"""`W4-06` — nửa dưới của một lượt chat, và cách đóng khung SSE.

Phần chạm Postgres nằm ở `tests/integration/test_chat_stream.py`; ở đây là
những thứ kiểm được mà không cần hạ tầng: hợp đồng khung, thứ tự khung, và
chuyện gì xảy ra khi dòng token đứt.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from typing import Any

import pytest

from rag_core.llm import ChatMessage, LLMChunk, LLMError, LLMResponse
from rag_core.retrieval.filters import MetadataFilter
from rag_core.schemas import Chunk, DocumentMetadata, RetrievedChunk, TokenUsage
from serving.api.sse import encode
from serving.core.auth import Principal
from serving.core.chat import SYSTEM_PROMPT, ChatEvent, ChatService, ChatTurn

PRINCIPAL = Principal(tenant_id="acme", key_id="k1", scopes=frozenset())


def _hit(n: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"c{n}",
            doc_id=f"d{n}",
            content=text,
            chunk_index=0,
            metadata=DocumentMetadata(
                source_url=f"https://example.test/{n}", license="CC-BY-4.0", title=f"Tài liệu {n}"
            ),
        ),
        score=1.0 / n,
        rank=n,
    )


def _turn(**kwargs: Any) -> ChatTurn:
    defaults: dict[str, Any] = {
        "principal": PRINCIPAL,
        "conversation_id": "conv1",
        "user_message_id": "m1",
        "question": "RRF là gì?",
        "history": [],
        "contexts": [_hit(1, "RRF là reciprocal rank fusion."), _hit(2, "k=1 thắng.")],
        "bundle_version": "0.2.0",
    }
    return ChatTurn(**{**defaults, **kwargs})


class FakeLLM:
    name = "fake"
    model = "fake-model"

    def __init__(self, deltas: Sequence[str] = ("Xin ", "chào"), fail_after: int | None = None):
        self.deltas = list(deltas)
        self.fail_after = fail_after
        self.seen: list[ChatMessage] = []

    async def astream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra_body: Any = None,
    ) -> AsyncIterator[LLMChunk]:
        self.seen = list(messages)
        for i, piece in enumerate(self.deltas):
            if self.fail_after is not None and i == self.fail_after:
                raise LLMError("provider đứt")
            yield LLMChunk(delta=piece)
        yield LLMChunk(
            final=LLMResponse(
                text="".join(self.deltas),
                model="fake-model-served",
                model_requested="fake-model",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=2, cost_usd=0.0001),
                finish_reason="stop",
            )
        )


class CapturingService(ChatService):
    """`ChatService` không chạm DB: chỉ ghi lại thứ lẽ ra đã được lưu."""

    saved: list[dict[str, Any]]

    def _schedule_save(self, turn: ChatTurn, text: str, model: str, finish_reason: str) -> None:
        self.saved.append({"text": text, "model": model, "finish_reason": finish_reason})


def _service(llm: Any) -> CapturingService:
    service = CapturingService(registry=None, sessions=None, llm=llm)  # type: ignore[arg-type]
    service.saved = []
    return service


async def _drain(service: ChatService, turn: ChatTurn) -> list[tuple[str, dict[str, Any]]]:
    return [(e.event, e.data) async for e in service.stream_turn(turn)]


# ---------------------------------------------------------------------------
# 1. Hợp đồng khung
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_frame_order_is_meta_sources_deltas_done() -> None:
    events = await _drain(_service(FakeLLM()), _turn())

    assert [name for name, _ in events] == ["meta", "sources", "delta", "delta", "done"]


@pytest.mark.asyncio
async def test_sources_arrive_before_the_first_token() -> None:
    """UI hiện được nguồn trong lúc chữ còn đang chảy.

    Đảo thứ tự (gửi `sources` ở cuối) vẫn "chạy", và vẫn hỏng đúng cái nó tồn
    tại để làm: người đọc thấy một khẳng định trước, nguồn của nó sau.
    """
    events = await _drain(_service(FakeLLM()), _turn())
    names = [name for name, _ in events]

    assert names.index("sources") < names.index("delta")
    sources = events[1][1]["sources"]
    assert [s["n"] for s in sources] == [1, 2]
    assert sources[0]["chunk_id"] == "c1"


@pytest.mark.asyncio
async def test_done_carries_usage_and_the_model_that_actually_served() -> None:
    events = await _drain(_service(FakeLLM()), _turn())
    done = events[-1][1]

    assert done["model"] == "fake-model-served"
    assert done["usage"]["completion_tokens"] == 2
    assert done["finish_reason"] == "stop"
    assert done["ttfb_ms"] is not None and done["ttfb_ms"] <= done["total_ms"]


@pytest.mark.asyncio
async def test_the_prompt_numbers_contexts_the_same_way_the_sources_frame_does() -> None:
    """`[1]` trong prompt phải là `n = 1` trong khung `sources`.

    Lệch một chỗ ở đây thì mọi trích dẫn của model trỏ sai nguồn — và câu trả
    lời vẫn đọc như thật, vì nó *có* trích dẫn.
    """
    llm = FakeLLM()
    events = await _drain(_service(llm), _turn())

    user_turn = llm.seen[-1].content
    assert "[1] RRF là reciprocal rank fusion." in user_turn
    assert "[2] k=1 thắng." in user_turn
    assert llm.seen[0].content == SYSTEM_PROMPT
    assert events[1][1]["sources"][0]["n"] == 1


@pytest.mark.asyncio
async def test_history_sits_between_the_system_prompt_and_the_question() -> None:
    llm = FakeLLM()
    history = [
        ChatMessage(role="user", content="câu cũ"),
        ChatMessage(role="assistant", content="đáp cũ"),
    ]
    await _drain(_service(llm), _turn(history=history))

    assert [m.role for m in llm.seen] == ["system", "user", "assistant", "user"]
    assert llm.seen[1].content == "câu cũ"


# ---------------------------------------------------------------------------
# 2. ⭐ Dòng token đứt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_midstream_failure_becomes_an_error_frame_not_a_silent_stop() -> None:
    """Sau `200 OK` không còn status nào để trả. Im lặng ở đây = nửa câu trả lời
    trông y hệt một câu trả lời ngắn đã xong."""
    events = await _drain(_service(FakeLLM(["a", "b", "c"], fail_after=2)), _turn())
    names = [name for name, _ in events]

    assert names[-1] == "error"
    assert "done" not in names
    assert events[-1][1]["partial_chars"] == 2


@pytest.mark.asyncio
async def test_the_partial_answer_is_still_saved_when_the_stream_breaks() -> None:
    service = _service(FakeLLM(["a", "b", "c"], fail_after=2))
    await _drain(service, _turn())

    assert service.saved == [{"text": "ab", "model": "fake-model", "finish_reason": "error"}]


async def _read_two_deltas(gen: AsyncGenerator[ChatEvent, None]) -> list[str]:
    got: list[str] = []
    while len(got) < 2:
        event = await anext(gen)
        if event.event == "delta":
            got.append(event.data["text"])
    return got


@pytest.mark.asyncio
async def test_a_cancellation_while_awaiting_the_next_token_saves_the_partial_answer() -> None:
    """⭐ Token đã trả tiền rồi. Vứt chúng đi là trả tiền cho một thứ không tồn tại.

    ⚠️ Đây là đường huỷ **thứ nhất**, và cách viết test cho nó không hiển nhiên:
    `raise CancelledError` trong thân `async for` **không** đi vào generator —
    generator chỉ đang treo ở `yield`, nó không nằm trong chuỗi `await`. Lần
    viết đầu của test này làm đúng thế và nó đỏ vì lý do ấy, chứ không phải vì
    mã sai.

    Cái mô phỏng đúng việc task bị huỷ *trong lúc chờ token kế tiếp* là
    `athrow()`: nó ném vào đúng chỗ generator đang treo.
    """
    service = _service(FakeLLM(["a", "b", "c", "d"]))
    gen = service.stream_turn(_turn())
    assert await _read_two_deltas(gen) == ["a", "b"]

    with pytest.raises(asyncio.CancelledError):
        await gen.athrow(asyncio.CancelledError())

    assert service.saved == [
        {"text": "ab", "model": "fake-model", "finish_reason": "client_disconnect"}
    ]


@pytest.mark.asyncio
async def test_an_abandoned_generator_still_saves_when_it_is_finally_closed() -> None:
    """Đường huỷ **thứ hai**: việc huỷ rơi vào lúc đang `send`, không phải lúc
    đang chờ token.

    Khi đó generator bị **bỏ rơi** ở `yield` và chỉ được đóng sau đó (`aclose()`,
    trong thực tế là do GC của asyncio). Nó nhận `GeneratorExit`, không phải
    `CancelledError` — hai exception khác nhau, cùng một nguyên nhân, và một
    `except asyncio.CancelledError` đơn độc bỏ lọt đúng một nửa số ca.
    """
    service = _service(FakeLLM(["a", "b", "c", "d"]))
    gen = service.stream_turn(_turn())
    assert await _read_two_deltas(gen) == ["a", "b"]

    await gen.aclose()

    assert service.saved == [
        {"text": "ab", "model": "fake-model", "finish_reason": "client_disconnect"}
    ]


@pytest.mark.asyncio
async def test_an_empty_answer_is_not_written_as_an_empty_row() -> None:
    """Một hàng rỗng trong lịch sử không phân biệt được với việc model im lặng."""
    service = ChatService(registry=None, sessions=None, llm=FakeLLM([]))  # type: ignore[arg-type]
    events = await _drain(service, _turn())

    assert events[-1][1]["finish_reason"] == "empty"


# ---------------------------------------------------------------------------
# 3. Đóng khung SSE
# ---------------------------------------------------------------------------


def test_a_newline_in_the_payload_does_not_split_the_frame() -> None:
    """⭐ Cái bẫy của SSE: dòng trống là dấu hết khung.

    Gửi text thô thì câu trả lời đầu tiên có xuống dòng — tức gần như mọi câu
    trả lời — tới client thành hai sự kiện, cái thứ hai không có tên.
    """
    frame = encode("delta", {"text": "dòng một\n\ndòng hai"}).decode("utf-8")

    assert frame.count("\n\n") == 1, "payload đã tách khung làm đôi"
    assert frame.endswith("\n\n")
    assert frame.startswith("event: delta\ndata: ")
    body = json.loads(frame.split("data: ", 1)[1])
    assert body["text"] == "dòng một\n\ndòng hai"


def test_vietnamese_is_not_escaped_into_ascii() -> None:
    assert "chào" in encode("delta", {"text": "chào"}).decode("utf-8")


def test_a_non_serialisable_value_does_not_kill_the_stream() -> None:
    """`default=str` thay vì `TypeError`: một khung xấu tốt hơn một stream chết
    giữa chừng vì `datetime` lọt vào payload."""
    frame = encode("x", {"f": MetadataFilter(tenant_id="a")}).decode("utf-8")

    assert "tenant_id='a'" in frame
    assert frame.endswith("\n\n") and frame.count("\n\n") == 1
