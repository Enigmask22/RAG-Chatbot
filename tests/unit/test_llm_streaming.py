"""`W4-06` — `OpenAICompatProvider.astream`.

Không chạm mạng: `httpx.MockTransport` dựng nguyên một phản hồi
`text/event-stream`, nên phần được kiểm ở đây đúng là phần mã của mình —
tách khung, gom `usage`, và **luật thử lại**.

Luật thử lại là thứ đáng kiểm nhất. Nó không có triệu chứng nào khi sai: một
lần thử lại sau khi đã phát token cho ra một câu trả lời trôi chảy, mạch lạc, và
là hai câu trả lời khác nhau nối vào nhau.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable

import httpx
import pytest

from rag_core.llm import ChatMessage, LLMError, OpenAICompatProvider

MESSAGES = [ChatMessage(role="user", content="xin chào")]


def _sse(*events: dict[str, object]) -> bytes:
    lines = [f"data: {json.dumps(e)}\n\n" for e in events]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


def _delta(text: str) -> dict[str, object]:
    return {"model": "m-served", "choices": [{"delta": {"content": text}}]}


def _finish(reason: str = "stop") -> dict[str, object]:
    return {"model": "m-served", "choices": [{"delta": {}, "finish_reason": reason}]}


def _usage(prompt: int = 100, completion: int = 20) -> dict[str, object]:
    return {
        "model": "m-served",
        "choices": [],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


def _provider(
    handler: Callable[[httpx.Request], httpx.Response], **kwargs: object
) -> OpenAICompatProvider:
    transport = httpx.MockTransport(handler)
    return OpenAICompatProvider(
        "m-asked",
        api_key="k",
        base_url="https://example.invalid",
        async_client=httpx.AsyncClient(transport=transport),
        max_retries=2,
        **kwargs,  # type: ignore[arg-type]
    )


async def _collect(provider: OpenAICompatProvider) -> list[object]:
    return [chunk async for chunk in provider.astream(MESSAGES)]


# ---------------------------------------------------------------------------
# 1. Hợp đồng của `LLMChunk`
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deltas_arrive_one_by_one_and_the_last_chunk_carries_the_whole_answer() -> None:
    body = _sse(_delta("Xin "), _delta("chào "), _delta("bạn"), _finish(), _usage())
    provider = _provider(lambda r: httpx.Response(200, content=body))

    chunks = await _collect(provider)

    assert [c.delta for c in chunks[:3]] == ["Xin ", "chào ", "bạn"]  # type: ignore[attr-defined]
    assert all(c.final is None for c in chunks[:-1])  # type: ignore[attr-defined]
    final = chunks[-1].final  # type: ignore[attr-defined]
    assert final is not None
    assert final.text == "Xin chào bạn"
    assert final.finish_reason == "stop"


@pytest.mark.asyncio
async def test_the_served_model_is_read_from_the_stream_not_from_what_we_asked() -> None:
    """Quy tắc cứng #1 phải đúng ở đường stream y như ở `complete()`.

    100% traffic production đi qua đây, nên một lỗ ở đường này lớn hơn hẳn cùng
    một lỗ ở đường kia.
    """
    body = _sse(_delta("a"), _finish(), _usage())
    provider = _provider(lambda r: httpx.Response(200, content=body))

    final = (await _collect(provider))[-1].final  # type: ignore[attr-defined]

    assert final.model == "m-served"
    assert final.model_requested == "m-asked"
    assert final.model_drifted is True


@pytest.mark.asyncio
async def test_usage_survives_the_empty_choices_chunk_that_carries_it() -> None:
    """Mẩu mang `usage` của OpenAI-compat có `choices` **rỗng**.

    Đọc `usage` bên trong nhánh `if choices:` là một dòng trông đúng và làm mọi
    con số chi phí của đường serving bằng 0 — im lặng, vì `TokenUsage(0, 0)` hợp
    lệ về hình dạng.
    """
    body = _sse(_delta("a"), _finish(), _usage(prompt=1_000_000, completion=1_000_000))
    provider = _provider(lambda r: httpx.Response(200, content=body))

    final = (await _collect(provider))[-1].final  # type: ignore[attr-defined]

    assert final.usage.prompt_tokens == 1_000_000
    assert final.usage.completion_tokens == 1_000_000


@pytest.mark.asyncio
async def test_include_usage_is_requested_or_the_provider_never_sends_it() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, content=_sse(_delta("a"), _finish(), _usage()))

    await _collect(_provider(handler))

    assert seen["stream"] is True
    assert seen["stream_options"] == {"include_usage": True}
    assert seen["temperature"] == 0.0


@pytest.mark.asyncio
async def test_a_line_that_is_not_json_is_skipped_not_fatal() -> None:
    """Provider chèn `: keep-alive` và dòng trống. Chết vì chúng là chết vì lịch sự."""
    raw = b": ping\n\n" + _sse(_delta("a"), _finish(), _usage())
    provider = _provider(lambda r: httpx.Response(200, content=raw))

    final = (await _collect(provider))[-1].final  # type: ignore[attr-defined]

    assert final.text == "a"


# ---------------------------------------------------------------------------
# 2. ⭐⭐ Luật thử lại — ranh giới là token đầu tiên
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_429_before_the_first_token_is_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, content=b"slow down")
        return httpx.Response(200, content=_sse(_delta("ok"), _finish(), _usage()))

    provider = _provider(handler)
    provider._max_retries = 1  # bỏ backoff dài

    final = (await _collect(provider))[-1].final  # type: ignore[attr-defined]

    assert calls["n"] == 2
    assert final.text == "ok"


@pytest.mark.asyncio
async def test_a_break_after_the_first_token_is_never_retried() -> None:
    """⭐ Cái test quan trọng nhất file này.

    Thử lại ở đây cho ra `"Xin chàoXin chào bạn"` — một chuỗi hợp lệ, đọc trôi
    chảy, và không phải cái model nào nói ra. Không có exception, không có log,
    và không có cách nào biết nó đã xảy ra khi nhìn vào dữ liệu đã lưu.
    """
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1

        async def broken() -> AsyncIterator[bytes]:
            yield b"data: " + json.dumps(_delta("Xin chào")).encode() + b"\n\n"
            raise httpx.ReadError("kết nối đứt")

        return httpx.Response(200, content=broken())

    provider = _provider(handler)

    got: list[str] = []
    with pytest.raises(LLMError, match="Không thử lại"):
        async for chunk in provider.astream(MESSAGES):
            got.append(chunk.delta)

    assert calls["n"] == 1, "đã phát token rồi mà vẫn gọi lại provider"
    assert got == ["Xin chào"], "phần đã sinh phải tới được người gọi"


@pytest.mark.asyncio
async def test_a_400_is_not_retried_because_it_will_fail_the_same_way() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, content=b"model not found")

    with pytest.raises(LLMError, match="400"):
        await _collect(_provider(handler))

    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_giving_up_says_how_many_times_it_tried() -> None:
    provider = _provider(lambda r: httpx.Response(503, content=b"nope"))
    provider._max_retries = 1

    with pytest.raises(LLMError, match="thất bại sau 2 lần"):
        await _collect(provider)
