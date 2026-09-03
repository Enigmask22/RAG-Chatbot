"""`W3-04`/`TD-32` — vòng chạy job: checkpoint, nối lượt lùi, ghi lô hỏng.

Nhóm này lấp một lỗ hổng có thật: `run_requests`, `load_done_keys` và
`load_contexts` **chưa từng có test nào** dù chúng quyết định hai thứ đắt nhất
của job — có phải trả tiền lại cho công đã làm không, và artifact có bị ghi dữ
liệu sai không.

Đường nối giữa lượt gộp và lượt lùi (`--skip-done-chunks`) đặc biệt đáng canh:
sai một chiều thì trả tiền lại cho 13.000 chunk đã xong, sai chiều kia thì
~2.800 chunk bị bỏ lại vĩnh viễn mà chỉ hiện ra dưới dạng `coverage` thiếu vài
phần trăm ở `apply_contexts`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from pipeline.indexing.contextualize import (
    load_contexts,
    load_done_keys,
    read_requests,
    run_requests,
    write_requests,
)
from rag_core.chunking.contextual import ECHO_SEPARATOR, ContextRequest
from rag_core.llm.base import ChatMessage, LLMProvider, LLMResponse, ModelPricing
from rag_core.llm.budget import CostBudget
from rag_core.schemas import TokenUsage

PRICING = ModelPricing(input_per_1m_usd=1.0, output_per_1m_usd=2.0)


# ⚠️ Mở đầu phải KHÁC HẲN nhau từng cái. Bản đầu dùng `f"mở đầu của {cid}"`, và
# hai chuỗi ấy giống nhau 90% nên phép hoán đổi không phân biệt được — test
# "bắt được lệch thứ tự" xanh vì fixture, không vì mã. Đây là lần thứ năm trong
# dự án một fixture đều tăm tắp làm phép kiểm mất hiệu lực (`W3-01` §2, `W3-05`
# §9, `W3-07`, và chính lượt canary của `TD-32`).
OPENINGS = [
    "Ngân sách đầu tư công tỉnh",
    "Rủi ro khí hậu ven biển",
    "Thị trường lao động phi chính thức",
    "Trái phiếu doanh nghiệp phát hành",
    "Hệ thống an sinh xã hội",
    "Năng suất nông nghiệp đồng bằng",
    "Chuyển đổi số khu vực công",
    "Giáo dục đại học và kỹ năng",
]


def request(key: str, chunk_ids: Sequence[str], *, doc_id: str = "doc") -> ContextRequest:
    return ContextRequest(
        key=key,
        chunk_ids=tuple(chunk_ids),
        doc_id=doc_id,
        messages=(
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content=f"user {key}"),
        ),
        est_prompt_tokens=100,
        echoes=tuple(OPENINGS[i % len(OPENINGS)] for i in range(len(chunk_ids))),
    )


class FakeProvider(LLMProvider):
    """Provider giả ghi lại `max_tokens` từng lời gọi — đó là thứ test cần soi."""

    name = "fake"
    model = "fake-model"

    def __init__(self, reply: str | None = None) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        seed: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append({"max_tokens": max_tokens, "messages": messages})
        text = self.reply if self.reply is not None else "ngữ cảnh"
        return LLMResponse(
            text=text,
            model=self.model,
            model_requested=self.model,
            usage=TokenUsage(prompt_tokens=800, completion_tokens=80, cost_usd=0.0008),
            finish_reason="stop",
        )


def batch_reply(req: ContextRequest, contexts: Sequence[str]) -> str:
    return "\n".join(
        f"[{i}] {echo}{ECHO_SEPARATOR}{ctx}"
        for i, (echo, ctx) in enumerate(zip(req.echoes, contexts, strict=True), start=1)
    )


def run(
    requests: Sequence[ContextRequest],
    out: Path,
    provider: LLMProvider,
    **kwargs: Any,
) -> Any:
    return run_requests(
        requests,
        provider,
        out,
        budget=CostBudget(0.0, name="test"),
        max_tokens=kwargs.pop("max_tokens", 200),
        concurrency=1,
        progress_every=0,
        **kwargs,
    )


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --------------------------------------------------------------------------
# 1. Vòng đời gói request
# --------------------------------------------------------------------------


def test_requests_round_trip_through_the_job_bundle(tmp_path: Path) -> None:
    original = [request("k1", ["c1", "c2"]), request("k2", ["c3"])]
    path = tmp_path / "requests.jsonl.gz"
    assert write_requests(path, original) == 2
    assert read_requests(path) == original


def test_the_old_single_chunk_schema_still_loads(tmp_path: Path) -> None:
    """⚠️ `data/contexts/requests.jsonl.gz` đã dựng **trước** khi có chế độ gộp.

    Nó ghi `chunk_id` số ít và không có `echoes`. File ấy chính là gói dùng cho
    lượt lùi cuối cùng, nên đọc không được nó nghĩa là mất luôn đường lùi — và
    triệu chứng sẽ là "0 request" chứ không phải một lỗi nói ra điều đó.
    """
    path = tmp_path / "old.jsonl"
    path.write_text(
        json.dumps(
            {
                "key": "k1",
                "chunk_id": "c1",
                "doc_id": "doc",
                "system": "sys",
                "user": "user k1",
                "est_prompt_tokens": 42,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = read_requests(path)
    assert [r.chunk_ids for r in loaded] == [("c1",)]
    assert loaded[0].echoes == ()
    assert loaded[0].n_chunks == 1


# --------------------------------------------------------------------------
# 2. Checkpoint
# --------------------------------------------------------------------------


def test_done_keys_ignore_rows_without_a_context(tmp_path: Path) -> None:
    path = tmp_path / "contexts.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"key": "a", "chunk_id": "c1", "context": "có"}),
                json.dumps({"key": "b", "chunk_id": "c2", "context": "   "}),
                "{ dòng hỏng",
            ]
        ),
        encoding="utf-8",
    )
    assert load_done_keys(path) == {"a"}


def test_contexts_are_keyed_by_chunk_not_by_request(tmp_path: Path) -> None:
    """Một khoá request phụ trách nhiều chunk, nên khoá-theo-request không ánh xạ được."""
    path = tmp_path / "contexts.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"key": "a", "chunk_id": "c1", "context": "một"}),
                json.dumps({"key": "a", "chunk_id": "c2", "context": "hai"}),
            ]
        ),
        encoding="utf-8",
    )
    assert load_contexts(path) == {"c1": "một", "c2": "hai"}


def test_contexts_from_another_configuration_are_filtered_out(tmp_path: Path) -> None:
    """Đây là chỗ giữ lại tính vô hiệu hoá theo cấu hình sau khi bỏ khoá-theo-request.

    Artifact có thể lẫn dòng của hai cấu hình (chạy gộp rồi chạy lùi). Lọc theo
    tập khoá của lượt hiện tại là cách duy nhất để `apply_contexts` không dán một
    ngữ cảnh sinh bằng prompt khác lên chunk.
    """
    path = tmp_path / "contexts.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"key": "cũ", "chunk_id": "c1", "context": "cũ"}),
                json.dumps({"key": "mới", "chunk_id": "c2", "context": "mới"}),
            ]
        ),
        encoding="utf-8",
    )
    assert load_contexts(path, keys={"mới"}) == {"c2": "mới"}


def test_a_second_run_skips_what_the_first_finished(tmp_path: Path) -> None:
    out = tmp_path / "contexts.jsonl"
    requests = [request("k1", ["c1"]), request("k2", ["c2"])]
    provider = FakeProvider()
    run(requests, out, provider)
    assert len(provider.calls) == 2

    again = FakeProvider()
    report = run(requests, out, again)
    assert again.calls == []
    assert report.n_skipped == 2


# --------------------------------------------------------------------------
# 3. Nối lượt gộp với lượt lùi
# --------------------------------------------------------------------------


def test_fallback_skips_chunks_the_batched_pass_already_covered(tmp_path: Path) -> None:
    """⭐ Đây là đường nối. Không có nó thì lượt lùi chạy lại **toàn bộ** 15.814 chunk."""
    out = tmp_path / "contexts.jsonl"
    batched = request("b1", ["c1", "c2"])
    run([batched], out, FakeProvider(batch_reply(batched, ["một", "hai"])))
    assert set(load_contexts(out)) == {"c1", "c2"}

    fallback = FakeProvider()
    report = run(
        [request("s1", ["c1"]), request("s2", ["c2"])],
        out,
        fallback,
        skip_done_chunks=True,
    )
    assert fallback.calls == []
    assert report.n_requests == 2


def test_fallback_still_runs_a_chunk_the_batched_pass_missed(tmp_path: Path) -> None:
    """Chiều ngược lại, và là chiều nguy hiểm hơn: bỏ sót thì im lặng."""
    out = tmp_path / "contexts.jsonl"
    batched = request("b1", ["c1", "c2"])
    run([batched], out, FakeProvider(batch_reply(batched, ["một", "hai"])))

    fallback = FakeProvider()
    run([request("s3", ["c3"])], out, fallback, skip_done_chunks=True)
    assert len(fallback.calls) == 1
    assert "c3" in load_contexts(out)


def test_without_the_flag_the_fallback_would_redo_everything(tmp_path: Path) -> None:
    """Ghim rằng cờ là **tuỳ chọn**, để lượt chạy thường vẫn vô hiệu hoá theo cấu hình.

    Khoá đổi khi prompt đổi, và đó là cách "đổi `prompt_version` thì sinh lại"
    hoạt động. Bật `skip_done_chunks` mặc định sẽ âm thầm phá cơ chế ấy.
    """
    out = tmp_path / "contexts.jsonl"
    batched = request("b1", ["c1", "c2"])
    run([batched], out, FakeProvider(batch_reply(batched, ["một", "hai"])))

    fallback = FakeProvider()
    run([request("s1", ["c1"])], out, fallback)
    assert len(fallback.calls) == 1


# --------------------------------------------------------------------------
# 4. Lô hỏng
# --------------------------------------------------------------------------


def test_a_rejected_batch_writes_nothing_to_the_artifact(tmp_path: Path) -> None:
    """Từ chối là từ chối **cả lô** — ghi phần "có vẻ đúng" là đúng thứ chốt chặn để chặn."""
    out = tmp_path / "contexts.jsonl"
    req = request("b1", ["c1", "c2"])
    swapped = (
        f"[1] {req.echoes[1]}{ECHO_SEPARATOR}ngữ cảnh của chunk 2\n"
        f"[2] {req.echoes[0]}{ECHO_SEPARATOR}ngữ cảnh của chunk 1"
    )
    report = run([req], out, FakeProvider(swapped))
    assert report.n_rejected == 1
    assert report.n_done == 0
    assert not out.exists() or out.read_text(encoding="utf-8").strip() == ""


def test_a_rejected_batch_keeps_the_raw_response_for_offline_reparsing(tmp_path: Path) -> None:
    """⚠️ Bản đầu chỉ ghi lý do, và nó tốn tiền thật.

    Lượt canary đầu của `TD-32` từ chối 109/110 lô vì chốt chặn quá chặt; sau khi
    sửa chốt chặn thì 92 lô lẽ ra bóc được — nhưng câu trả lời đã không còn nên
    phải gọi lại. Lưu văn bản biến "sửa parser" từ việc **mua lại** dữ liệu thành
    việc **bóc lại** dữ liệu đã mua.
    """
    out = tmp_path / "contexts.jsonl"
    req = request("b1", ["c1", "c2"])
    swapped = f"[1] {req.echoes[1]}{ECHO_SEPARATOR}x\n[2] {req.echoes[0]}{ECHO_SEPARATOR}y"
    run([req], out, FakeProvider(swapped))
    failures = rows(out.with_suffix(".failures.jsonl"))
    assert failures[0]["response_text"] == swapped
    assert failures[0]["chunk_ids"] == ["c1", "c2"]


def test_a_rejected_batch_is_retried_on_the_next_run(tmp_path: Path) -> None:
    """Lỗi ghi sang sidecar chứ không vào artifact chính, nên lần sau tự thử lại."""
    out = tmp_path / "contexts.jsonl"
    req = request("b1", ["c1", "c2"])
    run([req], out, FakeProvider("không đúng định dạng gì cả"))

    good = FakeProvider(batch_reply(req, ["một", "hai"]))
    report = run([req], out, good)
    assert len(good.calls) == 1
    assert report.n_done == 2


# --------------------------------------------------------------------------
# 5. Trần output và kế toán
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_chunks", [1, 4, 8])
def test_output_cap_scales_with_the_batch(tmp_path: Path, n_chunks: int) -> None:
    """⚠️ Lô 8 chunk chạy với trần của một chunk sẽ bị cắt lời ở chunk thứ hai.

    Triệu chứng khi ấy là `BatchParseError` hàng loạt — một thông báo không hề
    nói ra nguyên nhân, và trên một job tính tiền theo token thì nó đắt.
    """
    req = request("b", [f"c{i}" for i in range(n_chunks)])
    provider = FakeProvider(batch_reply(req, ["ctx"] * n_chunks) if n_chunks > 1 else "ctx")
    run([req], tmp_path / "out.jsonl", provider, max_tokens=200)
    assert provider.calls[0]["max_tokens"] == 200 * n_chunks


def test_batch_usage_is_split_across_the_chunks_it_produced(tmp_path: Path) -> None:
    """Cộng lại phải đúng bằng chi phí thật; `batch_size` nói rõ đó là phần chia."""
    out = tmp_path / "contexts.jsonl"
    req = request("b1", ["c1", "c2", "c3", "c4"])
    run([req], out, FakeProvider(batch_reply(req, ["a", "b", "c", "d"])))
    written = rows(out)
    assert len(written) == 4
    assert {r["batch_size"] for r in written} == {4}
    assert sum(r["cost_usd"] for r in written) == pytest.approx(0.0008)
    assert sum(r["prompt_tokens"] for r in written) == pytest.approx(800)


def test_cost_per_1000_counts_chunks_not_calls(tmp_path: Path) -> None:
    """Con số DoD hỏi là *cost/1000 **chunk***, và gộp làm hai thứ ấy khác nhau 8 lần."""
    out = tmp_path / "contexts.jsonl"
    req = request("b1", [f"c{i}" for i in range(8)])
    report = run([req], out, FakeProvider(batch_reply(req, ["ctx"] * 8)))
    assert report.n_done == 8
    assert report.cost_per_1000 == pytest.approx(0.0008 / 8 * 1000)
