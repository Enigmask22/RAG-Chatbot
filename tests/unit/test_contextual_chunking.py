"""`W3-04` — sinh ngữ cảnh định vị cho chunk trước khi embed.

DoD: *cost/1000 chunk được log; có flag tắt; fail 1 chunk không làm sập cả job.*

Ba nửa hỏng theo ba kiểu khác nhau nên tách ba nhóm: **dựng prompt** hỏng thì
tốn tiền GPU rồi mới biết (và bất biến tiền tố vỡ thì tốn gấp đôi); **dán ngữ
cảnh** hỏng thì index sai mà không ai thấy; **trần chi phí** hỏng thì hoá đơn
mới là thứ báo.
"""

from __future__ import annotations

import inspect
import types
from collections.abc import Sequence

import pytest

from rag_core.chunking.contextual import (
    CONTEXT_KEY,
    CONTEXT_SYSTEM_PROMPT,
    ContextRequest,
    ContextualConfig,
    EnrichStats,
    apply_contexts,
    build_requests,
    original_content,
)
from rag_core.chunking.tokens import TokenCounter
from rag_core.llm.budget import BudgetExceeded, CostBudget
from rag_core.llm.tokenizer import HFTokenCounter
from rag_core.schemas import Chunk, Document, DocumentMetadata, Language

METADATA = DocumentMetadata(source_url="https://example.org/doc", license="CC BY 4.0")

HEAD = (
    "Vietnam Development Report 2024. World Bank Group, Hanoi.\n\n"
    "This report reviews public investment management in Viet Nam over the "
    "period 2015 to 2023, with a focus on provincial execution rates.\n\n"
)

# Câu **khác nhau** từng câu, cố ý: fixture đều tăm tắp đã ba lần làm hỏng phép
# kiểm trong W3 (`W3-01` §2, `W3-05` §9, `W3-07`). Ở đây nếu mọi đoạn giống nhau
# thì phép kiểm "cửa sổ lân cận chứa đúng vùng quanh chunk" luôn xanh.
BODY = "".join(
    f"Section body sentence {i:03d} about disbursement in province {i:03d}. " for i in range(400)
)
TEXT = HEAD + BODY


class CharCounter:
    """`TokenCounter` giả với mật độ khai báo được — đơn vị là ký tự/token.

    Cần khai báo được vì `build_requests` cắt cửa sổ **theo mật độ đo được**, và
    một bộ đếm cố định 1 ký tự = 1 token sẽ giấu mất mọi lỗi nhân/chia.
    """

    def __init__(self, chars_per_token: float = 4.0) -> None:
        self.chars_per_token = chars_per_token
        self.calls = 0

    @property
    def max_sequence_tokens(self) -> int | None:
        return 40960

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        self.calls += 1
        return [max(1, round(len(t) / self.chars_per_token)) for t in texts]


def document(text: str = TEXT) -> Document:
    return Document(doc_id="doc", content=text, metadata=METADATA)


def chunk_at(index: int, start: int, length: int = 300, text: str = TEXT) -> Chunk:
    return Chunk(
        chunk_id=f"doc::{index:05d}",
        doc_id="doc",
        content=text[start : start + length],
        chunk_index=index,
        start_char=start,
        end_char=start + length,
    )


def config(**overrides: object) -> ContextualConfig:
    base: dict[str, object] = {"model": "qwen3-8b", "head_tokens": 100, "window_tokens": 200}
    base.update(overrides)
    return ContextualConfig(**base)  # type: ignore[arg-type]


def build(chunks: Sequence[Chunk], **overrides: object) -> list[ContextRequest]:
    return build_requests(document(), chunks, config=config(**overrides), counter=CharCounter())


# --------------------------------------------------------------------------
# 1. Dựng prompt
# --------------------------------------------------------------------------


def test_disabled_produces_no_calls() -> None:
    """Cờ tắt của DoD: tắt là **không lời gọi nào**, không phải lời gọi rỗng."""
    assert build([chunk_at(0, 0), chunk_at(1, 4000)], enabled=False) == []


def test_one_request_per_chunk() -> None:
    chunks = [chunk_at(0, 0), chunk_at(1, 4000), chunk_at(2, 8000)]
    requests = build(chunks)
    assert [r.chunk_id for r in requests] == [c.chunk_id for c in chunks]


def test_system_prompt_is_identical_across_requests() -> None:
    """Tiền tố dùng chung cho **cả job**. Lệch một ký tự là mất cache toàn cục."""
    requests = build([chunk_at(0, 0), chunk_at(1, 4000), chunk_at(2, 8000)])
    systems = {r.messages[0].content for r in requests}
    assert systems == {CONTEXT_SYSTEM_PROMPT}


def test_document_head_is_a_shared_prefix_of_every_user_message() -> None:
    """⭐ Bất biến quyết định thời gian chạy pod, nên phải có test riêng.

    vLLM cache theo tiền tố token. Nếu phần `<document_head>` không đứng đầu và
    không giống hệt nhau giữa các chunk cùng tài liệu thì ~2.150 token/lời gọi
    bị prefill lại 15.814 lần thay vì 60 lần. Không có gì đỏ khi điều đó xảy ra —
    job chỉ chạy lâu gấp đôi.
    """
    requests = build([chunk_at(0, 4000), chunk_at(1, 8000), chunk_at(2, 12000)])
    texts = [r.user_text for r in requests]
    head_block = texts[0].split("</document_head>")[0]

    assert "Vietnam Development Report 2024" in head_block
    for text in texts:
        assert text.startswith(head_block), "document_head phải là tiền tố chung"


def test_neighbourhood_never_repeats_text_already_in_the_head() -> None:
    """Không trả tiền prefill hai lần cho cùng một văn bản.

    Chunk nằm ngay đầu tài liệu là ca khó: vùng lân cận của nó **chồng** lên
    `document_head`. Phần chồng phải bị cắt bỏ, phần nằm sau head thì giữ — nên
    phép kiểm là "không lặp", không phải "không có vùng lân cận".
    """
    request = build([chunk_at(0, 0, length=100)])[0]
    head_block = request.user_text.split("<document_head>")[1].split("</document_head>")[0]
    window = request.user_text.split("<neighbourhood>")[1].split("</neighbourhood>")[0]

    assert window.strip(), "vùng nằm SAU head vẫn phải được đưa vào"
    assert window.strip() not in head_block
    # Chunk ở vị trí 0 thì nội dung nó NẰM TRONG head, nên `<chunk>` lặp lại
    # phần văn bản ấy — không tránh được và cũng không nên tránh: bỏ head đi cho
    # riêng mấy chunk đầu là phá vỡ bất biến tiền tố dùng chung, đổi lấy vài
    # trăm token ở đúng chỗ head đang được cache sẵn.


def test_no_neighbourhood_block_when_the_window_falls_inside_the_head() -> None:
    request = build([chunk_at(0, 0, length=50)], window_tokens=0)[0]
    assert "<neighbourhood>" not in request.user_text


def test_neighbourhood_spans_both_sides_of_a_mid_document_chunk() -> None:
    chunk = chunk_at(0, 20000, length=300)
    request = build([chunk])[0]
    window = request.user_text.split("<neighbourhood>")[1].split("</neighbourhood>")[0]

    before = TEXT[:20000]
    after = TEXT[20300:]
    assert window.split(".")[0].strip() in before, "phải có phần đứng TRƯỚC chunk"
    assert any(line.strip() and line.strip() in after for line in window.split(".")[-3:])


def test_window_size_follows_the_measured_density() -> None:
    """Cắt theo ký tự nhưng ngân sách khai bằng token → mật độ phải được dùng thật.

    Đo trên **riêng khối `document_head`**, không trên cả prompt: chunk và thẻ
    là phần cố định, gộp vào thì tỉ lệ 3× bị pha loãng còn 1,9× và ngưỡng phải
    nới tới mức không còn phân biệt được gì. Ngân sách để lớn (1000 token) để
    biên ±200 ký tự của `_snap_right` không nuốt mất tín hiệu.
    """
    chunk = chunk_at(0, 20000, length=300)

    def head_len(chars_per_token: float) -> int:
        request = build_requests(
            document(),
            [chunk],
            config=config(head_tokens=1000),
            counter=CharCounter(chars_per_token=chars_per_token),
        )[0]
        return len(request.user_text.split("<document_head>")[1].split("</document_head>")[0])

    assert head_len(6.0) / head_len(2.0) == pytest.approx(3.0, abs=0.15)


def test_chunk_without_span_is_rejected_loudly() -> None:
    orphan = Chunk(chunk_id="doc::00000", doc_id="doc", content="abc", chunk_index=0)
    with pytest.raises(ValueError, match="thiếu span"):
        build([orphan])


# --------------------------------------------------------------------------
# 2. Khoá — vừa là khoá checkpoint vừa là khoá cache
# --------------------------------------------------------------------------


def test_key_is_stable_across_identical_builds() -> None:
    chunks = [chunk_at(0, 4000)]
    assert build(chunks)[0].key == build(chunks)[0].key


@pytest.mark.parametrize(
    "override",
    [
        {"prompt_version": "ctx-v2"},
        {"model": "deepseek-v4-flash"},
        {"max_context_tokens": 200},
        {"head_tokens": 400},
        {"window_tokens": 400},
    ],
)
def test_key_changes_when_anything_that_changes_the_answer_changes(
    override: dict[str, object],
) -> None:
    """Đổi cấu hình mà khoá không đổi = dùng lại ngữ cảnh sinh bởi cấu hình khác."""
    chunks = [chunk_at(0, 20000)]
    assert build(chunks)[0].key != build(chunks, **override)[0].key


def test_key_differs_between_chunks() -> None:
    requests = build([chunk_at(0, 4000), chunk_at(1, 8000)])
    assert requests[0].key != requests[1].key


# --------------------------------------------------------------------------
# 3. Dán ngữ cảnh — nửa "một chunk hỏng không làm sập job"
# --------------------------------------------------------------------------


def enriched(
    context: str = "Bối cảnh: báo cáo World Bank 2024 về đầu tư công.",
) -> tuple[list[Chunk], EnrichStats, Chunk]:
    chunks = [chunk_at(0, 20000)]
    requests = build(chunks)
    out, stats = apply_contexts(chunks, requests, {requests[0].key: context})
    return out, stats, chunks[0]


def test_context_is_prepended_and_recorded() -> None:
    out, stats, before = enriched()
    assert out[0].content.startswith("Bối cảnh: báo cáo World Bank 2024 về đầu tư công.\n\n")
    assert out[0].content.endswith(before.content)
    assert out[0].extra[CONTEXT_KEY] == "Bối cảnh: báo cáo World Bank 2024 về đầu tư công."
    assert stats.n_enriched == 1 and stats.n_missing == 0


def test_token_count_is_cleared_rather_than_left_stale() -> None:
    """Số cũ đếm trên văn bản cũ. Giữ lại là một con số sai trông như đúng."""
    chunks = [chunk_at(0, 20000).model_copy(update={"token_count": 77})]
    requests = build(chunks)
    out, _ = apply_contexts(chunks, requests, {requests[0].key: "Bối cảnh."})
    assert out[0].token_count is None


def test_missing_context_keeps_the_chunk_and_counts_it() -> None:
    """DoD: một chunk hỏng không làm sập job. Nó cũng không được biến mất im lặng."""
    chunks = [chunk_at(0, 4000), chunk_at(1, 8000)]
    requests = build(chunks)
    out, stats = apply_contexts(chunks, requests, {requests[1].key: "Bối cảnh."})

    assert len(out) == 2
    assert out[0].content == chunks[0].content
    assert CONTEXT_KEY not in out[0].extra
    assert stats.n_missing == 1 and stats.n_enriched == 1
    assert stats.missing_chunk_ids == [chunks[0].chunk_id]
    assert stats.coverage == pytest.approx(0.5)


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_blank_context_is_not_pasted(blank: str) -> None:
    """LLM trả rỗng → chunk giữ nguyên, và được đếm **riêng** với ca thiếu hẳn."""
    chunks = [chunk_at(0, 20000)]
    requests = build(chunks)
    out, stats = apply_contexts(chunks, requests, {requests[0].key: blank})
    assert out[0].content == chunks[0].content
    assert stats.n_empty == 1 and stats.n_missing == 0


def test_enrichment_changes_content_hash() -> None:
    """Ghim tương tác với `W3-07`: lượt build đầu sau khi bật KHÔNG mượn lại được."""
    out, _, before = enriched()
    assert out[0].content_hash != before.content_hash


def test_original_content_round_trips_exactly() -> None:
    out, _, before = enriched()
    assert original_content(out[0]) == before.content


def test_original_content_is_identity_for_untouched_chunks() -> None:
    plain = chunk_at(0, 4000)
    assert original_content(plain) == plain.content


def test_original_content_survives_a_chunk_edited_after_enrichment() -> None:
    """Phòng thủ: nếu tiền tố không còn khớp thì trả nguyên văn, không cắt bừa."""
    out, _, _ = enriched()
    tampered = out[0].model_copy(update={"content": "văn bản đã bị sửa"})
    assert original_content(tampered) == "văn bản đã bị sửa"


# --------------------------------------------------------------------------
# 4. Trần chi phí
# --------------------------------------------------------------------------


def test_budget_blocks_before_the_call_that_would_exceed_it() -> None:
    budget = CostBudget(1.00, name="ctx")
    budget.charge(0.95)
    budget.reserve(0.04)
    with pytest.raises(BudgetExceeded, match=r"đã tiêu \$0\.9500"):
        budget.reserve(0.06)


def test_budget_without_cap_is_explicit_not_accidental() -> None:
    unlimited = CostBudget(0.0)
    unlimited.charge(1_000.0)
    unlimited.reserve(1_000.0)
    assert unlimited.unlimited and unlimited.remaining_usd == float("inf")


def test_cost_per_1000_is_the_number_the_dod_asks_for() -> None:
    budget = CostBudget(10.0)
    for _ in range(4):
        budget.charge(0.0005)
    assert budget.cost_per_1000() == pytest.approx(0.5)
    assert budget.calls == 4


def test_budget_counter_is_thread_safe() -> None:
    """`NEW-06` chốt job LLM dài phải chạy song song; bộ đếm không khoá thì trần sai."""
    import threading

    budget = CostBudget(0.0)
    barrier = threading.Barrier(8)

    def work() -> None:
        barrier.wait()
        for _ in range(500):
            budget.charge(0.001)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert budget.calls == 4000
    assert budget.spent_usd == pytest.approx(4.0)


# --------------------------------------------------------------------------
# 5. Ghim `HFTokenCounter` vào Protocol thật
# --------------------------------------------------------------------------


def test_hf_token_counter_satisfies_the_token_counter_protocol() -> None:
    """⭐ Khuôn của `W3-05`: đừng để cả hai bên đối chiếu đều do test dựng ra.

    `max_sequence_tokens` là **property** trong `TokenCounter`. Bản đầu của
    `HFTokenCounter` khai nó thành method và **không có gì đỏ** — `CharCounter`
    trong chính file này cũng sai y hệt, nên fake và lớp thật khớp nhau hoàn hảo
    trong khi cả hai cùng sai. `isinstance` không bắt được (`runtime_checkable`
    chỉ kiểm có thuộc tính hay không, method cũng thoả `hasattr`), nên phép ghim
    thật nằm ở phép gán có kiểu bên dưới — `mypy` mới là thứ đỏ.
    """
    counter: TokenCounter = HFTokenCounter("Qwen/Qwen3-8B", max_tokens=40960)
    assert isinstance(counter, TokenCounter)
    assert counter.max_sequence_tokens == 40960
    assert not isinstance(type(counter).max_sequence_tokens, types.FunctionType), (
        "phải là property, không phải method"
    )


def test_hf_token_counter_does_not_import_transformers_at_module_import() -> None:
    """Cùng lý do với `pipeline/ingest/app.py`: import nặng phải nằm trong hàm."""
    module = inspect.getmodule(HFTokenCounter)
    assert module is not None
    source = inspect.getsource(module)
    head = source.split("class HFTokenCounter")[0]
    assert "import transformers" not in head
    assert "from transformers" not in head


# --------------------------------------------------------------------------
# 6. Ngôn ngữ — ràng buộc mà dry-run đo được là bị bỏ qua 83% lần
# --------------------------------------------------------------------------


def document_in(language: Language) -> Document:
    return Document(
        doc_id="doc",
        content=TEXT,
        metadata=DocumentMetadata(
            source_url="https://example.org/doc", license="CC BY 4.0", lang=language
        ),
    )


@pytest.mark.parametrize(
    ("language", "name"), [(Language.EN, "English"), (Language.VI, "Vietnamese")]
)
def test_prompt_names_the_language_instead_of_asking_the_model_to_infer_it(
    language: Language, name: str
) -> None:
    """⭐⭐ Dry-run 30 request trên tài liệu tiếng Anh với prompt "same language as
    <chunk>": **15 tiếng Pháp, 10 tiếng Trung, 5 tiếng Anh**. Ngôn ngữ có sẵn
    trong manifest — bắt model suy ra thứ mình đã biết là tự thêm chỗ hỏng.
    """
    request = build_requests(
        document_in(language),
        [chunk_at(0, 20000)],
        config=config(),
        counter=CharCounter(),
    )[0]
    assert f"in {name}." in request.user_text
    assert "same language as" not in request.user_text


def test_language_instruction_sits_after_the_chunk() -> None:
    """Ngoài tiền tố dùng chung (không tốn prefill) và gần cuối nhất (được đọc chót)."""
    request = build_requests(
        document_in(Language.EN), [chunk_at(0, 20000)], config=config(), counter=CharCounter()
    )[0]
    text = request.user_text
    assert text.index("in English.") > text.index("</chunk>")


def test_unknown_language_falls_back_instead_of_naming_a_wrong_one() -> None:
    request = build_requests(
        document_in(Language.UNKNOWN), [chunk_at(0, 20000)], config=config(), counter=CharCounter()
    )[0]
    assert "same language as <chunk>" in request.user_text


def test_language_is_part_of_the_cache_key() -> None:
    """Đổi ngôn ngữ là đổi câu trả lời, nên phải sinh lại chứ không dùng lại."""
    chunks = [chunk_at(0, 20000)]
    keys = {
        build_requests(document_in(lang), chunks, config=config(), counter=CharCounter())[0].key
        for lang in (Language.EN, Language.VI, Language.UNKNOWN)
    }
    assert len(keys) == 3
