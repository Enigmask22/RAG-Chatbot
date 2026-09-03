"""`TD-32` — gộp nhiều chunk vào một lời gọi, và chốt chặn chống lệch thứ tự.

Chế độ gộp tồn tại vì một phép đo: GLM-5.3-Flash **không** có prefix caching
(2,1% ở concurrency 16, **0,1% ở concurrency 1**, trong khi tiền tố dùng chung là
54,8% mỗi prompt). Không có cache thì khối `<document_head>` bị trả tiền lại cho
từng chunk trong ~185 chunk của tài liệu, và cả corpus tốn ~$10,6 thay vì ~$2,5.

⚠️⚠️ Nhưng gộp mở ra một lỗi mà chế độ một-chunk không có: **model trả đủ N dòng,
đúng số thứ tự, mỗi dòng một câu hợp lệ — nhưng gán ngữ cảnh của passage 2 cho
dòng 1.** Không có gì đỏ. Chunk nhận nhầm ngữ cảnh đi thẳng vào vector và chỉ
hiện ra dưới dạng metric tệ hơn mà không ai truy được vì sao.

Nên phần lớn nhóm test này không kiểm "bóc tách có chạy không" mà kiểm **cái gì
bị TỪ CHỐI**. Một parser khoan dung ở đây là parser ghi dữ liệu sai vào index.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from rag_core.chunking.contextual import (
    BATCH_SYSTEM_PROMPT,
    CONTEXT_SYSTEM_PROMPT,
    ECHO_SEPARATOR,
    ECHO_WORDS,
    BatchParseError,
    ContextRequest,
    build_requests,
    parse_response,
)
from rag_core.schemas import Chunk

from .test_contextual_chunking import CharCounter, chunk_at, config, document


def chunks_at(n: int, *, start: int = 4000, length: int = 300) -> list[Chunk]:
    """`n` chunk **liền kề**, đúng hình dạng mà chế độ gộp nhắm tới."""
    return [chunk_at(i, start + i * length, length) for i in range(n)]


def build(chunks: Sequence[Chunk], **overrides: object) -> list[ContextRequest]:
    return build_requests(document(), chunks, config=config(**overrides), counter=CharCounter())


def reply(request: ContextRequest, contexts: Sequence[str]) -> str:
    """Trả lời hợp lệ, echo lấy từ chính request — dùng làm mốc cho các ca hỏng."""
    return "\n".join(
        f"[{i}] {echo}{ECHO_SEPARATOR}{text}"
        for i, (echo, text) in enumerate(zip(request.echoes, contexts, strict=True), start=1)
    )


# --------------------------------------------------------------------------
# 1. Gom nhóm và dựng prompt
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_chunks", "batch_size", "expected"),
    [(8, 8, [8]), (15, 8, [8, 7]), (16, 8, [8, 8]), (3, 8, [3]), (5, 1, [1, 1, 1, 1, 1])],
)
def test_chunks_are_grouped_into_batches(
    n_chunks: int, batch_size: int, expected: list[int]
) -> None:
    """Nhóm cuối ngắn hơn là chuyện bình thường; bỏ sót nó thì không.

    Cắt theo `range(0, n, B)` nên nhóm cuối tự nhiên ngắn. Test này ghim rằng
    **mọi** chunk đều nằm trong đúng một nhóm — mất chunk cuối là lỗi im lặng,
    nó chỉ hiện ra thành `coverage` thiếu vài phần trăm ở `apply_contexts`.
    """
    requests = build(chunks_at(n_chunks), batch_size=batch_size)
    assert [r.n_chunks for r in requests] == expected
    assert sum(r.n_chunks for r in requests) == n_chunks


def test_every_chunk_appears_exactly_once_across_batches() -> None:
    requests = build(chunks_at(15), batch_size=4)
    seen = [cid for r in requests for cid in r.chunk_ids]
    assert seen == [c.chunk_id for c in chunks_at(15)]
    assert len(set(seen)) == len(seen)


def test_batch_of_one_still_uses_the_single_chunk_prompt() -> None:
    """`batch_size=1` phải đi **đường cũ**, không phải đường gộp với `n=1`.

    Prompt một-chunk là thứ đã sinh ra 860 ngữ cảnh dùng được. Bắt nó đi qua định
    dạng đánh số + echo chỉ để thống nhất hình thức là đổi một đường đã kiểm lấy
    một đường chưa kiểm mà không được gì.
    """
    request = build(chunks_at(1), batch_size=1)[0]
    assert request.messages[0].content == CONTEXT_SYSTEM_PROMPT
    assert "<chunk>" in request.user_text
    assert "<passages>" not in request.user_text


def test_batch_prompt_numbers_every_passage() -> None:
    request = build(chunks_at(6), batch_size=6)[0]
    assert request.messages[0].content == BATCH_SYSTEM_PROMPT
    for i in range(1, 7):
        assert f"[{i}]" in request.user_text


def test_batch_prompt_does_not_repeat_the_group_text_in_before_or_after() -> None:
    """⭐ `<before>`/`<after>` là **phần ngoài** nhóm, không phải cửa sổ bao trùm.

    Các chunk trong nhóm liền kề nhau nên chúng đã là vùng lân cận của nhau. Đưa
    thêm một `<neighbourhood>` bao trùm như chế độ một-chunk sẽ lặp lại toàn bộ
    văn bản của nhóm lần thứ hai trong cùng một prompt — tức trả tiền hai lần cho
    đúng thứ mà chế độ gộp sinh ra để khỏi phải trả nhiều lần.
    """
    group = chunks_at(6)
    request = build(group, batch_size=6)[0]
    user = request.user_text
    before = user.split("<before>", 1)[1].split("</before>", 1)[0] if "<before>" in user else ""
    after = user.split("<after>", 1)[1].split("</after>", 1)[0] if "<after>" in user else ""
    for chunk in group:
        needle = chunk.content.strip()[:60]
        assert needle not in before
        assert needle not in after


def test_head_appears_once_per_batch_not_once_per_chunk() -> None:
    """Đây chính là khoản tiết kiệm. Head xuất hiện hai lần nghĩa là không tiết kiệm gì."""
    request = build(chunks_at(8), batch_size=8)[0]
    assert request.user_text.count("<document_head>") == 1


def test_batching_cuts_estimated_tokens_per_chunk() -> None:
    """⭐ Ghim chính lý do tồn tại của `TD-32`, bằng số chứ không bằng lời."""
    single = build(chunks_at(8), batch_size=1)
    batched = build(chunks_at(8), batch_size=8)
    per_chunk_single = sum(r.est_prompt_tokens for r in single) / 8
    per_chunk_batched = sum(r.est_prompt_tokens for r in batched) / 8
    assert per_chunk_batched < per_chunk_single / 2


def test_batches_of_one_document_share_the_head_prefix() -> None:
    """Bất biến tiền tố vẫn phải giữ — vLLM cache theo tiền tố token."""
    requests = build(chunks_at(16), batch_size=8)
    a, b = requests[0].user_text, requests[1].user_text
    shared = len(a) - len(a.lstrip()) if a == b else _common_prefix(a, b)
    assert shared > 200


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b, strict=False):
        if ca != cb:
            break
        n += 1
    return n


def test_echoes_are_recorded_for_every_chunk_in_the_batch() -> None:
    group = chunks_at(5)
    request = build(group, batch_size=5)[0]
    assert len(request.echoes) == 5
    for echo, chunk in zip(request.echoes, group, strict=True):
        assert echo == " ".join(chunk.content.split()[:ECHO_WORDS])


def test_batch_size_is_part_of_the_cache_key() -> None:
    """Đổi `batch_size` là đổi prompt, nên phải sinh lại chứ không dùng lại artifact."""
    single = build(chunks_at(2), batch_size=1)[0]
    batched = build(chunks_at(2), batch_size=2)[0]
    assert single.key != batched.key


# --------------------------------------------------------------------------
# 2. Bóc tách — ca chạy được
# --------------------------------------------------------------------------


def test_single_mode_returns_the_text_as_is() -> None:
    request = build(chunks_at(1), batch_size=1)[0]
    assert parse_response(request, "  Bối cảnh.  ") == {request.chunk_ids[0]: "Bối cảnh."}


def test_batch_happy_path_maps_each_context_to_its_chunk() -> None:
    request = build(chunks_at(3), batch_size=3)[0]
    out = parse_response(request, reply(request, ["ctx một", "ctx hai", "ctx ba"]))
    assert out == dict(zip(request.chunk_ids, ["ctx một", "ctx hai", "ctx ba"], strict=True))


@pytest.mark.parametrize("marker", ["[{i}]", "{i}.", "({i})", "{i}:"])
def test_numbering_format_is_tolerated(marker: str) -> None:
    """Khoan dung với **hình thức** đánh số, nghiêm khắc với **nội dung** — hai chuyện khác nhau.

    Từ chối cả lô vì model viết `1.` thay vì `[1]` là tự tạo tỉ lệ hỏng trên một
    job 15.814 chunk mà không đổi lấy được an toàn nào: số thứ tự vẫn đọc ra
    đúng, và thứ thật sự canh việc gán đúng chunk là echo chứ không phải dấu ngoặc.
    """
    request = build(chunks_at(2), batch_size=2)[0]
    body = "\n".join(
        marker.format(i=i) + f" {echo}{ECHO_SEPARATOR}ctx {i}"
        for i, echo in enumerate(request.echoes, start=1)
    )
    assert len(parse_response(request, body)) == 2


def test_context_spanning_several_lines_is_joined_not_rejected() -> None:
    request = build(chunks_at(2), batch_size=2)[0]
    body = (
        f"[1] {request.echoes[0]}{ECHO_SEPARATOR}Câu một.\nCâu hai của cùng passage.\n"
        f"[2] {request.echoes[1]}{ECHO_SEPARATOR}Câu ba."
    )
    out = parse_response(request, body)
    assert out[request.chunk_ids[0]] == "Câu một. Câu hai của cùng passage."
    assert out[request.chunk_ids[1]] == "Câu ba."


def test_context_containing_the_separator_keeps_everything_after_the_first() -> None:
    """Cắt ở lần xuất hiện ĐẦU TIÊN, nên `||` trong ngữ cảnh vô hại."""
    request = build(chunks_at(1), batch_size=1)[0]
    request = build(chunks_at(2), batch_size=2)[0]
    body = (
        f"[1] {request.echoes[0]}{ECHO_SEPARATOR}a || b\n[2] {request.echoes[1]}{ECHO_SEPARATOR}c"
    )
    assert parse_response(request, body)[request.chunk_ids[0]] == "a || b"


# --------------------------------------------------------------------------
# 3. Bóc tách — cái gì phải BỊ TỪ CHỐI
# --------------------------------------------------------------------------


def test_missing_a_line_rejects_the_whole_batch() -> None:
    """Từ chối cả lô, kể cả phần bóc được.

    Thiếu một dòng nghĩa là **không biết** dòng nào ứng với passage nào — model
    có thể đã bỏ passage 2 hoặc đã đánh số lệch. Ghi phần "có vẻ đúng" là đúng
    kiểu lỗi mà echo sinh ra để chặn.
    """
    request = build(chunks_at(3), batch_size=3)[0]
    body = reply(request, ["a", "b", "c"]).split("\n")
    with pytest.raises(BatchParseError, match="bóc ra"):
        parse_response(request, "\n".join([body[0], body[2]]))


def test_an_extra_line_rejects_the_whole_batch() -> None:
    request = build(chunks_at(2), batch_size=2)[0]
    body = reply(request, ["a", "b"]) + f"\n[3] x{ECHO_SEPARATOR}thừa"
    with pytest.raises(BatchParseError, match="bóc ra"):
        parse_response(request, body)


def test_zero_based_numbering_rejects_the_batch() -> None:
    """Đánh số từ 0 làm mọi ngữ cảnh lệch đúng một chunk — ca xấu nhất, và nó im lặng."""
    request = build(chunks_at(3), batch_size=3)[0]
    body = "\n".join(
        f"[{i}] {echo}{ECHO_SEPARATOR}ctx" for i, echo in enumerate(request.echoes, start=0)
    )
    with pytest.raises(BatchParseError):
        parse_response(request, body)


def test_a_line_without_the_separator_rejects_the_batch() -> None:
    request = build(chunks_at(2), batch_size=2)[0]
    body = f"[1] chỉ có văn bản không có ngăn cách\n[2] {request.echoes[1]}{ECHO_SEPARATOR}b"
    with pytest.raises(BatchParseError, match="ngữ cảnh"):
        parse_response(request, body)


def test_an_echo_matching_nothing_is_rejected_because_it_cannot_be_confirmed() -> None:
    """⭐⭐ Không có bằng chứng lệch **không phải** bằng chứng không lệch.

    Bản đầu cho ca này đi qua với lý do "echo không giống passage nào nên không
    kết luận được". Một ca thật trong lượt canary phá lý lẽ ấy: echo
    `"hiện có. Việt Nam có thể"` trong khi passage `[2]` mở đầu bằng `"hỗ trợ
    theo các"` — hai đoạn khác hẳn nhau, và không phép so nào nói được nó thuộc
    về đâu.

    Đường lùi một-chunk **không thể lệch** và giá của nó biết trước, nên khi
    không xác nhận được thì trả về đường lùi: đổi một rủi ro im lặng không chặn
    trên lấy một khoản chi đo được.
    """
    request = build(chunks_at(2), batch_size=2)[0]
    body = (
        f"[1] hoàn toàn không liên quan gì{ECHO_SEPARATOR}a"
        f"\n[2] {request.echoes[1]}{ECHO_SEPARATOR}b"
    )
    with pytest.raises(BatchParseError, match="không xác nhận được"):
        parse_response(request, body)


def test_a_passage_too_short_to_identify_is_not_judged() -> None:
    """Ngoại lệ duy nhất, và nó nằm ở **văn bản gốc** chứ không ở câu model trả lời.

    Passage mở đầu bằng rác OCR quá ngắn thì không phép so nào định danh được nó
    — `"g g n"` (3 ký tự sau khi ép) từng cho điểm cao hơn với passage 4 (0,86)
    so với passage 1 đúng của nó (0,75). Ở đó chốt chặn im lặng không xét, thay
    vì báo đạt hay báo hỏng.
    """
    request = build(chunks_at(2), batch_size=2)[0]
    stubby = ContextRequest(
        key=request.key,
        chunk_ids=request.chunk_ids,
        doc_id=request.doc_id,
        messages=request.messages,
        est_prompt_tokens=request.est_prompt_tokens,
        echoes=("g g n", "h b g"),
    )
    body = f"[1] bất kỳ điều gì{ECHO_SEPARATOR}a\n[2] gì cũng được{ECHO_SEPARATOR}b"
    assert len(parse_response(stubby, body)) == 2


def test_swapped_passages_are_caught_by_the_echo() -> None:
    """⭐⭐ Đúng lỗi mà `ECHO_WORDS` tồn tại để bắt, dựng lại nguyên hình dạng.

    Model mô tả passage 2 ở dòng 1 và passage 1 ở dòng 2. Đủ số dòng, đúng số thứ
    tự, mỗi dòng một câu hợp lệ — **không một phép kiểm cấu trúc nào bắt được**.
    Chỉ echo bắt được, vì dòng 1 chép lại 4 từ đầu của passage 2.
    """
    request = build(chunks_at(2), batch_size=2)[0]
    swapped = (
        f"[1] {request.echoes[1]}{ECHO_SEPARATOR}ngữ cảnh của passage 2\n"
        f"[2] {request.echoes[0]}{ECHO_SEPARATOR}ngữ cảnh của passage 1"
    )
    with pytest.raises(BatchParseError, match="echo"):
        parse_response(request, swapped)


def test_echo_comparison_ignores_case_punctuation_and_spacing() -> None:
    """Khoan dung đúng chỗ: model chép lại có thể đổi hoa thường hoặc thêm dấu nháy.

    Nếu chốt chặn từ chối vì một dấu phẩy thì nó sẽ bị ai đó tắt đi, và khi ấy
    nó không còn bảo vệ gì nữa.
    """
    request = build(chunks_at(2), batch_size=2)[0]
    noisy = request.echoes[0].upper().replace(" ", "  ") + ","
    body = f'[1] "{noisy}"{ECHO_SEPARATOR}a\n[2] {request.echoes[1]}{ECHO_SEPARATOR}b'
    assert len(parse_response(request, body)) == 2


def test_missing_echoes_on_the_request_skips_the_check_rather_than_passing_it() -> None:
    """Request cũ không mang `echoes`. Chốt chặn không kiểm được thì **im lặng không kiểm**.

    Ranh giới quan trọng: không kiểm được ≠ đã kiểm và đạt. Ca này còn bóc tách
    được, nhưng nó không được phép mang tiếng là đã qua chốt chặn.
    """
    request = build(chunks_at(2), batch_size=2)[0]
    bare = ContextRequest(
        key=request.key,
        chunk_ids=request.chunk_ids,
        doc_id=request.doc_id,
        messages=request.messages,
        est_prompt_tokens=request.est_prompt_tokens,
    )
    body = f"[1] bất kỳ{ECHO_SEPARATOR}a\n[2] gì cũng được{ECHO_SEPARATOR}b"
    assert len(parse_response(bare, body)) == 2
    assert bare.echoes == ()


def test_empty_response_yields_nothing_rather_than_raising() -> None:
    """Rỗng là "model không nói gì", không phải "định dạng hỏng" — hai đường xử lý khác nhau."""
    single = build(chunks_at(1), batch_size=1)[0]
    assert parse_response(single, "   ") == {}


# --------------------------------------------------------------------------
# 4. Trần output
# --------------------------------------------------------------------------


def test_n_chunks_reports_the_batch_size() -> None:
    """Vòng chạy nhân `--max-tokens` với con số này.

    Không nhân thì lô 8 chunk chạy với trần của một chunk, bị cắt lời ở chunk thứ
    hai, và triệu chứng là `BatchParseError` hàng loạt — một thông báo không hề
    nói ra nguyên nhân.
    """
    assert build(chunks_at(8), batch_size=8)[0].n_chunks == 8
    assert build(chunks_at(1), batch_size=1)[0].n_chunks == 1
