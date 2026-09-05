"""`W4-09` — parse block CITATIONS + xác minh quote theo chunk.

Ba ca của DoD nằm ở `TestVerifyAgainstTheRightChunk`: quote thật, quote bịa,
quote sai chunk. Phần còn lại ghim hai hợp đồng mà DoD không nói ra nhưng mọi
thứ phía trên dựa vào: (1) JSON hỏng không được trả block vào văn bản nhìn
thấy; (2) chuỗi delta đã stream và `text` sau cắt là CÙNG một chuỗi — bất biến
của `CitationHoldback`, kiểm bằng cách cắt cùng một văn bản ở mọi vị trí.
"""

from __future__ import annotations

import pytest

from rag_core.generation import (
    MARKER,
    CitationHoldback,
    CitationReport,
    split_citation_block,
    verify_citations,
)
from rag_core.schemas import Chunk, DocumentMetadata

# ---------------------------------------------------------------------------
# Dữ liệu chung
# ---------------------------------------------------------------------------


def _chunk(n: int, content: str) -> Chunk:
    return Chunk(
        chunk_id=f"c{n}",
        doc_id=f"d{n}",
        content=content,
        chunk_index=0,
        section_path=[f"Chương {n}"],
        metadata=DocumentMetadata(
            source_url=f"https://example.test/doc-{n}",
            license="CC-BY-4.0",
            title=f"Tài liệu {n}",
        ),
    )


CHUNKS = [
    _chunk(1, "Tăng trưởng đạt 7,09 phần trăm năm 2024, cao hơn năm trước."),
    _chunk(2, "Xuất khẩu tăng lên 405,5 tỷ đô la trong cùng kỳ."),
]


def _block(*claims: str) -> str:
    return "CITATIONS: [" + ", ".join(claims) + "]"


# ---------------------------------------------------------------------------
# 1. split_citation_block — cắt và validate
# ---------------------------------------------------------------------------


class TestSplitCitationBlock:
    def test_a_wellformed_block_is_cut_and_parsed(self) -> None:
        parsed = split_citation_block(
            "Trả lời [1].\n" + _block('{"n": 1, "quote": "7,09 phần trăm"}')
        )
        assert parsed.block == "ok"
        assert parsed.text == "Trả lời [1]."
        assert [(c.n, c.quote) for c in parsed.claims] == [(1, "7,09 phần trăm")]

    def test_no_block_is_absent_and_the_text_is_untouched(self) -> None:
        parsed = split_citation_block("Câu trả lời không có block.")
        assert parsed.block == "absent"
        assert parsed.text == "Câu trả lời không có block."
        assert parsed.claims == ()

    def test_broken_json_is_invalid_and_never_leaks_into_the_text(self) -> None:
        """Nửa JSON hỏng trên màn hình người dùng tệ hơn một câu thiếu đuôi."""
        parsed = split_citation_block('Trả lời.\nCITATIONS: [{"n": 1, "quote"')
        assert parsed.block == "invalid"
        assert parsed.text == "Trả lời."
        assert parsed.error is not None

    def test_an_empty_tail_after_the_marker_is_invalid(self) -> None:
        parsed = split_citation_block("Trả lời.\nCITATIONS:")
        assert parsed.block == "invalid"
        assert parsed.text == "Trả lời."

    def test_an_empty_array_is_ok_and_means_no_citations(self) -> None:
        """`CITATIONS: []` là câu trả lời hợp lệ của một lời từ chối."""
        parsed = split_citation_block("Không đủ thông tin.\nCITATIONS: []")
        assert parsed.block == "ok"
        assert parsed.claims == ()

    def test_a_midline_marker_is_ordinary_text(self) -> None:
        text = "Xem mục CITATIONS: trong tài liệu."
        parsed = split_citation_block(text)
        assert parsed.block == "absent"
        assert parsed.text == text

    def test_a_marker_at_byte_zero_leaves_an_empty_answer(self) -> None:
        parsed = split_citation_block('CITATIONS: [{"n": 1, "quote": "x"}]')
        assert parsed.block == "ok"
        assert parsed.text == ""

    def test_exactly_one_preceding_newline_is_swallowed(self) -> None:
        """Nuốt đúng MỘT `\\n` — để text sau cắt khớp từng byte với chuỗi delta
        đã stream (holdback cũng chỉ giữ lại đúng một newline trước marker)."""
        parsed = split_citation_block("Trả lời.\n\nCITATIONS: []")
        assert parsed.text == "Trả lời.\n"

    def test_extra_fields_in_a_claim_are_rejected(self) -> None:
        """`extra="forbid"`: field lạ là JSON không đúng hợp đồng, không phải
        phần thưởng thêm — model gõ sai tên field mà vẫn qua là bug âm thầm."""
        parsed = split_citation_block("X.\n" + _block('{"n": 1, "quote": "q", "score": 1}'))
        assert parsed.block == "invalid"

    def test_n_zero_is_rejected_because_sources_are_numbered_from_one(self) -> None:
        parsed = split_citation_block("X.\n" + _block('{"n": 0, "quote": "q"}'))
        assert parsed.block == "invalid"

    def test_only_the_first_line_start_marker_wins(self) -> None:
        """Marker thứ hai nằm TRONG block của marker đầu — phần sau marker đầu
        là JSON (hỏng, vì chứa marker thứ hai), không phải văn bản."""
        parsed = split_citation_block("A.\nCITATIONS: []\nCITATIONS: []")
        assert parsed.text == "A."
        assert parsed.block == "invalid"


# ---------------------------------------------------------------------------
# 2. verify_citations — ba ca của DoD, và các ca quanh nó
# ---------------------------------------------------------------------------


def _verify(*claims: str) -> CitationReport:
    parsed = split_citation_block("X [1].\n" + _block(*claims))
    assert parsed.block == "ok", parsed.error
    return verify_citations(parsed, CHUNKS)


class TestVerifyAgainstTheRightChunk:
    def test_a_real_quote_is_verified(self) -> None:
        report = _verify('{"n": 1, "quote": "đạt 7,09 phần trăm năm 2024"}')
        assert [c.verified for c in report.citations] == [True]
        assert report.verified_count == 1

    def test_a_fabricated_quote_is_flagged_not_dropped(self) -> None:
        """Bịa phải HIỆN RA với `verified=False` — vứt nó đi là im lặng đúng
        kiểu mà hạng mục này tồn tại để chặn."""
        report = _verify('{"n": 1, "quote": "GDP giảm 3 phần trăm"}')
        assert len(report.citations) == 1
        assert report.citations[0].verified is False
        assert report.citations[0].quote == "GDP giảm 3 phần trăm"

    def test_a_quote_from_the_wrong_chunk_is_not_verified(self) -> None:
        """Quote nguyên văn 100%% — nhưng của chunk [2], cite thành [1]. Số
        nguồn sai dẫn người đọc tới nhầm tài liệu, tệ ngang quote bịa."""
        report = _verify('{"n": 1, "quote": "405,5 tỷ đô la"}')
        assert report.citations[0].verified is False

    def test_the_same_quote_cited_correctly_is_verified(self) -> None:
        report = _verify('{"n": 2, "quote": "405,5 tỷ đô la"}')
        assert report.citations[0].verified is True

    def test_an_out_of_range_n_has_no_chunk_to_pin_to(self) -> None:
        """`n=9` không có chunk nào để gắn — dựng một `Citation` với `chunk_id`
        bịa để nhét vào danh sách là tự làm điều mình đang bắt model."""
        report = _verify('{"n": 9, "quote": "bất kỳ"}')
        assert report.citations == ()
        assert report.invalid_ns == (9,)

    def test_whitespace_differences_do_not_reject_a_real_quote(self) -> None:
        """Stream trả markdown nên xuống dòng/khoảng trắng không ổn định."""
        report = _verify('{"n": 1, "quote": "đạt  7,09\\nphần trăm"}')
        assert report.citations[0].verified is True


class TestEllipsisQuotes:
    """`NEW-08`/`TD-64`: dấu lược là "bỏ một quãng", không phải chữ.

    `W5-02` đo được matcher chuỗi-con từ chối oan **19/67** lỗi cấp quote —
    toàn bộ là hai mẩu nguyên văn nối bằng `...`, một cách trích dẫn hợp lệ
    trong văn viết. Đây là phép đo mà docstring `citations.py` đòi trước khi
    nới bất kỳ quy tắc nào.
    """

    def test_two_real_segments_joined_by_an_ellipsis_are_verified(self) -> None:
        report = _verify('{"n": 1, "quote": "Tăng trưởng đạt ... cao hơn năm trước"}')
        assert report.citations[0].verified is True

    def test_the_unicode_and_bracketed_forms_count_too(self) -> None:
        for mark in ("…", "[...]", "[…]"):
            report = _verify(f'{{"n": 1, "quote": "Tăng trưởng {mark} năm 2024"}}')
            assert report.citations[0].verified is True, mark

    def test_segments_in_the_wrong_order_are_rejected(self) -> None:
        """Thứ tự là phần giữ độ chặt: hai mẩu có thật nhưng đảo chiều là một
        câu mà chunk không nói."""
        report = _verify('{"n": 1, "quote": "cao hơn năm trước ... Tăng trưởng đạt"}')
        assert report.citations[0].verified is False

    def test_one_fabricated_segment_rejects_the_whole_quote(self) -> None:
        report = _verify('{"n": 1, "quote": "Tăng trưởng đạt ... GDP giảm 3 phần trăm"}')
        assert report.citations[0].verified is False

    def test_a_quote_that_is_only_an_ellipsis_says_nothing(self) -> None:
        """Trước sửa này `"..." in content` có thể `True` — một quote không
        nói gì cả mà được đóng dấu verified."""
        report = _verify('{"n": 1, "quote": "..."}')
        assert report.citations[0].verified is False

    def test_overlapping_segments_cannot_be_counted_twice(self) -> None:
        """`find` tiếp tục từ cuối mảnh trước: hai mảnh trỏ vào CÙNG một đoạn
        chunk không được tính là hai bằng chứng."""
        report = _verify('{"n": 1, "quote": "đạt 7,09 phần trăm ... đạt 7,09 phần trăm"}')
        assert report.citations[0].verified is False

    def test_case_differences_do_reject(self) -> None:
        """Đổi hoa thường là SỬA CHỮ — đúng loại 'sửa nhẹ' mà xác minh tồn tại
        để bắt. Nới quy tắc này phải kèm phép đo cho thấy nó từ chối oan."""
        report = _verify('{"n": 1, "quote": "tăng trưởng đạt 7,09"}')
        assert report.citations[0].verified is False

    def test_chunk_identity_and_metadata_are_resolved_on_our_side(self) -> None:
        report = _verify('{"n": 2, "quote": "405,5 tỷ đô la"}')
        citation = report.citations[0]
        assert citation.chunk_id == "c2"
        assert citation.doc_id == "d2"
        assert citation.source_url == "https://example.test/doc-2"
        assert citation.section_path == ["Chương 2"]

    def test_the_frame_counts_claims_not_just_citations(self) -> None:
        report = _verify(
            '{"n": 1, "quote": "đạt 7,09 phần trăm năm 2024"}',
            '{"n": 9, "quote": "ngoài phạm vi"}',
        )
        frame = report.as_frame()
        assert frame["total"] == 2
        assert frame["verified"] == 1
        assert frame["invalid_ns"] == [9]


# ---------------------------------------------------------------------------
# 3. CitationHoldback — bất biến: emitted == split(...).text, ở MỌI cách cắt
# ---------------------------------------------------------------------------

SAMPLE = "Trả lời có [1] nguồn.\nDòng hai.\n" + _block('{"n": 1, "quote": "q"}')


def _stream(text: str, cuts: list[int]) -> str:
    holdback = CitationHoldback()
    out: list[str] = []
    last = 0
    for cut in [*cuts, len(text)]:
        out.append(holdback.feed(text[last:cut]))
        last = cut
    out.append(holdback.flush())
    return "".join(out)


class TestCitationHoldback:
    def test_every_two_way_split_emits_exactly_the_visible_text(self) -> None:
        """Marker bị cắt đôi ở MỌI vị trí — kể cả giữa chữ `CITATIONS:`."""
        expected = split_citation_block(SAMPLE).text
        for cut in range(len(SAMPLE) + 1):
            assert _stream(SAMPLE, [cut]) == expected, f"cắt tại {cut}"

    def test_character_by_character_streaming_holds_the_block(self) -> None:
        expected = split_citation_block(SAMPLE).text
        assert _stream(SAMPLE, list(range(1, len(SAMPLE)))) == expected

    def test_nothing_of_the_marker_ever_reaches_a_delta(self) -> None:
        holdback = CitationHoldback()
        emitted = [holdback.feed(SAMPLE[i : i + 3]) for i in range(0, len(SAMPLE), 3)]
        emitted.append(holdback.flush())
        assert "CITAT" not in "".join(emitted)

    def test_text_without_a_block_is_emitted_in_full(self) -> None:
        text = "Không có block nào ở đây.\nKể cả dòng hai."
        assert _stream(text, [5, 17]) == text

    def test_a_midline_marker_is_streamed_as_ordinary_text(self) -> None:
        text = "Xem mục CITATIONS: nhé."
        assert _stream(text, [10]) == text

    def test_a_marker_at_stream_start_hides_everything(self) -> None:
        assert _stream('CITATIONS: [{"n": 1, "quote": "x"}]', [4]) == ""

    def test_an_unfinished_marker_at_stream_end_is_flushed_as_text(self) -> None:
        """Model chết đúng giữa chữ `CITATIONS` — phần giữ lại là văn bản
        thường, flush phải trả nó ra thay vì nuốt mất."""
        text = "Trả lời.\nCITATION"
        assert _stream(text, [12]) == text


@pytest.mark.parametrize("marker", [MARKER])
def test_the_marker_constant_is_what_the_prompt_promises(marker: str) -> None:
    """`SYSTEM_PROMPT` (serving) viết mẫu `CITATIONS:` — đổi hằng số bên này mà
    quên bên kia thì mọi block đều thành `absent` và không gì đỏ ngoài test này."""
    from serving.core.chat import SYSTEM_PROMPT

    assert marker in SYSTEM_PROMPT
