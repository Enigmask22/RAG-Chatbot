"""Test cho `pipeline.goldenset.anchor` — neo nhãn nháp vào văn bản gốc.

Phần khó nhất và cũng dễ sai nhất là `find_ignoring_whitespace`: trích dẫn được
model viện dẫn đã được đối chiếu với `chunk.content`, mà content thì bị splitter
đổi khoảng trắng (nối bằng `"\\n"` hoặc `" "` bất kể nguyên bản ngăn nhau bằng gì).
So thô bằng `str.find` sẽ trượt phần lớn, và trượt thì span phải lấy cả chunk —
mất độ chính xác chứ không lỗi, tức không có triệu chứng.
"""

from __future__ import annotations

import pytest

from pipeline.eval.golden import GoldenQuery, QueryCategory
from pipeline.goldenset.anchor import (
    anchor_drafts,
    build_chunk_index,
    find_ignoring_whitespace,
)
from pipeline.goldenset.schema import DraftProvenance, GoldenDraft
from rag_core.schemas import Chunk, Language


def _chunk(doc: str, index: int, start: int, end: int, content: str = "noi dung") -> Chunk:
    return Chunk(
        chunk_id=f"{doc}::{index:05d}",
        doc_id=doc,
        content=content,
        chunk_index=index,
        start_char=start,
        end_char=end,
    )


def _draft(
    qid: str,
    *,
    chunk_ids: list[str],
    quotes: list[str],
    category: QueryCategory = QueryCategory.FACTOID,
) -> GoldenDraft:
    ids = [] if category is QueryCategory.UNANSWERABLE else chunk_ids
    return GoldenDraft(
        query=GoldenQuery(
            query_id=qid,
            query=f"cau hoi {qid}",
            category=category,
            lang=Language.VI,
            relevant_chunk_ids=ids,
        ),
        provenance=DraftProvenance(
            generator_model="fake",
            generator_model_requested="fake",
            category_requested=category,
            source_chunk_ids=ids,
            supporting_quotes=quotes,
            quotes_verified=True,
        ),
    )


class TestFindIgnoringWhitespace:
    def test_exact_match(self) -> None:
        text = "Đầu tiên là câu này. Rồi câu khác."
        assert find_ignoring_whitespace(text, "Rồi câu khác.") == (21, 34)

    def test_newline_in_document_space_in_quote(self) -> None:
        """Đúng trường hợp thật: splitter đã đổi `"\\n"` thành `" "`."""
        text = "Tăng trưởng GDP\nđạt 6,7% trong năm 2017."
        found = find_ignoring_whitespace(text, "Tăng trưởng GDP đạt 6,7%")
        assert found is not None
        assert text[found[0] : found[1]] == "Tăng trưởng GDP\nđạt 6,7%"

    def test_collapses_runs_of_whitespace(self) -> None:
        text = "cột một     cột hai"
        assert find_ignoring_whitespace(text, "cột một cột hai") == (0, 19)

    def test_quote_with_extra_whitespace(self) -> None:
        text = "một hai ba"
        assert find_ignoring_whitespace(text, "  một   hai  ba  ") == (0, 10)

    def test_tab_and_newline_are_equivalent(self) -> None:
        text = "a\tb\nc"
        assert find_ignoring_whitespace(text, "a b c") == (0, 5)

    def test_not_found(self) -> None:
        assert find_ignoring_whitespace("abc", "xyz") is None

    def test_empty_needle(self) -> None:
        assert find_ignoring_whitespace("abc", "   ") is None

    def test_region_restricts_the_search(self) -> None:
        """Trích dẫn trùng chữ ở hai chỗ: `region` quyết định lấy chỗ nào."""
        text = "mục tiêu tăng trưởng. " * 3
        one = len("mục tiêu tăng trưởng. ")
        found = find_ignoring_whitespace(text, "mục tiêu", region=(one, 2 * one))
        assert found is not None and found[0] == one

    def test_region_outside_returns_none(self) -> None:
        assert find_ignoring_whitespace("abcdef", "abc", region=(3, 6)) is None

    def test_region_clamped_to_bounds(self) -> None:
        assert find_ignoring_whitespace("abc", "abc", region=(-10, 999)) == (0, 3)

    def test_empty_region_returns_none(self) -> None:
        assert find_ignoring_whitespace("abc", "a", region=(2, 2)) is None

    def test_offsets_are_absolute_not_region_relative(self) -> None:
        """Trả offset tương đối với `region` là lỗi âm thầm: span neo lệch."""
        text = "xxxxx" + "cần tìm"
        found = find_ignoring_whitespace(text, "cần tìm", region=(5, len(text)))
        assert found == (5, len(text))


class TestBuildChunkIndex:
    def test_maps_by_chunk_id(self) -> None:
        idx = build_chunk_index([_chunk("d", 0, 0, 10), _chunk("d", 1, 10, 20)])
        assert set(idx) == {"d::00000", "d::00001"}

    def test_duplicate_chunk_id_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="hai lần"):
            build_chunk_index([_chunk("d", 0, 0, 10), _chunk("d", 0, 10, 20)])


class TestAnchorDrafts:
    def test_narrows_span_to_the_quote(self) -> None:
        text = "Mở đầu không liên quan. GDP đạt 6,7% năm 2017. Kết thúc."
        idx = build_chunk_index([_chunk("d", 0, 0, len(text))])
        draft = _draft("a", chunk_ids=["d::00000"], quotes=["GDP đạt 6,7% năm 2017."])

        out, report = anchor_drafts([draft], idx, {"d": text})
        spans = out[0].query.relevant_spans
        assert len(spans) == 1
        assert text[spans[0].start : spans[0].end] == "GDP đạt 6,7% năm 2017."
        assert report.narrowed_by_quote == 1
        assert report.widened_to_chunk == 0

    def test_falls_back_to_chunk_span(self) -> None:
        text = "Nội dung tài liệu đủ dài để làm một chunk."
        idx = build_chunk_index([_chunk("d", 0, 5, 20)])
        draft = _draft("a", chunk_ids=["d::00000"], quotes=["câu này không có trong tài liệu"])

        out, report = anchor_drafts([draft], idx, {"d": text})
        assert out[0].query.relevant_spans[0].start == 5
        assert out[0].query.relevant_spans[0].end == 20
        assert report.widened_to_chunk == 1

    def test_short_quote_is_not_used_for_narrowing(self) -> None:
        """Chuỗi ngắn xuất hiện ở nhiều chỗ — thu span theo nó là neo vào chỗ sai."""
        text = "GDP tăng. " * 10
        idx = build_chunk_index([_chunk("d", 0, 0, len(text))])
        draft = _draft("a", chunk_ids=["d::00000"], quotes=["GDP"])

        out, report = anchor_drafts([draft], idx, {"d": text})
        assert report.widened_to_chunk == 1
        assert out[0].query.relevant_spans[0].end == len(text)

    def test_quote_search_is_confined_to_the_chunk(self) -> None:
        """Trích dẫn khớp ở chunk khác thì KHÔNG được lấy — span sẽ neo lệch."""
        text = "phần đầu: chỉ số quan trọng nhất. phần sau: chỉ số quan trọng nhất."
        idx = build_chunk_index([_chunk("d", 1, 33, len(text))])
        draft = _draft("a", chunk_ids=["d::00001"], quotes=["chỉ số quan trọng nhất"])

        out, _ = anchor_drafts([draft], idx, {"d": text})
        span = out[0].query.relevant_spans[0]
        assert span.start >= 33, "phải khớp trong chunk của mình, không phải lần đầu của tài liệu"

    def test_multiple_chunks_one_quote(self) -> None:
        """Câu aggregation: 1 trích dẫn, 3 chunk → 1 span thu hẹp + 2 span rộng."""
        text = "A" * 100 + "trích dẫn dài đủ để vượt ngưỡng" + "B" * 100
        chunks = [
            _chunk("d", 0, 0, 100),
            _chunk("d", 1, 100, 131),
            _chunk("d", 2, 131, len(text)),
        ]
        draft = _draft(
            "a",
            chunk_ids=["d::00000", "d::00001", "d::00002"],
            quotes=["trích dẫn dài đủ để vượt ngưỡng"],
        )
        out, report = anchor_drafts([draft], build_chunk_index(chunks), {"d": text})
        assert len(out[0].query.relevant_spans) == 3
        assert report.narrowed_by_quote == 1
        assert report.widened_to_chunk == 2

    def test_unanswerable_gets_no_spans(self) -> None:
        draft = _draft("u", chunk_ids=[], quotes=[], category=QueryCategory.UNANSWERABLE)
        out, report = anchor_drafts([draft], {}, {})
        assert out[0].query.relevant_spans == []
        assert report.answerable == 0

    def test_missing_chunk_is_reported(self) -> None:
        draft = _draft("a", chunk_ids=["d::00099"], quotes=["x" * 30])
        out, report = anchor_drafts([draft], {}, {"d": "abc"})
        assert report.missing_chunk_ids == ["d::00099"]
        assert report.unanchored_query_ids == ["a"]
        assert out[0].query.relevant_spans == []

    def test_chunk_without_offsets_counts_as_missing(self) -> None:
        stale = Chunk(chunk_id="d::00000", doc_id="d", content="x", chunk_index=0)
        draft = _draft("a", chunk_ids=["d::00000"], quotes=["x" * 30])
        _, report = anchor_drafts([draft], {"d::00000": stale}, {"d": "abc"})
        assert report.missing_chunk_ids == ["d::00000"]

    def test_original_chunk_ids_are_kept(self) -> None:
        """Giữ nhãn cũ là điều kiện để đối chiếu ánh xạ span có ra đúng tập cũ."""
        text = "abcdefghij" * 10
        idx = build_chunk_index([_chunk("d", 0, 0, len(text))])
        draft = _draft("a", chunk_ids=["d::00000"], quotes=[])
        out, _ = anchor_drafts([draft], idx, {"d": text})
        assert out[0].query.relevant_chunk_ids == ["d::00000"]

    def test_duplicate_spans_collapsed(self) -> None:
        """Hai chunk cùng khớp một trích dẫn thì chỉ ra một span."""
        text = "trích dẫn dài đủ để vượt ngưỡng tối thiểu"
        chunks = [_chunk("d", 0, 0, len(text)), _chunk("d", 1, 0, len(text))]
        draft = _draft(
            "a", chunk_ids=["d::00000", "d::00001"], quotes=["trích dẫn dài đủ để vượt ngưỡng"]
        )
        out, _ = anchor_drafts([draft], build_chunk_index(chunks), {"d": text})
        assert len(out[0].query.relevant_spans) == 1

    def test_order_is_preserved(self) -> None:
        text = "abcdefghij" * 20
        idx = build_chunk_index([_chunk("d", i, i * 50, (i + 1) * 50) for i in range(4)])
        drafts = [_draft(q, chunk_ids=[f"d::{i:05d}"], quotes=[]) for i, q in enumerate("zabc")]
        out, _ = anchor_drafts(drafts, idx, {"d": text})
        assert [d.query.query_id for d in out] == list("zabc")

    def test_document_text_missing_falls_back_to_chunk_span(self) -> None:
        idx = build_chunk_index([_chunk("d", 0, 10, 40)])
        draft = _draft("a", chunk_ids=["d::00000"], quotes=["x" * 30])
        out, report = anchor_drafts([draft], idx, {})
        assert out[0].query.relevant_spans[0].start == 10
        assert report.widened_to_chunk == 1
