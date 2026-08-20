"""Test cho `pipeline.eval.spans` — ánh xạ nhãn span → chunk_id của index đang đo.

Đây là lớp làm golden set độc lập với cấu hình chunking (`TD-12`). Bất biến quan
trọng nhất mà file này canh: **ánh xạ không được im lặng biến một câu khó thành
câu bị loại**. `evaluate_run` bỏ qua câu có `relevant_chunk_ids` rỗng (nó coi đó
là câu unanswerable), nên nếu ánh xạ trả về rỗng mà không giữ nhãn cũ thì câu khó
nhất tự động rơi khỏi tập đo và recall tăng lên — đúng cái bẫy mà `W1-11` đã dựng
hàng rào để tránh.
"""

from __future__ import annotations

import pytest

from pipeline.eval.golden import GoldenQuery, QueryCategory
from pipeline.eval.spans import (
    chunks_by_document,
    resolve_queries,
    resolve_spans,
)
from rag_core.schemas import Chunk, Language, TextSpan


def _chunk(doc: str, index: int, start: int, end: int) -> Chunk:
    return Chunk(
        chunk_id=f"{doc}::{index:05d}",
        doc_id=doc,
        content="x" * max(1, end - start),
        chunk_index=index,
        start_char=start,
        end_char=end,
    )


def _tiled(doc: str, size: int, n: int) -> list[Chunk]:
    """`n` chunk kề nhau, mỗi chunk `size` ký tự."""
    return [_chunk(doc, i, i * size, (i + 1) * size) for i in range(n)]


def _span(doc: str, start: int, end: int) -> TextSpan:
    return TextSpan(doc_id=doc, start=start, end=end)


def _query(
    qid: str,
    *,
    spans: list[TextSpan] | None = None,
    chunk_ids: list[str] | None = None,
    category: QueryCategory = QueryCategory.FACTOID,
) -> GoldenQuery:
    return GoldenQuery(
        query_id=qid,
        query=f"cau hoi {qid}",
        category=category,
        lang=Language.VI,
        relevant_chunk_ids=chunk_ids or [],
        relevant_spans=spans or [],
    )


class TestChunksByDocument:
    def test_groups_and_sorts_by_offset(self) -> None:
        chunks = [_chunk("d", 2, 200, 300), _chunk("d", 0, 0, 100), _chunk("e", 0, 0, 50)]
        by_doc = chunks_by_document(chunks)
        assert set(by_doc) == {"d", "e"}
        assert [c.start_char for c in by_doc["d"]] == [0, 200]

    def test_drops_chunks_without_offsets(self) -> None:
        good = _chunk("d", 0, 0, 100)
        bad = Chunk(chunk_id="d::00001", doc_id="d", content="x", chunk_index=1)
        by_doc = chunks_by_document([good, bad])
        assert [c.chunk_id for c in by_doc["d"]] == ["d::00000"]

    def test_warns_when_offsets_missing(self, caplog: pytest.LogCaptureFixture) -> None:
        """Cả index thiếu offset đọc lên giống hệt 'retrieval quá tệ' nếu không log."""
        bad = Chunk(chunk_id="d::00000", doc_id="d", content="x", chunk_index=0)
        with caplog.at_level("WARNING"):
            chunks_by_document([bad])
        assert "không có offset" in caplog.text
        assert "--recreate" in caplog.text


class TestResolveSpans:
    def test_span_inside_one_chunk(self) -> None:
        by_doc = chunks_by_document(_tiled("d", 1000, 5))
        res = resolve_spans([_span("d", 2100, 2250)], by_doc)
        assert res.chunk_ids == ["d::00002"]
        assert res.unmatched == []

    def test_span_straddling_two_chunks_matches_both(self) -> None:
        """Chồng 50/50 thì cả hai chunk đều đạt ngưỡng — đúng, cả hai đều chứa nửa."""
        by_doc = chunks_by_document(_tiled("d", 1000, 3))
        res = resolve_spans([_span("d", 950, 1050)], by_doc)
        assert res.chunk_ids == ["d::00000", "d::00001"]

    def test_marginal_overlap_is_rejected(self) -> None:
        """Liếm 10/200 ký tự ở biên: gần như chắc chắn không chứa câu trả lời."""
        by_doc = chunks_by_document(_tiled("d", 1000, 3))
        res = resolve_spans([_span("d", 990, 1190)], by_doc)
        assert res.chunk_ids == ["d::00001"]

    def test_chunk_fully_inside_a_wide_span(self) -> None:
        """Điều kiện đối xứng: chunk nhỏ nằm trọn trong span rộng vẫn phải khớp.

        Không có nhánh này thì hạ `chunk_size` (việc đầu tiên của `W2` theo
        `TD-11`) sẽ làm 40/226 câu mất hết nhãn — đã đo thật.
        """
        by_doc = chunks_by_document(_tiled("d", 200, 10))
        res = resolve_spans([_span("d", 0, 1000)], by_doc)
        assert res.chunk_ids == [f"d::{i:05d}" for i in range(5)]

    def test_multiple_spans_are_unioned_in_order(self) -> None:
        by_doc = chunks_by_document(_tiled("d", 1000, 6))
        res = resolve_spans([_span("d", 4100, 4200), _span("d", 100, 200)], by_doc)
        assert res.chunk_ids == ["d::00004", "d::00000"], "phải giữ thứ tự span đầu vào"

    def test_duplicate_matches_appear_once(self) -> None:
        by_doc = chunks_by_document(_tiled("d", 1000, 3))
        res = resolve_spans([_span("d", 100, 200), _span("d", 300, 400)], by_doc)
        assert res.chunk_ids == ["d::00000"]

    def test_span_in_unknown_document_is_unmatched(self) -> None:
        by_doc = chunks_by_document(_tiled("d", 1000, 3))
        span = _span("khong-co", 0, 100)
        res = resolve_spans([span], by_doc)
        assert res.chunk_ids == []
        assert res.unmatched == [span]

    def test_never_matches_across_documents(self) -> None:
        """Không kiểm `doc_id` thì hai tài liệu cùng offset sẽ khớp nhãn của nhau."""
        by_doc = chunks_by_document(_tiled("d", 1000, 3) + _tiled("e", 1000, 3))
        res = resolve_spans([_span("d", 100, 200)], by_doc)
        assert res.chunk_ids == ["d::00000"]

    def test_per_span_records_each_span_separately(self) -> None:
        by_doc = chunks_by_document(_tiled("d", 1000, 3))
        res = resolve_spans([_span("d", 100, 200), _span("khong-co", 0, 50)], by_doc)
        assert res.per_span[("d", 100, 200)] == ["d::00000"]
        assert res.per_span[("khong-co", 0, 50)] == []

    def test_empty_spans(self) -> None:
        res = resolve_spans([], chunks_by_document(_tiled("d", 1000, 3)))
        assert res.chunk_ids == [] and res.unmatched == []

    @pytest.mark.parametrize("ratio", [0.0, -0.1, 1.5])
    def test_rejects_invalid_ratio(self, ratio: float) -> None:
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            resolve_spans([], {}, min_overlap_ratio=ratio)

    def test_ratio_one_requires_full_containment(self) -> None:
        by_doc = chunks_by_document(_tiled("d", 1000, 3))
        assert resolve_spans([_span("d", 950, 1050)], by_doc, min_overlap_ratio=1.0).chunk_ids == []
        assert resolve_spans([_span("d", 100, 200)], by_doc, min_overlap_ratio=1.0).chunk_ids == [
            "d::00000"
        ]


class TestResolveQueries:
    def test_recomputes_labels_from_spans(self) -> None:
        by_doc = chunks_by_document(_tiled("d", 500, 10))
        q = _query("a", spans=[_span("d", 2100, 2250)], chunk_ids=["d::00002"])
        out, report = resolve_queries([q], by_doc)
        assert out[0].relevant_chunk_ids == ["d::00004"]
        assert report.resolved == 1
        assert report.label_changed == 1

    def test_label_unchanged_when_chunking_matches(self) -> None:
        by_doc = chunks_by_document(_tiled("d", 1000, 5))
        q = _query("a", spans=[_span("d", 2100, 2250)], chunk_ids=["d::00002"])
        _, report = resolve_queries([q], by_doc)
        assert report.label_changed == 0

    def test_queries_without_spans_are_left_alone(self) -> None:
        by_doc = chunks_by_document(_tiled("d", 1000, 3))
        q = _query("a", chunk_ids=["d::00099"])
        out, report = resolve_queries([q], by_doc)
        assert out[0].relevant_chunk_ids == ["d::00099"]
        assert report.kept_chunk_ids == 1
        assert report.resolved == 0

    def test_unmatched_span_keeps_the_old_label(self) -> None:
        """Bất biến chính: không được biến câu khó thành câu bị loại.

        `evaluate_run` bỏ qua câu có `relevant_chunk_ids` rỗng. Nếu ánh xạ trả
        rỗng mà không giữ nhãn cũ thì câu đó rơi khỏi tập đo và recall tăng lên.
        """
        by_doc = chunks_by_document(_tiled("d", 1000, 3))
        q = _query("a", spans=[_span("khac", 0, 100)], chunk_ids=["d::00000"])
        out, report = resolve_queries([q], by_doc)
        assert out[0].relevant_chunk_ids == ["d::00000"], "phải giữ nhãn cũ, không được rỗng"
        assert report.unmatched_queries == ["a"]

    def test_original_queries_are_not_mutated(self) -> None:
        by_doc = chunks_by_document(_tiled("d", 500, 10))
        q = _query("a", spans=[_span("d", 2100, 2250)], chunk_ids=["d::00002"])
        resolve_queries([q], by_doc)
        assert q.relevant_chunk_ids == ["d::00002"]

    def test_unanswerable_passes_through(self) -> None:
        by_doc = chunks_by_document(_tiled("d", 1000, 3))
        q = _query("u", category=QueryCategory.UNANSWERABLE)
        out, report = resolve_queries([q], by_doc)
        assert out[0].relevant_chunk_ids == []
        assert report.kept_chunk_ids == 1

    def test_report_records_the_ratio_used(self) -> None:
        by_doc = chunks_by_document(_tiled("d", 1000, 3))
        q = _query("a", spans=[_span("d", 100, 200)])
        _, report = resolve_queries([q], by_doc, min_overlap_ratio=0.7)
        assert report.min_overlap_ratio == 0.7

    def test_survives_a_chunk_size_change_without_losing_labels(self) -> None:
        """Kịch bản `W2`: cùng nhãn span, index chunk lại nhỏ hơn."""
        span = _span("d", 1200, 1400)
        q = _query("a", spans=[span], chunk_ids=["d::00001"])

        big, _ = resolve_queries([q], chunks_by_document(_tiled("d", 1000, 5)))
        small, _ = resolve_queries([q], chunks_by_document(_tiled("d", 300, 20)))

        assert big[0].relevant_chunk_ids == ["d::00001"]
        assert small[0].relevant_chunk_ids == ["d::00004"]
        assert big[0].relevant_chunk_ids != small[0].relevant_chunk_ids
