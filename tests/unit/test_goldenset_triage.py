"""Test cho `pipeline.goldenset.triage`.

Dùng retriever giả có thể lập trình từng câu trả lời, nên test được đúng những
kịch bản khó dựng bằng index thật: chunk_id chết, câu unanswerable trúng đậm,
câu trả lời được mà retriever trượt hoàn toàn.

Điều quan trọng nhất mà file này canh là **tính bất đối xứng** giữa hai loại tín
hiệu (xem docstring của module). Nếu ai đó "cải tiến" triage bằng cách xếp
`answerable_but_not_retrieved` lên đầu hàng đợi, hoặc cho nó một
`suggested_decision` khác `accept`, thì test ở đây phải đỏ — vì đó chính là con
đường dẫn tới một baseline bị thổi phồng ở `W1-13`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from pipeline.eval.golden import GoldenQuery, QueryCategory
from pipeline.goldenset.schema import DraftProvenance, GoldenDraft
from pipeline.goldenset.triage import (
    TriageFlag,
    review_priority,
    triage_drafts,
    write_decisions_template,
    write_review_queue,
    write_triage,
)
from rag_core.retrieval.base import Retriever
from rag_core.schemas import Chunk, Language, RetrievedChunk


def _chunk(chunk_id: str, content: str = "noi dung mac dinh") -> Chunk:
    doc_id = chunk_id.split("::")[0]
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        chunk_index=0,
    )


def _draft(
    query_id: str,
    *,
    category: QueryCategory = QueryCategory.FACTOID,
    chunk_ids: list[str] | None = None,
    quotes_verified: bool = True,
    quote: str = "mot doan trich dan",
    answer: str = "dap an tham chieu",
) -> GoldenDraft:
    ids = [] if category is QueryCategory.UNANSWERABLE else (chunk_ids or ["doc-a::00001"])
    return GoldenDraft(
        query=GoldenQuery(
            query_id=query_id,
            query=f"cau hoi cho {query_id}",
            category=category,
            lang=Language.VI,
            relevant_chunk_ids=ids,
            reference_answer=answer,
        ),
        provenance=DraftProvenance(
            generator_model="fake-model",
            generator_model_requested="fake-model",
            category_requested=category,
            source_chunk_ids=ids,
            supporting_quotes=[quote],
            quotes_verified=quotes_verified,
        ),
    )


class FakeRetriever(Retriever):
    """Retriever giả: `plan` map query → list (chunk_id, score).

    Có `fetch_chunks` để test được cả nhánh phát hiện chunk_id chết; `known` là
    tập chunk_id "tồn tại trong index", tách khỏi `plan` vì một chunk có thể tồn
    tại mà không được truy hồi ra.
    """

    def __init__(
        self,
        plan: dict[str, list[tuple[str, float]]],
        known: set[str] | None = None,
    ) -> None:
        self.plan = plan
        self.known = known
        self.calls: list[tuple[str, int]] = []
        self.fetch_calls: list[list[str]] = []
        self.name = "fake"

    def retrieve(
        self, query: str, top_k: int = 10, *, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        hits = self.plan.get(query, [])[:top_k]
        return [
            RetrievedChunk(chunk=_chunk(cid, f"text cua {cid}"), score=score, rank=i)
            for i, (cid, score) in enumerate(hits, start=1)
        ]

    def fetch_chunks(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        self.fetch_calls.append(list(chunk_ids))
        pool = (
            self.known
            if self.known is not None
            else {cid for hits in self.plan.values() for cid, _ in hits}
        )
        return {cid: _chunk(cid) for cid in chunk_ids if cid in pool}


class RetrieverWithoutFetch(Retriever):
    """Retriever không có `fetch_chunks` — triage phải chạy được, chỉ bỏ 1 phép kiểm."""

    def retrieve(
        self, query: str, top_k: int = 10, *, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        return [RetrievedChunk(chunk=_chunk("doc-a::00001"), score=0.5, rank=1)]


class TestCalibratedThreshold:
    """Ngưỡng nghi ngờ phải đến từ dữ liệu, không phải từ một hằng số."""

    def test_threshold_is_median_of_answerable_top_scores(self) -> None:
        drafts = [
            _draft("f1", chunk_ids=["doc-a::00001"]),
            _draft("f2", chunk_ids=["doc-b::00001"]),
            _draft("f3", chunk_ids=["doc-c::00001"]),
        ]
        plan = {
            "cau hoi cho f1": [("doc-a::00001", 0.20)],
            "cau hoi cho f2": [("doc-b::00001", 0.50)],
            "cau hoi cho f3": [("doc-c::00001", 0.90)],
        }
        _, summary = triage_drafts(drafts, FakeRetriever(plan))
        assert summary.score_threshold == pytest.approx(0.50)

    def test_threshold_follows_the_quantile_argument(self) -> None:
        drafts = [_draft(f"f{i}", chunk_ids=[f"doc-{i}::00001"]) for i in range(5)]
        plan = {f"cau hoi cho f{i}": [(f"doc-{i}::00001", 0.1 * (i + 1))] for i in range(5)}
        _, low = triage_drafts(drafts, FakeRetriever(plan), score_quantile=0.0)
        _, high = triage_drafts(drafts, FakeRetriever(plan), score_quantile=1.0)
        assert low.score_threshold == pytest.approx(0.1)
        assert high.score_threshold == pytest.approx(0.5)

    def test_unanswerable_scores_excluded_from_calibration(self) -> None:
        """Nếu tính cả nhóm unanswerable vào ngưỡng thì ngưỡng tự bị kéo lệch."""
        drafts = [
            _draft("f1", chunk_ids=["doc-a::00001"]),
            _draft("u1", category=QueryCategory.UNANSWERABLE),
            _draft("u2", category=QueryCategory.UNANSWERABLE),
        ]
        plan = {
            "cau hoi cho f1": [("doc-a::00001", 0.40)],
            "cau hoi cho u1": [("doc-z::00001", 0.99)],
            "cau hoi cho u2": [("doc-z::00002", 0.98)],
        }
        _, summary = triage_drafts(drafts, FakeRetriever(plan))
        assert summary.score_threshold == pytest.approx(0.40)

    def test_no_answerable_questions_means_no_threshold(self) -> None:
        drafts = [_draft("u1", category=QueryCategory.UNANSWERABLE)]
        plan = {"cau hoi cho u1": [("doc-z::00001", 0.99)]}
        results, summary = triage_drafts(drafts, FakeRetriever(plan))
        assert summary.score_threshold is None
        # Không có ngưỡng thì không được đoán — tuyệt đối không gắn cờ.
        assert TriageFlag.UNANSWERABLE_BUT_RETRIEVED not in results[0].flags


class TestSignalA:
    """`unanswerable` + retriever tự tin → bằng chứng nhãn sai. Đây là `TD-09`."""

    def test_flags_unanswerable_above_threshold(self) -> None:
        drafts = [
            _draft("f1", chunk_ids=["doc-a::00001"]),
            _draft("u1", category=QueryCategory.UNANSWERABLE),
        ]
        plan = {
            "cau hoi cho f1": [("doc-a::00001", 0.40)],
            "cau hoi cho u1": [("doc-b::00001", 0.85)],
        }
        results, _ = triage_drafts(drafts, FakeRetriever(plan))
        u1 = next(r for r in results if r.query_id == "u1")
        assert TriageFlag.UNANSWERABLE_BUT_RETRIEVED in u1.flags
        assert u1.suggested_decision == "recheck_category"

    def test_does_not_flag_unanswerable_below_threshold(self) -> None:
        drafts = [
            _draft("f1", chunk_ids=["doc-a::00001"]),
            _draft("u1", category=QueryCategory.UNANSWERABLE),
        ]
        plan = {
            "cau hoi cho f1": [("doc-a::00001", 0.60)],
            "cau hoi cho u1": [("doc-b::00001", 0.10)],
        }
        results, _ = triage_drafts(drafts, FakeRetriever(plan))
        u1 = next(r for r in results if r.query_id == "u1")
        assert u1.flags == []
        assert u1.suggested_decision == "accept"

    def test_unanswerable_with_empty_retrieval_is_not_flagged(self) -> None:
        drafts = [
            _draft("f1", chunk_ids=["doc-a::00001"]),
            _draft("u1", category=QueryCategory.UNANSWERABLE),
        ]
        plan = {"cau hoi cho f1": [("doc-a::00001", 0.60)]}
        results, _ = triage_drafts(drafts, FakeRetriever(plan))
        u1 = next(r for r in results if r.query_id == "u1")
        assert u1.top_score is None
        assert u1.flags == []


class TestSignalB:
    """Nhóm test canh **bất đối xứng**. Đỏ ở đây = nguy cơ thổi phồng baseline."""

    def test_missed_gold_is_flagged_but_default_stays_accept(self) -> None:
        drafts = [
            _draft("f1", chunk_ids=["doc-a::00001"]),
            _draft("f2", chunk_ids=["doc-b::00001"]),
        ]
        plan = {
            "cau hoi cho f1": [("doc-a::00001", 0.70)],
            # f2: retriever trả về chunk khác hoàn toàn
            "cau hoi cho f2": [("doc-z::00009", 0.65)],
        }
        results, _ = triage_drafts(drafts, FakeRetriever(plan, known={"doc-b::00001"}))
        f2 = next(r for r in results if r.query_id == "f2")
        assert TriageFlag.ANSWERABLE_BUT_NOT_RETRIEVED in f2.flags
        assert f2.gold_rank is None
        # Điểm cốt tử: đề xuất mặc định KHÔNG được là loại bỏ.
        assert f2.suggested_decision == "accept"

    def test_missed_gold_sorts_to_the_end_of_the_queue(self) -> None:
        """Xếp tín hiệu (B) lên đầu sẽ dụ người review loại đúng câu khó nhất."""
        hard = _draft("hard", chunk_ids=["doc-b::00001"])
        dead = _draft("dead", chunk_ids=["doc-gone::00001"])
        plan = {
            "cau hoi cho hard": [("doc-z::00009", 0.65)],
            "cau hoi cho dead": [("doc-z::00009", 0.65)],
        }
        results, _ = triage_drafts([hard, dead], FakeRetriever(plan, known={"doc-b::00001"}))
        order = [r.query_id for r in sorted(results, key=review_priority)]
        assert order == ["dead", "hard"]

    def test_dead_chunk_pointer_is_the_one_exception(self) -> None:
        """chunk_id không tồn tại trong index thì luôn phải sửa, không phải câu khó."""
        drafts = [_draft("f1", chunk_ids=["doc-gone::00001"])]
        plan = {"cau hoi cho f1": [("doc-z::00009", 0.65)]}
        results, _ = triage_drafts(drafts, FakeRetriever(plan, known=set()))
        assert TriageFlag.GOLD_CHUNK_MISSING in results[0].flags
        assert results[0].missing_chunk_ids == ["doc-gone::00001"]
        assert results[0].suggested_decision == "fix_chunk_ids"

    def test_partially_dead_pointer_lists_only_the_missing_one(self) -> None:
        drafts = [_draft("f1", chunk_ids=["doc-a::00001", "doc-gone::00002"])]
        plan = {"cau hoi cho f1": [("doc-a::00001", 0.70)]}
        results, _ = triage_drafts(drafts, FakeRetriever(plan, known={"doc-a::00001"}))
        assert results[0].missing_chunk_ids == ["doc-gone::00002"]


class TestOtherFlags:
    def test_unverified_quote_is_flagged(self) -> None:
        drafts = [_draft("f1", chunk_ids=["doc-a::00001"], quotes_verified=False)]
        plan = {"cau hoi cho f1": [("doc-a::00001", 0.70)]}
        results, _ = triage_drafts(drafts, FakeRetriever(plan))
        assert TriageFlag.QUOTE_UNVERIFIED in results[0].flags
        assert results[0].suggested_decision == "recheck_quote"

    def test_trivially_easy_needs_both_rank_1_and_high_score(self) -> None:
        drafts = [_draft(f"f{i}", chunk_ids=[f"doc-{i}::00001"]) for i in range(5)]
        plan = {f"cau hoi cho f{i}": [(f"doc-{i}::00001", 0.1 * (i + 1))] for i in range(5)}
        results, summary = triage_drafts(drafts, FakeRetriever(plan))
        easy = [r.query_id for r in results if TriageFlag.TRIVIALLY_EASY in r.flags]
        # Ngưỡng "quá dễ" là phân vị 0.9 của {0.1..0.5} = 0.46 → chỉ f4 (0.5) vượt.
        assert easy == ["f4"]
        assert summary.trivial_threshold == pytest.approx(0.46)

    def test_rank_2_is_never_trivially_easy(self) -> None:
        drafts = [_draft(f"f{i}", chunk_ids=[f"doc-{i}::00001"]) for i in range(3)]
        plan = {
            "cau hoi cho f0": [("doc-0::00001", 0.9)],
            "cau hoi cho f1": [("doc-x::00001", 0.9), ("doc-1::00001", 0.89)],
            "cau hoi cho f2": [("doc-2::00001", 0.1)],
        }
        results, _ = triage_drafts(drafts, FakeRetriever(plan))
        f1 = next(r for r in results if r.query_id == "f1")
        assert f1.gold_rank == 2
        assert TriageFlag.TRIVIALLY_EASY not in f1.flags

    def test_gold_rank_is_the_best_of_several(self) -> None:
        drafts = [_draft("f1", chunk_ids=["doc-a::00003", "doc-a::00001"])]
        plan = {
            "cau hoi cho f1": [
                ("doc-z::00009", 0.9),
                ("doc-a::00001", 0.8),
                ("doc-a::00003", 0.7),
            ]
        }
        results, _ = triage_drafts(drafts, FakeRetriever(plan))
        assert results[0].gold_rank == 2
        assert results[0].gold_score == pytest.approx(0.8)


class TestRobustness:
    def test_works_without_fetch_chunks(self) -> None:
        drafts = [_draft("f1", chunk_ids=["doc-a::00001"])]
        results, _ = triage_drafts(drafts, RetrieverWithoutFetch())
        assert results[0].missing_chunk_ids == []
        assert TriageFlag.GOLD_CHUNK_MISSING not in results[0].flags

    def test_fetch_chunks_can_be_disabled(self) -> None:
        drafts = [_draft("f1", chunk_ids=["doc-gone::00001"])]
        plan = {"cau hoi cho f1": [("doc-z::00009", 0.5)]}
        results, _ = triage_drafts(drafts, FakeRetriever(plan, known=set()), fetch_chunks=False)
        assert TriageFlag.GOLD_CHUNK_MISSING not in results[0].flags

    def test_results_keep_input_order(self) -> None:
        ids = ["z9", "a1", "m5"]
        drafts = [_draft(i, chunk_ids=[f"doc-{i}::00001"]) for i in ids]
        plan = {f"cau hoi cho {i}": [(f"doc-{i}::00001", 0.5)] for i in ids}
        results, _ = triage_drafts(drafts, FakeRetriever(plan))
        assert [r.query_id for r in results] == ids

    def test_top_k_is_passed_through(self) -> None:
        drafts = [_draft("f1", chunk_ids=["doc-a::00001"])]
        fake = FakeRetriever({"cau hoi cho f1": [("doc-a::00001", 0.5)]})
        triage_drafts(drafts, fake, top_k=7)
        assert fake.calls == [("cau hoi cho f1", 7)]

    def test_fetch_chunks_called_once_for_all_drafts(self) -> None:
        """266 câu × 1 round-trip Qdrant mỗi câu là thứ phải tránh."""
        drafts = [_draft(f"f{i}", chunk_ids=["doc-a::00001"]) for i in range(20)]
        plan = {f"cau hoi cho f{i}": [("doc-a::00001", 0.5)] for i in range(20)}
        fake = FakeRetriever(plan)
        triage_drafts(drafts, fake)
        assert fake.fetch_calls == [["doc-a::00001"]], (
            "phải gọi đúng 1 lần, với danh sách id đã khử trùng"
        )

    def test_empty_drafts(self) -> None:
        results, summary = triage_drafts([], FakeRetriever({}))
        assert results == []
        assert summary.total == 0
        assert summary.score_threshold is None

    def test_priority_is_deterministic_for_equal_risk(self) -> None:
        drafts = [_draft(i, chunk_ids=[f"doc-{i}::00001"]) for i in ["b", "a", "c"]]
        plan = {f"cau hoi cho {i}": [(f"doc-{i}::00001", 0.5)] for i in ["b", "a", "c"]}
        results, _ = triage_drafts(drafts, FakeRetriever(plan))
        order = [r.query_id for r in sorted(results, key=review_priority)]
        assert order == ["a", "b", "c"]


class TestSummary:
    def test_counts_are_split_by_answerability(self) -> None:
        drafts = [
            _draft("f1", chunk_ids=["doc-a::00001"]),
            _draft("f2", chunk_ids=["doc-b::00001"]),
            _draft("u1", category=QueryCategory.UNANSWERABLE),
        ]
        plan = {
            "cau hoi cho f1": [("doc-a::00001", 0.5)],
            "cau hoi cho f2": [("doc-z::00009", 0.5)],
            "cau hoi cho u1": [("doc-z::00009", 0.5)],
        }
        _, summary = triage_drafts(drafts, FakeRetriever(plan, known={"doc-b::00001"}))
        assert summary.total == 3
        assert summary.answerable == 2
        assert summary.unanswerable == 1
        assert summary.gold_found_in_top_k == 1
        assert summary.by_category["factoid"] == 2

    def test_summary_json_round_trips(self) -> None:
        drafts = [_draft("f1", chunk_ids=["doc-a::00001"])]
        plan = {"cau hoi cho f1": [("doc-a::00001", 0.5)]}
        _, summary = triage_drafts(drafts, FakeRetriever(plan))
        parsed = json.loads(summary.to_json())
        assert parsed["total"] == 1
        assert parsed["top_k"] == 20


class TestOutputs:
    def test_triage_jsonl_is_one_line_per_query(self, tmp_path: Path) -> None:
        drafts = [_draft(f"f{i}", chunk_ids=[f"doc-{i}::00001"]) for i in range(3)]
        plan = {f"cau hoi cho f{i}": [(f"doc-{i}::00001", 0.5)] for i in range(3)}
        results, _ = triage_drafts(drafts, FakeRetriever(plan))
        out = tmp_path / "triage.jsonl"
        assert write_triage(out, results) == 3
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        assert json.loads(lines[0])["query_id"] == "f0"

    def test_review_queue_carries_the_asymmetry_warning(self, tmp_path: Path) -> None:
        """Cảnh báo trong queue là phần chống lỗi phương pháp — không được rơi mất."""
        drafts = [_draft("f1", chunk_ids=["doc-a::00001"])]
        plan = {"cau hoi cho f1": [("doc-a::00001", 0.5)]}
        results, summary = triage_drafts(drafts, FakeRetriever(plan))
        out = tmp_path / "queue.md"
        write_review_queue(out, results, summary, drafts=drafts)
        text = out.read_text(encoding="utf-8")
        assert "không phải lý do để loại câu hỏi" in text
        assert "thiên vị" in text

    def test_review_queue_shows_gold_text_and_quote(self, tmp_path: Path) -> None:
        drafts = [_draft("f1", chunk_ids=["doc-a::00001"], quote="trich dan cua toi")]
        plan = {"cau hoi cho f1": [("doc-a::00001", 0.5)]}
        results, summary = triage_drafts(drafts, FakeRetriever(plan))
        out = tmp_path / "queue.md"
        write_review_queue(
            out,
            results,
            summary,
            gold_texts={"doc-a::00001": "toan van chunk da gan"},
            drafts=drafts,
        )
        text = out.read_text(encoding="utf-8")
        assert "toan van chunk da gan" in text
        assert "trich dan cua toi" in text
        assert "dap an tham chieu" in text

    def test_review_queue_marks_the_gold_chunk_in_top_k(self, tmp_path: Path) -> None:
        drafts = [_draft("f1", chunk_ids=["doc-a::00001"])]
        plan = {
            "cau hoi cho f1": [("doc-z::00009", 0.9), ("doc-a::00001", 0.8)],
        }
        results, summary = triage_drafts(drafts, FakeRetriever(plan))
        out = tmp_path / "queue.md"
        write_review_queue(out, results, summary, drafts=drafts)
        assert "⬅️ đã gán" in out.read_text(encoding="utf-8")

    def test_decisions_csv_has_blank_decision_column(self, tmp_path: Path) -> None:
        drafts = [_draft("f1", chunk_ids=["doc-a::00001"])]
        plan = {"cau hoi cho f1": [("doc-a::00001", 0.5)]}
        results, _ = triage_drafts(drafts, FakeRetriever(plan))
        out = tmp_path / "decisions.csv"
        assert write_decisions_template(out, results) == 1
        with out.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["query_id"] == "f1"
        assert rows[0]["decision"] == ""
        assert rows[0]["suggested_decision"] == "accept"

    def test_decisions_csv_refuses_to_overwrite(self, tmp_path: Path) -> None:
        """Mất 6 giờ công review vì chạy lại một lệnh là chuyện không được xảy ra."""
        drafts = [_draft("f1", chunk_ids=["doc-a::00001"])]
        plan = {"cau hoi cho f1": [("doc-a::00001", 0.5)]}
        results, _ = triage_drafts(drafts, FakeRetriever(plan))
        out = tmp_path / "decisions.csv"
        write_decisions_template(out, results)
        with pytest.raises(FileExistsError, match="đã tồn tại"):
            write_decisions_template(out, results)

    def test_decisions_csv_is_in_review_order(self, tmp_path: Path) -> None:
        hard = _draft("hard", chunk_ids=["doc-b::00001"])
        dead = _draft("dead", chunk_ids=["doc-gone::00001"])
        plan = {
            "cau hoi cho hard": [("doc-z::00009", 0.65)],
            "cau hoi cho dead": [("doc-z::00009", 0.65)],
        }
        results, _ = triage_drafts([hard, dead], FakeRetriever(plan, known={"doc-b::00001"}))
        out = tmp_path / "decisions.csv"
        write_decisions_template(out, results)
        with out.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert [r["query_id"] for r in rows] == ["dead", "hard"]
