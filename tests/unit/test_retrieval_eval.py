"""W1-12 — lớp tổng hợp: breakdown theo nhóm/ngôn ngữ và xuất báo cáo."""

from __future__ import annotations

import json

import pytest

from pipeline.eval.golden import GoldenQuery, QueryCategory, load_golden_set, write_golden_set
from pipeline.eval.retrieval_eval import evaluate_run, run_retrieval_eval
from rag_core.schemas import Chunk, Language, RetrievedChunk


def _query(qid: str, category: QueryCategory, lang: Language, relevant: list[str]) -> GoldenQuery:
    return GoldenQuery(
        query_id=qid,
        query=f"Câu hỏi {qid}",
        category=category,
        lang=lang,
        relevant_chunk_ids=relevant,
    )


QUERIES = [
    _query("q1", QueryCategory.FACTOID, Language.VI, ["c1"]),
    _query("q2", QueryCategory.FACTOID, Language.EN, ["c2"]),
    _query("q3", QueryCategory.MULTI_HOP, Language.VI, ["c3", "c4"]),
    _query("q4", QueryCategory.UNANSWERABLE, Language.VI, []),
]


class TestEvaluateRun:
    def test_unanswerable_excluded_from_scoring(self) -> None:
        report = evaluate_run(QUERIES, {"q1": ["c1"], "q2": ["c2"], "q3": ["c3", "c4"]})
        assert report.n_queries == 4
        assert report.n_scored == 3
        assert report.n_skipped_unanswerable == 1

    def test_perfect_run_scores_one(self) -> None:
        report = evaluate_run(QUERIES, {"q1": ["c1"], "q2": ["c2"], "q3": ["c3", "c4"]})
        assert report.overall["recall@10"] == pytest.approx(1.0)
        assert report.overall["mrr"] == pytest.approx(1.0)

    def test_missing_query_counts_as_empty_retrieval(self) -> None:
        """Thiếu query_id trong kết quả là truy hồi rỗng, không phải bỏ qua —
        im lặng bỏ qua sẽ làm điểm cao lên một cách sai."""
        report = evaluate_run(QUERIES, {"q1": ["c1"]})
        assert report.n_scored == 3
        assert report.overall["recall@10"] == pytest.approx(1 / 3)

    def test_breakdown_by_category(self) -> None:
        report = evaluate_run(QUERIES, {"q1": ["c1"], "q2": ["c2"], "q3": ["zz"]})
        assert report.by_category["factoid"].n_queries == 2
        assert report.by_category["factoid"].metrics["recall@10"] == pytest.approx(1.0)
        assert report.by_category["multi_hop"].metrics["recall@10"] == pytest.approx(0.0)
        assert "unanswerable" not in report.by_category

    def test_breakdown_by_language(self) -> None:
        report = evaluate_run(QUERIES, {"q1": ["c1"], "q2": ["zz"], "q3": ["c3", "c4"]})
        assert report.by_language["vi"].n_queries == 2
        assert report.by_language["en"].metrics["recall@10"] == pytest.approx(0.0)

    def test_latency_percentiles(self) -> None:
        report = evaluate_run(
            QUERIES,
            {"q1": ["c1"], "q2": ["c2"], "q3": ["c3"]},
            latencies_ms={"q1": 10.0, "q2": 20.0, "q3": 30.0},
        )
        assert report.latency_ms["p50"] == pytest.approx(20.0)
        assert report.latency_ms["max"] == pytest.approx(30.0)


class TestReportRendering:
    def test_markdown_has_tables(self) -> None:
        markdown = evaluate_run(QUERIES, {"q1": ["c1"]}, run_name="baseline").to_markdown()
        assert "# Retrieval eval — `baseline`" in markdown
        assert "## Tổng thể" in markdown
        assert "## Theo nhóm truy vấn" in markdown
        assert "unanswerable" in markdown  # phải giải thích vì sao bị loại

    def test_json_is_parseable_and_complete(self) -> None:
        payload = json.loads(evaluate_run(QUERIES, {"q1": ["c1"]}, run_name="baseline").to_json())
        assert payload["run_name"] == "baseline"
        assert payload["n_scored"] == 3
        assert "recall@10" in payload["overall"]
        # Báo cáo phải tự mô tả môi trường chạy, nếu không thì 2 tuần sau không
        # biết con số sinh ra ở đâu.
        assert payload["environment"]["python"]


class TestRunWithRetriever:
    def test_calls_retriever_and_scores(self) -> None:
        class StubRetriever:
            name = "stub"

            def retrieve(self, query: str, top_k: int = 10, *, filters=None):  # type: ignore[no-untyped-def]
                mapping = {"Câu hỏi q1": ["c1"], "Câu hỏi q2": ["c2"], "Câu hỏi q3": ["c3", "c4"]}
                return [
                    RetrievedChunk(
                        chunk=Chunk(chunk_id=cid, doc_id="d", content="x", chunk_index=0),
                        score=1.0 / rank,
                        rank=rank,
                    )
                    for rank, cid in enumerate(mapping.get(query, []), start=1)
                ]

        report = run_retrieval_eval(StubRetriever(), QUERIES, run_name="stub")  # type: ignore[arg-type]
        assert report.overall["recall@10"] == pytest.approx(1.0)
        assert report.config["retriever"] == "stub"
        assert report.latency_ms["max"] >= 0.0

    def test_warms_up_before_timing(self) -> None:
        """Truy vấn đầu tiên không được tính giờ.

        Model embedding nạp lazy, nên lần gọi đầu gánh cả việc nạp trọng số —
        đo thật trên máy này là 15 giây so với p50 31 ms. Không bỏ nó đi thì p95
        (ngưỡng của gate hiệu năng W5/W6) là con số của việc khởi động, không
        phải của việc truy hồi.
        """
        calls: list[str] = []

        class CountingRetriever:
            name = "counting"

            def retrieve(self, query: str, top_k: int = 10, *, filters=None):  # type: ignore[no-untyped-def]
                calls.append(query)
                return []

        run_retrieval_eval(CountingRetriever(), QUERIES, run_name="w")  # type: ignore[arg-type]
        assert len(calls) == len(QUERIES) + 1
        assert calls[0] == calls[1], "lượt warm-up phải là chính truy vấn đầu tiên"

    def test_warmup_can_be_turned_off(self) -> None:
        calls: list[str] = []

        class CountingRetriever:
            name = "counting"

            def retrieve(self, query: str, top_k: int = 10, *, filters=None):  # type: ignore[no-untyped-def]
                calls.append(query)
                return []

        run_retrieval_eval(CountingRetriever(), QUERIES, run_name="w", warmup=False)  # type: ignore[arg-type]
        assert len(calls) == len(QUERIES)


class TestGoldenSet:
    def test_round_trip(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "golden.jsonl"
        write_golden_set(path, QUERIES)
        assert load_golden_set(path) == QUERIES

    def test_unanswerable_must_have_no_relevant_ids(self) -> None:
        with pytest.raises(ValueError, match="phân loại sai"):
            GoldenQuery(
                query_id="q",
                query="x",
                category=QueryCategory.UNANSWERABLE,
                relevant_chunk_ids=["c1"],
            )

    def test_answerable_must_have_relevant_ids(self) -> None:
        with pytest.raises(ValueError, match="ít nhất một"):
            GoldenQuery(query_id="q", query="x", category=QueryCategory.FACTOID)

    def test_duplicate_query_id_rejected(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "golden.jsonl"
        write_golden_set(path, [QUERIES[0], QUERIES[0]])
        with pytest.raises(ValueError, match="query_id trùng"):
            load_golden_set(path)

    def test_error_reports_line_number(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "golden.jsonl"
        path.write_text('{"query_id": "q1"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match=r":1 —"):
            load_golden_set(path)


class TestLabelDigest:
    """`W2-03` — băm tập nhãn, để `compare.py` từ chối so hai lần chạy dùng nhãn
    khác nhau. Xem `QueryScore.relevant_digest`."""

    def test_written_for_every_scored_query(self) -> None:
        report = evaluate_run(QUERIES, {"q1": ["c1"], "q2": ["c2"], "q3": ["c3", "c4"]})
        assert all(row.relevant_digest for row in report.per_query)

    def test_same_labels_same_digest(self) -> None:
        a = evaluate_run([_query("q", QueryCategory.FACTOID, Language.VI, ["c1", "c2"])], {})
        b = evaluate_run([_query("q", QueryCategory.FACTOID, Language.VI, ["c1", "c2"])], {})
        assert a.per_query[0].relevant_digest == b.per_query[0].relevant_digest

    def test_order_does_not_matter(self) -> None:
        """Tập nhãn, không phải danh sách nhãn: ánh xạ span sinh nhãn theo thứ tự
        chunk trong tài liệu, và thứ tự đó không mang thông tin về độ liên quan."""
        a = evaluate_run([_query("q", QueryCategory.FACTOID, Language.VI, ["c1", "c2"])], {})
        b = evaluate_run([_query("q", QueryCategory.FACTOID, Language.VI, ["c2", "c1"])], {})
        assert a.per_query[0].relevant_digest == b.per_query[0].relevant_digest

    def test_different_labels_different_digest(self) -> None:
        a = evaluate_run([_query("q", QueryCategory.FACTOID, Language.VI, ["c1"])], {})
        b = evaluate_run([_query("q", QueryCategory.FACTOID, Language.VI, ["c9"])], {})
        assert a.per_query[0].relevant_digest != b.per_query[0].relevant_digest

    def test_same_count_different_labels_is_caught(self) -> None:
        """Đúng ca mà `n_relevant` một mình không thấy."""
        a = evaluate_run([_query("q", QueryCategory.FACTOID, Language.VI, ["c1", "c2"])], {})
        b = evaluate_run([_query("q", QueryCategory.FACTOID, Language.VI, ["c1", "c3"])], {})
        assert a.per_query[0].n_relevant == b.per_query[0].n_relevant
        assert a.per_query[0].relevant_digest != b.per_query[0].relevant_digest

    def test_digest_is_in_the_jsonl(self) -> None:
        report = evaluate_run(QUERIES, {"q1": ["c1"]})
        row = json.loads(report.to_jsonl().splitlines()[0])
        assert len(row["relevant_digest"]) == 16
