"""Smoke eval của CI, phần không cần Qdrant — `W5-09`.

Bài quan trọng nhất là `TestTheEmbedderRefusesToGuess`: nó khoá lý do
`FrozenEmbedder` tồn tại. Một embedder giả **trả về được** cho mọi đầu vào là
một cổng vẫn ra số sau khi đã thôi đo thứ nó nói là đo.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pipeline.eval.smoke import (
    FIXTURE_VERSION,
    SMOKE_PREFIX,
    FrozenEmbedder,
    SmokeFixture,
    SmokeQuery,
    SmokeResult,
    UnknownTextError,
    compare_to_baseline,
    load_fixture,
    run_smoke,
)

FIXTURE = Path("data/eval/smoke/smoke_v1.jsonl.gz")
BASELINE = Path("data/eval/smoke/baseline.json")


def _vector(seed: int, dim: int = 4) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {
        "dense": rng.random(dim, dtype=np.float32),
        "sparse": {"indices": [seed, seed + 1], "values": [1.0, 0.5]},
    }


# ---------------------------------------------------------------------------
# 1. ⭐⭐ Embedder không được đoán
# ---------------------------------------------------------------------------


class TestTheEmbedderRefusesToGuess:
    def test_an_unknown_string_raises(self) -> None:
        embedder = FrozenEmbedder({"đã biết": _vector(1)}, dimension=4, model_name="m")
        with pytest.raises(UnknownTextError, match="không có vector"):
            embedder.embed_query_hybrid("chưa biết")

    def test_it_does_not_fall_back_to_zeros(self) -> None:
        """Một `zeros(dim)` chạy trót lọt qua Qdrant và cho ra một con số. Con
        số ấy đo không gì cả, và nó sẽ nằm cạnh những con số thật."""
        embedder = FrozenEmbedder({}, dimension=4, model_name="m")
        with pytest.raises(UnknownTextError):
            embedder.embed_documents(["gì đó"])

    def test_a_known_string_comes_back_byte_for_byte(self) -> None:
        stored = _vector(7)
        embedder = FrozenEmbedder({"q": stored}, dimension=4, model_name="m")
        dense, sparse = embedder.embed_query_hybrid("q")
        assert np.allclose(dense, stored["dense"])
        assert sparse.indices == (7, 8)

    def test_it_records_what_it_was_asked_for(self) -> None:
        """`seen` là cách chứng minh smoke đã hỏi đúng những chuỗi đã đóng
        băng — chứ không phải một tập con vì ai đó thêm bộ lọc ở giữa."""
        embedder = FrozenEmbedder({"a": _vector(1), "b": _vector(2)}, dimension=4, model_name="m")
        embedder.embed_query_hybrid("a")
        embedder.embed_query_hybrid("b")
        assert embedder.seen == ["a", "b"]

    def test_it_declares_sparse_so_the_hybrid_branch_is_reachable(self) -> None:
        """`QdrantHybridRetriever` **chết lúc dựng** nếu provider không khai
        sparse — nên thiếu thuộc tính này thì smoke lặng lẽ không còn là smoke
        của nhánh hybrid nữa, nó là lỗi khởi tạo."""
        embedder = FrozenEmbedder({}, dimension=4, model_name="m")
        assert embedder.sparse_vocab_size is not None
        assert embedder.dimension == 4


# ---------------------------------------------------------------------------
# 2. ⭐ Tên metric không được trùng tên metric thật
# ---------------------------------------------------------------------------


class TestTheMetricNamesAreFenced:
    def test_every_metric_carries_the_prefix(self) -> None:
        """300 chunk vs 15.814 chunk: cùng công thức, khác tập. `W5-05` đã trả
        giá một lần cho hai con số cùng tên (`p95_latency_ms`)."""
        result = _fake_run()
        assert result.metrics
        assert all(name.startswith(SMOKE_PREFIX) for name in result.metrics)

    def test_the_prefix_is_not_empty(self) -> None:
        assert SMOKE_PREFIX and SMOKE_PREFIX.endswith("_")


class _StubRetriever:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    def retrieve(self, query: str, top_k: int = 10) -> list[Any]:
        class _Hit:
            def __init__(self, chunk_id: str) -> None:
                self.chunk = type("C", (), {"chunk_id": chunk_id})()

        return [_Hit(cid) for cid in self.order[:top_k]]


def _fixture(queries: list[SmokeQuery]) -> SmokeFixture:
    return SmokeFixture(
        version=FIXTURE_VERSION,
        built_at="2026-09-05T00:00:00+00:00",
        source_collection="c",
        embedding_model="m",
        dimension=4,
        queries=queries,
        chunks=[],
        query_vectors={},
        chunk_vectors={},
    )


def _fake_run(order: list[str] | None = None) -> SmokeResult:
    fixture = _fixture([SmokeQuery("q1", "câu hỏi", "factoid", "vi", ("c1",))])
    return run_smoke(_StubRetriever(order or ["c1", "c2"]), fixture, top_k=10)


class TestScoring:
    def test_a_perfect_hit_scores_one(self) -> None:
        result = _fake_run(["c1", "c2", "c3"])
        assert result.metrics[f"{SMOKE_PREFIX}mrr"] == 1.0
        assert result.metrics[f"{SMOKE_PREFIX}recall@10"] == 1.0

    def test_rank_matters_to_mrr_but_not_to_recall(self) -> None:
        """Chính chỗ này làm cổng nhạy với một thay đổi **thứ hạng** — thứ mà
        `recall@10` một mình không thấy."""
        result = _fake_run(["c9", "c8", "c1"])
        assert result.metrics[f"{SMOKE_PREFIX}recall@10"] == 1.0
        assert result.metrics[f"{SMOKE_PREFIX}mrr"] == pytest.approx(1 / 3)

    def test_a_query_without_labels_is_refused_not_skipped(self) -> None:
        """Ba metric đều trả `None` khi không có nhãn; một `None` lọt vào phép
        trung bình sẽ nổ giữa CI thay vì ở đây."""
        fixture = _fixture([SmokeQuery("q1", "câu", "unanswerable", "vi", ())])
        with pytest.raises(ValueError, match="không có nhãn"):
            run_smoke(_StubRetriever(["c1"]), fixture, top_k=10)


# ---------------------------------------------------------------------------
# 3. Cổng: khi nào đỏ
# ---------------------------------------------------------------------------


class TestTheGate:
    def _baseline(self, **metrics: float) -> dict[str, Any]:
        return {"n_queries": 1, "metrics": metrics or {f"{SMOKE_PREFIX}mrr": 0.90}}

    def test_a_drop_beyond_tolerance_is_a_failure(self) -> None:
        result = SmokeResult(metrics={f"{SMOKE_PREFIX}mrr": 0.80}, n_queries=1)
        failures = compare_to_baseline(result, self._baseline(), tolerance=0.02)
        assert len(failures) == 1
        assert "0.8000 < 0.9000" in failures[0]

    def test_a_drop_inside_tolerance_passes(self) -> None:
        result = SmokeResult(metrics={f"{SMOKE_PREFIX}mrr": 0.89}, n_queries=1)
        assert compare_to_baseline(result, self._baseline(), tolerance=0.02) == []

    def test_an_improvement_never_fails(self) -> None:
        result = SmokeResult(metrics={f"{SMOKE_PREFIX}mrr": 0.99}, n_queries=1)
        assert compare_to_baseline(result, self._baseline(), tolerance=0.02) == []

    def test_a_missing_metric_is_a_failure_not_a_pass(self) -> None:
        """⚠️ Hướng hỏng: baseline có dòng mà kết quả không có nghĩa là phép đo
        ấy **thôi chạy**. Bỏ qua nó là cách một cổng tự tháo từng chốt một."""
        result = SmokeResult(metrics={}, n_queries=1)
        failures = compare_to_baseline(result, self._baseline(), tolerance=0.02)
        assert failures and "thiếu metric" in failures[0]

    def test_a_different_query_count_is_a_failure(self) -> None:
        """Fixture đổi mà baseline không đổi thì mọi con số so với nhau đều vô
        nghĩa — và chúng vẫn so được, đó mới là vấn đề."""
        result = SmokeResult(metrics={f"{SMOKE_PREFIX}mrr": 0.95}, n_queries=29)
        failures = compare_to_baseline(result, self._baseline(), tolerance=0.02)
        assert any("số câu đổi" in f for f in failures)

    def test_a_brand_new_metric_warns_instead_of_failing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        result = SmokeResult(
            metrics={f"{SMOKE_PREFIX}mrr": 0.95, f"{SMOKE_PREFIX}new": 0.1}, n_queries=1
        )
        with caplog.at_level("WARNING"):
            assert compare_to_baseline(result, self._baseline(), tolerance=0.02) == []
        assert "chưa có trong baseline" in caplog.text


# ---------------------------------------------------------------------------
# 4. Fixture đã commit
# ---------------------------------------------------------------------------


class TestTheCommittedFixture:
    def test_it_loads(self) -> None:
        fixture = load_fixture(FIXTURE)
        assert fixture.version == FIXTURE_VERSION
        assert len(fixture.queries) == 30
        assert len(fixture.chunks) == len(fixture.chunk_vectors)

    def test_every_query_has_a_vector_and_a_label(self) -> None:
        fixture = load_fixture(FIXTURE)
        for query in fixture.queries:
            assert query.query in fixture.query_vectors, query.query_id
            assert query.relevant_chunk_ids, query.query_id

    def test_every_label_points_at_a_chunk_in_the_fixture(self) -> None:
        """⚠️ Nhãn trỏ ra ngoài tập chunk = một câu không thể đạt recall 1,0, và
        cổng sẽ có một trần thấp hơn 1 mà không ai biết vì sao."""
        fixture = load_fixture(FIXTURE)
        have = {chunk.chunk_id for chunk in fixture.chunks}
        missing = {cid for q in fixture.queries for cid in q.relevant_chunk_ids if cid not in have}
        assert not missing, f"nhãn trỏ ra ngoài fixture: {sorted(missing)[:5]}"

    def test_there_are_enough_distractors_for_the_gate_to_mean_something(self) -> None:
        """Không có nhiễu thì `recall@10` luôn bằng 1 — một cổng luôn xanh."""
        fixture = load_fixture(FIXTURE)
        relevant = {cid for q in fixture.queries for cid in q.relevant_chunk_ids}
        assert len(fixture.chunks) >= 5 * len(relevant)

    def test_the_baseline_matches_the_fixture(self) -> None:
        fixture = load_fixture(FIXTURE)
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        assert baseline["n_queries"] == len(fixture.queries)
        assert baseline["built_at"] == fixture.built_at
        assert all(name.startswith(SMOKE_PREFIX) for name in baseline["metrics"])

    def test_the_gate_is_not_already_saturated(self) -> None:
        """Một baseline toàn 1,0 là một cổng chỉ bắt được thảm hoạ. Ở đây
        `mrr ≈ 0,83` nên còn chỗ để một lần tụt vừa phải lộ ra."""
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        assert all(0.3 < value < 0.99 for value in baseline["metrics"].values())


class TestTheFixtureFormat:
    def test_a_future_version_is_refused(self) -> None:
        with pytest.raises(ValueError, match="phiên bản"):
            SmokeFixture(
                version=FIXTURE_VERSION + 1,
                built_at="",
                source_collection="",
                embedding_model="",
                dimension=1,
                queries=[],
                chunks=[],
                query_vectors={},
                chunk_vectors={},
            )

    def test_a_missing_npz_says_which_file(self, tmp_path: Path) -> None:
        meta = tmp_path / "x.jsonl.gz"
        with gzip.open(meta, "wt", encoding="utf-8") as handle:
            handle.write("{}\n")
        with pytest.raises(FileNotFoundError, match=r"x\.npz"):
            load_fixture(meta)
