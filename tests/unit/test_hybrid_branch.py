"""Test cho `QdrantHybridRetriever` và `build_branch(..., "hybrid")` — `W2-04`.

`fuse()` được tách khỏi `retrieve()` chính để test được ở đây: nó nhận hai danh
sách `RetrievedChunk` và trả về danh sách đã hợp nhất, không chạm mạng. Phần chạm
Qdrant thật ở `tests/integration/test_hybrid_retriever.py`.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_core.embedding import HashingEmbeddingProvider
from rag_core.retrieval import (
    DEFAULT_CANDIDATE_K,
    RRF_K,
    QdrantDenseRetriever,
    QdrantHybridRetriever,
    Retriever,
    build_branch,
)
from rag_core.schemas import (
    Chunk,
    DocType,
    DocumentMetadata,
    Language,
    RetrievalMode,
    RetrievedChunk,
)


def _store(*, sparse: bool = True) -> QdrantDenseRetriever:
    """Store không kết nối — `client` là lazy nên dựng thôi thì không chạm mạng."""
    return QdrantDenseRetriever(
        HashingEmbeddingProvider(dimension=64, sparse=sparse),
        collection="rag_test_hybrid",
    )


def _hit(key: str, score: float, rank: int, mode: RetrievalMode) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=key,
        doc_id=key.split("::")[0],
        content=f"nội dung của {key}",
        chunk_index=0,
        metadata=DocumentMetadata(
            source_url="https://example.org/x",
            license="CC BY 4.0",
            lang=Language.VI,
            doc_type=DocType.DEV_REPORT,
        ),
    )
    return RetrievedChunk(
        chunk=chunk,
        score=score,
        rank=rank,
        mode=mode,
        dense_score=score if mode is RetrievalMode.DENSE else None,
        sparse_score=score if mode is RetrievalMode.SPARSE else None,
    )


def _dense(*keys: str) -> list[RetrievedChunk]:
    """Điểm cosine giảm dần, thang [0, 1]."""
    return [_hit(key, 0.9 - 0.05 * i, i + 1, RetrievalMode.DENSE) for i, key in enumerate(keys)]


def _sparse(*keys: str) -> list[RetrievedChunk]:
    """Điểm dot product — thang khác hẳn dense, cố ý."""
    return [_hit(key, 38.0 - 2.0 * i, i + 1, RetrievalMode.SPARSE) for i, key in enumerate(keys)]


class TestBuildBranch:
    def test_hybrid_is_now_supported(self) -> None:
        branch = build_branch(_store(), "hybrid")
        assert isinstance(branch, QdrantHybridRetriever)
        assert isinstance(branch, Retriever)

    def test_reranked_still_points_at_w2_05(self) -> None:
        with pytest.raises(NotImplementedError, match="W2-05"):
            build_branch(_store(), "reranked")

    def test_options_reach_the_retriever(self) -> None:
        branch = build_branch(_store(), "hybrid", k=10, candidate_k=200, weights=(2.0, 1.0))
        assert isinstance(branch, QdrantHybridRetriever)
        assert (branch.k, branch.candidate_k, branch.weights) == (10, 200, (2.0, 1.0))

    def test_none_options_fall_back_to_defaults(self) -> None:
        """CLI truyền `None` cho cờ không đặt — không được ghi đè mặc định."""
        branch = build_branch(_store(), "hybrid", k=None, candidate_k=None, weights=None)
        assert isinstance(branch, QdrantHybridRetriever)
        assert branch.k == RRF_K
        assert branch.candidate_k is None

    def test_rrf_options_on_a_non_hybrid_branch_are_an_ERROR(self) -> None:
        """Không được im lặng bỏ qua. `--rrf-k 10 --retrieval-mode dense` mà vẫn
        chạy thì nó vào bảng ablation `W2-08` như một dòng hợp lệ, trong khi nó
        không đo cái nó nói là đang đo."""
        with pytest.raises(ValueError, match="không nhận tham số"):
            build_branch(_store(), "dense", k=10)
        with pytest.raises(ValueError, match="không nhận tham số"):
            build_branch(_store(), "sparse", candidate_k=100)

    def test_none_options_on_a_non_hybrid_branch_are_fine(self) -> None:
        """Người dùng không đặt cờ thì không phải lỗi — đó là đường chạy mặc định
        của `make eval-retrieval MODE=dense`."""
        store = _store()
        assert build_branch(store, "dense", k=None, candidate_k=None, weights=None) is store


class TestName:
    def test_name_carries_the_parameters(self) -> None:
        """Hai lần chạy khác tham số phải khác tên, nếu không thì bảng `W2-08` có
        hai dòng trùng nhãn và không ai biết dòng nào là dòng nào."""
        assert build_branch(_store(), "hybrid").name == "qdrant-hybrid:rag_test_hybrid:rrf60-c50"
        assert (
            build_branch(_store(), "hybrid", k=10, candidate_k=100).name
            == "qdrant-hybrid:rag_test_hybrid:rrf10-c100"
        )

    def test_weights_appear_in_the_name(self) -> None:
        assert build_branch(_store(), "hybrid", weights=(2.0, 1.0)).name.endswith("-w2:1")

    def test_names_of_the_three_branches_differ(self) -> None:
        store = _store()
        names = {build_branch(store, m).name for m in ("dense", "sparse", "hybrid")}
        assert len(names) == 3


class TestFailsEarly:
    def test_dense_only_provider_is_refused_at_construction(self) -> None:
        """Chết lúc dựng, không lúc truy vấn đầu — `_eval_against_index` quét
        toàn bộ chunk của corpus **trước** khi truy vấn câu đầu tiên."""
        with pytest.raises(ValueError, match="không có gì để hợp nhất"):
            build_branch(_store(sparse=False), "hybrid")

    def test_candidate_k_zero_is_refused(self) -> None:
        with pytest.raises(ValueError, match="candidate_k phải ≥ 1"):
            build_branch(_store(), "hybrid", candidate_k=0)


class TestDepth:
    @staticmethod
    def _branch(candidate_k: int | None = None) -> QdrantHybridRetriever:
        return QdrantHybridRetriever(_store(), candidate_k=candidate_k)

    def test_default_depth(self) -> None:
        assert self._branch()._depth(10) == DEFAULT_CANDIDATE_K

    def test_never_shallower_than_top_k(self) -> None:
        """Lấy ít hơn số kết quả phải trả về thì danh sách hợp nhất ngắn hơn
        `top_k` một cách vô cớ."""
        assert self._branch(candidate_k=5)._depth(100) == 100

    def test_explicit_depth_wins_when_deeper(self) -> None:
        assert self._branch(candidate_k=200)._depth(10) == 200


class TestFuse:
    def _branch(self, **kw: Any) -> QdrantHybridRetriever:
        return QdrantHybridRetriever(_store(), **kw)

    def test_mode_is_hybrid(self) -> None:
        fused = self._branch().fuse(_dense("a", "b"), _sparse("b", "c"), top_k=10)
        assert all(r.mode is RetrievalMode.HYBRID for r in fused)

    def test_union_of_both_branches(self) -> None:
        fused = self._branch().fuse(_dense("a", "b"), _sparse("b", "c"), top_k=10)
        assert {r.chunk.chunk_id for r in fused} == {"a", "b", "c"}

    def test_agreement_wins(self) -> None:
        """`b` hạng 2 ở cả hai nhánh thắng `a` hạng 1 của một nhánh — chính tính
        chất mà `W2-04` mua."""
        fused = self._branch().fuse(_dense("a", "b"), _sparse("c", "b"), top_k=10)
        assert fused[0].chunk.chunk_id == "b"

    def test_score_is_the_rrf_score_not_a_branch_score(self) -> None:
        fused = self._branch().fuse(_dense("a"), _sparse("a"), top_k=10)
        assert fused[0].score == pytest.approx(2 / 61)

    def test_branch_scores_are_preserved(self) -> None:
        """`RetrievedChunk` giữ `dense_score`/`sparse_score` riêng chính để RRF
        không làm mất thông tin nhánh nào đã kéo chunk lên."""
        fused = self._branch().fuse(_dense("a", "b"), _sparse("b"), top_k=10)
        by_key = {r.chunk.chunk_id: r for r in fused}
        assert by_key["b"].dense_score == pytest.approx(0.85)
        assert by_key["b"].sparse_score == pytest.approx(38.0)

    def test_missing_branch_score_is_none_not_zero(self) -> None:
        """`None` = "nhánh này không tới được chunk"; `0.0` = "tìm ra và thấy
        không liên quan". Gộp hai thứ đó là đúng cái bẫy `TD-11`."""
        fused = self._branch().fuse(_dense("a"), _sparse("b"), top_k=10)
        by_key = {r.chunk.chunk_id: r for r in fused}
        assert by_key["a"].sparse_score is None
        assert by_key["b"].dense_score is None

    def test_incomparable_score_scales_do_not_leak_into_the_ranking(self) -> None:
        """Điểm sparse ở đây lớn hơn dense ~40 lần (đúng như thực tế: dot product
        không có trần, cosine có). Nếu thang điểm lọt vào phép hợp nhất thì nhánh
        sparse sẽ luôn thắng. Hợp nhất theo thứ hạng nên nó không lọt."""
        fused = self._branch().fuse(_dense("a"), _sparse("b"), top_k=10)
        assert [r.chunk.chunk_id for r in fused] == ["a", "b"]
        assert fused[0].score == pytest.approx(fused[1].score)

    def test_ranks_are_contiguous_from_one(self) -> None:
        fused = self._branch().fuse(_dense("a", "b", "c"), _sparse("d", "e"), top_k=10)
        assert [r.rank for r in fused] == [1, 2, 3, 4, 5]

    def test_top_k_truncates(self) -> None:
        fused = self._branch().fuse(_dense("a", "b", "c"), _sparse("d", "e"), top_k=2)
        assert len(fused) == 2
        assert [r.rank for r in fused] == [1, 2]

    def test_empty_sparse_branch_falls_back_to_dense(self) -> None:
        """Trạng thái thật khi truy vấn không trùng token nào (`W2-03`)."""
        fused = self._branch().fuse(_dense("a", "b"), [], top_k=10)
        assert [r.chunk.chunk_id for r in fused] == ["a", "b"]
        assert all(r.sparse_score is None for r in fused)

    def test_empty_dense_branch_falls_back_to_sparse(self) -> None:
        fused = self._branch().fuse([], _sparse("a", "b"), top_k=10)
        assert [r.chunk.chunk_id for r in fused] == ["a", "b"]

    def test_both_empty(self) -> None:
        assert self._branch().fuse([], [], top_k=10) == []

    def test_k_is_honoured(self) -> None:
        fused = self._branch(k=0).fuse(_dense("a"), [], top_k=10)
        assert fused[0].score == pytest.approx(1.0)

    def test_weights_are_honoured(self) -> None:
        fused = self._branch(weights=(3.0, 1.0)).fuse(_dense("a"), _sparse("a"), top_k=10)
        assert fused[0].score == pytest.approx(3 / 61 + 1 / 61)

    def test_dense_wins_ties(self) -> None:
        """`W2-03` đo được dense mạnh hơn (`hit_rate@10` 0,6268 vs 0,5120), nên
        nghiêng về nó khi không phân biệt được là tiên nghiệm đúng. Test này canh
        thứ tự truyền vào `reciprocal_rank_fusion` không bị đảo."""
        fused = self._branch().fuse(_dense("d"), _sparse("s"), top_k=10)
        assert [r.chunk.chunk_id for r in fused] == ["d", "s"]

    def test_deterministic(self) -> None:
        branch = self._branch()
        args = (_dense("a", "b", "c"), _sparse("c", "b", "a"))
        first = [r.chunk.chunk_id for r in branch.fuse(*args, top_k=10)]
        for _ in range(10):
            assert [r.chunk.chunk_id for r in branch.fuse(*args, top_k=10)] == first

    def test_chunk_object_survives_the_round_trip(self) -> None:
        fused = self._branch().fuse(_dense("a"), [], top_k=10)
        assert fused[0].chunk.content == "nội dung của a"


class TestForwarding:
    def test_fetch_doc_chunks_is_forwarded(self) -> None:
        """Hố im lặng của `W2-03`, lần thứ hai: thiếu method này thì eval harness
        lặng lẽ chấm lần chạy hybrid bằng bộ nhãn khác lần chạy dense."""
        branch = build_branch(_store(), "hybrid")
        assert callable(getattr(branch, "fetch_doc_chunks", None))

    def test_verify_schema_is_forwarded(self) -> None:
        assert callable(getattr(build_branch(_store(), "hybrid"), "verify_schema", None))

    def test_no_blanket_getattr_forwarding(self) -> None:
        branch = build_branch(_store(), "hybrid")
        assert not hasattr(branch, "upsert")
