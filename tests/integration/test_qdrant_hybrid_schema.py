"""`W2-02` — một collection Qdrant chứa cả dense và sparse, query được độc lập.

Dùng `HashingEmbeddingProvider` chứ không phải BGE-M3: bài test ở đây nói về
**schema và đường ghi/đọc của Qdrant**, không về chất lượng embedding. Bắt nó nạp
2,2GB trọng số và cần GPU sẽ làm `make test-integration` không chạy được trên CI,
để đổi lấy đúng con số không ai đọc.

Chất lượng embedding BGE-M3 đã có test riêng ở `tests/unit/test_bge_m3.py`.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from qdrant_client import models

from rag_core.embedding import HashingEmbeddingProvider
from rag_core.retrieval import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    QdrantDenseRetriever,
    chunk_point_id,
)
from rag_core.schemas import Chunk, DocType, DocumentMetadata, Language, RetrievalMode

pytest.importorskip("qdrant_client", reason="cần extra `qdrant`: uv sync --extra qdrant")
pytestmark = pytest.mark.integration

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")

# Từ khoá lạ (`GSO-2024-XII`) cố ý chỉ xuất hiện ở một chunk: đó là loại truy vấn
# mà `W2-03` nói sparse phải bắt được. Ở đây chỉ cần nó tồn tại để phân biệt hai
# nhánh; việc chứng minh sparse thắng dense trên nó là việc của `W2-03`.
_TEXTS = {
    "budget": "Ngân sách nhà nước năm 2024 tăng chi đầu tư công cho hạ tầng giao thông.",
    "code": "Mã báo cáo GSO-2024-XII ghi nhận số liệu thống kê quý bốn.",
    "football": "Đội bóng ghi bàn thắng quyết định ở phút cuối của trận chung kết.",
}


def _chunk(key: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"doc-{key}::00000",
        doc_id=f"doc-{key}",
        content=text,
        chunk_index=0,
        metadata=DocumentMetadata(
            source_url=f"https://example.org/{key}",
            license="CC BY 4.0",
            lang=Language.VI,
            doc_type=DocType.DEV_REPORT,
        ),
        extra={"tenant_id": "t1"},
    )


CHUNKS = [_chunk(key, text) for key, text in _TEXTS.items()]


def _store(collection: str, *, sparse: bool) -> QdrantDenseRetriever:
    return QdrantDenseRetriever(
        HashingEmbeddingProvider(dimension=256, sparse=sparse),
        collection=collection,
        url=QDRANT_URL,
    )


@pytest.fixture
def hybrid() -> Iterator[QdrantDenseRetriever]:
    collection = f"test_hybrid_{uuid.uuid4().hex[:8]}"
    store = _store(collection, sparse=True)
    try:
        store.ensure_collection(recreate=True)
    except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
        pytest.fail(f"Không kết nối được Qdrant tại {QDRANT_URL}: {exc}. Chạy `make up` trước.")
    try:
        yield store
    finally:
        store.client.delete_collection(collection)


@pytest.fixture
def dense_only() -> Iterator[QdrantDenseRetriever]:
    collection = f"test_denseonly_{uuid.uuid4().hex[:8]}"
    store = _store(collection, sparse=False)
    try:
        store.ensure_collection(recreate=True)
    except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
        pytest.fail(f"Không kết nối được Qdrant tại {QDRANT_URL}: {exc}. Chạy `make up` trước.")
    try:
        yield store
    finally:
        store.client.delete_collection(collection)


class TestSchema:
    def test_one_collection_holds_both_vector_kinds(self, hybrid: QdrantDenseRetriever) -> None:
        """DoD của `W2-02`. Hai collection riêng cũng "chạy" nhưng thì mỗi truy
        vấn hybrid thành hai lần đi mạng và hai bộ point phải giữ đồng bộ tay."""
        dense_sizes, sparse_names = hybrid.live_schema()
        assert dense_sizes == {DENSE_VECTOR_NAME: 256}
        assert sparse_names == frozenset({SPARSE_VECTOR_NAME})

    def test_dense_only_provider_creates_no_sparse_config(
        self, dense_only: QdrantDenseRetriever
    ) -> None:
        _, sparse_names = dense_only.live_schema()
        assert sparse_names == frozenset()

    def test_sparse_index_has_no_idf_modifier(self, hybrid: QdrantDenseRetriever) -> None:
        """Trọng số của BGE-M3 là **đã học**. Chồng thêm IDF của Qdrant lên là
        nhân đôi phép hạ bậc từ phổ biến, và nó hỏng theo kiểu im lặng: điểm vẫn
        ra số, chỉ là sai. IDF dành cho nhánh BM25 thô ở `W2-03`.
        """
        params = hybrid.client.get_collection(hybrid.collection).config.params
        sparse = (params.sparse_vectors or {})[SPARSE_VECTOR_NAME]
        assert sparse.modifier in (None, "none")

    def test_writes_sparse_follows_the_provider(
        self, hybrid: QdrantDenseRetriever, dense_only: QdrantDenseRetriever
    ) -> None:
        assert hybrid.writes_sparse is True
        assert dense_only.writes_sparse is False


class TestSchemaGuard:
    """`ensure_collection` phải chết ở giây đầu, không phải giữa job 15.000 chunk."""

    def test_adding_sparse_to_an_existing_dense_collection_is_refused(
        self, dense_only: QdrantDenseRetriever
    ) -> None:
        upgraded = _store(dense_only.collection, sparse=True)
        with pytest.raises(RuntimeError, match="không khớp cấu hình"):
            upgraded.ensure_collection()

    def test_the_error_says_what_to_run(self, dense_only: QdrantDenseRetriever) -> None:
        """Một thông báo chỉ nói "schema mismatch" thì người đọc vẫn phải đi đọc
        code. Nó phải nói ra `--recreate`."""
        upgraded = _store(dense_only.collection, sparse=True)
        with pytest.raises(RuntimeError, match="--recreate"):
            upgraded.ensure_collection()

    def test_dimension_change_is_refused(self, dense_only: QdrantDenseRetriever) -> None:
        """Ca đổi model embedding: 256 → 512 chiều."""
        wider = QdrantDenseRetriever(
            HashingEmbeddingProvider(dimension=512, sparse=False),
            collection=dense_only.collection,
            url=QDRANT_URL,
        )
        with pytest.raises(RuntimeError, match="số chiều dense"):
            wider.ensure_collection()

    def test_dense_only_provider_on_hybrid_collection_is_refused(
        self, hybrid: QdrantDenseRetriever
    ) -> None:
        """Chiều ngược lại cũng phải chặn: eval bằng provider dense-only trên
        index hybrid cho ra con số trông bình thường trong khi nửa index bị bỏ."""
        downgraded = _store(hybrid.collection, sparse=False)
        with pytest.raises(RuntimeError, match="nửa index"):
            downgraded.ensure_collection()

    def test_recreate_is_the_escape_hatch(self, dense_only: QdrantDenseRetriever) -> None:
        upgraded = _store(dense_only.collection, sparse=True)
        upgraded.ensure_collection(recreate=True)
        _, sparse_names = upgraded.live_schema()
        assert sparse_names == frozenset({SPARSE_VECTOR_NAME})

    def test_matching_schema_passes_through(self, hybrid: QdrantDenseRetriever) -> None:
        hybrid.upsert(CHUNKS)
        hybrid.ensure_collection()  # không recreate → dữ liệu còn nguyên
        assert hybrid.count() == len(CHUNKS)


class TestUpsert:
    def test_every_point_carries_both_vectors(self, hybrid: QdrantDenseRetriever) -> None:
        hybrid.upsert(CHUNKS)
        points = hybrid.client.retrieve(
            collection_name=hybrid.collection,
            ids=[chunk_point_id(c.chunk_id) for c in CHUNKS],
            with_vectors=True,
        )
        assert len(points) == len(CHUNKS)
        for point in points:
            vectors = point.vector
            assert isinstance(vectors, dict)
            dense = vectors[DENSE_VECTOR_NAME]
            sparse = vectors[SPARSE_VECTOR_NAME]
            assert isinstance(dense, list) and len(dense) == 256
            assert isinstance(sparse, models.SparseVector)
            assert len(sparse.indices) > 0

    def test_upsert_is_still_idempotent(self, hybrid: QdrantDenseRetriever) -> None:
        hybrid.upsert(CHUNKS)
        hybrid.upsert(CHUNKS)
        assert hybrid.count() == len(CHUNKS)

    def test_batching_does_not_misalign_vectors_with_chunks(
        self, hybrid: QdrantDenseRetriever
    ) -> None:
        """`upsert` giờ index theo offset thay vì `zip(strict=True)`, nên lệch
        hàng sẽ gán embedding cho sai chunk. Batch nhỏ hơn số chunk để đi qua
        nhiều lô.
        """
        hybrid.upsert(CHUNKS, batch_size=2)
        for chunk in CHUNKS:
            results = hybrid.retrieve_sparse(chunk.content, top_k=1)
            assert results[0].chunk.chunk_id == chunk.chunk_id

    def test_empty_upsert(self, hybrid: QdrantDenseRetriever) -> None:
        assert hybrid.upsert([]) == 0


class TestIndependentQueries:
    """ "Query được độc lập" của DoD: hai nhánh, hai điểm, cùng một collection."""

    def test_dense_branch(self, hybrid: QdrantDenseRetriever) -> None:
        hybrid.upsert(CHUNKS)
        results = hybrid.retrieve("ngân sách đầu tư công", top_k=3)
        assert results[0].chunk.chunk_id == "doc-budget::00000"
        assert results[0].mode is RetrievalMode.DENSE
        assert results[0].dense_score is not None
        assert results[0].sparse_score is None, "nhánh dense không được bịa điểm sparse"

    def test_sparse_branch(self, hybrid: QdrantDenseRetriever) -> None:
        hybrid.upsert(CHUNKS)
        results = hybrid.retrieve_sparse("mã báo cáo GSO-2024-XII", top_k=3)
        assert results[0].chunk.chunk_id == "doc-code::00000"
        assert results[0].mode is RetrievalMode.SPARSE
        assert results[0].sparse_score is not None
        assert results[0].dense_score is None

    def test_the_two_branches_give_different_scores(self, hybrid: QdrantDenseRetriever) -> None:
        """Nếu hai nhánh cho cùng điểm thì `W2-04` (RRF) không có gì để hợp nhất
        — và nghĩa là một trong hai đang đọc sai named vector."""
        hybrid.upsert(CHUNKS)
        query = "ngân sách nhà nước 2024"
        dense = hybrid.retrieve(query, top_k=3)
        sparse = hybrid.retrieve_sparse(query, top_k=3)
        assert dense and sparse
        assert dense[0].score != sparse[0].score

    def test_ranks_are_contiguous_from_one(self, hybrid: QdrantDenseRetriever) -> None:
        """MRR/nDCG đọc `rank` như vị trí thật, nên dãy có lỗ là tính sai lặng lẽ."""
        hybrid.upsert(CHUNKS)
        for results in (
            hybrid.retrieve("ngân sách", top_k=10),
            hybrid.retrieve_sparse("ngân sách", top_k=10),
        ):
            assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_sparse_respects_metadata_filter(self, hybrid: QdrantDenseRetriever) -> None:
        """Filter phải áp ở tầng Qdrant cho **cả hai** nhánh, không chỉ dense —
        nếu không thì `W2-06` (cô lập tenant) có một lỗ đúng bằng nhánh sparse."""
        hybrid.upsert(CHUNKS)
        results = hybrid.retrieve_sparse(
            "mã báo cáo GSO-2024-XII", top_k=5, filters={"doc_id": "doc-football"}
        )
        assert [r.chunk.doc_id for r in results] == ["doc-football"] * len(results)

    def test_sparse_query_with_no_lexical_overlap_returns_nothing(
        self, hybrid: QdrantDenseRetriever
    ) -> None:
        """Đặc tính thật của sparse, và là lý do `W2-04` cần cả hai nhánh: không
        trùng token thì sparse không trả gì, trong khi dense vẫn đoán được."""
        hybrid.upsert(CHUNKS)
        query = "zzzqqq khongtontai"
        assert hybrid.retrieve_sparse(query, top_k=5) == []
        assert hybrid.retrieve(query, top_k=5) != []

    def test_sparse_on_dense_only_store_refuses(self, dense_only: QdrantDenseRetriever) -> None:
        dense_only.upsert(CHUNKS)
        with pytest.raises(RuntimeError, match="không sinh sparse"):
            dense_only.retrieve_sparse("ngân sách", top_k=3)
