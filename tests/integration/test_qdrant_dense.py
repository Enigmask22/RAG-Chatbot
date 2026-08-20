"""W1-07 — dense retriever trên Qdrant: upsert, tìm kiếm, idempotent, lọc."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

from rag_core.embedding import HashingEmbeddingProvider
from rag_core.retrieval import DENSE_VECTOR_NAME, QdrantDenseRetriever, chunk_point_id
from rag_core.schemas import Chunk, DocType, DocumentMetadata, Language, RetrievalMode

pytest.importorskip("qdrant_client", reason="cần extra `qdrant`: uv sync --extra qdrant")
pytestmark = pytest.mark.integration

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")

_TOPICS = {
    "budget": "Ngân sách nhà nước năm 2024 tăng chi đầu tư công cho hạ tầng giao thông.",
    "budget_en": "The state budget for 2024 increases public investment in transport infrastructure.",
    "football": "Đội bóng ghi bàn thắng quyết định ở phút cuối của trận chung kết.",
    "weather": "Dự báo thời tiết cho thấy mưa lớn kéo dài trên khu vực miền Trung.",
}


def _chunk(key: str, text: str, lang: Language, doc_type: DocType, tenant: str) -> Chunk:
    return Chunk(
        chunk_id=f"doc-{key}::00000",
        doc_id=f"doc-{key}",
        content=text,
        chunk_index=0,
        metadata=DocumentMetadata(
            source_url=f"https://example.org/{key}",
            license="CC BY 4.0",
            lang=lang,
            doc_type=doc_type,
        ),
        extra={"tenant_id": tenant},
    )


CHUNKS = [
    _chunk("budget", _TOPICS["budget"], Language.VI, DocType.DEV_REPORT, "t1"),
    _chunk("budget_en", _TOPICS["budget_en"], Language.EN, DocType.DEV_REPORT, "t1"),
    _chunk("football", _TOPICS["football"], Language.VI, DocType.OTHER, "t1"),
    _chunk("weather", _TOPICS["weather"], Language.VI, DocType.OTHER, "t2"),
]


@pytest.fixture
def retriever() -> Iterator[QdrantDenseRetriever]:
    collection = f"test_dense_{uuid.uuid4().hex[:8]}"
    store = QdrantDenseRetriever(
        HashingEmbeddingProvider(dimension=256), collection=collection, url=QDRANT_URL
    )
    try:
        store.ensure_collection(recreate=True)
    except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
        pytest.fail(f"Không kết nối được Qdrant tại {QDRANT_URL}: {exc}. Chạy `make up` trước.")
    try:
        yield store
    finally:
        store.client.delete_collection(collection)


class TestCollection:
    def test_uses_named_vector(self, retriever: QdrantDenseRetriever) -> None:
        """Named vector từ đầu — W2 thêm sparse vào cùng collection mà không
        phải build lại toàn bộ index."""
        info = retriever.client.get_collection(retriever.collection)
        vectors = info.config.params.vectors
        assert isinstance(vectors, dict) and DENSE_VECTOR_NAME in vectors

    def test_ensure_collection_is_safe_to_call_twice(self, retriever: QdrantDenseRetriever) -> None:
        retriever.upsert(CHUNKS)
        retriever.ensure_collection()  # không recreate → dữ liệu còn nguyên
        assert retriever.count() == len(CHUNKS)


class TestUpsertAndSearch:
    def test_search_returns_relevant_chunk_first(self, retriever: QdrantDenseRetriever) -> None:
        retriever.upsert(CHUNKS)
        results = retriever.retrieve("ngân sách đầu tư công", top_k=3)

        assert results
        assert results[0].chunk.chunk_id == "doc-budget::00000"
        assert results[0].mode is RetrievalMode.DENSE

    def test_scores_are_descending_and_ranks_sequential(
        self, retriever: QdrantDenseRetriever
    ) -> None:
        # MRR và nDCG phụ thuộc trực tiếp vào hai tính chất này — chúng là một
        # phần của hợp đồng `Retriever`, không phải chi tiết cài đặt.
        retriever.upsert(CHUNKS)
        results = retriever.retrieve("ngân sách", top_k=4)

        assert [r.rank for r in results] == list(range(1, len(results) + 1))
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert all(r.dense_score == r.score for r in results)

    def test_payload_round_trips_full_chunk(self, retriever: QdrantDenseRetriever) -> None:
        retriever.upsert(CHUNKS)
        top = retriever.retrieve("ngân sách", top_k=1)[0]
        assert top.chunk.metadata is not None
        assert top.chunk.metadata.license == "CC BY 4.0"
        assert top.chunk.content == _TOPICS["budget"]

    def test_top_k_is_respected(self, retriever: QdrantDenseRetriever) -> None:
        retriever.upsert(CHUNKS)
        assert len(retriever.retrieve("ngân sách", top_k=2)) == 2

    def test_empty_upsert(self, retriever: QdrantDenseRetriever) -> None:
        assert retriever.upsert([]) == 0


class TestIdempotency:
    def test_upsert_twice_does_not_duplicate(self, retriever: QdrantDenseRetriever) -> None:
        """Tính idempotent nằm ở tầng store (point ID sinh xác định từ chunk_id),
        không phải ở script build index — đây là điều kiện cho `W1-08`."""
        retriever.upsert(CHUNKS)
        first = retriever.count()
        retriever.upsert(CHUNKS)
        assert retriever.count() == first == len(CHUNKS)

    def test_point_id_is_deterministic(self) -> None:
        assert chunk_point_id("doc-a::00001") == chunk_point_id("doc-a::00001")
        assert chunk_point_id("doc-a::00001") != chunk_point_id("doc-a::00002")

    def test_reupsert_updates_content(self, retriever: QdrantDenseRetriever) -> None:
        retriever.upsert(CHUNKS)
        edited = CHUNKS[0].model_copy(update={"content": "Nội dung ngân sách đã được sửa lại."})
        retriever.upsert([edited])

        assert retriever.count() == len(CHUNKS)
        top = retriever.retrieve("ngân sách", top_k=1)[0]
        assert "sửa lại" in top.chunk.content


class TestFiltering:
    def test_filter_by_language(self, retriever: QdrantDenseRetriever) -> None:
        retriever.upsert(CHUNKS)
        results = retriever.retrieve("budget investment", top_k=5, filters={"lang": "en"})
        assert results
        assert all(
            r.chunk.metadata is not None and r.chunk.metadata.lang is Language.EN for r in results
        )

    def test_filter_isolates_tenants(self, retriever: QdrantDenseRetriever) -> None:
        # Rò dữ liệu chéo tenant là lỗi nghiêm trọng nhất mà tầng này có thể mắc.
        retriever.upsert(CHUNKS)
        results = retriever.retrieve("thời tiết mưa lớn", top_k=5, filters={"tenant_id": "t1"})
        assert all(r.chunk.doc_id != "doc-weather" for r in results)

    def test_filter_accepts_list(self, retriever: QdrantDenseRetriever) -> None:
        retriever.upsert(CHUNKS)
        results = retriever.retrieve("ngân sách", top_k=5, filters={"doc_type": ["dev_report"]})
        assert {r.chunk.doc_id for r in results} == {"doc-budget", "doc-budget_en"}


def test_delete_by_doc(retriever: QdrantDenseRetriever) -> None:
    retriever.upsert(CHUNKS)
    retriever.delete_by_doc("doc-weather")
    assert retriever.count() == len(CHUNKS) - 1


class TestFetchChunks:
    """`fetch_chunks` là chỗ duy nhất phát hiện golden set trỏ tới chunk đã mất."""

    def test_returns_chunks_keyed_by_chunk_id(self, retriever: QdrantDenseRetriever) -> None:
        retriever.upsert(CHUNKS)
        found = retriever.fetch_chunks(["doc-budget::00000", "doc-weather::00000"])
        assert set(found) == {"doc-budget::00000", "doc-weather::00000"}
        assert found["doc-budget::00000"].content == _TOPICS["budget"]

    def test_missing_ids_are_absent_not_an_error(self, retriever: QdrantDenseRetriever) -> None:
        """Người gọi cần biết id NÀO thiếu — đó là lý do trả dict chứ không phải list."""
        retriever.upsert(CHUNKS)
        wanted = ["doc-budget::00000", "doc-khong-ton-tai::00000"]
        found = retriever.fetch_chunks(wanted)
        assert set(wanted) - set(found) == {"doc-khong-ton-tai::00000"}

    def test_empty_input_does_not_hit_qdrant(self, retriever: QdrantDenseRetriever) -> None:
        assert retriever.fetch_chunks([]) == {}

    def test_duplicate_ids_are_collapsed(self, retriever: QdrantDenseRetriever) -> None:
        retriever.upsert(CHUNKS)
        found = retriever.fetch_chunks(["doc-budget::00000"] * 3)
        assert len(found) == 1

    def test_deleted_chunk_disappears(self, retriever: QdrantDenseRetriever) -> None:
        """Đúng kịch bản build lại index với cấu hình chunking khác."""
        retriever.upsert(CHUNKS)
        retriever.delete_by_doc("doc-weather")
        assert retriever.fetch_chunks(["doc-weather::00000"]) == {}
