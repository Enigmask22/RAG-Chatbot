"""Lưu trữ và truy hồi dense trên Qdrant.

Hai quyết định đặt từ đầu để khỏi phải migrate index ở W2:

* **Dùng named vector (`dense`) ngay cả khi mới có một loại vector.** W2 sẽ thêm
  sparse vào cùng collection. Collection tạo bằng vector vô danh không thể thêm
  named vector mà không build lại toàn bộ index — vài giờ GPU cho một việc lẽ ra
  miễn phí nếu quyết định đúng từ đầu.
* **Point ID là UUIDv5 sinh từ `chunk_id`.** Qdrant chỉ nhận UUID hoặc số nguyên
  làm ID. Sinh xác định từ `chunk_id` khiến việc upsert lại cùng một chunk ghi đè
  đúng point cũ thay vì tạo bản trùng — đây chính là tính idempotent mà `W1-08`
  yêu cầu, và nó phải nằm ở tầng này chứ không phải ở script build index.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..embedding.base import EmbeddingProvider
from ..schemas import Chunk, RetrievalMode, RetrievedChunk
from .base import Retriever

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

__all__ = ["DENSE_VECTOR_NAME", "QdrantDenseRetriever", "chunk_point_id"]

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "dense"
_ID_NAMESPACE = uuid.UUID("6f4f2f7a-6f0c-5d3f-9a1e-0c7d9b2a4e11")


def chunk_point_id(chunk_id: str) -> str:
    """ID xác định cho một chunk — cùng `chunk_id` luôn cho cùng point ID."""
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


class QdrantDenseRetriever(Retriever):
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        *,
        collection: str = "rag_chunks",
        url: str = "http://127.0.0.1:6333",
        api_key: str | None = None,
        client: QdrantClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.embeddings = embeddings
        self.collection = collection
        self.name = f"qdrant-dense:{collection}"
        self._client = client
        self._url = url
        self._api_key = api_key
        self._timeout = timeout

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(
                url=self._url, api_key=self._api_key, timeout=int(self._timeout)
            )
        return self._client

    # ------------------------------------------------------------ collection

    def ensure_collection(self, *, recreate: bool = False) -> None:
        from qdrant_client import models

        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if exists:
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=self.embeddings.dimension,
                    distance=models.Distance.COSINE,
                )
            },
        )
        # Index payload cho các field sẽ lọc ở W2-06. Tạo sớm thì rẻ; tạo sau khi
        # đã có vài trăm nghìn point thì Qdrant phải quét lại toàn bộ.
        for field, schema in (
            ("chunk_id", "keyword"),
            ("doc_id", "keyword"),
            ("lang", "keyword"),
            ("doc_type", "keyword"),
            ("tenant_id", "keyword"),
        ):
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=schema,
            )

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    # ------------------------------------------------------------ ghi

    def upsert(self, chunks: Sequence[Chunk], *, batch_size: int = 128) -> int:
        """Ghi chunk vào collection. Gọi lại với cùng chunk thì không sinh bản trùng."""
        from qdrant_client import models

        if not chunks:
            return 0

        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = list(chunks[start : start + batch_size])
            vectors = self.embeddings.embed_documents([c.content for c in batch])
            points = [
                models.PointStruct(
                    id=chunk_point_id(chunk.chunk_id),
                    vector={DENSE_VECTOR_NAME: np.asarray(vec, dtype=np.float32).tolist()},
                    payload=self._payload(chunk),
                )
                for chunk, vec in zip(batch, vectors, strict=True)
            ]
            self.client.upsert(collection_name=self.collection, points=points, wait=True)
            total += len(points)
        return total

    @staticmethod
    def _payload(chunk: Chunk) -> dict[str, Any]:
        # Lưu nguyên chunk để dựng lại được object đầy đủ lúc truy hồi, cộng vài
        # field phẳng ở cấp trên cùng để Qdrant lọc được mà không cần nested path.
        payload: dict[str, Any] = {"chunk": chunk.model_dump(mode="json")}
        payload["chunk_id"] = chunk.chunk_id
        payload["doc_id"] = chunk.doc_id
        payload["content_hash"] = chunk.content_hash
        if chunk.metadata is not None:
            payload["lang"] = chunk.metadata.lang.value
            payload["doc_type"] = chunk.metadata.doc_type.value
        if "tenant_id" in chunk.extra:
            payload["tenant_id"] = chunk.extra["tenant_id"]
        return payload

    def delete_by_doc(self, doc_id: str) -> None:
        from qdrant_client import models

        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
                    ]
                )
            ),
            wait=True,
        )

    # ------------------------------------------------------------ đọc

    def _build_filter(self, filters: dict[str, Any] | None) -> Any:
        from qdrant_client import models

        if not filters:
            return None
        # Chú thích kiểu tường minh: `list` là invariant nên `list[FieldCondition]`
        # không khớp `list[Condition]` mà `Filter.must` mong đợi.
        conditions: list[models.Condition] = []
        for key, value in filters.items():
            if isinstance(value, list | tuple | set):
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchAny(any=list(value)))
                )
            else:
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                )
        return models.Filter(must=conditions)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        vector = np.asarray(self.embeddings.embed_query(query), dtype=np.float32).tolist()
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            using=DENSE_VECTOR_NAME,
            limit=top_k,
            query_filter=self._build_filter(filters),
            with_payload=True,
        )

        results: list[RetrievedChunk] = []
        for rank, point in enumerate(response.points, start=1):
            payload = point.payload or {}
            raw_chunk = payload.get("chunk")
            if raw_chunk is None:  # pragma: no cover - point ghi bởi phiên bản cũ
                logger.warning("Point %s thiếu payload `chunk`, bỏ qua", point.id)
                continue
            results.append(
                RetrievedChunk(
                    chunk=Chunk.model_validate(raw_chunk),
                    score=float(point.score),
                    rank=rank,
                    mode=RetrievalMode.DENSE,
                    dense_score=float(point.score),
                )
            )
        return results
