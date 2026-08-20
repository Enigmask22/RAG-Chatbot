"""Nhánh hybrid — dense + sparse hợp nhất bằng RRF. `W2-04`.

Bốn quyết định của tầng này, cả bốn đều có số đo đứng sau:

**Embed truy vấn MỘT lần.** Gọi `store.retrieve()` rồi `store.retrieve_sparse()`
sẽ chạy forward pass hai lần — 12,6 ms mỗi lần (`W2-03` §8), tức +12,6 ms cho
đúng một kết quả. `embed_query_hybrid()` cho cả dense và sparse từ một pass, y
như phía tài liệu ở `W2-01`. Đây là lý do lớp này đi thẳng xuống
`client.query_batch_points` thay vì dùng lại hai method của store.

**Một request HTTP cho cả hai nhánh.** `query_batch_points` gửi hai truy vấn
trong một lần, nên Qdrant tự chạy song song được và ta trả một round trip thay vì
hai.

**Tự cài RRF, không dùng `Fusion.RRF` của Qdrant.** Qdrant có fusion server-side
nhưng `k` của nó **không cấu hình được** và nó không trả thứ hạng của từng nhánh.
`W2-08` cần quét `k`, và `RetrievedChunk.dense_score`/`sparse_score` tồn tại để
tách đóng góp của mỗi nhánh. Có test integration đối chiếu với bản của Qdrant —
tự cài không có nghĩa là tự tin.

**`candidate_k` sâu hơn `top_k`, và đó là một cần điều khiển thật.** RRF chỉ thấy
được sự đồng thuận giữa hai nhánh trong phạm vi nó nhìn. Một chunk ở hạng 45 của
dense và hạng 3 của sparse có điểm `1/105 + 1/63 = 0,0254`, cao hơn một chunk chỉ
dense tìm ra ở hạng 2 (`1/62 = 0,0161`) — nhưng chỉ khi `candidate_k ≥ 45`. Lấy
nông thì mất đúng cái lợi mà hợp nhất mang lại.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..schemas import Chunk, RetrievalMode, RetrievedChunk
from .base import Retriever
from .qdrant_store import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    build_filter,
    points_to_chunks,
)
from .rrf import RRF_K, reciprocal_rank_fusion

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .qdrant_store import QdrantDenseRetriever

__all__ = ["DEFAULT_CANDIDATE_K", "QdrantHybridRetriever"]

#: Số ứng viên lấy từ **mỗi** nhánh trước khi hợp nhất. Xem docstring module về
#: lý do nó phải sâu hơn `top_k`.
DEFAULT_CANDIDATE_K = 50


class QdrantHybridRetriever(Retriever):
    """Truy hồi bằng cả hai nhánh trên cùng collection, hợp nhất theo thứ hạng."""

    def __init__(
        self,
        store: QdrantDenseRetriever,
        *,
        k: int = RRF_K,
        candidate_k: int | None = None,
        weights: tuple[float, float] | None = None,
    ) -> None:
        if not store.writes_sparse:
            # Chết lúc dựng, không lúc truy vấn đầu — cùng lý lẽ với
            # `QdrantSparseRetriever` (`W2-03`) và `ensure_collection` (`W2-02`).
            raise ValueError(
                f"Provider {store.embeddings.name!r} không sinh sparse vector "
                f"(sparse_vocab_size is None) nên không có gì để hợp nhất với nhánh "
                f"dense. Dùng `mode=dense`, hoặc bật sparse ở provider."
            )
        if candidate_k is not None and candidate_k < 1:
            raise ValueError(f"candidate_k phải ≥ 1, nhận {candidate_k}")
        self.store = store
        self.k = k
        self.candidate_k = candidate_k
        self.weights = weights
        # Tên mang cả `k` và `candidate_k`: hai lần chạy khác tham số phải khác
        # tên, nếu không thì bảng ablation `W2-08` có hai dòng trùng nhãn.
        suffix = f"rrf{k}-c{candidate_k or DEFAULT_CANDIDATE_K}"
        if weights is not None:
            suffix += f"-w{weights[0]:g}:{weights[1]:g}"
        self.name = f"qdrant-hybrid:{store.collection}:{suffix}"

    def _depth(self, top_k: int) -> int:
        """Số ứng viên lấy mỗi nhánh. Không bao giờ nông hơn `top_k`."""
        return max(top_k, self.candidate_k or DEFAULT_CANDIDATE_K)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        from qdrant_client import models

        hybrid = self.store.embeddings.embed_query_hybrid(query)
        if hybrid is None:  # pragma: no cover - bug provider, xem `__init__`
            raise RuntimeError(
                f"Provider {self.store.embeddings.name!r} khai sparse_vocab_size="
                f"{self.store.embeddings.sparse_vocab_size} nhưng embed_query_hybrid trả None"
            )
        dense_vector, sparse_vector = hybrid
        depth = self._depth(top_k)
        query_filter = build_filter(filters)

        responses = self.store.client.query_batch_points(
            collection_name=self.store.collection,
            requests=[
                models.QueryRequest(
                    query=np.asarray(dense_vector, dtype=np.float32).tolist(),
                    using=DENSE_VECTOR_NAME,
                    limit=depth,
                    filter=query_filter,
                    with_payload=True,
                ),
                models.QueryRequest(
                    query=models.SparseVector(**sparse_vector.as_qdrant()),
                    using=SPARSE_VECTOR_NAME,
                    limit=depth,
                    filter=query_filter,
                    with_payload=True,
                ),
            ],
        )
        dense_hits = points_to_chunks(responses[0].points, mode=RetrievalMode.DENSE)
        sparse_hits = points_to_chunks(responses[1].points, mode=RetrievalMode.SPARSE)
        return self.fuse(dense_hits, sparse_hits, top_k=top_k)

    def fuse(
        self,
        dense_hits: Sequence[RetrievedChunk],
        sparse_hits: Sequence[RetrievedChunk],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Hợp nhất hai danh sách đã xếp hạng. Tách khỏi `retrieve()` để test được.

        Nhánh **dense đứng trước** trong danh sách vào, và đó là một quyết định có
        số đo: điểm RRF bằng nhau xảy ra thường xuyên (một chunk ở hạng 3 của
        dense và một chunk khác ở hạng 3 của sparse có cùng điểm), và quy tắc
        tie-break thứ ba của `reciprocal_rank_fusion` ưu tiên danh sách đứng
        trước. `W2-03` đo được dense `hit_rate@10` 0,6268 vs sparse 0,5120, nên
        khi không phân biệt được thì nghiêng về dense là tiên nghiệm đúng.
        """
        chunks: dict[str, Chunk] = {}
        dense_scores: dict[str, float] = {}
        sparse_scores: dict[str, float] = {}
        for hit in dense_hits:
            chunks[hit.chunk.chunk_id] = hit.chunk
            dense_scores[hit.chunk.chunk_id] = hit.score
        for hit in sparse_hits:
            chunks[hit.chunk.chunk_id] = hit.chunk
            sparse_scores[hit.chunk.chunk_id] = hit.score

        fused = reciprocal_rank_fusion(
            [
                [hit.chunk.chunk_id for hit in dense_hits],
                [hit.chunk.chunk_id for hit in sparse_hits],
            ],
            k=self.k,
            weights=list(self.weights) if self.weights is not None else None,
            limit=top_k,
        )
        return [
            RetrievedChunk(
                chunk=chunks[item.key],
                score=item.score,
                rank=item.rank,
                mode=RetrievalMode.HYBRID,
                # Điểm gốc của mỗi nhánh được giữ nguyên, `None` nếu nhánh đó
                # không tìm ra chunk này. `None` ≠ 0,0: điểm 0 nghĩa là "tìm ra và
                # thấy không liên quan", còn `None` nghĩa là "không tới được" —
                # phân biệt đó là thứ `W2-08` cần để tách đóng góp.
                dense_score=dense_scores.get(item.key),
                sparse_score=sparse_scores.get(item.key),
            )
            for item in fused
        ]

    def fetch_doc_chunks(self, doc_ids: Sequence[str], *, batch: int = 512) -> list[Chunk]:
        """Chuyển tiếp — bắt buộc, xem `sparse.py` về hố im lặng của `getattr`."""
        return self.store.fetch_doc_chunks(doc_ids, batch=batch)

    def verify_schema(self) -> None:
        self.store.verify_schema()
