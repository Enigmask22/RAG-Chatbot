"""Nhánh sparse như một `Retriever` độc lập — `W2-03`.

`W2-02` đã cho `retrieve_sparse()` chạy được ở tầng store, nhưng chạy được và
**đo được** là hai chuyện khác nhau: eval harness (`pipeline/eval/retrieval_eval.py`)
chỉ biết gọi `Retriever.retrieve()`. Chừng nào sparse chưa mang hình dạng đó thì
không có con số nào cho câu "nhánh sparse đóng góp gì", và mọi phát biểu về nó
chỉ là suy luận từ thiết kế.

Đây là lớp **bọc**, không phải một store thứ hai: nó dùng lại đúng
`QdrantDenseRetriever` đang mở kết nối. Ba lý do:

* Một `QdrantClient` cho cả hai nhánh. Hai object store trên cùng collection là
  hai connection pool và hai bản cache schema — không có lợi ích nào bù lại.
* `W2-04` (RRF) sẽ bọc **cùng** store đó và gọi cả hai nhánh. Nếu sparse là một
  store riêng thì RRF phải tự đồng bộ hai đối tượng và tự tin rằng chúng trỏ vào
  cùng một chỗ.
* Điểm của mỗi nhánh vẫn về đúng ô của nó (`dense_score` / `sparse_score`), nên
  bảng ablation `W2-08` tách được đóng góp.

⚠️ `fetch_doc_chunks` **phải** được chuyển tiếp. Eval harness lấy nó bằng
`getattr` để tính lại nhãn golden set theo span (`TD-12`); thiếu nó thì harness
lặng lẽ rơi về `relevant_chunk_ids` ghi trong file, và lần chạy sparse sẽ được
chấm bằng **bộ nhãn khác** với lần chạy dense. Hai con số vẫn hiện ra, vẫn so
được bằng mắt, và vẫn vô nghĩa. Có test canh việc chuyển tiếp này.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..schemas import Chunk, RetrievedChunk
from .base import Retriever

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .qdrant_store import QdrantDenseRetriever

__all__ = ["QdrantSparseRetriever"]


class QdrantSparseRetriever(Retriever):
    """Truy hồi chỉ bằng trọng số lexical, trên cùng collection với nhánh dense."""

    def __init__(self, store: QdrantDenseRetriever) -> None:
        if not store.writes_sparse:
            # Chết ở lúc dựng, không phải ở truy vấn đầu. `_eval_against_index`
            # quét toàn bộ chunk của corpus để ánh xạ span **trước** khi truy vấn
            # câu đầu tiên, nên lỗi ở truy vấn đầu là lỗi đến sau vài giây quét vô
            # ích — và với `W2-07` (grid nhiều dòng) thì là sau vài phút.
            raise ValueError(
                f"Provider {store.embeddings.name!r} không sinh sparse vector "
                f"(sparse_vocab_size is None) nên không có nhánh sparse để truy hồi. "
                f"Với BGE-M3 thì có sẵn; với `HashingEmbeddingProvider` phải bật "
                f"`sparse=True`."
            )
        self.store = store
        self.name = f"qdrant-sparse:{store.collection}"

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        return self.store.retrieve_sparse(query, top_k, filters=filters)

    def fetch_doc_chunks(self, doc_ids: Sequence[str], *, batch: int = 512) -> list[Chunk]:
        """Chuyển tiếp nguyên vẹn — xem cảnh báo ở docstring của module.

        Không dùng `__getattr__` để chuyển tiếp mọi thứ: một wrapper "trong suốt"
        thì mypy không kiểm được gì, và cái hố mà docstring vừa nói tới sẽ mở lại
        ngay lần đổi tên method ở store.
        """
        return self.store.fetch_doc_chunks(doc_ids, batch=batch)

    def verify_schema(self) -> None:
        """Chuyển tiếp — để người gọi kiểm schema mà không phải với tới `.store`."""
        self.store.verify_schema()
