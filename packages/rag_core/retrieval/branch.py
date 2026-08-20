"""Chọn nhánh truy hồi từ một chuỗi tên — `W2-03`.

Tồn tại vì từ `W2-03` trở đi, "index nào" và "truy hồi thế nào" là hai câu hỏi
tách nhau: cùng một collection `rag_bgem3` phục vụ được nhánh dense, nhánh
sparse, và (từ `W2-04`) nhánh hybrid. Trước đó chúng dính vào nhau nên
`IndexConfig` một mình là đủ để dựng retriever.

Cố ý **không** đưa nhánh truy hồi vào `IndexConfig`: nó không quyết định vector
nào được ghi, nên nó không thuộc `fingerprint` — mà một trường nằm trong
`IndexConfig` nhưng ngoài `fingerprint` là thứ phải giải thích mỗi lần đọc lại
(đã có hai trường như vậy là `device` và `batch_size`, đủ rồi). Nhánh truy hồi là
tham số của **lần đo**, nên nó là cờ dòng lệnh và sau này là một trường của ma
trận thí nghiệm ở `W2-07`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..schemas import RetrievalMode
from .base import Retriever

if TYPE_CHECKING:
    from .qdrant_store import QdrantDenseRetriever

__all__ = ["SUPPORTED_MODES", "build_branch"]

#: Nhánh dựng được ở thời điểm hiện tại. `HYBRID` vào đây ở `W2-04`, `RERANKED`
#: ở `W2-05` — liệt kê tường minh để `--retrieval-mode hybrid` báo "chưa cài"
#: thay vì báo "tên không hợp lệ", hai chuyện rất khác nhau khi đang gỡ lỗi.
SUPPORTED_MODES: tuple[RetrievalMode, ...] = (RetrievalMode.DENSE, RetrievalMode.SPARSE)


def build_branch(store: QdrantDenseRetriever, mode: RetrievalMode | str) -> Retriever:
    """Bọc `store` thành retriever của nhánh được nêu.

    `RetrievalMode.DENSE` trả về chính `store` — nó *là* retriever dense, bọc
    thêm một lớp chỉ để đối xứng sẽ làm mọi log và mọi `retriever.name` đổi nghĩa
    so với các lần chạy W1/W2-01/W2-02, tức không so được số cũ nữa.
    """
    try:
        resolved = RetrievalMode(mode)
    except ValueError:
        raise ValueError(
            f"Nhánh truy hồi không hợp lệ: {mode!r}. Hợp lệ: {[m.value for m in RetrievalMode]}"
        ) from None

    if resolved is RetrievalMode.DENSE:
        return store
    if resolved is RetrievalMode.SPARSE:
        from .sparse import QdrantSparseRetriever

        return QdrantSparseRetriever(store)

    raise NotImplementedError(
        f"Nhánh {resolved.value!r} là tên hợp lệ nhưng chưa cài. "
        f"Hiện có: {[m.value for m in SUPPORTED_MODES]} "
        f"(hybrid ở `W2-04`, reranked ở `W2-05`)."
    )
