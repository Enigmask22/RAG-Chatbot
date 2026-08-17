"""Interface của tầng truy hồi."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..schemas import RetrievedChunk

__all__ = ["Retriever"]


class Retriever(ABC):
    """Nguồn trả về chunk liên quan cho một truy vấn.

    Mọi nhánh (dense, sparse, hybrid, reranked) đều dùng chung interface này để
    eval harness chạy được trên nhánh bất kỳ mà không cần biết bên trong là gì —
    đó là điều kiện để bảng ablation ở W2 so sánh công bằng.
    """

    name: str

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Trả tối đa `top_k` chunk, sắp xếp giảm dần theo độ liên quan.

        `rank` bắt đầu từ 1 và liên tục. Metric xếp hạng (MRR, nDCG) phụ thuộc
        vào điều này nên nó là một phần của hợp đồng, không phải chi tiết cài đặt.
        """
