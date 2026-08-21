"""Interface của tầng rerank — `W2-05`.

Tồn tại vì `W2-04` đo được một kết luận cụ thể: hybrid là **bộ sinh ứng viên**
tốt (`recall@20` 0,6324 → 0,6754, có ý nghĩa) và **bộ xếp hạng cuối** tệ
(`hit_rate@1` đứng im ở 0,3397). Hai việc đó cần hai loại model khác nhau —
bi-encoder embed truy vấn và tài liệu **độc lập** nên nó không bao giờ thấy được
sự tương tác giữa hai bên; cross-encoder đọc cả cặp trong một forward pass.

Cái giá là không cache được: bi-encoder embed 15.814 chunk một lần rồi tìm bằng
ANN, còn cross-encoder phải chạy `len(candidates)` forward pass **cho mỗi truy
vấn**. Nên nó chỉ dùng được ở tầng thứ hai, trên một pool đã hẹp.

Quy ước cho mọi implementation:

* `score` trả đúng `len(texts)` điểm, **theo thứ tự đầu vào** — không sắp xếp.
* Điểm cao hơn = liên quan hơn.
* Batch phải cho kết quả y hệt khi gọi từng cái một (có test canh, giống hợp
  đồng của `EmbeddingProvider`).
* Điểm **không so được giữa các truy vấn khác nhau**. Cross-encoder chấm một
  cặp, không chấm một tài liệu; ngưỡng cứng kiểu `score > 0,5` là sai ở đây.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

__all__ = ["Reranker"]


class Reranker(ABC):
    """Chấm lại độ liên quan của một danh sách ứng viên với truy vấn."""

    #: Tên định danh dùng trong log, `retriever.name` và MLflow. Phải mang đủ
    #: những tham số **làm đổi điểm** để hai lần chạy khác nhau không trùng nhãn.
    name: str

    @abstractmethod
    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Điểm liên quan của từng `text` với `query`, theo đúng thứ tự đầu vào.

        Cố ý **không** sắp xếp và **không** nhận `RetrievedChunk`. Việc sắp xếp
        cần cả quy tắc tie-break lẫn các đối tượng `Chunk` đi kèm, và cả hai
        thuộc tầng retriever — xem `rag_core/retrieval/reranked.py`. Nếu hàm này
        trả về danh sách đã sắp thì tie-break nằm bên trong lớp bọc model, chỗ
        không có test nào nhìn vào.
        """
