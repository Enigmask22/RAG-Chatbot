"""Interface của tầng embedding.

Mục đích của lớp trừu tượng này là **đổi model chỉ bằng config, không sửa code**.
Ablation ở W2 sẽ quét qua nhiều model; nếu chỗ nào cũng gọi thẳng
`SentenceTransformer` thì mỗi lần đổi model là một lần sửa code, và không thể
tin rằng hai nhánh ablation chạy cùng một đường code.

Quy ước chung cho mọi implementation:

* `embed_documents` trả mảng `(n, dim)` `float32`, thứ tự khớp đầu vào.
* `embed_query` trả mảng `(dim,)`.
* Batch phải cho **kết quả y hệt** khi gọi từng cái một (có test canh). Nghe hiển
  nhiên nhưng padding trong một số backend làm sai điều này.
* Nếu `normalize=True` thì vector có norm 1, khi đó dot product == cosine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray

__all__ = ["EmbeddingProvider", "FloatArray"]

FloatArray = NDArray[np.float32]


class EmbeddingProvider(ABC):
    """Nguồn sinh vector cho text."""

    #: Tên định danh dùng trong log, cache key và MLflow. Phải đủ để tái lập.
    name: str

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Số chiều của vector."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> FloatArray:
        """Embed một lô text. Trả mảng shape `(len(texts), dimension)`."""

    def embed_query(self, text: str) -> FloatArray:
        """Embed một truy vấn. Trả mảng shape `(dimension,)`.

        Mặc định dùng lại `embed_documents`. Model bất đối xứng (E5, BGE với
        instruction prefix) phải override để thêm prefix đúng cho truy vấn —
        quên bước này làm tụt recall rất nhiều mà không có lỗi nào báo ra.
        """
        return cast(FloatArray, self.embed_documents([text])[0])

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, dim={self.dimension})"


def l2_normalize(matrix: FloatArray) -> FloatArray:
    """Chuẩn hoá L2 theo hàng, an toàn với vector 0."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return cast(FloatArray, (matrix / norms).astype(np.float32))
