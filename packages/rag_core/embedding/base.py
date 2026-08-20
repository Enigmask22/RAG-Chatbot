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
* `max_sequence_tokens` và `count_tokens` phơi ra **giới hạn cửa sổ** của model.
  Hai thứ này không phục vụ việc embed; chúng tồn tại để việc cắt bớt text
  không thể xảy ra âm thầm nữa — xem `TD-11` và `truncation.py`.
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

    # ------------------------------------------------------- giới hạn cửa sổ

    @property
    def max_sequence_tokens(self) -> int | None:
        """Số token tối đa model thật sự đọc. `None` = provider không biết.

        Lý do property này tồn tại: `sentence-transformers` **cắt âm thầm** ở
        `max_seq_length` — không cảnh báo, không lỗi, phần text vượt quá chỉ
        đơn giản không tới được vector. Ở baseline `W1-13` chỗ này làm **15,7%
        văn bản** của corpus không bao giờ được embed, và không một con số nào
        trong report lộ ra điều đó (`TD-11`).

        `None` là "không biết", **không phải** "không có giới hạn". Người gọi
        phải phân biệt hai thứ đó, vì coi "không biết" thành "không giới hạn"
        đúng là cách bug ban đầu trốn được sáu tuần.
        """
        return None

    def count_tokens(self, texts: Sequence[str]) -> list[int] | None:
        """Số token của từng text theo tokenizer **thật** của model.

        Trả `None` nếu provider không đếm được. Đừng quy ước thành `0` — mọi
        model sẽ trông như "không cắt gì".

        Phải đếm **kèm special token** (`[CLS]`/`[SEP]`) vì đó là thứ đem so
        với `max_sequence_tokens`; bỏ hai token đó ra là báo thiếu đúng hai
        token ở mọi chunk sát ngưỡng.
        """
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, dim={self.dimension})"


def l2_normalize(matrix: FloatArray) -> FloatArray:
    """Chuẩn hoá L2 theo hàng, an toàn với vector 0."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return cast(FloatArray, (matrix / norms).astype(np.float32))
