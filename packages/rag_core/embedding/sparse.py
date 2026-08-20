"""Sparse vector — biểu diễn "chỉ lưu chiều khác 0".

Vì sao cần một kiểu riêng thay vì `dict[int, float]`: sparse vector sẽ đi qua ba
tầng (provider → Qdrant upsert → retriever) và mỗi tầng có quy ước riêng về thứ
tự và về việc có được chứa số 0 hay không. Một `dict` lỏng lẻo thì không tầng
nào giữ được bất biến; sai ở đây biểu hiện thành "sparse retrieval hoạt động
nhưng tệ", không thành lỗi.

Bất biến, cưỡng chế lúc khởi tạo:

* `indices` **tăng nghiêm ngặt** — không trùng, đã sắp. Trùng index nghĩa là hai
  trọng số cho cùng một token, và tầng dưới sẽ lặng lẽ chọn một cái.
* `values` **dương** thực sự. "Sparse" đúng nghĩa là chỉ lưu chỗ khác 0; giữ lại
  entry bằng 0 làm phồng payload và làm mọi phép đếm "bao nhiêu token khớp" sai.
* Cùng độ dài.

⚠️ `SparseVector` **rỗng** là hợp lệ và KHÁC với `None`. Rỗng = "đã tính, không
token nào có trọng số dương" (text chỉ gồm special token). `None` = "provider
này không sinh sparse". Gộp hai thứ đó đúng là cách `TD-11` trốn được — xem
`EmbeddingProvider.max_sequence_tokens`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = ["SparseVector"]


@dataclass(frozen=True, slots=True)
class SparseVector:
    """Vector thưa: `indices[i]` mang trọng số `values[i]`.

    Với BGE-M3 thì `indices` là **token id** trong vocab 250.002 của XLM-R, nên
    hai vector chỉ so được với nhau khi cùng tokenizer. Không có chỗ nào trong
    kiểu này ghi lại tokenizer — đó là việc của `IndexConfig.fingerprint`.
    """

    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"indices và values phải cùng độ dài, nhận {len(self.indices)} và "
                f"{len(self.values)}"
            )
        # Không `strict=True`: hai dãy này cố ý lệch nhau đúng một phần tử.
        for prev, cur in zip(self.indices, self.indices[1:], strict=False):
            if cur <= prev:
                raise ValueError(
                    f"indices phải tăng nghiêm ngặt (không trùng, đã sắp); {prev} → {cur}"
                )
        if self.indices and self.indices[0] < 0:
            raise ValueError(f"index phải không âm, nhận {self.indices[0]}")
        for value in self.values:
            if value <= 0.0:
                raise ValueError(
                    f"values phải dương — entry bằng 0 không phải sparse, nhận {value}"
                )

    @classmethod
    def from_weights(cls, weights: Mapping[int, float]) -> SparseVector:
        """Dựng từ `{token_id: trọng số}`, tự bỏ entry ≤ 0 và tự sắp.

        Đây là đường vào duy nhất nên dùng: nó biến ba bất biến ở trên thành
        chuyện của kiểu dữ liệu chứ không phải chuyện người gọi phải nhớ.
        """
        kept = sorted((int(k), float(v)) for k, v in weights.items() if v > 0.0)
        return cls(
            indices=tuple(idx for idx, _ in kept),
            values=tuple(val for _, val in kept),
        )

    def __len__(self) -> int:
        return len(self.indices)

    def as_dict(self) -> dict[int, float]:
        return dict(zip(self.indices, self.values, strict=True))

    def as_qdrant(self) -> dict[str, list[int] | list[float]]:
        """Dạng `qdrant_client.models.SparseVector` nhận vào."""
        return {"indices": list(self.indices), "values": list(self.values)}

    def dot(self, other: SparseVector) -> float:
        """Tích vô hướng — điểm khớp lexical giữa truy vấn và tài liệu.

        Duyệt hợp nhất hai danh sách đã sắp, `O(n + m)`, không dựng dict trung
        gian. Với 250k chiều và vài trăm entry mỗi vector thì đây là phép nóng
        của `W2-03`.
        """
        total = 0.0
        i = j = 0
        while i < len(self.indices) and j < len(other.indices):
            left, right = self.indices[i], other.indices[j]
            if left == right:
                total += self.values[i] * other.values[j]
                i += 1
                j += 1
            elif left < right:
                i += 1
            else:
                j += 1
        return total

    def top(self, n: int) -> tuple[tuple[int, float], ...]:
        """`n` chiều nặng nhất — để đọc bằng mắt trong report và test.

        Tie-break theo index tăng để kết quả xác định.
        """
        ranked = sorted(zip(self.indices, self.values, strict=True), key=lambda p: (-p[1], p[0]))
        return tuple(ranked[:n])
