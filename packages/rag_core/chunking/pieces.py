"""`TextPiece` — một mảnh text kèm vùng xuất xứ trong tài liệu gốc.

Tách ra file riêng để `fixed.py` và `base.py` dùng chung mà không phải import
vòng: `base.Chunker._enforce_size` gọi splitter của `fixed`, còn `fixed.FixedSizeChunker`
kế thừa `base.Chunker`.

## Hợp đồng, và chỗ dễ hiểu sai nhất

`piece.text` **không** bắt buộc bằng `document[piece.start:piece.end]`.

Nghe như một khiếm khuyết, nhưng nó là điều kiện để giữ nguyên hành vi:

* Splitter đệ quy làm `[s for s in text.split(sep) if s]` — **bỏ mảnh rỗng** —
  rồi nối lại bằng `sep.join(...)`. Với `"A\\n\\nB"` tách theo `"\\n"` thì kết quả
  là `"A\\nB"`, ngắn hơn nguyên bản một ký tự.
* Splitter ngữ nghĩa nối các câu bằng dấu cách, bất kể nguyên bản ngăn nhau bằng
  gì.
* `_enforce_size` gộp mảnh nhỏ bằng `"\\n"`, cũng bất kể nguyên bản.

Ép `text` thành substring nguyên văn sẽ **đổi nội dung chunk** → đổi vector → đổi
mọi con số baseline. Nên hợp đồng đúng là: `[start, end)` là **vùng mà mảnh này
được dẫn ra từ**, có thể rộng hơn `text` vài ký tự khoảng trắng.

Điều đó đủ cho mục đích duy nhất của span: ánh xạ nhãn golden set qua các cấu
hình chunking khác nhau (`TD-12`). Nó **không** đủ để dùng span làm chỉ dẫn cắt,
và không chỗ nào trong dự án dùng nó như vậy.
"""

from __future__ import annotations

from typing import NamedTuple

__all__ = ["TextPiece", "merge_pieces", "shift"]


class TextPiece(NamedTuple):
    """Một mảnh text và vùng xuất xứ của nó. `end` loại trừ."""

    text: str
    start: int
    end: int


def shift(pieces: list[TextPiece], offset: int) -> list[TextPiece]:
    """Dịch mọi span thêm `offset`.

    Dùng khi splitter chạy đệ quy trên một mảnh con: nó trả span tính từ đầu mảnh
    con, còn người gọi cần span tính từ đầu tài liệu.
    """
    if offset == 0:
        return pieces
    return [TextPiece(p.text, p.start + offset, p.end + offset) for p in pieces]


def merge_pieces(pieces: list[TextPiece], joiner: str) -> TextPiece:
    """Nối text bằng `joiner`, span là hợp của các span.

    Span **hợp** chứ không phải tổng độ dài text: đó chính là chỗ `text` và span
    tách nhau ra, vì `joiner` có thể khác ký tự ngăn cách trong nguyên bản.

    Raises:
        ValueError: danh sách rỗng.
    """
    if not pieces:
        raise ValueError("không nối được danh sách mảnh rỗng")
    return TextPiece(
        joiner.join(p.text for p in pieces),
        min(p.start for p in pieces),
        max(p.end for p in pieces),
    )
