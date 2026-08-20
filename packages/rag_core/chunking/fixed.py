"""Chunking đệ quy theo ký tự — viết lại `RecursiveCharacterTextSplitter`.

Thuật toán: thử lần lượt danh sách separator từ "thô" tới "mịn"
(`đoạn văn → dòng → câu → từ → ký tự`). Với separator đầu tiên xuất hiện trong
text, cắt theo nó rồi gộp các mảnh lại thành chunk sát `chunk_size` nhất có thể;
mảnh nào tự nó đã vượt `chunk_size` thì đệ quy xuống separator mịn hơn.

Viết tay thay vì dùng LangChain: chỉ ~60 dòng, bỏ được một cây phụ thuộc nặng và
quan trọng hơn là test được từng nhánh (mảnh dài không có separator, text rỗng,
separator không tồn tại).

Từ `W1-11` splitter trả `TextPiece` (text + vùng xuất xứ) thay vì `str`, để golden
set neo được vào văn bản gốc thay vì vào `chunk_id` thuần vị trí — xem `TD-12` và
docstring của `pieces.py`. `split_recursive` vẫn còn, trả `list[str]` như trước,
và **phải** cho ra đúng chuỗi cũ: đó là bất biến có test canh trên corpus thật.
"""

from __future__ import annotations

from typing import ClassVar

from .base import Chunker, ChunkingStrategy
from .pieces import TextPiece, merge_pieces, shift

__all__ = ["FixedSizeChunker", "split_recursive", "split_recursive_pieces"]


def _merge_splits(
    pieces: list[TextPiece], separator: str, chunk_size: int, chunk_overlap: int
) -> list[TextPiece]:
    """Gộp các mảnh nhỏ thành chunk sát `chunk_size`, chừa `chunk_overlap` ký tự.

    Phần tính `total` giữ nguyên từng dòng so với bản chỉ-có-text: nó điều khiển
    chỗ cắt, nên lệch một ký tự là ra bộ chunk khác và số baseline khác.
    """
    sep_len = len(separator)
    merged: list[TextPiece] = []
    window: list[TextPiece] = []
    total = 0

    for piece in pieces:
        piece_len = len(piece.text)
        addition = piece_len + (sep_len if window else 0)

        if window and total + addition > chunk_size:
            merged.append(_strip_piece(merge_pieces(window, separator)))
            # Bỏ dần mảnh ở đầu cửa sổ cho tới khi phần giữ lại đủ nhỏ để làm
            # overlap và mảnh mới nhét vừa. `window` co lại mỗi vòng nên luôn dừng.
            while window and (
                total > chunk_overlap or total + piece_len + (sep_len if window else 0) > chunk_size
            ):
                removed = window.pop(0)
                total -= len(removed.text) + (sep_len if window else 0)

        window.append(piece)
        total += piece_len + (sep_len if len(window) > 1 else 0)

    if window:
        merged.append(_strip_piece(merge_pieces(window, separator)))

    return [m for m in merged if m.text]


def _strip_piece(piece: TextPiece) -> TextPiece:
    """`.strip()` phần text, thu span vào tương ứng.

    Thu span theo số ký tự khoảng trắng bị cắt hai đầu là **xấp xỉ**: khoảng trắng
    trong `text` đã nối có thể không tương ứng 1-1 với khoảng trắng trong nguyên
    bản. Chấp nhận được vì span là vùng xuất xứ (xem `pieces.py`), và thu vào
    luôn cho span hẹp hơn — hướng an toàn khi ánh xạ nhãn.
    """
    text = piece.text
    stripped = text.strip()
    if stripped == text:
        return piece
    lead = len(text) - len(text.lstrip())
    trail = len(text) - len(text.rstrip())
    start = piece.start + lead
    end = piece.end - trail
    if end <= start:  # text toàn khoảng trắng
        return TextPiece(stripped, piece.start, piece.end)
    return TextPiece(stripped, start, end)


def _split_keeping_offsets(text: str, separator: str) -> list[TextPiece]:
    """Tương đương `[s for s in text.split(separator) if s]`, có kèm offset.

    Con trỏ vẫn nhảy qua mảnh rỗng dù mảnh đó bị bỏ — nếu không thì offset của
    mọi mảnh sau một chuỗi separator liền nhau (`"\\n\\n\\n"`) đều lệch.
    """
    if separator == "":
        return [TextPiece(ch, i, i + 1) for i, ch in enumerate(text)]

    out: list[TextPiece] = []
    cursor = 0
    step = len(separator)
    for raw in text.split(separator):
        if raw:
            out.append(TextPiece(raw, cursor, cursor + len(raw)))
        cursor += len(raw) + step
    return out


def split_recursive_pieces(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextPiece]:
    """Bản trả `TextPiece` của `split_recursive`. Span tính từ đầu `text`."""
    if not text:
        return []
    if not separators:
        return [TextPiece(text, 0, len(text))]

    # Chọn separator thô nhất còn xuất hiện trong text.
    separator = separators[-1]
    remaining: list[str] = []
    for i, candidate in enumerate(separators):
        if candidate == "":
            separator = ""
            remaining = []
            break
        if candidate in text:
            separator = candidate
            remaining = separators[i + 1 :]
            break

    splits = _split_keeping_offsets(text, separator)

    final: list[TextPiece] = []
    pending: list[TextPiece] = []
    for piece in splits:
        if len(piece.text) < chunk_size:
            pending.append(piece)
            continue

        if pending:
            final.extend(_merge_splits(pending, separator, chunk_size, chunk_overlap))
            pending = []

        if remaining:
            final.extend(
                shift(
                    split_recursive_pieces(piece.text, remaining, chunk_size, chunk_overlap),
                    piece.start,
                )
            )
        else:
            # Hết separator để cắt mịn hơn: đành giữ nguyên mảnh quá khổ.
            final.append(piece)

    if pending:
        final.extend(_merge_splits(pending, separator, chunk_size, chunk_overlap))

    return final


def split_recursive(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Chỉ phần text. Giữ lại vì nhiều test và code gọi đọc dễ hơn khi không cần span."""
    return [p.text for p in split_recursive_pieces(text, separators, chunk_size, chunk_overlap)]


class FixedSizeChunker(Chunker):
    """Nhanh, xác định, không cần model. Là fallback của mọi chiến lược khác."""

    strategy: ClassVar[ChunkingStrategy] = ChunkingStrategy.FIXED

    def split_pieces(self, text: str) -> list[TextPiece]:
        return split_recursive_pieces(
            text,
            separators=list(self.config.separators),
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
