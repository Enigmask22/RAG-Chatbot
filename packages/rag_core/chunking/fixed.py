"""Chunking đệ quy theo ký tự — viết lại `RecursiveCharacterTextSplitter`.

Thuật toán: thử lần lượt danh sách separator từ "thô" tới "mịn"
(`đoạn văn → dòng → câu → từ → ký tự`). Với separator đầu tiên xuất hiện trong
text, cắt theo nó rồi gộp các mảnh lại thành chunk sát `chunk_size` nhất có thể;
mảnh nào tự nó đã vượt `chunk_size` thì đệ quy xuống separator mịn hơn.

Viết tay thay vì dùng LangChain: chỉ ~60 dòng, bỏ được một cây phụ thuộc nặng và
quan trọng hơn là test được từng nhánh (mảnh dài không có separator, text rỗng,
separator không tồn tại).
"""

from __future__ import annotations

from typing import ClassVar

from .base import Chunker, ChunkingStrategy

__all__ = ["FixedSizeChunker", "split_recursive"]


def _merge_splits(
    splits: list[str], separator: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Gộp các mảnh nhỏ thành chunk sát `chunk_size`, chừa `chunk_overlap` ký tự."""
    sep_len = len(separator)
    merged: list[str] = []
    window: list[str] = []
    total = 0

    for piece in splits:
        piece_len = len(piece)
        addition = piece_len + (sep_len if window else 0)

        if window and total + addition > chunk_size:
            merged.append(separator.join(window).strip())
            # Bỏ dần mảnh ở đầu cửa sổ cho tới khi phần giữ lại đủ nhỏ để làm
            # overlap và mảnh mới nhét vừa. `window` co lại mỗi vòng nên luôn dừng.
            while window and (
                total > chunk_overlap or total + piece_len + (sep_len if window else 0) > chunk_size
            ):
                removed = window.pop(0)
                total -= len(removed) + (sep_len if window else 0)

        window.append(piece)
        total += piece_len + (sep_len if len(window) > 1 else 0)

    if window:
        merged.append(separator.join(window).strip())

    return [m for m in merged if m]


def split_recursive(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    if not text:
        return []
    if not separators:
        return [text]

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

    splits = list(text) if separator == "" else [s for s in text.split(separator) if s]

    final: list[str] = []
    pending: list[str] = []
    for piece in splits:
        if len(piece) < chunk_size:
            pending.append(piece)
            continue

        if pending:
            final.extend(_merge_splits(pending, separator, chunk_size, chunk_overlap))
            pending = []

        if remaining:
            final.extend(split_recursive(piece, remaining, chunk_size, chunk_overlap))
        else:
            # Hết separator để cắt mịn hơn: đành giữ nguyên mảnh quá khổ.
            final.append(piece)

    if pending:
        final.extend(_merge_splits(pending, separator, chunk_size, chunk_overlap))

    return final


class FixedSizeChunker(Chunker):
    """Nhanh, xác định, không cần model. Là fallback của mọi chiến lược khác."""

    strategy: ClassVar[ChunkingStrategy] = ChunkingStrategy.FIXED

    def split_text(self, text: str) -> list[str]:
        return split_recursive(
            text,
            separators=list(self.config.separators),
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
