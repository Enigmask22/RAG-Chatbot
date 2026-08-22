"""Small-to-big: embed mảnh nhỏ, trả về khối lớn chứa nó.

Lý do kỹ thuật này tồn tại: mảnh càng nhỏ thì vector càng đúng chủ đề (một chunk
1000 ký tự nói ba chuyện thì vector của nó không nói chuyện nào rõ), nhưng mảnh
càng nhỏ thì mảnh trả về cho LLM càng thiếu ngữ cảnh. Small-to-big tách hai vai
đó ra: **child** để tìm, **parent** để đọc.

## Quyết định lớn: parent KHÔNG nằm trong index

Cách làm quen thuộc là index cả parent lẫn child rồi lọc parent ra khỏi kết quả
tìm kiếm. Ở dự án này làm vậy tốn ba thứ: thêm một field phẳng trong payload,
thêm một field trong `MetadataFilter`, và một lượt backfill cho collection đã
build — tức chạm đúng tầng mà `W2-06` vừa đo latency xong. Và nếu một đường tìm
kiếm nào đó quên filter thì parent lọt vào kết quả, âm thầm.

Ở đây parent **không phải một point**. Nó là **tập các child của nó**:

```
parent = ghép các child cùng `parent_chunk_id`, xếp theo `chunk_index`
```

`parent_chunk_id` vì thế là một **khoá gom nhóm**, không phải một id lấy được
bằng `fetch_chunks`. Id các anh em nằm trong `extra["parent_children"]`, nên tầng
lắp ngữ cảnh chỉ cần một lời gọi `fetch_chunks` là dựng lại được parent — không
nhân bản văn bản, không thêm point, không đổi payload, không đổi một con số nào
đã công bố.

## Hệ quả: child trong cùng một parent KHÔNG chồng lấn

`chunk_overlap` giữa các child cùng parent bị ép về 0. Không phải để cho tiện:
overlap tồn tại để một câu bị cắt đôi vẫn còn nguyên ở một trong hai chunk — mà
đó **đúng là vấn đề small-to-big giải quyết**, vì trúng child nào cũng trả về cả
parent. Giữ overlap ở đây thì phần chồng lấn sẽ xuất hiện **hai lần** trong
parent ghép lại, và LLM đọc một đoạn văn lặp.

Overlap **giữa các parent** thì vẫn giữ (`chunk_overlap` của config).

## Cái không dựng lại được nguyên vẹn

Parent ghép từ child sai lệch đúng bằng chỗ splitter bỏ ký tự separator ở mối
nối — `W3-06` §10 đo được là **60 dấu chấm** trên một văn bản 60 câu khi cắt
theo `". "`. Nên parent ghép lại **không** bằng `document[start:end]`, đúng theo
hợp đồng đã ghi ở `pieces.py`. `extra["parent_start"]`/`["parent_end"]` giữ vùng
xuất xứ thật để ai cần nguyên bản thì đọc từ tài liệu.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from ..schemas import Chunk, Document
from .base import Chunker, ChunkingStrategy
from .fixed import split_recursive_pieces
from .pieces import TextPiece, shift

__all__ = ["PARENT_CHILDREN_KEY", "PARENT_END_KEY", "PARENT_START_KEY", "ParentChildChunker"]

PARENT_CHILDREN_KEY = "parent_children"
"""`extra` của child: id mọi anh em cùng parent, theo thứ tự đọc."""

PARENT_START_KEY = "parent_start"
PARENT_END_KEY = "parent_end"
"""Vùng xuất xứ của parent trong `Document.content`."""


def parent_id(doc_id: str, index: int) -> str:
    """Khoá gom nhóm của parent thứ `index`.

    `p` ở giữa để không bao giờ đụng `chunk_id` của child (`{doc_id}::{i:05d}`) —
    hai không gian tên đi chung trong `extra` và trong log, nên chúng phải phân
    biệt được bằng mắt.
    """
    return f"{doc_id}::p{index:05d}"


class ParentChildChunker(Chunker):
    """Cắt hai tầng: parent theo `chunk_size * parent_size_multiple`, rồi child.

    Chỉ **child** được trả về — xem docstring module. Mỗi child mang
    `parent_chunk_id` và id các anh em, đủ để `retrieval.context` dựng lại parent.
    """

    strategy: ClassVar[ChunkingStrategy] = ChunkingStrategy.PARENT_CHILD

    def split_pieces(self, text: str) -> list[TextPiece]:
        """Chỉ phần child, phẳng — hợp đồng của lớp cha.

        Quan hệ cha-con cần `doc_id` để đặt tên nhóm nên nó được dựng ở `chunk`.
        Hàm này vẫn phải cho ra **đúng** dãy mảnh ấy, nếu không thì `split_text`
        và `chunk` nói hai chuyện khác nhau.
        """
        return [child for _, child in self._two_level(text)]

    def _parent_pieces(self, text: str) -> list[TextPiece]:
        return split_recursive_pieces(
            text,
            separators=list(self.sizing.separators),
            chunk_size=self.sizing.chunk_size * self.config.parent_size_multiple,
            chunk_overlap=self.sizing.chunk_overlap,
        )

    def _child_pieces(self, parent: TextPiece) -> list[TextPiece]:
        """Cắt một parent thành child, **overlap 0** — xem docstring module.

        Ép kích thước (`min_chunk_size`/`max_chunk_size`) áp **trong phạm vi một
        parent**, không trên danh sách gộp: gộp một child ngắn vào child liền
        trước qua ranh giới parent thì chunk sinh ra thuộc hai parent cùng lúc và
        `parent_chunk_id` của nó là một lời nói dối. Cùng khuôn với `W3-03`
        (`section_path`) và với lỗi bản POC gộp qua ranh giới tài liệu.
        """
        pieces = split_recursive_pieces(
            parent.text,
            separators=list(self.sizing.separators),
            chunk_size=self.sizing.chunk_size,
            chunk_overlap=0,
        )
        kept = [p for p in pieces if p.text.strip()]
        return shift(self._enforce_size(kept), parent.start) or [parent]

    def _two_level(self, text: str) -> list[tuple[int, TextPiece]]:
        """`(chỉ số parent, mảnh child)` theo thứ tự đọc."""
        out: list[tuple[int, TextPiece]] = []
        index = 0
        for parent in self._parent_pieces(text):
            if not parent.text.strip():
                continue
            for child in self._child_pieces(parent):
                out.append((index, child))
            index += 1
        return out

    def chunk(self, documents: Sequence[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            limit = self._begin_sizing(doc.content)
            try:
                pairs = self._two_level(doc.content)
                if limit is not None:
                    pairs = self._fit_children(pairs, limit)
            finally:
                self._end_sizing()

            spans = self._parent_spans(pairs)
            children = [
                Chunk(
                    chunk_id=f"{doc.doc_id}::{position:05d}",
                    doc_id=doc.doc_id,
                    content=piece.text,
                    chunk_index=position,
                    parent_chunk_id=parent_id(doc.doc_id, group),
                    metadata=doc.metadata,
                    start_char=piece.start,
                    end_char=piece.end,
                    extra={
                        PARENT_START_KEY: spans[group][0],
                        PARENT_END_KEY: spans[group][1],
                    },
                )
                for position, (group, piece) in enumerate(pairs)
            ]
            chunks.extend(self._link_siblings(children))
        return chunks

    def _fit_children(
        self, pairs: list[tuple[int, TextPiece]], limit: int
    ) -> list[tuple[int, TextPiece]]:
        """Áp trần token lên child, giữ nguyên nhóm parent (`W3-06`)."""
        fitted = self._fit_tokens([piece for _, piece in pairs], limit)
        return [(pairs[source][0], piece) for source, piece in fitted]

    @staticmethod
    def _parent_spans(pairs: list[tuple[int, TextPiece]]) -> dict[int, tuple[int, int]]:
        spans: dict[int, tuple[int, int]] = {}
        for group, piece in pairs:
            low, high = spans.get(group, (piece.start, piece.end))
            spans[group] = (min(low, piece.start), max(high, piece.end))
        return spans

    @staticmethod
    def _link_siblings(children: list[Chunk]) -> list[Chunk]:
        """Điền `extra["parent_children"]` — phải làm SAU khi có `chunk_id`.

        Đây là lý do `chunk` được ghi đè thay vì dùng nguyên lớp cha: `chunk_id`
        do lớp cha đánh theo vị trí, nên danh sách anh em chỉ tồn tại sau khi cả
        tài liệu đã được đánh số.
        """
        groups: dict[str | None, list[str]] = {}
        for child in children:
            groups.setdefault(child.parent_chunk_id, []).append(child.chunk_id)
        return [
            child.model_copy(
                update={
                    "extra": {
                        **child.extra,
                        PARENT_CHILDREN_KEY: groups[child.parent_chunk_id],
                    }
                }
            )
            for child in children
        ]
