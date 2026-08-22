"""Chunking theo cấu trúc: cắt ở ranh giới heading trước, cắt theo ký tự sau.

Đây là chunker đầu tiên **tiêu thụ** thứ `W3-01` dựng ra
(`LoadedDocument.headings`) thay vì đoán lại cấu trúc từ văn bản. Ranh giới đó
quan trọng: cùng một tài liệu ba cấp heading cho ra ba kiểu đánh số khác nhau
tuỳ định dạng nguồn (`.docx` cho level 1/2/3, `.md`/`.html` cho `title` rồi
level 1/2), và `docling_backend._normalise_depth` đã hoà chúng lại. Dò lại `#`
trong markdown ở tầng này là dựng **quy tắc thứ hai**, và hai quy tắc sẽ lệch
nhau đúng ở chỗ khó thấy nhất — xem `reports/tasks/w3-01-docling-loader.md` §4.

## Ba quyết định có hệ quả

**1. Gộp mảnh ngắn qua ranh giới section thì `section_path` sẽ nói dối.**
`Chunker._enforce_size` gộp mảnh nhỏ hơn `min_chunk_size` vào mảnh liền trước.
Áp thẳng lên danh sách section thì một `Điều 4` ngắn bị gộp vào cuối `Điều 3`,
và chunk sinh ra mang span bắt đầu trong `Điều 3` → `section_path` là
`[Chương I, Điều 3]` trong khi nửa sau nội dung thuộc `Điều 4`. Không lỗi, không
cảnh báo, và citation trỏ sai điều luật. Cùng khuôn với lỗi bản POC gộp chunk
qua ranh giới **tài liệu** (điểm 2 ở docstring `base.py`), thấp hơn một cấp.

Lối ra ở đây không phải cấm gộp — cấm gộp thì văn bản pháp luật ra hàng nghìn
chunk một dòng — mà là **hạ `section_path` xuống tổ tiên chung** của các section
bị gộp. Gộp `Điều 3` với `Điều 4` cho ra `[Chương I]`: nông hơn, nhưng đúng.

**2. Ranh giới lùi về đầu dòng.** `Heading.start_char` trỏ vào **chữ** của
heading, vì `_collect` định vị bằng `text.find(nội_dung)` — nên với markdown
`## Điều 3` nó trỏ vào `Đ`, bỏ lại `## ` ở cuối chunk trước. Cắt ở đầu dòng thì
marker đi cùng heading của nó và chunk trước không kết thúc bằng `##` lơ lửng.

**3. Không prepend heading vào `content`.** `Chunk.section_header` đã có sẵn cho
việc đó và thuộc về tầng embed — chunker nhét heading vào text thì `start_char`
/`end_char` không còn là vùng xuất xứ nữa, và `TD-12` mất chỗ neo.

## Cái mà chunker này KHÔNG làm được trên corpus hiện tại

Cả 60 tài liệu corpus là `.txt` → `load_plain` → `headings = ()`. Nên trên corpus
hôm nay `StructureChunker` **thoái hoá về fixed** ở 60/60 tài liệu và
`section_path` rỗng ở 100% chunk. Đó không phải hỏng: text thuần **không mang**
cấu trúc máy đọc được, và đoán heading bằng regex chính là quy tắc thứ hai ở
trên. Đo được điều đó là kết quả của `W3-03`, không phải phần chưa làm — xem
`reports/tasks/w3-03-structure-chunker.md` §3.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence
from typing import ClassVar

from ..loaders.base import LoadedDocument
from ..schemas import Chunk, Document, DocumentMetadata
from .base import Chunker, ChunkingConfig, ChunkingStrategy
from .fixed import split_recursive_pieces
from .pieces import TextPiece, merge_pieces, shift
from .tokens import TokenCounter

__all__ = ["StructureChunker", "common_ancestor", "section_boundaries"]

logger = logging.getLogger(__name__)


def common_ancestor(left: Sequence[str], right: Sequence[str]) -> list[str]:
    """Tiền tố chung dài nhất của hai đường dẫn section.

    Là `section_path` đúng cho một chunk gộp từ hai section: nó khẳng định đúng
    phần mà cả hai nửa nội dung đều thoả, và không khẳng định gì hơn.
    """
    out: list[str] = []
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        out.append(a)
    return out


def section_boundaries(document: LoadedDocument) -> list[tuple[int, int]]:
    """Cặp `(vị trí cắt, vị trí hỏi đường dẫn)` cho từng section, sắp tăng dần.

    Hai số này **khác nhau**, và đó là chỗ sai dễ nhất ở cả module. Vị trí cắt
    lùi về **đầu dòng** để marker `##` đi cùng heading của nó. Nhưng
    `section_path_at` phải hỏi tại **chữ** của heading (`Heading.start_char`):
    hỏi tại `#` thì với hàm ấy heading này *chưa* bắt đầu, nên nó trả về đường
    dẫn của section **liền trước** — mọi chunk lệch đi đúng một section, không
    lỗi, không cảnh báo.

    Đo trên `wb1.pdf` khi hỏi nhầm chỗ: **486/587 chunk** mang `section_path`
    của section trước. Hỏi nhầm chỗ làm **11/24** phép kiểm ở
    `tests/unit/test_structure_chunker.py` đỏ, nên nó có bị chặn — nhưng thứ tìm
    ra nó là phép đối chiếu với `section_path_at` trên tài liệu thật, chạy trước
    khi bộ test kia tồn tại.

    Heading không định vị được (`start_char < 0`) bị bỏ qua — nó không có vị trí
    thì không cắt ở đâu được. Vẫn giữ trong `headings` để `section_path_at` thấy,
    xem docstring `Heading`.
    """
    text = document.text
    probes: dict[int, int] = {0: 0}
    for heading in document.headings:
        if not heading.located:
            continue
        start = min(heading.start_char, len(text))
        cut = text.rfind("\n", 0, start) + 1
        probes[cut] = max(probes.get(cut, -1), start)
    return sorted(probes.items())


class StructureChunker(Chunker):
    """Cắt theo heading, rồi ép kích thước **trong phạm vi từng section**.

    Cấu trúc phải được khai báo trước bằng `bind`, vì `Document` không mang
    heading — nó chỉ có `content` và `metadata`. Tài liệu không được `bind` thì
    rơi về splitter đệ quy và **bị đếm** (`documents_without_structure`), không
    im lặng trả `section_path` rỗng như thể tài liệu vốn không có cấu trúc.
    """

    strategy: ClassVar[ChunkingStrategy] = ChunkingStrategy.STRUCTURE

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        *,
        token_counter: TokenCounter | None = None,
    ) -> None:
        super().__init__(config, token_counter=token_counter)
        self._structures: dict[str, LoadedDocument] = {}
        self.documents_without_structure = 0
        self.documents_with_mismatched_text = 0
        self._paths: list[list[str]] = []

    # ------------------------------------------------------------ khai báo

    def bind(self, doc_id: str, document: LoadedDocument) -> None:
        """Khai báo cấu trúc của một tài liệu trước khi chunk nó."""
        self._structures[doc_id] = document

    def chunk_loaded(
        self,
        document: LoadedDocument,
        *,
        doc_id: str,
        metadata: DocumentMetadata,
    ) -> list[Chunk]:
        """`bind` + `chunk` trong một lời gọi — đường đi không quên được `bind`."""
        self.bind(doc_id, document)
        return self.chunk([Document(doc_id=doc_id, content=document.text, metadata=metadata)])

    # --------------------------------------------------------------- cắt

    def split_pieces(self, text: str) -> list[TextPiece]:
        """Splitter **mù cấu trúc**, chỉ là fallback.

        Phần việc thật nằm ở `_prepare_pieces`, vì cắt theo heading cần
        `LoadedDocument` chứ không cần `text`. Hợp đồng của lớp cha vẫn phải có
        phương thức này, nên nó trả đúng thứ mà một tài liệu không có heading
        đáng được nhận.
        """
        return split_recursive_pieces(
            text,
            separators=list(self.sizing.separators),
            chunk_size=self.sizing.chunk_size,
            chunk_overlap=self.sizing.chunk_overlap,
        )

    def _prepare_pieces(self, doc: Document) -> list[TextPiece]:
        structure = self._usable_structure(doc)
        if structure is None:
            pieces = super()._prepare_pieces(doc)
            self._paths = [[] for _ in pieces]
            return pieces

        limit = self._begin_sizing(doc.content)
        try:
            pieces, paths = self._split_sections(doc.content, structure)
            if limit is None:
                self._paths = paths
                return pieces
            # Trần token cắt một mảnh thành nhiều mảnh, nên `paths` phải giãn
            # theo — cả ba mảnh con thừa kế `section_path` của mảnh mẹ.
            fitted = self._fit_tokens(pieces, limit)
            self._paths = [paths[source] for source, _ in fitted]
            return [piece for _, piece in fitted]
        finally:
            self._end_sizing()

    def _usable_structure(self, doc: Document) -> LoadedDocument | None:
        """Cấu trúc dùng được cho tài liệu này, hoặc `None` kèm lý do đã đếm.

        Phép kiểm `structure.text != doc.content` là phép kiểm quan trọng nhất ở
        module này. `Heading.start_char` là offset **trong `LoadedDocument.text`**;
        nếu đường đi từ loader tới `Document` có thêm bước chuẩn hoá nào (đổi
        xuống dòng, cắt khoảng trắng) thì mọi offset lệch đi, mà chunker vẫn chạy
        trơn tru — chỉ là mỗi chunk mang `section_path` của một chỗ khác. Đúng
        khuôn `TD-12`. Lệch thì rơi về cắt theo ký tự: **mất** `section_path` còn
        hơn `section_path` **sai**.
        """
        structure = self._structures.get(doc.doc_id)
        if structure is None:
            self.documents_without_structure += 1
            logger.warning(
                "%s: chưa `bind` cấu trúc, StructureChunker rơi về cắt theo ký tự",
                doc.doc_id,
            )
            return None
        if structure.text != doc.content:
            self.documents_with_mismatched_text += 1
            logger.error(
                "%s: `Document.content` khác `LoadedDocument.text` (%d vs %d ký tự) — "
                "offset heading không còn dùng được, rơi về cắt theo ký tự",
                doc.doc_id,
                len(doc.content),
                len(structure.text),
            )
            return None
        if not any(h.located for h in structure.headings):
            return None
        return structure

    def _split_sections(
        self, text: str, structure: LoadedDocument
    ) -> tuple[list[TextPiece], list[list[str]]]:
        """Cắt theo heading, ép kích thước từng section, trả kèm path mỗi mảnh."""
        sections = [(cut, probe) for cut, probe in section_boundaries(structure) if cut < len(text)]
        edges = [*(cut for cut, _ in sections), len(text)]

        out: list[TextPiece] = []
        paths: list[list[str]] = []

        for (start, end), (_, probe) in zip(itertools.pairwise(edges), sections, strict=True):
            body = text[start:end]
            if not body.strip():
                continue
            path = list(structure.section_path_at(probe))

            section_pieces = shift([p for p in self._section_pieces(body) if p.text.strip()], start)
            for piece in section_pieces:
                if not self._merge_into_previous(out, paths, piece, path):
                    out.append(piece)
                    paths.append(path)
        return out, paths

    def _section_pieces(self, body: str) -> list[TextPiece]:
        """Ép kích thước **bên trong** một section. Span tính từ đầu `body`."""
        if len(body) <= self.sizing.max_chunk_size:
            return [TextPiece(body, 0, len(body))]
        return split_recursive_pieces(
            body,
            separators=list(self.sizing.separators),
            chunk_size=self.sizing.chunk_size,
            chunk_overlap=self.sizing.chunk_overlap,
        )

    def _merge_into_previous(
        self,
        out: list[TextPiece],
        paths: list[list[str]],
        piece: TextPiece,
        path: list[str],
    ) -> bool:
        """Gộp mảnh quá ngắn vào mảnh trước, hạ `section_path` xuống tổ tiên chung."""
        if not self.config.structure_merge_short_sections:
            return False
        if len(piece.text) >= self.sizing.min_chunk_size or not out:
            return False

        merged = merge_pieces([out[-1], piece], "\n")
        if len(merged.text) > self.sizing.max_chunk_size:
            return False

        out[-1] = merged
        paths[-1] = common_ancestor(paths[-1], path)
        return True

    def _section_path_for(self, doc: Document, index: int) -> list[str]:
        return self._paths[index] if index < len(self._paths) else []
