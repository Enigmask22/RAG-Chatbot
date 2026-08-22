"""Lắp ngữ cảnh: từ child truy hồi được, dựng lại parent, và **gộp trùng**.

Nửa còn lại của small-to-big (`W3-05`). `chunking/parent_child.py` cắt; module
này ghép.

## Gộp trùng là phần dễ bỏ sót, và nó là phần có giá

Top-10 child rất hay nằm chụm trong vài parent — đó chính là dấu hiệu truy hồi
đang đúng chỗ. Không gộp thì cùng một đoạn văn 1024 token đi vào prompt ba lần:
tốn token, và tệ hơn là **thiên lệch sự chú ý của LLM** về phía đoạn được lặp,
không phải vì nó đúng hơn mà vì chunker cắt nó thành nhiều mảnh hơn.

## Parent thiếu anh em thì phải NÓI, không được ghép lặng lẽ

`fetch_chunks` nhận `filters`, và filter là thứ có thể **giấu bớt** anh em: một
tenant khác, một `doc_type` bị loại, hoặc chunk đã bị xoá khi re-index. Ghép
những mảnh còn lại rồi trả về như một parent nguyên vẹn là dựng ra một đoạn văn
chưa từng tồn tại — hai đoạn không liền nhau dán vào nhau, không dấu vết.
`AssembledParent.complete` và `missing_children` để người gọi biết.

## Filter phải đi xuyên qua

`QdrantDenseRetriever.fetch_chunks` đã cảnh báo từ `W1-07`: lấy chunk **theo id**
không đi qua tầng filter của truy vấn vector, nên một `chunk_id` đoán được sẽ trả
về nội dung đầy đủ dù mọi truy vấn đều đã lọc đúng. Mở rộng ngữ cảnh là **đúng
cái đường vòng đó**, và `W3-05` là chỗ tiêu thụ đầu tiên. Nên `expand_to_parents`
**bắt buộc** nhận `filters` và chuyển thẳng xuống; người gọi ở tầng serving phải
truyền đúng filter đã dùng cho lượt search.

Ranh giới cần nói rõ: `filters` chỉ chắn **đường lấy thêm anh em**. Các child đã
có trong `results` được tin là hợp lệ, vì chúng đến từ lượt search mà người gọi
đã (phải) lọc rồi. Đo trên index thật với `lang=en`: 32 anh em bị chặn, 9/10
parent thành `complete=False` — không parent nào bị ghép lặng lẽ.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..chunking.parent_child import PARENT_CHILDREN_KEY
from ..schemas import Chunk, RetrievedChunk
from .filters import FilterSpec

__all__ = ["AssembledParent", "ChunkFetcher", "assemble_text", "expand_to_parents"]

logger = logging.getLogger(__name__)

JOINER = "\n"
"""Ghép anh em bằng xuống dòng, cùng quy ước với `merge_pieces` ở tầng chunking."""


@runtime_checkable
class ChunkFetcher(Protocol):
    """Phần `QdrantDenseRetriever` mà module này cần. Đủ nhỏ để test không cần Qdrant.

    `runtime_checkable` không phải để trang trí: bản đầu của `W3-05` khai method
    này là `get_by_ids` — một cái tên **không tồn tại** ở lớp thật — và 27 unit
    test vẫn xanh vì fake trong test khai đúng cái tên sai ấy. Protocol cấu trúc
    chỉ ràng buộc được khi có **một bên thật** đối chiếu, nên
    `tests/unit/test_parent_child.py` ghim `isinstance(QdrantDenseRetriever(...),
    ChunkFetcher)`.
    """

    def fetch_chunks(
        self, chunk_ids: Sequence[str], *, filters: FilterSpec = None
    ) -> Mapping[str, Chunk]: ...


@dataclass(frozen=True, slots=True)
class AssembledParent:
    """Một parent đã ghép, kèm dấu vết để không ai nhầm nó là nguyên vẹn."""

    parent_id: str
    doc_id: str
    text: str
    children: tuple[Chunk, ...]
    """Các child đã ghép, theo thứ tự đọc."""
    hit_children: tuple[str, ...]
    """`chunk_id` của những child **thật sự** được truy hồi trả về."""
    missing_children: tuple[str, ...] = ()
    """Anh em khai báo mà không lấy được — bị filter chặn, hoặc đã bị xoá."""
    best_rank: int = 0
    best_score: float = 0.0
    metadata: object | None = None
    section_path: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Không thiếu anh em nào. `False` thì `text` là văn bản **đứt quãng**."""
        return not self.missing_children

    @property
    def expansion_ratio(self) -> float:
        """Số ký tự parent chia số ký tự các child đã trúng. 1.0 = không nở ra."""
        hit_chars = sum(
            len(c.content) for c in self.children if c.chunk_id in set(self.hit_children)
        )
        return len(self.text) / hit_chars if hit_chars else 0.0


def assemble_text(children: Sequence[Chunk]) -> str:
    """Ghép các child theo `chunk_index`.

    ⚠️ Kết quả **không** bằng `document[parent_start:parent_end]`: splitter bỏ ký
    tự separator ở mỗi mối nối (`W3-06` §10 đo được 60 dấu chấm trên 60 câu). Ai
    cần nguyên bản thì đọc tài liệu bằng `extra["parent_start"]`/`["parent_end"]`.
    """
    return JOINER.join(c.content for c in sorted(children, key=lambda c: c.chunk_index))


def expand_to_parents(
    results: Sequence[RetrievedChunk],
    fetcher: ChunkFetcher | None = None,
    *,
    filters: FilterSpec = None,
    max_parents: int | None = None,
) -> list[AssembledParent]:
    """Gom child theo parent, lấy nốt anh em, ghép, **gộp trùng**, giữ thứ hạng.

    Thứ tự đầu ra theo **thứ hạng tốt nhất** trong nhóm: parent chứa kết quả số 1
    đứng đầu. Cần nêu rõ vì đây là chỗ dễ đổi thầm lặng — xếp theo số child trúng
    (một cách đo "độ chụm" nghe rất hợp lý) sẽ đẩy một parent bị cắt vụn lên trên
    một parent đúng hơn nhưng liền mạch, tức để **cấu hình chunker** quyết định
    thứ hạng.

    `fetcher=None` thì chỉ ghép những child đã có trong `results` và mọi parent
    thiếu anh em sẽ được đánh dấu `complete=False`.

    Chunk không có `parent_chunk_id` (mọi chunker trước `W3-05`) đi qua nguyên
    vẹn, mỗi cái thành một "parent" một-child — nên hàm này an toàn với index cũ.
    """
    groups: dict[str, list[RetrievedChunk]] = {}
    for item in results:
        key = item.chunk.parent_chunk_id or item.chunk.chunk_id
        groups.setdefault(key, []).append(item)

    wanted = _siblings_to_fetch(groups)
    fetched: Mapping[str, Chunk] = {}
    if wanted and fetcher is not None:
        fetched = fetcher.fetch_chunks(sorted(wanted), filters=filters)
        if len(fetched) < len(wanted):
            logger.info(
                "Mở rộng ngữ cảnh: %d/%d anh em lấy được (filter chặn hoặc đã bị xoá)",
                len(fetched),
                len(wanted),
            )

    out: list[AssembledParent] = []
    for key, hits in groups.items():
        out.append(_assemble_one(key, hits, fetched))
    out.sort(key=lambda p: p.best_rank)
    return out[:max_parents] if max_parents is not None else out


def _declared_siblings(chunk: Chunk) -> list[str]:
    raw = chunk.extra.get(PARENT_CHILDREN_KEY)
    if isinstance(raw, str):  # payload cũ có thể lưu dạng chuỗi phân cách
        return [part for part in raw.split(",") if part]
    if isinstance(raw, list):
        return [str(part) for part in raw]
    return []


def _siblings_to_fetch(groups: Mapping[str, list[RetrievedChunk]]) -> set[str]:
    """Anh em đã khai báo mà chưa có trong kết quả truy hồi."""
    wanted: set[str] = set()
    for hits in groups.values():
        have = {item.chunk.chunk_id for item in hits}
        for item in hits:
            wanted |= set(_declared_siblings(item.chunk)) - have
    return wanted


def _assemble_one(
    key: str, hits: Sequence[RetrievedChunk], fetched: Mapping[str, Chunk]
) -> AssembledParent:
    best = min(hits, key=lambda item: item.rank)
    have: dict[str, Chunk] = {item.chunk.chunk_id: item.chunk for item in hits}

    declared: list[str] = []
    for item in hits:
        for sibling in _declared_siblings(item.chunk):
            if sibling not in declared:
                declared.append(sibling)
    if not declared:
        declared = sorted(have)

    children: list[Chunk] = []
    missing: list[str] = []
    for chunk_id in declared:
        chunk = have.get(chunk_id) or fetched.get(chunk_id)
        if chunk is None:
            missing.append(chunk_id)
            continue
        children.append(chunk)

    return AssembledParent(
        parent_id=key,
        doc_id=best.chunk.doc_id,
        text=assemble_text(children),
        children=tuple(sorted(children, key=lambda c: c.chunk_index)),
        hit_children=tuple(item.chunk.chunk_id for item in sorted(hits, key=lambda i: i.rank)),
        missing_children=tuple(missing),
        best_rank=best.rank,
        best_score=best.score,
        metadata=best.chunk.metadata,
        section_path=list(best.chunk.section_path),
    )
