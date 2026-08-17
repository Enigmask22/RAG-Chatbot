"""Chọn chunk từ index thật để làm nguyên liệu sinh câu hỏi.

Hai quyết định định hình cả chất lượng golden set:

**Lấy chunk từ Qdrant, không tự chunk lại corpus.** Chunk lại thì rẻ hơn và cho
ra `chunk_id` giống hệt *nếu* config trùng — nhưng "nếu" đó chính là chỗ hỏng.
Golden set trỏ tới `chunk_id` không tồn tại trong index thì recall bằng 0 mà
không có lỗi nào báo ra. Nguồn sự thật phải là index sẽ bị đo.

**Lọc chunk rác trước khi đưa cho model.** Mục lục, số trang, danh mục tài liệu
tham khảo, bảng số trần trụi — model vẫn sinh ra câu hỏi từ chúng, và những câu
đó không đo được gì cả. Bộ lọc ở đây cố ý thô: nó chỉ loại thứ rõ ràng không
phải văn xuôi, phần tinh vi để người review lo ở `W1-11`.

Quét hai lượt để không phải nạp 15.814 chunk vào RAM: lượt một lấy `chunk_id` +
`doc_id` + `lang` (payload nhỏ), chọn xong mới `retrieve` nội dung của đúng số
chunk cần.
"""

from __future__ import annotations

import logging
import random
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pipeline.eval.golden import QueryCategory
from rag_core.schemas import Chunk, Language

if TYPE_CHECKING:
    from rag_core.retrieval.qdrant_store import QdrantDenseRetriever

__all__ = [
    "ChunkGroup",
    "ChunkRef",
    "gutter_ratio",
    "hydrate",
    "is_prose_like",
    "load_chunk_refs",
    "mean_words_per_line",
    "plan_groups",
    "sample_groups",
]

logger = logging.getLogger(__name__)

_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_WORD_RE = re.compile(r"\w+", re.UNICODE)
#: Khoảng trắng dài ở GIỮA một dòng — dấu vết của máng phân cột trong PDF.
_GUTTER_RE = re.compile(r"\S {3,}\S")

#: Nhóm cần nhiều hơn một chunk, và cần chúng thuộc cùng một tài liệu.
MULTI_CHUNK_CATEGORIES: dict[QueryCategory, int] = {
    QueryCategory.MULTI_HOP: 2,
    QueryCategory.AGGREGATION: 3,
}


@dataclass(frozen=True)
class ChunkRef:
    """Con trỏ nhẹ tới một chunk trong index — chưa có nội dung."""

    chunk_id: str
    doc_id: str
    lang: str
    point_id: str


@dataclass
class ChunkGroup:
    """Một lô chunk sẽ đi vào đúng một lời gọi LLM."""

    category: QueryCategory
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks]

    @property
    def doc_ids(self) -> list[str]:
        return sorted({c.doc_id for c in self.chunks})

    @property
    def lang(self) -> Language:
        langs = {c.metadata.lang for c in self.chunks if c.metadata is not None}
        return next(iter(langs)) if len(langs) == 1 else Language.MIXED


def gutter_ratio(text: str) -> float:
    """Tỉ lệ dòng có khoảng trắng dài ở giữa — dấu vết của bố cục nhiều cột.

    Bản `.txt` mà World Bank trích sẵn giữ nguyên vị trí ký tự của trang PDF hai
    cột, nên cột trái và cột phải bị **đan xen vào nhau theo từng dòng**. Đọc
    liền mạch thì thành thứ như: "ividual indexes on new orders, output,
    minus the number of existing firms temporarily". Câu văn trông vẫn ra câu
    văn nên mọi bộ lọc theo tỉ lệ chữ cái đều cho qua.
    """
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return 0.0
    return sum(1 for line in lines if _GUTTER_RE.search(line)) / len(lines)


def mean_words_per_line(text: str) -> float:
    """Số từ trung bình trên mỗi dòng không rỗng.

    Phân biệt văn xuôi với chú thích biểu đồ và trang bìa: bản `.txt` của World
    Bank ngắt dòng ở khoảng 80 ký tự nên văn xuôi cho ~13–15 từ/dòng, còn chú
    thích trục và danh mục cho 2–5. Cả hai đều qua được bộ lọc tỉ lệ chữ cái.
    """
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return 0.0
    return sum(len(_WORD_RE.findall(line)) for line in lines) / len(lines)


def is_prose_like(
    text: str,
    *,
    min_words: int = 60,
    min_letter_ratio: float = 0.6,
    max_gutter_ratio: float = 0.3,
    min_words_per_line: float = 10.0,
) -> bool:
    """Đúng khi đoạn text trông như văn xuôi đủ để đặt câu hỏi lên nó.

    `min_letter_ratio` loại bảng số và mục lục: một dòng "3.2 ... 45" gần như
    toàn chữ số và dấu chấm. `min_words` loại tiêu đề rời và số trang.
    `max_gutter_ratio` loại phần bị trộn hai cột — xem `gutter_ratio`.

    Đo trên corpus hiện tại: **27,8% chunk** có `gutter_ratio > 0.3`. Đó là con
    số đáng kể, và nó **chỉ ảnh hưởng việc chọn mẫu sinh câu hỏi**, không ảnh
    hưởng index: chunk đó vẫn nằm trong Qdrant và retriever vẫn có quyền trả về.
    Ta chỉ từ chối *đặt câu hỏi* lên chúng, vì câu hỏi sinh từ văn bản trộn cột
    không đo được gì ngoài chính lỗi trích xuất. Docling ở `W3-01` mới là cách
    sửa thật.
    """
    stripped = text.strip()
    if len(_WORD_RE.findall(stripped)) < min_words:
        return False
    visible = [ch for ch in stripped if not ch.isspace()]
    if not visible:
        return False
    letters = sum(1 for ch in visible if _LETTER_RE.match(ch))
    if letters / len(visible) < min_letter_ratio:
        return False
    if gutter_ratio(stripped) > max_gutter_ratio:
        return False
    return mean_words_per_line(stripped) >= min_words_per_line


def load_chunk_refs(retriever: QdrantDenseRetriever, *, page_size: int = 2000) -> list[ChunkRef]:
    """Quét toàn bộ collection, chỉ lấy phần metadata cần để chọn mẫu."""
    refs: list[ChunkRef] = []
    offset: Any = None
    while True:
        points, offset = retriever.client.scroll(
            collection_name=retriever.collection,
            limit=page_size,
            offset=offset,
            with_payload=["chunk_id", "doc_id", "lang"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id")
            if not chunk_id:  # pragma: no cover - point ghi bởi phiên bản cũ
                continue
            refs.append(
                ChunkRef(
                    chunk_id=str(chunk_id),
                    doc_id=str(payload.get("doc_id", "")),
                    lang=str(payload.get("lang", Language.UNKNOWN.value)),
                    point_id=str(point.id),
                )
            )
        if offset is None:
            break
    logger.info("Quét được %d chunk trong `%s`", len(refs), retriever.collection)
    return refs


def _chunk_index(chunk_id: str) -> int:
    """Số thứ tự chunk trong tài liệu, đọc từ `{doc_id}::{index:05d}`."""
    _, _, tail = chunk_id.rpartition("::")
    return int(tail) if tail.isdigit() else 0


def _by_document(refs: Sequence[ChunkRef]) -> dict[str, list[ChunkRef]]:
    grouped: dict[str, list[ChunkRef]] = {}
    for ref in refs:
        grouped.setdefault(ref.doc_id, []).append(ref)
    for items in grouped.values():
        items.sort(key=lambda r: r.chunk_id)
    return grouped


def plan_groups(
    refs: Sequence[ChunkRef],
    quotas: dict[QueryCategory, int],
    *,
    seed: int = 20260817,
    languages: Sequence[str] = (),
    skip_leading_chunks: int = 6,
) -> list[tuple[QueryCategory, list[ChunkRef]]]:
    """Chọn xem mỗi lời gọi LLM sẽ được đưa những chunk nào.

    Rải đều theo tài liệu bằng cách đi vòng tròn qua danh sách tài liệu thay vì
    bốc ngẫu nhiên toàn cục: bốc ngẫu nhiên trên 15.814 chunk thì một tài liệu
    dài chiếm phần lớn mẫu, và golden set hoá ra chỉ đo được vài tài liệu.

    `skip_leading_chunks` bỏ phần đầu mỗi tài liệu. Báo cáo World Bank mở đầu
    bằng bìa, trang bản quyền, lời cảm ơn và mục lục — chúng là văn xuôi hợp lệ
    nên mọi bộ lọc hình thức đều cho qua, nhưng câu hỏi sinh từ đó chỉ đo được
    khả năng tìm lại đoạn "© 2022 International Bank for Reconstruction".

    `seed` cố định để chạy lại cho ra đúng cùng một tập chunk — không có nó thì
    hai lần sinh nháp không so được với nhau, và tiền gọi API đã tiêu là bỏ phí.
    """
    pool = [
        r
        for r in refs
        if (not languages or r.lang in set(languages))
        and _chunk_index(r.chunk_id) >= skip_leading_chunks
    ]
    if not pool:
        raise ValueError(f"Không có chunk nào khớp languages={list(languages)}")

    by_doc = _by_document(pool)
    doc_ids = sorted(by_doc)
    rng = random.Random(seed)
    rng.shuffle(doc_ids)

    used: set[str] = set()
    plan: list[tuple[QueryCategory, list[ChunkRef]]] = []
    doc_cursor = 0

    for category in sorted(quotas, key=lambda c: c.value):
        needed_groups = quotas[category]
        if category is QueryCategory.UNANSWERABLE:
            # Câu unanswerable không gắn với chunk nào theo định nghĩa; prompt
            # của nó vẫn cần chunk để biết corpus nói về chủ đề gì, nhưng chunk
            # đó KHÔNG được ghi vào `relevant_chunk_ids`.
            size = 1
        else:
            size = MULTI_CHUNK_CATEGORIES.get(category, 1)

        made = 0
        attempts = 0
        max_attempts = needed_groups * 8 + len(doc_ids)
        while made < needed_groups and attempts < max_attempts:
            attempts += 1
            doc_id = doc_ids[doc_cursor % len(doc_ids)]
            doc_cursor += 1
            candidates = [r for r in by_doc[doc_id] if r.chunk_id not in used]
            if len(candidates) < size:
                continue
            if size == 1:
                picked = [rng.choice(candidates)]
            else:
                # Chunk liền kề trong cùng tài liệu: multi-hop cần hai mẩu thông
                # tin có liên hệ, hai chunk ngẫu nhiên ở hai đầu tài liệu thường
                # chẳng nối được với nhau thành một câu hỏi có nghĩa.
                start = rng.randrange(0, len(candidates) - size + 1)
                picked = candidates[start : start + size]
            used.update(r.chunk_id for r in picked)
            plan.append((category, picked))
            made += 1

        if made < needed_groups:
            logger.warning(
                "Nhóm %s chỉ dựng được %d/%d lô — corpus không đủ chunk phù hợp",
                category.value,
                made,
                needed_groups,
            )
    return plan


def hydrate(
    retriever: QdrantDenseRetriever,
    plan: Sequence[tuple[QueryCategory, list[ChunkRef]]],
    *,
    min_words: int = 60,
    batch_size: int = 256,
) -> list[ChunkGroup]:
    """Lấy nội dung thật của các chunk đã chọn, bỏ lô nào có chunk rác."""
    point_ids = [ref.point_id for _, refs in plan for ref in refs]
    contents: dict[str, Chunk] = {}
    for start in range(0, len(point_ids), batch_size):
        records = retriever.client.retrieve(
            collection_name=retriever.collection,
            ids=point_ids[start : start + batch_size],
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            payload = record.payload or {}
            raw = payload.get("chunk")
            if raw is None:  # pragma: no cover - point ghi bởi phiên bản cũ
                continue
            chunk = Chunk.model_validate(raw)
            contents[chunk.chunk_id] = chunk

    groups: list[ChunkGroup] = []
    dropped = 0
    for category, refs in plan:
        chunks = [contents[r.chunk_id] for r in refs if r.chunk_id in contents]
        if len(chunks) != len(refs):
            dropped += 1
            continue
        if not all(is_prose_like(c.content, min_words=min_words) for c in chunks):
            dropped += 1
            continue
        groups.append(ChunkGroup(category=category, chunks=chunks))

    if dropped:
        logger.info("Bỏ %d lô vì chunk không phải văn xuôi hoặc thiếu nội dung", dropped)
    return groups


def sample_groups(
    retriever: QdrantDenseRetriever,
    quotas: dict[QueryCategory, int],
    *,
    seed: int = 20260817,
    languages: Sequence[str] = (),
    min_words: int = 60,
    overshoot: float = 3.0,
) -> list[ChunkGroup]:
    """Chọn và nạp nội dung cho các lô chunk.

    Lập kế hoạch dư `overshoot` lần vì bộ lọc văn xuôi sẽ loại bớt — không dư
    thì mỗi nhóm ra ít hơn hạn mức và phân bố category bị lệch.

    Hệ số 3,0 đến từ số đo thật: trên corpus hiện tại bộ lọc loại khoảng 60% lô
    (trộn cột, chú thích biểu đồ, trang bìa). Phần dư chỉ tốn thêm vài lần đọc
    Qdrant, **không** tốn thêm tiền API — lô thừa bị cắt trước khi gọi model.
    """
    refs = load_chunk_refs(retriever)
    inflated = {c: max(1, round(n * overshoot)) for c, n in quotas.items()}
    plan = plan_groups(refs, inflated, seed=seed, languages=languages)
    groups = hydrate(retriever, plan, min_words=min_words)
    return list(_trim_to_quota(groups, quotas))


def _trim_to_quota(
    groups: Sequence[ChunkGroup], quotas: dict[QueryCategory, int]
) -> Iterator[ChunkGroup]:
    remaining = dict(quotas)
    for group in groups:
        if remaining.get(group.category, 0) > 0:
            remaining[group.category] -= 1
            yield group
