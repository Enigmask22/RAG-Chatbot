"""Ánh xạ nhãn golden set từ **vùng ký tự** sang `chunk_id` của index đang được đo.

Đây là mảnh làm cho golden set độc lập với cấu hình chunking (`TD-12`). Nhãn được
neo vào văn bản gốc — thứ bất biến, có sha256 trong manifest và được kiểm lại mỗi
lần build index. Còn `chunk_id` thì thuộc về một index cụ thể, và được tính lại ở
mỗi lần eval.

## Quy tắc: chồng nhau **đáng kể**, xét theo cả hai phía

Một chunk được tính là liên quan tới một đoạn bằng chứng khi:

    overlap / span.length >= ratio   HOẶC   overlap / chunk.length >= ratio

mặc định `ratio = 0.5`.

Phải là **hoặc**, không phải chỉ điều kiện đầu. Đo thật trên 226 câu đã neo:
với `chunk_size=400`, mọi span rộng bằng cả chunk cũ (~1000 ký tự) đều mất nhãn
— **40/226 câu không khớp chunk nào** — vì không chunk 400 ký tự nào chứa nổi
500 ký tự. Mà đó đúng là kịch bản `W2` sẽ chạy (`TD-11` nói phải hạ `chunk_size`).
Điều kiện thứ hai xử lý đúng trường hợp đó: một chunk nằm **trọn trong** vùng
bằng chứng thì rõ ràng là liên quan, dù nó chỉ chiếm một phần nhỏ của vùng.

Vì sao không nới thành "chồng nhau tí nào cũng tính": chunk chỉ liếm qua vài ký
tự ở biên vùng bằng chứng gần như chắc chắn không chứa câu trả lời, và tính nó là
liên quan sẽ thổi phồng recall.

Vì sao ngưỡng vẫn an toàn khi span rất rộng: span luôn bị chặn bởi
`max_chunk_size` (1500 ký tự ở baseline) vì nó được dẫn ra từ một chunk. Không có
span nào rộng bằng cả tài liệu để làm mọi chunk thành "liên quan".

## Nhiều span thay vì một span rộng

Câu `aggregation` cần ba đoạn rời nhau thì `relevant_spans` có ba phần tử. Gộp
thành một span từ đầu đoạn một tới cuối đoạn ba sẽ làm mọi chunk nằm giữa — kể
cả chunk không liên quan gì — vượt ngưỡng 0,5 và được tính là đúng.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline.eval.golden import GoldenQuery
from rag_core.schemas import Chunk, TextSpan

__all__ = [
    "DEFAULT_MIN_OVERLAP_RATIO",
    "QueryResolution",
    "SpanResolution",
    "chunks_by_document",
    "resolve_queries",
    "resolve_spans",
]

logger = logging.getLogger(__name__)

DEFAULT_MIN_OVERLAP_RATIO = 0.5
"""Phần đoạn bằng chứng phải nằm trong chunk để chunk đó được tính là liên quan."""


class SpanResolution:
    """Kết quả ánh xạ, kèm phần đủ để biết vì sao ra thế.

    Không dùng pydantic: đây là kết quả tính toán trong bộ nhớ, không đi qua ranh
    giới nào cần validate, và `Chunk` bên trong đã được validate rồi.
    """

    __slots__ = ("chunk_ids", "per_span", "unmatched")

    def __init__(
        self,
        chunk_ids: list[str],
        unmatched: list[TextSpan],
        per_span: dict[tuple[str, int, int], list[str]],
    ) -> None:
        self.chunk_ids = chunk_ids
        self.unmatched = unmatched
        """Span không khớp chunk nào. **Luôn phải xem xét**: nghĩa là bằng chứng
        đã bị cấu hình chunking hiện tại cắt vụn tới mức không chunk nào chứa nổi
        một nửa nó. Câu đó sẽ bị chấm 0 recall và lý do không phải là retrieval."""
        self.per_span = per_span

    def __repr__(self) -> str:  # pragma: no cover - chỉ để debug
        return f"SpanResolution({len(self.chunk_ids)} chunk, {len(self.unmatched)} span không khớp)"


def chunks_by_document(chunks: Iterable[Chunk]) -> dict[str, list[Chunk]]:
    """Nhóm chunk theo `doc_id`, bỏ chunk không có span.

    Chunk thiếu span là point ghi trước `W1-11`. Bỏ qua **và cảnh báo**, chứ không
    im lặng: nếu cả index đều thiếu span thì mọi span sẽ không khớp gì, và không
    có dòng log nào thì hiện tượng đó đọc lên giống hệt "retrieval quá tệ".
    """
    out: dict[str, list[Chunk]] = {}
    n_missing = 0
    for chunk in chunks:
        if chunk.start_char is None or chunk.end_char is None:
            n_missing += 1
            continue
        out.setdefault(chunk.doc_id, []).append(chunk)
    if n_missing:
        logger.warning(
            "%d chunk không có offset ký tự — build lại index bằng `make index BUNDLE=... "
            "--recreate` để ánh xạ span hoạt động",
            n_missing,
        )
    for chunk_list in out.values():
        chunk_list.sort(key=lambda c: (c.start_char or 0, c.chunk_index))
    return out


def resolve_spans(
    spans: Sequence[TextSpan],
    by_doc: dict[str, list[Chunk]],
    *,
    min_overlap_ratio: float = DEFAULT_MIN_OVERLAP_RATIO,
) -> SpanResolution:
    """Tìm những `chunk_id` chứa phần lớn mỗi đoạn bằng chứng.

    Args:
        spans: các đoạn bằng chứng của một câu hỏi.
        by_doc: chunk của index đang đo, nhóm theo `doc_id` (xem `chunks_by_document`).
        min_overlap_ratio: phần của **span** phải nằm trong chunk.

    Returns:
        `SpanResolution`. `chunk_ids` giữ thứ tự xuất hiện và đã khử trùng lặp —
        thứ tự xác định để hai lần chạy cho cùng kết quả, và metric xếp hạng phía
        sau không được phụ thuộc vào thứ tự lặp của một `set`.
    """
    if not 0.0 < min_overlap_ratio <= 1.0:
        raise ValueError(f"min_overlap_ratio phải trong (0, 1], nhận {min_overlap_ratio}")

    ordered: list[str] = []
    seen: set[str] = set()
    unmatched: list[TextSpan] = []
    per_span: dict[tuple[str, int, int], list[str]] = {}

    for span in spans:
        matched: list[str] = []
        for chunk in by_doc.get(span.doc_id, ()):
            chunk_span = chunk.span
            if chunk_span is None:  # pragma: no cover - đã lọc ở chunks_by_document
                continue
            overlap = span.overlap(chunk_span)
            if not overlap:
                continue
            if (
                overlap / span.length >= min_overlap_ratio
                or overlap / chunk_span.length >= min_overlap_ratio
            ):
                matched.append(chunk.chunk_id)
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    ordered.append(chunk.chunk_id)
        per_span[(span.doc_id, span.start, span.end)] = matched
        if not matched:
            unmatched.append(span)

    return SpanResolution(ordered, unmatched, per_span)


class QueryResolution(BaseModel):
    """Thống kê của một lượt ánh xạ cả golden set — đi vào `EvalReport.config`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    resolved: int
    """Số câu có nhãn được tính lại từ span."""
    kept_chunk_ids: int
    """Số câu không có span, giữ nguyên `relevant_chunk_ids` sẵn có."""
    unmatched_queries: list[str] = Field(default_factory=list)
    """Câu có span nhưng **không** span nào khớp chunk nào của index này.

    Phải nhìn vào con số này trước khi đọc recall: nó nghĩa là cấu hình chunking
    hiện tại cắt bằng chứng vụn tới mức không chunk nào chứa nổi, nên câu bị chấm
    0 vì lý do **không phải** retrieval.
    """
    min_overlap_ratio: float
    label_changed: int
    """Số câu mà nhãn tính từ span khác tập `relevant_chunk_ids` ghi trong file.

    Với đúng cấu hình đã gán nhãn thì con số này phải gần 0. Lớn nghĩa là index
    đang đo được chunk khác lúc gán nhãn — đó là chuyện bình thường (và chính là
    lý do span tồn tại), nhưng phải hiện ra trong report chứ không được ẩn đi.
    """


def resolve_queries(
    queries: Sequence[GoldenQuery],
    by_doc: dict[str, list[Chunk]],
    *,
    min_overlap_ratio: float = DEFAULT_MIN_OVERLAP_RATIO,
) -> tuple[list[GoldenQuery], QueryResolution]:
    """Tính lại `relevant_chunk_ids` từ `relevant_spans` cho index đang đo.

    Câu không có span thì giữ nguyên nhãn cũ — golden set có thể trộn hai kiểu
    trong giai đoạn chuyển tiếp, và im lặng bỏ những câu chưa neo sẽ làm tập đo
    nhỏ đi mà không ai biết.

    Trả về `GoldenQuery` mới thay vì sửa tại chỗ: `GoldenQuery` là `frozen`, và
    quan trọng hơn, giữ bản gốc nguyên vẹn để so được nhãn cũ với nhãn mới.
    """
    out: list[GoldenQuery] = []
    unmatched: list[str] = []
    n_resolved = n_kept = n_changed = 0

    for query in queries:
        if not query.relevant_spans:
            out.append(query)
            n_kept += 1
            continue

        res = resolve_spans(query.relevant_spans, by_doc, min_overlap_ratio=min_overlap_ratio)
        n_resolved += 1
        if res.unmatched and not res.chunk_ids:
            unmatched.append(query.query_id)
        if set(res.chunk_ids) != set(query.relevant_chunk_ids):
            n_changed += 1
        # `relevant_chunk_ids` rỗng sẽ bị `evaluate_run` bỏ qua như câu
        # unanswerable, nên khi không khớp gì thì giữ nhãn cũ và để
        # `unmatched_queries` nói rõ — im lặng biến câu khó thành câu bị loại là
        # đúng kiểu thổi phồng recall mà cả W1-11 đã tránh.
        new_ids = res.chunk_ids or list(query.relevant_chunk_ids)
        out.append(query.model_copy(update={"relevant_chunk_ids": new_ids}))

    return out, QueryResolution(
        resolved=n_resolved,
        kept_chunk_ids=n_kept,
        unmatched_queries=unmatched,
        min_overlap_ratio=min_overlap_ratio,
        label_changed=n_changed,
    )
