"""Golden set — tập truy vấn có nhãn, dùng làm chuẩn cho mọi lần đo.

Đây là tài sản giá trị nhất của cả dự án và cũng là thứ tốn công người nhất. Vì
vậy schema chặt ngay từ đầu: sai một nhãn thì mọi con số phía sau đều lệch mà
không có cách nào phát hiện.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_core.schemas import Language, TextSpan

__all__ = ["GoldenQuery", "QueryCategory", "load_golden_set", "write_golden_set"]


class QueryCategory(StrEnum):
    """Bảy nhóm truy vấn. Breakdown theo nhóm chính là đầu ra hữu ích của eval —
    một con số tổng che mất việc hệ thống giỏi factoid nhưng hỏng hoàn toàn ở
    multi-hop."""

    FACTOID = "factoid"
    MULTI_HOP = "multi_hop"
    AGGREGATION = "aggregation"
    TABLE_LOOKUP = "table_lookup"
    CROSS_LINGUAL = "cross_lingual"
    UNANSWERABLE = "unanswerable"
    ADVERSARIAL = "adversarial"


class GoldenQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    category: QueryCategory
    lang: Language = Language.UNKNOWN
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    """Nhãn theo `chunk_id` — **chỉ đúng với đúng một cấu hình chunking**.

    Giữ lại vì tập nháp `W1-10` sinh ra theo dạng này và vì eval chạy nhanh hơn
    khi không phải ánh xạ. Nhưng `relevant_spans` mới là nhãn bền: `chunk_id` của
    dự án là `f"{doc_id}::{index:05d}"`, thuần vị trí, nên đổi `chunk_size` là mọi
    id trỏ vào văn bản khác **mà vẫn tồn tại** → không phép kiểm nào cảnh báo
    (`TD-12`).
    """
    relevant_spans: list[TextSpan] = Field(default_factory=list)
    """Nhãn theo vùng ký tự trong văn bản gốc — bền qua mọi cấu hình chunking.

    Một span cho mỗi mảnh bằng chứng, **không** phải một span lớn phủ hết. Câu
    `aggregation` cần ba đoạn rời nhau thì có ba span; gộp thành một span rộng sẽ
    làm mọi chunk nằm giữa cũng thành "liên quan".
    """
    reference_answer: str | None = None
    notes: str | None = None
    reviewed_by_human: bool = False

    @model_validator(mode="after")
    def _check_relevance(self) -> GoldenQuery:
        if self.category is QueryCategory.UNANSWERABLE:
            if self.relevant_chunk_ids or self.relevant_spans:
                raise ValueError(
                    f"{self.query_id}: câu unanswerable không được có relevant_chunk_ids "
                    "hay relevant_spans — nếu có thì nó trả lời được, phân loại sai"
                )
        elif not self.relevant_chunk_ids and not self.relevant_spans:
            raise ValueError(
                f"{self.query_id}: nhóm {self.category.value} bắt buộc phải có "
                "ít nhất một relevant_chunk_id hoặc relevant_span"
            )
        if len(set(self.relevant_chunk_ids)) != len(self.relevant_chunk_ids):
            raise ValueError(f"{self.query_id}: relevant_chunk_ids bị trùng")
        seen = {(s.doc_id, s.start, s.end) for s in self.relevant_spans}
        if len(seen) != len(self.relevant_spans):
            raise ValueError(f"{self.query_id}: relevant_spans bị trùng")
        return self


def load_golden_set(path: str | Path) -> list[GoldenQuery]:
    """Đọc file JSONL. Lỗi ở dòng nào thì báo rõ số dòng đó."""
    queries: list[GoldenQuery] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                query = GoldenQuery.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"{path}:{line_no} — {exc}") from exc
            if query.query_id in seen:
                raise ValueError(f"{path}:{line_no} — query_id trùng: {query.query_id}")
            seen.add(query.query_id)
            queries.append(query)
    return queries


def write_golden_set(path: str | Path, queries: Sequence[GoldenQuery]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for query in queries:
            handle.write(query.model_dump_json() + "\n")


def iter_categories(queries: Sequence[GoldenQuery]) -> Iterator[tuple[QueryCategory, int]]:
    counts: dict[QueryCategory, int] = {}
    for query in queries:
        counts[query.category] = counts.get(query.category, 0) + 1
    yield from sorted(counts.items(), key=lambda kv: kv[0].value)


def category_distribution(queries: Sequence[GoldenQuery]) -> dict[str, int]:
    return {category.value: count for category, count in iter_categories(queries)}


def json_summary(queries: Sequence[GoldenQuery]) -> str:
    return json.dumps(
        {
            "n_queries": len(queries),
            "categories": category_distribution(queries),
            "reviewed": sum(1 for q in queries if q.reviewed_by_human),
        },
        ensure_ascii=False,
        indent=2,
    )
