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

from rag_core.schemas import Language

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
    reference_answer: str | None = None
    notes: str | None = None
    reviewed_by_human: bool = False

    @model_validator(mode="after")
    def _check_relevance(self) -> GoldenQuery:
        if self.category is QueryCategory.UNANSWERABLE:
            if self.relevant_chunk_ids:
                raise ValueError(
                    f"{self.query_id}: câu unanswerable không được có relevant_chunk_ids — "
                    "nếu có thì nó trả lời được, phân loại sai"
                )
        elif not self.relevant_chunk_ids:
            raise ValueError(
                f"{self.query_id}: nhóm {self.category.value} bắt buộc phải có "
                "ít nhất một relevant_chunk_id"
            )
        if len(set(self.relevant_chunk_ids)) != len(self.relevant_chunk_ids):
            raise ValueError(f"{self.query_id}: relevant_chunk_ids bị trùng")
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
