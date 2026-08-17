"""Bản nháp golden set — `GoldenQuery` cộng phần xuất xứ của lần sinh.

Vì sao draft là một schema riêng chứ không dùng thẳng `GoldenQuery`:

* `GoldenQuery` là **hợp đồng của tập đã đóng băng**. Nó cố ý không có chỗ cho
  "model nào sinh ra câu này", "trích dẫn có kiểm chứng được không" — những thứ
  chỉ cần trong lúc review.
* Bước review tay ở `W1-11` là bước **rút gọn**: draft → `GoldenQuery`. Tách hai
  schema làm cho việc đó là một phép chiếu tường minh, thay vì xoá field lung tung.
* Draft giữ `supporting_quote` đã đối chiếu với chunk. Câu nào trích dẫn **không**
  tìm thấy trong chunk là câu model bịa — người review nên xem những câu đó trước.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from pipeline.eval.golden import GoldenQuery, QueryCategory

__all__ = [
    "DraftProvenance",
    "GoldenDraft",
    "load_drafts",
    "normalize_for_dedupe",
    "write_drafts",
]

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+", re.UNICODE)


def normalize_for_dedupe(text: str) -> str:
    """Hạ chữ, bỏ dấu câu, gom khoảng trắng — dạng chuẩn để so hai câu hỏi."""
    return _SPACE_RE.sub(" ", _PUNCT_RE.sub(" ", text.lower())).strip()


class DraftProvenance(BaseModel):
    """Mọi thứ cần để trả lời 'câu này từ đâu ra' hai tháng sau."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    generator_model: str = Field(min_length=1)
    """Model **thực tế** đã sinh, đọc từ response — không phải model đã yêu cầu."""

    generator_model_requested: str = Field(min_length=1)
    category_requested: QueryCategory
    """Nhóm đã yêu cầu model sinh. Lệch với `query.category` là tín hiệu model
    không làm đúng việc được giao — người review cần biết."""

    source_chunk_ids: list[str] = Field(default_factory=list)
    """Toàn bộ chunk đã đưa vào prompt, kể cả chunk model không dùng."""

    supporting_quotes: list[str] = Field(default_factory=list)
    quotes_verified: bool = False
    """`True` khi mọi trích dẫn đều tìm thấy trong chunk tương ứng."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    generated_at: str = ""
    batch_id: str = ""


class GoldenDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: GoldenQuery
    provenance: DraftProvenance

    @property
    def category_drifted(self) -> bool:
        return self.query.category is not self.provenance.category_requested

    @property
    def needs_close_review(self) -> bool:
        """Câu nên đọc kỹ trước: trích dẫn không kiểm chứng được hoặc lệch nhóm.

        Không tự loại chúng đi — model bịa trích dẫn mà câu hỏi vẫn dùng được là
        chuyện thường. Chỉ xếp lên đầu hàng đợi review.
        """
        return not self.provenance.quotes_verified or self.category_drifted

    @property
    def dedupe_key(self) -> str:
        return normalize_for_dedupe(self.query.query)


def write_drafts(path: str | Path, drafts: Sequence[GoldenDraft]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for draft in drafts:
            handle.write(draft.model_dump_json() + "\n")
    return len(drafts)


def load_drafts(path: str | Path) -> list[GoldenDraft]:
    """Đọc JSONL. Dòng nào hỏng thì báo kèm số dòng, không bỏ qua im lặng."""
    source = Path(path)
    if not source.exists():
        return []
    drafts: list[GoldenDraft] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                drafts.append(GoldenDraft.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{source}:{line_no} — {exc}") from exc
    return drafts


def drafts_summary(drafts: Iterable[GoldenDraft]) -> dict[str, object]:
    items = list(drafts)
    by_category: dict[str, int] = {}
    by_lang: dict[str, int] = {}
    for draft in items:
        by_category[draft.query.category.value] = by_category.get(draft.query.category.value, 0) + 1
        by_lang[draft.query.lang.value] = by_lang.get(draft.query.lang.value, 0) + 1
    return {
        "n_drafts": len(items),
        "by_category": dict(sorted(by_category.items())),
        "by_language": dict(sorted(by_lang.items())),
        "needs_close_review": sum(1 for d in items if d.needs_close_review),
        "quotes_unverified": sum(1 for d in items if not d.provenance.quotes_verified),
        "category_drifted": sum(1 for d in items if d.category_drifted),
        "total_cost_usd": round(sum(d.provenance.cost_usd for d in items), 6),
        "total_tokens": sum(
            d.provenance.prompt_tokens + d.provenance.completion_tokens for d in items
        ),
    }


def summary_json(drafts: Iterable[GoldenDraft]) -> str:
    return json.dumps(drafts_summary(drafts), ensure_ascii=False, indent=2)
