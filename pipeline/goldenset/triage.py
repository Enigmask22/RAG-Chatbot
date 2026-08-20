"""Xếp hàng đợi review cho `W1-11`, dùng chính retriever thật làm nguồn tín hiệu.

266 câu nháp, review tay ước lượng 6–8 giờ. Module này không thay người review —
nó chỉ làm hai việc: xếp thứ tự đọc theo rủi ro, và gom sẵn mọi thứ cần để phán
xét một câu vào cùng một chỗ (text chunk đã gán, trích dẫn model viện dẫn, top-k
mà retriever thật trả về). Phần lớn thời gian review tay bị mất vào việc tra cứu
qua lại, không phải vào việc phán xét.

---

## Bất đối xứng phải giữ đúng

Có hai tín hiệu từ retriever, và **chúng không đối xứng**. Lẫn hai cái này là
cách chắc chắn nhất để tự thổi phồng baseline ở `W1-13`.

**(A) Câu `unanswerable` mà retriever tìm được chunk điểm cao** — bằng chứng
*mạnh* rằng nhãn sai. "Corpus không trả lời được câu này" là một mệnh đề về
**corpus**; một hit mạnh phản chứng nó trực tiếp. Model sinh câu hỏi chỉ nhìn
thấy vài chunk trong một tài liệu, nên nó không có cách nào biết 59 tài liệu còn
lại có trả lời được hay không. Đây chính là `TD-09`.

**(B) Câu trả lời được mà retriever *không* tìm ra chunk đã gán** — **không**
phải bằng chứng nhãn sai. Đây đúng là thứ eval tồn tại để đo. Loại những câu này
ra khỏi golden set thì tập còn lại chỉ gồm câu mà hệ thống hiện tại đã trả lời
được, recall baseline bị đẩy lên, và mọi con số "cải thiện +X%" về sau đo trên
một tập đã chọn thiên vị theo đúng hệ thống đang muốn đánh giá.

Nên: (A) được xếp lên đầu hàng đợi kèm đề xuất đổi nhãn. (B) chỉ được ghi lại như
thông tin, xếp **cuối**, và kèm cảnh báo rằng hành động mặc định là *giữ nguyên*.

Ngoại lệ duy nhất của (B): `chunk_id` đã gán **không tồn tại trong index**. Đó
không phải câu khó, đó là con trỏ chết — luôn phải sửa.

## Ngưỡng được hiệu chuẩn, không phải hằng số

"Điểm cao" của cosine similarity phụ thuộc vào model embedding và vào corpus.
Ghim một hằng số kiểu `0.8` là đoán. Thay vào đó ngưỡng được lấy từ **phân bố
điểm top-1 của chính những câu trả lời được trong tập này** (mặc định: trung vị).
Một câu `unanswerable` có điểm top-1 cao hơn ngưỡng đó nghĩa là retriever tự tin
về nó ngang với những câu mà ta biết chắc corpus trả lời được.
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from pipeline.eval.golden import QueryCategory
from pipeline.goldenset.schema import GoldenDraft
from rag_core.retrieval.base import Retriever

__all__ = [
    "DEFAULT_TOP_K",
    "RetrievedBrief",
    "TriageFlag",
    "TriageResult",
    "TriageSummary",
    "review_priority",
    "triage_drafts",
    "write_decisions_template",
    "write_review_queue",
    "write_triage",
]

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 20
"""Đủ sâu để biết chunk đã gán xếp thứ mấy, đủ nông để queue còn đọc được."""

_QUEUE_PREVIEW = 3
"""Số chunk top-k in vào hàng đợi review. Nhiều hơn thì người đọc bỏ qua hết."""


class TriageFlag(StrEnum):
    """Tín hiệu gắn vào một câu nháp. Một câu có thể mang nhiều tín hiệu."""

    GOLD_CHUNK_MISSING = "gold_chunk_missing"
    """`relevant_chunk_ids` trỏ tới chunk không có trong index. Con trỏ chết —
    luôn phải sửa, bất kể câu hỏi tốt hay dở."""

    UNANSWERABLE_BUT_RETRIEVED = "unanswerable_but_retrieved"
    """Nhãn `unanswerable` nhưng retriever tự tin. Tín hiệu (A) — xem lại nhãn."""

    QUOTE_UNVERIFIED = "quote_unverified"
    """Trích dẫn model viện dẫn không tìm thấy trong chunk nó cite. Có thể model
    bịa, có thể chỉ là khác khoảng trắng — phải người đọc mới phân biệt được."""

    ANSWERABLE_BUT_NOT_RETRIEVED = "answerable_but_not_retrieved"
    """Tín hiệu (B). **Không** phải lỗi. Ghi lại vì nó cho biết trước rằng
    baseline sẽ trượt câu này; hành động mặc định là giữ nguyên."""

    TRIVIALLY_EASY = "trivially_easy"
    """Chunk đã gán nằm ngay hạng 1 với điểm rất cao. Không phải lỗi, nhưng nếu
    cả tập toàn loại này thì eval mất khả năng phân biệt giữa hai hệ thống."""


# Thứ tự đọc. Số nhỏ = đọc trước. `GOLD_CHUNK_MISSING` lên đầu vì nó là lỗi dữ
# liệu chắc chắn; `ANSWERABLE_BUT_NOT_RETRIEVED` xuống cuối **có chủ ý** — xem
# docstring module, xếp nó lên cao sẽ dụ người review đi loại đúng những câu khó
# mà eval cần nhất.
_FLAG_PRIORITY: dict[TriageFlag, int] = {
    TriageFlag.GOLD_CHUNK_MISSING: 0,
    TriageFlag.UNANSWERABLE_BUT_RETRIEVED: 1,
    TriageFlag.QUOTE_UNVERIFIED: 2,
    TriageFlag.TRIVIALLY_EASY: 8,
    TriageFlag.ANSWERABLE_BUT_NOT_RETRIEVED: 9,
}

_TRIVIAL_RANK = 1
_TRIVIAL_SCORE_QUANTILE = 0.9
"""Điểm top-1 nằm trong 10% cao nhất của tập → coi là quá dễ."""


class _Pass1(NamedTuple):
    """Kết quả lượt truy hồi, giữ nguyên để lượt 2 gắn cờ.

    Phải là hai lượt: ngưỡng nghi ngờ được hiệu chuẩn từ phân bố điểm của cả tập,
    nên không biết được trước khi truy hồi xong.
    """

    draft: GoldenDraft
    briefs: list[RetrievedBrief]
    gold_hits: list[RetrievedBrief]
    top_score: float | None


class RetrievedBrief(BaseModel):
    """Một kết quả truy hồi, gọn đủ để in vào hàng đợi và lưu vào JSONL."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    doc_id: str
    score: float
    rank: int = Field(ge=1)
    preview: str
    """Đoạn đầu của chunk. Người review cần thấy nội dung, không phải id."""


class TriageResult(BaseModel):
    """Kết quả triage của một câu nháp."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    query: str
    category: QueryCategory
    lang: str
    gold_chunk_ids: list[str] = Field(default_factory=list)
    missing_chunk_ids: list[str] = Field(default_factory=list)
    gold_rank: int | None = None
    """Hạng tốt nhất mà một chunk đã gán đạt được, `None` nếu không có trong top-k."""
    top_score: float | None = None
    gold_score: float | None = None
    quotes_verified: bool = True
    flags: list[TriageFlag] = Field(default_factory=list)
    retrieved: list[RetrievedBrief] = Field(default_factory=list)

    @property
    def suggested_decision(self) -> str:
        """Đề xuất **mặc định**, không phải phán quyết. Người review vẫn quyết."""
        if TriageFlag.GOLD_CHUNK_MISSING in self.flags:
            return "fix_chunk_ids"
        if TriageFlag.UNANSWERABLE_BUT_RETRIEVED in self.flags:
            return "recheck_category"
        if TriageFlag.QUOTE_UNVERIFIED in self.flags:
            return "recheck_quote"
        return "accept"


class TriageSummary(BaseModel):
    """Thống kê cả lượt triage — dùng cho report và cho việc chọn ngưỡng."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total: int
    top_k: int
    answerable: int
    unanswerable: int
    score_threshold: float | None
    """Ngưỡng đã hiệu chuẩn từ tập này, `None` khi không đủ dữ liệu."""
    trivial_threshold: float | None
    flag_counts: dict[str, int] = Field(default_factory=dict)
    by_category: dict[str, int] = Field(default_factory=dict)
    gold_found_in_top_k: int = 0
    """Số câu trả lời được mà chunk đã gán xuất hiện trong top-k. **Không** phải
    recall baseline — golden set chưa đóng băng, và top-k ở đây rộng hơn k của eval."""

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False, indent=2)


def _quantile(values: Sequence[float], q: float) -> float | None:
    """Phân vị theo nội suy tuyến tính. Tự viết để khỏi kéo scipy vào pipeline."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _preview(text: str, limit: int = 220) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit].rstrip() + "…"


def triage_drafts(
    drafts: Sequence[GoldenDraft],
    retriever: Retriever,
    *,
    top_k: int = DEFAULT_TOP_K,
    score_quantile: float = 0.5,
    fetch_chunks: bool = True,
) -> tuple[list[TriageResult], TriageSummary]:
    """Chạy retriever lên mọi câu nháp rồi gắn tín hiệu.

    Hai lượt, và phải là hai lượt: ngưỡng cho câu `unanswerable` được hiệu chuẩn
    từ phân bố điểm của **những câu trả lời được trong cùng tập này**, nên không
    thể biết ngưỡng trước khi truy hồi xong.

    Args:
        drafts: các câu nháp cần triage.
        retriever: retriever thật, đã trỏ vào index đã build.
        top_k: độ sâu truy hồi.
        score_quantile: phân vị của điểm top-1 (trên tập trả lời được) dùng làm
            ngưỡng nghi ngờ cho nhóm `unanswerable`. 0.5 = trung vị.
        fetch_chunks: kiểm xem `relevant_chunk_ids` có thật trong index không.
            Cần `retriever` có `fetch_chunks`; không có thì bỏ qua phép kiểm này.

    Returns:
        Danh sách kết quả (theo đúng thứ tự đầu vào) và bản thống kê.
    """
    pass1: list[_Pass1] = []
    answerable_top_scores: list[float] = []

    for draft in drafts:
        q = draft.query
        hits = retriever.retrieve(q.query, top_k=top_k)
        briefs = [
            RetrievedBrief(
                chunk_id=h.chunk.chunk_id,
                doc_id=h.chunk.doc_id,
                score=h.score,
                rank=h.rank,
                preview=_preview(h.chunk.content),
            )
            for h in hits
        ]
        gold = set(q.relevant_chunk_ids)
        gold_hits = [b for b in briefs if b.chunk_id in gold]
        top_score = briefs[0].score if briefs else None

        if q.category is not QueryCategory.UNANSWERABLE and top_score is not None:
            answerable_top_scores.append(top_score)

        pass1.append(_Pass1(draft, briefs, gold_hits, top_score))

    threshold = _quantile(answerable_top_scores, score_quantile)
    trivial_threshold = _quantile(answerable_top_scores, _TRIVIAL_SCORE_QUANTILE)
    missing_map = _missing_chunk_ids(drafts, retriever) if fetch_chunks else {}

    results: list[TriageResult] = []
    flag_counts: dict[str, int] = {}
    by_category: dict[str, int] = {}
    gold_found = 0

    for draft, briefs, gold_hits, top_score in pass1:
        q = draft.query
        flags: list[TriageFlag] = []
        missing = missing_map.get(q.query_id, [])
        best_gold = min(gold_hits, key=lambda b: b.rank) if gold_hits else None

        if missing:
            flags.append(TriageFlag.GOLD_CHUNK_MISSING)
        if not draft.provenance.quotes_verified:
            flags.append(TriageFlag.QUOTE_UNVERIFIED)

        if q.category is QueryCategory.UNANSWERABLE:
            if threshold is not None and top_score is not None and top_score >= threshold:
                flags.append(TriageFlag.UNANSWERABLE_BUT_RETRIEVED)
        elif q.relevant_chunk_ids:
            if best_gold is None:
                flags.append(TriageFlag.ANSWERABLE_BUT_NOT_RETRIEVED)
            else:
                gold_found += 1
                if (
                    best_gold.rank == _TRIVIAL_RANK
                    and trivial_threshold is not None
                    and best_gold.score >= trivial_threshold
                ):
                    flags.append(TriageFlag.TRIVIALLY_EASY)

        result = TriageResult(
            query_id=q.query_id,
            query=q.query,
            category=q.category,
            lang=str(q.lang),
            gold_chunk_ids=list(q.relevant_chunk_ids),
            missing_chunk_ids=missing,
            gold_rank=best_gold.rank if best_gold else None,
            top_score=top_score,
            gold_score=best_gold.score if best_gold else None,
            quotes_verified=draft.provenance.quotes_verified,
            flags=flags,
            retrieved=briefs,
        )
        results.append(result)

        by_category[q.category.value] = by_category.get(q.category.value, 0) + 1
        for flag in flags:
            flag_counts[flag.value] = flag_counts.get(flag.value, 0) + 1

    n_unanswerable = sum(1 for d in drafts if d.query.category is QueryCategory.UNANSWERABLE)
    summary = TriageSummary(
        total=len(drafts),
        top_k=top_k,
        answerable=len(drafts) - n_unanswerable,
        unanswerable=n_unanswerable,
        score_threshold=threshold,
        trivial_threshold=trivial_threshold,
        flag_counts=flag_counts,
        by_category=by_category,
        gold_found_in_top_k=gold_found,
    )
    return results, summary


def _missing_chunk_ids(drafts: Sequence[GoldenDraft], retriever: Retriever) -> dict[str, list[str]]:
    """Với mỗi câu, những `relevant_chunk_ids` không tồn tại trong index.

    Gọi `fetch_chunks` một lần cho toàn bộ id thay vì mỗi câu một lần: 266 câu
    trỏ tới ~500 chunk nhưng chỉ vài trăm id phân biệt, và một round-trip Qdrant
    đắt hơn nhiều so với việc dựng cái set.
    """
    fetch = getattr(retriever, "fetch_chunks", None)
    if fetch is None:
        logger.warning(
            "Retriever %s không có `fetch_chunks` — bỏ qua phép kiểm chunk_id chết.",
            type(retriever).__name__,
        )
        return {}

    wanted = sorted({cid for d in drafts for cid in d.query.relevant_chunk_ids})
    if not wanted:
        return {}
    found = set(fetch(wanted))
    out: dict[str, list[str]] = {}
    for d in drafts:
        gone = [cid for cid in d.query.relevant_chunk_ids if cid not in found]
        if gone:
            out[d.query.query_id] = gone
    return out


def review_priority(result: TriageResult) -> tuple[int, int, str]:
    """Khoá sắp xếp hàng đợi: (rủi ro cao nhất, số tín hiệu giảm dần, query_id).

    `query_id` ở cuối để thứ tự **xác định** — hai lần chạy cho cùng một file, nếu
    không thì diff của queue toàn nhiễu và không ai đọc nữa.
    """
    best = min((_FLAG_PRIORITY[f] for f in result.flags), default=5)
    return (best, -len(result.flags), result.query_id)


def write_triage(path: str | Path, results: Sequence[TriageResult]) -> int:
    """Ghi kết quả triage đầy đủ ra JSONL (máy đọc)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        for r in results:
            fh.write(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return len(results)


def write_review_queue(
    path: str | Path,
    results: Sequence[TriageResult],
    summary: TriageSummary,
    *,
    gold_texts: dict[str, str] | None = None,
    drafts: Sequence[GoldenDraft] | None = None,
) -> int:
    """Ghi hàng đợi review dạng Markdown (người đọc).

    Cố ý tách khỏi `write_decisions_template`: file này tối ưu cho việc **đọc**,
    file kia tối ưu cho việc **ghi**. Nhồi cả hai vào một file JSONL thì người
    review phải sửa JSON bằng tay trên 266 dòng — vừa chậm vừa dễ làm hỏng file.
    """
    by_id = {d.query.query_id: d for d in (drafts or [])}
    ordered = sorted(results, key=review_priority)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Hàng đợi review golden set (`W1-11`)",
        "",
        f"> {summary.total} câu · top-k {summary.top_k} · "
        f"ngưỡng nghi ngờ (hiệu chuẩn) {_fmt(summary.score_threshold)}",
        "",
        "Thứ tự đọc đã xếp theo rủi ro. Ghi quyết định vào "
        "`decisions_v1.csv`, **không** sửa file này.",
        "",
        "## Đọc tín hiệu thế nào",
        "",
        "| Tín hiệu | Nghĩa | Hành động mặc định |",
        "|---|---|---|",
        "| `gold_chunk_missing` | `chunk_id` không có trong index — con trỏ chết "
        "| `fix_chunk_ids`: chọn lại từ danh sách top-k bên dưới |",
        "| `unanswerable_but_retrieved` | Nhãn nói corpus không trả lời được, "
        "nhưng retriever tự tin ngang mức những câu trả lời được "
        "| `recheck_category`: đọc chunk top-1, nếu nó trả lời được thì đổi nhãn |",
        "| `quote_unverified` | Trích dẫn model viện dẫn không khớp text chunk "
        "| `recheck_quote`: model bịa thì `reject`, chỉ khác khoảng trắng thì `accept` |",
        "| `trivially_easy` | Chunk đã gán ở hạng 1, điểm rất cao | `accept`. "
        "Chỉ đáng lo ở mức phân bố cả tập, không ở mức từng câu |",
        "| `answerable_but_not_retrieved` | Retriever không tìm ra chunk đã gán "
        "| **`accept`.** Đây là câu khó, không phải câu sai — xem cảnh báo dưới |",
        "",
        "> ⚠️ **`answerable_but_not_retrieved` không phải lý do để loại câu hỏi.**",
        "> Đó đúng là thứ eval tồn tại để đo. Loại chúng đi thì golden set chỉ còn",
        "> câu mà hệ thống hiện tại đã trả lời được, recall baseline bị đẩy lên, và",
        "> mọi con số cải thiện về sau đo trên một tập đã chọn thiên vị.",
        "> Chỉ `reject` khi bản thân **câu hỏi** dở (tối nghĩa, sai sự thật, trùng ý).",
        "",
        "---",
        "",
    ]

    for i, r in enumerate(ordered, start=1):
        flags = " ".join(f"`{f.value}`" for f in r.flags) or "—"
        lines += [
            f"## {i}. `{r.query_id}` · {r.category.value} · {r.lang}",
            "",
            f"**Đề xuất:** `{r.suggested_decision}` · **Tín hiệu:** {flags}",
            "",
            f"> {r.query}",
            "",
            f"- điểm top-1: {_fmt(r.top_score)} · hạng của chunk đã gán: "
            f"{r.gold_rank if r.gold_rank else '**không có trong top-k**'}"
            f" (điểm {_fmt(r.gold_score)})",
        ]
        draft = by_id.get(r.query_id)
        if draft is not None and draft.query.reference_answer:
            lines.append(f"- đáp án tham chiếu: {_preview(draft.query.reference_answer, 300)}")
        if draft is not None and draft.provenance.supporting_quotes:
            mark = "✅" if r.quotes_verified else "⚠️ KHÔNG khớp chunk"
            quote = _preview(draft.provenance.supporting_quotes[0], 300)
            lines.append(f"- trích dẫn model viện dẫn ({mark}): {quote}")
        if r.missing_chunk_ids:
            lines.append(f"- ⚠️ chunk_id không có trong index: `{'`, `'.join(r.missing_chunk_ids)}`")
        lines.append("")

        if r.gold_chunk_ids:
            lines += ["**Chunk đã gán:**", ""]
            for cid in r.gold_chunk_ids:
                text = (gold_texts or {}).get(cid)
                body = _preview(text, 400) if text else "*(không lấy được text)*"
                lines.append(f"- `{cid}` — {body}")
            lines.append("")

        lines += [f"**Top-{min(_QUEUE_PREVIEW, len(r.retrieved))} retriever trả về:**", ""]
        for b in r.retrieved[:_QUEUE_PREVIEW]:
            mark = " ⬅️ đã gán" if b.chunk_id in set(r.gold_chunk_ids) else ""
            lines.append(f"{b.rank}. `{b.chunk_id}` ({b.score:.4f}){mark} — {b.preview}")
        lines += ["", "---", ""]

    target.write_text("\n".join(lines), encoding="utf-8")
    return len(ordered)


_DECISION_FIELDS = (
    "query_id",
    "category",
    "flags",
    "suggested_decision",
    "decision",
    "new_category",
    "new_relevant_chunk_ids",
    "notes",
    "query",
)


def write_decisions_template(path: str | Path, results: Sequence[TriageResult]) -> int:
    """Ghi CSV để người review điền quyết định.

    Không ghi đè nếu file đã tồn tại — mất 6 giờ công review vì một lần chạy lại
    lệnh là chuyện không được phép xảy ra.

    Raises:
        FileExistsError: file đã có. Xoá hoặc đổi tên bằng tay nếu thật sự muốn.
    """
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            f"{target} đã tồn tại. Không ghi đè để khỏi mất quyết định đã điền — "
            "đổi tên file cũ nếu thật sự muốn bắt đầu lại."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(results, key=review_priority)
    with target.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_DECISION_FIELDS))
        writer.writeheader()
        for r in ordered:
            writer.writerow(
                {
                    "query_id": r.query_id,
                    "category": r.category.value,
                    "flags": " ".join(f.value for f in r.flags),
                    "suggested_decision": r.suggested_decision,
                    "decision": "",
                    "new_category": "",
                    "new_relevant_chunk_ids": "",
                    "notes": "",
                    "query": r.query,
                }
            )
    return len(ordered)


def _fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "—"


def log_summary(summary: TriageSummary, results: Iterable[TriageResult]) -> None:
    """In thống kê ra log, kèm phần dễ đọc sai được nói rõ."""
    log = logging.getLogger("triage")
    log.info(
        "Triage %d câu (top-k %d) · ngưỡng nghi ngờ %s · ngưỡng 'quá dễ' %s",
        summary.total,
        summary.top_k,
        _fmt(summary.score_threshold),
        _fmt(summary.trivial_threshold),
    )
    for flag, count in sorted(summary.flag_counts.items(), key=lambda kv: -kv[1]):
        log.info("  %-32s %d", flag, count)
    if summary.answerable:
        pct = 100.0 * summary.gold_found_in_top_k / summary.answerable
        log.info(
            "  chunk đã gán có trong top-%d: %d/%d (%.1f%%) — KHÔNG phải recall baseline",
            summary.top_k,
            summary.gold_found_in_top_k,
            summary.answerable,
            pct,
        )
    n_median = statistics.median([len(r.flags) for r in results]) if summary.total else 0
    log.info("  trung vị số tín hiệu mỗi câu: %s", n_median)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: chạy retriever thật lên tập nháp rồi sinh hàng đợi review."""
    import argparse

    from pipeline.goldenset.schema import load_drafts
    from pipeline.indexing.config import load_index_config
    from rag_core.settings import get_settings

    parser = argparse.ArgumentParser(
        description="Triage tập nháp golden set bằng retriever thật (W1-11).",
    )
    parser.add_argument("--drafts", type=Path, default=Path("data/golden/draft_v1.jsonl"))
    parser.add_argument("--index-config", type=Path, default=Path("configs/indexing/baseline.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/golden/review"))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--score-quantile",
        type=float,
        default=0.5,
        help="Phân vị điểm top-1 của câu trả lời được, dùng làm ngưỡng nghi ngờ "
        "cho nhóm unanswerable. Cao hơn = ít cảnh báo hơn.",
    )
    parser.add_argument(
        "--force-decisions",
        action="store_true",
        help="Ghi đè `decisions_*.csv` đang có. MẤT quyết định đã điền.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    drafts = load_drafts(args.drafts)
    if not drafts:
        logger.error("%s không có câu nào.", args.drafts)
        return 1
    logger.info("Đọc %d câu nháp từ %s", len(drafts), args.drafts)

    settings = get_settings()
    index_config = load_index_config(args.index_config)
    retriever = index_config.build_retriever(
        index_config.build_embeddings(),
        url=settings.qdrant_url,
        api_key=(settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None),
    )

    # Warm-up: lần truy hồi đầu phải trả tiền nạp model sentence-transformers.
    # Không hâm nóng thì con số đầu tiên lệch hàng chục giây — đúng cái bẫy đã
    # làm p95 của `W1-08` đọc thành 15.219 ms.
    retriever.retrieve("warmup", top_k=1)

    results, summary = triage_drafts(
        drafts,
        retriever,
        top_k=args.top_k,
        score_quantile=args.score_quantile,
    )
    log_summary(summary, results)

    gold_texts: dict[str, str] = {}
    fetch = getattr(retriever, "fetch_chunks", None)
    if fetch is not None:
        wanted = sorted({cid for d in drafts for cid in d.query.relevant_chunk_ids})
        gold_texts = {cid: chunk.content for cid, chunk in fetch(wanted).items()}

    out: Path = args.out_dir
    write_triage(out / "triage_v1.jsonl", results)
    write_review_queue(out / "queue_v1.md", results, summary, gold_texts=gold_texts, drafts=drafts)
    (out / "triage_summary.json").write_text(summary.to_json(), encoding="utf-8")

    decisions = out / "decisions_v1.csv"
    if args.force_decisions and decisions.exists():
        decisions.unlink()
    try:
        n = write_decisions_template(decisions, results)
        logger.info("Đã tạo %s với %d dòng chờ điền", decisions, n)
    except FileExistsError as exc:
        logger.warning("%s", exc)

    logger.info("Xong. Đọc %s, điền %s", out / "queue_v1.md", decisions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
