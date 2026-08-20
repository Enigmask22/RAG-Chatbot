"""Đóng băng golden set: nháp + quyết định của người → `golden_v1.jsonl`.

`W1-11` là bước **rút gọn** có chủ ý: `GoldenDraft` (có xuất xứ, có trích dẫn, có
cờ nghi vấn) → `GoldenQuery` (chỉ còn hợp đồng của tập đã đóng băng). Module này
thực hiện phép chiếu đó và từ chối làm nếu dữ liệu chưa đủ điều kiện.

## Vì sao "quyết định" phải là một file riêng

Người review đọc `queue_v1.md` (tối ưu cho đọc) và ghi vào `decisions_v1.csv`
(tối ưu cho ghi). Không gộp thành một file JSONL cho người sửa tay: 266 dòng JSON
sửa bằng tay là vừa chậm vừa dễ làm hỏng cả file, và một dấu ngoặc lệch làm mất
toàn bộ công review.

## Ba từ vựng khác nhau, không được lẫn

* `suggested_decision` (triage sinh ra): `accept` · `fix_chunk_ids` ·
  `recheck_category` · `recheck_quote`. Đây là **câu hỏi đặt cho người review**.
* `decision` (người điền): `accept` · `reject` · `edit`. Đây là **câu trả lời**.
* Ô `decision` để trống = chưa review. Không được coi là `accept`.

Cố ý không cho `recheck_*` làm giá trị của `decision`: nếu một cờ nghi vấn có thể
tự trở thành quyết định thì cả cơ chế triage chỉ còn là trang trí.

## Không tự động sửa gì

Freeze không đoán hộ. `fix_chunk_ids` mà người review không điền
`new_relevant_chunk_ids` thì báo lỗi, chứ không lấy top-1 của retriever điền vào.
Lấy top-1 làm nhãn nghĩa là dạy golden set trả lời đúng theo hệ thống hiện tại —
đúng cái vòng lặp tự khen mà `W1-13` phải tránh.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import stat
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from pipeline.eval.golden import GoldenQuery, QueryCategory, write_golden_set
from pipeline.goldenset.schema import GoldenDraft

__all__ = [
    "MIN_FROZEN_QUESTIONS",
    "Decision",
    "FreezeError",
    "FreezeReport",
    "ReviewDecision",
    "freeze_golden_set",
    "load_decisions",
    "sha256_of_file",
    "verify_frozen",
]

logger = logging.getLogger(__name__)

MIN_FROZEN_QUESTIONS = 150
"""Ngưỡng của `W1-11`. Dưới ngưỡng thì breakdown theo 7 nhóm mất ý nghĩa thống kê."""

_ID_SEP_CHARS = ",;"


class Decision(StrEnum):
    """Quyết định của người review. Cố ý chỉ có ba giá trị."""

    ACCEPT = "accept"
    REJECT = "reject"
    EDIT = "edit"


class FreezeError(RuntimeError):
    """Dữ liệu chưa đủ điều kiện đóng băng. Thông báo luôn liệt kê `query_id`."""


class ReviewDecision(BaseModel):
    """Một dòng trong `decisions_*.csv` đã được người review điền."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    decision: Decision
    new_category: QueryCategory | None = None
    new_relevant_chunk_ids: list[str] = Field(default_factory=list)
    notes: str = ""


class FreezeReport(BaseModel):
    """Kết quả đóng băng — đủ để dán vào report mà không phải mở lại file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    drafts_total: int
    reviewed: int
    accepted: int
    edited: int
    rejected: int
    frozen: int
    by_category: dict[str, int] = Field(default_factory=dict)
    by_lang: dict[str, int] = Field(default_factory=dict)
    sha256: str = ""
    path: str = ""

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False, indent=2)

    def log_summary(self) -> None:
        log = logging.getLogger("freeze")
        log.info(
            "Đóng băng %d/%d câu (nhận %d · sửa %d · loại %d · chưa review %d)",
            self.frozen,
            self.drafts_total,
            self.accepted,
            self.edited,
            self.rejected,
            self.drafts_total - self.reviewed,
        )
        for cat, n in sorted(self.by_category.items(), key=lambda kv: -kv[1]):
            log.info("  %-16s %d", cat, n)
        log.info("  sha256 %s", self.sha256)


def _split_ids(raw: str) -> list[str]:
    """Tách danh sách chunk_id do người gõ tay.

    Chấp nhận cả dấu phẩy, chấm phẩy và khoảng trắng vì người điền CSV sẽ dùng
    cả ba, và khoảng trắng dư là chuyện bình thường khi copy từ queue Markdown.
    Bỏ luôn dấu backtick — trong `queue_v1.md` các id được bọc backtick, nên
    copy-paste gần như luôn kéo theo.
    """
    cleaned = raw.replace("`", " ")
    for ch in _ID_SEP_CHARS:
        cleaned = cleaned.replace(ch, " ")
    return list(dict.fromkeys(cleaned.split()))


def load_decisions(path: str | Path) -> dict[str, ReviewDecision]:
    """Đọc CSV quyết định, bỏ qua dòng chưa điền.

    Raises:
        FreezeError: giá trị `decision` không thuộc ba giá trị cho phép, hoặc
            `query_id` xuất hiện hai lần với hai quyết định khác nhau.
    """
    source = Path(path)
    if not source.exists():
        raise FreezeError(
            f"Không thấy {source}. Chạy `python -m pipeline.goldenset.triage` để sinh "
            "file quyết định, rồi điền cột `decision`."
        )

    out: dict[str, ReviewDecision] = {}
    allowed = {d.value for d in Decision}
    with source.open("r", encoding="utf-8-sig", newline="") as fh:
        for line_no, row in enumerate(csv.DictReader(fh), start=2):
            qid = (row.get("query_id") or "").strip()
            raw = (row.get("decision") or "").strip().lower()
            if not qid or not raw:
                continue
            if raw not in allowed:
                raise FreezeError(
                    f"{source}:{line_no} — `decision` = {raw!r} không hợp lệ. "
                    f"Chỉ nhận: {', '.join(sorted(allowed))}. "
                    "Giá trị kiểu `recheck_category` là câu hỏi triage đặt ra cho "
                    "người review, không phải câu trả lời."
                )
            new_cat_raw = (row.get("new_category") or "").strip().lower()
            try:
                new_cat = QueryCategory(new_cat_raw) if new_cat_raw else None
            except ValueError as exc:
                raise FreezeError(
                    f"{source}:{line_no} — `new_category` = {new_cat_raw!r} không phải "
                    f"nhóm hợp lệ. Chọn trong: {', '.join(c.value for c in QueryCategory)}."
                ) from exc

            decision = ReviewDecision(
                query_id=qid,
                decision=Decision(raw),
                new_category=new_cat,
                new_relevant_chunk_ids=_split_ids(row.get("new_relevant_chunk_ids") or ""),
                notes=(row.get("notes") or "").strip(),
            )
            existing = out.get(qid)
            if existing is not None and existing != decision:
                raise FreezeError(
                    f"{source}:{line_no} — `{qid}` xuất hiện hai lần với hai quyết định "
                    "khác nhau. Sửa file trước khi đóng băng."
                )
            out[qid] = decision
    return out


def _apply(draft: GoldenDraft, decision: ReviewDecision) -> GoldenQuery:
    """Chiếu một nháp đã được chấp nhận thành `GoldenQuery`.

    Raises:
        FreezeError: quyết định `edit` không nói rõ sửa gì, hoặc kết quả vi phạm
            bất biến của `GoldenQuery`.
    """
    q = draft.query
    category = decision.new_category or q.category
    if decision.decision is Decision.EDIT:
        if decision.new_category is None and not decision.new_relevant_chunk_ids:
            raise FreezeError(
                f"{q.query_id}: `decision=edit` nhưng cả `new_category` và "
                "`new_relevant_chunk_ids` đều trống — không biết phải sửa gì. "
                "Dùng `accept` nếu giữ nguyên."
            )
        chunk_ids = decision.new_relevant_chunk_ids or list(q.relevant_chunk_ids)
    else:
        chunk_ids = list(q.relevant_chunk_ids)

    # Đổi nhãn sang/khỏi `unanswerable` kéo theo `relevant_chunk_ids` phải đổi.
    # `GoldenQuery` sẽ chặn, nhưng thông báo của nó không nói được rằng nguyên
    # nhân là việc đổi nhãn ở dòng CSV này.
    if category is QueryCategory.UNANSWERABLE and chunk_ids:
        if decision.new_relevant_chunk_ids:
            raise FreezeError(
                f"{q.query_id}: đổi sang `unanswerable` nhưng vẫn điền "
                "`new_relevant_chunk_ids`. Câu không trả lời được thì không có chunk nào "
                "trả lời nó — bỏ trống ô đó."
            )
        chunk_ids = []  # nháp có chunk_id, nhãn mới nói là không — nhãn mới thắng
    if category is not QueryCategory.UNANSWERABLE and not chunk_ids:
        raise FreezeError(
            f"{q.query_id}: nhóm `{category.value}` bắt buộc có `relevant_chunk_ids`, "
            "nhưng nháp không có và `new_relevant_chunk_ids` để trống. "
            "Điền chunk_id (copy từ `queue_v1.md`) hoặc `reject` câu này."
        )

    try:
        return GoldenQuery(
            query_id=q.query_id,
            query=q.query,
            category=category,
            lang=q.lang,
            relevant_chunk_ids=chunk_ids,
            reference_answer=q.reference_answer,
            notes=decision.notes or q.notes,
            reviewed_by_human=True,
        )
    except Exception as exc:
        raise FreezeError(f"{q.query_id}: {exc}") from exc


def sha256_of_file(path: str | Path) -> str:
    """sha256 của nội dung file, đọc theo khối để không nạp cả file vào RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def freeze_golden_set(
    drafts: Sequence[GoldenDraft],
    decisions: dict[str, ReviewDecision],
    out_path: str | Path,
    *,
    min_questions: int = MIN_FROZEN_QUESTIONS,
    require_all_categories: bool = True,
    read_only: bool = True,
) -> FreezeReport:
    """Ghi `golden_v1.jsonl` từ nháp + quyết định, kèm checksum.

    Args:
        drafts: toàn bộ nháp (kể cả câu bị loại — cần để đếm cho đúng).
        decisions: quyết định theo `query_id`. Thiếu = chưa review, bị bỏ qua.
        out_path: đường dẫn file JSONL đầu ra.
        min_questions: ngưỡng tối thiểu. Đặt 0 để bỏ kiểm (chỉ dùng khi thử).
        require_all_categories: đòi đủ 7 nhóm.
        read_only: bỏ bit ghi của file sau khi ghi xong.

    Raises:
        FreezeError: quyết định trỏ tới `query_id` không tồn tại, không đủ số câu,
            thiếu nhóm, hoặc một câu vi phạm bất biến của `GoldenQuery`.
    """
    by_id = {d.query.query_id: d for d in drafts}
    unknown = sorted(set(decisions) - set(by_id))
    if unknown:
        shown = ", ".join(unknown[:5])
        raise FreezeError(
            f"{len(unknown)} `query_id` trong file quyết định không có trong tập nháp: "
            f"{shown}{'…' if len(unknown) > 5 else ''}. "
            "Có thể file quyết định thuộc một lượt sinh nháp khác."
        )

    frozen: list[GoldenQuery] = []
    problems: list[str] = []
    accepted = edited = rejected = 0

    for draft in drafts:
        decision = decisions.get(draft.query.query_id)
        if decision is None:
            continue
        if decision.decision is Decision.REJECT:
            rejected += 1
            continue
        try:
            frozen.append(_apply(draft, decision))
        except FreezeError as exc:
            problems.append(str(exc))
            continue
        if decision.decision is Decision.EDIT:
            edited += 1
        else:
            accepted += 1

    if problems:
        head = "\n  - ".join(problems[:10])
        more = f"\n  … và {len(problems) - 10} lỗi nữa" if len(problems) > 10 else ""
        raise FreezeError(f"{len(problems)} câu không đóng băng được:\n  - {head}{more}")

    by_category: dict[str, int] = {}
    by_lang: dict[str, int] = {}
    for q in frozen:
        by_category[q.category.value] = by_category.get(q.category.value, 0) + 1
        by_lang[str(q.lang)] = by_lang.get(str(q.lang), 0) + 1

    if len(frozen) < min_questions:
        raise FreezeError(
            f"Chỉ có {len(frozen)} câu được chấp nhận, `W1-11` đòi ≥ {min_questions}. "
            f"Còn {len(drafts) - len(decisions)} câu chưa review."
        )
    if require_all_categories:
        missing = [c.value for c in QueryCategory if c.value not in by_category]
        if missing:
            raise FreezeError(
                f"Thiếu hoàn toàn nhóm: {', '.join(missing)}. "
                "Golden set thiếu một nhóm thì breakdown theo nhóm ở eval sẽ im lặng "
                "bỏ qua nó, và không ai nhận ra là hệ thống chưa bao giờ được đo ở đó."
            )

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _unlock(target)
    write_golden_set(target, frozen)
    digest = sha256_of_file(target)
    (target.parent / f"{target.name}.sha256").write_text(
        f"{digest}  {target.name}\n", encoding="utf-8"
    )
    if read_only:
        _lock(target)

    return FreezeReport(
        drafts_total=len(drafts),
        reviewed=len(decisions),
        accepted=accepted,
        edited=edited,
        rejected=rejected,
        frozen=len(frozen),
        by_category=by_category,
        by_lang=by_lang,
        sha256=digest,
        path=str(target),
    )


def _lock(path: Path) -> None:
    """Bỏ bit ghi. Không phải bảo mật — chỉ là cái gờ chống sửa nhầm.

    Golden set là thước đo; sửa nó làm mọi metric lịch sử mất nghĩa mà không có
    dấu vết nào. `chmod` chặn được `>` và một lần lưu vô ý từ editor, và đó đúng
    là hai cách nó bị hỏng trên thực tế.
    """
    try:
        os.chmod(path, stat.S_IREAD)
    except OSError as exc:  # pragma: no cover - hệ file không hỗ trợ
        logger.warning("Không đặt được read-only cho %s: %s", path, exc)


def _unlock(path: Path) -> None:
    """Trả lại bit ghi trước khi ghi đè, nếu file đã bị lock từ lượt trước."""
    if path.exists():
        try:
            os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
        except OSError as exc:  # pragma: no cover
            logger.warning("Không mở được quyền ghi cho %s: %s", path, exc)


def verify_frozen(path: str | Path) -> str:
    """Đối chiếu file đã đóng băng với checksum đi kèm.

    Returns:
        sha256 hiện tại của file.

    Raises:
        FreezeError: thiếu file checksum, hoặc nội dung đã đổi.
    """
    target = Path(path)
    sidecar = target.parent / f"{target.name}.sha256"
    if not sidecar.exists():
        raise FreezeError(f"Không thấy {sidecar} — không xác minh được {target}.")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = sha256_of_file(target)
    if actual != expected:
        raise FreezeError(
            f"{target} đã bị sửa sau khi đóng băng.\n"
            f"  checksum ghi lại: {expected}\n"
            f"  checksum hiện tại: {actual}\n"
            "Mọi metric đo bằng file này trước đây không còn so sánh được với bây giờ."
        )
    return actual


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: `python -m pipeline.goldenset.freeze`."""
    import argparse

    from pipeline.goldenset.schema import load_drafts

    parser = argparse.ArgumentParser(description="Đóng băng golden set (W1-11).")
    parser.add_argument("--drafts", type=Path, default=Path("data/golden/draft_v1.jsonl"))
    parser.add_argument(
        "--decisions", type=Path, default=Path("data/golden/review/decisions_v1.csv")
    )
    parser.add_argument("--out", type=Path, default=Path("data/golden/golden_v1.jsonl"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--min-questions",
        type=int,
        default=MIN_FROZEN_QUESTIONS,
        help="Đặt 0 để bỏ kiểm ngưỡng (chỉ dùng khi thử, đừng dùng cho golden_v1).",
    )
    parser.add_argument("--allow-missing-categories", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Chỉ đối chiếu file đã đóng băng với checksum, không ghi gì.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.verify:
        try:
            digest = verify_frozen(args.out)
        except FreezeError as exc:
            logger.error("%s", exc)
            return 1
        logger.info("%s khớp checksum (%s)", args.out, digest)
        return 0

    try:
        drafts = load_drafts(args.drafts)
        decisions = load_decisions(args.decisions)
        report = freeze_golden_set(
            drafts,
            decisions,
            args.out,
            min_questions=args.min_questions,
            require_all_categories=not args.allow_missing_categories,
        )
    except FreezeError as exc:
        logger.error("%s", exc)
        return 1

    report.log_summary()
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report.to_json(), encoding="utf-8")
        logger.info("Đã ghi %s", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
