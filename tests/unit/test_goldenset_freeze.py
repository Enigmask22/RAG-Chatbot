"""Test cho `pipeline.goldenset.freeze`.

Hai bất biến mà file này canh chặt nhất:

1. **Freeze không đoán hộ người review.** Một quyết định `edit` không nói rõ sửa
   gì phải thành lỗi, chứ không được lặng lẽ lấy top-1 của retriever điền vào —
   đó là cách dạy golden set trả lời đúng theo hệ thống hiện tại.
2. **Ô `decision` trống không phải `accept`.** Nếu chưa-review bị tính là
   chấp nhận thì chạy freeze ngay sau triage sẽ ra một `golden_v1` gồm 266 câu
   chưa ai đọc, và không có dấu hiệu nào cho thấy điều đó.
"""

from __future__ import annotations

import csv
import os
import stat
from pathlib import Path

import pytest

from pipeline.eval.golden import GoldenQuery, QueryCategory, load_golden_set
from pipeline.goldenset.freeze import (
    Decision,
    FreezeError,
    ReviewDecision,
    freeze_golden_set,
    load_decisions,
    sha256_of_file,
    verify_frozen,
)
from pipeline.goldenset.schema import DraftProvenance, GoldenDraft
from rag_core.schemas import Language, TextSpan

_ALL_CATEGORIES = list(QueryCategory)

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


def _draft(
    query_id: str,
    *,
    category: QueryCategory = QueryCategory.FACTOID,
    chunk_ids: list[str] | None = None,
    lang: Language = Language.VI,
) -> GoldenDraft:
    ids = (
        [] if category is QueryCategory.UNANSWERABLE else (chunk_ids or [f"doc-{query_id}::00001"])
    )
    return GoldenDraft(
        query=GoldenQuery(
            query_id=query_id,
            query=f"cau hoi {query_id}",
            category=category,
            lang=lang,
            relevant_chunk_ids=ids,
            reference_answer="dap an",
        ),
        provenance=DraftProvenance(
            generator_model="fake",
            generator_model_requested="fake",
            category_requested=category,
            source_chunk_ids=ids,
        ),
    )


def _full_set(n_per_category: int = 25) -> list[GoldenDraft]:
    """Tập nháp đủ 7 nhóm và vượt ngưỡng 150 câu."""
    out: list[GoldenDraft] = []
    for cat in _ALL_CATEGORIES:
        for i in range(n_per_category):
            out.append(_draft(f"{cat.value}-{i:03d}", category=cat))
    return out


def _accept_all(drafts: list[GoldenDraft]) -> dict[str, ReviewDecision]:
    return {
        d.query.query_id: ReviewDecision(query_id=d.query.query_id, decision=Decision.ACCEPT)
        for d in drafts
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_DECISION_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in _DECISION_FIELDS})
    return path


class TestLoadDecisions:
    def test_blank_decision_is_skipped_not_accepted(self, tmp_path: Path) -> None:
        """Bất biến số 2 — chưa review KHÔNG phải chấp nhận."""
        p = _write_csv(
            tmp_path / "d.csv",
            [
                {"query_id": "a", "decision": "accept"},
                {"query_id": "b", "decision": ""},
                {"query_id": "c", "decision": "   "},
            ],
        )
        decisions = load_decisions(p)
        assert set(decisions) == {"a"}

    def test_rejects_a_triage_suggestion_as_decision(self, tmp_path: Path) -> None:
        """`recheck_category` là câu hỏi, không phải câu trả lời."""
        p = _write_csv(tmp_path / "d.csv", [{"query_id": "a", "decision": "recheck_category"}])
        with pytest.raises(FreezeError, match="không hợp lệ"):
            load_decisions(p)

    def test_decision_is_case_insensitive(self, tmp_path: Path) -> None:
        p = _write_csv(tmp_path / "d.csv", [{"query_id": "a", "decision": "ACCEPT"}])
        assert load_decisions(p)["a"].decision is Decision.ACCEPT

    def test_missing_file_names_the_fix(self, tmp_path: Path) -> None:
        with pytest.raises(FreezeError, match=r"pipeline\.goldenset\.triage"):
            load_decisions(tmp_path / "khong-co.csv")

    def test_invalid_new_category_lists_the_valid_ones(self, tmp_path: Path) -> None:
        p = _write_csv(
            tmp_path / "d.csv",
            [{"query_id": "a", "decision": "edit", "new_category": "khong-ton-tai"}],
        )
        with pytest.raises(FreezeError, match="factoid"):
            load_decisions(p)

    def test_duplicate_query_id_with_same_decision_is_fine(self, tmp_path: Path) -> None:
        p = _write_csv(
            tmp_path / "d.csv",
            [{"query_id": "a", "decision": "accept"}, {"query_id": "a", "decision": "accept"}],
        )
        assert set(load_decisions(p)) == {"a"}

    def test_duplicate_query_id_with_conflicting_decision_fails(self, tmp_path: Path) -> None:
        p = _write_csv(
            tmp_path / "d.csv",
            [{"query_id": "a", "decision": "accept"}, {"query_id": "a", "decision": "reject"}],
        )
        with pytest.raises(FreezeError, match="hai quyết định"):
            load_decisions(p)


class TestChunkIdParsing:
    """Người điền CSV bằng tay, và họ copy từ Markdown có backtick."""

    @pytest.mark.parametrize(
        "raw",
        [
            "doc-a::00001,doc-a::00002",
            "doc-a::00001; doc-a::00002",
            "doc-a::00001 doc-a::00002",
            "`doc-a::00001`, `doc-a::00002`",
            "  doc-a::00001 ,, doc-a::00002  ",
        ],
    )
    def test_separators_and_backticks(self, tmp_path: Path, raw: str) -> None:
        p = _write_csv(
            tmp_path / "d.csv",
            [{"query_id": "a", "decision": "edit", "new_relevant_chunk_ids": raw}],
        )
        assert load_decisions(p)["a"].new_relevant_chunk_ids == [
            "doc-a::00001",
            "doc-a::00002",
        ]

    def test_duplicates_collapsed_preserving_order(self, tmp_path: Path) -> None:
        p = _write_csv(
            tmp_path / "d.csv",
            [
                {
                    "query_id": "a",
                    "decision": "edit",
                    "new_relevant_chunk_ids": "b::2, a::1, b::2",
                }
            ],
        )
        assert load_decisions(p)["a"].new_relevant_chunk_ids == ["b::2", "a::1"]


class TestNoGuessing:
    """Bất biến số 1 — freeze không tự điền nhãn."""

    def test_edit_without_any_change_is_an_error(self, tmp_path: Path) -> None:
        drafts = _full_set()
        decisions = _accept_all(drafts)
        target = drafts[0].query.query_id
        decisions[target] = ReviewDecision(query_id=target, decision=Decision.EDIT)
        with pytest.raises(FreezeError, match="không biết phải sửa gì"):
            freeze_golden_set(drafts, decisions, tmp_path / "g.jsonl")

    def test_answerable_without_chunk_ids_is_an_error(self, tmp_path: Path) -> None:
        """Câu factoid mà nháp không có chunk_id → phải điền hoặc reject, không đoán."""
        drafts = _full_set()
        broken = _draft("broken-001", category=QueryCategory.UNANSWERABLE)
        drafts.append(broken)
        decisions = _accept_all(drafts)
        decisions["broken-001"] = ReviewDecision(
            query_id="broken-001",
            decision=Decision.EDIT,
            new_category=QueryCategory.FACTOID,
        )
        with pytest.raises(FreezeError, match="bắt buộc có `relevant_chunk_ids`"):
            freeze_golden_set(drafts, decisions, tmp_path / "g.jsonl")

    def test_unanswerable_with_chunk_ids_filled_is_an_error(self, tmp_path: Path) -> None:
        drafts = _full_set()
        target = drafts[0].query.query_id
        decisions = _accept_all(drafts)
        decisions[target] = ReviewDecision(
            query_id=target,
            decision=Decision.EDIT,
            new_category=QueryCategory.UNANSWERABLE,
            new_relevant_chunk_ids=["doc-a::00001"],
        )
        with pytest.raises(FreezeError, match="không có chunk nào"):
            freeze_golden_set(drafts, decisions, tmp_path / "g.jsonl")

    def test_relabel_to_unanswerable_clears_chunk_ids(self, tmp_path: Path) -> None:
        """Nhãn mới thắng: nháp có chunk_id, đổi sang unanswerable thì bỏ hết."""
        drafts = _full_set()
        target = next(d.query.query_id for d in drafts if d.query.category is QueryCategory.FACTOID)
        decisions = _accept_all(drafts)
        decisions[target] = ReviewDecision(
            query_id=target,
            decision=Decision.EDIT,
            new_category=QueryCategory.UNANSWERABLE,
        )
        freeze_golden_set(drafts, decisions, tmp_path / "g.jsonl")
        frozen = {q.query_id: q for q in load_golden_set(tmp_path / "g.jsonl")}
        assert frozen[target].category is QueryCategory.UNANSWERABLE
        assert frozen[target].relevant_chunk_ids == []


class TestGates:
    def test_below_min_questions_is_refused(self, tmp_path: Path) -> None:
        drafts = _full_set(n_per_category=2)  # 14 câu
        with pytest.raises(FreezeError, match="đòi ≥ 150"):
            freeze_golden_set(drafts, _accept_all(drafts), tmp_path / "g.jsonl")

    def test_error_says_how_many_are_unreviewed(self, tmp_path: Path) -> None:
        drafts = _full_set()
        partial = dict(list(_accept_all(drafts).items())[:20])
        with pytest.raises(FreezeError, match="chưa review"):
            freeze_golden_set(drafts, partial, tmp_path / "g.jsonl")

    def test_missing_category_is_refused(self, tmp_path: Path) -> None:
        drafts = [d for d in _full_set(30) if d.query.category is not QueryCategory.TABLE_LOOKUP]
        with pytest.raises(FreezeError, match="table_lookup"):
            freeze_golden_set(drafts, _accept_all(drafts), tmp_path / "g.jsonl")

    def test_missing_category_error_explains_the_consequence(self, tmp_path: Path) -> None:
        drafts = [d for d in _full_set(30) if d.query.category is not QueryCategory.TABLE_LOOKUP]
        with pytest.raises(FreezeError, match="im lặng bỏ qua"):
            freeze_golden_set(drafts, _accept_all(drafts), tmp_path / "g.jsonl")

    def test_can_be_relaxed_for_a_trial_run(self, tmp_path: Path) -> None:
        drafts = _full_set(n_per_category=2)
        report = freeze_golden_set(
            drafts, _accept_all(drafts), tmp_path / "g.jsonl", min_questions=0
        )
        assert report.frozen == 14

    def test_unknown_query_id_in_decisions_is_refused(self, tmp_path: Path) -> None:
        drafts = _full_set()
        decisions = _accept_all(drafts)
        decisions["cau-tu-luot-khac"] = ReviewDecision(
            query_id="cau-tu-luot-khac", decision=Decision.ACCEPT
        )
        with pytest.raises(FreezeError, match="một lượt sinh nháp khác"):
            freeze_golden_set(drafts, decisions, tmp_path / "g.jsonl")

    def test_all_problems_reported_together(self, tmp_path: Path) -> None:
        """Sửa từng lỗi một rồi chạy lại 12 lần là cách chắc chắn làm người ta bỏ."""
        drafts = _full_set()
        decisions = _accept_all(drafts)
        for d in drafts[:3]:
            decisions[d.query.query_id] = ReviewDecision(
                query_id=d.query.query_id, decision=Decision.EDIT
            )
        with pytest.raises(FreezeError, match="3 câu không đóng băng được"):
            freeze_golden_set(drafts, decisions, tmp_path / "g.jsonl")


class TestOutput:
    def test_frozen_queries_are_marked_reviewed(self, tmp_path: Path) -> None:
        drafts = _full_set()
        freeze_golden_set(drafts, _accept_all(drafts), tmp_path / "g.jsonl")
        queries = load_golden_set(tmp_path / "g.jsonl")
        assert queries
        assert all(q.reviewed_by_human for q in queries)

    def test_rejected_questions_are_excluded(self, tmp_path: Path) -> None:
        drafts = _full_set()
        decisions = _accept_all(drafts)
        dropped = [d.query.query_id for d in drafts[:10]]
        for qid in dropped:
            decisions[qid] = ReviewDecision(query_id=qid, decision=Decision.REJECT)
        report = freeze_golden_set(drafts, decisions, tmp_path / "g.jsonl")
        assert report.rejected == 10
        assert report.frozen == len(drafts) - 10
        ids = {q.query_id for q in load_golden_set(tmp_path / "g.jsonl")}
        assert not (ids & set(dropped))

    def test_report_counts_add_up(self, tmp_path: Path) -> None:
        drafts = _full_set()
        decisions = _accept_all(drafts)
        decisions[drafts[0].query.query_id] = ReviewDecision(
            query_id=drafts[0].query.query_id,
            decision=Decision.EDIT,
            new_relevant_chunk_ids=["doc-x::00001"],
        )
        decisions[drafts[1].query.query_id] = ReviewDecision(
            query_id=drafts[1].query.query_id, decision=Decision.REJECT
        )
        report = freeze_golden_set(drafts, decisions, tmp_path / "g.jsonl")
        assert report.accepted + report.edited + report.rejected == report.reviewed
        assert report.accepted + report.edited == report.frozen
        assert report.edited == 1
        assert report.rejected == 1

    def test_edit_applies_new_chunk_ids(self, tmp_path: Path) -> None:
        drafts = _full_set()
        target = drafts[0].query.query_id
        decisions = _accept_all(drafts)
        decisions[target] = ReviewDecision(
            query_id=target,
            decision=Decision.EDIT,
            new_relevant_chunk_ids=["doc-moi::00007"],
        )
        freeze_golden_set(drafts, decisions, tmp_path / "g.jsonl")
        frozen = {q.query_id: q for q in load_golden_set(tmp_path / "g.jsonl")}
        assert frozen[target].relevant_chunk_ids == ["doc-moi::00007"]

    def test_notes_are_carried_over(self, tmp_path: Path) -> None:
        drafts = _full_set()
        target = drafts[0].query.query_id
        decisions = _accept_all(drafts)
        decisions[target] = ReviewDecision(
            query_id=target, decision=Decision.ACCEPT, notes="so lieu can doi chieu lai"
        )
        freeze_golden_set(drafts, decisions, tmp_path / "g.jsonl")
        frozen = {q.query_id: q for q in load_golden_set(tmp_path / "g.jsonl")}
        assert frozen[target].notes == "so lieu can doi chieu lai"

    def test_distribution_recorded(self, tmp_path: Path) -> None:
        drafts = _full_set()
        report = freeze_golden_set(drafts, _accept_all(drafts), tmp_path / "g.jsonl")
        assert set(report.by_category) == {c.value for c in QueryCategory}
        assert sum(report.by_category.values()) == report.frozen
        assert report.by_lang == {"vi": report.frozen}


class TestChecksumAndLock:
    def test_sidecar_checksum_matches(self, tmp_path: Path) -> None:
        drafts = _full_set()
        out = tmp_path / "g.jsonl"
        report = freeze_golden_set(drafts, _accept_all(drafts), out)
        sidecar = tmp_path / "g.jsonl.sha256"
        assert sidecar.exists()
        assert sidecar.read_text(encoding="utf-8").split()[0] == report.sha256
        assert report.sha256 == sha256_of_file(out)

    def test_verify_passes_on_untouched_file(self, tmp_path: Path) -> None:
        drafts = _full_set()
        out = tmp_path / "g.jsonl"
        report = freeze_golden_set(drafts, _accept_all(drafts), out)
        assert verify_frozen(out) == report.sha256

    def test_verify_detects_tampering(self, tmp_path: Path) -> None:
        drafts = _full_set()
        out = tmp_path / "g.jsonl"
        freeze_golden_set(drafts, _accept_all(drafts), out)
        os.chmod(out, stat.S_IREAD | stat.S_IWRITE)
        with out.open("a", encoding="utf-8") as fh:
            fh.write("\n")
        with pytest.raises(FreezeError, match="không còn so sánh được"):
            verify_frozen(out)

    def test_verify_without_sidecar_fails(self, tmp_path: Path) -> None:
        (tmp_path / "g.jsonl").write_text("{}\n", encoding="utf-8")
        with pytest.raises(FreezeError, match="không xác minh được"):
            verify_frozen(tmp_path / "g.jsonl")

    def test_file_is_read_only_after_freeze(self, tmp_path: Path) -> None:
        drafts = _full_set()
        out = tmp_path / "g.jsonl"
        freeze_golden_set(drafts, _accept_all(drafts), out)
        assert not os.access(out, os.W_OK)

    def test_read_only_can_be_disabled(self, tmp_path: Path) -> None:
        drafts = _full_set()
        out = tmp_path / "g.jsonl"
        freeze_golden_set(drafts, _accept_all(drafts), out, read_only=False)
        assert os.access(out, os.W_OK)

    def test_refreeze_overwrites_a_locked_file(self, tmp_path: Path) -> None:
        """Đóng băng lại phải chạy được, nếu không thì lock thành cái bẫy."""
        drafts = _full_set()
        out = tmp_path / "g.jsonl"
        freeze_golden_set(drafts, _accept_all(drafts), out)
        decisions = _accept_all(drafts)
        decisions[drafts[0].query.query_id] = ReviewDecision(
            query_id=drafts[0].query.query_id, decision=Decision.REJECT
        )
        second = freeze_golden_set(drafts, decisions, out)
        assert second.frozen == len(drafts) - 1
        assert verify_frozen(out) == second.sha256

    def test_checksum_changes_when_content_changes(self, tmp_path: Path) -> None:
        drafts = _full_set()
        a = freeze_golden_set(drafts, _accept_all(drafts), tmp_path / "a.jsonl")
        decisions = _accept_all(drafts)
        decisions[drafts[0].query.query_id] = ReviewDecision(
            query_id=drafts[0].query.query_id, decision=Decision.REJECT
        )
        b = freeze_golden_set(drafts, decisions, tmp_path / "b.jsonl")
        assert a.sha256 != b.sha256


class TestSpansSurviveFreeze:
    """`relevant_spans` là thứ làm golden set độc lập với chunking (`TD-12`).

    Làm rơi nó ở bước freeze thì toàn bộ công neo span mất sạch, mà không có
    triệu chứng nào: file vẫn hợp lệ, eval vẫn chạy, chỉ là chấm bằng nhãn
    `chunk_id` chỉ đúng với đúng một cấu hình chunking.
    """

    @staticmethod
    def _with_spans(qid: str) -> GoldenDraft:
        draft = _draft(qid)
        return draft.model_copy(
            update={
                "query": draft.query.model_copy(
                    update={"relevant_spans": [TextSpan(doc_id="doc-x", start=10, end=200)]}
                )
            }
        )

    def test_accept_carries_spans_through(self, tmp_path: Path) -> None:
        drafts = _full_set()
        target = drafts[0].query.query_id
        drafts[0] = self._with_spans(target)
        freeze_golden_set(drafts, _accept_all(drafts), tmp_path / "g.jsonl")
        frozen = {q.query_id: q for q in load_golden_set(tmp_path / "g.jsonl")}
        assert len(frozen[target].relevant_spans) == 1
        assert frozen[target].relevant_spans[0].start == 10

    def test_manual_chunk_ids_drop_the_spans(self, tmp_path: Path) -> None:
        """Ánh xạ span ghi đè chunk_id, nên giữ span cũ sẽ bỏ sửa tay một cách âm thầm."""
        drafts = _full_set()
        target = drafts[0].query.query_id
        drafts[0] = self._with_spans(target)
        decisions = _accept_all(drafts)
        decisions[target] = ReviewDecision(
            query_id=target,
            decision=Decision.EDIT,
            new_relevant_chunk_ids=["doc-nguoi-chon::00007"],
        )
        freeze_golden_set(drafts, decisions, tmp_path / "g.jsonl")
        frozen = {q.query_id: q for q in load_golden_set(tmp_path / "g.jsonl")}
        assert frozen[target].relevant_chunk_ids == ["doc-nguoi-chon::00007"]
        assert frozen[target].relevant_spans == []

    def test_category_only_edit_keeps_spans(self, tmp_path: Path) -> None:
        drafts = _full_set()
        target = next(d.query.query_id for d in drafts if d.query.category is QueryCategory.FACTOID)
        idx = next(i for i, d in enumerate(drafts) if d.query.query_id == target)
        drafts[idx] = self._with_spans(target)
        decisions = _accept_all(drafts)
        decisions[target] = ReviewDecision(
            query_id=target, decision=Decision.EDIT, new_category=QueryCategory.MULTI_HOP
        )
        freeze_golden_set(drafts, decisions, tmp_path / "g.jsonl")
        frozen = {q.query_id: q for q in load_golden_set(tmp_path / "g.jsonl")}
        assert frozen[target].category is QueryCategory.MULTI_HOP
        assert len(frozen[target].relevant_spans) == 1

    def test_relabel_to_unanswerable_drops_spans(self, tmp_path: Path) -> None:
        drafts = _full_set()
        target = next(d.query.query_id for d in drafts if d.query.category is QueryCategory.FACTOID)
        idx = next(i for i, d in enumerate(drafts) if d.query.query_id == target)
        drafts[idx] = self._with_spans(target)
        decisions = _accept_all(drafts)
        decisions[target] = ReviewDecision(
            query_id=target,
            decision=Decision.EDIT,
            new_category=QueryCategory.UNANSWERABLE,
        )
        freeze_golden_set(drafts, decisions, tmp_path / "g.jsonl")
        frozen = {q.query_id: q for q in load_golden_set(tmp_path / "g.jsonl")}
        assert frozen[target].relevant_spans == []
