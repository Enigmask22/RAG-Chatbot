"""`W5-04` — lấy mẫu phân tầng, cam kết bằng hash, và cái bẫy thứ tự dòng.

Bài quan trọng nhất trong file này là `TestTheBlindFileIsActuallyBlind`. Nó tồn
tại vì bản đầu của `stratified_sample` **không** mù: nó ghi ra theo tầng, nên
người gán nhãn đọc tới mục thứ 18/50 là suy ra được ranh giới và biết trước
judge đã nói gì. File không chứa nhãn nào — chỉ thứ tự dòng chứa. Đó là loại rò
rỉ mà một bài test "file không có cột nhãn" sẽ không bao giờ bắt được.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from itertools import pairwise
from pathlib import Path
from typing import ClassVar

import pytest

from pipeline.eval.answer_run import AnswerRecord
from pipeline.eval.calibration import (
    RUBRICS,
    UNPARSEABLE,
    Rubric,
    Sampled,
    allocate,
    build_questions,
    parse_allocation,
    sha256_of,
    stratified_sample,
)
from pipeline.eval.judge import JudgeQuestion, JudgeVerdict


def question(ref: str) -> JudgeQuestion:
    return JudgeQuestion(
        prompt_id="judge-faithfulness",
        labels=("SUPPORTED", "CONTRADICTED", "NOT_FOUND", "NO_CLAIM"),
        variables={"context": f"ngữ cảnh {ref}", "claim": f"mệnh đề {ref}"},
        ref=ref,
    )


def verdict(ref: str, label: str | None) -> JudgeVerdict:
    return JudgeVerdict(
        ref=ref, label=label, reason="vì thế", served_model="m", cached=True, cost_usd=0.0
    )


def population(counts: dict[str, int]) -> tuple[list[JudgeQuestion], list[JudgeVerdict]]:
    """`{"SUPPORTED": 40, ...}` → hai danh sách song song, ref có thứ tự xác định."""
    questions: list[JudgeQuestion] = []
    verdicts: list[JudgeVerdict] = []
    for label, n in counts.items():
        for i in range(n):
            ref = f"q{label[:2].lower()}{i:03d}#s0"
            questions.append(question(ref))
            verdicts.append(verdict(ref, None if label == UNPARSEABLE else label))
    return questions, verdicts


class TestAllocation:
    def test_takes_the_minimum_of_wanted_and_available(self) -> None:
        assert allocate({"A": 400, "B": 5}, {"A": 30, "B": 15}) == {"A": 30, "B": 5}

    def test_a_stratum_with_nothing_in_it_disappears_rather_than_erroring(self) -> None:
        """`CONTRADICTED` có 0 mục trong lần chạy thật — phải bỏ, không phải nổ."""
        assert allocate({"A": 10}, {"A": 5, "CONTRADICTED": 5}) == {"A": 5}

    def test_a_short_stratum_is_not_compensated_from_another(self) -> None:
        """Bù tự động sẽ lặng lẽ đổi trọng số của tầng nhận bù.

        Muốn 15 + 15, chỉ có 15 + 5 ⇒ lấy 20, **không** phải 25 bằng cách bốc
        thêm 10 từ tầng A. Cỡ mẫu hụt là chuyện phải thấy.
        """
        taken = allocate({"A": 400, "B": 5}, {"A": 15, "B": 15})
        assert sum(taken.values()) == 20

    def test_parse_allocation_round_trips(self) -> None:
        assert parse_allocation("SUPPORTED=30, NO_CLAIM=15") == {"SUPPORTED": 30, "NO_CLAIM": 15}

    @pytest.mark.parametrize("bad", ["SUPPORTED", "SUPPORTED=x", "SUPPORTED=-3", ""])
    def test_parse_allocation_rejects_garbage(self, bad: str) -> None:
        with pytest.raises(ValueError):
            parse_allocation(bad)


class TestStratifiedSample:
    COUNTS: ClassVar[dict[str, int]] = {"SUPPORTED": 402, "NO_CLAIM": 26, "NOT_FOUND": 5}
    WANTED: ClassVar[dict[str, int]] = {
        "SUPPORTED": 30,
        "NO_CLAIM": 15,
        "NOT_FOUND": 5,
        "CONTRADICTED": 5,
    }

    def picked(self, seed: int = 20260906) -> list[Sampled]:
        questions, verdicts = population(self.COUNTS)
        return stratified_sample(questions, verdicts, self.WANTED, seed=seed)

    def test_weight_is_population_over_sample_per_stratum(self) -> None:
        by_stratum = {s.stratum: s.weight for s in self.picked()}
        assert by_stratum["SUPPORTED"] == pytest.approx(402 / 30)
        assert by_stratum["NO_CLAIM"] == pytest.approx(26 / 15)
        assert by_stratum["NOT_FOUND"] == pytest.approx(1.0)

    def test_weights_restore_the_population_counts_exactly(self) -> None:
        """Đây là điều kiện để ước lượng quy về quần thể có nghĩa."""
        totals: dict[str, float] = {}
        for s in self.picked():
            totals[s.stratum] = totals.get(s.stratum, 0.0) + s.weight
        assert totals == pytest.approx({k: float(v) for k, v in self.COUNTS.items()})

    def test_same_seed_same_sample(self) -> None:
        assert [s.ref for s in self.picked(7)] == [s.ref for s in self.picked(7)]

    def test_different_seed_different_sample(self) -> None:
        assert [s.ref for s in self.picked(7)] != [s.ref for s in self.picked(8)]

    def test_adding_a_query_at_the_end_of_the_run_does_not_reshuffle_the_sample(self) -> None:
        """Mẫu phải bám vào `ref`, không bám vào thứ tự dòng trong file run.

        Nếu không, thêm một truy vấn vào cuối golden set là đủ để mẫu hiệu chỉnh
        đổi hoàn toàn — và mọi nhãn tay đã bỏ công gán trở thành vô dụng.
        """
        questions, verdicts = population(self.COUNTS)
        shuffled = list(zip(questions, verdicts, strict=True))[::-1]
        a = stratified_sample(questions, verdicts, self.WANTED, seed=5)
        b = stratified_sample(
            [q for q, _ in shuffled], [v for _, v in shuffled], self.WANTED, seed=5
        )
        assert sorted(s.ref for s in a) == sorted(s.ref for s in b)

    def test_unparseable_becomes_its_own_stratum(self) -> None:
        questions, verdicts = population({"SUPPORTED": 10, UNPARSEABLE: 3})
        picked = stratified_sample(questions, verdicts, {UNPARSEABLE: 3}, seed=1)
        assert {s.stratum for s in picked} == {UNPARSEABLE}
        assert all(s.judge_label is None for s in picked)


class TestTheBlindFileIsActuallyBlind:
    """⭐⭐ Hồi quy cho một lỗi bắt được **khi đang dùng**, không phải khi viết.

    Bản đầu ghi mẫu theo tầng: 5 `NOT_FOUND`, rồi 15 `NO_CLAIM`, rồi 30
    `SUPPORTED`. File mù đúng là không có cột nhãn — nhưng **vị trí dòng** thì
    có, và nó đủ để suy ra toàn bộ phán quyết của judge từ mục thứ 18 trở đi.
    """

    def test_row_order_carries_no_information_about_the_judge_label(self) -> None:
        questions, verdicts = population({"SUPPORTED": 402, "NO_CLAIM": 26, "NOT_FOUND": 5})
        picked = stratified_sample(
            questions, verdicts, {"SUPPORTED": 30, "NO_CLAIM": 15, "NOT_FOUND": 5}, seed=20260906
        )
        strata = [s.stratum for s in picked]
        assert len(picked) == 50

        # Nếu còn gom cụm theo tầng thì số lần "đổi tầng" khi đi dọc danh sách
        # sẽ bằng đúng (số tầng − 1) = 2. Trộn đều thì con số ấy lớn hơn hẳn.
        switches = sum(1 for a, b in pairwise(strata) if a != b)
        assert switches > 10, f"thứ tự vẫn gom cụm theo tầng ({switches} lần đổi)"

    def test_a_stratum_is_not_confined_to_one_contiguous_block(self) -> None:
        questions, verdicts = population({"SUPPORTED": 402, "NO_CLAIM": 26, "NOT_FOUND": 5})
        picked = stratified_sample(
            questions, verdicts, {"SUPPORTED": 30, "NO_CLAIM": 15, "NOT_FOUND": 5}, seed=20260906
        )
        positions = [i for i, s in enumerate(picked) if s.stratum == "NOT_FOUND"]
        assert max(positions) - min(positions) > len(picked) // 3, (
            "5 mục NOT_FOUND vẫn nằm sát nhau — người gán nhãn sẽ nhận ra khối"
        )

    def test_shuffle_is_deterministic_for_a_given_seed(self) -> None:
        questions, verdicts = population({"SUPPORTED": 100, "NOT_FOUND": 5})
        wanted = {"SUPPORTED": 10, "NOT_FOUND": 5}
        first = [s.ref for s in stratified_sample(questions, verdicts, wanted, seed=3)]
        again = [s.ref for s in stratified_sample(questions, verdicts, wanted, seed=3)]
        assert first == again, "trộn phải tất định, nếu không thì mẫu không tái lập được"


class TestBuildQuestions:
    @staticmethod
    def record(**over: object) -> AnswerRecord:
        base: dict[str, object] = {
            "query_id": "q1",
            "query": "GDP tăng bao nhiêu?",
            "category": "factoid",
            "lang": "vi",
            "answer": "GDP tăng 7,09% [1]. Con số này cao hơn dự báo trước đó [1].",
            "route": "retrieve",
            "rewritten": None,
            "prompt_spec": "chat-system@v2",
            "bundle_version": "0.2.0",
            "model": "deepseek-v4-flash",
            "finish_reason": "stop",
            "sources": [{"n": 1, "chunk_id": "c1"}],
        }
        base.update(over)
        return AnswerRecord(**base)  # type: ignore[arg-type]

    CHUNKS: ClassVar[dict[str, str]] = {"c1": "GDP Việt Nam tăng 7,09% trong năm 2024."}

    def test_faithfulness_questions_come_from_the_metrics_module_not_a_copy(self) -> None:
        """Chép lại logic ở đây sẽ làm khoá cache lệch ⇒ trượt hàng trăm lượt.

        Kiểm bằng cách so trực tiếp với `faithfulness_questions`: hai bên phải
        ra **cùng** danh sách biến, không chỉ cùng số lượng.
        """
        from pipeline.eval.generation_metrics import faithfulness_questions

        record = self.record()
        mine = build_questions("faithfulness", [record], self.CHUNKS)
        theirs = [q for _, q in faithfulness_questions(record, self.CHUNKS) if q is not None]
        assert [q.variables for q in mine] == [q.variables for q in theirs]
        assert [q.ref for q in mine] == [q.ref for q in theirs]

    def test_non_retrieve_routes_are_excluded(self) -> None:
        assert (
            build_questions("faithfulness", [self.record(route="no_retrieval")], self.CHUNKS) == []
        )

    def test_relevancy_questions_skip_empty_answers(self) -> None:
        records = [self.record(), self.record(query_id="q2", answer="   ")]
        refs = [q.ref for q in build_questions("relevancy", records, self.CHUNKS)]
        assert refs == ["q1"]

    def test_unknown_rubric_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="rubric lạ"):
            build_questions("hallucination", [], {})

    def test_every_rubric_declares_a_numerator_inside_its_denominator(self) -> None:
        """Nếu không, `rate_of` sẽ ném — nhưng chỉ lúc chạy `score`, sau khi đã
        bỏ công gán nhãn tay xong."""
        for name, rubric in RUBRICS.items():
            assert set(rubric.numerator) <= set(rubric.denominator), name
            assert set(rubric.denominator) <= set(rubric.labels), name

    def test_rubric_is_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            RUBRICS["faithfulness"].prompt_id = "x"  # type: ignore[misc]
        assert isinstance(RUBRICS["faithfulness"], Rubric)


class TestSealedCommitment:
    def test_hash_changes_when_one_verdict_changes(self, tmp_path: Path) -> None:
        """Cam kết chỉ có giá trị nếu sửa một ký tự cũng làm nó đổi."""
        path = tmp_path / "sealed.jsonl"
        rows = [{"ref": "a", "judge_label": "SUPPORTED"}, {"ref": "b", "judge_label": "NOT_FOUND"}]
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in rows), encoding="utf-8"
        )
        before = sha256_of(path)
        rows[1]["judge_label"] = "SUPPORTED"
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in rows), encoding="utf-8"
        )
        assert sha256_of(path) != before
