"""Test cho phép so mức cải thiện **giữa các nhóm** (`W2-09`).

Mỗi lớp dưới đây ghim một chuyện đo được trong lúc làm `W2-09`, không phải một
khả năng lý thuyết. Cái đáng nhất: câu hỏi của DoD — "category nào cải thiện
nhiều nhất" — **không có câu trả lời** trên `golden_v1`, và nó không có câu trả
lời kể cả khi bỏ hiệu chỉnh đa so sánh.
"""

from __future__ import annotations

import pytest

from pipeline.eval.compare import MIN_TAIL_RESAMPLES, RunScores
from pipeline.eval.contrast import (
    SCALE_FREE_METRICS,
    ContrastRow,
    format_contrast,
    group_contrast,
    group_deltas,
    is_scale_free,
    unpaired_bootstrap,
)


def _run(
    name: str,
    hits: dict[str, list[float]],
    *,
    labels: dict[str, int] | None = None,
    metric: str = "hit_rate@5",
) -> RunScores:
    """Dựng một `RunScores` từ `{category: [điểm từng câu]}`."""
    scores: dict[str, dict[str, float]] = {}
    n_relevant: dict[str, int] = {}
    category: dict[str, str] = {}
    for cat, values in hits.items():
        for i, value in enumerate(values):
            qid = f"{cat}-{i}"
            scores[qid] = {metric: value}
            n_relevant[qid] = (labels or {}).get(cat, 1)
            category[qid] = cat
    return RunScores(
        name=name,
        scores=scores,
        n_relevant=n_relevant,
        relevant_digest=dict.fromkeys(scores, "d"),
        category=category,
        lang=dict.fromkeys(scores, "vi"),
    )


class TestTheAnswerIsASetNotAName:
    """ "Nhóm nào cải thiện nhiều nhất" là phép chọn cực đại, y như `W2-08`."""

    def test_two_groups_within_noise_are_both_in_the_set(self) -> None:
        base = _run("base", {"a": [0.0] * 40, "b": [0.0] * 40})
        # a sửa được 24/40, b sửa được 22/40 — chênh 2 câu.
        cand = _run(
            "cand",
            {"a": [1.0] * 24 + [0.0] * 16, "b": [1.0] * 22 + [0.0] * 18},
        )
        result = group_contrast(base, cand, "category", "hit_rate@5")

        assert result.top.group == "a"
        assert result.members == ("a", "b")
        assert result.beaten == ()

    def test_a_group_far_enough_ahead_is_reported_as_ahead(self) -> None:
        base = _run("base", {"a": [0.0] * 40, "b": [0.0] * 40})
        cand = _run("cand", {"a": [1.0] * 40, "b": [0.0] * 40})
        result = group_contrast(base, cand, "category", "hit_rate@5")

        assert result.beaten == ("b",)
        assert result.members == ("a",)


class TestGroupsAreDisjointSoTheBootstrapIsUnpaired:
    """Hai nhóm không có cặp nào để ghép — ghép là bịa ra tương quan."""

    def test_resampling_keeps_each_group_at_its_own_size(self) -> None:
        # Nhóm phải có mẫu số riêng: bên 4 câu chỉ nhận 5 giá trị trung bình.
        low, high, _, _ = unpaired_bootstrap([1.0] * 4, [0.0] * 40, iterations=2000)
        assert low == high == 1.0

    def test_the_smaller_group_widens_the_interval(self) -> None:
        wide = unpaired_bootstrap([1.0, 0.0] * 2, [1.0, 0.0] * 20, iterations=2000)
        narrow = unpaired_bootstrap([1.0, 0.0] * 20, [1.0, 0.0] * 20, iterations=2000)
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])

    def test_empty_group_is_an_error_not_an_empty_interval(self) -> None:
        with pytest.raises(ValueError, match="ít nhất một câu"):
            unpaired_bootstrap([], [1.0], iterations=100)


class TestMetricsWithALabelDenominatorCannotBeRanked:
    """Hàng rào mới của `W2-09` — `compare.py` không có chỗ nào thấy được nó."""

    def test_ndcg_is_refused_when_groups_have_different_label_counts(self) -> None:
        base = _run(
            "base", {"a": [0.0] * 20, "b": [0.0] * 20}, labels={"a": 1, "b": 3}, metric="ndcg@10"
        )
        cand = _run(
            "cand", {"a": [1.0] * 20, "b": [0.5] * 20}, labels={"a": 1, "b": 3}, metric="ndcg@10"
        )
        result = group_contrast(base, cand, "category", "ndcg@10", iterations=500)

        assert not result.scale_free
        assert all(not row.comparable for row in result.rows)
        assert result.unresolved == ("b",)

    def test_refusal_is_a_property_of_the_metric_not_of_the_data(self) -> None:
        # Cùng số nhãn hai bên vẫn từ chối: thang đo của `recall@k` phụ thuộc số
        # nhãn về NGUYÊN TẮC, và một luật bật/tắt theo dữ liệu là luật chọn được.
        base = _run(
            "base", {"a": [0.0] * 20, "b": [0.0] * 20}, labels={"a": 2, "b": 2}, metric="recall@5"
        )
        cand = _run(
            "cand", {"a": [1.0] * 20, "b": [0.5] * 20}, labels={"a": 2, "b": 2}, metric="recall@5"
        )
        assert not group_contrast(base, cand, "category", "recall@5", iterations=500).scale_free

    def test_hit_rate_and_mrr_and_precision_survive(self) -> None:
        for metric in SCALE_FREE_METRICS:
            assert is_scale_free(metric), metric
        for metric in ("recall@1", "recall@20", "ndcg@10", "map@20"):
            assert not is_scale_free(metric), metric

    def test_refused_rows_never_enter_the_winner_set(self) -> None:
        """Đúng lỗi đắt nhất của `W2-08`, ghim lại ở trục nhóm."""
        base = _run(
            "base", {"a": [0.0] * 20, "b": [0.0] * 20}, labels={"a": 1, "b": 3}, metric="ndcg@10"
        )
        cand = _run(
            "cand", {"a": [1.0] * 20, "b": [0.9] * 20}, labels={"a": 1, "b": 3}, metric="ndcg@10"
        )
        result = group_contrast(base, cand, "category", "ndcg@10", iterations=500)

        assert "b" not in result.members
        assert "b" not in result.tied


class TestUnderpoweredHereMeansSomethingElseThanInCompare:
    """Cùng tên, hai nguyên nhân, hai cách chữa — nên phải tách."""

    def test_a_thin_tail_is_flagged_and_it_is_curable_by_raising_b(self) -> None:
        # `ci_low` phải nằm ngoài một bước lưới (1/40 = 0,025), nếu không thì
        # hàng bị `ci_within_resolution` bắt trước và test đo nhầm cờ.
        thin = ContrastRow(
            top="a",
            other="b",
            metric="hit_rate@5",
            gap=0.2,
            n_top=40,
            n_other=40,
            ci_low=0.05,
            ci_high=0.4,
            tail=MIN_TAIL_RESAMPLES - 1,
            min_increment=1.0,
        )
        thick = ContrastRow(
            top="a",
            other="b",
            metric="hit_rate@5",
            gap=0.2,
            n_top=40,
            n_other=40,
            ci_low=0.05,
            ci_high=0.4,
            tail=MIN_TAIL_RESAMPLES,
            min_increment=1.0,
        )
        assert thin.underpowered and thin.verdict == "KHÔNG KẾT LUẬN"
        assert not thick.underpowered and thick.verdict == "hơn thật"


class TestTheGridStepComesFromTheSmallerGroup:
    """`gap` là hiệu hai trung bình, nên bên nhỏ hơn quyết định bước thô nhất."""

    def test_four_question_group_has_a_quarter_point_step(self) -> None:
        row = ContrastRow(
            top="a",
            other="table_lookup",
            metric="hit_rate@5",
            gap=0.15,
            n_top=43,
            n_other=4,
            ci_low=0.01,
            ci_high=0.6,
            tail=250,
            min_increment=1.0,
        )
        assert row.grid_step == pytest.approx(0.25)
        assert row.ci_within_resolution
        assert row.verdict == "KHÔNG KẾT LUẬN"

    def test_a_finer_metric_gets_a_finer_step(self) -> None:
        row = ContrastRow(
            top="a",
            other="b",
            metric="precision@20",
            gap=0.15,
            n_top=43,
            n_other=40,
            ci_low=0.01,
            ci_high=0.6,
            tail=250,
            min_increment=0.05,
        )
        assert row.grid_step == pytest.approx(0.05 / 40)
        assert not row.ci_within_resolution


class TestRankingUsesDeltaNotTheFinalScore:
    """Nhóm điểm cao nhất và nhóm cải thiện nhiều nhất là hai nhóm khác nhau."""

    def test_the_group_that_ends_highest_is_not_the_group_that_gained_most(self) -> None:
        base = _run("base", {"cao": [1.0] * 20, "thap": [0.0] * 20})
        cand = _run("cand", {"cao": [1.0] * 20, "thap": [1.0] * 16 + [0.0] * 4})
        ranked = group_deltas(base, cand, "category", "hit_rate@5")

        assert ranked[0].group == "thap"
        assert ranked[0].candidate < ranked[1].candidate


class TestTheTableSaysWhatMadeEachRowUnreadable:
    def test_refusal_prints_the_scale_ratio(self) -> None:
        base = _run(
            "base", {"a": [0.0] * 20, "b": [0.0] * 20}, labels={"a": 1, "b": 3}, metric="ndcg@10"
        )
        cand = _run(
            "cand", {"a": [1.0] * 20, "b": [0.5] * 20}, labels={"a": 1, "b": 3}, metric="ndcg@10"
        )
        text = format_contrast(group_contrast(base, cand, "category", "ndcg@10", iterations=500))
        assert "3.00×" in text
        assert "không xếp hạng được" in text

    def test_a_scale_free_table_reports_the_set(self) -> None:
        base = _run("base", {"a": [0.0] * 40, "b": [0.0] * 40})
        cand = _run("cand", {"a": [1.0] * 24 + [0.0] * 16, "b": [1.0] * 22 + [0.0] * 18})
        text = format_contrast(group_contrast(base, cand, "category", "hit_rate@5"))
        assert "Tập cải thiện nhiều nhất" in text
        assert "không phân biệt được" in text


class TestCorrectionIsNotWhatMakesGroupsTie:
    """Đối chứng đã chạy trên dữ liệu thật: bỏ hiệu chỉnh vẫn hoà."""

    def test_turning_correction_off_changes_alpha_but_not_this_verdict(self) -> None:
        # Ba nhóm, vì với hai nhóm thì `family = 1` và hiệu chỉnh không làm gì —
        # test hai nhóm sẽ "xanh" mà không kiểm được điều nó định kiểm.
        base = _run("base", {"a": [0.0] * 40, "b": [0.0] * 40, "c": [0.0] * 40})
        cand = _run(
            "cand",
            {
                "a": [1.0] * 24 + [0.0] * 16,
                "b": [1.0] * 22 + [0.0] * 18,
                "c": [1.0] * 21 + [0.0] * 19,
            },
        )
        # KHÔNG nêu `iterations`: hiệu chỉnh đưa α về 0,025 và `resolve_iterations`
        # phải tự nâng `B` cho đủ đuôi. Nêu một `B` nhỏ cho nhanh thì cả hai hàng
        # thành `KHÔNG KẾT LUẬN` và test "xanh" mà không kiểm được gì — đúng quyết
        # định (b) của `W2-09` tự hiện ra trong test của chính nó.
        strict = group_contrast(base, cand, "category", "hit_rate@5")
        loose = group_contrast(base, cand, "category", "hit_rate@5", correct=False)

        assert strict.alpha < loose.alpha
        assert strict.members == loose.members == ("a", "b", "c")

    def test_one_group_is_not_a_comparison(self) -> None:
        base = _run("base", {"a": [0.0] * 10})
        cand = _run("cand", {"a": [1.0] * 10})
        with pytest.raises(ValueError, match="không có gì để so"):
            group_contrast(base, cand, "category", "hit_rate@5", iterations=100)
