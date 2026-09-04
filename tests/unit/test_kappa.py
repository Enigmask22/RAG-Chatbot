"""`W5-04` — thống kê đồng thuận, đối chiếu với giá trị tính tay.

Mọi con số kỳ vọng ở đây được dẫn ra trong docstring của chính bài test, đủ để
kiểm lại bằng máy tính bỏ túi. Đó là điều kiện để `judge-calibration.md` là một
báo cáo chứ không phải một lời khai: nếu module tính sai, bài test phải đỏ chứ
không phải "cả hai cùng sai theo cùng một cách".
"""

from __future__ import annotations

import math

import pytest

from pipeline.eval.kappa import (
    Confusion,
    Pair,
    accuracy_of,
    bootstrap_ci,
    cohen_kappa,
    confusion,
    expected_agreement,
    observed_agreement,
    pabak,
    per_label,
    rate_of,
)

YN = ("YES", "NO")
FAITH = ("SUPPORTED", "CONTRADICTED", "NOT_FOUND", "NO_CLAIM")


def grid(labels: tuple[str, ...], cells: dict[tuple[str, str], int]) -> list[Pair]:
    """Dựng danh sách cặp từ một ma trận đếm — để test viết được như bảng giấy."""
    pairs: list[Pair] = []
    for (a, b), count in cells.items():
        assert a in labels and b in labels
        pairs.extend(Pair(a=a, b=b, ref=f"{a}/{b}#{i}") for i in range(count))
    return pairs


class TestCohenKappaAgainstHandComputedValues:
    def test_textbook_binary_example(self) -> None:
        """20/5/10/15 trên 50 mục.

        Po = (20+15)/50 = 0,70
        biên người  = 25 YES, 25 NO;  biên judge = 30 YES, 20 NO
        Pe = (25·30 + 25·20)/50² = (750+500)/2500 = 0,50
        κ  = (0,70 − 0,50)/(1 − 0,50) = **0,40**
        """
        cm = confusion(
            grid(YN, {("YES", "YES"): 20, ("YES", "NO"): 5, ("NO", "YES"): 10, ("NO", "NO"): 15}),
            YN,
        )
        assert observed_agreement(cm) == pytest.approx(0.70)
        assert expected_agreement(cm) == pytest.approx(0.50)
        assert cohen_kappa(cm) == pytest.approx(0.40)

    def test_total_disagreement_is_minus_one(self) -> None:
        """0/10/10/0: Po = 0, mọi biên = 0,5 ⇒ Pe = 0,50 ⇒ κ = −1,0.

        Kappa âm không phải lỗi tính — nó nghĩa là hai bên đồng thuận **kém hơn**
        cả khi gán nhãn ngẫu nhiên.
        """
        cm = confusion(grid(YN, {("YES", "NO"): 10, ("NO", "YES"): 10}), YN)
        assert observed_agreement(cm) == pytest.approx(0.0)
        assert cohen_kappa(cm) == pytest.approx(-1.0)

    def test_the_kappa_paradox_high_agreement_low_kappa(self) -> None:
        """85/5/5/5: Po = 0,90 nhưng κ = 0,444 vì biên độ lệch 90/10.

        Pe = (90·90 + 10·10)/100² = (8100+100)/10000 = 0,82
        κ  = (0,90 − 0,82)/0,18 = 0,08/0,18 = **0,4444…**
        PABAK = 2·0,90 − 1 = **0,80**

        Khoảng cách 0,44 vs 0,80 chính là độ lớn của nghịch lý. Báo cáo nào chỉ
        in một trong hai đều đang kể nửa câu chuyện.
        """
        cm = confusion(
            grid(YN, {("YES", "YES"): 85, ("YES", "NO"): 5, ("NO", "YES"): 5, ("NO", "NO"): 5}),
            YN,
        )
        assert observed_agreement(cm) == pytest.approx(0.90)
        assert expected_agreement(cm) == pytest.approx(0.82)
        assert cohen_kappa(cm) == pytest.approx(0.08 / 0.18)
        assert pabak(cm) == pytest.approx(0.80)

    def test_kappa_is_symmetric_in_the_two_raters(self) -> None:
        pairs = grid(YN, {("YES", "YES"): 12, ("YES", "NO"): 7, ("NO", "YES"): 3, ("NO", "NO"): 18})
        swapped = [Pair(a=p.b, b=p.a) for p in pairs]
        assert cohen_kappa(confusion(pairs, YN)) == pytest.approx(
            cohen_kappa(confusion(swapped, YN))
        )


class TestKappaIsUndefinedNotPerfect:
    def test_both_raters_use_one_label_gives_none(self) -> None:
        """Ca mặc định của tập này, không phải ca hiếm.

        Nếu cả hai bên gán `SUPPORTED` cho cả 30 mục thì Po = 1 và Pe = 1: κ là
        0/0. Trả `1.0` sẽ in ra "κ = 1,00 ✅" cho một phép đo **không chứa thông
        tin nào**.
        """
        cm = confusion(grid(FAITH, {("SUPPORTED", "SUPPORTED"): 30}), FAITH)
        assert observed_agreement(cm) == pytest.approx(1.0)
        assert expected_agreement(cm) == pytest.approx(1.0)
        assert cohen_kappa(cm) is None
        assert pabak(cm) == pytest.approx(1.0), "PABAK vẫn đọc được ca này"

    def test_empty_input_gives_none_everywhere(self) -> None:
        cm = confusion([], FAITH)
        assert cm.total == 0
        assert observed_agreement(cm) is None
        assert expected_agreement(cm) is None
        assert cohen_kappa(cm) is None
        assert pabak(cm) is None
        assert accuracy_of([]) is None

    def test_a_single_disagreement_swings_kappa_far_on_skewed_margins(self) -> None:
        """Vì sao 50 mẫu ngẫu nhiên đều không đủ: mẫu số `1 − Pe` rất nhỏ.

        49 trùng `SUPPORTED` + 1 bất đồng ⇒ κ nhảy từ *không xác định* xuống một
        con số thấp. Trên biên độ 98/2 thì một mục đổi nhãn kéo κ đi hơn 0,3.
        """
        base = grid(FAITH, {("SUPPORTED", "SUPPORTED"): 49})
        assert cohen_kappa(confusion(base, FAITH)) is None
        one_off = [*base, Pair(a="NOT_FOUND", b="SUPPORTED")]
        kappa = cohen_kappa(confusion(one_off, FAITH))
        assert kappa is not None
        assert kappa == pytest.approx(0.0), "một bất đồng duy nhất, không có ô chéo hiếm ⇒ κ=0"
        two_off = [*base, Pair(a="NOT_FOUND", b="NOT_FOUND")]
        kappa2 = cohen_kappa(confusion(two_off, FAITH))
        assert kappa2 == pytest.approx(1.0), "cùng cỡ mẫu, đổi một ô ⇒ κ nhảy 0 → 1"


class TestStratifiedWeighting:
    """Ví dụ hiệu chỉnh trên đúng biên độ thật của `w5-answers-v1`.

    Quần thể: 402 `SUPPORTED`, 5 `NOT_FOUND` (bỏ `NO_CLAIM` cho gọn).
    Mẫu: 30 từ tầng `SUPPORTED` (w = 402/30 = 13,4), 5 từ tầng `NOT_FOUND` (w = 1).
    Người bất đồng 1 mục ở mỗi tầng.

    Ma trận có trọng số (hàng = người, cột = judge):

        |                | judge S | judge NF |
        | người S        |  388,6  |    1,0   |  → 389,6
        | người NF       |   13,4  |    4,0   |  →  17,4
        |                |  402,0  |    5,0   |  → 407,0

    Po = (388,6 + 4,0)/407 = **0,9646**
    Pe = (389,6·402 + 17,4·5)/407² = 156 706,2/165 649 = **0,9460**
    κ  = (0,9646 − 0,9460)/(1 − 0,9460) = **0,3446**
    """

    @staticmethod
    def sample() -> list[Pair]:
        w = 402 / 30
        pairs = [
            *(Pair("SUPPORTED", "SUPPORTED", stratum="SUPPORTED", weight=w) for _ in range(29)),
            Pair("NOT_FOUND", "SUPPORTED", stratum="SUPPORTED", weight=w),
            *(Pair("NOT_FOUND", "NOT_FOUND", stratum="NOT_FOUND", weight=1.0) for _ in range(4)),
            Pair("SUPPORTED", "NOT_FOUND", stratum="NOT_FOUND", weight=1.0),
        ]
        assert len(pairs) == 35
        return pairs

    def test_weighted_matrix_reproduces_the_population_margins(self) -> None:
        cm = confusion(self.sample(), FAITH)
        assert cm.total == pytest.approx(407.0)
        assert cm.n_items == 35, "35 quan sát thật, không phải 407"
        assert cm.cell("SUPPORTED", "SUPPORTED") == pytest.approx(388.6)
        assert cm.cell("NOT_FOUND", "SUPPORTED") == pytest.approx(13.4)
        judge_totals = dict(zip(cm.labels, cm.col_totals, strict=True))
        assert judge_totals["SUPPORTED"] == pytest.approx(402.0)
        assert judge_totals["NOT_FOUND"] == pytest.approx(5.0)

    def test_population_kappa_is_hand_computed(self) -> None:
        cm = confusion(self.sample(), FAITH)
        assert observed_agreement(cm) == pytest.approx(0.9646191646, abs=1e-9)
        assert expected_agreement(cm) == pytest.approx(0.9460135588, abs=1e-9)
        assert cohen_kappa(cm) == pytest.approx(0.3446347900, abs=1e-9)

    def test_sample_kappa_and_population_kappa_differ(self) -> None:
        """Bỏ trọng số đi là trả lời một câu hỏi khác — và ra số khác hẳn."""
        unweighted = [Pair(p.a, p.b, stratum=p.stratum) for p in self.sample()]
        sample_kappa = cohen_kappa(confusion(unweighted, FAITH))
        pop_kappa = cohen_kappa(confusion(self.sample(), FAITH))
        assert sample_kappa is not None and pop_kappa is not None
        assert not math.isclose(sample_kappa, pop_kappa, abs_tol=0.05)

    def test_faithfulness_recomputed_from_each_side(self) -> None:
        """Cùng một công thức, đổi nguồn nhãn — nên hai số so trực tiếp được.

        Judge: 402/407 = 0,98771 (đúng con số `W5-01` đã báo cáo).
        Người: 389,6/407 = 0,95725.
        """
        pairs = self.sample()
        den = ("SUPPORTED", "CONTRADICTED", "NOT_FOUND")
        judge = rate_of(pairs, side="b", numerator=("SUPPORTED",), denominator=den)
        human = rate_of(pairs, side="a", numerator=("SUPPORTED",), denominator=den)
        assert judge == pytest.approx(402 / 407, abs=1e-9)
        assert human == pytest.approx(389.6 / 407, abs=1e-9)

    def test_no_claim_is_excluded_from_the_faithfulness_denominator(self) -> None:
        pairs = [
            Pair("SUPPORTED", "SUPPORTED"),
            Pair("NOT_FOUND", "NOT_FOUND"),
            *(Pair("NO_CLAIM", "NO_CLAIM") for _ in range(20)),
        ]
        den = ("SUPPORTED", "CONTRADICTED", "NOT_FOUND")
        assert rate_of(pairs, side="a", numerator=("SUPPORTED",), denominator=den) == pytest.approx(
            0.5
        ), "20 câu NO_CLAIM không được kéo tỉ lệ về 1/22"


class TestPerLabel:
    def test_precision_and_recall_are_hand_computed(self) -> None:
        """Judge gọi `NOT_FOUND` 4 lần, đúng 3; người gán `NOT_FOUND` 6 lần.

        precision(NOT_FOUND) = 3/4 = 0,75 · recall = 3/6 = 0,50 · F1 = 0,60
        """
        pairs = grid(
            FAITH,
            {
                ("SUPPORTED", "SUPPORTED"): 40,
                ("NOT_FOUND", "NOT_FOUND"): 3,
                ("NOT_FOUND", "SUPPORTED"): 3,
                ("SUPPORTED", "NOT_FOUND"): 1,
            },
        )
        stats = per_label(confusion(pairs, FAITH))["NOT_FOUND"]
        assert stats["precision"] == pytest.approx(0.75)
        assert stats["recall"] == pytest.approx(0.50)
        assert stats["f1"] == pytest.approx(0.60)
        assert stats["support_a"] == pytest.approx(6.0)
        assert stats["support_b"] == pytest.approx(4.0)

    def test_label_nobody_used_gives_none_not_zero(self) -> None:
        """`CONTRADICTED` không xuất hiện lần nào: 0/0, không phải 0%.

        In `0.00` cho nhãn không ai dùng sẽ đọc thành "judge trượt sạch nhãn
        này", trong khi sự thật là chưa có gì để đo.
        """
        stats = per_label(confusion(grid(FAITH, {("SUPPORTED", "SUPPORTED"): 10}), FAITH))
        assert stats["CONTRADICTED"]["precision"] is None
        assert stats["CONTRADICTED"]["recall"] is None
        assert stats["CONTRADICTED"]["f1"] is None

    def test_kappa_cannot_distinguish_two_opposite_failure_modes(self) -> None:
        """Cùng κ, hai kiểu sai trái ngược — lý do `per_label` phải tồn tại.

        A: judge bỏ sót `NOT_FOUND` (recall thấp, precision cao).
        B: judge báo động giả `NOT_FOUND` (precision thấp, recall cao).
        Hai ma trận là chuyển vị của nhau nên κ bằng nhau, nhưng một cái làm
        faithfulness **cao hơn** thực tế còn cái kia làm nó **thấp hơn**.
        """
        miss = grid(
            FAITH,
            {
                ("SUPPORTED", "SUPPORTED"): 40,
                ("NOT_FOUND", "SUPPORTED"): 5,
                ("NOT_FOUND", "NOT_FOUND"): 5,
            },
        )
        false_alarm = [Pair(a=p.b, b=p.a) for p in miss]
        assert cohen_kappa(confusion(miss, FAITH)) == pytest.approx(
            cohen_kappa(confusion(false_alarm, FAITH))
        )
        assert per_label(confusion(miss, FAITH))["NOT_FOUND"]["recall"] == pytest.approx(0.5)
        assert per_label(confusion(miss, FAITH))["NOT_FOUND"]["precision"] == pytest.approx(1.0)
        assert per_label(confusion(false_alarm, FAITH))["NOT_FOUND"]["recall"] == pytest.approx(1.0)
        assert per_label(confusion(false_alarm, FAITH))["NOT_FOUND"]["precision"] == pytest.approx(
            0.5
        )


class TestGuardRails:
    def test_label_outside_the_declared_set_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="ngoài tập khai báo"):
            confusion([Pair("SUPPORTED", "MAYBE", ref="x1")], FAITH)

    def test_the_offending_ref_is_named(self) -> None:
        with pytest.raises(ValueError, match="x1"):
            confusion([Pair("SUPPORTED", "MAYBE", ref="x1")], FAITH)

    def test_duplicate_labels_rejected(self) -> None:
        with pytest.raises(ValueError, match="trùng"):
            confusion([], ("A", "B", "A"))

    def test_non_positive_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="weight"):
            Pair("SUPPORTED", "SUPPORTED", weight=0.0)

    def test_rate_of_rejects_numerator_outside_denominator(self) -> None:
        with pytest.raises(ValueError, match="tập con"):
            rate_of([], side="a", numerator=("NO_CLAIM",), denominator=("SUPPORTED",))

    def test_rate_of_rejects_unknown_side(self) -> None:
        with pytest.raises(ValueError, match="side"):
            rate_of([], side="human", numerator=("A",), denominator=("A",))

    def test_confusion_as_dict_reports_both_counts(self) -> None:
        cm = confusion([Pair("SUPPORTED", "SUPPORTED", weight=13.4)], FAITH)
        payload = cm.as_dict()
        assert payload["n_items"] == 1
        assert payload["total_weight"] == pytest.approx(13.4)


class TestBootstrap:
    @staticmethod
    def skewed() -> list[Pair]:
        return [
            *(Pair("SUPPORTED", "SUPPORTED", stratum="S") for _ in range(45)),
            *(Pair("NOT_FOUND", "SUPPORTED", stratum="NF") for _ in range(2)),
            *(Pair("NOT_FOUND", "NOT_FOUND", stratum="NF") for _ in range(3)),
        ]

    def test_same_seed_gives_the_same_interval(self) -> None:
        a = bootstrap_ci(self.skewed(), lambda p: accuracy_of(p), n_resamples=200, seed=7)
        b = bootstrap_ci(self.skewed(), lambda p: accuracy_of(p), n_resamples=200, seed=7)
        assert a == b

    def test_different_seed_draws_a_different_sequence(self) -> None:
        """Kiểm trên **dãy** lần lấy mẫu, không trên hai đầu khoảng.

        Bản đầu của bài này so `(lo, hi)` và nó xanh giả: độ chính xác trên 50
        mục chỉ nhận bội của 0,02, nên hai seed khác nhau vẫn rơi vào đúng hai
        phân vị ấy. Một bài test "seed có tác dụng" mà bất động trước seed thì
        không kiểm gì cả.
        """

        def trace(pairs: object) -> float:
            value = accuracy_of(pairs)  # type: ignore[arg-type]
            assert value is not None
            seen.append(value)
            return value

        seen: list[float] = []
        bootstrap_ci(self.skewed(), trace, n_resamples=200, seed=7)
        first, seen = seen, []
        bootstrap_ci(self.skewed(), trace, n_resamples=200, seed=8)
        assert first != seen

    def test_point_estimate_lies_inside_the_interval(self) -> None:
        out = bootstrap_ci(self.skewed(), lambda p: accuracy_of(p), n_resamples=500, seed=11)
        assert out["lo"] <= out["point"] <= out["hi"]
        assert out["point"] == pytest.approx(48 / 50)

    def test_resampling_preserves_each_stratum_size(self) -> None:
        """Nếu lấy lại mẫu trên toàn bộ 50 cặp thì có lần tầng `NF` rỗng.

        Kiểm bằng một `stat` chỉ đếm phần tử của tầng hiếm: mọi lần lấy mẫu phải
        cho đúng 5, không bao giờ 0.
        """
        seen: list[float] = []

        def count_nf(pairs: object) -> float:
            n = float(len([p for p in pairs if p.stratum == "NF"]))  # type: ignore[attr-defined]
            seen.append(n)
            return n

        bootstrap_ci(self.skewed(), count_nf, n_resamples=100, seed=3)
        assert set(seen) == {5.0}

    def test_undefined_resamples_are_counted_not_dropped_silently(self) -> None:
        """κ không xác định ở một số lần lấy mẫu là một sự thật phải in ra."""
        pairs = [
            *(Pair("SUPPORTED", "SUPPORTED", stratum="S") for _ in range(9)),
            Pair("NOT_FOUND", "NOT_FOUND", stratum="S"),
        ]
        out = bootstrap_ci(
            pairs, lambda p: cohen_kappa(confusion(p, FAITH)), n_resamples=300, seed=5
        )
        assert out["n_undefined"] > 0, "phải có lần lấy mẫu trúng toàn SUPPORTED"
        assert out["n_undefined"] < 300
        assert out["lo"] is not None

    def test_all_undefined_yields_no_interval_rather_than_a_fake_one(self) -> None:
        pairs = [Pair("SUPPORTED", "SUPPORTED", stratum="S") for _ in range(10)]
        out = bootstrap_ci(
            pairs, lambda p: cohen_kappa(confusion(p, FAITH)), n_resamples=50, seed=5
        )
        assert out["n_undefined"] == 50
        assert out["lo"] is None and out["hi"] is None and out["point"] is None

    def test_invalid_level_rejected(self) -> None:
        with pytest.raises(ValueError, match="level"):
            bootstrap_ci([Pair("A", "A")], lambda p: 1.0, level=1.0)

    def test_invalid_resample_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_resamples"):
            bootstrap_ci([Pair("A", "A")], lambda p: 1.0, n_resamples=0)


class TestConfusionAccessors:
    def test_cell_lookup_matches_matrix_indexing(self) -> None:
        cm = confusion(grid(YN, {("YES", "NO"): 3}), YN)
        assert cm.cell("YES", "NO") == 3.0
        assert cm.cell("NO", "YES") == 0.0

    def test_row_and_column_totals_sum_to_the_grand_total(self) -> None:
        cm = confusion(
            grid(YN, {("YES", "YES"): 4, ("YES", "NO"): 6, ("NO", "YES"): 1, ("NO", "NO"): 9}), YN
        )
        assert sum(cm.row_totals) == pytest.approx(cm.total)
        assert sum(cm.col_totals) == pytest.approx(cm.total)
        assert isinstance(cm, Confusion)
