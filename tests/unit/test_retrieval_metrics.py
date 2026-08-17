"""W1-12 — metric truy hồi, đối chiếu với giá trị tính tay.

Mọi kỳ vọng ở đây được viết dưới dạng công thức tường minh với thứ hạng ghi rõ,
hoặc dưới dạng phân số tính tay. Cố ý **không** gọi lại hàm đang test để sinh ra
kỳ vọng — làm vậy thì bài test chỉ chứng minh hàm bằng chính nó.
"""

from __future__ import annotations

import math

import pytest

from pipeline.eval.metrics import (
    average_precision_at_k,
    dedupe_preserving_order,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Trúng ở hạng 2 và hạng 4.
RETRIEVED = ["c1", "c2", "c3", "c4", "c5"]
RELEVANT = ["c2", "c4"]


class TestRecall:
    @pytest.mark.parametrize(
        ("k", "expected"),
        [
            (1, 0.0),  # top1 = [c1], trúng 0/2
            (2, 0.5),  # top2 = [c1,c2], trúng 1/2
            (4, 1.0),  # top4 chứa cả c2 và c4
            (10, 1.0),  # k lớn hơn số kết quả vẫn hợp lệ
        ],
    )
    def test_hand_computed(self, k: int, expected: float) -> None:
        assert recall_at_k(RETRIEVED, RELEVANT, k) == pytest.approx(expected)


class TestPrecision:
    @pytest.mark.parametrize(
        ("k", "expected"),
        [
            (1, 0.0),
            (2, 1 / 2),
            (4, 2 / 4),
            (5, 2 / 5),
            (10, 2 / 10),  # mẫu số là k: trả về ít hơn k là thất bại của retriever
        ],
    )
    def test_hand_computed(self, k: int, expected: float) -> None:
        assert precision_at_k(RETRIEVED, RELEVANT, k) == pytest.approx(expected)


class TestHitRate:
    @pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 1.0), (5, 1.0)])
    def test_hand_computed(self, k: int, expected: float) -> None:
        assert hit_rate_at_k(RETRIEVED, RELEVANT, k) == pytest.approx(expected)


class TestReciprocalRank:
    def test_first_hit_at_rank_two(self) -> None:
        assert reciprocal_rank(RETRIEVED, RELEVANT) == pytest.approx(1 / 2)

    def test_first_hit_at_rank_one(self) -> None:
        assert reciprocal_rank(["c2", "c1"], RELEVANT) == pytest.approx(1.0)

    def test_no_hit_within_k(self) -> None:
        assert reciprocal_rank(RETRIEVED, RELEVANT, k=1) == pytest.approx(0.0)

    def test_mean_over_queries(self) -> None:
        cases = [
            (["c2", "c9"], ["c2"]),  # 1/1
            (["c9", "c9b", "c4"], ["c4"]),  # 1/3
        ]
        assert mean_reciprocal_rank(cases) == pytest.approx((1.0 + 1 / 3) / 2)

    def test_mean_ignores_unanswerable(self) -> None:
        cases: list[tuple[list[str], list[str]]] = [(["c2"], ["c2"]), (["c9"], [])]
        assert mean_reciprocal_rank(cases) == pytest.approx(1.0)


class TestNDCG:
    def test_hand_computed_with_gap(self) -> None:
        # DCG  = 1/log2(2+1) + 1/log2(4+1)   (trúng ở hạng 2 và 4)
        # IDCG = 1/log2(1+1) + 1/log2(2+1)   (lý tưởng: hạng 1 và 2)
        dcg = 1 / math.log2(3) + 1 / math.log2(5)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        assert ndcg_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(dcg / idcg)

    def test_perfect_ranking_is_one(self) -> None:
        assert ndcg_at_k(["c2", "c4", "c1"], RELEVANT, 5) == pytest.approx(1.0)

    def test_reversed_ranking_scores_lower(self) -> None:
        """Đảo thứ hạng phải làm điểm tụt — đây là điều phân biệt nDCG với recall."""
        good = ndcg_at_k(["c2", "c4", "c1", "c3", "c5"], RELEVANT, 5)
        bad = ndcg_at_k(["c1", "c3", "c5", "c2", "c4"], RELEVANT, 5)
        assert good is not None and bad is not None
        assert good > bad
        # recall thì không phân biệt được hai trường hợp này
        assert recall_at_k(["c2", "c4", "c1", "c3", "c5"], RELEVANT, 5) == recall_at_k(
            ["c1", "c3", "c5", "c2", "c4"], RELEVANT, 5
        )

    def test_idcg_truncated_at_k(self) -> None:
        # 3 tài liệu liên quan nhưng k=2 → trần lý tưởng chỉ tính 2 tài liệu đầu.
        relevant = ["a", "b", "c"]
        dcg = 1 / math.log2(2) + 1 / math.log2(3)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        assert ndcg_at_k(["a", "b", "z"], relevant, 2) == pytest.approx(dcg / idcg)

    def test_graded_gains(self) -> None:
        gains = {"c2": 1.0, "c4": 3.0}
        dcg = 1.0 / math.log2(3) + 3.0 / math.log2(5)
        idcg = 3.0 / math.log2(2) + 1.0 / math.log2(3)
        assert ndcg_at_k(RETRIEVED, RELEVANT, 5, gains=gains) == pytest.approx(dcg / idcg)

    def test_no_hit_is_zero(self) -> None:
        assert ndcg_at_k(["x", "y"], RELEVANT, 5) == pytest.approx(0.0)


class TestAveragePrecision:
    def test_hand_computed(self) -> None:
        # Trúng ở hạng 2 (precision 1/2) và hạng 4 (precision 2/4); chia cho min(2,5)
        assert average_precision_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx((0.5 + 0.5) / 2)

    def test_perfect_is_one(self) -> None:
        assert average_precision_at_k(["c2", "c4"], RELEVANT, 5) == pytest.approx(1.0)


class TestUnanswerableIsExcluded:
    """Truy vấn không có tài liệu liên quan trả `None`, không trả `0.0`.

    Quy ước thành 0 sẽ kéo tụt điểm một cách vô nghĩa; quy ước thành 1 thì thổi
    phồng. Nhóm này đo riêng bằng refusal correctness ở W5-02.
    """

    @pytest.mark.parametrize(
        "fn", [recall_at_k, precision_at_k, hit_rate_at_k, ndcg_at_k, average_precision_at_k]
    )
    def test_returns_none(self, fn) -> None:  # type: ignore[no-untyped-def]
        assert fn(RETRIEVED, [], 5) is None

    def test_reciprocal_rank_returns_none(self) -> None:
        assert reciprocal_rank(RETRIEVED, []) is None


class TestEdgeCases:
    def test_duplicates_do_not_double_count(self) -> None:
        """Retriever trả cùng một chunk hai lần không được tính công hai lần."""
        assert precision_at_k(["c2", "c2", "c2"], ["c2"], 3) == pytest.approx(1 / 3)
        assert recall_at_k(["c2", "c2"], ["c2", "c4"], 2) == pytest.approx(0.5)

    def test_dedupe_preserves_order(self) -> None:
        assert dedupe_preserving_order(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]

    def test_empty_retrieval_scores_zero_not_none(self) -> None:
        # Không trả gì cũng là một kết quả — phải bị chấm 0, không được bỏ qua.
        assert recall_at_k([], RELEVANT, 5) == pytest.approx(0.0)
        assert ndcg_at_k([], RELEVANT, 5) == pytest.approx(0.0)

    def test_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="k phải"):
            recall_at_k(RETRIEVED, RELEVANT, 0)
