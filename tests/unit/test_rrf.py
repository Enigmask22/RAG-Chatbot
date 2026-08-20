"""Test cho `reciprocal_rank_fusion` — `W2-04`, hàm thuần.

Đây là chỗ duy nhất trong đường truy hồi mà đúng/sai kiểm được bằng **số tính
tay**, nên nó phải được kiểm bằng số tính tay. Một phép hợp nhất sai không hỏng
ồn ào — nó chỉ làm mọi metric tệ đi vài phần trăm, và trên `golden_v1` thì vài
phần trăm nằm dưới ngưỡng phân giải (`TD-11`: 209 câu chỉ phân giải được ≥ 6 điểm
`hit_rate`). Tức là một lỗi ở đây sẽ **không** bị eval bắt.
"""

from __future__ import annotations

import pytest

from rag_core.retrieval import RRF_K, FusedItem, reciprocal_rank_fusion


def _keys(items: list[FusedItem]) -> list[str]:
    return [item.key for item in items]


class TestHandComputed:
    """Số tính tay, không phải số do chính hàm sinh ra."""

    def test_single_list_preserves_order(self) -> None:
        fused = reciprocal_rank_fusion([["a", "b", "c"]])
        assert _keys(fused) == ["a", "b", "c"]
        assert [item.rank for item in fused] == [1, 2, 3]

    def test_single_list_scores(self) -> None:
        fused = reciprocal_rank_fusion([["a", "b"]], k=60)
        assert fused[0].score == pytest.approx(1 / 61)
        assert fused[1].score == pytest.approx(1 / 62)

    def test_agreement_beats_a_single_first_place(self) -> None:
        """Tính chất trung tâm của RRF, và là lý do nó đáng làm.

        `b` đứng hạng 2 ở **cả hai** danh sách: 1/62 + 1/62 = 0,032258.
        `a` đứng hạng 1 ở đúng một danh sách: 1/61 = 0,016393.
        Đồng thuận thắng — dù `a` là hạng nhất của nhánh mạnh hơn.
        """
        fused = reciprocal_rank_fusion([["a", "b"], ["c", "b"]], k=60)
        assert _keys(fused)[0] == "b"
        assert fused[0].score == pytest.approx(2 / 62)

    def test_deep_agreement_beats_shallow_solo(self) -> None:
        """Con số biện minh cho `candidate_k` sâu hơn `top_k`.

        Chunk ở hạng 45 của dense và hạng 3 của sparse: 1/105 + 1/63 = 0,025397.
        Chunk chỉ dense tìm ra ở hạng 2: 1/62 = 0,016129.
        Cái sâu-nhưng-đồng-thuận thắng — nhưng chỉ khi ta lấy đủ sâu để thấy nó.
        """
        dense = [f"d{i}" for i in range(1, 46)]  # "d45" ở hạng 45
        sparse = ["s1", "s2", "d45"]  # "d45" ở hạng 3
        fused = reciprocal_rank_fusion([dense, sparse], k=60)
        d45 = next(item for item in fused if item.key == "d45")
        d2 = next(item for item in fused if item.key == "d2")
        assert d45.score == pytest.approx(1 / 105 + 1 / 63)
        assert d2.score == pytest.approx(1 / 62)
        assert d45.rank < d2.rank

    def test_shallow_pool_hides_that_agreement(self) -> None:
        """Cùng dữ liệu, lấy nông hơn: `d45` biến mất và `d2` lên trước.

        Đây là mặt bù của `candidate_k` nhỏ, đo bằng chính ví dụ ở test trên.
        """
        dense = [f"d{i}" for i in range(1, 21)]  # chỉ tới hạng 20
        fused = reciprocal_rank_fusion([dense, ["s1", "s2", "d45"]], k=60)
        d45 = next(item for item in fused if item.key == "d45")
        d2 = next(item for item in fused if item.key == "d2")
        assert d45.ranks == (None, 3), "dense không còn với tới nó"
        assert d2.rank < d45.rank

    def test_three_lists(self) -> None:
        """Hàm không giả định đúng hai nhánh — `W2-05` sẽ thêm nhánh reranked."""
        fused = reciprocal_rank_fusion([["a"], ["a"], ["a"]], k=60)
        assert fused[0].score == pytest.approx(3 / 61)


class TestKParameter:
    def test_default_is_the_paper_value(self) -> None:
        assert RRF_K == 60

    def test_large_k_flattens_the_gap_between_top_ranks(self) -> None:
        """Ý nghĩa thật của `k`: "tôi tin thứ hạng đến mức nào".

        `k = 0` thì hạng 1 hơn hạng 2 gấp đôi; `k = 60` thì hơn 1,6%.
        """
        flat = reciprocal_rank_fusion([["a", "b"]], k=60)
        steep = reciprocal_rank_fusion([["a", "b"]], k=0)
        assert flat[0].score / flat[1].score == pytest.approx(62 / 61)
        assert steep[0].score / steep[1].score == pytest.approx(2.0)

    def test_k_changes_who_wins(self) -> None:
        """`k` không phải tham số trang trí — nó đảo được kết quả.

        `a` hạng 1 một danh sách; `b` hạng 3 ở cả hai.
        `k = 60`: b = 2/63 = 0,03175 > a = 1/61 = 0,01639 → b thắng.
        `k = 0` : b = 2/3 = 0,6667 > a = 1/1 = 1,0 → **a** thắng.
        """
        lists = [["a", "x", "b"], ["y", "z", "b"]]
        assert _keys(reciprocal_rank_fusion(lists, k=60))[0] == "b"
        assert _keys(reciprocal_rank_fusion(lists, k=0))[0] == "a"

    def test_negative_k_is_refused(self) -> None:
        with pytest.raises(ValueError, match="k phải không âm"):
            reciprocal_rank_fusion([["a"]], k=-1)


class TestEmptyLists:
    def test_one_empty_list_is_valid_not_an_error(self) -> None:
        """Trạng thái **thật** của nhánh sparse khi truy vấn không trùng token
        (`W2-03` có test). Coi nó là lỗi thì mọi truy vấn kiểu đó sẽ chết thay vì
        rơi về nhánh còn lại — tức mất đúng cái lợi mà hợp nhất mang lại."""
        fused = reciprocal_rank_fusion([["a", "b"], []], k=60)
        assert _keys(fused) == ["a", "b"]
        assert fused[0].ranks == (1, None)

    def test_all_lists_empty(self) -> None:
        assert reciprocal_rank_fusion([[], []]) == []

    def test_no_lists_at_all(self) -> None:
        assert reciprocal_rank_fusion([]) == []

    def test_empty_list_does_not_shift_the_other(self) -> None:
        """Danh sách rỗng không được làm lệch điểm của danh sách còn lại."""
        alone = reciprocal_rank_fusion([["a", "b", "c"]])
        with_empty = reciprocal_rank_fusion([["a", "b", "c"], []])
        assert [(i.key, i.score) for i in alone] == [(i.key, i.score) for i in with_empty]


class TestTieBreak:
    """Điểm bằng nhau xảy ra **thường xuyên**, không phải ca biên: một khoá ở hạng
    3 của danh sách A và một khoá khác ở hạng 3 của danh sách B có cùng điểm."""

    def test_ties_are_broken_toward_the_first_list(self) -> None:
        """Nhánh dense được truyền vào trước, và `W2-03` đo được nó mạnh hơn
        (`hit_rate@10` 0,6268 vs 0,5120) — nên khi không phân biệt được thì
        nghiêng về nó là tiên nghiệm đúng, không phải lựa chọn tuỳ tiện."""
        fused = reciprocal_rank_fusion([["d"], ["s"]], k=60)
        assert fused[0].score == pytest.approx(fused[1].score)
        assert _keys(fused) == ["d", "s"]

    def test_swapping_list_order_swaps_the_tie(self) -> None:
        """Chứng minh quy tắc trên thực sự do thứ tự danh sách quyết định."""
        assert _keys(reciprocal_rank_fusion([["s"], ["d"]], k=60)) == ["s", "d"]

    def test_same_list_ties_are_impossible_so_key_order_is_last_resort(self) -> None:
        """Trong cùng một danh sách không thể có hai khoá cùng hạng, nên quy tắc
        cuối (theo chữ) chỉ tới khi hai khoá cùng điểm, cùng `best_rank`, cùng
        danh sách đầu tiên — tức cùng đúng một tập hạng."""
        fused = reciprocal_rank_fusion([["b", "a"], ["a", "b"]], k=60)
        assert fused[0].score == pytest.approx(fused[1].score)
        assert _keys(fused) == ["a", "b"], "hết quy tắc thì xếp theo chữ"

    def test_deterministic_across_calls(self) -> None:
        lists = [["a", "b", "c", "d"], ["d", "c", "b", "a"]]
        first = _keys(reciprocal_rank_fusion(lists, k=60))
        for _ in range(20):
            assert _keys(reciprocal_rank_fusion(lists, k=60)) == first

    def test_deterministic_regardless_of_dict_insertion_history(self) -> None:
        """Kết quả không được phụ thuộc thứ tự khoá gặp lần đầu — nếu phụ thuộc
        thì hai lần chạy trên cùng dữ liệu vẫn có thể khác nhau khi Qdrant trả về
        cùng tập point theo thứ tự khác."""
        a = reciprocal_rank_fusion([["x", "y"], ["y", "x"]], k=60)
        b = reciprocal_rank_fusion([["y", "x"], ["x", "y"]], k=60)
        assert [i.score for i in a] == [i.score for i in b]


class TestScaleInvariance:
    def test_only_rank_matters_not_score(self) -> None:
        """Lý do cả tầng này tồn tại: hàm **không nhận** điểm gốc làm đầu vào.

        Dense cho cosine ∈ [−1,1], sparse cho dot product không có trần (đo thật:
        0,6682 vs 0,2938). Không có cách chuẩn hoá nào không đưa vào một tham số
        ẩn phụ thuộc kết quả. Thứ hạng thì không có vấn đề đó — và cách chắc chắn
        nhất để giữ tính chất ấy là không cho điểm đi vào hàm.
        """
        import inspect

        params = set(inspect.signature(reciprocal_rank_fusion).parameters)
        assert "scores" not in params
        assert params == {"rankings", "k", "weights", "limit"}


class TestWeights:
    def test_equal_weights_match_the_default(self) -> None:
        lists = [["a", "b"], ["b", "c"]]
        assert reciprocal_rank_fusion(lists) == reciprocal_rank_fusion(lists, weights=[1.0, 1.0])

    def test_weight_scales_the_contribution(self) -> None:
        fused = reciprocal_rank_fusion([["a"], ["a"]], k=60, weights=[2.0, 1.0])
        assert fused[0].score == pytest.approx(2 / 61 + 1 / 61)

    def test_weighting_flips_a_lopsided_agreement(self) -> None:
        """Cân lệch **đảo được** khi sự đồng thuận lệch về nhánh yếu.

        `a` chỉ dense tìm ra, hạng 1. `b` ở hạng 40 của dense và hạng 1 của sparse.
        Đều nhau: b = 1/100 + 1/61 = 0,02639 > a = 1/61 = 0,01639 → b thắng.
        Cân dense ×3: a = 3/61 = 0,04918 > b = 3/100 + 1/61 = 0,04639 → a thắng.
        """
        dense = ["a", *(f"d{i}" for i in range(2, 40)), "b"]
        assert len(dense) == 40 and dense.index("b") + 1 == 40, "b phải ở hạng 40"
        lists = [dense, ["b"]]
        assert _keys(reciprocal_rank_fusion(lists, k=60))[0] == "b"
        assert _keys(reciprocal_rank_fusion(lists, k=60, weights=[3.0, 1.0]))[0] == "a"

    def test_same_depth_agreement_needs_an_absurd_weight_to_flip(self) -> None:
        """Tính chất của RRF mà tôi đã tính sai lần đầu, nên nó thành một test.

        `a` hạng 1 chỉ ở dense; `b` hạng 3 ở **cả hai**. Cân dense lên cũng cân
        luôn phần dense **của `b`**, nên: a = w₀/61 vs b = (w₀ + w₁)/63. Muốn a
        thắng thì cần `w₀/w₁ > 30,5` — tức trọng số 3:1 hay 5:1 hoàn toàn không
        đủ. Hệ quả cho `W2-08`: quét `weights` trong khoảng hợp lý sẽ **gần như
        không đổi** thứ hạng, và đó là lý do đừng kỳ vọng nhiều ở cần này.
        """
        lists = [["a", "x", "b"], ["y", "z", "b"]]
        for ratio in (2.0, 3.0, 5.0, 30.0):
            top = _keys(reciprocal_rank_fusion(lists, k=60, weights=[ratio, 1.0]))[0]
            assert top == "b", f"tỉ lệ {ratio} vẫn chưa đủ để đảo"
        assert _keys(reciprocal_rank_fusion(lists, k=60, weights=[31.0, 1.0]))[0] == "a"

    def test_zero_weight_silences_a_branch_without_removing_it(self) -> None:
        """`ranks` vẫn ghi lại rằng nhánh đó tìm ra chunk — thông tin cho `W2-08`
        không được mất chỉ vì trọng số bằng 0."""
        fused = reciprocal_rank_fusion([["a"], ["b"]], k=60, weights=[1.0, 0.0])
        b = next(item for item in fused if item.key == "b")
        assert b.score == 0.0
        assert b.ranks == (None, 1)
        assert _keys(fused) == ["a", "b"]

    def test_wrong_number_of_weights_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"2 danh sách"):
            reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])

    def test_negative_weight_is_refused(self) -> None:
        with pytest.raises(ValueError, match="weights phải không âm"):
            reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0, -1.0])


class TestRanksField:
    def test_records_the_rank_in_each_input_list(self) -> None:
        fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
        by_key = {item.key: item for item in fused}
        assert by_key["a"].ranks == (1, None)
        assert by_key["b"].ranks == (2, 1)
        assert by_key["c"].ranks == (None, 2)

    def test_none_means_not_reachable_not_rank_zero(self) -> None:
        """`None` ≠ 0 — cùng bài học với `SparseVector` rỗng ≠ `None` ở `TD-11`.
        "Nhánh này không tới được chunk" khác "nhánh này xếp nó hạng 0"."""
        item = reciprocal_rank_fusion([["a"], []], k=60)[0]
        assert item.ranks == (1, None)
        assert None in item.ranks

    def test_sources_counts_branches(self) -> None:
        by_key = {i.key: i for i in reciprocal_rank_fusion([["a", "b"], ["b"]], k=60)}
        assert by_key["b"].sources == 2
        assert by_key["a"].sources == 1

    def test_best_rank(self) -> None:
        by_key = {i.key: i for i in reciprocal_rank_fusion([["x", "b"], ["b"]], k=60)}
        assert by_key["b"].best_rank == 1


class TestLimitAndRanks:
    def test_limit_truncates_after_fusing_not_before(self) -> None:
        """Cắt trước khi hợp nhất sẽ bỏ mất chính những chunk đồng thuận ở hạng
        sâu — thứ mà RRF tồn tại để tìm ra."""
        dense = [f"d{i}" for i in range(1, 11)]
        fused = reciprocal_rank_fusion([dense, ["d10"]], k=60, limit=3)
        assert len(fused) == 3
        assert fused[0].key == "d10", "d10 lên hạng 1 nhờ đồng thuận, dù dense xếp nó hạng 10"

    def test_ranks_stay_contiguous_after_limit(self) -> None:
        fused = reciprocal_rank_fusion([["a", "b", "c", "d"]], limit=2)
        assert [item.rank for item in fused] == [1, 2]

    def test_limit_zero(self) -> None:
        assert reciprocal_rank_fusion([["a"]], limit=0) == []

    def test_limit_larger_than_the_union(self) -> None:
        assert len(reciprocal_rank_fusion([["a"], ["b"]], limit=99)) == 2

    def test_negative_limit_is_refused(self) -> None:
        with pytest.raises(ValueError, match="limit phải không âm"):
            reciprocal_rank_fusion([["a"]], limit=-1)

    def test_scores_are_non_increasing(self) -> None:
        fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "d", "a"]], k=60)
        scores = [item.score for item in fused]
        assert scores == sorted(scores, reverse=True)


class TestDuplicateKeys:
    def test_duplicate_within_one_list_is_an_error(self) -> None:
        """Trùng khoá trong **cùng** một danh sách là bug ở tầng trên — hai point
        Qdrant mang cùng `chunk_id`. Im lặng bỏ bớt sẽ làm điểm RRF trông hợp lý
        trong khi index đang có bản trùng, tức che đúng thứ `W1-08` bỏ ba tầng
        công sức ra để chống."""
        with pytest.raises(ValueError, match="xuất hiện hai lần"):
            reciprocal_rank_fusion([["a", "b", "a"]])

    def test_the_error_names_both_positions(self) -> None:
        with pytest.raises(ValueError, match="hạng 1 và 3"):
            reciprocal_rank_fusion([["a", "b", "a"]])

    def test_same_key_in_different_lists_is_the_normal_case(self) -> None:
        assert len(reciprocal_rank_fusion([["a"], ["a"]])) == 1
