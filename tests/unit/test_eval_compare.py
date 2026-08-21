"""Test cho `pipeline.eval.compare` — kiểm định giữa hai lần chạy eval.

Module này quyết định câu "cấu hình nào thắng" của cả W2, nên bất biến quan
trọng nhất không phải là số p đúng mà là **nó phải từ chối trả lời khi câu hỏi
sai**: so `recall@k` giữa hai `chunk_size` khác nhau là so hai thước đo khác
nhau, và nó tụt 29,6% kể cả khi truy hồi y nguyên (`TD-11`).

`mcnemar_exact` được so với giá trị **tính tay** từ phân bố nhị thức, cố ý không
gọi lại chính công thức đang test để sinh kỳ vọng.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.eval.compare import (
    DEFAULT_ALPHA,
    MIN_TAIL_RESAMPLES,
    BootstrapBounds,
    ComparisonRow,
    RunScores,
    bootstrap_intervals,
    bootstrap_resample,
    compare_by_group,
    compare_runs,
    format_grouped_table,
    format_table,
    load_per_query,
    mcnemar_exact,
    min_achievable_p,
    paired_bootstrap,
)


class TestMcnemarExact:
    def test_no_discordant_pairs_means_no_evidence(self) -> None:
        """Câu cả hai đều đúng hoặc cả hai đều sai không phân biệt được gì."""
        assert mcnemar_exact(0, 0) == 1.0

    @pytest.mark.parametrize(
        ("b", "c", "expected"),
        [
            (1, 0, 1.0),  # 2 * C(1,1)/2 = 1.0
            (2, 0, 0.5),  # 2 * C(2,2)/4
            (3, 0, 0.25),  # 2 * C(3,3)/8
            (5, 0, 2 / 32),  # 2 * C(5,5)/32
            (2, 1, 1.0),  # 2 * (C(3,2)+C(3,3))/8 = 2 * 4/8
            (10, 0, 2 / 1024),
        ],
    )
    def test_against_hand_computed_binomial(self, b: int, c: int, expected: float) -> None:
        assert mcnemar_exact(b, c) == pytest.approx(expected)

    def test_symmetric(self) -> None:
        """Đổi vai hai hệ thống không đổi p — test hai phía."""
        assert mcnemar_exact(12, 5) == mcnemar_exact(5, 12)

    def test_never_exceeds_one(self) -> None:
        for n in range(1, 30):
            assert mcnemar_exact(n // 2, n - n // 2) <= 1.0

    def test_the_actual_td11_numbers_are_not_significant(self) -> None:
        """Con số thật của `TD-11`: 16↔13 câu đổi chiều là nhiễu, không phải tụt.

        Bảng recall gợi ra "tụt 26%". Thực tế là 3 câu, p = 0,71.
        """
        assert mcnemar_exact(16, 13) > 0.5

    def test_resolution_of_the_current_golden_set(self) -> None:
        """Ghi lại độ phân giải: ~30 câu đổi chiều cần lệch ~12 câu mới đo được.

        Trên 209 câu đó là ~5,7 điểm `hit_rate`. Mọi mức chênh nhỏ hơn là không
        đo được, và bảng ablation của `W2-08` phải nói ra điều đó.
        """
        assert mcnemar_exact(21, 9) < 0.05
        assert mcnemar_exact(20, 10) > 0.05


class TestPairedBootstrap:
    def test_deterministic_for_a_fixed_seed(self) -> None:
        """CI nhảy nhót mỗi lần gọi thì không dùng để quyết định được."""
        diffs = [0.1, -0.2, 0.3, 0.0, -0.1] * 20
        assert paired_bootstrap(diffs, iterations=500) == paired_bootstrap(diffs, iterations=500)

    def test_constant_positive_difference_excludes_zero(self) -> None:
        lo, hi = paired_bootstrap([0.5] * 50, iterations=500)
        assert lo > 0 and hi > 0

    def test_zero_mean_difference_straddles_zero(self) -> None:
        lo, hi = paired_bootstrap([1.0, -1.0] * 50, iterations=500)
        assert lo < 0 < hi

    def test_empty(self) -> None:
        assert paired_bootstrap([]) == (0.0, 0.0)

    def test_more_data_narrows_the_interval(self) -> None:
        small = paired_bootstrap([0.2, -0.1] * 10, iterations=800)
        large = paired_bootstrap([0.2, -0.1] * 200, iterations=800)
        assert (large[1] - large[0]) < (small[1] - small[0])


def _run(name: str, hits: dict[str, float], *, n_relevant: int = 1) -> RunScores:
    return RunScores(
        name=name,
        scores={qid: {"hit_rate@5": v, "recall@5": v, "mrr": v} for qid, v in hits.items()},
        n_relevant=dict.fromkeys(hits, n_relevant),
    )


class TestCompareRuns:
    def test_refuses_recall_when_label_count_changes(self) -> None:
        """Bất biến số một của module này.

        Nhãn neo theo span nên hạ `chunk_size` làm số nhãn/câu tăng, và recall@k
        có mẫu số là chính con số đó. Không có hàng rào này thì `W2-08` sẽ xếp
        hạng cấu hình theo số nhãn của chúng thay vì theo chất lượng.
        """
        base = _run("base", {"q1": 1.0, "q2": 0.0}, n_relevant=1)
        cand = _run("cand", {"q1": 1.0, "q2": 0.0}, n_relevant=2)

        rows = {r.metric: r for r in compare_runs(base, cand, iterations=200)}
        assert rows["recall@5"].comparable is False
        assert "1.00 → 2.00" in rows["recall@5"].note
        assert rows["recall@5"].verdict == "KHÔNG SO ĐƯỢC"
        assert rows["hit_rate@5"].comparable is True, "hit_rate không có mẫu số đó"

    def test_recall_is_comparable_when_labels_match(self) -> None:
        base = _run("base", {"q1": 1.0, "q2": 0.0})
        cand = _run("cand", {"q1": 0.0, "q2": 1.0})
        rows = {r.metric: r for r in compare_runs(base, cand, iterations=200)}
        assert rows["recall@5"].comparable is True

    def test_binary_metric_uses_mcnemar(self) -> None:
        base = _run("base", {f"q{i}": 1.0 for i in range(10)})
        cand = _run("cand", dict.fromkeys([f"q{i}" for i in range(10)], 0.0))
        row = next(r for r in compare_runs(base, cand, iterations=200) if r.metric == "hit_rate@5")
        assert row.test == "McNemar exact"
        assert (row.n_baseline_only, row.n_candidate_only) == (10, 0)
        assert row.p_value is not None and row.p_value < 0.01
        assert row.verdict == "khác biệt thật"

    def test_precision_at_1_uses_mcnemar_like_hit_rate_at_1(self) -> None:
        """`precision@1` bằng `hit_rate@1` từng chữ số — phải cùng một kiểm định.

        `W2-05` phát hiện việc này bằng một mâu thuẫn quan sát được: cùng con số
        0,5598 → 0,5789, McNemar cho `p = 0,125` (0↔4 câu) còn bootstrap cho CI95
        không chứa 0. Hai kiểm định cho hai kết luận về **cùng một số** là một hố
        im lặng — người đọc sẽ trích dòng nào thuận với mình.
        """
        ids = [f"q{i}" for i in range(10)]
        base = RunScores(
            name="base",
            scores={qid: {"precision@1": 0.0, "precision@10": 0.1} for qid in ids},
            n_relevant=dict.fromkeys(ids, 1),
        )
        cand = RunScores(
            name="cand",
            scores={qid: {"precision@1": 1.0, "precision@10": 0.2} for qid in ids},
            n_relevant=dict.fromkeys(ids, 1),
        )
        rows = {r.metric: r for r in compare_runs(base, cand, iterations=200)}
        assert rows["precision@1"].test == "McNemar exact"
        assert (rows["precision@1"].n_baseline_only, rows["precision@1"].n_candidate_only) == (
            0,
            10,
        )

    def test_precision_at_10_is_not_caught_by_the_precision_at_1_rule(self) -> None:
        """Bẫy tiền tố: `"precision@10".startswith("precision@1")` là `True`.

        Bản sửa đầu của tôi ở `W2-05` đưa `precision@1` vào `BINARY_PREFIXES` và
        đẩy luôn `precision@10` sang McNemar — một metric nhận 0; 0,1; 0,2… nên
        McNemar không áp được. Test này canh đúng cái bẫy đó, không canh cách sửa.
        """
        ids = [f"q{i}" for i in range(10)]
        base = RunScores(
            name="base",
            scores={qid: {"precision@1": 0.0, "precision@10": 0.1} for qid in ids},
            n_relevant=dict.fromkeys(ids, 1),
        )
        cand = RunScores(
            name="cand",
            scores={qid: {"precision@1": 1.0, "precision@10": 0.3} for qid in ids},
            n_relevant=dict.fromkeys(ids, 1),
        )
        rows = {r.metric: r for r in compare_runs(base, cand, iterations=200)}
        assert "bootstrap" in rows["precision@10"].test

    def test_continuous_metric_uses_bootstrap(self) -> None:
        base = _run("base", {f"q{i}": 0.5 for i in range(20)})
        cand = _run("cand", {f"q{i}": 0.5 for i in range(20)})
        row = next(r for r in compare_runs(base, cand, iterations=200) if r.metric == "mrr")
        assert "bootstrap" in row.test
        assert row.ci_low == 0.0 and row.ci_high == 0.0

    def test_only_shared_queries_are_compared(self, caplog: pytest.LogCaptureFixture) -> None:
        """So 209 câu với 200 câu rồi kết luận là một dạng tự chọn mẫu."""
        base = _run("base", {"q1": 1.0, "q2": 1.0, "q3": 1.0})
        cand = _run("cand", {"q1": 1.0, "q4": 0.0})
        with caplog.at_level("WARNING", logger="pipeline.eval.compare"):
            rows = compare_runs(base, cand, iterations=200)
        assert "không cùng tập truy vấn" in caplog.text
        row = next(r for r in rows if r.metric == "hit_rate@5")
        assert (row.n_baseline_only, row.n_candidate_only) == (0, 0), "chỉ còn q1, cả hai đều đúng"

    def test_no_shared_queries_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="không có truy vấn nào chung"):
            compare_runs(_run("a", {"q1": 1.0}), _run("b", {"q2": 1.0}), iterations=200)

    def test_metric_filter(self) -> None:
        base = _run("base", {"q1": 1.0})
        rows = compare_runs(base, base, metrics=["mrr"], iterations=200)
        assert [r.metric for r in rows] == ["mrr"]

    def test_missing_metric_is_skipped_not_zero_filled(self) -> None:
        """Metric chỉ có ở một bên thì bỏ qua, không coi bên kia bằng 0."""
        base = RunScores(name="b", scores={"q1": {"mrr": 1.0}}, n_relevant={"q1": 1})
        cand = RunScores(name="c", scores={"q1": {"hit_rate@5": 1.0}}, n_relevant={"q1": 1})
        assert compare_runs(base, cand, iterations=200) == []


class TestLoadPerQuery:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "run-per-query.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(row)
                for row in (
                    {
                        "query_id": "q1",
                        "category": "factoid",
                        "lang": "vi",
                        "n_relevant": 2,
                        "n_retrieved": 20,
                        "scores": {"hit_rate@5": 1.0},
                    },
                    {
                        "query_id": "q2",
                        "category": "factoid",
                        "lang": "en",
                        "n_relevant": 1,
                        "n_retrieved": 20,
                        "scores": {"hit_rate@5": 0.0},
                    },
                )
            ),
            encoding="utf-8",
        )
        run = load_per_query(path)
        assert run.query_ids == {"q1", "q2"}
        assert run.mean_relevant(["q1", "q2"]) == 1.5

    def test_duplicate_query_id_is_an_error(self, tmp_path: Path) -> None:
        path = tmp_path / "dup-per-query.jsonl"
        row = {
            "query_id": "q1",
            "category": "c",
            "lang": "vi",
            "n_relevant": 1,
            "n_retrieved": 1,
            "scores": {},
        }
        path.write_text(json.dumps(row) + "\n" + json.dumps(row), encoding="utf-8")
        with pytest.raises(ValueError, match="hai lần"):
            load_per_query(path)

    def test_blank_lines_are_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "blank-per-query.jsonl"
        row = {
            "query_id": "q1",
            "category": "c",
            "lang": "vi",
            "n_relevant": 1,
            "n_retrieved": 1,
            "scores": {},
        }
        path.write_text("\n\n" + json.dumps(row) + "\n\n", encoding="utf-8")
        assert load_per_query(path).query_ids == {"q1"}

    def test_missing_file_says_how_to_get_it(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="phải chạy lại"):
            load_per_query(tmp_path / "khong-co.jsonl")


def _run_with_labels(name: str, hits: dict[str, float], labels: dict[str, str]) -> RunScores:
    return RunScores(
        name=name,
        scores={qid: {"hit_rate@5": v, "recall@5": v, "mrr": v} for qid, v in hits.items()},
        n_relevant=dict.fromkeys(hits, 1),
        relevant_digest=labels,
    )


class TestLabelDigestGuard:
    """Hàng rào thứ ba, thêm ở `W2-03`.

    Hai hàng rào cũ canh *số* nhãn và *tập truy vấn*. Cả hai đều không thấy
    trường hợp hai lần chạy có **cùng số** nhãn nhưng nhãn **khác nhau** — chuyện
    xảy ra ngay khi một lần chạy tính lại nhãn theo span còn lần kia rơi về nhãn
    ghi sẵn trong file, vì retriever thiếu `fetch_doc_chunks`.
    """

    def test_same_count_different_labels_refuses_everything(self) -> None:
        base = _run_with_labels("base", {"q1": 1.0, "q2": 0.0}, {"q1": "aaaa", "q2": "bbbb"})
        cand = _run_with_labels("cand", {"q1": 1.0, "q2": 0.0}, {"q1": "aaaa", "q2": "cccc"})
        rows = compare_runs(base, cand, iterations=200)
        assert rows, "vẫn phải trả về dòng để bảng nói lý do, không im lặng trả rỗng"
        assert all(not r.comparable for r in rows)
        assert all("tập nhãn khác nhau" in r.note for r in rows)
        assert all(r.verdict == "KHÔNG SO ĐƯỢC" for r in rows)

    def test_refuses_hit_rate_too_not_just_recall(self) -> None:
        """Khác hàng rào số nhãn: ở đó `hit_rate` vẫn so được vì nó không có mẫu
        số là số nhãn. Ở đây thì không — nhãn khác nhau nghĩa là hai bên đang
        chấm hai bài toán khác nhau, `hit_rate` cũng vô nghĩa."""
        base = _run_with_labels("base", {"q1": 1.0}, {"q1": "aaaa"})
        cand = _run_with_labels("cand", {"q1": 1.0}, {"q1": "zzzz"})
        rows = {r.metric: r for r in compare_runs(base, cand, iterations=200)}
        assert rows["hit_rate@5"].comparable is False

    def test_does_not_drop_the_bad_queries_and_compare_the_rest(self) -> None:
        """Lọc bỏ câu lệch rồi so phần còn lại là đúng cái tự chọn mẫu mà hàng
        rào #2 của module cấm. Một câu lệch là cả lần so bỏ."""
        hits = {f"q{i}": 1.0 for i in range(20)}
        base_labels = {q: "same" for q in hits}
        cand_labels = dict(base_labels) | {"q7": "khac"}
        rows = compare_runs(
            _run_with_labels("base", hits, base_labels),
            _run_with_labels("cand", hits, cand_labels),
            iterations=200,
        )
        assert all(not r.comparable for r in rows)
        assert "1/20 câu" in rows[0].note

    def test_matching_labels_pass_through(self) -> None:
        labels = {"q1": "aaaa", "q2": "bbbb"}
        base = _run_with_labels("base", {"q1": 1.0, "q2": 0.0}, labels)
        cand = _run_with_labels("cand", {"q1": 0.0, "q2": 1.0}, dict(labels))
        rows = {r.metric: r for r in compare_runs(base, cand, iterations=200)}
        assert rows["hit_rate@5"].comparable is True
        assert rows["recall@5"].comparable is True

    def test_still_reports_the_numbers_while_refusing(self) -> None:
        """Từ chối kết luận không có nghĩa là ẩn số — người đọc cần thấy mức chênh
        để biết hố này có đáng đi sửa hay không."""
        base = _run_with_labels("base", {"q1": 1.0, "q2": 1.0}, {"q1": "a", "q2": "b"})
        cand = _run_with_labels("cand", {"q1": 0.0, "q2": 0.0}, {"q1": "a", "q2": "X"})
        row = next(r for r in compare_runs(base, cand, iterations=200) if r.metric == "mrr")
        assert row.baseline == 1.0
        assert row.candidate == 0.0
        assert row.delta == -1.0

    def test_missing_digest_is_unknown_not_equal(self, caplog: pytest.LogCaptureFixture) -> None:
        """File của lần chạy trước `W2-03` không có băm. Không biết thì không kết
        luận — nhưng phải nói ra là không biết, nếu không thì hàng rào này chỉ là
        một thứ trang trí mà không ai biết là nó đang tắt."""
        base = _run("base", {"q1": 1.0, "q2": 0.0})
        cand = _run_with_labels("cand", {"q1": 1.0, "q2": 0.0}, {"q1": "a", "q2": "b"})
        with caplog.at_level("WARNING", logger="pipeline.eval.compare"):
            rows = compare_runs(base, cand, iterations=200)
        assert "không ghi `relevant_digest`" in caplog.text
        assert all(r.comparable for r in rows), "không biết thì không từ chối"

    def test_empty_string_digest_does_not_match_empty_string(self) -> None:
        base = _run_with_labels("base", {"q1": 1.0}, {"q1": ""})
        cand = _run_with_labels("cand", {"q1": 1.0}, {"q1": ""})
        rows = compare_runs(base, cand, iterations=200)
        assert all(r.comparable for r in rows), "hai bên đều không biết, không phải đều khớp"

    def test_load_per_query_reads_the_digest(self, tmp_path: Path) -> None:
        path = tmp_path / "d-per-query.jsonl"
        path.write_text(
            json.dumps(
                {
                    "query_id": "q1",
                    "category": "c",
                    "lang": "vi",
                    "n_relevant": 1,
                    "n_retrieved": 1,
                    "scores": {},
                    "relevant_digest": "deadbeef",
                }
            ),
            encoding="utf-8",
        )
        run = load_per_query(path)
        assert run.relevant_digest == {"q1": "deadbeef"}
        assert run.has_digests is True

    def test_old_file_without_digest_loads_fine(self, tmp_path: Path) -> None:
        path = tmp_path / "old-per-query.jsonl"
        path.write_text(
            json.dumps(
                {
                    "query_id": "q1",
                    "category": "c",
                    "lang": "vi",
                    "n_relevant": 1,
                    "n_retrieved": 1,
                    "scores": {},
                }
            ),
            encoding="utf-8",
        )
        run = load_per_query(path)
        assert run.relevant_digest == {"q1": ""}
        assert run.has_digests is False


# ============================================================ chia nhóm (W2-08-prep)


def _rows(
    scores: dict[str, dict[str, float]],
    *,
    category: dict[str, str] | None = None,
    lang: dict[str, str] | None = None,
    n_relevant: int = 2,
) -> RunScores:
    """`RunScores` dựng tay — không chạm đĩa."""
    return RunScores(
        name="r",
        scores=scores,
        n_relevant=dict.fromkeys(scores, n_relevant),
        relevant_digest=dict.fromkeys(scores, "same"),
        category=category or {},
        lang=lang or {},
    )


class TestMinAchievableP:
    """`p` nhỏ nhất McNemar có thể trả về — cơ sở của cờ `KHÔNG ĐỦ LỰC`."""

    @pytest.mark.parametrize(
        ("n", "expected"),
        [(0, 1.0), (1, 1.0), (2, 0.5), (3, 0.25), (4, 0.125), (5, 0.0625), (6, 0.03125)],
    )
    def test_matches_the_binomial_tail(self, n: int, expected: float) -> None:
        assert min_achievable_p(n) == pytest.approx(expected)

    def test_six_discordant_pairs_is_the_first_that_can_reach_significance(self) -> None:
        """Con số quyết định: nhóm dưới 6 câu đổi chiều là vô vọng ở α=0,05.

        `table_lookup` của `golden_v1` có **4** câu, nên nó không bao giờ đạt được
        ý nghĩa trên bất kỳ metric nhị phân nào — không phải "chưa đạt", mà là
        không thể.
        """
        assert min_achievable_p(5) >= 0.05
        assert min_achievable_p(6) < 0.05

    def test_it_agrees_with_mcnemar_at_the_extreme(self) -> None:
        """Định nghĩa phải khớp hàm thật, không phải khớp công thức tôi nhớ."""
        for n in range(1, 9):
            assert mcnemar_exact(n, 0) == pytest.approx(min_achievable_p(n))
            assert mcnemar_exact(0, n) == pytest.approx(min_achievable_p(n))


class TestUnderpoweredIsNotTheSameAsNoEffect:
    def test_a_row_that_cannot_reach_alpha_says_so(self) -> None:
        row = ComparisonRow(
            metric="hit_rate@5",
            baseline=0.3,
            candidate=0.2,
            delta=-0.1,
            test="McNemar exact",
            p_value=0.125,
            n_baseline_only=4,
            n_candidate_only=0,
            n_queries=43,
        )
        assert row.underpowered
        assert row.verdict == "KHÔNG ĐỦ LỰC"

    def test_a_row_with_power_that_found_nothing_says_something_else(self) -> None:
        """Phân biệt cả bài test này tồn tại để bảo vệ."""
        row = ComparisonRow(
            metric="hit_rate@5",
            baseline=0.5,
            candidate=0.5,
            delta=0.0,
            test="McNemar exact",
            p_value=1.0,
            n_baseline_only=10,
            n_candidate_only=10,
            n_queries=209,
        )
        assert not row.underpowered
        assert row.verdict == "trong ngưỡng nhiễu"

    def test_tightening_alpha_can_make_a_powered_row_powerless(self) -> None:
        """Cùng dữ liệu, ngưỡng khắt khe hơn → hết lực. Đo được ở `cross_lingual`.

        `hit_rate@10` có 5↔1 (trần `p` = 0,03125): có lực ở α=0,05 nhưng **không**
        ở α đã hiệu chỉnh cho 90 phép kiểm. Cờ này phải phân biệt được, không thì
        nó chỉ là "n nhỏ" đội lốt.
        """

        def row(alpha: float) -> ComparisonRow:
            return ComparisonRow(
                metric="hit_rate@10",
                baseline=0.44,
                candidate=0.35,
                delta=-0.09,
                test="McNemar exact",
                p_value=0.2188,
                n_baseline_only=5,
                n_candidate_only=1,
                n_queries=43,
                alpha=alpha,
            )

        assert not row(DEFAULT_ALPHA).underpowered
        assert row(DEFAULT_ALPHA / 90).underpowered

    def test_a_non_comparable_row_is_not_relabelled(self) -> None:
        """`KHÔNG SO ĐƯỢC` (nhãn lệch) phải thắng, nó là vấn đề nghiêm trọng hơn."""
        row = ComparisonRow(
            metric="recall@5",
            baseline=0.1,
            candidate=0.2,
            delta=0.1,
            test="—",
            comparable=False,
            n_queries=4,
        )
        assert row.verdict == "KHÔNG SO ĐƯỢC"


class TestCiWithinResolutionIsNotEvidenceOfNoEffect:
    """Biên CI sát 0 = biên giới của lưới rời rạc, không phải kết luận âm.

    Đo được: trong bảng `bgem3 → bgem3-rrf-k1-c20` chia theo category, **4/4**
    dòng `recall@5` có một biên ghim đúng `0,0000` và cả bốn từng bị dán "trong
    ngưỡng nhiễu". Đổi seed dịch biên đúng **một bước lưới** (1/43 = 0,0233) chứ
    không dịch trơn, và tăng 10.000 → 50.000 iterations không đổi gì.

    Luật đổi ở `W2-08` từ "biên `== 0`" sang "biên cách 0 dưới **một bước lưới**",
    vì luật cũ bỏ lọt đúng ca quyết định người thắng của bảng ablation. Và bước
    lưới là `min_increment / n`, **không** phải `1/n` — xem lớp dưới.
    """

    def _row(self, lo: float, hi: float, *, increment: float = 1.0) -> ComparisonRow:
        return ComparisonRow(
            metric="recall@5",
            baseline=0.3,
            candidate=0.21,
            delta=-0.09,
            test="bootstrap cặp (10000)",
            ci_low=lo,
            ci_high=hi,
            n_queries=43,
            min_increment=increment,
        )

    def test_upper_bound_exactly_zero_gives_no_conclusion(self) -> None:
        row = self._row(-0.2558, 0.0)
        assert row.ci_within_resolution
        assert row.verdict == "KHÔNG KẾT LUẬN"

    def test_lower_bound_exactly_zero_gives_no_conclusion(self) -> None:
        assert self._row(0.0, 0.1765).verdict == "KHÔNG KẾT LUẬN"

    def test_an_interval_that_merely_straddles_zero_is_noise(self) -> None:
        """Bao 0 ở cả hai phía là kết luận thật; sát 0 thì không."""
        row = self._row(-0.05, 0.03)
        assert not row.ci_within_resolution
        assert row.verdict == "trong ngưỡng nhiễu"

    def test_an_interval_excluding_zero_still_wins(self) -> None:
        """Ca này là lý do `min_increment` mặc định KHÔNG phải 1,0.

        `[−0,1630, −0,0054]` là `ndcg@10` của `cross_lingual` ở `W2-04` — dẫn chứng
        đã công bố. Với `min_increment = 1,0` thì bước lưới là `1/43 = 0,0233` và
        biên `0,0054` bị dán `KHÔNG KẾT LUẬN`, tức luật mới **xoá một kết quả
        thật**. Độ hạt thật của `ndcg@10` ở đó là ~0,0068, nên bước lưới là
        `0,00016` và kết luận giữ nguyên.
        """
        assert self._row(-0.1630, -0.0054, increment=0.0068).verdict == "khác biệt thật"
        # Chính ca sai: coi metric liên tục như nhị phân.
        assert self._row(-0.1630, -0.0054, increment=1.0).verdict == "KHÔNG KẾT LUẬN"

    def test_unknown_granularity_never_raises_the_flag(self) -> None:
        """`min_increment` mặc định là "chưa biết", và chưa biết thì không gắn cờ."""
        row = self._row(-0.1630, -0.0054, increment=0.0)
        assert row.grid_step == 0.0
        assert not row.ci_within_resolution
        assert row.verdict == "khác biệt thật"


class TestGridStepComesFromTheMetricNotFromN:
    """Bước lưới là `min_increment / n`. Đo được: luật `1/n` cho 13/14 dương giả.

    `precision@20` nhận giá trị bội của 1/20, nên bước thật của nó nhỏ hơn `1/n`
    **hai mươi lần**. Áp `1/n` lên nó làm mọi hiệu dưới 0,0048 bị gắn cờ, và khi
    tôi chạy thử luật đó trên 14 file `compare/` đã công bố thì **13** file đổi
    kết luận — gần hết là `precision@k`/`recall@k` bị gắn cờ oan.
    """

    def _row(self, increment: float) -> ComparisonRow:
        return ComparisonRow(
            metric="precision@20",
            baseline=0.0187,
            candidate=0.0244,
            delta=0.0057,
            test="bootstrap cặp (10000)",
            ci_low=0.0024,
            ci_high=0.0119,
            n_queries=209,
            min_increment=increment,
        )

    def test_binary_metric_keeps_one_over_n(self) -> None:
        assert self._row(1.0).grid_step == pytest.approx(1 / 209)

    def test_coarse_metric_has_a_finer_step(self) -> None:
        assert self._row(0.05).grid_step == pytest.approx(0.05 / 209)

    def test_the_false_positive_the_naive_rule_produced(self) -> None:
        """Cùng một hàng: luật `1/n` gắn cờ, luật đúng thì không."""
        assert self._row(1.0).ci_within_resolution is True
        assert self._row(0.05).ci_within_resolution is False
        assert self._row(0.05).verdict == "khác biệt thật"


class TestBootstrapIntervalsIsOneResampling:
    def test_paired_bootstrap_is_a_thin_wrapper(self) -> None:
        """Hai đường phải cho **cùng** con số, không thì bảng lệch theo cách vô hình."""
        diffs = [0.0] * 30 + [0.5, -0.25, 0.75, -0.5]
        assert (
            paired_bootstrap(diffs, iterations=500, seed=7)
            == bootstrap_intervals(diffs, (DEFAULT_ALPHA,), iterations=500, seed=7)[DEFAULT_ALPHA]
        )

    def test_several_alphas_come_from_the_same_sorted_draws(self) -> None:
        """Khoảng khắt khe hơn phải **chứa** khoảng rộng hơn — cùng một dãy mẫu.

        Gọi bootstrap hai lần với hai alpha có thể vi phạm điều này; lấy hai phân
        vị từ một lần sắp thì không thể.
        """
        diffs = [0.0] * 40 + [0.4, -0.2, 0.6, -0.3, 0.1]
        out = bootstrap_intervals(diffs, (0.05, 0.001), iterations=2000, seed=3)
        wide_lo, wide_hi = out[0.001]
        narrow_lo, narrow_hi = out[0.05]
        assert wide_lo <= narrow_lo
        assert wide_hi >= narrow_hi

    def test_empty_diffs_give_zero_for_every_alpha(self) -> None:
        assert bootstrap_intervals([], (0.05, 0.001)) == {0.05: (0.0, 0.0), 0.001: (0.0, 0.0)}


class TestGrouping:
    @pytest.fixture
    def pair(self) -> tuple[RunScores, RunScores]:
        cat = {"q1": "factoid", "q2": "factoid", "q3": "cross_lingual", "q4": ""}
        lang = {"q1": "vi", "q2": "en", "q3": "en", "q4": "vi"}
        base = _rows(
            {q: {"hit_rate@5": 1.0, "recall@5": 0.5} for q in cat},
            category=cat,
            lang=lang,
        )
        cand = _rows(
            {q: {"hit_rate@5": 0.0, "recall@5": 0.25} for q in cat},
            category=cat,
            lang=lang,
        )
        return base, cand

    def test_groups_are_ordered_largest_first(self, pair: tuple[RunScores, RunScores]) -> None:
        """Nhóm to trước: người đọc gặp kết luận có lực trước, gặp nhóm 4 câu cuối."""
        base, _ = pair
        assert list(base.groups("category")) == ["factoid", "", "cross_lingual"]

    def test_an_unlabelled_query_becomes_its_own_group_not_a_silent_drop(
        self, pair: tuple[RunScores, RunScores]
    ) -> None:
        base, _ = pair
        assert base.groups("category")[""] == ["q4"]
        assert sum(len(v) for v in base.groups("category").values()) == 4

    def test_an_invalid_dimension_raises_instead_of_giving_one_empty_group(self) -> None:
        """Chia theo trường không có sẽ cho **một nhóm rỗng**, đọc y như 'không khác biệt'."""
        with pytest.raises(ValueError, match="Chiều chia nhóm"):
            _rows({"q1": {"mrr": 1.0}}).groups("doc_type")

    def test_subset_narrows_every_parallel_dict(self, pair: tuple[RunScores, RunScores]) -> None:
        """Thiếu một dict là một hàng rào của `compare_runs` chạy trên tập sai."""
        base, _ = pair
        small = base.subset(["q1", "q3"])
        assert small.query_ids == {"q1", "q3"}
        assert set(small.n_relevant) == {"q1", "q3"}
        assert set(small.relevant_digest) == {"q1", "q3"}
        assert set(small.category) == {"q1", "q3"}
        assert set(small.lang) == {"q1", "q3"}
        assert small.name == base.name

    def test_subset_ignores_ids_that_are_not_there(self, pair: tuple[RunScores, RunScores]) -> None:
        base, _ = pair
        assert base.subset(["q1", "khong-co"]).query_ids == {"q1"}

    def test_every_group_gets_a_table(self, pair: tuple[RunScores, RunScores]) -> None:
        base, cand = pair
        out = compare_by_group(base, cand, "category", iterations=200)
        assert set(out) == {"factoid", "(không nhãn)", "cross_lingual"}

    def test_n_queries_is_recorded_per_group(self, pair: tuple[RunScores, RunScores]) -> None:
        """43 câu và 209 câu không cùng độ tin cậy; bảng không in `n` là bảng mời so sai."""
        base, cand = pair
        out = compare_by_group(base, cand, "category", iterations=200)
        assert {k: v[0].n_queries for k, v in out.items()} == {
            "factoid": 2,
            "(không nhãn)": 1,
            "cross_lingual": 1,
        }


class TestMultipleComparisonCorrection:
    """Chia nhóm **là** một cuộc tìm kiếm, nên nó phải tự hiệu chỉnh."""

    @pytest.fixture
    def pair(self) -> tuple[RunScores, RunScores]:
        cat = {f"q{i}": ("a" if i < 10 else "b") for i in range(20)}
        base = _rows({q: {"mrr": 0.5, "hit_rate@5": 1.0} for q in cat}, category=cat)
        cand = _rows({q: {"mrr": 0.6, "hit_rate@5": 1.0} for q in cat}, category=cat)
        return base, cand

    def test_alpha_is_divided_by_groups_times_metrics(
        self, pair: tuple[RunScores, RunScores]
    ) -> None:
        base, cand = pair
        out = compare_by_group(base, cand, "category", iterations=200)
        row = out["a"][0]
        assert row.family_size == 2 * 2
        assert row.alpha == pytest.approx(DEFAULT_ALPHA / 4)

    def test_no_correction_leaves_alpha_alone(self, pair: tuple[RunScores, RunScores]) -> None:
        base, cand = pair
        out = compare_by_group(base, cand, "category", iterations=200, correct=False)
        assert out["a"][0].alpha == pytest.approx(DEFAULT_ALPHA)
        assert out["a"][0].family_size == 1

    def test_family_size_counts_tests_attempted_not_tests_that_worked(
        self, pair: tuple[RunScores, RunScores]
    ) -> None:
        """`m` phải tính được TRƯỚC khi chạy.

        Nếu nó phụ thuộc số hàng so được thì chính `m` trở thành một lựa chọn dựa
        trên dữ liệu — đúng cái hiệu chỉnh tồn tại để chặn.
        """
        base, cand = pair
        out = compare_by_group(base, cand, "category", metrics=["mrr"], iterations=200)
        assert out["a"][0].family_size == 2 * 1

    def test_a_pairwise_comparison_is_never_corrected(self) -> None:
        """`compare_runs` mặc định giữ nguyên α — điều kiện, không phải tình cờ.

        Đổi ngưỡng ở đó sẽ lặng lẽ viết lại kết luận của mọi bảng đã công bố từ
        `W2-01` đến `W2-07`.
        """
        base = _rows({f"q{i}": {"mrr": 0.5} for i in range(20)})
        cand = _rows({f"q{i}": {"mrr": 0.6} for i in range(20)})
        (row,) = compare_runs(base, cand, iterations=200)
        assert row.alpha == pytest.approx(DEFAULT_ALPHA)
        assert row.family_size == 1


class TestGroupedTableTellsTheReaderWhatItIs:
    @pytest.fixture
    def table(self) -> str:
        cat = {f"q{i}": ("a" if i < 6 else "b") for i in range(12)}
        base = _rows({q: {"mrr": 0.5, "hit_rate@5": 1.0} for q in cat}, category=cat)
        cand = _rows({q: {"mrr": 0.6, "hit_rate@5": 0.0} for q in cat}, category=cat)
        groups = compare_by_group(base, cand, "category", iterations=300)
        return format_grouped_table(groups, baseline="A", candidate="B", dimension="category")

    def test_it_states_the_family_size_and_the_adjusted_alpha(self, table: str) -> None:
        assert "phép kiểm" in table
        assert "Bonferroni" in table

    def test_it_states_how_many_false_positives_are_expected_without_correction(
        self, table: str
    ) -> None:
        """Con số này là lý do hiệu chỉnh tồn tại; nó phải ở đầu file, không ở cuối."""
        assert "thuần do ngẫu nhiên" in table

    def test_it_explains_every_verdict_that_is_not_a_number(self, table: str) -> None:
        for label in ("KHÔNG ĐỦ LỰC", "KHÔNG SO ĐƯỢC", "KHÔNG KẾT LUẬN"):
            assert label in table

    def test_every_group_shows_its_n(self, table: str) -> None:
        assert "n = 6" in table

    def test_rows_carry_the_n_column(self, table: str) -> None:
        assert "| n |" in table


class TestPairwiseTableIsUnchanged:
    """Bảng một-cặp phải giữ nguyên hình dạng: 12 file trong `compare/` dùng nó."""

    def test_no_n_column_by_default(self) -> None:
        base = _rows({f"q{i}": {"mrr": 0.5} for i in range(10)})
        cand = _rows({f"q{i}": {"mrr": 0.6} for i in range(10)})
        rows = compare_runs(base, cand, iterations=200)
        table = format_table(rows, baseline="A", candidate="B")
        assert "| n |" not in table
        assert table.splitlines()[0] == "| metric | A | B | Δ | kiểm định | kết luận |"

    def test_n_column_appears_on_request(self) -> None:
        base = _rows({f"q{i}": {"mrr": 0.5} for i in range(10)})
        cand = _rows({f"q{i}": {"mrr": 0.6} for i in range(10)})
        rows = compare_runs(base, cand, iterations=200)
        assert "| n |" in format_table(rows, baseline="A", candidate="B", show_n=True)


class TestLoadReadsTheGroupingColumns:
    def test_category_and_lang_come_from_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "r-per-query.jsonl"
        path.write_text(
            json.dumps(
                {
                    "query_id": "q1",
                    "category": "cross_lingual",
                    "lang": "en",
                    "n_relevant": 2,
                    "n_retrieved": 5,
                    "scores": {"mrr": 0.5},
                    "relevant_digest": "d",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        run = load_per_query(path)
        assert run.category == {"q1": "cross_lingual"}
        assert run.lang == {"q1": "en"}

    def test_a_file_without_those_columns_still_loads(self, tmp_path: Path) -> None:
        """File của lần chạy cũ không có `category`; thiếu thì là chuỗi rỗng, không nổ."""
        path = tmp_path / "old-per-query.jsonl"
        path.write_text(
            json.dumps({"query_id": "q1", "n_relevant": 1, "scores": {"mrr": 1.0}}) + "\n",
            encoding="utf-8",
        )
        run = load_per_query(path)
        assert run.category == {"q1": ""}
        assert run.groups("category") == {"": ["q1"]}


class TestRealGoldenSetShape:
    """Ghim hình dạng thật của `golden_v1` — nó là lý do các cờ trên tồn tại."""

    def test_table_lookup_can_never_reach_significance(self) -> None:
        """4 câu → trần `p` = 0,125. Không phải "chưa đạt", là **không thể**."""
        run = load_per_query("plans/reports/runs/e1-bgem3-dense-per-query.jsonl")
        groups = run.groups("category")
        assert len(groups["table_lookup"]) == 4
        assert min_achievable_p(len(groups["table_lookup"])) >= DEFAULT_ALPHA

    def test_cross_lingual_is_a_fifth_of_the_set(self) -> None:
        run = load_per_query("plans/reports/runs/e1-bgem3-dense-per-query.jsonl")
        groups = run.groups("category")
        assert len(groups["cross_lingual"]) == 43
        assert sum(len(v) for v in groups.values()) == 209

    def test_both_languages_are_large_enough_to_test(self) -> None:
        run = load_per_query("plans/reports/runs/e1-bgem3-dense-per-query.jsonl")
        assert {k: len(v) for k, v in run.groups("lang").items()} == {"vi": 127, "en": 82}


class TestIdenticalIsTheMostCertainRowNotTheLeast:
    """0 câu khác nhau là hàng chắc chắn nhất trong bảng, không phải mơ hồ nhất.

    Đo được ở `W2-08`: `recall@5` giữa `rrf k=0` và `rrf k=1` có **0/209** câu
    khác nhau và `Δ = 0`, nên CI là `[0, 0]`. Luật cũ thấy "một biên đúng 0" và
    dán `KHÔNG KẾT LUẬN`; luật `underpowered` cũng thoả (`min_achievable_p(0) = 1`).
    Cả hai đều đọc bằng chứng mạnh nhất có thể thành bằng chứng yếu nhất.
    """

    def _row(self, *, binary: bool) -> ComparisonRow:
        common = {
            "metric": "recall@5",
            "baseline": 0.5088,
            "candidate": 0.5088,
            "delta": 0.0,
            "n_queries": 209,
            "n_baseline_only": 0,
            "n_candidate_only": 0,
        }
        if binary:
            return ComparisonRow(test="McNemar exact", p_value=1.0, **common)  # type: ignore[arg-type]
        return ComparisonRow(
            test="bootstrap cặp (10000)",
            ci_low=0.0,
            ci_high=0.0,
            **common,  # type: ignore[arg-type]
        )

    def test_bootstrap_row_with_no_differing_query(self) -> None:
        row = self._row(binary=False)
        assert row.identical
        assert row.verdict == "TRÙNG KHỚP"

    def test_binary_row_with_no_differing_query(self) -> None:
        """Hàng McNemar 0↔0 cũng trùng khớp — không phải `KHÔNG ĐỦ LỰC`."""
        row = self._row(binary=True)
        assert row.identical
        assert row.underpowered, "trần p của 0 câu đổi chiều vẫn là 1,0"
        assert row.verdict == "TRÙNG KHỚP", "thứ tự nhánh: identical phải chạy trước"

    def test_a_nonzero_delta_is_never_identical(self) -> None:
        row = ComparisonRow(
            metric="mrr",
            baseline=0.5,
            candidate=0.5,
            delta=1e-9,
            test="bootstrap cặp (10000)",
            ci_low=0.0,
            ci_high=0.0,
            n_queries=209,
        )
        assert not row.identical


class TestDirectionSplitIsNotTheSameAsNoEffect:
    """Trung bình nói một hướng, đếm câu nói hướng ngược — có chữ riêng: TRÁI CHIỀU.

    Đây là ca quyết định người thắng của bảng ablation `W2-08`. `rc50 → rc100`,
    `ndcg@10`: `Δ = +0,0255`, CI95 `[+0,0075, +0,0478]` loại 0 hẳn hoi, nhưng đếm
    câu là **+10 tốt hơn / −11 xấu đi**. Trung bình dương vì mấy câu thắng thắng
    đậm hơn mấy câu thua thua — hợp lệ, nhưng không phải "hệ thống tốt hơn".
    """

    def _row(self, better: int, worse: int, delta: float) -> ComparisonRow:
        return ComparisonRow(
            metric="ndcg@10",
            baseline=0.6481,
            candidate=0.6481 + delta,
            delta=delta,
            test="bootstrap cặp (10000)",
            ci_low=0.0075,
            ci_high=0.0478,
            n_candidate_only=better,
            n_baseline_only=worse,
            n_queries=209,
            min_increment=0.0068,
        )

    def test_the_measured_case(self) -> None:
        row = self._row(better=10, worse=11, delta=0.0255)
        assert row.direction_split
        assert row.verdict == "TRÁI CHIỀU"

    def test_counts_agreeing_with_delta_are_not_flagged(self) -> None:
        row = self._row(better=21, worse=0, delta=0.0255)
        assert not row.direction_split
        assert row.verdict == "khác biệt thật"

    def test_an_exact_tie_in_counts_carries_no_direction(self) -> None:
        """10↔10 không nói gì về hướng, nên nó cũng phải bị gắn cờ."""
        assert self._row(better=10, worse=10, delta=0.0255).direction_split

    def test_it_is_a_separate_verdict_from_unreadable_intervals(self) -> None:
        """`TRÁI CHIỀU` khác `KHÔNG KẾT LUẬN`: ở đây khoảng ĐỌC ĐƯỢC, nó nói hai điều."""
        split = self._row(better=10, worse=11, delta=0.0255)
        assert not split.ci_within_resolution
        assert not split.mc_unstable
        assert split.verdict == "TRÁI CHIỀU"

    def test_a_noise_row_is_not_relabelled(self) -> None:
        """Khoảng đã chứa 0 thì hai cờ này không được hỏi — thêm chữ, không thêm tin."""
        row = ComparisonRow(
            metric="ndcg@10",
            baseline=0.5,
            candidate=0.51,
            delta=0.01,
            test="bootstrap cặp (10000)",
            ci_low=-0.02,
            ci_high=0.04,
            n_candidate_only=5,
            n_baseline_only=9,
            n_queries=209,
            min_increment=0.0068,
        )
        assert row.direction_split, "cờ vẫn mô tả đúng dữ liệu"
        assert row.verdict == "trong ngưỡng nhiễu", "nhưng kết luận không đổi"


class TestMonteCarloInstabilityIsADifferentLimitFromTheLattice:
    """Biên đọc từ quá ít mẫu lại thì chính nó dao động qua 0.

    Đo được ở `W2-08` trên `rc50 → rc100`, `ndcg@10`, α đã hiệu chỉnh cho 39 phép
    kiểm (0,00128) và B = 10.000 → đuôi chỉ **6** mẫu. Sáu seed cho biên dưới nhận
    cả dấu `+` lẫn `−`; ở B = 50.000 (đuôi 32) và B = 200.000 (đuôi 128) nó âm
    nhất quán, tức khoảng thật sự **chứa** 0.

    ⚠️ Và nó **ngược** ghi chú của `W2-08-prep` ("đừng chữa bằng cách tăng
    iterations"). Ghi chú đó đúng cho ca metric nhị phân thưa; ca này là metric
    liên tục và tăng `B` **đảo** kết luận. Hai giới hạn khác nhau, cùng một triệu
    chứng — test này ghim rằng chúng là hai cờ khác nhau.
    """

    def test_a_bound_whose_own_jitter_spans_zero(self) -> None:
        row = ComparisonRow(
            metric="ndcg@10",
            baseline=0.6481,
            candidate=0.6736,
            delta=0.0255,
            test="bootstrap cặp (10000)",
            ci_low=0.0003,
            ci_high=0.0638,
            ci_jitter=(-0.0008, 0.0005),
            n_queries=209,
            min_increment=0.0068,
            alpha=0.05 / 39,
            family_size=39,
        )
        assert row.mc_unstable
        assert not row.ci_within_resolution, "0,0003 cách 0 gấp 9 lần bước lưới 3,2e-05"
        assert row.verdict == "KHÔNG KẾT LUẬN"

    def test_a_stable_bound_is_left_alone(self) -> None:
        """Dẫn chứng `cross_lingual` của `W2-04`: biên −0,0193, dao động không chạm 0."""
        row = ComparisonRow(
            metric="map@20",
            baseline=0.3,
            candidate=0.22,
            delta=-0.08,
            test="bootstrap cặp (10000)",
            ci_low=-0.1235,
            ci_high=-0.0193,
            ci_jitter=(-0.0201, -0.0186),
            n_queries=43,
            min_increment=0.0093,
            alpha=0.05 / 90,
            family_size=90,
        )
        assert not row.mc_unstable
        assert row.verdict == "khác biệt thật"

    def test_rows_without_a_bootstrap_have_no_jitter(self) -> None:
        row = ComparisonRow(
            metric="hit_rate@1",
            baseline=0.5,
            candidate=0.52,
            delta=0.02,
            test="McNemar exact",
            p_value=0.125,
            n_candidate_only=4,
            n_queries=209,
        )
        assert row.ci_jitter is None
        assert not row.mc_unstable

    def test_tail_size_shrinks_with_the_corrected_alpha(self) -> None:
        """Con số làm cờ này cần thiết: α/2 × B là bao nhiêu mẫu lại thật sự."""
        assert int(DEFAULT_ALPHA / 2 * 10_000) == 250
        assert int((DEFAULT_ALPHA / 39) / 2 * 10_000) == 6
        assert 6 < MIN_TAIL_RESAMPLES <= 250

    def test_resample_reports_its_own_tail_and_jitter(self) -> None:
        diffs = [0.0] * 190 + [0.3] * 10 + [-0.25] * 9
        bounds = bootstrap_resample(diffs, (DEFAULT_ALPHA,), iterations=2_000)[DEFAULT_ALPHA]
        assert bounds.tail == int(DEFAULT_ALPHA / 2 * 2_000)
        lo_lo, lo_hi = bounds.low_jitter
        assert lo_lo <= bounds.low <= lo_hi, "biên phải nằm trong khoảng dao động của nó"
        hi_lo, hi_hi = bounds.high_jitter
        assert hi_lo <= bounds.high <= hi_hi

    def test_near_jitter_picks_the_bound_that_decides(self) -> None:
        bounds = BootstrapBounds(
            low=0.0003,
            high=0.0638,
            low_jitter=(-0.0008, 0.0005),
            high_jitter=(0.0600, 0.0670),
            tail=6,
        )
        assert bounds.near_jitter() == (-0.0008, 0.0005)

    def test_thin_wrapper_agrees_with_the_rich_call(self) -> None:
        """`bootstrap_intervals` phải là lớp mỏng — hai đường không được lệch nhau."""
        diffs = [0.0] * 180 + [0.5] * 15 + [-0.4] * 14
        alphas = (DEFAULT_ALPHA, DEFAULT_ALPHA / 39)
        thin = bootstrap_intervals(diffs, alphas, iterations=3_000)
        rich = bootstrap_resample(diffs, alphas, iterations=3_000)
        assert thin == {a: (b.low, b.high) for a, b in rich.items()}


class TestEveryBootstrapRowKnowsItsOwnGranularity:
    """`compare_runs` phải điền `min_increment` và `ci_jitter` cho MỌI hàng bootstrap.

    Mặc định của hai trường đó là "chưa biết" (không gắn cờ), nên một hàng đi ra
    khỏi `compare_runs` mà thiếu chúng là một hàng **im lặng tắt** hai cờ.
    """

    def test_bootstrap_rows_are_fully_populated(self) -> None:
        base = _run("b", {"q1": 0.0, "q2": 0.5, "q3": 1.0, "q4": 0.25, "q5": 0.75})
        cand = _run("c", {"q1": 0.5, "q2": 0.5, "q3": 0.5, "q4": 0.75, "q5": 0.25})
        rows = compare_runs(base, cand, metrics=["mrr"])
        (row,) = rows
        assert row.test.startswith("bootstrap")
        assert row.min_increment > 0.0
        assert row.ci_jitter is not None
        assert row.n_discordant > 0, "đếm câu phải có cho cả hàng bootstrap"

    def test_mcnemar_rows_do_not_pretend_to_have_an_interval(self) -> None:
        base = _run("b", {"q1": 0.0, "q2": 1.0, "q3": 0.0})
        cand = _run("c", {"q1": 1.0, "q2": 1.0, "q3": 0.0})
        (row,) = compare_runs(base, cand, metrics=["hit_rate@5"])
        assert row.ci_jitter is None
        assert row.p_value is not None
