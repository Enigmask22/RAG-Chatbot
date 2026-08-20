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
    RunScores,
    compare_runs,
    load_per_query,
    mcnemar_exact,
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
