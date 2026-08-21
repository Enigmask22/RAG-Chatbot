"""Test cho bảng ablation N ô — và cho ba cái bẫy đã đo được ở `W2-08`.

Mỗi lớp dưới đây tồn tại vì một lỗi **đã xảy ra** trong lúc làm `W2-08`, không
phải vì một khả năng lý thuyết. Cái đắt nhất: bản đầu của `winner_set` đưa ô tệ
nhất bảng vào tập thắng rồi báo nó là phương án rẻ nhất.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.eval.ablation import (
    PRIMARY_METRICS,
    AblationCell,
    ablation_vs_baseline,
    discover_runs,
    format_ablation_table,
    format_winner_set,
    load_cells,
    winner_set,
)
from pipeline.eval.compare import RunScores

METRICS = ("ndcg@10", "hit_rate@1", "mrr")


def _cell(
    name: str,
    *,
    scores: dict[str, dict[str, float]],
    p95: float,
    n_relevant: int = 1,
    digest: str = "d",
    chunk: int | None = 1000,
) -> AblationCell:
    ids = list(scores)
    overall = {
        m: sum(scores[q].get(m, 0.0) for q in ids) / len(ids)
        for m in {m for q in ids for m in scores[q]}
    }
    return AblationCell(
        name=name,
        scores=RunScores(
            name=name,
            scores=scores,
            n_relevant=dict.fromkeys(ids, n_relevant),
            relevant_digest=dict.fromkeys(ids, digest),
            category=dict.fromkeys(ids, "factoid"),
            lang=dict.fromkeys(ids, "vi"),
        ),
        overall=overall,
        p95_ms=p95,
        n_relevant_mean=float(n_relevant),
        embedding_model="BAAI/bge-m3",
        retrieval_mode="reranked",
        branch_options={"rerank_candidates": 50},
        chunk_size=chunk,
    )


def _ladder(level: float, n: int = 40) -> dict[str, dict[str, float]]:
    """`n` câu, một tỉ lệ `level` trong đó đúng — dựng chênh lệch lớn tuỳ ý."""
    good = round(level * n)
    return {
        f"q{i}": {
            "ndcg@10": 1.0 if i < good else 0.0,
            "hit_rate@1": 1.0 if i < good else 0.0,
            "mrr": 1.0 if i < good else 0.0,
        }
        for i in range(n)
    }


class TestIncomparableIsNotEquivalent:
    """Cái bẫy đắt nhất của `W2-08`, ghim lại để nó không sống lại.

    Bản đầu định nghĩa tập tương đương là "không bị đánh bại". Ô `chunk550` có
    nhãn khác nên **mọi** phép so bị từ chối, và "bị từ chối" đọc y như "hoà" —
    nên nó vào tập thắng. Rồi vì nó là dense thuần (46 ms) nó thành **thành viên
    rẻ nhất**, tức công cụ đề xuất cấu hình có `ndcg@10 = 0,1215` thay cho cấu
    hình có `0,6736`.
    """

    @pytest.fixture
    def cells(self) -> list[AblationCell]:
        return [
            _cell("top", scores=_ladder(0.9), p95=1000.0),
            _cell("mid", scores=_ladder(0.2), p95=500.0),
            # Nhãn khác (digest khác) → mọi metric bị từ chối. Và nó là ô tệ nhất.
            _cell("otherlabels", scores=_ladder(0.05), p95=40.0, digest="KHAC"),
        ]

    def test_the_incomparable_cell_is_its_own_bucket(self, cells: list[AblationCell]) -> None:
        result = winner_set(cells, rank_by="ndcg@10", metrics=METRICS)
        assert [c.name for c in result.incomparable] == ["otherlabels"]
        assert "otherlabels" not in {c.name for c in result.equivalent}
        assert "otherlabels" not in {c.name for c in result.contested}
        assert "otherlabels" not in {c.name for c in result.beaten}

    def test_it_never_becomes_the_cheapest_recommendation(self, cells: list[AblationCell]) -> None:
        """Chính hồi quy: ô 40 ms không được thành `cheapest` chỉ vì không so được."""
        result = winner_set(cells, rank_by="ndcg@10", metrics=METRICS)
        assert result.cheapest.name != "otherlabels"
        assert {c.name for c in result.members} <= {"top", "mid"}

    def test_the_report_names_it_and_says_why(self, cells: list[AblationCell]) -> None:
        text = format_winner_set(winner_set(cells, rank_by="ndcg@10", metrics=METRICS))
        assert "Không so được" in text
        assert "otherlabels" in text


class TestFourBucketsAreExhaustiveAndDisjoint:
    """Mọi ô ngoài đỉnh bảng phải vào **đúng một** rổ. Không rơi, không trùng."""

    @pytest.fixture
    def cells(self) -> list[AblationCell]:
        return [
            _cell("top", scores=_ladder(0.90), p95=1200.0),
            _cell("near", scores=_ladder(0.875), p95=600.0),
            _cell("far", scores=_ladder(0.20), p95=50.0),
            _cell("nolabels", scores=_ladder(0.30), p95=45.0, digest="X"),
        ]

    def test_partition(self, cells: list[AblationCell]) -> None:
        result = winner_set(cells, rank_by="ndcg@10", metrics=METRICS)
        buckets = [result.equivalent, result.contested, result.beaten, result.incomparable]
        names = [c.name for bucket in buckets for c in bucket]
        assert sorted(names) == ["far", "near", "nolabels"]
        assert len(names) == len(set(names)), "không ô nào vào hai rổ"
        assert result.top.name == "top"

    def test_members_is_the_strict_set(self, cells: list[AblationCell]) -> None:
        """`members` = đỉnh bảng + `equivalent`, **không** gộp `contested`."""
        result = winner_set(cells, rank_by="ndcg@10", metrics=METRICS)
        assert {c.name for c in result.members} == {"top"} | {c.name for c in result.equivalent}
        assert not ({c.name for c in result.contested} & {c.name for c in result.members})


class TestBeatenNeedsEveryUsableMetricToAgree:
    """`beaten` là "mọi metric so được đều nói kém"; một phần thì là `contested`.

    Đo được ở `W2-08`: `rc100 → rc50` cho `mrr` "khác biệt thật" (biên CI −0,0007,
    kiểm định dấu `p = 0,45`) nhưng `ndcg@10` `TRÁI CHIỀU` và `hit_rate@1`
    `KHÔNG ĐỦ LỰC` — tức phép kiểm **exact** duy nhất nói là không đo được. Dán
    "bị đánh bại" thì kết luận phải trả 1,91× độ trễ.
    """

    def test_a_cell_beaten_on_every_metric_is_beaten(self) -> None:
        cells = [
            _cell("top", scores=_ladder(0.95), p95=1000.0),
            _cell("weak", scores=_ladder(0.10), p95=50.0),
        ]
        result = winner_set(cells, rank_by="ndcg@10", metrics=METRICS)
        assert [c.name for c in result.beaten] == ["weak"]
        assert not result.contested

    def test_a_cell_nobody_can_separate_is_equivalent(self) -> None:
        same = _ladder(0.5)
        cells = [
            _cell("top", scores=same, p95=1000.0),
            _cell("twin", scores=dict(same), p95=60.0),
        ]
        result = winner_set(cells, rank_by="ndcg@10", metrics=METRICS)
        assert [c.name for c in result.equivalent] == ["twin"]
        assert result.cheapest.name == "twin"
        assert result.speedup == pytest.approx(1000.0 / 60.0)

    def test_the_price_of_the_cheapest_member_is_reported(self) -> None:
        same = _ladder(0.5)
        cells = [
            _cell("top", scores=same, p95=1163.9),
            _cell("half", scores=dict(same), p95=608.9),
        ]
        text = format_winner_set(winner_set(cells, rank_by="ndcg@10", metrics=METRICS))
        assert "1.91×" in text, "tỉ lệ giá là kết luận kỹ thuật của bảng"


class TestRankingDependsOnTheMetric:
    """Metric xếp hạng phải nêu tường minh vì thứ hạng đổi theo nó.

    Đo được trên 14 ô thật: `e1-bgem3-dense` là hạng 10 theo `ndcg@10` nhưng hạng
    **8** theo `hit_rate@1`, đổi chỗ với hai ô `rrf`.
    """

    @pytest.fixture
    def cells(self) -> list[AblationCell]:
        a = {f"q{i}": {"ndcg@10": 0.9, "hit_rate@1": 0.1, "mrr": 0.5} for i in range(20)}
        b = {f"q{i}": {"ndcg@10": 0.1, "hit_rate@1": 0.9, "mrr": 0.5} for i in range(20)}
        return [
            _cell("ndcg-winner", scores=a, p95=100.0),
            _cell("hit-winner", scores=b, p95=100.0),
        ]

    def test_top_changes_with_rank_by(self, cells: list[AblationCell]) -> None:
        assert winner_set(cells, rank_by="ndcg@10", metrics=METRICS).top.name == "ndcg-winner"
        assert winner_set(cells, rank_by="hit_rate@1", metrics=METRICS).top.name == "hit-winner"

    def test_a_cell_beating_the_top_on_another_metric_is_recorded(
        self, cells: list[AblationCell]
    ) -> None:
        """`conflicts` phải là một mục trong kết quả, không phải một giả định."""
        result = winner_set(cells, rank_by="ndcg@10", metrics=METRICS)
        assert "hit-winner" in result.conflicts
        assert "hit_rate@1" in result.conflicts["hit-winner"]
        text = format_winner_set(result)
        assert "Thứ hạng phụ thuộc metric" in text


class TestTheTwoTablesUseTwoThresholds:
    """Bảng so-với-baseline **không** hiệu chỉnh; tập tương đương thì có.

    Hai loại suy luận khác nhau — cùng lý lẽ với `compare_runs` vs
    `compare_by_group` ở `W2-08-prep`. Đây là chỗ dễ bị "dọn cho nhất quán" nhất,
    nên nó có test.
    """

    @pytest.fixture
    def cells(self) -> list[AblationCell]:
        return [
            _cell("top", scores=_ladder(0.9), p95=1000.0),
            _cell("mid", scores=_ladder(0.5), p95=500.0),
            _cell("base", scores=_ladder(0.1), p95=50.0),
        ]

    def test_baseline_panel_stays_at_five_percent(self, cells: list[AblationCell]) -> None:
        by_name = {c.name: c for c in cells}
        rows = ablation_vs_baseline(cells, by_name["base"], metrics=METRICS)
        assert set(rows) == {"top", "mid"}
        for cell_rows in rows.values():
            for row in cell_rows:
                assert row.alpha == pytest.approx(0.05)
                assert row.family_size == 1

    def test_winner_set_divides_alpha(self, cells: list[AblationCell]) -> None:
        result = winner_set(cells, rank_by="ndcg@10", metrics=METRICS)
        assert result.family_size == 2 * len(METRICS)
        assert result.alpha == pytest.approx(0.05 / (2 * len(METRICS)))
        for cell_rows in result.rows.values():
            for row in cell_rows:
                assert row.family_size == result.family_size

    def test_the_report_says_which_table_is_which(self, cells: list[AblationCell]) -> None:
        text = format_winner_set(winner_set(cells, rank_by="ndcg@10", metrics=METRICS))
        assert "tìm kiếm" in text
        assert "Bonferroni" in text


class TestTableShowsWhatMakesRowsComparable:
    def test_label_count_column_is_present(self) -> None:
        cells = [
            _cell("a", scores=_ladder(0.5), p95=50.0, n_relevant=1),
            _cell("b", scores=_ladder(0.4), p95=60.0, n_relevant=2),
        ]
        table = format_ablation_table(cells, rank_by="ndcg@10", metrics=METRICS)
        assert "nhãn/câu" in table
        assert "1.0000" in table and "2.0000" in table

    def test_config_label_names_the_dimension_not_the_run(self) -> None:
        cell = _cell("whatever", scores=_ladder(0.5), p95=50.0, chunk=550)
        assert "bge-m3" in cell.label
        assert "reranked" in cell.label
        assert "chunk=550" in cell.label

    def test_rows_are_ordered_by_the_declared_metric(self) -> None:
        cells = [
            _cell("low", scores=_ladder(0.2), p95=50.0),
            _cell("high", scores=_ladder(0.8), p95=50.0),
        ]
        table = format_ablation_table(cells, rank_by="ndcg@10", metrics=METRICS)
        assert table.index("`high`") < table.index("`low`")


class TestLoadingReadsWhatTheGridWrote:
    """`load_cells` phải đọc đúng hình dạng file mà `retrieval_eval` ghi ra."""

    def _write(self, directory: Path, name: str, *, p95: float, chunk: int) -> None:
        (directory / f"{name}-retrieval.json").write_text(
            json.dumps(
                {
                    "run_name": name,
                    "overall": {"ndcg@10": 0.5, "hit_rate@1": 0.3, "mrr": 0.4},
                    "latency_ms": {"p95": p95},
                    "n_relevant_mean": 1.3828,
                    "config": {
                        "embedding_model": "BAAI/bge-m3",
                        "retrieval_mode": "hybrid",
                        "branch_options": {"k": 1},
                        "chunking": {"chunk_size": chunk},
                    },
                }
            ),
            encoding="utf-8",
        )
        (directory / f"{name}-per-query.jsonl").write_text(
            json.dumps(
                {
                    "query_id": "q1",
                    "scores": {"ndcg@10": 0.5, "hit_rate@1": 0.0, "mrr": 0.4},
                    "n_relevant": 1,
                    "relevant_digest": "abc",
                    "category": "factoid",
                    "lang": "vi",
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def test_discover_uses_the_prefix(self, tmp_path: Path) -> None:
        self._write(tmp_path, "e1-alpha", p95=10.0, chunk=1000)
        self._write(tmp_path, "e1-beta", p95=20.0, chunk=550)
        self._write(tmp_path, "old-gamma", p95=30.0, chunk=1000)
        assert discover_runs(tmp_path, "e1-") == ["e1-alpha", "e1-beta"]

    def test_an_empty_prefix_is_an_error_not_an_empty_table(self, tmp_path: Path) -> None:
        """Bảng rỗng đọc y như "không có khác biệt" — phải nổ."""
        with pytest.raises(FileNotFoundError, match="make exp"):
            discover_runs(tmp_path, "nothing-")

    def test_latency_and_chunk_size_survive_the_round_trip(self, tmp_path: Path) -> None:
        self._write(tmp_path, "e1-alpha", p95=608.9, chunk=550)
        (cell,) = load_cells(tmp_path, ["e1-alpha"])
        assert cell.p95_ms == pytest.approx(608.9)
        assert cell.chunk_size == 550
        assert cell.n_relevant_mean == pytest.approx(1.3828)
        assert cell.metric("ndcg@10") == pytest.approx(0.5)


class TestPrimaryMetricsAreDeclaredNotDiscovered:
    def test_the_default_family_is_three_metrics(self) -> None:
        """Thêm metric là nới họ phép kiểm ra, tức tự làm yếu chính mình."""
        assert PRIMARY_METRICS == ("ndcg@10", "hit_rate@1", "mrr")

    def test_exactly_one_of_them_avoids_the_bootstrap(self) -> None:
        """`hit_rate@1` đi McNemar exact — dòng duy nhất không phụ thuộc Monte Carlo.

        Đó là lý do nó nằm trong bộ ba: khi hai dòng bootstrap chạm giới hạn số
        mẫu lại (đã xảy ra ở `rc50 → rc100`), dòng này vẫn nói được điều gì mà
        không cần tin vào một lần lấy mẫu nào.
        """
        from pipeline.eval.compare import BINARY_METRICS, BINARY_PREFIXES

        binary = [
            m for m in PRIMARY_METRICS if m.startswith(BINARY_PREFIXES) or m in BINARY_METRICS
        ]
        assert binary == ["hit_rate@1"]


class TestTheRealGridStillHasTwelvePlusCells:
    """DoD `W2-08` đòi ≥ 12 tổ hợp có kết quả đầy đủ. Ghim vào dữ liệu thật."""

    RUNS = Path("plans/reports/runs")

    @pytest.mark.skipif(
        not (Path("plans/reports/runs") / "e1-baseline-dense-retrieval.json").exists(),
        reason="chưa chạy `make exp`",
    )
    def test_at_least_twelve_cells_with_per_query_scores(self) -> None:
        names = discover_runs(self.RUNS, "e1-")
        assert len(names) >= 12
        for name in names:
            assert (self.RUNS / f"{name}-per-query.jsonl").exists(), (
                f"{name} thiếu điểm từng câu — không có `p`/CI thì không lên bảng được"
            )

    @pytest.mark.skipif(
        not (Path("plans/reports/runs") / "e1-chunk550-dense-retrieval.json").exists(),
        reason="chưa chạy `make exp`",
    )
    def test_the_chunk_size_cell_has_a_different_label_distribution(self) -> None:
        """Chiều `chunk_size` **thật sự** không so được — không phải giả định.

        `G2` ghi "đổi chunk_size thì dùng hit_rate@k/MRR". Đo ra thì chặt hơn thế:
        nhãn neo theo span nên đổi `chunk_size` đổi luôn **tập** nhãn, không chỉ
        số lượng, và hàng rào băm của `W2-03` từ chối **cả 15** metric.
        """
        cells = load_cells(self.RUNS, ["e1-baseline-dense", "e1-chunk550-dense"])
        base, chunk = cells
        assert base.n_relevant_mean != chunk.n_relevant_mean
        shared = sorted(base.scores.query_ids & chunk.scores.query_ids)
        differing = [
            q for q in shared if base.scores.relevant_digest[q] != chunk.scores.relevant_digest[q]
        ]
        assert len(differing) == len(shared), "mọi câu đều đổi tập nhãn"
