"""`W2-07` — experiment runner.

DoD nêu hai thứ phải test: "expand grid đúng số tổ hợp" và "resume". Cái thứ nhất
là một `itertools.product` và tests của nó ngắn. **Toàn bộ trọng lượng của file
này nằm ở cái thứ hai**, vì "resume" có một cách cài đúng và nhiều cách cài trông
đúng:

* Bỏ qua ô có file báo cáo → đổi tham số rồi resume thì số cũ vào bảng mới.
* Bỏ qua ô có tên trong state → đổi `chunk_size` của index rồi build lại thì mọi
  ô vẫn bị bỏ qua, và bảng trộn hai index.
* Ghi state trước khi ghi báo cáo → crash giữa hai bước để lại ô `done` không có
  báo cáo, và resume bỏ qua nó mãi mãi.

Cả ba đều cho một grid "chạy xong" với những con số sai và không có gì báo lỗi —
cùng họ với `MatchAny(any=[])` ở `W2-06` và `ensure_collection` ở `W2-02`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from pydantic import ValidationError

from pipeline.experiments.backfill import backfill
from pipeline.experiments.config import (
    BRANCH_OPTIONS,
    SPEED_ONLY_OPTIONS,
    ExperimentCell,
    ExperimentConfig,
    MatrixBlock,
    cell_table,
    expand,
    golden_digest,
    load_experiment_config,
)
from pipeline.experiments.runner import (
    CellRecord,
    ExperimentState,
    PreflightError,
    _write_report,
    cell_params,
    plan_cells,
    preflight,
    report_metrics,
    report_params,
)
from pipeline.experiments.tracking import (
    METRIC_NAME_MAP,
    NullTracker,
    SafeTracker,
    TrackingUnavailable,
    mlflow_metric_name,
    open_tracker,
)
from rag_core.retrieval import HYBRID_OPTIONS, RERANK_OPTIONS
from rag_core.schemas import RetrievalMode

if TYPE_CHECKING:
    from collections.abc import Sequence

# --------------------------------------------------------------------- fixtures


def _index_yaml(directory: Path, name: str, *, chunk_size: int = 1000) -> Path:
    """Config index tối thiểu — đủ để `load_index_config` tính `fingerprint`."""
    path = directory / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump({"name": name, "tenant_id": "test", "chunking": {"chunk_size": chunk_size}}),
        encoding="utf-8",
    )
    return path


def _block(*, mode: str = "hybrid", options: dict[str, Any] | None = None) -> MatrixBlock:
    """`MatrixBlock` dựng qua `model_validate`, tức đúng đường mà YAML đi.

    Gọi constructor với `("a.yaml",)` cũng chạy — pydantic tự ép `str` → `Path` —
    nhưng `mypy --strict` đúng khi từ chối: kiểu khai là `tuple[Path, ...]`. Và ép
    ngầm ở test là chỗ che mất chuyện file config thật đi qua một đường khác.
    """
    return MatrixBlock.model_validate(
        {"index_config": ["a.yaml"], "retrieval_mode": [mode], "options": options or {}}
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "configs").mkdir()
    (tmp_path / "runs").mkdir()
    golden = tmp_path / "golden.jsonl"
    golden.write_text('{"query_id": "q1"}\n', encoding="utf-8")
    return tmp_path


def _config(workspace: Path, matrix: list[dict[str, Any]], **extra: Any) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "test-exp",
            "golden": str(workspace / "golden.jsonl"),
            "out_dir": str(workspace / "runs"),
            "state_dir": str(workspace / "state"),
            "matrix": matrix,
            **extra,
        }
    )


# ------------------------------------------------------- DoD: expand đúng số ô


class TestExpandsToTheStatedNumberOfCells:
    """DoD `W2-08` nói "≥ 12 tổ hợp", nên số ô phải suy được từ file config."""

    def test_one_block_is_the_product_of_its_axes(self, workspace: Path) -> None:
        config = _config(
            workspace,
            [
                {
                    "index_config": [str(_index_yaml(workspace / "configs", "a"))],
                    "retrieval_mode": ["hybrid"],
                    "options": {"k": [0, 1, 60], "candidate_k": [20, 50]},
                }
            ],
        )
        assert len(expand(config)) == 1 * 1 * 3 * 2

    def test_blocks_add_up(self, workspace: Path) -> None:
        first = _index_yaml(workspace / "configs", "a")
        second = _index_yaml(workspace / "configs", "b", chunk_size=550)
        config = _config(
            workspace,
            [
                {"index_config": [str(first), str(second)], "retrieval_mode": ["dense"]},
                {
                    "index_config": [str(first)],
                    "retrieval_mode": ["hybrid"],
                    "options": {"k": [1, 60]},
                },
            ],
        )
        assert len(expand(config)) == 2 + 2

    def test_the_real_w2_08_grid_has_at_least_twelve_cells(self) -> None:
        """Ghim chính con số DoD lên file config thật, không lên một fixture.

        Grid đi qua `--dry-run` không chứng minh nó *đủ lớn*; DoD `W2-08` đòi ≥ 12
        tổ hợp và đó là một tính chất của `configs/eval/exp-001-retrieval.yaml`.
        """
        config = load_experiment_config("configs/eval/exp-001-retrieval.yaml")
        assert len(expand(config)) >= 12

    def test_the_real_grid_has_no_duplicate_run_names(self) -> None:
        cells = expand(load_experiment_config("configs/eval/exp-001-retrieval.yaml"))
        names = [cell.run_name for cell in cells]
        assert len(set(names)) == len(names)

    def test_order_follows_the_yaml_so_resume_is_reproducible(self, workspace: Path) -> None:
        config = _config(
            workspace,
            [
                {
                    "index_config": [str(_index_yaml(workspace / "configs", "a"))],
                    "retrieval_mode": ["hybrid"],
                    "options": {"k": [60, 1, 0]},
                }
            ],
        )
        # `k` được sắp theo TÊN tham số, còn giá trị giữ đúng thứ tự YAML.
        assert [cell.branch_options["k"] for cell in expand(config)] == [60, 1, 0]


# --------------------------------------------- grid không đo được gì thì phải nổ


class TestRejectsGridsThatMeasureNothing:
    def test_empty_axis_raises_instead_of_producing_no_cells(self) -> None:
        """`k: []` cho tích Descartes = 0 ô, tức grid im lặng thành rỗng."""
        with pytest.raises(ValidationError) as err:
            _block(options={"k": []})
        assert "0 ô" in str(err.value)

    def test_unknown_option_names_the_valid_ones(self) -> None:
        with pytest.raises(ValidationError) as err:
            _block(options={"candidat_k": [20]})
        message = str(err.value)
        assert "candidat_k" in message
        assert "candidate_k" in message

    def test_speed_only_option_cannot_be_an_axis(self) -> None:
        """Quét một knob không đổi kết quả sinh ra dòng bảng đọc như phát hiện."""
        (name,) = SPEED_ONLY_OPTIONS
        with pytest.raises(ValidationError) as err:
            _block(mode="reranked", options={name: [16, 64]})
        assert "knob" in str(err.value)

    def test_speed_only_option_may_still_be_pinned(self) -> None:
        """Ghim một giá trị khác mặc định là chuyện khác với quét nó."""
        (name,) = SPEED_ONLY_OPTIONS
        block = _block(mode="reranked", options={name: [64]})
        assert len(list(block.cells())) == 1

    def test_option_on_a_branch_that_ignores_it_raises_during_expand(self, workspace: Path) -> None:
        """`dense × k=1` nổ ở `expand()`, tức trước preflight và trước mọi model.

        `MatrixBlock` một mình **không** bắt được ca này: nó chỉ kiểm *tên* tham số
        có hợp lệ hay không, và `k` là tên hợp lệ. Chuyện "nhánh này có nhận `k`
        không" chỉ trả lời được khi đã ghép index × mode × option thành một ô, tức
        ở `ExperimentCell`. Nên phép kiểm nằm ở đó, và `run_experiment` gọi
        `expand()` ở dòng đầu chính vì thế.
        """
        config = _config(
            workspace,
            [
                {
                    "index_config": [str(_index_yaml(workspace / "configs", "a"))],
                    "retrieval_mode": ["dense"],
                    "options": {"k": [1]},
                }
            ],
        )
        with pytest.raises(ValidationError) as err:
            expand(config)
        assert "dense" in str(err.value)

    def test_a_valid_rerank_base_expands(self, workspace: Path) -> None:
        config = _config(
            workspace,
            [
                {
                    "index_config": [str(_index_yaml(workspace / "configs", "a"))],
                    "retrieval_mode": ["reranked"],
                    "options": {"base": ["dense"]},
                }
            ],
        )
        assert len(expand(config)) == 1

    def test_rerank_base_reranked_raises(self, workspace: Path) -> None:
        """Xếp lại hai lần bằng cùng model: không đổi thứ hạng, chỉ đôi chi phí."""
        config = _config(
            workspace,
            [
                {
                    "index_config": [str(_index_yaml(workspace / "configs", "a"))],
                    "retrieval_mode": ["reranked"],
                    "options": {"base": ["reranked"]},
                }
            ],
        )
        with pytest.raises(ValidationError, match="hai lần"):
            expand(config)

    def test_two_cells_with_the_same_run_name_raise(self, workspace: Path) -> None:
        """Cùng tên = ghi lên cùng ba file, ô sau xoá ô trước, bảng vẫn đủ dòng."""
        index = str(_index_yaml(workspace / "configs", "a"))
        config = _config(
            workspace,
            [
                {"index_config": [index], "retrieval_mode": ["dense"]},
                {"index_config": [index], "retrieval_mode": ["dense"]},
            ],
        )
        with pytest.raises(ValueError, match="cùng `run_name`"):
            expand(config)

    def test_pinned_options_are_left_out_of_the_name_but_kept_in_the_cell(
        self, workspace: Path
    ) -> None:
        """Giá trị ghim không phân biệt ô nào với ô nào, nên nó không vào tên.

        Nhưng nó **phải** ở lại trong `branch_options`, vì đó là thứ đi vào
        `config` của báo cáo và là thứ làm báo cáo tự mô tả được chính nó.
        """
        config = _config(
            workspace,
            [
                {
                    "index_config": [str(_index_yaml(workspace / "configs", "a"))],
                    "retrieval_mode": ["reranked"],
                    "options": {"rerank_candidates": [20, 50], "rerank_dtype": ["float16"]},
                }
            ],
        )
        cells = expand(config)
        assert [c.run_name for c in cells] == ["a-reranked-rc20", "a-reranked-rc50"]
        assert all(c.branch_options["rerank_dtype"] == "float16" for c in cells)


class TestOptionNamesComeFromRagCore:
    """Danh sách tham số hợp lệ **không** được chép lại ở tầng pipeline."""

    def test_branch_options_is_exactly_the_union_rag_core_declares(self) -> None:
        assert BRANCH_OPTIONS == HYBRID_OPTIONS | RERANK_OPTIONS

    def test_hybrid_options_matches_the_real_constructor_signature(self) -> None:
        """Thêm tham số cho `QdrantHybridRetriever` mà quên `HYBRID_OPTIONS` → đỏ.

        Không có test này thì `HYBRID_OPTIONS` là một bản chép tay của chữ ký, và
        bản chép tay sẽ lệch — lệch theo hướng **từ chối một tham số hợp lệ**.
        """
        import inspect

        from rag_core.retrieval import QdrantHybridRetriever

        signature = inspect.signature(QdrantHybridRetriever.__init__)
        keyword_only = {
            name
            for name, param in signature.parameters.items()
            if param.kind is inspect.Parameter.KEYWORD_ONLY
        }
        assert keyword_only == set(HYBRID_OPTIONS)


class TestBranchValidationCannotDrift:
    """`check_branch_options` và `build_branch` phải nổ ở cùng những đầu vào.

    Đây là lý do `check_branch_options` được tách ra ở `W2-07`. Nếu nó lệch khỏi
    `build_branch` thì preflight **cho qua** những ô sẽ chết giữa grid — đúng cái
    chế độ hỏng hạng mục này tồn tại để chặn, chỉ là được dựng lên bởi chính bản
    sửa của nó.
    """

    CASES: tuple[tuple[str, dict[str, Any]], ...] = (
        ("dense", {}),
        ("sparse", {}),
        ("hybrid", {}),
        ("hybrid", {"k": 1, "candidate_k": 20}),
        ("hybrid", {"k": None}),
        ("dense", {"k": 1}),
        ("sparse", {"candidate_k": 100}),
        ("hybrid", {"candidat_k": 20}),
        ("hybrid", {"rerank_candidates": 50}),
        ("reranked", {}),
        ("reranked", {"base": "dense"}),
        ("reranked", {"base": "reranked"}),
        ("reranked", {"base": "dense", "k": 1}),
        ("reranked", {"base": "hybrid", "k": 1, "candidate_k": 20}),
        ("nonsense", {}),
    )

    @pytest.mark.parametrize(("mode", "options"), CASES)
    def test_both_agree(self, mode: str, options: dict[str, Any]) -> None:
        from rag_core.embedding import HashingEmbeddingProvider
        from rag_core.retrieval import QdrantDenseRetriever, build_branch, check_branch_options

        def raises(call: Any) -> bool:
            try:
                call()
            except ValueError:
                return True
            except Exception:
                # `build_branch` với nhánh `reranked` sẽ cố tải cross-encoder thật.
                # Đó không phải bất đồng về *tính hợp lệ* — nó có nghĩa là phép
                # kiểm đã cho qua, tức trùng với `check_branch_options`.
                return False
            return False

        store = QdrantDenseRetriever(
            HashingEmbeddingProvider(dimension=32, sparse=True), collection="rag_test_drift"
        )
        assert raises(lambda: check_branch_options(mode, options)) == raises(
            lambda: build_branch(store, mode, **options)
        ), f"{mode} {options}: preflight và build_branch không đồng ý"


# ----------------------------------------------------------------- fingerprint


class TestFingerprintIsTheIdentityOfACell:
    """Resume so `fingerprint`. Mọi thứ đổi kết quả phải đổi nó."""

    def _cell(self, **overrides: Any) -> ExperimentCell:
        base: dict[str, Any] = {
            "run_name": "x",
            "index_config": Path("configs/indexing/a.yaml"),
            "retrieval_mode": RetrievalMode.HYBRID,
            "branch_options": {"k": 1},
            "top_k": 20,
            "min_overlap_ratio": 0.5,
            "golden": Path("g.jsonl"),
        }
        return ExperimentCell.model_validate(base | overrides)

    def test_same_inputs_give_the_same_fingerprint(self) -> None:
        args = {"index_fingerprint": "abc", "golden_digest": "def"}
        assert self._cell().fingerprint(**args) == self._cell().fingerprint(**args)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"branch_options": {"k": 60}},
            {"retrieval_mode": RetrievalMode.DENSE, "branch_options": {}},
            {"top_k": 50},
            {"min_overlap_ratio": 0.8},
            {"index_config": Path("configs/indexing/b.yaml")},
        ],
        ids=["option", "mode", "top_k", "overlap", "index_path"],
    )
    def test_every_field_that_changes_results_changes_the_fingerprint(
        self, overrides: dict[str, Any]
    ) -> None:
        args = {"index_fingerprint": "abc", "golden_digest": "def"}
        assert self._cell().fingerprint(**args) != self._cell(**overrides).fingerprint(**args)

    def test_rebuilding_the_index_differently_changes_the_fingerprint(self) -> None:
        """Chuỗi YAML của ô không đổi một ký tự, nhưng kết quả thì đổi.

        Đây là ca mà resume theo tên file **không** bắt được: `bgem3.yaml` được
        sửa `chunk_size` rồi build lại, và mọi ô đọc index đó phải chạy lại.
        """
        cell = self._cell()
        assert cell.fingerprint(index_fingerprint="v1", golden_digest="g") != cell.fingerprint(
            index_fingerprint="v2", golden_digest="g"
        )

    def test_a_new_golden_set_changes_every_fingerprint(self) -> None:
        """`TD-13` sẽ review lại golden set và ghi lại **cùng đường dẫn**."""
        cell = self._cell()
        assert cell.fingerprint(index_fingerprint="v", golden_digest="g1") != cell.fingerprint(
            index_fingerprint="v", golden_digest="g2"
        )

    def test_windows_and_posix_paths_agree(self) -> None:
        """`fingerprint` không được đổi theo hệ điều hành đang chạy.

        Nếu nó đổi thì grid chạy trên laptop Windows và grid chạy trên pod Linux
        (`W0-05`) coi mọi ô là khác nhau, và không lần nào resume được lần kia.
        """
        args = {"index_fingerprint": "abc", "golden_digest": "def"}
        posix = self._cell(index_config=Path("configs/indexing/a.yaml"))
        windows = self._cell(index_config=Path("configs\\indexing\\a.yaml"))
        assert posix.fingerprint(**args) == windows.fingerprint(**args)


class TestGoldenDigestHashesContent:
    def test_editing_the_file_changes_the_digest(self, tmp_path: Path) -> None:
        path = tmp_path / "g.jsonl"
        path.write_text("a\n", encoding="utf-8")
        before = golden_digest(path)
        path.write_text("b\n", encoding="utf-8")
        assert golden_digest(path) != before


# ---------------------------------------------------------------------- resume


def _fingerprints(cells: Sequence[ExperimentCell], value: str = "idx") -> dict[str, str]:
    return {str(cell.index_config): value for cell in cells}


class TestResume:
    @pytest.fixture
    def grid(self, workspace: Path) -> tuple[ExperimentConfig, tuple[ExperimentCell, ...]]:
        config = _config(
            workspace,
            [
                {
                    "index_config": [str(_index_yaml(workspace / "configs", "a"))],
                    "retrieval_mode": ["hybrid"],
                    "options": {"k": [1, 60]},
                }
            ],
        )
        return config, expand(config)

    def _done(self, cell: ExperimentCell, fingerprint: str) -> CellRecord:
        return CellRecord(
            run_name=cell.run_name,
            fingerprint=fingerprint,
            status="done",
            finished_at="2026-08-21T00:00:00+00:00",
        )

    def test_a_finished_cell_is_skipped(
        self, grid: tuple[ExperimentConfig, tuple[ExperimentCell, ...]]
    ) -> None:
        _, cells = grid
        marks = _fingerprints(cells)
        first = cells[0].fingerprint(index_fingerprint="idx", golden_digest="g")
        state = ExperimentState(
            experiment="e", cells={cells[0].run_name: self._done(cells[0], first)}
        )
        todo, skipped = plan_cells(cells, state, marks, "g")
        assert skipped == [cells[0].run_name]
        assert [cell.run_name for cell, _ in todo] == [cells[1].run_name]

    def test_a_cell_whose_parameters_changed_runs_again(
        self, grid: tuple[ExperimentConfig, tuple[ExperimentCell, ...]]
    ) -> None:
        """Cái bẫy chính: state có ô này, nhưng nó đã chạy với tham số khác."""
        _, cells = grid
        state = ExperimentState(
            experiment="e", cells={cells[0].run_name: self._done(cells[0], "fingerprint-cũ")}
        )
        todo, skipped = plan_cells(cells, state, _fingerprints(cells), "g")
        assert skipped == []
        assert len(todo) == 2

    def test_rebuilding_the_index_reruns_everything(
        self, grid: tuple[ExperimentConfig, tuple[ExperimentCell, ...]]
    ) -> None:
        _, cells = grid
        state = ExperimentState(
            experiment="e",
            cells={
                cell.run_name: self._done(
                    cell, cell.fingerprint(index_fingerprint="v1", golden_digest="g")
                )
                for cell in cells
            },
        )
        _, skipped = plan_cells(cells, state, _fingerprints(cells, "v1"), "g")
        assert len(skipped) == 2
        todo, skipped = plan_cells(cells, state, _fingerprints(cells, "v2"), "g")
        assert skipped == []
        assert len(todo) == 2

    def test_a_failed_cell_is_retried(
        self, grid: tuple[ExperimentConfig, tuple[ExperimentCell, ...]]
    ) -> None:
        """`failed` không phải `done`. Một ô chết vì OOM phải được thử lại."""
        _, cells = grid
        record = self._done(
            cells[0], cells[0].fingerprint(index_fingerprint="idx", golden_digest="g")
        )
        record.status = "failed"
        state = ExperimentState(experiment="e", cells={cells[0].run_name: record})
        todo, skipped = plan_cells(cells, state, _fingerprints(cells), "g")
        assert skipped == []
        assert len(todo) == 2

    def test_no_resume_runs_everything(
        self, grid: tuple[ExperimentConfig, tuple[ExperimentCell, ...]]
    ) -> None:
        _, cells = grid
        state = ExperimentState(
            experiment="e",
            cells={
                cell.run_name: self._done(
                    cell, cell.fingerprint(index_fingerprint="idx", golden_digest="g")
                )
                for cell in cells
            },
        )
        todo, skipped = plan_cells(cells, state, _fingerprints(cells), "g", resume=False)
        assert skipped == []
        assert len(todo) == 2

    def test_cells_are_grouped_by_index_keeping_declaration_order(self, workspace: Path) -> None:
        """Gom để nạp model một lần, nhưng thứ tự phải suy được từ file config."""
        first = str(_index_yaml(workspace / "configs", "a"))
        second = str(_index_yaml(workspace / "configs", "b", chunk_size=550))
        config = _config(
            workspace,
            [
                {"index_config": [first, second], "retrieval_mode": ["dense"]},
                {"index_config": [first], "retrieval_mode": ["sparse"]},
            ],
        )
        cells = expand(config)
        todo, _ = plan_cells(cells, ExperimentState(experiment="e"), _fingerprints(cells), "g")
        indexes = [Path(cell.index_config).stem for cell, _ in todo]
        # `a` xuất hiện trước trong YAML nên cả nhóm của nó chạy trước, và trong
        # nhóm thì dense (khối 1) trước sparse (khối 2).
        assert indexes == ["a", "a", "b"]
        assert [cell.retrieval_mode.value for cell, _ in todo] == ["dense", "sparse", "dense"]


# ------------------------------------------------------------------- preflight


class TestPreflightFailsBeforeAnyCellRuns:
    def _cells(self, workspace: Path, **extra: Any) -> tuple[ExperimentConfig, Any]:
        config = _config(
            workspace,
            [
                {
                    "index_config": [str(_index_yaml(workspace / "configs", "a"))],
                    "retrieval_mode": ["dense"],
                }
            ],
            **extra,
        )
        return config, expand(config)

    def test_clean_grid_returns_fingerprints_and_digest(self, workspace: Path) -> None:
        config, cells = self._cells(workspace)
        fingerprints, digest, tracker = preflight(config, cells, ExperimentState(experiment="e"))
        assert set(fingerprints) == {str(cells[0].index_config)}
        assert len(digest) == 64
        # Không khai `tracking_uri` nên tracker là `NullTracker` — và preflight là
        # chỗ mở nó, nên nó phải đi ra cùng hai giá trị kia.
        assert isinstance(tracker, NullTracker)

    def test_missing_golden_set_is_caught(self, workspace: Path) -> None:
        config, cells = self._cells(workspace, golden=str(workspace / "nope.jsonl"))
        with pytest.raises(PreflightError, match="golden set"):
            preflight(config, cells, ExperimentState(experiment="e"))

    def test_unreadable_index_config_is_caught_without_loading_a_model(
        self, workspace: Path
    ) -> None:
        bad = workspace / "configs" / "bad.yaml"
        bad.write_text("chunking: {chunk_size: -5}\n", encoding="utf-8")
        config = _config(workspace, [{"index_config": [str(bad)], "retrieval_mode": ["dense"]}])
        with pytest.raises(PreflightError, match="không đọc được"):
            preflight(config, expand(config), ExperimentState(experiment="e"))

    def test_an_existing_report_this_state_does_not_own_is_refused(self, workspace: Path) -> None:
        """Kiểm quan trọng nhất — và nó bắt lỗi thật ngay lần dry-run đầu tiên.

        `plans/reports/runs/` đang giữ 57 file bằng chứng của `W2-01`…`W2-05`.
        Grid `exp-001` sinh ra ô tên `bgem3-sparse`, trùng đúng báo cáo tiêu đề
        của `W2-03`, và không có kiểm này thì nó bị ghi đè không tiếng động.
        """
        config, cells = self._cells(workspace)
        (workspace / "runs" / f"{cells[0].run_name}-retrieval.json").write_text("{}", "utf-8")
        with pytest.raises(PreflightError, match="bằng chứng của một lần chạy khác"):
            preflight(config, cells, ExperimentState(experiment="e"))

    def test_a_report_this_state_does_own_is_fine(self, workspace: Path) -> None:
        """Resume phải chạy được: ô của chính grid này thì ghi đè là đúng."""
        config, cells = self._cells(workspace)
        (workspace / "runs" / f"{cells[0].run_name}-retrieval.json").write_text("{}", "utf-8")
        state = ExperimentState(
            experiment="e",
            cells={
                cells[0].run_name: CellRecord(
                    run_name=cells[0].run_name,
                    fingerprint="f",
                    status="done",
                    finished_at="2026-08-21T00:00:00+00:00",
                )
            },
        )
        preflight(config, cells, state)

    def test_force_overrides_the_ownership_check(self, workspace: Path) -> None:
        config, cells = self._cells(workspace)
        (workspace / "runs" / f"{cells[0].run_name}-retrieval.json").write_text("{}", "utf-8")
        preflight(config, cells, ExperimentState(experiment="e"), force=True)

    def test_all_problems_are_reported_at_once(self, workspace: Path) -> None:
        """Sửa một lỗi rồi chạy lại để thấy lỗi kế tiếp = một lần nạp model mỗi vòng."""
        bad = workspace / "configs" / "bad.yaml"
        bad.write_text("chunking: {chunk_size: -5}\n", encoding="utf-8")
        config = _config(
            workspace,
            [{"index_config": [str(bad)], "retrieval_mode": ["dense"]}],
            golden=str(workspace / "nope.jsonl"),
        )
        with pytest.raises(PreflightError) as err:
            preflight(config, expand(config), ExperimentState(experiment="e"))
        assert "2 vấn đề" in str(err.value)


# ----------------------------------------------------------------------- state


class TestStateSurvivesInterruption:
    def test_save_then_load_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "s.json"
        state = ExperimentState(experiment="e", golden_digest="d")
        state.cells["x"] = CellRecord(run_name="x", fingerprint="f", status="done", finished_at="t")
        state.save(path)
        assert ExperimentState.load(path, "e").cells["x"].fingerprint == "f"

    def test_save_leaves_no_temp_file_behind(self, tmp_path: Path) -> None:
        path = tmp_path / "s.json"
        ExperimentState(experiment="e").save(path)
        assert [p.name for p in tmp_path.iterdir()] == ["s.json"]

    def test_a_truncated_state_file_starts_over_instead_of_crashing(self, tmp_path: Path) -> None:
        """Ctrl+C đúng lúc ghi để lại JSON cắt dở. Chạy lại tệ hơn resume, nhưng
        tốt hơn nhiều so với chết và bắt người dùng tự xoá file."""
        path = tmp_path / "s.json"
        path.write_text('{"experiment": "e", "cells": {"x": ', encoding="utf-8")
        assert ExperimentState.load(path, "e").cells == {}

    def test_missing_state_is_an_empty_state_not_an_error(self, tmp_path: Path) -> None:
        assert ExperimentState.load(tmp_path / "nope.json", "e").cells == {}

    def test_state_records_the_metrics_needed_to_read_it_without_the_reports(
        self, tmp_path: Path
    ) -> None:
        """State phải tự nói được ô nào cho số gì, không thì nó chỉ là danh sách tên."""
        path = tmp_path / "s.json"
        state = ExperimentState(experiment="e")
        state.cells["x"] = CellRecord(
            run_name="x",
            fingerprint="f",
            status="done",
            finished_at="t",
            metrics={"ndcg@10": 0.6481},
        )
        state.save(path)
        assert json.loads(path.read_text(encoding="utf-8"))["cells"]["x"]["metrics"]["ndcg@10"]


# -------------------------------------------------------------------- tracking


class TestTracking:
    """Ba đường ra của `open_tracker`, và chúng khác nhau có chủ đích.

    Lượt chạy grid đầu tiên của `W2-07` là lý do khối này tồn tại: mlflow 3.15 từ
    chối `file:./mlruns`, `build_tracker` cũ rơi về `NullTracker` kèm một cảnh
    báo, và grid chạy trọn 14 ô **không ghi gì lên MLflow** — Evidence của DoD
    không tồn tại, và cảnh báo duy nhất là dòng 19 của một log 2320 dòng.
    """

    def test_no_uri_means_no_tracking_silently(self) -> None:
        """`tracking_uri: null` là một lựa chọn tường minh, không phải một lỗi."""
        assert isinstance(open_tracker(None, "e"), NullTracker)
        assert isinstance(open_tracker("", "e"), NullTracker)

    def test_missing_mlflow_degrades_to_null_instead_of_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Extra `tracking` là tuỳ chọn có chủ đích.

        Grid 40 phút không được chết ở giây thứ nhất vì thiếu thư viện xem lại.
        Đây là đường DUY NHẤT còn được rơi về `NullTracker` khi đã khai `uri`.
        """
        import builtins

        real = builtins.__import__

        def fake(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "mlflow":
                raise ImportError("no mlflow")
            return real(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake)
        assert isinstance(open_tracker("sqlite:///x.db", "e"), NullTracker)

    def test_an_unusable_uri_raises_so_preflight_catches_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Khai một đích đến mà không tới được là **lỗi config**.

        Nó phải nổ ở giây thứ nhất, không im lặng ở phút thứ 25. Đúng ca thật:
        mlflow >= 3 bỏ file store, nên `file:./mlruns` không mở được nữa.
        """
        import pipeline.experiments.tracking as module

        def boom(uri: str, experiment: str) -> None:
            raise RuntimeError("file store is in maintenance mode")

        monkeypatch.setattr(module, "MlflowTracker", boom)
        with pytest.raises(TrackingUnavailable) as err:
            open_tracker("file:./mlruns", "e")
        # Thông báo phải nói cả hai đường ra, không chỉ báo là hỏng.
        assert "tracking_uri: null" in str(err.value)
        assert "sqlite" in str(err.value)

    def test_preflight_reports_a_bad_tracking_uri_as_a_problem(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pipeline.experiments.tracking as module

        def boom(uri: str, experiment: str) -> None:
            raise RuntimeError("nope")

        monkeypatch.setattr(module, "MlflowTracker", boom)
        config = _config(
            workspace,
            [
                {
                    "index_config": [str(_index_yaml(workspace / "configs", "a"))],
                    "retrieval_mode": ["dense"],
                }
            ],
            tracking_uri="file:./mlruns",
        )
        with pytest.raises(PreflightError, match="Không mở được MLflow"):
            preflight(config, expand(config), ExperimentState(experiment="e"))

    def test_null_run_is_a_usable_context_manager(self) -> None:
        with open_tracker(None, "e").start_run("r", {}) as run:
            run.log_params({"a": 1})
            run.log_metrics({"m": 1.0})
            run.set_failed("boom")
            assert run.run_id is None


class TestTrackingFailuresNeverLoseResults:
    """Một ô đã chạy 131 giây không được mất vì tracking server chết giữa grid."""

    class _Broken:
        def start_run(self, run_name: str, tags: dict[str, str]) -> Any:
            raise RuntimeError("server đã tắt")

    class _BrokenRun:
        run_id = "abc"

        def log_params(self, params: dict[str, Any]) -> None:
            raise RuntimeError("boom")

        def log_metrics(self, metrics: dict[str, float]) -> None:
            raise RuntimeError("boom")

        def log_artifact(self, path: Path) -> None:
            raise RuntimeError("boom")

        def set_failed(self, error: str) -> None:
            raise RuntimeError("boom")

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            raise RuntimeError("boom")

    def test_start_run_failing_gives_a_usable_null_run(self) -> None:
        with SafeTracker(self._Broken()).start_run("r", {}) as run:
            run.log_metrics({"m": 1.0})
            assert run.run_id is None

    def test_every_log_call_survives_a_broken_run(self) -> None:
        class Fine:
            def start_run(self, run_name: str, tags: dict[str, str]) -> Any:
                return TestTrackingFailuresNeverLoseResults._BrokenRun()

        with SafeTracker(Fine()).start_run("r", {}) as run:
            run.log_params({"a": 1})
            run.log_metrics({"m": 1.0})
            run.log_artifact(Path("x"))
            run.set_failed("e")
            # `run_id` đi qua `_guard` nên nó vẫn trả về giá trị thật khi được.
            assert run.run_id == "abc"


class TestCellTable:
    def test_every_cell_gets_a_row(self, workspace: Path) -> None:
        config = _config(
            workspace,
            [
                {
                    "index_config": [str(_index_yaml(workspace / "configs", "a"))],
                    "retrieval_mode": ["hybrid"],
                    "options": {"k": [1, 60]},
                }
            ],
        )
        cells = expand(config)
        table = cell_table(cells)
        assert all(cell.run_name in table for cell in cells)
        assert len(table.splitlines()) == len(cells) + 2


# ---------------------------------------------------- MLflow là view, không nguồn


_PAYLOAD: dict[str, Any] = {
    "run_name": "e1-bgem3-dense",
    "overall": {"ndcg@10": 0.4442, "hit_rate@1": 0.3397},
    "latency_ms": {"p50": 30.1, "p95": 46.3},
    "n_scored": 209,
    "n_skipped_unanswerable": 33,
    "n_relevant_mean": 1.4258,
    "config": {
        "index_config": "configs/indexing/bgem3.yaml",
        "index_fingerprint": "0123456789abcdef",
        "collection": "rag_bgem3",
        "embedding_model": "BAAI/bge-m3",
        "retrieval_mode": "hybrid",
        "retriever": "qdrant-hybrid:rag_bgem3:rrf1-c20",
        "top_k": 20,
        "branch_options": {"k": 1, "candidate_k": 20},
        "chunking": {"chunk_size": 1000, "chunk_overlap": 100, "strategy": "recursive"},
    },
}


class TestMlflowIsAViewNotTheSourceOfTruth:
    """`tracking.py` tuyên bố ba file báo cáo là đủ. Khối này kiểm câu đó.

    Nếu không dựng lại được bảng MLflow từ `plans/reports/runs/` thì MLflow đã
    lặng lẽ trở thành nguồn sự thật, và `W2-09` sẽ dựa vào một chỗ mà repo không
    tái lập được.
    """

    def test_params_come_from_the_report_alone(self) -> None:
        params = report_params(_PAYLOAD)
        assert params["index_config"] == "bgem3.yaml"
        assert params["collection"] == "rag_bgem3"
        assert params["retrieval_mode"] == "hybrid"
        assert params["opt.k"] == 1
        assert params["chunk.chunk_size"] == 1000

    def test_index_fingerprint_is_shortened_not_dropped(self) -> None:
        """Đủ để nhóm các ô cùng index trên UI, không đủ dài để làm ngợp bảng."""
        assert report_params(_PAYLOAD)["index_fingerprint"] == "0123456789ab"

    def test_chunking_is_dumped_whole_so_a_new_field_cannot_be_forgotten(self) -> None:
        params = report_params(_PAYLOAD)
        assert {k for k in params if k.startswith("chunk.")} == {
            "chunk.chunk_size",
            "chunk.chunk_overlap",
            "chunk.strategy",
        }

    def test_metrics_include_the_label_distribution_that_g2_depends_on(self) -> None:
        """`n_relevant_mean` là chiều `chunk_size` nhìn từ phía nhãn.

        Nó là mẫu số của recall@k, và là lý do `compare.py` từ chối so hai index
        khác `chunk_size`. Có nó trên bảng MLflow thì hai nhóm ô không so được với
        nhau tự tách ra bằng mắt.
        """
        metrics = report_metrics(_PAYLOAD)
        assert metrics["n_relevant_mean"] == 1.4258
        assert metrics["latency_p95"] == 46.3
        assert metrics["ndcg@10"] == 0.4442

    def test_a_report_without_optional_fields_does_not_crash(self) -> None:
        """Báo cáo cũ (trước `W2-07`) không có `n_relevant_mean`."""
        metrics = report_metrics({"overall": {"ndcg@10": 0.1}})
        assert metrics == {"ndcg@10": 0.1}

    def test_cell_params_carry_what_the_report_cannot_say(self) -> None:
        """Ba trường này là một lỗ hổng tái lập của định dạng báo cáo (`TD-19`)."""
        cell = ExperimentCell(
            run_name="x",
            index_config=Path("a.yaml"),
            retrieval_mode=RetrievalMode.DENSE,
            top_k=20,
            min_overlap_ratio=0.5,
            golden=Path("data/golden/golden_v1.jsonl"),
        )
        params = cell_params(cell, "abcdef0123456789")
        assert params == {
            "golden": "golden_v1.jsonl",
            "min_overlap_ratio": 0.5,
            "cell_fingerprint": "abcdef012345",
        }

    def test_the_report_file_holds_exactly_what_the_live_path_logged(self, tmp_path: Path) -> None:
        """Đây là điều làm hai đường **không thể** lệch nhau.

        `_run_one` log param từ `json.loads(report.to_json())`; `backfill` log từ
        `json.loads(file)`. Chúng trùng nhau khi và chỉ khi `_write_report` ghi
        đúng `to_json()` — nên đó là thứ được ghim, không phải sự cẩn thận của tôi.
        """
        from pipeline.eval.retrieval_eval import EvalReport

        report = EvalReport(
            run_name="r",
            created_at="2026-08-21T00:00:00+00:00",
            n_queries=1,
            n_scored=1,
            n_skipped_unanswerable=0,
            overall={"ndcg@10": 0.5},
        )
        _write_report(report, tmp_path)
        written = (tmp_path / "r-retrieval.json").read_text(encoding="utf-8")
        assert written == report.to_json()
        assert report_params(json.loads(written)) == report_params(json.loads(report.to_json()))

    def test_write_report_leaves_no_temp_files(self, tmp_path: Path) -> None:
        from pipeline.eval.retrieval_eval import EvalReport

        _write_report(
            EvalReport(
                run_name="r",
                created_at="t",
                n_queries=0,
                n_scored=0,
                n_skipped_unanswerable=0,
                overall={},
            ),
            tmp_path,
        )
        assert sorted(p.suffix for p in tmp_path.iterdir()) == [".json", ".jsonl", ".md"]


class TestBackfillSkipsWhatItCannotSee:
    def test_dry_run_counts_reports_without_touching_mlflow(
        self, workspace: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        index = str(_index_yaml(workspace / "configs", "a"))
        config = _config(
            workspace,
            [{"index_config": [index], "retrieval_mode": ["dense", "sparse"]}],
            tracking_uri="sqlite:///nope.db",
        )
        config_path = workspace / "exp.yaml"
        config_path.write_text(
            yaml.safe_dump(json.loads(config.model_dump_json())), encoding="utf-8"
        )
        cells = expand(config)
        (workspace / "runs" / f"{cells[0].run_name}-retrieval.json").write_text(
            json.dumps(_PAYLOAD), encoding="utf-8"
        )
        with caplog.at_level("INFO"):
            assert backfill(config_path, dry_run=True) == 0
        # Một ô có báo cáo, một ô chưa — backfill một grid đang chạy dở là hợp lý,
        # nên ô thiếu là thông tin chứ không phải lỗi.
        assert "1/2" in caplog.text
        assert cells[1].run_name in caplog.text


class TestMetricNamesSurviveMlflow:
    """MLflow từ chối `@`, và **mọi** metric của dự án này mang `@`.

    Đã hỏng thật: `backfill` log 14 ô, param vào hết, **metric không một cái nào**,
    và `SafeTracker` biến chuyện đó thành 14 dòng cảnh báo rồi báo "Đã log 14 ô".
    Bảng MLflow: 14 run, **0 cột metric**.
    """

    def test_at_becomes_something_a_human_can_read(self) -> None:
        assert mlflow_metric_name("ndcg@10") == "ndcg_at_10"
        assert mlflow_metric_name("hit_rate@1") == "hit_rate_at_1"

    def test_names_already_legal_are_untouched(self) -> None:
        for name in ("latency_p95", "n_scored", "n_relevant_mean", "mrr"):
            assert mlflow_metric_name(name) == name

    def test_every_metric_the_eval_produces_is_legal_after_mapping(self) -> None:
        """Đây là phép ghim thật, không phải bài test cho hàm đổi tên.

        `DEFAULT_K_VALUES` × họ metric là toàn bộ tên mà `evaluate_run` sinh ra.
        Thêm một metric mới mang ký tự MLflow không nhận thì test này đỏ **trước**
        khi một grid 13 phút ghi ra một bảng rỗng.
        """
        import re

        from pipeline.eval.retrieval_eval import DEFAULT_K_VALUES

        legal = re.compile(r"^[A-Za-z0-9_\-./ ]+$")
        names = ["mrr", "n_scored", "n_relevant_mean", "latency_p50", "latency_p95"]
        for k in DEFAULT_K_VALUES:
            names += [
                f"{family}@{k}" for family in ("hit_rate", "recall", "precision", "ndcg", "map")
            ]
        for name in names:
            mapped = mlflow_metric_name(name)
            assert legal.match(mapped), f"{name!r} → {mapped!r} vẫn không hợp lệ với MLflow"

    def test_the_map_is_a_display_layer_only(self) -> None:
        """File báo cáo giữ nguyên `@` — nó là nguồn sự thật, `compare.py` đọc nó."""
        assert "@" in METRIC_NAME_MAP
        assert set(report_metrics(_PAYLOAD)) & {"ndcg@10", "hit_rate@1"}
