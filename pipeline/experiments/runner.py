"""Chạy một grid tuần tự, resume được, ghi MLflow. `W2-07`.

## Ba thứ hạng mục này thật ra làm

`itertools.product` là một dòng. Phần còn lại là chỗ mất thời gian thật:

1. **Preflight.** Kiểm **toàn bộ** grid trước khi chạy ô đầu tiên. Một grid 12 ô
   mất ~40 phút; phát hiện ô thứ 11 gõ sai tên tham số ở phút thứ 33 là chế độ
   hỏng cần chặn, và cách chặn không phải "cẩn thận hơn" mà là kiểm trước. Kiểm
   được mà không nạp model nhờ `check_branch_options` (tách ra ở hạng mục này).

2. **Resume theo `fingerprint`, không theo tên file.** Xem
   `ExperimentCell.fingerprint`: "ô này đã chạy" và "ô này đã chạy *với đúng tham
   số hiện tại*" là hai câu khác nhau, và chỉ câu thứ hai cho phép bỏ qua.

3. **Gom ô theo index.** Đo được: 14 ô → **3** lần quét nhãn span thay vì 14.
   ⚠️ Không phải để tiết kiệm lần nạp model — `rag_core` đã có `lru_cache` trên cả
   ba loại model từ `W1`, nên model vốn đã được chia sẻ toàn tiến trình (dự đoán
   `D2` của tôi sai ở chỗ này). Cái gom lại thật sự mua là **quét nhãn span** và
   một tính chất: mỗi model nạp đúng một lần **bất kể** `lru_cache` to bằng bao
   nhiêu. Grid quét 5 model với `maxsize=4` mà chạy xen kẽ sẽ đá nhau ra khỏi
   cache và nạp lại liên tục; gom lại thì chuyện đó không xảy ra.

## Điều grid **không** đo tốt bằng probe riêng: độ trễ

Một grid chạy 40 phút liên tục trên GPU laptop, và `p95` của ô cuối chịu trạng
thái nhiệt khác ô đầu. `warmup=True` của `run_retrieval_eval` xử lý phần nạp
model, không xử lý phần đó. `W2-04` §6 và `W2-06` §5 đều đã dạy cùng một điều:
số độ trễ đáng tin là số đến từ một phép đo **được phân rã và có đối chứng thứ
tự**, không phải số đi kèm miễn phí với một phép đo khác. Cột `p95` trong bảng
grid dùng để *sàng*, không dùng để kết luận.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ..eval.golden import load_golden_set
from ..eval.retrieval_eval import IndexSession, open_index
from ..indexing.config import load_index_config
from .config import (
    ExperimentCell,
    ExperimentConfig,
    cell_table,
    expand,
    golden_digest,
    load_experiment_config,
)
from .tracking import NullTracker, TrackingUnavailable, open_tracker

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..eval.retrieval_eval import EvalReport
    from .tracking import Tracker

logger = logging.getLogger(__name__)

__all__ = [
    "CellRecord",
    "ExperimentState",
    "PreflightError",
    "cell_params",
    "git_sha",
    "plan_cells",
    "preflight",
    "report_metrics",
    "report_params",
    "run_experiment",
]


class PreflightError(Exception):
    """Grid không hợp lệ. Gom **tất cả** vấn đề, không báo lần lượt.

    Cùng lý lẽ với `Settings` ở `W1-01`: sửa một lỗi rồi chạy lại để thấy lỗi kế
    tiếp là vòng lặp mà mỗi vòng ở đây tốn một lần nạp model.
    """


class CellRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_name: str
    fingerprint: str
    status: str
    finished_at: str
    duration_s: float = 0.0
    mlflow_run_id: str | None = None
    error: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class ExperimentState(BaseModel):
    """File state của một grid. Là thứ duy nhất quyết định ô nào được bỏ qua."""

    model_config = ConfigDict(extra="forbid")

    experiment: str
    updated_at: str = ""
    golden_digest: str = ""
    cells: dict[str, CellRecord] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path, experiment: str) -> ExperimentState:
        if not path.exists():
            return cls(experiment=experiment)
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            # State hỏng không được làm mất grid: chạy lại từ đầu tệ hơn một chút
            # so với resume, nhưng tốt hơn nhiều so với chết và bắt người dùng tự
            # xoá file. Cảnh báo tường minh vì "grid chạy lại toàn bộ" mà không
            # nói lý do thì trông y như resume bị hỏng.
            logger.warning("State %s không đọc được (%s) — bắt đầu lại từ đầu.", path, exc)
            return cls(experiment=experiment)

    def save(self, path: Path) -> None:
        """Ghi **nguyên tử**. Crash giữa lúc ghi state không được sinh ra state rác.

        `os.replace` là atomic trên cùng một filesystem, kể cả Windows. Ghi trực
        tiếp thì một lần Ctrl+C đúng lúc để lại JSON cắt dở, và `load` ở trên sẽ
        coi cả grid là chưa chạy — mất 40 phút cho một lần bấm nhầm.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now(UTC).isoformat(timespec="seconds")
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except Exception:  # pragma: no cover - máy không có git
        return "unknown"
    return out.stdout.strip()


def preflight(
    config: ExperimentConfig,
    cells: Sequence[ExperimentCell],
    state: ExperimentState,
    *,
    force: bool = False,
) -> tuple[dict[str, str], str, Tracker]:
    """Kiểm cả grid, trả `(index_fingerprint theo đường dẫn, golden_digest, tracker)`.

    Gom mọi vấn đề rồi nổ một lần. Bốn nhóm được kiểm, và mỗi nhóm ứng với một
    cách grid có thể chạy 40 phút rồi cho kết quả vô dụng:

    1. **Golden set thiếu** → không có gì để chấm.
    2. **Config index không đọc được** → ô đó chết giữa grid.
    3. **File báo cáo đã tồn tại mà state này không sở hữu** → ô sẽ **ghi đè bằng
       chứng đã công bố**. `plans/reports/runs/` đang giữ 57 file của `W2-01`…
       `W2-05`; một ô tên `bgem3` sẽ xoá kết quả tiêu đề của `W2-01` và không có
       gì hỏi lại. Đây là kiểm quan trọng nhất trong bốn cái.
    4. **Tham số nhánh không hợp lệ** — đã kiểm ở `ExperimentCell`, nên tới đây
       là xong; ghi lại để danh sách này đọc được như một danh sách đầy đủ.
    5. **`tracking_uri` khai mà không mở được** → xem `open_tracker`. Thêm vào đây
       sau khi lượt chạy grid đầu tiên chạy trọn 14 ô mà không ghi được gì lên
       MLflow, tức Evidence của DoD không tồn tại và cảnh báo duy nhất nằm ở dòng
       19 của một log 2320 dòng.
    """
    problems: list[str] = []
    tracker: Tracker = NullTracker()
    try:
        tracker = open_tracker(config.tracking_uri, config.mlflow_experiment)
    except TrackingUnavailable as exc:
        problems.append(str(exc))

    if not config.golden.exists():
        problems.append(
            f"Không thấy golden set tại {config.golden}. Golden set được tạo ở "
            "`W1-10`/`W1-11` — chưa có thì chưa chạy eval được."
        )

    fingerprints: dict[str, str] = {}
    for path in dict.fromkeys(cell.index_config for cell in cells):
        try:
            fingerprints[str(path)] = load_index_config(path).fingerprint
        except Exception as exc:
            problems.append(f"Config index {path} không đọc được: {exc}")

    if not force:
        for cell in cells:
            report = config.out_dir / f"{cell.run_name}-retrieval.json"
            if report.exists() and cell.run_name not in state.cells:
                problems.append(
                    f"{report} đã tồn tại nhưng state của thí nghiệm này không có ô "
                    f"{cell.run_name!r} — nó là bằng chứng của một lần chạy khác và ô "
                    f"này sẽ ghi đè. Đổi `label` của khối, hoặc `--force` nếu đúng ý "
                    f"là ghi đè."
                )

    if problems:
        raise PreflightError(
            f"Grid {config.name!r} có {len(problems)} vấn đề, không chạy ô nào:\n  - "
            + "\n  - ".join(problems)
        )

    return fingerprints, golden_digest(config.golden), tracker


def plan_cells(
    cells: Sequence[ExperimentCell],
    state: ExperimentState,
    fingerprints: dict[str, str],
    digest: str,
    *,
    resume: bool = True,
) -> tuple[list[tuple[ExperimentCell, str]], list[str]]:
    """Chia grid thành (phải chạy, bỏ qua), gom theo index và giữ thứ tự khai báo.

    Gom **ổn định**: index nào xuất hiện trước thì cả nhóm của nó chạy trước, và
    trong nhóm giữ nguyên thứ tự YAML. Thứ tự chạy phải suy ra được từ file config
    — nếu không thì hai lần resume cho hai thứ tự khác nhau và log không so được.
    """
    todo: list[tuple[ExperimentCell, str]] = []
    skipped: list[str] = []
    for cell in cells:
        fingerprint = cell.fingerprint(
            index_fingerprint=fingerprints[str(cell.index_config)], golden_digest=digest
        )
        record = state.cells.get(cell.run_name)
        if resume and record is not None and record.status == "done":
            if record.fingerprint == fingerprint:
                skipped.append(cell.run_name)
                continue
            # Nói ra lý do. Một ô chạy lại mà không giải thích trông y như resume
            # hỏng, và người dùng sẽ đi sửa resume thay vì hiểu là tham số đã đổi.
            logger.info(
                "Ô %s chạy lại: fingerprint đổi (%s… → %s…) — tham số ô, config "
                "index, hoặc golden set đã khác lần trước.",
                cell.run_name,
                record.fingerprint[:8],
                fingerprint[:8],
            )
        todo.append((cell, fingerprint))

    order: dict[str, int] = {}
    for cell, _ in todo:
        order.setdefault(str(cell.index_config), len(order))
    todo.sort(key=lambda item: order[str(item[0].index_config)])
    return todo, skipped


def _release(session: IndexSession | None) -> None:
    """Trả bộ đệm CUDA trung gian trước khi mở index kế tiếp.

    ⚠️ **Hàm này KHÔNG giải phóng trọng số model, và bản đầu của nó nói là có.**
    Cả ba loại model trong `rag_core` đều nạp qua `lru_cache`
    (`embedding.huggingface._load_model` maxsize=4, `embedding.bge_m3._load_sparse_head`
    maxsize=4, `reranking.cross_encoder` maxsize=2), nên cache giữ tham chiếu mạnh
    và `del` + `gc.collect()` ở đây không chạm được vào chúng. Đo được: 4517/8188
    MiB VRAM trong lúc grid chạy nhánh `reranked`, đúng bằng BGE-M3 + cross-encoder
    cùng nằm đó.

    Nó vẫn làm một việc thật: bỏ store/client và ép `empty_cache()` trả lại phần
    bộ đệm hoạt hoá của những truy vấn index trước — không nhỏ với batch 512 token.

    **Hệ quả quan trọng hơn chính hàm này:** trần VRAM của một grid do ba con số
    `maxsize` ở `rag_core` quyết định, **không** do runner. Một grid quét 4 model
    embedding sẽ giữ 4 × 2,2 GB và OOM trên card 8 GB, và runner không có cách nào
    ngăn. Đây là đầu vào cụ thể cho `W0-06` (ngân sách VRAM).
    """
    if session is None:
        return
    del session
    gc.collect()
    try:
        import torch
    except ImportError:  # pragma: no cover - máy không có torch
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_report(report: EvalReport, out_dir: Path) -> list[Path]:
    """Ghi ba file của một ô, **nguyên tử từng file**.

    Crash giữa lúc ghi JSON để lại một file cắt dở mà `compare.py` sẽ đọc và
    `json.loads` sẽ nổ ở một chỗ chẳng liên quan gì. Ghi tmp rồi `os.replace` cho
    mỗi file, và state chỉ được cập nhật **sau** cả ba — nên một ô crash giữa
    đường luôn được chạy lại nguyên vẹn, không bao giờ nửa vời.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    payloads = {
        f"{report.run_name}-retrieval.md": report.to_markdown(),
        f"{report.run_name}-retrieval.json": report.to_json(),
        f"{report.run_name}-per-query.jsonl": report.to_jsonl() + "\n",
    }
    for name, text in payloads.items():
        target = out_dir / name
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
        written.append(target)
    return written


def report_params(payload: dict[str, Any]) -> dict[str, Any]:
    """Param MLflow suy **từ chính file báo cáo**, không từ đối tượng đang sống.

    Đây là chỗ `pipeline/experiments/backfill.py` và đường chạy trực tiếp gặp
    nhau, và việc chúng gặp nhau ở đây là có chủ đích: `tracking.py` tuyên bố ba
    file báo cáo là đủ để dựng lại view MLflow, và cách duy nhất để câu đó luôn
    đúng là **cả hai đường đọc cùng một nguồn**. Nếu đường trực tiếp lấy param từ
    `IndexSession` còn backfill lấy từ JSON thì hai bảng MLflow sẽ lệch dần, và
    lệch theo hướng không ai kiểm.
    """
    config = payload.get("config", {})
    chunking = config.get("chunking", {})
    params: dict[str, Any] = {
        "index_config": Path(str(config.get("index_config", ""))).name,
        "index_fingerprint": str(config.get("index_fingerprint", ""))[:12],
        "collection": config.get("collection"),
        "embedding_model": config.get("embedding_model"),
        "retrieval_mode": config.get("retrieval_mode"),
        "retriever": config.get("retriever"),
        "top_k": config.get("top_k"),
    }
    # Dump **toàn bộ** `ChunkingConfig` thay vì chọn tay vài trường: chiều
    # `chunk_size` của `W2-08` là chiều duy nhất đổi phân bố nhãn (`G2`), nên khi
    # lọc bảng MLflow theo nó thì phải thấy đủ những trường đi kèm — và liệt kê
    # tay ở đây sẽ lệch ngay lần `ChunkingConfig` có trường mới.
    params.update({f"chunk.{name}": value for name, value in chunking.items()})
    params.update({f"opt.{k}": v for k, v in config.get("branch_options", {}).items()})
    return params


def cell_params(cell: ExperimentCell, fingerprint: str) -> dict[str, Any]:
    """Param chỉ **ô** biết, không có trong báo cáo.

    ⚠️ Cả ba trường ở đây là một lỗ hổng tái lập của chính định dạng báo cáo, phát
    hiện khi viết `backfill.py`: một file `*-retrieval.json` **không nói được nó
    được đo trên golden set nào** và với `min_overlap_ratio` bao nhiêu. Hôm nay
    chỉ có một golden set nên không ai thấy; `TD-13` sẽ tạo ra cái thứ hai với
    **cùng đường dẫn**, và lúc đó hai báo cáo cạnh nhau sẽ không phân biệt được.
    Đã ghi thành `TD-19`.
    """
    return {
        "golden": cell.golden.name,
        "min_overlap_ratio": cell.min_overlap_ratio,
        "cell_fingerprint": fingerprint[:12],
    }


def report_metrics(payload: dict[str, Any]) -> dict[str, float]:
    """Metric MLflow suy từ file báo cáo. Cùng lý lẽ như `report_params`."""
    metrics = {name: float(value) for name, value in payload.get("overall", {}).items()}
    metrics.update(
        {f"latency_{name}": float(v) for name, v in payload.get("latency_ms", {}).items()}
    )
    for name in ("n_scored", "n_skipped_unanswerable", "n_relevant_mean"):
        value = payload.get(name)
        if value is not None:
            # `n_relevant_mean` là chiều `chunk_size` nhìn từ phía nhãn: nó đổi mẫu
            # số của recall@k, và đó là lý do `compare.py` từ chối so hai index
            # khác `chunk_size` (`G2`). Có nó trên bảng MLflow thì hai nhóm ô không
            # so được với nhau tự tách ra bằng mắt.
            metrics[name] = float(value)
    return metrics


def run_experiment(
    config: ExperimentConfig,
    *,
    resume: bool = True,
    force: bool = False,
    keep_going: bool = False,
    dry_run: bool = False,
) -> ExperimentState:
    """Chạy hết grid. Trả về state cuối cùng."""
    cells = expand(config)
    state = ExperimentState.load(config.state_path, config.name)

    logger.info("Grid %r: %d ô\n%s", config.name, len(cells), cell_table(cells))
    if dry_run:
        # Vẫn preflight: `--dry-run` mà không kiểm gì thì nó chỉ là một cách in
        # `itertools.product` ra màn hình, và người dùng sẽ tin nó nhiều hơn mức
        # nó đáng tin.
        preflight(config, cells, state, force=force)
        logger.info("Preflight sạch. `--dry-run` nên không chạy ô nào.")
        return state

    fingerprints, digest, tracker = preflight(config, cells, state, force=force)
    if state.golden_digest and state.golden_digest != digest:
        logger.warning(
            "Golden set đã đổi từ lần chạy trước (%s… → %s…) — mọi ô sẽ chạy lại. "
            "Số cũ và số mới đo hai tập câu hỏi khác nhau nên không trộn được.",
            state.golden_digest[:8],
            digest[:8],
        )
    state.golden_digest = digest

    todo, skipped = plan_cells(cells, state, fingerprints, digest, resume=resume)
    if skipped:
        logger.info("Bỏ qua %d ô đã xong: %s", len(skipped), ", ".join(skipped))
    if not todo:
        logger.info("Không còn ô nào phải chạy.")
        return state

    queries = load_golden_set(config.golden)
    base_tags = {"task": "W2-07", "experiment": config.name, "git_sha": git_sha(), **config.tags}

    session: IndexSession | None = None
    open_path: str | None = None
    failures = 0
    try:
        for number, (cell, fingerprint) in enumerate(todo, start=1):
            if open_path != str(cell.index_config):
                _release(session)
                session = None
                logger.info("Mở index %s", cell.index_config)
                session = open_index(cell.index_config)
                open_path = str(cell.index_config)
            assert session is not None
            failures += _run_one(
                cell,
                fingerprint,
                session=session,
                queries=queries,
                config=config,
                state=state,
                tracker=tracker,
                tags=base_tags,
                position=f"{number}/{len(todo)}",
                keep_going=keep_going,
            )
    finally:
        _release(session)

    done = sum(1 for record in state.cells.values() if record.status == "done")
    logger.info(
        "Grid %r: %d/%d ô xong · %d ô lỗi · state %s",
        config.name,
        done,
        len(cells),
        failures,
        config.state_path,
    )
    return state


def _run_one(
    cell: ExperimentCell,
    fingerprint: str,
    *,
    session: IndexSession,
    queries: Sequence[Any],
    config: ExperimentConfig,
    state: ExperimentState,
    tracker: Tracker,
    tags: dict[str, str],
    position: str,
    keep_going: bool,
) -> int:
    """Chạy một ô, ghi file rồi ghi state. Trả 1 nếu lỗi (và `keep_going`)."""
    logger.info("[%s] %s (%s)", position, cell.run_name, cell.retrieval_mode.value)
    started = time.perf_counter()
    with tracker.start_run(cell.run_name, {**tags, "run_name": cell.run_name}) as run:
        try:
            report = session.eval_branch(
                queries,
                run_name=cell.run_name,
                top_k=cell.top_k,
                mode=cell.retrieval_mode.value,
                branch_options=cell.branch_options,
                min_overlap_ratio=cell.min_overlap_ratio,
            )
        except Exception as exc:
            if not keep_going:
                raise
            # `--keep-going`: một ô chết không được làm mất 11 ô đã chạy. Ghi
            # `failed` vào state để lần resume sau chạy lại đúng ô đó.
            logger.exception("Ô %s lỗi — bỏ qua vì --keep-going.", cell.run_name)
            run.set_failed(f"{type(exc).__name__}: {exc}")
            state.cells[cell.run_name] = CellRecord(
                run_name=cell.run_name,
                fingerprint=fingerprint,
                status="failed",
                finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
                duration_s=round(time.perf_counter() - started, 2),
                error=f"{type(exc).__name__}: {exc}",
            )
            state.save(config.state_path)
            return 1

        # Đi qua JSON đã tuần tự hoá thay vì qua `EvalReport` đang sống: đó là
        # đúng byte mà `backfill.py` sẽ đọc lại, nên hai đường không thể lệch.
        payload = json.loads(report.to_json())
        metrics = report_metrics(payload)
        run.log_params(report_params(payload) | cell_params(cell, fingerprint))
        run.log_metrics(metrics)
        # File trước, state sau. Đảo lại thì một crash giữa hai bước để lại một ô
        # `done` không có báo cáo, và resume sẽ bỏ qua nó mãi mãi.
        for path in _write_report(report, config.out_dir):
            run.log_artifact(path)
        state.cells[cell.run_name] = CellRecord(
            run_name=cell.run_name,
            fingerprint=fingerprint,
            status="done",
            finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
            duration_s=round(time.perf_counter() - started, 2),
            mlflow_run_id=run.run_id,
            metrics={
                name: round(metrics[name], 4)
                for name in ("hit_rate@1", "hit_rate@5", "ndcg@10", "recall@5", "latency_p95")
                if name in metrics
            },
        )
        state.save(config.state_path)
    logger.info(
        "[%s] %s xong · nDCG@10 %.4f · hit@1 %.4f · p95 %.1f ms · %.1f s",
        position,
        cell.run_name,
        metrics.get("ndcg@10", float("nan")),
        metrics.get("hit_rate@1", float("nan")),
        metrics.get("latency_p95", float("nan")),
        time.perf_counter() - started,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chạy ma trận thí nghiệm truy hồi (W2-07)")
    parser.add_argument("--config", type=Path, required=True, help="YAML ở `configs/eval/`")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="In bảng grid + chạy preflight, không chạy ô nào và không nạp model.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Chạy lại cả những ô state ghi là đã xong. KHÔNG xoá state.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Một ô lỗi thì ghi `failed` và chạy tiếp, thay vì dừng cả grid.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Cho phép ghi đè file báo cáo mà state của thí nghiệm này không sở hữu. "
        "⚠️ `plans/reports/runs/` đang giữ bằng chứng của W2-01…W2-05.",
    )
    parser.add_argument("--tracking-uri", help="Ghi đè `tracking_uri` của file config.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr
    )
    # Một grid 14 ô ghi ~90 KB log, và ở lượt chạy đầu **hơn 95%** trong đó là
    # `HTTP Request: HEAD https://huggingface.co/...` của httpx khi
    # sentence-transformers kiểm cache. Mười ba dòng tiến độ thật bị chôn trong
    # đó, tức DoD "1 lệnh chạy hết grid" đạt về chức năng mà không đạt về việc
    # đọc được nó đã chạy gì. Hạ từng logger tường minh chứ không hạ root: cảnh
    # báo của `_resolve_span_labels` ("N câu có span nhưng không khớp chunk nào")
    # là thứ phải thấy được.
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    config = load_experiment_config(args.config, tracking_uri=args.tracking_uri)
    try:
        state = run_experiment(
            config,
            resume=not args.no_resume,
            force=args.force,
            keep_going=args.keep_going,
            dry_run=args.dry_run,
        )
    except PreflightError as exc:
        logger.error("%s", exc)
        return 2
    failed = [name for name, record in state.cells.items() if record.status == "failed"]
    if failed:
        logger.error("%d ô lỗi: %s", len(failed), ", ".join(sorted(failed)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
