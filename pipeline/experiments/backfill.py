"""Dựng lại view MLflow từ file báo cáo đã có. `W2-07`.

`tracking.py` tuyên bố một điều khá mạnh: *"`NullTracker` là chế độ chạy hợp lệ,
không phải chế độ suy giảm. Nếu MLflow là chỗ duy nhất giữ một con số thì con số
đó không tái lập được từ repo."*

Module này là **phép kiểm** của câu đó. Nếu ba file mỗi ô trong
`plans/reports/runs/` thật sự đủ, thì phải dựng lại được toàn bộ bảng MLflow từ
chúng mà không chạy lại một truy vấn nào. Nếu không dựng lại được thì câu trên
sai và MLflow đã lặng lẽ trở thành nguồn sự thật.

Nó ra đời từ một lần hỏng thật, không từ suy đoán: lượt chạy grid đầu tiên của
`W2-07` chạy trọn 14 ô — đúng số, đủ file — nhưng mlflow 3.15 từ chối
`file:./mlruns` nên tracker rơi về `NullTracker` và **không ghi gì**. Hai đường
sửa: chạy lại 25 phút, hoặc chứng minh là không cần chạy lại. Đường thứ hai vừa
nhanh hơn vừa kiểm được một khẳng định kiến trúc.

⚠️ Điều nó **không** phục hồi được: `duration_s` và `mlflow_run_id` của lần chạy
gốc. Cái đầu nằm ở state file nên vẫn đọc được; cái sau không tồn tại vì run gốc
chưa từng được tạo. Cũng không phục hồi được thứ tự thời gian thật của grid —
`created_at` trong báo cáo có, nhưng MLflow đặt `start_time` là lúc backfill.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import expand, load_experiment_config
from .runner import ExperimentState, cell_params, git_sha, report_metrics, report_params
from .tracking import TrackingUnavailable, open_tracker

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

__all__ = ["backfill", "main"]


def backfill(config_path: Path, *, tracking_uri: str | None = None, dry_run: bool = False) -> int:
    """Log lại mọi ô **đã có báo cáo** lên MLflow. Trả số ô đã log."""
    config = load_experiment_config(config_path, tracking_uri=tracking_uri)
    cells = expand(config)
    state = ExperimentState.load(config.state_path, config.name)

    found: list[tuple[str, Path]] = []
    missing: list[str] = []
    for cell in cells:
        report = config.out_dir / f"{cell.run_name}-retrieval.json"
        if report.exists():
            found.append((cell.run_name, report))
        else:
            missing.append(cell.run_name)

    if missing:
        # Danh sách này là thông tin, không phải lỗi: backfill một grid đang chạy
        # dở là chuyện hợp lý.
        logger.warning("%d ô chưa có báo cáo, bỏ qua: %s", len(missing), ", ".join(missing))
    logger.info("Có báo cáo cho %d/%d ô.", len(found), len(cells))
    if dry_run:
        return 0

    tracker = open_tracker(config.tracking_uri, config.mlflow_experiment)
    tags = {
        "task": "W2-07",
        "experiment": config.name,
        "git_sha": git_sha(),
        "backfilled": "true",
        **config.tags,
    }
    by_name = {cell.run_name: cell for cell in cells}
    logged = 0
    for run_name, report in found:
        payload = json.loads(report.read_text(encoding="utf-8"))
        record = state.cells.get(run_name)
        # `fingerprint` lấy từ state khi có. Không có thì để rỗng thay vì tính lại:
        # tính lại đòi `index_fingerprint` **lúc chạy gốc**, mà index có thể đã
        # được build lại từ lúc đó — một con số tự tin mà sai.
        fingerprint = record.fingerprint if record is not None else ""
        params: dict[str, Any] = report_params(payload) | cell_params(
            by_name[run_name], fingerprint
        )
        if record is not None:
            params["duration_s"] = record.duration_s
        with tracker.start_run(run_name, {**tags, "run_name": run_name}) as run:
            run.log_params(params)
            run.log_metrics(report_metrics(payload))
            for artifact in (
                report,
                report.with_name(f"{run_name}-retrieval.md"),
                config.out_dir / f"{run_name}-per-query.jsonl",
            ):
                if artifact.exists():
                    run.log_artifact(artifact)
        logged += 1
        logger.info("Đã log %s", run_name)
    return logged


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dựng lại view MLflow từ báo cáo đã có, không chạy lại eval (W2-07)"
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML ở `configs/eval/`")
    parser.add_argument("--tracking-uri", help="Ghi đè `tracking_uri` của file config.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Chỉ đếm ô có báo cáo, không ghi MLflow."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr
    )
    for noisy in ("httpx", "httpcore", "urllib3", "alembic", "mlflow"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        logged = backfill(args.config, tracking_uri=args.tracking_uri, dry_run=args.dry_run)
    except TrackingUnavailable as exc:
        logger.error("%s", exc)
        return 2
    logger.info("Đã log %d ô lên MLflow.", logged)
    return 0


if __name__ == "__main__":
    sys.exit(main())
