"""Experiment runner — `W2-07`.

Một file YAML mô tả ma trận thí nghiệm, một lệnh chạy hết, resume được khi crash.
Tồn tại vì `W2-08` (ablation ≥ 12 tổ hợp): gõ 12 lệnh `make eval-retrieval` bằng
tay thì lần thứ 7 sẽ lệch một tham số và không ai biết là đã lệch — đúng lý lẽ
`IndexConfig` được viết ra ở `W1-08`, chỉ là ở tầng *phép đo* thay vì tầng index.
"""

from .config import (
    ExperimentCell,
    ExperimentConfig,
    MatrixBlock,
    expand,
    load_experiment_config,
)
from .runner import CellRecord, ExperimentState, PreflightError, run_experiment

__all__ = [
    "CellRecord",
    "ExperimentCell",
    "ExperimentConfig",
    "ExperimentState",
    "MatrixBlock",
    "PreflightError",
    "expand",
    "load_experiment_config",
    "run_experiment",
]
