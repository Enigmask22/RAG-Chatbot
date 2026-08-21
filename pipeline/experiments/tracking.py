"""Ghi một lần chạy lên MLflow — hoặc không, và grid vẫn chạy.

MLflow là thứ **duy nhất** của `W2-07` mà thiếu nó thì kết quả không mất gì: ba
file báo cáo trong `plans/reports/runs/` vẫn là nguồn sự thật, `compare.py` vẫn
đọc chúng, `G2` vẫn kiểm định được. MLflow thêm vào một chỗ *duyệt* 12 ô cạnh
nhau, không thêm dữ liệu.

Nên nó là dependency **tuỳ chọn** (`pip install -e ".[tracking]"`), và thiếu nó
là một dòng log chứ không phải một exception. Lý lẽ: một grid 40 phút không được
chết ở giây thứ nhất vì thiếu một thư viện chỉ dùng để xem lại.

Đổi lại thì phải nói rõ: **`NullTracker` là chế độ chạy hợp lệ, không phải chế độ
suy giảm.** Nếu MLflow là chỗ duy nhất giữ một con số thì con số đó không tái lập
được từ repo, và đó là thứ `W2-09` không được dựa vào.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from types import TracebackType

logger = logging.getLogger(__name__)

__all__ = [
    "METRIC_NAME_MAP",
    "MlflowTracker",
    "NullTracker",
    "RunHandle",
    "SafeTracker",
    "Tracker",
    "TrackingUnavailable",
    "mlflow_metric_name",
    "open_tracker",
]


class TrackingUnavailable(Exception):
    """`tracking_uri` có nhưng không mở được — **lỗi config**, thuộc preflight."""


#: Ký tự MLflow **không** nhận trong tên metric, và cái nó được đổi thành.
#: MLflow chỉ cho `[A-Za-z0-9_-./ ]`, nên `@` bị từ chối — mà **mọi** metric của dự
#: án này mang `@`: `hit_rate@1`, `ndcg@10`, `recall@5`, `map@20`.
#:
#: Đã hỏng thật, và hỏng theo cách đáng ghi: `backfill` log 14 ô, param vào hết,
#: **metric không một cái nào**, và `SafeTracker` biến chuyện đó thành 14 dòng
#: cảnh báo rồi báo "Đã log 14 ô". Bảng MLflow có 14 run và **0 cột metric**.
#:
#: Bài học không phải "MLflow khó tính" mà là: **khoan dung đúng với lỗi nhất thời
#: và sai với lỗi hệ thống, và ở chỗ gọi thì hai loại giống nhau.** Nên cách sửa
#: bền là xoá cả lớp lỗi ở nguồn (đổi tên ở đây) cộng một test ghim rằng mọi tên
#: metric mà eval sinh ra đều hợp lệ sau khi đổi — chứ không phải nới `SafeTracker`.
#:
#: `_at_` chứ không phải `.`: `ndcg_at_10` đọc ra tiếng người, còn `ndcg.10` bị
#: MLflow hiểu là phân cấp và nhóm chung với `ndcg.20` trong UI.
METRIC_NAME_MAP = {"@": "_at_"}


def mlflow_metric_name(name: str) -> str:
    """Tên metric hợp lệ với MLflow. **Chỉ đổi ở tầng hiển thị.**

    File báo cáo trong `plans/reports/runs/` giữ nguyên `@` — chúng là nguồn sự
    thật và `compare.py` đọc chúng. Đổi tên ở đây là thích ứng với một ràng buộc
    của MLflow, không phải đổi từ vựng của dự án.
    """
    for bad, good in METRIC_NAME_MAP.items():
        name = name.replace(bad, good)
    return "".join(ch if (ch.isalnum() or ch in "_-./ ") else "_" for ch in name)


class RunHandle(Protocol):
    """Một run đang mở. `run_id` là `None` khi không theo dõi gì."""

    @property
    def run_id(self) -> str | None: ...
    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float]) -> None: ...
    def log_artifact(self, path: Path) -> None: ...
    def set_failed(self, error: str) -> None: ...
    def __enter__(self) -> RunHandle: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


class Tracker(Protocol):
    def start_run(self, run_name: str, tags: dict[str, str]) -> RunHandle: ...


class _NullRun:
    run_id: None = None

    def log_params(self, params: dict[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: dict[str, float]) -> None:
        return None

    def log_artifact(self, path: Path) -> None:
        return None

    def set_failed(self, error: str) -> None:
        return None

    def __enter__(self) -> _NullRun:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


class NullTracker:
    """Không theo dõi gì. Dùng khi `tracking_uri` rỗng hoặc thiếu mlflow."""

    def start_run(self, run_name: str, tags: dict[str, str]) -> RunHandle:
        return _NullRun()


class _MlflowRun:
    def __init__(self, active: Any) -> None:
        import mlflow

        self._mlflow = mlflow
        self._active = active

    @property
    def run_id(self) -> str | None:
        run_id: str = self._active.info.run_id
        return run_id

    def log_params(self, params: dict[str, Any]) -> None:
        # MLflow ép param thành chuỗi và **cắt** ở 6000 ký tự (im lặng ở bản cũ).
        # Không có param nào ở đây dài thế, nhưng `reranker_model` + đường dẫn
        # config là thứ sẽ dài ra, nên cắt tường minh kèm dấu để thấy được.
        self._mlflow.log_params(
            {key: _short(value) for key, value in params.items() if value is not None}
        )

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._mlflow.log_metrics({mlflow_metric_name(k): v for k, v in metrics.items()})

    def log_artifact(self, path: Path) -> None:
        self._mlflow.log_artifact(str(path))

    def set_failed(self, error: str) -> None:
        self._mlflow.set_tag("failure", _short(error))

    def __enter__(self) -> _MlflowRun:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # `status` phải phản ánh sự thật: một ô chết mà run vẫn FINISHED thì
        # MLflow UI hiển thị 12 run xanh cho một grid có 11 ô chạy được.
        status = "FINISHED" if exc_type is None else "FAILED"
        self._mlflow.end_run(status=status)


class MlflowTracker:
    def __init__(self, uri: str, experiment: str) -> None:
        import mlflow

        self._mlflow = mlflow
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        self.uri = uri
        self.experiment = experiment

    def start_run(self, run_name: str, tags: dict[str, str]) -> RunHandle:
        return _MlflowRun(self._mlflow.start_run(run_name=run_name, tags=tags))


class SafeTracker:
    """Bọc một tracker sao cho lỗi theo dõi **không bao giờ** làm mất kết quả eval.

    Nếu MLflow là một *view* chứ không phải nguồn sự thật (xem docstring module)
    thì một ô đã chạy 131 giây và đã ghi đủ ba file báo cáo không được đánh dấu
    `failed` chỉ vì tracking server chết ở giữa. Trước lớp này thì
    `tracker.start_run` nổ giữa grid sẽ làm đúng thế.
    """

    def __init__(self, inner: Tracker) -> None:
        self._inner = inner

    def start_run(self, run_name: str, tags: dict[str, str]) -> RunHandle:
        try:
            return _SafeRun(self._inner.start_run(run_name, tags))
        except Exception as exc:
            logger.warning(
                "Không mở được run MLflow %r (%s) — ô này chạy không log.", run_name, exc
            )
            return _NullRun()


class _SafeRun:
    def __init__(self, inner: RunHandle) -> None:
        self._inner = inner

    @property
    def run_id(self) -> str | None:
        value = self._guard(lambda: self._inner.run_id)
        return value if isinstance(value, str) else None

    def log_params(self, params: dict[str, Any]) -> None:
        self._guard(lambda: self._inner.log_params(params))

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._guard(lambda: self._inner.log_metrics(metrics))

    def log_artifact(self, path: Path) -> None:
        self._guard(lambda: self._inner.log_artifact(path))

    def set_failed(self, error: str) -> None:
        self._guard(lambda: self._inner.set_failed(error))

    def __enter__(self) -> _SafeRun:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._guard(lambda: self._inner.__exit__(exc_type, exc, tb))

    def _guard(self, call: Callable[[], Any]) -> Any:
        try:
            return call()
        except Exception as exc:
            logger.warning("Lỗi khi ghi MLflow (%s) — bỏ qua, kết quả vẫn nằm ở file báo cáo.", exc)
            return None


def open_tracker(uri: str | None, experiment: str) -> Tracker:
    """Mở tracker. **Nổ `TrackingUnavailable` khi có URI mà không dùng được.**

    Ba đường ra, và chúng khác nhau có chủ đích — lần chạy grid đầu tiên của
    `W2-07` là lý do:

    1. **`uri` rỗng** → `NullTracker`, im lặng. Đó là một lựa chọn tường minh.
    2. **Thiếu mlflow** → `NullTracker` + cảnh báo. Extra `tracking` là **tuỳ chọn
       có chủ đích**, và một grid 40 phút không được chết vì thiếu thư viện dùng để
       xem lại.
    3. **Có `uri` mà không mở được** → **nổ**, để preflight bắt.

    Đường 3 trước đây cũng rơi về `NullTracker` + cảnh báo, và nó đã hỏng đúng
    theo cách đáng ghi: mlflow 3.15 **từ chối** `file:./mlruns` (file store vào
    maintenance mode, đòi `sqlite:///…`). Grid chạy trọn 14 ô, đúng số, đủ file —
    và **không ghi gì lên MLflow**, tức Evidence của DoD không tồn tại. Cảnh báo
    thì có: **dòng 19 trong 2320**, nổ ở giây 0 rồi bị chôn dưới 2270 dòng HTTP.

    Nên: khai một đích đến mà không tới được là **lỗi config**, và lỗi config
    thuộc preflight — biết ở giây thứ nhất, không ở phút thứ 25. Muốn chạy không
    theo dõi thì đặt `tracking_uri: null`, tường minh. Lỗi **giữa grid** (server
    chết ở ô thứ 7) thì không preflight được và `SafeTracker` lo.
    """
    if not uri:
        logger.info("Không có `tracking_uri` — chạy grid mà không theo dõi MLflow.")
        return NullTracker()
    try:
        tracker = MlflowTracker(uri, experiment)
    except ImportError:
        logger.warning(
            "Có `tracking_uri=%s` nhưng chưa cài mlflow — grid vẫn chạy và vẫn ghi "
            "đủ ba file báo cáo mỗi ô. Muốn có UI thì `uv sync --extra tracking`.",
            uri,
        )
        return NullTracker()
    except Exception as exc:
        raise TrackingUnavailable(
            f"Không mở được MLflow tại {uri!r}: {exc}\n"
            "Đặt `tracking_uri: null` trong file config nếu đúng ý là chạy không "
            "theo dõi. Với mlflow >= 3 thì `file:` không còn được nhận — dùng "
            "`sqlite:///mlflow.db`."
        ) from exc
    logger.info("MLflow: %s · experiment %r", tracker.uri, tracker.experiment)
    return SafeTracker(tracker)


def _short(value: Any, limit: int = 480) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"
