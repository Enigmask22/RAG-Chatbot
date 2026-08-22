"""Kiểu dữ liệu qua dây và trạng thái job. `W3-08`.

Tách khỏi `app.py` và `tasks.py` vì cả ba tiến trình khác nhau đều đọc chúng:
API ghi, worker cập nhật, client đọc. Một `TypedDict` trong file nào đó của
FastAPI sẽ không dùng lại được ở worker.
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "IngestRequest",
    "JobState",
    "JobStatus",
    "config_dir",
    "resolve_config",
]


def config_dir() -> Path:
    """Thư mục config được phép đọc, lấy từ settings (`INGEST_CONFIG_DIR`).

    Hàm chứ không phải hằng: test và container cần đổi nó, và một hằng đọc lúc
    import thì đã chốt trước khi biến môi trường kịp có tác dụng.
    """
    from rag_core.settings import get_settings

    return get_settings().ingest_config_dir


_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class IngestRequest(BaseModel):
    """Yêu cầu index. Nhận **tên** config, không nhận đường dẫn."""

    model_config = ConfigDict(extra="forbid")

    config: str = Field(min_length=1, max_length=63)
    """Tên file trong `configs/indexing/`, không có đuôi `.yaml`.

    ⚠️ **Không** phải đường dẫn, và đó là một ràng buộc bảo mật chứ không phải
    tiện lợi: nhận đường dẫn thì `../../.env` hay một YAML bất kỳ trên đĩa đều
    trở thành đầu vào hợp lệ, và config index có `corpus_dir` — tức đọc được cả
    thư mục. `_SAFE_NAME` chặn từ tầng schema, `resolve_config` kiểm lại một lần
    nữa rằng đường dẫn cuối cùng vẫn nằm trong `CONFIG_DIR`.
    """

    doc_ids: tuple[str, ...] = ()
    """Rỗng = toàn bộ manifest. Có giá trị = chỉ index bấy nhiêu tài liệu."""

    recreate: bool = False
    """Xoá collection rồi build lại. Cố ý phơi ra: nếu không có thì cách duy nhất
    để sửa một index lệch fingerprint là ssh vào máy chạy tay."""

    @field_validator("config")
    @classmethod
    def _name_not_path(cls, value: str) -> str:
        if not _SAFE_NAME.match(value):
            raise ValueError(
                f"config phải là TÊN (chữ thường, số, `-`, `_`), nhận {value!r}. "
                "Đây là tên file trong configs/indexing/, không phải đường dẫn."
            )
        return value


def resolve_config(name: str, *, root: Path | None = None) -> Path:
    """`"bgem3"` → `configs/indexing/bgem3.yaml`, có kiểm thoát thư mục.

    Kiểm hai lần là cố ý. `_SAFE_NAME` đã chặn `/` và `..`, nhưng nó là validator
    của một model mà worker có thể được gọi trực tiếp (test, CLI, một job cũ nằm
    trong Redis từ trước khi validator tồn tại). `resolve_path` ở đây là hàng rào
    cuối, và nó kiểm **kết quả** chứ không kiểm đầu vào — thứ duy nhất không phụ
    thuộc vào việc mình đã nghĩ ra đủ ký tự xấu hay chưa.
    """
    base = (root or config_dir()).resolve()
    target = (base / f"{name}.yaml").resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"config {name!r} trỏ ra ngoài {base}")
    if not target.is_file():
        available = sorted(p.stem for p in base.glob("*.yaml"))
        raise FileNotFoundError(f"không có config {name!r}. Đang có: {', '.join(available)}")
    return target


class JobStatus(BaseModel):
    """Trạng thái một job, đủ để trả thẳng ra `GET /ingest/{id}`."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    state: JobState = JobState.QUEUED
    config: str = ""
    attempt: int = 0
    """Lần thử thứ mấy. `> 1` nghĩa là đã có một worker chết giữa chừng."""

    documents_total: int = 0
    documents_done: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0

    queued_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    error: str = ""

    @property
    def progress(self) -> float:
        """0..1. `documents_total == 0` trả 0,0 chứ không phải 1,0.

        Khác biệt ấy quan trọng: một job vừa vào hàng đợi và một job đã xong
        không được trông giống nhau ở thanh tiến độ.
        """
        if self.documents_total <= 0:
            return 0.0
        return min(1.0, self.documents_done / self.documents_total)

    @property
    def terminal(self) -> bool:
        return self.state in (JobState.DONE, JobState.FAILED)
