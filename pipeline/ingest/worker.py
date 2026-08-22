"""Worker arq. `W3-08`.

    uv run arq pipeline.ingest.worker.WorkerSettings

## Ba con số, và vì sao chúng là những con số ấy

* `job_timeout = 2 giờ`. Build sạch corpus hiện tại mất **397 s** (`W3-07` §3),
  nhưng timeout không phải để đo cái đã biết — nó là trần cho corpus lớn gấp
  mười. Đặt sát quá thì một job đang chạy đúng bị giết ở phút cuối, và arq sẽ
  **chạy lại** nó: không mất dữ liệu (idempotent) nhưng tốn gấp đôi và trông y
  hệt một lỗi hạ tầng.
* `max_tries = MAX_TRIES` (3), lấy thẳng từ `tasks.py` chứ không viết lại con số:
  `tasks.is_transient` so `attempt < MAX_TRIES` để quyết định có ném `Retry` hay
  không, nên hai chỗ lệch nhau sẽ cho một job ném `Retry` mà arq đã thôi thử —
  job biến mất khỏi hàng đợi ở trạng thái `queued`, vĩnh viễn.
  ⚠️ Con số này **không** có nghĩa "mọi lỗi được thử 3 lần":
  arq chỉ thử lại với `Retry`, `RetryJob` và `CancelledError`
  (`arq/worker.py:613-633`) — một `Exception` thường là hỏng hẳn ngay lần đầu.
  Tôi đã tưởng ngược, và test là chỗ lộ ra. Việc phân loại lỗi nào đáng thử lại
  vì thế nằm ở `tasks.is_transient`, và mặc định của arq hoá ra **đúng**: thử
  lại một job hỏng vì config sai chỉ cho ba lần hỏng y hệt nhau.
* `max_jobs = 1`. Hai job cùng lúc trên một GPU nghĩa là hai bản BGE-M3 (2,2 GB
  mỗi bản) tranh VRAM — cùng ngân sách mà `W0-06` đang đếm. Song song ở đây mua
  được rất ít vì công việc vốn đã bão hoà GPU.

## Ba đường quay lại hàng đợi, và chúng khác nhau

| chuyện gì xảy ra | arq làm gì | mất bao lâu để nhận ra |
|---|---|---|
| lỗi hạ tầng (Qdrant sập) | `tasks.py` ném `Retry` | ngay |
| worker tắt êm (Ctrl-C) | `CancelledError` ⇒ tự thử lại | ngay |
| worker **chết hẳn** | khoá `in-progress` hết hạn | **`job_timeout + 10s`** |
| lỗi tất định (sai config) | hỏng hẳn, không thử lại | ngay |

## Retry khi worker chết — cơ chế thật, không phải lời hứa

arq đặt một khoá "đang chạy" cho mỗi job với hạn bằng `job_timeout`. Worker chết
thì khoá hết hạn và job quay lại hàng đợi với `job_try` tăng lên. Nghĩa là phục
hồi **không tức thì**: nó mất đúng `job_timeout` để nhận ra. Đó là cái giá của
việc không có heartbeat riêng, và nó phải được nói ra chứ không để người vận
hành tự phát hiện lúc 2 giờ sáng.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from arq.connections import RedisSettings

from .tasks import INGEST_TASK, MAX_TRIES, ingest_job

__all__ = ["WorkerSettings", "queue_name", "redis_settings"]

logger = logging.getLogger(__name__)


def redis_settings() -> RedisSettings:
    from rag_core.settings import get_settings

    return RedisSettings.from_dsn(get_settings().redis_url)


def queue_name() -> str:
    from rag_core.settings import get_settings

    return get_settings().ingest_queue


async def startup(ctx: dict[str, Any]) -> None:
    """Đưa cấu hình hạ tầng vào `ctx` **một lần**, không đọc lại mỗi job.

    `get_settings()` đọc `.env`; gọi nó trong thân job nghĩa là mỗi lượt chạy lại
    có thể thấy một cấu hình khác, và một job retry sẽ ghi vào Qdrant khác với
    lượt đầu mà không ai biết.
    """
    from rag_core.settings import get_settings

    settings = get_settings()
    ctx["qdrant_url"] = settings.qdrant_url
    ctx["qdrant_api_key"] = (
        settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    )
    logger.info("Worker ingest sẵn sàng · qdrant=%s", settings.qdrant_url)


class WorkerSettings:
    functions: ClassVar[list[Any]] = [ingest_job]
    on_startup = startup
    max_tries = MAX_TRIES
    job_timeout = 2 * 60 * 60
    keep_result = 24 * 60 * 60
    max_jobs = 1

    @staticmethod
    def redis_settings() -> RedisSettings:
        return redis_settings()

    @staticmethod
    def queue_name() -> str:  # arq đọc thuộc tính này lúc khởi động worker
        return queue_name()


# arq tra hàm theo `__name__`; ghim lại để đổi tên hàm không âm thầm làm mọi job
# đang nằm trong hàng đợi thành "không tìm thấy".
assert ingest_job.__name__ == INGEST_TASK
