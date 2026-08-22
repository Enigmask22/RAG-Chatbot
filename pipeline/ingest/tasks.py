"""Công việc chạy nền: một job = một lượt `build_index`. `W3-08`.

## `build_index` là hàm đồng bộ, và nó phải ở lại như thế

Nó chiếm CPU/GPU hàng trăm giây. Gọi thẳng trong coroutine thì vòng lặp sự kiện
của worker đứng hình: `GET /ingest/{id}` vẫn trả lời (đó là tiến trình khác)
nhưng worker không nhận được tín hiệu dừng, không cập nhật được tiến độ, và
health check của chính nó chết. Nên nó chạy trong `asyncio.to_thread`.

Hệ quả kéo theo là callback tiến độ chạy **trong thread khác**, nên nó không
được chạm client Redis async trực tiếp — xem `_progress_bridge`.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Future
from datetime import UTC, datetime
from typing import Any

from .schemas import IngestRequest, JobState, resolve_config
from .store import JobStore

__all__ = ["INGEST_TASK", "MAX_TRIES", "ingest_job", "is_transient"]

MAX_TRIES = 3
"""Phải khớp `WorkerSettings.max_tries` — arq dừng ở đó bất kể `Retry` nói gì."""

logger = logging.getLogger(__name__)

INGEST_TASK = "ingest_job"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


_TRANSIENT_NAMES = frozenset(
    {"ConnectionError", "ResponseHandlingException", "UnexpectedResponse", "ServiceException"}
)


def is_transient(exc: BaseException) -> bool:
    """Lỗi có đáng thử lại không.

    ## arq KHÔNG tự thử lại exception thường

    Điều này tôi đã tưởng ngược, và test là chỗ lộ ra. `arq/worker.py:613-633`
    chỉ thử lại với `Retry`, `RetryJob` và `asyncio.CancelledError` (tức worker
    tắt hoặc job quá hạn). Một `Exception` bất kỳ ⇒ `finish = True`, hỏng hẳn,
    `max_tries` không đụng tới.

    Hoá ra đó là mặc định **đúng**, và việc phải tự phân loại là một cải thiện
    chứ không phải một chỗ vá: thử lại một job hỏng vì thiếu config là ba lần
    hỏng y hệt nhau, ba lần log, và một hàng đợi che mất lỗi thật. Chỉ lỗi **hạ
    tầng** mới đáng thử lại.

    Nhận diện theo **tên lớp** thay vì `isinstance`: bắt đúng kiểu của
    `qdrant_client` sẽ buộc module này import nó, mà cả điểm của `tasks.py` là
    không kéo phụ thuộc nặng vào tiến trình nào chưa cần.
    """
    seen: BaseException | None = exc
    while seen is not None:
        if isinstance(seen, ConnectionError | TimeoutError | OSError):
            return True
        if type(seen).__name__ in _TRANSIENT_NAMES:
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def _progress_bridge(store: JobStore, job_id: str, loop: asyncio.AbstractEventLoop) -> Any:
    """Callback gọi được từ **thread của `build_index`**, ghi vào Redis async.

    `run_coroutine_threadsafe` là cách duy nhất đúng ở đây: client Redis async
    thuộc về vòng lặp, và gọi nó từ thread khác sẽ hỏng theo kiểu ngẫu nhiên (dữ
    liệu lẫn giữa hai lệnh trên cùng một socket) chứ không phải hỏng ngay.

    Không `await` kết quả: tiến độ là thông tin phụ, và chặn vòng chunk lại để
    chờ Redis là để cái phụ điều khiển cái chính. Nhưng cũng **không** vứt
    Future đi — lỗi im lặng ở đây nghĩa là thanh tiến độ đứng yên mà không ai
    biết vì sao.
    """

    def on_done(future: Future[Any]) -> None:
        error = future.exception()
        if error is not None:
            logger.warning("Không ghi được tiến độ job %s: %s", job_id, error)

    def report(done: int, total: int) -> None:
        future = asyncio.run_coroutine_threadsafe(
            store.patch(job_id, documents_done=done, documents_total=total), loop
        )
        future.add_done_callback(on_done)

    return report


async def ingest_job(ctx: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Một lượt index. Chạy lại được nguyên vẹn — xem `pipeline/ingest/__init__.py`."""
    from ..indexing.build_index import build_index
    from ..indexing.config import load_index_config

    job_id = str(ctx["job_id"])
    attempt = int(ctx.get("job_try", 1))
    store = JobStore(ctx["redis"])
    loop = asyncio.get_running_loop()

    request = IngestRequest.model_validate(payload)
    await store.patch(
        job_id,
        state=JobState.RUNNING,
        attempt=attempt,
        started_at=_now(),
        error="",
    )
    if attempt > 1:
        # Không phải cảnh báo thừa: đây là dấu vết DUY NHẤT còn lại của một
        # worker đã chết giữa chừng. Job chạy lại sẽ thành công và trạng thái
        # cuối cùng trông y hệt một lượt chạy suôn sẻ.
        logger.warning("Job %s chạy lại lần %d — lượt trước không kết thúc", job_id, attempt)

    try:
        config = load_index_config(resolve_config(request.config))
        report = await asyncio.to_thread(
            build_index,
            config,
            qdrant_url=ctx["qdrant_url"],
            qdrant_api_key=ctx.get("qdrant_api_key"),
            recreate=request.recreate,
            on_progress=_progress_bridge(store, job_id, loop),
            only_doc_ids=list(request.doc_ids) or None,
        )
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        if is_transient(exc) and attempt < MAX_TRIES:
            from arq.worker import Retry

            logger.warning("Job %s gặp lỗi hạ tầng, thử lại (lần %d): %s", job_id, attempt, detail)
            # Quay về QUEUED chứ không FAILED: job này SẼ chạy lại, và để nó nằm
            # ở FAILED nghĩa là client thấy "hỏng" rồi vài giây sau thấy "xong" —
            # một chuỗi trạng thái không giải thích được.
            await store.patch(job_id, state=JobState.QUEUED, error=detail)
            raise Retry(defer=2**attempt) from exc
        logger.exception("Job %s hỏng", job_id)
        await store.fail(job_id, detail, finished_at=_now())
        raise

    await store.patch(
        job_id,
        state=JobState.DONE,
        documents_total=report.n_documents,
        documents_done=report.n_documents,
        chunks_embedded=report.n_chunks_embedded,
        chunks_reused=report.n_chunks_reused,
        finished_at=_now(),
    )
    return {
        "collection": report.collection,
        "chunks_written": report.n_chunks_written,
        "chunks_embedded": report.n_chunks_embedded,
        "chunks_reused": report.n_chunks_reused,
    }
