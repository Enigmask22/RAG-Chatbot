"""Trạng thái job trên Redis. `W3-08`.

Trạng thái sống ở Redis chứ không trong bộ nhớ tiến trình API, vì API và worker
là **hai tiến trình khác nhau** — và vì `GET /ingest/{id}` phải trả lời được sau
khi API restart. Một dict trong RAM sẽ hoạt động hoàn hảo trong test một tiến
trình rồi hỏng ngay lần deploy đầu.

TTL 24 giờ: đủ để người dùng quay lại xem một job đêm qua, không đủ để Redis
biến thành cơ sở dữ liệu lâu dài mà không ai định làm.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .schemas import JobState, JobStatus

if TYPE_CHECKING:
    from redis.asyncio import Redis

__all__ = ["JOB_TTL_SECONDS", "JobStore", "job_key"]

logger = logging.getLogger(__name__)

JOB_TTL_SECONDS = 24 * 60 * 60

_PREFIX = "ingest:job:"


def job_key(job_id: str) -> str:
    return f"{_PREFIX}{job_id}"


class JobStore:
    """Đọc/ghi `JobStatus`. Mỏng có chủ ý — nó là một cái hộp, không phải một tầng."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def put(self, status: JobStatus) -> None:
        await self._redis.set(job_key(status.job_id), status.model_dump_json(), ex=JOB_TTL_SECONDS)

    async def get(self, job_id: str) -> JobStatus | None:
        raw: Any = await self._redis.get(job_key(job_id))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            return JobStatus.model_validate_json(raw)
        except Exception:
            # Schema đã đổi giữa hai lần deploy. Trạng thái job là dữ liệu dùng
            # một lần rồi bỏ, nên coi như không có còn hơn làm sập endpoint —
            # nhưng phải nói ra, vì im lặng thì mọi job cũ "biến mất" không lý do.
            logger.warning("Trạng thái job %s không đọc được, coi như không có", job_id)
            return None

    async def patch(self, job_id: str, **fields: object) -> JobStatus | None:
        """Đọc–sửa–ghi. Không có khoá, và đó là một đánh đổi có ý thức.

        Mỗi job chỉ có **một** worker ghi vào nó tại một thời điểm (arq đảm bảo
        điều đó bằng khoá riêng của nó), còn API chỉ đọc. Nên đường đua duy nhất
        còn lại là hai lần thử của cùng một job chồng lên nhau sau khi worker
        chết — và ở đó bản ghi sau đúng hơn bản ghi trước, nên ghi đè là hành vi
        mong muốn chứ không phải mất mát.
        """
        current = await self.get(job_id)
        if current is None:
            return None
        updated = current.model_copy(update=dict(fields))
        await self.put(updated)
        return updated

    async def fail(self, job_id: str, error: str, *, finished_at: str) -> None:
        await self.patch(
            job_id,
            state=JobState.FAILED,
            error=error[:2000],
            finished_at=finished_at,
        )
