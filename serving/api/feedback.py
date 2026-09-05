"""`POST /feedback` và hàng đợi review — `W5-08`.

Mỏng như `chat.py`: mọi quyết định nằm ở `serving/core/feedback.py`.

## ⭐ Thân request KHÔNG có `trace_id`, và đó là điểm chính của hạng mục

Xem docstring `serving.core.feedback`. Tóm tắt: khoá gắn điểm Langfuse suy ra
từ hàng `message` mà RLS đã xác nhận là của tenant này, chứ không từ một trường
client gửi lên. Có test ghim rằng `FeedbackRequest` **không** khai trường ấy
(`extra="forbid"` biến một client cũ gửi kèm thành `422`, chứ không im lặng bỏ
qua).
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal, get_args

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from serving.api.security import principal_of
from serving.core.auth import Principal
from serving.core.chat import ChatService
from serving.core.feedback import (
    MessageNotFound,
    NotAnAnswer,
    export_candidates,
    record_feedback,
    review_queue,
)
from serving.db.models import FEEDBACK_REASONS

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter(tags=["feedback"])

Reason = Literal["wrong", "incomplete", "not_found", "citation", "language", "slow", "other"]
"""⚠️ Phải trùng `FEEDBACK_REASONS`. `Literal` không nhận một tuple runtime, nên
đây là bản sao thứ ba của danh sách — và có test đếm lại nó, vì hai bản sao lệch
nhau nghĩa là API nhận một giá trị mà `CheckConstraint` từ chối, tức 500 chứ
không 422."""

if set(get_args(Reason)) != set(FEEDBACK_REASONS):  # pragma: no cover - bất biến
    # Kiểm lúc **import** chứ không chỉ trong test: một test bỏ qua được, còn
    # đây thì server không khởi động nổi. Danh sách này là hợp đồng giữa ba nơi
    # (API, `CheckConstraint`, nhãn Prometheus) và lệch nhau là 500 ở production.
    raise RuntimeError(
        f"Reason {sorted(get_args(Reason))} lệch FEEDBACK_REASONS {sorted(FEEDBACK_REASONS)}"
    )


def get_service(request: Request) -> ChatService:
    service: ChatService = request.app.state.chat
    return service


def get_principal(request: Request) -> Principal:
    return principal_of(request)


ServiceDep = Annotated[ChatService, Depends(get_service)]
PrincipalDep = Annotated[Principal, Depends(get_principal)]


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(min_length=1, max_length=32)
    """Id của **câu trả lời**, lấy từ `answer_message_id` trong khung `meta`."""

    rating: Literal[-1, 1]
    """Chỉ 👍/👎. Xem `Feedback.__table_args__` cho lý do không dùng thang 1–5."""

    reason: Reason | None = None
    comment: str | None = Field(default=None, max_length=2000)


def _sessions(service: ChatService) -> Any:
    if service.sessions is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "chưa cấu hình Postgres")
    return service.sessions


@router.post("/feedback", status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    body: FeedbackRequest, request: Request, service: ServiceDep, principal: PrincipalDep
) -> dict[str, Any]:
    """Ghi 👍/👎 cho một câu trả lời.

    Idempotent theo `(tenant, message_id)`: gọi lại ghi đè, không nhân đôi.
    """
    sessions = _sessions(service)
    try:
        result = await record_feedback(
            sessions,
            principal,
            message_id=body.message_id,
            rating=body.rating,
            reason=body.reason,
            comment=body.comment,
            sink=getattr(request.app.state, "trace_sink", None),
        )
    except MessageNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except NotAnAnswer as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    metrics = getattr(request.app.state, "metrics", None)
    if metrics is not None:
        try:
            metrics.feedback.labels(rating=str(result.rating), reason=result.reason or "none").inc()
        except Exception:  # pragma: no cover - phòng thân
            logger.warning("không ghi được metric feedback", exc_info=True)

    return {
        "id": result.id,
        "message_id": result.message_id,
        "rating": result.rating,
        "reason": result.reason,
        # ⭐ Nói ra điểm đã được xếp hàng hay chưa. Im lặng ở đây nghĩa là người
        # tích hợp suy ra "Langfuse đã có điểm", và họ sẽ suy ra thế kể cả khi
        # `LANGFUSE_*` chưa được cấu hình lần nào.
        "scored": result.scored,
        "replaced": not result.created,
        "trace_id": result.trace_id,
    }


@router.get("/admin/feedback")
async def list_feedback(
    service: ServiceDep,
    principal: PrincipalDep,
    rating: Annotated[int | None, Query(ge=-1, le=1)] = -1,
    reason: Reason | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> dict[str, Any]:
    """Hàng đợi review: những lượt bị chấm, mới nhất trước.

    `rating=0` = lấy cả 👍 lẫn 👎 (mẫu số). Mặc định chỉ 👎 — xem `review_queue`.
    """
    sessions = _sessions(service)
    items = await review_queue(
        sessions,
        principal,
        rating=None if rating == 0 else rating,
        reason=reason,
        limit=limit,
    )
    return {"count": len(items), "items": [item.as_dict() for item in items]}


@router.get("/admin/feedback/candidates", response_class=PlainTextResponse)
async def candidates(
    service: ServiceDep,
    principal: PrincipalDep,
    rating: Annotated[int | None, Query(ge=-1, le=1)] = -1,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> PlainTextResponse:
    """Cùng dữ liệu, đóng gói thành JSONL ứng viên golden set.

    ⚠️ **Không** phải golden query — thiếu `category`, `relevant_spans`,
    `reference_answer`. Xem `GoldenCandidate`. Endpoint này là đường lấy file
    khi không có quyền vào DB; đường chính là
    `python -m serving.core.feedback export`.
    """
    sessions = _sessions(service)
    rows = await export_candidates(
        sessions, principal, rating=None if rating == 0 else rating, limit=limit
    )
    body = "".join(row.model_dump_json() + "\n" for row in rows)
    return PlainTextResponse(
        body,
        media_type="application/x-ndjson; charset=utf-8",
        headers={"X-Candidate-Count": str(len(rows))},
    )
