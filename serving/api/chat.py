"""`POST /chat` (SSE) và `GET /conversations/{id}` — `W4-06`.

File này cố ý **mỏng**: nó dịch giữa HTTP và `ChatService`, không chứa logic nào
của một lượt hỏi–đáp. Đường phân giới "còn trả được status" / "chỉ còn khung
SSE" nằm ở `serving/core/chat.py`, và ở đây nó hiện ra thành một điều rất cụ
thể: `await service.prepare(...)` chạy **trước** khi `StreamingResponse` được
tạo, nên mọi exception của nó còn thành `HTTPException` được.

⚠️ Đảo hai dòng đó (đưa `prepare()` vào trong generator "cho gọn") là một thay
đổi trông vô hại, và nó biến mọi lỗi 404/403/503 của hạng mục này thành
`200 OK` kèm một khung lỗi mà client mặc định sẽ bỏ qua. Có test ghim
(`test_a_missing_conversation_is_404_not_a_200_with_an_error_frame`).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime, time, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from rag_core.llm import BudgetExceeded
from rag_core.retrieval.filters import MetadataFilter
from serving.api.security import principal_of
from serving.api.sse import SSE_HEADERS, encode
from serving.core.auth import CrossTenantError, Principal
from serving.core.chat import ChatService, ConversationNotFound, GenerationUnavailable
from serving.core.chat import load_history as _load_history

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _seconds_to_utc_midnight() -> int:
    """Bao nhiêu giây nữa thì ngân sách ngày được nạp lại. Xem `DailyBudget`."""
    now = datetime.now(UTC)
    tomorrow = datetime.combine(now.date() + timedelta(days=1), time.min, tzinfo=UTC)
    return max(1, int((tomorrow - now).total_seconds()))


def get_service(request: Request) -> ChatService:
    service: ChatService = request.app.state.chat
    return service


def get_principal(request: Request) -> Principal:
    return principal_of(request)


# Khai ở tầng module, không trong factory — cùng lý do đã ghi ở `health.py`.
ServiceDep = Annotated[ChatService, Depends(get_service)]
PrincipalDep = Annotated[Principal, Depends(get_principal)]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=8000)
    """8000 ký tự ≈ 2–3k token. Không giới hạn thì một `POST` duy nhất tiêu hết
    ngân sách ngày của cả tenant, và hạn mức theo *số request* của `W4-04` không
    thấy điều đó."""

    conversation_id: str | None = Field(default=None, max_length=32)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: MetadataFilter | None = None
    """⚠️ `tenant_id` mà client gửi ở đây **không** được tin: `tenant_filter()`
    ghi đè nó bằng tenant của token, và từ chối nếu hai bên khác nhau."""


@router.post("/chat")
async def chat(
    body: ChatRequest, service: ServiceDep, principal: PrincipalDep
) -> StreamingResponse:
    """Một lượt hỏi–đáp, trả về dạng `text/event-stream`.

    Khung: `meta` → `sources` → `delta`* → (`done` | `error`).

    ⭐ Client **phải** đợi khung kết thúc (`done` hoặc `error`). Một dòng
    `delta` dừng lại không nói được điều gì cả: nó giống hệt nhau khi model nói
    xong, khi kết nối đứt, và khi provider hết hạn mức giữa chừng.
    """
    try:
        turn = await service.prepare(
            principal,
            question=body.message,
            conversation_id=body.conversation_id,
            top_k=body.top_k,
            filters=body.filters,
        )
    except ConversationNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except BudgetExceeded as exc:
        # ⭐ `429`, không `503`: trần chi phí là một **hạn mức**, và nó hết cho
        # tới nửa đêm UTC chứ không phải "thử lại sau vài giây". `Retry-After`
        # nói ra con số ấy để client không phải đoán — cùng khuôn với hạn mức
        # nhịp của `W4-04`.
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            str(exc),
            headers={"Retry-After": str(_seconds_to_utc_midnight())},
        ) from exc
    except CrossTenantError as exc:
        # Lời của lỗi cố ý **không** nhắc tenant mà client đã xin (`W4-04`).
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
    except GenerationUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except Exception as exc:
        # Truy hồi nằm trong `prepare()`, nên Qdrant chết tới được đây — và đây
        # là chỗ **cuối cùng** nó còn biến thành một status đọc được bằng máy.
        logger.exception("chuẩn bị lượt chat thất bại")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"{type(exc).__name__}: {exc}"
        ) from exc

    async def frames() -> AsyncIterator[bytes]:
        async for event in service.stream_turn(turn):
            yield encode(event.event, event.data)

    return StreamingResponse(
        frames(),
        media_type="text/event-stream; charset=utf-8",
        headers={**SSE_HEADERS, "X-Conversation-Id": turn.conversation_id},
    )


@router.get("/admin/llm")
def llm_status(service: ServiceDep) -> dict[str, Any]:
    """Trạng thái bộ định tuyến LLM — `W4-08`.

    Ở file này chứ không ở `admin.py` vì `admin.py` có prefix `/admin/bundle` và
    tài nguyên ở đây là `ChatService`, không phải registry. Scope `admin` vẫn
    được ép **tự động**: `W4-04` kiểm theo tiền tố đường dẫn ở middleware, chứ
    không bằng một dependency gắn tay từng route — nên một route admin mới không
    thể quên hàng rào.

    ⭐ Đây là chỗ duy nhất nhìn thấy cầu dao. Không có nó thì "primary đang bị
    cắt, mọi câu trả lời đến từ nhánh dự phòng" là một trạng thái **không quan
    sát được**: request vẫn 200, câu trả lời vẫn có, chỉ có model khác và hoá
    đơn khác.
    """
    router_obj = getattr(service.llm, "status", None)
    if router_obj is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "chưa cấu hình bộ định tuyến LLM")
    result: dict[str, Any] = router_obj()
    return result


@router.get("/conversations/{conversation_id}")
async def conversation(
    conversation_id: str, service: ServiceDep, principal: PrincipalDep
) -> dict[str, Any]:
    """Đọc lại lịch sử — nửa thứ hai của DoD ("sống sót qua restart container").

    Không có phân trang. Với `MAX_HISTORY_MESSAGES` = 10 ở đường ghi thì một hội
    thoại thật vẫn dài hơn thế nhiều, nên đây là nợ chứ không phải một quyết
    định: một hội thoại 500 lượt trả về một phản hồi vài MB.
    """
    if service.sessions is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "chưa cấu hình Postgres")
    try:
        messages = await _load_history(service.sessions, principal, conversation_id)
    except ConversationNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return {"conversation_id": conversation_id, "messages": messages}
