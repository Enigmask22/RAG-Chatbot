"""`request_id` + dòng access, dạng ASGI thuần — `W4-03`.

## ⚠️ Vì sao **không** phải `BaseHTTPMiddleware`

Cách mặc định mà mọi ví dụ FastAPI viết middleware là kế thừa
`starlette.middleware.base.BaseHTTPMiddleware`. Nó gói phản hồi vào một cặp
memory stream của anyio, và hệ quả là nó **giữ lại** phản hồi dạng dòng: với một
`StreamingResponse`, client nhận token khi middleware đã nhả chứ không phải khi
handler `yield`. `W4-06` là `POST /chat` **SSE**.

Nếu viết `BaseHTTPMiddleware` ở đây thì bug ấy xuất hiện ở `W4-06` — cách xa
nguyên nhân, và biểu hiện của nó ("stream về một cục ở cuối") trông đúng như một
lỗi của tầng sinh. Middleware ASGI thuần chỉ bọc `send`, không đụng vào luồng dữ
liệu, nên nó vô hình với SSE.

## Vì sao tự trả 500 thay vì để `ServerErrorMiddleware` lo

Thứ tự middleware của Starlette là `ServerErrorMiddleware` → (middleware của
ứng dụng) → `ExceptionMiddleware` → router. Một exception không bắt được sẽ đi
**xuyên qua** middleware này lên trên, và phản hồi 500 do lớp ngoài cùng gửi —
tức nó **không** đi qua `send` đã bọc ở đây, nên đúng cái phản hồi mà người vận
hành cần truy vết lại là phản hồi duy nhất thiếu `X-Request-ID`.

Nên chỗ này bắt lấy, ghi log kèm traceback, và trả một JSON có `request_id`
trong **thân** phản hồi — để người dùng cuối dán được mã ấy vào báo lỗi.

⚠️ Ngoại lệ: nếu phản hồi **đã bắt đầu** (đang giữa một stream SSE) thì không
còn gửi được status khác nữa. Lúc đó ném tiếp là cách duy nhất trung thực.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

from serving.core.logging import bind_request_id, reset_request_id

__all__ = ["REQUEST_ID_HEADER", "RequestContextMiddleware"]

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "x-request-id"

#: Mã đến từ ngoài chỉ được dùng lại nếu nó *nhìn giống* một mã truy vết.
#:
#: JSON encoding đã vô hiệu hoá phần nguy hiểm nhất — một `\\n` trong header
#: không tách được dòng log ra làm hai vì `json.dumps` escape nó. Nên phép kiểm
#: này **không** phải để chống log injection; nó chặn hai thứ còn lại: một header
#: dài vô hạn nhân lên trong mọi dòng log của request đó, và việc dội ngược một
#: chuỗi tuỳ ý vào header phản hồi.
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


def _incoming_request_id(scope: dict[str, Any]) -> str | None:
    """Lấy `X-Request-ID` của upstream, nếu nó dùng được.

    Dùng lại mã của upstream thay vì luôn sinh mới là điều kiện để một dấu vết
    xuyên qua nhiều dịch vụ: load balancer → API này → worker ingest.
    """
    for raw_name, raw_value in scope.get("headers", []):
        if raw_name != REQUEST_ID_HEADER.encode():
            continue
        candidate = raw_value.decode("latin-1")
        return candidate if _SAFE_REQUEST_ID.match(candidate) else None
    return None


class RequestContextMiddleware:
    """Gắn `request_id` cho mọi request, dội nó về header, và ghi một dòng access."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _incoming_request_id(scope) or uuid.uuid4().hex
        header = (REQUEST_ID_HEADER.encode(), request_id.encode("ascii"))
        token = bind_request_id(request_id)
        started = False
        status = 500
        start = time.perf_counter()

        async def send_wrapper(message: Any) -> None:
            nonlocal started, status
            if message["type"] == "http.response.start":
                started = True
                status = message["status"]
                message["headers"] = [*message.get("headers", []), header]
            await send(message)

        try:
            try:
                await self.app(scope, receive, send_wrapper)
            except Exception:
                logger.exception("request lỗi không bắt được")
                if started:
                    # Stream đã chạy: không đổi được status nữa, và giả vờ thành
                    # công thì tệ hơn là để kết nối đứt.
                    raise
                await _send_error(send, header, request_id)
                status = 500
        finally:
            # ⚠️ Đo ở đây nên với phản hồi dạng dòng, `duration_ms` là **thời
            # gian tới token cuối**, không phải thời gian tới byte đầu. Hai con
            # số ấy khác nhau rất xa ở SSE; `W4-06` phải log thêm TTFB riêng.
            logger.info(
                "%s %s → %d",
                scope.get("method", "?"),
                scope.get("path", "?"),
                status,
                extra={
                    "http_method": scope.get("method"),
                    # Cố ý **không** có query string: nó là dữ liệu người dùng
                    # nhập, và dòng access là thứ được giữ lâu nhất, chuyển đi xa
                    # nhất trong mọi loại log.
                    "http_path": scope.get("path"),
                    "http_status": status,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                },
            )
            reset_request_id(token)


async def _send_error(send: Any, header: tuple[bytes, bytes], request_id: str) -> None:
    body = json.dumps(
        {"detail": "Lỗi nội bộ", "request_id": request_id}, ensure_ascii=False
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 500,
            "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"content-length", str(len(body)).encode("ascii")),
                header,
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
