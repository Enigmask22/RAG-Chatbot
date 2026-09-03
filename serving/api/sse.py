"""Đóng khung `text/event-stream`. `W4-06`.

Nhỏ, và có file riêng vì luật khung của SSE có đúng một cái bẫy đắt:

**Một `\\n` trong `data` tách khung làm hai.** Giao thức phân tách sự kiện bằng
một dòng trống, nên nếu payload là text thô thì câu trả lời đầu tiên có xuống
dòng — tức gần như mọi câu trả lời — sẽ tới client dưới dạng hai sự kiện, cái
thứ hai không có tên khung. Client hiển thị đúng một nửa, không có lỗi ở đâu cả.

Cách chặn ở đây **không** phải là tự escape mà là: mọi payload đều đi qua
`json.dumps`, thứ đã escape `\\n` thành `\\\\n` theo đặc tả JSON. Nên luật của
module này là *"không có đường nào gửi text thô"*, và đó là một luật kiểm được
bằng việc đọc chữ ký hàm.

`ensure_ascii=False` để tiếng Việt đi qua nguyên vẹn; UTF-8 đã khai trong
`Content-Type` và mọi trình duyệt đọc SSE bằng UTF-8 theo đặc tả.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

__all__ = ["SSE_HEADERS", "encode"]

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    # `no-transform` không thừa: một proxy được phép nén lại phản hồi, và nén
    # nghĩa là gom đủ một khối trước khi gửi — tức stream về một cục ở cuối, đúng
    # triệu chứng mà `middleware.py` đã tránh ở tầng ứng dụng.
    "Connection": "keep-alive",
    # Không thuộc chuẩn nào; nginx đọc nó. Bỏ ra thì proxy đệm 4–8 KB trước khi
    # nhả byte đầu, và với câu trả lời ngắn hơn thế thì "streaming" không tồn tại.
    "X-Accel-Buffering": "no",
}


def encode(event: str, data: Mapping[str, Any]) -> bytes:
    """Một khung SSE hoàn chỉnh, đã kết thúc bằng dòng trống.

    Không nhận `str` cho `data` — xem docstring module.
    """
    payload = json.dumps(dict(data), ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n".encode()
