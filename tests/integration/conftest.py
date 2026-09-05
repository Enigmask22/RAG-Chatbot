"""Fixture chung cho integration: `database`/`workspace` của `test_chat_stream`.

Nạp ở conftest thay vì import trong từng file test: một fixture import rồi lại
xuất hiện làm tham số test là F811 (redefinition) với ruff — còn qua conftest
thì pytest tự phân giải theo tên, không cần import nào ở file dùng.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest

from serving.__main__ import needs_selector_loop
from tests.integration.test_chat_stream import database, workspace

__all__ = ["database", "workspace"]

_POLICY = "WindowsSelectorEventLoopPolicy"
"""Tên lớp policy, giữ trong một **biến**.

⚠️ `getattr(asyncio, "WindowsSelectorEventLoopPolicy")` với chuỗi hằng vẫn bị
mypy phân giải như một truy cập thuộc tính thường, nên nó báo `attr-defined`
trên Linux — nơi lớp ấy không tồn tại. Một biến thì không.
"""


@pytest.fixture(scope="session", autouse=True)
def _selector_loop() -> Iterator[None]:
    """Vòng lặp sự kiện mà `psycopg` async chạy được — `W4-06`, gom lại ở `W5-09`.

    Trên Windows mặc định của asyncio là `ProactorEventLoop`, thứ driver async
    của `psycopg` v3 **từ chối**. `TestClient` dựng vòng lặp qua
    `anyio.start_blocking_portal` → `asyncio.run`, và `asyncio.run` **có** đi
    qua policy — khác uvicorn, thứ trả thẳng một `loop_factory` (xem docstring
    `serving/__main__.py`).

    ## ⚠️ Hai chi tiết làm bản trước đỏ trên Linux, và CI tìm ra ở lượt chạy đầu

    Ba module tự chép fixture này, mỗi bản mở đầu bằng
    `if not sys.platform.startswith("win"): return`. mypy **thu hẹp**
    `sys.platform` theo `--platform`, nên trên Linux mọi dòng sau đó là
    `unreachable` — và `warn_unreachable` biến nó thành lỗi. `make lint` xanh
    trên Windows, đỏ trên Linux: phán quyết của cổng phụ thuộc **hệ điều hành**,
    không chỉ phụ thuộc mã.

    Chữa bằng hai thứ, cả hai đều cần:

    * hỏi qua `needs_selector_loop()` — một hàm trả `bool`, mypy không thu hẹp
      được, nên nhánh dưới vẫn *reachable* ở mọi nền tảng;
    * lấy lớp policy bằng `getattr` kèm **một biến** tên `_POLICY` —
      `asyncio.WindowsSelectorEventLoopPolicy` không tồn tại trên Linux, nên
      một tham chiếu tường minh (kể cả `getattr` với chuỗi hằng, thứ mypy vẫn
      phân giải) sẽ thành `attr-defined` ngay khi nhánh trở nên reachable.
    """
    if not needs_selector_loop():
        yield
        return
    # ⚠️ `getattr` với **biến**, không phải với chuỗi hằng: ruff `B009` tự sửa
    # `getattr(x, "hằng")` thành `x.hằng`, và bản tự sửa ấy chính là thứ mypy
    # từ chối trên Linux. Hai linter kéo ngược nhau, và bên tự sửa thắng im lặng.
    policy_cls = getattr(asyncio, _POLICY)
    previous = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(policy_cls())
    try:
        yield
    finally:
        asyncio.set_event_loop_policy(previous)
