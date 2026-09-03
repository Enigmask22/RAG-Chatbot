"""Điểm vào của Serving Plane — `uv run python -m serving`.

## ⭐⭐ Vì sao cần một điểm vào riêng thay vì gọi thẳng `uvicorn`

`W4-06` là hạng mục đầu tiên chạm Postgres **bất đồng bộ**, và lần chạy thật đầu
tiên trả về:

```
InterfaceError: Psycopg cannot use the 'ProactorEventLoop' to run in async mode
```

Trên Windows, vòng lặp sự kiện mặc định của asyncio là `ProactorEventLoop`, và
driver async của `psycopg` v3 chỉ chạy trên `SelectorEventLoop`. Nên `POST /chat`
trả `503` ở **mọi** request trên máy dev, trong khi container Linux của `W4-13`
sẽ chạy đúng ngay từ lần đầu.

⚠️ Đó là hướng lệch **ngược** với hướng quen thuộc: không phải "chạy được ở máy
tôi, hỏng ở production" mà là "hỏng ở máy tôi, chạy được ở production". Nó nguy
hiểm theo cách riêng — cám dỗ là kết luận "chuyện của Windows thôi" và bỏ qua,
rồi mọi test tích hợp từ `W4-06` trở đi không chạy được ở nơi duy nhất mà tôi
thực sự chạy chúng.

⚠️ Và cách chữa hiển nhiên **không** hoạt động. `asyncio.set_event_loop_policy(
WindowsSelectorEventLoopPolicy())` là câu trả lời mọi nơi đưa ra, nhưng uvicorn
không đi qua policy: `uvicorn/loops/asyncio.py` trả thẳng một `loop_factory`, và
trên win32 nó là `ProactorEventLoop` trừ khi bật subprocess. Đổi policy xong thì
lỗi **giữ nguyên từng chữ** — tôi đã đo đúng điều đó trước khi đọc mã uvicorn.

Nên phải tự chạy `Server.serve()` bằng vòng lặp của mình. Vì thế: một điểm vào.

⚠️ `SelectorEventLoop` trên Windows giới hạn 512 socket và không hỗ trợ
subprocess. Cả hai đều không chạm tới việc chạy một tiến trình API ở máy dev, và
trên Linux nhánh này không chạy.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence

__all__ = ["main", "needs_selector_loop"]


def needs_selector_loop() -> bool:
    """`True` khi vòng lặp mặc định của nền tảng này không chạy được `psycopg`."""
    return sys.platform == "win32"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m serving")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--app",
        default="serving.api.app:app",
        help="đường dẫn ASGI. Dùng `--factory` nếu nó là hàm dựng app.",
    )
    parser.add_argument("--factory", action="store_true")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    import uvicorn

    server = uvicorn.Server(
        uvicorn.Config(
            args.app,
            host=args.host,
            port=args.port,
            factory=args.factory,
            log_level=args.log_level,
            # `--reload` cố ý **không** có mặt: nó dựng một tiến trình con, và
            # tiến trình con ấy không đi qua `main()` nên nó lấy lại vòng lặp
            # mặc định — tức đúng lỗi ở đầu file, quay lại một cách im lặng.
        )
    )
    if needs_selector_loop():
        asyncio.run(server.serve(), loop_factory=asyncio.SelectorEventLoop)
    else:
        server.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
