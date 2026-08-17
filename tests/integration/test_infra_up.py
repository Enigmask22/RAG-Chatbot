"""W1-05 — `make up` phải dựng được Qdrant + Postgres + Redis và cả ba đều trả lời.

Cố ý ping bằng **giao thức thật của từng service** chứ không chỉ mở TCP socket:
một cổng mở chỉ chứng minh có tiến trình nào đó đang nghe, không chứng minh
Postgres đã sẵn sàng nhận kết nối. Cách này cũng không cần thêm dependency nào.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import urllib.error
import urllib.request

import pytest

pytestmark = pytest.mark.integration

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "127.0.0.1")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379

_HINT = "Chưa chạy `make up`, hoặc service chưa healthy."


def test_qdrant_is_healthy() -> None:
    try:
        with urllib.request.urlopen(f"{QDRANT_URL}/healthz", timeout=5) as response:
            body = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError) as exc:
        pytest.fail(f"Qdrant không trả lời tại {QDRANT_URL}: {exc}. {_HINT}")
    assert "passed" in body.lower()


def test_qdrant_reports_version() -> None:
    with urllib.request.urlopen(f"{QDRANT_URL}/", timeout=5) as response:
        payload = json.loads(response.read())
    # Ghim phiên bản lớn: nâng cấp Qdrant đổi hành vi named vector/sparse index,
    # và đó là thứ W2 phụ thuộc trực tiếp.
    assert payload["version"].startswith("1."), payload


def test_postgres_speaks_postgres_protocol() -> None:
    # Gói SSLRequest: độ dài 8, mã 80877103. Server Postgres trả đúng 1 byte 'S' hoặc 'N'.
    try:
        with socket.create_connection((POSTGRES_HOST, POSTGRES_PORT), timeout=5) as sock:
            sock.sendall(struct.pack("!ii", 8, 80877103))
            reply = sock.recv(1)
    except OSError as exc:
        pytest.fail(f"Postgres không trả lời tại {POSTGRES_HOST}:{POSTGRES_PORT}: {exc}. {_HINT}")
    assert reply in (b"S", b"N"), f"Cổng có mở nhưng không phải Postgres: {reply!r}"


def test_redis_responds_to_ping() -> None:
    try:
        with socket.create_connection((REDIS_HOST, REDIS_PORT), timeout=5) as sock:
            sock.sendall(b"PING\r\n")
            reply = sock.recv(64)
    except OSError as exc:
        pytest.fail(f"Redis không trả lời tại {REDIS_HOST}:{REDIS_PORT}: {exc}. {_HINT}")
    assert reply.startswith(b"+PONG"), reply
