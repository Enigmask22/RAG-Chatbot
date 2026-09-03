"""`W4-03` — log JSON mang `request_id`, và phép thử `/ready`.

Hai nhóm, hai kiểu rủi ro khác nhau:

* **log** — hỏng ở đây không làm gì đỏ. Một dòng log biến mất, một secret lọt ra,
  hay tiếng Việt thành `\\u1ea1` chỉ lộ ra khi đã cần đọc log, tức lúc đang có sự cố.
* **probe** — hỏng ở đây *khuếch đại* sự cố: không hạn giờ thì `/ready` treo theo
  phụ thuộc, không nhớ tạm thì chính phép thử là tải.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time

import pytest
from pydantic import SecretStr

from serving.core.logging import (
    JsonFormatter,
    bind_request_id,
    configure_logging,
    reset_request_id,
)
from serving.core.probes import Check, ReadinessProbes


def emit(**kwargs: object) -> dict[str, object]:
    """Chạy một bản ghi qua formatter thật rồi parse lại."""
    record = logging.LogRecord(
        name="serving.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=kwargs.pop("msg", "xin chào"),
        args=(),
        exc_info=kwargs.pop("exc_info", None),  # type: ignore[arg-type]
    )
    record.__dict__.update(kwargs)
    parsed: dict[str, object] = json.loads(JsonFormatter().format(record))
    return parsed


# ---------------------------------------------------------------------------
# 1. Định dạng
# ---------------------------------------------------------------------------


def test_a_record_is_exactly_one_json_line() -> None:
    line = JsonFormatter().format(
        logging.LogRecord("x", logging.INFO, __file__, 1, "một\nhai", (), None)
    )
    assert "\n" not in line
    assert json.loads(line)["msg"] == "một\nhai"


def test_vietnamese_stays_readable() -> None:
    """`ensure_ascii=True` vẫn cho JSON hợp lệ nhưng biến mọi chữ có dấu thành
    `\\uXXXX` — tức `grep` trên log thành vô dụng, ở một repo mà mọi thông báo
    lỗi đều là tiếng Việt."""
    assert emit(msg="ngữ cảnh thiếu")["msg"] == "ngữ cảnh thiếu"


def test_extra_fields_land_at_the_top_level() -> None:
    """Phẳng chứ không lồng: truy vấn `level=ERROR AND collection=rag_bgem3_ctx`
    trong log aggregator chỉ viết được khi trường ấy ở cùng cấp."""
    assert emit(collection="rag_bgem3_ctx")["collection"] == "rag_bgem3_ctx"


def test_a_colliding_extra_cannot_overwrite_a_core_field() -> None:
    """Một `extra={"level": ...}` vô ý không được phép làm hỏng đúng trường mà
    bộ lọc mức độ nghiêm trọng dựa vào."""
    assert emit(level="ok")["level"] == "INFO"


def test_an_unserialisable_extra_does_not_swallow_the_record() -> None:
    """Formatter ném thì `logging` nuốt bản ghi thành `--- Logging error ---`.
    Một dòng format xấu vẫn hơn một dòng biến mất."""
    assert "path" in emit(path=object())


def test_a_secret_never_reaches_the_log_in_clear_text() -> None:
    """⭐ `SecretStr` giữ lời hứa của nó xuống tận đây, và điều đó phụ thuộc vào
    `default=str` — `json.dumps` gọi `str()`, và `str()` của `SecretStr` là dấu sao."""
    line = json.dumps(emit(api_key=SecretStr("sk-thật-đấy")), ensure_ascii=False)
    assert "sk-thật-đấy" not in line
    assert "**" in line


def test_the_request_id_rides_along_without_being_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    """⭐ Cả lý do tồn tại của `ContextVar` ở đây: `rag_core` và `qdrant-client`
    không biết gì về HTTP, nhưng dòng log của chúng vẫn phải truy được về request."""
    assert "request_id" not in emit()
    token = bind_request_id("abc123")
    try:
        assert emit()["request_id"] == "abc123"
    finally:
        reset_request_id(token)
    assert "request_id" not in emit()


def test_an_exception_keeps_its_traceback() -> None:
    try:
        raise ValueError("vỡ")
    except ValueError:
        import sys

        payload = emit(exc_info=sys.exc_info())
    assert "ValueError: vỡ" in str(payload["exc"])


# ---------------------------------------------------------------------------
# 2. configure_logging
# ---------------------------------------------------------------------------


def test_pre_existing_handlers_are_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    """⭐ `uvicorn` tự gọi `dictConfig` lúc khởi động. Cùng sống với handler của
    ta thì mỗi dòng ra **hai** lần — một bản JSON, một bản văn xuôi — nên log
    vừa gấp đôi vừa không parse được."""
    root = logging.getLogger()
    saved = list(root.handlers)
    noise = io.StringIO()
    root.addHandler(logging.StreamHandler(noise))
    try:
        stream = io.StringIO()
        configure_logging("INFO", stream=stream)
        logging.getLogger("serving.test").info("một lần thôi")
        assert noise.getvalue() == ""
        assert len(json.loads(stream.getvalue().strip())) >= 4
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in saved:
            root.addHandler(handler)


def test_the_uvicorn_access_log_is_silenced() -> None:
    """Nó là văn xuôi cố định (không đi qua formatter của ta) và nó ghi cả query
    string. Dòng access thay thế do middleware sinh ra."""
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    logging.getLogger("uvicorn.access").info("GET /chat?q=bí+mật HTTP/1.1 200")
    assert stream.getvalue() == ""


# ---------------------------------------------------------------------------
# 3. ReadinessProbes
# ---------------------------------------------------------------------------


def test_a_passing_check_is_ok() -> None:
    probes = ReadinessProbes(checks={"qdrant": lambda: None}, ttl_s=0.0)
    (result,) = asyncio.run(probes.run())
    assert (result.name, result.ok, result.detail) == ("qdrant", True, None)


def test_the_failure_reason_survives_to_the_response() -> None:
    def down() -> None:
        raise ConnectionError("connection refused")

    probes = ReadinessProbes(checks={"qdrant": down}, ttl_s=0.0)
    (result,) = asyncio.run(probes.run())
    assert result.ok is False
    assert result.detail == "connection refused"


def test_a_silent_exception_still_names_itself() -> None:
    """Một exception không có lời (`raise TimeoutError()`) vẫn phải nói được nó
    là gì, nếu không thì `/ready` trả `"detail": ""`."""

    class Rỗng(Exception):
        pass

    probes = ReadinessProbes(checks={"x": _raiser(Rỗng())}, ttl_s=0.0)
    (result,) = asyncio.run(probes.run())
    assert result.detail == "Rỗng"


def test_a_hung_dependency_does_not_hang_ready() -> None:
    """⭐⭐ Không có hạn giờ thì orchestrator không nhận 503 — nó nhận **timeout
    của chính probe**, và hai thứ đó ánh xạ khác nhau ở mọi hệ điều phối."""
    probes = ReadinessProbes(checks={"qdrant": lambda: time.sleep(1.5)}, timeout_s=0.05, ttl_s=0.0)

    async def poll() -> tuple[float, bool, str]:
        started = time.perf_counter()
        (result,) = await probes.run()
        return time.perf_counter() - started, result.ok, str(result.detail)

    # ⚠️ Đo **bên trong** vòng lặp, không đo quanh `asyncio.run`. Luồng bị bỏ chờ
    # vẫn sống, và `asyncio.run` khi đóng có `shutdown_default_executor()` — nó
    # *đợi* luồng ấy. Tức phép đo quanh `asyncio.run` đo thời gian **tắt tiến
    # trình**, không đo thời gian `/ready` trả lời.
    elapsed, ok, detail = asyncio.run(poll())
    assert ok is False
    assert "quá hạn" in detail
    assert elapsed < 0.5


def test_the_qdrant_client_gives_up_before_the_probe_does() -> None:
    """⭐ Ràng buộc giữa hai hằng số ở hai file, và nó thật sự quan trọng.

    `wait_for` bỏ *chờ* nhưng không giết được luồng đang chờ socket. Nếu hạn giờ
    của client **không** ngắn hơn hạn giờ của probe thì mỗi chu kỳ TTL bỏ lại một
    luồng, và những luồng ấy còn làm **treo cả lúc tắt tiến trình** —
    `shutdown_default_executor()` đợi chúng, nên một lệnh deploy trở thành một
    lần chờ hết hạn giờ.
    """
    from serving.core.runtime import QdrantRuntimeBuilder

    assert QdrantRuntimeBuilder(url="x").qdrant_timeout_s < ReadinessProbes(checks={}).timeout_s


def test_repeated_polling_does_not_hit_the_dependency_every_time() -> None:
    """10 replica × mỗi 3 giây gửi vào một Qdrant *đang yếu* thì phép thử sức
    khoẻ là một phần của nguyên nhân."""
    calls: list[int] = []
    probes = ReadinessProbes(checks={"qdrant": lambda: calls.append(1)}, ttl_s=60.0)

    async def poll() -> None:
        for _ in range(5):
            await probes.run()

    asyncio.run(poll())
    assert calls == [1]


def test_a_failure_is_cached_just_like_a_success() -> None:
    """⭐ Chỉ nhớ kết quả tốt nghe an toàn hơn, nhưng nó nghĩa là phụ thuộc bị dội
    mạnh nhất đúng lúc nó yếu nhất."""
    calls: list[int] = []

    def down() -> None:
        calls.append(1)
        raise ConnectionError("nope")

    probes = ReadinessProbes(checks={"qdrant": down}, ttl_s=60.0)

    async def poll() -> None:
        for _ in range(4):
            await probes.run()

    asyncio.run(poll())
    assert calls == [1]


def test_force_bypasses_the_cache() -> None:
    calls: list[int] = []
    probes = ReadinessProbes(checks={"qdrant": lambda: calls.append(1)}, ttl_s=60.0)

    async def poll() -> None:
        await probes.run()
        await probes.run(force=True)

    asyncio.run(poll())
    assert calls == [1, 1]


def test_simultaneous_polls_collapse_into_one_run() -> None:
    """⚠️ Cách hỏng mà một cache TTL ngây thơ bỏ sót: mọi lượt gọi đến *trước khi*
    lượt đầu kịp ghi cache đều thấy cache rỗng, nên chúng cùng chạy probe. Đúng
    lúc đông nhất — sau một lần deploy, mọi replica cùng khởi động."""
    calls: list[int] = []

    def slow() -> None:
        time.sleep(0.05)
        calls.append(1)

    probes = ReadinessProbes(checks={"qdrant": slow}, ttl_s=60.0)

    async def storm() -> None:
        await asyncio.gather(*(probes.run() for _ in range(8)))

    asyncio.run(storm())
    assert calls == [1]


def _raiser(exc: Exception) -> Check:
    def check() -> None:
        raise exc

    return check
