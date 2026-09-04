"""`W4-12` — che PII trong log.

Câu hỏi của phần này không phải "regex có khớp email không" mà là **"một dòng
log do thư viện bên thứ ba sinh ra có được che không"** — vì đó là chỗ PII thật
sự lọt ra, và là chỗ mà kỷ luật "nhớ gọi `redact_pii()`" không với tới.
"""

from __future__ import annotations

import io
import json
import logging

import pytest

from rag_core.generation import RedactingFilter, redact_pii
from serving.core.logging import configure_logging

pytestmark = pytest.mark.security


class TestRedactPii:
    def test_an_email_is_replaced(self) -> None:
        assert redact_pii("gửi tới khanh.le@example.vn nhé") == "gửi tới [email] nhé"

    def test_a_vietnamese_mobile_is_replaced(self) -> None:
        assert redact_pii("gọi 0912345678") == "gọi [sđt]"
        assert redact_pii("gọi +84912345678") == "gọi [sđt]"

    def test_a_twelve_digit_id_is_replaced(self) -> None:
        assert redact_pii("CCCD 001099012345 của ông A") == "CCCD [cccd] của ông A"

    def test_a_card_number_passing_luhn_is_replaced(self) -> None:
        assert redact_pii("thẻ 4539 1488 0343 6467") == "thẻ [thẻ]"

    def test_a_long_number_failing_luhn_survives(self) -> None:
        """⭐ Phép kiểm Luhn là thứ giữ luật thẻ khỏi nuốt số liệu kinh tế.

        Corpus của dự án đầy giá trị VND viết liền. Che chúng đi thì log thôi
        nói được điều mà log tồn tại để nói — và không ai nhận ra, vì một con
        số bị che trông y hệt một con số đáng che.
        """
        assert redact_pii("tổng vốn 1234567890123 đồng") == "tổng vốn 1234567890123 đồng"

    def test_a_year_or_percentage_is_untouched(self) -> None:
        assert redact_pii("GDP 2025 tăng 7,5% so với 2024") == "GDP 2025 tăng 7,5% so với 2024"

    def test_a_decimal_number_is_not_mistaken_for_a_phone(self) -> None:
        assert redact_pii("tỉ giá 24.850,75 đồng") == "tỉ giá 24.850,75 đồng"

    def test_redaction_is_idempotent(self) -> None:
        once = redact_pii("mail a@b.vn")
        assert redact_pii(once) == once


class TestRedactingFilterOnRecords:
    def _record(self, msg: str, *args: object, **extra: object) -> logging.LogRecord:
        record = logging.LogRecord("x", logging.INFO, __file__, 1, msg, args or (), None)
        for key, value in extra.items():
            setattr(record, key, value)
        RedactingFilter().filter(record)
        return record

    def test_the_message_is_redacted(self) -> None:
        assert "[email]" in self._record("liên hệ a@b.vn").getMessage()

    def test_lazy_format_arguments_are_redacted(self) -> None:
        """`logger.info("user %s", email)` — PII nằm trong `args`, không trong
        `msg`. Che mỗi `msg` là che đúng nửa không chứa dữ liệu."""
        assert "[email]" in self._record("user %s", "a@b.vn").getMessage()

    def test_extra_fields_are_redacted(self) -> None:
        """`extra=` đi thẳng lên cấp một của dòng JSON (xem `JsonFormatter`), nên
        một `extra={"q": <câu hỏi người dùng>}` là một đường rò trọn vẹn."""
        record = self._record("ok", question="gọi 0912345678")
        assert record.question == "gọi [sđt]"  # type: ignore[attr-defined]

    def test_the_logger_name_is_never_rewritten(self) -> None:
        """Che tên logger thì bộ lọc theo module trong log aggregator chết —
        và không ai nghi ngờ bộ che PII."""
        record = self._record("ok")
        assert record.name == "x"


class TestConfiguredLoggingRedacts:
    def test_a_third_party_logger_is_redacted_end_to_end(self) -> None:
        """⭐ Phép kiểm thật của hạng mục này.

        `httpx` chưa bao giờ nghe nói tới `redact_pii`, và nó ghi URL kèm query
        string. Filter gắn trên **handler** nên mọi bản ghi tới được chỗ ghi ra
        đều đã đi qua nó — kể cả của thư viện chưa ai nghĩ tới.
        """
        stream = io.StringIO()
        configure_logging("INFO", stream=stream)
        try:
            logging.getLogger("httpx").info(
                "HTTP Request: GET https://api.example/v1?user=a@b.vn&phone=0912345678"
            )
            line = json.loads(stream.getvalue().strip().splitlines()[-1])
        finally:
            logging.getLogger().handlers.clear()

        assert "a@b.vn" not in line["msg"]
        assert "0912345678" not in line["msg"]
        assert "[email]" in line["msg"] and "[sđt]" in line["msg"]
        assert line["logger"] == "httpx"
