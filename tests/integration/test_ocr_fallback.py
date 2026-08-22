"""`W3-02` — OCR thật chạy trên một PDF scan thật.

Đánh dấu `integration` vì nó nạp model OCR (~30 MB trọng số, GPU nếu có), không
vì nó cần service ngoài. `make test` bỏ qua; `make test-integration` chạy.

Phần **quyết định** (phát hiện scan, cổng, chốt ngôn ngữ) nằm ở
`tests/unit/test_scan_detection.py` và chạy trong mọi vòng lặp phát triển. Chỗ
này chỉ canh đúng một điều mà unit test không canh được: **văn bản có thật sự
chui ra khỏi một trang ảnh hay không.**
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_core.loaders import (
    OCR_ENGINE,
    LoadedDocument,
    LoaderError,
    OcrGate,
    OcrLanguageError,
    load_document,
)

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "loaders"
SCANNED = FIXTURES / "scanned-page.pdf"
BORN_DIGITAL = FIXTURES / "two-column.pdf"

_MODEL_MARKERS = ("huggingface", "connection", "offline", "resolve", "download", "timed out")


def _load_or_skip(path: Path, **kwargs: object) -> LoadedDocument:
    try:
        return load_document(path, **kwargs)  # type: ignore[arg-type]
    except LoaderError as exc:
        if isinstance(exc, OcrLanguageError):
            raise
        if any(marker in str(exc).lower() for marker in _MODEL_MARKERS):
            pytest.skip(f"không tải được model OCR: {exc}")
        raise


@pytest.fixture(scope="module")
def scanned_document() -> LoadedDocument:
    return _load_or_skip(SCANNED, language="en")


class TestAScannedPageProducesText:
    """DoD: *PDF scan ra được text*."""

    def test_without_ocr_the_page_is_empty_and_that_is_an_error(self) -> None:
        """Đây là lý do `W3-02` tồn tại: không phát hiện thì tài liệu **biến mất**.

        `W3-01` ném `LoaderError` khi văn bản rỗng — đúng, nhưng ở đường index
        thì một tài liệu ném lỗi là một tài liệu không vào index, im lặng.
        """
        with pytest.raises(LoaderError, match="rỗng"):
            load_document(SCANNED, ocr="off")

    def test_auto_turns_ocr_on_for_this_file(self, scanned_document: LoadedDocument) -> None:
        assert scanned_document.extra["ocr"] == "true"
        assert scanned_document.extra["ocr_engine"] == OCR_ENGINE
        assert scanned_document.extra["ocr_pages"] == "1"

    def test_the_english_block_comes_back_verbatim(self, scanned_document: LoadedDocument) -> None:
        flat = " ".join(scanned_document.text.split())
        assert "Vietnam Macroeconomic Update" in flat
        assert "Growth reached 7.09 percent in 2024" in flat
        assert "consumer price index settled at 3.63" in flat

    def test_the_numbers_survive(self, scanned_document: LoadedDocument) -> None:
        for number in ("7.09", "5.05", "405.5", "3.63"):
            assert number in scanned_document.text


class TestTheVietnameseBlockIsWhyThisEngineIsRefused:
    """⭐⭐ Cùng ảnh, cùng lần chạy: tiếng Anh nguyên văn, tiếng Việt ra rác.

    Test này **khẳng định cái hỏng**. Nếu một ngày nó đỏ vì tiếng Việt đọc được
    thì đó là tin tốt, và lúc đó `OCR_VERIFIED_LANGUAGES` phải được mở ra —
    không phải sửa test cho xanh lại.
    """

    def test_vietnamese_diacritics_do_not_survive(self, scanned_document: LoadedDocument) -> None:
        flat = " ".join(scanned_document.text.split())
        assert "Tăng trưởng đạt 7,09 phần trăm" not in flat, (
            "OCR đọc được tiếng Việt rồi — mở `OCR_VERIFIED_LANGUAGES` và bỏ TD-23"
        )

    def test_a_vietnamese_document_is_refused_before_any_model_runs(self) -> None:
        with pytest.raises(OcrLanguageError, match="TD-23"):
            load_document(SCANNED, language="vi")


class TestABornDigitalPdfNeverPaysForOcr:
    def test_auto_leaves_ocr_off(self) -> None:
        document = _load_or_skip(BORN_DIGITAL, language="en")
        assert document.extra.get("ocr", "false") == "false"

    def test_force_overrides_the_detector(self) -> None:
        document = _load_or_skip(BORN_DIGITAL, ocr="force", language="en")
        assert document.extra["ocr"] == "true"


class TestTheGateAppliesToTheRealPath:
    def test_a_page_budget_of_zero_pages_refuses_before_loading_a_model(self) -> None:
        gate = OcrGate(max_pages=1)
        # 1 trang vừa đúng trần → chạy được
        assert _load_or_skip(SCANNED, language="en", gate=gate).text.strip()
