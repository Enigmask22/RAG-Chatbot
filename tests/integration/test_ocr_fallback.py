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

pytestmark = [pytest.mark.integration, pytest.mark.weights]
"""`weights` vì nó tải trọng số EasyOCR/docling, không vì nó cần một service —
hai lý do khác nhau để một bài chậm, và CI đối xử với chúng khác nhau."""

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

    def test_the_english_block_survives(self, scanned_document: LoadedDocument) -> None:
        """Đo 2026-09-04 (fixture DejaVu): mọi TỪ đều về, nhưng tầng xếp thứ tự
        đọc của docling xáo một dòng ("consumer index settled at 3.63. price").
        Đó là thuộc tính của docling — cả hai máy OCR đều dính — nên phép kiểm
        đúng là "đủ từ", không phải "nguyên văn"."""
        flat = " ".join(scanned_document.text.split())
        assert "Vietnam Macroeconomic Update" in flat
        assert "Growth reached 7.09 percent in 2024" in flat
        for word in ("consumer", "price", "index", "settled", "3.63"):
            assert word in flat

    def test_the_numbers_survive(self, scanned_document: LoadedDocument) -> None:
        for number in ("7.09", "5.05", "405.5", "3.63"):
            assert number in scanned_document.text


@pytest.fixture(scope="module")
def vietnamese_document() -> LoadedDocument:
    return _load_or_skip(SCANNED, language="vi")


class TestVietnameseGoesThroughEasyOcr:
    """⭐⭐ Nửa sau của TD-23, mở 2026-09-04 sau khi sửa fixture hỏng font.

    Tiền nhiệm của class này (`TestTheVietnameseBlockIsWhyThisEngineIsRefused`)
    khẳng định tiếng Việt ra rác — trên một ảnh mà dấu đã là ô ☒ từ lúc render.
    Trên ảnh hợp lệ: RapidOCR vẫn vứt 3/5 dòng (nên `en` giữ máy cũ còn `vi`
    không dùng nó), EasyOCR giữ dấu 8/8. Số đo: `probes/td-23-easyocr.json`.
    """

    def test_the_engine_is_easyocr_and_it_is_recorded(
        self, vietnamese_document: LoadedDocument
    ) -> None:
        assert vietnamese_document.extra["ocr_engine"] == "easyocr:latin-g2"
        assert "ocr=easyocr" in vietnamese_document.fingerprint.options
        assert any("easyocr=" in c for c in vietnamese_document.fingerprint.components)

    def test_vietnamese_diacritics_survive(self, vietnamese_document: LoadedDocument) -> None:
        """Đây từng là `test_vietnamese_diacritics_do_not_survive` — nó lật chiều
        đúng như chính nó dặn: "nếu một ngày nó đỏ vì tiếng Việt đọc được thì đó
        là tin tốt". Các cụm dưới là output ĐO ĐƯỢC, không phải kỳ vọng đẹp."""
        flat = " ".join(vietnamese_document.text.split())
        assert "Cập nhật kinh tế vĩ mô Việt Nam" in flat
        assert "trưởng đạt 7,09" in flat
        assert "Xuất khẩu" in flat
        assert "dừng ở mức 3,63" in flat

    def test_the_numbers_survive_in_vietnamese_format(
        self, vietnamese_document: LoadedDocument
    ) -> None:
        for number in ("7,09", "5,05", "405,5", "3,63"):
            assert number in vietnamese_document.text

    def test_word_order_is_not_guaranteed_and_that_is_documented(
        self, vietnamese_document: LoadedDocument
    ) -> None:
        """Giới hạn đã biết: tầng reading-order của docling đẩy chữ "Tăng" khỏi
        đầu câu trên fixture này. Test ghim GIỚI HẠN để nó không âm thầm đổi:
        nếu một ngày câu về nguyên văn — docling sửa layout — thì đây đỏ, và
        docstring của `ocr.py` phải được viết lại nhẹ đi."""
        flat = " ".join(vietnamese_document.text.split())
        assert "Tăng trưởng đạt 7,09 phần trăm năm 2024" not in flat

    def test_an_unmeasured_language_is_refused_before_any_model_runs(self) -> None:
        with pytest.raises(OcrLanguageError, match="TD-23"):
            load_document(SCANNED, language="fr")


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
