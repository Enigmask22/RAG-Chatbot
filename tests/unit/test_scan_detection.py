"""`W3-02` — phát hiện trang scan, cổng OCR, và cái chốt ngôn ngữ.

Không test nào ở đây chạy OCR: OCR thật nằm ở
`tests/integration/test_ocr_fallback.py`. Chỗ này canh phần **quyết định** —
và quyết định mới là phần quan trọng, vì chi phí hoá ra không phải lý do
`W3-02` tồn tại (xem `ocr.SECONDS_PER_PAGE`: cận biên 0,35 s/trang, không phải
70 s). Lý do là **đúng/sai**: không phát hiện thì tài liệu ảnh biến mất khỏi
index, và bật OCR bừa thì tài liệu tiếng Việt vào index dưới dạng rác.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from rag_core.loaders import (
    MIN_CHARS_PER_IN2,
    SCAN_PAGE_RATIO,
    OcrBudgetError,
    OcrGate,
    OcrLanguageError,
    PageText,
    ScanReport,
    detect_scan,
    engine_for,
    ocr_supports,
    page_texts,
    require_ocr_support,
)

pytestmark = pytest.mark.weights
"""Ba module này parse tài liệu thật bằng docling, tức chúng cần **trọng số
model tải về** — xem marker `weights` trong `pyproject.toml`. Vẫn chạy trong
`make test` ở máy đã `make install`; CI nhanh loại chúng ra và
`tests/unit/test_ci_tiers.py` khoá danh sách ấy lại."""


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "loaders"
BORN_DIGITAL = FIXTURES / "two-column.pdf"
SCANNED = FIXTURES / "scanned-page.pdf"


def _page(number: int, characters: int, width: float = 612.0, height: float = 792.0) -> PageText:
    return PageText(number=number, characters=characters, width_pt=width, height_pt=height)


class TestDensityIsPerAreaNotPerPage:
    """Cùng một lượng chữ trên slide 16:9 và trên A4 là hai mật độ khác nhau."""

    def test_same_text_on_a_bigger_page_has_lower_density(self) -> None:
        letter = _page(1, 900)
        slide = _page(1, 900, width=960.0, height=540.0)
        assert letter.density > slide.density

    def test_area_is_in_square_inches(self) -> None:
        # Letter = 8,5 × 11 = 93,5 in²
        assert _page(1, 0).area_in2 == pytest.approx(93.5, abs=0.01)

    def test_a_zero_area_page_does_not_divide_by_zero(self) -> None:
        assert _page(1, 100, width=0.0, height=0.0).density == 0.0


class TestTheThresholdSeparatesRealDocumentsFromScans:
    """Ngưỡng đo từ PDF World Bank thật, không đoán — xem docstring của `scan.py`."""

    def test_the_born_digital_fixture_has_a_text_layer(self) -> None:
        report = detect_scan(BORN_DIGITAL)
        assert report.empty_pages == ()
        assert not report.needs_ocr

    def test_the_scanned_fixture_has_none(self) -> None:
        report = detect_scan(SCANNED)
        assert report.empty_pages == (1,)
        assert report.median_density == 0.0
        assert report.needs_ocr

    def test_page_numbers_are_one_based(self) -> None:
        assert page_texts(SCANNED)[0].number == 1

    def test_the_sparsest_real_page_measured_stays_above_the_threshold(self) -> None:
        """Trang thưa nhất **có text layer** trong mẫu thật là 3,03 ký tự/in²."""
        assert MIN_CHARS_PER_IN2 < 3.03


class TestOneEmptyPageIsNotAScannedDocument:
    """⭐ Báo cáo born-digital THẬT vẫn có trang trống: 1/129 và 4/112.

    Luật "có một trang thiếu text ⇒ scan" sẽ đẩy 100% báo cáo World Bank vào OCR.
    """

    def test_a_single_empty_page_out_of_many_does_not_trigger_ocr(self) -> None:
        pages = (_page(1, 0), *(_page(i, 3000) for i in range(2, 130)))
        assert ScanReport(pages).empty_ratio == pytest.approx(1 / 129, abs=1e-6)
        assert not ScanReport(pages).needs_ocr

    def test_four_empty_pages_out_of_112_still_does_not(self) -> None:
        pages = tuple(_page(i, 0 if i <= 4 else 3000) for i in range(1, 113))
        assert not ScanReport(pages).needs_ocr

    def test_an_all_empty_document_does(self) -> None:
        assert ScanReport(tuple(_page(i, 0) for i in range(1, 20))).needs_ocr

    def test_the_threshold_sits_between_the_two_measured_clusters(self) -> None:
        assert 4 / 112 < SCAN_PAGE_RATIO < 1.0

    def test_an_empty_document_never_needs_ocr(self) -> None:
        assert not ScanReport(()).needs_ocr
        assert ScanReport(()).empty_ratio == 0.0


class TestEachLanguageGetsTheEngineThatWasMeasuredForIt:
    """⭐⭐ Chọn máy theo ngôn ngữ, và chỉ ngôn ngữ ĐÃ ĐO mới được chạy.

    Bản đầu của class này khẳng định "tiếng Việt bị từ chối" — dựa trên một phép
    đo mà 2026-09-04 lộ ra là chấm trên fixture hỏng font (dấu là ô ☒ ngay trong
    ảnh). Đo lại trên ảnh hợp lệ: RapidOCR vẫn vứt 3/5 dòng tiếng Việt, còn
    EasyOCR giữ dấu 8/8 — nên `vi` giờ đi EasyOCR thay vì bị chặn. Lý do từ chối
    ngôn ngữ CHƯA đo thì giữ nguyên: rác trông như nội dung, nguy hiểm hơn rỗng.
    """

    def test_english_stays_on_the_default_engine(self) -> None:
        """Đổi máy là đổi văn bản xuất ra, tức đổi vân tay parse — `en` không
        được âm thầm dời đi chỉ vì có máy mới."""
        assert ocr_supports("en")
        assert require_ocr_support("en") == "rapidocr"
        assert engine_for("en") == "rapidocr"

    def test_vietnamese_goes_to_easyocr(self) -> None:
        assert ocr_supports("vi")
        assert require_ocr_support("vi", name="bao-cao.pdf") == "easyocr"

    def test_an_unmeasured_language_is_refused(self) -> None:
        """EasyOCR *tuyên bố* đọc được tiếng Pháp — nhưng tuyên bố không phải
        phép đo, và bài học của chính TD-23 là hai thứ đó khác nhau."""
        assert not ocr_supports("fr")
        with pytest.raises(OcrLanguageError, match="TD-23"):
            require_ocr_support("fr", name="rapport.pdf")

    def test_the_refusal_message_carries_the_measurement(self) -> None:
        with pytest.raises(OcrLanguageError, match="3/5 dòng"):
            require_ocr_support("zh", name="bao-cao.pdf")

    def test_case_and_whitespace_do_not_change_the_routing(self) -> None:
        assert engine_for(" VI ") == "easyocr"
        assert engine_for(" EN ") == "rapidocr"
        assert not ocr_supports(" FR ")

    def test_unknown_language_is_allowed_but_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING", logger="rag_core.loaders.ocr"):
            engine = require_ocr_support(None, name="khong-ro.pdf")
        assert engine == "rapidocr"
        assert "chưa biết ngôn ngữ" in caplog.text


class TestTheGateStopsTwoThingsAtOnce:
    def test_a_document_over_the_page_budget_is_refused(self) -> None:
        gate = OcrGate(max_pages=50)
        with pytest.raises(OcrBudgetError, match="129 trang"):
            gate.check_budget(129, name="scan.pdf")

    def test_the_budget_message_says_how_long_it_would_take(self) -> None:
        with pytest.raises(OcrBudgetError, match="phút"):
            OcrGate(max_pages=10).check_budget(200)

    def test_a_document_within_budget_passes(self) -> None:
        with OcrGate(max_pages=5).reserve(5):
            pass

    def test_concurrency_is_one_by_default(self) -> None:
        gate = OcrGate()
        assert gate.max_concurrent == 1

        entered = threading.Event()
        release = threading.Event()
        overlapped = threading.Event()

        def hold() -> None:
            with gate.reserve(1):
                entered.set()
                release.wait(timeout=5)

        def intrude() -> None:
            entered.wait(timeout=5)
            if gate._semaphore.acquire(blocking=False):
                overlapped.set()
                gate._semaphore.release()

        first = threading.Thread(target=hold)
        second = threading.Thread(target=intrude)
        first.start()
        second.start()
        second.join(timeout=5)
        release.set()
        first.join(timeout=5)

        assert not overlapped.is_set(), "hai job OCR chạy song song — cổng không giữ"

    def test_the_slot_is_returned_even_when_the_body_raises(self) -> None:
        gate = OcrGate()
        with pytest.raises(RuntimeError), gate.reserve(1):
            raise RuntimeError("hỏng giữa chừng")
        with gate.reserve(1):  # nếu khe bị rò thì dòng này treo
            pass

    def test_a_second_job_waits_instead_of_failing(self) -> None:
        gate = OcrGate(max_concurrent=1)
        order: list[str] = []

        def worker(tag: str, hold: float) -> None:
            with gate.reserve(1):
                order.append(f"{tag}-vào")
                time.sleep(hold)
                order.append(f"{tag}-ra")

        first = threading.Thread(target=worker, args=("A", 0.15))
        first.start()
        time.sleep(0.03)
        second = threading.Thread(target=worker, args=("B", 0.0))
        second.start()
        first.join(timeout=5)
        second.join(timeout=5)

        assert order == ["A-vào", "A-ra", "B-vào", "B-ra"]

    @pytest.mark.parametrize(("concurrent", "pages"), [(0, 10), (1, 0), (-1, 10)])
    def test_nonsense_limits_are_rejected_at_construction(
        self, concurrent: int, pages: int
    ) -> None:
        with pytest.raises(ValueError):
            OcrGate(max_concurrent=concurrent, max_pages=pages)
