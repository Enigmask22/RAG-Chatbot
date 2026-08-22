"""Phát hiện trang scan bằng mật độ text layer. `W3-02`.

```
python -m rag_core.loaders.scan bao-cao.pdf   # in bảng mật độ từng trang
```

**Vì sao cần.** Không phải vì chi phí — xem `ocr.SECONDS_PER_PAGE`, con số "hai
bậc độ lớn" của `W3-01` hoá ra là chi phí **nạp model một lần**, còn giá mỗi
trang chỉ gấp ~3 lần. Lý do thật là **đúng/sai**: tài liệu ảnh mà không phát
hiện thì docling trả về rỗng và loader ném lỗi, tức tài liệu **biến mất khỏi
index** thay vì được OCR. Và phát hiện là chỗ duy nhất chặn được việc bật OCR
cho tiếng Việt (`ocr.py`). pypdfium2 (đã có sẵn theo docling) đọc text layer
trực tiếp, không dựng ảnh, không chạy model.

**Hai ngưỡng, và cả hai đo từ PDF thật.** Bài học `W3-01`: fixture tự sinh nằm
ngoài phân bố của tài liệu thật, nên hiệu chỉnh ngưỡng trên nó là đo chính cái
generator của mình. Ở đây tải hai báo cáo World Bank thật (CC BY, cùng nguồn với
corpus — 129 và 112 trang) rồi đo mật độ **ký tự trên inch²**:

| tài liệu | trang | min | p05 | p50 | p95 | max | trang < 1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `wb1.pdf` (thật) | 129 | 0,65 | 12,61 | 31,16 | 45,89 | 54,04 | **1** |
| `wb2.pdf` (thật) | 112 | 0,00 | 4,33 | 34,10 | 46,59 | 86,71 | **4** |
| `two-column.pdf` (fixture) | 1 | — | — | **8,17** | — | — | 0 |
| `scanned-page.pdf` (fixture) | 1 | — | — | **0,00** | — | — | 1 |

Hai điều rút ra, và điều thứ hai mới là điều quan trọng:

1. `MIN_CHARS_PER_IN2 = 1.0` — trang có text layer thật thấp nhất trong mẫu là
   **3,03** (bìa), còn trang không có text layer là **0,00–0,96**. Khoảng trống
   giữa hai bên rộng, ngưỡng 1,0 nằm giữa. ⚠️ Fixture born-digital của tôi ở
   **8,17**, tức **thấp hơn p05 của tài liệu thật** — hiệu chỉnh trên nó thì
   ngưỡng đã đặt sai chỗ.
2. ⭐ **Báo cáo born-digital THẬT vẫn có trang trống**: 1/129 và 4/112 (bìa, ảnh
   tràn trang, trang phân cách). Nên luật *"có một trang thiếu text layer ⇒ tài
   liệu là scan"* sẽ đẩy **100%** báo cáo World Bank vào OCR. Quyết định phải
   theo **tỉ lệ**: tài liệu thật cho **0,8%** và **3,6%**, scan thuần cho
   **100%**. `SCAN_PAGE_RATIO = 0.5` nằm giữa hai cụm cách nhau rất xa.

⚠️ Cái luật tỉ lệ đánh đổi đi: một báo cáo 129 trang có đúng 1 trang scan mang
nội dung thật sẽ **không** được OCR. Chấp nhận có ý thức — `ScanReport.pages`
giữ nguyên danh sách trang thưa để tầng trên quyết định khác nếu muốn.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from .base import LoaderError

__all__ = [
    "MIN_CHARS_PER_IN2",
    "SCAN_PAGE_RATIO",
    "PageText",
    "ScanReport",
    "detect_scan",
    "page_texts",
]

logger = logging.getLogger(__name__)

MIN_CHARS_PER_IN2 = 1.0
"""Dưới mức này coi như trang **không có text layer**. Xem bảng ở docstring."""

SCAN_PAGE_RATIO = 0.5
"""Tỉ lệ trang thiếu text layer để coi cả tài liệu là scan."""

_POINTS_PER_IN2 = 72.0 * 72.0


@dataclass(frozen=True, slots=True)
class PageText:
    """Một trang, và lượng text có sẵn trong đó."""

    number: int
    """1-based, để khớp với số trang người đọc thấy."""

    characters: int
    width_pt: float
    height_pt: float

    @property
    def area_in2(self) -> float:
        return self.width_pt * self.height_pt / _POINTS_PER_IN2

    @property
    def density(self) -> float:
        """Ký tự trên inch².

        Chuẩn hoá theo **diện tích** chứ không dùng số ký tự tuyệt đối: một slide
        16:9 và một trang A4 khác diện tích gần ba lần, nên cùng một ngưỡng đếm
        thô sẽ nghiêm với cái này và lỏng với cái kia.
        """
        area = self.area_in2
        return self.characters / area if area > 0 else 0.0

    def is_empty(self, threshold: float = MIN_CHARS_PER_IN2) -> bool:
        return self.density < threshold


@dataclass(frozen=True, slots=True)
class ScanReport:
    pages: tuple[PageText, ...]
    char_threshold: float = MIN_CHARS_PER_IN2
    ratio_threshold: float = SCAN_PAGE_RATIO

    @property
    def empty_pages(self) -> tuple[int, ...]:
        return tuple(p.number for p in self.pages if p.is_empty(self.char_threshold))

    @property
    def empty_ratio(self) -> float:
        return len(self.empty_pages) / len(self.pages) if self.pages else 0.0

    @property
    def needs_ocr(self) -> bool:
        return bool(self.pages) and self.empty_ratio >= self.ratio_threshold

    @property
    def median_density(self) -> float:
        if not self.pages:
            return 0.0
        values = sorted(p.density for p in self.pages)
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return (values[middle - 1] + values[middle]) / 2

    def summary(self) -> str:
        if not self.pages:
            return "0 trang"
        verdict = "CẦN OCR" if self.needs_ocr else "có text layer"
        return (
            f"{len(self.pages)} trang · {len(self.empty_pages)} trang thiếu text "
            f"({self.empty_ratio:.1%}) · mật độ p50 {self.median_density:.2f} ký tự/in² "
            f"→ {verdict}"
        )


def page_texts(path: str | Path) -> tuple[PageText, ...]:
    """Đọc text layer từng trang. Không dựng ảnh, không chạy model."""
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
        raise LoaderError(
            "Thiếu pypdfium2. Cài extra `ingestion`: `uv sync --extra ingestion`."
        ) from exc

    target = Path(path)
    if not target.is_file():
        raise LoaderError(f"Không thấy file: {target}")

    document = pdfium.PdfDocument(str(target))
    try:
        pages: list[PageText] = []
        for index, page in enumerate(document, start=1):
            width, height = page.get_size()
            text = page.get_textpage().get_text_bounded()
            pages.append(
                PageText(
                    number=index,
                    characters=len(text.strip()),
                    width_pt=float(width),
                    height_pt=float(height),
                )
            )
    finally:
        document.close()
    return tuple(pages)


def detect_scan(
    path: str | Path,
    *,
    char_threshold: float = MIN_CHARS_PER_IN2,
    ratio_threshold: float = SCAN_PAGE_RATIO,
) -> ScanReport:
    report = ScanReport(
        pages=page_texts(path),
        char_threshold=char_threshold,
        ratio_threshold=ratio_threshold,
    )
    logger.debug("%s: %s", Path(path).name, report.summary())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="In mật độ text layer từng trang.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--per-page", action="store_true", help="in từng trang, không chỉ tổng kết")
    parser.add_argument("--char-threshold", type=float, default=MIN_CHARS_PER_IN2)
    parser.add_argument("--ratio-threshold", type=float, default=SCAN_PAGE_RATIO)
    args = parser.parse_args(argv)

    for path in args.paths:
        report = detect_scan(
            path,
            char_threshold=args.char_threshold,
            ratio_threshold=args.ratio_threshold,
        )
        print(f"{path.name}: {report.summary()}")
        if args.per_page:
            for page in report.pages:
                flag = "  ← thiếu text" if page.is_empty(args.char_threshold) else ""
                print(
                    f"    trang {page.number:4d}  {page.density:8.2f} ký tự/in²  "
                    f"({page.characters:6d} ký tự / {page.area_in2:.1f} in²){flag}"
                )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
