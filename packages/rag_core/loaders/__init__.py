"""Chọn loader theo phần mở rộng, và quyết định có OCR hay không. `W3-01` · `W3-02`.

```python
from rag_core.loaders import load_document
doc = load_document("bao-cao.pdf", language="vi")
doc.text            # markdown
doc.headings        # đã chuẩn hoá độ sâu giữa các định dạng
doc.as_metadata()   # vân tay parse, để đi vào DocumentMetadata.extra
```

**Bảng định tuyến, và vì sao `.txt` nằm riêng.**

| đuôi | loader | vì sao |
|---|---|---|
| `.txt` | `plain` | hàm đồng nhất — cả corpus hiện tại nằm ở đây |
| `.md` `.markdown` `.html` `.htm` `.pdf` `.docx` `.pptx` `.xlsx` | `docling` | cần parse thật |

Ranh giới đó **không** phải để tiết kiệm: nó là điều kiện để mọi con số của
`W2` còn giá trị. 60 tài liệu corpus đều là `.txt`, mọi `TextSpan` của
`golden_v1` neo vào văn bản decode thẳng từ byte, nên `.txt` mà đi qua bất kỳ bộ
parse nào cũng làm toàn bộ nhãn lệch. Đo được: cho 60 file ấy qua backend
markdown của docling thì **0/60** giữ nguyên nội dung. Chi tiết ở
`plans/reports/tasks/w3-01-docling-loader.md`.

⚠️ Loader này **chưa** được nối vào `pipeline/indexing/corpus_loader.py`. Nối
vào là việc của `W3-07` (incremental re-index) và nó cần `TD-22` trước — xem
`base.py` về chỗ chuỗi toàn vẹn bị hở khi có parser đứng giữa.

**Đường OCR (`W3-02`)** đi qua ba cửa, theo đúng thứ tự rẻ → đắt:

1. `scan.detect_scan` — đọc text layer bằng pypdfium2, không dựng ảnh, không
   model. Ngưỡng đo từ hai báo cáo World Bank thật, xem `scan.py`.
2. `ocr.require_ocr_support` — **từ chối** nếu ngôn ngữ nằm ngoài tập đã đo.
   Máy OCR hiện có đọc tiếng Anh và trả rác cho tiếng Việt, nên với corpus của
   dự án này thì bật OCR bừa còn tệ hơn không bật.
3. `ocr.OcrGate` — trần song song (VRAM) và trần số trang (~12 s/trang).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import replace
from pathlib import Path
from typing import Literal

from .base import (
    Heading,
    LoadedDocument,
    LoaderError,
    ParseFingerprint,
    UnsupportedFormatError,
    detect_format,
)
from .docling_backend import DOCLING_FORMATS, docling_version, load_with_docling
from .ocr import (
    DEFAULT_GATE,
    OCR_ENGINE,
    OCR_ENGINES,
    OCR_VERIFIED_LANGUAGES,
    OcrBudgetError,
    OcrGate,
    OcrLanguageError,
    engine_for,
    ocr_supports,
    require_ocr_support,
)
from .plain import PLAIN_FORMATS, load_plain
from .scan import MIN_CHARS_PER_IN2, SCAN_PAGE_RATIO, PageText, ScanReport, detect_scan, page_texts

__all__ = [
    "DEFAULT_GATE",
    "DOCLING_FORMATS",
    "MIN_CHARS_PER_IN2",
    "OCR_ENGINE",
    "OCR_ENGINES",
    "OCR_VERIFIED_LANGUAGES",
    "PLAIN_FORMATS",
    "SCAN_PAGE_RATIO",
    "SUPPORTED_FORMATS",
    "Heading",
    "LoadedDocument",
    "LoaderError",
    "OcrBudgetError",
    "OcrGate",
    "OcrLanguageError",
    "OcrMode",
    "PageText",
    "ParseFingerprint",
    "ScanReport",
    "UnsupportedFormatError",
    "detect_format",
    "detect_scan",
    "docling_version",
    "engine_for",
    "load_document",
    "loader_for",
    "ocr_supports",
    "page_texts",
    "require_ocr_support",
]

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = frozenset(PLAIN_FORMATS | DOCLING_FORMATS)


def loader_for(path: str | Path) -> str:
    """`"plain"` hoặc `"docling"`. Ném `UnsupportedFormatError` nếu không nhận."""
    suffix = detect_format(path)
    if suffix in PLAIN_FORMATS:
        return "plain"
    if suffix in DOCLING_FORMATS:
        return "docling"
    raise UnsupportedFormatError(
        f"{Path(path).name}: chưa có loader cho `{suffix or 'không có đuôi'}`. "
        f"Đang hỗ trợ: {', '.join(sorted(SUPPORTED_FORMATS))}"
    )


OcrMode = Literal["off", "auto", "force"]


def load_document(
    path: str | Path,
    *,
    ocr: OcrMode | bool = "auto",
    language: str | None = None,
    gate: OcrGate | None = None,
) -> LoadedDocument:
    """Đọc một file bất kỳ trong các định dạng được hỗ trợ.

    `source_sha256` luôn tính từ **byte trên đĩa**, kể cả với docling — đó là
    nửa còn lại của vân tay: byte vào là gì, và parse bằng gì.

    `ocr`:

    * `"off"` — không bao giờ OCR. Đây là hành vi của `W3-01`.
    * `"auto"` (mặc định) — đo mật độ text layer trước (`scan.detect_scan`, rẻ,
      không dựng ảnh); chỉ bật OCR khi tài liệu thật sự thiếu text.
    * `"force"` — OCR bất kể phát hiện nói gì.

    ⚠️ `language` **không** phải gợi ý mà là một cái chốt chọn máy: `"en"` đi
    RapidOCR (mặc định docling), `"vi"` đi EasyOCR — máy duy nhất đã ĐO là giữ
    được dấu tiếng Việt (2026-09-04, sau khi phát hiện phép đo cũ chấm trên
    fixture hỏng font). Ngôn ngữ chưa đo thì hàm này **từ chối** thay vì trả
    rác. Bỏ trống thì chạy máy mặc định kèm cảnh báo — xem `ocr.require_ocr_support`.
    """
    target = Path(path)
    if not target.is_file():
        raise LoaderError(f"Không thấy file: {target}")

    kind = loader_for(target)
    source_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
    if kind == "plain":
        return load_plain(target, source_sha256=source_sha256)

    mode: OcrMode = ("force" if ocr else "off") if isinstance(ocr, bool) else ocr
    if mode not in ("off", "auto", "force"):
        raise ValueError(f"ocr phải là 'off' | 'auto' | 'force', nhận {ocr!r}")

    # Chỉ PDF mới có khái niệm "trang không có text layer". DOCX/PPTX/XLSX/HTML/MD
    # mang text theo cấu trúc, không có gì để OCR.
    report: ScanReport | None = None
    if mode == "auto" and detect_format(target) == ".pdf":
        report = detect_scan(target)
        logger.debug("%s: %s", target.name, report.summary())

    use_ocr = mode == "force" or (report is not None and report.needs_ocr)
    if not use_ocr:
        return load_with_docling(target, source_sha256=source_sha256, ocr_engine=None)

    engine = require_ocr_support(language, name=target.name)
    pages = len(report.pages) if report is not None else len(page_texts(target))
    with (gate or DEFAULT_GATE).reserve(pages, name=target.name):
        document = load_with_docling(target, source_sha256=source_sha256, ocr_engine=engine)

    extra = dict(document.extra)
    extra.update(
        {
            "ocr": "true",
            "ocr_engine": OCR_ENGINES[engine],
            "ocr_language": language or "unknown",
            "ocr_pages": str(pages),
        }
    )
    return replace(document, extra=extra)
