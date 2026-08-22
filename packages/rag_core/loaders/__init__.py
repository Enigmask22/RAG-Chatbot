"""Chọn loader theo phần mở rộng. `W3-01`.

```python
from rag_core.loaders import load_document
doc = load_document("bao-cao.pdf")
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
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .base import (
    Heading,
    LoadedDocument,
    LoaderError,
    ParseFingerprint,
    UnsupportedFormatError,
    detect_format,
)
from .docling_backend import DOCLING_FORMATS, docling_version, load_with_docling
from .plain import PLAIN_FORMATS, load_plain

__all__ = [
    "DOCLING_FORMATS",
    "PLAIN_FORMATS",
    "SUPPORTED_FORMATS",
    "Heading",
    "LoadedDocument",
    "LoaderError",
    "ParseFingerprint",
    "UnsupportedFormatError",
    "detect_format",
    "docling_version",
    "load_document",
    "loader_for",
]

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


def load_document(path: str | Path, *, ocr: bool = False) -> LoadedDocument:
    """Đọc một file bất kỳ trong các định dạng được hỗ trợ.

    `source_sha256` luôn tính từ **byte trên đĩa**, kể cả với docling — đó là
    nửa còn lại của vân tay: byte vào là gì, và parse bằng gì.
    """
    target = Path(path)
    if not target.is_file():
        raise LoaderError(f"Không thấy file: {target}")

    kind = loader_for(target)
    source_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
    if kind == "plain":
        return load_plain(target, source_sha256=source_sha256)
    return load_with_docling(target, source_sha256=source_sha256, ocr=ocr)
