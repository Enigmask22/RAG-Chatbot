"""Đường `.txt` — **hàm đồng nhất**, và đó là toàn bộ giá trị của nó.

Cả corpus hiện tại (60 tài liệu World Bank) là `.txt`, và mọi con số của `W2`
cùng mọi `TextSpan` của `golden_v1` neo vào `Document.content` sinh ra từ
`payload.decode("utf-8")`. Loader này giữ nguyên đúng phép biến đổi ấy, không
strip, không normalise newline, không normalise unicode. Một ký tự đổi là một
offset lệch, và span lệch thì recall sai mà không có gì đỏ.

Không phải chuyện thận trọng thừa: docling **có** đổi ký tự. Cùng một dấu gạch
dài `—` trong cùng một tài liệu, backend `.md` giữ nguyên còn backend `.html`
trả về `-`. Nếu `.txt` đi qua docling thì không đoán được nó đổi những gì.

Tiện là docling cũng không nhận `.txt` — `InputFormat` không có định dạng đó —
nên ranh giới này được chính thư viện ép chứ không chỉ do quy ước.
"""

from __future__ import annotations

from pathlib import Path

from .base import LoadedDocument, LoaderError, ParseFingerprint

__all__ = ["PLAIN_FORMATS", "load_plain"]

PLAIN_FORMATS = frozenset({".txt"})

_FINGERPRINT = ParseFingerprint(
    loader="plain",
    library="stdlib",
    # Không phải version của Python: phép biến đổi là `bytes.decode("utf-8")`,
    # không có tham số nào để đổi và không phụ thuộc thư viện nào. Nâng version
    # ở đây chỉ khi chính hành vi đọc đổi — và lúc đó là cố ý.
    library_version="1",
    options=("encoding=utf-8",),
)


def load_plain(path: str | Path, *, source_sha256: str) -> LoadedDocument:
    payload = Path(path).read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LoaderError(f"{Path(path).name}: không decode được bằng utf-8 ({exc})") from exc
    if not text.strip():
        raise LoaderError(f"{Path(path).name}: file rỗng sau khi decode")
    return LoadedDocument(text=text, source_sha256=source_sha256, fingerprint=_FINGERPRINT)
