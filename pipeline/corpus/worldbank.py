"""Adapter cho World Bank Documents & Reports API (WDS v3).

API công khai, không cần key: `https://search.worldbank.org/api/v3/wds`.

Hai điều đáng lưu ý khi dùng:

* **Không truyền tham số `fl`.** Nó có vẻ để chọn field trả về, nhưng thực tế lại
  làm rụng mất `pdfurl`/`txturl` — đúng hai field cần nhất. Cứ lấy đủ rồi lọc ở
  phía mình.
* **Ưu tiên `txturl` hơn `pdfurl`.** Bản `.txt` là text đã trích sẵn, dùng được
  ngay ở W1. PDF phải đợi Docling ở `W3-01`. Lấy text trước nghĩa là baseline
  `W1-13` không bị chặn bởi chất lượng bộ trích xuất PDF.

ADB **không** có adapter ở đây: `adb.org` trả 403 cho mọi truy cập tự động. Tài
liệu ADB phải tải tay rồi đưa vào bằng nguồn `seed_list` (xem `scripts/fetch_corpus.py`).
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from rag_core.schemas import Language

__all__ = ["WdsDocument", "search_wds"]

logger = logging.getLogger(__name__)

WDS_ENDPOINT = "https://search.worldbank.org/api/v3/wds"
USER_AGENT = "rag-platform-corpus-fetcher/0.1 (+research; contact via repo)"

_LANG_MAP = {
    "english": Language.EN,
    "vietnamese": Language.VI,
}


@dataclass(frozen=True)
class WdsDocument:
    guid: str
    title: str
    doc_type: str
    major_doc_type: str
    language: Language
    published_at: str
    landing_url: str
    text_url: str
    pdf_url: str
    disclosure_status: str

    @property
    def download_url(self) -> str:
        """Ưu tiên bản text; chỉ rơi về PDF khi không có text."""
        return self.text_url or self.pdf_url

    @property
    def extension(self) -> str:
        return ".txt" if self.text_url else ".pdf"

    @property
    def is_public(self) -> bool:
        # Chỉ nhận tài liệu đã công bố công khai. WDS có cả bản ghi hạn chế.
        return self.disclosure_status.strip().lower() in {"disclosed", "public", ""}


def _first_title(raw: dict[str, Any]) -> str:
    display = raw.get("display_title")
    if isinstance(display, str) and display.strip():
        return display.strip()
    docna = raw.get("docna")
    if isinstance(docna, dict):
        for value in docna.values():
            if isinstance(value, dict) and value.get("docna"):
                return str(value["docna"]).strip()
    return ""


def _parse_document(raw: dict[str, Any]) -> WdsDocument | None:
    guid = str(raw.get("guid") or raw.get("id") or "").strip()
    if not guid:
        return None
    language = _LANG_MAP.get(str(raw.get("lang", "")).strip().lower(), Language.UNKNOWN)
    return WdsDocument(
        guid=guid,
        title=_first_title(raw),
        doc_type=str(raw.get("docty", "")).strip(),
        major_doc_type=str(raw.get("majdocty", "")).strip(),
        language=language,
        published_at=str(raw.get("docdt", "")).strip(),
        landing_url=str(raw.get("url", "")).strip(),
        text_url=str(raw.get("txturl", "")).strip(),
        pdf_url=str(raw.get("pdfurl", "")).strip(),
        disclosure_status=str(raw.get("disclstat", "")).strip(),
    )


def _fetch_page(params: dict[str, str], timeout: float) -> dict[str, Any]:
    url = f"{WDS_ENDPOINT}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    return payload


def search_wds(
    qterm: str,
    *,
    language: str | None = None,
    rows_per_page: int = 20,
    max_pages: int = 10,
    timeout: float = 45.0,
    extra_params: dict[str, str] | None = None,
) -> Iterator[WdsDocument]:
    """Duyệt kết quả tìm kiếm theo trang, trả từng tài liệu đã parse.

    Là generator để người gọi dừng ngay khi đủ số lượng — không tải thừa trang.
    """
    offset = 0
    for page in range(max_pages):
        params = {
            "format": "json",
            "qterm": qterm,
            "rows": str(rows_per_page),
            "os": str(offset),
        }
        if language:
            params["lang_exact"] = language
        params.update(extra_params or {})

        payload = _fetch_page(params, timeout)
        documents = payload.get("documents", {})
        if not isinstance(documents, dict):
            return

        found = 0
        for key, raw in documents.items():
            if key == "facets" or not isinstance(raw, dict):
                continue
            found += 1
            parsed = _parse_document(raw)
            if parsed is not None:
                yield parsed

        if found == 0:
            logger.info("Hết kết quả ở trang %d cho truy vấn %r", page + 1, qterm)
            return
        offset += rows_per_page
