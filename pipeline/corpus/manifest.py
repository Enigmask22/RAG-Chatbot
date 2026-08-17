"""Manifest corpus — sổ đăng ký mọi tài liệu, kèm nguồn và giấy phép.

Đây là chỗ **quy tắc cứng #3 được thực thi bằng code**, không phải bằng lời hứa:
corpus phải công khai và giấy phép phải cho phép redistribute, vì repo sẽ public,
demo HF Spaces public, và một phần xử lý chạy trên máy thuê bên thứ ba — ba kênh
công bố dữ liệu. Không entry nào vào được manifest nếu thiếu `source_url` hoặc
mang giấy phép ngoài danh sách cho phép.

Một điểm dễ bỏ sót: **giấy phép có `ND` (NoDerivatives) bị từ chối.** Pipeline này
cắt tài liệu thành chunk, sinh context bằng LLM (`W3-04`) rồi ghép vào câu trả
lời — đó là tác phẩm phái sinh theo đúng nghĩa của điều khoản. `ND` cho phép
phát tán nguyên bản chứ không cho phép làm việc đó.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_core.schemas import DocType, Language

__all__ = [
    "LICENSE_ALLOWLIST",
    "CorpusEntry",
    "load_manifest",
    "validate_manifest",
    "write_manifest",
]

LICENSE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "CC BY 4.0",
        "CC BY 3.0",
        "CC BY 3.0 IGO",
        "CC BY-SA 4.0",
        "CC BY-NC 4.0",
        "CC BY-NC-SA 4.0",
        "CC0 1.0",
        "Public Domain",
        "OGL v3",
        "Vietnam Government Work",
    }
)
"""Giấy phép cho phép redistribute **và** cho phép tạo tác phẩm phái sinh.

`CC BY-NC*` được chấp nhận vì dự án phi thương mại; nếu sau này đem đi thương mại
hoá thì phải rà lại danh sách này trước.
"""

_FIELDS = (
    "doc_id",
    "relative_path",
    "source_url",
    "landing_url",
    "license",
    "license_url",
    "title",
    "lang",
    "doc_type",
    "source",
    "published_at",
    "sha256",
    "bytes",
    "fetched_at",
    "notes",
)


class CorpusEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    license: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)
    bytes: int = Field(ge=1)
    source: str = Field(min_length=1)
    title: str = ""
    landing_url: str = ""
    license_url: str = ""
    lang: Language = Language.UNKNOWN
    doc_type: DocType = DocType.OTHER
    published_at: str = ""
    fetched_at: str = ""
    notes: str = ""

    @field_validator("license")
    @classmethod
    def _license_must_be_allowed(cls, value: str) -> str:
        if value not in LICENSE_ALLOWLIST:
            raise ValueError(
                f"Giấy phép {value!r} không nằm trong danh sách cho phép. "
                f"Chỉ nhận: {sorted(LICENSE_ALLOWLIST)}. "
                "Giấy phép có ND (NoDerivatives) bị từ chối vì chunking + sinh context "
                "bằng LLM là tạo tác phẩm phái sinh."
            )
        return value

    @field_validator("source_url", "landing_url")
    @classmethod
    def _url_must_be_http(cls, value: str) -> str:
        if value and not value.startswith(("http://", "https://")):
            raise ValueError(f"URL phải bắt đầu bằng http(s): {value!r}")
        return value

    @classmethod
    def now_iso(cls) -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")


def validate_manifest(entries: Sequence[CorpusEntry]) -> None:
    """Kiểm tra ràng buộc ở mức tập hợp (pydantic chỉ lo từng dòng)."""
    seen_ids: dict[str, int] = {}
    seen_hashes: dict[str, str] = {}
    for index, entry in enumerate(entries):
        if entry.doc_id in seen_ids:
            raise ValueError(
                f"doc_id trùng {entry.doc_id!r} ở dòng {index + 1} "
                f"và dòng {seen_ids[entry.doc_id] + 1}"
            )
        seen_ids[entry.doc_id] = index
        if entry.sha256 in seen_hashes:
            # Cùng nội dung dưới hai doc_id khác nhau sẽ làm golden set có hai
            # chunk_id "đúng" cho cùng một đoạn văn — recall bị tính sai.
            raise ValueError(
                f"Nội dung trùng: {entry.doc_id!r} giống hệt "
                f"{seen_hashes[entry.sha256]!r} (sha256 {entry.sha256[:12]})"
            )
        seen_hashes[entry.sha256] = entry.doc_id


def write_manifest(path: str | Path, entries: Iterable[CorpusEntry]) -> int:
    rows = list(entries)
    validate_manifest(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_FIELDS))
        writer.writeheader()
        for entry in rows:
            record = entry.model_dump(mode="json")
            writer.writerow({field: record[field] for field in _FIELDS})
    return len(rows)


def load_manifest(path: str | Path) -> list[CorpusEntry]:
    """Đọc manifest. Dòng nào sai thì báo kèm số dòng trong file."""
    source = Path(path)
    if not source.exists():
        return []
    entries: list[CorpusEntry] = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        for line_no, row in enumerate(csv.DictReader(handle), start=2):
            try:
                entries.append(CorpusEntry.model_validate(row))
            except Exception as exc:
                raise ValueError(f"{source}:{line_no} — {exc}") from exc
    validate_manifest(entries)
    return entries
