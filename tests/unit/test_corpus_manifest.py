"""W0-03 — manifest corpus: ép nguồn công khai + giấy phép cho phép redistribute."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pipeline.corpus.manifest import (
    LICENSE_ALLOWLIST,
    CorpusEntry,
    load_manifest,
    validate_manifest,
    write_manifest,
)
from rag_core.schemas import DocType, Language


def _entry(**overrides: object) -> CorpusEntry:
    payload: dict[str, object] = {
        "doc_id": "wb-123456",
        "relative_path": "bao-cao-123456.txt",
        "source_url": "https://documents.worldbank.org/curated/en/123456/text/doc.txt",
        "license": "CC BY 3.0 IGO",
        "sha256": "a" * 64,
        "bytes": 12345,
        "source": "worldbank_wds",
        "title": "Vietnam Development Report",
        "lang": Language.EN,
        "doc_type": DocType.DEV_REPORT,
    }
    payload.update(overrides)
    return CorpusEntry.model_validate(payload)


class TestRequiredProvenance:
    def test_rejects_missing_source_url(self) -> None:
        with pytest.raises(ValidationError):
            _entry(source_url="")

    def test_rejects_missing_license(self) -> None:
        with pytest.raises(ValidationError):
            _entry(license="")

    def test_rejects_non_http_source(self) -> None:
        with pytest.raises(ValidationError, match="http"):
            _entry(source_url="C:/Users/LENOVO/Documents/tai-lieu-khach-hang.pdf")

    def test_rejects_extra_field(self) -> None:
        with pytest.raises(ValidationError):
            _entry(internal_only="khách hàng Enigmas")


class TestLicenseGate:
    """Quy tắc cứng #3 phải được thực thi bằng code, không bằng lời hứa."""

    def test_rejects_unlisted_license(self) -> None:
        with pytest.raises(ValidationError, match="không nằm trong danh sách"):
            _entry(license="All rights reserved")

    def test_rejects_no_derivatives_license(self) -> None:
        # Chunking + sinh context bằng LLM (W3-04) là tạo tác phẩm phái sinh,
        # nên ND không dùng được dù nó cho phép phát tán nguyên bản.
        assert not any("ND" in lic.split() or "-ND" in lic for lic in LICENSE_ALLOWLIST)
        with pytest.raises(ValidationError):
            _entry(license="CC BY-ND 4.0")

    @pytest.mark.parametrize("license_name", sorted(LICENSE_ALLOWLIST))
    def test_accepts_allowlisted(self, license_name: str) -> None:
        assert _entry(license=license_name).license == license_name


class TestSetLevelValidation:
    def test_rejects_duplicate_doc_id(self) -> None:
        with pytest.raises(ValueError, match="doc_id trùng"):
            validate_manifest([_entry(), _entry(sha256="b" * 64)])

    def test_rejects_duplicate_content(self) -> None:
        """Cùng nội dung dưới hai doc_id làm golden set có hai `chunk_id` đúng
        cho một đoạn văn — recall bị tính sai mà không có dấu hiệu nào."""
        with pytest.raises(ValueError, match="Nội dung trùng"):
            validate_manifest([_entry(), _entry(doc_id="wb-999", relative_path="khac.txt")])

    def test_accepts_distinct_entries(self) -> None:
        validate_manifest(
            [_entry(), _entry(doc_id="wb-999", relative_path="khac.txt", sha256="b" * 64)]
        )


class TestCsvRoundTrip:
    def test_round_trip(self, tmp_path: Path) -> None:
        entries = [
            _entry(),
            _entry(doc_id="wb-999", relative_path="b.txt", sha256="b" * 64, lang=Language.VI),
        ]
        path = tmp_path / "corpus_manifest.csv"
        assert write_manifest(path, entries) == 2
        assert load_manifest(path) == entries

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_manifest(tmp_path / "chua-co.csv") == []

    def test_bad_row_reports_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "m.csv"
        write_manifest(path, [_entry()])
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("CC BY 3.0 IGO", "All rights reserved"), encoding="utf-8")
        with pytest.raises(ValueError, match=r":2 —"):
            load_manifest(path)

    def test_write_refuses_invalid_set(self, tmp_path: Path) -> None:
        # Manifest hỏng không được ghi ra đĩa nửa vời.
        with pytest.raises(ValueError):
            write_manifest(tmp_path / "m.csv", [_entry(), _entry()])
        assert not (tmp_path / "m.csv").exists()
