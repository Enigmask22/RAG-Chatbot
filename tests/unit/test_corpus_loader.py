"""W1-08 — manifest → `Document`, có kiểm tra toàn vẹn.

Bài test quan trọng nhất ở đây là `TestIntegrity`: file trên đĩa lệch manifest
phải làm cả job dừng. Lý do không phải là sạch sẽ — `chunk_id` được sinh theo
`{doc_id}::{index}`, nên nội dung đổi khiến golden set trỏ sang đoạn văn khác
mà mọi thứ vẫn "chạy được", chỉ có recall là sai và không ai biết vì sao.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from pipeline.corpus.manifest import CorpusEntry, write_manifest
from pipeline.indexing.corpus_loader import (
    CorpusIntegrityError,
    load_documents,
    select_entries,
)
from rag_core.schemas import DocType, Language


def _write(path: Path, text: str) -> None:
    """Ghi bằng bytes chứ không dùng `write_text`.

    Trên Windows `write_text` tự đổi xuống dòng sang CRLF, làm sha256 lệch
    manifest. `scripts/fetch_corpus.py` ghi bytes thô nên test phải làm y hệt.
    """
    path.write_bytes(text.encode("utf-8"))


def _entry(doc_id: str, text: str, **kwargs: object) -> tuple[CorpusEntry, str]:
    payload = text.encode("utf-8")
    fields: dict[str, object] = {
        "doc_id": doc_id,
        "relative_path": f"{doc_id}.txt",
        "source_url": f"https://example.org/{doc_id}",
        "license": "CC BY 4.0",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "source": "test",
        "lang": "vi",
        "doc_type": "dev_report",
    }
    fields.update(kwargs)
    return CorpusEntry.model_validate(fields), text


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Corpus 3 tài liệu: 2 tiếng Việt (1 legal), 1 tiếng Anh."""
    specs = [
        _entry("d-b", "Nội dung tài liệu B về ngân sách nhà nước.", lang="vi"),
        _entry("d-a", "Document A about public investment.", lang="en"),
        _entry("d-c", "Điều 1. Phạm vi điều chỉnh của văn bản.", lang="vi", doc_type="legal"),
    ]
    for entry, text in specs:
        _write(tmp_path / entry.relative_path, text)
    write_manifest(tmp_path / "manifest.csv", [entry for entry, _ in specs])
    return tmp_path


class TestSelect:
    def test_sorts_by_doc_id_so_slicing_is_reproducible(self) -> None:
        """`max_documents=N` phải luôn chọn đúng N tài liệu đó.

        Nếu cắt theo thứ tự dòng trong manifest thì thêm một tài liệu ở đầu file
        cũng làm lần chạy thử lấy tập khác — không so được với lần trước.
        """
        entries = [_entry(f"d-{c}", "x")[0] for c in "cab"]
        assert [e.doc_id for e in select_entries(entries, max_documents=2)] == ["d-a", "d-b"]

    def test_filters_by_language(self) -> None:
        entries = [
            _entry("d-1", "x", lang="vi")[0],
            _entry("d-2", "y", lang="en")[0],
        ]
        assert [e.doc_id for e in select_entries(entries, languages=["en"])] == ["d-2"]

    def test_filters_by_doc_type(self) -> None:
        entries = [
            _entry("d-1", "x", doc_type="legal")[0],
            _entry("d-2", "y", doc_type="dev_report")[0],
        ]
        assert [e.doc_id for e in select_entries(entries, doc_types=["legal"])] == ["d-1"]

    def test_empty_filter_keeps_everything(self) -> None:
        entries = [_entry("d-1", "x")[0], _entry("d-2", "y")[0]]
        assert len(select_entries(entries)) == 2


class TestLoad:
    def test_maps_manifest_fields_onto_metadata(self, corpus: Path) -> None:
        docs = load_documents(corpus / "manifest.csv", corpus, max_documents=1)
        doc = docs[0]
        assert doc.doc_id == "d-a"
        assert doc.metadata.lang is Language.EN
        assert doc.metadata.doc_type is DocType.DEV_REPORT
        assert doc.metadata.license == "CC BY 4.0"
        assert doc.metadata.source_url == "https://example.org/d-a"
        assert doc.metadata.source_path == "d-a.txt"

    def test_keeps_provenance_in_extra(self, corpus: Path) -> None:
        """Giữ lại `sha256` của manifest trong metadata để truy ngược được về sau."""
        doc = load_documents(corpus / "manifest.csv", corpus, max_documents=1)[0]
        assert doc.metadata.extra["corpus_source"] == "test"
        assert len(doc.metadata.extra["manifest_sha256"]) == 64

    def test_language_filter_applies(self, corpus: Path) -> None:
        docs = load_documents(corpus / "manifest.csv", corpus, languages=["vi"])
        assert {d.doc_id for d in docs} == {"d-b", "d-c"}

    def test_parses_published_at(self, tmp_path: Path) -> None:
        entry, text = _entry("d-1", "Nội dung.", published_at="2024-03-15")
        _write(tmp_path / entry.relative_path, text)
        write_manifest(tmp_path / "m.csv", [entry])
        doc = load_documents(tmp_path / "m.csv", tmp_path)[0]
        assert doc.metadata.published_at is not None
        assert doc.metadata.published_at.year == 2024
        assert doc.metadata.published_at.tzinfo is not None

    def test_survives_unparsable_published_at(self, tmp_path: Path) -> None:
        """Ngày hỏng làm mất một field metadata, không đáng để hỏng cả job index."""
        entry, text = _entry("d-1", "Nội dung.", published_at="không rõ")
        _write(tmp_path / entry.relative_path, text)
        write_manifest(tmp_path / "m.csv", [entry])
        assert load_documents(tmp_path / "m.csv", tmp_path)[0].metadata.published_at is None


class TestIntegrity:
    def test_rejects_content_that_drifted_from_manifest(self, corpus: Path) -> None:
        _write(corpus / "d-a.txt", "Nội dung đã bị sửa sau khi tải.")
        with pytest.raises(CorpusIntegrityError, match="khác manifest"):
            load_documents(corpus / "manifest.csv", corpus)

    def test_can_be_disabled_explicitly(self, corpus: Path) -> None:
        _write(corpus / "d-a.txt", "Nội dung đã bị sửa.")
        docs = load_documents(corpus / "manifest.csv", corpus, verify_hash=False)
        assert len(docs) == 3

    def test_reports_missing_file_with_fix(self, corpus: Path) -> None:
        (corpus / "d-a.txt").unlink()
        with pytest.raises(CorpusIntegrityError, match="fetch_corpus"):
            load_documents(corpus / "manifest.csv", corpus)

    def test_rejects_empty_manifest(self, tmp_path: Path) -> None:
        with pytest.raises(CorpusIntegrityError, match="fetch_corpus"):
            load_documents(tmp_path / "khong-co.csv", tmp_path)

    def test_reports_filter_that_matches_nothing(self, corpus: Path) -> None:
        with pytest.raises(CorpusIntegrityError, match="Bộ lọc"):
            load_documents(corpus / "manifest.csv", corpus, languages=["fr"])

    def test_rejects_whitespace_only_document(self, tmp_path: Path) -> None:
        entry, _ = _entry("d-1", "   \n  \n")
        _write(tmp_path / entry.relative_path, "   \n  \n")
        write_manifest(tmp_path / "m.csv", [entry])
        with pytest.raises(CorpusIntegrityError, match="rỗng"):
            load_documents(tmp_path / "m.csv", tmp_path)


class TestRealManifest:
    """Manifest thật trong repo phải đọc được — nó là đầu vào của `make index`."""

    def test_repo_manifest_parses(self) -> None:
        manifest = Path(__file__).resolve().parents[2] / "data" / "corpus_manifest.csv"
        if not manifest.exists():
            pytest.skip("chưa chạy scripts/fetch_corpus.py")
        with manifest.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows, "manifest rỗng"
        assert all(row["license"] for row in rows), "có dòng thiếu giấy phép"
