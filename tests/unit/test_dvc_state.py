"""Test cho `pipeline.corpus.dvc_state` — canh hai cơ chế versioning không lệch.

Phần lớn test ở đây dựng file `.dvc` và manifest giả trong tmp_path. Có đúng một
test chạm vào dữ liệu thật của repo (`test_repo_corpus_is_consistent`) và nó được
skip khi corpus chưa `dvc pull` về, để `make test` trên clone sạch vẫn xanh.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from pydantic import ValidationError

from pipeline.corpus.dvc_state import (
    DvcOut,
    DvcTrackingError,
    load_dvc_out,
    verify_corpus_tracking,
)

_MANIFEST_FIELDS = (
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


def _write_dvc(path: Path, *, nfiles: int, md5: str = "a" * 32 + ".dir", size: int = 1234) -> Path:
    """Ghi file `.dvc` tối thiểu nhưng đúng format DVC sinh ra."""
    path.write_text(
        f"outs:\n- md5: {md5}\n  size: {size}\n  nfiles: {nfiles}\n  hash: md5\n  path: corpus\n",
        encoding="utf-8",
    )
    return path


def _write_manifest(path: Path, relative_paths: list[str]) -> Path:
    """Ghi manifest CSV hợp lệ với đúng số dòng cần thiết."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        for i, rel in enumerate(relative_paths):
            writer.writerow(
                {
                    "doc_id": f"wb-{i:04d}",
                    "relative_path": rel,
                    "source_url": f"https://example.org/{i}.txt",
                    "landing_url": f"https://example.org/{i}",
                    "license": "CC BY 4.0",
                    "license_url": "https://creativecommons.org/licenses/by/4.0/",
                    "title": f"Tài liệu {i}",
                    "lang": "vi",
                    "doc_type": "dev_report",
                    "source": "test",
                    "published_at": "2026-01-01",
                    "sha256": f"{i:064d}",
                    "bytes": 10,
                    "fetched_at": "2026-01-01T00:00:00+00:00",
                    "notes": "",
                }
            )
    return path


def _make_corpus(root: Path, names: list[str], *, gitkeep: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(b"noi dung")
    if gitkeep:
        (root / ".gitkeep").write_bytes(b"")
    return root


class TestLoadDvcOut:
    def test_parses_real_dvc_format(self, tmp_path: Path) -> None:
        p = _write_dvc(tmp_path / "corpus.dvc", nfiles=61, md5="9aeb1b77" + "0" * 24 + ".dir")
        out = load_dvc_out(p)
        assert out.nfiles == 61
        assert out.path == "corpus"
        assert out.hash_md5.endswith(".dir")

    def test_accepts_unknown_keys(self, tmp_path: Path) -> None:
        """Phiên bản DVC mới thêm khoá; đó không phải lỗi cấu hình của dự án."""
        p = tmp_path / "corpus.dvc"
        p.write_text(
            "outs:\n"
            "- md5: abc.dir\n"
            "  size: 1\n"
            "  nfiles: 1\n"
            "  path: corpus\n"
            "  remote: local\n"
            "  push: true\n"
            "  files: []\n",
            encoding="utf-8",
        )
        assert load_dvc_out(p).nfiles == 1

    def test_missing_file_names_the_fix(self, tmp_path: Path) -> None:
        with pytest.raises(DvcTrackingError, match=r"dvc add data/corpus"):
            load_dvc_out(tmp_path / "khong-ton-tai.dvc")

    def test_rejects_invalid_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "corpus.dvc"
        p.write_text("outs:\n- md5: [unclosed\n", encoding="utf-8")
        with pytest.raises(DvcTrackingError, match="không phải YAML hợp lệ"):
            load_dvc_out(p)

    def test_rejects_non_mapping(self, tmp_path: Path) -> None:
        p = tmp_path / "corpus.dvc"
        p.write_text("- chi la mot list\n", encoding="utf-8")
        with pytest.raises(DvcTrackingError, match="phải là mapping YAML"):
            load_dvc_out(p)

    def test_rejects_missing_outs(self, tmp_path: Path) -> None:
        p = tmp_path / "corpus.dvc"
        p.write_text("wdir: .\n", encoding="utf-8")
        with pytest.raises(DvcTrackingError, match=r"đúng 1 entry `outs`, thấy 0"):
            load_dvc_out(p)

    def test_rejects_multiple_outs(self, tmp_path: Path) -> None:
        p = tmp_path / "corpus.dvc"
        p.write_text(
            "outs:\n"
            "- md5: a.dir\n  size: 1\n  nfiles: 1\n  path: corpus\n"
            "- md5: b.dir\n  size: 1\n  nfiles: 1\n  path: golden\n",
            encoding="utf-8",
        )
        with pytest.raises(DvcTrackingError, match=r"đúng 1 entry `outs`, thấy 2"):
            load_dvc_out(p)

    def test_out_is_frozen(self, tmp_path: Path) -> None:
        out = load_dvc_out(_write_dvc(tmp_path / "c.dvc", nfiles=2))
        with pytest.raises(ValidationError, match="frozen"):
            out.nfiles = 99

    def test_md5_read_from_alias_not_field_name(self, tmp_path: Path) -> None:
        """`hash_md5` phải đọc từ khoá `md5`; đặt tên khác dễ lặng lẽ nhận None."""
        out = DvcOut.model_validate({"md5": "xyz.dir", "size": 1, "nfiles": 1, "path": "corpus"})
        assert out.hash_md5 == "xyz.dir"


class TestCountAgreement:
    """Phép so số lượng — chỗ duy nhất mà cả sha256 và md5 đều mù."""

    def test_agreeing_counts_pass(self, tmp_path: Path) -> None:
        root = _make_corpus(tmp_path / "corpus", ["a.txt", "b.txt"])
        dvc = _write_dvc(tmp_path / "corpus.dvc", nfiles=3)  # 2 tài liệu + .gitkeep
        manifest = _write_manifest(tmp_path / "m.csv", ["a.txt", "b.txt"])

        report = verify_corpus_tracking(dvc, manifest, root)
        assert report.tracked_documents == 2
        assert report.manifest_documents == 2
        assert report.non_document_files == 1

    def test_dvc_added_a_file_manifest_never_saw(self, tmp_path: Path) -> None:
        """Kịch bản thật: kéo file vào `data/corpus/` rồi `dvc add`, quên manifest.

        Không có gì nổ ra ở đường chạy bình thường — build index đọc theo manifest
        nên bỏ qua file lạ, `dvc status` vẫn sạch. Đây là test duy nhất bắt được.
        """
        root = _make_corpus(tmp_path / "corpus", ["a.txt", "b.txt", "la-mat.txt"])
        dvc = _write_dvc(tmp_path / "corpus.dvc", nfiles=4)
        manifest = _write_manifest(tmp_path / "m.csv", ["a.txt", "b.txt"])

        with pytest.raises(DvcTrackingError, match="không có trong manifest"):
            verify_corpus_tracking(dvc, manifest, root)

    def test_manifest_lists_a_file_that_is_gone(self, tmp_path: Path) -> None:
        root = _make_corpus(tmp_path / "corpus", ["a.txt"])
        dvc = _write_dvc(tmp_path / "corpus.dvc", nfiles=2)
        manifest = _write_manifest(tmp_path / "m.csv", ["a.txt", "bi-xoa.txt"])

        with pytest.raises(DvcTrackingError, match=r"dvc pull"):
            verify_corpus_tracking(dvc, manifest, root)

    def test_stale_dvc_file_detected_without_disk(self, tmp_path: Path) -> None:
        """`.dvc` cũ so với manifest — bắt được cả khi chưa `dvc pull`."""
        dvc = _write_dvc(tmp_path / "corpus.dvc", nfiles=51)  # 50 tài liệu
        manifest = _write_manifest(tmp_path / "m.csv", [f"d{i}.txt" for i in range(60)])

        with pytest.raises(
            DvcTrackingError, match="DVC theo dõi 50 tài liệu nhưng manifest khai 60"
        ):
            verify_corpus_tracking(dvc, manifest, corpus_dir=None)

    def test_gitkeep_absent_is_handled(self, tmp_path: Path) -> None:
        """Không có `.gitkeep` thì nfiles bằng đúng số tài liệu, không lệch 1."""
        root = _make_corpus(tmp_path / "corpus", ["a.txt", "b.txt"], gitkeep=False)
        dvc = _write_dvc(tmp_path / "corpus.dvc", nfiles=2)
        manifest = _write_manifest(tmp_path / "m.csv", ["a.txt", "b.txt"])

        report = verify_corpus_tracking(dvc, manifest, root)
        assert report.non_document_files == 0
        assert report.tracked_documents == 2

    def test_gitkeep_is_not_counted_as_a_document(self, tmp_path: Path) -> None:
        """Nếu trừ sai file phụ trợ thì test này là chỗ lộ ra."""
        root = _make_corpus(tmp_path / "corpus", ["a.txt"])
        dvc = _write_dvc(tmp_path / "corpus.dvc", nfiles=2)
        manifest = _write_manifest(tmp_path / "m.csv", ["a.txt"])

        assert verify_corpus_tracking(dvc, manifest, root).tracked_documents == 1

    def test_nested_documents_are_matched_by_relative_path(self, tmp_path: Path) -> None:
        """Manifest dùng đường dẫn tương đối, nên file trong thư mục con phải khớp."""
        root = tmp_path / "corpus"
        (root / "vi").mkdir(parents=True)
        (root / "vi" / "a.txt").write_bytes(b"x")
        (root / ".gitkeep").write_bytes(b"")
        dvc = _write_dvc(tmp_path / "corpus.dvc", nfiles=2)
        manifest = _write_manifest(tmp_path / "m.csv", ["vi/a.txt"])

        assert verify_corpus_tracking(dvc, manifest, root).tracked_documents == 1

    def test_error_message_carries_the_numbers(self, tmp_path: Path) -> None:
        """Thông báo phải đủ để sửa mà không cần mở lại code."""
        dvc = _write_dvc(tmp_path / "corpus.dvc", nfiles=11)
        manifest = _write_manifest(tmp_path / "m.csv", [f"d{i}.txt" for i in range(3)])

        with pytest.raises(DvcTrackingError) as exc:
            verify_corpus_tracking(dvc, manifest, corpus_dir=None)
        msg = str(exc.value)
        assert "10 tài liệu" in msg and "khai 3" in msg
        assert "fetch_corpus.py" in msg


class TestReport:
    def test_summary_is_human_readable(self, tmp_path: Path) -> None:
        dvc = _write_dvc(tmp_path / "c.dvc", nfiles=61, size=15_457_683)
        manifest = _write_manifest(tmp_path / "m.csv", [f"d{i}.txt" for i in range(60)])
        summary = verify_corpus_tracking(dvc, manifest, corpus_dir=None).summary()
        assert "61 file" in summary
        assert "60 tài liệu" in summary
        assert "14.7 MiB" in summary

    def test_report_is_frozen(self, tmp_path: Path) -> None:
        dvc = _write_dvc(tmp_path / "c.dvc", nfiles=2)
        manifest = _write_manifest(tmp_path / "m.csv", ["a.txt"])
        report = verify_corpus_tracking(dvc, manifest, corpus_dir=None)
        with pytest.raises(ValidationError, match="frozen"):
            report.dvc_nfiles = 5


class TestRealRepo:
    """Chạy trên chính dữ liệu của repo — bắt lệch thật, không phải lệch giả lập."""

    def test_repo_dvc_file_is_consistent_with_manifest(self) -> None:
        """Không cần corpus trên đĩa: chỉ so `.dvc` với manifest."""
        dvc = Path("data/corpus.dvc")
        manifest = Path("data/corpus_manifest.csv")
        if not dvc.is_file() or not manifest.is_file():
            pytest.skip("Chạy ngoài repo hoặc corpus chưa được DVC theo dõi.")

        report = verify_corpus_tracking(dvc, manifest, corpus_dir=None)
        assert report.tracked_documents == report.manifest_documents

    def test_repo_corpus_on_disk_matches_manifest(self) -> None:
        """Bản đầy đủ — skip trên clone sạch chưa `dvc pull`."""
        root = Path("data/corpus")
        dvc = Path("data/corpus.dvc")
        if not dvc.is_file() or not any(root.glob("*.txt")):
            pytest.skip("Corpus chưa có trên đĩa — chạy `dvc pull` hoặc `make corpus`.")

        report = verify_corpus_tracking(dvc, "data/corpus_manifest.csv", root)
        assert report.manifest_documents == 60
