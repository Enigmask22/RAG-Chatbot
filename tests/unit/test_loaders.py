"""`W3-01` — loader 6 định dạng.

Fixture do `scripts/make_loader_fixtures.py` sinh, mỗi định dạng một file, cùng
một nội dung logic (h1 → h2 → h3 + một bảng 4×3) để so được giữa các backend.

Test PDF cần model bố cục của docling. Máy chưa có cache HF thì **skip** chứ
không đỏ — nhưng chỉ skip khi lỗi đúng là lỗi tải model; mọi lỗi khác vẫn phải
đỏ, nếu không thì một hồi quy thật cũng lặng lẽ thành "skipped".
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rag_core.loaders import (
    DOCLING_FORMATS,
    PLAIN_FORMATS,
    Heading,
    LoadedDocument,
    LoaderError,
    ParseFingerprint,
    UnsupportedFormatError,
    detect_format,
    load_document,
    loader_for,
)
from rag_core.loaders.plain import load_plain

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "loaders"

PDF = FIXTURES / "two-column.pdf"
DOCX = FIXTURES / "chuong-i.docx"
PPTX = FIXTURES / "chuong-i.pptx"
XLSX = FIXTURES / "chi-tieu.xlsx"
HTML = FIXTURES / "chuong-i.html"
MD = FIXTURES / "chuong-i.md"

ALL_FIXTURES = (PDF, DOCX, PPTX, XLSX, HTML, MD)

# Ba định dạng mang được phân cấp heading thật. PPTX không mang (mỗi slide một
# `title`), XLSX không có heading nào — cả hai đều đã đo, xem `docling_backend`.
HIERARCHICAL = (DOCX, HTML, MD)

_MODEL_MARKERS = ("huggingface", "connection", "offline", "resolve", "download", "timed out")


def _load_or_skip(path: Path) -> LoadedDocument:
    try:
        return load_document(path)
    except LoaderError as exc:
        message = str(exc).lower()
        if any(marker in message for marker in _MODEL_MARKERS):
            pytest.skip(f"không tải được model docling: {exc}")
        raise


@pytest.fixture(scope="module")
def pdf_document() -> LoadedDocument:
    return _load_or_skip(PDF)


# --------------------------------------------------------------------------
class TestEveryFormatInTheDoDLoads:
    """DoD liệt kê 6 định dạng — mỗi cái phải ra được văn bản, không chỉ 5 cái dễ."""

    @pytest.mark.parametrize("path", ALL_FIXTURES, ids=lambda p: p.suffix)
    def test_fixture_exists(self, path: Path) -> None:
        assert path.is_file(), f"thiếu fixture {path.name}; chạy `make loader-fixtures`"

    @pytest.mark.parametrize("path", ALL_FIXTURES, ids=lambda p: p.suffix)
    def test_loads_to_non_empty_text(self, path: Path) -> None:
        document = _load_or_skip(path)
        assert document.text.strip()
        assert document.fingerprint.loader == "docling"

    @pytest.mark.parametrize("path", ALL_FIXTURES, ids=lambda p: p.suffix)
    def test_source_hash_is_of_the_bytes_on_disk(self, path: Path) -> None:
        document = _load_or_skip(path)
        assert document.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


class TestTheHeadingDepthIsNormalisedBecauseBackendsDisagree:
    """Cùng một tài liệu h1→h2→h3 ra ba kiểu đánh số tuỳ định dạng nguồn.

    `.docx` cho `section_header` level 1/2/3; `.md` và `.html` cho `title` (không
    level) rồi `section_header` level 1/2. Tin thẳng `item.level` thì cùng một
    heading là cấp 3 khi tới từ DOCX và cấp 2 khi tới từ HTML — tức `W3-03` sẽ
    dựng `section_path` khác nhau cho cùng một nội dung.
    """

    @pytest.mark.parametrize("path", HIERARCHICAL, ids=lambda p: p.suffix)
    def test_three_levels_come_out_as_depths_one_two_three(self, path: Path) -> None:
        document = _load_or_skip(path)
        assert [h.depth for h in document.headings] == [1, 2, 3]

    @pytest.mark.parametrize("path", HIERARCHICAL, ids=lambda p: p.suffix)
    def test_every_heading_is_located_in_the_exported_text(self, path: Path) -> None:
        document = _load_or_skip(path)
        assert all(h.located for h in document.headings)

    def test_the_deepest_section_path_is_the_same_across_formats(self) -> None:
        paths = {}
        for path in HIERARCHICAL:
            document = _load_or_skip(path)
            deepest = document.headings[-1]
            paths[path.suffix] = document.section_path_at(deepest.start_char)
        assert len({len(value) for value in paths.values()}) == 1, paths
        assert all(len(value) == 3 for value in paths.values()), paths

    def test_pptx_carries_no_hierarchy_and_that_is_the_format_not_a_bug(self) -> None:
        document = _load_or_skip(PPTX)
        assert document.headings, "PPTX vẫn phải có heading (mỗi slide một title)"
        assert document.max_depth == 1

    def test_xlsx_has_no_heading_at_all(self) -> None:
        document = _load_or_skip(XLSX)
        assert document.headings == ()


class TestTablesSurviveAsStructure:
    """DoD: "bảng giữ được cấu trúc". Đếm bảng, và kiểm cả 4 hàng có mặt."""

    WITH_TABLES = (DOCX, PPTX, XLSX, HTML, MD)

    @pytest.mark.parametrize("path", WITH_TABLES, ids=lambda p: p.suffix)
    def test_at_least_one_table_is_recognised(self, path: Path) -> None:
        assert _load_or_skip(path).table_count >= 1

    @pytest.mark.parametrize("path", WITH_TABLES, ids=lambda p: p.suffix)
    def test_every_row_of_the_table_survives(self, path: Path) -> None:
        text = _load_or_skip(path).text
        for label in ("Tăng trưởng GDP", "Lạm phát CPI", "Xuất khẩu"):
            assert label in text, f"{path.suffix}: mất hàng {label!r}"

    @pytest.mark.parametrize("path", WITH_TABLES, ids=lambda p: p.suffix)
    def test_the_table_is_exported_as_a_markdown_pipe_table(self, path: Path) -> None:
        if path is PPTX:
            pytest.skip("PPTX đưa cùng dữ liệu vào cả bullet lẫn bảng — xem docstring")
        text = _load_or_skip(path).text
        assert "|" in text and "---" in text


class TestTwoColumnPdfIsReadInReadingOrderNotStreamOrder:
    """DoD: "PDF 2 cột đọc đúng reading order".

    Test này chỉ có nghĩa nếu fixture **thật sự** sai thứ tự trong content
    stream — nếu không thì mọi bộ trích text đều "đúng" và assertion không phân
    biệt được gì. Nên kiểm cả hai vế.
    """

    def test_the_fixture_really_is_out_of_order_in_the_content_stream(self) -> None:
        raw = PDF.read_bytes()
        first_beta = raw.find(b"(Beta. The right column continues)")
        second_alpha_line = raw.find(b"(first half of the argument and it)")
        assert first_beta >= 0 and second_alpha_line >= 0, (
            "không tìm thấy dòng mong đợi trong content stream — fixture đã đổi cách wrap"
        )
        assert first_beta < second_alpha_line, (
            "fixture không còn xen kẽ hai cột — test thứ tự đọc mất ý nghĩa"
        )

    def test_the_whole_left_column_comes_before_the_right_one(
        self, pdf_document: LoadedDocument
    ) -> None:
        flat = " ".join(pdf_document.text.split())
        end_of_left = flat.rfind("prose in a report tends to.")
        start_of_right = flat.find("Beta.")
        assert end_of_left >= 0 and start_of_right >= 0, flat[:200]
        assert end_of_left < start_of_right, f"đọc theo thứ tự content stream: {flat[:200]}"

    def test_the_title_becomes_a_heading(self, pdf_document: LoadedDocument) -> None:
        assert any("Two Column" in h.text for h in pdf_document.headings)


class TestPlainTextIsTheIdentityFunction:
    """`.txt` không được đi qua bộ parse nào — cả corpus và mọi span nằm ở đó."""

    def test_txt_routes_to_plain(self) -> None:
        assert loader_for("bat-ky.txt") == "plain"

    def test_docling_does_not_claim_txt(self) -> None:
        assert ".txt" not in DOCLING_FORMATS
        assert PLAIN_FORMATS.isdisjoint(DOCLING_FORMATS)

    def test_text_is_byte_for_byte_the_decoded_file(self, tmp_path: Path) -> None:
        payload = "Dòng một\r\n\r\n  Dòng hai — có dấu gạch dài  \n\n\n".encode()
        target = tmp_path / "raw.txt"
        target.write_bytes(payload)

        document = load_document(target)

        assert document.text == payload.decode("utf-8")
        assert document.text_sha256 == hashlib.sha256(payload).hexdigest()

    def test_an_em_dash_is_not_normalised_away(self, tmp_path: Path) -> None:
        """docling backend HTML đổi `—` thành `-`; đường plain thì không."""
        target = tmp_path / "dash.txt"
        target.write_text("Chương I — Tổng quan", encoding="utf-8")
        assert "—" in load_document(target).text

    def test_empty_file_is_rejected(self, tmp_path: Path) -> None:
        target = tmp_path / "rong.txt"
        target.write_text("   \n\n", encoding="utf-8")
        with pytest.raises(LoaderError, match="rỗng"):
            load_document(target)

    def test_non_utf8_is_rejected_with_a_readable_message(self, tmp_path: Path) -> None:
        target = tmp_path / "latin.txt"
        target.write_bytes(b"\xff\xfe kh\xf4ng ph\xe3i utf-8")
        with pytest.raises(LoaderError, match="utf-8"):
            load_plain(target, source_sha256="x")


class TestTheFingerprintCoversEverythingThatChangesTheText:
    """Ba đầu vào quyết định văn bản xuất ra; manifest mới ghim được một."""

    def _fp(self, **kwargs: object) -> ParseFingerprint:
        base: dict[str, object] = {
            "loader": "docling",
            "library": "docling",
            "library_version": "2.121.0",
            "options": ("ocr=false",),
        }
        base.update(kwargs)
        return ParseFingerprint(**base)  # type: ignore[arg-type]

    def test_a_version_bump_changes_the_digest(self) -> None:
        assert self._fp().digest != self._fp(library_version="2.122.0").digest

    def test_an_option_change_changes_the_digest(self) -> None:
        assert self._fp().digest != self._fp(options=("ocr=true",)).digest

    def test_option_order_does_not_change_the_digest(self) -> None:
        left = self._fp(options=("ocr=false", "table_structure=true"))
        right = self._fp(options=("table_structure=true", "ocr=false"))
        assert left.digest == right.digest

    def test_the_plain_path_has_a_stable_digest(self, tmp_path: Path) -> None:
        target = tmp_path / "a.txt"
        target.write_text("xin chào", encoding="utf-8")
        first = load_document(target).fingerprint.digest
        second = load_document(target).fingerprint.digest
        assert first == second

    def test_metadata_carries_both_hashes(self, tmp_path: Path) -> None:
        target = tmp_path / "a.txt"
        target.write_text("xin chào", encoding="utf-8")
        metadata = load_document(target).as_metadata()
        assert metadata["parser"] == "plain"
        assert metadata["text_sha256"] == load_document(target).text_sha256
        assert len(metadata["parse_fingerprint"]) == 16


class TestSectionPathAt:
    """Cái `W3-03` sẽ gọi cho từng chunk."""

    def _doc(self, headings: tuple[Heading, ...]) -> LoadedDocument:
        return LoadedDocument(
            text="x" * 500,
            source_sha256="0" * 64,
            fingerprint=ParseFingerprint("plain", "stdlib", "1"),
            headings=headings,
        )

    def test_nesting_pops_back_to_the_right_level(self) -> None:
        document = self._doc(
            (
                Heading(1, "Chương I", 0),
                Heading(2, "Điều 1", 50),
                Heading(3, "Khoản 1", 100),
                Heading(2, "Điều 2", 200),
            )
        )
        assert document.section_path_at(150) == ("Chương I", "Điều 1", "Khoản 1")
        assert document.section_path_at(250) == ("Chương I", "Điều 2")

    def test_before_the_first_heading_the_path_is_empty(self) -> None:
        document = self._doc((Heading(1, "Chương I", 40),))
        assert document.section_path_at(10) == ()

    def test_an_unlocated_heading_is_skipped_not_guessed(self) -> None:
        document = self._doc((Heading(1, "Chương I", 0), Heading(2, "Điều 1", -1)))
        assert document.section_path_at(400) == ("Chương I",)


class TestUnsupportedInput:
    def test_an_unknown_extension_names_what_is_supported(self) -> None:
        with pytest.raises(UnsupportedFormatError, match=r"\.pdf"):
            loader_for("kho-luu.zip")

    def test_a_file_without_extension_is_refused(self) -> None:
        with pytest.raises(UnsupportedFormatError):
            loader_for("README")

    def test_a_missing_file_is_a_loader_error(self, tmp_path: Path) -> None:
        with pytest.raises(LoaderError, match="Không thấy file"):
            load_document(tmp_path / "khong-ton-tai.md")

    def test_detect_format_lowercases(self) -> None:
        assert detect_format("BAO-CAO.PDF") == ".pdf"
