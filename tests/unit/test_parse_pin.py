"""`TD-22` — chuỗi toàn vẹn khi có parser đứng giữa byte và `Document.content`.

Chế độ hỏng phải chặn được, nguyên văn từ `TD-22`: *`sha256` vẫn khớp manifest,
`iter_documents` vẫn xanh, mọi `chunk_id` vẫn tồn tại, không test nào đỏ — và
mọi con số recall sau đó sai.*

Dựng lại mà không phải nâng cấp thư viện thật: chép **nguyên byte** một tài liệu
sang tên `.md`. Byte không đổi một bit nên `sha256` khớp tuyệt đối, nhưng `.md`
định tuyến sang docling và văn bản parse ra là một thứ khác — đúng hình dạng của
một lượt đổi parser, và đúng chỗ mà kiểm-theo-byte mù.

Tách khỏi `test_corpus_loader.py` (`W1-08`) vì hai file kiểm hai thứ khác nhau:
ở đó là *file trên đĩa có đúng file đã tải về không*, ở đây là *văn bản parse ra
có đúng văn bản golden set neo vào không*. Hai câu hỏi độc lập, và `TD-22` tồn
tại chính vì chúng từng bị coi là một.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pytest

from pipeline.corpus.manifest import CorpusEntry, load_manifest, write_manifest
from pipeline.indexing.corpus_loader import ParsePinError, load_documents
from rag_core.loaders import ParseFingerprint, load_document
from rag_core.loaders.docling_backend import component_versions

pytestmark = pytest.mark.weights
"""Ba module này parse tài liệu thật bằng docling, tức chúng cần **trọng số
model tải về** — xem marker `weights` trong `pyproject.toml`. Vẫn chạy trong
`make test` ở máy đã `make install`; CI nhanh loại chúng ra và
`tests/unit/test_ci_tiers.py` khoá danh sách ấy lại."""


MARKDOWN = "# Tiêu đề\n\nMột đoạn văn bản về đầu tư công.\n"


def _write_bytes(path: Path, text: str) -> None:
    """Ghi bằng bytes. `write_text` trên Windows đổi xuống dòng sang CRLF."""
    path.write_bytes(text.encode("utf-8"))


def _entry(doc_id: str, text: str, suffix: str) -> CorpusEntry:
    payload = text.encode("utf-8")
    return CorpusEntry.model_validate(
        {
            "doc_id": doc_id,
            "relative_path": f"{doc_id}{suffix}",
            "source_url": f"https://example.org/{doc_id}",
            "license": "CC BY 4.0",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "source": "test",
            "lang": "vi",
            "doc_type": "dev_report",
        }
    )


@pytest.fixture
def as_markdown(tmp_path: Path) -> tuple[CorpusEntry, Path]:
    entry = _entry("d-md", MARKDOWN, ".md")
    _write_bytes(tmp_path / entry.relative_path, MARKDOWN)
    return entry, tmp_path


@pytest.fixture
def as_text(tmp_path: Path) -> tuple[CorpusEntry, Path]:
    entry = _entry("d-txt", MARKDOWN, ".txt")
    _write_bytes(tmp_path / entry.relative_path, MARKDOWN)
    return entry, tmp_path


def _pinned(entry: CorpusEntry, room: Path) -> CorpusEntry:
    loaded = load_document(room / entry.relative_path, ocr="off")
    return entry.model_copy(
        update={
            "text_sha256": loaded.text_sha256,
            "parse_fingerprint": loaded.fingerprint.canonical,
        }
    )


class TestHamDongNhatKhongCanGhim:
    """`.txt` chưa ghim vẫn nạp được, và đó là một luật có bằng chứng."""

    def test_text_sha256_bang_dung_sha256_byte(self, as_text: tuple[CorpusEntry, Path]) -> None:
        entry, room = as_text
        loaded = load_document(room / entry.relative_path, ocr="off")
        assert loaded.text_sha256 == entry.sha256 == loaded.source_sha256

    def test_txt_chua_ghim_van_nap_duoc(self, as_text: tuple[CorpusEntry, Path]) -> None:
        entry, room = as_text
        write_manifest(room / "m.csv", [entry])
        docs = load_documents(room / "m.csv", room)
        assert docs[0].content == MARKDOWN

    def test_corpus_that_van_la_ham_dong_nhat(self) -> None:
        """60/60 tài liệu repo phải còn nằm trên đường đồng nhất.

        Đây là điều kiện để mọi con số `W2` và mọi `TextSpan` của `golden_v1` còn
        giá trị. Ngày nào nó đỏ thì ngày đó có tài liệu đi qua parser, và lúc ấy
        phải ghim chứ không phải sửa test.
        """
        root = Path(__file__).resolve().parents[2]
        manifest = root / "data" / "corpus_manifest.csv"
        if not manifest.exists():
            pytest.skip("chưa chạy scripts/fetch_corpus.py")
        entries = load_manifest(manifest)
        pinned = [e for e in entries if e.text_sha256]
        assert pinned, "manifest chưa ghim — chạy `make corpus-pin`"
        assert all(e.text_sha256 == e.sha256 for e in pinned), (
            "có tài liệu không còn đồng nhất byte↔văn bản"
        )


class TestChoHoBiChan:
    def test_dinh_dang_khac_txt_chua_ghim_thi_dung(
        self, as_markdown: tuple[CorpusEntry, Path]
    ) -> None:
        entry, room = as_markdown
        write_manifest(room / "m.csv", [entry])
        with pytest.raises(ParsePinError, match="chưa ghim"):
            load_documents(room / "m.csv", room)

    def test_hash_byte_van_khop_khi_van_ban_da_lech(
        self, as_markdown: tuple[CorpusEntry, Path]
    ) -> None:
        """Chốt của cả `TD-22`: kiểm theo byte KHÔNG bắt được lỗi này."""
        entry, room = as_markdown
        assert entry.sha256 == hashlib.sha256(MARKDOWN.encode("utf-8")).hexdigest()
        loaded = load_document(room / entry.relative_path, ocr="off")
        assert loaded.source_sha256 == entry.sha256, "byte phải khớp tuyệt đối"
        assert loaded.text_sha256 != entry.sha256, "văn bản parse ra phải đã khác"

        write_manifest(room / "m.csv", [entry])
        with pytest.raises(ParsePinError):
            load_documents(room / "m.csv", room, verify_hash=True)

    def test_ghim_dung_thi_nap_duoc_va_ghi_lai_parser(
        self, as_markdown: tuple[CorpusEntry, Path]
    ) -> None:
        entry, room = as_markdown
        pinned = _pinned(entry, room)
        write_manifest(room / "m.csv", [pinned])
        docs = load_documents(room / "m.csv", room)
        extra = docs[0].metadata.extra
        # `as_metadata()` được dựng ở `W3-01` rồi nằm không tới tận đây; nối vào
        # nghĩa là từ một `chunk_id` bất kỳ truy ngược được lần parse sinh ra nó.
        assert extra["parser"] == "docling"
        assert extra["text_sha256"] == pinned.text_sha256
        assert (
            extra["parse_fingerprint"]
            == load_document(room / entry.relative_path, ocr="off").fingerprint.digest
        )


class TestThongBaoPhaiChiDungThuPham:
    def test_neu_ten_goi_da_dich_chuyen(self, as_markdown: tuple[CorpusEntry, Path]) -> None:
        entry, room = as_markdown
        good = _pinned(entry, room)
        drifted = good.model_copy(
            update={
                "text_sha256": "0" * 64,
                "parse_fingerprint": good.parse_fingerprint.replace(
                    "docling-core=", "docling-core=9.9.9|dead-"
                ),
            }
        )
        write_manifest(room / "m.csv", [drifted])
        with pytest.raises(ParsePinError) as caught:
            load_documents(room / "m.csv", room)
        message = str(caught.value)
        assert "docling-core=9.9.9" in message, "không nêu giá trị đã ghim"
        assert "mất:" in message and "thêm:" in message

    def test_parser_doi_ma_van_ban_khong_doi_thi_chi_canh_bao(
        self, as_markdown: tuple[CorpusEntry, Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Không phải lỗi. Nhưng im lặng thì lần lệch THẬT sau sẽ chỉ sai thủ phạm."""
        entry, room = as_markdown
        good = _pinned(entry, room)
        stale = good.model_copy(
            update={"parse_fingerprint": good.parse_fingerprint + "|dead-component=1"}
        )
        write_manifest(room / "m.csv", [stale])
        with caplog.at_level(logging.WARNING):
            docs = load_documents(room / "m.csv", room)
        assert len(docs) == 1
        assert "parser đã đổi" in caplog.text
        assert "dead-component=1" in caplog.text


class TestVanTayPhaiDayDu:
    """Tầng hai của `TD-22`: ghim đúng một số version là chưa ghim gì cả.

    `W3-01` ghi version của `docling`. Nhưng `DoclingDocument.export_to_markdown`
    — hàm sinh ra `LoadedDocument.text` — sống trong **`docling-core`**, gói riêng
    mà `docling==2.121.0` cho chạy từ `>=2.91.0` tới `<3.0.0`. Nên trước thay đổi
    này, `docling-core` dịch chuyển là markdown đổi trong khi vân tay đứng yên.
    """

    def test_vantay_pdf_ghim_ca_goi_phu(self) -> None:
        names = {p.split("=")[0].split("@")[0] for p in component_versions(".pdf", ocr_engine=None)}
        assert "docling-core" in names, "gói serialise markdown phải được ghim"
        assert {"docling-parse", "pypdfium2", "docling-ibm-models"} <= names

    def test_vantay_pdf_ghim_ca_trong_so_model(self) -> None:
        """Model bố cục tải theo `revision='main'` — một nhánh di động.

        Ghim chuỗi `"main"` là không ghim gì; phải ghim commit SHA đã phân giải.
        """
        parts = component_versions(".pdf", ocr_engine=None)
        assert any(p.startswith("docling-layout-heron@") for p in parts)
        assert any(p.startswith("docling-models@") for p in parts)

    def test_duong_khong_pdf_khong_ghim_thua(self) -> None:
        """Ghi thừa cũng có giá: một cảnh báo kêu suốt là một cảnh báo bị tắt."""
        parts = component_versions(".docx", ocr_engine=None)
        assert all(p.startswith("docling-core=") for p in parts)
        assert not any("rapidocr" in p or "pypdfium2" in p for p in parts)

    def test_ocr_them_dung_goi_cua_tung_may(self) -> None:
        """Hai máy OCR cho hai văn bản khác nhau trên cùng ảnh (đo 2026-09-04) —
        nên máy nào chạy phải đọc được từ vân tay, không phải chỉ "có OCR"."""
        rapid = component_versions(".pdf", ocr_engine="rapidocr")
        easy = component_versions(".pdf", ocr_engine="easyocr")
        off = component_versions(".pdf", ocr_engine=None)
        assert any("rapidocr=" in p for p in rapid)
        assert any("easyocr=" in p for p in easy)
        assert not any("rapidocr=" in p or "easyocr=" in p for p in off)
        assert not any("easyocr=" in p for p in rapid)
        assert not any("rapidocr=" in p for p in easy)

    def test_components_di_vao_digest(self) -> None:
        """Thêm `components` mà quên đưa vào `canonical` thì cả thay đổi này vô nghĩa."""
        base = ParseFingerprint(loader="x", library="y", library_version="1")
        with_component = ParseFingerprint(
            loader="x", library="y", library_version="1", components=("z=2",)
        )
        assert base.digest != with_component.digest
        assert "z=2" in with_component.canonical
        # Thứ tự lời gọi không được đổi digest — `components` phải được sắp.
        shuffled = ParseFingerprint(
            loader="x", library="y", library_version="1", components=("b=1", "a=1")
        )
        ordered = ParseFingerprint(
            loader="x", library="y", library_version="1", components=("a=1", "b=1")
        )
        assert shuffled.digest == ordered.digest
