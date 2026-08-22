"""`W3-03` — chunker cắt theo heading, và `section_path` phải nói đúng sự thật.

Phép kiểm trung tâm của file này không phải "path bằng hằng số tôi viết trong
fixture" mà là **path phải nhất quán với `LoadedDocument.section_path_at`** ở
từng chunk. Lý do: bản đầu của chunker hỏi `section_path_at` tại *đầu dòng*
heading thay vì tại *chữ* của heading, nên mọi chunk mang path của section liền
trước — 486/587 chunk trên `wb1.pdf`. Hằng số viết tay trong một fixture ba
heading vẫn xanh với lỗi ấy nếu người viết fixture cũng lệch cùng chiều; ràng
buộc với `section_path_at` thì không.

Phép kiểm ấy cũng đã bắt lỗi lần thứ hai, ở chính script kiểm chứng: bản kiểm
đầu hỏi tại **đầu** chunk, tức lặp lại đúng cái lỗi nó đi kiểm (chunk mở đầu
bằng `## Heading` thì đầu chunk là `#`). Nên ở đây hỏi tại **ký tự cuối** của
chunk — vị trí duy nhất chắc chắn nằm trong section mà chunk kết thúc.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from rag_core.chunking import (
    ChunkingConfig,
    ChunkingStrategy,
    FixedSizeChunker,
    StructureChunker,
    build_chunker,
    common_ancestor,
    section_boundaries,
)
from rag_core.loaders.base import Heading, LoadedDocument, ParseFingerprint
from rag_core.schemas import Chunk, Document, DocumentMetadata

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "loaders"

METADATA = DocumentMetadata(
    source_url="https://example.org/chuong-i",
    license="CC BY 4.0",
)

FINGERPRINT = ParseFingerprint("test", "test", "1")


def loaded(text: str, *headings: Heading) -> LoadedDocument:
    return LoadedDocument(
        text=text,
        source_sha256="0" * 64,
        fingerprint=FINGERPRINT,
        headings=headings,
    )


def document(text: str, doc_id: str = "doc") -> Document:
    return Document(doc_id=doc_id, content=text, metadata=METADATA)


def chunk_with(structure: LoadedDocument, *, doc_id: str = "doc", **config: object) -> list[Chunk]:
    """Chunk `structure.text` bằng `StructureChunker` với config đã ghi đè."""
    chunker = StructureChunker(ChunkingConfig(strategy=ChunkingStrategy.STRUCTURE, **config))  # type: ignore[arg-type]
    chunker.bind(doc_id, structure)
    return chunker.chunk([document(structure.text, doc_id)])


def path_is_truthful(chunk: Chunk, structure: LoadedDocument) -> bool:
    """`section_path` phải là **tiền tố** của đường dẫn tại ký tự cuối của chunk.

    Bằng nhau với chunk nằm gọn trong một section; ngắn hơn với chunk gộp qua
    ranh giới, vì lúc đó path đã tụt xuống tổ tiên chung. Dài hơn — hoặc lệch —
    là nói dối.
    """
    assert chunk.end_char is not None
    truth = list(structure.section_path_at(chunk.end_char - 1))
    return truth[: len(chunk.section_path)] == chunk.section_path


# `# h1` (2) · `## h2` (35) · `### h3` (72); thân mỗi section ~40 ký tự
THREE_LEVEL = (
    "# Chương I\n\n"  # 0..11
    "Mở đầu chương, đủ dài để không bị gộp đi đâu cả.\n\n"
    "## Điều 1\n\n"
    "Thân của điều một, cũng đủ dài để đứng riêng.\n\n"
    "### Khoản a\n\n"
    "Thân của khoản a, dòng cuối cùng của tài liệu.\n"
)


def three_level() -> LoadedDocument:
    return loaded(
        THREE_LEVEL,
        Heading(1, "Chương I", THREE_LEVEL.index("Chương I")),
        Heading(2, "Điều 1", THREE_LEVEL.index("Điều 1")),
        Heading(3, "Khoản a", THREE_LEVEL.index("Khoản a")),
    )


class TestDoDSectionPathTrenTaiLieuBaCapHeading:
    """DoD `W3-03`: `section_path` đúng trên tài liệu có 3 cấp heading."""

    def test_ba_chunk_ba_cap(self) -> None:
        structure = three_level()
        chunks = chunk_with(structure, min_chunk_size=0)

        assert [c.section_path for c in chunks] == [
            ["Chương I"],
            ["Chương I", "Điều 1"],
            ["Chương I", "Điều 1", "Khoản a"],
        ]

    def test_moi_chunk_nhat_quan_voi_section_path_at(self) -> None:
        structure = three_level()
        for chunk in chunk_with(structure, min_chunk_size=0):
            assert path_is_truthful(chunk, structure), chunk.section_path

    def test_section_header_ghep_lai_thanh_chuoi_de_prepend(self) -> None:
        chunks = chunk_with(three_level(), min_chunk_size=0)
        assert chunks[-1].section_header == "Chương I > Điều 1 > Khoản a"

    def test_chunker_khong_nhet_heading_vao_content(self) -> None:
        """Heading nằm trong `content` vì nó vốn ở đó, không phải do chunker thêm."""
        structure = three_level()
        for chunk in chunk_with(structure, min_chunk_size=0):
            assert chunk.start_char is not None and chunk.end_char is not None
            assert chunk.content == structure.text[chunk.start_char : chunk.end_char]


class TestTaiLieuKhongCoHeading:
    """Nhánh thứ hai của DoD — và là 60/60 tài liệu corpus hôm nay."""

    def test_section_path_rong_va_khong_lỗi(self) -> None:
        text = "Một tài liệu phẳng.\n\n" * 20
        chunks = chunk_with(loaded(text))
        assert chunks
        assert all(c.section_path == [] for c in chunks)

    def test_thoai_hoa_ve_dung_ket_qua_cua_fixed_chunker(self) -> None:
        """Không heading thì phải ra **đúng** bộ chunk mà `FixedSizeChunker` ra.

        Nếu lệch thì đổi `strategy` sang `structure` trên corpus `.txt` hiện tại
        sẽ âm thầm đổi baseline, trong khi chẳng có cấu trúc nào được dùng.
        """
        text = "Đoạn văn số một. " * 300
        config = ChunkingConfig(strategy=ChunkingStrategy.STRUCTURE)
        structure_chunks = chunk_with(loaded(text))
        fixed_chunks = FixedSizeChunker(config).chunk([document(text)])

        assert [c.content for c in structure_chunks] == [c.content for c in fixed_chunks]
        assert [c.span for c in structure_chunks] == [c.span for c in fixed_chunks]

    def test_heading_khong_dinh_vi_duoc_cung_la_khong_co_cau_truc(self) -> None:
        text = "Một tài liệu phẳng.\n\n" * 20
        structure = loaded(text, Heading(1, "Không tìm thấy", -1))
        assert all(c.section_path == [] for c in chunk_with(structure))


class TestGopQuaRanhGioiSectionThiPhaiTutXuongToTienChung:
    """Chỗ mà `section_path` dễ nói dối nhất."""

    def test_common_ancestor_la_tien_to_chung_dai_nhat(self) -> None:
        assert common_ancestor(["A", "B", "C"], ["A", "B", "D"]) == ["A", "B"]
        assert common_ancestor(["A"], ["B"]) == []
        assert common_ancestor([], ["A"]) == []
        assert common_ancestor(["A", "B"], ["A", "B"]) == ["A", "B"]

    def test_hai_section_ngan_gop_lai_thi_path_tut_xuong_cha_chung(self) -> None:
        text = "# Chương I\n\n## Điều 3\n\nNgắn.\n\n## Điều 4\n\nCũng ngắn.\n"
        structure = loaded(
            text,
            Heading(1, "Chương I", text.index("Chương I")),
            Heading(2, "Điều 3", text.index("Điều 3")),
            Heading(2, "Điều 4", text.index("Điều 4")),
        )
        chunks = chunk_with(structure, min_chunk_size=200, max_chunk_size=2000)

        assert len(chunks) == 1
        assert "Điều 3" in chunks[0].content and "Điều 4" in chunks[0].content
        assert chunks[0].section_path == ["Chương I"], (
            "chunk chứa cả hai điều mà lại khai đúng một điều thì citation nói dối"
        )

    def test_tat_gop_thi_moi_section_giu_duong_dan_rieng(self) -> None:
        text = "# Chương I\n\n## Điều 3\n\nNgắn.\n\n## Điều 4\n\nCũng ngắn.\n"
        structure = loaded(
            text,
            Heading(1, "Chương I", text.index("Chương I")),
            Heading(2, "Điều 3", text.index("Điều 3")),
            Heading(2, "Điều 4", text.index("Điều 4")),
        )
        chunks = chunk_with(structure, min_chunk_size=200, structure_merge_short_sections=False)
        assert [c.section_path[-1] for c in chunks] == ["Chương I", "Điều 3", "Điều 4"]

    def test_moi_chunk_gop_van_phai_nhat_quan_voi_section_path_at(self) -> None:
        structure = three_level()
        for chunk in chunk_with(structure, min_chunk_size=1000, max_chunk_size=4000):
            assert path_is_truthful(chunk, structure)


class TestRanhGioiCatLuiVeDauDong:
    def test_marker_markdown_di_cung_heading_cua_no(self) -> None:
        chunks = chunk_with(three_level(), min_chunk_size=0)
        assert chunks[1].content.startswith("## Điều 1")
        assert not chunks[0].content.rstrip().endswith("#")

    def test_section_boundaries_tra_cap_cat_va_cho_hoi(self) -> None:
        """Hai số phải **khác** nhau: cắt ở `#`, hỏi ở chữ đầu của heading."""
        structure = three_level()
        pairs = section_boundaries(structure)
        cut, probe = pairs[1]
        assert structure.text[cut] == "#"
        assert structure.text.startswith("Điều 1", probe)
        assert cut < probe

    def test_hoi_tai_vi_tri_cat_se_tra_ve_section_TRUOC(self) -> None:
        """Ghim lại cái bẫy — nếu docling đổi cách xuất marker, test này sẽ nói."""
        structure = three_level()
        cut, probe = section_boundaries(structure)[1]
        assert structure.section_path_at(cut) == ("Chương I",)
        assert structure.section_path_at(probe) == ("Chương I", "Điều 1")


class TestMotHeadingKhongDinhViDuocKhongLamMuPhanConLai:
    """Lỗi `break` của `W3-01`, phát hiện khi `W3-03` là chỗ tiêu thụ đầu tiên.

    Trên `wb1.pdf` (129 trang, 6/128 heading không định vị được) bản cũ trả
    đường dẫn khác ở **575/587 chunk**.
    """

    def test_heading_sau_mot_heading_mat_tich_van_co_hieu_luc(self) -> None:
        text = "# Chương I\n\nThân.\n\n## Điều 4\n\nThân điều bốn.\n"
        structure = loaded(
            text,
            Heading(1, "Chương I", text.index("Chương I")),
            Heading(2, "Điều 3", -1),  # docling không định vị được
            Heading(2, "Điều 4", text.index("Điều 4")),
        )
        assert structure.section_path_at(len(text) - 1) == ("Chương I", "Điều 4")

    def test_chunker_van_gan_duoc_path_cho_section_sau_do(self) -> None:
        text = "# Chương I\n\nThân đủ dài để đứng riêng một chunk.\n\n## Điều 4\n\nThân điều bốn cũng đủ dài.\n"
        structure = loaded(
            text,
            Heading(1, "Chương I", text.index("Chương I")),
            Heading(2, "Điều 3", -1),
            Heading(2, "Điều 4", text.index("Điều 4")),
        )
        chunks = chunk_with(structure, min_chunk_size=0)
        assert chunks[-1].section_path == ["Chương I", "Điều 4"]


class TestKhongBindThiPhaiDemChuKhongImLang:
    def test_dem_va_canh_bao(self, caplog: pytest.LogCaptureFixture) -> None:
        chunker = StructureChunker(ChunkingConfig(strategy=ChunkingStrategy.STRUCTURE))
        with caplog.at_level(logging.WARNING, logger="rag_core.chunking.structure"):
            chunks = chunker.chunk([document("Nội dung không khai cấu trúc. " * 40)])

        assert chunker.documents_without_structure == 1
        assert all(c.section_path == [] for c in chunks)
        assert "chưa `bind`" in caplog.text

    def test_chunk_loaded_khong_the_quen_bind(self) -> None:
        structure = three_level()
        chunker = StructureChunker(
            ChunkingConfig(strategy=ChunkingStrategy.STRUCTURE, min_chunk_size=0)
        )
        chunks = chunker.chunk_loaded(structure, doc_id="doc", metadata=METADATA)

        assert chunker.documents_without_structure == 0
        assert chunks[-1].section_path == ["Chương I", "Điều 1", "Khoản a"]


class TestContentKhacTextThiOffsetHetGiaTri:
    """Guard quan trọng nhất: offset heading chỉ đúng trong `LoadedDocument.text`."""

    def test_lech_thi_roi_ve_cat_theo_ky_tu_va_bi_dem(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        structure = three_level()
        chunker = StructureChunker(
            ChunkingConfig(strategy=ChunkingStrategy.STRUCTURE, min_chunk_size=0)
        )
        chunker.bind("doc", structure)

        # Một bước "chuẩn hoá" vô hại: đổi CRLF. Mọi offset lệch từ đó trở đi.
        drifted = "\r\n".join(structure.text.split("\n"))
        with caplog.at_level(logging.ERROR, logger="rag_core.chunking.structure"):
            chunks = chunker.chunk([document(drifted)])

        assert chunker.documents_with_mismatched_text == 1
        assert all(c.section_path == [] for c in chunks)
        assert "offset heading không còn dùng được" in caplog.text


class TestCacChunkerKhacKhongDoiHanhVi:
    """Tách `_prepare_pieces`/`_section_path_for` ra khỏi `chunk` không được đổi gì."""

    def test_fixed_chunker_van_tra_section_path_rong(self) -> None:
        chunks = FixedSizeChunker(ChunkingConfig()).chunk([document("Câu văn. " * 400)])
        assert chunks and all(c.section_path == [] for c in chunks)

    def test_build_chunker_map_duoc_strategy_moi(self) -> None:
        chunker = build_chunker(ChunkingConfig(strategy=ChunkingStrategy.STRUCTURE))
        assert isinstance(chunker, StructureChunker)
        assert chunker.name.startswith("structure:")


@pytest.mark.parametrize("name", ["chuong-i.md", "chuong-i.html", "chuong-i.docx"])
class TestBaDinhDangNguonRaCungMotSectionPath:
    """Payoff của `_normalise_depth`: cùng nội dung, ba định dạng, một đường dẫn.

    `.md` xuất `#`/`##`/`###` còn `.docx` xuất `##`/`###`/`####` — số dấu thăng
    khác nhau mà `section_path` phải giống nhau. Đây là hạng mục chứng minh
    `W3-03` **không** dựng quy tắc độ sâu thứ hai bằng cách dò `#`.
    """

    @pytest.mark.integration
    def test_ba_cap_va_khop_voi_section_path_at(self, name: str) -> None:
        from rag_core.loaders import load_document

        structure = load_document(FIXTURES / name)
        chunks = chunk_with(structure, min_chunk_size=0, structure_merge_short_sections=False)

        depths = [len(c.section_path) for c in chunks]
        assert depths == [1, 2, 3], f"{name}: {depths}"
        assert [c.section_path[-1] for c in chunks][1:] == [
            "Điều 1. Phạm vi điều chỉnh",
            "Khoản 1. Chỉ tiêu vĩ mô",
        ]
        for chunk in chunks:
            assert path_is_truthful(chunk, structure)
