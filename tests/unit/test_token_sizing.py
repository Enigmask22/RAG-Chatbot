"""`W3-06` — kích thước chunk tính bằng token, và trần token là bảo đảm cứng.

DoD đòi *"không chunk nào vượt max token của model"* trên **3 corpus mẫu**. Ba
mẫu ở đây chọn theo ba **chế độ hỏng** khác nhau chứ không phải ba đoạn văn khác
nhau:

1. `EN_PROSE` — văn xuôi bình thường, nhiều separator. Splitter đệ quy dư sức.
2. `VI_PROSE` — cùng độ dài ký tự nhưng **mật độ token khác**, để bắt lỗi hiệu
   chuẩn một lần rồi dùng chung cho mọi tài liệu.
3. `NO_SEPARATOR` — một khối chữ liền không khoảng trắng. Đây là mẫu duy nhất
   thật sự kiểm được **điều kiện dừng**: splitter đệ quy trả về nguyên mảnh, nên
   nếu không có đường cắt cứng thì `fit_to_budget` lặp vô hạn.

Bộ đếm token ở đây là giả và **xác định** — cố ý. Dùng tokenizer thật thì test
phụ thuộc trọng số 2,2 GB và không chạy trong CI; và thứ cần ghim là *thuật toán
cắt lại*, không phải *tokenizer của HuggingFace*. Con số thật đo bằng tokenizer
thật nằm ở `reports/tasks/w3-06-token-sizing.md`.
"""

from __future__ import annotations

import itertools
import string
from collections.abc import Sequence

import pytest

from rag_core.chunking import (
    ChunkingConfig,
    ChunkingStrategy,
    FixedSizeChunker,
    StructureChunker,
    build_chunker,
    calibrate_density,
    fit_to_budget,
)
from rag_core.chunking.pieces import TextPiece
from rag_core.chunking.tokens import TokenCounter, TokenSizingUnavailable, _slices
from rag_core.loaders.base import Heading, LoadedDocument, ParseFingerprint
from rag_core.schemas import Document, DocumentMetadata

METADATA = DocumentMetadata(source_url="https://example.org/x", license="CC BY 4.0")

SPECIAL_TOKENS = 2
"""`[CLS]` + `[SEP]`, đúng theo hợp đồng `EmbeddingProvider.count_tokens`."""


class FakeCounter:
    """Đếm token = `ceil(len/chars_per_token)` + special token. Xác định, không cần model."""

    def __init__(self, chars_per_token: float = 4.0, limit: int | None = 64) -> None:
        self.chars_per_token = chars_per_token
        self._limit = limit
        self.calls = 0

    @property
    def max_sequence_tokens(self) -> int | None:
        return self._limit

    def count_tokens(self, texts: Sequence[str]) -> list[int] | None:
        self.calls += 1
        return [
            -(-len(t) // int(self.chars_per_token)) + SPECIAL_TOKENS if t else SPECIAL_TOKENS
            for t in texts
        ]


class BlindCounter:
    """Provider không đếm được token — `count_tokens` trả `None` theo hợp đồng."""

    @property
    def max_sequence_tokens(self) -> int | None:
        return 256

    def count_tokens(self, texts: Sequence[str]) -> list[int] | None:
        return None


EN_PROSE = (
    "The report examines how public research institutes and private firms "
    "interact in the national innovation system. "
) * 60

VI_PROSE = (
    "Báo cáo phân tích cách các viện nghiên cứu công lập và doanh nghiệp tư nhân "
    "tương tác trong hệ thống đổi mới sáng tạo quốc gia. "
) * 60

NO_SEPARATOR = "x" * 12_000

CORPORA = {"en": EN_PROSE, "vi": VI_PROSE, "no-separator": NO_SEPARATOR}


def token_config(**overrides: object) -> ChunkingConfig:
    base: dict[str, object] = {
        "strategy": ChunkingStrategy.FIXED,
        "size_unit": "tokens",
        "chunk_size": 48,
        "chunk_overlap": 4,
        "min_chunk_size": 8,
        "max_chunk_size": 64,
        "neighbor_context_chars": 0,
    }
    base.update(overrides)
    return ChunkingConfig(**base)  # type: ignore[arg-type]


def counted_any(counter: TokenCounter, texts: Sequence[str]) -> list[int]:
    counts = counter.count_tokens(texts)
    assert counts is not None
    return counts


def counted(counter: FakeCounter, texts: Sequence[str]) -> list[int]:
    counts = counter.count_tokens(texts)
    assert counts is not None
    return counts


@pytest.mark.parametrize("name", sorted(CORPORA))
class TestDoDKhongChunkNaoVuotTranToken:
    def test_moi_chunk_nam_trong_cua_so_cua_model(self, name: str) -> None:
        counter = FakeCounter(limit=64)
        chunker = FixedSizeChunker(token_config(), token_counter=counter)
        chunks = chunker.chunk([Document(doc_id=name, content=CORPORA[name], metadata=METADATA)])

        assert chunks
        over = [n for n in counted(counter, [c.content for c in chunks]) if n > 64]
        assert over == [], f"{name}: {len(over)} chunk vượt trần, lớn nhất {max(over, default=0)}"

    def test_span_khong_bo_sot_ky_tu_NOI_DUNG_nao(self, name: str) -> None:
        """Cắt lại vì trần token không được làm bốc hơi nội dung.

        Khẳng định chính xác là **không bỏ sót ký tự nội dung**, chứ không phải
        "ghép lại bằng nguyên bản". Splitter đệ quy bỏ ký tự separator ở mỗi mối
        nối — cắt theo `". "` thì dấu chấm ở chỗ cắt biến mất khỏi cả `content`
        lẫn span. Đó là hành vi **có sẵn từ `W1-11`**, không phải do `W3-06`: chế
        độ ký tự với `chunk_size=200` mất đúng 60 dấu chấm trên cùng văn bản này.
        Xem `test_mat_dau_cham_o_moi_noi_la_hanh_vi_co_san`.
        """
        text = CORPORA[name]
        counter = FakeCounter(limit=64)
        chunker = FixedSizeChunker(token_config(), token_counter=counter)
        chunks = chunker.chunk([Document(doc_id=name, content=text, metadata=METADATA)])

        covered: set[int] = set()
        for chunk in chunks:
            assert chunk.start_char is not None and chunk.end_char is not None
            covered |= set(range(chunk.start_char, chunk.end_char))
        dropped = {text[i] for i in range(len(text)) if i not in covered}
        allowed = set(string.whitespace) | {"."}
        assert dropped <= allowed, f"{name}: bỏ sót ký tự nội dung {sorted(dropped)}"

    def test_ghep_lai_dung_nguyen_ban_khi_khong_co_moi_noi(self, name: str) -> None:
        """Không separator thì phép ghép phải khít tuyệt đối."""
        if name != "no-separator":
            pytest.skip("chỉ mẫu không separator mới ghép khít được")
        counter = FakeCounter(limit=64)
        chunker = FixedSizeChunker(token_config(chunk_overlap=0), token_counter=counter)
        chunks = chunker.chunk([Document(doc_id=name, content=CORPORA[name], metadata=METADATA)])
        assert "".join(c.content for c in chunks) == CORPORA[name]


class TestKichThuocThucSuBAMVaoNganSachToken:
    """Trần token là bảo đảm; **đích** `chunk_size` mới là thứ dễ trượt âm thầm.

    Bộ test đầu của `W3-06` chỉ kiểm trần, và nó **xanh** trong khi `FixedSizeChunker`
    vẫn đọc `self.config.chunk_size` (tức đọc 256 **token** như thể là 256 **ký
    tự**) — vì trần do một đường code khác (`_fit_tokens`) bảo đảm. Trên corpus
    thật, chunk tiếng Việt ra p50 = 71 token thay vì 256. Test này ghim cái đích.
    """

    @pytest.mark.parametrize("name", sorted(CORPORA))
    def test_p50_bam_sat_chunk_size_da_khai_bao(self, name: str) -> None:
        counter = FakeCounter(chars_per_token=4.0, limit=512)
        config = token_config(chunk_size=48, min_chunk_size=8, max_chunk_size=96)
        chunker = FixedSizeChunker(config, token_counter=counter)
        chunks = chunker.chunk([Document(doc_id=name, content=CORPORA[name], metadata=METADATA)])

        counts = sorted(counted(counter, [c.content for c in chunks]))
        p50 = counts[len(counts) // 2]
        assert 0.5 * config.chunk_size <= p50 <= 1.2 * config.chunk_size, (
            f"{name}: p50 = {p50} token, xin {config.chunk_size} — chunker đang trộn đơn vị"
        )

    def test_moi_chunker_deu_doc_sizing_chu_khong_doc_config(self) -> None:
        """Ghim bằng chính source: `self.config.<kích thước>` là lỗi trộn đơn vị."""
        import re
        from pathlib import Path as _Path

        pattern = re.compile(
            r"self\.config\.(chunk_size|chunk_overlap|min_chunk_size|max_chunk_size|separators)"
        )
        root = _Path(__file__).resolve().parents[2] / "packages" / "rag_core" / "chunking"
        offenders = []
        for path in sorted(root.glob("*.py")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                # `_begin_sizing` cố ý đọc con số KHAI BÁO để nhân với mật độ.
                if pattern.search(line) and "density" not in line and "limit = " not in line:
                    offenders.append(f"{path.name}:{number}")
        assert offenders == [], f"đọc self.config thay vì self.sizing: {offenders}"


class TestTranCungThapHonThiModelThang:
    def test_max_chunk_size_lon_hon_cua_so_model_bi_ha_xuong(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        counter = FakeCounter(limit=32)
        chunker = FixedSizeChunker(
            token_config(chunk_size=100, max_chunk_size=200), token_counter=counter
        )
        with caplog.at_level("WARNING", logger="rag_core.chunking.base"):
            chunks = chunker.chunk([Document(doc_id="d", content=EN_PROSE, metadata=METADATA)])

        assert max(counted(counter, [c.content for c in chunks])) <= 32
        assert "vượt cửa sổ của model" in caplog.text


class TestThieuBoDemThiPhaiNemLoi:
    def test_khong_co_token_counter(self) -> None:
        chunker = FixedSizeChunker(token_config())
        with pytest.raises(TokenSizingUnavailable, match="token_counter"):
            chunker.chunk([Document(doc_id="d", content=EN_PROSE, metadata=METADATA)])

    def test_provider_khong_dem_duoc_token(self) -> None:
        chunker = FixedSizeChunker(token_config(), token_counter=BlindCounter())
        with pytest.raises(TokenSizingUnavailable, match="count_tokens"):
            chunker.chunk([Document(doc_id="d", content=EN_PROSE, metadata=METADATA)])

    def test_khong_lang_le_roi_ve_dem_ky_tu(self) -> None:
        """Rơi về sẽ cho ra bộ chunk hợp lệ nhưng khác hẳn cái được yêu cầu."""
        chunker = FixedSizeChunker(token_config())
        with pytest.raises(TokenSizingUnavailable):
            chunker.chunk([Document(doc_id="d", content=EN_PROSE, metadata=METADATA)])


class TestChuoiKyTuKhongCoSeparator:
    """Mẫu duy nhất kiểm được điều kiện dừng."""

    def test_cat_cung_theo_ky_tu_thay_vi_lap_vo_han(self) -> None:
        counter = FakeCounter(limit=64)
        piece = TextPiece(NO_SEPARATOR, 0, len(NO_SEPARATOR))
        fitted = fit_to_budget(
            [piece], limit=64, counter=counter, separators=["\n\n", "\n", ". ", " "]
        )

        assert len(fitted) > 1
        assert all(source == 0 for source, _ in fitted)
        assert max(counted(counter, [p.text for _, p in fitted])) <= 64

    def test_span_van_liên_tuc_va_khop_van_ban_goc(self) -> None:
        counter = FakeCounter(limit=64)
        piece = TextPiece(NO_SEPARATOR, 100, 100 + len(NO_SEPARATOR))
        fitted = [p for _, p in fit_to_budget([piece], limit=64, counter=counter, separators=[" "])]

        assert fitted[0].start == 100
        assert fitted[-1].end == 100 + len(NO_SEPARATOR)
        for left, right in itertools.pairwise(fitted):
            assert left.end == right.start


class TestMatDauChamOMoiNoiLaHanhViCoSan:
    """Ghim lại rằng chỗ mất mát đó có từ `W1-11`, không phải do `W3-06` gây ra.

    Nếu ai đó sửa splitter để giữ separator thì test này đỏ — và đó là tin tốt,
    nhưng nó đổi **mọi** bộ chunk đã công bố, nên phải là một quyết định có báo
    cáo chứ không phải một lần dọn dẹp.
    """

    def test_che_do_ky_tu_mat_dung_bay_nhieu_dau_cham(self) -> None:
        chunker = FixedSizeChunker(
            ChunkingConfig(chunk_size=200, chunk_overlap=0, min_chunk_size=0, max_chunk_size=200)
        )
        chunks = chunker.chunk([Document(doc_id="d", content=EN_PROSE, metadata=METADATA)])
        rebuilt = "".join("".join(c.content.split()) for c in chunks)
        source = "".join(EN_PROSE.split())
        assert len(source) - len(rebuilt) == EN_PROSE.count(". ")


class TestChiSoManhGocDeMangSongSongGianTheo:
    def test_ba_manh_con_deu_tro_ve_manh_me(self) -> None:
        counter = FakeCounter(limit=32)
        pieces = [TextPiece("a" * 20, 0, 20), TextPiece("b" * 400, 20, 420)]
        fitted = fit_to_budget([*pieces], limit=32, counter=counter, separators=[""])

        sources = [source for source, _ in fitted]
        assert sources[0] == 0
        assert sources.count(1) > 1
        assert sources == sorted(sources)

    def test_structure_chunker_giu_dung_section_path_sau_khi_cat_lai(self) -> None:
        text = "# Chương I\n\n" + ("Thân của chương một. " * 200)
        structure = LoadedDocument(
            text=text,
            source_sha256="0" * 64,
            fingerprint=ParseFingerprint("test", "test", "1"),
            headings=(Heading(1, "Chương I", text.index("Chương I")),),
        )
        counter = FakeCounter(limit=64)
        chunker = StructureChunker(
            token_config(strategy=ChunkingStrategy.STRUCTURE), token_counter=counter
        )
        chunker.bind("doc", structure)
        chunks = chunker.chunk([Document(doc_id="doc", content=text, metadata=METADATA)])

        assert len(chunks) > 1
        assert all(c.section_path == ["Chương I"] for c in chunks)
        assert max(counted(counter, [c.content for c in chunks])) <= 64


class WordCounter:
    """Đếm token kiểu "mỗi từ một token" — mật độ **đổi theo nội dung**.

    `FakeCounter` cho mật độ hằng số nên không kiểm được gì về hiệu chuẩn: mọi
    lát cắt đều ra cùng một con số. Bộ đếm này cho `"A B C D "` ra 2 ký tự/token
    còn một từ dài 29 ký tự ra 29 ký tự/token.
    """

    @property
    def max_sequence_tokens(self) -> int | None:
        return 512

    def count_tokens(self, texts: Sequence[str]) -> list[int] | None:
        return [max(1, len(t.split())) + SPECIAL_TOKENS for t in texts]


DENSE_HEAD = "A B C D " * 400
SPARSE_BODY = "khongkhoangtrangchutdaidangke " * 400


class TestHieuChuanMatDo:
    def test_lay_mau_rai_deu_chu_khong_chi_dau_tai_lieu(self) -> None:
        """Đầu tài liệu là bìa/mục lục; mật độ ở đó không đại diện cho phần thân."""
        counter = WordCounter()
        head_only = calibrate_density(DENSE_HEAD, counter, samples=12, sample_chars=200)
        body_only = calibrate_density(SPARSE_BODY, counter, samples=12, sample_chars=200)
        whole = calibrate_density(DENSE_HEAD + SPARSE_BODY, counter, samples=12, sample_chars=200)

        assert head_only < body_only
        assert head_only < whole < body_only, (
            f"hiệu chuẩn ra {whole:.2f}, ngoài khoảng [{head_only:.2f}, {body_only:.2f}] — "
            "nghĩa là nó chỉ nhìn một đầu tài liệu"
        )

    def test_dung_TRUNG_BINH_chu_khong_phai_phan_vi_thap(self) -> None:
        """Ngược với `truncation.py`, và đó là chủ ý — xem docstring `tokens.py`.

        Phân vị thấp sẽ bám lát cắt dày token nhất, tức xấp xỉ `min`. Trung bình
        thì phải nằm hẳn phía trên đó.
        """
        counter = WordCounter()
        text = DENSE_HEAD + SPARSE_BODY
        density = calibrate_density(text, counter, samples=12, sample_chars=200)

        slice_densities = sorted(
            len(part) / count
            for part, count in zip(
                _slices(text, 12, 200),
                counted_any(counter, _slices(text, 12, 200)),
                strict=True,
            )
        )
        assert density > slice_densities[1], (
            f"hiệu chuẩn ra {density:.2f}, không cao hơn phân vị thấp "
            f"{slice_densities[1]:.2f} — đang chừa biên an toàn mà `fit_to_budget` "
            "đã lo rồi"
        )

    def test_tai_lieu_ngan_hon_mot_lat_van_hieu_chuan_duoc(self) -> None:
        assert calibrate_density("một câu ngắn", WordCounter(), sample_chars=2000) > 0

    def test_bo_dem_mu_thi_nem_loi(self) -> None:
        with pytest.raises(TokenSizingUnavailable, match="mật độ"):
            calibrate_density(EN_PROSE, BlindCounter())


class TestNgucCanhHangXomVaTranTokenLoaiTruNhau:
    def test_config_tu_choi_ngay_luc_dung(self) -> None:
        with pytest.raises(ValueError, match="mâu thuẫn"):
            token_config(neighbor_context_chars=100)

    def test_che_do_ky_tu_van_dung_duoc_ngu_canh_hang_xom(self) -> None:
        config = ChunkingConfig(neighbor_context_chars=100)
        assert config.neighbor_context_chars == 100


class TestCheDoKyTuKhongDoiGiCa:
    """Mặc định phải giữ nguyên mọi con số đã công bố."""

    def test_size_unit_mac_dinh_la_chars(self) -> None:
        assert ChunkingConfig().size_unit == "chars"

    def test_co_token_counter_nhung_van_dem_ky_tu(self) -> None:
        counter = FakeCounter(limit=8)  # trần rất thấp — nếu bị áp thì kết quả sẽ khác
        text = "Câu văn dài. " * 300
        with_counter = FixedSizeChunker(ChunkingConfig(), token_counter=counter).chunk(
            [Document(doc_id="d", content=text, metadata=METADATA)]
        )
        without = FixedSizeChunker(ChunkingConfig()).chunk(
            [Document(doc_id="d", content=text, metadata=METADATA)]
        )
        assert [c.content for c in with_counter] == [c.content for c in without]
        assert counter.calls == 0, "chế độ ký tự không được gọi tokenizer lần nào"

    def test_sizing_bang_config_o_che_do_ky_tu(self) -> None:
        chunker = FixedSizeChunker(ChunkingConfig(chunk_size=777))
        assert chunker.sizing is chunker.config

    def test_sizing_tra_ve_config_sau_moi_tai_lieu(self) -> None:
        """Trạng thái theo tài liệu để sót lại là tài liệu sau bị chunk sai mật độ."""
        counter = FakeCounter(limit=64)
        chunker = FixedSizeChunker(token_config(), token_counter=counter)
        chunker.chunk([Document(doc_id="d", content=EN_PROSE, metadata=METADATA)])
        assert chunker.sizing is chunker.config


class TestBuildChunkerChuyenTiepBoDem:
    def test_embeddings_tro_thanh_token_counter(self) -> None:
        counter = FakeCounter(limit=64)
        chunker = build_chunker(token_config(), counter)  # type: ignore[arg-type]
        chunks = chunker.chunk([Document(doc_id="d", content=EN_PROSE, metadata=METADATA)])
        assert max(counted(counter, [c.content for c in chunks])) <= 64

    def test_embedding_provider_thoa_giao_thuc_token_counter(self) -> None:
        from rag_core.embedding.hashing import HashingEmbeddingProvider

        assert isinstance(HashingEmbeddingProvider(dimension=32), TokenCounter)
