"""`W3-05` — small-to-big: tìm bằng child, đọc bằng parent, và gộp trùng.

DoD: *retrieve child → context assembly trả parent, dedupe parent trùng.*

Hai nửa được test riêng vì chúng hỏng theo hai kiểu khác nhau: chunker hỏng thì
quan hệ cha-con sai ngay lúc index (và phải build lại mới sửa được); assembly
hỏng thì quan hệ vẫn đúng mà prompt sai (sửa được mà không đụng index).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import pytest

from rag_core.chunking import ChunkingConfig, ChunkingStrategy, ParentChildChunker, build_chunker
from rag_core.chunking.parent_child import (
    PARENT_CHILDREN_KEY,
    PARENT_END_KEY,
    PARENT_START_KEY,
    parent_id,
)
from rag_core.retrieval import AssembledParent, assemble_text, expand_to_parents
from rag_core.retrieval.filters import FilterSpec
from rag_core.retrieval.qdrant_store import QdrantDenseRetriever
from rag_core.schemas import Chunk, Document, DocumentMetadata, RetrievedChunk

METADATA = DocumentMetadata(source_url="https://example.org/doc", license="CC BY 4.0")

# 60 câu **khác nhau**, mỗi câu ~95 ký tự.
#
# ⚠️ Bản đầu của fixture này lặp đúng một câu 60 lần, và ba phép kiểm "đoạn văn
# không xuất hiện hai lần trong parent" trở thành vô nghĩa — `str.count` đếm được
# 60 bản sao của mọi thứ. Cùng bài học với fixture PDF hai cột ở `W3-01` §2: một
# fixture đều tăm tắp đo cái generator, không đo cái cần đo.
PROSE = "".join(
    f"Đoạn {i:02d}: chương trình hợp tác tập trung vào năng lực đổi mới của doanh nghiệp. "
    for i in range(60)
)


def config(**overrides: object) -> ChunkingConfig:
    base: dict[str, object] = {
        "strategy": ChunkingStrategy.PARENT_CHILD,
        "chunk_size": 200,
        "chunk_overlap": 20,
        "min_chunk_size": 50,
        "max_chunk_size": 400,
        "parent_size_multiple": 4,
        "neighbor_context_chars": 0,
    }
    base.update(overrides)
    return ChunkingConfig(**base)  # type: ignore[arg-type]


def chunk_prose(text: str = PROSE, **overrides: object) -> list[Chunk]:
    chunker = ParentChildChunker(config(**overrides))
    return chunker.chunk([Document(doc_id="doc", content=text, metadata=METADATA)])


def by_parent(chunks: Sequence[Chunk]) -> dict[str | None, list[Chunk]]:
    groups: dict[str | None, list[Chunk]] = {}
    for chunk in chunks:
        groups.setdefault(chunk.parent_chunk_id, []).append(chunk)
    return groups


def retrieved(chunks: Sequence[Chunk]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk=chunk, score=1.0 - 0.01 * rank, rank=rank)
        for rank, chunk in enumerate(chunks, start=1)
    ]


class RecordingFetcher:
    """`ChunkFetcher` giả, ghi lại filter đã nhận và giấu được chunk theo yêu cầu."""

    def __init__(self, chunks: Sequence[Chunk], *, hidden: Sequence[str] = ()) -> None:
        self.index = {c.chunk_id: c for c in chunks}
        self.hidden = set(hidden)
        self.calls: list[tuple[tuple[str, ...], FilterSpec]] = []

    def fetch_chunks(
        self, chunk_ids: Sequence[str], *, filters: FilterSpec = None
    ) -> Mapping[str, Chunk]:
        self.calls.append((tuple(chunk_ids), filters))
        return {
            cid: self.index[cid]
            for cid in chunk_ids
            if cid in self.index and cid not in self.hidden
        }


class TestChunkerDungQuanHeChaCon:
    def test_moi_child_deu_co_parent_va_danh_sach_anh_em(self) -> None:
        chunks = chunk_prose()
        assert len(chunks) > 4
        for chunk in chunks:
            assert chunk.parent_chunk_id is not None
            siblings = chunk.extra[PARENT_CHILDREN_KEY]
            assert chunk.chunk_id in siblings

    def test_danh_sach_anh_em_khop_voi_nhom_that(self) -> None:
        chunks = chunk_prose()
        for parent, members in by_parent(chunks).items():
            ids = [c.chunk_id for c in members]
            for chunk in members:
                assert chunk.extra[PARENT_CHILDREN_KEY] == ids, parent

    def test_parent_lon_hon_child_va_gan_dung_boi_so(self) -> None:
        chunks = chunk_prose()
        groups = by_parent(chunks)
        assert len(groups) >= 2
        # Không đòi đúng 4 child/parent: splitter cắt theo separator nên biên
        # thực tế dao động. Đòi parent thật sự GOM nhiều child lại.
        assert sum(len(m) for m in groups.values()) / len(groups) >= 2.0

    def test_khoa_gom_nhom_khong_dung_do_khong_gian_ten_voi_chunk_id(self) -> None:
        chunks = chunk_prose()
        child_ids = {c.chunk_id for c in chunks}
        parents = {c.parent_chunk_id for c in chunks}
        assert parents & child_ids == set()
        assert all(p is not None and "::p" in p for p in parents)
        assert parent_id("doc", 0) == "doc::p00000"

    def test_span_parent_bao_tron_span_moi_child(self) -> None:
        for members in by_parent(chunk_prose()).values():
            low = min(c.start_char or 0 for c in members)
            high = max(c.end_char or 0 for c in members)
            for chunk in members:
                assert chunk.extra[PARENT_START_KEY] == low
                assert chunk.extra[PARENT_END_KEY] == high


class TestChildTrongMotParentKhongChongLan:
    """Overlap giữa anh em làm đoạn chồng lấn xuất hiện HAI LẦN trong parent ghép."""

    def test_span_cac_child_khong_giao_nhau(self) -> None:
        for parent, members in by_parent(chunk_prose()).items():
            ordered = sorted(members, key=lambda c: c.chunk_index)
            for left, right in itertools.pairwise(ordered):
                assert (left.end_char or 0) <= (right.start_char or 0), parent

    def test_parent_ghep_lai_khong_lap_doan_van(self) -> None:
        for members in by_parent(chunk_prose()).values():
            text = assemble_text(members)
            for chunk in members:
                assert text.count(chunk.content) == 1

    def test_overlap_giua_cac_PARENT_thi_van_giu(self) -> None:
        """`chunk_overlap` chưa bị vô hiệu — nó chỉ chuyển lên cấp parent."""
        assert config().chunk_overlap == 20


class TestDoDChildTruyHoiRaParent:
    def test_mot_child_trung_thi_tra_ve_ca_parent(self) -> None:
        chunks = chunk_prose()
        group = next(m for m in by_parent(chunks).values() if len(m) >= 3)
        fetcher = RecordingFetcher(chunks)

        parents = expand_to_parents(retrieved([group[1]]), fetcher)

        assert len(parents) == 1
        assert parents[0].complete
        assert parents[0].children == tuple(sorted(group, key=lambda c: c.chunk_index))
        for sibling in group:
            assert sibling.content in parents[0].text

    def test_parent_dai_hon_child_da_trung(self) -> None:
        chunks = chunk_prose()
        group = next(m for m in by_parent(chunks).values() if len(m) >= 3)
        parents = expand_to_parents(retrieved([group[1]]), RecordingFetcher(chunks))
        assert parents[0].expansion_ratio > 2.0

    def test_giu_thu_hang_theo_child_tot_nhat_cua_nhom(self) -> None:
        groups = list(by_parent(chunk_prose()).values())
        # hạng 1 thuộc nhóm thứ hai, hạng 2 thuộc nhóm thứ nhất
        hits = [
            RetrievedChunk(chunk=groups[1][0], score=0.9, rank=1),
            RetrievedChunk(chunk=groups[0][0], score=0.8, rank=2),
        ]
        parents = expand_to_parents(hits)
        assert [p.best_rank for p in parents] == [1, 2]
        assert parents[0].parent_id == groups[1][0].parent_chunk_id


class TestGopTrungParent:
    def test_ba_child_cung_parent_ra_MOT_parent(self) -> None:
        chunks = chunk_prose()
        group = next(m for m in by_parent(chunks).values() if len(m) >= 3)
        parents = expand_to_parents(retrieved(group[:3]), RecordingFetcher(chunks))

        assert len(parents) == 1
        assert parents[0].hit_children == tuple(c.chunk_id for c in group[:3])

    def test_van_ban_parent_khong_lap_du_ba_child_cung_trung(self) -> None:
        chunks = chunk_prose()
        group = next(m for m in by_parent(chunks).values() if len(m) >= 3)
        parents = expand_to_parents(retrieved(group[:3]), RecordingFetcher(chunks))
        for chunk in group:
            assert parents[0].text.count(chunk.content) == 1

    def test_hai_parent_khac_nhau_khong_bi_gop(self) -> None:
        groups = list(by_parent(chunk_prose()).values())
        hits = retrieved([groups[0][0], groups[1][0]])
        parents = expand_to_parents(hits)
        assert len({p.parent_id for p in parents}) == 2

    def test_max_parents_cat_theo_thu_hang(self) -> None:
        groups = list(by_parent(chunk_prose()).values())
        hits = retrieved([g[0] for g in groups[:3]])
        parents = expand_to_parents(hits, max_parents=2)
        assert [p.best_rank for p in parents] == [1, 2]


class TestFilterPhaiDiXuyenQua:
    """`fetch_chunks` là một đường vòng qua filter — `W1-07` đã cảnh báo, đây là chỗ dùng."""

    def test_filter_duoc_chuyen_nguyen_van_xuong_fetcher(self) -> None:
        chunks = chunk_prose()
        group = next(m for m in by_parent(chunks).values() if len(m) >= 3)
        fetcher = RecordingFetcher(chunks)
        spec = {"tenant_id": "acme"}

        expand_to_parents(retrieved([group[0]]), fetcher, filters=spec)

        assert fetcher.calls, "không gọi fetch_chunks thì anh em không bao giờ được lấy"
        assert all(call_filters == spec for _, call_filters in fetcher.calls)

    def test_anh_em_bi_filter_giau_thi_parent_KHONG_hoan_chinh(self) -> None:
        chunks = chunk_prose()
        group = next(m for m in by_parent(chunks).values() if len(m) >= 3)
        hidden = group[-1].chunk_id
        fetcher = RecordingFetcher(chunks, hidden=[hidden])

        parents = expand_to_parents(retrieved([group[0]]), fetcher, filters={"tenant_id": "acme"})

        assert parents[0].complete is False
        assert parents[0].missing_children == (hidden,)
        assert group[-1].content not in parents[0].text

    def test_khong_co_fetcher_thi_chi_ghep_cai_da_co_va_danh_dau(self) -> None:
        chunks = chunk_prose()
        group = next(m for m in by_parent(chunks).values() if len(m) >= 3)
        parents = expand_to_parents(retrieved([group[0]]))
        assert parents[0].complete is False
        assert set(parents[0].missing_children) == {c.chunk_id for c in group[1:]}


class TestIndexCuVanChayDuoc:
    """Mọi chunker trước `W3-05` không sinh `parent_chunk_id`."""

    def test_chunk_khong_co_parent_di_qua_nguyen_ven(self) -> None:
        plain = build_chunker(ChunkingConfig(strategy=ChunkingStrategy.FIXED)).chunk(
            [Document(doc_id="doc", content=PROSE, metadata=METADATA)]
        )
        parents = expand_to_parents(retrieved(plain[:5]))

        assert len(parents) == 5
        assert [p.text for p in parents] == [c.content for c in plain[:5]]
        assert all(p.complete for p in parents)

    def test_khong_goi_fetcher_khi_khong_co_anh_em_nao_de_lay(self) -> None:
        plain = build_chunker(ChunkingConfig(strategy=ChunkingStrategy.FIXED)).chunk(
            [Document(doc_id="doc", content=PROSE, metadata=METADATA)]
        )
        fetcher = RecordingFetcher(plain)
        expand_to_parents(retrieved(plain[:3]), fetcher)
        assert fetcher.calls == []


class TestEpKichThuocKhongGopQuaRanhGioiParent:
    """Cùng khuôn `W3-03`: gộp qua ranh giới thì nhãn của chunk là lời nói dối."""

    def test_child_ngan_dau_parent_khong_bi_gop_vao_parent_truoc(self) -> None:
        # `min_chunk_size` rất lớn để ép mọi cơ hội gộp xuất hiện.
        chunks = chunk_prose(min_chunk_size=180, chunk_size=200, max_chunk_size=400)
        for chunk in chunks:
            siblings = chunk.extra[PARENT_CHILDREN_KEY]
            assert chunk.chunk_id in siblings
        # Không child nào mang span vượt ra ngoài span parent của chính nó.
        for chunk in chunks:
            assert chunk.extra[PARENT_START_KEY] <= (chunk.start_char or 0)
            assert (chunk.end_char or 0) <= chunk.extra[PARENT_END_KEY]

    def test_span_parent_khong_chong_len_nhau_ngoai_phan_overlap(self) -> None:
        chunks = chunk_prose(chunk_overlap=0)
        spans = sorted({(c.extra[PARENT_START_KEY], c.extra[PARENT_END_KEY]) for c in chunks})
        for left, right in itertools.pairwise(spans):
            assert left[1] <= right[0], f"{left} chồng {right} dù chunk_overlap=0"


class TestSplitPiecesVaChunkNoiCungMotChuyen:
    def test_split_pieces_tra_dung_day_child(self) -> None:
        chunker = ParentChildChunker(config())
        pieces = chunker.split_pieces(PROSE)
        chunks = chunker.chunk([Document(doc_id="doc", content=PROSE, metadata=METADATA)])
        assert [p.text for p in pieces] == [c.content for c in chunks]


class TestKichThuocParentPhaiLaLuaChonCoY:
    def test_parent_size_multiple_1_bi_tu_choi(self) -> None:
        with pytest.raises(ValueError, match="parent_size_multiple"):
            config(parent_size_multiple=1)

    def test_boi_so_lon_hon_cho_parent_it_hon_va_to_hon(self) -> None:
        small = by_parent(chunk_prose(parent_size_multiple=2))
        large = by_parent(chunk_prose(parent_size_multiple=8))
        assert len(large) < len(small)


class TestAssembledParentKhongNoiDoi:
    def test_expansion_ratio_bang_1_khi_khong_co_anh_em(self) -> None:
        plain = build_chunker(ChunkingConfig(strategy=ChunkingStrategy.FIXED)).chunk(
            [Document(doc_id="doc", content=PROSE, metadata=METADATA)]
        )
        parent = expand_to_parents(retrieved(plain[:1]))[0]
        assert parent.expansion_ratio == pytest.approx(1.0)

    def test_complete_la_False_khi_thieu_du_mot_anh_em(self) -> None:
        empty = AssembledParent(
            parent_id="p",
            doc_id="d",
            text="",
            children=(),
            hit_children=(),
            missing_children=("x",),
        )
        assert empty.complete is False


class TestProtocolKhopVoiLopThat:
    """Cái mà 27 test đầu tiên KHÔNG bắt được: tên method không tồn tại.

    Bản đầu của `context.py` gọi `fetcher.get_by_ids(...)`. Lớp thật
    (`QdrantDenseRetriever`) không có method nào tên vậy — method thật là
    `fetch_chunks` — nhưng `RecordingFetcher` ở trên cũng khai `get_by_ids`, nên
    fake và Protocol khớp nhau hoàn hảo trong khi cả hai cùng sai. Lỗi chỉ lộ ra
    ở lần chạy trên index thật, sau 328 giây build.

    Bài học chung: **một Protocol cấu trúc không ràng buộc được gì nếu cả hai bên
    đối chiếu đều do test dựng ra.** Phải có một bên thật.
    """

    @staticmethod
    def _real_retriever() -> QdrantDenseRetriever:
        from rag_core.embedding.hashing import HashingEmbeddingProvider

        # Client là lazy property nên khởi tạo không chạm mạng.
        return QdrantDenseRetriever(HashingEmbeddingProvider(8), collection="unused")

    def test_lop_that_thoa_man_chunk_fetcher(self) -> None:
        from rag_core.retrieval.context import ChunkFetcher

        assert isinstance(self._real_retriever(), ChunkFetcher)

    def test_chu_ky_khop_ca_ten_tham_so(self) -> None:
        """`isinstance` chỉ kiểm tên method. Filter phải là **keyword** `filters`."""
        import inspect

        from rag_core.retrieval.context import ChunkFetcher

        real = inspect.signature(QdrantDenseRetriever.fetch_chunks)
        protocol = inspect.signature(ChunkFetcher.fetch_chunks)
        assert list(real.parameters) == list(protocol.parameters)
        assert real.parameters["filters"].kind is inspect.Parameter.KEYWORD_ONLY
