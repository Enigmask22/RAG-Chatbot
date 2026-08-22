"""`W3-07` — sửa một trang trong tài liệu 100 trang, embed lại bao nhiêu chunk?

DoD: *sửa 1 trang trong 100 trang → chỉ embed lại chunk bị ảnh hưởng.* Cách đo
mà DoD chỉ định là **đếm số lần gọi embed**, nên ở đây có hai lớp đếm độc lập:

* `BuildReport.n_chunks_embedded` — con số `build_index` tự báo.
* `CountingEmbeddings` — bọc provider và đếm **text** thật sự đi qua nó.

Hai lớp vì một mình con số tự báo không chứng minh được gì: nó là biến đếm của
chính đoạn code đang được kiểm. Nếu `upsert_reusing` báo "mượn lại 40" mà vẫn gọi
provider 40 lần thì chỉ lớp thứ hai thấy.

## Vì sao tầng bỏ-qua-theo-tài-liệu (`W1-08`) không đủ

`build_index` đã bỏ qua tài liệu có `Document.content_hash` không đổi từ `W1-08`.
Nhưng sửa **một trang** thì hash tài liệu đổi, nên cả tài liệu bị chunk lại và
embed lại. Với tài liệu 234.939 ký tự của corpus thật thì đó là ~250 chunk cho
một dòng sửa.

## Ca khó: chèn thêm chữ — và giới hạn thật của kỹ thuật này

`chunk_id` là `{doc_id}::{index:05d}` — thuần vị trí. Chèn vào **giữa** tài liệu
làm mọi chunk phía sau đổi chỉ số, nên so theo **vị trí** sẽ kết luận "tất cả đã
đổi". Khớp theo `content_hash` tránh được cái đó.

Nhưng nó **không** cứu được mọi thứ, và đo được ranh giới:

| ca | mượn lại |
|---|---|
| sửa tại chỗ (không đổi ranh giới chunk) | ~99% |
| nối thêm vào cuối | ~98% |
| chèn ở **giữa**, chunk gói 3 câu | **51,5%** |
| chèn ở **đầu** (vị trí 5/300), chunk gói 3 câu | **2,0%** |

Vì `_enforce_size`/splitter đóng gói **tham lam** theo thứ tự: chèn một câu làm
mọi chunk phía sau **gói lại khác đi**, nên nội dung chúng thật sự khác — không
phải chỉ đổi chỉ số. Luật rút ra: *mượn lại được đúng phần đứng TRƯỚC điểm sửa*.

⚠️ Điều này chỉ vô hình khi mỗi chunk chứa đúng một đơn vị tách (một câu vừa khít
`chunk_size`). Fixture đầu tiên của file này rơi đúng vào đó và cho 99% ở **mọi**
ca — một con số đẹp mà vô nghĩa. `TestChenVaoGiua` cố ý dùng câu ngắn để chunk
gói 3 câu, tức chế độ mà corpus thật đang ở (chunk 1000 ký tự ≈ 7 câu).
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

pytest.importorskip("qdrant_client", reason="cần extra `qdrant`: uv sync --extra qdrant")

from pipeline.corpus.manifest import CorpusEntry, write_manifest
from pipeline.indexing.build_index import BuildReport, IndexState, build_index
from pipeline.indexing.config import IndexConfig
from rag_core.embedding import HashingEmbeddingProvider
from rag_core.embedding.base import FloatArray, HybridVectors
from rag_core.retrieval.qdrant_store import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    QdrantDenseRetriever,
)
from rag_core.schemas import DocType, Language

pytestmark = pytest.mark.integration

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")

_LINE = (
    "Ngân sách nhà nước năm 2024 tăng chi đầu tư công cho hạ tầng giao thông "
    "tại các tỉnh miền Trung và đồng bằng sông Cửu Long. "
)


def _pages(count: int, *, edited: int | None = None, inserted_at: int | None = None) -> str:
    """Tài liệu `count` "trang", mỗi trang một câu **khác nhau**.

    Trang phải khác nhau, nếu không mọi `content_hash` trùng nhau và phép đo trở
    thành vô nghĩa — đúng bài học fixture của `W3-01` §2 và `W3-05` §9.
    """
    out = []
    for index in range(count):
        body = f"Trang {index:03d}. {_LINE}"
        if index == edited:
            body = f"Trang {index:03d}. ĐÃ SỬA MỘT DÒNG. {_LINE}"
        out.append(body)
        if index == inserted_at:
            out.append(f"Trang chèn thêm {index:03d}X. {_LINE}")
    return "".join(out)


class CountingEmbeddings(HashingEmbeddingProvider):
    """Đếm text thật sự đi qua model. Lớp kiểm chứng độc lập với báo cáo của code."""

    def __init__(self, dimension: int = 64, *, sparse: bool = False) -> None:
        super().__init__(dimension, sparse=sparse)
        self.texts_embedded = 0
        self.calls = 0

    def embed_documents(self, texts: Sequence[str]) -> FloatArray:
        self.calls += 1
        self.texts_embedded += len(texts)
        return super().embed_documents(texts)

    def embed_documents_hybrid(self, texts: Sequence[str]) -> HybridVectors | None:
        result = super().embed_documents_hybrid(texts)
        if result is not None:
            # Đường hybrid đi vòng khác `embed_documents`, nên phải đếm riêng —
            # nếu không thì với provider sinh sparse, bộ đếm luôn bằng 0 và mọi
            # assert trong file này thành vô nghĩa.
            self.calls += 1
            self.texts_embedded += len(texts)
        return result

    def reset(self) -> None:
        self.texts_embedded = 0
        self.calls = 0


def _write_corpus(root: Path, bodies: dict[str, str]) -> Path:
    entries = []
    for doc_id, text in bodies.items():
        payload = f"{doc_id}. {text}".encode()
        path = root / f"{doc_id}.txt"
        path.write_bytes(payload)
        entries.append(
            CorpusEntry(
                doc_id=doc_id,
                relative_path=path.name,
                source_url=f"https://example.org/{doc_id}",
                license="CC BY 4.0",
                sha256=hashlib.sha256(payload).hexdigest(),
                bytes=len(payload),
                source="test",
                lang=Language.VI,
                doc_type=DocType.DEV_REPORT,
            )
        )
    manifest = root / "manifest.csv"
    write_manifest(manifest, entries)
    return manifest


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[tuple[Path, str]]:
    collection = f"test_incr_{uuid.uuid4().hex[:8]}"
    yield tmp_path, collection
    client = QdrantDenseRetriever(
        HashingEmbeddingProvider(dimension=64), collection=collection, url=QDRANT_URL
    ).client
    if client.collection_exists(collection):
        client.delete_collection(collection)


@pytest.fixture
def counter() -> CountingEmbeddings:
    return CountingEmbeddings(dimension=64)


def _config(root: Path, collection: str) -> IndexConfig:
    return IndexConfig.model_validate(
        {
            "name": collection,
            "collection": collection,
            "manifest_path": root / "manifest.csv",
            "corpus_dir": root,
            "embedding_model": "hashing:64",
            "use_cache": False,
            "state_dir": root / "state",
            # `min_chunk_size` PHẢI khai tường minh: mặc định là 200, bằng đúng
            # `chunk_size` ở đây, và khi đó mọi mảnh đều bị gộp tới
            # `max_chunk_size` — 100 "trang" ra 10 chunk thay vì ~73. Xem
            # `ChunkingConfig._warn_if_chunk_size_is_decorative`, phát hiện ở
            # chính lần chạy đầu của file test này.
            "chunking": {
                "strategy": "fixed",
                "chunk_size": 200,
                "chunk_overlap": 0,
                "min_chunk_size": 50,
                "max_chunk_size": 400,
            },
        }
    )


def _build(config: IndexConfig, embeddings: CountingEmbeddings) -> BuildReport:
    return build_index(config, qdrant_url=QDRANT_URL, embeddings=embeddings)


class TestSuaMotTrang:
    """DoD: sửa 1 trang trong 100 → chỉ embed lại chunk bị ảnh hưởng."""

    def test_chi_embed_lai_phan_da_sua(
        self, workspace: tuple[Path, str], counter: CountingEmbeddings
    ) -> None:
        root, collection = workspace
        config = _config(root, collection)

        _write_corpus(root, {"d-1": _pages(100)})
        first = _build(config, counter)
        assert first.n_chunks_embedded == first.n_chunks_written
        assert first.n_chunks_reused == 0
        assert counter.texts_embedded == first.n_chunks_embedded
        total = first.n_chunks_written

        counter.reset()
        _write_corpus(root, {"d-1": _pages(100, edited=42)})
        second = _build(config, counter)

        assert second.n_documents_skipped == 0, "hash tài liệu phải đã đổi"
        assert second.n_chunks_written == total, "vẫn ghi đủ chunk"
        # Con số của DoD. Ngưỡng rộng rãi — điều đang kiểm là "tỉ lệ với phần
        # đã sửa", không phải một hằng số phụ thuộc chunk_size.
        assert second.n_chunks_embedded <= 5, (
            f"sửa một trang mà embed lại {second.n_chunks_embedded}/{total} chunk"
        )
        assert second.n_chunks_reused >= total - 5
        assert counter.texts_embedded == second.n_chunks_embedded, (
            "báo cáo nói mượn lại nhưng provider vẫn bị gọi"
        )

    def test_nhanh_hon_build_lai_it_nhat_10_lan(
        self, workspace: tuple[Path, str], counter: CountingEmbeddings
    ) -> None:
        """`G3`: reprocess sau sửa nhỏ phải rẻ hơn full rebuild ≥ 10×.

        Đo bằng **số chunk phải embed**, không bằng đồng hồ: đồng hồ trên
        `HashingEmbeddingProvider` đo tốc độ hash chứ không đo thứ tốn tiền thật,
        và một ngưỡng thời gian trên CI là một test bập bênh.
        """
        root, collection = workspace
        config = _config(root, collection)

        _write_corpus(root, {"d-1": _pages(100)})
        full = _build(config, counter).n_chunks_embedded

        counter.reset()
        _write_corpus(root, {"d-1": _pages(100, edited=42)})
        incremental = _build(config, counter).n_chunks_embedded

        assert incremental * 10 <= full, f"chỉ rẻ hơn {full / max(incremental, 1):.1f}×"


class TestChenVaoGiua:
    """Giới hạn thật của kỹ thuật này, đo bằng số chứ không phát biểu bằng lời.

    Dùng câu **ngắn** để mỗi chunk gói 3 câu — chế độ mà corpus thật đang ở
    (chunk 1000 ký tự ≈ 7 câu). Fixture `_pages` không dựng được ca này: một
    "trang" 146 ký tự vừa khít chunk 200, nên ranh giới chunk do **nội dung**
    quyết định và mọi ca chèn đều cho 99% — đẹp mà vô nghĩa.
    """

    @staticmethod
    def _sentences(count: int, *, extra_at: int | None = None) -> str:
        out = []
        for index in range(count):
            out.append(f"Cau {index:04d} noi ve ngan sach dau tu cong nam hai ngan hai tu. ")
            if index == extra_at:
                out.append("Cau chen them noi ve ha tang giao thong o mien Trung. ")
        return "".join(out)

    def _packs_three(self, root: Path, config: IndexConfig, counter: CountingEmbeddings) -> int:
        _write_corpus(root, {"d-1": self._sentences(300)})
        chunks = _build(config, counter).n_chunks_written
        assert 2.5 <= 300 / chunks <= 3.5, (
            f"{300 / chunks:.1f} câu/chunk — fixture không dựng được chế độ gói nhiều câu, "
            "nên mọi kết luận bên dưới sẽ nói về một chế độ khác"
        )
        return chunks

    def test_chen_o_giua_muon_lai_duoc_phan_dung_truoc(
        self, workspace: tuple[Path, str], counter: CountingEmbeddings
    ) -> None:
        root, collection = workspace
        config = _config(root, collection)
        self._packs_three(root, config, counter)

        counter.reset()
        _write_corpus(root, {"d-1": self._sentences(300, extra_at=150)})
        second = _build(config, counter)

        # ~một nửa: đúng phần đứng trước điểm chèn. Không phải "gần như tất cả",
        # và cũng không phải "không được gì" — hai kết luận sai theo hai hướng.
        assert 0.35 <= second.n_chunks_reused / second.n_chunks_written <= 0.65, (
            f"mượn lại {second.n_chunks_reused}/{second.n_chunks_written}"
        )
        assert counter.texts_embedded == second.n_chunks_embedded

    def test_chen_o_dau_thi_gan_nhu_khong_muon_lai_duoc(
        self, workspace: tuple[Path, str], counter: CountingEmbeddings
    ) -> None:
        """Ghim **giới hạn**, để sau này không ai phát biểu quá lên.

        Chèn ở vị trí 5/300 làm mọi chunk phía sau gói lại khác đi, nên nội dung
        chúng thật sự khác — `content_hash` không cứu được, và không nên cứu: các
        chunk ấy **đúng là** văn bản mới. Lối ra thật là chunking theo nội dung
        (ranh giới do hash cục bộ quyết định), không phải tra hash tinh hơn.
        """
        root, collection = workspace
        config = _config(root, collection)
        total = self._packs_three(root, config, counter)

        counter.reset()
        _write_corpus(root, {"d-1": self._sentences(300, extra_at=5)})
        second = _build(config, counter)

        assert second.n_chunks_reused < 0.1 * total, (
            f"mượn lại {second.n_chunks_reused}/{total} — nếu con số này đã tốt lên "
            "thì chunker đã đổi, và tài liệu của W3-07 phải được đo lại"
        )


class TestKhongCoStateCuThiVanDung:
    def test_state_truoc_w3_07_khong_co_chunk_hashes(
        self, workspace: tuple[Path, str], counter: CountingEmbeddings
    ) -> None:
        """State cũ thiếu `chunk_hashes` → embed lại toàn bộ, không phải lỗi."""
        root, collection = workspace
        config = _config(root, collection)

        _write_corpus(root, {"d-1": _pages(40)})
        first = _build(config, counter)

        state = IndexState.load(config.state_path)
        assert state is not None
        state.documents["d-1"] = state.documents["d-1"].model_copy(update={"chunk_hashes": []})
        state.save(config.state_path)

        counter.reset()
        _write_corpus(root, {"d-1": _pages(40, edited=5)})
        second = _build(config, counter)

        assert second.n_chunks_reused == 0
        assert second.n_chunks_embedded == first.n_chunks_written


class TestVectorMuonLaiPhaiDungVector:
    def test_vector_giu_nguyen_qua_lan_index_lai(
        self, workspace: tuple[Path, str], counter: CountingEmbeddings
    ) -> None:
        """Mượn nhầm vector là chế độ hỏng tệ nhất: index vẫn đầy, kết quả vẫn ra.

        Nên phải so **giá trị vector** trước/sau, không chỉ so số lượng point.
        """
        root, collection = workspace
        config = _config(root, collection)
        store = QdrantDenseRetriever(counter, collection=collection, url=QDRANT_URL)

        _write_corpus(root, {"d-1": _pages(40)})
        _build(config, counter)
        state = IndexState.load(config.state_path)
        assert state is not None
        untouched = [f"d-1::{i:05d}" for i in range(5)]
        before = store.fetch_vectors(untouched)
        assert len(before) == len(untouched)

        _write_corpus(root, {"d-1": _pages(40, edited=30)})
        _build(config, counter)
        after = store.fetch_vectors(untouched)

        for chunk_id in untouched:
            assert after[chunk_id]["dense"] == pytest.approx(before[chunk_id]["dense"])

    def test_sparse_khong_bi_bo_quen_khi_muon_lai(self, workspace: tuple[Path, str]) -> None:
        """Provider sinh sparse thì vector mượn lại phải mang **cả hai** nhánh.

        Mượn mà chỉ chép `dense` sẽ tạo point thiếu `sparse`: nhánh sparse im lặng
        trả rỗng cho đúng những chunk ấy, số point vẫn đủ, không gì báo. Đây là
        chế độ hỏng mà `W2-02` đã phải dựng cả `schema_problems` để chặn ở tầng
        collection — ở tầng point thì chỉ phép kiểm này thấy.
        """
        root, collection = workspace
        hybrid = CountingEmbeddings(dimension=64, sparse=True)
        config = _config(root, collection)
        store = QdrantDenseRetriever(hybrid, collection=collection, url=QDRANT_URL)

        _write_corpus(root, {"d-1": _pages(40)})
        _build(config, hybrid)
        untouched = [f"d-1::{i:05d}" for i in range(5)]

        _write_corpus(root, {"d-1": _pages(40, edited=30)})
        second = _build(config, hybrid)
        assert second.n_chunks_reused > 0, "không mượn lại thì test này không kiểm gì"

        after = store.fetch_vectors(untouched)
        for chunk_id in untouched:
            assert DENSE_VECTOR_NAME in after[chunk_id]
            assert SPARSE_VECTOR_NAME in after[chunk_id], "point mượn lại bị mất nhánh sparse"
