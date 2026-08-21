"""`W2-03` — nhánh sparse như một `Retriever`, chạy trên Qdrant thật.

Chia làm hai phần vì hai phần trả lời hai câu hỏi khác nhau, và gộp chúng lại là
cách tự lừa mình:

* **Phần cơ học** (`HashingEmbeddingProvider`, không cần GPU) — lớp bọc có gọi
  đúng nhánh không, filter có xuống tới Qdrant không, `rank` có liên tục không,
  truy vấn không trùng token thì trả gì. Chạy trong `make test-integration`.
* **Phần tính chất** (`@pytest.mark.gpu`, BGE-M3 thật) — hai nhánh khác nhau ở
  chỗ nào, đo bằng model thật thay vì suy từ thiết kế.

⚠️ **Không phần nào ở đây chứng minh DoD của `W2-03`** ("từ khoá lạ mà dense miss
thì sparse hit"), và không được đọc như thế:

* `HashingEmbeddingProvider` băm token thành bucket, nên "dense" của nó cũng là
  lexical — nó không có tính chất *ngữ nghĩa* nào để nói về.
* Corpus BGE-M3 ở đây chỉ có 7 chunk, tức dense chỉ có 6 đối thủ. Đo thật cho
  thấy dense **cũng** tra đúng mã ở hạng 1 (xem `_CODE_TEXTS`).

Bằng chứng cho DoD là known-item search trên index thật 15.814 chunk:
`plans/reports/probes/w2-03-known-item.json` (sparse hit@10 0,5098 vs dense 0,0784,
McNemar `p = 4,8e-07`). Một bài test đơn vị không đứng thay được cho phép đo đó.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

from rag_core.embedding import HashingEmbeddingProvider
from rag_core.retrieval import (
    QdrantDenseRetriever,
    QdrantSparseRetriever,
    build_branch,
)
from rag_core.schemas import Chunk, DocType, DocumentMetadata, Language, RetrievalMode

pytest.importorskip("qdrant_client", reason="cần extra `qdrant`: uv sync --extra qdrant")
pytestmark = pytest.mark.integration

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")


def _chunk(key: str, text: str, *, tenant: str = "t1") -> Chunk:
    return Chunk(
        chunk_id=f"doc-{key}::00000",
        doc_id=f"doc-{key}",
        content=text,
        chunk_index=0,
        metadata=DocumentMetadata(
            source_url=f"https://example.org/{key}",
            license="CC BY 4.0",
            lang=Language.VI,
            doc_type=DocType.DEV_REPORT,
        ),
        extra={"tenant_id": tenant},
    )


# ---------------------------------------------------------------- phần cơ học

_TEXTS = {
    "budget": "Ngân sách nhà nước năm 2024 tăng chi đầu tư công cho hạ tầng giao thông.",
    "code": "Mã báo cáo GSO-2024-XII ghi nhận số liệu thống kê quý bốn của tổng cục.",
    "football": "Đội bóng ghi bàn thắng quyết định ở phút cuối của trận chung kết.",
}
CHUNKS = [_chunk(key, text) for key, text in _TEXTS.items()]


@pytest.fixture
def branch() -> Iterator[QdrantSparseRetriever]:
    collection = f"test_sparsebranch_{uuid.uuid4().hex[:8]}"
    store = QdrantDenseRetriever(
        HashingEmbeddingProvider(dimension=256, sparse=True),
        collection=collection,
        url=QDRANT_URL,
    )
    try:
        store.ensure_collection(recreate=True)
    except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
        pytest.fail(f"Không kết nối được Qdrant tại {QDRANT_URL}: {exc}. Chạy `make up` trước.")
    try:
        store.upsert(CHUNKS)
        yield QdrantSparseRetriever(store)
    finally:
        store.client.delete_collection(collection)


class TestItIsARetriever:
    """Điểm của cả `W2-03`: eval harness chỉ biết gọi `Retriever.retrieve()`."""

    def test_retrieve_uses_the_sparse_branch(self, branch: QdrantSparseRetriever) -> None:
        results = branch.retrieve("ngân sách đầu tư công", top_k=3)
        assert results
        assert all(r.mode is RetrievalMode.SPARSE for r in results)
        assert all(r.sparse_score is not None for r in results)
        assert all(r.dense_score is None for r in results), (
            "điểm sparse không được rơi vào ô dense — `W2-08` tách đóng góp bằng hai ô đó"
        )

    def test_top_hit_is_the_lexically_closest_chunk(self, branch: QdrantSparseRetriever) -> None:
        assert branch.retrieve("ngân sách đầu tư công", top_k=1)[0].chunk.doc_id == "doc-budget"

    def test_ranks_are_contiguous_from_one(self, branch: QdrantSparseRetriever) -> None:
        """Hợp đồng của `Retriever`, không phải chi tiết cài đặt: nDCG và MRR đọc
        `rank` như vị trí thật nên một lỗ trong dãy làm chúng sai âm thầm."""
        results = branch.retrieve("ngân sách thống kê bóng đá", top_k=10)
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_scores_are_non_increasing(self, branch: QdrantSparseRetriever) -> None:
        scores = [r.score for r in branch.retrieve("ngân sách thống kê", top_k=10)]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_is_respected(self, branch: QdrantSparseRetriever) -> None:
        assert len(branch.retrieve("ngân sách thống kê bóng đá", top_k=2)) == 2

    def test_build_branch_gives_a_working_retriever(self, branch: QdrantSparseRetriever) -> None:
        """Đường mà `make eval-retrieval MODE=sparse` thực sự đi."""
        assert build_branch(branch.store, "sparse").retrieve("ngân sách", top_k=1)

    def test_dense_branch_still_works_on_the_same_store(
        self, branch: QdrantSparseRetriever
    ) -> None:
        """Một store, hai nhánh, một kết nối. Nếu nhánh sparse làm hỏng nhánh
        dense thì `W2-04` không có gì để hợp nhất."""
        dense = build_branch(branch.store, "dense").retrieve("ngân sách đầu tư công", top_k=1)
        assert dense and dense[0].mode is RetrievalMode.DENSE


class TestExactLookup:
    """Ca mà DoD của `W2-03` nêu tên: tra một mã chính xác."""

    def test_rare_code_finds_exactly_its_chunk(self, branch: QdrantSparseRetriever) -> None:
        results = branch.retrieve("GSO-2024-XII", top_k=3)
        assert results[0].chunk.doc_id == "doc-code"

    def test_no_token_overlap_returns_nothing(self, branch: QdrantSparseRetriever) -> None:
        """Mặt bù của nhánh sparse, và là nửa lý do `W2-04` tồn tại.

        Dense luôn trả về *một cái gì đó* (mọi vector đều có cosine với mọi
        vector); sparse trả **rỗng**. Hai nhánh hỏng theo hai kiểu khác nhau nên
        hợp nhất chúng có ý nghĩa — chứ không phải để thêm một dòng vào CV.
        """
        assert branch.retrieve("zzzqqq khongtontai", top_k=5) == []
        dense = build_branch(branch.store, "dense").retrieve("zzzqqq khongtontai", top_k=5)
        assert dense, "dense vẫn trả về, chỉ là trả về thứ không liên quan"

    def test_empty_query_returns_nothing_not_an_error(self, branch: QdrantSparseRetriever) -> None:
        assert branch.retrieve("", top_k=5) == []


class TestFilterAndFetch:
    def test_filter_is_applied_at_qdrant(self, branch: QdrantSparseRetriever) -> None:
        """Không post-filter: `W2-06` (cô lập tenant) dựa vào chỗ này, và một lỗ
        ở nhánh sparse sẽ không lộ ra ở bất kỳ test dense nào."""
        assert branch.retrieve("ngân sách", top_k=5, filters={"tenant_id": "t1"})
        assert branch.retrieve("ngân sách", top_k=5, filters={"tenant_id": "khac"}) == []

    def test_filter_by_doc_id(self, branch: QdrantSparseRetriever) -> None:
        results = branch.retrieve("ngân sách thống kê", top_k=5, filters={"doc_id": "doc-code"})
        assert {r.chunk.doc_id for r in results} == {"doc-code"}

    def test_fetch_doc_chunks_works_through_the_wrapper(
        self, branch: QdrantSparseRetriever
    ) -> None:
        """Đây là hố im lặng của `W2-03`, kiểm trên Qdrant thật.

        Eval harness lấy method này bằng `getattr` để tính lại nhãn golden set
        theo span. Thiếu nó thì harness rơi về nhãn ghi sẵn trong file, và lần
        chạy sparse bị chấm bằng bộ nhãn **khác** lần chạy dense — hai con số vẫn
        hiện ra, vẫn xếp hạng được bằng mắt, và vẫn vô nghĩa.
        """
        chunks = branch.fetch_doc_chunks(["doc-code", "doc-budget"])
        assert {c.doc_id for c in chunks} == {"doc-code", "doc-budget"}


class TestRefusals:
    def test_sparse_branch_needs_a_sparse_provider(self) -> None:
        """Chết lúc dựng retriever, trước khi quét span và trước truy vấn đầu."""
        store = QdrantDenseRetriever(
            HashingEmbeddingProvider(dimension=256, sparse=False),
            collection="khong-dung-den",
            url=QDRANT_URL,
        )
        with pytest.raises(ValueError, match="không sinh sparse vector"):
            build_branch(store, "sparse")


# -------------------------------------------------------- phần chất lượng (GPU)

# Sáu mã gần giống nhau + một chunk khác hẳn chủ đề.
#
# ⚠️ Corpus này **không** chứng minh được "dense miss, sparse hit". Đo thật trên
# nó (2026-08-20, xem `reports/tasks/w2-03-sparse-retriever.md` §4): truy vấn
# `GSO-2024-XII` thì **cả hai** nhánh đặt đúng chunk ở hạng 1 — dense 0,6947 vs
# 0,6411 cho ứng viên gần nhất, một khoảng cách rõ ràng. Giả định ban đầu của
# tôi là dense sẽ lẫn giữa những mã gần giống nhau; nó sai, và với 7 chunk thì
# sai là điều phải chờ đợi: dense chỉ có 6 đối thủ. DoD phải đo trên index thật
# 15.814 chunk, không đo ở đây.
#
# Thứ corpus này **đo được** là mặt bù của sparse, và đó là thứ đáng test: truy
# vấn diễn đạt lại bằng từ khác thì sparse mất gần hết recall trong khi dense
# không. Đó là nửa lý do `W2-04` tồn tại.
_CODE_TEXTS = {
    "gso-2024-xii": "Mã báo cáo GSO-2024-XII ghi nhận số liệu thống kê quý bốn năm 2024.",
    "gso-2023-xi": "Mã báo cáo GSO-2023-XI ghi nhận số liệu thống kê quý ba năm 2023.",
    "gso-2022-x": "Mã báo cáo GSO-2022-X ghi nhận số liệu thống kê quý hai năm 2022.",
    "wb-2024-xii": "Mã báo cáo WB-2024-XII ghi nhận số liệu thống kê quý bốn năm 2024.",
    "gso-2024-xi": "Mã báo cáo GSO-2024-XI ghi nhận số liệu thống kê quý ba năm 2024.",
    "gso-2024-x": "Mã báo cáo GSO-2024-X ghi nhận số liệu thống kê quý hai năm 2024.",
    "bongda": "Đội bóng ghi bàn thắng quyết định ở phút cuối của trận chung kết.",
}
_TARGET = "GSO-2024-XII"


@pytest.fixture(scope="module")
def bge_store() -> Iterator[QdrantDenseRetriever]:
    torch = pytest.importorskip("torch", reason="cần extra `ml`")
    if not torch.cuda.is_available():
        pytest.skip("cần GPU — BGE-M3 trên CPU quá chậm cho test")
    from rag_core.embedding.bge_m3 import BgeM3EmbeddingProvider

    collection = f"test_sparsebge_{uuid.uuid4().hex[:8]}"
    store = QdrantDenseRetriever(
        BgeM3EmbeddingProvider(batch_size=8),
        collection=collection,
        url=QDRANT_URL,
    )
    store.ensure_collection(recreate=True)
    try:
        store.upsert([_chunk(key, text) for key, text in _CODE_TEXTS.items()])
        yield store
    finally:
        store.client.delete_collection(collection)


_PARAPHRASE = "kết quả thi đấu thể thao"
"""Truy vấn về `bongda` mà **không dùng lại từ nào** của chunk đó.

Chunk viết "Đội bóng ghi bàn thắng quyết định ở phút cuối của trận chung kết";
truy vấn nói cùng chuyện bằng từ khác. Đây là chỗ hai nhánh tách nhau rõ nhất.
"""


@pytest.mark.gpu
class TestRealModelBranchCharacter:
    """Tính chất của hai nhánh, đo bằng model thật.

    ⚠️ Không phải bằng chứng cho DoD của `W2-03` — xem chú thích ở `_CODE_TEXTS`.
    """

    def test_sparse_picks_the_exact_code(self, bge_store: QdrantDenseRetriever) -> None:
        """Sparse tra đúng mã. Nói được chừng đó, không nói được "hơn dense"."""
        results = QdrantSparseRetriever(bge_store).retrieve(_TARGET, top_k=7)
        assert results[0].chunk.doc_id == "doc-gso-2024-xii", "thứ tự sparse: " + ", ".join(
            r.chunk.doc_id for r in results
        )

    def test_dense_picks_it_too_and_that_is_the_honest_result(
        self, bge_store: QdrantDenseRetriever
    ) -> None:
        """Canh **kết quả âm**, để nó không lặng lẽ biến mất khỏi bộ test.

        Trên 7 chunk thì dense cũng tra đúng mã. Một test kiểu
        `rank_sparse <= rank_dense` sẽ *pass* ở đây — vì hai bên bằng nhau — và
        đọc như một chiến thắng. Đó là cách một kết quả âm bị dựng thành dương.
        """
        dense = [r.chunk.doc_id for r in bge_store.retrieve(_TARGET, 7)]
        assert dense[0] == "doc-gso-2024-xii", f"dense={dense}"

    def test_sparse_recall_collapses_on_a_paraphrase(self, bge_store: QdrantDenseRetriever) -> None:
        """Mặt bù thật, đo được, và là nửa lý do `W2-04` tồn tại.

        Hỏi 4, sparse trả **1** — nó chỉ tới được chunk có token trùng, còn 6
        chunk kia không trùng chữ nào nên không tồn tại đối với nó. Dense trả đủ
        4. Hai nhánh không chỉ *xếp hạng* khác nhau, chúng có **tập ứng viên**
        khác nhau; đó là điều RRF phải xử lý.
        """
        sparse = QdrantSparseRetriever(bge_store).retrieve(_PARAPHRASE, top_k=4)
        dense = bge_store.retrieve(_PARAPHRASE, top_k=4)
        assert len(dense) == 4
        assert len(sparse) < len(dense), f"sparse trả {len(sparse)}, dense trả {len(dense)}"
        assert sparse[0].chunk.doc_id == "doc-bongda", "cái ít ỏi nó trả về vẫn đúng"

    def test_both_branches_miss_the_same_paraphrase_of_a_number(
        self, bge_store: QdrantDenseRetriever
    ) -> None:
        """Ghi lại một chỗ **cả hai** đều sai, vì nó nói `W2-04` không cứu được gì.

        "ba tháng cuối năm 2024" = quý bốn 2024 = `gso-2024-xii` / `wb-2024-xii`.
        Cả hai nhánh đặt `gso-2024-xi` (quý **ba**) ở hạng 1. Hợp nhất hai danh
        sách cùng sai một kiểu thì vẫn sai — chỗ này là việc của reranker
        (`W2-05`), không phải của RRF.
        """
        query = "thống kê ba tháng cuối năm 2024"
        dense = [r.chunk.doc_id for r in bge_store.retrieve(query, 4)]
        sparse = [r.chunk.doc_id for r in QdrantSparseRetriever(bge_store).retrieve(query, 4)]
        assert dense[0] == "doc-gso-2024-xi", f"dense={dense}"
        assert sparse[0] == "doc-gso-2024-xi", f"sparse={sparse}"
