"""Test cho `build_branch` và `QdrantSparseRetriever` — `W2-03`, không cần Qdrant.

Phần đắt của nhánh sparse là truy vấn thật (`tests/integration/test_sparse_retriever.py`).
Phần *dễ hỏng* lại là phần rẻ: chọn đúng nhánh, chết đúng lúc, và **chuyển tiếp
`fetch_doc_chunks`**. Cái cuối là hố im lặng — thiếu nó thì eval harness rơi về
nhãn ghi sẵn trong file và lần chạy sparse được chấm bằng bộ nhãn khác lần chạy
dense, mà bảng số vẫn hiện ra bình thường.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_core.embedding import HashingEmbeddingProvider
from rag_core.retrieval import (
    QdrantDenseRetriever,
    QdrantSparseRetriever,
    Retriever,
    build_branch,
)
from rag_core.schemas import Chunk, RetrievalMode, RetrievedChunk


def _store(*, sparse: bool) -> QdrantDenseRetriever:
    """Store không kết nối — `client` là lazy nên chỉ dựng thôi thì không chạm mạng."""
    return QdrantDenseRetriever(
        HashingEmbeddingProvider(dimension=64, sparse=sparse),
        collection="rag_test_branch",
    )


class TestBuildBranch:
    def test_dense_returns_the_store_itself(self) -> None:
        """Không bọc thêm một lớp chỉ để đối xứng: `retriever.name` đi vào mọi
        report, nên đổi nó là làm số cũ của W1/W2-01/W2-02 không so được nữa."""
        store = _store(sparse=False)
        assert build_branch(store, "dense") is store
        assert build_branch(store, RetrievalMode.DENSE) is store

    def test_sparse_wraps(self) -> None:
        store = _store(sparse=True)
        branch = build_branch(store, "sparse")
        assert isinstance(branch, QdrantSparseRetriever)
        assert branch.store is store

    def test_both_branches_are_retrievers(self) -> None:
        store = _store(sparse=True)
        assert isinstance(build_branch(store, "dense"), Retriever)
        assert isinstance(build_branch(store, "sparse"), Retriever)

    def test_names_differ_so_reports_do_not_collide(self) -> None:
        store = _store(sparse=True)
        assert build_branch(store, "dense").name == "qdrant-dense:rag_test_branch"
        assert build_branch(store, "sparse").name == "qdrant-sparse:rag_test_branch"

    def test_unknown_mode_lists_the_valid_ones(self) -> None:
        with pytest.raises(ValueError, match="không hợp lệ"):
            build_branch(_store(sparse=True), "bm25")

    def test_every_mode_is_now_implemented(self) -> None:
        """Hai thông báo khác nhau cho hai chuyện khác nhau: "tên sai" gửi người
        đọc đi tra chính tả, "chưa cài" gửi họ đi xem hạng mục đang làm.

        Test này đã trỏ vào `hybrid` (`W2-04` cài), rồi `reranked` (`W2-05` cài).
        Cả hai lần, việc nó **đỏ** là thứ nhắc rằng `SUPPORTED_MODES` cần cập
        nhật. Giờ không còn tên nào để trỏ vào, nên nó canh chính điều đó — và nó
        sẽ đỏ đúng lúc ai đó thêm một mode mới vào `RetrievalMode` mà quên cài.
        """
        from rag_core.retrieval import SUPPORTED_MODES

        assert set(SUPPORTED_MODES) == set(RetrievalMode)

    def test_hybrid_is_implemented_since_w2_04(self) -> None:
        assert build_branch(_store(sparse=True), "hybrid") is not None


class TestFailsEarly:
    def test_sparse_branch_on_dense_only_provider_refuses_at_construction(self) -> None:
        """Chết lúc dựng, không lúc truy vấn đầu.

        `_eval_against_index` quét toàn bộ chunk của corpus để ánh xạ span
        **trước** khi truy vấn câu đầu, nên lỗi ở truy vấn đầu là lỗi đến sau
        vài giây quét vô ích — với grid của `W2-07` thì là sau vài phút.
        """
        with pytest.raises(ValueError, match="không sinh sparse vector"):
            build_branch(_store(sparse=False), "sparse")

    def test_the_message_says_how_to_fix_it(self) -> None:
        with pytest.raises(ValueError, match="sparse=True"):
            QdrantSparseRetriever(_store(sparse=False))


class _FakeStore:
    """Chỉ mang đủ bề mặt mà lớp bọc dùng tới."""

    def __init__(self) -> None:
        self.embeddings = HashingEmbeddingProvider(dimension=64, sparse=True)
        self.collection = "fake"
        self.writes_sparse = True
        self.sparse_calls: list[tuple[str, int, dict[str, Any] | None]] = []
        self.doc_calls: list[tuple[tuple[str, ...], int]] = []
        self.dense_calls = 0
        self.verified = 0

    def retrieve(self, query: str, top_k: int = 10, **_: Any) -> list[RetrievedChunk]:
        self.dense_calls += 1
        return []

    def retrieve_sparse(
        self, query: str, top_k: int = 10, *, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        self.sparse_calls.append((query, top_k, filters))
        return []

    def fetch_doc_chunks(self, doc_ids: Any, *, batch: int = 512) -> list[Chunk]:
        self.doc_calls.append((tuple(doc_ids), batch))
        return []

    def verify_schema(self) -> None:
        self.verified += 1


class TestDelegation:
    def _wrap(self) -> tuple[QdrantSparseRetriever, _FakeStore]:
        store = _FakeStore()
        return QdrantSparseRetriever(store), store  # type: ignore[arg-type]

    def test_retrieve_goes_to_the_sparse_branch_not_the_dense_one(self) -> None:
        branch, store = self._wrap()
        branch.retrieve("ngân sách", 7)
        assert store.sparse_calls == [("ngân sách", 7, None)]
        assert store.dense_calls == 0, "gọi nhầm nhánh thì bảng ablation đo hai lần cùng một thứ"

    def test_filters_are_passed_through(self) -> None:
        """Filter phải xuống tới Qdrant, nếu không thì `W2-06` (cô lập tenant) có
        một lỗ đúng bằng nhánh sparse."""
        branch, store = self._wrap()
        branch.retrieve("q", filters={"tenant_id": "acme"})
        assert store.sparse_calls[0][2] == {"tenant_id": "acme"}

    def test_default_top_k_matches_the_interface(self) -> None:
        branch, store = self._wrap()
        branch.retrieve("q")
        assert store.sparse_calls[0][1] == 10

    def test_fetch_doc_chunks_is_forwarded(self) -> None:
        """Hố im lặng của `W2-03`. Eval harness lấy method này bằng `getattr` để
        tính lại nhãn theo span (`TD-12`); thiếu nó thì harness rơi về nhãn ghi
        sẵn trong file và lần chạy sparse bị chấm bằng bộ nhãn KHÁC lần chạy
        dense — hai con số vẫn hiện ra và vẫn vô nghĩa.
        """
        branch, store = self._wrap()
        branch.fetch_doc_chunks(["doc-a", "doc-b"], batch=8)
        assert store.doc_calls == [(("doc-a", "doc-b"), 8)]

    def test_the_harness_duck_type_check_finds_it(self) -> None:
        """Canh đúng phép thử mà `_resolve_span_labels` thực sự làm."""
        branch, _ = self._wrap()
        assert callable(getattr(branch, "fetch_doc_chunks", None))

    def test_verify_schema_is_forwarded(self) -> None:
        branch, store = self._wrap()
        branch.verify_schema()
        assert store.verified == 1

    def test_no_blanket_getattr_forwarding(self) -> None:
        """Chuyển tiếp tường minh, không `__getattr__`. Một wrapper "trong suốt"
        thì mypy không kiểm được gì, và hố ở test trên mở lại ngay lần đổi tên
        method tiếp theo ở store."""
        branch, _ = self._wrap()
        assert not hasattr(branch, "upsert")
        assert not hasattr(branch, "retrieve_sparse")
