"""Test cho `schema_problems` — `W2-02`.

Hàm này là hàng rào giữa "đổi cấu hình" và "ghi vào collection sai schema". Nó
thuần và không chạm Qdrant, nên mọi ca lệch test được trong `make test`.

Vì sao nó tồn tại: trước `W2-02`, `ensure_collection` thấy collection tồn tại là
trả về ngay. Chạy provider sinh sparse lên collection dense-only sẽ chết ở lần
upsert **đầu tiên** — tức sau khi đã nạp 2,2GB trọng số và chunk xong tài liệu
đầu — và thông báo của Qdrant không nói phải làm gì để sửa.
"""

from __future__ import annotations

import logging

import pytest

from rag_core.embedding import HashingEmbeddingProvider
from rag_core.retrieval import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, schema_problems
from rag_core.retrieval.qdrant_store import QdrantDenseRetriever
from rag_core.schemas import Chunk


def _problems(
    dense: dict[str, int] | None = None,
    sparse: frozenset[str] = frozenset(),
    *,
    want_dimension: int = 1024,
    want_sparse: bool = False,
) -> list[str]:
    return schema_problems(
        dense_sizes={DENSE_VECTOR_NAME: 1024} if dense is None else dense,
        sparse_names=sparse,
        want_dimension=want_dimension,
        want_sparse=want_sparse,
    )


class TestMatching:
    def test_dense_only_matches(self) -> None:
        assert _problems() == []

    def test_hybrid_matches(self) -> None:
        assert _problems(sparse=frozenset({SPARSE_VECTOR_NAME}), want_sparse=True) == []

    def test_extra_dense_vector_names_are_tolerated(self) -> None:
        """Collection có thêm named vector khác không phải lỗi — `W3-05`
        (parent-child) sẽ thêm, và nó không ảnh hưởng nhánh đang dùng."""
        assert _problems({DENSE_VECTOR_NAME: 1024, "parent": 1024}) == []


class TestDenseMismatch:
    def test_anonymous_vector_collection(self) -> None:
        """Collection của phiên bản trước dùng vector vô danh — mọi truy vấn
        `using="dense"` sẽ lỗi."""
        problems = _problems({})
        assert len(problems) == 1
        assert "thiếu named vector" in problems[0]
        assert "vector vô danh" in problems[0]

    def test_wrong_dense_name(self) -> None:
        problems = _problems({"embedding": 1024})
        assert "thiếu named vector" in problems[0]
        assert "embedding" in problems[0], "phải nói collection đang có tên gì"

    def test_dimension_mismatch_is_the_change_model_case(self) -> None:
        """Ca xảy ra mỗi lần đổi model embedding: 768 (PhoBERT) → 1024 (BGE-M3).

        Qdrant vẫn từ chối upsert, nhưng chỉ *sau khi* đã nạp model và chunk xong.
        """
        problems = _problems({DENSE_VECTOR_NAME: 768}, want_dimension=1024)
        assert len(problems) == 1
        assert "768" in problems[0] and "1024" in problems[0]

    def test_missing_name_does_not_also_report_dimension(self) -> None:
        """Một nguyên nhân, một dòng. Báo cả "thiếu tên" lẫn "sai chiều" cho cùng
        một sự việc làm người đọc đi tìm hai vấn đề."""
        assert len(_problems({})) == 1


class TestSparseMismatch:
    def test_provider_writes_sparse_but_collection_has_no_room(self) -> None:
        """Ca mới của `W2-02`, và là ca nguy hiểm nhất vì nó chưa từng xảy ra."""
        problems = _problems(want_sparse=True)
        assert len(problems) == 1
        assert SPARSE_VECTOR_NAME in problems[0]

    def test_collection_has_sparse_but_provider_does_not(self) -> None:
        """KHÔNG được bỏ qua ca này. Nó nghĩa là đang eval bằng provider
        dense-only trên index hybrid: con số trông bình thường trong khi một nửa
        index không được dùng tới — đúng kiểu hỏng im lặng của `TD-11`.
        """
        problems = _problems(sparse=frozenset({SPARSE_VECTOR_NAME}), want_sparse=False)
        assert len(problems) == 1
        assert "nửa index" in problems[0]

    def test_unrelated_sparse_name_does_not_count(self) -> None:
        problems = _problems(sparse=frozenset({"bm25"}), want_sparse=True)
        assert len(problems) == 1
        assert SPARSE_VECTOR_NAME in problems[0]

    def test_both_dense_and_sparse_wrong_reports_both(self) -> None:
        problems = _problems({DENSE_VECTOR_NAME: 768}, want_dimension=1024, want_sparse=True)
        assert len(problems) == 2


class TestHashingProviderSparse:
    """`HashingEmbeddingProvider` sinh được sparse khi bật `sparse=True` — để
    `W2-02`…`W2-04` test được schema hybrid, sparse retriever và RRF mà không cần
    GPU + 2,2GB trọng số."""

    def test_off_by_default(self) -> None:
        """Mặc định TẮT, có chủ ý: `name` đi vào cache key của semantic chunker và
        vào MLflow, nên bật sẵn sẽ đổi tên của mọi provider mặc định — tức vô hiệu
        cache chunk và làm mọi test W1 đổi nghĩa âm thầm."""
        provider = HashingEmbeddingProvider(dimension=64)
        assert provider.sparse_vocab_size is None
        assert provider.embed_documents_hybrid(["a"]) is None
        assert provider.embed_query_hybrid("a") is None

    def test_reports_sparse_vocab_when_on(self) -> None:
        assert HashingEmbeddingProvider(dimension=64, sparse=True).sparse_vocab_size == 65_536

    def test_name_says_whether_sparse_is_on(self) -> None:
        """Hai cấu hình khác nhau không được mang cùng một `name`, vì nó là cache
        key. Và tên của bản mặc định phải giữ nguyên như W1."""
        assert HashingEmbeddingProvider(dimension=64).name == "hashing-64d"
        assert HashingEmbeddingProvider(dimension=64, sparse=True).name == "hashing-64d+sparse"

    def test_dense_part_is_unchanged_by_turning_sparse_on(self) -> None:
        """Bất biến hồi quy: bật sparse không được làm lệch dense, nếu không thì
        mọi test dùng provider này ở W1 đổi nghĩa mà không ai biết."""
        import numpy as np

        texts = ["ngân sách nhà nước", "the state budget 2024"]
        with_sparse = HashingEmbeddingProvider(dimension=64, sparse=True).embed_documents(texts)
        without = HashingEmbeddingProvider(dimension=64).embed_documents(texts)
        assert np.array_equal(with_sparse, without)

    def test_hybrid_dense_matches_embed_documents(self) -> None:
        import numpy as np

        provider = HashingEmbeddingProvider(dimension=64, sparse=True)
        texts = ["ngân sách", "đầu tư công"]
        hybrid = provider.embed_documents_hybrid(texts)
        assert hybrid is not None
        assert np.array_equal(hybrid.dense, provider.embed_documents(texts))

    def test_sparse_reflects_lexical_overlap(self) -> None:
        """Nếu sparse không phản ánh trùng lặp từ vựng thì nó là vector ngẫu
        nhiên và mọi test dùng nó sẽ pass kể cả khi thuật toán sai."""
        provider = HashingEmbeddingProvider(dimension=64, sparse=True)
        result = provider.embed_documents_hybrid(
            ["ngân sách đầu tư công", "ngân sách nhà nước", "trận đấu bóng đá"]
        )
        assert result is not None
        a, b, c = result.sparse
        assert a.dot(b) > a.dot(c)

    def test_query_and_document_sparse_agree(self) -> None:
        provider = HashingEmbeddingProvider(dimension=64, sparse=True)
        docs = provider.embed_documents_hybrid(["ngân sách nhà nước"])
        query = provider.embed_query_hybrid("ngân sách nhà nước")
        assert docs is not None and query is not None
        assert docs.sparse[0] == query[1]

    def test_repeated_token_uses_max_not_sum(self) -> None:
        """Khớp cách BGE-M3 gộp: trọng số là "token này quan trọng thế nào",
        không phải "nó xuất hiện bao nhiêu lần"."""
        provider = HashingEmbeddingProvider(dimension=64, sparse=True)
        once = provider.embed_query_hybrid("ngân")
        twice = provider.embed_query_hybrid("ngân ngân ngân")
        assert once is not None and twice is not None
        assert len(twice[1]) == 1
        assert twice[1].values[0] > once[1].values[0], "log1p(3) > log1p(1)"

    def test_empty_text_gives_empty_sparse_not_error(self) -> None:
        provider = HashingEmbeddingProvider(dimension=64, sparse=True)
        result = provider.embed_query_hybrid("")
        assert result is not None
        assert len(result[1]) == 0

    def test_empty_batch(self) -> None:
        result = HashingEmbeddingProvider(dimension=64, sparse=True).embed_documents_hybrid([])
        assert result is not None
        assert result.dense.shape == (0, 64)
        assert result.sparse == []

    def test_rejects_tiny_dimension_still(self) -> None:
        with pytest.raises(ValueError, match="dimension quá nhỏ"):
            HashingEmbeddingProvider(dimension=4)


class TestTenantOnEveryPoint:
    """`TD-40` — `_payload` là chỗ duy nhất biết field nào được làm phẳng.

    Thuần, không chạm Qdrant: mọi ca kiểm được trong `make test`. Phần "nó có
    thật sự tới Qdrant không" nằm ở `tests/integration/test_metadata_filter.py`.
    """

    @staticmethod
    def _store(tenant: str | None) -> QdrantDenseRetriever:
        return QdrantDenseRetriever(
            HashingEmbeddingProvider(dimension=8), collection="c", tenant_id=tenant
        )

    @staticmethod
    def _chunk(**extra: str) -> Chunk:
        return Chunk(chunk_id="d::0", doc_id="d", content="nội dung", chunk_index=0, extra=extra)

    def test_the_store_tenant_lands_on_every_point(self) -> None:
        assert self._store("acme")._payload(self._chunk())["tenant_id"] == "acme"

    def test_a_store_without_a_tenant_writes_no_tenant_field(self) -> None:
        """Hành vi cũ giữ nguyên — index đo đạc chạy ngoài đường serving vẫn dựng được.

        ⚠️ Và đó chính là hình dạng của `TD-40`: point không có field này **vô
        hình** với mọi request đã xác thực, không báo lỗi ở đâu. Chỗ chặn là
        `IndexConfig.tenant_id` bắt buộc, không phải chỗ này.
        """
        assert "tenant_id" not in self._store(None)._payload(self._chunk())

    def test_a_chunk_can_carry_its_own_tenant_when_the_store_has_none(self) -> None:
        assert self._store(None)._payload(self._chunk(tenant_id="t1"))["tenant_id"] == "t1"

    def test_the_store_wins_when_the_chunk_disagrees(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """⭐ Một field có **hai** nguồn sự thật là chỗ tenant sai xuất hiện lúc
        không ai nhìn. Store được dựng từ config sở hữu collection, nên một chunk
        khai khác là bug ở tầng gọi — và nó phải để lại dấu vết."""
        with caplog.at_level(logging.WARNING):
            payload = self._store("acme")._payload(self._chunk(tenant_id="globex"))

        assert payload["tenant_id"] == "acme"
        assert "globex" in caplog.text and "acme" in caplog.text

    def test_the_tenant_is_not_part_of_the_retriever_name(self) -> None:
        """`name` là chuỗi "mọi thứ làm đổi con số" (`TD-38`); tenant quyết định
        *ai đọc được*, không phải kết quả của một lần eval. Đưa nó vào đó sẽ làm
        mọi bundle đã ký hỏng chữ ký vì một lý do không liên quan."""
        assert self._store("acme").name == self._store("globex").name
