"""W1-06 — abstraction embedding: shape, determinism, batch == single."""

from __future__ import annotations

import numpy as np
import pytest

from rag_core.embedding import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
    build_embedding_provider,
    l2_normalize,
)

# Không xuất từ `rag_core.embedding.__init__` — import thẳng từ module con.
from rag_core.embedding.huggingface import HuggingFaceEmbeddingProvider


@pytest.fixture
def provider() -> HashingEmbeddingProvider:
    return HashingEmbeddingProvider(dimension=64)


TEXTS = ["ngân sách nhà nước", "public investment budget", "đội bóng ghi bàn"]


class TestContract:
    def test_shape_and_dtype(self, provider: EmbeddingProvider) -> None:
        vectors = provider.embed_documents(TEXTS)
        assert vectors.shape == (len(TEXTS), provider.dimension)
        assert vectors.dtype == np.float32

    def test_query_shape(self, provider: EmbeddingProvider) -> None:
        assert provider.embed_query("ngân sách").shape == (provider.dimension,)

    def test_empty_input(self, provider: EmbeddingProvider) -> None:
        assert provider.embed_documents([]).shape == (0, provider.dimension)

    def test_deterministic(self, provider: EmbeddingProvider) -> None:
        np.testing.assert_array_equal(
            provider.embed_documents(TEXTS), provider.embed_documents(TEXTS)
        )

    def test_batch_equals_single(self, provider: EmbeddingProvider) -> None:
        """Nghe hiển nhiên nhưng padding trong một số backend làm sai điều này,
        và hậu quả là điểm truy hồi đổi theo batch size — cực khó truy."""
        batched = provider.embed_documents(TEXTS)
        one_by_one = np.vstack([provider.embed_documents([t]) for t in TEXTS])
        np.testing.assert_allclose(batched, one_by_one, rtol=1e-6, atol=1e-7)

    def test_order_preserved(self, provider: EmbeddingProvider) -> None:
        vectors = provider.embed_documents(TEXTS)
        for i, text in enumerate(TEXTS):
            np.testing.assert_allclose(vectors[i], provider.embed_query(text), rtol=1e-6)

    def test_normalized(self, provider: EmbeddingProvider) -> None:
        norms = np.linalg.norm(provider.embed_documents(TEXTS), axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


class TestSemanticSanity:
    """Provider hashing phải phản ánh tương đồng từ vựng thật — nếu không thì
    mọi test dùng nó (semantic chunking, metric) đều thành vô nghĩa."""

    def test_similar_texts_are_closer(self, provider: EmbeddingProvider) -> None:
        a = provider.embed_query("ngân sách nhà nước tăng chi đầu tư công")
        b = provider.embed_query("ngân sách nhà nước tăng chi thường xuyên")
        c = provider.embed_query("đội bóng ghi bàn thắng phút cuối")
        assert float(a @ b) > float(a @ c)

    def test_identical_text_similarity_is_one(self, provider: EmbeddingProvider) -> None:
        v = provider.embed_query("ngân sách")
        assert float(v @ v) == pytest.approx(1.0, rel=1e-6)


class TestHelpers:
    def test_l2_normalize_handles_zero_vector(self) -> None:
        out = l2_normalize(np.zeros((2, 4), dtype=np.float32))
        assert not np.isnan(out).any()

    def test_rejects_tiny_dimension(self) -> None:
        with pytest.raises(ValueError, match="va chạm hash"):
            HashingEmbeddingProvider(dimension=4)

    def test_factory_builds_hashing_provider(self) -> None:
        provider = build_embedding_provider("hashing:32")
        assert provider.dimension == 32

    def test_provider_name_is_reproducible(self, provider: EmbeddingProvider) -> None:
        # `name` đi vào cache key và vào MLflow — phải đủ để tái lập.
        assert provider.name == "hashing-64d"


class TestConcurrentEncodeIsSerialised:
    """`W5-01`: cùng lỗ hổng với cross-encoder, cùng nguyên nhân.

    Tokenizer "fast" là đối tượng Rust dùng `RefCell`; `sentence-transformers`
    sửa cấu hình truncation/padding của nó ngay trong lời gọi. Đường serving
    embed câu hỏi trong `asyncio.to_thread`, nên hai request đồng thời là đủ để
    một trong hai nhận `RuntimeError: Already borrowed`.
    """

    def test_same_model_and_device_share_one_lock(self) -> None:
        first = HuggingFaceEmbeddingProvider("mo-hinh/gia", device="cpu")
        second = HuggingFaceEmbeddingProvider("mo-hinh/gia", device="cpu")
        assert first is not second
        assert first.lock is second.lock

    def test_khac_model_thi_khac_khoa(self) -> None:
        first = HuggingFaceEmbeddingProvider("mo-hinh/gia", device="cpu")
        other = HuggingFaceEmbeddingProvider("mo-hinh/khac", device="cpu")
        assert first.lock is not other.lock

    def test_two_threads_never_enter_encode_together(self) -> None:
        import threading
        import time

        inside = 0
        overlaps = 0
        guard = threading.Lock()

        class SpyModel:
            def encode(self, texts: list[str], **_: object) -> np.ndarray:
                nonlocal inside, overlaps
                with guard:
                    inside += 1
                    if inside > 1:
                        overlaps += 1
                time.sleep(0.01)
                with guard:
                    inside -= 1
                return np.ones((len(texts), 4), dtype=np.float32)

        provider = HuggingFaceEmbeddingProvider("mo-hinh/gia", device="cpu")
        provider._model = SpyModel()  # type: ignore[assignment]

        threads = [
            threading.Thread(target=provider.embed_documents, args=(["a", "b"],)) for _ in range(6)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert overlaps == 0, f"{overlaps} lần hai luồng cùng ở trong `encode`"
