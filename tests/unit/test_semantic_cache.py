"""`W4-10` — semantic cache: ngưỡng, hàng rào chữ số, namespace, và suy giảm thành miss.

Redis ở đây là fake dict-based: hành vi cần kiểm là của `SemanticCache`
(so khớp, guard, trim, fail-open), không phải của Redis. Đường Redis thật nằm ở
`tests/integration/test_semantic_cache.py`.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from serving.core.semantic_cache import (
    CachedAnswer,
    SemanticCache,
    digit_tokens,
    embedder_of,
)

# ---------------------------------------------------------------------------
# Fake hạ tầng
# ---------------------------------------------------------------------------


class FakeRedis:
    """Đủ bốn lệnh mà `SemanticCache` dùng. Lưu bytes như Redis thật."""

    def __init__(self) -> None:
        self.data: dict[str, dict[bytes, bytes]] = {}
        self.ttls: dict[str, int] = {}

    async def hgetall(self, key: str) -> dict[bytes, bytes]:
        return dict(self.data.get(key, {}))

    async def hset(self, key: str, field: str, value: str) -> None:
        self.data.setdefault(key, {})[field.encode()] = value.encode()

    async def hdel(self, key: str, *fields: str) -> None:
        for field in fields:
            self.data.get(key, {}).pop(field.encode(), None)

    async def expire(self, key: str, seconds: int) -> None:
        self.ttls[key] = seconds


class BrokenRedis:
    async def hgetall(self, key: str) -> dict[bytes, bytes]:
        raise ConnectionError("redis chết")

    async def hset(self, key: str, field: str, value: str) -> None:
        raise ConnectionError("redis chết")

    async def hdel(self, key: str, *fields: str) -> None:
        raise ConnectionError("redis chết")

    async def expire(self, key: str, seconds: int) -> None:
        raise ConnectionError("redis chết")


def _vector(direction: int, dim: int = 8) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[direction] = 1.0
    return v


def _blend(a: np.ndarray, b: np.ndarray, weight: float) -> np.ndarray:
    """Vector có cosine ĐÚNG bằng `weight` với `a` (khi a ⊥ b, chuẩn hoá)."""
    mixed = weight * a + float(np.sqrt(1.0 - weight**2)) * b
    return (mixed / np.linalg.norm(mixed)).astype(np.float32)


async def _store(
    cache: SemanticCache, question: str, vector: np.ndarray, text: str = "đáp"
) -> None:
    await cache.store(
        "acme",
        "0.2.0",
        question,
        vector,
        text=text,
        sources=[{"n": 1, "chunk_id": "c1"}],
        citations_frame={"block": "ok", "citations": []},
        model="fake-model",
    )


# ---------------------------------------------------------------------------
# 1. digit_tokens — hàng rào tất định
# ---------------------------------------------------------------------------


class TestDigitTokens:
    def test_a_digit_swap_changes_the_tokens(self) -> None:
        """`7,5%` vs `5,7%` — cosine đo được 0,9112, chỉ hàng rào này chặn."""
        assert digit_tokens("đạt 7,5% đúng không") != digit_tokens("đạt 5,7% đúng không")

    def test_a_year_change_changes_the_tokens(self) -> None:
        assert digit_tokens("GDP năm 2025") != digit_tokens("GDP năm 2024")

    def test_order_does_not_matter_it_is_a_multiset(self) -> None:
        assert digit_tokens("từ 2024 đến 2025") == digit_tokens("từ 2025 về 2024")

    def test_no_digits_is_an_empty_guard(self) -> None:
        assert digit_tokens("lạm phát ở mức nào") == ()


# ---------------------------------------------------------------------------
# 2. lookup — ngưỡng, guard, namespace
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_identical_question_hits_with_similarity_one() -> None:
    cache = SemanticCache(FakeRedis(), threshold=0.96)
    v = _vector(0)
    await _store(cache, "lạm phát ở mức nào?", v, text="ổn định")

    hit = await cache.lookup("acme", "0.2.0", "lạm phát ở mức nào?", v)

    assert isinstance(hit, CachedAnswer)
    assert hit.text == "ổn định"
    assert hit.similarity == 1.0
    assert hit.question == "lạm phát ở mức nào?"


@pytest.mark.asyncio
async def test_below_the_threshold_is_a_miss() -> None:
    """0,96 đến từ phép đo: bẫy cao nhất 0,9410 — 0,95 của plan chỉ chừa 0,009."""
    cache = SemanticCache(FakeRedis(), threshold=0.96)
    await _store(cache, "câu gốc", _vector(0))

    near = _blend(_vector(0), _vector(1), 0.95)
    assert await cache.lookup("acme", "0.2.0", "câu gần", near) is None

    nearer = _blend(_vector(0), _vector(1), 0.97)
    assert await cache.lookup("acme", "0.2.0", "câu gần hơn", nearer) is not None


@pytest.mark.asyncio
async def test_cosine_cannot_override_a_digit_mismatch() -> None:
    """Vector TRÙNG HỆT (cosine 1,0) mà chữ số khác vẫn phải miss — hai câu chỉ
    khác một con số là hai câu hỏi khác nhau, và embedding không thấy điều đó
    đủ rõ (phép đo: 0,9112 cho hoán vị `7,5`/`5,7`)."""
    cache = SemanticCache(FakeRedis(), threshold=0.96)
    v = _vector(0)
    await _store(cache, "tín dụng đạt 7,5% đúng không?", v)

    assert await cache.lookup("acme", "0.2.0", "tín dụng đạt 5,7% đúng không?", v) is None
    assert await cache.lookup("acme", "0.2.0", "tín dụng đạt 7,5% đúng không?", v) is not None


@pytest.mark.asyncio
async def test_the_best_entry_wins_not_the_first_above_threshold() -> None:
    cache = SemanticCache(FakeRedis(), threshold=0.90)
    await _store(cache, "câu A", _blend(_vector(0), _vector(1), 0.97), text="A")
    await _store(cache, "câu B", _vector(0), text="B")

    hit = await cache.lookup("acme", "0.2.0", "câu hỏi", _vector(0))

    assert hit is not None
    assert hit.text == "B"


@pytest.mark.asyncio
async def test_tenants_never_share_a_namespace() -> None:
    """Cache hit xuyên tenant không phải hit — nó là rò dữ liệu."""
    cache = SemanticCache(FakeRedis(), threshold=0.96)
    v = _vector(0)
    await _store(cache, "câu hỏi", v)

    assert await cache.lookup("khac", "0.2.0", "câu hỏi", v) is None


@pytest.mark.asyncio
async def test_a_new_bundle_version_sees_an_empty_cache() -> None:
    """DoD "invalidate khi đổi bundle": không phải một lệnh xoá mà là thuộc
    tính của khoá — bundle mới nhìn vào namespace trống."""
    cache = SemanticCache(FakeRedis(), threshold=0.96)
    v = _vector(0)
    await _store(cache, "câu hỏi", v)

    assert await cache.lookup("acme", "0.2.0", "câu hỏi", v) is not None
    assert await cache.lookup("acme", "0.3.0", "câu hỏi", v) is None


# ---------------------------------------------------------------------------
# 3. Suy giảm — mọi lỗi hạ tầng là miss, không phải 500
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_dead_redis_is_a_miss_on_lookup_and_silent_on_store() -> None:
    cache = SemanticCache(BrokenRedis(), threshold=0.96)

    assert await cache.lookup("acme", "0.2.0", "câu", _vector(0)) is None
    await _store(cache, "câu", _vector(0))  # không ném


@pytest.mark.asyncio
async def test_a_corrupted_entry_is_skipped_not_fatal() -> None:
    redis = FakeRedis()
    cache = SemanticCache(redis, threshold=0.96)
    v = _vector(0)
    await _store(cache, "câu lành", v, text="lành")
    redis.data[SemanticCache._key("acme", "0.2.0")][b"hong"] = b"khong phai json"

    hit = await cache.lookup("acme", "0.2.0", "câu lành", v)

    assert hit is not None
    assert hit.text == "lành"


@pytest.mark.asyncio
async def test_a_dimension_mismatch_is_skipped() -> None:
    """Đổi embedding model là đổi số chiều — entry cũ không so được thì bỏ
    qua, không ném. (Namespace theo bundle đã chặn hầu hết ca này; đây là
    lưới đỡ cho ca hai bundle trùng version nhưng khác model — một lỗi vận
    hành, và cache không phải chỗ nó được phép nổ.)"""
    redis = FakeRedis()
    cache = SemanticCache(redis, threshold=0.5)
    await _store(cache, "câu", _vector(0, dim=8))

    assert await cache.lookup("acme", "0.2.0", "câu", _vector(0, dim=16)) is None


@pytest.mark.asyncio
async def test_a_zero_vector_never_hits_nor_stores(caplog: pytest.LogCaptureFixture) -> None:
    redis = FakeRedis()
    cache = SemanticCache(redis, threshold=0.96)
    zero = np.zeros(8, dtype=np.float32)

    await _store(cache, "câu", zero)
    assert not redis.data
    assert await cache.lookup("acme", "0.2.0", "câu", zero) is None


# ---------------------------------------------------------------------------
# 4. store — chuẩn hoá, TTL, trim
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_normalizes_the_vector_so_dot_is_cosine() -> None:
    redis = FakeRedis()
    cache = SemanticCache(redis, threshold=0.96)
    await _store(cache, "câu", _vector(0) * 7.3)

    hit = await cache.lookup("acme", "0.2.0", "câu", _vector(0) * 0.2)

    assert hit is not None
    assert hit.similarity == 1.0


@pytest.mark.asyncio
async def test_every_store_refreshes_the_namespace_ttl() -> None:
    redis = FakeRedis()
    cache = SemanticCache(redis, threshold=0.96, ttl_s=123)
    await _store(cache, "câu", _vector(0))

    assert redis.ttls[SemanticCache._key("acme", "0.2.0")] == 123


@pytest.mark.asyncio
async def test_the_oldest_entry_is_evicted_at_the_cap() -> None:
    redis = FakeRedis()
    cache = SemanticCache(redis, threshold=0.96, max_entries=2)
    await _store(cache, "câu một", _vector(0), text="một")
    await _store(cache, "câu hai", _vector(1), text="hai")
    await _store(cache, "câu ba", _vector(2), text="ba")

    key = SemanticCache._key("acme", "0.2.0")
    remaining = {json.loads(v)["t"] for v in redis.data[key].values()}
    assert remaining == {"hai", "ba"}


# ---------------------------------------------------------------------------
# 5. embedder_of — đào xuyên lớp wrap
# ---------------------------------------------------------------------------


class _Embedder:
    def embed_query(self, text: str) -> Any:
        return _vector(0)


class _Store:
    embeddings = _Embedder()


class TestEmbedderOf:
    def test_a_bare_store_retriever_exposes_its_embedder(self) -> None:
        class Bare:
            store = _Store()

        assert embedder_of(Bare()) is Bare.store.embeddings

    def test_a_reranked_wrapper_is_pierced_through_base(self) -> None:
        class Bare:
            store = _Store()

        class Wrapped:
            base = Bare()

        assert embedder_of(Wrapped()) is Wrapped.base.store.embeddings

    def test_a_fake_without_embedder_disables_the_cache(self) -> None:
        class NoStore:
            pass

        assert embedder_of(NoStore()) is None

    def test_a_retriever_that_is_its_own_store_exposes_its_embedder(self) -> None:
        """`NEW-08`/`AU-13`: `QdrantDenseRetriever` không có `.store` — chính
        nó LÀ store, `.embeddings` nằm trực tiếp trên nó. Trước sửa này một
        bundle `mode: dense` làm cache tắt câm lặng (trả `None`, không log) —
        fixture cũ có `.store` nên không bắt được, vì nó không cùng hình dạng
        với class thật."""

        class DenseShaped:
            embeddings = _Embedder()

        assert embedder_of(DenseShaped()) is DenseShaped.embeddings

    def test_the_real_dense_retriever_shape_is_recognised(self) -> None:
        """Ghim bằng CLASS THẬT, không phải fixture cùng-hình-dạng-hôm-nay:
        nếu `QdrantDenseRetriever` đổi cấu trúc thuộc tính, bài này đỏ đúng
        ngày đó thay vì cache lại tắt câm lần nữa."""
        from rag_core.retrieval.qdrant_store import QdrantDenseRetriever

        embedder = _Embedder()
        dense = object.__new__(QdrantDenseRetriever)  # không cần client thật
        dense.embeddings = embedder  # type: ignore[assignment]

        assert embedder_of(dense) is embedder
