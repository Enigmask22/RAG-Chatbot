"""Semantic cache trên Redis cho `/chat`. `W4-10`.

Ba quyết định, cả ba đến từ phép đo `probes/w4-10-cosine-threshold.json`
(BGE-M3, 30 cặp câu tiếng Việt chia ba lớp):

1. **Ngưỡng 0,96, không phải 0,95 của plan.** Hai phân bố paraphrase và
   "gần giống nhưng đổi đáp án" chồng lên nhau gần hết (p50: 0,8717 vs 0,8659)
   — không tồn tại ngưỡng tách được chúng. 0,96 nằm trên bẫy cao nhất đo được
   (0,9410 — "Thu ngân sách" vs "Chi ngân sách") một biên 0,019; cái giá là chỉ
   2/10 paraphrase thật vượt được ngưỡng. Cache này vì thế là cache **bảo
   thủ**: giá trị chính là câu hỏi lặp lại nguyên văn (reload UI, demo), không
   phải paraphrase tự do.
2. **Hàng rào token chữ số.** "đạt 7,5%" vs "đạt 5,7%" là hai câu hỏi khác
   nhau ở đúng một hoán vị chữ số, cosine 0,9112. Mọi token mang chữ số của
   hai câu phải trùng nhau như **multiset** — tất định, không ngưỡng nào chỉnh
   được nó.
3. **Namespace = `{tenant}:{bundle_version}`.** Đổi bundle là đổi index, đổi
   prompt, đổi mọi thứ phía sau câu trả lời — nên "invalidate khi đổi bundle"
   không phải một lệnh xoá mà là một **thuộc tính của khoá**: bundle mới nhìn
   vào một namespace trống. Tenant trong khoá vì một câu trả lời cache của
   tenant này trả cho tenant kia là rò dữ liệu, không phải cache hit.

Hỏng thì hỏng về phía **miss**: Redis chết, entry vỡ, vector lệch chiều — tất
cả đều trả "không có" và request đi đường đầy đủ. Một cache không bao giờ được
phép là lý do `/chat` trả lỗi.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

__all__ = ["CachedAnswer", "SemanticCache", "digit_tokens", "embedder_of"]

logger = logging.getLogger(__name__)

_DIGIT_TOKEN = re.compile(r"\d[\d.,%/]*")


def digit_tokens(text: str) -> tuple[str, ...]:
    """Multiset (đã sắp) các token mang chữ số. `"7,5%" != "5,7%"` là mục đích."""
    return tuple(sorted(_DIGIT_TOKEN.findall(text)))


def embedder_of(retriever: Any) -> Any | None:
    """Đào embedder ra khỏi retriever, xuyên qua lớp wrap của reranker.

    Duck-typing có chủ đích: `RerankedRetriever.base` → `.store.embeddings`.
    Không thấy thì trả `None` và cache tự tắt — một retriever giả trong test
    không có embedder là chuyện bình thường, không phải lỗi cấu hình.
    """
    for candidate in (retriever, getattr(retriever, "base", None)):
        store = getattr(candidate, "store", None)
        embeddings = getattr(store, "embeddings", None)
        if embeddings is not None and hasattr(embeddings, "embed_query"):
            return embeddings
    return None


class _RedisLike(Protocol):
    async def hgetall(self, key: str) -> dict[bytes, bytes]: ...
    async def hset(self, key: str, field: str, value: str) -> Any: ...
    async def hdel(self, key: str, *fields: str) -> Any: ...
    async def expire(self, key: str, seconds: int) -> Any: ...


@dataclass(frozen=True)
class CachedAnswer:
    """Một lượt trả lời đã cache, đủ để phát lại nguyên bộ khung SSE."""

    question: str
    text: str
    sources: list[dict[str, Any]]
    citations_frame: dict[str, Any] | None
    model: str
    similarity: float


class SemanticCache:
    """Tra và ghi câu trả lời theo vector câu hỏi. Mọi lỗi hạ tầng = miss."""

    def __init__(
        self,
        redis: _RedisLike,
        *,
        threshold: float = 0.96,
        ttl_s: int = 86_400,
        max_entries: int = 128,
    ) -> None:
        self.redis = redis
        self.threshold = threshold
        self.ttl_s = ttl_s
        self.max_entries = max_entries

    @staticmethod
    def _key(tenant: str, bundle_version: str) -> str:
        return f"semcache:{tenant}:{bundle_version}"

    async def lookup(
        self, tenant: str, bundle_version: str, question: str, vector: np.ndarray
    ) -> CachedAnswer | None:
        try:
            raw = await self.redis.hgetall(self._key(tenant, bundle_version))
        except Exception as exc:
            logger.warning("semantic cache lookup hỏng (coi như miss): %s", exc)
            return None
        if not raw:
            return None
        guard = digit_tokens(question)
        query = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return None
        query = query / norm
        best: tuple[float, dict[str, Any]] | None = None
        for value in raw.values():
            try:
                entry = json.loads(value)
                stored = np.frombuffer(base64.b64decode(entry["v"]), dtype=np.float32)
            except Exception:
                continue
            if stored.shape != query.shape:
                continue
            similarity = float(np.dot(stored, query))
            if similarity < self.threshold:
                continue
            if digit_tokens(entry["q"]) != guard:
                # Cosine nói "giống" nhưng chữ số nói "khác" — chữ số thắng.
                continue
            if best is None or similarity > best[0]:
                best = (similarity, entry)
        if best is None:
            return None
        similarity, entry = best
        return CachedAnswer(
            question=entry["q"],
            text=entry["t"],
            sources=entry.get("s", []),
            citations_frame=entry.get("c"),
            model=entry.get("m", "cache"),
            similarity=round(similarity, 4),
        )

    async def store(
        self,
        tenant: str,
        bundle_version: str,
        question: str,
        vector: np.ndarray,
        *,
        text: str,
        sources: list[dict[str, Any]],
        citations_frame: dict[str, Any] | None,
        model: str,
    ) -> None:
        query = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return
        entry = {
            "q": question,
            "v": base64.b64encode((query / norm).tobytes()).decode("ascii"),
            "t": text,
            "s": sources,
            "c": citations_frame,
            "m": model,
            "ts": time.time(),
        }
        key = self._key(tenant, bundle_version)
        try:
            await self.redis.hset(key, uuid.uuid4().hex, json.dumps(entry, ensure_ascii=False))
            await self.redis.expire(key, self.ttl_s)
            await self._trim(key)
        except Exception as exc:
            logger.warning("semantic cache store hỏng (bỏ qua): %s", exc)

    async def _trim(self, key: str) -> None:
        """Giữ tối đa `max_entries` — xoá entry cũ nhất theo `ts`.

        Đọc cả hash để trim là O(N), nhưng N bị chính hàm này chặn ở 128 và
        lookup cũng đã đọc cả hash — chi phí cùng bậc với đường nóng.
        """
        raw = await self.redis.hgetall(key)
        if len(raw) <= self.max_entries:
            return
        aged: list[tuple[float, bytes]] = []
        for field, value in raw.items():
            try:
                aged.append((float(json.loads(value).get("ts", 0.0)), field))
            except Exception:
                aged.append((0.0, field))
        aged.sort()
        doomed = [
            field.decode() if isinstance(field, bytes) else field
            for _, field in aged[: len(raw) - self.max_entries]
        ]
        if doomed:
            await self.redis.hdel(key, *doomed)
