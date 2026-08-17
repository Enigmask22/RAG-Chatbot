"""Embedding provider bằng hashing trick — không cần model, không cần mạng.

Vì sao có thứ này thay vì một fake trả vector ngẫu nhiên: unit test của semantic
chunking và của metric cần embedding **có ý nghĩa tương đồng thật** (hai câu
cùng chủ đề phải gần nhau). Vector ngẫu nhiên deterministic thoả mãn "chạy được"
nhưng biến bài test thành vô nghĩa — nó sẽ pass kể cả khi thuật toán sai.

Đây là bag-of-words hashing đã chuẩn hoá: đủ để phản ánh tương đồng từ vựng,
chạy trong micro giây, không phụ thuộc `torch`. Dùng cho test và cho CI. **Không
dùng cho eval thật** — không nắm được ngữ nghĩa.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

from .base import EmbeddingProvider, FloatArray, l2_normalize

__all__ = ["HashingEmbeddingProvider"]

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class HashingEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimension: int = 256, *, lowercase: bool = True) -> None:
        if dimension < 8:
            raise ValueError("dimension quá nhỏ, va chạm hash sẽ làm kết quả vô nghĩa")
        self._dimension = dimension
        self._lowercase = lowercase
        self.name = f"hashing-{dimension}d"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self._dimension

    def _embed_one(self, text: str) -> FloatArray:
        vec = np.zeros(self._dimension, dtype=np.float32)
        source = text.lower() if self._lowercase else text
        counts = Counter(_TOKEN_RE.findall(source))
        for token, count in counts.items():
            # log(1+tf): giảm ảnh hưởng của từ lặp nhiều lần, giống tf-idf
            vec[self._bucket(token)] += float(np.log1p(count))
        return vec

    def embed_documents(self, texts: Sequence[str]) -> FloatArray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        matrix = np.vstack([self._embed_one(t) for t in texts]).astype(np.float32)
        return l2_normalize(matrix)
