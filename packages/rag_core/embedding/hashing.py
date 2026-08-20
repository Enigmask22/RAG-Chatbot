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

from .base import EmbeddingProvider, FloatArray, HybridVectors, l2_normalize
from .sparse import SparseVector

__all__ = ["HASHING_SPARSE_VOCAB", "HashingEmbeddingProvider"]

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

#: Kích thước không gian sparse giả lập. 2^16 đủ thưa để va chạm hash không làm
#: hỏng ý nghĩa của test, và đủ nhỏ để đọc bằng mắt lúc debug.
HASHING_SPARSE_VOCAB = 65_536


class HashingEmbeddingProvider(EmbeddingProvider):
    """Provider hashing — nay sinh **cả** dense và sparse.

    Sparse ở đây không phải thêm cho đủ: `W2-02`…`W2-04` cần một provider sinh
    sparse mà **không** cần GPU và không cần 2,2GB trọng số, nếu không thì mọi
    test của schema hybrid, sparse retriever và RRF đều phải chạy BGE-M3 thật.
    Phải bật tường minh bằng `sparse=True` — xem `__init__`.

    Cấu trúc của nó cũng đúng với thứ đang mô phỏng: dense là bag-of-words đã
    băm xuống `dimension` chiều, sparse là **cùng** bag-of-words băm xuống một
    không gian rộng hơn (`HASHING_SPARSE_VOCAB`) mà không cộng dồn — tức là
    tương đồng từ vựng thật, giống trọng số lexical của BGE-M3 về mặt hành vi
    (khớp token nào thì được điểm), dù không giống về mặt cách sinh ra.

    ⚠️ Vẫn **không dùng cho eval thật** — nó không nắm được ngữ nghĩa.
    """

    def __init__(
        self,
        dimension: int = 256,
        *,
        lowercase: bool = True,
        sparse: bool = False,
    ) -> None:
        if dimension < 8:
            raise ValueError("dimension quá nhỏ, va chạm hash sẽ làm kết quả vô nghĩa")
        self._dimension = dimension
        self._lowercase = lowercase
        # Mặc định **tắt**, và đó là quyết định có chủ ý. `name` đi vào cache key
        # của semantic chunker (`chunking/semantic.py`) và vào MLflow, nên bật sẵn
        # sparse sẽ đổi tên của mọi provider mặc định — tức vô hiệu cache chunk và
        # làm mọi test hiện có đổi nghĩa một cách âm thầm. Sparse là năng lực thêm
        # cho một nhu cầu cụ thể (`W2-02`…`W2-04`), nên phải xin thì mới có.
        self._sparse = sparse
        self.name = f"hashing-{dimension}d" + ("+sparse" if sparse else "")

    @property
    def dimension(self) -> int:
        return self._dimension

    def _bucket(self, token: str, modulus: int) -> int:
        """Băm token vào `[0, modulus)`. Cùng một digest, hai không gian đích:
        `dimension` cho dense, `HASHING_SPARSE_VOCAB` cho sparse."""
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % modulus

    def _embed_one(self, text: str) -> FloatArray:
        vec = np.zeros(self._dimension, dtype=np.float32)
        source = text.lower() if self._lowercase else text
        counts = Counter(_TOKEN_RE.findall(source))
        for token, count in counts.items():
            # log(1+tf): giảm ảnh hưởng của từ lặp nhiều lần, giống tf-idf
            vec[self._bucket(token, self._dimension)] += float(np.log1p(count))
        return vec

    def embed_documents(self, texts: Sequence[str]) -> FloatArray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        matrix = np.vstack([self._embed_one(t) for t in texts]).astype(np.float32)
        return l2_normalize(matrix)

    # --------------------------------------------------------------- sparse

    @property
    def sparse_vocab_size(self) -> int | None:
        return HASHING_SPARSE_VOCAB if self._sparse else None

    def _sparse_one(self, text: str) -> SparseVector:
        source = text.lower() if self._lowercase else text
        weights: dict[int, float] = {}
        for token, count in Counter(_TOKEN_RE.findall(source)).items():
            bucket = self._bucket(token, HASHING_SPARSE_VOCAB)
            value = float(np.log1p(count))
            # `max` chứ không `+=`, khớp cách BGE-M3 gộp trọng số theo token: hai
            # token khác nhau va vào cùng bucket không được cộng lại thành một
            # token "quan trọng gấp đôi".
            if value > weights.get(bucket, 0.0):
                weights[bucket] = value
        return SparseVector.from_weights(weights)

    def embed_documents_hybrid(self, texts: Sequence[str]) -> HybridVectors | None:
        if not self._sparse:
            return None
        return HybridVectors(
            dense=self.embed_documents(texts),
            sparse=[self._sparse_one(t) for t in texts],
        )

    def embed_query_hybrid(self, text: str) -> tuple[FloatArray, SparseVector] | None:
        if not self._sparse:
            return None
        return self.embed_query(text), self._sparse_one(text)
