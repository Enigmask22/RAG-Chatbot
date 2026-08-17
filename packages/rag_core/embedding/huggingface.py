"""Embedding provider chạy model HuggingFace cục bộ qua `sentence-transformers`.

Import `sentence_transformers` được để **lazy**: cả `torch` lẫn trọng số model
đều nặng, mà phần lớn unit test không cần tới. Nạp lúc khởi tạo provider chứ
không phải lúc import module giữ `make test` ở mức vài giây.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .base import EmbeddingProvider, FloatArray, l2_normalize

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

__all__ = ["HuggingFaceEmbeddingProvider", "resolve_device"]


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:  # pragma: no cover - phụ thuộc vào môi trường
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=4)
def _load_model(model_name: str, device: str) -> SentenceTransformer:
    """Nạp model một lần cho mỗi cặp (model, device).

    Giới hạn 4 để một lần ablation quét nhiều model không âm thầm giữ hết trong
    VRAM — 4060 8GB không chịu nổi và lỗi OOM sẽ xuất hiện ở giữa job.
    """
    from sentence_transformers import SentenceTransformer

    return cast("SentenceTransformer", SentenceTransformer(model_name, device=device))


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Bọc `SentenceTransformer` sau interface `EmbeddingProvider`.

    `query_prefix`/`document_prefix` phục vụ các model bất đối xứng (E5, BGE).
    Để rỗng với model đối xứng như `vietnamese-bi-encoder`.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        batch_size: int = 32,
        normalize: bool = True,
        query_prefix: str = "",
        document_prefix: str = "",
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.normalize = normalize
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self._model_kwargs = model_kwargs or {}
        self.name = f"{model_name}@{self.device}"
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = _load_model(self.model_name, self.device)
        return self._model

    @property
    def dimension(self) -> int:
        # sentence-transformers 5.x đổi tên `get_sentence_embedding_dimension`
        # thành `get_embedding_dimension`. Thử tên mới trước rồi mới lùi về tên
        # cũ để chạy được trên cả hai — image RunPod thường ghim bản khác laptop.
        getter = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        dim = getter()
        if dim is None:  # pragma: no cover - model hỏng
            raise RuntimeError(f"Không đọc được số chiều của model {self.model_name!r}")
        return int(dim)

    def _encode(self, texts: Sequence[str]) -> FloatArray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
            **self._model_kwargs,
        )
        matrix = cast(FloatArray, np.asarray(vectors, dtype=np.float32))
        return l2_normalize(matrix) if self.normalize else matrix

    def embed_documents(self, texts: Sequence[str]) -> FloatArray:
        prefixed = [self.document_prefix + t for t in texts] if self.document_prefix else texts
        return self._encode(prefixed)

    def embed_query(self, text: str) -> FloatArray:
        return cast(FloatArray, self._encode([self.query_prefix + text])[0])
