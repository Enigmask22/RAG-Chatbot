"""Tầng embedding — đổi model bằng config, không sửa code."""

from .base import EmbeddingProvider, FloatArray, l2_normalize
from .hashing import HashingEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "FloatArray",
    "HashingEmbeddingProvider",
    "build_embedding_provider",
    "l2_normalize",
]


def build_embedding_provider(
    model_name: str,
    *,
    device: str = "auto",
    batch_size: int = 32,
    normalize: bool = True,
    **kwargs: object,
) -> EmbeddingProvider:
    """Factory theo tên model — điểm duy nhất mà config biến thành provider.

    Tiền tố `hashing:` chọn provider hashing dùng cho test/CI (ví dụ
    `hashing:256`). Mọi tên khác coi là model HuggingFace.
    """
    if model_name.startswith("hashing:"):
        return HashingEmbeddingProvider(dimension=int(model_name.split(":", 1)[1]))

    from .huggingface import HuggingFaceEmbeddingProvider

    return HuggingFaceEmbeddingProvider(
        model_name,
        device=device,
        batch_size=batch_size,
        normalize=normalize,
        **kwargs,  # type: ignore[arg-type]
    )
