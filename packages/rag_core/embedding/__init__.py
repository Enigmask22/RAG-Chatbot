"""Tầng embedding — đổi model bằng config, không sửa code."""

from .base import EmbeddingProvider, FloatArray, HybridVectors, l2_normalize
from .bge_m3 import BGE_M3_MODEL  # chỉ hằng tên model — `torch` vẫn nạp lazy
from .hashing import HashingEmbeddingProvider
from .sparse import SparseVector
from .truncation import TruncationStats, measure_truncation, token_stats

__all__ = [
    "BGE_M3_MODEL",
    "EmbeddingProvider",
    "FloatArray",
    "HashingEmbeddingProvider",
    "HybridVectors",
    "SparseVector",
    "TruncationStats",
    "build_embedding_provider",
    "l2_normalize",
    "measure_truncation",
    "token_stats",
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
    `hashing:256`). `BAAI/bge-m3` được nhận diện riêng vì nó cần provider sinh
    thêm sparse — chọn theo **tên model** chứ không thêm một cờ `use_bge_m3`
    trong config, để không tồn tại được cấu hình `model=bge-m3, use_bge_m3=false`
    vừa hợp lệ về cú pháp vừa vô nghĩa về nội dung.

    Mọi tên khác coi là model HuggingFace dense-only.
    """
    if model_name.startswith("hashing:"):
        return HashingEmbeddingProvider(dimension=int(model_name.split(":", 1)[1]))

    if model_name == BGE_M3_MODEL:
        from .bge_m3 import BgeM3EmbeddingProvider

        return BgeM3EmbeddingProvider(
            model_name,
            device=device,
            batch_size=batch_size,
            normalize=normalize,
            **kwargs,  # type: ignore[arg-type]
        )

    from .huggingface import HuggingFaceEmbeddingProvider

    return HuggingFaceEmbeddingProvider(
        model_name,
        device=device,
        batch_size=batch_size,
        normalize=normalize,
        **kwargs,  # type: ignore[arg-type]
    )
