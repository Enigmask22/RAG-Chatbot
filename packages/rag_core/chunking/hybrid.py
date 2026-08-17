"""Chiến lược hybrid: semantic khi kham nổi, fixed khi không.

Semantic chunking tốn một lượt embed toàn bộ câu của tài liệu, nên bản POC chỉ
bật nó khi lô có ít tài liệu. Giữ nguyên quy tắc đó (`hybrid_max_docs_for_semantic`)
vì `W1-13` phải tái lập được hệ thống hiện tại — nhưng ghi lại rằng ngưỡng "số
tài liệu" là một xấp xỉ tồi cho chi phí thật (một tài liệu 500 trang vẫn lọt qua
ngưỡng ≤ 5 tài liệu). Ablation ở W2/W3 sẽ đo xem có nên thay bằng ngưỡng theo số
câu hay không.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import ClassVar

from ..embedding.base import EmbeddingProvider
from ..schemas import Chunk, Document
from .base import Chunker, ChunkingConfig, ChunkingStrategy
from .fixed import FixedSizeChunker
from .semantic import SemanticChunker

__all__ = ["HybridChunker"]

logger = logging.getLogger(__name__)


class HybridChunker(Chunker):
    strategy: ClassVar[ChunkingStrategy] = ChunkingStrategy.HYBRID

    def __init__(
        self,
        embeddings: EmbeddingProvider | None = None,
        config: ChunkingConfig | None = None,
    ) -> None:
        super().__init__(config)
        self._fixed = FixedSizeChunker(self.config)
        self._semantic = SemanticChunker(embeddings, self.config) if embeddings else None
        self.last_strategy_used: str = "fixed"

    def _use_semantic(self, n_documents: int) -> bool:
        return (
            self._semantic is not None and n_documents <= self.config.hybrid_max_docs_for_semantic
        )

    def split_text(self, text: str) -> list[str]:
        return self._delegate(1).split_text(text)

    def _delegate(self, n_documents: int) -> Chunker:
        if self._use_semantic(n_documents) and self._semantic is not None:
            return self._semantic
        return self._fixed

    def chunk(self, documents: Sequence[Document]) -> list[Chunk]:
        delegate = self._delegate(len(documents))
        try:
            chunks = delegate.chunk(documents)
            self.last_strategy_used = delegate.strategy.value
            return chunks
        except Exception:
            if delegate is self._fixed:
                raise
            # Semantic hỏng (model lỗi, OOM) thì vẫn phải ra được chunk — nhưng
            # log ở mức exception, không nuốt. Job index chạy 2 tiếng rồi âm thầm
            # rơi về fixed mà không ai biết là kịch bản tệ nhất cho eval.
            logger.exception("Semantic chunking thất bại, rơi về fixed chunking")
            chunks = self._fixed.chunk(documents)
            self.last_strategy_used = "fixed_fallback"
            return chunks
