"""Chunking — ba chiến lược sau một interface chung."""

from ..embedding.base import EmbeddingProvider
from .base import Chunker, ChunkingConfig, ChunkingStrategy
from .cache import CachedChunker, CacheStats, SQLiteChunkCache
from .fixed import FixedSizeChunker, split_recursive
from .hybrid import HybridChunker
from .parent_child import ParentChildChunker, parent_id
from .semantic import SemanticChunker, split_sentences
from .structure import StructureChunker, common_ancestor, section_boundaries
from .tokens import TokenCounter, TokenSizingUnavailable, calibrate_density, fit_to_budget

__all__ = [
    "CacheStats",
    "CachedChunker",
    "Chunker",
    "ChunkingConfig",
    "ChunkingStrategy",
    "FixedSizeChunker",
    "HybridChunker",
    "ParentChildChunker",
    "SQLiteChunkCache",
    "SemanticChunker",
    "StructureChunker",
    "TokenCounter",
    "TokenSizingUnavailable",
    "build_chunker",
    "calibrate_density",
    "common_ancestor",
    "fit_to_budget",
    "parent_id",
    "section_boundaries",
    "split_recursive",
    "split_sentences",
]


def build_chunker(
    config: ChunkingConfig | None = None,
    embeddings: EmbeddingProvider | None = None,
) -> Chunker:
    """Factory theo `config.strategy` — điểm duy nhất map config sang chunker."""
    cfg = config or ChunkingConfig()
    if cfg.strategy is ChunkingStrategy.FIXED:
        return FixedSizeChunker(cfg, token_counter=embeddings)
    if cfg.strategy is ChunkingStrategy.STRUCTURE:
        return StructureChunker(cfg, token_counter=embeddings)
    if cfg.strategy is ChunkingStrategy.PARENT_CHILD:
        return ParentChildChunker(cfg, token_counter=embeddings)
    if cfg.strategy is ChunkingStrategy.SEMANTIC:
        if embeddings is None:
            raise ValueError("strategy=semantic bắt buộc phải có EmbeddingProvider")
        return SemanticChunker(embeddings, cfg)
    return HybridChunker(embeddings, cfg)
