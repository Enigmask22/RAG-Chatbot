"""Chunking — ba chiến lược sau một interface chung."""

from ..embedding.base import EmbeddingProvider
from .base import Chunker, ChunkingConfig, ChunkingStrategy
from .cache import CachedChunker, CacheStats, SQLiteChunkCache
from .fixed import FixedSizeChunker, split_recursive
from .hybrid import HybridChunker
from .semantic import SemanticChunker, split_sentences
from .structure import StructureChunker, common_ancestor, section_boundaries

__all__ = [
    "CacheStats",
    "CachedChunker",
    "Chunker",
    "ChunkingConfig",
    "ChunkingStrategy",
    "FixedSizeChunker",
    "HybridChunker",
    "SQLiteChunkCache",
    "SemanticChunker",
    "StructureChunker",
    "build_chunker",
    "common_ancestor",
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
        return FixedSizeChunker(cfg)
    if cfg.strategy is ChunkingStrategy.STRUCTURE:
        return StructureChunker(cfg)
    if cfg.strategy is ChunkingStrategy.SEMANTIC:
        if embeddings is None:
            raise ValueError("strategy=semantic bắt buộc phải có EmbeddingProvider")
        return SemanticChunker(embeddings, cfg)
    return HybridChunker(embeddings, cfg)
