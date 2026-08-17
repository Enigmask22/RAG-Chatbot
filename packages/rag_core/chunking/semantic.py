"""Chunking theo ngữ nghĩa — cắt ở chỗ chủ đề đổi, không cắt theo số ký tự.

Cách làm: tách câu → embed từng câu kèm cửa sổ ngữ cảnh hai bên → đo khoảng cách
cosine giữa hai cửa sổ liền kề → cắt ở những chỗ khoảng cách vượt phân vị
`semantic_threshold_percentile`.

Dùng ngưỡng theo **phân vị** chứ không phải hằng số tuyệt đối là có lý do: phân
bố khoảng cách khác nhau rất nhiều giữa các model embedding và giữa tiếng Việt
với tiếng Anh. Ngưỡng tuyệt đối chỉnh vừa cho một model sẽ hỏng khi ablation đổi
model — mà đúng đó là việc W2 sẽ làm.
"""

from __future__ import annotations

import re
from typing import ClassVar

import numpy as np

from ..embedding.base import EmbeddingProvider
from .base import Chunker, ChunkingConfig, ChunkingStrategy

__all__ = ["SemanticChunker", "split_sentences"]

# Cắt sau dấu kết câu (kể cả dấu toàn rộng) hoặc ở dòng trống.
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？…])\s+|\n{2,}")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s and s.strip()]


class SemanticChunker(Chunker):
    strategy: ClassVar[ChunkingStrategy] = ChunkingStrategy.SEMANTIC

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        config: ChunkingConfig | None = None,
    ) -> None:
        super().__init__(config)
        self.embeddings = embeddings

    @property
    def name(self) -> str:
        return f"{self.strategy.value}:{self.embeddings.name}:{self.config.config_hash[:12]}"

    def _context_windows(self, sentences: list[str]) -> list[str]:
        """Ghép mỗi câu với `semantic_buffer_size` câu hai bên.

        Embed câu đơn lẻ cho tín hiệu rất nhiễu — câu ngắn kiểu "Điều 5." gần như
        không mang nội dung. Cửa sổ ngữ cảnh làm mượt tín hiệu đó.
        """
        buffer = self.config.semantic_buffer_size
        if buffer == 0:
            return sentences
        windows: list[str] = []
        for i in range(len(sentences)):
            lo = max(0, i - buffer)
            hi = min(len(sentences), i + buffer + 1)
            windows.append(" ".join(sentences[lo:hi]))
        return windows

    def split_text(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        if len(sentences) < self.config.semantic_min_sentences:
            return [text.strip()] if text.strip() else []

        vectors = np.asarray(
            self.embeddings.embed_documents(self._context_windows(sentences)),
            dtype=np.float64,
        )
        norms = np.linalg.norm(vectors, axis=1)
        norms[norms == 0.0] = 1.0
        unit = vectors / norms[:, None]

        # distances[i] = độ "lệch chủ đề" giữa câu i và câu i+1
        distances = 1.0 - np.sum(unit[:-1] * unit[1:], axis=1)
        if distances.size == 0:
            return [text.strip()]

        threshold = float(np.percentile(distances, self.config.semantic_threshold_percentile))
        breakpoints = [i for i, d in enumerate(distances) if float(d) > threshold]

        chunks: list[str] = []
        start = 0
        for bp in breakpoints:
            chunks.append(" ".join(sentences[start : bp + 1]).strip())
            start = bp + 1
        if start < len(sentences):
            chunks.append(" ".join(sentences[start:]).strip())

        return [c for c in chunks if c]
