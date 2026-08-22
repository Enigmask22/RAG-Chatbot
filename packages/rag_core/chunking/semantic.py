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
from .pieces import TextPiece, merge_pieces

__all__ = ["SemanticChunker", "split_sentence_pieces", "split_sentences"]

# Cắt sau dấu kết câu (kể cả dấu toàn rộng) hoặc ở dòng trống.
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？…])\s+|\n{2,}")


def split_sentence_pieces(text: str) -> list[TextPiece]:
    """Tách câu kèm vùng xuất xứ.

    Tương đương `[s.strip() for s in _SENTENCE_RE.split(text) if s and s.strip()]`.
    Dùng `finditer` thay vì `split` để biết vị trí: `re.split` với pattern không
    có nhóm bắt trả về đúng các đoạn giữa hai match, nên hai cách cho cùng danh
    sách text — có test canh điều đó.
    """
    out: list[TextPiece] = []
    pos = 0
    for match in _SENTENCE_RE.finditer(text):
        _append_sentence(out, text[pos : match.start()], pos)
        pos = match.end()
    _append_sentence(out, text[pos:], pos)
    return out


def _append_sentence(out: list[TextPiece], segment: str, offset: int) -> None:
    """Thêm một câu đã strip, thu span vào đúng phần khoảng trắng bị cắt."""
    stripped = segment.strip()
    if not stripped:
        return
    lead = len(segment) - len(segment.lstrip())
    trail = len(segment) - len(segment.rstrip())
    out.append(TextPiece(stripped, offset + lead, offset + len(segment) - trail))


def split_sentences(text: str) -> list[str]:
    """Chỉ phần text. Giữ lại vì phần lớn chỗ gọi không cần span."""
    return [p.text for p in split_sentence_pieces(text)]


class SemanticChunker(Chunker):
    strategy: ClassVar[ChunkingStrategy] = ChunkingStrategy.SEMANTIC

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        config: ChunkingConfig | None = None,
    ) -> None:
        # `EmbeddingProvider` thoả sẵn giao thức `TokenCounter`, nên chunker nào
        # có model embedding thì đo được kích thước theo token (`W3-06`).
        super().__init__(config, token_counter=embeddings)
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

    def split_pieces(self, text: str) -> list[TextPiece]:
        pieces = split_sentence_pieces(text)
        if len(pieces) < self.config.semantic_min_sentences:
            stripped = text.strip()
            if not stripped:
                return []
            lead = len(text) - len(text.lstrip())
            trail = len(text) - len(text.rstrip())
            return [TextPiece(stripped, lead, len(text) - trail)]

        sentences = [p.text for p in pieces]
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
            return [TextPiece(text.strip(), pieces[0].start, pieces[-1].end)]

        threshold = float(np.percentile(distances, self.config.semantic_threshold_percentile))
        breakpoints = [i for i, d in enumerate(distances) if float(d) > threshold]

        # Câu đã strip rồi nối bằng dấu cách, nên `.strip()` ngoài là no-op —
        # giữ lại để text khớp từng byte với bản trước `W1-11`.
        groups: list[TextPiece] = []
        start = 0
        for bp in breakpoints:
            groups.append(_join_sentences(pieces[start : bp + 1]))
            start = bp + 1
        if start < len(pieces):
            groups.append(_join_sentences(pieces[start:]))

        return [g for g in groups if g.text]


def _join_sentences(group: list[TextPiece]) -> TextPiece:
    joined = merge_pieces(group, " ")
    return TextPiece(joined.text.strip(), joined.start, joined.end)
