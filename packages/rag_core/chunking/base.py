"""Interface chung và hậu xử lý dùng chung cho mọi chiến lược chunking.

Port từ `enhanced_chunking.py` của bản POC, với ba thay đổi có chủ ý — ghi lại ở
đây vì chúng ảnh hưởng tới con số baseline:

1. **`config_hash` không làm tròn tham số nữa.** Bản cũ làm tròn `chunk_size` về
   bội 100 và `overlap` về bội 25 "cho cache ổn định hơn". Hậu quả là
   `chunk_size=1000` và `chunk_size=1049` dùng chung cache entry — nghĩa là một
   vòng ablation quét chunk_size sẽ đọc lại kết quả của cấu hình khác và báo hai
   cấu hình cho ra số y hệt. Đúng thứ làm hỏng toàn bộ eval. Giờ hash trên
   nguyên văn config.
2. **Hậu xử lý áp theo từng tài liệu**, không áp trên danh sách gộp. Bản cũ gộp
   chunk quá nhỏ vào chunk liền trước kể cả khi chunk đó thuộc tài liệu khác.
3. **Thứ tự hậu xử lý:** ép kích thước trước, thêm ngữ cảnh hàng xóm sau (bản cũ
   làm ngược). Ngữ cảnh hàng xóm cố ý vượt `max_chunk_size` vì đó là phần đệm;
   cắt nhỏ nó lần nữa chỉ tạo ra các chunk trùng lặp nội dung.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..schemas import Chunk, Document

__all__ = ["Chunker", "ChunkingConfig", "ChunkingStrategy"]


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class ChunkingConfig(BaseModel):
    """Tham số chunking. Bất biến để `config_hash` luôn khớp với kết quả."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: ChunkingStrategy = ChunkingStrategy.HYBRID

    # --- fixed / recursive ---
    chunk_size: int = Field(default=1000, ge=50)
    chunk_overlap: int = Field(default=100, ge=0)
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

    # --- ràng buộc kích thước, áp cho mọi chiến lược ---
    min_chunk_size: int = Field(default=200, ge=0)
    max_chunk_size: int = Field(default=1500, ge=50)

    # --- semantic ---
    semantic_buffer_size: int = Field(default=1, ge=0)
    semantic_threshold_percentile: float = Field(default=85.0, ge=0.0, le=100.0)
    semantic_min_sentences: int = Field(default=3, ge=2)

    # --- hybrid ---
    hybrid_max_docs_for_semantic: int = Field(default=5, ge=0)

    # --- ngữ cảnh hàng xóm (hành vi của bản POC) ---
    neighbor_context_chars: int = Field(default=0, ge=0)
    """Nối thêm bấy nhiêu ký tự từ chunk trước và chunk sau vào mỗi chunk.

    Mặc định **tắt**. Bản POC luôn bật ở mức 100 ký tự, nên config baseline của
    `W1-13` phải đặt lại thành 100 để tái lập đúng hệ thống hiện tại. Để mặc
    định tắt vì kỹ thuật này thổi phồng số ký tự được embed lên ~20% và làm nhiễu
    metric precision — muốn dùng thì phải là lựa chọn có ý thức, không phải mặc
    định thừa hưởng.
    """

    @model_validator(mode="after")
    def _check_sizes(self) -> ChunkingConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap phải nhỏ hơn chunk_size, nếu không sẽ lặp vô hạn")
        if self.max_chunk_size < self.chunk_size:
            raise ValueError("max_chunk_size không được nhỏ hơn chunk_size")
        if self.min_chunk_size >= self.max_chunk_size:
            raise ValueError("min_chunk_size phải nhỏ hơn max_chunk_size")
        return self

    @property
    def config_hash(self) -> str:
        """Hash toàn bộ config — dùng làm một nửa của cache key."""
        payload = self.model_dump_json()
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Chunker(ABC):
    """Lớp cha của mọi chiến lược chunking.

    Lớp con chỉ cần cài `split_text`. Việc dựng `Chunk`, ép kích thước, thêm ngữ
    cảnh hàng xóm và đánh `chunk_id` do lớp cha lo — để mọi chiến lược cho ra
    output cùng hình dạng và ablation so sánh được đúng phần thuật toán khác nhau.
    """

    strategy: ClassVar[ChunkingStrategy]

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()
        self._planned_documents: int | None = None

    @property
    def name(self) -> str:
        return f"{self.strategy.value}:{self.config.config_hash[:12]}"

    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """Cắt một tài liệu thành các đoạn text thô, chưa hậu xử lý."""

    @property
    def planned_documents(self) -> int | None:
        """Số tài liệu của cả lô, nếu người gọi đã khai báo qua `prepare`."""
        return self._planned_documents

    def prepare(self, n_documents: int) -> None:
        """Khai báo kích thước lô **thật** sắp chunk.

        Cần có vì cache hoạt động **theo từng tài liệu**: `CachedChunker` gọi
        `chunk([doc])` 60 lần cho một corpus 60 tài liệu. Chunker nào ra quyết
        định dựa trên số tài liệu trong lô (`HybridChunker`) sẽ thấy `n=1` mỗi
        lần và chọn sai nhánh — im lặng, không lỗi, chỉ là số baseline sai.

        Script build index gọi hàm này một lần với tổng số tài liệu rồi mới lặp
        từng tài liệu; khai báo tường minh luôn thắng suy đoán theo lô.
        """
        self._planned_documents = n_documents

    def _batch_size_for_decision(self, n_in_call: int) -> int:
        return self._planned_documents if self._planned_documents is not None else n_in_call

    def chunk(self, documents: Sequence[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            pieces = [p for p in self.split_text(doc.content) if p.strip()]
            pieces = self._enforce_size(pieces)
            pieces = self._apply_neighbor_context(pieces)
            for index, piece in enumerate(pieces):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}::{index:05d}",
                        doc_id=doc.doc_id,
                        content=piece,
                        chunk_index=index,
                        metadata=doc.metadata,
                    )
                )
        return chunks

    # ------------------------------------------------------------ hậu xử lý

    def _enforce_size(self, pieces: list[str]) -> list[str]:
        """Gộp đoạn quá nhỏ vào đoạn trước, cắt đoạn quá lớn bằng splitter cố định."""
        from .fixed import split_recursive  # import cục bộ để tránh vòng lặp import

        out: list[str] = []
        for piece in pieces:
            if len(piece) > self.config.max_chunk_size:
                out.extend(
                    split_recursive(
                        piece,
                        separators=list(self.config.separators),
                        chunk_size=self.config.chunk_size,
                        chunk_overlap=self.config.chunk_overlap,
                    )
                )
                continue

            if len(piece) < self.config.min_chunk_size and out:
                merged = out[-1] + "\n" + piece
                if len(merged) <= self.config.max_chunk_size:
                    out[-1] = merged
                    continue

            out.append(piece)
        return out

    def _apply_neighbor_context(self, pieces: list[str]) -> list[str]:
        window = self.config.neighbor_context_chars
        if window == 0 or len(pieces) <= 1:
            return pieces

        out: list[str] = []
        for i, piece in enumerate(pieces):
            parts: list[str] = []
            if i > 0:
                parts.append(pieces[i - 1][-window:])
            parts.append(piece)
            if i < len(pieces) - 1:
                parts.append(pieces[i + 1][:window])
            out.append("\n".join(parts))
        return out
