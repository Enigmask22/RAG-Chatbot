"""Interface chung và hậu xử lý dùng chung cho mọi chiến lược chunking.

Port từ `legacy/enhanced_chunking.py` của bản POC, với ba thay đổi có chủ ý — ghi lại ở
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
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..schemas import Chunk, Document
from .pieces import TextPiece, merge_pieces, shift
from .tokens import TokenCounter, TokenSizingUnavailable, calibrate_density, fit_to_budget

__all__ = ["Chunker", "ChunkingConfig", "ChunkingStrategy"]

logger = logging.getLogger(__name__)


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    STRUCTURE = "structure"
    PARENT_CHILD = "parent_child"


class ChunkingConfig(BaseModel):
    """Tham số chunking. Bất biến để `config_hash` luôn khớp với kết quả."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: ChunkingStrategy = ChunkingStrategy.HYBRID

    size_unit: Literal["chars", "tokens"] = "chars"
    """Đơn vị của `chunk_size`/`chunk_overlap`/`min_chunk_size`/`max_chunk_size`.

    Mặc định `chars` để mọi config đã công bố giữ nguyên kết quả. Đổi sang
    `tokens` là **đổi bộ chunk**, tức đổi index và mọi con số baseline — phải là
    lựa chọn có ý thức, không phải mặc định thừa hưởng (cùng lý lẽ với
    `neighbor_context_chars`).

    Ở chế độ `tokens`, chunker bắt buộc phải có bộ đếm token (`token_counter`,
    thường chính là `EmbeddingProvider`); thiếu thì **ném lỗi** chứ không lặng lẽ
    rơi về đếm ký tự — xem `tokens.TokenSizingUnavailable`. Ngân sách bên trong
    vẫn quy về ký tự bằng mật độ đo trên chính tài liệu đó, nhưng **trần token là
    bảo đảm cứng**: mảnh nào còn vượt sẽ bị cắt lại. Xem docstring `tokens.py`.
    """

    # --- fixed / recursive ---
    chunk_size: int = Field(default=1000, ge=1)
    chunk_overlap: int = Field(default=100, ge=0)
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

    # --- ràng buộc kích thước, áp cho mọi chiến lược ---
    min_chunk_size: int = Field(default=200, ge=0)
    max_chunk_size: int = Field(default=1500, ge=1)

    # --- semantic ---
    semantic_buffer_size: int = Field(default=1, ge=0)
    semantic_threshold_percentile: float = Field(default=85.0, ge=0.0, le=100.0)
    semantic_min_sentences: int = Field(default=3, ge=2)

    # --- hybrid ---
    hybrid_max_docs_for_semantic: int = Field(default=5, ge=0)

    # --- parent-child (W3-05) ---
    parent_size_multiple: int = Field(default=4, ge=2)
    """Parent lớn gấp bấy nhiêu lần child. `chunk_size` là kích thước **child**.

    Mặc định 4 ứng với cặp mà plan nêu: child 256 token → parent ~1024 token.
    Cận dưới 2 vì `parent_size_multiple=1` nghĩa là parent trùng child, tức tắt
    hẳn small-to-big — muốn tắt thì đổi `strategy`, đừng làm nó thành một cấu
    hình trông vẫn giống bật.
    """

    # --- structure (W3-03) ---
    structure_merge_short_sections: bool = True
    """Cho phép gộp section ngắn hơn `min_chunk_size` vào chunk liền trước.

    Bật thì `section_path` của chunk gộp **tụt xuống tổ tiên chung** của các
    section bị gộp, chứ không giữ đường dẫn của section đầu. Tắt thì mỗi section
    là ít nhất một chunk, kể cả section chỉ có một dòng.

    Đánh đổi: tắt cho `section_path` sâu nhất nhưng sinh ra rất nhiều chunk vụn
    ở văn bản pháp luật (mỗi khoản một chunk); bật cho chunk to đều hơn nhưng
    đường dẫn nông hơn. Là knob để ablation ở `W3-06` đo, không phải để đoán.
    """

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
        # Cận dưới 50 vốn là cận dưới tính bằng KÝ TỰ — `W3-06` mới lộ ra điều
        # đó, khi `chunk_size=48` **token** (một ngân sách hoàn toàn hợp lý với
        # BGE-M3) bị chặn bởi một ràng buộc chưa từng nói nó đo bằng gì.
        if self.size_unit == "chars" and (self.chunk_size < 50 or self.max_chunk_size < 50):
            raise ValueError("với size_unit='chars', chunk_size và max_chunk_size phải ≥ 50 ký tự")
        if self.size_unit == "tokens" and self.neighbor_context_chars > 0:
            raise ValueError(
                "size_unit='tokens' và neighbor_context_chars > 0 mâu thuẫn nhau: "
                "ngữ cảnh hàng xóm CỐ Ý vượt max_chunk_size (nó là phần đệm chép "
                "từ chunk bên cạnh), nên nó phá đúng cái trần cứng mà chế độ token "
                "dựng lên. Chọn một trong hai."
            )
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

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        *,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.config = config or ChunkingConfig()
        self._planned_documents: int | None = None
        self._token_counter = token_counter
        self._sizing = self.config
        self._warned_limit = False

    @property
    def sizing(self) -> ChunkingConfig:
        """Config **đã quy về ký tự** cho tài liệu đang chunk.

        Bằng chính `self.config` ở chế độ `chars`. Ở chế độ `tokens` nó là bản
        sao có `chunk_size`/`min`/`max` đã nhân với mật độ `ký tự/token` đo trên
        tài liệu đó — nên mọi chỗ trong chunker phải đọc `self.sizing`, không đọc
        `self.config`, nếu không sẽ trộn hai đơn vị.

        `self.config` vẫn là thứ khai báo, và vẫn là thứ đi vào `config_hash`.
        """
        return self._sizing

    @property
    def name(self) -> str:
        return f"{self.strategy.value}:{self.config.config_hash[:12]}"

    @abstractmethod
    def split_pieces(self, text: str) -> list[TextPiece]:
        """Cắt text thành các mảnh thô, mỗi mảnh kèm vùng xuất xứ.

        Đây là phương thức lớp con phải cài, thay cho `split_text` từ `W1-11`.
        Span cần thiết để golden set neo được vào văn bản gốc thay vì vào
        `chunk_id` thuần vị trí (`TD-12`).

        ⚠️ `piece.text` **không** buộc bằng `text[piece.start:piece.end]` — xem
        docstring của `pieces.py`. Ép nó thành substring nguyên văn sẽ đổi nội
        dung chunk, tức đổi cả index và mọi con số baseline.
        """

    def split_text(self, text: str) -> list[str]:
        """Chỉ phần text. Tiện ích dẫn xuất, không phải điểm mở rộng."""
        return [p.text for p in self.split_pieces(text)]

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
            pieces = self._apply_neighbor_context(self._prepare_pieces(doc))
            for index, piece in enumerate(pieces):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}::{index:05d}",
                        doc_id=doc.doc_id,
                        content=piece.text,
                        chunk_index=index,
                        section_path=self._section_path_for(doc, index),
                        metadata=doc.metadata,
                        start_char=piece.start,
                        end_char=piece.end,
                    )
                )
        return chunks

    # -------------------------------------------------------- điểm mở rộng

    def _prepare_pieces(self, doc: Document) -> list[TextPiece]:
        """Mảnh cuối cùng của **một** tài liệu, đã lọc rỗng và ép kích thước.

        Tách khỏi `chunk` để `W3-03` thay được bước này mà không phải chép lại
        phần dựng `Chunk`. `StructureChunker` ép kích thước **theo từng section**
        chứ không trên danh sách gộp — cùng lý do mà `_enforce_size` được áp theo
        từng tài liệu chứ không trên cả lô (xem điểm 2 ở docstring module): gộp
        một mảnh nhỏ vào mảnh liền trước qua ranh giới section thì chunk sinh ra
        mang `section_path` của section cũ mà nội dung thuộc section mới.
        """
        limit = self._begin_sizing(doc.content)
        try:
            pieces = [p for p in self.split_pieces(doc.content) if p.text.strip()]
            pieces = self._enforce_size(pieces)
            if limit is None:
                return pieces
            return [p for _, p in self._fit_tokens(pieces, limit)]
        finally:
            self._end_sizing()

    # ---------------------------------------------------- kích thước theo token

    def _begin_sizing(self, text: str) -> int | None:
        """Đặt `self.sizing` cho tài liệu này; trả trần token, `None` nếu đếm ký tự.

        Người gọi **phải** gọi `_end_sizing` trong `finally`: `self._sizing` là
        trạng thái theo từng tài liệu, để sót lại là tài liệu sau bị chunk bằng
        mật độ của tài liệu trước — sai âm thầm, đúng khuôn `TD-12`.
        """
        self._sizing = self.config
        if self.config.size_unit != "tokens":
            return None

        counter = self._token_counter
        if counter is None:
            raise TokenSizingUnavailable(
                "size_unit='tokens' nhưng chunker không có `token_counter`. "
                "Truyền `EmbeddingProvider` vào (`build_chunker(config, embeddings)`)."
            )

        density = calibrate_density(text, counter)
        model_limit = counter.max_sequence_tokens
        limit = self.config.max_chunk_size
        if model_limit is not None and model_limit < limit:
            # Cảnh báo một lần cho mỗi chunker: nó là tính chất của cấu hình, không
            # phải của tài liệu, nên in lại ở mỗi tài liệu chỉ làm ngập log.
            if not self._warned_limit:
                self._warned_limit = True
                logger.warning(
                    "max_chunk_size=%d token vượt cửa sổ của model (%d) — dùng %d làm trần",
                    limit,
                    model_limit,
                    model_limit,
                )
            limit = model_limit

        self._sizing = self.config.model_copy(
            update={
                "size_unit": "chars",
                "chunk_size": max(50, round(self.config.chunk_size * density)),
                "chunk_overlap": max(0, round(self.config.chunk_overlap * density)),
                "min_chunk_size": max(0, round(self.config.min_chunk_size * density)),
                "max_chunk_size": max(50, round(limit * density)),
            }
        )
        return limit

    def _end_sizing(self) -> None:
        self._sizing = self.config

    def _fit_tokens(self, pieces: list[TextPiece], limit: int) -> list[tuple[int, TextPiece]]:
        """Bảo đảm cứng: không mảnh nào vượt `limit` token."""
        counter = self._token_counter
        assert counter is not None  # `_begin_sizing` đã kiểm
        return fit_to_budget(
            pieces,
            limit=limit,
            counter=counter,
            separators=self.sizing.separators,
            chunk_overlap=self.sizing.chunk_overlap,
        )

    def _section_path_for(self, doc: Document, index: int) -> list[str]:
        """Đường dẫn heading của mảnh thứ `index`. Rỗng với chunker không nhìn cấu trúc.

        `Chunk.section_path` có mặt trong schema từ `W1-01` nhưng tới `W3-03` mới
        có chunker điền được. Mặc định rỗng — **không** phải rỗng vì thiếu sót mà
        vì fixed/semantic/hybrid cắt theo ký tự và câu, chúng không biết gì về
        heading để mà điền.

        Nhận **chỉ số** chứ không nhận `TextPiece`: hai mảnh trong cùng một tài
        liệu có thể trùng span (splitter đệ quy chồng lấn), nên tra theo span là
        tra nhầm. `_apply_neighbor_context` giữ nguyên số lượng và thứ tự, nên
        chỉ số khớp 1-1 với thứ tự `_prepare_pieces` trả về.
        """
        return []

    # ------------------------------------------------------------ hậu xử lý

    def _enforce_size(self, pieces: list[TextPiece]) -> list[TextPiece]:
        """Gộp đoạn quá nhỏ vào đoạn trước, cắt đoạn quá lớn bằng splitter cố định."""
        from .fixed import split_recursive_pieces  # import cục bộ để tránh vòng lặp

        out: list[TextPiece] = []
        for piece in pieces:
            if len(piece.text) > self.sizing.max_chunk_size:
                out.extend(
                    shift(
                        split_recursive_pieces(
                            piece.text,
                            separators=list(self.sizing.separators),
                            chunk_size=self.sizing.chunk_size,
                            chunk_overlap=self.sizing.chunk_overlap,
                        ),
                        piece.start,
                    )
                )
                continue

            if len(piece.text) < self.sizing.min_chunk_size and out:
                merged = merge_pieces([out[-1], piece], "\n")
                if len(merged.text) <= self.sizing.max_chunk_size:
                    out[-1] = merged
                    continue

            out.append(piece)
        return out

    def _apply_neighbor_context(self, pieces: list[TextPiece]) -> list[TextPiece]:
        """Nối đệm từ hai chunk lân cận vào `text`, **giữ nguyên span**.

        Span vẫn là vùng của riêng chunk, không gồm phần đệm — và đó là điều
        đúng: đệm là bản sao text của chunk khác, gán nó vào span của chunk này
        thì mỗi chunk sẽ "sở hữu" một vùng chồng lên hai chunk bên cạnh, và mọi
        phép ánh xạ nhãn theo span sẽ khớp thừa ba lần.
        """
        window = self.config.neighbor_context_chars
        if window == 0 or len(pieces) <= 1:
            return pieces

        out: list[TextPiece] = []
        for i, piece in enumerate(pieces):
            parts: list[str] = []
            if i > 0:
                parts.append(pieces[i - 1].text[-window:])
            parts.append(piece.text)
            if i < len(pieces) - 1:
                parts.append(pieces[i + 1].text[:window])
            out.append(TextPiece("\n".join(parts), piece.start, piece.end))
        return out
