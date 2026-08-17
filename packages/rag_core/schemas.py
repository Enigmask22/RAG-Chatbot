"""Hợp đồng dữ liệu dùng chung cho cả hai plane.

Đây là *thứ duy nhất* mà pipeline và serving thống nhất với nhau về hình dạng dữ
liệu. Vì vậy mọi model ở đây đều:

* `extra="forbid"` — payload thừa field là lỗi, không im lặng bỏ qua. Một field
  gõ sai tên mà bị nuốt sẽ thành bug âm thầm trong index đã build xong.
* `frozen=True` với `Document`/`Chunk` — sau khi tạo thì bất biến, để hash nội
  dung luôn khớp với nội dung.
* round-trip được: `model_validate_json(x.model_dump_json()) == x`.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "Answer",
    "Chunk",
    "Citation",
    "DocType",
    "Document",
    "DocumentMetadata",
    "Language",
    "QueryRequest",
    "RetrievalMode",
    "RetrievedChunk",
    "TokenUsage",
]

NonEmptyStr = Annotated[str, Field(min_length=1)]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def sha256_of(text: str) -> str:
    """Hash nội dung đã chuẩn hoá — dùng cho cache key và dedupe.

    Chuẩn hoá xuống dòng và bỏ khoảng trắng thừa ở hai đầu mỗi dòng để cùng một
    tài liệu tải lại từ nguồn khác (CRLF vs LF) không sinh hash khác nhau.
    """
    normalized = "\n".join(line.strip() for line in text.replace("\r\n", "\n").split("\n"))
    return hashlib.sha256(normalized.strip().encode("utf-8")).hexdigest()


class Language(StrEnum):
    """Ngôn ngữ chính của tài liệu/chunk/truy vấn."""

    VI = "vi"
    EN = "en"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class DocType(StrEnum):
    """Loại tài liệu — dùng để breakdown metric và để lọc metadata.

    Ba giá trị đầu ứng với ba nguồn corpus đã chốt: mỗi nguồn phục vụ một nhóm
    metric khác nhau, nên breakdown theo `doc_type` là đầu ra chính của eval.
    """

    DEV_REPORT = "dev_report"
    """Báo cáo tổ chức phát triển (World Bank / ADB) — prose dài, song ngữ."""

    LEGAL = "legal"
    """Văn bản pháp luật — heading nhiều cấp, kiểm chứng `section_path`."""

    ANNUAL_REPORT = "annual_report"
    """Báo cáo thường niên doanh nghiệp — nhiều bảng, kiểm chứng `table_lookup`."""

    OTHER = "other"


class RetrievalMode(StrEnum):
    """Nhánh truy hồi đã sinh ra kết quả — cần cho việc phân tích ablation."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    RERANKED = "reranked"


class DocumentMetadata(BaseModel):
    """Xuất xứ của một tài liệu.

    `source_url` và `license` là **bắt buộc**: corpus phải công khai và cho phép
    redistribute, vì repo public + demo public + máy GPU thuê đều là kênh công
    bố dữ liệu. Bắt buộc ở tầng schema thì không thể quên khi thêm tài liệu mới.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_url: NonEmptyStr
    license: NonEmptyStr
    source_path: str | None = None
    title: str | None = None
    lang: Language = Language.UNKNOWN
    doc_type: DocType = DocType.OTHER
    published_at: datetime | None = None
    ingested_at: datetime = Field(default_factory=_utcnow)
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """Một tài liệu nguồn sau khi đã trích xuất text, trước khi chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: NonEmptyStr
    content: NonEmptyStr
    metadata: DocumentMetadata

    @property
    def content_hash(self) -> str:
        return sha256_of(self.content)


class Chunk(BaseModel):
    """Một đơn vị được embed và index.

    `section_path` là đường dẫn heading (ví dụ
    `["Chương II", "Điều 15", "Khoản 2"]`). Nó có mặt ngay từ schema nền dù
    structure-aware chunker mãi tới W3 mới sinh ra được — chunker cũ để rỗng.
    Đặt trước như vậy để lên W3 không phải migrate lại toàn bộ index.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: NonEmptyStr
    doc_id: NonEmptyStr
    content: NonEmptyStr
    chunk_index: int = Field(ge=0)
    section_path: list[str] = Field(default_factory=list)
    parent_chunk_id: str | None = None
    token_count: int | None = Field(default=None, ge=0)
    metadata: DocumentMetadata | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return sha256_of(self.content)

    @property
    def section_header(self) -> str:
        """Chuỗi heading để prepend vào text lúc embed."""
        return " > ".join(self.section_path)


class RetrievedChunk(BaseModel):
    """Một chunk kèm điểm số của lần truy hồi cụ thể.

    Giữ `dense_score`/`sparse_score`/`rerank_score` tách riêng thay vì chỉ một
    `score` tổng: khi phân tích ablation cần biết nhánh nào đã kéo chunk lên,
    và RRF làm mất thông tin đó nếu không lưu lại.
    """

    model_config = ConfigDict(extra="forbid")

    chunk: Chunk
    score: float
    rank: int = Field(ge=1)
    mode: RetrievalMode = RetrievalMode.DENSE
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None


class Citation(BaseModel):
    """Một trích dẫn trong câu trả lời, kèm kết quả xác minh.

    `verified=False` nghĩa là `quote` **không** tìm thấy trong chunk được cite —
    tức mô hình đã bịa. Câu trả lời vẫn trả về nhưng phải đánh dấu, không được
    im lặng bỏ qua.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: NonEmptyStr
    doc_id: NonEmptyStr
    quote: NonEmptyStr
    verified: bool = False
    source_url: str | None = None
    section_path: list[str] = Field(default_factory=list)


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Answer(BaseModel):
    """Đầu ra của serving plane.

    `model` là **model thực tế đã phục vụ request**, đọc từ response của
    provider — không phải model mình yêu cầu. Router có fallback, và một metric
    dịch chuyển vì âm thầm rơi sang model khác là loại bug rất khó truy.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    model: NonEmptyStr
    prompt_version: str | None = None
    bundle_version: str | None = None
    usage: TokenUsage | None = None
    latency_ms: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_refusal_consistency(self) -> Answer:
        if self.refused and self.citations:
            raise ValueError("Câu trả lời đã từ chối thì không được kèm citation")
        if not self.refused and not self.text.strip():
            raise ValueError("Câu trả lời không từ chối thì `text` không được rỗng")
        return self


class QueryRequest(BaseModel):
    """Đầu vào của serving plane."""

    model_config = ConfigDict(extra="forbid")

    query: NonEmptyStr
    top_k: int = Field(default=10, ge=1, le=200)
    conversation_id: str | None = None
    tenant_id: str | None = None
    lang: Language | None = None
    doc_types: list[DocType] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    stream: bool = True

    @field_validator("query")
    @classmethod
    def _strip_query(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("query không được chỉ gồm khoảng trắng")
        return stripped
