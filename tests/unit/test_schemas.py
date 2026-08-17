"""W1-02 — hợp đồng dữ liệu: round-trip và từ chối payload sai."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_core.schemas import (
    Answer,
    Chunk,
    Citation,
    DocType,
    Document,
    DocumentMetadata,
    Language,
    QueryRequest,
    RetrievalMode,
    RetrievedChunk,
    TokenUsage,
    sha256_of,
)


@pytest.fixture
def meta() -> DocumentMetadata:
    return DocumentMetadata(
        source_url="https://example.org/doc",
        license="CC BY 4.0",
        lang=Language.VI,
        doc_type=DocType.LEGAL,
    )


@pytest.fixture
def chunk(meta: DocumentMetadata) -> Chunk:
    return Chunk(
        chunk_id="doc-1::00003",
        doc_id="doc-1",
        content="Nội dung chunk thử nghiệm.",
        chunk_index=3,
        section_path=["Chương II", "Điều 15"],
        metadata=meta,
    )


class TestRoundTrip:
    def test_document_round_trip(self, meta: DocumentMetadata) -> None:
        doc = Document(doc_id="doc-1", content="Xin chào thế giới.", metadata=meta)
        assert Document.model_validate_json(doc.model_dump_json()) == doc

    def test_chunk_round_trip(self, chunk: Chunk) -> None:
        assert Chunk.model_validate_json(chunk.model_dump_json()) == chunk

    def test_retrieved_chunk_round_trip(self, chunk: Chunk) -> None:
        rc = RetrievedChunk(
            chunk=chunk, score=0.87, rank=1, mode=RetrievalMode.HYBRID, dense_score=0.9
        )
        assert RetrievedChunk.model_validate_json(rc.model_dump_json()) == rc

    def test_answer_round_trip(self, chunk: Chunk) -> None:
        answer = Answer(
            text="Câu trả lời.",
            citations=[
                Citation(
                    chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, quote="thử nghiệm", verified=True
                )
            ],
            model="deepseek-chat",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=20, cost_usd=0.0001),
        )
        assert Answer.model_validate_json(answer.model_dump_json()) == answer

    def test_query_request_round_trip(self) -> None:
        req = QueryRequest(query="Ngân sách 2024 là bao nhiêu?", top_k=5, doc_types=[DocType.LEGAL])
        assert QueryRequest.model_validate_json(req.model_dump_json()) == req


class TestRejectsBadPayload:
    def test_extra_field_rejected(self, meta: DocumentMetadata) -> None:
        # Field gõ sai tên bị nuốt im lặng là nguồn bug âm thầm trong index.
        with pytest.raises(ValidationError, match="extra_forbidden"):
            Document.model_validate(
                {"doc_id": "d", "content": "x", "metadata": meta.model_dump(), "titel": "typo"}
            )

    def test_metadata_requires_source_and_license(self) -> None:
        with pytest.raises(ValidationError):
            DocumentMetadata(source_url="https://example.org")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            DocumentMetadata(license="CC BY 4.0")  # type: ignore[call-arg]

    def test_empty_content_rejected(self, meta: DocumentMetadata) -> None:
        with pytest.raises(ValidationError):
            Document(doc_id="d", content="", metadata=meta)

    def test_negative_chunk_index_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Chunk(chunk_id="c", doc_id="d", content="x", chunk_index=-1)

    def test_rank_starts_at_one(self, chunk: Chunk) -> None:
        with pytest.raises(ValidationError):
            RetrievedChunk(chunk=chunk, score=0.5, rank=0)

    def test_whitespace_query_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(query="   ")

    def test_query_is_stripped(self) -> None:
        assert QueryRequest(query="  ngân sách  ").query == "ngân sách"

    def test_refused_answer_cannot_have_citations(self, chunk: Chunk) -> None:
        with pytest.raises(ValidationError, match="từ chối"):
            Answer(
                text="Tôi không tìm thấy thông tin.",
                refused=True,
                citations=[Citation(chunk_id=chunk.chunk_id, doc_id="doc-1", quote="x")],
                model="deepseek-chat",
            )

    def test_non_refused_answer_cannot_be_empty(self) -> None:
        with pytest.raises(ValidationError, match="rỗng"):
            Answer(text="   ", model="deepseek-chat")

    def test_document_is_immutable(self, meta: DocumentMetadata) -> None:
        doc = Document(doc_id="d", content="x", metadata=meta)
        with pytest.raises(ValidationError):
            doc.content = "y"


class TestContentHash:
    def test_stable_across_line_endings(self, meta: DocumentMetadata) -> None:
        # Cùng một tài liệu tải lại từ nguồn khác không được sinh hash khác.
        crlf = Document(doc_id="d", content="dòng một\r\ndòng hai", metadata=meta)
        lf = Document(doc_id="d", content="dòng một\ndòng hai", metadata=meta)
        assert crlf.content_hash == lf.content_hash

    def test_differs_on_real_change(self, meta: DocumentMetadata) -> None:
        a = Document(doc_id="d", content="ngân sách 2024", metadata=meta)
        b = Document(doc_id="d", content="ngân sách 2025", metadata=meta)
        assert a.content_hash != b.content_hash

    def test_sha256_of_is_deterministic(self) -> None:
        assert sha256_of("xin chào") == sha256_of("  xin chào  ")


def test_section_header_joins_path(chunk: Chunk) -> None:
    assert chunk.section_header == "Chương II > Điều 15"


def test_token_usage_total() -> None:
    assert TokenUsage(prompt_tokens=10, completion_tokens=5).total_tokens == 15
