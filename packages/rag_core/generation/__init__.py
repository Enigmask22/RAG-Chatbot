"""Tầng sinh câu trả lời: hợp đồng văn bản giữa prompt và mã đọc lại nó. `W4-09`."""

from .citations import (
    MARKER,
    CitationClaim,
    CitationHoldback,
    CitationReport,
    ParsedAnswer,
    split_citation_block,
    verify_citations,
)

__all__ = [
    "MARKER",
    "CitationClaim",
    "CitationHoldback",
    "CitationReport",
    "ParsedAnswer",
    "split_citation_block",
    "verify_citations",
]
