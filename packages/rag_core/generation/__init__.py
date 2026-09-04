"""Tầng sinh câu trả lời: hợp đồng văn bản giữa prompt và mã đọc lại nó. `W4-09`/`W4-11`."""

from .citations import (
    MARKER,
    CitationClaim,
    CitationHoldback,
    CitationReport,
    ParsedAnswer,
    split_citation_block,
    verify_citations,
)
from .prompts import (
    Prompt,
    PromptIntegrityError,
    PromptNotFoundError,
    PromptRegistry,
    default_registry,
    sha256_of,
    stamp,
)

__all__ = [
    "MARKER",
    "CitationClaim",
    "CitationHoldback",
    "CitationReport",
    "ParsedAnswer",
    "Prompt",
    "PromptIntegrityError",
    "PromptNotFoundError",
    "PromptRegistry",
    "default_registry",
    "sha256_of",
    "split_citation_block",
    "stamp",
    "verify_citations",
]
