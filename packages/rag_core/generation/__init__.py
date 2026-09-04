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
from .guardrails import (
    INJECTION_RULES,
    RedactingFilter,
    context_nonce,
    normalise_for_scan,
    redact_pii,
    scan_injection,
    wrap_context,
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
    "INJECTION_RULES",
    "MARKER",
    "CitationClaim",
    "CitationHoldback",
    "CitationReport",
    "ParsedAnswer",
    "Prompt",
    "PromptIntegrityError",
    "PromptNotFoundError",
    "PromptRegistry",
    "RedactingFilter",
    "context_nonce",
    "default_registry",
    "normalise_for_scan",
    "redact_pii",
    "scan_injection",
    "sha256_of",
    "split_citation_block",
    "stamp",
    "verify_citations",
    "wrap_context",
]
