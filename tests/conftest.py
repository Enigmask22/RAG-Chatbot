"""Fixture dùng chung."""

from __future__ import annotations

import random

import pytest

from rag_core.embedding import HashingEmbeddingProvider
from rag_core.schemas import DocType, Document, DocumentMetadata, Language

_VI_WORDS = [
    "ngân",
    "sách",
    "đầu",
    "tư",
    "công",
    "tăng",
    "trưởng",
    "kinh",
    "tế",
    "báo",
    "cáo",
    "thường",
    "niên",
    "doanh",
    "thu",
    "lợi",
    "nhuận",
]


@pytest.fixture
def metadata() -> DocumentMetadata:
    return DocumentMetadata(
        source_url="https://openknowledge.worldbank.org/example",
        license="CC BY 3.0 IGO",
        title="Tài liệu thử nghiệm",
        lang=Language.VI,
        doc_type=DocType.DEV_REPORT,
    )


@pytest.fixture
def embeddings() -> HashingEmbeddingProvider:
    return HashingEmbeddingProvider(dimension=128)


@pytest.fixture
def short_document(metadata: DocumentMetadata) -> Document:
    text = " ".join(
        f"Đoạn {i} nói về chủ đề số {i // 4} trong báo cáo ngân sách." for i in range(20)
    )
    return Document(doc_id="doc-short", content=text, metadata=metadata)


@pytest.fixture
def large_document(metadata: DocumentMetadata) -> Document:
    """Xấp xỉ 50 trang (~100k ký tự) — đủ để lộ hành vi bậc hai nếu có."""
    rng = random.Random(1337)
    paragraphs = []
    for p in range(400):
        sentences = [
            " ".join(rng.choice(_VI_WORDS) for _ in range(rng.randint(8, 20))) + "."
            for _ in range(rng.randint(3, 6))
        ]
        paragraphs.append(f"Mục {p}. " + " ".join(sentences))
    return Document(doc_id="doc-large", content="\n\n".join(paragraphs), metadata=metadata)
