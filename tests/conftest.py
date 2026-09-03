"""Fixture dùng chung."""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator

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


@pytest.fixture(autouse=True)
def _restore_root_logging() -> Iterator[None]:
    """⭐ Trả lại cấu hình logging gốc sau **mỗi** test.

    `create_app()` gọi `configure_logging()`, và hàm đó **gỡ sạch handler của
    root logger** — đúng như nó phải làm ở production, nơi uvicorn đã tự
    `dictConfig` trước đó. Nhưng một trong những handler bị gỡ là handler mà
    `caplog` của pytest vừa gắn vào, và nó không được gắn lại; nó cũng hạ mức
    root xuống `CRITICAL` khi test dựng app với `log_level="CRITICAL"`.

    Hệ quả: **mọi** test dùng `caplog` chạy *sau* một test dựng app đều thấy
    `caplog.text` rỗng.

    ⚠️ Lỗi này có từ `W4-03` và không ai thấy, vì `pytest-randomly` đổi thứ tự
    file mỗi lần chạy: nó chỉ đỏ khi `tests/integration/test_health.py` rơi
    trước `tests/unit/test_chunking.py`. Con số "1613 passed" của những phiên
    trước vì thế đúng cho *thứ tự hôm đó* chứ không đúng cho bộ test.

    Cách chữa nằm ở đây chứ không phải trong `create_app`: gỡ handler là hành vi
    **đúng** của production, và thêm một cờ `configure_logs=False` vào mã
    production để chiều test là để một khác biệt test-vs-prod đi vào mã.
    """
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    try:
        yield
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)
