"""W1-03 — ba chiến lược chunking sau một interface chung."""

from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from rag_core.chunking import (
    ChunkingConfig,
    ChunkingStrategy,
    FixedSizeChunker,
    HybridChunker,
    SemanticChunker,
    build_chunker,
    split_recursive,
    split_sentences,
)
from rag_core.embedding import HashingEmbeddingProvider
from rag_core.schemas import Document, DocumentMetadata


@pytest.fixture
def config() -> ChunkingConfig:
    return ChunkingConfig(chunk_size=300, chunk_overlap=50, min_chunk_size=80, max_chunk_size=600)


def _all_chunkers(
    config: ChunkingConfig, embeddings: HashingEmbeddingProvider
) -> list[tuple[str, object]]:
    return [
        ("fixed", FixedSizeChunker(config)),
        ("semantic", SemanticChunker(embeddings, config)),
        ("hybrid", HybridChunker(embeddings, config)),
    ]


class TestSharedInterface:
    """Điều kiện để ablation so sánh công bằng: mọi chiến lược cho ra cùng hình dạng output."""

    @pytest.mark.parametrize("name", ["fixed", "semantic", "hybrid"])
    def test_produces_valid_chunks(
        self,
        name: str,
        config: ChunkingConfig,
        embeddings: HashingEmbeddingProvider,
        short_document: Document,
    ) -> None:
        chunker = dict(_all_chunkers(config, embeddings))[name]
        chunks = chunker.chunk([short_document])  # type: ignore[attr-defined]

        assert chunks, "chunker không được trả danh sách rỗng cho tài liệu có nội dung"
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        assert len({c.chunk_id for c in chunks}) == len(chunks)
        assert all(c.doc_id == short_document.doc_id for c in chunks)
        assert all(c.content.strip() for c in chunks)
        assert all(c.metadata == short_document.metadata for c in chunks)

    def test_no_streamlit_import(self) -> None:
        # Bản POC gọi thẳng `st.warning` trong chunker, khiến không dùng lại được
        # ngoài UI. Test này canh việc đó không quay lại.
        import sys

        import rag_core.chunking as pkg  # noqa: F401

        assert "streamlit" not in sys.modules


class TestEdgeCases:
    @pytest.mark.parametrize("name", ["fixed", "semantic", "hybrid"])
    def test_single_sentence_document(
        self,
        name: str,
        config: ChunkingConfig,
        embeddings: HashingEmbeddingProvider,
        metadata: DocumentMetadata,
    ) -> None:
        doc = Document(doc_id="d", content="Một câu duy nhất.", metadata=metadata)
        chunker = dict(_all_chunkers(config, embeddings))[name]
        chunks = chunker.chunk([doc])  # type: ignore[attr-defined]
        assert len(chunks) == 1
        assert chunks[0].content == "Một câu duy nhất."

    @pytest.mark.parametrize("name", ["fixed", "semantic", "hybrid"])
    def test_whitespace_only_document(
        self,
        name: str,
        config: ChunkingConfig,
        embeddings: HashingEmbeddingProvider,
        metadata: DocumentMetadata,
    ) -> None:
        doc = Document(doc_id="d", content="   \n\n   \t  ", metadata=metadata)
        chunker = dict(_all_chunkers(config, embeddings))[name]
        assert chunker.chunk([doc]) == []  # type: ignore[attr-defined]

    def test_empty_document_list(
        self, config: ChunkingConfig, embeddings: HashingEmbeddingProvider
    ) -> None:
        assert HybridChunker(embeddings, config).chunk([]) == []

    def test_large_document_is_linear_enough(
        self, config: ChunkingConfig, large_document: Document
    ) -> None:
        """~50 trang phải xong trong vài giây — canh hành vi bậc hai."""
        started = time.perf_counter()
        chunks = FixedSizeChunker(config).chunk([large_document])
        elapsed = time.perf_counter() - started

        assert len(chunks) > 100
        assert elapsed < 5.0, f"chunking 50 trang mất {elapsed:.1f}s — nghi ngờ thuật toán bậc hai"
        assert all(len(c.content) <= config.max_chunk_size for c in chunks)


class TestFixedSplitter:
    def test_respects_chunk_size(self) -> None:
        text = "\n\n".join(f"Đoạn văn số {i} với một ít nội dung." for i in range(50))
        pieces = split_recursive(text, ["\n\n", "\n", " ", ""], chunk_size=200, chunk_overlap=20)
        assert all(len(p) <= 200 for p in pieces)
        assert "".join(p.replace(" ", "") for p in pieces).count("Đoạnvănsố0") >= 1

    def test_long_word_without_separator(self) -> None:
        # Không có chỗ nào để cắt "đẹp" — vẫn phải cắt được nhờ separator rỗng.
        text = "x" * 500
        pieces = split_recursive(text, ["\n\n", " ", ""], chunk_size=100, chunk_overlap=0)
        assert len(pieces) >= 5
        assert all(len(p) <= 100 for p in pieces)

    def test_no_matching_separator_keeps_oversized_piece(self) -> None:
        # Danh sách separator không chứa "" thì không cắt mịn hơn được nữa.
        pieces = split_recursive("y" * 300, ["\n\n"], chunk_size=100, chunk_overlap=0)
        assert pieces == ["y" * 300]

    def test_empty_text(self) -> None:
        assert split_recursive("", ["\n", ""], chunk_size=100, chunk_overlap=0) == []


class TestSemanticChunker:
    @staticmethod
    def _two_topic_doc(metadata: DocumentMetadata) -> Document:
        block_a = " ".join(["Ngân sách nhà nước tăng chi đầu tư công."] * 4)
        block_b = " ".join(["Đội bóng ghi bàn thắng ở phút cuối trận đấu."] * 4)
        return Document(doc_id="d", content=f"{block_a} {block_b}", metadata=metadata)

    @staticmethod
    def _config(buffer: int) -> ChunkingConfig:
        return ChunkingConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=2000,
            max_chunk_size=4000,
            min_chunk_size=0,
            semantic_buffer_size=buffer,
            semantic_threshold_percentile=80.0,
        )

    def test_splits_exactly_at_topic_change(self, metadata: DocumentMetadata) -> None:
        """Không có cửa sổ ngữ cảnh thì ranh giới phải sắc: không chunk nào lẫn hai chủ đề."""
        chunks = SemanticChunker(HashingEmbeddingProvider(256), self._config(buffer=0)).chunk(
            [self._two_topic_doc(metadata)]
        )

        assert len(chunks) >= 2
        for chunk in chunks:
            assert not ("Ngân sách" in chunk.content and "bàn thắng" in chunk.content)

    def test_context_buffer_blurs_boundary_by_one_sentence(
        self, metadata: DocumentMetadata
    ) -> None:
        """Với `semantic_buffer_size=1`, ranh giới lệch được ±1 câu — và đó là
        đánh đổi cố ý, không phải lỗi.

        Cửa sổ ngữ cảnh làm mượt tín hiệu (câu ngắn kiểu "Điều 5." tự nó gần như
        không mang nội dung), cái giá là điểm cắt nhoè đi đúng bằng bán kính cửa
        sổ. Test này ghim hành vi đó lại để lần sau ai thấy chunk biên bị lẫn chủ
        đề thì biết là đã biết, không phải hồi quy mới.
        """
        chunks = SemanticChunker(HashingEmbeddingProvider(256), self._config(buffer=1)).chunk(
            [self._two_topic_doc(metadata)]
        )

        assert len(chunks) >= 2
        mixed = [c for c in chunks if "Ngân sách" in c.content and "bàn thắng" in c.content]
        assert len(mixed) <= 1, "chỉ đúng chunk ở biên được phép lẫn hai chủ đề"

    def test_too_few_sentences_returns_whole_text(
        self,
        config: ChunkingConfig,
        embeddings: HashingEmbeddingProvider,
        metadata: DocumentMetadata,
    ) -> None:
        doc = Document(doc_id="d", content="Câu một. Câu hai.", metadata=metadata)
        chunks = SemanticChunker(embeddings, config).chunk([doc])
        assert len(chunks) == 1

    def test_split_sentences_handles_vietnamese(self) -> None:
        assert split_sentences("Câu một. Câu hai! Câu ba?") == ["Câu một.", "Câu hai!", "Câu ba?"]

    def test_split_sentences_ignores_blank(self) -> None:
        assert split_sentences("  \n\n  ") == []


class TestHybridChunker:
    def test_uses_semantic_for_small_batch(
        self, embeddings: HashingEmbeddingProvider, short_document: Document
    ) -> None:
        chunker = HybridChunker(embeddings, ChunkingConfig(hybrid_max_docs_for_semantic=5))
        chunker.chunk([short_document])
        assert chunker.last_strategy_used == "semantic"

    def test_falls_back_to_fixed_for_large_batch(
        self, embeddings: HashingEmbeddingProvider, short_document: Document
    ) -> None:
        chunker = HybridChunker(embeddings, ChunkingConfig(hybrid_max_docs_for_semantic=2))
        chunker.chunk([short_document] * 3)
        assert chunker.last_strategy_used == "fixed"

    def test_falls_back_when_semantic_raises(
        self, short_document: Document, caplog: pytest.LogCaptureFixture
    ) -> None:
        class BrokenEmbeddings(HashingEmbeddingProvider):
            def embed_documents(self, texts):  # type: ignore[no-untyped-def]
                raise RuntimeError("CUDA out of memory")

        chunker = HybridChunker(BrokenEmbeddings(64), ChunkingConfig())
        chunks = chunker.chunk([short_document])

        assert chunks, "fallback phải vẫn cho ra chunk"
        assert chunker.last_strategy_used == "fixed_fallback"
        # Không được nuốt lỗi: job index chạy 2h rồi âm thầm đổi chiến lược là
        # kịch bản tệ nhất cho eval.
        assert "Semantic chunking thất bại" in caplog.text

    def test_works_without_embeddings(self, short_document: Document) -> None:
        chunker = HybridChunker(None, ChunkingConfig())
        assert chunker.chunk([short_document])
        assert chunker.last_strategy_used == "fixed"


class TestNeighborContext:
    def test_disabled_by_default(self, short_document: Document) -> None:
        assert ChunkingConfig().neighbor_context_chars == 0

    def test_adds_context_from_both_sides(self, metadata: DocumentMetadata) -> None:
        config = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED,
            chunk_size=100,
            chunk_overlap=0,
            min_chunk_size=0,
            max_chunk_size=200,
            neighbor_context_chars=20,
        )
        text = "\n\n".join(f"{'ABCDEFGHIJ' * 8}-{i}" for i in range(3))
        chunks = FixedSizeChunker(config).chunk(
            [Document(doc_id="d", content=text, metadata=metadata)]
        )

        assert len(chunks) >= 3
        # Chunk giữa dài hơn chunk gốc vì được đệm cả hai phía.
        assert len(chunks[1].content) > len(chunks[1].content.split("\n")[1])


class TestConfigHash:
    def test_differs_for_nearby_values(self) -> None:
        """Bản POC làm tròn chunk_size về bội 100 nên 1000 và 1049 dùng chung
        cache — đủ để một vòng ablation báo hai cấu hình cho kết quả y hệt."""
        a = ChunkingConfig(chunk_size=1000)
        b = ChunkingConfig(chunk_size=1049)
        assert a.config_hash != b.config_hash

    def test_stable_for_same_config(self) -> None:
        assert (
            ChunkingConfig(chunk_size=512).config_hash == ChunkingConfig(chunk_size=512).config_hash
        )

    def test_rejects_overlap_ge_chunk_size(self) -> None:
        with pytest.raises(ValidationError, match="lặp vô hạn"):
            ChunkingConfig(chunk_size=200, chunk_overlap=200)

    def test_rejects_min_ge_max(self) -> None:
        with pytest.raises(ValidationError):
            ChunkingConfig(min_chunk_size=1500, max_chunk_size=1500)


class TestFactory:
    def test_semantic_requires_embeddings(self) -> None:
        with pytest.raises(ValueError, match="EmbeddingProvider"):
            build_chunker(ChunkingConfig(strategy=ChunkingStrategy.SEMANTIC))

    @pytest.mark.parametrize(
        ("strategy", "expected"),
        [
            (ChunkingStrategy.FIXED, FixedSizeChunker),
            (ChunkingStrategy.SEMANTIC, SemanticChunker),
            (ChunkingStrategy.HYBRID, HybridChunker),
        ],
    )
    def test_builds_right_type(
        self, strategy: ChunkingStrategy, expected: type, embeddings: HashingEmbeddingProvider
    ) -> None:
        assert isinstance(build_chunker(ChunkingConfig(strategy=strategy), embeddings), expected)
