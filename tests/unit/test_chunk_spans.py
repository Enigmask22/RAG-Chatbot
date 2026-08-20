"""Test cho span ký tự của chunk (`W1-11`, `TD-12`).

Ba nhóm bất biến, xếp theo mức quan trọng:

1. **Nội dung chunk không đổi.** Refactor sang `split_pieces` không được sửa một
   byte nào của `content`, nếu không thì mọi con số baseline đã đo trước đó vô
   nghĩa. Nhóm `TestTextUnchanged` canh bằng cách so `split_text` với đúng biểu
   thức của bản cũ.
2. **Span nằm trong biên và theo thứ tự.** Span sai làm nhãn golden set trỏ lệch,
   và đó là kiểu lỗi không có triệu chứng.
3. **Span là vùng xuất xứ, không phải chỉ dẫn cắt.** Có test khẳng định tường minh
   rằng `content` có thể khác `doc[start:end]` — để không ai "sửa" nó thành bằng
   nhau rồi vô tình đổi nội dung chunk.
"""

from __future__ import annotations

import re

import pytest

from rag_core.chunking.base import ChunkingConfig, ChunkingStrategy
from rag_core.chunking.fixed import (
    FixedSizeChunker,
    split_recursive,
    split_recursive_pieces,
)
from rag_core.chunking.pieces import TextPiece, merge_pieces, shift
from rag_core.chunking.semantic import split_sentence_pieces, split_sentences
from rag_core.schemas import Chunk, Document, DocumentMetadata, Language, TextSpan

_SEPS = ["\n\n", "\n", ". ", " ", ""]
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？…])\s+|\n{2,}")


def _doc(content: str, doc_id: str = "d") -> Document:
    return Document(
        doc_id=doc_id,
        content=content,
        metadata=DocumentMetadata(
            source_url="https://example.org/x",
            license="CC BY 4.0",
            lang=Language.VI,
        ),
    )


_TEXTS = [
    "Câu một. Câu hai. Câu ba.",
    "Đoạn một.\n\nĐoạn hai.\n\n\nĐoạn ba với nhiều chữ hơn để vượt ngưỡng.",
    "khongcoseparatornaodaymotchuoirataidaidedebuocxuongkytu" * 5,
    "A\n\nB",
    "   khoảng trắng hai đầu   ",
    "Dòng 1\nDòng 2\nDòng 3\n" * 20,
    "Mixed. Text\n\nwith\nvarious   spacing. And numbers 1,2,3.",
]


class TestTextUnchanged:
    """Bất biến số 1 — nội dung không được đổi một byte."""

    @pytest.mark.parametrize("text", _TEXTS)
    def test_split_recursive_matches_pieces(self, text: str) -> None:
        pieces = split_recursive_pieces(text, _SEPS, chunk_size=50, chunk_overlap=10)
        assert split_recursive(text, _SEPS, 50, 10) == [p.text for p in pieces]

    @pytest.mark.parametrize("text", _TEXTS)
    def test_split_sentences_matches_old_expression(self, text: str) -> None:
        """`finditer` phải cho đúng danh sách mà `re.split` cho."""
        expected = [s.strip() for s in _SENTENCE_RE.split(text) if s and s.strip()]
        assert split_sentences(text) == expected

    @pytest.mark.parametrize("text", _TEXTS)
    def test_chunker_split_text_matches_pieces(self, text: str) -> None:
        chunker = FixedSizeChunker(ChunkingConfig(chunk_size=60, chunk_overlap=10))
        assert chunker.split_text(text) == [p.text for p in chunker.split_pieces(text)]

    def test_empty_separator_list_returns_whole_text(self) -> None:
        assert split_recursive_pieces("abc", [], 10, 0) == [TextPiece("abc", 0, 3)]

    def test_empty_text(self) -> None:
        assert split_recursive_pieces("", _SEPS, 10, 0) == []


class TestSpanBounds:
    """Bất biến số 2 — span trong biên, tăng dần."""

    @pytest.mark.parametrize("text", _TEXTS)
    def test_spans_within_document(self, text: str) -> None:
        for p in split_recursive_pieces(text, _SEPS, chunk_size=50, chunk_overlap=10):
            assert 0 <= p.start < p.end <= len(text), f"{p.start},{p.end} ngoài [0,{len(text)}]"

    @pytest.mark.parametrize("text", _TEXTS)
    def test_starts_are_non_decreasing(self, text: str) -> None:
        starts = [p.start for p in split_recursive_pieces(text, _SEPS, 50, 10)]
        assert starts == sorted(starts)

    @pytest.mark.parametrize("text", _TEXTS)
    def test_sentence_spans_within_document(self, text: str) -> None:
        for p in split_sentence_pieces(text):
            assert 0 <= p.start < p.end <= len(text)

    def test_sentence_span_points_at_the_sentence(self) -> None:
        """Với text mà splitter không nối lại gì, span phải trỏ đúng chữ."""
        text = "Câu một. Câu hai. Câu ba."
        for p in split_sentence_pieces(text):
            assert text[p.start : p.end] == p.text

    def test_offsets_survive_consecutive_separators(self) -> None:
        """`"\\n\\n\\n"` làm `split` sinh mảnh rỗng bị bỏ — con trỏ vẫn phải nhảy qua."""
        text = "AAA\n\n\nBBB"
        pieces = split_recursive_pieces(text, ["\n"], chunk_size=3, chunk_overlap=0)
        assert [p.text for p in pieces] == ["AAA", "BBB"]
        assert text[pieces[-1].start : pieces[-1].end] == "BBB"

    def test_recursion_offsets_are_absolute(self) -> None:
        """Mảnh quá khổ được đệ quy xuống separator mịn hơn: span phải là tuyệt đối."""
        head = "x" * 10 + "\n"
        text = head + "aaa bbb ccc ddd eee fff"
        pieces = split_recursive_pieces(text, ["\n", " ", ""], chunk_size=8, chunk_overlap=0)
        for p in pieces:
            assert p.start >= 0 and p.end <= len(text)
        tail = [p for p in pieces if p.start >= len(head)]
        assert tail, "phải có mảnh nằm sau phần đầu"


class TestSpanIsProvenanceNotSlice:
    """Bất biến số 3 — nói tường minh rằng span KHÔNG phải chỉ dẫn cắt."""

    def test_dropped_empty_split_makes_text_shorter_than_span(self) -> None:
        """Đây là ví dụ chuẩn: `"A\\n\\nB"` tách theo `"\\n"` cho `"A\\nB"`."""
        text = "A\n\nB"
        pieces = split_recursive_pieces(text, ["\n"], chunk_size=100, chunk_overlap=0)
        assert len(pieces) == 1
        p = pieces[0]
        assert p.text == "A\nB"
        assert text[p.start : p.end] == "A\n\nB"
        assert len(p.text) < p.end - p.start

    def test_semantic_join_replaces_original_separator(self) -> None:
        pieces = split_sentence_pieces("Một.\n\nHai.")
        joined = merge_pieces(pieces, " ")
        assert joined.text == "Một. Hai."
        assert joined.start == 0
        assert joined.end == len("Một.\n\nHai.")


class TestNeighborContextKeepsCoreSpan:
    def test_span_excludes_the_padding(self) -> None:
        """Nếu span gồm cả đệm thì mỗi chunk 'sở hữu' vùng của hai chunk bên cạnh."""
        text = "aaaa. " * 40
        base = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED,
            chunk_size=60,
            chunk_overlap=0,
            min_chunk_size=0,
        )
        plain = FixedSizeChunker(base).chunk([_doc(text)])
        padded = FixedSizeChunker(base.model_copy(update={"neighbor_context_chars": 20})).chunk(
            [_doc(text)]
        )

        assert len(plain) == len(padded)
        for a, b in zip(plain, padded, strict=True):
            assert (a.start_char, a.end_char) == (b.start_char, b.end_char)
            assert len(b.content) > len(a.content), "đệm phải làm text dài ra"


class TestEnforceSizeSpans:
    def test_merge_unions_the_spans(self) -> None:
        """Gộp mảnh nhỏ vào mảnh trước: span mới phải phủ cả hai."""
        cfg = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED,
            chunk_size=60,
            chunk_overlap=0,
            min_chunk_size=50,
            max_chunk_size=400,
        )
        text = "\n".join(["dòng ngắn"] * 12)
        chunks = FixedSizeChunker(cfg).chunk([_doc(text)])
        assert chunks
        for c in chunks:
            assert c.start_char is not None and c.end_char is not None
            assert c.end_char - c.start_char >= len(c.content) - 2

    def test_oversized_piece_keeps_absolute_offsets(self) -> None:
        cfg = ChunkingConfig(
            strategy=ChunkingStrategy.FIXED,
            chunk_size=100,
            chunk_overlap=0,
            min_chunk_size=0,
            max_chunk_size=120,
        )
        prefix = "mở đầu ngắn.\n\n"
        text = prefix + ("từ " * 200)
        chunks = FixedSizeChunker(cfg).chunk([_doc(text)])
        assert len(chunks) > 2
        for c in chunks:
            assert c.end_char is not None and c.end_char <= len(text)
        assert any(c.start_char is not None and c.start_char > len(prefix) for c in chunks)


class TestChunkSpanProperty:
    def test_span_built_from_offsets(self) -> None:
        c = Chunk(chunk_id="d::0", doc_id="d", content="x", chunk_index=0, start_char=3, end_char=9)
        assert c.span == TextSpan(doc_id="d", start=3, end=9)

    def test_span_is_none_without_offsets(self) -> None:
        assert Chunk(chunk_id="d::0", doc_id="d", content="x", chunk_index=0).span is None

    def test_span_is_none_if_only_one_offset(self) -> None:
        """Nửa vời còn tệ hơn không có: phải là `None`, không phải đoán nốt nửa kia."""
        c = Chunk(chunk_id="d::0", doc_id="d", content="x", chunk_index=0, start_char=3)
        assert c.span is None

    def test_chunker_populates_offsets(self) -> None:
        chunks = FixedSizeChunker(ChunkingConfig(chunk_size=60, chunk_overlap=10)).chunk(
            [_doc("abc. " * 40)]
        )
        assert chunks
        assert all(c.start_char is not None and c.end_char is not None for c in chunks)


class TestTextSpan:
    def test_overlap_partial(self) -> None:
        assert (
            TextSpan(doc_id="d", start=0, end=10).overlap(TextSpan(doc_id="d", start=5, end=20))
            == 5
        )

    def test_overlap_contained(self) -> None:
        assert (
            TextSpan(doc_id="d", start=0, end=100).overlap(TextSpan(doc_id="d", start=40, end=50))
            == 10
        )

    def test_no_overlap_when_disjoint(self) -> None:
        assert (
            TextSpan(doc_id="d", start=0, end=10).overlap(TextSpan(doc_id="d", start=10, end=20))
            == 0
        )

    def test_no_overlap_across_documents(self) -> None:
        """Không kiểm `doc_id` thì hai tài liệu khác nhau sẽ khớp nhãn của nhau."""
        assert (
            TextSpan(doc_id="a", start=0, end=100).overlap(TextSpan(doc_id="b", start=0, end=100))
            == 0
        )

    def test_overlap_is_symmetric(self) -> None:
        a = TextSpan(doc_id="d", start=3, end=17)
        b = TextSpan(doc_id="d", start=10, end=25)
        assert a.overlap(b) == b.overlap(a)

    def test_length(self) -> None:
        assert TextSpan(doc_id="d", start=4, end=11).length == 7

    def test_rejects_empty_span(self) -> None:
        with pytest.raises(Exception, match=r"span rỗng|greater than"):
            TextSpan(doc_id="d", start=5, end=5)

    def test_rejects_reversed_span(self) -> None:
        with pytest.raises(Exception, match=r"đảo ngược|span rỗng"):
            TextSpan(doc_id="d", start=9, end=4)

    def test_is_frozen(self) -> None:
        span = TextSpan(doc_id="d", start=0, end=5)
        with pytest.raises(Exception, match="frozen"):
            span.start = 1


class TestPieceHelpers:
    def test_shift_moves_both_ends(self) -> None:
        assert shift([TextPiece("a", 1, 2)], 10) == [TextPiece("a", 11, 12)]

    def test_shift_zero_returns_same_list(self) -> None:
        pieces = [TextPiece("a", 1, 2)]
        assert shift(pieces, 0) is pieces

    def test_merge_joins_text_and_unions_span(self) -> None:
        merged = merge_pieces([TextPiece("a", 0, 1), TextPiece("b", 5, 6)], "-")
        assert merged == TextPiece("a-b", 0, 6)

    def test_merge_uses_min_start_and_max_end(self) -> None:
        """Không giả định danh sách đã sắp thứ tự."""
        merged = merge_pieces([TextPiece("b", 5, 6), TextPiece("a", 0, 1)], "-")
        assert (merged.start, merged.end) == (0, 6)

    def test_merge_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="rỗng"):
            merge_pieces([], "-")
