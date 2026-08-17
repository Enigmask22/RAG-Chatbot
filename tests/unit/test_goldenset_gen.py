"""W1-10 — parse response, xác minh trích dẫn, dựng bản nháp.

Trọng tâm: những gì code phải bắt được **trước** khi tới tay người review. Mỗi
câu lọt qua với `chunk_id` sai là một câu làm sai recall mà không có triệu chứng.
"""

from __future__ import annotations

import json

import pytest

from pipeline.eval.golden import QueryCategory
from pipeline.goldenset.generate import (
    ParsedQuestion,
    _build_draft,
    _resolve_category,
    parse_response,
    query_id_for,
    quote_found,
)
from pipeline.goldenset.sampling import ChunkGroup
from rag_core.schemas import Chunk, DocType, DocumentMetadata, Language

_TEXT_A = (
    "Tỷ lệ nghèo đa chiều của Việt Nam giảm từ 9,2 phần trăm năm 2016 xuống "
    "còn 4,4 phần trăm vào năm 2020, theo số liệu của Tổng cục Thống kê."
)
_TEXT_B = (
    "Đầu tư công cho hạ tầng giao thông chiếm khoảng 5,7 phần trăm GDP trong "
    "giai đoạn 2016-2020, cao hơn mức trung bình của khu vực Đông Á."
)


def _chunk(index: int, text: str, lang: Language = Language.VI) -> Chunk:
    return Chunk(
        chunk_id=f"doc-1::{index:05d}",
        doc_id="doc-1",
        content=text,
        chunk_index=index,
        metadata=DocumentMetadata(
            source_url="https://example.org/doc-1",
            license="CC BY 4.0",
            lang=lang,
            doc_type=DocType.DEV_REPORT,
        ),
    )


def _group(category: QueryCategory, *texts: str) -> ChunkGroup:
    return ChunkGroup(
        category=category, chunks=[_chunk(i, t) for i, t in enumerate(texts, start=10)]
    )


def _payload(**item: object) -> str:
    base: dict[str, object] = {
        "query": "Tỷ lệ nghèo đa chiều của Việt Nam năm 2020 là bao nhiêu?",
        "lang": "vi",
        "used_chunks": [1],
        "quote": "giảm từ 9,2 phần trăm năm 2016 xuống còn 4,4 phần trăm",
        "reference_answer": "4,4 phần trăm",
    }
    base.update(item)
    return json.dumps({"questions": [base]}, ensure_ascii=False)


class TestParseResponse:
    def test_reads_a_well_formed_answer(self) -> None:
        questions, bad_index, bad_schema = parse_response(_payload(), n_chunks=1)
        assert (bad_index, bad_schema) == (0, 0)
        assert questions[0].used_chunks == [1]
        assert questions[0].lang is Language.VI

    def test_strips_markdown_fences(self) -> None:
        """Model bị dặn đừng rào ```json vẫn rào — đó là lỗi hình thức, không
        phải lý do vứt cả lời gọi đã trả tiền."""
        fenced = f"```json\n{_payload()}\n```"
        questions, _, bad_schema = parse_response(fenced, n_chunks=1)
        assert len(questions) == 1
        assert bad_schema == 0

    def test_accepts_bare_list(self) -> None:
        raw = json.dumps([{"query": "Câu hỏi?", "used_chunks": [1], "lang": "vi"}])
        questions, _, _ = parse_response(raw, n_chunks=1)
        assert len(questions) == 1

    def test_accepts_used_chunks_as_single_int(self) -> None:
        questions, _, _ = parse_response(_payload(used_chunks=1), n_chunks=1)
        assert questions[0].used_chunks == [1]

    def test_dedupes_repeated_indices(self) -> None:
        questions, _, _ = parse_response(_payload(used_chunks=[1, 1, 2]), n_chunks=2)
        assert questions[0].used_chunks == [1, 2]

    @pytest.mark.parametrize("indices", [[0], [3], [-1], [1, 9]])
    def test_rejects_indices_outside_the_range_given(self, indices: list[int]) -> None:
        """Chỉ số ngoài khoảng là thứ DUY NHẤT không sửa được ở bước review —
        không biết model định trỏ vào đâu, nên phải bỏ câu."""
        questions, bad_index, _ = parse_response(_payload(used_chunks=indices), n_chunks=2)
        assert questions == []
        assert bad_index == 1

    def test_rejects_non_numeric_index(self) -> None:
        questions, bad_index, _ = parse_response(_payload(used_chunks=["một"]), n_chunks=2)
        assert (questions, bad_index) == ([], 1)

    def test_counts_broken_json_as_schema_failure(self) -> None:
        assert parse_response("không phải json", n_chunks=1) == ([], 0, 1)

    def test_counts_missing_questions_array(self) -> None:
        assert parse_response(json.dumps({"ket_qua": []}), n_chunks=1) == ([], 0, 1)

    def test_skips_item_without_query(self) -> None:
        raw = json.dumps({"questions": [{"used_chunks": [1]}, {"query": "  ", "used_chunks": [1]}]})
        questions, _, bad_schema = parse_response(raw, n_chunks=1)
        assert (questions, bad_schema) == ([], 2)

    def test_unknown_language_falls_back_instead_of_crashing(self) -> None:
        questions, _, _ = parse_response(_payload(lang="klingon"), n_chunks=1)
        assert questions[0].lang is Language.UNKNOWN

    def test_empty_used_chunks_is_allowed_for_unanswerable(self) -> None:
        questions, bad_index, _ = parse_response(_payload(used_chunks=[]), n_chunks=1)
        assert questions[0].used_chunks == []
        assert bad_index == 0


class TestQuoteVerification:
    def test_finds_exact_quote(self) -> None:
        assert quote_found("giảm từ 9,2 phần trăm năm 2016", [_chunk(1, _TEXT_A)])

    def test_ignores_whitespace_differences(self) -> None:
        """Chunk giữ nguyên xuống dòng của PDF, model chép lại thành một dòng.
        Đó là khác biệt hình thức, không phải dấu hiệu bịa đặt."""
        chunk = _chunk(1, "Tỷ lệ nghèo\n   đa chiều   giảm mạnh trong giai đoạn này.")
        assert quote_found("Tỷ lệ nghèo đa chiều giảm mạnh", [chunk])

    def test_case_insensitive(self) -> None:
        assert quote_found("TỶ LỆ NGHÈO ĐA CHIỀU CỦA VIỆT NAM", [_chunk(1, _TEXT_A)])

    def test_rejects_invented_quote(self) -> None:
        assert not quote_found("Con số này tăng gấp đôi sau đó", [_chunk(1, _TEXT_A)])

    def test_rejects_quote_too_short_to_prove_anything(self) -> None:
        """Trích dẫn 5 ký tự khớp được với gần như mọi văn bản — nó không chứng
        minh điều gì, nên coi như chưa kiểm chứng."""
        assert not quote_found("Việt", [_chunk(1, _TEXT_A)])

    def test_searches_across_all_cited_chunks(self) -> None:
        chunks = [_chunk(1, _TEXT_A), _chunk(2, _TEXT_B)]
        assert quote_found("chiếm khoảng 5,7 phần trăm GDP", chunks)


class TestQueryId:
    def test_same_question_gives_same_id(self) -> None:
        """Chạy lại sinh cùng câu thì cùng ID, nên hai lần chạy gộp được mà
        không nhân đôi."""
        a = query_id_for(QueryCategory.FACTOID, "Tỷ lệ nghèo năm 2020?")
        b = query_id_for(QueryCategory.FACTOID, "  tỷ lệ nghèo năm 2020?  ")
        assert a == b

    def test_different_category_gives_different_id(self) -> None:
        assert query_id_for(QueryCategory.FACTOID, "x?") != query_id_for(
            QueryCategory.ADVERSARIAL, "x?"
        )

    def test_id_carries_the_category_name(self) -> None:
        assert query_id_for(QueryCategory.MULTI_HOP, "x?").startswith("multi_hop-")


class TestCategoryResolution:
    @pytest.mark.parametrize("requested", [QueryCategory.MULTI_HOP, QueryCategory.AGGREGATION])
    def test_downgrades_to_factoid_when_only_one_chunk_used(self, requested: QueryCategory) -> None:
        """Câu `multi_hop` chỉ dựa vào một chunk thì nó là `factoid`.

        Giữ nguyên nhãn sai làm cột `multi_hop` trong bảng breakdown báo một
        năng lực mà hệ thống chưa từng được đo.
        """
        assert _resolve_category(requested, 1) is QueryCategory.FACTOID

    def test_keeps_multi_hop_when_two_chunks_used(self) -> None:
        assert _resolve_category(QueryCategory.MULTI_HOP, 2) is QueryCategory.MULTI_HOP

    def test_unanswerable_never_downgrades(self) -> None:
        assert _resolve_category(QueryCategory.UNANSWERABLE, 0) is QueryCategory.UNANSWERABLE

    def test_factoid_stays_factoid(self) -> None:
        assert _resolve_category(QueryCategory.FACTOID, 1) is QueryCategory.FACTOID


def _draft(group: ChunkGroup, parsed: ParsedQuestion) -> object:
    return _build_draft(
        parsed,
        group,
        response_model="deepseek-chat",
        requested_model="deepseek-chat",
        cost_usd=0.001,
        prompt_tokens=100,
        completion_tokens=20,
        batch_id="test",
    )


class TestBuildDraft:
    def test_maps_indices_to_real_chunk_ids(self) -> None:
        """Model chỉ trả về chỉ số; việc ánh xạ sang `chunk_id` do code làm.

        Cho model tự viết id là mở đường cho cả golden set trỏ vào hư không.
        """
        group = _group(QueryCategory.FACTOID, _TEXT_A)
        draft = _draft(
            group,
            ParsedQuestion(
                query="Tỷ lệ nghèo đa chiều Việt Nam năm 2020?",
                lang=Language.VI,
                used_chunks=[1],
                quote="giảm từ 9,2 phần trăm năm 2016",
                reference_answer="4,4%",
            ),
        )
        assert draft is not None
        assert draft.query.relevant_chunk_ids == ["doc-1::00010"]  # type: ignore[attr-defined]
        assert draft.provenance.quotes_verified  # type: ignore[attr-defined]

    def test_marks_unverified_quote_for_close_review(self) -> None:
        group = _group(QueryCategory.FACTOID, _TEXT_A)
        draft = _draft(
            group,
            ParsedQuestion(
                query="Tỷ lệ nghèo đa chiều Việt Nam năm 2020?",
                lang=Language.VI,
                used_chunks=[1],
                quote="một câu model tự nghĩ ra hoàn toàn",
                reference_answer="4,4%",
            ),
        )
        assert draft is not None
        assert not draft.provenance.quotes_verified  # type: ignore[attr-defined]
        assert draft.needs_close_review  # type: ignore[attr-defined]

    def test_unanswerable_gets_no_relevant_chunks(self) -> None:
        """Dù model có trả về chỉ số chunk, câu unanswerable vẫn phải rỗng —
        `GoldenQuery` từ chối nếu không."""
        group = _group(QueryCategory.UNANSWERABLE, _TEXT_A)
        draft = _draft(
            group,
            ParsedQuestion(
                query="GDP Việt Nam năm 2045 dự báo là bao nhiêu?",
                lang=Language.VI,
                used_chunks=[1],
                quote="",
                reference_answer="Không có trong corpus",
            ),
        )
        assert draft is not None
        assert draft.query.relevant_chunk_ids == []  # type: ignore[attr-defined]
        assert draft.query.category is QueryCategory.UNANSWERABLE  # type: ignore[attr-defined]

    def test_drops_answerable_question_with_no_chunk(self) -> None:
        group = _group(QueryCategory.FACTOID, _TEXT_A)
        assert (
            _draft(
                group,
                ParsedQuestion(
                    query="Câu hỏi không viện dẫn đoạn nào?",
                    lang=Language.VI,
                    used_chunks=[],
                    quote="",
                    reference_answer="",
                ),
            )
            is None
        )

    def test_records_category_drift(self) -> None:
        group = _group(QueryCategory.MULTI_HOP, _TEXT_A, _TEXT_B)
        draft = _draft(
            group,
            ParsedQuestion(
                query="Tỷ lệ nghèo đa chiều Việt Nam năm 2020?",
                lang=Language.VI,
                used_chunks=[1],
                quote="giảm từ 9,2 phần trăm năm 2016",
                reference_answer="4,4%",
            ),
        )
        assert draft is not None
        assert draft.query.category is QueryCategory.FACTOID  # type: ignore[attr-defined]
        assert draft.provenance.category_requested is QueryCategory.MULTI_HOP  # type: ignore[attr-defined]
        assert draft.category_drifted  # type: ignore[attr-defined]

    def test_keeps_every_source_chunk_even_the_unused_ones(self) -> None:
        """Biết model được đưa gì mà không dùng là thông tin để chỉnh prompt."""
        group = _group(QueryCategory.MULTI_HOP, _TEXT_A, _TEXT_B)
        draft = _draft(
            group,
            ParsedQuestion(
                query="Ghép hai thông tin: tỷ lệ nghèo và đầu tư công giai đoạn 2016-2020?",
                lang=Language.VI,
                used_chunks=[1, 2],
                quote="chiếm khoảng 5,7 phần trăm GDP",
                reference_answer="...",
            ),
        )
        assert draft is not None
        assert len(draft.provenance.source_chunk_ids) == 2  # type: ignore[attr-defined]
