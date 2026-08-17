"""W1-10 — khử câu hỏi trùng ý.

Vì sao cần: model được đưa 300 chunk khác nhau vẫn sinh ra hàng chục câu dạng
"Tỷ lệ nghèo ở Việt Nam năm 2020 là bao nhiêu?". Chúng không sai, nhưng làm
**cùng một phép đo bị đếm nhiều lần** — nhóm nào model thích viết sẽ chi phối
con số tổng và bảng breakdown thành vô nghĩa.
"""

from __future__ import annotations

import pytest

from pipeline.eval.golden import GoldenQuery, QueryCategory
from pipeline.goldenset.dedupe import deduplicate_drafts, jaccard, tokenize
from pipeline.goldenset.schema import DraftProvenance, GoldenDraft, normalize_for_dedupe
from rag_core.embedding import HashingEmbeddingProvider
from rag_core.schemas import Language


def _draft(
    query: str,
    *,
    category: QueryCategory = QueryCategory.FACTOID,
    chunk_ids: list[str] | None = None,
    verified: bool = True,
    query_id: str | None = None,
) -> GoldenDraft:
    ids = chunk_ids if chunk_ids is not None else ["c1"]
    if category is QueryCategory.UNANSWERABLE:
        ids = []
    return GoldenDraft(
        query=GoldenQuery(
            query_id=query_id or f"q-{abs(hash(query)) % 10**8}",
            query=query,
            category=category,
            lang=Language.VI,
            relevant_chunk_ids=ids,
        ),
        provenance=DraftProvenance(
            generator_model="deepseek-chat",
            generator_model_requested="deepseek-chat",
            category_requested=category,
            quotes_verified=verified,
        ),
    )


class TestPrimitives:
    def test_normalize_strips_punctuation_and_case(self) -> None:
        assert normalize_for_dedupe("Tỷ lệ NGHÈO,  năm 2020?") == "tỷ lệ nghèo năm 2020"

    def test_jaccard_of_identical_sets_is_one(self) -> None:
        assert jaccard(tokenize("a b c"), tokenize("c b a")) == 1.0

    def test_jaccard_of_disjoint_sets_is_zero(self) -> None:
        assert jaccard(tokenize("a b"), tokenize("c d")) == 0.0

    def test_jaccard_with_empty_set(self) -> None:
        assert jaccard(frozenset(), tokenize("a")) == 0.0


class TestExactDuplicates:
    def test_removes_question_differing_only_in_punctuation(self) -> None:
        result = deduplicate_drafts(
            [
                _draft("Tỷ lệ nghèo của Việt Nam năm 2020 là bao nhiêu?"),
                _draft("tỷ lệ nghèo của việt nam năm 2020 là bao nhiêu"),
            ]
        )
        assert len(result.kept) == 1
        assert result.n_removed == 1

    def test_keeps_genuinely_different_questions(self) -> None:
        result = deduplicate_drafts(
            [
                _draft("Tỷ lệ nghèo của Việt Nam năm 2020 là bao nhiêu?"),
                _draft("Đầu tư công cho hạ tầng giao thông chiếm bao nhiêu phần trăm GDP?"),
            ]
        )
        assert len(result.kept) == 2


class TestNearDuplicates:
    def test_removes_reworded_question(self) -> None:
        result = deduplicate_drafts(
            [
                _draft("Tỷ lệ nghèo đa chiều của Việt Nam năm 2020 là bao nhiêu phần trăm?"),
                _draft("Năm 2020 tỷ lệ nghèo đa chiều của Việt Nam là bao nhiêu phần trăm?"),
            ],
            jaccard_threshold=0.8,
        )
        assert len(result.kept) == 1

    def test_threshold_one_keeps_everything_but_exact_copies(self) -> None:
        result = deduplicate_drafts(
            [
                _draft("Tỷ lệ nghèo đa chiều của Việt Nam năm 2020 là bao nhiêu phần trăm?"),
                _draft("Năm 2020 tỷ lệ nghèo đa chiều của Việt Nam là bao nhiêu phần trăm?"),
            ],
            jaccard_threshold=1.01,
        )
        assert len(result.kept) == 2

    def test_records_which_draft_absorbed_which(self) -> None:
        """Ghi lại cặp (bản bỏ, bản giữ) để kiểm lại quyết định thay vì tin mù."""
        keeper = _draft("Tỷ lệ nghèo đa chiều Việt Nam 2020 bao nhiêu phần trăm?", query_id="giu")
        result = deduplicate_drafts(
            [
                keeper,
                _draft("Tỷ lệ nghèo đa chiều Việt Nam 2020 bao nhiêu phần trăm", query_id="bo"),
            ]
        )
        assert result.removed[0][1] == "giu"


class TestCategoryIsolation:
    def test_same_wording_in_two_categories_both_survive(self) -> None:
        """Một câu factoid và một câu adversarial dùng chung phần lớn từ vựng vẫn
        là hai phép đo khác nhau — gộp lại là làm hỏng chính bảng breakdown."""
        text = "Tỷ lệ nghèo đa chiều của Việt Nam năm 2020 là bao nhiêu phần trăm?"
        result = deduplicate_drafts(
            [
                _draft(text, category=QueryCategory.FACTOID),
                _draft(text, category=QueryCategory.ADVERSARIAL),
            ]
        )
        assert len(result.kept) == 2


class TestRichnessWins:
    def test_keeps_the_draft_with_more_relevant_chunks(self) -> None:
        """Khi trùng, giữ bản mang nhiều thông tin hơn — không phải bản đến trước."""
        poor = _draft("Câu hỏi giống hệt nhau về ngân sách", chunk_ids=["c1"], query_id="ngheo")
        rich = _draft(
            "Câu hỏi giống hệt nhau về ngân sách", chunk_ids=["c1", "c2"], query_id="giau"
        )
        result = deduplicate_drafts([poor, rich])
        assert [d.query.query_id for d in result.kept] == ["giau"]

    def test_verified_quote_beats_unverified_when_chunks_tie(self) -> None:
        unverified = _draft("Câu hỏi trùng về đầu tư công", verified=False, query_id="chua")
        verified = _draft("Câu hỏi trùng về đầu tư công", verified=True, query_id="roi")
        result = deduplicate_drafts([unverified, verified])
        assert [d.query.query_id for d in result.kept] == ["roi"]


class TestSemanticPass:
    def test_optional_embedding_pass_runs_and_keeps_distinct_questions(self) -> None:
        """Lượt ngữ nghĩa là tuỳ chọn, và không được gộp hai câu khác chủ đề."""
        result = deduplicate_drafts(
            [
                _draft("Tỷ lệ nghèo của Việt Nam năm 2020 là bao nhiêu?"),
                _draft("Sản lượng cà phê xuất khẩu của Tây Nguyên năm 2019?"),
            ],
            embeddings=HashingEmbeddingProvider(dimension=256),
        )
        assert len(result.kept) == 2

    def test_semantic_pass_catches_lexically_different_paraphrase(self) -> None:
        embeddings = HashingEmbeddingProvider(dimension=256)
        text = "Tỷ lệ nghèo của Việt Nam năm 2020 là bao nhiêu?"
        result = deduplicate_drafts(
            [_draft(text, query_id="a"), _draft(text + " ", query_id="b")],
            jaccard_threshold=1.01,
            embeddings=embeddings,
            cosine_threshold=0.99,
        )
        assert len(result.kept) == 1


class TestEdgeCases:
    def test_empty_input(self) -> None:
        result = deduplicate_drafts([])
        assert result.kept == [] and result.n_removed == 0

    def test_single_draft_survives(self) -> None:
        assert len(deduplicate_drafts([_draft("Một câu duy nhất?")]).kept) == 1

    @pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
    def test_never_loses_a_draft_without_recording_it(self, threshold: float) -> None:
        """Bất biến: giữ + bỏ luôn bằng tổng đầu vào. Mất câu mà không ghi lại
        là mất dữ liệu đã trả tiền để sinh ra."""
        drafts = [
            _draft("Tỷ lệ nghèo Việt Nam 2020?"),
            _draft("Tỷ lệ nghèo Việt Nam 2020 là mấy?"),
            _draft("Đầu tư công chiếm bao nhiêu GDP?"),
        ]
        result = deduplicate_drafts(drafts, jaccard_threshold=threshold)
        assert len(result.kept) + result.n_removed == len(drafts)
