"""Test cho `RerankedRetriever`, `CrossEncoderReranker` và `build_branch(..., "reranked")` — `W2-05`.

Không tải model ở đây: `Reranker` là một ABC có đúng một method, nên một bản giả
xác định cho phép test toàn bộ logic xếp hạng, tie-break, trần `top_n` và việc
chuyển tiếp — tức là chỗ bug thật sẽ nằm. Phần chạm cross-encoder thật ở
`tests/integration/test_reranked_retriever.py` (đánh dấu `gpu`).

Hai test của DoD (thứ tự score giảm dần, batch == single) có ở cả hai tầng: ở đây
với bản giả để chúng luôn chạy trong `make test`, và ở tầng integration với model
thật để hợp đồng được kiểm trên thứ thật sự được phục vụ.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from rag_core.embedding import HashingEmbeddingProvider
from rag_core.reranking import (
    BGE_RERANKER_V2_M3,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_MAX_LENGTH,
    CrossEncoderReranker,
    Reranker,
)
from rag_core.retrieval import (
    DEFAULT_RERANK_CANDIDATES,
    QdrantDenseRetriever,
    RerankedRetriever,
    Retriever,
    build_branch,
)
from rag_core.schemas import (
    Chunk,
    DocType,
    DocumentMetadata,
    Language,
    RetrievalMode,
    RetrievedChunk,
)


def _chunk(key: str, content: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=key,
        doc_id=key.split("::")[0],
        content=content or f"nội dung của {key}",
        chunk_index=0,
        metadata=DocumentMetadata(
            source_url="https://example.org/x",
            license="CC BY 4.0",
            lang=Language.VI,
            doc_type=DocType.DEV_REPORT,
        ),
    )


class FakeBase(Retriever):
    """Nhánh nền giả: trả về đúng danh sách đã cho, cắt theo `top_k`.

    Ghi lại `top_k` đã được yêu cầu vì đó là chỗ `candidates` biến thành độ sâu
    thật của pool, và cũng là chỗ tương tác với `candidate_k` của hybrid xảy ra.
    """

    def __init__(self, keys: Sequence[str], *, name: str = "fake-base") -> None:
        self.name = name
        self.keys = list(keys)
        self.requested: list[int] = []
        self.filters_seen: list[dict[str, Any] | None] = []

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        self.requested.append(top_k)
        self.filters_seen.append(filters)
        return [
            RetrievedChunk(
                chunk=_chunk(key),
                score=1.0 - 0.01 * index,
                rank=index + 1,
                mode=RetrievalMode.HYBRID,
                dense_score=0.5 - 0.01 * index,
                sparse_score=9.0 - index,
            )
            for index, key in enumerate(self.keys[:top_k])
        ]

    def fetch_doc_chunks(self, doc_ids: Sequence[str], *, batch: int = 512) -> list[Chunk]:
        self.fetch_batch = batch
        return [_chunk(f"{doc_id}::0") for doc_id in doc_ids]

    def verify_schema(self) -> None:
        self.verified = True


class ScriptedReranker(Reranker):
    """Reranker giả: điểm tra từ một dict theo nội dung chunk."""

    def __init__(self, table: dict[str, float], *, default: float = 0.0) -> None:
        self.name = "scripted"
        self.table = table
        self.default = default
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls.append((query, tuple(texts)))
        return [self.table.get(text, self.default) for text in texts]


def _by_key(table: dict[str, float]) -> ScriptedReranker:
    """Bảng điểm theo `chunk_id` thay vì theo nội dung, gọn hơn khi viết test."""
    return ScriptedReranker({f"nội dung của {key}": value for key, value in table.items()})


# --------------------------------------------------------------------------- #
# DoD: thứ tự score giảm dần
# --------------------------------------------------------------------------- #


class TestOrdering:
    def test_scores_come_back_descending(self) -> None:
        base = FakeBase(["a", "b", "c", "d"])
        reranker = _by_key({"a": 0.1, "b": 0.9, "c": 0.4, "d": 0.7})
        results = RerankedRetriever(base, reranker).retrieve("q", 4)

        assert [hit.chunk.chunk_id for hit in results] == ["b", "d", "c", "a"]
        assert [hit.score for hit in results] == [0.9, 0.7, 0.4, 0.1]

    def test_rank_is_one_based_and_contiguous(self) -> None:
        """Hợp đồng của `Retriever`, và MRR/nDCG phụ thuộc trực tiếp vào nó."""
        base = FakeBase(["a", "b", "c"])
        results = RerankedRetriever(base, _by_key({"a": 0.2, "b": 0.3, "c": 0.1})).retrieve("q", 3)
        assert [hit.rank for hit in results] == [1, 2, 3]

    def test_negative_scores_order_correctly(self) -> None:
        """Logit thô là số âm rất thường xuyên — sắp xếp phải không giả định dấu."""
        base = FakeBase(["a", "b", "c"])
        reranker = _by_key({"a": -8.5, "b": -0.3, "c": -11.0})
        results = RerankedRetriever(base, reranker).retrieve("q", 3)
        assert [hit.chunk.chunk_id for hit in results] == ["b", "a", "c"]

    def test_reranker_can_completely_reverse_the_base(self) -> None:
        """Đây là lý do tầng này tồn tại: nó được phép bác bỏ nhánh nền hoàn toàn."""
        base = FakeBase(["a", "b", "c", "d", "e"])
        reranker = _by_key({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0})
        results = RerankedRetriever(base, reranker).retrieve("q", 5)
        assert [hit.chunk.chunk_id for hit in results] == ["e", "d", "c", "b", "a"]

    def test_a_deep_candidate_can_reach_rank_one(self) -> None:
        """Cụ thể hoá cái `W2-04` để lại: biến vùng phủ ở hạng sâu thành thứ hạng."""
        base = FakeBase([f"c{i}" for i in range(50)])
        reranker = _by_key({"c47": 9.9})
        results = RerankedRetriever(base, reranker, candidates=50).retrieve("q", 10)
        assert results[0].chunk.chunk_id == "c47"


# --------------------------------------------------------------------------- #
# Tie-break: điểm bằng nhau thì giữ thứ tự nhánh nền
# --------------------------------------------------------------------------- #


class TestTieBreak:
    def test_equal_scores_keep_base_order(self) -> None:
        base = FakeBase(["a", "b", "c"])
        results = RerankedRetriever(base, ScriptedReranker({}, default=0.5)).retrieve("q", 3)
        assert [hit.chunk.chunk_id for hit in results] == ["a", "b", "c"]

    def test_all_tied_is_a_no_op_on_ordering(self) -> None:
        """Reranker không phân biệt được gì thì tiên nghiệm của bộ sinh được giữ.

        Quan trọng vì `W2-04` đã gặp ties thật với điểm RRF: khi hai ứng viên
        không phân biệt được, "giữ nguyên" là quyết định đúng duy nhất — đảo thứ
        tự theo hash của dict là cách âm thầm làm kết quả không tái lập được.
        """
        keys = [f"c{i}" for i in range(20)]
        base = FakeBase(keys)
        results = RerankedRetriever(base, ScriptedReranker({}, default=0.0)).retrieve("q", 20)
        assert [hit.chunk.chunk_id for hit in results] == keys

    def test_partial_tie_breaks_within_the_tied_group_only(self) -> None:
        base = FakeBase(["a", "b", "c", "d"])
        reranker = _by_key({"a": 0.1, "b": 0.5, "c": 0.5, "d": 0.9})
        results = RerankedRetriever(base, reranker).retrieve("q", 4)
        assert [hit.chunk.chunk_id for hit in results] == ["d", "b", "c", "a"]


# --------------------------------------------------------------------------- #
# `candidates` — trần cứng của nhánh
# --------------------------------------------------------------------------- #


class TestCandidatePool:
    def test_default_pool_depth_is_asked_of_the_base(self) -> None:
        base = FakeBase([f"c{i}" for i in range(100)])
        RerankedRetriever(base, ScriptedReranker({})).retrieve("q", 10)
        assert base.requested == [DEFAULT_RERANK_CANDIDATES]

    def test_candidates_sets_the_pool_depth(self) -> None:
        base = FakeBase([f"c{i}" for i in range(100)])
        RerankedRetriever(base, ScriptedReranker({}), candidates=25).retrieve("q", 10)
        assert base.requested == [25]

    def test_pool_never_shallower_than_top_k(self) -> None:
        """Không thể trả 40 kết quả đã xếp từ một pool 10 ứng viên."""
        base = FakeBase([f"c{i}" for i in range(100)])
        RerankedRetriever(base, ScriptedReranker({}), candidates=10).retrieve("q", 40)
        assert base.requested == [40]

    def test_a_chunk_outside_the_pool_can_never_be_returned(self) -> None:
        """Trần cứng: reranker chấm `c60` cao nhất nhưng pool chỉ tới `c19`.

        Đây là số phải đo TRƯỚC khi kết luận reranker tốt hay tệ — `hit_rate@1`
        sau rerank bị chặn trên bởi `hit_rate@candidates` của nhánh nền.
        """
        base = FakeBase([f"c{i}" for i in range(100)])
        reranker = _by_key({"c60": 99.0})
        results = RerankedRetriever(base, reranker, candidates=20).retrieve("q", 5)
        assert "c60" not in [hit.chunk.chunk_id for hit in results]

    def test_candidates_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="candidates phải"):
            RerankedRetriever(FakeBase([]), ScriptedReranker({}), candidates=0)


# --------------------------------------------------------------------------- #
# `top_n` — cái bẫy làm mọi metric @k mất nghĩa
# --------------------------------------------------------------------------- #


class TestTopN:
    def test_default_does_not_cap_below_top_k(self) -> None:
        base = FakeBase([f"c{i}" for i in range(50)])
        results = RerankedRetriever(base, ScriptedReranker({})).retrieve("q", 20)
        assert len(results) == 20

    def test_top_n_caps_the_returned_list(self) -> None:
        base = FakeBase([f"c{i}" for i in range(50)])
        results = RerankedRetriever(base, ScriptedReranker({}), top_n=6).retrieve("q", 20)
        assert len(results) == 6

    def test_top_n_larger_than_top_k_does_not_expand(self) -> None:
        base = FakeBase([f"c{i}" for i in range(50)])
        results = RerankedRetriever(base, ScriptedReranker({}), top_n=30).retrieve("q", 5)
        assert len(results) == 5

    def test_top_n_below_top_k_is_the_documented_trap(self) -> None:
        """`top_n=6` + `--top-k 20` = `recall@20` bị chặn ở 6 chunk.

        Test này không sửa cái bẫy, nó **ghim** nó: hành vi là đúng theo thiết kế
        (ngân sách context lúc phục vụ), nhưng ai dùng nó cho phép đo sẽ thấy
        recall@20 tụt mà không hiểu vì sao. Ghim ở đây để lần sau ai đọc test
        cũng gặp đúng câu giải thích đó.
        """
        base = FakeBase([f"c{i}" for i in range(50)])
        capped = RerankedRetriever(base, ScriptedReranker({}), top_n=6).retrieve("q", 20)
        uncapped = RerankedRetriever(base, ScriptedReranker({})).retrieve("q", 20)
        assert len(capped) == 6
        assert len(uncapped) == 20
        # Pool vẫn sâu như nhau — mất mát nằm ở lượt TRẢ VỀ, không ở lượt chấm.
        assert base.requested == [50, 50]

    def test_top_n_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="top_n phải"):
            RerankedRetriever(FakeBase([]), ScriptedReranker({}), top_n=0)


# --------------------------------------------------------------------------- #
# Điểm được giữ lại
# --------------------------------------------------------------------------- #


class TestScoresPreserved:
    def test_score_and_rerank_score_are_the_same_number(self) -> None:
        base = FakeBase(["a", "b"])
        results = RerankedRetriever(base, _by_key({"a": 0.25, "b": 0.75})).retrieve("q", 2)
        assert [(hit.score, hit.rerank_score) for hit in results] == [(0.75, 0.75), (0.25, 0.25)]

    def test_mode_is_reranked(self) -> None:
        base = FakeBase(["a"])
        results = RerankedRetriever(base, ScriptedReranker({})).retrieve("q", 1)
        assert results[0].mode is RetrievalMode.RERANKED

    def test_branch_scores_survive_the_rerank(self) -> None:
        """`W2-08` cần trả lời "reranker kéo lên chunk mà nhánh nào tìm ra?"."""
        base = FakeBase(["a", "b"])
        results = RerankedRetriever(base, _by_key({"a": 0.1, "b": 0.2})).retrieve("q", 2)
        top = results[0]
        assert top.chunk.chunk_id == "b"
        assert top.dense_score == pytest.approx(0.49)
        assert top.sparse_score == pytest.approx(8.0)


# --------------------------------------------------------------------------- #
# Ca biên và hợp đồng
# --------------------------------------------------------------------------- #


class TestEdgeCases:
    def test_empty_pool_returns_empty_without_calling_the_reranker(self) -> None:
        """Cross-encoder trên pool rỗng là một forward pass tốn công vô ích."""
        reranker = ScriptedReranker({})
        assert RerankedRetriever(FakeBase([]), reranker).retrieve("q", 10) == []
        assert reranker.calls == []

    def test_pool_smaller_than_top_k_returns_what_exists(self) -> None:
        base = FakeBase(["a", "b"])
        assert len(RerankedRetriever(base, ScriptedReranker({})).retrieve("q", 10)) == 2

    def test_reranker_sees_chunk_content_not_chunk_id(self) -> None:
        base = FakeBase(["a", "b"])
        reranker = ScriptedReranker({})
        RerankedRetriever(base, reranker).retrieve("truy vấn thật", 2)
        query, texts = reranker.calls[0]
        assert query == "truy vấn thật"
        assert texts == ("nội dung của a", "nội dung của b")

    def test_wrong_score_count_raises_instead_of_misaligning(self) -> None:
        """Hợp đồng bị vi phạm thì điểm gán lệch chunk và kết quả vẫn trông hợp lệ."""

        class ShortReranker(Reranker):
            name = "short"

            def score(self, query: str, texts: Sequence[str]) -> list[float]:
                return [1.0]

        with pytest.raises(RuntimeError, match="điểm cho"):
            RerankedRetriever(FakeBase(["a", "b", "c"]), ShortReranker()).retrieve("q", 3)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_scores_raise_instead_of_sorting_arbitrarily(self, bad: float) -> None:
        """NaN so với mọi thứ đều False nên `sorted` trả thứ tự tuỳ ý, im lặng.

        Chế độ hỏng thật của fp16: một overflow trung gian là đủ, và hậu quả trông
        y như "model kém" trên bảng metric.
        """
        base = FakeBase(["a", "b", "c"])
        reranker = _by_key({"a": 0.5, "b": bad, "c": 0.1})
        with pytest.raises(RuntimeError, match="không hữu hạn"):
            RerankedRetriever(base, reranker).retrieve("q", 3)

    def test_filters_reach_the_base_untouched(self) -> None:
        """`W2-06` (tenant isolation) phụ thuộc vào chuyện này không bị đánh rơi."""
        base = FakeBase(["a"])
        filters = {"tenant_id": "acme"}
        RerankedRetriever(base, ScriptedReranker({})).retrieve("q", 1, filters=filters)
        assert base.filters_seen == [filters]


class TestForwarding:
    def test_fetch_doc_chunks_is_forwarded(self) -> None:
        """Thiếu method này thì eval âm thầm lùi về nhãn đã lưu — xem `sparse.py`."""
        base = FakeBase(["a"])
        branch = RerankedRetriever(base, ScriptedReranker({}))
        assert [c.chunk_id for c in branch.fetch_doc_chunks(["d1", "d2"])] == ["d1::0", "d2::0"]
        assert base.fetch_batch == 512

    def test_verify_schema_is_forwarded(self) -> None:
        base = FakeBase(["a"])
        RerankedRetriever(base, ScriptedReranker({})).verify_schema()
        assert base.verified is True

    def test_verify_schema_is_a_no_op_when_base_has_none(self) -> None:
        class BareBase(Retriever):
            name = "bare"

            def retrieve(
                self,
                query: str,
                top_k: int = 10,
                *,
                filters: dict[str, Any] | None = None,
            ) -> list[RetrievedChunk]:
                return []

        RerankedRetriever(BareBase(), ScriptedReranker({})).verify_schema()


# --------------------------------------------------------------------------- #
# `name` — nhãn phải mang đủ để hai lần chạy khác nhau không trùng
# --------------------------------------------------------------------------- #


class TestName:
    def test_name_carries_base_and_reranker_and_pool(self) -> None:
        base = FakeBase([], name="qdrant-hybrid:rag_bgem3:rrf1-c20")
        branch = RerankedRetriever(base, ScriptedReranker({}), candidates=50)
        assert branch.name == "reranked[qdrant-hybrid:rag_bgem3:rrf1-c20]:scripted:n50"

    def test_name_shows_top_n_only_when_set(self) -> None:
        base = FakeBase([], name="b")
        assert RerankedRetriever(base, ScriptedReranker({})).name.endswith(
            f":n{DEFAULT_RERANK_CANDIDATES}"
        )
        assert RerankedRetriever(base, ScriptedReranker({}), top_n=6).name.endswith("-top6")

    def test_different_pool_depths_get_different_names(self) -> None:
        base = FakeBase([], name="b")
        names = {
            RerankedRetriever(base, ScriptedReranker({}), candidates=depth).name
            for depth in (20, 50, 100)
        }
        assert len(names) == 3


# --------------------------------------------------------------------------- #
# `CrossEncoderReranker` — phần dựng được mà không tải model
# --------------------------------------------------------------------------- #


class TestCrossEncoderConstruction:
    def test_default_model_is_bge_reranker_v2_m3(self) -> None:
        assert CrossEncoderReranker(device="cpu").model_name == BGE_RERANKER_V2_M3

    def test_construction_does_not_load_the_model(self) -> None:
        """2,2GB trọng số không được nạp chỉ vì ai đó dựng object để đọc `name`."""
        assert CrossEncoderReranker(device="cpu")._model is None

    def test_max_length_is_in_the_name_but_batch_size_is_not(self) -> None:
        """`max_length` cắt nội dung nên nó đổi điểm; `batch_size` chỉ đổi tốc độ.

        Cùng lý lẽ với `IndexConfig.fingerprint` (`W1-06`), nơi `device` và
        `batch_size` bị loại khỏi fingerprint vì đúng lý do này.
        """
        slow = CrossEncoderReranker(device="cpu", batch_size=1, max_length=512)
        fast = CrossEncoderReranker(device="cpu", batch_size=64, max_length=512)
        assert slow.name == fast.name
        assert CrossEncoderReranker(device="cpu", max_length=256).name != slow.name

    def test_name_carries_model_and_device(self) -> None:
        name = CrossEncoderReranker("some/model", device="cpu").name
        assert name == f"some/model@cpu:L{DEFAULT_RERANK_MAX_LENGTH}"

    def test_activation_appears_in_the_name_only_when_not_default(self) -> None:
        assert ":sigmoid" not in CrossEncoderReranker(device="cpu").name
        assert CrossEncoderReranker(device="cpu", activation="sigmoid").name.endswith(":sigmoid")

    def test_default_activation_is_raw_logits(self) -> None:
        """Sigmoid đơn điệu nên không đổi thứ hạng, nhưng bão hoà ở float32."""
        assert CrossEncoderReranker(device="cpu").activation == "none"

    def test_defaults_are_the_documented_constants(self) -> None:
        reranker = CrossEncoderReranker(device="cpu")
        assert reranker.batch_size == DEFAULT_RERANK_BATCH_SIZE
        assert reranker.max_length == DEFAULT_RERANK_MAX_LENGTH

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"batch_size": 0}, "batch_size phải"),
            ({"max_length": 0}, "max_length phải"),
            ({"activation": "relu"}, "activation phải"),
        ],
    )
    def test_bad_arguments_fail_at_construction(self, kwargs: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            CrossEncoderReranker(device="cpu", **kwargs)

    def test_auto_dtype_is_fp16_on_cuda_and_fp32_elsewhere(self) -> None:
        """`auto` phải được phân giải NGAY, không để chữ "auto" lọt vào `name`.

        Một nhãn ghi "auto" thì đọc log xong vẫn không biết model đã chạy ở độ
        chính xác nào — mà fp16 và fp32 cho điểm khác nhau, nên đó là hai lần
        chạy khác nhau đang dùng chung một tên.
        """
        assert CrossEncoderReranker(device="cpu").dtype is None
        assert CrossEncoderReranker(device="cuda").dtype == "float16"
        assert CrossEncoderReranker(device="cuda:1").dtype == "float16"
        assert "auto" not in CrossEncoderReranker(device="cuda").name

    def test_dtype_is_in_the_name_because_it_changes_scores(self) -> None:
        names = {
            CrossEncoderReranker(device="cuda", dtype=dtype).name
            for dtype in (None, "float16", "bfloat16", "float32")
        }
        assert len(names) == 4

    def test_explicit_none_dtype_overrides_auto_on_cuda(self) -> None:
        """Cần đường lùi về fp32 trên GPU để lấy số nền so sánh với fp16."""
        assert CrossEncoderReranker(device="cuda", dtype=None).dtype is None

    def test_bad_dtype_fails_at_construction(self) -> None:
        with pytest.raises(ValueError, match="dtype phải"):
            CrossEncoderReranker(device="cpu", dtype="int8")

    def test_empty_input_returns_empty_without_loading_the_model(self) -> None:
        reranker = CrossEncoderReranker(device="cpu")
        assert reranker.score("q", []) == []
        assert reranker.count_pair_tokens("q", []) == []
        assert reranker._model is None


# --------------------------------------------------------------------------- #
# `build_branch(..., "reranked")`
# --------------------------------------------------------------------------- #


def _store(*, sparse: bool = True) -> QdrantDenseRetriever:
    """Store không kết nối — `client` là lazy nên dựng thôi thì không chạm mạng."""
    return QdrantDenseRetriever(
        HashingEmbeddingProvider(dimension=64, sparse=sparse),
        collection="rag_test_rerank",
    )


class TestBuildBranch:
    def test_reranked_is_implemented_since_w2_05(self) -> None:
        branch = build_branch(_store(), "reranked", rerank_device="cpu")
        assert isinstance(branch, RerankedRetriever)

    def test_default_base_is_hybrid(self) -> None:
        branch = build_branch(_store(), "reranked", rerank_device="cpu")
        assert isinstance(branch, RerankedRetriever)
        assert "qdrant-hybrid" in branch.base.name

    @pytest.mark.parametrize(
        ("base_mode", "marker"),
        [("dense", "qdrant-dense:"), ("sparse", "qdrant-sparse:"), ("hybrid", "qdrant-hybrid:")],
    )
    def test_base_is_selectable(self, base_mode: str, marker: str) -> None:
        branch = build_branch(_store(), "reranked", base=base_mode, rerank_device="cpu")
        assert isinstance(branch, RerankedRetriever)
        assert branch.base.name.startswith(marker)

    def test_fusion_options_reach_the_hybrid_base(self) -> None:
        branch = build_branch(
            _store(), "reranked", base="hybrid", k=1, candidate_k=20, rerank_device="cpu"
        )
        assert isinstance(branch, RerankedRetriever)
        assert branch.base.name.endswith("rrf1-c20")

    def test_fusion_options_are_still_rejected_by_a_non_hybrid_base(self) -> None:
        """Phép kiểm của `build_branch` áp dụng nguyên vẹn qua lượt đệ quy."""
        with pytest.raises(ValueError, match="không nhận tham số"):
            build_branch(_store(), "reranked", base="dense", k=1, rerank_device="cpu")

    def test_rerank_options_are_rejected_by_other_branches(self) -> None:
        """`--rerank-candidates 50 --retrieval-mode hybrid` phải nổ, không bị bỏ qua."""
        with pytest.raises(ValueError, match="chỉ có nghĩa với nhánh 'reranked'"):
            build_branch(_store(), "hybrid", rerank_candidates=50)

    def test_rerank_options_rejected_by_dense_too(self) -> None:
        with pytest.raises(ValueError, match="không nhận tham số"):
            build_branch(_store(), "dense", rerank_top_n=6)

    def test_every_rerank_option_is_actually_consumed(self) -> None:
        """`RERANK_OPTIONS` mà thừa một tên thì tên đó bị nhận rồi bỏ im lặng."""
        from rag_core.retrieval import RERANK_OPTIONS

        supplied = dict.fromkeys(RERANK_OPTIONS - {"base", "reranker_model"}, None)
        supplied.update(
            rerank_device="cpu",
            rerank_batch_size=2,
            rerank_max_length=64,
            rerank_activation="none",
            rerank_dtype="float32",
            rerank_candidates=7,
            rerank_top_n=3,
        )
        assert not [name for name, value in supplied.items() if value is None], (
            "có tên trong RERANK_OPTIONS mà test này chưa gán giá trị — "
            "hoặc nó vô dụng, hoặc test này đã lạc hậu"
        )
        branch = build_branch(_store(), "reranked", **supplied)
        assert isinstance(branch, RerankedRetriever)

    def test_reranked_base_reranked_is_refused(self) -> None:
        with pytest.raises(ValueError, match="không có nghĩa"):
            build_branch(_store(), "reranked", base="reranked", rerank_device="cpu")

    def test_garbage_base_gets_the_invalid_branch_message(self) -> None:
        """Không phải một `ValueError` trần của enum — thông điệp phải nêu tên hợp lệ."""
        with pytest.raises(ValueError, match="Nhánh truy hồi không hợp lệ"):
            build_branch(_store(), "reranked", base="bogus", rerank_device="cpu")

    def test_rerank_knobs_reach_the_reranker(self) -> None:
        branch = build_branch(
            _store(),
            "reranked",
            reranker_model="some/model",
            rerank_device="cpu",
            rerank_batch_size=4,
            rerank_max_length=256,
            rerank_activation="sigmoid",
            rerank_dtype="bfloat16",
            rerank_candidates=30,
            rerank_top_n=6,
        )
        assert isinstance(branch, RerankedRetriever)
        reranker = branch.reranker
        assert isinstance(reranker, CrossEncoderReranker)
        assert (reranker.model_name, reranker.device) == ("some/model", "cpu")
        assert (reranker.batch_size, reranker.max_length) == (4, 256)
        assert reranker.activation == "sigmoid"
        assert reranker.dtype == "bfloat16"
        assert (branch.candidates, branch.top_n) == (30, 6)

    def test_name_of_the_whole_branch_is_reproducible(self) -> None:
        branch = build_branch(
            _store(),
            "reranked",
            base="hybrid",
            k=1,
            candidate_k=20,
            rerank_device="cpu",
            rerank_candidates=50,
        )
        assert branch.name == (
            "reranked[qdrant-hybrid:rag_test_rerank:rrf1-c20]"
            f":{BGE_RERANKER_V2_M3}@cpu:L{DEFAULT_RERANK_MAX_LENGTH}:n50"
        )

    def test_sparse_base_still_fails_early_without_sparse_vectors(self) -> None:
        """Nhánh nền chết lúc dựng, không lúc truy vấn đầu — kể cả qua rerank."""
        with pytest.raises(ValueError, match="không sinh sparse vector"):
            build_branch(_store(sparse=False), "reranked", base="sparse", rerank_device="cpu")

    def test_every_retrieval_mode_is_now_buildable(self) -> None:
        """`SUPPORTED_MODES` đã bằng `RetrievalMode` — không còn tên hợp lệ mà chưa cài."""
        from rag_core.retrieval import SUPPORTED_MODES

        assert set(SUPPORTED_MODES) == set(RetrievalMode)
