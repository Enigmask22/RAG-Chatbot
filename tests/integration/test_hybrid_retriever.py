"""`W2-04` — nhánh hybrid trên Qdrant thật, và phép đối chiếu với RRF của Qdrant.

Phần đáng nhất của file này là `TestAgainstQdrantNativeRRF`: Qdrant 1.15 có
`Fusion.RRF` server-side, nên ta có một **bản tham chiếu độc lập** để đối chiếu.
Tự cài RRF (vì cần `k` cấu hình được và cần thứ hạng từng nhánh) không có nghĩa
là tự tin — nếu hai bản cho thứ tự khác nhau thì một trong hai sai, và biết được
điều đó là rẻ.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from qdrant_client import models

from rag_core.embedding import HashingEmbeddingProvider
from rag_core.retrieval import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    QdrantDenseRetriever,
    QdrantHybridRetriever,
    QdrantSparseRetriever,
    build_branch,
    points_to_chunks,
    reciprocal_rank_fusion,
)
from rag_core.schemas import Chunk, DocType, DocumentMetadata, Language, RetrievalMode

pytest.importorskip("qdrant_client", reason="cần extra `qdrant`: uv sync --extra qdrant")
pytestmark = pytest.mark.integration

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")

# Corpus dựng có chủ đích để truy vấn `_TIE_FREE_QUERY` cho điểm **phân biệt** ở
# **cả hai** nhánh: doc `a`…`d` chia sẻ số lượng token truy vấn giảm dần, `f` chỉ
# chia sẻ "thống kê", `e` không chia sẻ gì. Không có chuyện đó thì phép đối chiếu
# với Qdrant ở dưới lung lay — hai chunk bằng điểm thì thứ tự giữa chúng không
# thuộc hợp đồng của ai cả, và bản đầu của file này đỏ ngẫu nhiên vì lý do đó.
#
# `f` cũng mang một mã lạ (`GSO-2024-XII`) cho ca "chỉ sparse tới được" — cố ý đặt
# vào một doc **đã** có token trùng với `_TIE_FREE_QUERY`, để thêm nó không tạo ra
# một doc điểm-0 thứ hai và làm hỏng tính phân biệt.
_TEXTS = {
    "a": "Ngân sách nhà nước đầu tư công hạ tầng giao thông thống kê quý bốn năm 2024.",
    "b": "Ngân sách nhà nước đầu tư công hạ tầng giao thông trong kế hoạch trung hạn.",
    "c": "Ngân sách nhà nước đầu tư công cho các dự án trọng điểm của địa phương.",
    "d": "Ngân sách được phân bổ theo nghị quyết của quốc hội khoá mới.",
    "e": "Chương trình tiêm chủng mở rộng cho trẻ dưới một tuổi ở vùng cao.",
    "f": "Thống kê tỉ lệ nhập học bậc trung học phổ thông theo mã báo cáo GSO-2024-XII.",
}

#: Truy vấn duy nhất trong file này cho điểm phân biệt ở cả hai nhánh. Mọi phép so
#: **thứ tự** với Qdrant phải dùng nó; các test khác dùng truy vấn nào cũng được.
_TIE_FREE_QUERY = "ngân sách đầu tư công hạ tầng giao thông thống kê"


def _chunk(key: str, text: str, *, tenant: str = "t1") -> Chunk:
    return Chunk(
        chunk_id=f"doc-{key}::00000",
        doc_id=f"doc-{key}",
        content=text,
        chunk_index=0,
        metadata=DocumentMetadata(
            source_url=f"https://example.org/{key}",
            license="CC BY 4.0",
            lang=Language.VI,
            doc_type=DocType.DEV_REPORT,
        ),
        extra={"tenant_id": tenant},
    )


CHUNKS = [_chunk(key, text) for key, text in _TEXTS.items()]


@pytest.fixture
def store() -> Iterator[QdrantDenseRetriever]:
    collection = f"test_hybrid_{uuid.uuid4().hex[:8]}"
    retriever = QdrantDenseRetriever(
        HashingEmbeddingProvider(dimension=256, sparse=True),
        collection=collection,
        url=QDRANT_URL,
    )
    try:
        retriever.ensure_collection(recreate=True)
    except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
        pytest.fail(f"Không kết nối được Qdrant tại {QDRANT_URL}: {exc}. Chạy `make up` trước.")
    try:
        retriever.upsert(CHUNKS)
        yield retriever
    finally:
        retriever.client.delete_collection(collection)


@pytest.fixture
def hybrid(store: QdrantDenseRetriever) -> QdrantHybridRetriever:
    return QdrantHybridRetriever(store)


class TestItIsARetriever:
    def test_retrieve_returns_hybrid_results(self, hybrid: QdrantHybridRetriever) -> None:
        results = hybrid.retrieve("ngân sách nhà nước đầu tư công", top_k=3)
        assert results
        assert all(r.mode is RetrievalMode.HYBRID for r in results)

    def test_ranks_are_contiguous_from_one(self, hybrid: QdrantHybridRetriever) -> None:
        results = hybrid.retrieve("ngân sách thống kê", top_k=10)
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_scores_are_non_increasing(self, hybrid: QdrantHybridRetriever) -> None:
        scores = [r.score for r in hybrid.retrieve("ngân sách thống kê", top_k=10)]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_is_respected(self, hybrid: QdrantHybridRetriever) -> None:
        assert len(hybrid.retrieve("ngân sách thống kê tiêm chủng", top_k=2)) == 2

    def test_build_branch_path(self, store: QdrantDenseRetriever) -> None:
        """Đường mà `make eval-retrieval MODE=hybrid` thực sự đi."""
        assert build_branch(store, "hybrid").retrieve("ngân sách", top_k=2)

    def test_deterministic_across_calls(self, hybrid: QdrantHybridRetriever) -> None:
        first = [r.chunk.chunk_id for r in hybrid.retrieve("ngân sách thống kê", top_k=7)]
        for _ in range(5):
            assert [
                r.chunk.chunk_id for r in hybrid.retrieve("ngân sách thống kê", top_k=7)
            ] == first


class TestOneForwardPassOneRequest:
    def test_query_is_embedded_exactly_once(
        self, store: QdrantDenseRetriever, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lý do lớp này không gọi `retrieve()` + `retrieve_sparse()`.

        Hai method đó embed truy vấn hai lần — 12,6 ms mỗi lần (`W2-03` §8), tức
        trả gấp đôi tiền forward pass cho đúng một kết quả. Đây là phiên bản phía
        truy vấn của quyết định "một forward pass" ở `W2-01`.
        """
        calls: list[str] = []
        original = store.embeddings.embed_query_hybrid

        def counted_hybrid(text: str) -> Any:
            calls.append(text)
            return original(text)

        monkeypatch.setattr(store.embeddings, "embed_query_hybrid", counted_hybrid)
        QdrantHybridRetriever(store).retrieve("ngân sách", top_k=3)
        assert calls == ["ngân sách"], f"đã gọi provider {len(calls)} lần"

        # Canh **giao diện** retriever dùng, không canh nội bộ provider. Bản đầu
        # của test này đếm cả `embed_query` và thấy 2 lần gọi — nhưng lần thứ hai
        # đến từ *bên trong* `HashingEmbeddingProvider.embed_query_hybrid`, không
        # phải từ retriever. Nội bộ provider là hợp đồng của provider:
        # `BgeM3EmbeddingProvider.embed_query_hybrid` chạy đúng một `_forward`, và
        # điều đó có test riêng ở `tests/unit/test_bge_m3.py`.

    def test_both_branches_go_in_one_request(
        self, store: QdrantDenseRetriever, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Một round trip, không hai — và Qdrant tự chạy song song được."""
        batch_calls: list[int] = []
        single_calls: list[str] = []
        original = store.client.query_batch_points
        single = store.client.query_points

        def counted_batch(*args: Any, **kwargs: Any) -> Any:
            batch_calls.append(len(kwargs.get("requests") or ()))
            return original(*args, **kwargs)

        def counted_single(*args: Any, **kwargs: Any) -> Any:
            single_calls.append("query_points")
            return single(*args, **kwargs)

        monkeypatch.setattr(store.client, "query_batch_points", counted_batch)
        monkeypatch.setattr(store.client, "query_points", counted_single)
        QdrantHybridRetriever(store).retrieve("ngân sách", top_k=3)
        assert batch_calls == [2], "một request mang đúng hai truy vấn"
        assert single_calls == []


class TestComplementarity:
    def test_hybrid_reaches_what_only_sparse_reaches(self, store: QdrantDenseRetriever) -> None:
        """Nếu hybrid không với được chunk mà chỉ sparse tìm ra thì cả `W2-04` vô
        nghĩa. Dùng truy vấn mã lạ — `W2-03` đo được đó là địa hạt của sparse."""
        sparse_top = QdrantSparseRetriever(store).retrieve("GSO-2024-XII", top_k=1)
        hybrid_ids = [
            r.chunk.chunk_id for r in QdrantHybridRetriever(store).retrieve("GSO-2024-XII", 7)
        ]
        assert sparse_top[0].chunk.chunk_id in hybrid_ids

    def test_hybrid_reaches_what_only_dense_reaches(self, store: QdrantDenseRetriever) -> None:
        dense_top = store.retrieve("ngân sách nhà nước đầu tư công", top_k=1)
        hybrid_ids = [
            r.chunk.chunk_id
            for r in QdrantHybridRetriever(store).retrieve("ngân sách nhà nước đầu tư công", 7)
        ]
        assert dense_top[0].chunk.chunk_id in hybrid_ids

    def test_union_is_never_smaller_than_either_branch(self, store: QdrantDenseRetriever) -> None:
        query = "ngân sách thống kê tiêm chủng"
        dense = {r.chunk.chunk_id for r in store.retrieve(query, 20)}
        sparse = {r.chunk.chunk_id for r in QdrantSparseRetriever(store).retrieve(query, 20)}
        hybrid = {r.chunk.chunk_id for r in QdrantHybridRetriever(store).retrieve(query, 20)}
        assert hybrid == dense | sparse

    def test_a_query_with_no_lexical_overlap_still_works(self, store: QdrantDenseRetriever) -> None:
        """Sparse trả rỗng (`W2-03`), hybrid phải rơi về dense chứ không chết."""
        query = "zzzqqq khongtontai"
        assert QdrantSparseRetriever(store).retrieve(query, 5) == []
        results = QdrantHybridRetriever(store).retrieve(query, 5)
        assert results
        assert all(r.sparse_score is None for r in results)


class TestBranchScores:
    def test_scores_of_both_branches_survive(self, hybrid: QdrantHybridRetriever) -> None:
        results = hybrid.retrieve("ngân sách đầu tư công thống kê", top_k=7)
        assert any(r.dense_score is not None and r.sparse_score is not None for r in results), (
            "phải có ít nhất một chunk cả hai nhánh tìm ra, nếu không thì test này vô nghĩa"
        )

    def test_score_is_rrf_not_a_branch_score(self, hybrid: QdrantHybridRetriever) -> None:
        """Điểm RRF với hai nhánh, `k=60`, luôn ≤ 2/61 ≈ 0,0328. Điểm dense là
        cosine (tới 1,0) và sparse là dot product (không trần), nên nếu `score`
        mang điểm nhánh thì nó sẽ vượt ngưỡng này ngay."""
        for result in hybrid.retrieve("ngân sách", top_k=5):
            assert 0 < result.score <= 2 / 61

    def test_filter_applies_to_both_branches(self, hybrid: QdrantHybridRetriever) -> None:
        """`W2-06` (cô lập tenant) dựa vào chỗ này. Một lỗ ở **một** nhánh của
        hybrid sẽ không lộ ra ở test của nhánh kia."""
        assert hybrid.retrieve("ngân sách", top_k=5, filters={"tenant_id": "t1"})
        assert hybrid.retrieve("ngân sách", top_k=5, filters={"tenant_id": "khac"}) == []


class TestCandidateDepth:
    """`candidate_k` phải đi tới `limit` của **cả hai** truy vấn gửi cho Qdrant.

    Hiệu ứng của độ sâu lên thứ hạng đã có test bằng số tính tay
    (`test_rrf.py::test_deep_agreement_beats_shallow_solo` và
    `test_shallow_pool_hides_that_agreement`). Ở đây chỉ kiểm đường dây: một cần
    điều khiển không tới được dây thì mọi lý lẽ về nó là vô nghĩa.
    """

    @staticmethod
    def _limits(
        store: QdrantDenseRetriever,
        monkeypatch: pytest.MonkeyPatch,
        *,
        candidate_k: int | None,
        top_k: int,
    ) -> list[int]:
        seen: list[int] = []
        original = store.client.query_batch_points

        def spy(*args: Any, **kwargs: Any) -> Any:
            for request in kwargs.get("requests") or ():
                seen.append(int(request.limit))
            return original(*args, **kwargs)

        monkeypatch.setattr(store.client, "query_batch_points", spy)
        QdrantHybridRetriever(store, candidate_k=candidate_k).retrieve("ngân sách", top_k=top_k)
        return seen

    def test_candidate_k_reaches_both_queries(
        self, store: QdrantDenseRetriever, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert self._limits(store, monkeypatch, candidate_k=37, top_k=5) == [37, 37]

    def test_default_depth_reaches_the_wire(
        self, store: QdrantDenseRetriever, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert self._limits(store, monkeypatch, candidate_k=None, top_k=5) == [50, 50]

    def test_depth_is_raised_to_top_k(
        self, store: QdrantDenseRetriever, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lấy ít hơn số kết quả phải trả về thì danh sách hợp nhất ngắn hơn
        `top_k` một cách vô cớ."""
        assert self._limits(store, monkeypatch, candidate_k=1, top_k=80) == [80, 80]


class TestAgainstQdrantNativeRRF:
    """Đối chiếu với `Fusion.RRF` server-side của Qdrant — bản tham chiếu độc lập.

    Tự cài RRF (vì cần `k` cấu hình được và cần thứ hạng từng nhánh) không có
    nghĩa là tự tin. Qdrant 1.15 có fusion server-side, nên có một bản khác để
    đối chiếu, và nó rẻ.

    ⚠️ Phép đối chiếu **không** so được thứ tự trong nhóm bằng điểm. Corpus ở đây
    có điểm trùng thật (hai chunk chia đúng cùng tập token của truy vấn thì điểm
    sparse bằng nhau; chunk không trùng token nào thì điểm dense bằng 0), và thứ
    tự trong nhóm bằng điểm không thuộc hợp đồng của Qdrant — nó đi qua đường
    prefetch chứ không phải đường query thường, và hai đường đó không hứa cùng
    thứ tự. Nên: so **tập điểm** (mạnh, không phụ thuộc tie), và so thứ tự chỉ
    trên đoạn đầu **trước** chỗ bằng điểm đầu tiên.
    """

    @staticmethod
    def _native(
        store: QdrantDenseRetriever, query: str, depth: int, limit: int
    ) -> list[tuple[str, float]]:
        hybrid = store.embeddings.embed_query_hybrid(query)
        assert hybrid is not None
        dense_vector, sparse_vector = hybrid
        response = store.client.query_points(
            collection_name=store.collection,
            prefetch=[
                models.Prefetch(
                    query=list(map(float, dense_vector)),
                    using=DENSE_VECTOR_NAME,
                    limit=depth,
                ),
                models.Prefetch(
                    query=models.SparseVector(**sparse_vector.as_qdrant()),
                    using=SPARSE_VECTOR_NAME,
                    limit=depth,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [
            (c.chunk.chunk_id, c.score)
            for c in points_to_chunks(response.points, mode=RetrievalMode.HYBRID)
        ]

    @staticmethod
    def _ours(
        store: QdrantDenseRetriever, query: str, depth: int, limit: int, k: int
    ) -> list[tuple[str, float]]:
        return [
            (r.chunk.chunk_id, r.score)
            for r in QdrantHybridRetriever(store, k=k, candidate_k=depth).retrieve(
                query, top_k=limit
            )
        ]

    @staticmethod
    def _prefix_before_first_tie(rows: list[tuple[str, float]]) -> list[str]:
        keys: list[str] = []
        for i, (key, score) in enumerate(rows):
            if i + 1 < len(rows) and rows[i + 1][1] == pytest.approx(score):
                break
            keys.append(key)
        return keys

    def test_native_fusion_is_available(self, store: QdrantDenseRetriever) -> None:
        assert self._native(store, _TIE_FREE_QUERY, 7, 7)

    def test_qdrant_uses_k_equal_1_not_the_paper_value_60(
        self, store: QdrantDenseRetriever
    ) -> None:
        """Phát hiện làm nên lý lẽ cho việc tự cài, và nó đo được chứ không phỏng
        đoán: suy `k` của Qdrant từ chính điểm nó trả về.

        `k = 1` là **rất dốc** — hạng 1 hơn hạng 2 gấp 1,5 lần, còn `k = 60` của
        bài báo thì chỉ hơn 1,6%. Nói cách khác Qdrant tin thứ hạng của từng nhánh
        gần như tuyệt đối. Đó là một lựa chọn hợp lệ, nhưng nó **không** phải giá
        trị của bài báo và nó **không** cấu hình được — nên `W2-08` không quét
        được `k` nếu dùng bản của Qdrant.
        """
        native = dict(self._native(store, _TIE_FREE_QUERY, 7, 7))
        matched: list[int] = []
        for k in range(0, 101):
            ours = dict(self._ours(store, _TIE_FREE_QUERY, 7, 7, k))
            if ours.keys() == native.keys() and all(
                ours[key] == pytest.approx(score, rel=1e-6) for key, score in native.items()
            ):
                matched.append(k)
        assert matched == [1], f"k khớp: {matched}"

    def test_our_scores_match_qdrant_exactly_at_k_1(self, store: QdrantDenseRetriever) -> None:
        """So **tập điểm**, không so thứ tự — mạnh và không phụ thuộc tie.

        Nếu công thức của ta khác Qdrant thì tập điểm lệch ngay, kể cả khi thứ tự
        tình cờ trùng nhau.
        """
        # `rel=1e-6`, không chặt hơn: trường `score` của Qdrant là **float32**
        # (nó trả 0,6666667 cho chỗ ta tính 0,6666666666666666). Đòi chặt hơn thế
        # là đòi độ chính xác mà đường truyền không mang được.
        native = dict(self._native(store, _TIE_FREE_QUERY, 7, 7))
        ours = dict(self._ours(store, _TIE_FREE_QUERY, 7, 7, 1))
        assert ours.keys() == native.keys(), "tập chunk lệch"
        for key, score in native.items():
            assert ours[key] == pytest.approx(score, rel=1e-6), key

    def test_ordering_matches_down_to_the_first_tie(self, store: QdrantDenseRetriever) -> None:
        """Thứ tự khớp trên đoạn mà thứ tự **có nghĩa**.

        Cắt ở chỗ bằng điểm đầu tiên: từ đó trở đi hai thứ tự đều đúng, và đòi
        chúng giống nhau là đòi một thứ Qdrant không hứa.
        """
        native = self._native(store, _TIE_FREE_QUERY, 7, 7)
        ours = self._ours(store, _TIE_FREE_QUERY, 7, 7, 1)
        prefix = self._prefix_before_first_tie(native)
        assert len(prefix) >= 5, f"corpus phải cho ít nhất 5 hạng phân biệt, được {len(prefix)}"
        assert [key for key, _ in ours][: len(prefix)] == prefix

    def test_our_default_k_is_the_paper_value_and_that_is_the_point(self) -> None:
        """Ta mặc định `k = 60` (bài báo), Qdrant cố định `k = 1`. Hai lựa chọn
        khác nhau, và `W2-04` đo cả hai — xem `reports/w2-04-rrf.md` §5."""
        from rag_core.retrieval import RRF_K

        assert RRF_K == 60


class TestRefusals:
    def test_dense_only_provider_is_refused(self) -> None:
        store = QdrantDenseRetriever(
            HashingEmbeddingProvider(dimension=256, sparse=False),
            collection="khong-dung-den",
            url=QDRANT_URL,
        )
        with pytest.raises(ValueError, match="không có gì để hợp nhất"):
            build_branch(store, "hybrid")


class TestFuseUsesRealRanks:
    def test_fusion_input_is_the_rank_from_qdrant(self, store: QdrantDenseRetriever) -> None:
        """`points_to_chunks` đánh `rank` liên tục từ 1, và RRF lấy **chính** thứ
        hạng đó làm đầu vào. Test này nối hai tầng lại để một lỗ trong dãy rank
        không đi lọt vào phép hợp nhất."""
        query = "ngân sách thống kê"
        dense = store.retrieve(query, 7)
        sparse = QdrantSparseRetriever(store).retrieve(query, 7)
        expected = reciprocal_rank_fusion(
            [[c.chunk.chunk_id for c in dense], [c.chunk.chunk_id for c in sparse]],
            k=60,
            limit=7,
        )
        actual = QdrantHybridRetriever(store, candidate_k=7).retrieve(query, top_k=7)
        assert [i.key for i in expected] == [r.chunk.chunk_id for r in actual]
        assert [i.score for i in expected] == pytest.approx([r.score for r in actual])
