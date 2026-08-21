"""`W2-05` — nhánh reranked trên Qdrant thật, và cross-encoder thật.

Ba tầng, và chúng kiểm ba thứ khác nhau:

* `TestWiring` (`integration`) — Qdrant thật + reranker giả. Kiểm **đường dây**:
  `build_branch` → nhánh nền → pool → xếp lại → filter đi tới đâu. Reranker giả
  vì ở đây câu hỏi không phải "model tốt không", mà "ứng viên có tới được model
  không, và đúng thứ tự không".
* `TestPoolDepthReachesTheWire` (`integration`) — `rerank_candidates` có thật sự
  biến thành `limit` trong request Qdrant hay không. `W2-04` đã học bài này:
  một tham số "có tác dụng" theo test đơn vị mà không tới được dây thì vô nghĩa.
* `TestRealCrossEncoder` (`gpu`) — model thật, 2,2GB. Kiểm **hợp đồng** của
  `Reranker`: batch == single, thứ tự giảm dần, sigmoid đơn điệu. Và đo hai thứ
  không đoán được: sigmoid có bão hoà không, cặp có bị cắt ở 512 không.
"""

from __future__ import annotations

import math
import os
import uuid
from collections.abc import Iterator, Sequence
from typing import Any

import pytest

from rag_core.embedding import HashingEmbeddingProvider
from rag_core.reranking import CrossEncoderReranker, Reranker
from rag_core.retrieval import (
    QdrantDenseRetriever,
    RerankedRetriever,
    build_branch,
)
from rag_core.schemas import Chunk, DocType, DocumentMetadata, Language, RetrievalMode

pytest.importorskip("qdrant_client", reason="cần extra `qdrant`: uv sync --extra qdrant")
pytestmark = pytest.mark.integration

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")

#: Corpus nhỏ, dùng lại tinh thần của `W2-04`: token trùng với truy vấn giảm dần
#: từ `a` xuống `d` để nhánh nền cho thứ tự **phân biệt**. Reranker giả ở đây sẽ
#: cố ý đảo thứ tự đó, nên phép so mới có nghĩa.
_TEXTS = {
    "a": "Ngân sách nhà nước đầu tư công hạ tầng giao thông thống kê quý bốn năm 2024.",
    "b": "Ngân sách nhà nước đầu tư công hạ tầng giao thông trong kế hoạch trung hạn.",
    "c": "Ngân sách nhà nước đầu tư công cho các dự án trọng điểm của địa phương.",
    "d": "Ngân sách được phân bổ theo nghị quyết của quốc hội khoá mới.",
    "e": "Chương trình tiêm chủng mở rộng cho trẻ dưới một tuổi ở vùng cao.",
    "f": "Thống kê tỉ lệ nhập học bậc trung học phổ thông theo mã báo cáo GSO-2024-XII.",
    # `g` thuộc tenant khác — chỉ tồn tại cho test filter, xem `_TENANTS`.
    "g": "Ngân sách đầu tư công hạ tầng giao thông của một khách hàng khác.",
}

#: Tenant của từng chunk. Mặc định `t1`; chỉ `g` khác, và đó là điểm của nó.
_TENANTS = {"g": "t2"}

_QUERY = "ngân sách đầu tư công hạ tầng giao thông thống kê"


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


CHUNKS = [_chunk(key, text, tenant=_TENANTS.get(key, "t1")) for key, text in _TEXTS.items()]


class SpyReranker(Reranker):
    """Reranker giả chấm theo một thứ tự ưu tiên cho trước, và ghi lại đầu vào."""

    def __init__(self, preferred: Sequence[str] = ()) -> None:
        self.name = "spy"
        self.preferred = list(preferred)
        self.seen: list[tuple[str, ...]] = []

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.seen.append(tuple(texts))
        scores: list[float] = []
        for text in texts:
            match = next((i for i, key in enumerate(self.preferred) if _TEXTS[key] == text), None)
            scores.append(0.0 if match is None else float(len(self.preferred) - match))
        return scores


@pytest.fixture
def store() -> Iterator[QdrantDenseRetriever]:
    collection = f"test_rerank_{uuid.uuid4().hex[:8]}"
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


class TestWiring:
    def test_reranked_returns_reranked_mode(self, store: QdrantDenseRetriever) -> None:
        branch = RerankedRetriever(store, SpyReranker(["d"]))
        results = branch.retrieve(_QUERY, top_k=3)
        assert results
        assert all(hit.mode is RetrievalMode.RERANKED for hit in results)

    def test_reranker_overrides_the_base_ordering(self, store: QdrantDenseRetriever) -> None:
        """Phép so có nghĩa: nhánh nền xếp `a` trước, reranker kéo `d` lên đầu."""
        base_order = [hit.chunk.chunk_id for hit in store.retrieve(_QUERY, 6)]
        branch = RerankedRetriever(store, SpyReranker(["d", "e"]))
        reranked_order = [hit.chunk.chunk_id for hit in branch.retrieve(_QUERY, 6)]

        assert base_order != reranked_order
        assert reranked_order[:2] == ["doc-d::00000", "doc-e::00000"]

    def test_ranks_are_contiguous_from_one(self, store: QdrantDenseRetriever) -> None:
        branch = RerankedRetriever(store, SpyReranker(["c", "a"]))
        results = branch.retrieve(_QUERY, top_k=5)
        assert [hit.rank for hit in results] == list(range(1, len(results) + 1))

    def test_scores_are_non_increasing(self, store: QdrantDenseRetriever) -> None:
        branch = RerankedRetriever(store, SpyReranker(["f", "b", "d"]))
        scores = [hit.score for hit in branch.retrieve(_QUERY, top_k=6)]
        assert scores == sorted(scores, reverse=True)

    def test_build_branch_path_over_hybrid(self, store: QdrantDenseRetriever) -> None:
        """Đúng đường mà `make eval-retrieval MODE=reranked` đi, trừ model thật."""
        branch = build_branch(store, "reranked", base="hybrid", k=1, rerank_device="cpu")
        assert isinstance(branch, RerankedRetriever)
        # Thay reranker thật bằng bản giả: ở đây kiểm đường dây, không kiểm model.
        branch.reranker = SpyReranker(["e"])
        assert branch.retrieve(_QUERY, top_k=3)[0].chunk.chunk_id == "doc-e::00000"

    def test_deterministic_across_calls(self, store: QdrantDenseRetriever) -> None:
        branch = RerankedRetriever(store, SpyReranker(["b", "f"]))
        first = [hit.chunk.chunk_id for hit in branch.retrieve(_QUERY, 6)]
        for _ in range(5):
            assert [hit.chunk.chunk_id for hit in branch.retrieve(_QUERY, 6)] == first

    def test_filters_reach_qdrant_not_just_the_wrapper(self, store: QdrantDenseRetriever) -> None:
        """`W2-06` dựa vào chuyện này: filter phải lọc TRƯỚC khi vào pool.

        Lọc sau khi rerank thì cross-encoder đã đọc nội dung của tenant khác —
        tiền đã trả và dữ liệu đã bị chạm, dù kết quả cuối trông đúng.
        """
        reranker = SpyReranker(["g"])
        branch = RerankedRetriever(store, reranker)
        results = branch.retrieve(_QUERY, top_k=6, filters={"tenant_id": "t1"})
        assert "doc-g::00000" not in [hit.chunk.chunk_id for hit in results]
        assert _TEXTS["g"] not in reranker.seen[0]

    def test_the_reranker_sees_the_full_pool_not_just_top_k(
        self, store: QdrantDenseRetriever
    ) -> None:
        reranker = SpyReranker([])
        RerankedRetriever(store, reranker, candidates=7).retrieve(_QUERY, top_k=2)
        assert len(reranker.seen[0]) == 7

    def test_branch_scores_from_a_hybrid_base_survive(self, store: QdrantDenseRetriever) -> None:
        branch = build_branch(store, "reranked", base="hybrid", k=1, rerank_device="cpu")
        assert isinstance(branch, RerankedRetriever)
        branch.reranker = SpyReranker(["a"])
        top = branch.retrieve(_QUERY, top_k=3)[0]
        assert top.rerank_score is not None
        # Ít nhất một nhánh phải nhận là mình tìm ra nó, nếu không thì đóng góp
        # của từng nhánh đã bị đánh rơi ở đâu đó giữa fuse và rerank.
        assert top.dense_score is not None or top.sparse_score is not None


class TestPoolDepthReachesTheWire:
    """`rerank_candidates` có biến thành `limit` trong request Qdrant không.

    `W2-04` học được bài này bằng một test đỏ: `candidate_k=1` và `candidate_k=7`
    cho cùng kết quả vì `_depth` không bao giờ xuống dưới `top_k`, nên test "pool
    sâu hơn đổi thứ hạng" không đo cái nó tưởng. Cách chắc chắn là xem dây.
    """

    def test_candidates_becomes_the_dense_limit(
        self, store: QdrantDenseRetriever, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[int] = []
        original = store.client.query_points

        def spy(*args: Any, **kwargs: Any) -> Any:
            seen.append(int(kwargs["limit"]))
            return original(*args, **kwargs)

        monkeypatch.setattr(store.client, "query_points", spy)
        RerankedRetriever(store, SpyReranker([]), candidates=13).retrieve(_QUERY, top_k=3)
        assert seen == [13]

    def test_candidates_deepens_the_hybrid_fusion_pool_too(
        self, store: QdrantDenseRetriever, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """⚠️ Tương tác được ghi trong docstring của `_depth`, ở đây là bằng chứng.

        `candidate_k=2` nhưng `rerank_candidates=9` thì **cả hai** truy vấn của
        nhánh hybrid lấy 9, không phải 2 — tức pool hợp nhất bị làm sâu theo. Vô
        hại ở `k=1` (`W2-04` đo `candidate_k` không có tác dụng ở đó) nhưng không
        vô hại ở `k` lớn, nên nó phải là một sự thật được ghim chứ không phải một
        chi tiết bị phát hiện lại sau này.
        """
        limits: list[int] = []
        original = store.client.query_batch_points

        def spy(*args: Any, **kwargs: Any) -> Any:
            limits.extend(int(request.limit or 0) for request in kwargs["requests"])
            return original(*args, **kwargs)

        monkeypatch.setattr(store.client, "query_batch_points", spy)
        branch = build_branch(
            store, "reranked", base="hybrid", k=1, candidate_k=2, rerank_device="cpu"
        )
        assert isinstance(branch, RerankedRetriever)
        branch.reranker = SpyReranker([])
        branch.candidates = 9
        branch.retrieve(_QUERY, top_k=3)
        assert limits == [9, 9]


# --------------------------------------------------------------------------- #
# Cross-encoder thật. 2,2GB trọng số — `make test-gpu`.
# --------------------------------------------------------------------------- #

_PAIRS = [
    "Tổng sản phẩm trong nước (GDP) năm 2023 tăng 5,05% so với năm trước.",
    "Chỉ số giá tiêu dùng bình quân năm 2023 tăng 3,25% so với bình quân 2022.",
    "Chương trình tiêm chủng mở rộng cho trẻ dưới một tuổi ở các tỉnh vùng cao.",
    "Diện tích rừng trồng mới tập trung năm 2023 đạt 298,2 nghìn ha.",
    "Báo cáo tình hình kinh tế xã hội quý IV và năm 2023 của Tổng cục Thống kê.",
]
_GDP_QUERY = "Tăng trưởng GDP của Việt Nam năm 2023 là bao nhiêu phần trăm?"


@pytest.fixture(scope="module")
def gpu_reranker() -> CrossEncoderReranker:
    """Reranker như nó thật sự được phục vụ: `dtype="auto"` → **fp16** trên CUDA."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("cần GPU")
    return CrossEncoderReranker(device="cuda", batch_size=4)


@pytest.fixture(scope="module")
def fp32_reranker() -> CrossEncoderReranker:
    """Bản fp32 — chỉ dùng cho những test cần **số học xác định**.

    Cần fixture riêng vì mặc định fp16 làm hai test dưới mất nghĩa theo hai kiểu
    đối ngược: "batch == single" trở nên **sai** (padding khác thì tổng khác ở chữ
    số thấp), còn "max_length đổi điểm" trở nên **đúng một cách vô nghĩa** — nhiễu
    fp16 (~0,005) đã lớn hơn ngưỡng 1e-3 nên nó pass dù `max_length` có tới được
    model hay không. Một test pass vì lý do sai tệ hơn một test đỏ.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("cần GPU")
    return CrossEncoderReranker(device="cuda", batch_size=4, dtype="float32")


@pytest.mark.gpu
class TestRealCrossEncoder:
    def test_scores_come_back_one_per_text_in_order(
        self, gpu_reranker: CrossEncoderReranker
    ) -> None:
        scores = gpu_reranker.score(_GDP_QUERY, _PAIRS)
        assert len(scores) == len(_PAIRS)

    def test_it_puts_the_right_passage_first(self, gpu_reranker: CrossEncoderReranker) -> None:
        """Phép kiểm ngữ nghĩa duy nhất ở đây, và nó cố ý dễ.

        Câu hỏi về GDP, một đoạn nói đúng con số GDP, bốn đoạn khác chủ đề. Nếu
        model không thắng ca này thì nó bị nạp sai — đó là điều test này bắt.
        Còn "reranker có tốt hơn hybrid không" là câu hỏi của golden set 209 câu,
        không phải của năm đoạn tự chọn (bài học `W2-03`, nơi một corpus 7 chunk
        làm tôi tin sai về dense).
        """
        scores = gpu_reranker.score(_GDP_QUERY, _PAIRS)
        assert scores.index(max(scores)) == 0

    def test_batch_equals_single(self, fp32_reranker: CrossEncoderReranker) -> None:
        """DoD của `W2-05`. Padding trong batch làm sai điều này ở một số backend.

        Cùng hợp đồng với `EmbeddingProvider` (`W1-05`): `batch_size` chỉ được đổi
        tốc độ, không đổi số. Đây cũng là lý do `batch_size` không nằm trong
        `CrossEncoderReranker.name`.

        Đo trên **fp32**: hợp đồng này nói về cách gom batch, và kiểm nó ở fp16 là
        trộn hai câu hỏi (gom batch có đúng không / fp16 mất bao nhiêu chữ số) vào
        một assertion không tách được. Phần fp16 có test riêng ngay dưới.
        """
        batched = fp32_reranker.score(_GDP_QUERY, _PAIRS)
        one_at_a_time = [fp32_reranker.score(_GDP_QUERY, [text])[0] for text in _PAIRS]
        assert batched == pytest.approx(one_at_a_time, abs=1e-4)

    def test_batch_size_one_and_sixty_four_agree(self, fp32_reranker: CrossEncoderReranker) -> None:
        small = CrossEncoderReranker(device="cuda", batch_size=1, dtype="float32")
        large = CrossEncoderReranker(device="cuda", batch_size=64, dtype="float32")
        assert small.score(_GDP_QUERY, _PAIRS) == pytest.approx(
            large.score(_GDP_QUERY, _PAIRS), abs=1e-4
        )

    def test_fp16_keeps_the_order_even_though_the_numbers_move(
        self, gpu_reranker: CrossEncoderReranker, fp32_reranker: CrossEncoderReranker
    ) -> None:
        """Hợp đồng thật của mặc định fp16: **thứ hạng** giữ, chữ số thấp thì không.

        Đo được trên 60 câu × pool 50: trùng top-1 98,3%, lệch điểm max 0,0154 trên
        khoảng logit ~19,5 (0,08%). Ở đây chỉ có 5 đoạn nên nó **không** thay thế
        phép đo đó — nó canh việc mặc định fp16 không hỏng thầm lặng (NaN, inf,
        hoặc một thứ tự đảo lộn).
        """
        raw = fp32_reranker.score(_GDP_QUERY, _PAIRS)
        half = gpu_reranker.score(_GDP_QUERY, _PAIRS)
        assert all(math.isfinite(value) for value in half)
        assert half.index(max(half)) == raw.index(max(raw))
        assert half == pytest.approx(raw, abs=0.05)

    def test_sigmoid_is_monotone_so_the_order_is_identical(
        self, gpu_reranker: CrossEncoderReranker
    ) -> None:
        """Lý do mặc định là logit thô: sigmoid không thêm gì cho việc xếp hạng.

        Nếu test này đỏ thì hoặc sigmoid đã bão hoà (mất phân biệt), hoặc
        `activation` đang được áp hai lần ở đâu đó — cả hai đều là bug.
        """
        squashed = CrossEncoderReranker(device=gpu_reranker.device, activation="sigmoid")
        raw = gpu_reranker.score(_GDP_QUERY, _PAIRS)
        wrapped = squashed.score(_GDP_QUERY, _PAIRS)

        def order_of(values: Sequence[float]) -> list[int]:
            return sorted(range(len(values)), key=lambda index: -values[index])

        assert order_of(raw) == order_of(wrapped)
        assert all(0.0 <= value <= 1.0 for value in wrapped)

    def test_deterministic_across_calls(self, gpu_reranker: CrossEncoderReranker) -> None:
        first = gpu_reranker.score(_GDP_QUERY, _PAIRS)
        for _ in range(3):
            assert gpu_reranker.score(_GDP_QUERY, _PAIRS) == pytest.approx(first, abs=1e-5)

    def test_count_pair_tokens_does_not_truncate(self, gpu_reranker: CrossEncoderReranker) -> None:
        """Nếu nó cắt thì phép đo truncation trả về hằng số — bài học `TD-11`."""
        long_text = "ngân sách nhà nước đầu tư công " * 400
        counts = gpu_reranker.count_pair_tokens(_GDP_QUERY, [long_text])
        assert counts[0] > gpu_reranker.max_length

    def test_max_length_actually_changes_the_score(
        self, fp32_reranker: CrossEncoderReranker
    ) -> None:
        """Bằng chứng cho việc `max_length` phải nằm trong `name`.

        Cắt ở 32 token thì model đọc một cặp khác hẳn cắt ở 512, nên điểm phải
        khác. Nếu nó không khác thì `max_length` không tới được model và nhãn
        đang nói dối về cấu hình.

        Ngưỡng **0,5** chứ không phải 1e-3, và đo trên fp32: cắt bỏ 90% một đoạn
        phải đổi điểm *nhiều*, nên một ngưỡng chặt chỉ khiến test pass nhờ nhiễu số
        học thay vì nhờ điều nó muốn chứng minh.

        ⚠️ Bản `max_length=32` chạy trên **CPU**, và đó là một quyết định về ngân
        sách VRAM, không phải tuỳ tiện. `max_length` nằm trong khoá cache của
        `_load_cross_encoder`, nên nó là một **model thứ ba** trong cùng phiên
        pytest: bge-m3 3,3GB + fp16 1,15GB + fp32 2,3GB + bản này 2,3GB = 9,05GB
        trên một GPU 8,0GB. Lần chạy đầu của file này **OOM đúng vì vậy**. Điều
        test này khẳng định — `max_length` có tới được model không — hoàn toàn
        không phụ thuộc device, và ngưỡng 0,5 cách xa nhiễu fp32 giữa hai device
        (~1e-4) nhiều bậc. Chạy nó trên CPU đưa nó ra khỏi ngân sách VRAM hẳn.
        """
        short = CrossEncoderReranker(device="cpu", max_length=32, dtype="float32")
        deltas = [
            abs(a - b)
            for a, b in zip(
                short.score(_GDP_QUERY, _PAIRS),
                fp32_reranker.score(_GDP_QUERY, _PAIRS),
                strict=True,
            )
        ]
        assert max(deltas) > 0.5
