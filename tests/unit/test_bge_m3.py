"""Test cho `rag_core.embedding.bge_m3` — `W2-01`.

Chia hai tầng có chủ ý:

* **Không cần model** — `_collapse` và `_plan_batches` là logic thuần, chạy trong
  `make test` (~ms). `device="cpu"` để `resolve_device` trả về ngay, không nạp
  `torch`, giữ vòng lặp phát triển ở vài giây.
* **Cần model + GPU** (`@pytest.mark.gpu`, `make test-gpu`) — 2,2GB trọng số.
  Bài quan trọng nhất ở đây là **dense phải khớp `SentenceTransformer.encode()`**:
  `_forward` cố ý không đi qua `model.encode()` để dense và sparse ra từ một
  forward pass, nên phải có gì đó canh rằng nó không lệch khỏi đường chuẩn.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag_core.embedding.bge_m3 import BGE_M3_MODEL, BgeM3EmbeddingProvider, _collapse
from rag_core.embedding.sparse import SparseVector

# `device="cpu"` (không phải "auto") là điều kiện để dựng provider mà không
# import `torch` — `resolve_device` trả về ngay ở nhánh đầu.
CPU_KWARGS = {"device": "cpu"}


def _provider(**kwargs: object) -> BgeM3EmbeddingProvider:
    return BgeM3EmbeddingProvider(**{**CPU_KWARGS, **kwargs})  # type: ignore[arg-type]


class TestCollapse:
    """`_collapse`: `(input_ids, attention_mask, weights)` → `SparseVector`."""

    @staticmethod
    def _call(
        ids: list[int], mask: list[int], weights: list[float], unused: set[int]
    ) -> SparseVector:
        return _collapse(
            np.array(ids), np.array(mask), np.array(weights, dtype=np.float32), frozenset(unused)
        )

    def test_takes_max_not_sum_for_repeated_token(self) -> None:
        """Định nghĩa của BGE-M3 là max, và nó có nghĩa.

        Trọng số trả lời "token này quan trọng thế nào cho đoạn text", không phải
        "nó xuất hiện bao nhiêu lần". Đổi sang sum thì token lặp nhiều lấn át và
        sparse retrieval biến thành đếm tần suất thô.
        """
        vec = self._call([7, 7, 7], [1, 1, 1], [0.2, 0.9, 0.4], unused=set())
        assert vec.as_dict() == {7: pytest.approx(0.9)}

    def test_special_tokens_are_dropped(self) -> None:
        """`[CLS]` thường là chiều nặng nhất — để lại thì mọi cặp text khớp nhau ở
        đúng chiều đó và điểm sparse gần thành hằng số."""
        vec = self._call([0, 42, 2], [1, 1, 1], [0.99, 0.3, 0.88], unused={0, 2})
        assert vec.as_dict() == {42: pytest.approx(0.3)}

    def test_padding_is_dropped_even_with_positive_weight(self) -> None:
        """Vị trí bị mask vẫn qua matmul nên vẫn có trọng số dương — phải bỏ theo
        `attention_mask`, không thể dựa vào trọng số bằng 0."""
        vec = self._call([5, 9], [1, 0], [0.4, 0.7], unused=set())
        assert vec.as_dict() == {5: pytest.approx(0.4)}

    def test_non_positive_weights_are_dropped(self) -> None:
        vec = self._call([1, 2, 3], [1, 1, 1], [0.0, -0.5, 0.2], unused=set())
        assert vec.indices == (3,)

    def test_all_dropped_gives_empty_vector_not_none(self) -> None:
        """Text chỉ gồm special token: rỗng là kết quả hợp lệ, không phải `None`."""
        vec = self._call([0, 2], [1, 1], [0.9, 0.9], unused={0, 2})
        assert isinstance(vec, SparseVector)
        assert len(vec) == 0

    def test_result_is_sorted_by_token_id(self) -> None:
        vec = self._call([90, 10, 50], [1, 1, 1], [0.1, 0.2, 0.3], unused=set())
        assert vec.indices == (10, 50, 90)

    def test_mismatched_lengths_raise(self) -> None:
        """`zip(strict=True)` — lệch độ dài nghĩa là tokenizer và forward pass
        không nói về cùng một batch."""
        with pytest.raises(ValueError):
            self._call([1, 2], [1], [0.5, 0.5], unused=set())


class TestPlanBatches:
    """Chia batch — bất biến mà `_forward` dựa vào để không trả thiếu hàng."""

    def test_every_index_appears_exactly_once(self) -> None:
        lengths = [10, 500, 3, 8000, 42, 42, 1]
        batches = _provider(batch_size=2, max_batch_tokens=8192)._plan_batches(lengths, 8192)
        flat = [i for b in batches for i in b]
        assert sorted(flat) == list(range(len(lengths)))
        assert len(flat) == len(set(flat))

    def test_respects_batch_size(self) -> None:
        batches = _provider(batch_size=3, max_batch_tokens=10**9)._plan_batches([5] * 10, 8192)
        assert [len(b) for b in batches] == [3, 3, 3, 1]

    def test_respects_token_budget(self) -> None:
        """`batch_size` một mình không chặn được gì: cửa sổ 8192 nghĩa là một
        batch 16 câu dài 8192 token là 131k token và OOM ngay."""
        batches = _provider(batch_size=16, max_batch_tokens=1000)._plan_batches([500] * 6, 8192)
        assert [len(b) for b in batches] == [2, 2, 2]

    def test_longest_sequence_lands_in_the_first_batch(self) -> None:
        """Cấu hình sẽ OOM thì phải OOM ở giây thứ nhất, không phải ở phút thứ ba
        của một job 31.000 chunk."""
        lengths = [10, 20, 7000, 15]
        batches = _provider(batch_size=1)._plan_batches(lengths, 8192)
        assert batches[0] == [2]

    def test_sorted_descending_by_length(self) -> None:
        lengths = [10, 900, 50, 300]
        batches = _provider(batch_size=1)._plan_batches(lengths, 8192)
        assert [b[0] for b in batches] == [1, 3, 2, 0]

    def test_length_is_clamped_to_the_window(self) -> None:
        """Text 40.000 token vẫn chỉ tốn 8192 sau `truncation=True`; không clamp
        thì ngân sách tưởng đã vượt và mỗi batch còn đúng một câu."""
        batches = _provider(batch_size=8, max_batch_tokens=8192)._plan_batches([40_000] * 2, 8192)
        assert [len(b) for b in batches] == [1, 1]
        batches = _provider(batch_size=8, max_batch_tokens=16_384)._plan_batches([40_000] * 2, 8192)
        assert [len(b) for b in batches] == [2]

    def test_short_sequences_still_get_full_batches(self) -> None:
        """Điều làm trần token đáng có: chunk ngắn không bị phạt."""
        batches = _provider(batch_size=64, max_batch_tokens=8192)._plan_batches([40] * 64, 8192)
        assert len(batches) == 1

    def test_empty_input(self) -> None:
        assert _provider()._plan_batches([], 8192) == []

    def test_zero_length_text_does_not_divide_by_zero(self) -> None:
        batches = _provider(batch_size=4)._plan_batches([0, 0, 0], 8192)
        assert sorted(i for b in batches for i in b) == [0, 1, 2]


class TestConfiguration:
    def test_rejects_non_positive_token_budget(self) -> None:
        with pytest.raises(ValueError, match="max_batch_tokens phải dương"):
            _provider(max_batch_tokens=0)

    def test_no_instruction_prefix(self) -> None:
        """BGE-M3 **không** dùng instruction prefix, khác BGE v1.5 và E5. Thêm
        prefix là làm lệch phân bố đầu vào so với lúc train, không báo lỗi."""
        p = _provider()
        assert p.query_prefix == ""
        assert p.document_prefix == ""

    def test_sparse_head_defaults_to_the_model_repo(self) -> None:
        assert _provider().sparse_head_name == BGE_M3_MODEL

    def test_sparse_head_is_overridable(self) -> None:
        """`W2-08` cần thử checkpoint fine-tune mà không phải fork provider."""
        assert _provider(sparse_head_name="me/my-m3").sparse_head_name == "me/my-m3"

    def test_factory_picks_this_provider_by_model_name(self) -> None:
        """Chọn theo tên model, không theo cờ riêng — để không tồn tại được cấu
        hình `model=bge-m3, use_bge_m3=false` vừa hợp lệ vừa vô nghĩa."""
        from rag_core.embedding import build_embedding_provider

        provider = build_embedding_provider(BGE_M3_MODEL, device="cpu")
        assert isinstance(provider, BgeM3EmbeddingProvider)

    def test_factory_still_returns_dense_only_for_other_models(self) -> None:
        from rag_core.embedding import build_embedding_provider
        from rag_core.embedding.huggingface import HuggingFaceEmbeddingProvider

        provider = build_embedding_provider(
            "bkai-foundation-models/vietnamese-bi-encoder", device="cpu"
        )
        assert isinstance(provider, HuggingFaceEmbeddingProvider)
        assert not isinstance(provider, BgeM3EmbeddingProvider)


class TestBaseProviderDefaults:
    """Năng lực sparse là tuỳ chọn: provider dense-only phải trả `None`."""

    def test_dense_only_provider_reports_no_sparse(self) -> None:
        from rag_core.embedding import HashingEmbeddingProvider

        p = HashingEmbeddingProvider(dimension=64)
        assert p.sparse_vocab_size is None
        assert p.embed_documents_hybrid(["a"]) is None
        assert p.embed_query_hybrid("a") is None


# --------------------------------------------------------------------------
# Cần model thật (2,2GB) + GPU. `make test-gpu`.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gpu_provider() -> BgeM3EmbeddingProvider:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("cần GPU")
    return BgeM3EmbeddingProvider(device="cuda", batch_size=8)


VI = "Tăng trưởng GDP của Việt Nam năm 2023 đạt 5,05%."
EN = "Vietnam's GDP growth reached 5.05 percent in 2023."
OTHER = "Lạm phát được kiểm soát ở mức 3,25%."


@pytest.mark.gpu
class TestRealModel:
    def test_window_is_8192(self, gpu_provider: BgeM3EmbeddingProvider) -> None:
        """Lý do tồn tại của `W2-01`: cửa sổ 8192 xoá truncation mà không phải
        hạ `chunk_size` (`TD-11` đã đo hạ `chunk_size` là đánh đổi)."""
        assert gpu_provider.max_sequence_tokens == 8192

    def test_sparse_vocab_size_is_memoised(self, gpu_provider: BgeM3EmbeddingProvider) -> None:
        """Hồi quy cho một bug hiệu năng tìm ra ở `W2-04`, và cách tìm ra nó đáng
        ghi lại: phân rã độ trễ của `retrieve_sparse` thì **81,7 ms không thuộc
        thành phần nào** — không phải embed, không phải Qdrant, không phải dựng
        `Chunk`. Chỗ thiếu đó là đây.

        `len(tokenizer)` gọi `get_vocab()`, tức dựng lại dict 250.002 phần tử, mất
        **64 ms**. Thuộc tính này bị đọc qua `QdrantDenseRetriever.writes_sparse` ở
        **mỗi** truy vấn sparse và **mỗi lô** upsert (124 lô × 64 ms ≈ 8 giây cho
        một lần build index).

        Canh bằng thời gian chứ không bằng số lần gọi: cái phải giữ là "đọc thuộc
        tính này rẻ", và một bản cài khác vẫn có thể vi phạm nó theo cách khác.
        """
        import time

        assert gpu_provider.sparse_vocab_size == 250_002  # lần đầu được phép đắt
        started = time.perf_counter()
        for _ in range(50):
            assert gpu_provider.sparse_vocab_size == 250_002
        per_call_ms = (time.perf_counter() - started) * 1000.0 / 50
        assert per_call_ms < 1.0, f"{per_call_ms:.1f} ms mỗi lần đọc — chưa nhớ kết quả"

    def test_query_hybrid_runs_exactly_one_forward_pass(
        self, gpu_provider: BgeM3EmbeddingProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tuyên bố mà `W2-04` dựa vào, canh ở đúng tầng nó thuộc về.

        `QdrantHybridRetriever` gọi `embed_query_hybrid` một lần thay vì gọi
        `retrieve()` + `retrieve_sparse()`, và cả lý lẽ đó chỉ đúng nếu **provider**
        cũng chỉ chạy một forward pass. Đo được ở `W2-03` §8: embed một truy vấn
        tốn 12,6 ms, tức chạy hai lần là +12,6 ms cho đúng một kết quả.

        Bản đầu của test tương ứng ở tầng integration đếm cả `embed_query` và đỏ —
        vì `HashingEmbeddingProvider` gọi lại nó bên trong. Nội bộ provider là hợp
        đồng của provider, và đây là chỗ kiểm nó.
        """
        batches: list[int] = []
        original = gpu_provider._forward

        def counted(texts: object) -> object:
            batches.append(len(texts))  # type: ignore[arg-type]
            return original(texts)  # type: ignore[arg-type]

        monkeypatch.setattr(gpu_provider, "_forward", counted)
        gpu_provider.embed_query_hybrid("Tăng trưởng GDP của Việt Nam năm 2023")
        assert batches == [1], f"chạy {len(batches)} forward pass: {batches}"

    def test_dimensions(self, gpu_provider: BgeM3EmbeddingProvider) -> None:
        assert gpu_provider.dimension == 1024
        assert gpu_provider.sparse_vocab_size == 250_002

    def test_dense_matches_sentence_transformers(
        self, gpu_provider: BgeM3EmbeddingProvider
    ) -> None:
        """Bài quan trọng nhất của module.

        `_forward` cố ý không gọi `model.encode()` — dense và sparse phải ra từ
        cùng một forward pass. Cái giá là dense giờ có đường code riêng, và hai
        đường sinh dense song song là cách chắc chắn để hai nhánh ablation vô
        tình đo hai thứ khác nhau. Test này là thứ chặn điều đó.
        """
        texts = [VI, EN, OTHER, "The quick brown fox. " * 40]
        ours = gpu_provider.embed_documents(texts)
        reference = gpu_provider.model.encode(
            texts, batch_size=8, convert_to_numpy=True, normalize_embeddings=True
        )
        assert np.abs(ours - reference).max() < 1e-5

    def test_dense_is_normalized(self, gpu_provider: BgeM3EmbeddingProvider) -> None:
        norms = np.linalg.norm(gpu_provider.embed_documents([VI, EN]), axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_order_is_preserved_through_length_sorted_batching(
        self, gpu_provider: BgeM3EmbeddingProvider
    ) -> None:
        """`_plan_batches` sắp lại theo độ dài; sai phép sắp ngược thì embedding
        gán cho sai chunk và không có gì báo lỗi."""
        texts = ["ngắn", VI * 20, "cũng ngắn", EN * 5]
        batched = gpu_provider.embed_documents_hybrid(texts)
        for i, text in enumerate(texts):
            single = gpu_provider.embed_documents_hybrid([text])
            assert np.abs(batched.dense[i] - single.dense[0]).max() < 1e-5
            assert batched.sparse[i].indices == single.sparse[0].indices

    def test_sparse_is_not_empty_for_vi_and_en(self, gpu_provider: BgeM3EmbeddingProvider) -> None:
        result = gpu_provider.embed_documents_hybrid([VI, EN])
        assert all(len(s) > 0 for s in result.sparse)

    def test_sparse_picks_content_words(self, gpu_provider: BgeM3EmbeddingProvider) -> None:
        """Sparse phải nặng ở từ mang nội dung, không ở hư từ — nếu không thì nó
        không thêm gì cho dense và `W2-04` không có gì để hợp nhất."""
        _, sparse = gpu_provider.embed_query_hybrid(VI)
        top = [gpu_provider.model.tokenizer.decode([tid]) for tid, _ in sparse.top(6)]
        assert "2023" in top
        assert any("GDP" in t for t in top)

    def test_sparse_scores_same_meaning_above_different_meaning(
        self, gpu_provider: BgeM3EmbeddingProvider
    ) -> None:
        """Dấu hiệu cho `cross_lingual` = 0 của baseline: vocab dùng chung của
        BGE-M3 làm "GDP"/"2023" khớp được qua hai ngôn ngữ."""
        result = gpu_provider.embed_documents_hybrid([VI, EN, OTHER])
        vi, en, other = result.sparse
        assert vi.dot(en) > vi.dot(other) * 5

    def test_same_input_gives_bit_identical_output(
        self, gpu_provider: BgeM3EmbeddingProvider
    ) -> None:
        """Tính xác định của đường index — đã xác nhận ba lần ở W1, giữ tiếp."""
        texts = [VI, EN, OTHER]
        first = gpu_provider.embed_documents_hybrid(texts)
        second = gpu_provider.embed_documents_hybrid(texts)
        assert np.array_equal(first.dense, second.dense)
        assert first.sparse == second.sparse

    def test_no_truncation_at_baseline_chunk_size(
        self, gpu_provider: BgeM3EmbeddingProvider
    ) -> None:
        """`chunk_size=1000` ký tự ở mật độ xấu nhất đo được (~0,25 token/ký tự
        cho tiếng Anh) vẫn còn cách 8192 rất xa."""
        counts = gpu_provider.count_tokens(["a b " * 250])
        assert counts is not None
        assert max(counts) < 8192

    def test_empty_input(self, gpu_provider: BgeM3EmbeddingProvider) -> None:
        result = gpu_provider.embed_documents_hybrid([])
        assert result.dense.shape == (0, 1024)
        assert result.sparse == []
