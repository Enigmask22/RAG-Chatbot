"""Test cho `rag_core.embedding.truncation` — phép đo của `TD-11`.

Phép đo này canh một lỗi **không có triệu chứng**, nên bản thân nó cũng có thể
sai mà không ai biết. Ba bất biến quan trọng nhất mà file này giữ:

1. Text **đúng bằng** giới hạn thì KHÔNG bị cắt (lỗi off-by-one ở đây làm mọi
   con số lệch theo hướng bi quan, và sẽ bị đọc thành "đã sửa xong TD-11").
2. `truncated_ratio` và `tokens_lost_ratio` là **hai** đại lượng khác nhau. Gộp
   chúng lại là mất đúng thông tin cần để quyết định.
3. `None` = "không đo được", không bao giờ được biến thành `0`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import FrozenInstanceError

import pytest

from pipeline.indexing.build_index import BuildReport
from rag_core.embedding.hashing import HashingEmbeddingProvider
from rag_core.embedding.truncation import (
    TruncationStats,
    mean_tokens_per_char,
    measure_truncation,
    token_stats,
)
from rag_core.schemas import Chunk


class TestTokenStats:
    def test_counts_by_hand(self) -> None:
        """Số tính tay: 4 text, giới hạn 10, hai text vượt."""
        stats = token_stats([5, 10, 12, 20], limit=10)
        assert stats.n_texts == 4
        assert stats.n_truncated == 2, "12 và 20 vượt, 10 thì không"
        assert stats.tokens_total == 47
        assert stats.tokens_kept == 5 + 10 + 10 + 10

    def test_exactly_at_limit_is_not_truncated(self) -> None:
        """Bất biến 1. `>` chứ không phải `>=` — model đọc trọn 256 token."""
        stats = token_stats([256, 256, 256], limit=256)
        assert stats.n_truncated == 0
        assert stats.tokens_lost_ratio == 0.0
        assert stats.is_healthy

    def test_one_over_the_limit_is_truncated(self) -> None:
        stats = token_stats([257], limit=256)
        assert stats.n_truncated == 1
        assert stats.tokens_kept == 256

    def test_two_ratios_tell_different_stories(self) -> None:
        """Bất biến 2 — đây là lý do có hai property thay vì một.

        Trường hợp A: mọi chunk bị cắt, mỗi chunk mất 1 token → 100% chunk bị
        cắt nhưng gần như không mất nội dung.
        Trường hợp B: một chunk bị cắt mất một nửa → 10% chunk nhưng mất nhiều
        nội dung hơn hẳn tính trên mỗi chunk bị ảnh hưởng.
        """
        a = token_stats([101] * 10, limit=100)
        b = token_stats([200] + [50] * 9, limit=100)

        assert a.truncated_ratio == 1.0
        assert a.tokens_lost_ratio < 0.01

        assert b.truncated_ratio == pytest.approx(0.1)
        assert b.tokens_lost_ratio > a.tokens_lost_ratio

    def test_tokens_lost_ratio_by_hand(self) -> None:
        # tổng 300, giữ 200 → mất 1/3
        stats = token_stats([200, 100], limit=100)
        assert stats.tokens_kept == 200
        assert stats.tokens_lost_ratio == pytest.approx(1 / 3)

    def test_percentiles(self) -> None:
        """Quy ước nearest-rank, **giống hệt** `build_index._percentile`.

        Không phải quy ước duy nhất đúng, nhưng phải là quy ước duy nhất trong
        repo: hai công thức phân vị khác nhau trong cùng một báo cáo sẽ cho hai
        con số p95 lệch nhau mà không ai truy được vì sao.
        """
        stats = token_stats(list(range(1, 101)), limit=1000)
        assert stats.token_p50 == 51
        assert stats.token_p95 == 95
        assert stats.token_max == 100

    def test_percentile_matches_build_index(self) -> None:
        from pipeline.indexing.build_index import _percentile

        counts = [3, 1, 4, 1, 5, 9, 2, 6]
        stats = token_stats(counts, limit=99)
        assert stats.token_p50 == _percentile([float(c) for c in counts], 50)
        assert stats.token_p95 == _percentile([float(c) for c in counts], 95)

    def test_empty_input_is_not_an_error(self) -> None:
        stats = token_stats([], limit=256)
        assert stats.n_texts == 0
        assert stats.truncated_ratio == 0.0
        assert stats.tokens_lost_ratio == 0.0
        assert stats.token_max == 0

    @pytest.mark.parametrize("limit", [0, -1])
    def test_rejects_nonpositive_limit(self, limit: int) -> None:
        with pytest.raises(ValueError, match="phải dương"):
            token_stats([1, 2], limit=limit)

    def test_tokens_per_char_needs_chars_total(self) -> None:
        assert token_stats([100], limit=256).tokens_per_char == 0.0
        assert token_stats([100], limit=256, chars=[400]).tokens_per_char == 0.25

    def test_as_dict_has_every_number_the_report_prints(self) -> None:
        keys = set(token_stats([10], limit=20, chars=[40]).as_dict())
        assert keys == {
            "limit",
            "n_texts",
            "n_truncated",
            "truncated_ratio",
            "tokens_total",
            "tokens_kept",
            "chars_total",
            "tokens_lost_ratio",
            "tokens_per_char",
            "chars_per_token_p05",
            "char_budget",
            "token_p50",
            "token_p95",
            "token_max",
        }

    def test_summary_mentions_both_ratios(self) -> None:
        text = token_stats([300, 100], limit=256).summary()
        assert "1/2" in text
        assert "%" in text

    def test_summary_on_empty(self) -> None:
        assert "không có text" in token_stats([], limit=8).summary()

    def test_frozen(self) -> None:
        stats = token_stats([1], limit=2)
        with pytest.raises(FrozenInstanceError):
            stats.limit = 99  # type: ignore[misc]

    def test_accepts_any_sequence(self) -> None:
        assert token_stats((5, 5), limit=4).n_truncated == 2


class TestCharBudget:
    """Hiệu chuẩn `chunk_size` — chỗ tôi đã viết SAI ở lần đầu.

    Bản đầu dùng token/ký tự **trung bình**, và nó trả lời sai câu hỏi: nó cho
    ra `chunk_size` mà chunk *trung bình* vừa khít cửa sổ, tức ngưỡng mà một nửa
    số chunk vẫn bị cắt. Trên corpus thật nó gợi ý 946 ký tự trong khi ở 1000 ký
    tự đã có 56,9% chunk bị cắt — một con số vô lý mà trông hoàn toàn hợp lý.
    """

    def test_uses_the_dense_tail_not_the_mean(self) -> None:
        # nửa dày (2 ký tự/token, ví dụ tiếng Anh bị PhoBERT xé vụn),
        # nửa thưa (10 ký tự/token). Trung bình 6 — không mô tả nửa nào cả.
        counts = [50] * 50 + [50] * 50
        chars = [100] * 50 + [500] * 50
        stats = token_stats(counts, limit=100, chars=chars)

        assert stats.chars_per_token_p05 == pytest.approx(2.0)
        assert stats.char_budget(special_tokens=2) == 196

        mean_density = sum(chars) / sum(counts)
        assert mean_density == pytest.approx(6.0)
        naive = int((100 - 2) * mean_density)
        assert stats.char_budget(special_tokens=2) < naive / 2, (
            "ngân sách theo trung bình rộng gấp 3 — đó là bug đã sửa"
        )

    def test_budget_is_zero_without_char_lengths(self) -> None:
        """Không có độ dài ký tự thì không suy được gì — trả 0, không đoán."""
        assert token_stats([10, 20], limit=256).char_budget() == 0

    def test_mismatched_lengths_is_an_error(self) -> None:
        """Lệch độ dài = mật độ tính trên hai tập text khác nhau."""
        with pytest.raises(ValueError, match="cùng độ dài"):
            token_stats([1, 2, 3], limit=10, chars=[10, 20])

    def test_zero_token_chunks_do_not_divide_by_zero(self) -> None:
        stats = token_stats([0, 10], limit=10, chars=[0, 100])
        assert stats.chars_per_token_p05 == pytest.approx(10.0)

    def test_special_tokens_are_subtracted(self) -> None:
        stats = token_stats([10], limit=100, chars=[100])
        assert stats.char_budget(special_tokens=0) > stats.char_budget(special_tokens=2)


class TestMeanTokensPerChar:
    def test_weighted_not_averaged_per_chunk(self) -> None:
        """Trung bình có trọng số, không phải trung bình của các tỉ lệ.

        Chunk ngắn có tỉ lệ token/ký tự lệch cao (special token chiếm phần lớn).
        Lấy trung bình các tỉ lệ sẽ để chunk ngắn quyết định con số, và
        `chunk_size` gợi ý ra nhỏ hơn thực tế cần.
        """
        counts = [10, 100]
        lengths = [10, 400]
        assert mean_tokens_per_char(counts, lengths) == pytest.approx(110 / 410)
        naive = (10 / 10 + 100 / 400) / 2
        assert mean_tokens_per_char(counts, lengths) < naive

    def test_zero_chars(self) -> None:
        assert mean_tokens_per_char([], []) == 0.0


class TestMeasureTruncation:
    def test_returns_none_when_provider_has_no_limit(self) -> None:
        """Bất biến 3: provider không biết giới hạn → `None`, không phải "sạch"."""
        provider = HashingEmbeddingProvider(dimension=32)
        assert provider.max_sequence_tokens is None
        assert measure_truncation(provider, ["a", "b"]) is None

    def test_base_provider_counts_nothing_by_default(self) -> None:
        provider = HashingEmbeddingProvider(dimension=32)
        assert provider.count_tokens(["a"]) is None

    def test_uses_provider_tokenizer(self) -> None:
        stats = measure_truncation(_FakeProvider(limit=5), ["a b c", "a b c d e f g"])
        assert isinstance(stats, TruncationStats)
        assert stats.limit == 5
        assert stats.n_truncated == 1

    def test_batching_does_not_change_the_result(self) -> None:
        texts = [" ".join(["w"] * n) for n in range(1, 40)]
        provider = _FakeProvider(limit=10)
        one = measure_truncation(provider, texts, batch_size=1)
        big = measure_truncation(provider, texts, batch_size=1000)
        assert one == big

    def test_chars_total_comes_from_the_texts(self) -> None:
        stats = measure_truncation(_FakeProvider(limit=100), ["abcd", "ef"])
        assert stats is not None
        assert stats.chars_total == 6

    def test_provider_that_stops_counting_gives_none(self) -> None:
        """Đếm được nửa đường rồi thôi thì phải bỏ cả phép đo.

        Thống kê trên một tập con không rõ là tập nào còn tệ hơn không có
        thống kê — nó vẫn trông như một con số.
        """
        assert measure_truncation(_FlakyProvider(), ["a"] * 10, batch_size=2) is None

    def test_empty_texts(self) -> None:
        stats = measure_truncation(_FakeProvider(limit=8), [])
        assert stats is not None
        assert stats.n_texts == 0


class TestBuildIndexGlue:
    """Phần nối vào `build_index` — nơi con số này thật sự phải xuất hiện.

    Đo được mà không in ra thì `TD-11` vẫn trốn được lần nữa, nên ba đường log
    phải phân biệt rõ ba trạng thái khác nhau: có số, không đo được, và không
    có chunk mới để đo.
    """

    def test_accumulates_across_batches(self) -> None:
        from pipeline.indexing.build_index import _accumulate_tokens

        provider = _FakeProvider(limit=8)
        counts = _accumulate_tokens([], provider, [_chunk("a b"), _chunk("c d e")])
        assert counts == [4, 5]
        counts = _accumulate_tokens(counts, provider, [_chunk("f")])
        assert counts == [4, 5, 3]

    def test_none_is_sticky(self) -> None:
        """Một lần không đếm được là bỏ cả phép đo, không đếm tiếp phần còn lại."""
        from pipeline.indexing.build_index import _accumulate_tokens

        assert _accumulate_tokens(None, _FakeProvider(limit=8), [_chunk("a")]) is None

    def test_provider_without_limit_never_starts_counting(self) -> None:
        from pipeline.indexing.build_index import _accumulate_tokens

        provider = HashingEmbeddingProvider(dimension=32)
        assert _accumulate_tokens([], provider, [_chunk("a")]) is None

    def test_warns_when_chunks_were_truncated(self, caplog: pytest.LogCaptureFixture) -> None:
        """WARNING, không phải INFO — xem docstring của `_log_truncation`."""
        report = _report(truncation=token_stats([300, 100], limit=256, chars=[900, 300]).as_dict())
        with caplog.at_level("WARNING", logger="pipeline.indexing"):
            report.log_summary()
        assert "cắt token" in caplog.text

    def test_no_warning_when_everything_fits(self, caplog: pytest.LogCaptureFixture) -> None:
        report = _report(truncation=token_stats([100], limit=256, chars=[300]).as_dict())
        with caplog.at_level("WARNING", logger="pipeline.indexing"):
            report.log_summary()
        assert caplog.text == "", "chunk vừa cửa sổ thì không có gì phải cảnh báo"

    def test_unmeasurable_is_not_reported_as_clean(self, caplog: pytest.LogCaptureFixture) -> None:
        """Bất biến 3 ở tầng log: 'không biết' phải đọc ra được là 'không biết'."""
        report = _report(truncation={}, n_chunks_written=10)
        with caplog.at_level("WARNING", logger="pipeline.indexing"):
            report.log_summary()
        assert "KHÔNG đo được" in caplog.text
        assert "không phải" in caplog.text

    def test_nothing_written_says_so(self, caplog: pytest.LogCaptureFixture) -> None:
        """Chạy lại khi cache đủ: không có chunk mới, khác hẳn 'không đo được'."""
        report = _report(truncation={}, n_chunks_written=0)
        with caplog.at_level("WARNING", logger="pipeline.indexing"):
            report.log_summary()
        assert caplog.text == ""

    def test_truncation_survives_json_round_trip(self) -> None:
        import json

        report = _report(truncation=token_stats([300], limit=256, chars=[900]).as_dict())
        payload = json.loads(report.to_json())
        assert payload["truncation"]["tokens_lost_ratio"] > 0


def _chunk(text: str) -> Chunk:
    return Chunk(chunk_id="d::00000", doc_id="d", content=text, chunk_index=0)


def _report(*, truncation: dict[str, float], n_chunks_written: int = 2) -> BuildReport:
    return BuildReport(
        config_name="t",
        collection="c",
        fingerprint="f" * 64,
        embedding_model="m",
        embedding_device="cpu",
        embedding_dim=8,
        chunker_name="ch",
        n_documents=1,
        n_documents_indexed=1,
        n_documents_skipped=0,
        n_documents_removed=0,
        n_chunks_written=n_chunks_written,
        n_stale_points_deleted=0,
        collection_count=n_chunks_written,
        chars_in=1000,
        chars_out=1200,
        truncation=truncation,
    )


class _FakeProvider(HashingEmbeddingProvider):
    """Provider có giới hạn token, đếm token = số từ + 2 special token."""

    def __init__(self, *, limit: int) -> None:
        super().__init__(dimension=32)
        self._limit = limit

    @property
    def max_sequence_tokens(self) -> int | None:
        return self._limit

    def count_tokens(self, texts: Sequence[str]) -> list[int] | None:
        return [len(t.split()) + 2 for t in texts]


class _FlakyProvider(_FakeProvider):
    """Đếm được lô đầu rồi trả `None` — mô phỏng tokenizer chết giữa job."""

    def __init__(self) -> None:
        super().__init__(limit=8)
        self._calls = 0

    def count_tokens(self, texts: Sequence[str]) -> list[int] | None:
        self._calls += 1
        if self._calls > 1:
            return None
        return super().count_tokens(texts)
