"""Đo phần văn bản bị model embedding cắt bỏ — nợ kỹ thuật `TD-11`.

Vì sao cần một module riêng cho một phép chia:

`sentence-transformers` cắt input ở `max_seq_length` **không cảnh báo, không
lỗi**. Chunk dài 340 token vào một model 256 token thì 84 token cuối đơn giản
không tồn tại đối với vector — và mọi con số trong `BuildReport` vẫn đẹp: đủ
chunk, đủ chiều, đủ tài liệu. Ở baseline `W1-13`, 56,8% chunk bị cắt và 15,7%
toàn bộ văn bản đem embed không bao giờ tới được vector, mà không một dòng log
nào lộ ra.

Bài học đóng vào đây: **một lỗi không có triệu chứng thì phải tự tạo triệu
chứng cho nó**. Nên phép đo này không phải script phân tích một lần rồi bỏ —
nó chạy trong mỗi lần build index và nằm trong report JSON, để mọi cấu hình
W2/W3 về sau đều mang theo con số này.

Hai điểm dễ đo sai:

* **Ngưỡng so sánh phải tính cả special token.** `[CLS]` + `[SEP]` chiếm 2 chỗ
  trong đúng cái cửa sổ 256 đó. Đếm không kèm chúng là báo thiếu 2 token ở mọi
  chunk sát ngưỡng, và những chunk *vừa đúng* 256 sẽ bị coi là không bị cắt.
* **"Không biết giới hạn" khác "không có giới hạn".** Provider trả `None` thì
  phép đo phải **bỏ qua kèm cảnh báo**, chứ không được kết luận "0% bị cắt".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .base import EmbeddingProvider

__all__ = ["TruncationStats", "mean_tokens_per_char", "measure_truncation", "token_stats"]


def _percentile(values: Sequence[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(pct / 100.0 * (len(ordered) - 1)))
    return int(ordered[index])


def _percentile_float(values: Sequence[float], pct: float) -> float:
    """Cùng quy ước nearest-rank với `_percentile`, cho giá trị thực."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(pct / 100.0 * (len(ordered) - 1)))
    return float(ordered[index])


@dataclass(frozen=True)
class TruncationStats:
    """Kết quả đo trên một tập text, ứng với một giới hạn token cụ thể."""

    limit: int
    n_texts: int
    n_truncated: int
    tokens_total: int
    """Tổng token nếu model đọc hết — tức số token *đáng lẽ* phải embed."""
    tokens_kept: int
    """Tổng token model thật sự đọc: `sum(min(n, limit))`."""
    chars_total: int
    token_p50: int
    token_p95: int
    token_max: int
    chars_per_token_p05: float = 0.0
    """Phân vị **thấp** của mật độ `ký tự/token` — trường hợp xé vụn nhất.

    Đây là con số dùng để suy `chunk_size`, và nó phải là phân vị thấp chứ
    không phải trung bình. Lấy trung bình trả lời câu hỏi "`chunk_size` nào làm
    chunk *trung bình* vừa khít cửa sổ" — tức đúng ngưỡng mà một nửa số chunk
    vẫn bị cắt. Đo thật trên corpus này: trung bình cho ra 946 ký tự trong khi
    ở 1000 ký tự đã có 56,9% chunk bị cắt. Một gợi ý sai theo hướng rất dễ tin.
    """

    @property
    def truncated_ratio(self) -> float:
        """Tỉ lệ **chunk** bị cắt. Dễ đọc nhưng không nói mất bao nhiêu nội dung."""
        return self.n_truncated / self.n_texts if self.n_texts else 0.0

    @property
    def tokens_lost_ratio(self) -> float:
        """Tỉ lệ **token** không tới được vector. Đây mới là con số đáng lo.

        Tách khỏi `truncated_ratio` vì hai số này kể hai câu chuyện khác nhau:
        90% chunk bị cắt mất mỗi chunk 1 token là chuyện nhỏ; 10% chunk bị cắt
        mất mỗi chunk một nửa là mất hẳn nội dung.
        """
        return 1.0 - self.tokens_kept / self.tokens_total if self.tokens_total else 0.0

    def char_budget(self, *, special_tokens: int = 2, quantile: float = 5.0) -> int:
        """Số ký tự tối đa cho một chunk để ~95% chunk nằm trong cửa sổ.

        `quantile` chỉ để ghi lại ý định — mật độ đã được chốt ở
        `chars_per_token_p05` lúc đo. Tham số `special_tokens` trừ chỗ của
        `[CLS]`/`[SEP]`.

        **Không** thay được cho việc build lại rồi đo lại: quan hệ ký tự↔token
        không hoàn toàn tuyến tính (`min_chunk_size`, `max_chunk_size` và ngữ
        cảnh hàng xóm đều bóp méo nó). Đây là điểm khởi đầu để thử, không phải
        câu trả lời.
        """
        del quantile  # có mặt để đọc code hiểu con số p05 tới từ đâu
        if self.chars_per_token_p05 <= 0:
            return 0
        return int((self.limit - special_tokens) * self.chars_per_token_p05)

    @property
    def tokens_per_char(self) -> float:
        """Dùng để hiệu chuẩn `chunk_size` (tính bằng **ký tự**) về token.

        Tỉ lệ này phụ thuộc ngôn ngữ và tokenizer, nên phải **đo** chứ không
        đoán: PhoBERT xé chữ tiếng Anh vụn hơn tiếng Việt rất nhiều.
        """
        return self.tokens_total / self.chars_total if self.chars_total else 0.0

    @property
    def is_healthy(self) -> bool:
        return self.n_truncated == 0

    def as_dict(self) -> dict[str, float]:
        return {
            "limit": float(self.limit),
            "n_texts": float(self.n_texts),
            "n_truncated": float(self.n_truncated),
            "truncated_ratio": round(self.truncated_ratio, 4),
            "tokens_total": float(self.tokens_total),
            "tokens_kept": float(self.tokens_kept),
            "chars_total": float(self.chars_total),
            "tokens_lost_ratio": round(self.tokens_lost_ratio, 4),
            "tokens_per_char": round(self.tokens_per_char, 4),
            "chars_per_token_p05": round(self.chars_per_token_p05, 3),
            "char_budget": float(self.char_budget()),
            "token_p50": float(self.token_p50),
            "token_p95": float(self.token_p95),
            "token_max": float(self.token_max),
        }

    def summary(self) -> str:
        if not self.n_texts:
            return "không có text nào để đo"
        return (
            f"{self.n_truncated}/{self.n_texts} chunk bị cắt "
            f"({100 * self.truncated_ratio:.1f}%) · "
            f"mất {100 * self.tokens_lost_ratio:.1f}% token · "
            f"token p50 {self.token_p50} · p95 {self.token_p95} · max {self.token_max} "
            f"(giới hạn {self.limit})"
        )


def token_stats(
    counts: Sequence[int],
    *,
    limit: int,
    chars: Sequence[int] = (),
) -> TruncationStats:
    """Phần thuần số học, tách khỏi model để test được không cần `torch`.

    `counts` là số token của từng text (đã kèm special token). `chars` là độ dài
    ký tự của **cùng** các text đó, theo cùng thứ tự — cần để tính mật độ. `limit`
    là `max_sequence_tokens` của model; text đúng bằng `limit` **không** bị cắt.
    """
    if limit <= 0:
        raise ValueError(f"limit phải dương, nhận {limit}")
    counts = list(counts)
    chars = list(chars)
    if chars and len(chars) != len(counts):
        raise ValueError(
            f"counts và chars phải cùng độ dài, nhận {len(counts)} và {len(chars)} — "
            "lệch nhau nghĩa là mật độ được tính trên hai tập text khác nhau"
        )
    densities = [c / t for c, t in zip(chars, counts, strict=True) if t > 0] if chars else []
    return TruncationStats(
        limit=limit,
        n_texts=len(counts),
        n_truncated=sum(1 for c in counts if c > limit),
        tokens_total=sum(counts),
        tokens_kept=sum(min(c, limit) for c in counts),
        chars_total=sum(chars),
        token_p50=_percentile(counts, 50),
        token_p95=_percentile(counts, 95),
        token_max=max(counts) if counts else 0,
        chars_per_token_p05=_percentile_float(densities, 5.0),
    )


def measure_truncation(
    provider: EmbeddingProvider,
    texts: Sequence[str],
    *,
    batch_size: int = 256,
) -> TruncationStats | None:
    """Đo trên `texts` bằng tokenizer thật của `provider`.

    Trả `None` khi provider không biết giới hạn hoặc không đếm được token —
    người gọi phải cảnh báo, **không** được coi là "sạch".

    Chia lô vì tokenizer nhanh nhưng dựng list Python cho vài chục nghìn text
    một lượt thì tốn bộ nhớ vô ích; ở đây chỉ cần các con số cộng dồn.
    """
    limit = provider.max_sequence_tokens
    if limit is None:
        return None

    counts: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = provider.count_tokens(texts[start : start + batch_size])
        if batch is None:
            return None
        counts.extend(batch)

    return token_stats(counts, limit=limit, chars=[len(t) for t in texts])


def mean_tokens_per_char(counts: Sequence[int], lengths: Sequence[int]) -> float:
    """Trung bình có trọng số của token/ký tự — dùng để suy `chunk_size`.

    Cố ý **không** lấy trung bình của các tỉ lệ từng chunk: chunk ngắn có tỉ lệ
    lệch mạnh (special token chiếm phần lớn) và sẽ kéo con số đi sai hướng.
    """
    total_chars = sum(lengths)
    return sum(counts) / total_chars if total_chars else 0.0
