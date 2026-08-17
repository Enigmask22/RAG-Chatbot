"""Metric xếp hạng cho truy hồi — Recall@k, Precision@k, MRR, nDCG@k, HitRate.

Toàn bộ module này **không gọi LLM nào**. Đó là điểm quan trọng nhất của nó:
nhóm metric quyết định phần lớn chất lượng RAG đo được bằng embedding + reranker
chạy trên máy cá nhân, không tốn một đồng API và không phụ thuộc nhà cung cấp.
Gate `G1` có một mục riêng kiểm chứng điều này bằng cách chạy eval với API key rỗng.

Ba quyết định về ngữ nghĩa, ghi rõ vì chúng thay đổi con số:

1. **Truy vấn không có tài liệu liên quan (`unanswerable`) bị loại khỏi mọi
   metric xếp hạng** — hàm trả `None`, không trả `0.0`. Recall trên tập rỗng là
   không xác định; quy ước thành 0 sẽ kéo tụt điểm một cách vô nghĩa, quy ước
   thành 1 thì thổi phồng. Nhóm này được đo riêng bằng refusal correctness ở W5.
2. **Danh sách trả về được khử trùng lặp, giữ nguyên thứ tự.** Retriever trả
   cùng một chunk hai lần không được tính công hai lần.
3. **nDCG dùng độ liên quan nhị phân theo mặc định**, IDCG tính trên
   `min(|relevant|, k)` — tức trần lý tưởng bị chặn bởi k, đúng chuẩn.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "average_precision_at_k",
    "dedupe_preserving_order",
    "hit_rate_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]


def dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _prepare(
    retrieved: Sequence[str], relevant: Iterable[str], k: int
) -> tuple[list[str], set[str]]:
    if k < 1:
        raise ValueError("k phải ≥ 1")
    return dedupe_preserving_order(retrieved)[:k], set(relevant)


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float | None:
    """Tỉ lệ tài liệu liên quan lọt vào top-k. `None` nếu không có tài liệu liên quan."""
    top, rel = _prepare(retrieved, relevant, k)
    if not rel:
        return None
    return len(rel & set(top)) / len(rel)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float | None:
    """Tỉ lệ kết quả trong top-k thực sự liên quan.

    Mẫu số là `k` chứ không phải `len(top)`: trả về ít hơn k kết quả là một dạng
    thất bại của retriever, không phải hoàn cảnh giảm nhẹ.
    """
    top, rel = _prepare(retrieved, relevant, k)
    if not rel:
        return None
    return len(rel & set(top)) / k


def hit_rate_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float | None:
    """1.0 nếu có ít nhất một tài liệu liên quan trong top-k."""
    top, rel = _prepare(retrieved, relevant, k)
    if not rel:
        return None
    return 1.0 if rel & set(top) else 0.0


def reciprocal_rank(
    retrieved: Sequence[str], relevant: Iterable[str], k: int | None = None
) -> float | None:
    """`1 / thứ hạng` của tài liệu liên quan đầu tiên; 0.0 nếu không có trong top-k."""
    rel = set(relevant)
    if not rel:
        return None
    top = dedupe_preserving_order(retrieved)
    if k is not None:
        top = top[:k]
    for rank, item in enumerate(top, start=1):
        if item in rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int,
    *,
    gains: Mapping[str, float] | None = None,
) -> float | None:
    """nDCG@k với discount `1/log2(rank+1)`.

    `gains` cho phép độ liên quan có mức (ví dụ 2 = trả lời trực tiếp,
    1 = liên quan một phần). Không truyền thì coi mọi tài liệu liên quan là 1.0.
    """
    top, rel = _prepare(retrieved, relevant, k)
    if not rel:
        return None

    def gain(chunk_id: str) -> float:
        if chunk_id not in rel:
            return 0.0
        return float(gains[chunk_id]) if gains and chunk_id in gains else 1.0

    dcg = sum(gain(cid) / math.log2(rank + 1) for rank, cid in enumerate(top, start=1))

    ideal_gains = sorted((gain(cid) for cid in rel), reverse=True)[:k]
    idcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal_gains, start=1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def average_precision_at_k(
    retrieved: Sequence[str], relevant: Iterable[str], k: int
) -> float | None:
    """AP@k — trung bình precision tại mỗi vị trí trúng. Gộp lại thành MAP."""
    top, rel = _prepare(retrieved, relevant, k)
    if not rel:
        return None
    hits = 0
    total = 0.0
    for rank, cid in enumerate(top, start=1):
        if cid in rel:
            hits += 1
            total += hits / rank
    denominator = min(len(rel), k)
    return total / denominator if denominator else 0.0


def mean_reciprocal_rank(
    cases: Iterable[tuple[Sequence[str], Iterable[str]]], k: int | None = None
) -> float | None:
    """MRR trên nhiều truy vấn. Truy vấn không có tài liệu liên quan bị bỏ qua."""
    values = [
        rr
        for retrieved, relevant in cases
        if (rr := reciprocal_rank(retrieved, relevant, k)) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)
