"""`TD-37` — RRF có trọng số: cân tĩnh, hay cân theo độ tin từng truy vấn?

`TD-35` truy ra rằng thủ phạm của hồi quy `cross_lingual` là **tầng hợp nhất**,
không phải reranker: sparse tìm được 1/43 câu `cross_lingual` (`hit_rate@50` =
0,0233) nhưng `k=1` vẫn cho hạng 1 của nó trọng số ½. Nhánh ấy không tệ — ở
`factoid` nó đạt 0,8676 — nó chỉ **mù với loại truy vấn này**, và RRF không có
chỗ nào để biết điều đó.

`TD-35` đề xuất chữa bằng **định tuyến** (`W4-07`). `TD-37` phản biện: định tuyến
là chữa triệu chứng, và nếu cách chữa đúng là cân trọng số theo độ tin thì
`W4-07` đang giải sai bài toán. Hạng mục này quyết định điều đó **trước** khi
`W4-07` được dựng.

## Vì sao đo được mà không cần reranker và không tốn đồng nào

`TD-35` đã xác định thiệt hại nằm ở **trần pool** (`hit_rate@50` của nhánh nền):
dense **+0,0465**, sparse **0,0000**, hybrid **−0,0233** — ô âm duy nhất trong 18.
Reranker chỉ chọn lại trong pool nó nhận; nó không tạo ra được ứng viên không có
ở đó. Nên trần pool là đúng đại lượng, và nó đo được bằng hai lượt truy hồi rồi
**hợp nhất lại offline** bao nhiêu lần tuỳ ý.

## Dự đoán (viết TRƯỚC khi chạy)

* **P1** — Quét trọng số tĩnh sẽ **đánh đổi đơn điệu**: hạ sparse xuống thì
  `cross_lingual` lên và `factoid` xuống, tức nó trượt trên đúng đường đánh đổi
  mà định tuyến trượt. Không có bữa trưa miễn phí ở trọng số tĩnh.
* **P2** — Điểm của nhánh sparse **có** tách được hai nhóm: `cross_lingual` gần
  như không trùng token nào với tài liệu nên đỉnh phân bố của nó thấp và bẹt.
* **P3** — Cân thích ứng theo tín hiệu ở P2 sẽ **thắng cả hai** đầu của P1 trên
  trần pool tổng.
* **P4** — Nhưng dense-only sẽ **nằm trong nhiễu** so với bản thích ứng tốt
  nhất: ở độ sâu 50, đóng góp *thực* của sparse có thể bằng 0 sau khi thôi cho
  nó làm hỏng.

Chạy:

    python scripts/td37_probe.py --report plans/reports/probes/td-37.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.eval.golden import GoldenQuery, load_golden_set
from pipeline.eval.metrics import hit_rate_at_k
from pipeline.eval.retrieval_eval import _resolve_span_labels
from pipeline.eval.spans import DEFAULT_MIN_OVERLAP_RATIO
from pipeline.indexing.config import load_index_config
from rag_core.retrieval import reciprocal_rank_fusion
from rag_core.retrieval.sparse import QdrantSparseRetriever
from rag_core.settings import get_settings

logger = logging.getLogger("td37")

POOL = 50
RRF_K = 1
"""`k` của điểm vận hành (`W2-04` đo được `k=60` của bài báo là giá trị tệ nhất)."""


@dataclass(frozen=True)
class Branch:
    """Một lượt truy hồi đã cache: khoá theo thứ hạng, và điểm thô."""

    keys: tuple[str, ...]
    scores: tuple[float, ...]

    @property
    def top_score(self) -> float:
        return self.scores[0] if self.scores else 0.0

    @property
    def peakedness(self) -> float:
        """Đỉnh cao hơn phần thân bao nhiêu lần.

        Dùng tỉ lệ chứ không dùng hiệu: điểm sparse là tích vô hướng của trọng số
        không âm nên nó **không có trần** và tỉ lệ thuận với độ dài truy vấn — một
        ngưỡng trên giá trị tuyệt đối sẽ đo độ dài câu hỏi chứ không đo độ tin.
        """
        if len(self.scores) < 5:
            return 0.0
        body = statistics.median(self.scores[1:10]) or 1e-9
        return self.scores[0] / body


def collect(
    config_path: Path, queries: Sequence[GoldenQuery]
) -> tuple[list[Branch], list[Branch], Sequence[GoldenQuery]]:
    """Một lượt duy nhất qua Qdrant; mọi luật hợp nhất sau đó chạy offline."""
    index_config = load_index_config(config_path)
    settings = get_settings()
    store = index_config.build_retriever(
        index_config.build_embeddings(),
        url=settings.qdrant_url,
        api_key=(settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None),
    )
    store.verify_schema()
    sparse = QdrantSparseRetriever(store)

    resolved, _ = _resolve_span_labels(store, queries, DEFAULT_MIN_OVERLAP_RATIO)

    dense_pools: list[Branch] = []
    sparse_pools: list[Branch] = []
    for i, query in enumerate(resolved, 1):
        d = store.retrieve(query.query, top_k=POOL)
        s = sparse.retrieve(query.query, top_k=POOL)
        dense_pools.append(Branch(tuple(h.chunk.chunk_id for h in d), tuple(h.score for h in d)))
        sparse_pools.append(Branch(tuple(h.chunk.chunk_id for h in s), tuple(h.score for h in s)))
        if i % 50 == 0:
            logger.info("  %d/%d truy vấn", i, len(resolved))
    return dense_pools, sparse_pools, resolved


# ---------------------------------------------------------------------------
# Luật hợp nhất — tất cả chạy offline trên pool đã cache
# ---------------------------------------------------------------------------

Rule = Callable[[Branch, Branch], tuple[float, float]]
"""Trả `(w_dense, w_sparse)` cho một truy vấn cụ thể."""


def static(w_sparse: float) -> Rule:
    def rule(dense: Branch, sparse: Branch) -> tuple[float, float]:
        return 1.0, w_sparse

    return rule


def adaptive(threshold: float) -> Rule:
    """Tắt hẳn sparse khi phân bố điểm của nó bẹt.

    ⭐ Tín hiệu phải **quan sát được lúc chạy**: nó chỉ nhìn vào điểm của chính
    nhánh sparse cho truy vấn ấy — không nhìn nhãn, không nhìn ngôn ngữ, không
    cần một bộ phân loại phải huấn luyện và bảo trì. Đó là điểm khác biệt với
    định tuyến của `W4-07`.
    """

    def rule(dense: Branch, sparse: Branch) -> tuple[float, float]:
        return (1.0, 0.0) if sparse.peakedness < threshold else (1.0, 1.0)

    return rule


def fuse(dense: Branch, sparse: Branch, rule: Rule) -> list[str]:
    w_dense, w_sparse = rule(dense, sparse)
    fused = reciprocal_rank_fusion(
        [list(dense.keys), list(sparse.keys)], k=RRF_K, weights=(w_dense, w_sparse)
    )
    return [item.key for item in fused]


def per_query(
    queries: Sequence[GoldenQuery],
    dense_pools: Sequence[Branch],
    sparse_pools: Sequence[Branch],
    rule: Rule,
    depth: int,
) -> dict[str, float]:
    """`hit_rate@depth` cho **từng** truy vấn — nguyên liệu của phép kiểm dấu."""
    out: dict[str, float] = {}
    for query, dense, sparse in zip(queries, dense_pools, sparse_pools, strict=True):
        value = hit_rate_at_k(fuse(dense, sparse, rule), query.relevant_chunk_ids, depth)
        if value is not None:
            out[query.query_id] = value
    return out


def evaluate(
    queries: Sequence[GoldenQuery],
    dense_pools: Sequence[Branch],
    sparse_pools: Sequence[Branch],
    rule: Rule,
) -> dict[str, dict[str, float]]:
    """`hit_rate@50` (trần pool) và `@10`, tổng và theo nhóm."""
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for query, dense, sparse in zip(queries, dense_pools, sparse_pools, strict=True):
        keys = fuse(dense, sparse, rule)
        for depth in (10, 50):
            value = hit_rate_at_k(keys, query.relevant_chunk_ids, depth)
            if value is None:
                continue
            buckets["OVERALL"][f"hit_rate@{depth}"].append(value)
            buckets[query.category.value][f"hit_rate@{depth}"].append(value)
    return {
        name: {metric: statistics.fmean(values) for metric, values in metrics.items()}
        for name, metrics in buckets.items()
    }


def discordant(base: dict[str, float], variant: dict[str, float]) -> dict[str, float]:
    """Số câu **đổi chiều**, và `p` của phép kiểm dấu hai phía.

    ⭐ Trung bình không đủ: `TD-13` đo được rằng một chênh lệch 0,02 trên 209 câu
    là 4–5 câu, và 4↔0 có `p` sàn = 0,125 — tức *không kết luận được* dù bảng
    trông thuyết phục. Chỉ cặp bất đồng mới nói được điều đó.
    """
    wins = sum(1 for q in base if variant.get(q, 0.0) > base[q])
    losses = sum(1 for q in base if variant.get(q, 0.0) < base[q])
    n = wins + losses
    if n == 0:
        return {"wins": 0.0, "losses": 0.0, "p": 1.0}
    # Nhị thức đối xứng, hai phía. `math.comb` chính xác nên không cần scipy.
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / (2**n)
    return {"wins": float(wins), "losses": float(losses), "p": min(1.0, 2 * tail)}


# ---------------------------------------------------------------------------
# Tín hiệu: có phân biệt được "sparse dùng được" với "sparse mù" không
# ---------------------------------------------------------------------------


def signal_analysis(
    queries: Sequence[GoldenQuery], sparse_pools: Sequence[Branch]
) -> dict[str, Any]:
    """AUC giữa `peakedness` và việc nhánh sparse **có** tìm ra gì đúng không.

    AUC vì nó không cần chọn ngưỡng: 0,5 = tín hiệu vô dụng, 1,0 = tách hoàn hảo.
    Chọn ngưỡng là bước sau, và chỉ đáng làm nếu AUC nói rằng có gì để chọn.
    """
    rows: list[tuple[float, bool, str]] = []
    for query, sparse in zip(queries, sparse_pools, strict=True):
        hit = hit_rate_at_k(list(sparse.keys), query.relevant_chunk_ids, POOL)
        if hit is None:
            continue
        rows.append((sparse.peakedness, hit == 1.0, query.category.value))

    positives = [p for p, hit, _ in rows if hit]
    negatives = [p for p, hit, _ in rows if not hit]
    auc = float("nan")
    if positives and negatives:
        wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
        auc = wins / (len(positives) * len(negatives))

    by_category: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for peak, hit, category in rows:
        grouped[category].append((peak, hit))
    for category, values in grouped.items():
        by_category[category] = {
            "n": float(len(values)),
            "sparse_hit_rate@50": statistics.fmean(1.0 if h else 0.0 for _, h in values),
            "median_peakedness": statistics.median(p for p, _ in values),
        }
    return {
        "auc_peakedness_vs_sparse_hit": auc,
        "n_sparse_useful": len(positives),
        "n_sparse_blind": len(negatives),
        "by_category": by_category,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/indexing/bgem3-contextual.yaml")
    )
    parser.add_argument("--golden", type=Path, default=Path("data/golden/golden_v1.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("plans/reports/probes/td-37.json"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    queries = load_golden_set(args.golden)
    logger.info("Truy hồi hai nhánh cho %d truy vấn…", len(queries))
    dense_pools, sparse_pools, resolved = collect(args.config, queries)

    rules: dict[str, Rule] = {
        f"w={w:g}": static(w) for w in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0)
    }
    for threshold in (1.05, 1.15, 1.3, 1.6, 2.0):
        rules[f"adaptive_peak<{threshold}"] = adaptive(threshold)

    results = {
        name: evaluate(resolved, dense_pools, sparse_pools, rule) for name, rule in rules.items()
    }
    signal = signal_analysis(resolved, sparse_pools)

    # Điểm vận hành hiện tại là trọng số đều; mọi phép so đều so với nó.
    base = {
        depth: per_query(resolved, dense_pools, sparse_pools, static(1.0), depth)
        for depth in (10, 50)
    }
    tests = {
        name: {
            f"@{depth}": discordant(
                base[depth], per_query(resolved, dense_pools, sparse_pools, rule, depth)
            )
            for depth in (10, 50)
        }
        for name, rule in rules.items()
    }

    report: dict[str, Any] = {
        "n_queries": len(resolved),
        "pool": POOL,
        "rrf_k": RRF_K,
        "baseline": "w=1 (trọng số đều, điểm vận hành W3)",
        "signal": signal,
        "rules": results,
        "sign_test_vs_baseline": tests,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("\nAUC(peakedness → sparse hữu ích) = %.3f", signal["auc_peakedness_vs_sparse_hit"])
    logger.info(
        "%-22s %8s %8s %10s %8s %14s",
        "luật",
        "@50",
        "@10",
        "cross_ling",
        "factoid",
        "@50 thắng↔thua p",
    )
    for name, table in results.items():
        test = tests[name]["@50"]
        logger.info(
            "%-22s %8.4f %8.4f %10.4f %8.4f %6d↔%-3d p=%.3f",
            name,
            table["OVERALL"]["hit_rate@50"],
            table["OVERALL"]["hit_rate@10"],
            table.get("cross_lingual", {}).get("hit_rate@50", float("nan")),
            table.get("factoid", {}).get("hit_rate@50", float("nan")),
            int(test["wins"]),
            int(test["losses"]),
            test["p"],
        )
    logger.info("\nBáo cáo: %s", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
