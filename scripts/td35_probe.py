"""`TD-35` — truy cơ chế làm `cross_lingual` đi ngược *chỉ sau* reranker.

`exp-002` đo được một thứ không tự giải thích: dán ngữ cảnh làm `cross_lingual`
**15/15 metric dương ở tầng dense** rồi **11/15 âm ở tầng reranked**. Nghi phạm
duy nhất nêu ra khi ấy là cửa sổ 512 của reranker (tỉ lệ cắt tăng 87×), nhưng
0,72% quá nhỏ để giải thích một nhóm 43 câu đổi dấu — nên nó là *giả thuyết chưa
kiểm*, không phải kết luận.

Ba phép đo, mỗi phép loại được một khả năng:

1. **Tỉ lệ cắt theo nhóm và theo ngôn ngữ tài liệu.** Nếu `cross_lingual` không
   cắt nhiều hơn hẳn thì truncation không phải cơ chế, và đoạn tương ứng của
   `exp-002` §3 phải sửa.
2. **Trần pool theo nhóm** (`hit_rate@50` của nhánh nền, có và không ngữ cảnh).
   Phân biệt "reranker nhận tập ứng viên tệ hơn" với "reranker chấm sai".
3. ⭐ **Ablation quyết định — cùng pool, khác văn bản.** Lấy pool từ collection
   có ngữ cảnh rồi rerank **hai lần**: một lần với văn bản đã dán, một lần với
   văn bản đã **bóc ngữ cảnh ra** (`original_content`). Cùng truy vấn, cùng ứng
   viên, cùng model, cùng thứ tự nền — khác đúng một thứ. Đây là phép duy nhất
   tách được "pool đổi" khỏi "văn bản đổi", vì hai lượt eval đã chạy ở `exp-002`
   đổi **cả hai** cùng lúc.

Chạy:

    python scripts/td35_probe.py --report plans/reports/probes/td-35.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pipeline.eval.golden import GoldenQuery, load_golden_set
from pipeline.eval.metrics import hit_rate_at_k, ndcg_at_k, reciprocal_rank
from pipeline.eval.retrieval_eval import _resolve_span_labels
from pipeline.eval.spans import DEFAULT_MIN_OVERLAP_RATIO
from pipeline.indexing.config import load_index_config
from rag_core.chunking.contextual import original_content
from rag_core.reranking import CrossEncoderReranker
from rag_core.retrieval import Retriever, build_branch
from rag_core.schemas import RetrievedChunk
from rag_core.settings import get_settings

DEFAULT_SEED = 20260820
POOL = 50
TOP_N = 6


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


#: Tham số nhánh hybrid của điểm vận hành (`W2-04`: `k=1` thắng, `k=60` tệ nhất).
HYBRID_OPTIONS = {"k": 1, "candidate_k": 20}


def _store(config_path: Path) -> Any:
    index_config = load_index_config(config_path)
    settings = get_settings()
    embeddings = index_config.build_embeddings()
    store = index_config.build_retriever(
        embeddings,
        url=settings.qdrant_url,
        api_key=(settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None),
    )
    store.verify_schema()
    return store


def _build(store: Any, base: str) -> Retriever:
    """⚠️ Nhánh nền phải khớp với lượt eval đang đi truy nguyên nhân.

    Lượt đầu của phép đo này chạy nền `dense` trong khi `exp-002` chạy nền
    `hybrid`, và chênh lệch giữa hai nền ấy (0,5669 vs 0,5041 trên
    `cross_lingual`) **lớn hơn chính hiện tượng** đang đi tìm — tức phép đo đo
    một hệ thống khác với hệ thống được báo cáo.
    """
    return build_branch(store, base, **(HYBRID_OPTIONS if base == "hybrid" else {}))


def collect_pools(branch: Retriever, queries: Sequence[GoldenQuery]) -> list[list[RetrievedChunk]]:
    return [branch.retrieve(query.query, top_k=POOL) for query in queries]


# ---------------------------------------------------------------------------
# 1. Truncation, chia theo nhóm truy vấn và theo ngôn ngữ TÀI LIỆU
# ---------------------------------------------------------------------------


def _summarise(counts: Sequence[int], max_length: int) -> dict[str, float]:
    if not counts:
        return {}
    return {
        "pairs": float(len(counts)),
        "p50_tokens": float(statistics.median(counts)),
        "p95_tokens": _percentile([float(c) for c in counts], 0.95),
        "max_tokens": float(max(counts)),
        "truncated_ratio": sum(1 for c in counts if c > max_length) / len(counts),
    }


def truncation_breakdown(
    reranker: CrossEncoderReranker,
    queries: Sequence[GoldenQuery],
    pools: Sequence[Sequence[RetrievedChunk]],
) -> dict[str, Any]:
    """Cùng một tập cặp, cắt theo hai trục khác nhau.

    Hai trục **không** thay thế được nhau: nhóm truy vấn là thuộc tính của câu
    hỏi, ngôn ngữ là thuộc tính của tài liệu được truy hồi. Giả thuyết
    "tiếng Việt tokenise vụn hơn nên chạm trần trước" chỉ đo được ở trục thứ hai.
    """
    overall: list[int] = []
    by_category: dict[str, list[int]] = defaultdict(list)
    by_doc_lang: dict[str, list[int]] = defaultdict(list)

    for query, pool in zip(queries, pools, strict=True):
        counts = reranker.count_pair_tokens(query.query, [h.chunk.content for h in pool])
        overall.extend(counts)
        by_category[query.category.value].extend(counts)
        for hit, count in zip(pool, counts, strict=True):
            meta = hit.chunk.metadata
            by_doc_lang[meta.lang.value if meta is not None else "unknown"].append(count)

    return {
        "max_length": float(reranker.max_length),
        "overall": _summarise(overall, reranker.max_length),
        "by_category": {
            name: _summarise(values, reranker.max_length)
            for name, values in sorted(by_category.items())
        },
        "by_doc_language": {
            name: _summarise(values, reranker.max_length)
            for name, values in sorted(by_doc_lang.items())
        },
    }


# ---------------------------------------------------------------------------
# 2. Trần pool theo nhóm
# ---------------------------------------------------------------------------


def ceiling_by_category(
    queries: Sequence[GoldenQuery], pools: Sequence[Sequence[RetrievedChunk]]
) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for query, pool in zip(queries, pools, strict=True):
        ids = [hit.chunk.chunk_id for hit in pool]
        value = hit_rate_at_k(ids, query.relevant_chunk_ids, POOL)
        if value is not None:
            buckets[query.category.value].append(value)
    return {
        name: {"n": float(len(values)), f"hit_rate@{POOL}": sum(values) / len(values)}
        for name, values in sorted(buckets.items())
    }


# ---------------------------------------------------------------------------
# 3. ⭐ Ablation: cùng pool, khác văn bản
# ---------------------------------------------------------------------------


def _score_ordering(
    queries: Sequence[GoldenQuery], orderings: Sequence[Sequence[str]]
) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for query, ids in zip(queries, orderings, strict=True):
        rel = query.relevant_chunk_ids
        values = {
            "hit_rate@1": hit_rate_at_k(ids, rel, 1),
            "hit_rate@5": hit_rate_at_k(ids, rel, 5),
            f"hit_rate@{TOP_N}": hit_rate_at_k(ids, rel, TOP_N),
            "ndcg@10": ndcg_at_k(ids, rel, 10),
            "mrr": reciprocal_rank(ids, rel),
        }
        for group in (query.category.value, "__all__"):
            for name, value in values.items():
                if value is not None:
                    buckets[group][name].append(value)
    return {
        group: {"n": float(len(next(iter(metrics.values()))))}
        | {name: sum(vals) / len(vals) for name, vals in sorted(metrics.items())}
        for group, metrics in sorted(buckets.items())
    }


def rerank_ablation(
    reranker: CrossEncoderReranker,
    queries: Sequence[GoldenQuery],
    pools: Sequence[Sequence[RetrievedChunk]],
) -> dict[str, Any]:
    """Xếp lại **đúng cùng một pool** bằng hai phiên bản văn bản của chính nó.

    ⚠️ Điểm phải giữ chặt: `stripped` **không** phải pool của collection không
    ngữ cảnh. Nó là pool có ngữ cảnh, chỉ bóc phần chữ do LLM viết ra. Nếu lấy
    pool kia thì tập ứng viên đổi theo và phép đo lại lẫn hai nguyên nhân — đúng
    cái mà `exp-002` đã lẫn.
    """
    enriched_order: list[list[str]] = []
    stripped_order: list[list[str]] = []
    n_identical = 0
    n_pairs = 0

    for query, pool in zip(queries, pools, strict=True):
        ids = [hit.chunk.chunk_id for hit in pool]
        enriched = [hit.chunk.content for hit in pool]
        stripped = [original_content(hit.chunk) for hit in pool]
        n_pairs += len(ids)
        n_identical += sum(1 for a, b in zip(enriched, stripped, strict=True) if a == b)

        for texts, sink in ((enriched, enriched_order), (stripped, stripped_order)):
            scores = reranker.score(query.query, texts)
            order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
            sink.append([ids[i] for i in order])

    return {
        "n_pairs": float(n_pairs),
        "n_texts_identical": float(n_identical),
        "identical_ratio": n_identical / n_pairs if n_pairs else float("nan"),
        "enriched": _score_ordering(queries, enriched_order),
        "stripped": _score_ordering(queries, stripped_order),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/indexing/bgem3-contextual.yaml")
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/indexing/bgem3.yaml"),
        help="Collection không ngữ cảnh — chỉ dùng cho trần pool và truncation nền.",
    )
    parser.add_argument("--golden", type=Path, default=Path("data/golden/golden_v1.jsonl"))
    parser.add_argument(
        "--base",
        default="hybrid",
        choices=["dense", "sparse", "hybrid"],
        help="Phải khớp nhánh nền của lượt eval đang truy — `exp-002` dùng hybrid.",
    )
    parser.add_argument(
        "--branches",
        default="dense,sparse,hybrid",
        help="Các nhánh đo trần pool. Rẻ vì không rerank.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    reranker = CrossEncoderReranker(
        device=args.device, batch_size=args.batch_size, max_length=args.max_length
    )
    report: dict[str, Any] = {
        "reranker": reranker.name,
        "device": reranker.device,
        "seed": DEFAULT_SEED,
        "pool": float(POOL),
    }

    loaded = load_golden_set(args.golden)
    sweep = [name.strip() for name in args.branches.split(",") if name.strip()]

    for label, config_path in (("contextual", args.config), ("baseline", args.baseline_config)):
        store = _store(config_path)
        branch = _build(store, args.base)
        queries, _ = _resolve_span_labels(branch, loaded, DEFAULT_MIN_OVERLAP_RATIO)
        pools = collect_pools(branch, queries)
        section: dict[str, Any] = {
            "config": str(config_path),
            "base": args.base,
            "branch": branch.name,
            "n_queries": float(len(queries)),
            "truncation": truncation_breakdown(reranker, queries, pools),
            "ceiling_by_category": ceiling_by_category(queries, pools),
        }
        # Trần pool của TỪNG nhánh nền. Rẻ (không rerank) và là phép duy nhất
        # tách được "tầng nào làm hỏng `cross_lingual`" — dense và sparse đi vào
        # RRF theo hai đường khác nhau, nên gộp chúng lại thì không quy được.
        section["ceiling_by_branch"] = {
            name: ceiling_by_category(queries, collect_pools(_build(store, name), queries))
            for name in sweep
        }
        if label == "contextual" and not args.skip_ablation:
            section["ablation"] = rerank_ablation(reranker, queries, pools)
        report[label] = section

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
