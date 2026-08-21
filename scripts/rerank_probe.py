"""Đo bốn thứ về nhánh reranked mà `retrieval_eval` không đo được — `W2-05`.

Tồn tại vì mỗi con số ở đây trả lời một câu hỏi khác nhau, và cả bốn đều là loại
câu hỏi mà bỏ qua thì dễ kết luận sai:

1. **Trần vùng phủ** (`--pool`). Reranker chỉ xếp lại những gì nhánh nền đưa cho,
   nên `hit_rate@1` sau rerank bị chặn trên bởi `hit_rate@pool` của nhánh nền.
   `retrieval_eval` chỉ chấm ở `k ∈ {1, 5, 10, 20}` nên nó không nói được trần ở
   `pool = 50`. Không có số này thì "reranker cải thiện 5 điểm" là một câu vô
   nghĩa — 5 trên bao nhiêu điểm còn lại?
2. **Truncation ở `max_length`.** `TD-11` là cả một tuần đi sai hướng vì một giả
   định về truncation không được đo. Đo lại ở đây, cho **cặp** (truy vấn, chunk).
3. **Bão hoà sigmoid.** Lý lẽ để mặc định dùng logit thô là "sigmoid bão hoà ở
   float32 và sinh ties nhân tạo". Nếu nó không thật thì lý lẽ đó sai và phải nói.
4. **Độ trễ** (DoD: rerank 50 → 6 trong < 400 ms trên GPU, và có CPU fallback).

Chạy:

    python scripts/rerank_probe.py --config configs/indexing/bgem3.yaml
    python scripts/rerank_probe.py --device cpu --latency-only   # đo fallback
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pipeline.eval.golden import GoldenQuery, load_golden_set
from pipeline.eval.retrieval_eval import _resolve_span_labels
from pipeline.eval.spans import DEFAULT_MIN_OVERLAP_RATIO
from pipeline.indexing.config import load_index_config
from rag_core.reranking import CrossEncoderReranker
from rag_core.retrieval import Retriever, build_branch
from rag_core.schemas import RetrievedChunk
from rag_core.settings import get_settings

#: Cùng seed với `pipeline/eval/compare.py` và `known_item_probe.py`.
DEFAULT_SEED = 20260820

#: `top_n` của DoD — số chunk thực sự đi vào prompt.
DOD_TOP_N = 6


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def _build(config_path: Path, *, base: str, rrf_k: int, candidate_k: int) -> Retriever:
    index_config = load_index_config(config_path)
    settings = get_settings()
    embeddings = index_config.build_embeddings()
    store = index_config.build_retriever(
        embeddings,
        url=settings.qdrant_url,
        api_key=(settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None),
    )
    store.verify_schema()
    options: dict[str, Any] = {}
    if base == "hybrid":
        options = {"k": rrf_k, "candidate_k": candidate_k}
    return build_branch(store, base, **options)


def measure_ceiling(
    branch: Retriever, queries: Sequence[GoldenQuery], depths: Sequence[int]
) -> tuple[dict[int, float], list[list[RetrievedChunk]]]:
    """`hit_rate@depth` của nhánh nền — trần cứng của mọi phép xếp lại sau nó.

    Trả luôn pool đã lấy để phần đo sau không phải truy vấn Qdrant lần nữa: mọi
    con số trong lần chạy này phải nói về **cùng một** tập ứng viên, nếu không thì
    "trần" và "truncation" đang nói về hai thứ khác nhau.
    """
    max_depth = max(depths)
    pools: list[list[RetrievedChunk]] = []
    hits = {depth: 0 for depth in depths}
    for query in queries:
        pool = branch.retrieve(query.query, max_depth)
        pools.append(pool)
        relevant = set(query.relevant_chunk_ids)
        if not relevant:
            continue
        ids = [hit.chunk.chunk_id for hit in pool]
        for depth in depths:
            if relevant & set(ids[:depth]):
                hits[depth] += 1
    scored = sum(1 for query in queries if query.relevant_chunk_ids)
    return {depth: count / scored for depth, count in hits.items()}, pools


def measure_truncation(
    reranker: CrossEncoderReranker,
    queries: Sequence[GoldenQuery],
    pools: Sequence[Sequence[RetrievedChunk]],
) -> dict[str, float]:
    counts: list[int] = []
    for query, pool in zip(queries, pools, strict=True):
        counts.extend(reranker.count_pair_tokens(query.query, [h.chunk.content for h in pool]))
    if not counts:
        return {}
    over = sum(1 for value in counts if value > reranker.max_length)
    return {
        "pairs": float(len(counts)),
        "max_length": float(reranker.max_length),
        "p50_tokens": float(statistics.median(counts)),
        "p95_tokens": float(_percentile([float(c) for c in counts], 0.95)),
        "max_tokens": float(max(counts)),
        "truncated_ratio": over / len(counts),
    }


def measure_saturation(
    reranker: CrossEncoderReranker,
    queries: Sequence[GoldenQuery],
    pools: Sequence[Sequence[RetrievedChunk]],
    *,
    limit: int,
) -> dict[str, float]:
    """Logit nào sẽ mất phân biệt nếu áp sigmoid trong float32.

    `sigmoid(x)` chạm 1.0 của float32 từ khoảng `x ≈ 16,64` và chạm 0.0 từ
    khoảng `x ≈ -103`. Ngưỡng trên là ngưỡng đáng lo: hai chunk logit 18 và 25
    đều thành 1,0, tức một tie nhân tạo ở đúng chỗ quan trọng nhất — top của
    danh sách.
    """
    logits: list[float] = []
    for query, pool in list(zip(queries, pools, strict=True))[:limit]:
        logits.extend(reranker.score(query.query, [h.chunk.content for h in pool]))
    if not logits:
        return {}
    high = sum(1 for value in logits if value > 16.64)
    low = sum(1 for value in logits if value < -103.0)
    return {
        "logits": float(len(logits)),
        "min": min(logits),
        "max": max(logits),
        "p50": float(statistics.median(logits)),
        "would_saturate_high_ratio": high / len(logits),
        "would_saturate_low_ratio": low / len(logits),
    }


def measure_latency(
    reranker: CrossEncoderReranker,
    queries: Sequence[str],
    pools: Sequence[Sequence[RetrievedChunk]],
    *,
    pool_size: int,
    rounds: int,
    sample: int,
) -> dict[str, float]:
    """Chỉ phần **rerank**, tách khỏi phần truy hồi.

    Tách vì `W2-04` học được rằng đo cả cục thì không quy kết được gì: ba harness
    khác cấu trúc cho ba câu trả lời lệch nhau 2–6×, và bug 64 ms chỉ hiện ra khi
    các phần buộc phải cộng lại đúng. Ở đây phần truy hồi đã đo ở `W2-04`
    (hybrid p50 31,3 ms), nên cái còn thiếu là đúng phần này.
    """
    work = [
        (query, [hit.chunk.content for hit in pool[:pool_size]])
        for query, pool in list(zip(queries, pools, strict=True))[:sample]
        if pool
    ]
    for query, texts in work[:3]:
        reranker.score(query, texts)  # warm-up: lần đầu gồm cả nạp kernel CUDA

    samples: list[float] = []
    for _ in range(rounds):
        for query, texts in work:
            started = time.perf_counter()
            reranker.score(query, texts)
            samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "pool_size": float(pool_size),
        "batch_size": float(reranker.batch_size),
        "n": float(len(samples)),
        "p50_ms": float(statistics.median(samples)),
        "p95_ms": _percentile(samples, 0.95),
        "max_ms": max(samples),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/indexing/bgem3.yaml"))
    parser.add_argument("--golden", type=Path, default=Path("data/golden/golden_v1.jsonl"))
    parser.add_argument("--base", default="hybrid", choices=["dense", "sparse", "hybrid"])
    parser.add_argument("--rrf-k", type=int, default=1, help="Cấu hình thắng của `W2-04`.")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--pool", type=int, default=50, help="Độ sâu pool = trần của reranker.")
    parser.add_argument("--device", default="auto", help="`cuda`/`cpu`; `cpu` để đo fallback.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="`auto` = fp16 trên CUDA. Truyền `float32` để lấy số nền so sánh.",
    )
    parser.add_argument(
        "--saturation-sample",
        type=int,
        default=40,
        help="Số truy vấn dùng cho phép đo logit (mỗi câu tốn một lượt rerank).",
    )
    parser.add_argument("--latency-sample", type=int, default=30)
    parser.add_argument("--latency-rounds", type=int, default=3)
    parser.add_argument(
        "--latency-only",
        action="store_true",
        help="Bỏ qua Qdrant và golden set — chỉ đo độ trễ trên text tự sinh. Dùng "
        "để đo CPU fallback mà không phải chờ nhánh nền chạy trên CPU.",
    )
    parser.add_argument("--report", type=Path, help="Ghi kết quả ra JSON.")
    args = parser.parse_args(argv)

    reranker = CrossEncoderReranker(
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=args.dtype,
    )
    report: dict[str, Any] = {
        "reranker": reranker.name,
        "device": reranker.device,
        "seed": DEFAULT_SEED,
    }

    if args.latency_only:
        # Text tự sinh dài xấp xỉ chunk thật (1000 ký tự) để phép đo không lạc quan
        # hơn thực tế chỉ vì đầu vào ngắn hơn.
        filler = "Ngân sách nhà nước đầu tư công cho hạ tầng giao thông năm 2024. "
        fake = [
            "Tăng trưởng GDP của Việt Nam năm 2023 là bao nhiêu phần trăm?"
        ] * args.latency_sample
        pools = [_synthetic_pool(filler, args.pool) for _ in fake]
        report["latency"] = {
            str(size): measure_latency(
                reranker,
                fake,
                pools,
                pool_size=size,
                rounds=args.latency_rounds,
                sample=args.latency_sample,
            )
            for size in (DOD_TOP_N, 20, args.pool)
        }
        report["note"] = "latency-only: text tự sinh, không phải corpus thật"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.report:
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
        return 0

    branch = _build(args.config, base=args.base, rrf_k=args.rrf_k, candidate_k=args.candidate_k)
    report["base_branch"] = branch.name
    loaded = load_golden_set(args.golden)
    queries, resolution = _resolve_span_labels(branch, loaded, DEFAULT_MIN_OVERLAP_RATIO)
    if resolution is not None:
        report["span_resolution"] = json.loads(resolution.model_dump_json())

    depths = sorted({1, 5, 10, 20, args.pool})
    ceiling, pools = measure_ceiling(branch, queries, depths)
    report["ceiling_hit_rate"] = {f"@{depth}": value for depth, value in ceiling.items()}
    report["truncation"] = measure_truncation(reranker, queries, pools)
    report["saturation"] = measure_saturation(
        reranker, queries, pools, limit=args.saturation_sample
    )
    report["latency"] = {
        str(size): measure_latency(
            reranker,
            [query.query for query in queries],
            pools,
            pool_size=size,
            rounds=args.latency_rounds,
            sample=args.latency_sample,
        )
        for size in (DOD_TOP_N, 20, args.pool)
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    return 0


def _synthetic_pool(filler: str, size: int) -> list[RetrievedChunk]:
    from rag_core.schemas import Chunk, RetrievalMode

    return [
        RetrievedChunk(
            chunk=Chunk(
                chunk_id=f"synthetic::{index:05d}",
                doc_id="synthetic",
                content=(filler * 16)[:1000],
                chunk_index=index,
            ),
            score=1.0 - 0.001 * index,
            rank=index + 1,
            mode=RetrievalMode.HYBRID,
        )
        for index in range(size)
    ]


if __name__ == "__main__":
    raise SystemExit(main())
