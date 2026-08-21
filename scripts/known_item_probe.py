"""Known-item search — bằng chứng cho DoD của `W2-03`, không cần nhãn người.

DoD của `W2-03` là: "truy vấn từ khoá lạ (mã số, tên riêng) mà dense miss thì
sparse hit". `golden_v1` không trả lời được câu đó — 209 câu đều là câu hỏi tự
nhiên do LLM sinh, không có câu nào tra mã. Và một corpus tổng hợp bảy chunk cũng
không trả lời được: dense chỉ có sáu đối thủ nên nó tra đúng mã một cách dễ dàng
(đo thật, xem `tests/integration/test_sparse_retriever.py`).

Cách đo ở đây dựa vào một tính chất của bài toán known-item: nếu truy vấn là một
chuỗi **xuất hiện nguyên văn** trong corpus thì "đúng" kiểm được bằng **so
chuỗi** — một kết quả đúng khi nội dung của nó chứa chính chuỗi đó. Không cần
người gán nhãn, và không có chỗ cho phán đoán chen vào.

Chọn từ khoá: mã dạng chữ-in-hoa + số (`P171645`, `TF097373`, `SEDP-2016-2020` —
project ID, trust fund ID, tên chương trình của World Bank), giữ lại mã xuất hiện
ở **1–3 chunk**. Nhiều hơn thì nó là mã của cả một tài liệu và bài toán trở thành
"tìm tài liệu", không còn là known-item.

Chạy:

    python scripts/known_item_probe.py --config configs/indexing/bgem3.yaml \\
        --report plans/reports/probes/w2-03-known-item.json
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import random
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_core.retrieval import QdrantDenseRetriever
    from rag_core.schemas import RetrievedChunk

logger = logging.getLogger("known_item_probe")

DEFAULT_SEED = 20260820
DEFAULT_TOP_K = 10
DEFAULT_MAX_QUERIES = 120

#: Mã tài liệu/dự án: chữ in hoa kèm số. Hai dạng — liền (`P171645`, `TF097373`)
#: và có phân cách (`SEDP-2016-2020`, `VIE/02/009`).
CODE_PATTERN = re.compile(r"\b(?:[A-Z]{1,4}\d{4,7}|[A-Z]{2,5}[-/]\d{2,5}(?:[-/][A-Z0-9]{1,5})?)\b")

#: Số chunk tối đa mà một mã được xuất hiện để còn tính là "lạ".
MAX_DOC_FREQUENCY = 3

#: Mã quá ngắn (`VN01`) trùng với vô số thứ khác, không phải known-item.
MIN_CODE_LENGTH = 6


@dataclass(frozen=True)
class ProbeRow:
    """Một truy vấn known-item và thứ hạng mà mỗi nhánh đặt bằng chứng vào."""

    code: str
    doc_frequency: int
    dense_rank: int | None
    sparse_rank: int | None
    hybrid_rank: int | None
    #: `None` khi không chạy nhánh reranked (`--rerank`), khác hẳn với "chạy mà
    #: không tìm ra" — nên `reranked_returned` là chỗ phân biệt hai ca đó.
    reranked_rank: int | None = None
    dense_returned: int = 0
    sparse_returned: int = 0
    hybrid_returned: int = 0
    reranked_returned: int = 0


def scan_contents(store: QdrantDenseRetriever, *, batch: int = 2048) -> dict[str, str]:
    """Đọc nội dung mọi chunk trong collection. Cần vì tiêu chí đúng là so chuỗi."""
    from rag_core.schemas import Chunk

    contents: dict[str, str] = {}
    offset: Any = None
    while True:
        points, offset = store.client.scroll(
            collection_name=store.collection,
            limit=batch,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            raw = (point.payload or {}).get("chunk")
            if raw is not None:
                chunk = Chunk.model_validate(raw)
                contents[chunk.chunk_id] = chunk.content
        if offset is None:
            return contents


def code_frequency(contents: dict[str, str]) -> collections.Counter[str]:
    """Số **chunk** mà mỗi mã xuất hiện (không phải số lần xuất hiện)."""
    frequency: collections.Counter[str] = collections.Counter()
    for text in contents.values():
        frequency.update(set(CODE_PATTERN.findall(text)))
    return frequency


def rare_codes(frequency: collections.Counter[str]) -> list[str]:
    """Mã đủ lạ để bài toán còn là known-item. Xem `MAX_DOC_FREQUENCY`."""
    return sorted(
        code
        for code, n in frequency.items()
        if 1 <= n <= MAX_DOC_FREQUENCY and len(code) >= MIN_CODE_LENGTH
    )


def _rank_of(code: str, hits: Sequence[RetrievedChunk]) -> int | None:
    """Hạng của kết quả đúng đầu tiên, hoặc `None` nếu không có trong danh sách.

    "Đúng" = nội dung chứa chính chuỗi truy vấn. Cố ý **không** so `chunk_id` với
    một chunk nguồn định trước: một mã xuất hiện ở 2–3 chunk thì cả 2–3 đều là câu
    trả lời đúng, và chấm chỉ một cái là hạ điểm cả hai nhánh một cách vô cớ.
    """
    for hit in hits:
        if code in hit.chunk.content:
            return hit.rank
    return None


def summarise(ranks: Sequence[int | None], *, n: int) -> dict[str, float]:
    found = [r for r in ranks if r is not None]
    return {
        "hit@k": len(found) / n if n else float("nan"),
        "hit@1": sum(1 for r in found if r == 1) / n if n else float("nan"),
        "mrr": sum(1.0 / r for r in found) / n if n else float("nan"),
        "median_rank": float(statistics.median(found)) if found else float("nan"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--config", type=Path, default=Path("configs/indexing/bgem3.yaml"))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-queries", type=int, default=DEFAULT_MAX_QUERIES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rrf-k", type=int, help="`k` của RRF cho nhánh hybrid (mặc định 60)")
    parser.add_argument(
        "--candidate-k", type=int, help="Số ứng viên mỗi nhánh cho hybrid (mặc định 50)"
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Đo thêm nhánh reranked (`W2-05`). Opt-in vì nó nạp 2,2GB trọng số và "
        "chấm 50 cặp cho mỗi mã — câu hỏi nó trả lời là: cross-encoder có lấy lại "
        "được thứ hạng mà RRF làm mất ở đây không?",
    )
    parser.add_argument("--rerank-candidates", type=int, help="Pool cho reranker (mặc định 50)")
    parser.add_argument("--rerank-dtype", help="`auto`/`float16`/`float32` cho reranker")
    parser.add_argument("--report", type=Path, help="Ghi JSON kết quả đầy đủ")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(message)s")
    # Mỗi truy vấn là hai request HTTP; ở mức INFO thì httpx in ra 2×51 dòng và
    # bảng kết quả trôi mất khỏi màn hình.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from rag_core.retrieval import QdrantHybridRetriever, QdrantSparseRetriever, build_branch
    from rag_core.settings import get_settings

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.eval.compare import mcnemar_exact
    from pipeline.indexing.config import load_index_config

    config = load_index_config(args.config)
    settings = get_settings()
    store = config.build_retriever(
        config.build_embeddings(),
        url=settings.qdrant_url,
        api_key=(settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None),
    )
    store.verify_schema()
    sparse = build_branch(store, "sparse")
    assert isinstance(sparse, QdrantSparseRetriever)
    # `W2-04`: RRF có **giữ** được chỗ sparse thắng hay pha loãng nó? Câu hỏi này
    # đo được ở đây và không đo được trên `golden_v1` (209 câu không có câu tra mã).
    hybrid = build_branch(store, "hybrid", k=args.rrf_k, candidate_k=args.candidate_k)
    assert isinstance(hybrid, QdrantHybridRetriever)
    # `W2-05`: `W2-04` §8 đo được RRF trọng số đều làm hit@1 tra mã tụt 0,3529 →
    # 0,0980. Reranker có lấy lại được chỗ đó không? Dự đoán ghi trước: **không**,
    # vì cross-encoder cũng là model subword cùng họ nên nó gặp đúng `TD-18`.
    reranked = (
        build_branch(
            store,
            "reranked",
            base="hybrid",
            k=args.rrf_k,
            candidate_k=args.candidate_k,
            rerank_candidates=args.rerank_candidates,
            rerank_dtype=args.rerank_dtype,
        )
        if args.rerank
        else None
    )

    logger.info("Đang quét %s...", store.collection)
    contents = scan_contents(store)
    frequency = code_frequency(contents)
    codes = rare_codes(frequency)
    logger.info(
        "%d chunk · %d mã xuất hiện ở 1–%d chunk",
        len(contents),
        len(codes),
        MAX_DOC_FREQUENCY,
    )
    if not codes:
        logger.error("Không có mã nào đủ lạ trong corpus này — không đo được.")
        return 1

    queries = random.Random(args.seed).sample(codes, min(args.max_queries, len(codes)))
    rows: list[ProbeRow] = []
    for i, code in enumerate(queries, 1):
        dense_hits = store.retrieve(code, args.top_k)
        sparse_hits = sparse.retrieve(code, args.top_k)
        hybrid_hits = hybrid.retrieve(code, args.top_k)
        reranked_hits = reranked.retrieve(code, args.top_k) if reranked is not None else []
        rows.append(
            ProbeRow(
                code=code,
                doc_frequency=frequency[code],
                dense_rank=_rank_of(code, dense_hits),
                sparse_rank=_rank_of(code, sparse_hits),
                hybrid_rank=_rank_of(code, hybrid_hits),
                reranked_rank=_rank_of(code, reranked_hits) if reranked is not None else None,
                dense_returned=len(dense_hits),
                sparse_returned=len(sparse_hits),
                hybrid_returned=len(hybrid_hits),
                reranked_returned=len(reranked_hits),
            )
        )
        if i % 20 == 0:
            logger.info("  %d/%d", i, len(queries))

    n = len(rows)
    dense_only = sum(1 for r in rows if r.dense_rank and not r.sparse_rank)
    sparse_only = sum(1 for r in rows if r.sparse_rank and not r.dense_rank)
    payload = {
        "n_queries": n,
        "top_k": args.top_k,
        "seed": args.seed,
        "rrf_k": args.rrf_k,
        "candidate_k": args.candidate_k,
        "collection": store.collection,
        "index_fingerprint": config.fingerprint,
        "dense": summarise([r.dense_rank for r in rows], n=n),
        "sparse": summarise([r.sparse_rank for r in rows], n=n),
        "hybrid": summarise([r.hybrid_rank for r in rows], n=n),
        "retriever_names": {
            "dense": store.name,
            "sparse": sparse.name,
            "hybrid": hybrid.name,
            **({"reranked": reranked.name} if reranked is not None else {}),
        },
        "hybrid_vs_sparse": {
            # Câu hỏi của `W2-04`: hợp nhất có làm mất chỗ sparse thắng không?
            "sparse_only": sum(1 for r in rows if r.sparse_rank and not r.hybrid_rank),
            "hybrid_only": sum(1 for r in rows if r.hybrid_rank and not r.sparse_rank),
            "mcnemar_p": mcnemar_exact(
                sum(1 for r in rows if r.sparse_rank and not r.hybrid_rank),
                sum(1 for r in rows if r.hybrid_rank and not r.sparse_rank),
            ),
        },
        "discordant": {
            "dense_only": dense_only,
            "sparse_only": sparse_only,
            "both": sum(1 for r in rows if r.dense_rank and r.sparse_rank),
            "neither": sum(1 for r in rows if not r.dense_rank and not r.sparse_rank),
            "mcnemar_p": mcnemar_exact(dense_only, sparse_only),
        },
        "rows": [asdict(r) for r in rows],
    }
    if reranked is not None:
        payload["reranked"] = summarise([r.reranked_rank for r in rows], n=n)
        # Hai phép so, và chúng trả lời hai câu khác nhau: so với hybrid là "xếp
        # lại có giúp gì không", so với sparse là "có lấy lại được nhánh đang
        # thắng ở đây không". Câu thứ hai mới là câu của `TD-18`.
        for label, other in (("hybrid", "hybrid_rank"), ("sparse", "sparse_rank")):
            only_other = sum(1 for r in rows if getattr(r, other) and not r.reranked_rank)
            only_reranked = sum(1 for r in rows if r.reranked_rank and not getattr(r, other))
            payload[f"reranked_vs_{label}"] = {
                f"{label}_only": only_other,
                "reranked_only": only_reranked,
                "mcnemar_p": mcnemar_exact(only_other, only_reranked),
            }

    logger.info("─" * 62)
    logger.info("Known-item search · %d truy vấn · top-%d", n, args.top_k)
    branches = (
        ("dense", "sparse", "hybrid", "reranked")
        if reranked
        else (
            "dense",
            "sparse",
            "hybrid",
        )
    )
    for branch in branches:
        stats = payload[branch]
        assert isinstance(stats, dict)
        logger.info(
            "  %-7s hit@%d %.4f · hit@1 %.4f · MRR %.4f · hạng trung vị %s",
            branch,
            args.top_k,
            stats["hit@k"],
            stats["hit@1"],
            stats["mrr"],
            stats["median_rank"],
        )
    disc = payload["discordant"]
    assert isinstance(disc, dict)
    logger.info(
        "  chỉ dense %d · chỉ sparse %d · cả hai %d · không bên nào %d · McNemar p=%.3g",
        disc["dense_only"],
        disc["sparse_only"],
        disc["both"],
        disc["neither"],
        disc["mcnemar_p"],
    )
    hvs = payload["hybrid_vs_sparse"]
    assert isinstance(hvs, dict)
    logger.info(
        "  hybrid vs sparse: chỉ sparse %d · chỉ hybrid %d · McNemar p=%.3g",
        hvs["sparse_only"],
        hvs["hybrid_only"],
        hvs["mcnemar_p"],
    )
    if reranked is not None:
        for label in ("hybrid", "sparse"):
            block = payload[f"reranked_vs_{label}"]
            assert isinstance(block, dict)
            logger.info(
                "  reranked vs %s: chỉ %s %d · chỉ reranked %d · McNemar p=%.3g",
                label,
                label,
                block[f"{label}_only"],
                block["reranked_only"],
                block["mcnemar_p"],
            )
    logger.info("─" * 62)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Đã ghi %s", args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
