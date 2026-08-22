"""Đo small-to-big trên index thật: gộp trùng được bao nhiêu, ngữ cảnh nở mấy lần.

Ba câu hỏi mà unit test không trả lời được:

1. Top-k child thật sự **chụm** vào bao nhiêu parent? (Nếu k child luôn rơi vào k
   parent khác nhau thì việc gộp trùng là code chết.)
2. Mở rộng sang parent làm prompt phình lên mấy lần — tính bằng **token**, vì đó
   mới là thứ trả tiền?
3. Có bao nhiêu parent bị **thiếu anh em**? Trên index sạch phải là 0; khác 0 là
   dấu hiệu index và `extra["parent_children"]` lệch nhau.

⚠️ Script này **không** đo chất lượng truy hồi. Bộ chunk của `pc256` khác baseline
nên nhãn golden ánh xạ khác — xem `TD-20`/`TD-25`. Ở đây chỉ đo **cấu trúc**, thứ
không phụ thuộc nhãn.

Dùng:

    uv run python scripts/parent_child_probe.py --config configs/indexing/pc256.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from rag_core.retrieval import expand_to_parents

logger = logging.getLogger("parent_child_probe")

DEFAULT_GOLDEN = Path("data/golden/golden_v1.jsonl")


@dataclass(frozen=True)
class QueryOutcome:
    children: int
    parents: int
    child_tokens: int
    parent_tokens: int
    incomplete: int

    @property
    def dedupe_ratio(self) -> float:
        """1 − parent/child. 0 = không gộp được gì; 0,5 = k child rơi vào k/2 parent."""
        return 1.0 - self.parents / self.children if self.children else 0.0

    @property
    def token_growth(self) -> float:
        return self.parent_tokens / self.child_tokens if self.child_tokens else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(pct / 100 * len(ordered)))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/indexing/pc256.yaml"))
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="chỉ chạy N câu đầu")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from pipeline.eval.golden import load_golden_set
    from pipeline.eval.retrieval_eval import open_index

    session = open_index(args.config)
    queries = load_golden_set(args.golden)
    if args.limit:
        queries = queries[: args.limit]
    logger.info("%d câu · top_k=%d · %s", len(queries), args.top_k, session.config.collection)

    outcomes: list[QueryOutcome] = []
    for query in queries:
        results = session.store.retrieve(query.query, top_k=args.top_k)
        if not results:
            continue
        parents = expand_to_parents(results, session.store)
        child_texts = [item.chunk.content for item in results]
        parent_texts = [p.text for p in parents]
        counts = session.embeddings.count_tokens(child_texts + parent_texts) or []
        split = len(child_texts)
        outcomes.append(
            QueryOutcome(
                children=len(results),
                parents=len(parents),
                child_tokens=sum(counts[:split]),
                parent_tokens=sum(counts[split:]),
                incomplete=sum(1 for p in parents if not p.complete),
            )
        )

    if not outcomes:
        logger.error("Không câu nào trả về kết quả — index rỗng?")
        return 1

    dedupe = [o.dedupe_ratio for o in outcomes]
    growth = [o.token_growth for o in outcomes]
    summary = {
        "collection": session.config.collection,
        "queries": len(outcomes),
        "top_k": args.top_k,
        "parents_mean": round(statistics.mean(o.parents for o in outcomes), 2),
        "parents_min": min(o.parents for o in outcomes),
        "dedupe_ratio_mean": round(statistics.mean(dedupe), 4),
        "dedupe_ratio_p50": round(percentile(dedupe, 50), 4),
        "queries_with_any_dedupe": sum(1 for o in outcomes if o.parents < o.children),
        "child_tokens_mean": round(statistics.mean(o.child_tokens for o in outcomes), 1),
        "parent_tokens_mean": round(statistics.mean(o.parent_tokens for o in outcomes), 1),
        "token_growth_mean": round(statistics.mean(growth), 3),
        "token_growth_p95": round(percentile(growth, 95), 3),
        "parents_incomplete": sum(o.incomplete for o in outcomes),
    }
    for key, value in summary.items():
        logger.info("%-26s %s", key, value)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("→ %s", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
