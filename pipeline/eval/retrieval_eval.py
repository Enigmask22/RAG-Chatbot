"""Chạy metric truy hồi trên golden set và xuất báo cáo Markdown + JSON.

Thiết kế cố ý tách làm hai lớp:

* `evaluate_run` nhận **kết quả đã truy hồi sẵn** (`dict[query_id, list[chunk_id]]`).
  Không chạm mạng, không chạm model → test được bằng fixture tính tay.
* `run_retrieval_eval` mới là lớp gọi retriever thật.

Tách như vậy để bài test của metric không phụ thuộc Qdrant, và để so lại số cũ
mà không cần chạy lại retrieval (chỉ cần file kết quả thô).
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_core.retrieval.base import Retriever

from .golden import GoldenQuery, load_golden_set
from .metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from .spans import (
    DEFAULT_MIN_OVERLAP_RATIO,
    QueryResolution,
    chunks_by_document,
    resolve_queries,
)

logger = logging.getLogger(__name__)

__all__ = ["EvalReport", "QueryScore", "evaluate_run", "run_retrieval_eval"]

DEFAULT_K_VALUES = (1, 5, 10, 20)


@dataclass(frozen=True)
class GroupMetrics:
    n_queries: int
    metrics: dict[str, float]


@dataclass
class QueryScore:
    """Điểm của **một** truy vấn — thứ cần để kiểm định cặp giữa hai lần chạy.

    Không có tầng này thì không so được hai config nào một cách có ý nghĩa:
    chênh 0,2153 vs 0,2010 ở `hit_rate@5` trên 209 câu là **3 câu**, và không
    ai biết đó là cải thiện hay là nhiễu. `pipeline/eval/compare.py` đọc chính
    file này để chạy McNemar và bootstrap cặp.

    `n_relevant` có mặt vì nó là **cái bẫy** phát hiện ở `TD-11`: nhãn neo theo
    span nên đổi `chunk_size` làm số nhãn mỗi câu thay đổi, và recall@k có mẫu
    số là chính con số đó. Ghi lại để công cụ so sánh từ chối so recall khi hai
    lần chạy có phân bố nhãn khác nhau.
    """

    query_id: str
    category: str
    lang: str
    n_relevant: int
    n_retrieved: int
    scores: dict[str, float]


@dataclass
class EvalReport:
    run_name: str
    created_at: str
    n_queries: int
    n_scored: int
    n_skipped_unanswerable: int
    overall: dict[str, float]
    by_category: dict[str, GroupMetrics] = field(default_factory=dict)
    by_language: dict[str, GroupMetrics] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    per_query: list[QueryScore] = field(default_factory=list)

    def to_jsonl(self) -> str:
        """Một dòng JSON cho mỗi truy vấn — định dạng để so sánh và để diff."""
        return "\n".join(
            json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) for row in self.per_query
        )

    def to_json(self) -> str:
        payload = asdict(self)
        payload["by_category"] = {k: asdict(v) for k, v in self.by_category.items()}
        payload["by_language"] = {k: asdict(v) for k, v in self.by_language.items()}
        # Điểm từng câu đi ra file JSONL riêng: 209 dòng × 15 metric làm file
        # tóm tắt không đọc được nữa, mà file tóm tắt là thứ người ta mở trước.
        payload.pop("per_query", None)
        payload["n_relevant_mean"] = round(
            _mean([float(row.n_relevant) for row in self.per_query]), 4
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def to_markdown(self) -> str:
        lines: list[str] = [
            f"# Retrieval eval — `{self.run_name}`",
            "",
            f"- Thời điểm chạy: `{self.created_at}`",
            f"- Số truy vấn: **{self.n_queries}** "
            f"(chấm điểm {self.n_scored}, bỏ qua {self.n_skipped_unanswerable} câu unanswerable)",
        ]
        if self.config:
            lines.append(f"- Config: `{json.dumps(self.config, ensure_ascii=False)}`")
        if self.environment:
            env = ", ".join(f"{k}={v}" for k, v in sorted(self.environment.items()))
            lines.append(f"- Môi trường: {env}")

        lines += ["", "## Tổng thể", "", "| Metric | Giá trị |", "|---|---:|"]
        lines += [f"| {name} | {value:.4f} |" for name, value in sorted(self.overall.items())]

        if self.latency_ms:
            lines += ["", "## Độ trễ truy hồi (ms)", "", "| Phân vị | ms |", "|---|---:|"]
            lines += [f"| {name} | {value:.1f} |" for name, value in self.latency_ms.items()]

        for title, groups in (
            ("Theo nhóm truy vấn", self.by_category),
            ("Theo ngôn ngữ", self.by_language),
        ):
            if not groups:
                continue
            metric_names = sorted({m for g in groups.values() for m in g.metrics})
            header = "| Nhóm | n | " + " | ".join(metric_names) + " |"
            sep = "|---|---:|" + "---:|" * len(metric_names)
            lines += ["", f"## {title}", "", header, sep]
            for name, group in sorted(groups.items()):
                cells = " | ".join(
                    f"{group.metrics.get(m, float('nan')):.4f}" for m in metric_names
                )
                lines.append(f"| {name} | {group.n_queries} | {cells} |")

        lines += [
            "",
            "> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi",
            "> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).",
            "",
        ]
        return "\n".join(lines)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _score_one(
    retrieved: Sequence[str], relevant: Sequence[str], k_values: Sequence[int]
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for k in k_values:
        for label, value in (
            (f"recall@{k}", recall_at_k(retrieved, relevant, k)),
            (f"precision@{k}", precision_at_k(retrieved, relevant, k)),
            (f"hit_rate@{k}", hit_rate_at_k(retrieved, relevant, k)),
        ):
            if value is not None:
                scores[label] = value
    max_k = max(k_values)
    ndcg = ndcg_at_k(retrieved, relevant, 10 if 10 in k_values else max_k)
    if ndcg is not None:
        scores[f"ndcg@{10 if 10 in k_values else max_k}"] = ndcg
    ap = average_precision_at_k(retrieved, relevant, max_k)
    if ap is not None:
        scores[f"map@{max_k}"] = ap
    rr = reciprocal_rank(retrieved, relevant, max_k)
    if rr is not None:
        scores["mrr"] = rr
    return scores


def _aggregate(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    names = sorted({name for row in rows for name in row})
    return {name: _mean([row[name] for row in rows if name in row]) for name in names}


def evaluate_run(
    queries: Sequence[GoldenQuery],
    retrieved_by_query: Mapping[str, Sequence[str]],
    *,
    run_name: str = "unnamed",
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    latencies_ms: Mapping[str, float] | None = None,
    config: dict[str, Any] | None = None,
) -> EvalReport:
    """Chấm điểm một lần chạy đã có sẵn kết quả truy hồi.

    Thiếu `query_id` trong `retrieved_by_query` được coi là **truy hồi rỗng** chứ
    không phải bỏ qua: retriever không trả gì cũng là một kết quả, và im lặng bỏ
    qua sẽ làm điểm cao lên một cách sai.
    """
    per_query: list[tuple[GoldenQuery, dict[str, float]]] = []
    rows: list[QueryScore] = []
    skipped = 0
    for query in queries:
        if not query.relevant_chunk_ids:
            skipped += 1
            continue
        retrieved = list(retrieved_by_query.get(query.query_id, []))
        scores = _score_one(retrieved, query.relevant_chunk_ids, k_values)
        per_query.append((query, scores))
        rows.append(
            QueryScore(
                query_id=query.query_id,
                category=str(query.category.value),
                lang=str(query.lang.value),
                n_relevant=len(query.relevant_chunk_ids),
                n_retrieved=len(retrieved),
                scores=scores,
            )
        )

    overall = _aggregate([scores for _, scores in per_query])

    def group_by(key: str) -> dict[str, GroupMetrics]:
        buckets: dict[str, list[dict[str, float]]] = {}
        for query, scores in per_query:
            name = str(
                getattr(query, key).value
                if hasattr(getattr(query, key), "value")
                else getattr(query, key)
            )
            buckets.setdefault(name, []).append(scores)
        return {
            name: GroupMetrics(n_queries=len(rows), metrics=_aggregate(rows))
            for name, rows in buckets.items()
        }

    latency_summary: dict[str, float] = {}
    if latencies_ms:
        values = sorted(latencies_ms.values())
        latency_summary = {
            "mean": _mean(values),
            "p50": values[len(values) // 2],
            "p95": values[min(len(values) - 1, int(len(values) * 0.95))],
            "max": values[-1],
        }
        if len(values) > 1:
            latency_summary["stdev"] = statistics.stdev(values)

    return EvalReport(
        run_name=run_name,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        n_queries=len(queries),
        n_scored=len(per_query),
        n_skipped_unanswerable=skipped,
        overall=overall,
        by_category=group_by("category"),
        by_language=group_by("lang"),
        latency_ms=latency_summary,
        config=config or {},
        environment={"python": platform.python_version(), "platform": platform.platform()},
        per_query=rows,
    )


def run_retrieval_eval(
    retriever: Retriever,
    queries: Sequence[GoldenQuery],
    *,
    run_name: str,
    top_k: int = 20,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    config: dict[str, Any] | None = None,
    warmup: bool = True,
) -> EvalReport:
    """Chạy eval bằng retriever thật, đo cả độ trễ.

    `warmup=True` gọi một truy vấn bỏ đi trước khi bấm giờ. Không có nó thì lần
    truy vấn đầu tiên gánh cả việc nạp model embedding (~15 giây đo được trên
    máy này, so với p50 là 31 ms) và p95 trở thành số vô nghĩa — mà p95 chính
    là ngưỡng của gate hiệu năng ở W5/W6.
    """
    if warmup and queries:
        retriever.retrieve(queries[0].query, top_k=top_k)

    retrieved: dict[str, list[str]] = {}
    latencies: dict[str, float] = {}
    for query in queries:
        started = time.perf_counter()
        results = retriever.retrieve(query.query, top_k=top_k)
        latencies[query.query_id] = (time.perf_counter() - started) * 1000.0
        retrieved[query.query_id] = [r.chunk.chunk_id for r in results]

    full_config = {"retriever": retriever.name, "top_k": top_k, **(config or {})}
    return evaluate_run(
        queries,
        retrieved,
        run_name=run_name,
        k_values=k_values,
        latencies_ms=latencies,
        config=full_config,
    )


def _eval_against_index(
    index_config_path: Path,
    queries: Sequence[GoldenQuery],
    *,
    run_name: str,
    top_k: int,
    min_overlap_ratio: float = DEFAULT_MIN_OVERLAP_RATIO,
) -> EvalReport:
    """Dựng retriever từ chính config đã build index rồi chạy eval.

    Dùng lại **cùng một file config** thay vì khai báo model/collection lần nữa
    ở phía eval. Nếu hai bên khai báo riêng thì sớm muộn cũng lệch — eval sẽ đo
    một index được build bằng model khác, và không có gì báo lỗi cả.
    """
    # Import cục bộ: eval chấm điểm từ file `--retrieved` không cần qdrant-client.
    from rag_core.settings import get_settings

    from ..indexing.config import load_index_config

    index_config = load_index_config(index_config_path)
    settings = get_settings()
    embeddings = index_config.build_embeddings()
    retriever = index_config.build_retriever(
        embeddings,
        url=settings.qdrant_url,
        api_key=(settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None),
    )

    config: dict[str, Any] = {
        "index_config": str(index_config_path),
        "index_fingerprint": index_config.fingerprint,
        "collection": index_config.collection_name,
        "embedding_model": index_config.embedding_model,
        "chunking": json.loads(index_config.chunking.model_dump_json()),
    }

    # Nhãn theo span được tính lại theo ĐÚNG index đang đo. Đây là chỗ golden set
    # thoát khỏi việc phụ thuộc cấu hình chunking (`TD-12`): `relevant_chunk_ids`
    # ghi trong file chỉ đúng với cấu hình lúc gán nhãn, còn span thì luôn đúng.
    queries, resolution = _resolve_span_labels(retriever, queries, min_overlap_ratio)
    if resolution is not None:
        config["span_resolution"] = json.loads(resolution.model_dump_json())

    return run_retrieval_eval(
        retriever,
        queries,
        run_name=run_name,
        top_k=top_k,
        config=config,
    )


def _resolve_span_labels(
    retriever: Retriever,
    queries: Sequence[GoldenQuery],
    min_overlap_ratio: float,
) -> tuple[Sequence[GoldenQuery], QueryResolution | None]:
    """Tính lại nhãn từ span, nếu golden set có span và retriever đọc được chunk.

    Trả `(queries, None)` khi không có gì để làm — cố ý không im lặng bỏ qua
    trường hợp *có* span mà retriever không lấy được chunk: chỗ đó phải cảnh báo,
    vì recall sẽ được đo bằng nhãn cũ mà người đọc report lại tưởng là nhãn span.
    """
    doc_ids = sorted({s.doc_id for q in queries for s in q.relevant_spans})
    if not doc_ids:
        return queries, None

    fetch = getattr(retriever, "fetch_doc_chunks", None)
    if fetch is None:
        logger.warning(
            "Golden set có span nhưng %s không có `fetch_doc_chunks` — chấm điểm "
            "bằng `relevant_chunk_ids` sẵn có, tức nhãn chỉ đúng nếu index này "
            "được build bằng đúng cấu hình chunking lúc gán nhãn.",
            type(retriever).__name__,
        )
        return queries, None

    by_doc = chunks_by_document(fetch(doc_ids))
    resolved, resolution = resolve_queries(queries, by_doc, min_overlap_ratio=min_overlap_ratio)
    logger.info(
        "Ánh xạ span → chunk: %d câu tính lại · %d câu giữ nhãn cũ · %d câu đổi nhãn",
        resolution.resolved,
        resolution.kept_chunk_ids,
        resolution.label_changed,
    )
    if resolution.unmatched_queries:
        logger.warning(
            "%d câu có span nhưng không khớp chunk nào của index này: %s. "
            "Chúng bị chấm bằng nhãn cũ — cấu hình chunking hiện tại cắt bằng "
            "chứng quá vụn.",
            len(resolution.unmatched_queries),
            ", ".join(resolution.unmatched_queries[:5]),
        )
    return resolved, resolution


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chạy eval retrieval trên golden set")
    parser.add_argument("--golden", type=Path, default=Path("data/golden/golden_v1.jsonl"))
    parser.add_argument("--retrieved", type=Path, help="JSON {query_id: [chunk_id, ...]} đã có sẵn")
    parser.add_argument(
        "--index-config",
        type=Path,
        help="YAML config index (configs/indexing/*.yaml) — truy hồi trực tiếp từ Qdrant",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=DEFAULT_MIN_OVERLAP_RATIO,
        help="Phần chồng nhau tối thiểu khi ánh xạ `relevant_spans` sang chunk_id "
        "của index đang đo. Xét theo cả hai phía (span và chunk).",
    )
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--out-dir", type=Path, default=Path("plans/reports"))
    args = parser.parse_args(argv)

    if not args.golden.exists():
        parser.error(
            f"Không thấy golden set tại {args.golden}. "
            "Golden set được tạo ở W1-10/W1-11 — chưa có thì chưa chạy eval được."
        )

    queries = load_golden_set(args.golden)

    if args.index_config is not None:
        report = _eval_against_index(
            args.index_config,
            queries,
            run_name=args.run_name,
            top_k=args.top_k,
            min_overlap_ratio=args.min_overlap_ratio,
        )
    elif args.retrieved is not None:
        retrieved = json.loads(args.retrieved.read_text(encoding="utf-8"))
        report = evaluate_run(queries, retrieved, run_name=args.run_name)
    else:
        parser.error(
            "Cần một trong hai: `--index-config` để truy hồi thật từ Qdrant, "
            "hoặc `--retrieved` với file kết quả đã có sẵn để chấm lại."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.out_dir / f"{args.run_name}-retrieval.md"
    json_path = args.out_dir / f"{args.run_name}-retrieval.json"
    jsonl_path = args.out_dir / f"{args.run_name}-per-query.jsonl"
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    json_path.write_text(report.to_json(), encoding="utf-8")
    jsonl_path.write_text(report.to_jsonl() + "\n", encoding="utf-8")
    print(f"Đã ghi {md_path}, {json_path} và {jsonl_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
