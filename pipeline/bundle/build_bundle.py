"""Đóng gói một cấu hình **đã đo** thành `RagBundle` (`W4-01`).

Công việc thật của file này không phải là chép trường từ chỗ này sang chỗ kia —
đó là phần dễ. Nó là **từ chối đóng gói một lời nói dối**.

Một bundle là một phát biểu: *"những con số eval này thuộc về đúng cấu hình
này"*. Phát biểu ấy sai một cách rất tự nhiên và rất im lặng:

* chạy eval trên `rag_bgem3`, rồi đóng gói với `configs/.../bgem3-contextual.yaml`
  vì đó là file mở sẵn trong editor;
* build lại index sau khi eval xong, rồi đóng gói — collection cùng tên, nội
  dung khác;
* đổi một tham số chunking, quên chạy lại eval, đóng gói bằng số cũ.

Cả ba đều cho ra một manifest hợp lệ, ký được, checksum khớp. Nên phép kiểm phải
nằm ở đây, ở chỗ *sinh*, và nó phải so **vân tay** chứ không so tên: tên
collection dùng lại được, vân tay thì không.

Chạy:

    python -m pipeline.bundle.build_bundle \\
        --index-config configs/indexing/bgem3-contextual.yaml \\
        --index-report plans/reports/probes/index-bgem3-contextual.json \\
        --eval-run plans/reports/runs/bgem3-ctx-rr-c50-retrieval.json \\
        --generator deepseek-chat@2026-09 \\
        --version 0.1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipeline.indexing.config import IndexConfig, load_index_config
from rag_core.bundle import (
    BundleComponents,
    BundleValidationError,
    ChunkingComponent,
    EmbeddingComponent,
    EvalReport,
    GateRecord,
    GateStatus,
    IndexComponent,
    RagBundle,
    RerankComponent,
    RetrievalComponent,
    save_bundle,
)

__all__ = ["build_bundle", "main"]

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path("bundles")

#: Tham số reranker không nằm trong `branch_options` của báo cáo eval — chúng bị
#: nén vào chuỗi `retriever`. Đọc lại từ đó thay vì mặc định, vì mặc định ở đây
#: là đúng cái mà `test_bundle.py` cấm.
_RERANK_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=10", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        raise BundleValidationError(
            "không đọc được `git rev-parse HEAD`. Bundle phải truy ngược được về "
            "commit đã sinh ra nó, nên đây là lỗi cứng chứ không phải trường bỏ trống."
        ) from exc
    return out.stdout.strip()


def _parse_rerank(retriever_name: str, branch_options: dict[str, Any]) -> RerankComponent | None:
    """Bóc cấu hình reranker ra khỏi chuỗi `retriever` của báo cáo eval.

    Ví dụ chuỗi: ``reranked[qdrant-hybrid:rag_bgem3_ctx:rrf1-c20]:
    BAAI/bge-reranker-v2-m3@cuda:L512:float16:n50``

    ⚠️ Phân tích chuỗi là cách làm dở, và nó ở đây vì báo cáo eval của `W2` ghi
    tham số reranker vào tên thay vì vào `branch_options` — sửa chỗ đó là sửa
    định dạng của 15 file báo cáo đã công bố. Đổi lại, hàm này **ném** khi không
    bóc được, chứ không lặng lẽ rơi về giá trị mặc định.
    """
    if not retriever_name.startswith("reranked["):
        return None
    tail = retriever_name.split("]:", 1)[1] if "]:" in retriever_name else ""
    parts = tail.split(":")
    model = parts[0].split("@")[0] if parts and parts[0] else _RERANK_DEFAULT_MODEL

    max_length: int | None = None
    candidates: int | None = None
    for piece in parts[1:]:
        if piece.startswith("L") and piece[1:].isdigit():
            max_length = int(piece[1:])
        elif piece.startswith("n") and piece[1:].isdigit():
            candidates = int(piece[1:])
    if max_length is None or candidates is None:
        raise BundleValidationError(
            f"không bóc được `max_length`/`candidates` từ tên nhánh {retriever_name!r}. "
            "Đoán giá trị ở đây sẽ tạo ra một bundle mô tả sai hệ thống đã đo."
        )
    top_n = int(branch_options.get("rerank_top_n", 6))
    return RerankComponent(model=model, candidates=candidates, top_n=top_n, max_length=max_length)


def _check_provenance(
    config: IndexConfig, index_report: dict[str, Any], eval_run: dict[str, Any]
) -> None:
    """Ba nguồn phải nói về **cùng một** index. So vân tay, không so tên.

    Tên collection dùng lại được — `rag_bgem3_ctx` hôm nay và `rag_bgem3_ctx`
    sau khi build lại với `chunk_size` khác là hai index khác nhau mang cùng một
    tên. Vân tay thì không dùng lại được.
    """
    eval_config = eval_run.get("config", {})
    expected = config.fingerprint

    mismatches: list[str] = []
    for source, value in (
        ("báo cáo build index", index_report.get("fingerprint")),
        ("lượt chạy eval", eval_config.get("index_fingerprint")),
    ):
        if value != expected:
            mismatches.append(f"  {source}: {value}")
    if mismatches:
        raise BundleValidationError(
            f"vân tay index không khớp với `{config.name}` ({expected}):\n"
            + "\n".join(mismatches)
            + "\nSố eval thuộc về một index khác với config đang đóng gói. Chạy "
            "lại eval trên index này thay vì đóng gói chéo."
        )

    if eval_config.get("collection") != config.collection:
        raise BundleValidationError(
            f"eval chạy trên collection {eval_config.get('collection')!r} nhưng "
            f"config nói {config.collection!r}."
        )


def build_bundle(
    *,
    config: IndexConfig,
    index_report: dict[str, Any],
    eval_run: dict[str, Any],
    version: str,
    generator: str,
    notes: str | None = None,
) -> RagBundle:
    _check_provenance(config, index_report, eval_run)

    eval_config: dict[str, Any] = eval_run.get("config", {})
    branch_options: dict[str, Any] = dict(eval_config.get("branch_options", {}))
    # `base` mô tả nhánh nền của reranked; nó là `mode`, không phải một option.
    base_mode = str(branch_options.pop("base", eval_config.get("retrieval_mode", "dense")))
    branch_options.pop("rerank_candidates", None)
    branch_options.pop("rerank_top_n", None)

    components = BundleComponents(
        chunking=ChunkingComponent(
            strategy=config.chunking.strategy,
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
            contextual=config.contextual.enabled,
            chunking_fingerprint=config.chunking_fingerprint,
        ),
        embedding=EmbeddingComponent(
            model=config.embedding_model,
            dim=int(index_report["embedding_dim"]),
            normalize=config.embedding_normalize,
        ),
        index=IndexComponent(
            backend="qdrant",
            collection=config.collection,
            fingerprint=config.fingerprint,
            n_chunks=int(index_report["collection_count"]),
            n_documents=int(index_report["n_documents_indexed"]),
        ),
        retrieval=RetrievalComponent(
            mode=base_mode,
            top_k=int(eval_config.get("top_k", 20)),
            options=branch_options,
        ),
        rerank=_parse_rerank(str(eval_config.get("retriever", "")), eval_config),
        # Tầng sinh chưa được dựng (`W4-08`/`W4-11`). Bundle khai thiếu thay vì
        # bịa — xem `BundleComponents.prompt`.
        prompt=None,
        generation=None,
    )

    latency = eval_run.get("latency_ms", {})
    report = EvalReport(
        golden_set=Path(eval_config.get("golden", "golden_v1")).stem,
        n_queries=int(eval_run["n_scored"]),
        evaluated_with_generator=generator,
        retrieval_metrics={str(k): float(v) for k, v in eval_run.get("overall", {}).items()},
        p95_latency_ms=float(latency["p95"]) if "p95" in latency else None,
    )

    return RagBundle(
        bundle_version=version,
        created_at=datetime.now(UTC),
        git_sha=_git_sha(),
        components=components,
        eval=report,
        gate=GateRecord(status=GateStatus.NOT_RUN),
        notes=notes,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-config", type=Path, required=True)
    parser.add_argument("--index-report", type=Path, required=True)
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--version", required=True, help="semver, ví dụ `0.1.0`.")
    parser.add_argument(
        "--generator",
        required=True,
        help="`evaluated_with_generator` — bắt buộc, vì gate chỉ so like-for-like.",
    )
    parser.add_argument("--notes")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    bundle = build_bundle(
        config=load_index_config(args.index_config),
        index_report=json.loads(args.index_report.read_text(encoding="utf-8")),
        eval_run=json.loads(args.eval_run.read_text(encoding="utf-8")),
        version=args.version,
        generator=args.generator,
        notes=args.notes,
    )
    path = save_bundle(bundle, args.root, overwrite=args.overwrite)
    logger.info("đã ghi bundle %s → %s", bundle.bundle_version, path)
    logger.info("serves_generation=%s", bundle.serves_generation)
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
