"""`W3-07` — sửa một dòng trong corpus thật, đo lại phải embed bao nhiêu.

Test tích hợp đã chứng minh **cơ chế** đúng trên corpus dựng sẵn. Script này trả
lời câu hỏi mà test không trả lời được: trên **corpus thật**, với **model thật**,
tiết kiệm được bao nhiêu thời gian đồng hồ.

    uv run python scripts/incremental_probe.py --report plans/reports/probes/w3-07-incremental.json

⚠️ Không đụng `data/corpus/` hay `data/corpus_manifest.csv`: chép sang thư mục
tạm rồi sửa ở đó. Sửa corpus thật là làm hỏng `sha256` của manifest và mọi span
của golden set.

Ba lượt build vào một collection dùng một lần rồi xoá:

1. **sạch** — chưa có state, embed toàn bộ. Đây là mẫu số của `G3`.
2. **không sửa gì** — tầng bỏ-qua-theo-tài-liệu của `W1-08`; phải là 0 embed.
3. **sửa một dòng** ở giữa tài liệu **dài nhất** — con số của `W3-07`.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

logger = logging.getLogger("incremental_probe")

DEFAULT_CONFIG = Path("configs/indexing/bgem3.yaml")
NEEDLE = "\n\nGHI CHÚ SỬA MỘT DÒNG ĐỂ ĐO RE-INDEX TĂNG DẦN.\n\n"


@dataclass
class Pass:
    label: str
    documents_indexed: int
    chunks_written: int
    chunks_embedded: int
    chunks_reused: int
    seconds_embed: float
    seconds_total: float


def _repin(manifest: Path, corpus_dir: Path) -> None:
    """Tính lại `sha256`/`text_sha256` sau khi sửa file. Không có bước này thì
    `iter_documents` dừng ở kiểm toàn vẹn — đúng hành vi của nó (`W1-08`)."""
    import hashlib

    from pipeline.corpus.manifest import load_manifest, write_manifest
    from rag_core.loaders import load_document

    rows = []
    for entry in load_manifest(manifest):
        payload = (corpus_dir / entry.relative_path).read_bytes()
        loaded = load_document(corpus_dir / entry.relative_path, ocr="off")
        rows.append(
            entry.model_copy(
                update={
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                    "text_sha256": loaded.text_sha256,
                    "parse_fingerprint": loaded.fingerprint.canonical,
                }
            )
        )
    write_manifest(manifest, rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--collection", default="rag_incr_probe")
    parser.add_argument("--limit", type=int, default=0, help="chỉ dùng N tài liệu đầu")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from pipeline.corpus.manifest import load_manifest
    from pipeline.indexing.build_index import build_index
    from pipeline.indexing.config import load_index_config
    from rag_core.settings import get_settings

    settings = get_settings()
    url = settings.qdrant_url
    source = load_index_config(args.config)

    with tempfile.TemporaryDirectory(prefix="incr-probe-") as tmp:
        room = Path(tmp)
        corpus = room / "corpus"
        shutil.copytree(source.corpus_dir, corpus)
        manifest = room / "manifest.csv"
        shutil.copyfile(source.manifest_path, manifest)

        overrides = {
            "name": args.collection,
            "collection": args.collection,
            "manifest_path": manifest,
            "corpus_dir": corpus,
            "state_dir": room / "state",
            "use_cache": False,  # cache chunk sẽ che mất chi phí chunk lại
        }
        if args.limit:
            overrides["max_documents"] = args.limit
        config = source.model_copy(update=overrides)

        # Nạp model MỘT lần và dùng lại cho cả ba lượt: nạp BGE-M3 là 2,2 GB, và
        # tính nó vào `seconds_total` sẽ làm mọi tỉ lệ vô nghĩa.
        embeddings = config.build_embeddings()

        def run(label: str) -> Pass:
            t0 = time.perf_counter()
            report = build_index(
                config,
                qdrant_url=url,
                recreate=(label == "sạch"),
                embeddings=embeddings,
            )
            return Pass(
                label=label,
                documents_indexed=report.n_documents_indexed,
                chunks_written=report.n_chunks_written,
                chunks_embedded=report.n_chunks_embedded,
                chunks_reused=report.n_chunks_reused,
                seconds_embed=round(report.seconds.get("upsert", 0.0), 1),
                seconds_total=round(time.perf_counter() - t0, 1),
            )

        passes = [run("sạch"), run("không sửa gì")]

        entries = load_manifest(manifest)
        if args.limit:
            entries = sorted(entries, key=lambda e: e.doc_id)[: args.limit]
        biggest = max(entries, key=lambda e: e.bytes)
        target = corpus / biggest.relative_path
        payload = target.read_bytes()
        middle = len(payload) // 2
        target.write_bytes(payload[:middle] + NEEDLE.encode("utf-8") + payload[middle:])
        logger.info(
            "Sửa %s (%d byte) — chèn %d byte vào giữa",
            biggest.doc_id,
            biggest.bytes,
            len(NEEDLE.encode("utf-8")),
        )
        _repin(manifest, corpus)
        passes.append(run("sửa một dòng"))

        try:
            from rag_core.retrieval.qdrant_store import QdrantDenseRetriever

            QdrantDenseRetriever(
                embeddings, collection=args.collection, url=url
            ).client.delete_collection(args.collection)
        except Exception as exc:  # pragma: no cover - dọn dẹp, không phải kết quả
            logger.warning("Không xoá được collection tạm: %s", exc)

    clean, untouched, edited = passes
    logger.info("")
    logger.info("%-16s %9s %9s %9s %9s", "lượt", "embed", "mượn", "giây", "tổng giây")
    for item in passes:
        logger.info(
            "%-16s %9d %9d %9.1f %9.1f",
            item.label,
            item.chunks_embedded,
            item.chunks_reused,
            item.seconds_embed,
            item.seconds_total,
        )
    if untouched.chunks_embedded:
        logger.error(
            "⚠️ lượt 'không sửa gì' vẫn embed %d chunk — tầng bỏ-qua-theo-tài-liệu "
            "của W1-08 đã hỏng, và mọi con số W3-07 bên dưới đo trên nền sai",
            untouched.chunks_embedded,
        )
    speedup = clean.seconds_embed / edited.seconds_embed if edited.seconds_embed else 0.0
    ratio = clean.chunks_embedded / edited.chunks_embedded if edited.chunks_embedded else 0.0
    logger.info("")
    logger.info("sửa một dòng vs build sạch: embed ít hơn %.0f× · nhanh hơn %.1f×", ratio, speedup)
    logger.info("G3 (≥10×): %s", "ĐẠT" if speedup >= 10 else "CHƯA ĐẠT")

    summary = {
        "config": str(args.config),
        "document_edited": biggest.doc_id,
        "document_bytes": biggest.bytes,
        "passes": [asdict(p) for p in passes],
        "untouched_pass_embedded": untouched.chunks_embedded,
        "embed_ratio": round(ratio, 2),
        "speedup": round(speedup, 2),
        "g3_met": speedup >= 10,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("→ %s", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
