"""Dựng lại chế độ hỏng của `TD-22` trên tài liệu thật, rồi cho xem nó bị chặn.

Chế độ hỏng cần chứng minh, nguyên văn từ `TD-22`: *`sha256` vẫn khớp manifest,
`iter_documents` vẫn xanh, mọi `chunk_id` vẫn tồn tại, không test nào đỏ — và mọi
con số recall sau đó sai.*

Cách dựng lại mà không phải nâng cấp thư viện thật: lấy **đúng** một tài liệu
corpus, chép nguyên **byte** sang một tên file `.md`. Byte không đổi một bit nên
`sha256` khớp tuyệt đối — nhưng `.md` định tuyến sang docling thay vì `plain`, và
văn bản parse ra là một thứ khác. Đó chính xác là hình dạng của việc đổi parser.

    uv run python scripts/parse_pin_probe.py

Bốn cảnh, theo thứ tự:

1. `.txt` — đường hàng đồng nhất, mọi thứ khớp.
2. `.md` **chưa ghim** — hash byte khớp, văn bản đã khác. Đây là chỗ bản cũ đi
   tiếp trong im lặng.
3. `.md` **đã ghim** — nạp được, và ghi lại được văn bản nào đã được duyệt.
4. `.md` đã ghim nhưng **môi trường dịch chuyển** — thông báo lỗi phải chỉ đúng
   gói nào đã đổi, không bắt người đọc tự dò hai chuỗi 9 phần.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

logger = logging.getLogger("parse_pin_probe")

DEFAULT_MANIFEST = Path("data/corpus_manifest.csv")
DEFAULT_CORPUS = Path("data/corpus")


def _smallest(entries: list) -> object:
    return min(entries, key=lambda e: e.bytes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from pipeline.corpus.manifest import load_manifest
    from pipeline.indexing.corpus_loader import ParsePinError, iter_documents
    from rag_core.loaders import load_document

    entries = load_manifest(args.manifest)
    entry = _smallest(entries)
    src = args.corpus_dir / entry.relative_path
    logger.info("Tài liệu: %s (%s, %d byte)", entry.doc_id, entry.relative_path, entry.bytes)

    out: dict[str, object] = {"doc_id": entry.doc_id, "bytes": entry.bytes}

    with tempfile.TemporaryDirectory() as tmp:
        room = Path(tmp)
        as_md = room / (Path(entry.relative_path).stem + ".md")
        shutil.copyfile(src, as_md)  # NGUYÊN byte

        plain = load_document(src, ocr="off")
        docling = load_document(as_md, ocr="off")

        logger.info("")
        logger.info("── 1. cùng byte, hai loader ──────────────────────────────")
        logger.info(
            "  sha256 byte      %s  ==  %s", plain.source_sha256[:16], docling.source_sha256[:16]
        )
        logger.info("  khớp manifest    %s", plain.source_sha256 == entry.sha256)
        logger.info(
            "  text_sha256      %s  vs  %s", plain.text_sha256[:16], docling.text_sha256[:16]
        )
        logger.info("  ký tự            %d  vs  %d", len(plain.text), len(docling.text))
        logger.info("  vân tay          %s", plain.fingerprint.canonical)
        logger.info("                   %s", docling.fingerprint.canonical)
        out["bytes_identical"] = plain.source_sha256 == docling.source_sha256
        out["text_identical"] = plain.text_sha256 == docling.text_sha256
        out["chars_plain"] = len(plain.text)
        out["chars_docling"] = len(docling.text)
        out["fingerprint_plain"] = plain.fingerprint.canonical
        out["fingerprint_docling"] = docling.fingerprint.canonical

        md_entry = entry.model_copy(
            update={"relative_path": as_md.name, "text_sha256": "", "parse_fingerprint": ""}
        )

        logger.info("")
        logger.info("── 2. `.md` CHƯA ghim ────────────────────────────────────")
        try:
            list(iter_documents([md_entry], room, ocr="off"))
            logger.error("  ❌ nạp trót lọt — chỗ hở VẪN mở")
            out["unpinned_blocked"] = False
        except ParsePinError as exc:
            logger.info("  ✅ bị chặn: %s", str(exc)[:150])
            out["unpinned_blocked"] = True

        logger.info("")
        logger.info("── 3. `.md` ĐÃ ghim ──────────────────────────────────────")
        pinned = md_entry.model_copy(
            update={
                "text_sha256": docling.text_sha256,
                "parse_fingerprint": docling.fingerprint.canonical,
            }
        )
        docs = list(iter_documents([pinned], room, ocr="off"))
        logger.info(
            "  ✅ nạp được, %d ký tự, parser=%s",
            len(docs[0].content),
            docs[0].metadata.extra["parser"],
        )
        out["pinned_loads"] = len(docs) == 1

        logger.info("")
        logger.info("── 4. đã ghim, nhưng môi trường dịch chuyển ──────────────")
        drifted = pinned.model_copy(
            update={
                "text_sha256": "0" * 64,
                "parse_fingerprint": docling.fingerprint.canonical.replace(
                    "docling-core=", "docling-core=2.91.0#"
                ).split("#")[0],
            }
        )
        try:
            list(iter_documents([drifted], room, ocr="off"))
            logger.error("  ❌ không phát hiện")
            out["drift_blocked"] = False
        except ParsePinError as exc:
            logger.info("  ✅ %s", str(exc).split("Nghĩa là")[-1].split("Mọi TextSpan")[0].strip())
            out["drift_blocked"] = True
            out["drift_message"] = str(exc)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("")
        logger.info("→ %s", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
