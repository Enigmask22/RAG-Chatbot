"""Ghim `text_sha256` + `parse_fingerprint` vào manifest corpus. `TD-22`.

Hai chế độ, và mặc định là chế độ **không sửa gì**:

    uv run python scripts/pin_manifest.py            # chỉ báo cáo (mặc định)
    uv run python scripts/pin_manifest.py --write    # ghi vào manifest

Chế độ báo cáo trả lời một câu hỏi mà `TD-22` đặt ra nhưng chưa ai đo: **hôm nay
hai cột này có nói thêm được gì so với `sha256` không?** Với `.txt` thì không —
`bytes.decode("utf-8").encode("utf-8")` trả lại đúng byte cũ, nên `text_sha256`
phải trùng khít `sha256`. Script in ra tỉ lệ trùng để chuyện đó là một **số đo**
chứ không phải một lập luận.

⚠️ `--write` phải chạy **sau khi** đã xác nhận văn bản parse ra vẫn dùng được.
Ghim là tuyên bố "đây là văn bản đúng"; ghim đè lên một lần parse đã hỏng thì
manifest hết tác dụng cảnh báo mà không ai biết.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

logger = logging.getLogger("pin_manifest")

DEFAULT_MANIFEST = Path("data/corpus_manifest.csv")
DEFAULT_CORPUS = Path("data/corpus")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--write", action="store_true", help="ghi vào manifest")
    parser.add_argument("--ocr", default="off", choices=("off", "auto", "force"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from pipeline.corpus.manifest import load_manifest, write_manifest
    from rag_core.loaders import load_document, loader_for

    entries = load_manifest(args.manifest)
    if not entries:
        logger.error("Manifest %s rỗng hoặc không tồn tại", args.manifest)
        return 1

    updated = []
    identical = 0
    changed: list[str] = []
    loaders: Counter[str] = Counter()

    for entry in entries:
        path = args.corpus_dir / entry.relative_path
        if not path.exists():
            logger.error("%s: thiếu file %s", entry.doc_id, path)
            return 1
        kind = loader_for(path)
        loaders[kind] += 1
        loaded = load_document(path, ocr=args.ocr, language=None)

        if loaded.text_sha256 == entry.sha256:
            identical += 1
        if entry.text_sha256 and entry.text_sha256 != loaded.text_sha256:
            changed.append(entry.doc_id)

        updated.append(
            entry.model_copy(
                update={
                    "text_sha256": loaded.text_sha256,
                    "parse_fingerprint": loaded.fingerprint.canonical,
                }
            )
        )

    total = len(entries)
    logger.info("%d tài liệu · loader: %s", total, dict(loaders))
    logger.info(
        "text_sha256 == sha256 (byte): %d/%d (%.1f%%) — phép biến đổi là hàm đồng nhất",
        identical,
        total,
        100 * identical / total,
    )
    fingerprints = Counter(e.parse_fingerprint for e in updated)
    for canonical, count in fingerprints.most_common():
        logger.info("  %4d × %s", count, canonical)
    if changed:
        logger.warning(
            "⚠️ %d tài liệu có văn bản KHÁC cái đã ghim: %s",
            len(changed),
            ", ".join(changed[:5]) + (" …" if len(changed) > 5 else ""),
        )

    if not args.write:
        logger.info("(chế độ báo cáo — thêm `--write` để ghi)")
        return 0

    write_manifest(args.manifest, updated)
    logger.info("→ đã ghi %d dòng vào %s", len(updated), args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
