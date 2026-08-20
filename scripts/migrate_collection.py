#!/usr/bin/env python
"""So schema của collection Qdrant đang sống với schema mà một config cần (`W2-02`).

    uv run python scripts/migrate_collection.py --config configs/indexing/bgem3.yaml
    uv run python scripts/migrate_collection.py --config ... --run   # build lại thật

**Script này cố ý KHÔNG migrate tại chỗ, vì không thể.** Qdrant không cho thêm
named vector vào collection đã tồn tại. Thêm `sparse` vào `rag_bgem3` bắt buộc
phải tạo collection mới và ghi lại toàn bộ point.

Vậy tại sao vẫn có script: nó trả lời đúng câu hỏi mà người vận hành thật sự cần
trước khi chạy một job 400 giây — *"collection đang nằm trong Qdrant có phải thứ
config này mô tả không, và nếu không thì phải làm gì?"* Đây là phiên bản cho
**schema** của câu hỏi mà `IndexConfig.fingerprint` trả lời cho **nội dung**.

Một câu hỏi hợp lý: sao không copy dense vector cũ sang collection mới rồi chỉ
tính thêm sparse? Vì với BGE-M3 thì sparse ra từ **cùng** một forward pass với
dense — tính được sparse tức là đã tính lại dense. Copy chẳng tiết kiệm gì, mà
lại thêm một đường ghi thứ hai vào index. Phần thật sự đắt (chunking) đã có cache.

Mã trả về:
    0 — schema khớp, không cần làm gì (hoặc `--run` đã build lại xong)
    1 — schema lệch và **chưa** làm gì (in ra lệnh phải chạy)
    2 — collection chưa tồn tại
    3 — lỗi kết nối / cấu hình
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from pipeline.indexing.config import load_index_config
from rag_core.retrieval import SPARSE_VECTOR_NAME, schema_problems
from rag_core.settings import get_settings

logger = logging.getLogger("migrate_collection")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML của IndexConfig")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Build lại thật khi schema lệch (tương đương `build_index --recreate`)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_index_config(args.config)
    settings = get_settings()
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    embeddings = config.build_embeddings()
    retriever = config.build_retriever(embeddings, url=settings.qdrant_url, api_key=api_key)

    try:
        exists = retriever.client.collection_exists(retriever.collection)
    except Exception as exc:
        logger.error("Không kết nối được Qdrant: %s. Chạy `make up` trước.", exc)
        return 3

    want_sparse = retriever.writes_sparse
    logger.info("config `%s` → collection `%s`", config.name, retriever.collection)
    logger.info(
        "  cần             dense %d chiều%s",
        embeddings.dimension,
        f" + named vector {SPARSE_VECTOR_NAME!r}" if want_sparse else " (không sparse)",
    )

    if not exists:
        logger.warning("  collection chưa tồn tại — chưa có gì để so")
        logger.info("Chạy:  make index BUNDLE=%s", config.name)
        return 2

    dense_sizes, sparse_names = retriever.live_schema()
    logger.info(
        "  đang có         dense %s · sparse %s · %d point",
        dense_sizes or "(vector vô danh)",
        sorted(sparse_names) or "(không)",
        retriever.count(),
    )

    problems = schema_problems(
        dense_sizes=dense_sizes,
        sparse_names=sparse_names,
        want_dimension=embeddings.dimension,
        want_sparse=want_sparse,
    )
    if not problems:
        logger.info("✅ Schema khớp. Không cần build lại.")
        return 0

    for problem in problems:
        logger.warning("  ✗ %s", problem)
    logger.warning("Qdrant không sửa được schema tại chỗ — phải ghi lại toàn bộ point.")

    if not args.run:
        logger.info("Chạy lệnh sau (hoặc thêm `--run` vào script này):")
        logger.info("  python -m pipeline.indexing.build_index --config %s --recreate", args.config)
        logger.info(
            "Muốn giữ index cũ để đối chiếu thì đổi `collection` trong config thành tên khác."
        )
        return 1

    # Import muộn: `--run` là đường duy nhất cần tới nó, và nó kéo theo cả chuỗi
    # nạp corpus + chunker mà nhánh chỉ-kiểm-tra không dùng.
    from pipeline.indexing.build_index import build_index

    logger.warning("--run: XOÁ collection `%s` và ghi lại từ đầu", retriever.collection)
    report = build_index(
        config,
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=api_key,
        recreate=True,
    )
    report.log_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
