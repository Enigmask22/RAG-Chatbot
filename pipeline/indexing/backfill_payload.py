"""Migrate payload của collection đã build. `W2-06`.

Chạy: `make backfill-payload BUNDLE=bgem3`

Vì sao cần một đường migrate riêng thay vì build lại index: `W2-06` thêm
`published_at` vào payload phẳng và vào `PAYLOAD_INDEXES`, nhưng các collection
dựng trước đó không có nó. Build lại thì đúng nhưng tốn ~380 s và **embed lại
toàn bộ** — tức đưa vector vào diện có thể đổi, và mọi con số eval đã công bố từ
`W2-01` đến `W2-05` đều đo trên chính những vector đó. Backfill payload không
chạm vector, nên nó không thể làm sai một con số nào.

Hai việc, tách rời vì hỏng theo hai kiểu khác nhau:

1. **Thiếu payload index** → filter vẫn cho **kết quả đúng**, Qdrant chỉ lùi về
   quét toàn bộ collection. Hỏng về hiệu năng, không về đúng/sai, nên không có
   test nào tự phát hiện được.
2. **Thiếu field trong payload** → filter cho **kết quả sai**: point không có
   `published_at` thì không khớp `DatetimeRange`, nên `published_after=2020` trả
   0 kết quả trên toàn corpus. Đúng chế độ hỏng im lặng mà `W2-06` tồn tại để
   chặn, chỉ là lần này do dữ liệu cũ chứ không do code.

Chạy lại lần thứ hai là no-op (so payload hiện có với payload đúng rồi chỉ ghi
phần lệch), nên gọi nó nhiều lần không sao.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from rag_core.settings import get_settings

from .config import load_index_config

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/indexing/bgem3.yaml"))
    parser.add_argument("--collection", help="Ghi đè tên collection")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ báo index còn thiếu và số point sẽ đổi, không ghi gì.",
    )
    parser.add_argument("--report", type=Path, help="Ghi kết quả ra JSON.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")

    config = load_index_config(args.config, collection=args.collection)
    settings = get_settings()
    # `.get_secret_value()` là bắt buộc: `api_key` mong đợi `str`, và truyền
    # thẳng `SecretStr` sẽ gửi lên header chuỗi `SecretStr('**********')`.
    # Qdrant local không cần key nên lỗi này sẽ im lặng cho tới lúc deploy.
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    embeddings = config.build_embeddings()
    store = config.build_retriever(embeddings, url=settings.qdrant_url, api_key=api_key)

    total = store.count()
    logger.info("Collection %s · %d point", store.collection, total)

    if args.dry_run:
        # `payload_schema` cho biết index nào đã có; phần point thì chỉ đếm được
        # bằng cách quét, nên dry-run chỉ báo index và tổng số point.
        from rag_core.retrieval import PAYLOAD_INDEXES

        have = set(store.client.get_collection(store.collection).payload_schema or {})
        missing = [field for field, _ in PAYLOAD_INDEXES if field not in have]
        logger.info("Index còn thiếu: %s", missing or "không có")
        logger.info("dry-run — không ghi gì. Bỏ `--dry-run` để chạy thật.")
        payload = {
            "collection": store.collection,
            "points": total,
            "missing_indexes": missing,
            "dry_run": True,
        }
    else:
        created = store.ensure_payload_indexes()
        logger.info("Đã tạo payload index: %s", created or "không có gì thiếu")
        updated = store.backfill_flat_payload(batch=args.batch)
        logger.info("Đã cập nhật payload của %d/%d point", updated, total)
        payload = {
            "collection": store.collection,
            "points": total,
            "created_indexes": created,
            "updated_points": updated,
            "dry_run": False,
        }

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        logger.info("Đã ghi %s", args.report)
    return 0


if __name__ == "__main__":  # pragma: no cover - đường CLI
    sys.exit(main())
