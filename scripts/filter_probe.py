"""Giá của metadata filter trên corpus thật. `W2-06`.

Chạy: `make filter-probe BUNDLE=bgem3`

Vì sao cần script này chứ không chỉ cần test: mọi test của `W2-06` kiểm **đúng
hay sai**, và filter thiếu payload index vẫn cho kết quả **đúng** — Qdrant lùi về
quét toàn bộ collection. Hỏng về hiệu năng không làm test nào đỏ, nên nó chỉ lộ
ra khi có người đo.

Ba câu hỏi:

1. **Lọc có tốn gì không?** So p50/p95 của cùng truy vấn có và không có filter.
2. **Độ chọn lọc ảnh hưởng thế nào?** Một filter giữ 90% point và một filter giữ
   5% có thể nhanh chậm ngược nhau: HNSW phải đi xa hơn để tìm đủ `top_k` khi
   filter chặt, nhưng lại xét ít ứng viên hơn khi filter lỏng.
3. **`DatetimeRange` có đắt hơn khớp-chính-xác không?** Khoảng phải so hai đầu
   trên chuỗi RFC3339 thay vì tra một bảng keyword.

Dùng chính truy vấn của `golden_v1` để số so được với `reports/runs/`, và
warm-up trước khi đo — bài học `NEW-04`: p95 đo lẫn thời gian nạp model cho ra
15.219 ms trong khi p50 là 31 ms.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipeline.indexing.config import load_index_config
from rag_core.retrieval import MetadataFilter, build_branch
from rag_core.settings import get_settings

logger = logging.getLogger(__name__)

DEFAULT_GOLDEN = Path("data/golden/golden_v1.jsonl")

#: (tên, filter). `None` là mốc so sánh. Các mốc thời gian chọn theo phân bố thật
#: của corpus (World Bank, 2003–2025) để mỗi ca có độ chọn lọc khác nhau rõ rệt.
CASES: tuple[tuple[str, MetadataFilter | None], ...] = (
    ("không filter", None),
    ("lang=en", MetadataFilter(lang="en")),  # type: ignore[arg-type]
    ("lang=vi", MetadataFilter(lang="vi")),  # type: ignore[arg-type]
    ("doc_type=dev_report", MetadataFilter(doc_type="dev_report")),  # type: ignore[arg-type]
    ("published_after=2020", MetadataFilter(published_after=datetime(2020, 1, 1, tzinfo=UTC))),
    (
        "khoảng 2020–2023",
        MetadataFilter(
            published_after=datetime(2020, 1, 1, tzinfo=UTC),
            published_before=datetime(2023, 12, 31, tzinfo=UTC),
        ),
    ),
    (
        "lang=en + khoảng",
        MetadataFilter(
            lang="en",  # type: ignore[arg-type]
            published_after=datetime(2020, 1, 1, tzinfo=UTC),
        ),
    ),
    # Filter không khớp gì: corpus thật không có `tenant_id`. Đây không phải ca
    # nhân tạo — nó là ca "tenant mới, chưa có tài liệu", và nó cho biết đường
    # rỗng có nhanh hay Qdrant vẫn quét hết.
    ("tenant không tồn tại", MetadataFilter(tenant_id="không-có-tenant-này")),
    # ⚠️ Đối chứng thứ tự, **bắt buộc**. Các ca trên chạy tuần tự và ca "không
    # filter" đứng đầu, nên nếu còn hiệu ứng warm-up (cache HNSW, GPU lên xung)
    # thì nó bị phạt và chiều giảm dần của bảng chỉ là chiều của thời gian, không
    # phải của độ chọn lọc. Lặp lại đúng ca đầu ở **cuối**: hai con số gần nhau
    # thì thứ tự không phải nguyên nhân; lệch nhiều thì cả bảng vô nghĩa.
    ("không filter (lặp cuối)", None),
)


def load_queries(path: Path, limit: int) -> list[str]:
    out: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            question = row.get("question") or row.get("query")
            if question:
                out.append(str(question))
            if len(out) >= limit:
                break
    return out


def selectivity(store: Any, flt: MetadataFilter | None) -> int:
    """Số point khớp filter — mẫu số để đọc con số độ trễ."""
    from rag_core.retrieval import build_filter

    return int(
        store.client.count(store.collection, count_filter=build_filter(flt), exact=True).count
    )


def _stats(times: list[float], prefix: str) -> dict[str, float]:
    ordered = sorted(times)
    return {
        f"{prefix}p50_ms": statistics.median(ordered),
        f"{prefix}p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        f"{prefix}mean_ms": statistics.fmean(ordered),
    }


def measure(branch: Any, queries: Sequence[str], flt: MetadataFilter | None, top_k: int) -> dict:
    """Đo cả `retrieve()` đầu-cuối **và** riêng phần Qdrant.

    ⚠️ Phần "riêng Qdrant" là bắt buộc, không phải thêm cho đủ. Đo đầu-cuối thì
    p50 của **cùng một ca** lệch ±11 ms giữa hai lượt trong cùng tiến trình,
    trong khi p95 đứng im trong 1,3 ms ở cả 9 ca — tức biến động nằm ở phần
    **không** phụ thuộc filter (embed truy vấn trên GPU), và nó lớn hơn hẳn thứ
    đang muốn đo. Cùng cơ chế đã làm `W2-03` §8 quy sai nguyên nhân: các con số
    không cộng lại đúng thì phải tách ra, không phải tăng số mẫu.

    Nên embed **một lần** ngoài vòng đo, rồi bấm giờ đúng lời gọi Qdrant.
    """
    from rag_core.retrieval import DENSE_VECTOR_NAME, build_filter

    store = getattr(branch, "store", branch)
    query_filter = build_filter(flt)

    end_to_end: list[float] = []
    returned: list[int] = []
    for question in queries:
        started = time.perf_counter()
        hits = branch.retrieve(question, top_k, filters=flt)
        end_to_end.append((time.perf_counter() - started) * 1000)
        returned.append(len(hits))

    # Embed trước, ngoài vòng bấm giờ — đây là hằng số đang che mất phép đo.
    vectors = [store.embeddings.embed_query(question) for question in queries]
    search_only: list[float] = []
    for vector in vectors:
        started = time.perf_counter()
        store.client.query_points(
            collection_name=store.collection,
            query=list(vector),
            using=DENSE_VECTOR_NAME,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        search_only.append((time.perf_counter() - started) * 1000)

    return {
        **_stats(end_to_end, ""),
        **_stats(search_only, "search_"),
        "mean_returned": statistics.fmean(returned),
        "n": len(end_to_end),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/indexing/bgem3.yaml"))
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--mode", default="dense", help="Nhánh truy hồi cần đo.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--sample", type=int, default=40, help="Số truy vấn mỗi ca.")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")

    config = load_index_config(args.config)
    settings = get_settings()
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    embeddings = config.build_embeddings()
    store = config.build_retriever(embeddings, url=settings.qdrant_url, api_key=api_key)
    branch = build_branch(store, args.mode)

    queries = load_queries(args.golden, args.sample)
    if not queries:
        parser.error(f"Không đọc được truy vấn nào từ {args.golden}")
    total = store.count()

    # Warm-up: nạp model embedding + làm nóng HNSW. Không có nó thì ca đầu tiên
    # mang toàn bộ thời gian nạp và trông như "không filter là ca chậm nhất".
    for question in queries[:3]:
        branch.retrieve(question, args.top_k)

    rows: list[dict[str, Any]] = []
    for label, flt in CASES:
        matching = selectivity(store, flt)
        stats = measure(branch, queries, flt, args.top_k)
        row = {
            "case": label,
            "matching_points": matching,
            "selectivity": matching / total if total else 0.0,
            **stats,
        }
        rows.append(row)
        logger.info(
            "%-24s %6d point (%5.1f%%)  e2e p50 %6.1f p95 %6.1f │ chỉ-Qdrant p50 %6.2f p95 %6.2f",
            label,
            matching,
            100 * row["selectivity"],
            stats["p50_ms"],
            stats["p95_ms"],
            stats["search_p50_ms"],
            stats["search_p95_ms"],
        )

    # So trên **chỉ-Qdrant**, không trên đầu-cuối: xem docstring của `measure`.
    baseline = rows[0]["search_p50_ms"]
    logger.info("--- chỉ-Qdrant, so với 'không filter' (%.2f ms):", baseline)
    for row in rows[1:]:
        logger.info(
            "%-24s %+6.2f ms  (%+.0f%%)",
            row["case"],
            row["search_p50_ms"] - baseline,
            100 * (row["search_p50_ms"] / baseline - 1),
        )

    payload = {
        "collection": store.collection,
        "branch": branch.name,
        "total_points": total,
        "top_k": args.top_k,
        "queries": len(queries),
        "cases": rows,
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
