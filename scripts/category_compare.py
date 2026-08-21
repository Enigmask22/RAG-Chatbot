"""Script bằng chứng cho `reports/tasks/w2-05-reranker.md` §6.4b — kiểm định theo category.

Vì sao file này tồn tại và vì sao nó **tạm thời**: `pipeline/eval/compare.py` chỉ
so trên toàn bộ tập đo, không có chiều category; bảng `by_category` trong file
JSON có số nhưng không có kiểm định. Nên một mức tụt khu trú ở 20% tập đo đi qua
được cả hai — đúng cái đã xảy ra với `cross_lingual` ở `W2-04`.

Đây **không** phải bản cài thứ hai của `compare.py`: nó gọi lại
`mcnemar_exact`/`paired_bootstrap` của chính module đó, chỉ thêm phần lọc theo
`category`. Khi `compare.py` có `--category` (điều kiện của DoD `W2-09`) thì
**xoá file này** — hai công cụ trả lời cùng một câu hỏi là hai câu trả lời khác
nhau đang chờ xảy ra.

Chạy: `python scripts/category_compare.py`
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pipeline.eval.compare import DEFAULT_BOOTSTRAP, DEFAULT_SEED, mcnemar_exact, paired_bootstrap

logger = logging.getLogger(__name__)

#: Trùng với `--dir` mặc định của `pipeline/eval/compare.py`. Suy từ vị trí file
#: chứ không phải từ CWD: chạy script từ thư mục khác vẫn phải tìm đúng chỗ.
RUNS = Path(__file__).resolve().parent.parent / "plans" / "reports" / "runs"

#: Cặp (nền, ứng viên) cần đo. Ba dòng đầu dựng nên §6.4b; dòng cuối là phép
#: đối chứng trả lời "reranker có vá lại phần hybrid làm hỏng không".
PAIRS = (
    ("bgem3", "bgem3-rrf-k1-c20"),
    ("bgem3", "bgem3-rr-c50"),
    ("bgem3-rrf-k1-c20", "bgem3-rr-c50"),
    ("bgem3-rr-dense-c50", "bgem3-rr-c50"),
)

BINARY = ("hit_rate@1", "hit_rate@5", "hit_rate@10")
CONTINUOUS = ("recall@5", "recall@20", "ndcg@10")


def load(run: str, category: str | None) -> dict[str, dict[str, float]]:
    """Điểm từng truy vấn, lọc theo `category` (None = tất cả)."""
    out: dict[str, dict[str, float]] = {}
    with (RUNS / f"{run}-per-query.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if category is None or row["category"] == category:
                out[row["query_id"]] = row["scores"]
    return out


def report(base: str, cand: str, category: str | None) -> None:
    a, b = load(base, category), load(cand, category)
    ids = sorted(set(a) & set(b))
    logger.info("### %s (%d câu)  %s → %s", category or "TOÀN BỘ", len(ids), base, cand)
    for metric in BINARY:
        # McNemar cho metric nhị phân — cùng quy tắc route như `compare.py`.
        a_only = sum(1 for i in ids if a[i][metric] > 0.5 and b[i][metric] <= 0.5)
        b_only = sum(1 for i in ids if b[i][metric] > 0.5 and a[i][metric] <= 0.5)
        mean_a = sum(a[i][metric] for i in ids) / len(ids)
        mean_b = sum(b[i][metric] for i in ids) / len(ids)
        logger.info(
            "  %-12s %.4f → %.4f  Δ=%+.4f  p=%.4f  %d↔%d",
            metric,
            mean_a,
            mean_b,
            mean_b - mean_a,
            mcnemar_exact(a_only, b_only),
            a_only,
            b_only,
        )
    for metric in CONTINUOUS:
        mean_a = sum(a[i][metric] for i in ids) / len(ids)
        mean_b = sum(b[i][metric] for i in ids) / len(ids)
        lo, hi = paired_bootstrap(
            [b[i][metric] - a[i][metric] for i in ids],
            iterations=DEFAULT_BOOTSTRAP,
            seed=DEFAULT_SEED,
        )
        verdict = "khác biệt thật" if (lo > 0 or hi < 0) else "nhiễu"
        logger.info(
            "  %-12s %.4f → %.4f  Δ=%+.4f  CI95 [%+.4f, %+.4f]  %s",
            metric,
            mean_a,
            mean_b,
            mean_b - mean_a,
            lo,
            hi,
            verdict,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # ⚠️ 43 câu `cross_lingual` có ngưỡng phân giải thô hơn 209 câu nhiều — in kèm
    # `n` là bắt buộc, và Δ một mình không được dùng để tuyên bố người thắng.
    for category in ("cross_lingual", None):
        for base, cand in PAIRS:
            report(base, cand, category)


if __name__ == "__main__":
    main()
