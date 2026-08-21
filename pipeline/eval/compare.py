"""So hai lần chạy eval — có kiểm định, và có hàng rào chống so sai.

Vì sao module này tồn tại, nói bằng con số của chính hôm nay:

`hit_rate@5` đi từ 0,2153 (baseline) xuống 0,2010 (chunk550). Nghe như "tụt
6,7%". Trên 209 câu, đó là **45 câu xuống 42 câu** — chênh **3 câu**. Không có
kiểm định thì bảng ablation 12 dòng của `W2-08` sẽ xếp hạng 12 cấu hình bằng
những mức chênh cỡ này, và cái thắng chỉ là cái may.

Hai hàng rào, đều học từ lỗi thật:

1. **Từ chối so `recall@k` / `nDCG@k` / `MAP@k` khi số nhãn mỗi câu khác nhau.**
   Nhãn golden set neo theo span (`TD-12`) nên đổi `chunk_size` làm một span phủ
   nhiều chunk hơn: 1,38 → 1,96 nhãn/câu khi hạ 1000 → 550. `recall@k` có mẫu số
   là chính con số đó, nên nó tụt 25,8% **kể cả khi chất lượng không đổi** — và
   29,6% là mức tụt lý thuyết nếu truy hồi y nguyên. Suýt đọc thành "hạ
   chunk_size làm tụt recall 26%".
2. **Chỉ so trên tập truy vấn giao nhau**, và báo rõ nếu hai lần chạy không
   cùng tập câu. So 209 câu với 200 câu rồi kết luận là một dạng tự chọn mẫu.

Kiểm định dùng thư viện chuẩn, không cần `scipy`:

* **McNemar exact** cho metric nhị phân (`hit_rate@k`) — đúng bài toán "cùng một
  bộ câu hỏi, hai hệ thống, đếm câu đổi chiều".
* **Bootstrap cặp** cho metric liên tục (`mrr`, `precision@k`) — khoảng tin cậy
  của *hiệu*, lấy mẫu lại theo truy vấn với seed cố định.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "BINARY_METRICS",
    "ComparisonRow",
    "RunScores",
    "compare_runs",
    "format_table",
    "load_per_query",
    "main",
    "mcnemar_exact",
    "paired_bootstrap",
]

logger = logging.getLogger("pipeline.eval.compare")

#: Metric có mẫu số là số nhãn liên quan — không so được khi phân bố nhãn đổi.
CARDINALITY_SENSITIVE = ("recall@", "ndcg@", "map@")

#: Metric nhận giá trị 0/1 trên từng câu → dùng McNemar.
#:
#: `precision@1` vào đây từ `W2-05`, và nó vào vì một **mâu thuẫn quan sát được**:
#: `precision@1` bằng `hit_rate@1` từng chữ số (top-1 chỉ có một chỗ, nên nó liên
#: quan hay không là 0/1), nhưng nó đi đường bootstrap và cho kết luận **khác**.
#: So `bgem3-rr-c50` với `bgem3-rr-c100`: cùng con số 0,5598 → 0,5789, McNemar cho
#: `p = 0,125` (0↔4 câu đổi chiều — bốn lần tung xu cùng mặt thì không kết luận
#: được gì) còn bootstrap cho CI95 `[+0,0048, +0,0383]`, tức "khác biệt thật".
#: Bootstrap tự tin quá mức trên một metric rời rạc có rất ít câu đổi chiều, và
#: hai kiểm định cho hai câu trả lời về cùng một số là một hố im lặng —
#: người đọc sẽ trích dòng nào thuận với mình.
#:
#: ⚠️ Chỉ `precision@1`, không phải `precision@` chung: `precision@5` nhận
#: 0; 0,2; 0,4… nên McNemar không áp được. Và `recall@1` cũng **không** vào đây —
#: nó là `0` hoặc `1/n_relevant`, tức không nhị phân, dù mẫu câu khác 0 của nó
#: trùng với `hit_rate@1`.
#:
#: ⚠️⚠️ `precision@1` phải khớp **đúng tên**, không phải tiền tố: bản sửa đầu của
#: tôi đưa nó vào `BINARY_PREFIXES` và `"precision@10".startswith("precision@1")`
#: là `True`, nên `precision@10` — nhận 0; 0,1; 0,2… — bị đẩy sang McNemar ngay
#: trong lần chạy tiếp theo. Hai tập tách riêng vì hai cơ chế khớp khác nhau.
BINARY_PREFIXES = ("hit_rate@",)

#: Metric nhị phân nhưng tên không có tiền tố dùng chung được — xem cảnh báo trên.
BINARY_METRICS = frozenset({"precision@1"})

DEFAULT_BOOTSTRAP = 10_000
DEFAULT_SEED = 20260820


@dataclass(frozen=True)
class RunScores:
    """Điểm từng câu của một lần chạy, đọc từ `{run}-per-query.jsonl`."""

    name: str
    scores: dict[str, dict[str, float]]
    n_relevant: dict[str, int]
    relevant_digest: dict[str, str] = field(default_factory=dict)
    """Băm tập nhãn từng câu. Rỗng = lần chạy có trước `W2-03`, chưa ghi băm."""

    @property
    def query_ids(self) -> set[str]:
        return set(self.scores)

    @property
    def has_digests(self) -> bool:
        return any(self.relevant_digest.values())

    def mean_relevant(self, ids: Sequence[str]) -> float:
        return sum(self.n_relevant[q] for q in ids) / len(ids) if ids else 0.0


@dataclass(frozen=True)
class ComparisonRow:
    metric: str
    baseline: float
    candidate: float
    delta: float
    test: str
    p_value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n_baseline_only: int = 0
    n_candidate_only: int = 0
    comparable: bool = True
    note: str = ""

    @property
    def verdict(self) -> str:
        if not self.comparable:
            return "KHÔNG SO ĐƯỢC"
        if self.p_value is not None:
            return "khác biệt thật" if self.p_value < 0.05 else "trong ngưỡng nhiễu"
        if self.ci_low is not None and self.ci_high is not None:
            return (
                "khác biệt thật"
                if (self.ci_low > 0 and self.ci_high > 0) or (self.ci_low < 0 and self.ci_high < 0)
                else "trong ngưỡng nhiễu"
            )
        return "—"


def load_per_query(path: str | Path, *, name: str = "") -> RunScores:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Không thấy {source}. File này do `retrieval_eval` ghi ra "
            "(`{run_name}-per-query.jsonl`) — lần chạy cũ trước khi có nó thì phải chạy lại."
        )
    scores: dict[str, dict[str, float]] = {}
    n_relevant: dict[str, int] = {}
    digests: dict[str, str] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = row["query_id"]
        if qid in scores:
            raise ValueError(f"{source}: truy vấn {qid} xuất hiện hai lần")
        scores[qid] = row["scores"]
        n_relevant[qid] = int(row["n_relevant"])
        # `.get`: file của lần chạy trước `W2-03` không có trường này. Coi là
        # "không biết" chứ không phải "trùng nhau" — xem `_label_mismatch`.
        digests[qid] = str(row.get("relevant_digest", ""))
    return RunScores(
        name=name or source.stem,
        scores=scores,
        n_relevant=n_relevant,
        relevant_digest=digests,
    )


def mcnemar_exact(n_a_only: int, n_b_only: int) -> float:
    """p-value hai phía của McNemar exact test.

    Chỉ những câu **đổi chiều** mang thông tin: câu cả hai đều đúng hoặc cả hai
    đều sai không phân biệt được hai hệ thống. Dưới giả thuyết không, mỗi câu
    đổi chiều nghiêng về bên nào là như nhau → nhị thức(n, 0.5).

    Dùng exact chứ không xấp xỉ chi-square vì `n` ở đây thường bé (3–15 câu), và
    đó chính là vùng mà xấp xỉ sai nhất.
    """
    n = n_a_only + n_b_only
    if n == 0:
        return 1.0
    extreme = max(n_a_only, n_b_only)
    tail: float = sum(math.comb(n, k) for k in range(extreme, n + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def paired_bootstrap(
    diffs: Sequence[float],
    *,
    iterations: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Khoảng tin cậy phần trăm cho trung bình hiệu, lấy mẫu lại theo truy vấn.

    Seed cố định để hai lần chạy công cụ cho cùng con số — một khoảng tin cậy
    nhảy nhót mỗi lần gọi thì không dùng để quyết định được.
    """
    if not diffs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(diffs)
    means: list[float] = []
    for _ in range(iterations):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(alpha / 2 * iterations)]
    hi = means[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return (lo, hi)


def _mean_of(run: RunScores, ids: Sequence[str], metric: str) -> float:
    values = [run.scores[qid][metric] for qid in ids if metric in run.scores[qid]]
    return sum(values) / len(values) if values else float("nan")


def _label_mismatch(baseline: RunScores, candidate: RunScores, shared: Sequence[str]) -> list[str]:
    """Các câu mà hai lần chạy dùng **tập nhãn khác nhau**.

    Hàng rào thứ ba của module này, thêm ở `W2-03`. Hai hàng rào cũ canh *số*
    nhãn và *tập truy vấn*; cả hai đều không thấy trường hợp hai lần chạy có cùng
    số nhãn nhưng nhãn khác nhau. Đó không phải giả thuyết: eval harness lấy
    `fetch_doc_chunks` bằng `getattr` để tính lại nhãn theo span, nên một
    retriever thiếu method đó **lặng lẽ** rơi về nhãn ghi sẵn trong file. Lúc ấy
    `hit_rate` hai bên đo hai bài toán khác nhau và bảng so vẫn hiện ra bình
    thường.

    Câu thiếu băm (file của lần chạy trước `W2-03`) không tính là lệch — không
    biết thì không kết luận. `has_digests` là chỗ để cảnh báo việc không biết.
    """
    return [
        qid
        for qid in shared
        if baseline.relevant_digest.get(qid)
        and candidate.relevant_digest.get(qid)
        and baseline.relevant_digest[qid] != candidate.relevant_digest[qid]
    ]


def compare_runs(
    baseline: RunScores,
    candidate: RunScores,
    *,
    metrics: Sequence[str] = (),
    iterations: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> list[ComparisonRow]:
    shared = sorted(baseline.query_ids & candidate.query_ids)
    if not shared:
        raise ValueError("Hai lần chạy không có truy vấn nào chung — không so được gì")
    only_base = len(baseline.query_ids - candidate.query_ids)
    only_cand = len(candidate.query_ids - baseline.query_ids)
    if only_base or only_cand:
        logger.warning(
            "Hai lần chạy không cùng tập truy vấn: chỉ %s có %d câu, chỉ %s có %d câu. "
            "So trên %d câu chung.",
            baseline.name,
            only_base,
            candidate.name,
            only_cand,
            len(shared),
        )

    base_labels = baseline.mean_relevant(shared)
    cand_labels = candidate.mean_relevant(shared)
    labels_differ = abs(cand_labels - base_labels) > 0.01
    expected_drop = 100 * (1 - base_labels / cand_labels) if cand_labels else 0.0

    if not (baseline.has_digests and candidate.has_digests):
        logger.warning(
            "Ít nhất một lần chạy không ghi `relevant_digest` (%s: %s · %s: %s) — "
            "không kiểm được hai bên có dùng cùng bộ nhãn hay không. File này do "
            "lần chạy trước W2-03 sinh ra; chạy lại `make eval-retrieval` để có băm.",
            baseline.name,
            "có" if baseline.has_digests else "không",
            candidate.name,
            "có" if candidate.has_digests else "không",
        )
    mismatched = _label_mismatch(baseline, candidate, shared)
    if mismatched:
        # Từ chối **toàn bộ**, không lọc bỏ câu lệch rồi so phần còn lại: lọc theo
        # kết quả là đúng cái tự chọn mẫu mà hàng rào #2 của module này nói tới.
        logger.error(
            "%d/%d câu có tập nhãn KHÁC nhau giữa %s và %s (ví dụ: %s). "
            "Hai lần chạy đang đo hai bài toán khác nhau — không so được metric nào.",
            len(mismatched),
            len(shared),
            baseline.name,
            candidate.name,
            ", ".join(mismatched[:5]),
        )
        note = (
            f"{len(mismatched)}/{len(shared)} câu có tập nhãn khác nhau "
            f"(băm `relevant_digest` lệch) — hai lần chạy không cùng bài toán"
        )
        return [
            ComparisonRow(
                metric=metric,
                baseline=_mean_of(baseline, shared, metric),
                candidate=_mean_of(candidate, shared, metric),
                delta=_mean_of(candidate, shared, metric) - _mean_of(baseline, shared, metric),
                test="—",
                comparable=False,
                note=note,
            )
            for metric in (
                list(metrics)
                or sorted(
                    {m for qid in shared for m in baseline.scores[qid]}
                    & {m for qid in shared for m in candidate.scores[qid]}
                )
            )
        ]

    names = list(metrics) or sorted(
        {m for qid in shared for m in baseline.scores[qid]}
        & {m for qid in shared for m in candidate.scores[qid]}
    )

    rows: list[ComparisonRow] = []
    for metric in names:
        pairs = [
            (baseline.scores[qid][metric], candidate.scores[qid][metric])
            for qid in shared
            if metric in baseline.scores[qid] and metric in candidate.scores[qid]
        ]
        if not pairs:
            continue
        mean_b = sum(a for a, _ in pairs) / len(pairs)
        mean_c = sum(b for _, b in pairs) / len(pairs)

        if labels_differ and metric.startswith(CARDINALITY_SENSITIVE):
            rows.append(
                ComparisonRow(
                    metric=metric,
                    baseline=mean_b,
                    candidate=mean_c,
                    delta=mean_c - mean_b,
                    test="—",
                    comparable=False,
                    note=(
                        f"số nhãn/câu đổi {base_labels:.2f} → {cand_labels:.2f}; "
                        f"mẫu số là số nhãn nên metric tụt {expected_drop:.1f}% "
                        "kể cả khi truy hồi y nguyên"
                    ),
                )
            )
            continue

        if metric.startswith(BINARY_PREFIXES) or metric in BINARY_METRICS:
            b_only = sum(1 for a, b in pairs if a > b)
            c_only = sum(1 for a, b in pairs if b > a)
            rows.append(
                ComparisonRow(
                    metric=metric,
                    baseline=mean_b,
                    candidate=mean_c,
                    delta=mean_c - mean_b,
                    test="McNemar exact",
                    p_value=mcnemar_exact(b_only, c_only),
                    n_baseline_only=b_only,
                    n_candidate_only=c_only,
                )
            )
            continue

        diffs = [b - a for a, b in pairs]
        lo, hi = paired_bootstrap(diffs, iterations=iterations, seed=seed)
        rows.append(
            ComparisonRow(
                metric=metric,
                baseline=mean_b,
                candidate=mean_c,
                delta=mean_c - mean_b,
                test=f"bootstrap cặp ({iterations})",
                ci_low=lo,
                ci_high=hi,
            )
        )
    return rows


def format_table(rows: Sequence[ComparisonRow], *, baseline: str, candidate: str) -> str:
    out = [
        f"| metric | {baseline} | {candidate} | Δ | kiểm định | kết luận |",
        "|---|---:|---:|---:|---|---|",
    ]
    for r in rows:
        if not r.comparable:
            detail = f"⚠️ {r.note}"
        elif r.p_value is not None:
            detail = f"p={r.p_value:.3f} · {r.n_baseline_only}↔{r.n_candidate_only} câu đổi chiều"
        else:
            detail = f"CI95 [{r.ci_low:+.4f}, {r.ci_high:+.4f}]"
        out.append(
            f"| `{r.metric}` | {r.baseline:.4f} | {r.candidate:.4f} | "
            f"{r.delta:+.4f} | {detail} | {r.verdict} |"
        )
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="So hai lần chạy retrieval eval, có kiểm định thống kê"
    )
    parser.add_argument("baseline", help="Tên run hoặc đường dẫn tới *-per-query.jsonl")
    parser.add_argument("candidate", help="Tên run hoặc đường dẫn")
    parser.add_argument("--dir", type=Path, default=Path("plans/reports"))
    parser.add_argument("--metrics", nargs="*", default=[])
    parser.add_argument("--iterations", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, help="Ghi bảng Markdown ra file")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)-7s %(message)s")

    def resolve(name: str) -> Path:
        direct = Path(name)
        return direct if direct.exists() else args.dir / f"{name}-per-query.jsonl"

    base = load_per_query(resolve(args.baseline), name=args.baseline)
    cand = load_per_query(resolve(args.candidate), name=args.candidate)
    rows = compare_runs(
        base, cand, metrics=args.metrics, iterations=args.iterations, seed=args.seed
    )
    table = format_table(rows, baseline=args.baseline, candidate=args.candidate)
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n", encoding="utf-8")
        logger.info("Đã ghi %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
