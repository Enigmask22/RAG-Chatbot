"""So mức cải thiện **giữa các nhóm** — và câu "nhóm nào cải thiện nhiều nhất".

DoD `W2-09` đòi "ít nhất 2 nhận xét về category nào cải thiện nhiều nhất".
`compare_by_group` (`W2-08-prep`) trả lời được một nửa: nó cho `Δ` của **từng**
nhóm kèm kiểm định `Δ ≠ 0`. Nhưng "nhóm nào **nhiều nhất**" là một câu khác — nó
so `Δ_A` với `Δ_B`, và không có hàng nào trong bảng đó kiểm chuyện ấy.

## Xếp hạng theo `Δ` là phép chọn cực đại, lần thứ hai

`W2-08` học điều này cho **cấu hình**: xếp 14 ô rồi lấy dòng đầu là chọn cực đại
trên 14 ước lượng nhiễu, nên câu trả lời phải là một **tập**. Cùng lập luận, cùng
kết luận, chỉ đổi trục: 6 nhóm, mỗi nhóm một `Δ` có sai số riêng, và nhóm nhỏ
nhất có **4 câu**. Nên module này cũng trả về một tập chứ không một tên.

Khác `ablation.py` ở đúng một chỗ, và chỗ đó đổi phép kiểm: hai **cấu hình** đo
trên cùng bộ câu hỏi nên bootstrap **cặp**; hai **nhóm** là hai tập câu **rời
nhau** nên bootstrap **không cặp** — lấy mẫu lại độc lập trong từng nhóm rồi trừ
hai trung bình. Dùng nhầm phép cặp ở đây thì phải ghép câu nhóm này với câu nhóm
kia, tức bịa ra một tương quan không tồn tại.

## Hàng rào mới: mẫu số của metric đổi giữa các NHÓM

`compare.py` có ba hàng rào, và cả ba canh trục "hai lần chạy". Không cái nào
thấy được chuyện này:

    factoid      1,0147 nhãn/câu
    aggregation  2,4231 nhãn/câu      → chênh 2,39×

`recall@k`, `nDCG@k`, `MAP@k` có mẫu số là số nhãn. Trong **một** nhóm thì hai
lần chạy dùng đúng bộ nhãn ấy nên `Δ` của nhóm đó hợp lệ — hàng rào băm `W2-03`
nhìn đúng chỗ và im lặng, hợp lý. Nhưng đem `Δ recall@5` của `factoid` **so với**
của `aggregation` là so hai thang đo: sửa được một câu ở `factoid` đẩy recall lên
1,0, còn ở `aggregation` chỉ đẩy lên 0,33.

Đo được ngay trong dữ liệu: xếp theo `hit_rate@5` (0/1, không mẫu số) thì
`aggregation` hạng **2**; xếp theo `ndcg@10` thì nó hạng **5**. Nhóm bị xê dịch
nhiều nhất đúng là nhóm nhiều nhãn nhất.

⚠️ Hệ quả cho `G2`: `ndcg@10` — metric mà cả `W2` cam kết — **không** trả lời
được câu "nhóm nào cải thiện nhiều nhất". Còn 9/15 metric trả lời được.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pipeline.eval.compare import (
    CARDINALITY_SENSITIVE,
    DEFAULT_ALPHA,
    DEFAULT_BOOTSTRAP,
    DEFAULT_SEED,
    GROUP_DIMENSIONS,
    MIN_TAIL_RESAMPLES,
    RunScores,
    load_per_query,
    resolve_iterations,
)

__all__ = [
    "SCALE_FREE_METRICS",
    "ContrastResult",
    "ContrastRow",
    "GroupDelta",
    "format_contrast",
    "group_contrast",
    "group_deltas",
    "is_scale_free",
    "main",
    "unpaired_bootstrap",
]

logger = logging.getLogger("pipeline.eval.contrast")


def is_scale_free(metric: str) -> bool:
    """`Δ` của metric này so được **giữa hai nhóm có số nhãn khác nhau** không?

    Không cùng câu hỏi với `comparable` của `compare.py`. Ở đó câu hỏi là "hai
    lần chạy có dùng cùng bộ nhãn không" và trả lời được bằng băm. Ở đây bộ nhãn
    hai bên **cố ý** khác — chúng là hai tập câu khác nhau — nên câu hỏi thành
    "metric này có phụ thuộc số nhãn không", và đó là thuộc tính của **metric**,
    không của dữ liệu. Nên hàm này không nhận dữ liệu.
    """
    return not metric.startswith(CARDINALITY_SENSITIVE)


#: Chín metric mà `Δ` so được giữa các nhóm: `hit_rate@k` (0/1 mỗi câu),
#: `precision@k` (mẫu số là `k`), `mrr` (mẫu số là **hạng**). Sáu metric còn lại
#: — `recall@k`, `ndcg@10`, `map@20` — có mẫu số là số nhãn.
#:
#: Danh sách này chỉ để tài liệu hoá và để test ghim; đường quyết định thật là
#: `is_scale_free`, vì nó đúng cả với metric thêm sau này.
SCALE_FREE_METRICS = (
    "hit_rate@1",
    "hit_rate@5",
    "hit_rate@10",
    "hit_rate@20",
    "mrr",
    "precision@1",
    "precision@5",
    "precision@10",
    "precision@20",
)


@dataclass(frozen=True)
class GroupDelta:
    """`Δ` của một nhóm, kèm **từng** hiệu để bootstrap lấy mẫu lại được."""

    group: str
    metric: str
    n: int
    baseline: float
    candidate: float
    diffs: tuple[float, ...]
    labels_per_query: float

    @property
    def delta(self) -> float:
        return self.candidate - self.baseline

    @property
    def n_better(self) -> int:
        return sum(1 for d in self.diffs if d > 0)

    @property
    def n_worse(self) -> int:
        return sum(1 for d in self.diffs if d < 0)


def unpaired_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    *,
    alpha: float = DEFAULT_ALPHA,
    iterations: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, tuple[float, float], int]:
    """Khoảng tin cậy của `mean(left) − mean(right)` cho hai mẫu **rời nhau**.

    Trả `(biên dưới, biên trên, dao động của biên gần 0 nhất, số mẫu trong đuôi)`.

    Lấy mẫu lại **độc lập** trong từng bên, đúng `n` của bên đó — không ghép cặp,
    vì không có cặp nào để ghép. Giữ nguyên `n` từng bên cũng là chỗ nhóm 4 câu
    tự khai ra nó là nhóm 4 câu: trung bình bootstrap của nó chỉ nhận 5 giá trị.

    Dao động của biên tính như `W2-08`: số mẫu dưới biên là `Binomial(B, α/2)`,
    sd `sqrt(tail)`, nên đọc lại biên ở `tail ± sqrt(tail)` của **chính dãy đã
    sắp** — không tốn thêm một lần lấy mẫu nào.
    """
    if not left or not right:
        raise ValueError("Cả hai nhóm phải có ít nhất một câu.")
    if iterations == DEFAULT_BOOTSTRAP:
        iterations = resolve_iterations(alpha, iterations)
    rng = random.Random(seed)
    n_l, n_r = len(left), len(right)
    stats: list[float] = []
    for _ in range(iterations):
        m_l = sum(left[rng.randrange(n_l)] for _ in range(n_l)) / n_l
        m_r = sum(right[rng.randrange(n_r)] for _ in range(n_r)) / n_r
        stats.append(m_l - m_r)
    stats.sort()

    lo_i = max(0, int(iterations * alpha / 2) - 1)
    hi_i = min(iterations - 1, int(iterations * (1 - alpha / 2)) - 1)
    low, high = stats[lo_i], stats[hi_i]

    spread = max(1, round(math.sqrt(max(lo_i, 1))))
    near_i = lo_i if abs(low) <= abs(high) else hi_i
    jitter = (
        stats[max(0, near_i - spread)],
        stats[min(iterations - 1, near_i + spread)],
    )
    return low, high, jitter, lo_i


def _scale_gap(left: GroupDelta, right: GroupDelta) -> float:
    """Hai thang đo lệch nhau bao nhiêu **lần**, in ra để người đọc tự cân.

    Hàng rào là **nhị phân** — lệch một chút cũng từ chối — và con số này không
    đổi điều đó. Nó có mặt vì hai lần từ chối trông giống hệt nhau trong bảng mà
    thật ra rất khác nhau: `cross_lingual` vs `adversarial` lệch **1,00×** còn
    `cross_lingual` vs `aggregation` lệch **2,22×**. Người đọc phải thấy được
    cái nào là sát nút.

    Tỉ số lớn/nhỏ chứ không phải phần trăm: phần trăm buộc phải chọn lấy bên nào
    làm mẫu số, và cùng một cặp cho 54,9% hay 122% tuỳ chọn — một con số mà đổi
    hơn hai lần theo cách viết thì không phải con số để đọc lướt.

    Vì sao vẫn nhị phân thay vì đặt ngưỡng dung sai: `compare.py` đã từ chối
    `CARDINALITY_SENSITIVE` khi số nhãn đổi **bất kể** đổi bao nhiêu, và một
    ngưỡng dung sai ở đây là một con số bịa ra cạnh một luật đã có. Nhất quán
    với luật cũ đáng hơn vài dòng thu lại được.
    """
    a, b = left.labels_per_query, right.labels_per_query
    if not a or not b:
        return float("inf")
    return max(a, b) / min(a, b)


@dataclass(frozen=True)
class ContrastRow:
    """Một dòng "đỉnh bảng hơn nhóm này bao nhiêu" — hai nhóm **rời nhau**."""

    top: str
    other: str
    metric: str
    gap: float
    n_top: int
    n_other: int
    ci_low: float | None = None
    ci_high: float | None = None
    ci_jitter: tuple[float, float] | None = None
    tail: int = 0
    min_increment: float = 0.0
    comparable: bool = True
    note: str = ""
    alpha: float = DEFAULT_ALPHA
    family_size: int = 1

    @property
    def grid_step(self) -> float:
        """Bước phân giải: dịch **một** câu của nhóm nhỏ hơn đổi `gap` bao nhiêu.

        Nhóm nhỏ quyết định, không phải tổng hai nhóm: `gap` là hiệu hai trung
        bình, nên bước thô nhất đến từ bên có mẫu số nhỏ nhất. Với `table_lookup`
        (4 câu) và `hit_rate@5` thì bước là **0,25** — mọi khoảng hẹp hơn 0,25
        quanh 0 là chuyện của lưới, không của dữ liệu.
        """
        n = min(self.n_top, self.n_other)
        return self.min_increment / n if n else 0.0

    @property
    def ci_within_resolution(self) -> bool:
        if not self.comparable or self.ci_low is None or self.ci_high is None:
            return False
        return min(abs(self.ci_low), abs(self.ci_high)) < self.grid_step

    @property
    def mc_unstable(self) -> bool:
        if self.ci_jitter is None:
            return False
        lo, hi = self.ci_jitter
        return lo <= 0.0 <= hi

    @property
    def underpowered(self) -> bool:
        """Đuôi mỏng hơn `MIN_TAIL_RESAMPLES` thì biên đọc từ quá ít mẫu lại.

        ⚠️ Khác `compare.py`, và khác ở chỗ quan trọng. Ở đó `underpowered` là
        **trần `p` của McNemar**, một tính chất của *dữ liệu* — thêm mẫu lại
        không chữa được, chỉ thêm **câu hỏi** mới chữa được. Ở đây không có
        đường McNemar nào (hai nhóm rời nhau thì không câu nào "đổi chiều"), nên
        thứ duy nhất đo được là đuôi có đủ dày không — một tính chất của **số
        mẫu lại**, và nâng `B` chữa được. Cùng tên, hai nguyên nhân, hai cách
        chữa; nếu không tách ra thì người đọc sẽ nâng `B` cho ca không chữa
        được và bỏ tay với ca chữa được.
        """
        return self.comparable and 0 < self.tail < MIN_TAIL_RESAMPLES

    @property
    def verdict(self) -> str:
        if not self.comparable:
            return "KHÔNG SO ĐƯỢC"
        if self.ci_low is None or self.ci_high is None:
            return "—"
        excludes_zero = (self.ci_low > 0 and self.ci_high > 0) or (
            self.ci_low < 0 and self.ci_high < 0
        )
        unreadable = self.ci_within_resolution or self.mc_unstable or self.underpowered
        if excludes_zero:
            return "KHÔNG KẾT LUẬN" if unreadable else "hơn thật"
        return "KHÔNG KẾT LUẬN" if unreadable else "không phân biệt được"


@dataclass(frozen=True)
class ContrastResult:
    metric: str
    dimension: str
    deltas: tuple[GroupDelta, ...]
    rows: tuple[ContrastRow, ...]
    alpha: float
    family_size: int
    scale_free: bool

    @property
    def top(self) -> GroupDelta:
        return self.deltas[0]

    @property
    def tied(self) -> tuple[str, ...]:
        """Nhóm **không phân biệt được** với đỉnh bảng — câu trả lời thật."""
        return tuple(r.other for r in self.rows if r.verdict == "không phân biệt được")

    @property
    def beaten(self) -> tuple[str, ...]:
        return tuple(r.other for r in self.rows if r.verdict == "hơn thật")

    @property
    def unresolved(self) -> tuple[str, ...]:
        """`KHÔNG KẾT LUẬN` **và** `KHÔNG SO ĐƯỢC` — cả hai đều không phải "hoà".

        Rổ riêng, không gộp vào `tied`. Đây đúng là lỗi đắt nhất của `W2-08`:
        gộp "không có thông tin" vào "thông tin là bằng nhau" đưa ô tệ nhất bảng
        lên làm người thắng.
        """
        return tuple(r.other for r in self.rows if r.verdict in ("KHÔNG KẾT LUẬN", "KHÔNG SO ĐƯỢC"))

    @property
    def members(self) -> tuple[str, ...]:
        """Tập "cải thiện nhiều nhất": đỉnh bảng + những nhóm hoà với nó."""
        return (self.top.group, *self.tied)


def group_deltas(
    baseline: RunScores,
    candidate: RunScores,
    dimension: str,
    metric: str,
) -> list[GroupDelta]:
    """`Δ` từng nhóm cho **một** metric, xếp giảm dần theo `Δ`."""
    if dimension not in GROUP_DIMENSIONS:
        raise ValueError(f"Chiều không hợp lệ: {dimension!r}. Hợp lệ: {list(GROUP_DIMENSIONS)}.")
    shared = sorted(baseline.query_ids & candidate.query_ids)
    if not shared:
        raise ValueError("Hai lần chạy không có câu nào chung.")

    base = baseline.subset(shared)
    out: list[GroupDelta] = []
    for value, ids in base.groups(dimension).items():
        usable = [q for q in ids if metric in base.scores[q] and metric in candidate.scores[q]]
        if not usable:
            logger.warning("Nhóm %s không có câu nào có metric %s — bỏ qua.", value, metric)
            continue
        diffs = tuple(candidate.scores[q][metric] - base.scores[q][metric] for q in usable)
        out.append(
            GroupDelta(
                group=value or "(không nhãn)",
                metric=metric,
                n=len(usable),
                baseline=sum(base.scores[q][metric] for q in usable) / len(usable),
                candidate=sum(candidate.scores[q][metric] for q in usable) / len(usable),
                diffs=diffs,
                labels_per_query=base.mean_relevant(usable),
            )
        )
    return sorted(out, key=lambda g: -g.delta)


def group_contrast(
    baseline: RunScores,
    candidate: RunScores,
    dimension: str,
    metric: str,
    *,
    iterations: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    correct: bool = True,
) -> ContrastResult:
    """Nhóm nào cải thiện nhiều nhất — trả về một **tập**, không một tên.

    Hiệu chỉnh Bonferroni cho `số nhóm − 1` phép so (mỗi nhóm còn lại so với đỉnh
    bảng), vì đây là một **cuộc tìm kiếm**: đỉnh bảng chọn *bằng dữ liệu*. Không
    hiệu chỉnh cho cả `C(k,2)` cặp — module này không hỏi mọi cặp, nó hỏi "ai
    không phân biệt được với người dẫn đầu", và đó là `k − 1` câu hỏi.

    `correct=False` chỉ để test dựng ca đối chứng — đừng dùng khi báo cáo.
    """
    deltas = group_deltas(baseline, candidate, dimension, metric)
    if len(deltas) < 2:
        raise ValueError(f"Chiều {dimension!r} chỉ có {len(deltas)} nhóm — không có gì để so.")

    family = (len(deltas) - 1) if correct else 1
    alpha = DEFAULT_ALPHA / family
    scale_free = is_scale_free(metric)
    if not scale_free:
        logger.error(
            "Metric %s có mẫu số là SỐ NHÃN, mà số nhãn/câu đổi theo nhóm (%s). "
            "`Δ` hai nhóm nằm trên hai thang đo — không xếp hạng được.",
            metric,
            " · ".join(f"{g.group} {g.labels_per_query:.4f}" for g in deltas),
        )
    logger.info(
        "Chiều %s: %d nhóm → %d phép so với đỉnh bảng. alpha = %.5g%s",
        dimension,
        len(deltas),
        len(deltas) - 1,
        alpha,
        "" if correct else "  ⚠️ KHÔNG hiệu chỉnh (correct=False)",
    )

    top = deltas[0]
    rows: list[ContrastRow] = []
    for other in deltas[1:]:
        if not scale_free:
            rows.append(
                ContrastRow(
                    top=top.group,
                    other=other.group,
                    metric=metric,
                    gap=top.delta - other.delta,
                    n_top=top.n,
                    n_other=other.n,
                    comparable=False,
                    note=(
                        f"mẫu số là số nhãn, mà `{top.group}` có "
                        f"{top.labels_per_query:.4f} nhãn/câu còn `{other.group}` có "
                        f"{other.labels_per_query:.4f} — hai thang đo, lệch "
                        f"**{_scale_gap(top, other):.2f}×**"
                    ),
                    alpha=alpha,
                    family_size=family,
                )
            )
            continue

        low, high, jitter, tail = unpaired_bootstrap(
            top.diffs, other.diffs, alpha=alpha, iterations=iterations, seed=seed
        )
        nonzero = [abs(d) for d in (*top.diffs, *other.diffs) if d]
        rows.append(
            ContrastRow(
                top=top.group,
                other=other.group,
                metric=metric,
                gap=top.delta - other.delta,
                n_top=top.n,
                n_other=other.n,
                ci_low=low,
                ci_high=high,
                ci_jitter=jitter,
                tail=tail,
                min_increment=min(nonzero) if nonzero else 0.0,
                alpha=alpha,
                family_size=family,
            )
        )

    return ContrastResult(
        metric=metric,
        dimension=dimension,
        deltas=tuple(deltas),
        rows=tuple(rows),
        alpha=alpha,
        family_size=family,
        scale_free=scale_free,
    )


def _row_detail(row: ContrastRow) -> str:
    """Ô "kiểm định": mỗi cờ in ra **con số làm nó bật**, không chỉ in tên nó."""
    if not row.comparable:
        return row.note
    bits = [f"CI{100 * (1 - row.alpha):.4g}% [{row.ci_low:+.4f}, {row.ci_high:+.4f}]"]
    if row.underpowered:
        bits.append(f"**đuôi {row.tail} mẫu lại** < {MIN_TAIL_RESAMPLES} — biên đọc từ quá ít mẫu")
    if row.ci_within_resolution:
        near = min(abs(row.ci_low or 0.0), abs(row.ci_high or 0.0))
        bits.append(
            f"**biên cách 0 dưới một bước lưới** ({near:.5f} < "
            f"{row.min_increment:.4g}/{min(row.n_top, row.n_other)} = {row.grid_step:.5f})"
        )
    if row.mc_unstable and row.ci_jitter is not None:
        bits.append(
            f"**biên không ổn định**: chính nó dao động "
            f"[{row.ci_jitter[0]:+.4f}, {row.ci_jitter[1]:+.4f}]"
        )
    return " · ".join(bits)


def format_contrast(result: ContrastResult) -> str:
    """Bảng Markdown: xếp hạng nhóm + tập "cải thiện nhiều nhất"."""
    out = [f"### `{result.metric}` theo `{result.dimension}`", ""]
    if not result.scale_free:
        out += [
            f"> ⚠️ **`{result.metric}` không xếp hạng được giữa các nhóm.** Mẫu số của "
            "nó là số nhãn, mà số nhãn/câu đổi theo nhóm — `Δ` hai nhóm nằm trên hai "
            "thang đo. Bảng dưới in `Δ` để tham khảo, **không** để xếp hạng.",
            "",
        ]
    out += [
        "| hạng | nhóm | n | nhãn/câu | mốc | sau | Δ | tốt↔xấu |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for i, g in enumerate(result.deltas, 1):
        mark = " ⬅ đỉnh bảng" if i == 1 else ""
        out.append(
            f"| {i} | `{g.group}`{mark} | {g.n} | {g.labels_per_query:.4f} | "
            f"{g.baseline:.4f} | {g.candidate:.4f} | {g.delta:+.4f} | "
            f"{g.n_better}↔{g.n_worse} |"
        )

    out += [
        "",
        f"Đỉnh bảng **chọn bằng dữ liệu**, nên {len(result.rows)} phép so dưới đây "
        f"hiệu chỉnh Bonferroni: α = {result.alpha:.5g}.",
        "",
        f"| `{result.top.group}` hơn | khoảng cách | n | kiểm định | kết luận |",
        "|---|---:|---:|---|---|",
    ]
    for row in result.rows:
        out.append(
            f"| `{row.other}` | {row.gap:+.4f} | {row.n_top}/{row.n_other} | "
            f"{_row_detail(row)} | {row.verdict} |"
        )

    if result.scale_free:
        members = " · ".join(f"`{m}`" for m in result.members)
        out += [
            "",
            f"➡️ **Tập cải thiện nhiều nhất: {members}** "
            f"({len(result.members)}/{len(result.deltas)} nhóm) · "
            f"hơn thật {len(result.beaten)} nhóm · "
            f"không kết luận được {len(result.unresolved)} nhóm.",
        ]
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Nhóm nào cải thiện nhiều nhất — so Δ giữa các nhóm rời nhau"
    )
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--dir", type=Path, default=Path("plans/reports/runs"))
    parser.add_argument("--by", default="category", choices=GROUP_DIMENSIONS)
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["hit_rate@5", "hit_rate@1", "mrr"],
        help="Chỉ metric KHÔNG có mẫu số là số nhãn mới xếp hạng được giữa các nhóm.",
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)-7s %(message)s")

    def resolve(name: str) -> Path:
        direct = Path(name)
        return direct if direct.exists() else args.dir / f"{name}-per-query.jsonl"

    base = load_per_query(resolve(args.baseline), name=args.baseline)
    cand = load_per_query(resolve(args.candidate), name=args.candidate)

    parts = [
        f"# `{args.baseline}` → `{args.candidate}`: nhóm nào cải thiện nhiều nhất",
        "",
        f"- Chia theo `{args.by}`. Xếp nhóm theo `Δ`, rồi kiểm **từng** nhóm còn lại có "
        "phân biệt được với đỉnh bảng hay không — câu trả lời là một **tập**.",
        "- Bootstrap **không cặp**: hai nhóm là hai tập câu rời nhau, không có cặp nào để ghép.",
        "- ⚠️ `Δ` của `recall@k`/`ndcg@k`/`map@k` **không** xếp hạng được giữa các nhóm "
        "— mẫu số là số nhãn, mà số nhãn/câu đổi theo nhóm.",
        "",
    ]
    for metric in args.metrics:
        result = group_contrast(
            base,
            cand,
            args.by,
            metric,
            iterations=args.iterations,
            seed=args.seed,
        )
        parts += [format_contrast(result), ""]

    text = "\n".join(parts)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        logger.info("Đã ghi %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
