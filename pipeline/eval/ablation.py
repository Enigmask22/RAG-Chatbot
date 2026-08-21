"""Bảng ablation N cấu hình — và câu "cấu hình nào thắng" hỏi cho đúng.

`compare.py` so **hai** lần chạy. Hạng mục `W2-08` cần so **mười bốn**, và đó
không phải cùng một bài toán nhân lên mười ba lần:

## Vì sao "cấu hình thắng" không phải một dòng

Xếp 14 cấu hình theo `ndcg@10` rồi lấy dòng đầu là **chọn cực đại trên 14 ước
lượng nhiễu**. Con số của cái thắng bị lệch lên có hệ thống (nó thắng một phần vì
nó may), và `TD-11` đã đo ngưỡng phân giải của `golden_v1`: với 209 câu, phải
chênh **≥ 6 điểm tuyệt đối** mới phát hiện được. Ba cấu hình đầu bảng chênh nhau
2,5 điểm — tức nằm trong vùng tung đồng xu.

Nên câu trả lời của module này là một **tập**: những cấu hình *không phân biệt
được* với đỉnh bảng. Cách phát biểu đó tránh đúng chỗ lệch nói trên — nó không
tuyên bố "cái này tốt nhất" (một tuyên bố mà phép chọn cực đại làm cho thiên
lệch), nó tuyên bố "không phân biệt được mấy cái này", và tuyên bố ấy an toàn
theo hướng đúng.

Rồi **giá** chọn người thắng trong tập đó. Đo được ở `W2-08`: pool 50 và pool 100
không phân biệt được, mà pool 100 tốn **1,91×** độ trễ.

## Ba cái bẫy đã đo, mỗi cái có một chỗ trong code

1. **`KHÔNG SO ĐƯỢC` không phải "không phân biệt được".** Bản đầu của tôi định
   nghĩa tập tương đương là "không bị đánh bại", và `chunk550` — ô **tệ nhất
   bảng** (`ndcg@10` 0,1215 so với 0,6736) — lọt vào tập, rồi vì nó là dense
   thuần nên nó thành **thành viên rẻ nhất** và suýt được đề xuất. Nhãn của nó
   khác nên mọi phép so đều bị từ chối, và "bị từ chối" trông y như "hoà". Ba rổ
   riêng, không hai: `equivalent`, `beaten`, `incomparable`.
2. **Metric xếp hạng phải nêu tường minh**, vì thứ hạng đổi theo metric: trong
   14 ô này `k=2` và dense đổi chỗ giữa `ndcg@10` và `hit_rate@1`.
3. **Bảng so-với-baseline và tập tương đương là hai loại suy luận**, nên hai
   ngưỡng — xem `ablation_vs_baseline` và `winner_set`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pipeline.eval.compare import (
    DEFAULT_ALPHA,
    DEFAULT_BOOTSTRAP,
    DEFAULT_SEED,
    ComparisonRow,
    RunScores,
    compare_runs,
    load_per_query,
)

__all__ = [
    "PRIMARY_METRICS",
    "AblationCell",
    "WinnerSet",
    "ablation_vs_baseline",
    "discover_runs",
    "format_ablation_table",
    "format_winner_set",
    "load_cells",
    "main",
    "winner_set",
]

logger = logging.getLogger("pipeline.eval.ablation")

#: Metric chính của bảng ablation, **nêu trước khi xem số**.
#:
#: Ba cái, và mỗi cái có một lý do chứ không phải "cho nhiều thông tin hơn":
#:
#: * `ndcg@10` — metric của gate `G2`, tức thứ cả `W2` đã cam kết đo.
#: * `hit_rate@1` — thứ người dùng thấy, và là metric tiêu đề của `W2-05`. Nó đi
#:   đường **McNemar exact**, nên nó là dòng duy nhất trong ba dòng không phụ
#:   thuộc vào một lần lấy mẫu Monte Carlo nào.
#: * `mrr` — metric xếp hạng **không** chia cho số nhãn, tức cái duy nhất trong ba
#:   cái mà `G2` cho phép dùng khi `chunk_size` đổi.
#:
#: Ba cái là một họ 39 phép kiểm cho câu hỏi người thắng (13 × 3). Thêm metric là
#: nới họ ra, tức tự làm yếu chính mình — xem `winner_set`.
PRIMARY_METRICS = ("ndcg@10", "hit_rate@1", "mrr")


@dataclass(frozen=True)
class AblationCell:
    """Một ô của ma trận: điểm từng câu, số tổng, độ trễ, và cấu hình sinh ra nó."""

    name: str
    scores: RunScores
    overall: Mapping[str, float]
    p95_ms: float
    n_relevant_mean: float
    embedding_model: str
    retrieval_mode: str
    branch_options: Mapping[str, object]
    chunk_size: int | None

    def metric(self, name: str) -> float:
        return float(self.overall.get(name, float("nan")))

    @property
    def label(self) -> str:
        """Mô tả ngắn cấu hình — chiều nào đang thay đổi, không phải tên run."""
        bits = [self.embedding_model.split("/")[-1], self.retrieval_mode]
        for key in ("k", "base", "rerank_candidates"):
            value = self.branch_options.get(key)
            if value is not None:
                bits.append(f"{key}={value}")
        if self.chunk_size is not None:
            bits.append(f"chunk={self.chunk_size}")
        return " · ".join(bits)


def discover_runs(directory: Path, prefix: str) -> list[str]:
    """Tên các run có báo cáo trong `directory`, theo tiền tố.

    Đọc từ `*-retrieval.json` chứ không từ `*-per-query.jsonl`: báo cáo tổng là
    thứ mang độ trễ và cấu hình, và một ô thiếu nó thì không lên bảng được.
    """
    suffix = "-retrieval.json"
    found = sorted(p.name[: -len(suffix)] for p in directory.glob(f"{prefix}*{suffix}"))
    if not found:
        raise FileNotFoundError(
            f"Không thấy `{prefix}*-retrieval.json` nào trong {directory}. "
            "Chạy `make exp EXP=exp-001-retrieval` trước."
        )
    return found


def load_cells(directory: Path, names: Sequence[str]) -> list[AblationCell]:
    cells: list[AblationCell] = []
    for name in names:
        report = json.loads((directory / f"{name}-retrieval.json").read_text(encoding="utf-8"))
        config = report.get("config", {})
        chunking = config.get("chunking") or {}
        cells.append(
            AblationCell(
                name=name,
                scores=load_per_query(directory / f"{name}-per-query.jsonl", name=name),
                overall=report["overall"],
                p95_ms=float(report.get("latency_ms", {}).get("p95", float("nan"))),
                n_relevant_mean=float(report.get("n_relevant_mean", float("nan"))),
                embedding_model=str(config.get("embedding_model", "?")),
                retrieval_mode=str(config.get("retrieval_mode", "?")),
                branch_options=dict(config.get("branch_options") or {}),
                chunk_size=chunking.get("chunk_size"),
            )
        )
    return cells


def ablation_vs_baseline(
    cells: Sequence[AblationCell],
    baseline: AblationCell,
    *,
    metrics: Sequence[str] = PRIMARY_METRICS,
    iterations: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[ComparisonRow]]:
    """Mỗi ô so với **một** baseline đã nêu trước. `alpha = 0,05`, KHÔNG hiệu chỉnh.

    Đây là một **panel giả thuyết nêu trước**, không phải một cuộc tìm kiếm: ma
    trận đã được khai trong `configs/eval/exp-001-retrieval.yaml` *trước* khi chạy
    ô nào, và câu hỏi của mỗi hàng ("cấu hình này có hơn baseline không") là một
    giả thuyết riêng, không phải một lựa chọn từ 13 cái.

    Và đây là chỗ hiệu chỉnh **không** thay đổi gì trên thực tế, đo được: hiệu ứng
    ở bảng này cỡ `ndcg@10` 0,16 → 0,67 với `p` cỡ `1e-20`. Hiệu chỉnh cho 39 phép
    kiểm chia `alpha` cho 39; không hàng nào đổi kết luận. Rủi ro đa so sánh nằm
    **toàn bộ** ở `winner_set`, nơi các mức chênh là 1–5 câu — nên hiệu chỉnh ở
    đúng chỗ đó và nói ra ở chỗ này.
    """
    out: dict[str, list[ComparisonRow]] = {}
    for cell in cells:
        if cell.name == baseline.name:
            continue
        out[cell.name] = compare_runs(
            baseline.scores,
            cell.scores,
            metrics=list(metrics),
            iterations=iterations,
            seed=seed,
        )
    return out


@dataclass(frozen=True)
class WinnerSet:
    """Kết quả của câu hỏi "cấu hình nào thắng", phát biểu dưới dạng ba rổ."""

    top: AblationCell
    equivalent: list[AblationCell]
    contested: list[AblationCell]
    beaten: list[AblationCell]
    incomparable: list[AblationCell]
    rows: Mapping[str, Sequence[ComparisonRow]]
    conflicts: Mapping[str, Sequence[str]]
    """Ô **thắng** đỉnh bảng ở một metric chính nào đó — thứ hạng phụ thuộc metric."""

    rank_by: str
    alpha: float
    family_size: int

    @property
    def members(self) -> list[AblationCell]:
        """Tập tương đương **chặt**: đỉnh bảng cộng những ô không metric nào bác.

        Cố ý **không** gộp `contested` vào đây. Một ô bị một metric trong ba nói là
        kém thì nó không "tương đương", nhưng cũng chưa "bị đánh bại" — và nếu nó
        rẻ hơn nhiều thì việc chọn nó là một **phán quyết kỹ thuật**, không phải
        một phép tính. Gộp vào đây là lấy phán quyết đó ra khỏi tay người đọc.
        """
        return [self.top, *self.equivalent]

    @property
    def cheapest(self) -> AblationCell:
        """Thành viên rẻ nhất của tập — chỉ trong `members`, không tính `incomparable`."""
        return min(self.members, key=lambda c: c.p95_ms)

    @property
    def speedup(self) -> float:
        """Đỉnh bảng đắt hơn thành viên rẻ nhất bao nhiêu lần."""
        cheap = self.cheapest.p95_ms
        return self.top.p95_ms / cheap if cheap else float("nan")


def winner_set(
    cells: Sequence[AblationCell],
    *,
    rank_by: str = "ndcg@10",
    metrics: Sequence[str] = PRIMARY_METRICS,
    iterations: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> WinnerSet:
    """Đỉnh bảng theo `rank_by`, rồi so nó với **mọi** ô còn lại. Có hiệu chỉnh.

    Hiệu chỉnh Bonferroni trên `(số ô − 1) × số metric` vì đây **là** một cuộc tìm
    kiếm: mục đích của nó đúng là chọn ra một cái trong nhiều cái. Cùng lý lẽ với
    `compare_by_group` của `W2-08-prep`, và ngược lý lẽ với `ablation_vs_baseline`
    ở trên.

    ## Bốn rổ, không hai

    Phân loại theo **số metric chính** nói "kém", không theo "có tồn tại một cái":

    * `equivalent` — 0 metric nào phân biệt được với đỉnh bảng.
    * `beaten` — **mọi** metric so được đều nói kém.
    * `contested` — các metric chính **không đồng ý**. Đây không phải rổ rác: nó
      là chỗ nên dừng lại đọc, và ở `W2-08` nó chứa đúng cấu hình rẻ hơn 1,91×.
    * `incomparable` — nhãn khác nhau nên không so được metric nào.

    Mọi kết luận không phải "khác biệt thật" — `trong ngưỡng nhiễu`, `KHÔNG ĐỦ
    LỰC`, `KHÔNG KẾT LUẬN`, `TRÁI CHIỀU`, `TRÙNG KHỚP` — đều **không** tính là
    kém, và đó là toàn bộ điểm của việc có nhiều chữ khác nhau.

    ⚠️ `KHÔNG SO ĐƯỢC` là một rổ **riêng**, không phải "không bị đánh bại". Xem
    docstring module: gộp nó vào `equivalent` đưa ô tệ nhất bảng lên làm thành
    viên rẻ nhất của tập thắng.

    `conflicts` ghi ô nào **hơn** đỉnh bảng có ý nghĩa ở một metric chính khác. Nó
    phải là một mục trong kết quả chứ không phải một giả định: nếu nó không rỗng
    thì câu "cấu hình nào thắng" không có câu trả lời độc lập với metric, và đó là
    kết luận, không phải lỗi.
    """
    ranked = sorted(cells, key=lambda c: -c.metric(rank_by))
    top = ranked[0]
    others = ranked[1:]
    family = max(1, len(others) * len(metrics))
    alpha = DEFAULT_ALPHA / family
    logger.info(
        "Đỉnh bảng theo %s: %s (%.4f). Họ = %d ô × %d metric = %d phép kiểm, α = %.5g",
        rank_by,
        top.name,
        top.metric(rank_by),
        len(others),
        len(metrics),
        family,
        alpha,
    )

    rows: dict[str, list[ComparisonRow]] = {}
    equivalent: list[AblationCell] = []
    contested: list[AblationCell] = []
    beaten: list[AblationCell] = []
    incomparable: list[AblationCell] = []
    conflicts: dict[str, list[str]] = {}

    for cell in others:
        cell_rows = compare_runs(
            top.scores,
            cell.scores,
            metrics=list(metrics),
            iterations=iterations,
            seed=seed,
            family_size=family,
        )
        rows[cell.name] = cell_rows
        if all(not row.comparable for row in cell_rows):
            incomparable.append(cell)
            continue
        # `delta = cell − top`, nên `delta < 0` là ô này kém đỉnh bảng.
        worse = [r for r in cell_rows if r.verdict == "khác biệt thật" and r.delta < 0]
        better = [r for r in cell_rows if r.verdict == "khác biệt thật" and r.delta > 0]
        if better:
            conflicts[cell.name] = [r.metric for r in better]
        # Ba rổ theo **số metric** nói "kém", không theo "có hay không có một cái":
        #
        # * 0/3  → `equivalent`. Không metric nào phân biệt được.
        # * đủ   → `beaten`. Mọi metric so được đều nói kém.
        # * còn lại → `contested`. Các metric chính **không đồng ý**, và đó là kết
        #   quả cần đọc, không phải chỗ để chọn một phía.
        #
        # Đo được ở `W2-08` trên đúng cặp quyết định cả bảng: `rc100 → rc50` cho
        # `mrr` "khác biệt thật" (nhưng biên CI là −0,0007 và kiểm định dấu
        # `p = 0,45`), `ndcg@10` `TRÁI CHIỀU`, `hit_rate@1` `KHÔNG ĐỦ LỰC` — tức
        # phép kiểm **exact** duy nhất trong ba cái nói là không đo được. Dán nó
        # "bị đánh bại" thì bảng kết luận phải trả 1,91× độ trễ; dán "tương đương"
        # thì bỏ qua một metric đã nói ngược. Cả hai đều là chọn hộ người đọc.
        usable = [r for r in cell_rows if r.comparable]
        if not worse:
            equivalent.append(cell)
        elif len(worse) == len(usable):
            beaten.append(cell)
        else:
            contested.append(cell)

    return WinnerSet(
        top=top,
        equivalent=equivalent,
        contested=contested,
        beaten=beaten,
        incomparable=incomparable,
        rows=rows,
        conflicts=conflicts,
        rank_by=rank_by,
        alpha=alpha,
        family_size=family,
    )


def format_ablation_table(
    cells: Sequence[AblationCell],
    *,
    rank_by: str,
    metrics: Sequence[str] = PRIMARY_METRICS,
    baseline: AblationCell | None = None,
    comparisons: Mapping[str, Sequence[ComparisonRow]] | None = None,
) -> str:
    """Một hàng mỗi ô: số tổng, độ trễ, và kết luận so với baseline từng metric chính.

    Cột `nhãn/câu` **bắt buộc** có mặt: nó là thứ cho biết hàng nào so được với
    hàng nào. Hai hàng khác nhau ở cột đó thì `recall`/`nDCG`/`MAP` của chúng có
    mẫu số khác nhau, và bảng phải nói ra thay vì để người đọc trừ hai con số.
    """
    ranked = sorted(cells, key=lambda c: -c.metric(rank_by))
    verdicts: dict[str, dict[str, str]] = {}
    for name, rows in (comparisons or {}).items():
        verdicts[name] = {row.metric: row.verdict for row in rows}

    head = ["#", "run", "cấu hình", "nhãn/câu", *[f"`{m}`" for m in metrics], "p95 ms"]
    if comparisons:
        head.append("vs baseline")
    out = [
        "| " + " | ".join(head) + " |",
        "|" + "---|" * 4 + "---:|" * len(metrics) + "---:|" + ("---|" if comparisons else ""),
    ]
    for i, cell in enumerate(ranked, 1):
        cells_text = [
            str(i),
            f"`{cell.name}`",
            cell.label,
            f"{cell.n_relevant_mean:.4f}",
            *[f"{cell.metric(m):.4f}" for m in metrics],
            f"{cell.p95_ms:.1f}",
        ]
        if comparisons:
            if baseline is not None and cell.name == baseline.name:
                cells_text.append("*(baseline)*")
            else:
                got = verdicts.get(cell.name, {})
                shown = [f"`{m.split('@')[0]}`:{got.get(m, '—')}" for m in metrics]
                cells_text.append(" · ".join(shown))
        out.append("| " + " | ".join(cells_text) + " |")
    return "\n".join(out)


def _rank_table(cells: Sequence[AblationCell], metrics: Sequence[str]) -> str:
    """Thứ hạng theo từng metric chính, cạnh nhau — để thấy nó có đổi hay không."""
    orders = {m: [c.name for c in sorted(cells, key=lambda c: -c.metric(m))] for m in metrics}
    out = [
        "| hạng | " + " | ".join(f"`{m}`" for m in metrics) + " |",
        "|---:|" + "---|" * len(metrics),
    ]
    for i in range(len(cells)):
        row = [str(i + 1)] + [f"`{orders[m][i]}`" for m in metrics]
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def format_winner_set(result: WinnerSet, *, metrics: Sequence[str] = PRIMARY_METRICS) -> str:
    from pipeline.eval.compare import format_table

    out = [
        f"## Tập tương đương của `{result.top.name}` (xếp theo `{result.rank_by}`)",
        "",
        f"- Họ **{result.family_size} phép kiểm**, α đã hiệu chỉnh Bonferroni = "
        f"**{result.alpha:.5g}**. Đây là một cuộc **tìm kiếm** (chọn một trong "
        f"{result.family_size // max(len(metrics), 1) + 1} ô), nên nó hiệu chỉnh — khác bảng "
        "so-với-baseline ở trên, là một panel giả thuyết nêu trước.",
        f"- **Không phân biệt được với đỉnh bảng: {len(result.members)} ô.** "
        f"Các metric chính **không đồng ý**: {len(result.contested)} ô. "
        f"Kém đỉnh bảng ở mọi metric: {len(result.beaten)} ô. "
        f"**Không so được** (nhãn khác): {len(result.incomparable)} ô.",
        '- ⚠️ `KHÔNG SO ĐƯỢC` là rổ riêng, **không** phải "không phân biệt được". '
        "Gộp hai cái đó đưa ô tệ nhất bảng lên làm thành viên rẻ nhất của tập thắng "
        "— đã xảy ra ở bản đầu, xem docstring module.",
        "",
        "### Tập tương đương, xếp theo giá",
        "",
        "| run | cấu hình | " + " | ".join(f"`{m}`" for m in metrics) + " | p95 ms |",
        "|---|---|" + "---:|" * (len(metrics) + 1),
    ]
    for cell in sorted(result.members, key=lambda c: c.p95_ms):
        mark = " ⬅ đỉnh bảng" if cell.name == result.top.name else ""
        out.append(
            f"| `{cell.name}`{mark} | {cell.label} | "
            + " | ".join(f"{cell.metric(m):.4f}" for m in metrics)
            + f" | {cell.p95_ms:.1f} |"
        )
    cheap = result.cheapest
    if cheap.name != result.top.name:
        out += [
            "",
            f"➡️ **Đỉnh bảng đắt hơn {result.speedup:.2f}× thành viên rẻ nhất của chính "
            f"tập nó** (`{result.top.name}` {result.top.p95_ms:.1f} ms vs "
            f"`{cheap.name}` {cheap.p95_ms:.1f} ms) mà **không** phân biệt được về "
            "chất lượng. Kết luận kỹ thuật của bảng ablation là ô rẻ, không phải ô đầu.",
        ]
    else:
        out += [
            "",
            "➡️ Đỉnh bảng **cũng là** ô rẻ nhất trong tập tương đương — không có gì phải đánh đổi.",
        ]

    if result.contested:
        out += [
            "",
            "### ⚠️ Tranh chấp: các metric chính không đồng ý",
            "",
            "Những ô sau bị **một phần** metric chính nói là kém đỉnh bảng, phần còn "
            "lại thì không. Đây là chỗ phán quyết kỹ thuật thật sự nằm, và công cụ "
            "**cố ý không** quyết hộ: cột giá cho biết mua sự chắc chắn ấy tốn bao nhiêu.",
            "",
            "| run | cấu hình | metric nói kém | metric không kết luận | p95 ms | so đỉnh bảng |",
            "|---|---|---|---|---:|---:|",
        ]
        for cell in sorted(result.contested, key=lambda c: c.p95_ms):
            cell_rows = list(result.rows[cell.name])
            worse = [r.metric for r in cell_rows if r.verdict == "khác biệt thật" and r.delta < 0]
            other = [f"{r.metric} ({r.verdict})" for r in cell_rows if r.metric not in worse]
            ratio = result.top.p95_ms / cell.p95_ms if cell.p95_ms else float("nan")
            out.append(
                f"| `{cell.name}` | {cell.label} | "
                + ", ".join(f"`{m}`" for m in worse)
                + " | "
                + ", ".join(f"`{m}`" for m in other)
                + f" | {cell.p95_ms:.1f} | **{ratio:.2f}× rẻ hơn** |"
            )

    if result.conflicts:
        out += [
            "",
            "### ⚠️ Thứ hạng phụ thuộc metric",
            "",
            "Những ô sau **hơn** đỉnh bảng có ý nghĩa ở một metric chính khác, nên "
            'câu "cấu hình nào thắng" không có câu trả lời độc lập với metric:',
            "",
        ]
        for name, hit in result.conflicts.items():
            out.append(f"- `{name}`: " + ", ".join(f"`{m}`" for m in hit))

    if result.incomparable:
        out += [
            "",
            "### Không so được (nhãn khác nhau)",
            "",
            *[
                f"- `{c.name}` — {c.n_relevant_mean:.4f} nhãn/câu vs "
                f"{result.top.n_relevant_mean:.4f} của đỉnh bảng"
                for c in result.incomparable
            ],
        ]

    out += ["", "### Từng ô, ba metric chính", ""]
    for name, rows in result.rows.items():
        out += [
            f"#### `{result.top.name}` → `{name}`",
            "",
            format_table(list(rows), baseline=result.top.name, candidate=name, show_n=True),
            "",
        ]
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bảng ablation N cấu hình, có p/CI từng dòng và một tập tương đương"
    )
    parser.add_argument("--dir", type=Path, default=Path("plans/reports/runs"))
    parser.add_argument("--prefix", default="e1-", help="Tiền tố tên run của grid cần đọc")
    parser.add_argument("--runs", nargs="*", default=[], help="Nêu tường minh, bỏ qua --prefix")
    parser.add_argument("--baseline", required=True, help="Ô làm mốc cho bảng so-với-baseline")
    parser.add_argument(
        "--rank-by",
        default="ndcg@10",
        help="Metric xếp hạng. PHẢI nêu tường minh vì thứ hạng đổi theo metric.",
    )
    parser.add_argument("--metrics", nargs="*", default=list(PRIMARY_METRICS))
    parser.add_argument("--iterations", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)-7s %(message)s")

    names = list(args.runs) or discover_runs(args.dir, args.prefix)
    cells = load_cells(args.dir, names)
    by_name = {c.name: c for c in cells}
    if args.baseline not in by_name:
        parser.error(f"Không thấy ô {args.baseline!r}. Có: {', '.join(sorted(by_name))}")
    if args.rank_by not in cells[0].overall:
        parser.error(
            f"Không có metric {args.rank_by!r} trong báo cáo. "
            f"Có: {', '.join(sorted(cells[0].overall))}"
        )
    baseline = by_name[args.baseline]

    comparisons = ablation_vs_baseline(
        cells, baseline, metrics=args.metrics, iterations=args.iterations, seed=args.seed
    )
    result = winner_set(
        cells,
        rank_by=args.rank_by,
        metrics=args.metrics,
        iterations=args.iterations,
        seed=args.seed,
    )

    parts = [
        f"# Ablation: {len(cells)} cấu hình, mốc `{baseline.name}`",
        "",
        f"- Xếp theo `{args.rank_by}`. Metric chính: "
        + ", ".join(f"`{m}`" for m in args.metrics)
        + ".",
        "- Bảng dưới **không** hiệu chỉnh đa so sánh: nó là một panel giả thuyết nêu "
        "trước (ma trận đã khai trong config trước khi chạy ô nào), và hiệu ứng ở đây "
        "cỡ `p ~ 1e-20` nên hiệu chỉnh không đổi hàng nào. Chỗ hiệu chỉnh là **tập "
        "tương đương** bên dưới, nơi mức chênh là 1–5 câu.",
        "",
        "## Bảng ablation",
        "",
        format_ablation_table(
            cells,
            rank_by=args.rank_by,
            metrics=args.metrics,
            baseline=baseline,
            comparisons=comparisons,
        ),
        "",
        "## Thứ hạng có đổi theo metric không",
        "",
        _rank_table(cells, args.metrics),
        "",
        format_winner_set(result, metrics=args.metrics),
    ]
    text = "\n".join(parts)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        logger.info("Đã ghi %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
