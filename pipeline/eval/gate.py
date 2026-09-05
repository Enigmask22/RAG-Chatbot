"""Gate phát hành: một bundle có được đi tiếp hay không. `W5-05`.

Đây là chỗ duy nhất trong dự án nói **KHÔNG**. Mọi hạng mục trước đó sinh ra số;
hạng mục này biến số thành một quyết định có exit code. Bốn quyết định thiết kế
dưới đây quyết định nó có nói KHÔNG đúng chỗ hay không.

## ⭐⭐ 1. "Không so được" là một phán quyết THỨ BA, không phải một kiểu FAIL

Cám dỗ là hai trạng thái: PASS và FAIL. Nhưng ba câu dưới đây khác hẳn nhau:

* *"đo được, và tệ hơn champion"* → FAIL. Hành động: sửa hệ thống.
* *"đo được, và đạt"* → PASS.
* *"hai con số này không đặt cạnh nhau được"* → **INCOMPARABLE**. Hành động: sửa
  **phép đo**, không sửa hệ thống.

Gộp cái thứ ba vào FAIL sẽ đẩy người ta đi tối ưu một hệ thống không có gì sai.
Gộp nó vào PASS thì tệ hơn nhiều: nó thả một bundle chưa ai so với gì cả.

Cả ba đều exit khác 0 trừ PASS — nhưng chúng in ra ba câu khác nhau, và
`GateVerdict.status` phân biệt được bằng máy.

## ⭐⭐ 2. Chuỗi bằng nhau chưa đủ để kết luận "cùng một giám khảo"

`evaluated_with_generator` tồn tại từ `W4-01` để ép so like-for-like. Nhưng cả
hai bundle đang có trên đĩa đều ghi `deepseek-chat@2026-09`, và
`DEEPSEEK_ALIASES` trong `rag_core.llm` nói rằng `deepseek-chat` là **con trỏ
phía server** — `W5-03` đo được nó được phục vụ bởi `deepseek-v4-flash`, cùng
model với `deepseek-reasoner`.

Nghĩa là hai chuỗi bằng nhau **không** chứng minh hai lần đo dùng cùng model.
Trường sinh ra để bảo đảm danh tính đang mang một danh tính không ổn định. Đây
đúng là lý do quy tắc cứng #1 cấm OpenRouter preset, chỉ kín đáo hơn — nên gate
từ chối cả hai, và `reject_alias_identity` mặc định bật.

## ⭐⭐ 3. Kiểm tính hợp lệ của phép đo TRƯỚC khi so với ngưỡng

`W5-04` đo được điều này bằng tiền thật: **một judge hỏng không cho điểm thấp,
nó cho điểm tuyệt đối**. Bật suy luận làm mất 32/50 phán quyết vì chuỗi suy luận
ăn hết `max_tokens`; ca bị mất có ngữ cảnh dài gần gấp đôi nên toàn bộ mệnh đề
thất bại nằm trong nhóm bị loại, và 18 ca sống sót đều `SUPPORTED` ⇒
faithfulness `1,0000`.

Một gate chỉ so `faithfulness >= 0,92` sẽ **thả** lần chạy ấy với điểm cao nhất
có thể. Nên `validity` chạy trước `absolute`, và bundle nào khai
`generation_metrics` mà không khai `unjudged_rate` thì đỏ: "không biết" phải đỏ
khi chế độ hỏng đã đo lệch về phía điểm cao.

## ⭐ 4. Ngưỡng nằm trong YAML, nhưng LUẬT nằm trong mã

`gate.yaml` chọn được con số; nó **không** chọn được có kiểm hay không. Không có
cờ nào tắt `comparability` hay bỏ qua một metric — vì cái cờ ấy sẽ được bật vào
đúng ngày người ta cần nó nhất. Muốn nới thì sửa con số và commit, để `git blame`
còn nhìn thấy.

Mỗi ngưỡng trong YAML mang một trường `why`. Nó vào thẳng báo cáo HTML, nên
người đọc kết quả FAIL thấy ngay lý lẽ chứ không phải một con số trần trụi.
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from rag_core.bundle import RagBundle
from rag_core.llm import DEEPSEEK_ALIASES

__all__ = [
    "DEFAULT_THRESHOLDS",
    "GateStatus",
    "GateVerdict",
    "Rule",
    "RuleOutcome",
    "Thresholds",
    "evaluate_gate",
    "judge_identity",
    "load_thresholds",
    "main",
    "render_html",
]

DEFAULT_THRESHOLDS = Path("configs/eval/gate.yaml")

_PRESET = "@preset/"


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCOMPARABLE = "INCOMPARABLE"
    """Điều kiện tiên quyết không thoả — xem điểm 1 ở docstring module."""

    @property
    def exit_code(self) -> int:
        """PASS = 0. FAIL = 1. INCOMPARABLE = **2**, không phải 1.

        Hai chế độ hỏng khác nhau đáng hai exit code khác nhau: CI phân biệt
        được "chất lượng tụt" với "phép đo không đặt cạnh nhau được" mà không
        phải parse stdout.
        """
        return {GateStatus.PASS: 0, GateStatus.FAIL: 1, GateStatus.INCOMPARABLE: 2}[self]


class RuleOutcome(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    """Luật không áp dụng được — ví dụ luật hồi quy khi không có champion.

    Bỏ hẳn khỏi báo cáo thì người đọc tưởng đã kiểm. In ra SKIP kèm lý do thì
    họ thấy đúng cái chưa được kiểm.
    """


@dataclass(frozen=True)
class Rule:
    """Một phán quyết đơn, đủ để in ra một dòng người đọc hiểu được."""

    group: str
    name: str
    outcome: RuleOutcome
    detail: str
    why: str = ""
    value: float | None = None
    threshold: float | None = None
    champion: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "name": self.name,
            "outcome": self.outcome.value,
            "detail": self.detail,
            "why": self.why,
            "value": self.value,
            "threshold": self.threshold,
            "champion": self.champion,
        }


@dataclass
class GateVerdict:
    status: GateStatus
    candidate: str
    champion: str | None
    rules: list[Rule] = field(default_factory=list)

    @property
    def failures(self) -> list[Rule]:
        return [rule for rule in self.rules if rule.outcome is RuleOutcome.FAIL]

    def counts(self) -> dict[str, int]:
        out = {outcome.value: 0 for outcome in RuleOutcome}
        for rule in self.rules:
            out[rule.outcome.value] += 1
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "exit_code": self.status.exit_code,
            "candidate": self.candidate,
            "champion": self.champion,
            "counts": self.counts(),
            "rules": [rule.as_dict() for rule in self.rules],
        }


# ------------------------------------------------------------------- ngưỡng


@dataclass(frozen=True)
class Thresholds:
    require_same: tuple[str, ...]
    reject_alias_identity: bool
    max_unjudged_rate: float
    require_unjudged_rate: bool
    min_judge_kappa: float | None
    absolute: Mapping[str, Mapping[str, Any]]
    max_drop: Mapping[str, float]
    require_metrics_present: bool

    @staticmethod
    def from_mapping(raw: Mapping[str, Any]) -> Thresholds:
        comparability = raw.get("comparability") or {}
        validity = raw.get("validity") or {}
        regression = raw.get("regression") or {}
        return Thresholds(
            require_same=tuple(comparability.get("require_same") or ()),
            reject_alias_identity=bool(comparability.get("reject_alias_identity", True)),
            max_unjudged_rate=float(validity.get("max_unjudged_rate", 1.0)),
            require_unjudged_rate=bool(validity.get("require_unjudged_rate", True)),
            min_judge_kappa=(
                None
                if validity.get("min_judge_kappa") is None
                else float(validity["min_judge_kappa"])
            ),
            absolute=dict(raw.get("absolute") or {}),
            max_drop=dict(regression.get("max_drop") or {}),
            require_metrics_present=bool(regression.get("require_metrics_present", True)),
        )


def load_thresholds(path: str | Path = DEFAULT_THRESHOLDS) -> Thresholds:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: ngưỡng phải là một ánh xạ")
    return Thresholds.from_mapping(raw)


# ----------------------------------------------------------------- danh tính


def judge_identity(bundle: RagBundle) -> str:
    """Chuỗi định danh giám khảo: `(model, rubrics, reasoning)` — `TD-66`.

    Không dùng riêng `model`: `W5-01` đo được cùng model + rubric v1→v2 đưa
    `uncited_grounding` từ `0,427` lên `0,856`. Rubric là một phần của thước đo
    y hệt như model, và `reasoning` thì `W5-04` đo được là phần quyết định nhất.
    """
    judge = bundle.eval.judge
    if judge is None:
        return "<không có judge>"
    rubrics = ",".join(sorted(judge.rubrics)) or "<không khai rubric>"
    reasoning = "?" if judge.reasoning is None else str(judge.reasoning).lower()
    return f"{judge.model}|{rubrics}|reasoning={reasoning}"


def _alias_problem(identity: str) -> str:
    """Chuỗi danh tính này có phải một con trỏ phía server không.

    So trên **phần slug trước `@`**: `deepseek-chat@2026-09` là bí danh mang
    thêm một cái nhãn ngày tháng, và cái nhãn ấy không làm nó ngừng là con trỏ.
    """
    if _PRESET in identity:
        return f"{identity!r} chứa OpenRouter preset — quy tắc cứng #1"
    slug = identity.split("@", 1)[0].strip()
    if slug in DEEPSEEK_ALIASES:
        return (
            f"{identity!r} là bí danh: {slug!r} hiện được phục vụ bởi "
            f"{DEEPSEEK_ALIASES[slug]!r} (đo ở `W5-03`) và sẽ đổi khi nhà cung cấp "
            "ra bản mới. Hai bundle cùng ghi chuỗi này có thể đã đo bằng hai model "
            "khác nhau."
        )
    return ""


# ---------------------------------------------------------------------- luật


def _comparability(
    candidate: RagBundle, champion: RagBundle | None, limits: Thresholds
) -> list[Rule]:
    rules: list[Rule] = []
    readers = {
        "golden_set": lambda b: b.eval.golden_set,
        "evaluated_with_generator": lambda b: b.eval.evaluated_with_generator,
        "judge_identity": judge_identity,
    }

    if limits.reject_alias_identity:
        problem = _alias_problem(candidate.eval.evaluated_with_generator)
        if champion is not None and not problem:
            problem = _alias_problem(champion.eval.evaluated_with_generator)
        rules.append(
            Rule(
                group="comparability",
                name="generator không phải bí danh",
                outcome=RuleOutcome.FAIL if problem else RuleOutcome.PASS,
                detail=problem or f"{candidate.eval.evaluated_with_generator!r} là slug tường minh",
                why=(
                    "Trường này tồn tại để bảo đảm so like-for-like. Một con trỏ "
                    "phía server làm hai chuỗi bằng nhau mà hai phép đo thì không."
                ),
            )
        )

    for key in limits.require_same:
        reader = readers.get(key)
        if reader is None:
            rules.append(
                Rule("comparability", key, RuleOutcome.FAIL, f"luật lạ trong ngưỡng: {key!r}")
            )
            continue
        mine = reader(candidate)
        if champion is None:
            rules.append(
                Rule("comparability", key, RuleOutcome.SKIP, "không có champion để so", value=None)
            )
            continue
        theirs = reader(champion)
        same = mine == theirs
        rules.append(
            Rule(
                group="comparability",
                name=key,
                outcome=RuleOutcome.PASS if same else RuleOutcome.FAIL,
                detail=(f"{mine!r}" if same else f"ứng viên {mine!r} ≠ champion {theirs!r}"),
                why=(
                    "Chênh lệch giữa hai bundle chỉ quy được cho hệ thống khi mọi "
                    "thứ khác giữ nguyên."
                ),
            )
        )
    return rules


def _validity(candidate: RagBundle, limits: Thresholds) -> list[Rule]:
    rules: list[Rule] = []
    report = candidate.eval
    if not report.generation_metrics:
        return [
            Rule(
                "validity",
                "tỉ lệ chưa chấm được",
                RuleOutcome.SKIP,
                "bundle không mang metric tầng sinh",
            )
        ]

    if limits.require_unjudged_rate and not report.unjudged_rate:
        rules.append(
            Rule(
                group="validity",
                name="khai tỉ lệ chưa chấm được",
                outcome=RuleOutcome.FAIL,
                detail=(
                    f"{len(report.generation_metrics)} metric tầng sinh nhưng `unjudged_rate` rỗng"
                ),
                why=(
                    "Không khai thì không kiểm được. `W5-04` đo được chế độ hỏng của "
                    "judge lệch về phía điểm CAO (faithfulness 1,0000 khi mất 64% phán "
                    "quyết), nên 'không biết' phải đỏ."
                ),
            )
        )
    for name, rate in sorted(report.unjudged_rate.items()):
        ok = rate <= limits.max_unjudged_rate
        rules.append(
            Rule(
                group="validity",
                name=f"chưa chấm được · {name}",
                outcome=RuleOutcome.PASS if ok else RuleOutcome.FAIL,
                detail=(
                    f"{rate:.1%} phán quyết không đọc được (trần {limits.max_unjudged_rate:.0%})"
                ),
                why=(
                    "Một metric tính trên phần còn lại chỉ đúng nếu việc mất phán "
                    "quyết độc lập với nhãn — `W5-04` đo được nó không độc lập."
                ),
                value=rate,
                threshold=limits.max_unjudged_rate,
            )
        )

    if limits.min_judge_kappa is not None:
        judge = report.judge
        kappa = judge.kappa_vs_human if judge else None
        if kappa is None:
            rules.append(
                Rule(
                    group="validity",
                    name="judge đã hiệu chỉnh",
                    outcome=RuleOutcome.FAIL,
                    detail="`judge.kappa_vs_human` chưa khai",
                    why="Một judge chưa đối chiếu với người là một thước đo chưa hiệu chuẩn.",
                    threshold=limits.min_judge_kappa,
                )
            )
        else:
            rules.append(
                Rule(
                    group="validity",
                    name="judge đã hiệu chỉnh",
                    outcome=(
                        RuleOutcome.PASS if kappa >= limits.min_judge_kappa else RuleOutcome.FAIL
                    ),
                    detail=f"κ vs người = {kappa:.3f} (tối thiểu {limits.min_judge_kappa})",
                    why="`W5-04` đo 0,737 trên 50 mệnh đề gán nhãn tay.",
                    value=kappa,
                    threshold=limits.min_judge_kappa,
                )
            )
    return rules


def _metrics_of(bundle: RagBundle) -> dict[str, float]:
    """Gộp hai bảng metric. Trùng khoá là **lỗi**, không phải chuyện bên nào thắng."""
    merged = dict(bundle.eval.retrieval_metrics)
    clash = set(merged) & set(bundle.eval.generation_metrics)
    if clash:
        raise ValueError(
            f"{bundle.bundle_version}: metric trùng tên ở hai bảng {sorted(clash)} — "
            "không quyết được con số nào là con số được gate."
        )
    merged.update(bundle.eval.generation_metrics)
    # ⚠️ Hai con số độ trễ, và nhầm chúng là một lỗi `W5-05` đã mắc rồi sửa:
    # `p95_latency_ms` là truy hồi thuần (`759 ms`), `p95_end_to_end_ms` là cái
    # người dùng chịu (`4706 ms`). Đem cái đầu so với ngân sách end-to-end thì
    # gate cho qua một hệ thống vượt ngân sách 34%.
    if bundle.eval.p95_latency_ms is not None:
        merged["p95_latency_ms"] = bundle.eval.p95_latency_ms
    if bundle.eval.p95_end_to_end_ms is not None:
        merged["p95_end_to_end_ms"] = bundle.eval.p95_end_to_end_ms
    return merged


def _absolute(values: Mapping[str, float], limits: Thresholds) -> list[Rule]:
    rules: list[Rule] = []
    for name, spec in limits.absolute.items():
        why = str(spec.get("why", ""))
        if name not in values:
            rules.append(
                Rule(
                    group="absolute",
                    name=name,
                    outcome=RuleOutcome.FAIL,
                    detail="bundle không mang metric này",
                    why=why or "Ngưỡng đã khai nhưng không có số để so — không phải PASS.",
                )
            )
            continue
        value = values[name]
        if "min" in spec:
            floor = float(spec["min"])
            rules.append(
                Rule(
                    group="absolute",
                    name=name,
                    outcome=RuleOutcome.PASS if value >= floor else RuleOutcome.FAIL,
                    detail=f"{value:.4f} {'≥' if value >= floor else '<'} {floor}",
                    why=why,
                    value=value,
                    threshold=floor,
                )
            )
        if "max" in spec:
            ceiling = float(spec["max"])
            rules.append(
                Rule(
                    group="absolute",
                    name=name,
                    outcome=RuleOutcome.PASS if value <= ceiling else RuleOutcome.FAIL,
                    detail=f"{value:.4f} {'≤' if value <= ceiling else '>'} {ceiling}",
                    why=why,
                    value=value,
                    threshold=ceiling,
                )
            )
    return rules


def _regression(
    values: Mapping[str, float], champion: RagBundle | None, limits: Thresholds
) -> list[Rule]:
    if champion is None:
        return [Rule("regression", "so với champion", RuleOutcome.SKIP, "không có champion để so")]
    base = _metrics_of(champion)
    rules: list[Rule] = []
    for name, allowed in sorted(limits.max_drop.items()):
        if name not in base:
            rules.append(
                Rule("regression", name, RuleOutcome.SKIP, "champion không mang metric này")
            )
            continue
        if name not in values:
            outcome = RuleOutcome.FAIL if limits.require_metrics_present else RuleOutcome.SKIP
            rules.append(
                Rule(
                    group="regression",
                    name=name,
                    outcome=outcome,
                    detail="champion có metric này, ứng viên thiếu",
                    why=(
                        "Metric biến mất giữa hai lần chạy là cách một lần eval hỏng "
                        "đi qua gate mà trông như không có gì xảy ra."
                    ),
                    champion=base[name],
                )
            )
            continue
        drop = base[name] - values[name]
        rules.append(
            Rule(
                group="regression",
                name=name,
                outcome=RuleOutcome.PASS if drop <= allowed else RuleOutcome.FAIL,
                detail=(
                    f"{values[name]:.4f} vs champion {base[name]:.4f} "
                    f"({'−' if drop > 0 else '+'}{abs(drop):.4f}, cho phép tụt {allowed})"
                ),
                why="Bắt hướng đi, không bắt mức. Ngưỡng cỡ sai số bootstrap của `W2-08`.",
                value=values[name],
                threshold=allowed,
                champion=base[name],
            )
        )
    return rules


def evaluate_gate(
    candidate: RagBundle,
    champion: RagBundle | None,
    limits: Thresholds,
) -> GateVerdict:
    """Chạy cả bốn nhóm luật và gộp thành một phán quyết.

    Nhóm `comparability` hỏng thì **vẫn chạy nốt** các nhóm sau. Lý do: người
    đọc báo cáo cần thấy cả hai thứ cùng lúc — "không so được" **và** "ngoài ra
    thì các ngưỡng tuyệt đối thế nào" — chứ không phải sửa một lỗi rồi mới biết
    còn lỗi gì. Chỉ `status` là bị `INCOMPARABLE` chiếm quyền.
    """
    values = _metrics_of(candidate)
    rules = [
        *_comparability(candidate, champion, limits),
        *_validity(candidate, limits),
        *_absolute(values, limits),
        *_regression(values, champion, limits),
    ]
    blocked = any(
        rule.group == "comparability" and rule.outcome is RuleOutcome.FAIL for rule in rules
    )
    failed = any(rule.outcome is RuleOutcome.FAIL for rule in rules)
    if blocked:
        status = GateStatus.INCOMPARABLE
    elif failed:
        status = GateStatus.FAIL
    else:
        status = GateStatus.PASS
    return GateVerdict(
        status=status,
        candidate=candidate.bundle_version,
        champion=champion.bundle_version if champion else None,
        rules=rules,
    )


# ---------------------------------------------------------------------- HTML

_BADGE = {
    GateStatus.PASS: ("#0a7c42", "#e6f6ed"),
    GateStatus.FAIL: ("#b42318", "#fdecea"),
    GateStatus.INCOMPARABLE: ("#8a5a00", "#fdf3e2"),
}
_MARK = {RuleOutcome.PASS: "✅", RuleOutcome.FAIL: "❌", RuleOutcome.SKIP: "⏭"}
_GROUP_TITLE = {
    "comparability": "1 · So được với nhau không",
    "validity": "2 · Phép đo có dùng được không",
    "absolute": "3 · Ngưỡng tuyệt đối",
    "regression": "4 · Hồi quy so với champion",
}


def _cell(value: float | None) -> str:
    if value is None:
        return "—"
    if math.isfinite(value) and abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.4f}"


def render_html(verdict: GateVerdict, *, thresholds_path: str = "") -> str:
    """Một file HTML tự chứa. Không CDN, không font ngoài — báo cáo gate phải mở
    được trên một máy không có mạng, ba tháng sau, từ một artifact CI."""
    colour, wash = _BADGE[verdict.status]
    counts = verdict.counts()
    rows: list[str] = []
    for group in ("comparability", "validity", "absolute", "regression"):
        group_rules = [rule for rule in verdict.rules if rule.group == group]
        if not group_rules:
            continue
        rows.append(f"<h2>{html.escape(_GROUP_TITLE[group])}</h2>")
        rows.append(
            "<table><thead><tr><th></th><th>Luật</th><th>Chi tiết</th>"
            "<th class='num'>Giá trị</th><th class='num'>Ngưỡng</th>"
            "<th class='num'>Champion</th></tr></thead><tbody>"
        )
        for rule in group_rules:
            why = f"<div class='why'>{html.escape(rule.why)}</div>" if rule.why else ""
            rows.append(
                f"<tr class='{rule.outcome.value.lower()}'>"
                f"<td class='mark'>{_MARK[rule.outcome]}</td>"
                f"<td><code>{html.escape(rule.name)}</code></td>"
                f"<td>{html.escape(rule.detail)}{why}</td>"
                f"<td class='num'>{_cell(rule.value)}</td>"
                f"<td class='num'>{_cell(rule.threshold)}</td>"
                f"<td class='num'>{_cell(rule.champion)}</td></tr>"
            )
        rows.append("</tbody></table>")

    champion = html.escape(verdict.champion or "— (không có champion để so)")
    note = ""
    if verdict.status is GateStatus.INCOMPARABLE:
        note = (
            "<p class='note'><strong>INCOMPARABLE không phải một kiểu FAIL.</strong> "
            "Nó nói rằng hai con số này không đặt cạnh nhau được — việc phải làm là "
            "sửa <em>phép đo</em>, không phải sửa hệ thống. Các nhóm luật còn lại vẫn "
            "chạy hết và in ở dưới, để không phải sửa một lỗi rồi mới biết còn lỗi gì."
            "</p>"
        )
    return f"""<meta charset="utf-8">
<title>Gate · bundle {html.escape(verdict.candidate)} · {verdict.status.value}</title>
<style>
 body {{ font: 15px/1.55 ui-sans-serif, system-ui, "Segoe UI", sans-serif;
        max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.4rem; margin-bottom: .2rem; }}
 h2 {{ font-size: 1.05rem; margin: 2rem 0 .5rem; color: #444; }}
 .badge {{ display:inline-block; padding:.35rem .9rem; border-radius:999px;
           font-weight:700; letter-spacing:.03em; color:{colour}; background:{wash};
           border:1px solid {colour}33; }}
 .meta {{ color:#555; margin:.6rem 0 0; }}
 .meta code {{ background:#f4f4f5; padding:.1rem .35rem; border-radius:4px; }}
 .note {{ background:#fdf3e2; border-left:4px solid #8a5a00; padding:.7rem .9rem;
          margin:1.2rem 0; }}
 table {{ border-collapse: collapse; width: 100%; }}
 th, td {{ border-bottom:1px solid #e6e6e8; padding:.5rem .6rem; text-align:left;
           vertical-align: top; }}
 th {{ font-size:.8rem; text-transform:uppercase; letter-spacing:.04em; color:#666; }}
 td.num, th.num {{ text-align:right; font-variant-numeric: tabular-nums;
                   white-space:nowrap; }}
 td.mark {{ width:1.6rem; }}
 tr.fail {{ background:#fdecea55; }}
 tr.skip {{ color:#777; }}
 .why {{ font-size:.85rem; color:#666; margin-top:.25rem; }}
 code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.9em; }}
 footer {{ margin-top:2.5rem; font-size:.85rem; color:#777; }}
</style>
<h1>Gate · bundle <code>{html.escape(verdict.candidate)}</code></h1>
<p><span class="badge">{verdict.status.value}</span>
   <span class="meta">exit code <code>{verdict.status.exit_code}</code></span></p>
<p class="meta">champion: <code>{champion}</code> ·
   {counts["PASS"]} qua · {counts["FAIL"]} hỏng · {counts["SKIP"]} bỏ qua ·
   ngưỡng: <code>{html.escape(thresholds_path)}</code></p>
{note}
{"".join(rows)}
<footer>Sinh bởi <code>pipeline.eval.gate</code> (`W5-05`). Mỗi ngưỡng mang trường
<code>why</code> lấy từ chính file ngưỡng — một con số không có lý lẽ là một con số
sẽ bị hạ xuống vào lúc cần nó nhất.</footer>
"""


# ----------------------------------------------------------------------- CLI


def _resolve_bundle(spec: str, root: Path) -> RagBundle:
    from rag_core.bundle import bundle_dir_name, load_bundle

    path = Path(spec)
    if path.is_file():
        return load_bundle(path)
    candidate = root / bundle_dir_name(spec) / "manifest.json"
    if not candidate.is_file():
        raise SystemExit(f"không tìm thấy bundle {spec!r} (đã thử {path} và {candidate})")
    return load_bundle(candidate)


def _pick_champion(candidate: RagBundle, root: Path) -> RagBundle | None:
    """Champion = bản có version cao nhất **thấp hơn** ứng viên.

    Không lấy "bản mới nhất": nếu ứng viên chính là bản mới nhất thì nó sẽ tự so
    với chính mình và mọi luật hồi quy đều qua — một gate luôn xanh.
    """
    from rag_core.bundle import list_bundles

    earlier = [b for b in list_bundles(root) if b.version_key < candidate.version_key]
    return max(earlier, key=lambda b: b.version_key) if earlier else None


def _text_summary(verdict: GateVerdict) -> str:
    lines = [f"GATE {verdict.status.value} · bundle {verdict.candidate}"]
    lines.append(f"  champion: {verdict.champion or '—'}")
    for rule in verdict.rules:
        lines.append(f"  {_MARK[rule.outcome]} [{rule.group}] {rule.name}: {rule.detail}")
    if verdict.status is GateStatus.INCOMPARABLE:
        lines.append(
            "  ⚠️ INCOMPARABLE — sửa PHÉP ĐO, không phải sửa hệ thống. Xem nhóm comparability."
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import logging

    parser = argparse.ArgumentParser(description="W5-05 — gate phát hành cho một bundle")
    parser.add_argument("--bundle", required=True, help="version (`0.2.1`) hoặc đường dẫn manifest")
    parser.add_argument("--champion", default=None, help="version/đường dẫn; mặc định tự chọn")
    parser.add_argument(
        "--no-champion",
        action="store_true",
        help="Chỉ chạy ngưỡng tuyệt đối + tính hợp lệ. Luật hồi quy in SKIP, không im lặng.",
    )
    parser.add_argument("--root", type=Path, default=Path("bundles"))
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--html", type=Path, default=None)
    parser.add_argument("--json", dest="json_out", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    limits = load_thresholds(args.thresholds)
    candidate = _resolve_bundle(args.bundle, args.root)
    if args.no_champion:
        champion = None
    elif args.champion:
        champion = _resolve_bundle(args.champion, args.root)
    else:
        champion = _pick_champion(candidate, args.root)

    verdict = evaluate_gate(candidate, champion, limits)

    if args.html:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(
            render_html(verdict, thresholds_path=str(args.thresholds)), encoding="utf-8"
        )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(verdict.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(_text_summary(verdict))
    if args.html:
        print(f"đã ghi {args.html}")
    if args.json_out:
        print(f"đã ghi {args.json_out}")
    return verdict.status.exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
