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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "BINARY_METRICS",
    "GROUP_DIMENSIONS",
    "ComparisonRow",
    "RunScores",
    "bootstrap_intervals",
    "compare_by_group",
    "compare_runs",
    "format_grouped_table",
    "format_table",
    "load_per_query",
    "main",
    "mcnemar_exact",
    "min_achievable_p",
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
DEFAULT_ALPHA = 0.05

#: Chiều chia nhóm hợp lệ — đúng hai trường mà `QueryScore` mang theo từng câu.
#: Không nhận chiều tuỳ ý: chia theo một trường không có trong file sẽ cho **một
#: nhóm rỗng** thay vì một lỗi, và một bảng rỗng trông y như "không có khác biệt".
GROUP_DIMENSIONS = ("category", "lang")


@dataclass(frozen=True)
class RunScores:
    """Điểm từng câu của một lần chạy, đọc từ `{run}-per-query.jsonl`."""

    name: str
    scores: dict[str, dict[str, float]]
    n_relevant: dict[str, int]
    relevant_digest: dict[str, str] = field(default_factory=dict)
    """Băm tập nhãn từng câu. Rỗng = lần chạy có trước `W2-03`, chưa ghi băm."""

    category: dict[str, str] = field(default_factory=dict)
    lang: dict[str, str] = field(default_factory=dict)
    """Hai chiều chia nhóm, đọc từ `*-per-query.jsonl`. Xem `GROUP_DIMENSIONS`."""

    @property
    def query_ids(self) -> set[str]:
        return set(self.scores)

    def groups(self, dimension: str) -> dict[str, list[str]]:
        """`{giá trị: [query_id]}` theo một chiều, giữ thứ tự nhóm to → nhỏ.

        To → nhỏ có chủ đích: nhóm đầu bảng là nhóm có lực kiểm định lớn nhất, nên
        người đọc gặp kết luận đáng tin trước và gặp `table_lookup` (4 câu) cuối.
        """
        if dimension not in GROUP_DIMENSIONS:
            raise ValueError(
                f"Chiều chia nhóm không hợp lệ: {dimension!r}. Hợp lệ: {list(GROUP_DIMENSIONS)}."
            )
        source: dict[str, str] = getattr(self, dimension)
        buckets: dict[str, list[str]] = {}
        for qid in sorted(self.scores):
            buckets.setdefault(source.get(qid, ""), []).append(qid)
        return dict(sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])))

    def subset(self, query_ids: Sequence[str]) -> RunScores:
        """Thu về một tập câu, giữ nguyên `name`.

        Thu nhỏ **dữ liệu** thay vì thêm tham số lọc vào `compare_runs`: mọi hàng
        rào của `compare_runs` (tập câu chung, `n_relevant` trung bình, băm nhãn)
        phải áp trên **đúng** tập đang so, và cách chắc chắn nhất để không quên một
        hàng rào nào là không cho `compare_runs` biết là có chuyện lọc.
        """
        keep = [qid for qid in query_ids if qid in self.scores]
        return RunScores(
            name=self.name,
            scores={qid: self.scores[qid] for qid in keep},
            n_relevant={qid: self.n_relevant[qid] for qid in keep},
            relevant_digest={qid: self.relevant_digest.get(qid, "") for qid in keep},
            category={qid: self.category.get(qid, "") for qid in keep},
            lang={qid: self.lang.get(qid, "") for qid in keep},
        )

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

    n_queries: int = 0
    """Số câu thật sự vào phép so. Bắt buộc hiện ra ở bảng chia nhóm.

    43 câu `cross_lingual` có ngưỡng phân giải thô hơn 209 câu **rất** nhiều, và
    `table_lookup` chỉ có **4** câu. Một bảng chia nhóm không in `n` là một bảng
    mời người đọc so những con số không cùng độ tin cậy.
    """

    alpha: float = DEFAULT_ALPHA
    """Ngưỡng đã dùng cho **hàng này**. Khác `DEFAULT_ALPHA` khi có hiệu chỉnh."""

    family_size: int = 1
    """Số phép kiểm trong họ. `> 1` nghĩa là `alpha` đã bị chia — xem `compare_by_group`."""

    @property
    def n_discordant(self) -> int:
        """Số câu **đổi chiều** — thứ duy nhất McNemar dùng."""
        return self.n_baseline_only + self.n_candidate_only

    @property
    def ci_touches_zero(self) -> bool:
        """CI có **một biên đúng bằng 0** — biên giới, không phải kết luận âm.

        Đo được ở `W2-08-prep`, và nó không phải chuyện lẻ: trong bảng chia nhóm
        `bgem3 → bgem3-rrf-k1-c20`, **4/4** dòng `recall@5` có một biên ghim đúng
        `0,0000`, và cả bốn vì thế bị dán "trong ngưỡng nhiễu".

        Nguyên nhân là **rời rạc, không phải ngẫu nhiên**: phần lớn câu có hiệu
        bằng 0, nên phân bố trung bình bootstrap nằm trên một **lưới** bước `1/n`
        và một phân vị cực đoan rơi đúng lên điểm 0 của lưới đó. Đo trực tiếp: đổi
        seed làm biên dịch **đúng một bước lưới** (0,0233 = 1/43) chứ không dịch
        trơn, và tăng từ 10.000 lên 50.000 iterations **không thay đổi gì** — nên
        đây không phải chỗ chữa bằng cách lấy nhiều mẫu hơn. (Ghi lại để lần sau
        không ai "sửa" nó bằng cách tăng iterations.)

        Kết luận đúng của một hàng như thế **không** phải "không có khác biệt" mà
        là "không kết luận được": việc khoảng chứa 0 phụ thuộc vào đúng một bước
        phân giải của dữ liệu.
        """
        if not self.comparable or self.ci_low is None or self.ci_high is None:
            return False
        return self.ci_low == 0.0 or self.ci_high == 0.0

    @property
    def underpowered(self) -> bool:
        """`True` khi phép kiểm **không thể** đạt ý nghĩa dù kết quả thế nào.

        Đây là phân biệt quan trọng nhất mà bảng chia nhóm phải nói ra, và nó là
        một con số **tính được**, không phải một phỏng đoán về cỡ mẫu:

        McNemar exact chỉ dùng `n` câu đổi chiều, và `p` nhỏ nhất nó trả về được là
        `2 / 2**n` (mọi câu đổi chiều cùng một hướng). Với `n = 4` thì `p_min =
        0,125`, tức **không kết quả nào** cho ra ý nghĩa ở mức 0,05. `table_lookup`
        có 4 câu; nó không bao giờ đạt được ý nghĩa trên bất kỳ metric nhị phân nào.

        Không có cờ này thì "trong ngưỡng nhiễu" của `table_lookup` đọc như *bằng
        chứng không có hiệu ứng*, trong khi nó là *bằng chứng không có lực kiểm
        định* — hai chuyện trái ngược nhau dẫn tới cùng một dòng chữ.
        """
        if not self.comparable or self.p_value is None:
            return False
        return min_achievable_p(self.n_discordant) >= self.alpha

    @property
    def verdict(self) -> str:
        if not self.comparable:
            return "KHÔNG SO ĐƯỢC"
        if self.underpowered:
            return "KHÔNG ĐỦ LỰC"
        if self.p_value is not None:
            return "khác biệt thật" if self.p_value < self.alpha else "trong ngưỡng nhiễu"
        if self.ci_low is not None and self.ci_high is not None:
            if (self.ci_low > 0 and self.ci_high > 0) or (self.ci_low < 0 and self.ci_high < 0):
                return "khác biệt thật"
            # Biên đúng 0 = biên giới của lưới rời rạc, không phải bằng chứng âm.
            # Xem `ci_touches_zero`. Gộp nó vào "trong ngưỡng nhiễu" là chỗ mà một
            # giới hạn phân giải bị đọc thành một kết luận.
            return "KHÔNG KẾT LUẬN" if self.ci_touches_zero else "trong ngưỡng nhiễu"
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
    categories: dict[str, str] = {}
    langs: dict[str, str] = {}
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
        categories[qid] = str(row.get("category", ""))
        langs[qid] = str(row.get("lang", ""))
    return RunScores(
        name=name or source.stem,
        scores=scores,
        n_relevant=n_relevant,
        relevant_digest=digests,
        category=categories,
        lang=langs,
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


def min_achievable_p(n_discordant: int) -> float:
    """`p` nhỏ nhất McNemar exact có thể trả về với `n` câu đổi chiều.

    Xảy ra khi **mọi** câu đổi chiều nghiêng cùng một hướng: `2 * (1/2**n)`. Dùng
    để phân biệt "không tìm thấy hiệu ứng" với "không thể tìm thấy hiệu ứng" —
    xem `ComparisonRow.underpowered`.

    Bảng để khỏi phải tính nhẩm: n=0 → 1,0 · n=1 → 1,0 · n=2 → 0,5 · n=3 → 0,25 ·
    n=4 → 0,125 · n=5 → 0,0625 · **n=6 → 0,03125** (chỗ đầu tiên qua được 0,05).
    Nên một nhóm dưới **6 câu** không bao giờ đạt ý nghĩa trên metric nhị phân, và
    dưới 11 câu thì không bao giờ đạt được ở mức đã hiệu chỉnh Bonferroni cho 90
    phép kiểm (0,05/90 = 0,00056; cần n ≥ 12).
    """
    if n_discordant <= 0:
        return 1.0
    # `float(...)`: typeshed khai `int.__pow__` trả `Any` vì số mũ có thể âm, nên
    # `mypy --strict` từ chối. Ở đây `n_discordant >= 1` nên nó luôn là int dương.
    return min(1.0, 2.0 / float(2**n_discordant))


def bootstrap_intervals(
    diffs: Sequence[float],
    alphas: Sequence[float],
    *,
    iterations: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> dict[float, tuple[float, float]]:
    """Nhiều khoảng tin cậy từ **một** lần lấy mẫu lại.

    Bảng chia nhóm cần cả khoảng 95% (để đọc) và khoảng đã hiệu chỉnh Bonferroni
    (để kết luận). Gọi `paired_bootstrap` hai lần sẽ lấy mẫu lại hai lần — gấp đôi
    chi phí cho **cùng một** phân bố, và với hai `seed` giống nhau thì hai kết quả
    vẫn đến từ cùng dãy số ngẫu nhiên nên chẳng độc lập gì. Sắp một lần, lấy phân
    vị nhiều lần.

    Seed cố định để hai lần chạy công cụ cho cùng con số — một khoảng tin cậy nhảy
    nhót mỗi lần gọi thì không dùng để quyết định được.
    """
    if not diffs:
        return {alpha: (0.0, 0.0) for alpha in alphas}
    rng = random.Random(seed)
    n = len(diffs)
    means: list[float] = []
    for _ in range(iterations):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    out: dict[float, tuple[float, float]] = {}
    for alpha in alphas:
        lo = means[int(alpha / 2 * iterations)]
        hi = means[min(iterations - 1, int((1 - alpha / 2) * iterations))]
        out[alpha] = (lo, hi)
    return out


def paired_bootstrap(
    diffs: Sequence[float],
    *,
    iterations: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float]:
    """Khoảng tin cậy phần trăm cho trung bình hiệu, lấy mẫu lại theo truy vấn.

    Giữ nguyên chữ ký từ `W2-01` vì mọi con số đã công bố đi qua nó. Phần lấy mẫu
    chuyển sang `bootstrap_intervals`; hàm này là một lớp mỏng, nên hai đường
    **không thể** cho hai kết quả khác nhau (có test ghim).
    """
    return bootstrap_intervals(diffs, (alpha,), iterations=iterations, seed=seed)[alpha]


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
    family_size: int = 1,
) -> list[ComparisonRow]:
    """So hai lần chạy trên **toàn bộ** tập câu chung của chúng.

    `family_size > 1` chia `alpha` theo Bonferroni. Mặc định là 1, tức **hành vi
    của mọi lần chạy từ `W2-01` đến `W2-07` không đổi một chữ số nào** — đó là
    điều kiện, không phải sự tình cờ: đổi ngưỡng ở đây sẽ lặng lẽ viết lại kết
    luận của những bảng đã công bố. Chỗ dùng hiệu chỉnh là `compare_by_group`,
    nơi việc hỏi nhiều câu hỏi cùng lúc **là mục đích**.
    """
    alpha = DEFAULT_ALPHA / family_size
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
                n_queries=len(shared),
                alpha=alpha,
                family_size=family_size,
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
                    n_queries=len(pairs),
                    alpha=alpha,
                    family_size=family_size,
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
                    n_queries=len(pairs),
                    alpha=alpha,
                    family_size=family_size,
                )
            )
            continue

        diffs = [b - a for a, b in pairs]
        # Một lần lấy mẫu lại, hai phân vị: khoảng 95% để **đọc**, khoảng đã hiệu
        # chỉnh để **kết luận**. Khi `family_size == 1` thì hai cái là một.
        intervals = bootstrap_intervals(
            diffs, (DEFAULT_ALPHA, alpha), iterations=iterations, seed=seed
        )
        lo, hi = intervals[alpha]
        note = ""
        if family_size > 1:
            raw_lo, raw_hi = intervals[DEFAULT_ALPHA]
            note = f"CI95 thô [{raw_lo:+.4f}, {raw_hi:+.4f}]"
        rows.append(
            ComparisonRow(
                metric=metric,
                baseline=mean_b,
                candidate=mean_c,
                delta=mean_c - mean_b,
                test=f"bootstrap cặp ({iterations})",
                ci_low=lo,
                ci_high=hi,
                n_queries=len(pairs),
                alpha=alpha,
                family_size=family_size,
                note=note,
            )
        )
    return rows


def compare_by_group(
    baseline: RunScores,
    candidate: RunScores,
    dimension: str,
    *,
    metrics: Sequence[str] = (),
    iterations: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    correct: bool = True,
) -> dict[str, list[ComparisonRow]]:
    """So theo từng nhóm của một chiều, **có hiệu chỉnh đa so sánh**.

    ## Vì sao hàm này phải hiệu chỉnh mà `compare_runs` thì không

    Hai hàm trả lời hai loại câu hỏi khác nhau, và đó là toàn bộ lý do chúng khác
    ngưỡng:

    * `compare_runs` kiểm **một giả thuyết đã nêu trước**: "cấu hình B khác cấu
      hình A". Bảng metric là một panel cố định, không phải một cuộc tìm kiếm.
    * `compare_by_group` **là** một cuộc tìm kiếm. DoD `W2-09` đòi "category nào
      cải thiện nhiều nhất", tức chọn ra cái lớn nhất trong 6 nhóm × 15 metric =
      **90 phép kiểm**. Ở mức 0,05 và không hiệu chỉnh thì kỳ vọng **4,5 kết quả
      "có ý nghĩa" thuần do ngẫu nhiên** — nên câu trả lời cho "category nào cải
      thiện nhiều nhất" gần như *bảo đảm* tìm ra một cái, kể cả khi không có gì.

    Bonferroni (`alpha / m`), không Holm, dù Holm mạnh hơn một chút. Lý do: Holm
    dùng ngưỡng **khác nhau cho từng hạng**, và một khoảng tin cậy bootstrap không
    biểu diễn được điều đó bằng một khoảng duy nhất — bảng sẽ có `p` hiệu chỉnh
    theo Holm cạnh CI hiệu chỉnh theo Bonferroni, hai luật trong một bảng. Nhất
    quán trên cả bảng đáng hơn một chút lực kiểm định.

    `m` đếm số phép kiểm **thử**, tức `số nhóm × số metric`, kể cả những hàng sau
    đó bị đánh KHÔNG SO ĐƯỢC. Hơi bảo thủ, và cố ý: `m` phải tính được **trước**
    khi chạy, còn nếu nó phụ thuộc kết quả thì chính nó thành một lựa chọn dựa
    trên dữ liệu.

    `correct=False` chỉ để test dựng được ca đối chứng — đừng dùng khi báo cáo.
    """
    groups = baseline.groups(dimension)
    names = list(metrics) or sorted(
        {m for qid in baseline.scores for m in baseline.scores[qid]}
        & {m for qid in candidate.scores for m in candidate.scores[qid]}
    )
    family = len(groups) * len(names) if correct else 1
    logger.info(
        "Chia theo %s: %d nhóm × %d metric = %d phép kiểm. alpha = %.5g%s",
        dimension,
        len(groups),
        len(names),
        len(groups) * len(names),
        DEFAULT_ALPHA / family,
        "" if correct else "  ⚠️ KHÔNG hiệu chỉnh (correct=False)",
    )
    out: dict[str, list[ComparisonRow]] = {}
    for value, query_ids in groups.items():
        label = value or "(không nhãn)"
        try:
            out[label] = compare_runs(
                baseline.subset(query_ids),
                candidate.subset(query_ids),
                metrics=names,
                iterations=iterations,
                seed=seed,
                family_size=family,
            )
        except ValueError as exc:
            # Nhóm không có câu chung: ghi lại rồi đi tiếp. Bỏ im lặng thì bảng
            # thiếu một nhóm và không ai biết là thiếu.
            logger.warning("Nhóm %s bỏ qua: %s", label, exc)
    return out


def _detail(row: ComparisonRow) -> str:
    """Ô "kiểm định" của một hàng — chỗ duy nhất quyết định người đọc thấy gì."""
    if not row.comparable:
        return f"⚠️ {row.note}"
    if row.p_value is not None:
        text = f"p={row.p_value:.4g} · {row.n_baseline_only}↔{row.n_candidate_only} câu đổi chiều"
        if row.underpowered:
            # Nói ra trần, không chỉ nói là không đạt: `p` nhỏ nhất có thể của
            # `n` câu đổi chiều là một con số, và nó là lý do hàng này vô vọng.
            text += (
                f" · **trần `p` = {min_achievable_p(row.n_discordant):.4g}** ở α={row.alpha:.4g}"
            )
        return text
    label = "CI95" if row.family_size == 1 else f"CI{100 * (1 - row.alpha):.4g}%"
    text = f"{label} [{row.ci_low:+.4f}, {row.ci_high:+.4f}]"
    if row.note:
        text += f" · {row.note}"
    if row.ci_touches_zero:
        step = 1.0 / row.n_queries if row.n_queries else 0.0
        text += f" · **biên đúng 0** (bước lưới 1/{row.n_queries} = {step:.4f})"
    return text


def format_table(
    rows: Sequence[ComparisonRow],
    *,
    baseline: str,
    candidate: str,
    show_n: bool = False,
) -> str:
    """Bảng Markdown. `show_n=True` thêm cột `n` — bắt buộc cho bảng chia nhóm."""
    head = f"| metric | {baseline} | {candidate} | Δ |"
    rule = "|---|---:|---:|---:|"
    if show_n:
        head += " n |"
        rule += "---:|"
    out = [head + " kiểm định | kết luận |", rule + "---|---|"]
    for r in rows:
        cells = f"| `{r.metric}` | {r.baseline:.4f} | {r.candidate:.4f} | {r.delta:+.4f} |"
        if show_n:
            cells += f" {r.n_queries} |"
        out.append(f"{cells} {_detail(r)} | {r.verdict} |")
    return "\n".join(out)


def format_grouped_table(
    groups: Mapping[str, Sequence[ComparisonRow]],
    *,
    baseline: str,
    candidate: str,
    dimension: str,
) -> str:
    """Một khối bảng cho mỗi nhóm, kèm phần mở đầu nói rõ đã hỏi bao nhiêu câu hỏi.

    Phần mở đầu **không** phải trang trí. Bảng chia nhóm là một cuộc tìm kiếm, và
    một bảng tìm kiếm không nói ra cỡ họ phép kiểm là một bảng mời người đọc trích
    dòng thuận với mình. Ghi ngay đầu file để không đọc được bảng mà bỏ qua nó.
    """
    first = next(iter(groups.values()), ())
    family = first[0].family_size if first else 1
    alpha = first[0].alpha if first else DEFAULT_ALPHA
    total = sum(len(rows) for rows in groups.values())
    every = [row for rows in groups.values() for row in rows]
    boundary = sum(1 for row in every if row.ci_touches_zero)
    powerless = sum(1 for row in every if row.underpowered)

    out = [
        f"# So theo `{dimension}`: `{baseline}` → `{candidate}`",
        "",
        f"- **{len(groups)} nhóm × {total // max(len(groups), 1)} metric = {total} phép kiểm.**",
    ]
    if family > 1:
        out += [
            f"- Ngưỡng đã hiệu chỉnh Bonferroni: **α = {alpha:.4g}** (từ "
            f"{DEFAULT_ALPHA} chia cho {family}).",
            f"- Không hiệu chỉnh thì ở α={DEFAULT_ALPHA} kỳ vọng **"
            f'{DEFAULT_ALPHA * total:.1f} kết quả "có ý nghĩa" thuần do ngẫu nhiên** — '
            'tức câu hỏi "nhóm nào cải thiện nhiều nhất" gần như bảo đảm tìm ra một '
            "cái, kể cả khi không có gì.",
        ]
    else:
        out.append(
            f"- ⚠️ **KHÔNG hiệu chỉnh đa so sánh** (`--no-correct`). Ở α={DEFAULT_ALPHA} "
            f'kỳ vọng {DEFAULT_ALPHA * total:.1f} kết quả "có ý nghĩa" thuần do ngẫu '
            "nhiên. Đừng trích bảng này để tuyên bố người thắng."
        )
    out += [
        "- `KHÔNG ĐỦ LỰC` = phép kiểm **không thể** đạt ý nghĩa dù kết quả thế nào "
        "(`p` nhỏ nhất McNemar cho `n` câu đổi chiều đã ≥ α). Nó **không** phải "
        '"không có khác biệt".',
        "- `KHÔNG SO ĐƯỢC` = phân bố nhãn hai bên khác nhau nên mẫu số của metric khác "
        "nhau (`G2`/`TD-11`).",
        "- `KHÔNG KẾT LUẬN` = khoảng tin cậy có **một biên đúng bằng 0**. Với metric "
        "rời rạc thưa, phân bố bootstrap nằm trên lưới bước `1/n` và phân vị cực đoan "
        'rơi đúng lên 0 — nên việc khoảng "chứa 0" phụ thuộc đúng một bước phân '
        "giải. Đã đo: đổi seed dịch biên **đúng một bước lưới**, và tăng 10.000 → "
        "50.000 iterations **không đổi gì**, nên đừng chữa bằng cách tăng iterations.",
        "",
    ]
    if powerless or boundary:
        # Đếm ngay đầu file: nếu phần lớn bảng không kết luận được thì đó là điều
        # đầu tiên người đọc phải biết, không phải điều họ tự cộng ra khi đọc hết.
        out += [
            f"- ⚠️ **{powerless}/{total} hàng `KHÔNG ĐỦ LỰC` và {boundary}/{total} hàng "
            f"`KHÔNG KẾT LUẬN`.** Chia nhỏ tập đo rồi hiệu chỉnh cho "
            f"{family} phép kiểm để lại rất ít lực; đọc bảng này để **loại** giả "
            "thuyết thì được, để **chọn người thắng** thì không.",
            "",
        ]
    for label, rows in groups.items():
        out += [
            f"## `{dimension} = {label}` — n = {rows[0].n_queries if rows else 0}",
            "",
            format_table(rows, baseline=baseline, candidate=candidate, show_n=True),
            "",
        ]
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="So hai lần chạy retrieval eval, có kiểm định thống kê"
    )
    parser.add_argument("baseline", help="Tên run hoặc đường dẫn tới *-per-query.jsonl")
    parser.add_argument("candidate", help="Tên run hoặc đường dẫn")
    parser.add_argument("--dir", type=Path, default=Path("plans/reports/runs"))
    parser.add_argument("--metrics", nargs="*", default=[])
    parser.add_argument("--iterations", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, help="Ghi bảng Markdown ra file")
    parser.add_argument("--log-level", default="INFO")

    # --- hai chiều chia nhóm, và chúng KHÁC nhau về mặt suy luận thống kê -------
    parser.add_argument(
        "--by",
        choices=GROUP_DIMENSIONS,
        help="Quét **mọi** nhóm của chiều này và in một bảng cho từng nhóm. Đây là "
        "một cuộc TÌM KIẾM, nên nó tự hiệu chỉnh đa so sánh (Bonferroni trên "
        "`số nhóm × số metric`). Dùng cho DoD `W2-09` ('category nào cải thiện "
        "nhiều nhất').",
    )
    parser.add_argument(
        "--category",
        help="Chỉ so trên MỘT category đã nêu trước. Là một giả thuyết đơn, nên "
        "KHÔNG hiệu chỉnh — khác `--by category`, và khác có chủ đích: nêu tên "
        "nhóm trước khi xem số là kiểm một giả thuyết, quét hết rồi chọn cái to "
        "nhất là chọn mẫu.",
    )
    parser.add_argument("--lang", help="Như `--category`, nhưng theo ngôn ngữ truy vấn.")
    parser.add_argument(
        "--no-correct",
        action="store_true",
        help="Tắt hiệu chỉnh của `--by`. ⚠️ Chỉ để đối chứng, đừng dùng khi báo cáo.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)-7s %(message)s")

    if args.by and (args.category or args.lang):
        parser.error(
            "`--by` và `--category`/`--lang` không đi cùng nhau: một cái quét mọi "
            "nhóm (có hiệu chỉnh), cái kia so một nhóm đã nêu trước (không hiệu "
            "chỉnh). Gộp lại thì không nói được kết quả thuộc loại suy luận nào."
        )

    def resolve(name: str) -> Path:
        direct = Path(name)
        return direct if direct.exists() else args.dir / f"{name}-per-query.jsonl"

    base = load_per_query(resolve(args.baseline), name=args.baseline)
    cand = load_per_query(resolve(args.candidate), name=args.candidate)

    if args.by:
        groups = compare_by_group(
            base,
            cand,
            args.by,
            metrics=args.metrics,
            iterations=args.iterations,
            seed=args.seed,
            correct=not args.no_correct,
        )
        table = format_grouped_table(
            groups, baseline=args.baseline, candidate=args.candidate, dimension=args.by
        )
    else:
        for dimension, wanted in (("category", args.category), ("lang", args.lang)):
            if not wanted:
                continue
            available = base.groups(dimension)
            if wanted not in available:
                # Nêu tên nhóm không tồn tại phải **nổ**, không trả bảng rỗng: một
                # bảng rỗng đọc y như "nhóm này không có khác biệt".
                parser.error(
                    f"Không có {dimension} = {wanted!r} trong {args.baseline}. "
                    f"Có: {', '.join(f'{k} ({len(v)} câu)' for k, v in available.items())}"
                )
            ids = available[wanted]
            base, cand = base.subset(ids), cand.subset(ids)
            logger.info("Lọc %s = %s → %d câu", dimension, wanted, len(ids))
        rows = compare_runs(
            base, cand, metrics=args.metrics, iterations=args.iterations, seed=args.seed
        )
        subset = bool(args.category or args.lang)
        table = format_table(rows, baseline=args.baseline, candidate=args.candidate, show_n=subset)
        if subset:
            chosen = [("category", args.category), ("lang", args.lang)]
            named = " · ".join(f"`{dim} = {val}`" for dim, val in chosen if val)
            table = (
                f"> Tập con đã nêu trước: {named} — "
                f"**{rows[0].n_queries if rows else 0} câu**. Một giả thuyết, "
                "không hiệu chỉnh đa so sánh.\n\n" + table
            )
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n", encoding="utf-8")
        logger.info("Đã ghi %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
