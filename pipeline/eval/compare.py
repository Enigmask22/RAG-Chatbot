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
    "MIN_TAIL_RESAMPLES",
    "RESOLUTION_FLAGS",
    "BootstrapBounds",
    "ComparisonRow",
    "RunScores",
    "bootstrap_intervals",
    "bootstrap_resample",
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

#: Tên ba cờ nói "phép so này chạm giới hạn phân giải của dữ liệu, không phải
#: giới hạn của hệ thống được đo". Gom lại thành một hằng số để bảng nào in
#: chú giải cũng in đủ cả ba — thiếu một cái là bảng mời người đọc kết luận sai
#: theo đúng hướng mà cái đó canh.
RESOLUTION_FLAGS = (
    "underpowered",
    "ci_within_resolution",
    "direction_split",
    "mc_unstable",
)

#: Số mẫu lại tối thiểu trong đuôi để một biên phân vị đọc được.
#:
#: Biên dưới của khoảng `alpha` là phần tử thứ `alpha/2 * B` của dãy đã sắp. Với
#: `alpha = 0,05` và `B = 10.000` thì đó là phần tử thứ 250 — đọc được. Với
#: `alpha` đã hiệu chỉnh Bonferroni cho 39 phép kiểm (0,00128) thì đó là phần tử
#: thứ **6**, và một quyết định kiến trúc dựa vào 6 mẫu lại là dựa vào nhiễu.
#:
#: Con số 30 chọn từ đo thật ở `W2-08` (`rc50 → rc100`, 6 seed × 3 mức `B`):
#: đuôi 6 mẫu → **biên dưới đổi dấu theo seed**; đuôi 32 → dấu ổn định; đuôi 128
#: → sd của biên tụt còn 1/5. Xem `ComparisonRow.mc_unstable`.
MIN_TAIL_RESAMPLES = 30

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

    min_increment: float = 0.0
    """|hiệu| khác 0 **nhỏ nhất** trên một câu — độ hạt của metric, đo từ dữ liệu.

    Mặc định `0,0` nghĩa **"chưa biết"**, và khi chưa biết thì `grid_step` bằng 0
    nên `ci_within_resolution` **không bật**. Đó là chủ ý: `compare_runs` luôn
    điền trường này cho mọi hàng bootstrap (có test ghim), nên mặc định chỉ gặp ở
    hàng dựng tay.

    ⚠️ Bản đầu mặc định `1,0` ("coi như nhị phân") và nó làm một test đã có đổi
    kết luận: một khoảng `[−0,1630, −0,0054]` của `ndcg@10` trên 43 câu bị dán
    `KHÔNG KẾT LUẬN` vì `1/43 = 0,0233 > 0,0054`, trong khi độ hạt thật của
    `ndcg@10` ở đó là `0,0068/43 = 0,00016`. Một cờ **đoán** ngưỡng của chính nó
    là một cờ bật theo phỏng đoán — mà đó đúng là loại lỗi cờ này được dựng để
    bắt.
    """

    ci_jitter: tuple[float, float] | None = None
    """Khoảng dao động của **biên gần 0 nhất**, do chính vị trí phân vị gây ra.

    Xem `mc_unstable`. `None` với hàng không đi đường bootstrap.
    """

    @property
    def n_discordant(self) -> int:
        """Số câu **đổi chiều** — thứ duy nhất McNemar dùng."""
        return self.n_baseline_only + self.n_candidate_only

    @property
    def grid_step(self) -> float:
        """Bước phân giải của trung bình: hiệu **nhỏ nhất một câu tạo ra được**, chia `n`.

        Mức chênh nhỏ nhất có thể tồn tại giữa hai lần chạy là "đúng một câu đổi,
        đổi ít nhất có thể" — tức `min_increment / n`.

        ⚠️ Bản đầu của tôi dùng `1/n` cho mọi metric, và nó **sai theo hướng dương
        giả**. `1/n` đúng cho metric nhị phân (một câu đổi thì đổi cả 1,0), nhưng
        `precision@20` nhận giá trị bội của 1/20 nên bước thật của nó nhỏ hơn **20
        lần**. Áp `1/n` lên nó thì mọi hiệu nhỏ hơn 0,0048 bị dán "dưới một bước
        lưới" — đo được: luật `1/n` làm **13/14** file `compare/` đã công bố đổi
        kết luận, và phần lớn là `precision@k`/`recall@k` bị gắn cờ oan.

        Hàng McNemar không đi qua đây (nó dùng `p_value`), nên `min_increment` chỉ
        cần đúng cho hàng bootstrap — và `compare_runs` đo nó từ chính các hiệu.
        """
        if not self.n_queries:
            return 0.0
        return self.min_increment / self.n_queries

    @property
    def sign_p(self) -> float | None:
        """`p` của kiểm định dấu trên các câu có hiệu khác 0.

        Với hàng McNemar thì đây **chính là** `p_value` (cùng một phép kiểm, cùng
        hai con số đếm). Với hàng bootstrap thì nó là một phép kiểm thứ hai trên
        cùng dữ liệu, và giá trị của nó nằm ở chỗ nó **bỏ qua độ lớn**: bootstrap
        đọc "trung bình hiệu khác 0", kiểm định dấu đọc "số câu tốt hơn nhiều hơn
        số câu xấu đi". Hai câu đó có thể trái nhau một cách hợp lệ (thắng ít câu
        nhưng thắng đậm), nên nó **không** dùng để phủ quyết bootstrap — xem
        `direction_split` để biết chỗ duy nhất nó được dùng.
        """
        if not self.comparable:
            return None
        return mcnemar_exact(self.n_baseline_only, self.n_candidate_only)

    @property
    def identical(self) -> bool:
        """Không một câu nào khác nhau — hai lần chạy **trùng khớp** trên metric này.

        Phải là một kết luận riêng, không được gộp vào "trong ngưỡng nhiễu" hay
        "KHÔNG KẾT LUẬN": 0 câu khác nhau trên 209 câu là hàng **chắc chắn nhất**
        trong bảng, không phải hàng mơ hồ nhất. Đo được ở `W2-08`: `recall@5` giữa
        `rrf k=0` và `rrf k=1` có `0/209` câu khác nhau và `Δ = 0`, và luật cũ dán
        nó là `KHÔNG KẾT LUẬN` vì CI là `[0, 0]` nên "có một biên đúng 0".
        """
        return self.comparable and self.n_discordant == 0 and self.delta == 0.0

    @property
    def direction_split(self) -> bool:
        """Trung bình nói một hướng, **đếm câu** nói hướng ngược lại (hoặc không nói gì).

        Đo được ở `W2-08` trên đúng cặp quyết định người thắng của cả bảng
        ablation. `rerank_candidates` 50 → 100, `ndcg@10`: `Δ = +0,0255` và
        CI99,87% `[+0,0003, +0,0638]` loại 0, tức "khác biệt thật". Nhưng đếm câu
        là **+10 tốt hơn / −11 xấu đi**: *nhiều câu bị làm hỏng hơn số câu được
        sửa*. Trung bình dương vì mấy câu thắng thắng đậm hơn mấy câu thua thua.

        Đó **có thể** là một kết quả thật (thắng ít câu nhưng thắng đậm là chuyện
        hợp lệ), nên cờ này không nói "sai" — nó nói **"đừng đọc dòng này thành
        hệ thống tốt hơn"**. Với một quyết định kiến trúc trả bằng 1,91× độ trễ
        thì một hiệu ứng mà đa số câu đi ngược hướng không đủ để chi tiền.

        Vì thế nó có kết luận **riêng** là `TRÁI CHIỀU`, không dùng chung
        `KHÔNG KẾT LUẬN` với hai cờ về giới hạn đọc số. Ở đây khoảng tin cậy
        *đọc được* hẳn hoi — nó đọc ra hai điều đối nhau, và đó là thông tin.

        `<= 0` chứ không `< 0`: đếm hoà đúng bằng nhau (10↔10) mang **không**
        thông tin nào về hướng, nên nó cũng phải bị gắn cờ.
        """
        if not self.comparable or self.delta == 0.0 or self.n_discordant == 0:
            return False
        majority = self.n_candidate_only - self.n_baseline_only
        return majority * self.delta <= 0

    @property
    def mc_unstable(self) -> bool:
        """Biên gần 0 nhất **không giữ được dấu** khi xét dao động của chính nó.

        Số mẫu lại nằm dưới biên dưới là một biến ngẫu nhiên nhị thức `B(B, α/2)`,
        nên độ lệch chuẩn của nó là `sqrt(tail)`. Đọc lại biên ở hai vị trí
        `tail ± sqrt(tail)` của **đúng dãy đã sắp đó** cho khoảng dao động của
        biên, **không tốn thêm một lần lấy mẫu nào**. Nếu khoảng ấy chứa 0 thì
        việc CI "loại 0" là chuyện của số mẫu lại, không phải của dữ liệu.

        ## Vì sao cần cờ này khi đã có `ci_within_resolution`

        Hai cờ bắt hai giới hạn khác nhau, và ca quyết định người thắng của bảng
        ablation `W2-08` chỉ bị bắt bởi cờ này:

        `rerank_candidates` 50 → 100, `ndcg@10`, α=0,00128, B=10.000 → đuôi **6
        mẫu**. Đo 6 seed: biên dưới nhận cả dấu `+` lẫn `−`. Ở B=50.000 (đuôi 32)
        và B=200.000 (đuôi 128) nó **âm nhất quán**, tức CI thật sự **chứa** 0.
        Kết luận "khác biệt thật" ở B=10.000 là một tạo tác Monte Carlo, và nó là
        thứ đã chọn ra cấu hình thắng cho cả bảng.

        Bước lưới của `ndcg@10` là `0,0068/209 = 3,2e-05`, còn biên là `+0,0003` —
        **gấp 9 lần** bước lưới, nên `ci_within_resolution` để nó đi qua. Đúng: đây
        không phải giới hạn phân giải của dữ liệu, mà là giới hạn số mẫu lại.

        ⚠️ Và nó **ngược** với ghi chú của `W2-08-prep` ("đừng chữa bằng cách tăng
        iterations"). Ghi chú đó đúng cho ca nó đo — metric nhị phân thưa, phân bố
        nằm trên lưới, tăng `B` chứng minh được là không đổi gì. Ca này là metric
        liên tục và tăng `B` **đảo kết luận**. Hai giới hạn khác nhau, cùng một
        triệu chứng "biên sát 0", và cách phân biệt là `min_increment`.
        """
        if self.ci_jitter is None:
            return False
        lo, hi = self.ci_jitter
        return lo <= 0.0 <= hi

    @property
    def ci_within_resolution(self) -> bool:
        """CI có một biên **gần 0 hơn một bước lưới** — biên giới, không phải kết luận.

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

        ## Vì sao luật là `< 1/n` chứ không phải `== 0` (sửa ở `W2-08`)

        Bản đầu của cờ này, viết ở `W2-08-prep`, kiểm **đúng** biên `== 0`. Nó bắt
        được ca đã gặp lúc đó và bỏ lọt đúng ca quyết định người thắng của bảng
        ablation `W2-08`: `rerank_candidates` 50 → 100 cho `ndcg@10` CI99,87%
        `[+0,0003, +0,0638]` và `mrr` `[+0,0007, +0,0625]`. Biên `+0,0003` là
        **1/16 của một bước lưới** (`1/209 = 0,0048`) — cùng hiện tượng y nguyên,
        nhưng `+0,0003 != 0` nên luật cũ để nó đi qua thành "khác biệt thật", và
        cấu hình thắng của cả bảng được quyết bởi con số đó.

        Việc biên rơi *đúng* 0 hay *gần* 0 là chuyện của lưới, không phải chuyện
        của hệ thống được đo. Một luật phân biệt hai ca đó đang phân biệt một tai
        nạn số học.

        ⚠️ Nhưng bước lưới **không** phải `1/n` cho mọi metric — xem `grid_step`.
        Và ca `ndcg@10` nói trên hoá ra không thuộc cờ này mà thuộc `mc_unstable`:
        biên `+0,0003` của nó cách 0 **gấp 9 lần** bước lưới `3,2e-05`. Hai giới
        hạn khác nhau; đừng gộp.

        Một luật, hai hướng: nó bắt cả ca **chứa 0 sát rìa** (đọc thành "không có
        hiệu ứng" thì sai) và ca **loại 0 sát rìa** (đọc thành "có hiệu ứng" thì
        sai). Cả hai đều là "kết luận đang dựa vào ít hơn một bước phân giải".
        """
        if not self.comparable or self.ci_low is None or self.ci_high is None:
            return False
        return min(abs(self.ci_low), abs(self.ci_high)) < self.grid_step

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

        ## ⚠️ Cờ này CỐ Ý không áp cho hàng bootstrap

        `p_value is None` (hàng bootstrap) trả `False`, và đó là một quyết định,
        không phải một chỗ chưa làm. `min_achievable_p` là **trần của kiểm định
        dấu** — một phép kiểm chỉ đếm hướng. Bootstrap dùng cả **độ lớn**, nên nó
        có lực hơn hẳn trên cùng số câu, và áp trần của phép kiểm yếu hơn lên nó
        sẽ **xoá những kết quả thật**: đo được ở `W2-08`, `recall@5` của
        `rc50 → rc100` có 10 câu khác nhau (trần dấu 0,00195 ≥ α=0,00128) nhưng
        `+9/−1` là một hướng rất rõ.

        Hai cờ dành cho hàng bootstrap — `ci_within_resolution` và
        `direction_split` — đọc **khoảng đã quan sát được**, không đọc một cái trần
        giả định. Đó là chỗ khác nhau, và nó là lý do có ba cờ chứ không phải một.
        """
        if not self.comparable or self.p_value is None:
            return False
        return min_achievable_p(self.n_discordant) >= self.alpha

    @property
    def verdict(self) -> str:
        """Kết luận một hàng. **Thứ tự các nhánh là nội dung**, không phải hình thức.

        Từ chặt tới lỏng: không so được → trùng khớp → không đủ lực → có/không có
        khác biệt. Mỗi nhánh trước loại bỏ một cách đọc sai của nhánh sau, nên đổi
        thứ tự là đổi kết luận: `identical` phải đứng trước `underpowered` vì
        "0 câu đổi chiều" vừa là trùng khớp hoàn toàn *vừa* thoả trần
        `min_achievable_p(0) = 1 >= alpha`, và nhánh nào chạy trước thì thắng.
        """
        if not self.comparable:
            return "KHÔNG SO ĐƯỢC"
        if self.identical:
            return "TRÙNG KHỚP"
        if self.underpowered:
            return "KHÔNG ĐỦ LỰC"
        if self.p_value is not None:
            return "khác biệt thật" if self.p_value < self.alpha else "trong ngưỡng nhiễu"
        if self.ci_low is not None and self.ci_high is not None:
            excludes_zero = (self.ci_low > 0 and self.ci_high > 0) or (
                self.ci_low < 0 and self.ci_high < 0
            )
            # Hai cờ này chỉ được hỏi ở đây, khi khoảng đang **tuyên bố một
            # hướng**. Hỏi chúng ở nhánh "chứa 0" thì thành gắn cờ cho một hàng
            # vốn đã không kết luận gì — thêm chữ, không thêm thông tin.
            unreadable = self.ci_within_resolution or self.mc_unstable
            if excludes_zero and not (unreadable or self.direction_split):
                return "khác biệt thật"
            # Hai chuyện khác nhau, hai chữ khác nhau. `KHÔNG KẾT LUẬN` = *không
            # đọc được* khoảng này. `TRÁI CHIỀU` = đọc được, và nó nói hai điều
            # đối nhau: trung bình lên mà đa số câu bị làm hỏng. Gộp lại thì mất
            # đúng phần thông tin đáng giá nhất — xem `direction_split`.
            if excludes_zero:
                return "KHÔNG KẾT LUẬN" if unreadable else "TRÁI CHIỀU"
            # Biên sát 0 = biên giới của lưới rời rạc, không phải bằng chứng âm.
            # Xem `ci_within_resolution`. Gộp nó vào "trong ngưỡng nhiễu" là chỗ
            # mà một giới hạn phân giải bị đọc thành một kết luận.
            return "KHÔNG KẾT LUẬN" if self.ci_within_resolution else "trong ngưỡng nhiễu"
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


@dataclass(frozen=True)
class BootstrapBounds:
    """Một khoảng tin cậy **kèm mức dao động của chính hai biên nó**."""

    low: float
    high: float
    low_jitter: tuple[float, float]
    high_jitter: tuple[float, float]
    tail: int
    """Số mẫu lại nằm dưới biên dưới. Xem `MIN_TAIL_RESAMPLES`."""

    def near_jitter(self) -> tuple[float, float]:
        """Dao động của biên **gần 0 nhất** — biên quyết định khoảng có loại 0 hay không."""
        return self.low_jitter if abs(self.low) <= abs(self.high) else self.high_jitter


def bootstrap_resample(
    diffs: Sequence[float],
    alphas: Sequence[float],
    *,
    iterations: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> dict[float, BootstrapBounds]:
    """Nhiều khoảng tin cậy từ **một** lần lấy mẫu lại, kèm dao động của từng biên.

    Bảng chia nhóm cần cả khoảng 95% (để đọc) và khoảng đã hiệu chỉnh Bonferroni
    (để kết luận). Gọi `paired_bootstrap` hai lần sẽ lấy mẫu lại hai lần — gấp đôi
    chi phí cho **cùng một** phân bố, và với hai `seed` giống nhau thì hai kết quả
    vẫn đến từ cùng dãy số ngẫu nhiên nên chẳng độc lập gì. Sắp một lần, lấy phân
    vị nhiều lần.

    Seed cố định để hai lần chạy công cụ cho cùng con số — một khoảng tin cậy nhảy
    nhót mỗi lần gọi thì không dùng để quyết định được.

    ## Dao động của biên, tính từ đúng dãy đã sắp — không tốn thêm lần lấy mẫu nào

    Số mẫu lại nằm dưới biên dưới là `B(B, alpha/2)`, độ lệch chuẩn `sqrt(tail)`.
    Đọc lại biên ở `tail ± sqrt(tail)` cho khoảng dao động của nó. Nếu khoảng ấy
    chứa 0 thì việc khoảng tin cậy loại 0 là chuyện của số mẫu lại, không phải của
    dữ liệu — xem `ComparisonRow.mc_unstable`, và xem con số đã đo ở `W2-08` nơi
    một cấu hình thắng được quyết định bởi **6** mẫu lại trên 10.000.
    """
    if not diffs:
        flat = (0.0, 0.0)
        return {alpha: BootstrapBounds(0.0, 0.0, flat, flat, 0) for alpha in alphas}
    rng = random.Random(seed)
    n = len(diffs)
    means: list[float] = []
    for _ in range(iterations):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()

    def at(index: int) -> float:
        return means[max(0, min(iterations - 1, index))]

    out: dict[float, BootstrapBounds] = {}
    for alpha in alphas:
        lo_i = int(alpha / 2 * iterations)
        hi_i = int((1 - alpha / 2) * iterations)
        # Hai đuôi đối xứng: số mẫu **dưới** biên dưới và số mẫu **trên** biên trên
        # đều là `lo_i`, nên độ lệch chuẩn của cả hai vị trí là `sqrt(lo_i)`. Một
        # `spread` dùng cho cả hai — tính riêng cho từng biên là chỗ để hai biên
        # lệch luật mà không ai thấy.
        spread = max(1, round(math.sqrt(max(lo_i, 1))))
        out[alpha] = BootstrapBounds(
            low=at(lo_i),
            high=at(hi_i),
            low_jitter=(at(lo_i - spread), at(lo_i + spread)),
            high_jitter=(at(hi_i - spread), at(hi_i + spread)),
            tail=lo_i,
        )
    return out


def bootstrap_intervals(
    diffs: Sequence[float],
    alphas: Sequence[float],
    *,
    iterations: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> dict[float, tuple[float, float]]:
    """Chỉ hai biên, không kèm dao động. Lớp mỏng của `bootstrap_resample`.

    Giữ chữ ký từ `W2-08-prep` vì mọi con số đã công bố đi qua nó, và giữ nó là
    **lớp mỏng** để hai đường không thể cho hai kết quả khác nhau (có test ghim).
    """
    return {
        alpha: (bounds.low, bounds.high)
        for alpha, bounds in bootstrap_resample(
            diffs, alphas, iterations=iterations, seed=seed
        ).items()
    }


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
        # Đếm câu mỗi bên thắng — **một** chỗ duy nhất, dùng cho cả hai đường kiểm
        # định. Trước `W2-08` chỉ đường McNemar đếm, nên hàng bootstrap không có
        # cách nào biết trung bình của nó đến từ 10 câu hay 137 câu; `Δ = +0,0255`
        # của `ndcg@10` trông y như nhau ở cả hai. Tính riêng cho từng đường thì
        # hai con số ấy lệch nhau được mà không ai thấy.
        n_base_better = sum(1 for a, b in pairs if a > b)
        n_cand_better = sum(1 for a, b in pairs if b > a)

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
            rows.append(
                ComparisonRow(
                    metric=metric,
                    baseline=mean_b,
                    candidate=mean_c,
                    delta=mean_c - mean_b,
                    test="McNemar exact",
                    p_value=mcnemar_exact(n_base_better, n_cand_better),
                    n_baseline_only=n_base_better,
                    n_candidate_only=n_cand_better,
                    n_queries=len(pairs),
                    alpha=alpha,
                    family_size=family_size,
                )
            )
            continue

        diffs = [b - a for a, b in pairs]
        # Một lần lấy mẫu lại, hai phân vị: khoảng 95% để **đọc**, khoảng đã hiệu
        # chỉnh để **kết luận**. Khi `family_size == 1` thì hai cái là một.
        intervals = bootstrap_resample(
            diffs, (DEFAULT_ALPHA, alpha), iterations=iterations, seed=seed
        )
        bounds = intervals[alpha]
        lo, hi = bounds.low, bounds.high
        nonzero = [abs(d) for d in diffs if d != 0.0]
        note = ""
        if family_size > 1:
            raw = intervals[DEFAULT_ALPHA]
            note = f"CI95 thô [{raw.low:+.4f}, {raw.high:+.4f}]"
        rows.append(
            ComparisonRow(
                metric=metric,
                baseline=mean_b,
                candidate=mean_c,
                delta=mean_c - mean_b,
                test=f"bootstrap cặp ({iterations})",
                ci_low=lo,
                ci_high=hi,
                ci_jitter=bounds.near_jitter(),
                min_increment=min(nonzero) if nonzero else 0.0,
                n_baseline_only=n_base_better,
                n_candidate_only=n_cand_better,
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
    """Ô "kiểm định" của một hàng — chỗ duy nhất quyết định người đọc thấy gì.

    Nguyên tắc: mỗi cờ phải in ra **con số làm nó bật**, không chỉ in tên nó.
    "KHÔNG ĐỦ LỰC" một mình là một lời khẳng định phải tin; "trần `p` = 0,125" là
    một con số kiểm lại được bằng giấy bút.
    """
    if not row.comparable:
        return f"⚠️ {row.note}"
    if row.identical:
        return f"**0/{row.n_queries} câu khác nhau** — hai lần chạy cho kết quả trùng khớp"
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
    # Đếm câu đi kèm **mọi** hàng bootstrap, không chỉ hàng bị gắn cờ: nó là thứ
    # cho biết trung bình này đến từ 10 câu hay 137 câu, và người đọc cần nó để
    # tự thấy hàng nào mỏng trước khi công cụ nói ra.
    text += f" · {row.n_baseline_only}↔{row.n_candidate_only} câu khác nhau"
    sign = row.sign_p
    if sign is not None:
        text += f" (p dấu={sign:.4g})"
    if row.note:
        text += f" · {row.note}"
    if row.ci_within_resolution:
        near = min(abs(row.ci_low or 0.0), abs(row.ci_high or 0.0))
        # In cả `min_increment`, không chỉ `n`: bước lưới là `min_increment/n`, và
        # bản đầu in "< 1/209" cho một metric có bước 0,2/209 — một thông điệp tự
        # nhận là đang so với một con số khác con số nó thật sự so.
        text += (
            f" · **biên cách 0 dưới một bước lưới** ({near:.5f} < "
            f"{row.min_increment:.4g}/{row.n_queries} = {row.grid_step:.5f})"
        )
    if row.direction_split:
        text += (
            f" · **đếm câu đi ngược Δ** ({row.n_candidate_only} câu tốt hơn vs "
            f"{row.n_baseline_only} câu xấu đi, mà Δ = {row.delta:+.4f})"
        )
    if row.mc_unstable and row.ci_jitter is not None:
        jlo, jhi = row.ci_jitter
        text += (
            f" · **biên không ổn định**: chính nó dao động [{jlo:+.4f}, {jhi:+.4f}] "
            "nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu"
        )
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
    boundary = sum(
        1 for row in every if row.ci_within_resolution or row.direction_split or row.mc_unstable
    )
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
        "- `KHÔNG KẾT LUẬN` = một trong hai chuyện. (a) Khoảng tin cậy có **một biên "
        "gần 0 hơn một bước lưới `1/n`**: với metric rời rạc thưa, phân bố bootstrap "
        "nằm trên lưới bước `1/n`, nên việc khoảng chứa hay loại 0 phụ thuộc đúng một "
        "bước phân giải. Đã đo: đổi seed dịch biên **đúng một bước lưới**, và tăng "
        "10.000 → 50.000 iterations **không đổi gì** — đừng chữa bằng cách tăng "
        "iterations. (b) **Đếm câu đi ngược `Δ`**: khoảng nói một hướng mà số câu tốt "
        "(b) **Biên không ổn định**: biên gần 0 nhất được đọc từ quá ít mẫu lại nên "
        "chính nó dao động qua 0. Đã đo ở `W2-08`: α đã hiệu chỉnh cho 39 phép kiểm "
        "để lại **6** mẫu lại trong đuôi của 10.000, và ở đó biên dưới **đổi dấu "
        "theo seed**; tăng lên 50.000 thì nó âm nhất quán, tức khoảng thật sự chứa 0.",
        "- `TRÁI CHIỀU` = khoảng tin cậy đọc được **và** nó nói hai điều đối nhau: "
        "trung bình đi một hướng, còn **số câu** thì đi hướng ngược lại (thắng ít câu "
        "nhưng thắng đậm). Không phải lỗi, nhưng đừng đọc thành `hệ thống tốt hơn` — "
        "nhất là khi đổi lấy độ trễ.",
        "- `TRÙNG KHỚP` = **0 câu khác nhau**. Đây là hàng chắc chắn nhất trong bảng, "
        "không phải hàng mơ hồ nhất — đừng đọc nó cùng nhóm với các cờ trên.",
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
