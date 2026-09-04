"""Thống kê mức đồng thuận giữa hai người chấm. `W5-04`.

Module này **không biết gì về LLM, về RAG, hay về file nào**. Nó nhận các cặp
nhãn và trả ra số. Lý do tách ra: mọi con số trong `judge-calibration.md` phải
kiểm lại được bằng tay trên giấy, và điều đó chỉ khả thi nếu phép tính không
dính vào việc đọc cache hay gọi mạng.

## ⭐⭐ 1. Cohen's kappa **không xác định** khi cả hai bên dùng đúng một nhãn

Đây không phải ca hiếm ở đây — nó là ca mặc định. Phân bố nhãn faithfulness của
`w5-answers-v1` là 402 `SUPPORTED` / 26 `NO_CLAIM` / 5 `NOT_FOUND` / 0
`CONTRADICTED`. Với biên độ ấy, `Pe = 0,866`: mẫu số `1 - Pe` chỉ còn `0,134`,
nên **một bất đồng thêm trên 50 mẫu kéo kappa đi ~0,15**.

Đẩy tới cùng, nếu hai bên cùng gán `SUPPORTED` cho cả 50 mẫu thì `Po = 1`,
`Pe = 1`, và kappa là `0/0`. Cám dỗ là trả `1.0` — "họ đồng ý hoàn toàn mà".
Nhưng `0/0` ở đây nghĩa là **không có thông tin**: hai cái đồng hồ đứng yên
cũng chỉ cùng một giờ. Trả `1.0` là biến một phép đo trống thành điểm tuyệt đối,
và nó sẽ đi thẳng vào một bảng báo cáo dưới dạng "kappa = 1,00 ✅".

Nên `cohen_kappa` trả `None`, và mọi thứ gọi nó phải xử `None` một cách tường
minh. `PABAK` tồn tại bên cạnh chính để đọc được những ca ấy.

## ⭐ 2. Nhãn ở đây là **danh định**, không phải thứ tự — nên không có trọng số ô

`SUPPORTED` / `CONTRADICTED` / `NOT_FOUND` / `NO_CLAIM` không xếp được thành
thang. Nhầm `SUPPORTED` thành `NOT_FOUND` không "gần đúng hơn" nhầm thành
`CONTRADICTED`. Weighted kappa (linear/quadratic) đòi thang thứ tự; áp nó lên
nhãn danh định là tạo ra một con số cao hơn một cách vô căn cứ. Module này cố ý
**không** có nó.

Trọng số duy nhất ở đây là `Pair.weight` — trọng số **lấy mẫu**, không phải
trọng số ô. Xem điểm 3.

## ⭐⭐ 3. Mẫu phân tầng cần hai con số, không phải một

Chấm 50 mẫu lấy ngẫu nhiên đều từ một quần thể 92,8% một nhãn thì kỳ vọng chỉ
có **0,58 mẫu `NOT_FOUND`** — nhánh judge dễ sai nhất thường không có mẫu nào.
Nên mẫu phải phân tầng, và khi đã phân tầng thì kappa tính trên 50 mẫu ấy
**không phải** kappa của quần thể: tầng hiếm đã bị bơm lên.

Hai con số, mỗi con số trả lời một câu khác nhau:

* **kappa mẫu** (`weight=1`) — "ở chỗ khó, judge và người hợp nhau tới đâu?"
* **kappa quần thể** (`weight = |tầng| / |mẫu của tầng|`, ước lượng
  Horvitz–Thompson) — "nếu chấm tay cả 433 mệnh đề thì kappa là bao nhiêu?"

Báo cáo chỉ một trong hai là nói dối bằng cách bỏ bớt. Con số đầu nghe tệ hơn
thực tế; con số sau nghe tốt hơn chỗ khó.

## ⭐ 4. Bootstrap phải lấy lại mẫu **trong từng tầng**

Lấy lại mẫu trên toàn bộ 50 cặp sẽ sinh ra những mẫu lặp có 0 phần tử ở tầng
`NOT_FOUND` — tức là bootstrap trên một thiết kế lấy mẫu khác với thiết kế thật.
Khoảng tin cậy ra sẽ rộng hơn thực tế và bị lệch. `bootstrap_ci` vì vậy nhóm
theo `Pair.stratum` và lấy lại mẫu có hoàn lại **trong** từng nhóm, giữ nguyên
cỡ mỗi tầng.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "Confusion",
    "Pair",
    "accuracy_of",
    "bootstrap_ci",
    "cohen_kappa",
    "confusion",
    "expected_agreement",
    "observed_agreement",
    "pabak",
    "per_label",
    "rate_of",
]


@dataclass(frozen=True)
class Pair:
    """Một mệnh đề đã được cả hai bên chấm.

    `a` theo quy ước là **người** (hoặc bên được coi là chuẩn khi tính
    precision/recall), `b` là bên còn lại. Kappa đối xứng nên với nó thứ tự
    không đổi gì; `per_label` thì có.
    """

    a: str
    b: str
    stratum: str = ""
    weight: float = 1.0
    ref: str = ""

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(f"weight phải > 0, nhận {self.weight}")


@dataclass(frozen=True)
class Confusion:
    """Ma trận nhầm lẫn có trọng số. Hàng = `a`, cột = `b`.

    Ô là `float` chứ không `int` vì mẫu phân tầng đưa vào trọng số lấy mẫu. Một
    ma trận `weight=1` toàn phần thì mọi ô vẫn là số nguyên đúng nghĩa.
    """

    labels: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]
    n_items: int
    """Số cặp **thật sự** đã chấm — không phải tổng trọng số.

    Giữ riêng vì mọi lời khai về độ tin cậy phải dựa trên số quan sát thật, chứ
    không dựa trên 433 "quan sát" mà 30 trong số đó được nhân lên 13,4 lần.
    """

    @property
    def total(self) -> float:
        return sum(sum(row) for row in self.matrix)

    @property
    def row_totals(self) -> tuple[float, ...]:
        return tuple(sum(row) for row in self.matrix)

    @property
    def col_totals(self) -> tuple[float, ...]:
        return tuple(sum(col) for col in zip(*self.matrix, strict=True))

    def cell(self, a: str, b: str) -> float:
        return self.matrix[self.labels.index(a)][self.labels.index(b)]

    def as_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "matrix": [[round(v, 4) for v in row] for row in self.matrix],
            "n_items": self.n_items,
            "total_weight": round(self.total, 4),
        }


def confusion(pairs: Sequence[Pair], labels: Sequence[str]) -> Confusion:
    """Dựng ma trận nhầm lẫn. Nhãn ngoài `labels` là **lỗi**, không bị bỏ qua.

    Bỏ qua lặng lẽ một nhãn lạ sẽ làm mẫu số tụt đi mà con số vẫn trông hợp lệ —
    đúng loại hỏng mà một module metric không được phép có.
    """
    index = {label: i for i, label in enumerate(labels)}
    if len(index) != len(labels):
        raise ValueError(f"tập nhãn trùng: {labels}")
    grid = [[0.0] * len(labels) for _ in labels]
    for pair in pairs:
        for side, value in (("a", pair.a), ("b", pair.b)):
            if value not in index:
                raise ValueError(
                    f"nhãn ngoài tập khai báo ở {side} của {pair.ref or '<không ref>'}: "
                    f"{value!r} ∉ {list(labels)}"
                )
        grid[index[pair.a]][index[pair.b]] += pair.weight
    return Confusion(
        labels=tuple(labels),
        matrix=tuple(tuple(row) for row in grid),
        n_items=len(pairs),
    )


def observed_agreement(cm: Confusion) -> float | None:
    """`Po` — tỉ trọng nằm trên đường chéo."""
    total = cm.total
    if total <= 0:
        return None
    return sum(cm.matrix[i][i] for i in range(len(cm.labels))) / total


def expected_agreement(cm: Confusion) -> float | None:
    """`Pe` — mức đồng thuận nếu hai bên gán nhãn độc lập, giữ nguyên biên độ."""
    total = cm.total
    if total <= 0:
        return None
    rows, cols = cm.row_totals, cm.col_totals
    return sum(rows[i] * cols[i] for i in range(len(cm.labels))) / (total * total)


def cohen_kappa(cm: Confusion) -> float | None:
    """`(Po - Pe) / (1 - Pe)`, hoặc `None` khi không xác định.

    `None` xảy ra khi `Pe == 1`: cả hai bên chỉ dùng đúng một nhãn. Xem điểm 1
    ở docstring module — đó là "không có thông tin", không phải "đồng thuận hoàn
    hảo", và trả `1.0` ở đây là cách một báo cáo tự khen mình.
    """
    po, pe = observed_agreement(cm), expected_agreement(cm)
    if po is None or pe is None:
        return None
    if abs(1.0 - pe) < 1e-12:
        return None
    return (po - pe) / (1.0 - pe)


def pabak(cm: Confusion) -> float | None:
    """Prevalence-adjusted bias-adjusted kappa: `2·Po − 1`.

    Đây là kappa mà ta *sẽ* thu được nếu biên độ cân bằng hoàn toàn. Nó không
    thay thế kappa — nó là cách đọc được những ca kappa sụp vì biên độ lệch.
    Chênh lệch lớn giữa `pabak` và `cohen_kappa` chính là **độ lớn của nghịch lý
    kappa** trên tập này, và đó là con số phải nằm trong báo cáo.
    """
    po = observed_agreement(cm)
    return None if po is None else 2.0 * po - 1.0


def per_label(cm: Confusion) -> dict[str, dict[str, float | None]]:
    """Precision/recall/F1 của `b` (judge) khi coi `a` (người) là chuẩn.

    Tồn tại vì kappa là **một** con số cho **cả** ma trận: nó không phân biệt
    "judge bỏ sót mọi `NOT_FOUND`" với "judge báo động giả `NOT_FOUND` khắp
    nơi", trong khi hai lỗi ấy có hệ quả trái ngược nhau lên con số faithfulness.
    """
    out: dict[str, dict[str, float | None]] = {}
    rows, cols = cm.row_totals, cm.col_totals
    for i, label in enumerate(cm.labels):
        tp = cm.matrix[i][i]
        precision = tp / cols[i] if cols[i] > 0 else None
        recall = tp / rows[i] if rows[i] > 0 else None
        if precision and recall:
            f1: float | None = 2 * precision * recall / (precision + recall)
        else:
            f1 = None if precision is None or recall is None else 0.0
        out[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support_a": rows[i],
            "support_b": cols[i],
        }
    return out


# ------------------------------------------------------------------ tỉ lệ dẫn xuất


def accuracy_of(pairs: Sequence[Pair]) -> float | None:
    """Tỉ lệ hai bên trùng nhau, có trọng số. Bằng `Po` nhưng không cần ma trận."""
    total = sum(p.weight for p in pairs)
    if total <= 0:
        return None
    return sum(p.weight for p in pairs if p.a == p.b) / total


def rate_of(
    pairs: Sequence[Pair],
    *,
    side: str,
    numerator: Iterable[str],
    denominator: Iterable[str],
) -> float | None:
    """Tỉ lệ `numerator / denominator` trên nhãn của một bên, có trọng số.

    Đây là cách tính lại **chính con số faithfulness** từ nhãn tay: tử số
    `{SUPPORTED}`, mẫu số `{SUPPORTED, CONTRADICTED, NOT_FOUND}` — đúng quy ước
    của `score_faithfulness`, chỉ đổi nguồn nhãn. Nhờ vậy "judge nói 0,9877"
    và "người nói bao nhiêu" là hai con số **so sánh trực tiếp được**, chứ không
    phải hai định nghĩa khác nhau đặt cạnh nhau.
    """
    if side not in ("a", "b"):
        raise ValueError(f"side phải là 'a' hoặc 'b', nhận {side!r}")
    num, den = frozenset(numerator), frozenset(denominator)
    if not num <= den:
        raise ValueError(f"tử số phải là tập con của mẫu số: {sorted(num - den)} thừa")
    total = sum(p.weight for p in pairs if getattr(p, side) in den)
    if total <= 0:
        return None
    return sum(p.weight for p in pairs if getattr(p, side) in num) / total


# ---------------------------------------------------------------------- bootstrap


def bootstrap_ci(
    pairs: Sequence[Pair],
    stat: Callable[[Sequence[Pair]], float | None],
    *,
    n_resamples: int = 2000,
    seed: int = 20260905,
    level: float = 0.95,
) -> dict[str, Any]:
    """Khoảng tin cậy percentile, lấy lại mẫu **trong từng tầng**.

    Trả kèm `n_undefined` — số lần lấy lại mẫu mà `stat` trả `None`. Con số ấy
    không được giấu: nếu 300/2000 lần lấy mẫu cho kappa không xác định thì
    khoảng tin cậy in ra chỉ mô tả 1700 lần còn lại, và đó là một sự thật khác
    hẳn về độ ổn định của phép đo.
    """
    if not 0 < level < 1:
        raise ValueError(f"level phải nằm trong (0,1), nhận {level}")
    if n_resamples < 1:
        raise ValueError("n_resamples phải ≥ 1")
    buckets: dict[str, list[Pair]] = {}
    for pair in pairs:
        buckets.setdefault(pair.stratum, []).append(pair)

    rng = random.Random(seed)
    values: list[float] = []
    undefined = 0
    for _ in range(n_resamples):
        draw: list[Pair] = []
        for bucket in buckets.values():
            draw.extend(rng.choices(bucket, k=len(bucket)))
        value = stat(draw)
        if value is None:
            undefined += 1
        else:
            values.append(value)

    point = stat(pairs)
    if not values:
        return {
            "point": point,
            "lo": None,
            "hi": None,
            "n_resamples": n_resamples,
            "n_undefined": undefined,
            "level": level,
        }
    values.sort()
    tail = (1.0 - level) / 2.0
    lo = values[min(len(values) - 1, int(tail * len(values)))]
    hi = values[min(len(values) - 1, int((1.0 - tail) * len(values)))]
    return {
        "point": point,
        "lo": lo,
        "hi": hi,
        "median": statistics.median(values),
        "n_resamples": n_resamples,
        "n_undefined": undefined,
        "level": level,
    }
