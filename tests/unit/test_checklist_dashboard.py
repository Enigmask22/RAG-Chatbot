"""Dashboard của `CHECKLIST.md` phải khớp với chính các mục nó tổng kết. `NEW-07`.

`CHECKLIST.md` §1 đã ba lần ghi nhận **cùng một lỗi**: đổi trạng thái một task ở
§2–§9 rồi quên cập nhật bảng tổng ở §1. Mỗi lần lại thêm một ghi chú cảnh báo, và
lần sau vẫn sai — lần này là lần thứ tư (dòng W0, W3 và W4 đều lệch).

Ba ghi chú không sửa được một việc phải làm bằng tay. Cái sửa được là một phép
đếm chạy tự động, và đó là file này. Cùng nguyên tắc với `W4-11` (đổi prompt =
tăng version là **cơ chế**, không phải quy ước) và với `test_architecture_-
boundaries.py`: rẻ, và đỏ đúng lúc lỗi mới xảy ra.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

CHECKLIST = Path(__file__).resolve().parents[2] / "plans" / "CHECKLIST.md"

#: Nhãn cột trong bảng §1 → ký hiệu trạng thái ở §2–§9.
COLUMNS: dict[str, str] = {
    "done": "x",
    "cho_test": "!",
    "dang_lam": "~",
    "bi_chan": "?",
    "hoan": "-",
    "todo": " ",
}

_SECTION = re.compile(r"^## \d+\. (W\d)\b|^## \d+\. (Task thêm mới)")
_BULLET = re.compile(r"^- \[(.)\] `((?:W\d|NEW)-\d{2}(?:-prep)?)`")
_TABLE_ROW = re.compile(r"^\|\s*`(NEW-\d{2})`\s*\|.*\|\s*`\[(.)\]`\s*\|")
#: Dòng dashboard: `| W4 · Serving Plane | 13 | 1 | 0 | 1 | 0 | 11 | ... |`
_NUMBERS = r"".join([r"\s*(\d+)\s*\|"] * (1 + len(COLUMNS)))
_DASH_ROW = re.compile(r"^\|\s*(W\d)\s*·[^|]*\|" + _NUMBERS)
_DASH_NEW = re.compile(r"^\|\s*§9 Task thêm mới[^|]*\|" + _NUMBERS)
#: Hai dòng tổng dùng `**…**`, nên số nằm giữa dấu sao.
_BOLD = r"".join([r"\s*\*\*(\d+)\*\*\s*\|"] * (1 + len(COLUMNS)))
_DASH_TOTAL = re.compile(r"^\|\s*\*\*Tổng (backlog gốc|cộng)\*\*\s*\|" + _BOLD)


def _parse() -> tuple[dict[str, Counter[str]], dict[str, tuple[int, ...]]]:
    """Trả `(đếm thật theo mục, dòng dashboard theo mục)`."""
    counted: dict[str, Counter[str]] = {}
    dashboard: dict[str, tuple[int, ...]] = {}
    section: str | None = None
    for line in CHECKLIST.read_text(encoding="utf-8").splitlines():
        header = _SECTION.match(line)
        if header:
            section = header.group(1) or "NEW"
            counted.setdefault(section, Counter())
            continue
        if (dash := _DASH_ROW.match(line)) is not None:
            dashboard[dash.group(1)] = tuple(int(g) for g in dash.groups()[1:])
            continue
        if (dash_new := _DASH_NEW.match(line)) is not None:
            dashboard["NEW"] = tuple(int(g) for g in dash_new.groups())
            continue
        if section is None:
            continue
        if (bullet := _BULLET.match(line)) is not None:
            counted[section][bullet.group(1)] += 1
        elif (row := _TABLE_ROW.match(line)) is not None:
            counted[section][row.group(2)] += 1
    return counted, dashboard


def test_moi_muc_deu_co_mot_dong_dashboard() -> None:
    counted, dashboard = _parse()
    assert set(counted) == set(dashboard), (
        f"mục có task nhưng không có dòng dashboard (hoặc ngược lại): "
        f"{set(counted) ^ set(dashboard)}"
    )


def test_dashboard_khop_voi_so_dem_that() -> None:
    """Đây là phép kiểm mà ba dòng cảnh báo ở §1 đã thay thế không thành công."""
    counted, dashboard = _parse()
    lech: list[str] = []
    for section, row in sorted(dashboard.items()):
        actual = counted[section]
        total, *by_status = row
        if total != sum(actual.values()):
            lech.append(f"{section}: cột Tổng ghi {total}, đếm được {sum(actual.values())}")
        for value, (name, symbol) in zip(by_status, COLUMNS.items(), strict=True):
            if value != actual[symbol]:
                lech.append(
                    f"{section}: cột {name} ghi {value}, đếm được {actual[symbol]} "
                    f"(ký hiệu `[{symbol}]`)"
                )
    assert not lech, "Dashboard §1 lệch với §2–§9:\n  " + "\n  ".join(lech)


def test_khong_task_nao_mang_ky_hieu_la() -> None:
    """Một ký hiệu mới (`[>]`, `[/]`…) rơi ra ngoài mọi cột thì hàng ngang không
    còn cộng lại bằng cột Tổng — đúng cách mà `[-]` của `W0-05` đã sống sót im
    lặng cho tới 2026-09-04."""
    counted, _ = _parse()
    known = set(COLUMNS.values())
    la = {
        f"{section}:[{symbol}]"
        for section, statuses in counted.items()
        for symbol in statuses
        if symbol not in known
    }
    assert not la, f"ký hiệu trạng thái không nằm trong bảng quy ước §0: {sorted(la)}"


def test_hai_dong_tong_cong_lai_dung() -> None:
    """Lỗi sổ sách đầu tiên của §1 là một dòng tổng không cộng lại — nó sống được
    vì không ai cộng lại bằng tay lần thứ hai."""
    counted, _ = _parse()
    totals: dict[str, tuple[int, ...]] = {}
    for line in CHECKLIST.read_text(encoding="utf-8").splitlines():
        if (m := _DASH_TOTAL.match(line)) is not None:
            totals[m.group(1)] = tuple(int(g) for g in m.groups()[1:])
    assert set(totals) == {"backlog gốc", "cộng"}, f"thiếu dòng tổng: {sorted(totals)}"

    goc: Counter[str] = Counter()
    for section, statuses in counted.items():
        if section != "NEW":
            goc.update(statuses)
    tat_ca: Counter[str] = Counter()
    for statuses in counted.values():
        tat_ca.update(statuses)

    for name, actual in (("backlog gốc", goc), ("cộng", tat_ca)):
        row = totals[name]
        assert row[0] == sum(actual.values()), (
            f"dòng '{name}': cột Tổng ghi {row[0]}, đếm được {sum(actual.values())}"
        )
        assert sum(row[1:]) == row[0], (
            f"dòng '{name}': các cột trạng thái cộng lại {sum(row[1:])} ≠ Tổng {row[0]}"
        )
        for value, symbol in zip(row[1:], COLUMNS.values(), strict=True):
            assert value == actual[symbol], (
                f"dòng '{name}', ký hiệu `[{symbol}]`: ghi {value}, đếm được {actual[symbol]}"
            )


def test_khong_co_ma_task_trung() -> None:
    """Hai dòng cùng `W3-09` thì mọi phép đếm phía sau đều sai mà không ai thấy."""
    seen: list[str] = []
    section: str | None = None
    for line in CHECKLIST.read_text(encoding="utf-8").splitlines():
        header = _SECTION.match(line)
        if header:
            section = header.group(1) or "NEW"
            continue
        if section is None:
            continue
        if (bullet := _BULLET.match(line)) is not None:
            seen.append(bullet.group(2))
        elif (row := _TABLE_ROW.match(line)) is not None:
            seen.append(row.group(1))
    trung = [code for code, n in Counter(seen).items() if n > 1]
    assert not trung, f"mã task trùng trong CHECKLIST.md: {sorted(trung)}"
