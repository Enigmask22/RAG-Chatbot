"""Các target trong `Makefile` phải chạy được, không chỉ tồn tại.

Nhóm test này tồn tại vì cả hai target `ctx-run-glm`/`ctx-run-deepseek` hỏng từ
lúc được tạo: chúng chứa `\n` **theo nghĩa đen** thay vì nối dòng, nên shell
nhận `n` làm một tham số rời. Không ai phát hiện suốt một buổi vì mọi phép đo
đều gọi thẳng CLI — tức lệnh được ghi trong `RUNPOD.md` như đường chính thức
chưa từng được chạy lần nào.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

MAKEFILE = Path(__file__).resolve().parents[2] / "Makefile"
BACKSLASH = chr(92)


def _recipe_lines() -> list[tuple[int, str]]:
    """Các dòng công thức (bắt đầu bằng TAB), kèm số dòng."""
    text = MAKEFILE.read_text(encoding="utf-8")
    return [(i, ln) for i, ln in enumerate(text.split("\n"), 1) if ln.startswith("\t")]


def test_makefile_exists() -> None:
    assert MAKEFILE.is_file()


def test_no_literal_backslash_n_in_recipes() -> None:
    """`\n` giữa công thức là dấu hiệu heredoc đã ăn mất ký tự nối dòng.

    Ngoại lệ duy nhất hợp lệ là chuỗi định dạng của `printf`/`awk`, nơi `\n`
    được truyền cho chương trình con một cách có chủ ý.
    """
    bad = [
        (i, ln)
        for i, ln in _recipe_lines()
        if BACKSLASH + "n" in ln and not re.search(r"(printf|awk|sed)", ln)
    ]
    assert not bad, f"công thức chứa {BACKSLASH}n theo nghĩa đen: {bad}"


@pytest.mark.parametrize(
    "target",
    ["ctx-dry", "ctx-prepare", "ctx-run-glm", "ctx-run-deepseek", "job-bundle", "job-verify"],
)
def test_w3_04_targets_are_declared(target: str) -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    assert re.search(rf"^{re.escape(target)}:", text, re.M), f"thiếu target {target}"


@pytest.mark.parametrize("target", ["ctx-run-glm", "ctx-run-deepseek"])
def test_paid_targets_declare_a_cost_cap(target: str) -> None:
    """Một target tiêu tiền mà không có trần là một target chạy tới hết số dư."""
    text = MAKEFILE.read_text(encoding="utf-8")
    body = text.split(f"\n{target}:", 1)[1].split("\n.PHONY", 1)[0]
    assert "--cost-cap" in body
