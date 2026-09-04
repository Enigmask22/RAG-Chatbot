"""Stamp prompt sau khi sửa template — `W4-11`.

    uv run python scripts/prompt_stamp.py            # stamp mọi prompt trong registry
    uv run python scripts/prompt_stamp.py <file.yaml> [...]

Đây là đường sửa prompt hợp lệ DUY NHẤT: sửa `template` trong YAML rồi chạy
lệnh này. Nó đẩy cặp (version, sha256) cũ vào `history`, tăng version, ghi hash
mới. Sửa tay `version`/`sha256` không có lý do gì cả — loader từ chối mọi tổ
hợp mà lệnh này không sinh ra được.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "packages"))

from rag_core.generation.prompts import (
    DEFAULT_PROMPT_DIR,
    _parse,
    sha256_of,
    stamp,
)


def main(argv: list[str]) -> int:
    paths = [pathlib.Path(a) for a in argv] or sorted(DEFAULT_PROMPT_DIR.glob("*.yaml"))
    if not paths:
        print(f"không có file YAML nào trong {DEFAULT_PROMPT_DIR}")
        return 1
    for path in paths:
        before = _parse(path)
        changed = sha256_of(before.template) != before.sha256
        after = stamp(path)
        if changed:
            print(f"{path.name}: v{before.version} -> {after.spec} (sha256 {after.sha256[:12]}…)")
        else:
            print(f"{path.name}: {after.spec} — không đổi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
