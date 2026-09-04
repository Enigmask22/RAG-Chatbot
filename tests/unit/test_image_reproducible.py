"""Image serving phải cài từ `uv.lock`, không giải lại phụ thuộc. `W5-01`.

## Vì sao file này tồn tại

`W4-13` viết Dockerfile với `uv pip install -r pyproject.toml` và kèm chú thích:
*"`uv` thay pip: cùng resolver với môi trường dev, nên image **không thể lệch
phiên bản so với `uv.lock`** mà không ai thấy."*

Chú thích ấy sai. `-r pyproject.toml` **giải lại** từ đầu mỗi lần build; `uv.lock`
được `COPY` vào chỉ để làm khoá cache lớp. Bốn lần rebuild trong vài giờ đủ để
image trôi:

    lock / dev  : transformers 5.15.0 · sentence-transformers 5.7.0 · torch 2.13.0
    image       : transformers 5.16.1 · sentence-transformers 6.0.1  · torch 2.14.0

Hậu quả: cross-encoder nạp ra dtype trộn và **mọi** request có truy hồi trả 503.
`GET /admin/bundle` vẫn báo `runtime_drift: null` suốt lúc đó — phép kiểm danh
tính `TD-38` soi model/device/dtype/max_length, không soi phiên bản thư viện.

Một chú thích không kiểm được là một lời hứa. Đây là phép kiểm.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = _ROOT / "serving" / "Dockerfile"
LOCKFILE = _ROOT / "uv.lock"

#: Ba gói mà phiên bản của chúng **đổi được con số eval**, không chỉ đổi API.
#: `torch`/`transformers`/`sentence-transformers` quyết định cách trọng số được
#: nạp (dtype), cách tokenizer cắt, và cách cross-encoder chấm.
VERSION_CRITICAL = ("torch", "transformers", "sentence-transformers")


def locked_versions() -> dict[str, set[str]]:
    """Phiên bản `uv.lock` ghi cho từng gói. Một gói có thể có nhiều bản
    (ví dụ `torch` 2.13.0 từ PyPI và 2.13.0+cu126 từ index PyTorch)."""
    data = tomllib.loads(LOCKFILE.read_text(encoding="utf-8"))
    out: dict[str, set[str]] = {}
    for package in data.get("package", []):
        out.setdefault(package["name"], set()).add(package["version"])
    return out


def test_dockerfile_cai_tu_lockfile() -> None:
    commands = "\n".join(
        line
        for line in DOCKERFILE.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "uv sync" in commands, "Dockerfile phải cài bằng `uv sync`, không phải `uv pip install`"
    assert "--locked" in commands, (
        "thiếu `--locked`: `uv sync` không có cờ này vẫn có thể cập nhật lockfile "
        "trong lúc build, tức build lại tự sửa cái nó lẽ ra phải tuân theo"
    )


def test_dockerfile_khong_giai_lai_tu_pyproject() -> None:
    """Đây là chính xác dòng đã làm image trôi, ghim lại để nó không quay về."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    install_lines = [
        line
        for line in text.splitlines()
        if "-r pyproject.toml" in line and not line.lstrip().startswith("#")
    ]
    assert not install_lines, (
        "`-r pyproject.toml` giải lại phụ thuộc mỗi lần build — image sẽ trôi khỏi "
        f"môi trường đã đo mà không có gì báo: {install_lines}"
    )


def test_lockfile_duoc_copy_truoc_khi_cai() -> None:
    """`uv sync --locked` mà không có `uv.lock` trong context thì build đỏ —
    nhưng đỏ ở chỗ khó đọc. Ghim thứ tự để lỗi hiện ra ở đây trước."""
    # Chỉ xét dòng lệnh, bỏ chú thích: chính docstring của Dockerfile nhắc tới
    # cả `uv.lock` lẫn `uv sync`, và một phép kiểm đọc trúng chú thích thì nó
    # đang kiểm văn bản chứ không kiểm build.
    lines = [
        line
        for line in DOCKERFILE.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]
    copy_at = next((i for i, line in enumerate(lines) if "uv.lock" in line), None)
    sync_at = next((i for i, line in enumerate(lines) if "uv sync" in line), None)
    assert copy_at is not None, "Dockerfile không COPY `uv.lock`"
    assert sync_at is not None, "Dockerfile không gọi `uv sync`"
    assert copy_at < sync_at, "`uv.lock` phải được COPY trước bước `uv sync`"


@pytest.mark.parametrize("package", VERSION_CRITICAL)
def test_goi_quyet_dinh_con_so_deu_co_trong_lock(package: str) -> None:
    """Nếu một trong ba gói này biến khỏi lock thì `--locked` không còn ghim gì
    cho nó, và bảng số đo mất neo mà không ai thấy."""
    versions = locked_versions()
    assert package in versions, f"`{package}` không có trong uv.lock"
    assert versions[package], f"`{package}` trong uv.lock không có version"


def test_moi_gioi_han_tren_deu_vang_mat_la_co_y() -> None:
    """`sentence-transformers>=3.0` không có trần trên — và đó chính là cửa mà
    bản 6.0.1 đã đi qua.

    Không siết trần ở `pyproject.toml`: trần trên chặn nâng cấp và tạo ra một
    loại nợ khác. Cái phải đúng là **image dùng lock**, và đó là việc của các
    test phía trên. Test này chỉ ghim rằng lựa chọn ấy có ý thức.
    """
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    unbounded = [
        spec
        for group in extras.values()
        for spec in group
        if any(spec.startswith(name) for name in VERSION_CRITICAL) and "<" not in spec
    ]
    assert unbounded, (
        "Có ai đó đã thêm trần trên cho gói quyết định con số. Không sai, nhưng "
        "hãy sửa docstring này: lý lẽ 'image dùng lock nên không cần trần' đã đổi."
    )
