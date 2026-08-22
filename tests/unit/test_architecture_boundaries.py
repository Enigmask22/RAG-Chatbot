"""Canh chiều phụ thuộc giữa hai plane.

Cả kiến trúc tồn tại là để **Pipeline Plane tách rời Serving Plane**: pipeline
chạy trên GPU thuê để thử nghiệm, serving chạy production, hai bên chỉ nối nhau
qua artifact `RagBundle` có version. Một dòng `from pipeline...` lọt vào
`serving/` là đủ để ranh giới đó biến mất — và nó sẽ lọt vào rất tự nhiên, vì
code cần dùng lại thì nằm sẵn ở đó.

Test này rẻ và bắt được đúng lúc lỗi mới xảy ra, thay vì lúc đã phải gỡ rối
nguyên một tuần công việc.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _imported_roots(package_dir: Path) -> dict[str, list[str]]:
    """Map tên gói gốc được import → danh sách vị trí import."""
    found: dict[str, list[str]] = {}
    for source_file in sorted(package_dir.rglob("*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                # `level > 0` là import tương đối, luôn nằm trong chính gói đó.
                if node.level and node.level > 0:
                    continue
                names = [node.module or ""]
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                rel = source_file.relative_to(_REPO_ROOT).as_posix()
                found.setdefault(root, []).append(f"{rel}:{node.lineno}")
    return found


@pytest.mark.parametrize("forbidden", ["pipeline", "serving"])
def test_rag_core_does_not_import_planes(forbidden: str) -> None:
    imports = _imported_roots(_REPO_ROOT / "packages" / "rag_core")
    assert forbidden not in imports, (
        f"rag_core import `{forbidden}` tại {imports.get(forbidden)} — "
        "thư viện dùng chung không được phụ thuộc ngược lên plane"
    )


def test_serving_does_not_import_pipeline() -> None:
    imports = _imported_roots(_REPO_ROOT / "serving")
    assert "pipeline" not in imports, (
        f"serving import `pipeline` tại {imports.get('pipeline')} — "
        "hai plane chỉ được nối nhau qua artifact RagBundle"
    )


def test_pipeline_does_not_import_serving() -> None:
    imports = _imported_roots(_REPO_ROOT / "pipeline")
    assert "serving" not in imports, (
        f"pipeline import `serving` tại {imports.get('serving')} — "
        "pipeline phải chạy được độc lập trên máy GPU thuê, nơi không có serving stack"
    )


def test_rag_core_stays_dependency_light() -> None:
    """`rag_core` chỉ được phụ thuộc thư viện nhẹ ở tầng module.

    Torch và qdrant-client đều được import **lazy** bên trong hàm. Kéo chúng lên
    đầu file làm `make test` chậm đi hàng chục giây và buộc CI phải cài GPU stack
    chỉ để chạy unit test.
    """
    heavy = {"torch", "sentence_transformers", "qdrant_client", "transformers", "docling"}
    imports = _imported_roots(_REPO_ROOT / "packages" / "rag_core")

    # Import trong `if TYPE_CHECKING:` không tính — chúng không chạy lúc runtime.
    offenders: dict[str, list[str]] = {}
    for name in heavy & set(imports):
        locations = [
            loc
            for loc in imports[name]
            if not _is_type_checking_or_local(
                Path(_REPO_ROOT / loc.rsplit(":", 1)[0]), int(loc.rsplit(":", 1)[1])
            )
        ]
        if locations:
            offenders[name] = locations

    assert not offenders, f"phụ thuộc nặng phải import lazy: {offenders}"


def _is_type_checking_or_local(source_file: Path, lineno: int) -> bool:
    """Đúng nếu dòng import nằm trong `if TYPE_CHECKING:` hoặc trong thân hàm."""
    tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            is_tc = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
            )
            if is_tc and any(getattr(child, "lineno", None) == lineno for child in ast.walk(node)):
                return True
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and any(
            isinstance(child, ast.Import | ast.ImportFrom) and child.lineno == lineno
            for child in ast.walk(node)
        ):
            return True
    return False
