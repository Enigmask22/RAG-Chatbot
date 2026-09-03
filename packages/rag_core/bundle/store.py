"""Đọc/ghi `RagBundle` trên đĩa. Ba luật, và cả ba đều là luật *từ chối*.

1. **Không ghi đè một version đã tồn tại.** "Bundle bất biến" là một câu về hệ
   thống file, không phải về schema. Nếu `v1.4.0` ghi đè được thì câu "số đo này
   thuộc về `v1.4.0`" mất nghĩa, và gate ở `W5` gác một cái tên chứ không gác
   một artifact.
2. **Không nạp bundle chưa ký hoặc sai chữ ký**, trừ khi người gọi nêu tường
   minh là muốn bỏ qua. Mặc định phải là chặt, vì đường mặc định là đường mà
   serving đi.
3. **Không nạp bundle mà tên thư mục khác `bundle_version` bên trong.** Đây là
   cách một bản rollback đi nhầm chỗ: thư mục nói `v1.3.2`, manifest nói
   `v1.4.0`, checksum khớp hoàn toàn — vì checksum bảo vệ nội dung, không bảo vệ
   chỗ đặt.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import BundleValidationError, RagBundle, parse_semver

__all__ = [
    "BUNDLE_DIR_PREFIX",
    "MANIFEST_NAME",
    "bundle_dir_name",
    "latest_bundle",
    "list_bundles",
    "load_bundle",
    "save_bundle",
]

MANIFEST_NAME = "manifest.json"
BUNDLE_DIR_PREFIX = "rag-bundle-v"


def bundle_dir_name(version: str) -> str:
    parse_semver(version)  # từ chối sớm: tên thư mục sai không sửa được sau khi ghi
    return f"{BUNDLE_DIR_PREFIX}{version}"


def _version_from_dir(path: Path) -> str | None:
    if not path.name.startswith(BUNDLE_DIR_PREFIX):
        return None
    return path.name[len(BUNDLE_DIR_PREFIX) :]


def save_bundle(bundle: RagBundle, root: Path, *, overwrite: bool = False) -> Path:
    """Ký rồi ghi vào `root/rag-bundle-v<version>/manifest.json`.

    Ký ở đây chứ không bắt người gọi tự ký: một đường ghi mà quên ký sinh ra
    bundle không nạp được, và lỗi ấy chỉ lộ ra ở phía đọc, thường là trên máy
    khác, thường là lúc deploy.

    `overwrite` có mặt cho test và cho việc sinh lại bundle mẫu; nó **không** có
    cờ dòng lệnh tương ứng, để việc ghi đè luôn là một câu viết trong mã chứ
    không phải một phím gõ nhầm.
    """
    directory = root / bundle_dir_name(bundle.bundle_version)
    manifest = directory / MANIFEST_NAME
    if manifest.exists() and not overwrite:
        raise BundleValidationError(
            f"bundle {bundle.bundle_version} đã tồn tại: {manifest}. "
            "Bundle là bất biến — tăng version thay vì ghi đè, nếu không thì "
            "mọi số đo đã công bố cho version này không còn trỏ vào đâu cả."
        )
    directory.mkdir(parents=True, exist_ok=True)
    signed = bundle.signed()
    manifest.write_text(
        json.dumps(json.loads(signed.model_dump_json()), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_bundle(path: Path, *, verify: bool = True) -> RagBundle:
    """Nạp từ file manifest hoặc từ thư mục bundle.

    `verify=False` tồn tại cho đúng một việc: chẩn đoán một bundle đã hỏng
    (`đọc được nhưng sai chữ ký` khác `không đọc nổi`). Không dùng nó ở đường
    serving.
    """
    manifest = path / MANIFEST_NAME if path.is_dir() else path
    if not manifest.is_file():
        raise FileNotFoundError(f"không thấy manifest bundle: {manifest}")

    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleValidationError(f"manifest không phải JSON hợp lệ: {manifest} ({exc})") from exc

    bundle = RagBundle.model_validate(raw)

    declared = _version_from_dir(manifest.parent)
    if declared is not None and declared != bundle.bundle_version:
        raise BundleValidationError(
            f"thư mục nói version {declared!r} nhưng manifest nói "
            f"{bundle.bundle_version!r}: {manifest}. Checksum bảo vệ *nội dung*, "
            "không bảo vệ *chỗ đặt* — nên phép kiểm này phải nằm ngoài checksum."
        )

    if verify:
        # ⭐ Truyền `raw` chứ không để nó băm lại model đã validate: đó là toàn bộ
        # cách giải `TD-36`. Xem docstring của `RagBundle.verify_checksum`.
        bundle.verify_checksum(raw)
    return bundle


def list_bundles(root: Path, *, verify: bool = True) -> list[RagBundle]:
    """Mọi bundle trong `root`, **sắp theo thứ tự semver** chứ không theo tên file.

    Sắp theo tên thì `v1.10.0` đứng trước `v1.9.0`, và "bản trước đó" — thứ mà
    rollback cần — trỏ vào nhầm bundle. Lỗi này chỉ xuất hiện ở lần release thứ
    mười, tức lâu sau khi mọi test thủ công đã thôi được chạy.
    """
    if not root.is_dir():
        return []
    found = [
        load_bundle(child, verify=verify)
        for child in sorted(root.iterdir())
        if child.is_dir() and (child / MANIFEST_NAME).is_file()
    ]
    return sorted(found, key=lambda item: item.version_key)


def latest_bundle(root: Path, *, verify: bool = True) -> RagBundle | None:
    found = list_bundles(root, verify=verify)
    return found[-1] if found else None
