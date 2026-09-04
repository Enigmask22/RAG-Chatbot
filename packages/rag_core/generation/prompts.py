"""Prompt registry — `W4-11`: một prompt không đánh số là một biến số không đo được.

## Vì sao "đổi prompt = tăng version" phải là một CƠ CHẾ

Trước hạng mục này dự án có ba prompt hằng số trong mã (`SYSTEM_PROMPT`,
`NO_RETRIEVAL_SYSTEM_PROMPT`, `REWRITE_SYSTEM_PROMPT`). Đổi một chữ trong đó là
đổi hệ thống đang được đo — mọi con số eval trước không còn so được với số sau —
và không có gì trong log nói ra điều đó đã xảy ra. Một quy ước "nhớ tăng version
khi sửa" không chữa được, vì kiểu hỏng ở đây chính là *quên*.

Nên registry này làm cho việc quên **không nạp được**:

* `sha256` trong file là hash của `template` — loader tính lại và **từ chối**
  khi lệch. Sửa nội dung mà không đi qua `stamp()` thì server không lên.
* `version` phải lớn hơn **mọi** version trong `history`. Dùng lại một số
  version cho nội dung khác là điều `stamp()` không bao giờ sinh ra và loader
  không bao giờ nhận.
* Đường sửa hợp lệ duy nhất: sửa `template` → chạy `scripts/prompt_stamp.py` —
  nó đẩy cặp `(version, sha256)` cũ vào `history`, tăng version, ghi hash mới.
  Con người không gõ tay version hay hash ở bất kỳ bước nào.

## Cái gì KHÔNG nằm ở đây

* `LANGUAGE_DIRECTIVE` — mảnh ghép theo từng lượt; nhét vào prompt có version
  sẽ chẻ hash thành một bản mỗi ngôn ngữ (xem `QueryPlan.directive`).
* `CLARIFY_TEXT` — text trả cho người dùng do mã chọn, không phải prompt cho model.
* `CONTEXT_SYSTEM_PROMPT`/`BATCH_SYSTEM_PROMPT` của contextual chunking —
  chúng đã có cơ chế riêng: nội dung system thật được băm vào fingerprint cache
  (`W3-03`), và $5,47 ngữ cảnh đã sinh gắn với fingerprint ấy. Migrate chúng
  không mua được gì hôm nay ngoài rủi ro lệch fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from rag_core.bundle.schema import PromptComponent

__all__ = [
    "Prompt",
    "PromptIntegrityError",
    "PromptNotFoundError",
    "PromptRegistry",
    "default_registry",
    "sha256_of",
    "stamp",
]

DEFAULT_PROMPT_DIR = Path(__file__).parent / "prompts"
"""Prompt đi theo package chứ không theo repo root: Docker của `W4-13` copy
`packages/` là có luôn, không cần một volume hay một biến môi trường thứ N."""


def sha256_of(template: str) -> str:
    """Hash của **nội dung template**, không phải của file.

    Băm file thì đổi một dòng `description` cũng vỡ hash — tức hash thôi nói về
    thứ được đo (nội dung đưa cho model) mà nói về cả sổ sách quanh nó, và người
    ta sẽ tắt phép kiểm trong tuần đầu. Cùng lý lẽ với checksum của bundle
    (`W4-01`: băm model đã validate, không băm byte).
    """
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


class PromptNotFoundError(LookupError):
    """Không có file prompt ấy trong thư mục registry."""


class PromptIntegrityError(ValueError):
    """File prompt tự mâu thuẫn — nội dung đổi mà version/hash chưa đi qua `stamp`."""


class _HistoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = Field(ge=1)
    sha256: str = Field(min_length=64, max_length=64)


class _PromptFile(BaseModel):
    """Schema của một file YAML. `extra="forbid"`: một trường gõ nhầm tên
    (`verison`) phải là lỗi nạp, không phải một trường bị lặng lẽ bỏ qua."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    version: int = Field(ge=1)
    description: str = ""
    sha256: str = Field(min_length=64, max_length=64)
    history: list[_HistoryEntry] = Field(default_factory=list)
    template: str = Field(min_length=1)


@dataclass(frozen=True)
class Prompt:
    """Một prompt đã nạp và đã kiểm — bất biến, như mọi thứ khác trên đường eval."""

    id: str
    version: int
    sha256: str
    text: str

    @property
    def spec(self) -> str:
        """`chat-system@v3` — chuỗi đi vào log và khung `meta` của mỗi lượt."""
        return f"{self.id}@v{self.version}"

    def component(self) -> PromptComponent:
        """Cầu sang `RagBundle`: đúng cái ô `prompt.hash` mà schema đã chừa sẵn
        từ `W4-01` thay vì nhét `"todo"`."""
        return PromptComponent(id=self.id, version=self.version, hash=f"sha256:{self.sha256}")


def _parse(path: Path) -> _PromptFile:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise PromptIntegrityError(f"{path.name}: không phải một mapping YAML")
    return _PromptFile.model_validate(raw)


def _verify(spec: _PromptFile, path: Path) -> Prompt:
    """Mọi cách một file prompt có thể nói dối, mỗi cách một thông điệp chỉ đường."""
    if spec.id != path.stem:
        raise PromptIntegrityError(
            f"{path.name}: khai `id: {spec.id}` nhưng tên file là {path.stem!r} — "
            "id là địa chỉ tra cứu, hai thứ lệch nhau thì `get()` trả về prompt khác tên"
        )
    actual = sha256_of(spec.template)
    if actual != spec.sha256:
        raise PromptIntegrityError(
            f"{path.name}: nội dung template (sha256 {actual[:12]}…) không khớp hash đã khai "
            f"({spec.sha256[:12]}…) — template đã bị sửa mà chưa qua stamp. "
            "Chạy `uv run python scripts/prompt_stamp.py` để tăng version và ghi hash mới."
        )
    for entry in spec.history:
        if entry.version >= spec.version:
            raise PromptIntegrityError(
                f"{path.name}: history chứa version {entry.version} >= version hiện tại "
                f"{spec.version} — version phải tăng đơn điệu, một số dùng lại là hai nội "
                "dung khác nhau mang cùng một tên"
            )
    seen_versions = [entry.version for entry in spec.history]
    if len(seen_versions) != len(set(seen_versions)):
        raise PromptIntegrityError(f"{path.name}: history có version trùng nhau")
    return Prompt(id=spec.id, version=spec.version, sha256=spec.sha256, text=spec.template)


class PromptRegistry:
    """Nạp + kiểm + cache. Một instance cho một thư mục; `default_registry()`
    là instance chung cho thư mục mặc định."""

    def __init__(self, root: Path | str = DEFAULT_PROMPT_DIR) -> None:
        self.root = Path(root)
        self._cache: dict[str, Prompt] = {}

    def get(self, prompt_id: str) -> Prompt:
        if prompt_id in self._cache:
            return self._cache[prompt_id]
        path = self.root / f"{prompt_id}.yaml"
        if not path.is_file():
            raise PromptNotFoundError(f"không có prompt {prompt_id!r} trong {self.root}")
        prompt = _verify(_parse(path), path)
        self._cache[prompt_id] = prompt
        return prompt

    def all(self) -> list[Prompt]:
        """Mọi prompt trong thư mục, sắp theo id — cho dòng log lúc khởi động."""
        return [self.get(path.stem) for path in sorted(self.root.glob("*.yaml"))]


@lru_cache(maxsize=1)
def default_registry() -> PromptRegistry:
    return PromptRegistry()


def stamp(path: Path) -> Prompt:
    """Đường sửa prompt hợp lệ duy nhất: đọc file, nếu template đã đổi so với
    hash khai thì **tự** đẩy `(version, sha256)` cũ vào history, tăng version,
    ghi hash mới. Không đổi gì thì không chạm file.

    Trả về `Prompt` sau khi stamp (đã qua đúng bộ kiểm của loader — stamp mà
    sinh ra một file loader từ chối là một bug của stamp, và test ghim điều đó).
    """
    spec = _parse(path)
    actual = sha256_of(spec.template)
    if actual == spec.sha256:
        return _verify(spec, path)

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["history"] = [*data.get("history", []), {"version": spec.version, "sha256": spec.sha256}]
    data["version"] = spec.version + 1
    data["sha256"] = actual
    path.write_text(_render(data), encoding="utf-8")
    return _verify(_parse(path), path)


def _render(data: dict[str, Any]) -> str:
    """Ghi lại file theo đúng một khuôn — template là block scalar `|-` để diff
    đọc được như văn bản, không phải một chuỗi JSON-escape dài một dòng."""
    lines = [
        f"id: {data['id']}",
        f"version: {data['version']}",
    ]
    description = str(data.get("description", "")).strip()
    if description:
        lines.append(f"description: {_quote(description)}")
    # Hash luôn trong ngoặc kép: một hex toàn chữ số ("000…") sẽ bị YAML đọc
    # thành int, và "0e5" hợp lệ của hex là một float hợp lệ của YAML 1.1.
    lines.append(f'sha256: "{data["sha256"]}"')
    history = data.get("history", [])
    if history:
        lines.append("history:")
        for entry in history:
            lines.append(f"  - version: {entry['version']}")
            lines.append(f'    sha256: "{entry["sha256"]}"')
    else:
        lines.append("history: []")
    lines.append("template: |-")
    for template_line in str(data["template"]).splitlines():
        lines.append(f"  {template_line}" if template_line else "")
    return "\n".join(lines) + "\n"


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)
