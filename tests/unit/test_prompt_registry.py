"""`W4-11` — prompt registry: "đổi prompt = tăng version" là cơ chế, không phải lời dặn.

Hai nửa: (1) loader từ chối mọi file mà `stamp()` không sinh ra được — sửa nội
dung không stamp thì server không lên; (2) ba prompt thật của serving nạp được,
đúng nội dung các phép đo `W4-07`/`W4-09` đã gắn vào.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_core.bundle.schema import PromptComponent
from rag_core.generation.prompts import (
    DEFAULT_PROMPT_DIR,
    Prompt,
    PromptIntegrityError,
    PromptNotFoundError,
    PromptRegistry,
    _render,
    default_registry,
    sha256_of,
    stamp,
)

# ---------------------------------------------------------------------------
# Dựng file
# ---------------------------------------------------------------------------

TEMPLATE = "Bạn là trợ lý.\n\nQuy tắc:\n1. Trả lời ngắn gọn.\n2. Không suy đoán."


def _write(
    root: Path, prompt_id: str = "demo", template: str = TEMPLATE, **overrides: object
) -> Path:
    data: dict[str, object] = {
        "id": prompt_id,
        "version": 1,
        "description": "prompt thử",
        "sha256": sha256_of(template),
        "history": [],
        "template": template,
        **overrides,
    }
    path = root / f"{prompt_id}.yaml"
    path.write_text(_render(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. Nạp
# ---------------------------------------------------------------------------


class TestLoad:
    def test_the_hash_covers_every_byte_without_normalisation(self) -> None:
        """⭐ Sinh từ một phép tiêm SỐNG SÓT: `sha256_of` strip() nội dung trước
        khi băm không làm đỏ test nào — template thật không có whitespace ở
        biên (YAML `|-` còn cắt newline cuối), và mọi test khác so hash qua
        chính hàm bị tiêm nên tự nhất quán. Hợp đồng phải ghim trực tiếp:
        model nhìn thấy BYTE, hai template khác nhau một khoảng trắng biên là
        hai prompt khác nhau, hash không được phép nói chúng là một."""
        assert sha256_of("a") != sha256_of(" a ")
        assert sha256_of("a") != sha256_of("a\n")
        assert sha256_of("a") != sha256_of("\na")

    def test_a_stamped_file_loads_with_its_exact_text(self, tmp_path: Path) -> None:
        """Round-trip qua `_render` phải giữ nguyên byte — template có dòng
        trống ở giữa, đúng hình dạng của prompt thật."""
        _write(tmp_path)
        prompt = PromptRegistry(tmp_path).get("demo")

        assert prompt.text == TEMPLATE
        assert prompt.version == 1
        assert prompt.spec == "demo@v1"
        assert prompt.sha256 == sha256_of(TEMPLATE)

    def test_the_registry_caches_by_id(self, tmp_path: Path) -> None:
        _write(tmp_path)
        registry = PromptRegistry(tmp_path)
        assert registry.get("demo") is registry.get("demo")

    def test_a_missing_prompt_names_the_directory(self, tmp_path: Path) -> None:
        with pytest.raises(PromptNotFoundError, match="khong-co"):
            PromptRegistry(tmp_path).get("khong-co")

    def test_component_bridges_to_the_bundle_schema(self, tmp_path: Path) -> None:
        """Đúng cái ô `prompt.hash` mà `W4-01` chừa sẵn thay vì `"todo"`."""
        _write(tmp_path)
        component = PromptRegistry(tmp_path).get("demo").component()

        assert isinstance(component, PromptComponent)
        assert component.version == 1
        assert component.hash == f"sha256:{sha256_of(TEMPLATE)}"


# ---------------------------------------------------------------------------
# 2. Loader từ chối — mỗi cách một file có thể nói dối
# ---------------------------------------------------------------------------


class TestRefusals:
    def test_content_edited_without_stamp_does_not_load(self, tmp_path: Path) -> None:
        """Cơ chế trung tâm của cả hạng mục: sửa template mà quên stamp thì
        KHÔNG nạp được — và thông điệp chỉ đúng đường sửa."""
        path = _write(tmp_path)
        path.write_text(
            path.read_text(encoding="utf-8").replace("Không suy đoán", "Được suy đoán"),
            encoding="utf-8",
        )

        with pytest.raises(PromptIntegrityError, match="prompt_stamp"):
            PromptRegistry(tmp_path).get("demo")

    def test_an_id_that_disagrees_with_the_filename_is_refused(self, tmp_path: Path) -> None:
        path = _write(tmp_path, prompt_id="demo")
        path.rename(tmp_path / "khac.yaml")

        with pytest.raises(PromptIntegrityError, match="id"):
            PromptRegistry(tmp_path).get("khac")

    def test_a_history_version_at_or_above_current_is_refused(self, tmp_path: Path) -> None:
        """Version phải tăng đơn điệu — một số dùng lại là hai nội dung khác
        nhau mang cùng một tên."""
        _write(
            tmp_path,
            version=2,
            history=[{"version": 2, "sha256": "0" * 64}],
        )
        with pytest.raises(PromptIntegrityError, match="đơn điệu"):
            PromptRegistry(tmp_path).get("demo")

    def test_duplicate_history_versions_are_refused(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            version=3,
            history=[{"version": 1, "sha256": "0" * 64}, {"version": 1, "sha256": "1" * 64}],
        )
        with pytest.raises(PromptIntegrityError, match="trùng"):
            PromptRegistry(tmp_path).get("demo")

    def test_a_misspelled_field_is_an_error_not_a_silent_skip(self, tmp_path: Path) -> None:
        """`verison: 2` bị lặng lẽ bỏ qua nghĩa là người sửa tin rằng mình đã
        tăng version trong khi không có gì đổi — `extra="forbid"` chặn từ gốc."""
        path = _write(tmp_path)
        path.write_text(
            path.read_text(encoding="utf-8") + "verison: 2\n",
            encoding="utf-8",
        )
        with pytest.raises(ValidationError, match="verison"):
            PromptRegistry(tmp_path).get("demo")

    def test_an_empty_template_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "demo.yaml"
        path.write_text(
            f"id: demo\nversion: 1\nsha256: {sha256_of('')}\nhistory: []\ntemplate: ''\n",
            encoding="utf-8",
        )
        with pytest.raises(ValidationError):
            PromptRegistry(tmp_path).get("demo")


# ---------------------------------------------------------------------------
# 3. stamp — đường sửa hợp lệ duy nhất
# ---------------------------------------------------------------------------


class TestStamp:
    def test_editing_content_bumps_the_version_and_keeps_history(self, tmp_path: Path) -> None:
        """DoD "hash đổi khi nội dung đổi" + "đổi prompt = tăng version", trong
        một chuyển động: stamp sau khi sửa cho ra v2 với hash mới, và cặp
        (v1, hash cũ) nằm lại trong history."""
        path = _write(tmp_path)
        old_hash = sha256_of(TEMPLATE)
        path.write_text(
            path.read_text(encoding="utf-8").replace("Không suy đoán", "Được suy đoán"),
            encoding="utf-8",
        )

        prompt = stamp(path)

        assert prompt.version == 2
        assert prompt.sha256 != old_hash
        assert "Được suy đoán" in prompt.text
        reloaded = PromptRegistry(tmp_path).get("demo")
        assert reloaded == prompt  # file sau stamp qua đúng bộ kiểm của loader
        history = path.read_text(encoding="utf-8")
        assert f'sha256: "{old_hash}"' in history  # bản cũ còn dấu vết

    def test_stamp_without_changes_touches_nothing(self, tmp_path: Path) -> None:
        path = _write(tmp_path)
        before = path.read_text(encoding="utf-8")

        prompt = stamp(path)

        assert prompt.version == 1
        assert path.read_text(encoding="utf-8") == before

    def test_two_edits_stack_two_history_entries(self, tmp_path: Path) -> None:
        path = _write(tmp_path)
        for marker in ("lần một", "lần hai"):
            content = path.read_text(encoding="utf-8")
            path.write_text(content.replace("trợ lý", f"trợ lý {marker}"), encoding="utf-8")
            stamp(path)

        prompt = PromptRegistry(tmp_path).get("demo")
        assert prompt.version == 3
        assert prompt.text.count("lần") == 2


# ---------------------------------------------------------------------------
# 4. Registry thật — ba prompt của serving
# ---------------------------------------------------------------------------


class TestShippedPrompts:
    def test_the_three_serving_prompts_load(self) -> None:
        specs = {p.id: p for p in default_registry().all()}
        assert set(specs) == {"chat-system", "chat-no-retrieval", "query-rewrite"}
        assert all(isinstance(p, Prompt) for p in specs.values())

    def test_every_shipped_file_is_consistent(self) -> None:
        """Chạy đúng bộ kiểm của loader trên từng file trong repo — một lần sửa
        tay lọt qua review sẽ đỏ ở ĐÂY chứ không phải lúc server không lên."""
        for path in sorted(DEFAULT_PROMPT_DIR.glob("*.yaml")):
            PromptRegistry(DEFAULT_PROMPT_DIR).get(path.stem)

    def test_chat_system_still_carries_the_citation_contract(self) -> None:
        """Nội dung migrate phải giữ nguyên hợp đồng `W4-09`: marker và luật
        ngôn ngữ nằm trong template, không thất lạc trên đường vào YAML."""
        prompt = default_registry().get("chat-system")
        assert "CITATIONS:" in prompt.text
        assert "ngôn ngữ" in prompt.text

    def test_default_registry_is_a_singleton(self) -> None:
        assert default_registry() is default_registry()

    def test_the_chat_system_hash_is_pinned_to_a_literal(self) -> None:
        """Ghim hash bằng LITERAL, không bằng `sha256_of` — mọi test khác so
        hash qua chính hàm ấy, nên một `sha256_of` bị đổi cách chuẩn hoá (vd
        strip() nội dung trước khi băm) tự nhất quán và không test nào thấy.
        Literal này là điểm neo ngoài hệ: đổi MỘT byte của template `W4-07`/
        `W4-09` đã đo, hoặc đổi cách băm, đều đỏ ở đây.

        Khi bump version có chủ đích: cập nhật literal này cùng lúc — đó chính
        là hành vi muốn có, người sửa phải nhìn thấy chỗ con số eval gãy."""
        prompt = default_registry().get("chat-system")
        assert prompt.sha256 == "ae5ea143004e8e6f56d5acfa899e80ff6a80a9df3b570f61c211b02ce51c7769"
        assert prompt.version == 2  # v1 -> v2 ở `W4-12` (ranh giới dữ liệu)
