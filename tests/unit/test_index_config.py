"""W1-08 — `IndexConfig`: fingerprint, factory, đọc YAML.

Trọng tâm là `fingerprint`. Nó là thứ duy nhất trả lời được "index trong Qdrant
có phải do config này sinh ra không", nên phải đúng theo cả hai chiều: đổi thứ
ảnh hưởng tới vector thì fingerprint phải đổi, đổi thứ chỉ ảnh hưởng tốc độ thì
phải giữ nguyên. Sai chiều nào cũng tệ — chiều một cho phép trộn hai cấu hình
vào một collection, chiều hai bắt build lại vài giờ GPU mỗi lần đổi máy.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from pipeline.indexing.config import IndexConfig, load_index_config
from rag_core.chunking import ChunkingConfig, ChunkingStrategy
from rag_core.chunking.cache import CachedChunker
from rag_core.embedding import HashingEmbeddingProvider

REPO_ROOT = Path(__file__).resolve().parents[2]


def _config(**kwargs: object) -> IndexConfig:
    base: dict[str, object] = {"name": "t", "embedding_model": "hashing:64"}
    base.update(kwargs)
    return IndexConfig.model_validate(base)


class TestFingerprint:
    def test_stable_across_identical_configs(self) -> None:
        assert _config().fingerprint == _config().fingerprint

    @pytest.mark.parametrize(
        "field,value",
        [
            ("embedding_model", "hashing:128"),
            ("embedding_normalize", False),
            ("languages", ("vi",)),
            ("doc_types", ("legal",)),
            ("max_documents", 5),
            ("embedding_kwargs", {"query_prefix": "query: "}),
        ],
    )
    def test_changes_when_content_affecting_field_changes(self, field: str, value: object) -> None:
        assert _config().fingerprint != _config(**{field: value}).fingerprint

    def test_changes_when_chunking_changes(self) -> None:
        other = ChunkingConfig(chunk_size=900)
        assert _config().fingerprint != _config(chunking=other).fingerprint

    @pytest.mark.parametrize(
        "field,value",
        [
            ("embedding_device", "cpu"),
            ("embedding_batch_size", 8),
            ("upsert_batch_size", 16),
            ("cache_path", Path("/tmp/other.sqlite3")),
            ("state_dir", Path("/tmp/state")),
            ("use_cache", False),
            ("collection", "khac"),
            ("manifest_path", Path("khac.csv")),
        ],
    )
    def test_unchanged_when_only_speed_or_location_changes(self, field: str, value: object) -> None:
        """Chạy trên laptop hay trên GPU thuê phải ra cùng một index về mặt logic.

        Nếu `device` vào fingerprint thì mỗi lần đổi máy là một lần build lại
        toàn bộ — đúng thứ kiến trúc hai plane sinh ra để tránh.
        """
        assert _config().fingerprint == _config(**{field: value}).fingerprint

    def test_neighbor_context_is_part_of_fingerprint(self) -> None:
        """Sai lệch nguy hiểm nhất so với POC phải được fingerprint bắt được."""
        poc = ChunkingConfig(neighbor_context_chars=100)
        assert _config().fingerprint != _config(chunking=poc).fingerprint


class TestDerived:
    def test_collection_defaults_to_name(self) -> None:
        assert _config(name="baseline").collection_name == "rag_baseline"

    def test_explicit_collection_wins(self) -> None:
        assert _config(collection="riêng").collection_name == "riêng"

    def test_state_path_is_per_run(self) -> None:
        assert _config(name="abl-01").state_path.name == "abl-01.json"

    def test_rejects_unknown_field(self) -> None:
        """`extra=forbid`: gõ sai tên trường trong YAML phải báo, không im lặng bỏ qua."""
        with pytest.raises(ValidationError):
            IndexConfig.model_validate({"name": "t", "embeding_model": "typo"})

    def test_rejects_bad_device(self) -> None:
        with pytest.raises(ValidationError, match="embedding_device"):
            _config(embedding_device="tpu")


class TestFactories:
    def test_builds_hashing_provider_without_torch(self) -> None:
        provider = _config(embedding_model="hashing:64").build_embeddings()
        assert isinstance(provider, HashingEmbeddingProvider)
        assert provider.dimension == 64

    def test_wraps_chunker_in_cache_when_enabled(self, tmp_path: Path) -> None:
        config = _config(use_cache=True, cache_path=tmp_path / "c.sqlite3")
        chunker = config.build_chunker(HashingEmbeddingProvider(64))
        assert isinstance(chunker, CachedChunker)

    def test_no_cache_wrapper_when_disabled(self, tmp_path: Path) -> None:
        config = _config(use_cache=False, cache_path=tmp_path / "c.sqlite3")
        assert not isinstance(config.build_chunker(HashingEmbeddingProvider(64)), CachedChunker)


class TestLoadYaml:
    def test_reads_nested_chunking(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "name": "x",
                    "embedding_model": "hashing:64",
                    "chunking": {"strategy": "fixed", "chunk_size": 800},
                }
            ),
            encoding="utf-8",
        )
        config = load_index_config(path)
        assert config.chunking.strategy is ChunkingStrategy.FIXED
        assert config.chunking.chunk_size == 800

    def test_overrides_skip_none(self, tmp_path: Path) -> None:
        """CLI truyền `None` cho cờ không đặt — không được ghi đè giá trị trong file."""
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({"name": "x", "collection": "giu_nguyen"}), "utf-8")
        config = load_index_config(path, collection=None, max_documents=7)
        assert config.collection == "giu_nguyen"
        assert config.max_documents == 7

    def test_missing_file_points_at_template(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"configs/indexing/baseline\.yaml"):
            load_index_config(tmp_path / "khong-co.yaml")

    def test_rejects_non_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_index_config(path)


class TestShippedConfigs:
    """Config đi kèm repo phải hợp lệ — hỏng thì `make index` chết ở dòng đầu."""

    @pytest.mark.parametrize("name", ["baseline", "smoke"])
    def test_parses(self, name: str) -> None:
        config = load_index_config(REPO_ROOT / "configs" / "indexing" / f"{name}.yaml")
        assert config.name == name

    def test_baseline_keeps_poc_neighbor_context(self) -> None:
        """Bảo vệ cảnh báo lớn nhất của `W1-13`.

        Mặc định của `ChunkingConfig` là 0, bản POC luôn chạy ở 100. Ai đó dọn
        dẹp config mà bỏ dòng này đi thì baseline sẽ đo một hệ thống chưa từng
        tồn tại, và không có triệu chứng nào ngoài việc con số hơi khác.
        """
        config = load_index_config(REPO_ROOT / "configs" / "indexing" / "baseline.yaml")
        assert config.chunking.neighbor_context_chars == 100
