"""`W3-04` — nối ngữ cảnh vào `build_index`, và ba thứ phải từ chối build.

Dán ngữ cảnh sai còn tệ hơn không dán: index trông bình thường, số chunk đúng,
build xanh, và mọi metric chỉ đơn giản là tệ hơn mà không ai truy ra vì sao. Nên
nhóm test này gần như chỉ kiểm **cái gì làm build DỪNG**.

Cái nguy hiểm nhất không phải "thiếu ngữ cảnh" mà là **ngữ cảnh của bộ chunk
khác**: `chunk_id` là `doc::index`, nên ngữ cảnh sinh cho `chunk_size=1000` vẫn
khớp id với chunk của `chunk_size=550` trong khi nội dung khác hẳn. Coverage báo
100%.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.indexing.build_index import _check_coverage, _load_contexts, _merge_enrich
from pipeline.indexing.config import IndexConfig
from rag_core.chunking.contextual import EnrichStats

FINGERPRINT = "c0ffee0000000001"
OTHER = "deadbeef00000002"


def write_contexts(path: Path, *, n: int = 4, cfg: str = FINGERPRINT) -> Path:
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "key": f"k{i}",
                    "chunk_id": f"doc::{i:05d}",
                    "doc_id": "doc",
                    "cfg": cfg,
                    "context": f"Ngữ cảnh của đoạn {i}.",
                },
                ensure_ascii=False,
            )
            for i in range(n)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def config(tmp_path: Path, **contextual: object) -> IndexConfig:
    base: dict[str, object] = {"enabled": True, "contexts_path": tmp_path / "contexts.jsonl"}
    base.update(contextual)
    return IndexConfig(name="t", tenant_id="t", contextual=base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# 1. Hai vân tay, hai mục đích
# --------------------------------------------------------------------------


def test_enabling_contextual_changes_the_vector_fingerprint() -> None:
    """Bật là đổi mọi vector, nên state cũ phải bị từ chối — `--recreate` mới chạy được."""
    off = IndexConfig(name="t", tenant_id="t")
    on = IndexConfig(name="t", tenant_id="t", contextual={"enabled": True})  # type: ignore[arg-type]
    assert off.fingerprint != on.fingerprint


def test_enabling_contextual_does_NOT_change_the_chunking_fingerprint() -> None:
    """⭐ Đây là lý do phải có hai vân tay riêng.

    Nếu dùng chung một vân tay thì bật Contextual Retrieval sẽ làm chính artifact
    ngữ cảnh trở thành "sai vân tay" — tức bật lên là tự vô hiệu hoá thứ vừa bật.
    """
    off = IndexConfig(name="t", tenant_id="t")
    on = IndexConfig(name="t", tenant_id="t", contextual={"enabled": True})  # type: ignore[arg-type]
    assert off.chunking_fingerprint == on.chunking_fingerprint


def test_changing_chunk_size_changes_the_chunking_fingerprint() -> None:
    a = IndexConfig(name="t", tenant_id="t", chunking={"chunk_size": 1000})  # type: ignore[arg-type]
    b = IndexConfig(name="t", tenant_id="t", chunking={"chunk_size": 550})  # type: ignore[arg-type]
    assert a.chunking_fingerprint != b.chunking_fingerprint


def test_embedding_model_is_in_the_chunking_fingerprint() -> None:
    """`HybridChunker` mượn tokenizer của embedding model để đo kích thước."""
    a = IndexConfig(name="t", tenant_id="t", embedding_model="BAAI/bge-m3")
    b = IndexConfig(
        name="t", tenant_id="t", embedding_model="bkai-foundation-models/vietnamese-bi-encoder"
    )
    assert a.chunking_fingerprint != b.chunking_fingerprint


def test_device_is_not_in_the_chunking_fingerprint() -> None:
    """Chạy trên CPU hay GPU không đổi bộ chunk — cùng lý lẽ với `fingerprint`."""
    a = IndexConfig(name="t", tenant_id="t", embedding_device="cpu")
    b = IndexConfig(name="t", tenant_id="t", embedding_device="cuda")
    assert a.chunking_fingerprint == b.chunking_fingerprint


# --------------------------------------------------------------------------
# 2. Nạp artifact
# --------------------------------------------------------------------------


def test_disabled_returns_none_not_an_empty_dict() -> None:
    """⚠️ `None` = tắt, `{}` = không thể xảy ra (hàm ném trước đó).

    Nếu tắt cũng trả `{}` thì `if contexts:` ở chỗ gọi sẽ đối xử "tắt" giống
    "bật nhưng không nạp được gì", và cái sau phải là lỗi.
    """
    assert _load_contexts(IndexConfig(name="t", tenant_id="t")) is None


def test_enabled_without_the_artifact_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ctx-run-glm"):
        _load_contexts(config(tmp_path))


def test_contexts_from_another_chunking_configuration_are_refused(tmp_path: Path) -> None:
    """⭐⭐ Chốt chặn chính của cả nhóm test này.

    Artifact đầy đủ, đọc được, `chunk_id` khớp hết — nhưng nó sinh cho bộ chunk
    khác. Không có phép kiểm này thì build chạy xanh và index nhận 15.814 câu mô
    tả sai đoạn.
    """
    write_contexts(tmp_path / "contexts.jsonl", cfg=OTHER)
    with pytest.raises(RuntimeError, match="vân tay"):
        _load_contexts(config(tmp_path))


def test_the_refusal_message_says_how_many_contexts_were_found(tmp_path: Path) -> None:
    """Phân biệt "file rỗng" với "file đầy nhưng sai vân tay" — hai cách sửa khác nhau."""
    write_contexts(tmp_path / "contexts.jsonl", n=7, cfg=OTHER)
    with pytest.raises(RuntimeError, match="7 ngữ cảnh"):
        _load_contexts(config(tmp_path))


def test_the_fingerprint_check_can_be_switched_off_explicitly(tmp_path: Path) -> None:
    """Lối thoát phải có, nhưng phải **tường minh** chứ không phải mặc định."""
    write_contexts(tmp_path / "contexts.jsonl", cfg=OTHER)
    loaded = _load_contexts(config(tmp_path, require_fingerprint=False))
    assert loaded is not None
    assert len(loaded) == 4


def test_matching_fingerprint_loads(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    write_contexts(tmp_path / "contexts.jsonl", cfg=cfg.chunking_fingerprint)
    loaded = _load_contexts(cfg)
    assert loaded is not None
    assert loaded["doc::00002"] == "Ngữ cảnh của đoạn 2."


# --------------------------------------------------------------------------
# 3. Ngưỡng phủ
# --------------------------------------------------------------------------


def test_coverage_below_the_threshold_stops_the_build() -> None:
    """`apply_contexts` cố ý không ném khi thiếu; tầng build là nơi vạch ranh giới.

    Thiếu 1 chunk và thiếu 8.000 chunk trông giống hệt nhau ở phía build: cả hai
    chạy xong, cả hai không báo gì, và chỉ một trong hai cho ra index dùng được.
    """
    enrich = EnrichStats(n_chunks=1000, n_enriched=800, n_missing=200)
    enrich.missing_chunk_ids = [f"doc::{i:05d}" for i in range(20)]
    with pytest.raises(RuntimeError, match="ctx-coverage"):
        _check_coverage(IndexConfig(name="t", tenant_id="t", contextual={"enabled": True}), enrich)  # type: ignore[arg-type]


def test_coverage_at_the_threshold_passes() -> None:
    enrich = EnrichStats(n_chunks=100, n_enriched=95, n_missing=5)
    _check_coverage(IndexConfig(name="t", tenant_id="t", contextual={"enabled": True}), enrich)  # type: ignore[arg-type]


def test_the_threshold_is_configurable() -> None:
    enrich = EnrichStats(n_chunks=100, n_enriched=80, n_missing=20)
    cfg = IndexConfig(name="t", tenant_id="t", contextual={"enabled": True, "min_coverage": 0.5})  # type: ignore[arg-type]
    _check_coverage(cfg, enrich)


# --------------------------------------------------------------------------
# 4. Cộng dồn thống kê qua các tài liệu
# --------------------------------------------------------------------------


def test_stats_accumulate_across_documents() -> None:
    total = EnrichStats()
    _merge_enrich(total, EnrichStats(n_chunks=10, n_enriched=9, n_missing=1))
    _merge_enrich(total, EnrichStats(n_chunks=5, n_enriched=5))
    assert (total.n_chunks, total.n_enriched, total.n_missing) == (15, 14, 1)
    assert total.coverage == pytest.approx(14 / 15)


def test_the_missing_example_list_stays_bounded() -> None:
    """Log ví dụ, không log danh sách. 8.000 chunk_id trong một dòng log là vô dụng."""
    total = EnrichStats()
    for batch in range(10):
        part = EnrichStats(n_chunks=100, n_missing=100)
        part.missing_chunk_ids = [f"d{batch}::{i:05d}" for i in range(20)]
        _merge_enrich(total, part)
    assert total.n_missing == 1000
    assert len(total.missing_chunk_ids) == 20
