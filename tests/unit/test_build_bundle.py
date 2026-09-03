"""`W4-01` — bên **sinh** bundle, nơi một lời nói dối được tạo ra chứ không bị phát hiện.

Checksum bảo vệ manifest khỏi bị sửa *sau* khi ký. Nó không bảo vệ được gì trước
một manifest **sai từ lúc ký** — và đó là kiểu sai dễ xảy ra hơn nhiều: chạy eval
trên index này, đóng gói bằng config kia, ký, và mọi phép kiểm phía sau đều xanh.

Nên nhóm test này gần như chỉ hỏi một câu: builder có từ chối đóng gói khi ba
nguồn (config, báo cáo build, lượt eval) **không nói về cùng một index** không.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pipeline.bundle.build_bundle import _parse_rerank, build_bundle
from pipeline.indexing.config import IndexConfig
from rag_core.bundle import BundleValidationError

RERANKED_NAME = (
    "reranked[qdrant-hybrid:rag_bgem3_ctx:rrf1-c20]:BAAI/bge-reranker-v2-m3@cuda:L512:float16:n50"
)


def make_config(**overrides: Any) -> IndexConfig:
    base: dict[str, Any] = {
        "name": "bgem3-contextual",
        "collection": "rag_bgem3_ctx",
        "chunking": {"strategy": "hybrid", "chunk_size": 1000, "chunk_overlap": 100},
        "embedding_model": "BAAI/bge-m3",
        "contextual": {"enabled": True},
    }
    base.update(overrides)
    return IndexConfig(**base)


def make_index_report(config: IndexConfig, **overrides: Any) -> dict[str, Any]:
    report: dict[str, Any] = {
        "fingerprint": config.fingerprint,
        "collection": config.collection,
        "embedding_dim": 1024,
        "collection_count": 15814,
        "n_documents_indexed": 60,
    }
    report.update(overrides)
    return report


def make_eval_run(config: IndexConfig, **config_overrides: Any) -> dict[str, Any]:
    run_config: dict[str, Any] = {
        "retriever": RERANKED_NAME,
        "top_k": 20,
        "index_fingerprint": config.fingerprint,
        "collection": config.collection,
        "retrieval_mode": "reranked",
        "branch_options": {"k": 1, "candidate_k": 20, "base": "hybrid"},
        "golden": "data/golden/golden_v1.jsonl",
    }
    run_config.update(config_overrides)
    return {
        "n_scored": 209,
        "overall": {"ndcg@10": 0.6888, "hit_rate@5": 0.8086},
        "latency_ms": {"p95": 809.27},
        "config": run_config,
    }


def build(config: IndexConfig | None = None, **kwargs: Any) -> Any:
    config = config or make_config()
    args: dict[str, Any] = {
        "config": config,
        "index_report": make_index_report(config),
        "eval_run": make_eval_run(config),
        "version": "0.1.0",
        "generator": "deepseek-chat@2026-09",
    }
    args.update(kwargs)
    return build_bundle(**args)


# ---------------------------------------------------------------------------
# 1. ⭐⭐ Từ chối đóng gói chéo
# ---------------------------------------------------------------------------


def test_eval_from_a_different_index_is_refused() -> None:
    """Kiểu sai nguy hiểm nhất, và kiểu duy nhất checksum không bắt được.

    Số eval đo trên `rag_bgem3` đem đóng gói với config `bgem3-contextual` cho ra
    một manifest hợp lệ, ký được, checksum khớp — và nói dối về hệ thống nào đạt
    được những con số ấy.
    """
    config = make_config()
    other = make_config(contextual={"enabled": False})
    with pytest.raises(BundleValidationError, match="vân tay index không khớp"):
        build(config, eval_run=make_eval_run(other))


def test_index_report_from_a_different_build_is_refused() -> None:
    """Build lại index sau khi eval xong: cùng tên collection, khác nội dung."""
    config = make_config()
    with pytest.raises(BundleValidationError, match="báo cáo build index"):
        build(config, index_report=make_index_report(config, fingerprint="0" * 64))


def test_the_error_names_every_source_that_disagrees() -> None:
    """Một nguồn lệch và hai nguồn lệch là hai tình huống khác nhau."""
    config = make_config()
    other = make_config(chunking={"chunk_size": 550, "chunk_overlap": 50})
    with pytest.raises(BundleValidationError) as excinfo:
        build(
            config,
            index_report=make_index_report(other),
            eval_run=make_eval_run(other),
        )
    message = str(excinfo.value)
    assert "báo cáo build index" in message
    assert "lượt chạy eval" in message


def test_collection_rename_is_caught_even_when_fingerprints_agree() -> None:
    """Vân tay không gồm tên collection (đúng), nên tên phải được kiểm riêng.

    ⚠️ Nếu bỏ phép kiểm này thì bundle trỏ serving vào một collection **không
    tồn tại** trong khi mọi vân tay đều khớp.
    """
    config = make_config()
    with pytest.raises(BundleValidationError, match="collection"):
        build(config, eval_run=make_eval_run(config, collection="rag_bgem3"))


def test_matching_sources_produce_a_bundle() -> None:
    bundle = build()
    assert bundle.components.index.collection == "rag_bgem3_ctx"
    assert bundle.eval.retrieval_metrics["ndcg@10"] == pytest.approx(0.6888)


# ---------------------------------------------------------------------------
# 2. Bóc tham số reranker khỏi tên nhánh
# ---------------------------------------------------------------------------


def test_rerank_parameters_come_from_the_run_not_from_defaults() -> None:
    component = _parse_rerank(RERANKED_NAME, {})
    assert component is not None
    assert (component.model, component.max_length, component.candidates) == (
        "BAAI/bge-reranker-v2-m3",
        512,
        50,
    )


def test_a_branch_name_missing_the_window_is_an_error_not_a_default() -> None:
    """⭐ Đoán `max_length=512` ở đây sẽ tạo ra bundle mô tả sai hệ thống đã đo —
    và `W3-04` vừa đo được rằng cửa sổ ấy là ràng buộc thật, không phải chi tiết."""
    with pytest.raises(BundleValidationError, match="không bóc được"):
        _parse_rerank("reranked[qdrant-dense:c]:BAAI/bge-reranker-v2-m3@cuda", {})


def test_a_non_reranked_branch_has_no_rerank_component() -> None:
    """`None` ở đây nghĩa là *tắt rerank*, và đó là mô tả đúng của nhánh dense."""
    assert _parse_rerank("qdrant-dense:rag_bgem3", {}) is None


def test_base_branch_becomes_the_mode_not_an_option() -> None:
    """`base` mô tả nhánh nền của reranked; để nó nằm trong `options` thì
    `build_branch` sẽ nhận một tham số nó không hiểu và nổ lúc serving load."""
    bundle = build()
    assert bundle.components.retrieval.mode == "hybrid"
    assert "base" not in bundle.components.retrieval.options
    assert bundle.components.retrieval.options == {"k": 1, "candidate_k": 20}


# ---------------------------------------------------------------------------
# 3. Bundle hôm nay khai thiếu tầng sinh — và khai thật
# ---------------------------------------------------------------------------


def test_todays_bundle_is_retrieval_only() -> None:
    bundle = build()
    assert bundle.serves_generation is False
    assert bundle.components.prompt is None
    assert bundle.eval.generation_metrics == {}


def test_generator_is_required_by_the_signature() -> None:
    """Không có mặc định nào cho `evaluated_with_generator`, kể cả ở tầng CLI."""
    with pytest.raises(TypeError):
        build_bundle(  # type: ignore[call-arg]
            config=make_config(),
            index_report=make_index_report(make_config()),
            eval_run=make_eval_run(make_config()),
            version="0.1.0",
        )


# ---------------------------------------------------------------------------
# 4. Bundle mẫu đã commit phải còn hợp lệ
# ---------------------------------------------------------------------------

SAMPLE = Path(__file__).resolve().parents[2] / "bundles" / "rag-bundle-v0.1.0" / "manifest.json"


@pytest.mark.skipif(not SAMPLE.is_file(), reason="chưa sinh bundle mẫu")
def test_the_committed_sample_bundle_still_loads() -> None:
    """Bundle mẫu là *bằng chứng* của `W4-01`, nên nó phải chịu được mọi thay đổi
    schema sau này — hoặc bị sinh lại một cách có ý thức."""
    from rag_core.bundle import load_bundle

    bundle = load_bundle(SAMPLE)
    assert bundle.bundle_version == "0.1.0"
    assert bundle.components.index.n_chunks == 15814
    assert bundle.eval.retrieval_metrics["ndcg@10"] == pytest.approx(0.6888, abs=1e-4)


def test_the_sample_bundle_matches_the_real_index_config() -> None:
    """Ghim rằng bundle mẫu được sinh từ artifact thật, không phải gõ tay."""
    if not SAMPLE.is_file():  # pragma: no cover
        pytest.skip("chưa sinh bundle mẫu")
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    from pipeline.indexing.config import load_index_config

    config = load_index_config(Path("configs/indexing/bgem3-contextual.yaml"))
    assert raw["components"]["index"]["fingerprint"] == config.fingerprint
    assert raw["components"]["chunking"]["chunking_fingerprint"] == config.chunking_fingerprint


# ---------------------------------------------------------------------------
# TD-38 — bundle mới phải mang danh tính của hệ thống đã đo
# ---------------------------------------------------------------------------


def test_the_measured_retriever_name_is_copied_verbatim() -> None:
    """⭐ Chép nguyên, **không** dựng lại từ các trường đã bóc ra.

    Dựng lại nghĩa là có một *bản sao thứ hai* của quy ước đặt tên sống trong
    `build_bundle`, và hai bản sao sẽ lệch nhau ở đúng lần `rag_core` đổi hậu tố
    — lúc không ai nhìn, và theo hướng làm phép kiểm danh tính ở serving đỏ giả.
    """
    assert build().components.retriever_name == RERANKED_NAME


def test_an_eval_report_without_a_retriever_name_cannot_be_packaged() -> None:
    """Schema cho phép `None` để bundle sinh trước `TD-38` còn nạp được; phía
    **sinh** thì không, vì một bundle mới thiếu trường này nạp xanh trên mọi cấu
    hình sai. Chỗ duy nhất được phép để trống là quá khứ."""
    config = make_config()
    with pytest.raises(BundleValidationError, match="TD-38"):
        build(config, eval_run=make_eval_run(config, retriever=""))
