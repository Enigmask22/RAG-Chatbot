"""`W5-05` — gate phát hành.

Ba nhóm bài, và nhóm đầu là nhóm dễ bỏ sót nhất: **INCOMPARABLE không phải một
kiểu FAIL**. Một gate gộp hai thứ đó lại vẫn "chặn được bundle xấu" trong mọi
bài test hình thức, nhưng nó nói sai việc phải làm — và với `G5` thì đó chính là
điều duy nhất được yêu cầu chứng minh bằng test.

Số trong fixture là số **đã đo thật** của `0.2.1` (`W5-01`/`W5-02`/`W5-04`), nên
khi một bài đỏ thì nó đỏ ở đúng chỗ hệ thống thật đang đứng.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from pipeline.eval.gate import (
    DEFAULT_THRESHOLDS,
    GateStatus,
    GateVerdict,
    RuleOutcome,
    Thresholds,
    evaluate_gate,
    judge_identity,
    load_thresholds,
    main,
    render_html,
)
from rag_core.bundle import (
    BundleComponents,
    ChunkingComponent,
    EmbeddingComponent,
    EvalReport,
    GateRecord,
    GenerationComponent,
    IndexComponent,
    JudgeSpec,
    PromptComponent,
    RagBundle,
    RerankComponent,
    RetrievalComponent,
    save_bundle,
)
from rag_core.bundle import (
    GateStatus as BundleGateStatus,
)

MEASURED_RETRIEVAL = {
    "ndcg@10": 0.7079,
    "recall@5": 0.7847,
    "hit_rate@5": 0.8230,
}
MEASURED_GENERATION = {
    "faithfulness": 0.9877,
    "citation_accuracy": 0.8308,
    "refusal_accuracy": 0.9091,
    "answer_relevancy": 0.7479,
}


def judge(**over: Any) -> JudgeSpec:
    base: dict[str, Any] = {
        "model": "deepseek-v4-flash",
        "temperature": 0.0,
        "kappa_vs_human": 0.7368,
        "rubrics": ("judge-answer-relevancy@v1", "judge-faithfulness@v2"),
        "reasoning": False,
    }
    base.update(over)
    return JudgeSpec(**base)


def bundle(version: str = "0.2.1", **eval_over: Any) -> RagBundle:
    report: dict[str, Any] = {
        "golden_set": "golden_v1",
        "n_queries": 209,
        "evaluated_with_generator": "deepseek-v4-flash",
        "judge": judge(),
        "retrieval_metrics": dict(MEASURED_RETRIEVAL),
        "generation_metrics": dict(MEASURED_GENERATION),
        "unjudged_rate": {"faithfulness": 0.0, "answer_relevancy": 0.0},
        "p95_latency_ms": 759.0,
        "p95_end_to_end_ms": 2900.0,
    }
    report.update(eval_over)
    return RagBundle(
        bundle_version=version,
        created_at=datetime(2026, 9, 5, tzinfo=UTC),
        git_sha="87f912b",
        components=BundleComponents(
            chunking=ChunkingComponent(
                strategy="hybrid",
                chunk_size=1000,
                chunk_overlap=100,
                contextual=True,
                chunking_fingerprint="c7ca3e6fc4da29a5",
            ),
            embedding=EmbeddingComponent(model="BAAI/bge-m3", dim=1024, normalize=True),
            index=IndexComponent(
                backend="qdrant",
                collection="rag_bgem3_ctx",
                fingerprint="a" * 64,
                n_chunks=15814,
                n_documents=60,
            ),
            retrieval=RetrievalComponent(mode="dense", top_k=20),
            rerank=RerankComponent(
                model="BAAI/bge-reranker-v2-m3", candidates=50, top_n=None, max_length=512
            ),
            prompt=PromptComponent(id="chat-system", version=2, hash="ae5ea143"),
            generation=GenerationComponent(
                primary="deepseek-v4-flash", max_tokens=1024, temperature=0.0
            ),
        ),
        eval=EvalReport(**report),
        gate=GateRecord(status=BundleGateStatus.NOT_RUN),
    )


def limits(**over: Any) -> Thresholds:
    base: dict[str, Any] = {
        "require_same": ("golden_set", "evaluated_with_generator", "judge_identity"),
        "reject_alias_identity": True,
        "max_unjudged_rate": 0.05,
        "require_unjudged_rate": True,
        "min_judge_kappa": 0.6,
        "absolute": {
            "ndcg@10": {"min": 0.60, "why": "vì thế"},
            "faithfulness": {"min": 0.92},
            "citation_accuracy": {"min": 0.85},
            "p95_end_to_end_ms": {"max": 3500},
        },
        "max_drop": {"ndcg@10": 0.01, "faithfulness": 0.01},
        "require_metrics_present": True,
    }
    base.update(over)
    return Thresholds(**base)


def outcome_of(verdict: GateVerdict, group: str, name: str) -> RuleOutcome:
    matches = [r for r in verdict.rules if r.group == group and r.name == name]
    assert matches, f"không có luật {group}/{name}: {[(r.group, r.name) for r in verdict.rules]}"
    return matches[0].outcome


class TestHappyPath:
    def test_a_bundle_that_meets_everything_passes(self) -> None:
        good = bundle(generation_metrics={**MEASURED_GENERATION, "citation_accuracy": 0.90})
        verdict = evaluate_gate(good, bundle("0.2.0"), limits())
        assert verdict.status is GateStatus.PASS, [r.detail for r in verdict.failures]
        assert verdict.status.exit_code == 0

    def test_the_real_bundle_fails_on_citation_accuracy(self) -> None:
        """Số thật của `0.2.1`. `TD-64` — cố ý không hạ ngưỡng cho vừa."""
        verdict = evaluate_gate(bundle(), None, limits())
        assert verdict.status is GateStatus.FAIL
        assert verdict.status.exit_code == 1
        assert outcome_of(verdict, "absolute", "citation_accuracy") is RuleOutcome.FAIL
        assert outcome_of(verdict, "absolute", "faithfulness") is RuleOutcome.PASS


class TestIncomparableIsNotAKindOfFail:
    """`G5`: gate phải **từ chối** so hai bundle khác `evaluated_with_generator`."""

    def test_a_different_generator_blocks_the_comparison(self) -> None:
        champion = bundle("0.2.0", evaluated_with_generator="qwen3-8b")
        verdict = evaluate_gate(bundle(), champion, limits())
        assert verdict.status is GateStatus.INCOMPARABLE
        assert outcome_of(verdict, "comparability", "evaluated_with_generator") is RuleOutcome.FAIL

    def test_incomparable_has_its_own_exit_code(self) -> None:
        """CI phân biệt "chất lượng tụt" với "phép đo không đặt cạnh nhau được"
        mà không phải parse stdout."""
        assert GateStatus.INCOMPARABLE.exit_code == 2
        assert GateStatus.FAIL.exit_code == 1
        assert GateStatus.PASS.exit_code == 0

    def test_a_blocked_comparison_still_reports_every_other_rule(self) -> None:
        """Người đọc cần thấy cả hai thứ cùng lúc, không phải sửa một lỗi rồi
        mới biết còn lỗi gì."""
        champion = bundle("0.2.0", evaluated_with_generator="qwen3-8b")
        verdict = evaluate_gate(bundle(), champion, limits())
        assert verdict.status is GateStatus.INCOMPARABLE
        assert outcome_of(verdict, "absolute", "citation_accuracy") is RuleOutcome.FAIL
        assert outcome_of(verdict, "absolute", "ndcg@10") is RuleOutcome.PASS

    def test_a_different_golden_set_blocks_the_comparison(self) -> None:
        champion = bundle("0.2.0", golden_set="golden_v2")
        assert evaluate_gate(bundle(), champion, limits()).status is GateStatus.INCOMPARABLE

    def test_a_quality_regression_is_fail_not_incomparable(self) -> None:
        """Đối chứng: nếu FAIL cũng ra INCOMPARABLE thì hai bài trên vô nghĩa."""
        champion = bundle("0.2.0", retrieval_metrics={**MEASURED_RETRIEVAL, "ndcg@10": 0.80})
        verdict = evaluate_gate(bundle(), champion, limits())
        assert verdict.status is GateStatus.FAIL
        assert outcome_of(verdict, "regression", "ndcg@10") is RuleOutcome.FAIL


class TestJudgeIdentity:
    """`TD-66`: `(model, rubrics, reasoning)`, không chỉ `model`."""

    def test_same_model_different_rubric_is_a_different_judge(self) -> None:
        """`W5-01` đo được: cùng model, rubric v1→v2 đưa `uncited_grounding` từ
        0,427 lên 0,856. Rubric là một phần của thước đo."""
        champion = bundle("0.2.0", judge=judge(rubrics=("judge-faithfulness@v1",)))
        verdict = evaluate_gate(bundle(), champion, limits())
        assert verdict.status is GateStatus.INCOMPARABLE
        assert outcome_of(verdict, "comparability", "judge_identity") is RuleOutcome.FAIL

    def test_same_model_different_reasoning_is_a_different_judge(self) -> None:
        """`W5-04`: bật suy luận đưa faithfulness từ 0,9877 lên 1,0000."""
        champion = bundle("0.2.0", judge=judge(reasoning=True))
        assert evaluate_gate(bundle(), champion, limits()).status is GateStatus.INCOMPARABLE

    def test_a_bundle_with_no_judge_is_not_comparable_to_one_with_a_judge(self) -> None:
        champion = bundle("0.2.0", judge=None, generation_metrics={}, unjudged_rate={})
        assert evaluate_gate(bundle(), champion, limits()).status is GateStatus.INCOMPARABLE

    def test_identity_string_names_all_three_parts(self) -> None:
        text = judge_identity(bundle())
        assert "deepseek-v4-flash" in text
        assert "judge-faithfulness@v2" in text
        assert "reasoning=false" in text

    def test_unknown_reasoning_is_not_silently_read_as_false(self) -> None:
        assert "reasoning=?" in judge_identity(bundle("0.2.0", judge=judge(reasoning=None)))


class TestAliasIdentity:
    """⭐⭐ Chuỗi bằng nhau chưa đủ nếu chuỗi ấy là con trỏ phía server."""

    def test_the_alias_both_real_bundles_carry_is_rejected(self) -> None:
        """`0.1.0` và `0.2.0` đều ghi `deepseek-chat@2026-09`, và `deepseek-chat`
        là bí danh — `W5-03` đo được nó được phục vụ bởi `deepseek-v4-flash`."""
        old = bundle("0.2.0", evaluated_with_generator="deepseek-chat@2026-09")
        verdict = evaluate_gate(old, None, limits())
        assert outcome_of(verdict, "comparability", "generator không phải bí danh") is (
            RuleOutcome.FAIL
        )
        assert verdict.status is GateStatus.INCOMPARABLE

    def test_a_champion_carrying_an_alias_also_blocks(self) -> None:
        champion = bundle("0.2.0", evaluated_with_generator="deepseek-chat@2026-09")
        assert evaluate_gate(bundle(), champion, limits()).status is GateStatus.INCOMPARABLE

    def test_an_openrouter_preset_is_rejected_too(self) -> None:
        preset = bundle(evaluated_with_generator="@preset/my-luna-pro")
        assert evaluate_gate(preset, None, limits()).status is GateStatus.INCOMPARABLE

    def test_an_explicit_slug_passes(self) -> None:
        verdict = evaluate_gate(bundle(), None, limits())
        assert outcome_of(verdict, "comparability", "generator không phải bí danh") is (
            RuleOutcome.PASS
        )


class TestValidityBeforeQuality:
    """`TD-67`: một judge hỏng không cho điểm thấp — nó cho điểm tuyệt đối."""

    def test_the_reasoning_arm_of_w5_04_is_caught_despite_perfect_scores(self) -> None:
        """Tái dựng lần chạy thật: 64% phán quyết mất, và faithfulness = 1,0000.

        Không có luật này thì gate **thả** nó với điểm cao nhất có thể.
        """
        broken = bundle(
            generation_metrics={**MEASURED_GENERATION, "faithfulness": 1.0},
            unjudged_rate={"faithfulness": 0.64},
        )
        verdict = evaluate_gate(broken, None, limits())
        assert outcome_of(verdict, "absolute", "faithfulness") is RuleOutcome.PASS
        assert outcome_of(verdict, "validity", "chưa chấm được · faithfulness") is RuleOutcome.FAIL
        assert verdict.status is GateStatus.FAIL

    def test_not_declaring_the_unjudged_rate_is_red_not_green(self) -> None:
        """ "Không biết" phải đỏ, vì chế độ hỏng đã đo lệch về phía điểm CAO."""
        silent = bundle(unjudged_rate={})
        verdict = evaluate_gate(silent, None, limits())
        assert outcome_of(verdict, "validity", "khai tỉ lệ chưa chấm được") is RuleOutcome.FAIL

    def test_a_retrieval_only_bundle_skips_the_rule_instead_of_failing(self) -> None:
        """Không có metric tầng sinh thì không có judge để hỏng."""
        retrieval_only = bundle(generation_metrics={}, unjudged_rate={}, judge=None)
        verdict = evaluate_gate(retrieval_only, None, limits(absolute={}, min_judge_kappa=None))
        assert outcome_of(verdict, "validity", "tỉ lệ chưa chấm được") is RuleOutcome.SKIP
        assert verdict.status is GateStatus.PASS

    def test_an_uncalibrated_judge_fails(self) -> None:
        assert (
            outcome_of(
                evaluate_gate(bundle(judge=judge(kappa_vs_human=None)), None, limits()),
                "validity",
                "judge đã hiệu chỉnh",
            )
            is RuleOutcome.FAIL
        )

    def test_kappa_below_the_bar_fails(self) -> None:
        """`W5-04` đo GLM ở 0,371 — dưới ngưỡng 0,6."""
        weak = bundle(judge=judge(kappa_vs_human=0.371))
        assert (
            outcome_of(evaluate_gate(weak, None, limits()), "validity", "judge đã hiệu chỉnh")
            is RuleOutcome.FAIL
        )


class TestMissingMetrics:
    def test_a_declared_threshold_with_no_number_is_fail_not_skip(self) -> None:
        """DoD: "thiếu metric FAIL". Bỏ qua là cách một lần eval hỏng đi qua gate."""
        gone = bundle(
            generation_metrics={k: v for k, v in MEASURED_GENERATION.items() if k != "faithfulness"}
        )
        verdict = evaluate_gate(gone, None, limits())
        assert outcome_of(verdict, "absolute", "faithfulness") is RuleOutcome.FAIL
        detail = next(r.detail for r in verdict.failures if r.name == "faithfulness")
        assert "không mang metric này" in detail

    def test_a_metric_the_champion_has_and_the_candidate_lost_is_fail(self) -> None:
        champion = bundle("0.2.0")
        gone = bundle(
            retrieval_metrics={k: v for k, v in MEASURED_RETRIEVAL.items() if k != "ndcg@10"}
        )
        verdict = evaluate_gate(gone, champion, limits())
        assert outcome_of(verdict, "regression", "ndcg@10") is RuleOutcome.FAIL

    def test_that_same_case_can_be_downgraded_only_by_editing_the_thresholds(self) -> None:
        champion = bundle("0.2.0")
        gone = bundle(
            retrieval_metrics={k: v for k, v in MEASURED_RETRIEVAL.items() if k != "ndcg@10"}
        )
        verdict = evaluate_gate(gone, champion, limits(require_metrics_present=False))
        assert outcome_of(verdict, "regression", "ndcg@10") is RuleOutcome.SKIP

    def test_a_metric_name_in_both_tables_is_an_error_not_a_silent_winner(self) -> None:
        clash = bundle(
            retrieval_metrics={**MEASURED_RETRIEVAL, "faithfulness": 0.1},
        )
        with pytest.raises(ValueError, match="trùng tên"):
            evaluate_gate(clash, None, limits())


class TestLatencyIsTwoDifferentNumbers:
    """⚠️ Lỗi `W5-05` đã mắc rồi sửa: `p95_latency_ms` là **truy hồi thuần**."""

    def test_the_retrieval_p95_does_not_satisfy_the_end_to_end_budget(self) -> None:
        """759 ms truy hồi không chứng minh gì về ngân sách 3500 ms end-to-end."""
        no_e2e = bundle(p95_latency_ms=759.0, p95_end_to_end_ms=None)
        verdict = evaluate_gate(no_e2e, None, limits())
        assert outcome_of(verdict, "absolute", "p95_end_to_end_ms") is RuleOutcome.FAIL

    def test_the_measured_end_to_end_p95_of_the_real_run_is_over_budget(self) -> None:
        """4706 ms trên 242 request thật (`W5-01`)."""
        real = bundle(p95_end_to_end_ms=4706.5)
        verdict = evaluate_gate(real, None, limits())
        assert outcome_of(verdict, "absolute", "p95_end_to_end_ms") is RuleOutcome.FAIL


class TestShippedThresholds:
    def test_the_shipped_yaml_loads(self) -> None:
        assert DEFAULT_THRESHOLDS.exists()
        shipped = load_thresholds(DEFAULT_THRESHOLDS)
        assert "evaluated_with_generator" in shipped.require_same
        assert shipped.reject_alias_identity is True
        assert shipped.max_unjudged_rate == pytest.approx(0.05)

    def test_every_absolute_threshold_states_why(self) -> None:
        """Một con số không có lý lẽ là một con số sẽ bị hạ xuống lúc 11 giờ đêm."""
        shipped = load_thresholds(DEFAULT_THRESHOLDS)
        missing = [name for name, spec in shipped.absolute.items() if not spec.get("why")]
        assert not missing, f"ngưỡng thiếu `why`: {missing}"

    def test_every_absolute_threshold_declares_a_direction(self) -> None:
        shipped = load_thresholds(DEFAULT_THRESHOLDS)
        for name, spec in shipped.absolute.items():
            assert "min" in spec or "max" in spec, name

    def test_the_shipped_thresholds_fail_the_real_bundle_the_way_the_report_says(self) -> None:
        """Chốt đúng hai chỗ đỏ đã công bố: `TD-64` và ngân sách end-to-end."""
        verdict = evaluate_gate(bundle(p95_end_to_end_ms=4706.5), None, load_thresholds())
        failed = {rule.name for rule in verdict.failures}
        assert failed == {"citation_accuracy", "p95_end_to_end_ms"}, failed


class TestReportAndCli:
    def test_html_is_self_contained(self) -> None:
        page = render_html(evaluate_gate(bundle(), None, limits()))
        assert "<style>" in page
        assert "http://" not in page and "https://" not in page, "báo cáo gate không được gọi mạng"

    def test_html_carries_the_why_of_each_threshold(self) -> None:
        page = render_html(evaluate_gate(bundle(), None, limits()))
        assert "vì thế" in page

    def test_html_escapes_content(self) -> None:
        page = render_html(
            evaluate_gate(bundle(evaluated_with_generator="<script>x"), None, limits())
        )
        assert "<script>x" not in page
        assert "&lt;script&gt;" in page

    def test_cli_exit_code_matches_the_verdict(self, tmp_path: Path) -> None:
        root = tmp_path / "bundles"
        save_bundle(bundle("0.2.1").signed(), root)
        code = main(
            [
                "--bundle",
                "0.2.1",
                "--no-champion",
                "--root",
                str(root),
                "--thresholds",
                str(DEFAULT_THRESHOLDS),
                "--html",
                str(tmp_path / "gate.html"),
                "--json",
                str(tmp_path / "gate.json"),
            ]
        )
        assert code == 1
        assert (tmp_path / "gate.html").exists()
        assert (tmp_path / "gate.json").exists()

    def test_champion_is_the_highest_version_below_the_candidate(self, tmp_path: Path) -> None:
        """Không lấy "bản mới nhất": ứng viên thường CHÍNH LÀ bản mới nhất, và
        khi ấy nó tự so với mình — một gate luôn xanh."""
        root = tmp_path / "bundles"
        for version in ("0.1.0", "0.2.0", "0.2.1"):
            save_bundle(bundle(version).signed(), root)
        from pipeline.eval.gate import _pick_champion, _resolve_bundle

        picked = _pick_champion(_resolve_bundle("0.2.1", root), root)
        assert picked is not None and picked.bundle_version == "0.2.0"

    def test_the_first_bundle_has_no_champion_and_that_is_not_an_error(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "bundles"
        save_bundle(bundle("0.1.0").signed(), root)
        from pipeline.eval.gate import _pick_champion, _resolve_bundle

        assert _pick_champion(_resolve_bundle("0.1.0", root), root) is None
