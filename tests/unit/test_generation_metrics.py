"""`W5-01` — metric tầng sinh trên fixture tính tay.

Không mạng, không judge thật: `Judge` nhận provider giả, nên mọi con số ở đây
kiểm được bằng phép nhẩm. Đó là điều kiện để một metric còn dùng được — nếu
không tự tính lại được điểm trên 5 câu, thì điểm trên 242 câu là một con số phải
tin chứ không phải một con số đọc được.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest

from pipeline.eval.answer_run import AnswerRecord
from pipeline.eval.generation_metrics import (
    Aggregate,
    citation_accuracy,
    citation_coverage,
    citation_validity,
    claims_of,
    context_precision,
    context_recall,
    derived,
    faithfulness_questions,
    markers,
    score_faithfulness,
    score_misattribution,
    score_refusal,
    score_relevancy,
    score_uncited_grounding,
    split_sentences,
    strip_markers,
)
from pipeline.eval.golden import GoldenQuery, QueryCategory
from pipeline.eval.judge import Judge, JudgeConfig
from rag_core.llm import ChatMessage, LLMProvider, LLMResponse
from rag_core.schemas import TokenUsage

# ---------------------------------------------------------------- tách câu


class TestSplitSentences:
    def test_tach_cau_don_gian(self) -> None:
        assert split_sentences("Câu một. Câu hai! Câu ba?") == [
            "Câu một.",
            "Câu hai!",
            "Câu ba?",
        ]

    def test_khong_tach_giua_mot_con_so(self) -> None:
        r"""⭐ Tiếng Việt dùng dấu chấm làm phân cách hàng nghìn.

        Tách ở đó cắt `1.234.567` thành hai "câu" và cả hai mảnh đều vô nghĩa
        với judge — nó sẽ chấm `NOT_FOUND` cho một con số hoàn toàn có thật.

        Cơ chế giữ được điều này là `\s+` trong `_SENTENCE_END`, **không** phải
        một phép kiểm chữ số riêng: phép tiêm `G1` bỏ phép kiểm ấy đi mà test
        vẫn xanh, vì nó là mã không chạy được. Đã gỡ; test ở lại và giờ mới thật
        sự kiểm cơ chế đang chạy.
        """
        text = "Thu ngân sách đạt 1.234.567 tỷ đồng. Vượt dự toán."
        assert split_sentences(text) == [
            "Thu ngân sách đạt 1.234.567 tỷ đồng.",
            "Vượt dự toán.",
        ]

    @pytest.mark.parametrize("abbrev", ["TP.", "v.v.", "GS.", "No."])
    def test_khong_tach_sau_tu_viet_tat(self, abbrev: str) -> None:
        text = f"Số liệu {abbrev} Hồ Chí Minh cho thấy tăng trưởng. Kết thúc."
        assert len(split_sentences(text)) == 2

    def test_gach_dau_dong_la_ranh_gioi_y(self) -> None:
        """`- ý một [1]\\n- ý hai [2]` là hai mệnh đề, không phải một câu dài."""
        text = "Kết quả:\n- GDP tăng 7,09% [1]\n- Lạm phát 3,63% [2]"
        assert split_sentences(text) == [
            "Kết quả:",
            "GDP tăng 7,09% [1]",
            "Lạm phát 3,63% [2]",
        ]

    def test_van_ban_rong_cho_danh_sach_rong(self) -> None:
        assert split_sentences("   \n  ") == []

    def test_cau_cuoi_khong_co_dau_cham_van_duoc_giu(self) -> None:
        assert split_sentences("Một câu. Câu cuối không chấm") == [
            "Một câu.",
            "Câu cuối không chấm",
        ]


class TestMarkers:
    def test_lay_so_nguon_theo_thu_tu_khu_trung(self) -> None:
        assert markers("Ý này [3] và ý kia [1], nhắc lại [3].") == (3, 1)

    def test_khong_co_marker_tra_tuple_rong(self) -> None:
        assert markers("Không trích gì cả.") == ()

    def test_strip_bo_marker_va_khoang_trang_thua(self) -> None:
        assert strip_markers("GDP tăng 7,09% [1] [2].") == "GDP tăng 7,09%."


# ------------------------------------------------------------------ fixture


def record(
    query_id: str,
    answer: str,
    *,
    category: str = "factoid",
    chunk_ids: Sequence[str] = ("c1", "c2"),
    citations: Sequence[dict[str, Any]] = (),
    route: str = "retrieve",
) -> AnswerRecord:
    return AnswerRecord(
        query_id=query_id,
        query=f"câu hỏi {query_id}?",
        category=category,
        lang="vi",
        answer=answer,
        route=route,
        rewritten=None,
        prompt_spec="chat-system@v2",
        bundle_version="0.2.0",
        model="deepseek-v4-flash",
        finish_reason="stop",
        sources=[
            {"n": i, "chunk_id": cid, "doc_id": "d1", "score": 0.9}
            for i, cid in enumerate(chunk_ids, start=1)
        ],
        citations=list(citations),
        citation_block="ok" if citations else "absent",
    )


def golden(query_id: str, relevant: Sequence[str], category: str = "factoid") -> GoldenQuery:
    return GoldenQuery(
        query_id=query_id,
        query=f"câu hỏi {query_id}?",
        category=QueryCategory(category),
        relevant_chunk_ids=list(relevant),
    )


class LabelProvider(LLMProvider):
    """Trả nhãn theo bảng tra dựa trên nội dung MỆNH ĐỀ trong prompt."""

    name = "labels"
    model = "labels-model"

    def __init__(self, table: Mapping[str, str], default: str = "NOT_FOUND") -> None:
        self.table = dict(table)
        self.default = default
        self.calls = 0

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        seed: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        prompt = str(messages[0].content)
        label = self.default
        for needle, value in self.table.items():
            if needle in prompt:
                label = value
                break
        return LLMResponse(
            text=json.dumps({"verdict": label, "reason": "giả"}, ensure_ascii=False),
            model=self.model,
            model_requested=self.model,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, cost_usd=0.0),
            finish_reason="stop",
        )


def make_judge(tmp_path: Path, table: Mapping[str, str], default: str = "NOT_FOUND") -> Judge:
    config = JudgeConfig(
        model="deepseek-v4-flash", cache_path=tmp_path / "j.sqlite3", concurrency=1
    )
    return Judge(config, LabelProvider(table, default))


# ------------------------------------------------------- metric tất định


class TestContextMetrics:
    def test_precision_va_recall_tinh_tay_duoc(self) -> None:
        records = [record("q1", "trả lời", chunk_ids=("c1", "c9"))]
        labels = {"q1": golden("q1", ["c1", "c2"])}
        # top-5, 1/5 kết quả liên quan; 1/2 nhãn được lấy về.
        assert context_precision(records, labels, k=5).value == pytest.approx(0.2)
        assert context_recall(records, labels, k=5).value == pytest.approx(0.5)

    def test_cau_unanswerable_bi_loai_chu_khong_tinh_0(self) -> None:
        """`metrics.py` chốt từ `W1-08`: recall trên tập rỗng là **không xác
        định**. Quy thành 0 kéo tụt điểm vô nghĩa, quy thành 1 thì thổi phồng."""
        records = [
            record("q1", "trả lời", chunk_ids=("c1",)),
            record("q2", "không đủ thông tin", category="unanswerable", chunk_ids=("c7",)),
        ]
        labels = {
            "q1": golden("q1", ["c1"]),
            "q2": GoldenQuery(query_id="q2", query="?", category=QueryCategory.UNANSWERABLE),
        }
        agg = context_recall(records, labels, k=5)
        assert agg.n == 1
        assert agg.n_no_evidence == 1
        assert agg.value == pytest.approx(1.0)

    def test_breakdown_theo_nhom(self) -> None:
        records = [
            record("q1", "a", chunk_ids=("c1",), category="factoid"),
            record("q2", "b", chunk_ids=("c9",), category="multi_hop"),
        ]
        labels = {
            "q1": golden("q1", ["c1"], "factoid"),
            "q2": golden("q2", ["c2"], "multi_hop"),
        }
        breakdown = context_recall(records, labels, k=5).breakdown()
        assert breakdown["factoid"]["value"] == 1.0
        assert breakdown["multi_hop"]["value"] == 0.0


class TestCitationMetrics:
    def test_coverage_dem_menh_de_co_marker(self) -> None:
        answer = (
            "GDP Việt Nam năm 2024 tăng bảy phẩy không chín phần trăm [1]. "
            "Con số này cao hơn hẳn mức tăng trưởng của năm hai nghìn hai ba."
        )
        agg = citation_coverage([record("q1", answer)])
        assert agg.n == 2
        assert agg.value == pytest.approx(0.5)

    def test_coverage_bo_qua_nhanh_khong_truy_hoi(self) -> None:
        """Nhánh chào hỏi không có nguồn nào để trích — đòi nó trích là đòi sai."""
        agg = citation_coverage(
            [record("q1", "Chào bạn, tôi có thể giúp gì?", route="no_retrieval")]
        )
        assert agg.n == 0
        assert agg.n_no_evidence == 1

    def test_validity_dem_verified_cua_w4_09(self) -> None:
        citations = [
            {"chunk_id": "c1", "doc_id": "d", "quote": "x", "verified": True},
            {"chunk_id": "c2", "doc_id": "d", "quote": "y", "verified": False},
        ]
        agg = citation_validity([record("q1", "trả lời [1][2].", citations=citations)])
        assert agg.value == pytest.approx(0.5)

    def test_khong_co_citation_thi_khong_vao_mau_so(self) -> None:
        agg = citation_validity([record("q1", "trả lời không trích gì.")])
        assert agg.n == 0
        assert agg.n_no_evidence == 1


# -------------------------------------------------------- metric có judge


class TestFaithfulness:
    chunks: ClassVar[dict[str, str]] = {
        "c1": "GDP năm 2024 tăng 7,09%.",
        "c2": "Lạm phát năm 2024 là 3,63%.",
    }

    def test_chi_dua_dung_chunk_duoc_trich_cho_judge(self) -> None:
        """⭐⭐ Đây là quyết định trung tâm của module: chấm với **nguồn đã trích**,
        không phải hợp mọi nguồn. Chấm với hợp trả lời một câu yếu hơn và giấu
        mất lỗi gán nhầm nguồn."""
        answer = "Tổng sản phẩm quốc nội năm 2024 tăng bảy phẩy không chín phần trăm [1]."
        pairs = faithfulness_questions(record("q1", answer), self.chunks)
        assert len(pairs) == 1
        question = pairs[0][1]
        assert question is not None
        assert "GDP năm 2024" in question.variables["context"]
        assert "Lạm phát" not in question.variables["context"]

    def test_union_dua_moi_chunk(self) -> None:
        answer = "Tổng sản phẩm quốc nội năm 2024 tăng bảy phẩy không chín phần trăm [1]."
        pairs = faithfulness_questions(record("q1", answer), self.chunks, union=True)
        question = pairs[0][1]
        assert question is not None
        assert "Lạm phát" in question.variables["context"]

    def test_cau_khong_trich_nguon_khong_bi_tinh_la_khong_trung_thuc(self, tmp_path: Path) -> None:
        """⭐ Nó là một lỗi **khác** — vi phạm luật 2 — và có phép đo riêng.

        Gộp hai thứ làm mất khả năng phân biệt "model bịa" với "model quên
        trích", trong khi cách chữa hai bên hoàn toàn khác nhau.
        """
        answer = (
            "Tổng sản phẩm quốc nội năm 2024 tăng bảy phẩy không chín phần trăm [1]. "
            "Còn câu này thì không trích nguồn nào và dài hơn hai lăm ký tự."
        )
        judge = make_judge(tmp_path, {"Tổng sản phẩm": "SUPPORTED"})
        agg, detail = score_faithfulness([record("q1", answer)], self.chunks, judge)
        assert agg.n == 1
        assert agg.n_no_evidence == 1
        assert agg.value == pytest.approx(1.0)
        assert len(detail) == 1

    def test_chunk_thieu_trong_sidecar_khong_bi_tinh_la_that_bai(self, tmp_path: Path) -> None:
        """Index đã đổi kể từ lần chạy là lỗi của **bằng chứng**, không phải của
        câu trả lời. Tính nó thành `NOT_FOUND` là đổ lỗi nhầm chỗ."""
        answer = "Tổng sản phẩm quốc nội năm 2024 tăng bảy phẩy không chín phần trăm [1]."
        judge = make_judge(tmp_path, {})
        agg, _ = score_faithfulness([record("q1", answer)], {}, judge)
        assert agg.n == 0
        assert agg.n_no_evidence == 1

    def test_phan_quyet_khong_doc_duoc_bi_loai_khoi_ca_tu_va_mau(self, tmp_path: Path) -> None:
        answer = "Tổng sản phẩm quốc nội năm 2024 tăng bảy phẩy không chín phần trăm [1]."
        config = JudgeConfig(
            model="deepseek-v4-flash", cache_path=tmp_path / "j.sqlite3", concurrency=1
        )
        judge = Judge(config, LabelProvider({}, default="NHAN_LA"))
        agg, _ = score_faithfulness([record("q1", answer)], self.chunks, judge)
        assert agg.n == 0
        assert agg.n_unjudged == 1
        assert agg.value is None

    def test_gan_nham_nguon_chi_hoi_lai_nhung_menh_de_da_truot(self, tmp_path: Path) -> None:
        """⭐ Mệnh đề đã được chunk nó trích chống đỡ thì **theo định nghĩa**
        không thể gán nhầm — hỏi lại nó là mua một câu trả lời đã biết.

        Ở đây mệnh đề nói về lạm phát nhưng trích `[1]` (chunk GDP): vòng theo
        nguồn trích trả `NOT_FOUND`, vòng hợp trả `SUPPORTED` ⇒ nội dung có
        thật, **số nguồn sai**.
        """
        answer = "Lạm phát năm 2024 ở mức ba phẩy sáu ba phần trăm [1]."
        records = [record("q1", answer)]
        cited = make_judge(tmp_path / "a", {}, default="NOT_FOUND")
        agg_cited, detail = score_faithfulness(records, self.chunks, cited)
        assert agg_cited.value == pytest.approx(0.0)

        union = make_judge(tmp_path / "b", {"Lạm phát": "SUPPORTED"})
        agg_mis, examples = score_misattribution(records, self.chunks, union, detail)
        assert agg_mis.value == pytest.approx(1.0)
        assert [c.query_id for c in examples] == ["q1"]
        assert union.provider.calls == 1  # type: ignore[union-attr]

    def test_menh_de_da_duoc_chong_do_khong_ton_them_loi_goi_nao(self, tmp_path: Path) -> None:
        answer = "Tổng sản phẩm quốc nội năm 2024 tăng bảy phẩy không chín phần trăm [1]."
        records = [record("q1", answer)]
        cited = make_judge(tmp_path / "a", {"Tổng sản phẩm": "SUPPORTED"})
        _, detail = score_faithfulness(records, self.chunks, cited)

        union = make_judge(tmp_path / "b", {})
        agg_mis, examples = score_misattribution(records, self.chunks, union, detail)
        assert union.provider.calls == 0  # type: ignore[union-attr]
        assert agg_mis.n == 1
        assert agg_mis.value == pytest.approx(0.0)
        assert examples == []

    def test_cau_tu_choi_khong_bi_tinh_la_khong_trung_thuc(self, tmp_path: Path) -> None:
        """⭐⭐ Phép đo suýt sai, và đây là chỗ nó được chặn.

        Vòng chấm đầu cho `uncited_grounding = 0,427`; đọc ví dụ thật thì gần hết
        là câu meta — model đang **tuân thủ luật 3** của `chat-system`. Chấm một
        lời từ chối trung thực thành `NOT_FOUND` là đếm nó thành một lỗi.
        """
        answer = "Dựa trên ngữ cảnh được cung cấp, tôi không đủ thông tin để trả lời."
        judge = make_judge(tmp_path, {"không đủ thông tin": "NO_CLAIM"})
        agg, _ = score_uncited_grounding([record("q1", answer)], self.chunks, judge)
        assert agg.n == 0
        assert agg.n_not_a_claim == 1
        assert agg.value is None


class TestRelevancy:
    def test_refusal_khong_vao_tu_so_nhung_o_lai_mau_so(self, tmp_path: Path) -> None:
        """Với câu trả lời được, từ chối là một lần trượt: người dùng hỏi và
        không nhận được gì. Quy ước cộng điểm nằm ở mã, không ở judge."""
        records = [
            record("q1", "GDP tăng 7,09% [1]."),
            record("q2", "Tôi không đủ thông tin để trả lời."),
        ]
        judge = make_judge(tmp_path, {"q1": "RELEVANT", "q2": "REFUSAL"}, default="IRRELEVANT")
        agg, labels = score_relevancy(records, judge)
        assert labels == {"q1": "RELEVANT", "q2": "REFUSAL"}
        assert agg.n == 2
        assert agg.value == pytest.approx(0.5)

    def test_cau_tra_loi_rong_khong_vao_mau_so(self, tmp_path: Path) -> None:
        judge = make_judge(tmp_path, {})
        agg, labels = score_relevancy([record("q1", "   ")], judge)
        assert agg.n == 0
        assert agg.n_no_evidence == 1
        assert labels == {}


class TestAggregateContract:
    def test_mau_so_rong_tra_none_chu_khong_tra_0(self) -> None:
        """`0.0` và "chưa đo được" là hai điều hoàn toàn khác nhau, và gộp chúng
        là cách chắc chắn để một ô trống trong báo cáo trông như một điểm kém."""
        assert Aggregate("x").value is None
        assert Aggregate("x").as_dict()["value"] is None

    def test_moi_tong_hop_deu_khai_n_unjudged(self) -> None:
        keys = set(Aggregate("x").as_dict())
        assert {
            "value",
            "n",
            "n_unjudged",
            "n_no_evidence",
            "n_not_a_claim",
            "by_category",
        } == keys


class TestClaimFiltering:
    def test_cau_qua_ngan_khong_phai_menh_de(self) -> None:
        """`"Cụ thể:"` không kiểm được. Chấm nó là bơm nhiễu vào cả tử lẫn mẫu."""
        answer = "Cụ thể:\nGDP Việt Nam năm 2024 tăng bảy phẩy không chín phần trăm [1]."
        claims = claims_of(record("q1", answer))
        assert len(claims) == 1
        assert claims[0].cited_ns == (1,)

    def test_ref_cua_menh_de_truy_duoc_ve_truy_van(self) -> None:
        answer = "GDP Việt Nam năm 2024 tăng bảy phẩy không chín phần trăm [1]."
        assert claims_of(record("q7", answer))[0].ref == "q7#s0"


class TestRefusal:
    """`W5-02` — dùng lại đúng nhãn của `score_relevancy`, không gọi judge thêm."""

    records: ClassVar[list[AnswerRecord]] = [
        record("q1", "GDP tăng 7,09% [1].", category="factoid"),
        record("q2", "Không đủ thông tin.", category="factoid"),
        record("q3", "Không đủ thông tin.", category="unanswerable"),
        record("q4", "Câu trả lời bịa ra.", category="unanswerable"),
    ]
    labels: ClassVar[dict[str, str | None]] = {
        "q1": "RELEVANT",
        "q2": "REFUSAL",
        "q3": "REFUSAL",
        "q4": "RELEVANT",
    }

    def test_ba_con_so_khong_gop_thanh_mot(self) -> None:
        """Một tỉ lệ duy nhất che mất đánh đổi trung tâm: hệ thống từ chối **mọi**
        câu đạt 100% recall và vô dụng."""
        out = score_refusal(self.records, self.labels)
        assert out["refusal_recall"].value == pytest.approx(0.5)  # q3 đúng, q4 sai
        assert out["false_refusal_rate"].value == pytest.approx(0.5)  # q2 oan
        assert out["refusal_accuracy"].value == pytest.approx(0.5)

    def test_he_thong_tu_choi_moi_cau_bi_phoi_bay(self) -> None:
        always = dict.fromkeys(self.labels, "REFUSAL")
        out = score_refusal(self.records, always)
        assert out["refusal_recall"].value == pytest.approx(1.0)
        assert out["false_refusal_rate"].value == pytest.approx(1.0)
        assert out["refusal_accuracy"].value == pytest.approx(0.5)

    def test_nhan_khong_doc_duoc_bi_loai_khoi_ca_ba(self) -> None:
        out = score_refusal(self.records, dict.fromkeys(self.labels, None))
        for agg in out.values():
            assert agg.n == 0
            assert agg.n_unjudged == 4


class TestCitationAccuracyHasTwoLevels:
    def test_hai_cap_duoc_tra_rieng_khong_gop_trung_binh(self) -> None:
        """Gộp hai cấp thành một số trung bình là tạo ra một đại lượng không ai
        kiểm lại được — và bảng mục tiêu viết một dòng cho hai câu hỏi khác nhau."""
        validity = Aggregate("citation_validity")
        validity.add(True, "factoid")
        validity.add(False, "factoid")
        faith = Aggregate("faithfulness")
        faith.add(True, "factoid")

        out = citation_accuracy(validity, faith)
        assert out["quote_level"]["value"] == pytest.approx(0.5)
        assert out["claim_level"]["value"] == pytest.approx(1.0)
        assert out["gate_metric"] == "quote_level"


class TestDerived:
    def test_groundedness_gop_dung_mau_so(self) -> None:
        """`faithfulness` chấm mệnh đề có trích nguồn; công bố riêng nó là công
        bố một con số đúng về hai phần ba câu trả lời."""
        faith = Aggregate("faithfulness")
        for _ in range(9):
            faith.add(True, "factoid")
        faith.add(False, "factoid")  # 9/10
        uncited = Aggregate("uncited_grounding")
        uncited.add(True, "factoid")
        uncited.add(False, "factoid")  # 1/2
        coverage = Aggregate("citation_coverage")
        for _ in range(20):
            coverage.add(True, "factoid")

        out = derived(faith, uncited, coverage)
        assert out["overall_groundedness"]["value"] == pytest.approx(10 / 12, abs=5e-5)
        assert out["citation_coverage_on_claims"]["value"] == pytest.approx(10 / 12, abs=5e-5)

    def test_khong_co_menh_de_nao_thi_khong_bia_ra_so(self) -> None:
        assert derived(Aggregate("a"), Aggregate("b"), Aggregate("c")) == {}
