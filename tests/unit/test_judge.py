"""`W5-03` — judge: cache, trần chi phí, phán quyết không đọc được, quy tắc cấm preset.

Mọi test ở đây chạy với provider giả: không key, không mạng, không tiền. Đó là
điều kiện để bộ test của judge còn chạy được ở CI, nơi `G1` đã ép rằng eval phải
chạy được với API key rỗng.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from pipeline.eval.judge import (
    Judge,
    JudgeCache,
    JudgeConfig,
    JudgeConfigError,
    JudgeQuestion,
    _canonical_key,
    _parse_verdict,
    judge_registry,
)
from rag_core.llm import (
    DEEPSEEK_PRICING,
    GLM_BASE_URL,
    GLM_PRICING,
    ChatMessage,
    CostBudget,
    LLMProvider,
    LLMResponse,
)
from rag_core.schemas import TokenUsage

FAITHFULNESS_LABELS = ("SUPPORTED", "CONTRADICTED", "NOT_FOUND")


class ScriptedProvider(LLMProvider):
    """Trả lần lượt các phản hồi đã soạn sẵn và đếm số lời gọi."""

    name = "scripted"
    model = "scripted-model"

    def __init__(
        self,
        replies: Sequence[str],
        *,
        cost: float = 0.001,
        finish_reason: str = "stop",
    ) -> None:
        self.replies = list(replies)
        self.cost = cost
        self.finish_reason = finish_reason
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
                "seed": seed,
                "extra_body": extra_body,
                "messages": list(messages),
            }
        )
        index = min(len(self.calls) - 1, len(self.replies) - 1)
        return LLMResponse(
            text=self.replies[index],
            model=self.model,
            model_requested=self.model,
            usage=TokenUsage(prompt_tokens=600, completion_tokens=40, cost_usd=self.cost),
            finish_reason=self.finish_reason,
        )


def verdict_json(label: str, reason: str = "vì ngữ cảnh nói vậy") -> str:
    return json.dumps({"verdict": label, "reason": reason}, ensure_ascii=False)


def make_judge(
    tmp_path: Path,
    replies: Sequence[str],
    **overrides: Any,
) -> tuple[Judge, ScriptedProvider]:
    config = JudgeConfig(
        model="deepseek-v4-flash",
        cache_path=tmp_path / "judge.sqlite3",
        concurrency=1,
        **overrides,
    )
    provider = ScriptedProvider(replies)
    return Judge(config, provider), provider


def question(claim: str = "GDP tăng 7,09%.", ref: str = "q1#c1") -> JudgeQuestion:
    return JudgeQuestion(
        prompt_id="judge-faithfulness",
        labels=FAITHFULNESS_LABELS,
        variables={"context": "GDP năm 2024 tăng 7,09%.", "claim": claim},
        ref=ref,
    )


# ------------------------------------------------------------------ cấu hình


def test_preset_bi_tu_choi_truoc_moi_loi_goi_mang() -> None:
    """Quy tắc cứng #1, ép bằng mã. Ném lúc dựng config, không phải lúc gọi."""
    with pytest.raises(JudgeConfigError, match="preset"):
        JudgeConfig(model="@preset/my-eval-judge")


def test_preset_bi_bat_ca_khi_nam_giua_slug() -> None:
    """`startswith` là chưa đủ — OpenRouter chấp nhận cả `openrouter/@preset/x`."""
    with pytest.raises(JudgeConfigError, match="preset"):
        JudgeConfig(model="openrouter/@preset/my-eval-judge")


def test_bi_danh_cua_deepseek_cung_bi_tu_choi() -> None:
    """Cùng lý do với preset: `deepseek-reasoner` là con trỏ phía server.

    Đây là điểm mà quy tắc của dự án được áp theo *lý do* chứ không theo tên nhà
    cung cấp — `DEEPSEEK_ALIASES` ghi rõ nó trỏ đi đâu và rằng chỗ trỏ sẽ đổi.
    """
    with pytest.raises(JudgeConfigError, match="bí danh"):
        JudgeConfig(model="deepseek-reasoner")


def test_bi_danh_dung_duoc_khi_khai_tuong_minh() -> None:
    config = JudgeConfig(model="deepseek-reasoner", allow_alias=True)
    assert config.model == "deepseek-reasoner"


def test_khong_co_duong_nao_dat_temperature_khac_khong() -> None:
    """`temperature` không phải field. Một mặc định 0 là một thứ đặt lại được."""
    assert not hasattr(JudgeConfig(model="deepseek-v4-flash"), "temperature")
    with pytest.raises(TypeError):
        JudgeConfig(model="deepseek-v4-flash", temperature=0.7)  # type: ignore[call-arg]


def test_judge_goi_model_voi_temperature_0(tmp_path: Path) -> None:
    judge, provider = make_judge(tmp_path, [verdict_json("SUPPORTED")])
    judge.ask(question())
    assert provider.calls[0]["temperature"] == 0.0
    assert provider.calls[0]["seed"] == judge.config.seed


def test_mac_dinh_la_tat_suy_luan(tmp_path: Path) -> None:
    """Mặc định do phép đo chọn, không do trực giác — xem bảng ở `JudgeConfig`.

    Suy luận bật không mua thêm một phán quyết đúng nào trên 12 mẫu, mà đổi lại
    5,4× chi phí và 5/72 lời gọi cụt ở `max_tokens`.
    """
    judge, provider = make_judge(tmp_path, [verdict_json("SUPPORTED")])
    judge.ask(question())
    assert judge.config.reasoning is False
    assert provider.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_bat_suy_luan_thi_khong_gui_extra_body(tmp_path: Path) -> None:
    judge, provider = make_judge(tmp_path, [verdict_json("SUPPORTED")], reasoning=True)
    judge.ask(question())
    assert provider.calls[0]["extra_body"] is None


def test_model_mac_dinh_la_slug_that_khong_phai_bi_danh() -> None:
    """`deepseek-reasoner` đo được là **được phục vụ bởi** `deepseek-v4-flash` —
    ghim nó là ghim một con trỏ, và trả 5,4× tiền cho cùng một model."""
    from pipeline.eval.judge import DEFAULT_JUDGE_MODEL
    from rag_core.llm import DEEPSEEK_ALIASES

    assert DEFAULT_JUDGE_MODEL not in DEEPSEEK_ALIASES
    JudgeConfig()  # dựng được mà không cần `allow_alias`


def test_max_tokens_va_concurrency_phai_hop_le() -> None:
    with pytest.raises(JudgeConfigError, match="max_tokens"):
        JudgeConfig(model="deepseek-v4-flash", max_tokens=0)
    with pytest.raises(JudgeConfigError, match="concurrency"):
        JudgeConfig(model="deepseek-v4-flash", concurrency=0)


# --------------------------------------------------------------------- cache


def test_lan_hai_tra_ve_tu_cache_khong_goi_model(tmp_path: Path) -> None:
    judge, provider = make_judge(tmp_path, [verdict_json("SUPPORTED")])
    first = judge.ask(question())
    second = judge.ask(question())
    assert len(provider.calls) == 1
    assert first.label == second.label == "SUPPORTED"
    assert first.cached is False
    assert second.cached is True
    assert second.cost_usd == 0.0
    assert judge.stats.hits == 1
    assert judge.stats.misses == 1


def test_cache_dung_chung_giua_hai_ref_khac_nhau(tmp_path: Path) -> None:
    """Cùng (mệnh đề, ngữ cảnh) ở hai truy vấn là **một** phép chấm.

    Nếu `ref` lọt vào khoá thì cùng một cặp bị chấm hai lần — vừa tốn tiền, vừa
    mở đường cho hai phán quyết khác nhau về cùng một thứ.
    """
    judge, provider = make_judge(tmp_path, [verdict_json("SUPPORTED")])
    judge.ask(question(ref="q1#c1"))
    hit = judge.ask(question(ref="q7#c3"))
    assert len(provider.calls) == 1
    assert hit.cached is True
    assert hit.ref == "q7#c3"


def test_doi_rubric_lam_mat_cache(tmp_path: Path) -> None:
    """Sửa rubric = câu hỏi khác ⇒ phán quyết cũ không dùng lại được.

    Kiểm bằng cách đổi `sha256` trong khoá: đó chính là thứ registry `W4-11` ép
    phải khớp với nội dung template.
    """
    base = {
        "prompt_spec": "judge-faithfulness@v1",
        "model": "deepseek-v4-flash",
        "max_tokens": 512,
        "seed": 1,
        "reasoning": True,
        "json_mode": True,
        "labels": FAITHFULNESS_LABELS,
        "variables": {"claim": "x", "context": "y"},
    }
    first = _canonical_key(prompt_sha256="a" * 64, **base)  # type: ignore[arg-type]
    second = _canonical_key(prompt_sha256="b" * 64, **base)  # type: ignore[arg-type]
    assert first != second


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("model", "deepseek-chat"),
        ("max_tokens", 1024),
        ("seed", 2),
        ("reasoning", False),
        ("json_mode", False),
        ("labels", ("YES", "NO")),
    ],
)
def test_moi_thu_doi_duoc_cau_tra_loi_deu_nam_trong_khoa(field_name: str, value: object) -> None:
    """Một tham số đổi được kết quả mà không nằm trong khoá = cache trả lời sai
    cho một câu hỏi khác, và không có gì báo."""
    base: dict[str, Any] = {
        "prompt_spec": "judge-faithfulness@v1",
        "prompt_sha256": "a" * 64,
        "model": "deepseek-v4-flash",
        "max_tokens": 512,
        "seed": 1,
        "reasoning": True,
        "json_mode": True,
        "labels": FAITHFULNESS_LABELS,
        "variables": {"claim": "x", "context": "y"},
    }
    changed = {**base, field_name: value}
    assert _canonical_key(**base) != _canonical_key(**changed)


def test_thu_tu_bien_khong_doi_khoa() -> None:
    """Khoá phải theo *nội dung*, không theo thứ tự dict — nếu không thì cache
    trượt ngẫu nhiên tuỳ thứ tự chèn."""
    base: dict[str, Any] = {
        "prompt_spec": "s",
        "prompt_sha256": "a" * 64,
        "model": "m",
        "max_tokens": 512,
        "seed": 1,
        "reasoning": True,
        "json_mode": True,
        "labels": ("A", "B"),
    }
    one = _canonical_key(**base, variables={"claim": "x", "context": "y"})
    two = _canonical_key(**base, variables={"context": "y", "claim": "x"})
    assert one == two


def test_frozen_cache_bien_mot_lan_trat_thanh_loi(tmp_path: Path) -> None:
    """Chế độ "tái lập lại đúng con số đã báo cáo"."""
    judge, provider = make_judge(tmp_path, [verdict_json("SUPPORTED")])
    judge.ask(question())

    frozen = Judge(
        JudgeConfig(
            model="deepseek-v4-flash",
            cache_path=tmp_path / "judge.sqlite3",
            concurrency=1,
            frozen_cache=True,
        ),
        provider,
    )
    assert frozen.ask(question()).cached is True
    with pytest.raises(JudgeConfigError, match="frozen_cache"):
        frozen.ask(question(claim="Lạm phát 3,63%."))


def test_digest_doi_khi_them_phan_quyet(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path / "judge.sqlite3")
    empty = cache.digest()
    cache.put("k1", label="SUPPORTED", reason="r", served_model="m", raw="{}", cost_usd=0.0)
    one = cache.digest()
    cache.put("k2", label="NOT_FOUND", reason="r", served_model="m", raw="{}", cost_usd=0.0)
    assert len({empty, one, cache.digest()}) == 3
    assert len(cache) == 2


def test_cache_hong_thi_dung_lai_chu_khong_lam_sap(tmp_path: Path) -> None:
    path = tmp_path / "judge.sqlite3"
    path.write_bytes(b"day khong phai file sqlite")
    cache = JudgeCache(path)
    assert len(cache) == 0


# ------------------------------------------------- phán quyết không đọc được


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "không có JSON ở đây",
        '{"verdict": ',
        '["SUPPORTED"]',
        '{"reason": "quên mất verdict"}',
        '{"verdict": 1}',
        '{"verdict": "MAYBE"}',
    ],
)
def test_cac_dang_dau_ra_khong_doc_duoc(raw: str) -> None:
    label, reason = _parse_verdict(raw, FAITHFULNESS_LABELS)
    assert label is None
    assert reason


@pytest.mark.parametrize(
    "raw",
    [
        '{"verdict": "SUPPORTED", "reason": "ok"}',
        '```json\n{"verdict": "supported", "reason": "ok"}\n```',
        'Phán quyết: {"verdict": " Supported ", "reason": "ok"}',
        '{"verdict": "SUPPORTED"}',
    ],
)
def test_cac_dang_dau_ra_doc_duoc(raw: str) -> None:
    label, _ = _parse_verdict(raw, FAITHFULNESS_LABELS)
    assert label == "SUPPORTED"


def test_nhan_ngoai_tap_khai_bao_la_khong_doc_duoc_chu_khong_phai_nhan_moi() -> None:
    """Judge trả `PARTIAL` cho một rubric ba nhãn nghĩa là nó đã trả lời một câu
    hỏi khác câu được hỏi. Nhận bừa nhãn ấy là để một nhãn không ai định nghĩa
    đi thẳng vào phép tính."""
    label, reason = _parse_verdict(verdict_json("PARTIAL"), FAITHFULNESS_LABELS)
    assert label is None
    assert "ngoài tập" in reason


def test_mot_lan_sua_roi_thoi(tmp_path: Path) -> None:
    judge, provider = make_judge(tmp_path, ["rác", verdict_json("NOT_FOUND")])
    verdict = judge.ask(question())
    assert len(provider.calls) == 2
    assert verdict.label == "NOT_FOUND"
    assert judge.stats.repairs == 1
    assert judge.stats.unparseable == 0
    # Chi phí của **cả hai** lời gọi được cộng vào, không chỉ lời gọi thành công.
    assert verdict.cost_usd == pytest.approx(0.002)


def test_cut_o_max_tokens_khong_tieu_them_mot_loi_goi_sua(tmp_path: Path) -> None:
    """⭐ Nguyên nhân đã biết chắc thì không mua lại nó lần thứ hai.

    Prompt sửa nói "hãy trả JSON" — nhưng model đang trả JSON, nó chỉ hết chỗ
    trước khi tới đó. Gọi lại với đúng `max_tokens` ấy thì cụt ở đúng chỗ ấy.
    """
    config = JudgeConfig(
        model="deepseek-v4-flash", cache_path=tmp_path / "j.sqlite3", concurrency=1
    )
    provider = ScriptedProvider(['{"verdict": "SUPPO'], finish_reason="length")
    judge = Judge(config, provider)
    verdict = judge.ask(question())
    assert len(provider.calls) == 1
    assert verdict.error == "truncated"
    assert judge.stats.truncated == 1
    assert judge.stats.repairs == 0
    assert judge.stats.unparseable == 1


def test_sua_khong_duoc_thi_thanh_unparseable_chu_khong_thanh_nhan_xau(
    tmp_path: Path,
) -> None:
    judge, provider = make_judge(tmp_path, ["rác", "vẫn rác"])
    verdict = judge.ask(question())
    assert len(provider.calls) == 2
    assert verdict.label is None
    assert verdict.judged is False
    assert verdict.error == "unparseable"
    assert judge.stats.unparseable == 1


def test_phan_quyet_khong_doc_duoc_khong_bi_ghi_cache(tmp_path: Path) -> None:
    """Nếu ghi, lần chạy sau "tái lập" đúng cái hỏng ấy mà không tốn lời gọi nào
    để phát hiện — một lỗi tạm thời bị đóng băng thành kết quả vĩnh viễn."""
    judge, _ = make_judge(tmp_path, ["rác", "vẫn rác"])
    judge.ask(question())
    assert len(judge.cache) == 0

    healthy, provider = make_judge(tmp_path, [verdict_json("SUPPORTED")])
    assert healthy.ask(question()).label == "SUPPORTED"
    assert len(provider.calls) == 1


# ---------------------------------------------------------------- trần chi phí


def test_tran_chi_phi_chan_truoc_khi_goi(tmp_path: Path) -> None:
    judge, provider = make_judge(tmp_path, [verdict_json("SUPPORTED")], cap_usd=0.0000001)
    verdict = judge.ask(question())
    assert provider.calls == []
    assert verdict.label is None
    assert verdict.error == "budget"
    assert judge.stats.budget_stopped == 1


def test_cham_tran_thi_cac_cau_con_lai_khong_goi_them(tmp_path: Path) -> None:
    """Chạm trần một lần là dừng hẳn, không phải thử-và-trượt từng câu.

    Không có chốt này thì một job 200 câu sẽ gọi `reserve` 200 lần sau khi đã
    hết tiền — vô hại về tiền, nhưng nó biến "hết ngân sách" thành 200 dòng log
    thay vì một."""
    judge, provider = make_judge(tmp_path, [verdict_json("SUPPORTED")], cap_usd=0.0000001)
    verdicts = [judge.ask(question(claim=f"mệnh đề {i}", ref=f"r{i}")) for i in range(5)]
    assert provider.calls == []
    assert all(v.error == "budget" for v in verdicts)
    assert judge.stats.budget_stopped == 1


def test_uoc_luong_chi_phi_cao_hon_thuc_te(tmp_path: Path) -> None:
    """Trần sai về phía chặn sớm thì mất vài lời gọi; sai về phía nới thì mất tiền."""
    judge, _ = make_judge(tmp_path, [verdict_json("SUPPORTED")])
    text = "x" * 3000
    estimate = judge._estimate_usd(text)
    actual_worst_case = judge.config.pricing.cost(1000, judge.config.max_tokens)
    assert estimate >= actual_worst_case


def test_budget_duoc_ghi_no_tu_ben_ngoai(tmp_path: Path) -> None:
    """Trần dùng chung được giữa nhiều judge — một job nhiều rubric vẫn có **một**
    trần, không phải mỗi rubric một trần."""
    shared = CostBudget(0.0015, name="chung")
    config = JudgeConfig(
        model="deepseek-v4-flash", cache_path=tmp_path / "j.sqlite3", concurrency=1
    )
    judge = Judge(config, ScriptedProvider([verdict_json("SUPPORTED")]), budget=shared)
    judge.ask(question(claim="một"))
    assert shared.spent_usd == pytest.approx(0.001)
    assert judge.ask(question(claim="hai")).error == "budget"


# ------------------------------------------------------------------ vận hành


def test_ask_many_giu_dung_thu_tu(tmp_path: Path) -> None:
    """Nơi gọi ghép phán quyết với mệnh đề theo chỉ số. Một danh sách trả về theo
    thứ tự hoàn thành sẽ gán nhãn của câu này cho câu khác — sai lặng lẽ, và sai
    khác nhau mỗi lần chạy."""
    labels = ["SUPPORTED", "NOT_FOUND", "CONTRADICTED", "SUPPORTED"]
    config = JudgeConfig(
        model="deepseek-v4-flash", cache_path=tmp_path / "j.sqlite3", concurrency=4
    )

    class ByClaim(ScriptedProvider):
        def complete(self, messages: Sequence[ChatMessage], **kwargs: Any) -> LLMResponse:
            tail = str(messages[0].content).rsplit("MỆNH ĐỀ:\n", 1)[1]
            index = int(tail.splitlines()[0].strip())
            self.calls.append({})
            return LLMResponse(
                text=verdict_json(labels[index]),
                model="scripted-model",
                model_requested="scripted-model",
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, cost_usd=0.0),
            )

    judge = Judge(config, ByClaim([]))
    verdicts = judge.ask_many([question(claim=str(i), ref=f"r{i}") for i in range(len(labels))])
    assert [v.label for v in verdicts] == labels
    assert [v.ref for v in verdicts] == [f"r{i}" for i in range(len(labels))]


def test_model_thuc_te_da_phuc_vu_duoc_ghi_lai(tmp_path: Path) -> None:
    judge, _ = make_judge(tmp_path, [verdict_json("SUPPORTED")])
    verdict = judge.ask(question())
    assert verdict.served_model == "scripted-model"
    assert judge.stats.served_models == {"scripted-model": 1}


def test_stats_bao_ca_ti_le_trung_cache(tmp_path: Path) -> None:
    judge, _ = make_judge(tmp_path, [verdict_json("SUPPORTED")])
    judge.ask(question())
    judge.ask(question())
    judge.ask(question())
    assert judge.stats.hit_rate == pytest.approx(2 / 3)
    assert judge.stats.as_dict()["served_models"] == {"scripted-model": 1}


def test_rubric_con_bien_chua_dien_thi_no_som(tmp_path: Path) -> None:
    """Thiếu một biến thì prompt gửi đi mang chuỗi `{{claim}}` nguyên văn, và
    judge sẽ chấm một mệnh đề rỗng — rồi trả về một nhãn trông hợp lệ."""
    judge, _ = make_judge(tmp_path, [verdict_json("SUPPORTED")])
    with pytest.raises(KeyError, match="claim"):
        judge.ask(
            JudgeQuestion(
                prompt_id="judge-faithfulness",
                labels=FAITHFULNESS_LABELS,
                variables={"context": "chỉ có ngữ cảnh"},
            )
        )


def test_tap_nhan_rong_hoac_trung_bi_tu_choi() -> None:
    with pytest.raises(ValueError, match="tập nhãn"):
        JudgeQuestion(prompt_id="x", labels=(), variables={})
    with pytest.raises(ValueError, match="trùng"):
        JudgeQuestion(prompt_id="x", labels=("A", "A"), variables={})


# ------------------------------------------------------------------- rubric


def test_rubric_faithfulness_nap_duoc_va_khop_hash() -> None:
    """Rubric đi qua đúng registry của `W4-11`: sửa nội dung mà quên tăng version
    thì loader từ chối nạp, và version ấy nằm trong khoá cache."""
    prompt = judge_registry().get("judge-faithfulness")
    assert prompt.spec.startswith("judge-faithfulness@v")
    assert "{{context}}" in prompt.text
    assert "{{claim}}" in prompt.text
    for label in FAITHFULNESS_LABELS:
        assert label in prompt.text


def test_rubric_noi_ro_ngu_canh_la_du_lieu_khong_phai_chi_thi() -> None:
    """`W4-12` đo được rằng chính mấy dòng chữ này làm gần hết phần việc chống
    tiêm. Judge đọc nguyên văn chunk corpus, nên nó cần đúng lớp bảo vệ ấy."""
    text = judge_registry().get("judge-faithfulness").text
    assert "dữ liệu" in text and "chỉ thị" in text


# ------------------------------------------------- họ model (W5-04, cross-check)


def test_ho_suy_ra_tu_slug_chu_khong_phai_mot_field_khai_rieng() -> None:
    """Không có đường nào khai `family` lệch với `model` — vì đó là lỗi câm.

    DeepSeek **nhận** `reasoning_effort` rồi bỏ qua (đo ở `W3-04`): một config
    khai sai họ vẫn trả phán quyết, vẫn ghi cache, chỉ là dưới điều kiện khác
    lời khai. Suy ra từ slug thì trường hợp ấy không dựng lên được.
    """
    assert not hasattr(JudgeConfig(), "family_override")
    assert JudgeConfig().family == "deepseek"
    assert JudgeConfig(model="glm-5.3-flash", base_url=GLM_BASE_URL).family == "glm"


def test_model_la_de_thi_bao_loi_chu_khong_am_tham_tinh_gia_0() -> None:
    with pytest.raises(JudgeConfigError, match="chưa biết họ"):
        _ = JudgeConfig(model="mistral-large-2411", base_url="https://x").family


def test_moi_ho_lay_dung_bang_gia_cua_no() -> None:
    """Sai bảng giá không làm gì đỏ — nó chỉ làm báo cáo chi phí sai."""
    assert JudgeConfig().pricing == DEEPSEEK_PRICING["deepseek-v4-flash"]
    glm = JudgeConfig(model="glm-5.3-flash", base_url=GLM_BASE_URL)
    assert glm.pricing == GLM_PRICING["glm-5.3-flash"]
    assert glm.pricing != JudgeConfig().pricing


def test_glm_khong_tat_duoc_suy_luan_nen_hai_nhanh_khong_doi_xung() -> None:
    """`reasoning=False` với GLM nghĩa là **thấp nhất cho phép**, không phải tắt.

    `glm-5.3-flash` trả HTTP 400 khi bị yêu cầu tắt. Bất đối xứng này đi thẳng
    vào cách đọc phần cross-check của `W5-04`, nên nó phải có một bài test giữ.
    """
    deepseek = JudgeConfig()
    glm = JudgeConfig(model="glm-5.3-flash", base_url=GLM_BASE_URL)
    assert deepseek.reasoning_body == {"thinking": {"type": "disabled"}}
    assert glm.reasoning_body == {"reasoning_effort": "low"}
    assert "thinking" not in (glm.reasoning_body or {})
    assert JudgeConfig(reasoning=True).reasoning_body is None


def test_ho_khac_deepseek_ma_quen_doi_base_url_thi_do_ngay(tmp_path: Path) -> None:
    """Không chặn thì lỗi rơi xuống thành 404 **giữa** một lần chấm dở tiền."""
    with pytest.raises(JudgeConfigError, match="base_url"):
        JudgeConfig(model="glm-5.3-flash", cache_path=tmp_path / "c.sqlite3")


def test_ho_khong_can_vao_khoa_cache_vi_model_da_o_do() -> None:
    """Thêm `family` vào khoá sẽ **huỷ sạch** 1664 phán quyết của `W5-01`.

    Và nó không mua gì: họ là hàm của model, mà model đã nằm trong khoá.
    """
    import inspect

    from pipeline.eval.judge import _canonical_key

    assert "family" not in inspect.signature(_canonical_key).parameters
    assert "model" in inspect.signature(_canonical_key).parameters
