"""`W3-04` — ba backend, và những thứ phải khai đủ cho **từng** cái.

Nhóm test này tồn tại vì một lỗi cụ thể: `--backend glm` mà quên `--model` gửi
slug mặc định `Qwen/Qwen3-8B` sang Z.ai và nhận 20 lần `HTTP 400 modelCode: does
not exist`. Ở laptop thì vô hại; trên GPU thuê thì đó là tiền thuê đổi lấy một
file lỗi. Bất kỳ bảng nào khai theo backend đều phải phủ **đủ** danh sách
backend, và đó là thứ kiểm được mà không cần gọi mạng.
"""

from __future__ import annotations

import pytest

from pipeline.indexing.contextualize import (
    DEFAULT_MODEL,
    MIN_REASONING,
    VLLM_BASE_URL,
    build_provider,
)
from rag_core.llm import DEEPSEEK_PRICING, GLM_BASE_URL, GLM_PRICING

BACKENDS = ("glm", "deepseek", "vllm")


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_backend_has_a_default_model(backend: str) -> None:
    assert DEFAULT_MODEL[backend]


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_backend_declares_how_to_minimise_reasoning(backend: str) -> None:
    """Bỏ sót một backend ở đây = trả tiền cho chuỗi suy luận rồi vứt đi.

    Đo được mức đó đắt cỡ nào: DeepSeek 83% completion token, GLM 68%
    (165/243). Không có gì đỏ khi bảng thiếu một dòng.
    """
    assert MIN_REASONING[backend]


def test_backend_tables_cover_exactly_the_cli_choices() -> None:
    """Thêm backend mà quên một bảng thì `KeyError` xảy ra lúc chạy, không lúc test."""
    assert set(DEFAULT_MODEL) == set(BACKENDS)
    assert set(MIN_REASONING) == set(BACKENDS)


def test_glm_and_deepseek_minimise_reasoning_by_DIFFERENT_parameters() -> None:
    """⚠️ Đặt nhầm nhà thì tham số **được nhận và không có tác dụng**, hoặc bị từ chối.

    Đo 2026-09-03: `chat_template_kwargs` với DeepSeek → vẫn 159 token suy luận
    (nhận rồi bỏ qua). `thinking={"type":"disabled"}` với GLM → HTTP 400 mã 1210
    *"This model always engages in thinking and cannot be disabled"*. Nên hai
    dòng này **phải** khác nhau; giống nhau nghĩa là một trong hai đang sai.
    """
    assert MIN_REASONING["glm"] != MIN_REASONING["deepseek"]
    assert MIN_REASONING["glm"] == {"reasoning_effort": "low"}
    assert "thinking" in MIN_REASONING["deepseek"]


def test_glm_cannot_be_asked_to_disable_thinking_entirely() -> None:
    """GLM-5.3-Flash chỉ nhận `low`/`high`/`max`. Tên `MIN_REASONING` phản ánh đúng thế."""
    assert MIN_REASONING["glm"]["reasoning_effort"] in {"low", "high", "max"}


def test_glm_pricing_is_the_published_list_price_not_the_promotion() -> None:
    """Khuyến mại $0,075/1M input hết hạn 2026-09-09.

    Ghi giá khuyến mại vào bảng thì mọi báo cáo chi phí đọc sai ngay hôm sau mà
    không ai sửa lại con số.
    """
    glm = GLM_PRICING["glm-5.3-flash"]
    assert (glm.input_per_1m_usd, glm.output_per_1m_usd, glm.cached_input_per_1m_usd) == (
        0.15,
        0.50,
        0.03,
    )


def test_glm_is_cheaper_than_deepseek_on_every_axis() -> None:
    """Ghim lý do đổi provider. Nếu bảng giá đổi làm điều này sai thì phải biết."""
    glm = GLM_PRICING["glm-5.3-flash"]
    deepseek = DEEPSEEK_PRICING["deepseek-chat"]
    assert glm.input_per_1m_usd < deepseek.input_per_1m_usd
    assert glm.output_per_1m_usd < deepseek.output_per_1m_usd
    assert glm.cached_input_per_1m_usd is not None
    assert deepseek.cached_input_per_1m_usd is not None
    assert glm.cached_input_per_1m_usd < deepseek.cached_input_per_1m_usd


def test_glm_provider_uses_zai_endpoint_and_glm_pricing() -> None:
    provider = build_provider("glm", model="glm-5.3-flash", api_key="test-key")
    assert GLM_BASE_URL.removeprefix("https://") in provider.name
    assert provider.pricing == GLM_PRICING["glm-5.3-flash"]  # type: ignore[attr-defined]


def test_vllm_provider_needs_no_api_key_and_prices_at_zero() -> None:
    """Quy tắc cứng #2 ở tầng code: đường chạy trên máy thuê không nhận key.

    Giá 0 vì pod tính theo giờ chứ không theo token — `--gpu-hourly-usd` mới là
    chỗ quy thời gian thành tiền.
    """
    provider = build_provider("vllm", model="Qwen/Qwen3-8B")
    assert VLLM_BASE_URL.removeprefix("http://") in provider.name
    assert provider.pricing.input_per_1m_usd == 0.0  # type: ignore[attr-defined]


@pytest.mark.parametrize("backend", ["glm", "deepseek"])
def test_paid_backends_refuse_to_start_without_a_key(backend: str) -> None:
    with pytest.raises(SystemExit, match="API_KEY"):
        build_provider(backend, model=DEFAULT_MODEL[backend], api_key="")


def test_unknown_backend_names_the_valid_ones() -> None:
    with pytest.raises(SystemExit, match="glm"):
        build_provider("openai", model="gpt-4", api_key="k")
