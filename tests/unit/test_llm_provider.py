"""W1-10 — client LLM tương thích OpenAI.

Chạy bằng `httpx.MockTransport` nên không chạm mạng và không tốn tiền. Thứ được
kiểm ở đây không phải "gọi được API" mà là ba đảm bảo khiến kết quả eval tin
được: model thực tế đã phục vụ, chi phí tính đúng, và retry đúng chỗ.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from rag_core.llm import (
    DEEPSEEK_ALIASES,
    DEEPSEEK_PRICING,
    DEFAULT_DEEPSEEK_MODEL,
    ChatMessage,
    LLMError,
    ModelPricing,
    OpenAICompatProvider,
    build_deepseek_provider,
)

MESSAGES = [ChatMessage(role="user", content="xin chào")]


def _body(
    text: str = "ok",
    *,
    model: str = "deepseek-chat",
    prompt_tokens: int = 1000,
    completion_tokens: int = 100,
    cached: int | None = None,
) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    if cached is not None:
        usage["prompt_cache_hit_tokens"] = cached
    return {
        "model": model,
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": usage,
    }


def _provider(handler: Any, **kwargs: Any) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        kwargs.pop("model", "deepseek-chat"),
        api_key="test-key",
        base_url="https://api.example.com",
        pricing=kwargs.pop("pricing", DEEPSEEK_PRICING["deepseek-chat"]),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


class TestRequestShape:
    def test_sends_temperature_zero_by_default(self) -> None:
        """Mặc định phải xác định. Sampling là lựa chọn có ý thức ở nơi gọi."""
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=_body())

        _provider(handler).complete(MESSAGES)
        assert seen["temperature"] == 0.0
        assert seen["stream"] is False

    def test_passes_seed_and_json_mode(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=_body())

        _provider(handler).complete(MESSAGES, json_mode=True, seed=42, max_tokens=500)
        assert seen["response_format"] == {"type": "json_object"}
        assert seen["seed"] == 42
        assert seen["max_tokens"] == 500

    def test_sends_bearer_token(self) -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers["Authorization"]
            return httpx.Response(200, json=_body())

        _provider(handler).complete(MESSAGES)
        assert seen["auth"] == "Bearer test-key"


class TestPresetGuard:
    def test_rejects_openrouter_preset(self) -> None:
        """Quy tắc cứng #1 của dự án, ép bằng code chứ không bằng lời nhắc.

        Preset là cấu hình phía server: nó đổi lúc nào không ai biết, và khi đổi
        thì mọi con số eval đo trước đó không còn so được với số sau.
        """
        with pytest.raises(LLMError, match="preset"):
            OpenAICompatProvider(
                "@preset/my-luna-pro", api_key="k", base_url="https://openrouter.ai/api/v1"
            )

    def test_allows_explicit_slug(self) -> None:
        provider = OpenAICompatProvider(
            "deepseek/deepseek-chat", api_key="k", base_url="https://openrouter.ai/api/v1"
        )
        assert provider.model == "deepseek/deepseek-chat"


class TestCost:
    def test_computes_cost_from_pricing_table(self) -> None:
        pricing = ModelPricing(input_per_1m_usd=1.0, output_per_1m_usd=2.0)
        response = _provider(
            lambda r: httpx.Response(
                200, json=_body(prompt_tokens=1_000_000, completion_tokens=500_000)
            ),
            pricing=pricing,
        ).complete(MESSAGES)
        assert response.usage.cost_usd == pytest.approx(1.0 + 1.0)

    def test_cached_prompt_tokens_are_billed_at_the_cheaper_rate(self) -> None:
        """DeepSeek báo phần prompt được cache ở field riêng và tính rẻ hơn nhiều.

        Bỏ qua nó thì báo cáo chi phí cao hơn thực tế vài lần — mà chi phí là
        một nửa của câu hỏi 'cải thiện này có đáng không'.
        """
        pricing = ModelPricing(
            input_per_1m_usd=10.0, output_per_1m_usd=0.0, cached_input_per_1m_usd=1.0
        )
        response = _provider(
            lambda r: httpx.Response(
                200,
                json=_body(prompt_tokens=1_000_000, completion_tokens=0, cached=900_000),
            ),
            pricing=pricing,
        ).complete(MESSAGES)
        # 100k token tính giá đầy đủ + 900k token tính giá cache
        assert response.usage.cost_usd == pytest.approx(0.1 * 10.0 + 0.9 * 1.0)

    def test_reads_openai_style_cached_tokens(self) -> None:
        pricing = ModelPricing(input_per_1m_usd=10.0, cached_input_per_1m_usd=0.0)
        body = _body(prompt_tokens=1_000_000, completion_tokens=0)
        body["usage"]["prompt_tokens_details"] = {"cached_tokens": 1_000_000}
        response = _provider(lambda r: httpx.Response(200, json=body), pricing=pricing).complete(
            MESSAGES
        )
        assert response.usage.cost_usd == pytest.approx(0.0)

    def test_pricing_declared_for_every_model_the_project_may_request(self) -> None:
        assert set(DEEPSEEK_PRICING) >= {"deepseek-chat", "deepseek-v4-flash", "deepseek-reasoner"}

    def test_default_model_has_pricing(self) -> None:
        """Mặc định mà thiếu giá thì mọi báo cáo chi phí âm thầm báo $0."""
        assert DEFAULT_DEEPSEEK_MODEL in DEEPSEEK_PRICING


class TestModelDrift:
    def test_reports_model_actually_served(self) -> None:
        response = _provider(
            lambda r: httpx.Response(200, json=_body(model="deepseek-chat-0324"))
        ).complete(MESSAGES)
        assert response.model == "deepseek-chat-0324"
        assert response.model_requested == "deepseek-chat"

    def test_flags_drift_when_a_different_family_answers(self) -> None:
        """Router có fallback; một metric dịch chuyển vì rơi sang model khác là
        loại bug tốn nhiều ngày nhất để truy."""
        response = _provider(
            lambda r: httpx.Response(200, json=_body(model="gpt-4o-mini"))
        ).complete(MESSAGES)
        assert response.model_drifted

    def test_no_drift_when_same_model(self) -> None:
        response = _provider(lambda r: httpx.Response(200, json=_body())).complete(MESSAGES)
        assert not response.model_drifted


class TestRetry:
    def test_retries_on_429_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("rag_core.llm.openai_compat.time.sleep", lambda _: None)
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(429, text="rate limited")
            return httpx.Response(200, json=_body("thành công"))

        response = _provider(handler).complete(MESSAGES)
        assert response.text == "thành công"
        assert calls["n"] == 3

    def test_does_not_retry_on_400(self) -> None:
        """4xx không phải 429 là lỗi của mình — thử lại chỉ tốn tiền và làm chậm
        việc phát hiện sai key hay sai tên model."""
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(400, text="model không tồn tại")

        with pytest.raises(LLMError, match="400"):
            _provider(handler).complete(MESSAGES)
        assert calls["n"] == 1

    def test_gives_up_after_max_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("rag_core.llm.openai_compat.time.sleep", lambda _: None)
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503, text="quá tải")

        with pytest.raises(LLMError, match="thất bại sau"):
            _provider(handler, max_retries=2).complete(MESSAGES)
        assert calls["n"] == 3

    def test_retries_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("rag_core.llm.openai_compat.time.sleep", lambda _: None)
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ReadTimeout("hết giờ")
            return httpx.Response(200, json=_body())

        assert _provider(handler).complete(MESSAGES).text == "ok"


class TestMalformedResponse:
    def test_raises_on_missing_choices(self) -> None:
        with pytest.raises(LLMError, match="không đúng dạng"):
            _provider(lambda r: httpx.Response(200, json={"model": "x"})).complete(MESSAGES)

    def test_tolerates_missing_usage(self) -> None:
        body = {"model": "deepseek-chat", "choices": [{"message": {"content": "hi"}}]}
        response = _provider(lambda r: httpx.Response(200, json=body)).complete(MESSAGES)
        assert response.usage.total_tokens == 0
        assert response.usage.cost_usd == 0.0

    def test_tolerates_null_content(self) -> None:
        body = _body()
        body["choices"][0]["message"]["content"] = None
        response = _provider(lambda r: httpx.Response(200, json=body)).complete(MESSAGES)
        assert response.text == ""


class TestFactory:
    def test_deepseek_factory_attaches_pricing(self) -> None:
        provider = build_deepseek_provider("deepseek-chat", api_key="k")
        assert provider.pricing == DEEPSEEK_PRICING["deepseek-chat"]

    def test_default_model_is_a_real_slug_not_an_alias(self) -> None:
        """`deepseek-chat` là bí danh do server nắm — cùng loại vấn đề với
        OpenRouter preset, chỉ kín đáo hơn vì tên trông như một model cụ thể.

        Đo bằng bí danh thì con số tháng này không so được với tháng sau, và
        không có gì báo. Xác nhận thật trên API 2026-08-17: cả `deepseek-chat`
        lẫn `deepseek-reasoner` đều được phục vụ bởi `deepseek-v4-flash`.
        """
        assert DEFAULT_DEEPSEEK_MODEL not in DEEPSEEK_ALIASES

    def test_unknown_model_gets_zero_pricing_not_a_crash(self) -> None:
        """Model lạ vẫn gọi được, chỉ là chi phí báo 0 — thà báo 0 rõ ràng còn
        hơn đoán giá sai rồi ghi vào báo cáo."""
        provider = build_deepseek_provider("deepseek-experimental", api_key="k")
        assert provider.pricing.cost(1_000_000, 1_000_000) == 0.0
