"""Client cho mọi API tương thích OpenAI `/chat/completions` (DeepSeek, OpenRouter).

Cố ý tự viết bằng `httpx` thay vì kéo SDK `openai`:

* Chỉ dùng đúng một endpoint. SDK mang theo cả một cây phụ thuộc và một lớp
  trừu tượng nữa để gỡ khi có lỗi.
* Cần **kiểm soát chính xác** phần retry và phần đọc `usage`. `prompt_cache_hit`
  của DeepSeek là field riêng, không có trong schema chuẩn — mà bỏ qua nó thì
  báo cáo chi phí sai vài lần.
* Log được nguyên response khi parse hỏng, không bị SDK nuốt mất.

Retry chỉ áp cho lỗi **tạm thời** (429, 5xx, timeout, đứt kết nối). 4xx khác là
lỗi của mình — thử lại chỉ tốn tiền và làm chậm việc phát hiện.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ..schemas import TokenUsage
from .base import ChatMessage, LLMError, LLMProvider, LLMResponse, ModelPricing

if TYPE_CHECKING:
    import httpx

__all__ = ["OpenAICompatProvider"]

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class OpenAICompatProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str,
        pricing: ModelPricing | None = None,
        timeout: float = 120.0,
        max_retries: int = 4,
        client: httpx.Client | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        if model.startswith("@preset/"):
            # Quy tắc cứng #1 của dự án, ép bằng code chứ không bằng lời nhắc:
            # preset là cấu hình phía server, đổi lúc nào không ai biết, và khi
            # đổi thì mọi con số eval đo trước đó không còn so được với số sau.
            raise LLMError(
                f"Không được dùng OpenRouter preset trên đường eval: {model!r}. "
                "Ghim slug model tường minh (ví dụ `deepseek/deepseek-chat`) để "
                "kết quả tái lập được."
            )
        self.model = model
        self.name = f"{base_url.rstrip('/').removeprefix('https://')}:{model}"
        self.pricing = pricing or ModelPricing()
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = client
        self._extra_headers = extra_headers or {}

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------ gọi

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
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if seed is not None:
            payload["seed"] = seed
        if extra_body:
            payload.update(extra_body)

        started = time.perf_counter()
        body = self._post_with_retry("/chat/completions", payload)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return self._parse(body, latency_ms)

    def _post_with_retry(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import httpx

        url = f"{self._base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self.client.post(url, json=payload, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                logger.warning("Lỗi mạng khi gọi %s (lần %d): %s", self.model, attempt + 1, exc)
            else:
                if response.status_code < 400:
                    parsed: dict[str, Any] = response.json()
                    return parsed
                if response.status_code not in _RETRYABLE_STATUS:
                    # 4xx khác là lỗi của mình (sai key, sai tên model, prompt
                    # quá dài). Thử lại chỉ tốn tiền và làm chậm việc phát hiện.
                    raise LLMError(
                        f"{self.model} trả HTTP {response.status_code}: {response.text[:500]}"
                    )
                last_error = LLMError(f"HTTP {response.status_code}: {response.text[:200]}")
                logger.warning(
                    "%s trả HTTP %d (lần %d/%d)",
                    self.model,
                    response.status_code,
                    attempt + 1,
                    self._max_retries + 1,
                )

            if attempt < self._max_retries:
                # Backoff có jitter: nhiều worker cùng bị 429 mà chờ đúng bằng
                # nhau thì lần thử lại cũng dồn vào cùng một thời điểm.
                delay = min(30.0, 2.0**attempt) * (0.5 + random.random())
                time.sleep(delay)

        raise LLMError(
            f"Gọi {self.model} thất bại sau {self._max_retries + 1} lần: {last_error}"
        ) from last_error

    def _parse(self, body: dict[str, Any], latency_ms: float) -> LLMResponse:
        try:
            choice = body["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Response không đúng dạng chat/completions: {body!r}"[:800]) from exc

        raw_usage = body.get("usage") or {}
        prompt_tokens = int(raw_usage.get("prompt_tokens", 0))
        completion_tokens = int(raw_usage.get("completion_tokens", 0))
        # DeepSeek báo phần prompt được cache ở field riêng và tính giá rẻ hơn
        # nhiều. Bỏ qua nó thì báo cáo chi phí cao hơn thực tế vài lần.
        cached_tokens = int(
            raw_usage.get("prompt_cache_hit_tokens")
            or (raw_usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            or 0
        )

        # Model suy luận (deepseek-v4-flash) tiêu phần lớn `completion_tokens`
        # vào chuỗi suy luận KHÔNG xuất hiện trong `content`. Không ghi lại con
        # số này thì `max_tokens` trông như thừa thãi trong khi content bị cắt cụt.
        reasoning_tokens = int(
            (raw_usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        )

        served_model = str(body.get("model") or self.model)
        response = LLMResponse(
            text=text,
            model=served_model,
            model_requested=self.model,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=self.pricing.cost(prompt_tokens, completion_tokens, cached_tokens),
            ),
            finish_reason=choice.get("finish_reason"),
            latency_ms=latency_ms,
            raw={"cached_tokens": cached_tokens, "reasoning_tokens": reasoning_tokens},
        )
        if response.model_drifted:
            logger.warning(
                "Model trôi: yêu cầu %r nhưng được phục vụ bởi %r. "
                "Số đo của lần chạy này không so được với lần trước.",
                response.model_requested,
                response.model,
            )
        return response
