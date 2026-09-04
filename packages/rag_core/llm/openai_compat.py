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

import asyncio
import json
import logging
import random
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ..schemas import TokenUsage
from .base import ChatMessage, LLMChunk, LLMError, LLMProvider, LLMResponse, ModelPricing

if TYPE_CHECKING:
    import httpx

__all__ = ["OpenAICompatProvider", "PermanentLLMError"]

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class PermanentLLMError(LLMError):
    """4xx **không** thử lại được: request của mình sai, không phải nhà cung cấp chết.

    Từ ngoài nhìn vào, "sai tên model" và "provider sập" trước đây là cùng một
    `LLMError`, và `W4-08` cần tách chúng: mở cầu dao vì một request sai của
    chính mình là **tự cắt** nhà cung cấp chính trong 30 giây cho một lỗi mà
    thử lại bao nhiêu lần cũng thế.

    Vẫn là `LLMError`, nên mọi `except LLMError` đã viết trước đây không đổi
    hành vi.
    """


class _Transient(LLMError):
    """Lỗi *tạm thời* của đường stream — thử lại được, và chỉ trước token đầu.

    Tồn tại vì đường stream không dùng chung `_post_with_retry`, nên nếu không
    phân loại thì một `LLMError` "sai tên model" (400) sẽ bị thử lại 5 lần với
    backoff — mất gần một phút để báo một lỗi đã biết chắc từ lần đầu. Đó đúng
    là cái mà `_RETRYABLE_STATUS` chặn ở đường kia.
    """


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
        async_client: httpx.AsyncClient | None = None,
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
        self._async_client = async_client
        self._extra_headers = extra_headers or {}

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    @property
    def async_client(self) -> httpx.AsyncClient:
        """Client riêng cho đường stream — **không** dùng chung với `client`.

        `httpx.Client` và `httpx.AsyncClient` là hai lớp khác nhau; và kể cả nếu
        không, một tiến trình serving vừa gọi `complete()` đồng bộ trong
        threadpool vừa `astream()` trên vòng lặp sự kiện sẽ tranh nhau cùng một
        pool kết nối từ hai mô hình đồng thời.
        """
        if self._async_client is None:
            import httpx

            # `read` rộng hơn `connect` rất nhiều là có chủ đích: giữa hai token
            # của một model đang suy luận có thể im lặng vài chục giây, còn một
            # kết nối không mở nổi trong 10 s thì thử lại là đúng.
            self._async_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout, connect=10.0)
            )
        return self._async_client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

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

    # ---------------------------------------------------------------- stream

    async def astream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        """Cài `StreamingLLM` — `W4-06`.

        ## ⭐⭐ Retry ở đây chỉ hợp lệ **trước token đầu tiên**

        `_post_with_retry` thử lại tối đa 5 lần vì ở đường không-stream một lần
        thử lại là vô hại: người gọi chưa thấy gì cả. Ở đây thì không. Sau khi
        đã `yield` một mẩu, người gọi có thể đã đẩy nó qua SSE tới trình duyệt —
        và thử lại nghĩa là **sinh lại từ đầu**, tức người dùng thấy nửa câu trả
        lời này nối vào nửa câu trả lời kia. Kết quả là một đoạn văn trôi chảy,
        mạch lạc, và không phải cái mà model nào nói ra.

        Nên có đúng hai chế độ, và ranh giới là mẩu đầu tiên:

        * chưa `yield` gì ⇒ thử lại được (429/5xx/timeout — ca thường gặp nhất,
          vì lỗi hạn mức xảy ra lúc mở kết nối)
        * đã `yield` ⇒ `LLMError`, không thử lại, và phần đã sinh **không** bị
          vứt: người gọi giữ nó và đánh dấu câu trả lời là dở dang.

        ## `stream_options.include_usage`

        Thiếu tham số này thì OpenAI-compat trả về stream **không có `usage`**,
        và mọi con số chi phí của đường serving thành 0 — một cách im lặng, vì
        `TokenUsage(0, 0)` hợp lệ về hình dạng. Đó đúng là chế độ hỏng mà quy tắc
        #3 của module `base` tồn tại để chặn.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra_body:
            payload.update(extra_body)

        started = time.perf_counter()
        emitted = False
        parts: list[str] = []
        served_model = self.model
        finish_reason: str | None = None
        usage: dict[str, Any] = {}
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                async for event in self._raw_events(payload):
                    choices = event.get("choices") or []
                    if choices:
                        choice = choices[0]
                        piece = (choice.get("delta") or {}).get("content") or ""
                        finish_reason = choice.get("finish_reason") or finish_reason
                        if piece:
                            emitted = True
                            parts.append(piece)
                            yield LLMChunk(delta=piece)
                    # `model` và `usage` đến ở những mẩu khác nhau, và mẩu mang
                    # `usage` của OpenAI-compat có `choices` **rỗng** — nên đọc
                    # chúng ngoài nhánh trên, không phải trong.
                    served_model = str(event.get("model") or served_model)
                    if event.get("usage"):
                        usage = dict(event["usage"])
            except _Transient as exc:
                if emitted:
                    raise LLMError(
                        f"{self.model} đứt stream sau {len(''.join(parts))} ký tự: {exc}. "
                        "Không thử lại: sinh lại từ đầu sẽ nối hai câu trả lời khác nhau."
                    ) from exc
                last_error = exc
                logger.warning(
                    "stream %s hỏng trước token đầu (lần %d): %s", self.model, attempt + 1, exc
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(min(30.0, 2.0**attempt) * (0.5 + random.random()))
                    continue
                raise LLMError(
                    f"Mở stream {self.model} thất bại sau {self._max_retries + 1} lần: {last_error}"
                ) from last_error
            else:
                break

        latency_ms = (time.perf_counter() - started) * 1000.0
        body = {
            "choices": [{"message": {"content": "".join(parts)}, "finish_reason": finish_reason}],
            "model": served_model,
            "usage": usage,
        }
        # Đi qua đúng `_parse` của đường không-stream: `prompt_cache_hit_tokens`
        # của DeepSeek, `reasoning_tokens`, và cảnh báo model trôi đều là logic
        # **đã có** — chép lại nó ở đây là tạo ra hai định nghĩa của cùng một
        # con số, và cái thứ hai sẽ là cái không ai sửa khi giá đổi.
        yield LLMChunk(final=self._parse(body, latency_ms))

    async def _raw_events(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Đọc `text/event-stream` thành từng object JSON.

        ⚠️ `aiter_lines()` chứ không `aiter_bytes()`: httpx đã lo phần ghép mẩu
        TCP bị cắt giữa dòng. Tự ghép bằng tay ở đây là chỗ mà một token tiếng
        Việt bị cắt đôi giữa hai byte UTF-8 sẽ hỏng — hiếm, và chỉ hỏng dưới tải.
        """
        import httpx

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **self._extra_headers,
        }
        try:
            async with self.async_client.stream(
                "POST", url, json=payload, headers=headers
            ) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", "replace")[:500]
                    message = f"{self.model} trả HTTP {response.status_code}: {detail}"
                    if response.status_code in _RETRYABLE_STATUS:
                        raise _Transient(message)
                    # 4xx khác là lỗi của mình (sai key, sai slug, prompt quá
                    # dài). Cùng lý lẽ với `_post_with_retry`, và ở đây nó còn
                    # đắt hơn: người dùng đang nhìn một ô chat trống.
                    raise PermanentLLMError(message)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        parsed = json.loads(data)
                    except json.JSONDecodeError:
                        logger.warning("bỏ qua mẩu SSE không phải JSON: %r", data[:200])
                        continue
                    if isinstance(parsed, dict):
                        yield parsed
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise _Transient(f"lỗi mạng khi stream {self.model}: {exc}") from exc

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
                    raise PermanentLLMError(
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
