"""Interface của tầng LLM — dùng chung cho pipeline (sinh dữ liệu, judge) và serving.

Ba thứ được ép ở tầng interface vì thiếu chúng thì không tái lập được kết quả,
mà lại không có triệu chứng gì khi thiếu:

1. **`model` trả về là model đã thực sự phục vụ request**, đọc từ response chứ
   không phải model mình yêu cầu. Router có fallback; một metric dịch chuyển vì
   âm thầm rơi sang model khác là loại bug tốn nhiều ngày nhất để truy.
2. **`temperature` mặc định là 0.** Mọi thứ nằm trên đường eval phải xác định.
   Muốn sampling thì phải là lựa chọn có ý thức ở nơi gọi.
3. **`usage` luôn có mặt**, kèm chi phí quy ra USD. Không đo được tiền thì không
   trả lời được câu "cải thiện 3% recall này giá bao nhiêu".

Quy tắc cứng của dự án: **không dùng OpenRouter preset (`@preset/...`)** ở bất
kỳ đâu trên đường eval. Preset là cấu hình phía server, đổi lúc nào không biết,
và khi đổi thì con số cũ không còn so được với số mới. Luôn ghim slug tường minh.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..schemas import TokenUsage

__all__ = ["ChatMessage", "LLMError", "LLMProvider", "LLMResponse", "ModelPricing"]

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Provider trả lỗi không thể tự khắc phục bằng cách thử lại."""


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ModelPricing(BaseModel):
    """Giá theo 1 triệu token, khai báo tường minh chứ không hard-code trong code.

    Giá của provider đổi theo thời gian. Ghi vào config và log lại cùng kết quả
    thì báo cáo chi phí cũ vẫn đọc được sau khi giá đã đổi.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_per_1m_usd: float = Field(default=0.0, ge=0.0)
    output_per_1m_usd: float = Field(default=0.0, ge=0.0)
    cached_input_per_1m_usd: float | None = Field(default=None, ge=0.0)

    def cost(self, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
        billable_input = max(0, prompt_tokens - cached_tokens)
        cached_rate = (
            self.cached_input_per_1m_usd
            if self.cached_input_per_1m_usd is not None
            else self.input_per_1m_usd
        )
        return (
            billable_input * self.input_per_1m_usd
            + cached_tokens * cached_rate
            + completion_tokens * self.output_per_1m_usd
        ) / 1_000_000


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    model: str = Field(min_length=1)
    """Model **thực tế** đã phục vụ request, lấy từ response của provider."""

    model_requested: str = Field(min_length=1)
    usage: TokenUsage
    finish_reason: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def model_drifted(self) -> bool:
        """`True` khi provider phục vụ bằng model khác model đã yêu cầu."""
        return self.model.split(":")[0] != self.model_requested.split(":")[0]


class LLMProvider(ABC):
    """Nguồn sinh text. Mọi provider đều phải tuân đúng ba điều ở docstring module."""

    name: str
    model: str

    @abstractmethod
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
        """Gọi model một lượt. `temperature=0` là mặc định, cố ý.

        `extra_body` trộn thẳng vào payload để chạm tới tham số **ngoài chuẩn**
        của từng provider. Có mặt vì một lý do đo được: model lai suy luận tiêu
        phần lớn `max_tokens` vào chuỗi suy luận không xuất hiện trong `content`
        — dry-run `W3-04` đo **83%** (9.003/10.885 token) và **6/30 request trả
        rỗng** vì lý do đó. Tắt suy luận là tham số riêng của từng nhà: vLLM/Qwen3
        dùng `chat_template_kwargs`, DeepSeek dùng tên khác. Không có đường
        truyền tham số thô thì không tắt được, và cái giá là trả tiền cho phần
        suy luận rồi vứt đi.

        ⚠️ Tham số đi qua đây **không** được kiểm; nó vẫn nằm trên đường eval nên
        phải ghi vào báo cáo của lần chạy.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"
