"""Bộ đếm token của **generator**, tách khỏi bộ đếm của embedding model.

Vì sao phải là hai thứ khác nhau: `W3-06` đo trên cùng một bộ 15.814 chunk và
thấy đổi tokenizer là đổi số token tới **47%** (p50 EN 313 với PhoBERT vs 212
với BGE-M3). Ở `W3-04` con số cần canh là **cửa sổ ngữ cảnh của Qwen3-8B**
(`max_position_embeddings = 40960`, đọc từ `config.json`), nên đếm bằng
tokenizer của BGE-M3 là canh nhầm thước — dù `EmbeddingProvider` đã thoả sẵn
giao thức `TokenCounter` và cắm vào thì chạy trơn.

Đo được trên corpus thật bằng đúng tokenizer `Qwen/Qwen3-8B`: **EN 5,10** ký
tự/token, **VI 4,37**. Chiều lệch giống BGE-M3 (tiếng Việt tốn token hơn) nhưng
độ lớn khác, và `W3-06` đã cho thấy chiều ấy **đảo dấu** với PhoBERT — nên không
có tỉ lệ nào mang đi dùng chung được.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

__all__ = ["HFTokenCounter"]

logger = logging.getLogger(__name__)


class HFTokenCounter:
    """`TokenCounter` bọc quanh một `AutoTokenizer` của Hugging Face.

    Thoả giao thức `rag_core.chunking.tokens.TokenCounter` nên dùng lại được
    `calibrate_density` và `fit_to_budget` mà không phải viết lại phép cắt.

    `transformers` nạp **lười** trong `tokenizer`: `pipeline/ingest/app.py` có
    test chạy ở tiến trình con ghim rằng import đường HTTP không kéo theo
    `transformers`, và một `import` ở đầu file này là đủ phá nó.

    ⚠️ `max_sequence_tokens` là **property** trong `TokenCounter`, không phải
    method. Bản đầu của lớp này viết thành method và không có gì đỏ — `mypy`
    chỉ so khi có chỗ nối lớp thật với Protocol, mà lúc ấy chưa có chỗ nào. Đúng
    hình dạng lỗi `ChunkFetcher` của `W3-05`; chỗ ghim nằm ở
    `test_contextual_chunking.py`.
    """

    def __init__(self, model: str, *, max_tokens: int | None = None) -> None:
        self.model = model
        self._max_tokens = max_tokens
        self._tokenizer: Any | None = None

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        return self._tokenizer

    @property
    def max_sequence_tokens(self) -> int | None:
        """Trần cửa sổ, **lấy từ tham số dựng chứ không từ tokenizer**.

        ⚠️ `tokenizer.model_max_length` của `Qwen/Qwen3-8B` báo **131072**, còn
        `config.json` của chính model ghi `max_position_embeddings = 40960` và
        `rope_scaling = None`. Tin con số của tokenizer là dựng prompt dài gấp
        3,2× cửa sổ model thật sự có. Nên trần phải do người gọi khai, đọc từ
        `config.json`, và mặc định là `None` (không biết) thay vì một con số sai.
        """
        return self._max_tokens

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        if not texts:
            return []
        encoded = self.tokenizer(list(texts), add_special_tokens=False)["input_ids"]
        return [len(ids) for ids in encoded]
