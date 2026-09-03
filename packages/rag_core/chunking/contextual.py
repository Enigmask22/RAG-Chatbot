"""Contextual Retrieval — gắn 1–2 câu định vị vào chunk **trước khi** embed.

Một chunk cắt ra khỏi giữa báo cáo thường mất hết chủ ngữ: "Tỷ lệ này tăng lên
7,2% trong giai đoạn đó" không nói tỷ lệ gì, của ai, giai đoạn nào. Câu hỏi thật
lại dùng đúng những từ bị mất. Cách chữa của Anthropic: cho LLM đọc tài liệu rồi
viết một hai câu định vị, dán lên đầu chunk, embed cả cụm.

## Vì sao KHÔNG nhét cả tài liệu vào prompt như công thức gốc

Công thức gốc đặt nguyên tài liệu vào prompt và dựa vào prompt caching. Đo trên
corpus này bằng đúng tokenizer `Qwen/Qwen3-8B`:

| | token |
|---|---:|
| tài liệu p50 | **32.176** |
| tài liệu p90 | **138.423** |
| tài liệu lớn nhất | **188.987** |
| cửa sổ Qwen3-8B (`max_position_embeddings`) | **40.960** |

**28/60 tài liệu (47%) không lọt nổi vào cửa sổ**, và tài liệu lớn nhất vượt
4,6×. Kể cả nếu lọt, 15.814 chunk × prompt 32K là ~500 tỷ token prefill — không
phải vấn đề tiền mà là vấn đề tuần lễ.

Lựa chọn hiển nhiên thứ hai — cửa sổ theo **section**, dùng `section_path` của
`W3-03` — bị `TD-24` chặn: `make structure-corpus` đo **0/60** tài liệu có
heading máy đọc được, vì corpus là bản `.txt` World Bank trích sẵn.

Nên cửa sổ ở đây ghép **hai** nguồn, và ghép theo đúng thứ tự đó:

1. `<document_head>` — phần đầu tài liệu (tiêu đề, tóm tắt, khung cảnh). Trả lời
   *"tài liệu nào, về ai, năm nào"*. **Giống hệt nhau cho mọi chunk cùng tài
   liệu.**
2. `<neighbourhood>` — đoạn văn bao quanh chunk, lấy qua `start_char`/`end_char`.
   Trả lời *"mục này đang bàn gì"*. Đổi theo từng chunk.

## Thứ tự trong prompt là một quyết định về hiệu năng, không phải thẩm mỹ

vLLM tự động cache theo **tiền tố token**. Phần bất biến phải nằm trước, nếu
không thì cache không bao giờ trúng. Nên toàn bộ chỉ dẫn nằm ở `system` (bất
biến trên **cả job**), rồi `document_head` (bất biến trong **một tài liệu**), rồi
mới tới phần đổi theo chunk. Sắp xếp cách khác thì phần prefill dùng chung
~2.150 token/lời gọi bị tính lại cho từng chunk trong số 15.814 lời gọi.

## Ngữ cảnh viết bằng ngôn ngữ của chunk

Corpus song ngữ, và vector nằm trong không gian của BGE-M3. Dán một câu tiếng
Anh lên chunk tiếng Việt là trộn hai ngôn ngữ trong một vector mà **không có số
đo nào** nói rằng như vậy tốt hơn. Giả thuyết ngược lại (ngữ cảnh tiếng Anh giúp
nhóm `cross_lingual`) là một ô của `W3-09`, không phải mặc định.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from ..llm.base import ChatMessage
from ..schemas import Chunk, Document
from .tokens import TokenCounter, calibrate_density

__all__ = [
    "CONTEXT_SYSTEM_PROMPT",
    "ContextRequest",
    "ContextualConfig",
    "EnrichStats",
    "apply_contexts",
    "build_requests",
    "original_content",
]

logger = logging.getLogger(__name__)

CONTEXT_KEY = "context"
"""Khoá trong `Chunk.extra` giữ câu ngữ cảnh đã dán."""

CONTEXT_SEPARATOR = "\n\n"
"""Ngăn cách ngữ cảnh với nội dung gốc. Cố định vì `original_content` bóc theo nó."""


CONTEXT_SYSTEM_PROMPT = """\
You situate a passage inside the document it was taken from, so that a search \
engine can find the passage from a question that does not reuse its wording.

You are given:
  <document_head>  the opening of the document — title, summary, framing
  <neighbourhood>  the surrounding passage (may be absent)
  <chunk>          the passage itself

Write ONE or TWO sentences saying what the passage is about and where it sits in \
the document. Name the document, the organisation, the country, the time period \
and the section topic whenever the material states them — those are the words a \
question will use and the passage itself usually leaves out.

Rules:
- Write in the SAME language as <chunk>.
- Do not repeat sentences from <chunk>.
- Do not state anything that is not in the material given.
- Output only the sentences. No preamble, no quotes, no labels, no markdown.\
"""


class ContextualConfig(BaseModel):
    """Tham số của bước sinh ngữ cảnh. Nằm trong khoá cache, nên đổi là sinh lại."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    """Cờ tắt mà DoD yêu cầu. Tắt thì `build_requests` trả rỗng và index như cũ."""

    model: str = ""
    """Slug model sinh ngữ cảnh. Nằm trong khoá vì đổi model là đổi ngữ cảnh."""

    head_tokens: int = Field(default=2000, ge=0)
    window_tokens: int = Field(default=1500, ge=0)
    max_context_tokens: int = Field(default=120, ge=16)
    prompt_version: str = Field(default="ctx-v1", min_length=1)


@dataclass(frozen=True)
class ContextRequest:
    """Một lời gọi LLM cần thực hiện, kèm khoá định danh nó.

    `key` băm **toàn bộ** thứ quyết định câu trả lời (chỉ dẫn, tham số, văn bản
    đưa vào), nên nó vừa là khoá checkpoint vừa là khoá cache: chạy lại job với
    cùng cấu hình thì mọi `key` đã có trong artifact được bỏ qua, còn đổi
    `chunk_size` hay `prompt_version` thì `key` lệch và ngữ cảnh được sinh lại —
    không cần ai nhớ phải xoá cache.
    """

    key: str
    chunk_id: str
    doc_id: str
    messages: tuple[ChatMessage, ...]
    est_prompt_tokens: int

    @property
    def user_text(self) -> str:
        return self.messages[-1].content


@dataclass
class EnrichStats:
    """Kết quả của một lượt dán ngữ cảnh, để log và để test bám vào."""

    n_chunks: int = 0
    n_enriched: int = 0
    n_missing: int = 0
    n_empty: int = 0
    missing_chunk_ids: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return self.n_enriched / self.n_chunks if self.n_chunks else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "n_chunks": self.n_chunks,
            "n_enriched": self.n_enriched,
            "n_missing": self.n_missing,
            "n_empty": self.n_empty,
            "coverage": round(self.coverage, 4),
        }


def _snap_left(text: str, position: int) -> int:
    """Lùi về khoảng trắng gần nhất để không cắt giữa từ. Bó trong 200 ký tự."""
    limit = max(0, position - 200)
    while position > limit and not text[position - 1].isspace():
        position -= 1
    return position


def _snap_right(text: str, position: int) -> int:
    limit = min(len(text), position + 200)
    while position < limit and not text[position].isspace():
        position += 1
    return position


def _window_for(text: str, *, start: int, end: int, head_chars: int, window_chars: int) -> str:
    """Đoạn bao quanh chunk, đã trừ đi phần `document_head` đã đưa vào prompt.

    Trừ đi chứ không để chồng: chunk nằm ở đầu tài liệu thì `head` đã chứa cả nó
    lẫn vùng lân cận, và đưa lại lần nữa là trả tiền prefill hai lần cho cùng một
    văn bản — trên 15.814 lời gọi thì đó là khoản có thật.
    """
    half = window_chars // 2
    w_start = max(head_chars, start - half)
    w_end = min(len(text), end + half)
    if w_end <= w_start:
        return ""
    return text[_snap_left(text, w_start) : _snap_right(text, w_end)].strip()


def build_requests(
    document: Document,
    chunks: Sequence[Chunk],
    *,
    config: ContextualConfig,
    counter: TokenCounter,
) -> list[ContextRequest]:
    """Dựng lời gọi LLM cho từng chunk của **một** tài liệu.

    Hiệu chuẩn mật độ ký tự/token **một lần cho mỗi tài liệu** bằng
    `calibrate_density` rồi cắt theo ký tự, thay vì đếm token cho từng lát cắt
    ứng viên. Tỉ lệ đo được rất khác nhau giữa hai ngôn ngữ (EN 5,10 · VI 4,37
    với tokenizer Qwen3) nên một hằng số dùng chung sẽ làm prompt tiếng Việt dài
    hơn khai báo 17%.

    Raises:
        ValueError: chunk không có `start_char`/`end_char` — không dựng được cửa
            sổ lân cận, và im lặng bỏ qua sẽ cho ngữ cảnh chỉ dựa vào phần đầu
            tài liệu mà không ai biết.
    """
    if not config.enabled or not chunks:
        return []

    text = document.content
    density = calibrate_density(text, counter)
    head_chars = int(config.head_tokens * density)
    window_chars = int(config.window_tokens * density)
    head = text[: _snap_right(text, head_chars)].strip() if head_chars else ""

    requests: list[ContextRequest] = []
    for chunk in chunks:
        if chunk.start_char is None or chunk.end_char is None:
            raise ValueError(
                f"{chunk.chunk_id}: thiếu span, không dựng được cửa sổ lân cận. "
                "Mọi chunker từ W1-11 đều sinh offset — chunk này tới từ đâu?"
            )
        window = _window_for(
            text,
            start=chunk.start_char,
            end=chunk.end_char,
            head_chars=len(head),
            window_chars=window_chars,
        )
        user = _render_user(head=head, window=window, chunk_text=chunk.content)
        messages = (
            ChatMessage(role="system", content=CONTEXT_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user),
        )
        requests.append(
            ContextRequest(
                key=_key_for(config, user),
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                messages=messages,
                est_prompt_tokens=int((len(CONTEXT_SYSTEM_PROMPT) + len(user)) / max(density, 1.0)),
            )
        )
    return requests


def _render_user(*, head: str, window: str, chunk_text: str) -> str:
    """Phần bất biến theo tài liệu đứng TRƯỚC phần đổi theo chunk — xem docstring module."""
    parts = [f"<document_head>\n{head}\n</document_head>"] if head else []
    if window:
        parts.append(f"<neighbourhood>\n{window}\n</neighbourhood>")
    parts.append(f"<chunk>\n{chunk_text}\n</chunk>")
    return "\n\n".join(parts)


def _key_for(config: ContextualConfig, user: str) -> str:
    payload = "\n".join(
        (
            config.prompt_version,
            config.model,
            str(config.max_context_tokens),
            CONTEXT_SYSTEM_PROMPT,
            user,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def apply_contexts(
    chunks: Sequence[Chunk],
    requests: Sequence[ContextRequest],
    contexts: Mapping[str, str],
) -> tuple[list[Chunk], EnrichStats]:
    """Dán ngữ cảnh vào chunk. Thiếu ngữ cảnh thì **giữ nguyên chunk**, không ném.

    Đây là nửa "fail 1 chunk không làm sập cả job" của DoD, ở phía tiêu thụ: job
    sinh có thể bỏ sót vài chunk (LLM trả rỗng, lỗi tất định, chạm trần chi phí)
    và index vẫn phải build được. Số chunk bị bỏ sót nằm trong `EnrichStats` để
    người đọc báo cáo thấy được, thay vì tự tin là 100%.

    ⚠️ Dán ngữ cảnh **đổi `content`, tức đổi `content_hash`**, nên lượt build đầu
    tiên sau khi bật sẽ embed lại toàn bộ — `W3-07` không mượn lại được gì. Từ
    lượt thứ hai trở đi thì bình thường, miễn là artifact ngữ cảnh ổn định.
    """
    by_chunk = {r.chunk_id: r.key for r in requests}
    stats = EnrichStats(n_chunks=len(chunks))
    out: list[Chunk] = []

    for chunk in chunks:
        key = by_chunk.get(chunk.chunk_id)
        context = (contexts.get(key) or "").strip() if key else ""
        if not context:
            if key is None or key not in contexts:
                stats.n_missing += 1
                if len(stats.missing_chunk_ids) < 20:
                    stats.missing_chunk_ids.append(chunk.chunk_id)
            else:
                stats.n_empty += 1
            out.append(chunk)
            continue

        out.append(
            chunk.model_copy(
                update={
                    "content": f"{context}{CONTEXT_SEPARATOR}{chunk.content}",
                    # Số cũ đếm trên văn bản cũ. Giữ lại là nói dối; đếm lại thì
                    # cần tokenizer của embedding model, mà chỗ này không có.
                    "token_count": None,
                    "extra": {**chunk.extra, CONTEXT_KEY: context},
                }
            )
        )
        stats.n_enriched += 1

    if stats.n_missing:
        logger.warning(
            "Thiếu ngữ cảnh cho %d/%d chunk (ví dụ: %s)",
            stats.n_missing,
            stats.n_chunks,
            ", ".join(stats.missing_chunk_ids[:5]),
        )
    return out, stats


def original_content(chunk: Chunk) -> str:
    """Nội dung gốc của chunk, đã bóc câu ngữ cảnh ra.

    Cần cho trích dẫn: câu trả lời phải dẫn lại **văn bản trong tài liệu**, không
    phải câu do một LLM khác viết ra ở bước index. Bóc theo tiền tố thay vì lưu
    thêm một bản gốc trong payload — 15.814 chunk × ~1000 ký tự là 15 MB trùng
    lặp cho một phép cắt chuỗi xác định.
    """
    context = chunk.extra.get(CONTEXT_KEY)
    if not isinstance(context, str) or not context:
        return chunk.content
    prefix = f"{context}{CONTEXT_SEPARATOR}"
    return chunk.content[len(prefix) :] if chunk.content.startswith(prefix) else chunk.content
