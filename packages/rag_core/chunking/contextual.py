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
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from pydantic import BaseModel, ConfigDict, Field

from ..llm.base import ChatMessage
from ..schemas import Chunk, Document
from .tokens import TokenCounter, calibrate_density

__all__ = [
    "BATCH_SYSTEM_PROMPT",
    "CONTEXT_SYSTEM_PROMPT",
    "BatchParseError",
    "ContextRequest",
    "ContextualConfig",
    "EnrichStats",
    "apply_contexts",
    "build_requests",
    "original_content",
    "parse_response",
]

logger = logging.getLogger(__name__)

_NUMBERED = re.compile(r"^\s*[\[(]?(\d{1,2})[\]).:]\s*(.*)$")
"""Dấu mở đầu một dòng trả lời của chế độ gộp. Khoan dung với `[1]`, `1.`, `(1)`."""

CONTEXT_KEY = "context"
"""Khoá trong `Chunk.extra` giữ câu ngữ cảnh đã dán."""

CONTEXT_SEPARATOR = "\n\n"
"""Ngăn cách ngữ cảnh với nội dung gốc. Cố định vì `original_content` bóc theo nó."""

LANGUAGE_NAMES: dict[str, str] = {"vi": "Vietnamese", "en": "English"}
"""Tên ngôn ngữ để **gọi thẳng** trong prompt.

⚠️⚠️ Bản đầu của prompt viết *"Write in the SAME language as `<chunk>`"* và để
model tự suy. Dry-run 30 request trên một tài liệu tiếng Anh: **15 ngữ cảnh
tiếng Pháp, 10 tiếng Trung, đúng 5 tiếng Anh — 17%**. Không lỗi nào, không cảnh
báo nào; index sẽ nhận 15.814 chunk mang một câu tiếng Pháp dán lên đầu và mọi
metric chỉ đơn giản là tệ hơn mà không ai truy ra vì sao.

Ngôn ngữ nằm sẵn ở `DocumentMetadata.lang`, tới từ manifest. Bắt model suy ra
thứ mình đã biết là tự thêm một chỗ hỏng. `unknown`/`mixed` mới quay về cách cũ,
và khi đó nó là lối thoát cuối chứ không phải mặc định."""


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
- Do not repeat sentences from <chunk>.
- Do not state anything that is not in the material given.
- Output only the sentences. No preamble, no quotes, no labels, no markdown.\
"""


ECHO_SEPARATOR = " || "
"""Ngăn phần echo với phần ngữ cảnh trong một dòng trả lời của chế độ gộp.

Chuỗi này phải là thứ **không xuất hiện trong văn bản tự nhiên**, vì parser cắt
theo lần xuất hiện ĐẦU TIÊN của nó: ngữ cảnh có chứa `||` thì phần sau vẫn còn
nguyên, còn echo có chứa `||` thì mới hỏng — và echo là 4 từ lấy từ chính tài
liệu, nên rủi ro nằm ở phía đã được kiểm bởi `_check_echo`."""

ECHO_WORDS = 4
"""Số từ đầu passage mà model phải chép lại.

⚠️⚠️ Đây là **chốt chặn chống lệch thứ tự**, không phải trang trí. Gộp N chunk
vào một lời gọi thì model trả về N dòng, và nếu nó gán ngữ cảnh của passage 2 cho
dòng 1 thì **không có gì đỏ**: đủ số dòng, đủ số thứ tự, mỗi dòng một câu hợp lệ.
Chunk nhận nhầm ngữ cảnh đi thẳng vào vector và chỉ hiện ra dưới dạng metric tệ
hơn mà không ai truy được vì sao.

Bắt model chép lại 4 từ đầu của đúng passage nó đang mô tả biến lỗi im lặng ấy
thành lỗi kiểm được: `_check_echo` so chuỗi ấy với văn bản thật. Giá phải trả là
~24 token output mỗi lô, tức **~$0,02 cho cả corpus** — rẻ hơn một lần phải chạy
lại vì nghi ngờ."""

BATCH_SYSTEM_PROMPT = """\
You situate passages inside the document they were taken from, so that a search \
engine can find each passage from a question that does not reuse its wording.

You are given:
  <document_head>  the opening of the document — title, summary, framing
  <before>         the text immediately preceding the passages (may be absent)
  <passages>       numbered passages, in document order
  <after>          the text immediately following the passages (may be absent)

For EACH passage write ONE or TWO sentences saying what it is about and where \
it sits in the document.

⚠️ Each sentence you write is stored separately and later read ON ITS OWN, with \
no access to the other passages, to your other answers, or to this prompt. So \
EVERY line must stand alone and must name, whenever the material states them:
  - the document title
  - the organisation that published it
  - the country and the time period
  - the section or topic the passage belongs to
Repeat those in every single line. It will feel redundant across the lines you \
write together; it is not redundant, because no reader ever sees two of them \
at once. A line saying only "this passage continues the previous section" is \
useless on its own and counts as a failure.

Rules:
- Do not repeat sentences from the passage.
- Do not state anything that is not in the material given.
- Treat each passage separately. Do not merge them, do not skip any.\
"""


class BatchParseError(ValueError):
    """Trả lời của chế độ gộp không bóc tách được, hoặc bóc ra sai passage.

    Ném ra thay vì trả về phần bóc được: một lô hỏng nửa chừng nghĩa là **không
    biết** dòng nào ứng với passage nào, và ghi bừa phần "có vẻ đúng" là đúng
    kiểu lỗi mà `ECHO_WORDS` sinh ra để chặn.
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

    chunk_fingerprint: str = ""
    """Vân tay của cấu hình đã sinh ra bộ chunk này, do phía gọi cung cấp.

    `rag_core` không biết `IndexConfig`, nên nó chỉ **mang theo** chuỗi này chứ
    không tính ra nó. Mục đích duy nhất: đi kèm ngữ cảnh vào artifact để lúc dán
    còn kiểm được rằng chunk hiện tại đúng là chunk đã sinh ra nó.

    Không nằm trong khoá cache — đổi cấu hình chunk thì nội dung chunk đổi, nên
    prompt đổi, nên khoá đã đổi sẵn. Ở đây nó là **bằng chứng**, không phải khoá.
    """

    batch_size: int = Field(default=1, ge=1, le=32)
    """Số chunk mỗi lời gọi LLM. `1` là chế độ một-chunk-một-lời-gọi ban đầu.

    ⭐ Đây là đòn bẩy chi phí lớn nhất khi nhà cung cấp **không** có prefix
    caching. Đo được trên GLM-5.3-Flash: `<document_head>` chiếm 49% mỗi prompt
    và bị trả tiền lại cho từng chunk trong ~185 chunk của tài liệu, vì cache
    trúng chỉ 0,1% ngay cả khi chạy tuần tự. Gộp 8 chunk thì head chia cho 8, và
    vùng lân cận của chúng chồng lấn nên gộp lại gần như một dải liền — đưa cả
    corpus từ ~$10,6 xuống ~$2,5.

    Nằm trong khoá cache vì đổi nó là đổi prompt, tức đổi ngữ cảnh sinh ra."""


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
    chunk_ids: tuple[str, ...]
    doc_id: str
    messages: tuple[ChatMessage, ...]
    est_prompt_tokens: int
    chunk_fingerprint: str = ""
    echoes: tuple[str, ...] = ()
    """4 từ đầu của từng chunk, để `parse_response` kiểm chốt chặn — xem `ECHO_WORDS`.

    Nằm trong request chứ không tra lại từ corpus, vì pod **chỉ** nhận
    `requests.jsonl` (quy tắc cứng #2). Bốn từ mỗi chunk là ~30 byte, tức cả
    corpus thêm ~0,5 MB — rẻ hơn nhiều so với việc phải mang corpus lên pod chỉ
    để kiểm được thứ tự.
    """

    @property
    def user_text(self) -> str:
        return self.messages[-1].content

    @property
    def n_chunks(self) -> int:
        """Số chunk lời gọi này phụ trách. `1` là chế độ một-chunk-một-lời-gọi.

        Vòng chạy nhân `--max-tokens` với số này. Không nhân thì một lô 8 chunk
        chạy với trần 256 token bị cắt lời ở chunk thứ hai, và triệu chứng là
        `BatchParseError` hàng loạt chứ không phải một thông báo nói ra điều đó.
        """
        return len(self.chunk_ids)


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
    language = document.metadata.lang.value if document.metadata else ""
    density = calibrate_density(text, counter)
    head_chars = int(config.head_tokens * density)
    window_chars = int(config.window_tokens * density)
    head = text[: _snap_right(text, head_chars)].strip() if head_chars else ""

    for chunk in chunks:
        if chunk.start_char is None or chunk.end_char is None:
            raise ValueError(
                f"{chunk.chunk_id}: thiếu span, không dựng được cửa sổ lân cận. "
                "Mọi chunker từ W1-11 đều sinh offset — chunk này tới từ đâu?"
            )

    groups = [
        list(chunks[i : i + config.batch_size]) for i in range(0, len(chunks), config.batch_size)
    ]
    requests: list[ContextRequest] = []
    for group in groups:
        system, user = _render(
            group, text=text, head=head, window_chars=window_chars, language=language
        )
        requests.append(
            ContextRequest(
                key=_key_for(config, system, user),
                chunk_ids=tuple(c.chunk_id for c in group),
                doc_id=group[0].doc_id,
                messages=(
                    ChatMessage(role="system", content=system),
                    ChatMessage(role="user", content=user),
                ),
                est_prompt_tokens=int((len(system) + len(user)) / max(density, 1.0)),
                chunk_fingerprint=config.chunk_fingerprint,
                echoes=tuple(_first_words(c.content) for c in group),
            )
        )
    return requests


def _render(
    group: Sequence[Chunk], *, text: str, head: str, window_chars: int, language: str
) -> tuple[str, str]:
    """Chọn giữa prompt một-chunk và prompt gộp, trả về `(system, user)`.

    Nhóm một phần tử **vẫn đi đường một-chunk** chứ không phải đường gộp với
    `n=1`: prompt một-chunk là thứ đã sinh ra 860 ngữ cảnh dùng được, và bắt nó
    đi qua định dạng đánh số + echo chỉ để thống nhất hình thức là đổi một đường
    đã kiểm lấy một đường chưa kiểm mà không được gì.
    """
    if len(group) == 1:
        chunk = group[0]
        window = _window_for(
            text,
            start=chunk.start_char or 0,
            end=chunk.end_char or 0,
            head_chars=len(head),
            window_chars=window_chars,
        )
        return CONTEXT_SYSTEM_PROMPT, _render_user(
            head=head, window=window, chunk_text=chunk.content, language=language
        )
    return BATCH_SYSTEM_PROMPT, _render_batch_user(
        group, text=text, head=head, window_chars=window_chars, language=language
    )


def _render_batch_user(
    group: Sequence[Chunk], *, text: str, head: str, window_chars: int, language: str
) -> str:
    """Prompt gộp: `<before>` và `<after>` là **phần ngoài** nhóm, không chồng lấn.

    Các chunk trong nhóm liền kề nhau, nên chúng đã là vùng lân cận của nhau —
    chunk giữa nhóm nhìn thấy hàng xóm thật của nó ngay trong `<passages>`. Chỉ
    hai đầu nhóm là còn thiếu, và đó đúng là phần `<before>`/`<after>` bù vào.
    Đưa cả một `<neighbourhood>` bao trùm như chế độ một-chunk sẽ lặp lại toàn bộ
    văn bản của nhóm lần thứ hai trong cùng một prompt.
    """
    start = min(c.start_char or 0 for c in group)
    end = max(c.end_char or 0 for c in group)
    half = window_chars // 2

    before_start = max(len(head), start - half)
    before = text[_snap_left(text, before_start) : start].strip() if before_start < start else ""
    after = text[end : _snap_right(text, min(len(text), end + half))].strip()

    parts = [f"<document_head>\n{head}\n</document_head>"] if head else []
    if before:
        parts.append(f"<before>\n{before}\n</before>")
    passages = "\n\n".join(f"[{i}]\n{chunk.content}" for i, chunk in enumerate(group, start=1))
    parts.append(f"<passages>\n{passages}\n</passages>")
    if after:
        parts.append(f"<after>\n{after}\n</after>")

    n = len(group)
    name = LANGUAGE_NAMES.get(language)
    in_language = f" Write in {name}." if name else " Write in the language of the passages."
    parts.append(
        f"Now write the situating sentences for each of the {n} passages.{in_language}\n"
        f"Output exactly {n} lines and nothing else. Line i must be:\n"
        f"[i] first {ECHO_WORDS} words of passage i{ECHO_SEPARATOR}the sentences\n"
        f"Copy those first {ECHO_WORDS} words exactly as they appear in passage i.\n"
        "Every line must name the document, the organisation and the period on its "
        "own — the lines are stored separately and never read together."
    )
    return "\n\n".join(parts)


def _render_user(*, head: str, window: str, chunk_text: str, language: str) -> str:
    """Phần bất biến theo tài liệu đứng TRƯỚC phần đổi theo chunk — xem docstring module.

    Câu lệnh cuối đặt **sau** `<chunk>` có chủ ý kép: nó nằm ngoài tiền tố dùng
    chung nên không tốn gì cho prefix caching, và nó là thứ model đọc gần nhất
    trước khi trả lời — đúng chỗ cần cho ràng buộc bị bỏ qua nhiều nhất.
    """
    parts = [f"<document_head>\n{head}\n</document_head>"] if head else []
    if window:
        parts.append(f"<neighbourhood>\n{window}\n</neighbourhood>")
    parts.append(f"<chunk>\n{chunk_text}\n</chunk>")
    name = LANGUAGE_NAMES.get(language)
    parts.append(
        f"Now write the situating sentences for <chunk>, in {name}."
        if name
        else "Now write the situating sentences for <chunk>, in the same language as <chunk>."
    )
    return "\n\n".join(parts)


def _key_for(config: ContextualConfig, system: str, user: str) -> str:
    """Băm **cả** `system`, vì hai chế độ dùng hai chỉ dẫn khác nhau.

    ⚠️ Bản đầu băm hằng số `CONTEXT_SYSTEM_PROMPT` thay vì `system` thực tế. Khi
    chỉ có một chế độ thì hai thứ đó luôn bằng nhau nên lỗi không quan sát được;
    thêm chế độ gộp thì nó thành "đổi chỉ dẫn mà khoá không đổi", tức artifact cũ
    được dùng lại cho một prompt khác hẳn.
    """
    payload = "\n".join(
        (
            config.prompt_version,
            config.model,
            str(config.max_context_tokens),
            system,
            user,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalise(text: str) -> str:
    """Gấp khoảng trắng và bỏ dấu câu ở hai đầu, để so echo không vấp vào định dạng."""
    return " ".join(text.split()).strip("\"'`*_-–—.,:;!?()[]{}").casefold()


def _first_words(text: str, n: int = ECHO_WORDS) -> str:
    return " ".join(text.split()[:n])


def parse_response(request: ContextRequest, text: str) -> dict[str, str]:
    """Bóc trả lời của LLM thành `chunk_id -> context`.

    Chế độ một-chunk trả thẳng. Chế độ gộp thì bóc theo dấu `[i]` và **kiểm
    echo**: dòng `i` phải chép lại đúng 4 từ đầu của passage `i`.

    Raises:
        BatchParseError: thiếu/thừa dòng, số thứ tự không phải `1..n`, dòng rỗng,
            hoặc echo không khớp. Cả lô bị từ chối — xem `BatchParseError`.
    """
    body = text.strip()
    if request.n_chunks == 1:
        return {request.chunk_ids[0]: body} if body else {}

    numbered = _split_numbered(body)
    expected = set(range(1, request.n_chunks + 1))
    if set(numbered) != expected:
        raise BatchParseError(f"chờ {sorted(expected)} dòng, bóc ra {sorted(numbered)}")

    out: dict[str, str] = {}
    for index, chunk_id in enumerate(request.chunk_ids, start=1):
        echo, _, context = numbered[index].partition(ECHO_SEPARATOR)
        context = context.strip() if context else ""
        if not context:
            raise BatchParseError(f"dòng [{index}] không có phần ngữ cảnh sau {ECHO_SEPARATOR!r}")
        _check_alignment(index, echo, request.echoes)
        out[chunk_id] = context
    return out


def _split_numbered(body: str) -> dict[int, str]:
    """Gom các dòng theo dấu `[i]`, nối cả phần xuống dòng vào dòng đang mở.

    Nối thay vì đòi đúng một dòng mỗi passage: model xuống dòng giữa hai câu là
    chuyện thường và hoàn toàn vô hại, còn từ chối cả lô vì một ký tự xuống dòng
    là tự tạo ra tỉ lệ hỏng không cần thiết trên một job 15.814 chunk.
    """
    out: dict[int, list[str]] = {}
    current: int | None = None
    for raw in body.splitlines():
        match = _NUMBERED.match(raw)
        if match:
            current = int(match.group(1))
            out.setdefault(current, []).append(match.group(2).strip())
        elif current is not None and raw.strip():
            out[current].append(raw.strip())
    return {i: " ".join(parts).strip() for i, parts in out.items() if " ".join(parts).strip()}


MIN_ECHO_CHARS = 10
"""Phần mở đầu của passage phải dài tối thiểu bằng này thì mới định danh được nó.

Đo trên dữ liệu thật: `"g g n"` (3 ký tự sau khi ép) tình cờ giống passage 4
(0,86) hơn passage 1 (0,75) mà nhìn bằng mắt thì rõ ràng là passage 1
(`"hB g g N"`). Chuỗi càng ngắn thì `SequenceMatcher` càng dễ cho điểm cao ngẫu
nhiên — theo **cả hai** chiều, nên nó vừa bỏ sót vừa báo động giả.

Đây là ngoại lệ **duy nhất** được đi qua chốt chặn mà không bị xét: khi chính
văn bản gốc không đủ để định danh thì không phép so nào cứu được."""

MIN_CONFIRM = 0.30
"""Điểm giống tối thiểu để coi là đã **xác nhận** dòng `i` nói về passage `i`.

⚠️ Ranh giới quan trọng và nó đi ngược trực giác tiết kiệm: dưới ngưỡng này thì
echo *không giống passage nào*, và tôi từng định cho qua với lý do "không có
bằng chứng lệch". Nhưng không có bằng chứng lệch **không phải** bằng chứng không
lệch — một ca thật trong lượt canary có echo `"hiện có. Việt Nam có thể"` trong
khi passage `[2]` mở đầu bằng `"hỗ trợ theo các"`, tức hai đoạn khác hẳn nhau.

Đường lùi một-chunk **không thể lệch** và giá của nó biết trước. Nên khi không
xác nhận được, trả về đường lùi là đổi một rủi ro im lặng không chặn trên lấy
một khoản chi đo được."""

ALIGNMENT_MARGIN = 0.05
"""Biên tuyệt đối khi so với passage giống nhất — chống nhiễu làm đổi thứ hạng sát nút."""

ALIGNMENT_RELATIVE = 0.25
"""Biên tương đối, cần thiết khi **nhiều passage cùng khớp cao**.

Corpus có tiêu đề chạy lặp ở đầu nhiều đoạn, nên một echo hợp lệ có thể giống
passage khác gần bằng: đo được `0,89` cho passage `[4]` so với `0,81` cho passage
`[8]` đúng của nó. Biên tuyệt đối 0,05 từ chối ca ấy, còn biên tương đối thì
không — vì `0,89 − 0,81` nhỏ so với chính `0,89`."""


def _similarity(a: str, b: str) -> float:
    """Độ giống ở mức ký tự, đã bỏ hết khoảng trắng.

    Bỏ khoảng trắng vì corpus là OCR hai cột: `"tương đương trước đại dịch"` nằm
    trong tài liệu dưới dạng `"tương trướcđại đươngtrước đạidịch"`. So theo từ
    thì hai chuỗi ấy gần như không giao nhau, so theo ký tự thì chúng rất giống —
    và chúng **đúng là** cùng một đoạn.
    """
    return SequenceMatcher(None, _squash(a), _squash(b)).ratio()


def _squash(text: str) -> str:
    return "".join(_normalise(text).split())[:60]


def _check_alignment(index: int, echo: str, echoes: Sequence[str]) -> None:
    """⭐⭐ Chốt chặn chống lệch thứ tự: dòng `i` phải **xác nhận được** là nói về passage `i`.

    So chuỗi model chép lại với phần mở đầu của **mọi** passage trong lô, rồi đòi
    passage `i` vừa đủ giống (`MIN_CONFIRM`) vừa không thua passage nào khác một
    cách rõ rệt (`ALIGNMENT_MARGIN`/`ALIGNMENT_RELATIVE`).

    ⚠️ Bản đầu so **bằng nhau đúng từng chữ** với 4 từ đầu, và nó từ chối
    **109/110 lô** ở lượt canary. Đọc lại 18 ca thì không ca nào lệch thật —
    model chỉ đang dọn văn bản OCR: bỏ số trang (`"55 phụ lục 2"` → `"phụ lục
    2"`), gỡ chữ đan xen hai cột (`"tương trướcđại đươngtrước đạidịch"` → `"tương
    đương trước đại dịch"`). Chốt chặn hỏi *"có chép đúng từng chữ không"* trong
    khi câu cần hỏi là *"có chỉ đúng passage không"*; hai câu ấy chỉ trùng nhau
    khi văn bản sạch.

    So với **tất cả** passage thay vì chỉ passage `i` cũng mạnh hơn hẳn: hoán đổi
    hai dòng làm cực đại rơi sang passage kia nên bị bắt, trong khi nhiễu OCR tác
    động lên mọi phép so như nhau nên không đổi thứ hạng.
    """
    if index > len(echoes):
        return
    own = echoes[index - 1] if echoes else ""
    if len(_squash(own)) < MIN_ECHO_CHARS:
        return

    scores = [_similarity(echo, other) for other in echoes]
    mine = scores[index - 1]
    best = max(scores)
    if mine >= MIN_CONFIRM and mine >= best - max(ALIGNMENT_MARGIN, ALIGNMENT_RELATIVE * best):
        return

    winner = scores.index(best) + 1
    raise BatchParseError(
        f"dòng [{index}] echo {echo!r} không xác nhận được là passage [{index}] "
        f"(giống {mine:.2f}; passage [{winner}] giống {best:.2f})"
    )


def apply_contexts(
    chunks: Sequence[Chunk],
    contexts: Mapping[str, str],
) -> tuple[list[Chunk], EnrichStats]:
    """Dán ngữ cảnh vào chunk. Thiếu ngữ cảnh thì **giữ nguyên chunk**, không ném.

    `contexts` khoá theo **`chunk_id`**, không theo khoá băm của request. Ở chế
    độ gộp, một request phụ trách nhiều chunk nên ánh xạ một-một không còn;
    việc loại bỏ ngữ cảnh cũ sinh bởi cấu hình khác chuyển sang
    `load_contexts(path, keys=...)`, nơi có đủ thông tin để làm.

    Đây là nửa "fail 1 chunk không làm sập cả job" của DoD, ở phía tiêu thụ: job
    sinh có thể bỏ sót vài chunk (LLM trả rỗng, lỗi tất định, chạm trần chi phí)
    và index vẫn phải build được. Số chunk bị bỏ sót nằm trong `EnrichStats` để
    người đọc báo cáo thấy được, thay vì tự tin là 100%.

    ⚠️ Dán ngữ cảnh **đổi `content`, tức đổi `content_hash`**, nên lượt build đầu
    tiên sau khi bật sẽ embed lại toàn bộ — `W3-07` không mượn lại được gì. Từ
    lượt thứ hai trở đi thì bình thường, miễn là artifact ngữ cảnh ổn định.
    """
    stats = EnrichStats(n_chunks=len(chunks))
    out: list[Chunk] = []

    for chunk in chunks:
        context = (contexts.get(chunk.chunk_id) or "").strip()
        if not context:
            if chunk.chunk_id not in contexts:
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
