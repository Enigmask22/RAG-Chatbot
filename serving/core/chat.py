"""Một lượt hỏi–đáp: truy hồi → sinh theo dòng → ghi lại. `W4-06`.

## ⭐⭐ Đường phân giới: chỗ nào còn trả được HTTP status, chỗ nào thì không

Đây là quyết định kiến trúc của cả hạng mục, và nó là một **đường thật trong
mã** chứ không phải một lời dặn.

Sau khi byte đầu tiên của `200 OK` đã ra khỏi socket, không còn cách nào biến
một lỗi thành `503`. Nếu LLM chết ở token thứ 50, thứ client nhận được là một
dòng SSE dừng lại — và điều đó **trông y hệt** một câu trả lời ngắn đã kết thúc
bình thường. Không có mã lỗi, không có exception, và người dùng đọc nửa câu như
thể đó là toàn bộ câu trả lời.

Nên module này tách làm hai nửa:

| | `prepare()` | `stream_turn()` |
|---|---|---|
| chạy khi | chưa gửi byte nào | đang gửi |
| lỗi biểu hiện thành | **HTTP status thật** (404/403/503) | khung SSE `event: error` |
| gồm | kiểm tenant, mở hội thoại, **truy hồi** | gọi LLM, ghi message trợ lý |

⭐ **Truy hồi nằm ở nửa trên** là lựa chọn có chủ đích, và nó tốn ~600 ms TTFB.
Đổi lại: Qdrant chết trả `503` với `Retry-After` đọc được bằng máy, thay vì một
khung `event: error` mà mọi client phải tự viết mã xử lý. Nửa dưới càng mỏng thì
càng ít thứ chỉ hỏng được theo kiểu không nói ra được.

Hệ quả cho `W4-09`: xác minh citation cần **toàn bộ** câu trả lời, nên nó nằm ở
nửa dưới, và nó phải chấp nhận rằng một citation bịa chỉ báo được bằng một khung
SSE. Đó là lý do khung ở đây tên là `sources` (cái đã đưa cho model) chứ không
phải `citations` (cái đã kiểm) — hai thứ khác nhau, và gộp tên chúng lại là cách
chắc chắn để `W4-09` trở nên vô hình với client.

## ⭐ Ngắt kết nối giữa chừng: token đã trả tiền rồi

Người dùng đóng tab ở giây thứ 3 của một câu trả lời 10 giây. Tiền đã tiêu, và
phần đã sinh vẫn là dữ liệu thật. Nhưng lúc đó Starlette **huỷ** task đang chạy
generator này, nên mọi `await` trong `finally` bị huỷ ngay lập tức — tức đoạn mã
"lưu lại trước khi thoát" viết theo cách hiển nhiên nhất sẽ **không bao giờ
chạy**, và nó không bao giờ chạy đúng ở chỗ khó nhìn thấy nhất.

Cách chặn là `asyncio.create_task` — hàm **đồng bộ**, không treo, nên nó chạy
xong kể cả trong lúc bị huỷ, và task nó tạo ra sống độc lập với request. Ba chi
tiết đi kèm:

* Giữ **tham chiếu mạnh** tới task (`_PENDING`), nếu không GC thu nó giữa chừng
  và việc ghi biến mất — im lặng, ngẫu nhiên, chỉ dưới tải.
* Task tự mở phiên DB **của riêng nó**. Phiên của request đã bị đóng lúc đó.
* ⚠️ Tiến trình tắt trong lúc còn task chờ ⇒ mất bản ghi. Giới hạn thật, và cách
  chữa đúng là outbox — `W5`, không phải một `sleep` ở chỗ shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_core.generation import (
    CitationHoldback,
    default_registry,
    split_citation_block,
    verify_citations,
)
from rag_core.llm import BudgetExceeded, ChatMessage, LLMError, StreamingLLM
from rag_core.retrieval.filters import MetadataFilter
from rag_core.schemas import RetrievedChunk
from serving.core.auth import Principal, tenant_filter
from serving.core.registry import ActiveBundle, BundleRegistry, NoBundleLoadedError
from serving.core.semantic_cache import CachedAnswer, SemanticCache, embedder_of
from serving.core.understanding import QueryPlan, QueryUnderstanding, detect_language
from serving.db.engine import atenant_session
from serving.db.models import Conversation, Message

__all__ = [
    "CHAT_NO_RETRIEVAL",
    "CHAT_SYSTEM",
    "MAX_HISTORY_MESSAGES",
    "NO_RETRIEVAL_SYSTEM_PROMPT",
    "SYSTEM_PROMPT",
    "ChatEvent",
    "ChatService",
    "ChatTurn",
    "ConversationNotFound",
    "GenerationUnavailable",
]

logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 10
"""Số message cũ đưa vào prompt.

Không phải một con số tuỳ tiện: lịch sử **không giới hạn** là một quả bom chi
phí có ngòi nổ chậm — hội thoại thứ 200 của một khách hàng gửi lại toàn bộ 199
lượt trước ở *mỗi* lượt, nên giá một câu hỏi tăng tuyến tính theo số câu đã hỏi,
cho tới khi nó vượt cửa sổ ngữ cảnh và request bắt đầu trả 400.

⚠️ Cắt theo **số message** chứ không theo token là một xấp xỉ thô, và nó sai
theo hướng nguy hiểm khi 10 message ấy đều dài. `HFTokenCounter` (`W1-10`) chờ
sẵn để thay bằng ngân sách token thật — `TD-51` (bản đầu của docstring này hứa
"W4-07/W4-11" nhưng việc ấy không thuộc DoD hạng mục nào).
"""

_PROMPTS = default_registry()
"""`W4-11`: prompt nạp từ registry YAML (`rag_core/generation/prompts/`), có
version + hash. Đổi nội dung mà không qua `scripts/prompt_stamp.py` thì import
này NÉM và server không lên — cố ý fail-fast, khác với bundle nạp lỗi: prompt
là package data đóng trong image, một file hỏng là một bản build hỏng, không
phải một trạng thái runtime chữa được bằng `/admin/bundle/reload`."""

CHAT_SYSTEM = _PROMPTS.get("chat-system")
"""Prompt nhánh RETRIEVE. Nội dung giữ nguyên byte so với hằng số cũ — các phép
đo `W4-07` (chỉ thị ngôn ngữ) và `W4-09` (block CITATIONS) gắn với đúng nội
dung này, đổi byte nào là các con số ấy thôi so được.

Câu cuối của template là hàng rào injection hạng nhẹ, ghi ra để nó không bị
nhầm là đã xong: một dòng chỉ dẫn không chặn được tài liệu cố tình chiếm quyền.
`W4-12` mới là chỗ đó, với bộ payload để đo.

⭐ Luật 4 ("trả lời bằng đúng ngôn ngữ của câu hỏi") không hoạt động một mình:
`W4-07` đo được 8/8 câu tiếng Anh nhận trả lời tiếng Việt khi thiếu dòng chỉ
thị cuối lượt người dùng — xem `QueryPlan.directive`."""

CHAT_NO_RETRIEVAL = _PROMPTS.get("chat-no-retrieval")
"""⭐ Prompt RIÊNG cho nhánh `NO_RETRIEVAL`, không phải `SYSTEM_PROMPT` với ngữ
cảnh rỗng: đưa `"hello"` vào đó cùng một khối ngữ cảnh trống thì model làm ĐÚNG
điều được bảo — nó trả lời rằng không đủ thông tin để chào lại. Luật đúng, ngữ
cảnh đúng, kết quả vô lý — và không có gì trong log nói ra."""

SYSTEM_PROMPT = CHAT_SYSTEM.text
"""Tên cũ giữ dạng `str` cho mọi chỗ đã dùng; nguồn sự thật là `CHAT_SYSTEM`."""

NO_RETRIEVAL_SYSTEM_PROMPT = CHAT_NO_RETRIEVAL.text


def cache_namespace(bundle_version: str) -> str:
    """`W4-11`: version prompt phải nằm trong namespace cache như bundle_version.

    Registry vừa biến prompt thành một biến số có version thì semantic cache
    phải invalidate theo nó: một câu trả lời sinh dưới `chat-system@v1` KHÔNG
    phải câu trả lời của `chat-system@v2`, và phát lại nó là phát lại kết quả
    của một hệ thống đã không còn tồn tại — đúng kiểu hỏng mà namespace-theo-
    bundle của `W4-10` sinh ra để chặn, chỉ là ở một trục khác.
    """
    return f"{bundle_version}+{CHAT_SYSTEM.spec}"


def cache_eligible(plan: QueryPlan, history: Sequence[ChatMessage]) -> bool:
    """Lượt nào được phép chạm cache. `W4-10`.

    Chỉ lượt ĐẦU hội thoại, tự đủ nghĩa, có truy hồi: câu hỏi giữa hội thoại
    mang nghĩa từ lịch sử, và hai người dùng có cùng một câu chữ giữa hai hội
    thoại khác nhau thì KHÔNG có cùng một câu hỏi. Bản viết lại cũng loại —
    nó phụ thuộc lịch sử theo định nghĩa.
    """
    return plan.retrieves and not history and not plan.rewritten


_PENDING: set[asyncio.Task[None]] = set()
"""Tham chiếu mạnh tới các task ghi đang chạy — xem §"Ngắt kết nối" ở docstring.

`asyncio` chỉ giữ tham chiếu **yếu** tới task, nên một task không ai giữ có thể
bị GC dọn giữa chừng. Tài liệu chuẩn nói đúng điều này, và triệu chứng của việc
bỏ qua nó là những lần ghi biến mất ngẫu nhiên dưới tải.
"""


class ConversationNotFound(LookupError):
    """Không có hội thoại ấy — **hoặc** nó thuộc tenant khác.

    ⭐ Cố ý không phân biệt hai ca. RLS làm cho câu `SELECT` trả rỗng ở cả hai,
    và giữ nguyên sự mập mờ ấy trong phản hồi là điều đúng: một `403` cho hội
    thoại của người khác và `404` cho hội thoại không tồn tại biến endpoint này
    thành máy dò xem một `conversation_id` có tồn tại hay không.
    """


class GenerationUnavailable(RuntimeError):
    """Chưa cấu hình được nguồn sinh text, hoặc bundle chưa nạp."""


@dataclass(frozen=True)
class ChatEvent:
    """Một khung SSE. `event` là tên khung, `data` được JSON hoá nguyên vẹn."""

    event: str
    data: dict[str, Any]


@dataclass
class ChatTurn:
    """Mọi thứ đã giải quyết xong **trước** khi byte đầu tiên rời đi."""

    principal: Principal
    conversation_id: str
    user_message_id: str
    plan: QueryPlan
    history: list[ChatMessage]
    contexts: list[RetrievedChunk]
    bundle_version: str
    cached: CachedAnswer | None = None
    """`W4-10`: lượt này được trả từ cache — `stream_turn` phát lại thay vì gọi model."""
    cache_vector: Any | None = None
    """Vector câu hỏi đã embed cho lần tra cache TRƯỢT — giữ lại để ghi cache
    sau khi stream thành công, khỏi embed lần thứ ba."""
    max_tokens: int = 1024
    started: float = field(default_factory=time.perf_counter)

    @property
    def question(self) -> str:
        """Chuỗi **đã đưa vào truy hồi** — viết lại rồi nếu `W4-07` có viết lại.

        ⚠️ Không phải chuỗi người dùng gõ; cái đó là `plan.original`, và nó mới
        là cái được ghi vào cột `content` của message. Gộp hai thứ này lại là
        cách chắc chắn để lịch sử hội thoại hiện ra một câu hỏi không ai hỏi.
        """
        return self.plan.question

    def prompt_spec(self) -> str | None:
        """Prompt nào đã đứng sau lượt này — `chat-system@v1`, hoặc `None` cho
        nhánh CLARIFY (không gọi model). Đi vào khung `meta`: mỗi lượt tự khai
        biến số prompt của mình, để một con số eval sau này truy được nó sinh
        dưới prompt nào mà không phải đoán từ ngày giờ."""
        if self.plan.route == "clarify":
            return None
        return CHAT_SYSTEM.spec if self.plan.retrieves else CHAT_NO_RETRIEVAL.spec

    def sources(self) -> list[dict[str, Any]]:
        """Cái đã **đưa cho model**, đánh số khớp với `[n]` trong prompt."""
        out: list[dict[str, Any]] = []
        for n, hit in enumerate(self.contexts, start=1):
            meta = hit.chunk.metadata
            out.append(
                {
                    "n": n,
                    "chunk_id": hit.chunk.chunk_id,
                    "doc_id": hit.chunk.doc_id,
                    "title": meta.title if meta else None,
                    "source_url": meta.source_url if meta else None,
                    "section_path": hit.chunk.section_path,
                    "score": round(hit.score, 6),
                }
            )
        return out

    def prompt(self) -> list[ChatMessage]:
        directive = self.plan.directive()
        if not self.plan.retrieves:
            # Nhánh `NO_RETRIEVAL`: không ngữ cảnh, không luật trích nguồn, và
            # dùng chuỗi **gốc** — không có gì để viết lại trong một lời chào.
            return [
                ChatMessage(role="system", content=NO_RETRIEVAL_SYSTEM_PROMPT),
                *self.history,
                ChatMessage(role="user", content=f"{self.plan.original}{directive}"),
            ]
        blocks = [f"[{n}] {hit.chunk.content}" for n, hit in enumerate(self.contexts, start=1)]
        context = "\n\n".join(blocks) if blocks else "(không tìm thấy tài liệu liên quan)"
        # ⭐⭐ **Cả hai** chuỗi, và thứ tự này là kết quả của một lần chạy thật.
        #
        # Bản đầu chỉ đưa câu **gốc**, với lý lẽ: truy hồi cần một chuỗi tự đủ
        # nghĩa để so vector, còn model đã có lịch sử ở ngay trên và nên thấy
        # đúng thứ người dùng vừa gõ. Lý lẽ nghe đúng và **sai trong thực tế**:
        # với `"cái đó thì sao?"`, `deepseek-v4-flash` truy hồi ra đúng 5 chunk
        # về di cư lao động rồi trả lời *"tôi không đủ thông tin để trả lời câu
        # hỏi 'cái đó thì sao?' vì câu hỏi không nêu rõ 'cái đó' là gì"*. Lịch
        # sử có trong prompt; model vẫn áp luật 3 lên chuỗi mơ hồ trước mắt nó.
        #
        # Đưa **mỗi** bản viết lại thì câu trả lời lại nói về một câu hỏi người
        # dùng không gõ. Đưa cả hai giữ được cả hai: người dùng thấy chữ của
        # mình, model có bản đã giải nghĩa, và một bản viết lại lệch chủ đề nằm
        # ngay cạnh bản gốc để model tự thấy.
        question = f"CÂU HỎI: {self.plan.original}"
        if self.plan.rewritten:
            question += f'\n(Hiểu đầy đủ theo hội thoại: "{self.plan.question}")'
        return [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            *self.history,
            ChatMessage(role="user", content=f"NGỮ CẢNH:\n{context}\n\n{question}{directive}"),
        ]


@dataclass
class ChatService:
    """Người điều phối một lượt. Không biết gì về HTTP hay SSE — đó là `api/chat.py`.

    `llm` khai kiểu `StreamingLLM` (Protocol) chứ không phải một lớp cụ thể, nên
    `W4-08` cắm router vào đây mà không sửa một dòng nào trong file này.
    """

    registry: BundleRegistry
    sessions: async_sessionmaker[AsyncSession] | None
    llm: StreamingLLM | None
    top_k: int = 5
    max_tokens: int = 1024

    cache: SemanticCache | None = None
    """`W4-10`. `None` = tắt. Mọi lỗi cache đều suy giảm thành miss — cache
    không bao giờ được phép là lý do `/chat` trả lỗi."""

    understanding: QueryUnderstanding = field(default_factory=QueryUnderstanding)
    """`W4-07`. Mặc định là bản **không có LLM**: luật vẫn chạy đủ (định tuyến +
    ngôn ngữ), chỉ viết lại đa lượt là không có. Đó là mức suy giảm đúng — ba
    việc kia miễn phí và tất định, không có lý do gì để chúng phụ thuộc vào việc
    cấu hình được một provider."""

    extra_body: Mapping[str, Any] | None = None
    """⭐⭐ Tham số ngoài chuẩn của provider — trong thực tế là `MIN_REASONING`.

    Lần chạy thật đầu tiên của endpoint này, với `max_tokens = 1024` và một
    prompt 1.613 token, trả về **0 ký tự**: toàn bộ 1024 token completion đi vào
    chuỗi suy luận của `deepseek-v4-flash`, thứ không xuất hiện trong `content`.
    `finish_reason` = `"empty"`, hoá đơn $0,0015, câu trả lời không tồn tại.

    Đó là **đúng** phát hiện của `W3-04` (83% token vào suy luận, 6/30 request
    trả rỗng), xuất hiện lại ở đường request. Khác biệt là ở đó nó tốn tiền, còn
    ở đây nó là một endpoint hỏng — và nó hỏng theo kiểu tệ nhất: `200 OK`, dòng
    SSE hợp lệ, không một khung `error` nào.

    Đặt ở `ChatService` chứ không nhét cứng vào provider: bảng đúng phụ thuộc
    **nhà cung cấp**, và `app.py` là chỗ duy nhất biết nhà nào đang được dùng.
    `W4-08` sẽ chuyển nó vào router, nơi mỗi nhánh fallback mang bảng của mình.
    """

    # ---------------------------------------------------------- nửa trên

    async def prepare(
        self,
        principal: Principal,
        *,
        question: str,
        conversation_id: str | None = None,
        top_k: int | None = None,
        filters: MetadataFilter | None = None,
    ) -> ChatTurn:
        """Mọi thứ còn hỏng thành một HTTP status tử tế. Xem bảng ở docstring module."""
        if self.llm is None:
            raise GenerationUnavailable(
                "chưa cấu hình LLM cho serving — đặt `DEEPSEEK_API_KEY` rồi khởi động lại"
            )
        if self.sessions is None:
            raise GenerationUnavailable("chưa cấu hình Postgres cho serving")
        # ⭐ `W4-08`: hỏi trần chi phí **trước** khi tốn một lượt truy hồi, và
        # trước khi byte đầu tiên rời đi. Sau `200 OK` thì "hết ngân sách" chỉ
        # còn là một dòng SSE dừng lại — cùng đường phân giới ở docstring module.
        # `BudgetExceeded` bay thẳng lên `api/chat.py` và thành `429`.
        check = getattr(self.llm, "assert_within_budget", None)
        if callable(check):
            check()

        try:
            snapshot: ActiveBundle = self.registry.active
        except NoBundleLoadedError as exc:
            # Ảnh chụp **một lần** cho cả lượt — luật 2 của `W4-02`. Đọc
            # `registry.active` lần thứ hai ở giữa lượt có thể trả về runtime
            # khác nếu có reload chen vào, và khi đó câu trả lời trích chunk của
            # index này bằng điểm số của index kia.
            raise GenerationUnavailable(str(exc)) from exc

        history = await self._history(principal, conversation_id)

        # ⭐ Lần gọi **đầu tiên** của `tenant_filter()` từ `W4-04`. Trước dòng
        # này nó chỉ có test; từ đây nó là thứ đứng giữa truy vấn của một khách
        # hàng và corpus của mọi khách hàng còn lại.
        #
        # ⚠️ Gọi **trước** khi rẽ nhánh theo `route`, và luôn gọi kể cả khi lượt
        # này không truy hồi: hàm này không chỉ *thu hẹp* filter, nó còn **từ
        # chối** filter trỏ sang tenant khác. Bỏ nó ở nhánh `no_retrieval` thì
        # cùng một request nhận `403` hay `200` tuỳ vào việc người dùng có chào
        # hỏi hay không — một chỗ dò danh sách tenant, và là một hành vi bảo mật
        # phụ thuộc vào bộ phân loại câu hỏi.
        scoped = tenant_filter(principal, filters)

        plan = await self.understanding.plan(question, history)

        # ⭐ `W4-10`: tra cache TRƯỚC khi truy hồi — hit thì tiết kiệm cả lượt
        # embed+rerank (~800 ms) lẫn lượt model. Vector tra trượt được giữ lại
        # trên turn để ghi cache sau khi stream thành công. Mọi lỗi ở đây suy
        # giảm thành miss; đường đầy đủ không phụ thuộc cache sống hay chết.
        cached: CachedAnswer | None = None
        cache_vector: Any | None = None
        if self.cache is not None and cache_eligible(plan, history):
            embedder = embedder_of(snapshot.retriever)
            if embedder is not None:
                cache_vector = await asyncio.to_thread(embedder.embed_query, plan.question)
                assert cache_vector is not None
                cached = await self.cache.lookup(
                    principal.tenant_id,
                    cache_namespace(snapshot.version),
                    plan.question,
                    cache_vector,
                )

        contexts: list[RetrievedChunk] = []
        if plan.retrieves and cached is None:
            contexts = list(
                await asyncio.to_thread(
                    # ⚠️ `retrieve()` là **đồng bộ** và tốn hàng trăm mili giây
                    # (embed trên GPU + cross-encoder). Gọi thẳng trong
                    # `async def` thì suốt khoảng đó vòng lặp sự kiện không chạy
                    # gì khác — kể cả `/health`, và orchestrator đọc đúng điều
                    # đó là "tiến trình chết". Cùng lý lẽ đã làm cho ba handler
                    # của `admin.py` là `def` chứ không `async def`.
                    snapshot.retriever.retrieve,
                    plan.question,
                    top_k or self.top_k,
                    filters=scoped,
                )
            )

        resolved_id, user_message_id = await self._open_turn(
            principal, conversation_id, plan, snapshot.version
        )
        return ChatTurn(
            principal=principal,
            conversation_id=resolved_id,
            user_message_id=user_message_id,
            plan=plan,
            history=history,
            contexts=contexts,
            bundle_version=snapshot.version,
            max_tokens=self.max_tokens,
            cached=cached,
            cache_vector=cache_vector if cached is None else None,
        )

    # ---------------------------------------------------------- nửa dưới

    async def stream_turn(self, turn: ChatTurn) -> AsyncGenerator[ChatEvent, None]:
        """Từ đây trở đi mọi lỗi chỉ còn là một khung SSE.

        ⚠️ Kiểu trả về là `AsyncGenerator`, **không** phải `AsyncIterator`, và đó
        là một phần của hợp đồng chứ không phải một chi tiết: người gọi phải
        đóng được nó (`aclose`/`athrow`), vì hai đường huỷ ở §"Ngắt kết nối"
        chính là hai cách generator này kết thúc trong thực tế.
        """
        assert self.llm is not None  # `prepare()` đã kiểm; giữ mypy yên tâm
        if turn.cached is not None:
            # ⭐ `W4-10`: phát lại nguyên bộ khung từ cache. `meta.cache` nói RÕ
            # đây là câu trả lời của câu hỏi NÀO và giống bao nhiêu — một cache
            # hit sai (hai câu gần nhau nhưng khác đáp án) phải truy được từ
            # client, không phải chỉ từ log server.
            cached = turn.cached
            yield ChatEvent(
                "meta",
                {
                    "conversation_id": turn.conversation_id,
                    "message_id": turn.user_message_id,
                    "bundle_version": turn.bundle_version,
                    "model": cached.model,
                    "prompt": turn.prompt_spec(),
                    **turn.plan.as_meta(),
                    "cache": {
                        "hit": True,
                        "similarity": cached.similarity,
                        "matched_question": cached.question,
                    },
                },
            )
            yield ChatEvent("sources", {"sources": cached.sources})
            yield ChatEvent("delta", {"text": cached.text})
            if cached.citations_frame is not None:
                yield ChatEvent("citations", cached.citations_frame)
            elapsed = round((time.perf_counter() - turn.started) * 1000.0, 2)
            yield ChatEvent(
                "done",
                {
                    "finish_reason": "cache",
                    "model": cached.model,
                    "usage": {},
                    "ttfb_ms": elapsed,
                    "total_ms": elapsed,
                    "language_mismatch": False,
                },
            )
            self._schedule_save(turn, cached.text, f"cache:{cached.model}", "cache")
            return

        yield ChatEvent(
            "meta",
            {
                "conversation_id": turn.conversation_id,
                "message_id": turn.user_message_id,
                "bundle_version": turn.bundle_version,
                "model": self.llm.model,
                "prompt": turn.prompt_spec(),
                **turn.plan.as_meta(),
            },
        )
        yield ChatEvent("sources", {"sources": turn.sources()})

        if turn.plan.route == "clarify":
            # ⭐ Nhánh duy nhất **không** gọi model. Text lấy từ bảng trong mã,
            # nên nó tất định, miễn phí, và không thể sai ngôn ngữ đã phát hiện.
            #
            # Vẫn đi qua đúng bộ khung SSE (`delta` rồi `done`) chứ không phải
            # một dạng phản hồi riêng: client đã viết mã cho bốn khung ấy, và
            # thêm khung thứ năm cho một nhánh nội bộ là bắt mọi người tiêu thụ
            # phải biết về bộ phân loại câu hỏi.
            text = turn.plan.clarify_text()
            yield ChatEvent("delta", {"text": text})
            yield ChatEvent(
                "done",
                {
                    "finish_reason": "clarify",
                    "model": None,
                    "usage": {},
                    "ttfb_ms": round((time.perf_counter() - turn.started) * 1000.0, 2),
                    "total_ms": round((time.perf_counter() - turn.started) * 1000.0, 2),
                    "language_mismatch": False,
                },
            )
            self._schedule_save(turn, text, "rule:clarify", "clarify")
            return

        parts: list[str] = []
        emitted: list[str] = []
        holdback = CitationHoldback()
        served_model = self.llm.model
        # ⭐ `"unknown"` chứ không phải `"client_disconnect"`, và đó là một lựa
        # chọn có chủ đích sau một phép tiêm lỗi **không** đỏ: nếu khởi tạo bằng
        # `"client_disconnect"` thì khối `except` bên dưới chỉ gán lại đúng giá
        # trị nó đã có — tức nó là chú thích chứ không phải hành vi, và xoá nó đi
        # không test nào thấy.
        #
        # Với `"unknown"`, hai đường huỷ *phải* tự khai tên mình, và một đường
        # thoát thứ ba xuất hiện trong tương lai sẽ được ghi lại là **không
        # biết** thay vì bị gán nhầm cho người dùng.
        finish_reason = "unknown"
        usage: dict[str, Any] = {}
        ttfb_ms: float | None = None
        try:
            async for chunk in self.llm.astream(
                turn.prompt(),
                temperature=0.0,
                max_tokens=turn.max_tokens,
                extra_body=self.extra_body,
            ):
                if chunk.delta:
                    if ttfb_ms is None:
                        ttfb_ms = (time.perf_counter() - turn.started) * 1000.0
                    parts.append(chunk.delta)
                    # ⭐ Block `CITATIONS:` không được rò vào khung `delta`, kể
                    # cả khi marker bị cắt đôi giữa hai delta. `parts` giữ bản
                    # thô để parse; `emitted` là đúng những gì người dùng thấy —
                    # và cũng là bản được ghi vào Postgres, vì hai thứ đó lệch
                    # nhau là một bug không ai truy được từ log.
                    visible = holdback.feed(chunk.delta)
                    if visible:
                        # `append` TRƯỚC `yield`: generator có thể bị huỷ đúng
                        # tại điểm yield — sau khi khung đã rời đi. Append sau
                        # thì bản lưu thiếu đúng mẩu cuối người dùng đã thấy
                        # (hai test cancellation của `W4-06` bắt được điều này
                        # khi thứ tự bị viết ngược trong lúc làm `W4-09`).
                        emitted.append(visible)
                        yield ChatEvent("delta", {"text": visible})
                if chunk.final is not None:
                    served_model = chunk.final.model
                    finish_reason = chunk.final.finish_reason or "stop"
                    usage = {
                        "prompt_tokens": chunk.final.usage.prompt_tokens,
                        "completion_tokens": chunk.final.usage.completion_tokens,
                        "cost_usd": round(chunk.final.usage.cost_usd, 6),
                    }
            tail = holdback.flush()
            if tail:
                emitted.append(tail)
                yield ChatEvent("delta", {"text": tail})
        except (asyncio.CancelledError, GeneratorExit):
            # Client đóng kết nối. Ném tiếp là **bắt buộc**: nuốt một
            # `CancelledError` làm hỏng cơ chế huỷ của cả vòng lặp sự kiện, và
            # nuốt một `GeneratorExit` cho `RuntimeError: async generator
            # ignored GeneratorExit`.
            finish_reason = "client_disconnect"
            raise
        except BudgetExceeded as exc:
            # Trần cạn **giữa** stream: phép kiểm ở `prepare()` chỉ hỏi "đã cạn
            # chưa", còn `astream` giữ chỗ theo ước lượng của chính lời gọi này.
            # Từ đây trở đi không còn HTTP status nào nữa, nên nó phải là một
            # khung `error` có tên riêng chứ không lặng lẽ dừng dòng token.
            finish_reason = "budget"
            logger.warning("hết ngân sách giữa lượt: %s", exc)
            yield ChatEvent(
                "error",
                {"detail": f"BudgetExceeded: {exc}", "partial_chars": len("".join(emitted))},
            )
        except LLMError as exc:
            finish_reason = "error"
            logger.warning("stream hỏng giữa chừng: %s", exc)
            yield ChatEvent(
                "error",
                {
                    "detail": f"{type(exc).__name__}: {exc}",
                    "partial_chars": len("".join(emitted)),
                },
            )
        else:
            finish_reason = finish_reason if parts else "empty"
            # ⭐⭐ Khung `sources` (đã đưa gì cho model) đã phát từ đầu; đây là
            # khung `citations` — model TUYÊN BỐ đã dùng gì, sau khi đối chiếu
            # từng quote với đúng chunk nó chỉ vào. Một quote bịa không thể là
            # một HTTP status nữa (đường phân giới `W4-06`): nó là
            # `verified: false` trong khung này, to và rõ, không im lặng.
            parsed = split_citation_block("".join(parts))
            if turn.plan.retrieves or parsed.block != "absent":
                report = verify_citations(parsed, [hit.chunk for hit in turn.contexts])
                claimed = len(report.citations) + len(report.invalid_ns)
                if report.block != "ok" or report.verified_count < claimed:
                    logger.warning(
                        "citations: block=%s verified=%d/%d (conversation %s)",
                        report.block,
                        report.verified_count,
                        claimed,
                        turn.conversation_id,
                    )
                yield ChatEvent("citations", report.as_frame())
            if (
                self.cache is not None
                and turn.cache_vector is not None
                and finish_reason == "stop"
                and emitted
            ):
                # Ghi cache là việc phụ — chạy nền như đường ghi Postgres, và
                # cùng lý do phải giữ tham chiếu mạnh (xem `_PENDING`).
                store_task = asyncio.get_running_loop().create_task(
                    self.cache.store(
                        turn.principal.tenant_id,
                        cache_namespace(turn.bundle_version),
                        turn.plan.question,
                        turn.cache_vector,
                        text="".join(emitted),
                        sources=turn.sources(),
                        citations_frame=report.as_frame() if turn.plan.retrieves else None,
                        model=served_model,
                    )
                )
                _PENDING.add(store_task)
                store_task.add_done_callback(_PENDING.discard)
            answer_language = detect_language(parsed.text)
            # ⭐⭐ Chỗ chỉ dẫn ngôn ngữ trở thành một **con số**.
            #
            # `W4-06` đo được rằng luật 4 của prompt bị model bỏ qua (hỏi tiếng
            # Anh, đáp tiếng Việt), và `W4-07` không sửa được điều đó — một dòng
            # chỉ dẫn vẫn chỉ là một dòng chỉ dẫn. Cái đổi là từ đây thất bại ấy
            # **đếm được**: cả câu hỏi lẫn câu trả lời đều đi qua cùng một bộ
            # phát hiện, nên chênh lệch xuất hiện trong khung `done` và trong
            # log thay vì chỉ xuất hiện với người dùng.
            #
            # `unknown` ở bất kỳ bên nào ⇒ **không** báo lệch: không biết không
            # phải là biết-khác.
            mismatch = (
                turn.plan.language != "unknown"
                and answer_language != "unknown"
                and answer_language != turn.plan.language
            )
            if mismatch:
                logger.warning(
                    "câu trả lời lệch ngôn ngữ: hỏi %s, đáp %s (conversation %s)",
                    turn.plan.language,
                    answer_language,
                    turn.conversation_id,
                )
            yield ChatEvent(
                "done",
                {
                    "finish_reason": finish_reason,
                    "model": served_model,
                    "usage": usage,
                    "ttfb_ms": round(ttfb_ms, 2) if ttfb_ms is not None else None,
                    "total_ms": round((time.perf_counter() - turn.started) * 1000.0, 2),
                    "language_mismatch": mismatch,
                },
            )
        finally:
            # ⚠️ **Đồng bộ, không `await`.** Xem §"Ngắt kết nối" ở docstring
            # module: ở đây có thể đang bị huỷ, và một `await` trong lúc bị huỷ
            # không chạy tới nơi.
            #
            # Ghi `emitted` chứ không phải `parts`: block CITATIONS là giao thức
            # giữa model và mã, không phải nội dung — lịch sử đọc lại từ DB phải
            # là đúng những gì người dùng đã thấy trên màn hình.
            self._schedule_save(turn, "".join(emitted), served_model, finish_reason)

    # ---------------------------------------------------------------- DB

    async def _history(
        self, principal: Principal, conversation_id: str | None
    ) -> list[ChatMessage]:
        if conversation_id is None:
            return []
        assert self.sessions is not None
        async with atenant_session(self.sessions, principal.tenant_id) as session:
            exists = await session.scalar(
                select(Conversation.id).where(Conversation.id == conversation_id)
            )
            if exists is None:
                raise ConversationNotFound(f"không có hội thoại {conversation_id!r}")
            rows = (
                await session.scalars(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    # Lấy **mới nhất** rồi đảo lại, chứ không `LIMIT` từ đầu:
                    # `ORDER BY created_at ASC LIMIT 10` cho 10 message **đầu
                    # tiên** của hội thoại, tức prompt càng ngày càng lạc đề khi
                    # cuộc trò chuyện dài ra — và nó vẫn trông như đang hoạt động.
                    .order_by(Message.created_at.desc(), Message.id.desc())
                    .limit(MAX_HISTORY_MESSAGES)
                )
            ).all()
        return [
            ChatMessage(role=row.role, content=row.content)  # type: ignore[arg-type]
            for row in reversed(rows)
            if row.role in ("user", "assistant") and row.content
        ]

    async def _open_turn(
        self,
        principal: Principal,
        conversation_id: str | None,
        plan: QueryPlan,
        bundle_version: str,
    ) -> tuple[str, str]:
        """Tạo hội thoại nếu cần, rồi ghi câu hỏi. Ghi **trước** khi sinh.

        Ghi câu hỏi ở cuối lượt thì một lần crash giữa stream làm chính câu hỏi
        biến mất — người dùng tải lại trang và thấy câu mình vừa gõ không còn ở
        đâu cả. Mất câu trả lời thì họ hỏi lại được; mất câu hỏi thì lịch sử nói
        dối về chuyện đã xảy ra.
        """
        assert self.sessions is not None
        async with atenant_session(self.sessions, principal.tenant_id) as session:
            if conversation_id is None:
                conversation = Conversation(
                    tenant_id=principal.tenant_id,
                    title=plan.original[:120],
                    bundle_version=bundle_version,
                )
                session.add(conversation)
                await session.flush()
                conversation_id = conversation.id
            message = Message(
                tenant_id=principal.tenant_id,
                conversation_id=conversation_id,
                role="user",
                # Chuỗi người dùng **thật sự gõ**. Ghi bản viết lại vào đây thì
                # lượt sau đọc lịch sử ra một câu hỏi không ai hỏi, và bước viết
                # lại của lượt sau sẽ dựa trên đó — sai số cộng dồn qua từng lượt.
                content=plan.original,
                route=plan.route,
                rewritten_query=plan.question if plan.rewritten else None,
            )
            session.add(message)
            await session.flush()
            message_id = message.id
            await session.commit()
        return conversation_id, message_id

    def _schedule_save(self, turn: ChatTurn, text: str, model: str, finish_reason: str) -> None:
        task = asyncio.create_task(self._save(turn, text, model, finish_reason))
        _PENDING.add(task)
        task.add_done_callback(_PENDING.discard)

    async def _save(self, turn: ChatTurn, text: str, model: str, finish_reason: str) -> None:
        if not text:
            # Không sinh được chữ nào: một hàng rỗng trong lịch sử tệ hơn là
            # không có hàng nào — nó hiện ra như một câu trả lời trống và không
            # phân biệt được với việc model im lặng.
            logger.warning(
                "không ghi message trợ lý cho %s: rỗng (%s)",
                turn.conversation_id,
                finish_reason,
            )
            return
        assert self.sessions is not None
        try:
            async with atenant_session(self.sessions, turn.principal.tenant_id) as session:
                session.add(
                    Message(
                        tenant_id=turn.principal.tenant_id,
                        conversation_id=turn.conversation_id,
                        role="assistant",
                        content=text,
                        citations=turn.sources(),
                        latency_ms=int((time.perf_counter() - turn.started) * 1000.0),
                        model=model,
                        finish_reason=finish_reason,
                    )
                )
                await session.commit()
        except Exception:
            # Task này chạy ngoài request; một exception ở đây không có ai bắt
            # và `asyncio` chỉ in nó lúc GC dọn task — tức nó lạc khỏi
            # `request_id` và khỏi mọi dòng access.
            logger.exception(
                "ghi message trợ lý thất bại (%s, %s)", turn.conversation_id, finish_reason
            )


async def load_history(
    sessions: async_sessionmaker[AsyncSession], principal: Principal, conversation_id: str
) -> list[dict[str, Any]]:
    """Đọc lại một hội thoại — thứ chứng minh câu DoD "sống sót qua restart"."""
    async with atenant_session(sessions, principal.tenant_id) as session:
        exists = await session.scalar(
            select(Conversation.id).where(Conversation.id == conversation_id)
        )
        if exists is None:
            raise ConversationNotFound(f"không có hội thoại {conversation_id!r}")
        rows: Sequence[Message] = (
            await session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at, Message.id)
            )
        ).all()
    return [
        {
            "id": row.id,
            "role": row.role,
            "content": row.content,
            "created_at": row.created_at.isoformat(),
            "citations": row.citations,
            "model": row.model,
            "finish_reason": row.finish_reason,
            "latency_ms": row.latency_ms,
            "route": row.route,
            "rewritten_query": row.rewritten_query,
        }
        for row in rows
    ]
