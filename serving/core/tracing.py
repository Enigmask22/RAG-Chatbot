"""Cây span cho một lượt `/chat` — `W5-06`.

File này **không** biết Langfuse là gì. Nó dựng một cây span trong bộ nhớ rồi
giao cho một `TraceSink`; `serving/core/langfuse.py` là cái sink duy nhất hiện
có, và nó có thể vắng mặt hoàn toàn mà không một dòng nào ở đây phải đổi.

## ⭐⭐ Vì sao không gọi thẳng SDK ở đường request

Ba lý do, và cả ba đều là ràng buộc đã có sẵn trong dự án chứ không phải sở
thích:

1. **Đường request không được chờ một hệ thống quan sát.** Nếu span được đẩy đi
   ngay lúc nó đóng, thì Langfuse chậm 2 giây là `/chat` chậm 2 giây. Cây được
   gom trọn trong bộ nhớ rồi đẩy **một lần** ở cuối, trong một task nền — cùng
   khuôn với `_schedule_save` của `W4-06` và đường ghi cache của `W4-10`.
2. **Đếm span phải test được mà không cần hạ tầng.** DoD của hạng mục này là
   "đủ span"; nếu phép đếm ấy chỉ kiểm được bằng cách gọi một server thật thì
   nó sẽ không bao giờ chạy trong CI, và một span biến mất sẽ không ai thấy.
3. **Phải kiểm soát chính xác cái gì rời khỏi tiến trình.** Xem `redact()`.
   Một SDK tự động chụp prompt là một SDK tự động chụp cả `nonce` của `W4-12`.

Đây cũng đúng lý lẽ đã khiến `rag_core/llm` viết client httpx thay vì dùng SDK
`openai`: cần đúng một endpoint, và cần biết chính xác nội dung gói tin.

## ⭐⭐ Không có gì ở đây được phép làm hỏng một lượt chat

Quan sát là việc phụ. Mọi hàm công khai của module này nuốt exception và ghi
log ở mức `WARNING` — cùng hợp đồng với `SemanticCache`. Một chỗ đo hỏng làm
mất một trace; một chỗ đo *ném* làm mất một câu trả lời đã trả tiền rồi.

## ⭐ Cha–con đi bằng `ContextVar`, không bằng một cái stack

Truy hồi chạy trong `asyncio.to_thread`, nên cây span phải nối được qua ranh
giới thread. `contextvars` đi qua được ranh giới ấy (`to_thread` sao chép
context), còn một cái stack `list` dùng chung thì vừa không đi qua được vừa
hỏng câm khi có hai người ghi.

Và có **thật** một chỗ hai người ghi: `QueryUnderstanding._rewrite` bọc lời gọi
trong `asyncio.wait_for`, thứ huỷ được cái *chờ* chứ không huỷ được cái
*thread*. Quá hạn ⇒ request đi tiếp trong khi thread kia vẫn chạy nốt. Với
`ContextVar` thì thread mồ côi ấy ghi vào bản sao context của riêng nó và
không đụng vào ai; danh sách span chung vẫn có khoá vì nó là chỗ duy nhất hai
bên gặp nhau.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

__all__ = [
    "NONCE_MASK",
    "FanoutSink",
    "Span",
    "SpanKind",
    "Trace",
    "TraceSink",
    "current_trace",
    "redact",
    "trace_scope",
]

logger = logging.getLogger(__name__)

SpanKind = Literal["span", "generation"]
Level = Literal["DEFAULT", "WARNING", "ERROR"]

NONCE_MASK = "«nonce»"
"""Chỗ đứng thay cho mã phiên của `W4-12` trong mọi thứ rời khỏi tiến trình.

## ⭐⭐ Vì sao phải che, khi mã ấy chỉ sống một lượt

`W4-12` dựng hàng rào tiêm bằng một điều duy nhất không giả được: 16 ký tự hex
sinh ra **sau** khi tài liệu đã nằm trong index, nên nội dung tài liệu không
đóng được khối dữ liệu để mở một khối chỉ thị giả. Đó là lớp *duy nhất* của
hàng rào không phụ thuộc vào việc model có hợp tác hay không (`TD-53`).

Ghi nguyên prompt vào trace là đem lớp ấy sang một hệ thống thứ hai — một hệ
thống có xác thực khác, có người xem khác, và là loại hệ thống người ta chụp
màn hình rồi dán vào báo cáo. DoD của chính hạng mục này là *"Evidence:
screenshot trace"*.

Đúng là mã hết hiệu lực khi lượt kết thúc, nên rò một mã cũ không mở được cửa
nào. Nhưng che nó **không tốn gì**: người đọc trace cần biết khối ngữ cảnh có
được bọc hay không, chứ không cần biết bọc bằng chuỗi nào. Đổi một thứ không
mất gì lấy một thứ có thể mất là phép đổi nên làm kể cả khi rủi ro nhỏ.
"""

_NONCE_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{16}(?![0-9a-f])")
"""Đúng hình dạng `context_nonce()` sinh ra: 16 hex, không dính hex hai đầu.

⚠️ Đây là một bộ lọc theo **hình dạng**, nên nó cũng che một chuỗi hex 16 ký tự
tình cờ nằm trong tài liệu. Hướng nhầm ấy là hướng đúng: che thừa một chuỗi vô
hại làm trace xấu đi một chút, để lọt một mã thật làm hàng rào yếu đi thật.
"""

_MAX_TEXT = 4_000
"""Trần ký tự cho mỗi trường text đi vào trace.

Một khối ngữ cảnh 5 chunk là ~10k ký tự, và nhân với số span thì một trace
thành vài trăm KB. Cắt ở đây chứ không ở sink: sink nào cũng phải chịu cùng
trần này, và một trần đặt ở chỗ xuất là một trần quên được."""


def redact(value: Any) -> Any:
    """Che `nonce` và cắt ngắn — áp cho mọi thứ đi vào `input`/`output` của span."""
    if isinstance(value, str):
        masked = _NONCE_RE.sub(NONCE_MASK, value)
        if len(masked) > _MAX_TEXT:
            return f"{masked[:_MAX_TEXT]}… (cắt {len(masked) - _MAX_TEXT} ký tự)"
        return masked
    if isinstance(value, Mapping):
        return {str(key): redact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


@dataclass
class Usage:
    """Token và tiền của một bước. Mọi trường `None` = **chưa đo**, không phải 0.

    ⚠️ Phân biệt này là cả điểm của một bảng chi phí. `W5-04` đã trả giá một lần
    cho việc đọc "không biết" thành một con số: nhánh judge hỏng cho
    faithfulness `1,0000` vì 32 phán quyết mất được lặng lẽ bỏ khỏi mẫu số. Ở
    đây, một bước có `cost_usd = None` phải hiện ra là *chưa đo* chứ không cộng
    `0.0` vào tổng — nếu không thì tổng chi phí của trace luôn là một cận dưới
    trông như một phép đo.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None

    @property
    def empty(self) -> bool:
        return (
            self.prompt_tokens is None and self.completion_tokens is None and self.cost_usd is None
        )


@dataclass
class Span:
    """Một bước. Đóng bằng `end()`, hoặc tự đóng khi ra khỏi `with`."""

    name: str
    trace: Trace
    kind: SpanKind = "span"
    parent_id: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    _t0: float = field(default_factory=time.perf_counter, repr=False)
    duration_ms: float | None = None

    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    level: Level = "DEFAULT"
    status_message: str | None = None

    model: str | None = None
    usage: Usage = field(default_factory=Usage)

    @property
    def closed(self) -> bool:
        return self.end_time is not None

    def set(self, **fields: Any) -> None:
        """Gán thêm metadata. Không đóng span."""
        try:
            self.metadata.update({key: redact(value) for key, value in fields.items()})
        except Exception:  # pragma: no cover - phòng thân, xem docstring module
            logger.warning("tracing: không gán được metadata cho span %r", self.name)

    def end(
        self,
        *,
        output: Any = None,
        level: Level | None = None,
        status: str | None = None,
        model: str | None = None,
        usage: Usage | None = None,
        **fields: Any,
    ) -> None:
        """Đóng span. Gọi lần thứ hai **không** ghi đè lần thứ nhất.

        ⭐ Bất biến ấy là thứ làm cho `with` lồng trong một đường thoát ngoại lệ
        vẫn ra số đúng: khối `except` đóng span kèm lý do, rồi `__exit__` chạy
        và phải không làm gì. Ngược lại thì mọi span hỏng đều mang thời lượng
        của cái `finally` chứ không phải của công việc.
        """
        if self.closed:
            return
        try:
            self.duration_ms = round((time.perf_counter() - self._t0) * 1000.0, 3)
            self.end_time = datetime.now(UTC)
            if output is not None:
                self.output = redact(output)
            if level is not None:
                self.level = level
            if status is not None:
                self.status_message = status[:500]
            if model is not None:
                self.model = model
            if usage is not None:
                self.usage = usage
            if fields:
                self.set(**fields)
        except Exception:  # pragma: no cover - phòng thân
            logger.warning("tracing: không đóng được span %r", self.name)
            self.end_time = datetime.now(UTC)

    def __enter__(self) -> Span:
        self._token = _current_parent.set(self)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Literal[False]:
        try:
            _current_parent.reset(self._token)
        except (ValueError, AttributeError):
            # Token của context khác — xảy ra khi span mở ở coroutine này và
            # đóng ở coroutine kia. Không phải lỗi cần ném; cái quan trọng là
            # span vẫn được đóng ngay dưới đây.
            logger.debug("tracing: token cha lệch context ở span %r", self.name)
        if exc is not None and not self.closed:
            self.end(level="ERROR", status=f"{type(exc).__name__}: {exc}")
        else:
            self.end()
        return False


class TraceSink(Protocol):
    """Nơi cây span đi tới. Hợp đồng: **không chặn**, và **không ném**."""

    def submit(self, trace: Trace) -> None: ...


@dataclass
class FanoutSink:
    """Một cây span, nhiều người tiêu thụ. `W5-07`.

    ## ⭐⭐ Thứ tự ở đây là một quyết định, không phải một danh sách

    `MetricsSink` phải đứng **trước** `LangfuseSink`. Sink thứ hai vứt trace khi
    hàng đợi đầy — có chủ đích, xem docstring của nó — và hàng đợi đầy đúng vào
    lúc hệ thống bận nhất. Đảo thứ tự thì bộ đếm RED cũng mất đúng phần tải cao,
    và một bảng thiếu đúng lúc có sự cố là một bảng nói ngược.

    Ràng buộc kèm theo: sink đứng trước phải **rẻ và đồng bộ**. `MetricsSink`
    chỉ cộng số nguyên trong bộ nhớ nên nó đủ điều kiện; một sink ghi đĩa hay
    gọi mạng thì không, và nó phải xuống cuối hàng.

    Một sink ném không được phép chặn sink sau nó — cả hai đã hứa không ném, và
    ở đây là chỗ lời hứa ấy được cưỡng chế.
    """

    sinks: tuple[TraceSink, ...]

    def submit(self, trace: Trace) -> None:
        for sink in self.sinks:
            try:
                sink.submit(trace)
            except Exception:
                logger.warning(
                    "tracing: sink %s từ chối trace %s",
                    type(sink).__name__,
                    trace.id,
                    exc_info=True,
                )


@dataclass
class Trace:
    """Một lượt `/chat`, từ lúc handler nhận request tới lúc khung cuối rời đi."""

    name: str = "chat"
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str | None = None
    """`conversation_id`. Đặt sau, vì `prepare()` mới là chỗ nó được cấp."""

    user_id: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    input: Any = None
    output: Any = None
    level: Level = "DEFAULT"
    status_message: str | None = None

    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None

    spans: list[Span] = field(default_factory=list)
    sink: TraceSink | None = None

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _finished: bool = field(default=False, repr=False)

    # ------------------------------------------------------------------ span

    def span(
        self,
        name: str,
        *,
        kind: SpanKind = "span",
        input: Any = None,
        parent: Span | None = None,
        **metadata: Any,
    ) -> Span:
        """Mở một span con. Cha mặc định là span đang mở trong context này."""
        try:
            owner = parent if parent is not None else _current_parent.get()
            child = Span(
                name=name,
                trace=self,
                kind=kind,
                parent_id=owner.id if owner is not None else None,
                input=redact(input),
                metadata={key: redact(value) for key, value in metadata.items()},
            )
            with self._lock:
                self.spans.append(child)
            return child
        except Exception:  # pragma: no cover - phòng thân
            logger.warning("tracing: không mở được span %r", name)
            return Span(name=name, trace=self)

    def find(self, name: str) -> Span | None:
        """Span **đầu tiên** mang tên này. Chỉ dùng cho test và cho log."""
        with self._lock:
            return next((s for s in self.spans if s.name == name), None)

    def children_of(self, span: Span) -> list[Span]:
        with self._lock:
            return [s for s in self.spans if s.parent_id == span.id]

    # ------------------------------------------------------------------ tổng

    def total_cost_usd(self) -> float | None:
        """Tổng tiền của trace, hoặc `None` nếu **không** bước nào khai chi phí.

        ⭐ `None` chứ không phải `0.0`, và các bước không khai bị **bỏ qua** chứ
        không cộng 0. Xem docstring `Usage`: một tổng gộp cả cái chưa đo là một
        cận dưới đội lốt một phép đo.
        """
        with self._lock:
            costs = [s.usage.cost_usd for s in self.spans if s.usage.cost_usd is not None]
        return round(sum(costs), 8) if costs else None

    def unmeasured_cost_steps(self) -> list[str]:
        """Tên các bước **gọi model** nhưng không khai được chi phí.

        Đây là mẫu số của con số ở trên. Một trace có `total_cost_usd` nhưng
        danh sách này không rỗng thì tổng ấy chưa phải tổng.
        """
        with self._lock:
            return [s.name for s in self.spans if s.kind == "generation" and s.usage.empty]

    # ------------------------------------------------------------------ đóng

    def finish(
        self,
        *,
        output: Any = None,
        level: Level | None = None,
        status: str | None = None,
    ) -> None:
        """Đóng trace và đẩy cho sink. **Idempotent** — gọi lần hai là no-op.

        ⭐ Bất biến ấy không phải để cho gọn. Trace được đóng ở hai chỗ khác
        nhau và cả hai đều có thể chạy: `api/chat.py` đóng khi `prepare()` ném
        (lúc đó chưa có `ChatTurn` nào), còn `stream_turn` đóng ở `finally` của
        chính nó. Không idempotent thì một lượt hỏng sinh **hai** trace, và
        trace thứ hai không có span nào — tức bảng của người vận hành có một
        nửa số dòng là rác đúng vào lúc có sự cố.
        """
        with self._lock:
            if self._finished:
                return
            self._finished = True
            open_spans = [s for s in self.spans if not s.closed]
        for orphan in open_spans:
            # Span còn mở lúc trace đóng nghĩa là một đường thoát nào đó không
            # đi qua `__exit__`. Đóng nó và **nói ra** — một span im lặng lấy
            # thời lượng của cả trace là một con số sai trông như một con số.
            orphan.end(level="WARNING", status="span chưa đóng khi trace kết thúc")
        try:
            self.end_time = datetime.now(UTC)
            if output is not None:
                self.output = redact(output)
            if level is not None:
                self.level = level
            if status is not None:
                self.status_message = status[:500]
            if open_spans:
                self.metadata["unclosed_spans"] = [s.name for s in open_spans]
            self.metadata.setdefault("span_count", len(self.spans))
        except Exception:  # pragma: no cover - phòng thân
            logger.warning("tracing: không hoàn tất được trace %s", self.id)
        if self.sink is None:
            return
        try:
            self.sink.submit(self)
        except Exception:
            # Sink hứa không ném; hứa không phải là bảo đảm.
            logger.warning("tracing: sink từ chối trace %s", self.id, exc_info=True)


# ------------------------------------------------------------------ context

_current_trace: ContextVar[Trace | None] = ContextVar("rag_trace", default=None)
_current_parent: ContextVar[Span | None] = ContextVar("rag_span", default=None)


def current_trace() -> Trace | None:
    """Trace của lượt đang chạy, hoặc `None` khi không ai bật quan sát."""
    return _current_trace.get()


@contextmanager
def trace_scope(trace: Trace | None, parent: Span | None = None) -> Iterator[None]:
    """Gắn `trace` (và tuỳ chọn một span cha) vào context hiện tại.

    Dùng ở đúng **hai** chỗ, và cả hai đều hẹp có chủ đích:

    * quanh lời gọi `asyncio.to_thread(retriever.retrieve, …)`, để ba lớp bọc
      của `instrument_retriever` nối được vào đúng cây dù chúng chạy ở thread
      khác và không có cách nào nhận thêm tham số (chữ ký `Retriever.retrieve`
      là hợp đồng của `rag_core`);
    * trong test.

    ⚠️ Cố ý **không** đặt ở middleware. Bọc cả tiến trình thì mọi `/health` cũng
    có trace, và quan trọng hơn: một `ContextVar` sống suốt vòng đời request là
    một chỗ để hai lượt chồng lên nhau khi ai đó tái dùng nhầm.
    """
    trace_token = _current_trace.set(trace)
    parent_token = _current_parent.set(parent)
    try:
        yield
    finally:
        _current_parent.reset(parent_token)
        _current_trace.reset(trace_token)


def hits_summary(hits: Sequence[Any], *, limit: int = 10) -> list[dict[str, Any]]:
    """Rút `RetrievedChunk` thành phần đi vào trace — **điểm số, không nội dung**.

    ⭐ Giữ cả `dense_score` lẫn `sparse_score` bên cạnh `score`, vì đó là thứ
    duy nhất trả lời được câu hỏi mà một trace truy hồi sinh ra để trả lời:
    *chunk này lọt vào nhờ nhánh nào?* Bỏ chúng đi thì span `retrieve` và span
    `rerank` chỉ còn là hai danh sách id, và chênh lệch giữa hai danh sách ấy —
    đóng góp thật của reranker — không đọc được từ trace nữa.
    """
    out: list[dict[str, Any]] = []
    for hit in hits[:limit]:
        try:
            row: dict[str, Any] = {
                "rank": getattr(hit, "rank", None),
                "chunk_id": hit.chunk.chunk_id,
                "doc_id": hit.chunk.doc_id,
                "score": round(float(hit.score), 6),
            }
            for label in ("dense_score", "sparse_score", "rerank_score"):
                value = getattr(hit, label, None)
                if value is not None:
                    row[label] = round(float(value), 6)
            out.append(row)
        except Exception:  # pragma: no cover - phòng thân
            logger.warning("tracing: không tóm tắt được một hit")
    return out
