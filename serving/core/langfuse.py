"""Đẩy cây span sang Langfuse tự dựng — `W5-06`.

Dùng httpx thẳng vào `POST /api/public/ingestion` thay vì SDK `langfuse`, cùng
lý lẽ đã viết cho `rag_core/llm/openai_compat.py`: cần đúng **một** endpoint,
và cần biết chính xác byte nào rời khỏi tiến trình. SDK Langfuse hiện đại kéo
theo cả OpenTelemetry và một bộ chụp tự động — mà "chụp tự động" ở đây nghĩa là
chụp cả `nonce` của `W4-12`, thứ `tracing.redact()` sinh ra để chặn.

## ⭐⭐ Một hệ thống quan sát không được phép giết thứ nó quan sát

Hai chế độ hỏng, và cả hai đều là chế độ hỏng của **hàng đợi**, không phải của
mạng:

* **Chặn.** Đẩy đồng bộ ở cuối `stream_turn` thì Langfuse chậm 2 giây là mọi
  câu trả lời chậm thêm 2 giây — và Langfuse chậm đúng vào lúc hệ thống đang
  tải cao, tức đúng lúc không được chậm.
* **Phình.** Hàng đợi không trần + Langfuse chết = mọi trace nằm lại trong RAM
  cho tới khi tiến trình bị OOM. Một endpoint `/chat` chết vì bộ đếm span của
  chính nó là một cách hỏng đặc biệt khó chấp nhận.

Nên: một thread nền, một `queue.Queue(maxsize=…)`, và **vứt trace mới khi đầy**
— có đếm. `dropped` đi vào `/admin/tracing`, vì một con số bị vứt mà không ai
biết thì bảng Langfuse trở thành một mẫu thiên lệch: nó mất đúng những lượt
xảy ra lúc hệ thống bận nhất.

⚠️ Vứt **cái mới**, không vứt cái cũ. Lúc nghẽn, trace cũ đã chờ lâu nhất là
trace gần nhất với sự cố đang diễn ra; đổi nó lấy một trace vừa đến là bỏ đúng
bằng chứng cần đọc.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import queue
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

from rag_core.generation.guardrails import redact_pii
from serving.core.tracing import Span, Trace

__all__ = ["LangfuseSink", "Score", "build_sink", "encode_score", "score_id"]

logger = logging.getLogger(__name__)

_INGESTION_PATH = "/api/public/ingestion"


def _redact(value: Any) -> Any:
    """`redact_pii` đệ quy trên mọi chuỗi trong một cấu trúc JSON-được.

    ## ⚠️ `NEW-08`/`AU-05`: Langfuse là biên phải redact, không phải Postgres

    `RedactingFilter` chỉ phủ Python logging; câu hỏi người dùng đi vào
    `trace.input` và comment feedback đi vào `score.comment` **không** qua nó.
    Mà `TD-73` đã ghi: mọi tenant vào MỘT project Langfuse, tenant ở đó là nhãn
    chứ không phải hàng rào — tức đây chính là mặt phẳng mà PII của khách này
    đọc được bởi người xem project. Postgres thì ngược lại: có RLS, và người
    dùng phải đọc lại được nguyên văn câu của mình — nên redact ở nguồn là sai
    chỗ, redact ở biên xuất là đúng chỗ.

    Chỉ áp lên các trường mang **nội dung người dùng** (input/output/comment/
    statusMessage), không áp lên id/tên — `redact_pii` thay chuỗi chữ số dài,
    và một `trace_id` bị thay là một điểm số không bao giờ gắn được vào trace.
    """
    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_redact(v) for v in value]
    return value


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def _usage_body(span: Span) -> dict[str, Any] | None:
    """`usage` theo giao thức ingestion. `None` khi bước này **chưa đo** gì.

    ⭐ Không dựng `{"input": 0, "output": 0, "totalCost": 0}` cho một bước không
    có số. Langfuse cộng mọi `totalCost` nó nhận được, nên một số 0 khai hộ sẽ
    biến "chưa đo" thành "miễn phí" trong đúng cái bảng người ta mở ra để trả
    lời *một câu hỏi tốn bao nhiêu*.
    """
    usage = span.usage
    if usage.empty:
        return None
    body: dict[str, Any] = {"unit": "TOKENS"}
    if usage.prompt_tokens is not None:
        body["input"] = usage.prompt_tokens
    if usage.completion_tokens is not None:
        body["output"] = usage.completion_tokens
    if usage.prompt_tokens is not None and usage.completion_tokens is not None:
        body["total"] = usage.prompt_tokens + usage.completion_tokens
    if usage.cost_usd is not None:
        body["totalCost"] = usage.cost_usd
    return body


def encode_trace(trace: Trace) -> list[dict[str, Any]]:
    """Cây span → danh sách sự kiện ingestion. Thuần, không I/O, test được."""
    now = _iso(trace.end_time) or _iso(trace.start_time)
    events: list[dict[str, Any]] = [
        {
            "id": uuid.uuid4().hex,
            "timestamp": now,
            "type": "trace-create",
            "body": {
                "id": trace.id,
                "name": trace.name,
                "userId": trace.user_id,
                "sessionId": trace.session_id,
                "timestamp": _iso(trace.start_time),
                "input": _redact(trace.input),
                "output": _redact(trace.output),
                "tags": trace.tags,
                "metadata": {
                    **trace.metadata,
                    "level": trace.level,
                    "status_message": trace.status_message,
                    "total_cost_usd": trace.total_cost_usd(),
                    # Xem `Trace.unmeasured_cost_steps`: tổng ở trên chỉ là một
                    # tổng khi danh sách này rỗng.
                    "unmeasured_cost_steps": trace.unmeasured_cost_steps(),
                },
            },
        }
    ]
    for span in trace.spans:
        body: dict[str, Any] = {
            "id": span.id,
            "traceId": trace.id,
            "parentObservationId": span.parent_id,
            "name": span.name,
            "startTime": _iso(span.start_time),
            "endTime": _iso(span.end_time),
            "input": _redact(span.input),
            "output": _redact(span.output),
            "metadata": {**span.metadata, "duration_ms": span.duration_ms},
            "level": span.level,
            "statusMessage": _redact(span.status_message),
        }
        if span.kind == "generation":
            body["model"] = span.model
            usage = _usage_body(span)
            if usage is not None:
                body["usage"] = usage
        events.append(
            {
                "id": uuid.uuid4().hex,
                "timestamp": _iso(span.end_time) or now,
                "type": "generation-create" if span.kind == "generation" else "span-create",
                "body": body,
            }
        )
    return events


# ------------------------------------------------------------------- điểm số


@dataclass(frozen=True)
class Score:
    """Một điểm số gắn vào trace — `W5-08`.

    Đi qua **cùng** hàng đợi với trace, không qua một đường riêng: một endpoint
    feedback chặn 300 ms vì Langfuse chậm là một nút 👎 mà người dùng bấm hai
    lần.
    """

    trace_id: str
    name: str
    value: float
    comment: str | None = None
    data_type: str = "NUMERIC"


def score_id(trace_id: str, name: str) -> str:
    """Id tất định cho một cặp (trace, tên điểm).

    ⭐⭐ **Một luật idempotent cho hai kho.** Postgres upsert theo
    `(tenant_id, message_id)`; Langfuse upsert theo `id` của điểm. Nếu id ở đây
    ngẫu nhiên thì đổi 👎 thành 👍 để lại **hai** điểm trái ngược trên cùng một
    trace, và cái trung bình hiện trên bảng Langfuse không tương ứng với bất kỳ
    hàng nào trong Postgres. Hai kho phải cùng một khái niệm "lần chấm này".
    """
    return hashlib.sha256(f"{trace_id}:{name}".encode()).hexdigest()[:32]


def encode_score(score: Score) -> list[dict[str, Any]]:
    """Điểm số → sự kiện ingestion. Thuần, không I/O."""
    return [
        {
            "id": uuid.uuid4().hex,
            "timestamp": _iso(datetime.now(UTC)),
            "type": "score-create",
            "body": {
                "id": score_id(score.trace_id, score.name),
                "traceId": score.trace_id,
                "name": score.name,
                "value": score.value,
                "dataType": score.data_type,
                # Comment feedback đã redact ở nguồn (`record_feedback`), nhưng
                # biên này không được *phụ thuộc* vào điều đó — hai lớp rẻ hơn
                # một lần rò.
                "comment": _redact(score.comment),
            },
        }
    ]


@dataclass
class LangfuseSink:
    """Thread nền + hàng đợi có trần. `submit()` không chặn và không ném."""

    host: str
    public_key: str
    secret_key: str
    max_queue: int = 256
    timeout_s: float = 10.0
    client: httpx.Client | None = None

    _queue: queue.Queue[Trace | Score | None] = field(init=False, repr=False)
    _worker: threading.Thread | None = field(default=None, init=False, repr=False)
    _sent: int = field(default=0, init=False)
    _failed: int = field(default=0, init=False)
    _dropped: int = field(default=0, init=False)
    _scored: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._queue = queue.Queue(maxsize=self.max_queue)
        if self.client is None:
            self.client = httpx.Client(
                base_url=self.host.rstrip("/"),
                timeout=self.timeout_s,
                auth=(self.public_key, self.secret_key),
                headers={"Content-Type": "application/json"},
            )
        self._worker = threading.Thread(target=self._run, name="langfuse-sink", daemon=True)
        self._worker.start()

    # ---------------------------------------------------------------- công khai

    def submit(self, trace: Trace) -> None:
        self._enqueue(trace)

    def submit_score(self, score: Score) -> None:
        """Xếp một điểm số vào cùng hàng đợi. Không chặn, không ném.

        ⚠️ Xếp **sau** trace của cùng lượt, và Langfuse nhận điểm trỏ vào một
        trace chưa tồn tại (`traceId` là một chuỗi, không phải khoá ngoại) — nó
        ghép lại khi trace tới. Nhưng nếu trace bị vứt vì hàng đợi đầy thì điểm
        ấy treo vĩnh viễn, nên `dropped` phải đọc kèm `scored`.
        """
        self._enqueue(score)

    def _enqueue(self, item: Trace | Score) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self._dropped += 1
            if self._dropped % 50 == 1:
                # Không log mỗi lần: lúc nghẽn thì chính dòng log là thứ tiếp
                # theo làm nghẽn. Log lần đầu rồi mỗi 50 lần.
                logger.warning(
                    "tracing: hàng đợi Langfuse đầy (%d), đã vứt %d trace",
                    self.max_queue,
                    self._dropped,
                )
        except Exception:  # pragma: no cover - phòng thân
            logger.warning("tracing: không xếp được trace vào hàng đợi")

    def status(self) -> dict[str, Any]:
        """Đủ để một lệnh `curl` trả lời "trace có thật sự tới nơi không"."""
        return {
            "host": self.host,
            "queued": self._queue.qsize(),
            "sent": self._sent,
            "scored": self._scored,
            "failed": self._failed,
            "dropped": self._dropped,
        }

    def close(self, *, timeout_s: float = 5.0) -> None:
        """Đẩy nốt rồi dừng. Gọi ở `lifespan` lúc tắt.

        ⚠️ Có trần thời gian. Một Langfuse chết không được phép giữ tiến trình
        không tắt được — cùng đánh đổi mà `W4-06` đã chọn cho task ghi Postgres,
        và cùng giới hạn thật: tắt lúc còn hàng đợi ⇒ mất trace.
        """
        with contextlib.suppress(queue.Full):  # hàng đợi đầy ⇒ worker vẫn thoát
            self._queue.put_nowait(None)
        if self._worker is not None:
            self._worker.join(timeout=timeout_s)
        if self.client is not None:
            self.client.close()

    # ------------------------------------------------------------------ nền

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._send(item)
            except Exception:
                self._failed += 1
                logger.warning("tracing: đẩy %s thất bại", _label(item), exc_info=True)
            finally:
                self._queue.task_done()

    def _send(self, item: Trace | Score) -> None:
        assert self.client is not None
        batch = encode_score(item) if isinstance(item, Score) else encode_trace(item)
        payload = {"batch": batch}
        response = self.client.post(_INGESTION_PATH, content=json.dumps(payload, default=str))
        if response.status_code >= 400:
            self._failed += 1
            # ⚠️ **Không** log `response.request.headers` — ở đó có Basic auth
            # mang `secret_key`. Một dòng log gỡ rối là cách phổ biến nhất để
            # một khoá đi vào một hệ thống log có quyền đọc rộng hơn nó.
            logger.warning(
                "tracing: Langfuse trả %d cho %s: %s",
                response.status_code,
                _label(item),
                response.text[:300],
            )
            return
        if isinstance(item, Score):
            self._scored += 1
        else:
            self._sent += 1


def build_sink(settings: Any) -> LangfuseSink | None:
    """`None` = tắt quan sát. Bật **chỉ** khi có đủ cả hai khoá.

    ⭐ Không có cờ `tracing_enabled` riêng, và đó là có chủ đích: một cờ bật/tắt
    bên cạnh một cặp khoá cho **bốn** trạng thái cấu hình trong đó hai là mâu
    thuẫn ("bật nhưng không có khoá", "tắt nhưng có khoá"), và cái người ta gặp
    trong thực tế là trạng thái thứ nhất — mọi thứ trông như đã bật, không
    trace nào tới nơi, không lỗi nào được ném.
    """
    host = _plain(getattr(settings, "langfuse_host", ""))
    public_key = _plain(getattr(settings, "langfuse_public_key", ""))
    secret_key = _plain(getattr(settings, "langfuse_secret_key", ""))
    if not (host and public_key and secret_key):
        logger.info("tracing: chưa cấu hình Langfuse (thiếu host/khoá) — không có trace")
        return None
    if not _is_local(host):
        # ⚠️⚠️ Trace mang **câu hỏi của người dùng** và **điểm số kèm id chunk**
        # của corpus. Gửi ra một host ngoài là một quyết định về dữ liệu, không
        # phải một dòng cấu hình. Không chặn — chặn thì người có quyền làm việc
        # ấy không làm được — nhưng phải nói ra, mỗi lần khởi động.
        logger.warning(
            "tracing: LANGFUSE_HOST=%s KHÔNG phải máy cục bộ — nội dung câu hỏi "
            "và metadata corpus sẽ rời khỏi máy này",
            host,
        )
    return LangfuseSink(
        host=host,
        public_key=public_key,
        secret_key=secret_key,
        max_queue=int(getattr(settings, "langfuse_queue_size", 256)),
    )


def _label(item: Trace | Score) -> str:
    if isinstance(item, Score):
        return f"score {item.name} của trace {item.trace_id}"
    return f"trace {item.id}"


def _plain(value: Any) -> str:
    """Mở `SecretStr` nếu có. Đây là **chỗ duy nhất** khoá thành `str` trần —
    giữ nó ở một hàm để mọi đường khác vẫn cầm kiểu che được lúc log."""
    reveal = getattr(value, "get_secret_value", None)
    if callable(reveal):
        revealed: str = reveal()
        return revealed
    return str(value or "")


def _is_local(host: str) -> bool:
    lowered = host.lower()
    # `host.docker.internal` = máy chủ đang chạy container này, tức vẫn là
    # "cục bộ" theo nghĩa dữ liệu không rời khỏi máy. Không có nó ở đây thì
    # đường chạy trong compose cảnh báo nhầm ở MỌI lần khởi động — và một
    # cảnh báo luôn sai là một cảnh báo không ai đọc nữa.
    marks = ("localhost", "127.0.0.1", "://langfuse", "[::1]", "host.docker.internal")
    return any(mark in lowered for mark in marks)
