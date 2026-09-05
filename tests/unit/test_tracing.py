"""Cây span, che dữ liệu, kế toán chi phí, và sink — `W5-06`.

Không có bài nào ở đây cần Langfuse chạy. Đó là điều kiện để phép đếm span của
DoD sống được trong CI: một phép kiểm chỉ chạy khi có hạ tầng là một phép kiểm
sẽ không chạy.
"""

from __future__ import annotations

import json
import threading
from typing import Any

import httpx
import pytest

from serving.core.langfuse import LangfuseSink, build_sink, encode_trace
from serving.core.tracing import (
    NONCE_MASK,
    Trace,
    Usage,
    current_trace,
    hits_summary,
    redact,
    trace_scope,
)

NONCE = "a1b2c3d4e5f60718"


# ---------------------------------------------------------------------------
# 1. Che dữ liệu
# ---------------------------------------------------------------------------


class TestRedaction:
    def test_the_injection_nonce_never_leaves_the_process(self) -> None:
        """`W4-12`: 16 hex ấy là lớp DUY NHẤT của hàng rào không phụ thuộc model.

        DoD của chính hạng mục này là *"screenshot trace"* — tức trace là thứ
        được chụp và dán đi. Xem `tracing.NONCE_MASK`.
        """
        prompt = f"<ctx nonce={NONCE}>tài liệu</ctx nonce={NONCE}>"
        assert NONCE not in redact(prompt)
        assert redact(prompt).count(NONCE_MASK) == 2

    def test_it_reaches_inside_nested_structures(self) -> None:
        payload = {"messages": [{"role": "system", "content": f"nonce={NONCE}"}]}
        assert NONCE not in str(redact(payload))

    def test_a_shorter_hex_run_is_left_alone(self) -> None:
        """Bộ lọc theo hình dạng phải khớp ĐÚNG 16, không phải "ít nhất 16"."""
        assert redact("deadbeef") == "deadbeef"
        assert redact("a" * 15) == "a" * 15

    def test_a_longer_hex_run_is_left_alone_too(self) -> None:
        """Một sha256 trong tài liệu không phải một nonce, và che nó đi làm mất
        đúng thứ người ta cần đọc khi gỡ lỗi bundle."""
        digest = "0" * 64
        assert redact(digest) == digest

    def test_long_text_is_truncated_and_says_so(self) -> None:
        out = redact("x" * 10_000)
        assert isinstance(out, str)
        assert len(out) < 5_000
        assert "cắt" in out

    def test_non_text_values_pass_through_untouched(self) -> None:
        assert redact(3) == 3
        assert redact(None) is None
        assert redact(0.5) == 0.5


# ---------------------------------------------------------------------------
# 2. Cây span
# ---------------------------------------------------------------------------


class TestTree:
    def test_a_span_opened_inside_another_is_its_child(self) -> None:
        trace = Trace()
        with trace.span("outer") as outer, trace.span("inner") as inner:
            assert inner.parent_id == outer.id
        assert trace.children_of(outer) == [inner]

    def test_a_top_level_span_has_no_parent(self) -> None:
        trace = Trace()
        with trace.span("solo") as span:
            pass
        assert span.parent_id is None

    def test_leaving_the_with_block_closes_the_span(self) -> None:
        trace = Trace()
        with trace.span("s") as span:
            assert not span.closed
        # `Span.__exit__` khai `Literal[False]`, nên mypy biết khối `with` không
        # nuốt ngoại lệ và coi mọi dòng sau nó là **đến được**. Đọc `span` từ
        # `trace` thay vì từ biến của `with` giữ đúng ý mà không cần `type: ignore`.
        closed = trace.find("s")
        assert closed is not None
        assert closed.closed
        assert closed.duration_ms is not None

    def test_an_exception_marks_the_span_error_and_still_propagates(self) -> None:
        trace = Trace()
        with pytest.raises(ValueError), trace.span("s"):
            raise ValueError("nổ")
        span = trace.find("s")
        assert span is not None
        assert span.level == "ERROR"
        assert "nổ" in (span.status_message or "")

    def test_ending_a_span_twice_keeps_the_first_answer(self) -> None:
        """Khối `except` đóng span kèm lý do, rồi `__exit__` chạy. Nếu lần hai
        ghi đè thì mọi span hỏng mang thời lượng của `finally`, không phải của
        công việc."""
        trace = Trace()
        span = trace.span("s")
        span.end(output="thật", level="ERROR", status="lý do thật")
        first = span.duration_ms
        span.end(output="ghi đè", level="DEFAULT")
        assert span.output == "thật"
        assert span.level == "ERROR"
        assert span.status_message == "lý do thật"
        assert span.duration_ms == first

    def test_a_span_survives_the_thread_boundary(self) -> None:
        """`retrieve()` chạy trong `asyncio.to_thread`; nếu `ContextVar` không
        đi qua được ranh giới ấy thì mọi span truy hồi thành mồ côi."""
        trace = Trace()
        seen: list[Any] = []

        def worker() -> None:
            seen.append(current_trace())

        with trace_scope(trace):
            thread = threading.Thread(target=_with_context(worker))
            thread.start()
            thread.join()
        assert seen == [trace]

    def test_the_context_is_reset_when_the_scope_ends(self) -> None:
        trace = Trace()
        with trace_scope(trace):
            assert current_trace() is trace
        assert current_trace() is None


def _with_context(fn: Any) -> Any:
    """`threading.Thread` KHÔNG sao chép context; `asyncio.to_thread` thì có.
    Bài test trên mô phỏng đúng hành vi của cái thứ hai."""
    import contextvars

    ctx = contextvars.copy_context()
    return lambda: ctx.run(fn)


# ---------------------------------------------------------------------------
# 3. Đóng trace
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.traces: list[Trace] = []

    def submit(self, trace: Trace) -> None:
        self.traces.append(trace)


class TestFinish:
    def test_finishing_twice_submits_once(self) -> None:
        """Một lượt hỏng đóng trace ở `api/chat.py` **và** ở `finally` của
        `stream_turn`. Không idempotent thì bảng người vận hành có một nửa số
        dòng là trace rỗng, đúng vào lúc có sự cố."""
        sink = _Recorder()
        trace = Trace(sink=sink)
        trace.finish(status="một")
        trace.finish(status="hai")
        assert len(sink.traces) == 1
        assert trace.status_message == "một"

    def test_an_unclosed_span_is_closed_and_named(self) -> None:
        trace = Trace()
        span = trace.span("bỏ quên")
        trace.finish()
        assert span.closed
        assert span.level == "WARNING"
        assert trace.metadata["unclosed_spans"] == ["bỏ quên"]

    def test_a_sink_that_raises_does_not_take_the_turn_down(self) -> None:
        class Angry:
            def submit(self, trace: Trace) -> None:
                raise RuntimeError("sink chết")

        Trace(sink=Angry()).finish()  # không ném là toàn bộ nội dung bài này

    def test_a_trace_with_no_sink_is_still_a_valid_tree(self) -> None:
        trace = Trace()
        with trace.span("s"):
            pass
        trace.finish()
        assert trace.metadata["span_count"] == 1


# ---------------------------------------------------------------------------
# 4. Kế toán chi phí — "chưa đo" khác "0"
# ---------------------------------------------------------------------------


class TestCost:
    def test_a_step_that_measured_nothing_is_not_counted_as_free(self) -> None:
        """Bước `rewrite` quá hạn vẫn bị nhà cung cấp tính tiền (`wait_for` huỷ
        cái *chờ*, không huỷ cái *thread*). Cộng 0 cho nó là biến một khoản chi
        không quan sát được thành một khoản chi bằng không."""
        trace = Trace()
        paid = trace.span("completion", kind="generation")
        paid.end(usage=Usage(cost_usd=0.002))
        lost = trace.span("rewrite", kind="generation")
        lost.end(level="WARNING", status="quá hạn")
        assert trace.total_cost_usd() == pytest.approx(0.002)
        assert trace.unmeasured_cost_steps() == ["rewrite"]

    def test_a_trace_where_nothing_measured_reports_none_not_zero(self) -> None:
        trace = Trace()
        trace.span("rewrite", kind="generation").end()
        assert trace.total_cost_usd() is None

    def test_plain_spans_never_appear_in_the_unmeasured_list(self) -> None:
        """Một span `rerank` không khai chi phí vì nó chạy trên GPU của chính
        mình, không vì phép đo hỏng."""
        trace = Trace()
        trace.span("rerank").end()
        assert trace.unmeasured_cost_steps() == []


# ---------------------------------------------------------------------------
# 5. Tóm tắt hit
# ---------------------------------------------------------------------------


class _Chunk:
    chunk_id = "c1"
    doc_id = "d1"


class _Hit:
    chunk = _Chunk()
    score = 0.5
    rank = 1
    dense_score = 0.4
    sparse_score = 0.1
    rerank_score = 0.5


class TestHitsSummary:
    def test_both_branch_scores_survive_into_the_trace(self) -> None:
        """Bỏ `dense_score`/`sparse_score` đi thì span `retrieve` và span
        `rerank` chỉ còn là hai danh sách id, và đóng góp thật của reranker
        không đọc được từ trace nữa."""
        row = hits_summary([_Hit()])[0]
        assert row["dense_score"] == 0.4
        assert row["sparse_score"] == 0.1
        assert row["rerank_score"] == 0.5

    def test_it_carries_no_chunk_text(self) -> None:
        """Nội dung chunk đã có trong span `prompt`; nhân nó lên 50 ứng viên
        rerank là một trace vài trăm KB cho mỗi câu hỏi."""
        assert "content" not in hits_summary([_Hit()])[0]

    def test_the_list_is_capped(self) -> None:
        assert len(hits_summary([_Hit()] * 50)) == 10


# ---------------------------------------------------------------------------
# 6. Mã hoá sang giao thức ingestion
# ---------------------------------------------------------------------------


def _finished_trace() -> Trace:
    trace = Trace(name="chat", session_id="conv1", user_id="acme", input="RRF là gì?")
    with trace.span("understand"):
        pass
    gen = trace.span("completion", kind="generation")
    gen.end(
        model="deepseek-v4-flash",
        usage=Usage(prompt_tokens=100, completion_tokens=20, cost_usd=0.0003),
    )
    trace.finish(output="xong")
    return trace


class TestEncode:
    def test_the_first_event_creates_the_trace(self) -> None:
        events = encode_trace(_finished_trace())
        assert events[0]["type"] == "trace-create"
        assert events[0]["body"]["sessionId"] == "conv1"

    def test_pii_in_the_question_never_reaches_langfuse(self) -> None:
        """`NEW-08`/`AU-05`: `RedactingFilter` chỉ phủ logging — câu hỏi người
        dùng vào `trace.input` không qua nó. Mà `TD-73` đã ghi: mọi tenant vào
        MỘT project Langfuse, tenant ở đó là nhãn chứ không phải hàng rào — tức
        đây là mặt phẳng mà PII phải bị chặn ở biên, kể cả trong span con và
        trong `statusMessage` (một exception có thể mang nguyên câu hỏi)."""
        trace = Trace(
            name="chat",
            session_id="conv1",
            user_id="acme",
            input="Số của tôi là 0912345678, email toi@example.com",
        )
        with trace.span("understand", input="gọi lại 0912345678 giúp"):
            pass
        trace.finish(output="đã ghi nhận toi@example.com")

        blob = json.dumps(encode_trace(trace), ensure_ascii=False)
        assert "0912345678" not in blob
        assert "toi@example.com" not in blob

    def test_ids_survive_redaction_untouched(self) -> None:
        """`redact_pii` thay chuỗi chữ số dài — áp bừa lên cả cây thì một
        `trace_id` toàn số bị thay và điểm số không bao giờ gắn được vào
        trace. Redact chỉ đụng các trường nội dung; id phải nguyên vẹn."""
        trace = _finished_trace()
        events = encode_trace(trace)
        assert events[0]["body"]["id"] == trace.id
        span_ids = {e["body"]["id"] for e in events[1:]}
        assert span_ids == {span.id for span in trace.spans}

    def test_every_span_becomes_one_event(self) -> None:
        trace = _finished_trace()
        assert len(encode_trace(trace)) == 1 + len(trace.spans)

    def test_a_generation_carries_model_and_usage(self) -> None:
        body = _event(encode_trace(_finished_trace()), "completion")
        assert body["model"] == "deepseek-v4-flash"
        assert body["usage"] == {
            "unit": "TOKENS",
            "input": 100,
            "output": 20,
            "total": 120,
            "totalCost": 0.0003,
        }

    def test_a_plain_span_carries_no_usage_key_at_all(self) -> None:
        """Langfuse cộng mọi `totalCost` nó nhận được. Một `{"totalCost": 0}`
        khai hộ biến "chưa đo" thành "miễn phí" trong bảng chi phí."""
        assert "usage" not in _event(encode_trace(_finished_trace()), "understand")

    def test_an_unmeasured_generation_carries_no_usage_either(self) -> None:
        trace = Trace()
        trace.span("rewrite", kind="generation").end()
        trace.finish()
        assert "usage" not in _event(encode_trace(trace), "rewrite")

    def test_nesting_survives_the_encoding(self) -> None:
        """Cây span đúng trong bộ nhớ **không** có nghĩa là Langfuse thấy cây.

        `parentObservationId` là trường duy nhất mang quan hệ cha–con qua giao
        thức ingestion. Mất nó thì server nhận đủ số quan sát, trả `207`, và vẽ
        ra một danh sách **phẳng** — `rerank` thôi nằm trong `retrieval`, và
        bảng thời lượng bắt đầu cộng trùng. Không có bài này thì phép tiêm bỏ
        trường ấy đi vẫn xanh (`L2`).
        """
        trace = Trace()
        with trace.span("retrieval") as outer, trace.span("rerank") as inner:
            pass
        trace.finish()
        bodies = {e["body"]["name"]: e["body"] for e in encode_trace(trace)[1:]}
        assert bodies["retrieval"]["parentObservationId"] is None
        assert bodies["rerank"]["parentObservationId"] == outer.id
        assert bodies["rerank"]["id"] == inner.id

    def test_the_trace_metadata_names_the_steps_it_could_not_price(self) -> None:
        trace = Trace()
        trace.span("rewrite", kind="generation").end()
        trace.finish()
        meta = encode_trace(trace)[0]["body"]["metadata"]
        assert meta["unmeasured_cost_steps"] == ["rewrite"]
        assert meta["total_cost_usd"] is None


def _event(events: list[dict[str, Any]], name: str) -> dict[str, Any]:
    body: dict[str, Any] = next(e["body"] for e in events[1:] if e["body"]["name"] == name)
    return body


# ---------------------------------------------------------------------------
# 7. Sink
# ---------------------------------------------------------------------------


class TestSink:
    def test_a_trace_reaches_the_ingestion_endpoint(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(207, json={"successes": [], "errors": []})

        sink = _sink(handler)
        sink.submit(_finished_trace())
        sink.close()
        assert len(seen) == 1
        assert seen[0].url.path == "/api/public/ingestion"
        assert sink.status()["sent"] == 1

    def test_a_dead_langfuse_never_reaches_the_caller(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("không cắm được")

        sink = _sink(handler)
        sink.submit(_finished_trace())
        sink.close()
        assert sink.status()["failed"] == 1

    def test_a_full_queue_drops_and_counts_instead_of_growing(self) -> None:
        """Hàng đợi không trần + Langfuse chết = `/chat` chết vì bộ đếm span của
        chính nó."""
        sink = _sink(_blocked, max_queue=1)
        for _ in range(6):
            sink.submit(_finished_trace())
        # 1 chỗ trong hàng đợi + 1 cái worker đã nhấc ra và đang kẹt ở `_send`.
        assert sink.status()["dropped"] >= 4
        assert sink._queue.qsize() <= 1
        _RELEASE.set()
        sink.close()

    def test_submit_never_raises_even_on_a_broken_queue(self) -> None:
        sink = _sink(lambda r: httpx.Response(207, json={}))
        broken, sink._queue = sink._queue, None  # type: ignore[assignment]
        try:
            sink.submit(_finished_trace())  # không ném là toàn bộ nội dung bài
        finally:
            sink._queue = broken
            sink.close()

    def test_a_4xx_is_logged_and_counted_not_raised(self) -> None:
        sink = _sink(lambda r: httpx.Response(401, text="unauthorised"))
        sink.submit(_finished_trace())
        sink.close()
        assert sink.status()["failed"] == 1
        assert sink.status()["sent"] == 0


_RELEASE = threading.Event()


def _blocked(request: httpx.Request) -> httpx.Response:
    """Giữ worker lại ở lời gọi mạng, để hàng đợi thật sự đầy."""
    _RELEASE.wait(timeout=5.0)
    return httpx.Response(207, json={})


def _sink(handler: Any, *, max_queue: int = 16) -> LangfuseSink:
    return LangfuseSink(
        host="http://127.0.0.1:3000",
        public_key="pk",
        secret_key="sk",
        max_queue=max_queue,
        client=httpx.Client(
            base_url="http://127.0.0.1:3000", transport=httpx.MockTransport(handler)
        ),
    )


# ---------------------------------------------------------------------------
# 8. Bật/tắt
# ---------------------------------------------------------------------------


class _Cfg:
    def __init__(self, **kwargs: Any) -> None:
        self.langfuse_host = kwargs.get("host", "http://127.0.0.1:3000")
        self.langfuse_public_key = kwargs.get("public", "pk")
        self.langfuse_secret_key = kwargs.get("secret", "sk")
        self.langfuse_queue_size = 8


class TestBuildSink:
    def test_no_keys_means_no_sink(self) -> None:
        assert build_sink(_Cfg(secret="")) is None
        assert build_sink(_Cfg(public="")) is None
        assert build_sink(_Cfg(host="")) is None

    def test_a_secretstr_is_unwrapped(self) -> None:
        from pydantic import SecretStr

        sink = build_sink(_Cfg(secret=SecretStr("sk-real")))
        assert sink is not None
        assert sink.secret_key == "sk-real"
        sink.close()

    def test_a_non_local_host_is_flagged_loudly(self, caplog: Any) -> None:
        """Trace mang câu hỏi của người dùng. Trỏ nó ra một host ngoài là một
        quyết định về dữ liệu, không phải một dòng cấu hình."""
        sink = build_sink(_Cfg(host="https://cloud.langfuse.com"))
        assert sink is not None
        sink.close()
        assert any("KHÔNG phải máy cục bộ" in r.message for r in caplog.records)

    def test_the_docker_host_gateway_counts_as_local(self, caplog: Any) -> None:
        """Container API gọi Langfuse qua `host.docker.internal` (hai compose
        project ⇒ hai network ⇒ không có tên service nào để trỏ tới). Dữ liệu
        vẫn không rời khỏi máy, nên cảnh báo ở đây là cảnh báo **luôn sai** —
        và một cảnh báo luôn sai là một cảnh báo không ai đọc nữa."""
        sink = build_sink(_Cfg(host="http://host.docker.internal:3000"))
        assert sink is not None
        sink.close()
        assert not any("KHÔNG phải máy cục bộ" in r.message for r in caplog.records)

    def test_a_compose_service_name_counts_as_local(self, caplog: Any) -> None:
        sink = build_sink(_Cfg(host="http://langfuse-web:3000"))
        assert sink is not None
        sink.close()
        assert not any("KHÔNG phải máy cục bộ" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 9. Bọc chuỗi truy hồi
# ---------------------------------------------------------------------------


class _Base:
    name = "fake-hybrid"
    k = 60

    def __init__(self) -> None:
        self.store = _Store()

    def retrieve(self, query: str, top_k: int = 10, *, filters: Any = None) -> list[Any]:
        return [_Hit()]


class _Store:
    embeddings = type("E", (), {"name": "bge-m3", "embed_query": staticmethod(lambda q: [0.0])})()


class _Reranker:
    name = "bge-reranker"

    def score(self, query: str, texts: Any) -> list[float]:
        return [1.0 for _ in texts]


class _Reranked:
    name = "reranked[fake-hybrid]:bge-reranker:n50"

    def __init__(self) -> None:
        self.base: Any = _Base()
        self.reranker: Any = _Reranker()

    def retrieve(self, query: str, top_k: int = 10, *, filters: Any = None) -> list[Any]:
        pool: list[Any] = self.base.retrieve(query, 50, filters=filters)
        self.reranker.score(query, [h.chunk.chunk_id for h in pool])
        return pool


class TestInstrument:
    def test_the_original_chain_is_left_untouched(self) -> None:
        """`ActiveBundle` là ảnh chụp bất biến và có thể đang được nhiều request
        cầm (`W4-02`). Sửa nó tại chỗ cũng có nghĩa là bọc lần thứ hai sẽ lồng
        một lớp bọc vào trong một lớp bọc, mãi mãi."""
        from serving.core.instrument import instrument_retriever

        original = _Reranked()
        base, reranker = original.base, original.reranker
        instrument_retriever(original)  # type: ignore[arg-type]
        assert original.base is base
        assert original.reranker is reranker

    def test_copying_a_wrapper_does_not_recurse(self) -> None:
        """Hồi quy cho một lỗi thật mà chính bài test này tìm ra.

        `copy.copy` dựng thực thể **không qua `__init__`** rồi tra
        `__setstate__`; với một `__getattr__` uỷ quyền ngây thơ thì `_inner`
        chưa tồn tại và lời tra ấy gọi lại chính nó tới `RecursionError`. Chỗ
        gọi `copy.copy` là `instrument_retriever`, nên lỗi nổ đúng khi bọc một
        chuỗi đã bọc. Xem `_delegate`.
        """
        import copy as copy_module

        from serving.core.instrument import instrument_retriever

        wrapped = instrument_retriever(_Reranked())  # type: ignore[arg-type]
        assert copy_module.copy(wrapped) is not wrapped
        assert instrument_retriever(wrapped).name == wrapped.name

    def test_the_wrapper_does_not_hide_the_embedder_from_the_cache(self) -> None:
        """`embedder_of()` của `W4-10` đào `retriever.base.store.embeddings` bằng
        duck-typing. Một lớp bọc không uỷ quyền làm semantic cache tắt **lặng
        lẽ**: miss 100%, không log, không lỗi, chỉ có hoá đơn cao hơn."""
        from serving.core.instrument import instrument_retriever
        from serving.core.semantic_cache import embedder_of

        wrapped = instrument_retriever(_Reranked())  # type: ignore[arg-type]
        assert embedder_of(wrapped) is not None

    def test_the_wrapper_keeps_the_name_the_bundle_declared(self) -> None:
        """`retriever_name` trong manifest là một phần của danh tính bundle
        (`W5-05` gate so trên nó). Một lớp bọc đổi tên là một bundle khác."""
        from serving.core.instrument import instrument_retriever

        assert instrument_retriever(_Reranked()).name == _Reranked().name  # type: ignore[arg-type]

    def test_a_bare_retriever_still_gets_one_span(self) -> None:
        from serving.core.instrument import instrument_retriever

        class Bare:
            name = "bare"

            def retrieve(self, query: str, top_k: int = 10, *, filters: Any = None) -> list[Any]:
                return []

        trace = Trace()
        with trace_scope(trace):
            instrument_retriever(Bare()).retrieve("q", 5)  # type: ignore[arg-type]
        assert [s.name for s in trace.spans] == ["retrieve"]

    def test_with_no_trace_the_wrapper_is_a_straight_pass_through(self) -> None:
        from serving.core.instrument import instrument_retriever

        assert len(instrument_retriever(_Reranked()).retrieve("q", 5)) == 1  # type: ignore[arg-type]


class TestPrecomputedThroughTheTracedChain:
    """`NEW-08`/`AU-06`, hồi hai — lỗi mà PROBE bắt được chứ không phải test.

    Production bọc cả chuỗi truy hồi bằng `TracedRetriever` (`W5-06`), nên
    `retriever.base` của server thật là `TracedRetriever(hybrid)` chứ không
    phải `QdrantHybridRetriever`: bản đầu của `wants_precomputed` isinstance
    trên class trần — mọi unit test xanh, còn server thật lặng lẽ đi đường
    embed-đôi cũ. Hai bài dưới dựng ĐÚNG chuỗi mà `instrument_retriever` dựng.
    """

    def _real_chain(self) -> Any:
        from rag_core.embedding import HashingEmbeddingProvider
        from rag_core.retrieval import (
            QdrantDenseRetriever,
            QdrantHybridRetriever,
            RerankedRetriever,
        )
        from serving.core.instrument import instrument_retriever

        store = QdrantDenseRetriever(
            HashingEmbeddingProvider(dimension=16, sparse=True),
            collection="rag_test_traced",
        )

        class _Reranker:
            name = "rr-fake"

            def score(self, query: str, texts: list[str]) -> list[float]:
                return [0.0] * len(texts)

        return instrument_retriever(RerankedRetriever(QdrantHybridRetriever(store), _Reranker()))  # type: ignore[arg-type]

    def test_an_instrumented_chain_still_qualifies_for_precomputed(self) -> None:
        from serving.core.chat import wants_precomputed
        from serving.core.semantic_cache import embedder_of

        chain = self._real_chain()
        embedder = embedder_of(chain)
        assert embedder is not None, "lớp bọc phải để embedder_of đào xuyên qua"
        assert wants_precomputed(chain, embedder), (
            "chuỗi production LUÔN bị bọc — không nhận ra nó là đường "
            "embed-một-lần không bao giờ chạy thật"
        )

    def test_the_traced_layer_forwards_the_pair_to_its_inner(self) -> None:
        from serving.core.instrument import TracedRetriever

        recorded: dict[str, Any] = {}

        class _Inner:
            name = "inner"

            def retrieve(
                self,
                query: str,
                top_k: int = 10,
                *,
                filters: Any = None,
                precomputed: Any = None,
            ) -> list[Any]:
                recorded["precomputed"] = precomputed
                return []

        traced = TracedRetriever(_Inner(), "retrieve")  # type: ignore[arg-type]
        traced.retrieve("q", 5, precomputed=("d", "s"))
        assert recorded["precomputed"] == ("d", "s")

    def test_the_traced_layer_leaves_a_strict_inner_alone_when_there_is_no_pair(self) -> None:
        from serving.core.instrument import TracedRetriever

        class _StrictInner:
            name = "strict"

            def retrieve(self, query: str, top_k: int = 10, *, filters: Any = None) -> list[Any]:
                return []

        traced = TracedRetriever(_StrictInner(), "retrieve")  # type: ignore[arg-type]
        assert traced.retrieve("q", 5) == []
