"""Vòng phản hồi end-to-end — `W5-08`. Postgres thật, RLS thật, app thật.

Ba thứ ở đây không giả lập được, và cả ba là lý do module này tồn tại:

* **RLS.** "Tenant khác không chấm được câu trả lời của tôi" là hành vi của
  policy Postgres, không của mã Python — một mock sẽ xanh với một policy bị
  gỡ bỏ.
* **Upsert.** `ON CONFLICT (tenant_id, message_id)` cần chỉ mục duy nhất **thật**
  của `0004`; thiếu nó thì câu lệnh không lỗi, nó chỉ thôi là upsert.
* **Cầu nối `answer_message_id`.** Id phát ra trong khung `meta` phải là đúng id
  mà task nền ghi xuống — hai chỗ trong hai luồng khác nhau, nối bằng một
  dataclass.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from rag_core.settings import Settings
from serving.api.app import create_app
from serving.core.auth import digest_of
from serving.core.feedback import SCORE_NAME
from serving.core.langfuse import Score, score_id
from tests.integration.chat_app import write_keys
from tests.integration.test_bundle_reload import write_bundle
from tests.integration.test_tracing import FakeLLM, _always_ready, _fake_runtime

pytestmark = pytest.mark.integration

KEY = "rag_acme_feedback_key"
ADMIN_KEY = "rag_acme_feedback_admin"
OTHER_KEY = "rag_globex_feedback_key"


@pytest.fixture(scope="module", autouse=True)
def _selector_loop() -> Iterator[None]:
    """Xem `test_tracing.py::_selector_loop` — cùng lý do, cùng cách chữa."""
    import asyncio
    import sys

    if not sys.platform.startswith("win"):
        yield
        return
    previous = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        yield
    finally:
        asyncio.set_event_loop_policy(previous)


class _CountingSink:
    """Đứng thay `LangfuseSink`: đếm điểm thay vì gửi chúng đi."""

    def __init__(self) -> None:
        self.traces: list[Any] = []
        self.scores: list[Score] = []

    def submit(self, trace: Any) -> None:
        self.traces.append(trace)

    def submit_score(self, score: Score) -> None:
        self.scores.append(score)

    def status(self) -> dict[str, Any]:
        return {"host": "fake", "queued": 0, "sent": len(self.traces), "scored": len(self.scores)}


@pytest.fixture(scope="module")
def feedback_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("feedback")
    write_bundle(root / "bundles", "0.2.0")
    write_keys(
        root / "api-keys.json",
        {
            digest_of(KEY): {
                "tenant_id": "acme",
                "key_id": "acme-fb",
                "scopes": [],
                "rate_limit_per_minute": 10_000,
            },
            digest_of(ADMIN_KEY): {
                "tenant_id": "acme",
                "key_id": "acme-fb-admin",
                "scopes": ["admin"],
                "rate_limit_per_minute": 10_000,
            },
            digest_of(OTHER_KEY): {
                "tenant_id": "globex",
                "key_id": "globex-fb",
                "scopes": ["admin"],
                "rate_limit_per_minute": 10_000,
            },
        },
    )
    return root


@pytest.fixture(autouse=True)
def _empty_tables(database: Any) -> None:
    """Mỗi bài bắt đầu từ ba bảng rỗng.

    ⚠️ `database` dọn ở **đầu và cuối module**, nên không có nó thì mọi phép
    đếm ở đây ("hàng đợi có đúng 1 mục") đo tổng của các bài chạy trước — và nó
    xanh khi chạy một mình, đỏ khi chạy cả module. Kiểu phụ thuộc thứ tự ấy đắt
    nhất lúc gỡ, vì bài đỏ không phải bài sai.
    """
    from sqlalchemy import text

    with database.begin() as conn:
        for table in ("feedback", "message", "conversation"):
            conn.execute(text(f"DELETE FROM {table}"))


@pytest.fixture
def app(feedback_workspace: Path, database: Any) -> Iterator[tuple[TestClient, _CountingSink]]:
    settings = Settings(
        bundle_root=feedback_workspace / "bundles",
        bundle_version="0.2.0",
        api_keys_file=feedback_workspace / "api-keys.json",
        chat_cache=False,
        chat_rewrite=False,
    )
    api = create_app(
        settings=settings,
        build_runtime=_fake_runtime,
        probe_factory=lambda registry: _always_ready(),
    )
    sink = _CountingSink()
    api.state.trace_sink = sink
    with TestClient(api) as client:
        api.state.chat.llm = FakeLLM()
        api.state.chat.sink = sink
        yield client, sink


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------


def _frames(response: httpx.Response) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    name = ""
    for line in response.text.splitlines():
        if line.startswith("event: "):
            name = line[len("event: ") :]
        elif line.startswith("data: "):
            out.append((name, json.loads(line[len("data: ") :])))
    return out


def _turn(client: TestClient, message: str = "RRF là gì?", *, key: str = KEY) -> dict[str, Any]:
    """Một lượt `/chat` trọn vẹn; trả về khung `meta`.

    ⚠️ `_drain_saves` phải chạy sau: hàng trợ lý được ghi trong một task nền,
    nên id trong khung `meta` trỏ vào một hàng chưa tồn tại cho tới lúc ấy. Đó
    không phải một chi tiết của test — nó là `TD-78` nhìn từ phía client.
    """
    response = client.post(
        "/chat", json={"message": message}, headers={"Authorization": f"Bearer {key}"}
    )
    response.read()
    assert response.status_code == 200, response.text
    frames = dict(_frames(response))
    _drain_saves(client)
    return frames["meta"]


def _drain_saves(client: TestClient) -> None:
    """Đợi task ghi Postgres chạy xong.

    Không có `await` nào ở đây bắt được nó: `_schedule_save` cố ý là đồng bộ
    (xem §"Ngắt kết nối" của `serving/core/chat.py`). Một request rẻ khác là
    cách ép vòng lặp sự kiện quay thêm vài vòng.
    """
    from serving.core.chat import _PENDING

    for _ in range(50):
        if not _PENDING:
            return
        client.get("/health")
    raise AssertionError("task ghi message không kết thúc")


def _rate(
    client: TestClient, message_id: str, rating: int, *, key: str = KEY, **extra: Any
) -> httpx.Response:
    response: httpx.Response = client.post(
        "/feedback",
        json={"message_id": message_id, "rating": rating, **extra},
        headers={"Authorization": f"Bearer {key}"},
    )
    return response


# ---------------------------------------------------------------------------
# 1. DoD: một câu 👎 đi hết đường
# ---------------------------------------------------------------------------


class TestTheHappyPath:
    def test_a_thumbs_down_lands_in_postgres_and_in_langfuse(self, app: Any) -> None:
        """Câu DoD, viết ra thành một chuỗi bốn phép kiểm."""
        client, sink = app
        meta = _turn(client)

        response = _rate(
            client, meta["answer_message_id"], -1, reason="not_found", comment="báo cáo có nói mà"
        )
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["rating"] == -1
        assert body["reason"] == "not_found"
        assert body["replaced"] is False

        # (1) trace_id lấy từ HÀNG, và nó khớp trace của đúng lượt ấy
        assert body["trace_id"] == meta["trace_id"]
        # (2) điểm đã xếp hàng sang Langfuse
        assert body["scored"] is True
        (score,) = sink.scores
        assert score.trace_id == meta["trace_id"]
        assert score.value == -1.0
        assert score.name == SCORE_NAME
        assert score.comment == "not_found · báo cáo có nói mà"
        # (3) hàng đợi review nhìn thấy nó
        queue = client.get(
            "/admin/feedback", headers={"Authorization": f"Bearer {ADMIN_KEY}"}
        ).json()
        assert queue["count"] == 1
        (item,) = queue["items"]
        assert item["message_id"] == meta["answer_message_id"]
        assert item["question"] == "RRF là gì?"
        assert item["answer"] == "RRF hợp nhất thứ hạng."
        assert item["model"] == "fake-model-served"
        assert item["bundle_version"] == "0.2.0"
        assert item["retrieved_chunk_ids"]

    def test_the_candidate_file_can_be_written_from_it(self, app: Any, tmp_path: Path) -> None:
        """Nửa sau của DoD: *"câu 👎 xuất ra được file candidate"*."""
        client, _ = app
        meta = _turn(client, "Câu hỏi sẽ bị chấm kém?")
        assert _rate(client, meta["answer_message_id"], -1, reason="wrong").status_code == 201

        response = client.get(
            "/admin/feedback/candidates", headers={"Authorization": f"Bearer {ADMIN_KEY}"}
        )
        assert response.status_code == 200
        assert response.headers["X-Candidate-Count"] == "1"
        (line,) = response.text.splitlines()
        candidate = json.loads(line)
        assert candidate["query"] == "Câu hỏi sẽ bị chấm kém?"
        assert candidate["rating"] == -1
        assert candidate["reviewed_by_human"] is False
        # ⭐⭐ Và nó KHÔNG mang nhãn — xem `GoldenCandidate`.
        assert "relevant_chunk_ids" not in candidate
        assert "reference_answer" not in candidate

        path = tmp_path / "candidates.jsonl"
        path.write_text(response.text, encoding="utf-8")
        assert len(path.read_text(encoding="utf-8").splitlines()) == 1

    def test_a_thumbs_up_is_recorded_but_stays_out_of_the_review_queue(self, app: Any) -> None:
        """Hàng đợi review là danh sách **việc phải làm**; một lượt hài lòng
        không phải việc phải làm."""
        client, _ = app
        meta = _turn(client)
        assert _rate(client, meta["answer_message_id"], 1).status_code == 201

        headers = {"Authorization": f"Bearer {ADMIN_KEY}"}
        assert client.get("/admin/feedback", headers=headers).json()["count"] == 0
        assert client.get("/admin/feedback?rating=0", headers=headers).json()["count"] == 1


# ---------------------------------------------------------------------------
# 2. ⭐⭐ Khoá nối không đến từ người gọi
# ---------------------------------------------------------------------------


class TestTheJoinKeyIsProven:
    def test_a_tenant_cannot_rate_another_tenants_answer(self, app: Any) -> None:
        """RLS lọc trước, nên hàng của `acme` **không tồn tại** với `globex` —
        404, không 403. Hướng ấy cũng đúng về mặt rò rỉ: một 403 xác nhận rằng
        id ấy có thật."""
        client, sink = app
        meta = _turn(client)
        before = len(sink.scores)

        response = _rate(client, meta["answer_message_id"], -1, key=OTHER_KEY)
        assert response.status_code == 404
        assert len(sink.scores) == before, "không được gắn điểm vào trace của tenant khác"

    def test_the_endpoint_refuses_a_body_that_carries_a_trace_id(self, app: Any) -> None:
        """`extra="forbid"`. Nếu một ngày trường này được nhận thì bài này đỏ,
        và người thêm nó phải đọc lý do trước khi xoá bài."""
        client, _ = app
        meta = _turn(client)
        response = client.post(
            "/feedback",
            json={
                "message_id": meta["answer_message_id"],
                "rating": -1,
                "trace_id": "0" * 32,
            },
            headers={"Authorization": f"Bearer {KEY}"},
        )
        assert response.status_code == 422

    def test_rating_the_question_instead_of_the_answer_is_refused(self, app: Any) -> None:
        """Khung `meta` mang **hai** id, và chấm nhầm cái kia là lỗi dễ nhất
        của người tích hợp — nên nó phải là 422 với lời giải thích, không phải
        một hàng hợp lệ vô nghĩa."""
        client, _ = app
        meta = _turn(client)
        response = _rate(client, meta["message_id"], -1)
        assert response.status_code == 422
        assert "role='user'" in response.json()["detail"]

    def test_an_unknown_message_is_404(self, app: Any) -> None:
        client, _ = app
        _turn(client)
        assert _rate(client, "f" * 32, -1).status_code == 404


# ---------------------------------------------------------------------------
# 3. Idempotent ở CẢ HAI kho
# ---------------------------------------------------------------------------


class TestOneRatingPerAnswer:
    def test_a_double_click_does_not_double_the_queue(self, app: Any) -> None:
        client, _ = app
        meta = _turn(client)
        first = _rate(client, meta["answer_message_id"], -1, reason="wrong")
        second = _rate(client, meta["answer_message_id"], -1, reason="wrong")
        assert first.status_code == second.status_code == 201
        assert first.json()["replaced"] is False
        assert second.json()["replaced"] is True
        assert first.json()["id"] == second.json()["id"]

        queue = client.get(
            "/admin/feedback", headers={"Authorization": f"Bearer {ADMIN_KEY}"}
        ).json()
        assert queue["count"] == 1

    def test_changing_your_mind_overwrites_in_both_stores(self, app: Any) -> None:
        """⭐⭐ Một luật idempotent cho hai kho: Postgres theo
        `(tenant, message)`, Langfuse theo `score_id()`."""
        client, sink = app
        meta = _turn(client)
        _rate(client, meta["answer_message_id"], -1, reason="wrong")
        _rate(client, meta["answer_message_id"], 1)

        headers = {"Authorization": f"Bearer {ADMIN_KEY}"}
        assert client.get("/admin/feedback?rating=0", headers=headers).json()["count"] == 1
        (item,) = client.get("/admin/feedback?rating=0", headers=headers).json()["items"]
        assert item["rating"] == 1
        assert item["reason"] is None

        assert [s.value for s in sink.scores] == [-1.0, 1.0]
        ids = {score_id(s.trace_id, s.name) for s in sink.scores}
        assert len(ids) == 1, "hai điểm phải cùng id, nếu không Langfuse giữ cả hai"

    def test_a_re_rating_bubbles_back_to_the_top_of_the_queue(self, app: Any) -> None:
        """⭐ Đổi ý là một tín hiệu **mới**, và hàng đợi sắp theo thời gian.

        Giữ nguyên `created_at` cũ thì lần chấm lại chìm xuống dưới những lượt
        đã xảy ra sau nó, và người review không bao giờ thấy nó nổi lên — hàng
        đợi vẫn đúng số lượng, chỉ sai thứ tự. Phép tiêm `F7` sống sót vì không
        bài nào nhìn tới thứ tự.
        """
        client, _ = app
        first = _turn(client, "câu hỏi thứ nhất?")
        second = _turn(client, "câu hỏi thứ hai?")
        _rate(client, first["answer_message_id"], -1, reason="wrong")
        _rate(client, second["answer_message_id"], -1, reason="slow")

        headers = {"Authorization": f"Bearer {ADMIN_KEY}"}
        order = [
            i["message_id"] for i in client.get("/admin/feedback", headers=headers).json()["items"]
        ]
        assert order[0] == second["answer_message_id"]

        _rate(client, first["answer_message_id"], -1, reason="citation")
        order = [
            i["message_id"] for i in client.get("/admin/feedback", headers=headers).json()["items"]
        ]
        assert order[0] == first["answer_message_id"], "lần chấm lại phải nổi lên đầu"


# ---------------------------------------------------------------------------
# 4. Cầu nối `answer_message_id` và hai cột citations của `0004`
# ---------------------------------------------------------------------------


class TestTheAnswerRow:
    def test_the_id_in_the_meta_frame_is_the_row_that_gets_written(self, app: Any) -> None:
        """Sinh ở `prepare()`, phát ở khung đầu, ghi ở một task nền sau khung
        cuối. Ba chỗ, một giá trị."""
        client, _ = app
        meta = _turn(client)
        history = client.get(
            f"/conversations/{meta['conversation_id']}",
            headers={"Authorization": f"Bearer {KEY}"},
        ).json()
        assistant = [m for m in history["messages"] if m["role"] == "assistant"]
        assert [m["id"] for m in assistant] == [meta["answer_message_id"]]

    def test_history_carries_the_trace_id_and_both_citation_columns(self, app: Any) -> None:
        """`TD-50` đóng ở đây: đọc lại một câu trả lời cũ mang theo **kết quả
        xác minh** của nó, không chỉ danh sách nguồn đã đưa vào."""
        client, _ = app
        meta = _turn(client)
        history = client.get(
            f"/conversations/{meta['conversation_id']}",
            headers={"Authorization": f"Bearer {KEY}"},
        ).json()
        (answer,) = [m for m in history["messages"] if m["role"] == "assistant"]
        assert answer["trace_id"] == meta["trace_id"]
        assert answer["sources"], "nguồn đã đưa cho model"
        assert answer["citations"] is not None, "khung xác minh của W4-09"
        assert "block" in answer["citations"]

    def test_the_two_columns_are_not_the_same_thing(self, app: Any) -> None:
        """⚠️ Bài này là chỗ lỗi của `0001` chết: một cột tên `citations` chứa
        `sources()` trông đúng cho tới khi ai đó hỏi *"citation nào đã được xác
        minh"*."""
        client, _ = app
        meta = _turn(client)
        history = client.get(
            f"/conversations/{meta['conversation_id']}",
            headers={"Authorization": f"Bearer {KEY}"},
        ).json()
        (answer,) = [m for m in history["messages"] if m["role"] == "assistant"]
        assert isinstance(answer["sources"], list)
        assert isinstance(answer["citations"], dict)


# ---------------------------------------------------------------------------
# 5. Hàng rào và số đo
# ---------------------------------------------------------------------------


class TestTheEdges:
    def test_the_review_queue_needs_the_admin_scope(self, app: Any) -> None:
        client, _ = app
        response = client.get("/admin/feedback", headers={"Authorization": f"Bearer {KEY}"})
        assert response.status_code == 403

    def test_posting_feedback_does_not(self, app: Any) -> None:
        """Người dùng thường phải chấm được — nếu không thì không có tín hiệu nào."""
        client, _ = app
        meta = _turn(client)
        assert _rate(client, meta["answer_message_id"], -1).status_code == 201

    def test_feedback_needs_a_key_at_all(self, app: Any) -> None:
        client, _ = app
        meta = _turn(client)
        response = client.post(
            "/feedback", json={"message_id": meta["answer_message_id"], "rating": -1}
        )
        assert response.status_code == 401

    def test_the_metric_counts_it_by_rating_and_reason(self, app: Any) -> None:
        client, _ = app
        meta = _turn(client)
        _rate(client, meta["answer_message_id"], -1, reason="citation")

        body = client.get("/metrics", headers={"Authorization": f"Bearer {KEY}"}).text
        line = next(
            ln
            for ln in body.splitlines()
            if ln.startswith('rag_feedback_total{rating="-1",reason="citation"}')
        )
        assert line.endswith(" 1.0")

    def test_an_unknown_reason_is_422_not_500(self, app: Any) -> None:
        """Ba bản sao của danh sách lý do phải trùng nhau; nếu chúng lệch thì
        Postgres từ chối `INSERT` và người dùng nhận 500."""
        client, _ = app
        meta = _turn(client)
        response = _rate(client, meta["answer_message_id"], -1, reason="hallucination")
        assert response.status_code == 422

    def test_a_comment_longer_than_the_cap_is_refused(self, app: Any) -> None:
        client, _ = app
        meta = _turn(client)
        assert _rate(client, meta["answer_message_id"], -1, comment="x" * 2001).status_code == 422

    @pytest.mark.asyncio
    async def test_the_core_layer_guards_the_vocabulary_too_not_only_the_api(
        self, app: Any
    ) -> None:
        """⭐ `Literal` của FastAPI chặn đường HTTP, nhưng `record_feedback` còn
        có **một** người gọi khác: CLI xuất ứng viên và bất kỳ script vận hành
        nào. Bỏ phép kiểm ở lõi thì một mã lạ đi thẳng tới `CheckConstraint`
        của Postgres và quay ra thành `IntegrityError` — tức 500 thay vì một
        lời từ chối đọc được.

        Phép tiêm `F19` sống sót vì mọi bài khác đi qua HTTP, nơi `Literal` đã
        chặn trước.
        """
        from serving.core.auth import Principal
        from serving.core.feedback import record_feedback

        client, _ = app
        meta = _turn(client)
        sessions = client.app.state.chat.sessions
        principal = Principal(tenant_id="acme", key_id="direct")

        with pytest.raises(ValueError, match="reason không hợp lệ"):
            await record_feedback(
                sessions,
                principal,
                message_id=meta["answer_message_id"],
                rating=-1,
                reason="hallucination",
            )
        with pytest.raises(ValueError, match="rating phải là"):
            await record_feedback(
                sessions, principal, message_id=meta["answer_message_id"], rating=0
            )
