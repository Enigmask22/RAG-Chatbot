"""Vòng phản hồi, phần không cần Postgres — `W5-08`.

Bài quan trọng nhất ở đây là `TestACandidateIsNotAGoldenQuery`: nó khoá một
tính chất **âm** (ứng viên phải KHÔNG mang nhãn), và một tính chất âm là thứ
biến mất dễ nhất khi ai đó "làm cho tiện dùng hơn".
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from pipeline.eval.golden import GoldenQuery
from serving.api.feedback import FeedbackRequest, Reason
from serving.core.feedback import (
    SCORE_NAME,
    GoldenCandidate,
    ReviewItem,
    _score_comment,
    _to_item,
    to_candidate,
    write_candidates,
)
from serving.core.langfuse import Score, encode_score, score_id
from serving.db.models import FEEDBACK_REASONS


def _item(**overrides: Any) -> ReviewItem:
    base: dict[str, Any] = {
        "feedback_id": "fb0123456789abcdef",
        "created_at": datetime(2026, 9, 5, 12, 0, tzinfo=UTC),
        "rating": -1,
        "reason": "not_found",
        "comment": "báo cáo có nói mà",
        "conversation_id": "conv1",
        "message_id": "msg1",
        "question": "GDP Việt Nam 2016 tăng bao nhiêu?",
        "rewritten_query": None,
        "answer": "Tôi không tìm thấy thông tin này trong tài liệu.",
        "model": "deepseek-v4-flash",
        "finish_reason": "stop",
        "route": "retrieve",
        "latency_ms": 2500,
        "trace_id": "t" * 32,
        "bundle_version": "0.2.1",
        "retrieved_chunk_ids": ["wb-1::00094", "wb-1::00095"],
        "cited_chunk_ids": [],
        "citations_verified": 0,
        "citations_claimed": 0,
    }
    base.update(overrides)
    return ReviewItem(**base)


# ---------------------------------------------------------------------------
# 1. ⭐⭐ Ứng viên KHÔNG phải một câu golden
# ---------------------------------------------------------------------------


class TestACandidateIsNotAGoldenQuery:
    def test_it_carries_no_field_that_looks_like_a_label(self) -> None:
        """Ba tên bị cấm, và lý do nằm ở docstring `GoldenCandidate`.

        Điền `relevant_chunk_ids` bằng thứ hệ thống đã truy hồi nghĩa là chấm
        hệ thống bằng chính lỗi của nó — trên đúng những hàng tồn tại *vì* nó
        đã sai.
        """
        fields = set(GoldenCandidate.model_fields)
        assert not [f for f in fields if f.startswith("relevant_")]
        assert "reference_answer" not in fields
        assert "category" not in fields

    def test_the_golden_schema_refuses_it(self) -> None:
        """Phép kiểm mạnh hơn một danh sách tên: đưa thẳng ứng viên cho
        `GoldenQuery` và đòi nó **đỏ**.

        `GoldenQuery` khai `extra="forbid"`, nên nếu ai đó thêm một trường nhãn
        vào ứng viên thì bài này vẫn đỏ ở phía thiếu, còn nếu họ làm ứng viên
        khớp hoàn toàn thì nó xanh — và đó chính là lúc phải dừng lại.
        """
        candidate = to_candidate(_item())
        with pytest.raises(Exception):  # noqa: B017 - pydantic.ValidationError
            GoldenQuery.model_validate(candidate.model_dump())

    def test_the_three_missing_fields_are_exactly_the_human_work(self) -> None:
        needed = set(GoldenQuery.model_fields) - set(GoldenCandidate.model_fields)
        assert {"category", "relevant_spans", "reference_answer"} <= needed

    def test_what_the_system_did_is_named_as_behaviour_not_as_truth(self) -> None:
        candidate = to_candidate(_item())
        assert candidate.retrieved_chunk_ids == ["wb-1::00094", "wb-1::00095"]
        assert candidate.system_answer.startswith("Tôi không tìm thấy")
        assert candidate.reviewed_by_human is False

    def test_the_query_is_what_the_user_typed(self) -> None:
        """Không phải chuỗi đã viết lại: một golden set chứa câu hỏi do máy
        viết lại đo hệ thống trên phân bố truy vấn của chính nó."""
        candidate = to_candidate(_item(rewritten_query="GDP Việt Nam năm 2016"))
        assert candidate.query == "GDP Việt Nam 2016 tăng bao nhiêu?"
        assert candidate.rewritten_query == "GDP Việt Nam năm 2016"


# ---------------------------------------------------------------------------
# 2. Đọc một lượt bị chấm ra thành ReviewItem
# ---------------------------------------------------------------------------


class _Row:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


def _message(**kw: Any) -> Any:
    base: dict[str, Any] = {
        "id": "msg1",
        "conversation_id": "conv1",
        "role": "assistant",
        "content": "Câu trả lời.",
        "retrieved_sources": [{"n": 1, "chunk_id": "c1"}, {"n": 2, "chunk_id": "c2"}],
        "citations_verified": {
            "block": "ok",
            "verified": 1,
            "citations": [
                {"n": 1, "chunk_id": "c1", "verified": True},
                {"n": 2, "chunk_id": "c2", "verified": False},
            ],
        },
        "trace_id": "t" * 32,
        "model": "m",
        "finish_reason": "stop",
        "latency_ms": 10,
        "route": None,
        "created_at": datetime(2026, 9, 5, tzinfo=UTC),
    }
    base.update(kw)
    return _Row(**base)


def _feedback(**kw: Any) -> Any:
    base: dict[str, Any] = {
        "id": "fb1",
        "created_at": datetime(2026, 9, 5, tzinfo=UTC),
        "rating": -1,
        "reason": "citation",
        "comment": None,
    }
    base.update(kw)
    return _Row(**base)


class TestReadingARatedTurn:
    def test_it_separates_what_was_given_from_what_was_claimed(self) -> None:
        """⭐ Hai danh sách, không một. Đó là toàn bộ lý do `0004` tách cột."""
        item = _to_item(_feedback(), _message(), "0.2.1", None)
        assert item.retrieved_chunk_ids == ["c1", "c2"]
        assert item.cited_chunk_ids == ["c1", "c2"]
        assert item.citations_claimed == 2
        assert item.citations_verified == 1

    def test_a_row_written_before_0004_reads_as_unknown_not_as_zero(self) -> None:
        """`NULL` ≠ "không citation nào". Gộp hai thứ ấy làm một báo cáo
        "tỉ lệ citation xác minh được" tụt theo tuổi của dữ liệu."""
        item = _to_item(_feedback(), _message(citations_verified=None), None, None)
        assert item.citations_verified is None
        assert item.citations_claimed is None
        assert item.cited_chunk_ids == []

    def test_the_question_comes_from_the_paired_user_message(self) -> None:
        question = _message(
            id="msg0",
            role="user",
            content="Hỏi gì đó?",
            rewritten_query="Hỏi rõ hơn?",
            route="retrieve",
        )
        item = _to_item(_feedback(), _message(), "0.2.1", question)
        assert item.question == "Hỏi gì đó?"
        assert item.rewritten_query == "Hỏi rõ hơn?"
        assert item.route == "retrieve"

    def test_a_source_without_a_chunk_id_is_skipped_not_stringified(self) -> None:
        item = _to_item(
            _feedback(),
            _message(retrieved_sources=[{"n": 1}, {"n": 2, "chunk_id": "c9"}]),
            None,
            None,
        )
        assert item.retrieved_chunk_ids == ["c9"]


# ---------------------------------------------------------------------------
# 3. Điểm số Langfuse
# ---------------------------------------------------------------------------


class TestScore:
    def test_the_id_is_deterministic_so_changing_your_mind_overwrites(self) -> None:
        """⭐⭐ Một luật idempotent cho hai kho — xem `score_id`."""
        first = encode_score(Score(trace_id="abc", name=SCORE_NAME, value=-1.0))
        second = encode_score(Score(trace_id="abc", name=SCORE_NAME, value=1.0))
        assert first[0]["body"]["id"] == second[0]["body"]["id"]
        assert first[0]["body"]["value"] == -1.0
        assert second[0]["body"]["value"] == 1.0

    def test_different_traces_get_different_ids(self) -> None:
        assert score_id("a", SCORE_NAME) != score_id("b", SCORE_NAME)

    def test_the_envelope_is_a_score_create_pointing_at_the_trace(self) -> None:
        (event,) = encode_score(Score(trace_id="t1", name=SCORE_NAME, value=1.0, comment="hay"))
        assert event["type"] == "score-create"
        assert event["body"]["traceId"] == "t1"
        assert event["body"]["comment"] == "hay"
        assert event["body"]["dataType"] == "NUMERIC"

    def test_the_comment_joins_reason_and_free_text(self) -> None:
        assert _score_comment("wrong", "sai số liệu") == "wrong · sai số liệu"

    def test_pii_in_a_comment_is_redacted_at_the_langfuse_boundary(self) -> None:
        """`NEW-08`/`AU-05`: lớp thứ hai — comment đã redact ở nguồn
        (`record_feedback`), nhưng biên xuất không được *phụ thuộc* điều đó:
        một `Score` dựng từ đường khác vẫn phải sạch khi rời hệ thống."""
        (event,) = encode_score(
            Score(
                trace_id="t1",
                name=SCORE_NAME,
                value=-1.0,
                comment="gọi tôi qua 0912345678 hoặc toi@example.com",
            )
        )
        assert "0912345678" not in event["body"]["comment"]
        assert "toi@example.com" not in event["body"]["comment"]
        assert _score_comment("wrong", None) == "wrong"
        assert _score_comment(None, None) is None


class TestTheSinkCarriesBothKinds:
    def test_a_score_and_a_trace_share_one_queue(self) -> None:
        """⭐ Cùng hàng đợi, cùng trần, cùng bộ đếm `dropped` — một đường thứ
        hai sang Langfuse là một chỗ thứ hai để chặn `POST /feedback`."""
        from serving.core.langfuse import LangfuseSink

        sent: list[dict[str, Any]] = []

        class _Client:
            def post(self, path: str, *, content: str) -> Any:
                sent.append(json.loads(content))
                return _Row(status_code=207, text="{}")

            def close(self) -> None:
                pass

        sink = LangfuseSink(
            host="http://localhost",
            public_key="p",
            secret_key="s",
            client=_Client(),  # type: ignore[arg-type]
        )
        sink.submit_score(Score(trace_id="t1", name=SCORE_NAME, value=-1.0))
        sink.close()

        assert [e["type"] for batch in sent for e in batch["batch"]] == ["score-create"]
        assert sink.status()["scored"] == 1
        assert sink.status()["sent"] == 0


# ---------------------------------------------------------------------------
# 4. Danh sách lý do: ba bản sao phải trùng nhau
# ---------------------------------------------------------------------------


class TestTheReasonVocabulary:
    def test_the_api_literal_matches_the_database_check(self) -> None:
        """Lệch nhau nghĩa là API nhận một giá trị mà Postgres từ chối — 500,
        không phải 422. Kiểm này cũng chạy lúc import `serving.api.feedback`."""
        from typing import get_args

        assert set(get_args(Reason)) == set(FEEDBACK_REASONS)

    def test_every_reason_points_at_one_component(self) -> None:
        """Tiêu chí trong docstring `FEEDBACK_REASONS`, viết thành một phép đếm:
        bảy mã, và `other` là mã cuối (nó là cửa thoát, không phải một nhãn)."""
        assert len(FEEDBACK_REASONS) == 7
        assert FEEDBACK_REASONS[-1] == "other"

    def test_an_unknown_reason_is_rejected_by_the_request_model(self) -> None:
        with pytest.raises(Exception):  # noqa: B017 - pydantic.ValidationError
            FeedbackRequest(message_id="m", rating=-1, reason="hallucination")  # type: ignore[arg-type]


class TestTheRequestModel:
    def test_it_has_no_trace_id_field(self) -> None:
        """⭐⭐ Khoá nối không đến từ người gọi — xem docstring module
        `serving.core.feedback`."""
        assert "trace_id" not in FeedbackRequest.model_fields

    def test_a_client_that_sends_one_anyway_is_rejected_loudly(self) -> None:
        """`extra="forbid"`: 422, không phải im lặng bỏ qua. Im lặng nghĩa là
        người tích hợp tin rằng trường ấy có tác dụng."""
        with pytest.raises(Exception):  # noqa: B017 - pydantic.ValidationError
            FeedbackRequest(message_id="m", rating=-1, trace_id="deadbeef")  # type: ignore[call-arg]

    def test_only_thumbs_up_or_down(self) -> None:
        for bad in (0, 5, -2):
            with pytest.raises(Exception):  # noqa: B017 - pydantic.ValidationError
                FeedbackRequest(message_id="m", rating=bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 5. File ứng viên
# ---------------------------------------------------------------------------


class TestTheCandidateFile:
    def test_it_is_jsonl_one_candidate_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "candidates.jsonl"
        written = write_candidates([to_candidate(_item()), to_candidate(_item())], path)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert written == 2
        assert len(lines) == 2
        assert json.loads(lines[0])["rating"] == -1

    def test_vietnamese_survives(self, tmp_path: Path) -> None:
        path = tmp_path / "c.jsonl"
        write_candidates([to_candidate(_item())], path)
        assert "không tìm thấy" in path.read_text(encoding="utf-8")

    def test_the_candidate_id_traces_back_to_the_feedback_row(self) -> None:
        candidate = to_candidate(_item(feedback_id="abcdef012345ffff"))
        assert candidate.candidate_id == "fb-abcdef012345"
