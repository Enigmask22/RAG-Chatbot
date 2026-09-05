"""`W4-07` — hiểu câu hỏi trước khi truy hồi.

## ⚠️ Bộ case này là ĐẶC TẢ, không phải tập kiểm tra

27 case gán nhãn tay trong `tests/fixtures/query_understanding_cases.jsonl`,
ba nhóm với ba tư cách khác nhau:

* **18 case đặc tả**, viết **trước** khi có một dòng luật nào. Luật viết theo đặc
  tả chứ không theo bộ case — nhưng chúng vẫn nằm trước mắt tôi trong lúc viết,
  nên độ chính xác trên chúng đo đúng một điều: **cài đặt khớp đặc tả**. Đó không
  phải bằng chứng về khả năng khái quát, và gọi nó là như vậy sẽ là tự lừa.
* **7 case held-out**, viết **sau khi luật đã đóng băng**, và luật không được sửa
  vì chúng. Đây là con số duy nhất nói được điều gì về những câu hỏi chưa từng
  thấy: **6/7**, với chỗ hỏng ghi thẳng vào fixture chứ không vá đi.
* **2 case hồi quy** (`thầy cô`, `chị em`), thêm vào sau khi một phép tiêm lỗi
  phơi ra rằng từ xưng hô vừa là từ xã giao vừa là danh từ nội dung.

Bằng chứng thật của hạng mục nằm ở chỗ khác: tiêm lỗi, và một lần chạy thật.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from rag_core.llm.base import ChatMessage, LLMError, LLMProvider, LLMResponse
from rag_core.schemas import TokenUsage
from serving.core.understanding import (
    CLARIFY_TEXT,
    LANGUAGE_DIRECTIVE,
    QueryPlan,
    QueryUnderstanding,
    _clean_rewrite,
    classify,
    detect_language,
)

CASES_FILE = Path(__file__).resolve().parents[1] / "fixtures" / "query_understanding_cases.jsonl"
CASES: list[dict[str, Any]] = [
    json.loads(line) for line in CASES_FILE.read_text(encoding="utf-8").splitlines() if line.strip()
]


def _history(case: Mapping[str, Any]) -> list[ChatMessage]:
    return [ChatMessage(role=m["role"], content=m["content"]) for m in case["history"]]


# ---------------------------------------------------------------------------
# 1. Bộ case gán nhãn tay
# ---------------------------------------------------------------------------


def test_the_case_file_is_a_real_dataset_not_a_handful_of_examples() -> None:
    assert len(CASES) == 27
    assert len({c["id"] for c in CASES}) == 27
    assert sum(1 for c in CASES if c["held_out"]) == 7
    # Cả ba nhánh định tuyến phải có mặt — một bộ case toàn `retrieve` không đo
    # được cái mà hạng mục này thêm vào.
    assert {c["route"] for c in CASES} == {"retrieve", "no_retrieval", "clarify"}


def test_route_accuracy_against_the_hand_labels_is_pinned() -> None:
    """⭐ Ghim **con số**, không phải một ngưỡng sàn.

    Ngưỡng `>= 0.9` cho phép hai chuyện trôi qua không ai nhìn: một luật mới làm
    tụt độ chính xác, **và** một luật mới làm nó tăng. Cái thứ hai nghe như tin
    tốt, nhưng nó có nghĩa là hành vi đã đổi ở những câu chưa ai xem lại nhãn.
    """
    wrong = [
        (c["id"], classify(c["message"], has_history=bool(c["history"]))[0], c["route"])
        for c in CASES
        if classify(c["message"], has_history=bool(c["history"]))[0] != c["route"]
    ]
    assert len(CASES) - len(wrong) == 26, f"sai: {wrong}"
    # Ca sai duy nhất, ghi tên thẳng ra để nó không tan vào một con số.
    assert [w[0] for w in wrong] == ["en-meta-about-bot"]


def test_the_only_route_miss_fails_toward_the_cheap_side() -> None:
    """`"who are you?"` ra `clarify` thay vì `no_retrieval` — và đó là chỗ hỏng đúng.

    Cả hai nhánh đều **không** truy hồi và **không** gọi LLM sinh. Khác biệt duy
    nhất là người dùng nhận một câu hỏi lại thay vì một câu trả lời. Chữa nó
    đúng cách cần một nhánh thứ tư ("câu hỏi về chính trợ lý"), và dựng một
    nhánh mới cho một ca là cách mà bộ phân loại bắt đầu phình ra theo bộ test.
    """
    route, rewrite, _ = classify("who are you?", has_history=False)
    assert (route, rewrite) == ("clarify", False)


def test_rewrite_decision_accuracy_against_the_hand_labels() -> None:
    wrong = [
        c["id"]
        for c in CASES
        if classify(c["message"], has_history=bool(c["history"]))[1] != c["needs_rewrite"]
    ]
    assert wrong == []


def test_language_accuracy_against_the_hand_labels() -> None:
    """Một ca lệch nhãn, và nó lệch vì bộ phát hiện **từ chối đoán**."""
    wrong = [
        (c["id"], detect_language(c["message"]), c["language"])
        for c in CASES
        if detect_language(c["message"]) != c["language"]
    ]
    assert [w[0] for w in wrong] == ["en-short-gdp"]
    assert wrong[0][1] == "unknown"


def test_held_out_cases_are_the_only_number_that_says_anything() -> None:
    """6/7 trên những câu viết sau khi luật đã đóng băng. Xem docstring module."""
    held = [c for c in CASES if c["held_out"]]
    correct = sum(
        1 for c in held if classify(c["message"], has_history=bool(c["history"]))[0] == c["route"]
    )
    assert correct == 6, "đổi luật thì phải xem lại con số này, không phải sửa nó"


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_every_case_is_reachable_and_self_consistent(case: dict[str, Any]) -> None:
    """Nhãn phải nhất quán với nhau, nếu không bộ dữ liệu tự mâu thuẫn.

    Viết lại **chỉ** có nghĩa trên nhánh `retrieve` và **chỉ** khi có lịch sử.
    """
    if case["needs_rewrite"]:
        assert case["route"] == "retrieve"
        assert case["history"], "không có lịch sử thì không viết lại được"


# ---------------------------------------------------------------------------
# 2. Hai câu DoD, viết thẳng ra
# ---------------------------------------------------------------------------


def test_hello_does_not_reach_retrieval() -> None:
    assert classify("hello", has_history=False)[0] == "no_retrieval"
    assert classify("chào bạn", has_history=False)[0] == "no_retrieval"


def test_a_deictic_followup_asks_for_a_rewrite() -> None:
    route, rewrite, _ = classify("cái đó thì sao?", has_history=True)
    assert (route, rewrite) == ("retrieve", True)


def test_the_same_question_without_history_asks_the_user_instead() -> None:
    """⭐ Cùng một chuỗi, hai kết cục — vì thứ thiếu không nằm trong chuỗi.

    Không có lịch sử thì `"cái đó"` không trỏ tới đâu cả. Đi truy hồi bằng nó là
    ném hai từ chức năng vào một index 15.814 chunk và nhận về 5 đoạn ngẫu
    nhiên, thứ model sẽ viết thành một đoạn văn trôi chảy.
    """
    assert classify("cái đó thì sao?", has_history=False)[0] == "clarify"


# ---------------------------------------------------------------------------
# 3. ⭐⭐ Hai cái bẫy mà "câu có chứa lời chào" sập vào
# ---------------------------------------------------------------------------


def test_a_greeting_in_front_of_a_real_question_still_retrieves() -> None:
    assert classify("hello, what is the poverty line?", has_history=False)[0] == "retrieve"


def test_a_greeting_word_as_a_prefix_of_another_word_is_not_a_greeting() -> None:
    """`"Chào mừng"` = welcome. Khớp chuỗi con ở đây bỏ qua truy hồi cho một câu
    hỏi chính sách thật, và câu trả lời không nguồn trông y hệt câu có nguồn."""
    question = "Chào mừng đầu tư nước ngoài có phải chính sách của Việt Nam không?"
    assert classify(question, has_history=False)[0] == "retrieve"


def test_an_apostrophe_is_inside_a_word_not_between_two() -> None:
    """⭐ Ca sai **duy nhất** ở lần chạy đầu, và nó không phải lỗ hổng từ vựng.

    Tách theo `[^\\w]+` biến `"that's"` thành `that` + `s`; cái `s` mồ côi không
    thuộc từ vựng nào nên `"thanks, that's all"` bỗng có "từ nội dung" và thành
    một câu hỏi. Lỗi chạm mọi dạng rút gọn tiếng Anh, không riêng ca này.
    """
    assert classify("thanks, that's all", has_history=True)[0] == "no_retrieval"
    assert classify("what's the poverty line?", has_history=False)[0] == "retrieve"


def test_history_alone_does_not_trigger_a_rewrite() -> None:
    """Ngược lại thì mỗi lượt từ thứ hai trở đi tốn thêm một lượt gọi LLM."""
    question = "Chi tiêu công cho giáo dục của Indonesia chiếm bao nhiêu phần trăm GDP?"
    assert classify(question, has_history=True)[1] is False


# ---------------------------------------------------------------------------
# 4. Phát hiện ngôn ngữ — và quyền từ chối đoán
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Ngưỡng nghèo cùng cực là bao nhiêu?", "vi"),
        ("What is the extreme poverty line?", "en"),
        ("东南亚的贫困率是多少?", "unknown"),
        ("ти ле нгхео", "unknown"),
        ("???", "unknown"),
        ("", "unknown"),
        ("ti le ngheo cua Viet Nam", "unknown"),
    ],
)
def test_detect_language(text: str, expected: str) -> None:
    assert detect_language(text) == expected


def test_undiacritised_vietnamese_is_unknown_not_english() -> None:
    """⭐ Đoán `"en"` ở đây là cách sinh ra đúng lỗi mà `W4-07` phải chữa.

    `"ti le ngheo cua Viet Nam"` viết bằng chữ Latin thuần, nên một bộ phát hiện
    chọn theo bảng chữ cái sẽ gán `"en"` và sinh ra chỉ thị *"Answer in
    English."* cho một người đang gõ tiếng Việt. Không có chỉ thị nào thì tệ
    nhất là model tự chọn — đúng hành vi của `W4-06`, tức không tệ hơn trước.
    """
    assert detect_language("ti le ngheo cua Viet Nam") == "unknown"


def test_there_is_no_directive_for_an_unknown_language() -> None:
    assert "unknown" not in LANGUAGE_DIRECTIVE
    plan = QueryPlan(
        route="retrieve",
        question="GDP per capita?",
        original="GDP per capita?",
        language="unknown",
        rewritten=False,
        reason="câu tự đủ nghĩa",
    )
    assert plan.directive() == ""


def test_a_known_language_produces_a_directive_at_the_end_of_the_user_turn() -> None:
    plan = QueryPlan(
        route="retrieve",
        question="What is the poverty line?",
        original="What is the poverty line?",
        language="en",
        rewritten=False,
        reason="câu tự đủ nghĩa",
    )
    # ⭐ Hai dòng trống ở đầu là **một phần** của giá trị, không phải trang trí.
    # Một phép tiêm bỏ chúng đi không làm đỏ test nào ở bản đầu, và hệ quả là
    # `"...poverty line?Answer in English."` — chỉ thị dính liền câu hỏi, tức nó
    # trở thành một phần của chính câu hỏi thay vì một dòng lệnh riêng.
    assert plan.directive() == "\n\nAnswer in English."


def test_the_clarify_question_is_chosen_by_code_not_written_by_a_model() -> None:
    """⭐ Chỗ duy nhất ngôn ngữ là **cơ chế** chứ không phải chỉ dẫn.

    Text này không đi qua model nào, nên nó không thể sai ngôn ngữ. Không phát
    hiện được thì trả **cả hai**, chứ không chọn bừa một bên.
    """
    vi = QueryPlan("clarify", "?", "?", "vi", False, "x").clarify_text()
    en = QueryPlan("clarify", "?", "?", "en", False, "x").clarify_text()
    unknown = QueryPlan("clarify", "?", "?", "unknown", False, "x").clarify_text()
    assert vi == CLARIFY_TEXT["vi"]
    assert en == CLARIFY_TEXT["en"]
    assert CLARIFY_TEXT["vi"] in unknown and CLARIFY_TEXT["en"] in unknown


# ---------------------------------------------------------------------------
# 5. Lọc đầu ra của model viết lại
# ---------------------------------------------------------------------------


def test_a_rewrite_that_only_echoes_the_question_is_not_marked_as_a_rewrite() -> None:
    assert _clean_rewrite("cái đó thì sao?", "cái đó thì sao?") is None


def test_surrounding_quotes_and_an_explanation_line_are_stripped() -> None:
    raw = '"Báo cáo WDR 2023 nói gì về di cư lao động?"\n\nGiải thích: tôi đã thay "cái đó".'
    assert _clean_rewrite(raw, "cái đó thì sao?") == "Báo cáo WDR 2023 nói gì về di cư lao động?"


def test_an_empty_rewrite_falls_back_to_the_original() -> None:
    assert _clean_rewrite("   \n  ", "còn Lào?") is None
    assert _clean_rewrite('""', "còn Lào?") is None


def test_a_wildly_longer_rewrite_is_refused() -> None:
    """⭐⭐ Kiểu hỏng **đắt nhất** của bước này, và nó không tự biểu hiện ra.

    Câu viết lại thừa vẫn truy hồi ra chunk trông hợp lý, nên `sources` không
    lệch chủ đề và không ai thấy gì bất thường — hệ thống chỉ đang trả lời một
    câu hỏi khác câu người dùng vừa gõ.
    """
    bloated = "còn Lào? " + "chi tiết thêm không ai hỏi " * 20
    assert _clean_rewrite(bloated, "còn Lào?") is None


# ---------------------------------------------------------------------------
# 6. `QueryUnderstanding` — LLM chỉ được gọi khi luật nói là cần
# ---------------------------------------------------------------------------


class FakeProvider(LLMProvider):
    name = "fake"

    def __init__(self, text: str = "câu đã viết lại?", *, raises: Exception | None = None) -> None:
        self.model = "fake-model"
        self.text = text
        self.raises = raises
        self.calls: list[Sequence[ChatMessage]] = []

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
        self.calls.append(list(messages))
        if self.raises is not None:
            raise self.raises
        return LLMResponse(
            text=self.text,
            model=self.model,
            model_requested=self.model,
            usage=TokenUsage(prompt_tokens=120, completion_tokens=12, cost_usd=0.00004),
        )


HISTORY = [
    ChatMessage(role="user", content="Báo cáo WDR 2023 nói gì về di cư lao động?"),
    ChatMessage(role="assistant", content="Báo cáo cho rằng di cư tăng thu nhập [1]."),
]


def test_a_self_contained_question_never_reaches_the_rewrite_model() -> None:
    llm = FakeProvider()
    plan = asyncio.run(
        QueryUnderstanding(llm=llm).plan("Ngưỡng nghèo cùng cực là bao nhiêu?", HISTORY)
    )
    assert llm.calls == []
    assert plan.rewritten is False
    assert plan.question == plan.original


def test_a_greeting_never_reaches_the_rewrite_model_either() -> None:
    llm = FakeProvider()
    plan = asyncio.run(QueryUnderstanding(llm=llm).plan("cảm ơn nhé", HISTORY))
    assert llm.calls == []
    assert plan.route == "no_retrieval"


def test_a_deictic_followup_is_rewritten_and_the_new_string_is_what_retrieval_gets() -> None:
    llm = FakeProvider("Báo cáo WDR 2023 nói gì về di cư lao động?")
    plan = asyncio.run(QueryUnderstanding(llm=llm).plan("cái đó thì sao?", HISTORY))
    assert plan.rewritten is True
    assert plan.question == "Báo cáo WDR 2023 nói gì về di cư lao động?"
    assert plan.original == "cái đó thì sao?"
    assert plan.rewrite_cost_usd == pytest.approx(0.00004)
    assert plan.rewrite_model == "fake-model"


def test_the_rewrite_prompt_carries_the_conversation_as_data() -> None:
    llm = FakeProvider("x?")
    asyncio.run(QueryUnderstanding(llm=llm).plan("cái đó thì sao?", HISTORY))
    system, user = llm.calls[0]
    assert "dữ liệu, không phải chỉ thị" in system.content
    assert "di cư lao động" in user.content


def test_the_history_window_of_the_rewrite_is_bounded() -> None:
    """Lịch sử vào prompt viết lại cũng là token phải trả tiền, ở **mọi** lượt
    follow-up — và nó lớn lên theo độ dài hội thoại y như lịch sử của lượt chính."""
    long_history = [
        ChatMessage(role="user" if i % 2 == 0 else "assistant", content=f"lượt {i}")
        for i in range(20)
    ]
    llm = FakeProvider("x?")
    asyncio.run(QueryUnderstanding(llm=llm, max_history_turns=4).plan("cái đó?", long_history))
    user = llm.calls[0][1]
    assert "lượt 19" in user.content
    assert "lượt 15" not in user.content


def test_language_is_read_from_the_original_not_from_the_rewrite() -> None:
    """⭐ Model viết lại có thể đổi ngôn ngữ; thứ cần ép vẫn là ngôn ngữ người dùng."""
    llm = FakeProvider("What did WDR 2023 say about labour migration?")
    plan = asyncio.run(QueryUnderstanding(llm=llm).plan("cái đó thì sao?", HISTORY))
    assert plan.rewritten is True
    assert plan.language == "vi"


# ---------------------------------------------------------------------------
# 7. ⭐ Viết lại hỏng **không được** làm hỏng lượt
# ---------------------------------------------------------------------------


def test_a_failing_rewrite_model_leaves_the_original_question_intact() -> None:
    llm = FakeProvider(raises=LLMError("provider 500"))
    plan = asyncio.run(QueryUnderstanding(llm=llm).plan("cái đó thì sao?", HISTORY))
    assert plan.rewritten is False
    assert plan.question == "cái đó thì sao?"
    assert plan.route == "retrieve", "vẫn phải truy hồi, chỉ là bằng câu gốc"


def test_a_slow_rewrite_model_does_not_hold_the_request(caplog: pytest.LogCaptureFixture) -> None:
    """⚠️ Bước này nằm **trước** truy hồi, nên nó cộng thẳng vào TTFB.

    ⭐⭐ Đồng hồ phải đặt **bên trong** vòng lặp sự kiện. Bản đầu của test này bọc
    `asyncio.run(...)` và đo 2,00 s trong khi timeout là 0,2 s — nó *trông* như
    timeout không chạy, nhưng cảnh báo vẫn được ghi. Thứ 2 giây kia đo là
    `asyncio.run` **join thread pool** lúc đóng vòng lặp, không phải thời gian
    request chờ.

    Và đó chính là cảnh báo đã ghi trong mã: `wait_for` huỷ được cái *chờ*, không
    huỷ được cái *thread*. Trong uvicorn (vòng lặp sống mãi) người dùng đi tiếp
    ngay, còn thread kia vẫn chạy nốt và vẫn bị provider tính tiền.

    Cùng họ với `M7` của `W4-06`: **cái sai ở chỗ đặt đồng hồ, không ở ngưỡng.**
    """
    release = threading.Event()

    class SlowProvider(FakeProvider):
        def complete(self, *args: Any, **kwargs: Any) -> LLMResponse:
            release.wait(10.0)
            return super().complete(*args, **kwargs)

    async def measure() -> tuple[QueryPlan, float]:
        started = time.perf_counter()
        plan = await QueryUnderstanding(llm=SlowProvider(), timeout_s=0.2).plan(
            "cái đó thì sao?", HISTORY
        )
        elapsed = time.perf_counter() - started
        # Thả thread ra **trước** khi vòng lặp đóng: `asyncio.run` join thread
        # pool ở bước cuối, nên quên dòng này thì chính test này treo 10 giây.
        release.set()
        return plan, elapsed

    with caplog.at_level(logging.WARNING, logger="serving.core.understanding"):
        plan, elapsed = asyncio.run(measure())

    assert plan.rewritten is False
    assert plan.question == "cái đó thì sao?"
    assert elapsed < 1.0, f"request chờ hết {elapsed:.2f}s — timeout không có tác dụng"
    assert "dùng câu gốc" in caplog.text


def test_without_a_rewrite_model_the_turn_still_works_and_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="serving.core.understanding"):
        plan = asyncio.run(QueryUnderstanding(llm=None).plan("cái đó thì sao?", HISTORY))
    assert plan.rewritten is False
    assert plan.route == "retrieve"
    assert "chưa cấu hình LLM" in caplog.text


def test_the_plan_reaches_the_sse_frame_in_a_shape_a_client_can_read() -> None:
    plan = asyncio.run(QueryUnderstanding().plan("  hello  ", []))
    assert plan.as_meta() == {
        "route": "no_retrieval",
        "language": "en",
        "rewritten": False,
        "question": "hello",
        "rewrite_ms": None,
    }


# ---------------------------------------------------------------------------
# `W5-06` — bước viết lại trong trace
# ---------------------------------------------------------------------------


class TestRewriteSpan:
    """Span `rewrite` mở **bên trong** `_rewrite`, không dựng lại từ `QueryPlan`.

    Dựng từ bên ngoài bằng `plan.rewrite_ms` sẽ phải bịa ra mốc bắt đầu; ở đây
    span đo đúng lời gọi model, và nó là span duy nhất của lượt có thể **quá
    hạn mà vẫn tốn tiền**.
    """

    def test_a_successful_rewrite_reports_its_own_tokens_and_cost(self) -> None:
        from serving.core.tracing import Trace, trace_scope

        trace = Trace()
        with trace_scope(trace):
            asyncio.run(QueryUnderstanding(llm=FakeProvider()).plan("cái đó thì sao?", HISTORY))
        span = trace.find("rewrite")
        assert span is not None
        assert span.kind == "generation"
        assert span.usage.prompt_tokens == 120
        assert span.usage.cost_usd == pytest.approx(0.00004)
        assert trace.unmeasured_cost_steps() == []

    def test_a_timed_out_rewrite_reports_no_cost_rather_than_zero(self) -> None:
        """⭐⭐ `wait_for` huỷ được cái *chờ*, **không** huỷ được cái *thread*:
        lời gọi kia vẫn chạy nốt và vẫn bị provider tính tiền. Chi phí thật của
        bước này là *chưa biết*.

        Ghi `0.0` biến một khoản chi không quan sát được thành một khoản chi
        bằng không — và tổng chi phí của trace trở thành một cận dưới đội lốt
        một phép đo. `unmeasured_cost_steps()` là chỗ điều đó phải hiện ra.
        """
        from serving.core.tracing import Trace, trace_scope

        release = threading.Event()

        class SlowProvider(FakeProvider):
            def complete(self, *args: Any, **kwargs: Any) -> LLMResponse:
                release.wait(5.0)
                return super().complete(*args, **kwargs)

        trace = Trace()
        try:
            with trace_scope(trace):
                asyncio.run(
                    QueryUnderstanding(llm=SlowProvider(), timeout_s=0.2).plan(
                        "cái đó thì sao?", HISTORY
                    )
                )
        finally:
            release.set()
        span = trace.find("rewrite")
        assert span is not None
        assert span.level == "WARNING"
        assert span.usage.cost_usd is None, "0.0 nghĩa là miễn phí; ở đây là chưa biết"
        assert trace.unmeasured_cost_steps() == ["rewrite"]
        assert trace.total_cost_usd() is None

    def test_a_failing_rewrite_reports_no_cost_either(self) -> None:
        from serving.core.tracing import Trace, trace_scope

        trace = Trace()
        with trace_scope(trace):
            asyncio.run(
                QueryUnderstanding(llm=FakeProvider(raises=LLMError("provider 500"))).plan(
                    "cái đó thì sao?", HISTORY
                )
            )
        span = trace.find("rewrite")
        assert span is not None
        assert span.level == "WARNING"
        assert span.usage.empty

    def test_no_rewrite_means_no_span(self) -> None:
        """Phần lớn lượt không viết lại. Một span `rewrite` rỗng ở mỗi lượt là
        một dòng vô nghĩa nhân với toàn bộ traffic."""
        from serving.core.tracing import Trace, trace_scope

        trace = Trace()
        with trace_scope(trace):
            asyncio.run(QueryUnderstanding(llm=FakeProvider()).plan("RRF là gì?", []))
        assert trace.find("rewrite") is None
