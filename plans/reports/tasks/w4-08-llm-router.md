# `W4-08` — bộ định tuyến LLM, và ba loại "hỏng" không được đối xử giống nhau

> **Ngày:** 2026-09-04 · **Mã:** `rag_core/llm/router.py` (mới),
> `rag_core/llm/openai_compat.py`, `rag_core/settings.py`, `serving/api/app.py`,
> `serving/api/chat.py`, `serving/core/chat.py`
> **Test:** `tests/unit/test_llm_router.py` (**27**),
> `tests/unit/test_chat_service.py` (+1), `tests/integration/test_chat_stream.py` (+4)
> **Bằng chứng:** `reports/probes/w4-08-router-real.json` (DeepSeek → GLM, hai
> nhà cung cấp thật) · **Chi phí API:** $0,00033

## 1. ⭐⭐ Chuyển nhà cung cấp GIỮA STREAM nối hai câu trả lời khác nhau

Đây là ràng buộc quan trọng nhất của cả hạng mục, và cách viết hiển nhiên vi
phạm nó:

```python
for route in routes:                      # SAI
    try:
        async for chunk in route.astream(...):
            yield chunk
        return
    except LLMError:
        continue                          # ← đã gửi 50 token của route 1 rồi
```

Người dùng nhận nửa câu trả lời của DeepSeek nối vào nửa câu trả lời của GLM:
một đoạn văn **trôi chảy, mạch lạc, và không model nào nói ra**, kèm
`finish_reason: stop`. Không mã lỗi, không dấu vết.

Ranh giới là **mẩu đầu tiên** — y hệt ranh giới retry mà `OpenAICompatProvider`
đã dựng một tầng dưới ở `W4-06`. Và test cho nó rất dễ thành test rỗng, nên ở
đây có **hai** test kẹp lấy ranh giới:

| | hỏng **trước** mẩu đầu | hỏng **sau** mẩu đầu |
|---|---|---|
| chuyển nhánh | ✅ có | ❌ không |
| người gọi thấy | câu trả lời của nhánh dự phòng | `LLMError`, phần đã sinh giữ nguyên |

Chỉ một trong hai thì không đo được gì: test "không bao giờ chuyển nhánh" cũng
xanh nếu router **luôn** bỏ fallback. Đo trên đường thật (SSE, uvicorn) chứ
không chỉ ở unit: `CHÍNH-0 CHÍNH-1 ` rồi `event: error`, và chuỗi của nhánh dự
phòng **không** xuất hiện.

## 2. ⭐ Ba loại hỏng, ba cách đối xử

| loại | chuyển nhánh? | tính cầu dao? | vì sao |
|---|---|---|---|
| `BudgetExceeded` | **không** | **không** | hết tiền mà chuyển nhánh là tiêu **thêm** |
| `PermanentLLMError` (4xx của mình) | có | **không** | request của mình sai, không phải nhà chết |
| 429/5xx/timeout/mạng | có | **có** | đây mới là bằng chứng nhà cung cấp hỏng |

Hàng giữa cần một lớp lỗi mới. Trước hạng mục này, "sai tên model" và "provider
sập" là **cùng một** `LLMError` nhìn từ ngoài — nên mở cầu dao cho nó nghĩa là
**tự cắt** nhà cung cấp chính 30 giây vì một lỗi mà thử lại bao nhiêu lần cũng
thế. `PermanentLLMError(LLMError)` tách hai ca ra; vì nó vẫn là `LLMError` nên
không một `except` nào đã viết trước đây đổi hành vi.

Và chuyển nhánh cho một 400 **là** đúng, vì hai route mang `extra_body` khác
nhau — xem §3.

## 3. `extra_body` theo **từng route**, đo lại chứ không trích lại

`W4-06` đặt `MIN_REASONING` ở `ChatService` vì lúc đó chỉ có một nhà. Với hai
nhà, một bảng dùng chung hỏng **im lặng** ở nhà này và **ồn ào** ở nhà kia. Gọi
GLM bằng tham số của DeepSeek, đo lại hôm nay trên API thật:

```
HTTP 400 {"code":"1210","message":"This model always engages in thinking
          and cannot be disabled; please use low, high, or max"}
```

Nên `Route.extra_body`, và `ChatService.extra_body` / `QueryUnderstanding
.extra_body` giờ để trống: một giá trị ở tầng lời gọi **ghi đè** bảng của mọi
nhánh, tức áp tham số DeepSeek lên nhánh GLM. Đúng cái mà `W4-06` đã ghi trước
là sẽ phải chuyển.

## 4. ⭐ Half-open cho **đúng một** request, và một chỗ tinh hơn thế

Cầu dao mở → sau thời gian nguội → cho **một** request thăm dò. Cho tất cả thì
lúc cooldown hết, 200 request đang chờ cùng lao vào một nhà vẫn đang chết — cầu
dao thành bộ tạo đợt lỗi định kỳ thay vì một cái van. Test tuần tự vẫn thấy "nó
thử lại được" nên vẫn xanh; thứ phân biệt là gọi `allow()` **hai lần** trước khi
`record()`.

⭐⭐ Chỗ tinh hơn thì một phép tiêm lỗi mới tìm ra (§6): coi `neutral` như
`success`. Nghe vô hại — cả hai đều "không phải lỗi của nhà cung cấp" — nhưng
`success` **xoá bộ đếm** và **đóng mạch**, nên một lượt hết ngân sách chen vào
giữa chuỗi hỏng đưa bộ đếm về 0 và cầu dao **không bao giờ mở**, kể cả khi nhà
cung cấp đã chết hẳn.

## 5. "Từ chối có thông báo rõ" là một HTTP status, không phải một stream chết

Trần chi phí ngày cạn ⇒ `BudgetExceeded`. Nếu để nó rơi vào trong stream thì
client nhận `200 OK` và một dòng token trống — đúng chế độ hỏng mà đường phân
giới của `W4-06` tồn tại để chặn. Nên:

* `LLMRouter.assert_within_budget()` gọi ở đầu `prepare()`, **trước** cả truy
  hồi ⇒ `429` kèm `Retry-After` đếm tới nửa đêm UTC.
* Phép kiểm ấy chỉ trả lời *"đã cạn chưa"*: ở thời điểm `prepare()` prompt chưa
  tồn tại nên không ước được giá lời gọi sắp tới. Lời gọi **vượt** trần vì thế
  vẫn xảy ra sau `200 OK`, và ở đó nó là khung `error` với `finish_reason`
  riêng — `budget`, không gộp vào `error`: hết tiền và provider chết cần hai
  hành động khác nhau, gộp lại thì cả hai không đếm được.

⚠️ Một lỗ nhỏ đã bịt: mẩu mang `usage` là mẩu **cuối**. Stream đứt giữa chừng
thì nó không bao giờ tới, và ghi 0 nghĩa là một client ngắt kết nối ở **mọi**
request sẽ không bao giờ chạm trần trong khi token đã tiêu thật. Nên đứt sau khi
đã sinh chữ ⇒ ghi phần đã giữ chỗ.

⚠️⚠️ Và trần này đếm **trong tiến trình**: N replica ⇒ trần thật N×, restart ⇒
về 0. Cùng giới hạn với hạn mức nhịp của `W4-04` (`TD-39`). Nó chặn được ca nó
sinh ra để chặn — một vòng lặp hỏng đốt sạch ngân sách trong mười phút — nhưng
nó **không phải** một trần đúng nghĩa. Ghi ra để không ai đọc
`chat_daily_budget_usd` mà tưởng đã có trần thật → `TD-47`.

## 6. Tiêm lỗi — 12 phép, 10 đỏ ngay, 2 phơi ra lỗ thật

| tiêm vào | kết quả |
|---|---|
| chuyển nhánh kể cả khi đã `yield` | ✅ đỏ |
| `BudgetExceeded` bị nuốt thành lỗi route | ✅ đỏ |
| `PermanentLLMError` tính vào cầu dao | ✅ đỏ |
| half-open cho **mọi** request qua | ✅ đỏ |
| không tính tiền khi stream đứt giữa chừng | ✅ đỏ |
| bỏ qua / đảo ưu tiên `extra_body` của route | ✅ đỏ (2 phép) |
| log model **yêu cầu** thay vì model **đã phục vụ** | ✅ đỏ |
| cầu dao mở ở lần hỏng thứ 4 thay vì thứ 3 | ✅ đỏ |
| bỏ kiểm ngân sách ở `prepare()` | ✅ đỏ (test tích hợp) |
| **`neutral` được coi như thành công** | ⚠️ **xanh** → lỗ thật, xem §4 |
| **ngân sách cuộn theo giờ máy thay vì UTC** | ⚠️ **xanh** |

Phép thứ hai sống sót vì test cuộn ngày **monkeypatch chính `_today`**. Không
chữa được bằng cách so `_today()` với ngày hiện tại: máy này ở UTC+7 nên hai
lịch chỉ khác nhau 7 giờ mỗi ngày — một test như vậy **xanh phần lớn thời gian
kể cả khi có bug**, và đó là dạng test tệ hơn không có test. Chữa bằng cách đóng
băng đồng hồ ở đúng chỗ hai lịch lệch nhau (23:30 UTC = 06:30 hôm sau, giờ máy).

Sau khi bịt, 12/12 đỏ.

## 7. Lần chạy thật — DeepSeek → GLM, hai nhà cung cấp thật

| | kết quả |
|---|---|
| nhánh chính không kết nối được | GLM trả lời **625 ký tự**, $0,000103; cầu dao `closed`, 1 lần hỏng |
| nhánh chính **sai slug** (HTTP 400 thật) | chuyển nhánh, cầu dao `closed`, **0** lần hỏng |
| tham số DeepSeek đưa cho GLM | HTTP 400 `1210` — xem §3 |
| ba lần hỏng liên tiếp | cầu dao **`open`**, request thứ tư không chạm nhánh chính |

Hàng thứ hai là hàng đáng nhất: nó kiểm rằng `PermanentLLMError` phân loại đúng
trên một **phản hồi thật** chứ không trên một giả định về mã lỗi.

⚠️ Bốn độ trễ đo được là 2944 / 2289 / 3240 / 2010 ms. Request thứ tư nhanh
nhất, nhưng **n = 1 mỗi ô** nên tôi không kết luận gì từ đó — bằng chứng cứng là
`state: open`, không phải con số độ trễ. Ở đây lợi ích của cầu dao vốn nhỏ vì
lỗi DNS trả về ngay; nó chỉ lớn khi nhà cung cấp **treo** thay vì từ chối.

⚠️ Một hạn chế lộ ra: lỗi cuối "mọi route đều không phục vụ được" gộp mọi lý do
thành một `LLMError` phẳng, nên kiểu `PermanentLLMError` **không** còn đọc được
ở đó. Dòng log từng route vẫn ghi đúng phân loại; người gọi thì không phân biệt
được. Chấp nhận hôm nay, ghi ra để không ai dựa vào nó.

## 8. Dự đoán — 5/6 đúng, và cái sai là cái tôi tự tin nhất về *chỗ*

| | dự đoán | thực tế |
|---|---|---|
| P1 | fallback giữa stream là chỗ nguy nhất; test cho nó dễ rỗng | ✅ cài đúng từ đầu, và phép tiêm xác nhận test **có** sống |
| P2 | `BudgetExceeded` không bị `except LLMError` bắt nhầm | ✅ đúng, không có bug |
| P3 | `extra_body` phải theo từng route | ✅ đúng, và HTTP 400 thật xác nhận |
| P4 | phép tiêm half-open **không** bị bộ test đầu bắt | ❌ bị bắt (test hai-`allow()` viết sẵn). Nhưng một chỗ **khác** của cầu dao (`neutral`) thì sống sót — đúng hướng, sai cơ chế |
| P5 | trần trong tiến trình = `TD-39` lặp lại | ✅ đúng, đã ghi thành `TD-47` |
| P6 | retry mặc định của provider quá chậm cho đường request | ✅ đúng → `chat_max_retries = 1` |

## 9. Còn lại

* **Fallback mặc định là `none`, có chủ đích.** Một fallback chưa từng chạy là
  một fallback chưa biết có chạy không, và lần đầu nó chạy sẽ là lúc nhà chính
  đang chết — chỗ tệ nhất để phát hiện sai key hoặc sai slug. `GET /admin/llm`
  là chỗ duy nhất nhìn thấy nó đã phục vụ bao nhiêu request.
* **Trần chi phí không phân theo tenant.** Một khách hàng đốt hết ngân sách thì
  mọi khách hàng còn lại bị `429`. Đúng cho một tenant, sai ngay khi có hai →
  cùng chỗ chữa với `TD-47` (bộ đếm dùng chung có khoá theo tenant).
* **Cầu dao không có nửa mở theo tỉ lệ.** Nó là bật/tắt: 3 lần hỏng liên tiếp là
  mở hẳn. Một nhà cung cấp hỏng 40% request sẽ **không bao giờ** mở mạch, và đó
  là ca thật hơn ca chết hẳn. Cần cửa sổ trượt theo tỉ lệ lỗi, không phải đếm
  liên tiếp.
* **`vLLM profile` của DoD chưa dựng.** Nó là "(tuỳ chọn)" trong plan, và với
  kiến trúc hiện tại nó chỉ là thêm một `Route` trỏ vào một endpoint
  OpenAI-compat — không cần mã mới. Chưa có endpoint nào để trỏ tới.
