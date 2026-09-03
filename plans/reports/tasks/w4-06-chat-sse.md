# `W4-06` — `POST /chat` SSE, và đường phân giới của một phản hồi dạng dòng

> **Ngày:** 2026-09-03 · **Mã:** `serving/core/chat.py`, `serving/api/{chat,sse}.py`,
> `serving/__main__.py`, `rag_core/llm` (`astream`), `alembic/versions/0002_*`
> **Test:** `tests/integration/test_chat_stream.py` (15, cần uvicorn + Postgres thật),
> `tests/unit/test_chat_service.py` (13), `tests/unit/test_llm_streaming.py` (9)
> **Kiểm thật:** cả stack — bundle `0.2.0` → BGE-M3 GPU → Qdrant → reranker →
> DeepSeek → Postgres. §7

## 1. ⭐⭐ Quyết định của cả hạng mục: một đường thật trong mã

Sau khi byte đầu tiên của `200 OK` rời socket, không còn cách nào biến một lỗi
thành `503`. LLM chết ở token thứ 50 ⇒ client thấy một dòng SSE dừng lại, và
điều đó **trông y hệt** một câu trả lời ngắn đã kết thúc bình thường.

Nên `ChatService` tách làm hai nửa, và ranh giới là một dòng gọi hàm:

| | `prepare()` | `stream_turn()` |
|---|---|---|
| chạy khi | chưa gửi byte nào | đang gửi |
| lỗi thành | **HTTP status thật** | khung SSE `event: error` |
| gồm | kiểm tenant, mở hội thoại, **truy hồi** | gọi LLM, ghi message |

⭐ **Truy hồi nằm ở nửa trên** dù nó tốn ~800 ms TTFB. Đổi lại, Qdrant chết trả
`503` đọc được bằng máy thay vì một khung lỗi mà mọi client phải tự viết mã xử
lý. Nửa dưới càng mỏng thì càng ít thứ chỉ hỏng được theo kiểu không nói ra được.

Hệ quả: khung tên là **`sources`** (cái đã đưa cho model), không phải
`citations` (cái đã kiểm). `W4-09` thêm cái thứ hai; gộp tên chúng từ bây giờ là
cách chắc chắn để `W4-09` trở nên vô hình với client.

## 2. ⭐⭐ Ngắt kết nối giữa chừng — và test đầu tiên viết sai chỗ

Người dùng đóng tab ở giây thứ 3. Tiền đã tiêu, phần đã sinh là dữ liệu thật.

Cách viết hiển nhiên — `await self._save(...)` trong `finally` — **không bao giờ
chạy**: lúc đó task đang bị huỷ, nên `await` bị huỷ ngay. Phép tiêm lỗi xác nhận
đúng điều đó (M8, §6). Cách chặn là `asyncio.create_task`: hàm **đồng bộ**, chạy
xong kể cả trong lúc bị huỷ, và task nó tạo sống độc lập với request — kèm một
tham chiếu mạnh (`_PENDING`) vì `asyncio` chỉ giữ tham chiếu yếu.

⚠️ **Có hai đường huỷ, không phải một**, và test đầu tiên tôi viết không chạm
đường nào:

```python
async for event in gen:        # raise CancelledError ở ĐÂY
    ...                        # generator chỉ đang treo ở `yield` — nó không
                               # nằm trong chuỗi await, nên nó không nhận gì cả
```

Hai đường thật là `athrow(CancelledError)` (huỷ rơi lúc đang chờ token kế tiếp)
và `aclose()` → `GeneratorExit` (huỷ rơi lúc đang `send`, generator bị bỏ rơi
rồi mới đóng). Chúng là **hai exception khác nhau**, và một `except
asyncio.CancelledError` đơn độc bỏ lọt đúng một nửa số ca.

⭐ Và một phép tiêm không đỏ đã sửa chính thiết kế: ban đầu `finish_reason` khởi
tạo bằng `"client_disconnect"`, nên khối `except` chỉ gán lại giá trị nó đã có —
tức nó là **chú thích, không phải hành vi**, và xoá đi không test nào thấy. Đổi
khởi tạo thành `"unknown"` làm khối ấy gánh việc thật, và cho một đường thoát
thứ ba trong tương lai được ghi là *không biết* thay vì bị gán nhầm cho người
dùng.

## 3. ⭐⭐ Lần chạy thật đầu tiên trả về **0 ký tự**

Truy hồi đúng, 4 chunk, reranker chấm điểm, `200 OK`, dòng SSE hợp lệ. Và:

```
done: finish_reason=empty · completion_tokens=1024 · cost=$0.001511 · #delta=0
```

Toàn bộ 1024 token đi vào **chuỗi suy luận** của `deepseek-v4-flash`, thứ không
xuất hiện trong `content`. Đó là **đúng** phát hiện của `W3-04` (83% token vào
suy luận, 6/30 request trả rỗng), xuất hiện lại ở đường request — nơi nó không
còn là lãng phí mà là một **endpoint hỏng**, hỏng theo kiểu tệ nhất: không mã
lỗi, không exception, không khung `error`.

`W3-04` đã đo sẵn cách chữa cho từng nhà (`MIN_REASONING`), nhưng bảng ấy sống ở
`pipeline/indexing/contextualize.py` — và **`serving` không được import
`pipeline`** (`test_architecture_boundaries`). Nâng nó lên `rag_core.llm`, một
định nghĩa cho cả hai plane. Chép sang là tạo hai bảng số đo cho cùng một thứ,
và cái thứ hai sẽ là cái không ai sửa khi provider đổi hành vi.

Sau khi nối: **396 mẩu, TTFB 787 ms, `finish_reason=stop`, $0,000863** (§7).

💡 Thứ *báo* được chuyện này là `finish_reason="empty"` cộng với luật "không ghi
message rỗng". Nếu `finish_reason` chỉ có `stop`/`error`, hàng trong DB sẽ là
một câu trả lời trống và không phân biệt được với việc model im lặng.

## 4. Hai cột mới trên `message` (migration `0002`)

* **`model`** — quy tắc cứng #1 ("log model thực tế đã phục vụ") tới giờ chỉ
  sống trong một dòng log, tức nó biến mất theo chính sách giữ log. Sau khi
  `W4-08` bật fallback, không có cột này thì không tách được hai nhóm traffic ra
  để so.
* **`finish_reason`** — `stop` / `length` / `client_disconnect` / `error` /
  `empty` / `unknown`. Ba trong số đó chỉ tồn tại ở đường stream.

`nullable=True`, **không backfill**: message cũ không có cách nào biết model nào
sinh ra chúng, và điền một giá trị đoán vào đó là biến "không biết" thành "biết
sai".

## 5. ⭐ `psycopg` async không chạy trên vòng lặp mặc định của Windows

Lần chạy thật đầu tiên:

```
InterfaceError: Psycopg cannot use the 'ProactorEventLoop' to run in async mode
```

⚠️ Hướng lệch **ngược** với hướng quen thuộc: không phải "chạy ở máy tôi, hỏng ở
production" mà là "hỏng ở máy tôi, chạy được ở production" (container Linux của
`W4-13` không gặp). Cám dỗ là gạt đi như chuyện của Windows — nhưng nó chặn mọi
test tích hợp từ `W4-06` trở đi ở **nơi duy nhất tôi thực sự chạy chúng**.

⚠️ Và cách chữa hiển nhiên không hoạt động. `asyncio.set_event_loop_policy(
WindowsSelectorEventLoopPolicy())` là câu trả lời mọi nơi đưa ra; uvicorn **không
đi qua policy** — `uvicorn/loops/asyncio.py` trả thẳng một `loop_factory`, và
trên win32 nó là `ProactorEventLoop`. Lỗi giữ nguyên từng chữ. Tôi đo được điều
đó trước khi đọc mã uvicorn.

Nên có `serving/__main__.py`: dựng `uvicorn.Server` rồi tự chạy nó bằng
`asyncio.run(..., loop_factory=SelectorEventLoop)`. `make serve` đổi theo.

## 6. Tiêm lỗi — 11 phép, 9 đỏ ngay, **2 phơi ra lỗ thật**

| # | tiêm vào | kết quả |
|---|---|---|
| M1 | bỏ `stream_options.include_usage` | ✅ đỏ |
| M2 | đọc `usage` trong nhánh `if choices:` | ✅ đỏ |
| M3 | thử lại kể cả sau khi đã phát token | ✅ đỏ |
| M4 | `encode()` gửi text thô thay vì JSON | ✅ đỏ |
| M5 | bỏ `except (CancelledError, GeneratorExit)` | ⚠️ **không đỏ** → §2 |
| M6 | gửi `sources` sau khi stream xong | ✅ đỏ (3 test) |
| M7 | gọi `retrieve()` thẳng trên event loop | ⚠️ **không đỏ** → dưới |
| M8 | `await self._save(...)` trong `finally` | ✅ đỏ |
| M9 | bỏ `tenant_filter`, dùng filter của client | ✅ đỏ |
| M10 | lịch sử `ORDER BY … ASC` (giữ 10 message **đầu**) | ✅ đỏ |
| M11 | bỏ `X-Accel-Buffering` | ✅ đỏ |

⭐ **M7 là bài học đắt nhất, và nó lặp lại `W4-03`.** Test đo `/health` *bên
trong* khối `with client.stream(...)` — nhưng `stream()` chỉ trả về khi **header**
đã tới, mà header của SSE ra sau khi `prepare()` (gồm cả truy hồi) đã xong. Phép
đo rơi đúng vào lúc vòng lặp đã rảnh trở lại: test xanh kể cả khi truy hồi chặn
event loop. Cùng khuôn với `test_a_hung_dependency_does_not_hang_ready`: **cái
sai không nằm ở ngưỡng mà ở chỗ đặt đồng hồ.** Sửa bằng cách bắn `/chat` từ một
luồng khác; phép tiêm lập tức cho **1298 ms** so với ngưỡng 500 ms.

⚠️ Và một lỗi cùng họ ở chính bộ test này: mọi tiến trình uvicorn phụ dùng chung
một hằng số `PORT`, nên chúng **không bind được** và vòng chờ khởi động hỏi
`/health` của server **cũ**. Bốn test chạy trên server sai; ba trong số đó vẫn
xanh vì hành vi mặc định giống nhau. Một test xanh có hai cách giải thích, và
"nó chưa từng chạy thứ tôi tưởng" là cách phải loại trừ trước (`W4-05` §7).

## 7. Chạy thật — cả stack, dữ liệu thật, tiền thật

```
/ready  → 200 · bundle 0.2.0 · qdrant 5,91 ms · postgres 137,5 ms
POST /chat "What are the main constraints on private investment in Vietnam?"
  sources  4 chunk, rerank score 3,79 / 3,65 / 3,37 / 2,59
  delta    396 mẩu · TTFB 787 ms · tổng 3.223 ms
  done     finish_reason=stop · deepseek-v4-flash · 1567+400 tok · $0,000863
GET /conversations/{id}
  user      cit=0
  assistant model=deepseek-v4-flash finish=stop lat=3226ms cit=4
```

Câu trả lời trích `[1][2][3]` đúng vị trí, tiếng Việt, cho một câu hỏi tiếng
Anh — tức đường cross-lingual mà cả `W2-01`/`TD-37` xây đã chạy tới tận đầu ra.

⚠️ **Nhưng nó trả lời sai ngôn ngữ.** Luật 4 của prompt viết *"trả lời bằng đúng
ngôn ngữ của câu hỏi"* và model không theo. Đó là chuyện của `W4-07` (phát hiện
ngôn ngữ) và `W4-11` (prompt có version) — ghi lại ở đây vì nó là bằng chứng
rằng một dòng chỉ dẫn trong prompt **không phải** một cơ chế.

### ⚠️⚠️ Và lần chạy thật đầu tiên tìm ra một lỗ chặn `W4-13`

Request đầu trả `sources: []`. Nguyên nhân: **15.814 chunk trong Qdrant không có
`tenant_id` nào** — đường index chưa bao giờ ghi field ấy. Với `tenant_filter()`
của `W4-04` áp lên mọi request, toàn bộ corpus **vô hình với mọi người dùng đã
xác thực**.

Đó chính xác là hành vi `W2-06` đã ghi và ghim bằng test (*"point thiếu
`tenant_id` không khớp bất kỳ filter tenant nào"*) — đúng, an toàn, và cho tới
hôm nay chưa ai chạm vào. Nó chỉ lộ ra khi cả ba tầng cùng chạy một lần.

Tạm gán cho toàn corpus để có bằng chứng chạy thật (không chạm vector nào, đúng
tiền lệ `backfill-payload` của `W2-06`):

```
POST /collections/rag_bgem3_ctx/points/payload?wait=true
{"payload": {"tenant_id": "public"}, "filter": {}}      → 15.814/15.814
```

**`TD-40`**: đường index phải tự ghi `tenant_id`, nếu không lần build lại kế
tiếp xoá nó đi — im lặng, và triệu chứng là "mọi câu hỏi đều không tìm thấy gì".

## 8. Ba quyết định nhỏ

* **Hai transaction ngắn, không một transaction dài ôm cả stream.** `SET LOCAL`
  chết cùng transaction, nên một `commit()` giữa chừng **tháo tenant ra** và mọi
  câu sau đó bị RLS chặn — hướng hỏng tốt, triệu chứng đánh lừa. Và độc lập với
  điều đó: giữ một transaction mở suốt 30 giây sinh token là ghim một kết nối
  Postgres cho mỗi người đang gõ chuyện.
* **Câu hỏi được ghi *trước* khi sinh.** Mất câu trả lời thì người dùng hỏi lại
  được; mất câu hỏi thì lịch sử nói dối về chuyện đã xảy ra.
* **Retry của `astream` chỉ hợp lệ trước token đầu tiên.** Sau đó, thử lại nghĩa
  là sinh lại từ đầu — kết quả là một đoạn văn trôi chảy, mạch lạc, và không
  phải cái mà model nào nói ra.

## 9. Một lỗi có sẵn mà bộ test tự che

`create_app()` gọi `configure_logging()`, thứ **gỡ sạch handler của root
logger** — gồm cả handler mà `caplog` vừa gắn. Mọi test dùng `caplog` chạy *sau*
một test dựng app đều thấy `caplog.text` rỗng.

Lỗi này có từ `W4-03`. Không ai thấy vì `pytest-randomly` đổi thứ tự file mỗi
lần: nó chỉ đỏ khi `test_health.py` rơi trước `test_chunking.py`. ⚠️ Con số
"1613 passed" của những phiên trước vì thế đúng cho **thứ tự hôm đó**, không
đúng cho bộ test. Chữa bằng một fixture `autouse` khôi phục root logger — chứ
không phải thêm một cờ `configure_logs=False` vào mã production để chiều test.

Bộ test giờ chạy xanh ở **cả hai** chế độ: `1856 passed, 3 skipped` với
`-p no:randomly` và với thứ tự ngẫu nhiên.

## 10. Còn nợ

* **`TD-40`** — đường index không ghi `tenant_id` (§7). Chặn `W4-13`.
* **Tiến trình tắt khi còn task ghi đang chờ ⇒ mất bản ghi.** Giới hạn thật của
  `create_task`; cách chữa đúng là outbox, `W5` — không phải một `sleep` ở chỗ
  shutdown.
* **`GET /conversations/{id}` không phân trang** — một hội thoại 500 lượt trả về
  vài MB.
* **Cắt lịch sử theo *số message*, không theo token.** Sai theo hướng nguy hiểm
  khi 10 message ấy đều dài. `HFTokenCounter` (`W1-10`) đã có sẵn để thay.
* **Không có heartbeat SSE.** Proxy có idle timeout sẽ cắt một stream im lặng
  lâu; hiện khoảng im lâu nhất là TTFB ~800 ms nên chưa chạm, nhưng `W4-08` với
  fallback + retry sẽ kéo nó dài ra.
* **`SYSTEM_PROMPT` là hằng số trong mã** — `W4-11`. Và nó **không** ép được
  ngôn ngữ đầu ra (§7).
