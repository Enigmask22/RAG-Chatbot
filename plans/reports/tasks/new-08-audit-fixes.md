# `NEW-08` — Gói vá audit: 9 món, $0, và một hồi quy thật lộ ra giữa chừng

> Nguồn: audit toàn cục `reports/tasks/audit-pre-w510.md` (2026-09-05, theo yêu
> cầu của bạn trước khi `W5-10` tự động hoá đường promote). Gói này trả
> `AU-01`…`AU-07` + `AU-13` + `TD-64`; phần còn lại của audit nằm ở `TD-84`
> kèm chỗ trả từng mục.

## 0. Kết quả một dòng

**10** lỗi vá xong (9 của audit + 1 do chính probe của gói này tìm ra — §9),
30 test mới, tiêm lỗi **22 phép: 21 đỏ, 1 sống sót có chủ đích kèm chốt bù đo
sống PASS**. `TD-64` **đóng bằng cách sửa phép đo chứ không sửa model**:
citation quote 0,8308 → **0,8662 ≥ 0,85 ✅**. Migration `0005` thuận nghịch đã
kiểm. Chi phí $0 (+~$0,006 hai lượt probe).

## 1. `AU-01` — cầu dao đếm "khách đóng tab" thành "nhà cung cấp hỏng"

`router.astream` khởi tạo `outcome = "failure"`; `CancelledError`/`GeneratorExit`
là `BaseException` nên không rơi vào `except Exception` nào — nhưng `finally`
vẫn gọi `breaker.record(outcome)` với giá trị khởi tạo. Comment ngay đó xử lý
đường huỷ cho phần **tiền** (đúng) và bỏ sót phần **cầu dao**: ba người dùng
liên tiếp bỏ ngang (đúng lúc provider chậm — lúc người ta hay bỏ nhất) là mạch
mở 30 giây cho một route khoẻ mạnh.

Vá: nhánh `except (asyncio.CancelledError, GeneratorExit)` tường minh, ghi
`"neutral"`. **Không phải `"success"`** — mutation M2 phơi ra phân biệt này:
`success` reset bộ đếm, tức một client hay đóng tab giữ một provider hỏng thật
mãi ở `closed`. Test `test_a_cancelled_stream_does_not_reset_the_failure_counter`
kẹp đúng ranh giới ấy (2 lỗi thật + 1 huỷ + 1 lỗi thật = vẫn mở mạch).

## 2. `AU-02` — khoá semantic cache thiếu `top_k` lẫn `filters`

Cùng câu hỏi với `top_k=5` và `top_k=20` là hai lượt sinh trên hai bộ ngữ cảnh;
request mang `filters={"doc_type": …}` nhận nguyên văn câu trả lời của lượt
không filter là vi phạm **phạm vi dữ liệu được yêu cầu**, không phải chuyện
hiệu năng. Hai trục xử lý khác nhau, có chủ đích:

* `top_k` vào **namespace** (`{version}+{prompt}+k{top_k}`): client dùng
  `top_k` khác mặc định *một cách nhất quán* vẫn giữ được cache của nó.
* `filters` vào **điều kiện loại** (`cache_eligible`): cache này bảo thủ có
  chủ đích (xem docstring `semantic_cache.py`), ca có filter hiếm tới mức một
  khoá riêng chỉ nuôi entry chết. Băm filter vào khoá là độ phức tạp mua về
  số hit ~0.

`ChatTurn.resolved_top_k` giữ cho đầu **ghi** dùng đúng namespace của đầu
**đọc** — mutation M5 chứng minh lỗ này có thật trước khi test
`test_the_cache_is_stored_under_the_top_k_that_produced_the_answer` bịt nó.

## 3. `AU-03` — exception nội bộ rò ra client, và một test dựa vào chính chỗ rò

Ba chỗ cùng pattern `f"{type(exc).__name__}: {exc}"`: catch-all 503 của
`api/chat.py`, khung SSE `error` cho `LLMError`, và `_refused` của admin.
Qdrant rớt kết nối là client đọc được tên service, port, tên collection.

Vá hai chỗ đầu: client nhận thông báo chung + `request_id`/`trace_id` để đối
chiếu; nguyên văn ở lại `logger` và trace (Langfuse — mặt phẳng vận hành).
**Giữ nguyên `admin.py` có chủ đích**: route đã cần scope `admin`, chi tiết
lỗi ở đó là chức năng cho người vận hành (comment trong mã giải thích vì sao
tên lớp exception cần thiết — "timed out" trần không nói được *cái gì* hết giờ).

⚠️ **Hồi quy thật, và nó dạy đúng bài của audit**:
`test_a_primary_that_dies_mid_stream_becomes_an_error_frame_not_a_spliced_answer`
đỏ sau bản vá — nó nhận diện *đường lỗi nào bắn ra* bằng cách đọc lời nội bộ
của router trong `detail` (`"không ai nói"`). Tức là test **dựa vào chính chỗ
rò**. Nó lọt qua lượt chạy giữa chừng vì tôi chỉ chạy `test_feedback.py` +
`test_chat_stream.py` một phần — đúng lỗi "thay lệnh rộng bằng lệnh hẹp" của
`W5-06`; trọn bộ integration bắt được ngay. Sửa test: đường lỗi chứng minh
bằng **hành vi** (text không bị nối, không `done`), và test giờ canh luôn
tính kín (`"không ai nói" not in detail`).

## 4. `AU-04` — DSN không URL-encode: mật khẩu chứa `@` là kết nối sang host khác

`_dsn` là f-string trần. `s3cret@evil.com/db` làm SQLAlchemy parse `evil.com`
thành host — kể cả trên DSN migration mang quyền superuser, không lỗi nào cảnh
báo. Vá bằng `quote_plus` cho cả user lẫn password; test round-trip qua
`make_url` với mật khẩu chứa `@ : / %`.

## 5. `AU-05` — PII đi vào Langfuse không redact (leo thang `TD-73`)

`RedactingFilter` chỉ phủ Python logging; câu hỏi người dùng (`trace.input`),
output, `statusMessage` của span (một exception có thể mang nguyên câu hỏi) và
comment feedback đi thẳng sang Langfuse — nơi `TD-73` đã ghi là mọi tenant vào
**một** project. Redact ở **biên** (`_redact` đệ quy trong `encode_trace`/
`encode_score`), không ở nguồn Postgres: RLS đã rào Postgres và người dùng
phải đọc lại được nguyên văn câu của mình. Riêng **comment feedback** redact
thêm **tại nguồn** vì nó chảy đi ba ngả (review queue, Langfuse, file ứng viên
golden — thứ sẽ sống trong git). Redact chỉ đụng trường nội dung, không đụng
id — `redact_pii` thay chuỗi chữ số dài, và một `trace_id` bị thay là một điểm
số vĩnh viễn không gắn được vào trace (có test ghim).

## 6. `AU-06` — query embed HAI lần mỗi lượt cache-eligible

`prepare()` gọi `embed_query` cho cache rồi — khi miss — `retrieve()` gọi
`embed_query_hybrid` trên **cùng chuỗi** lần nữa: +12,6 ms và giữ khoá model
(`TD-63`, thứ đã gây 503 dưới tải) **hai lần** mỗi lượt. Ngưỡng cosine 0,96
làm đa số lượt eligible là miss, nên đây là đường nóng chứ không phải ca hiếm.
Hai agent review độc lập cùng tìm ra.

Vá: một `embed_query_hybrid` cho cả hai — phần dense tra cache (an toàn vì
BGE-M3 để prefix rỗng có chủ đích và cả hai đường cùng `_forward` chuẩn hoá
L2), cặp vector truyền vào `retrieve(precomputed=…)` xuyên qua
`RerankedRetriever` → `QdrantHybridRetriever`. Điều kiện chọn đường nằm trong
`wants_precomputed()` — `isinstance` với class thật chứ không duck-typing, vì
truyền một cặp vector vào một retriever hiểu sai nó là loại lỗi *trông vẫn
chạy*. Mọi retriever giả trong test rơi về đường cũ, nên hành vi các bài
`W4-10` không đổi một byte.

**M14 (dây nối trong `prepare()`) là mutation sống sót có chủ đích** — không
có test service-level cho `prepare()` (cần registry + sessions + bundle thật).
Chốt bù: probe đo sống trên app thật, bundle 0.2.1, đếm forward pass bằng
proxy quanh đúng instance embedder — xem §9 và
`probes/new08-au06-single-embed.json`.

## 7. `AU-07` — ghép câu hỏi–câu trả lời theo đồng hồ, lệch được khi hai lượt chồng nhau

`_questions_for` chọn user message *muộn nhất* trước `created_at` của answer.
Hàng user ghi ngay ở `_open_turn`; hàng assistant ghi trong task nền **sau**
khi stream xong — đa tab hoặc retry là đủ để ứng viên golden mang **câu hỏi B
dán lên câu trả lời của A**, sai không dấu vết. Khoá thật đã có sẵn trên
`ChatTurn.user_message_id` từ `W4-06`; migration `0005` chỉ là việc ghi nó
xuống. Join theo id, suy luận thời gian chỉ còn là fallback cho hàng cũ —
và test fallback ghim rằng nó **tồn tại**, không ghim rằng nó đúng: rủi ro
chọn nhầm là thuộc tính của dữ liệu cũ. Không backfill: backfill bằng đúng
suy luận bị thay thế là đóng dấu "khoá thật" lên một phép đoán.

Kịch bản chồng lấn test bằng cách **dựng tay đúng thứ tự ghi thật** (ba INSERT
với `created_at` tường minh) — TestClient đồng bộ không tạo được hai stream
chồng nhau thật, và điều khiển đồng hồ mới là thứ làm bài test tất định.

## 8. `TD-64` — 19 "lỗi" citation là từ chối oan của matcher, không phải của model

Docstring `citations.py` tự đặt điều kiện: *"nới lỏng quy tắc nào phải kèm một
phép đo cho thấy quy tắc hiện tại từ chối oan"*. `W5-02` chính là phép đo đó:
19/67 lỗi cấp quote là hai mẩu **nguyên văn** nối bằng `...` — một cách trích
dẫn hợp lệ trong văn viết, matcher chuỗi-con coi là quote bịa. Đề xuất cũ của
nợ (cấm dấu lược trong prompt) là sửa ngược: cấm một cách trích dẫn đúng để
chiều một matcher hẹp.

`_quote_matches`: tách theo `...`/`…`/`[...]`/`[…]`, mọi mảnh phải khớp nguyên
văn **theo đúng thứ tự, không chồng lấn** (`find` tiếp từ cuối mảnh trước).
Thứ tự là phần giữ độ chặt; quote toàn dấu lược trả `False` (trước đây
`"..." in content` có thể `True` — một quote không nói gì được đóng dấu
verified).

Chấm lại 396 quote của `w5-answers-v1` bằng matcher mới ($0, thuần chuỗi trên
sidecar đã lưu): **0,8308 → 0,8662 ≥ 0,85 ✅** · 14/19 quote dấu lược được cứu
(5 còn lỗi chép sai bên trong mảnh — lỗi thật của model) · **0** quote từng
xanh bị rớt · 10 lỗi gắn nhầm số nguồn giữ nguyên (đúng chế độ hỏng `W4-09`
sinh ra để bắt). `probes/new08-td64-rescore.json`.

## 9. Đo sống `AU-06` — và lượt probe ĐẦU bắt được lỗi thật mà 27 test mới trượt qua

App thật (bundle 0.2.1, BGE-M3 GPU, Redis, DeepSeek), proxy đếm quanh đúng
instance embedder của retriever đang phục vụ.

**Lượt probe đầu: `embed_query_hybrid=1, embed_query=1` — tức `wants_precomputed`
trả `False` trên server thật.** Nguyên nhân: production bọc cả chuỗi truy hồi
bằng `TracedRetriever` (`instrument_retriever` của `W5-06`), nên
`retriever.base` của server thật là một `TracedRetriever(hybrid)` chứ không
phải `QdrantHybridRetriever` — `isinstance` fail **lặng lẽ**, đường
embed-một-lần không bao giờ chạy, và mọi unit test xanh vì chúng dựng class
trần. Đây chính xác là loại lỗi mà mutation M14 sống sót cảnh báo, và là lý do
chốt bù phải là một phép đo **trên hệ đang phục vụ** chứ không phải thêm test
cùng hình dạng với các test đã có.

Vá hai chỗ: `_unwrap_traced` trong `wants_precomputed` (bóc `_inner` ở cả hai
tầng), và `TracedRetriever.retrieve` chuyển tiếp `precomputed` (cùng luật
"chỉ truyền khi có" với reranked). Hai bài test mới dựng **đúng chuỗi mà
`instrument_retriever` dựng** — không phải fixture cùng-hình-dạng-hôm-nay —
cộng hai mutation M21/M22 xác nhận chúng có răng.

**Lượt probe sau vá — PASS cả hai lượt** (`probes/new08-au06-single-embed.json`):

| | `embed_query_hybrid` | `embed_query` |
|---|---:|---:|
| lượt 1 (miss, 768 frame trả thật) | **1** | **0** |
| lượt 2 (cùng câu, cộng dồn) | 2 (+1 cho lookup) | 0 |

Trước `NEW-08` mỗi lượt như vậy là **2** forward pass và **2** lần giữ khoá
`TD-63`.

## 10. Test và tiêm lỗi

| | |
|---|---|
| test mới | 30 (unit 25 · integration 5) |
| tiêm lỗi | **22 phép: 21 đỏ · 1 sống sót CÓ CHỦ ĐÍCH** (M14, chốt bù §9 PASS) |
| bộ mặc định | **2 291 xanh** (3 skip) |
| integration | **282 xanh** (tầng CI, 0 đỏ) · gpu **13/13** (đổi chữ ký `RerankedRetriever`) |
| lint | ruff + format + mypy + `mypy --platform linux` sạch |
| migration `0005` | `upgrade → downgrade -1 → upgrade` sạch, cột kiểm bằng inspector |

Ba chuyện của lượt tiêm đáng ghi hơn con số:

* **Lượt đầu "20/20 đỏ" có một cái đỏ GIẢ**: M14 bị "giết" bởi test
  spliced-answer — chính cái test đang đỏ sẵn vì hồi quy §3 (harness chạy
  integration khi unit xanh, và gặp cái đỏ có sẵn). Phiên bản mutation của bài
  học `W5-08`: *một phép đo đúng quy trình vẫn vô nghĩa khi nền không xanh*.
* **M14 tiêm lại trên nền xanh: SỐNG SÓT, đúng dự đoán ghi trước** — không có
  test service-level cho `prepare()`. Chốt bù là probe §9, và probe đã chứng
  minh nó xứng đáng: bắt được lỗi TracedRetriever mà test không thấy.
* **M2 dạy một phân biệt tôi chưa test**: huỷ ghi `"success"` (thay vì
  `"neutral"`) *reset* bộ đếm lỗi — một client hay đóng tab giữ một provider
  hỏng thật mãi ở `closed`. Test kẹp cạnh ấy được thêm TRƯỚC lượt tiêm, và
  M2 xác nhận nó có răng.

## 11. Quyết định giữ nguyên (có chủ đích, người sau đừng "sửa" lại)

* `admin.py` `_refused` **giữ** chi tiết exception — sau scope `admin`, cho
  người vận hành (§3).
* `BudgetExceeded` trong khung SSE **giữ** nguyên văn — lời của chính mình,
  client cần biết *vì sao* bị từ chối.
* `compare_to_baseline` của smoke vẫn một chiều — chặn regression, không chặn
  improvement (xác nhận lại ở audit là chủ đích, không phải bug).

## 12. Nợ liên quan

Đóng: ~~`TD-64`~~. Giảm nhẹ: `TD-73` (PII đã redact ở biên; *nhãn-không-phải-
hàng-rào* vẫn nguyên). Mới: `TD-84` (12 mục audit chưa vá, mỗi mục kèm chỗ
trả — `AU-12`→`W5-10` · `AU-08/09/10`→`W6-06` · `AU-11/14…17`→`W6-05` · còn
lại nhặt khi chạm file).
