# `W3-08` — Worker ingestion: ba đường quay lại hàng đợi, và chỉ một trong số đó tôi đoán đúng

> **Trạng thái:** xong · **Ngày:** 2026-08-22 · **Nhánh:** `feat/w1-foundation`
> **Code:** `pipeline/ingest/` (`app.py`, `worker.py`, `tasks.py`, `store.py`, `schemas.py`)
> **Test:** `tests/integration/test_ingest_job.py` (20)
> **Lệnh:** `make ingest-api` · `make ingest-worker`

## 0. DoD

| vế | đo được |
|---|---|
| `POST /ingest` trả `job_id` < 200 ms | **p50 3,47 ms · p95 4,26 ms · max 7,77 ms** (50 lượt) — dư **47×** ở p95 (§2) |
| `GET /ingest/{id}` có progress | có, và tiến độ đơn điệu tăng theo từng tài liệu (§3) |
| retry khi worker chết | có, nhưng **không như tôi tưởng** — ba đường khác nhau, mỗi đường một cơ chế (§4) |

## 1. Dự đoán ghi trước khi làm

| # | Dự đoán | Kết quả |
|---|---|---|
| E1 | `POST` dưới 200 ms là ràng buộc phải thiết kế để đạt | ⚠️ **nửa đúng** — đạt dư 47×, nhưng chỉ vì endpoint **không làm gì** (§2) |
| E2 | `max_tries = 3` nghĩa là mọi lỗi được thử 3 lần | ❌ **sai** — arq **không** thử lại `Exception` thường (§4) |
| E3 | `doc_ids` là một bộ lọc corpus, dùng lại `select_entries` là xong | ❌ **sai nguy hiểm** — làm vậy thì một `POST` xoá 2/3 index (§5) |
| E4 | Trạng thái job để trong RAM của API là đủ cho bản đầu | ❌ **sai** — API và worker là **hai tiến trình** (§6) |
| E5 | Đặt API ở `serving/` cho đúng tên | ❌ **sai** — nó ghi vào index, tức thuộc pipeline plane (§7) |
| E6 | Test viết xong là chạy | ❌ **sai** — ba lỗi cô lập/thứ tự, mỗi lỗi hỏng theo một kiểu không giống nguyên nhân (§8) |

**0 đúng hẳn, 1 nửa đúng, 5 sai.** Hạng mục nhiều dự đoán sai nhất từ đầu dự án.

## 2. `POST` nhanh vì nó không làm gì — và đó là toàn bộ thiết kế

Đo trên 50 lượt sau khi làm nóng:

| | p50 | p95 | max |
|---|---:|---:|---:|
| `POST /ingest` | **3,47 ms** | 4,26 ms | 7,77 ms |
| `GET /ingest/{id}` | 1,26 ms | 1,59 ms | 1,66 ms |

Endpoint làm đúng ba việc: kiểm tên config, ghi một bản trạng thái vào Redis, đẩy
job. Hai lượt Redis cục bộ.

Ràng buộc thật không phải "viết cho nhanh" mà là **giữ cho nó không bao giờ nặng
lên**. Nên có một test chạy trong **tiến trình con** kiểm rằng
`import pipeline.ingest.app` không kéo theo `torch`/`sentence_transformers`/
`docling`. Không có nó thì một lần thêm `from ..indexing.build_index import ...`
lên đầu file là đủ để thời gian khởi động API đi từ mili giây lên vài giây và
tiến trình giữ thêm 2 GB — mà mọi test khác vẫn xanh.

Thứ tự trong endpoint cũng có chủ ý: **ghi trạng thái trước, đẩy job sau**. Ngược
lại thì worker có thể nhặt job và gọi `patch` vào một bản ghi chưa tồn tại —
`patch` trả `None` và mọi tiến độ của job đó biến mất, im lặng. Đổi lại,
`enqueue_job` hỏng sẽ để lại một bản ghi `queued` mồ côi; đó là cái giá rẻ hơn.

## 3. Tiến độ

`build_index` nhận `on_progress(done, total)`, gọi một lần với `(0, N)` **trước**
khi làm gì và một lần sau mỗi tài liệu. Báo tổng số trước là cần thiết: không có
nó thì thanh tiến độ của một job vừa bắt đầu và của một job đã xong trông giống
hệt nhau (`0/0`), nên `JobStatus.progress` trả `0.0` chứ không phải `1.0` khi
`documents_total == 0`.

Callback chạy **trong thread của `build_index`** (hàm đó đồng bộ và chiếm CPU/GPU
hàng trăm giây, nên nó nằm trong `asyncio.to_thread`). Client Redis async thuộc
về vòng lặp sự kiện; gọi nó từ thread khác làm hỏng dữ liệu trên socket theo kiểu
ngẫu nhiên chứ không hỏng ngay. `_progress_bridge` đi qua
`asyncio.run_coroutine_threadsafe` — và **không** vứt `Future` đi, vì lỗi im lặng
ở đây nghĩa là thanh tiến độ đứng yên mà không ai biết vì sao.

## 4. ⭐ Ba đường quay lại hàng đợi, và `max_tries` không phải cái tôi tưởng

Tôi viết một test để chứng minh `max_tries = 3` hoạt động: xoá manifest, chạy
worker, kỳ vọng `attempt == 2`. Nó đỏ ở `attempt == 1`, và buộc đọc mã nguồn arq.

`arq/worker.py:613-633` chỉ thử lại với `Retry`, `RetryJob` và
`asyncio.CancelledError`. Một `Exception` bất kỳ ⇒ `finish = True`, hỏng hẳn ngay
lần đầu, `max_tries` không đụng tới.

Hoá ra đó là mặc định **đúng**, và việc phải tự phân loại là một cải thiện chứ
không phải một chỗ vá: thử lại một job hỏng vì thiếu config chỉ cho ba lần hỏng y
hệt nhau, ba lần log, và một hàng đợi che mất lỗi thật.

Bảng đầy đủ, mỗi dòng có test riêng:

| chuyện gì xảy ra | cơ chế | mất bao lâu để nhận ra |
|---|---|---|
| lỗi hạ tầng (Qdrant sập) | `tasks.is_transient` ⇒ ném `Retry` | ngay |
| worker tắt êm (Ctrl-C) | `CancelledError` ⇒ arq tự thử lại | ngay |
| worker **chết hẳn** | khoá `in-progress` hết hạn | **`job_timeout + 10 s`** |
| lỗi tất định (sai config) | hỏng hẳn, **không** thử lại | ngay |

`is_transient` nhận diện theo **tên lớp** chứ không `isinstance`: bắt đúng kiểu
của `qdrant_client` sẽ buộc `tasks.py` import nó, mà cả điểm của module này là
không kéo phụ thuộc nặng vào tiến trình chưa cần. Nó cũng đi hết chuỗi
`__cause__`/`__context__`, vì lỗi kết nối hầu như luôn bị bọc lại.

Test cho dòng thứ nhất dùng một **cổng đóng thật** (`127.0.0.1:59999`), không
phải exception dựng sẵn: `attempt` đi tới đúng `MAX_TRIES` rồi mới `failed`.

### Chỗ **không** được kiểm, nói thẳng

Dòng "worker chết hẳn" được xác minh bằng **đọc mã nguồn**, không bằng giết tiến
trình: job ở lại hàng đợi trong lúc chạy, worker giữ `in-progress:{job_id}` hạn
`job_timeout + 10 s` (`arq/worker.py:450-465`), và `job_try` được `INCR` ở **mỗi**
lần nhặt (`:482`). Nửa của **mình** — `job_try > 1` phải đi vào
`JobStatus.attempt` và phải được log — thì có test, dựng lại bằng cách đặt trước
đúng khoá đếm mà arq dùng. → `TD-29`.

## 5. ⚠️ `doc_ids` suýt thành một endpoint xoá index

Phản xạ đầu: `doc_ids` là một bộ lọc corpus, dùng lại `select_entries` là xong.

Nhưng `build_index` **xoá** mọi tài liệu có trong state mà không thấy trong lượt
chạy — đúng hành vi của nó, vì bộ lọc corpus (`languages`/`doc_types`/
`max_documents`) nói *tài liệu nào THUỘC index*. Còn "index lại đúng tài liệu
này" là một câu hoàn toàn khác. Diễn đạt nó bằng bộ lọc thì
`POST /ingest {"doc_ids": ["d-1"]}` xoá 59 tài liệu kia, im lặng, và trả `200`.

Nên `only_doc_ids` là **phạm vi của một lượt chạy**: nó không đi vào
`fingerprint` (nó không đổi *index này là gì*) và nó **tắt** bước gỡ tài liệu
vắng mặt. Có test đi tìm đúng chế độ hỏng ấy: index 3 tài liệu, index lại 1, rồi
kiểm cả 3 vẫn còn trong collection.

## 6. Trạng thái phải sống ngoài tiến trình

Một `dict` trong `app.state` sẽ qua **mọi** test một tiến trình rồi hỏng ở lần
deploy đầu, vì API và worker là hai tiến trình. Nên trạng thái ở Redis, TTL 24
giờ, và có một test dựng hẳn một app **thứ hai** để đọc job do app thứ nhất tạo —
test mà một `dict` không qua được.

## 7. Vì sao `pipeline/ingest/` chứ không `serving/`

`serving/` là Serving Plane: đường phục vụ truy vấn, nối với Pipeline Plane chỉ
qua một `RagBundle` bất biến có version. Một endpoint **ghi vào index** không
thuộc về đó — nó là điều khiển từ xa của chính pipeline. Đặt nhầm chỗ thì ranh
giới mà cả kiến trúc dựa vào bị nhoè ngay ở hạng mục **đầu tiên** chạm tới HTTP,
và `W4` sẽ thừa hưởng chỗ nhoè đó.

## 8. ⚠️ Ba lỗi test, và cả ba hỏng theo kiểu không giống nguyên nhân

* **`Annotated[...]` khai trong hàm** → FastAPI trả `422 Field required: store`.
  `from __future__ import annotations` biến annotation thành chuỗi và FastAPI
  phân giải chúng ở **không gian tên module**, nên alias cục bộ không tìm thấy và
  tham số bị coi là **query param**. Không lỗi import, không gợi ý.
* **Test dùng chung hàng đợi Redis** → worker của test này nhặt job của test kia,
  hỏng vì không thấy config, và cả hai đỏ vì một lý do không liên quan. Sửa bằng
  `INGEST_QUEUE` cấu hình được — thứ hai môi trường dùng chung một Redis cũng cần,
  nên nó không phải một cái nạng cho test.
* **Thứ tự fixture** → `client` được dựng trước `workspace`, app nối vào hàng đợi
  **mặc định**, mọi job nằm im ở `queued`. Sửa bằng cách khai phụ thuộc tường minh
  dù không dùng giá trị, kèm lý do trong docstring.

## 9. Cái cố ý **không** làm

* **Không nhận upload file.** `POST /ingest` nhận **tên config**, không nhận tài
  liệu. Nhận upload là mở một đường vòng qua `LICENSE_ALLOWLIST` (Quy tắc cứng
  #3) và buộc serving plane ghi vào manifest — hai thứ đều đắt hơn nhiều so với
  cái nó mua.
* **`max_jobs = 1`.** Hai job cùng lúc là hai bản BGE-M3 (2,2 GB mỗi bản) tranh
  VRAM. Song song mua được rất ít vì công việc vốn đã bão hoà GPU.
* **Không có endpoint huỷ job.** arq có `abort`; chưa có ai cần, và một nút huỷ
  chưa được kiểm là một nút không dùng được.

## 10. Còn lại gì

* `TD-29`: đường "worker chết hẳn" chưa được kiểm bằng cách giết tiến trình thật.
* API chưa có xác thực. Chấp nhận được khi nó bind `127.0.0.1`, **không** chấp
  nhận được khi `W4-13` đóng gói Docker — phải vào cùng lúc với việc mở cổng.
* `job_timeout = 2 giờ` nghĩa là phục hồi sau khi worker chết mất tới 2 giờ mới
  bắt đầu. Đổi được bằng heartbeat riêng, nhưng đó là thêm một cơ chế; hôm nay
  ghi lại con số còn hơn giả vờ nó nhỏ.
