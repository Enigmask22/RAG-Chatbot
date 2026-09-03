# `W4-03` — FastAPI skeleton, và cái giá của việc chỉ chạy test giả

> **Ngày:** 2026-09-03 · **Mã:** `serving/api/{app,health,admin,middleware}.py`,
> `serving/core/{logging,probes,runtime}.py`
> **Test:** `tests/integration/test_health.py` (26) + `tests/unit/test_serving_logging.py` (19)
> **Kiểm thật:** uvicorn thật trên `:8077`, Qdrant **tắt** — xem §7

## 1. DoD, và một nửa của nó cố tình chưa làm

DoD: *"`/ready` chỉ 200 khi bundle + Qdrant + DB sẵn sàng"*.

Bundle và Qdrant: xong. **Postgres thì không**, và không phải vì thiếu thời gian:
schema DB là `W4-05`, nên hiện **không một request nào** của tiến trình này chạm
Postgres. Một phép thử sẵn sàng cho phụ thuộc mà mình không dùng sai theo cả hai
chiều — nó rút pod khỏi load balancer vì một thứ không ảnh hưởng tới request nào,
và nó báo "sẵn sàng" cho một tầng lưu trữ mà ta chưa biết mình cần gì ở đó.

`ReadinessProbes` nhận một dict phép thử, nên chỗ nối cho `W4-05` là **một dòng**.
Hạng mục để `[~]`.

## 2. ⭐⭐ `/health` và `/ready` không phải hai tên của một thứ

Đây là chỗ dễ làm sai nhất, và làm sai không lộ ra cho tới lần sự cố đầu tiên —
lúc đó nó *khuếch đại* sự cố.

| | `/health` (liveness) | `/ready` (readiness) |
|---|---|---|
| câu hỏi | "tiến trình này còn cứu được không?" | "gửi traffic vào đây được chưa?" |
| trả lời sai ⇒ | **khởi động lại container** | rút khỏi load balancer |
| chạm phụ thuộc | **không** | có |

Cho `/health` hỏi Qdrant thì: Qdrant chớp 30 giây → mọi replica trả 503 →
orchestrator **giết toàn bộ chúng cùng lúc**. Việc khởi động lại không chữa được
Qdrant, mà mỗi replica mới mất vài phút nạp 3,3 GB trọng số bge-m3. **Một trục
trặc 30 giây của phụ thuộc trở thành một sự cố nhiều phút của chính mình**, và
thứ gây ra nó là phép thử sức khoẻ.

`test_health_stays_200_while_qdrant_is_down` là test đắt giá nhất hạng mục này.

## 3. ⭐ Khởi động không được chết vì bundle nạp lỗi

Phản xạ là fail-fast, và ở job offline nó đúng. Ở container serving nó sai vì ba
lý do cộng lại: (1) crashloop, và log của pod biến mất trước khi ai kịp đọc;
(2) không còn `/ready` nào để **hỏi vì sao**; (3) không còn
`POST /admin/bundle/reload` — tức cách chữa duy nhất là deploy lại, kể cả khi
nguyên nhân chỉ là gõ nhầm một số phiên bản.

Lên-nhưng-chưa-sẵn-sàng giữ nguyên phần an toàn (load balancer đọc `/ready` nên
không có traffic) mà đổi được cả ba.

⚠️ **Đánh đổi thật, phải vào checklist deploy `W5`:** điều trên chỉ đúng khi
`/ready` *thực sự* được khai làm readiness probe. Quên khai thì mô hình này biến
một pod hỏng thành một pod **nhận traffic và trả 500**.

## 4. Ba quyết định nhỏ, mỗi cái chặn một cách hỏng

### `ReadinessProbes` có TTL — nếu không, chính phép thử là tải

10 replica × mỗi 3 giây = 200 lượt `get_collection`/phút gửi vào một Qdrant *đang
yếu*. Kết quả **hỏng cũng được nhớ** đúng bằng thời gian như kết quả tốt: chỉ nhớ
kết quả tốt nghe an toàn hơn, nhưng nó nghĩa là phụ thuộc bị dội mạnh nhất đúng
lúc nó yếu nhất. Cache có **double-check trong khoá** — thiếu nó thì mọi lượt đến
trước khi lượt đầu kịp ghi cache đều chạy probe, tức đúng lúc đông nhất.

### Middleware là ASGI thuần, không `BaseHTTPMiddleware`

`BaseHTTPMiddleware` gói phản hồi vào memory stream của anyio và **giữ lại** phản
hồi dạng dòng. `W4-06` là `POST /chat` SSE: bug sẽ hiện ra ở đó, cách nguyên nhân
vài hạng mục, và biểu hiện của nó ("token về một cục ở cuối") trông đúng như lỗi
của tầng sinh.

⚠️ Tính chất này **không có test** — và phát hiện ra điều đó có ích cho `W4-06`:
`TestClient` **tự nó đã đệm**. Cùng đoạn mã kiểm, chạy trên một `FastAPI()` trắng
không middleware nào, vẫn gộp cả hai mẩu vào lần `iter_bytes()` đầu. Nên test
viết ở đây sẽ **xanh với cả hai cách cài đặt** — tệ hơn là không có.
`test_the_test_client_cannot_prove_streaming_works` ghim đúng giới hạn ấy, kèm
kết luận: **SSE của `W4-06` phải xác minh bằng uvicorn thật.**

### Route admin là `def`, không `async def`

`registry.activate()` chặn hàng chục giây (nạp trọng số). `async def` thì suốt
khoảng đó vòng lặp sự kiện không chạy gì khác — kể cả `/health` — và
orchestrator đọc đúng điều đó là "tiến trình chết", rồi giết nó **giữa lúc đang
reload**. FastAPI chạy handler đồng bộ trong threadpool; `anyio` sao chép context
nên `request_id` vẫn theo được.

## 5. Log: `logging` chuẩn, không structlog

Kế hoạch ghi "structlog". Lý do đổi đo được: **phần lớn dòng log của một tiến
trình serving không do mã của ta sinh ra** — `uvicorn`, `qdrant-client`, `httpx`,
`sentence-transformers` đều ghi qua `logging` chuẩn. Dùng structlog vẫn phải cấu
hình `logging` chuẩn để bắc cầu chúng sang, và phần cầu nối dài hơn chính cái
formatter nó thay thế. Thứ hạng mục cần là hai điều — mọi dòng là JSON, mỗi dòng
mang `request_id` — và cả hai là một formatter cộng một `ContextVar`.

Hai chi tiết đáng ghi:

* **`ensure_ascii=False` + stream UTF-8.** `json.dumps` mặc định biến mỗi chữ có
  dấu thành `\uXXXX`, tức `grep` trên log thành vô dụng ở một repo mà mọi thông
  báo lỗi đều là tiếng Việt. Bật nó lên thì **trên Windows stderr mặc định không
  phải UTF-8** và dòng đầu tiên có dấu ném `UnicodeEncodeError` *bên trong*
  `logging`, chỗ lỗi bị nuốt thành `--- Logging error ---`. §7 xác nhận đường này
  chạy đúng trên máy thật.
* **Gỡ handler có sẵn.** `uvicorn` tự gọi `dictConfig`; cùng sống thì mỗi dòng ra
  hai lần — một bản JSON, một bản văn xuôi — nên log vừa gấp đôi vừa không parse
  được. `uvicorn.access` bị tắt hẳn (văn xuôi cố định, và nó ghi cả query
  string); dòng access thay thế do middleware sinh, **không** có query string.

## 6. `RuntimeBuilder` thật — và món nợ `W4-02` hoá ra đã trả một nửa

`W4-02` ghi nợ "cache model theo danh tính". Đọc lại `rag_core` thì
`_load_cross_encoder` **đã là** `lru_cache(maxsize=2)`, và embedding cũng cache
tương tự. Hai bundle chỉ khác `top_k`/`rrf_k` dùng chung đúng một bản trọng số,
nên giữ bản trước để rollback thường tốn gần như 0.

⚠️ **Nhưng khoá cache rộng hơn tưởng.** Nó là bộ **bốn** `(model, device,
max_length, dtype)`, mà `max_length` **nằm trong** `RerankComponent`. Hai bundle
khác nhau *chỉ ở* `rerank.max_length` là **hai** model trong cache — 2,2 GB nữa
trên GPU 8 GB đã có 3,3 GB của bge-m3. Phiên pytest của `W2-05` đã OOM đúng theo
tổ hợp đó. Đổi bundle theo hướng ấy lúc đang chạy là cách thật để OOM một tiến
trình đang phục vụ.

Builder hỏi lại ba câu mà manifest nêu và hệ thống trả lời được: **số chiều**
(model nạp lên vs `embedding.dim`), **schema collection** (`verify_schema`, dùng
lại `W2-02`), **số điểm** (vs `index.n_chunks` — không phải phép kiểm toàn vẹn mà
là phép kiểm **danh tính**: các số trong `bundle.eval` được đo trên một index có
đúng bằng ấy chunk).

⚠️ **Câu thứ tư hỏi không được** — `index.fingerprint` chỉ tồn tại trong
`IndexState` của Pipeline Plane, và Serving Plane không được phép có file đó. Một
collection build lại với chunking khác nhưng ra **đúng bằng ấy** chunk qua được
cả ba phép kiểm. → `TD-38`.

## 7. ⭐⭐ Chạy thật, và nó tìm ra một lỗi mà 45 test xanh không thấy

`uv run uvicorn serving.api.app:app --port 8077`, Qdrant **tắt**. Bốn điều xác
nhận được và một lỗi:

1. API lên sau ~20 giây (nạp bge-m3 xong rồi mới hỏng ở `verify_schema`), `/health`
   200, `/ready` 503 kèm lý do. §3 đúng trên máy thật.
2. `QdrantRuntimeBuilder` chạy hết đường với **manifest thật** `0.1.0`: dựng
   provider, qua phép kiểm số chiều, dựng nhánh hybrid + reranker, tới tận mạng.
3. Log JSON ra `stderr` thật của Windows giữ nguyên cả dấu tiếng Việt lẫn emoji:
   `{"level": "WARNING", "msg": "🔓 route /admin/bundle/* CHƯA có xác thực…"}`.
4. Dòng access: `{"http_method": "POST", "http_path": "/admin/bundle/reload",
   "http_status": 503, "duration_ms": 3391.96, "request_id": "c37122a9…"}`.

**Lỗi:** `POST /admin/bundle/reload` trả `500 {"detail": "Lỗi nội bộ"}`. Qdrant
tắt ⇒ `verify_schema` ném `qdrant_client…ResponseHandlingException("timed out")`
— **không** phải `BundleRuntimeError`, **không** phải `ValueError` — nên nó xuyên
qua mọi `except` trong `reload()` và ra ngoài thành 500 của middleware. Người vận
hành nhận đúng thứ vô dụng nhất: không lý do, và **không** có câu quan trọng nhất
là *"bản cũ vẫn đang phục vụ"*.

Mọi test trước đó xanh vì **builder giả chỉ biết ném đúng những lớp lỗi mà mã đã
bắt**. Đó là giới hạn của đồ giả, và nó chỉ lộ ra khi chạy thật.

Sửa: thêm nhánh bắt-tất → **503** (lỗi phân loại được đã bị bắt ở trên, nên tới
đây là "chưa biết", và "chưa biết" thường là hạ tầng — có thể thử lại), và
`detail` mang **tên lớp exception** vì lời của nhiều lỗi hạ tầng là một chuỗi
trần như `"timed out"`, một mình không nói được cái gì hết giờ. Xác nhận lại trên
tiến trình thật:

```
POST /admin/bundle/reload {"version":"0.1.0"}  → HTTP 503
{"detail":"ResponseHandlingException: timed out","still_serving":null,
 "note":"bundle đang phục vụ không bị chạm tới"}
POST /admin/bundle/reload {"version":"9.9.9"}  → HTTP 404
{"detail":"FileNotFoundError: không thấy manifest bundle: bundles\\rag-bundle-v9.9.9",…}
```

## 8. Kiểm rằng test bắt được gì — tiêm lại 6 lỗi

| lỗi tiêm vào | test đỏ |
|---|---|
| `/health` gọi probe | `test_health_stays_200_while_qdrant_is_down` |
| khởi động ném thay vì log | `test_a_bundle_that_fails_to_build_does_not_kill_startup` |
| bỏ double-check trong khoá probe | `test_simultaneous_polls_collapse_into_one_run` |
| formatter bỏ `extra` | 3 test, gồm `…_secret_never_reaches_the_log…` |
| `/ready` probe cả khi chưa có bundle | `test_ready_does_not_probe_qdrant_before_a_bundle_exists` |
| `_store_of` bỏ nhánh `.base` | `test_the_probe_reaches_the_qdrant_store_through_every_wrapper` |
| bỏ nhánh bắt-tất của `reload` | `test_an_unclassified_failure_is_still_a_refusal_not_a_500` |

## 9. 🔓 Còn nợ

* **Xác thực route admin** — ba route đổi được hệ thống đang phục vụ và hiện ai
  gọi cũng được. `test_admin_routes_are_still_open` ghim **lỗ hổng**, không ghim
  hành vi mong muốn: nó sẽ đỏ khi `W4-04` gắn auth, buộc phải xoá một cách có ý
  thức. Phải vào **cùng lúc** với `W4-04`.
* **`TD-38`** — bundle chưa mô tả đủ runtime: `index.fingerprint` không kiểm được
  từ Serving Plane, và `rerank.dtype`/`device` không có trong manifest dù chúng
  đổi kết quả (fp16 vs fp32 trùng top-1 98,3%, `W2-05`).
* **Phép thử Postgres** — vào cùng `W4-05` (§1).
* **`Retriever` ABC không khai `verify_schema`/`fetch_doc_chunks`** dù mọi cài đặt
  đều có; `runtime.py` phải `cast`. Việc của `rag_core`, không phải của `W4-03`.
