# `W5-06` — Langfuse tự dựng, và con số 725 ms hoá ra là hai con số

> **DoD**: 1 query hiện đủ span trong Langfuse UI, có cost · **Test**:
> `tests/integration/test_tracing.py` (assert số span) · **Evidence**: trace thật
>
> **Kết quả**: đủ · 7 span cho một lượt `/chat` · `$0,001299` một câu · toàn bộ
> hạng mục tốn **$0,009161** — 7 trace thật, trong đó 6 tới được model.
>
> ⚠️ **Không có ảnh chụp màn hình.** DoD viết *"Evidence: screenshot trace"*;
> thứ ở đây là bản đọc lại qua `GET /api/public/traces/{id}`. Nó mạnh hơn một
> ảnh (kiểm được bằng máy, không cắt xén được, và một bài test đối chiếu với
> chính endpoint ấy), nhưng nó **không** chứng minh giao diện Langfuse vẽ đúng
> cây — Langfuse UI đọc cùng API này, nên rủi ro còn lại là rủi ro của Langfuse.

---

## 0. Một lượt thật, đọc lại từ chính Langfuse

Không phải ảnh chụp màn hình — đọc lại bằng `GET /api/public/traces/{id}`, thứ
kiểm được bằng máy và không cắt xén được:

```
trace  7d6453c9055b4023803ac2c3065c8de3   name=chat
       session=2ec487b0fc80…  user=public  tags=[tenant:public]
       cost=$0.001299        latency=9.892 s   span_count=9

SPAN               TYPE              ms   chi tiết
cache.lookup       SPAN         416.424   hit=false
  embed.query      SPAN         413.349   bkai-vietnamese-bi-encoder
understand         SPAN           0.074   route=retrieve · language=vi · rewritten=false
retrieval          SPAN          7398.4   n_hits=5 · truncated=false
  retrieve.hybrid  SPAN          44.362   n_hits=50
  rerank           SPAN        7353.524   n_candidates=50 · n_scores=50
prompt             SPAN           0.070   chat-system@v2 · 5 chunk · 9 392 ký tự
completion         GENERATION  2035.369   deepseek-v4-flash  in=4199 out=197  $0.001299
citations          SPAN           0.029   block=ok
```

Bằng chứng đầy đủ 7 trace: [`runs/w5-06-trace-spans.json`](../runs/w5-06-trace-spans.json).

---

## 1. ⭐⭐ "Truy hồi 725 ms" là **44 ms tìm + 685 ms xếp lại**

`W4-13` đo được truy hồi tốn 725 ms và ghi nó là *một* con số. Nó là hai, và tỉ
lệ giữa chúng đảo ngược việc nên đi tối ưu cái gì:

| bước | p50 (đã nóng, n=5) | phần của truy hồi |
|---|---:|---:|
| `retrieve.hybrid` — Qdrant dense+sparse+RRF, 50 ứng viên | **44,8 ms** | 6,1 % |
| `rerank` — `bge-reranker-v2-m3` chấm 50 ứng viên trên GPU | **685,3 ms** | 92,8 % |
| `retrieval` (tổng) | 738,1 ms | 100 % |

Cả hạng mục `W2` (hybrid, RRF, `candidate_k`, trọng số nhánh) tối ưu đúng
**6 %** ngân sách truy hồi. Cross-encoder chiếm phần còn lại, và trước
hôm nay không có phép đo nào tách được nó ra — vì mã chỉ có **một** lời gọi
`retriever.retrieve()` và một đồng hồ quanh nó.

Cùng họ lỗi với `p95_latency_ms` / `p95_end_to_end_ms` của `W5-05`: một cái tên
gộp hai đại lượng, và cái gộp luôn nghiêng về phía dễ chịu hơn.

**Hệ quả cho `W6-05`**: mục tiêu p95 ≤ 3500 ms end-to-end không đạt được bằng
cách chỉnh Qdrant. Đòn bẩy là `DEFAULT_RERANK_CANDIDATES = 50` — và giờ có một
con số để đánh đổi nó với `ndcg@10`, thay vì một cảm giác.

---

## 2. ⭐⭐ Lần chạy đầu sau deploy tốn **10,7×** thời gian rerank

| | `rerank` | `retrieval` | end-to-end |
|---|---:|---:|---:|
| request **đầu tiên** sau khi container lên | **7 353 ms** | 7 398 ms | 9,89 s |
| p50 đã nóng (n=5) | 685 ms | 738 ms | 3,95 s |

Trọng số cross-encoder nạp lúc `activate()`, nhưng kernel CUDA và bộ nhớ đệm
của nó chỉ khởi tạo ở lời gọi `score()` **đầu tiên**. `/ready` xanh trước đó,
nên load balancer đã gửi traffic vào.

Đây là một chế độ hỏng không quan sát được trước hạng mục này: nó xảy ra đúng
**một lần** mỗi lần deploy, và một p95 trên hàng nghìn request pha loãng nó tới
mức vô hình. Trace thì không pha loãng gì cả — nó là một dòng, và dòng ấy đọc
được ngay. → `TD-72`

---

## 3. ⭐⭐ Trace hữu ích nhất trong bảy trace là trace của một request **hỏng**

Trong lúc dựng, một lượt trả `503` vì `psycopg` không chạy được trên
`ProactorEventLoop`. Cái mà log để lại là một dòng `InterfaceError`. Cái mà
trace để lại:

```
cache.lookup   485 ms      ✔
retrieval     8245 ms      ✔  (retrieve.hybrid 76 · rerank 8168)
—— rồi chết ——             level=ERROR  status="503 InterfaceError: …"
prepare_ms     null        (không tới được dòng gán)
cost           null
```

Nó nói ra được rằng request đã đi hết truy hồi — tức đã tiêu 8,2 giây GPU — rồi
mới chết ở Postgres. Không có trace thì "503" là toàn bộ thông tin.

Điều làm nó tồn tại là một quyết định nhỏ: **trace mở ở `api/chat.py`, trước
`prepare()`**, chứ không mở cùng `ChatTurn`. `ChatTurn` chỉ ra đời ở *dòng cuối*
của `prepare()`, nên nếu trace sinh ra cùng nó thì mọi lượt 404/403/429/503 đều
không có trace — và hệ quan sát sẽ phủ **đúng những request đã chạy trót lọt**.
Một hệ thống như thế không im lặng; nó nói rằng mọi thứ đều ổn.

Cái giá: trace được đóng ở **hai** chỗ (`except` của handler, `finally` của
`stream_turn`), nên `Trace.finish()` phải idempotent — có test, và có phép tiêm
`T7`.

---

## 4. ⭐⭐ Cây span chỉ mịn được tới đúng chỗ mã có mối nối

DoD đòi `rewrite → retrieve kèm score → rerank → prompt → completion`. Nhìn từ
`prepare()`, truy hồi là **một** lời gọi trong một `asyncio.to_thread`. Một lời
gọi, một thời lượng.

Có hai lối để lấy hai span từ đó. Lối thứ nhất là bổ đôi con số ấy theo một tỉ
lệ nào đó — nó cho ra hai con số **trông như** hai phép đo, và không ai đọc
trace biết được rằng chúng là một phép đo bị chia. Lối thứ hai là đi tìm mối
nối thật.

Mối nối có sẵn: `RerankedRetriever` giữ `base` và `reranker` là hai thuộc tính
công khai, và thân `retrieve()` của nó đúng là `base.retrieve()` rồi
`reranker.score()`. Nên mỗi lớp được bọc riêng và mỗi span đo `perf_counter`
quanh đúng lời gọi của lớp ấy. Không có phép chia nào — và đó là lý do bảng ở
§1 là một phép đo.

**Hệ quả phải nói ra: không có span `embed` ở đường truy hồi.**
`QdrantHybridRetriever.retrieve` gọi `embed_query_hybrid` trong *thân hàm*, nên
tách nó ra đòi sửa `rag_core` — thứ cố ý không biết gì về quan sát. Thời gian
embed nằm trong `retrieve.hybrid` (44,8 ms tổng cho cả embed + hai nhánh + RRF,
nên nó nhỏ). Đường **cache** thì có `embed.query` thật, vì ở đó `prepare()` tự
gọi embedder — và nó đo được 413 ms lúc lạnh, 13 ms lúc nóng.

---

## 5. ⭐⭐ Ghi nguyên prompt vào trace là mang `nonce` của `W4-12` sang một hệ thống khác

`W4-12` dựng hàng rào tiêm bằng một điều duy nhất không giả được: 16 ký tự hex
sinh ra **sau** khi tài liệu đã nằm trong index. `TD-53` ghi rằng đó là lớp
*duy nhất* không phụ thuộc vào việc model có hợp tác hay không.

Span `prompt` mang nguyên văn thứ đã gửi cho model, nên nó cũng mang mã ấy. Và
DoD của chính hạng mục này là *"Evidence: screenshot trace"* — tức trace là
loại dữ liệu được chụp và dán đi.

Mã hết hiệu lực khi lượt kết thúc, nên rò một mã cũ không mở được cửa nào. Che
nó vẫn đúng, vì **che không tốn gì**: người đọc trace cần biết khối ngữ cảnh
*có được bọc hay không*, không cần biết bọc bằng chuỗi nào.

Đo trên 7 trace thật: **0** chuỗi 16-hex lọt ra, 9 dấu `«nonce»` mỗi trace.

⚠️ Bộ lọc theo **hình dạng** (`(?<![0-9a-f])[0-9a-f]{16}(?![0-9a-f])`), nên nó
cũng che một chuỗi hex 16 ký tự tình cờ trong tài liệu. Hướng nhầm ấy là hướng
đúng.

---

## 6. ⭐ `TD-55` trả xong, và con số ước lượng đúng

`TD-55` nói khung `done` khai thiếu ~18% thời gian thật, vì `total_ms` đếm từ
`ChatTurn.started` — thứ được đặt **sau** embed + truy hồi + rerank.

Trace bắt đầu ở handler HTTP nên nó đo được cả phần ấy:

```
p50 prepare_ms   787,3 ms
p50 latency      3 953  ms
phần khung `done` không thấy   19,92 %
```

Ước lượng 18% của `TD-55` đúng. Cái đổi là nó thôi là một ước lượng: mỗi trace
mang `prepare_ms` của chính nó, và khung `done` giờ khai nó ra để hai đồng hồ
không còn lệch một cách im lặng.

---

## 7. ⭐ Một hệ quan sát không được phép giết thứ nó quan sát

Hai chế độ hỏng, và cả hai là chế độ hỏng của **hàng đợi**, không phải của mạng:

* **Chặn** — đẩy đồng bộ ở cuối `stream_turn` thì Langfuse chậm 2 giây là mọi
  câu trả lời chậm thêm 2 giây, và Langfuse chậm đúng lúc hệ thống đang tải cao.
* **Phình** — hàng đợi không trần + Langfuse chết = mọi trace nằm lại trong RAM
  cho tới khi tiến trình OOM. Một `/chat` chết vì bộ đếm span của chính nó.

Nên: một thread nền, `queue.Queue(maxsize=256)`, **vứt trace mới khi đầy** kèm
bộ đếm. Vứt cái *mới* chứ không vứt cái *cũ*: lúc nghẽn, trace đã chờ lâu nhất
là trace gần nhất với sự cố đang diễn ra.

`GET /admin/tracing` khai `queued/sent/failed/dropped`. Không có nó thì quan sát
là thứ duy nhất trong hệ thống không quan sát được: hàng đợi đầy, khoá sai, host
sai — cả ba trông y hệt nhau từ phía `/chat` (200, câu trả lời đúng, bảng trống).

---

## 8. ⭐ "Chưa đo" không phải "miễn phí"

`Usage` khai `int | None`, và `Trace.total_cost_usd()` **bỏ qua** bước không có
số thay vì cộng 0. Kèm `unmeasured_cost_steps()` — danh sách các bước gọi model
mà không khai được chi phí. Một tổng có danh sách ấy không rỗng thì chưa phải
một tổng.

Chỗ này có một khách hàng thật: `QueryUnderstanding._rewrite` bọc lời gọi trong
`asyncio.wait_for`, thứ huỷ được cái *chờ* nhưng không huỷ được cái *thread*.
Quá hạn ⇒ request đi tiếp bằng câu gốc **trong khi lời gọi kia vẫn chạy nốt và
vẫn bị tính tiền**. Chi phí thật của bước đó là *chưa biết*; ghi `0.0` biến một
khoản chi không quan sát được thành một khoản chi bằng không.

Cùng bài học `W5-04` đã trả giá: faithfulness `1,0000` vì 32 phán quyết mất
được lặng lẽ bỏ khỏi mẫu số.

Tương tự, một **cache hit** không khai `usage`: nó không tốn token, nhưng nó
cũng không phải "$0 cho câu hỏi này" — câu trả lời ấy đã trả tiền một lần ở lượt
trước. Ghi 0 làm bảng chi phí tụt theo tỉ lệ trúng cache thay vì theo giá thật.

---

## 9. Ba lỗi thật gặp trong lúc dựng

**⭐⭐ Proxy uỷ quyền tự đệ quy tới chết khi bị sao chép.** `TracedRetriever`
uỷ quyền bằng `return getattr(self._inner, item)`. `copy.copy` dựng thực thể mới
**không qua `__init__`** rồi tra `__setstate__`; `_inner` chưa có trong
`__dict__` nên `__getattr__("_inner")` gọi lại chính nó — 961 tầng rồi
`RecursionError`. Chỗ gọi `copy.copy` là chính `instrument_retriever`, nên lỗi
nổ đúng khi bọc một chuỗi **đã bọc**, tức đúng kịch bản mà lối "sao chép thay vì
sửa tại chỗ" được chọn để phục vụ. Chữa bằng `object.__getattribute__` (không
rơi vào `__getattr__`). Tìm ra bởi một test viết cho phép tiêm `I2`, không bởi
một lượt chạy.

**⭐ YAML nuốt khoá mã hoá.** `ENCRYPTION_KEY: 0000…0` (64 số 0, không nháy) bị
YAML đọc là **số**, chuẩn hoá thành `0`, và Langfuse chết với *"must be 256
bits, 64 string characters"*. Cấu hình nhìn đúng trên màn hình, sai trong
container — và chỉ hỏng với những khoá toàn chữ số.

**⭐ `start_period: 180s` biến một lỗi cấu hình chết ngay thành một lần khởi
động chậm.** Trong `start_period`, Docker báo `health: starting` chứ không báo
`unhealthy`, nên `up --wait` ngồi đợi đủ ba phút trước khi nói bất cứ điều gì.
Gặp **hai lần** ở đúng file này (`ENCRYPTION_KEY`, rồi
`LANGFUSE_INIT_USER_EMAIL: dev@localhost` bị Zod từ chối vì thiếu TLD). Cách gỡ
là đọc `make logs-langfuse`, không đọc cột `STATUS`.

Và một chi tiết đã được dự án ghi sẵn mà tôi vẫn đâm vào: `uvicorn
serving.api.app:app` **không** chạy được `psycopg` trên Windows. `serving/__main__.py`
tồn tại đúng vì lý do đó, và docstring của nó ghi rõ rằng đổi
`event_loop_policy` **không đủ cho uvicorn** (uvicorn trả thẳng một
`loop_factory`). Nhưng nó **đủ** cho `TestClient`, thứ dựng vòng lặp qua
`anyio.start_blocking_portal` → `asyncio.run`. Hai đường vào, hai cách chữa cho
cùng một lỗi — đo được cả hai chứ không suy từ một cái ra cái kia.

---

## 10. Quyết định thiết kế

**httpx thẳng vào `/api/public/ingestion`, không dùng SDK `langfuse`.** Cùng lý
lẽ đã viết cho `rag_core/llm/openai_compat.py`: cần đúng một endpoint, và cần
biết chính xác byte nào rời khỏi tiến trình. SDK hiện đại kéo theo cả
OpenTelemetry và một bộ chụp tự động — mà "chụp tự động" ở đây nghĩa là chụp cả
`nonce` (§5). Đổi lại, bộ mã hoá viết tay có thể sai schema và vẫn nhận `207`,
nên có một bài test đối chiếu với **server thật** rồi đọc lại `totalCost`.

**Cây span dựng trong bộ nhớ, sink là một `Protocol`.** Nhờ vậy phép đếm span
của DoD chạy được **không cần hạ tầng** — 44 unit test, 0 container. Một phép
kiểm chỉ chạy khi có Docker là một phép kiểm sẽ không chạy trong `W5-09`.

**`ChatTurn.trace` khai kiểu `Trace`, không `Trace | None`.** Một `Optional` ở
đây sinh ra khoảng một tá `if trace is not None` rải khắp hai nửa của lượt, và
mỗi cái là một chỗ để một span biến mất khi ai đó quên. Không thu trace thì vẫn
dựng cây, chỉ là nó rơi vào GC.

**Cha–con đi bằng `ContextVar`, và `stream_turn` **không** dùng nó.** Truy hồi
chạy trong `asyncio.to_thread`, thứ sao chép context — nên `ContextVar` là cách
duy nhất nối span qua ranh giới thread khi chữ ký `Retriever.retrieve` là hợp
đồng của `rag_core`. Nhưng một async generator chạy trong context của người
*tiêu thụ* nó, nên một `with trace.span(...)` bọc quanh một `yield` sẽ để biến
cha ở trạng thái đặt trong suốt thời gian sinh chữ. Nửa dưới vì thế mở/đóng span
tường minh.

**Bọc bằng `copy.copy`, không sửa runtime tại chỗ.** `ActiveBundle` là ảnh chụp
bất biến có thể đang được nhiều request cầm (`W4-02`).

**Langfuse ở một compose project riêng.** Qdrant/Postgres/Redis là thứ `/chat`
không chạy được nếu thiếu; Langfuse là thứ `/chat` chạy y hệt khi thiếu. Trộn
chúng nghĩa là `make down` gỡ luôn quan sát, và mỗi lần chạy test tích hợp phải
đánh thức ClickHouse + MinIO. Khoá API **khai trước** bằng `LANGFUSE_INIT_*`:
một hạ tầng mà bước cuối là "vào UI bấm New API key" là một hạ tầng `W5-09`
không dựng lại được.

---

## 11. Test và tiêm lỗi

| | số bài |
|---|---:|
| `tests/unit/test_tracing.py` | 46 |
| `tests/unit/test_query_understanding.py` (span `rewrite`) | +4 |
| `tests/integration/test_tracing.py` | 22 |
| **bộ mặc định** | **2 157** xanh, 3 skip (+50) |
| **bộ integration** | **254** xanh, 0 hỏng |

⭐ Bộ integration chạy **hết** lần này (7 phút 46) — nó là thứ `W5-05` phải ghi
là *chưa kiểm lại được* vì Docker Desktop tắt. Cảnh báo ấy giờ đóng.

Bộ integration đi qua **app thật + Postgres thật**, với một chuỗi truy hồi giả
mang **đúng hình dạng** của chuỗi thật (`.base` + `.reranker`), nên
`instrument_retriever` được chạy bằng đúng nhánh mã mà production đi qua. Hai
bài cuối đối chiếu với **Langfuse thật** và bỏ qua khi chưa `make up-langfuse`.

Phép đếm span của DoD viết dạng **bộ đầy đủ**, không phải "ít nhất": một span
mới xuất hiện mà không ai cập nhật bài test ấy là một span không ai quyết định
thêm.

**Tiêm lỗi: 26 phép, 26 đỏ** — nhưng **không** ngay lần đầu. Lượt một có **ba
phép sống sót**, và cả ba là lỗ hổng thật:

| phép | vì sao sống sót | đã bịt bằng |
|---|---|---|
| `L2` bỏ `parentObservationId` khỏi gói tin | mọi bài đều kiểm cây **trong bộ nhớ**; không bài nào kiểm rằng cây ấy còn nguyên sau khi mã hoá. Mất trường ấy thì Langfuse nhận đủ số quan sát, trả `207`, và vẽ một danh sách **phẳng** | `test_nesting_survives_the_encoding` |
| `C4` span `rewrite` khai `cost = 0` khi quá hạn | span `rewrite` **không có bài test nào** — nó nằm trong `understanding.py`, ngoài file test của hạng mục | 4 bài `TestRewriteSpan` |
| `C1` bỏ `trace.finish()` ở nhánh 404 | không sống sót — chuỗi tìm kiếm của tôi lỗi thời sau `ruff format` | (sửa chuỗi, đỏ ngay) |

Hai lỗ hổng thật ấy có cùng một hình dạng: **bài test dừng lại ở ranh giới của
module đang được viết.** Cây span đúng tới lúc nó rời `tracing.py`; span
`rewrite` không được kiểm vì nó sống ở file khác.

---

## 12. Nợ mới

| mã | nợ |
|---|---|
| `TD-72` | **Rerank lạnh tốn 10,7× rerank nóng** (7 353 ms vs 685 ms), và `/ready` xanh trước lời gọi `score()` đầu tiên — nên request đầu sau mỗi lần deploy nhận 9,9 s. Cần một lượt làm nóng ở `lifespan`, và `/ready` phải đợi nó. Chưa làm vì nó kéo dài thời gian khởi động container và đó là một đánh đổi thuộc `W6-05` |
| `TD-73` | **Trace không mang `tenant` vào Langfuse như một hàng rào, chỉ như một nhãn.** Mọi tenant ghi vào **một** project Langfuse; ai đọc được project ấy đọc được câu hỏi của mọi khách hàng. Đủ cho một corpus công khai (quy tắc cứng #3), không đủ cho dữ liệu thật. Lối ra: một project Langfuse mỗi tenant, hoặc tắt `input`/`output` khi tenant ≠ `public` |
| `TD-74` | **Prompt trong trace bị cắt ở 4 000 ký tự** trong khi prompt thật là 9 392. Người gỡ lỗi một câu trả lời sai không đọc được nửa sau ngữ cảnh. Trần này bảo vệ bộ nhớ và băng thông; đúng lối ra là cắt theo **chunk** (giữ đủ 5 khối, mỗi khối cắt riêng) chứ không cắt theo chuỗi ghép |
