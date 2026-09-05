# `W5-08` — Vòng phản hồi: 👍/👎 → Postgres → điểm Langfuse → ứng viên golden set

**Ngày**: 2026-09-05 · **Chi phí**: ~$0,006 (4 lượt sinh thật; **$0,001419** là
phần đo được, đọc lại từ Langfuse) · **Trạng thái**: xong

Đây là hạng mục đầu tiên mà tín hiệu chảy **ngược**: mọi thứ khác đi từ corpus
ra câu trả lời, cái này đi từ người dùng trở lại tập đo.

---

## 0. Lượt chạy thật, đọc lại từ ba nơi khác nhau

4 câu qua server thật (bundle `0.2.1`, `deepseek-v4-flash`), chấm, rồi đọc lại
từ **Langfuse**, từ **`/admin/feedback`**, và từ **file JSONL**.

| | | |
|---|---|---|
| lượt | 4 (3 trúng cache, 1 sinh thật 3 583 ms) | `finish_reason` = `cache`×3, `stop` |
| chấm | 5 lần bấm (4 câu + 1 lần đổi ý) | `201` cả 5 |
| hàng Postgres | **4** | `UNIQUE (tenant, message)` nuốt lần đổi ý |
| điểm trong Langfuse | **4** / 5 lần gửi | `score_id()` tất định — lần thứ 5 ghi đè |
| hàng đợi review | **3** 👎 (4 nếu tính cả 👍) | `GET /admin/feedback` |
| ứng viên xuất ra | **3** | `X-Candidate-Count: 3` |
| `dropped` | **0** | `GET /admin/tracing` |

`GET /api/public/traces/{id}` trả về điểm **nằm trong trace**:

```
e0c1ff03  spans=4  cost=None       scores=[('user_feedback', -1)]
2c5eafd9  spans=4  cost=None       scores=[('user_feedback',  1)]
24da24f9  spans=4  cost=None       scores=[('user_feedback', -1)]
846de832  spans=9  cost=0.001419   scores=[('user_feedback', -1)]
```

⭐ Ba trace đầu có `cost=None`, không phải `cost=0` — chúng là lượt phát lại từ
cache, và `W5-06` đã quyết rằng một câu trả lời đã trả tiền ở lượt trước thì
**không** được ghi $0 vào lượt này. Hệ quả cho hàng đợi review: một câu 👎 vào
câu trả lời từ cache **không quy được chi phí**, và ô ấy phải trống chứ không
phải bằng 0.

Bằng chứng: `plans/reports/runs/w5-08-feedback-loop.json`.

---

## 1. ⭐⭐ Khoá nối không được đến từ người gọi

Điểm số Langfuse gắn theo `traceId`. Cách hiển nhiên là để client gửi kèm
`trace_id` — nó **đã có sẵn** trong khung `meta` mà tôi vừa thêm ở hạng mục
này. Một dòng, không ai phản đối trong review.

Và nó mở đúng lỗ mà `TD-73` đã ghi: **tenant trong Langfuse là một nhãn, không
phải một hàng rào**. Bên ấy không kiểm gì cả, nên một tenant gửi `trace_id` của
tenant khác sẽ gắn 👎 lên trace của người khác — và hàng đợi review của họ nhận
rác mà không có một dòng log nào nói ra, vì về phía chúng ta mọi request đều
hợp lệ.

Nên endpoint **không nhận** `trace_id`. Nó nhận `message_id`, đọc hàng ấy qua
`atenant_session` (RLS lọc theo tenant của token), rồi lấy `trace_id` **từ hàng
đó** — cột `message.trace_id` mới của `0004`. Khoá nối trở thành một thứ người
gọi phải *chứng minh sở hữu* thay vì một thứ họ *khai*.

Đây là lần thứ tư cùng một lý lẽ trong dự án, và lần này nó chặn một chiều
khác:

| | chặn gì | cơ chế |
|---|---|---|
| `W2-06` → `W4-04` | đọc chéo tenant qua filter | `tenant_filter()` ghi đè |
| `W4-05` | đọc/ghi chéo tenant trong SQL | `FORCE ROW LEVEL SECURITY` |
| **`W5-08`** | **ghi chéo tenant sang một hệ thống KHÁC** | khoá suy ra từ hàng đã qua RLS |

Hai bài ghim nó: `test_a_tenant_cannot_rate_another_tenants_answer` (RLS thật,
Postgres thật — trả **404**, không 403: một 403 xác nhận rằng id ấy có thật) và
`test_the_endpoint_refuses_a_body_that_carries_a_trace_id` (`extra="forbid"` ⇒
422, không im lặng bỏ qua — im lặng nghĩa là người tích hợp tin rằng trường ấy
có tác dụng).

---

## 2. ⭐⭐ Cột `citations` của `0001` chưa bao giờ chứa citations

`serving/core/chat.py` mở đầu bằng một cảnh báo tôi tự viết ở `W4-06`:

> Đó là lý do khung ở đây tên là `sources` (cái đã đưa cho model) chứ không
> phải `citations` (cái đã kiểm) — hai thứ khác nhau, và **gộp tên chúng lại**
> là cách chắc chắn để `W4-09` trở nên vô hình với client.

Rồi `_save()`, cách đó một nghìn dòng trong cùng file, ghi:

```python
citations=turn.sources()
```

Đúng cái bẫy vừa mô tả. Không có gì bắt được, vì một cột JSONB nhận mọi hình
dạng và cả hai thứ đều là "list of dict có `chunk_id`".

Hệ quả không phải thẩm mỹ: `W4-09` xác minh từng quote rồi phát kết quả ra
khung SSE, và **kết quả ấy chưa từng được ghi xuống**. Một câu trả lời đọc lại
từ lịch sử trông y hệt nhau dù mọi citation của nó là thật hay bịa. Đó là nội
dung thật của `TD-50`, và nó đóng ở đây vì phân biệt *"model bịa nguồn"* với
*"truy hồi lấy nhầm tài liệu"* là câu hỏi **đầu tiên** người review một câu 👎
phải trả lời.

`0004` tách hai cột: `citations` → `retrieved_sources`, thêm
`citations_verified`. Không backfill, và lần này vì lý do mạnh hơn `0002`/`0003`
— ở đó backfill là *đoán*, ở đây nó **bất khả**: xác minh cần nguyên văn chunk
tại thời điểm trả lời, mà chunk ấy có thể đã bị reindex.

💡 Bài học chung: **một cảnh báo viết trong docstring không phải là một hàng
rào.** Nó không chạy. Cùng file, cùng tác giả, cùng tuần — vẫn đi vào đúng cái
bẫy được mô tả ở dòng 30. Cái chặn được là kiểu dữ liệu hoặc một phép kiểm, và
ở đây phải là **tên cột**, vì kiểu thì giống nhau.

---

## 3. ⭐⭐ Lỗi thật, tìm ra bởi lượt chạy — và nó chế ra đúng triệu chứng cần tìm

Lượt chạy đầu cho ra một file ứng viên như thế này:

```
fb-00d99bdc3ecb  wrong  | retrieved: []  | cited: ['wb-676…::00019', …3 chunk]
```

**3 citation, 0 chunk được truy hồi.** Tức một citation trỏ vào tài liệu chưa
từng được đưa cho model — thứ trông y hệt một citation bịa, và là chính xác thứ
mà người review mở file này ra để tìm.

Nguyên nhân: cả ba lượt ấy **trúng cache**. Một lượt trúng cache không chạy
truy hồi, nên `contexts` rỗng và `sources()` trả `[]`. Khung SSE thì vẫn đầy đủ
— nó phát `cached.sources`. Nên từ `W4-10` tới giờ, mỗi lần trúng cache là hàng
Postgres mất sạch nguồn **trong khi client nhìn thấy chúng**.

Vô hình suốt bốn hạng mục, vì `W4-10` chỉ kiểm khung SSE — đúng chỗ dữ liệu vẫn
đúng — và bộ test đơn vị tắt cache.

Sửa: `ChatTurn.persisted_sources()`. Sau khi sửa, cả ba lượt cache đều ghi
`src=5`.

⚠️ Điểm đáng ghi không phải là bug, mà là **hướng** của nó: một công cụ săn ảo
giác tự chế ra một ca ảo giác giả. Nếu tôi không chạy thật mà chỉ đọc test, ca
đầu tiên trong file ứng viên sẽ dẫn một người thật đi tìm một lỗi bộ sinh không
tồn tại. Cùng họ với `TD-55` và với độ chệch của `W5-07`: **sai theo hướng làm
người đọc tin nhầm**, chứ không sai theo hướng ồn ào.

---

## 4. ⭐⭐ Ứng viên **không** được mang hình dạng của một câu golden

Cám dỗ là xuất thẳng ra `GoldenQuery`: điền `relevant_chunk_ids` bằng những
chunk hệ thống đã truy hồi, `reference_answer` bằng câu hệ thống đã trả lời,
`category` đoán từ độ dài. File chạy được ngay, và nó **phá hỏng chính thứ nó
bổ sung vào**.

Lý do: hàng này tồn tại *bởi vì* hệ thống đã sai. Lấy đầu ra của hệ thống làm
nhãn nghĩa là chấm hệ thống bằng chính lỗi của nó — `ndcg@10` sẽ **tăng** khi
truy hồi giữ nguyên hành vi sai và **giảm** khi nó được sửa. Một bộ eval có
tính chất ấy tệ hơn không có bộ eval nào, vì nó vẫn ra số.

Nên `GoldenCandidate` **không có** trường nào tên `relevant_*`, không
`reference_answer`, không `category`. Cái hệ thống đã làm nằm ở
`retrieved_chunk_ids` / `system_answer`, và tên của chúng nói rõ đó là *hành vi
cần xét*, không phải *nhãn đúng*.

Cách khoá nó là một bài test cho một tính chất **âm**, và tính chất âm là thứ
biến mất dễ nhất khi ai đó "làm cho tiện dùng hơn":

```python
with pytest.raises(Exception):
    GoldenQuery.model_validate(candidate.model_dump())
```

Ba trường một người phải điền để nó thành `GoldenQuery` — `category`,
`relevant_spans`, `reference_answer` — không tự động hoá được cái nào. **Đó
chính là công việc**, và một file trông như đã xong sẽ làm nó không bao giờ
được làm. CLI in ra đúng câu ấy sau mỗi lần xuất.

---

## 5. ⭐ Một luật idempotent cho **hai** kho

Một cú double-click là hai request. Hai kho, hai cơ chế, và chúng phải nói cùng
một khái niệm "lần chấm này":

| kho | khoá | cơ chế |
|---|---|---|
| Postgres | `(tenant_id, message_id)` | `UNIQUE` + `ON CONFLICT DO UPDATE` |
| Langfuse | `sha256(f"{trace_id}:{name}")[:32]` | upsert theo `id` của điểm |

Nếu chỉ có một trong hai thì đổi 👎 thành 👍 để lại **một hàng đúng** trong
Postgres và **hai điểm mâu thuẫn** trong Langfuse — và cái người ta nhìn là
Langfuse. Đo được ở lượt chạy thật: 5 lần gửi → **4** điểm tồn tại, và điểm còn
lại trên trace ấy mang giá trị của lần chấm **sau** (`wrong`), không phải lần
đầu.

⚠️ `UNIQUE` chứ không phải một phép kiểm `SELECT ... WHERE NOT EXISTS` ở tầng
ứng dụng: hai request đồng thời đi qua phép kiểm ấy cùng lúc và cả hai đều ghi.

⚠️ Khoá là `(tenant, message)`, không phải `(user, message)`, vì mô hình xác
thực của `W4-04` chỉ biết tới tenant. Một tenant nhiều người dùng ⇒ ghi đè lẫn
nhau → `TD-79`.

`xmax = 0` trong `RETURNING` là cách Postgres nói "hàng này vừa được `INSERT`,
không phải `UPDATE`"; phản hồi khai nó ra thành `replaced`, để người bấm 👎 lần
thứ hai biết lần đầu có vào hay không.

---

## 6. ⭐ `answer_message_id`: khung `meta` tới giờ không mang khoá nào chấm được

`W4-06` phát `message_id` trong khung `meta`, và đó là id của **message người
dùng**. Hàng trợ lý ra đời trong một task nền *sau khi* stream kết thúc, nên
client không bao giờ nhìn thấy id của đúng cái nó muốn chấm.

Cách hiển nhiên — gắn feedback vào message người dùng — hỏng đúng chỗ nó cần
đúng: cùng một câu hỏi hỏi lại lần hai cho **hai** câu trả lời khác nhau, và cả
hai chấm vào cùng một hàng.

Nên id được **sinh trước** ở `prepare()`, phát ra trong khung đầu tiên, rồi
`_save()` dùng lại. Ba chỗ trong hai luồng, một giá trị; có bài ghim bằng cách
đọc lại lịch sử và so.

⚠️ Id này là một **lời hứa**, không phải một sự thật: `_save()` bỏ qua câu trả
lời rỗng, nên với một lượt model im lặng thì hàng ấy không bao giờ tồn tại và
feedback vào nó trả 404 — **đúng lượt đáng nhận 👎 nhất**. `TD-78`.

---

## 7. ⭐ Bảy mã lý do, và tiêu chí để một mã đáng tồn tại

Tiêu chí: nó phải **chỉ vào một bộ phận** của pipeline.

| mã | chỉ vào |
|---|---|
| `wrong` | bộ sinh |
| `incomplete` | truy hồi / `top_k` |
| `not_found` | truy hồi (tài liệu có mà hệ thống nói không có) |
| `citation` | `W4-09` |
| `language` | prompt (luật 4, thứ `W4-06` đo được là bị bỏ qua) |
| `slow` | serving |
| `other` | cửa thoát, đi kèm text tự do |

Một mã không chỉ được vào đâu thì nó chỉ là `other` viết dài hơn, và nó làm cho
bảng thống kê lý do trông chi tiết mà không dùng được.

Danh sách sống ở **ba** nơi (`Literal` của API, `CheckConstraint` của Postgres,
nhãn Prometheus). Ba bản sao sẽ lệch, và cái lệch đầu tiên là một `INSERT` bị
Postgres từ chối ở production — tức **500**, không phải 422. Nên có một phép so
chạy lúc **import** `serving.api.feedback`: lệch ⇒ server không khởi động nổi.
Mạnh hơn một bài test, vì một bài test bỏ qua được.

---

## 8. ⭐ Điểm số được phép trượt; hàng Postgres thì không

`submit_score()` xếp hàng, không gửi — cùng hàng đợi, cùng trần, cùng bộ đếm
`dropped` với trace. Một đường thứ hai sang Langfuse là một chỗ thứ hai để chặn
`POST /feedback`, và một nút 👎 chặn 300 ms là một nút được bấm hai lần.

Nếu Langfuse chết thì feedback **vẫn nằm trong Postgres** và hàng đợi review vẫn
chạy. Chiều ngược lại không chấp nhận được: mất hàng Postgres mà vẫn có điểm
Langfuse nghĩa là một tín hiệu người dùng chỉ còn sống trong một hệ quan sát có
TTL.

Phản hồi HTTP khai `scored` để người tích hợp không suy ra rằng im lặng nghĩa là
xong — **và trường ấy đã làm đúng việc của nó ngay trong lượt chạy đầu tiên**:
nó báo `false` cho cả ba câu, vì tiến trình server hôm ấy chưa có `LANGFUSE_*`.
Không có trường đó thì tôi đã viết vào báo cáo rằng điểm đã tới Langfuse, rồi
đi đọc một bảng trống.

`status()` thêm bộ đếm `scored` riêng: gộp vào `sent` thì "trace có tới nơi
không" và "điểm có tới nơi không" trở thành một câu hỏi. Bài hợp đồng của
`W5-06` (`assert sink.status() == {...}`) đỏ ngay khi thêm — đúng việc của một
danh sách đóng.

---

## 9. Hàng đợi review, và một giới hạn phải nói ra

`GET /admin/feedback` mặc định `rating=-1`. Hàng đợi review là một danh sách
**việc phải làm**, và một lượt 👍 không phải việc phải làm — trộn vào thì danh
sách dài ra theo tỉ lệ hài lòng, tức càng tốt càng khó dùng. `rating=0` lấy tất
cả, cho phép đếm mẫu số.

⚠️ **RLS vẫn áp cho route admin.** Một key `admin` thuộc về **một** tenant, nên
đây là hàng đợi của tenant ấy, không phải của cả hệ thống. Đó là hành vi đúng,
nhưng nó nghĩa là **chưa có góc nhìn vận hành toàn cục nào**, và không được
nhầm hai thứ đó với nhau.

Câu hỏi ghép với câu trả lời bằng **hai** truy vấn rồi ghép trong Python, không
bằng `LATERAL` mỗi hàng: hàng đợi có trần 50, và một `LATERAL` ở đây là thứ
chạy đúng cho tới ngày ai đó bỏ trần.

Metric `rag_feedback_total{rating,reason}` — ⚠️ mẫu số của nó **không** phải
`rag_chat_turns`: gần như không ai bấm nút, nên đây là một mẫu tự chọn. Dòng
`HELP` nói ra điều đó, cùng khuôn với `rag_refusals_suspected` của `W5-07`.

---

## 10. Test và tiêm lỗi

| | |
|---|---|
| `tests/unit/test_feedback.py` | **23** |
| `tests/integration/test_feedback.py` | **20** (Postgres + RLS + app thật) |
| `tests/unit/test_chat_service.py` | +1 (hồi quy cho lỗi §3) |
| tiêm lỗi | **20/20 đỏ** |
| bộ mặc định | **2 225** xanh (3 skip) |
| bộ integration | **295** xanh, 0 đỏ |
| `make lint` | sạch cả ba lệnh, kể cả `mypy` **trần** (nó bắt 5 lỗi kiểu trong đúng hai file test mới) |

Ba thứ không giả lập được, và là lý do bộ integration tồn tại: **RLS** ("tenant
khác không chấm được" là hành vi của policy Postgres — một mock sẽ xanh với một
policy bị gỡ), **upsert** (`ON CONFLICT` cần chỉ mục duy nhất thật; thiếu nó thì
câu lệnh không lỗi, nó chỉ thôi là upsert), và **cầu nối `answer_message_id`**
(hai luồng, một giá trị).

**Lượt tiêm thứ nhất: 2 sống sót, và 2 không tiêm được.**

*Hai không tiêm được* là lỗi trong chính công cụ tiêm: backup đặt tên theo
`path.name`, mà `serving/api/feedback.py` và `serving/core/feedback.py` có
**cùng** `.name` — nên hai file dùng chung một file backup, bản sao thứ hai đè
bản thứ nhất, và cả hai đọc `sources` ra nội dung của một file. Phép khôi phục
lẽ ra đã ghi nội dung file này đè lên file kia; nó không xảy ra chỉ vì phép
tiêm vào file bị đè đúng lúc ấy không khớp chuỗi nên bị bỏ qua trước khi ghi.
⚠️ Một công cụ kiểm chất lượng hỏng theo hướng **báo an toàn** (hai phép bị bỏ
qua, tổng vẫn "không ai sống sót") là hỏng đúng hướng tệ nhất.

*Hai sống sót thật*, và cả hai chỉ vào một khoảng trống có thật:

* **`F7`** — bỏ `created_at = now()` khỏi nhánh `ON CONFLICT`. Không bài nào
  nhìn tới **thứ tự** của hàng đợi, nên một lần đổi ý chìm xuống dưới những
  lượt xảy ra sau nó: đúng số lượng, sai thứ tự, và người review không bao giờ
  thấy nó nổi lên.
* **`F19`** — bỏ phép kiểm `reason` ở lõi. Sống sót vì mọi bài đều đi qua HTTP,
  nơi `Literal` của FastAPI đã chặn trước. Nhưng `record_feedback` còn một
  người gọi khác — CLI và mọi script vận hành — và ở đó một mã lạ đi thẳng tới
  `CheckConstraint` rồi quay ra thành `IntegrityError`, tức **500** thay vì một
  lời từ chối đọc được.

Thêm hai bài, lượt hai **20/20 đỏ**.

⚠️ Một bài đỏ trong lúc dựng vì phụ thuộc thứ tự: fixture `database` dọn bảng ở
**đầu và cuối module**, nên mọi phép đếm ("hàng đợi có đúng 1 mục") thật ra đo
tổng của các bài chạy trước. Xanh khi chạy một mình, đỏ khi chạy cả module —
kiểu hỏng đắt nhất lúc gỡ, vì bài đỏ không phải bài sai. Thêm một fixture
`_empty_tables` cấp hàm.

---

## 11. Nợ mới

| id | nội dung |
|---|---|
| `TD-78` | **Không chấm được một câu trả lời rỗng.** `_save()` cố ý bỏ qua text rỗng, nên `answer_message_id` phát ra trong `meta` trỏ vào một hàng không tồn tại và feedback trả 404 — đúng lượt (model im lặng, `finish_reason="empty"`) đáng nhận 👎 nhất. Cũng có một cửa sổ đua: hàng ghi ở task nền, nên một cú bấm rất nhanh cũng 404 |
| `TD-79` | **Feedback khoá theo tenant, không theo người dùng.** `W4-04` không có định danh người dùng, nên một tenant nhiều người dùng thì lần chấm sau ghi đè lần trước — và hàng đợi review đếm thiếu |
| `TD-80` | **Chưa có đường promote ứng viên → golden set.** File JSONL xuất ra được, nhưng bước điền `category`/`relevant_spans`/`reference_answer` rồi trộn vào `data/golden/` là thao tác tay, chưa có công cụ và chưa có phép kiểm trùng lặp với `pipeline/goldenset/dedupe.py` |

---

## 12. `TD-50` — đóng

Cột `citations_verified` + `retrieved_sources` (migration `0004`), **không**
backfill (bất khả, xem §2). `GET /conversations/{id}` trả cả hai khoá riêng
biệt, và một bài integration đòi chúng có **kiểu khác nhau** (`list` vs `dict`)
— nếu ai đó gộp lại thì bài ấy đỏ.
