# Audit toàn cục trước `W5-10` — 2026-09-05

> Yêu cầu: rà soát mọi vấn đề tồn đọng + giám sát toàn bộ mã nguồn tìm lỗ hổng và
> giảm hiệu năng, trước khi bước vào `W5-10` (auto-promote — điểm mà lỗi nền móng
> bắt đầu tự nhân bản).
>
> Phương pháp: (1) kiểm kê sổ sách từ `plans/CHECKLIST.md` — 35 nợ mở, 4 gate,
> bảng metric §1; (2) ba lượt review độc lập quét thẳng mã nguồn theo ba trục
> **bảo mật · hiệu năng · đúng đắn**; (3) bốn phát hiện nặng nhất được kiểm chéo
> lại bằng tay trên mã trước khi vào báo cáo này. Chi phí: $0, read-only.

---

## 0. Kết luận một đoạn

Nền móng vững hơn tôi dè chừng — ba lượt review xác nhận ~40 chỗ "đã kiểm, ổn"
(RLS, tenant filter, SSE, idempotency, RRF, budget, bundle checksum...). Nhưng
audit tìm ra **6 lỗi mới đáng sửa trước khi nhận traffic thật**, trong đó 4 đã
kiểm chéo tay và chắc chắn: cầu dao LLM đếm nhầm *khách đóng tab* thành *nhà
cung cấp hỏng*; khoá semantic cache **không chứa `top_k`/`filters`**; chi tiết
exception nội bộ rò ra client; DSN Postgres không URL-encode mật khẩu. Bên cạnh
đó, **hai con số đã tuyên bố ngưỡng đang trượt** (p95 end-to-end 4.706 > 3.500 ms;
citation quote 0,8308 < 0,85) và chưa có task nào đang kéo chúng.

---

## 1. Phát hiện MỚI — chưa có trong sổ nợ

Đánh số `AU-xx` để trỏ tới được. ✔ = đã kiểm chéo tay trên mã.

### Nhóm 1 — sửa trước khi nhận traffic thật

**`AU-01` ✔ Cầu dao đếm "khách ngắt kết nối" thành "nhà cung cấp hỏng"** —
`packages/rag_core/llm/router.py:381` (`astream`). `outcome` khởi tạo `"failure"`;
`CancelledError`/`GeneratorExit` là `BaseException` nên không rơi vào
`except Exception` nào, và `finally` gọi `route.breaker.record(outcome)` với giá
trị khởi tạo. Comment trong code chỉ xử lý đường huỷ cho phần **tiền**
(`_charge(reserved)` — đúng), bỏ sót phần **cầu dao**. Kịch bản:
`failure_threshold=3`, ba người dùng liên tiếp đóng tab giữa stream (chính lúc
provider chậm là lúc người ta hay bỏ) → mạch mở 30 giây cho một route khoẻ mạnh,
mọi `/chat` trong 30 giây đó nhận khung `error`. Sự cố tự gây, kích hoạt bởi
hành vi client. **Sửa**: bắt tường minh
`except (asyncio.CancelledError, GeneratorExit): outcome = "neutral"; raise`
trước `finally`.

**`AU-02` ✔ Khoá semantic cache không chứa `top_k` lẫn `filters`** —
`serving/core/chat.py:160` (`cache_eligible`) + `serving/core/semantic_cache.py`
(`lookup`/`store` chỉ nhận tenant · namespace · question · vector). Request có
`filters={"doc_type": "circular"}` có thể nhận **nguyên văn** câu trả lời cache
từ một request không filter (cùng tenant, cosine ≥ 0,96). Với `filters` đây là
vi phạm phạm vi dữ liệu được yêu cầu, không phải khác biệt hiệu năng. Không test
nào phủ ca hai lượt trùng khác `top_k`/`filters`. **Sửa**: đưa `top_k` + hash
JSON chuẩn hoá của `filters` vào khoá, hoặc thêm điều kiện vào `cache_eligible`
(chỉ cache request không filter, `top_k` mặc định — rẻ hơn và giữ hit-rate cho
ca phổ biến).

**`AU-03` ✔ Chi tiết exception nội bộ rò ra client** — `serving/api/chat.py:147`,
`serving/core/chat.py:899` (khung SSE `error`), `serving/api/admin.py:142`:
cùng pattern `f"{type(exc).__name__}: {exc}"`. Qdrant rớt kết nối là client đọc
được tên service nội bộ, tên collection, topology. Middleware `_send_error` đã
làm đúng (thông báo chung + `request_id`) nhưng ba chỗ này đi vòng qua nó.
**Sửa**: trả thông báo chung + `request_id`, log đầy đủ phía server.

**`AU-04` ✔ DSN Postgres không URL-encode user/password** —
`packages/rag_core/settings.py:254` (`_dsn`, f-string trần). Mật khẩu chứa
`@`/`/`/`:`/`%` làm SQLAlchemy parse ra **host khác** mà không lỗi nào cảnh
báo — DSN migration còn mang quyền superuser. **Sửa**: `urllib.parse.quote_plus`
cho cả user lẫn password, kèm một test với mật khẩu chứa `@`.

**`AU-05` PII vào Langfuse không qua `redact_pii()`** — leo thang của `TD-73`.
`serving/core/langfuse.py:97` (`trace.input` = câu hỏi người dùng) và `:180`
(`score.comment` = comment feedback) gửi thẳng; `RedactingFilter` chỉ phủ Python
logging. `TD-73` mới ghi "tenant là nhãn"; thực tế còn tệ hơn: PII của người
dùng nằm trong Langfuse không redact. Cùng họ: comment feedback lưu Postgres
cũng không redact (`serving/core/feedback.py:128`) và đi vào file xuất ứng viên
golden. **Sửa**: áp `redact_pii()` lên `trace.input`/`output`/`score.comment`
trước khi encode; cập nhật mô tả `TD-73`.

**`AU-06` ✔ Query bị embed HAI lần trên mỗi lượt cache-eligible** —
`serving/core/chat.py:560` (`embed_query` cho cache) rồi khi miss,
`retrieve()` → `hybrid.py:102` `embed_query_hybrid` embed lại **cùng câu hỏi**.
Docstring `hybrid.py` tự tuyên bố bất biến "embed MỘT lần" — tầng orchestration
phá đúng bất biến đó. Ngưỡng cosine 0,96 chỉ cho ~2/10 paraphrase qua nên **đa
số** lượt eligible đi đúng đường lãng phí này; tệ hơn: giữ khoá `TD-63` (thứ đã
gây 503 dưới tải) **hai lần** mỗi lượt. Hai agent độc lập cùng tìm ra. **Sửa**:
tính `embed_query_hybrid` một lần/lượt, dùng phần dense tra cache, truyền vector
tính sẵn vào `retrieve()` (điểm nối fuse-only đã có ở `hybrid.py:135`).

### Nhóm 2 — lỗi thật nhưng điều kiện kích hoạt hẹp hơn

**`AU-07` `review_queue` ghép câu hỏi–câu trả lời theo `created_at`, lệch được
khi hai lượt chồng nhau** — `serving/core/feedback.py:313` (`_questions_for`
chọn user message *muộn nhất* ≤ thời điểm answer). User message ghi ngay lúc
`_open_turn`, assistant ghi **sau khi stream xong** (task nền) — hai request
cùng `conversation_id` chạy song song (đa tab, retry) là ứng viên golden mang
**câu hỏi B dán lên câu trả lời của A**, sai không dấu vết. Mọi test hiện có đều
tuần tự. **Sửa**: `ChatTurn.user_message_id` đã có sẵn — ghi nó vào hàng
assistant lúc `_save()` rồi join theo khoá thật, bỏ suy luận theo thời gian.
*Nên sửa trước khi `TD-80` (đường promote ứng viên) được xây trên dữ liệu này.*

**`AU-08` `GET /conversations/{id}` trả toàn bộ, không phân trang** —
`serving/core/chat.py:1183` (`load_history` không `LIMIT`, trong khi `_history()`
nội bộ cắt 10). Hội thoại 500 lượt × 8 KB là ~4 MB serialize một cục mỗi
request. **Sửa**: `limit` mặc định 50 + cursor, theo mẫu sẵn có ở feedback.

**`AU-09` Error body của provider có thể chứa echo `Authorization`** —
`packages/rag_core/llm/openai_compat.py:291,338`: `response.text[:500]` vào
thẳng message exception → log. **Sửa**: scrub pattern `Bearer ...` trước khi
nhét vào exception.

**`AU-10` Ingest API (port 8001) không có xác thực** — `pipeline/ingest/app.py`:
`POST /ingest` nhận cả `recreate=true` (xoá collection). Giảm nhẹ hiện tại: bind
`127.0.0.1`, Docker không expose. Khác `TD-42` (tenant scoping sau auth) — đây
là "ai được gọi". **Sửa**: API key check như serving plane, hoặc ghi tường minh
vào runbook rằng 8001 phải sau firewall.

**`AU-11` Cache miss không có single-flight** — N request trùng câu hỏi đồng
thời = N lần retrieval + N lần generation trả tiền, thay vì 1. Rủi ro cost dưới
burst. **Đo trước khi đầu tư**: load-test M request trùng, đếm LLM call thật —
thuộc `W6-05`.

**`AU-12` `smoke.py` hardcode `DEFAULT_BUNDLE = v0.2.1`** —
`pipeline/eval/smoke.py:85`; CI và Makefile không truyền `--bundle`. Bump bundle
mà quên sửa hằng số là cổng PR gác **cấu hình cũ** trong khi serving chạy cấu
hình mới — bug meta làm giảm độ tin của chính cơ chế gate. **Sửa**: một nguồn
sự thật (con trỏ `current`) cho cả `app.py` lẫn `smoke.py` — khớp tự nhiên với
`W5-10` (promote = cập nhật con trỏ).

**`AU-13` Cache tắt câm lặng nếu mode `dense` thuần** (latent) —
`semantic_cache.py:53` `embedder_of` duck-typing `.store.embeddings`;
`QdrantDenseRetriever` không có `.store` → trả `None` → cache tắt không log.
Không bundle nào hiện dùng `dense` thuần nên chưa kích hoạt. **Sửa** khi chạm
file này: kiểm thêm `.embeddings` trực tiếp + test bằng class thật thay fixture.

### Nhóm 3 — ghi nhận, xử lý khi có số đo

`AU-14` Pool Postgres async dùng mặc định (5+10), chung cho foreground lẫn task
nền `_save` — expose thành settings, đo ở `W6-05` ·
`AU-15` Executor `to_thread` dùng chung giữa LLM-rewrite (timeout 6 s nhưng
thread không bị huỷ) và embed/retrieve/probe — provider chậm làm `/ready` xếp
hàng; cấp executor riêng bounded cho rewrite ·
`AU-16` `lookup()` decode/cosine 128 entry đồng bộ trên event loop — profile
trước, sửa sau ·
`AU-17` `/ready` xanh không chứng minh pool async thật đã mở kết nối (cùng họ
`TD-72`) — warm bằng `SELECT 1` lúc startup ·
`AU-18` Redis cache không có trong `/ready` lẫn `/admin/*` — không phân biệt
được "cache nguội" với "cache chết cả tuần"; thêm `/admin/cache` theo mẫu
`/admin/tracing` ·
`AU-19` Tokenizer đếm budget của contextualize ghim `Qwen/Qwen3-8B` bất kể
router failover sang model khác — assert lúc khởi động job ·
`AU-20` Escape mật khẩu trong DDL migration chỉ thay `'` → dùng
`psycopg.sql.Literal` (`alembic/versions/0001:180`) ·
`AU-21` `Feedback` thiếu index `(tenant_id, rating, created_at)` cho
`review_queue` — chỉ đáng khi bảng lớn, `EXPLAIN` trước.

### Đã kiểm, ổn (rút gọn — ba lượt review cùng xác nhận)

So khớp API key bằng SHA-256 dict lookup (không timing attack) · SSE không tiêm
được event name, `data` luôn qua `json.dumps` · NDJSON export escape chuẩn JSON ·
`tenant_filter()` ghi đè từ token **trước** mọi nhánh route, `CrossTenantError`
thay vì ghi đè câm · RLS `FORCE` + role `NOBYPASSRLS` + `SET LOCAL` bind param ·
middleware default-deny, chỉ `/health`,`/ready` public · CI không lộ secret cho
fork · compose bind `127.0.0.1`, container non-root · cache namespace theo
tenant:bundle nên swap bundle không trả lời cũ · RRF/`_depth()` không co pool
dưới `top_k`, dedup đúng, filter pushdown thật · router không double-billing,
charge phần đã giữ chỗ khi stream đứt · bundle checksum băm payload trên đĩa,
từ chối ghi đè · `_save` nền giữ tham chiếu mạnh, session mở/đóng đúng chỗ ·
SSE flush từng phần, middleware ASGI thuần · Prometheus không có label cardinality
theo tenant/query · `compare_to_baseline` một chiều là **chủ đích** (chặn
regression, không chặn improvement).

---

## 2. Con số đã tuyên bố ngưỡng đang TRƯỢT — và chưa ai kéo

| Metric | Đo được | Ngưỡng | Chủ hiện tại |
|---|---|---|---|
| p95 end-to-end | **4.706 ms** (242 req, `W5-05`) | ≤ 3.500 | `W5-11`/`W6-05` — đòn bẩy đã xác định (tầng sinh + `DEFAULT_RERANK_CANDIDATES`), chưa kéo |
| Citation accuracy cấp quote | **0,8308** | ≥ 0,85 | `TD-64` — **19/67 lỗi là dấu lược `...`**: chuẩn hoá ellipsis khi so khớp là việc thuần code, $0, có thể trả một mình phần thiếu 0,019 |
| nDCG@10 vs `G6` | 0,6888 | ≥ 0,82 | Nghẽn cấu trúc `cross_lingual` (20% tập đo) — không lấp bằng tinh chỉnh; cần quyết định ở `W5-11`/`W6`: đổi ngưỡng có căn cứ, hay đổi tầng nền |

Sổ sách kèm theo: bảng §1 cột "Hiện tại" vẫn đứng ở `bgem3-rr-c50` (`W2-05`,
nDCG 0,6481) trong khi index đang phục vụ là contextual (0,6888 ở `exp-002`) —
đúng lỗi "cột đứng lại" mà chính bảng này đã cảnh báo một lần.

## 3. Gate treo và đường trả

- **`G1` 🟡** — chờ đúng `TD-13`: **bạn** đọc lại 33 câu `unanswerable` + 43 câu
  `cross_lingual` rồi `make goldenset-freeze --reviewer human`. ~1 buổi, mở khoá
  chữ "human-verified" cho mọi báo cáo sau.
- **`G2` 3/4** — ô cuối (p95 ≤ 3.500) giờ **có số và số nói trượt** (§2). Ô này
  nên được cập nhật trạng thái tường minh thay vì để trống như thể chưa đo.
- **`G4` 2,5/3** — `TD-56`: sửa câu chữ gate ("≤ 5 phút **khi cache ấm**").
- **`W3-09` 2/5 ô** — 3 ô còn lại bị `TD-20` chặn (đổi chunking = đổi tập nhãn);
  DoD gốc đã trả lời xong. Quyết định cần ghi: hoặc làm hạ tầng `TD-20` (đắt),
  hoặc thu hẹp phạm vi ô về DoD gốc và đóng `[~]` có ghi chú.

## 4. 35 nợ mở — nhìn theo cụm thay vì theo số

1. **Cụm multi-tenant nửa vời** (nặng nhất cho chữ "production"): `TD-39`
   rate-limit theo tiến trình · `TD-42` ingest bỏ qua tenant · `TD-43` `/ready`
   không soi tenant coverage · `TD-44` một key = một tenant · `TD-47` trần ngân
   sách trong RAM · `TD-79` feedback khoá theo tenant. Chung một nguyên nhân:
   trạng thái sống trong tiến trình + mô hình xác thực chưa có khái niệm người
   dùng. Redis đã có sẵn trong stack — `TD-39`/`TD-47` trả được bằng nó.
2. **Cụm gate/promote** (chặn `W5-10`): `TD-71` (ghi phán quyết vào bundle) +
   `TD-82` (đúc lại manifest, phải cùng lúc) + `AU-12` (con trỏ bundle) +
   `TD-70` (bí danh generator làm gate trả `INCOMPARABLE`).
3. **Cụm eval-trung-thực**: `TD-13` (golden chưa human-verified — rẻ, việc của
   bạn) · `TD-64` (`AU` đề xuất sửa ellipsis) · `TD-65`/`TD-67`/`TD-68` (judge)
   · `TD-31` (chất lượng ngữ cảnh) · `AU-07` (ứng viên golden lệch câu hỏi).
4. **Cụm hiệu năng có đòn bẩy đã đo**: p95 (tầng sinh, `W5-11`) · `TD-63`+`AU-06`
   (khoá + embed đôi) · `TD-72`+`AU-17` (khởi động lạnh) · `TD-75`/`TD-76`
   (metrics đa tiến trình) — phần lớn hội tụ về `W6-05`.
5. **Cụm chấp nhận có ghi chú** (không đáng tiền lúc này): `TD-05`, `TD-10`,
   `TD-18`, `TD-25`…`TD-29`, `TD-33`/`34`, `TD-51`…`TD-54`, `TD-59`…`TD-62`,
   `TD-74`, `TD-77`, `TD-81`, `TD-83`, `W0-03`.

## 5. Đề xuất thứ tự làm

1. **Vá `AU-01`…`AU-06` + `AU-07` thành một task `NEW-08`** trước `W5-10` —
   toàn bộ đều nhỏ, tất định, test được, $0; ước lượng một phiên làm việc.
   Kèm `TD-64` (chuẩn hoá ellipsis) vì nó kéo một ngưỡng đang trượt về xanh.
2. **`W5-10` như kế hoạch**, gộp `TD-71` + `TD-82` + `AU-12` (cùng đụng manifest
   và con trỏ bundle — làm một lần đỡ đúc lại hai lần).
3. **`TD-13`** — việc của bạn, ~1 buổi, nên xen song song.
4. `W5-11` trả p95 + cost có thẩm quyền; `W6-05` mang theo `AU-11`/`AU-14`/
   `AU-15`; `W6-06` (security pass) đối chiếu lại `AU-03`/`AU-05`/`AU-09`/`AU-10`.

> Chưa sửa gì trong mã ở audit này — toàn bộ là read-only. File này chưa commit.
