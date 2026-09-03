# `W4-05` — Postgres, Alembic, và lỗ tenant thứ ba

> **Ngày:** 2026-09-03 · **Mã:** `serving/db/{models,engine}.py`, `alembic/`
> **Test:** `tests/integration/test_migrations.py` (13, cần Postgres thật)
> **Kiểm thật:** `/ready` = **200** với cả ba nhánh xanh trên hệ thống đầy đủ — §6

## 1. ⭐⭐ Lỗ tenant thứ ba, và lần này nó nằm trong SQL

Cùng một lỗ hổng đã xuất hiện ba lần ở ba tầng:

| tầng | cách quên | cách chặn |
|---|---|---|
| `rag_core` (`W2-06`) | `retrieve(query)` không filter | ⚠️ **không chặn được ở đó** |
| HTTP (`W4-04`) | route quên `Depends` | middleware chặn theo mặc định |
| **SQL (`W4-05`)** | `WHERE conversation_id = :id` quên `AND tenant_id` | **Row-Level Security** |

Điểm chung là **hướng hỏng**, không phải cơ chế. `SELECT * FROM message WHERE
conversation_id = :id` là câu SQL tự nhiên nhất trên đời, nó chạy đúng trên môi
trường dev một-tenant, và ở production nó trả về hội thoại của khách hàng khác.
Với RLS, cùng câu ấy trả **rỗng** — thiếu kết quả thì người dùng thấy và báo lại,
dư kết quả thì không ai thấy, kể cả người bị rò.

## 2. ⭐⭐ Cái bẫy: RLS **tự báo cáo là đã đóng**

Hai lớp bẫy, và tôi rơi vào cả hai.

**Lớp một — `ENABLE` không đủ, phải `FORCE`.** `ENABLE ROW LEVEL SECURITY` bật
policy cho mọi role *trừ chủ sở hữu bảng*, và ứng dụng kết nối bằng chính owner.

**Lớp hai — và đây mới là lớp thật sự đắt.** Sau khi `FORCE`, năm test **cấu
hình** xanh hết (`relrowsecurity = t`, `relforcerowsecurity = t`) trong khi bốn
test **hành vi** đỏ. Nguyên nhân: `POSTGRES_USER` của image postgres là
**superuser**, và superuser bỏ qua RLS **hoàn toàn** — kể cả `FORCE`.

```
rag|t          ← usesuper
```

Nếu chỉ viết test cấu hình thì hạng mục này đã kết thúc với một bảng kiểm toàn
xanh và **không một policy nào có tác dụng**. Đó là lý do phải có cả hai loại
test, và là lý do loại thứ hai đáng giá hơn.

**Cách sửa:** một role riêng cho ứng dụng — `rag_app`, `NOSUPERUSER NOBYPASSRLS`,
chỉ có DML trên năm bảng cộng `SELECT` trên `alembic_version`. Migration chạy
bằng role owner; đường request chạy bằng `rag_app`.

`GRANT SELECT ON alembic_version` là hạt nhỏ nhưng có chủ đích: `/ready` phải
trả lời được "schema trong DB có đúng schema mã này giả định không", và lối sai
là cho tiến trình serving dùng DSN của owner — tức mang credential superuser vào
tiến trình nhận request từ Internet.

## 3. "DB sẵn sàng" nghĩa là **migration đã chạy**

`W4-03` để trống nhánh DB của `/ready`. Câu hỏi thật không phải "cắm được
không".

`SELECT 1` trả lời "cắm được". Cách hệ thống này hỏng là: image mới lên,
`alembic upgrade head` chưa chạy (hoặc chạy lỗi và bị bỏ qua), pod báo sẵn sàng,
nhận traffic, rồi mọi request chết bằng `column … does not exist` — với một
`SELECT 1` xanh suốt.

Nên phép thử so `alembic_version` trong DB với **head của thư mục migration
trong chính image này**, đọc từ `ScriptDirectory` chứ không ghim hằng số (hằng số
phải sửa tay mỗi lần thêm migration, và lần quên đầu tiên biến phép thử thành
luôn-xanh). Nhiều head là **lỗi**, không phải hai lựa chọn.

## 4. `SET LOCAL`, không `SET SESSION`

Policy đọc `app.tenant_id` từ tham số phiên. Connection pool **tái dùng** kết
nối, nên một `SET SESSION` sót lại làm request kế tiếp mang tenant của request
trước. Đó là rò chéo tenant do một dòng cấu hình, và nó chỉ xảy ra **khi có
tải** — tức không bao giờ ở máy dev.

## 5. Bốn quyết định nhỏ

* **`psycopg` v3, không `asyncpg`** — một driver phục vụ cả hai chế độ (Alembic
  đồng bộ, request async). Dùng asyncpg thì migration vẫn cần thêm một driver
  đồng bộ, tức hai driver phải đồng ý với nhau về kiểu dữ liệu.
* **`timestamptz`, không `timestamp`** — cái sau bỏ lặng lẽ phần offset khi ghi.
* **`ingest_job` KHÔNG thay Redis của `W3-08`** — Redis giữ hàng đợi và trạng
  thái đang chạy; bảng này giữ bản ghi bền sau khi Redis hết TTL. Nguồn sự thật:
  Redis trong lúc job chạy, Postgres sau khi nó kết thúc.
* **`conversation.bundle_version`** — sau một lần hot-swap (`W4-02`) hoặc
  rollback, cùng một câu hỏi cho câu trả lời khác; không có trường này thì không
  cách nào biết bản nào đã trả lời.
* **`downgrade` không `DROP ROLE`** — role là đối tượng của cả cluster, và
  `downgrade` của một schema không có quyền quyết định chuyện ngoài schema.

## 6. ⭐⭐ Chạy thật — và nó bắt một bug trong chính mã `W4-01` của tôi

Lần chạy đầu với đủ Qdrant + Postgres, bundle **không nạp được**:

```
bundle 0.1.0 được eval trên
  reranked[qdrant-hybrid:rag_bgem3_ctx:rrf1-c20]:…@cuda:L512:float16:n50
nhưng máy này dựng ra
  reranked[qdrant-hybrid:rag_bgem3_ctx:rrf1-c20]:…@cuda:L512:float16:n50-top6
```

Phép kiểm danh tính của `TD-38` — viết xong **cùng ngày** — bắt được **hai** bug
trong `_parse_rerank` của `W4-01`:

1. `top_n = int(branch_options.get("rerank_top_n", 6))` — **bịa ra 6** khi lần
   eval không nêu, ngay dưới một docstring nói rằng *"đoán giá trị ở đây sẽ tạo
   ra một bundle mô tả sai hệ thống đã đo"*.
2. Tham số tên là `branch_options` nhưng chỗ gọi truyền `eval_config`, trong khi
   `retrieval_eval.py` ghi `rerank_top_n` vào `branch_options`. Nên giá trị thật
   **không bao giờ được đọc** — kể cả khi lần eval có nêu.

Không test nào của `W4-01` bắt được: cả `6` lẫn `None` đều hợp lệ về hình dạng,
và bundle round-trip xanh. Thứ bắt được là một phép so chuỗi trên hệ thống thật.

Sửa: `RerankComponent.top_n` thành `int | None` (**vẫn bắt buộc**, không mặc
định — `None` là câu "không cắt thêm"), `_parse_rerank` không đoán, chỗ gọi
truyền đúng dict. Sinh lại bundle mẫu. Sau đó:

```json
{"ready":true,"checks":{"bundle":{"ok":true},"qdrant":{"ok":true,"duration_ms":7.47},
 "postgres":{"ok":true,"duration_ms":137.3}},"active":"0.1.0"}
```

`runtime_drift: null` — runtime khớp **chính xác** hệ thống đã eval. Đây là lần
đầu cả Serving Plane chạy đủ: bundle → BGE-M3 trên GPU → Qdrant → kiểm migration
→ auth.

## 7. Tiêm lại lỗi — và hai lần tiêm **không** đỏ, vì hai lý do khác nhau

| lỗi tiêm vào | kết quả |
|---|---|
| bỏ `FORCE ROW LEVEL SECURITY` | 5 test đỏ ✅ |
| `top_n` mặc định về 6 | `test_an_unset_top_n_is_recorded_as_none_not_guessed` ✅ |
| truyền `eval_config` thay `raw_options` | `test_an_explicit_top_n_is_kept` ✅ |
| đổi role thành `BYPASSRLS` | **không đỏ** — nhưng đó là **phép tiêm hỏng** |
| xoá `WITH CHECK` khỏi policy | **không đỏ** — và đó là **chú thích của tôi sai** |

**Phép tiêm hỏng:** tôi chỉ sửa nhánh `CREATE ROLE`, mà role đã tồn tại nên
migration đi nhánh `ALTER ROLE` — không mutate gì. Chạy tay `ALTER ROLE rag_app
BYPASSRLS` thì đúng hai test hành vi đỏ ngay. Cùng họ với phép tiêm sửa nhầm
dòng comment ở `W3-04`: **một phép tiêm không đỏ có hai cách giải thích, và phải
loại trừ cách thứ hai trước.**

**Chú thích sai:** tôi viết *"thiếu `WITH CHECK` thì tenant ghi được hàng của
tenant khác"*. Sai — Postgres dùng chính biểu thức `USING` cho chiều ghi khi
`WITH CHECK` vắng mặt. Giữ lại `WITH CHECK` vì nó là hàng rào cho tương lai
(ngày nào `USING` được nới ra cho một role đọc chéo tenant, thiếu `WITH CHECK` sẽ
nới luôn chiều ghi — im lặng), nhưng chú thích đã được sửa lại cho đúng.

## 8. Còn nợ

* **`downgrade` chỉ được kiểm trên DB rỗng** — nó chứng minh **DDL đảo được**,
  không chứng minh **deploy lùi lại được**: lùi một migration có dữ liệu là mất
  dữ liệu, và không test nào ở đây nói khác. Ghi vào runbook `W5`.
* **Chưa có engine async** — đường request của `W4-06` cần
  `create_async_engine`; `psycopg` v3 làm được bằng cùng DSN, nhưng
  `tenant_session` sẽ cần một bản async song song.
* **Kho API key vẫn ở file** (`W4-04`) — giờ đã có DB để chuyển vào, và thu hồi
  key sẽ thành một câu `UPDATE` thay vì sửa file + restart.
