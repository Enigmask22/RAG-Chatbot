# `TD-40` — 15.814 chunk không có chủ, và ba tuần không ai thấy

> **Ngày:** 2026-09-04 · **Mã:** `pipeline/indexing/config.py`,
> `rag_core/retrieval/qdrant_store.py`, 8 file `configs/indexing/*.yaml`
> **Test:** `tests/unit/test_index_config.py::TestTenant` (7),
> `tests/unit/test_qdrant_schema.py::TestTenantOnEveryPoint` (5),
> `tests/integration/test_metadata_filter.py::TestTenantOwnershipAtWriteTime` (2)
> **Bằng chứng:** `reports/probes/td-40-backfill.json` — 15.814/15.814 point

## 1. Nợ này không phải một bug

Không có dòng mã nào sai. `W2-06` đã ghi đúng hành vi và ghim nó bằng test:
*"point thiếu `tenant_id` không khớp bất kỳ filter tenant nào"* — an toàn, đúng
hướng đã chọn, và có chủ đích. `W4-04` áp `tenant_filter()` lên mọi request,
cũng đúng.

Cái thiếu là **một dòng cấu hình**, và hai thứ đúng cộng lại thành một hệ thống
mà mọi câu hỏi đều trả `sources: []`.

Nó không lộ ra được ở bất kỳ phép đo nào đã chạy, vì đường eval **không truyền
filter** — 14 ô ablation, 209 câu golden, `hit_rate@50` = 0,8517, tất cả đo trên
cùng những point vô hình ấy và tất cả đều đúng. Chỉ có một cấu hình duy nhất
phơi nó ra: `rag_core` filter → HTTP tenant → payload Qdrant, cùng chạy một lần.
Cấu hình đó tồn tại lần đầu ở `W4-06`.

💡 Bài học không phải về Qdrant: **một tính chất chỉ hỏng khi ba tầng cùng chạy
thì không tầng nào kiểm được nó.**

## 2. ⭐⭐ Chỗ ép là config, không phải chữ ký hàm

Phản xạ đầu tiên là làm `upsert(chunks, *, tenant_id)` thành tham số bắt buộc —
"không quên được vì trình biên dịch không cho". Đếm lại thì có **đúng một** chỗ
gọi ở production (`build_index.py:615`) và **36** chỗ trong test. Tỉ lệ đó nói
rằng chữ ký hàm là hàng rào sai chỗ: nó bắt 36 đoạn mã không bao giờ chạy ở
production phải khai một thứ chúng không quan tâm, và bảo vệ đúng một dòng.

Hàng rào đúng là `IndexConfig.tenant_id`, **bắt buộc, không mặc định**. Không
chạy được `build_index` mà không khai tenant, và quyết định "corpus này thuộc
về ai" được viết ra ở nơi người ta thật sự quyết định nó.

## 3. ⭐ Vì sao **không** có mặc định

`tenant_id: str = "public"` chạy được, không phá test nào, và đóng lỗ. Nó cũng
đảo chiều hỏng:

| | quên tenant, không mặc định | quên tenant, mặc định `"public"` |
|---|---|---|
| chuyện xảy ra | không ai đọc được | tài liệu riêng vào kho công khai |
| ai thấy | người dùng, ngay | không ai |
| sửa được không | có | **không** — dữ liệu đã phát tán |

Đúng lý lẽ của `W2-06` (*"filter quá chặt → người dùng thấy; quá lỏng → không ai
thấy, kể cả người bị rò"*), chỉ lần này ở phía **ghi**. Có test ghim
(`test_there_is_no_default_tenant`), và phép tiêm "cho nó một mặc định" làm đỏ
5 test.

## 4. `tenant_id` **không** vào `fingerprint`

Nó là field payload, không chạm vector nào — đúng lý lẽ đã cho phép `W2-06`
backfill `published_at` mà mọi con số từ `W2-01` đến `W2-05` vẫn đúng nguyên.

Đưa nó vào `fingerprint` sẽ làm mọi bundle đã ký hỏng chữ ký vì một lý do không
liên quan tới chất lượng truy hồi, và buộc build lại 15.814 chunk để đổi đúng
một chuỗi trong payload. Cùng họ với quyết định của `TD-38` về `retriever.name`:
chuỗi ấy gom **những gì làm đổi con số**, và tenant quyết định *ai đọc được*,
không phải *kết quả ra sao*. Hai test ghim cả hai chỗ.

## 5. Một nguồn sự thật cho một field

`_payload` giờ đọc tenant từ **store**, và store lấy từ config. `chunk.extra
["tenant_id"]` vẫn dùng được khi store không khai gì (đường multi-tenant mà
`test_metadata_filter.py` đã kiểm từ `W2-06`), nhưng khi cả hai cùng có mặt thì
**store thắng và chỗ lệch được ghi WARNING**.

Không phải để cho gọn: một field có hai nguồn sự thật là chỗ mà tenant sai xuất
hiện lúc không ai nhìn, và im lặng ở đó nghĩa là dữ liệu đã sang tay trước khi
có ai đọc log.

## 6. Đường vá — và nó chạy trên 15.814 point thật

Lần chạy thật của `W4-06` được cứu bằng một lệnh `curl` gán payload thẳng vào
Qdrant. Đó là bằng chứng tồi: nó chứng minh Qdrant nhận field, không chứng minh
**dự án có đường sửa**.

Nên tôi gỡ nó ra rồi làm lại bằng đường thật:

```
POST /points/payload/delete {"keys":["tenant_id"],"filter":{}}   → 15.814 point sạch
count(tenant_id = public)                                        → 0
uv run python -m pipeline.indexing.backfill_payload \
    --config configs/indexing/bgem3-contextual.yaml              → 15.814/15.814
count(tenant_id = public)                                        → 15.814
chạy lại lần hai                                                  → 0/15.814
```

`backfill_flat_payload` (`W2-06`) dựng lại payload phẳng từ object `chunk` lồng
bên trong và **không chạm vector nào** — nên nó không thể làm sai một con số eval
nào, và nó no-op ở lần chạy thứ hai nên gọi được từ trong script build.

Sau đó, `POST /chat` với key tenant `public`, corpus thật, DeepSeek thật:

```
sources  4 chunk · rerank 2,30 / 1,90 / 1,60 / 1,55
done     stop · 2800+189 tok · $0,000964 · TTFB 930 ms
```

⭐ Và câu trả lời là *"không đủ thông tin"* kèm `[1][2][3][4]` — luật 3 của
prompt hoạt động. Một lần từ chối đúng đắn có giá trị hơn một lần bịa trôi chảy,
và nó chỉ đo được khi có dữ liệu thật ở đầu vào.

## 7. Tiêm lỗi — 4/4 đỏ

| tiêm vào | kết quả |
|---|---|
| cho `tenant_id` một mặc định `"public"` | ✅ 5 test đỏ |
| đưa `tenant_id` vào `fingerprint` | ✅ đỏ |
| `build_retriever` quên truyền `tenant_id` | ✅ đỏ |
| `chunk.extra` thắng tenant của store | ✅ đỏ |

⭐ Phép tiêm thứ ba là cái đáng nhất. Không có
`test_the_tenant_reaches_the_store_that_build_retriever_makes`, thì: config bắt
buộc vẫn đúng, mọi test `_payload` vẫn xanh, mọi config đã khai tenant — và index
dựng ra **vẫn không có tenant**. `TD-40` quay lại nguyên vẹn qua một dòng nối mà
không phép kiểm nào nhìn.

## 8. Còn lại

* **Corpus công khai được gán `tenant_id: public` trong cả 8 config.** Đó là một
  quyết định thật, không phải giá trị điền cho đủ: corpus World Bank là dữ liệu
  công khai, license cho phép redistribute (quy tắc cứng #3), và mọi key hiện có
  đọc được nó. Ngày có khách hàng thật, `public` là *một* tenant chứ không phải
  "mọi người" — và lúc đó cần một luật "key nào đọc được tenant nào", thứ hiện
  không tồn tại (một key = đúng một tenant).
* **`W3-08` (ingest job) chưa nhận tenant từ request.** Nó dựng `IndexConfig` từ
  file, nên nó thừa hưởng `tenant_id` của config — tức mọi tài liệu upload vào
  cùng một tenant bất kể ai upload. Đúng cho hôm nay (một corpus), sai ngay khi
  có khách hàng thứ hai.
* **Không có phép kiểm "collection này có point nào thiếu tenant không"** ở
  `/ready`. Đếm được bằng một truy vấn `count` với `is_empty`, và nó sẽ bắt đúng
  loại sự cố này ở lần deploy sau thay vì ở lần người dùng đầu tiên hỏi.
