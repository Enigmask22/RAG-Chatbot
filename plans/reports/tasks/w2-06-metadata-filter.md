# `W2-06` — Metadata filter và cách ly tenant

> 2026-08-21 · DoD: *filter áp ở tầng Qdrant (không post-filter), không rò dữ liệu
> chéo tenant* · Test: `tests/integration/test_metadata_filter.py` — ca tenant isolation

## 0. Câu hỏi của hạng mục này

Không phải "làm sao lọc theo metadata" — phần đó đã có từ `W1-07`. Câu hỏi là:
**còn đường nào vào dữ liệu mà không đi qua filter?**

Đó là câu hỏi khác hẳn, và nó cho một câu trả lời khác hẳn.

## 1. Dự đoán ghi trước khi đo

| # | Dự đoán | Kết quả |
|---|---|---|
| D1 | Đường search (`retrieve`) đã kín cả 4 nhánh; chỉ cần thêm test | ✅ đúng |
| D2 | Thiếu chính là `date range` — `build_filter` không có khoảng | ✅ đúng, nhưng không phải phần quan trọng |
| D3 | Sẽ phải build lại index (~380 s) để thêm `published_at` | ❌ **sai** — backfill payload đủ, và không chạm vector |
| D4 | Lọc làm truy vấn **chậm** hơn vì Qdrant phải kiểm thêm điều kiện | ❌ **sai** — không đo được khác biệt (1,8%) |
| D5 | Filter chặt (ít point khớp) đắt hơn filter lỏng | ❌ **sai** — không có quan hệ; ca 0 point thì **nhanh gấp đôi** |

`D3` sai theo hướng có lợi và đáng ghi lại vì lý do: tôi mặc định "đổi payload =
build lại index", trong khi payload và vector là **hai** thứ Qdrant cập nhật độc
lập. `set_payload` không đọc, không ghi, không chạm vector — nên mọi con số eval
từ `W2-01` đến `W2-05` vẫn đúng nguyên vẹn sau migrate. Nếu build lại thì chúng
**có thể** đúng và tôi sẽ phải chứng minh điều đó.

## 2. ⭐ Cái thực sự thiếu: đường fetch

`retrieve()` áp filter ở Qdrant từ `W1-07`, và `W2-05` đã có test canh nó áp
**trước** khi vào pool rerank. Nhưng store có hai method nữa:

| method | dùng ở đâu | filter trước `W2-06` |
|---|---|---|
| `retrieve()` | truy vấn vector | ✅ có |
| `fetch_chunks(chunk_ids)` | phân giải nhãn golden set; **`W4`: giải citation** | ❌ **không có tham số** |
| `fetch_doc_chunks(doc_ids)` | ánh xạ span (`TD-12`); **`W4`: mở rộng ngữ cảnh** | ❌ **không có tham số** |

Hai dòng dưới là hai đường vòng hoàn chỉnh. Ở `W4-09` (citation verification) và
`W4-06` (chat), tầng serving sẽ gọi đúng chúng để lấy nội dung chunk theo id —
và một `chunk_id` lấy từ câu trả lời cũ, từ log, hoặc đoán được sẽ trả về **nội
dung đầy đủ của tenant khác**, dù mọi truy vấn vector đều lọc đúng.

Điều làm nó khó thấy: đường search và đường fetch trông giống nhau ở tầng gọi,
nhưng `client.retrieve(ids=...)` của Qdrant **không nhận filter** — Qdrant không
hỗ trợ. Nên "thêm filter vào chỗ đó" không phải thêm một tham số, mà là chuyển
sang `scroll` với điều kiện trên `chunk_id`. Giữ hai đường code là có chủ ý và
được ghi trong docstring: đường không filter dùng `retrieve` (nóng, dùng cho
phân giải nhãn, lấy đúng point theo id, không phân trang).

`test_fetch_paths_agree_with_the_search_path` canh chỗ dễ lệch nhất: **ba** cách
dựng filter cho cùng một câu hỏi phải cho cùng một tập chunk.

## 3. Hướng của chế độ hỏng, và vì sao nó quyết định mọi mặc định

Với filter thường (lọc `lang` để đo breakdown), hỏng-thành-rỗng chỉ gây nhầm lẫn.
Với `tenant_id`, hai hướng hỏng **không đối xứng**:

* Quá chặt → thiếu kết quả. Người dùng thấy, báo lại, sửa được.
* Quá lỏng → **dữ liệu tenant khác lọt ra**. Không ai thấy, kể cả người bị rò.

Nên `MetadataFilter` nghiêng hết về hướng thứ nhất, và mỗi quy tắc dưới đây ứng
với một cách viết filter cho 0 kết quả **mà không báo lỗi**:

| Viết sai | Trước `W2-06` | Sau |
|---|---|---|
| `{"tenant": "acme"}` (thiếu `_id`) | Qdrant trả **0 kết quả** | `ValueError` kèm danh sách khoá hợp lệ |
| `doc_type=[]` | `MatchAny(any=[])` khớp **rỗng** | `ValidationError` |
| `after=2024, before=2020` | khoảng rỗng, 0 kết quả | `ValidationError` |
| chunk không có `tenant_id` | không khớp filter nào | *giữ nguyên* — có test ghim |

Ba dòng đầu là code; dòng thứ tư là hành vi `MatchValue` của Qdrant, và nó đúng
hướng — nhưng nó là hành vi của **thư viện bên thứ ba**, nên nó có test riêng
(`test_a_chunk_without_tenant_matches_no_tenant_filter`) thay vì chỉ được tin.

`MetadataFilter` cũng `frozen=True`, và đó là quyết định bảo mật chứ không phải
thẩm mỹ: nếu `W4` kiểm `filters.tenant_id == token.tenant` rồi truyền tiếp, một
filter đổi được cho phép nới nó ra **sau** khi đã kiểm.

## 4. ⚠️ Cái `W2-06` KHÔNG đóng được

Nó **không** ép người gọi phải truyền `tenant_id`. `retrieve(query)` không filter
vẫn thấy tất cả — và `rag_core` không thể biết như vậy là đúng (eval chạy trên
toàn corpus, và phải như thế) hay là một lỗ rò (serving quên truyền tenant).

Chỗ ép được là tầng serving, nơi tenant đến từ **token đã xác thực**: `W4-04`.
Ghi vào docstring của `filters.py` và có một test **ghim hành vi hiện tại**
(`test_fetch_chunks_without_a_filter_still_returns_everything`) chứ không ghim
hành vi mong muốn — để tới `W4` không ai đọc `W2-06` xong tưởng chuyện này đã
xong.

## 5. Giá của việc lọc: **không có**. Và tôi đã gần như báo cáo nhiễu.

`make filter-probe BUNDLE=bgem3` · nhánh dense · 200 truy vấn của `golden_v1` mỗi
ca · `rag_bgem3` 15.814 point · `probes/w2-06-filter-probe.json`.

| ca | point khớp | | **chỉ Qdrant** p50 | e2e p50 |
|---|---:|---:|---:|---:|
| không filter | 15.814 | 100,0% | **30,37 ms** | 37,7 |
| `lang=en` | 9.393 | 59,4% | **30,25 ms** | 33,0 |
| `lang=vi` | 6.421 | 40,6% | **30,20 ms** | 41,6 |
| `doc_type=dev_report` | 15.814 | 100,0% | **30,00 ms** | 43,9 |
| `published_after=2020` | 8.996 | 56,9% | **29,98 ms** | 42,6 |
| khoảng 2020–2023 | 6.040 | 38,2% | **30,03 ms** | 31,4 |
| `lang=en` + khoảng | 3.238 | 20,5% | **30,46 ms** | 32,0 |
| tenant không tồn tại | 0 | 0,0% | **15,39 ms** | 31,1 |
| *không filter (lặp cuối)* | 15.814 | 100,0% | **30,54 ms** | 39,4 |

**Tám ca có kết quả nằm trong 29,98–30,54 ms — trải 0,56 ms, tức 1,8%.** Từ filter
giữ 20,5% point tới không filter gì: **không đo được khác biệt**. `D4` và `D5` đều
sai, và sai theo hướng có lợi.

Ngoại lệ duy nhất là ca **0 point khớp: 15,39 ms, đúng một nửa.** Qdrant nhận ra
điều kiện có cardinality bằng 0 và cắt sớm thay vì đi HNSW. Đó không phải ca nhân
tạo — nó là "tenant mới, chưa có tài liệu", và biết nó *nhanh* thay vì *chậm* là
thông tin có ích cho `W4`.

### ⚠️ Tôi đã gần như báo cáo nhiễu như một kết quả

Hai lượt đo đầu chỉ có cột đầu-cuối, và chúng cho bảng này:

| lượt | không filter | `lang=en` | `doc_type` (100% khớp) | khoảng |
|---|---:|---:|---:|---:|
| n=40, lượt 1 | 44,8 | 38,9 | 43,5 | 34,0 |
| n=40, lượt 2 | 46,1 | 35,6 | **33,8** | **44,0** |
| n=200 | 32,5 | 38,2 | 45,0 | 32,6 |

Đọc riêng lượt 1 thì nó **đơn điệu theo độ chọn lọc** và tôi đã suýt viết "lọc
làm truy vấn nhanh hơn, càng chặt càng nhanh". Ba thứ chặn lại, theo thứ tự:

1. **Đối chứng thứ tự.** Các ca chạy tuần tự với "không filter" đứng đầu, nên
   chiều giảm dần có thể chỉ là chiều của warm-up. Thêm một lần lặp đúng ca đầu ở
   **cuối** bảng: 45,2 vs 46,1 — lệch 2%, nên thứ tự không phải nguyên nhân.
2. **Cùng một ca, hai lượt, lệch ±11 ms.** `doc_type` đi 43,5 → 33,8; khoảng đi
   34,0 → 44,0. Thứ tự trong bảng là **hoán vị**, không phải xu hướng.
3. **p95 đứng im trong khi p50 nhảy.** 46,7–49,2 ms ở cả 9 ca, trải 2,8%. Một đại
   lượng mà p50 lệch 30% còn p95 lệch 3% thì phần biến động **không** nằm ở thứ
   đang đo.

Cách sửa **không** phải tăng số mẫu — n=200 vẫn cho cùng một ca hai giá trị 32,5
và 43,9. Cách sửa là **phân rã**: embed truy vấn một lần ngoài vòng bấm giờ, rồi
chỉ bấm giờ lời gọi Qdrant. Xong thì trải từ 30% tụt xuống 1,8%.

Đây đúng là bài học `W2-04` §6, lần thứ hai: **khi các con số không cộng lại
đúng thì tách chúng ra, đừng lấy thêm mẫu.** Ở `W2-04` nó tìm ra bug 64 ms; ở đây
nó tìm ra rằng câu trả lời là "không tốn gì" chứ không phải "tốn âm".

⚠️ Và một quan sát ngoài lề đáng cho `W2-08`: **`doc_type=dev_report` khớp
15.814/15.814 point.** Toàn bộ corpus là một `doc_type`, nên `doc_type` hiện là
một chiều filter **chết** — dùng nó làm một dòng ablation sẽ cho hai dòng giống
nhau. `lang` thì thật (59,4% / 40,6%).

## 6. Migrate index đang có

`published_at` là field payload phẳng mới, nên 15.814 point của `rag_bgem3`
(build ở `W2-02`) không có nó. Point thiếu field **không khớp** `DatetimeRange`,
nên trước migrate thì `published_after=2020` trả 0 kết quả trên toàn corpus —
đúng chế độ hỏng im lặng của §3, chỉ là lần này do dữ liệu cũ chứ không do code.

`make backfill-payload BUNDLE=bgem3`:

| | |
|---|---|
| point | **15.814** |
| payload index tạo mới | `published_at` (kiểu `datetime`) |
| point cập nhật | **15.814 / 15.814** |
| vector chạm tới | **0** |

Hai việc tách rời vì hỏng theo hai kiểu khác nhau:

1. **Thiếu payload index** → filter vẫn **đúng**, Qdrant lùi về quét toàn bộ.
   Hỏng về hiệu năng nên **không test nào đỏ**. Đây là lý do §5 tồn tại.
2. **Thiếu field trong payload** → filter **sai** (trả rỗng).

Chạy lại là no-op (so payload hiện có với payload đúng rồi chỉ ghi phần lệch).

⚠️ `published_at` phải là index kiểu `datetime`, không phải `keyword`. Index
keyword *cũng dựng được* trên cùng field và mọi truy vấn khớp-chính-xác vẫn chạy
— rồi `DatetimeRange` không dùng được nó và Qdrant quét. Có test canh
(`test_published_at_is_indexed_as_datetime_not_keyword`).

## 7. Một chỗ nới nửa vời mà mypy bắt được

Tôi nới `build_filter` và `fetch_*` để nhận `MetadataFilter`, nhưng để nguyên
`Retriever.retrieve(filters: dict[str, Any] | None)`. Kết quả: truyền một
`MetadataFilter` vào `retrieve()` **chạy đúng** nhưng là lỗi kiểu — tức API mới
chỉ dùng được ở một nửa số điểm vào, và nửa còn lại chỉ hỏng khi có người chạy
`mypy`.

`mypy --strict` bắt 25 lỗi, gồm cả 4 chỗ vi phạm Liskov ở các `Retriever` giả
trong test. Sửa bằng một alias dùng chung:

```python
type FilterSpec = MetadataFilter | dict[str, Any] | None
```

`dict` giữ lại để mọi chỗ gọi cũ và mọi nguồn ngoài (YAML, query string) dùng
được mà không phải import gì — nó **vẫn** đi qua `MetadataFilter.model_validate`
nên vẫn được kiểm khoá. Khác biệt: `MetadataFilter` cho type checker bắt tên
field sai, `dict` chỉ nổ lúc chạy.

## 8. Test

| file | số | kiểm gì |
|---|---:|---|
| `tests/unit/test_metadata_filter.py` | 24 | mọi cách viết sai đều nổ; `FILTER_FIELDS` ↔ `PAYLOAD_INDEXES` khớp **hai chiều** |
| `tests/integration/test_metadata_filter.py` | 22 | cách ly tenant trên **cả 4 nhánh**; đường fetch; filter tới được request |

Ba chỗ đáng nói:

**Điểm neo.** `test_without_a_filter_both_tenants_are_visible` chạy **trước** mọi
test cách ly. Không có nó thì "có filter → không thấy `t2`" có thể xanh vì truy
vấn vốn đã không trả `t2` (nội dung khác chủ đề), và cả file mất nghĩa. Vì thế
nội dung 8 chunk của hai tenant **cố ý gần giống nhau**, và `t2` phủ hết mọi ô
`lang`/`doc_type`/ngày của `t1` để không filter nào loại nó một cách tình cờ.

**Cả bốn nhánh, không phải một.** `dense` dùng `query_points`; `hybrid` dùng
`query_batch_points` và truyền filter vào **từng** `QueryRequest`; `reranked`
chuyển tiếp xuống nhánh nền. Ba nhánh xanh không nói gì về nhánh thứ tư. Riêng
hybrid có test đếm: **hai** sub-request phải **đều** mang filter — quên một cái
thì rò một nửa, và nửa bị rò sẽ *không hiện ra trong kết quả* nếu RRF không đẩy
nó lên top-k, tức lỗ tồn tại rất lâu mà mọi test kết quả vẫn xanh.

**Filter ở server, không phải ở client.** Chữ đầu của DoD, và nó **không suy ra
được từ kết quả** — lọc sau cũng cho kết quả đúng. Nên phải theo dõi chính
request (`monkeypatch` lên `query_points`/`query_batch_points`).

## 9. Còn lại gì

* `W4-04` phải ép `tenant_id` từ token — §4.
* `fetch_chunks`/`fetch_doc_chunks` nhận filter nhưng **không đòi**; tầng serving
  phải luôn truyền. Không có gì trong `rag_core` bắt được việc quên.
* `W2-08` có thêm hai chiều dùng được: `lang` và `doc_type` — nhưng lọc theo
  chúng **đổi tập câu hỏi được chấm**, nên đó là breakdown chứ không phải một
  dòng ablation so được với dòng khác.
