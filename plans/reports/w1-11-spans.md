# `TD-12` đã trả — golden set neo theo span ký tự

> Ngày: 2026-08-20 · Nhánh: `feat/w1-foundation` · Commit `0b87cfe`
> Trạng thái: **xong**, đã chạy thật trên corpus 60 tài liệu và index 15.814 chunk

---

## 1. Vấn đề, và vì sao nó tệ hơn một lỗi bình thường

`chunk_id` được sinh ở `packages/rag_core/chunking/base.py`:

```python
chunk_id=f"{doc.doc_id}::{index:05d}"
```

Thuần vị trí. Đổi `chunk_size` là mọi nhãn golden set trỏ vào văn bản khác.

Điều làm nó tệ hơn một lỗi bình thường: **id vẫn tồn tại.** Không có gì báo lỗi.
Đo thật trên 226 câu đã gán nhãn, ba cấu hình chunking:

| `chunk_size` | số chunk | nhãn `chunk_id` cũ "còn hợp lệ" |
|---|---|---|
| 1000 (lúc gán nhãn) | 15.814 | 226/226 |
| 600 | 28.215 | **226/226** ← trỏ vào văn bản khác |
| 400 | 44.530 | **226/226** ← trỏ vào văn bản khác |

`fetch_chunks` báo sạch. `gold_chunk_missing` = 0. `make goldenset-verify` xanh —
checksum canh nội dung *file golden set*, không canh việc index nó trỏ tới có còn
như cũ. Recall vẫn ra một con số, chỉ là con số đó vô nghĩa.

Và `TD-11` nói việc **đầu tiên** của `W2` là hạ `chunk_size`. Nên golden set đóng
băng hôm nay sẽ hỏng ở ngay bước sau, im lặng.

Đây là lỗi thiết kế của tôi ở `W1-07`: chọn `chunk_id` thuần vị trí mà không nghĩ
tới việc cả `W2` và `W3` đều là về việc đổi cách chunk.

---

## 2. Cách sửa

Nhãn neo vào **văn bản gốc**, không vào chunk. Văn bản gốc là thứ bất biến duy
nhất trong hệ này: `data/corpus_manifest.csv` ghi sha256 từng tài liệu và
`iter_documents` kiểm lại ở mỗi lần build index.

```
GoldenQuery.relevant_spans = [(doc_id, start, end), ...]
                    ↓  eval tự ánh xạ theo index ĐANG ĐO
      chunk_size=1000 → doc::00023
      chunk_size=600  → doc::00038, doc::00039
```

`relevant_chunk_ids` **được giữ lại**, không xoá. Nó vẫn là nhãn đúng cho index
hiện tại, và giữ nó là điều kiện để đối chiếu xem ánh xạ span có dựng lại đúng
tập cũ hay không — chính phép đối chiếu đó là bằng chứng ở mục 5.

---

## 3. Hợp đồng của span: **vùng xuất xứ**, không phải chỉ dẫn cắt

Đây là chỗ dễ hiểu sai nhất, và hiểu sai thì sửa "cho đúng" sẽ làm đổ mọi số
baseline.

`chunk.content` **không** bằng `document.content[start_char:end_char]`.

Vì sao không thể ép cho bằng:

* Splitter đệ quy làm `[s for s in text.split(sep) if s]` — **bỏ mảnh rỗng** — rồi
  nối lại bằng `sep.join(...)`. `"A\n\nB"` tách theo `"\n"` cho ra `"A\nB"`, ngắn
  hơn nguyên bản một ký tự.
* Splitter ngữ nghĩa nối các câu bằng dấu cách, bất kể nguyên bản ngăn nhau bằng gì.
* `_enforce_size` gộp mảnh nhỏ bằng `"\n"`, cũng bất kể nguyên bản.

Ép `content` thành substring nguyên văn = đổi nội dung chunk = đổi vector = đổi
mọi con số baseline. Nên hợp đồng là: `[start, end)` là **vùng mà chunk được dẫn
ra từ**, có thể rộng hơn `content` vài ký tự khoảng trắng. Đủ cho mục đích duy
nhất của span, và không chỗ nào trong dự án dùng nó làm chỉ dẫn cắt.

Có test khẳng định tường minh điều này
(`TestSpanIsProvenanceNotSlice::test_dropped_empty_split_makes_text_shorter_than_span`)
để không ai "sửa" nó thành bằng nhau.

**Một quyết định nhỏ nhưng quan trọng:** `_apply_neighbor_context` làm `content`
dài ra nhưng **giữ nguyên span**. Đệm là bản sao text của chunk khác; gán nó vào
span của chunk này thì mỗi chunk sẽ "sở hữu" một vùng chồng lên hai chunk bên
cạnh, và mọi phép ánh xạ nhãn sẽ khớp thừa ba lần.

---

## 4. Bằng chứng nội dung chunk không đổi

Bất biến số một của cả lần refactor này. Chốt digest **trước** khi sửa dòng nào,
trên corpus thật 60 tài liệu:

| chiến lược | tài liệu | chunk | sha256 nội dung — trước | — sau |
|---|---|---|---|---|
| `fixed` | 60 | 15.814 | `b381634d51e39365` | `b381634d51e39365` ✅ |
| `semantic` | 6 | 1.970 | `e00bc87aaffe792e` | `e00bc87aaffe792e` ✅ |
| `hybrid` | 60 | 15.814 | `b381634d51e39365` | `b381634d51e39365` ✅ |

Digest băm `chunk_id` + `content` của từng chunk theo thứ tự. Khớp cả ba nghĩa là
không một byte nào đổi, nên số baseline đo trước đó vẫn so sánh được.

Bên cạnh đó, 530 unit test cũ + mới xanh, trong đó `TestTextUnchanged` so trực
tiếp `split_recursive` và `split_sentences` với **đúng biểu thức của bản cũ**.

Span cũng phải hợp lý, không chỉ tồn tại:

```
15.814 chunk / 60 tài liệu
  thiếu span             : 0
  span ngoài biên        : 0
  span chồng chunk trước : 8.230 (tổng 546.653 ký tự — chunk_overlap=100 là chủ ý)
  tỉ lệ phủ tài liệu     : p10=0,971 p50=0,989 min=0,922
```

Phần hụt ~1% là khoảng trắng giữa các chunk bị `.strip()` cắt.

---

## 5. Bằng chứng nhãn span sống qua việc đổi chunking

266 nháp → 226 câu trả lời được → **299 span**, trong đó **186 (62%) thu về đúng
câu trích dẫn**, 113 rộng bằng cả chunk. 0 chunk_id không dựng lại được, 0 câu
không neo được.

Rồi ánh xạ ngược lại, ở ba cấu hình chunking:

| `chunk_size` | khớp đúng tập nhãn cũ | ra tập khác | **không khớp gì** |
|---|---|---|---|
| 1000 (lúc gán nhãn) | **216/226** + 10 rộng ra | 0 | **0** |
| 600 | 0 | 226 ✅ đúng | **0** |
| 400 | 0 | 226 ✅ đúng | **0** |

Đọc bảng này:

* Ở **1000**, ánh xạ span dựng lại đúng nhãn cũ cho 216/226. 10 câu còn lại ra
  *nhiều* chunk hơn — do `chunk_overlap=100` làm một trích dẫn nằm trong vùng
  chồng của hai chunk. Cả hai chunk đều thật sự chứa bằng chứng, nên đó là đúng.
* Ở **600** và **400**, nhãn ra tập chunk khác — đúng như phải thế, vì chunk đã
  khác. Và **không câu nào mất nhãn**.

So với dòng "226/226 id cũ vẫn hợp lệ" ở mục 1: cùng một sự thay đổi, nhãn
`chunk_id` thì hỏng âm thầm, nhãn span thì đi theo.

### Một khiếm khuyết bị lộ ra nhờ chạy ở `chunk_size=400`

Quy tắc ban đầu là `overlap / span.length >= 0.5`. Ở 400 thì **40/226 câu không
khớp chunk nào**: span rộng bằng cả chunk cũ (~1000 ký tự) mà không chunk 400 ký
tự nào chứa nổi 500 ký tự.

Sửa thành đối xứng:

```
overlap / span.length >= ratio   HOẶC   overlap / chunk.length >= ratio
```

Nhánh thứ hai xử lý đúng trường hợp chunk nằm **trọn trong** vùng bằng chứng —
rõ ràng là liên quan, dù chỉ chiếm một phần nhỏ của vùng. Sau khi sửa: 0 câu mất
nhãn ở cả ba cấu hình.

Ngưỡng vẫn an toàn khi span rộng, vì span luôn bị chặn bởi `max_chunk_size` (1500
ký tự) — không có span nào rộng bằng cả tài liệu để làm mọi chunk thành "liên quan".

Nếu chỉ test ở 1000 thì khiếm khuyết này không lộ ra, và nó sẽ nổ đúng lúc `W2`
hạ `chunk_size`.

---

## 6. Đường đầu-cuối đã chạy thật

Build lại index (`--recreate`, 170s, 93,1 chunk/giây) để point mang offset — kiểm
551/551 chunk của một tài liệu đều có span, tổng collection 15.814.

Rồi chạy `retrieval_eval` với nhãn span:

```json
"span_resolution": {
  "resolved": 226, "kept_chunk_ids": 40, "unmatched_queries": [],
  "min_overlap_ratio": 0.5, "label_changed": 10
}
```

`recall@5 0,1615 · recall@20 0,2507 · mrr 0,1543 · p50 20,9 ms`

⚠️ **Đây không phải baseline** — 266 nháp chưa qua review tay. Nhưng nó khớp với
kết quả triage (28,8% chunk đã gán nằm trong top-20 → recall@20 ≈ 0,25), tức hai
đường đo độc lập cho cùng một con số.

`label_changed: 10` chính là 10 câu ở mục 5. `unmatched_queries: []` là điều phải
kiểm trước khi đọc recall: nếu nó không rỗng thì những câu đó bị chấm 0 vì lý do
**không phải** retrieval.

---

## 7. Hai chỗ hỏng âm thầm đã chặn thêm

**Cache trả về offset `None`.** `SQLiteChunkCache` dùng `TypeAdapter(list[Chunk])`.
Thêm field **optional** vào `Chunk` thì `validate_json` trên payload cũ **không**
báo lỗi — nó chỉ điền `None`. Nên cái guard `ValidationError` ở `get()` không bắt
được, và cache cũ sẽ lặng lẽ trả về chunk không có offset. Đã đưa version vào
**tên bảng** (`chunk_cache_v2`, kèm `DROP TABLE` bảng cũ) để entry cũ trở thành
*không tồn tại* thay vì *hỏng ngầm*.

**Ánh xạ trả rỗng biến câu khó thành câu bị loại.** `evaluate_run` bỏ qua câu có
`relevant_chunk_ids` rỗng — nó coi đó là câu unanswerable. Nếu `resolve_queries`
trả rỗng khi không khớp gì thì đúng những câu khó nhất tự động rơi khỏi tập đo và
recall tăng lên. Cùng loại bẫy mà `W1-11` đã dựng hàng rào để tránh. Nên khi không
khớp gì thì **giữ nhãn cũ** và ghi vào `unmatched_queries`. Có test
(`test_unmatched_span_keeps_the_old_label`).

---

## 8. Số liệu

| | |
|---|---|
| Test | **530 unit** (409 → 530, +121) · **38 integration** |
| Lint | `ruff check` · `ruff format --check` · `mypy --strict` sạch, 74 file |
| File mới | `chunking/pieces.py` · `eval/spans.py` · `goldenset/anchor.py` + 3 file test |
| Neo | 226/226 câu · 299 span · 186 thu theo trích dẫn (62%) |
| Build lại index | 170s · 93,1 chunk/giây · 15.814 chunk đều có span |

---

## 9. Lệnh tái lập

```bash
make goldenset-anchor            # nháp → span, ~4s (chunk lại corpus, không cần Qdrant)
make index BUNDLE=baseline       # cần --recreate lần đầu để point mang offset
make goldenset-triage
# review tay → decisions_v1.csv
make goldenset-freeze
make eval-retrieval              # đọc `config.span_resolution` trước khi đọc recall
```

---

## 10. Còn lại

* **113/299 span vẫn rộng bằng cả chunk.** Phần lớn là trường hợp đúng: câu
  `aggregation`/`multi_hop` có một trích dẫn nhưng ba chunk nguồn, nên hai chunk
  còn lại không có trích dẫn riêng để thu hẹp. Bước review tay có thể thu hẹp
  thêm, nhưng không bắt buộc — ánh xạ đã chứng minh chạy đúng với span rộng.
* **`min_overlap_ratio = 0.5` chưa được điều chỉnh theo dữ liệu.** Nó là một tham
  số của phép đo, nên đáng đưa vào ablation ở `W2`: cùng golden set, quét ratio,
  xem recall nhạy tới mức nào. Nếu nhạy thì đó là điều phải công bố trong report
  baseline.
* Nhãn `chunk_id` vẫn còn trong file. Xoá được sau `W2-01`, khi đã có ít nhất một
  lần đo chứng minh đường span đủ tin.
