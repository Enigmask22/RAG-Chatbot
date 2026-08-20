# `W2-02` — một collection Qdrant, hai loại vector

**Ngày:** 2026-08-20 · **Nhánh:** `feat/w1-foundation` · **Collection:** `rag_bgem3`

> **Kết luận một dòng.** `rag_bgem3` giờ mang cả named vector `dense` (1024-d) và
> `sparse`, query được độc lập, và sparse chỉ tốn thêm **8,8 giây** trên tổng 389
> giây index (+2,3%) cùng **+19%** dung lượng vector.
>
> Kiểm định quan trọng nhất: dense sau khi build lại **bit-identical** với trước —
> 15/15 metric không lệch một chữ số, 0/209 câu đổi điểm. Đường ghi hybrid mới
> không làm lệch nhánh dense.

---

## 1. Sparse gần như miễn phí, và đó là hệ quả của một quyết định ở `W2-01`

| | dense-only (`W2-01`) | dense + sparse (`W2-02`) | |
|---|---:|---:|---|
| embed + ghi | 380,4 s | **389,2 s** | +8,8 s (+2,3%) |
| thông lượng | 39,0 chunk/s | 38,2 chunk/s | −2,1% |
| chunk | 15.814 | 15.814 | — |

Con số 2,3% không phải may. Sparse của BGE-M3 là một `Linear(1024 → 1)` đặt lên
**cùng** `last_hidden_state` đã dùng cho dense, và `W2-01` đã cho hai thứ đó ra từ
một forward pass. Nếu tầng store gọi `embed_documents()` rồi gọi tiếp một hàm
sparse riêng thì đây sẽ là **+380 giây**, không phải +8,8 — trả gấp đôi tiền
forward pass cho đúng một kết quả. Nên `upsert` có `_embed_batch` gọi provider
**một** lần.

## 2. Dense không đổi một chữ số

Đường ghi mới (`_embed_batch` → `embed_documents_hybrid`) là một đường code khác
với đường của `W2-01` (`embed_documents`). `W2-01` đã tuyên bố "`_forward` là
đường sinh dense duy nhất"; đây là chỗ kiểm tra tuyên bố đó trên 15.814 chunk
thật thay vì trên 4 câu test:

| kiểm tra | kết quả |
|---|---|
| 15 metric tổng thể | **0 metric lệch** |
| điểm từng câu (209 câu) | **0 câu đổi** |
| `n_relevant_mean` | 1,3828 (không đổi) |
| fingerprint | `0eaaf9265487eabb` (không đổi — `chunking` + model không đổi) |

Nếu chỉ so bảng metric tổng thể thì hai lần chạy khác nhau vẫn có thể trùng số do
bù trừ. Đó là lý do so cả `*-per-query.jsonl` — hạ tầng thêm ở `TD-11` giờ dùng
được cho việc này.

## 3. Sparse trên dữ liệu thật

Mẫu 3.000 chunk, `entry` = số token có trọng số dương:

| | giá trị |
|---|---:|
| trung bình | **95,9** entry/chunk |
| p50 | 100 |
| p95 | 147 |
| max | 195 |
| **min** | **3** |
| mật độ | 0,0384% của vocab 250.002 |

Đọc được hai điều từ đây.

**ReLU thật sự chọn lọc.** Chunk có p50 **218 token** (`W2-01` §3) mà chỉ ~96
token còn trọng số dương — model loại bỏ khoảng **55%** token. Sparse của BGE-M3
không phải bag-of-words có trọng số; nó là một phép chọn.

**`min = 3` là một cảnh báo cho `W2-04`.** Có chunk chỉ còn 3 token dương, tức
nhánh sparse gần như không tìm thấy nó bằng bất kỳ truy vấn nào. Đây là mặt bù
của đặc tính đo được trong test: **không trùng token thì sparse trả về rỗng**,
trong khi dense vẫn đoán được. Hai nhánh hỏng theo hai kiểu khác nhau, và đó
chính là lý do RRF đáng làm chứ không phải chỉ để có thêm một dòng trong CV.

### 3.1 Chi phí: lưu trữ và độ trễ

| | dung lượng |
|---|---:|
| dense 1024-d × 15.814 | 61,8 MB |
| sparse (96 entry × 8 byte) | **+11,6 MB (+19%)** |

Độ trễ truy hồi **dense** trước/sau khi thêm sparse index vào cùng collection
(209 truy vấn, đo 3 lần):

| | trước (`W2-01`) | sau (`W2-02`) | |
|---|---:|---:|---|
| mean | 27,7 ms | 33,2 ms | +20% |
| **p50** | **23,7 ms** | **31,5 ms** | **+33%** |
| p95 | 46,0 ms | 46,6 ms | +1,3% |
| max | 50,4 ms | 49,5 ms | — |

Mức tăng p50 **tái lập được** (31,3 · 31,6 ở hai lần đo lại) nên không phải nhiễu.
Đây là khoản phải trả thật, và nó cho biết một điều về cấu trúc của con số: **p95
gần như không đổi** vì p95 bị chi phối bởi forward pass embed truy vấn của BGE-M3
(biến động lớn), trong khi p50 phản ánh phép tìm trong Qdrant.

⚠️ **Chưa tách được nguyên nhân**: +7,8 ms có thể là chi phí của sparse index trong
cùng collection, hoặc là trạng thái segment sau khi vừa build lại (8 segment, Qdrant
tối ưu ở background). Chưa đo để phân biệt. Với ngưỡng `G2` (p95 < 3500 ms) thì 46,6
ms còn rất nhiều chỗ, nên chưa đáng đào — nhưng phải đo lại ở `W2-04`, nơi mỗi truy
vấn sẽ đi **cả hai** nhánh.

### 3.2 Hai nhánh cho thứ hạng khác nhau

Truy vấn `"Tăng trưởng GDP của Việt Nam năm 2023"` trên `rag_bgem3`:

| # | dense | sparse |
|---|---|---|
| 1 | `wb-099619110162342004::00004` (0,6682) | `wb-099142309042522486::00158` (0,2938) |
| 2 | `wb-099142309042522486::00158` (0,6634) | `wb-099142309042522486::00038` (0,2707) |
| 3 | `wb-099619110162342004::00000` (0,6615) | `wb-676661480599107823::00019` (0,2392) |

Chỉ **1/3** chunk trùng nhau, và nó đứng thứ 2 ở dense vs thứ 1 ở sparse. Nếu hai
nhánh cho cùng thứ hạng thì `W2-04` không có gì để hợp nhất — và sẽ nghĩa là một
trong hai đang đọc sai named vector.

⚠️ Thang điểm hai nhánh **không so được với nhau**: dense là cosine (đã chuẩn hoá,
∈ [−1, 1]), sparse là **dot product** của trọng số không âm nên không có trần.
Đây chính là lý do RRF hợp nhất theo **thứ hạng** chứ không theo điểm.

## 4. `ensure_collection` giờ kiểm tra schema

Đây là phần không có trong DoD nhưng là phần đáng nhất của `W2-02`.

Trước đó `ensure_collection` thấy collection tồn tại là **trả về ngay**. Chạy
config `bgem3` (provider sinh sparse) lên collection `rag_bgem3` cũ (dense-only)
sẽ chết ở lần upsert **đầu tiên** — sau khi đã nạp 2,2GB trọng số, chunk xong tài
liệu đầu, và với một thông báo của Qdrant không nói phải làm gì.

Bốn ca lệch, mỗi ca hỏng một kiểu, đều có test:

| ca | hỏng thế nào nếu không chặn |
|---|---|
| thiếu named vector `dense` | collection phiên bản cũ dùng vector vô danh → mọi truy vấn `using="dense"` lỗi |
| số chiều khác | ca của mỗi lần đổi model (768 → 1024); Qdrant từ chối upsert *sau khi* đã nạp model |
| provider sinh sparse, collection không có chỗ | ca **mới**; chết giữa job |
| collection có sparse, provider chỉ dense | ⚠️ **hỏng im lặng** — eval ra số trông bình thường trong khi nửa index không được dùng |

Ca cuối cố ý **không** được bỏ qua. Nó là biến thể của đúng cái bẫy `TD-11`: hệ
thống chạy, số ra, không ai biết là sai.

Thông báo lỗi nói ra `--recreate` và cả lối thoát thứ hai (đổi `collection` trong
config để giữ index cũ mà đối chiếu). Có test canh chính chuỗi `--recreate` xuất
hiện — một thông báo chỉ nói "schema mismatch" thì người đọc vẫn phải đi đọc code.

`schema_problems()` là **hàm thuần**, không chạm Qdrant, nên 11 ca lệch test được
trong `make test` (~3 giây) chứ không cần server.

## 5. Script migrate — và vì sao nó cố ý không migrate

`scripts/migrate_collection.py`. Qdrant **không** cho thêm named vector vào
collection đã tồn tại, nên "migrate tại chỗ" là chuyện không tồn tại. Script này
làm thứ nó làm được:

```
$ python scripts/migrate_collection.py --config configs/indexing/bgem3.yaml
  config `bgem3` → collection `rag_bgem3`
    cần             dense 1024 chiều + named vector 'sparse'
    đang có         dense {'dense': 1024} · sparse (không) · 15814 point
  ✗ provider sinh sparse nhưng collection không có named vector 'sparse'
  Qdrant không sửa được schema tại chỗ — phải ghi lại toàn bộ point.
  Chạy lệnh sau (hoặc thêm `--run` vào script này):
    python -m pipeline.indexing.build_index --config configs/indexing/bgem3.yaml --recreate
$ echo $?
1
```

Sau khi build lại: `✅ Schema khớp. Không cần build lại.` (mã 0).

Đây là phiên bản cho **schema** của câu hỏi mà `IndexConfig.fingerprint` trả lời
cho **nội dung**: *"thứ đang nằm trong Qdrant có phải thứ config này mô tả
không?"* — câu hỏi bắt buộc trước khi so hai con số eval.

Một phương án đã cân nhắc rồi bỏ: copy dense vector cũ sang collection mới rồi
chỉ tính thêm sparse. Không tiết kiệm gì, vì tính được sparse của BGE-M3 tức là đã
tính lại dense (§1). Phần thật sự đắt — chunking — đã có cache (hit 60/60).

## 6. `HashingEmbeddingProvider` nay sinh được sparse

`W2-02`…`W2-04` cần một provider sinh sparse mà **không** cần GPU và 2,2GB trọng
số, nếu không thì mọi test của schema hybrid, sparse retriever và RRF đều phải
chạy BGE-M3 thật — và `make test-integration` sẽ không chạy được trên CI.

Nó cũng đúng với thứ đang mô phỏng: dense là bag-of-words băm xuống `dimension`
chiều, sparse là **cùng** bag-of-words băm xuống 2^16 chiều. Gộp trọng số bằng
`max` chứ không `+=`, khớp cách BGE-M3 gộp.

**Mặc định `sparse=False`**, và đó là quyết định có chủ ý: `name` của provider đi
vào cache key của semantic chunker (`chunking/semantic.py`) và vào MLflow, nên bật
sẵn sẽ đổi tên của mọi provider mặc định — vô hiệu cache chunk và làm mọi test W1
đổi nghĩa âm thầm. Có test hồi quy canh `embed_documents()` cho **kết quả y hệt**
khi bật/tắt sparse, và canh `name` mặc định vẫn là `hashing-64d`.

(Lần đầu tôi để mặc định `True` và hai test W1 đỏ ngay — `test_provider_name_is_
reproducible` và một test của chính `W2-01`. Đó là cách phát hiện ra `name` là
cache key.)

## 7. Thứ CHƯA làm

* **Chưa có sparse retriever đúng nghĩa.** `retrieve_sparse()` là phép truy vấn ở
  tầng store, không phải một `Retriever` mà eval harness chạy được. `W2-03` mới
  có, và mới có **số** cho câu hỏi "sparse đóng góp gì".
* **Chưa hợp nhất.** `W2-04` (RRF). §3.2 chỉ cho thấy hai nhánh *khác nhau*, chưa
  cho thấy hợp nhất thì tốt hơn.
* **Chưa đo sparse trên golden set.** Cố ý để `W2-03` làm, kèm `p`/CI qua
  `make eval-compare` — mọi so sánh W2 phải qua đó.
* **Chưa có BM25 thô.** Nhánh đó sẽ cần `modifier=Modifier.IDF`; nhánh BGE-M3 thì
  **không** (§8).

## 8. Chi tiết dễ sai: không dùng `Modifier.IDF` cho BGE-M3

Qdrant có `SparseVectorParams(modifier=Modifier.IDF)` để tự áp IDF lên sparse
vector. Trọng số của BGE-M3 là **đã học** — model tự quyết token nào quan trọng,
và phần "hạ bậc từ phổ biến" đã nằm trong trọng số. Chồng IDF của Qdrant lên là
nhân đôi phép đó, và nó hỏng theo kiểu tệ nhất: điểm vẫn ra số, chỉ là sai.

`Modifier.IDF` dành cho nhánh BM25 thô ở `W2-03`, nơi giá trị đầu vào là **tần
suất** chứ không phải trọng số. Có test integration canh `modifier` của
`rag_bgem3` là `None`.

## 9. Tái lập

```bash
make test                                          # 666 unit (22 ca mới: 11 schema + 11 hashing sparse)
make up && make test-integration                   # 59 integration (21 ca hybrid)
python scripts/migrate_collection.py --config configs/indexing/bgem3.yaml
python -m pipeline.indexing.build_index --config configs/indexing/bgem3.yaml \
  --recreate --report plans/reports/index-bgem3.json   # ~414s trên RTX 4060
make eval-retrieval BUNDLE=bgem3                   # phải khớp số cũ từng chữ số
```

## 10. Bước tiếp

1. **`W2-03` sparse / BM25 retriever** — bọc `retrieve_sparse()` thành `Retriever`
   để eval harness chạy được, rồi đo trên golden set. DoD của nó là truy vấn từ
   khoá lạ (mã số, tên riêng) mà dense miss thì sparse hit; §3 cho thấy `min = 3
   entry` là mặt bù phải đo cùng.
2. **`W2-04` RRF** — hợp nhất theo **thứ hạng**, không theo điểm (§3.2).
3. `W2-08` vẫn phải có ma trận hai chiều `chunk_size` × `embedding` (`W2-01` §5).
