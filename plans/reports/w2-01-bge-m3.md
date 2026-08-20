# `W2-01` — BGE-M3: dense 1024-d + sparse lexical weights

**Ngày:** 2026-08-20 · **Nhánh:** `feat/w1-foundation` · **Config:** `configs/indexing/bgem3.yaml`

> **Kết luận một dòng.** Đổi model embedding sang BGE-M3, giữ nguyên toàn bộ
> chunking của baseline: **mọi metric truy hồi tăng có ý nghĩa thống kê**
> (nDCG@10 0,1621 → 0,4442, `p < 0,001` hoặc CI95 không chứa 0 ở cả 15 metric),
> và nhóm `cross_lingual` từ recall@5 **bằng 0** lên **0,3023**.
>
> ⚠️ **Nhưng đừng gán mức tăng này cho việc sửa `TD-11`.** Lần chạy này đổi cùng
> lúc *model*, *cửa sổ ngữ cảnh* và *tính đa ngữ*. `TD-11` đã đo riêng phần cửa
> sổ: xoá truncation hoàn toàn (56,9% → 0,4%) với PhoBERT cho `p = 0,711`, tức
> **gần như không đóng góp gì**. Vậy +153% ở đây là của **model**, không phải của
> việc hết bị cắt. Xem §5.

---

## 1. Vì sao chọn BGE-M3, và vì sao giữ `chunk_size=1000`

`TD-11` để lại hai số đo dẫn thẳng tới đây:

1. **Hạ `chunk_size` là đánh đổi, không phải thu hồi nội dung.** Baseline bị cắt
   nhưng mỗi vector vẫn đọc ~950 ký tự; `chunk550` không bị cắt nhưng mỗi vector
   chỉ đọc 678. Cửa sổ **8192 token** của BGE-M3 lấy được phần lợi — hết
   truncation — mà *không* phải trả phần ngữ cảnh.
2. **Baseline có `cross_lingual` recall@5 = 0.** PhoBERT là model đơn ngữ; nhóm
   43 câu hỏi chéo ngôn ngữ không có cách nào hoạt động. Đây là lỗ hổng **cấu
   trúc**, không phải chuyện tinh chỉnh.

Giữ `chunk_size=1000` còn có một lợi ích về **đo lường** mà `TD-11` dạy cho:
nhãn neo theo span nên đổi `chunk_size` làm số nhãn/câu đổi theo, và mẫu số của
recall là chính con số đó. Giữ chunking y nguyên thì recall@k/nDCG/MAP **so được
trực tiếp** — và đã xác nhận: `n_relevant_mean` **1,3828 ở cả hai lần chạy**,
`span_resolution` giống nhau từng trường (`resolved: 209`, `label_changed: 9`).
`pipeline/eval/compare.py` không từ chối metric nào, khác hẳn `TD-11`.

## 2. Đã dựng gì

| Thành phần | Chỗ | Ghi chú |
|---|---|---|
| `SparseVector` | `rag_core/embedding/sparse.py` | Kiểu bất biến, cưỡng chế 3 bất biến lúc khởi tạo |
| `HybridVectors` | `rag_core/embedding/base.py` | NamedTuple `(dense, sparse)` |
| Năng lực sparse tuỳ chọn | `EmbeddingProvider` | `sparse_vocab_size`, `embed_documents_hybrid`, `embed_query_hybrid` — mặc định `None` |
| `BgeM3EmbeddingProvider` | `rag_core/embedding/bge_m3.py` | Dense + sparse **một** forward pass |
| Chọn provider theo tên model | `rag_core/embedding/__init__.py` | `BAAI/bge-m3` → provider hybrid |
| `embedding_max_batch_tokens` | `IndexConfig` | Knob VRAM, **ngoài** `fingerprint` |
| `make test-gpu` | `Makefile` | Test cần trọng số thật |
| Test | `test_sparse_vector.py` (21) · `test_bge_m3.py` (23 CPU + 11 GPU) | |

### 2.1 Sparse của BGE-M3 không nằm trong `sentence-transformers`

`modules.json` của repo chỉ có `Transformer → Pooling → Normalize`, tức ST chỉ
trả dense. Sparse là một `Linear(1024 → 1)` đặt lên `last_hidden_state`, trọng số
ở `sparse_linear.pt` (`{weight: (1,1024), bias: (1,)}`) — ST không nạp file đó.

Cách làm: nạp thẳng `sparse_linear.pt`, và **dùng lại chính `XLMRobertaModel` mà
ST đã nạp** (`self.model[0].auto_model`) thay vì mở bản thứ hai. 2,2GB trọng số
nhân đôi thì 4060 8GB hết chỗ.

Trọng số gộp theo token bằng **max** qua các vị trí, bỏ `[CLS]`/`[SEP]`/`[PAD]`/
`[UNK]`. Max chứ không phải sum vì trọng số trả lời *"token này quan trọng thế
nào cho đoạn text"*, không phải *"nó xuất hiện bao nhiêu lần"* — đổi sang sum thì
sparse retrieval biến thành đếm tần suất thô. Bỏ `[CLS]` là bắt buộc: nó thường
là chiều nặng nhất, để lại thì mọi cặp text khớp nhau ở đúng chiều đó và điểm
sparse gần thành hằng số.

### 2.2 `_forward` là đường code duy nhất

Cái giá của việc không đi qua `model.encode()` là dense giờ có đường code riêng —
và hai đường sinh dense song song là cách chắc chắn để hai nhánh ablation vô tình
đo hai thứ khác nhau. Nên:

* `_encode` (dense-only) cũng gọi `_forward` rồi bỏ phần sparse.
* Có test canh `embed_documents()` khớp `SentenceTransformer.encode()`. Đo được:
  **max |Δ| = 1,5e-8**, cosine = 1,0000.
* Phải làm lại phép **sắp batch theo độ dài** mà `encode()` vốn tự làm: padding
  tới câu dài nhất trong batch nên trộn câu 40 token với câu 500 token làm hầu
  hết phép tính đổ vào padding.

Sắp giảm dần còn một lợi ích thứ hai: câu dài nhất rơi vào batch **đầu**, nên
cấu hình sẽ OOM thì OOM ở giây thứ nhất chứ không phải ở phút thứ ba của một job
15.000 chunk.

### 2.3 `batch_size` một mình không chặn được VRAM khi cửa sổ là 8192

16 câu × 8192 token = 131k token — OOM ngay. Nên có thêm trần
`max_batch_tokens` tính theo **độ dài thật** của batch, mặc định 8192. Chunk ngắn
vẫn chạy full batch; chỉ batch có câu dài mới bị chia nhỏ.

### 2.4 `None` ≠ rỗng, lần thứ hai

Cùng bài học của `TD-11` (`max_sequence_tokens`), áp cho sparse:

* `embed_documents_hybrid()` trả **`None`** = "provider này không sinh sparse".
* Trả `SparseVector` **rỗng** = "đã tính, không token nào có trọng số dương"
  (text chỉ gồm special token) — một kết quả hợp lệ.

Gộp hai thứ đó thì `W2-03` sẽ không phân biệt được "provider chỉ có dense" với
"sparse retrieval trả 0 kết quả", và cả hai trông giống nhau: im lặng.

## 3. Truncation: 56,9% → **0,0%**

Chạy `make truncation BUNDLE=bgem3` **trước** khi eval, đúng như kế hoạch chốt:

| | chunk | bị cắt | token mất | token/ký tự |
|---|---:|---:|---:|---:|
| **baseline** (PhoBERT, 256) | 15.814 | **56,9%** | **15,4%** | — |
| `en` | 9.393 | 65,3% | 19,4% | 0,244 |
| `vi` | 6.421 | 44,7% | 7,5% | 0,188 |
| **bgem3** (8192) | 15.814 | **0,0%** | **0,0%** | — |
| `en` | 9.393 | 0,0% | 0,0% | **0,172** |
| `vi` | 6.421 | 0,0% | 0,0% | 0,185 |

`0/15814` chunk vượt cửa sổ. Token/chunk p50 218 · p95 295 · **max 734** — còn
cách 8192 hơn mười lần.

**Phát hiện phụ đáng ghi: tokenizer BGE-M3 hiệu quả hơn hẳn cho tiếng Anh.**
0,172 vs 0,244 token/ký tự của PhoBERT, tức **giảm 30% số token cho cùng một
lượng text tiếng Anh**. Chiều bất đối xứng cũng đảo: với PhoBERT thì `en` tốn
nhiều token hơn `vi` (0,244 > 0,188), với BGE-M3 thì `en` **tốn ít hơn** (0,172 <
0,185). Đây là gốc của việc `en` bị cắt nặng nhất ở baseline.

`TD-17` (tài liệu dùng `Ê` làm dấu cách, 0,63 token/ký tự với PhoBERT) không còn
gây truncation ở cửa sổ 8192 — nhưng văn bản vẫn không có ranh giới từ, nên nợ
đó **vẫn mở** và vẫn sửa ở `W3-01`.

## 4. Kết quả eval

209/242 câu được chấm (33 câu `unanswerable` loại khỏi metric xếp hạng).
Kiểm định bằng `make eval-compare BASE=baseline CAND=bgem3`: McNemar exact cho
metric nhị phân, bootstrap cặp 10.000 lần (seed 20260820) cho metric liên tục.

| metric | baseline | bgem3 | Δ | Δ tương đối | kiểm định |
|---|---:|---:|---:|---:|---|
| `hit_rate@1` | 0,1196 | 0,3397 | +0,2201 | +184% | `p < 0,001` · 8↔54 |
| `hit_rate@5` | 0,2153 | 0,5455 | +0,3301 | +153% | `p < 0,001` · 5↔74 |
| `hit_rate@10` | 0,2775 | 0,6268 | +0,3493 | +126% | `p < 0,001` · 4↔77 |
| `hit_rate@20` | 0,3110 | 0,6746 | +0,3636 | +117% | `p < 0,001` · 5↔81 |
| `recall@5` | 0,1746 | 0,4769 | +0,3022 | +173% | CI95 [+0,2392, +0,3652] |
| `recall@10` | 0,2257 | 0,5813 | +0,3557 | +158% | CI95 [+0,2911, +0,4195] |
| `mrr` | 0,1660 | 0,4394 | +0,2734 | +165% | CI95 [+0,2156, +0,3325] |
| **`ndcg@10`** | **0,1621** | **0,4442** | **+0,2820** | **+174%** | CI95 [+0,2297, +0,3346] |
| `map@20` | 0,1349 | 0,3853 | +0,2504 | +186% | CI95 [+0,1988, +0,3035] |
| `precision@5` | 0,0459 | 0,1340 | +0,0880 | +192% | CI95 [+0,0699, +0,1072] |

**Cả 15 metric đều "khác biệt thật".** Ở `hit_rate@5`, 74 câu BGE-M3 tìm được mà
baseline miss, chỉ 5 câu ngược lại — không phải chuyện phân giải nữa.

Nhắc lại vì nó quan trọng: `golden_v1` chỉ phân giải được mức chênh ≥ **6 điểm**
`hit_rate` (`TD-11` §8). Mức chênh ở đây là **33 điểm**, gấp năm lần ngưỡng. Đây
là loại kết quả mà tập 209 câu **đủ sức** phát hiện — khác hẳn `TD-11`.

### 4.1 Theo nhóm truy vấn (`hit_rate@5`)

| nhóm | n | baseline | bgem3 | Δ | đổi chiều | p |
|---|---:|---:|---:|---:|---|---:|
| `cross_lingual` | 43 | **0,0000** | 0,3256 | +0,3256 | 0↔14 | **0,0001** |
| `aggregation` | 26 | 0,2308 | 0,7692 | +0,5385 | 1↔15 | **0,0005** |
| `factoid` | 68 | 0,3088 | 0,6029 | +0,2941 | 3↔23 | **0,0001** |
| `multi_hop` | 34 | 0,3529 | 0,6765 | +0,3235 | 1↔12 | **0,0034** |
| `adversarial` | 34 | 0,1765 | 0,4412 | +0,2647 | 0↔9 | **0,0039** |
| `table_lookup` | 4 | 0,0000 | 0,2500 | +0,2500 | 0↔1 | 1,0000 |

Năm trong sáu nhóm cải thiện có ý nghĩa. `table_lookup` có **n = 4** — không có
lực thống kê nào ở cỡ đó; con số ở đó không nói gì và không nên được kể như
thắng hay thua.

### 4.2 Lỗ hổng cấu trúc `cross_lingual` đã đóng

| metric | baseline | bgem3 |
|---|---:|---:|
| `recall@5` | **0,0000** | 0,3023 |
| `recall@10` | 0,0233 | 0,4419 |
| `hit_rate@10` | 0,0233 | 0,4419 |
| `mrr` | 0,0058 | 0,2028 |

43/242 câu (18% golden set) đi từ **không hoạt động** sang hoạt động. `TD-11` chỉ
nhích nhóm này lên 2/43 câu; đây là 14/43 ở `@5` và 19/43 ở `@10`.

Sparse cũng cho thấy cùng cơ chế, đo trực tiếp trên cặp câu VI/EN cùng nghĩa:
tích vô hướng sparse **0,168** cho cặp cùng nghĩa vs **0,011** cho cặp khác nghĩa
cùng tiếng Việt vs **0,004** cho cặp không liên quan. Vocab dùng chung của
XLM-R làm `GDP`, `2023`, `05` khớp thẳng qua hai ngôn ngữ.

### 4.3 Theo ngôn ngữ — lần này không triệt tiêu

| lang | n | baseline | bgem3 | Δ | p |
|---|---:|---:|---:|---:|---:|
| `en` | 82 | 0,1707 | 0,5122 | +0,3415 | **< 0,0001** |
| `vi` | 127 | 0,2441 | 0,5669 | +0,3228 | **< 0,0001** |

Khác hẳn `TD-11`, nơi `en` khá lên và `vi` tụt rồi triệt tiêu nhau ở mức tổng.
Ở đây hai ngôn ngữ cùng tăng, và gần bằng nhau.

## 5. Mức tăng này KHÔNG phải công của việc sửa `TD-11`

Đây là chỗ dễ kể sai nhất, nên nói rõ.

Lần chạy này đổi **ba** thứ cùng lúc: model (PhoBERT base → XLM-R large), cửa sổ
(256 → 8192 token), và tính đa ngữ (đơn ngữ → đa ngữ). Bảng số ở §4 không tách
được ba thứ đó.

Nhưng `TD-11` **đã tách phần cửa sổ ra rồi**, và câu trả lời là *gần như không*:

| thí nghiệm | truncation | `hit_rate@5` | kết luận |
|---|---:|---:|---|
| baseline | 56,9% | 0,2153 | — |
| `chunk550` (PhoBERT, hết cắt) | 0,4% | 0,2010 | `p = 0,711` — **không khác** |
| `bgem3` (BGE-M3, hết cắt) | 0,0% | 0,5455 | `p < 0,001` |

Hai dòng dưới **cùng** đưa truncation về ~0. Một dòng không đổi gì, dòng kia
+153%. Vậy biến giải thích không phải truncation — nó là **model**.

Vai trò thật của cửa sổ 8192 là **cho phép giữ `chunk_size=1000`**: không có nó,
muốn hết bị cắt thì phải hạ `chunk_size`, mất ngữ cảnh mỗi vector *và* làm
recall@k không so được nữa. Cửa sổ dài là thứ làm phép đo sạch, không phải thứ
tạo ra mức tăng.

`W2-08` phải có ma trận **hai chiều** `chunk_size` × `embedding` để chốt điều
này bằng số thay vì bằng suy luận từ hai thí nghiệm rời.

### 5.1 Một cảnh báo còn lại về tính khách quan

BGE-M3 được train trên corpus đa ngữ quy mô lớn (gồm MIRACL, mC4) và corpus của
dự án là tài liệu World Bank về Việt Nam — thể loại văn bản mà model rất có thể
đã thấy nhiều. Đây **không** phải rò rỉ tập test (nhãn do LLM sinh từ chunk của
chính corpus này, và chunk giống nhau ở cả hai lần chạy), nhưng nó có nghĩa mức
tăng này chưa chắc giữ nguyên trên corpus đóng của doanh nghiệp. Nói trong
interview thì nói kèm câu này.

Và nhắc lại `TD-13`: `golden_v1` vẫn **review bằng model**. So sánh *tương đối*
giữa hai cấu hình vẫn hợp lệ vì cùng một thước đo và nhãn bit-identical; con số
*tuyệt đối* thì chưa được gọi là "human-verified".

## 6. Chi phí

| | baseline | bgem3 |
|---|---:|---:|
| số chiều | 768 | **1024** (+33%) |
| chunk | 15.814 | 15.814 |
| ký tự embed | 17.745.511 | 17.745.511 |
| thời gian embed+ghi | — | **380,4 s** (39,0 chunk/giây) |
| VRAM lúc index (`batch_size=16`) | — | **~3,3 GB** / 8 GB |
| độ trễ truy hồi p95 | 32,8 ms | **46,0 ms** (+40%) |
| dung lượng vector | ~48,6 MB | ~64,8 MB |

`batch_size` phải hạ 64 → 16: BGE-M3 là XLM-R large (24 lớp, hidden 1024), nặng
hơn PhoBERT base nhiều. Đây là knob tốc độ nên **không** vào `fingerprint` —
chạy trên GPU thuê với batch lớn hơn vẫn ra cùng một index về mặt logic.

+13 ms p95 gần như toàn bộ là chi phí embed truy vấn bằng model lớn hơn, không
phải chi phí tìm kiếm trong Qdrant. Với ngưỡng `G4` (p95 < 3 s toàn tuyến) thì
46 ms còn rất nhiều chỗ, nhưng đây là khoản phải trả thật và `W2-05` (reranker)
sẽ cộng thêm.

## 7. Thứ CHƯA được kiểm chứng đầu-cuối

Nói rõ để không ai đọc report này rồi tưởng hybrid search đã chạy:

* **Sparse vector đã dựng và đã test, nhưng chưa đi vào Qdrant.** Eval ở §4 là
  **dense-only**. `W2-02` mới thêm named vector sparse vào collection, `W2-03`
  mới có retriever dùng nó, `W2-04` mới hợp nhất hai nhánh.
* Nên toàn bộ mức tăng ở §4 là của **dense BGE-M3**. Phần sparse còn nguyên chưa
  tiêu.
* Collection `rag_bgem3` hiện chỉ có named vector `dense` 1024-d. `W2-02` sẽ phải
  build lại collection để thêm sparse — Qdrant không thêm named vector vào
  collection đã tồn tại mà không tạo lại.

## 8. Trạng thái `G2`

| Điều kiện `G2` | Ngưỡng | Đo được | |
|---|---|---|---|
| nDCG@10 tăng so baseline | ≥ +0,08 | **+0,2820** | ✅ gấp 3,5× ngưỡng |
| Mọi dòng ablation có `p`/CI | — | có ở cả 15 metric | ✅ |
| recall@k chỉ dùng khi nhãn/câu khớp | — | 1,3828 = 1,3828 | ✅ |
| Ablation ≥ 12 tổ hợp, 2 chiều | — | chưa | ⬜ `W2-08` |
| Hybrid + rerank | — | chưa | ⬜ `W2-02`…`W2-05` |

## 9. Tái lập

```bash
make truncation BUNDLE=bgem3                  # 0/15814 chunk bị cắt
python -m pipeline.indexing.build_index \n  --config configs/indexing/bgem3.yaml --recreate \n  --report plans/reports/index-bgem3.json   # ~405s trên RTX 4060
make eval-retrieval BUNDLE=bgem3              # ~30s
make eval-compare BASE=baseline CAND=bgem3    # McNemar + bootstrap
make test && make test-gpu                    # 644 unit + 11 gpu
```

⚠️ `--recreate` là bắt buộc: đổi model là đổi số chiều 768 → 1024. Không có nó
thì Qdrant từ chối upsert, hoặc tệ hơn là dùng lại collection cũ và so hai thước
đo khác nhau.

## 10. Bước tiếp

1. **`W2-02` Qdrant named vectors + sparse index.** Sparse đã có, giờ cần chỗ
   chứa. Build lại `rag_bgem3` với cả hai named vector.
2. **`W2-03` sparse retriever**, `W2-04` RRF. Ở đây mới biết sparse của BGE-M3
   đóng góp gì trên số liệu — §4.2 chỉ là dấu hiệu đo trên 3 cặp câu.
3. **`W2-08` ma trận hai chiều** `chunk_size` × `embedding`, để §5 thành số thay
   vì thành suy luận từ hai thí nghiệm rời.
4. `TD-13` vẫn mở, và giờ càng đáng làm: mức tăng lớn thế này thì con số tuyệt
   đối bắt đầu được dùng để kể chuyện, mà nó vẫn dựa trên nhãn review bằng model.
