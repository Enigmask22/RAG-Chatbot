# `W2-05` — Cross-encoder reranker: trần là 0,7799 và ngân sách 400 ms không đủ

**Ngày**: 2026-08-21 · **Nhánh**: `feat/w1-foundation` · **Index**: `rag_bgem3` (15.814 chunk, fingerprint không đổi từ `W2-02`)
**Model**: `BAAI/bge-reranker-v2-m3` (XLM-R large: 24 lớp, hidden 1024, FFN 4096, vocab 250.002) · **GPU**: RTX 4060 Laptop 8GB

---

## 0. Câu hỏi của hạng mục này

`W2-04` kết thúc bằng một kết luận có hình dạng rất cụ thể: **hybrid là bộ sinh
ứng viên tốt và bộ xếp hạng cuối tệ.** `recall@20` đi từ 0,6324 lên 0,6770 có ý
nghĩa thống kê, trong khi `hit_rate@1` **đứng im ở 0,3397** — đúng con số của
dense. Vùng phủ tăng mà thứ hạng không đổi nghĩa là bằng chứng đã có mặt trong
danh sách nhưng không ở trên cùng.

Cross-encoder là công cụ cho đúng chỗ đó, vì lý do kiến trúc chứ không phải vì
nó "mạnh hơn": bi-encoder embed truy vấn và tài liệu **độc lập**, nên nó không
bao giờ nhìn thấy tương tác giữa hai bên; cross-encoder đọc cả cặp trong một
forward pass. Cái giá là mất khả năng cache — bi-encoder embed 15.814 chunk một
lần rồi tìm bằng ANN, còn cross-encoder chạy `candidates` forward pass **cho mỗi
truy vấn**. Đó là lý do nó chỉ dùng được ở tầng thứ hai, và đó cũng là gốc của
phát hiện đắt nhất trong hạng mục này (§5).

---

## 1. Dự đoán ghi trước khi đo

Bảy dự đoán, ghi vào file trước khi chạy một lệnh eval nào. `W2-03` và `W2-04`
đều làm vậy và đều sai; giá trị của việc ghi trước không phải là đúng, mà là
không sửa được sau khi thấy số.

| | Dự đoán | Thực tế | |
|---|---|---|---|
| **D1** trần vùng phủ | `hit_rate@50` 0,74–0,78 | **0,7799** | ✅ sát mép trên |
| **D2** `hit_rate@1` | +8…+15 điểm, **có ý nghĩa** | §6 | |
| **D3** `recall@20` | +0,01…+0,04 | §6 | |
| **D4** known-item | vẫn dưới 0,20, `TD-18` không được xoá | §7 | |
| **D5** độ trễ | 150–300 ms cho 50 cặp | **1794,7 ms** fp32 · **510,1 ms** fp16 | ❌ sai về bậc |
| **D6** bão hoà sigmoid | ≥1% cặp | **0,0%** | ❌ lý lẽ của tôi không đúng |
| **D7** truncation ở 512 | dưới 5% | **0,008%** (1/12.100) | ✅ đúng xa hơn dự đoán |
| **D5b** fp16 speedup | 1,7–2,2× | **3,52×** | ❌ vẫn thấp, cùng chiều sai với D5 |

---

## 2. Trần vùng phủ — con số làm mọi phát biểu sau này có nghĩa

Reranker chỉ xếp lại những gì nhánh nền đưa cho. Chunk đúng không nằm trong pool
thì không có phép xếp nào cứu được, nên `hit_rate@1` sau rerank **bị chặn trên
bởi `hit_rate@candidates` của nhánh nền**. `retrieval_eval` chỉ chấm ở
`k ∈ {1, 5, 10, 20}` nên nó không nói được trần ở `pool = 50` — đó là lý do
`scripts/rerank_probe.py` tồn tại.

Nhánh nền `qdrant-hybrid:rag_bgem3:rrf1-c20` (cấu hình thắng của `W2-04`),
242 câu, 209 câu có nhãn:

| depth | 1 | 5 | 10 | 20 | **50** |
|---|---:|---:|---:|---:|---:|
| `hit_rate` | 0,3397 | 0,5742 | 0,6555 | 0,7129 | **0,7799** |

⚠️ Các cột ≤ 20 ở đây **không** trùng khít số của `bgem3-rrf-k1-c20` trong
`W2-04` (`hit_rate@20` 0,7177 vs 0,7129 ở đây), và chênh lệch đó không phải
nhiễu. Bảng này lấy pool **50** rồi cắt xuống N, còn `W2-04` gọi nhánh nền với
`top_k = N`; với hybrid thì `top_k` đi vào `_depth` nên **độ sâu pool hợp nhất
khác nhau**, và RRF thấy một tập ứng viên khác. Đúng cái tương tác đã ghi trong
docstring của `RerankedRetriever._depth` và có test canh ở tầng dây
(`test_candidates_deepens_the_hybrid_fusion_pool_too`). Chênh 0,0048 = **1 câu**,
tức nó cũng không phản chứng kết luận "`candidate_k` không có tác dụng ở `k=1`"
của `W2-04` — nó chỉ nói kết luận đó đúng ở độ phân giải mà `golden_v1` cho phép.
Cột duy nhất dùng để nói về trần là **@50**, và nó được đo ở đúng độ sâu mà
nhánh reranked thật sự dùng.

**Dư địa là 44 điểm**, từ 0,3397 lên 0,7799. Đây là mẫu số phải nhớ khi đọc §6:
"+5 điểm" nghĩa là 5 trên 44, không phải 5 trên 66. Và nó cũng nói ngay một điều
về giới hạn: **22% câu hỏi không có bằng chứng nào trong 50 ứng viên đầu**, tức
1/5 golden set nằm ngoài tầm với của mọi reranker, mọi `k`, mọi trọng số. Phần
đó thuộc `W3` (chất lượng parse) và `TD-18`, không thuộc tầng xếp hạng.

Đường cong cũng đáng đọc: từ 20 lên 50 chỉ thêm 6,7 điểm, trong khi từ 1 lên 5
thêm 23,4 điểm. Pool sâu hơn mua vùng phủ với hiệu suất giảm dần — và §5 cho
thấy nó bán độ trễ với hiệu suất **tăng** dần theo tuyến tính.

---

## 3. Truncation ở `max_length=512` — đo, không tin

`TD-11` là cả một tuần đi sai hướng vì một giả định về truncation không được đo,
nên `CrossEncoderReranker.count_pair_tokens` có mặt từ đầu, với `truncation=False`
tường minh. (Mặc định của tokenizer là **cắt** ở `model_max_length`, và khi đó
phép đo "bao nhiêu phần trăm bị cắt" trả về hằng số 0 cho mọi corpus — cùng cái
bẫy đã ghi ở `HuggingFaceEmbeddingProvider.count_tokens`.)

12.100 cặp (242 câu × pool 50):

| | p50 | p95 | max | vượt 512 |
|---|---:|---:|---:|---:|
| token mỗi cặp | 287 | 352 | 542 | **1 cặp** (0,008%) |

Cửa sổ của model là 8192 nhưng `max_length=512` **không phải một đánh đổi** ở
corpus này — nó là dư. Và vì chi phí attention là bậc hai theo độ dài, hạ trần
xuống là cách rẻ nhất để mua tốc độ *nếu* nó cắt gì; ở đây nó không cắt gì, nên
nó cũng không mua được gì. Con số này đóng một hướng tối ưu trước khi tôi kịp
mất thời gian vào nó.

⚠️ `max_length` vẫn nằm trong `CrossEncoderReranker.name`, và có test canh việc
nó **thật sự đổi điểm** (`max_length=32` cho điểm khác 512). Cùng lý lẽ với
`IndexConfig.fingerprint` (`W1-06`): tham số cắt nội dung thì đổi kết quả, nên nó
phải xuất hiện trong nhãn. `batch_size` thì không — nó chỉ đổi tốc độ, và có test
canh `batch_size=1` với `batch_size=64` cho cùng số.

---

## 4. Sigmoid không bão hoà — một lý lẽ của tôi bị phản chứng

Mặc định của `CrossEncoderReranker` là **logit thô** (`activation="none"`), và lý
do tôi viết vào docstring là: "sigmoid bão hoà — `float32` cho `sigmoid(x) == 1.0`
từ khoảng `x > 17`, biến các điểm khác nhau thành ties nhân tạo ở đúng chỗ quan
trọng nhất, top của danh sách."

Đo 2000 logit thật (40 câu × pool 50):

| min | p50 | max | vượt ngưỡng bão hoà trên (+16,64) | dưới (−103) |
|---:|---:|---:|---:|---:|
| −10,873 | −1,652 | +8,668 | **0,0%** | **0,0%** |

Toàn bộ phân bố nằm gọn trong vùng sigmoid còn phân biệt tốt. **Lý lẽ của tôi
không đúng với model này trên corpus này.** Nó không phải sai về nguyên lý —
`sigmoid` *có* bão hoà ở float32 — nhưng tôi đã dùng một sự thật về kiểu số để
kết luận về một phân bố mà tôi chưa đo.

Mặc định logit thô **vẫn giữ**, với lý do yếu hơn và đúng: sigmoid đơn điệu nên
nó không thêm gì cho việc xếp hạng, và bỏ một phép biến đổi là bỏ một chỗ có thể
sai. Docstring đã được sửa lại cho khớp. Test `test_sigmoid_is_monotone_so_the_order_is_identical`
giữ nguyên và giờ nó có thêm một nghĩa: nếu nó đỏ thì hoặc phân bố logit đã đổi
(corpus/model khác), hoặc `activation` đang bị áp hai lần.

---

## 5. Độ trễ — DoD 400 ms **không đạt**, và lý do là số học

DoD của `W2-05` viết: *"rerank 50 → 6 trong < 400ms trên GPU; có CPU fallback"*.
Nó được viết ở `W0` khi lập kế hoạch, trước khi có bất kỳ phép đo nào.

### 5.1 Dự đoán của tôi sai về **bậc**, không phải về biên

Tôi ghi trước: 150–300 ms. Đo được (fp32, corpus thật, pool 50): **1794,7 ms**.
Sai gấp 6–12 lần. Nguyên nhân của cái sai đó đáng ghi lại hơn con số: tôi đã
nghĩ về reranker như **một** forward pass, trong khi nó là **`candidates`**
forward pass. Đó chính là tính chất đã viết trong docstring của `Reranker` —
"cross-encoder phải chạy `len(candidates)` forward pass cho mỗi truy vấn" — nên
tôi đã viết đúng cơ chế rồi vẫn dự đoán như thể nó không tồn tại.

Số lượng phép tính nói ngay ra điều đó. Encoder XLM-R large, bỏ embedding:
24 lớp × (4 × 1024² attention + 2 × 1024 × 4096 FFN) = **302,0M tham số**.
Pool 50 × 287 token (p50 ở §3) = 14.350 token. Chi phí matmul:
`2 × 302,0e6 × 14.350` = **8,67 TFLOP** cho **một** truy vấn. Đó là con số lẽ ra
phải tính trước khi đoán.

### 5.2 fp16 lấy 3,52× — nhiều hơn tôi đoán, và gần như không đổi thứ hạng

60 câu × pool 50, một tiến trình, xả model giữa hai lượt, không có gì khác chạm GPU:

| | p50 | | |
|---|---:|---|---|
| fp32 | 1794,7 ms | | |
| **fp16** | **510,1 ms** | **3,52×** | |

Và mức đổi kết quả:

| trùng top-1 | trùng thứ tự top-10 | top-20 | vị trí lệch trong top-10 (TB) | lệch điểm max |
|---:|---:|---:|---:|---:|
| **98,3%** (59/60) | 95,0% | 91,7% | **0,067** | 0,0154 (**0,08%** khoảng logit) |

Nên `dtype="auto"` (= fp16 trên CUDA, fp32 ở nơi khác) là mặc định, và nó nằm
trong `CrossEncoderReranker.name` để không có lần chạy nào không biết mình đã
chạy ở độ chính xác nào. 8,67 TFLOP trong 510 ms = **17,0 TFLOPS hiệu dụng**.

⚠️ **Tôi đã quy sai nguyên nhân một lần và sửa lại ở đây.** Bản đo đầu tiên
(text tự sinh) cho 1278,7 ms; bản corpus thật cho 1869,7 ms trong một lần chạy
**bị nhiễm** (một tiến trình pytest thứ hai nạp bge-m3, VRAM lên 7934/8188 MiB,
`max_ms` chạm **171.759 ms**). Tôi kết luận rằng chênh lệch là do nhiễm. Sai:
lần đo sạch ở trên cho 1794,7 ms. Con số "nhanh" 1278,7 ms là số **lạc quan** vì
text tự sinh dài đều nhau nên batch không có padding thừa — corpus thật lệch độ
dài, và đó là chi phí thật. Số bị nhiễm vẫn không dùng được, nhưng lý do nó khác
không phải lý do tôi nói.

### 5.3 `batch_size` không mua được gì — và cơ chế giải thích tại sao

40 câu, fp16, p50 ms:

| `batch_size` | pool 20 | pool 30 | pool 40 | pool 50 |
|---:|---:|---:|---:|---:|
| 8 | 212,4 | 317,0 | 420,0 | 526,6 |
| **16** | 213,7 | **315,0** | **420,0** | **524,4** |
| 32 | 212,7 | 324,0 | 434,8 | 539,8 |
| 64 | 213,5 | 327,1 | 441,9 | 551,1 |

Từ 8 đến 16 là nhiễu; từ 32 lên 64 thì **tệ dần**, đều đặn ở mọi pool. Cơ chế:
`CrossEncoder.predict` của sentence-transformers gom batch theo **độ dài đã sắp**,
nên batch nhỏ chỉ pad tới độ dài lớn nhất *trong batch đó*. Batch 64 với pool 50
nhét cả 50 cặp vào một batch và pad tất cả tới cặp dài nhất — tối đa hoá phần
tính toán vô ích. GPU đã bão hoà từ batch 8 với chuỗi ~300 token, nên batch lớn
hơn không thêm song song, chỉ thêm padding.

Đây cũng là bằng chứng thứ hai cho việc **`batch_size` không thuộc `name`**: nó
không đổi kết quả (có test canh), và giờ biết là nó gần như không đổi cả tốc độ.

### 5.4 Kết luận về DoD: không đạt, và đây là ngân sách thật

Độ trễ tuyến tính theo pool, khớp rất chặt: **`≈ 10,4 ms × pool + 5 ms`**.

| ngân sách | pool vừa | |
|---|---:|---|
| 400 ms (DoD) | **38** | 50 cặp tốn 524 ms — **vượt 31%** |
| 220 ms | 20 | |

**DoD như đã viết không đạt trên RTX 4060 Laptop.** Tôi không sửa lại DoD cho
khớp số đo; tôi ghi rằng nó không đạt và ghi cái gì thì đạt. Ba cần điều khiển đã
thử và cái nào cũng đã cạn:

* `max_length` — **đã cạn**: §3 đo được nó chỉ cắt 1/12.100 cặp, nên hạ trần
  không giảm được tính toán thật.
* `dtype` — **đã dùng hết**: fp16 cho 3,52×; bf16 không nhanh hơn fp16 trên Ada.
* `batch_size` — **không có gì để lấy**: §5.3.

Còn lại đúng một cần: **`candidates`**, và nó không phải một phép tối ưu — nó là
**trả bằng vùng phủ**. §2 đã cho giá: pool 50 → 20 làm trần `hit_rate` tụt từ
0,7799 xuống 0,7129 (−6,7 điểm). §6 nói phần đó có thành mất mát thật hay không.

### 5.5 CPU fallback: có, và chậm 38×

DoD cũng đòi "có CPU fallback". Nó có — `resolve_device` dùng lại của tầng
embedding, nên không có CUDA thì trả `"cpu"` thay vì nổ, và `dtype="auto"` cố ý
**không** chọn fp16 ở đó (phần lớn CPU không có kernel fp16 nên PyTorch lùi về
emulate và chạy *chậm hơn* fp32 — nhanh trên GPU không suy ra nhanh ở mọi nơi).

| pool | GPU fp16 | CPU fp32 | chậm hơn |
|---:|---:|---:|---:|
| 6 | ~65 ms | 2.288 ms | 35× |
| 20 | 213 ms | 7.790 ms | 37× |
| 50 | 510 ms | **19.446 ms** | **38×** |

19,4 **giây** cho một truy vấn. Nên phát biểu đúng về CPU fallback là: nó tồn tại
để code **chạy đúng** ở nơi không có GPU (CI, máy dev, container không có driver),
không phải để phục vụ. Nếu serving mất GPU thì cách đúng là **tắt tầng rerank** và
lùi về hybrid ở 31 ms, chứ không phải chạy reranker trên CPU và để truy vấn treo
20 giây. Đó là một quyết định của `W4`, và nó có số để dựa vào.

(Đo trên text tự sinh dài đều nhau — xem cảnh báo ở §5.2, loại đầu vào này cho số
*lạc quan* hơn corpus thật.)

---

## 6. Chất lượng — mức cải thiện lớn nhất của W2 sau chính BGE-M3

Nhánh nền `hybrid k=1, candidate_k=20` (cấu hình thắng `W2-04`), fp16,
`golden_v1` 209 câu có nhãn:

| run | hit@1 | hit@5 | hit@10 | hit@20 | MRR | nDCG@10 | MAP@20 | recall@20 | p50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 0,3397 | 0,5455 | 0,6268 | 0,6746 | 0,4394 | 0,4442 | 0,3853 | 0,6324 | 30,3 |
| hybrid k=1 c=20 | 0,3397 | 0,5742 | 0,6555 | 0,7177 | 0,4436 | 0,4563 | 0,3996 | 0,6770 | 31,3 |
| **rerank c=20** | **0,5407** | 0,7033 | 0,7129 | 0,7177 | 0,6128 | 0,6075 | 0,5698 | 0,6770 | 232,8 |
| **rerank c=50** | **0,5598** | 0,7512 | 0,7703 | 0,7751 | 0,6440 | 0,6481 | 0,6051 | 0,7424 | 534,4 |
| rerank c=100 | 0,5789 | 0,7895 | 0,8134 | **0,8182** | 0,6694 | 0,6736 | 0,6274 | 0,7775 | 1044,3 |
| rerank c=50, nền **dense** | 0,5455 | 0,7273 | 0,7416 | 0,7464 | 0,6265 | 0,6268 | 0,5894 | 0,7121 | 538,0 |

`hit_rate@1` **0,3397 → 0,5598 = +22,0 điểm**. Đặt cạnh trần ở §2: dư địa là 44
điểm, nên reranker lấy được **đúng một nửa phần còn lấy được**. nDCG@10 +42%
tương đối. Đây là mức nhảy lớn nhất của W2 sau `W2-01` (BGE-M3).

Và `hit_rate@20` của `rerank c=50` là **0,7751** so với trần `hit_rate@50` =
0,7799 — tức **99,4%** những câu mà bằng chứng *có mặt đâu đó trong 50 ứng viên*
thì nó đã nằm trong top-20. Sau con số đó, làm pool sâu hơn không còn là chuyện
xếp hạng nữa; nó là chuyện `hit_rate@pool` của nhánh nền, tức `W3` và `TD-18`.

### 6.1 Một phép kiểm nội tại tình cờ, và nó rất mạnh

`rerank c=20` cho `hit_rate@20` **0,7177** và `recall@20` **0,6770** — **trùng
khít từng chữ số** với hybrid c=20. Đúng như bắt buộc phải vậy: với
`candidates=20` và `top_k=20`, nhánh reranked trả về **cùng một tập**, chỉ khác
thứ tự. Metric theo *tập* (`hit_rate@20`, `recall@20`) phải bất biến; metric theo
*hạng* thì đổi — và `hit_rate@1` nhảy 0,3397 → 0,5407 **thuần do đảo thứ tự**.

Phép kiểm này không được thiết kế; nó rơi ra từ bảng. Nhưng nó canh đúng chỗ dễ
sai nhất của cài đặt: `RerankedRetriever.retrieve` dựng lại `RetrievedChunk` từ
đầu (chunk, rank, mode, ba trường điểm). Gán lệch chunk cho điểm, đánh rơi một
ứng viên, hay off-by-one ở `rank` đều sẽ làm hai con số @20 kia lệch đi. Chúng
không lệch.

### 6.2 `candidates`: 20 lấy 91% mức lợi với 44% chi phí

| `candidates` | hit@1 | nDCG@10 | recall@20 | p50 ms | phần lợi hit@1 giữ được |
|---:|---:|---:|---:|---:|---:|
| 20 | 0,5407 | 0,6075 | 0,6770 | 232,8 | **91,4%** |
| 50 | 0,5598 | 0,6481 | 0,7424 | 534,4 | 100% |
| 100 | 0,5789 | 0,6736 | 0,7775 | 1044,3 | 108,7% |

Mỗi lần nhân đôi pool: `hit_rate@1` thêm **đúng 1,9 điểm** (0,5407 → 0,5598 →
0,5789) trong khi độ trễ cũng nhân đôi. Nhưng `hit_rate@20` thì thêm 5,7 rồi 4,3
điểm — hai metric phản ứng rất khác nhau, đúng như cơ chế: pool sâu hơn nhập thêm
ứng viên đúng từ hạng sâu vào *danh sách*, còn việc đưa cái đúng nhất lên *đầu*
thì đã làm được từ pool 20.

`hit_rate@20` ở c=100 là **0,8182**, tức nó **vượt trần `hit_rate@50` = 0,7799**
của §2. Đó không phải nghịch lý — trần ở §2 là trần *của pool 50*. Nó là bằng
chứng trực tiếp rằng pool sâu đang nhập thêm vùng phủ, không chỉ đảo thứ tự.

### 6.3 Kiểm định — 15/15 có ý nghĩa, và McNemar lệch một chiều

`pipeline/eval/compare.py`, McNemar exact + paired bootstrap 10.000 vòng, seed 20260820:

| so sánh | `hit_rate@1` | `hit_rate@5` | `hit_rate@10` | `recall@20` | số metric có ý nghĩa |
|---|---|---|---|---|---:|
| dense → rerank c50 | +0,2201 · 9↔55 | +0,2057 · **0↔43** | +0,1435 · **0↔30** | +0,1100 | **15/15** |
| hybrid k=1 → rerank c50 | +0,2201 · **7↔53** | +0,1770 · 2↔39 | +0,1148 · 1↔25 | +0,0654 | **15/15** |

`hit_rate@5` từ dense: **43 câu được sửa, 0 câu bị làm hỏng.** Khác hẳn `W2-04`
nơi chỉ 3/15 metric đạt ý nghĩa và mức tăng nằm dưới ngưỡng phân giải của
`golden_v1`; ở đây mọi thứ vượt xa ngưỡng đó.

⚠️ Vẫn giữ nguyên cảnh báo của `W2-04`: **"15/15 đều dương" không phải 15 phép
thử độc lập** — các metric này tương quan mạnh. Điều làm kết quả này đáng tin
không phải số 15, mà là các bảng discordance: 0↔43 và 0↔30 là loại số không xuất
hiện từ nhiễu.

**Pool 20 → 50 → 100 tách được hai chuyện khác nhau:**

| | `hit_rate@1` | `hit_rate@10` | `recall@20` |
|---|---|---|---|
| c20 → c50 | +0,0191 · **p=0,219 (nhiễu)** | +0,0574 · p<0,001 | +0,0654 có ý nghĩa |
| c50 → c100 | +0,0191 · **p=0,125 (nhiễu)** | +0,0431 · p=0,004 | +0,0351 có ý nghĩa |

Pool sâu hơn mua **chất lượng danh sách**, không mua **chất lượng hạng nhất** —
và đó là kiểm định, không phải lập luận. Cơ chế khớp: từ pool 20 trở lên,
reranker đã đưa được cái đúng nhất *lên đầu* rồi; thêm ứng viên chỉ nhập thêm
những chunk đúng khác vào *danh sách*.

### 6.4 ⭐ Sau khi có reranker, tầng hybrid không còn đo được

Đây là kết quả kiến trúc đáng nhất của hạng mục, và nó là loại kết quả rất dễ
không đi tìm.

| nhánh nền (cùng reranker, cùng c=50) | hit@1 | hit@10 | nDCG@10 | recall@20 | p50 |
|---|---:|---:|---:|---:|---:|
| dense | 0,5455 | 0,7416 | 0,6268 | 0,7121 | 538,0 ms |
| hybrid k=1 c=20 | 0,5598 | 0,7703 | 0,6481 | 0,7424 | 534,4 ms |

`hit_rate@1` `p = 0,453` (2↔5) · `hit_rate@10` `p = 0,180` (4↔10) · `recall@20`
CI95 **[−0,0008, +0,0630]** — **13/15 metric trong ngưỡng nhiễu.**

Ở `W2-04`, hybrid cho `recall@20` hơn dense **có ý nghĩa** (+0,0446). Sau khi
thêm reranker, phần đóng góp đó không còn phát hiện được. Cách đọc đúng: hai
tầng đang sửa **cùng một khuyết điểm** của dense (thứ hạng), nên chúng chồng lên
nhau; reranker sửa mạnh hơn nên nó hấp thụ gần hết phần hybrid mang lại.

⚠️ **Nhưng đây không phải lý lẽ để bỏ hybrid**, và ba lý do đều là số đo:

1. **Hybrid gần như miễn phí**: 534,4 ms so với 538,0 ms — chi phí hợp nhất nằm
   trong nhiễu, vì `W2-04` đã đo Qdrant chạy hai nhánh song song trong một
   request (30,2 ms = bằng dense một mình).
2. **Cả 15 metric vẫn cùng chiều về phía hybrid.** "Không có ý nghĩa" ở đây nghĩa
   là `golden_v1` với 209 câu không phân giải được, không phải là bằng nhau.
3. **`golden_v1` không đo được thứ hybrid/sparse thật sự mang lại.** 209 câu đều
   là câu hỏi tự nhiên; chỗ sparse thắng áp đảo là tra mã tài liệu (`W2-03`:
   hit@10 0,0784 → 0,5098), và đó là §7.

Cái đúng để kết luận: **thứ tự ưu tiên của hai tầng là rõ ràng.** Nếu chỉ được
chọn một, chọn reranker (+22,0 điểm `hit_rate@1`, 15/15 có ý nghĩa) chứ không
chọn hybrid (+0,0 điểm `hit_rate@1`, 3/15 có ý nghĩa). Giữ cả hai vì tầng thứ hai
miễn phí, không vì nó đo được.

### 6.5 Một hố im lặng trong `compare.py` mà hạng mục này lộ ra

`precision@1` **bằng `hit_rate@1` từng chữ số** — top-1 chỉ có một chỗ nên nó liên
quan hay không là 0/1. Nhưng `BINARY_PREFIXES` chỉ có `"hit_rate@"`, nên hai
metric giống nhau đi hai đường kiểm định khác nhau. So `c50` với `c100` làm nó
hiện ra: cùng con số 0,5598 → 0,5789, McNemar cho `p = 0,125` (0↔4 câu đổi chiều
— bốn lần tung xu cùng mặt thì không kết luận được gì) còn bootstrap cho CI95
`[+0,0048, +0,0383]`, tức "khác biệt thật". Người đọc bảng sẽ trích dòng nào thuận
với mình.

Đã sửa: `precision@1` đi McNemar. Và **bản sửa đầu của tôi tự tạo một bug mới** —
tôi thêm `"precision@1"` vào `BINARY_PREFIXES`, và `"precision@10".startswith(
"precision@1")` là `True`, nên `precision@10` (nhận 0; 0,1; 0,2…) bị đẩy sang
McNemar ngay ở lần chạy tiếp theo. Tách thành `BINARY_METRICS` khớp **đúng tên**.
Có hai test, và test thứ hai canh **chính cái bẫy tiền tố** chứ không canh cách sửa.

---

## 7. ⭐ Known-item: D4 sai, và `TD-18` là bài toán **truy hồi**, không phải biểu diễn

Dự đoán D4 của tôi: *"Cross-encoder cũng là model subword BGE, nên nó gặp đúng
vấn đề của `TD-18`: `P171645` bị xé thành `['▁P','171','645']`. Tôi đoán hit@1
known-item sau rerank **vẫn dưới 0,20**, tức nợ `TD-18` không được xoá."*

51 mã tài liệu lạ (xuất hiện ở 1–3 chunk), tiêu chí đúng = **chuỗi truy vấn có
mặt nguyên văn trong nội dung chunk** (không cần nhãn người), seed 20260820:

| nhánh | hit@10 | hit@1 | MRR | hạng trung vị |
|---|---:|---:|---:|---:|
| dense | 0,0784 | 0,0196 | 0,0276 | 6,5 |
| sparse | 0,5098 | 0,3529 | 0,4134 | 1,0 |
| hybrid `k=1 c=20` | 0,4706 | 0,0980 | 0,2500 | 2,0 |
| **reranked** (nền hybrid, c=50) | **0,6471** | **0,5490** | **0,5840** | **1,0** |

| | chỉ nhánh kia | chỉ reranked | McNemar |
|---|---:|---:|---|
| reranked vs hybrid | 1 | **10** | `p = 0,0117` |
| reranked vs sparse | 1 | **8** | `p = 0,0391` |

**Reranked thắng cả sparse** — nhánh vốn thống trị bài toán này từ `W2-03` — và
thắng có ý nghĩa. `hit_rate@1` 0,3529 → 0,5490.

### Vì sao lý lẽ của tôi sai

Tôi lẫn hai việc: **truy hồi** một mã và **nhận ra** một mã. Vocab subword phá
việc thứ nhất chứ không phá việc thứ hai.

Chấm điểm sparse là một tích vô hướng trên **túi** subword: `['▁P','171','645']`
mất hết thông tin về **thứ tự và liền kề**, nên một chunk chứa `171` ở một chỗ và
`645` ở chỗ khác ghi điểm y như chunk chứa đúng `P171645`. Trên 15.814 chunk tài
liệu thống kê thì loại chunk thứ nhất có rất nhiều.

Cross-encoder không chấm bằng túi. Nó có attention đầy đủ trên **cả cặp**, nên nó
thấy được ba mảnh subword ấy xuất hiện **liền nhau, đúng thứ tự** ở cả truy vấn
và tài liệu. Đó là một bài toán dễ hơn hẳn. Điều kiện duy nhất là chunk phải **có
trong pool**, và pool 50 sâu hơn top-10 mà `W2-03` đã đo.

Con số xác nhận đúng cơ chế đó: reranked tìm ra **33/51 mã**, trong khi hợp của
dense và sparse ở top-10 (`W2-03`) chỉ có 26/51. **7 mã được cứu từ vùng sâu của
pool** — chúng vẫn được truy hồi ra, chỉ là bị xếp hạng sai, đúng như `W2-03` đã
kết luận ("giới hạn thật là xếp hạng sai, không phải không trả gì").

### `TD-18` thu hẹp nhưng không đóng

18/51 mã (**35%**) vẫn không tìm ra ở top-10, vì chúng không có trong top-50 của
nhánh nền — reranker không thể xếp cái nó không được thấy (§2). Nên `TD-18` vẫn
mở, nhưng phát biểu của nó phải sửa:

* ❌ cũ: *"RRF (`W2-04`) lẫn reranker (`W2-05`) đều không sửa được — cả hai chỉ xếp
  lại thứ tự những gì đã được tìm ra."*
* ✅ mới: **reranker sửa được phần lớn** (hit@1 0,0980 → 0,5490 so với nhánh nền,
  và vượt cả sparse). Phần còn lại là 35% mã **không vào được pool 50**, và đó
  đúng là chỗ cần một nhánh khớp đúng — filter payload khớp chuỗi (rẻ) hoặc BM25
  mức từ (đắt, cần đổi schema).

Nói cách khác: `TD-18` từ "cần một nhánh truy hồi mới" thu lại thành "cần một
nhánh truy hồi mới **cho 35% ca còn lại**", và giá của việc không làm nó đã giảm
mạnh. Đó là một thay đổi về ưu tiên, không phải một nợ được xoá.

---

## 8. Điểm vận hành nên chọn

Ba cấu hình, cả ba đều đo thật, và chúng phục vụ ba mục đích khác nhau:

| | `candidates` | hit@1 | hit@10 | recall@20 | p50 | dùng khi |
|---|---:|---:|---:|---:|---:|---|
| **A** | 20 | 0,5407 | 0,7129 | 0,6770 | **233 ms** | có ngân sách độ trễ; lấy 91% mức lợi hạng nhất |
| **B** | 50 | 0,5598 | 0,7703 | 0,7424 | 534 ms | cân bằng; là mặc định của thư viện |
| **C** | 100 | 0,5789 | **0,8182** (@20) | 0,7775 | 1044 ms | offline / batch, hoặc khi generator cần vùng phủ |

Khuyến nghị cho `W4` (serving): **A** cho đường tương tác. `hit_rate@1` giữa A và
B **không phân biệt được** (`p = 0,219`) trong khi độ trễ hơn gấp đôi, nên với
một trợ lý mà người dùng đọc câu trả lời đầu tiên thì A là chỗ đúng. Chuyển sang
B/C chỉ khi generator thật sự tiêu được 20 chunk và `recall@20` là thứ ràng buộc
chất lượng câu trả lời — điều đó phải đo ở `W4`, không suy từ đây.

⚠️ **`top_n` không phải cần điều khiển của phép đo.** DoD nói "50 → 6", và 6 là
số chunk đi vào prompt. Nhưng đặt `--rerank-top-n 6` rồi chấm `recall@20` thì
`recall@20` bị chặn ở 6 chunk và mọi metric @k > 6 mất nghĩa. Nên mặc định của
`RerankedRetriever.top_n` là `None` (không chặn thêm) và có một test ghim đúng cái
bẫy đó (`test_top_n_below_top_k_is_the_documented_trap`). `top_n` thuộc tầng
serving; nó không xuất hiện trong bất kỳ con số nào của report này.

---

## 9. Kiến trúc và quyết định cài đặt

`RerankedRetriever` bọc **một `Retriever` bất kỳ**, không bọc `QdrantDenseRetriever`.
Đó là lý do `--rerank-base dense|sparse|hybrid` tồn tại và §6.4 đo được — nếu tầng
này gắn cứng vào hybrid thì câu hỏi "reranker có cần hybrid không" sẽ không hỏi
được mà không sửa code.

`build_branch(store, "reranked", ...)` **gọi lại chính nó** cho nhánh nền, nên mọi
phép kiểm tham số của `W2-03`/`W2-04` áp dụng nguyên vẹn ở tầng dưới:
`--rerank-base dense --rrf-k 1` vẫn nổ, vì `k` không có nghĩa với dense. Một lần
chạy ablation gõ sai cờ phải **dừng**, không phải vào bảng `W2-08` như một dòng
hợp lệ đo cái khác với nhãn của nó.

`SUPPORTED_MODES` giờ **bằng** `RetrievalMode`. Hai test cũ khẳng định `reranked`
raise `NotImplementedError` đã **đỏ** ở lần chạy đầu của hạng mục này — đúng như
thiết kế, và là lần thứ hai cơ chế đó hoạt động (`W2-04` với `hybrid`). Chúng được
viết lại thành một test canh chính bất biến `set(SUPPORTED_MODES) == set(RetrievalMode)`,
nên nó sẽ đỏ đúng lúc ai đó thêm mode mới mà quên cài.

**Tie-break: điểm bằng nhau thì giữ thứ tự nhánh nền.** `sorted` của Python là ổn
định và khoá sắp xếp mang cả chỉ số gốc, nên khi cross-encoder không phân biệt
được hai ứng viên thì tiên nghiệm của bộ sinh được giữ. Ties là chuyện thật —
`W2-04` đã gặp với điểm RRF.

**Chết khi có điểm không hữu hạn.** `NaN` so với mọi thứ đều `False` nên `sorted`
trả về một thứ tự **tuỳ ý mà không có gì báo**, và hậu quả trên bảng metric trông
y như "model kém". Đây là chế độ hỏng thật của fp16 (một overflow trung gian là
đủ), nên `retrieve()` raise thay vì xếp bừa. Có test cho `nan`, `+inf`, `-inf`.

**Điểm của nhánh nền được giữ lại.** `dense_score`/`sparse_score` đi qua lượt
rerank nguyên vẹn, để `W2-08` trả lời được "reranker kéo lên chunk mà nhánh nào
tìm ra?".

### Cái gì vào `name`, cái gì không

`reranked[<nền>]:<model>@<device>:L<max_length>[:<dtype>][:<activation>]:n<candidates>[-top<n>]`

Vào: `max_length` (cắt nội dung → đổi điểm, có test), `dtype` (đổi chữ số thấp,
§5.2), `activation`, `candidates`, `top_n`, và **toàn bộ nhãn của nhánh nền**.
Không vào: `batch_size` — nó không đổi kết quả (có test) và §5.3 cho thấy nó gần
như không đổi cả tốc độ. Cùng lý lẽ với `IndexConfig.fingerprint` ở `W1-06`.

⚠️ `dtype="auto"` được **phân giải ngay lúc dựng** thành tên dtype thật, không để
lại chữ "auto" trong `name`: một nhãn ghi "auto" thì đọc log xong vẫn không biết
lần chạy đó ở độ chính xác nào — mà fp32 và fp16 cho điểm khác nhau.

---

## 10. VRAM — số thật, chưa phải `W0-06` chính thức

| trạng thái | VRAM / 8188 MiB |
|---|---:|
| bge-m3 (fp32) một mình | ~3.300 |
| bge-m3 + reranker **fp32** | **5.685** |
| bge-m3 + reranker **fp16** | ~3.900 |
| trên + một tiến trình pytest thứ hai nạp bge-m3 | **7.934** ⚠️ |

Dòng cuối là tai nạn, và nó đắt: một lần đo bị nhiễm tới mức `max_ms` chạm
**171.759 ms** (172 giây cho một lượt rerank 50 cặp) vì máy thrash. Đó là lý do
`_load_cross_encoder` giới hạn `lru_cache(maxsize=2)` chứ không phải 4 như tầng
embedding, và là lý do mọi phép so fp16/fp32 trong report này **xả model giữa hai
lượt** thay vì nạp song song.

Với fp16 làm mặc định thì tầng rerank chỉ thêm ~600 MiB, tức còn dư ~4,3 GB. Điều
đó nói một chuyện có ích cho `W4`: **vẫn không đủ cho một generator LLM local**
(Qwen3-8B ở 4-bit là ~5,5 GB), nên kiến trúc "generator qua API" đứng nguyên.

⚠️ **Và tôi đã làm test suite OOM thật.** `max_length` nằm trong khoá cache của
`_load_cross_encoder`, nên bản `max_length=32` của một test là **model thứ ba**
trong cùng phiên pytest: bge-m3 3,3GB + fp16 1,15GB + fp32 2,3GB + 2,3GB =
**9,05 GB trên một GPU 8,0 GB**. Lần chạy đầy đủ đầu tiên đỏ với
`CUDACachingAllocator ... OOM`, và nó kéo theo 4 test của `test_sparse_retriever.py`
lỗi liên đới — tức một test mới có thể làm đỏ những test không liên quan gì.

Sửa: bản `max_length=32` chạy trên **CPU**. Điều test đó khẳng định (`max_length`
có tới được model không) không phụ thuộc device, và ngưỡng 0,5 cách nhiễu fp32
giữa hai device (~1e-4) nhiều bậc.

Bài học cho `W0-06`, và nó nâng nợ đó từ "nên làm" lên "cần thiết":
`lru_cache(maxsize=2)` cho reranker là một **lời hứa phần cứng này không giữ
được** khi bge-m3 cũng thường trú. `W0-06` phải đo ngân sách theo *tổ hợp model
đồng thời*, không theo từng model một.

---

## 11. Còn lại gì

1. **`W2-06` metadata filter** — việc tiếp theo theo checklist. Nhánh reranked đã
   chuyển tiếp `filters` xuống nhánh nền và có test integration canh việc filter
   được áp **ở Qdrant, trước khi vào pool** (`test_filters_reach_qdrant_not_just_the_wrapper`):
   lọc sau khi rerank thì cross-encoder đã đọc nội dung của tenant khác — tiền đã
   trả và dữ liệu đã bị chạm, dù kết quả cuối trông đúng.
2. **`W2-07`/`W2-08`** — nhánh reranked thêm ba chiều cho ma trận:
   `rerank_base` (§6.4 cho thấy nó có thể không đáng quét trên `golden_v1` nhưng
   đáng quét trên known-item), `candidates` (§6.3: 20/50/100 tách được hai loại
   metric), và `dtype`. **Không** quét `batch_size` (§5.3) và **không** quét
   `max_length` (§3) — cả hai đã đo là không có gì để lấy. Ghi lại để `W2-08`
   không đốt thời gian vào ô trống.
3. **`TD-18` sửa phát biểu** (§7): còn 35% mã không vào được pool 50. Phương án rẻ
   (filter payload khớp chuỗi) phải đo trước phương án đắt (BM25 mức từ).
4. **Trần 0,7799 là câu hỏi của `W3`.** 22% golden set không có bằng chứng trong
   50 ứng viên đầu của nhánh nền. Không tầng xếp hạng nào chạm được phần đó — nó
   thuộc chất lượng parse (`W3-01` Docling, `TD-17`) và `TD-18`.
5. **DoD 400 ms** (§5.4): không đạt trên RTX 4060 Laptop; 400 ms mua được pool 38.
   Khuyến nghị §8 là pool 20 ở 233 ms, vốn *vừa* ngân sách — nhưng đó là đổi bài
   toán, không phải đạt DoD, và hai chuyện đó không được lẫn.
