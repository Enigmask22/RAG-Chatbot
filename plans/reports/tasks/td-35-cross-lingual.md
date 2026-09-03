# `TD-35` — vì sao `cross_lingual` đi ngược, và chỉ sau reranker

> **Ngày:** 2026-09-03 · **Nợ mở từ:** `exp-002-contextual.md` §3
> **Mã:** `scripts/td35_probe.py` · **Dữ liệu:** `probes/td-35.json` (nền hybrid),
> `probes/td-35-dense.json` (nền dense)
> **Câu hỏi:** dán ngữ cảnh làm `cross_lingual` **15/15 metric dương ở dense**
> (0,2538 → 0,3172) nhưng **11/15 âm ở reranked** (0,5191 → 0,5041) — và đó là
> **toàn bộ** hàng âm của bảng 90 hàng.

## 1. Dự đoán ghi TRƯỚC khi đo

| | dự đoán | kết quả |
|---|---|---|
| `D1` | Tỉ lệ cắt của `cross_lingual` **không** cao hơn hẳn các nhóm khác | ✅ **đúng** — 0,558%, hạng 3/7, thấp hơn `factoid` (0,618%) |
| `D2` | Tài liệu tiếng Việt cắt nhiều hơn tiếng Anh 2–4× | ✅ **đúng, còn mạnh hơn** — 0,863% vs 0,067%, **12,9×** |
| `D3` | Trần pool của `cross_lingual` **tăng** sau khi dán ngữ cảnh | ⚠️ **đúng ở dense, SAI ở hybrid** — và chỗ sai chính là câu trả lời |
| `D4` | Rerank cùng pool bằng văn bản đã bóc ngữ cảnh sẽ **lấy lại** phần tụt | ❌ **sai** — chênh lệch chỉ −0,0061 nDCG, `hit_rate` hoà tuyệt đối |
| `D5` | Cơ chế là **pha loãng** văn bản trong cặp cross-encoder | ❌ **sai** — hệ quả trực tiếp của `D4` sai |

**4/5 sai hoặc chỉ đúng một nửa, gồm cả hai dự đoán tôi tự tin nhất.** Và lời
giải thích tôi đã viết vào `exp-002` §3 — truncation — bị phản chứng ở §2.

⚠️ Sửa lại phát biểu cũ: `exp-002` §3 nói *"nguyên nhân nằm ở quãng giữa dense
và reranked"*. **Đúng về vị trí, sai về cơ chế.** Quãng ấy có hai tầng chứ không
một, và thủ phạm là tầng tôi không nhìn tới.

## 2. Truncation: đúng như dự đoán, và **không** liên quan

| nhóm | tỉ lệ cắt (có ngữ cảnh) | Δ nDCG@10 ở exp-002 |
|---|---:|---|
| `factoid` | **0,618%** ← cao nhất | **+0,0554** ← tăng nhiều nhất |
| `multi_hop` | 0,529% | +0,0797 |
| `cross_lingual` | 0,558% | **−0,0150** ← nhóm duy nhất âm |
| `adversarial` | 0,471% | +0,0503 |
| `unanswerable` | 0,364% | — |
| `aggregation` | 0,077% | +0,0374 |

Nhóm cắt **nhiều nhất** là nhóm cải thiện **nhiều nhất**. Không có quan hệ nào ở
đây, kể cả quan hệ ngược.

Theo ngôn ngữ tài liệu thì giả thuyết tokenise lại **đúng rõ rệt**:

| | p50 token | p95 | tỉ lệ cắt |
|---|---:|---:|---:|
| tài liệu `en` | 344 | 411 | **0,067%** |
| tài liệu `vi` | 386 | 460 | **0,863%** |

Tiếng Việt cắt nhiều gấp **12,9×** tiếng Anh — nhưng nó không rơi vào
`cross_lingual` nhiều hơn các nhóm khác, nên nó không giải thích được gì ở đây.
Nó vẫn là một con số đáng giữ: p95 tiếng Việt ở **460/512** nghĩa là biên còn
lại rất mỏng, và bất kỳ lần làm dài chunk nào nữa cũng sẽ đâm vào trần.

## 3. ⭐⭐ Thủ phạm là tầng HYBRID, không phải reranker

Trần pool `hit_rate@50` — reranker không bao giờ xếp được thứ không có trong
pool, nên đây là chặn trên cứng của nó:

| nhóm | dense Δ | sparse Δ | **hybrid Δ** |
|---|---:|---:|---:|
| `factoid` | +0,1029 | +0,0588 | +0,0588 |
| `multi_hop` | +0,1176 | +0,0588 | +0,1471 |
| `aggregation` | +0,0769 | 0,0000 | +0,0385 |
| `adversarial` | +0,0588 | +0,0882 | +0,0294 |
| `table_lookup` | +0,2500 | 0,0000 | 0,0000 |
| **`cross_lingual`** | **+0,0465** | 0,0000 | **−0,0233** |

⭐ **Một ô âm duy nhất trong 18 ô, và nó là giao của `cross_lingual` với
`hybrid`.** Dán ngữ cảnh làm pool dense của `cross_lingual` **tốt lên**
(0,6279 → 0,6744), làm pool sparse **đứng yên**, và làm pool **hợp nhất** của hai
cái đó **tệ đi** (0,6047 → 0,5814).

### Vì sao hợp nhất hai thứ không xấu đi lại cho ra thứ xấu hơn

Con số nói ra ngay: **sparse tìm được 1/43 câu `cross_lingual`** (`hit_rate@50` =
0,0233). Nhánh ấy về cơ bản không biết gì về nhóm này — nhưng RRF `k=1` vẫn cho
hạng 1 của nó trọng số ½, lớn hơn mọi hạng từ 2 trở xuống của nhánh dense.

Dán ngữ cảnh thêm 32% văn bản tiếng Việt vào mỗi chunk, tức thêm **khối lượng từ
vựng để sparse bắt trúng**. Sparse không tìm ra thêm câu đúng nào, nó chỉ **tự
tin hơn vào những chunk sai** — và RRF `k=1` đẩy đúng những chunk ấy lên, hất
những chunk dense vừa mới tìm được ra khỏi pool 50.

Reranker sau đó làm đúng việc của nó trên một tập ứng viên đã tệ hơn.

### Ablation quyết định: reranker được minh oan

Xếp lại **đúng cùng một pool** bằng hai phiên bản văn bản của chính nó (bóc câu
ngữ cảnh ra bằng `original_content`, giữ nguyên tập ứng viên):

| nhóm | nDCG@10 có ngữ cảnh | bóc ngữ cảnh | Δ |
|---|---:|---:|---:|
| `adversarial` | 0,6235 | 0,5698 | **+0,0537** |
| `aggregation` | 0,6797 | 0,6462 | +0,0335 |
| `factoid` | 0,8150 | 0,7866 | +0,0284 |
| `multi_hop` | 0,7647 | 0,7479 | +0,0168 |
| **`cross_lingual`** | 0,5041 | 0,5102 | **−0,0061** |
| *toàn bộ* | 0,6888 | 0,6652 | +0,0236 |

`hit_rate@1` và `hit_rate@5` của `cross_lingual` **hoà tuyệt đối** (0,4419 và
0,5581 ở cả hai arm). Nghĩa là văn bản đã dán không giúp cross-encoder ở nhóm
này, nhưng cũng **không làm hại** — −0,0061 nhỏ hơn nhiều so với −0,0150 cần
giải thích, và ngược dấu với mọi nhóm khác chỉ ở mức nhiễu.

**Chốt kiểm tính hợp lệ:** arm `enriched` tái lập số của lượt eval đã công bố
**đúng tới chữ số thứ sáu** — nDCG@10 toàn bộ 0,688824 vs 0,688824, và
`cross_lingual` 0,504070 vs 0,504070. Nên đây đúng là hệ thống đã báo cáo, không
phải một hệ thống gần giống.

⚠️ Lượt đo đầu tiên chạy nền **dense** trong khi `exp-002` chạy nền **hybrid**,
và chênh lệch giữa hai nền (0,5669 vs 0,5041 trên `cross_lingual`) **lớn hơn cả
hiện tượng đang đi tìm**. Nếu dừng ở đó thì báo cáo này đã kết luận về một hệ
thống khác với hệ thống được báo cáo. Đó là lý do `_build` giờ nhận `--base` kèm
docstring nói vì sao.

## 4. ⭐ Đây là lần thứ TƯ cùng một phát hiện xuất hiện

| lúc nào | phát biểu |
|---|---|
| `W2-04` | nhánh hybrid làm tụt `cross_lingual` |
| `W2-08-prep` | sống sót hiệu chỉnh Bonferroni 90 phép kiểm: nDCG@10 CI99,94% [−0,1630, −0,0054] |
| `W2-09` | bậc hybrid là **bậc duy nhất** trên cả đường tới đỉnh bảng làm một nhóm tệ đi — **17 câu tệ đi vs 1 câu tốt lên** |
| `W2-05` | hệ quả vận hành: mất GPU thì lùi về **dense**, không phải hybrid |

Contextual Retrieval **không tạo ra vấn đề mới** — nó làm to một vấn đề đã đo
được ba lần, bằng cách cho nhánh sparse thêm chữ để sai một cách tự tin hơn.

Điều đó cũng có nghĩa là dự đoán `D3` sai vì tôi đọc `exp-002` §3 mà không đọc
`W2-09` — tài liệu đã ghi sẵn đúng câu trả lời, chỉ là ở tầng khác.

## 5. Hệ quả có thể làm ngay: định tuyến theo nhóm truy vấn

Cùng collection có ngữ cảnh, chỉ đổi nhánh nền của reranker:

| nhóm | n | nền hybrid | nền dense | Δ |
|---|---:|---:|---:|---:|
| **`cross_lingual`** | 43 | 0,5041 | **0,5669** | **+0,0628** |
| `aggregation` | 26 | 0,6797 | 0,7015 | +0,0217 |
| `multi_hop` | 34 | 0,7647 | 0,7650 | +0,0004 |
| `adversarial` | 34 | 0,6235 | 0,6235 | 0,0000 |
| `table_lookup` | 4 | 0,5000 | 0,5000 | 0,0000 |
| **`factoid`** | 68 | **0,8150** | 0,7817 | **−0,0333** |
| *toàn bộ* | 209 | 0,6888 | 0,6937 | +0,0049 |

Hai nhóm kéo ngược nhau và bảng tổng gần như hoà (+0,0049) — tức **chọn một nền
duy nhất là bỏ đi một trong hai**. Đây đúng là hình dạng bài toán mà `W4-07`
(routing) tồn tại để giải, và giờ nó có một con số cụ thể để nhắm: +0,0628 cho
`cross_lingual` nếu định tuyến đúng, đổi lấy việc phải phân loại được truy vấn.

### Kiểm định thật: dấu nhất quán, **không phân giải được**

Ô eval `bgem3-ctx-dense-rr-c50` đã chạy và so bằng `compare.py`
(`compare/bycategory-ctx-rr-c50-vs-ctx-dense-rr-c50.md`, Bonferroni 90 phép kiểm):

| nhóm | Δ nDCG@10 | CI95 **thô** | phán quyết ở α đã hiệu chỉnh |
|---|---:|---|---|
| `cross_lingual` | **+0,0628** | [+0,0073, +0,1363] | **KHÔNG KẾT LUẬN** |
| `factoid` | **−0,0333** | [−0,0739, −0,0019] | **KHÔNG KẾT LUẬN** |
| 4 nhóm còn lại | ≤ +0,0217 | — | trong ngưỡng nhiễu / trùng khớp |

Cả hai hàng đáng chú ý đều **loại 0 ở CI95 thô** và **mất khi hiệu chỉnh** — và
lý do sâu hơn ngưỡng: `cross_lingual` chỉ có **0↔4 câu khác nhau**, tức trần `p`
của phép kiểm dấu là **0,125**. Bốn câu thì không đạt ý nghĩa được dù kết quả
thế nào. Đây đúng là `TD-13` chứ không phải phát hiện mới; `W2-09` đã đo ngưỡng
phân giải ~±0,22 và tính ra cần **~440 câu**.

Bảng **tổng** thì 15/15 nằm trong nhiễu (nDCG@10 +0,0049, `hit_rate@1` hoà tuyệt
đối) — và nền dense **nhanh hơn 51,6 ms** ở p95 (757,7 vs 809,3).

⭐ Nên phát biểu đúng là: **ở điểm vận hành có reranker, tầng hybrid tốn 51,6 ms
và không mua được gì đo được ở bảng tổng** — lần thứ **tư** kết quả này lặp lại
sau `W2-05`, `W2-08` (0/24 ở ba độ sâu pool) và `W2-09`, lần này trên index có
ngữ cảnh. Nó **không** trung tính theo nhóm: nó đổi `factoid` lấy
`cross_lingual`, và cả hai chiều đều dưới ngưỡng phân giải của `golden_v1`.

⚠️ Vì thế **không** được rút bảng §5 ra làm căn cứ đổi điểm vận hành. Nó là một
giả thuyết có dấu nhất quán, và điều kiện để kiểm nó là `TD-13`.

## 6. Kết luận cho `TD-35`

1. **Truncation không phải cơ chế** — đoạn tương ứng ở `exp-002` §3 đã sửa.
2. **Reranker không phải thủ phạm** — cùng pool, văn bản đã dán không làm hại
   `cross_lingual` (−0,0061, `hit_rate` hoà tuyệt đối).
3. **Thủ phạm là tầng hợp nhất RRF**, nơi một nhánh sparse gần như mù với nhóm
   này (1/43) được trọng số ½ bởi `k=1`, và ngữ cảnh cho nó thêm chữ để sai tự
   tin hơn.
4. Hiện tượng **không mới**: đây là lần thứ tư nó được đo, và `W2-09` đã ghi
   đúng cơ chế ở một tầng khác.

**Còn để mở:** `cross_lingual` chỉ có `hit_rate@50` = **0,0233** ở nhánh sparse.
Một nhánh phục vụ 1/43 câu mà vẫn được RRF cho trọng số ngang nhánh kia là một
lỗi thiết kế của phép hợp nhất, không phải của nhánh. Cách chữa đúng có thể
không phải "định tuyến sang dense" mà là **RRF có trọng số theo độ tin của từng
nhánh** — chưa đo, ghi thành `TD-37`.
