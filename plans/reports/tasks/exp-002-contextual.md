# `exp-002` — Contextual Retrieval: đo thật trên 209 câu

> **Ngày:** 2026-09-03 · **Index:** `rag_bgem3_ctx` (15.814 chunk, 100% có ngữ cảnh)
> **So với:** `rag_bgem3` — **cùng chunking đúng từng tham số**, chỉ khác một thứ
> **Run:** `bgem3-ctx`, `bgem3-ctx-rr-c50` · **Evidence:** `compare/cmp-bgem3-vs-bgem3-ctx.md`,
> `compare/cmp-rr-c50-vs-ctx-rr-c50.md`, `compare/bycategory-rr-c50-vs-ctx-rr-c50.md`

## 1. Câu trả lời

**`G3`, tiêu chí 1 — Contextual Retrieval thắng, và thắng ở cả hai tầng.**

| | nDCG@10 | Δ | metric có ý nghĩa |
|---|---:|---:|---:|
| dense: `bgem3` → `bgem3-ctx` | 0,4442 → **0,5019** | +0,0577 | **13/15** |
| reranked: `rr-c50` → `ctx-rr-c50` | 0,6481 → **0,6888** | +0,0407 | **9/15** |

**`W3-09` DoD — giảm retrieval failure bao nhiêu:**

| | failure@5 | giảm tương đối |
|---|---|---:|
| dense | 45,45% → 38,28% | **15,8%** |
| reranked *(điểm vận hành)* | 24,88% → 19,14% | **23,1%** |

Không metric nào ở bảng tổng đi ngược chiều, ở cả hai tầng.

## 2. ⭐⭐ Reranker KHÔNG hấp thụ lợi ích này

`W2-05` đã đo một kết quả kiến trúc: sau khi có reranker, **tầng hybrid không còn
đo được** (13/15 metric rơi vào nhiễu, và 0/24 khi quét ba độ sâu pool ở
`W2-08`). Lời giải thích khi ấy — reranker đọc lại toàn văn nên nó tự sửa được
những gì tầng dưới xếp sai — dự đoán rằng Contextual Retrieval **cũng** sẽ bị hấp
thụ: ngữ cảnh chỉ là thêm chữ vào chunk, mà reranker vốn đọc cả chunk.

Dự đoán ấy **sai**. Ở điểm vận hành có reranker, `hit_rate@5` vẫn tăng
**0,7512 → 0,8086**, và phân rã theo câu cho **3↔15** — mười lăm câu được sửa,
ba câu bị làm hỏng.

Vì sao khác nhau: tầng hybrid thêm **một cách xếp hạng khác** trên cùng một tập
văn bản, còn Contextual Retrieval thêm **văn bản chưa từng có ở đó**. Reranker
đọc lại được thứ nhất; nó không đọc lại được thứ nó chưa bao giờ nhìn thấy.

⚠️ Nói cho đủ: đây là suy diễn hợp lý sau khi thấy số, không phải giả thuyết ghi
trước. Dự đoán ghi trước của tôi là "sẽ bị hấp thụ" và nó sai.

## 3. ⚠️ `cross_lingual` là nhóm duy nhất đi ngược, và chỉ sau reranker

Quét theo nhóm, hiệu chỉnh Bonferroni cho 90 phép kiểm:

| nhóm | n | nDCG@10 dense | nDCG@10 reranked |
|---|---:|---|---|
| `factoid` | 68 | 0,5241 → 0,5818 | 0,7596 → 0,8150 |
| **`cross_lingual`** | 43 | 0,2538 → **0,3172** | 0,5191 → **0,5041** |
| `adversarial` | 34 | 0,3974 → 0,4486 | 0,5732 → 0,6235 |
| `multi_hop` | 34 | 0,5522 → 0,5883 | 0,6850 → 0,7647 |
| `aggregation` | 26 | 0,5235 → 0,5816 | 0,6423 → 0,6797 |
| `table_lookup` | 4 | 0,0967 → 0,3289 | 0,5000 → 0,5000 |

Ở tầng reranked, **cả 11 hàng có `Δ` âm trong toàn bảng 90 hàng đều thuộc
`cross_lingual`**, và 5 nhóm còn lại dương ở mọi metric. Không hàng nào đạt ý
nghĩa ở α đã hiệu chỉnh — nhưng dấu nhất quán tuyệt đối thì không phải chuyện
ngẫu nhiên dễ bỏ qua.

⭐ **Và ở tầng dense thì `cross_lingual` là 15/15 metric DƯƠNG** (0,2538 →
0,3172, mức tăng lớn thứ hai bảng). Nên nguyên nhân **không** nằm ở việc dán ngữ
cảnh, mà nằm ở **quãng giữa dense và reranked**.

### Đo được cái gì ở quãng đó

Reranker có cửa sổ cứng **512 token**. Dán ngữ cảnh làm chunk dài thêm 32,2%
(`inflation` = 1,3217), và đo lại bằng `rerank_probe`:

| | không ngữ cảnh | có ngữ cảnh |
|---|---:|---:|
| token p50 | 287 | **365** |
| token p95 | 352 | **453** |
| token max | 542 | **895** |
| tỉ lệ cặp bị cắt | 0,008% | **0,72%** |

Tỉ lệ cắt tăng **87×**, và p95 giờ ở 453/512 — phân bố đã áp sát trần.

⚠️⚠️ **Nhưng 0,72% quá nhỏ để giải thích một nhóm 43 câu đổi dấu ở 11/15
metric.** Hai sự thật này nhất quán với nhau và **chưa** chứng minh nhân quả.
Ghi lại ở `TD-35` kèm phép đo cần làm: tỉ lệ cắt **theo nhóm** và **theo ngôn
ngữ tài liệu** (tiếng Việt tokenise tệ hơn nên chạm trần trước). Nếu tỉ lệ cắt
của `cross_lingual` không cao hơn hẳn thì giả thuyết này sai và phải tìm chỗ khác.

⚠️ Cũng phải nói: `W2-09` đã đo ngưỡng phân giải giữa hai nhóm là **~±0,22**, cần
golden set **~440 câu**. Với `n = 43` thì `cross_lingual` **không thể** kết luận
được ở bảng này dù kết quả thế nào — đó là `TD-13`, không phải phát hiện mới.

## 4. Chi phí của phần thắng này

| | giá trị |
|---|---:|
| sinh ngữ cảnh (một lần) | **~$5,90** cho 15.814 chunk |
| chunk dài thêm | +32,2% ký tự |
| truncation ở tầng embed | **0/15.814** — BGE-M3 cửa sổ 8.192, token p95 chỉ 398 |
| truncation ở tầng rerank | 0,008% → **0,72%** |
| thời gian build lại index | 585s *(embed lại toàn bộ, `W3-07` không mượn được gì ở lượt đầu)* |
| độ trễ truy vấn | **không đổi** — ngữ cảnh nằm trong vector, không thêm bước nào |

⭐ Điểm đáng chú ý về kiến trúc: đây là mức cải thiện **mua bằng chi phí index một
lần**, không mua bằng độ trễ. Khác hẳn reranker (`W2-05`: +524 ms mỗi truy vấn) —
và hai thứ cộng dồn được với nhau.

## 5. Còn thiếu để đóng `W3-09`

Hạng mục gốc yêu cầu **5 chiến lược chunking**; ở đây mới có 2 ô (có/không ngữ
cảnh) trên cùng một chiến lược `hybrid`. Ba ô còn lại (`fixed`, `recursive`,
`semantic` × contextual) chưa chạy. Chúng không cần thêm tiền API — ngữ cảnh đã
sinh cho `chunk_size=1000`, nhưng đổi chiến lược là **đổi bộ chunk**, tức
`chunking_fingerprint` lệch và phải sinh lại ngữ cảnh cho từng ô. Đó là ~$5,90
mỗi ô, nên phải cân nhắc chứ không chạy mặc định.

`G3` tiêu chí 1 thì **đã trả lời được** bằng bảng ở §1.
