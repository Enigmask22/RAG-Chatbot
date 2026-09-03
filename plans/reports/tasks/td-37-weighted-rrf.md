# `TD-37` — RRF có trọng số, và một câu hỏi thiết kế bị đóng bằng số

> **Ngày:** 2026-09-03 · **Mã:** `scripts/td37_probe.py`
> **Bằng chứng:** `reports/probes/td-37.json`, `runs/bgem3-ctx-rr-c50-w025-*`,
> `compare/cmp-ctx-rr-c50-vs-w025.md`, `compare/bycategory-ctx-rr-c50-vs-w025.md`
> **Kết quả:** bundle `v0.2.0` — điểm vận hành mới

## 1. Câu hỏi

`TD-35` truy ra rằng thủ phạm của hồi quy `cross_lingual` là **tầng hợp nhất**:
sparse tìm được 1/43 câu `cross_lingual` (`hit_rate@50` = 0,0233) nhưng RRF `k=1`
vẫn cho hạng 1 của nó trọng số ½. Nó đề xuất chữa bằng **định tuyến** (`W4-07`).

`TD-37` phản biện: nếu cách chữa đúng là cân trọng số theo độ tin thì `W4-07`
đang giải sai bài toán. Hạng mục này quyết định điều đó **trước** khi `W4-07`
được dựng — đúng như ghi chú nợ yêu cầu.

## 2. Đo được mà không tốn đồng nào

`TD-35` đã xác định thiệt hại nằm ở **trần pool**, và reranker không tạo ra được
ứng viên không có trong pool nó nhận. Nên: **một** lượt truy hồi hai nhánh cho
209 câu, cache lại thứ hạng, rồi hợp nhất lại **offline** với bao nhiêu luật tuỳ
ý. 16 luật trong vài giây, không GPU, không API.

## 3. Dự đoán vs kết quả — 2/4 sai, và cái sai quan trọng hơn cái đúng

| | dự đoán | thực tế |
|---|---|---|
| **P1** | quét trọng số tĩnh chỉ trượt trên đường đánh đổi | ❌ **SAI** — `w=0,2–0,25` thắng cả hai đầu |
| **P2** | phân bố điểm sparse tách được hai nhóm | ❌ **gần như SAI** — AUC 0,688, và median gần bằng nhau |
| **P3** | cân thích ứng thắng cả hai đầu của P1 | ❌ **SAI** — nó *là* cái trượt trên đường đánh đổi |
| **P4** | dense-only nằm trong nhiễu so với bản thích ứng tốt nhất | ✅ đúng, chính xác bằng nhau |

Trần pool `hit_rate@50`, 209 câu, so với điểm vận hành (`w=1`):

```
luật                     @50      @10  cross_ling  factoid   thắng↔thua  p
w=0   (dense-only)    0.8325   0.6842      0.6744   0.8971        6↔5   1.000
w=0.1                 0.8469   0.7033      0.6744   0.9265        6↔2   0.289
w=0.2                 0.8517   0.7225      0.6744   0.9265        6↔1   0.125
w=0.25                0.8517   0.7225      0.6744   0.9265        6↔1   0.125
w=0.5                 0.8325   0.7368      0.6279   0.9265        2↔1   1.000
w=1   (điểm vận hành) 0.8278   0.7177      0.5814   0.9412        0↔0      —
adaptive (tốt nhất)   0.8325   0.6842      0.6744   0.8971        6↔5   1.000
```

⚠️ **Không ô nào đạt ý nghĩa thống kê** — 6↔1 cho `p` = 0,125, đúng bằng **trần
`p`** của 7 cặp bất đồng. Đó là `TD-13` lần thứ n. Nhưng cái đáng tin ở đây không
phải một ô: nó là **đường liều–đáp ứng đơn điệu** chạy suốt 11 mức trọng số
(6↔5 → 6↔4 → 6↔2 → 6↔2 → 6↔1 → 6↔1 → 5↔1 → 4↔1 → 2↔1 → 1↔0 → 0↔0). Nhiễu không
xếp thành hình đó.

## 4. ⭐⭐ Vì sao đề xuất của chính `TD-37` **không** cài được

`TD-37` đề xuất "cân theo độ tin từng nhánh, ước lượng lúc chạy". Phép đo nói
rằng tín hiệu ấy **không tồn tại** ở chỗ tự nhiên nhất để tìm nó:

| nhóm | `sparse hit_rate@50` | `median peakedness` |
|---|---:|---:|
| factoid | **0,8676** | 1,2049 |
| cross_lingual | **0,0233** | 1,1539 |

Sparse thất bại **37 lần** thường xuyên hơn ở `cross_lingual`, mà phân bố điểm
của nó trông **gần như y hệt**. Nói cách khác: nhánh sparse **tự tin và sai** —
và đó chính xác là lý do lỗi này im lặng suốt bốn lần đo (`W2-04`, `W2-08-prep`,
`W2-09`, `W2-05`).

Hệ quả trực tiếp: mọi luật **bật/tắt** — dù là ngưỡng tự động ở đây hay định
tuyến của `W4-07` — đều chỉ trượt trên đường đánh đổi, vì chúng phải *quyết định*
đúng/sai cho mỗi câu bằng một tín hiệu không phân biệt được. Một **trọng số nhỏ**
không phải quyết định gì cả: nó giữ lại phần đóng góp của sparse ở chỗ nhánh ấy
đúng, mà không cho hạng 1 của nó áp đảo khi nó sai.

## 5. Xác nhận qua reranker — 15/15 metric dương

Trần pool là proxy; điểm vận hành có reranker. Chạy eval thật với
`--rrf-weights 1 0.25`:

| metric | `w=1` | `w=0,25` | Δ | kết luận |
|---|---:|---:|---:|---|
| `ndcg@10` | 0,6888 | **0,7079** | +0,0191 | khác biệt thật |
| `map@20` | 0,6449 | 0,6636 | +0,0186 | khác biệt thật |
| `recall@20` | 0,7967 | 0,8246 | +0,0279 | khác biệt thật |
| `recall@10` | 0,7759 | 0,8022 | +0,0263 | khác biệt thật |
| `hit_rate@1` | 0,6077 | 0,6220 | +0,0144 | không đủ lực |

**15/15 metric dương, 6 đạt "khác biệt thật", 0 âm.**

## 6. ⭐ So thẳng với định tuyến — cùng mức lợi, 1/3,6 cái giá

nDCG@10 theo nhóm, so với điểm vận hành:

| nhóm | `w=0,25` | định tuyến sang nền dense (`TD-35`) |
|---|---:|---:|
| **cross_lingual** | **+0,0622** | +0,0628 |
| adversarial | +0,0294 | — |
| aggregation | +0,0191 | — |
| multi_hop | +0,0130 | — |
| table_lookup | 0,0000 | — |
| **factoid** | **−0,0093** | **−0,0333** |

Cùng mức lợi trên nhóm cần cứu, **1/3,6** cái giá trên nhóm phải trả, và nó còn
kéo ba nhóm khác lên thay vì để nguyên. ⚠️ Mọi hàng theo nhóm đều **KHÔNG KẾT
LUẬN** ở α đã hiệu chỉnh Bonferroni cho 90 phép kiểm — `TD-13`, như mọi lần.

Và cái giá **kỹ thuật** thì không so được: một tham số cấu hình, so với một hệ
thống phân loại truy vấn phải huấn luyện, đo, và bảo trì.

## 7. Hệ quả cho `W4-07`

`TD-35` viết *"routing đang giải sai bài toán"* như một nghi ngờ. Nó đúng, nhưng
không đúng vì lý do đã nêu: không phải vì cân theo độ tin tốt hơn (nó **không cài
được**), mà vì **bản thân việc quyết định nhị phân** là hình dạng sai cho một
nhánh chỉ *đôi khi* hữu ích.

`W4-07` **không mất lý do tồn tại** — rewrite đa lượt và `NO_RETRIEVAL` cho
"hello" là hai việc khác, và cả hai vẫn cần. Thứ bị gỡ khỏi phạm vi của nó là
**định tuyến dense-vs-hybrid theo loại truy vấn**: mục tiêu ấy đã đạt bằng một
dòng cấu hình, và làm lại nó bằng routing sẽ tốn hơn để được ít hơn.

## 8. Điểm vận hành mới — bundle `v0.2.0`

Đây là bundle **thứ hai** được promote, tức đúng cái mà `TD-36` nói phải giải
quyết trước — và nó đã được giải cùng ngày. Chạy thật với cả hai bundle trên đĩa:

```
khởi động → 0.2.0 (semver cao nhất)
  reranked[qdrant-hybrid:rag_bgem3_ctx:rrf1-c20-w1:0.25]:…@cuda:L512:float16:n50
reload 0.1.0 → 451 ms          ← không nạp lại trọng số, `lru_cache` của W4-02
rollback     → 0.2.0
/ready       → true, rollback_to = 0.1.0
```

⭐ **451 ms** so với ~20 giây khởi động lạnh: đó là món nợ "cache model theo danh
tính" của `W4-02` — hoá ra đã có sẵn — được **đo** lần đầu. Hai bundle này dùng
chung đúng một bản trọng số vì chúng chỉ khác `weights`.

Phép kiểm danh tính của `TD-38` cũng đi qua đường thật lần đầu với **hai** tên
khác nhau, gồm cả phép chuyển `weights` từ list (JSON) về tuple.

## 9. Còn lại

* **Không chạy lại toàn bộ `W2-08`** với trọng số mới. Bảng ablation cũ vẫn đúng
  cho câu hỏi nó hỏi (`k` nào thắng ở trọng số đều); nó chỉ không còn mô tả điểm
  vận hành. Chạy lại 12 ô để cập nhật một bảng không ai gác là chi phí không có
  người mua.
* **`w = 0,2` và `w = 0,25` bằng nhau** trên mọi metric đo được. Chọn 0,25 vì nó
  là ¼ tròn; không có bằng chứng nào tách hai giá trị này, và nói ra điều đó tốt
  hơn là giả vờ đã tối ưu.
* **Trọng số này gắn với corpus này.** Tỉ lệ 59,4% vi / 40,6% en và 43/209 câu
  cross-lingual là thứ quyết định điểm tối ưu; một corpus một ngôn ngữ gần như
  chắc chắn muốn `w` cao hơn. Nó nằm trong bundle chứ không nằm trong mã, và đó
  đúng là chỗ của nó.
