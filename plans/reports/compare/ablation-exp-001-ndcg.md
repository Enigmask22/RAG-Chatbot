# Ablation: 14 cấu hình, mốc `e1-baseline-dense`

- Xếp theo `ndcg@10`. Metric chính: `ndcg@10`, `hit_rate@1`, `mrr`.
- Bảng dưới **không** hiệu chỉnh đa so sánh: nó là một panel giả thuyết nêu trước (ma trận đã khai trong config trước khi chạy ô nào), và hiệu ứng ở đây cỡ `p ~ 1e-20` nên hiệu chỉnh không đổi hàng nào. Chỗ hiệu chỉnh là **tập tương đương** bên dưới, nơi mức chênh là 1–5 câu.

## Bảng ablation

| # | run | cấu hình | nhãn/câu | `ndcg@10` | `hit_rate@1` | `mrr` | p95 ms | vs baseline |
|---|---|---|---|---:|---:|---:|---:|---|
| 1 | `e1-rr-bgem3-reranked-onhybrid-rc100` | bge-m3 · reranked · base=hybrid · rerank_candidates=100 · chunk=1000 | 1.3828 | 0.6736 | 0.5789 | 0.6694 | 1163.9 | `ndcg`:khác biệt thật · `hit_rate`:khác biệt thật · `mrr`:khác biệt thật |
| 2 | `e1-rr-bgem3-reranked-ondense-rc100` | bge-m3 · reranked · base=dense · rerank_candidates=100 · chunk=1000 | 1.3828 | 0.6624 | 0.5742 | 0.6595 | 1182.3 | `ndcg`:khác biệt thật · `hit_rate`:khác biệt thật · `mrr`:khác biệt thật |
| 3 | `e1-rr-bgem3-reranked-onhybrid-rc50` | bge-m3 · reranked · base=hybrid · rerank_candidates=50 · chunk=1000 | 1.3828 | 0.6481 | 0.5598 | 0.6440 | 608.9 | `ndcg`:khác biệt thật · `hit_rate`:khác biệt thật · `mrr`:khác biệt thật |
| 4 | `e1-rr-bgem3-reranked-ondense-rc50` | bge-m3 · reranked · base=dense · rerank_candidates=50 · chunk=1000 | 1.3828 | 0.6268 | 0.5455 | 0.6265 | 618.5 | `ndcg`:khác biệt thật · `hit_rate`:khác biệt thật · `mrr`:khác biệt thật |
| 5 | `e1-rr-bgem3-reranked-onhybrid-rc20` | bge-m3 · reranked · base=hybrid · rerank_candidates=20 · chunk=1000 | 1.3828 | 0.5823 | 0.5263 | 0.5902 | 276.5 | `ndcg`:khác biệt thật · `hit_rate`:khác biệt thật · `mrr`:khác biệt thật |
| 6 | `e1-rr-bgem3-reranked-ondense-rc20` | bge-m3 · reranked · base=dense · rerank_candidates=20 · chunk=1000 | 1.3828 | 0.5676 | 0.5072 | 0.5756 | 267.6 | `ndcg`:khác biệt thật · `hit_rate`:khác biệt thật · `mrr`:khác biệt thật |
| 7 | `e1-rrf-bgem3-hybrid-k0` | bge-m3 · hybrid · k=0 · chunk=1000 | 1.3828 | 0.4582 | 0.3493 | 0.4481 | 47.1 | `ndcg`:khác biệt thật · `hit_rate`:khác biệt thật · `mrr`:khác biệt thật |
| 8 | `e1-rrf-bgem3-hybrid-k1` | bge-m3 · hybrid · k=1 · chunk=1000 | 1.3828 | 0.4563 | 0.3397 | 0.4436 | 49.1 | `ndcg`:khác biệt thật · `hit_rate`:khác biệt thật · `mrr`:khác biệt thật |
| 9 | `e1-rrf-bgem3-hybrid-k2` | bge-m3 · hybrid · k=2 · chunk=1000 | 1.3828 | 0.4521 | 0.3301 | 0.4362 | 46.7 | `ndcg`:khác biệt thật · `hit_rate`:khác biệt thật · `mrr`:khác biệt thật |
| 10 | `e1-bgem3-dense` | bge-m3 · dense · chunk=1000 | 1.3828 | 0.4442 | 0.3397 | 0.4394 | 46.3 | `ndcg`:khác biệt thật · `hit_rate`:khác biệt thật · `mrr`:khác biệt thật |
| 11 | `e1-rrf-bgem3-hybrid-k60` | bge-m3 · hybrid · k=60 · chunk=1000 | 1.3828 | 0.4313 | 0.3014 | 0.4080 | 48.0 | `ndcg`:khác biệt thật · `hit_rate`:khác biệt thật · `mrr`:khác biệt thật |
| 12 | `e1-bgem3-sparse` | bge-m3 · sparse · chunk=1000 | 1.3828 | 0.3733 | 0.2919 | 0.3623 | 44.2 | `ndcg`:khác biệt thật · `hit_rate`:khác biệt thật · `mrr`:khác biệt thật |
| 13 | `e1-baseline-dense` | vietnamese-bi-encoder · dense · chunk=1000 | 1.3828 | 0.1621 | 0.1196 | 0.1660 | 42.8 | *(baseline)* |
| 14 | `e1-chunk550-dense` | vietnamese-bi-encoder · dense · chunk=550 | 1.9617 | 0.1215 | 0.0861 | 0.1414 | 46.1 | `ndcg`:KHÔNG SO ĐƯỢC · `hit_rate`:KHÔNG SO ĐƯỢC · `mrr`:KHÔNG SO ĐƯỢC |

## Thứ hạng có đổi theo metric không

| hạng | `ndcg@10` | `hit_rate@1` | `mrr` |
|---:|---|---|---|
| 1 | `e1-rr-bgem3-reranked-onhybrid-rc100` | `e1-rr-bgem3-reranked-onhybrid-rc100` | `e1-rr-bgem3-reranked-onhybrid-rc100` |
| 2 | `e1-rr-bgem3-reranked-ondense-rc100` | `e1-rr-bgem3-reranked-ondense-rc100` | `e1-rr-bgem3-reranked-ondense-rc100` |
| 3 | `e1-rr-bgem3-reranked-onhybrid-rc50` | `e1-rr-bgem3-reranked-onhybrid-rc50` | `e1-rr-bgem3-reranked-onhybrid-rc50` |
| 4 | `e1-rr-bgem3-reranked-ondense-rc50` | `e1-rr-bgem3-reranked-ondense-rc50` | `e1-rr-bgem3-reranked-ondense-rc50` |
| 5 | `e1-rr-bgem3-reranked-onhybrid-rc20` | `e1-rr-bgem3-reranked-onhybrid-rc20` | `e1-rr-bgem3-reranked-onhybrid-rc20` |
| 6 | `e1-rr-bgem3-reranked-ondense-rc20` | `e1-rr-bgem3-reranked-ondense-rc20` | `e1-rr-bgem3-reranked-ondense-rc20` |
| 7 | `e1-rrf-bgem3-hybrid-k0` | `e1-rrf-bgem3-hybrid-k0` | `e1-rrf-bgem3-hybrid-k0` |
| 8 | `e1-rrf-bgem3-hybrid-k1` | `e1-bgem3-dense` | `e1-rrf-bgem3-hybrid-k1` |
| 9 | `e1-rrf-bgem3-hybrid-k2` | `e1-rrf-bgem3-hybrid-k1` | `e1-bgem3-dense` |
| 10 | `e1-bgem3-dense` | `e1-rrf-bgem3-hybrid-k2` | `e1-rrf-bgem3-hybrid-k2` |
| 11 | `e1-rrf-bgem3-hybrid-k60` | `e1-rrf-bgem3-hybrid-k60` | `e1-rrf-bgem3-hybrid-k60` |
| 12 | `e1-bgem3-sparse` | `e1-bgem3-sparse` | `e1-bgem3-sparse` |
| 13 | `e1-baseline-dense` | `e1-baseline-dense` | `e1-baseline-dense` |
| 14 | `e1-chunk550-dense` | `e1-chunk550-dense` | `e1-chunk550-dense` |

## Tập tương đương của `e1-rr-bgem3-reranked-onhybrid-rc100` (xếp theo `ndcg@10`)

- Họ **39 phép kiểm**, α đã hiệu chỉnh Bonferroni = **0.0012821**. Đây là một cuộc **tìm kiếm** (chọn một trong 14 ô), nên nó hiệu chỉnh — khác bảng so-với-baseline ở trên, là một panel giả thuyết nêu trước.
- **Không phân biệt được với đỉnh bảng: 2 ô.** Các metric chính **không đồng ý**: 3 ô. Kém đỉnh bảng ở mọi metric: 8 ô. **Không so được** (nhãn khác): 1 ô.
- ⚠️ `KHÔNG SO ĐƯỢC` là rổ riêng, **không** phải "không phân biệt được". Gộp hai cái đó đưa ô tệ nhất bảng lên làm thành viên rẻ nhất của tập thắng — đã xảy ra ở bản đầu, xem docstring module.

### Tập tương đương, xếp theo giá

| run | cấu hình | `ndcg@10` | `hit_rate@1` | `mrr` | p95 ms |
|---|---|---:|---:|---:|---:|
| `e1-rr-bgem3-reranked-onhybrid-rc100` ⬅ đỉnh bảng | bge-m3 · reranked · base=hybrid · rerank_candidates=100 · chunk=1000 | 0.6736 | 0.5789 | 0.6694 | 1163.9 |
| `e1-rr-bgem3-reranked-ondense-rc100` | bge-m3 · reranked · base=dense · rerank_candidates=100 · chunk=1000 | 0.6624 | 0.5742 | 0.6595 | 1182.3 |

➡️ Đỉnh bảng **cũng là** ô rẻ nhất trong tập tương đương — không có gì phải đánh đổi.

### ⚠️ Tranh chấp: các metric chính không đồng ý

Những ô sau bị **một phần** metric chính nói là kém đỉnh bảng, phần còn lại thì không. Đây là chỗ phán quyết kỹ thuật thật sự nằm, và công cụ **cố ý không** quyết hộ: cột giá cho biết mua sự chắc chắn ấy tốn bao nhiêu.

| run | cấu hình | metric nói kém | metric không kết luận | p95 ms | so đỉnh bảng |
|---|---|---|---|---:|---:|
| `e1-rr-bgem3-reranked-onhybrid-rc20` | bge-m3 · reranked · base=hybrid · rerank_candidates=20 · chunk=1000 | `ndcg@10`, `mrr` | `hit_rate@1 (trong ngưỡng nhiễu)` | 276.5 | **4.21× rẻ hơn** |
| `e1-rr-bgem3-reranked-onhybrid-rc50` | bge-m3 · reranked · base=hybrid · rerank_candidates=50 · chunk=1000 | `mrr` | `ndcg@10 (TRÁI CHIỀU)`, `hit_rate@1 (KHÔNG ĐỦ LỰC)` | 608.9 | **1.91× rẻ hơn** |
| `e1-rr-bgem3-reranked-ondense-rc50` | bge-m3 · reranked · base=dense · rerank_candidates=50 · chunk=1000 | `ndcg@10`, `mrr` | `hit_rate@1 (KHÔNG ĐỦ LỰC)` | 618.5 | **1.88× rẻ hơn** |

### Không so được (nhãn khác nhau)

- `e1-chunk550-dense` — 1.9617 nhãn/câu vs 1.3828 của đỉnh bảng

### Từng ô, ba metric chính

#### `e1-rr-bgem3-reranked-onhybrid-rc100` → `e1-rr-bgem3-reranked-ondense-rc100`

| metric | e1-rr-bgem3-reranked-onhybrid-rc100 | e1-rr-bgem3-reranked-ondense-rc100 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `ndcg@10` | 0.6736 | 0.6624 | -0.0113 | 209 | CI99.87% [-0.0472, +0.0153] · 9↔11 câu khác nhau (p dấu=0.8238) · CI95 thô [-0.0313, +0.0063] · **đếm câu đi ngược Δ** (11 câu tốt hơn vs 9 câu xấu đi, mà Δ = -0.0113) | trong ngưỡng nhiễu |
| `hit_rate@1` | 0.5789 | 0.5742 | -0.0048 | 209 | p=1 · 3↔2 câu đổi chiều · **trần `p` = 0.0625** ở α=0.001282 | KHÔNG ĐỦ LỰC |
| `mrr` | 0.6694 | 0.6595 | -0.0099 | 209 | CI99.87% [-0.0497, +0.0223] · 8↔7 câu khác nhau (p dấu=1) · CI95 thô [-0.0310, +0.0096] | trong ngưỡng nhiễu |

#### `e1-rr-bgem3-reranked-onhybrid-rc100` → `e1-rr-bgem3-reranked-onhybrid-rc50`

| metric | e1-rr-bgem3-reranked-onhybrid-rc100 | e1-rr-bgem3-reranked-onhybrid-rc50 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `ndcg@10` | 0.6736 | 0.6481 | -0.0255 | 209 | CI99.87% [-0.0638, -0.0003] · 10↔11 câu khác nhau (p dấu=1) · CI95 thô [-0.0478, -0.0075] · **đếm câu đi ngược Δ** (11 câu tốt hơn vs 10 câu xấu đi, mà Δ = -0.0255) | TRÁI CHIỀU |
| `hit_rate@1` | 0.5789 | 0.5598 | -0.0191 | 209 | p=0.125 · 4↔0 câu đổi chiều · **trần `p` = 0.125** ở α=0.001282 | KHÔNG ĐỦ LỰC |
| `mrr` | 0.6694 | 0.6440 | -0.0254 | 209 | CI99.87% [-0.0625, -0.0007] · 10↔6 câu khác nhau (p dấu=0.4545) · CI95 thô [-0.0471, -0.0079] | khác biệt thật |

#### `e1-rr-bgem3-reranked-onhybrid-rc100` → `e1-rr-bgem3-reranked-ondense-rc50`

| metric | e1-rr-bgem3-reranked-onhybrid-rc100 | e1-rr-bgem3-reranked-ondense-rc50 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `ndcg@10` | 0.6736 | 0.6268 | -0.0469 | 209 | CI99.87% [-0.0981, -0.0098] · 17↔11 câu khác nhau (p dấu=0.3449) · CI95 thô [-0.0750, -0.0219] | khác biệt thật |
| `hit_rate@1` | 0.5789 | 0.5455 | -0.0335 | 209 | p=0.03906 · 8↔1 câu đổi chiều · **trần `p` = 0.003906** ở α=0.001282 | KHÔNG ĐỦ LỰC |
| `mrr` | 0.6694 | 0.6265 | -0.0429 | 209 | CI99.87% [-0.0937, -0.0060] · 15↔7 câu khác nhau (p dấu=0.1338) · CI95 thô [-0.0715, -0.0178] | khác biệt thật |

#### `e1-rr-bgem3-reranked-onhybrid-rc100` → `e1-rr-bgem3-reranked-onhybrid-rc20`

| metric | e1-rr-bgem3-reranked-onhybrid-rc100 | e1-rr-bgem3-reranked-onhybrid-rc20 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `ndcg@10` | 0.6736 | 0.5823 | -0.0913 | 209 | CI99.87% [-0.1522, -0.0390] · 35↔13 câu khác nhau (p dấu=0.002088) · CI95 thô [-0.1281, -0.0576] | khác biệt thật |
| `hit_rate@1` | 0.5789 | 0.5263 | -0.0526 | 209 | p=0.007385 · 13↔2 câu đổi chiều | trong ngưỡng nhiễu |
| `mrr` | 0.6694 | 0.5902 | -0.0792 | 209 | CI99.87% [-0.1395, -0.0273] · 32↔7 câu khác nhau (p dấu=7.025e-05) · CI95 thô [-0.1155, -0.0463] | khác biệt thật |

#### `e1-rr-bgem3-reranked-onhybrid-rc100` → `e1-rr-bgem3-reranked-ondense-rc20`

| metric | e1-rr-bgem3-reranked-onhybrid-rc100 | e1-rr-bgem3-reranked-ondense-rc20 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `ndcg@10` | 0.6736 | 0.5676 | -0.1060 | 209 | CI99.87% [-0.1761, -0.0468] · 38↔16 câu khác nhau (p dấu=0.003838) · CI95 thô [-0.1454, -0.0694] | khác biệt thật |
| `hit_rate@1` | 0.5789 | 0.5072 | -0.0718 | 209 | p=0.0002747 · 16↔1 câu đổi chiều | khác biệt thật |
| `mrr` | 0.6694 | 0.5756 | -0.0938 | 209 | CI99.87% [-0.1621, -0.0392] · 32↔6 câu khác nhau (p dấu=2.434e-05) · CI95 thô [-0.1324, -0.0580] | khác biệt thật |

#### `e1-rr-bgem3-reranked-onhybrid-rc100` → `e1-rrf-bgem3-hybrid-k0`

| metric | e1-rr-bgem3-reranked-onhybrid-rc100 | e1-rrf-bgem3-hybrid-k0 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `ndcg@10` | 0.6736 | 0.4582 | -0.2155 | 209 | CI99.87% [-0.2909, -0.1438] · 100↔14 câu khác nhau (p dấu=3.488e-17) · CI95 thô [-0.2619, -0.1703] | khác biệt thật |
| `hit_rate@1` | 0.5789 | 0.3493 | -0.2297 | 209 | p=2.43e-10 · 55↔7 câu đổi chiều | khác biệt thật |
| `mrr` | 0.6694 | 0.4481 | -0.2213 | 209 | CI99.87% [-0.3074, -0.1399] · 86↔11 câu khác nhau (p dấu=1.434e-15) · CI95 thô [-0.2733, -0.1701] | khác biệt thật |

#### `e1-rr-bgem3-reranked-onhybrid-rc100` → `e1-rrf-bgem3-hybrid-k1`

| metric | e1-rr-bgem3-reranked-onhybrid-rc100 | e1-rrf-bgem3-hybrid-k1 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `ndcg@10` | 0.6736 | 0.4563 | -0.2173 | 209 | CI99.87% [-0.2927, -0.1432] · 100↔14 câu khác nhau (p dấu=3.488e-17) · CI95 thô [-0.2639, -0.1722] | khác biệt thật |
| `hit_rate@1` | 0.5789 | 0.3397 | -0.2392 | 209 | p=7.638e-11 · 57↔7 câu đổi chiều | khác biệt thật |
| `mrr` | 0.6694 | 0.4436 | -0.2257 | 209 | CI99.87% [-0.3121, -0.1412] · 88↔11 câu khác nhau (p dấu=4.53e-16) · CI95 thô [-0.2780, -0.1748] | khác biệt thật |

#### `e1-rr-bgem3-reranked-onhybrid-rc100` → `e1-rrf-bgem3-hybrid-k2`

| metric | e1-rr-bgem3-reranked-onhybrid-rc100 | e1-rrf-bgem3-hybrid-k2 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `ndcg@10` | 0.6736 | 0.4521 | -0.2216 | 209 | CI99.87% [-0.2977, -0.1474] · 101↔13 câu khác nhau (p dấu=4.774e-18) · CI95 thô [-0.2680, -0.1761] | khác biệt thật |
| `hit_rate@1` | 0.5789 | 0.3301 | -0.2488 | 209 | p=2.383e-11 · 59↔7 câu đổi chiều | khác biệt thật |
| `mrr` | 0.6694 | 0.4362 | -0.2332 | 209 | CI99.87% [-0.3188, -0.1516] · 91↔11 câu khác nhau (p dấu=7.967e-17) · CI95 thô [-0.2856, -0.1822] | khác biệt thật |

#### `e1-rr-bgem3-reranked-onhybrid-rc100` → `e1-bgem3-dense`

| metric | e1-rr-bgem3-reranked-onhybrid-rc100 | e1-bgem3-dense | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `ndcg@10` | 0.6736 | 0.4442 | -0.2295 | 209 | CI99.87% [-0.3066, -0.1549] · 110↔18 câu khác nhau (p dấu=2.654e-17) · CI95 thô [-0.2766, -0.1829] | khác biệt thật |
| `hit_rate@1` | 0.5789 | 0.3397 | -0.2392 | 209 | p=3.914e-10 · 59↔9 câu đổi chiều | khác biệt thật |
| `mrr` | 0.6694 | 0.4394 | -0.2300 | 209 | CI99.87% [-0.3166, -0.1421] · 90↔15 câu khác nhau (p dấu=3.276e-14) · CI95 thô [-0.2831, -0.1778] | khác biệt thật |

#### `e1-rr-bgem3-reranked-onhybrid-rc100` → `e1-rrf-bgem3-hybrid-k60`

| metric | e1-rr-bgem3-reranked-onhybrid-rc100 | e1-rrf-bgem3-hybrid-k60 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `ndcg@10` | 0.6736 | 0.4313 | -0.2423 | 209 | CI99.87% [-0.3176, -0.1662] · 106↔12 câu khác nhau (p dấu=5.775e-20) · CI95 thô [-0.2901, -0.1956] | khác biệt thật |
| `hit_rate@1` | 0.5789 | 0.3014 | -0.2775 | 209 | p=2.443e-13 · 64↔6 câu đổi chiều | khác biệt thật |
| `mrr` | 0.6694 | 0.4080 | -0.2614 | 209 | CI99.87% [-0.3460, -0.1772] · 97↔11 câu khác nhau (p dấu=2.391e-18) · CI95 thô [-0.3145, -0.2094] | khác biệt thật |

#### `e1-rr-bgem3-reranked-onhybrid-rc100` → `e1-bgem3-sparse`

| metric | e1-rr-bgem3-reranked-onhybrid-rc100 | e1-bgem3-sparse | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `ndcg@10` | 0.6736 | 0.3733 | -0.3003 | 209 | CI99.87% [-0.3891, -0.2175] · 108↔11 câu khác nhau (p dấu=3.524e-21) · CI95 thô [-0.3549, -0.2449] | khác biệt thật |
| `hit_rate@1` | 0.5789 | 0.2919 | -0.2871 | 209 | p=7.256e-14 · 66↔6 câu đổi chiều | khác biệt thật |
| `mrr` | 0.6694 | 0.3623 | -0.3071 | 209 | CI99.87% [-0.4014, -0.2153] · 100↔10 câu khác nhau (p dấu=8.01e-20) · CI95 thô [-0.3660, -0.2464] | khác biệt thật |

#### `e1-rr-bgem3-reranked-onhybrid-rc100` → `e1-baseline-dense`

| metric | e1-rr-bgem3-reranked-onhybrid-rc100 | e1-baseline-dense | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `ndcg@10` | 0.6736 | 0.1621 | -0.5115 | 209 | CI99.87% [-0.6016, -0.4154] · 153↔8 câu khác nhau (p dấu=6.77e-36) · CI95 thô [-0.5684, -0.4538] | khác biệt thật |
| `hit_rate@1` | 0.5789 | 0.1196 | -0.4593 | 209 | p=1.981e-22 · 104↔8 câu đổi chiều | khác biệt thật |
| `mrr` | 0.6694 | 0.1660 | -0.5034 | 209 | CI99.87% [-0.6003, -0.3993] · 143↔9 câu khác nhau (p dấu=3.502e-32) · CI95 thô [-0.5652, -0.4386] | khác biệt thật |

#### `e1-rr-bgem3-reranked-onhybrid-rc100` → `e1-chunk550-dense`

| metric | e1-rr-bgem3-reranked-onhybrid-rc100 | e1-chunk550-dense | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `ndcg@10` | 0.6736 | 0.1215 | -0.5522 | 209 | ⚠️ 209/209 câu có tập nhãn khác nhau (băm `relevant_digest` lệch) — hai lần chạy không cùng bài toán | KHÔNG SO ĐƯỢC |
| `hit_rate@1` | 0.5789 | 0.0861 | -0.4928 | 209 | ⚠️ 209/209 câu có tập nhãn khác nhau (băm `relevant_digest` lệch) — hai lần chạy không cùng bài toán | KHÔNG SO ĐƯỢC |
| `mrr` | 0.6694 | 0.1414 | -0.5280 | 209 | ⚠️ 209/209 câu có tập nhãn khác nhau (băm `relevant_digest` lệch) — hai lần chạy không cùng bài toán | KHÔNG SO ĐƯỢC |

