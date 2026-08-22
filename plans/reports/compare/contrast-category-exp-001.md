# `e1-baseline-dense` → `e1-rr-bgem3-reranked-onhybrid-rc100`: nhóm nào cải thiện nhiều nhất

- Chia theo `category`. Xếp nhóm theo `Δ`, rồi kiểm **từng** nhóm còn lại có phân biệt được với đỉnh bảng hay không — câu trả lời là một **tập**.
- Bootstrap **không cặp**: hai nhóm là hai tập câu rời nhau, không có cặp nào để ghép.
- ⚠️ `Δ` của `recall@k`/`ndcg@k`/`map@k` **không** xếp hạng được giữa các nhóm — mẫu số là số nhãn, mà số nhãn/câu đổi theo nhóm.

### `hit_rate@5` theo `category`

| hạng | nhóm | n | nhãn/câu | mốc | sau | Δ | tốt↔xấu |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `cross_lingual` ⬅ đỉnh bảng | 43 | 1.0930 | 0.0000 | 0.6512 | +0.6512 | 28↔0 |
| 2 | `aggregation` | 26 | 2.4231 | 0.2308 | 0.8462 | +0.6154 | 16↔0 |
| 3 | `adversarial` | 34 | 1.0882 | 0.1765 | 0.7647 | +0.5882 | 20↔0 |
| 4 | `factoid` | 68 | 1.0147 | 0.3088 | 0.8529 | +0.5441 | 37↔0 |
| 5 | `table_lookup` | 4 | 1.0000 | 0.0000 | 0.5000 | +0.5000 | 2↔0 |
| 6 | `multi_hop` | 34 | 2.0294 | 0.3529 | 0.8529 | +0.5000 | 17↔0 |

Đỉnh bảng **chọn bằng dữ liệu**, nên 5 phép so dưới đây hiệu chỉnh Bonferroni: α = 0.01.

| `cross_lingual` hơn | khoảng cách | n | kiểm định | kết luận |
|---|---:|---:|---|---|
| `aggregation` | +0.0358 | 43/26 | CI99% [-0.2728, +0.3444] | không phân biệt được |
| `adversarial` | +0.0629 | 43/34 | CI99% [-0.2237, +0.3495] | không phân biệt được |
| `factoid` | +0.1070 | 43/68 | CI99% [-0.1392, +0.3447] | không phân biệt được |
| `table_lookup` | +0.1512 | 43/4 | CI99% [-0.4419, +0.7442] | không phân biệt được |
| `multi_hop` | +0.1512 | 43/34 | CI99% [-0.1354, +0.4378] | không phân biệt được |

➡️ **Tập cải thiện nhiều nhất: `cross_lingual` · `aggregation` · `adversarial` · `factoid` · `table_lookup` · `multi_hop`** (6/6 nhóm) · hơn thật 0 nhóm · không kết luận được 0 nhóm.

### `hit_rate@1` theo `category`

| hạng | nhóm | n | nhãn/câu | mốc | sau | Δ | tốt↔xấu |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `cross_lingual` ⬅ đỉnh bảng | 43 | 1.0930 | 0.0000 | 0.5116 | +0.5116 | 22↔0 |
| 2 | `table_lookup` | 4 | 1.0000 | 0.0000 | 0.5000 | +0.5000 | 2↔0 |
| 3 | `factoid` | 68 | 1.0147 | 0.1471 | 0.6176 | +0.4706 | 36↔4 |
| 4 | `aggregation` | 26 | 2.4231 | 0.1923 | 0.6538 | +0.4615 | 14↔2 |
| 5 | `multi_hop` | 34 | 2.0294 | 0.2059 | 0.6471 | +0.4412 | 16↔1 |
| 6 | `adversarial` | 34 | 1.0882 | 0.0882 | 0.4706 | +0.3824 | 14↔1 |

Đỉnh bảng **chọn bằng dữ liệu**, nên 5 phép so dưới đây hiệu chỉnh Bonferroni: α = 0.01.

| `cross_lingual` hơn | khoảng cách | n | kiểm định | kết luận |
|---|---:|---:|---|---|
| `table_lookup` | +0.0116 | 43/4 | CI99% [-0.6047, +0.6279] | không phân biệt được |
| `factoid` | +0.0410 | 43/68 | CI99% [-0.2308, +0.3143] | không phân biệt được |
| `aggregation` | +0.0501 | 43/26 | CI99% [-0.3122, +0.4428] | không phân biệt được |
| `multi_hop` | +0.0705 | 43/34 | CI99% [-0.2394, +0.3865] | không phân biệt được |
| `adversarial` | +0.1293 | 43/34 | CI99% [-0.1772, +0.4405] | không phân biệt được |

➡️ **Tập cải thiện nhiều nhất: `cross_lingual` · `table_lookup` · `factoid` · `aggregation` · `multi_hop` · `adversarial`** (6/6 nhóm) · hơn thật 0 nhóm · không kết luận được 0 nhóm.

### `mrr` theo `category`

| hạng | nhóm | n | nhãn/câu | mốc | sau | Δ | tốt↔xấu |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `cross_lingual` ⬅ đỉnh bảng | 43 | 1.0930 | 0.0058 | 0.5724 | +0.5666 | 30↔0 |
| 2 | `aggregation` | 26 | 2.4231 | 0.2258 | 0.7449 | +0.5190 | 17↔2 |
| 3 | `factoid` | 68 | 1.0147 | 0.2162 | 0.7206 | +0.5044 | 49↔4 |
| 4 | `table_lookup` | 4 | 1.0000 | 0.0000 | 0.5000 | +0.5000 | 2↔0 |
| 5 | `adversarial` | 34 | 1.0882 | 0.1361 | 0.5990 | +0.4629 | 23↔1 |
| 6 | `multi_hop` | 34 | 2.0294 | 0.2720 | 0.7221 | +0.4501 | 22↔2 |

Đỉnh bảng **chọn bằng dữ liệu**, nên 5 phép so dưới đây hiệu chỉnh Bonferroni: α = 0.01.

| `cross_lingual` hơn | khoảng cách | n | kiểm định | kết luận |
|---|---:|---:|---|---|
| `aggregation` | +0.0475 | 43/26 | CI99% [-0.2611, +0.3690] | không phân biệt được |
| `factoid` | +0.0621 | 43/68 | CI99% [-0.1729, +0.2934] | không phân biệt được |
| `table_lookup` | +0.0666 | 43/4 | CI99% [-0.5296, +0.6647] | không phân biệt được |
| `adversarial` | +0.1036 | 43/34 | CI99% [-0.1623, +0.3683] | không phân biệt được |
| `multi_hop` | +0.1165 | 43/34 | CI99% [-0.1558, +0.3845] | không phân biệt được |

➡️ **Tập cải thiện nhiều nhất: `cross_lingual` · `aggregation` · `factoid` · `table_lookup` · `adversarial` · `multi_hop`** (6/6 nhóm) · hơn thật 0 nhóm · không kết luận được 0 nhóm.

