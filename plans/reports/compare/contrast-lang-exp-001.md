# `e1-baseline-dense` → `e1-rr-bgem3-reranked-onhybrid-rc100`: nhóm nào cải thiện nhiều nhất

- Chia theo `lang`. Xếp nhóm theo `Δ`, rồi kiểm **từng** nhóm còn lại có phân biệt được với đỉnh bảng hay không — câu trả lời là một **tập**.
- Bootstrap **không cặp**: hai nhóm là hai tập câu rời nhau, không có cặp nào để ghép.
- ⚠️ `Δ` của `recall@k`/`ndcg@k`/`map@k` **không** xếp hạng được giữa các nhóm — mẫu số là số nhãn, mà số nhãn/câu đổi theo nhóm.

### `hit_rate@5` theo `lang`

| hạng | nhóm | n | nhãn/câu | mốc | sau | Δ | tốt↔xấu |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `en` ⬅ đỉnh bảng | 82 | 1.5000 | 0.1707 | 0.7561 | +0.5854 | 48↔0 |
| 2 | `vi` | 127 | 1.3071 | 0.2441 | 0.8110 | +0.5669 | 72↔0 |

Đỉnh bảng **chọn bằng dữ liệu**, nên 1 phép so dưới đây hiệu chỉnh Bonferroni: α = 0.05.

| `en` hơn | khoảng cách | n | kiểm định | kết luận |
|---|---:|---:|---|---|
| `vi` | +0.0184 | 82/127 | CI95% [-0.1185, +0.1546] | không phân biệt được |

➡️ **Tập cải thiện nhiều nhất: `en` · `vi`** (2/2 nhóm) · hơn thật 0 nhóm · không kết luận được 0 nhóm.

### `hit_rate@1` theo `lang`

| hạng | nhóm | n | nhãn/câu | mốc | sau | Δ | tốt↔xấu |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `vi` ⬅ đỉnh bảng | 127 | 1.3071 | 0.1181 | 0.5906 | +0.4724 | 64↔4 |
| 2 | `en` | 82 | 1.5000 | 0.1220 | 0.5610 | +0.4390 | 40↔4 |

Đỉnh bảng **chọn bằng dữ liệu**, nên 1 phép so dưới đây hiệu chỉnh Bonferroni: α = 0.05.

| `vi` hơn | khoảng cách | n | kiểm định | kết luận |
|---|---:|---:|---|---|
| `en` | +0.0334 | 127/82 | CI95% [-0.1271, +0.1947] | không phân biệt được |

➡️ **Tập cải thiện nhiều nhất: `vi` · `en`** (2/2 nhóm) · hơn thật 0 nhóm · không kết luận được 0 nhóm.

### `mrr` theo `lang`

| hạng | nhóm | n | nhãn/câu | mốc | sau | Δ | tốt↔xấu |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `vi` ⬅ đỉnh bảng | 127 | 1.3071 | 0.1752 | 0.6840 | +0.5088 | 90↔5 |
| 2 | `en` | 82 | 1.5000 | 0.1518 | 0.6467 | +0.4949 | 53↔4 |

Đỉnh bảng **chọn bằng dữ liệu**, nên 1 phép so dưới đây hiệu chỉnh Bonferroni: α = 0.05.

| `vi` hơn | khoảng cách | n | kiểm định | kết luận |
|---|---:|---:|---|---|
| `en` | +0.0138 | 127/82 | CI95% [-0.1188, +0.1466] | không phân biệt được |

➡️ **Tập cải thiện nhiều nhất: `vi` · `en`** (2/2 nhóm) · hơn thật 0 nhóm · không kết luận được 0 nhóm.

