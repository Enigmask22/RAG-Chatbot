> ⚠️ **15 hàng, KHÔNG hiệu chỉnh đa so sánh** — mỗi hàng là một phép kiểm ở α = 0.05. 15 metric này không độc lập (đo lại ở `W2-09`: **7** phép kiểm hiệu dụng, `|r|` trung bình 0,45–0,83), nên số hàng "có ý nghĩa" **thuần do ngẫu nhiên** mà bảng này chờ đợi là ≈ **0.35**.
> Đọc **cả bảng** thì được; rút một hàng thuận nhất ra trích thì đó là chỗ con số trên biến thành kết luận sai.

| metric | bgem3 | bgem3-ctx | Δ | kiểm định | kết luận |
|---|---:|---:|---:|---|---|
| `hit_rate@1` | 0.3397 | 0.3971 | +0.0574 | p=0.05761 · 11↔23 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@10` | 0.6268 | 0.6842 | +0.0574 | p=0.0357 · 8↔20 câu đổi chiều | khác biệt thật |
| `hit_rate@20` | 0.6746 | 0.7464 | +0.0718 | p=0.00813 · 7↔22 câu đổi chiều | khác biệt thật |
| `hit_rate@5` | 0.5455 | 0.6172 | +0.0718 | p=0.01353 · 9↔24 câu đổi chiều | khác biệt thật |
| `map@20` | 0.3853 | 0.4489 | +0.0636 | CI95 [+0.0265, +0.1030] · 41↔78 câu khác nhau (p dấu=0.0008881) | khác biệt thật |
| `mrr` | 0.4394 | 0.4921 | +0.0527 | CI95 [+0.0125, +0.0943] · 31↔60 câu khác nhau (p dấu=0.003113) | khác biệt thật |
| `ndcg@10` | 0.4442 | 0.5019 | +0.0577 | CI95 [+0.0223, +0.0947] · 37↔67 câu khác nhau (p dấu=0.004233) | khác biệt thật |
| `precision@1` | 0.3397 | 0.3971 | +0.0574 | p=0.05761 · 11↔23 câu đổi chiều | trong ngưỡng nhiễu |
| `precision@10` | 0.0818 | 0.0890 | +0.0072 | CI95 [+0.0014, +0.0134] · 13↔27 câu khác nhau (p dấu=0.03848) | khác biệt thật |
| `precision@20` | 0.0445 | 0.0493 | +0.0048 | CI95 [+0.0019, +0.0079] · 10↔29 câu khác nhau (p dấu=0.003378) | khác biệt thật |
| `precision@5` | 0.1340 | 0.1579 | +0.0239 | CI95 [+0.0115, +0.0373] · 10↔35 câu khác nhau (p dấu=0.0002471) | khác biệt thật |
| `recall@1` | 0.2512 | 0.3030 | +0.0518 | CI95 [+0.0064, +0.0997] · 11↔23 câu khác nhau (p dấu=0.05761) | khác biệt thật |
| `recall@10` | 0.5813 | 0.6348 | +0.0534 | CI95 [+0.0064, +0.1021] · 13↔27 câu khác nhau (p dấu=0.03848) | khác biệt thật |
| `recall@20` | 0.6324 | 0.7081 | +0.0758 | CI95 [+0.0287, +0.1236] · 10↔29 câu khác nhau (p dấu=0.003378) | khác biệt thật |
| `recall@5` | 0.4769 | 0.5654 | +0.0885 | CI95 [+0.0359, +0.1419] · 10↔35 câu khác nhau (p dấu=0.0002471) | khác biệt thật |
