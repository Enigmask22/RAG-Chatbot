> ⚠️ **15 hàng, KHÔNG hiệu chỉnh đa so sánh** — mỗi hàng là một phép kiểm ở α = 0.05. 15 metric này không độc lập (đo lại ở `W2-09`: **7** phép kiểm hiệu dụng, `|r|` trung bình 0,45–0,83), nên số hàng "có ý nghĩa" **thuần do ngẫu nhiên** mà bảng này chờ đợi là ≈ **0.35**.
> Đọc **cả bảng** thì được; rút một hàng thuận nhất ra trích thì đó là chỗ con số trên biến thành kết luận sai.

| metric | bgem3 | bgem3-rr-c50 | Δ | kiểm định | kết luận |
|---|---:|---:|---:|---|---|
| `hit_rate@1` | 0.3397 | 0.5598 | +0.2201 | p=3.542e-09 · 9↔55 câu đổi chiều | khác biệt thật |
| `hit_rate@10` | 0.6268 | 0.7703 | +0.1435 | p=1.863e-09 · 0↔30 câu đổi chiều | khác biệt thật |
| `hit_rate@20` | 0.6746 | 0.7751 | +0.1005 | p=9.537e-07 · 0↔21 câu đổi chiều | khác biệt thật |
| `hit_rate@5` | 0.5455 | 0.7512 | +0.2057 | p=2.274e-13 · 0↔43 câu đổi chiều | khác biệt thật |
| `map@20` | 0.3853 | 0.6051 | +0.2198 | CI95 [+0.1714, +0.2693] · 17↔101 câu khác nhau (p dấu=1.006e-15) | khác biệt thật |
| `mrr` | 0.4394 | 0.6440 | +0.2046 | CI95 [+0.1538, +0.2555] · 14↔81 câu khác nhau (p dấu=1.24e-12) | khác biệt thật |
| `ndcg@10` | 0.4442 | 0.6481 | +0.2039 | CI95 [+0.1602, +0.2490] · 16↔101 câu khác nhau (p dấu=2.867e-16) | khác biệt thật |
| `precision@1` | 0.3397 | 0.5598 | +0.2201 | p=3.542e-09 · 9↔55 câu đổi chiều | khác biệt thật |
| `precision@10` | 0.0818 | 0.1010 | +0.0191 | CI95 [+0.0134, +0.0249] · 1↔40 câu khác nhau (p dấu=3.82e-11) | khác biệt thật |
| `precision@20` | 0.0445 | 0.0510 | +0.0065 | CI95 [+0.0043, +0.0089] · 1↔28 câu khác nhau (p dấu=1.118e-07) | khác biệt thật |
| `precision@5` | 0.1340 | 0.1914 | +0.0574 | CI95 [+0.0450, +0.0708] · 1↔60 câu khác nhau (p dấu=5.378e-17) | khác biệt thật |
| `recall@1` | 0.2512 | 0.4474 | +0.1962 | CI95 [+0.1340, +0.2592] · 9↔55 câu khác nhau (p dấu=3.542e-09) | khác biệt thật |
| `recall@10` | 0.5813 | 0.7352 | +0.1539 | CI95 [+0.1093, +0.2018] · 1↔40 câu khác nhau (p dấu=3.82e-11) | khác biệt thật |
| `recall@20` | 0.6324 | 0.7424 | +0.1100 | CI95 [+0.0718, +0.1523] · 1↔28 câu khác nhau (p dấu=1.118e-07) | khác biệt thật |
| `recall@5` | 0.4769 | 0.7026 | +0.2257 | CI95 [+0.1730, +0.2799] · 1↔60 câu khác nhau (p dấu=5.378e-17) | khác biệt thật |
