> ⚠️ **15 hàng, KHÔNG hiệu chỉnh đa so sánh** — mỗi hàng là một phép kiểm ở α = 0.05. 15 metric này không độc lập (đo lại ở `W2-09`: **7** phép kiểm hiệu dụng, `|r|` trung bình 0,45–0,83), nên số hàng "có ý nghĩa" **thuần do ngẫu nhiên** mà bảng này chờ đợi là ≈ **0.35**.
> Đọc **cả bảng** thì được; rút một hàng thuận nhất ra trích thì đó là chỗ con số trên biến thành kết luận sai.

| metric | bgem3 | bgem3-sparse | Δ | kiểm định | kết luận |
|---|---:|---:|---:|---|---|
| `hit_rate@1` | 0.3397 | 0.2919 | -0.0478 | p=0.1325 · 23↔13 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@10` | 0.6268 | 0.5120 | -0.1148 | p=0.001842 · 40↔16 câu đổi chiều | khác biệt thật |
| `hit_rate@20` | 0.6746 | 0.5311 | -0.1435 | p=0.0001005 · 44↔14 câu đổi chiều | khác biệt thật |
| `hit_rate@5` | 0.5455 | 0.4593 | -0.0861 | p=0.01753 · 35↔17 câu đổi chiều | khác biệt thật |
| `map@20` | 0.3853 | 0.3333 | -0.0520 | CI95 [-0.0970, -0.0069] · 66↔52 câu khác nhau (p dấu=0.2313) | khác biệt thật |
| `mrr` | 0.4394 | 0.3623 | -0.0771 | CI95 [-0.1273, -0.0267] · 62↔36 câu khác nhau (p dấu=0.01117) | khác biệt thật |
| `ndcg@10` | 0.4442 | 0.3733 | -0.0709 | CI95 [-0.1190, -0.0225] · 60↔48 câu khác nhau (p dấu=0.2898) | khác biệt thật |
| `precision@1` | 0.3397 | 0.2919 | -0.0478 | p=0.1325 · 23↔13 câu đổi chiều | trong ngưỡng nhiễu |
| `precision@10` | 0.0818 | 0.0660 | -0.0158 | CI95 [-0.0249, -0.0072] · 43↔20 câu khác nhau (p dấu=0.005152) | khác biệt thật |
| `precision@20` | 0.0445 | 0.0352 | -0.0093 | CI95 [-0.0139, -0.0048] · 46↔18 câu khác nhau (p dấu=0.0006174) | khác biệt thật |
| `precision@5` | 0.1340 | 0.1167 | -0.0172 | CI95 [-0.0344, -0.0000] · 36↔25 câu khác nhau (p dấu=0.2) · **biên cách 0 dưới một bước lưới** (0.00000 < 0.2/209 = 0.00096) | KHÔNG KẾT LUẬN |
| `recall@1` | 0.2512 | 0.2225 | -0.0287 | CI95 [-0.0750, +0.0167] · 23↔13 câu khác nhau (p dấu=0.1325) | trong ngưỡng nhiễu |
| `recall@10` | 0.5813 | 0.4721 | -0.1093 | CI95 [-0.1754, -0.0439] · 43↔20 câu khác nhau (p dấu=0.005152) | khác biệt thật |
| `recall@20` | 0.6324 | 0.5024 | -0.1300 | CI95 [-0.1970, -0.0638] · 46↔18 câu khác nhau (p dấu=0.0006174) | khác biệt thật |
| `recall@5` | 0.4769 | 0.4171 | -0.0598 | CI95 [-0.1188, -0.0008] · 36↔25 câu khác nhau (p dấu=0.2) · **biên cách 0 dưới một bước lưới** (0.00080 < 0.3333/209 = 0.00159) · **biên không ổn định**: chính nó dao động [-0.0016, +0.0008] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
