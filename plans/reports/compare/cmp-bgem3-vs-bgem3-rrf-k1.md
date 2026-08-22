> ⚠️ **15 hàng, KHÔNG hiệu chỉnh đa so sánh** — mỗi hàng là một phép kiểm ở α = 0.05. 15 metric này không độc lập (đo lại ở `W2-09`: **7** phép kiểm hiệu dụng, `|r|` trung bình 0,45–0,83), nên số hàng "có ý nghĩa" **thuần do ngẫu nhiên** mà bảng này chờ đợi là ≈ **0.35**.
> Đọc **cả bảng** thì được; rút một hàng thuận nhất ra trích thì đó là chỗ con số trên biến thành kết luận sai.

| metric | bgem3 | bgem3-rrf-k1 | Δ | kiểm định | kết luận |
|---|---:|---:|---:|---|---|
| `hit_rate@1` | 0.3397 | 0.3397 | +0.0000 | p=1 · 9↔9 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@10` | 0.6268 | 0.6555 | +0.0287 | p=0.2863 · 8↔14 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@20` | 0.6746 | 0.7129 | +0.0383 | p=0.09625 · 5↔13 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@5` | 0.5455 | 0.5742 | +0.0287 | p=0.2101 · 5↔11 câu đổi chiều | trong ngưỡng nhiễu |
| `map@20` | 0.3853 | 0.3987 | +0.0134 | CI95 [-0.0104, +0.0377] · 56↔51 câu khác nhau (p dấu=0.6992) · **đếm câu đi ngược Δ** (51 câu tốt hơn vs 56 câu xấu đi, mà Δ = +0.0134) | trong ngưỡng nhiễu |
| `mrr` | 0.4394 | 0.4425 | +0.0031 | CI95 [-0.0230, +0.0296] · 48↔36 câu khác nhau (p dấu=0.2299) · **đếm câu đi ngược Δ** (36 câu tốt hơn vs 48 câu xấu đi, mà Δ = +0.0031) | trong ngưỡng nhiễu |
| `ndcg@10` | 0.4442 | 0.4557 | +0.0116 | CI95 [-0.0129, +0.0369] · 50↔47 câu khác nhau (p dấu=0.8392) · **đếm câu đi ngược Δ** (47 câu tốt hơn vs 50 câu xấu đi, mà Δ = +0.0116) | trong ngưỡng nhiễu |
| `precision@1` | 0.3397 | 0.3397 | +0.0000 | p=1 · 9↔9 câu đổi chiều | trong ngưỡng nhiễu |
| `precision@10` | 0.0818 | 0.0847 | +0.0029 | CI95 [-0.0024, +0.0081] · 13↔19 câu khác nhau (p dấu=0.3771) | trong ngưỡng nhiễu |
| `precision@20` | 0.0445 | 0.0471 | +0.0026 | CI95 [+0.0005, +0.0048] · 6↔17 câu khác nhau (p dấu=0.03469) | khác biệt thật |
| `precision@5` | 0.1340 | 0.1445 | +0.0105 | CI95 [+0.0010, +0.0211] · 7↔19 câu khác nhau (p dấu=0.02896) · **biên cách 0 dưới một bước lưới** (0.00096 < 0.2/209 = 0.00096) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0010] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@1` | 0.2512 | 0.2584 | +0.0072 | CI95 [-0.0263, +0.0407] · 9↔9 câu khác nhau (p dấu=1) · **đếm câu đi ngược Δ** (9 câu tốt hơn vs 9 câu xấu đi, mà Δ = +0.0072) | trong ngưỡng nhiễu |
| `recall@10` | 0.5813 | 0.6013 | +0.0199 | CI95 [-0.0247, +0.0646] · 13↔19 câu khác nhau (p dấu=0.3771) | trong ngưỡng nhiễu |
| `recall@20` | 0.6324 | 0.6754 | +0.0431 | CI95 [+0.0056, +0.0813] · 6↔17 câu khác nhau (p dấu=0.03469) | khác biệt thật |
| `recall@5` | 0.4769 | 0.5088 | +0.0319 | CI95 [-0.0032, +0.0678] · 7↔19 câu khác nhau (p dấu=0.02896) | trong ngưỡng nhiễu |
