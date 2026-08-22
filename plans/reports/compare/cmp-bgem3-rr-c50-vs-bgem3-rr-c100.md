> ⚠️ **15 hàng, KHÔNG hiệu chỉnh đa so sánh** — mỗi hàng là một phép kiểm ở α = 0.05. 15 metric này không độc lập (đo lại ở `W2-09`: **7** phép kiểm hiệu dụng, `|r|` trung bình 0,45–0,83), nên số hàng "có ý nghĩa" **thuần do ngẫu nhiên** mà bảng này chờ đợi là ≈ **0.35**.
> Đọc **cả bảng** thì được; rút một hàng thuận nhất ra trích thì đó là chỗ con số trên biến thành kết luận sai.

| metric | bgem3-rr-c50 | bgem3-rr-c100 | Δ | kiểm định | kết luận |
|---|---:|---:|---:|---|---|
| `hit_rate@1` | 0.5598 | 0.5789 | +0.0191 | p=0.125 · 0↔4 câu đổi chiều · **trần `p` = 0.125** ở α=0.05 | KHÔNG ĐỦ LỰC |
| `hit_rate@10` | 0.7703 | 0.8134 | +0.0431 | p=0.003906 · 0↔9 câu đổi chiều | khác biệt thật |
| `hit_rate@20` | 0.7751 | 0.8182 | +0.0431 | p=0.003906 · 0↔9 câu đổi chiều | khác biệt thật |
| `hit_rate@5` | 0.7512 | 0.7895 | +0.0383 | p=0.02148 · 1↔9 câu đổi chiều | khác biệt thật |
| `map@20` | 0.6051 | 0.6274 | +0.0223 | CI95 [+0.0053, +0.0438] · 12↔10 câu khác nhau (p dấu=0.8318) · **đếm câu đi ngược Δ** (10 câu tốt hơn vs 12 câu xấu đi, mà Δ = +0.0223) | TRÁI CHIỀU |
| `mrr` | 0.6440 | 0.6694 | +0.0254 | CI95 [+0.0079, +0.0471] · 6↔10 câu khác nhau (p dấu=0.4545) | khác biệt thật |
| `ndcg@10` | 0.6481 | 0.6736 | +0.0255 | CI95 [+0.0075, +0.0478] · 11↔10 câu khác nhau (p dấu=1) · **đếm câu đi ngược Δ** (10 câu tốt hơn vs 11 câu xấu đi, mà Δ = +0.0255) | TRÁI CHIỀU |
| `precision@1` | 0.5598 | 0.5789 | +0.0191 | p=0.125 · 0↔4 câu đổi chiều · **trần `p` = 0.125** ở α=0.05 | KHÔNG ĐỦ LỰC |
| `precision@10` | 0.1010 | 0.1048 | +0.0038 | CI95 [+0.0010, +0.0067] · 1↔9 câu khác nhau (p dấu=0.02148) | khác biệt thật |
| `precision@20` | 0.0510 | 0.0531 | +0.0022 | CI95 [+0.0007, +0.0038] · 1↔10 câu khác nhau (p dấu=0.01172) | khác biệt thật |
| `precision@5` | 0.1914 | 0.1990 | +0.0077 | CI95 [+0.0019, +0.0134] · 1↔9 câu khác nhau (p dấu=0.02148) | khác biệt thật |
| `recall@1` | 0.4474 | 0.4665 | +0.0191 | CI95 [+0.0048, +0.0383] · 0↔4 câu khác nhau (p dấu=0.125) | khác biệt thật |
| `recall@10` | 0.7352 | 0.7679 | +0.0327 | CI95 [+0.0104, +0.0582] · 1↔9 câu khác nhau (p dấu=0.02148) | khác biệt thật |
| `recall@20` | 0.7424 | 0.7775 | +0.0351 | CI95 [+0.0128, +0.0614] · 1↔10 câu khác nhau (p dấu=0.01172) | khác biệt thật |
| `recall@5` | 0.7026 | 0.7352 | +0.0327 | CI95 [+0.0112, +0.0590] · 1↔9 câu khác nhau (p dấu=0.02148) | khác biệt thật |
