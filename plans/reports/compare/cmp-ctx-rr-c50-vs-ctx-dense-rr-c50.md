> ⚠️ **15 hàng, KHÔNG hiệu chỉnh đa so sánh** — mỗi hàng là một phép kiểm ở α = 0.05. 15 metric này không độc lập (đo lại ở `W2-09`: **7** phép kiểm hiệu dụng, `|r|` trung bình 0,45–0,83), nên số hàng "có ý nghĩa" **thuần do ngẫu nhiên** mà bảng này chờ đợi là ≈ **0.35**.
> Đọc **cả bảng** thì được; rút một hàng thuận nhất ra trích thì đó là chỗ con số trên biến thành kết luận sai.

| metric | bgem3-ctx-rr-c50 | bgem3-ctx-dense-rr-c50 | Δ | kiểm định | kết luận |
|---|---:|---:|---:|---|---|
| `hit_rate@1` | 0.6077 | 0.6077 | +0.0000 | p=1 · 3↔3 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@10` | 0.8134 | 0.8134 | +0.0000 | p=1 · 5↔5 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@20` | 0.8278 | 0.8230 | -0.0048 | p=1 · 6↔5 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@5` | 0.8086 | 0.8038 | -0.0048 | p=1 · 5↔4 câu đổi chiều | trong ngưỡng nhiễu |
| `map@20` | 0.6449 | 0.6496 | +0.0047 | CI95 [-0.0185, +0.0291] · 10↔18 câu khác nhau (p dấu=0.1849) | trong ngưỡng nhiễu |
| `mrr` | 0.6912 | 0.6876 | -0.0036 | CI95 [-0.0271, +0.0205] · 7↔7 câu khác nhau (p dấu=1) · **đếm câu đi ngược Δ** (7 câu tốt hơn vs 7 câu xấu đi, mà Δ = -0.0036) | trong ngưỡng nhiễu |
| `ndcg@10` | 0.6888 | 0.6937 | +0.0049 | CI95 [-0.0195, +0.0300] · 8↔13 câu khác nhau (p dấu=0.3833) | trong ngưỡng nhiễu |
| `precision@1` | 0.6077 | 0.6077 | +0.0000 | p=1 · 3↔3 câu đổi chiều | trong ngưỡng nhiễu |
| `precision@10` | 0.1067 | 0.1091 | +0.0024 | CI95 [-0.0014, +0.0067] · 7↔12 câu khác nhau (p dấu=0.3593) | trong ngưỡng nhiễu |
| `precision@20` | 0.0555 | 0.0562 | +0.0007 | CI95 [-0.0012, +0.0029] · 8↔11 câu khác nhau (p dấu=0.6476) | trong ngưỡng nhiễu |
| `precision@5` | 0.2105 | 0.2115 | +0.0010 | CI95 [-0.0057, +0.0077] · 6↔7 câu khác nhau (p dấu=1) | trong ngưỡng nhiễu |
| `recall@1` | 0.4856 | 0.4880 | +0.0024 | CI95 [-0.0191, +0.0239] · 3↔3 câu khác nhau (p dấu=1) · **đếm câu đi ngược Δ** (3 câu tốt hơn vs 3 câu xấu đi, mà Δ = +0.0024) | trong ngưỡng nhiễu |
| `recall@10` | 0.7759 | 0.7887 | +0.0128 | CI95 [-0.0183, +0.0447] · 7↔12 câu khác nhau (p dấu=0.3593) | trong ngưỡng nhiễu |
| `recall@20` | 0.7967 | 0.8054 | +0.0088 | CI95 [-0.0215, +0.0407] · 8↔11 câu khác nhau (p dấu=0.6476) | trong ngưỡng nhiễu |
| `recall@5` | 0.7663 | 0.7679 | +0.0016 | CI95 [-0.0263, +0.0303] · 6↔7 câu khác nhau (p dấu=1) | trong ngưỡng nhiễu |
