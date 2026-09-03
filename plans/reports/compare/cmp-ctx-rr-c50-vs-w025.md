> ⚠️ **15 hàng, KHÔNG hiệu chỉnh đa so sánh** — mỗi hàng là một phép kiểm ở α = 0.05. 15 metric này không độc lập (đo lại ở `W2-09`: **7** phép kiểm hiệu dụng, `|r|` trung bình 0,45–0,83), nên số hàng "có ý nghĩa" **thuần do ngẫu nhiên** mà bảng này chờ đợi là ≈ **0.35**.
> Đọc **cả bảng** thì được; rút một hàng thuận nhất ra trích thì đó là chỗ con số trên biến thành kết luận sai.

| metric | bgem3-ctx-rr-c50 | bgem3-ctx-rr-c50-w025 | Δ | kiểm định | kết luận |
|---|---:|---:|---:|---|---|
| `hit_rate@1` | 0.6077 | 0.6220 | +0.0144 | p=0.25 · 0↔3 câu đổi chiều · **trần `p` = 0.25** ở α=0.05 | KHÔNG ĐỦ LỰC |
| `hit_rate@10` | 0.8134 | 0.8325 | +0.0191 | p=0.2188 · 1↔5 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@20` | 0.8278 | 0.8469 | +0.0191 | p=0.2188 · 1↔5 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@5` | 0.8086 | 0.8230 | +0.0144 | p=0.375 · 1↔4 câu đổi chiều · **trần `p` = 0.0625** ở α=0.05 | KHÔNG ĐỦ LỰC |
| `map@20` | 0.6449 | 0.6636 | +0.0186 | CI95 [+0.0029, +0.0386] · 5↔17 câu khác nhau (p dấu=0.0169) | khác biệt thật |
| `mrr` | 0.6912 | 0.7047 | +0.0135 | CI95 [-0.0006, +0.0323] · 2↔8 câu khác nhau (p dấu=0.1094) | trong ngưỡng nhiễu |
| `ndcg@10` | 0.6888 | 0.7079 | +0.0191 | CI95 [+0.0027, +0.0391] · 4↔12 câu khác nhau (p dấu=0.07681) | khác biệt thật |
| `precision@1` | 0.6077 | 0.6220 | +0.0144 | p=0.25 · 0↔3 câu đổi chiều · **trần `p` = 0.25** ở α=0.05 | KHÔNG ĐỦ LỰC |
| `precision@10` | 0.1067 | 0.1105 | +0.0038 | CI95 [+0.0005, +0.0077] · 3↔11 câu khác nhau (p dấu=0.05737) | khác biệt thật |
| `precision@20` | 0.0555 | 0.0577 | +0.0022 | CI95 [+0.0005, +0.0041] · 3↔12 câu khác nhau (p dấu=0.03516) | khác biệt thật |
| `precision@5` | 0.2105 | 0.2153 | +0.0048 | CI95 [-0.0000, +0.0105] · 2↔7 câu khác nhau (p dấu=0.1797) · **biên cách 0 dưới một bước lưới** (0.00000 < 0.2/209 = 0.00096) | KHÔNG KẾT LUẬN |
| `recall@1` | 0.4856 | 0.5000 | +0.0144 | CI95 [+0.0000, +0.0335] · 0↔3 câu khác nhau (p dấu=0.25) · **biên cách 0 dưới một bước lưới** (0.00000 < 1/209 = 0.00478) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@10` | 0.7759 | 0.8022 | +0.0263 | CI95 [+0.0024, +0.0526] · 3↔11 câu khác nhau (p dấu=0.05737) | khác biệt thật |
| `recall@20` | 0.7967 | 0.8246 | +0.0279 | CI95 [+0.0040, +0.0542] · 3↔12 câu khác nhau (p dấu=0.03516) | khác biệt thật |
| `recall@5` | 0.7663 | 0.7847 | +0.0183 | CI95 [-0.0024, +0.0423] · 2↔7 câu khác nhau (p dấu=0.1797) | trong ngưỡng nhiễu |
