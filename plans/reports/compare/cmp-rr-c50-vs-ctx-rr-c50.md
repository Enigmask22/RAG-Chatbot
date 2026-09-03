> ⚠️ **15 hàng, KHÔNG hiệu chỉnh đa so sánh** — mỗi hàng là một phép kiểm ở α = 0.05. 15 metric này không độc lập (đo lại ở `W2-09`: **7** phép kiểm hiệu dụng, `|r|` trung bình 0,45–0,83), nên số hàng "có ý nghĩa" **thuần do ngẫu nhiên** mà bảng này chờ đợi là ≈ **0.35**.
> Đọc **cả bảng** thì được; rút một hàng thuận nhất ra trích thì đó là chỗ con số trên biến thành kết luận sai.

| metric | bgem3-rr-c50 | bgem3-ctx-rr-c50 | Δ | kiểm định | kết luận |
|---|---:|---:|---:|---|---|
| `hit_rate@1` | 0.5598 | 0.6077 | +0.0478 | p=0.1539 · 15↔25 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@10` | 0.7703 | 0.8134 | +0.0431 | p=0.06357 · 5↔14 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@20` | 0.7751 | 0.8278 | +0.0526 | p=0.0266 · 5↔16 câu đổi chiều | khác biệt thật |
| `hit_rate@5` | 0.7512 | 0.8086 | +0.0574 | p=0.007538 · 3↔15 câu đổi chiều | khác biệt thật |
| `map@20` | 0.6051 | 0.6449 | +0.0398 | CI95 [+0.0036, +0.0763] · 32↔49 câu khác nhau (p dấu=0.07479) | khác biệt thật |
| `mrr` | 0.6440 | 0.6912 | +0.0472 | CI95 [+0.0067, +0.0881] · 26↔39 câu khác nhau (p dấu=0.136) | khác biệt thật |
| `ndcg@10` | 0.6481 | 0.6888 | +0.0407 | CI95 [+0.0061, +0.0760] · 34↔44 câu khác nhau (p dấu=0.3082) | khác biệt thật |
| `precision@1` | 0.5598 | 0.6077 | +0.0478 | p=0.1539 · 15↔25 câu đổi chiều | trong ngưỡng nhiễu |
| `precision@10` | 0.1010 | 0.1067 | +0.0057 | CI95 [+0.0000, +0.0115] · 10↔19 câu khác nhau (p dấu=0.136) · **biên cách 0 dưới một bước lưới** (0.00000 < 0.1/209 = 0.00048) | KHÔNG KẾT LUẬN |
| `precision@20` | 0.0510 | 0.0555 | +0.0045 | CI95 [+0.0017, +0.0077] · 6↔21 câu khác nhau (p dấu=0.005925) | khác biệt thật |
| `precision@5` | 0.1914 | 0.2105 | +0.0191 | CI95 [+0.0077, +0.0316] · 7↔23 câu khác nhau (p dấu=0.005223) | khác biệt thật |
| `recall@1` | 0.4474 | 0.4856 | +0.0383 | CI95 [-0.0112, +0.0885] · 15↔25 câu khác nhau (p dấu=0.1539) | trong ngưỡng nhiễu |
| `recall@10` | 0.7352 | 0.7759 | +0.0407 | CI95 [+0.0016, +0.0829] · 10↔19 câu khác nhau (p dấu=0.136) · **biên cách 0 dưới một bước lưới** (0.00159 < 0.3333/209 = 0.00159) | KHÔNG KẾT LUẬN |
| `recall@20` | 0.7424 | 0.7967 | +0.0542 | CI95 [+0.0152, +0.0949] · 6↔21 câu khác nhau (p dấu=0.005925) | khác biệt thật |
| `recall@5` | 0.7026 | 0.7663 | +0.0638 | CI95 [+0.0247, +0.1037] · 7↔23 câu khác nhau (p dấu=0.005223) | khác biệt thật |
