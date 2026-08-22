> ⚠️ **15 hàng, KHÔNG hiệu chỉnh đa so sánh** — mỗi hàng là một phép kiểm ở α = 0.05. 15 metric này không độc lập (đo lại ở `W2-09`: **7** phép kiểm hiệu dụng, `|r|` trung bình 0,45–0,83), nên số hàng "có ý nghĩa" **thuần do ngẫu nhiên** mà bảng này chờ đợi là ≈ **0.35**.
> Đọc **cả bảng** thì được; rút một hàng thuận nhất ra trích thì đó là chỗ con số trên biến thành kết luận sai.

| metric | bgem3-rr-dense-c50 | bgem3-rr-c50 | Δ | kiểm định | kết luận |
|---|---:|---:|---:|---|---|
| `hit_rate@1` | 0.5455 | 0.5598 | +0.0144 | p=0.4531 · 2↔5 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@10` | 0.7416 | 0.7703 | +0.0287 | p=0.1796 · 4↔10 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@20` | 0.7464 | 0.7751 | +0.0287 | p=0.1796 · 4↔10 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@5` | 0.7273 | 0.7512 | +0.0239 | p=0.2668 · 4↔9 câu đổi chiều | trong ngưỡng nhiễu |
| `map@20` | 0.5894 | 0.6051 | +0.0157 | CI95 [-0.0076, +0.0401] · 13↔16 câu khác nhau (p dấu=0.7111) | trong ngưỡng nhiễu |
| `mrr` | 0.6265 | 0.6440 | +0.0175 | CI95 [-0.0072, +0.0439] · 9↔12 câu khác nhau (p dấu=0.6636) | trong ngưỡng nhiễu |
| `ndcg@10` | 0.6268 | 0.6481 | +0.0213 | CI95 [-0.0032, +0.0473] · 13↔16 câu khác nhau (p dấu=0.7111) | trong ngưỡng nhiễu |
| `precision@1` | 0.5455 | 0.5598 | +0.0144 | p=0.4531 · 2↔5 câu đổi chiều | trong ngưỡng nhiễu |
| `precision@10` | 0.0967 | 0.1010 | +0.0043 | CI95 [+0.0005, +0.0081] · 4↔13 câu khác nhau (p dấu=0.04904) | khác biệt thật |
| `precision@20` | 0.0493 | 0.0510 | +0.0017 | CI95 [-0.0002, +0.0036] · 5↔12 câu khác nhau (p dấu=0.1435) | trong ngưỡng nhiễu |
| `precision@5` | 0.1866 | 0.1914 | +0.0048 | CI95 [-0.0019, +0.0115] · 4↔9 câu khác nhau (p dấu=0.2668) | trong ngưỡng nhiễu |
| `recall@1` | 0.4354 | 0.4474 | +0.0120 | CI95 [-0.0096, +0.0359] · 2↔5 câu khác nhau (p dấu=0.4531) | trong ngưỡng nhiễu |
| `recall@10` | 0.7002 | 0.7352 | +0.0351 | CI95 [+0.0048, +0.0678] · 4↔13 câu khác nhau (p dấu=0.04904) | khác biệt thật |
| `recall@20` | 0.7121 | 0.7424 | +0.0303 | CI95 [-0.0008, +0.0630] · 5↔12 câu khác nhau (p dấu=0.1435) · **biên cách 0 dưới một bước lưới** (0.00080 < 0.3333/209 = 0.00159) · **biên không ổn định**: chính nó dao động [-0.0008, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@5` | 0.6786 | 0.7026 | +0.0239 | CI95 [-0.0048, +0.0550] · 4↔9 câu khác nhau (p dấu=0.2668) | trong ngưỡng nhiễu |
