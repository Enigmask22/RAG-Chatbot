> ⚠️ **15 hàng, KHÔNG hiệu chỉnh đa so sánh** — mỗi hàng là một phép kiểm ở α = 0.05. 15 metric này không độc lập (đo lại ở `W2-09`: **7** phép kiểm hiệu dụng, `|r|` trung bình 0,45–0,83), nên số hàng "có ý nghĩa" **thuần do ngẫu nhiên** mà bảng này chờ đợi là ≈ **0.35**.
> Đọc **cả bảng** thì được; rút một hàng thuận nhất ra trích thì đó là chỗ con số trên biến thành kết luận sai.

| metric | bgem3-rr-c20 | bgem3-rr-c50 | Δ | kiểm định | kết luận |
|---|---:|---:|---:|---|---|
| `hit_rate@1` | 0.5407 | 0.5598 | +0.0191 | p=0.2188 · 1↔5 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@10` | 0.7129 | 0.7703 | +0.0574 | p=0.0004883 · 0↔12 câu đổi chiều | khác biệt thật |
| `hit_rate@20` | 0.7177 | 0.7751 | +0.0574 | p=0.001831 · 1↔13 câu đổi chiều | khác biệt thật |
| `hit_rate@5` | 0.7033 | 0.7512 | +0.0478 | p=0.006348 · 1↔11 câu đổi chiều | khác biệt thật |
| `map@20` | 0.5698 | 0.6051 | +0.0353 | CI95 [+0.0132, +0.0606] · 14↔17 câu khác nhau (p dấu=0.7201) | khác biệt thật |
| `mrr` | 0.6128 | 0.6440 | +0.0312 | CI95 [+0.0100, +0.0556] · 5↔13 câu khác nhau (p dấu=0.09625) | khác biệt thật |
| `ndcg@10` | 0.6075 | 0.6481 | +0.0406 | CI95 [+0.0174, +0.0664] · 12↔16 câu khác nhau (p dấu=0.5716) | khác biệt thật |
| `precision@1` | 0.5407 | 0.5598 | +0.0191 | p=0.2188 · 1↔5 câu đổi chiều | trong ngưỡng nhiễu |
| `precision@10` | 0.0938 | 0.1010 | +0.0072 | CI95 [+0.0038, +0.0110] · 1↔16 câu khác nhau (p dấu=0.0002747) | khác biệt thật |
| `precision@20` | 0.0474 | 0.0510 | +0.0036 | CI95 [+0.0017, +0.0055] · 2↔17 câu khác nhau (p dấu=0.0007286) | khác biệt thật |
| `precision@5` | 0.1809 | 0.1914 | +0.0105 | CI95 [+0.0029, +0.0182] · 3↔14 câu khác nhau (p dấu=0.01273) | khác biệt thật |
| `recall@1` | 0.4282 | 0.4474 | +0.0191 | CI95 [+0.0000, +0.0431] · 1↔5 câu khác nhau (p dấu=0.2188) · **biên cách 0 dưới một bước lưới** (0.00000 < 1/209 = 0.00478) · **biên không ổn định**: chính nó dao động [-0.0048, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@10` | 0.6738 | 0.7352 | +0.0614 | CI95 [+0.0311, +0.0949] · 1↔16 câu khác nhau (p dấu=0.0002747) | khác biệt thật |
| `recall@20` | 0.6770 | 0.7424 | +0.0654 | CI95 [+0.0343, +0.0997] · 2↔17 câu khác nhau (p dấu=0.0007286) | khác biệt thật |
| `recall@5` | 0.6555 | 0.7026 | +0.0470 | CI95 [+0.0167, +0.0805] · 3↔14 câu khác nhau (p dấu=0.01273) | khác biệt thật |
