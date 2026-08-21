| metric | bgem3 | bgem3-rrf | Δ | kiểm định | kết luận |
|---|---:|---:|---:|---|---|
| `hit_rate@1` | 0.3397 | 0.3014 | -0.0383 | p=0.1686 · 17↔9 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@10` | 0.6268 | 0.5742 | -0.0526 | p=0.06143 · 20↔9 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@20` | 0.6746 | 0.6746 | +0.0000 | p=1 · 11↔11 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@5` | 0.5455 | 0.4689 | -0.0766 | p=0.01385 · 27↔11 câu đổi chiều | khác biệt thật |
| `map@20` | 0.3853 | 0.3583 | -0.0270 | CI95 [-0.0635, +0.0079] · 53↔58 câu khác nhau (p dấu=0.7044) · **đếm câu đi ngược Δ** (58 câu tốt hơn vs 53 câu xấu đi, mà Δ = -0.0270) | trong ngưỡng nhiễu |
| `mrr` | 0.4394 | 0.3871 | -0.0523 | CI95 [-0.0916, -0.0145] · 50↔39 câu khác nhau (p dấu=0.2891) | khác biệt thật |
| `ndcg@10` | 0.4442 | 0.4021 | -0.0421 | CI95 [-0.0785, -0.0071] · 48↔47 câu khác nhau (p dấu=1) | khác biệt thật |
| `precision@1` | 0.3397 | 0.3014 | -0.0383 | p=0.1686 · 17↔9 câu đổi chiều | trong ngưỡng nhiễu |
| `precision@10` | 0.0818 | 0.0742 | -0.0077 | CI95 [-0.0144, -0.0014] · 25↔13 câu khác nhau (p dấu=0.07295) | khác biệt thật |
| `precision@20` | 0.0445 | 0.0445 | +0.0000 | CI95 [-0.0029, +0.0026] · 13↔15 câu khác nhau (p dấu=0.8506) | trong ngưỡng nhiễu |
| `precision@5` | 0.1340 | 0.1187 | -0.0153 | CI95 [-0.0306, -0.0010] · 28↔18 câu khác nhau (p dấu=0.1839) | khác biệt thật |
| `recall@1` | 0.2512 | 0.2360 | -0.0152 | CI95 [-0.0542, +0.0239] · 17↔9 câu khác nhau (p dấu=0.1686) | trong ngưỡng nhiễu |
| `recall@10` | 0.5813 | 0.5327 | -0.0486 | CI95 [-0.0981, -0.0016] · 25↔13 câu khác nhau (p dấu=0.07295) | khác biệt thật |
| `recall@20` | 0.6324 | 0.6396 | +0.0072 | CI95 [-0.0343, +0.0478] · 13↔15 câu khác nhau (p dấu=0.8506) | trong ngưỡng nhiễu |
| `recall@5` | 0.4769 | 0.4242 | -0.0526 | CI95 [-0.1053, -0.0016] · 28↔18 câu khác nhau (p dấu=0.1839) · **biên cách 0 dưới một bước lưới** (0.00159 < 0.3333/209 = 0.00159) | KHÔNG KẾT LUẬN |
