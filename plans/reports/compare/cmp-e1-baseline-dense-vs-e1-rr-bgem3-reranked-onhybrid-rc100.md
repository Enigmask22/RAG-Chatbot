> ⚠️ **15 hàng, KHÔNG hiệu chỉnh đa so sánh** — mỗi hàng là một phép kiểm ở α = 0.05. 15 metric này không độc lập (đo lại ở `W2-09`: **7** phép kiểm hiệu dụng, `|r|` trung bình 0,45–0,83), nên số hàng "có ý nghĩa" **thuần do ngẫu nhiên** mà bảng này chờ đợi là ≈ **0.35**.
> Đọc **cả bảng** thì được; rút một hàng thuận nhất ra trích thì đó là chỗ con số trên biến thành kết luận sai.

| metric | e1-baseline-dense | e1-rr-bgem3-reranked-onhybrid-rc100 | Δ | kiểm định | kết luận |
|---|---:|---:|---:|---|---|
| `hit_rate@1` | 0.1196 | 0.5789 | +0.4593 | p=1.981e-22 · 8↔104 câu đổi chiều | khác biệt thật |
| `hit_rate@10` | 0.2775 | 0.8134 | +0.5359 | p=3.852e-34 · 0↔112 câu đổi chiều | khác biệt thật |
| `hit_rate@20` | 0.3110 | 0.8182 | +0.5072 | p=2.465e-32 · 0↔106 câu đổi chiều | khác biệt thật |
| `hit_rate@5` | 0.2153 | 0.7895 | +0.5742 | p=1.505e-36 · 0↔120 câu đổi chiều | khác biệt thật |
| `map@20` | 0.1349 | 0.6274 | +0.4926 | CI95 [+0.4333, +0.5511] · 10↔153 câu khác nhau (p dấu=5.034e-34) | khác biệt thật |
| `mrr` | 0.1660 | 0.6694 | +0.5034 | CI95 [+0.4386, +0.5652] · 9↔143 câu khác nhau (p dấu=3.502e-32) | khác biệt thật |
| `ndcg@10` | 0.1621 | 0.6736 | +0.5115 | CI95 [+0.4539, +0.5684] · 8↔153 câu khác nhau (p dấu=6.77e-36) | khác biệt thật |
| `precision@1` | 0.1196 | 0.5789 | +0.4593 | p=1.981e-22 · 8↔104 câu đổi chiều | khác biệt thật |
| `precision@10` | 0.0306 | 0.1048 | +0.0742 | CI95 [+0.0646, +0.0833] · 0↔128 câu khác nhau (p dấu=5.877e-39) | khác biệt thật |
| `precision@20` | 0.0187 | 0.0531 | +0.0344 | CI95 [+0.0297, +0.0390] · 1↔120 câu khác nhau (p dấu=9.178e-35) | khác biệt thật |
| `precision@5` | 0.0459 | 0.1990 | +0.1531 | CI95 [+0.1340, +0.1713] · 1↔133 câu khác nhau (p dấu=1.24e-38) | khác biệt thật |
| `recall@1` | 0.0877 | 0.4665 | +0.3788 | CI95 [+0.3110, +0.4458] · 8↔104 câu khác nhau (p dấu=1.981e-22) | khác biệt thật |
| `recall@10` | 0.2257 | 0.7679 | +0.5423 | CI95 [+0.4777, +0.6045] · 0↔128 câu khác nhau (p dấu=5.877e-39) | khác biệt thật |
| `recall@20` | 0.2663 | 0.7775 | +0.5112 | CI95 [+0.4466, +0.5742] · 1↔120 câu khác nhau (p dấu=9.178e-35) | khác biệt thật |
| `recall@5` | 0.1746 | 0.7352 | +0.5606 | CI95 [+0.4976, +0.6220] · 1↔133 câu khác nhau (p dấu=1.24e-38) | khác biệt thật |
