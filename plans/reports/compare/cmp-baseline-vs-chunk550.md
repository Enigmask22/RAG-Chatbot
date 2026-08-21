| metric | baseline | chunk550 | Δ | kiểm định | kết luận |
|---|---:|---:|---:|---|---|
| `hit_rate@1` | 0.1196 | 0.0861 | -0.0335 | p=0.2478 · 17↔10 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@10` | 0.2775 | 0.2584 | -0.0191 | p=0.5847 · 17↔13 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@20` | 0.3110 | 0.2823 | -0.0287 | p=0.3915 · 20↔14 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@5` | 0.2153 | 0.2010 | -0.0144 | p=0.7111 · 16↔13 câu đổi chiều | trong ngưỡng nhiễu |
| `map@20` | 0.1349 | 0.0921 | -0.0427 | ⚠️ số nhãn/câu đổi 1.38 → 1.96; mẫu số là số nhãn nên metric tụt 29.5% kể cả khi truy hồi y nguyên | KHÔNG SO ĐƯỢC |
| `mrr` | 0.1660 | 0.1414 | -0.0246 | CI95 [-0.0624, +0.0126] | trong ngưỡng nhiễu |
| `ndcg@10` | 0.1621 | 0.1215 | -0.0407 | ⚠️ số nhãn/câu đổi 1.38 → 1.96; mẫu số là số nhãn nên metric tụt 29.5% kể cả khi truy hồi y nguyên | KHÔNG SO ĐƯỢC |
| `precision@1` | 0.1196 | 0.0861 | -0.0335 | p=0.2478 · 17↔10 câu đổi chiều | trong ngưỡng nhiễu |
| `precision@10` | 0.0306 | 0.0282 | -0.0024 | CI95 [-0.0081, +0.0033] | trong ngưỡng nhiễu |
| `precision@20` | 0.0187 | 0.0165 | -0.0022 | CI95 [-0.0055, +0.0010] | trong ngưỡng nhiễu |
| `precision@5` | 0.0459 | 0.0431 | -0.0029 | CI95 [-0.0144, +0.0077] | trong ngưỡng nhiễu |
| `recall@1` | 0.0877 | 0.0524 | -0.0353 | ⚠️ số nhãn/câu đổi 1.38 → 1.96; mẫu số là số nhãn nên metric tụt 29.5% kể cả khi truy hồi y nguyên | KHÔNG SO ĐƯỢC |
| `recall@10` | 0.2257 | 0.1695 | -0.0561 | ⚠️ số nhãn/câu đổi 1.38 → 1.96; mẫu số là số nhãn nên metric tụt 29.5% kể cả khi truy hồi y nguyên | KHÔNG SO ĐƯỢC |
| `recall@20` | 0.2663 | 0.1919 | -0.0745 | ⚠️ số nhãn/câu đổi 1.38 → 1.96; mẫu số là số nhãn nên metric tụt 29.5% kể cả khi truy hồi y nguyên | KHÔNG SO ĐƯỢC |
| `recall@5` | 0.1746 | 0.1295 | -0.0451 | ⚠️ số nhãn/câu đổi 1.38 → 1.96; mẫu số là số nhãn nên metric tụt 29.5% kể cả khi truy hồi y nguyên | KHÔNG SO ĐƯỢC |
