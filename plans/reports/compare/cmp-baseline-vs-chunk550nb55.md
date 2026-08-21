| metric | baseline | chunk550nb55 | Δ | kiểm định | kết luận |
|---|---:|---:|---:|---|---|
| `hit_rate@1` | 0.1196 | 0.0861 | -0.0335 | p=0.2649 · 18↔11 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@10` | 0.2775 | 0.2440 | -0.0335 | p=0.2962 · 20↔13 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@20` | 0.3110 | 0.2919 | -0.0191 | p=0.6076 · 19↔15 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@5` | 0.2153 | 0.1770 | -0.0383 | p=0.2295 · 21↔13 câu đổi chiều | trong ngưỡng nhiễu |
| `map@20` | 0.1349 | 0.0953 | -0.0395 | ⚠️ số nhãn/câu đổi 1.38 → 1.96; mẫu số là số nhãn nên metric tụt 29.5% kể cả khi truy hồi y nguyên | KHÔNG SO ĐƯỢC |
| `mrr` | 0.1660 | 0.1343 | -0.0317 | CI95 [-0.0744, +0.0093] · 39↔32 câu khác nhau (p dấu=0.4767) | trong ngưỡng nhiễu |
| `ndcg@10` | 0.1621 | 0.1180 | -0.0442 | ⚠️ số nhãn/câu đổi 1.38 → 1.96; mẫu số là số nhãn nên metric tụt 29.5% kể cả khi truy hồi y nguyên | KHÔNG SO ĐƯỢC |
| `precision@1` | 0.1196 | 0.0861 | -0.0335 | p=0.2649 · 18↔11 câu đổi chiều | trong ngưỡng nhiễu |
| `precision@10` | 0.0306 | 0.0268 | -0.0038 | CI95 [-0.0100, +0.0024] · 24↔17 câu khác nhau (p dấu=0.3489) | trong ngưỡng nhiễu |
| `precision@20` | 0.0187 | 0.0170 | -0.0017 | CI95 [-0.0050, +0.0017] · 26↔20 câu khác nhau (p dấu=0.4614) | trong ngưỡng nhiễu |
| `precision@5` | 0.0459 | 0.0392 | -0.0067 | CI95 [-0.0191, +0.0057] · 21↔16 câu khác nhau (p dấu=0.5114) | trong ngưỡng nhiễu |
| `recall@1` | 0.0877 | 0.0582 | -0.0295 | ⚠️ số nhãn/câu đổi 1.38 → 1.96; mẫu số là số nhãn nên metric tụt 29.5% kể cả khi truy hồi y nguyên | KHÔNG SO ĐƯỢC |
| `recall@10` | 0.2257 | 0.1591 | -0.0666 | ⚠️ số nhãn/câu đổi 1.38 → 1.96; mẫu số là số nhãn nên metric tụt 29.5% kể cả khi truy hồi y nguyên | KHÔNG SO ĐƯỢC |
| `recall@20` | 0.2663 | 0.2055 | -0.0608 | ⚠️ số nhãn/câu đổi 1.38 → 1.96; mẫu số là số nhãn nên metric tụt 29.5% kể cả khi truy hồi y nguyên | KHÔNG SO ĐƯỢC |
| `recall@5` | 0.1746 | 0.1244 | -0.0502 | ⚠️ số nhãn/câu đổi 1.38 → 1.96; mẫu số là số nhãn nên metric tụt 29.5% kể cả khi truy hồi y nguyên | KHÔNG SO ĐƯỢC |
