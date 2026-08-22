# So theo `category`: `e1-baseline-dense` → `e1-rr-bgem3-reranked-onhybrid-rc100`

- **6 nhóm × 15 metric = 90 phép kiểm.**
- Ngưỡng đã hiệu chỉnh Bonferroni: **α = 0.0005556** (từ 0.05 chia cho 90).
- Không hiệu chỉnh thì ở α=0.05 kỳ vọng **4.5 kết quả "có ý nghĩa" thuần do ngẫu nhiên** — tức câu hỏi "nhóm nào cải thiện nhiều nhất" gần như bảo đảm tìm ra một cái, kể cả khi không có gì.
- `KHÔNG ĐỦ LỰC` = phép kiểm **không thể** đạt ý nghĩa dù kết quả thế nào (`p` nhỏ nhất McNemar cho `n` câu đổi chiều đã ≥ α). Nó **không** phải "không có khác biệt".
- `KHÔNG SO ĐƯỢC` = phân bố nhãn hai bên khác nhau nên mẫu số của metric khác nhau (`G2`/`TD-11`).
- `KHÔNG KẾT LUẬN` = một trong hai chuyện. (a) Khoảng tin cậy có **một biên gần 0 hơn một bước lưới `1/n`**: với metric rời rạc thưa, phân bố bootstrap nằm trên lưới bước `1/n`, nên việc khoảng chứa hay loại 0 phụ thuộc đúng một bước phân giải. Đã đo: đổi seed dịch biên **đúng một bước lưới**, và tăng 10.000 → 50.000 iterations **không đổi gì** — đừng chữa bằng cách tăng iterations. (b) **Đếm câu đi ngược `Δ`**: khoảng nói một hướng mà số câu tốt (b) **Biên không ổn định**: biên gần 0 nhất được đọc từ quá ít mẫu lại nên chính nó dao động qua 0. Đã đo ở `W2-08`: α đã hiệu chỉnh cho 39 phép kiểm để lại **6** mẫu lại trong đuôi của 10.000, và ở đó biên dưới **đổi dấu theo seed**; tăng lên 50.000 thì nó âm nhất quán, tức khoảng thật sự chứa 0.
- `TRÁI CHIỀU` = khoảng tin cậy đọc được **và** nó nói hai điều đối nhau: trung bình đi một hướng, còn **số câu** thì đi hướng ngược lại (thắng ít câu nhưng thắng đậm). Không phải lỗi, nhưng đừng đọc thành `hệ thống tốt hơn` — nhất là khi đổi lấy độ trễ.
- `TRÙNG KHỚP` = **0 câu khác nhau**. Đây là hàng chắc chắn nhất trong bảng, không phải hàng mơ hồ nhất — đừng đọc nó cùng nhóm với các cờ trên.

- ⚠️ **6/90 hàng `KHÔNG ĐỦ LỰC` và 11/90 hàng `KHÔNG KẾT LUẬN`.** Chia nhỏ tập đo rồi hiệu chỉnh cho 90 phép kiểm để lại rất ít lực; đọc bảng này để **loại** giả thuyết thì được, để **chọn người thắng** thì không.

## `category = factoid` — n = 68

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | e1-baseline-dense | e1-rr-bgem3-reranked-onhybrid-rc100 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.1471 | 0.6176 | +0.4706 | 68 | p=1.857e-07 · 4↔36 câu đổi chiều | khác biệt thật |
| `hit_rate@10` | 0.3824 | 0.8824 | +0.5000 | 68 | p=1.164e-10 · 0↔34 câu đổi chiều | khác biệt thật |
| `hit_rate@20` | 0.3971 | 0.8824 | +0.4853 | 68 | p=2.328e-10 · 0↔33 câu đổi chiều | khác biệt thật |
| `hit_rate@5` | 0.3088 | 0.8529 | +0.5441 | 68 | p=1.455e-11 · 0↔37 câu đổi chiều | khác biệt thật |
| `map@20` | 0.2152 | 0.7182 | +0.5030 | 68 | CI99.94% [+0.3046, +0.6915] · 4↔49 câu khác nhau (p dấu=7.054e-11) · CI95 thô [+0.3885, +0.6140] | khác biệt thật |
| `mrr` | 0.2162 | 0.7206 | +0.5044 | 68 | CI99.94% [+0.3051, +0.6929] · 4↔49 câu khác nhau (p dấu=7.054e-11) · CI95 thô [+0.3897, +0.6155] | khác biệt thật |
| `ndcg@10` | 0.2530 | 0.7596 | +0.5066 | 68 | CI99.94% [+0.3230, +0.6877] · 4↔49 câu khác nhau (p dấu=7.054e-11) · CI95 thô [+0.3999, +0.6107] | khác biệt thật |
| `precision@1` | 0.1471 | 0.6176 | +0.4706 | 68 | p=1.857e-07 · 4↔36 câu đổi chiều | khác biệt thật |
| `precision@10` | 0.0382 | 0.0897 | +0.0515 | 68 | CI99.94% [+0.0309, +0.0721] · 0↔35 câu khác nhau (p dấu=5.821e-11) · CI95 thô [+0.0397, +0.0632] | khác biệt thật |
| `precision@20` | 0.0199 | 0.0449 | +0.0250 | 68 | CI99.94% [+0.0147, +0.0353] · 0↔34 câu khác nhau (p dấu=1.164e-10) · CI95 thô [+0.0191, +0.0309] | khác biệt thật |
| `precision@5` | 0.0618 | 0.1735 | +0.1118 | 68 | CI99.94% [+0.0676, +0.1559] · 0↔37 câu khác nhau (p dấu=1.455e-11) · CI95 thô [+0.0882, +0.1353] | khác biệt thật |
| `recall@1` | 0.1471 | 0.6103 | +0.4632 | 68 | CI99.94% [+0.2059, +0.7059] · 4↔36 câu khác nhau (p dấu=1.857e-07) · CI95 thô [+0.3162, +0.6029] | khác biệt thật |
| `recall@10` | 0.3750 | 0.8824 | +0.5074 | 68 | CI99.94% [+0.3015, +0.7132] · 0↔35 câu khác nhau (p dấu=5.821e-11) · CI95 thô [+0.3897, +0.6250] | khác biệt thật |
| `recall@20` | 0.3897 | 0.8824 | +0.4926 | 68 | CI99.94% [+0.2868, +0.7059] · 0↔34 câu khác nhau (p dấu=1.164e-10) · CI95 thô [+0.3750, +0.6103] | khác biệt thật |
| `recall@5` | 0.3088 | 0.8529 | +0.5441 | 68 | CI99.94% [+0.3382, +0.7500] · 0↔37 câu khác nhau (p dấu=1.455e-11) · CI95 thô [+0.4265, +0.6618] | khác biệt thật |

## `category = cross_lingual` — n = 43

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | e1-baseline-dense | e1-rr-bgem3-reranked-onhybrid-rc100 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.0000 | 0.5116 | +0.5116 | 43 | p=4.768e-07 · 0↔22 câu đổi chiều | khác biệt thật |
| `hit_rate@10` | 0.0233 | 0.6744 | +0.6512 | 43 | p=7.451e-09 · 0↔28 câu đổi chiều | khác biệt thật |
| `hit_rate@20` | 0.0465 | 0.6977 | +0.6512 | 43 | p=7.451e-09 · 0↔28 câu đổi chiều | khác biệt thật |
| `hit_rate@5` | 0.0000 | 0.6512 | +0.6512 | 43 | p=7.451e-09 · 0↔28 câu đổi chiều | khác biệt thật |
| `map@20` | 0.0058 | 0.5693 | +0.5635 | 43 | CI99.94% [+0.3236, +0.7958] · 0↔30 câu khác nhau (p dấu=1.863e-09) · CI95 thô [+0.4263, +0.6987] | khác biệt thật |
| `mrr` | 0.0058 | 0.5724 | +0.5666 | 43 | CI99.94% [+0.3243, +0.8004] · 0↔30 câu khác nhau (p dấu=1.863e-09) · CI95 thô [+0.4283, +0.7029] | khác biệt thật |
| `ndcg@10` | 0.0083 | 0.5958 | +0.5875 | 43 | CI99.94% [+0.3436, +0.8144] · 0↔29 câu khác nhau (p dấu=3.725e-09) · CI95 thô [+0.4509, +0.7211] | khác biệt thật |
| `precision@1` | 0.0000 | 0.5116 | +0.5116 | 43 | p=4.768e-07 · 0↔22 câu đổi chiều | khác biệt thật |
| `precision@10` | 0.0023 | 0.0767 | +0.0744 | 43 | CI99.94% [+0.0419, +0.1070] · 0↔28 câu khác nhau (p dấu=7.451e-09) · CI95 thô [+0.0558, +0.0930] | khác biệt thật |
| `precision@20` | 0.0023 | 0.0395 | +0.0372 | 43 | CI99.94% [+0.0209, +0.0535] · 0↔28 câu khác nhau (p dấu=7.451e-09) · CI95 thô [+0.0279, +0.0465] | khác biệt thật |
| `precision@5` | 0.0000 | 0.1488 | +0.1488 | 43 | CI99.94% [+0.0837, +0.2140] · 0↔28 câu khác nhau (p dấu=7.451e-09) · CI95 thô [+0.1116, +0.1860] | khác biệt thật |
| `recall@1` | 0.0000 | 0.4767 | +0.4767 | 43 | CI99.94% [+0.2326, +0.7209] · 0↔22 câu khác nhau (p dấu=4.768e-07) · CI95 thô [+0.3372, +0.6163] | khác biệt thật |
| `recall@10` | 0.0233 | 0.6744 | +0.6512 | 43 | CI99.94% [+0.3953, +0.8837] · 0↔28 câu khác nhau (p dấu=7.451e-09) · CI95 thô [+0.5116, +0.7907] | khác biệt thật |
| `recall@20` | 0.0465 | 0.6977 | +0.6512 | 43 | CI99.94% [+0.3953, +0.8837] · 0↔28 câu khác nhau (p dấu=7.451e-09) · CI95 thô [+0.5116, +0.7907] | khác biệt thật |
| `recall@5` | 0.0000 | 0.6512 | +0.6512 | 43 | CI99.94% [+0.3953, +0.8837] · 0↔28 câu khác nhau (p dấu=7.451e-09) · CI95 thô [+0.5116, +0.7907] | khác biệt thật |

## `category = adversarial` — n = 34

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | e1-baseline-dense | e1-rr-bgem3-reranked-onhybrid-rc100 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.0882 | 0.4706 | +0.3824 | 34 | p=0.0009766 · 1↔14 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@10` | 0.2059 | 0.7647 | +0.5588 | 34 | p=3.815e-06 · 0↔19 câu đổi chiều | khác biệt thật |
| `hit_rate@20` | 0.2647 | 0.7647 | +0.5000 | 34 | p=1.526e-05 · 0↔17 câu đổi chiều | khác biệt thật |
| `hit_rate@5` | 0.1765 | 0.7647 | +0.5882 | 34 | p=1.907e-06 · 0↔20 câu đổi chiều | khác biệt thật |
| `map@20` | 0.1241 | 0.5774 | +0.4534 | 34 | CI99.94% [+0.2014, +0.7093] · 1↔24 câu khác nhau (p dấu=1.55e-06) · CI95 thô [+0.3084, +0.5970] | khác biệt thật |
| `mrr` | 0.1361 | 0.5990 | +0.4629 | 34 | CI99.94% [+0.2059, +0.7197] · 1↔23 câu khác nhau (p dấu=2.98e-06) · CI95 thô [+0.3157, +0.6079] | khác biệt thật |
| `ndcg@10` | 0.1392 | 0.6258 | +0.4866 | 34 | CI99.94% [+0.2391, +0.7341] · 1↔24 câu khác nhau (p dấu=1.55e-06) · CI95 thô [+0.3440, +0.6275] | khác biệt thật |
| `precision@1` | 0.0882 | 0.4706 | +0.3824 | 34 | p=0.0009766 · 1↔14 câu đổi chiều | trong ngưỡng nhiễu |
| `precision@10` | 0.0206 | 0.0824 | +0.0618 | 34 | CI99.94% [+0.0294, +0.0941] · 0↔20 câu khác nhau (p dấu=1.907e-06) · CI95 thô [+0.0441, +0.0794] | khác biệt thật |
| `precision@20` | 0.0147 | 0.0426 | +0.0279 | 34 | CI99.94% [+0.0118, +0.0471] · 0↔17 câu khác nhau (p dấu=1.526e-05) · CI95 thô [+0.0176, +0.0382] | khác biệt thật |
| `precision@5` | 0.0353 | 0.1588 | +0.1235 | 34 | CI99.94% [+0.0588, +0.1882] · 0↔20 câu khác nhau (p dấu=1.907e-06) · CI95 thô [+0.0882, +0.1588] | khác biệt thật |
| `recall@1` | 0.0735 | 0.4412 | +0.3676 | 34 | CI99.94% [+0.0588, +0.6765] · 1↔14 câu khác nhau (p dấu=0.0009766) · CI95 thô [+0.1912, +0.5441] | khác biệt thật |
| `recall@10` | 0.1912 | 0.7500 | +0.5588 | 34 | CI99.94% [+0.2647, +0.8235] · 0↔20 câu khác nhau (p dấu=1.907e-06) · CI95 thô [+0.3971, +0.7206] | khác biệt thật |
| `recall@20` | 0.2647 | 0.7647 | +0.5000 | 34 | CI99.94% [+0.2059, +0.7941] · 0↔17 câu khác nhau (p dấu=1.526e-05) · CI95 thô [+0.3235, +0.6765] | khác biệt thật |
| `recall@5` | 0.1618 | 0.7353 | +0.5735 | 34 | CI99.94% [+0.2794, +0.8529] · 0↔20 câu khác nhau (p dấu=1.907e-06) · CI95 thô [+0.4118, +0.7353] | khác biệt thật |

## `category = multi_hop` — n = 34

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | e1-baseline-dense | e1-rr-bgem3-reranked-onhybrid-rc100 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.2059 | 0.6471 | +0.4412 | 34 | p=0.0002747 · 1↔16 câu đổi chiều | khác biệt thật |
| `hit_rate@10` | 0.4118 | 0.8824 | +0.4706 | 34 | p=3.052e-05 · 0↔16 câu đổi chiều | khác biệt thật |
| `hit_rate@20` | 0.4412 | 0.8824 | +0.4412 | 34 | p=6.104e-05 · 0↔15 câu đổi chiều | khác biệt thật |
| `hit_rate@5` | 0.3529 | 0.8529 | +0.5000 | 34 | p=1.526e-05 · 0↔17 câu đổi chiều | khác biệt thật |
| `map@20` | 0.1810 | 0.6295 | +0.4485 | 34 | CI99.94% [+0.2214, +0.6758] · 3↔27 câu khác nhau (p dấu=8.43e-06) · CI95 thô [+0.3178, +0.5790] | khác biệt thật |
| `mrr` | 0.2720 | 0.7221 | +0.4501 | 34 | CI99.94% [+0.1821, +0.7078] · 2↔22 câu khác nhau (p dấu=3.588e-05) · CI95 thô [+0.2972, +0.6017] | khác biệt thật |
| `ndcg@10` | 0.2238 | 0.6947 | +0.4709 | 34 | CI99.94% [+0.2530, +0.6876] · 2↔28 câu khác nhau (p dấu=8.68e-07) · CI95 thô [+0.3449, +0.5971] | khác biệt thật |
| `precision@1` | 0.2059 | 0.6471 | +0.4412 | 34 | p=0.0002747 · 1↔16 câu đổi chiều | khác biệt thật |
| `precision@10` | 0.0559 | 0.1588 | +0.1029 | 34 | CI99.94% [+0.0559, +0.1471] · 0↔24 câu khác nhau (p dấu=1.192e-07) · CI95 thô [+0.0765, +0.1294] | khác biệt thật |
| `precision@20` | 0.0353 | 0.0794 | +0.0441 | 34 | CI99.94% [+0.0206, +0.0676] · 0↔20 câu khác nhau (p dấu=1.907e-06) · CI95 thô [+0.0309, +0.0588] | khác biệt thật |
| `precision@5` | 0.0882 | 0.2882 | +0.2000 | 34 | CI99.94% [+0.1059, +0.2882] · 1↔26 câu khác nhau (p dấu=4.172e-07) · CI95 thô [+0.1471, +0.2529] | khác biệt thật |
| `recall@1` | 0.1029 | 0.3186 | +0.2157 | 34 | CI99.94% [+0.0490, +0.3676] · 1↔16 câu khác nhau (p dấu=0.0002747) · CI95 thô [+0.1225, +0.3039] | khác biệt thật |
| `recall@10` | 0.2745 | 0.7843 | +0.5098 | 34 | CI99.94% [+0.2794, +0.7353] · 0↔24 câu khác nhau (p dấu=1.192e-07) · CI95 thô [+0.3775, +0.6422] | khác biệt thật |
| `recall@20` | 0.3480 | 0.7843 | +0.4363 | 34 | CI99.94% [+0.2010, +0.6765] · 0↔20 câu khác nhau (p dấu=1.907e-06) · CI95 thô [+0.2990, +0.5735] | khác biệt thật |
| `recall@5` | 0.2157 | 0.7108 | +0.4951 | 34 | CI99.94% [+0.2647, +0.7157] · 1↔26 câu khác nhau (p dấu=4.172e-07) · CI95 thô [+0.3627, +0.6225] | khác biệt thật |

## `category = aggregation` — n = 26

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | e1-baseline-dense | e1-rr-bgem3-reranked-onhybrid-rc100 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.1923 | 0.6538 | +0.4615 | 26 | p=0.004181 · 2↔14 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@10` | 0.3846 | 0.8846 | +0.5000 | 26 | p=0.0002441 · 0↔13 câu đổi chiều | khác biệt thật |
| `hit_rate@20` | 0.4615 | 0.8846 | +0.4231 | 26 | p=0.0009766 · 0↔11 câu đổi chiều · **trần `p` = 0.0009766** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@5` | 0.2308 | 0.8462 | +0.6154 | 26 | p=3.052e-05 · 0↔16 câu đổi chiều | khác biệt thật |
| `map@20` | 0.1128 | 0.5687 | +0.4559 | 26 | CI99.94% [+0.2234, +0.6942] · 2↔21 câu khác nhau (p dấu=6.604e-05) · CI95 thô [+0.3216, +0.5929] | khác biệt thật |
| `mrr` | 0.2258 | 0.7449 | +0.5190 | 26 | CI99.94% [+0.1569, +0.8373] · 2↔17 câu khác nhau (p dấu=0.0007286) · CI95 thô [+0.3178, +0.7091] | khác biệt thật |
| `ndcg@10` | 0.1532 | 0.6392 | +0.4860 | 26 | CI99.94% [+0.2527, +0.7242] · 1↔21 câu khác nhau (p dấu=1.097e-05) · CI95 thô [+0.3509, +0.6221] | khác biệt thật |
| `precision@1` | 0.1923 | 0.6538 | +0.4615 | 26 | p=0.004181 · 2↔14 câu đổi chiều | trong ngưỡng nhiễu |
| `precision@10` | 0.0423 | 0.1577 | +0.1154 | 26 | CI99.94% [+0.0577, +0.1808] · 0↔19 câu khác nhau (p dấu=3.815e-06) · CI95 thô [+0.0808, +0.1500] | khác biệt thật |
| `precision@20` | 0.0288 | 0.0808 | +0.0519 | 26 | CI99.94% [+0.0212, +0.0827] · 1↔19 câu khác nhau (p dấu=4.005e-05) · CI95 thô [+0.0346, +0.0692] | khác biệt thật |
| `precision@5` | 0.0462 | 0.3000 | +0.2538 | 26 | CI99.94% [+0.1308, +0.3846] · 0↔20 câu khác nhau (p dấu=1.907e-06) · CI95 thô [+0.1846, +0.3231] | khác biệt thật |
| `recall@1` | 0.0897 | 0.2949 | +0.2051 | 26 | CI99.94% [+0.0000, +0.3846] · 2↔14 câu khác nhau (p dấu=0.004181) · CI95 thô [+0.0962, +0.3077] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.3333/26 = 0.01282) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0064] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@10` | 0.1859 | 0.6667 | +0.4808 | 26 | CI99.94% [+0.2436, +0.7244] · 0↔19 câu khác nhau (p dấu=3.815e-06) · CI95 thô [+0.3397, +0.6218] | khác biệt thật |
| `recall@20` | 0.2436 | 0.6859 | +0.4423 | 26 | CI99.94% [+0.1987, +0.6859] · 1↔19 câu khác nhau (p dấu=4.005e-05) · CI95 thô [+0.3013, +0.5833] | khác biệt thật |
| `recall@5` | 0.1026 | 0.6346 | +0.5321 | 26 | CI99.94% [+0.2885, +0.7692] · 0↔20 câu khác nhau (p dấu=1.907e-06) · CI95 thô [+0.3910, +0.6667] | khác biệt thật |

## `category = table_lookup` — n = 4

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | e1-baseline-dense | e1-rr-bgem3-reranked-onhybrid-rc100 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.0000 | 0.5000 | +0.5000 | 4 | p=0.5 · 0↔2 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@10` | 0.0000 | 0.5000 | +0.5000 | 4 | p=0.5 · 0↔2 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@20` | 0.0000 | 0.5000 | +0.5000 | 4 | p=0.5 · 0↔2 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@5` | 0.0000 | 0.5000 | +0.5000 | 4 | p=0.5 · 0↔2 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `map@20` | 0.0000 | 0.5000 | +0.5000 | 4 | CI99.94% [+0.0000, +1.0000] · 0↔2 câu khác nhau (p dấu=0.5) · CI95 thô [+0.0000, +1.0000] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/4 = 0.25000) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `mrr` | 0.0000 | 0.5000 | +0.5000 | 4 | CI99.94% [+0.0000, +1.0000] · 0↔2 câu khác nhau (p dấu=0.5) · CI95 thô [+0.0000, +1.0000] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/4 = 0.25000) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `ndcg@10` | 0.0000 | 0.5000 | +0.5000 | 4 | CI99.94% [+0.0000, +1.0000] · 0↔2 câu khác nhau (p dấu=0.5) · CI95 thô [+0.0000, +1.0000] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/4 = 0.25000) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `precision@1` | 0.0000 | 0.5000 | +0.5000 | 4 | p=0.5 · 0↔2 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `precision@10` | 0.0000 | 0.0500 | +0.0500 | 4 | CI99.94% [+0.0000, +0.1000] · 0↔2 câu khác nhau (p dấu=0.5) · CI95 thô [+0.0000, +0.1000] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.1/4 = 0.02500) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `precision@20` | 0.0000 | 0.0250 | +0.0250 | 4 | CI99.94% [+0.0000, +0.0500] · 0↔2 câu khác nhau (p dấu=0.5) · CI95 thô [+0.0000, +0.0500] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.05/4 = 0.01250) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `precision@5` | 0.0000 | 0.1000 | +0.1000 | 4 | CI99.94% [+0.0000, +0.2000] · 0↔2 câu khác nhau (p dấu=0.5) · CI95 thô [+0.0000, +0.2000] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.2/4 = 0.05000) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@1` | 0.0000 | 0.5000 | +0.5000 | 4 | CI99.94% [+0.0000, +1.0000] · 0↔2 câu khác nhau (p dấu=0.5) · CI95 thô [+0.0000, +1.0000] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/4 = 0.25000) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@10` | 0.0000 | 0.5000 | +0.5000 | 4 | CI99.94% [+0.0000, +1.0000] · 0↔2 câu khác nhau (p dấu=0.5) · CI95 thô [+0.0000, +1.0000] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/4 = 0.25000) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@20` | 0.0000 | 0.5000 | +0.5000 | 4 | CI99.94% [+0.0000, +1.0000] · 0↔2 câu khác nhau (p dấu=0.5) · CI95 thô [+0.0000, +1.0000] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/4 = 0.25000) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@5` | 0.0000 | 0.5000 | +0.5000 | 4 | CI99.94% [+0.0000, +1.0000] · 0↔2 câu khác nhau (p dấu=0.5) · CI95 thô [+0.0000, +1.0000] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/4 = 0.25000) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |

