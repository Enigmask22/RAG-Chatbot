# So theo `category`: `bgem3` → `bgem3-rrf-k1-c20`

- **6 nhóm × 15 metric = 90 phép kiểm.**
- Ngưỡng đã hiệu chỉnh Bonferroni: **α = 0.0005556** (từ 0.05 chia cho 90).
- Không hiệu chỉnh thì ở α=0.05 kỳ vọng **4.5 kết quả "có ý nghĩa" thuần do ngẫu nhiên** — tức câu hỏi "nhóm nào cải thiện nhiều nhất" gần như bảo đảm tìm ra một cái, kể cả khi không có gì.
- `KHÔNG ĐỦ LỰC` = phép kiểm **không thể** đạt ý nghĩa dù kết quả thế nào (`p` nhỏ nhất McNemar cho `n` câu đổi chiều đã ≥ α). Nó **không** phải "không có khác biệt".
- `KHÔNG SO ĐƯỢC` = phân bố nhãn hai bên khác nhau nên mẫu số của metric khác nhau (`G2`/`TD-11`).
- `KHÔNG KẾT LUẬN` = một trong hai chuyện. (a) Khoảng tin cậy có **một biên gần 0 hơn một bước lưới `1/n`**: với metric rời rạc thưa, phân bố bootstrap nằm trên lưới bước `1/n`, nên việc khoảng chứa hay loại 0 phụ thuộc đúng một bước phân giải. Đã đo: đổi seed dịch biên **đúng một bước lưới**, và tăng 10.000 → 50.000 iterations **không đổi gì** — đừng chữa bằng cách tăng iterations. (b) **Đếm câu đi ngược `Δ`**: khoảng nói một hướng mà số câu tốt (b) **Biên không ổn định**: biên gần 0 nhất được đọc từ quá ít mẫu lại nên chính nó dao động qua 0. Đã đo ở `W2-08`: α đã hiệu chỉnh cho 39 phép kiểm để lại **6** mẫu lại trong đuôi của 10.000, và ở đó biên dưới **đổi dấu theo seed**; tăng lên 50.000 thì nó âm nhất quán, tức khoảng thật sự chứa 0.
- `TRÁI CHIỀU` = khoảng tin cậy đọc được **và** nó nói hai điều đối nhau: trung bình đi một hướng, còn **số câu** thì đi hướng ngược lại (thắng ít câu nhưng thắng đậm). Không phải lỗi, nhưng đừng đọc thành `hệ thống tốt hơn` — nhất là khi đổi lấy độ trễ.
- `TRÙNG KHỚP` = **0 câu khác nhau**. Đây là hàng chắc chắn nhất trong bảng, không phải hàng mơ hồ nhất — đừng đọc nó cùng nhóm với các cờ trên.

- ⚠️ **30/90 hàng `KHÔNG ĐỦ LỰC` và 25/90 hàng `KHÔNG KẾT LUẬN`.** Chia nhỏ tập đo rồi hiệu chỉnh cho 90 phép kiểm để lại rất ít lực; đọc bảng này để **loại** giả thuyết thì được, để **chọn người thắng** thì không.

## `category = factoid` — n = 68

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | bgem3 | bgem3-rrf-k1-c20 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.3529 | 0.4265 | +0.0735 | 68 | p=0.125 · 1↔6 câu đổi chiều · **trần `p` = 0.01562** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@10` | 0.7059 | 0.7941 | +0.0882 | 68 | p=0.1094 · 2↔8 câu đổi chiều · **trần `p` = 0.001953** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@20` | 0.7500 | 0.8529 | +0.1029 | 68 | p=0.03906 · 1↔8 câu đổi chiều · **trần `p` = 0.003906** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@5` | 0.6029 | 0.6765 | +0.0735 | 68 | p=0.0625 · 0↔5 câu đổi chiều · **trần `p` = 0.0625** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `map@20` | 0.4681 | 0.5340 | +0.0659 | 68 | CI99.94% [-0.0093, +0.1498] · 9↔24 câu khác nhau (p dấu=0.01353) · CI95 thô [+0.0222, +0.1120] | trong ngưỡng nhiễu |
| `mrr` | 0.4718 | 0.5364 | +0.0646 | 68 | CI99.94% [-0.0102, +0.1488] · 9↔23 câu khác nhau (p dấu=0.02006) · CI95 thô [+0.0210, +0.1108] | trong ngưỡng nhiễu |
| `ndcg@10` | 0.5241 | 0.5923 | +0.0682 | 68 | CI99.94% [-0.0071, +0.1499] · 8↔22 câu khác nhau (p dấu=0.01612) · CI95 thô [+0.0240, +0.1141] | trong ngưỡng nhiễu |
| `precision@1` | 0.3529 | 0.4265 | +0.0735 | 68 | p=0.125 · 1↔6 câu đổi chiều · **trần `p` = 0.01562** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `precision@10` | 0.0721 | 0.0809 | +0.0088 | 68 | CI99.94% [-0.0059, +0.0250] · 2↔8 câu khác nhau (p dấu=0.1094) · CI95 thô [+0.0000, +0.0176] | trong ngưỡng nhiễu |
| `precision@20` | 0.0382 | 0.0434 | +0.0051 | 68 | CI99.94% [-0.0015, +0.0132] · 1↔8 câu khác nhau (p dấu=0.03906) · CI95 thô [+0.0015, +0.0096] | trong ngưỡng nhiễu |
| `precision@5` | 0.1235 | 0.1382 | +0.0147 | 68 | CI99.94% [+0.0000, +0.0412] · 0↔5 câu khác nhau (p dấu=0.0625) · CI95 thô [+0.0029, +0.0294] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.2/68 = 0.00294) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@1` | 0.3456 | 0.4191 | +0.0735 | 68 | CI99.94% [-0.0441, +0.2206] · 1↔6 câu khác nhau (p dấu=0.125) · CI95 thô [+0.0000, +0.1471] | trong ngưỡng nhiễu |
| `recall@10` | 0.7059 | 0.7941 | +0.0882 | 68 | CI99.94% [-0.0588, +0.2500] · 2↔8 câu khác nhau (p dấu=0.1094) · CI95 thô [+0.0000, +0.1765] | trong ngưỡng nhiễu |
| `recall@20` | 0.7500 | 0.8529 | +0.1029 | 68 | CI99.94% [-0.0294, +0.2647] · 1↔8 câu khác nhau (p dấu=0.03906) · CI95 thô [+0.0294, +0.1912] | trong ngưỡng nhiễu |
| `recall@5` | 0.6029 | 0.6765 | +0.0735 | 68 | CI99.94% [+0.0000, +0.2059] · 0↔5 câu khác nhau (p dấu=0.0625) · CI95 thô [+0.0147, +0.1471] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/68 = 0.01471) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |

## `category = cross_lingual` — n = 43

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | bgem3 | bgem3-rrf-k1-c20 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.0930 | 0.0465 | -0.0465 | 43 | p=0.5 · 2↔0 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@10` | 0.4419 | 0.3488 | -0.0930 | 43 | p=0.2188 · 5↔1 câu đổi chiều · **trần `p` = 0.03125** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@20` | 0.5116 | 0.4651 | -0.0465 | 43 | p=0.5 · 2↔0 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@5` | 0.3256 | 0.2326 | -0.0930 | 43 | p=0.125 · 4↔0 câu đổi chiều · **trần `p` = 0.125** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `map@20` | 0.1963 | 0.1306 | -0.0657 | 43 | CI99.94% [-0.1295, -0.0190] · 19↔1 câu khác nhau (p dấu=4.005e-05) · CI95 thô [-0.0999, -0.0361] | khác biệt thật |
| `mrr` | 0.2028 | 0.1349 | -0.0679 | 43 | CI99.94% [-0.1388, -0.0186] · 19↔1 câu khác nhau (p dấu=4.005e-05) · CI95 thô [-0.1057, -0.0359] | khác biệt thật |
| `ndcg@10` | 0.2538 | 0.1707 | -0.0831 | 43 | CI99.94% [-0.1631, -0.0086] · 17↔1 câu khác nhau (p dấu=0.000145) · CI95 thô [-0.1279, -0.0403] | khác biệt thật |
| `precision@1` | 0.0930 | 0.0465 | -0.0465 | 43 | p=0.5 · 2↔0 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `precision@10` | 0.0535 | 0.0395 | -0.0140 | 43 | CI99.94% [-0.0372, +0.0070] · 7↔1 câu khác nhau (p dấu=0.07031) · CI95 thô [-0.0256, -0.0023] | trong ngưỡng nhiễu |
| `precision@20` | 0.0302 | 0.0279 | -0.0023 | 43 | CI99.94% [-0.0093, +0.0000] · 2↔0 câu khác nhau (p dấu=0.5) · CI95 thô [-0.0058, +0.0000] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.05/43 = 0.00116) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `precision@5` | 0.0744 | 0.0512 | -0.0233 | 43 | CI99.94% [-0.0744, +0.0000] · 4↔0 câu khác nhau (p dấu=0.125) · CI95 thô [-0.0512, -0.0047] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.2/43 = 0.00465) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@1` | 0.0814 | 0.0465 | -0.0349 | 43 | CI99.94% [-0.1512, +0.0000] · 2↔0 câu khác nhau (p dấu=0.5) · CI95 thô [-0.0930, +0.0000] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.5/43 = 0.01163) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@10` | 0.4419 | 0.3256 | -0.1163 | 43 | CI99.94% [-0.3256, +0.0698] · 7↔1 câu khác nhau (p dấu=0.07031) · CI95 thô [-0.2326, -0.0116] | trong ngưỡng nhiễu |
| `recall@20` | 0.5116 | 0.4651 | -0.0465 | 43 | CI99.94% [-0.1860, +0.0000] · 2↔0 câu khác nhau (p dấu=0.5) · CI95 thô [-0.1163, +0.0000] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/43 = 0.02326) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@5` | 0.3023 | 0.2093 | -0.0930 | 43 | CI99.94% [-0.2791, +0.0000] · 4↔0 câu khác nhau (p dấu=0.125) · CI95 thô [-0.1860, -0.0233] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/43 = 0.02326) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |

## `category = adversarial` — n = 34

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | bgem3 | bgem3-rrf-k1-c20 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.2941 | 0.2647 | -0.0294 | 34 | p=1 · 2↔1 câu đổi chiều · **trần `p` = 0.25** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@10` | 0.5000 | 0.5000 | +0.0000 | 34 | p=1 · 1↔1 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@20` | 0.5588 | 0.5882 | +0.0294 | 34 | p=1 · 1↔2 câu đổi chiều · **trần `p` = 0.25** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@5` | 0.4412 | 0.4706 | +0.0294 | 34 | p=1 · 0↔1 câu đổi chiều · **trần `p` = 1** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `map@20` | 0.3665 | 0.3447 | -0.0219 | 34 | CI99.94% [-0.1290, +0.0936] · 8↔5 câu khác nhau (p dấu=0.5811) · CI95 thô [-0.0814, +0.0402] | trong ngưỡng nhiễu |
| `mrr` | 0.3703 | 0.3519 | -0.0185 | 34 | CI99.94% [-0.1254, +0.0956] · 7↔4 câu khác nhau (p dấu=0.5488) · CI95 thô [-0.0775, +0.0428] | trong ngưỡng nhiễu |
| `ndcg@10` | 0.3974 | 0.3811 | -0.0163 | 34 | CI99.94% [-0.1079, +0.0802] · 6↔4 câu khác nhau (p dấu=0.7539) · CI95 thô [-0.0680, +0.0364] | trong ngưỡng nhiễu |
| `precision@1` | 0.2941 | 0.2647 | -0.0294 | 34 | p=1 · 2↔1 câu đổi chiều · **trần `p` = 0.25** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `precision@10` | 0.0559 | 0.0559 | +0.0000 | 34 | CI99.94% [-0.0147, +0.0147] · 1↔1 câu khác nhau (p dấu=1) · CI95 thô [-0.0088, +0.0088] | trong ngưỡng nhiễu |
| `precision@20` | 0.0309 | 0.0324 | +0.0015 | 34 | CI99.94% [-0.0074, +0.0118] · 1↔2 câu khác nhau (p dấu=1) · CI95 thô [-0.0029, +0.0059] | trong ngưỡng nhiễu |
| `precision@5` | 0.0941 | 0.1059 | +0.0118 | 34 | CI99.94% [+0.0000, +0.0471] · 0↔2 câu khác nhau (p dấu=0.5) · CI95 thô [+0.0000, +0.0294] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.2/34 = 0.00588) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@1` | 0.2794 | 0.2500 | -0.0294 | 34 | CI99.94% [-0.2353, +0.1471] · 2↔1 câu khác nhau (p dấu=1) · CI95 thô [-0.1176, +0.0588] | trong ngưỡng nhiễu |
| `recall@10` | 0.5000 | 0.5000 | +0.0000 | 34 | CI99.94% [-0.1471, +0.1471] · 1↔1 câu khác nhau (p dấu=1) · CI95 thô [-0.0882, +0.0882] | trong ngưỡng nhiễu |
| `recall@20` | 0.5441 | 0.5735 | +0.0294 | 34 | CI99.94% [-0.1471, +0.2353] · 1↔2 câu khác nhau (p dấu=1) · CI95 thô [-0.0588, +0.1176] | trong ngưỡng nhiễu |
| `recall@5` | 0.4265 | 0.4706 | +0.0441 | 34 | CI99.94% [+0.0000, +0.1912] · 0↔2 câu khác nhau (p dấu=0.5) · CI95 thô [+0.0000, +0.1176] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.5/34 = 0.01471) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |

## `category = multi_hop` — n = 34

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | bgem3 | bgem3-rrf-k1-c20 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.5588 | 0.5588 | +0.0000 | 34 | p=1 · 2↔2 câu đổi chiều · **trần `p` = 0.125** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@10` | 0.7059 | 0.7647 | +0.0588 | 34 | p=0.5 · 0↔2 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@20` | 0.7647 | 0.7941 | +0.0294 | 34 | p=1 · 0↔1 câu đổi chiều · **trần `p` = 1** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@5` | 0.6765 | 0.7353 | +0.0588 | 34 | p=0.5 · 0↔2 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `map@20` | 0.4855 | 0.5223 | +0.0368 | 34 | CI99.94% [-0.0685, +0.1471] · 10↔12 câu khác nhau (p dấu=0.8318) · CI95 thô [-0.0235, +0.0989] | trong ngưỡng nhiễu |
| `mrr` | 0.6184 | 0.6256 | +0.0072 | 34 | CI99.94% [-0.1147, +0.1621] · 6↔4 câu khác nhau (p dấu=0.7539) · CI95 thô [-0.0637, +0.0876] · **đếm câu đi ngược Δ** (4 câu tốt hơn vs 6 câu xấu đi, mà Δ = +0.0072) | trong ngưỡng nhiễu |
| `ndcg@10` | 0.5522 | 0.5861 | +0.0339 | 34 | CI99.94% [-0.0704, +0.1507] · 9↔11 câu khác nhau (p dấu=0.8238) · CI95 thô [-0.0255, +0.0965] | trong ngưỡng nhiễu |
| `precision@1` | 0.5588 | 0.5588 | +0.0000 | 34 | p=1 · 2↔2 câu đổi chiều · **trần `p` = 0.125** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `precision@10` | 0.1294 | 0.1382 | +0.0088 | 34 | CI99.94% [-0.0176, +0.0353] · 2↔5 câu khác nhau (p dấu=0.4531) · CI95 thô [-0.0059, +0.0235] | trong ngưỡng nhiễu |
| `precision@20` | 0.0706 | 0.0750 | +0.0044 | 34 | CI99.94% [+0.0000, +0.0147] · 0↔3 câu khác nhau (p dấu=0.25) · CI95 thô [+0.0000, +0.0103] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.05/34 = 0.00147) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `precision@5` | 0.2059 | 0.2412 | +0.0353 | 34 | CI99.94% [+0.0000, +0.0824] · 0↔6 câu khác nhau (p dấu=0.03125) · CI95 thô [+0.0118, +0.0647] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.2/34 = 0.00588) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@1` | 0.2745 | 0.2745 | +0.0000 | 34 | CI99.94% [-0.1029, +0.1029] · 2↔2 câu khác nhau (p dấu=1) · CI95 thô [-0.0588, +0.0588] | trong ngưỡng nhiễu |
| `recall@10` | 0.6324 | 0.6765 | +0.0441 | 34 | CI99.94% [-0.0882, +0.1765] · 2↔5 câu khác nhau (p dấu=0.4531) · CI95 thô [-0.0294, +0.1176] | trong ngưỡng nhiễu |
| `recall@20` | 0.6912 | 0.7353 | +0.0441 | 34 | CI99.94% [+0.0000, +0.1471] · 0↔3 câu khác nhau (p dấu=0.25) · CI95 thô [+0.0000, +0.1029] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.5/34 = 0.01471) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@5` | 0.5049 | 0.5931 | +0.0882 | 34 | CI99.94% [+0.0000, +0.2059] · 0↔6 câu khác nhau (p dấu=0.03125) · CI95 thô [+0.0294, +0.1618] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.5/34 = 0.01471) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |

## `category = aggregation` — n = 26

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | bgem3 | bgem3-rrf-k1-c20 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.5385 | 0.4615 | -0.0769 | 26 | p=0.5 · 2↔0 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@10` | 0.8462 | 0.8846 | +0.0385 | 26 | p=1 · 0↔1 câu đổi chiều · **trần `p` = 1** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@20` | 0.8462 | 0.8846 | +0.0385 | 26 | p=1 · 0↔1 câu đổi chiều · **trần `p` = 1** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@5` | 0.7692 | 0.8462 | +0.0769 | 26 | p=0.5 · 0↔2 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `map@20` | 0.4264 | 0.4406 | +0.0142 | 26 | CI99.94% [-0.0830, +0.1201] · 8↔8 câu khác nhau (p dấu=1) · CI95 thô [-0.0418, +0.0731] · **đếm câu đi ngược Δ** (8 câu tốt hơn vs 8 câu xấu đi, mà Δ = +0.0142) | trong ngưỡng nhiễu |
| `mrr` | 0.6620 | 0.6362 | -0.0257 | 26 | CI99.94% [-0.1629, +0.1121] · 5↔3 câu khác nhau (p dấu=0.7266) · CI95 thô [-0.1020, +0.0500] | trong ngưỡng nhiễu |
| `ndcg@10` | 0.5235 | 0.5340 | +0.0105 | 26 | CI99.94% [-0.0778, +0.1144] · 8↔8 câu khác nhau (p dấu=1) · CI95 thô [-0.0423, +0.0677] · **đếm câu đi ngược Δ** (8 câu tốt hơn vs 8 câu xấu đi, mà Δ = +0.0105) | trong ngưỡng nhiễu |
| `precision@1` | 0.5385 | 0.4615 | -0.0769 | 26 | p=0.5 · 2↔0 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `precision@10` | 0.1346 | 0.1423 | +0.0077 | 26 | CI99.94% [-0.0154, +0.0346] · 1↔3 câu khác nhau (p dấu=0.625) · CI95 thô [-0.0077, +0.0231] | trong ngưỡng nhiễu |
| `precision@20` | 0.0731 | 0.0769 | +0.0038 | 26 | CI99.94% [-0.0096, +0.0173] · 1↔3 câu khác nhau (p dấu=0.625) · CI95 thô [-0.0038, +0.0115] | trong ngưỡng nhiễu |
| `precision@5` | 0.2308 | 0.2538 | +0.0231 | 26 | CI99.94% [-0.0462, +0.0923] · 2↔5 câu khác nhau (p dấu=0.4531) · CI95 thô [-0.0154, +0.0615] | trong ngưỡng nhiễu |
| `recall@1` | 0.2564 | 0.2179 | -0.0385 | 26 | CI99.94% [-0.1538, +0.0000] · 2↔0 câu khác nhau (p dấu=0.5) · CI95 thô [-0.0962, +0.0000] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.5/26 = 0.01923) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@10` | 0.5769 | 0.6026 | +0.0256 | 26 | CI99.94% [-0.0897, +0.1474] · 1↔3 câu khác nhau (p dấu=0.625) · CI95 thô [-0.0385, +0.0897] | trong ngưỡng nhiễu |
| `recall@20` | 0.6218 | 0.6538 | +0.0321 | 26 | CI99.94% [-0.0577, +0.1474] · 1↔3 câu khác nhau (p dấu=0.625) · CI95 thô [-0.0192, +0.0897] | trong ngưỡng nhiễu |
| `recall@5` | 0.5000 | 0.5449 | +0.0449 | 26 | CI99.94% [-0.0769, +0.1731] · 2↔5 câu khác nhau (p dấu=0.4531) · CI95 thô [-0.0256, +0.1154] | trong ngưỡng nhiễu |

## `category = table_lookup` — n = 4

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | bgem3 | bgem3-rrf-k1-c20 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.0000 | 0.0000 | +0.0000 | 4 | **0/4 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `hit_rate@10` | 0.2500 | 0.5000 | +0.2500 | 4 | p=1 · 0↔1 câu đổi chiều · **trần `p` = 1** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@20` | 0.2500 | 0.5000 | +0.2500 | 4 | p=1 · 0↔1 câu đổi chiều · **trần `p` = 1** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@5` | 0.2500 | 0.2500 | +0.0000 | 4 | p=1 · 1↔1 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `map@20` | 0.0500 | 0.1667 | +0.1167 | 4 | CI99.94% [-0.0333, +0.5000] · 1↔1 câu khác nhau (p dấu=1) · CI95 thô [-0.0250, +0.3750] · **đếm câu đi ngược Δ** (1 câu tốt hơn vs 1 câu xấu đi, mà Δ = +0.1167) | trong ngưỡng nhiễu |
| `mrr` | 0.0500 | 0.1667 | +0.1167 | 4 | CI99.94% [-0.0333, +0.5000] · 1↔1 câu khác nhau (p dấu=1) · CI95 thô [-0.0250, +0.3750] · **đếm câu đi ngược Δ** (1 câu tốt hơn vs 1 câu xấu đi, mà Δ = +0.1167) | trong ngưỡng nhiễu |
| `ndcg@10` | 0.0967 | 0.2468 | +0.1501 | 4 | CI99.94% [-0.0306, +0.6309] · 1↔1 câu khác nhau (p dấu=1) · CI95 thô [-0.0230, +0.4732] · **đếm câu đi ngược Δ** (1 câu tốt hơn vs 1 câu xấu đi, mà Δ = +0.1501) | trong ngưỡng nhiễu |
| `precision@1` | 0.0000 | 0.0000 | +0.0000 | 4 | **0/4 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `precision@10` | 0.0250 | 0.0500 | +0.0250 | 4 | CI99.94% [+0.0000, +0.1000] · 0↔1 câu khác nhau (p dấu=1) · CI95 thô [+0.0000, +0.0750] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.1/4 = 0.02500) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `precision@20` | 0.0125 | 0.0250 | +0.0125 | 4 | CI99.94% [+0.0000, +0.0500] · 0↔1 câu khác nhau (p dấu=1) · CI95 thô [+0.0000, +0.0375] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.05/4 = 0.01250) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `precision@5` | 0.0500 | 0.0500 | +0.0000 | 4 | CI99.94% [-0.2000, +0.2000] · 1↔1 câu khác nhau (p dấu=1) · CI95 thô [-0.1500, +0.1500] | trong ngưỡng nhiễu |
| `recall@1` | 0.0000 | 0.0000 | +0.0000 | 4 | **0/4 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `recall@10` | 0.2500 | 0.5000 | +0.2500 | 4 | CI99.94% [+0.0000, +1.0000] · 0↔1 câu khác nhau (p dấu=1) · CI95 thô [+0.0000, +0.7500] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/4 = 0.25000) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@20` | 0.2500 | 0.5000 | +0.2500 | 4 | CI99.94% [+0.0000, +1.0000] · 0↔1 câu khác nhau (p dấu=1) · CI95 thô [+0.0000, +0.7500] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/4 = 0.25000) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@5` | 0.2500 | 0.2500 | +0.0000 | 4 | CI99.94% [-1.0000, +1.0000] · 1↔1 câu khác nhau (p dấu=1) · CI95 thô [-0.7500, +0.7500] | trong ngưỡng nhiễu |

