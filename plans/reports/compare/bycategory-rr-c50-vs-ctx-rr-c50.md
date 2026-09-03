# So theo `category`: `bgem3-rr-c50` → `bgem3-ctx-rr-c50`

- **6 nhóm × 15 metric = 90 phép kiểm.**
- Ngưỡng đã hiệu chỉnh Bonferroni: **α = 0.0005556** (từ 0.05 chia cho 90).
- Không hiệu chỉnh thì ở α=0.05 kỳ vọng **4.5 kết quả "có ý nghĩa" thuần do ngẫu nhiên** — tức câu hỏi "nhóm nào cải thiện nhiều nhất" gần như bảo đảm tìm ra một cái, kể cả khi không có gì.
- `KHÔNG ĐỦ LỰC` = phép kiểm **không thể** đạt ý nghĩa dù kết quả thế nào (`p` nhỏ nhất McNemar cho `n` câu đổi chiều đã ≥ α). Nó **không** phải "không có khác biệt".
- `KHÔNG SO ĐƯỢC` = phân bố nhãn hai bên khác nhau nên mẫu số của metric khác nhau (`G2`/`TD-11`).
- `KHÔNG KẾT LUẬN` = một trong hai chuyện. (a) Khoảng tin cậy có **một biên gần 0 hơn một bước lưới `1/n`**: với metric rời rạc thưa, phân bố bootstrap nằm trên lưới bước `1/n`, nên việc khoảng chứa hay loại 0 phụ thuộc đúng một bước phân giải. Đã đo: đổi seed dịch biên **đúng một bước lưới**, và tăng 10.000 → 50.000 iterations **không đổi gì** — đừng chữa bằng cách tăng iterations. (b) **Đếm câu đi ngược `Δ`**: khoảng nói một hướng mà số câu tốt (b) **Biên không ổn định**: biên gần 0 nhất được đọc từ quá ít mẫu lại nên chính nó dao động qua 0. Đã đo ở `W2-08`: α đã hiệu chỉnh cho 39 phép kiểm để lại **6** mẫu lại trong đuôi của 10.000, và ở đó biên dưới **đổi dấu theo seed**; tăng lên 50.000 thì nó âm nhất quán, tức khoảng thật sự chứa 0.
- `TRÁI CHIỀU` = khoảng tin cậy đọc được **và** nó nói hai điều đối nhau: trung bình đi một hướng, còn **số câu** thì đi hướng ngược lại (thắng ít câu nhưng thắng đậm). Không phải lỗi, nhưng đừng đọc thành `hệ thống tốt hơn` — nhất là khi đổi lấy độ trễ.
- `TRÙNG KHỚP` = **0 câu khác nhau**. Đây là hàng chắc chắn nhất trong bảng, không phải hàng mơ hồ nhất — đừng đọc nó cùng nhóm với các cờ trên.

- ⚠️ **28/90 hàng `KHÔNG ĐỦ LỰC` và 20/90 hàng `KHÔNG KẾT LUẬN`.** Chia nhỏ tập đo rồi hiệu chỉnh cho 90 phép kiểm để lại rất ít lực; đọc bảng này để **loại** giả thuyết thì được, để **chọn người thắng** thì không.

## `category = factoid` — n = 68

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | bgem3-rr-c50 | bgem3-ctx-rr-c50 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.6176 | 0.6618 | +0.0441 | 68 | p=0.5811 · 5↔8 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@10` | 0.8824 | 0.9412 | +0.0588 | 68 | p=0.125 · 0↔4 câu đổi chiều · **trần `p` = 0.125** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@20` | 0.8824 | 0.9412 | +0.0588 | 68 | p=0.125 · 0↔4 câu đổi chiều · **trần `p` = 0.125** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@5` | 0.8529 | 0.9265 | +0.0735 | 68 | p=0.0625 · 0↔5 câu đổi chiều · **trần `p` = 0.0625** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `map@20` | 0.7182 | 0.7728 | +0.0546 | 68 | CI99.94% [-0.0514, +0.1722] · 7↔14 câu khác nhau (p dấu=0.1892) · CI95 thô [-0.0061, +0.1191] | trong ngưỡng nhiễu |
| `mrr` | 0.7206 | 0.7716 | +0.0509 | 68 | CI99.94% [-0.0612, +0.1697] · 7↔14 câu khác nhau (p dấu=0.1892) · CI95 thô [-0.0123, +0.1169] | trong ngưỡng nhiễu |
| `ndcg@10` | 0.7596 | 0.8150 | +0.0553 | 68 | CI99.94% [-0.0339, +0.1602] · 7↔14 câu khác nhau (p dấu=0.1892) · CI95 thô [+0.0032, +0.1126] | trong ngưỡng nhiễu |
| `precision@1` | 0.6176 | 0.6618 | +0.0441 | 68 | p=0.5811 · 5↔8 câu đổi chiều | trong ngưỡng nhiễu |
| `precision@10` | 0.0897 | 0.0956 | +0.0059 | 68 | CI99.94% [+0.0000, +0.0176] · 0↔4 câu khác nhau (p dấu=0.125) · CI95 thô [+0.0015, +0.0118] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.1/68 = 0.00147) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `precision@20` | 0.0449 | 0.0478 | +0.0029 | 68 | CI99.94% [+0.0000, +0.0088] · 0↔4 câu khác nhau (p dấu=0.125) · CI95 thô [+0.0007, +0.0059] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.05/68 = 0.00074) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `precision@5` | 0.1735 | 0.1882 | +0.0147 | 68 | CI99.94% [+0.0000, +0.0412] · 0↔5 câu khác nhau (p dấu=0.0625) · CI95 thô [+0.0029, +0.0294] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.2/68 = 0.00294) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@1` | 0.6103 | 0.6618 | +0.0515 | 68 | CI99.94% [-0.1176, +0.2353] · 5↔8 câu khác nhau (p dấu=0.5811) · CI95 thô [-0.0441, +0.1544] | trong ngưỡng nhiễu |
| `recall@10` | 0.8824 | 0.9412 | +0.0588 | 68 | CI99.94% [+0.0000, +0.1765] · 0↔4 câu khác nhau (p dấu=0.125) · CI95 thô [+0.0147, +0.1176] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/68 = 0.01471) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@20` | 0.8824 | 0.9412 | +0.0588 | 68 | CI99.94% [+0.0000, +0.1765] · 0↔4 câu khác nhau (p dấu=0.125) · CI95 thô [+0.0147, +0.1176] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/68 = 0.01471) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@5` | 0.8529 | 0.9265 | +0.0735 | 68 | CI99.94% [+0.0000, +0.2059] · 0↔5 câu khác nhau (p dấu=0.0625) · CI95 thô [+0.0147, +0.1471] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/68 = 0.01471) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |

## `category = cross_lingual` — n = 43

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | bgem3-rr-c50 | bgem3-ctx-rr-c50 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.4419 | 0.4419 | +0.0000 | 43 | p=1 · 2↔2 câu đổi chiều · **trần `p` = 0.125** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@10` | 0.5814 | 0.5581 | -0.0233 | 43 | p=1 · 3↔2 câu đổi chiều · **trần `p` = 0.0625** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@20` | 0.6047 | 0.5814 | -0.0233 | 43 | p=1 · 3↔2 câu đổi chiều · **trần `p` = 0.0625** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@5` | 0.5581 | 0.5581 | +0.0000 | 43 | p=1 · 2↔2 câu đổi chiều · **trần `p` = 0.125** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `map@20` | 0.4978 | 0.4876 | -0.0102 | 43 | CI99.94% [-0.1494, +0.1034] · 6↔6 câu khác nhau (p dấu=1) · CI95 thô [-0.0844, +0.0569] · **đếm câu đi ngược Δ** (6 câu tốt hơn vs 6 câu xấu đi, mà Δ = -0.0102) | trong ngưỡng nhiễu |
| `mrr` | 0.5009 | 0.4866 | -0.0143 | 43 | CI99.94% [-0.1519, +0.0988] · 6↔5 câu khác nhau (p dấu=1) · CI95 thô [-0.0879, +0.0524] | trong ngưỡng nhiễu |
| `ndcg@10` | 0.5191 | 0.5041 | -0.0150 | 43 | CI99.94% [-0.1566, +0.1027] · 6↔5 câu khác nhau (p dấu=1) · CI95 thô [-0.0911, +0.0540] | trong ngưỡng nhiễu |
| `precision@1` | 0.4419 | 0.4419 | +0.0000 | 43 | p=1 · 2↔2 câu đổi chiều · **trần `p` = 0.125** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `precision@10` | 0.0674 | 0.0651 | -0.0023 | 43 | CI99.94% [-0.0209, +0.0163] · 3↔2 câu khác nhau (p dấu=1) · CI95 thô [-0.0116, +0.0070] | trong ngưỡng nhiễu |
| `precision@20` | 0.0349 | 0.0337 | -0.0012 | 43 | CI99.94% [-0.0105, +0.0081] · 3↔2 câu khác nhau (p dấu=1) · CI95 thô [-0.0058, +0.0035] | trong ngưỡng nhiễu |
| `precision@5` | 0.1302 | 0.1256 | -0.0047 | 43 | CI99.94% [-0.0419, +0.0326] · 3↔2 câu khác nhau (p dấu=1) · CI95 thô [-0.0233, +0.0140] | trong ngưỡng nhiễu |
| `recall@1` | 0.4070 | 0.4070 | +0.0000 | 43 | CI99.94% [-0.1628, +0.1628] · 2↔2 câu khác nhau (p dấu=1) · CI95 thô [-0.0930, +0.0930] | trong ngưỡng nhiễu |
| `recall@10` | 0.5814 | 0.5581 | -0.0233 | 43 | CI99.94% [-0.2093, +0.1628] · 3↔2 câu khác nhau (p dấu=1) · CI95 thô [-0.1163, +0.0698] | trong ngưỡng nhiễu |
| `recall@20` | 0.6047 | 0.5814 | -0.0233 | 43 | CI99.94% [-0.2093, +0.1628] · 3↔2 câu khác nhau (p dấu=1) · CI95 thô [-0.1163, +0.0698] | trong ngưỡng nhiễu |
| `recall@5` | 0.5581 | 0.5465 | -0.0116 | 43 | CI99.94% [-0.1860, +0.1628] · 3↔2 câu khác nhau (p dấu=1) · CI95 thô [-0.1047, +0.0814] | trong ngưỡng nhiễu |

## `category = adversarial` — n = 34

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | bgem3-rr-c50 | bgem3-ctx-rr-c50 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.4412 | 0.5000 | +0.0588 | 34 | p=0.7266 · 3↔5 câu đổi chiều · **trần `p` = 0.007812** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@10` | 0.7059 | 0.7353 | +0.0294 | 34 | p=1 · 2↔3 câu đổi chiều · **trần `p` = 0.0625** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@20` | 0.7059 | 0.7353 | +0.0294 | 34 | p=1 · 2↔3 câu đổi chiều · **trần `p` = 0.0625** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@5` | 0.6765 | 0.7353 | +0.0588 | 34 | p=0.625 · 1↔3 câu đổi chiều · **trần `p` = 0.125** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `map@20` | 0.5272 | 0.5833 | +0.0561 | 34 | CI99.94% [-0.1583, +0.2681] · 6↔9 câu khác nhau (p dấu=0.6072) · CI95 thô [-0.0627, +0.1750] | trong ngưỡng nhiễu |
| `mrr` | 0.5446 | 0.5907 | +0.0461 | 34 | CI99.94% [-0.1686, +0.2588] · 6↔7 câu khác nhau (p dấu=1) · CI95 thô [-0.0716, +0.1657] | trong ngưỡng nhiễu |
| `ndcg@10` | 0.5732 | 0.6235 | +0.0503 | 34 | CI99.94% [-0.1523, +0.2462] · 6↔9 câu khác nhau (p dấu=0.6072) · CI95 thô [-0.0594, +0.1596] | trong ngưỡng nhiễu |
| `precision@1` | 0.4412 | 0.5000 | +0.0588 | 34 | p=0.7266 · 3↔5 câu đổi chiều · **trần `p` = 0.007812** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `precision@10` | 0.0765 | 0.0794 | +0.0029 | 34 | CI99.94% [-0.0206, +0.0265] · 2↔3 câu khác nhau (p dấu=1) · CI95 thô [-0.0088, +0.0147] | trong ngưỡng nhiễu |
| `precision@20` | 0.0382 | 0.0397 | +0.0015 | 34 | CI99.94% [-0.0103, +0.0132] · 2↔3 câu khác nhau (p dấu=1) · CI95 thô [-0.0044, +0.0074] | trong ngưỡng nhiễu |
| `precision@5` | 0.1412 | 0.1588 | +0.0176 | 34 | CI99.94% [-0.0235, +0.0647] · 1↔4 câu khác nhau (p dấu=0.375) · CI95 thô [-0.0059, +0.0412] | trong ngưỡng nhiễu |
| `recall@1` | 0.4118 | 0.4706 | +0.0588 | 34 | CI99.94% [-0.2353, +0.3529] · 3↔5 câu khác nhau (p dấu=0.7266) · CI95 thô [-0.0882, +0.2353] | trong ngưỡng nhiễu |
| `recall@10` | 0.6912 | 0.7353 | +0.0441 | 34 | CI99.94% [-0.1618, +0.2647] · 2↔3 câu khác nhau (p dấu=1) · CI95 thô [-0.0735, +0.1618] | trong ngưỡng nhiễu |
| `recall@20` | 0.6912 | 0.7353 | +0.0441 | 34 | CI99.94% [-0.1618, +0.2647] · 2↔3 câu khác nhau (p dấu=1) · CI95 thô [-0.0735, +0.1618] | trong ngưỡng nhiễu |
| `recall@5` | 0.6618 | 0.7353 | +0.0735 | 34 | CI99.94% [-0.1176, +0.2941] · 1↔4 câu khác nhau (p dấu=0.375) · CI95 thô [-0.0441, +0.1912] | trong ngưỡng nhiễu |

## `category = multi_hop` — n = 34

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | bgem3-rr-c50 | bgem3-ctx-rr-c50 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.6471 | 0.7647 | +0.1176 | 34 | p=0.3438 · 3↔7 câu đổi chiều · **trần `p` = 0.001953** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@10` | 0.8235 | 0.9412 | +0.1176 | 34 | p=0.125 · 0↔4 câu đổi chiều · **trần `p` = 0.125** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@20` | 0.8235 | 0.9706 | +0.1471 | 34 | p=0.0625 · 0↔5 câu đổi chiều · **trần `p` = 0.0625** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@5` | 0.8235 | 0.9412 | +0.1176 | 34 | p=0.125 · 0↔4 câu đổi chiều · **trần `p` = 0.125** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `map@20` | 0.6283 | 0.7042 | +0.0759 | 34 | CI99.94% [-0.0525, +0.2436] · 5↔10 câu khác nhau (p dấu=0.3018) · CI95 thô [-0.0028, +0.1649] | trong ngưỡng nhiễu |
| `mrr` | 0.7157 | 0.8403 | +0.1246 | 34 | CI99.94% [-0.0861, +0.3789] · 4↔8 câu khác nhau (p dấu=0.3877) · CI95 thô [-0.0007, +0.2623] | trong ngưỡng nhiễu |
| `ndcg@10` | 0.6850 | 0.7647 | +0.0797 | 34 | CI99.94% [-0.0617, +0.2679] · 6↔8 câu khác nhau (p dấu=0.7905) · CI95 thô [-0.0072, +0.1795] | trong ngưỡng nhiễu |
| `precision@1` | 0.6471 | 0.7647 | +0.1176 | 34 | p=0.3438 · 3↔7 câu đổi chiều · **trần `p` = 0.001953** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `precision@10` | 0.1529 | 0.1647 | +0.0118 | 34 | CI99.94% [-0.0176, +0.0529] · 2↔4 câu khác nhau (p dấu=0.6875) · CI95 thô [-0.0059, +0.0324] | trong ngưỡng nhiễu |
| `precision@20` | 0.0765 | 0.0868 | +0.0103 | 34 | CI99.94% [+0.0000, +0.0294] · 0↔5 câu khác nhau (p dấu=0.0625) · CI95 thô [+0.0029, +0.0206] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.05/34 = 0.00147) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `precision@5` | 0.2824 | 0.3294 | +0.0471 | 34 | CI99.94% [-0.0118, +0.1294] · 1↔7 câu khác nhau (p dấu=0.07031) · CI95 thô [+0.0118, +0.0882] | trong ngưỡng nhiễu |
| `recall@1` | 0.3186 | 0.3775 | +0.0588 | 34 | CI99.94% [-0.1029, +0.2206] · 3↔7 câu khác nhau (p dấu=0.3438) · CI95 thô [-0.0294, +0.1471] | trong ngưỡng nhiễu |
| `recall@10` | 0.7549 | 0.8137 | +0.0588 | 34 | CI99.94% [-0.0882, +0.2647] · 2↔4 câu khác nhau (p dấu=0.6875) · CI95 thô [-0.0294, +0.1618] | trong ngưỡng nhiễu |
| `recall@20` | 0.7549 | 0.8578 | +0.1029 | 34 | CI99.94% [+0.0000, +0.2941] · 0↔5 câu khác nhau (p dấu=0.0625) · CI95 thô [+0.0294, +0.2059] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.5/34 = 0.01471) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@5` | 0.6961 | 0.8137 | +0.1176 | 34 | CI99.94% [-0.0294, +0.3235] · 1↔7 câu khác nhau (p dấu=0.07031) · CI95 thô [+0.0294, +0.2206] | trong ngưỡng nhiễu |

## `category = aggregation` — n = 26

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | bgem3-rr-c50 | bgem3-ctx-rr-c50 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.6538 | 0.6923 | +0.0385 | 26 | p=1 · 2↔3 câu đổi chiều · **trần `p` = 0.0625** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@10` | 0.8462 | 0.8846 | +0.0385 | 26 | p=1 · 0↔1 câu đổi chiều · **trần `p` = 1** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@20` | 0.8462 | 0.9231 | +0.0769 | 26 | p=0.5 · 0↔2 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@5` | 0.8462 | 0.8846 | +0.0385 | 26 | p=1 · 0↔1 câu đổi chiều · **trần `p` = 1** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `map@20` | 0.5746 | 0.5961 | +0.0215 | 26 | CI99.94% [-0.1348, +0.1982] · 8↔10 câu khác nhau (p dấu=0.8145) · CI95 thô [-0.0707, +0.1197] | trong ngưỡng nhiễu |
| `mrr` | 0.7385 | 0.7853 | +0.0468 | 26 | CI99.94% [-0.1250, +0.2417] · 3↔5 câu khác nhau (p dấu=0.7266) · CI95 thô [-0.0526, +0.1545] | trong ngưỡng nhiễu |
| `ndcg@10` | 0.6423 | 0.6797 | +0.0374 | 26 | CI99.94% [-0.1028, +0.2174] · 9↔8 câu khác nhau (p dấu=1) · CI95 thô [-0.0488, +0.1348] · **đếm câu đi ngược Δ** (8 câu tốt hơn vs 9 câu xấu đi, mà Δ = +0.0374) | trong ngưỡng nhiễu |
| `precision@1` | 0.6538 | 0.6923 | +0.0385 | 26 | p=1 · 2↔3 câu đổi chiều · **trần `p` = 0.0625** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `precision@10` | 0.1577 | 0.1731 | +0.0154 | 26 | CI99.94% [-0.0269, +0.0615] · 3↔6 câu khác nhau (p dấu=0.5078) · CI95 thô [-0.0077, +0.0423] | trong ngưỡng nhiễu |
| `precision@20` | 0.0808 | 0.0962 | +0.0154 | 26 | CI99.94% [-0.0038, +0.0404] · 1↔7 câu khác nhau (p dấu=0.07031) · CI95 thô [+0.0038, +0.0288] | trong ngưỡng nhiễu |
| `precision@5` | 0.3000 | 0.3385 | +0.0385 | 26 | CI99.94% [-0.0462, +0.1462] · 2↔5 câu khác nhau (p dấu=0.4531) · CI95 thô [-0.0077, +0.0923] | trong ngưỡng nhiễu |
| `recall@1` | 0.2949 | 0.3141 | +0.0192 | 26 | CI99.94% [-0.1090, +0.1603] · 2↔3 câu khác nhau (p dấu=1) · CI95 thô [-0.0513, +0.0962] | trong ngưỡng nhiễu |
| `recall@10` | 0.6731 | 0.7500 | +0.0769 | 26 | CI99.94% [-0.1090, +0.2949] · 3↔6 câu khác nhau (p dấu=0.5078) · CI95 thô [-0.0321, +0.1987] | trong ngưỡng nhiễu |
| `recall@20` | 0.6923 | 0.8205 | +0.1282 | 26 | CI99.94% [-0.0256, +0.3397] · 1↔7 câu khác nhau (p dấu=0.07031) · CI95 thô [+0.0321, +0.2436] | trong ngưỡng nhiễu |
| `recall@5` | 0.6410 | 0.7308 | +0.0897 | 26 | CI99.94% [-0.0705, +0.3077] · 2↔5 câu khác nhau (p dấu=0.4531) · CI95 thô [-0.0128, +0.2051] | trong ngưỡng nhiễu |

## `category = table_lookup` — n = 4

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | bgem3-rr-c50 | bgem3-ctx-rr-c50 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.5000 | 0.5000 | +0.0000 | 4 | **0/4 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `hit_rate@10` | 0.5000 | 0.5000 | +0.0000 | 4 | **0/4 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `hit_rate@20` | 0.5000 | 0.5000 | +0.0000 | 4 | **0/4 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `hit_rate@5` | 0.5000 | 0.5000 | +0.0000 | 4 | **0/4 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `map@20` | 0.5000 | 0.5000 | +0.0000 | 4 | **0/4 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `mrr` | 0.5000 | 0.5000 | +0.0000 | 4 | **0/4 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `ndcg@10` | 0.5000 | 0.5000 | +0.0000 | 4 | **0/4 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `precision@1` | 0.5000 | 0.5000 | +0.0000 | 4 | **0/4 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `precision@10` | 0.0500 | 0.0500 | +0.0000 | 4 | **0/4 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `precision@20` | 0.0250 | 0.0250 | +0.0000 | 4 | **0/4 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `precision@5` | 0.1000 | 0.1000 | +0.0000 | 4 | **0/4 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `recall@1` | 0.5000 | 0.5000 | +0.0000 | 4 | **0/4 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `recall@10` | 0.5000 | 0.5000 | +0.0000 | 4 | **0/4 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `recall@20` | 0.5000 | 0.5000 | +0.0000 | 4 | **0/4 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `recall@5` | 0.5000 | 0.5000 | +0.0000 | 4 | **0/4 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |

