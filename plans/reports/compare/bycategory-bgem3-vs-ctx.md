# So theo `category`: `bgem3` → `bgem3-ctx`

- **6 nhóm × 15 metric = 90 phép kiểm.**
- Ngưỡng đã hiệu chỉnh Bonferroni: **α = 0.0005556** (từ 0.05 chia cho 90).
- Không hiệu chỉnh thì ở α=0.05 kỳ vọng **4.5 kết quả "có ý nghĩa" thuần do ngẫu nhiên** — tức câu hỏi "nhóm nào cải thiện nhiều nhất" gần như bảo đảm tìm ra một cái, kể cả khi không có gì.
- `KHÔNG ĐỦ LỰC` = phép kiểm **không thể** đạt ý nghĩa dù kết quả thế nào (`p` nhỏ nhất McNemar cho `n` câu đổi chiều đã ≥ α). Nó **không** phải "không có khác biệt".
- `KHÔNG SO ĐƯỢC` = phân bố nhãn hai bên khác nhau nên mẫu số của metric khác nhau (`G2`/`TD-11`).
- `KHÔNG KẾT LUẬN` = một trong hai chuyện. (a) Khoảng tin cậy có **một biên gần 0 hơn một bước lưới `1/n`**: với metric rời rạc thưa, phân bố bootstrap nằm trên lưới bước `1/n`, nên việc khoảng chứa hay loại 0 phụ thuộc đúng một bước phân giải. Đã đo: đổi seed dịch biên **đúng một bước lưới**, và tăng 10.000 → 50.000 iterations **không đổi gì** — đừng chữa bằng cách tăng iterations. (b) **Đếm câu đi ngược `Δ`**: khoảng nói một hướng mà số câu tốt (b) **Biên không ổn định**: biên gần 0 nhất được đọc từ quá ít mẫu lại nên chính nó dao động qua 0. Đã đo ở `W2-08`: α đã hiệu chỉnh cho 39 phép kiểm để lại **6** mẫu lại trong đuôi của 10.000, và ở đó biên dưới **đổi dấu theo seed**; tăng lên 50.000 thì nó âm nhất quán, tức khoảng thật sự chứa 0.
- `TRÁI CHIỀU` = khoảng tin cậy đọc được **và** nó nói hai điều đối nhau: trung bình đi một hướng, còn **số câu** thì đi hướng ngược lại (thắng ít câu nhưng thắng đậm). Không phải lỗi, nhưng đừng đọc thành `hệ thống tốt hơn` — nhất là khi đổi lấy độ trễ.
- `TRÙNG KHỚP` = **0 câu khác nhau**. Đây là hàng chắc chắn nhất trong bảng, không phải hàng mơ hồ nhất — đừng đọc nó cùng nhóm với các cờ trên.

- ⚠️ **25/90 hàng `KHÔNG ĐỦ LỰC` và 18/90 hàng `KHÔNG KẾT LUẬN`.** Chia nhỏ tập đo rồi hiệu chỉnh cho 90 phép kiểm để lại rất ít lực; đọc bảng này để **loại** giả thuyết thì được, để **chọn người thắng** thì không.

## `category = factoid` — n = 68

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | bgem3 | bgem3-ctx | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.3529 | 0.4265 | +0.0735 | 68 | p=0.3018 · 5↔10 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@10` | 0.7059 | 0.7647 | +0.0588 | 68 | p=0.3877 · 4↔8 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@20` | 0.7500 | 0.8676 | +0.1176 | 68 | p=0.03857 · 2↔10 câu đổi chiều | trong ngưỡng nhiễu |
| `hit_rate@5` | 0.6029 | 0.6912 | +0.0882 | 68 | p=0.2101 · 5↔11 câu đổi chiều | trong ngưỡng nhiễu |
| `map@20` | 0.4681 | 0.5313 | +0.0632 | 68 | CI99.94% [-0.0799, +0.2228] · 14↔26 câu khác nhau (p dấu=0.08069) · CI95 thô [-0.0206, +0.1503] | trong ngưỡng nhiễu |
| `mrr` | 0.4718 | 0.5338 | +0.0619 | 68 | CI99.94% [-0.0812, +0.2217] · 14↔25 câu khác nhau (p dấu=0.1081) · CI95 thô [-0.0218, +0.1490] | trong ngưỡng nhiễu |
| `ndcg@10` | 0.5241 | 0.5818 | +0.0578 | 68 | CI99.94% [-0.0775, +0.2081] · 13↔22 câu khác nhau (p dấu=0.1755) · CI95 thô [-0.0208, +0.1399] | trong ngưỡng nhiễu |
| `precision@1` | 0.3529 | 0.4265 | +0.0735 | 68 | p=0.3018 · 5↔10 câu đổi chiều | trong ngưỡng nhiễu |
| `precision@10` | 0.0721 | 0.0779 | +0.0059 | 68 | CI99.94% [-0.0103, +0.0250] · 4↔8 câu khác nhau (p dấu=0.3877) · CI95 thô [-0.0044, +0.0162] | trong ngưỡng nhiễu |
| `precision@20` | 0.0382 | 0.0441 | +0.0059 | 68 | CI99.94% [-0.0022, +0.0147] · 2↔10 câu khác nhau (p dấu=0.03857) · CI95 thô [+0.0015, +0.0110] | trong ngưỡng nhiễu |
| `precision@5` | 0.1235 | 0.1412 | +0.0176 | 68 | CI99.94% [-0.0206, +0.0588] · 5↔11 câu khác nhau (p dấu=0.2101) · CI95 thô [-0.0059, +0.0412] | trong ngưỡng nhiễu |
| `recall@1` | 0.3456 | 0.4191 | +0.0735 | 68 | CI99.94% [-0.1176, +0.2794] · 5↔10 câu khác nhau (p dấu=0.3018) · CI95 thô [-0.0294, +0.1912] | trong ngưỡng nhiễu |
| `recall@10` | 0.7059 | 0.7647 | +0.0588 | 68 | CI99.94% [-0.1029, +0.2500] · 4↔8 câu khác nhau (p dấu=0.3877) · CI95 thô [-0.0441, +0.1618] | trong ngưỡng nhiễu |
| `recall@20` | 0.7500 | 0.8676 | +0.1176 | 68 | CI99.94% [-0.0441, +0.2941] · 2↔10 câu khác nhau (p dấu=0.03857) · CI95 thô [+0.0294, +0.2206] | trong ngưỡng nhiễu |
| `recall@5` | 0.6029 | 0.6912 | +0.0882 | 68 | CI99.94% [-0.1029, +0.2941] · 5↔11 câu khác nhau (p dấu=0.2101) · CI95 thô [-0.0294, +0.2059] | trong ngưỡng nhiễu |

## `category = cross_lingual` — n = 43

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | bgem3 | bgem3-ctx | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.0930 | 0.2093 | +0.1163 | 43 | p=0.0625 · 0↔5 câu đổi chiều · **trần `p` = 0.0625** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@10` | 0.4419 | 0.4651 | +0.0233 | 43 | p=1 · 3↔4 câu đổi chiều · **trần `p` = 0.01562** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@20` | 0.5116 | 0.5116 | +0.0000 | 43 | p=1 · 3↔3 câu đổi chiều · **trần `p` = 0.03125** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@5` | 0.3256 | 0.3953 | +0.0698 | 43 | p=0.5078 · 3↔6 câu đổi chiều · **trần `p` = 0.003906** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `map@20` | 0.1963 | 0.2740 | +0.0778 | 43 | CI99.94% [-0.0287, +0.2234] · 6↔15 câu khác nhau (p dấu=0.07835) · CI95 thô [+0.0116, +0.1553] | trong ngưỡng nhiễu |
| `mrr` | 0.2028 | 0.2869 | +0.0841 | 43 | CI99.94% [-0.0269, +0.2328] · 6↔14 câu khác nhau (p dấu=0.1153) · CI95 thô [+0.0149, +0.1645] | trong ngưỡng nhiễu |
| `ndcg@10` | 0.2538 | 0.3172 | +0.0635 | 43 | CI99.94% [-0.0577, +0.2132] · 6↔13 câu khác nhau (p dấu=0.1671) · CI95 thô [-0.0094, +0.1439] | trong ngưỡng nhiễu |
| `precision@1` | 0.0930 | 0.2093 | +0.1163 | 43 | p=0.0625 · 0↔5 câu đổi chiều · **trần `p` = 0.0625** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `precision@10` | 0.0535 | 0.0535 | +0.0000 | 43 | CI99.94% [-0.0233, +0.0233] · 4↔4 câu khác nhau (p dấu=1) · CI95 thô [-0.0140, +0.0140] | trong ngưỡng nhiễu |
| `precision@20` | 0.0302 | 0.0302 | +0.0000 | 43 | CI99.94% [-0.0105, +0.0105] · 3↔3 câu khác nhau (p dấu=1) · CI95 thô [-0.0058, +0.0058] | trong ngưỡng nhiễu |
| `precision@5` | 0.0744 | 0.0930 | +0.0186 | 43 | CI99.94% [-0.0465, +0.0744] · 3↔8 câu khác nhau (p dấu=0.2266) · CI95 thô [-0.0140, +0.0512] | trong ngưỡng nhiễu |
| `recall@1` | 0.0814 | 0.1744 | +0.0930 | 43 | CI99.94% [+0.0000, +0.2558] · 0↔5 câu khác nhau (p dấu=0.0625) · CI95 thô [+0.0233, +0.1860] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.5/43 = 0.01163) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@10` | 0.4419 | 0.4535 | +0.0116 | 43 | CI99.94% [-0.2093, +0.2326] · 4↔4 câu khác nhau (p dấu=1) · CI95 thô [-0.1163, +0.1395] · **đếm câu đi ngược Δ** (4 câu tốt hơn vs 4 câu xấu đi, mà Δ = +0.0116) | trong ngưỡng nhiễu |
| `recall@20` | 0.5116 | 0.5116 | +0.0000 | 43 | CI99.94% [-0.2093, +0.2093] · 3↔3 câu khác nhau (p dấu=1) · CI95 thô [-0.1163, +0.1163] | trong ngưỡng nhiễu |
| `recall@5` | 0.3023 | 0.3953 | +0.0930 | 43 | CI99.94% [-0.1512, +0.3372] · 3↔8 câu khác nhau (p dấu=0.2266) · CI95 thô [-0.0465, +0.2326] | trong ngưỡng nhiễu |

## `category = adversarial` — n = 34

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | bgem3 | bgem3-ctx | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.2941 | 0.3529 | +0.0588 | 34 | p=0.625 · 1↔3 câu đổi chiều · **trần `p` = 0.125** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@10` | 0.5000 | 0.5588 | +0.0588 | 34 | p=0.5 · 0↔2 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@20` | 0.5588 | 0.6176 | +0.0588 | 34 | p=0.625 · 1↔3 câu đổi chiều · **trần `p` = 0.125** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@5` | 0.4412 | 0.5000 | +0.0588 | 34 | p=0.625 · 1↔3 câu đổi chiều · **trần `p` = 0.125** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `map@20` | 0.3665 | 0.4172 | +0.0506 | 34 | CI99.94% [-0.0971, +0.2163] · 5↔7 câu khác nhau (p dấu=0.7744) · CI95 thô [-0.0332, +0.1394] | trong ngưỡng nhiễu |
| `mrr` | 0.3703 | 0.4172 | +0.0468 | 34 | CI99.94% [-0.0988, +0.2102] · 5↔7 câu khác nhau (p dấu=0.7744) · CI95 thô [-0.0357, +0.1341] | trong ngưỡng nhiễu |
| `ndcg@10` | 0.3974 | 0.4486 | +0.0512 | 34 | CI99.94% [-0.0720, +0.1932] · 3↔6 câu khác nhau (p dấu=0.5078) · CI95 thô [-0.0196, +0.1279] | trong ngưỡng nhiễu |
| `precision@1` | 0.2941 | 0.3529 | +0.0588 | 34 | p=0.625 · 1↔3 câu đổi chiều · **trần `p` = 0.125** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `precision@10` | 0.0559 | 0.0618 | +0.0059 | 34 | CI99.94% [+0.0000, +0.0235] · 0↔2 câu khác nhau (p dấu=0.5) · CI95 thô [+0.0000, +0.0147] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.1/34 = 0.00294) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `precision@20` | 0.0309 | 0.0338 | +0.0029 | 34 | CI99.94% [-0.0059, +0.0147] · 1↔3 câu khác nhau (p dấu=0.625) · CI95 thô [-0.0029, +0.0088] | trong ngưỡng nhiễu |
| `precision@5` | 0.0941 | 0.1118 | +0.0176 | 34 | CI99.94% [-0.0235, +0.0647] · 1↔4 câu khác nhau (p dấu=0.375) · CI95 thô [-0.0059, +0.0412] | trong ngưỡng nhiễu |
| `recall@1` | 0.2794 | 0.3235 | +0.0441 | 34 | CI99.94% [-0.1471, +0.2500] · 1↔3 câu khác nhau (p dấu=0.625) · CI95 thô [-0.0588, +0.1471] | trong ngưỡng nhiễu |
| `recall@10` | 0.5000 | 0.5588 | +0.0588 | 34 | CI99.94% [+0.0000, +0.2353] · 0↔2 câu khác nhau (p dấu=0.5) · CI95 thô [+0.0000, +0.1471] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/34 = 0.02941) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@20` | 0.5441 | 0.6176 | +0.0735 | 34 | CI99.94% [-0.0588, +0.2941] · 1↔3 câu khác nhau (p dấu=0.625) · CI95 thô [-0.0147, +0.1765] | trong ngưỡng nhiễu |
| `recall@5` | 0.4265 | 0.5000 | +0.0735 | 34 | CI99.94% [-0.1324, +0.2941] · 1↔4 câu khác nhau (p dấu=0.375) · CI95 thô [-0.0294, +0.1912] | trong ngưỡng nhiễu |

## `category = multi_hop` — n = 34

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | bgem3 | bgem3-ctx | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.5588 | 0.5294 | -0.0294 | 34 | p=1 · 4↔3 câu đổi chiều · **trần `p` = 0.01562** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@10` | 0.7059 | 0.8235 | +0.1176 | 34 | p=0.125 · 0↔4 câu đổi chiều · **trần `p` = 0.125** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@20` | 0.7647 | 0.8824 | +0.1176 | 34 | p=0.125 · 0↔4 câu đổi chiều · **trần `p` = 0.125** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@5` | 0.6765 | 0.7941 | +0.1176 | 34 | p=0.125 · 0↔4 câu đổi chiều · **trần `p` = 0.125** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `map@20` | 0.4855 | 0.5170 | +0.0315 | 34 | CI99.94% [-0.1070, +0.1696] · 9↔16 câu khác nhau (p dấu=0.2295) · CI95 thô [-0.0454, +0.1088] | trong ngưỡng nhiễu |
| `mrr` | 0.6184 | 0.6448 | +0.0265 | 34 | CI99.94% [-0.1738, +0.2323] · 4↔9 câu khác nhau (p dấu=0.2668) · CI95 thô [-0.0863, +0.1414] | trong ngưỡng nhiễu |
| `ndcg@10` | 0.5522 | 0.5883 | +0.0361 | 34 | CI99.94% [-0.1031, +0.1758] · 9↔13 câu khác nhau (p dấu=0.5235) · CI95 thô [-0.0415, +0.1152] | trong ngưỡng nhiễu |
| `precision@1` | 0.5588 | 0.5294 | -0.0294 | 34 | p=1 · 4↔3 câu đổi chiều · **trần `p` = 0.01562** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `precision@10` | 0.1294 | 0.1382 | +0.0088 | 34 | CI99.94% [-0.0206, +0.0382] · 3↔6 câu khác nhau (p dấu=0.5078) · CI95 thô [-0.0088, +0.0265] | trong ngưỡng nhiễu |
| `precision@20` | 0.0706 | 0.0765 | +0.0059 | 34 | CI99.94% [-0.0059, +0.0191] · 1↔5 câu khác nhau (p dấu=0.2188) · CI95 thô [+0.0000, +0.0132] | trong ngưỡng nhiễu |
| `precision@5` | 0.2059 | 0.2529 | +0.0471 | 34 | CI99.94% [+0.0000, +0.1118] · 0↔7 câu khác nhau (p dấu=0.01562) · CI95 thô [+0.0176, +0.0824] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.2/34 = 0.00588) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@1` | 0.2745 | 0.2598 | -0.0147 | 34 | CI99.94% [-0.1471, +0.1176] · 4↔3 câu khác nhau (p dấu=1) · CI95 thô [-0.0882, +0.0588] | trong ngưỡng nhiễu |
| `recall@10` | 0.6324 | 0.6765 | +0.0441 | 34 | CI99.94% [-0.1029, +0.1912] · 3↔6 câu khác nhau (p dấu=0.5078) · CI95 thô [-0.0441, +0.1324] | trong ngưỡng nhiễu |
| `recall@20` | 0.6912 | 0.7500 | +0.0588 | 34 | CI99.94% [-0.0588, +0.1912] · 1↔5 câu khác nhau (p dấu=0.2188) · CI95 thô [+0.0000, +0.1324] | trong ngưỡng nhiễu |
| `recall@5` | 0.5049 | 0.6225 | +0.1176 | 34 | CI99.94% [+0.0000, +0.2794] · 0↔7 câu khác nhau (p dấu=0.01562) · CI95 thô [+0.0441, +0.2059] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.5/34 = 0.01471) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |

## `category = aggregation` — n = 26

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | bgem3 | bgem3-ctx | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.5385 | 0.5385 | +0.0000 | 26 | p=1 · 1↔1 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@10` | 0.8462 | 0.8462 | +0.0000 | 26 | p=1 · 1↔1 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@20` | 0.8462 | 0.8462 | +0.0000 | 26 | p=1 · 1↔1 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@5` | 0.7692 | 0.7692 | +0.0000 | 26 | **0/26 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `map@20` | 0.4264 | 0.5009 | +0.0745 | 26 | CI99.94% [-0.0174, +0.1838] · 7↔12 câu khác nhau (p dấu=0.3593) · CI95 thô [+0.0189, +0.1344] | trong ngưỡng nhiễu |
| `mrr` | 0.6620 | 0.6529 | -0.0091 | 26 | CI99.94% [-0.1543, +0.1058] · 2↔3 câu khác nhau (p dấu=1) · CI95 thô [-0.0860, +0.0553] · **đếm câu đi ngược Δ** (3 câu tốt hơn vs 2 câu xấu đi, mà Δ = -0.0091) | trong ngưỡng nhiễu |
| `ndcg@10` | 0.5235 | 0.5816 | +0.0581 | 26 | CI99.94% [-0.0290, +0.1626] · 6↔11 câu khác nhau (p dấu=0.3323) · CI95 thô [+0.0050, +0.1153] | trong ngưỡng nhiễu |
| `precision@1` | 0.5385 | 0.5385 | +0.0000 | 26 | p=1 · 1↔1 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `precision@10` | 0.1346 | 0.1538 | +0.0192 | 26 | CI99.94% [-0.0192, +0.0654] · 2↔6 câu khác nhau (p dấu=0.2891) · CI95 thô [-0.0038, +0.0423] | trong ngưỡng nhiễu |
| `precision@20` | 0.0731 | 0.0827 | +0.0096 | 26 | CI99.94% [-0.0115, +0.0327] · 3↔7 câu khác nhau (p dấu=0.3438) · CI95 thô [-0.0038, +0.0231] | trong ngưỡng nhiễu |
| `precision@5` | 0.2308 | 0.2615 | +0.0308 | 26 | CI99.94% [-0.0308, +0.0923] · 1↔5 câu khác nhau (p dấu=0.2188) · CI95 thô [+0.0000, +0.0692] | trong ngưỡng nhiễu |
| `recall@1` | 0.2564 | 0.2500 | -0.0064 | 26 | CI99.94% [-0.1026, +0.0641] · 1↔1 câu khác nhau (p dấu=1) · CI95 thô [-0.0577, +0.0385] · **đếm câu đi ngược Δ** (1 câu tốt hơn vs 1 câu xấu đi, mà Δ = -0.0064) | trong ngưỡng nhiễu |
| `recall@10` | 0.5769 | 0.6603 | +0.0833 | 26 | CI99.94% [-0.0833, +0.2692] · 2↔6 câu khác nhau (p dấu=0.2891) · CI95 thô [-0.0128, +0.1859] | trong ngưỡng nhiễu |
| `recall@20` | 0.6218 | 0.7115 | +0.0897 | 26 | CI99.94% [-0.0962, +0.3077] · 3↔7 câu khác nhau (p dấu=0.3438) · CI95 thô [-0.0192, +0.2115] | trong ngưỡng nhiễu |
| `recall@5` | 0.5000 | 0.5769 | +0.0769 | 26 | CI99.94% [-0.0385, +0.2244] · 1↔5 câu khác nhau (p dấu=0.2188) · CI95 thô [+0.0064, +0.1603] | trong ngưỡng nhiễu |

## `category = table_lookup` — n = 4

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | bgem3 | bgem3-ctx | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.0000 | 0.2500 | +0.2500 | 4 | p=1 · 0↔1 câu đổi chiều · **trần `p` = 1** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@10` | 0.2500 | 0.5000 | +0.2500 | 4 | p=1 · 0↔1 câu đổi chiều · **trần `p` = 1** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@20` | 0.2500 | 0.5000 | +0.2500 | 4 | p=1 · 0↔1 câu đổi chiều · **trần `p` = 1** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@5` | 0.2500 | 0.2500 | +0.0000 | 4 | **0/4 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `map@20` | 0.0500 | 0.2812 | +0.2313 | 4 | CI99.94% [+0.0000, +0.8000] · 0↔2 câu khác nhau (p dấu=0.5) · CI95 thô [+0.0000, +0.6000] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.125/4 = 0.03125) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `mrr` | 0.0500 | 0.2812 | +0.2313 | 4 | CI99.94% [+0.0000, +0.8000] · 0↔2 câu khác nhau (p dấu=0.5) · CI95 thô [+0.0000, +0.6000] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.125/4 = 0.03125) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `ndcg@10` | 0.0967 | 0.3289 | +0.2322 | 4 | CI99.94% [+0.0000, +0.6131] · 0↔2 câu khác nhau (p dấu=0.5) · CI95 thô [+0.0000, +0.4643] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.3155/4 = 0.07887) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `precision@1` | 0.0000 | 0.2500 | +0.2500 | 4 | p=1 · 0↔1 câu đổi chiều · **trần `p` = 1** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `precision@10` | 0.0250 | 0.0500 | +0.0250 | 4 | CI99.94% [+0.0000, +0.1000] · 0↔1 câu khác nhau (p dấu=1) · CI95 thô [+0.0000, +0.0750] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.1/4 = 0.02500) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `precision@20` | 0.0125 | 0.0250 | +0.0125 | 4 | CI99.94% [+0.0000, +0.0500] · 0↔1 câu khác nhau (p dấu=1) · CI95 thô [+0.0000, +0.0375] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.05/4 = 0.01250) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `precision@5` | 0.0500 | 0.0500 | +0.0000 | 4 | **0/4 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `recall@1` | 0.0000 | 0.2500 | +0.2500 | 4 | CI99.94% [+0.0000, +1.0000] · 0↔1 câu khác nhau (p dấu=1) · CI95 thô [+0.0000, +0.7500] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/4 = 0.25000) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@10` | 0.2500 | 0.5000 | +0.2500 | 4 | CI99.94% [+0.0000, +1.0000] · 0↔1 câu khác nhau (p dấu=1) · CI95 thô [+0.0000, +0.7500] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/4 = 0.25000) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@20` | 0.2500 | 0.5000 | +0.2500 | 4 | CI99.94% [+0.0000, +1.0000] · 0↔1 câu khác nhau (p dấu=1) · CI95 thô [+0.0000, +0.7500] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/4 = 0.25000) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@5` | 0.2500 | 0.2500 | +0.0000 | 4 | **0/4 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |

