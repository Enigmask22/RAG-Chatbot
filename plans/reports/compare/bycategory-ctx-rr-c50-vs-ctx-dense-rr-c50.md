# So theo `category`: `bgem3-ctx-rr-c50` → `bgem3-ctx-dense-rr-c50`

- **6 nhóm × 15 metric = 90 phép kiểm.**
- Ngưỡng đã hiệu chỉnh Bonferroni: **α = 0.0005556** (từ 0.05 chia cho 90).
- Không hiệu chỉnh thì ở α=0.05 kỳ vọng **4.5 kết quả "có ý nghĩa" thuần do ngẫu nhiên** — tức câu hỏi "nhóm nào cải thiện nhiều nhất" gần như bảo đảm tìm ra một cái, kể cả khi không có gì.
- `KHÔNG ĐỦ LỰC` = phép kiểm **không thể** đạt ý nghĩa dù kết quả thế nào (`p` nhỏ nhất McNemar cho `n` câu đổi chiều đã ≥ α). Nó **không** phải "không có khác biệt".
- `KHÔNG SO ĐƯỢC` = phân bố nhãn hai bên khác nhau nên mẫu số của metric khác nhau (`G2`/`TD-11`).
- `KHÔNG KẾT LUẬN` = một trong hai chuyện. (a) Khoảng tin cậy có **một biên gần 0 hơn một bước lưới `1/n`**: với metric rời rạc thưa, phân bố bootstrap nằm trên lưới bước `1/n`, nên việc khoảng chứa hay loại 0 phụ thuộc đúng một bước phân giải. Đã đo: đổi seed dịch biên **đúng một bước lưới**, và tăng 10.000 → 50.000 iterations **không đổi gì** — đừng chữa bằng cách tăng iterations. (b) **Đếm câu đi ngược `Δ`**: khoảng nói một hướng mà số câu tốt (b) **Biên không ổn định**: biên gần 0 nhất được đọc từ quá ít mẫu lại nên chính nó dao động qua 0. Đã đo ở `W2-08`: α đã hiệu chỉnh cho 39 phép kiểm để lại **6** mẫu lại trong đuôi của 10.000, và ở đó biên dưới **đổi dấu theo seed**; tăng lên 50.000 thì nó âm nhất quán, tức khoảng thật sự chứa 0.
- `TRÁI CHIỀU` = khoảng tin cậy đọc được **và** nó nói hai điều đối nhau: trung bình đi một hướng, còn **số câu** thì đi hướng ngược lại (thắng ít câu nhưng thắng đậm). Không phải lỗi, nhưng đừng đọc thành `hệ thống tốt hơn` — nhất là khi đổi lấy độ trễ.
- `TRÙNG KHỚP` = **0 câu khác nhau**. Đây là hàng chắc chắn nhất trong bảng, không phải hàng mơ hồ nhất — đừng đọc nó cùng nhóm với các cờ trên.

- ⚠️ **30/90 hàng `KHÔNG ĐỦ LỰC` và 35/90 hàng `KHÔNG KẾT LUẬN`.** Chia nhỏ tập đo rồi hiệu chỉnh cho 90 phép kiểm để lại rất ít lực; đọc bảng này để **loại** giả thuyết thì được, để **chọn người thắng** thì không.

## `category = factoid` — n = 68

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | bgem3-ctx-rr-c50 | bgem3-ctx-dense-rr-c50 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.6618 | 0.6471 | -0.0147 | 68 | p=1 · 1↔0 câu đổi chiều · **trần `p` = 1** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@10` | 0.9412 | 0.8971 | -0.0441 | 68 | p=0.25 · 3↔0 câu đổi chiều · **trần `p` = 0.25** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@20` | 0.9412 | 0.8971 | -0.0441 | 68 | p=0.25 · 3↔0 câu đổi chiều · **trần `p` = 0.25** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@5` | 0.9265 | 0.8824 | -0.0441 | 68 | p=0.25 · 3↔0 câu đổi chiều · **trần `p` = 0.25** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `map@20` | 0.7728 | 0.7434 | -0.0294 | 68 | CI99.94% [-0.1078, +0.0000] · 4↔0 câu khác nhau (p dấu=0.125) · CI95 thô [-0.0686, -0.0025] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.1667/68 = 0.00245) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `mrr` | 0.7716 | 0.7422 | -0.0294 | 68 | CI99.94% [-0.1078, +0.0000] · 4↔0 câu khác nhau (p dấu=0.125) · CI95 thô [-0.0686, -0.0025] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.1667/68 = 0.00245) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `ndcg@10` | 0.8150 | 0.7817 | -0.0333 | 68 | CI99.94% [-0.1161, +0.0000] · 4↔0 câu khác nhau (p dấu=0.125) · CI95 thô [-0.0739, -0.0019] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.1309/68 = 0.00193) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `precision@1` | 0.6618 | 0.6471 | -0.0147 | 68 | p=1 · 1↔0 câu đổi chiều · **trần `p` = 1** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `precision@10` | 0.0956 | 0.0912 | -0.0044 | 68 | CI99.94% [-0.0147, +0.0000] · 3↔0 câu khác nhau (p dấu=0.25) · CI95 thô [-0.0103, +0.0000] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.1/68 = 0.00147) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `precision@20` | 0.0478 | 0.0456 | -0.0022 | 68 | CI99.94% [-0.0074, +0.0000] · 3↔0 câu khác nhau (p dấu=0.25) · CI95 thô [-0.0051, +0.0000] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.05/68 = 0.00074) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `precision@5` | 0.1882 | 0.1794 | -0.0088 | 68 | CI99.94% [-0.0294, +0.0000] · 3↔0 câu khác nhau (p dấu=0.25) · CI95 thô [-0.0206, +0.0000] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.2/68 = 0.00294) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@1` | 0.6618 | 0.6471 | -0.0147 | 68 | CI99.94% [-0.0882, +0.0000] · 1↔0 câu khác nhau (p dấu=1) · CI95 thô [-0.0441, +0.0000] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/68 = 0.01471) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@10` | 0.9412 | 0.8971 | -0.0441 | 68 | CI99.94% [-0.1471, +0.0000] · 3↔0 câu khác nhau (p dấu=0.25) · CI95 thô [-0.1029, +0.0000] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/68 = 0.01471) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@20` | 0.9412 | 0.8971 | -0.0441 | 68 | CI99.94% [-0.1471, +0.0000] · 3↔0 câu khác nhau (p dấu=0.25) · CI95 thô [-0.1029, +0.0000] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/68 = 0.01471) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@5` | 0.9265 | 0.8824 | -0.0441 | 68 | CI99.94% [-0.1471, +0.0000] · 3↔0 câu khác nhau (p dấu=0.25) · CI95 thô [-0.1029, +0.0000] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/68 = 0.01471) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |

## `category = cross_lingual` — n = 43

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | bgem3-ctx-rr-c50 | bgem3-ctx-dense-rr-c50 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.4419 | 0.4884 | +0.0465 | 43 | p=0.5 · 0↔2 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@10` | 0.5581 | 0.6512 | +0.0930 | 43 | p=0.125 · 0↔4 câu đổi chiều · **trần `p` = 0.125** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@20` | 0.5814 | 0.6744 | +0.0930 | 43 | p=0.125 · 0↔4 câu đổi chiều · **trần `p` = 0.125** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@5` | 0.5581 | 0.6279 | +0.0698 | 43 | p=0.25 · 0↔3 câu đổi chiều · **trần `p` = 0.25** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `map@20` | 0.4876 | 0.5416 | +0.0541 | 43 | CI99.94% [+0.0000, +0.1953] · 0↔5 câu khác nhau (p dấu=0.0625) · CI95 thô [+0.0029, +0.1252] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.03409/43 = 0.00079) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `mrr` | 0.4866 | 0.5407 | +0.0541 | 43 | CI99.94% [+0.0000, +0.1953] · 0↔5 câu khác nhau (p dấu=0.0625) · CI95 thô [+0.0029, +0.1252] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.03409/43 = 0.00079) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `ndcg@10` | 0.5041 | 0.5669 | +0.0628 | 43 | CI99.94% [+0.0000, +0.2093] · 0↔4 câu khác nhau (p dấu=0.125) · CI95 thô [+0.0073, +0.1363] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.3155/43 = 0.00734) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `precision@1` | 0.4419 | 0.4884 | +0.0465 | 43 | p=0.5 · 0↔2 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `precision@10` | 0.0651 | 0.0744 | +0.0093 | 43 | CI99.94% [+0.0000, +0.0279] · 0↔4 câu khác nhau (p dấu=0.125) · CI95 thô [+0.0023, +0.0186] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.1/43 = 0.00233) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `precision@20` | 0.0337 | 0.0384 | +0.0047 | 43 | CI99.94% [+0.0000, +0.0140] · 0↔4 câu khác nhau (p dấu=0.125) · CI95 thô [+0.0012, +0.0093] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.05/43 = 0.00116) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `precision@5` | 0.1256 | 0.1395 | +0.0140 | 43 | CI99.94% [+0.0000, +0.0465] · 0↔3 câu khác nhau (p dấu=0.25) · CI95 thô [+0.0000, +0.0326] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.2/43 = 0.00465) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@1` | 0.4070 | 0.4535 | +0.0465 | 43 | CI99.94% [+0.0000, +0.1860] · 0↔2 câu khác nhau (p dấu=0.5) · CI95 thô [+0.0000, +0.1163] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/43 = 0.02326) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@10` | 0.5581 | 0.6512 | +0.0930 | 43 | CI99.94% [+0.0000, +0.2791] · 0↔4 câu khác nhau (p dấu=0.125) · CI95 thô [+0.0233, +0.1860] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/43 = 0.02326) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@20` | 0.5814 | 0.6744 | +0.0930 | 43 | CI99.94% [+0.0000, +0.2791] · 0↔4 câu khác nhau (p dấu=0.125) · CI95 thô [+0.0233, +0.1860] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/43 = 0.02326) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@5` | 0.5465 | 0.6163 | +0.0698 | 43 | CI99.94% [+0.0000, +0.2326] · 0↔3 câu khác nhau (p dấu=0.25) · CI95 thô [+0.0000, +0.1628] · **biên cách 0 dưới một bước lưới** (0.00000 < 1/43 = 0.02326) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |

## `category = adversarial` — n = 34

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | bgem3-ctx-rr-c50 | bgem3-ctx-dense-rr-c50 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.5000 | 0.5000 | +0.0000 | 34 | p=1 · 1↔1 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@10` | 0.7353 | 0.7353 | +0.0000 | 34 | p=1 · 1↔1 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@20` | 0.7353 | 0.7353 | +0.0000 | 34 | p=1 · 1↔1 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@5` | 0.7353 | 0.7353 | +0.0000 | 34 | p=1 · 1↔1 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `map@20` | 0.5833 | 0.5833 | +0.0000 | 34 | CI99.94% [-0.1471, +0.1471] · 1↔1 câu khác nhau (p dấu=1) · CI95 thô [-0.0882, +0.0882] | trong ngưỡng nhiễu |
| `mrr` | 0.5907 | 0.5907 | +0.0000 | 34 | CI99.94% [-0.1471, +0.1471] · 1↔1 câu khác nhau (p dấu=1) · CI95 thô [-0.0882, +0.0882] | trong ngưỡng nhiễu |
| `ndcg@10` | 0.6235 | 0.6235 | +0.0000 | 34 | CI99.94% [-0.1471, +0.1471] · 1↔1 câu khác nhau (p dấu=1) · CI95 thô [-0.0882, +0.0882] | trong ngưỡng nhiễu |
| `precision@1` | 0.5000 | 0.5000 | +0.0000 | 34 | p=1 · 1↔1 câu đổi chiều · **trần `p` = 0.5** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `precision@10` | 0.0794 | 0.0794 | +0.0000 | 34 | CI99.94% [-0.0147, +0.0147] · 1↔1 câu khác nhau (p dấu=1) · CI95 thô [-0.0088, +0.0088] | trong ngưỡng nhiễu |
| `precision@20` | 0.0397 | 0.0397 | +0.0000 | 34 | CI99.94% [-0.0074, +0.0074] · 1↔1 câu khác nhau (p dấu=1) · CI95 thô [-0.0044, +0.0044] | trong ngưỡng nhiễu |
| `precision@5` | 0.1588 | 0.1588 | +0.0000 | 34 | CI99.94% [-0.0294, +0.0294] · 1↔1 câu khác nhau (p dấu=1) · CI95 thô [-0.0176, +0.0176] | trong ngưỡng nhiễu |
| `recall@1` | 0.4706 | 0.4706 | +0.0000 | 34 | CI99.94% [-0.1471, +0.1471] · 1↔1 câu khác nhau (p dấu=1) · CI95 thô [-0.0882, +0.0882] | trong ngưỡng nhiễu |
| `recall@10` | 0.7353 | 0.7353 | +0.0000 | 34 | CI99.94% [-0.1471, +0.1471] · 1↔1 câu khác nhau (p dấu=1) · CI95 thô [-0.0882, +0.0882] | trong ngưỡng nhiễu |
| `recall@20` | 0.7353 | 0.7353 | +0.0000 | 34 | CI99.94% [-0.1471, +0.1471] · 1↔1 câu khác nhau (p dấu=1) · CI95 thô [-0.0882, +0.0882] | trong ngưỡng nhiễu |
| `recall@5` | 0.7353 | 0.7353 | +0.0000 | 34 | CI99.94% [-0.1471, +0.1471] · 1↔1 câu khác nhau (p dấu=1) · CI95 thô [-0.0882, +0.0882] | trong ngưỡng nhiễu |

## `category = multi_hop` — n = 34

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | bgem3-ctx-rr-c50 | bgem3-ctx-dense-rr-c50 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.7647 | 0.7353 | -0.0294 | 34 | p=1 · 1↔0 câu đổi chiều · **trần `p` = 1** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@10` | 0.9412 | 0.9118 | -0.0294 | 34 | p=1 · 1↔0 câu đổi chiều · **trần `p` = 1** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@20` | 0.9706 | 0.9412 | -0.0294 | 34 | p=1 · 1↔0 câu đổi chiều · **trần `p` = 1** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@5` | 0.9412 | 0.9118 | -0.0294 | 34 | p=1 · 1↔0 câu đổi chiều · **trần `p` = 1** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `map@20` | 0.7042 | 0.7002 | -0.0040 | 34 | CI99.94% [-0.0799, +0.0425] · 1↔5 câu khác nhau (p dấu=0.2188) · CI95 thô [-0.0417, +0.0240] · **đếm câu đi ngược Δ** (5 câu tốt hơn vs 1 câu xấu đi, mà Δ = -0.0040) | trong ngưỡng nhiễu |
| `mrr` | 0.8403 | 0.8113 | -0.0291 | 34 | CI99.94% [-0.1761, +0.0018] · 1↔1 câu khác nhau (p dấu=1) · CI95 thô [-0.0882, +0.0011] · **đếm câu đi ngược Δ** (1 câu tốt hơn vs 1 câu xấu đi, mà Δ = -0.0291) | trong ngưỡng nhiễu |
| `ndcg@10` | 0.7647 | 0.7650 | +0.0004 | 34 | CI99.94% [-0.0927, +0.0599] · 1↔3 câu khác nhau (p dấu=0.625) · CI95 thô [-0.0463, +0.0365] | trong ngưỡng nhiễu |
| `precision@1` | 0.7647 | 0.7353 | -0.0294 | 34 | p=1 · 1↔0 câu đổi chiều · **trần `p` = 1** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `precision@10` | 0.1647 | 0.1706 | +0.0059 | 34 | CI99.94% [-0.0147, +0.0265] · 1↔3 câu khác nhau (p dấu=0.625) · CI95 thô [-0.0059, +0.0176] | trong ngưỡng nhiễu |
| `precision@20` | 0.0868 | 0.0882 | +0.0015 | 34 | CI99.94% [-0.0074, +0.0118] · 1↔2 câu khác nhau (p dấu=1) · CI95 thô [-0.0029, +0.0059] | trong ngưỡng nhiễu |
| `precision@5` | 0.3294 | 0.3294 | +0.0000 | 34 | CI99.94% [-0.0294, +0.0294] · 1↔1 câu khác nhau (p dấu=1) · CI95 thô [-0.0176, +0.0176] | trong ngưỡng nhiễu |
| `recall@1` | 0.3775 | 0.3627 | -0.0147 | 34 | CI99.94% [-0.0882, +0.0000] · 1↔0 câu khác nhau (p dấu=1) · CI95 thô [-0.0441, +0.0000] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.5/34 = 0.01471) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `recall@10` | 0.8137 | 0.8431 | +0.0294 | 34 | CI99.94% [-0.0735, +0.1324] · 1↔3 câu khác nhau (p dấu=0.625) · CI95 thô [-0.0294, +0.0882] | trong ngưỡng nhiễu |
| `recall@20` | 0.8578 | 0.8676 | +0.0098 | 34 | CI99.94% [-0.0735, +0.0980] · 1↔2 câu khác nhau (p dấu=1) · CI95 thô [-0.0343, +0.0539] | trong ngưỡng nhiễu |
| `recall@5` | 0.8137 | 0.8137 | +0.0000 | 34 | CI99.94% [-0.0735, +0.0735] · 1↔1 câu khác nhau (p dấu=1) · CI95 thô [-0.0441, +0.0441] | trong ngưỡng nhiễu |

## `category = aggregation` — n = 26

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | bgem3-ctx-rr-c50 | bgem3-ctx-dense-rr-c50 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.6923 | 0.6923 | +0.0000 | 26 | **0/26 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `hit_rate@10` | 0.8846 | 0.8846 | +0.0000 | 26 | **0/26 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `hit_rate@20` | 0.9231 | 0.8846 | -0.0385 | 26 | p=1 · 1↔0 câu đổi chiều · **trần `p` = 1** ở α=0.0005556 | KHÔNG ĐỦ LỰC |
| `hit_rate@5` | 0.8846 | 0.8846 | +0.0000 | 26 | **0/26 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `map@20` | 0.5961 | 0.6264 | +0.0303 | 26 | CI99.94% [-0.0584, +0.1389] · 4↔7 câu khác nhau (p dấu=0.5488) · CI95 thô [-0.0211, +0.0877] | trong ngưỡng nhiễu |
| `mrr` | 0.7853 | 0.7821 | -0.0032 | 26 | CI99.94% [-0.0192, +0.0000] · 1↔0 câu khác nhau (p dấu=1) · CI95 thô [-0.0096, +0.0000] · **biên cách 0 dưới một bước lưới** (0.00000 < 0.08333/26 = 0.00321) · **biên không ổn định**: chính nó dao động [+0.0000, +0.0000] nên việc khoảng loại 0 là chuyện của số mẫu lại, không phải của dữ liệu | KHÔNG KẾT LUẬN |
| `ndcg@10` | 0.6797 | 0.7015 | +0.0217 | 26 | CI99.94% [-0.0606, +0.1115] · 2↔5 câu khác nhau (p dấu=0.4531) · CI95 thô [-0.0248, +0.0705] | trong ngưỡng nhiễu |
| `precision@1` | 0.6923 | 0.6923 | +0.0000 | 26 | **0/26 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `precision@10` | 0.1731 | 0.1808 | +0.0077 | 26 | CI99.94% [-0.0231, +0.0423] · 2↔4 câu khác nhau (p dấu=0.6875) · CI95 thô [-0.0115, +0.0269] | trong ngưỡng nhiễu |
| `precision@20` | 0.0962 | 0.0981 | +0.0019 | 26 | CI99.94% [-0.0154, +0.0192] · 3↔4 câu khác nhau (p dấu=1) · CI95 thô [-0.0077, +0.0115] | trong ngưỡng nhiễu |
| `precision@5` | 0.3385 | 0.3462 | +0.0077 | 26 | CI99.94% [-0.0385, +0.0615] · 1↔2 câu khác nhau (p dấu=1) · CI95 thô [-0.0154, +0.0308] | trong ngưỡng nhiễu |
| `recall@1` | 0.3141 | 0.3141 | +0.0000 | 26 | **0/26 câu khác nhau** — hai lần chạy cho kết quả trùng khớp | TRÙNG KHỚP |
| `recall@10` | 0.7500 | 0.7756 | +0.0256 | 26 | CI99.94% [-0.1218, +0.1731] · 2↔4 câu khác nhau (p dấu=0.6875) · CI95 thô [-0.0577, +0.1090] | trong ngưỡng nhiễu |
| `recall@20` | 0.8205 | 0.8397 | +0.0192 | 26 | CI99.94% [-0.1218, +0.1667] · 3↔4 câu khác nhau (p dấu=1) · CI95 thô [-0.0641, +0.1026] | trong ngưỡng nhiễu |
| `recall@5` | 0.7308 | 0.7436 | +0.0128 | 26 | CI99.94% [-0.0962, +0.1218] · 1↔2 câu khác nhau (p dấu=1) · CI95 thô [-0.0449, +0.0705] | trong ngưỡng nhiễu |

## `category = table_lookup` — n = 4

> Đã hiệu chỉnh Bonferroni cho **90** phép kiểm: α = 0.00055556.

| metric | bgem3-ctx-rr-c50 | bgem3-ctx-dense-rr-c50 | Δ | n | kiểm định | kết luận |
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

