# So theo `lang`: `bgem3` → `bgem3-rr-c50`

- **2 nhóm × 15 metric = 30 phép kiểm.**
- Ngưỡng đã hiệu chỉnh Bonferroni: **α = 0.001667** (từ 0.05 chia cho 30).
- Không hiệu chỉnh thì ở α=0.05 kỳ vọng **1.5 kết quả "có ý nghĩa" thuần do ngẫu nhiên** — tức câu hỏi "nhóm nào cải thiện nhiều nhất" gần như bảo đảm tìm ra một cái, kể cả khi không có gì.
- `KHÔNG ĐỦ LỰC` = phép kiểm **không thể** đạt ý nghĩa dù kết quả thế nào (`p` nhỏ nhất McNemar cho `n` câu đổi chiều đã ≥ α). Nó **không** phải "không có khác biệt".
- `KHÔNG SO ĐƯỢC` = phân bố nhãn hai bên khác nhau nên mẫu số của metric khác nhau (`G2`/`TD-11`).
- `KHÔNG KẾT LUẬN` = một trong hai chuyện. (a) Khoảng tin cậy có **một biên gần 0 hơn một bước lưới `1/n`**: với metric rời rạc thưa, phân bố bootstrap nằm trên lưới bước `1/n`, nên việc khoảng chứa hay loại 0 phụ thuộc đúng một bước phân giải. Đã đo: đổi seed dịch biên **đúng một bước lưới**, và tăng 10.000 → 50.000 iterations **không đổi gì** — đừng chữa bằng cách tăng iterations. (b) **Đếm câu đi ngược `Δ`**: khoảng nói một hướng mà số câu tốt (b) **Biên không ổn định**: biên gần 0 nhất được đọc từ quá ít mẫu lại nên chính nó dao động qua 0. Đã đo ở `W2-08`: α đã hiệu chỉnh cho 39 phép kiểm để lại **6** mẫu lại trong đuôi của 10.000, và ở đó biên dưới **đổi dấu theo seed**; tăng lên 50.000 thì nó âm nhất quán, tức khoảng thật sự chứa 0.
- `TRÁI CHIỀU` = khoảng tin cậy đọc được **và** nó nói hai điều đối nhau: trung bình đi một hướng, còn **số câu** thì đi hướng ngược lại (thắng ít câu nhưng thắng đậm). Không phải lỗi, nhưng đừng đọc thành `hệ thống tốt hơn` — nhất là khi đổi lấy độ trễ.
- `TRÙNG KHỚP` = **0 câu khác nhau**. Đây là hàng chắc chắn nhất trong bảng, không phải hàng mơ hồ nhất — đừng đọc nó cùng nhóm với các cờ trên.

- ⚠️ **1/30 hàng `KHÔNG ĐỦ LỰC` và 0/30 hàng `KHÔNG KẾT LUẬN`.** Chia nhỏ tập đo rồi hiệu chỉnh cho 30 phép kiểm để lại rất ít lực; đọc bảng này để **loại** giả thuyết thì được, để **chọn người thắng** thì không.

## `lang = vi` — n = 127

| metric | bgem3 | bgem3-rr-c50 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.3307 | 0.5748 | +0.2441 | 127 | p=1.636e-06 · 6↔37 câu đổi chiều | khác biệt thật |
| `hit_rate@10` | 0.6457 | 0.7874 | +0.1417 | 127 | p=7.629e-06 · 0↔18 câu đổi chiều | khác biệt thật |
| `hit_rate@20` | 0.7087 | 0.7953 | +0.0866 | 127 | p=0.0009766 · 0↔11 câu đổi chiều | khác biệt thật |
| `hit_rate@5` | 0.5669 | 0.7559 | +0.1890 | 127 | p=1.192e-07 · 0↔24 câu đổi chiều | khác biệt thật |
| `map@20` | 0.3914 | 0.6144 | +0.2230 | 127 | CI99.83% [+0.1258, +0.3250] · 11↔61 câu khác nhau (p dấu=1.549e-09) · CI95 thô [+0.1611, +0.2862] | khác biệt thật |
| `mrr` | 0.4451 | 0.6573 | +0.2122 | 127 | CI99.83% [+0.1095, +0.3184] · 9↔53 câu khác nhau (p dấu=1.051e-08) · CI95 thô [+0.1465, +0.2773] | khác biệt thật |
| `ndcg@10` | 0.4532 | 0.6604 | +0.2072 | 127 | CI99.83% [+0.1235, +0.3000] · 10↔61 câu khác nhau (p dấu=4.645e-10) · CI95 thô [+0.1530, +0.2634] | khác biệt thật |
| `precision@1` | 0.3307 | 0.5748 | +0.2441 | 127 | p=1.636e-06 · 6↔37 câu đổi chiều | khác biệt thật |
| `precision@10` | 0.0780 | 0.0969 | +0.0189 | 127 | CI99.83% [+0.0087, +0.0315] · 0↔23 câu khác nhau (p dấu=2.384e-07) · CI95 thô [+0.0118, +0.0260] | khác biệt thật |
| `precision@20` | 0.0429 | 0.0488 | +0.0059 | 127 | CI99.83% [+0.0020, +0.0106] · 0↔15 câu khác nhau (p dấu=6.104e-05) · CI95 thô [+0.0031, +0.0087] | khác biệt thật |
| `precision@5` | 0.1276 | 0.1795 | +0.0520 | 127 | CI99.83% [+0.0283, +0.0787] · 1↔34 câu khác nhau (p dấu=2.095e-09) · CI95 thô [+0.0362, +0.0677] | khác biệt thật |
| `recall@1` | 0.2585 | 0.4738 | +0.2152 | 127 | CI99.83% [+0.0827, +0.3491] · 6↔37 câu khác nhau (p dấu=1.636e-06) · CI95 thô [+0.1299, +0.2992] | khác biệt thật |
| `recall@10` | 0.6050 | 0.7572 | +0.1522 | 127 | CI99.83% [+0.0709, +0.2559] · 0↔23 câu khác nhau (p dấu=2.384e-07) · CI95 thô [+0.0971, +0.2152] | khác biệt thật |
| `recall@20` | 0.6627 | 0.7651 | +0.1024 | 127 | CI99.83% [+0.0315, +0.1929] · 0↔15 câu khác nhau (p dấu=6.104e-05) · CI95 thô [+0.0551, +0.1535] | khác biệt thật |
| `recall@5` | 0.4948 | 0.7100 | +0.2152 | 127 | CI99.83% [+0.1168, +0.3307] · 1↔34 câu khác nhau (p dấu=2.095e-09) · CI95 thô [+0.1496, +0.2861] | khác biệt thật |

## `lang = en` — n = 82

| metric | bgem3 | bgem3-rr-c50 | Δ | n | kiểm định | kết luận |
|---|---:|---:|---:|---:|---|---|
| `hit_rate@1` | 0.3537 | 0.5366 | +0.1829 | 82 | p=0.00149 · 3↔18 câu đổi chiều | khác biệt thật |
| `hit_rate@10` | 0.5976 | 0.7439 | +0.1463 | 82 | p=0.0004883 · 0↔12 câu đổi chiều | khác biệt thật |
| `hit_rate@20` | 0.6220 | 0.7439 | +0.1220 | 82 | p=0.001953 · 0↔10 câu đổi chiều · **trần `p` = 0.001953** ở α=0.001667 | KHÔNG ĐỦ LỰC |
| `hit_rate@5` | 0.5122 | 0.7439 | +0.2317 | 82 | p=3.815e-06 · 0↔19 câu đổi chiều | khác biệt thật |
| `map@20` | 0.3759 | 0.5907 | +0.2149 | 82 | CI99.83% [+0.1029, +0.3343] · 6↔40 câu khác nhau (p dấu=3.103e-07) · CI95 thô [+0.1405, +0.2954] | khác biệt thật |
| `mrr` | 0.4306 | 0.6234 | +0.1928 | 82 | CI99.83% [+0.0681, +0.3272] · 5↔28 câu khác nhau (p dấu=6.619e-05) · CI95 thô [+0.1128, +0.2786] | khác biệt thật |
| `ndcg@10` | 0.4301 | 0.6291 | +0.1989 | 82 | CI99.83% [+0.0937, +0.3178] · 6↔40 câu khác nhau (p dấu=3.103e-07) · CI95 thô [+0.1286, +0.2760] | khác biệt thật |
| `precision@1` | 0.3537 | 0.5366 | +0.1829 | 82 | p=0.00149 · 3↔18 câu đổi chiều | khác biệt thật |
| `precision@10` | 0.0878 | 0.1073 | +0.0195 | 82 | CI99.83% [+0.0061, +0.0341] · 1↔17 câu khác nhau (p dấu=0.000145) · CI95 thô [+0.0110, +0.0293] | khác biệt thật |
| `precision@20` | 0.0470 | 0.0543 | +0.0073 | 82 | CI99.83% [+0.0012, +0.0146] · 1↔13 câu khác nhau (p dấu=0.001831) · CI95 thô [+0.0030, +0.0116] | khác biệt thật |
| `precision@5` | 0.1439 | 0.2098 | +0.0659 | 82 | CI99.83% [+0.0341, +0.1024] · 0↔26 câu khác nhau (p dấu=2.98e-08) · CI95 thô [+0.0439, +0.0878] | khác biệt thật |
| `recall@1` | 0.2398 | 0.4065 | +0.1667 | 82 | CI99.83% [+0.0325, +0.3150] · 3↔18 câu khác nhau (p dấu=0.00149) · CI95 thô [+0.0793, +0.2602] | khác biệt thật |
| `recall@10` | 0.5447 | 0.7012 | +0.1565 | 82 | CI99.83% [+0.0528, +0.2825] · 1↔17 câu khác nhau (p dấu=0.000145) · CI95 thô [+0.0854, +0.2378] | khác biệt thật |
| `recall@20` | 0.5854 | 0.7073 | +0.1220 | 82 | CI99.83% [+0.0305, +0.2439] · 1↔13 câu khác nhau (p dấu=0.001831) · CI95 thô [+0.0569, +0.1951] | khác biệt thật |
| `recall@5` | 0.4492 | 0.6911 | +0.2419 | 82 | CI99.83% [+0.1179, +0.3801] · 0↔26 câu khác nhau (p dấu=2.98e-08) · CI95 thô [+0.1585, +0.3313] | khác biệt thật |

