# `W2-08-prep` — `--category`/`--lang` cho `compare.py`

> Trạng thái: **xong** · 2026-08-21 · Điều kiện của DoD `W2-09`
> Lệnh: `make eval-compare-by BASE=… CAND=… BY=category|lang` ·
> `make eval-compare-subset BASE=… CAND=… CAT=cross_lingual`
> Test: `tests/unit/test_eval_compare.py` — **82 case** (từ 39)
> Đã **xoá** `scripts/category_compare.py`

---

## 1. Vì sao hạng mục này đứng trước `W2-08`

DoD `W2-09` đòi "ít nhất 2 nhận xét về *category nào cải thiện nhiều nhất*".
`compare.py` không có chiều đó, nên `W2-05` phải viết `scripts/category_compare.py`
làm tạm — và chính chỗ thiếu đó đã để **một mức tụt có ý nghĩa của `W2-04` đi qua
không ai thấy** cho tới lúc tổng kết. Hai công cụ trả lời cùng một câu hỏi là hai
câu trả lời khác nhau đang chờ xảy ra, nên bản tạm bị xoá trong cùng commit.

Nhưng phần khó **không** phải viết bộ lọc. Nó là: *hỏi 6 nhóm × 15 metric = 90 câu
hỏi rồi chọn cái to nhất* là một quy trình chọn mẫu, và câu trả lời của nó gần như
**bảo đảm** tìm ra "người thắng" kể cả khi không có gì.

---

## 2. Hai lệnh, vì có hai loại suy luận

| Lệnh | Làm gì | Hiệu chỉnh? |
|---|---|---|
| `--category cross_lingual` | So trên **một** nhóm đã nêu trước | **Không** — một giả thuyết |
| `--by category` | Quét **mọi** nhóm, một bảng mỗi nhóm | **Có** — Bonferroni trên `nhóm × metric` |

Phân biệt này là nội dung chính của hạng mục, không phải chi tiết CLI. Nêu tên
nhóm **trước khi xem số** là kiểm một giả thuyết; quét hết rồi chọn cái to nhất là
chọn mẫu. Cùng một con số, hai mức bằng chứng khác nhau. CLI **từ chối** dùng cả
hai cùng lúc, vì lúc đó không nói được kết quả thuộc loại nào.

`compare_runs` (một cặp) **giữ nguyên `α = 0,05`**, và đó là điều kiện chứ không
phải tình cờ: đổi ngưỡng ở đó sẽ lặng lẽ viết lại kết luận của mọi bảng đã công bố
từ `W2-01` đến `W2-07`. Có test ghim.

**Bonferroni, không Holm**, dù Holm mạnh hơn: Holm dùng ngưỡng khác nhau cho từng
hạng, và một khoảng tin cậy bootstrap không biểu diễn được điều đó bằng một khoảng
duy nhất — bảng sẽ có `p` theo Holm cạnh CI theo Bonferroni, hai luật trong một
bảng. Nhất quán đáng hơn một chút lực kiểm định.

`m` đếm số phép kiểm **thử** (`nhóm × metric`), tính được **trước** khi chạy. Nếu
`m` phụ thuộc số hàng so được thì chính nó thành một lựa chọn dựa trên dữ liệu.

---

## 3. `KHÔNG ĐỦ LỰC` — phân biệt quan trọng nhất, và nó tính được

`golden_v1` có `table_lookup` = **4 câu**. McNemar exact chỉ dùng `n` câu **đổi
chiều**, và `p` nhỏ nhất nó trả về được là `2/2ⁿ`:

| n câu đổi chiều | 2 | 3 | 4 | 5 | **6** | 11 | **12** |
|---|---|---|---|---|---|---|---|
| trần `p` | 0,5 | 0,25 | **0,125** | 0,0625 | **0,03125** | 0,00098 | **0,00049** |

Nên **dưới 6 câu đổi chiều thì không bao giờ đạt ý nghĩa ở α = 0,05**, và dưới 12
thì không bao giờ đạt ở α đã hiệu chỉnh cho 90 phép kiểm (0,000556).

Không có cờ này thì "trong ngưỡng nhiễu" của `table_lookup` đọc như *bằng chứng
không có hiệu ứng*, trong khi nó là *bằng chứng không có lực kiểm định* — hai
chuyện trái ngược nhau dẫn tới cùng một dòng chữ. Cờ phân biệt được cả ca tinh tế:
`cross_lingual` `hit_rate@10` có `5↔1`, trần `p` = 0,031 — **có** lực ở α=0,05 và
**không** ở α đã hiệu chỉnh. Nó không phải "n nhỏ" đội lốt.

---

## 4. `KHÔNG KẾT LUẬN` — lỗi tôi tự tạo ra rồi phải đo mới thấy

Sau khi thêm hiệu chỉnh Bonferroni, bảng chia nhóm hiện `CI99,72% [+0,0000, +0,1765]`
— biên dưới **đúng bằng 0**. Và không phải một dòng: **4/4** dòng `recall@5` trong
bảng `bgem3 → bgem3-rrf-k1-c20` đều có một biên ghim đúng `0,0000`, nên cả bốn bị
dán "trong ngưỡng nhiễu".

Lo ngại đầu tiên của tôi là sai: tôi cho rằng 10.000 iterations không đủ để đọc
phân vị 0,14% (chỉ 13,9 điểm ở đuôi) và định tự động nâng lên ~72.000. **Đo trước
khi sửa** — 6 seed × 3 mức iterations trên cùng dữ liệu:

| iterations | điểm ở đuôi | dao động biên CI theo seed |
|---|---|---|
| 10.000 | 13,9 | **0,0233** |
| 50.000 | 69,4 | **0,0233** |
| 200.000 | 277,8 | 0,0000 |

`0,0233` = **đúng `1/43`**. Đó là **một bước lưới**, không phải sai số Monte Carlo:
phần lớn câu có hiệu bằng 0 nên phân bố trung bình bootstrap nằm trên một lưới bước
`1/n`, và phân vị cực đoan rơi đúng lên điểm 0 của lưới. Tăng iterations gấp 5 lần
**không đổi gì**. Ghi lại để lần sau không ai "sửa" nó bằng cách tăng iterations.

Nên kết luận đúng của một hàng như thế không phải "không có khác biệt" mà là
**"không kết luận được"**: việc khoảng chứa 0 phụ thuộc đúng một bước phân giải của
dữ liệu. Gộp nó vào "trong ngưỡng nhiễu" là chỗ một giới hạn phân giải bị đọc thành
một kết luận — cùng họ với `KHÔNG ĐỦ LỰC`, và cùng họ với `KHÔNG SO ĐƯỢC` ở `W2-03`.

⚠️ **Tự phê:** cờ này tồn tại vì chính bản hiệu chỉnh của tôi tạo ra vấn đề. Bonferroni
trên metric rời rạc thưa làm CI suy biến về biên 0, tức phép hiệu chỉnh biến hiệu
ứng thật thành không-phát-hiện **do cấu tạo**. Không đo thì tôi đã công bố một bảng
toàn "trong ngưỡng nhiễu" và tưởng đó là kết quả.

---

## 5. Kết quả: phát hiện `cross_lingual` của `W2-04` **sống sót**, nhưng dẫn chứng thì không

Chạy lại phát hiện đã công bố bằng công cụ mới, ở bài kiểm khắt khe nhất
(6 nhóm × 15 metric = **90 phép kiểm**, α = 0,000556):

| metric | bgem3 → bgem3-rrf-k1-c20 (43 câu `cross_lingual`) | kết luận |
|---|---|---|
| `map@20` | 0,1963 → 0,1306 · CI99,94% **[−0,1235, −0,0193]** | ✅ **khác biệt thật** |
| `mrr` | 0,2028 → 0,1349 · CI99,94% **[−0,1335, −0,0193]** | ✅ **khác biệt thật** |
| `ndcg@10` | 0,2538 → 0,1707 · CI99,94% **[−0,1630, −0,0054]** | ✅ **khác biệt thật** |
| `recall@5` | 0,3023 → 0,2093 · CI95 thô [−0,1860, −0,0233] | ⚠️ `KHÔNG KẾT LUẬN` (biên đúng 0) |
| `hit_rate@5` | `4↔0` · `p = 0,125` | ⚠️ `KHÔNG ĐỦ LỰC` (trần `p` = 0,125) |

**Kết luận đúng, dẫn chứng sai.** Nhánh hybrid của `W2-04` thật sự làm tụt
`cross_lingual`, và nó sống sót hiệu chỉnh cho **cả 90 phép kiểm** — bằng chứng
mạnh hơn lúc phát biểu. Nhưng hai metric mà CHECKLIST trích (`recall@5` CI95 và
`hit_rate@5` `4↔0`) **không** phải chỗ nó sống sót; ba metric sống sót
(`map@20`, `mrr`, `ndcg@10`) thì CHECKLIST không nhắc.

### Cùng khuôn, ba lần

Kiểm tiếp hai cặp còn lại mà bản tạm từng dựng kết luận:

* CHECKLIST: *"reranker vá lại đúng phần đó (còn `1↔0`, `p = 1,000`)"* — `1↔0` có
  **trần `p` = 1,0**, tức phép kiểm đó không mang một chút thông tin nào. Bằng chứng
  thật cho câu đó là cặp khác: `bgem3-rrf-k1-c20 → bgem3-rr-c50` trên
  `cross_lingual`, `hit_rate@5` **0↔14, `p` = 0,00012** — khổng lồ và thật.
* `W2-05` §: `hit_rate@1` của `c50 → c100` là `0↔4`, trần `p` = 0,125. Bảng cũ ghi
  "trong ngưỡng nhiễu"; docstring `BINARY_METRICS` thì đã lý giải đúng ("bốn lần
  tung xu cùng mặt thì không kết luận được gì"). **Cột kết luận và docstring của
  cùng một file đã nói hai điều khác nhau từ `W2-05`**; giờ chúng khớp.

💡 Khuôn chung: mọi dẫn chứng sai đều là **một `p` không có ý nghĩa được trích cạnh
một CI có ý nghĩa**, và nó trông như bằng chứng bổ trợ trong khi nó là số 0. Cờ
`KHÔNG ĐỦ LỰC` bắt được cả ba mà không cần ai đi tìm.

---

## 6. Đối chứng: công cụ **không** dán "không kết luận" cho mọi thứ

Nếu mọi bảng đều thành "không kết luận" thì cờ vô dụng. Chạy trên hiệu ứng lớn
(reranker, `bgem3 → bgem3-rr-c50`), chia theo `lang`, 30 phép kiểm:

```
1/30 hàng KHÔNG ĐỦ LỰC · 0/30 hàng KHÔNG KẾT LUẬN
lang = vi (127 câu): 15/15 khác biệt thật
lang = en ( 82 câu): 14/15 khác biệt thật
```

**29/30 sống sót Bonferroni.** So với `W2-04` (60/90 hàng không kết luận được), sự
đối lập chính là thông tin: `W2-05` là hiệu ứng lớn thật, `W2-04` là hiệu ứng biên
mà bảng tổng che đi.

---

## 7. Tái sinh 12 file bằng chứng — và script xác minh bắt được điều tôi đã cho là không xảy ra

Đổi định dạng `p` từ `{:.3f}` sang `{:.4g}`: `p=0.000` không phân biệt được `4e-9`
với `4,9e-4`, mà `4,9e-4` nằm ngay cạnh ngưỡng. Cải thiện thật, nhưng nó chạm 12
file `compare/` đã công bố.

Tôi tự nhủ "`family_size=1` nên hành vi một-cặp không đổi", và viết một script tái
sinh **so mọi ô trừ ô kiểm định** với bản cũ. Nó nổ ở 2 file:

* `hit_rate@1` + `precision@1` của `c50 → c100`: "trong ngưỡng nhiễu" → `KHÔNG ĐỦ LỰC`
* `recall@1` của `c20 → c50`: "trong ngưỡng nhiễu" → `KHÔNG KẾT LUẬN`

Tôi đã sai: `α` thì không đổi, nhưng **hai cờ mới áp cả ở `family_size=1`**. Cả 3
hàng đổi đều đúng và đều làm sắc thêm; không kết luận **dương** nào bị lật, vì cả
hai cờ chỉ biến "không tìm thấy khác biệt" thành "không thể tìm thấy". Và
`13/15 trong ngưỡng nhiễu` của `W2-05` (nền dense vs nền hybrid — cơ sở của kết
luận kiến trúc "sau khi có reranker, tầng hybrid không còn đo được") **còn nguyên
13, không hàng nào bị gắn cờ**.

💡 Bài học lặp lại từ `W2-07` §8: cách duy nhất biết một đợt sửa "không đổi con số
nào" là **có một script nói không**. Ở đây nó nói không, và nó đúng.

---

## 8. Việc còn lại

* ⚠️ **Bảng một-cặp vẫn không hiệu chỉnh cho 15 metric.** Đó là một lựa chọn có
  chủ đích (§2) nhưng nó là một giới hạn thật: một bảng 15 metric ở α=0,05 kỳ vọng
  0,75 kết quả dương giả. Không sửa ở đây vì nó sẽ viết lại kết luận của `W2-01`…
  `W2-07`; **quyết định đó thuộc về `W2-09`**, và phải nói ra trong report chứ
  không im lặng.
* `W2-08` giờ chạy được: 14 ô của `make exp` + `make eval-compare-by` cho `p`/CI
  từng dòng. ⚠️ `e1-chunk550-dense` không so được recall/nDCG/MAP với 13 ô kia.
* `TD-19` (báo cáo không nói được nó đo trên golden set nào) — trước `TD-13`.
* ⚠️ Nhóm `table_lookup` (4 câu) sẽ mãi là `KHÔNG ĐỦ LỰC`. Nó thuộc `TD-16` (nhóm
  thiếu đa dạng) mở rộng: golden set cần thêm câu `table_lookup` mới đo được nhóm
  đó, và đó là việc của `TD-13`/nguồn (b)(c).
