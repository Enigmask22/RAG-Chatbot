# `W2-09` — exp-001-retrieval: tổng kết, và câu hỏi của DoD không có câu trả lời

**Ngày:** 2026-08-22 · **Nhánh:** `feat/w1-foundation` · **Hạng mục cuối của `W2`**

DoD: *bảng delta vs baseline + ít nhất 2 nhận xét về category nào cải thiện nhiều
nhất.* Bảng delta có ngay §1. Nhận xét có §4. Nhưng cái đáng nhất của hạng mục này
là **câu hỏi thứ hai của DoD không trả lời được trên `golden_v1`**, và phải dựng
một công cụ mới mới biết là không trả lời được.

---

## 0. Dự đoán ghi trước khi chạy số: 4/7

| # | dự đoán | kết quả |
|---|---|---|
| D1 | Delta tổng đạt ý nghĩa **15/15** metric | ✅ 15/15, `p` từ 1,98e-22 tới 5,88e-39 |
| D2 | `cross_lingual` tăng nhiều nhất về `ndcg@10` | ✅ +0,5875 — **nhưng** `ndcg@10` không xếp hạng được giữa nhóm (§3) |
| D3 | `table_lookup` vẫn `KHÔNG ĐỦ LỰC` kể cả với delta khổng lồ | ✅ Δ = +0,5000 mà trần `p` = 0,5 |
| D4 | `adversarial` tăng ít nhất | ❌ `multi_hop` mới là ít nhất; `adversarial` hạng 3/6 |
| D5 | Hiệu chỉnh 15 metric đổi **≤ 5 hàng** | ❌ **33 hàng / 6 file** ở `m=15`, 26 hàng ở `m=7` |
| D6 | Tự nâng `B` đổi ít nhất 1 kết luận → phải nhận | ✅ đổi — **nhưng theo hướng ngược** với điều tôi định làm (§6) |
| D7 | `factoid` phần lớn nhờ reranker, `cross_lingual` phần lớn nhờ model | ❌ **model** thắng ở cả 6 category (§5) |

Ba lần sai lần này **không** cùng một hướng như `W2-08` (ở đó cả bốn là đánh giá
quá cao độ phân giải). D5 và D7 sai vì tôi suy từ cấu trúc mà không đo cấu trúc:
tôi cho rằng 15 metric gần độc lập và rằng reranker là bậc lớn nhất, cả hai đều
kiểm được bằng một phép đo mà tôi chưa từng chạy.

---

## 1. Bảng delta: `e1-baseline-dense` → `e1-rr-bgem3-reranked-onhybrid-rc100`

Mốc là ô baseline của chính grid (`vietnamese-bi-encoder` · dense · chunk=1000),
đỉnh bảng là ô thắng của `W2-08`. Cùng 209 câu, cùng bộ nhãn (1,3828 nhãn/câu cả
hai bên) nên `compare.py` không từ chối metric nào.

| metric | baseline | đỉnh bảng | Δ | đếm câu | kết luận |
|---|---:|---:|---:|---|---|
| `hit_rate@1` | 0,1196 | 0,5789 | **+0,4593** | 8↔104 | khác biệt thật |
| `hit_rate@5` | 0,2153 | 0,7895 | **+0,5742** | 0↔120 | khác biệt thật |
| `hit_rate@10` | 0,2775 | 0,8134 | **+0,5359** | 0↔112 | khác biệt thật |
| `hit_rate@20` | 0,3110 | 0,8182 | +0,5072 | 0↔106 | khác biệt thật |
| `mrr` | 0,1660 | 0,6694 | **+0,5034** | 9↔143 | khác biệt thật |
| `ndcg@10` | 0,1621 | 0,6736 | **+0,5115** | 8↔153 | khác biệt thật |
| `map@20` | 0,1349 | 0,6274 | +0,4926 | 10↔153 | khác biệt thật |
| `recall@5` | 0,1746 | 0,7352 | +0,5606 | 1↔133 | khác biệt thật |
| `recall@10` | 0,2257 | 0,7679 | +0,5423 | 0↔128 | khác biệt thật |
| `recall@20` | 0,2663 | 0,7775 | +0,5112 | 1↔120 | khác biệt thật |
| `recall@1` | 0,0877 | 0,4665 | +0,3788 | 8↔104 | khác biệt thật |
| `precision@1` | 0,1196 | 0,5789 | +0,4593 | 8↔104 | khác biệt thật |
| `precision@5` | 0,0459 | 0,1990 | +0,1531 | 1↔133 | khác biệt thật |
| `precision@10` | 0,0306 | 0,1048 | +0,0742 | 0↔128 | khác biệt thật |
| `precision@20` | 0,0187 | 0,0531 | +0,0344 | 1↔120 | khác biệt thật |

**15/15 có ý nghĩa.** Đây là bảng duy nhất trong cả `W2` mà hiệu chỉnh đa so sánh
không cần bàn: `p` nhỏ nhất là 5,88e-39, chia cho 15 hay cho 90 đều không chạm tới.

⭐ Con số đáng nhớ: `hit_rate@5` **0↔120** — 120 câu được sửa, **0 câu bị làm
hỏng**. Trên 209 câu thì đó là 57% bộ đo đổi từ trượt sang trúng mà không mất câu
nào. `ndcg@10` **0,1621 → 0,6736** là **4,2×**.

Evidence: [`compare/cmp-e1-baseline-dense-vs-e1-rr-bgem3-reranked-onhybrid-rc100.md`](../compare/cmp-e1-baseline-dense-vs-e1-rr-bgem3-reranked-onhybrid-rc100.md)

---

## 2. ⭐⭐ "Category nào cải thiện nhiều nhất" — không nhóm nào, và đó là kết quả

Bảng chia nhóm của `W2-08-prep` cho `Δ` từng nhóm kèm kiểm định `Δ ≠ 0`. Nó **không**
kiểm `Δ_A > Δ_B`, mà đó mới là câu DoD hỏi. Không có hàng nào trong bất kỳ bảng nào
đã công bố trả lời câu ấy.

Nên `W2-09` dựng `pipeline/eval/contrast.py`. Xếp 6 nhóm theo `Δ`, rồi kiểm từng
nhóm còn lại có phân biệt được với đỉnh bảng không — bootstrap **không cặp**, vì
hai nhóm là hai tập câu rời nhau và không có cặp nào để ghép.

### `hit_rate@5`, `e1-baseline-dense` → đỉnh bảng

| hạng | nhóm | n | Δ | tốt↔xấu | vs đỉnh bảng |
|---:|---|---:|---:|---|---|
| 1 | `cross_lingual` | 43 | **+0,6512** | 28↔0 | *(đỉnh bảng)* |
| 2 | `aggregation` | 26 | +0,6154 | 16↔0 | CI99% [−0,2728, +0,3444] → không phân biệt được |
| 3 | `adversarial` | 34 | +0,5882 | 20↔0 | CI99% [−0,2237, +0,3447] → không phân biệt được |
| 4 | `factoid` | 68 | +0,5441 | 37↔0 | CI99% [−0,1416, +0,3434] → không phân biệt được |
| 5 | `table_lookup` | 4 | +0,5000 | 2↔0 | CI99% [−0,4419, +0,7442] → không phân biệt được |
| 6 | `multi_hop` | 34 | +0,5000 | 17↔0 | CI99% [−0,1306, +0,4378] → không phân biệt được |

**Tập "cải thiện nhiều nhất" = cả 6 nhóm.** 0/5 phép so phân giải được. Lặp lại
trên `hit_rate@1` và `mrr`: cũng 0/5, cũng cả 6 nhóm.

### ⚠️ Đối chứng bắt buộc: đây là chuyện của dữ liệu, không phải của Bonferroni

Nếu "hoà 6/6" chỉ là hệ quả của ngưỡng đã hiệu chỉnh thì nó là kết luận của công
cụ, không phải của bộ đo. Nên đo lại ở α **thô** 0,05:

| metric | Bonferroni α=0,01 | thô α=0,05 | khoảng cách lớn nhất | nửa bề rộng CI95 |
|---|---|---|---:|---:|
| `hit_rate@5` | hoà 5/5 | **hoà 5/5** | +0,1512 | 0,2168 |
| `hit_rate@1` | hoà 5/5 | **hoà 5/5** | +0,1293 | 0,2370 |
| `mrr` | hoà 5/5 | **hoà 5/5** | +0,1165 | 0,2030 |

Khoảng cách xa nhất giữa hai category (+0,1512) chỉ bằng **0,70×** nửa bề rộng
khoảng tin cậy **chưa hiệu chỉnh**. Tức nó còn cách ngưỡng phát hiện được một
khoảng rộng, và hiệu chỉnh không dính dáng gì.

💡 **Ngưỡng phân giải giữa hai nhóm: ~±0,22 tuyệt đối** (43 câu vs 34 câu, α=0,05).
So với ~6 điểm của `TD-11` cho toàn bộ 209 câu: **chia 209 câu thành 6 nhóm rồi so
các nhóm với nhau tốn khoảng 3,6× độ phân giải.** Để phân giải được khoảng cách
0,15 hiện tại cần mỗi nhóm to gấp `(0,2168/0,15)² ≈ 2,1×`, tức golden set
**~440 câu**. Đó là một con số dùng được cho `TD-13`.

Evidence: [`compare/contrast-category-exp-001.md`](../compare/contrast-category-exp-001.md) ·
[`compare/contrast-lang-exp-001.md`](../compare/contrast-lang-exp-001.md)

### Chiều `lang` còn nói thêm một chuyện

Chia đôi 209 câu (127 `vi` / 82 `en`) là cách chia cho **nhiều lực nhất** có thể,
và nó vẫn không phân giải được: `hit_rate@5` `en` +0,5854 vs `vi` +0,5669, CI95
[−0,1185, +0,1546]. Và người dẫn đầu **đổi theo metric** — `en` dẫn ở `hit_rate@5`,
`vi` dẫn ở `hit_rate@1` (+0,4724 vs +0,4390). Một cái tên duy nhất cho câu "nhóm nào
nhiều nhất" sẽ phụ thuộc vào metric mà người viết tình cờ chọn.

---

## 3. ⚠️ Hàng rào mới: mẫu số của metric đổi giữa các NHÓM

`compare.py` có ba hàng rào và cả ba canh trục *"hai lần chạy"*. Không cái nào thấy
được điều này:

| category | nhãn/câu |
|---|---:|
| `factoid` | 1,0147 |
| `table_lookup` | 1,0000 |
| `adversarial` | 1,0882 |
| `cross_lingual` | 1,0930 |
| `multi_hop` | 2,0294 |
| `aggregation` | **2,4231** |

`recall@k`, `nDCG@k`, `MAP@k` có mẫu số là số nhãn. Trong **một** nhóm thì hai lần
chạy dùng đúng bộ nhãn ấy, nên `Δ` của nhóm đó hợp lệ và hàng rào băm của `W2-03`
im lặng — đúng, vì nó nhìn trục khác. Nhưng đem `Δ recall@5` của `factoid` **so
với** của `aggregation` là so hai thang đo: sửa một câu ở `factoid` đẩy recall lên
1,0, ở `aggregation` chỉ đẩy lên 0,33.

**Nó lộ ra ngay trong dữ liệu**, không cần lập luận:

| xếp theo | `aggregation` đứng hạng |
|---|---:|
| `hit_rate@5` (0/1, không mẫu số) | **2**/6 |
| `ndcg@10` (mẫu số là số nhãn) | **5**/6 |

Nhóm bị xê dịch nhiều nhất đúng là nhóm nhiều nhãn nhất. Nên `contrast.py` từ chối
6/15 metric khi xếp hạng giữa các nhóm, còn lại **9/15**.

⚠️ **Hệ quả cho `G2`**: `ndcg@10` — metric mà cả `W2` cam kết và là tiêu chí đầu
của gate — **không** trả lời được câu "nhóm nào cải thiện nhiều nhất". Dự đoán D2
của tôi ("`cross_lingual` tăng nhiều nhất về `ndcg@10`") đúng về con số và **sai về
loại câu hỏi**.

💡 Hàng rào để **nhị phân**: lệch 1,00× cũng từ chối như lệch 2,22×. Đặt ngưỡng dung
sai thì phải bịa ra một con số, mà `compare.py` đã từ chối `CARDINALITY_SENSITIVE`
bất kể lệch bao nhiêu — nhất quán với luật cũ đáng hơn. Bù lại bảng **in ra tỉ số
lệch** để người đọc thấy hàng nào là sát nút.

---

## 4. Hai nhận xét mà DoD đòi (và chúng không phải "category nào nhiều nhất")

Câu hỏi gốc không có câu trả lời (§2). Hai nhận xét thay thế, cả hai **có kiểm
định**:

### Nhận xét 1 — `cross_lingual` là nhóm duy nhất có một bậc làm nó TỆ ĐI

Bậc hybrid (`e1-bgem3-dense` → `e1-rrf-bgem3-hybrid-k1`) trên riêng 43 câu
`cross_lingual`, giả thuyết **nêu trước** (`W2-04`/`W2-05` đã nêu tên nhóm này):

| metric | dense | hybrid k=1 | Δ | kiểm định |
|---|---:|---:|---:|---|
| `ndcg@10` | 0,2538 | 0,1707 | **−0,0831** | CI95 [−0,1279, −0,0396] · 17↔1 |
| `mrr` | 0,2028 | 0,1349 | −0,0679 | CI95 [−0,1052, −0,0362] · 19↔1 |
| `map@20` | 0,1963 | 0,1306 | −0,0657 | CI95 [−0,1000, −0,0361] · 19↔1 |
| `recall@5` | 0,3023 | 0,2093 | −0,0930 | CI95 [−0,1860, −0,0233] · 4↔0 |

17–19 câu tệ đi, **1** câu tốt lên, tuỳ metric. Phát hiện của `W2-04` tái lập nguyên vẹn trên bộ
run của grid — và lần này nó là bậc *giữa* của chính đường dẫn tới đỉnh bảng, chứ
không phải một nhánh bên.

### Nhận xét 2 — mọi category đều nhận phần lớn nhất từ **một** bậc, và đó là bậc đổi model

`hit_rate@5`, bốn bậc liền kề (chỉ mô tả, không kiểm định — xem cảnh báo dưới):

| category | model<br>baseline→bge-m3 | hybrid<br>dense→k=1 | rerank<br>k=1→rc20 | pool<br>rc20→rc100 | tổng |
|---|---:|---:|---:|---:|---:|
| `cross_lingual` | **+0,3256** (50%) | −0,0930 (−14%) | +0,1628 (25%) | +0,2558 (39%) | +0,6512 |
| `aggregation` | **+0,5385** (88%) | +0,0769 (12%) | +0,0000 (0%) | +0,0000 (0%) | +0,6154 |
| `adversarial` | **+0,2647** (45%) | +0,0294 (5%) | +0,0882 (15%) | +0,2059 (35%) | +0,5882 |
| `factoid` | **+0,2941** (54%) | +0,0735 (14%) | +0,1324 (24%) | +0,0441 (8%) | +0,5441 |
| `table_lookup` | **+0,2500** (50%) | +0,0000 (0%) | +0,0000 (0%) | +0,2500 (50%) | +0,5000 |
| `multi_hop` | **+0,3235** (65%) | +0,0588 (12%) | +0,0000 (0%) | +0,1176 (24%) | +0,5000 |

⭐ **Bậc đổi embedding model chiếm phần lớn nhất ở cả 6/6 category** (45%–88%). Mọi
thứ sau nó — hybrid, reranker, pool sâu — cộng lại vẫn nhỏ hơn. Đây là kết quả kiến
trúc mạnh nhất của cả `W2`, và nó ngược dự đoán D7 của tôi.

Hai chỗ đáng chú ý trong bảng:

* `aggregation` nhận **0%** từ reranker và **0%** từ pool sâu. Nó có 2,42 nhãn/câu
  nên `hit_rate@5` bão hoà sớm — reranker không còn gì để sửa.
* `cross_lingual` (+39%) và `adversarial` (+35%) là hai nhóm nhận nhiều nhất từ
  **pool sâu**. Đây là chỗ duy nhất trong cả `W2` mà `rc100` kiếm được tiền của
  nó, và nó kiếm theo category — xem §7.

⚠️ **Bảng này mô tả, không suy luận.** 6 category × 4 bậc = 24 `Δ` chưa kiểm định
riêng, và §2 vừa đo được rằng chênh lệch **giữa** các nhóm cỡ 0,15 còn chưa phân
giải được thì chênh lệch giữa các **ô** trong bảng này càng không. Đọc nó để **nêu
giả thuyết** cho `W3`/`W4`, đừng đọc để chốt.

---

## 5. Quyết định (a) — bảng một-cặp có hiệu chỉnh cho 15 metric không? **Không**, và tiền đề của câu hỏi sai

`W2-08` để lại: *"kỳ vọng 0,75 dương giả mỗi bảng; sửa thì viết lại `W2-01`…`W2-07`,
không sửa thì phải nói ra."*

Con số 0,75 tính bằng `15 × 0,05`, tức coi 15 metric là **độc lập**. Chưa ai kiểm
tiền đề đó. Đo trên hiệu từng câu (Li & Ji 2005 trên ma trận tương quan; ngoặc là
tỉ số tham gia `(Σλ)²/Σλ²`):

| cặp | `m` | `n_eff` | `\|r\|` trung bình | trị riêng đầu |
|---|---:|---:|---:|---:|
| `baseline → bgem3` | 15 | 5,0 (1,9) | 0,666 | 10,42 |
| `bgem3 → bgem3-rrf` | 15 | 7,0 (2,9) | 0,454 | 7,72 |
| `bgem3 → bgem3-rr-c50` | 15 | 5,0 (2,3) | 0,583 | 9,25 |
| `rc50 → rc100` | 15 | **4,0** (1,4) | **0,833** | **12,69** |

⭐ **Cặp bị hiệu chỉnh cắn đau nhất đúng là cặp tương quan chặt nhất.** `rc50 → rc100`
có trị riêng đầu 12,69/15 — **một** chiều giải thích 85% phương sai — nên
Bonferroni-15 ở đó là hiệu chỉnh thừa khoảng 3,75×.

**Con số đúng là `7 × 0,05 =` 0,35, không phải 0,75.** Vấn đề nhỏ hơn hai lần so
với lúc phát biểu.

Và nếu hiệu chỉnh thật thì giá phải trả đã đo:

| `family_size` | file đổi | hàng đổi | trong đó `khác biệt thật` → nhiễu |
|---:|---:|---:|---:|
| 15 | 6/12 | **33** | 16 |
| 7 | 6/12 | 26 | 12 |

✅ **Hai chốt kiểm soát không đổi ở cả hai mức**: `baseline → bgem3` (tiêu đề
`W2-01`) và `bgem3 → bgem3-rr-c50` (tiêu đề `W2-05`) giữ nguyên **15/15**. Tức hiệu
chỉnh chỉ cắn vào những cặp hiệu ứng nhỏ mà dự án **đã** đánh dấu là mong manh.

**Quyết định: không hiệu chỉnh bảng một-cặp, nhưng bắt bảng tự khai.** Mọi bảng
không hiệu chỉnh giờ in ngay dòng đầu:

> ⚠️ **15 hàng, KHÔNG hiệu chỉnh đa so sánh** — […] **7** phép kiểm hiệu dụng […]
> nên số hàng "có ý nghĩa" **thuần do ngẫu nhiên** mà bảng này chờ đợi là ≈ **0.35**.
> Đọc **cả bảng** thì được; rút một hàng thuận nhất ra trích thì đó là chỗ con số
> trên biến thành kết luận sai.

💡 Lý do đặt cảnh báo **trong bảng** chứ không trong docstring: `W2-08-prep` đã chọn
không hiệu chỉnh và ghi lựa chọn ấy vào docstring `compare_by_group` — nơi người
đọc bảng không bao giờ tới. Ba lần trích sai đã công bố đều là **rút một hàng thuận
nhất trong 15 hàng ra trích**. Con số cảnh báo phải nằm ở chỗ hành vi ấy xảy ra.

---

## 6. ⭐⭐ Quyết định (b) — tự nâng `B`? Có. Nhưng luật hiển nhiên cho câu trả lời SAI

`W2-08` để lại: *"`MIN_TAIL_RESAMPLES = 30` mới chỉ để **giải thích** cờ, chưa để tự
nâng `B`. Đọc được α = 0,00128 cần ~47.000 mẫu lại (~4,7× chi phí). Đo rồi hãy
quyết."*

Đo bằng thứ thật sự quan trọng — **số thành viên của tập thắng**, chứ không phải
độ rộng của một khoảng:

| `B` | đuôi | tập thắng | ghi chú |
|---:|---:|---:|---|
| 10.000 | 5 | **2 ô** | con số đã công bố ở `W2-08` |
| 50.000 | 31 | **3 ô** | ⚠️ `rc50` lọt vào |
| 200.000 | 127 | **2 ô** | hội tụ |
| 400.000 | 255 | **2 ô** | hội tụ |

⚠️⚠️ **Luật hiển nhiên — "nâng `B` cho tới khi đuôi đủ 30 mẫu" — đưa `B` tới
~48.400, đúng vùng bất ổn, và cho câu trả lời mà `B = 10.000` đã trả lời đúng.**
Nâng nửa vời tệ hơn không nâng: nó đổi kết luận rồi dừng lại giữa đường.

Truy tới hàng gây ra chuyện, `mrr` của `rc50`:

| `B` | CI99,87% | dao động của biên | kết luận |
|---:|---|---|---|
| 10.000 | [−0,0625, −0,0007] | [−0,0009, −0,0005] | khác biệt thật |
| 50.000 | [−0,0638, −0,0002] | **[−0,0004, +0,0002]** | KHÔNG KẾT LUẬN |
| 200.000 | [−0,0640, −0,0002] | [−0,0003, −0,0001] | khác biệt thật |

Biên gần như không nhúc nhích (−0,0007 → −0,0002); thứ đổi là **dải dao động của
chính nó**, và ở 50.000 dải ấy vắt qua 0. Kết luận đi *đúng → không đọc được →
đúng*.

💡 **Chỗ `W2-08` chọn sai hằng số, nói cho chính xác:** nó đo ba mức đuôi (6/32/128)
với tiêu chí *"**dấu** của biên có ổn định không"*, và ở đuôi 32 dấu đúng là ổn
định. Nhưng thứ quyết định một ô nằm rổ nào không phải dấu của biên — mà là **dải
dao động** của nó có chứa 0 hay không. **Hai đại lượng ổn định ở hai tốc độ khác
nhau, và tôi chọn hằng số bằng cái ổn định nhanh hơn.**

**Đã sửa:**

* `MIN_TAIL_RESAMPLES` **30 → 128** (điểm hội tụ đo được, không phải điểm đầu tiên
  trông có vẻ ổn).
* `resolve_iterations(alpha, iterations)` tự nâng `B`, có trần `MAX_BOOTSTRAP =
  200.000`. Chạm trần thì cờ `mc_unstable` vẫn bật và hàng vẫn nói `KHÔNG KẾT LUẬN`
  — hết cách đọc thì nói là chưa đọc được.
* ⚠️ **Chỉ nâng khi người gọi để nguyên mặc định.** Một `B` nêu tường minh là lựa
  chọn có chủ đích, và tự nâng nó làm chính phép quét mất khả năng đo thứ nó đang
  đo — bốn con số trong bảng trên sẽ không đo được nếu hàm giẫm lên chúng.

### Kiểm chứng: đổi đúng một hàng, và hàng đó `W2-08` đã tự đo được rồi

Sinh lại toàn bộ 15 file `compare/` sau khi đổi hằng số: **0/15 file đổi kết luận**.
Bảng chia nhóm thật sự nâng — log ghi **60 lần** `B: 10.000 → 200.000` ở
α = 0,000556, nơi đuôi cũ chỉ có **2 mẫu**. Tức ba phát hiện `cross_lingual` của
`W2-08-prep` (`map@20`/`mrr`/`ndcg@10`) từng đọc từ đuôi 2 mẫu, và chúng **sống sót
nguyên vẹn** ở 200.000.

⚠️ **Bảng ablation thì đổi đúng một hàng** — nó là chỗ `α` hiệu chỉnh mạnh nhất nên
cũng là chỗ đuôi mỏng nhất:

| hàng | B = 10.000 | B = 200.000 |
|---|---|---|
| `ndcg@10` của `rc50` | CI99,87% [−0,0638, **−0,0003**] → `TRÁI CHIỀU` | CI99,87% [−0,0647, **+0,0004**] → `trong ngưỡng nhiễu` |

**Và đây không phải bất ngờ — `W2-08` §5 đã đo được đúng chuyện này** ("ở B=50.000
và 200.000 thì khoảng **chứa 0**") và ghi nó vào report, nhưng bảng mặc định vẫn
chạy ở 10.000 nên nó **mâu thuẫn với phép đo của chính nó**. Giờ hai chỗ khớp nhau.

✅ Kết luận cấu trúc không đổi: tập thắng vẫn **2 ô**, và `rc50` vẫn ở rổ **tranh
chấp** — vì `mrr` của nó vẫn "khác biệt thật". 14/15 hàng còn lại của bảng ablation
giữ nguyên kết luận.

---

## 7. Quyết định (c) — điểm vận hành `rc50` vs `rc100`

Ở `B` đã hội tụ (200.000), `rc50` **không** nằm trong tập tương đương — nó ở rổ
tranh chấp, và cuộc tranh chấp rất mỏng:

| metric chính | Δ (`rc100` − `rc50`) | kết luận ở α = 0,00128 |
|---|---:|---|
| `mrr` | +0,0254 | khác biệt thật (CI [−0,0640, −0,0002] cho chiều ngược) |
| `ndcg@10` | +0,0255 | trong ngưỡng nhiễu |
| `hit_rate@1` | +0,0191 | `KHÔNG ĐỦ LỰC` (trần `p` = 0,125) |

1/3 metric chính nói `rc100` hơn, 1/3 nói nhiễu, 1/3 không có lực. Giá: **608,9 →
1163,9 ms p95 = 1,91×**, tức **+555 ms**.

**Khuyến nghị: lấy `rc50` làm điểm vận hành mặc định của `W4`, và ghi rõ điều kiện
lật lại.** Ba lý do, tất cả đều là số đã đo:

1. Ưu thế của `rc100` là ~2,5 điểm `ndcg@10`, tức **~5 câu** trên 209 — ngay rìa đo
   được kể cả ở 200.000 mẫu lại.
2. `+555 ms` là **16% của toàn bộ ngân sách 3500 ms** end-to-end mà `G2` đặt ra, và
   phần sinh chưa được đo lần nào (`W4-13`).
3. Ưu thế thật của pool sâu là **vùng phủ**, không phải chất lượng xếp hạng
   (`W2-08`: `hit_rate@10` 0↔9 sạch, còn `ndcg@10`/`map@20` `TRÁI CHIỀU`) — và vùng
   phủ chỉ thành tiền nếu **bộ sinh đọc sâu hơn hạng 5**.

⚠️ **Điều kiện lật lại, nêu trước:** §4 đo được pool sâu đóng góp **39%** mức cải
thiện của `cross_lingual` và **35%** của `adversarial`. Nếu `W4` đo được bộ sinh
dùng được ngữ cảnh dưới hạng 5, hoặc nếu sản phẩm ưu tiên hai nhóm đó, thì `rc100`
đáng 1,91× — và lúc ấy quyết định này phải đổi. Phép đo chốt thuộc **`W4-13`**.

---

## 8. Một target Makefile chưa từng chạy được

`make eval-compare-subset CAT=… LANG=…` (thêm ở `W2-08-prep`) hỏng từ ngày viết:

```make
LANG ?=
… $(if $(LANG),--lang $(LANG))
```

`LANG` là **biến môi trường chuẩn** (`en_US.UTF-8`). Make thừa kế nó, nên `?=`
không bao giờ có tác dụng và `--lang` luôn nhận một chuỗi locale. Đã đổi thành
`QLANG`.

💡 Nó chỉ lộ ra ở `W2-09` vì đây là lần đầu target ấy được gọi thật. Một target
viết xong, xanh trong `make help`, và **chưa bao giờ chạy** — cùng họ với
`make eval EXP=exp_001` của `G2` (một target chưa bao giờ tồn tại, tìm ra ở
`W2-07`). Bài học: một dòng trong Makefile không phải bằng chứng nó chạy được.

---

## 9. Kết luận

1. **Đường từ baseline tới đỉnh bảng là 15/15 metric, `hit_rate@5` 0↔120 câu.**
   `ndcg@10` ×4,2. Không có gì mơ hồ ở tầng này.
2. ⭐⭐ **Câu "category nào cải thiện nhiều nhất" không có câu trả lời trên
   `golden_v1`** — cả 6 nhóm hoà, và hoà kể cả khi bỏ hiệu chỉnh. Cần ~440 câu mới
   phân giải được khoảng cách hiện tại.
3. ⭐ **Bậc đổi embedding model chiếm phần lớn nhất ở cả 6/6 category** (45–88%).
4. ⚠️ **`ndcg@10` không dùng để xếp hạng giữa các nhóm được** — mẫu số là số nhãn,
   mà `aggregation` có 2,42 nhãn/câu còn `factoid` có 1,01.
5. **Hiệu chỉnh 15 metric: không**, vì `n_eff` đo được là 4–7 chứ không phải 15 —
   nhưng mọi bảng giờ tự in ngân sách dương giả của nó.
6. ⭐⭐ **`MIN_TAIL_RESAMPLES` 30 → 128**, vì luật nâng `B` nửa vời cho câu trả lời
   sai mà `B` mặc định đã trả lời đúng.
7. **`rc50` là điểm vận hành đề xuất**, điều kiện lật lại đã nêu trước, chốt ở
   `W4-13`.

### Việc còn lại lộ ra từ đây

* **`TD-21` (mới)** — `contrast.py` mới có chiều `category`/`lang`. Chiều đáng giá
  nhất cho `W4` là **theo document/tenant**, mà `RunScores` chưa mang trường đó.
* **`TD-20`** — chiều `chunk_size` vẫn không có phép kiểm nào (`W2-08`).
* **`TD-13`** — con số **~440 câu** ở §2 là mục tiêu định lượng đầu tiên cho việc mở
  rộng golden set. Trước đó `TD-19` (một `runs/*.json` phải nói được nó đo trên
  golden set nào).
* **`W4-13`** — chốt `rc50` vs `rc100` cần phép đo end-to-end có bộ sinh.
