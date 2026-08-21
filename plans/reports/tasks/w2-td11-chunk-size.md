# `TD-11` — sửa truncation bằng cách hạ `chunk_size`: **không cải thiện gì đo được**

> Việc đầu tiên của W2. Kết luận đi trước:
>
> 1. Phép sửa **hoạt động đúng như thiết kế**: 56,9% → **0,4%** chunk bị cắt.
> 2. Truy hồi **không cải thiện**, và mức tụt trong bảng số hoá ra là **hai loại
>    nhiễu cộng lại** — nhiễu đo (mẫu số của recall đổi) và nhiễu thống kê (chênh
>    3 câu trên 209). Kiểm định McNemar: `p = 0,711`.
> 3. Phát hiện đáng giá nhất lại không phải về `chunk_size` mà về **thước đo**:
>    `golden_v1` chỉ phân giải được mức chênh **≥ 6 điểm `hit_rate`**. Mọi so
>    sánh mịn hơn thế trong W2 sẽ là tung đồng xu nếu không có kiểm định.
>
> `TD-11` giả định truncation là thứ đang kéo recall xuống. Giả định đó sai, và
> báo cáo giữ nguyên đường đi tới chỗ biết là nó sai.

- Ngày chạy: 2026-08-20
- Golden set: `golden_v1` (242 câu, chấm 209 — 33 câu `unanswerable` đo riêng ở `W5-02`)
- Model embedding: `vietnamese-bi-encoder` ở **cả ba** cấu hình. Không đổi model ở bước này.
- Evidence: `truncation-{baseline,chunk550}.json` · `reports/probes/index-chunk550.json` ·
  `{baseline,chunk550,chunk550nb55}-retrieval.{md,json}`

---

## 1. Trước hết: biến truncation thành thứ đo được

`sentence-transformers` cắt input ở `max_seq_length` **không cảnh báo, không lỗi**.
Đây là lý do `TD-11` sống được suốt W1: index đủ chunk, đủ chiều, mọi số trong
`BuildReport` đều đẹp, chỉ có phần đuôi văn bản là không tồn tại đối với vector.

Nên việc đầu tiên không phải hạ `chunk_size` mà là **tự tạo triệu chứng cho lỗi**:

| Thêm vào | Ở đâu | Làm gì |
|---|---|---|
| `EmbeddingProvider.max_sequence_tokens` | `rag_core/embedding/base.py` | Giới hạn cửa sổ; `None` = **không biết**, khác "không giới hạn" |
| `EmbeddingProvider.count_tokens` | như trên | Đếm bằng tokenizer **thật** của model, kèm special token |
| `rag_core/embedding/truncation.py` | mới | `TruncationStats` — phần thuần số học, test được không cần `torch` |
| `pipeline/indexing/truncation_report.py` | mới | `make truncation BUNDLE=x` — đo trên corpus thật, chia theo ngôn ngữ |
| `BuildReport.truncation` + `_log_truncation()` | `build_index.py` | **Mọi** lần build đều mang con số này, và in ở mức `WARNING` |

Ba chi tiết dễ làm phép đo sai, cả ba đều có test canh:

1. **`truncation=False` khi gọi tokenizer.** Mặc định tokenizer *cắt ở
   `model_max_length`* — tức nó trả về đúng con số ngưỡng cho mọi text dài, và
   "% bị cắt" trở thành hằng số 0. Phép đo sẽ báo "đã sửa xong" ở mọi cấu hình.
2. **Đếm kèm `[CLS]`/`[SEP]`.** Hai token đó chiếm chỗ thật trong cửa sổ 256.
3. **Text dài **đúng bằng** giới hạn thì KHÔNG bị cắt** (`>` chứ không `>=`).

Và hai đại lượng phải tách rời, không gộp:

- `truncated_ratio` — **bao nhiêu chunk** bị cắt.
- `tokens_lost_ratio` — **bao nhiêu nội dung** không tới được vector.

90% chunk bị cắt mất mỗi chunk 1 token là chuyện nhỏ; 10% chunk mất mỗi chunk một
nửa là mất hẳn nội dung. Một con số duy nhất không phân biệt được hai ca đó.

## 2. Baseline mất bao nhiêu, và ở đâu

`make truncation BUNDLE=baseline` — 15.814 chunk, cửa sổ 256 token:

| | chunk | bị cắt | token mất | token/ký tự |
|---|---:|---:|---:|---:|
| **tổng** | 15.814 | **56,9%** | **15,4%** | — |
| `en` | 9.393 | 65,3% | 19,4% | 0,244 |
| `vi` | 6.421 | 44,7% | 7,5% | 0,188 |

Phân bố token/chunk: p50 **274** · p95 375 · max 684 — tức **quá nửa số chunk đã
vượt ngưỡng ngay ở trung vị**.

Con số đáng chú ý nhất không phải 56,9% mà là **chênh lệch EN/VI**: tài liệu tiếng
Anh mất 19,4% token, gấp **2,6 lần** tiếng Việt. Nguyên nhân là tokenizer: PhoBERT
học trên tiếng Việt nên xé chữ tiếng Anh vụn hơn (0,244 vs 0,188 token/ký tự,
+30%). Điều này **cộng dồn** với thứ baseline đã đo được: nhóm `cross_lingual`
recall@5 = 0,0000. Câu hỏi tiếng Anh, tài liệu tiếng Việt, và riêng phía tiếng Anh
còn mất thêm phần đuôi.

*(Phép đo này thay con số lấy mẫu 1.200 chunk ở `W1-11` — 56,8% / 15,7%. Giờ chạy
trên toàn bộ 15.814 chunk: 56,9% / 15,4%. Sai lệch nằm trong khoảng lấy mẫu.)*

## 3. Hiệu chuẩn `chunk_size` — và một công thức sai trông rất hợp lý

`chunk_size` tính bằng **ký tự**, cửa sổ model tính bằng **token**, tỉ lệ quy đổi
phụ thuộc ngôn ngữ. Nên nó phải được **đo**.

Bản đầu tôi viết công thức dùng token/ký tự **trung bình**. Nó cho ra:

```
overall 946 ký tự · en 839 · vi 1152
```

Con số này **sai**, và sai theo hướng rất dễ tin: ở `chunk_size=1000` đã có 56,9%
chunk bị cắt, thì không thể nào hạ xuống 946 lại hết cắt. Lỗi là dùng trung bình —
nó trả lời "`chunk_size` nào làm chunk *trung bình* vừa khít cửa sổ", tức đúng
ngưỡng mà **một nửa** số chunk vẫn bị cắt.

Công thức đúng lấy **phân vị 5% của mật độ ký tự/token** (trường hợp text bị xé
vụn nhất), rồi trừ chỗ của special token và **hai lần** `neighbor_context_chars`
(ngữ cảnh dán cả trước và sau):

```
overall 582 ký tự · en 574 · vi 728  →  chọn 574 (an toàn cho mọi ngôn ngữ)
```

Con số này gần khít với "~600 ký tự" mà plan đoán — nhưng giờ nó là số **đo được**,
và có test hồi quy cho đúng cái bug trung bình/phân vị ở trên.

Chốt `chunk_size: 550` (574 trừ đệm). `max_chunk_size` hạ theo 1500 → 800: nó là
**cùng một đại lượng**, để nguyên thì hậu xử lý vẫn thả ra chunk 1500 ký tự và
chúng vẫn bị cắt.

## 4. Phép sửa hoạt động

| | baseline | chunk550 |
|---|---:|---:|
| chunk trong index | 15.814 | **31.155** (1,97×) |
| chunk bị cắt | 56,9% | **0,4%** |
| token không tới được vector | 15,4% | **0,1%** |
| token/chunk p50 | 274 | 171 |
| hệ số phình `neighbor_context` | 1,24× | **1,48×** |

Log build in ra đúng như thiết kế:

```
⚠️ cắt token     109/31155 chunk vượt 256 token (0.4%) · 0.1% token không tới được vector
```

## 5. Bảng số nói "tệ hơn" — và phần lớn là nhiễu

`make eval-retrieval BUNDLE=chunk550`, cùng 209 câu, nhãn được ánh xạ lại từ span:

| metric | baseline | chunk550 | Δ |
|---|---:|---:|---:|
| recall@5 | 0,1746 | 0,1295 | **−25,8%** |
| recall@20 | 0,2663 | 0,1919 | −28,0% |
| nDCG@10 | 0,1621 | 0,1215 | −25,1% |
| MAP@20 | 0,1349 | 0,0921 | −31,7% |
| MRR | 0,1660 | 0,1414 | −14,8% |
| hit_rate@1 | 0,1196 | 0,0861 | −28,0% |
| hit_rate@5 | 0,2153 | 0,2010 | **−6,7%** |
| precision@5 | 0,0459 | 0,0431 | −6,3% |
| p95 độ trễ | 39,9 ms | 42,6 ms | +6,8% |

### 5.1 Phần lớn mức tụt của recall là NHIỄU ĐO, không phải chất lượng

Nhãn của golden set neo theo **span ký tự** (`TD-12`), và eval ánh xạ span sang
`chunk_id` của index đang đo. Chunk nhỏ hơn thì một span phủ nhiều chunk hơn:

| | nhãn/câu trung bình | tổng nhãn trên 209 câu |
|---|---:|---:|
| baseline | 1,38 | 289 |
| chunk550 | **1,96** | 410 |

`recall@k = |tìm được ∩ liên quan| / |liên quan|`. Mẫu số phình 42% thì **kể cả khi
chất lượng truy hồi không đổi**, recall vẫn tụt `1 − 1,38/1,96 = 29,6%`. Thực tế
recall@5 tụt 25,8% — **gần trùng khít**.

Nên recall@k, nDCG@10 và MAP@20 **không so được trực tiếp giữa hai `chunk_size`**.
Các metric miễn nhiễm với mẫu số đó:

- `hit_rate@k` — "có ít nhất một chunk đúng trong top-k", mẫu số là 1.
- `precision@k` — mẫu số là `k`, cố định.
- `MRR` — thứ hạng của chunk đúng đầu tiên.

Đọc theo ba metric đó, mức tụt còn lại là **−7%** ở k=5, −14,8% ở MRR, −28% ở k=1
— nhỏ hơn hẳn con số −26% mà bảng recall gợi ra. Nhưng ngay cả phần này cũng chưa
được kết luận: −7% trên 209 câu là **3 câu**. Xem §8.

> ⚠️ Hệ quả cho `W2-08`: ma trận ablation có chiều `chunk_size` **bắt buộc** phải
> lấy `hit_rate@k` / `MRR` làm metric so sánh chính. Xếp hạng 12 tổ hợp theo
> recall@10 sẽ tự động thưởng cho cấu hình chunk to, vì chúng có ít nhãn hơn.

### 5.2 Chiều đơn điệu: ngữ cảnh thật sự đi vào mỗi vector

Đếm số ký tự **thật sự đi vào mỗi vector** — tức sau khi đã trừ phần bị cắt:

| | chunk | ký tự/chunk | token model đọc được | **ký tự tới được vector** |
|---|---:|---:|---:|---:|
| baseline | 15.814 | 1.122 | 210 | **950** |
| chunk550 | 31.155 | 679 | 156 | **678** |

Đây là chỗ mà cả `TD-11` lẫn plan đều suy luận sai. Baseline **bị cắt**, nhưng
mỗi vector của nó vẫn đọc ~950 ký tự. chunk550 **không bị cắt**, nhưng mỗi vector
chỉ đọc 678 ký tự — **ít hơn 29%**.

Tức "sửa truncation bằng cách hạ `chunk_size`" không phải là thu hồi 15,4% nội
dung đã mất. Nó là **đánh đổi**: từ "một số chunk mất phần đuôi" sang "mọi chunk
đều ngắn hơn". Tổng nội dung trong index tăng lên, nhưng ngữ cảnh trên mỗi vector
giảm đi — và số ứng viên cho cùng top-20 thì tăng gấp đôi (15.814 → 31.155).

Đây là **lời giải thích khớp với dữ liệu**, không phải nguyên nhân đã chứng minh:
§8 cho thấy mức chênh nằm dưới ngưỡng phân giải của golden set. Cái đứng được là
**chiều** — ba cấu hình xếp đúng theo số ký tự mỗi vector (§6) — chứ không phải
độ lớn.

## 6. Đối chứng: `neighbor_context_chars` phải tỉ lệ, không tuyệt đối

`chunk550` giữ `neighbor_context_chars: 100` "để chỉ đổi một thứ". Đó là **lỗi
thiết kế thí nghiệm của tôi**: giữ nguyên giá trị *tuyệt đối* đã làm đổi giá trị
*tương đối*, mà cái quyết định chất lượng vector là tương đối.

| | ngữ cảnh hàng xóm / chunk | hệ số phình đo được |
|---|---:|---:|
| baseline | 200/1000 = **20%** | 1,24× |
| chunk550 | 200/550 = **36%** | 1,48× |
| chunk550nb55 | 110/550 = **20%** | *(xem dưới)* |

Tức ở `chunk550`, **hơn một phần ba** mỗi vector là text của chunk bên cạnh. Điều
đó vừa pha loãng tín hiệu riêng của chunk, vừa làm các chunk cạnh nhau giống nhau
hơn. Nên phải có một lượt đối chứng với `neighbor_context_chars: 55` (110/550 =
20%, khớp tỉ lệ baseline).

`hit_rate@5` của ba cấu hình:

| | ngữ cảnh/chunk | ký tự tới được vector | hit_rate@5 |
|---|---:|---:|---:|
| baseline (`chunk_size` 1000, nb 100) | 20% | **950** | **0,2153** |
| chunk550 (nb 100) | 36% | 678 | 0,2010 |
| chunk550nb55 (nb 55) | 20% | 589 | 0,1770 |

**Giả thuyết của tôi không được số liệu ủng hộ.** Nếu `chunk550` tụt vì pha loãng
ngữ cảnh hàng xóm thì `chunk550nb55` phải lấy lại được phần đó — nó đi tiếp xuống.
(Mức chênh này cũng dưới ngưỡng ý nghĩa, `p = 0,359` — nên câu đúng là "không có
dấu hiệu nào cho thấy pha loãng là nguyên nhân", chứ không phải "giảm ngữ cảnh
hàng xóm làm tệ hơn".)

Cái còn lại giải thích được cả ba: **số ký tự thật sự đi vào mỗi vector**, và nó
đơn điệu — 950 → 678 → 589 ký tự, `hit_rate@5` 0,2153 → 0,2010 → 0,1770. Kể cả
khi một phần số ký tự đó là text của chunk bên cạnh, và kể cả khi baseline đang
bị cắt mất phần đuôi.

*(Ghi lại để không quên: `chunk550nb55` sinh ra vì tôi cho rằng thiết kế thí
nghiệm của `chunk550` có lỗi — giữ `neighbor_context_chars` tuyệt đối làm đổi giá
trị tương đối. Lập luận đó vẫn đúng; kết luận rút ra từ nó thì sai. Đo vẫn rẻ hơn
suy luận.)*

## 7. Phát hiện phụ: 1 tài liệu dùng `Ê` làm dấu cách

Trong lúc truy nguyên 109 chunk còn bị cắt ở `chunk550`, một tài liệu lộ ra:
`wb-099553007092621441` (*Điểm lại*, bản VI mới nhất) có **5,7% ký tự là `Ê`
(U+00CA)** — font PDF map glyph dấu cách thành `Ê`, nên văn bản **không có ranh
giới từ nào**.

Hệ quả xếp thành chuỗi: splitter không cắt được theo `" "` nên rơi xuống mức ký
tự → tokenizer nổ ra **0,63 token/ký tự** (bình thường 0,20) → 31/36 chunk của
tài liệu này vẫn bị cắt kể cả ở `chunk_size=550` → embedding của nó về cơ bản là
rác.

Phạm vi đã đo và nó hẹp: **1/60 tài liệu**, 36/31.155 chunk, **1/242 câu** golden
set (`factoid-caa304ba55`). Đã ghi `TD-17`, không sửa ở đây — chữa bằng cách hạ
`chunk_size` là chữa triệu chứng; chỗ sửa đúng là tầng đọc tài liệu (`W3-01`).

Cách phát hiện rẻ và nên thành cổng chất lượng corpus: **tỉ lệ ký tự whitespace**.
Corpus hiện tại p50 = 37,5%, thấp nhất 15,2%; tài liệu này 16,4% với 5,7% là `Ê`
giả dạng. Phép kiểm này bắt được cả họ lỗi "PDF trích ra chữ nhưng mất ranh giới".

## 8. Kiểm định: cả ba cấu hình KHÔNG phân biệt được

Đến đây bảng số nói "tụt 7%" và tôi gần như đã viết đúng câu đó vào báo cáo.
Nhưng 0,2153 → 0,2010 trên 209 câu là **45 câu xuống 42 câu** — chênh **3 câu**.
Nên phải có kiểm định trước khi kết luận, và eval chưa lưu điểm từng câu.

Đã thêm (xem §9). Kết quả `make eval-compare`:

| so sánh | `hit_rate@5` | câu đổi chiều | p (McNemar exact) | kết luận |
|---|---:|---|---:|---|
| baseline → chunk550 | 0,2153 → 0,2010 | 16↔13 | **0,711** | trong ngưỡng nhiễu |
| baseline → chunk550nb55 | 0,2153 → 0,1770 | 21↔13 | **0,229** | trong ngưỡng nhiễu |
| chunk550 → chunk550nb55 | 0,2010 → 0,1770 | 12↔7 | **0,359** | trong ngưỡng nhiễu |

`MRR` và `precision@k` (bootstrap cặp, 10.000 lượt, seed cố định) đều cho khoảng
tin cậy 95% **chứa 0**. Ví dụ MRR baseline → chunk550: `CI95 [−0,0624, +0,0126]`.

**Kết luận đúng: không có bằng chứng nào cho thấy ba cấu hình khác nhau.** Không
phải "hạ `chunk_size` làm tụt 7%". Mức tụt 26% của recall là nhiễu đo (§5.1),
và phần còn lại là nhiễu thống kê.

### 8.1 Phát hiện quan trọng nhất của cả bước này: độ phân giải của thước đo

Vì mọi so sánh đều ra "nhiễu", câu hỏi đổi thành: `golden_v1` **đo được** mức
chênh nhỏ nhất là bao nhiêu?

| câu đổi chiều | cần lệch tối thiểu | tương đương chênh `hit_rate` |
|---:|---:|---:|
| 10 | 8 câu | 0,038 |
| 20 | 10 câu | 0,048 |
| 30 | 12 câu | **0,057** |
| 40 | 14 câu | 0,067 |

Với 209 câu và `hit_rate@5` quanh 0,20, phải chênh khoảng **6 điểm tuyệt đối
(≈ 28% tương đối)** mới phát hiện được. Mọi thứ nhỏ hơn là không đo được — bất
kể bảng số trông chênh bao nhiêu phần trăm.

Hệ quả trực tiếp cho phần còn lại của W2:

- **`W2-08` không thể xếp hạng 12 cấu hình bằng mức chênh vài phần trăm.** Mỗi
  dòng ablation phải kèm `p` hoặc CI, và chỉ được tuyên bố người thắng khi kiểm
  định nói vậy. Nếu không thì cái "thắng" chỉ là cái may.
- **Ngưỡng `G2` (+0,08 nDCG tuyệt đối trên 0,1621, tức +49% tương đối) thì nằm
  trong tầm đo.** Gate đặt đủ cao — tình cờ, nhưng đúng.
- Muốn phân giải mịn hơn thì phải **thêm câu hỏi**, không phải thêm cấu hình.
  Việc này gộp được với `TD-13` (người review lại golden set).

## 9. Thêm vào hạ tầng eval

Bước này thêm hai thứ mà `W2-08`/`W2-09` không thể thiếu:

1. **`{run}-per-query.jsonl`** — điểm từng câu của mỗi lần chạy. Không có nó thì
   không kiểm định cặp được. `QueryScore` mang cả `n_relevant` vì đó là biến gây
   nhiễu ở §5.1.
2. **`pipeline/eval/compare.py`** (`make eval-compare BASE=x CAND=y`) — McNemar
   exact cho metric nhị phân, bootstrap cặp cho metric liên tục, và **hai hàng
   rào**:
   - **Từ chối so `recall@k`/`nDCG@k`/`MAP@k` khi số nhãn mỗi câu khác nhau**,
     kèm luôn con số "metric này tụt X% ngay cả khi truy hồi y nguyên". Đây là
     lỗi hôm nay, đóng thành hàng rào để không mắc lại.
   - Chỉ so trên **tập truy vấn giao nhau**, và cảnh báo nếu hai lần chạy khác
     tập câu — so 209 câu với 200 câu rồi kết luận là tự chọn mẫu.

   Cả hai kiểm định dùng thư viện chuẩn, không cần `scipy`. Seed cố định.

Kèm 66 unit test mới (`test_truncation.py` 38 · `test_eval_compare.py` 28).
`mcnemar_exact` được so với giá trị tính tay từ phân bố nhị thức, không gọi lại
chính công thức đang test.

## 10. Kết luận và bước tiếp

**`TD-11` đã trả xong phần đo, và giả thuyết của nó bị phản chứng.**

Truncation là thật (15,4% token của baseline không tới được vector, EN mất 19,4%).
Nhưng **sửa nó bằng cách hạ `chunk_size` không cải thiện gì đo được**, vì đó là
một đánh đổi chứ không phải một phép thu hồi: bớt cắt phần đuôi nhưng mọi vector
đều nhận ít ngữ cảnh hơn (950 → 678 ký tự). Và với bi-encoder này, ngữ cảnh mỗi
vector là chiều đơn điệu.

Cái vẫn còn giá trị:

- Một dấu hiệu **theo ngôn ngữ** đi đúng cơ chế: EN (bị cắt nặng nhất, 19,4%)
  `hit_rate@5` 0,1707 → 0,2073; VI (bị cắt nhẹ nhất, 7,5%) 0,2441 → 0,1969.
  Không cái nào đạt ngưỡng ý nghĩa riêng lẻ, nhưng chúng ngược chiều nhau đúng
  như dự đoán và triệt tiêu nhau ở mức tổng (VI là 127/209 câu = 61% tập đo).
- `cross_lingual` lần đầu khác 0: 0,0000 → 0,0233 → 0,0465 (0 → 1 → 2 câu / 43).

Cả hai chỉ về **cùng một chỗ**: cần model đa ngữ, cửa sổ dài. Đó là `W2-01`
(**BGE-M3**, cửa sổ 8192 token) — nó xoá truncation **mà không** phải hạ
`chunk_size`, tức lấy phần lợi mà không trả phần ngữ cảnh. Đây giờ là kết luận
**có số đỡ**, không còn là thứ tự trong plan.

Việc tiếp theo, theo thứ tự:

1. **`W2-01` BGE-M3** — giữ `chunk_size=1000` của baseline, chỉ đổi model. Đo
   truncation trước để xác nhận cửa sổ 8192 đưa nó về 0.
2. Không tiếp tục quét `chunk_size` với `vietnamese-bi-encoder`: hai điểm đo cho
   thấy chiều đơn điệu và mọi mức chênh đều dưới ngưỡng phân giải.
3. `TD-17` — 1 tài liệu dùng `Ê` làm dấu cách (§7), sửa ở `W3-01`.

### Tái lập

```bash
make truncation BUNDLE=baseline      # 56,9% chunk bị cắt · 15,4% token mất
make truncation BUNDLE=chunk550      # 0,4% · 0,1%
make index BUNDLE=chunk550           # ~257s trên RTX 4060 (31.155 chunk)
make eval-retrieval BUNDLE=chunk550  # ~20s
make eval-compare BASE=baseline CAND=chunk550
```

Collection `rag_chunk550` và `rag_chunk550nb55` vẫn nằm trong Qdrant để đối
chiếu lại; xoá được bằng `make down-clean` rồi build lại từ config.

*(Lượt chạy lại `baseline` sau khi thêm per-query cho **đúng từng chữ số** trên
cả 15 metric — thêm một lần xác nhận tính xác định.)*
