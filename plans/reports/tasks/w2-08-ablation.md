# `W2-08` — Ablation 14 tổ hợp, và câu "cấu hình nào thắng" hỏi sai ở đâu

> **DoD:** ≥ 12 tổ hợp có kết quả đầy đủ, xác định được cấu hình thắng **kèm `p`/CI
> cho từng dòng**. Evidence: MLflow run IDs.
>
> **Trạng thái: đạt**, và câu trả lời cho "cấu hình thắng" là một **tập** chứ không
> phải một dòng — vì hỏi dưới dạng một dòng là hỏi sai.
>
> Sinh lại: `make ablation` · Bảng:
> [`compare/ablation-exp-001-ndcg.md`](../compare/ablation-exp-001-ndcg.md) ·
> Dữ liệu: `runs/e1-*` (14 ô, `W2-07`) + MLflow `sqlite:///mlflow.db`

---

## 1. Dự đoán ghi TRƯỚC khi đo

| # | Dự đoán | Kết quả |
|---|---|---|
| `D1` | Cả 13 ô đều thắng baseline; hiệu chỉnh đa so sánh không đổi hàng nào | ✅ **đúng** — `p ~ 1e-20`, chia α cho 39 không lay chuyển gì |
| `D2` | Tập tương đương có ≥ 3 thành viên (hai `rc100` + `onhybrid-rc50`) | ❌ **sai** — tập **chặt** chỉ 2 ô; `rc50` rơi vào rổ tranh chấp |
| `D3` | `rc20` phân biệt được với `rc100` | ✅ **đúng**, và mạnh hơn tôi tưởng (`hit_rate@5` 0↔18) |
| `D4` | Nền dense vs hybrid không phân biệt được ở **cả ba** độ sâu pool | ✅ **đúng** — 0/24 hàng đạt ý nghĩa |
| `D5` | Cụm RRF không phân biệt được nhau, trừ `k=60` kém dense có ý nghĩa | ⚠️ **nửa sai** — cụm thì đúng, nhưng ở α đã hiệu chỉnh thì `k=60` **cũng** không phân biệt được với dense |
| `D6` | `chunk550` kém baseline **đo được** trên `mrr` | ❌ **sai, và sai kiểu không đoán trước được**: nó không so được **metric nào** |
| `D7` | Tập tương đương chứa một thành viên rẻ hơn đáng kể | ❌ **sai** cho tập chặt (cả hai ô ~1,17 s) |

**4/7 sai, và ba lần sai đều cùng một hướng: tôi đánh giá QUÁ CAO độ phân giải của
209 câu.** `D2`/`D7` cho là phân biệt được ít hơn thực tế công cụ nói, `D6` cho là
so được thứ vốn không so được. Ngược hẳn thiên lệch của `W2-05` (ở đó tôi đánh giá
thấp cả chi phí lẫn lợi ích của cross-encoder).

---

## 2. Bảng 14 ô — DoD "≥ 12 tổ hợp"

Bốn chiều thật, và **`chunk_size` × `embedding` có mặt cả hai** như `TD-11` đòi:
`chunk_size` (1000/550) × embedding (`vietnamese-bi-encoder`/`bge-m3`) × chế độ
truy hồi (dense/sparse/hybrid `k∈{0,1,2,60}`/reranked) × `rerank_candidates`
(20/50/100) × nền rerank (dense/hybrid).

| # | ô | `ndcg@10` | `hit_rate@1` | `mrr` | p95 ms |
|---:|---|---:|---:|---:|---:|
| 1 | `rr-onhybrid-rc100` | **0,6736** | **0,5789** | **0,6694** | 1163,9 |
| 2 | `rr-ondense-rc100` | 0,6624 | 0,5742 | 0,6595 | 1182,3 |
| 3 | `rr-onhybrid-rc50` | 0,6481 | 0,5598 | 0,6440 | 608,9 |
| 4 | `rr-ondense-rc50` | 0,6268 | 0,5455 | 0,6265 | 618,5 |
| 5 | `rr-onhybrid-rc20` | 0,5823 | 0,5263 | 0,5902 | 276,5 |
| 6 | `rr-ondense-rc20` | 0,5676 | 0,5072 | 0,5756 | 267,6 |
| 7 | `rrf-hybrid-k0` | 0,4582 | 0,3493 | 0,4481 | 47,1 |
| 8 | `rrf-hybrid-k1` | 0,4563 | 0,3397 | 0,4436 | 49,1 |
| 9 | `rrf-hybrid-k2` | 0,4521 | 0,3301 | 0,4362 | 46,7 |
| 10 | `bgem3-dense` | 0,4442 | 0,3397 | 0,4394 | 46,3 |
| 11 | `rrf-hybrid-k60` | 0,4313 | 0,3014 | 0,4080 | 48,0 |
| 12 | `bgem3-sparse` | 0,3733 | 0,2919 | 0,3623 | 44,2 |
| 13 | `baseline-dense` | 0,1621 | 0,1196 | 0,1660 | 42,8 |
| 14 | `chunk550-dense` | 0,1215 | 0,0861 | 0,1414 | 46,1 |

**Thứ hạng đổi theo metric**, nên metric xếp hạng phải nêu tường minh (`--rank-by`
là tham số bắt buộc trong tinh thần, có test): `bgem3-dense` là hạng **10** theo
`ndcg@10` nhưng hạng **8** theo `hit_rate@1`, đổi chỗ với `rrf-k1`/`rrf-k2`. Sáu ô
đầu và bốn ô cuối thì hạng giống nhau ở cả ba metric — tức phần bảng *không* ổn
định đúng là phần các ô sát nhau, y như phải thế.

---

## 3. "Cấu hình nào thắng" là một phép chọn cực đại, không phải một phép so

Xếp 14 ô rồi lấy dòng đầu là chọn max trên 14 ước lượng nhiễu: con số của cái
thắng **lệch lên có hệ thống** vì nó thắng một phần vì nó may. Và `TD-11` đã đo
ngưỡng phân giải của `golden_v1` — chênh dưới ~6 điểm tuyệt đối là không phát hiện
được. Ba ô đầu chênh nhau **2,5 điểm**.

Nên câu trả lời là bốn rổ, không một dòng:

| rổ | nghĩa | số ô |
|---|---|---:|
| tập tương đương (chặt) | **không** metric chính nào phân biệt được với đỉnh bảng | **2** |
| tranh chấp | ba metric chính **không đồng ý** | **3** |
| bị đánh bại | **mọi** metric so được đều nói kém | **8** |
| không so được | nhãn khác nhau → từ chối cả 15 metric | **1** |

Họ phép kiểm: 13 ô × 3 metric = **39**, α Bonferroni = **0,00128**. Bảng
so-với-baseline thì **không** hiệu chỉnh (panel giả thuyết nêu trước, ma trận khai
trong `configs/eval/exp-001-retrieval.yaml` trước khi chạy ô nào) — và hiệu chỉnh
ở đó **không đổi hàng nào** vì hiệu ứng cỡ `1e-20`. Rủi ro đa so sánh nằm **toàn
bộ** ở câu hỏi người thắng, nên hiệu chỉnh đúng chỗ đó.

### ⚠️ Lỗi đắt nhất của hạng mục: `KHÔNG SO ĐƯỢC` không phải "hoà"

Bản đầu của `winner_set` định nghĩa tập tương đương là "không bị đánh bại". Ô
`chunk550-dense` có nhãn khác nên **mọi** phép so bị từ chối — và "bị từ chối"
trông y như "không phân biệt được". Nó vào tập thắng. Rồi vì nó là dense thuần
(46,1 ms) nó thành **thành viên rẻ nhất**, tức công cụ đề xuất cấu hình có
`ndcg@10 = 0,1215` thay cho cấu hình có **0,6736** — ô **tệ nhất bảng** làm khuyến
nghị.

💡 Khuôn của lỗi này giống hệt `W2-02` (`ensure_collection` tin thay vì kiểm) và
`W2-06` (`MatchAny(any=[])` cho 0 kết quả mà không báo lỗi): **một nhánh "không có
thông tin" đi vào cùng đường với nhánh "thông tin là bằng nhau".** Ba rổ → bốn rổ,
và có test hồi quy ghim đúng ca 46 ms này.

---

## 4. Ba cờ mới, và cái đã quyết định người thắng

Bảng ablation làm lộ ra rằng `compare.py` — sau cả `W2-05` và `W2-08-prep` — vẫn
đọc quá tự tin đúng ở vùng mà bảng này sống: mức chênh 1–5 câu.

### 4a. `TRÁI CHIỀU` — trung bình nói một hướng, đếm câu nói hướng ngược

Cặp quyết định cả bảng, `rc50 → rc100`, `ndcg@10`:

```
Δ = +0,0255   CI95 [+0,0075, +0,0478]   →  loại 0 hẳn hoi
đếm câu: +10 tốt hơn / −11 xấu đi       →  nhiều câu bị làm hỏng HƠN số câu được sửa
kiểm định dấu: p = 1,0
```

Trung bình dương vì mấy câu thắng thắng **đậm hơn** mấy câu thua thua. Đó là một
kết quả hợp lệ — nhưng nó **không** là "hệ thống tốt hơn", và với một quyết định
trả bằng **1,91×** độ trễ thì nó không đủ để chi tiền. Nên nó có chữ riêng
`TRÁI CHIỀU`, **không** gộp vào `KHÔNG KẾT LUẬN`: ở đây khoảng tin cậy đọc được
hẳn hoi, nó đọc ra **hai điều đối nhau**, và đó là thông tin.

Cờ này bắt đúng **hai metric chất lượng xếp hạng** (`ndcg@10`, `map@20`) của cặp
đó, còn `recall`/`hit_rate` thì tăng sạch (0↔9, 1↔9, 1↔10). Xem §6.

### 4b. `TRÙNG KHỚP` — 0 câu khác nhau là hàng **chắc chắn nhất**, không phải mơ hồ nhất

`recall@5` giữa `rrf k=0` và `rrf k=1`: **0/209** câu khác nhau, `Δ = 0`, CI
`[0, 0]`. Luật cũ thấy "một biên đúng 0" → `KHÔNG KẾT LUẬN`; luật `underpowered`
cũng thoả (`min_achievable_p(0) = 1`). Cả hai đọc bằng chứng **mạnh nhất có thể**
thành bằng chứng yếu nhất. Thứ tự nhánh trong `verdict` là nội dung, không phải
hình thức — có test ghim rằng `identical` phải chạy **trước** `underpowered`.

### 4c. `mc_unstable` — biên CI đọc từ 6 mẫu lại

Xem §5. Đây là cờ tìm ra chuyện đáng nhất của hạng mục.

---

## 5. Người thắng của cả bảng do **6 mẫu lại trên 10.000** quyết định

Biên dưới của khoảng α là phần tử thứ `α/2 × B` của dãy đã sắp. Với α = 0,05 và
B = 10.000 đó là phần tử thứ **250** — đọc được. Với α đã hiệu chỉnh cho 39 phép
kiểm (0,00128) đó là phần tử thứ **6**.

Đo 6 seed × 3 mức `B` trên đúng cặp `rc50 → rc100`:

| metric | câu khác nhau | B=10.000 (đuôi 6) | B=50.000 (đuôi 32) | B=200.000 (đuôi 128) |
|---|---:|---|---|---|
| `ndcg@10` | 21 (+10/−11) | **dấu {+, −}** — đổi theo seed | {−} nhất quán | {−} nhất quán |
| `recall@5` | 10 (+9/−1) | **dấu {+, −, 0}** | {−, 0} | {−, 0} |
| `mrr` | 16 (+10/−6) | {+}, sd 0,00022 | {+, 0} | {+}, sd 0,00008 |
| `map@20` | 22 (+10/−12) | {−} | {−} | {−} |

**`ndcg@10` ở `B = 10.000` cho "khác biệt thật"; ở 50.000 và 200.000 thì khoảng
tin cậy CHỨA 0.** Kết luận "pool 100 hơn pool 50" ở mức đã hiệu chỉnh là một **tạo
tác Monte Carlo**, và nó là thứ đã chọn ra cấu hình thắng cho cả bảng ablation.

Cách vá **không** tốn thêm một lần lấy mẫu nào: số mẫu nằm dưới biên dưới là
`B(B, α/2)`, độ lệch chuẩn `sqrt(tail)`. Đọc lại biên ở `tail ± sqrt(tail)` của
**đúng dãy đã sắp đó** cho khoảng dao động của chính nó; nếu khoảng ấy chứa 0 thì
việc CI loại 0 là chuyện của số mẫu lại, không phải của dữ liệu.

### ⚠️ Và nó NGƯỢC ghi chú tôi tự viết ở `W2-08-prep`

`W2-08-prep` ghi: *"đừng chữa bằng cách tăng iterations"*, kèm phép đo chứng minh
10.000 → 50.000 **không đổi gì**. Ghi chú đó **đúng** — cho ca nó đo: metric nhị
phân thưa, phân bố nằm trên lưới bước `1/n`, phân vị cực đoan rơi đúng lên điểm
lưới. Ca của `W2-08` là metric **liên tục**, và ở đó tăng `B` **đảo** kết luận.

💡 **Hai giới hạn khác nhau cho ra cùng một triệu chứng "biên sát 0", và thứ phân
biệt chúng là độ hạt của metric.** Nếu tôi tin ghi chú của chính mình mà không đo
lại thì kết luận sai vẫn đứng. Đây là lần thứ hai trong `W2` một ghi chú đúng-cho-
ca-của-nó bị áp sang ca khác (lần đầu: `TD-11` "hạ `chunk_size` không cải thiện" bị
đọc thành công của `W2-01`).

---

## 6. Bậc thang pool: 20 → 50 phân giải được, 50 → 100 thì không — và đó là **vùng phủ vs xếp hạng**

| bước | `hit_rate@5` | `ndcg@10` | p95 | kết luận (α = 0,00128) |
|---|---|---|---|---|
| 20 → 50 | +0,0861 · **0↔18**, `p` = 7,6e-06 | +0,0658 · CI [+0,0249, +0,1201] | 276 → 609 ms | **khác biệt thật**, dư sức |
| 50 → 100 | +0,0383 · 1↔9, **trần `p` = 0,00195** | +0,0255 · CI [+0,0003, +0,0638] | 609 → 1164 ms | `KHÔNG ĐỦ LỰC` / `TRÁI CHIỀU` |

Ở mức **một cặp** (α = 0,05), `c50 → c100` cho **11/15** metric "khác biệt thật" —
và phần bị gắn cờ nói đúng chỗ nó xảy ra:

* **Vùng phủ tăng sạch**: `recall@10` 1↔9, `recall@20` 1↔10, `hit_rate@10` **0↔9**.
  Pool sâu hơn kéo thêm chunk đúng vào danh sách, và **không câu nào** mất.
* **Chất lượng xếp hạng thì không**: `ndcg@10` và `map@20` đều `TRÁI CHIỀU`
  (10↔11, 10↔12), `hit_rate@1`/`precision@1` `KHÔNG ĐỦ LỰC` (0↔4, trần 0,125).

Tức pool sâu hơn **nhập thêm bằng chứng đúng vào danh sách và đồng thời xáo lại
phần đầu** — một số câu tốt lên, một số xấu đi, tổng thì bằng nhau. Đó là câu
`W2-05` đã phát biểu ("pool sâu mua chất lượng danh sách, không mua chất lượng
hạng nhất") **sắc thêm một bậc**: không chỉ `hit_rate@1` đứng yên, mà chính
`nDCG`/`MAP` cũng có các thay đổi triệt tiêu nhau.

### ⚠️ Giới hạn của bộ ba metric chính, phải nói ra

`ndcg@10`, `hit_rate@1`, `mrr` **đều là metric chất lượng xếp hạng**; không cái nào
đo vùng phủ. Chúng được chọn *trước* khi xem số, mỗi cái có lý do (metric của
`G2`; thứ người dùng thấy và là đường **exact** duy nhất; metric không chia cho số
nhãn) — nhưng hệ quả là bộ ba này **loại trừ đúng chiều mà `rc100` thắng sạch**.

Nên phát biểu đúng của kết quả là: **`rc100` vs `rc50` là đánh đổi *vùng phủ* với
*chi phí*, không phải *chất lượng* với *chi phí*.** `rc100` mua +3,3 điểm
`recall@5` và +4,3 điểm `hit_rate@10` (đếm câu 1↔9, 0↔9 — sạch) bằng **1,91×** độ
trễ, và mua **không gì đo được** về chất lượng xếp hạng. Chọn cái nào phụ thuộc
việc bộ sinh có dùng được thêm bằng chứng trong ngữ cảnh hay không — đó là câu hỏi
của `W4`, không phải của `W2`.

---

## 7. Nền dense vs hybrid: **0/24** — `W2-05` tái lập ở cả ba độ sâu

| cùng `rerank_candidates` | số hàng đạt ý nghĩa | p95 dense → hybrid |
|---|---:|---|
| 20 | 0/8 | 267,6 → 276,5 ms |
| 50 | 0/8 | 618,5 → 608,9 ms |
| 100 | 0/8 | 1182,3 → 1163,9 ms |

`hit_rate@10` ở pool 20 cho **10↔10, `p` = 1,0** — đối xứng hoàn hảo. Và hybrid
**không** đắt hơn ở cả ba mức (hai mức còn *nhanh hơn* trong nhiễu đo). Kết luận
kiến trúc của `W2-05` không phải chuyện của một độ sâu pool: sau khi có reranker,
tầng hybrid không đo được ở **bất kỳ** độ sâu nào đã thử.

---

## 8. Chiều `chunk_size` **không so được** — và `p = 0,711` của `TD-11` dựa vào một hàng rào chưa tồn tại

`D6` sai theo hướng tôi không đoán trước. Chạy `e1-baseline-dense` vs
`e1-chunk550-dense`:

```
209/209 câu có tập nhãn KHÁC nhau  →  từ chối CẢ 15 metric
```

`G2` ghi *"đổi `chunk_size` thì phải dùng `hit_rate@k`/MRR"*, ngụ ý hai metric đó
còn dùng được. Đo ra thì chặt hơn thế: nhãn neo theo span (`TD-12`) nên đổi
`chunk_size` đổi luôn **tập** nhãn, không chỉ số lượng — và hàng rào băm
`relevant_digest` của `W2-03` (viết cho một lỗi *khác*: retriever thiếu
`fetch_doc_chunks` rơi về nhãn ghi sẵn) **bao trùm** hàng rào đếm nhãn của `TD-11`
và từ chối mọi metric.

### Và đây là phần đáng ghi lại

Chạy đúng cặp của `TD-11` (`baseline` vs `chunk550`, file sinh **trước** `W2-03`):

```
có digest: baseline=False  chunk550=False   →  hàng rào IM LẶNG
hit_rate@5  0,2153 → 0,2010   trong ngưỡng nhiễu      ← p = 0,711 đã công bố
ndcg@10     0,1621 → 0,1215   KHÔNG SO ĐƯỢC
```

**Con số `p = 0,711` của `TD-11` tồn tại được vì file của nó có trước hàng rào
băm.** Cùng phép so đó, với file có băm, bị **từ chối**. Hướng kết luận của `TD-11`
("hạ `chunk_size` không mua được gì") không đổi — nhưng bằng chứng cho nó **yếu
hơn** mức đã công bố: hôm nay công cụ nói phép so ấy không làm được.

⚠️ Ghi thành `TD-20`: chiều `chunk_size` cần một metric **bất biến theo chunking**
(đơn vị là span hoặc ký tự được phủ, không phải `chunk_id`) mới đo được. Đến lúc đó
thì ma trận `W2-08` vẫn thoả mục đích của `TD-11` — **thiết kế** có cả hai chiều
nên mức tăng của BGE-M3 không bị gán sai cho truncation (`e1-baseline-dense` vs
`e1-bgem3-dense` có nhãn **bit-identical**) — nhưng hàng `chunk550` không có `p`.

---

## 9. Luật `1/n` của tôi sai, và một script tái sinh nói ra

Bản đầu của cờ phân giải dùng bước lưới `1/n` cho **mọi** metric. Chạy nó trên 14
file `compare/` đã công bố: **13/14 file đổi kết luận**, phần lớn là
`precision@k`/`recall@k`.

Nguyên nhân: `1/n` đúng cho metric **nhị phân** (một câu đổi thì đổi cả 1,0), còn
`precision@20` nhận bội của 1/20 nên bước thật của nó nhỏ hơn **20 lần**. Luật đúng
là `min_increment / n` với `min_increment` = |hiệu| khác 0 nhỏ nhất, **đo từ chính
dữ liệu**. Sau khi sửa: **6/14 file**, 10 hàng, và kiểm từng hàng thì cả 10 đều
đúng.

Mặc định của `min_increment` cũng phải sửa: bản đầu để `1,0` ("coi như nhị phân")
và nó làm một test **đã có** đổi kết luận — khoảng `[−0,1630, −0,0054]` của
`ndcg@10` trên 43 câu (chính dẫn chứng `cross_lingual` của `W2-04`) bị dán
`KHÔNG KẾT LUẬN` vì `1/43 = 0,0233 > 0,0054`, trong khi độ hạt thật ở đó là
`0,0068/43 = 0,00016`. **Một cờ đoán ngưỡng của chính nó là một cờ bật theo phỏng
đoán** — mà đó đúng là loại lỗi cờ này được dựng để bắt. Mặc định giờ là `0,0` =
"chưa biết" = không gắn cờ, và có test ghim rằng `compare_runs` điền nó cho **mọi**
hàng bootstrap.

### Bốn số đã công bố bị chạm, cả bốn theo hướng sắc thêm

| chỗ | trước | sau |
|---|---|---|
| `W2-04` `k=1` vs dense | 3/15 đạt ý nghĩa | **2/15** (`precision@5` → `KHÔNG KẾT LUẬN`) |
| `W2-05` nền dense vs hybrid | "13/15 trong ngưỡng nhiễu" | 13/15 **không đạt ý nghĩa** = 12 nhiễu + 1 `KHÔNG KẾT LUẬN` |
| `W2-05` `c50 → c100` | `ndcg@10` tăng có ý nghĩa | **`TRÁI CHIỀU`** (10↔11) |
| `CHECKLIST` §2 | "`c=100` tốt hơn ở **mọi** metric" | tốt hơn ở **điểm tổng**; `nDCG`/`MAP` là `TRÁI CHIỀU` |

**Hai chốt kiểm soát giữ nguyên đúng từng chữ số**, và đó là bằng chứng cờ không
dán không-kết-luận cho mọi thứ: `cmp-baseline-vs-bgem3` vẫn **15/15** (kết quả tiêu
đề `W2-01`) và `cmp-bgem3-vs-bgem3-rr-c50` vẫn **15/15** (kết quả tiêu đề `W2-05`).

---

## 10. Kết luận

1. **Cấu hình thắng: reranked, pool 100, nền hybrid** — `ndcg@10` **0,6736**,
   `hit_rate@1` **0,5789**, p95 **1163,9 ms**. Nhưng nó chỉ phân biệt được với
   **8/13** ô còn lại; ô hạng 2 (`ondense-rc100`) không phân biệt được, và ba ô
   nữa thì ba metric chính **không đồng ý**.
2. **`rc50` rẻ hơn 1,91× và bị bác bởi đúng một metric trong ba**, mà metric ấy có
   biên CI `−0,0007` với kiểm định dấu `p = 0,45`, còn phép kiểm **exact** duy nhất
   trong bộ ba (`hit_rate@1`) nói `KHÔNG ĐỦ LỰC`. Phán quyết thuộc `W4`: nếu bộ
   sinh dùng được thêm bằng chứng thì trả 1,91×, nếu không thì `rc50`.
3. **Bậc thang pool có một bước phân giải được (20→50) và một bước không (50→100)**,
   và cái không phân giải được là *chất lượng xếp hạng* — *vùng phủ* thì vẫn tăng
   sạch.
4. **Tầng hybrid không đo được ở cả ba độ sâu pool** (0/24), và nó miễn phí.
5. **Chiều `chunk_size` không có `p`** với công cụ hiện tại → `TD-20`.
6. **Người thắng từng được quyết định bởi 6 mẫu lại**, và ba cờ mới trong
   `compare.py` là hệ quả.

### Việc còn lại

* **`W2-09`** phải chốt chuyện `W2-08-prep` và hạng mục này đều cố ý không chốt:
  bảng một-cặp **vẫn** không hiệu chỉnh cho 15 metric (kỳ vọng 0,75 dương giả mỗi
  bảng). Sửa thì viết lại kết luận `W2-01`…`W2-07`; không sửa thì phải nói ra.
* **`TD-20`** metric bất biến theo chunking — làm **trước** khi quét lại chiều
  `chunk_size`.
* **`TD-19`** (`golden`/`golden_digest` vào `EvalReport.config`) vẫn phải làm trước
  `TD-13`.
* Ngưỡng `MIN_TAIL_RESAMPLES = 30` hiện chỉ dùng để **giải thích** cờ, chưa dùng để
  tự nâng `B`. Nâng `B` cho α = 0,00128 cần ~47.000 mẫu lại, tức ~4,7× chi phí — đo
  rồi hãy làm, và `W2-09` là chỗ quyết.
