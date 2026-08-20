# `W2-03` — nhánh sparse thành `Retriever`, và con số cho câu "sparse đóng góp gì"

**Ngày:** 2026-08-20 · **Nhánh:** `feat/w1-foundation` · **Index:** `rag_bgem3`
(15.814 chunk, fingerprint `0eaaf9265487eabb`)

---

## 1. Tóm tắt

`retrieve_sparse()` của `W2-02` giờ là một `Retriever`, nên eval harness đo được
nó. Đo rồi thì hai câu trả lời, và chúng ngược nhau:

| Câu hỏi | Trả lời |
|---|---|
| Sparse có tốt hơn dense trên golden set? | **Không.** nDCG@10 0,4442 → 0,3733 (`CI95 [−0,1190, −0,0225]`) |
| Sparse có tra được mã tài liệu mà dense không? | **Có, áp đảo.** hit@10 0,0784 → **0,5098**, `p = 4,8e-07` |

Hai câu đó không mâu thuẫn: chúng đo hai loại truy vấn khác nhau, và cái thứ hai
mới là DoD của `W2-03`.

Con số quan trọng nhất của phiên là con số thứ ba: **hợp hai nhánh cho
`hit_rate@10 = 0,7033`** so với dense 0,6268. Đó là **trần lý thuyết** của
`W2-04` — RRF không thể vượt qua nó, chỉ có thể tiến gần. +0,0765 tuyệt đối
(+12,2% tương đối) là toàn bộ số tiền đang nằm trên bàn.

---

## 2. Việc đã làm

| Thành phần | Vai trò |
|---|---|
| `retrieval/sparse.py` · `QdrantSparseRetriever` | Bọc store thành `Retriever`, dùng lại **cùng** kết nối |
| `retrieval/branch.py` · `build_branch()` | Chọn nhánh từ tên. `HYBRID`/`RERANKED` báo "chưa cài", không báo "tên sai" |
| `--retrieval-mode` + `MODE=`/`RUN=` | Nhánh là tham số của lần **đo**, không phải của index |
| `QueryScore.relevant_digest` | Hàng rào nhãn — xem §7 |
| `scripts/known_item_probe.py` | Phép đo cho DoD, không cần nhãn người — xem §5 |
| `store.verify_schema()` trong `_eval_against_index` | Kiểm schema **trước** khi quét span |

Ba quyết định đáng nói:

**Lớp bọc, không phải store thứ hai.** `QdrantSparseRetriever` giữ một tham
chiếu tới `QdrantDenseRetriever` đang mở kết nối. `W2-04` sẽ bọc **cùng** store
đó và gọi cả hai nhánh; nếu sparse là một store riêng thì RRF phải tự đồng bộ hai
đối tượng và tự tin rằng chúng trỏ vào cùng một collection.

**`build_branch(store, "dense")` trả về chính `store`.** Không bọc thêm một lớp
chỉ để đối xứng: `retriever.name` đi vào mọi report, nên đổi nó là làm số cũ của
W1/`W2-01`/`W2-02` không so được nữa.

**Nhánh truy hồi KHÔNG vào `IndexConfig`.** Nó không quyết định vector nào được
ghi nên nó không thuộc `fingerprint` — và một trường nằm trong `IndexConfig`
nhưng ngoài `fingerprint` là thứ phải giải thích lại mỗi lần đọc. Đã có hai
trường như vậy (`device`, `batch_size`), đủ rồi. Ở `W2-07` nó sẽ là một chiều
của ma trận thí nghiệm.

---

## 3. Trên golden set: sparse kém hơn, và kém một cách có ý nghĩa

209 câu, cùng index, **cùng nhãn** (`n_relevant_mean` 1,3828 hai bên, 0/209 câu
lệch băm nhãn — xem §7).

| metric | dense | sparse | Δ | kiểm định |
|---|---:|---:|---:|---|
| `hit_rate@1` | 0,3397 | 0,2919 | −0,0478 | p=0,132 · 23↔13 — **nhiễu** |
| `hit_rate@5` | 0,5455 | 0,4593 | −0,0861 | p=0,018 · 35↔17 |
| `hit_rate@10` | 0,6268 | 0,5120 | −0,1148 | p=0,002 · 40↔16 |
| `hit_rate@20` | 0,6746 | 0,5311 | −0,1435 | p=0,000 · 44↔14 |
| `ndcg@10` | 0,4442 | 0,3733 | −0,0709 | CI95 [−0,1190, −0,0225] |
| `mrr` | 0,4394 | 0,3623 | −0,0771 | CI95 [−0,1273, −0,0267] |
| `map@20` | 0,3853 | 0,3333 | −0,0520 | CI95 [−0,0970, −0,0069] |
| `recall@10` | 0,5813 | 0,4721 | −0,1093 | CI95 [−0,1754, −0,0439] |
| `precision@1` | 0,3397 | 0,2919 | −0,0478 | CI95 [−0,1053, +0,0096] — **nhiễu** |

12/15 metric khác biệt thật, tất cả theo chiều sparse kém hơn. Bảng đầy đủ ở
`cmp-bgem3-vs-bgem3-sparse.md`.

Đây là kết quả **phải chờ đợi**, không phải thất bại: `golden_v1` gồm câu hỏi tự
nhiên do LLM sinh, tức đúng loại truy vấn mà dense sinh ra để xử lý. Nếu sparse
thắng ở đây thì phải đi kiểm lại nhánh dense.

Một chi tiết bác bỏ giả định của chính `W2-02`: sparse **luôn trả đủ 20** kết quả
trên cả 209 câu. Mặt bù "không trùng token thì sparse trả rỗng" là hiện tượng của
corpus nhỏ; trên 15.814 chunk thì mọi câu hỏi tiếng Việt hay tiếng Anh đều có
token trùng ở đâu đó. Cái giới hạn sparse ở đây là **xếp hạng sai**, không phải
**không tìm ra**.

---

## 4. Chỗ sparse đóng góp — và trần của `W2-04`

Cột "đổi chiều" mới là chỗ đáng đọc, không phải cột trung bình. Ở `hit_rate@10`:

| nhóm | n | chỉ dense | **chỉ sparse** | cả hai | không bên nào |
|---|---:|---:|---:|---:|---:|
| `factoid` | 68 | 7 | **10** | 41 | 10 |
| `cross_lingual` | 43 | 19 | 1 | **0** | 23 |
| `adversarial` | 34 | 6 | 1 | 11 | 16 |
| `multi_hop` | 34 | 4 | 2 | 20 | 8 |
| `aggregation` | 26 | 4 | 1 | 18 | 3 |
| `table_lookup` | 4 | 0 | 1 | 1 | 2 |

| ngôn ngữ | n | chỉ dense | **chỉ sparse** | cả hai | không bên nào |
|---|---:|---:|---:|---:|---:|
| `en` | 82 | 4 | **9** | 45 | 24 |
| `vi` | 127 | **36** | 7 | 46 | 38 |

Ba điều đọc ra được:

1. **`cross_lingual` là chỗ sparse chết hẳn**: 19 câu chỉ dense, 1 câu chỉ
   sparse, và **0 câu cả hai**. Cơ chế rõ ràng: câu hỏi tiếng Việt trên tài liệu
   tiếng Anh thì không có token nào trùng. Toàn bộ mức kém của sparse ở §3 chủ
   yếu đến từ 43 câu này (20% tập đo).
2. **Trên truy vấn tiếng Anh, sparse THẮNG**: 9 câu chỉ sparse vs 4 câu chỉ
   dense. Khi truy vấn và tài liệu cùng ngôn ngữ thì trùng lặp từ vựng là tín
   hiệu thật.
3. **`factoid` là chỗ sparse bù được nhiều nhất**: 10 vs 7. Câu hỏi dữ kiện có
   thực thể và con số, tức có neo từ vựng.

**Trần của RRF:** hợp hai nhánh cho `hit_rate@10 = 0,7033` vs dense 0,6268.
`W2-04` chỉ có thể tiến tới con số đó — và chỉ khi nó biết chọn *đúng* nhánh cho
từng câu, việc mà RRF (hợp nhất theo thứ hạng, không nhìn nội dung) không làm
được hoàn toàn. Ghi con số này lại **trước khi** làm `W2-04` để lúc đó không tự
diễn giải kết quả theo hướng có lợi.

---

## 5. DoD: known-item search — bằng chứng, và cách đo không cần nhãn người

DoD của `W2-03` là "truy vấn từ khoá lạ (mã số, tên riêng) mà dense miss thì
sparse hit". `golden_v1` **không trả lời được** câu đó: 209 câu đều là câu hỏi tự
nhiên, không có câu nào tra mã. Nên phải có phép đo riêng.

Cách đo dựa vào một tính chất của bài toán known-item: nếu truy vấn là chuỗi
**xuất hiện nguyên văn** trong corpus thì "đúng" kiểm được bằng **so chuỗi** —
không cần người gán nhãn, và không có chỗ cho phán đoán chen vào.

Lấy 51 mã dạng chữ-in-hoa + số xuất hiện ở 1–3 chunk (project ID, trust fund ID,
tên chương trình của World Bank). `make known-item`, seed 20260820:

| | dense | sparse |
|---|---:|---:|
| hit@10 | 0,0784 (4/51) | **0,5098 (26/51)** |
| hit@1 | 0,0196 (1) | **0,3529 (18)** |
| MRR | 0,0276 | **0,4134** |
| hạng trung vị khi tìm được | 6,5 | **1,0** |

Đổi chiều: **chỉ sparse 22 · chỉ dense 0** · cả hai 4 · không bên nào 25.
McNemar exact `p = 4,8e-07`.

**Không có một mã nào mà dense tìm ra và sparse không.** Đó là dạng mạnh nhất của
DoD: không chỉ sparse tốt hơn trung bình, mà nó **phủ trọn** khả năng của dense
trên loại truy vấn này.

Đây cũng là loại truy vấn có thật trong sản phẩm — người dùng dán mã dự án vào
hộp tìm kiếm — nên nó không phải một bài kiểm tra dựng lên cho vui.

---

## 6. Tại sao sparse vẫn miss một nửa: sparse học được không phải index khớp đúng

25/51 mã **không nhánh nào** tìm ra. Nhìn vào mã, có mẫu ngay:

| | sparse tìm được | sparse miss |
|---|---|---|
| ví dụ | `VIE-01`, `SEDP-2016-2020`, `ASEAN-518`, `DB2017`, `RIE2025` | `P171645`, `TCS210345`, `LL1131900`, `JRC107150`, `FY2003` |
| độ dài trung bình | 7,5 ký tự | 8,9 |
| số chữ số trung bình | 4,2 | 5,2 |

Tokenizer của BGE-M3 giải thích hết:

```
VIE-01          → ['▁', 'VIE', '-01']            ← 'VIE' là mảnh HIẾM
SEDP-2016-2020  → ['▁SE', 'DP', '-2016', '-2020']
ASEAN-518       → ['▁ASEAN', '-5', '18']
DB2017          → ['▁DB', '2017']

P171645         → ['▁P', '171', '645']           ← không mảnh nào hiếm
TCS210345       → ['▁T', 'CS', '210', '345']
LL1131900       → ['▁LL', '113', '1900']
FY2003          → ['▁F', 'Y', '2003']
```

Mã tìm được luôn có một **neo từ vựng** — một mảnh subword tự nó là từ hiếm.
Mã miss rã thành một chữ cái cộng vài khối ba chữ số, và `171`, `645`, `210` có
mặt khắp 15.814 chunk tài liệu thống kê.

**Kết luận thiết kế, và nó đổi kế hoạch:** sparse học được của BGE-M3 **không
phải** một index khớp đúng. Nó là một biểu diễn trên vocab **subword**, nên nó
mạnh ở "từ khoá hiếm" và yếu ở "chuỗi ký tự chính xác". Muốn tra mã đúng nghĩa
thì cần một trong hai thứ khác:

* **BM25 thô** trên token mức **từ** (không phải subword) — nhánh mà `W2-02` đã
  ghi là cần `modifier=Modifier.IDF`. Giờ nó có bằng chứng chứ không phải phỏng
  đoán: 25/51 mã hiện đang không tìm được bằng bất cứ nhánh nào.
* Hoặc **filter payload theo khớp chuỗi** — rẻ hơn nhiều, và Qdrant đã có
  `create_payload_index` sẵn từ `W1-07`.

Việc này chưa làm trong `W2-03` vì nó cần đổi schema collection (thêm named
vector sparse thứ hai) và build lại index — xem §10.

---

## 7. Hàng rào nhãn: một hố im lặng đã đóng trước khi nó kịp hại

Trước phiên này, `compare.py` có hai hàng rào: canh **số** nhãn mỗi câu, và canh
**tập truy vấn**. Cả hai đều không thấy được ca sau:

Eval harness lấy `fetch_doc_chunks` bằng `getattr` để tính lại nhãn golden set
theo span (`TD-12`). Một retriever **thiếu** method đó thì harness lặng lẽ rơi về
`relevant_chunk_ids` ghi trong file. Hai lần chạy có thể có **cùng số** nhãn mà
**nhãn khác nhau** — lúc đó `hit_rate` hai bên đo hai bài toán khác nhau, bảng so
vẫn hiện ra bình thường, và không có gì báo lỗi.

Lớp bọc `W2-03` là chỗ đầu tiên trong dự án có thể rơi vào hố đó. Nên:

* `QueryScore.relevant_digest` — băm sha256 của **tập** `chunk_id` liên quan
  (16 hex, không phụ thuộc thứ tự), ghi vào `*-per-query.jsonl`.
* `compare.py` từ chối **toàn bộ** phép so khi có câu lệch băm. Không lọc bỏ câu
  lệch rồi so phần còn lại — đó đúng là kiểu tự chọn mẫu mà hàng rào #2 cấm.
* Câu **thiếu** băm (file trước `W2-03`) tính là "không biết", không phải "khớp",
  và có cảnh báo nói rõ là hàng rào đang tắt.
* Chuyển tiếp `fetch_doc_chunks` **tường minh**, không `__getattr__`: một wrapper
  "trong suốt" thì mypy không kiểm được gì và hố mở lại ngay lần đổi tên method
  tiếp theo. Có test canh đúng phép thử `getattr` mà harness thực sự làm.

Đã chạy thật: 209/209 câu có băm ở cả hai lần chạy, **0 câu lệch**. Nên bảng ở
§3 và §4 là bảng so hai hệ thống trên **cùng** một bài toán.

Đây là biến thể thứ ba của cùng một bài học: `TD-11` (im lặng cắt text), `W2-02`
(im lặng bỏ nửa index), giờ là im lặng đổi nhãn. Ba lần đều là "hệ thống chạy, số
ra, không ai biết là sai".

---

## 8. Chi phí độ trễ: sparse chậm hơn dense 5,5× ở phép tìm

Từ chính hai lần chạy eval ở §3 (209 truy vấn, sau warm-up):

| | p50 | p95 | max |
|---|---:|---:|---:|
| dense | 30,2 ms | 45,7 ms | 49,7 ms |
| **sparse** | **113,4 ms** | **142,1 ms** | 202,8 ms |

Số dense khớp `W2-02` (p50 31,5 · p95 46,6) trong khoảng dao động giữa các lần
chạy, nên đây không phải máy đang chậm bất thường.

Đo riêng trên 60 truy vấn đầu để tách thành phần:

| thành phần | ms |
|---|---:|
| embed truy vấn (một forward pass, **dùng chung** cả hai nhánh) | 12,6 |
| tìm trong index dense (HNSW) | 17,8 |
| **tìm trong index sparse** | **97,8** |

Vector sparse của truy vấn chỉ có **31,8 entry** trung bình (p50 30 · min 13 ·
max 60) — ngắn hơn nhiều so với 95,9 entry/chunk phía tài liệu. Vậy mà 32 posting
list trên 15.814 point tốn 98 ms. Đây là số phải mang sang `W2-04`: mỗi truy vấn
hybrid sẽ là ~12,6 + 17,8 + 97,8 ≈ **128 ms** nếu chạy tuần tự, tức ~4× dense.

Ngưỡng `G2` là p95 < 3500 ms nên vẫn còn rất nhiều chỗ. Nhưng nó nói rằng nhánh
sparse — không phải model embedding — sẽ là thành phần **nặng nhất** của đường
truy hồi từ `W2-04` trở đi. Chỗ 12,6 ms embed dùng chung là phần lợi còn lại của
quyết định "một forward pass" ở `W2-01`.

⚠️ Con số 97,8 ms này **chưa được tối ưu gì cả** — chưa thử `on_disk`, chưa thử
giảm entry phía tài liệu bằng ngưỡng trọng số. Đọc nó là "chi phí hiện tại",
không phải "chi phí tối thiểu".

---

## 9. Kết quả âm và giới hạn — những chỗ tôi đã đoán sai

**Giả định sai #1: "dense lẫn giữa những mã gần giống nhau".** Tôi dựng một
corpus 7 chunk gồm sáu mã na ná nhau (`GSO-2024-XII`, `GSO-2023-XI`, …) và viết
test cho rằng sparse sẽ tra đúng còn dense thì không. Đo thật: **cả hai** đặt
đúng chunk ở hạng 1, dense với biên rõ ràng (0,6947 vs 0,6411). Với 6 đối thủ thì
đó là điều phải chờ đợi.

Cái tôi đã **không** làm: đổi assertion thành `rank_sparse <= rank_dense` để test
xanh. Nó sẽ pass — vì hai bên bằng nhau — và đọc như một chiến thắng. Thay vào
đó có một test **canh kết quả âm** (`test_dense_picks_it_too_and_that_is_the_
honest_result`), để nó không lặng lẽ biến mất.

**Giả định sai #2: "không trùng token thì sparse trả rỗng".** Đúng trên corpus 3
chunk, sai trên 15.814: sparse trả đủ 20 kết quả cho cả 209 câu. Mặt bù thật của
sparse là **xếp hạng sai**, không phải **không trả gì**.

**Một chỗ cả hai nhánh cùng sai.** Truy vấn "thống kê ba tháng cuối năm 2024" =
quý bốn; cả dense lẫn sparse đặt chunk quý **ba** ở hạng 1. Có test canh việc
này, vì nó nói `W2-04` không cứu được gì: hợp nhất hai danh sách cùng sai một
kiểu thì vẫn sai. Chỗ đó là việc của reranker (`W2-05`).

**Giới hạn của mọi con số ở §3–§4:** đo trên `golden_v1`, vốn **review bằng
model** (`TD-13`). So sánh *tương đối* giữa hai nhánh vẫn hợp lệ vì cùng một
thước đo; con số *tuyệt đối* thì chưa được gọi là "human-verified".

**Giới hạn của §5:** 51 mã là một mẫu nhỏ, và tất cả đều là mã của World Bank
trong một corpus 60 tài liệu. Chiều của kết quả (`p = 4,8e-07`, 22↔0) thì không
mong manh; độ lớn tuyệt đối thì chưa chắc giữ trên corpus khác.

---

## 10. Việc tiếp theo

1. **`W2-04` RRF fusion** — trần đã biết: `hit_rate@10` tối đa **0,7033** (dense
   0,6268). Hợp nhất theo **thứ hạng**, không theo điểm (`W2-02` quyết định 87).
   Phải đo lại độ trễ ở đó: ~128 ms/truy vấn theo §8.
2. **Cân nhắc `TD-18`: nhánh khớp đúng cho mã tài liệu.** §6 cho thấy 25/51 mã
   hiện không tìm được bằng bất cứ nhánh nào, và nguyên nhân là vocab subword —
   không phải thứ RRF hay reranker sửa được. Hai phương án: BM25 thô mức từ (cần
   named vector sparse thứ hai + build lại index), hoặc filter payload khớp chuỗi
   (rẻ hơn nhiều). Nên đo phương án rẻ trước.
3. **`W2-05` reranker** — chỗ duy nhất xử lý được ca "cả hai nhánh cùng sai" ở
   §9.
4. `TD-13` (review golden set bằng người) vẫn là nợ đáng nhất, và giờ nó chặn
   thêm một thứ: mọi con số §3–§4 đều mang chú thích ấy.

## Lệnh tái lập

```bash
make test                                     # 697 unit (16 ca branch, thuần)
make up && make test-integration              # 73 integration (14 ca sparse)
make test-gpu                                 # 15 gpu (4 ca BGE-M3 thật)

make eval-retrieval BUNDLE=bgem3 MODE=dense  RUN=bgem3
make eval-retrieval BUNDLE=bgem3 MODE=sparse RUN=bgem3-sparse
make eval-compare BASE=bgem3 CAND=bgem3-sparse
make known-item BUNDLE=bgem3                  # §5, seed 20260820
```
