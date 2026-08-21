# `W2-04` — RRF fusion: `k=60` của bài báo là lựa chọn tệ nhất, và một bug 64 ms

**Ngày:** 2026-08-20 · **Nhánh:** `feat/w1-foundation` · **Index:** `rag_bgem3`
(15.814 chunk, fingerprint `0eaaf9265487eabb`)

---

## 1. Tóm tắt

Ba kết quả, và cả ba đều ngược với thứ tôi tin lúc bắt đầu:

| | tôi dự đoán | đo được |
|---|---|---|
| RRF ở `k=60` (mặc định bài báo) | ngang dense (`hit_rate@10` ≈ 0,6268) | **0,5742 — kém rõ rệt** |
| `candidate_k` sâu hơn | tốt hơn (thấy được đồng thuận ở hạng sâu) | **tệ hơn**, và chỉ khi `k` lớn |
| Chi phí độ trễ hybrid | ~128 ms (4× dense) | **≈ dense một mình** |

Cấu hình thắng là **`k=1`, `candidate_k=20`**: `hit_rate@10` **0,6555** vs dense
0,6268, `hit_rate@20` **0,7177** vs 0,6746, và **cả 15 metric đều tốt hơn dense**
— nhưng chỉ **2/15** đạt ý nghĩa thống kê, nên đây là cải thiện **nhỏ**, phần lớn
nằm dưới ngưỡng phân giải của `golden_v1`.

> 📝 **Đính chính 2026-08-22 (`W2-08`)**: con số này là **3/15** khi công bố. `W2-08`
> thêm cờ phân giải vào `compare.py` và `precision@5` thành `KHÔNG KẾT LUẬN` — biên
> CI95 `+0,0010` nằm trong một bước lưới của chính metric ấy. Cùng chiều kết luận,
> yếu hơn một bậc. Xem `w2-08-ablation.md` §9.

Phần đáng nhất của phiên lại không phải RRF: phân rã độ trễ để kiểm một con số
không khớp đã tìm ra **một bug hiệu năng 64 ms/lần gọi**, và bug đó làm **sai hai
con số tôi đã công bố** ở `W2-02` và `W2-03`. Xem §6.

---

## 2. Dự đoán đã ghi TRƯỚC khi đo

Thứ tự này quan trọng, nên nó được ghi lại: `W2-03` để lại con số trần
`hit_rate@10` hợp hai nhánh = **0,7033** (dense 0,6268). Trước khi cài RRF tôi
lập luận thế này:

> RRF **xen kẽ** hai danh sách, nên top-10 hợp nhất chỉ chứa được khoảng top-5
> mỗi nhánh. Hợp@5 đo được là **0,6268** — đúng bằng dense@10. Vậy `hybrid@10` sẽ
> **ngang dense**, không phải bằng trần. Chỗ RRF có cơ hội thắng là nơi hai nhánh
> **đồng thuận** được cộng điểm, tức `hit_rate@1`, không phải `@10`.

Đường hợp đầy đủ (từ `*-per-query.jsonl` của `W2-03`):

| k | dense | sparse | hợp (trần) |
|---|---:|---:|---:|
| 1 | 0,3397 | 0,2919 | 0,4019 |
| 5 | 0,5455 | 0,4593 | 0,6268 |
| 10 | 0,6268 | 0,5120 | 0,7033 |
| 20 | 0,6746 | 0,5311 | 0,7416 |

**Dự đoán sai theo cả hai chiều.** Ở `k=60` thực tế là 0,5742 — *kém* dense, tệ
hơn dự đoán. Còn `hit_rate@1` thì **không** cải thiện ở bất kỳ cấu hình nào
(0,3397 → 0,3397 ở `k=1`, tụt ở mọi `k` khác). Lý do phần đầu sai: top-10 hợp
nhất **không phải** hợp của hai top-5, mà bị chiếm bởi những chunk *cả hai nhánh
đều xếp hạng trung bình* — và chúng thường không liên quan.

---

## 3. Bảng quét: `k` là cần điều khiển chính, và nhỏ thì tốt

11 lần chạy trên cùng index, cùng 209 câu, **cùng nhãn** (0 câu lệch băm — §7).

| lần chạy | `k` | `candidate_k` | hit@1 | hit@5 | **hit@10** | hit@20 | nDCG@10 | MRR | recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense (nền) | — | — | 0,3397 | 0,5455 | 0,6268 | 0,6746 | 0,4442 | 0,4394 | 0,5813 |
| sparse | — | — | 0,2919 | 0,4593 | 0,5120 | 0,5311 | 0,3733 | 0,3623 | 0,4721 |
| **hybrid** | **1** | **20** | 0,3397 | 0,5742 | **0,6555** | **0,7177** | **0,4563** | **0,4436** | 0,6013 |
| hybrid | 1 | 50 | 0,3397 | 0,5742 | 0,6555 | 0,7129 | 0,4557 | 0,4425 | 0,6013 |
| hybrid | 1 | 100 | 0,3349 | 0,5742 | 0,6555 | 0,7129 | 0,4530 | 0,4396 | 0,5997 |
| hybrid | 2 | 50 | 0,3301 | 0,5789 | 0,6603 | 0,7129 | 0,4530 | 0,4348 | 0,6037 |
| hybrid | 5 | 50 | 0,3206 | 0,5502 | 0,6603 | 0,7129 | 0,4443 | 0,4236 | 0,6053 |
| hybrid | 10 | 50 | 0,3062 | 0,5120 | 0,6411 | 0,7129 | 0,4305 | 0,4086 | 0,5893 |
| hybrid | 60 | 20 | 0,3014 | 0,5359 | 0,6364 | 0,7177 | 0,4313 | 0,4080 | 0,5853 |
| hybrid | 60 | 50 | 0,3014 | 0,4689 | 0,5742 | 0,6746 | 0,4021 | 0,3871 | 0,5327 |
| hybrid | 60 | 100 | 0,3014 | 0,4498 | 0,5024 | 0,6220 | 0,3785 | 0,3762 | 0,4697 |
| hybrid | 60 | 50 · w 2:1 | 0,2967 | 0,4880 | 0,6029 | 0,6794 | 0,4053 | 0,3848 | 0,5566 |

Ba điều đọc ra:

**`k` nhỏ thắng, đơn điệu.** Giữ `candidate_k=50`, nDCG@10 theo `k` = 1 → 2 → 5 →
10 → 60 cho **0,4557 → 0,4530 → 0,4443 → 0,4305 → 0,4021**. Giá trị 60 của bài
báo gốc là **đầu tệ nhất** của khoảng, và nó tệ hơn dense một mình.

Cơ chế thì rõ và tính được: `k` là "tôi tin thứ hạng đến mức nào". Với `k=60` thì
một chunk **cả hai** nhánh xếp hạng 3 được `2/63 = 0,0317`, đè lên chunk mà dense
xếp **hạng 1** (`1/61 = 0,0164`) — gấp **1,9 lần**. Với `k=1` thì hai bên bằng
nhau (`0,5` và `0,5`). Nói cách khác `k=60` cho phép **sự đồng thuận yếu lật đổ
một tín hiệu mạnh**, và trên tập này sự đồng thuận yếu chủ yếu là nhiễu.

**`candidate_k` chỉ có ảnh hưởng khi `k` lớn.** Ở `k=60`: c20/c50/c100 cho
`hit_rate@10` **0,6364 / 0,5742 / 0,5024** — chênh nhau tới 13 điểm. Ở `k=1`:
**0,6555 / 0,6555 / 0,6555** — không lệch một chữ số. Cùng một cơ chế: `k` nhỏ
làm ứng viên hạng sâu gần như vô giá trị, nên lấy sâu bao nhiêu cũng không đổi
kết quả. **Hệ quả thực dụng: với `k=1` thì chọn `candidate_k` nhỏ nhất** — nó
miễn phí về chất lượng và rẻ hơn về độ trễ.

Điều này ngược hẳn dự đoán của tôi, và ngược cả với hai bài test đơn vị tôi viết
để biện minh cho `candidate_k` sâu (`test_deep_agreement_beats_shallow_solo`).
Hai test đó **không sai** — số học của chúng đúng, chunk sâu-mà-đồng-thuận thật sự
được đẩy lên. Điều chúng không nói được là: những chunk ấy thường **không liên
quan**. Số học đúng, tiên đề sai.

**Weighted RRF gần như vô dụng, và đã đoán được trước.** Cân dense 2:1 ở `k=60`
đưa `hit_rate@10` 0,5742 → 0,6029 — có nhưng không đủ. Khớp với tính chất tính
được trong test đơn vị: muốn lật một sự đồng thuận **cùng độ sâu** thì cần tỉ lệ
trọng số **> 30,5:1**, vì cân dense lên cũng cân luôn phần dense của chunk đồng
thuận. Cần điều khiển này không đáng quét ở `W2-08`.

---

## 4. Cấu hình thắng có ý nghĩa thống kê tới đâu

`k=1, candidate_k=50` vs dense (`make eval-compare bgem3 bgem3-rrf-k1`):

| metric | dense | hybrid k=1 | Δ | kiểm định |
|---|---:|---:|---:|---|
| `recall@20` | 0,6324 | 0,6754 | **+0,0431** | CI95 [+0,0056, +0,0813] — **thật** |
| `precision@5` | 0,1340 | 0,1445 | **+0,0105** | CI95 [+0,0010, +0,0211] — **thật** |
| `precision@20` | 0,0445 | 0,0471 | **+0,0026** | CI95 [+0,0005, +0,0048] — **thật** |
| `hit_rate@20` | 0,6746 | 0,7129 | +0,0383 | p=0,096 · 5↔13 |
| `hit_rate@10` | 0,6268 | 0,6555 | +0,0287 | p=0,286 · 8↔14 |
| `hit_rate@5` | 0,5455 | 0,5742 | +0,0287 | p=0,210 · 5↔11 |
| `ndcg@10` | 0,4442 | 0,4557 | +0,0116 | CI95 [−0,0129, +0,0369] |
| `mrr` | 0,4394 | 0,4425 | +0,0031 | CI95 [−0,0230, +0,0296] |
| `hit_rate@1` | 0,3397 | 0,3397 | ±0,0000 | p=1,000 · 9↔9 |

**15/15 metric có dấu ≥ 0, nhưng chỉ 3 đạt ý nghĩa.** Đọc đúng: RRF `k=1` là một
cải thiện **nhỏ và thật nhưng phần lớn không chứng minh được** trên 209 câu —
đúng vùng mà `TD-11` đã cảnh báo (`golden_v1` chỉ phân giải được mức chênh ≥ 6
điểm `hit_rate`; ở đây là +2,9 điểm).

⚠️ Không được đọc "cả 15 metric đều dương" như một kiểm định. 15 metric này tương
quan rất mạnh với nhau (đều tính từ cùng một danh sách xếp hạng), nên nó **không**
là 15 phép thử độc lập và không có phép thử dấu nào áp được ở đây.

Chỗ cải thiện tập trung ở **`recall@20`** và **`hit_rate@20`** — tức RRF làm tốt
việc *mở rộng vùng phủ*, không làm tốt việc *đưa câu trả lời lên đầu*. `hit_rate@1`
đứng im xác nhận điều đó. Đây chính là hình dạng của một **bộ sinh ứng viên** tốt,
không phải một bộ xếp hạng cuối tốt — xem §8.

---

## 5. Qdrant dùng `k = 1`, không phải 60 — và đó là lý do tự cài

Qdrant 1.15 có `Fusion.RRF` server-side, nên có một bản tham chiếu độc lập để đối
chiếu. Suy `k` của nó từ chính điểm nó trả về (7 chunk, mọi điểm khớp `1/(1+rank)`
tới 6 chữ số):

```
code2   ranks=(1,1)  qdrant=1,000000  = 1/2 + 1/2   → k=1
code    ranks=(2,2)  qdrant=0,666667  = 1/3 + 1/3   → k=1
budget2 ranks=(3,4)  qdrant=0,450000  = 1/4 + 1/5   → k=1
football ranks=(5,None) qdrant=0,166667 = 1/6       → k=1
```

Bản của ta ở `k=1` cho **cùng tập điểm** với Qdrant (`rel=1e-6` — trường `score`
của Qdrant là float32) và **cùng thứ tự** trên đoạn trước chỗ bằng điểm đầu tiên.
Có test integration canh cả ba điều đó.

Hai điều rút ra:

1. **Bản cài của ta đúng.** Trùng khít với một bản độc lập là bằng chứng mạnh hơn
   bất kỳ số tính tay nào tôi tự đặt ra.
2. **Và vẫn phải tự cài.** `k` của Qdrant cố định — không cấu hình được. §3 cho
   thấy `k` là cần điều khiển *quan trọng nhất*, đưa nDCG@10 từ 0,4021 lên 0,4557.
   Dùng bản server-side thì `W2-08` không quét được nó, và `FusedItem.ranks`
   (thứ hạng từng nhánh) cũng không lấy ra được.

⚠️ Phép đối chiếu **không** so được thứ tự trong nhóm bằng điểm, và bản đầu của
test đỏ ngẫu nhiên vì lý do đó. Corpus test có điểm trùng thật (hai chunk chia
đúng cùng tập token thì điểm sparse bằng nhau; chunk không trùng token nào thì
điểm dense bằng 0), và thứ tự trong nhóm bằng điểm không thuộc hợp đồng của Qdrant
— nó đi qua đường `prefetch` chứ không phải đường `query` thường. Đã dựng lại
corpus để truy vấn đối chiếu cho điểm **phân biệt hoàn toàn** ở cả hai nhánh.

---

## 6. ⭐ Bug 64 ms, và hai con số đã công bố bị sai

Đây là phần đáng nhất của phiên, và nó đến từ việc **các con số không cộng lại
đúng**.

`W2-03` §8 báo: "tìm trong index sparse 97,8 ms vs dense 17,8 ms (5,5×)". Ở
`W2-04` tôi thấy hybrid — làm **nhiều việc hơn** sparse (hai phép tìm thay vì
một) — có p50 **31,3 ms**. Hai số đó không thể cùng đúng. Song song server-side
giải thích được nhiều nhất là "hybrid bằng nhánh chậm", không giải thích được
"hybrid nhanh hơn nhánh chậm 3 lần".

Phân rã đầy đủ trong **một** tiến trình, xen kẽ, sau warm-up chung:

| bước | p50 ms |
|---|---:|
| embed truy vấn (`embed_query`) | 12,7 |
| embed truy vấn (`embed_query_hybrid`) | 12,7 |
| Qdrant tìm dense (`query_points`) | 28,7 |
| **Qdrant tìm sparse (`query_points`)** | **15,4** |
| **Qdrant tìm CẢ HAI (`query_batch_points`)** | **30,2** |
| dựng 20 `Chunk` từ payload | 0,1 |
| `retrieve()` dense đầy đủ | 33,6 |
| `retrieve()` sparse đầy đủ **trước khi sửa** | **109,3** |
| `retrieve()` sparse đầy đủ **sau khi sửa** | **30,4** |
| `retrieve()` hybrid đầy đủ | 30,6 – 38,8 |

Phép kiểm quyết định: cộng các phần lại.

```
dense   phần 41,4 ms · trực tiếp 33,6 ms · lệch  −7,8 ms   (trong nhiễu)
hybrid  phần 42,9 ms · trực tiếp 38,8 ms · lệch  −4,1 ms   (trong nhiễu)
sparse  phần 28,1 ms · trực tiếp 109,3 ms · lệch +81,7 ms  ← không phải nhiễu
```

**81,7 ms không thuộc thành phần nào.** Khác biệt duy nhất giữa `retrieve_sparse`
và đường hybrid: nó kiểm `self.writes_sparse` ở **mỗi lần gọi**, còn hybrid kiểm
một lần lúc dựng. Và `writes_sparse` đọc `embeddings.sparse_vocab_size`, vốn là:

```python
return len(self.model.tokenizer)     # ← 64 ms
```

`len(tokenizer)` gọi `get_vocab()`, tức **dựng lại một dict 250.002 phần tử**. Đo
được **63,9 ms mỗi lần gọi**; `tokenizer.vocab_size` cho cùng con số trong 0,001
ms nhưng khác nghĩa (không tính token thêm vào), nên bản sửa là **nhớ kết quả**
chứ không phải đổi sang thuộc tính nhanh hơn.

### Hai con số đã công bố phải sửa

**`W2-03` §8 sai về quy kết.** Sự thật ngược lại: **tìm sparse (15,4 ms) RẺ HƠN
tìm dense (28,7 ms)**, và gửi cả hai trong một request batch tốn **30,2 ms** —
tức bằng dense một mình, vì Qdrant chạy song song và nhánh sparse lọt hết vào bên
trong nhánh dense. Kết luận "nhánh sparse sẽ là thành phần nặng nhất của đường
truy hồi" là **sai**.

**`W2-02` "sparse gần như miễn phí: +8,8 s trên 389 s"** — đúng kết luận, sai cơ
chế, và chi phí thật còn **thấp hơn**. `upsert` đọc `writes_sparse` một lần mỗi
lô: 15.814 chunk / `batch_size` 128 = **124 lô** × 64 ms ≈ **7,9 s**. Đã build
lại để đo: **389,2 s → 379,1 s**, tức sparse phía ghi thật sự **miễn phí** (§9).

### Bài học, và nó khác hai bài học trước

`TD-11`, `W2-02`, `W2-03` đều là "hệ thống chạy, số ra, không ai biết là sai" —
và cả ba được phát hiện bằng cách **kiểm một bất biến**. Cái này khác: nó được
phát hiện bằng cách **buộc các con số cộng lại đúng**. Ba harness khác cấu trúc
cho ba câu trả lời lệch nhau 2–6× cho cùng một việc; khi phương sai giữa các cách
đo lớn hơn hiệu ứng muốn quy kết, mọi quy kết đều vô nghĩa — và đó là dấu hiệu
phải đo lại chứ không phải chọn con số vừa mắt.

Có test hồi quy (`test_sparse_vocab_size_is_memoised`), canh bằng **thời gian**
chứ không bằng số lần gọi: điều phải giữ là "đọc thuộc tính này rẻ", và một bản
cài khác vẫn có thể vi phạm nó theo cách khác.

---

## 7. Bốn quyết định thiết kế, và số đo đứng sau mỗi cái

**Embed truy vấn một lần, không hai.** Gọi `store.retrieve()` rồi
`store.retrieve_sparse()` sẽ chạy forward pass hai lần — 12,7 ms mỗi lần, tức
+12,7 ms cho đúng một kết quả. Đây là phiên bản phía truy vấn của quyết định "một
forward pass" ở `W2-01`. Có test integration đếm số lần gọi provider, **và** một
test GPU đếm số `_forward` bên trong provider (`BgeM3EmbeddingProvider.
embed_query_hybrid` chạy đúng một).

Bản đầu của test integration đếm cả `embed_query` và **đỏ**: `HashingEmbedding
Provider.embed_query_hybrid` gọi lại `embed_query` bên trong. Đó là nội bộ
provider, không phải việc của retriever — nên test được chia đúng hai tầng thay vì
nới assertion cho nó xanh.

**Một request HTTP cho cả hai nhánh.** `query_batch_points` để Qdrant chạy song
song. §6 cho thấy điều đó không chỉ tiết kiệm một round trip mà làm nhánh sparse
**miễn phí hoàn toàn** (30,2 ms cho cả hai vs 28,7 ms cho dense một mình).

**Hàm hợp nhất là hàm thuần, không nhận điểm.** `reciprocal_rank_fusion(rankings,
k, weights, limit)` — không có tham số `scores`, và có test canh chính chữ ký đó.
Dense cho cosine ∈ [−1,1], sparse cho dot product không trần (đo thật: 0,6682 vs
0,2938); mọi phép chuẩn hoá đều phải chọn một cửa sổ, tức đưa vào một tham số ẩn
phụ thuộc kết quả. Cách chắc nhất để giữ tính bất biến theo thang là **không cho
điểm đi vào hàm**.

**Tie-break là quy tắc, không phải tình cờ.** Điểm bằng nhau xảy ra *thường
xuyên* — một chunk ở hạng 3 của dense và một chunk khác ở hạng 3 của sparse có
cùng điểm. Thứ tự: (1) điểm, (2) `best_rank`, (3) danh sách nào tìm ra trước —
người gọi đặt **dense trước** vì `W2-03` đo được nó mạnh hơn, (4) khoá theo chữ.
Quy tắc (4) hầu như không bao giờ tới, nhưng nó là thứ biến "gần như xác định"
thành "xác định".

---

## 8. Known-item: hybrid giữ vùng phủ, làm hỏng thứ hạng

`W2-03` để lại câu hỏi: RRF có **giữ** được chỗ sparse thắng (tra mã tài liệu)
hay pha loãng nó? Chạy lại `scripts/known_item_probe.py`, giờ đo cả ba nhánh, 51
mã tài liệu:

| | dense | sparse | hybrid k=60 | hybrid k=1 |
|---|---:|---:|---:|---:|
| hit@10 | 0,0784 | **0,5098** | 0,5098 | 0,4706 |
| hit@1 | 0,0196 | **0,3529** | 0,0980 | 0,1373 |
| MRR | 0,0276 | **0,4134** | 0,1995 | 0,2696 |
| hạng trung vị | 6,5 | **1,0** | 4,0 | 2,0 |

hybrid vs sparse ở `k=60`: chỉ sparse 3 · chỉ hybrid 3 · `p = 1` — **vùng phủ
giống nhau về mặt thống kê**. Nhưng thứ hạng thì tệ hơn hẳn: hit@1 **0,3529 →
0,0980**, hạng trung vị 1 → 4.

Đây là mặt bù cố hữu của hợp nhất theo thứ hạng với trọng số đều: nó **không biết
nhánh nào đáng tin cho truy vấn nào**. Trên truy vấn tra mã, dense đóng góp toàn
nhiễu (hit@10 = 0,0784) và RRF vẫn cho nó ngang quyền. `k=1` giảm nhẹ thiệt hại
(hạng trung vị 4 → 2) nhưng không xoá được.

**Kết luận kiến trúc, và nó khớp với §4:** hybrid là một **bộ sinh ứng viên** tốt
(mở rộng vùng phủ: `recall@20` +0,0431 có ý nghĩa, known-item giữ nguyên hit@10)
và một **bộ xếp hạng cuối** tệ (`hit_rate@1` đứng im, known-item MRR tụt một
nửa). Đó chính là hình dạng bài toán mà `W2-05` (cross-encoder reranker) tồn tại
để giải: cho nó vùng phủ rộng, để nó lo thứ tự.

Thứ **không** giải được bằng reranker: chọn nhánh theo truy vấn. Một bộ định
tuyến ("truy vấn này là tra mã → tin sparse") sẽ giữ được cả hai, nhưng đó không
phải RRF và không nằm trong `W2-04`.

---

## 9. Chi phí đo lại sau khi sửa bug: sparse thật sự miễn phí

Build lại `rag_bgem3` với **cùng** config, chỉ khác là đã sửa bug §6:

| | embed + ghi | thông lượng |
|---|---:|---:|
| `W2-01` dense-only | 380,4 s | 39,0 chunk/s |
| `W2-02` dense + sparse (**có bug**) | 389,2 s | 38,2 chunk/s |
| **`W2-04` dense + sparse (đã sửa)** | **379,1 s** | **39,3 chunk/s** |

Dense + sparse sau khi sửa (**379,1 s**) **nhanh hơn** dense-only của `W2-01`
(380,4 s) — tức chi phí thật của nhánh sparse phía ghi nằm **trong nhiễu đo**.
Không phải "+2,3%", mà là **0%**.

Số học ở §6 dự đoán tiết kiệm 7,9 s (124 lô × 64 ms); đo được **10,1 s**
(389,2 − 379,1). Phần lệch là dao động thông lượng giữa các lần chạy — chiều và
độ lớn thì khớp.

Điều này làm câu chuyện của `W2-01` mạnh hơn chứ không yếu đi: quyết định "một
forward pass" khiến sparse **hoàn toàn** miễn phí phía ghi, và §6 cho thấy nó cũng
miễn phí phía đọc (Qdrant chạy hai nhánh song song, 30,2 ms cho cả hai vs 28,7 ms
cho dense một mình). Cái +2,3% từng được kể như "cái giá rẻ phải trả" thực ra
không phải cái giá của sparse — nó là cái giá của một lời gọi `len()`.

**Index không đổi một chữ số sau khi build lại**: 209/209 câu, **0 câu đổi điểm**,
0 câu đổi nhãn. Fingerprint vẫn `0eaaf9265487eabb`. Nên mọi con số §3–§4 và §8 đo
trên cùng một index, trước và sau khi sửa.

---

## 10. Kết quả âm và giới hạn

**`k=60` — mặc định của bài báo gốc — là lựa chọn tệ nhất trong khoảng đã quét**,
và nó kém dense một mình có ý nghĩa (`hit_rate@5` `p = 0,014`; nDCG@10 CI95 không
chứa 0). Nếu tôi cài RRF với mặc định của bài báo, không quét `k`, và báo cáo "đã
làm hybrid search", thì kết luận sẽ là một **suy giảm** được trình bày như một
tính năng.

**`hit_rate@1` không cải thiện ở bất kỳ cấu hình nào.** Dự đoán của tôi là RRF sẽ
thắng ở đây (đồng thuận được cộng điểm). Sai: 0,3397 → 0,3397 ở `k=1`, và tụt ở
mọi `k` lớn hơn.

**Hai test đơn vị tôi viết để biện minh `candidate_k` sâu có số học đúng nhưng
tiên đề sai.** Chunk sâu-mà-đồng-thuận thật sự được đẩy lên; điều test không nói
được là chúng thường không liên quan. Tôi giữ cả hai test (chúng đặc tả đúng hành
vi của hàm) và thêm ghi chú trỏ vào §3 — một test đúng về hành vi vẫn có thể bị
đọc thành một lời khuyên sai.

**Weighted RRF không đáng quét ở `W2-08`** — cần tỉ lệ > 30,5:1 mới lật được một
sự đồng thuận cùng độ sâu.

**Giới hạn của mọi con số §3–§4:** đo trên `golden_v1`, vẫn **review bằng model**
(`TD-13`). So sánh *tương đối* giữa các cấu hình hợp lệ; con số *tuyệt đối* chưa
được gọi là "human-verified". Và mức cải thiện của cấu hình thắng (+2,9 điểm
`hit_rate@10`) **nhỏ hơn** ngưỡng phân giải của tập đo (≥ 6 điểm), nên thứ tự
giữa `k=1`, `k=2`, `k=5` **không** phân biệt được — đừng đọc bảng §3 như một xếp
hạng tinh.

**Giới hạn của §8:** 51 mã, một corpus 60 tài liệu World Bank. Chiều của kết quả
(hybrid giữ phủ, mất thứ hạng) thì rõ; độ lớn thì chưa chắc giữ ở nơi khác.

---

## 11. Việc tiếp theo

1. **`W2-05` cross-encoder reranker** — §8 cho thấy đây là mảnh còn thiếu, và có
   số cụ thể: hybrid `k=1` cho `recall@20` **0,6754** để reranker làm việc trên
   đó, so với 0,6324 của dense. Nhiệm vụ của nó là biến vùng phủ ấy thành thứ hạng.
2. **Chốt mặc định `k=1, candidate_k=20`** cho `W2-08`, và **không quét
   `weights`** (§3). Quét `k` thì chỉ cần vùng nhỏ: `k ∈ {0, 1, 2, 5}` — từ 10
   trở lên đã thấy rõ là tệ hơn.
3. **`TD-18`** (khớp đúng cho mã tài liệu) vẫn mở và §8 làm nó nặng thêm: RRF
   *không* sửa được nó, và còn làm thứ hạng của nhánh duy nhất giải được nó tệ đi.
4. **Ý tưởng đáng cân nhắc, chưa lên kế hoạch:** định tuyến theo truy vấn. §8 cho
   thấy tổn thất đến từ việc trọng số cố định trên một tập truy vấn không đồng
   nhất. Nhưng nó cần một bộ phân loại truy vấn, tức thêm một thứ phải eval — nên
   nó là câu hỏi của W3+ chứ không phải một cờ thêm vào `W2-04`.

## Lệnh tái lập

```bash
make test                                  # 40 ca RRF thuần + 34 ca hybrid
make up && make test-integration           # 25 ca hybrid trên Qdrant thật
make test-gpu                              # gồm test hồi quy bug 64 ms

make eval-retrieval BUNDLE=bgem3 MODE=hybrid RUN=bgem3-rrf-k1-c20 \
  RRF_ARGS="--rrf-k 1 --candidate-k 20"
make eval-compare BASE=bgem3 CAND=bgem3-rrf-k1
make known-item BUNDLE=bgem3               # §8, thêm --rrf-k 1 cho cột cuối
```
