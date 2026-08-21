# `W1-11` (phần máy) — triage golden set + ba phát hiện về retrieval

> Ngày: 2026-08-20 · Nhánh: `feat/w1-foundation`
> Trạng thái: **công cụ xong, đã chạy thật.** Phần review tay 266 câu vẫn thuộc về người.
> Trả xong `TD-09`. Thêm `TD-11` — **phát hiện quan trọng nhất của lượt này**.

---

## 1. Tóm tắt

Xây hai công cụ cho `W1-11` rồi chạy thật lên 266 câu nháp với chính index baseline.
Lượt chạy đó không chỉ xếp hàng đợi review — nó lộ ra ba điều về hệ thống retrieval
mà không phép thử nào trước đó thấy được, trong đó có một lỗi cấu hình âm thầm
làm **15,7% văn bản của corpus không bao giờ tới được vector**.

| Thành phần | Nội dung |
|---|---|
| `pipeline/goldenset/triage.py` | chạy retriever thật → tín hiệu + hàng đợi review + CSV quyết định |
| `pipeline/goldenset/freeze.py` | quyết định của người → `golden_v1.jsonl` + checksum + read-only |
| `rag_core/retrieval/qdrant_store.py` | thêm `fetch_chunks()` — lấy chunk theo id, phát hiện id chết |
| Test | **+71 unit** (409 tổng) · **+5 integration** (38 tổng) |
| Makefile | `goldenset-triage` · `goldenset-freeze` · `goldenset-verify` |

`ruff check`, `ruff format --check`, `mypy --strict` sạch trên 68 file.

---

## 2. Bất đối xứng: điều quan trọng nhất về thiết kế

Dùng retriever để dọn golden set là con dao hai lưỡi. Có hai tín hiệu, và **chúng
không đối xứng** — lẫn chúng là cách chắc chắn nhất để tự thổi phồng baseline ở `W1-13`.

**(A) Câu `unanswerable` mà retriever tìm được chunk điểm cao** → bằng chứng *mạnh*
rằng nhãn sai. "Corpus không trả lời được câu này" là mệnh đề về **corpus**, và một
hit mạnh phản chứng nó trực tiếp. Model sinh câu hỏi chỉ thấy vài chunk trong một
tài liệu; nó không có cách nào biết 59 tài liệu còn lại nói gì. Đây là `TD-09`.

**(B) Câu trả lời được mà retriever *không* tìm ra chunk đã gán** → **không** phải
bằng chứng nhãn sai. Đây đúng là thứ eval tồn tại để đo. Loại chúng ra thì golden
set chỉ còn câu mà hệ thống hiện tại đã trả lời được, recall baseline bị đẩy lên,
và mọi con số "+X%" về sau đo trên một tập đã chọn thiên vị theo đúng hệ thống
đang được đánh giá.

Nên trong code: (A) xếp ưu tiên 1, đề xuất `recheck_category`. (B) xếp ưu tiên **9
— cuối hàng đợi**, đề xuất `accept`, kèm cảnh báo in thẳng vào `queue_v1.md`.
`tests/unit/test_goldenset_triage.py::TestSignalB` canh cả hai:
`test_missed_gold_is_flagged_but_default_stays_accept` và
`test_missed_gold_sorts_to_the_end_of_the_queue`. Ai "cải tiến" theo hướng ngược
lại thì test đỏ.

Ngoại lệ duy nhất của (B): `chunk_id` không tồn tại trong index. Đó không phải câu
khó, đó là con trỏ chết → ưu tiên 0.

### Ngưỡng được hiệu chuẩn, không phải hằng số

"Điểm cao" của cosine phụ thuộc model embedding và corpus; ghim `0.8` là đoán.
Ngưỡng lấy từ **phân bố điểm top-1 của chính những câu trả lời được trong tập này**
(trung vị): **0,5797**. Ngưỡng "quá dễ" là phân vị 0,9: **0,6623**.

### Freeze không đoán hộ

`fix_chunk_ids` mà người review để trống `new_relevant_chunk_ids` thì **báo lỗi**,
chứ không lấy top-1 của retriever điền vào. Lấy top-1 làm nhãn nghĩa là dạy golden
set trả lời đúng theo hệ thống hiện tại. Ba từ vựng cố ý tách rời:

| | Giá trị | Là gì |
|---|---|---|
| `suggested_decision` (máy) | `accept` · `fix_chunk_ids` · `recheck_category` · `recheck_quote` | **câu hỏi** đặt cho người review |
| `decision` (người) | `accept` · `reject` · `edit` | **câu trả lời** |
| ô trống | — | chưa review. **Không** phải `accept` |

Điền `recheck_category` vào cột `decision` bị từ chối: nếu một cờ nghi vấn tự trở
thành quyết định thì cả cơ chế triage chỉ còn là trang trí.

---

## 3. Kết quả lượt chạy thật

```
$ make goldenset-triage
Đọc 266 câu nháp từ data\golden\draft_v1.jsonl
Triage 266 câu (top-k 20) · ngưỡng nghi ngờ 0.5797 · ngưỡng 'quá dễ' 0.6623
  answerable_but_not_retrieved     161
  quote_unverified                  16
  unanswerable_but_retrieved        15
  trivially_easy                     2
  chunk đã gán có trong top-20: 65/226 (28.8%) — KHÔNG phải recall baseline
real 0m24.250s
```

`gold_chunk_missing` = **0**: cả 226 câu trả lời được đều trỏ tới chunk còn tồn tại
trong index. Phép kiểm này chỉ có nghĩa vì index chưa build lại từ lúc sinh nháp;
để nó ở đây là để lần sau đổi cấu hình chunking thì lỗi nổ ra chứ không âm thầm.

### `TD-09` đã trả: 15/40 câu `unanswerable` bị nghi

15 trong 40 câu có điểm top-1 vượt ngưỡng hiệu chuẩn. **Đây là tín hiệu sàng lọc,
không phải phán quyết** — và ví dụ đầu hàng đợi cho thấy đúng vì sao:

> `unanswerable-171a4548af` · điểm top-1 **0,7287**
> "Theo báo cáo 'Vibrant Vietnam' […] tăng trưởng GDP dự kiến của Việt Nam vào năm
> 2030 là bao nhiêu?"
> top-1: *"Điều đó cũng đòi hỏi tốc độ tăng trưởng tương lai còn cao hơn tốc độ
> tăng trưởng đầy ấn tượng trước đây của Việt Nam kể từ thập kỷ 1990…"*

Chunk đó nói về tăng trưởng, đúng chủ đề, điểm cao — nhưng **không** đưa dự báo GDP
2030. Nhãn `unanswerable` có thể vẫn đúng. Đây là lý do code chỉ gắn cờ và bắt người
đọc quyết, thay vì tự đổi nhãn: điểm cao chứng minh *cùng chủ đề*, không chứng minh
*trả lời được*.

Kế hoạch cũ của `TD-09` viết "câu nào ra chunk điểm rất cao thì phân loại lại". Sau
khi chạy thật thì câu đó phải sửa: điểm cao → **đưa lên đầu hàng đợi review**, không
phân loại lại tự động.

---

## 4. Ba phát hiện về retrieval

28,8% chunk-đã-gán-trong-top-20 là con số thấp đáng ngờ, nhất là khi `W1-08` đo
recall@5 = 0,9167 trên golden set giả. Nên tôi không viết nó vào report rồi đi tiếp
— tôi đi tìm nguyên nhân. Ba giả thuyết, ba lần đo.

### 4.1 Model embedding là **đơn ngữ** — xác nhận

`bkai-foundation-models/vietnamese-bi-encoder` dựa trên PhoBERT, không có căn chỉnh
đa ngữ. Chia 226 câu trả lời được theo việc ngôn ngữ câu hỏi có khớp ngôn ngữ tài
liệu chứa chunk đã gán:

| | chunk đã gán trong top-20 |
|---|---|
| câu hỏi **cùng** ngôn ngữ với tài liệu | **58/136 = 42,6%** |
| câu hỏi **khác** ngôn ngữ với tài liệu | **7/90 = 7,8%** |

Chênh **5,5×**. Nhìn theo nhóm thì rõ hơn nữa:

| nhóm | trượt | ghi chú |
|---|---|---|
| `cross_lingual` | 44/46 = **95,7%** | nhóm này *định nghĩa* là hỏi khác ngôn ngữ tài liệu |
| `table_lookup` | 4/4 = 100% | n=4, không suy ra được gì |
| `adversarial` | 27/36 = 75,0% | 14/36 câu là vi→en |
| `factoid` | 51/78 = 65,4% | vi→vi: 20/38 trúng; vi→en: **0/10** |
| `multi_hop` | 19/34 = 55,9% | vi→vi: 9/10 trúng; vi→en: 1/9 |
| `aggregation` | 16/28 = 57,1% | |

Dòng `factoid` là bằng chứng gọn nhất: cùng model, cùng nhóm câu hỏi, chỉ đổi ngôn
ngữ tài liệu → 20/38 thành 0/10.

**Không phải lỗi cần sửa trước `W1-13`.** Baseline có nhiệm vụ đo hệ thống POC *như
nó đang là*, và POC dùng đúng model này. Nhưng nó cho biết trước rằng cột
`cross_lingual` của baseline sẽ gần bằng 0, và con số đó **có lời giải thích**, không
phải điều bí ẩn. Đây cũng chính là lý lẽ định lượng cho `W2-01` (BGE-M3, đa ngữ).

### 4.2 Corpus có tài liệu gần trùng — **bị loại**

Giả thuyết: corpus có 5 bản "Điểm lại" và 3 tập báo cáo ven biển, nội dung lặp qua
các năm, nên chunk top-1 có thể trả lời **tương đương** chunk đã gán → nhãn không
đầy đủ chứ retriever không dở. Đo Jaccard trên tập từ, giữa chunk đã gán và chunk
top-1, trên 78 câu trượt-mà-cùng-ngôn-ngữ:

| Jaccard | số câu |
|---|---|
| ≥ 0,5 (gần trùng) | **0** (0,0%) |
| 0,3–0,5 | 1 (1,3%) |
| 0,15–0,3 | 22 (28,2%) |
| < 0,15 (khác hẳn) | **55 (70,5%)** |

Giả thuyết bị loại. 70,5% chunk top-1 khác hẳn nội dung chunk đã gán, và chỉ 22/78
có top-1 cùng tài liệu. Đây là điểm yếu truy hồi thật, không phải hiện tượng gán nhãn.

Ghi lại phép đo *bị loại* này có chủ ý: không có nó thì con số 42,6% ở mục 4.1 vẫn
còn cách giải thích dễ chịu hơn, và ai đọc report cũng sẽ nghĩ tới nó.

### 4.3 91% chunk bị **cắt** lúc embed — `TD-11`

Phát hiện quan trọng nhất của lượt này.

```
max_seq_length của vietnamese-bi-encoder  = 256 token
chunk_size (baseline.yaml)                = 1000 ký tự
neighbor_context_chars                    = 100
```

`chunk_size` tính bằng **ký tự**, `max_seq_length` tính bằng **token**. Tiếng Việt
có dấu bị BPE của PhoBERT chẻ rất vụn, nên 1000 ký tự ≈ 340 token — vượt xa 256.
`sentence-transformers` **cắt âm thầm**, không cảnh báo, không lỗi.

Đo trên 1200 chunk lấy ngẫu nhiên từ index (seed 20260820):

| | token |
|---|---|
| p10 | 61 |
| p25 | 185 |
| **p50** | **275** |
| p75 | 335 |
| p90 | 365 |
| max | 622 |

> **56,8%** chunk bị cắt · **15,7%** toàn bộ token đem embed bị bỏ

Trên riêng 158 chunk *đã gán* thì tỉ lệ là **91,1%** (p50 = 340 token) — cao hơn vì
bộ lọc "prose-like" của `W1-10` cố ý chọn chunk văn xuôi dài, tránh chú thích biểu
đồ và trang bìa.

Chiều câu hỏi thì an toàn: p50 = 44 token, max = 105, **0/266 bị cắt**. Nên đây là
lỗi một chiều — chunk mất đuôi, câu hỏi nguyên vẹn.

Hệ quả: khoảng một phần tư nội dung của chunk trung vị không tồn tại đối với index.
Nếu câu trả lời nằm ở nửa sau một chunk thì không truy vấn nào tìm ra được nó, và
không có dấu hiệu gì cho thấy điều đó. Đây giải thích phần lớn con số 42,6% ở mục 4.1
sau khi đã trừ yếu tố ngôn ngữ.

**Vẫn không sửa trước `W1-13`.** Đây là hành vi thật của bản POC, và baseline phải
đo hành vi thật đó — sửa trước rồi gọi kết quả là "baseline" thì mọi so sánh về sau
đo lẫn cả cải tiến này vào. Sửa ở `W2`, và đây là hạng mục đầu tiên nên sửa vì nó
rẻ nhất: chỉ cần hạ `chunk_size` xuống ~600 ký tự, hoặc đổi model có cửa sổ dài hơn.

---

## 5. Vì sao hai file, không phải một

Người review **đọc** `queue_v1.md` và **ghi** vào `decisions_v1.csv`.

Gộp thành một JSONL cho người sửa tay thì 266 dòng JSON phải sửa bằng tay — vừa chậm
vừa dễ làm hỏng, và một dấu ngoặc lệch mất toàn bộ công review. Nên `queue_v1.md`
tối ưu cho đọc: mỗi câu có sẵn câu hỏi, đáp án tham chiếu, trích dẫn model viện dẫn
(kèm dấu ✅/⚠️), **toàn văn chunk đã gán**, và top-3 retriever trả về với chunk đã gán
được đánh dấu `⬅️ đã gán`. Không phải tra cứu qua lại — đó là chỗ phần lớn thời gian
review tay bị mất.

`write_decisions_template` **từ chối ghi đè** file đã tồn tại. Mất 6 giờ công review
vì chạy lại một lệnh là chuyện không được phép xảy ra; muốn làm lại thì `--force-decisions`.

Kích thước và cách version:

| file | cỡ | git |
|---|---|---|
| `decisions_v1.csv` | 68 KB | **commit** — công của người, không sinh lại được |
| `triage_summary.json` | 1 KB | **commit** — bằng chứng lần chạy |
| `queue_v1.md` | 604 KB | ignore — sinh lại 25s |
| `triage_v1.jsonl` | 2,1 MB | ignore — sinh lại 25s |

Hai file sinh lại được đều chỉ đúng với **một** index cụ thể, nên commit một bản cũ
còn tệ hơn không commit.

---

## 6. Chuyện phụ: index sống sót qua lần đổi tên

`make index` sau khi đổi tên workspace:

```
tài liệu   60 (index 0 · bỏ qua 60 · gỡ 0)
chunk      ghi 0 · xoá thừa 0 · tổng trong collection 15814
```

Đúng như dự đoán ở `reports/tasks/rename-workspace.md`: volume Docker sống vì
`infra/docker-compose.yml` khai `name: rag-platform` tường minh, và `.cache/` sống vì
`index_state/baseline.json` không chứa đường dẫn nào.

---

## 7. Việc còn lại — thuộc về người

266 câu, ước lượng 6–8 giờ. Thứ tự đã xếp sẵn trong `queue_v1.md`:

1. **15 câu `unanswerable_but_retrieved`** — đọc chunk top-1, nó có *thật sự* trả lời
   câu hỏi không? Cùng chủ đề là chưa đủ.
2. **16 câu `quote_unverified`** — model bịa thì `reject`; chỉ khác khoảng trắng thì `accept`.
3. **2 câu `trivially_easy`** — `accept`. Chỉ đáng lo ở mức phân bố cả tập, mà 2/226 thì không.
4. **161 câu `answerable_but_not_retrieved`** — hành động mặc định là **`accept`**. Chỉ
   `reject` khi bản thân *câu hỏi* dở (tối nghĩa, sai sự thật, trùng ý). Sau mục 4.1 và
   4.3 thì đã biết vì sao chúng trượt, và cả hai nguyên nhân đều **không** phải lỗi của
   câu hỏi.
5. 72 câu không cờ nào — đọc nhanh.

Rồi:

```bash
make goldenset-freeze     # đòi ≥150 câu và đủ 7 nhóm, nếu không thì từ chối
make goldenset-verify     # đối chiếu checksum
```

`table_lookup` chỉ có 4 câu. Nếu review loại mất cả 4 thì freeze sẽ **từ chối** vì
thiếu nhóm — và đó là hành vi đúng: golden set thiếu một nhóm thì breakdown ở eval im
lặng bỏ qua nó, và không ai nhận ra hệ thống chưa bao giờ được đo ở đó. Nhóm này phải
chờ nguồn (c) HOSE + `W3-01` mới đủ dày.

---

## 8. Lệnh tái lập

```bash
make up && make index                    # xác nhận: "index 0 · bỏ qua 60"
make goldenset-triage                    # ~25s, cần GPU cho embedding
# đọc data/golden/review/queue_v1.md, điền data/golden/review/decisions_v1.csv
make goldenset-freeze
```

---

## 9. Phát hiện thứ tư, tìm ra sau khi đã commit: `TD-12`

`TD-11` kết luận rằng việc đầu tiên của `W2` nên là hạ `chunk_size`. Nên tôi kiểm
xem điều đó làm gì với golden set sắp đóng băng.

`chunk_id` được sinh ở `packages/rag_core/chunking/base.py:144`:

```python
chunk_id=f"{doc.doc_id}::{index:05d}"
```

Thuần **vị trí**. Không có offset ký tự nào được lưu — `Chunk` có `chunk_index`,
`section_path`, `parent_chunk_id`, `token_count`, nhưng không có `start_char`/`end_char`.

Chunk lại cùng một tài liệu ở hai `chunk_size`:

```
tài liệu: wb-099000007072236813 · 40.508 ký tự
chunk_size=1000 → 46 chunk
chunk_size=600  → 69 chunk

chunk_id tồn tại ở CẢ HAI cấu hình: 46
  trong đó nội dung GIỐNG nhau : 0
  nội dung KHÁC nhau           : 46
chunk_id chỉ có ở bản 1000 (biến mất): 0
```

`wb-099000007072236813::00023`:

| | 110 ký tự đầu |
|---|---|
| `size=1000` | `'n. Significant investments\nwill be needed to upgrade and retrofit public assets…'` |
| `size=600` | `' exported rice 65% of aquaculture production 60% of exported fish…'` |

**Đây là dạng hỏng tệ nhất có thể.** Không phải con trỏ chết — con trỏ chết thì
`fetch_chunks` phát hiện và `gold_chunk_missing` gắn cờ. Đây là con trỏ **sai**: id
vẫn tồn tại, vẫn lấy ra được một chunk, chunk đó vẫn được chấm điểm recall. `0
chunk_id biến mất` nghĩa là mọi phép kiểm hiện có đều báo sạch.

Hệ quả cụ thể: `golden_v1` đóng băng hôm nay, `W2` hạ `chunk_size` theo `TD-11`, và
từ đó recall được đo so với ground truth sai — trong khi `make goldenset-verify` vẫn
xanh, vì checksum canh nội dung *file golden set*, không canh việc nội dung *index*
mà nó trỏ tới có còn như cũ.

Đây là lỗi thiết kế của tôi ở `W1-07`, không phải của bản POC: tôi chọn `chunk_id`
thuần vị trí mà không nghĩ tới việc cả `W2` và `W3` đều là về việc đổi cách chunk.

### Hai đường đi

**(a) Review bằng `chunk_id` như hiện tại, ánh xạ lại ở `W2`.** Ánh xạ được vì draft
giữ `supporting_quotes` đã đối chiếu với chunk — 250/266 câu có trích dẫn xác minh
được, tìm chunk mới chứa trích dẫn đó là xong. 16 câu còn lại làm tay. Không phải
sửa gì bây giờ, bắt đầu review ngay được.

**(b) Thêm `start_char`/`end_char` vào `Chunk`, gán nhãn theo span ký tự.** Golden set
trở thành *độc lập với cấu hình chunking*: eval nhận span rồi tự ánh xạ sang chunk
của bất kỳ index nào đang được đo. Chi phí máy nhỏ (build lại index 202s + triage
25s), nhưng phải sửa `rag_core.schemas.Chunk`, cả bốn chunker, và thêm bước ánh xạ
span→chunk vào `retrieval_eval`. Bù lại thì `W2`, `W3` và mọi lần đổi chunking sau
này không phải ánh xạ lại lần nào nữa.

Khuyến nghị: **(b)**. Toàn bộ `W2` và `W3` là về việc đổi cách chunk; một golden set
phải ánh xạ lại sau mỗi lần đổi đó là nợ phải trả nhiều lần, và mỗi lần đều là một
cơ hội để ground truth lệch âm thầm.
