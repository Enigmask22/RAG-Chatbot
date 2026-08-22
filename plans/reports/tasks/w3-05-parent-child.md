# `W3-05` — Small-to-big: parent không nằm trong index, và một Protocol nói dối suốt 27 test

> **Trạng thái:** xong · **Ngày:** 2026-08-22 · **Nhánh:** `feat/w1-foundation`
> **Code:** `packages/rag_core/chunking/parent_child.py`, `packages/rag_core/retrieval/context.py`
> **Config:** `configs/indexing/pc256.yaml`, `configs/indexing/pc128.yaml`
> **Test:** `tests/unit/test_parent_child.py` (29) · **Lệnh:** `make pc-probe PCCFG=pc256 PCK=10`

## 0. DoD và cách nó được đọc

DoD: *"retrieve child → context assembly trả parent, dedupe parent trùng."*

Ba mệnh đề, đo riêng, trên index thật `rag_pc256` (10.473 child, 60 tài liệu,
242 câu golden):

| mệnh đề | đo được |
|---|---|
| retrieve trả child | 10.473 point đều là child; **0** parent lọt vào kết quả (parent không phải point — §2) |
| assembly trả parent | 242/242 câu dựng lại được parent · **0** parent thiếu anh em trên 968 lượt chạy (§4) |
| dedupe parent trùng | có thật, nhưng **nhỏ**: 10,0% ở k=10 · 61% số câu gộp được ít nhất một lần (§5) |

## 1. Dự đoán ghi trước khi đo

| # | Dự đoán | Kết quả |
|---|---|---|
| E1 | Top-k child chụm mạnh vào ít parent, nên dedupe là phần có giá | ⚠️ **nửa đúng** — có thật ở 61% số câu nhưng chỉ tiết kiệm 10% (§5) |
| E2 | Không index parent thì không phải đụng payload/filter/backfill | ✅ đúng — build sạch, 0 thay đổi schema, 0 con số cũ bị đổi (§2) |
| E3 | Index sạch thì 0 parent thiếu anh em | ✅ đúng — 0/968 lượt (§4) |
| E4 | Ngữ cảnh nở đúng bằng `parent_size_multiple` = 4× | ❌ **sai, và sai ở chỗ sâu hơn** — đo được 3,45×, nhưng **độ nở là chỉ số đánh lừa**: chia đôi child làm nó gấp đôi trong khi prompt không đổi (§6) |
| E5 | 27 unit test + `mypy` sạch là đủ tin bản cài đặt | ❌ **sai nặng** — method được gọi **không tồn tại** ở lớp thật (§3) |
| E6 | `chunk_size=256` token là "small" | ❌ **sai** — child p50 **256 token**, chunk baseline **218 token**: child *to hơn* baseline (§6) |
| E7 | Filter đi xuyên `fetch_chunks` và parent thiếu anh em bị đánh dấu | ✅ đúng — đo trên store thật: 32 anh em bị chặn, 9/10 parent `complete=False` (§7) |

**3 đúng, 1 nửa đúng, 3 sai.**

## 2. Quyết định: parent KHÔNG phải một point

Cách quen thuộc là index cả parent lẫn child rồi lọc parent ra khỏi kết quả tìm
kiếm. Ở repo này cái giá của nó cụ thể được: thêm một field phẳng trong payload
(`qdrant_store._payload`), thêm một field trong `MetadataFilter`, một payload
index, và một lượt backfill cho collection `rag_bgem3` đã build — tức chạm đúng
tầng mà `W2-06` vừa đo latency xong. Và một đường tìm kiếm nào đó quên filter thì
parent lọt vào kết quả, âm thầm.

Thay vào đó **parent là tập các child của nó**: `parent_chunk_id` là một *khoá
gom nhóm* (`{doc_id}::p{k:05d}`), id anh em nằm trong `extra["parent_children"]`,
và assembly dựng lại parent bằng **một** lời gọi `fetch_chunks`.

Đo được: `n_chunks_written` = 10.473, `collection_count` = 10.473 — không point
thừa. Không field payload mới, không `MetadataFilter` mới, không backfill,
`rag_bgem3` không bị đụng.

Hệ quả bắt buộc: **child cùng một parent không chồng lấn** (`chunk_overlap=0`).
Không phải cho tiện — overlap tồn tại để một câu bị cắt đôi vẫn còn nguyên ở một
trong hai chunk, mà đó *đúng là* vấn đề small-to-big giải quyết. Giữ overlap thì
đoạn chồng lấn xuất hiện **hai lần** trong parent ghép lại. Overlap **giữa các
parent** vẫn giữ.

Và `_enforce_size` áp **trong phạm vi một parent**: gộp một child ngắn vào child
liền trước qua ranh giới parent thì chunk sinh ra thuộc hai parent cùng lúc và
`parent_chunk_id` của nó là một lời nói dối. Cùng khuôn với `W3-03`
(`section_path` qua ranh giới section) và với lỗi bản POC gộp qua ranh giới tài
liệu — lần thứ ba của **cùng một hình dạng lỗi**.

## 3. ⭐ Protocol khớp với một cái tên không tồn tại, và 27 test xanh

`context.py` gọi `fetcher.get_by_ids(...)`. Lớp thật `QdrantDenseRetriever`
**không có** method nào tên vậy — method thật là `fetch_chunks`. Lỗi chỉ lộ ra ở
lần chạy probe đầu tiên trên index thật, **sau 328 giây build**:

```
AttributeError: 'QdrantDenseRetriever' object has no attribute 'get_by_ids'
```

Vì sao 27 unit test không thấy: `RecordingFetcher` trong test cũng khai
`get_by_ids`. Fake và Protocol khớp nhau hoàn hảo **trong khi cả hai cùng sai**.
`mypy` cũng sạch, và đúng logic của nó: `expand_to_parents` nhận `ChunkFetcher`,
test truyền một object thoả `ChunkFetcher`, và không có chỗ nào trong code đã
kiểm kiểu nối Protocol này với lớp thật.

Bài học chung, và nó rộng hơn `W3-05`: **một Protocol cấu trúc không ràng buộc
được gì nếu cả hai bên đối chiếu đều do test dựng ra.** Phải có một bên thật.

Chi tiết khó chịu nhất: **khuôn để bịt đã có sẵn từ hạng mục ngay trước.**
`W3-06` khai `TokenCounter` cũng là Protocol, và `test_token_sizing.py:427` ghim
nó vào một lớp thật (`isinstance(HashingEmbeddingProvider(...), TokenCounter)`).
Tôi viết `ChunkFetcher` một ngày sau đó và không dùng lại khuôn ấy. Quét nốt hai
Protocol còn lại của repo (`Tracker`/`RunHandle` ở `pipeline/experiments/tracking.py`)
thì chúng có bên thật trong repo (`NullTracker`\`MlflowTracker`\`SafeTracker`) và
đã bị `W2-07` bắt chạy thật hai lần — nên `ChunkFetcher` là **trường hợp duy
nhất** mà cả hai bên đối chiếu đều do test dựng ra.

Cách bịt: `ChunkFetcher` thành `runtime_checkable`, và hai test ghim nó vào lớp
thật — `isinstance(QdrantDenseRetriever(...), ChunkFetcher)` (khởi tạo được mà
không chạm mạng vì `client` là lazy property) cộng một phép so `inspect.signature`
để bắt cả trường hợp đổi `filters` từ keyword-only sang positional. Tiêm lại lỗi
cũ: **đúng 2 test mới đỏ, 27 test cũ vẫn xanh** — tức khoảng trống được đo, không
phải được đoán.

## 4. Không parent nào bị ghép lặng lẽ

`fetch_chunks` là **đường vòng qua filter** mà `W1-07` đã cảnh báo bằng chữ trong
docstring: lấy chunk theo id không đi qua tầng filter của truy vấn vector. `W3-05`
là chỗ **tiêu thụ đầu tiên** của đường vòng đó, nên `expand_to_parents` bắt buộc
nhận `filters` và chuyển thẳng xuống.

Phần đi cùng, và nó là phần dễ bỏ: khi filter *giấu bớt* anh em, ghép những mảnh
còn lại rồi trả về như một parent nguyên vẹn là **dựng ra một đoạn văn chưa từng
tồn tại** — hai đoạn không liền nhau dán vào nhau, không dấu vết.
`AssembledParent.complete` và `missing_children` nói ra điều đó.

Trên index sạch, 4 lượt chạy × 242 câu = **968 lượt, 0 parent thiếu anh em** —
index và `extra["parent_children"]` khớp nhau.

## 5. ⭐ Dedupe có thật nhưng nhỏ, và cái đắt là token

`make pc-probe` trên 242 câu golden, `rag_pc256`:

| top-k | parent trung bình | dedupe | số câu gộp được ≥1 | token child | token parent | nở |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 2,85 | 5,0% | 36/242 (15%) | 835 | 2.993 | 3,61× |
| 5 | 4,64 | 7,3% | 75/242 (31%) | 1.383 | 4.861 | 3,53× |
| **10** | **9,00** | **10,0%** | **148/242 (61%)** | **2.750** | **9.471** | **3,45×** |
| 20 | 17,33 | 13,3% | 218/242 (90%) | 5.493 | 18.340 | 3,35× |

Đọc bảng này theo hai chiều:

**Dedupe không phải code chết, nhưng cũng không phải phần có giá.** Ở k=10, 61%
số câu gộp được ít nhất một lần, và tỉ lệ tăng đều theo k (15% → 90%). Nhưng nó
chỉ cắt được **1 parent trên 10**. Nếu không gộp thì độ nở sẽ là đúng 4,0×; gộp
kéo xuống 3,45×, tức tiết kiệm **~14% prompt**. Đáng giữ, không đáng gọi là lý do
tồn tại của module.

**Cái đắt là số token parent tuyệt đối.** Ở k=10, prompt đi từ 2.750 lên
**9.471 token** — bật parent expansion mà giữ nguyên k là nhân ngân sách ngữ cảnh
lên ~3,5 lần, và ở nhiều cấu hình serving thì đó là vượt trần, không phải "hơi
tốn".

⚠️ Cột **nở** trong bảng trên trông như thước đo tự nhiên của chi phí, và nó
**không phải**. §6 cho thấy vì sao: chia đôi child làm cột đó gấp đôi trong khi
prompt gần như y nguyên. Con số phải nhìn là **token parent**, không phải tỉ số.

Hệ quả trực tiếp cho `W3-09`: so "k=10 child" với "k=10 parent" là **so hai thứ
khác giá**. Ở cùng ngân sách token, k=3 với parent (2.993 token) mới là đối thủ
của k=10 không parent (2.750 token). Ablation phải ghép cặp theo **token**, không
theo k.

## 6. ⚠️ `pc256` không kiểm được nửa "small" — nên có `pc128`

Luận điểm của small-to-big có hai nửa: mảnh **nhỏ** cho vector đúng chủ đề hơn,
khối **lớn** cho LLM đủ ngữ cảnh. `pc256` chỉ kiểm được nửa sau.

Lý do, đo được: child của `pc256` có p50 **1.384 ký tự**, và với mật độ corpus
(~5,4 ký tự/token, `W3-06` §3) đó là **~256 token** — đúng như khai. Nhưng chunk
của baseline (`bgem3.yaml`, 1000 ký tự) đo được là **218 token** (`W3-06` §2). Tức
"child" của `pc256` **to hơn** chunk baseline 17%. Con số 256 được chọn vì nó
tròn, không vì nó nhỏ.

Nên `pc256` thật ra đo *"child cỡ baseline + parent 4×"*. Đó là một cấu hình hợp
lệ, chỉ không phải cấu hình mà tên kỹ thuật này hứa hẹn.

`configs/indexing/pc128.yaml` chia đôi child (128 token) và nhân đôi
`parent_size_multiple` (8), nên **parent vẫn ~1024 token y hệt** `pc256`. Đúng một
biến thay đổi: độ mịn của child.

Build ra **22.924 child**, p50 **634 ký tự ≈ 117 token** — bằng **54%** chunk
baseline, tức lần này "small" đúng nghĩa. Cùng 242 câu golden:

| index | child p50 | k | parent tb | dedupe | **token child** | **token parent** | nở |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pc256` | ~256 tok | 3 | 2,85 | 5,0% | 835 | **2.993** | 3,61× |
| `pc128` | ~117 tok | 3 | 2,80 | 6,8% | 419 | **3.042** | 7,38× |
| `pc256` | ~256 tok | 10 | 9,00 | 10,0% | 2.750 | **9.471** | 3,45× |
| `pc128` | ~117 tok | 10 | 8,70 | 13,0% | 1.370 | **9.519** | 6,98× |

⭐ **Độ nở là một chỉ số đánh lừa.** Chia đôi child làm "độ nở" **gấp đôi** (3,45×
→ 6,98×) trong khi prompt thật **không đổi**: 9.471 → 9.519 token, lệch **+0,5%**.
Ở k=3 cũng vậy (2.993 → 3.042, +1,6%).

Lý do thì hiển nhiên khi đã thấy: prompt = *số parent riêng biệt* × *cỡ parent*,
và cả hai đều được giữ nguyên có chủ ý. Child không có mặt trong công thức đó.
"Độ nở" chỉ đo child nhỏ đến đâu, không đo cái gì phải trả tiền — nên §5 lẽ ra
phải dẫn bằng **token parent tuyệt đối**, không phải bằng tỉ số.

Hệ quả thực tế, và nó đi ngược trực giác thông thường: **chọn child nhỏ hơn
không đắt hơn.** Chia đôi child cho thêm 3 điểm dedupe (10,0% → 13,0%; 164/242 câu
gộp được, so với 148) và vector mịn hơn, ở **cùng một ngân sách prompt**. Cái giá
nằm chỗ khác — 22.924 point thay vì 10.473, tức gấp đôi bộ nhớ index và 343 giây
build thay vì 328.

Cái vẫn **chưa** biết là nửa còn lại: child mịn hơn có làm truy hồi đúng hơn
không. Đó là `W3-09`, và giờ nó có hai index dựng sẵn để so.

## 7. Filter chặn thật, đo trên store thật

Sau §3 thì một phép kiểm bằng fake không còn đủ để tin. Chạy trực tiếp trên
`rag_pc256`, một câu trúng cả tài liệu EN lẫn VI, rồi mở rộng hai lần:

| | parent | anh em bị chặn | `complete` |
|---|---:|---:|---:|
| không filter | 10 | 0 | 10/10 |
| `MetadataFilter(lang=en)` | 10 | **32** | **1/10** |

9 parent bị đánh dấu `complete=False` với `missing_children` liệt kê đủ id, và
`text` chỉ chứa những child thật sự lấy được (1 child, 1.448 ký tự thay vì ~4).
Không parent nào bị ghép lặng lẽ.

Ranh giới cần nói rõ, và nó đã được ghi vào docstring: `filters` chỉ chắn **đường
lấy thêm anh em**. Các child đã có trong `results` được tin là hợp lệ, vì chúng
đến từ lượt search mà người gọi đã phải lọc rồi. Tầng serving ở `W4` phải truyền
**đúng** filter đã dùng cho lượt search — truyền thiếu thì đây là chỗ rò.

## 8. Cái cố ý **không** làm

* **Không đụng `bgem3.yaml`.** `pc256` là collection riêng. Bộ chunk khác hẳn nên
  nhãn golden ánh xạ khác (`TD-20`: đổi chunking là đổi *tập* nhãn, không chỉ số
  lượng), và mọi con số `W2` sẽ vô hiệu.
* **Không tuyên bố chất lượng truy hồi.** Probe này đo **cấu trúc** — thứ không
  phụ thuộc nhãn. Câu hỏi "small-to-big có truy hồi tốt hơn không" thuộc `W3-09`,
  và §5 vừa cho biết nó phải được thiết kế theo **ngân sách token**, không theo k.
* **Không nối vào `CachedChunker`/serving.** `W4` sẽ gọi `expand_to_parents`; nối
  bây giờ là nối vào một tầng chưa có.

## 9. Test

29 test trong `tests/unit/test_parent_child.py`, chia theo hai kiểu hỏng: chunker
hỏng thì quan hệ cha-con sai **ngay lúc index** (phải build lại mới sửa được);
assembly hỏng thì quan hệ vẫn đúng mà **prompt sai** (sửa không cần đụng index).

Một ghi chú về fixture: bản đầu của `PROSE` lặp đúng một câu 60 lần, và ba phép
kiểm "đoạn văn không xuất hiện hai lần trong parent" trở thành vô nghĩa —
`str.count` đếm được 60 bản sao của mọi thứ. Cùng bài học với fixture PDF hai cột
ở `W3-01` §2: **một fixture đều tăm tắp đo cái generator, không đo cái cần đo.**
Đã thay bằng 60 câu khác nhau.

## 10. Còn lại gì

* `TD-26` (mới): `ParentChildChunker` chưa nằm trong đường `CachedChunker`, và
  `expand_to_parents` chưa được nối vào serving — cùng dạng với `TD-24` của
  `StructureChunker`.
* `TD-25` vẫn mở, và `W3-05` làm nó cụ thể hơn theo hai cách: ablation `W3-09`
  phải ghép cặp theo **ngân sách token**, không theo `top_k` (§5); và nó đã có
  sẵn **hai** index để so (`rag_pc256`, `rag_pc128`) khác nhau đúng một biến.
* Nửa "small" mới được đo về **cấu trúc** (§6): child mịn hơn cho dedupe nhỉnh
  hơn ở **cùng** ngân sách prompt. Nó có làm vector đúng chủ đề hơn không thì
  phải chờ `W3-09` — và đó là câu hỏi duy nhất còn lại của kỹ thuật này.
