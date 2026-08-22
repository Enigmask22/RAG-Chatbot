# `W3-07` — Re-index tăng dần: mượn lại vector, và ranh giới mà nó không vượt được

> **Trạng thái:** xong · **Ngày:** 2026-08-22 · **Nhánh:** `feat/w1-foundation`
> **Code:** `packages/rag_core/retrieval/qdrant_store.py` (`upsert_reusing`, `fetch_vectors`),
> `pipeline/indexing/build_index.py` (`_reuse_map`, `DocState.chunk_hashes`)
> **Test:** `tests/integration/test_incremental_reindex.py` (7) · **Lệnh:** `make incr-probe`

## 0. DoD, và tầng đã có sẵn không giải quyết nó

DoD: *sửa 1 trang trong 100 trang → chỉ embed lại chunk bị ảnh hưởng.*

`build_index` đã bỏ qua tài liệu có `Document.content_hash` không đổi từ `W1-08`.
Nhưng sửa **một trang** thì hash của cả tài liệu đổi, nên tài liệu bị chunk lại
**và embed lại toàn bộ**. Với tài liệu dài nhất corpus (990.826 byte, 1.082
chunk) đó là **1.082 chunk cho một dòng sửa**.

Cái thiếu là một tầng nữa, ở mức **chunk**.

## 1. Dự đoán ghi trước khi làm

| # | Dự đoán | Kết quả |
|---|---|---|
| E1 | Nhớ `content_hash` từng chunk là đủ cho DoD | ✅ đúng — sửa tại chỗ: mượn lại ~99% (§3) |
| E2 | Khớp theo nội dung cứu được cả ca **chèn** vào giữa | ❌ **sai như đã phát biểu**, và chỗ sai là phần hay nhất: nó **có điều kiện** — 98,0% khi văn bản có xuống dòng đoạn, **2,0%** khi không (§4) |
| E3 | Phải dựng một cache vector riêng (file/SQLite) | ❌ **sai** — Qdrant đã giữ vector rồi, đọc lại là đủ (§2) |
| E4 | Corpus có nhiều chunk trùng nội dung, nên cache liên tài liệu đáng làm | ❌ **sai** — 51/15.814 chunk, **0,32%** (§6) |
| E5 | Fixture "100 trang, sửa 1 trang" là phép đo hợp lệ | ❌ **sai** — nó đo một chế độ chunking khác hẳn, và cho 99% ở **mọi** ca (§5) |
| E6 | `G3` (nhanh hơn ≥10×) sẽ đạt | ✅ đúng, và dư rất xa — **179,3×** trên corpus thật (§3) |

**1 đúng rưỡi, 4 sai, 1 đạt vượt xa.** Cái sai ở `E2` hoá ra là phát hiện chính của cả hạng mục (§4).

## 2. Không dựng cache vector — Qdrant đã là cache rồi

Phản xạ đầu là dựng một cache `(model, content_hash) → vector`. Với 15.814 chunk
× 1024 chiều float32 thì đó là **65 MB** trên đĩa, cộng một vòng đời cache nữa
phải bảo trì (TTL, eviction, version schema — đúng những thứ `chunking/cache.py`
đã phải làm).

Nhưng vector **đã nằm sẵn trong Qdrant**, gắn với point cũ. Cái thiếu chỉ là
đường đọc lại nó. Nên:

* `QdrantDenseRetriever.fetch_vectors(chunk_ids)` — lấy named vector theo id.
* `upsert_reusing(chunks, reuse={chunk_id mới: chunk_id cũ})` — lấy trước **một
  lần** cho cả lượt, embed những chunk còn lại, rồi ghi.
* `DocState.chunk_hashes` — `content_hash` từng chunk theo thứ tự, nguồn để dựng
  bản đồ `reuse`.

Không file mới, không schema mới, không backfill. Cùng khuôn quyết định với
`W3-05` (§2 ở đó): trước khi thêm một tầng lưu trữ, hỏi xem thứ mình cần đã nằm
đâu đó chưa.

Ba chi tiết đúng đắn phải giữ:

* **Lấy vector trước, ghi sau.** Nếu lấy xen kẽ với ghi thì một chunk mượn vector
  của point vừa bị chính lượt này ghi đè.
* **Mượn hụt thì rơi về embed**, và đếm vào `embedded` — không phải lỗi (point có
  thể đã bị xoá), nhưng im lặng bỏ qua chunk đó thì nó biến mất khỏi index.
* **Mượn cả named vector**, không riêng `dense`. Chép mỗi dense sẽ tạo point
  thiếu `sparse`: nhánh sparse im lặng trả rỗng cho đúng những chunk ấy, số point
  vẫn đủ, không gì báo. Có test riêng cho chuyện này.

## 3. Sửa tại chỗ — DoD đạt

`make incr-probe` chép corpus sang thư mục tạm, build ba lượt vào một collection
dùng một lần, và sửa **một dòng** ở giữa tài liệu dài nhất
(`wb-618951468329964080`, 990.826 byte — chèn 62 byte):

| lượt | embed | mượn lại | giây embed | tổng |
|---|---:|---:|---:|---:|
| sạch | 15.814 | 0 | 376,6 | 397,5 |
| không sửa gì | **0** | 0 | 0,0 | 0,6 |
| **sửa một dòng** | **6** | **1.076** | **2,1** | **2,8** |

Tài liệu ấy có 1.082 chunk. Sửa một dòng phải embed lại **6** trong số đó —
**0,55%**. So với build sạch: embed ít hơn **2.636×**, nhanh hơn **179,3×**.

Lượt "không sửa gì" cho 0 embed như phải thế: đó là tầng `W1-08` chứ không phải
tầng mới, và nó có mặt trong bảng để chứng minh con số của lượt ba **không** chỉ
là tầng cũ đội lốt.

Model được nạp **một lần** rồi dùng lại cho cả ba lượt (`build_index(...,
embeddings=...)`). Tính 2,2 GB trọng số vào `tổng giây` sẽ làm mọi tỉ lệ vô
nghĩa — 397,5 s so với 2,8 s là thời gian **làm việc**, không phải thời gian khởi
động.

## 4. ⭐ Ranh giới thật: mượn lại được đúng phần đứng TRƯỚC điểm sửa

Đây là chỗ tôi đoán sai, và đoán sai theo hướng lạc quan.

Lập luận ban đầu: `chunk_id` thuần vị trí nên chèn thêm chữ làm mọi chỉ số phía
sau dịch, và khớp theo **nội dung** sẽ nhìn xuyên qua chuyện đó. Nửa đầu đúng.
Nửa sau sai, vì splitter đóng gói **tham lam** theo thứ tự: chèn một câu làm mọi
chunk phía sau *gói lại khác đi*, nên nội dung chúng **thật sự khác** — không
phải chỉ đổi chỉ số. `content_hash` không cứu được, và **không nên** cứu: các
chunk ấy đúng là văn bản mới.

Đo trên 300 câu ngắn, chunk gói 3 câu:

| ca | mượn lại |
|---|---:|
| sửa tại chỗ (không đổi ranh giới) | ~99% |
| nối thêm vào cuối | ~98% |
| chèn ở **giữa** (câu 150/300) | **51,5%** |
| chèn ở **đầu** (câu 5/300) | **2,0%** |

51,5% ở ca giữa không phải ngẫu nhiên: nó **chính là** phần tài liệu đứng trước
điểm chèn.

### Nhưng corpus thật cho 0,55%, không phải 50%. Hai con số này phải hoà giải được.

§3 chèn 62 byte vào **giữa** một tài liệu 990.826 byte và chỉ 6/1.082 chunk phải
embed lại. Theo bảng trên thì lẽ ra phải hỏng một nửa. Chênh lệch ấy không được
phép bỏ qua — nó nghĩa là tôi chưa hiểu cơ chế.

Đo thẳng vào nghi ngờ: cùng 300 câu, cùng điểm chèn (câu 5/300), chỉ khác **có
xuống dòng đoạn hay không**:

| văn bản | mượn lại |
|---|---:|
| một mạch, không có `

` | **2,0%** |
| `

` mỗi 9 câu | **98,0%** |
| `

` mỗi 3 câu | **98,0%** |

Cơ chế: `separators` là `("

", "
", ". ", " ", "")` **theo thứ tự ưu tiên**,
và splitter đệ quy luôn cắt ở separator ưu tiên cao nhất còn dùng được. Nên mỗi
`

` là một **điểm đồng bộ lại**: đóng gói tham lam khởi động lại từ đó, và
thiệt hại của một lần chèn bị chặn trong khoảng cách tới lần xuống dòng đoạn kế
tiếp.

Phát biểu đúng, thay cho "mượn lại được phần đứng trước điểm sửa":

> Thiệt hại bị chặn bởi **khoảng cách tới separator ưu tiên cao nhất kế tiếp**.
> Văn bản có cấu trúc đoạn ⇒ hỏng cục bộ. Văn bản một mạch ⇒ hỏng tới cuối.

Điều đó nói ra loại tài liệu mà kỹ thuật này **không** giúp được: đầu ra OCR đã
mất cấu trúc dòng, bảng biểu chuyển thành văn xuôi, transcript không chấm câu —
đúng những thứ `W3-02` và `TD-23` đang bàn.

Lối ra cho ca xấu ấy không phải tra hash tinh hơn mà là **chunking theo nội dung**:
ranh giới do một hàm hash cục bộ quyết định (kiểu FastCDC/Rabin), tức mọi vị trí
đều là điểm đồng bộ tiềm năng chứ không chỉ chỗ có `

`. Đó là một chiến lược
chunking mới, không phải một tối ưu của `W3-07` → `TD-28`.

## 5. ⚠️ Fixture đầu tiên cho 99% ở MỌI ca, và con số đó vô nghĩa

Fixture đầu dựng "100 trang", mỗi trang 146 ký tự, `chunk_size=200`. Mọi ca —
sửa, nối, chèn — đều cho ~99% mượn lại. Trông như một kết quả tuyệt vời.

Nó vô nghĩa vì **mỗi trang vừa khít một chunk**: hai trang là 292 ký tự, vượt
200, nên splitter luôn cắt đúng ranh giới trang. Ranh giới chunk do *nội dung*
quyết định chứ không do *độ dài tích luỹ*, nên chèn thêm chữ không dịch được gì.
Fixture đo một chế độ mà corpus thật không ở trong (chunk 1000 ký tự ≈ 7 câu).

Lần thứ ba trong `W3`: `W3-01` §2 (PDF hai cột toàn dòng đều nhau), `W3-05` §9
(60 câu giống hệt nhau), giờ là đây. **Một fixture đều tăm tắp đo cái generator,
không đo cái cần đo** — và lần này nó còn cho ra con số *đẹp*, tức khó nghi ngờ
hơn hai lần trước.

## 6. ⭐ Chệch đi một chút: `chunk_size` có thể hoàn toàn vô tác dụng

Fixture đầu ra **10 chunk** cho "100 trang" thay vì ~73. Truy ra thì
`chunk_size=200` không được tôn trọng: `min_chunk_size` mặc định **cũng là 200**,
nên mọi mảnh đều bị coi là quá ngắn và `_enforce_size` gộp chúng tới tận
`max_chunk_size=1500`. Chunk trung bình đo được: **1.460 ký tự — gấp 7,3× con số
đã khai**, không một dòng log nào.

Không config sản phẩm nào dính (đều `chunk_size ≥ 256` với `min` khai tường minh,
hoặc 1000/200), nhưng **hai test tích hợp có sẵn của repo thì có** — chúng vẫn
xanh vì chỉ kiểm số lượng.

Sửa bằng **cảnh báo, không phải lỗi**, và ranh giới ấy có lý do: "cắt mịn rồi
đóng gói tới `max_chunk_size`" là một chiến lược hợp lệ và có test dùng đúng nó.
Cái sai không phải cấu hình mà là **sự im lặng**. Cảnh báo một lần cho mỗi bộ ba
kích thước, vì `_begin_sizing` `model_copy` config lại cho *mỗi tài liệu*.

## 7. Cái cố ý **không** làm: dedupe liên tài liệu

Tên hạng mục có chữ "content-hash dedupe", nên phải trả lời: corpus có bao nhiêu
chunk trùng nội dung, và gộp chúng lại có đáng không?

Đo trên đúng 15.814 chunk của baseline:

| | |
|---|---:|
| chunk | 15.814 |
| `content_hash` riêng biệt | 15.763 |
| hash xuất hiện > 1 lần | 30 (81 chunk) |
| trùng **giữa các tài liệu** | 6 hash |

Gộp hết cũng chỉ tiết kiệm **51/15.814 = 0,32%**, và ba hash lặp nhiều nhất là
các chunk bắt đầu bằng 70 dấu cách. Không đáng một tầng tra cứu liên tài liệu,
càng không đáng rủi ro gán một point cho nhiều `doc_id`. Ghi lại con số để lần
sau không phải đoán lại.

## 8. Test

7 test tích hợp, và điểm đáng nói là **hai lớp đếm độc lập**:

* `BuildReport.n_chunks_embedded` — con số `build_index` tự báo.
* `CountingEmbeddings` — bọc provider, đếm text thật sự đi qua nó.

Một mình con số tự báo không chứng minh được gì: nó là biến đếm của chính đoạn
code đang được kiểm. Nếu `upsert_reusing` báo "mượn lại 99" mà vẫn gọi provider
99 lần thì chỉ lớp thứ hai thấy. Đây là bài học `W3-05` §3 áp sang một trục khác:
*đừng để cả hai bên đối chiếu đều do code đang kiểm sinh ra.*

Một test ghim **giới hạn** (chèn ở đầu ⇒ mượn lại < 10%) thay vì ghim thành công.
Nếu con số đó tốt lên thì chunker đã đổi, và tài liệu này phải đo lại — đó là
điều cần biết, không phải điều cần im.

## 9. Còn lại gì

* `TD-28` (mới): ca chèn chỉ giải được bằng **chunking theo nội dung**.
* `G3` tiêu chí "reprocess nhanh hơn full rebuild ≥ 10×": **đạt, 179,3×** — có số ở
  `plans/reports/probes/w3-07-incremental.json`.
* `IndexState` giờ mang 15.814 hash → file state ~1,1 MB. Chấp nhận được, nhưng
  nó lớn tuyến tính theo số chunk và sẽ thành vấn đề ở corpus lớn hơn nhiều.
