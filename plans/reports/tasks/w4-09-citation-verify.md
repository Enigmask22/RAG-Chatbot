# `W4-09` — Structured output + citation verification

> 2026-09-04 · Serving Plane · DoD: output validate bằng pydantic; mỗi `quote`
> phải match trong chunk được cite, không match → `unverified`.
> Test: `tests/unit/test_citation_verify.py` (quote thật, quote bịa, quote sai chunk).
> Số đo thật: `probes/w4-09-citations-real.json` (~$0,0026, 3 lượt `/chat` DeepSeek).

## 0. Câu hỏi của hạng mục, và vì sao nó nằm ở nửa dưới đường phân giới

`W4-06` chia mọi cách hỏng thành hai nửa: trước byte đầu của `200 OK` (còn trả
được HTTP status) và sau nó (chỉ còn khung SSE). Xác minh citation cần **toàn bộ
câu trả lời** — nghĩa là nó chỉ tồn tại ở nửa dưới: một citation bịa không thể
là một `422`, nó chỉ có thể là một khung SSE nói thẳng `verified: false`.

Khung `sources` của `W4-06` cố ý mang tên đó vì nó là *cái đã đưa cho model*.
Hạng mục này thêm khung `citations` — *cái model tuyên bố đã dùng, sau khi đối
chiếu từng quote với đúng chunk nó chỉ vào*. Hai khung, hai nghĩa, và khoảng
cách giữa chúng chính là thứ `W5-02` sẽ đo trên diện rộng.

`text.split("Trả lời:")` của bản POC (`legacy/app.py:213`) không có ở đường
serving mới ngay từ `W4-06` — phần "bỏ hẳn" của DoD được thoả từ trước; phần
có nội dung là **structured block + verification**, và đây là nó.

## 1. Dự đoán ghi trước khi đo

Viết trước khi chạy bất kỳ phép đo nào với model thật:

- **P1** — deepseek-v4-flash `temp=0` sinh block `CITATIONS: [...]` đúng cú
  pháp JSON **ngay lần chạy thật đầu tiên** trên một câu trả lời được.
- **P2** — **Ít nhất một quote không match nguyên văn** (sau chuẩn hoá
  whitespace) trên một câu trả lời có ≥ 2 citation. Hai lý do: model hay "sửa
  nhẹ" khi trích tiếng Việt; và chunk của index `ctx` mang **câu ngữ cảnh tổng
  hợp dán đầu** — model có thể quote câu đó thay vì văn bản gốc.
- **P3** — Câu không trả lời được: model nói "không đủ thông tin" và **bỏ hẳn
  block** (thay vì in `CITATIONS: []` như prompt yêu cầu).
- **P4** — TTFB không đổi so với `W4-06` (~800 ms): block nằm cuối stream.
- **P5** — Không ký tự nào của chuỗi `CITATIONS:` lọt vào bất kỳ khung `delta`
  nào ở lần chạy thật.

**Chấm sau khi đo: 4/5 đúng, P3 SAI** — và cái sai mang tin tốt, xem §4.

## 2. Thiết kế: `n` chứ không phải `chunk_id`, và block là giao thức chứ không phải nội dung

`packages/rag_core/generation/citations.py` — đặt ở `rag_core` vì `W5-02`
(citation accuracy trên diện rộng) sẽ chạy đúng bộ này trên đường eval.

* **Model chép `n`, mình giải `chunk_id`.** Model đã thấy ngữ cảnh đánh số
  `[1]`, `[2]`… và vốn phải chép số ấy vào câu trả lời từ `W4-06`. Bắt nó chép
  một UUID 36 ký tự là thêm một chỗ chép sai mà không mua được gì — ánh xạ
  `n → chunk_id/doc_id/source_url` nằm ở phía mình, tất định.
* **Match sau chuẩn hoá whitespace, GIỮ nguyên hoa thường.** Stream trả
  markdown nên khoảng trắng không ổn định; còn đổi hoa thường là *sửa chữ* —
  đúng loại "sửa nhẹ" mà xác minh tồn tại để bắt. Có test ghim cả hai chiều.
* **`n` ngoài phạm vi đi vào `invalid_ns`, không đội lốt `Citation`.** Dựng một
  `Citation` với `chunk_id` bịa để nhét nó vào danh sách là tự làm đúng điều
  mình đang bắt model.
* **JSON hỏng không trả block về màn hình.** Nửa JSON vỡ trên màn hình người
  dùng tệ hơn một câu trả lời thiếu đuôi; block hỏng thành `block: "invalid"`
  kèm lỗi validate đầu tiên, văn bản nhìn thấy vẫn sạch.
* **Ba nghĩa của `block`**: `ok` / `absent` / `invalid` — `absent` là một giá
  trị chứ không phải một exception, vì model bỏ qua chỉ dẫn là tín hiệu vận
  hành cần đếm được, không phải một ngày bình thường.

## 3. ⭐⭐ `CitationHoldback` — bất biến "hai nơi cùng một chuỗi", kiểm bằng cắt mọi vị trí

Block không được rò vào khung `delta`, kể cả khi `CITATIONS:` bị cắt đôi giữa
hai delta (lần chạy thật: marker về trong 2 mảnh). Nhưng bất biến thật sự mạnh
hơn "không rò": **nối mọi chuỗi `feed()` + `flush()` phải khớp TỪNG BYTE với
`split_citation_block(full_text).text`** — vì chuỗi delta đã stream chính là
thứ được ghi vào Postgres, và màn hình với lịch sử lệch nhau là một bug không
truy được từ log. Test ghim bằng cách cắt cùng một văn bản **ở mọi vị trí**
(mọi cách cắt đôi + cắt từng ký tự) — chính test này định hình luật "nuốt đúng
MỘT newline trước marker" ở cả hai phía.

⚠️ **Một bug thứ tự bị bắt bởi test của `W4-06`, không phải test mới**: bản đầu
`yield` khung delta rồi mới `emitted.append` — generator có thể bị huỷ đúng tại
điểm yield, sau khi khung đã rời đi, và bản lưu thiếu đúng mẩu cuối người dùng
đã thấy. Hai test cancellation cũ đỏ ngay. `append` trước, `yield` sau.

## 4. Lần chạy thật (3 lượt `/chat`, ~$0,0026) — và P3 sai theo hướng có lợi

| lượt | block | verified | ghi chú |
|---|---|---|---|
| hỏi GDP **cả năm** 2024 (corpus không có) | `ok` | 0/0 | từ chối + `CITATIONS: []` |
| hỏi thủ đô Pháp (ngoài corpus) | `ok` | 0/0 | từ chối + `CITATIONS: []` |
| hỏi GDP **nửa đầu 2025** (corpus có) | `ok` | **1/2** | xem dưới |

* **P1 ✓** — cú pháp đúng cả 3/3 lượt, ngay lần đầu.
* **P3 ✗** — model **tuân thủ** cả khi từ chối: in `CITATIONS: []` thay vì bỏ
  block. Nhánh `absent` vẫn phải tồn tại (model khác/ngày khác), nhưng với
  deepseek-v4-flash nó không phải nhánh thường gặp như dự đoán.
* **P4 ✓** — TTFB 634 ms (`W4-06`: 787 ms). **P5 ✓** — 0 ký tự marker trong 66 khung delta.
* **P2 ✓, và ca thật đắt hơn kịch bản dự đoán.** Câu trả lời cite `[1][3]`,
  block có 2 claim:
  - `n=1` → quote 130 ký tự khớp **nguyên văn tuyệt đối** → `verified: true`.
  - `n=3` → `verified: false`, và khi đối chiếu tay thì nó là một **bản lai
    ghép**: lời văn lấy của chunk `00046` ("đạt 7,5% trong…" đổi thành "lên đến
    7,5%…"), gán cho chunk `00020` — mà vùng "7,5%" của `00020` lại nằm trong
    **câu ngữ cảnh tổng hợp** (Contextual Retrieval dán đầu chunk), lời văn
    thật ở đó là "nêu tăng trưởng GDP 7,5% nửa đầu năm 2025 (so với 6,5% cùng
    kỳ 2024)". Một trích dẫn *trông* có thật, chỉ vào một chunk *có thật*, và
    **không nằm trong chunk đó** — đúng lớp lỗi mà máy xác minh sinh ra để bắt,
    xuất hiện ngay ở lượt chạy thật thứ ba.

💡 Trong lúc truy ca `n=3` tôi đã đi nhầm đường mất một lượt đo: probe chỉ in
citation *cuối* (tail cắt mất dòng đầu) nên tôi tưởng "quote khớp Qdrant mà
serving nói false" và bắt đầu nghi ngờ serving đọc content khác. Sự thật: có
HAI citation, cái khớp là `n=1`. Bài học cũ, dạng mới: **đọc nguyên khung dữ
liệu trước khi dựng giả thuyết về hệ thống.**

## 5. Tiêm lỗi: 8/8 đỏ, mỗi phép đúng test chủ đích

| phép | kết quả |
|---|---|
| C1 mọi quote đều `verified` | ĐỎ — `test_a_fabricated_quote_is_flagged_not_dropped` |
| C2 bỏ chuẩn hoá whitespace | ĐỎ — test whitespace |
| C3 `flush` nuốt phần giữ lại | ĐỎ — test marker dở dang cuối stream |
| C4 marker giữa dòng cũng tính | ĐỎ — test mid-line |
| C5 bỏ holdback (block rò vào delta) | ĐỎ — test leak tầng service |
| C6 lưu bản thô thay vì bản đã phát | ĐỎ — cùng test trên (nó kiểm CẢ HAI vế) |
| C7 lệch một khi tra chunk theo `n` | ĐỎ — quote thật hoá `false` |
| C8 JSON hỏng trả block về màn hình | ĐỎ — test broken-json |

Backup phép tiêm bằng **copy file** — không `git checkout` (lỗi quy trình vừa
trả giá trong `TD-23` cùng phiên: checkout xoá sạch code chưa commit).

## 6. Hợp đồng khung SSE mới

`meta → sources → delta* → citations → done`. Khung `citations`:
`{citations: [{chunk_id, doc_id, quote, verified, source_url, section_path}],
invalid_ns, block, error, verified, total}`. Nhánh `no_retrieval`/`clarify`
**không** có khung này — không đưa gì cho model cite thì không có gì để xác
minh, và một khung rỗng lúc có lúc không chỉ dạy client rằng nó không đáng tin.
Bản ghi Postgres là **bản đã phát** (block bị cắt) — lịch sử đọc lại phải là
đúng thứ người dùng đã thấy.

## 7. Test & còn lại

Unit **+34** (`test_citation_verify.py` 28 + 6 tầng service), integration
**+2** (HTTP thật: một quote thật + một quote bịa qua marker cắt đôi; mode
`ok` đời `W4-06` → `block: "absent"`). Tổng **2026 xanh, 3 skip**, hai thứ tự.

* `message` trong Postgres chưa có cột citations — khung chỉ sống trong stream
  và log; `W5-08` (feedback) hoặc `W5-02` sẽ cần nó ở DB → ghi vào `TD-50`.
* Ngưỡng match là nhị phân (substring sau chuẩn hoá whitespace). Một quote sai
  một ký tự do OCR/format sẽ `false` — đúng theo thiết kế, nhưng chưa đo tỉ lệ
  từ chối oan trên diện rộng; đó là việc của `W5-02`, không phải của một lượt
  chạy tay.
* Key probe `rag_llxlP…` đã mint vào kho local trong lúc đo (server local,
  không có giá trị ngoài máy này); dọn kho key là việc chung của `W4-13` khi
  đóng compose.
