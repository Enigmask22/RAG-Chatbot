# `TD-22` — Ghim văn bản parse ra, và cái tên gói không làm việc gì cả

> **Trạng thái:** đóng · **Ngày:** 2026-08-22 · **Nhánh:** `feat/w1-foundation`
> **Code:** `pipeline/indexing/corpus_loader.py`, `pipeline/corpus/manifest.py`,
> `packages/rag_core/loaders/{base,docling_backend}.py`
> **Test:** `tests/unit/test_parse_pin.py` (13) · **Lệnh:** `make corpus-pin` · `make parse-pin-probe`

## 0. Nợ này nói gì, và vì sao nó chặn `W3-07`

Tới hết `W2`, phép biến đổi `byte → Document.content` là **hàm đồng nhất**
(`payload.decode("utf-8")`), nên ghim `sha256` của byte là ghim luôn văn bản. Từ
`W3-01` có parser đứng giữa, và lúc đó

```
content = parse(byte, phiên bản parser, tuỳ chọn parse)
```

manifest ghim đúng **một trong ba** đầu vào.

⚠️ Chế độ hỏng là loại tệ nhất: `sha256` vẫn khớp, `iter_documents` vẫn xanh, mọi
`chunk_id` vẫn tồn tại, **không test nào đỏ** — và mọi `TextSpan` của golden set
trỏ lệch. `W3-07` mở corpus cho định dạng khác `.txt`, tức mở đúng lỗ này, nên
nợ phải đóng trước.

## 1. Dự đoán ghi trước khi làm

| # | Dự đoán | Kết quả |
|---|---|---|
| E1 | Thêm `text_sha256` vào manifest là đủ | ❌ **sai** — thiếu nó thì không biết *cái gì* đã đổi; và vân tay hiện có tự nó đã hở (§3, §4) |
| E2 | Hôm nay `text_sha256` trùng `sha256` với mọi `.txt` | ✅ đúng — **60/60**, đo được (§2) |
| E3 | Vân tay `W3-01` đủ ghim môi trường parse | ❌ **sai nặng** — nó ghim gói **ô dù**, không ghim gói làm việc (§3) |
| E4 | Ghim version gói là đủ | ❌ **sai** — trọng số layout tải theo nhánh **di động** (§4) |
| E5 | Nối loader vào `corpus_loader` sẽ đổi văn bản của corpus hiện tại | ✅ đúng là **không** đổi — 60/60 trùng khít, 14.284.300 ký tự y nguyên (§5) |
| E6 | Luật "chưa ghim thì báo lỗi" áp cho mọi tài liệu | ⚠️ **nửa đúng** — áp cho mọi loader **trừ** `plain`, và có lý do (§6) |

**2 đúng, 1 nửa đúng, 3 sai.**

## 2. Hôm nay hai cột mới hoàn toàn thừa, và đó là điều phải đo

`make corpus-pin` (chế độ báo cáo, không ghi):

```
60 tài liệu · loader: {'plain': 60}
text_sha256 == sha256 (byte): 60/60 (100.0%) — phép biến đổi là hàm đồng nhất
    60 × plain|stdlib|1|encoding=utf-8
```

`bytes.decode("utf-8").encode("utf-8")` trả lại đúng byte cũ, nên `text_sha256`
**phải** trùng `sha256` — nhưng "phải" là một lập luận, còn 60/60 là một số đo,
và `W3-06` đã dạy rằng lập luận về mã hoá ký tự là chỗ dễ sai nhất (`read_text()`
âm thầm đổi CRLF → LF, chênh 14% số chunk).

Sự thừa ấy chính là thứ hết hạn khi có parser. Cột này không mua gì hôm nay; nó
mua tất cả vào ngày `W3-07` thêm tài liệu `.docx` đầu tiên.

## 3. ⭐ Vân tay `W3-01` ghim tên gói ô dù, không ghim gói làm việc

`W3-01` ghi đúng **một** số version: của `docling`. Nhưng gói phát tán tên
`docling` không phải gói làm ra văn bản. Đo trên môi trường này:

```
docling==2.121.0  yêu cầu:
    docling-core       >=2.91.0,<3.0.0     ← export_to_markdown() nằm ở ĐÂY
    docling-ibm-models >=3.13.0,<4
    docling-parse      >=7.12.0,<8.0.0
    pypdfium2          >=4.30.0,<6.0.0     ← hai major version
    rapidocr           >=3.9.1,<4.0.0
```

`LoadedDocument.text` là giá trị trả về của `DoclingDocument.export_to_markdown`,
và `inspect.getsourcefile` xác nhận hàm đó sống trong
`.venv/Lib/site-packages/docling_core/types/doc/document.py`. Nên `docling-core`
đi từ 2.91 lên 2.99 là markdown có thể đổi **trong khi vân tay không đổi một ký
tự** — đúng chế độ hỏng mà `TD-22` mô tả, chỉ là ở một trục tôi đã không nhìn.

`ParseFingerprint` nhận thêm `components`, và nó ghi **theo đường parse** chứ
không ghi tất:

| đường | components |
|---|---|
| `.docx`/`.md`/`.html`/`.pptx`/`.xlsx` | `docling-core=2.92.0` |
| `.pdf`, ocr tắt | `+ docling-ibm-models` `docling-parse` `pypdfium2` `+` trọng số (§4) |
| `.pdf`, ocr bật | `+ rapidocr=3.9.2` |

Ghi thừa cũng có giá, và nó là cái giá dễ coi thường: một lượt nâng `rapidocr`
làm mọi tài liệu DOCX báo "parser đã đổi" trong khi chúng chưa từng chạm OCR — và
**một cảnh báo kêu suốt là một cảnh báo bị tắt**.

## 4. ⭐⭐ Ghim đủ version gói vẫn chưa ghim được PDF

Tầng sâu hơn, và tôi chỉ tìm ra vì đã đi hỏi tiếp một tầng: pipeline PDF của
docling nạp trọng số từ Hugging Face, và **hai model được đối xử khác nhau**.

| model | repo | revision docling yêu cầu | nguồn |
|---|---|---|---|
| bảng | `docling-project/docling-models` | **`v2.3.0`** — tag cố định | `docling/models/stages/table_structure/table_structure_model.py:105` |
| bố cục | `docling-project/docling-layout-heron` | **`main`** — nhánh di động | `docling/datamodel/stage_model_specs.py:997-998` |

Model bố cục quyết định thứ tự đọc và cách chia khối của trang PDF, tức quyết
định thứ tự các đoạn trong markdown xuất ra — chính là thứ mà mọi offset span neo
vào. **Một lượt push lên `main` của repo ấy đổi văn bản parse ra mà không một con
số version nào trên máy này nhúc nhích.** Không `docling`, không `docling-core`,
không `uv.lock`.

Cái ghim được: **commit SHA đã phân giải**, đọc từ cache HF (`refs/<revision>`),
không phải chuỗi `"main"`. Chỉ đọc đĩa, không gọi mạng. Đo trên máy này:

```
docling-layout-heron@8f39ad3c0b4c     (refs/main)
docling-models@fc0f2d45e221           (refs/v2.3.0)
```

⚠️ Điều này **không** ngăn được việc trọng số bị đổi — nó chỉ đảm bảo lần đổi ấy
sẽ **ồn ào** thay vì im lặng. Muốn ngăn thì phải ghim `revision` bằng SHA lúc
gọi docling, và docling hiện chưa cho đường đó ở tầng `PdfFormatOption`.

## 5. Nối loader vào `corpus_loader`, và bằng chứng nó không đổi gì

`iter_documents` trước đây tự `payload.decode("utf-8")` tại chỗ — một đường thứ
hai song song với `rag_core.loaders`. Hai đường cho cùng kết quả với `.txt`,
nhưng chúng là **hai đường**, và hai đường thì sớm muộn lệch.

Giờ nó gọi `load_document`, tức đúng đường mọi chỗ khác dùng. Bằng chứng trung
tính:

```
Document.content == payload.decode("utf-8") : 60/60
tổng ký tự                                   : 14.284.300
```

14.284.300 khớp **từng chữ số** với `W3-06` §7 và với `index-pc256.json`
(`chars_in`) — cùng một corpus, đo bằng ba đường độc lập.

Đi kèm hai thứ nhỏ mà không nhỏ:

* `load_document` nhận `language` **từ manifest**. Cổng OCR của `W3-02` từ chối
  tiếng Việt (`TD-23`); manifest đã biết ngôn ngữ từng tài liệu nên không có lý
  do gì để cổng ấy phải đoán.
* `LoaderError` được bọc thành `CorpusIntegrityError` kèm `doc_id`. Trước đó một
  file rỗng báo `"d-1.txt: file rỗng"` — trong lượt 60 tài liệu thì tên file
  không đủ để biết dòng manifest nào phải sửa, và người gọi vốn chỉ bắt
  `CorpusIntegrityError`.
* `LoadedDocument.as_metadata()` được dựng ở `W3-01` rồi **nằm không** (chỉ một
  test đụng tới). Giờ nó vào `DocumentMetadata.extra`, nên từ một `chunk_id` bất
  kỳ truy ngược được lần parse đã sinh ra nó.

## 6. ⚠️ "Chưa ghim" không phải lúc nào cũng là lỗi — và luật phải có lý do

Cám dỗ là viết `if entry.text_sha256: kiểm`. Nhưng như thế thì **rỗng = bỏ qua**,
tức đúng chế độ hỏng đang đi sửa, chỉ đổi chỗ.

Luật dùng ở đây theo **loader**, không theo giá trị:

* `plain` (`.txt`) — chưa ghim thì chấp nhận. `sha256` của byte **đã** ghim văn
  bản (§2), không có gì thêm để kiểm.
* mọi loader khác — chưa ghim là **lỗi** (`ParsePinError`).

Nên `W3-07` về mặt vật lý không thêm được một tài liệu `.docx` mà quên ghim.

`ParsePinError` tách khỏi `CorpusIntegrityError` vì cách xử lý khác hẳn: hash
byte lệch nghĩa là **file** đã đổi (tải lại, hoặc điều tra); vân tay lệch nghĩa
là **môi trường** đã đổi trong khi file y nguyên, và lối ra là ghim lại sau khi
đã xác nhận văn bản mới dùng được.

Và một trường hợp thứ ba, cố ý **không** phải lỗi: vân tay đổi mà `text_sha256`
không đổi. Nâng `docling-core` mà markdown ra y hệt là chuyện tốt — nhưng nó
được log `WARNING`, vì để manifest mang vân tay cũ nghĩa là lần lệch **thật** sau
này sẽ chỉ sai thủ phạm.

## 7. Dựng lại chế độ hỏng trên tài liệu thật

`make parse-pin-probe` lấy tài liệu corpus nhỏ nhất và chép **nguyên byte** sang
tên `.md`. Byte không đổi một bit, nên đây là phép thử sạch nhất cho câu "kiểm
theo byte có bắt được không".

| | `.txt` (plain) | `.md` (docling) |
|---|---|---|
| sha256 byte | `8da126fbc1a136b7` | `8da126fbc1a136b7` — **trùng khít** |
| khớp manifest | ✅ | ✅ |
| `text_sha256` | `8da126fbc1a136b7` | `4c16f1465f383008` |
| ký tự | 4.688 | **3.673** (−21,6%) |

**21,6% văn bản biến mất, và kiểm-theo-byte không thấy gì.** Bốn cảnh của probe:

1. `.txt` — mọi thứ khớp.
2. `.md` **chưa ghim** → `ParsePinError: manifest chưa ghim text_sha256 mà tài
   liệu này đi qua loader 'docling'`. ✅
3. `.md` **đã ghim** → nạp được, `extra["parser"] == "docling"`. ✅
4. `.md` đã ghim, môi trường dịch chuyển → ✅ và thông báo là:
   `**môi trường parse** đã đổi: mất: docling-core=2.91.0 · thêm: docling-core=2.92.0`

Cảnh 4 là chỗ cả thiết kế trả cổ tức: thông báo nêu thẳng **`docling-core`** —
đúng cái gói mà trước thay đổi này vân tay không hề ghi. Vân tay của một PDF có 9
phần, nên `_fingerprint_delta` in chỗ **khác nhau** thay vì in cả hai chuỗi; bắt
người đọc tự dò bằng mắt đúng lúc đang hoảng là một thiết kế tồi.

## 8. Test

13 test trong `tests/unit/test_parse_pin.py`, tách khỏi `test_corpus_loader.py`
vì hai file hỏi hai câu độc lập: ở đó là *file trên đĩa có đúng file đã tải về
không*, ở đây là *văn bản parse ra có đúng văn bản golden set neo vào không*.
`TD-22` tồn tại chính vì hai câu ấy từng bị coi là một.

Tiêm lại chỗ hở (cho `_verify_parse` trả về ngay): **4 test đỏ**. Một test ghim
corpus thật — 60/60 phải còn nằm trên đường đồng nhất; ngày nào nó đỏ thì ngày đó
có tài liệu đi qua parser, và lúc ấy **phải ghim chứ không phải sửa test**.

⚠️ Ba assert đầu tiên tôi viết là vô nghĩa (một tautology `x == x`, một
`... or True`, một biến alias thừa) — đúng loại "test không bao giờ đỏ được" mà
`W3-05` §3 vừa viết cả một mục. Đã thay bằng phép kiểm thật, kèm một phép kiểm
`components` phải được **sắp** nên thứ tự lời gọi không đổi digest.

## 9. Còn lại gì

* `W3-07` hết bị chặn.
* `TD-27` (mới): ghim được commit SHA của trọng số **không** ngăn được nó đổi,
  chỉ làm lần đổi ấy ồn ào. Muốn ngăn thì docling phải cho truyền `revision`
  xuống `PdfFormatOption`.
* Corpus vẫn 60/60 là `.txt`, nên toàn bộ máy móc này **chưa** được chạy thật
  trên một tài liệu corpus nào đi qua parser. Nó được kiểm bằng một tài liệu thật
  chép sang `.md`, đó là phép thử tốt nhất có được hôm nay nhưng không thay được
  `TD-24`(2) — thêm DOCX/HTML thật vào corpus.
