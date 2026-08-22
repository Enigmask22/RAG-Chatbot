# `W3-01` — Docling loader: 6 định dạng, và một lỗ vừa mở ra trong chuỗi toàn vẹn

> 2026-08-22 · DoD: *PDF 2 cột đọc đúng reading order; bảng giữ được cấu trúc* ·
> Test: `tests/unit/test_loaders.py` với 6 file fixture (mỗi định dạng 1 file) ·
> Evidence: so sánh output trước/sau

## 0. Câu hỏi của hạng mục này

DoD hỏi hai câu về **chất lượng parse**. Cả hai đều trả lời được, và câu trả lời
là "đạt". Nhưng đi làm hai câu ấy thì lộ ra câu thứ ba mà DoD không hỏi, và câu
thứ ba mới là thứ quyết định `W3` được phép làm gì tiếp:

> **Chèn một bộ parse vào giữa byte và `Document.content` thì `golden_v1` còn
> đúng không?**

Đo được: **0/280 span sống sót**. Không phải "một số lệch" — **không span nào**.

## 1. Dự đoán ghi trước khi đo

| # | Dự đoán | Kết quả |
|---|---|---|
| D1 | Cho 60 file `.txt` corpus qua docling thì **0/60** byte-identical | ✅ đúng — nhưng tôi đoán đúng con số mà đoán sai **cơ chế** (xem §3) |
| D2 | Đường PDF cần tải model từ HF → unit test không offline hoàn toàn | ✅ đúng |
| D3 | PDF 2 cột content stream sai thứ tự → docling đọc **đúng** thứ tự | ❌ **sai ở lần đầu** — và cái sai đó là phần đáng giá nhất của hạng mục (§2) |
| D4 | Bảng trong XLSX/DOCX ra markdown pipe table đủ hàng/cột | ✅ đúng, 5/5 định dạng có bảng |
| D5 | typer 0.27.1 → 0.26.8 không phá gì | ✅ đúng — không chỗ nào trong repo import typer |
| D6 | Xuất markdown của chính file `.md` **không** byte-identical với đầu vào | ✅ đúng |
| D7 | Lấy được heading hierarchy từ document model, không cần regex | ✅ đúng — nhưng **`level` của nó không dùng thẳng được** (§4) |

**4/7 hoàn toàn đúng, 2 đúng-nhưng-thiếu, 1 sai.** Hai lần "đúng mà thiếu" (D1,
D7) cùng một kiểu: tôi đoán trúng *kết quả* rồi tưởng là đã hiểu *lý do*.

## 2. ⭐ Fixture "sạch" đo cái generator của tôi, không đo docling

Bản fixture PDF đầu tiên đúng như cách người ta hay dựng: hai cột, mỗi cột bốn
dòng ngắn bằng nhau, cách nhau đều, khoảng trắng rộng rãi. Content stream ghi
xen kẽ trái–phải–trái–phải, tức **sai thứ tự đọc có chủ đích** — nếu docling
chạy theo thứ tự content stream thì trả ra `Alpha 1, Beta 1, Alpha 2, Beta 2…`,
còn nếu nó phân tích bố cục thì trả trọn cột trái rồi mới tới cột phải.

Docling trả ra:

```
Alpha one: the left column begins Beta one: the right column begins
Alpha two: it continues downward Beta two: it continues downward …
```

Tức **đúng thứ tự content stream** — hỏng đúng cái mà DoD bảo phải không hỏng.

Trước khi kết luận "docling không đọc được hai cột", quét theo số dòng mỗi cột:

| dòng/cột | thứ tự đọc đúng? | chế độ hỏng |
|---:|---|---|
| 4 | ❌ | gộp cả hai cột thành một đoạn |
| 12 | ✅ | — |
| 24 | ❌ | **cột phải bị phân loại thành `table`** |
| 40 | ✅ | — |

**Không đơn điệu.** Một thư viện hỏng thật thì không hỏng kiểu 4 ✗ / 12 ✓ / 24 ✗
/ 40 ✓. Model bố cục của docling ăn **ảnh trang**, và một trang gồm mấy dòng
ngắn giống hệt nhau cách đều nhau nằm **ngoài phân bố** của thứ nó được huấn
luyện — nó rơi vào lòng chảo nào là chuyện của hình học ngẫu nhiên.

Đổi fixture sang **văn xuôi có độ dài dòng so le** (đúng dáng một đoạn được xếp
chữ thật) rồi quét 4 biến thể (rộng cột 30/34/38 ký tự, cỡ chữ 8/9/10):

| biến thể | thứ tự đọc đúng? | nhãn docling |
|---|---|---|
| w=34 s=9 | ✅ | `section_header`, `text`, `text` |
| w=38 s=9 | ✅ | `section_header`, `text`, `text` |
| w=34 s=10 | ✅ | `section_header`, `text`, `text` |
| w=30 s=8 | ✅ | `section_header`, `text`, `text` |

**4/4**, và cấu trúc đúng: một heading + **hai khối text**, mỗi cột một khối.

> 💡 **Bài học không phải "docling ổn".** Nó là: **một fixture được dựng cho
> "sạch" là một fixture nằm ngoài phân bố, và test chạy trên nó đo xem generator
> của tôi chạm vào biên quyết định của model ở chỗ nào — chứ không đo thư viện.**
> Nếu bản đầu tiên tình cờ rơi vào `n=12` thì tôi đã có một test xanh, một DoD
> đánh dấu ✅, và một kết luận không có cơ sở nào.
>
> Đây là lần thứ ba trong dự án cùng một khuôn "phép đo đầu tiên đo nhầm thứ":
> `W2-06` (bảng đơn điệu thuyết phục hoá ra là nhiễu của bước embed) và `W2-08`
> (người thắng do 6 mẫu lại quyết định). Khác biệt lần này: **thứ đo nhầm là do
> chính tôi dựng ra**, nên không có cách nào phát hiện bằng cách nhìn output.

Test giữ lại **cả hai vế**, vì chỉ khẳng định "cột trái ra trước" thì một
fixture ghi đúng thứ tự cũng làm test xanh:

```python
def test_the_fixture_really_is_out_of_order_in_the_content_stream(self):
    raw = PDF.read_bytes()
    assert raw.find(b"(Beta. The right column continues)") < raw.find(
        b"(first half of the argument and it)"
    ), "fixture không còn xen kẽ hai cột — test thứ tự đọc mất ý nghĩa"
```

## 3. ⭐⭐ 0/280 span golden sống sót — và mất là do **dồn dòng**, không phải mất chữ

Tới hết `W2`, `pipeline/indexing/corpus_loader.py` làm thế này:

```python
digest = hashlib.sha256(payload).hexdigest()   # đối chiếu manifest
text = payload.decode("utf-8")                 # ← hàm ĐỒNG NHẤT
```

Phép biến đổi giữa byte và `Document.content` là **hàm đồng nhất**, nên ghim
byte cũng chính là ghim nội dung, nên `TextSpan` của `golden_v1` (neo theo offset
ký tự) an toàn tuyệt đối. `W3-01` chèn một bộ parse vào giữa:

```
content = parse(bytes, phiên_bản_parser, tuỳ_chọn_parse)
```

Manifest ghim **một** trong ba đầu vào. `make loader-probe` cho 60 tài liệu
corpus qua backend markdown của docling:

| phép đo | kết quả |
|---|---|
| tài liệu parse được | 60/60 |
| **đồng nhất byte** | **0/60** |
| tỉ lệ độ dài | tb **0,9115** · [0,7770, 1,0016] |
| **dòng còn nguyên vẹn** | tb **0,0270** · min 0,0000 |
| **span `golden_v1` sống sót** | **0/280 (0,0%)** |

Ba con số này **không nhất quán với trực giác**, và chỗ không nhất quán chính là
phát hiện: văn bản chỉ ngắn đi **8,85%**, tức gần như toàn bộ chữ vẫn còn — mà
chỉ **2,70%** số dòng còn nguyên. Chữ không mất; **dòng bị dồn lại**:

```
TRƯỚC: ['Vietnam Transport Knowledge Series',
        'Supported by AUSTRALIA–WORLD BANK GROUP STRATEGIC PARTNERSHIP IN VIETNAM,',
        '             GOVERNMENT OF GERMANY and NDC PARTNERSHIP SUPPORT FACILITY']
SAU:   ['Vietnam Transport Knowledge Series Supported by AUSTRALIA–WORLD BANK GROUP
         STRATEGIC PARTNERSHIP IN VIETNAM, GOVERNMENT OF GERMANY and NDC …',
        '```']
```

Backend markdown nối các dòng liên tiếp thành đoạn, chèn thêm code fence, bỏ
`\r`. Mỗi lần dồn là mọi offset phía sau **trượt đi**, nên tại cùng byte 3000:

```
TRƯỚC: 'ents\t\r\nFigures\tand\tTables ………………vii\t\r\nForewords ……'
SAU:   'eral\tRepublic\tof\tGermany\t\tto\tthe\tSocialist\tRepublic\tof\tVietnam…… xiii'
```

Cùng một tài liệu, cùng offset, hai chỗ cách nhau vài trăm ký tự.

> ⚠️ **Đây là chế độ hỏng tệ nhất có thể**: `sha256` của file vẫn khớp manifest,
> `iter_documents` vẫn xanh, không test nào đỏ, mọi `chunk_id` vẫn tồn tại — và
> **mọi con số recall sau đó đều sai**. Đúng khuôn `TD-12` (đổi `chunk_size` giữ
> nguyên id mà đổi nội dung), chỉ khác trục: lần đó là chunker, lần này là
> parser.

### Hệ quả kiến trúc: `.txt` **không** đi qua docling

Bảng định tuyến của `rag_core.loaders`:

| đuôi | loader | vì sao |
|---|---|---|
| `.txt` | `plain` | hàm đồng nhất — **cả 60 tài liệu corpus nằm ở đây** |
| `.md` `.markdown` `.html` `.htm` `.pdf` `.docx` `.pptx` `.xlsx` | `docling` | cần parse thật |

Ranh giới ấy **không** phải để tiết kiệm mà là điều kiện để mọi con số của `W2`
còn giá trị. Tiện là chính docling ép nó: `InputFormat` **không có** `txt`, nên
đây là ràng buộc của thư viện chứ không phải quy ước của tôi.

⚠️ Nhưng ràng buộc ấy mỏng hơn vẻ ngoài: phép đo trên phải **đổi đuôi `.txt`
thành `.md`** mới đưa vào docling được — và đó đúng là kịch bản đáng lo thật.
Cùng một nội dung, vào repo dưới `.txt` thì ra một `Document.content`, dưới
`.md` thì ra một cái khác. Không có gì hôm nay chặn chuyện đó → **`TD-22`**.

## 4. Cùng một tài liệu, ba cách đánh số heading

D7 đúng: docling có sẵn phân cấp, không cần regex. Nhưng `item.level` **không
dùng thẳng được**. Cùng một nội dung logic (h1 → h2 → h3) qua 6 backend:

| nguồn | h1 | h2 | h3 |
|---|---|---|---|
| `.docx` | `section_header` **level 1** | level 2 | level 3 |
| `.md` / `.html` | **`title`** (không có level) | `section_header` **level 1** | level 2 |
| `.pptx` | `title` | `title` | `list_item` |
| `.xlsx` | — | — | — |

Tin thẳng `level` thì **cùng một heading là cấp 3 khi tới từ DOCX và cấp 2 khi
tới từ HTML**. Tức `W3-03` — DoD của nó là *"`section_path` đúng trên tài liệu có
3 cấp heading"* — sẽ dựng `section_path` khác nhau cho cùng một nội dung, chỉ vì
người gửi lưu file dưới định dạng khác.

Quy tắc chuẩn hoá (`_normalise_depth`): `title` chiếm **cấp 1**; `section_header
level L` nằm ở cấp **`L + 1` nếu đã gặp `title` trước đó**, ngược lại là cấp `L`.
Khớp cả bốn cột. Sau chuẩn hoá:

| nguồn | độ sâu đo được |
|---|---|
| `.docx` | `[1, 2, 3]` |
| `.html` | `[1, 2, 3]` |
| `.md` | `[1, 2, 3]` |
| `.pptx` | `[1, 1, 1]` |
| `.xlsx` | `[]` |

⚠️ **PPTX và XLSX không phải lỗi của quy tắc mà là của định dạng.** PPTX cho mỗi
slide một `title` nên không có cây phân cấp nào để dựng; XLSX không có heading,
kể cả tên sheet (`"Chỉ tiêu vĩ mô"`) cũng **mất**. `W3-03` phải biết trước điều
này thay vì phát hiện lúc `section_path` rỗng trên production.

## 5. Bảng: 5/5 giữ được cấu trúc

| fixture | vào | thời gian | ra (ký tự) | heading | độ sâu | bảng |
|---|---:|---:|---:|---:|---:|---:|
| `two-column.pdf` | 2.250 B | 11.267 ms¹ | 757 | 1 | 1 | 0 |
| `chuong-i.docx` | 37.137 B | 35 ms | 416 | 3 | 3 | 1 |
| `chuong-i.pptx` | 30.605 B | 31 ms | 481 | 3 | 1 | 1 |
| `chi-tieu.xlsx` | 5.087 B | 7 ms | 209 | 0 | 0 | 1 |
| `chuong-i.html` | 778 B | 13 ms | 413 | 3 | 3 | 1 |
| `chuong-i.md` | 439 B | 6 ms | 413 | 3 | 3 | 1 |

¹ Lần gọi đầu trong tiến trình = nạp model bố cục. Các lần sau **0,12–0,77 s**.

Cả 5 định dạng có bảng đều giữ đủ 3 hàng dữ liệu và xuất ra markdown pipe table
(trừ PPTX, nơi cùng dữ liệu nằm ở cả bullet lẫn bảng).

**Một ký tự bị nuốt, và chỉ ở một backend.** Dấu gạch dài `—` trong
`Chương I — Tổng quan kinh tế`: backend `.md` và `.docx` giữ nguyên, backend
`.html` trả về `-`. Cùng một ký tự, cùng một tài liệu, hai câu trả lời. Đó là lý
do đường `.txt` phải là hàm đồng nhất chứ không phải "chắc docling không đổi gì".

## 6. OCR mặc định TẮT, và đó là quyết định có số đo

Pipeline PDF của docling bật OCR sẵn. Lần chạy đầu tải ~30 MB trọng số RapidOCR
rồi chạy detect + recog trên từng trang:

| cấu hình | thời gian, fixture 1 trang |
|---|---:|
| `do_ocr=True` (mặc định của docling) | **70,56 s** |
| `do_ocr=False` (mặc định của `W3-01`) | **0,12–0,77 s** |

Khoảng **hai bậc độ lớn**, cho một PDF born-digital vốn đã có sẵn text layer —
tức toàn bộ chi phí ấy mua về đúng thứ đã có. Phát hiện scan rồi mới gọi OCR là
DoD của **`W3-02`**; ở đây tắt để `W3-01` không âm thầm trả giá đó cho mọi tài liệu.

> ⚠️ **Đính chính 2026-08-22 (`W3-02`): bảng trên là cold start vs cold start,
> KHÔNG phải chi phí mỗi trang — và tôi đã dùng nó như thể là.**
>
> Đo lại 5 lượt liên tiếp **trong cùng một tiến trình**: `12,67 · 0,34 · 0,34 ·
> 0,35 · 0,36` giây. Phần lớn cái 70 s là **nạp model một lần cho cả tiến
> trình**, lần đầu còn kèm tải ~30 MB trọng số. Chi phí **cận biên** là
> **0,35 s/trang** so với ~0,12 s khi tắt — **gấp ~3 lần, không gấp 500**.
>
> Quyết định `do_ocr=False` **không đổi** và vẫn đúng: chi phí cố định ~12 s mỗi
> tiến trình là thật, và tài liệu born-digital không cần trả nó. Cái đổi là **lý
> do**: `W3-02` đáng làm vì **đúng/sai** (không phát hiện thì PDF ảnh trả rỗng →
> `LoaderError` → tài liệu biến mất khỏi index, im lặng) và vì nó là chỗ duy
> nhất chặn được OCR chạy trên tài liệu tiếng Việt — không phải vì tiết kiệm
> thời gian. Xem `w3-02-ocr-fallback.md` §4.

## 7. Fixture phải sinh được lại, và lần đầu thì không

`.docx`/`.pptx`/`.xlsx` là zip: zip ghi mtime từng entry, OOXML ghi thêm ngày
tạo/sửa vào `docProps`. Chạy `make loader-fixtures` hai lần ra hai chuỗi byte
khác nhau → `git status` bẩn mỗi lần và sha256 trong test vô nghĩa.

Đóng băng timestamp zip về 1980-01-01 sửa được 5/6. `chi-tieu.xlsx` **vẫn nhảy**:
6 tiến trình cho 3 chuỗi byte. Khác nhau đúng một trường — `dcterms:modified`,
mà **openpyxl ghi đè bằng giờ hiện tại ngay trong lúc serialize**, sau khi tôi đã
set `book.properties.modified`. Phải sửa **sau** khi save, trong `_freeze_zip`.
Sau đó 3/3 tiến trình ra cùng một hash.

> 💡 Đặt tham số trước khi save không có nghĩa là tham số ấy sống sót qua save.

## 8. Một phép đo không chạy xong không phải là một phép đo chậm

Bản `loader_probe` đầu dùng `SequenceMatcher(None, baseline, parsed).ratio()`.
Thuật toán bậc hai, tài liệu ~500 KB, 60 tài liệu → treo, không phải chậm. Đổi
sang tỉ lệ **dòng** còn nguyên vẹn đếm bằng `Counter`: tuyến tính, và trả lời
đúng câu đang hỏi. Đếm theo **bội** chứ không theo tập — một dòng lặp 5 lần mà
bản parse chỉ giữ 2 thì mất 3, dùng `set` sẽ báo còn nguyên.

Chính con số ấy (2,70% dòng, 91,15% ký tự) mới lộ ra cơ chế "dồn dòng" ở §3.
`SequenceMatcher` nếu chạy xong cũng chỉ cho một số ~0,9 và tôi đã kết luận
"gần như không đổi" — **sai hoàn toàn**.

## 9. Test

`tests/unit/test_loaders.py` — **63 test** (62 passed, 1 skipped có chủ đích):

| nhóm | canh cái gì |
|---|---|
| `TestEveryFormatInTheDoDLoads` | 6 định dạng ra được văn bản; `source_sha256` là hash của **byte trên đĩa** |
| `TestTheHeadingDepthIsNormalisedBecauseBackendsDisagree` | `[1,2,3]` ở cả docx/html/md; PPTX phẳng; XLSX rỗng |
| `TestTablesSurviveAsStructure` | 5/5 có bảng, đủ 3 hàng, xuất ra pipe table |
| `TestTwoColumnPdfIsReadInReadingOrderNotStreamOrder` | **cả hai vế** — fixture thật sự sai thứ tự, và output thật sự đúng thứ tự |
| `TestPlainTextIsTheIdentityFunction` | `.txt` byte-for-byte; `—` không bị chuẩn hoá; docling không nhận `.txt` |
| `TestTheFingerprintCoversEverythingThatChangesTheText` | đổi version/tuỳ chọn → đổi digest; đổi **thứ tự** tuỳ chọn → không đổi |
| `TestSectionPathAt` | cái `W3-03` sẽ gọi: pop đúng cấp, trước heading đầu là rỗng, heading không định vị được thì bỏ qua |
| `TestUnsupportedInput` | đuôi lạ / không đuôi / file không tồn tại |

Test PDF **skip** khi không tải được model docling — nhưng chỉ skip khi thông
điệp lỗi đúng là lỗi tải model; mọi lỗi khác vẫn đỏ, nếu không thì một hồi quy
thật cũng lặng lẽ thành "skipped".

`tests/unit/test_architecture_boundaries.py`: thêm `docling` vào tập phụ thuộc
nặng phải import lazy.

Toàn bộ: **1277 test** — 1276 passed, 1 skipped, exit 0, 377,42 s. `make lint`
sạch: `ruff check` passed · `ruff format --check` 126 file · `mypy` 119 file.

> ⚠️ **Và fixture PDF suýt không vào được repo.** `.gitignore` có `*.pdf` (chặn
> tài liệu tải về), nên `two-column.pdf` bị loại — clone sạch sẽ làm
> `test_fixture_exists` đỏ, mà trên máy tôi thì mọi thứ xanh. Bắt được bằng
> `git add -A --dry-run` chứ không bằng test. Thêm ngoại lệ
> `!tests/fixtures/loaders/*.pdf`.

### ⚠️ Rồi tôi suýt ghi một lỗi chưa tái lập được vào lịch sử git

Thấy `.gitattributes` có `* text=auto`, tôi kết luận ngay: PDF viết tay toàn
ASCII không có byte NUL → git coi là text → checkout trên Windows đổi LF thành
CRLF → **bảng xref ghi offset tuyệt đối nên file hỏng**. Lập luận chặt, và tôi
đã commit nó như một bản vá.

Đo lại: clone ở đúng commit **trước** bản vá, với `core.autocrlf=true`:

```
$ git ls-files --eol tests/fixtures/loaders/two-column.pdf
i/lf    w/lf    attr/text=auto
$ sha256sum …/two-column.pdf     # trùng đúng hash trên đĩa
```

**Không đổi.** Heuristic của git tự nhận ra nó là binary. Bản vá đúng, nhưng lý
do tôi ghi cho nó thì sai — nên `.gitattributes` và commit message đã sửa lại:
dòng khai báo giữ lại vì nó biến *một điều đang đúng nhờ heuristic* thành *một
điều đúng vì được khai báo*, và heuristic ấy đọc **nội dung** file, mà nội dung
fixture thì `make loader-fixtures` có quyền đổi bất cứ lúc nào.

💡 Ba lỗi trong §7–§9 (`.gitignore`, `.gitattributes`, và cái này) đều cùng một
hình dạng: **"xanh trên máy tôi, chưa biết trên clone sạch"**. Hai cái đầu là lỗi
thật, cái thứ ba là lỗi tưởng tượng — và không phân biệt được bằng suy luận, chỉ
phân biệt được bằng cách clone ra thử.

## 10. Còn lại gì

* ⚠️ **`TD-22` (mới, chặn `W3-07`)** — manifest ghim `sha256` của byte nhưng
  không ghim `text_sha256`. Loader đã **khai báo** đủ ba đầu vào
  (`ParseFingerprint`: loader, version thư viện, tuỳ chọn) và trả kèm
  `text_sha256`, nhưng chưa có chỗ nào **đối chiếu**. Cần thêm cột vào
  `data/corpus_manifest.csv` + kiểm trong `iter_documents`. Cùng họ với `TD-19`.
* ⚠️ **Loader chưa được nối vào `pipeline/indexing/corpus_loader.py`.** Cố ý:
  nối vào trước `TD-22` là mở đúng cái lỗ §3 mô tả. Việc nối thuộc `W3-07`.
* ⚠️ **`W3-03` phải đọc §4 trước khi bắt đầu** — `item.level` không dùng thẳng
  được, và PPTX/XLSX không mang phân cấp.
* 💡 **Con số định hướng của `W2-05` vẫn nguyên**: trần `hit_rate@50` = 0,7799,
  tức 22% golden set không có bằng chứng trong 50 ứng viên đầu, và `W2-05` gán
  phần đó cho chất lượng parse (`W3-01`). **`W3-01` không đụng được vào con số
  đó**: corpus hiện tại là `.txt` thuần, không có bước parse nào để cải thiện.
  Nó chỉ đo được khi corpus có tài liệu PDF thật — tức sau `W3-07`.
* 💡 Fixture PDF là **tự sinh**, nên §2 chỉ chứng minh docling đọc đúng **một**
  bố cục hai cột dựng bằng tay. Muốn nói mạnh hơn thì cần PDF thật (World Bank,
  CC BY, đã có `source_url` trong manifest) — thuộc `W3-07`.
