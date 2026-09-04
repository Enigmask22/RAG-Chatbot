# `W3-02` — OCR fallback: đọc được tiếng Anh, **từ chối** tiếng Việt

> 2026-08-22 · DoD: *PDF scan ra được text; có queue control tránh OOM* ·
> Test: `tests/integration/test_ocr_fallback.py` với 1 PDF scan · Evidence: —

## 0. Câu hỏi của hạng mục này, và câu trả lời không như DoD giả định

DoD viết như thể OCR là một cái công tắc: phát hiện scan → bật → có text. Với
tiếng Anh thì đúng thế. Với **tiếng Việt** — ngôn ngữ chính của corpus — máy OCR
đi kèm docling trả về văn bản *trông như nội dung* nhưng sai hầu hết chữ, và
không tầng nào phía sau phân biệt được.

Nên `W3-02` giao hai thứ: một bộ phát hiện có ngưỡng đo từ tài liệu thật, và một
**cái chốt từ chối** — vì với corpus này, OCR bật lên còn tệ hơn OCR tắt.

## 1. Dự đoán ghi trước khi đo

| # | Dự đoán | Kết quả |
|---|---|---|
| E1 | pypdfium2 đủ rẻ để phát hiện scan, không cần docling | ✅ đúng |
| E2 | Scan ~0 ký tự, born-digital hàng nghìn → hai phân bố tách hẳn | ✅ đúng, nhưng **không phải chỗ tôi tưởng** (§2) |
| E3 | Ngưỡng phải theo diện tích (ký tự/in²), không theo số ký tự | ✅ đúng |
| E4 | Quyết định phải ở mức **trang**, không mức tài liệu | ⚠️ **nửa đúng** — phát hiện theo trang, quyết định theo **tỉ lệ** (§2) |
| E5 | docling `do_ocr=True` chỉ OCR vùng bitmap nên PDF lai tự đúng | — chưa kiểm được, không có PDF lai thật |
| E6 | OCR trên GPU concurrency > 1 sẽ đội VRAM; cổng 1 job là đủ | ✅ đúng |
| E7 | PDF World Bank thật nằm cùng phía với fixture born-digital của tôi | ❌ **sai** — fixture của tôi thấp hơn **p05** của tài liệu thật (§2) |

**4/7 đúng, 1 nửa đúng, 1 sai, 1 chưa kiểm được.** Và cái sai (E7) đúng là lỗi
`W3-01` vừa dạy: fixture tự sinh nằm ngoài phân bố của tài liệu thật.

## 2. ⭐ Ngưỡng: đo trên PDF thật, và hai điều bất ngờ

Tải hai báo cáo World Bank thật (CC BY, cùng nguồn với corpus), đo mật độ **ký
tự trên inch²** từng trang:

| tài liệu | trang | min | p05 | p50 | p95 | max | trang < 1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `wb1.pdf` (thật) | 129 | 0,65 | 12,61 | 31,16 | 45,89 | 54,04 | **1** |
| `wb2.pdf` (thật) | 112 | 0,00 | 4,33 | 34,10 | 46,59 | 86,71 | **4** |
| `two-column.pdf` (fixture) | 1 | — | — | **8,17** | — | — | 0 |
| `scanned-page.pdf` (fixture) | 1 | — | — | **0,00** | — | — | 1 |

**Bất ngờ 1 (E7 sai).** Fixture born-digital của tôi ở **8,17** — nằm giữa p05
của `wb2` (4,33) và p05 của `wb1` (12,61), tức **ở đuôi thưa** của phân bố thật,
không ở giữa. Hiệu chỉnh ngưỡng trên nó thì ngưỡng đã đặt lệch. Đây là lần thứ
hai liên tiếp fixture tự sinh không đại diện cho dữ liệu thật; lần này tôi đi
lấy dữ liệu thật **trước** khi chọn hằng số.

**Bất ngờ 2 (E4 nửa đúng), và nó quan trọng hơn.** Báo cáo born-digital **thật
vẫn có trang trống**: 1/129 và 4/112 — bìa, ảnh tràn trang, trang phân cách. Nên
luật hiển nhiên *"có một trang thiếu text layer ⇒ tài liệu là scan"* sẽ đẩy
**100%** báo cáo World Bank vào OCR.

Hai hằng số, cả hai nằm giữa hai cụm cách nhau rất xa:

| hằng số | giá trị | cụm dưới | cụm trên |
|---|---:|---|---|
| `MIN_CHARS_PER_IN2` | **1,0** | trang không text layer: 0,00–0,96 | trang có text thưa nhất: **3,03** (bìa) |
| `SCAN_PAGE_RATIO` | **0,5** | tài liệu thật: **0,8%** và **3,6%** | scan thuần: **100%** |

⚠️ **Cái luật tỉ lệ đánh đổi đi**, nói ra để không phải phát hiện sau: một báo
cáo 129 trang có đúng 1 trang scan mang nội dung thật sẽ **không** được OCR.
`ScanReport.pages` giữ nguyên danh sách trang thưa để tầng trên quyết định khác.

## 3. ⭐⭐ Máy OCR đọc tiếng Anh và trả rác cho tiếng Việt

Fixture scan có **hai đoạn cùng nội dung, khác ngôn ngữ**, trong cùng một ảnh —
cùng font, cùng độ phân giải, cùng một lần chạy. Đó là điều kiện để so mà không
lẫn biến nào khác.

| model rec | tiếng Anh | tiếng Việt |
|---|---|---|
| `ch` PP-OCRv6 (mặc định docling) | ✅ **nguyên văn** | `Tāng trng t 7,09 phān trām nām 2024` |
| `latin` PP-OCRv3 | ✅ **nguyên văn** | `Tng trXXng XXt 7,09 phn trm nm 2024` |

Bản gốc: `Tăng trưởng đạt 7,09 phần trăm năm 2024`.

Ba điều đọc ra từ bảng này:

1. Tiếng Anh về **nguyên văn** — cả năm dòng, kể cả `consumer price index
   settled at 3.63`.
2. **Con số sống sót cả hai bên** (7,09 · 5,05 · 405,5 · 3,63). Hỏng nằm đúng ở
   dấu và hình dạng chữ cái tiếng Việt.
3. Model `latin` — thứ tôi tìm tới vì tiếng Việt dùng chữ Latin — **tệ hơn**:
   nó bỏ hẳn dấu và chèn `XX` cho ký tự ngoài bộ chữ. `latin_dict.txt` không
   phủ tiếng Việt.

> ⚠️ **Với corpus tiếng Việt, OCR bật lên còn tệ hơn OCR tắt.** Tắt thì tài liệu
> rỗng, `W3-01` ném `LoaderError`, và có người nhìn thấy. Bật thì nó sinh ra văn
> bản trông như nội dung, đi thẳng vào embedding → index → citation, và **không
> phép kiểm nào ở tầng sau phân biệt được**. Một câu trả lời có citation trỏ vào
> `Tāng trng t 7,09 phān trām` vẫn là một câu trả lời có citation.

Nên `require_ocr_support("vi")` **ném lỗi**, và thông điệp lỗi mang theo chính
ví dụ trên — người đọc log phải thấy ngay đây không phải chuyện "thử bật cờ xem
sao".

### Vì sao không làm VLM như plan viết

Dòng `W3-02` trong plan ghi *"Qwen2.5-VL / Gemini Vision"*, và phép đo trên đúng
là lý do nên làm thế. Nhưng môi trường hiện tại **không chạy được đường đó**:

* **không có `OPENROUTER_API_KEY`** — đã kiểm qua `get_settings()`;
* **DeepSeek**, key duy nhất đang có, **không có model thị giác**;
* Gemini không nằm trong hai provider đã chốt của dự án.

Viết một đường gọi API trả phí mà không chạy thử được lần nào là đúng chế độ
hỏng `W2-07` đã ghi lại ("chạy xong, đúng số ô, **không có dữ liệu**"). Để lại
**`TD-23`** kèm điều kiện mở khoá.

Tesseract (`vie` traineddata đọc tiếng Việt tốt) là lối ra thứ hai và rẻ hơn,
nhưng nó là **binary hệ điều hành** chứ không phải wheel — thuộc Dockerfile,
không thuộc commit này. Đã kiểm: máy hiện tại không có `tesseract`, không có
`onnxruntime` (nên nhánh `latin` phải chạy qua engine torch).

## 4. ⚠️ Đính chính một con số của `W3-01`, và nó đổi luôn lý do `W3-02` tồn tại

`W3-01` ghi: *"OCR đắt hai bậc độ lớn — 70,56 s bật, 0,12–0,77 s tắt"*, và tôi
dùng đúng câu đó làm lý do phải phát hiện scan. Đo lại 5 lượt liên tiếp **trong
cùng một tiến trình**:

```
12,67   0,34   0,34   0,35   0,36     (giây)
```

Phần lớn cái 70 s ấy là **nạp model một lần cho cả tiến trình** — lần đầu còn
kèm tải ~30 MB trọng số. Chi phí **cận biên** là **0,35 s/trang** so với ~0,12 s
khi tắt: **gấp ~3 lần, không phải gấp 500**.

Hệ quả không chỉ là một con số sai. Nó đổi **lý do** cả hạng mục:

| lý do tôi tưởng | lý do thật |
|---|---|
| Tránh trả giá OCR 70 s/trang | Sai. 0,35 s/trang. |
| — | **Đúng/sai**: không phát hiện thì PDF ảnh trả về rỗng → `LoaderError` → tài liệu **biến mất khỏi index, im lặng** |
| — | **Chốt ngôn ngữ**: phát hiện là chỗ duy nhất chặn được OCR chạy trên tài liệu tiếng Việt |
| — | Chi phí **cố định** ~12 s/tiến trình vẫn tránh được cho tài liệu born-digital |

Và hằng số của cổng đổi theo: `max_pages` từ 50 lên **500**. Ở 0,35 s/trang thì
129 trang là **45 giây**, không phải 26 phút — trần 50 trang sẽ chặn nhầm mọi
báo cáo World Bank bình thường. Trần 500 là để chặn đầu vào bệnh lý (5.000
trang ≈ 30 phút), đúng việc của nó.

> 💡 Đây là lần thứ hai trong hai hạng mục liên tiếp một con số đúng-cho-ca-của-nó
> bị áp sang ca khác. `W3-01` đo **cold start có OCR vs cold start không OCR** —
> phép đo ấy không sai, nhưng nó không phải phép đo *chi phí mỗi trang*, mà tôi
> đã dùng nó như thế.

## 5. Queue control

`OcrGate` chặn hai thứ khác nhau qua cùng một cổng:

* `max_concurrent = 1` — OCR chạy detect + recog **trên GPU** (log RapidOCR:
  `Using GPU device with ID: 0`), mà ngân sách VRAM card 8 GB đã kín gần một nửa
  vì embedding + reranker (`W2-07` đo **3.900/8.188 MiB**). Và đây cũng là giới
  hạn thật chứ không chỉ thận trọng: converter của docling nằm sau `lru_cache`,
  dùng chung từ nhiều luồng không an toàn.
* `max_pages = 500` — xem §4.

Test canh **cả bốn** tính chất mà một semaphore dễ làm sai: không có hai job
chồng nhau, job thứ hai **đợi** chứ không lỗi, khe được trả lại kể cả khi thân
`with` ném lỗi, và tham số vô nghĩa bị từ chối ngay lúc dựng.

## 6. Fixture scan, và một lỗi lặp lại từ `W3-01`

`scanned-page.pdf` là một trang giấy trắng có chữ, render bằng Pillow rồi lưu
thành PDF **ảnh** (bilevel, 150 DPI, 7,8 KB). Font dùng
`ImageFont.load_default(size=…)` — Aileron đóng gói sẵn trong Pillow — chứ không
dùng font hệ điều hành, để fixture sinh trên máy khác ra cùng chuỗi byte.

⚠️ **Và nó không idempotent ở lần đầu, đúng lỗi đã mắc với openpyxl ở `W3-01`.**
Tôi ép `/CreationDate` về mốc cố định rồi tưởng xong. Ba lần chạy ra ba hash. Diff
ra: khác **đúng một byte**, ở `/ModDate` — trường thứ hai mà tôi không nghĩ tới.
Hai lần chạy cách nhau dưới một giây thì trùng, cách nhau hơn một giây thì lệch.

> 💡 Lần trước bài học là *"đặt tham số trước khi save không có nghĩa nó sống
> sót qua save"*. Lần này là phiên bản hẹp hơn và khó chịu hơn: **vá một trường
> thời gian rồi tưởng đã vá hết**. Cả hai lần đều chỉ lộ ra khi chạy lại **và
> chờ đủ lâu** — một vòng lặp nhanh sẽ báo xanh.

## 7. Test

| file | số test | canh cái gì |
|---|---:|---|
| `tests/unit/test_scan_detection.py` | **26** | mật độ theo diện tích · ngưỡng · **một trang trống ≠ tài liệu scan** · chốt ngôn ngữ · cổng |
| `tests/integration/test_ocr_fallback.py` | **9** | OCR thật: tiếng Anh nguyên văn, con số sống, **tiếng Việt hỏng**, born-digital không trả giá |

Hai chỗ đáng nói:

* `TestTheVietnameseBlockIsWhyThisEngineIsRefused` **khẳng định cái hỏng**. Nếu
  một ngày nó đỏ vì tiếng Việt đọc được thì đó là tin tốt, và việc phải làm là
  **mở `OCR_VERIFIED_LANGUAGES`**, không phải sửa test cho xanh lại. Câu đó nằm
  ngay trong thông điệp assert.
* `test_without_ocr_the_page_is_empty_and_that_is_an_error` ghim đúng chế độ
  hỏng mà `W3-02` tồn tại để chặn: `W3-01` ném `LoaderError` khi văn bản rỗng —
  đúng, nhưng ở đường index thì một tài liệu ném lỗi là một tài liệu **không vào
  index, im lặng**.

Toàn bộ: **1312 test** — 1311 passed, 1 skipped, exit 0, 387,38 s. `make lint`
sạch: `ruff check` passed · `ruff format --check` 130 file · `mypy` 123 file
(thêm `pypdfium2.*` vào `ignore_missing_imports` — nó không có `py.typed`).

Marker `integration` được nới nghĩa: *"cần service ngoài **hoặc trọng số
model**"*. Test OCR không cần Qdrant/Postgres/Redis, nhưng nó tải ~30 MB trọng
số và không thuộc vòng lặp `make test` vài giây.

## 8. DoD: đạt một nửa, và nửa kia là kết luận chứ không phải việc còn dở

* ✅ **"có queue control tránh OOM"** — `OcrGate`, 8 test.
* ✅ **"PDF scan ra được text"** — **với tiếng Anh**. Nguyên văn cả năm dòng.
* ❌ **với tiếng Việt** — và đây **không** phải phần chưa làm xong. Máy OCR duy
  nhất chạy được trong môi trường này đọc sai tiếng Việt, hai model đều sai, và
  quyết định đúng là **từ chối** chứ không phải cố. Mở khoá cần một trong hai:
  `OPENROUTER_API_KEY` (→ VLM) hoặc Tesseract `vie` trong image (→ `W4-13`).

Giống `W2-05` ghi thẳng *"DoD 400 ms KHÔNG đạt"*: nói ra thì nó là một ràng buộc
đã biết, giấu đi thì nó là rác trong index sáu tuần nữa.

## 9. Còn lại gì

* ⚠️ **`TD-23` (mới)** — OCR tiếng Việt. Hai lối, điều kiện mở khoá nêu trước:
  (a) VLM qua OpenRouter khi có key — pin slug tường minh, `temp=0`, log model
  thực tế đã phục vụ, có cost cap (Quy tắc cứng #1); (b) Tesseract + `vie` trong
  Dockerfile. Đo lại bằng đúng fixture này thì so được trực tiếp với bảng §3.
* ⚠️ **`E5` chưa kiểm được** — PDF **lai** (bìa scan + ruột text) là ca thường
  gặp nhất trong thực tế và tôi không có mẫu thật nào. `SCAN_PAGE_RATIO = 0.5`
  hôm nay chưa từng được thử trên một tài liệu nằm giữa hai cụm.
* 💡 `pypdfium2` trích text theo **thứ tự content stream**: cùng `two-column.pdf`
  nó trả `Alpha… Beta… Alpha… Beta…`, còn docling trả đúng thứ tự đọc. Tức phát
  hiện scan dùng được (nó chỉ **đếm** ký tự) nhưng đừng bao giờ lấy text từ
  đường đó — nó tái lập đúng lỗi `W3-01` §2.
* ⚠️ Loader vẫn **chưa** nối vào `pipeline/indexing/corpus_loader.py`; `TD-22`
  đứng trước, không đổi.


---

## 10. ⚠️⚠️ PHỤ LỤC 2026-09-04 — phép đo §3 KHÔNG HỢP LỆ, và kết luận đảo một nửa

Bạn chỉ đạo thử EasyOCR cho `TD-23`. Lần chạy đầu (fixture cũ) EasyOCR cũng trả
rác y hệt RapidOCR — `Tkng trkng MMt 7,09` — và **hai máy độc lập hỏng cùng một
kiểu** là mùi của một nghi phạm chung. Render fixture ra nhìn tận mắt: dấu tiếng
Việt là **ô ☒ ngay trong ảnh**. `ImageFont.load_default()` của Pillow (Aileron)
không có glyph tiếng Việt. Bảng §3 vì thế chấm hai máy OCR trên một đề bài hỏng:
"không OCR nào đọc được tiếng Việt" thật ra là "không OCR nào đọc được ký tự ☒".

Hai chi tiết lẽ ra phải gây nghi từ đầu, ghi lại để lần sau nhìn thấy sớm hơn:

1. Dòng tiêu đề VI của fixture được viết **không dấu** (`"Cap nhat kinh te..."`)
   — nhiều khả năng chính người viết fixture (tôi) đã gặp vấn đề font và lách
   qua nó một cách vô thức thay vì hỏi vì sao.
2. §3 tự khen điều kiện đo ("cùng ảnh, cùng lần chạy") — kiểm soát biến rất kỹ
   giữa hai *máy*, nhưng chưa bao giờ kiểm **đề bài**: không ai mở ảnh ra xem
   chữ trong đó có phải tiếng Việt không.

### Đo lại trên fixture hợp lệ (DejaVu Sans, commit trong repo)

| máy | tiếng Anh | tiếng Việt |
|---|---|---|
| `rapidocr` PP-OCRv6 | ✅ đủ từ, 1 dòng xáo trật tự | ❌ **vứt 3/5 dòng**, còn lại sai dấu (`mức`→`múc`) |
| `easyocr` latin-g2 `[vi,en]` | ✅ đủ từ | ✅ **dấu 8/8**, char-acc 0,91–0,97 (tệ nhất: `phần`→`phẩn`) |

Số đo: `probes/td-23-easyocr.json`. Ba hệ quả:

* **Kết luận về RapidOCR đứng vững** — nhưng giờ nó được đo đúng. `en` giữ máy
  cũ (đường cũ, vân tay cũ).
* **`vi` mở qua EasyOCR**: `engine_for("vi") == "easyocr"`, `OCR_VERIFIED_LANGUAGES`
  thành bảng theo máy, vân tay parse mang tên máy (`ocr=easyocr`) thay vì
  `ocr=true` — hai máy cho hai văn bản khác nhau trên cùng ảnh nên "true" là một
  vân tay nói dối. Ngôn ngữ chưa đo (fr, zh…) vẫn bị từ chối, lý do cũ giữ nguyên.
* **Giới hạn ghi thành test**: tầng reading-order của docling xáo một phần trật
  tự từ (cả hai máy, cả hai ngôn ngữ). `test_word_order_is_not_guaranteed…` ghim
  đúng giới hạn ấy để nó không âm thầm đổi.

Test `test_vietnamese_diacritics_do_not_survive` lật chiều đúng như docstring
của nó dặn ("nếu một ngày nó đỏ vì tiếng Việt đọc được thì đó là tin tốt").
Tiêm 3 lỗi: nuốt từ chối ngôn ngữ lạ, mất nhánh routing `vi`, và **mất
`lang=[vi,en]` tường minh** (rơi về mặc định docling `[fr,de,es,en]` — đúng bộ
không có tiếng Việt) — 3/3 đỏ, phép cuối chỉ integration test bắt được.

⚠️ Trong lúc làm phần này tôi còn mắc thêm một lỗi quy trình đáng ghi: dùng
`git checkout --` để khôi phục file sau phép tiêm — nhưng bản policy mới **chưa
commit**, nên lệnh đó xoá sạch chính phần sửa. May scratchpad còn script. Bài
học trùng với `mutate_router.py`: backup phép tiêm bằng **copy file**, không
bằng git, chính vì lý do này.
