# `W3-03` — Structure-aware chunker: `section_path` chỉ có giá trị khi nó không nói dối

> **Trạng thái:** xong · **Ngày:** 2026-08-22 · **Nhánh:** `feat/w1-foundation`
> **Code:** `packages/rag_core/chunking/structure.py`, `scripts/structure_probe.py`
> **Test:** `tests/unit/test_structure_chunker.py` (24)

## 0. Câu hỏi của hạng mục, và chỗ nó khác DoD

DoD viết: *"`section_path` **đúng** trên tài liệu có 3 cấp heading"*. Chữ "đúng"
mới là phần khó, vì `section_path` là thứ đi thẳng vào `Citation.section_path`
rồi hiện ra trước mặt người dùng: *"theo Chương I, Điều 15, Khoản 2…"*. Một
`section_path` **thiếu** thì người đọc thấy ngay là thiếu. Một `section_path`
**sai** thì không ai thấy — nó vẫn là một chuỗi heading có thật, lấy từ đúng tài
liệu ấy, chỉ trỏ nhầm chỗ.

Nên hạng mục này có hai nửa: cắt theo heading (dễ), và chứng minh đường dẫn gắn
vào mỗi chunk không nói dối (phần còn lại của báo cáo).

## 1. Dự đoán ghi trước khi đo

| # | Dự đoán | Kết quả |
|---|---|---|
| E1 | `headings` + `section_path_at` của `W3-01` dùng thẳng được, `W3-03` chỉ là bên tiêu thụ | ❌ **sai** — `section_path_at` có lỗi `break`, 575/587 chunk sai trên `wb1.pdf` (§4) |
| E2 | Ép kích thước bằng `_enforce_size` sẵn có là đủ | ❌ **sai** — gộp qua ranh giới section làm path nói dối; phải có luật tổ tiên chung (§2) |
| E3 | Báo cáo World Bank 129 trang cho ra cây heading nhiều cấp | ❌ **sai** — đúng **một** cấp (§5) |
| E4 | Cắt theo heading làm chunk vụn hơn hẳn so với fixed | ⚠️ **nửa đúng** — chỉ khi tắt gộp (min 1 ký tự); bật gộp thì 587 vs 595 (§5) |
| E5 | Corpus `.txt` hiện tại không có cấu trúc nào để dùng | ✅ đúng — **0/60** (§3) |
| E6 | Ba định dạng nguồn cùng nội dung ra cùng `section_path` nhờ `_normalise_depth` | ✅ đúng (§6) |
| E7 | Fixture ba heading không đủ để bắt lỗi định vị | ❌ **sai** — bản lỗi làm **11/24** test đỏ (§4) |

**3 sai, 1 nửa đúng, 3 đúng.** Ba cái sai đầu đều cùng một dạng: tôi tưởng
`W3-03` là hạng mục *tiêu thụ*, hoá ra nó là hạng mục *kiểm tra* — chỗ đầu tiên
mà những thứ `W3-01` dựng ra bị dùng thật, và cũng là chỗ đầu tiên chúng lộ ra.

## 2. ⭐ Gộp qua ranh giới section: chỗ `section_path` nói dối

`Chunker._enforce_size` gộp mảnh nhỏ hơn `min_chunk_size` vào mảnh liền trước.
Áp thẳng lên danh sách section thì:

```
## Điều 3
Nội dung ngắn.          ← 15 ký tự, dưới min_chunk_size
## Điều 4
Nội dung cũng ngắn.
```

→ một chunk, span bắt đầu trong `Điều 3`, `section_path = [Chương I, Điều 3]`,
mà nửa sau nội dung thuộc `Điều 4`. Không lỗi, không cảnh báo, và citation trỏ
sai điều luật.

Đây **đúng khuôn** với lỗi bản POC gộp chunk qua ranh giới *tài liệu*
(điểm 2 ở docstring `chunking/base.py`), chỉ thấp hơn một cấp — lần đó gộp nhầm
giữa hai tài liệu, lần này giữa hai section.

Cấm gộp không phải lối ra: tắt gộp trên `wb1.pdf` cho ra **735 chunk, chunk nhỏ
nhất 1 ký tự**, p05 = 39. Luật dùng ở đây là **hạ `section_path` xuống tổ tiên
chung** của các section bị gộp: `common_ancestor([Chương I, Điều 3], [Chương I,
Điều 4]) = [Chương I]`. Nông hơn, nhưng khẳng định đúng phần mà **cả hai** nửa
nội dung đều thoả.

Cái giá, đo trên hai báo cáo thật:

| tài liệu | chunk | có `section_path` | phải hạ xuống tổ tiên chung |
|---|---:|---:|---:|
| `wb1.pdf` (129 trang) | 587 | 541 (92,2%) | **46** |
| `wb2.pdf` (112 trang) | 474 | 456 (96,2%) | **18** |

64/1061 chunk (6,0%) đánh đổi một đường dẫn *sai* lấy một đường dẫn *nông hơn*.
Vì heading của PDF chỉ có một cấp (§5), tổ tiên chung ở đây luôn là `[]` — nên
trên PDF, luật này đọc ra là "gộp qua ranh giới thì bỏ hẳn đường dẫn".

## 3. ⭐ Corpus hiện tại: **0/60** tài liệu có cấu trúc để mà dùng

```
$ make structure-corpus
corpus: 60 tài liệu · có heading máy đọc được: 0 (0%)
```

Cả 60 tài liệu là `.txt` → `load_plain` → `headings = ()`. Nên hôm nay
`StructureChunker` **thoái hoá về fixed ở 60/60 tài liệu** và `section_path` rỗng
ở 100% chunk. Có một test ghim đúng điều đó (`test_thoai_hoa_ve_dung_ket_qua_cua_fixed_chunker`):
không heading thì output phải **trùng khít** `FixedSizeChunker`, để đổi
`strategy` sang `structure` trên corpus hiện tại không âm thầm dịch baseline.

0% không phải hỏng, và cũng không phải phần chưa làm — text thuần **không mang**
cấu trúc máy đọc được. Không tài liệu nào trong 60 có dòng markdown heading, và
"đoán heading bằng regex" chính là dựng **quy tắc độ sâu thứ hai** mà `W3-01` §4
đã cảnh báo.

Chỗ thú vị là hai đầu của cùng một tài liệu:

| cùng báo cáo *Vietnam STI 2020* | ký tự | heading |
|---|---:|---:|
| `.txt` từ endpoint text của World Bank (trong corpus) | 493 383 | **0** |
| `.pdf` gốc, qua docling | 501 638 | **128** |

Ràng buộc không nằm ở chunker mà ở **định dạng corpus**. `W3-07` nối loader vào
`corpus_loader.py`, và chỉ khi corpus có PDF/DOCX thật thì `W3-03` mới đo được
trên eval. → ghi vào `TD-24`.

## 4. ⚠️ Lỗi `break` của `W3-01`, và vì sao chỉ tới đây mới lộ

`LoadedDocument.section_path_at` bản `W3-01`:

```python
for heading in self.headings:
    if not heading.located or heading.start_char > offset:
        break
```

Docstring viết "heading không định vị được thì **bỏ qua**", code thì **dừng
hẳn**. Một heading không định vị được sẽ làm mù toàn bộ phần còn lại của tài
liệu. Và `_collect` sinh ra đúng tình huống ấy: `text.find` thất bại → `start_char
= -1`, nhưng `cursor` không đổi, nên heading kế tiếp vẫn định vị được.

Không phải giả thuyết: `wb1.pdf` có **6/128 heading không định vị được**, và ba
lý do khác nhau:

* `BOX 1. Conceptualizing innovation beyond formal R&D` — tiêu đề khung phụ,
  không xuất hiện nguyên vẹn trong markdown xuất ra.
* `23. According to some S&T experts…` — một **đoạn văn 170 ký tự** bị docling
  gán nhãn `section_header`, chứa cả soft hyphen `\xad`. Không định vị được lại
  là may: định vị được thì `section_path` sẽ mang nguyên đoạn văn ấy.
* `Pillar 3: Supply side - skills & knowledge` — chữ có thật trong văn bản nhưng
  nằm **trước** con trỏ, vì thứ tự `iterate_items()` không trùng thứ tự
  `export_to_markdown()`. Con trỏ tiến dần (chống heading trùng chữ, `W3-01` §4)
  đổi lấy chỗ hỏng này.

| tài liệu | heading không định vị được | chunk mà bản `break` trả đường dẫn khác |
|---|---:|---:|
| `wb1.pdf` | 6/128 | **575/587 (98,0%)** |
| `wb2.pdf` | 0/83 | 0/474 (0%) |

Sáu heading hỏng làm sai 98% chunk. `wb2` thì sạch tuyệt đối — nên nếu chỉ đo
một tài liệu, xác suất kết luận "không sao" là 1/2.

`W3-01` không thể tự bắt được: ở đó `section_path_at` không có ai gọi, nên nó là
code chưa từng chạy trên dữ liệu thật. `W3-03` là bên tiêu thụ đầu tiên.

**Còn E7 thì tôi đoán sai theo hướng bi quan.** Tôi tưởng một fixture ba heading
sẽ vẫn xanh với lỗi định vị. Thử lại bằng cách cấy lại bản lỗi: **11/24 test
đỏ**. Bộ test *có* chặn được. Nhưng thứ **tìm ra** lỗi là phép đối chiếu với
`section_path_at` trên tài liệu thật, chạy trước khi bộ test ấy tồn tại — nên
đây là may, không phải quy trình.

## 5. ⭐ Tài liệu thật chỉ có **một** cấp heading

Đây là kết quả bất ngờ nhất của hạng mục.

```
===== wb1.pdf =====
501638 ký tự · 128 heading (122 định vị được) · cấp [1] · 41 bảng
===== wb2.pdf =====
… 83 heading (83 định vị được) · cấp [1] …
```

Backend PDF của docling gán nhãn `section_header` nhưng **không suy ra cấp**:
mọi heading về cấp 1. Nên với PDF, `section_path` là một danh sách **một phần
tử** — "chunk này nằm trong mục nào", không phải một đường dẫn phân cấp.

Phân cấp chỉ tồn tại ở định dạng nào **tự mang nó**: DOCX (`Heading 1/2/3`),
HTML (`h1/h2/h3`), Markdown (`#/##/###`). Với PDF, phân cấp là thứ phải **suy
luận từ bố cục** (cỡ chữ, thụt lề, đánh số) và docling không làm.

Hệ quả cho kế hoạch: mục *"kiểm chứng `section_path`"* của `DocType.LEGAL` chỉ
có nghĩa nếu văn bản pháp luật vào corpus ở dạng **DOCX/HTML**, không phải PDF
scan hay PDF xuất từ máy in. Ghi vào `TD-24`.

Phân bố kích thước chunk (`chunk_size=1000`, `min=200`, `max=1500`):

| `wb1.pdf` | n | min | median | p95 | max |
|---|---:|---:|---:|---:|---:|
| structure (có gộp) | 587 | 172 | 832 | 1085 | 1497 |
| structure (không gộp) | 735 | **1** | 706 | 1000 | 1497 |
| fixed | 595 | 216 | 824 | 1042 | 1240 |

Với gộp bật, cắt theo heading gần như **không** đổi hình dạng chunk (587 vs 595)
— nó chỉ thêm `section_path` và dời chỗ cắt về ranh giới ngữ nghĩa. E4 sai ở
đúng chỗ đó: chunk vụn là hệ quả của *tắt gộp*, không phải của *cắt theo
heading*.

## 6. Ba định dạng, một `section_path`

Cùng nội dung logic (h1 → h2 → h3), ba nguồn, và số dấu thăng trong markdown
xuất ra **khác nhau** — `.md` cho `#`/`##`/`###`, `.docx` cho `##`/`###`/`####`:

| nguồn | `section_path` của chunk sâu nhất |
|---|---|
| `chuong-i.md` | `[Chương I — Tổng quan kinh tế, Điều 1. Phạm vi điều chỉnh, Khoản 1. Chỉ tiêu vĩ mô]` |
| `chuong-i.html` | `[Chương I - Tổng quan kinh tế, Điều 1…, Khoản 1…]` |
| `chuong-i.docx` | `[Chương I — Tổng quan kinh tế, Điều 1…, Khoản 1…]` |

Giống nhau **trừ** `—` → `-` ở bản HTML, đúng cái quirk `W3-01` §4 đã ghi. Đây
là bằng chứng `W3-03` **không** dựng quy tắc độ sâu thứ hai: nếu nó dò `#` thì
`.docx` đã ra bốn cấp.

`.pptx` ra ba đường dẫn **một phần tử** rời nhau (`[Chương I]`, `[Điều 1]`,
`[Khoản 2]`) vì PPTX không mang phân cấp; `.xlsx` ra `[]`. Cả hai đều là hành vi
đúng của `_normalise_depth`, đã ghi ở `W3-01`.

## 7. Hai bẫy tự đặt

**Vị trí cắt ≠ vị trí hỏi.** Ranh giới section lùi về **đầu dòng** để marker `##`
đi cùng heading của nó. Nhưng bản đầu của tôi hỏi `section_path_at` **tại chỗ
cắt** — tức tại ký tự `#`. Với `section_path_at`, heading ấy *chưa* bắt đầu, nên
nó trả về đường dẫn của section **liền trước**. Kết quả: **486/587 chunk** trên
`wb1.pdf` mang đường dẫn lệch đúng một section. Chạy trơn, không lỗi.
`section_boundaries` giờ trả **cặp** `(vị trí cắt, vị trí hỏi)` và tên hàm nói
thẳng rằng hai số ấy khác nhau.

**Script kiểm chứng lặp lại đúng lỗi nó đi kiểm.** Bản kiểm đầu hỏi
`section_path_at` tại **đầu** chunk — mà chunk mở đầu bằng `## Heading` thì đầu
chunk là `#`. Nó báo 486 chunk sai, sửa xong còn báo 54 chunk sai, và 54 cái đó
là *chunk đúng bị chấm sai*. Chuyển sang hỏi tại **ký tự cuối** của chunk — vị
trí duy nhất chắc chắn nằm trong section mà chunk kết thúc:

```
wb1: 587 chunk · khớp đúng 541 · tổ tiên chung 46 · SAI 0
wb2: 474 chunk · khớp đúng 456 · tổ tiên chung 18 · SAI 0
```

**0/1061 chunk nói dối** trên hai tài liệu 100+ trang.

## 8. Guard: `Document.content` phải bằng `LoadedDocument.text`

`Heading.start_char` là offset **trong `LoadedDocument.text`**. Nếu đường đi từ
loader tới `Document` có thêm bất kỳ bước chuẩn hoá nào — đổi CRLF, cắt khoảng
trắng — thì mọi offset lệch, chunker vẫn chạy trơn, và mỗi chunk mang
`section_path` của một chỗ khác. Đúng khuôn `TD-12`, đúng khuôn `TD-22`.

Nên `_usable_structure` so nguyên văn hai chuỗi, lệch thì **log `ERROR`, đếm vào
`documents_with_mismatched_text`, và rơi về cắt theo ký tự**: mất `section_path`
còn hơn `section_path` sai. Cùng lý lẽ với `require_ocr_support` ở `W3-02`.

Tương tự, tài liệu chưa `bind` bị đếm vào `documents_without_structure` và log
`WARNING` — không im lặng trả `[]` như thể tài liệu vốn không có cấu trúc.
`chunk_loaded()` là đường đi không quên được `bind`.

## 9. Test

24 test ở `tests/unit/test_structure_chunker.py`:

| nhóm | ghim cái gì |
|---|---|
| `TestDoDSectionPathTrenTaiLieuBaCapHeading` | DoD — 3 cấp, và mọi chunk nhất quán với `section_path_at` |
| `TestTaiLieuKhongCoHeading` | nhánh 2 của DoD; và output **trùng khít** `FixedSizeChunker` |
| `TestGopQuaRanhGioiSection…` | tổ tiên chung; và bật/tắt gộp cho hai hình dạng khác nhau |
| `TestRanhGioiCatLuiVeDauDong` | vị trí cắt ≠ vị trí hỏi, kèm một test ghim **cái bẫy** |
| `TestMotHeadingKhongDinhViDuoc…` | lỗi `break` của `W3-01` |
| `TestKhongBindThiPhaiDemChuKhongImLang` | đếm + cảnh báo, không im lặng |
| `TestContentKhacTextThiOffsetHetGiaTri` | guard §8 |
| `TestBaDinhDangNguonRaCungMotSectionPath` | `.md`/`.html`/`.docx` ra cùng đường dẫn (integration) |

Ba test cố ý **ghim cái bẫy** chứ không chỉ ghim hành vi đúng
(`test_hoi_tai_vi_tri_cat_se_tra_ve_section_TRUOC`): nếu docling đổi cách xuất
marker heading, chúng sẽ nói.

## 10. DoD và phần còn lại

✅ `section_path` đúng trên tài liệu 3 cấp heading — và "đúng" được định nghĩa
bằng nhất quán với `section_path_at`, kiểm trên 1061 chunk của hai tài liệu thật
(0 sai), không chỉ trên fixture.
✅ Tài liệu không heading — `section_path` rỗng, output trùng `FixedSizeChunker`.

Còn lại:

* **`TD-24`** (mới) — corpus không có tài liệu nào mang cấu trúc. `W3-03` chỉ đo
  được trên eval sau khi corpus có PDF/DOCX thật (`W3-07`). Và với PDF thì
  `section_path` chỉ sâu **một cấp**, nên tài liệu pháp luật cần vào ở dạng
  DOCX/HTML mới kiểm chứng được `DocType.LEGAL`.
* **`TD-22`** vẫn đứng trước `W3-07`.
* `StructureChunker` **chưa** nối vào `CachedChunker`: cache khoá theo
  `(content_hash, config_hash, chunker_name)` mà cấu trúc **không** nằm trong
  khoá nào cả. Hôm nay vô hại (cấu trúc là hàm của nội dung), nhưng đúng khuôn
  lỗi làm tròn `config_hash` của bản POC. Ghi vào `TD-24`.
* Chưa đo `section_path` có **cải thiện truy hồi** hay không — đó là `W3-06`, và
  nó cần `TD-24` xong trước.
