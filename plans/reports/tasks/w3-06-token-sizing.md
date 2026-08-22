# `W3-06` — Token-based sizing: DoD đã đạt sẵn, và cái đáng làm là chuyện khác

> **Trạng thái:** xong · **Ngày:** 2026-08-22 · **Nhánh:** `feat/w1-foundation`
> **Code:** `packages/rag_core/chunking/tokens.py`, `scripts/token_sizing_probe.py`
> **Test:** `tests/unit/test_token_sizing.py` (34) · **Lệnh:** `make token-probe`

## 0. Câu hỏi, và chỗ nó đã có sẵn câu trả lời

DoD viết: *"không chunk nào vượt max token của model"*. Với model đang dùng thì
điều đó **đã đúng từ `W2-01`** và đã được đo, chỉ là không ai gọi tên nó:
`plans/reports/probes/truncation-bgem3.json` ghi **0/15.814 chunk bị cắt**, token
lớn nhất **734** trên cửa sổ **8192** — dư **11,2×**.

Nên hạng mục này không phải "làm cho hết bị cắt". Nó là: *`chunk_size` tính bằng
ký tự có phải một cái núm dùng được không?* Câu trả lời đo được là **không**, và
đó mới là nội dung của `W3-06`.

## 1. Dự đoán ghi trước khi đo

| # | Dự đoán | Kết quả |
|---|---|---|
| E1 | Với BGE-M3, DoD đã đạt sẵn, không cần làm gì | ✅ đúng — 0/15.814, dư 11,2× (§2) |
| E2 | 1000 ký tự là lượng thông tin rất khác nhau giữa VI và EN | ⚠️ **nửa đúng** — 7,8% với BGE-M3, 30% với PhoBERT, và **đảo chiều** (§3) |
| E3 | Giá trị chính của token sizing là tránh bị cắt | ❌ **sai** — với BGE-M3 không có gì để tránh; giá trị là **san bằng giữa ngôn ngữ** (§4) |
| E4 | Dùng lại quy tắc p05 của `truncation.py` là đúng | ❌ **sai** — hụt 17% ngân sách đã khai (§5) |
| E5 | Test trần token là đủ để tin bản cài đặt | ❌ **sai** — 28 test xanh với một lỗi trộn đơn vị (§6) |
| E6 | Phải giữ mặc định `chars` để không đổi baseline | ✅ đúng — bật lên là 15.814 → 11.190 chunk (§8) |
| E7 | Chế độ token chậm hơn đáng kể | ✅ đúng — 8,0× và 41,3×, nhưng tuyệt đối không đáng kể (§7) |

**3 đúng, 1 nửa đúng, 3 sai.**

## 2. DoD, đo bằng tokenizer thật

`make token-probe` trên đúng 60 tài liệu corpus, đúng khối `chunking` của
`baseline.yaml`/`bgem3.yaml` (hai file dùng chung bộ số), tức **đúng 15.814 chunk
mà `W1-13` và `W2-01` đã index**:

| model | cửa sổ | chunk | token p50 | p95 | max | **vượt** |
|---|---:|---:|---:|---:|---:|---:|
| `BAAI/bge-m3` | 8192 | 15.814 | 218 | 295 | 734 | **0** |
| `vietnamese-bi-encoder` | 256 | 15.814 | 274 | 375 | 684 | **9.001 (56,9%)** |

Con số PhoBERT trùng **từng chữ số** với `truncation-baseline.json` của `TD-11`
(9.393/6.421 chunk theo ngôn ngữ, 6.133/2.868 bị cắt, p50 313/244, max 447/684) —
phép đo mới tái lập được phép đo cũ, nên tin được cả hai.

Bật `size_unit="tokens"` với PhoBERT: **21.823 chunk, 0 vượt**. Khác biệt không
phải "ít bị cắt hơn" mà là **không thể bị cắt**: trần là bất biến của thuật toán,
không phải hệ quả may mắn của một cấu hình.

## 3. ⭐ Ký tự không phải một đơn vị mang đi được

Cùng **một** bộ 15.814 chunk, hai tokenizer:

| model | ký tự/token EN | ký tự/token VI | token p50 EN | token p50 VI |
|---|---:|---:|---:|---:|
| `BAAI/bge-m3` | 5,83 | 5,41 | 212 | 243 |
| `vietnamese-bi-encoder` | 4,09 | 5,33 | **313** | 244 |

Hai điều:

* Đổi model là đổi số token của cùng một chunk tới **47%** (313 vs 212 ở tiếng
  Anh). `chunk_size` bằng ký tự là cái núm mà **ý nghĩa đổi khi đổi model**, và
  không có gì báo. `W2-01` đổi model và ghi "GIỮ NGUYÊN chunking của baseline" —
  đúng theo ký tự, và chunk quả thật giống hệt nhau; nhưng *lượng model đọc được
  trong mỗi chunk* thì đổi hẳn.
* **Chiều lệch giữa hai ngôn ngữ đảo dấu.** BGE-M3: tiếng Anh gói nhiều ký tự hơn
  mỗi token. PhoBERT: tiếng Việt, và khoảng cách rộng gấp bốn (30% vs 7,8%). Nên
  "tiếng Việt tốn token hơn" **không phải** tính chất của ngôn ngữ mà của **cặp
  (model, ngôn ngữ)**. Dự đoán E2 của tôi đúng về việc có chênh lệch, sai về
  nguyên nhân.

## 4. ⭐ Giá trị thật: san bằng kích thước giữa hai ngôn ngữ

So công bằng — cả hai bên **tắt** ngữ cảnh hàng xóm (§9 nói vì sao bắt buộc):

| BGE-M3 | EN p50 | VI p50 | chênh | EN p95 | VI p95 | chênh |
|---|---:|---:|---:|---:|---:|---:|
| `chars` 1000 | 174 | 199 | **+14,4%** | 219 | 254 | +16,0% |
| `tokens` 256 | 257 | 269 | **+4,7%** | 327 | 366 | +11,9% |

| PhoBERT | EN p50 | VI p50 | chênh |
|---|---:|---:|---:|
| `chars` 1000 | 256 | 199 | **−22,3%** |
| `tokens` 256 | 165 | 169 | **+2,4%** |

Với PhoBERT, cắt theo ký tự làm chunk tiếng Anh mang **nhiều hơn 28,6%** lượng
model đọc so với chunk tiếng Việt — hai nửa corpus được cắt bằng hai thước khác
nhau, mà mọi metric `cross_lingual` của `W1`/`W2` đều đo trên đúng cái đó. Cắt
theo token đưa chênh lệch về **2,4%**.

Đây là câu trả lời cho E3: cái token sizing mua về không phải "khỏi bị cắt" (BGE-M3
đã khỏi sẵn) mà là **hai nửa corpus dùng chung một thước**.

## 5. ⚠️ Dùng lại quy tắc p05 của `truncation.py` là sai — và lý do đảo chiều

`embedding/truncation.py::chars_per_token_p05` cố ý dùng **phân vị 5**, với lập
luận đã ghi và đúng: nó *gợi ý* một `chunk_size` mà **không có ai kiểm lại**, nên
lấy trung bình sẽ ra đúng ngưỡng mà một nửa số chunk vẫn bị cắt.

Tôi chép nguyên lập luận đó sang đây. Sai, vì ở đây **có** bước kiểm lại:

| ước lượng mật độ | token p50 thu được | chunk | vượt trần |
|---|---:|---:|---:|
| phân vị 5 | **213** (−17% so với 256 đã khai) | 13.986 | 0 |
| trung bình | **261** (+2%) | 11.190 | 0 |

Cả hai đều 0 vượt trần — vì trần do `fit_to_budget` giữ, không do ước lượng giữ.
Khác nhau ở chỗ p05 làm `chunk_size=256` thật ra có nghĩa là **213**, tức phá đúng
cái tính chất mà cả hạng mục này dựng lên. **Sự tồn tại của phép kiểm chứng làm
đổi ước lượng nào là đúng** — không phải một chi tiết cài đặt, mà là chỗ mà việc
tái sử dụng một lập luận đúng ở ngữ cảnh khác trở thành sai.

## 6. ⚠️ Lỗi trộn đơn vị của chính tôi, và 28 test xanh không thấy

`Chunker` mới có `self.sizing` (config đã quy về ký tự cho tài liệu hiện tại) bên
cạnh `self.config` (con số khai báo, và vẫn là thứ vào `config_hash`). Docstring
`sizing` viết thẳng: *"mọi chỗ trong chunker phải đọc `self.sizing`, không đọc
`self.config`, nếu không sẽ trộn hai đơn vị."*

Rồi tôi để `FixedSizeChunker` và `StructureChunker` đọc `self.config` — 12 chỗ.
Chúng nhận `chunk_size=256` **token** và dùng nó như 256 **ký tự**.

**28 test xanh.** Vì cả 28 test kiểm **trần**, mà trần do một đường code khác
(`_fit_tokens`) bảo đảm. Không test nào kiểm **đích**.

Chỗ lộ ra là corpus thật: chunk tiếng Việt ra **p50 = 71 token** thay vì 256, và
số chunk tiếng Việt *tăng* 36% trong khi tiếng Anh *giảm* 31% — một hình dạng
không giải thích nổi nếu bản cài đặt đúng. Đã thêm hai test:

* `test_p50_bam_sat_chunk_size_da_khai_bao` — p50 phải nằm trong `[0,5×; 1,2×]`
  ngân sách đã khai, trên cả 3 corpus mẫu. Cấy lại lỗi thì **3/3 đỏ**.
* `test_moi_chunker_deu_doc_sizing_chu_khong_doc_config` — quét chính source của
  `rag_core/chunking/*.py` tìm `self.config.<kích thước>`. Cấy lại lỗi thì đỏ.

Bài học ghi lại: **bảo đảm và đích là hai thứ khác nhau, và một bộ test chỉ kiểm
bảo đảm sẽ xanh trong khi đích trượt hoàn toàn.**

## 7. ⚠️ Probe đầu tiên đo trên một corpus không tồn tại

Bản probe đầu đọc corpus bằng `Path.read_text(encoding="utf-8")`. Trên Windows đó
là **universal newlines**: `\r\n` → `\n`.

| | ký tự | chunk |
|---|---:|---:|
| `read_text()` | 14.075.359 | **18.038** |
| `read_bytes().decode()` — đúng như `corpus_loader.py` | 14.284.300 | **15.814** |

Lệch 208.941 ký tự (1,5%) và **+14% số chunk**. Phép đo chạy trơn, ra bảng đẹp,
và đo một corpus không có thật. Chỉ phát hiện được vì con số không khớp
`index-baseline.json` đã công bố.

Đáng nhớ hơn: đây **đúng** cái ví dụ tôi dùng làm test cho guard
`_usable_structure` ở `W3-03` một hạng mục trước (*"một bước chuẩn hoá vô hại:
đổi CRLF"*). Viết được phép kiểm cho một chế độ hỏng không có nghĩa là nhận ra nó
khi nó xảy ra với mình.

Chi phí chế độ token, đo trên 60 tài liệu:

| model | `chars` | `tokens` | chậm hơn |
|---|---:|---:|---:|
| BGE-M3 | 0,3 s | 2,7 s | 8,0× |
| PhoBERT | 0,6 s | 23,5 s | 41,3× |

PhoBERT đắt hơn nhiều vì trần 256 **liên tục chạm**, nên vòng cắt lại chạy thật.
Tuyệt đối thì vài giây cho cả corpus — không đáng kể cạnh bước embed (`W1-08`:
hàng chục phút).

## 8. Cái cố ý **không** làm

**Không đổi `configs/indexing/bgem3.yaml` sang chế độ token.** Bật lên là
15.814 → 11.190 chunk, tức index khác, tức mọi con số `W2` không so được nữa. Đổi
`chunk_size` còn là chiều mà `TD-20` đã chỉ ra là **không kiểm định cặp được**
(nhãn neo theo span nên đổi chunking là đổi tập nhãn). Việc đó thuộc `W3-09`
(ablation #2) → **`TD-25`**.

Ba thứ giữ nguyên mặc định để không hạng mục nào đã công bố bị dịch:
`size_unit="chars"`, không có bộ đếm thì **không** gọi tokenizer lần nào (có test
đếm số lần gọi = 0), và `self.sizing is self.config` ở chế độ ký tự.

## 9. Ba ràng buộc mới, và vì sao chúng là ràng buộc

**Ngữ cảnh hàng xóm loại trừ trần token.** `neighbor_context_chars` **cố ý** vượt
`max_chunk_size` — nó là phần đệm chép từ chunk bên cạnh. Nó phá đúng cái trần mà
chế độ token dựng lên, và nó chạy *sau* bước kiểm. Config từ chối ngay lúc dựng.
⚠️ Baseline dùng `nb=100`, nên baseline **không** token-size được nếu không bỏ
padding — một quyết định, không phải một dòng config.
💡 Và lần so đầu tiên của tôi bị đúng cái này làm nhiễu: so baseline (`nb=100`)
với chế độ token (`nb=0`) là so hai thứ khác nhau **hai chỗ**, mà padding 200 ký
tự đủ để một mình nó giải thích chênh lệch.

**Không có bộ đếm thì ném lỗi, không rơi về đếm ký tự.** Rơi về sẽ cho ra một bộ
chunk hợp lệ, khác hẳn cái được yêu cầu, và không gì trong output lộ ra.

**Cận dưới `ge=50` hoá ra là cận dưới tính bằng KÝ TỰ.** `chunk_size=48` token là
ngân sách hoàn toàn hợp lý với BGE-M3, và bị pydantic chặn bởi một ràng buộc chưa
từng nói nó đo bằng gì. Đã chuyển vào validator, chỉ áp khi `size_unit="chars"`.

## 10. Test

34 test ở `tests/unit/test_token_sizing.py`. Ba corpus mẫu chọn theo **chế độ
hỏng** chứ không phải ba đoạn văn: văn xuôi EN (nhiều separator), văn xuôi VI
(mật độ token khác), và **một khối chữ liền không khoảng trắng** — mẫu duy nhất
kiểm được **điều kiện dừng**, vì splitter đệ quy trả nguyên mảnh nên không có
đường cắt cứng thì `fit_to_budget` lặp vô hạn.

Bộ đếm token trong test là **giả và xác định**: thứ cần ghim là thuật toán cắt
lại, không phải tokenizer của HuggingFace, và test không được phụ thuộc 2,2 GB
trọng số. Con số thật nằm ở báo cáo này, sinh lại bằng `make token-probe`.

Bất biến mạnh nhất: **span không bỏ sót ký tự nội dung nào**. Không khẳng định
"ghép lại đúng nguyên bản" vì điều đó **sai** — splitter bỏ ký tự separator ở mỗi
mối nối (cắt theo `". "` thì dấu chấm ở chỗ cắt biến mất khỏi cả `content` lẫn
span). Hành vi ấy có **từ `W1-11`**, không phải do `W3-06`: chế độ ký tự với
`chunk_size=200` mất đúng 60 dấu chấm trên cùng văn bản. Có test riêng ghim nó,
để ai sửa splitter thì biết mình đang đổi mọi bộ chunk đã công bố.

## 11. Còn lại gì

* **`TD-25`** (mới) — chưa biết token sizing có **cải thiện truy hồi** hay không.
  Nó san bằng kích thước giữa hai ngôn ngữ (§4), và đó là *giả thuyết* về
  `cross_lingual`, chưa phải kết quả. Thuộc `W3-09`, và vướng `TD-20`.
* `W3-06` **không đụng** tới `cross_lingual` hay bất kỳ metric nào — không lượt
  index nào được build lại trong hạng mục này, cố ý.
* Chế độ token chưa chạy qua `CachedChunker`: mật độ là hàm của nội dung nên
  khoá cache hiện tại vẫn đúng, nhưng nó cùng họ với ghi chú ở `TD-24`.
