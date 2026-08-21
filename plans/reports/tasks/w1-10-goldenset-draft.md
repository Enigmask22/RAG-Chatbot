# W1-10 — Sinh nháp golden set: bằng chứng nghiệm thu

> Ngày chạy: 2026-08-17 · DeepSeek `deepseek-v4-flash` · index `rag_baseline` (15.814 chunk)
> Phạm vi: `W1-10`. Đầu ra là **bản nháp**, không phải golden set — đóng băng ở `W1-11`.

## Cách tái lập

```bash
make up && make index          # cần index baseline trước
make goldenset-dry             # xem phân bố lô + prompt mẫu, KHÔNG tốn tiền
make goldenset-draft           # TỐN TIỀN API
```

## Kết quả

```
lời gọi          163 (hỏng 0)
câu              288 thô → 266 giữ lại (bỏ trùng 0 · sai chỉ số 0 · sai schema 22)
  adversarial     36        cross_lingual   46        multi_hop       34
  aggregation     28        factoid         78        table_lookup     4
                            unanswerable    40
ngôn ngữ         vi 167 · en 99      ·  trải trên 44/60 tài liệu
cần đọc kỹ       16 (trích dẫn không kiểm chứng được 16 · trôi nhóm 0)
token            222.088 vào · 506.567 ra
chi phí          $0,5821  ($0,00219 mỗi câu giữ lại)
thời gian        640,6s (song song 6)
```

- Đầu ra: [`data/golden/draft_v1.jsonl`](../../data/golden/draft_v1.jsonl) — **266 câu** (DoD ≥ 250 ✅)
- Báo cáo máy đọc được: [`reports/goldenset/goldenset-draft.json`](goldenset-draft.json)

| Kiểm tra | Lệnh | Kết quả |
|---|---|---|
| Lint | `ruff check .` + `ruff format --check .` | All checks passed |
| Type | `mypy` (strict) | Success: no issues found in **62** source files |
| Unit test | `pytest -m "not integration and not gpu"` | **317 passed** · 2,2s |

Test mới: `test_llm_provider.py` (**23 case**), `test_goldenset_gen.py` (**35 case**),
`test_goldenset_dedupe.py` (**19 case**), `test_goldenset_sampling.py` (**23 case**).
Toàn bộ chạy bằng `httpx.MockTransport` — không chạm mạng, không tốn tiền.

## Bốn phát hiện khi chạy thật

### 1. `deepseek-chat` là **bí danh**, không phải một model

Xác nhận trực tiếp trên API ngày 2026-08-17:

| Slug yêu cầu | Model thực tế phục vụ |
|---|---|
| `deepseek-chat` | `deepseek-v4-flash` |
| `deepseek-reasoner` | `deepseek-v4-flash` |
| `deepseek-v4-flash` | `deepseek-v4-flash` |

Đây **đúng là vấn đề mà quy tắc cứng #1 nói về OpenRouter preset**, chỉ kín đáo
hơn: tên trông như một model cụ thể nhưng thật ra là con trỏ do server nắm. Đo
bằng bí danh thì con số tháng này không so được với tháng sau và không có gì báo.

Đã xử lý: mặc định của dự án đổi sang slug thật `deepseek-v4-flash`; gọi bằng bí
danh vẫn được nhưng ghi cảnh báo; `LLMResponse.model` luôn là model **thực tế**
đã phục vụ, và mọi lần trôi đều được log.

### 2. Model suy luận làm hỏng chẩn đoán về `max_tokens`

Triệu chứng ban đầu: hàng loạt `Response không phải JSON hợp lệ` với nội dung
rỗng hoặc bị cắt giữa chuỗi. Nhìn thì tưởng model trả rác.

Đo lại một lời gọi: `completion_tokens = 1770` nhưng `len(text) = 515` ký tự.
`deepseek-v4-flash` tiêu **1.500–3.000 token cho chuỗi suy luận không nằm trong
`content`**, rồi mới viết JSON. Với ngân sách 2.000 thì phần suy luận ăn hết,
JSON bị cắt hoặc không kịp bắt đầu.

Đã xử lý: nâng `max_tokens` lên 6.000; đọc `completion_tokens_details.reasoning_tokens`;
và tách `finish_reason == "length"` thành một cảnh báo riêng nói thẳng "hết ngân
sách token, tăng `--max-tokens`" thay vì đổ lỗi cho JSON.

⚠️ **Còn 22/163 lời gọi (13,5%) vẫn bị cắt ở mức 6.000** — toàn bộ 6.000 token
đi vào suy luận. Xem `TD-08`.

### 3. Chạy tuần tự lãng phí một tiếng đồng hồ

Mỗi lời gọi ~25 giây, và gần như toàn bộ là **chờ mạng**. 163 lô tuần tự = hơn
một tiếng máy ngồi không. Chạy song song 6 luồng: **640 giây**.

Kết quả vẫn được ráp lại theo đúng thứ tự lô, nên hai lần chạy cùng seed cho ra
cùng một file.

### 4. Job trả tiền dài mà không có checkpoint

Lượt chạy đầu tiên treo ở phút thứ 40 và mất sạch. Đã thêm checkpoint ghi nối
sau mỗi lời gọi: chạy lại bỏ qua lô đã xong. Lô mà model trả về rỗng cũng được
ghi dấu — nếu không thì mỗi lần chạy lại đều trả tiền cho cùng một câu trả lời
rỗng đó.

## Chất lượng corpus: hai bộ lọc phải thêm

Prompt đầu tiên đưa cho model đoạn văn thế này:

```
ividual indexes on new orders, output,                minus the number of existing firms
employment, suppliers' delivery times (and stock of   suspending their operations; net entry
```

Bản `.txt` mà World Bank trích sẵn **giữ nguyên vị trí ký tự của trang PDF hai
cột**, nên cột trái và cột phải bị đan xen theo từng dòng. Nó gồm toàn từ tiếng
Anh hợp lệ nên mọi bộ lọc theo tỉ lệ chữ cái đều cho qua.

| Bộ lọc | Tín hiệu | Đo được trên corpus |
|---|---|---|
| `gutter_ratio ≤ 0.3` | khoảng trắng dài ở **giữa** dòng = máng phân cột | 27,8% chunk vượt ngưỡng |
| `mean_words_per_line ≥ 10` | văn xuôi ~13–15 từ/dòng, chú thích biểu đồ 2–5 | loại thêm ~19% |
| `skip_leading_chunks = 6` | bìa, trang bản quyền, lời cảm ơn, mục lục | 6 chunk đầu mỗi tài liệu |

Tổng cộng khoảng **60% lô bị loại**, nên hệ số lấy dư `overshoot` đặt 3,0.

Quan trọng: ba bộ lọc này **chỉ áp cho việc chọn mẫu sinh câu hỏi**, không áp cho
index. Chunk trộn cột vẫn nằm trong Qdrant và retriever vẫn có quyền trả về —
ta chỉ từ chối *đặt câu hỏi* lên chúng, vì câu hỏi sinh từ văn bản trộn cột chỉ
đo được chính lỗi trích xuất. Docling ở `W3-01` mới là cách sửa thật.

## Những gì code kiểm, thay vì đẩy hết cho người review

1. **Model không được tự viết `chunk_id`.** Nó chỉ trả về **chỉ số** của đoạn văn
   trong danh sách được đưa; ánh xạ sang `chunk_id` thật do code làm. Chỉ số
   ngoài khoảng → bỏ câu (0 ca trong lượt này).
2. **`quote` phải tìm thấy trong chunk được viện dẫn**, so sau khi gom khoảng
   trắng. Không khớp thì vẫn giữ câu nhưng đánh dấu — 16/266 câu.
3. **Hạ nhóm khi model không làm đúng việc.** `multi_hop` chỉ dùng một chunk thì
   nó là `factoid`; giữ nhãn sai sẽ làm cột `multi_hop` trong bảng breakdown báo
   một năng lực chưa từng được đo. 0 ca trong lượt này.
4. **`unanswerable` bị ép `relevant_chunk_ids` rỗng** ở tầng schema.

## Nhóm `table_lookup` chỉ có 4 câu

Đây là **kết quả đúng, không phải lỗi**. Prompt dặn model trả về danh sách rỗng
nếu đoạn văn không có bảng, và corpus hiện chỉ có bản `.txt` với bảng đã bị làm
phẳng. Nhóm này chỉ thật sự đo được khi có nguồn (c) báo cáo thường niên HOSE
và Docling (`W3-01`). Đừng nặn thêm câu cho đủ số — 4 câu thật tốt hơn 20 câu
hỏi về một cái bảng không còn là bảng.

## Việc của `W1-11`

266 nháp → ≥150 câu đã người xác nhận. Thứ tự đọc nên theo mức rủi ro:

1. **16 câu `needs_close_review`** (trích dẫn không kiểm chứng được) — đọc trước.
2. **40 câu `unanswerable`** — nhóm dễ sai nhất: model có thể vô tình hỏi thứ mà
   một tài liệu *khác* trong corpus trả lời được. Code không kiểm được điều này.
3. **34 câu `multi_hop`** — kiểm rằng bỏ một trong hai chunk thì thật sự không
   trả lời được.
4. Phần còn lại đọc lướt.

## Nợ kỹ thuật ghi nhận

| Nợ | Lý do chấp nhận | Kế hoạch trả |
|---|---|---|
| 22/163 lời gọi bị cắt ở `max_tokens=6000` | Vẫn đủ 266 câu, và nâng tiếp thì tốn tiền cho phần suy luận không dùng | Thử `--questions-per-call 1` (prompt ngắn hơn → suy luận ngắn hơn) khi cần thêm câu |
| `table_lookup` mới có 4 câu | Corpus chưa có bảng thật | Nguồn (c) HOSE + `W3-01` |
| Chưa kiểm được câu `unanswerable` có thật sự không trả lời được từ corpus | Cần chạy retrieval trên toàn corpus cho từng câu | Kiểm bằng chính retriever ở `W1-11`: câu nào retriever trả về chunk có điểm rất cao thì phân loại lại |
