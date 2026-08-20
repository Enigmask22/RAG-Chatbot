# `W1-11` + `W1-13` — golden_v1 đã đóng băng, baseline đã đo

> Ngày: 2026-08-20 · Nhánh: `feat/w1-foundation`
> `golden_v1.jsonl` · 242 câu · sha256 `f53ad84abea32d3f...`
> ⚠️ **Review bởi model, KHÔNG phải người.** Xem §1 trước khi dùng con số này ở đâu.

---

## 1. Ai đã review, và điều đó có nghĩa gì

Người dùng không có thời gian review tay nên **tôi (Claude Opus 5) đã review thay**.
Đây là một lựa chọn hợp lệ nhưng **không tương đương** người, và điều đó được ghi
thẳng vào dữ liệu chứ không chỉ nằm trong report:

```json
{"reviewed_by_human": false, "reviewed_by": "model:claude-opus-5", ...}
```

`freeze` chỉ đặt `reviewed_by_human=true` khi chạy với `--reviewer human`, và nó in
cảnh báo mỗi lần reviewer khác `human`. Lý do làm chặt như vậy: một cờ boolean sai ở
đây sẽ làm report, CV và câu trả lời phỏng vấn cùng nói sai một chuyện, và không ai
phát hiện được.

**Điểm mạnh của cách này:** DeepSeek-v4-flash sinh câu hỏi, Claude Opus 5 review —
đây là **cross-model**, không phải model tự chấm mình. Hai họ model khác nhau, khác
dữ liệu huấn luyện, khác lỗi hệ thống.

**Điểm yếu, nói thẳng:**

* Tôi kiểm được những gì **đối chiếu được với văn bản**: trích dẫn có thật không,
  nhãn nhóm có đúng không, câu hỏi có tự chứa không, corpus có trả lời được câu
  `unanswerable` không. Với những thứ đó tôi tra thẳng văn bản gốc, không đoán.
* Tôi **không** thay được người ở câu hỏi "đây có phải câu mà người dùng thật sẽ
  hỏi hệ thống này không". Đó là phán xét về sản phẩm, và nó thuộc về chủ dự án.
* Nếu tôi và DeepSeek cùng có một điểm mù (ví dụ cùng coi một câu hỏi tối nghĩa là
  rõ ràng), tôi không phát hiện ra được.

**Việc nên làm khi có thời gian:** đọc lại 33 câu `unanswerable` và 43 câu
`cross_lingual` — hai nhóm tôi loại nhiều nhất, tức hai nhóm tôi kém tự tin nhất.
Xong thì `make goldenset-freeze` lại với `--reviewer human`.

---

## 2. Kết quả review: 242 nhận / 24 loại

| Nhóm | nháp | nhận | loại |
|---|---:|---:|---:|
| factoid | 78 | 68 | 10 |
| cross_lingual | 46 | 43 | 3 |
| unanswerable | 40 | 33 | 7 |
| adversarial | 36 | 34 | 2 |
| multi_hop | 34 | 34 | 0 |
| aggregation | 28 | 26 | 2 |
| table_lookup | 4 | 4 | 0 |
| **tổng** | **266** | **242** | **24** |

### Vì sao loại — năm loại lỗi

**(a) 7 câu `unanswerable` mà corpus THẬT SỰ trả lời được.** Đây là `TD-09`, và nó
chỉ lộ ra khi tra thẳng văn bản gốc — retriever không đủ để kết luận vì nó là model
đơn ngữ và chỉ nhìn 256 token đầu mỗi chunk (`TD-11`). Ví dụ:

| Câu hỏi | Corpus có gì |
|---|---|
| "projected inflation rate for Vietnam in 2025" | *Điểm lại 9-2025*: "lạm phát vẫn nằm dưới chỉ tiêu của NHNN là **4,5-5% cho năm 2025**" |
| "tăng trưởng GDP Việt Nam năm 2030 dự kiến bao nhiêu" (2 câu) | Bảng CGE của CCDR: `real gdp ... 2030 → **5,45**` |
| "số thủ tục tiền kiểm được khuyến nghị bãi bỏ 2021–2025" | Báo cáo ghi rõ **1.584** thủ tục cấp phép trước khi hoạt động |
| "tỷ lệ nghèo đa chiều Việt Nam năm 2022" | *Cập nhật Tình trạng Nghèo* có đồ thị nghèo đa chiều TCTK tới 2022 |

Model sinh câu hỏi chỉ thấy vài chunk trong **một** tài liệu, nên nó không có cách
nào biết 59 tài liệu còn lại nói gì. Tỉ lệ sai 7/40 = **17,5%** trong nhóm này.

**(b) 4 câu không tự chứa.** `"According to the passage..."`, `"theo báo cáo này"`,
`"this study"`, `"theo hướng dẫn báo cáo"`. Lúc eval không có "passage" nào —
hệ thống phải tự truy hồi. Script sàng bắt được 0 câu ở lần đầu vì regex của tôi
thiếu mấy biến thể; ba câu còn lại tìm ra khi đọc tay.

**(c) 7 câu tra cấu trúc tài liệu, không phải nội dung.** Số trang của một mục,
tiêu đề của Hộp 3, "Table 6 tóm tắt gì", số hiệu working paper 5493, tác giả trong
danh mục tham khảo, ai xuất bản báo cáo về dementia. Số trang đặc biệt vô nghĩa —
chúng là artifact của việc làm phẳng PDF. `W1-10` đã lọc trang bìa khỏi **việc chọn
mẫu**, nhưng chú thích và thư mục tham khảo thì lọt qua.

**(d) 3 câu trùng ý.** Hai câu `adversarial` cùng về cải cách thuế TNDN 2003; hai câu
cùng về ai xây dựng kịch bản phát thải GTVT; hai câu `factoid` dẫn từ **cùng một
câu** về FDI/năng suất. Khử trùng lặp Jaccard 0,8 của `W1-10` không bắt được vì
*cách hỏi* khác nhau — chỉ *nguồn* mới trùng. Đó là lý do tôi thêm phép sàng theo
vùng span.

**(e) 3 câu quá mơ hồ hoặc vòng tròn.** "thành phố nào ngày càng được ưa chuộng"
(điền vào chỗ trống); "ngành nào bị ảnh hưởng nặng nhất" (có nhiều bản *Điểm lại*,
không xác định được bản nào); và một câu tả báo cáo rồi hỏi chính tiêu đề của nó.

---

## 3. Quy trình review — sáu phép kiểm máy + đọc tay 266 câu

Không dựa vào một phép nào. Mỗi phép kiểm chạy trên **toàn bộ** 266 câu:

| Phép kiểm | Kết quả |
|---|---|
| Trích dẫn có trong văn bản gốc? (bỏ qua khác biệt khoảng trắng) | 16 câu `quote_unverified` → **15 có thật**, 1 loại vì lý do khác |
| Con số trong đáp án có quanh vùng bằng chứng? | 12 cờ → **0 lỗi thật** (đều là số của tiền đề sai, hoặc nhóm không có span) |
| Câu hỏi tự chứa? | 1 câu qua regex + 3 câu qua đọc tay |
| `multi_hop`/`aggregation` có ≥2 span? | **60/60 đạt** |
| `cross_lingual` có thật khác ngôn ngữ? | **43/43 đạt** |
| `adversarial` có tiền đề sai trong đáp án? | **36/36 đạt** |
| Nhóm câu dẫn từ cùng vùng 200 ký tự | 15 nhóm → 3 câu trùng ý bị loại |
| Câu hỏi về cấu trúc tài liệu | 4 qua regex + 3 qua đọc tay |

Rồi đọc tay từng câu, kèm text bằng chứng thật lấy từ tài liệu gốc.

**`quote_unverified` không phải dấu hiệu model bịa** — đó là dấu hiệu **lỗi trích
xuất PDF**. Bốn nguyên nhân đo được:

* trộn hai cột chèn chữ vào giữa từ: `"bổ sung"` → `"bổ chuyên sung"`
* số chú thích chèn giữa câu: `"revenue collection 24 was 67 percent"`
* gạch nối cuối dòng: `"mid- November"`
* trích dẫn vắt qua biên chunk

Và một chỗ đáng ghi lại: **phép kiểm máy sửa lại nhận định của tôi.** Đọc tay tôi
tưởng `factoid-69f6fec136` (2.938 tỷ đồng PFES) không có căn cứ vì số đó không nằm
trong span. Phép kiểm grounding chứng minh nó có thật, chỉ nằm ngoài span đã thu hẹp.
Nếu tin mắt mình thì đã loại oan một câu tốt.

---

## 4. `W1-13` — baseline

```
$ make eval-retrieval BUNDLE=baseline
```

242 câu · **209 câu được chấm** (33 câu `unanswerable` trả `None`, đo riêng ở `W5-02`)
· index fingerprint `72c87744d258ed2c` · top-k 20

| Metric | Giá trị |
|---|---:|
| recall@1 | **0,0877** |
| recall@5 | **0,1746** |
| recall@10 | 0,2257 |
| recall@20 | 0,2663 |
| MRR | **0,1660** |
| nDCG@10 | 0,1621 |
| MAP@20 | 0,1349 |
| HitRate@5 | 0,2153 |
| p50 / p95 độ trễ | **22,5 / 39,9 ms** |

### Theo nhóm — chỗ có thông tin nhất

| Nhóm | n | recall@5 | recall@20 | MRR |
|---|---:|---:|---:|---:|
| factoid | 68 | **0,3088** | 0,3897 | 0,2162 |
| multi_hop | 34 | 0,2157 | 0,3480 | 0,2720 |
| adversarial | 34 | 0,1618 | 0,2647 | 0,1361 |
| aggregation | 26 | 0,1026 | 0,2436 | 0,2258 |
| cross_lingual | 43 | **0,0000** | 0,0465 | 0,0058 |
| table_lookup | 4 | 0,0000 | 0,0000 | 0,0000 |

| Ngôn ngữ | n | recall@5 | recall@20 |
|---|---:|---:|---:|
| vi | 127 | 0,2073 | 0,3333 |
| en | 82 | 0,1240 | 0,1626 |

### Đọc con số này thế nào

`recall@5 = 0,17` là **thấp**, và nó thấp có lý do — hai lý do đã được đo riêng ở
`reports/w1-11-triage.md`, không phải đoán:

1. **`cross_lingual` bằng 0.** 43/209 câu, tức 20% tập đo, dùng model embedding
   **đơn ngữ** (`vietnamese-bi-encoder`, nền PhoBERT). recall@5 = 0,0000 không phải
   lỗi đo — đó là con số đúng của một model không có căn chỉnh đa ngữ. Chỉ riêng
   nhóm này đã kéo recall@5 tổng thể xuống khoảng 20%.
2. **56,8% chunk bị cắt ở 256 token** (`TD-11`), làm 15,7% văn bản không bao giờ tới
   được vector.

**Cố ý không sửa hai thứ đó trước khi đo.** Baseline có nhiệm vụ đo bản POC *như nó
đang là*; sửa trước rồi gọi kết quả là "baseline" thì mọi so sánh "+X%" về sau đã
tính lẫn cải tiến này vào và không ai truy ra được.

Ngược lại, thấp như vậy là **tin tốt cho `W2`**: hai hạng mục đầu tiên (BGE-M3 đa ngữ,
hạ `chunk_size`) đều nhắm thẳng vào hai nguyên nhân đã định lượng được.

`table_lookup` bằng 0 với n=4 thì **không suy ra được gì** — corpus `.txt` đã làm
phẳng bảng, nhóm này chờ nguồn (c) HOSE + `W3-01`.

### DoD: chạy lại hai lần sai số <1%

Chạy lần hai với `DEEPSEEK_API_KEY=""` và `OPENROUTER_API_KEY=""`:

```
sai số lớn nhất trên 15 metric: 0,0000%  →  ĐẠT
```

Không phải "dưới 1%" mà là **bằng 0** — dense retrieval trên vector đã ghi là phép
tính xác định. Lượt chạy này đồng thời là bằng chứng cho gate `G1`: eval retrieval
chạy **không cần bất kỳ LLM API nào**.

---

## 5. Lỗi bắt được lúc chạy thật: `freeze` làm rơi `relevant_spans`

Phát hiện khi thấy `config.span_resolution` **không có** trong JSON của lượt đo đầu.
`_apply` dựng `GoldenQuery` mà không truyền `relevant_spans` → `golden_v1.jsonl` chỉ
còn `chunk_id`, tức toàn bộ công neo span của `TD-12` bị mất ở đúng bước cuối cùng.

Không có triệu chứng nào: file vẫn hợp lệ, eval vẫn chạy, số vẫn ra. Chỉ là nhãn lại
phụ thuộc cấu hình chunking, đúng thứ vừa bỏ công sửa.

Đã sửa, kèm một quyết định không hiển nhiên: **`edit` có điền
`new_relevant_chunk_ids` thì span bị BỎ.** Vì ánh xạ span *ghi đè* `relevant_chunk_ids`
ở eval, nên giữ span cũ sẽ làm chỉnh sửa tay của người review bị bỏ một cách âm thầm
— thứ tệ nhất có thể làm với công review tay. Đổi nhãn nhóm thôi thì span vẫn giữ.
Bốn test canh cả bốn nhánh (`TestSpansSurviveFreeze`).

Sau khi sửa: 209/209 câu trả lời được đều có span, `unmatched_queries: []`,
`label_changed: 9`.

---

## 6. Việc còn lại

| Việc | Ghi chú |
|---|---|
| Người đọc lại 33 câu `unanswerable` + 43 câu `cross_lingual` | Hai nhóm tôi loại nhiều nhất = hai nhóm tôi kém tự tin nhất. Xong thì freeze lại với `--reviewer human` |
| Dọn `reference_answer` bị lẫn ngữ cảnh sinh | ~6 câu có `"Theo đoạn 1"`, `"trong các đoạn văn được cung cấp"`. Không ảnh hưởng eval retrieval (không dùng `reference_answer`) nhưng sẽ nhiễu ở `W5` |
| `table_lookup` chỉ 4 câu | Freeze sẽ **từ chối** nếu nhóm này về 0. Chờ nguồn (c) HOSE + `W3-01` |
| `unanswerable` thiếu đa dạng | ~12/33 câu theo cùng một khuôn "hỏi về nước khác" (Thái Lan, Malaysia, Lào, Singapore, Hàn Quốc). Nhóm này đo ở `W5-02`, nên trước đó cần thêm khuôn khác |
| Nhãn liên quan chưa đầy đủ | Một số dữ kiện xuất hiện ở **hai** tài liệu (ví dụ "92% tài sản hệ thống tài chính" có trong cả *Financial sector assessment* và *Taking stock*). Hệ thống truy hồi bản kia sẽ bị chấm sai. Phép đo Jaccard ở `w1-11-triage.md` §4.2 cho thấy chuyện này hiếm (1/78) nhưng có thật |

---

## 7. Lệnh tái lập

```bash
make up && make index BUNDLE=baseline      # "index 0 · bỏ qua 60"
make goldenset-verify                      # checksum f53ad84abea32d3f…
make eval-retrieval BUNDLE=baseline
```

Đọc `config.span_resolution` trước khi đọc recall: `unmatched_queries` không rỗng
nghĩa là có câu bị chấm 0 vì lý do **không phải** retrieval.
