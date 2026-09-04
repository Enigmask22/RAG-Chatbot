# `W5-01` + `W5-02` — Eval tầng sinh: hai lỗi production trước khi có con số đầu tiên

*2026-09-04 · `pipeline/eval/answer_run.py`, `pipeline/eval/generation_metrics.py`,
`pipeline/eval/prompts/judge-*.yaml` · 39 test mới · chi phí $0,85*

---

## 0. Tóm tắt

| Metric | Giá trị | Mẫu số | Mục tiêu | |
|---|---|---|---|---|
| **Faithfulness** (mệnh đề có trích nguồn) | **0,9877** | 407 mệnh đề | ≥ 0,92 | ✅ |
| **Overall groundedness** (mọi mệnh đề thật) | **0,9568** | 532 mệnh đề | — | ✅ |
| **Citation accuracy — cấp quote** | **0,8308** | 396 trích dẫn | ≥ 0,85 | ❌ **thiếu 0,019** |
| **Citation accuracy — cấp mệnh đề** | **0,9877** | 407 mệnh đề | ≥ 0,85 | ✅ |
| **Refusal correctness** | **0,9091** | 242 truy vấn | ≥ 0,85 | ✅ |
| Answer relevancy | 0,7479 | 242 | — | (xem §6) |
| Context recall@5 | 0,7887 | 209 | — | |
| Context precision@5 | 0,2096 | 209 | — | |
| Misattribution | 0,0049 | 407 | — | |
| Citation coverage (mọi câu) | 0,6186 | 700 câu | — | |
| Citation coverage (chỉ mệnh đề thật) | 0,7650 | 532 mệnh đề | — | |

Nguồn: `runs/w5-answers-v1.jsonl` (242 câu qua HTTP thật, $0,2588) và
`runs/w5-answers-v1-generation.json` (947 phán quyết judge, $0,5083).

---

## 1. ⭐⭐ Harness tìm ra hai lỗi production trước khi in được con số nào

Cả hai đều nằm ngoài phạm vi `W5-01`. Cả hai đều **không** test nào của 13 hạng
mục `W4` bắt được. Và cả hai đều làm `POST /chat` trả 503.

### 1.1 Image trôi khỏi môi trường đã đo — và `runtime_drift` vẫn báo `null`

Lượt chạy thử 6 câu đầu tiên: 5/6 trả
`RuntimeError: mat1 and mat2 must have the same dtype, but got Half and Float`.

Truy ra: cross-encoder nạp với embedding fp16 nhưng `key.weight` fp32. Nguyên
nhân không nằm ở mã dự án:

```
uv.lock / môi trường dev : transformers 5.15.0 · sentence-transformers 5.7.0 · torch 2.13.0
container                 : transformers 5.16.1 · sentence-transformers 6.0.1  · torch 2.14.0
```

`serving/Dockerfile` của `W4-13` cài bằng `uv pip install -r pyproject.toml` —
**giải lại phụ thuộc từ đầu mỗi lần build**. `uv.lock` được `COPY` vào chỉ để làm
khoá cache lớp. Bốn lần rebuild trong vài giờ (chính bộ tiêm lỗi của `W4-13`) đủ
để `sentence-transformers>=3.0` nhảy lên bản 6.

⚠️ **Và chú thích ngay trên dòng ấy — do tôi viết ở `W4-13` — khẳng định điều
ngược lại**: *"cùng resolver với môi trường dev, nên image không thể lệch phiên
bản so với `uv.lock` mà không ai thấy."* Một chú thích không kiểm được là một lời
hứa, không phải một bảo đảm.

⚠️⚠️ **Suốt thời gian đó `GET /admin/bundle` trả `runtime_drift: null`.** Phép
kiểm danh tính `TD-38` soi tên model / thiết bị / dtype / max_length — **không**
soi phiên bản thư viện. Nên artifact vẫn khai "đúng hệ thống đã đo" trong khi hệ
thống đã khác, và khác tới mức không phục vụ được. → `TD-62`.

**Sửa**: `uv sync --locked --no-install-project`. `--locked` mạnh hơn `--frozen`
ở chỗ nó **đỏ** khi `uv.lock` không còn khớp `pyproject.toml`. Sau khi build lại,
container báo đúng `5.15.0 / 5.7.0 / 2.13.0`.

**Ghim lại bằng hai lớp**, vì một chú thích đã thất bại một lần:
* `tests/unit/test_image_reproducible.py` — tĩnh, chạy trong `make test`: Dockerfile
  phải dùng `uv sync --locked`, không được có `-r pyproject.toml`, `uv.lock` phải
  được `COPY` trước bước cài.
* `tests/e2e/test_smoke.py::test_the_container_runs_the_versions_the_lockfile_pins`
  — hỏi thẳng container rồi so với `uv.lock`. **Đã chứng minh đỏ**: sửa version
  `sentence-transformers` trong lock thành `6.0.1` ⇒ test đỏ với đúng thông điệp
  *"container chạy 5.7.0 nhưng uv.lock ghim ['6.0.1'] — image đã trôi khỏi môi
  trường đã đo"*.

### 1.2 Hai người dùng hỏi cùng lúc là đủ để hỏng

Sửa xong dtype thì lộ ra lỗi thứ hai: `RuntimeError: Already borrowed`, 4/6
request với 3 luồng, **0/8 khi chạy tuần tự**.

Tokenizer "fast" của HuggingFace là đối tượng Rust dùng `RefCell`, và
`sentence-transformers` sửa cấu hình truncation/padding của nó **ngay trong** lời
gọi. `ChatService.prepare()` đưa truy hồi vào `asyncio.to_thread`, nên hai
request đồng thời chạm cùng một tokenizer.

`W4-13` không thấy vì **mọi phép đo ở đó đều tuần tự** — 7 test e2e chạy nối
tiếp, probe độ trễ cố ý chạy tuần tự để không tự tạo tải.

⚠️ **Bản vá đầu tiên không đủ, và cách nó không đủ đáng ghi lại.** Tôi khoá
`HuggingFaceEmbeddingProvider._encode` và `CrossEncoderReranker.score` — container
vẫn hỏng 6/9. Lý do: `BgeM3EmbeddingProvider` **ghi đè** `_encode` và gọi thẳng
`_forward`, nên khoá đặt ở lớp cha bị đi vòng hoàn toàn. Một khoá ở lớp cha trông
như đang bảo vệ và không bảo vệ gì.

**Sửa**: khoá theo **khoá cache của model** (`lru_cache` cùng tham số), không theo
instance — hai `CrossEncoderReranker` khác nhau vẫn dùng chung một `CrossEncoder`.
Dùng `RLock` để một đường gọi lồng nhau về sau không tự khoá chính nó.

⚠️ Giá phải trả nói thẳng: các lời gọi bị **tuần tự hoá**. Đó là đánh đổi đúng ở
đây — một GPU 8 GB không chạy song song hai lượt forward nhanh hơn chạy lần lượt,
còn cách kia (mỗi luồng một tokenizer) nhân đôi VRAM để mua thứ phần cứng không
cho. Nhưng nó có nghĩa là **thông lượng của một worker bị chặn bởi GPU**, và đó
là số liệu mà `W6-05` (load test) phải đo.

Ba test hồi quy: `test_reranker.py::TestConcurrentCallsAreSerialised`,
`test_embedding.py::TestConcurrentEncodeIsSerialised`,
`test_bge_m3.py::TestConcurrentForwardIsSerialised` — cái cuối ghim đúng đường đi
vòng của lớp con. Đã chứng minh đỏ khi gỡ khoá.

---

## 2. ⭐⭐ Đo qua HTTP, không dựng lại đường sinh trong pipeline

`tests/unit/test_architecture_boundaries.py` cấm `pipeline` import `serving`. Nên
có hai cách lấy câu trả lời để chấm, và cách hiển nhiên là cách sai.

Dựng lại đường sinh bằng `rag_core` nghe gọn, nhưng đường sinh **thật** gồm bộ
định tuyến `W4-07`, bản viết lại câu hỏi, mốc nonce `W4-12` và `ChatTurn.prompt()`
— tất cả sống ở `serving/`. Một bản dựng lại sẽ trôi khỏi bản thật ngay lần sửa
prompt đầu tiên, và từ giây ấy báo cáo faithfulness nói về một prompt không ai
phục vụ. Đúng họ với lỗi mà `TD-38` sinh ra để chặn — và §1.1 vừa cho thấy họ lỗi
ấy có thật.

Nên harness gọi `POST /chat` như một client. Giá: eval sinh cần stack đang chạy,
một phụ thuộc mà eval truy hồi không có. Đổi lại: mọi con số nói về **hệ thống
người dùng gặp**.

**Tách "chạy" khỏi "chấm"**, đúng khuôn `evaluate_run` của `W1-08`:
`answer_run.py` chỉ sinh và ghi JSONL; `generation_metrics.py` chấm trên file.
Nên chấm lại bằng rubric mới không cần stack, và lần chạy tốn tiền xảy ra một lần.

**Sidecar nội dung chunk** (`w5-answers-v1-chunks.jsonl`, 934 chunk, thiếu 0):
khung SSE `sources` cố ý không mang nội dung chunk — gửi vài KB cho client mỗi
request là lãng phí, và đó là quyết định đúng của `W4-06`. Nhưng judge cần nó.
Lấy từ index lúc chấm thì con số phụ thuộc một index có thể đã đổi; nên nội dung
được đông cứng ngay sau lần chạy, thành file riêng (một chunk hay được 5–10 truy
vấn dùng chung).

**Lần chạy**: 242/242, **0 lỗi**, mọi câu đi nhánh `retrieve`, 1 lượt trúng cache
ngữ nghĩa (hai câu golden giống nhau ≥ 0,96 — đã xoá cache trước khi chạy nên
không phải nhiễu từ lần trước).

---

## 3. ⭐⭐ Tách mệnh đề bằng luật, và chấm với **đúng nguồn nó trích**

RAGAS phân rã câu trả lời thành mệnh đề nguyên tử bằng một lời gọi LLM nữa. Làm
vậy đặt **hai** model không tất định vào cùng một phép đo, và khi con số dịch
chuyển thì không quy được cho model nào. Ở đây phân rã là tách câu bằng luật:
tất định, miễn phí, chỉ còn một LLM trong chuỗi — và cái đó có cache.

⚠️ Giá: câu ghép chứa một mệnh đề đúng và một mệnh đề sai bị tính là **cả câu
không được chống đỡ**. Phép đo nghiêm hơn mức mệnh đề và lệch về phía báo
faithfulness *thấp hơn* thực tế. Với metric an toàn đó là chiều lệch đúng.

Luật 2 của `chat-system` bắt mỗi ý mang `[n]`, nên mỗi câu **tự khai** nó dựa vào
nguồn nào — và phép đo đúng là hỏi *nguồn ấy*. Chấm với hợp mọi chunk đã truy hồi
trả lời một câu yếu hơn ("có suy ra được từ thứ gì ta đã lấy về không"), cho điểm
cao hơn, và **giấu mất lỗi gán nhầm nguồn**.

`score_misattribution` đo phần bị giấu, nhưng **chỉ hỏi lại những mệnh đề đã
trượt** ở vòng theo-nguồn-trích: nếu chunk được trích đã chống đỡ mệnh đề thì
theo định nghĩa không có gì để gán nhầm. Chạy cả hai vòng đầy đủ tốn gấp 2,6× để
mua những câu trả lời đã biết.

**Kết quả: misattribution 0,0049** — 2/407. Số nguồn gần như luôn đúng.

---

## 4. ⭐⭐ Điểm mù: 38% nội dung nằm ngoài mọi phép kiểm

`faithfulness` chỉ chấm được mệnh đề **có** `[n]`. Trong lần chạy này là 433/700
câu. Tức hơn một phần ba nội dung câu trả lời không đi qua cơ chế `W4-09` lẫn
judge. Công bố `faithfulness = 0,95` mà không nói ra là báo một con số đúng về hai
phần ba câu trả lời rồi để người đọc tưởng nó nói về toàn bộ.

Nên phần không trích được hỏi lại với hợp mọi chunk (`score_uncited_grounding`).
Nó tách hai chế độ hỏng hoàn toàn khác nhau: **có căn cứ nhưng quên trích** (lỗi
hình thức, sửa bằng prompt) và **không có căn cứ** (bịa, đã lọt qua mọi hàng rào).

---

## 5. ⭐⭐ Phép đo suýt sai, và cái chặn nó là đọc 12 ví dụ thật

Vòng chấm đầu (rubric `judge-faithfulness@v1`):

```
uncited_grounding = 0,427   (n = 267)
```

Tôi đã định viết *"22% nội dung câu trả lời không có căn cứ"*. Rồi đọc ví dụ:

```
- Dựa trên ngữ cảnh được cung cấp:
- Based on the provided context:
- tôi không đủ thông tin để trả lời đầy đủ câu hỏi của bạn về…
- Lưu ý: Ngữ cảnh không cung cấp thông tin so sánh với thị trường Mỹ và EU…
- Do đó, tôi không thể trả lời đầy đủ câu hỏi của bạn.
```

Gần như toàn bộ là câu **meta**: model đang **tuân thủ luật 3** của `chat-system`
("nếu ngữ cảnh không đủ, hãy nói thẳng"). Judge chấm `NOT_FOUND` là **đúng theo
rubric v1** — ngữ cảnh quả thật không chứa lời khẳng định nào về việc chính nó
thiếu thông tin. Một lời từ chối trung thực đang bị đếm thành một lỗi trung thực.

**Sửa: rubric lên v2, thêm nhãn `NO_CLAIM`.** Và đây là lần đầu cơ chế của
`W4-11`/`W5-03` được thử ở chỗ nó sinh ra để phục vụ: `prompt_sha256` nằm trong
khoá cache judge, nên **đổi rubric tự động vô hiệu hoá toàn bộ phán quyết cũ**.
Không một verdict v1 nào lọt vào con số v2, và tôi không phải nhớ xoá cache.

| metric | rubric v1 | rubric v2 | |
|---|---|---|---|
| `faithfulness` | 0,9538 (n=433) | **0,9877** (n=407) | 26 câu là `NO_CLAIM` |
| `uncited_grounding` | 0,4270 (n=267) | **0,8560** (n=125) | 142 câu là `NO_CLAIM` |

**Cùng một hệ thống, cùng những câu trả lời, chênh 2×.** Con số là thuộc tính của
cặp *(hệ thống, rubric)*; công bố nó mà không công bố rubric là công bố một nửa.
Đó là lý do rubric có version, version nằm trong khoá cache, và cả hai đi vào
report.

**Tỉ trọng câu meta: 0,24** — gần một phần tư câu trong câu trả lời không phải
mệnh đề. Con số ấy làm `citation_coverage` tất định (0,6186, mẫu số = mọi câu) đọc
sai: luật 2 không đòi câu dẫn phải trích nguồn. Mẫu số đúng cho **mệnh đề thật**
là **0,7650**.

---

## 6. `W5-02` — từ chối đúng chỗ, và hai cấp của "độ chính xác trích dẫn"

### Refusal correctness

Dùng lại **đúng những nhãn** `score_relevancy` đã lấy — không một lời gọi judge
nào thêm. `REFUSAL` là nhãn của rubric chứ không phải phép dò từ khoá: "không đủ
thông tin", "các nguồn không nêu rõ", "I could not find" và mười cách nói khác
đều là cùng một hành vi, còn một danh sách từ khoá chỉ bắt được những cách nói
tôi nghĩ ra được.

| | giá trị | ý nghĩa |
|---|---|---|
| `refusal_recall` | **0,9091** (30/33) | trong 33 câu `unanswerable`, bao nhiêu câu được từ chối |
| `false_refusal_rate` | **0,0909** (19/209) | trong câu trả lời được, bao nhiêu câu bị từ chối oan |
| `refusal_accuracy` | **0,9091** (220/242) | quyết định đúng trên toàn tập — đối chiếu ngưỡng ≥ 0,85 ✅ |

Ba con số chứ không một, vì một tỉ lệ duy nhất che mất đánh đổi trung tâm: hệ
thống từ chối **mọi** câu đạt recall 100% và vô dụng; hệ thống không bao giờ từ
chối đạt 0% từ chối oan và bịa cho 33 câu không có đáp án.

Nhóm yếu nhất là `adversarial` (0,8529) và `cross_lingual` (0,8605).

### Citation accuracy — bảng mục tiêu viết một dòng cho hai câu hỏi

| cấp | giá trị | câu hỏi nó trả lời |
|---|---|---|
| **quote** (`W4-09`, tất định) | **0,8308** (329/396) | quote có nằm **nguyên văn** trong đúng chunk nó chỉ vào không |
| **mệnh đề** (judge) | **0,9877** (402/407) | chunk được trích có **chống đỡ** ý của câu không |

Gộp hai thành một số trung bình là tạo ra một đại lượng không ai kiểm lại được.
Ngưỡng ≥ 0,85 áp cho **cấp quote** — đó là câu hỏi mà một người dùng bấm vào
citation đang hỏi — và nó **KHÔNG ĐẠT**, thiếu 0,019.

**67 quote hỏng, phân loại được:**

| nguyên nhân | số | ghi chú |
|---|---|---|
| không khớp nguyên văn (diễn đạt lại) | 38 | model chép không đúng chữ |
| có dấu lược `...` nối hai mảnh | 19 | không phải nguyên văn theo định nghĩa |
| **quote đúng nhưng gắn nhầm số nguồn** | 10 | đúng chế độ hỏng `W4-09` sinh ra để bắt |

⚠️ Đã kiểm giả thuyết "phép đo sai chứ không phải model sai": chuẩn hoá NFKC +
gấp nháy cong/gạch dài **cứu được 0/67**. Con số 0,8308 là thật, không phải hiện
vật của phép so chuỗi.

---

## 7. Answer relevancy 0,7479 — và vì sao con số ấy đọc sai nếu đứng một mình

`REFUSAL` không vào tử số nhưng ở lại mẫu số. Với câu trả lời được, từ chối là
một lần trượt. Nhưng nhóm `unanswerable` (33 câu) **phải** từ chối, nên nhóm ấy
kéo điểm xuống 0,0606 một cách hoàn toàn đúng đắn.

Bỏ nhóm `unanswerable`: **179/209 = 0,8565**.

Breakdown: `table_lookup` 1,0 (n=4) · `factoid` 0,9412 · `cross_lingual` 0,8372 ·
`multi_hop` 0,8235 · `adversarial` 0,7941 · `aggregation` 0,7692.

---

## 8. Tái lập: toàn bộ eval chạy lại ở **$0**

```
uv run python -m pipeline.eval.generation_metrics \
    --run plans/reports/runs/w5-answers-v1.jsonl \
    --cache .cache/judge-w5.sqlite3 --frozen-cache
```

`--frozen-cache` biến mọi lượt trượt cache thành lỗi. Lần chạy kiểm chứng:
**947 hits / 0 misses / $0,000000**, cùng digest cache
`79d7df51897a1448…`, cùng mọi con số. Đó là cách trả lời "tái lập lại số của bạn
đi" bằng một lệnh chứ không bằng một lời hứa.

⚠️ Giới hạn: cache nằm ở `.cache/judge-w5.sqlite3`, không version cùng report
(`TD-61`). Mất file ấy thì `--frozen-cache` vô dụng và phải trả tiền lại.

---

## 9. Tiêm lỗi — 11/11 đỏ, và một phép tiêm nữa lộ ra mã chết

Mỗi phép tiêm đảo một **quy ước đo**. Chúng không làm chương trình nổ; chúng làm
nó trả một con số khác mà vẫn trông hợp lệ.

| | kết quả |
|---|---|
| G2 bỏ chặn tách sau từ viết tắt | ĐỎ |
| G3 `strip_markers` không bỏ gì | ĐỎ |
| G4 chấm với hợp mọi chunk thay vì nguồn đã trích | ĐỎ |
| G5 câu không trích nguồn tính là thất bại | ĐỎ |
| G6 `NO_CLAIM` tính là không được chống đỡ | ĐỎ |
| G7 phán quyết không đọc được tính là thất bại | ĐỎ |
| G8 gán nhầm hỏi lại **cả** mệnh đề đã được chống đỡ | ĐỎ |
| G9 `Aggregate` rỗng trả `0.0` thay vì `None` | ĐỎ |
| G10 từ chối oan tính là đúng | ĐỎ |
| G11 coverage tính cả nhánh không truy hồi | ĐỎ |
| G12 truy vấn `unanswerable` tính recall = 0 | ĐỎ |
| **G1** bỏ chặn tách giữa một con số | **XANH** → mã chết, đã gỡ |

**G1 sống sót vì phép kiểm nó gỡ không thể chạy.** `_SENTENCE_END` đòi dấu chấm
theo sau bởi khoảng trắng; phép kiểm `\d[.,]\d` đòi nó theo sau bởi chữ số. Hai
điều kiện loại trừ nhau. Cái thật sự bảo vệ `1.234.567` là `\s+` trong regex.

Cùng họ với `J2` của `W5-03`, và xử lý như thế: **gỡ**, không giữ. Một lớp bảo vệ
không có tác dụng còn tệ hơn không có. Test ở lại và giờ mới thật sự kiểm cơ chế
đang chạy. Sau khi gỡ, G1 không còn đối tượng để tiêm.

---

## 10. Chi phí

| khoản | USD |
|---|---|
| Lần chạy 242 câu qua container | 0,2588 |
| Judge vòng 1 (rubric v1, faithfulness + relevancy) | 0,2049 |
| Judge vòng 2 (thêm `uncited_grounding`) | 0,1721 |
| Judge vòng 3 (rubric v2 — cache tự vô hiệu, chấm lại) | 0,3311 |
| Vòng 4 `--frozen-cache` | **0,0000** |
| **Tổng** | **≈ 0,967** |

Vòng 3 là **giá của việc sửa một rubric sai**, và nó là giá đúng: thay thế cho
việc công bố `uncited_grounding = 0,427`.

---

## 11. Nợ mở

* **`TD-62`** — phép kiểm danh tính bundle không soi phiên bản thư viện. `W5-05`
  hoặc `W6-06`: đưa `torch`/`transformers`/`sentence-transformers` vào vân tay
  runtime, hoặc ít nhất vào `/admin/bundle` để `runtime_drift` nói được sự thật.
* **`TD-63`** — khoá tuần tự hoá làm thông lượng một worker bị chặn bởi GPU.
  `W6-05` (load test) phải đo trần thông lượng thật, không suy từ p95 một luồng.
* **`TD-64`** — `citation_accuracy` cấp quote **KHÔNG ĐẠT** (0,8308 vs 0,85). 19/67
  lỗi là dấu lược `...`; sửa luật 5 của `chat-system` để cấm lược là một thay đổi
  prompt **có thể đo được bằng chính harness này**. Thuộc `W5-05`/`W6-06`.
* `TD-61` (cache judge không version cùng report) — chạm trực tiếp §8.
