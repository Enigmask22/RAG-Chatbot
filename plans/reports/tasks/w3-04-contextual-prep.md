# `W3-04` + `W0-08` — Contextual Retrieval: phần chạy được ở laptop, và hai cái bẫy mà dry-run bắt được

> **Trạng thái:** chuẩn bị **xong**, lượt chạy GPU **chưa** (thuộc phần thủ công của bạn)
> **Ngày:** 2026-09-03 · **Nhánh:** `main`
> **Code:** `packages/rag_core/chunking/contextual.py`, `packages/rag_core/llm/{budget,tokenizer}.py`,
> `pipeline/indexing/{contextualize,job_bundle}.py`, `scripts/runpod_job.sh`
> **Test:** `tests/unit/test_contextual_chunking.py` (37) · `tests/security/test_no_secret_in_job_bundle.py` (26)
> **Probe:** `plans/reports/probes/w3-04-dryrun-deepseek.json`
> **Lệnh:** `make ctx-dry` · `make ctx-prepare` · `make ctx-run-api` · `make job-bundle` · `make job-verify`

## 0. Hạng mục này KHÔNG phải là "W3-04 xong"

Nó là mọi thứ có thể làm — và mọi thứ có thể **hỏng** — trước khi thuê GPU. Lý do
chia như vậy: pod tính tiền theo giờ, nên mỗi lỗi phát hiện trên pod đắt hơn cùng
lỗi ấy phát hiện ở laptop đúng bằng tiền thuê. Báo cáo này ghi lại hai lỗi mà
**không suy luận thiết kế nào bắt được** và chỉ lộ ra khi nhìn output thật.

Còn lại cho lượt chạy trên pod: sinh ngữ cảnh cho 15.814 chunk, build index, eval,
`W3-09`.

## 1. Dự đoán ghi trước khi viết code

Bốn dự đoán dưới đây được phát biểu trong lúc bàn kế hoạch, **trước** khi có dòng
code nào — nên chúng kiểm được.

| # | Dự đoán | Kết quả |
|---|---|---|
| E1 | Không nhét cả tài liệu vào prompt được; cửa sổ Qwen3-8B là 32K | ✅ đúng về kết luận, ❌ **sai con số**: `config.json` ghi `max_position_embeddings = 40960`. **28/60 tài liệu vượt** (§2) |
| E2 | Cửa sổ theo section bị `TD-24` chặn | ✅ đúng — 0/60 tài liệu có heading máy đọc được |
| E3 | Prefix caching bỏ được ~2,3× prefill (63M → 27,7M) | 🟡 **quá lạc quan**: đo được **1,95×** (64,3M → 33,0M), tức 49,0% (§3) |
| E4 | Chạy hết corpus trên DeepSeek tốn ~$17–18 | 🟡 lệch 50% — đo được **$11,85**. Trớ trêu: con số tôi đoán lại gần đúng với đường **đang hỏng** ($20,9 khi chưa tắt suy luận) |
| E5 | *(không có)* Rủi ro lớn nhất là kích thước cửa sổ | ❌ **sai hoàn toàn.** Hai lỗi thật sự nguy hiểm là **suy luận nuốt output** (§4) và **ngữ cảnh sai ngôn ngữ** (§5). Cả hai vô hình với suy luận thiết kế |

**Bài học của cả hạng mục nằm ở `E5`.** Tôi dành công sức thiết kế cho chiều duy
nhất mình *tính toán* được, và cả hai lỗi thật đều nằm ở chiều chỉ **nhìn output**
mới thấy. Không có dry-run 30 request giá $0,03, cả hai đi thẳng lên pod.

## 2. Vì sao công thức gốc không áp thẳng được

Contextual Retrieval của Anthropic đặt **nguyên tài liệu** vào prompt. Đo corpus
bằng đúng tokenizer `Qwen/Qwen3-8B`:

| | token |
|---|---:|
| tài liệu p50 | **32.176** |
| tài liệu p90 | **138.423** |
| tài liệu lớn nhất | **188.987** |
| tổng corpus | 2.947.725 |
| cửa sổ Qwen3-8B | **40.960** |

**28/60 (47%) không lọt vào cửa sổ**, lớn nhất vượt 4,6×. Và kể cả nếu lọt: 15.814
chunk × 32K = ~500 tỷ token prefill, không phải vấn đề tiền mà là vấn đề tuần lễ.

Lựa chọn thay thế hiển nhiên — cửa sổ theo **section**, dùng `section_path` của
`W3-03` — bị `TD-24` chặn: corpus là bản `.txt` World Bank trích sẵn, `make
structure-corpus` đo **0/60** tài liệu có heading máy đọc được.

Nên cửa sổ ghép hai nguồn: `<document_head>` (2.000 token đầu — trả lời *tài liệu
nào, của ai, năm nào*) + `<neighbourhood>` (1.500 token quanh chunk, lấy qua
`start_char`/`end_char` — trả lời *mục này đang bàn gì*). Prompt dài nhất đo được:
**5.248 token**, dư 7,8× so với cửa sổ.

Ký tự/token đo bằng tokenizer Qwen3: **EN 5,10 · VI 4,37**. Chênh 17% nên cắt cửa
sổ bằng một hằng số dùng chung sẽ làm prompt tiếng Việt lệch đúng chừng ấy —
`calibrate_density` gọi một lần cho mỗi tài liệu.

## 3. Thứ tự trong prompt là quyết định hiệu năng, và nó đo được

vLLM (và cả prompt cache của DeepSeek) cache theo **tiền tố token**. Nên chỉ dẫn
nằm ở `system` (bất biến trên cả job), rồi `document_head` (bất biến trong một tài
liệu), rồi mới tới phần đổi theo chunk, và câu lệnh cuối đặt **sau** `<chunk>`.

Đo trên gói thật, 15.814 request / 60 tài liệu:

| | token |
|---|---:|
| prefill nếu không cache | 64.369.811 |
| tiền tố dùng chung | **31.338.717 (48,7%)** |
| còn lại nếu cache trúng | 33.031.094 |

Có một test riêng ghim bất biến này (`test_document_head_is_a_shared_prefix_of_
every_user_message`), vì khi nó vỡ thì **không có gì đỏ** — job chỉ chạy lâu gấp
đôi. Tiêm lỗi đảo thứ tự khối prompt: đúng test đó đỏ, 31 test còn lại xanh.

⭐ **Và nó đo được một lần nữa, tình cờ.** Mẫu 20 request lấy **ngẫu nhiên** khắp
corpus cho cache trúng **10,5%**; 40 request liên tiếp cùng một tài liệu cho
**49,1%**. Prefix cache chỉ trúng khi tiền tố *vừa mới* đi qua — xáo trộn để "chia
tải" là vứt đi một nửa của 31,3 triệu token. `prepare` sinh đúng thứ tự sẵn, và
`run_requests` cảnh báo nếu đầu vào không gom theo tài liệu.

## 4. Bẫy thứ nhất: suy luận nuốt hết output

Dry-run đầu tiên, 30 request, `max_tokens=512`:

```
  sinh được      24
  lỗi 0 · rỗng 6 · cắt lời 7
  token          prompt 123.846 · out 10.885
                 suy luận 9.003 (KHÔNG có trong content)
  cost/1000      $1,3226
```

**83% completion token là chuỗi suy luận không xuất hiện trong `content`**, và 6/30
request trả rỗng vì suy luận ăn hết ngân sách trước khi kịp viết câu nào. Đây
không phải chuyện riêng của DeepSeek: **Qwen3-8B cũng là model lai suy luận**, nên
lỗi này đã lên tới pod nguyên vẹn.

Cách tắt là tham số riêng của từng nhà. Đo bốn cách, cùng prompt, `max_tokens=512`:

| cách | suy luận | completion | content |
|---|---:|---:|---:|
| không tắt | 275 | 328 | 255 ký tự |
| `thinking={"type":"disabled"}` | **0** | 138 | 361 |
| `reasoning_effort="none"` | **0** | 87 | 374 |
| `chat_template_kwargs={"enable_thinking":false}` | 159 | 219 | 288 |

⚠️ Dòng cuối là loại bẫy tệ nhất: tham số được **nhận, không lỗi, và không có tác
dụng**. Không đo thì tưởng đã tắt. (Với vLLM thì đúng nó mới là cơ chế — chat
template của Qwen3 đọc `enable_thinking`; DeepSeek chỉ đơn giản là không đọc.)

`LLMProvider.complete` nhận thêm `extra_body`; `NO_THINKING` khai theo từng
backend. Sau khi tắt:

```
  sinh được      40 / 40
  lỗi 0 · rỗng 0 · cắt lời 0
  token          out 3.199 (giảm 3,8×)
  cost/1000      $0,7492  (giảm 43%)
```

Và nội dung **dài hơn**, không ngắn đi.

## 5. Bẫy thứ hai: ngữ cảnh sinh ra bằng tiếng Pháp và tiếng Trung

Prompt bản đầu viết *"Write in the SAME language as `<chunk>`"*. Trên 30 chunk của
một tài liệu **tiếng Anh**:

| ngôn ngữ ngữ cảnh sinh ra | số |
|---|---:|
| tiếng Pháp | **15** |
| tiếng Trung | **10** |
| tiếng Anh | 5 |

**17% đúng.** Không lỗi, không cảnh báo, không có gì trong log. Nếu không mở file
ra đọc, index sẽ nhận 15.814 chunk mang một câu tiếng Pháp dán lên đầu, và mọi
metric của `W3-09` chỉ đơn giản là tệ hơn — với một nguyên nhân không ai truy ra.

Cách chữa không phải là prompt engineering: **ngôn ngữ nằm sẵn trong
`DocumentMetadata.lang`**, tới thẳng từ manifest. Bắt model suy ra thứ mình đã
biết là tự thêm một chỗ hỏng. Câu lệnh cuối giờ gọi tên nó (`"…in Vietnamese."`)
và đặt **sau** `<chunk>` — ngoài tiền tố dùng chung nên miễn phí về prefill, lại
là thứ model đọc gần nhất trước khi trả lời.

Sau khi sửa: **40/40 tiếng Anh · 20/20 tiếng Việt.** Mẫu:

> Đoạn trích nằm trong Chương 2 của báo cáo "Điểm lại Tháng 9/2025: Cập nhật tình
> hình kinh tế Việt Nam" do Ngân hàng Thế giới xuất bản, thuộc phần chuyên đề về
> phát triển nhân tài công nghệ cao.

Đó đúng là những từ mà câu hỏi sẽ dùng và chunk gốc không có.

## 6. Chi phí và thời gian dự phóng

**Đường API (đo thật, 60 request):** $0,7492/1000 chunk → **$11,85** cho cả corpus.
Cache trúng 49,1%.

**Đường GPU (tính, chưa đo):** 33,0M token prefill. Qwen3-8B bf16 ≈ 16,4 GB trọng
số; KV cache = 2 × 36 lớp × 8 KV head × 128 head_dim × 2 byte = **0,1406 MB/token**.
Trên 24 GB còn ~6,6 GB cho KV ≈ **47.000 token**, tức ~11 chuỗi 4.100-token cùng
lúc. Prefill ≈ 2 × 8,2e9 × 33,0e6 = 5,4e17 FLOP; ở ~40% MFU của 165 TFLOP/s bf16
thì ≈ **2,3 giờ**, cộng decode ~20 phút → **~2,5–3,5 giờ**.

⚠️ Cả đoạn trên là **phép tính, không phải phép đo**. MFU 40% là giả định; con số
thật chỉ có sau lượt chạy đầu. Nếu chật VRAM thì lượng FP8 hạ trọng số còn ~8,2 GB
và gấp đôi được batch — 4090 (sm89) hỗ trợ FP8 nguyên bản.

## 7. `W0-08` — gói job, và ba lớp chặn

DoD: *quét tarball, fail nếu thấy pattern API key.* ⚠️ Một bộ quét luôn trả "sạch"
cũng qua được DoD viết như vậy, nên nhóm test đầu tiên là **positive control**:
mỗi mẫu phải có một bí mật giả làm nó đỏ, và có test đòi mọi mẫu đều có ca thử.

Ba lớp, xếp theo độ mạnh giảm dần:

1. **`git archive HEAD`** — không thể chứa file chưa commit hoặc bị `.gitignore`.
   Đó là tính chất của công cụ, không phải của một danh sách loại trừ tôi nhớ ra.
   `.env`, `.venv/`, `.cache/`, `data/` đều rơi vào nhóm ấy.
2. **Danh sách tên cấm** — bắt file *đã commit nhầm từ trước* (`git archive` sẽ vui
   vẻ đóng gói một `.env` đã lỡ commit).
3. **Quét nội dung** — kể cả **bên trong** `requests.jsonl.gz`, vì đó là file duy
   nhất trong gói sinh ra từ dữ liệu cục bộ.

⚠️ **Bẫy đối xứng của lớp 1, và nó bật ra ngay lần dựng đầu:** gói thiếu đúng
`pipeline/indexing/contextualize.py` — chính cái job — vì code vừa viết xong chưa
commit. Triệu chứng sẽ là `ModuleNotFoundError` trên một cái pod đang tính tiền.
Nên có `REQUIRED_MEMBERS` + `BundleIncomplete`: gói thiếu file thì **không dựng
được**, và lỗi xảy ra ở laptop.

⭐ **Lần quét đầu trên gói thật báo 6 phát hiện — cả sáu là bí mật giả trong chính
file test của bộ quét.** True positive, và là bằng chứng đường quét chạy thật trên
gói thật chứ không chỉ trên đầu vào tổng hợp. Cách chữa **không** phải là miễn trừ
file đó (một cơ chế miễn trừ là thứ về sau sẽ che mất bí mật thật) mà là bỏ hẳn
`tests/` khỏi gói: pod chạy đúng một job và không chạy test bao giờ.

Kết quả gói thật: **7,6 MB · 122 file · 320,4 MB văn bản đã giải nén và quét ·
SẠCH**. Gói request nén 285 MB → **8,47 MB** (33×, vì `document_head` lặp lại ở mọi
dòng cùng tài liệu).

Và một test đọc `run_on_pod.sh` để ghim rằng nó không nhắc tới `DEEPSEEK_API_KEY`,
`OPENROUTER_API_KEY`, `HF_TOKEN`, hay `--backend deepseek` — quy tắc cứng #2 ở dạng
kiểm được, không phải ở dạng lời hứa.

## 8. Chưa xác minh

* **Chưa chạy vLLM lần nào.** Đường `--backend vllm` dùng chung client với đường
  DeepSeek đã chạy thật, nên nó không phải mã chưa từng chạy — nhưng
  `chat_template_kwargs` với Qwen3, `--enable-prefix-caching`, và toàn bộ ngân sách
  VRAM thì chưa có số đo nào.
* **Chưa build index nào bằng chunk có ngữ cảnh**, nên chưa biết nó cải thiện truy
  hồi hay không. Đó là `W3-09`.
* **Chưa đo `apply_contexts` trên corpus thật** — chỉ trên 60 chunk dry-run.
* Chất lượng ngữ cảnh mới được đọc bằng mắt trên ~8 mẫu. Không có phép đo tự động
  nào cho "câu này có định vị đúng không"; `W3-09` sẽ trả lời gián tiếp qua metric.

## 9. Chạy trên pod

```bash
# --- laptop ---
make ctx-prepare          # 15.814 request → data/contexts/requests.jsonl.gz (8,5 MB)
make job-bundle           # → dist/runpod-job.tar.gz (7,6 MB)
make job-verify           # PHẢI sạch trước khi đẩy
bash scripts/runpod_job.sh push

# --- pod: Secure Cloud · RTX 4090 24GB · network volume /workspace ---
#     HF_HOME=/workspace/.hf  · đặt spending limit TRƯỚC khi tạo pod
tar xzf runpod-job.tar.gz
pip install vllm httpx pydantic pydantic-settings pyyaml
GPU_HOURLY_USD=<giá thật> bash job/run_on_pod.sh

# --- laptop ---
runpodctl receive <mã>    # → data/contexts/contexts.jsonl
# rồi TERMINATE pod, thu hồi token
```

`run_on_pod.sh` đặt `--max-model-len 8192` (prompt dài nhất 5.248) thay vì 40.960:
khai thừa là bắt vLLM giữ chỗ KV cache cho độ dài không bao giờ dùng tới, mà trên
24 GB thì đúng chỗ ấy quyết định batch lớn hay nhỏ.
