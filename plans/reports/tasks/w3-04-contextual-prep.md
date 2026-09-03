# `W3-04` + `W0-08` — Contextual Retrieval: phần chạy được ở laptop, và hai cái bẫy mà dry-run bắt được

> **Trạng thái:** chuẩn bị **xong**, lượt chạy GPU **chưa** (thuộc phần thủ công của bạn)
> **Ngày:** 2026-09-03 · **Nhánh:** `main`
> **Code:** `packages/rag_core/chunking/contextual.py`, `packages/rag_core/llm/{budget,tokenizer}.py`,
> `pipeline/indexing/{contextualize,job_bundle}.py`, `scripts/runpod_job.sh`
> **Test:** `tests/unit/test_contextual_chunking.py` (37) · `tests/unit/test_contextualize_backends.py` (16)
> · `tests/security/test_no_secret_in_job_bundle.py` (26)
> **Probe:** `plans/reports/probes/w3-04-dryrun-deepseek.json`
> **Lệnh:** `make ctx-dry` · `make ctx-prepare` · `make ctx-run-glm` · `make job-bundle` · `make job-verify`
> **Runbook:** `RUNPOD.md`

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
| E4 | Chạy hết corpus trên DeepSeek tốn ~$17–18 | ❌ **lệch 2,8×** — đo được **$6,2** (cache ấm). Con số $11,85 tôi công bố ở bản đầu báo cáo này cũng sai, vì nó đo vùng khởi động cache (§6) |
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

## 6. Ba backend, và cái bẫy đo lường trong chính phép so sánh

Sau khi `W3-04` chạy được, ngân sách API trở thành ràng buộc thật, nên thêm
**GLM-5.3-Flash (Z.ai)**. Giá niêm yết rẻ hơn DeepSeek 1,8–2,3× ở cả ba trục
($0,15 / $0,50 / $0,03 so với $0,27 / $1,10 / $0,07).

GLM nói đúng giao thức OpenAI **kể cả phần `usage`** — nó trả
`prompt_tokens_details.cached_tokens` và `completion_tokens_details.reasoning_tokens`,
đúng hai field `_parse` đã đọc sẵn từ `W1-10`. Nên không có đường parse riêng;
chỉ thêm bảng giá và một `base_url`. (DeepSeek thì dùng tên khác —
`prompt_cache_hit_tokens` — nên đây là may, không phải mặc định.)

### GLM cũng là model suy luận, và nó KHÔNG tắt được

Cùng bẫy §4, nhưng lối thoát khác. API trả thẳng HTTP 400 mã `1210`:

> *This model always engages in thinking and cannot be disabled; please use low,
> high, or max*

Ba mức hợp lệ, đo trên cùng một prompt:

| `reasoning_effort` | suy luận | completion | content |
|---|---:|---:|---:|
| *(không đặt)* | 165 | 243 | 401 ký tự |
| `low` | **0** | **70** | 383 |
| `high` | 40 | 118 | 393 |
| `max` | 180 | 255 | 411 |

`low` cho `reasoning_content` **rỗng thật** (0 ký tự — đã đọc response thô để
kiểm, không tin mỗi con số `reasoning_tokens`), trong khi độ dài content gần như
không đổi. 3,5× output tiết kiệm được là tiết kiệm sạch.

⚠️ Và `chat_template_kwargs` — thứ **có** tác dụng với vLLM/Qwen3 — bị GLM từ chối
thẳng, trong khi DeepSeek nhận nó rồi **bỏ qua**. Ba nhà, ba hành vi khác nhau cho
cùng một ý định. Nên hằng số được đổi tên từ `NO_THINKING` thành **`MIN_REASONING`**:
gọi là "tắt" khi có nhà không tắt được là gieo một hiểu nhầm vào mọi báo cáo chi
phí về sau.

### ⚠️ ĐÃ BỊ THAY THẾ — đọc §6bis trước

> Toàn bộ tiểu mục này và bảng quyết định của nó **sai**. Phép "chạy lại đúng
> 40 request đó" không đo cache tiền tố mà đo cache khớp-nguyên-request, thứ
> không bao giờ xảy ra khi chạy thật. Giữ lại nguyên văn để đối chiếu; kết luận
> đúng ở **§6bis**.

### ~~Phép so sánh đầu tiên sai, vì nó đo vùng khởi động cache~~

Lượt đo đầu: GLM **$0,5954**/1000 vs DeepSeek **$0,7492**/1000 — chỉ rẻ 20%, mâu
thuẫn với bảng giá. Nguyên nhân nằm ngay trong báo cáo: **cache trúng 4,0% (GLM)
vs 49,1% (DeepSeek)**.

Chạy **lại đúng 40 request đó** lần thứ hai:

| | lượt 1 (cache nguội) | lượt 2 (cache ấm) |
|---|---:|---:|
| GLM · cache trúng | 4,0% | **94,8%** |
| GLM · cost/1000 | $0,5954 | **$0,1818** |
| DeepSeek · cache trúng | 49,1% | **98,3%** |
| DeepSeek · cost/1000 | $0,7492 | **$0,3949** |

Cả hai đều rẻ đi, GLM rẻ đi **3,3×**. Kết luận đúng: **GLM rẻ hơn 2,17×**, không
phải 20%.

Chỗ sai của phép đo đầu: 40 request ấy đều lấy từ **đầu một tài liệu**, tức toàn
bộ mẫu nằm trong vùng khởi động cache. Trên corpus thật mỗi tài liệu có ~264
chunk, nên vùng khởi động chiếm ~2% chứ không phải 100%. **Mẫu 40 request liên
tiếp đo đúng cái mà nó tình cờ phủ, và cái đó không phải chế độ vận hành.** Cùng
họ với ba lần fixture đồng nhất của `W3-01`/`W3-05`/`W3-07`, chỉ khác là lần này
sai lệch đến từ *vị trí* mẫu chứ không từ nội dung mẫu.

### Bảng quyết định

| đường | cost/1000 (ấm) | cả corpus | throughput (conc=6) | cả corpus |
|---|---:|---:|---:|---:|
| **GLM-5.3-Flash** | **$0,1818** | **~$2,9** | 1,54 req/s | ~2,9 h |
| DeepSeek-v4-flash | $0,3949 | ~$6,2 | 4,40 req/s | ~1,0 h |
| vLLM Qwen3-8B / 4090 | theo giờ | tiền thuê × 2,5–3,5 h *(tính)* | — | 2,5–3,5 h *(tính)* |

GLM rẻ hơn 2,2× nhưng chậm hơn 2,9× mỗi request; nâng `--concurrency` bù được
phần thời gian mà không đổi chi phí.

⭐ **Hệ quả cho kế hoạch:** ~$2,9 làm nhánh API thành lựa chọn chính chứ không còn
là fallback. Pod vẫn cần cho `W5-11` (ablation generator bắt buộc có vLLM), nhưng
`W3-04` **không còn bị GPU chặn**.

Chất lượng ngang nhau: GLM cho **40/40 EN · 20/20 VI** đúng ngôn ngữ, độ dài p50
447 ký tự (DeepSeek 433), nội dung nêu đúng tên báo cáo, tổ chức, năm, và mục.

## 6bis. ⭐⭐ GLM không có prefix caching, và cách tôi phát hiện ra

Chạy thật 860 request ngày 03/09/2026. Kết quả đảo ngược §6.

### Phép đo quyết định

Prompt của job chia sẻ tiền tố rất lớn — đo trực tiếp trên file request, không
đoán: giữa hai request liền kề cùng tài liệu, tiền tố chung **p50 = 9.006 ký tự
≈ 54,8%** mỗi prompt. Đó là trần lý thuyết của cache.

| phép đo | request | concurrency | cache trúng | cost/1000 |
|---|---:|---:|---:|---:|
| chạy thật, 5 tài liệu | 800 | 16 | **2,1%** | **$0,6196** |
| chạy thật, tuần tự | 60 | **1** | **0,1%** | $0,6777 |
| *trần lý thuyết* | — | — | *~55%* | — |

`--concurrency 1` là phép thử tách bạch: request thứ k+1 chỉ gửi **sau khi** k
trả về, nên nếu cache tiền tố tồn tại thì tiền tố chắc chắn đã nằm sẵn trong đó.
Vẫn 0,1%. Không phải chuyện đua tranh giữa các request song song — **GLM chỉ
cache khi request trùng gần như toàn bộ.**

Điều đó giải thích cả hai con số của §6: 94,8% là vì tôi gửi lại **request giống
hệt**; 4,0% là vì các request khác nhau. Cả hai đều đúng, và cả hai đều không
phải chế độ vận hành.

Ngược lại DeepSeek đo được **49,1%** trên tập request *khác nhau* — sát trần 55%.
Nên DeepSeek **có** prefix caching thật, và ở §6 tôi đã đọc nhầm chính con số đó
thành "cache còn nguội". Bằng chứng bác bỏ nằm trong bảng tôi tự in ra, cách chỗ
tôi viết kết luận sai đúng ba dòng.

⭐ **Bài học chung với `W3-01`/`W3-05`/`W3-07` nhưng ở dạng mới:** một phép đo lặp
lại chính nó đo **khả năng ghi nhớ**, không đo khả năng làm việc. Muốn đo cache
tiền tố thì đầu vào phải *khác nhau ở phần đuôi và giống nhau ở phần đầu* — đúng
hình dạng của workload thật, chứ không phải bản sao của một mẫu.

### Bảng quyết định (thay cho bảng ở §6)

| đường | cache tiền tố | cost/1000 | cả corpus 15.814 |
|---|---|---:|---:|
| GLM-5.3-Flash | ❌ không | $0,6196 *(đo)* | **~$10,6** |
| DeepSeek-v4-flash | ✅ 49,1% | $0,78 *(chiếu)* | ~$12,4 |
| vLLM Qwen3-8B / 4090 | ✅ tự động | theo giờ | tiền thuê × 2,5–3,5 h *(tính)* |

⭐ **Hệ quả cho kế hoạch — ngược hẳn §6:** đường API **không** rẻ. Với workload
chia sẻ 48,7% tiền tố, pod chạy vLLM là đường duy nhất thu được khoản đó, vì
automatic prefix caching của vLLM là prefix trie thật. `W3-04` **bị ngân sách
chặn lại**, không còn "không bị GPU chặn" như §6 kết luận.

### ⭐⭐ Cắt prompt cho rẻ: rẻ đúng 2× và hỏng 95%

Không có cache thì `<document_head>` (49% mỗi prompt) bị trả tiền lại từ đầu cho
từng chunk trong ~185 chunk của tài liệu. Nên thử `--head-tokens 500
--window-tokens 800`: prompt nhỏ **2,06×**, cost/1000 xuống **$0,3113**, cả
corpus còn ~$5,2.

40 request đầu trông hoàn hảo — cùng độ dài, cùng nội dung, 40/40 nêu đúng tổ
chức và năm. Nhưng 40 request đó **đều là chunk đầu tài liệu**. Chạy hết 385
request để có cả chunk sâu thì:

| | ngữ cảnh nêu **đúng tên tài liệu** |
|---|---:|
| head 2000 (mặc định) | 256/260 — sai **1,5%** |
| head 500 | 12/260 — sai **95%** |

Bản `slim` bịa tên báo cáo là «Việt Nam 2035: Từ chiến lược đến hành động» trong
khi tài liệu thật là «Thị trường lao động và sự bùng phát đại dịch COVID-19 ở
Việt Nam». Bìa là OCR lẫn banner chương trình đứng **trước** tên báo cáo thật;
head 500 thấy cả hai và chọn nhầm cái nổi bật hơn, head 2000 có thêm tiêu đề
chạy và mục lục phía sau để phân định.

⚠️⚠️ **Hai chỉ số bề mặt của tôi đạt 100% trên chính những ngữ cảnh bịa đặt.**
"Có nêu tên tổ chức" 354/354, "có nêu năm" 354/354 — vì ngữ cảnh bịa vẫn nói
"Ngân hàng Thế giới" và vẫn có một năm trong đó. Proxy đo cái **dễ đo** chứ không
đo cái **cần đo**; phải đối chiếu tên tài liệu với sự thật mới thấy. Cùng khuôn
với `KHÔNG SO ĐƯỢC` đi chung đường với "hoà" ở `W2-08`: một nhãn xanh trên một
thứ chưa từng được kiểm.

⭐ Và bẫy vị trí mẫu **lặp lại lần thứ hai trong cùng một buổi** — 40 chunk đầu
cho kết luận ngược với 385 chunk đủ. Lần này tôi đã ngờ và chạy tiếp; đó là khác
biệt duy nhất giữa nó và §6.

### Chất lượng lượt chạy thật (860 chunk, head 2000)

0 lỗi · 0 rỗng · 0 cắt lời · độ dài p50 389 ký tự · đúng ngôn ngữ tài liệu.

⚠️ Hai khiếm khuyết đo được, cả hai chưa chữa:
- **0,35% rò token tiếng Trung** giữa câu tiếng Việt (`討论`, `免责`) — GLM là
  model Trung Quốc. Ở quy mô corpus là ~55 chunk. Phát hiện bằng regex CJK nên
  chữa rẻ → `TD-33`.
- **1,5% nêu sai tên tài liệu** ngay cả với head 2000 → `TD-34`.

### Đường chưa làm: gộp nhiều chunk mỗi request

Gửi 8 chunk liền kề trong một request thì `<document_head>` chia cho 8, và vùng
lân cận của chúng chồng lấn nên gộp lại gần như một dải liền. Ước tính đưa corpus
về **~$2–2,5**. Đúng về kinh tế, nhưng đổi định dạng artifact, cần bóc tách 8 ngữ
cảnh trả về, cần chốt chặn chống lệch thứ tự, và một response hỏng làm mất 8
chunk thay vì 1 → `TD-32`.

### Đã chi

$0,6617 cho 860 ngữ cảnh dùng được + 385 ngữ cảnh `slim` để bác bỏ đường cắt
prompt.

---

### ⚠️ Một mặc định duy nhất cho ba backend là một cái bẫy

`--backend glm` mà quên `--model` gửi slug mặc định `Qwen/Qwen3-8B` sang Z.ai và
nhận **20/20 `HTTP 400 modelCode: does not exist`**. Ở laptop thì vô hại — xử lý
lỗi chạy đúng thiết kế: 20 lỗi ghi sang `.failures.jsonl`, artifact chính không
bẩn, chạy lại là thử lại. Trên pod thì đó là mấy phút tiền thuê đổi lấy một file
lỗi. Sửa bằng `DEFAULT_MODEL` theo backend + test đòi **mọi** bảng-theo-backend
phải phủ đủ danh sách backend.

### Đường GPU (tính, chưa đo)

33,0M token prefill. Qwen3-8B bf16 ≈ 16,4 GB trọng số; KV cache = 2 × 36 lớp × 8
KV head × 128 head_dim × 2 byte = **0,1406 MB/token**. Trên 24 GB còn ~6,6 GB cho
KV ≈ **47.000 token**, tức ~11 chuỗi 4.100-token cùng lúc. Prefill ≈ 2 × 8,2e9 ×
33,0e6 = 5,4e17 FLOP; ở ~40% MFU của 165 TFLOP/s bf16 thì ≈ **2,3 giờ**, cộng
decode ~20 phút → **~2,5–3,5 giờ**.

⚠️ Cả đoạn trên là **phép tính, không phải phép đo**. MFU 40% là giả định. Nếu
chật VRAM thì lượng hoá FP8 hạ trọng số còn ~8,2 GB — 4090 (sm89) hỗ trợ nguyên
bản.

### Số cũ (giữ lại để đối chiếu)

**Đường DeepSeek đo lần đầu, cache nguội:** $0,7492/1000 chunk → $11,85 cho cả
corpus. Cache trúng 49,1%.

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

Runbook đầy đủ — kể cả cách tạo Network Volume, hai cách chuyển file, và bảng chế
độ hỏng — nằm ở **`RUNPOD.md`** ở gốc repo.

Tóm tắt:

```bash
# laptop
make ctx-prepare && git status --short   # phải sạch: git archive chỉ đóng gói thứ đã commit
make job-bundle && make job-verify       # phải in SẠCH
bash scripts/runpod_job.sh push

# pod: Secure Cloud · 4090 24GB · volume /workspace · HF_HOME=/workspace/.hf
tar xzf runpod-job.tar.gz
pip install -q vllm httpx pydantic pydantic-settings
GPU_HOURLY_USD=<giá thật> bash job/run_on_pod.sh
```

⚠️ **Kiểm ở phút thứ ba**: `reasoning_tokens` phải bằng **0**, `cache_hit_rate`
phải trên **30%**, và ngôn ngữ ngữ cảnh phải khớp tài liệu. Cả ba đều **không làm
job đỏ** — job vẫn chạy tới cùng và trả ra một file trông hợp lệ.

`run_on_pod.sh` đặt `--max-model-len 8192` (prompt dài nhất 5.248) thay vì 40.960:
khai thừa là bắt vLLM giữ chỗ KV cache cho độ dài không bao giờ dùng tới, mà trên
24 GB thì đúng chỗ ấy quyết định batch lớn hay nhỏ.
