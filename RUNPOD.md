# Chạy `W3-04` trên GPU thuê (RunPod)

Hướng dẫn từng bước để sinh ngữ cảnh cho 15.814 chunk bằng `Qwen3-8B` trên một
pod RunPod. Mọi con số trong này đều đo được, không phải ước lượng — trừ những
chỗ ghi rõ là **tính**.

> **Đọc trước khi bắt đầu:** bảng quyết định dưới đây đã được **viết lại ngày
> 03/09/2026** sau khi chạy thật 860 request. Kết luận cũ ("đường API rẻ hơn
> nhiều, ~$2,9") là **sai** — nó dựa trên một phép đo cache không hợp lệ. Xem
> [§0](#0-cân-nhắc-trước-có-cần-thuê-gpu-không).

---

## 0. Cân nhắc trước: có cần thuê GPU không?

Ba đường đều cho ra cùng một artifact `contexts.jsonl`.

### ⭐⭐ GLM không làm prefix caching — và đó là chuyện quyết định tất cả

Prompt của job này chia sẻ tiền tố rất lớn: các chunk liền kề trong cùng một tài
liệu dùng chung khối `<document_head>`, đo được **p50 = 9.006 ký tự ≈ 54,8%** mỗi
prompt. Toàn corpus: 64,37M token prefill, trong đó 31,3M (48,7%) là tiền tố
dùng chung. Đường nào tái sử dụng được phần đó thì rẻ hơn gần hai lần.

Đo trên GLM-5.3-Flash:

| phép đo | request | concurrency | cache trúng |
|---|---:|---:|---:|
| chạy thật, 5 tài liệu | 800 | 16 | **2,1%** |
| chạy thật, tuần tự | 60 | **1** | **0,1%** |
| *trần lý thuyết từ tiền tố chung* | — | — | *~55%* |

Chạy `--concurrency 1` là phép thử quyết định: request thứ k+1 chỉ gửi sau khi k
đã xong, nên tiền tố chắc chắn đã nằm trong cache nếu cache tồn tại. Vẫn 0,1%.
**GLM chỉ cache khi request trùng gần như toàn bộ, không cache theo tiền tố.**

⚠️ Con số "94,8% cache trúng" trong bản trước của tài liệu này đến từ việc chạy
**lại đúng 40 request giống hệt nhau**. Đó là chặn trên của lợi ích cache và nó
không bao giờ xảy ra khi chạy thật, vì khi chạy thật mọi request đều khác nhau.
Cùng họ sai lầm với ba lần fixture đồng nhất của `W3-01`/`W3-05`/`W3-07`: **một
phép đo lặp lại chính nó đo khả năng ghi nhớ, không đo khả năng làm việc.**

Ngược lại, DeepSeek đo được **49,1%** trên tập request khác nhau — sát trần 55%.
Nên DeepSeek **có** prefix caching thật, và lần trước tôi đã đọc nhầm con số đó
thành "cache còn nguội".

### Bảng quyết định (đo thật, 03/09/2026)

| đường | cache tiền tố | cost/1000 chunk | cả corpus 15.814 |
|---|---|---:|---:|
| GLM-5.3-Flash | ❌ không | **$0,6196** *(đo, 800 req)* | **~$10,6** |
| DeepSeek-v4-flash | ✅ 49,1% | $0,78 *(chiếu từ hồ sơ token)* | ~$12,4 |
| vLLM Qwen3-8B trên 4090 | ✅ tự động, theo tiền tố | theo giờ | tiền thuê × 2,5–3,5 h *(tính)* |

**Hệ quả: đường API không còn là lựa chọn rẻ.** Với workload chia sẻ 48,7% tiền
tố, pod chạy vLLM là đường duy nhất thực sự thu được khoản đó — automatic prefix
caching của vLLM là prefix trie thật.

### Đã thử cắt prompt cho rẻ, và nó hỏng

Nếu không có cache thì khối `<document_head>` (49% mỗi prompt) bị trả tiền lại từ
đầu cho từng chunk. Cắt `--head-tokens 500 --window-tokens 800` làm prompt nhỏ
2,06× và cost/1000 xuống **$0,3113**. Nhưng chất lượng sụp:

| | ngữ cảnh nêu đúng tên tài liệu |
|---|---:|
| head 2000 (mặc định) | 256/260 — **sai 1,5%** |
| head 500 | 12/260 — **sai 95%** |

Bìa tài liệu là OCR lẫn banner chương trình «Việt Nam 2035…» đứng **trước** tên
báo cáo thật «THỊ TRƯỜNG LAO ĐỘNG…». Head 500 token thấy cả hai nhưng chọn nhầm
cái nổi bật hơn; head 2000 có thêm tiêu đề chạy và mục lục phía sau để phân định.

⚠️ **Hai chỉ số bề mặt tôi dùng lúc đầu đều đạt 100% trên chính những ngữ cảnh
bịa đặt** — "có nêu tên tổ chức" 354/354, "có nêu năm" 354/354. Ngữ cảnh bịa vẫn
nói "Ngân hàng Thế giới" và vẫn có một năm trong đó. Proxy chất lượng đo cái dễ
đo chứ không đo cái cần đo; phải so tên tài liệu với sự thật mới thấy.

### ✅ Đường đã làm: gộp 8 chunk mỗi request (`TD-32`)

Gửi 8 chunk liền kề trong một request thì `<document_head>` được chia cho 8, và
vùng lân cận của chúng chồng lấn nên gộp lại gần như một dải liền.

| | token/chunk | cost/1000 | cả corpus |
|---|---:|---:|---:|
| một chunk mỗi lời gọi | 4.070 | $0,6196 | ~$10,6 |
| **gộp 8** | **728** | **$0,1724** | **~$2,7** |

⭐ **Kết quả: `W3-04` chạy xong trên laptop trong ngân sách $5,62** — 15.058/15.814
chunk (95,2%), hết ~$5,47. **Không cần pod cho `W3-04`.**

⚠️ Gộp mở ra một lỗi im lặng: model trả đủ N dòng, đúng số thứ tự, mỗi dòng hợp
lệ, nhưng gán ngữ cảnh của passage 2 cho dòng 1. Chốt chặn là bắt mỗi dòng chép
lại 4 từ đầu của passage nó mô tả; **đo được nó bắt ~17–20% số lô**. Lô bị từ
chối lùi sang `b4` rồi `b1` qua `--skip-done-chunks`.

```bash
make ctx-prepare CTX_B=8        # corpus → data/contexts/requests-b8.jsonl.gz
make ctx-run-glm CTX_B=8 CAP=2.6
make ctx-fallback               # b4 rồi b1 cho phần bị từ chối
make ctx-coverage               # còn thiếu chunk nào
```

⚠️ **Dừng job phải kiểm bằng danh sách tiến trình.** Lệnh dừng của trình bao có
thể giết shell mà không giết python, và vì job mở file theo từng lần ghi nên đổi
tên artifact **không** tách nó khỏi tiến trình đang ghi. Một tiến trình zombie đã
tiêu 27% ngân sách vào ngữ cảnh phải vứt đi trước khi bị phát hiện.

### Pod còn cần cho việc gì

`W5-11` (ablation generator) **bắt buộc** có vLLM, không thay thế được bằng API.
`W0-05` (dựng môi trường GPU) là điều kiện của nó. Phần còn lại của tài liệu này
vẫn dùng được nguyên vẹn cho lượt ấy — chỉ đổi `--backend glm` thành
`--backend vllm`.

## 1. Chuẩn bị ở laptop (~5 phút, $0)

### 1.1. Sinh gói request

```bash
make ctx-prepare
```

Nạp BGE-M3, chunk 60 tài liệu **đúng như `build_index` sẽ chunk**, dựng 15.814
prompt, ghi ra `data/contexts/requests.jsonl.gz`.

Kết quả mong đợi:

```
tài liệu           60
request            15,814
prefill ước tính   64,369,811 token (không cache tiền tố)
  tiền tố dùng chung 31,338,717 token → còn ~33,031,094 nếu cache trúng
trần cửa sổ        40,960 token · dài nhất 5,248
```

Nếu số request khác 15.814 thì chunking đã đổi — dừng lại và tìm hiểu, đừng chạy
tiếp, vì ngữ cảnh sinh cho chunk không tồn tại sẽ không bao giờ khớp `key`.

File ra nặng **8,5 MB** (thô 285 MB, nén 33× vì `document_head` lặp ở mọi dòng).

### 1.2. Commit trước khi đóng gói

⚠️ **Bước này bắt buộc và dễ quên.** Gói job dựng bằng `git archive HEAD`, tức
**chỉ chứa thứ đã commit**. Đó là lý do `.env` không bao giờ lọt vào — và cũng là
lý do code bạn vừa sửa mà chưa commit cũng không lọt vào.

```bash
git status --short     # phải sạch
```

Nếu còn thay đổi chưa commit, gói sẽ thiếu file và `make job-bundle` **từ chối
dựng** với thông báo nêu tên file thiếu. Đó là chủ ý: lỗi này xảy ra ở laptop chứ
không phải dưới dạng `ModuleNotFoundError` trên pod đang tính tiền.

### 1.3. Đóng gói và quét

```bash
make job-bundle
make job-verify
```

Kết quả mong đợi:

```
dist/runpod-job.tar.gz  (7.6 MB)
122 file · 320.4 MB đã quét · SẠCH
```

`job-verify` giải nén cả file `.gz` bên trong rồi quét — 320 MB đó là nội dung
thật của gói request. **Không đẩy lên pod nếu nó không in `SẠCH`.**

Gói chứa: mã nguồn (đã bỏ `tests/`, `plans/`, `data/`, `infra/`),
`job/requests.jsonl.gz`, và `job/run_on_pod.sh`. Không có `.env`, không có
corpus, không có manifest, không có credential nào.

---

## 2. Tạo pod (~10 phút)

### 2.1. Đặt trần chi tiêu TRƯỚC

RunPod → **Settings → Billing → Spending Limit**. Đặt một con số bạn chấp nhận
mất (ví dụ $20). Làm bước này trước khi tạo pod, không phải sau.

### 2.2. Tạo Network Volume

RunPod → **Storage → Network Volume → New**.

| trường | giá trị | vì sao |
|---|---|---|
| Region | chọn một region **có RTX 4090 Secure Cloud** | volume dính chặt với một datacenter, đổi region sau là mất |
| Size | **60 GB** | trọng số Qwen3-8B bf16 ~16 GB + cache HF + chỗ thở |
| Name | `rag-gpu` | tuỳ |

Volume tồn tại độc lập với pod. Đó là mục đích: pod bật lần hai **không phải tải
lại 16 GB trọng số** — chính là DoD của `W0-05`.

### 2.3. Tạo Pod

RunPod → **Pods → Deploy** → chọn **Secure Cloud** (⚠️ **không** Community Cloud
— quy tắc cứng #2 của dự án).

| trường | giá trị |
|---|---|
| GPU | **RTX 4090 (24 GB)** ×1 |
| Type | **On-Demand** (không Spot — job này chưa resumable ở mức pod) |
| Template | `runpod/pytorch` bản CUDA 12.x mới nhất |
| Network Volume | gắn `rag-gpu` vào **`/workspace`** |
| Container Disk | 20 GB |
| Expose | không cần cổng nào — job chạy hoàn toàn trong pod |

**Environment variables** cần đặt đúng một biến:

```
HF_HOME=/workspace/.hf
```

⚠️ Không đặt biến này thì trọng số tải vào container disk và **mất khi pod bị
xoá** — mọi lần bật lại là tải lại 16 GB.

⚠️ **Không** đặt `DEEPSEEK_API_KEY`, `GLM_API_KEY`, `HF_TOKEN`, hay bất kỳ khoá
nào. Job trên pod không cần khoá nào cả (có test ghim điều đó). Qwen3-8B là model
công khai, không cần đăng nhập HF.

---

## 3. Chuyển gói lên pod

### Cách A — `runpodctl` (khuyến nghị)

Không cần khoá SSH, dùng mã một lần, không để lại credential trên pod.

Cài ở laptop: https://github.com/runpod/runpodctl

```bash
# laptop
bash scripts/runpod_job.sh push
```

Nó tự chạy `job-verify` một lần nữa rồi in ra một lệnh dạng:

```
runpodctl receive 1234-word-word-word
```

Mở **Web Terminal** của pod (nút `Connect` → `Start Web Terminal`) và chạy đúng
lệnh đó:

```bash
cd /workspace
runpodctl receive 1234-word-word-word
```

### Cách B — SSH/`scp`

Dùng **keypair riêng cho RunPod**, không dùng lại khoá GitHub:

```bash
# laptop, làm một lần
ssh-keygen -t ed25519 -f ~/.ssh/runpod_ed25519 -C "runpod"
# dán nội dung ~/.ssh/runpod_ed25519.pub vào RunPod → Settings → SSH Public Keys

# mỗi lần đẩy
scp -i ~/.ssh/runpod_ed25519 -P <SSH_PORT> \
    dist/runpod-job.tar.gz root@<POD_IP>:/workspace/
```

`<POD_IP>` và `<SSH_PORT>` lấy ở nút `Connect` của pod.

---

## 4. Chạy trên pod

```bash
cd /workspace
tar xzf runpod-job.tar.gz

# vLLM kéo theo torch riêng của nó; ba gói còn lại là toàn bộ thứ job cần
pip install -q vllm httpx pydantic pydantic-settings

# lần đầu: tải trọng số vào /workspace/.hf (~16 GB, 5-15 phút)
GPU_HOURLY_USD=0.69 bash job/run_on_pod.sh
```

Thay `0.69` bằng **giá thật/giờ** mà RunPod đang tính cho pod của bạn. Con số này
chỉ dùng để quy thời gian chạy thành `cost/1000 chunk` trong báo cáo — nó không
ảnh hưởng gì tới kết quả, nhưng thiếu nó thì báo cáo ghi chi phí bằng 0.

Script sẽ: bật `vllm serve Qwen/Qwen3-8B` với `--max-model-len 8192` và
`--enable-prefix-caching`, đợi `/health`, rồi chạy job và tắt vLLM khi xong.

### 4.1. ⚠️ Kiểm ở phút thứ ba, đừng đợi tới giờ thứ ba

Đây là phần quan trọng nhất của cả quy trình. Sau khoảng 50 request đầu, log in
một dòng tiến độ. Dừng lại và kiểm **hai** con số:

```bash
# ở một terminal khác trên pod
head -3 job/contexts.jsonl | python -c "
import json,sys
for line in sys.stdin:
    r = json.loads(line)
    print(r['chunk_id'], '| cached', r['cached_tokens'], '| out', r['completion_tokens'])
    print('   ', r['context'][:120])
"
```

| kiểm | phải là | nếu sai |
|---|---|---|
| `reasoning_tokens` trong `run-report.json` | **0** | `chat_template_kwargs` không có tác dụng với bản vLLM này → dừng job, thêm `/no_think` vào cuối prompt hoặc dùng `--reasoning-parser` phù hợp |
| `cache_hit_rate` sau ~200 request | **> 30%** | `--enable-prefix-caching` không ăn → job sẽ chạy lâu gấp đôi; kiểm log vLLM lúc khởi động |
| ngôn ngữ của `context` | **khớp tài liệu** (vi/en) | prompt hoặc metadata sai → dừng ngay, đây là lỗi làm hỏng cả index |

Lý do phải kiểm tay: cả ba đều **không làm job đỏ**. Job vẫn chạy tới cùng và trả
ra một file trông hợp lệ.

### 4.2. Nếu job đứt

Chạy lại đúng lệnh cũ. `contexts.jsonl` **chính là** checkpoint — job đọc lại và
bỏ qua mọi `key` đã có. Không mất gì ngoài các request đang bay dở.

### 4.3. Nếu hết VRAM

Qwen3-8B bf16 chiếm ~16,4 GB; KV cache tốn **0,1406 MB/token**, nên 24 GB chỉ còn
~6,6 GB cho KV ≈ 47.000 token ≈ 11 chuỗi 4.100-token cùng lúc *(tính, chưa đo)*.
Nếu vLLM báo OOM:

```bash
# giảm bộ nhớ dành cho model, hoặc lượng hoá FP8 (4090 hỗ trợ nguyên bản)
vllm serve Qwen/Qwen3-8B --max-model-len 8192 --enable-prefix-caching \
  --gpu-memory-utilization 0.88 --quantization fp8
```

FP8 hạ trọng số còn ~8,2 GB và gấp đôi được batch.

---

## 5. Kéo kết quả về và dọn

```bash
# trên pod
runpodctl send job/contexts.jsonl
runpodctl send job/run-report.json

# laptop
runpodctl receive <MÃ>
mv contexts.jsonl run-report.json data/contexts/
```

Rồi **ngay lập tức**:

1. **Terminate pod** (không phải Stop — Stop vẫn tính tiền container disk).
2. Giữ lại Network Volume nếu còn định chạy `W5-11`; xoá nếu không.
3. Nếu có tạo token HF fine-grained cho job này thì thu hồi.

Kiểm tra file nhận được:

```bash
uv run python -c "
import json, collections, re
rows = [json.loads(l) for l in open('data/contexts/contexts.jsonl', encoding='utf-8')]
print('so ngu canh:', len(rows))
print('rong       :', sum(1 for r in rows if not r['context'].strip()))
g = lambda t: 'vi' if re.search(r'[ăâđêôơư]|ạ|ế|ộ|ữ|ị', t, re.I) else 'en/khac'
print('ngon ngu   :', dict(collections.Counter(g(r['context']) for r in rows)))
"
```

Mong đợi ~15.814 dòng, 0 rỗng, và tỉ lệ vi/en xấp xỉ tỉ lệ corpus (20 tài liệu VI
/ 40 EN, nhưng tính theo chunk thì VI chiếm 6.421/15.814 ≈ 41%).

---

## 6. Bước tiếp theo

Sau khi có `contexts.jsonl`, phần còn lại chạy ở laptop: build index từ chunk đã
có ngữ cảnh, eval, rồi `W3-09` (ablation #2). Phần nối `apply_contexts` vào
`build_index` chưa làm — đó là việc kế tiếp của `W3-04`.

---

## Phụ lục: các chế độ hỏng đã gặp

| triệu chứng | nguyên nhân | cách chữa |
|---|---|---|
| `make job-bundle` báo thiếu file | code chưa commit | `git commit` rồi dựng lại |
| `job-verify` báo phát hiện | có bí mật hoặc file cấm trong gói | đọc tên file nó in ra; **không** thêm ngoại lệ |
| `HTTP 400 modelCode: does not exist` | `--model` không khớp backend | bỏ hẳn `--model`, để mặc định theo backend |
| job xong nhưng nhiều dòng "rỗng" | `max_tokens` bị chuỗi suy luận ăn hết | kiểm `reasoning_tokens`; xem §4.1 |
| `cache_hit_rate` rất thấp | request không gom theo tài liệu, hoặc prefix caching tắt | job có cảnh báo cho ca thứ nhất; ca thứ hai kiểm log vLLM |
| ngữ cảnh sai ngôn ngữ | `DocumentMetadata.lang` sai hoặc prompt cũ | dừng ngay — lỗi này làm hỏng cả index mà không metric nào chỉ ra nguyên nhân |
| pod bật lại phải tải lại trọng số | thiếu `HF_HOME=/workspace/.hf` | đặt biến rồi tạo lại pod |
