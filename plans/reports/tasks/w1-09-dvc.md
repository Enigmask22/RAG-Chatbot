# `W1-09` — DVC init + version corpus

> Ngày: 2026-08-20 · Nhánh: `feat/w1-foundation` · DVC 3.67.1
> Trạng thái: **xong**, DoD đạt đủ (`dvc status` sạch · `dvc pull` trên clone sạch lấy đủ file)

---

## 1. Đã làm gì

| Thành phần | Nội dung |
|---|---|
| `.dvc/` | `dvc init` · analytics tắt · `config` dùng chung **không chứa remote** |
| `data/corpus.dvc` | md5 `9aeb1b7717d72b9e099160420faed9ad.dir` · 61 file · 15.457.683 byte |
| `data/.gitignore` | do DVC sinh (`/corpus`) — đã bỏ dòng trùng ở `.gitignore` gốc |
| Remote | `local` → `D:/dvc-remote/rag-chatbot`, khai ở `.dvc/config.local` (không commit) |
| `pipeline/corpus/dvc_state.py` | đối chiếu DVC ↔ manifest, kèm CLI |
| `tests/unit/test_dvc_state.py` | **23 case** |
| `Makefile` | `data-pull` · `data-push` · `data-status` · `data-verify` · `data-track` |

Tổng test: **338 unit** (trước: 317) + 33 integration. `ruff check`, `ruff format --check`, `mypy --strict` sạch trên 64 file.

---

## 2. Bằng chứng DoD

### 2.1 `dvc status` sạch

```
$ uv run dvc status
Data and pipelines are up to date.

$ uv run dvc push
62 files pushed

$ uv run dvc status -r local
Cache and remote 'local' are in sync.
```

### 2.2 `dvc pull` trên clone sạch

Clone thật vào thư mục khác, không copy gì từ working tree:

```
$ git clone --branch feat/w1-foundation /d/studioproj/RAG-Chatbot <scratch>/clone-test

$ ls -A <scratch>/clone-test/data
.gitignore  corpus.dvc  corpus_manifest.csv  golden

$ ls -A <scratch>/clone-test/data/corpus
ls: cannot access ...: No such file or directory        ← đúng: git không mang corpus
```

Chưa cấu hình remote thì chết, và chết có thông báo:

```
$ uv run dvc --cd <scratch>/clone-test pull
No remote provided and no default remote set.
ERROR: failed to pull data from the cloud - Checkout failed for following targets:
data\corpus
```

Cấu hình remote rồi pull:

```
$ uv run dvc --cd <scratch>/clone-test remote add --local -d local /d/dvc-remote/rag-chatbot
$ uv run dvc --cd <scratch>/clone-test pull
A       data\corpus\
62 files fetched and 61 files added
real 0m1.945s

$ ls -A <scratch>/clone-test/data/corpus | wc -l
61                                    ← 60 tài liệu + .gitkeep

$ uv run dvc --cd <scratch>/clone-test status
Data and pipelines are up to date.
```

### 2.3 Nội dung phục hồi là **byte-identical**

Đây là phần đáng tin hơn cả `dvc status`: so sha256 từng tài liệu mà DVC vừa khôi phục
với sha256 mà `scripts/fetch_corpus.py` ghi vào manifest lúc tải từ World Bank.

```
sha256 khớp: 60/60
cross-check: DVC 9aeb1b7717d7… · 61 file (60 tài liệu + 1 phụ trợ) · 14.7 MiB · manifest 60 tài liệu
```

60/60. Hai cơ chế versioning độc lập nói cùng một chuyện.

---

## 3. Ba quyết định thiết kế

### 3.1 Remote **không** nằm trong `.dvc/config`

`dvc remote add -d local D:/dvc-remote/rag-chatbot` ghi thẳng đường dẫn vào
`.dvc/config`, và file đó được commit. Kết quả: mọi clone nhận một remote trỏ tới
ổ D: của máy này, và lỗi chỉ lộ ra lúc `dvc pull`.

Đây đúng loại lỗi vừa gặp ở lần đổi tên workspace — `.venv/_editable_impl_*.pth`
ghi đường dẫn tuyệt đối nên `import rag_core` trỏ vào chỗ không còn tồn tại. Dự án
này chạy trên ít nhất hai máy (laptop + pod RunPod thuê theo giờ) và repo sẽ public,
nên đường dẫn máy-cụ-thể trong file dùng chung là hỏng chắc chắn, chỉ là chưa hỏng.

Cách làm: `.dvc/config` chỉ chứa comment hướng dẫn; remote thật khai bằng
`dvc remote add --local`, đi vào `.dvc/config.local` mà `.dvc/.gitignore` đã chặn sẵn.

Giá phải trả: clone sạch không pull được ngay, phải cấu hình remote trước. Đó là
đánh đổi có ý thức — thất bại ồn ào (`No remote provided`) hơn thất bại âm thầm.

### 3.2 `data/golden/` giữ trong git, **không** đưa vào DVC

Checklist ghi `W1-09` gồm cả `data/golden`. Tôi làm khác, và đây là lý do.

Golden set là **thước đo**, không phải dữ liệu. Mọi con số trong `reports/runs/baseline-retrieval.md`
và mọi cửa promotion về sau đều đo bằng nó, nên khi một nhãn đổi thì toàn bộ metric
lịch sử mất nghĩa. Thứ cần nhất ở một file như vậy là **diff đọc được lúc review**:
thấy ngay ai đổi `relevant_chunk_ids` của câu nào, từ gì sang gì. DVC thay file bằng
một con hash, và diff biến thành "md5 đổi" — mất đúng thứ quan trọng nhất.

Kích thước không phải lý do để tránh git: `draft_v1.jsonl` là 284 KB text.

Tính tái lập vẫn nguyên vẹn. Một commit git ghim cả hai: corpus qua `data/corpus.dvc`,
golden set trực tiếp. `git checkout <sha> && dvc pull` cho đúng cặp dữ liệu đã sinh ra
metric của commit đó, bất kể golden set nằm ở cơ chế nào.

Ngưỡng phải đổi ý: khi `data/golden/` bắt đầu chứa artifact sinh ra chứ không phải
nhãn người viết (ví dụ ma trận điểm reranker, cache embedding của câu hỏi). Lúc đó
tách hai thư mục, không đổi cách quản golden set.

### 3.3 DVC thêm gì khi manifest đã có sha256

Câu hỏi đáng hỏi, vì `scripts/fetch_corpus.py` + manifest đã là một cơ chế
content-addressed hoàn chỉnh cho nguồn (a).

DVC thêm đúng một thứ không thay được: **nguồn (b) và (c) không tải lại được bằng
script.** Văn bản pháp luật Việt Nam và báo cáo thường niên HOSE phải chọn tay
(và ADB đã chặn truy cập tự động bằng 403 — ghi ở `W1-07`). Với những tài liệu đó
manifest chỉ chứng minh được "file này là file tôi đã dùng", không phục hồi được nó.
DVC thì phục hồi được.

Ngoài ra: một `dvc pull` 1,9 giây thay cho 60 request ra worldbank.org, và có đường
sống khi URL gốc chết — tài liệu tổ chức quốc tế bị đổi đường dẫn khá thường.

Ngược lại, manifest cho DVC một thứ mà DVC không có: giấy phép từng tài liệu, tức
quy tắc cứng #3 được ép bằng code. Hai cơ chế bổ nhau chứ không trùng nhau.

---

## 4. Hai nguồn sự thật → một chỗ mù, và test canh nó

Hệ quả của mục 3.3: corpus giờ có hai cơ chế versioning. Chỗ **cả hai đều mù** là
phép so số lượng.

Kịch bản: kéo một file vào `data/corpus/` rồi `dvc add data/corpus`, quên cập nhật
manifest. Từ đó "corpus" nghĩa là hai tập khác nhau tuỳ hỏi ai. Và không có lỗi nào
nổ ra:

* `build_index` đọc theo manifest → bỏ qua file lạ, báo thành công 60 tài liệu
* `dvc status` → sạch, vì `.dvc` đúng là hash của thứ trên đĩa
* `dvc push` → đem file lạ lên remote như một phần chính thức của corpus

`pipeline/corpus/dvc_state.py` biến sự lệch đó thành lỗi tường minh. Cố ý **không**
băm lại nội dung — sha256 từng tài liệu đã được `iter_documents` kiểm ở mỗi lần build
index, md5 thư mục đã được `dvc status` kiểm. Việc còn thiếu chỉ là phép đếm.

```
$ make data-verify
KHỚP: DVC 9aeb1b7717d7… · 61 file (60 tài liệu + 1 phụ trợ) · 14.7 MiB · manifest 60 tài liệu
```

`61 = 60 + 1`: `.gitkeep` được DVC đếm vào `nfiles` nhưng cố ý không có trong
manifest. Có test riêng cho cả hai chiều (`test_gitkeep_is_not_counted_as_a_document`,
`test_gitkeep_absent_is_handled`) vì cách "sửa" dễ nhất khi lệch 1 — nới điều kiện
thành xấp xỉ — làm test mất hết tác dụng.

---

## 5. Lỗi bắt được lúc chạy thật

**`load_manifest` trả `[]` cho đường dẫn không tồn tại, không raise.**

Phát hiện khi thử nghịch trên clone. Đây **không** phải lỗi của `load_manifest`:
`scripts/fetch_corpus.py:335` cần đúng hành vi đó cho lần chạy đầu, khi manifest
chưa tồn tại. `pipeline/indexing/corpus_loader.py:129` đã chặn đúng.

Chỗ thiếu là `dvc_state`. Gõ sai đường dẫn manifest sẽ cho lỗi:

```
DVC theo dõi 60 tài liệu nhưng manifest khai 0. Hai cơ chế versioning đã lệch nhau…
```

Thông báo này dẫn người đọc đi sai hoàn toàn — đi tìm sự lệch không tồn tại, trong khi
nguyên nhân là một lỗi chính tả. Đã thêm guard riêng + 2 test
(`test_missing_manifest_is_its_own_error`, `test_empty_manifest_is_its_own_error`).

---

## 6. Còn nợ

| Việc | Vì sao chưa làm |
|---|---|
| Remote dùng chung (HF dataset repo / GDrive) | `D:/dvc-remote` không sống sót nếu mất máy. Nguồn (a) còn đường phục hồi độc lập qua `fetch_corpus.py`, nên chưa gấp. **Thành bắt buộc trước khi thêm nguồn (b)/(c)** — những tài liệu đó không tải lại được bằng script, mất remote là mất corpus. Ghi thành `TD-10` |
| `dvc.yaml` (DVC pipeline stage) | `W1-09` chỉ yêu cầu version dữ liệu. Việc mô tả `corpus → index → eval` thành DAG thuộc `W4-01` (`RagBundle`), và nếu làm sớm sẽ phải viết lại theo bundle |

---

## 7. Lệnh tái lập

```bash
# Lần đầu trên một máy mới
uv sync --extra data --extra pipeline
uv run dvc remote add --local -d local /đường/dẫn/ngoài/repo
make data-pull
make data-verify        # phải in "KHỚP: … 60 tài liệu … manifest 60 tài liệu"

# Sau khi corpus đổi
make data-track         # dvc add + kiểm chéo manifest ngay
make data-push
```
