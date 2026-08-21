# `W2-07` — Experiment Runner + MLflow

> Trạng thái: **xong** · 2026-08-21
> DoD: 1 lệnh chạy hết grid ✅ · resume được khi crash giữa đường ✅ ·
> Test: `tests/unit/test_experiment_runner.py` — **80 case** ✅ · Evidence: §8 (bảng
> 14 ô đọc từ MLflow) + §8b (crash `SIGKILL` rồi resume)
> Lệnh: `make exp-dry` · `make exp` · `make exp-backfill` · `make mlflow-ui`

---

## 1. Dự đoán ghi TRƯỚC khi viết code

Lệ này bắt đầu từ `W2-05` (3/7 sai, cả 3 lệch cùng hướng) và `W2-06` (3/5 sai).
Ghi trước thì sai được đếm; ghi sau thì mọi thứ đều "như dự kiến".

| # | Dự đoán | Kết quả |
|---|---|---|
| `D1` | Phần khó không phải expand grid mà là **resume đúng** — phân biệt "ô đã chạy" với "ô đã chạy *với đúng tham số hiện tại*" | ✅ **Đúng.** 63 test, phần lớn nằm ở đây; xem §3 |
| `D2` | Nạp lại model là chi phí lớn nhất; gom ô theo `index_config` sẽ cắt được phần lớn (~15 s/lần × 12 ô ≈ 3 phút) | ❌ **Sai.** `rag_core` đã có `lru_cache` trên **cả ba** loại model từ `W1`. Gom lại mua **quét nhãn span** (14 → 3), không mua lần nạp model. Xem §5 |
| `D3` | Preflight **không** kiểm được tham số nhánh mà không nạp model, vì `build_branch` dựng cross-encoder trong lúc kiểm | ❌ **Sai.** Phần kiểm tách ra được thành hàm thuần (`check_branch_options`) và preflight dùng lại **đúng** hàm đó. Xem §2 |
| `D4` | `_resolve_span_labels` là hàm của `(index, min_overlap_ratio)` chứ không của nhánh | ✅ **Đúng**, và đo được: 14 ô → **3** lần quét |
| `D5` | MLflow là phần **nhàm nhất** — `log_param`/`log_metric` là API phẳng | ❌ **Sai, và sai nhiều nhất.** MLflow là chỗ sinh ra **hai** lỗi thật của hạng mục này, cả hai đều là loại chạy-xong-mà-không-có-dữ-liệu. Xem §6 và §6b |
| `D6` | MLflow kéo theo web stack và **không** cùng tồn tại êm với `torch` CUDA đã ghim | ❌ **Sai.** mlflow 3.15.1 + torch 2.13.0+cu126, `cuda.is_available() == True`, `uv sync --all-extras` chạy **3,9 s** nhờ hardlink từ `D:\uv-cache` |

**2/6 đúng, 4 sai.** Và bốn lần sai chia làm hai nhóm rõ rệt:

* **Ba lần đánh giá quá cao độ khó của phần *code*** — tưởng một thứ không tách
  ra được (`D3`), tưởng hai thư viện không cùng tồn tại được (`D6`), tưởng phải
  tự tối ưu thứ thư viện đã tối ưu (`D2`).
* **Một lần đánh giá quá thấp độ khó của phần *tích hợp*** (`D5`): tôi coi MLflow
  là API phẳng nên nhàm, và nó là chỗ duy nhất trong hạng mục này *thật sự* hỏng
  — hai lần, cả hai đều theo kiểu "chạy xong, đúng số ô, không có dữ liệu".

Nhìn chung thì tôi hiệu chuẩn tốt về việc *viết* code và kém về việc *nối* nó vào
một hệ thống bên ngoài mà tôi không kiểm soát phiên bản. Hướng lệch **ngược** với
`W2-05`, ở đó tôi đánh giá quá thấp chi phí cross-encoder ba lần liền.

---

## 2. `check_branch_options` — phần refactor mà `D3` nói là không làm được

Preflight phải trả lời "ô thứ 11 có hợp lệ không" **trước khi** chạy ô thứ nhất.
Luật "nhánh nào nhận tham số nào" chỉ có ở một chỗ — `build_branch` — nhưng nó chỉ
nói được câu trả lời bằng cách **dựng** retriever, và dựng nhánh `reranked` nạp một
cross-encoder 2,2 GB. Nên `D3` kết luận là không kiểm được mà không trả giá đó.

`D3` sai vì có đường thứ ba: **tách phần kiểm ra khỏi phần dựng.**

```python
def check_branch_options(mode, options) -> RetrievalMode:  # thuần, không dựng gì
def build_branch(store, mode, **options) -> Retriever:     # gọi hàm trên rồi dựng
```

Cách **sai** là chép luật sang tầng pipeline. Luật gồm cả phần đệ quy của
`reranked` (tham số ngoài `RERANK_OPTIONS` đi xuống nhánh nền và bị từ chối ở đó),
nên bản chép sẽ lệch dần và preflight sẽ **cho qua** những ô mà `build_branch` từ
chối — tức grid chạy 40 phút rồi chết ở ô cuối, đúng cái chế độ hỏng hạng mục này
tồn tại để chặn, chỉ là được dựng lên bởi chính bản sửa của nó. Có test
`TestBranchValidationCannotDrift` chạy **15 cặp đầu vào** qua cả hai đường và
khẳng định chúng nổ ở cùng những chỗ.

Phụ phẩm: thêm `HYBRID_OPTIONS` để kiểm được cả tham số **lạ** cho hybrid. Trước
đó `build_branch(store, "hybrid", candidat_k=100)` đi thẳng vào constructor và
nhận một `TypeError` trần — đúng là nổ, nhưng không liệt kê tham số hợp lệ, cùng
loại thiếu sót với khoá filter gõ sai ở `W2-06`. Giờ:

```
ValueError: Nhánh 'hybrid' không nhận tham số ['candidat_k'].
            Hợp lệ: ['candidate_k', 'k', 'weights'].
```

Và `HYBRID_OPTIONS` có test đối chiếu với **chữ ký thật** của
`QdrantHybridRetriever.__init__` qua `inspect`, vì không có test đó thì nó là một
bản chép tay của chữ ký và bản chép tay sẽ lệch — lệch theo hướng *từ chối một
tham số hợp lệ*.

---

## 3. Resume: một cách cài đúng, ba cách cài trông đúng

DoD chỉ nói "resume được khi crash giữa đường". Đọc như checklist thì nó là "lưu
danh sách ô đã xong". Ba cách làm thế đều cho một grid **"chạy xong" với số sai và
không có gì báo lỗi**:

| Cách cài | Hỏng khi nào | Kết quả |
|---|---|---|
| Bỏ qua ô **có file báo cáo** | Sửa `k: 60` → `k: 5` rồi resume | Số của `k=60` vào bảng dưới nhãn `k=5` |
| Bỏ qua ô **có tên trong state** | Sửa `chunk_size` của `bgem3.yaml`, build lại, resume | Mọi ô bị bỏ qua; bảng trộn hai index |
| Ghi state **trước** báo cáo | Ctrl+C giữa hai bước | Ô `done` không có báo cáo, resume bỏ qua nó mãi mãi |

Cách đúng là **fingerprint của ô**, và điểm mấu chốt là nó nhận hai thứ **từ
ngoài**:

```python
def fingerprint(self, *, index_fingerprint: str, golden_digest: str) -> str
```

Kết quả của một ô không chỉ phụ thuộc những gì viết trong ô. Build lại
`bgem3.yaml` với `chunk_size` khác, hoặc `TD-13` review lại golden set và ghi lại
**cùng đường dẫn** — cả hai đổi kết quả mà không đổi một ký tự nào trong YAML của
ô. `golden_digest` băm **nội dung** chính vì thế, không băm tên hay mtime.

Ba chi tiết nhỏ mà thiếu thì resume vẫn hỏng:

* **`failed` ≠ `done`.** Ô chết vì OOM phải được thử lại. Có test.
* **Thứ tự chạy suy được từ file config.** Gom theo index là gom **ổn định**
  (index xuất hiện trước trong YAML thì cả nhóm chạy trước, trong nhóm giữ thứ tự
  khai báo). Không thế thì hai lần resume cho hai thứ tự và log không so được.
* **Nói ra lý do chạy lại.** Một ô chạy lại mà không giải thích trông y như resume
  hỏng, và người dùng sẽ đi sửa resume thay vì hiểu là tham số đã đổi. Log in
  `fingerprint` cũ → mới.

Ghi file thì **báo cáo trước, state sau**, và mỗi file ghi qua `tmp` + `os.replace`.
State hỏng thì `load` cảnh báo rồi bắt đầu lại từ đầu thay vì chết — chạy lại tệ
hơn resume một chút, nhưng tốt hơn nhiều so với bắt người dùng tự xoá file.

Một test đáng nhắc riêng: `test_windows_and_posix_paths_agree`. `fingerprint`
chuẩn hoá `\` → `/`, vì nếu không thì grid chạy trên laptop Windows và grid chạy
trên pod Linux (`W0-05`) coi mọi ô là khác nhau và không lần nào resume được lần kia.

---

## 4. Preflight bắt được lỗi thật ở lần `--dry-run` **đầu tiên**

Lượt `--dry-run` đầu trên config thật dừng ngay:

```
ERROR Grid 'exp-001-retrieval' có 1 vấn đề, không chạy ô nào:
  - plans\reports\runs\bgem3-sparse-retrieval.json đã tồn tại nhưng state của thí
    nghiệm này không có ô 'bgem3-sparse' — nó là bằng chứng của một lần chạy khác
    và ô này sẽ ghi đè.
```

Grid sinh ô tên `bgem3-sparse`, trùng **đúng** báo cáo tiêu đề của `W2-03`. Không
có kiểm này thì `make exp` sẽ **ghi đè bằng chứng của một hạng mục đã xong**, và
`plans/reports/runs/` đang giữ 57 file như thế của `W2-01`…`W2-05`. Nó là loại
hỏng không hoàn tác được: file cũ không nằm trong git (chúng *có* nằm trong git,
nhưng người phát hiện ra sẽ là người đọc `git diff` sau đó, không phải người chạy
lệnh).

Sửa bằng `run_prefix: e1` ở cấp config, nên mọi ô của grid mang tiền tố và không
gian tên của grid tách khỏi không gian tên của các lần chạy tay.

Đường **không** chọn: để grid *dùng lại* những lần chạy cũ trùng tên. Nghe tiết
kiệm, nhưng state không sở hữu chúng nên không biết chúng được sinh bằng tham số
nào — tức không kiểm được là chúng có khớp ô hiện tại hay không, mà đó đúng là câu
`fingerprint` tồn tại để trả lời.

Preflight gom **tất cả** vấn đề rồi nổ một lần (cùng lý lẽ `Settings` ở `W1-01`):
sửa một lỗi rồi chạy lại để thấy lỗi kế tiếp là vòng lặp mà mỗi vòng ở đây tốn một
lần nạp model.

⚠️ **Điều preflight KHÔNG bắt được, và đây là giới hạn thật:** ô `sparse` trên một
index không có sparse vector. Đó là tính chất của **collection trong Qdrant**,
không của file config, nên nó chỉ lộ ra khi `QdrantSparseRetriever` được dựng.
Cùng họ với chuyện `W2-06` không ép được `tenant_id`: có những thứ tầng này không
có đủ thông tin để biết. `--keep-going` là cái van cho nó.

---

## 5. `D2` sai: model đã được chia sẻ sẵn, và `_release()` không làm được việc nó nói

Đo từ log grid thật:

| Mở index | Thời gian | Vì sao |
|---|---|---|
| `baseline.yaml` (lần đầu) | **27,8 s** | nạp model + kiểm cache HF qua mạng |
| `chunk550.yaml` | **0,4 s** | **cùng model** với baseline |
| `bgem3.yaml` | **11,1 s** | model khác, lần đầu |

`0,4 s` là con số phản chứng `D2`. Đi tìm thì thấy `rag_core` đã có `lru_cache`
trên **cả ba** loại model từ `W1`:

| Chỗ | `maxsize` |
|---|---|
| `embedding/huggingface.py::_load_model` | 4 |
| `embedding/bge_m3.py::_load_sparse_head` | 4 |
| `reranking/cross_encoder.py` | 2 |

Nên gom ô theo index **không** mua lần nạp model. Nó mua hai thứ khác:

1. **Quét nhãn span: 14 → 3.** Đo được đúng 3 dòng `Ánh xạ span → chunk` cho 14 ô.
   Và đây không chỉ là tối ưu — nó *nói ra* được điều `compare.py` cần: các ô cùng
   index dùng **cùng** nhãn nên so theo cặp được, ô khác index thì không.
2. **Mỗi model nạp đúng một lần bất kể `maxsize`.** Grid quét 5 model chạy xen kẽ
   với `maxsize=4` sẽ đá nhau ra khỏi cache và nạp lại liên tục; gom lại thì không.
   Đây là bảo hiểm, không phải tối ưu cho grid hiện tại.

### `_release()` — bản đầu của tôi nói sai

Docstring đầu tiên viết là nó ngăn grid giữ hai model embedding cùng lúc. **Sai.**
`lru_cache` giữ tham chiếu mạnh, nên `del` + `gc.collect()` trong runner không
chạm được vào trọng số. Đo trong lúc grid chạy nhánh `reranked`: **4517/8188 MiB**,
đúng bằng BGE-M3 + cross-encoder cùng nằm đó.

Nó vẫn làm một việc thật (trả bộ đệm hoạt hoá của index trước, không nhỏ với batch
512 token), nhưng docstring đã được sửa lại cho đúng, kèm hệ quả **quan trọng hơn
chính hàm đó**:

> ⚠️ Trần VRAM của một grid do ba con số `maxsize` ở `rag_core` quyết định,
> **không** do runner. Grid quét 4 model embedding giữ 4 × 2,2 GB và OOM trên card
> 8 GB, và runner không có cách nào ngăn.

Đây là đầu vào cụ thể cho `W0-06` (ngân sách VRAM) — và là lý do đáng chú ý: hạng
mục này *phát hiện* ra giới hạn đó chứ không *tạo* ra nó, nhưng nó là hạng mục đầu
tiên chạy đủ nhiều model liên tiếp để giới hạn đó có nghĩa.

---

## 6. MLflow, lỗi thứ nhất: grid chạy trọn 14 ô mà không ghi được gì

MLflow là thứ duy nhất của hạng mục này mà thiếu nó thì **không mất dữ liệu**: ba
file trong `plans/reports/runs/` vẫn là nguồn sự thật, `compare.py` vẫn đọc chúng,
`G2` vẫn kiểm định được. Nên bản đầu để nó là extra tuỳ chọn và **mọi** lỗi mở
tracker đều rơi về `NullTracker` kèm một cảnh báo — lý lẽ: một grid 40 phút không
được chết ở giây thứ nhất vì thiếu thư viện dùng để xem lại.

Lượt chạy grid đầu tiên cho thấy lý lẽ đó đúng một nửa. **mlflow 3.15 từ chối
`file:./mlruns`** (file store vào maintenance mode, đòi `sqlite:///`). Grid chạy
trọn 14 ô — đúng số, đủ file, exit 0 — và Evidence của DoD **không tồn tại**.
Cảnh báo thì có: **dòng 19 trong một log 2320 dòng**, nổ ở giây 0 rồi bị chôn dưới
2270 dòng HTTP (§9).

Sửa bằng cách **phân loại lỗi theo kiểu**, không theo mức nghiêm trọng:

| Tình huống | Xử lý | Lý do |
|---|---|---|
| `tracking_uri: null` | `NullTracker`, im lặng | Một lựa chọn tường minh |
| Thiếu mlflow | `NullTracker` + cảnh báo | Extra `tracking` là **tuỳ chọn có chủ đích** |
| Có URI mà không mở được | **`TrackingUnavailable` → preflight** | Khai một đích đến mà không tới được là **lỗi config**, và lỗi config thuộc preflight |
| Hỏng **giữa** grid (server chết ở ô 7) | `SafeTracker` nuốt + cảnh báo | Không preflight được, và một ô đã chạy 131 giây không được mất vì chuyện này |

Dòng thứ ba là chỗ đổi ý: nếu bạn *muốn* chạy không theo dõi thì viết
`tracking_uri: null`. Còn khai một đích đến thì việc tới được nó là điều kiện, và
điều kiện được kiểm ở giây thứ nhất.

### `backfill.py` — biến một khẳng định kiến trúc thành một phép kiểm

Còn 14 ô đã chạy mà chưa lên MLflow. Hai đường: chạy lại 25 phút, hoặc **chứng
minh là không cần chạy lại**. Đường thứ hai vừa nhanh hơn vừa kiểm được một câu mà
`tracking.py` đã tuyên bố:

> `NullTracker` là chế độ chạy hợp lệ, không phải chế độ suy giảm. Nếu MLflow là
> chỗ duy nhất giữ một con số thì con số đó không tái lập được từ repo.

`make exp-backfill` dựng lại toàn bộ bảng MLflow **từ file báo cáo**, không chạy
lại một truy vấn nào. Nếu nó không làm được thì câu trên sai và MLflow đã lặng lẽ
trở thành nguồn sự thật.

Để hai đường **không thể** lệch nhau, cả `_run_one` và `backfill` gọi
`report_params(json.loads(...))` trên **cùng byte**: đường trực tiếp đọc
`report.to_json()`, đường backfill đọc file, và có test ghim `_write_report` ghi
đúng `report.to_json()`. Không phải "tôi cẩn thận cho khớp" — là cùng một nguồn.

---

## 6b. MLflow, lỗi thứ hai: 14 run, **0 metric**, và một dòng log nói "Đã log 14 ô"

`make exp-backfill` chạy xong, báo "Đã log 14 ô lên MLflow". Kiểm lại:

```
run: 14 · cột metrics: []
```

**MLflow không nhận `@` trong tên metric** (chỉ cho `[A-Za-z0-9_-./ ]`) — và *mọi*
metric của dự án này mang `@`: `hit_rate@1`, `ndcg@10`, `recall@5`, `map@20`. Param
vào hết, metric không một cái nào.

Đáng ghi là **cách nó lọt**: `SafeTracker` — thứ tôi vừa viết ở §6 để lỗi tracking
không làm mất kết quả eval — nuốt đúng 14 lần và in 14 dòng cảnh báo, rồi kết thúc
bằng một dòng INFO "Đã log 14 ô".

> **Khoan dung đúng với lỗi nhất thời và sai với lỗi hệ thống, mà ở chỗ gọi thì hai
> loại giống nhau.**

Nên cách sửa **không** phải nới `SafeTracker` (nó đang làm đúng việc của nó cho
trường hợp nó được thiết kế cho), mà là **xoá cả lớp lỗi ở nguồn**: đổi tên ở tầng
adapter MLflow (`@` → `_at_`, nên `ndcg@10` → `ndcg_at_10`), cộng một test ghim
rằng **mọi** tên metric `evaluate_run` sinh ra đều hợp lệ sau khi đổi. Test đó là
phần bền: thêm một metric mới mang ký tự lạ thì nó đỏ **trước** khi một grid 13
phút ghi ra một bảng rỗng.

Đổi tên chỉ ở **tầng hiển thị**. File báo cáo giữ nguyên `@` — chúng là nguồn sự
thật và `compare.py` đọc chúng. `_at_` chứ không phải `.` vì `ndcg.10` bị MLflow
hiểu là phân cấp và nhóm chung với `ndcg.20` trong UI.

Một chi tiết không nhàm nữa: `_MlflowRun.__exit__` đặt `status="FAILED"` khi có
exception. Không có nó thì UI hiện 14 run xanh cho một grid có 11 ô chạy được.

## 7. Cấu trúc `matrix`: danh sách khối, không phải một tích Descartes

Không gian tham số **không phải hình hộp**: `k`/`candidate_k` chỉ có nghĩa với
`hybrid`, `rerank_candidates` chỉ với `reranked`. Tích đầy đủ sinh ra `dense × k=1`
— ô mà `build_branch` từ chối, đúng ra phải từ chối.

Hai đường thoát đều tệ hơn:

* **Sinh hết rồi lọc ô không hợp lệ.** "12 tổ hợp" trong DoD `W2-08` thành con số
  không đoán được từ file config, và một ô *bị lọc vì gõ sai tên* trông giống hệt
  một ô *bị lọc vì vô nghĩa về ngữ nghĩa*.
* **Cho `null` nghĩa là "chiều này không áp dụng".** Đọc được, nhưng số ô vẫn là
  tích và người đọc phải tự nhân trong đầu.

Nên `matrix` là **danh sách khối**, mỗi khối một tích nhỏ đồng nhất về nhánh —
đúng cấu trúc `matrix.include` của GitHub Actions, vì cùng lý do.

Hai quy tắc nhỏ đi kèm, cả hai là biến thể của bài học `W2-06`:

* **`options` luôn là list**, kể cả một giá trị. Cho phép cả scalar lẫn list thì
  phải đọc kiểu từng giá trị mới biết grid to bằng bao nhiêu.
* **`k: []` nổ**, không phải "bỏ qua chiều này" — tích Descartes với list rỗng cho
  **0 ô**, tức grid im lặng thành rỗng. Cùng lý lẽ `MatchAny(any=[])` ở `W2-06`.

### Chiều bị **từ chối**: `rerank_batch_size`

Quét một knob không đổi kết quả sinh ra hai dòng bảng chắc chắn giống nhau trong
phạm vi nhiễu — và một dòng như thế đọc y như một phát hiện. Điểm cần cẩn thận:
đây là lập luận **nhất quán**, không phải một khẳng định thực nghiệm mới.
`IndexConfig.fingerprint` cố ý loại `batch_size` với đúng lý lẽ đó từ `W1-08`, nên
dùng nó làm chiều ablation là tự mâu thuẫn với một quyết định đã ghi. Ghim một giá
trị vẫn được (list một phần tử) — đó là chuyện khác.

⚠️ Và chỗ dễ nhầm: **`rerank_device` KHÔNG bị từ chối.** Nó *trông* như knob tốc
độ, nhưng `rerank_dtype` mặc định là `auto` = fp16 trên CUDA và fp32 ở nơi khác,
nên đổi device **đổi cả dtype**, mà `W2-05` đo được fp16 đổi top-1 ở 1/60 câu. Quét
device là quét dtype một cách vô tình. Vì thế `exp-001` ghim `rerank_dtype:
[float16]` **tường minh** thay vì để `auto` — nếu không thì cùng một file config
cho hai kết quả khác nhau trên hai máy.

### Tên ô

Chỉ những chiều **thật sự biến thiên** trong khối vào tên:
`rr-bgem3-reranked-ondense-rc20-devcuda-dtfloat16` →
`e1-rr-bgem3-reranked-ondense-rc20`. Giá trị ghim giống nhau ở mọi ô của khối nên
nó chỉ làm tên dài mà không phân biệt được gì; nó vẫn vào `config.branch_options`
của báo cáo và vào param MLflow, nên không mất thông tin.

⚠️ Hệ quả phải biết: đổi một giá trị **ghim** (`float16` → `float32`) không đổi
tên, nên file báo cáo cũ bị ghi đè. `fingerprint` đổi nên resume chạy lại đúng ô
đó; chỉ là con số cũ không còn nằm cạnh con số mới để so. Muốn giữ cả hai thì cho
dtype thành chiều hai giá trị, hoặc đổi `label`.

Trùng tên thì **nổ**: hai ô cùng tên ghi lên cùng ba file, ô sau xoá ô trước, và
bảng vẫn đủ dòng — chỉ là hai dòng mang cùng một con số.

---

## 8. Grid thật: 14 ô, `0` lỗi, ~13 phút

`make exp` trên `configs/eval/exp-001-retrieval.yaml`. Bảng dưới đọc **từ MLflow**,
mà bảng MLflow lại được dựng từ file báo cáo bằng `make exp-backfill` — nên nó cũng
là bằng chứng cho §6.

| run | nDCG@10 | hit@1 | recall@5 | p95 (ms) | nhãn/câu |
|---|---|---|---|---|---|
| `e1-rr-…-onhybrid-rc100` | **0,6736** | **0,5789** | **0,7352** | 1163,9 | 1,3828 |
| `e1-rr-…-ondense-rc100` | 0,6624 | 0,5742 | 0,7185 | 1182,3 | 1,3828 |
| `e1-rr-…-onhybrid-rc50` | 0,6481 | 0,5598 | 0,7026 | 608,9 | 1,3828 |
| `e1-rr-…-ondense-rc50` | 0,6268 | 0,5455 | 0,6786 | 618,5 | 1,3828 |
| `e1-rr-…-onhybrid-rc20` | 0,5823 | 0,5263 | 0,6220 | 276,5 | 1,3828 |
| `e1-rr-…-ondense-rc20` | 0,5676 | 0,5072 | 0,6069 | 267,6 | 1,3828 |
| `e1-rrf-bgem3-hybrid-k0` | 0,4582 | 0,3493 | 0,5088 | 47,1 | 1,3828 |
| `e1-rrf-bgem3-hybrid-k1` | 0,4563 | 0,3397 | 0,5088 | 49,1 | 1,3828 |
| `e1-rrf-bgem3-hybrid-k2` | 0,4521 | 0,3301 | 0,5136 | 46,7 | 1,3828 |
| `e1-bgem3-dense` | 0,4442 | 0,3397 | 0,4769 | 46,3 | 1,3828 |
| `e1-rrf-bgem3-hybrid-k60` | 0,4313 | 0,3014 | 0,4753 | 48,0 | 1,3828 |
| `e1-bgem3-sparse` | 0,3733 | 0,2919 | 0,4171 | 44,2 | 1,3828 |
| `e1-baseline-dense` | 0,1621 | 0,1196 | 0,1746 | 42,8 | 1,3828 |
| `e1-chunk550-dense` | 0,1215 | 0,0861 | 0,1295 | 46,1 | **1,9617** |

⚠️ **Đây KHÔNG phải `W2-08`.** DoD của `W2-08` đòi `p`/CI **cho từng dòng**, tức
`compare.py`, chưa chạy. Bảng này là *bằng chứng runner chạy đúng*, không phải một
kết luận. Đọc nó như kết luận sẽ lặp lại đúng lỗi `TD-11`: chênh 0,6736 vs 0,6481
trên 209 câu là **5 câu**, và không ai biết đó là cải thiện hay nhiễu.

### Cột `nhãn/câu` làm đúng việc nó được thêm vào để làm

`chunk550` là **1,9617**; mười ba ô kia đều **1,3828**. Đó là `G2` hiện ra trong
một cột: `chunk_size` khác → số nhãn mỗi câu khác → mẫu số của recall@k khác, nên
`e1-chunk550-dense` **không so được** recall/nDCG/MAP với phần còn lại. Log của
grid nói cùng chuyện theo cách khác:

```
[1] baseline : 209 tính lại · 33 giữ nhãn cũ ·   9 đổi nhãn
[2] chunk550 : 209 tính lại · 33 giữ nhãn cũ · 209 đổi nhãn
[3] bgem3    : 209 tính lại · 33 giữ nhãn cũ ·   9 đổi nhãn
```

`compare.py` tự từ chối so, nhưng nó từ chối *lúc so*. Cột này làm chuyện đó thấy
được **trước** khi ai đó ghép hai dòng vào cùng một bảng.

### Năm con số tái lập đúng từng chữ số — phép kiểm mạnh nhất của đợt refactor

| Ô | Số đo | Số đã công bố |
|---|---|---|
| `e1-baseline-dense` | nDCG@10 **0,1621** | `W2-01` baseline ✅ |
| `e1-bgem3-dense` | nDCG@10 **0,4442** | `W2-01` BGE-M3 ✅ |
| `e1-bgem3-sparse` | nDCG@10 **0,3733** | `W2-03` ✅ |
| `e1-rrf-bgem3-hybrid-k1` | nDCG@10 **0,4563** | `W2-04` `k=1` ✅ |
| `e1-rr-…-onhybrid-rc50` | nDCG@10 **0,6481** · hit@1 **0,5598** | `W2-05` ✅ |

Đợt này refactor `_eval_against_index` → `IndexSession`, đổi
`_resolve_span_labels(retriever, …)` → `(store, …)`, và đưa việc ghi ba file qua
`tmp` + `os.replace`. Năm dòng trên là cách duy nhất biết những thay đổi đó **không
đổi một con số nào**.

Về chỗ đổi `retriever` → `store`: nếu nhánh `reranked` trước đây **không** có
`fetch_doc_chunks` thì nó đã rơi vào nhánh cảnh báo và được chấm bằng nhãn cũ,
nghĩa là số `W2-05` không so được với dense. Đã kiểm: cả bốn nhánh đều forward
(`sparse.py:20` còn ghi rõ *vì sao* phải forward), và mọi báo cáo `reranked` đã
công bố **đều có** `config.span_resolution`. Refactor không đổi số nào; nó biến
tính chất đó thành **cấu trúc** thay vì phụ thuộc mỗi wrapper nhớ forward.

---

## 8b. Chứng minh resume — và lần thử đầu không hợp lệ

Test đơn vị ghim logic resume, nhưng DoD nói "resume được khi **crash** giữa
đường", và đó là một phát biểu về tiến trình bị giết, không về một hàm.

### Lần thử thứ nhất: sai, và cái vạch ra là chính resume

`timeout -s KILL 55 uv run python -m pipeline.experiments.runner …` → exit 137, và
state có đúng 2 ô `done`. Trông như đúng ý. Nhưng lượt resume in:

```
Bỏ qua 3 ô đã xong: ct-baseline-dense, ct-chunk550-dense, ct-bgem3-dense
```

**Ba**, không phải hai. `timeout` giết `uv run` chứ **không** giết tiến trình python
con, nên nó chạy xong ô 3 và ghi state trong lúc tôi đang kiểm. Phép thử không đo
cái nó nói là đo. Đáng ghi vì thứ phát hiện ra là **một mâu thuẫn nội tại** giữa
hai quan sát (state 2 ô vs resume nói 3 ô) — không có lượt resume đó thì tôi đã báo
"crash-resume chạy đúng" dựa trên một tiến trình chưa từng bị crash.

### Lần thử thứ hai: gọi `python.exe` của venv trực tiếp

```
exit=137 (KILL, giữa ô 3)
state:  ct2-baseline-dense done · ct2-chunk550-dense done
đĩa:    6 file (2 ô × 3), KHÔNG file .tmp nào sót
resume: Bỏ qua 2 ô đã xong · [1/1] ct2-bgem3-dense xong · 3/3 ô xong · 0 ô lỗi
```

Đúng ba tính chất cần: state không nhận ô dở, `os.replace` không để lại file cắt
dở, và resume chạy **đúng một** ô còn thiếu.

### Và ca quan trọng hơn crash: đổi tham số rồi resume

Sửa `top_k: 20` → `10` rồi chạy lại:

```
Ô ct2-baseline-dense chạy lại: fingerprint đổi (8c3873c2… → e4e31794…)
Ô ct2-chunk550-dense chạy lại: fingerprint đổi (75586b6f… → 33620bb8…)
Ô ct2-bgem3-dense   chạy lại: fingerprint đổi (c6fd5b9e… → bba574d1…)
```

Đây là ca mà mọi cách cài "trông đúng" ở §3 đều **im lặng bỏ qua** và đưa số cũ
vào bảng mới. Và chú ý là nó *nói ra lý do* — không có dòng đó thì ba ô chạy lại
trông y như resume bị hỏng.

## 9. Log: 97% là nhiễu, và DoD chỉ đạt một nửa nếu không sửa

Lượt chạy đầu ghi **2320 dòng, trong đó 2270 dòng (97%) là
`HTTP Request: HEAD https://huggingface.co/...`** của httpx khi
sentence-transformers kiểm cache. Mười ba dòng tiến độ thật bị chôn trong đó.

DoD "1 lệnh chạy hết grid" đạt về **chức năng** mà không đạt về việc **đọc được nó
đã chạy gì** — và với một lệnh chạy 40 phút thì đó là cùng một yêu cầu. Sửa bằng
cách hạ từng logger tường minh, **không** hạ root: cảnh báo của
`_resolve_span_labels` ("N câu có span nhưng không khớp chunk nào của index này")
là thứ phải thấy được.

---

## 10. Việc còn lại

* **`W2-08`** giờ có dữ liệu 14 ô, nhưng DoD của nó đòi **`p`/CI cho từng dòng** —
  đó là `compare.py`, chưa chạy. Và `W2-09` đòi "category nào cải thiện nhiều
  nhất", nên **`--category`/`--lang` cho `compare.py` phải làm trước**. Thiếu chiều
  đó đã để một mức tụt có ý nghĩa của `W2-04` đi qua không ai thấy.
* **`TD-19`** (mới, từ §6): một file `runs/*-retrieval.json` **không nói được nó đo
  trên golden set nào**. Vô hại hôm nay vì chỉ có một golden set; `TD-13` sẽ tạo
  cái thứ hai với **cùng đường dẫn**. Phải làm **trước** `TD-13`, không phải sau.
  Phía grid đã an toàn — `fingerprint` băm `golden_digest` từ hạng mục này.
* **`W0-06`** có đầu vào cụ thể từ §5: trần VRAM do ba con số `maxsize` ở `rag_core`
  quyết định, không do runner.
* Preflight **không** bắt được ô `sparse` trên index không có sparse vector (§4) —
  đó là tính chất của collection, không của file config. `--keep-going` là cái van.
* Runner **không** chia sẻ `IndexSession` giữa các lần gọi `make exp`, nên resume
  vẫn nạp lại model một lần cho mỗi index còn việc. Không đáng sửa: 11,1 s một lần.
* Cột `p95` của grid dùng để **sàng**, không để kết luận: 13 phút chạy liên tục
  trên GPU laptop, không có đối chứng thứ tự. Số độ trễ đáng tin đến từ probe riêng
  (`W2-04` §6, `W2-06` §5).
