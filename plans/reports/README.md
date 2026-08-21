# `plans/reports/` — bản đồ

> Cập nhật 2026-08-21 · 102 file, chia 5 folder theo **ai sinh ra nó**, không theo chủ đề.

Trước đây 97 file nằm phẳng trong một thư mục, và ba loại rất khác nhau bị trộn
lẫn: tường thuật viết tay, output máy sinh, và artifact dữ liệu. Ranh giới dùng để
chia là **cái gì sinh ra file** — vì đó cũng là ranh giới của việc *được phép sửa
tay hay không*.

| Folder | Chứa gì | Ai sinh | Sửa tay? |
|---|---|---|---|
| [`tasks/`](tasks/) | Tường thuật một hạng mục `Wx-xx`: câu hỏi, dự đoán ghi trước, số đo, kết luận, cái gì sai | Tôi viết | ✅ đây là chỗ duy nhất |
| [`runs/`](runs/) | Output của **một lần chạy eval**: `<run>-retrieval.json` + `.md` + `<run>-per-query.jsonl` | `pipeline.eval.retrieval_eval` | ❌ sửa là làm hỏng bằng chứng |
| [`compare/`](compare/) | `cmp-<base>-vs-<cand>.md` — bảng có `p`/CI95 | `pipeline.eval.compare` | ❌ |
| [`probes/`](probes/) | Output script đo lẻ + report build index | `scripts/*_probe.py`, `pipeline.indexing.*` | ❌ |
| [`goldenset/`](goldenset/) | Provenance + checksum của tập nhãn | `pipeline.goldenset.*` | ❌ |

**Đọc thứ tự nào:** vào `tasks/` trước. Mọi file ở bốn folder còn lại đều được một
file trong `tasks/` dẫn tới kèm lời giải thích; đọc ngược lại thì chỉ thấy số mà
không thấy vì sao đo.

---

## Review theo `Wx-xx`

Bảng này là chỗ để trả lời "hạng mục đó đã làm những gì, và số nằm ở đâu".

| Hạng mục | Tường thuật | Lần chạy liên quan (`runs/`) | Kiểm định (`compare/`) | Đo lẻ (`probes/`) | Kết quả một dòng |
|---|---|---|---|---|---|
| `W1-08` build index | [`tasks/w1-08-build-index.md`](tasks/w1-08-build-index.md) | `baseline` | — | `index-baseline.json` | Index đầu tiên tái lập được, 15.814 chunk |
| `W1-09` DVC | [`tasks/w1-09-dvc.md`](tasks/w1-09-dvc.md) | — | — | — | Corpus phục hồi được từ hash |
| `W1-10` golden nháp | [`tasks/w1-10-goldenset-draft.md`](tasks/w1-10-goldenset-draft.md) | — | — | — | 242 câu sinh bằng DeepSeek |
| `W1-11` review nhãn | [`tasks/w1-11-triage.md`](tasks/w1-11-triage.md) · [`-review`](tasks/w1-11-review.md) · [`-spans`](tasks/w1-11-spans.md) | — | — | — | Nhãn neo theo **span** (`TD-12`), không theo `chunk_id` |
| `W1-13` baseline | [`tasks/w1-foundation.md`](tasks/w1-foundation.md) | `baseline` | — | — | nDCG@10 **0,1621** · `cross_lingual` recall@5 = **0** |
| `TD-11` chunk_size | [`tasks/w2-td11-chunk-size.md`](tasks/w2-td11-chunk-size.md) | `chunk550`, `chunk550nb55` | `cmp-baseline-vs-chunk550`, `cmp-baseline-vs-chunk550nb55`, `cmp-chunk550-vs-chunk550nb55` | `truncation-baseline/-chunk550`, `index-chunk550*` | **Kết quả âm**: truncation 56,9% → 0,4% mà `p = 0,711` |
| `W2-01` BGE-M3 | [`tasks/w2-01-bge-m3.md`](tasks/w2-01-bge-m3.md) | `bgem3` | `cmp-baseline-vs-bgem3` | `truncation-bgem3`, `index-bgem3.json` | nDCG@10 → **0,4442**, 15/15 có ý nghĩa |
| `W2-02` named vectors | [`tasks/w2-02-qdrant-hybrid.md`](tasks/w2-02-qdrant-hybrid.md) | — | — | `index-bgem3.json` | Một collection, hai loại vector; `ensure_collection` **kiểm** schema |
| `W2-03` sparse | [`tasks/w2-03-sparse-retriever.md`](tasks/w2-03-sparse-retriever.md) | `bgem3-sparse` | `cmp-bgem3-vs-bgem3-sparse` | `w2-03-known-item.json` | Kém dense trên golden set, **thắng áp đảo** ở tra mã |
| `W2-04` RRF | [`tasks/w2-04-rrf.md`](tasks/w2-04-rrf.md) | `bgem3-rrf` (k=60), `-k1`, `-k2`, `-k5`, `-k10`, `-c20`, `-c100`, `-k1-c20`, `-k1-c100`, `-w2` | `cmp-bgem3-vs-bgem3-rrf`, `cmp-bgem3-vs-bgem3-rrf-k1` | `w2-04-known-item.json`, `-k1.json` | **`k=60` của bài báo là lựa chọn tệ nhất**; `k=1` thắng nhưng chỉ 3/15 có ý nghĩa |
| `W2-05` reranker | [`tasks/w2-05-reranker.md`](tasks/w2-05-reranker.md) | `bgem3-rr-c20`, `-c50`, `-c100`, `-dense-c50` | `cmp-bgem3-vs-bgem3-rr-c50`, `cmp-bgem3-rr-c20-vs-...c50`, `...c50-vs-...c100`, `cmp-bgem3-rr-dense-c50-vs-bgem3-rr-c50`, `cmp-bgem3-rrf-k1-c20-vs-bgem3-rr-c50` | `w2-05-known-item.json`, `w2-05-rerank-probe.json` | `hit_rate@1` **+22,0 điểm**, 15/15 có ý nghĩa; **DoD 400 ms không đạt** |
| `W2-06` metadata filter | [`tasks/w2-06-metadata-filter.md`](tasks/w2-06-metadata-filter.md) | — | — | `w2-06-filter-probe.json`, `w2-06-backfill-bgem3.json` | Phần thiếu là **đường `fetch`**, không phải date range; lọc **không tốn gì** |
| `W2-07` experiment runner | [`tasks/w2-07-experiment-runner.md`](tasks/w2-07-experiment-runner.md) | `e1-*` (**14 ô**) | — | — | Resume khoá vào **`fingerprint` của ô**; grid tái lập **5** con số đã công bố đúng từng chữ số; MLflow hỏng **hai** lần |
| — (hành chính) | [`tasks/rename-workspace.md`](tasks/rename-workspace.md) | — | — | — | Đổi tên repo → `RAG-Chatbot` |

### Cách đọc tên một lần chạy

`bgem3-rr-dense-c50` = `bgem3` (index/embedding) + `rr` (reranked) + `dense`
(nhánh nền) + `c50` (`rerank_candidates`). Cấu hình **đầy đủ** luôn nằm trong
`runs/<run>-retrieval.json` ở khoá `config.branch_options` — tên chỉ là nhãn cho
người đọc, không phải nguồn sự thật. Nhánh truy hồi **không** thuộc
`IndexConfig.fingerprint` (nó không quyết định vector nào được ghi), nên nhiều
`run` khác nhau dùng chung một index.

**Tiền tố `e1-` = do `make exp` sinh ra, không phải gõ tay** (`run_prefix` của
`configs/eval/exp-001-retrieval.yaml`). Nó tồn tại vì một lý do cụ thể: ô
`bgem3-sparse` của grid trùng **đúng** tên báo cáo tiêu đề `W2-03` đang nằm ở đây,
và preflight của `W2-07` chặn lại ở lần `--dry-run` đầu tiên. Hai không gian tên
tách nhau: `e1-*` thuộc một grid có state ở `.cache/experiments/`, phần còn lại là
lần chạy tay.

⚠️ Tên ô grid **chỉ mang những chiều biến thiên** trong khối của nó, nên
`e1-rr-bgem3-reranked-ondense-rc20` không hiện `rerank_dtype=float16` dù ô đó có
ghim nó. Giá trị ghim vẫn nằm đủ trong `config.branch_options` — đọc file, đừng
đọc tên.

---

## Công cụ nào ghi vào đâu

Chạy lại thì file rơi đúng chỗ — không cần truyền đường dẫn:

| Lệnh | Ghi ra |
|---|---|
| `make eval-retrieval` / `make eval-rerank` | `runs/<RUN>-retrieval.{json,md}` + `runs/<RUN>-per-query.jsonl` |
| `make eval-compare BASE=x CAND=y` | `compare/cmp-x-vs-y.md` (đọc `runs/`) |
| `make index` | `probes/index-<BUNDLE>.json` |
| `make truncation` | `probes/truncation-<BUNDLE>.json` |
| `make known-item` | `probes/w2-03-known-item.json` |
| `make rerank-probe` | `probes/w2-05-rerank-probe.json` |
| `make filter-probe` | `probes/w2-06-filter-probe.json` |
| `make exp` | `runs/<run_prefix>-*` × 3 file mỗi ô + `.cache/experiments/<name>.json` + MLflow |
| `make exp-dry` | — (chỉ in bảng grid + preflight) |
| `make exp-backfill` | MLflow, **đọc từ** `runs/` — không chạy eval |
| `make backfill-payload` | `probes/w2-06-backfill-<BUNDLE>.json` |
| `make goldenset-anchor` / `-freeze` | `goldenset/goldenset-anchor.json` / `-v1.json` |
| `python scripts/category_compare.py` | *không ghi file* — in ra stdout, xem cảnh báo dưới |

Mặc định trong code cũng trỏ vào đây: `retrieval_eval.py --out-dir` và
`compare.py --dir` đều là `plans/reports/runs`.

---

## Chỗ còn hổng — đọc trước khi tin một con số

**✅ `probes/w2-05-rerank-probe.json` — chạy lại 2026-08-21, khớp.** File này từng
không tồn tại: `make rerank-probe` ghi ra đường dẫn đó, nhưng ở `W2-05` tôi chạy
script không truyền `--report`, nên **trần vùng phủ `hit_rate@50` = 0,7799** — con
số chặn trên mọi phát biểu khác của hạng mục — chỉ có ở dạng bảng viết tay. Đã chạy
lại: trần khớp **tới chữ số cuối** (`0.7799043062200957`), cả 5 cột; truncation
khớp tuyệt đối. Ba nhóm đó là hàm thuần của (index, golden set, tham số) nên phải
khớp — và phép đo lại đáng giá chính vì nó *có thể* đã lệch.

Hai thứ **đổi** khi chạy lại, cả hai đã ghi vào
[`tasks/w2-05-reranker.md`](tasks/w2-05-reranker.md):

* **§4 thiếu nhãn dtype.** Logit ra khác ở chữ số thấp (−10,875 vs −10,873) vì bản
  cũ đo ở **fp32** còn mặc định phục vụ là **fp16**. Kiểm bằng
  `--dtype float32`: tái lập đúng cả ba số cũ. Kết luận (0,0% bão hoà) không đổi ở
  cả hai độ chính xác. Bản thân file JSON *có* ghi dtype (`reranker` =
  `...@cuda:L512:float16`) — chỗ hỏng là bảng viết tay.
* **Ngân sách 400 ms mua pool 37, không phải 38.** Bản cũ tính bằng hằng số đã làm
  tròn; tính từ ba điểm đo được thì 37,7 → làm tròn xuống. Độ trễ pool 50 đo lại
  529,3 ms (n=90) vs 524,4 ms cũ, tức "524 ms" nên đọc là **520–530 ms**.

**⚠️ `compare/cmp-baseline-vs-bgem3.md` sinh lại ngày 2026-08-21, không phải khi
làm `W2-01`.** Lệnh so sánh *đã* chạy ở `W2-01` (số nằm trong report) nhưng file
output không được commit; phát hiện khi chia folder. Bản sinh lại **khớp từng chữ
số** với bảng trong `tasks/w2-01-bge-m3.md` và với `CHECKLIST.md`. Kèm một cảnh
báo thật của công cụ: `baseline-per-query.jsonl` **không có `relevant_digest`**
(nó có trước `W2-03`), nên với riêng cặp này cái chốt "hai bên dùng cùng bộ nhãn"
không kiểm được bằng băm — chỉ suy được gián tiếp qua `n_relevant_mean` 1,3828
bằng nhau ở cả hai lần chạy.

**⚠️ Toàn bộ `compare/` được sinh lại ngày 2026-08-21.** Không phải để dọn dẹp:
các file cũ có **trước** bản sửa `precision@1` ở `W2-05`, nên chúng vẫn đi đường
bootstrap cho một metric nhị phân — tức cùng một con số nhận hai kết luận khác
nhau tuỳ file được sinh lúc nào. Sinh lại chỉ cần `runs/*-per-query.jsonl`, không
cần GPU/Qdrant, và `compare.py` có seed cố định nên tái lập được. Mọi kết luận
(`khác biệt thật` / `trong ngưỡng nhiễu`) **không đổi** với bất kỳ dòng nào.

Một chỗ khác đi nhiều hơn một dòng: `cmp-chunk550-vs-chunk550nb55.md` trước có
**7** metric, giờ có **15**. Không có gì bị che — hồi `TD-11` tôi truyền `--metrics`
hẹp, chắc là bê nguyên lệnh của `baseline-vs-chunk550` (nơi nhãn *thật sự* khác:
1,3828 vs 1,9617). Nhưng `chunk550` và `chunk550nb55` chỉ khác
`neighbor_context_chars`, không khác biên chunk, nên `n_relevant_mean` **bằng
nhau** (1,9617) và cả 15 metric đều so được. Cả 15 đều cho "trong ngưỡng nhiễu",
đúng kết luận `TD-11` đã ghi. Guard vẫn chạy đúng ở chỗ cần: xem
`cmp-baseline-vs-chunk550.md`, sáu metric có mẫu số là số nhãn bị đánh **KHÔNG SO
ĐƯỢC** kèm lý do.

**⚠️ `runs/e1-*` (14 ô) KHÔNG phải `W2-08`.** DoD `W2-08` đòi `p`/CI **cho từng
dòng**, tức `compare.py`, và chưa chạy trên bộ này. 14 ô đó là *bằng chứng runner
chạy đúng* (`W2-07`), không phải một kết luận: chênh 0,6736 vs 0,6481 trên 209 câu
là **5 câu**, và đọc nó như kết luận là lặp lại đúng lỗi `TD-11`.

**⚠️ `e1-chunk550-dense` không so được recall/nDCG/MAP với 13 ô còn lại** —
`n_relevant_mean` **1,9617** vs **1,3828** (`G2`). Log của grid nói cùng chuyện:
`chunk550` đổi **209/209** nhãn, baseline và bgem3 đổi 9.

**⚠️ `mlflow.db` không nằm trong git** — nó là một *view*, dựng lại được bằng
`make exp-backfill` từ chính `runs/`. Nếu một con số chỉ có trên MLflow thì nó
không tái lập được từ repo, và `W2-09` không được dựa vào chỗ đó.

**⚠️ `scripts/category_compare.py` là bản tạm.** Nó lọc theo `category` rồi gọi
lại `mcnemar_exact`/`paired_bootstrap` của `compare.py`, vì `compare.py` **chưa
có** `--category`. Kiểm định theo category là điều kiện của DoD `W2-09`, và chính
việc thiếu nó đã để một mức tụt có ý nghĩa của `W2-04` đi qua không ai thấy (xem
[`tasks/w2-05-reranker.md`](tasks/w2-05-reranker.md) §6.4b). Khi `compare.py` có
`--category` thì **xoá script đó** — hai công cụ trả lời cùng một câu hỏi là hai
câu trả lời khác nhau đang chờ xảy ra.

---

## Quy tắc khi thêm file mới

1. **Tường thuật thì vào `tasks/`, tên `w<tuần>-<số>-<chủ đề>.md`.** Report của
   một hạng mục chưa làm cũng đã được `CHECKLIST.md` trỏ tới sẵn theo đường đó
   (`tasks/exp-001-retrieval.md`, `tasks/judge-calibration.md`,
   `tasks/loadtest.md`, `tasks/security-w4.md`, `tasks/vram-budget.md`…).
2. **Không sửa tay file ở `runs/`, `compare/`, `probes/`, `goldenset/`.** Cần đính
   chính thì viết vào `tasks/` — đã có tiền lệ: `tasks/w2-03-sparse-retriever.md`
   §8 giữ nguyên khối sai và đánh dấu "Nội dung gốc (SAI)" thay vì xoá, vì con số
   sai và lý do nó sai đều đáng đọc.
3. **Ba file của một lần chạy phải ở cùng chỗ và cùng prefix.** `compare.py` phân
   giải tên run thành `runs/<tên>-per-query.jsonl`; tách chúng ra là làm hỏng cách
   nó tìm file.
4. **Số vào `CHECKLIST.md` phải dẫn được về một file ở đây.** DoD của `W6-03` là
   "không có số nào không truy được về `reports/`" — và bảng dashboard đã sai ba
   lần vì bỏ bước đó (xem `CHECKLIST.md` §1).
