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
| [`compare/`](compare/) | `cmp-<base>-vs-<cand>.md` (một cặp) · `by<chiều>-*.md` (chia nhóm, có hiệu chỉnh) · `ablation-*.md` (**N ô** + tập tương đương) · `contrast-<chiều>-*.md` (**nhóm nào cải thiện nhiều nhất**) | `pipeline.eval.compare`, `pipeline.eval.ablation`, `pipeline.eval.contrast` | ❌ |
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
| `W2-04` RRF | [`tasks/w2-04-rrf.md`](tasks/w2-04-rrf.md) | `bgem3-rrf` (k=60), `-k1`, `-k2`, `-k5`, `-k10`, `-c20`, `-c100`, `-k1-c20`, `-k1-c100`, `-w2` | `cmp-bgem3-vs-bgem3-rrf`, `cmp-bgem3-vs-bgem3-rrf-k1` | `w2-04-known-item.json`, `-k1.json` | **`k=60` của bài báo là lựa chọn tệ nhất**; `k=1` thắng nhưng chỉ **2/15** có ý nghĩa (sửa từ 3/15 ở `W2-08`) |
| `W2-05` reranker | [`tasks/w2-05-reranker.md`](tasks/w2-05-reranker.md) | `bgem3-rr-c20`, `-c50`, `-c100`, `-dense-c50` | `cmp-bgem3-vs-bgem3-rr-c50`, `cmp-bgem3-rr-c20-vs-...c50`, `...c50-vs-...c100`, `cmp-bgem3-rr-dense-c50-vs-bgem3-rr-c50`, `cmp-bgem3-rrf-k1-c20-vs-bgem3-rr-c50` | `w2-05-known-item.json`, `w2-05-rerank-probe.json` | `hit_rate@1` **+22,0 điểm**, 15/15 có ý nghĩa; **DoD 400 ms không đạt** |
| `W2-06` metadata filter | [`tasks/w2-06-metadata-filter.md`](tasks/w2-06-metadata-filter.md) | — | — | `w2-06-filter-probe.json`, `w2-06-backfill-bgem3.json` | Phần thiếu là **đường `fetch`**, không phải date range; lọc **không tốn gì** |
| `W2-07` experiment runner | [`tasks/w2-07-experiment-runner.md`](tasks/w2-07-experiment-runner.md) | `e1-*` (**14 ô**) | — | — | Resume khoá vào **`fingerprint` của ô**; grid tái lập **5** con số đã công bố đúng từng chữ số; MLflow hỏng **hai** lần |
| `W2-08-prep` `--category`/`--lang` | [`tasks/w2-08-prep-compare-groups.md`](tasks/w2-08-prep-compare-groups.md) | — | `bycategory-*`, `bylang-*` | — | Phát hiện `cross_lingual` của `W2-04` **sống sót** hiệu chỉnh 90 phép kiểm, nhưng **cả ba dẫn chứng đã công bố** là `p` không có lực |
| `W2-08` ablation | [`tasks/w2-08-ablation.md`](tasks/w2-08-ablation.md) | `e1-*` (**14 ô**) | `ablation-exp-001-ndcg` | — | **Cấu hình thắng là một TẬP, không một dòng**; và người thắng từng do **6 mẫu lại trên 10.000** quyết định |
| `W2-09` tổng kết exp-001 | [`tasks/exp-001-retrieval.md`](tasks/exp-001-retrieval.md) | `e1-*` | `cmp-e1-*` · `bycategory-e1-*` · `contrast-category-exp-001` · `contrast-lang-exp-001` | — | **15/15 metric**, `hit_rate@5` **0↔120 câu**; và **câu "category nào cải thiện nhiều nhất" KHÔNG có câu trả lời** — cả 6 nhóm hoà |
| `W3-01` Docling loader | [`tasks/w3-01-docling-loader.md`](tasks/w3-01-docling-loader.md) | — | — | — | 6 định dạng, cả hai vế DoD đạt; và **chèn parser vào giữa byte và `Document.content` giết sạch golden set — 0/280 span sống sót** (`TD-22`) |
| `W3-02` OCR fallback | [`tasks/w3-02-ocr-fallback.md`](tasks/w3-02-ocr-fallback.md) | — | — | [`probes/td-23-easyocr.json`](probes/td-23-easyocr.json) | Phát hiện scan bằng mật độ text layer (ngưỡng đo từ PDF World Bank thật); ~~máy OCR trả rác cho tiếng Việt nên loader từ chối~~ ⚠️ **§10 (2026-09-04): phép đo §3 KHÔNG hợp lệ** — fixture render dấu thành ô ☒ (font Aileron không có glyph tiếng Việt). Đo lại trên ảnh hợp lệ: RapidOCR vẫn hỏng `vi`, **EasyOCR giữ dấu 8/8** → `vi` mở qua EasyOCR, chưa đo vẫn từ chối (`TD-23` mở một nửa) |
| `W3-03` Structure chunker | [`tasks/w3-03-structure-chunker.md`](tasks/w3-03-structure-chunker.md) | — | — | — | `section_path` đúng trên 1061 chunk của hai báo cáo thật (0 nói dối); và **PDF thật chỉ cho MỘT cấp heading**, **corpus 0/60 tài liệu có cấu trúc** (`TD-24`) |
| `W3-08` Ingestion worker | [`tasks/w3-08-ingest-worker.md`](tasks/w3-08-ingest-worker.md) | — | — | `w3-08-ingest-latency.json` | `POST` **p95 4,26 ms** (trần 200 ms, dư 47×); ⭐⭐ **`max_tries` không thử lại `Exception` thường** — arq chỉ thử lại `Retry`/`CancelledError`, nên phân loại lỗi chuyển vào `is_transient`; ⚠️ `doc_ids` suýt thành endpoint **xoá** 2/3 index (`TD-29`) |
| `W3-07` Re-index tăng dần | [`tasks/w3-07-incremental-reindex.md`](tasks/w3-07-incremental-reindex.md) | — | — | `w3-07-incremental.json` | Sửa một dòng ở tài liệu 1.082 chunk → embed lại **6** (nhanh hơn **179,3×**, `G3` đạt); **Qdrant đã là cache vector** nên không dựng cache mới; ⭐⭐ thiệt hại bị chặn bởi **khoảng cách tới `

` kế tiếp** — cùng văn bản, thêm xuống dòng đoạn: 2,0% → 98,0% mượn lại (`TD-28`) |
| `TD-22` Ghim parse | [`tasks/td-22-parse-pin.md`](tasks/td-22-parse-pin.md) | — | — | `td-22-parse-pin.json` | Manifest ghim `text_sha256` + `parse_fingerprint`; ⭐⭐ vân tay `W3-01` ghim **tên gói ô dù** (`export_to_markdown` nằm ở `docling-core`, thả tự do `>=2.91,<3.0`) và **trọng số bố cục tải theo nhánh `main`** (`TD-27`); chép nguyên byte sang `.md` → sha256 trùng khít, **−21,6% văn bản** |
| `W3-05` Parent-child | [`tasks/w3-05-parent-child.md`](tasks/w3-05-parent-child.md) | — | — | `w3-05-parent-child-pc{256,128}-k{3,5,10,20}.json`, `index-pc256.json`, `index-pc128.json` | **Parent không phải một point** (là *tập* các child, nên 0 thay đổi payload/filter/backfill); một **Protocol khớp với method KHÔNG tồn tại** mà 27 test vẫn xanh; và **độ nở ngữ cảnh là chỉ số đánh lừa** — chia đôi child làm nó gấp đôi trong khi prompt không đổi (`TD-26`) |
| `W3-06` Token sizing | [`tasks/w3-06-token-sizing.md`](tasks/w3-06-token-sizing.md) | — | — | — | DoD **đã đạt sẵn** với BGE-M3 (0/15.814, dư 11,2×); giá trị thật là **san bằng kích thước chunk giữa hai ngôn ngữ** (PhoBERT −22,3% → +2,4%), và **ký tự không phải đơn vị mang đi được** (`TD-25`) |
| `W4-01` RagBundle | [`tasks/w4-01-rag-bundle.md`](tasks/w4-01-rag-bundle.md) | — | — | — | Artifact có version + checksum nối hai plane; builder **từ chối đóng gói chéo** (eval của index này + config của index kia) |
| `W4-02` hot-swap bundle | [`tasks/w4-02-bundle-reload.md`](tasks/w4-02-bundle-reload.md) | — | — | — | Đổi bundle không restart; ảnh chụp **một lần** cho cả lượt, rollback một bước. Đo thật ở `TD-37`: 0.2.0 → 0.1.0 hết **451 ms** |
| `W4-03` khung FastAPI | [`tasks/w4-03-fastapi-skeleton.md`](tasks/w4-03-fastapi-skeleton.md) | — | — | — | `/health` vs `/ready` tách vai; ⭐ lần chạy **thật** tìm ra lỗi mà 45 test không thấy, và `test_a_hung_dependency_does_not_hang_ready` đặt đồng hồ **sai chỗ** |
| `W4-04` auth + hạn mức | [`tasks/w4-04-auth-ratelimit.md`](tasks/w4-04-auth-ratelimit.md) | — | — | — | Middleware chứ không `Depends` (quên = **chặn**, không phải mở); `tenant_filter()` **từ chối** filter trỏ sang tenant khác. Hạn mức đếm trong tiến trình (`TD-39`) |
| `W4-05` Postgres + RLS | [`tasks/w4-05-postgres-alembic.md`](tasks/w4-05-postgres-alembic.md) | — | — | — | ⭐⭐ **`FORCE ROW LEVEL SECURITY`**: chủ bảng được miễn policy, nên `ENABLE` một mình là trang trí trong khi `pg_tables.rowsecurity` vẫn báo `true` |
| `W4-12` Guardrails | [`tasks/security-w4.md`](tasks/security-w4.md) | 165 lượt gọi thật ($0,10) | tiêm 9: 8 đỏ + 1 tương đương | `probes/w4-12-*.json` (6 file) | ⭐⭐ k=1 trước model không tất định không phải phép đo (1/11 → 10/17 khi k=6); chữ trong prompt làm gần hết việc, nonce giá trị đo được = 0; dương tính giả 2/20.424 vs 2.515 nếu dùng luật một-từ-khoá |
| `W4-11` Prompt registry | [`tasks/w4-11-prompt-registry.md`](tasks/w4-11-prompt-registry.md) | — | — | — | ⭐ "Đổi prompt = tăng version" là cơ chế: loader từ chối nội dung chưa stamp; namespace cache mang version prompt; meta khai `prompt` theo route; M2 sống sót chỉ ra test tự chiếu qua chính hàm băm bị tiêm |
| `W4-10` Semantic cache | [`tasks/w4-10-semantic-cache.md`](tasks/w4-10-semantic-cache.md) | — | — | [`probes/w4-10-cosine-threshold.json`](probes/w4-10-cosine-threshold.json) + [`probes/w4-10-hitrate.json`](probes/w4-10-hitrate.json) | ⭐⭐ Không tồn tại ngưỡng tách paraphrase khỏi câu-đổi-đáp-án (hai phân bố chồng nhau, p50 0,8717 vs 0,8659) → ngưỡng 0,96 + hàng rào chữ số; 100 query Zipf hit 77/100, bẫy 0/10 |
| `W4-09` Citation verification | [`tasks/w4-09-citation-verify.md`](tasks/w4-09-citation-verify.md) | — | — | [`probes/w4-09-citations-real.json`](probes/w4-09-citations-real.json) | ⭐⭐ Khung `citations` = cái model tuyên bố đã dùng SAU đối chiếu (khác khung `sources`); lần chạy thật thứ ba cho ngay một quote lai ghép (`verified: false`); holdback giữ block khỏi delta, bất biến kiểm bằng cắt mọi vị trí |
| `W4-06` `POST /chat` SSE | [`tasks/w4-06-chat-sse.md`](tasks/w4-06-chat-sse.md) | — | — | — | ⭐⭐ Đường phân giới "còn trả được HTTP status" vs "chỉ còn khung SSE" là một đường **thật** trong mã; lần chạy thật đầu trả **0 ký tự** (1024/1024 token vào chuỗi suy luận) và tìm ra `TD-40` |
| `W4-07` hiểu câu hỏi | [`tasks/w4-07-query-understanding.md`](tasks/w4-07-query-understanding.md) | — | — | `w4-07-language-directive.json`, `w4-07-real-run.json` | ⭐⭐ Chỉ thị ngôn ngữ **8/8 → 0/8** (tỉ lệ nền là *tất cả*); ⭐⭐ tiêm lỗi phơi ra `"thầy cô"`/`"chị em"` bị xếp là chào hỏi ⇒ **không truy hồi**; ⭐⭐ lần chạy thật **đảo ngược** quyết định "model xem câu gốc". 27 case gán nhãn tay, chỉ **7 held-out (6/7)** là bằng chứng |
| `W4-08` LLM Router | [`tasks/w4-08-llm-router.md`](tasks/w4-08-llm-router.md) | — | — | `w4-08-router-real.json` | ⭐⭐ Chuyển nhà cung cấp **giữa stream** nối hai câu trả lời khác nhau — ranh giới là mẩu đầu tiên, có hai test kẹp lấy; ⭐ ba loại hỏng ba cách đối xử (`PermanentLLMError` mới); tiêm lỗi 12 phép, **2 sống sót và cả hai là lỗ thật**; chạy thật DeepSeek → GLM |
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
| `make loader-fixtures` | `tests/fixtures/loaders/*` (6 file, idempotent) |
| `make loader-probe` | — (in ra stdout: đồng nhất byte · dòng còn nguyên · span golden sống) |
| `make scan-probe SCAN=…` | — (in ra stdout: mật độ text layer từng trang) |
| `make exp` | `runs/<run_prefix>-*` × 3 file mỗi ô + `.cache/experiments/<name>.json` + MLflow |
| `make exp-dry` | — (chỉ in bảng grid + preflight) |
| `make exp-backfill` | MLflow, **đọc từ** `runs/` — không chạy eval |
| `make backfill-payload` | `probes/w2-06-backfill-<BUNDLE>.json` |
| `make goldenset-anchor` / `-freeze` | `goldenset/goldenset-anchor.json` / `-v1.json` |
| `make eval-compare-by` | `compare/by<BY>-<BASE>-vs-<CAND>.md` (có hiệu chỉnh Bonferroni) |
| `make eval-compare-subset` | *không ghi file* — in ra stdout (một nhóm nêu trước, không hiệu chỉnh) |
| `make ablation` | `compare/ablation-exp-001-<RANK_SLUG>.md` — **N ô** kèm tập tương đương (hiệu chỉnh trên `(số ô − 1) × số metric`) |
| `make contrast` | `compare/contrast-<BY>-exp-001.md` — **nhóm nào cải thiện nhiều nhất**, bootstrap **không cặp** (hiệu chỉnh trên `số nhóm − 1`) |

Mặc định trong code cũng trỏ vào đây: `retrieval_eval.py --out-dir` và
`compare.py --dir` đều là `plans/reports/runs`.

---

## Chỗ còn hổng — đọc trước khi tin một con số

⚠️ **Bảng trên thiếu toàn bộ `W4` cho tới 2026-09-04.** Bảy hạng mục `W4-01`…`W4-07`
có báo cáo đầy đủ trong `tasks/` từ 03/09 nhưng không có hàng nào ở đây, nên ai vào
bản đồ này để review đường serving sẽ kết luận là nó chưa được làm. Đã bổ sung cùng
lượt với `W4-07`. Bài học nhỏ và lặp lại được: **một bản đồ không được cập nhật cùng
lúc với thứ nó chỉ đường thì nó nói dối theo hướng khó phát hiện nhất** — im lặng.

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

**⚠️ Toàn bộ `compare/` được sinh lại lần nữa ngày 2026-08-22 (`W2-08`).** Ba cờ
mới — `TRÁI CHIỀU`, `TRÙNG KHỚP`, `mc_unstable` — đổi kết luận **10 hàng trong 6/14
file**, và kiểm từng hàng thì cả 10 đều đúng. Hai chốt kiểm soát giữ nguyên đúng
từng chữ số: `cmp-baseline-vs-bgem3` vẫn **15/15** và `cmp-bgem3-vs-bgem3-rr-c50`
vẫn **15/15**. Bản đầu của luật dùng bước lưới `1/n` và làm **13/14** file đổi kết
luận — phần lớn là dương giả trên `precision@k`/`recall@k`, vì `1/n` chỉ đúng cho
metric nhị phân. Chi tiết: `tasks/w2-08-ablation.md` §9.

**⚠️ Và lần sinh lại trước đó, ngày 2026-08-21.** Không phải để dọn dẹp:
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

**✅ `runs/e1-*` (14 ô) đã thành `W2-08` ngày 2026-08-22** — `p`/CI từng dòng nằm
ở [`compare/ablation-exp-001-ndcg.md`](compare/ablation-exp-001-ndcg.md), sinh bằng
`make ablation`. Cảnh báo cũ ở chỗ này ("chênh 0,6736 vs 0,6481 là **5 câu**, đọc nó
như kết luận là lặp lại lỗi `TD-11`") **hoá ra đúng**: 5 câu đó là `ndcg@10`
`TRÁI CHIỀU` (đếm câu +10/−11) và ở `B = 50.000` mẫu lại thì khoảng tin cậy **chứa
0**.

**⚠️ `e1-chunk550-dense` không so được metric NÀO với 13 ô còn lại** — không chỉ
`recall`/`nDCG`/`MAP`. `n_relevant_mean` **1,9617** vs **1,3828** (`G2`) là phần dễ
thấy, nhưng phần chặt hơn là **tập** nhãn: `chunk550` đổi **209/209** nhãn (baseline
và bgem3 đổi 9), nên hàng rào băm `relevant_digest` của `W2-03` từ chối cả 15 metric.
Chỉ dẫn của `G2` ("đổi `chunk_size` thì dùng `hit_rate@k`/MRR") **không thực hiện
được** với công cụ hiện tại → `TD-20`.

**⚠️ `mlflow.db` không nằm trong git** — nó là một *view*, dựng lại được bằng
`make exp-backfill` từ chính `runs/`. Nếu một con số chỉ có trên MLflow thì nó
không tái lập được từ repo, và `W2-09` không được dựa vào chỗ đó.

**✅ `scripts/category_compare.py` đã bị xoá** ở `W2-08-prep` — `compare.py` giờ có
`--by category|lang` (quét mọi nhóm, **có** hiệu chỉnh Bonferroni) và
`--category X`/`--lang Y` (một nhóm nêu trước, **không** hiệu chỉnh). Hai loại suy
luận khác nhau, và CLI từ chối dùng cả hai cùng lúc.

**⚠️ Đọc bảng `bycategory-*`/`bylang-*` thì phải biết ba nhãn không phải con số:**
`KHÔNG ĐỦ LỰC` = phép kiểm **không thể** đạt ý nghĩa (trần `p` của McNemar là
`2/2ⁿ`; `table_lookup` có 4 câu nên **vĩnh viễn** không đo được). `KHÔNG KẾT LUẬN` =
biên CI **đúng bằng 0**, tức việc khoảng chứa 0 phụ thuộc một bước lưới `1/n`.
`KHÔNG SO ĐƯỢC` = phân bố nhãn hai bên khác nhau (`G2`). Không cái nào nghĩa là
"không có khác biệt".

**⚠️ Ba dẫn chứng trong `CHECKLIST.md`/`w2-05-reranker.md` đã được sửa** ở
`W2-08-prep`: kết luận đúng, nhưng `recall@5` CI95 của `cross_lingual`, `hit_rate@5`
`4↔0`, và "reranker vá lại (1↔0, `p`=1,000)" đều là **`p` không có lực trích cạnh
một CI có nghĩa**. Xem `tasks/w2-08-prep-compare-groups.md` §5.

**⚠️ Bảng một-cặp (`cmp-*.md`) KHÔNG hiệu chỉnh cho 15 metric** — kỳ vọng 0,75 kết
quả dương giả mỗi bảng. Lựa chọn có chủ đích (sửa sẽ viết lại kết luận `W2-01`…
`W2-07`), nhưng là giới hạn thật; quyết định thuộc `W2-09`.

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
