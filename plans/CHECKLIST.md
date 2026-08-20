# CHECKLIST — RAG Platform Upgrade

> **Đây là nguồn sự thật duy nhất về tiến độ.** Plan kỹ thuật: [`2026-08-14-rag-upgrade-proposal.md`](2026-08-14-rag-upgrade-proposal.md)
> Cập nhật lần cuối: 2026-08-20 · Trạng thái tổng: **tuần 1: 13/13 task xong · `golden_v1` 242 câu (review bằng MODEL, không phải người) · baseline recall@5 = 0,1746 · gate `G1` 4/4 ⚠️ với một điều kiện · sẵn sàng vào `W2`**
> **W2 đang chạy — `TD-11` + `W2-01`…`W2-03` xong (3/9).** `TD-11`: hạ `chunk_size` đưa truncation 56,9% → 0,4% mà **không cải thiện gì đo được** (`p = 0,711`) — giả định của nợ đó bị phản chứng. `W2-01`: BGE-M3 (giữ `chunk_size=1000`) đưa **nDCG@10 0,1621 → 0,4442** và `cross_lingual` recall@5 **0 → 0,3023**, cả 15 metric đều có ý nghĩa thống kê. ⚠️ Mức tăng đó là của **model**, KHÔNG phải của việc hết truncation — `TD-11` đã đo riêng phần đó và nó cho `p = 0,711`. `W2-02`: `rag_bgem3` giờ mang cả named vector `dense` và `sparse` trong một collection, và `ensure_collection` kiểm tra schema thay vì tin. `W2-03`: sparse đã đo được — **kém dense trên golden set** (nDCG@10 0,4442 → 0,3733) nhưng **thắng áp đảo ở tra mã tài liệu** (hit@10 0,0784 → 0,5098, `p = 4,8e-07`, 22↔**0**). Con số đáng nhất: **trần của RRF là `hit_rate@10` 0,7033** vs dense 0,6268. Việc tiếp theo: `W2-04`. Xem §4 và `reports/w2-td11-chunk-size.md`, `w2-01-bge-m3.md`, `w2-02-qdrant-hybrid.md`, `w2-03-sparse-retriever.md`
> Nhật ký phiên làm việc (để nối tiếp khi ngắt giữa chừng): [`WORKLOG.md`](WORKLOG.md)

---

## 0. Cách dùng file này

### Ký hiệu trạng thái

| Ký hiệu | Nghĩa | Điều kiện |
|---|---|---|
| `[ ]` | **TODO** | Chưa bắt đầu |
| `[~]` | **ĐANG LÀM** | Đã bắt đầu, chưa xong. Ghi ngày bắt đầu ở cột Note |
| `[!]` | **XONG CODE, CHƯA PASS TEST** | Code chạy được nhưng test chưa viết / chưa xanh. **Không được coi là done** |
| `[?]` | **BỊ CHẶN / ĐANG ĐỢI** | Đợi input của bạn, đợi GPU, đợi quyết định. **Bắt buộc** ghi lý do vào §10 |
| `[x]` | **DONE** | Test xanh **và** có Evidence. Không có Evidence thì không được tick |
| `[-]` | **ĐÃ HỦY / GIẢM SCOPE** | Ghi lý do vào §12 Changelog |

### Định nghĩa Done chung (áp dụng cho mọi task code)

Một task chỉ được `[x]` khi đủ **cả 4**:

1. Code đã viết và chạy được.
2. Test tương ứng **xanh** (`pytest`), không skip, không xfail lén.
3. Có **Evidence**: đường dẫn tới report/log/screenshot/commit SHA — thứ mà 2 tuần sau vẫn xác minh lại được.
4. Không phá gate của tuần trước (`make gate` vẫn PASS nếu gate đã tồn tại).

### Ràng buộc môi trường (ảnh hưởng nhiều task)

- **GPU local: RTX 4060 Laptop 8GB.** Chỉ đủ cho `bge-m3` (~2.3GB) + `bge-reranker-v2-m3` (~2.2GB). **Không** đủ để thêm generator LLM.
- **Không có Claude API key.** Chỉ có **DeepSeek API** và **OpenRouter**.
- Phân vai LLM đầy đủ: xem **§3.6** của [plan](2026-08-14-rag-upgrade-proposal.md).
- **Quy tắc cứng #1:** không dùng OpenRouter preset (`@preset/...`) ở bất kỳ đâu trong đường eval — luôn pin slug tường minh, `temp=0`, seed cố định, và log model thực tế đã phục vụ request.
- **Quy tắc cứng #2:** GPU thuê = **RunPod Secure Cloud** (không dùng Community Cloud). Pod chỉ chạy job **GPU-bound, tự chứa** — **không mang API key lên đó**. Mọi thứ chạm API trả phí chạy ở laptop.
- **Quy tắc cứng #3:** corpus phải **công khai, license cho phép redistribute**. Repo public + demo public + máy thuê bên thứ ba = ba kênh công bố dữ liệu.

### Quy tắc cập nhật

- Task mới phát sinh trong lúc làm → thêm vào **§9 Task thêm mới**, cấp ID `NEW-xx`, **không** chèn lẫn vào backlog gốc (để so sánh được scope dự kiến vs scope thật).
- Đổi trạng thái → cập nhật luôn **§1 Dashboard** và ghi 1 dòng vào **§12 Changelog**.
- Gate tuần (`G1`–`G6`) là chốt cứng: **không sang tuần sau khi gate tuần hiện tại chưa PASS**, trừ khi ghi rõ quyết định bỏ qua vào Changelog.

---

## 1. Dashboard tiến độ

| Giai đoạn | Tổng | `[x]` Done | `[!]` Chờ test | `[~]` Đang làm | `[?]` Bị chặn | `[ ]` TODO | Gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| W0 · Chuẩn bị | 8 | 1 | 0 | 2 | 2 | 3 | — |
| W1 · Nền móng + Eval baseline | 13 | 13 | 0 | 0 | 0 | 0 | `G1` 🟡 |
| W2 · Retrieval upgrade | 9 | 3 | 0 | 0 | 0 | 6 | `G2` ⬜ |
| W3 · Ingestion + Chunking | 9 | 0 | 0 | 0 | 0 | 9 | `G3` ⬜ |
| W4 · Serving Plane | 13 | 0 | 0 | 0 | 0 | 13 | `G4` ⬜ |
| W5 · Eval đầy đủ + Observability | 11 | 0 | 0 | 0 | 0 | 11 | `G5` ⬜ |
| W6 · Hoàn thiện & trình bày | 8 | 0 | 0 | 0 | 0 | 8 | `G6` ⬜ |
| **Tổng backlog gốc** | **71** | **17** | **0** | **2** | **2** | **50** | 0/6 |
| §9 Task thêm mới (`NEW-xx`) | 6 | 6 | 0 | 0 | 0 | 0 | — |
| **Tổng cộng** | **77** | **23** | **0** | **2** | **2** | **50** | 0/6 |

Gate: ⬜ chưa chạy · 🟡 đã chạy FAIL · ✅ PASS

> ⚠️ Con số tổng cũ ("73") là lỗi sổ sách tích lại: `NEW-03`…`NEW-06` được thêm vào §9 mà
> không cộng vào tổng. Đã đếm lại trực tiếp từ các mục §2–§8 (**71**) và §9 (**6**).

**Chỉ số chất lượng (điền dần, lấy từ `plans/reports/`)**

| Metric | Baseline (hệ thống hiện tại) | Hiện tại | Mục tiêu | Nguồn |
|---|---|---|---|---|
| Recall@10 | **0,2257** | **0,5813** | ≥ 0.90 | `reports/bgem3-retrieval.md` |
| Recall@5 | **0,1746** | **0,4769** | — | `reports/bgem3-retrieval.md` |
| nDCG@10 | **0,1621** | **0,4442** | ≥ 0.82 | `reports/bgem3-retrieval.md` |
| MRR | **0,1660** | **0,4394** | ≥ 0.75 | `reports/bgem3-retrieval.md` |
| MAP@20 | **0,1349** | **0,3853** | — | `reports/bgem3-retrieval.md` |
| Faithfulness | *chưa đo* | — | ≥ 0.92 | `W5-01` |
| Citation accuracy | *chưa đo* | — | ≥ 0.85 | `W5-02` |
| Refusal correctness | *chưa đo* | — | ≥ 0.85 | `W5-02` (33 câu `unanswerable`) |
| p95 latency (truy hồi) | **32,8 ms** | **45,7 ms** | — | `reports/bgem3-retrieval.md` |
| p95 latency (end-to-end) | *chưa đo* | — | ≤ 3500 ms | `W4-13` / `W6-05` |
| Cost / query | *chưa đo* | — | ≤ $0.005 | `W5-11` |
| Judge–human kappa | *chưa đo* | — | ≥ 0.6 | `W5-04` |

> ✅ Cột **Baseline** đã điền ở `W1-13` (2026-08-20) — điều kiện để bắt đầu W2.
> Năm metric xếp hạng đo trên **209/242 câu** của `golden_v1`; 33 câu `unanswerable` trả
> `None` ở mọi metric xếp hạng và được đo riêng bằng refusal correctness (`W5-02`).
> **p95 là độ trễ truy hồi thuần** (embed câu hỏi + tìm trong Qdrant, sau warm-up), không
> phải end-to-end — chỉ so được với ngưỡng 3500 ms sau `W4-13`.
> ⚠️ Con số baseline cũ "39,9 ms" là lỗi sổ sách; `reports/baseline-retrieval.md` ghi **32,8 ms**.
> Cột **Hiện tại** = `bgem3` (`W2-01`, 2026-08-20), đo trên cùng 209 câu và **cùng nhãn**
> (`n_relevant_mean` 1,3828 ở cả hai lần chạy), nên recall@k/nDCG/MAP so được trực tiếp.
> ⚠️ Đừng đọc "Recall@10 ≥ 0,90" như một khoảng cách lấp được bằng tinh chỉnh retrieval:
> `cross_lingual` (20% tập đo) đang bằng **0** vì model embedding **đơn ngữ** — đổi model
> là đổi hẳn tầng nền, không phải tinh chỉnh tham số.

---

## 2. W0 · Chuẩn bị & quyết định

- [~] `W0-01` **Đọc & phê duyệt plan kỹ thuật** — Note: đang chờ bạn đọc lại sau khi sửa diagram (2026-08-14)
  · DoD: bạn xác nhận đồng ý hoặc yêu cầu sửa · Test: — · Evidence: —
- [?] `W0-02` **Quyết định tên + tạo GitHub repo mới** (giữ nguyên repo cũ đang link trong CV)
  · DoD: repo trống có README + LICENSE + `.gitignore`, branch `main` protected · Test: — · Evidence: URL repo
- [~] `W0-03` ⭐ **Chọn & thu thập corpus song ngữ VI–EN CÔNG KHAI** (50–100 tài liệu)
  · DoD: `data/corpus/` + `corpus_manifest.csv` (path, lang, doc_type, source_url, license); **mọi file đều có nguồn public + license cho phép redistribute** · Test: `pytest tests/unit/test_corpus_manifest.py` — reject dòng thiếu `source_url` hoặc `license` · Evidence: manifest
  · ⚠️ **Cấm dùng tài liệu thật của khách hàng Enigmas.** Repo sẽ public + demo HF Spaces public + một phần chạy trên máy thuê bên thứ ba → cả ba đều là công bố dữ liệu. Rủi ro NDA nặng hơn nhiều so với lộ API key.
  · **Đã chốt (2026-08-14): trộn 3 nguồn**, mỗi nguồn ~30 tài liệu — (a) báo cáo World Bank/ADB về Việt Nam (song song VI–EN, license CC BY), (b) văn bản pháp luật VN có bản dịch chính thức, (c) báo cáo thường niên doanh nghiệp niêm yết HOSE.
  · Lý do trộn: mỗi nguồn phục vụ một nhóm metric khác nhau — (a) prose/cross-lingual, (b) heading sâu → showcase `section_path` của structure-aware chunking, (c) bảng tài chính → test `table_lookup` + OCR. Breakdown theo `doc_type` là điểm bán chính của eval harness.
  · **Tiến độ 2026-08-17 — nguồn (a) XONG: 60 tài liệu World Bank** (40 EN + 20 VI, 15.5 MB, toàn bộ CC BY 3.0 IGO, xuất bản 2000–2026). Tải bằng `uv run python scripts/fetch_corpus.py --config configs/corpus/worldbank_vietnam.yaml`. Manifest: `data/corpus_manifest.csv`. Test: `tests/unit/test_corpus_manifest.py` — **23 case**
  · Lấy bản `.txt` mà World Bank trích sẵn (`txturl`) thay vì PDF → baseline `W1-13` không bị chặn bởi chất lượng Docling (`W3-01`)
  · ⚠️ **ADB trả HTTP 403 cho mọi truy cập tự động** — không viết adapter được. Tài liệu ADB phải tải tay rồi khai báo qua block `seed_list` (đã có mẫu trong config)
  · Còn thiếu: nguồn (b) văn bản pháp luật ~30 và (c) báo cáo thường niên HOSE ~30 — cả hai đều cần bạn chọn tay
- [x] `W0-04` **Chuẩn bị credential & biến môi trường** (`DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, `HF_TOKEN`, DB/Qdrant creds)
  · DoD: file mẫu `.env.example` đủ biến ✅ · file thật nằm trong `.gitignore` ✅ · startup fail-fast báo **tất cả** biến thiếu trong một lần thay vì lần lượt ✅ · Test: `tests/unit/test_settings.py` — **8 case**, gồm `test_secret_not_in_repr` và `test_password_not_in_repr_but_in_dsn` (giá trị bí mật không lọt vào log/repr nhưng vẫn vào được DSN) · Evidence: `reports/w1-foundation.md`
  · Đã dùng thật ở `W1-10` (gọi DeepSeek, $0,5821) và ở `W1-13` (chạy với key **rỗng** để chứng minh eval truy hồi không cần LLM API). Trước đây đánh `[?]` là **nhầm** — đã hết bị chặn từ 2026-08-17, changelog có ghi mà quên đổi ký hiệu
- [ ] `W0-05` **Dựng môi trường RunPod** — Secure Cloud, **RTX 4090 24GB**, Pod On-Demand, + **Network Volume** gắn `/workspace` với `HF_HOME=/workspace/.hf`
  · DoD: pod khởi động lại lần 2 **không phải tải lại trọng số**; chạy thử load `bge-m3` + `bge-reranker-v2-m3` + vLLM `Qwen3-8B`, ghi VRAM & thời gian · Test: — · Evidence: `reports/gpu-env.md`
  · Chỉ 3 task cần GPU thuê: `W3-04`, `W5-11`, fine-tune (tùy chọn). Tổng ~10–20 GPU-hour ≈ vài chục USD trở xuống.
  · Đặt **spending limit** trên tài khoản RunPod ngay từ đầu. Network volume gắn với 1 datacenter → chọn region rồi giữ nguyên.
  · Chuyển sang **Spot** để tiết kiệm **chỉ sau khi** `W3-04` và `W2-07` đã resumable.
- [ ] `W0-06` ⭐ **Đo ngân sách VRAM thật trên RTX 4060 8GB** + xác minh vLLM chạy qua Docker Desktop + WSL2 GPU passthrough
  · DoD: biết chính xác VRAM còn dư sau khi load bge-m3 + reranker; xác nhận được/không thể chạy thêm generator local · Test: `python scripts/vram_budget.py` in ra bảng · Evidence: `reports/vram-budget.md`
  · *Nếu VRAM dư < 3GB (kỳ vọng đúng như vậy) → chốt: serving dùng API generator, vLLM chỉ bật theo phiên trên GPU thuê.*
- [?] `W0-07` **Làm rõ `@preset/my-luna-pro`** — preset này resolve ra model slug nào, params gì
  · DoD: biết slug thật; chọn thêm 1 slug **khác họ DeepSeek** trên OpenRouter làm judge cross-check · Test: gọi 1 request, đọc field `model` trong response · Evidence: ghi vào `reports/llm-providers.md`
  · *Quy tắc: preset chỉ dùng cho demo tương tác, **cấm dùng trong eval** (config phía server có thể đổi ngầm).*
- [ ] `W0-08` ⭐ **Kênh chuyển dữ liệu an toàn cho RunPod + vệ sinh secret**
  · DoD: script `scripts/runpod_job.sh` đẩy corpus lên / kéo kết quả về bằng **HF private dataset repo + fine-grained token scoped 1 repo** (hoặc `runpodctl send/receive`); **không** copy `.env`; không cấu hình git credential trên pod · Test: `pytest tests/security/test_no_secret_in_job_bundle.py` — quét tarball job, fail nếu thấy pattern API key · Evidence: `reports/remote-gpu-hygiene.md`
  · Quy tắc: **pod chỉ chạy job GPU-bound tự chứa, không mang API key** (kể cả RunPod Secrets). SSH bằng keypair riêng cho RunPod. Thu hồi token ngay sau job; terminate pod khi xong.

---

## 3. W1 · Nền móng + Eval baseline

> Mục tiêu tuần: **có con số baseline tái lập bằng 1 lệnh.** Chưa tối ưu gì cả.

- [x] `W1-01` **Monorepo skeleton** — `packages/rag_core`, `pipeline/`, `serving/`, `apps/`, `tests/`; `pyproject.toml` (uv), ruff + mypy + pytest, pre-commit, `Makefile`
  · DoD: `make lint && make test` exit 0 · Test: `make lint`, `make test` · Evidence: `reports/w1-foundation.md` — ruff + mypy strict pass, 144 unit test
  · Đã dựng `packages/rag_core`, `pipeline/`, `serving/`, `apps/`, `infra/`, `configs/`, `tests/{unit,integration,security,e2e}`; uv + ruff + mypy strict + pytest + pre-commit + Makefile
  · ⚠️ **Yêu cầu Python ≥ 3.12** (không phải 3.11): stub numpy dùng cú pháp `type` chỉ có từ 3.12, mypy strict không chạy được với 3.11. Ảnh hưởng việc chọn image ở `W0-05`
- [x] `W1-02` **Pydantic schemas (hợp đồng dữ liệu)** — `Document`, `Chunk`, `RetrievedChunk`, `Citation`, `Answer`, `QueryRequest`
  · DoD: schema có validation + serialize/deserialize round-trip · Test: `tests/unit/test_schemas.py` — **20 case** · Evidence: `reports/w1-foundation.md`
  · `extra="forbid"` toàn bộ (field gõ sai tên bị nuốt = bug âm thầm trong index đã build); `Document`/`Chunk` bất biến; `DocumentMetadata` **bắt buộc** `source_url` + `license` — ràng buộc corpus công khai ép ngay ở tầng schema nên không thể quên khi thêm tài liệu
- [x] `W1-03` **Port `enhanced_chunking.py` → `rag_core/chunking/`** (fixed, semantic, hybrid) sau interface chung `Chunker.chunk(docs) -> list[Chunk]`
  · DoD: 3 strategy dùng chung interface, không import `streamlit` · Test: `tests/unit/test_chunking.py` — **35 case**, có doc rỗng, doc 1 câu, doc ~50 trang (< 5s) · Evidence: `reports/w1-foundation.md`
  · **Bỏ LangChain**: viết lại `RecursiveCharacterTextSplitter` + `SemanticChunker` thuần Python (~150 dòng) → test được từng nhánh, không phụ thuộc `langchain-experimental`
  · ⚠️ **4 sai lệch có chủ ý so với bản POC, ảnh hưởng số baseline `W1-13`** — chi tiết ở `reports/w1-foundation.md`. Quan trọng nhất: `neighbor_context_chars` mặc định **tắt**, config baseline phải đặt lại **100** mới tái lập đúng hệ thống hiện tại
- [x] `W1-04` **Thay cache `pickle` → SQLite content-hash cache** (an toàn, portable, có TTL + eviction)
  · DoD: hit/miss đúng khi đổi content hoặc đổi config; không dùng `pickle.load` · Test: `tests/unit/test_chunk_cache.py` — **16 case**: hit, miss theo content/config/chunker, TTL, LRU eviction, entry hỏng, file DB hỏng · Evidence: `reports/w1-foundation.md`
  · Có test quét AST toàn bộ `rag_core` chặn mọi `import pickle` quay lại
  · **Cache theo từng tài liệu**, không theo cả corpus (bản POC hash chuỗi nối của mọi tài liệu → sửa 1 file là mất sạch). Đây là nền cho `W3-07`
- [x] `W1-05` **`docker-compose.yml`: Qdrant + Postgres + Redis** + `make up` / `make down` + healthcheck
  · DoD: `make up` → cả 3 service healthy < 60s · Test: `tests/integration/test_infra_up.py` — **4 case**, ping bằng **giao thức thật** của từng service (không chỉ mở TCP socket) · Evidence: `reports/w1-foundation.md`
  · Tìm ra 2 lỗi chỉ lộ khi chạy thật: (a) `QDRANT__SERVICE__API_KEY` rỗng vẫn bật xác thực → 401 mọi request; (b) `localhost` trên Windows resolve `::1` trước → **mỗi request Qdrant chậm đúng 2 giây**; đổi sang `127.0.0.1` thì integration suite từ ~5 phút còn 30 giây
- [x] `W1-06` **Embedding provider abstraction** — `EmbeddingProvider` interface + impl HF local; baseline `vietnamese-bi-encoder`
  · DoD: đổi model chỉ qua config, không sửa code; batch + normalize configurable · Test: `tests/unit/test_embedding.py` — **14 case** · Evidence: `reports/w1-foundation.md`
  · Thêm `HashingEmbeddingProvider` để unit test chạy không cần `torch`. Cố ý **không** dùng fake trả vector ngẫu nhiên: test semantic chunking cần tương đồng từ vựng thật, vector ngẫu nhiên sẽ khiến test pass kể cả khi thuật toán sai
- [x] `W1-07` **Dense retriever trên Qdrant** — upsert + search + payload
  · DoD: index chunk, query trả top-k đúng thứ tự score · Test: `tests/integration/test_qdrant_dense.py` — **14 case**: named vector, thứ hạng liên tục, score giảm dần, idempotent, filter theo lang/tenant/doc_type, delete · Evidence: `reports/w1-foundation.md`
  · **Named vector `dense` ngay từ W1** — collection dùng vector vô danh không thêm được named vector mà không build lại toàn bộ index, mà `W2-02` sẽ thêm sparse
  · **Point ID = UUIDv5 sinh từ `chunk_id`** → tính idempotent nằm ở tầng store chứ không ở script build index; `W1-08` thừa hưởng sẵn
- [x] `W1-08` **`pipeline/indexing/build_index.py`** — corpus → chunk → embed → Qdrant collection, idempotent
  · DoD: chạy 2 lần không sinh duplicate; log số doc/chunk/thời gian · Test: `tests/integration/test_build_index.py` — **15 case** + `test_index_config.py` **28 case** + `test_corpus_loader.py` **16 case** · Evidence: `reports/w1-08-build-index.md` + `reports/index-baseline.json`
  · **Index baseline đã build**: 60 tài liệu → **15.814 chunk**, 768 chiều, trên `cuda`, 202s. Chạy lần hai: `index 0 · bỏ qua 60`, count không đổi
  · **Ba tầng idempotent**, point ID xác định của `W1-07` chỉ là tầng 1: (2) nhớ số chunk cũ để xoá phần đuôi thừa khi tài liệu **ngắn lại** — point mồ côi không trùng lặp với gì cả, nó chỉ trỏ tới văn bản không còn tồn tại; (3) so `fingerprint` để chặn trộn hai cấu hình vào một collection
  · **Bắt được 3 lỗi chỉ lộ khi chạy thật**: (a) `HybridChunker` luôn chọn semantic vì cache chia lô thành 1 tài liệu → sửa bằng `Chunker.prepare(n)`; (b) `HybridChunker.name` không phân biệt nhánh → hai lần chạy đọc nhầm cache của nhau; (c) **p95 độ trễ 15.219 ms là thời gian nạp model**, không phải truy hồi → thêm warm-up, p95 còn 98 ms
  · Đo được **hệ số 1.24x**: `neighbor_context_chars=100` của bản POC làm text đem embed phình từ 14,3 lên 17,7 triệu ký tự
  · `IndexConfig` là tiền thân của `RagBundle` (`W4-01`): `fingerprint` băm đúng thứ quyết định vector, **không** gồm `device`/`batch_size` để đổi máy không phải build lại — có test canh cả hai chiều
- [x] `W1-09` **DVC init + version corpus** — `data/corpus` (60 tài liệu, md5 `9aeb1b77…`, 14,7 MiB)
  · DoD: `dvc status` sạch ✅ · remote local hoạt động ✅ · Test: `tests/unit/test_dvc_state.py` **23 case** + phép thử clone sạch chạy thật · Evidence: `.dvc/` committed + `reports/w1-09-dvc.md`
  · **Clone sạch → `dvc pull` → 61 file trong 1,9s**, rồi so sha256 từng tài liệu với manifest: **60/60 khớp**. Đây là bằng chứng mạnh hơn `dvc status`, vì nó chứng minh nội dung DVC phục hồi là byte-identical với thứ tải từ World Bank
  · **Remote KHÔNG nằm trong `.dvc/config`** (file được commit) mà ở `.dvc/config.local`. Một URL `D:/...` commit vào config dùng chung thì mọi clone nhận remote trỏ vào ổ đĩa không tồn tại — cùng loại lỗi với đường dẫn tuyệt đối trong `.venv/*.pth` vừa gặp lúc đổi tên workspace
  · ⚠️ **Làm khác checklist: `data/golden` giữ trong git, không đưa vào DVC.** Golden set là thước đo, thứ cần nhất ở nó là diff đọc được lúc review (ai đổi nhãn câu nào, từ gì sang gì); DVC thay file bằng một hash nên mất đúng thứ đó. 284 KB text không phải lý do để tránh git. Tính tái lập không mất: một commit ghim cả hai. Lý lẽ đầy đủ ở report §3.2
  · Corpus giờ có **hai** cơ chế versioning (sha256/manifest + md5/DVC) và chỗ cả hai đều mù là phép so **số lượng**. `pipeline/corpus/dvc_state.py` canh chỗ đó: thêm file vào `data/corpus/` rồi `dvc add` mà quên manifest thì không lỗi nào nổ ra — build index bỏ qua file lạ, `dvc status` vẫn sạch, `dvc push` vẫn đem nó lên remote
  · Bắt được lúc chạy thật: `load_manifest` trả `[]` cho đường dẫn không tồn tại (chủ ý, `fetch_corpus.py` cần thế ở lần đầu) → gõ sai đường dẫn manifest sẽ báo thành "hai cơ chế lệch nhau", dẫn đi sai hướng. Đã thêm guard riêng
- [x] `W1-10` **Script sinh nháp golden set** — DeepSeek sinh Q + relevant_chunk_ids từ chunk thật, kèm phân loại category
  · DoD: **266 câu nháp** (≥250 ✅) · chi phí **$0,5821** ($0,00219/câu) · khử trùng lặp có · Test: `test_goldenset_gen.py` **35 case** + `test_goldenset_dedupe.py` **19 case** + `test_goldenset_sampling.py` **23 case** + `test_llm_provider.py` **23 case** · Evidence: `reports/w1-10-goldenset-draft.md` + `data/golden/draft_v1.jsonl`
  · Phân bố: factoid 78 · cross_lingual 46 · unanswerable 40 · adversarial 36 · multi_hop 34 · aggregation 28 · table_lookup 4. Ngôn ngữ vi 167 / en 99, trải trên **44/60 tài liệu**
  · ⚠️ **`deepseek-chat` là BÍ DANH**, thực tế phục vụ bởi `deepseek-v4-flash` (xác nhận trên API 2026-08-17; `deepseek-reasoner` cũng vậy). Đây đúng là vấn đề quy tắc cứng #1 nói về preset, chỉ kín đáo hơn. Mặc định dự án đã đổi sang slug thật
  · **Model không được tự viết `chunk_id`** — nó chỉ trả chỉ số đoạn văn, code ánh xạ sang id thật. `quote` được đối chiếu lại với chunk: 16/266 câu không kiểm chứng được → xếp đầu hàng đợi review
  · Ba bộ lọc chất lượng corpus phải thêm (trộn hai cột PDF 27,8% chunk · chú thích biểu đồ · trang bìa) — chi tiết ở report. Chúng **chỉ áp cho việc chọn mẫu**, không áp cho index
  · `table_lookup` chỉ 4 câu là **đúng, không phải lỗi**: corpus `.txt` đã làm phẳng bảng. Nhóm này chờ nguồn (c) HOSE + `W3-01`
- [x] `W1-11` **Review → freeze `golden_v1`** — **242 câu** (≥150 ✅), đủ 7 nhóm, sha256 `f53ad84abea32d3f…`, file read-only
  · ⚠️⚠️ **Review bởi MODEL, không phải người.** `reviewed_by_human=false`, `reviewed_by="model:claude-opus-5"`. `freeze` chỉ đặt `true` khi `--reviewer human` và cảnh báo mỗi lần khác. DeepSeek sinh → Claude review là **cross-model** (không phải tự chấm mình), nhưng **không** tương đương người. Đừng mô tả là "human-verified" ở CV hay phỏng vấn. Chi tiết + giới hạn: `reports/w1-11-review.md` §1
  · **Loại 24/266 câu**, năm loại lỗi: (a) **7 câu `unanswerable` mà corpus THẬT SỰ trả lời được** — lạm phát 2025 (4,5-5%), GDP 2030 (bảng CGE của CCDR = 5,45%), 1.584 thủ tục tiền kiểm, nghèo đa chiều 2022 → tỉ lệ sai 17,5% trong nhóm; (b) 4 câu không tự chứa ("According to the passage", "theo báo cáo này"); (c) 7 câu tra **cấu trúc tài liệu** (số trang, tiêu đề Hộp 3, số hiệu working paper, thư mục tham khảo); (d) 3 câu trùng ý — dedupe Jaccard không bắt vì chỉ *nguồn* trùng, *cách hỏi* khác; (e) 3 câu mơ hồ/vòng tròn
  · Quy trình: **6 phép kiểm máy chạy trên toàn bộ 266 câu** + đọc tay từng câu kèm text bằng chứng thật. `multi_hop`/`aggregation` 60/60 đủ ≥2 span · `cross_lingual` 43/43 thật khác ngôn ngữ · `adversarial` 36/36 có tiền đề sai · grounding **0 lỗi thật**
  · **`quote_unverified` KHÔNG phải dấu hiệu model bịa** mà là lỗi trích xuất PDF: trộn hai cột chèn chữ vào giữa từ (`"bổ sung"`→`"bổ chuyên sung"`), số chú thích chèn giữa câu, gạch nối cuối dòng. 15/16 trích dẫn có thật
  · Phép kiểm máy **sửa lại nhận định mắt thường của tôi** ở 1 câu (số 2.938 tỷ có thật, chỉ nằm ngoài span đã thu hẹp) — nếu tin mắt thì đã loại oan
  · Bắt được lúc chạy thật: **`freeze` làm rơi `relevant_spans`** → `golden_v1` mất hết công neo span của `TD-12`, không triệu chứng nào. Đã sửa + 4 test. Quyết định kèm theo: `edit` có điền `new_relevant_chunk_ids` thì **bỏ span**, vì ánh xạ span ghi đè chunk_id nên giữ span cũ sẽ âm thầm bỏ sửa tay của người review
  · DoD gốc ("người xác nhận") **chưa đạt**. Việc còn lại: người đọc lại 33 câu `unanswerable` + 43 câu `cross_lingual` (hai nhóm loại nhiều nhất = kém tự tin nhất) rồi freeze lại với `--reviewer human` — đủ 7 nhóm: factoid, multi_hop, aggregation, table_lookup, cross_lingual, unanswerable, adversarial
  · **Công cụ xong, đã chạy thật** (`reports/w1-11-triage.md`): `pipeline/goldenset/triage.py` (retriever thật → hàng đợi review) + `freeze.py` (quyết định → `golden_v1` + checksum + read-only). +71 unit test (409 tổng), +5 integration (38 tổng). `make goldenset-triage` chạy 24,3s
  · **Thứ tự đọc đã xếp sẵn** trong `queue_v1.md`: 15 câu `unanswerable_but_retrieved` → 16 câu `quote_unverified` → 2 câu `trivially_easy` → 161 câu `answerable_but_not_retrieved` (mặc định **accept**) → 72 câu không cờ. `gold_chunk_missing` = 0
  · ⚠️ **Bất đối xứng phải giữ đúng.** Câu `unanswerable` mà retriever tự tin = bằng chứng nhãn sai (mệnh đề về corpus bị phản chứng). Câu trả lời được mà retriever trượt **không** phải bằng chứng nhãn sai — đó là thứ eval tồn tại để đo, loại nó đi là tự thổi phồng recall baseline. Nên tín hiệu thứ hai xếp **cuối** hàng đợi, đề xuất `accept`, và có 2 test canh (`TestSignalB`)
  · Ngưỡng nghi ngờ **hiệu chuẩn từ dữ liệu**, không phải hằng số: trung vị điểm top-1 của các câu trả lời được = **0,5797**. Ghim `0.8` là đoán, vì "điểm cao" phụ thuộc model embedding và corpus
  · **Freeze không đoán hộ**: `fix_chunk_ids` mà người review để trống thì báo lỗi, không lấy top-1 của retriever điền vào — làm thế là dạy golden set trả lời đúng theo hệ thống hiện tại. Ba từ vựng tách rời: `suggested_decision` (máy hỏi) / `decision` (người trả lời: accept·reject·edit) / ô trống (chưa review, **không** phải accept)
  · **Nhãn đã neo theo span ký tự** trước khi review, để 6–8 giờ công không hỏng ở bước `W2` đầu tiên (`TD-12`). `make goldenset-anchor`. Nội dung chunk **không đổi một byte** — digest trên corpus thật khớp trước/sau, đó là bất biến số một của lần refactor đó
  · Hai file, không phải một: `queue_v1.md` tối ưu cho **đọc** (toàn văn chunk đã gán + top-3 + trích dẫn, không phải tra cứu qua lại), `decisions_v1.csv` tối ưu cho **ghi**. `write_decisions_template` từ chối ghi đè — mất 6 giờ công review vì chạy lại một lệnh là chuyện không được xảy ra
  · DoD: mỗi câu có `relevant_chunk_ids` đã người xác nhận; file read-only, có checksum · Test: `tests/unit/test_goldenset_schema.py` (schema + phân bố category + không rỗng chunk_ids trừ nhóm unanswerable) · Evidence: `data/golden/golden_v1.jsonl` + `reports/w1-11-review.md` (quy trình + phân bố) + `reports/goldenset-v1.json` (số liệu freeze) + `reports/w1-11-triage.md` + `reports/w1-11-spans.md`
- [x] `W1-12` **`pipeline/eval/retrieval_eval.py`** — Recall@{1,5,10,20}, Precision@k, MRR, nDCG@10, HitRate; breakdown theo category & language
  · DoD: bảng MD + JSON; metric đúng trên fixture có đáp án tính tay · Test: `tests/unit/test_retrieval_metrics.py` (**35 case**) + `tests/unit/test_retrieval_eval.py` (**19 case**) · Evidence: `reports/w1-foundation.md`
  · Kỳ vọng viết bằng công thức tường minh với thứ hạng ghi rõ; cố ý **không** gọi lại hàm đang test để sinh kỳ vọng
  · Có breakdown theo category & language. **Câu `unanswerable` trả `None` chứ không phải `0.0`** — recall trên tập rỗng là không xác định; quy ước thành 0 kéo tụt điểm vô nghĩa, thành 1 thì thổi phồng. Nhóm này đo riêng ở `W5-02`
  · Thiếu `query_id` trong kết quả = truy hồi rỗng (bị chấm 0), không phải bỏ qua — im lặng bỏ qua sẽ làm điểm cao lên một cách sai
  · ✅ Đã nối retriever thật ở `W1-08`: `make eval-retrieval` dùng `--index-config` và truy hồi trực tiếp từ Qdrant. `--retrieved` vẫn giữ để chấm lại kết quả cũ mà không phải chạy lại retrieval
- [x] `W1-13` ⭐ **Baseline đã đo** — 209/242 câu được chấm (33 câu `unanswerable` trả `None`, đo riêng ở `W5-02`)
  · **recall@1 0,0877 · recall@5 0,1746 · recall@10 0,2257 · recall@20 0,2663 · MRR 0,1660 · nDCG@10 0,1621 · MAP@20 0,1349 · HitRate@5 0,2153 · p50 22,5ms / p95 39,9ms**
  · DoD chạy lại 2 lần: **sai số 0,0000%** trên cả 15 metric (không phải "<1%" mà bằng 0 — dense retrieval trên vector đã ghi là phép tính xác định). Evidence: `reports/w1-11-review.md` + `reports/baseline-retrieval.{md,json}`
  · **Theo nhóm — chỗ có thông tin nhất**: factoid 0,3088 · multi_hop 0,2157 · adversarial 0,1618 · aggregation 0,1026 · **cross_lingual 0,0000** · table_lookup 0,0000 (n=4, không suy ra được gì). Theo ngôn ngữ: vi 0,2073 vs en 0,1240
  · **0,17 là thấp, và thấp CÓ LÝ DO đã định lượng**: (1) `cross_lingual` = 43/209 câu tức 20% tập đo, dùng model embedding đơn ngữ → recall@5 bằng 0 là con số đúng, không phải lỗi đo; (2) `TD-11` — 56,8% chunk bị cắt ở 256 token, 15,7% văn bản không tới được vector. Cố ý **không sửa trước khi đo**: baseline phải đo bản POC như nó đang là
  · Nhãn được tính từ **span**, không phải `chunk_id`: `span_resolution` = 209 câu tính lại · 0 câu không khớp · 9 câu đổi nhãn

### `G1` — Gate tuần 1 🟡 (4/4 về mặt kỹ thuật, **1 điều kiện chưa đạt**)

> ⚠️ Bốn hạng mục đều chạy được và có bằng chứng, nhưng golden set là **review bằng
> model**, không phải người. Gate này nên coi là **PASS có điều kiện**: đủ để đi tiếp
> `W2`, chưa đủ để gọi baseline là "human-verified" ở bất kỳ đâu.
>
> **Điều kiện để 🟡 → ✅ là `TD-13`** và chỉ nó: người đọc lại 33 câu `unanswerable` +
> 43 câu `cross_lingual`, rồi `make goldenset-freeze` với `--reviewer human`. Không cần
> làm lại gì khác — đo lại baseline sau đó mất **20 giây**.
- [x] `make eval-retrieval BUNDLE=baseline` chạy được — đường ống thông từ `make corpus` tới `make eval-retrieval`, trên `golden_v1` thật (242 câu)
  · Kiểm chứng bằng golden set giả sinh từ chính chunk trong index (13 câu, **không** commit): `recall@5 0.9167 · p50 32.6ms · p95 97.7ms`. Đây **không phải** baseline — câu hỏi chính là text của chunk nên gần như chắc chắn tìm lại được chính nó
- [x] Retrieval eval chạy **không cần bất kỳ LLM API nào** — đã chạy thật với `DEEPSEEK_API_KEY=""` và `OPENROUTER_API_KEY=""`, kết quả trùng khít lượt có key (sai số 0,0000%)
  · Đã đúng ở mức code: `pipeline/eval/metrics.py` không import provider LLM nào, và `tests/unit/test_architecture_boundaries.py` canh phụ thuộc. Còn thiếu lần chạy thật với API key rỗng — làm cùng `W1-13`
- [x] §1 Dashboard đã điền xong cột **Baseline** — recall@5 0,1746 · MRR 0,1660 · nDCG@10 0,1621 · p95 39,9ms
- [x] Coverage `rag_core/` ≥ 70% → **80%** (chỉ tính unit test; `reports/w1-08-build-index.md`)

---

## 4. W2 · Retrieval upgrade

> ### Đã làm: `TD-11` (2026-08-20) — kết quả **âm**, và nó đổi thứ tự phần còn lại
>
> Thứ tự ban đầu là `TD-11` trước (hạ `chunk_size`) rồi `W2-01` (BGE-M3), để tách cặp số
> trước/sau. Đã chạy: `chunk_size` 1000 → 550 đưa truncation **56,9% → 0,4%** chunk và
> **không cải thiện gì đo được** — McNemar `p = 0,711`, mọi CI bootstrap chứa 0.
>
> Ba thứ rút ra, cả ba đều đổi cách làm phần còn lại của W2:
>
> 1. **Hạ `chunk_size` là đánh đổi, không phải thu hồi.** Baseline bị cắt nhưng mỗi vector
>    vẫn đọc ~950 ký tự; `chunk550` không bị cắt nhưng mỗi vector chỉ đọc 678. Nên **không
>    quét thêm `chunk_size` với `vietnamese-bi-encoder`** — hai điểm đo đã cho thấy chiều.
> 2. **`golden_v1` chỉ phân giải được mức chênh ≥ 6 điểm `hit_rate` (≈28% tương đối).**
>    209 câu, `hit_rate@5` ≈ 0,20. Mọi so sánh mịn hơn thế **phải** có kiểm định, nếu không
>    thì cái thắng chỉ là cái may. Đã thêm `make eval-compare` + `*-per-query.jsonl`.
> 3. **recall@k/nDCG/MAP không so được giữa hai `chunk_size`.** Nhãn neo theo span nên chunk
>    nhỏ hơn làm số nhãn/câu tăng 1,38 → 1,96, và mẫu số của recall là chính con số đó →
>    tụt 29,6% kể cả khi truy hồi y nguyên. `compare.py` tự từ chối những metric này.
>
> ### `W2-01` xong (2026-08-20) — và nó dạy thêm một điều về quy kết nguyên nhân
>
> BGE-M3 giữ `chunk_size=1000`: **nDCG@10 0,1621 → 0,4442**, `cross_lingual` recall@5
> **0 → 0,3023**, cả 15 metric có ý nghĩa. Chi tiết ở `reports/w2-01-bge-m3.md`.
>
> ⚠️ **Nhưng mức tăng đó không phải của việc sửa truncation.** Ba thí nghiệm đặt cạnh nhau:
>
> | | truncation | `hit_rate@5` | |
> |---|---:|---:|---|
> | baseline (PhoBERT 256) | 56,9% | 0,2153 | — |
> | `chunk550` (PhoBERT, hết cắt) | 0,4% | 0,2010 | `p = 0,711` — **không khác** |
> | `bgem3` (BGE-M3, hết cắt) | 0,0% | 0,5455 | `p < 0,001` |
>
> Hai dòng dưới **cùng** đưa truncation về ~0; một dòng không đổi gì, dòng kia +153%. Vậy
> biến giải thích là **model**, không phải truncation. Vai trò thật của cửa sổ 8192 là *cho
> phép giữ `chunk_size=1000`* — tức làm phép đo sạch (nhãn bit-identical → recall so được),
> không phải tạo ra mức tăng.
>
> Hệ quả cho `W2-08`: ma trận ablation phải có **cả hai chiều** `chunk_size` × `embedding`,
> không thì bảng số sẽ gán toàn bộ mức tăng cho một trong hai một cách sai.
>
> ⚠️ Cảnh báo về tính khách quan: BGE-M3 train trên corpus đa ngữ lớn (MIRACL, mC4) và corpus
> dự án là tài liệu World Bank — thể loại model rất có thể đã thấy nhiều. Không phải rò rỉ tập
> test (nhãn sinh từ chunk của chính corpus, chunk giống nhau ở cả hai lần chạy), nhưng mức
> tăng chưa chắc giữ nguyên trên corpus đóng của doanh nghiệp. Nói trong interview thì nói kèm.
>
> ### `W2-02` xong (2026-08-20) — sparse đã có chỗ chứa
>
> `rag_bgem3` build lại với cả `dense` và `sparse`: **+8,8 s trên 389 s (+2,3%)**, dung lượng
> +19%. Rẻ như vậy vì `W2-01` cho hai loại vector ra từ một forward pass. Dense **bit-identical**
> sau khi build lại (0/209 câu đổi điểm). Chi tiết: `reports/w2-02-qdrant-hybrid.md`.
>
> Phần đáng nhất lại ngoài DoD: `ensure_collection` giờ **kiểm tra** schema. Trong 4 ca lệch có
> một ca **hỏng im lặng** — collection có sparse mà provider chỉ sinh dense: eval ra số trông
> bình thường trong khi nửa index không được dùng. Biến thể của đúng cái bẫy `TD-11`.
>
> ### `W2-03` xong (2026-08-20) — sparse có số, và hai câu trả lời ngược nhau
>
> | Câu hỏi | Trả lời |
> |---|---|
> | Sparse tốt hơn dense trên golden set? | **Không.** nDCG@10 0,4442 → 0,3733, 12/15 metric kém có ý nghĩa |
> | Sparse tra được mã tài liệu mà dense không? | **Có, áp đảo.** hit@10 0,0784 → 0,5098, `p = 4,8e-07`, 22↔**0** |
>
> Hai câu không mâu thuẫn — chúng đo hai loại truy vấn khác nhau, và câu thứ hai mới là DoD.
> `golden_v1` toàn câu hỏi tự nhiên nên nó **không đo được** DoD; phải dựng phép đo riêng
> (`scripts/known_item_probe.py` — known-item search, tiêu chí kiểm bằng **so chuỗi** nên
> không cần nhãn người).
>
> **Con số đáng nhất là con số thứ ba: trần của `W2-04` là `hit_rate@10 = 0,7033`** (dense
> 0,6268). Ghi lại **trước khi** làm RRF để lúc đó không tự diễn giải kết quả theo hướng có
> lợi. +0,0765 tuyệt đối là toàn bộ số tiền đang nằm trên bàn.
>
> Ba phát hiện đổi cách làm phần còn lại của W2:
>
> 1. **`cross_lingual` là chỗ sparse chết hẳn** (1↔19, và **0 câu cả hai**), còn trên truy
>    vấn **tiếng Anh sparse thắng** (9↔4). Lợi ích của RRF rất khác nhau theo ngôn ngữ, nên
>    bảng `W2-08` phải tách theo `lang`, không chỉ theo `category`.
> 2. **Sparse học được không phải index khớp đúng.** 25/51 mã không nhánh nào tìm ra, vì
>    vocab là **subword**: `P171645` → `['▁P','171','645']`, không mảnh nào hiếm. RRF và
>    reranker đều không sửa được chuyện này → `TD-18`.
> 3. **Tìm trong index sparse tốn 97,8 ms vs dense 17,8 ms.** Từ `W2-04` trở đi, nhánh sparse
>    là thành phần nặng nhất của đường truy hồi — không phải model embedding.
>
> ### Việc tiếp theo: `W2-04` — RRF, với trần đã biết trước
>
> Nếu RRF ra thấp hơn 0,6268 thì nó đang **làm hại** so với dense một mình, và đó là kết quả
> phải báo cáo chứ không phải tinh chỉnh `k` cho tới khi số đẹp.
>
> ⚠️ Mọi số W2 đo trên `golden_v1` — vốn **review bằng model** (`TD-13`). So sánh *tương đối*
> giữa các cấu hình vẫn hợp lệ (cùng một thước đo); con số *tuyệt đối* thì chưa được gọi là
> "human-verified".
>
> 💡 `make eval-retrieval` tự ánh xạ nhãn span → `chunk_id` của index đang đo (`TD-12`), nên
> đổi `chunk_size` **không** làm hỏng golden set — đã chứng minh trên `chunk550`: 209/209 câu
> ánh xạ được, 0 câu mất nhãn. Nhưng phải build bằng `--recreate` để point mang
> `start_char`/`end_char`, nếu không `span_resolution` rơi về nhãn cũ và im lặng đo sai.

- [x] `W2-01` **BGE-M3 provider** (dense + sparse lexical weights cùng lúc) — 2026-08-20
  · DoD: ✅ dense 1024-d + sparse dict từ **một** forward pass · ✅ cache model load (`lru_cache` cho cả backbone và `sparse_linear.pt`) · ✅ `max_sequence_tokens` = 8192 · Test: `tests/unit/test_bge_m3.py` (23 CPU + 11 GPU) + `test_sparse_vector.py` (21) · Evidence: `reports/w2-01-bge-m3.md`, `configs/indexing/bgem3.yaml`
  · **Kết quả: nDCG@10 0,1621 → 0,4442** (+174%), `hit_rate@5` 0,2153 → 0,5455, MRR 0,1660 → 0,4394. **Cả 15 metric có ý nghĩa** (`p < 0,001` hoặc CI95 không chứa 0). Nhãn bit-identical với baseline (`n_relevant_mean` 1,3828) nên recall@k so được — khác `TD-11`
  · **`cross_lingual` recall@5 0,0000 → 0,3023** (43 câu = 18% golden set đi từ *không hoạt động* sang hoạt động). Cả 5/6 nhóm cải thiện có ý nghĩa; `table_lookup` n=4 không có lực thống kê
  · Truncation **56,9% → 0,0%** (0/15814 chunk), token/chunk max 734 vs cửa sổ 8192. Tokenizer BGE-M3 tốn **ít hơn 30%** token cho text tiếng Anh (0,172 vs 0,244 token/ký tự) — gốc của việc `en` bị cắt nặng nhất ở baseline
  · ⚠️ **Mức tăng này KHÔNG phải công của việc sửa `TD-11`.** Lần chạy đổi cùng lúc model + cửa sổ + tính đa ngữ; `TD-11` đã tách riêng phần cửa sổ và nó cho `p = 0,711`. Vai trò thật của cửa sổ 8192 là **cho phép giữ `chunk_size=1000`**, tức làm phép đo sạch — không phải tạo ra mức tăng. `W2-08` phải chốt bằng ma trận 2 chiều
  · ⚠️ **Sparse chưa đi vào Qdrant.** Eval trên là dense-only; phần sparse đã dựng + test nhưng chưa tiêu, chờ `W2-02`…`W2-04`
  · Chi phí: `batch_size` 64 → 16 · VRAM ~3,3/8 GB · index 405 s · p95 truy hồi 32,8 → 46,0 ms · vector 768 → 1024 chiều
- [x] `W2-02` **Qdrant named vectors + sparse index** + script migrate collection — 2026-08-20
  · DoD: ✅ `rag_bgem3` chứa cả `dense` (1024-d) & `sparse`, query độc lập bằng `retrieve()` / `retrieve_sparse()` · Test: `tests/integration/test_qdrant_hybrid_schema.py` (21) + `tests/unit/test_qdrant_schema.py` (22, thuần) · Evidence: `reports/w2-02-qdrant-hybrid.md`, `scripts/migrate_collection.py`
  · **Sparse gần như miễn phí: +8,8 s trên 389 s (+2,3%)**, dung lượng +19% (dense 61,8 MB → +11,6 MB). Đó là hệ quả trực tiếp của việc `W2-01` cho dense và sparse ra từ **một** forward pass — gọi provider hai lần thì đây là +380 s
  · **Dense bit-identical sau khi build lại**: 15/15 metric không lệch một chữ số, **0/209 câu đổi điểm**. Đường ghi hybrid mới không làm lệch nhánh dense — kiểm tra tuyên bố "một đường code" của `W2-01` trên 15.814 chunk thật
  · Sparse trên dữ liệu thật: **95,9 entry/chunk** (p50 100 · p95 147 · max 195 · **min 3**), mật độ 0,0384% của vocab 250.002. ReLU loại ~55% token (chunk có p50 218 token) — sparse của BGE-M3 là một phép **chọn**, không phải bag-of-words có trọng số
  · ⭐ **Ngoài DoD: `ensure_collection` giờ KIỂM TRA schema** thay vì thấy tồn tại là trả về. 4 ca lệch đều có test, trong đó ca "collection có sparse mà provider chỉ dense" là **hỏng im lặng** — eval ra số trông bình thường trong khi nửa index không được dùng. `schema_problems()` là hàm thuần nên 12 ca test được trong `make test`
  · ⚠️ Thang điểm hai nhánh **không so được**: dense là cosine ∈ [−1,1], sparse là dot product không có trần. Đây là lý do `W2-04` phải hợp nhất theo **thứ hạng**
  · ⚠️ **KHÔNG dùng `Modifier.IDF`** cho sparse của BGE-M3 — trọng số đã học, chồng IDF lên là nhân đôi phép hạ bậc từ phổ biến và hỏng im lặng. IDF dành cho nhánh BM25 thô ở `W2-03`. Có test canh `modifier is None`
  · `HashingEmbeddingProvider` nay sinh được sparse (`sparse=True`, mặc định **tắt** vì `name` là cache key) — để `W2-03`/`W2-04` test được mà không cần GPU
  · ⚠️ **Chi phí độ trễ đo được**: p50 truy hồi dense 23,7 → **31,5 ms** (+33%, tái lập 3 lần) sau khi thêm sparse index vào cùng collection; p95 gần như không đổi (46,0 → 46,6 ms) vì p95 bị chi phối bởi forward pass embed truy vấn. **Chưa tách được** chi phí sparse index khỏi trạng thái segment sau build lại — phải đo lại ở `W2-04` nơi mỗi truy vấn đi cả hai nhánh
- [x] `W2-03` **Sparse retriever** (`retrieve_sparse()` → `Retriever`, đo được) — 2026-08-20
  · DoD: ✅ query từ khóa lạ mà dense miss thì sparse hit — **known-item search: hit@10 0,0784 → 0,5098**, hit@1 0,0196 → 0,3529, McNemar `p = 4,8e-07`, và **0 mã nào dense tìm ra mà sparse không** · Test: `tests/integration/test_sparse_retriever.py` (14 integration + 4 gpu) + `tests/unit/test_sparse_branch.py` (16) · Evidence: `reports/w2-03-sparse-retriever.md`, `w2-03-known-item.json`, `scripts/known_item_probe.py`
  · ⭐ **Trần lý thuyết của `W2-04`: hợp hai nhánh cho `hit_rate@10 = 0,7033`** vs dense 0,6268 (+12,2% tương đối). Ghi lại **trước khi** làm RRF để lúc đó không tự diễn giải kết quả theo hướng có lợi
  · **Trên golden set sparse KÉM hơn dense** và kém có ý nghĩa: nDCG@10 0,4442 → 0,3733 (CI95 [−0,1190, −0,0225]), `hit_rate@10` 0,6268 → 0,5120 (`p = 0,002`, 40↔16). 12/15 metric khác biệt thật, tất cả cùng chiều. Đây là kết quả **phải chờ đợi**: `golden_v1` là câu hỏi tự nhiên, đúng loại truy vấn dense sinh ra để xử lý
  · Chỗ sparse **bù được**: `en` 9↔4 (sparse **thắng** trên truy vấn tiếng Anh), `factoid` 10↔7. Chỗ sparse **chết hẳn**: `cross_lingual` 1↔19 và **0 câu cả hai** — câu tiếng Việt trên tài liệu tiếng Anh không có token nào trùng
  · ⚠️ **Sparse học được KHÔNG phải index khớp đúng.** 25/51 mã không nhánh nào tìm ra, và tokenizer giải thích hết: mã tìm được có neo từ vựng (`VIE-01` → `['▁','VIE','-01']`), mã miss rã thành chữ số chung (`P171645` → `['▁P','171','645']`). Đây là bằng chứng cho `TD-18`, không còn là phỏng đoán
  · ⚠️ **Chi phí: tìm trong index sparse 97,8 ms vs dense 17,8 ms (5,5×).** p50 toàn phần 30,2 → 113,4 ms. Embed truy vấn 12,6 ms **dùng chung** cả hai nhánh (lợi còn lại của "một forward pass" ở `W2-01`). Mỗi truy vấn hybrid ở `W2-04` ≈ 128 ms
  · ⭐ **Ngoài DoD: đóng một hố im lặng trong `compare.py`.** Harness lấy `fetch_doc_chunks` bằng `getattr`; retriever thiếu method đó thì nhãn rơi về nhãn cũ và hai lần chạy **cùng số nhãn nhưng khác nhãn** vẫn so được. Thêm `QueryScore.relevant_digest`; `compare.py` từ chối **toàn bộ** khi băm lệch. Đã chạy thật: 209/209 có băm, **0 lệch**
  · **Kết quả âm đã ghi lại, không xoá**: giả định "dense lẫn giữa các mã gần giống" **sai** — trên corpus 7 chunk cả hai đều tra đúng hạng 1. Có test canh chính kết quả âm đó, thay vì đổi assertion thành `rank_sparse <= rank_dense` (sẽ pass vì bằng nhau, và đọc như một chiến thắng)
  · Giả định "không trùng token thì sparse trả rỗng" cũng **sai** trên 15.814 chunk: sparse trả đủ 20 kết quả cho cả 209 câu. Giới hạn thật là **xếp hạng sai**, không phải **không trả gì**
  · Nhánh truy hồi **không** vào `IndexConfig` (không quyết định vector nào được ghi → không thuộc `fingerprint`). Là cờ `--retrieval-mode` / `MODE=`, và ở `W2-07` là một chiều của ma trận
- [ ] `W2-04` **RRF fusion** (`k=60`, configurable) — **việc tiếp theo**
  · DoD: deterministic, tie-break ổn định · Test: `tests/unit/test_rrf.py` — so với thứ hạng tính tay, case 1 list rỗng · Evidence: —
  · **Trần đã biết trước khi bắt đầu: `hit_rate@10` tối đa 0,7033** (dense 0,6268). RRF chỉ tiến gần được, vì nó hợp nhất theo thứ hạng chứ không nhìn nội dung
  · Hợp nhất theo **thứ hạng**, không theo điểm — thang hai nhánh không so được (`W2-02`)
  · Đo lại độ trễ ở đây: ~128 ms/truy vấn theo `W2-03` §8. Đây là chỗ tách được chi phí sparse index khỏi trạng thái segment mà `W2-02` chưa tách được
  · ⚠️ Có ca **cả hai nhánh cùng sai một kiểu** (có test): hợp nhất hai danh sách cùng sai thì vẫn sai. Chỗ đó là việc của `W2-05`
- [ ] `W2-05` **Cross-encoder reranker** `bge-reranker-v2-m3` (batch, top_n configurable)
  · DoD: rerank 50 → 6 trong < 400ms trên GPU; có CPU fallback · Test: `tests/unit/test_reranker.py` (thứ tự score giảm dần, batch == single) · Evidence: bench latency
- [ ] `W2-06` **Metadata filter** (tenant_id, doc_type, date range, lang)
  · DoD: filter áp ở tầng Qdrant (không post-filter), không rò dữ liệu chéo tenant · Test: `tests/integration/test_metadata_filter.py` — case tenant isolation · Evidence: —
- [ ] `W2-07` **Experiment Runner** — YAML config matrix → chạy tuần tự → MLflow tracking
  · DoD: 1 lệnh chạy hết grid, resume được khi crash giữa đường · Test: `tests/unit/test_experiment_runner.py` (expand grid đúng số tổ hợp, resume) · Evidence: MLflow UI screenshot
- [ ] `W2-08` **Ablation #1**: chunking × embedding × retrieval mode × rerank
  · DoD: ≥ 12 tổ hợp có kết quả đầy đủ, xác định được cấu hình thắng **kèm `p`/CI cho từng dòng** · Test: — · Evidence: MLflow run IDs
  · ⚠️ Ma trận phải có **cả hai chiều** `chunk_size` × `embedding`: BGE-M3 xoá luôn nguyên nhân truncation nên gộp hai chiều sẽ gán toàn bộ mức tăng cho model một cách sai
  · ⚠️ Chiều `chunk_size` **không dùng** recall@k/nDCG/MAP (mẫu số là số nhãn — xem `G2`). `pipeline/eval/compare.py` tự từ chối những metric đó khi phân bố nhãn khác nhau
- [ ] `W2-09` **Report `reports/exp-001-retrieval.md`** — bảng số vs baseline + phân tích tại sao thắng/thua
  · DoD: có bảng delta, có ít nhất 2 nhận xét về *category nào cải thiện nhiều nhất* · Test: — · Evidence: file report

### `G2` — Gate tuần 2 ⬜
- [x] nDCG@10 tăng so với baseline (kỳ vọng ≥ +0.08 tuyệt đối) — **đạt ở `W2-01`: +0,2820** (0,1621 → 0,4442), gấp 3,5× ngưỡng, CI95 [+0,2297, +0,3346]
  · ⚠️ **Chỉ so được khi số nhãn/câu không đổi** — nDCG có mẫu số là số nhãn, mà nhãn neo theo span nên đổi `chunk_size` là đổi mẫu số (`TD-11` đo được: 1,38 → 1,96 nhãn/câu khi hạ 1000 → 550, làm nDCG tụt 29,6% **kể cả khi truy hồi y nguyên**). Đổi `chunk_size` thì phải dùng `hit_rate@k`/`MRR`
  · Tin tốt: +0,08 tuyệt đối trên 0,1621 là **+49% tương đối** — nằm trong tầm phân giải của `golden_v1`
  · `W2-01` giữ `chunk_size=1000` nên nhãn bit-identical với baseline (1,3828 nhãn/câu cả hai bên) → nDCG so được trực tiếp, `compare.py` không từ chối metric nào
- [x] **Mọi dòng ablation có `p` hoặc CI95** (`make eval-compare`), và chỉ tuyên bố người thắng khi kiểm định nói vậy — hạ tầng xong ở `TD-11`, dùng thật ở `W2-01` (15/15 metric có `p`/CI)
  · Đo được ở `TD-11`: với 209 câu và `hit_rate@5` ≈ 0,20, phải chênh **≥ 6 điểm tuyệt đối (≈28% tương đối)** mới phát hiện được. Xếp hạng 12 tổ hợp bằng mức chênh vài phần trăm là tung đồng xu
- [ ] Có bảng ablation ≥ 12 dòng, tái lập được bằng `make eval EXP=exp_001`
- [ ] Không tổ hợp nào làm p95 latency vượt 3500 ms

---

## 5. W3 · Ingestion + Chunking nâng cao

- [ ] `W3-01` **Docling loader** (PDF, DOCX, PPTX, XLSX, HTML, MD) + bảng → Markdown + heading hierarchy
  · DoD: PDF 2 cột đọc đúng reading order; bảng giữ được cấu trúc · Test: `tests/unit/test_loaders.py` với 6 file fixture (mỗi định dạng 1 file) · Evidence: so sánh output trước/sau
- [ ] `W3-02` **Scan detection + OCR fallback** (text density < ngưỡng → Qwen2.5-VL / Gemini Vision)
  · DoD: PDF scan ra được text; có queue control tránh OOM (tái dùng kinh nghiệm SmartSchool) · Test: `tests/integration/test_ocr_fallback.py` với 1 PDF scan · Evidence: —
- [ ] `W3-03` **Structure-aware chunker** — cắt theo heading trước, mỗi chunk có `section_path`
  · DoD: `section_path` đúng trên tài liệu có 3 cấp heading · Test: `tests/unit/test_structure_chunker.py` (3 cấp heading, doc không heading) · Evidence: —
- [ ] `W3-04` ⭐ **Contextual Retrieval** — **vLLM offline batch (`Qwen3-8B`) trên GPU thuê** sinh 1–2 câu context prepend vào chunk trước khi embed; fallback DeepSeek API khi không có GPU; có cost cap
  · Chạy trên **GPU thuê 24GB, không mang API key** (vLLM chạy ngay trên box). ~2–4h.
  · DoD: cost/1000 chunk được log; có flag tắt; fail 1 chunk không làm sập cả job · Test: `tests/unit/test_contextual_chunking.py` (context không rỗng, fallback khi LLM lỗi, cost cap chặn) · Evidence: cost log
- [ ] `W3-05` **Parent-child (small-to-big)** — embed child 256 token, trả parent ~1024 token
  · DoD: retrieve child → context assembly trả parent, dedupe parent trùng · Test: `tests/unit/test_parent_child.py` · Evidence: —
- [ ] `W3-06` **Token-based sizing** theo tokenizer của embedding model (bỏ đếm ký tự)
  · DoD: không chunk nào vượt max token của model · Test: `tests/unit/test_token_sizing.py` (assert mọi chunk ≤ limit trên 3 corpus mẫu) · Evidence: —
- [ ] `W3-07` **Content-hash dedupe + incremental re-index**
  · DoD: sửa 1 trang trong 100 trang → chỉ embed lại chunk bị ảnh hưởng · Test: `tests/integration/test_incremental_reindex.py` (đếm số lần gọi embed) · Evidence: log so sánh
- [ ] `W3-08` **Async ingestion worker (arq)** + job status
  · DoD: `POST /ingest` trả `job_id` < 200ms; `GET /ingest/{id}` có progress; retry khi worker chết · Test: `tests/integration/test_ingest_job.py` · Evidence: —
- [ ] `W3-09` **Ablation #2 + report** `reports/exp-002-chunking.md` — 5 chiến lược chunking
  · DoD: xác định được contextual chunking giảm retrieval failure bao nhiêu % (có số) · Test: — · Evidence: file report

### `G3` — Gate tuần 3 ⬜
- [ ] Contextual/structure-aware chunking thắng hybrid cũ trên nDCG@10 (hoặc kết luận rõ là không, kèm số)
- [ ] Ingest được ≥ 5 định dạng file, có test fixture cho từng loại
- [ ] Reprocess sau sửa nhỏ nhanh hơn full rebuild ≥ 10× (có số đo)

---

## 6. W4 · Serving Plane

- [ ] `W4-01` **`RagBundle` schema + save/load/validate + checksum**
  · DoD: bundle thiếu field hoặc sai checksum bị reject; version theo semver; **bắt buộc có `evaluated_with_generator`** · Test: `tests/unit/test_bundle.py` (round-trip, reject checksum sai, reject thiếu eval report, reject thiếu `evaluated_with_generator`) · Evidence: bundle mẫu
- [ ] `W4-02` **Bundle loader hot-reload** + `POST /admin/bundle/reload` + rollback
  · DoD: đổi bundle không restart process, request đang chạy không bị lỗi; rollback 1 lệnh · Test: `tests/integration/test_bundle_reload.py` (reload giữa 2 request, rollback) · Evidence: —
- [ ] `W4-03` **FastAPI skeleton** — DI, `/health` + `/ready` (phân biệt rõ), structlog JSON + request_id
  · DoD: `/ready` chỉ 200 khi bundle + Qdrant + DB sẵn sàng · Test: `tests/integration/test_health.py` (ready=503 khi chưa load bundle) · Evidence: —
- [ ] `W4-04` **Auth (API key / JWT) + rate limit per tenant**
  · DoD: request không key → 401; vượt quota → 429 có `Retry-After` · Test: `tests/integration/test_auth_ratelimit.py` · Evidence: —
- [ ] `W4-05` **Postgres schema + Alembic** — conversation, message, document, ingest_job, feedback
  · DoD: `alembic upgrade head` từ DB trống; có downgrade · Test: `tests/integration/test_migrations.py` (up → down → up) · Evidence: —
- [ ] `W4-06` **`POST /chat` SSE streaming** + lưu conversation vào Postgres
  · DoD: token stream về client; chat history sống sót qua restart container · Test: `tests/integration/test_chat_stream.py` (nhận ≥ 2 chunk SSE, reload conv) · Evidence: GIF demo
- [ ] `W4-07` **Query Understanding** — multi-turn rewrite, language detect, routing (NO_RETRIEVAL / RETRIEVE / CLARIFY)
  · DoD: "cái đó thì sao?" được rewrite thành câu độc lập; "hello" không gọi retrieval · Test: `tests/unit/test_query_understanding.py` (bộ 15 case gán nhãn tay) · Evidence: —
- [ ] `W4-08` **LLM Router** DeepSeek (primary) → OpenRouter pinned slug (fallback) → vLLM profile (tùy chọn) + circuit breaker + retry + budget cap
  · DoD: primary lỗi 3 lần → fallback; vượt budget/ngày → từ chối có thông báo rõ; **log model thực tế đã phục vụ** mỗi request · Test: `tests/unit/test_llm_router.py` (mock 5xx, timeout, budget exceeded, assert log model) · Evidence: —
- [ ] `W4-09` ⭐ **Structured output + citation verification** — bỏ hẳn `text.split("Trả lời:")`
  · DoD: output JSON validate bằng pydantic; mỗi `quote` phải match được trong chunk được cite, không match → đánh dấu unverified · Test: `tests/unit/test_citation_verify.py` (quote thật, quote bịa, quote sai chunk) · Evidence: —
- [ ] `W4-10` **Redis semantic cache** (ngưỡng cosine ~0.95)
  · DoD: query gần giống → cache hit, có TTL, invalidate khi đổi bundle · Test: `tests/integration/test_semantic_cache.py` (hit paraphrase, miss câu khác chủ đề, invalidate) · Evidence: tỉ lệ hit đo trên 100 query
- [ ] `W4-11` **Prompt registry** — YAML có version + hash, loader
  · DoD: đổi prompt = tăng version; runtime log rõ prompt version đang dùng · Test: `tests/unit/test_prompt_registry.py` (load, hash đổi khi nội dung đổi) · Evidence: —
- [ ] `W4-12` **Guardrails** — phát hiện prompt injection trong **nội dung tài liệu**, tách ranh giới data/instruction, PII redaction ở log
  · DoD: PDF chứa "ignore previous instructions" không đổi được hành vi · Test: `tests/security/test_prompt_injection.py` (bộ 10 payload) + `test_pii_redact.py` · Evidence: `reports/security-w4.md`
- [ ] `W4-13` **`serving/Dockerfile` + compose integration + smoke e2e**
  · DoD: `make up` → API healthy → `curl /chat` trả stream có citation · Test: `tests/e2e/test_smoke.py` · Evidence: log e2e

### `G4` — Gate tuần 4 ⬜
- [ ] Từ clone sạch: `make up` → `curl /chat` trả câu trả lời **có citation đã verify** trong ≤ 5 phút setup
- [ ] Đổi retrieval config chỉ bằng `POST /admin/bundle/reload`, **không rebuild image**
- [ ] Chat history sống sót qua `docker compose restart`

---

## 7. W5 · Eval đầy đủ + Gate + Observability

- [ ] `W5-01` **Generation eval** — faithfulness, answer relevancy, context precision/recall (RAGAS hoặc tự implement)
  · DoD: chạy trên `golden_v1`, có breakdown theo category · Test: `tests/unit/test_generation_metrics.py` trên fixture có nhãn tay · Evidence: —
- [ ] `W5-02` **Citation accuracy + refusal correctness**
  · DoD: đo được % câu unanswerable mà hệ thống từ chối đúng · Test: `tests/unit/test_refusal_metric.py` · Evidence: —
- [ ] `W5-03` **LLM judge module** — judge = `deepseek-reasoner` **pinned**, `temp=0`; cache theo hash (query+answer+context), cost cap, retry
  · DoD: chạy lại eval lần 2 gần như 0 cost nhờ cache; **cấm mọi OpenRouter preset trong đường eval** · Test: `tests/unit/test_judge.py` (cache hit, cost cap, parse output lỗi, reject config dùng preset) · Evidence: cost log 2 lần chạy
- [ ] `W5-04` ⭐ **Judge calibration** — tự gán nhãn tay 50 mẫu, tính Cohen's kappa judge vs người; **cross-check bằng 1 judge khác họ** (OpenRouter, pin slug)
  · DoD: `reports/judge-calibration.md` có kappa vs người + agreement giữa 2 judge khác họ + phân tích case judge sai · Test: `tests/unit/test_kappa.py` (so với giá trị tính tay) · Evidence: file report
- [ ] `W5-05` **`pipeline/eval/gate.py`** — thresholds YAML, so với champion, xuất HTML report, exit code khác 0 khi FAIL
  · DoD: `make gate BUNDLE=x` cho PASS/FAIL rõ ràng · Test: `tests/unit/test_gate.py` (bundle tốt PASS, bundle tụt nDCG FAIL, thiếu metric FAIL) · Evidence: HTML report mẫu
- [ ] `W5-06` **Langfuse self-host + instrument full trace** (rewrite → retrieve kèm score → rerank → prompt → completion, cost/token per step)
  · DoD: 1 query hiện đủ span trong Langfuse UI, có cost · Test: `tests/integration/test_tracing.py` (assert số span) · Evidence: screenshot trace
- [ ] `W5-07` **Prometheus metrics + Grafana dashboard "RAG Health"**
  · DoD: dashboard JSON commit trong `infra/grafana/`; có RED metrics, cache hit rate, retrieval empty rate, refusal rate, cost/hour · Test: `tests/integration/test_metrics_endpoint.py` (các metric key tồn tại) · Evidence: screenshot dashboard
- [ ] `W5-08` **Feedback endpoint** 👍/👎 + lý do → Postgres → Langfuse score → hàng đợi review
  · DoD: câu 👎 xuất ra được file candidate để bổ sung golden set · Test: `tests/integration/test_feedback.py` · Evidence: —
- [ ] `W5-09` **CI GitHub Actions** — ruff, mypy, pytest, smoke eval 30 câu (< 5 phút, dùng model rẻ)
  · DoD: PR mở ra là CI chạy; smoke eval fail thì CI đỏ · Test: — · Evidence: CI run URL
- [ ] `W5-10` **Nightly full eval workflow** (self-hosted GPU runner hoặc trigger tay) + auto tạo PR promote khi PASS
  · DoD: chạy full `golden_v1`, đăng report vào PR comment · Test: — · Evidence: 1 lần chạy thật
- [ ] `W5-11` ⭐ **Generator ablation** — `Qwen3-8B (vLLM, pinned, temp=0)` vs `deepseek-chat` vs `OpenRouter pinned slug`, **cùng một retrieval stack**
  · **Tách nhánh theo nơi chạy:** nhánh vLLM trên GPU thuê (không key); nhánh DeepSeek/OpenRouter chạy **từ laptop** (không cần GPU). Judge luôn chạy từ laptop.
  · DoD: `reports/exp-003-generator.md` có bảng quality × p95 latency × cost/query; nêu rõ chọn model nào cho production và **vì sao**; ghi cảnh báo self-preference bias · Test: `tests/unit/test_bundle.py::test_evaluated_with_generator_required` (bundle thiếu field → reject) · Evidence: file report + MLflow run IDs

### `G5` — Gate tuần 5 ⬜
- [ ] **Test ngược:** mở 1 PR cố ý làm tụt retrieval (ví dụ `top_k=1`, tắt rerank) → **CI phải đỏ**
- [ ] Có `reports/judge-calibration.md` với kappa ≥ 0.6 (nếu < 0.6 → sửa prompt judge rồi đo lại)
- [ ] 1 query trace được end-to-end trong Langfuse kèm cost
- [ ] Gate **từ chối** so sánh 2 bundle khác `evaluated_with_generator` (có test chứng minh)

---

## 8. W6 · Hoàn thiện & trình bày

- [ ] `W6-01` **Web UI** — streaming, citation click → highlight chunk gốc, feedback 👍/👎, upload progress
  · DoD: người lạ dùng được không cần hướng dẫn · Test: `tests/e2e/test_ui_smoke.py` (Playwright) · Evidence: GIF
- [ ] `W6-02` **HF Spaces demo** (API-only, không GPU) + corpus mẫu sẵn
  · DoD: link public mở ra hỏi được ngay < 30s, có rate limit chống lạm dụng · Test: — · Evidence: URL Space
- [ ] `W6-03` **README** — kiến trúc (Mermaid), **bảng metric thật**, GIF demo, quickstart 1 lệnh
  · DoD: không có số nào không truy được về `reports/` · Test: — · Evidence: README
- [ ] `W6-04` **`ARCHITECTURE.md` + `EVALUATION.md` + `BUNDLE.md`**
  · DoD: `EVALUATION.md` giải thích được quy trình tạo golden set và vì sao tin được metric · Test: — · Evidence: —
- [ ] `W6-05` **Load test (locust)** → p50/p95/p99 theo concurrency
  · DoD: `reports/loadtest.md` có biểu đồ + điểm bão hòa · Test: — · Evidence: file report
- [ ] `W6-06` **Security pass** (`/ek:security` hoặc `/security-review`)
  · DoD: 0 finding severity cao chưa xử lý · Test: — · Evidence: `reports/security-final.md`
- [ ] `W6-07` **Cập nhật CV** — điền số thật vào bản nháp ở §8 của plan, đổi link repo
  · DoD: mọi con số trong CV có script tái lập · Test: — · Evidence: `main.tex` diff
- [ ] `W6-08` ⭐ **Sửa các claim sai trong CV hiện tại** — bỏ "FAISS" nếu không dùng thật; "hybrid search" giờ đúng nghĩa BM25+dense; thay số 3×/15×/80–90% bằng số có evidence
  · DoD: tự phỏng vấn thử: mọi dòng trong mục project đều trả lời được bằng file trong repo · Test: — · Evidence: —

### `G6` — Gate tuần 6 ⬜
- [ ] Recruiter mở link demo, hỏi 1 câu, nhận câu trả lời có citation — trong 30 giây
- [ ] README có bảng before/after với số thật của cả 8 metric
- [ ] Toàn bộ CV claim về project này đều có evidence trong repo

---

## 9. Task thêm mới (phát sinh ngoài plan gốc)

> Thêm task mới vào đây, ID `NEW-xx`, ghi rõ phát sinh từ đâu. Không chèn vào backlog gốc.

| ID | Task | Phát sinh từ | Trạng thái | Ngày thêm |
|---|---|---|---|---|
| `NEW-02` | **`scripts/fetch_corpus.py` + `pipeline/corpus/`** — adapter World Bank WDS + nguồn `seed_list`, manifest CSV ép giấy phép, chạy lại là no-op | `W0-03`: cần cách tải corpus tái lập được, không phải tải tay rồi quên mất lấy ở đâu | `[x]` | 2026-08-17 |
| `NEW-05` | **`packages/rag_core/llm/`** — client tương thích OpenAI tự viết bằng `httpx`, có bảng giá, phát hiện model trôi, retry chỉ cho lỗi tạm thời, chặn OpenRouter preset ở tầng constructor | `W1-10`: cần LLM cho cả pipeline (sinh dữ liệu, judge) lẫn serving (generator), nên nó là hợp đồng dùng chung chứ không phải tiện ích của một script | `[x]` | 2026-08-17 |
| `NEW-06` | **Checkpoint + chạy song song cho job LLM dài** | `W1-10`: lượt chạy đầu treo ở phút 40 và mất sạch. Song song 6 luồng đưa 163 lời gọi từ >1 giờ xuống 640 giây | `[x]` | 2026-08-17 |
| `NEW-03` | **`Chunker.prepare(n_documents)`** — người gọi khai báo kích thước lô thật trước khi cache chia nhỏ theo từng tài liệu | `W1-08`: không có nó thì `HybridChunker` luôn thấy `n=1` và luôn chọn semantic, tức baseline đo một chiến lược chunking khác hệ thống hiện tại | `[x]` | 2026-08-17 |
| `NEW-04` | **Warm-up trước khi đo độ trễ trong `run_retrieval_eval`** | `W1-08`: p95 đo được là 15.219 ms trong khi p50 là 31 ms — toàn bộ chênh lệch là thời gian nạp model. Ngưỡng gate hiệu năng W5/W6 dựa trên p95 | `[x]` | 2026-08-17 |
| `NEW-01` | **Test canh chiều phụ thuộc hai plane** — `tests/unit/test_architecture_boundaries.py`: quét AST chặn `rag_core → pipeline/serving`, `serving → pipeline`, `pipeline → serving`, và chặn import nặng (`torch`, `qdrant_client`) ở tầng module của `rag_core` | `W1-01`: ranh giới hai plane là lý do tồn tại của cả kiến trúc, mà một dòng `from pipeline...` lọt vào `serving/` sẽ xoá nó rất tự nhiên | `[x]` | 2026-08-17 |

---

## 10. Đang bị chặn / đang đợi

| ID | Đợi gì | Ai xử lý | Từ ngày | Ảnh hưởng |
|---|---|---|---|---|
| `W0-01` | Bạn đọc lại plan sau khi sửa diagram | Bạn | 2026-08-14 | **Không còn chặn gì** — W1 đã làm xong 13/13 theo plan hiện tại. Giữ lại vì §3.6 (phân vai LLM) và §8 (nháp CV) sẽ được dùng lại ở W5/W6 |
| `W0-02` | Quyết định tên repo mới + có tạo repo riêng hay nâng cấp in-place | Bạn | 2026-08-14 | Toàn bộ tuần 1 đã push lên nhánh `feat/w1-foundation` (`f5ec22b` … `b9fcbce`, 15 commit tính cả `f5ec22b`). `main` vẫn là bản POC cũ vì đang là link trong CV. Cần quyết: merge vào `main` hay tách repo mới. **Càng để lâu càng khó**: `W6-02` (demo HF Spaces) và `W6-07` (đổi link CV) đều trỏ vào quyết định này |
| `W0-03` | Nguồn (b) văn bản pháp luật ~30 và (c) báo cáo HOSE ~30 — cần chọn tay rồi khai báo qua `seed_list` | Bạn | 2026-08-14 | Nguồn (a) xong (60 tài liệu, đã index). Thiếu (b)/(c) thì golden set không có nhóm `table_lookup` và `section_path` |
| `W0-07` | `@preset/my-luna-pro` resolve ra model slug nào (cần biết để chọn judge cross-check khác họ DeepSeek) | Bạn | 2026-08-14 | Chặn `W5-04`, không chặn W1–W4 |
| `TD-13` | **Người đọc lại 33 câu `unanswerable` + 43 câu `cross_lingual`** rồi `make goldenset-freeze` với `--reviewer human` (~2–3h, hàng đợi đã xếp sẵn) | Bạn | 2026-08-20 | **Không chặn `W2`** — so sánh tương đối giữa các cấu hình vẫn hợp lệ. Chặn việc gọi baseline là "human-verified" ở README / CV / phỏng vấn, và chặn `G1` chuyển từ 🟡 sang ✅ |

---

## 11. Nợ kỹ thuật ghi nhận

> Những chỗ cố tình làm tạm để đi nhanh. Ghi lại để không tự lừa mình.

| ID | Nợ | Lý do chấp nhận | Kế hoạch trả |
|---|---|---|---|
| `TD-01` | `HuggingFaceEmbeddingProvider` coverage 0% | Cần `torch` (~2.5GB) mà unit test cố ý chạy không có GPU stack để `make test` giữ ở ~3 giây | **Đã trả một phần ở `W2-01`**: có `make test-gpu` + 11 test chạy model thật (`tests/unit/test_bge_m3.py`), trong đó bài canh dense khớp `SentenceTransformer.encode()`. Đường `_encode`/`count_tokens` của bản dense-only vẫn chưa có test GPU riêng — làm cùng `W0-06` (đo VRAM) |
| ~~`TD-02`~~ | ~~CLI `retrieval_eval.py` chưa nối retriever thật~~ | **Đã trả ở `W1-08`**: `--index-config` dựng retriever từ chính config đã build index | ✅ |
| ~~`TD-03`~~ | ~~`pipeline/indexing/` còn rỗng~~ | **Đã trả ở `W1-08`** | ✅ |
| `TD-04` | Bộ tải corpus loại 5 tài liệu vì "không giải mã được UTF-8" | Một số bản `.txt` của World Bank mã hoá cp1252. Loại bỏ là an toàn, và 60 tài liệu đã đủ cho nguồn (a) | Thử fallback cp1252 → utf-8 khi cần thêm tài liệu |
| `TD-05` | Giấy phép CC BY 3.0 IGO là khai báo **ở mức nguồn**, không kiểm chứng từng tài liệu | Script chỉ ép được rằng giấy phép nằm trong danh sách cho phép | Đọc tay trang giấy phép của ~5 tài liệu bất kỳ **trước khi** push repo lên public |
| `TD-06` | Toàn bộ `Document` nằm trong RAM khi build index | 60 tài liệu = 14 MB, tối ưu sớm là lãng phí | `W3-07` (re-index tăng dần) sẽ đọc theo luồng |
| ~~`TD-07`~~ | ~~`plans/` bị thêm vào `.gitignore`~~ | **Đã xử lý**: bỏ dòng đó, vì DoD của dự án bắt buộc mỗi task có đường dẫn Evidence | ✅ |
| `TD-08` | 22/163 lời gọi LLM bị cắt ở `max_tokens=6000` — toàn bộ ngân sách đi vào chuỗi suy luận | Vẫn đủ 266 câu; nâng tiếp là trả tiền cho phần suy luận không dùng tới | Thử `--questions-per-call 1` (prompt ngắn → suy luận ngắn) khi cần thêm câu |
| `TD-10` | DVC remote chỉ là thư mục local `D:/dvc-remote/rag-chatbot` | Nguồn (a) còn đường phục hồi độc lập: `scripts/fetch_corpus.py` tải lại từ World Bank rồi so sha256. Mất remote chưa mất corpus | **Bắt buộc dựng remote dùng chung trước khi thêm nguồn (b)/(c)** — văn bản pháp luật + báo cáo HOSE phải chọn tay, không script nào tải lại được, mất remote là mất corpus |
| ~~`TD-09`~~ | ~~Chưa kiểm được câu `unanswerable` có thật sự không trả lời được~~ | **Đã trả**: `triage.py` chạy retriever lên cả 40 câu, **15 câu** vượt ngưỡng hiệu chuẩn 0,5797 → đầu hàng đợi review | ✅ Nhưng kế hoạch cũ phải sửa: điểm cao **không** đủ để phân loại lại tự động. Ví dụ đầu hàng đợi có điểm 0,7287 mà chunk top-1 vẫn không trả lời được câu hỏi — điểm cao chứng minh *cùng chủ đề*, không chứng minh *trả lời được* |
| ~~`TD-12`~~ | ~~`chunk_id` thuần vị trí làm golden set hỏng âm thầm khi đổi chunking~~ | **Đã trả**: nhãn neo theo **span ký tự** trong văn bản gốc (`TextSpan`, `Chunk.start_char/end_char`), eval tự ánh xạ span → chunk_id của index đang đo. 226/226 câu đã neo, 299 span (62% thu về đúng câu trích dẫn). Đo ở 3 cấu hình chunking (1000/600/400): **0 câu mất nhãn**; ở 1000 dựng lại đúng nhãn cũ 216/226 | ✅ `reports/w1-11-spans.md` |
| ~~`TD-11`~~ ⭐ | ~~**56,8% chunk bị cắt lúc embed**~~ — `chunk_size=1000` **ký tự** ≈ 340 token, `vietnamese-bi-encoder` (PhoBERT) có `max_seq_length=256` **token**. `sentence-transformers` cắt âm thầm, không cảnh báo. **15,7% toàn bộ text đem embed không bao giờ tới được vector** (mẫu ngẫu nhiên 1200 chunk, seed 20260820; riêng chunk đã gán thì 91,1% vì bộ lọc prose-like chọn chunk dài) | **Cố ý không sửa trước `W1-13`**: đây là hành vi thật của bản POC, sửa trước rồi gọi kết quả là "baseline" thì mọi so sánh về sau đo lẫn cải tiến này vào. Chiều câu hỏi an toàn: p50 = 44 token, 0/266 bị cắt | **✅ Đã trả phần đo ở W2 (2026-08-20) — và giả định của nợ này bị phản chứng.** Hạ `chunk_size` 1000 → 550 đưa truncation từ 56,9% về **0,4%** chunk (token mất 15,4% → 0,1%) và **không cải thiện gì đo được**: McNemar `p = 0,711`, mọi CI bootstrap chứa 0. Lý do: baseline bị cắt nhưng mỗi vector vẫn đọc ~950 ký tự, chunk550 không bị cắt nhưng mỗi vector chỉ đọc 678 — hạ `chunk_size` là **đánh đổi**, không phải thu hồi. Còn lại là `W2-01` (BGE-M3, cửa sổ 8192): xoá truncation **mà không** phải hạ `chunk_size`. Evidence: `reports/w2-td11-chunk-size.md` |
| `TD-17` | **1/60 tài liệu dùng `Ê` (U+00CA) làm dấu cách** — `wb-099553007092621441` (*Điểm lại*, bản VI mới nhất): 5,7% ký tự của nó là `Ê`, tức văn bản **không có ranh giới từ**. Hệ quả: splitter không cắt được theo `" "` nên rơi xuống mức ký tự, tokenizer nổ ra 0,63 token/ký tự (bình thường 0,20) và 31/36 chunk vẫn bị cắt kể cả ở `chunk_size=550`. Embedding của tài liệu này về cơ bản là rác | Ảnh hưởng hẹp và đã đo: **1/242 câu** golden set trỏ vào nó (`factoid-caa304ba55`), 36/31.155 chunk của index. Sửa đúng chỗ là ở tầng đọc tài liệu, không phải ở chunking — chữa bằng cách hạ `chunk_size` là chữa triệu chứng | `W3-01` (Docling) đọc lại từ PDF gốc thay vì bản `.txt` mà World Bank trích. Trước đó thêm **cổng chất lượng corpus**: tỉ lệ whitespace < 10% thì cảnh báo — corpus hiện tại p50 là 37,5%, tài liệu này 16,4% với 5,7% là `Ê` giả dạng. Phép kiểm rẻ, và nó bắt được cả họ lỗi 'PDF trích ra chữ nhưng mất ranh giới'  · **Cập nhật `W2-01`:** cửa sổ 8192 của BGE-M3 làm tài liệu này **không còn gây truncation** (0/36 chunk), nhưng văn bản vẫn không có ranh giới từ nên nợ vẫn mở — triệu chứng mất, nguyên nhân còn. Sửa ở `W3-01` |
| `TD-13` ⭐ | **`golden_v1` review bằng MODEL, không phải người.** `reviewed_by_human=false`, `reviewed_by="model:claude-opus-5"` trong cả 242 dòng | Bạn không có ~6–8h để đọc tay, mà thiếu golden set thì cả W2–W6 không có thước đo. DeepSeek sinh → Claude review là **cross-model** nên không phải tự chấm mình; và mọi kết luận đều được đối chiếu với **văn bản gốc**, không phải với retriever (dùng retriever để lọc golden set là tự thổi phồng recall) | Người đọc lại **hai nhóm loại nhiều nhất** = hai nhóm model kém tự tin nhất: 33 câu `unanswerable` (đã tìm ra 7/40 sai nhãn, tỉ lệ sai 17,5%) + 43 câu `cross_lingual`. Rồi `make goldenset-freeze` với `--reviewer human`. Cho tới lúc đó: **cấm** chữ "human-verified" ở README, report, CV, phỏng vấn |
| `TD-14` | **7 `reference_answer` lẫn ngữ cảnh lúc sinh** — "Theo đoạn văn…", "Đoạn văn không nói…", "không có trong các đoạn văn được cung cấp" (`multi_hop-78a942bf28`, `multi_hop-d8e57b6cb1`, `multi_hop-6af19bc73a`, `adversarial-8ce6a8483a`, `adversarial-87b49c12d2`, `unanswerable-200ea2a853`, `unanswerable-599b31f973`) | **Vô hại cho eval truy hồi** (`W1-13` chỉ dùng `relevant_chunk_ids`/`relevant_spans`, không đọc `reference_answer`). Câu hỏi thì sạch: **0/242** câu có lỗi này | Dọn trước `W5-01`: judge so câu trả lời của hệ thống với `reference_answer`, mà "Theo đoạn văn 1" là thứ hệ thống thật không bao giờ nói được → judge sẽ trừ điểm oan. Làm cùng lượt người review `TD-13` |
| `TD-15` | **Nhãn liên quan chưa đầy đủ** — một dữ kiện có mặt ở nhiều tài liệu nhưng chỉ 1 chỗ được gán. Ví dụ "92% tài sản của các định chế tài chính" có trong **cả hai** *Financial sector assessment* và *Taking stock* | Hiếm: phép thử Jaccard ở `W1-11` cho **1/78** câu. Sửa cho đủ nghĩa là gán chéo toàn corpus × toàn golden set — đắt hơn giá trị mang lại ở giai đoạn này | Recall thật **cao hơn** số đo, tức baseline là **cận dưới** — sai theo chiều an toàn (không thổi phồng cải tiến). Nếu W2 đưa recall lên >0,6 mà vẫn có câu "trượt", kiểm chỗ này trước khi kết luận hệ thống sai |
| `TD-16` | **Nhóm `unanswerable` thiếu đa dạng** — 10/33 câu theo cùng một khuôn "hỏi về nước khác" (Thái Lan, Trung Quốc, Indonesia…) | 33 câu vẫn đủ để đo refusal correctness ở `W5-02`, và khuôn này đúng là một dạng câu không trả lời được | Thêm 2 khuôn khác khi có nguồn (b)/(c): hỏi về **mốc thời gian ngoài phạm vi** tài liệu, và hỏi số liệu **chi tiết hơn mức tài liệu có**. Nếu chỉ đo được một khuôn thì refusal correctness cao có thể chỉ nghĩa là hệ thống giỏi nhận ra tên nước lạ |
| `TD-18` | **Không có nhánh khớp đúng cho mã tài liệu.** `W2-03` đo được: 25/51 mã (project ID, trust fund ID) **không nhánh nào** tìm ra ở top-10. Nguyên nhân là vocab **subword** của BGE-M3: `P171645` → `['▁P','171','645']`, và `171`/`645` có mặt khắp 15.814 chunk tài liệu thống kê. Mã tìm được thì luôn có một mảnh subword tự nó là từ hiếm (`VIE-01` → `['▁','VIE','-01']`) | Ảnh hưởng **không** đo được trên `golden_v1` — 209 câu đều là câu hỏi tự nhiên, không có câu nào tra mã. Nhưng đây là loại truy vấn có thật trong sản phẩm (người dùng dán mã dự án vào hộp tìm kiếm), và **RRF (`W2-04`) lẫn reranker (`W2-05`) đều không sửa được** — cả hai chỉ xếp lại thứ tự những gì đã được tìm ra | Hai phương án, **đo cái rẻ trước**: (a) **filter payload khớp chuỗi** — Qdrant đã có `create_payload_index` từ `W1-07`, không cần build lại index; (b) **BM25 thô mức từ** (không phải subword) làm named vector sparse thứ hai — đây là chỗ cần `modifier=Modifier.IDF`, khác nhánh BGE-M3, nhưng phải đổi schema + build lại index. `scripts/known_item_probe.py` là phép đo sẵn có để so trước/sau |

---

## 12. Changelog

| Ngày | Thay đổi |
|---|---|
| 2026-08-20 | **`W2-03` xong — sparse có số, và hai câu trả lời ngược nhau.** `retrieve_sparse()` giờ là `Retriever` (`QdrantSparseRetriever` bọc chính store đang mở kết nối; `build_branch()` chọn nhánh, `--retrieval-mode`/`MODE=`). **Trên golden set sparse KÉM hơn dense có ý nghĩa**: nDCG@10 0,4442 → 0,3733 (CI95 [−0,1190, −0,0225]), `hit_rate@10` 0,6268 → 0,5120 (`p = 0,002`), 12/15 metric cùng chiều — kết quả **phải chờ đợi** vì `golden_v1` toàn câu hỏi tự nhiên. **Nhưng DoD thì đạt áp đảo**: `golden_v1` không đo được DoD nên dựng phép đo riêng `scripts/known_item_probe.py` (known-item search trên 51 mã tài liệu, tiêu chí kiểm bằng **so chuỗi** nên không cần nhãn người) — sparse hit@10 **0,0784 → 0,5098**, hit@1 0,0196 → 0,3529, McNemar `p = 4,8e-07`, và **0 mã nào dense tìm ra mà sparse không**. ⭐ **Con số đáng nhất: trần của `W2-04` là `hit_rate@10` 0,7033** vs dense 0,6268 — ghi lại trước khi làm RRF để lúc đó không tự diễn giải theo hướng có lợi. Chỗ sparse bù được: `en` 9↔4 (**sparse thắng** trên truy vấn tiếng Anh), `factoid` 10↔7; chỗ sparse chết hẳn: `cross_lingual` 1↔19 và **0 câu cả hai**. ⚠️ **Sparse học được KHÔNG phải index khớp đúng**: 25/51 mã không nhánh nào tìm ra, tokenizer giải thích hết (`VIE-01` → `['▁','VIE','-01']` có neo từ vựng và tìm được; `P171645` → `['▁P','171','645']` toàn chữ số chung và miss) → `TD-18`, giờ có bằng chứng chứ không phải phỏng đoán. ⚠️ Chi phí: tìm trong index sparse **97,8 ms vs dense 17,8 ms (5,5×)**, p50 toàn phần 30,2 → 113,4 ms; embed truy vấn 12,6 ms **dùng chung** cả hai nhánh. ⭐ **Ngoài DoD: đóng một hố im lặng trong `compare.py`** — harness lấy `fetch_doc_chunks` bằng `getattr`, nên retriever thiếu method đó thì nhãn rơi về nhãn cũ và hai lần chạy **cùng số nhãn nhưng khác nhãn** vẫn so được. Thêm `QueryScore.relevant_digest`; `compare.py` từ chối **toàn bộ** khi băm lệch (không lọc bỏ câu lệch rồi so phần còn lại — đó là tự chọn mẫu). Chạy thật: 209/209 có băm, **0 lệch**. **Hai giả định của tôi bị phản chứng và được giữ lại trong test**: "dense lẫn giữa các mã gần giống" (sai — trên 7 chunk cả hai đều đúng hạng 1; có test canh chính kết quả âm đó thay vì đổi assertion thành `rank_sparse <= rank_dense` để xanh) và "không trùng token thì sparse trả rỗng" (sai trên 15.814 chunk — sparse trả đủ 20 cho cả 209 câu; giới hạn thật là **xếp hạng sai**). +31 unit (697) + 14 integration (73) + 4 gpu (15). Evidence: `reports/w2-03-sparse-retriever.md`, `w2-03-known-item.json` |
| 2026-08-20 | **`W2-02` xong — một collection, hai loại vector.** `rag_bgem3` build lại với cả named vector `dense` (1024-d) và `sparse`, query độc lập qua `retrieve()` / `retrieve_sparse()`. **Sparse gần như miễn phí: +8,8 s trên 389 s (+2,3%)**, dung lượng +19% — hệ quả trực tiếp của việc `W2-01` cho hai loại vector ra từ **một** forward pass; gọi provider hai lần thì đây là +380 s. **Dense bit-identical sau khi build lại**: 15/15 metric không lệch một chữ số, 0/209 câu đổi điểm — kiểm tra tuyên bố "một đường code" của `W2-01` trên 15.814 chunk thật thay vì trên 4 câu test. Sparse thật: 95,9 entry/chunk (p50 100 · p95 147 · min **3**), mật độ 0,0384% của vocab 250.002, ReLU loại ~55% token. ⭐ **Ngoài DoD: `ensure_collection` giờ KIỂM TRA schema** thay vì thấy tồn tại là trả về — trước đó chạy provider sinh sparse lên collection dense-only sẽ chết ở *giữa* job 15.000 chunk. 4 ca lệch đều có test, trong đó ca "collection có sparse mà provider chỉ dense" là **hỏng im lặng**: eval ra số trông bình thường trong khi nửa index bị bỏ. `schema_problems()` thuần nên 12 ca test trong `make test`. Thêm `scripts/migrate_collection.py` — cố ý **không** migrate tại chỗ (Qdrant không cho thêm named vector), mà chẩn đoán lệch schema + in đúng lệnh phải chạy; mã trả về 0/1/2/3. `HashingEmbeddingProvider` nay sinh được sparse (mặc định **tắt** vì `name` là cache key của semantic chunker — để mặc định bật làm 2 test W1 đỏ ngay, đó là cách phát hiện). ⚠️ **Không dùng `Modifier.IDF`** cho BGE-M3: trọng số đã học, chồng IDF là nhân đôi phép hạ bậc và hỏng im lặng. ⚠️ Thang điểm hai nhánh không so được (cosine vs dot không trần) nên `W2-04` phải hợp nhất theo **thứ hạng**. +22 unit (666) + 21 integration (59). Evidence: `reports/w2-02-qdrant-hybrid.md` |
| 2026-08-20 | **`W2-01` xong — BGE-M3, và mức tăng lớn nhất của dự án tới giờ.** Đổi model embedding `vietnamese-bi-encoder` → `BAAI/bge-m3`, **giữ nguyên toàn bộ chunking của baseline**: nDCG@10 **0,1621 → 0,4442** (+174%), `hit_rate@5` 0,2153 → 0,5455, MRR 0,1660 → 0,4394, recall@10 0,2257 → 0,5813. **Cả 15 metric có ý nghĩa thống kê** (`p < 0,001` hoặc CI95 không chứa 0); ở `hit_rate@5` là 74 câu thắng vs 5 câu thua. `cross_lingual` recall@5 **0,0000 → 0,3023** — 43 câu (18% golden set) đi từ *không hoạt động* sang hoạt động. 5/6 nhóm truy vấn cải thiện có ý nghĩa. Truncation **56,9% → 0,0%**. Vì giữ `chunk_size=1000` nên nhãn **bit-identical** với baseline (`n_relevant_mean` 1,3828 cả hai bên) → recall@k/nDCG/MAP so được trực tiếp, `compare.py` không từ chối metric nào. ⚠️ **Mức tăng này là của model, KHÔNG phải của việc hết truncation**: `TD-11` đã tách riêng phần đó (`chunk550` cũng hết cắt) và nó cho `p = 0,711`; vai trò thật của cửa sổ 8192 là *cho phép giữ `chunk_size=1000`*, tức làm phép đo sạch. Hạ tầng thêm: `SparseVector` (bất biến cưỡng chế), `HybridVectors`, năng lực sparse tuỳ chọn trên `EmbeddingProvider` (mặc định `None` — `None` ≠ rỗng, bài học `TD-11` lần thứ hai), `BgeM3EmbeddingProvider` (dense + sparse **một** forward pass, có test canh dense khớp `SentenceTransformer.encode()` tới 1,5e-8), `embedding_max_batch_tokens` (knob VRAM, ngoài `fingerprint`), `make test-gpu`. ⚠️ **Sparse chưa đi vào Qdrant** — eval là dense-only, chờ `W2-02`. `G2` 2/4 điều kiện. +44 unit test (644 tổng) + 11 gpu test. Sửa lỗi sổ sách: p95 baseline là **32,8 ms** không phải 39,9 ms. Evidence: `reports/w2-01-bge-m3.md` |
| 2026-08-20 | **W2 bắt đầu — `TD-11` trả xong, kết quả ÂM.** Hạ `chunk_size` 1000 → 550 đưa truncation **56,9% → 0,4%** chunk (token mất 15,4% → 0,1%) và **không cải thiện gì đo được**: McNemar `p = 0,711`, mọi CI bootstrap chứa 0. Giả định của `TD-11` bị phản chứng — hạ `chunk_size` là **đánh đổi** (mỗi vector đọc 950 → 678 ký tự), không phải thu hồi nội dung. Ba thứ đổi cách làm phần còn lại của W2: (1) **không quét thêm `chunk_size`** với model đơn ngữ, đi thẳng `W2-01` BGE-M3 giữ `chunk_size=1000`; (2) **`golden_v1` chỉ phân giải được mức chênh ≥ 6 điểm `hit_rate`** nên mọi dòng ablation phải có `p`/CI — thêm `*-per-query.jsonl` + `pipeline/eval/compare.py` (McNemar exact + bootstrap cặp, không cần `scipy`); (3) **recall@k/nDCG/MAP không so được giữa hai `chunk_size`** vì nhãn neo theo span làm mẫu số đổi 1,38 → 1,96 nhãn/câu → tụt 29,6% kể cả khi truy hồi y nguyên, `compare.py` tự chặn. Thêm `make truncation` (đo `TD-11` tái lập được, chia theo ngôn ngữ: EN mất 19,4% token vs VI 7,5%) và `WARNING` truncation trong mọi lần build index. Thêm `TD-17` (1 tài liệu dùng `Ê` làm dấu cách). +66 unit test (600 tổng). Evidence: `reports/w2-td11-chunk-size.md` |
| 2026-08-20 | **Đóng sổ tuần 1, mở sổ tuần 2.** Điền cột **Baseline** ở §1 (recall@10 0,2257 · nDCG@10 0,1621 · MRR 0,1660 · p95 truy hồi 39,9 ms) và tách rõ metric *chưa đo* thuộc W4/W5 khỏi metric đã có số. `W0-04` `[?]` → `[x]` (đánh nhầm từ 2026-08-17, DoD đã đủ và đã dùng thật ở `W1-10`/`W1-13`). Đếm lại tổng task: **77** (71 backlog gốc + 6 `NEW-xx`), số cũ "73" là lỗi sổ sách vì `NEW-03`…`NEW-06` không được cộng vào. Chốt **thứ tự tấn công W2** vào §4: `TD-11` trước, `W2-01` sau, để cặp số trước/sau tách được phần của `chunk_size` khỏi phần của BGE-M3. Thêm `TD-13`…`TD-16` (review bằng model · 7 `reference_answer` lẫn ngữ cảnh · nhãn liên quan chưa đầy đủ · nhóm `unanswerable` thiếu đa dạng) |
| 2026-08-20 | **`W1-11` + `W1-13` xong — tuần 1 đủ 13/13.** `golden_v1` 242 câu (loại 24/266), baseline **recall@5 0,1746 · MRR 0,1660 · p95 39,9ms**, chạy lại 2 lần sai số **0,0000%**, chạy được với API key rỗng. ⚠️ Review bằng **model** (`reviewed_by_human=false`), không phải người — `G1` là PASS **có điều kiện**. Trả xong `TD-09` bằng cách tra thẳng văn bản gốc: 7/40 câu `unanswerable` sai nhãn. Bắt được `freeze` làm rơi `relevant_spans`. Thêm `reviewed_by` vào `GoldenQuery` và `--reviewer` vào `freeze`. 534 unit test. Evidence: `reports/w1-11-review.md` |
| 2026-08-20 | **`TD-12` đã trả — golden set neo theo span ký tự.** `TextSpan` + `Chunk.start_char/end_char`; `split_pieces()` thay `split_text()` làm API chính của Chunker; `pipeline/eval/spans.py` ánh xạ span → chunk_id của index đang đo; `pipeline/goldenset/anchor.py` chuyển 266 nháp sang span (299 span, 62% thu theo trích dẫn). Đo ở `chunk_size` 1000/600/400: **0 câu mất nhãn**, còn nhãn `chunk_id` cũ thì "hợp lệ" 226/226 ở cả ba mà trỏ vào văn bản khác. Bất biến: nội dung chunk khớp từng byte trước/sau (digest `b381634d51e39365`). Chạy `--recreate` để point mang offset (170s). +121 unit test (530), +5 integration (38). Evidence: `reports/w1-11-spans.md` |
| 2026-08-20 | **`W1-11` phần máy xong — triage + freeze, và ba phát hiện về retrieval.** Chạy retriever thật lên 266 câu nháp (24,3s): chỉ **65/226 = 28,8%** câu có chunk đã gán trong top-20. Điều tra ra ba nguyên nhân: (a) model embedding **đơn ngữ** → cùng ngôn ngữ 42,6% vs khác ngôn ngữ 7,8%, chênh 5,5× (`cross_lingual` trượt 95,7%); (b) giả thuyết "corpus gần trùng nên nhãn không đầy đủ" **bị loại** (0/78 câu có Jaccard ≥ 0,5); (c) **`TD-11`** — 56,8% chunk bị cắt ở 256 token, 15,7% text không tới được vector. Trả xong `TD-09` (15/40 câu unanswerable bị nghi). +71 unit test (409), +5 integration (38). Evidence: `reports/w1-11-triage.md` |
| 2026-08-20 | **`W1-09` xong — corpus đã version bằng DVC**: 60 tài liệu / 14,7 MiB, `dvc pull` trên clone sạch lấy đủ 61 file trong 1,9s và **60/60 sha256 khớp manifest**. Thêm `pipeline/corpus/dvc_state.py` canh lệch giữa hai cơ chế versioning + 23 test (317 → 338 unit). Remote đặt ở `.dvc/config.local` chứ không phải `.dvc/config`. Làm khác checklist: `data/golden` giữ trong git (lý lẽ ở `reports/w1-09-dvc.md` §3.2). Thêm `TD-10` |
| 2026-08-20 | **Dọn codebase + đổi tên repo**: bản POC chuyển vào `legacy/` (giữ lịch sử qua `git mv`), repo đổi tên `project_1.2_chatbot_rag` → `RAG-Chatbot`. Evidence: `reports/rename-workspace.md` |
| 2026-08-17 | **`W1-10` xong — 266 câu nháp golden set** ($0,5821 · 640s · 163 lời gọi). Thêm `packages/rag_core/llm/` và `pipeline/goldenset/`, 100 test mới. Phát hiện `deepseek-chat` chỉ là bí danh của `deepseek-v4-flash`; phát hiện model suy luận tiêu hết ngân sách token trước khi viết JSON; phát hiện 27,8% chunk bị trộn hai cột PDF. Thêm `NEW-05`, `NEW-06`, `TD-08`, `TD-09`; trả xong `TD-07`. `W0-04` hết bị chặn |
| 2026-08-17 | **`W1-08` xong — index baseline đã build**: 60 tài liệu → 15.814 chunk (768 chiều, cuda, 202s). Thêm `pipeline/indexing/` (config + corpus loader + build_index), `configs/indexing/{baseline,smoke}.yaml`, 59 test mới. Trả xong `TD-02`, `TD-03`; thêm `NEW-03`, `NEW-04`, `TD-06`, `TD-07`. Ghim torch CUDA (`cu126`) vì wheel PyPI trên Windows là bản CPU-only |
| 2026-08-17 | **Corpus nguồn (a) xong**: 60 tài liệu World Bank (40 EN + 20 VI) qua `scripts/fetch_corpus.py`. Thêm `pipeline/corpus/` (manifest ép giấy phép + adapter WDS) và 23 test. Phát hiện ADB chặn truy cập tự động (403) → phải tải tay qua `seed_list`. Thêm `NEW-02`, `TD-04`, `TD-05`; tổng 72 → 73 task |
| 2026-08-17 | **Hoàn thành 8/13 task tuần 1** (`W1-01`…`W1-07`, `W1-12`): 144 unit test + 18 integration test xanh, ruff + mypy strict pass, coverage `rag_core` 81%. Bằng chứng: `reports/w1-foundation.md`. Thêm `NEW-01`, `TD-01`…`TD-03`; tổng 71 → 72 task |
| 2026-08-17 | **Đổi yêu cầu Python từ ≥3.11 lên ≥3.12** — stub numpy dùng cú pháp `type` statement chỉ có từ 3.12, mypy strict không chạy được với `python_version=3.11`. Ảnh hưởng lựa chọn image ở `W0-05` |
| 2026-08-17 | **Bỏ LangChain khỏi `rag_core`** — viết lại recursive + semantic splitter thuần Python. Phụ thuộc nặng (`torch`, `qdrant-client`) chuyển sang optional extras để unit test chạy không cần GPU stack |
| 2026-08-14 | Tạo plan kỹ thuật `2026-08-14-rag-upgrade-proposal.md` (chốt 4 quyết định: LLM hybrid router · Docker Compose + HF Spaces · corpus VI–EN · 4–6 tuần) |
| 2026-08-14 | Thay diagram ASCII §2 bằng Mermaid + fallback text (ASCII bị vỡ khi trộn dấu tiếng Việt) |
| 2026-08-14 | Tạo `CHECKLIST.md` — 67 task, 6 gate, quy ước trạng thái & Definition of Done |
| 2026-08-14 | **Chuyển GPU thuê sang RunPod:** chốt Secure Cloud + RTX 4090 24GB + Pod On-Demand + Network Volume (tránh trả tiền GPU lúc tải trọng số). Spot chỉ dùng sau khi job resumable. Viết lại §3.6 mục GPU thuê; thêm rủi ro quên terminate pod |
| 2026-08-14 | **Bảo mật GPU thuê + corpus:** chốt quy tắc máy thuê không mang API key (job GPU-bound tự chứa); thứ tự nền tảng Kaggle → Colab → HF Jobs → vast.ai; bắt buộc corpus công khai (cấm tài liệu khách hàng Enigmas — rủi ro NDA). Thêm `W0-08`; siết `W0-03`, `W0-05`; tổng 69 → 70 task |
| 2026-08-14 | **Đổi nhà cung cấp LLM:** bỏ Claude (không có key) → DeepSeek primary + OpenRouter fallback + vLLM cho pipeline. Phát hiện GPU local là RTX 4060 8GB → không thể self-host generator ở serving. Thêm `W0-06`, `W0-07`, `W5-11` (generator ablation); tổng 67 → 69 task. Thêm §3.6 vào plan (phân vai LLM + ngân sách VRAM + 3 quy tắc tái lập) |
