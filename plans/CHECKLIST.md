# CHECKLIST — RAG Platform Upgrade

> **Đây là nguồn sự thật duy nhất về tiến độ.** Plan kỹ thuật: [`2026-08-14-rag-upgrade-proposal.md`](2026-08-14-rag-upgrade-proposal.md)
> Cập nhật lần cuối: 2026-08-20 · Trạng thái tổng: **tuần 1: 11/13 task xong · index baseline 15.814 chunk · 266 câu nháp đã triage, chờ review tay · corpus đã version bằng DVC · gate `G1` 2/4**
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
| W0 · Chuẩn bị | 8 | 1 | 0 | 1 | 3 | 3 | — |
| W1 · Nền móng + Eval baseline | 13 | 11 | 0 | 1 | 1 | 0 | `G1` ⬜ |
| W2 · Retrieval upgrade | 9 | 0 | 0 | 0 | 0 | 9 | `G2` ⬜ |
| W3 · Ingestion + Chunking | 9 | 0 | 0 | 0 | 0 | 9 | `G3` ⬜ |
| W4 · Serving Plane | 13 | 0 | 0 | 0 | 0 | 13 | `G4` ⬜ |
| W5 · Eval đầy đủ + Observability | 11 | 0 | 0 | 0 | 0 | 11 | `G5` ⬜ |
| W6 · Hoàn thiện & trình bày | 8 | 0 | 0 | 0 | 0 | 8 | `G6` ⬜ |
| **Tổng** | **73** | **12** | **0** | **2** | **6** | **53** | 0/6 |

Gate: ⬜ chưa chạy · 🟡 đã chạy FAIL · ✅ PASS

**Chỉ số chất lượng (điền dần, lấy từ `plans/reports/`)**

| Metric | Baseline (hệ thống hiện tại) | Hiện tại | Mục tiêu | Nguồn |
|---|---|---|---|---|
| Recall@10 | — | — | ≥ 0.90 | `reports/baseline.md` |
| nDCG@10 | — | — | ≥ 0.82 | — |
| MRR | — | — | ≥ 0.75 | — |
| Faithfulness | — | — | ≥ 0.92 | — |
| Citation accuracy | — | — | ≥ 0.85 | — |
| Refusal correctness | — | — | ≥ 0.85 | — |
| p95 latency | — | — | ≤ 3500 ms | — |
| Cost / query | — | — | ≤ $0.005 | — |
| Judge–human kappa | — | — | ≥ 0.6 | — |

> Cột **Baseline** phải được điền xong ở `W1-13` trước khi làm bất cứ việc tối ưu nào của W2.

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
- [?] `W0-04` **Chuẩn bị secrets & `.env.example`** (`DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, `HF_TOKEN`, DB/Qdrant creds)
  · DoD: `.env.example` đủ biến, `.env` trong `.gitignore`, startup fail-fast báo rõ biến nào thiếu · Test: `pytest tests/unit/test_settings.py` · Evidence: —
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
- [~] `W1-11` **Review tay → freeze `golden_v1` (≥150 câu)** — đủ 7 nhóm: factoid, multi_hop, aggregation, table_lookup, cross_lingual, unanswerable, adversarial
  · **Công cụ xong, đã chạy thật** (`reports/w1-11-triage.md`): `pipeline/goldenset/triage.py` (retriever thật → hàng đợi review) + `freeze.py` (quyết định → `golden_v1` + checksum + read-only). +71 unit test (409 tổng), +5 integration (38 tổng). `make goldenset-triage` chạy 24,3s
  · **Thứ tự đọc đã xếp sẵn** trong `queue_v1.md`: 15 câu `unanswerable_but_retrieved` → 16 câu `quote_unverified` → 2 câu `trivially_easy` → 161 câu `answerable_but_not_retrieved` (mặc định **accept**) → 72 câu không cờ. `gold_chunk_missing` = 0
  · ⚠️ **Bất đối xứng phải giữ đúng.** Câu `unanswerable` mà retriever tự tin = bằng chứng nhãn sai (mệnh đề về corpus bị phản chứng). Câu trả lời được mà retriever trượt **không** phải bằng chứng nhãn sai — đó là thứ eval tồn tại để đo, loại nó đi là tự thổi phồng recall baseline. Nên tín hiệu thứ hai xếp **cuối** hàng đợi, đề xuất `accept`, và có 2 test canh (`TestSignalB`)
  · Ngưỡng nghi ngờ **hiệu chuẩn từ dữ liệu**, không phải hằng số: trung vị điểm top-1 của các câu trả lời được = **0,5797**. Ghim `0.8` là đoán, vì "điểm cao" phụ thuộc model embedding và corpus
  · **Freeze không đoán hộ**: `fix_chunk_ids` mà người review để trống thì báo lỗi, không lấy top-1 của retriever điền vào — làm thế là dạy golden set trả lời đúng theo hệ thống hiện tại. Ba từ vựng tách rời: `suggested_decision` (máy hỏi) / `decision` (người trả lời: accept·reject·edit) / ô trống (chưa review, **không** phải accept)
  · Hai file, không phải một: `queue_v1.md` tối ưu cho **đọc** (toàn văn chunk đã gán + top-3 + trích dẫn, không phải tra cứu qua lại), `decisions_v1.csv` tối ưu cho **ghi**. `write_decisions_template` từ chối ghi đè — mất 6 giờ công review vì chạy lại một lệnh là chuyện không được xảy ra
  · DoD: mỗi câu có `relevant_chunk_ids` đã người xác nhận; file read-only, có checksum · Test: `tests/unit/test_goldenset_schema.py` (schema + phân bố category + không rỗng chunk_ids trừ nhóm unanswerable) · Evidence: `data/golden/golden_v1.jsonl` + `reports/goldenset-v1.md` (quy trình + phân bố)
- [x] `W1-12` **`pipeline/eval/retrieval_eval.py`** — Recall@{1,5,10,20}, Precision@k, MRR, nDCG@10, HitRate; breakdown theo category & language
  · DoD: bảng MD + JSON; metric đúng trên fixture có đáp án tính tay · Test: `tests/unit/test_retrieval_metrics.py` (**35 case**) + `tests/unit/test_retrieval_eval.py` (**19 case**) · Evidence: `reports/w1-foundation.md`
  · Kỳ vọng viết bằng công thức tường minh với thứ hạng ghi rõ; cố ý **không** gọi lại hàm đang test để sinh kỳ vọng
  · Có breakdown theo category & language. **Câu `unanswerable` trả `None` chứ không phải `0.0`** — recall trên tập rỗng là không xác định; quy ước thành 0 kéo tụt điểm vô nghĩa, thành 1 thì thổi phồng. Nhóm này đo riêng ở `W5-02`
  · Thiếu `query_id` trong kết quả = truy hồi rỗng (bị chấm 0), không phải bỏ qua — im lặng bỏ qua sẽ làm điểm cao lên một cách sai
  · ✅ Đã nối retriever thật ở `W1-08`: `make eval-retrieval` dùng `--index-config` và truy hồi trực tiếp từ Qdrant. `--retrieved` vẫn giữ để chấm lại kết quả cũ mà không phải chạy lại retrieval
- [?] `W1-13` ⭐ **Đo baseline hệ thống hiện tại** (chunking hybrid cũ + vietnamese-bi-encoder + dense k=5)
  · DoD: `reports/baseline.md` có đủ 8 metric + config + lệnh tái lập + thời gian chạy · Test: chạy lại 2 lần, sai số < 1% · Evidence: `plans/reports/baseline.md`

### `G1` — Gate tuần 1 ⬜ (2/4)
- [~] `make eval-retrieval BUNDLE=baseline` chạy được từ clone sạch bằng **1 lệnh** — đường ống đã thông (`make corpus && make up && make index && make eval-retrieval`), còn chặn bởi golden set thật (`W1-11`)
  · Kiểm chứng bằng golden set giả sinh từ chính chunk trong index (13 câu, **không** commit): `recall@5 0.9167 · p50 32.6ms · p95 97.7ms`. Đây **không phải** baseline — câu hỏi chính là text của chunk nên gần như chắc chắn tìm lại được chính nó
- [~] Retrieval eval chạy **không cần bất kỳ LLM API nào** (chỉ embedding + reranker local) — xác nhận bằng cách chạy với biến môi trường API key rỗng
  · Đã đúng ở mức code: `pipeline/eval/metrics.py` không import provider LLM nào, và `tests/unit/test_architecture_boundaries.py` canh phụ thuộc. Còn thiếu lần chạy thật với API key rỗng — làm cùng `W1-13`
- [?] §1 Dashboard đã điền xong cột **Baseline** — chặn bởi `W1-13`
- [x] Coverage `rag_core/` ≥ 70% → **80%** (chỉ tính unit test; `reports/w1-08-build-index.md`)

---

## 4. W2 · Retrieval upgrade

- [ ] `W2-01` **BGE-M3 provider** (dense + sparse lexical weights cùng lúc)
  · DoD: trả cả dense vector 1024-d và sparse dict; cache model load · Test: `tests/unit/test_bge_m3.py` (dim, sparse không rỗng, VI & EN) · Evidence: —
- [ ] `W2-02` **Qdrant named vectors + sparse index** + script migrate collection
  · DoD: 1 collection chứa cả dense & sparse, query được độc lập · Test: `tests/integration/test_qdrant_hybrid_schema.py` · Evidence: —
- [ ] `W2-03` **Sparse / BM25 retriever**
  · DoD: query từ khóa lạ (mã số, tên riêng) mà dense miss thì sparse hit · Test: `tests/integration/test_sparse_retriever.py` (có case exact-ID lookup) · Evidence: —
- [ ] `W2-04` **RRF fusion** (`k=60`, configurable)
  · DoD: deterministic, tie-break ổn định · Test: `tests/unit/test_rrf.py` — so với thứ hạng tính tay, case 1 list rỗng · Evidence: —
- [ ] `W2-05` **Cross-encoder reranker** `bge-reranker-v2-m3` (batch, top_n configurable)
  · DoD: rerank 50 → 6 trong < 400ms trên GPU; có CPU fallback · Test: `tests/unit/test_reranker.py` (thứ tự score giảm dần, batch == single) · Evidence: bench latency
- [ ] `W2-06` **Metadata filter** (tenant_id, doc_type, date range, lang)
  · DoD: filter áp ở tầng Qdrant (không post-filter), không rò dữ liệu chéo tenant · Test: `tests/integration/test_metadata_filter.py` — case tenant isolation · Evidence: —
- [ ] `W2-07` **Experiment Runner** — YAML config matrix → chạy tuần tự → MLflow tracking
  · DoD: 1 lệnh chạy hết grid, resume được khi crash giữa đường · Test: `tests/unit/test_experiment_runner.py` (expand grid đúng số tổ hợp, resume) · Evidence: MLflow UI screenshot
- [ ] `W2-08` **Ablation #1**: chunking × embedding × retrieval mode × rerank
  · DoD: ≥ 12 tổ hợp có kết quả đầy đủ, xác định được cấu hình thắng · Test: — · Evidence: MLflow run IDs
- [ ] `W2-09` **Report `reports/exp-001-retrieval.md`** — bảng số vs baseline + phân tích tại sao thắng/thua
  · DoD: có bảng delta, có ít nhất 2 nhận xét về *category nào cải thiện nhiều nhất* · Test: — · Evidence: file report

### `G2` — Gate tuần 2 ⬜
- [ ] nDCG@10 tăng so với baseline (kỳ vọng ≥ +0.08 tuyệt đối)
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
| `W0-01` | Bạn đọc lại plan sau khi sửa diagram | Bạn | 2026-08-14 | Chặn toàn bộ W1 |
| `W0-02` | Quyết định tên repo mới + có tạo repo riêng hay nâng cấp in-place | Bạn | 2026-08-14 | Code đã push lên nhánh `feat/w1-foundation` của repo cũ (`f5ec22b`, `c253fa5`). `main` chưa đụng tới vì đang là link trong CV. Cần quyết: merge vào `main` hay tách repo mới |
| `W0-03` | Nguồn (b) văn bản pháp luật ~30 và (c) báo cáo HOSE ~30 — cần chọn tay rồi khai báo qua `seed_list` | Bạn | 2026-08-14 | Nguồn (a) xong (60 tài liệu, đã index). Thiếu (b)/(c) thì golden set không có nhóm `table_lookup` và `section_path` |
| `W0-07` | `@preset/my-luna-pro` resolve ra model slug nào (cần biết để chọn judge cross-check khác họ DeepSeek) | Bạn | 2026-08-14 | Chặn `W5-04`, không chặn W1–W4 |

---

## 11. Nợ kỹ thuật ghi nhận

> Những chỗ cố tình làm tạm để đi nhanh. Ghi lại để không tự lừa mình.

| ID | Nợ | Lý do chấp nhận | Kế hoạch trả |
|---|---|---|---|
| `TD-01` | `HuggingFaceEmbeddingProvider` coverage 0% | Cần `torch` (~2.5GB) mà unit test cố ý chạy không có GPU stack để `make test` giữ ở ~3 giây | Test đánh dấu `@pytest.mark.gpu` khi làm `W0-06` (đo VRAM) và `W2-01` (BGE-M3) |
| ~~`TD-02`~~ | ~~CLI `retrieval_eval.py` chưa nối retriever thật~~ | **Đã trả ở `W1-08`**: `--index-config` dựng retriever từ chính config đã build index | ✅ |
| ~~`TD-03`~~ | ~~`pipeline/indexing/` còn rỗng~~ | **Đã trả ở `W1-08`** | ✅ |
| `TD-04` | Bộ tải corpus loại 5 tài liệu vì "không giải mã được UTF-8" | Một số bản `.txt` của World Bank mã hoá cp1252. Loại bỏ là an toàn, và 60 tài liệu đã đủ cho nguồn (a) | Thử fallback cp1252 → utf-8 khi cần thêm tài liệu |
| `TD-05` | Giấy phép CC BY 3.0 IGO là khai báo **ở mức nguồn**, không kiểm chứng từng tài liệu | Script chỉ ép được rằng giấy phép nằm trong danh sách cho phép | Đọc tay trang giấy phép của ~5 tài liệu bất kỳ **trước khi** push repo lên public |
| `TD-06` | Toàn bộ `Document` nằm trong RAM khi build index | 60 tài liệu = 14 MB, tối ưu sớm là lãng phí | `W3-07` (re-index tăng dần) sẽ đọc theo luồng |
| ~~`TD-07`~~ | ~~`plans/` bị thêm vào `.gitignore`~~ | **Đã xử lý**: bỏ dòng đó, vì DoD của dự án bắt buộc mỗi task có đường dẫn Evidence | ✅ |
| `TD-08` | 22/163 lời gọi LLM bị cắt ở `max_tokens=6000` — toàn bộ ngân sách đi vào chuỗi suy luận | Vẫn đủ 266 câu; nâng tiếp là trả tiền cho phần suy luận không dùng tới | Thử `--questions-per-call 1` (prompt ngắn → suy luận ngắn) khi cần thêm câu |
| `TD-10` | DVC remote chỉ là thư mục local `D:/dvc-remote/rag-chatbot` | Nguồn (a) còn đường phục hồi độc lập: `scripts/fetch_corpus.py` tải lại từ World Bank rồi so sha256. Mất remote chưa mất corpus | **Bắt buộc dựng remote dùng chung trước khi thêm nguồn (b)/(c)** — văn bản pháp luật + báo cáo HOSE phải chọn tay, không script nào tải lại được, mất remote là mất corpus |
| ~~`TD-09`~~ | ~~Chưa kiểm được câu `unanswerable` có thật sự không trả lời được~~ | **Đã trả**: `triage.py` chạy retriever lên cả 40 câu, **15 câu** vượt ngưỡng hiệu chuẩn 0,5797 → đầu hàng đợi review | ✅ Nhưng kế hoạch cũ phải sửa: điểm cao **không** đủ để phân loại lại tự động. Ví dụ đầu hàng đợi có điểm 0,7287 mà chunk top-1 vẫn không trả lời được câu hỏi — điểm cao chứng minh *cùng chủ đề*, không chứng minh *trả lời được* |
| `TD-11` ⭐ | **56,8% chunk bị cắt lúc embed** — `chunk_size=1000` **ký tự** ≈ 340 token, `vietnamese-bi-encoder` (PhoBERT) có `max_seq_length=256` **token**. `sentence-transformers` cắt âm thầm, không cảnh báo. **15,7% toàn bộ text đem embed không bao giờ tới được vector** (mẫu ngẫu nhiên 1200 chunk, seed 20260820; riêng chunk đã gán thì 91,1% vì bộ lọc prose-like chọn chunk dài) | **Cố ý không sửa trước `W1-13`**: đây là hành vi thật của bản POC, sửa trước rồi gọi kết quả là "baseline" thì mọi so sánh về sau đo lẫn cải tiến này vào. Chiều câu hỏi an toàn: p50 = 44 token, 0/266 bị cắt | **Hạng mục đầu tiên của `W2`** vì rẻ nhất: hạ `chunk_size` xuống ~600 ký tự, hoặc đổi sang model cửa sổ dài (BGE-M3 8192). Đo lại ngay sau `W1-13` để có cặp số trước/sau |

---

## 12. Changelog

| Ngày | Thay đổi |
|---|---|
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
