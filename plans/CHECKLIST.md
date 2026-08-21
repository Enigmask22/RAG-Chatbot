# CHECKLIST — RAG Platform Upgrade

> **Đây là nguồn sự thật duy nhất về tiến độ.** Plan kỹ thuật: [`2026-08-14-rag-upgrade-proposal.md`](2026-08-14-rag-upgrade-proposal.md)
> Cập nhật lần cuối: 2026-08-21 · Trạng thái tổng: **tuần 1: 13/13 task xong · `golden_v1` 242 câu (review bằng MODEL, không phải người) · baseline recall@5 = 0,1746 · gate `G1` 4/4 ⚠️ với một điều kiện · `W2` 7/9 · 1096 test**
> **W2 đang chạy — `TD-11` + `W2-01`…`W2-07` xong (7/9).** `TD-11`: hạ `chunk_size` đưa truncation 56,9% → 0,4% mà **không cải thiện gì đo được** (`p = 0,711`) — giả định của nợ đó bị phản chứng. `W2-01`: BGE-M3 (giữ `chunk_size=1000`) đưa **nDCG@10 0,1621 → 0,4442** và `cross_lingual` recall@5 **0 → 0,3023**, cả 15 metric đều có ý nghĩa thống kê. ⚠️ Mức tăng đó là của **model**, KHÔNG phải của việc hết truncation — `TD-11` đã đo riêng phần đó và nó cho `p = 0,711`. `W2-02`: `rag_bgem3` giờ mang cả named vector `dense` và `sparse` trong một collection, và `ensure_collection` kiểm tra schema thay vì tin. `W2-03`: sparse đã đo được — **kém dense trên golden set** (nDCG@10 0,4442 → 0,3733) nhưng **thắng áp đảo ở tra mã tài liệu** (hit@10 0,0784 → 0,5098, `p = 4,8e-07`, 22↔**0**). Con số đáng nhất: **trần của RRF là `hit_rate@10` 0,7033** vs dense 0,6268. `W2-04`: RRF — **`k=60` của bài báo là lựa chọn TỆ NHẤT** (kém dense có ý nghĩa); `k=1` thắng (`hit_rate@10` 0,6268 → 0,6555) nhưng chỉ 3/15 metric đạt ý nghĩa. Và phân rã độ trễ tìm ra **bug 64 ms/lần gọi** làm sai hai con số đã công bố ở `W2-02`/`W2-03` — sparse thật ra **miễn phí hoàn toàn**, cả phía ghi lẫn phía đọc. `W2-05`: cross-encoder reranker — **mức cải thiện lớn nhất của W2 sau chính BGE-M3**. `hit_rate@1` **0,3397 → 0,5598 (+22,0 điểm)**, nDCG@10 0,4563 → 0,6481, **15/15 metric có ý nghĩa** (`hit_rate@5` từ dense: **0↔43** câu — 43 câu được sửa, 0 câu bị làm hỏng). Trần vùng phủ đo trước là `hit_rate@50` = **0,7799**, nên reranker lấy **một nửa** dư địa 44 điểm. ⚠️ **DoD 400 ms KHÔNG đạt**: 50 cặp tốn 524–529 ms trên 4060 Laptop; 400 ms mua được pool 37. ⭐ Hai kết quả kiến trúc: **sau khi có reranker, tầng hybrid không còn đo được** (13/15 metric nhiễu) và **`TD-18` là bài toán truy hồi chứ không phải biểu diễn** (known-item hit@1 0,0980 → **0,5490**, thắng cả sparse). ⚠️ **Bổ sung khi tổng kết (2026-08-21)**: kiểm định theo category — chưa từng chạy vì `compare.py` không có chiều đó — cho thấy **nhánh hybrid của `W2-04` làm tụt `cross_lingual` CÓ Ý NGHĨA** (recall@5 0,3023 → 0,2093, CI95 [−0,1860, −0,0233], `hit_rate@5` **4↔0**), tức lấy lại gần ⅓ kết quả tiêu đề của `W2-01` trong khi bảng tổng vẫn xanh. Reranker **vá lại đúng phần đó** (còn 1↔0, `p = 1,000`), nên "hybrid miễn phí" chỉ đúng ở dạng **miễn phí và vô hại KHI CÓ reranker phía sau** — và chế độ suy giảm "mất GPU thì lùi về hybrid" là SAI. `W2-06`: metadata filter — phần thiếu **không** phải date range mà là **đường `fetch`**: `fetch_chunks`/`fetch_doc_chunks` không có tham số filter, và đó đúng là hai method `W4-09`/`W4-06` sẽ gọi để giải citation. Lọc **không tốn gì** (tám ca 20,5%–100% độ chọn lọc nằm trong 29,98–30,54 ms), nhưng phải **phân rã** phép đo mới thấy — hai lượt đầu cho một bảng đơn điệu thuyết phục mà là nhiễu của bước embed. `W2-07`: experiment runner — một lệnh chạy 14 ô, resume theo **`fingerprint` của ô** (không theo tên file báo cáo, vì "đã chạy" và "đã chạy *với đúng tham số hiện tại*" là hai câu khác nhau). ⭐ Grid tái lập **5** con số đã công bố **đúng từng chữ số**, tức chứng minh đợt refactor `IndexSession` không đổi con số nào. ⭐ Preflight bắt ngay ở lần `--dry-run` đầu một ô sắp **ghi đè bằng chứng `W2-03`**. ⚠️ **MLflow hỏng hai lần**, cả hai kiểu "chạy xong, đúng số ô, không có dữ liệu": mlflow 3 bỏ file store (14 ô chạy mà không ghi gì), và MLflow không nhận `@` trong tên metric (14 run, **0 cột metric**) — bài học: **khoan dung đúng với lỗi nhất thời và sai với lỗi hệ thống**. `make exp-backfill` dựng lại view MLflow từ file báo cáo, tức *kiểm* câu "MLflow là view không phải nguồn sự thật" thay vì tuyên bố nó. Việc tiếp theo: **`--category`/`--lang` cho `compare.py`**, rồi `W2-08`. Xem §4 và `reports/tasks/w2-td11-chunk-size.md`, `reports/tasks/w2-01-bge-m3.md`, `reports/tasks/w2-02-qdrant-hybrid.md`, `reports/tasks/w2-03-sparse-retriever.md`, `reports/tasks/w2-04-rrf.md`, `reports/tasks/w2-05-reranker.md`, `reports/tasks/w2-06-metadata-filter.md`, `reports/tasks/w2-07-experiment-runner.md`
> Nhật ký phiên làm việc (để nối tiếp khi ngắt giữa chừng): [`WORKLOG.md`](WORKLOG.md)
> Bản đồ báo cáo (**vào đây để review một `Wx-xx` đã làm gì**): [`reports/README.md`](reports/README.md) — 5 folder chia theo *ai sinh ra file*, kèm bảng `Wx-xx` → tường thuật + lần chạy + kiểm định + đo lẻ, và danh sách chỗ còn hổng

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
| W2 · Retrieval upgrade | 9 | 7 | 0 | 0 | 0 | 2 | `G2` ⬜ |
| W3 · Ingestion + Chunking | 9 | 0 | 0 | 0 | 0 | 9 | `G3` ⬜ |
| W4 · Serving Plane | 13 | 0 | 0 | 0 | 0 | 13 | `G4` ⬜ |
| W5 · Eval đầy đủ + Observability | 11 | 0 | 0 | 0 | 0 | 11 | `G5` ⬜ |
| W6 · Hoàn thiện & trình bày | 8 | 0 | 0 | 0 | 0 | 8 | `G6` ⬜ |
| **Tổng backlog gốc** | **71** | **21** | **0** | **2** | **2** | **47** | 0/6 |
| §9 Task thêm mới (`NEW-xx`) | 6 | 6 | 0 | 0 | 0 | 0 | — |
| **Tổng cộng** | **77** | **27** | **0** | **2** | **2** | **47** | 0/6 |

Gate: ⬜ chưa chạy · 🟡 đã chạy FAIL · ✅ PASS

> ⚠️ Con số tổng cũ ("73") là lỗi sổ sách tích lại: `NEW-03`…`NEW-06` được thêm vào §9 mà
> không cộng vào tổng. Đã đếm lại trực tiếp từ các mục §2–§8 (**71**) và §9 (**6**).
>
> ⚠️ **Lỗi sổ sách thứ hai, cùng loại** (sửa 2026-08-21): `W2-05` được tick `[x]` ở §4 nhưng
> dòng W2 ở đây vẫn ghi 4 done, nên tổng cộng đếm thiếu một task. Quy tắc §0 nói rõ "đổi
> trạng thái → cập nhật luôn §1 Dashboard" và tôi đã bỏ nửa sau. Cách kiểm không tốn gì:
> `grep -cE '^- \[x\]'` trên đúng khoảng dòng của từng mục §2–§8.

**Chỉ số chất lượng (điền dần, lấy từ `plans/reports/`)**

| Metric | Baseline (hệ thống hiện tại) | Hiện tại | Mục tiêu | Nguồn |
|---|---|---|---|---|
| Recall@10 | **0,2257** | **0,7352** | ≥ 0.90 | `reports/runs/bgem3-rr-c50-retrieval.json` |
| Recall@5 | **0,1746** | **0,7026** | — | `reports/runs/bgem3-rr-c50-retrieval.json` |
| nDCG@10 | **0,1621** | **0,6481** | ≥ 0.82 | `reports/runs/bgem3-rr-c50-retrieval.json` |
| MRR | **0,1660** | **0,6440** | ≥ 0.75 | `reports/runs/bgem3-rr-c50-retrieval.json` |
| MAP@20 | **0,1349** | **0,6051** | — | `reports/runs/bgem3-rr-c50-retrieval.json` |
| hit_rate@1 | **0,1196** | **0,5598** | — | `reports/runs/bgem3-rr-c50-retrieval.json` |
| Faithfulness | *chưa đo* | — | ≥ 0.92 | `W5-01` |
| Citation accuracy | *chưa đo* | — | ≥ 0.85 | `W5-02` |
| Refusal correctness | *chưa đo* | — | ≥ 0.85 | `W5-02` (33 câu `unanswerable`) |
| p95 latency (truy hồi) | **32,8 ms** | **604,0 ms** | — | `reports/runs/bgem3-rr-c50-retrieval.json` |
| p50 latency · nhánh hybrid | — | **31,3 ms** (= dense) | — | `reports/runs/bgem3-rrf-k1-c20-retrieval.json` |
| p50 latency · nhánh reranked | — | **534,4 ms** (c=50) · **232,8 ms** (c=20) | — | `reports/tasks/w2-05-reranker.md` §5 |
| p95 latency (end-to-end) | *chưa đo* | — | ≤ 3500 ms | `W4-13` / `W6-05` |
| Cost / query | *chưa đo* | — | ≤ $0.005 | `W5-11` |
| Judge–human kappa | *chưa đo* | — | ≥ 0.6 | `W5-04` |

> ✅ Cột **Baseline** đã điền ở `W1-13` (2026-08-20) — điều kiện để bắt đầu W2.
> Năm metric xếp hạng đo trên **209/242 câu** của `golden_v1`; 33 câu `unanswerable` trả
> `None` ở mọi metric xếp hạng và được đo riêng bằng refusal correctness (`W5-02`).
> **p95 là độ trễ truy hồi thuần** (embed câu hỏi + tìm trong Qdrant, sau warm-up), không
> phải end-to-end — chỉ so được với ngưỡng 3500 ms sau `W4-13`.
> ⚠️ Con số baseline cũ "39,9 ms" là lỗi sổ sách; `reports/runs/baseline-retrieval.md` ghi **32,8 ms**.
> Cột **Hiện tại** = **`bgem3-rr-c50`** (`W2-05`, 2026-08-21): BGE-M3 + hybrid RRF `k=1` +
> cross-encoder rerank trên pool 50. Đo trên cùng 209 câu và **cùng nhãn**
> (`n_relevant_mean` 1,3828 ở mọi lần chạy từ `W2-01` trở đi), nên recall@k/nDCG/MAP so được
> trực tiếp với baseline.
> ⚠️ Cột này **đứng lại ở `W2-01` suốt `W2-04` và `W2-05`** (sửa 2026-08-21). Hệ quả không
> phải mỹ quan: bảng này là chỗ duy nhất trong repo trả lời "đang ở đâu so với mục tiêu", và
> đọc nó thì nDCG@10 còn cách ngưỡng `G6` 0,82 một khoảng 0,38 trong khi thật ra chỉ còn 0,17.
> **`c=50` không phải cấu hình nhanh nhất cũng không phải tốt nhất** — nó là cấu hình `W2-05`
> báo cáo chính. `c=100` tốt hơn ở mọi metric (nDCG@10 **0,6736**, Recall@10 **0,7679**) nhưng
> tốn 1044 ms; `c=20` giữ 91% mức lợi hạng nhất với 233 ms và là điểm vận hành khuyến nghị cho
> `W4`. Chốt một con số vào bảng mà không nói kèm `candidates` là bỏ mất chiều đắt nhất.
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
  · DoD: file mẫu `.env.example` đủ biến ✅ · file thật nằm trong `.gitignore` ✅ · startup fail-fast báo **tất cả** biến thiếu trong một lần thay vì lần lượt ✅ · Test: `tests/unit/test_settings.py` — **8 case**, gồm `test_secret_not_in_repr` và `test_password_not_in_repr_but_in_dsn` (giá trị bí mật không lọt vào log/repr nhưng vẫn vào được DSN) · Evidence: `reports/tasks/w1-foundation.md`
  · Đã dùng thật ở `W1-10` (gọi DeepSeek, $0,5821) và ở `W1-13` (chạy với key **rỗng** để chứng minh eval truy hồi không cần LLM API). Trước đây đánh `[?]` là **nhầm** — đã hết bị chặn từ 2026-08-17, changelog có ghi mà quên đổi ký hiệu
- [ ] `W0-05` **Dựng môi trường RunPod** — Secure Cloud, **RTX 4090 24GB**, Pod On-Demand, + **Network Volume** gắn `/workspace` với `HF_HOME=/workspace/.hf`
  · DoD: pod khởi động lại lần 2 **không phải tải lại trọng số**; chạy thử load `bge-m3` + `bge-reranker-v2-m3` + vLLM `Qwen3-8B`, ghi VRAM & thời gian · Test: — · Evidence: `reports/tasks/gpu-env.md`
  · Chỉ 3 task cần GPU thuê: `W3-04`, `W5-11`, fine-tune (tùy chọn). Tổng ~10–20 GPU-hour ≈ vài chục USD trở xuống.
  · Đặt **spending limit** trên tài khoản RunPod ngay từ đầu. Network volume gắn với 1 datacenter → chọn region rồi giữ nguyên.
  · Chuyển sang **Spot** để tiết kiệm **chỉ sau khi** `W3-04` và `W2-07` đã resumable.
- [ ] `W0-06` ⭐ **Đo ngân sách VRAM thật trên RTX 4060 8GB** + xác minh vLLM chạy qua Docker Desktop + WSL2 GPU passthrough
  · DoD: biết chính xác VRAM còn dư sau khi load bge-m3 + reranker; xác nhận được/không thể chạy thêm generator local · Test: `python scripts/vram_budget.py` in ra bảng · Evidence: `reports/tasks/vram-budget.md`
  · *Nếu VRAM dư < 3GB (kỳ vọng đúng như vậy) → chốt: serving dùng API generator, vLLM chỉ bật theo phiên trên GPU thuê.*
- [?] `W0-07` **Làm rõ `@preset/my-luna-pro`** — preset này resolve ra model slug nào, params gì
  · DoD: biết slug thật; chọn thêm 1 slug **khác họ DeepSeek** trên OpenRouter làm judge cross-check · Test: gọi 1 request, đọc field `model` trong response · Evidence: ghi vào `reports/tasks/llm-providers.md`
  · *Quy tắc: preset chỉ dùng cho demo tương tác, **cấm dùng trong eval** (config phía server có thể đổi ngầm).*
- [ ] `W0-08` ⭐ **Kênh chuyển dữ liệu an toàn cho RunPod + vệ sinh secret**
  · DoD: script `scripts/runpod_job.sh` đẩy corpus lên / kéo kết quả về bằng **HF private dataset repo + fine-grained token scoped 1 repo** (hoặc `runpodctl send/receive`); **không** copy `.env`; không cấu hình git credential trên pod · Test: `pytest tests/security/test_no_secret_in_job_bundle.py` — quét tarball job, fail nếu thấy pattern API key · Evidence: `reports/tasks/remote-gpu-hygiene.md`
  · Quy tắc: **pod chỉ chạy job GPU-bound tự chứa, không mang API key** (kể cả RunPod Secrets). SSH bằng keypair riêng cho RunPod. Thu hồi token ngay sau job; terminate pod khi xong.

---

## 3. W1 · Nền móng + Eval baseline

> Mục tiêu tuần: **có con số baseline tái lập bằng 1 lệnh.** Chưa tối ưu gì cả.

- [x] `W1-01` **Monorepo skeleton** — `packages/rag_core`, `pipeline/`, `serving/`, `apps/`, `tests/`; `pyproject.toml` (uv), ruff + mypy + pytest, pre-commit, `Makefile`
  · DoD: `make lint && make test` exit 0 · Test: `make lint`, `make test` · Evidence: `reports/tasks/w1-foundation.md` — ruff + mypy strict pass, 144 unit test
  · Đã dựng `packages/rag_core`, `pipeline/`, `serving/`, `apps/`, `infra/`, `configs/`, `tests/{unit,integration,security,e2e}`; uv + ruff + mypy strict + pytest + pre-commit + Makefile
  · ⚠️ **Yêu cầu Python ≥ 3.12** (không phải 3.11): stub numpy dùng cú pháp `type` chỉ có từ 3.12, mypy strict không chạy được với 3.11. Ảnh hưởng việc chọn image ở `W0-05`
- [x] `W1-02` **Pydantic schemas (hợp đồng dữ liệu)** — `Document`, `Chunk`, `RetrievedChunk`, `Citation`, `Answer`, `QueryRequest`
  · DoD: schema có validation + serialize/deserialize round-trip · Test: `tests/unit/test_schemas.py` — **20 case** · Evidence: `reports/tasks/w1-foundation.md`
  · `extra="forbid"` toàn bộ (field gõ sai tên bị nuốt = bug âm thầm trong index đã build); `Document`/`Chunk` bất biến; `DocumentMetadata` **bắt buộc** `source_url` + `license` — ràng buộc corpus công khai ép ngay ở tầng schema nên không thể quên khi thêm tài liệu
- [x] `W1-03` **Port `enhanced_chunking.py` → `rag_core/chunking/`** (fixed, semantic, hybrid) sau interface chung `Chunker.chunk(docs) -> list[Chunk]`
  · DoD: 3 strategy dùng chung interface, không import `streamlit` · Test: `tests/unit/test_chunking.py` — **35 case**, có doc rỗng, doc 1 câu, doc ~50 trang (< 5s) · Evidence: `reports/tasks/w1-foundation.md`
  · **Bỏ LangChain**: viết lại `RecursiveCharacterTextSplitter` + `SemanticChunker` thuần Python (~150 dòng) → test được từng nhánh, không phụ thuộc `langchain-experimental`
  · ⚠️ **4 sai lệch có chủ ý so với bản POC, ảnh hưởng số baseline `W1-13`** — chi tiết ở `reports/tasks/w1-foundation.md`. Quan trọng nhất: `neighbor_context_chars` mặc định **tắt**, config baseline phải đặt lại **100** mới tái lập đúng hệ thống hiện tại
- [x] `W1-04` **Thay cache `pickle` → SQLite content-hash cache** (an toàn, portable, có TTL + eviction)
  · DoD: hit/miss đúng khi đổi content hoặc đổi config; không dùng `pickle.load` · Test: `tests/unit/test_chunk_cache.py` — **16 case**: hit, miss theo content/config/chunker, TTL, LRU eviction, entry hỏng, file DB hỏng · Evidence: `reports/tasks/w1-foundation.md`
  · Có test quét AST toàn bộ `rag_core` chặn mọi `import pickle` quay lại
  · **Cache theo từng tài liệu**, không theo cả corpus (bản POC hash chuỗi nối của mọi tài liệu → sửa 1 file là mất sạch). Đây là nền cho `W3-07`
- [x] `W1-05` **`docker-compose.yml`: Qdrant + Postgres + Redis** + `make up` / `make down` + healthcheck
  · DoD: `make up` → cả 3 service healthy < 60s · Test: `tests/integration/test_infra_up.py` — **4 case**, ping bằng **giao thức thật** của từng service (không chỉ mở TCP socket) · Evidence: `reports/tasks/w1-foundation.md`
  · Tìm ra 2 lỗi chỉ lộ khi chạy thật: (a) `QDRANT__SERVICE__API_KEY` rỗng vẫn bật xác thực → 401 mọi request; (b) `localhost` trên Windows resolve `::1` trước → **mỗi request Qdrant chậm đúng 2 giây**; đổi sang `127.0.0.1` thì integration suite từ ~5 phút còn 30 giây
- [x] `W1-06` **Embedding provider abstraction** — `EmbeddingProvider` interface + impl HF local; baseline `vietnamese-bi-encoder`
  · DoD: đổi model chỉ qua config, không sửa code; batch + normalize configurable · Test: `tests/unit/test_embedding.py` — **14 case** · Evidence: `reports/tasks/w1-foundation.md`
  · Thêm `HashingEmbeddingProvider` để unit test chạy không cần `torch`. Cố ý **không** dùng fake trả vector ngẫu nhiên: test semantic chunking cần tương đồng từ vựng thật, vector ngẫu nhiên sẽ khiến test pass kể cả khi thuật toán sai
- [x] `W1-07` **Dense retriever trên Qdrant** — upsert + search + payload
  · DoD: index chunk, query trả top-k đúng thứ tự score · Test: `tests/integration/test_qdrant_dense.py` — **14 case**: named vector, thứ hạng liên tục, score giảm dần, idempotent, filter theo lang/tenant/doc_type, delete · Evidence: `reports/tasks/w1-foundation.md`
  · **Named vector `dense` ngay từ W1** — collection dùng vector vô danh không thêm được named vector mà không build lại toàn bộ index, mà `W2-02` sẽ thêm sparse
  · **Point ID = UUIDv5 sinh từ `chunk_id`** → tính idempotent nằm ở tầng store chứ không ở script build index; `W1-08` thừa hưởng sẵn
- [x] `W1-08` **`pipeline/indexing/build_index.py`** — corpus → chunk → embed → Qdrant collection, idempotent
  · DoD: chạy 2 lần không sinh duplicate; log số doc/chunk/thời gian · Test: `tests/integration/test_build_index.py` — **15 case** + `test_index_config.py` **28 case** + `test_corpus_loader.py` **16 case** · Evidence: `reports/tasks/w1-08-build-index.md` + `reports/probes/index-baseline.json`
  · **Index baseline đã build**: 60 tài liệu → **15.814 chunk**, 768 chiều, trên `cuda`, 202s. Chạy lần hai: `index 0 · bỏ qua 60`, count không đổi
  · **Ba tầng idempotent**, point ID xác định của `W1-07` chỉ là tầng 1: (2) nhớ số chunk cũ để xoá phần đuôi thừa khi tài liệu **ngắn lại** — point mồ côi không trùng lặp với gì cả, nó chỉ trỏ tới văn bản không còn tồn tại; (3) so `fingerprint` để chặn trộn hai cấu hình vào một collection
  · **Bắt được 3 lỗi chỉ lộ khi chạy thật**: (a) `HybridChunker` luôn chọn semantic vì cache chia lô thành 1 tài liệu → sửa bằng `Chunker.prepare(n)`; (b) `HybridChunker.name` không phân biệt nhánh → hai lần chạy đọc nhầm cache của nhau; (c) **p95 độ trễ 15.219 ms là thời gian nạp model**, không phải truy hồi → thêm warm-up, p95 còn 98 ms
  · Đo được **hệ số 1.24x**: `neighbor_context_chars=100` của bản POC làm text đem embed phình từ 14,3 lên 17,7 triệu ký tự
  · `IndexConfig` là tiền thân của `RagBundle` (`W4-01`): `fingerprint` băm đúng thứ quyết định vector, **không** gồm `device`/`batch_size` để đổi máy không phải build lại — có test canh cả hai chiều
- [x] `W1-09` **DVC init + version corpus** — `data/corpus` (60 tài liệu, md5 `9aeb1b77…`, 14,7 MiB)
  · DoD: `dvc status` sạch ✅ · remote local hoạt động ✅ · Test: `tests/unit/test_dvc_state.py` **23 case** + phép thử clone sạch chạy thật · Evidence: `.dvc/` committed + `reports/tasks/w1-09-dvc.md`
  · **Clone sạch → `dvc pull` → 61 file trong 1,9s**, rồi so sha256 từng tài liệu với manifest: **60/60 khớp**. Đây là bằng chứng mạnh hơn `dvc status`, vì nó chứng minh nội dung DVC phục hồi là byte-identical với thứ tải từ World Bank
  · **Remote KHÔNG nằm trong `.dvc/config`** (file được commit) mà ở `.dvc/config.local`. Một URL `D:/...` commit vào config dùng chung thì mọi clone nhận remote trỏ vào ổ đĩa không tồn tại — cùng loại lỗi với đường dẫn tuyệt đối trong `.venv/*.pth` vừa gặp lúc đổi tên workspace
  · ⚠️ **Làm khác checklist: `data/golden` giữ trong git, không đưa vào DVC.** Golden set là thước đo, thứ cần nhất ở nó là diff đọc được lúc review (ai đổi nhãn câu nào, từ gì sang gì); DVC thay file bằng một hash nên mất đúng thứ đó. 284 KB text không phải lý do để tránh git. Tính tái lập không mất: một commit ghim cả hai. Lý lẽ đầy đủ ở report §3.2
  · Corpus giờ có **hai** cơ chế versioning (sha256/manifest + md5/DVC) và chỗ cả hai đều mù là phép so **số lượng**. `pipeline/corpus/dvc_state.py` canh chỗ đó: thêm file vào `data/corpus/` rồi `dvc add` mà quên manifest thì không lỗi nào nổ ra — build index bỏ qua file lạ, `dvc status` vẫn sạch, `dvc push` vẫn đem nó lên remote
  · Bắt được lúc chạy thật: `load_manifest` trả `[]` cho đường dẫn không tồn tại (chủ ý, `fetch_corpus.py` cần thế ở lần đầu) → gõ sai đường dẫn manifest sẽ báo thành "hai cơ chế lệch nhau", dẫn đi sai hướng. Đã thêm guard riêng
- [x] `W1-10` **Script sinh nháp golden set** — DeepSeek sinh Q + relevant_chunk_ids từ chunk thật, kèm phân loại category
  · DoD: **266 câu nháp** (≥250 ✅) · chi phí **$0,5821** ($0,00219/câu) · khử trùng lặp có · Test: `test_goldenset_gen.py` **35 case** + `test_goldenset_dedupe.py` **19 case** + `test_goldenset_sampling.py` **23 case** + `test_llm_provider.py` **23 case** · Evidence: `reports/tasks/w1-10-goldenset-draft.md` + `data/golden/draft_v1.jsonl`
  · Phân bố: factoid 78 · cross_lingual 46 · unanswerable 40 · adversarial 36 · multi_hop 34 · aggregation 28 · table_lookup 4. Ngôn ngữ vi 167 / en 99, trải trên **44/60 tài liệu**
  · ⚠️ **`deepseek-chat` là BÍ DANH**, thực tế phục vụ bởi `deepseek-v4-flash` (xác nhận trên API 2026-08-17; `deepseek-reasoner` cũng vậy). Đây đúng là vấn đề quy tắc cứng #1 nói về preset, chỉ kín đáo hơn. Mặc định dự án đã đổi sang slug thật
  · **Model không được tự viết `chunk_id`** — nó chỉ trả chỉ số đoạn văn, code ánh xạ sang id thật. `quote` được đối chiếu lại với chunk: 16/266 câu không kiểm chứng được → xếp đầu hàng đợi review
  · Ba bộ lọc chất lượng corpus phải thêm (trộn hai cột PDF 27,8% chunk · chú thích biểu đồ · trang bìa) — chi tiết ở report. Chúng **chỉ áp cho việc chọn mẫu**, không áp cho index
  · `table_lookup` chỉ 4 câu là **đúng, không phải lỗi**: corpus `.txt` đã làm phẳng bảng. Nhóm này chờ nguồn (c) HOSE + `W3-01`
- [x] `W1-11` **Review → freeze `golden_v1`** — **242 câu** (≥150 ✅), đủ 7 nhóm, sha256 `f53ad84abea32d3f…`, file read-only
  · ⚠️⚠️ **Review bởi MODEL, không phải người.** `reviewed_by_human=false`, `reviewed_by="model:claude-opus-5"`. `freeze` chỉ đặt `true` khi `--reviewer human` và cảnh báo mỗi lần khác. DeepSeek sinh → Claude review là **cross-model** (không phải tự chấm mình), nhưng **không** tương đương người. Đừng mô tả là "human-verified" ở CV hay phỏng vấn. Chi tiết + giới hạn: `reports/tasks/w1-11-review.md` §1
  · **Loại 24/266 câu**, năm loại lỗi: (a) **7 câu `unanswerable` mà corpus THẬT SỰ trả lời được** — lạm phát 2025 (4,5-5%), GDP 2030 (bảng CGE của CCDR = 5,45%), 1.584 thủ tục tiền kiểm, nghèo đa chiều 2022 → tỉ lệ sai 17,5% trong nhóm; (b) 4 câu không tự chứa ("According to the passage", "theo báo cáo này"); (c) 7 câu tra **cấu trúc tài liệu** (số trang, tiêu đề Hộp 3, số hiệu working paper, thư mục tham khảo); (d) 3 câu trùng ý — dedupe Jaccard không bắt vì chỉ *nguồn* trùng, *cách hỏi* khác; (e) 3 câu mơ hồ/vòng tròn
  · Quy trình: **6 phép kiểm máy chạy trên toàn bộ 266 câu** + đọc tay từng câu kèm text bằng chứng thật. `multi_hop`/`aggregation` 60/60 đủ ≥2 span · `cross_lingual` 43/43 thật khác ngôn ngữ · `adversarial` 36/36 có tiền đề sai · grounding **0 lỗi thật**
  · **`quote_unverified` KHÔNG phải dấu hiệu model bịa** mà là lỗi trích xuất PDF: trộn hai cột chèn chữ vào giữa từ (`"bổ sung"`→`"bổ chuyên sung"`), số chú thích chèn giữa câu, gạch nối cuối dòng. 15/16 trích dẫn có thật
  · Phép kiểm máy **sửa lại nhận định mắt thường của tôi** ở 1 câu (số 2.938 tỷ có thật, chỉ nằm ngoài span đã thu hẹp) — nếu tin mắt thì đã loại oan
  · Bắt được lúc chạy thật: **`freeze` làm rơi `relevant_spans`** → `golden_v1` mất hết công neo span của `TD-12`, không triệu chứng nào. Đã sửa + 4 test. Quyết định kèm theo: `edit` có điền `new_relevant_chunk_ids` thì **bỏ span**, vì ánh xạ span ghi đè chunk_id nên giữ span cũ sẽ âm thầm bỏ sửa tay của người review
  · DoD gốc ("người xác nhận") **chưa đạt**. Việc còn lại: người đọc lại 33 câu `unanswerable` + 43 câu `cross_lingual` (hai nhóm loại nhiều nhất = kém tự tin nhất) rồi freeze lại với `--reviewer human` — đủ 7 nhóm: factoid, multi_hop, aggregation, table_lookup, cross_lingual, unanswerable, adversarial
  · **Công cụ xong, đã chạy thật** (`reports/tasks/w1-11-triage.md`): `pipeline/goldenset/triage.py` (retriever thật → hàng đợi review) + `freeze.py` (quyết định → `golden_v1` + checksum + read-only). +71 unit test (409 tổng), +5 integration (38 tổng). `make goldenset-triage` chạy 24,3s
  · **Thứ tự đọc đã xếp sẵn** trong `queue_v1.md`: 15 câu `unanswerable_but_retrieved` → 16 câu `quote_unverified` → 2 câu `trivially_easy` → 161 câu `answerable_but_not_retrieved` (mặc định **accept**) → 72 câu không cờ. `gold_chunk_missing` = 0
  · ⚠️ **Bất đối xứng phải giữ đúng.** Câu `unanswerable` mà retriever tự tin = bằng chứng nhãn sai (mệnh đề về corpus bị phản chứng). Câu trả lời được mà retriever trượt **không** phải bằng chứng nhãn sai — đó là thứ eval tồn tại để đo, loại nó đi là tự thổi phồng recall baseline. Nên tín hiệu thứ hai xếp **cuối** hàng đợi, đề xuất `accept`, và có 2 test canh (`TestSignalB`)
  · Ngưỡng nghi ngờ **hiệu chuẩn từ dữ liệu**, không phải hằng số: trung vị điểm top-1 của các câu trả lời được = **0,5797**. Ghim `0.8` là đoán, vì "điểm cao" phụ thuộc model embedding và corpus
  · **Freeze không đoán hộ**: `fix_chunk_ids` mà người review để trống thì báo lỗi, không lấy top-1 của retriever điền vào — làm thế là dạy golden set trả lời đúng theo hệ thống hiện tại. Ba từ vựng tách rời: `suggested_decision` (máy hỏi) / `decision` (người trả lời: accept·reject·edit) / ô trống (chưa review, **không** phải accept)
  · **Nhãn đã neo theo span ký tự** trước khi review, để 6–8 giờ công không hỏng ở bước `W2` đầu tiên (`TD-12`). `make goldenset-anchor`. Nội dung chunk **không đổi một byte** — digest trên corpus thật khớp trước/sau, đó là bất biến số một của lần refactor đó
  · Hai file, không phải một: `queue_v1.md` tối ưu cho **đọc** (toàn văn chunk đã gán + top-3 + trích dẫn, không phải tra cứu qua lại), `decisions_v1.csv` tối ưu cho **ghi**. `write_decisions_template` từ chối ghi đè — mất 6 giờ công review vì chạy lại một lệnh là chuyện không được xảy ra
  · DoD: mỗi câu có `relevant_chunk_ids` đã người xác nhận; file read-only, có checksum · Test: `tests/unit/test_goldenset_schema.py` (schema + phân bố category + không rỗng chunk_ids trừ nhóm unanswerable) · Evidence: `data/golden/golden_v1.jsonl` + `reports/tasks/w1-11-review.md` (quy trình + phân bố) + `reports/goldenset/goldenset-v1.json` (số liệu freeze) + `reports/tasks/w1-11-triage.md` + `reports/tasks/w1-11-spans.md`
- [x] `W1-12` **`pipeline/eval/retrieval_eval.py`** — Recall@{1,5,10,20}, Precision@k, MRR, nDCG@10, HitRate; breakdown theo category & language
  · DoD: bảng MD + JSON; metric đúng trên fixture có đáp án tính tay · Test: `tests/unit/test_retrieval_metrics.py` (**35 case**) + `tests/unit/test_retrieval_eval.py` (**19 case**) · Evidence: `reports/tasks/w1-foundation.md`
  · Kỳ vọng viết bằng công thức tường minh với thứ hạng ghi rõ; cố ý **không** gọi lại hàm đang test để sinh kỳ vọng
  · Có breakdown theo category & language. **Câu `unanswerable` trả `None` chứ không phải `0.0`** — recall trên tập rỗng là không xác định; quy ước thành 0 kéo tụt điểm vô nghĩa, thành 1 thì thổi phồng. Nhóm này đo riêng ở `W5-02`
  · Thiếu `query_id` trong kết quả = truy hồi rỗng (bị chấm 0), không phải bỏ qua — im lặng bỏ qua sẽ làm điểm cao lên một cách sai
  · ✅ Đã nối retriever thật ở `W1-08`: `make eval-retrieval` dùng `--index-config` và truy hồi trực tiếp từ Qdrant. `--retrieved` vẫn giữ để chấm lại kết quả cũ mà không phải chạy lại retrieval
- [x] `W1-13` ⭐ **Baseline đã đo** — 209/242 câu được chấm (33 câu `unanswerable` trả `None`, đo riêng ở `W5-02`)
  · **recall@1 0,0877 · recall@5 0,1746 · recall@10 0,2257 · recall@20 0,2663 · MRR 0,1660 · nDCG@10 0,1621 · MAP@20 0,1349 · HitRate@5 0,2153 · p50 22,5ms / p95 39,9ms**
  · DoD chạy lại 2 lần: **sai số 0,0000%** trên cả 15 metric (không phải "<1%" mà bằng 0 — dense retrieval trên vector đã ghi là phép tính xác định). Evidence: `reports/tasks/w1-11-review.md` + `reports/baseline-retrieval.{md,json}`
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
- [x] Coverage `rag_core/` ≥ 70% → **80%** (chỉ tính unit test; `reports/tasks/w1-08-build-index.md`)

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
> **0 → 0,3023**, cả 15 metric có ý nghĩa. Chi tiết ở `reports/tasks/w2-01-bge-m3.md`.
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
> sau khi build lại (0/209 câu đổi điểm). Chi tiết: `reports/tasks/w2-02-qdrant-hybrid.md`.
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
>    reranker thì **sửa được phần lớn** — `W2-05` đo known-item hit@1 0,0980 → 0,5490, thắng cả
>    sparse. RRF thì không. Phần còn lại (35% mã không vào được pool 50) vẫn là `TD-18`.
> 3. **Tìm trong index sparse tốn 97,8 ms vs dense 17,8 ms.** Từ `W2-04` trở đi, nhánh sparse
>    là thành phần nặng nhất của đường truy hồi — không phải model embedding.
>
> ### `W2-04` xong (2026-08-20) — mặc định của bài báo là lựa chọn tệ nhất
>
> Tôi đã ghi trước: "nếu RRF ra thấp hơn 0,6268 thì nó đang làm hại, và đó là kết quả phải
> báo cáo chứ không phải tinh chỉnh `k` cho tới khi số đẹp". **Ở `k=60` nó ra 0,5742** — làm
> hại, có ý nghĩa thống kê. Nên đây là con số được báo cáo.
>
> Quét `k` rồi mới thấy nó là cần điều khiển chính: nDCG@10 theo `k` = 1→2→5→10→60 cho
> **0,4557 → 0,4530 → 0,4443 → 0,4305 → 0,4021**, đơn điệu. Cấu hình thắng `k=1, c=20`:
> `hit_rate@10` **0,6555** vs dense 0,6268 — nhưng chỉ 3/15 metric đạt ý nghĩa, tức cải
> thiện nhỏ và phần lớn dưới ngưỡng phân giải của `golden_v1`.
>
> Bốn thứ đổi cách làm phần còn lại của W2:
>
> 1. **`W2-08` quét `k ∈ {0,1,2,5}`, KHÔNG quét `weights`.** Từ `k=10` trở lên đã thấy rõ là
>    tệ hơn; còn weighted RRF cần tỉ lệ > 30,5:1 mới lật được gì (tính được, có test).
> 2. **`candidate_k` chỉ là cần điều khiển khi `k` lớn.** Ở `k=1` thì c20/c50/c100 cho cùng
>    một con số tới bốn chữ số thập phân — nên chọn cái nhỏ nhất, nó rẻ hơn và không mất gì.
> 3. **Hybrid là bộ SINH ứng viên, không phải bộ XẾP HẠNG.** `recall@20` +0,0431 có ý nghĩa,
>    `hit_rate@1` đứng im, và known-item mất một nửa MRR. Đó chính là hình dạng bài toán
>    `W2-05` tồn tại để giải — nên `W2-05` là việc tiếp theo, không phải tinh chỉnh RRF.
> 4. ⚠️ **Một bug 64 ms làm sai hai con số đã công bố.** `sparse_vocab_size` gọi
>    `len(tokenizer)` (dựng lại dict 250.002 phần tử) ở đường nóng. Sau khi sửa: sparse phía
>    **đọc** rẻ hơn dense (15,4 vs 28,7 ms) và phía **ghi** miễn phí hoàn toàn (389,2 → 379,1 s,
>    so với 380,4 s của dense-only). Cả `w2-02` và `w2-03` đã được đính chính tại chỗ.
>
> 💡 Cách tìm ra bug đó đáng nhớ hơn bản thân bug: `TD-11`, `W2-02`, `W2-03` đều được phát
> hiện bằng cách **kiểm một bất biến**; cái này được phát hiện bằng cách **buộc các con số
> cộng lại đúng**. Ba harness khác cấu trúc cho ba câu trả lời lệch nhau 2–6× cho cùng một
> việc — khi phương sai giữa các cách đo lớn hơn hiệu ứng muốn quy kết thì phải đo lại, không
> phải chọn con số vừa mắt.

> ### `W2-06` xong (2026-08-21) — và phần thiếu không phải phần tôi đi tìm
>
> Tôi vào hạng mục này để thêm `date range` và một ca isolation hai tenant. Cả hai đều làm,
> nhưng cái đáng nhất là thứ tìm được khi hỏi **"còn đường nào vào dữ liệu mà không qua
> filter?"** thay vì "làm sao lọc theo metadata":
>
> `retrieve()` đã lọc ở Qdrant từ `W1-07`. Nhưng `fetch_chunks(chunk_ids)` và
> `fetch_doc_chunks(doc_ids)` **không có tham số filter** — và đó đúng là hai method mà
> `W4-09` (giải citation) và `W4-06` (mở rộng ngữ cảnh) sẽ gọi. Một `chunk_id` lấy từ log
> hoặc từ câu trả lời cũ trả về **nội dung đầy đủ của tenant khác**, dù mọi truy vấn vector
> đều lọc đúng. Nó khó thấy vì hai đường trông giống nhau ở tầng gọi, nhưng
> `client.retrieve(ids=...)` của Qdrant **không nhận filter** — nên vá nó là chuyển sang
> `scroll`, không phải thêm một tham số.
>
> 💡 Bài học phương pháp, không phải bài học Qdrant: DoD viết "filter áp ở tầng Qdrant" và
> đường search đã thoả mãn nó từ trước. Đọc DoD như một checklist thì hạng mục này chỉ còn
> là viết test. Đọc nó như một **câu hỏi về bề mặt tấn công** thì nó tìm ra hai lỗ.
>
> ⚠️ Và `W2-06` **không** đóng được chuyện quan trọng nhất: nó không ép người gọi *phải*
> truyền `tenant_id`. `rag_core` không biết được "không filter" là đúng (eval chạy toàn
> corpus) hay là lỗ rò (serving quên). Chỗ ép là `W4-04`, nơi tenant đến từ token đã xác
> thực. Có test **ghim hành vi hiện tại** chứ không ghim hành vi mong muốn, để tới `W4`
> không ai đọc `W2-06` xong tưởng chuyện này đã xong.
>
> 💡 Con số vận hành: **lọc không tốn gì** — tám ca từ 20,5% tới 100% độ chọn lọc nằm trong
> 29,98–30,54 ms (trải 1,8%). Nhưng để đến được con số đó tôi phải **phân rã** phép đo, vì
> hai lượt đầu cho một bảng đơn điệu rất thuyết phục mà thật ra là nhiễu của bước embed.
> Xem `reports/tasks/w2-06-metadata-filter.md` §5.
>
> ### Việc tiếp theo: `--category`/`--lang` cho `compare.py`, RỒI `W2-08`
>
> `W2-07` đã chạy grid 14 ô và số nằm sẵn trong `plans/reports/runs/e1-*` + MLflow.
> Nhưng **bảng đó không phải `W2-08`**: DoD `W2-08` đòi `p`/CI **cho từng dòng**, và
> DoD `W2-09` đòi "nhận xét về *category nào cải thiện nhiều nhất*". Cả hai là
> `compare.py`, và `compare.py` **chưa có chiều category** — chính chỗ thiếu đó đã
> để một mức tụt có ý nghĩa của `W2-04` đi qua không ai thấy (xem đầu §4). Làm
> `--category`/`--lang` trước, rồi **xoá** `scripts/category_compare.py`. Phải in
> kèm `n`: 43 câu có ngưỡng phân giải thô hơn 209 câu nhiều.
>
> Grid `exp-001` có bốn chiều thật: `chunk_size` × `embedding` × nhánh truy hồi ×
> `rerank_candidates`. ⚠️ **`doc_type` không phải chiều dùng được**: `W2-06` đo được
> nó khớp 15.814/15.814 point. ⚠️ Và `e1-chunk550-dense` **không so được**
> recall/nDCG/MAP với 13 ô còn lại — `n_relevant_mean` 1,9617 vs 1,3828 (`G2`).
>
> ⚠️ **`TD-19` làm TRƯỚC `TD-13`**, không phải sau: một file `runs/*-retrieval.json`
> không nói được nó đo trên golden set nào, và `TD-13` sẽ tạo golden set thứ hai với
> **cùng đường dẫn**. Phía grid đã an toàn (`fingerprint` băm `golden_digest` từ
> `W2-07`); phía báo cáo lẻ thì chưa.
>
> 💡 `W2-05` cũng để lại một con số định hướng cho cả W3: **trần `hit_rate@50` = 0,7799**, tức
> **22% golden set không có bằng chứng trong 50 ứng viên đầu**. Không tầng xếp hạng nào chạm
> được phần đó — nó thuộc chất lượng parse (`W3-01`) và `TD-18`.
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
  · DoD: ✅ dense 1024-d + sparse dict từ **một** forward pass · ✅ cache model load (`lru_cache` cho cả backbone và `sparse_linear.pt`) · ✅ `max_sequence_tokens` = 8192 · Test: `tests/unit/test_bge_m3.py` (23 CPU + 11 GPU) + `test_sparse_vector.py` (21) · Evidence: `reports/tasks/w2-01-bge-m3.md`, `configs/indexing/bgem3.yaml`
  · **Kết quả: nDCG@10 0,1621 → 0,4442** (+174%), `hit_rate@5` 0,2153 → 0,5455, MRR 0,1660 → 0,4394. **Cả 15 metric có ý nghĩa** (`p < 0,001` hoặc CI95 không chứa 0). Nhãn bit-identical với baseline (`n_relevant_mean` 1,3828) nên recall@k so được — khác `TD-11`
  · **`cross_lingual` recall@5 0,0000 → 0,3023** (43 câu = 18% golden set đi từ *không hoạt động* sang hoạt động). Cả 5/6 nhóm cải thiện có ý nghĩa; `table_lookup` n=4 không có lực thống kê
  · Truncation **56,9% → 0,0%** (0/15814 chunk), token/chunk max 734 vs cửa sổ 8192. Tokenizer BGE-M3 tốn **ít hơn 30%** token cho text tiếng Anh (0,172 vs 0,244 token/ký tự) — gốc của việc `en` bị cắt nặng nhất ở baseline
  · ⚠️ **Mức tăng này KHÔNG phải công của việc sửa `TD-11`.** Lần chạy đổi cùng lúc model + cửa sổ + tính đa ngữ; `TD-11` đã tách riêng phần cửa sổ và nó cho `p = 0,711`. Vai trò thật của cửa sổ 8192 là **cho phép giữ `chunk_size=1000`**, tức làm phép đo sạch — không phải tạo ra mức tăng. `W2-08` phải chốt bằng ma trận 2 chiều
  · ⚠️ **Sparse chưa đi vào Qdrant.** Eval trên là dense-only; phần sparse đã dựng + test nhưng chưa tiêu, chờ `W2-02`…`W2-04`
  · Chi phí: `batch_size` 64 → 16 · VRAM ~3,3/8 GB · index 405 s · p95 truy hồi 32,8 → 46,0 ms · vector 768 → 1024 chiều
- [x] `W2-02` **Qdrant named vectors + sparse index** + script migrate collection — 2026-08-20
  · DoD: ✅ `rag_bgem3` chứa cả `dense` (1024-d) & `sparse`, query độc lập bằng `retrieve()` / `retrieve_sparse()` · Test: `tests/integration/test_qdrant_hybrid_schema.py` (21) + `tests/unit/test_qdrant_schema.py` (22, thuần) · Evidence: `reports/tasks/w2-02-qdrant-hybrid.md`, `scripts/migrate_collection.py`
  · **Sparse gần như miễn phí: +8,8 s trên 389 s (+2,3%)**, dung lượng +19% (dense 61,8 MB → +11,6 MB). Đó là hệ quả trực tiếp của việc `W2-01` cho dense và sparse ra từ **một** forward pass — gọi provider hai lần thì đây là +380 s
  · **Dense bit-identical sau khi build lại**: 15/15 metric không lệch một chữ số, **0/209 câu đổi điểm**. Đường ghi hybrid mới không làm lệch nhánh dense — kiểm tra tuyên bố "một đường code" của `W2-01` trên 15.814 chunk thật
  · Sparse trên dữ liệu thật: **95,9 entry/chunk** (p50 100 · p95 147 · max 195 · **min 3**), mật độ 0,0384% của vocab 250.002. ReLU loại ~55% token (chunk có p50 218 token) — sparse của BGE-M3 là một phép **chọn**, không phải bag-of-words có trọng số
  · ⭐ **Ngoài DoD: `ensure_collection` giờ KIỂM TRA schema** thay vì thấy tồn tại là trả về. 4 ca lệch đều có test, trong đó ca "collection có sparse mà provider chỉ dense" là **hỏng im lặng** — eval ra số trông bình thường trong khi nửa index không được dùng. `schema_problems()` là hàm thuần nên 12 ca test được trong `make test`
  · ⚠️ Thang điểm hai nhánh **không so được**: dense là cosine ∈ [−1,1], sparse là dot product không có trần. Đây là lý do `W2-04` phải hợp nhất theo **thứ hạng**
  · ⚠️ **KHÔNG dùng `Modifier.IDF`** cho sparse của BGE-M3 — trọng số đã học, chồng IDF lên là nhân đôi phép hạ bậc từ phổ biến và hỏng im lặng. IDF dành cho nhánh BM25 thô ở `W2-03`. Có test canh `modifier is None`
  · `HashingEmbeddingProvider` nay sinh được sparse (`sparse=True`, mặc định **tắt** vì `name` là cache key) — để `W2-03`/`W2-04` test được mà không cần GPU
  · ⚠️ **Chi phí độ trễ đo được**: p50 truy hồi dense 23,7 → **31,5 ms** (+33%, tái lập 3 lần) sau khi thêm sparse index vào cùng collection; p95 gần như không đổi (46,0 → 46,6 ms) vì p95 bị chi phối bởi forward pass embed truy vấn. **Chưa tách được** chi phí sparse index khỏi trạng thái segment sau build lại — phải đo lại ở `W2-04` nơi mỗi truy vấn đi cả hai nhánh
- [x] `W2-03` **Sparse retriever** (`retrieve_sparse()` → `Retriever`, đo được) — 2026-08-20
  · DoD: ✅ query từ khóa lạ mà dense miss thì sparse hit — **known-item search: hit@10 0,0784 → 0,5098**, hit@1 0,0196 → 0,3529, McNemar `p = 4,8e-07`, và **0 mã nào dense tìm ra mà sparse không** · Test: `tests/integration/test_sparse_retriever.py` (14 integration + 4 gpu) + `tests/unit/test_sparse_branch.py` (16) · Evidence: `reports/tasks/w2-03-sparse-retriever.md`, `reports/probes/w2-03-known-item.json`, `scripts/known_item_probe.py`
  · ⭐ **Trần lý thuyết của `W2-04`: hợp hai nhánh cho `hit_rate@10 = 0,7033`** vs dense 0,6268 (+12,2% tương đối). Ghi lại **trước khi** làm RRF để lúc đó không tự diễn giải kết quả theo hướng có lợi
  · **Trên golden set sparse KÉM hơn dense** và kém có ý nghĩa: nDCG@10 0,4442 → 0,3733 (CI95 [−0,1190, −0,0225]), `hit_rate@10` 0,6268 → 0,5120 (`p = 0,002`, 40↔16). 12/15 metric khác biệt thật, tất cả cùng chiều. Đây là kết quả **phải chờ đợi**: `golden_v1` là câu hỏi tự nhiên, đúng loại truy vấn dense sinh ra để xử lý
  · Chỗ sparse **bù được**: `en` 9↔4 (sparse **thắng** trên truy vấn tiếng Anh), `factoid` 10↔7. Chỗ sparse **chết hẳn**: `cross_lingual` 1↔19 và **0 câu cả hai** — câu tiếng Việt trên tài liệu tiếng Anh không có token nào trùng
  · ⚠️ **Sparse học được KHÔNG phải index khớp đúng.** 25/51 mã không nhánh nào tìm ra, và tokenizer giải thích hết: mã tìm được có neo từ vựng (`VIE-01` → `['▁','VIE','-01']`), mã miss rã thành chữ số chung (`P171645` → `['▁P','171','645']`). Đây là bằng chứng cho `TD-18`, không còn là phỏng đoán
  · ⚠️ **Chi phí: tìm trong index sparse 97,8 ms vs dense 17,8 ms (5,5×).** p50 toàn phần 30,2 → 113,4 ms. Embed truy vấn 12,6 ms **dùng chung** cả hai nhánh (lợi còn lại của "một forward pass" ở `W2-01`). Mỗi truy vấn hybrid ở `W2-04` ≈ 128 ms
  · ⭐ **Ngoài DoD: đóng một hố im lặng trong `compare.py`.** Harness lấy `fetch_doc_chunks` bằng `getattr`; retriever thiếu method đó thì nhãn rơi về nhãn cũ và hai lần chạy **cùng số nhãn nhưng khác nhãn** vẫn so được. Thêm `QueryScore.relevant_digest`; `compare.py` từ chối **toàn bộ** khi băm lệch. Đã chạy thật: 209/209 có băm, **0 lệch**
  · **Kết quả âm đã ghi lại, không xoá**: giả định "dense lẫn giữa các mã gần giống" **sai** — trên corpus 7 chunk cả hai đều tra đúng hạng 1. Có test canh chính kết quả âm đó, thay vì đổi assertion thành `rank_sparse <= rank_dense` (sẽ pass vì bằng nhau, và đọc như một chiến thắng)
  · Giả định "không trùng token thì sparse trả rỗng" cũng **sai** trên 15.814 chunk: sparse trả đủ 20 kết quả cho cả 209 câu. Giới hạn thật là **xếp hạng sai**, không phải **không trả gì**
  · Nhánh truy hồi **không** vào `IndexConfig` (không quyết định vector nào được ghi → không thuộc `fingerprint`). Là cờ `--retrieval-mode` / `MODE=`, và ở `W2-07` là một chiều của ma trận
- [x] `W2-04` **RRF fusion** (`k` configurable, mặc định 60) — 2026-08-20
  · DoD: ✅ deterministic (test canh 20 lần gọi cho cùng kết quả) · ✅ tie-break ổn định (4 quy tắc, mỗi quy tắc có test) · Test: `tests/unit/test_rrf.py` (40, thuần, số tính tay) + `test_hybrid_branch.py` (34) + `tests/integration/test_hybrid_retriever.py` (25) · Evidence: `reports/tasks/w2-04-rrf.md`
  · ⚠️ **`k=60` — mặc định của bài báo gốc — là lựa chọn TỆ NHẤT trong khoảng đã quét**, và nó kém dense một mình **có ý nghĩa** (`hit_rate@5` 0,5455 → 0,4689, `p = 0,014`; nDCG@10 CI95 không chứa 0). Cài RRF theo mặc định rồi báo "đã làm hybrid search" là trình bày một **suy giảm** như một tính năng
  · **Cấu hình thắng `k=1, candidate_k=20`**: `hit_rate@10` 0,6268 → **0,6555**, `hit_rate@20` 0,6746 → **0,7177**, `recall@20` 0,6324 → **0,6754**. Cả 15 metric tốt hơn dense nhưng **chỉ 3 đạt ý nghĩa** — cải thiện nhỏ và phần lớn dưới ngưỡng phân giải của `golden_v1` (+2,9 điểm vs ngưỡng ≥ 6). ⚠️ "15/15 đều dương" **không** là kiểm định: 15 metric này tương quan mạnh, không phải 15 phép thử độc lập
  · **`k` là cần điều khiển chính, đơn điệu, nhỏ thì tốt**: nDCG@10 theo `k` = 1→2→5→10→60 cho 0,4557→0,4530→0,4443→0,4305→0,4021. Cơ chế tính được: `k=60` cho chunk mà **cả hai** nhánh xếp hạng 3 (`2/63`) đè lên chunk dense xếp **hạng 1** (`1/61`) — gấp 1,9×, tức **đồng thuận yếu lật đổ tín hiệu mạnh**
  · **`candidate_k` chỉ có tác dụng khi `k` lớn**: ở `k=60` thì c20/c50/c100 cho 0,6364/0,5742/0,5024 (chênh 13 điểm); ở `k=1` thì 0,6555/0,6555/0,6555 (không lệch). Ngược hẳn dự đoán của tôi và ngược cả hai test đơn vị tôi viết để biện minh cho pool sâu — số học của chúng đúng, tiên đề sai (chunk sâu-đồng-thuận thật sự được đẩy lên, nhưng chúng thường **không liên quan**)
  · **Weighted RRF không đáng quét ở `W2-08`**: cần tỉ lệ **> 30,5:1** mới lật được một đồng thuận cùng độ sâu, vì cân dense lên cũng cân luôn phần dense của chunk đồng thuận (tính được, có test). Đo thật 2:1 cho +0,0287 — không đủ
  · ⭐ **Phát hiện đáng nhất không phải RRF: bug hiệu năng 64 ms/lần gọi**, tìm ra vì **các con số không cộng lại đúng**. `sparse_vocab_size` gọi `len(tokenizer)` → dựng lại dict 250.002 phần tử; `retrieve_sparse` đọc nó ở **mỗi** truy vấn và `upsert` ở **mỗi lô**. Nó làm **sai hai con số đã công bố** — xem hai dòng dưới
  · ⚠️ **`W2-03` §8 SAI về quy kết** (đã đính chính trong file đó): tìm sparse **15,4 ms**, RẺ HƠN tìm dense 28,7 ms; gửi cả hai trong một request batch tốn **30,2 ms** = bằng dense một mình vì Qdrant chạy song song. `retrieve()` sparse 109,3 → **30,4 ms**. Kết luận cũ "nhánh sparse sẽ là thành phần nặng nhất" là sai
  · ⚠️ **`W2-02` "+8,8 s (+2,3%)" đúng kết luận, sai cơ chế**: 124 lô × 64 ms ≈ 7,9 s là bug. Build lại sau khi sửa: **389,2 s → 379,1 s**, tức **nhanh hơn** cả dense-only của `W2-01` (380,4 s) — sparse phía ghi **miễn phí hoàn toàn**. Index bit-identical (0/209 câu đổi điểm)
  · ⭐ **Đối chiếu với `Fusion.RRF` của Qdrant** (bản tham chiếu độc lập): suy ra Qdrant dùng **`k = 1`**, không phải 60 của bài báo, và bản của ta trùng khít điểm của nó ở `k=1` (`rel=1e-6` — `score` của Qdrant là float32). Tự cài vẫn cần: `k` của Qdrant **không cấu hình được**, mà §3 cho thấy `k` là cần điều khiển quan trọng nhất
  · **Dự đoán ghi TRƯỚC khi đo, và sai cả hai chiều**: tôi dự `hybrid@10 ≈ 0,6268` (ngang dense, vì RRF xen kẽ nên top-10 ≈ hợp của hai top-5) và dự `hit_rate@1` sẽ cải thiện. Thực tế `k=60` cho 0,5742 (tệ hơn dự đoán) và `hit_rate@1` **không đổi ở mọi cấu hình**
  · ⚠️ **Bổ sung 2026-08-21 — hybrid làm tụt `cross_lingual` CÓ Ý NGHĨA, và `W2-04` không thấy vì `compare.py` không có chiều category.** 43 câu (20% tập đo): recall@5 **0,3023 → 0,2093** CI95 [−0,1860, −0,0233], nDCG@10 0,2538 → 0,1707 CI95 [−0,1279, −0,0396], `hit_rate@5` **4↔0** — hỏng 4 câu, sửa 0 câu. Cùng cơ chế với dòng trên, chỉ khác loại truy vấn: câu hỏi khác ngôn ngữ với tài liệu thì trùng từ vựng ≈ 0 **theo định nghĩa** nên sparse là nhiễu thuần, mà RRF đều vẫn cho nó quyền bầu ngang. `cross_lingual` 0 → 0,3023 là kết quả tiêu đề của `W2-01`; `W2-04` lấy lại gần ⅓ của nó trong khi bảng tổng vẫn xanh (166 câu còn lại pha loãng thành +0,0319 "trong ngưỡng nhiễu"). Chi tiết: `reports/tasks/w2-05-reranker.md` §6.4b
  · **Known-item: hybrid giữ vùng phủ, làm hỏng thứ hạng.** hit@10 0,5098 = sparse (3↔3, `p = 1`) nhưng hit@1 0,3529 → **0,0980**, hạng trung vị 1 → 4. RRF trọng số đều **không biết nhánh nào đáng tin cho truy vấn nào**; trên truy vấn tra mã, dense đóng góp toàn nhiễu mà vẫn ngang quyền
  · ⭐ **Kết luận kiến trúc**: hybrid là **bộ sinh ứng viên** tốt (`recall@20` +0,0431 có ý nghĩa) và **bộ xếp hạng cuối** tệ (`hit_rate@1` đứng im). Đó đúng là hình dạng bài toán `W2-05` tồn tại để giải
  · Nhánh dense đặt **trước** trong danh sách hợp nhất — tie-break ưu tiên danh sách đầu, và `W2-03` đo được dense mạnh hơn (0,6268 vs 0,5120) nên đó là tiên nghiệm đúng, không phải lựa chọn tuỳ tiện
  · Embed truy vấn **một** lần cho cả hai nhánh (12,7 ms), và **một** request HTTP cho cả hai truy vấn — phiên bản phía truy vấn của quyết định "một forward pass" ở `W2-01`
- [x] `W2-05` **Cross-encoder reranker** `bge-reranker-v2-m3` (batch, top_n configurable) — 2026-08-21
  · DoD: ⚠️ **KHÔNG đạt phần độ trễ** — rerank 50 cặp tốn **524 ms** trên RTX 4060 Laptop, vượt ngân sách 400 ms **31%**; 400 ms mua được pool **37**. ✅ CPU fallback có và đo rồi (19.446 ms cho pool 50, chậm **38×**) · Test: `tests/unit/test_reranker.py` (66) + `tests/integration/test_reranked_retriever.py` (11 integration + 9 gpu) · Evidence: `reports/tasks/w2-05-reranker.md`, `reports/probes/w2-05-rerank-probe.json`, `reports/probes/w2-05-known-item.json`, `scripts/rerank_probe.py`
  · ⭐ **Mức cải thiện lớn nhất của W2 sau chính BGE-M3**: `hit_rate@1` **0,3397 → 0,5598 (+22,0 điểm)**, `hit_rate@10` 0,6555 → 0,7703, nDCG@10 0,4563 → **0,6481** (+42% tương đối), MRR 0,4436 → 0,6440. **15/15 metric có ý nghĩa** cả với nền dense lẫn nền hybrid — khác hẳn `W2-04` (3/15). `hit_rate@5` từ dense: **0↔43** câu đổi chiều, tức **43 câu được sửa và 0 câu bị làm hỏng**
  · **Trần vùng phủ đo TRƯỚC khi làm** (`scripts/rerank_probe.py`): `hit_rate@50` của nhánh nền = **0,7799**, dư địa 44 điểm. Reranker lấy **đúng một nửa**. Và con số đó nói luôn giới hạn: **22% golden set không có bằng chứng trong 50 ứng viên đầu** — ngoài tầm với của mọi tầng xếp hạng, thuộc `W3`/`TD-18`
  · ⭐ **Kiến trúc 1: sau khi có reranker, tầng hybrid không còn đo được.** Cùng reranker c=50, nền dense vs nền hybrid cho **13/15 metric trong ngưỡng nhiễu** (`hit_rate@1` `p = 0,453`; `recall@20` CI95 [−0,0008, +0,0630]). Ở `W2-04` hybrid hơn dense **có ý nghĩa** — hai tầng sửa **cùng một khuyết điểm** và chồng lên nhau. **Vẫn giữ hybrid** vì nó miễn phí (534,4 vs 538,0 ms — Qdrant chạy hai nhánh song song trong một request), cả 15 metric vẫn cùng chiều, và `golden_v1` không đo được chỗ sparse thắng. Nhưng thứ tự ưu tiên rõ: nếu chỉ chọn một thì chọn reranker
  · ⭐ **Kiến trúc 2: `TD-18` là bài toán TRUY HỒI, không phải biểu diễn** — dự đoán của tôi sai hẳn. Known-item hit@1 **0,0980 → 0,5490**, hit@10 0,4706 → **0,6471**, và nó **thắng cả sparse** (0,3529 → 0,5490, McNemar `p = 0,0391`, 1↔8). Vocab subword phá việc **truy hồi** một mã (chấm điểm sparse là tích vô hướng trên *túi* subword nên các mảnh mất thông tin thứ tự/liền kề) nhưng **không** phá việc **nhận ra** nó (cross-encoder có attention trên cả cặp). Reranked tìm 33/51 mã vs 26/51 của hợp dense+sparse ở top-10 — **7 mã được cứu từ vùng sâu pool**
  · ⚠️ **Bổ sung 2026-08-21 — "hybrid miễn phí" phải nói ở dạng đầy đủ: miễn phí và vô hại KHI CÓ reranker phía sau.** Reranker vá đúng mức tụt `cross_lingual` mà `W2-04` gây ra: nền dense vs nền hybrid ở c=50 còn **1↔0** (`p = 1,000`) trên 43 câu đó, tức nhiễu; và so với chính nó không rerank thì hit@1 **0,0465 → 0,4419** (`p < 0,0001`, **0↔17**). Hệ quả: **chế độ suy giảm ghi ở §8 của report là SAI** — mất GPU thì lùi về **dense**, không phải hybrid, với lưu lượng có `cross_lingual`. Chọn đúng nhánh suy giảm cần phân loại truy vấn lúc chạy, tức `W4-07`
  · **Việc còn lại lộ ra từ đây:** `compare.py` cần `--category`/`--lang` để kiểm định theo tập con — hiện phải viết script tay. Đây là điều kiện của DoD `W2-09` ("ít nhất 2 nhận xét về *category nào cải thiện nhiều nhất*"): không kiểm định thì "cải thiện nhiều nhất" đúng là thứ `TD-11` đã dạy là vô nghĩa. Phải in kèm `n` — 43 câu có ngưỡng phân giải thô hơn 209 câu nhiều
  · **`candidates` tách được hai loại metric** (có kiểm định): 20→50 và 50→100 đều làm `hit_rate@1` **không** cải thiện có ý nghĩa (`p = 0,219` và `p = 0,125`) nhưng `hit_rate@10`/`recall@20` thì có. Pool sâu mua **chất lượng danh sách**, không mua **chất lượng hạng nhất**. Điểm vận hành khuyến nghị cho `W4`: **c=20 ở 233 ms** — giữ 91% mức lợi hạng nhất với 44% chi phí
  · **Ba cần điều khiển tối ưu đã cạn, và mỗi cái có số**: `max_length` chỉ cắt **1/12.100 cặp** (p50 287 token vs trần 512) nên hạ trần không giảm tính toán; fp16 lấy **3,52×** (1794,7 → 510,1 ms) và đổi top-1 ở đúng **1/60 câu**; `batch_size` **không mua được gì** (8/16 là nhiễu, 32/64 **tệ dần** vì `predict` gom batch theo độ dài đã sắp nên batch lớn tối đa hoá padding). Còn lại chỉ `candidates`, và nó là **trả bằng vùng phủ** chứ không phải tối ưu
  · ⚠️ **Dự đoán ghi trước: 3/7 sai, và cả 3 lần đều THẤP** — độ trễ (đoán 150–300 ms, thật 1794,7), fp16 (đoán 1,7–2,2×, thật 3,52×), `hit_rate@1` (đoán +8…+15, thật **+22,0**). Thiên lệch hệ thống về hiệu chuẩn với cross-encoder, cả chi phí lẫn lợi ích. D6 (bão hoà sigmoid) cũng sai: logit nằm trong [−10,87; +8,67], **0,0%** bão hoà — lý lẽ tôi dùng để biện minh cho mặc định logit thô bị phản chứng, mặc định giữ nhưng lý do phải hạ xuống mức yếu hơn và đúng
  · ⭐ **Ngoài DoD: đóng một hố im lặng thứ hai trong `compare.py`.** `precision@1` **bằng `hit_rate@1` từng chữ số** nhưng đi đường bootstrap, nên cùng một con số nhận hai kết luận trái nhau (McNemar `p = 0,125` "nhiễu" vs CI95 "khác biệt thật"). Đã route qua McNemar — và **bản sửa đầu tự tạo bug mới** vì `precision@10` khớp tiền tố `precision@1`, kéo luôn nó sang McNemar. Tách `BINARY_METRICS` khớp đúng tên; có test canh **chính cái bẫy tiền tố**
  · **Phép kiểm nội tại tình cờ**: `rerank c=20` cho `hit_rate@20`/`recall@20` **trùng khít** hybrid c=20 (0,7177 / 0,6770) — đúng như bắt buộc, vì cùng `candidates` và `top_k` thì tập trả về y hệt, chỉ khác thứ tự. Metric theo tập bất biến, metric theo hạng đổi 20 điểm. Nó canh đúng chỗ dễ sai nhất: `retrieve()` dựng lại `RetrievedChunk` từ đầu
  · `RerankedRetriever` bọc **một `Retriever` bất kỳ** nên `--rerank-base` là một chiều thật của `W2-08`; `build_branch` **gọi lại chính nó** cho nhánh nền nên mọi phép kiểm tham số của `W2-03`/`W2-04` áp dụng nguyên vẹn (`--rerank-base dense --rrf-k 1` vẫn nổ). `SUPPORTED_MODES` giờ **bằng** `RetrievalMode`
  · Chết khi điểm không hữu hạn: `NaN` so với mọi thứ đều `False` nên `sorted` trả thứ tự **tuỳ ý mà không báo**, và hậu quả trông y như "model kém". Chế độ hỏng thật của fp16 — có test cho `nan`/`±inf`
  · ⚠️ **`top_n` không phải cần điều khiển của phép đo**: đặt `--rerank-top-n 6` rồi chấm `recall@20` thì mọi metric @k > 6 mất nghĩa. Mặc định `None`, và có test **ghim** cái bẫy đó
  · VRAM: bge-m3 + reranker fp16 ~**3.900/8.188 MiB** (fp32 là 5.685). Còn dư ~4,3 GB — **vẫn không đủ** cho generator local (Qwen3-8B 4-bit ~5,5 GB), nên kiến trúc "generator qua API" đứng nguyên. `W0-06` vẫn phải đo chính thức
- [x] `W2-06` **Metadata filter** (tenant_id, doc_type, date range, lang) — 2026-08-21
  · DoD: ✅ filter áp ở tầng Qdrant, có test theo dõi **chính request** (`query_points`/`query_batch_points`) vì "kết quả đúng" không phân biệt được lọc-ở-server với lọc-ở-client · ✅ không rò chéo tenant, đo trên **cả 4 nhánh** · Test: `tests/unit/test_metadata_filter.py` (24) + `tests/integration/test_metadata_filter.py` (22) · Evidence: `reports/tasks/w2-06-metadata-filter.md`, `reports/probes/w2-06-filter-probe.json`, `reports/probes/w2-06-backfill-bgem3.json`
  · ⭐ **Phần thực sự thiếu không phải date range — là đường `fetch`.** `retrieve()` đã lọc ở Qdrant từ `W1-07`, nhưng `fetch_chunks(chunk_ids)` và `fetch_doc_chunks(doc_ids)` **không có tham số filter**. Đó đúng là hai method mà `W4-09` (giải citation) và `W4-06` (mở rộng ngữ cảnh) sẽ gọi — một `chunk_id` lấy từ log hoặc câu trả lời cũ trả về **nội dung đầy đủ của tenant khác**, dù mọi truy vấn vector đều lọc đúng. Khó thấy vì `client.retrieve(ids=...)` của Qdrant **không nhận filter**, nên vá nó là chuyển sang `scroll`, không phải thêm một tham số
  · **`MetadataFilter` (`extra="forbid"`, `frozen`) đóng bốn cách viết filter cho 0 kết quả mà không báo lỗi**: khoá gõ sai (`{"tenant": ...}` thiếu `_id` → Qdrant trả rỗng, trông y như "tenant chưa có tài liệu"), `[]` (Qdrant hiểu `MatchAny(any=[])` = khớp rỗng), khoảng ngược, và chunk thiếu `tenant_id`. Ba cái đầu giờ **nổ**; cái thứ tư là hành vi Qdrant và có test ghim thay vì được tin. `frozen` là quyết định **bảo mật**: `W4` kiểm `filters.tenant_id == token.tenant` rồi truyền tiếp thì filter đổi được cho phép nới ra *sau* khi đã kiểm
  · ⚠️ **`W2-06` KHÔNG ép được người gọi phải truyền `tenant_id`** — `retrieve(query)` không filter vẫn thấy tất cả, và `rag_core` không biết được thế là đúng (eval chạy toàn corpus) hay là lỗ rò (serving quên). Chỗ ép là `W4-04`, nơi tenant đến từ token đã xác thực. Có test **ghim hành vi hiện tại** chứ không ghim hành vi mong muốn
  · ⭐ **Lọc không tốn gì**: tám ca từ 20,5% tới 100% độ chọn lọc nằm trong **29,98–30,54 ms** (trải 1,8%) ở phần chỉ-Qdrant. Ca **0 point khớp nhanh gấp đôi** (15,39 ms) — Qdrant cắt sớm khi cardinality bằng 0, tức "tenant mới chưa có tài liệu" là đường *nhanh*. Dự đoán `D4`/`D5` của tôi (lọc chậm hơn; lọc chặt đắt hơn) **sai cả hai**
  · ⚠️ **Và tôi đã suýt báo cáo nhiễu**: hai lượt đo đầu (chỉ có cột đầu-cuối) cho một bảng **đơn điệu theo độ chọn lọc**, nhưng cùng một ca lệch ±11 ms giữa hai lượt trong cùng tiến trình, còn p95 chỉ lệch 2,8%. Cách sửa **không** phải tăng mẫu (n=200 vẫn cho 32,5 và 43,9 cho cùng ca) mà là **phân rã** — embed truy vấn ngoài vòng bấm giờ. Bài học `W2-04` §6 lần thứ hai: số không cộng lại đúng thì tách ra, đừng lấy thêm mẫu
  · **Migrate không cần build lại index** (dự đoán `D3` sai theo hướng có lợi): `published_at` là field payload mới nên 15.814 point của `rag_bgem3` thiếu nó, và point thiếu field **không khớp** `DatetimeRange` — tức `published_after=2020` trả 0 kết quả trên toàn corpus. `make backfill-payload` cập nhật **15.814/15.814** point và chạm **0** vector, nên mọi con số eval từ `W2-01` đến `W2-05` vẫn đúng nguyên vẹn. Build lại thì chúng *có thể* đúng và tôi sẽ phải chứng minh
  · `published_at` phải là index kiểu **`datetime`**, không phải `keyword`: index keyword *cũng dựng được* trên cùng field và mọi truy vấn khớp-chính-xác vẫn chạy, rồi `DatetimeRange` không dùng được nó và Qdrant quét. Hỏng chỉ về hiệu năng nên không test đúng/sai nào bắt được — đó là lý do `scripts/filter_probe.py` tồn tại
  · **`FILTER_FIELDS` ↔ `PAYLOAD_INDEXES` có test canh HAI chiều**: field lọc được mà không có index thì quét toàn bộ collection (đúng nhưng chậm); có index mà `_payload` không ghi thì mọi filter trên nó trả rỗng (sai). Hai chiều hỏng khác nhau nên cần hai test
  · ⚠️ **`doc_type` hiện là chiều filter CHẾT**: `doc_type=dev_report` khớp **15.814/15.814** point — toàn corpus là một loại. Dùng nó làm một dòng `W2-08` sẽ cho hai dòng giống nhau. `lang` thì thật (59,4% / 40,6%)
  · `mypy --strict` bắt một chỗ tôi **nới nửa vời**: `build_filter`/`fetch_*` nhận `MetadataFilter` nhưng `Retriever.retrieve` vẫn khai `dict`, nên API mới chỉ dùng được ở một nửa điểm vào — 25 lỗi, gồm 4 chỗ vi phạm Liskov ở `Retriever` giả trong test. Sửa bằng alias dùng chung `type FilterSpec = MetadataFilter | dict[str, Any] | None`
- [x] `W2-07` **Experiment Runner** — YAML config matrix → chạy tuần tự → MLflow tracking — 2026-08-21
  · DoD: 1 lệnh chạy hết grid ✅ (`make exp` — 14 ô, 0 lỗi, ~13 phút) · resume được khi crash giữa đường ✅ (`SIGKILL` giữa ô 3 → state đúng 2 ô, đĩa đúng 6 file không `.tmp`, resume chạy đúng `[1/1]`) · Test: `tests/unit/test_experiment_runner.py` — **80 case** · Evidence: `reports/tasks/w2-07-experiment-runner.md` §8 (bảng 14 ô đọc **từ MLflow**) + §8b
  · `pipeline/experiments/` — `config.py` (`MatrixBlock`/`ExperimentCell`/`expand`), `runner.py` (preflight + state + gom theo index), `tracking.py` (`open_tracker`/`SafeTracker`/`NullTracker`), `backfill.py`. Config: `configs/eval/exp-001-retrieval.yaml`
  · ⭐ **Resume phải khoá vào `fingerprint` của ô, không vào tên file báo cáo.** `ExperimentCell.fingerprint(index_fingerprint=…, golden_digest=…)` nhận hai thứ **từ ngoài** vì kết quả một ô không chỉ phụ thuộc những gì viết trong ô: build lại `bgem3.yaml` với `chunk_size` khác, hoặc `TD-13` ghi lại golden set ở **cùng đường dẫn** — cả hai đổi kết quả mà không đổi một ký tự YAML nào. Ba cách cài "trông đúng" (theo file báo cáo / theo tên trong state / ghi state trước báo cáo) đều cho grid "chạy xong" với số sai và không có gì báo lỗi
  · ⭐ **Preflight bắt lỗi thật ở lần `--dry-run` ĐẦU TIÊN**: ô `bgem3-sparse` của grid trùng **đúng** báo cáo tiêu đề `W2-03` trong `plans/reports/runs/` và sẽ ghi đè nó. Sửa bằng `run_prefix: e1`. Đường **không** chọn: để grid dùng lại lần chạy cũ trùng tên — state không sở hữu chúng nên không biết chúng sinh bằng tham số nào, đúng câu `fingerprint` tồn tại để trả lời
  · ⭐ **Tách `check_branch_options` khỏi `build_branch`** (`rag_core/retrieval/branch.py`) — dự đoán `D3` nói không làm được. Luật "nhánh nào nhận tham số nào" trước đây chỉ trả lời được bằng cách **dựng** retriever, mà dựng `reranked` nạp cross-encoder 2,2 GB. Không chép luật sang pipeline: `build_branch` **gọi** hàm đó, và có test chạy **15 cặp đầu vào** qua cả hai đường khẳng định chúng nổ ở cùng chỗ. Phụ phẩm: `HYBRID_OPTIONS` + thông báo liệt kê tham số hợp lệ (trước đó `candidat_k=100` nhận `TypeError` trần)
  · ⭐ **Grid tái lập 5 con số đã công bố ĐÚNG TỪNG CHỮ SỐ**: 0,1621 (`W2-01` baseline), 0,4442 (`W2-01` BGE-M3), 0,3733 (`W2-03`), 0,4563 (`W2-04` `k=1`), 0,6481/hit@1 0,5598 (`W2-05`). Đây là cách duy nhất biết đợt refactor (`_eval_against_index` → `IndexSession`, `_resolve_span_labels(retriever)` → `(store)`, ghi file qua `os.replace`) **không đổi một con số nào**
  · ⚠️ **MLflow hỏng HAI lần, cả hai theo kiểu "chạy xong, đúng số ô, không có dữ liệu".** (1) mlflow 3.15 **từ chối** `file:./mlruns` (file store maintenance mode) → tracker rơi về `NullTracker`, grid chạy trọn 14 ô, Evidence DoD **không tồn tại**, cảnh báo duy nhất là **dòng 19 trong log 2320 dòng**. (2) MLflow **không nhận `@`** trong tên metric — mà *mọi* metric ở đây mang `@` (`ndcg@10`, `hit_rate@1`) → 14 run, **0 cột metric**, và `SafeTracker` nuốt 14 lần rồi in "Đã log 14 ô"
  · 💡 Bài học từ (2): **khoan dung đúng với lỗi nhất thời và sai với lỗi hệ thống, mà ở chỗ gọi thì hai loại giống nhau.** Sửa không phải nới `SafeTracker` mà **xoá cả lớp lỗi ở nguồn** (đổi `@` → `_at_` ở tầng adapter, file báo cáo giữ nguyên `@`) + test ghim mọi tên metric `evaluate_run` sinh ra đều hợp lệ. Sửa từ (1): URI khai mà không mở được là **lỗi config → preflight**; chỉ *thiếu mlflow* còn rơi về `NullTracker`; lỗi **giữa** grid thì `SafeTracker` lo
  · ⭐ **`make exp-backfill` dựng lại view MLflow TỪ FILE BÁO CÁO**, không chạy lại eval. Nó là **phép kiểm** cho câu `tracking.py` tuyên bố ("MLflow là view, không phải nguồn sự thật") thay vì để câu đó là khẳng định — và nó cứu 14 ô khỏi phải chạy lại 25 phút. Hai đường không thể lệch: cả hai gọi `report_params(json.loads(…))` trên **cùng byte**, có test ghim `_write_report` ghi đúng `report.to_json()`
  · ⚠️ **`D2` sai: model đã được chia sẻ sẵn từ `W1`.** Mở `baseline.yaml` 27,8 s → `chunk550.yaml` (cùng model) **0,4 s** → `bgem3.yaml` 11,1 s. Nguyên nhân: `@lru_cache` có sẵn trên cả ba loại model (`_load_model` 4, `_load_sparse_head` 4, cross-encoder 2). Gom ô theo index mua **quét nhãn span 14 → 3**, không mua lần nạp model — cộng một tính chất: mỗi model nạp đúng một lần **bất kể** `maxsize`
  · ⚠️ Và nó làm docstring đầu của `_release()` thành **sai**: `lru_cache` giữ tham chiếu mạnh nên `del` + `gc.collect()` không chạm được trọng số (đo 4517/8188 MiB khi chạy `reranked`). Hệ quả quan trọng hơn: **trần VRAM của grid do ba con số `maxsize` ở `rag_core` quyết định, không do runner** — grid quét 4 model embedding sẽ OOM card 8 GB và runner không ngăn được. Đầu vào cho `W0-06`
  · **`matrix` là danh sách KHỐI, không phải một tích Descartes**: không gian tham số không phải hình hộp (`k` chỉ có nghĩa với `hybrid`, `rerank_candidates` chỉ với `reranked`). Sinh hết rồi lọc thì "12 tổ hợp" của DoD `W2-08` thành con số không đoán được từ file config, và ô *bị lọc vì gõ sai tên* trông giống hệt ô *bị lọc vì vô nghĩa*. `options` luôn là list (kể cả 1 giá trị); `k: []` **nổ** vì tích với list rỗng cho 0 ô
  · **Chiều bị từ chối: `rerank_batch_size`.** Lập luận **nhất quán**, không phải khẳng định thực nghiệm mới — `IndexConfig.fingerprint` cố ý loại `batch_size` từ `W1-08`, nên dùng nó làm chiều ablation là tự mâu thuẫn với một quyết định đã ghi. ⚠️ `rerank_device` **không** bị từ chối và đây là chỗ dễ nhầm: `rerank_dtype: auto` = fp16 trên CUDA / fp32 nơi khác, nên quét device là **quét dtype một cách vô tình** (và `W2-05` đo fp16 đổi top-1 ở 1/60 câu). `exp-001` vì thế ghim `rerank_dtype: [float16]` tường minh
  · **Log 2320 dòng, 2270 dòng (97%) là `HTTP Request: HEAD huggingface.co/…`.** DoD "1 lệnh chạy hết grid" đạt về chức năng mà không đạt về việc **đọc được nó đã chạy gì** — với một lệnh 13 phút thì đó là cùng một yêu cầu (và chính nó chôn mất cảnh báo MLflow). Hạ từng logger tường minh, **không** hạ root: cảnh báo span của `_resolve_span_labels` phải thấy được
  · ⚠️ **Phép thử crash đầu tiên KHÔNG hợp lệ**, và thứ vạch ra là chính resume: `timeout -s KILL` giết `uv run` chứ không giết python con, nên nó chạy xong ô 3; lượt resume in "Bỏ qua **3** ô" trong khi state lúc kiểm chỉ có 2 — một mâu thuẫn nội tại giữa hai quan sát. Làm lại bằng cách gọi `.venv/Scripts/python.exe` trực tiếp
  · ⚠️ **Bảng 14 ô KHÔNG phải `W2-08`**: DoD `W2-08` đòi `p`/CI **cho từng dòng**, tức `compare.py`, chưa chạy. Chênh 0,6736 vs 0,6481 trên 209 câu là **5 câu** — đọc nó như kết luận là lặp lại đúng lỗi `TD-11`
  · 💡 Cột `n_relevant_mean` trên MLflow: `chunk550` = **1,9617**, mười ba ô kia = **1,3828**. `G2` hiện ra trong một cột, và nó thấy được **trước** khi ai đó ghép hai dòng vào cùng bảng (`compare.py` cũng từ chối, nhưng nó từ chối *lúc so*)
  · ⚠️ Cột `p95` của grid dùng để **sàng**, không để kết luận: 13 phút chạy liên tục trên GPU laptop, không có đối chứng thứ tự. Số độ trễ đáng tin đến từ probe riêng (`W2-04` §6, `W2-06` §5)
  · Dự đoán ghi trước: **2/6 đúng, 4 sai**. Ba lần **đánh giá quá cao độ khó của phần code** (`D2`, `D3`, `D6`), một lần **đánh giá quá thấp độ khó của phần tích hợp** (`D5`: tưởng MLflow nhàm, nó hỏng hai lần). Hướng lệch **ngược** với `W2-05`
- [ ] `W2-08` **Ablation #1**: chunking × embedding × retrieval mode × rerank
  · DoD: ≥ 12 tổ hợp có kết quả đầy đủ, xác định được cấu hình thắng **kèm `p`/CI cho từng dòng** · Test: — · Evidence: MLflow run IDs
  · ⚠️ Ma trận phải có **cả hai chiều** `chunk_size` × `embedding`: BGE-M3 xoá luôn nguyên nhân truncation nên gộp hai chiều sẽ gán toàn bộ mức tăng cho model một cách sai
  · ⚠️ Chiều `chunk_size` **không dùng** recall@k/nDCG/MAP (mẫu số là số nhãn — xem `G2`). `pipeline/eval/compare.py` tự từ chối những metric đó khi phân bố nhãn khác nhau
- [ ] `W2-09` **Report `reports/tasks/exp-001-retrieval.md`** — bảng số vs baseline + phân tích tại sao thắng/thua
  · DoD: có bảng delta, có ít nhất 2 nhận xét về *category nào cải thiện nhiều nhất* · Test: — · Evidence: file report

### `G2` — Gate tuần 2 ⬜
- [x] nDCG@10 tăng so với baseline (kỳ vọng ≥ +0.08 tuyệt đối) — **đạt ở `W2-01`: +0,2820** (0,1621 → 0,4442), gấp 3,5× ngưỡng, CI95 [+0,2297, +0,3346]
  · ⚠️ **Chỉ so được khi số nhãn/câu không đổi** — nDCG có mẫu số là số nhãn, mà nhãn neo theo span nên đổi `chunk_size` là đổi mẫu số (`TD-11` đo được: 1,38 → 1,96 nhãn/câu khi hạ 1000 → 550, làm nDCG tụt 29,6% **kể cả khi truy hồi y nguyên**). Đổi `chunk_size` thì phải dùng `hit_rate@k`/`MRR`
  · Tin tốt: +0,08 tuyệt đối trên 0,1621 là **+49% tương đối** — nằm trong tầm phân giải của `golden_v1`
  · `W2-01` giữ `chunk_size=1000` nên nhãn bit-identical với baseline (1,3828 nhãn/câu cả hai bên) → nDCG so được trực tiếp, `compare.py` không từ chối metric nào
- [x] **Mọi dòng ablation có `p` hoặc CI95** (`make eval-compare`), và chỉ tuyên bố người thắng khi kiểm định nói vậy — hạ tầng xong ở `TD-11`, dùng thật ở `W2-01` (15/15 metric có `p`/CI)
  · Đo được ở `TD-11`: với 209 câu và `hit_rate@5` ≈ 0,20, phải chênh **≥ 6 điểm tuyệt đối (≈28% tương đối)** mới phát hiện được. Xếp hạng 12 tổ hợp bằng mức chênh vài phần trăm là tung đồng xu
- [ ] Có bảng ablation ≥ 12 dòng, tái lập được bằng `make exp EXP=exp-001-retrieval`
  · ⚠️ Tiêu chí này trước đây viết `make eval EXP=exp_001` — **một target chưa bao giờ tồn tại**. Sửa 2026-08-21 khi `W2-07` dựng runner thật. Bảng **đã có 14 dòng** và tái lập được (`plans/reports/runs/e1-*` + MLflow), nhưng tiêu chí vẫn `[ ]` vì `W2-08` sở hữu nó và DoD `W2-08` đòi thêm `p`/CI **cho từng dòng** — đánh dấu bây giờ sẽ nói W2 đi xa hơn thực tế
- [ ] Không tổ hợp nào làm p95 latency vượt 3500 ms
  · Số hiện có (chỉ **truy hồi**, chưa end-to-end): dense p95 **46,2 ms** · hybrid `k=1 c20` **48,0 ms** · reranked **604,0 ms** (c=50) / **263,9 ms** (c=20) / **1154,5 ms** (c=100). Còn nhiều chỗ, nhưng ngưỡng 3500 ms là end-to-end nên chỉ đối chiếu được sau `W4-13`
  · ⚠️ Con số dense cũ ở dòng này ("45,7 ms") **không có trong nguồn được dẫn**: `reports/runs/bgem3-retrieval.md` ghi 46,2. Nó lấy từ bảng ở `reports/tasks/w2-03-sparse-retriever.md` §8 — chính phần đã bị đánh dấu "Nội dung gốc (SAI)" sau khi tìm ra bug 64 ms. Phần dense của bảng đó không sai, nhưng dẫn số từ một khối đã gạch là cách chắc chắn để nó tái sinh
  · ⚠️ Đừng dùng số p95 của `W2-03` §8 — nó sai vì bug 64 ms, đã đính chính ở `W2-04` §6

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
- [ ] `W3-09` **Ablation #2 + report** `reports/tasks/exp-002-chunking.md` — 5 chiến lược chunking
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
  · DoD: PDF chứa "ignore previous instructions" không đổi được hành vi · Test: `tests/security/test_prompt_injection.py` (bộ 10 payload) + `test_pii_redact.py` · Evidence: `reports/tasks/security-w4.md`
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
  · DoD: `reports/tasks/judge-calibration.md` có kappa vs người + agreement giữa 2 judge khác họ + phân tích case judge sai · Test: `tests/unit/test_kappa.py` (so với giá trị tính tay) · Evidence: file report
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
  · DoD: `reports/tasks/exp-003-generator.md` có bảng quality × p95 latency × cost/query; nêu rõ chọn model nào cho production và **vì sao**; ghi cảnh báo self-preference bias · Test: `tests/unit/test_bundle.py::test_evaluated_with_generator_required` (bundle thiếu field → reject) · Evidence: file report + MLflow run IDs

### `G5` — Gate tuần 5 ⬜
- [ ] **Test ngược:** mở 1 PR cố ý làm tụt retrieval (ví dụ `top_k=1`, tắt rerank) → **CI phải đỏ**
- [ ] Có `reports/tasks/judge-calibration.md` với kappa ≥ 0.6 (nếu < 0.6 → sửa prompt judge rồi đo lại)
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
  · DoD: `reports/tasks/loadtest.md` có biểu đồ + điểm bão hòa · Test: — · Evidence: file report
- [ ] `W6-06` **Security pass** (`/ek:security` hoặc `/security-review`)
  · DoD: 0 finding severity cao chưa xử lý · Test: — · Evidence: `reports/tasks/security-final.md`
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
| ~~`TD-12`~~ | ~~`chunk_id` thuần vị trí làm golden set hỏng âm thầm khi đổi chunking~~ | **Đã trả**: nhãn neo theo **span ký tự** trong văn bản gốc (`TextSpan`, `Chunk.start_char/end_char`), eval tự ánh xạ span → chunk_id của index đang đo. 226/226 câu đã neo, 299 span (62% thu về đúng câu trích dẫn). Đo ở 3 cấu hình chunking (1000/600/400): **0 câu mất nhãn**; ở 1000 dựng lại đúng nhãn cũ 216/226 | ✅ `reports/tasks/w1-11-spans.md` |
| ~~`TD-11`~~ ⭐ | ~~**56,8% chunk bị cắt lúc embed**~~ — `chunk_size=1000` **ký tự** ≈ 340 token, `vietnamese-bi-encoder` (PhoBERT) có `max_seq_length=256` **token**. `sentence-transformers` cắt âm thầm, không cảnh báo. **15,7% toàn bộ text đem embed không bao giờ tới được vector** (mẫu ngẫu nhiên 1200 chunk, seed 20260820; riêng chunk đã gán thì 91,1% vì bộ lọc prose-like chọn chunk dài) | **Cố ý không sửa trước `W1-13`**: đây là hành vi thật của bản POC, sửa trước rồi gọi kết quả là "baseline" thì mọi so sánh về sau đo lẫn cải tiến này vào. Chiều câu hỏi an toàn: p50 = 44 token, 0/266 bị cắt | **✅ Đã trả phần đo ở W2 (2026-08-20) — và giả định của nợ này bị phản chứng.** Hạ `chunk_size` 1000 → 550 đưa truncation từ 56,9% về **0,4%** chunk (token mất 15,4% → 0,1%) và **không cải thiện gì đo được**: McNemar `p = 0,711`, mọi CI bootstrap chứa 0. Lý do: baseline bị cắt nhưng mỗi vector vẫn đọc ~950 ký tự, chunk550 không bị cắt nhưng mỗi vector chỉ đọc 678 — hạ `chunk_size` là **đánh đổi**, không phải thu hồi. Còn lại là `W2-01` (BGE-M3, cửa sổ 8192): xoá truncation **mà không** phải hạ `chunk_size`. Evidence: `reports/tasks/w2-td11-chunk-size.md` |
| `TD-17` | **1/60 tài liệu dùng `Ê` (U+00CA) làm dấu cách** — `wb-099553007092621441` (*Điểm lại*, bản VI mới nhất): 5,7% ký tự của nó là `Ê`, tức văn bản **không có ranh giới từ**. Hệ quả: splitter không cắt được theo `" "` nên rơi xuống mức ký tự, tokenizer nổ ra 0,63 token/ký tự (bình thường 0,20) và 31/36 chunk vẫn bị cắt kể cả ở `chunk_size=550`. Embedding của tài liệu này về cơ bản là rác | Ảnh hưởng hẹp và đã đo: **1/242 câu** golden set trỏ vào nó (`factoid-caa304ba55`), 36/31.155 chunk của index. Sửa đúng chỗ là ở tầng đọc tài liệu, không phải ở chunking — chữa bằng cách hạ `chunk_size` là chữa triệu chứng | `W3-01` (Docling) đọc lại từ PDF gốc thay vì bản `.txt` mà World Bank trích. Trước đó thêm **cổng chất lượng corpus**: tỉ lệ whitespace < 10% thì cảnh báo — corpus hiện tại p50 là 37,5%, tài liệu này 16,4% với 5,7% là `Ê` giả dạng. Phép kiểm rẻ, và nó bắt được cả họ lỗi 'PDF trích ra chữ nhưng mất ranh giới'  · **Cập nhật `W2-01`:** cửa sổ 8192 của BGE-M3 làm tài liệu này **không còn gây truncation** (0/36 chunk), nhưng văn bản vẫn không có ranh giới từ nên nợ vẫn mở — triệu chứng mất, nguyên nhân còn. Sửa ở `W3-01` |
| `TD-13` ⭐ | **`golden_v1` review bằng MODEL, không phải người.** `reviewed_by_human=false`, `reviewed_by="model:claude-opus-5"` trong cả 242 dòng | Bạn không có ~6–8h để đọc tay, mà thiếu golden set thì cả W2–W6 không có thước đo. DeepSeek sinh → Claude review là **cross-model** nên không phải tự chấm mình; và mọi kết luận đều được đối chiếu với **văn bản gốc**, không phải với retriever (dùng retriever để lọc golden set là tự thổi phồng recall) | Người đọc lại **hai nhóm loại nhiều nhất** = hai nhóm model kém tự tin nhất: 33 câu `unanswerable` (đã tìm ra 7/40 sai nhãn, tỉ lệ sai 17,5%) + 43 câu `cross_lingual`. Rồi `make goldenset-freeze` với `--reviewer human`. Cho tới lúc đó: **cấm** chữ "human-verified" ở README, report, CV, phỏng vấn |
| `TD-14` | **7 `reference_answer` lẫn ngữ cảnh lúc sinh** — "Theo đoạn văn…", "Đoạn văn không nói…", "không có trong các đoạn văn được cung cấp" (`multi_hop-78a942bf28`, `multi_hop-d8e57b6cb1`, `multi_hop-6af19bc73a`, `adversarial-8ce6a8483a`, `adversarial-87b49c12d2`, `unanswerable-200ea2a853`, `unanswerable-599b31f973`) | **Vô hại cho eval truy hồi** (`W1-13` chỉ dùng `relevant_chunk_ids`/`relevant_spans`, không đọc `reference_answer`). Câu hỏi thì sạch: **0/242** câu có lỗi này | Dọn trước `W5-01`: judge so câu trả lời của hệ thống với `reference_answer`, mà "Theo đoạn văn 1" là thứ hệ thống thật không bao giờ nói được → judge sẽ trừ điểm oan. Làm cùng lượt người review `TD-13` |
| `TD-15` | **Nhãn liên quan chưa đầy đủ** — một dữ kiện có mặt ở nhiều tài liệu nhưng chỉ 1 chỗ được gán. Ví dụ "92% tài sản của các định chế tài chính" có trong **cả hai** *Financial sector assessment* và *Taking stock* | Hiếm: phép thử Jaccard ở `W1-11` cho **1/78** câu. Sửa cho đủ nghĩa là gán chéo toàn corpus × toàn golden set — đắt hơn giá trị mang lại ở giai đoạn này | Recall thật **cao hơn** số đo, tức baseline là **cận dưới** — sai theo chiều an toàn (không thổi phồng cải tiến). Nếu W2 đưa recall lên >0,6 mà vẫn có câu "trượt", kiểm chỗ này trước khi kết luận hệ thống sai |
| `TD-16` | **Nhóm `unanswerable` thiếu đa dạng** — 10/33 câu theo cùng một khuôn "hỏi về nước khác" (Thái Lan, Trung Quốc, Indonesia…) | 33 câu vẫn đủ để đo refusal correctness ở `W5-02`, và khuôn này đúng là một dạng câu không trả lời được | Thêm 2 khuôn khác khi có nguồn (b)/(c): hỏi về **mốc thời gian ngoài phạm vi** tài liệu, và hỏi số liệu **chi tiết hơn mức tài liệu có**. Nếu chỉ đo được một khuôn thì refusal correctness cao có thể chỉ nghĩa là hệ thống giỏi nhận ra tên nước lạ |
| `TD-19` | **Một file `runs/*-retrieval.json` không nói được nó đo trên golden set nào.** `config` của báo cáo có `index_config`, `index_fingerprint`, `collection`, `embedding_model`, `retrieval_mode`, `branch_options`, `chunking`, `top_k` — nhưng **không** có `golden` và **không** có `min_overlap_ratio` | Phát hiện khi viết `pipeline/experiments/backfill.py` ở `W2-07`: dựng lại view MLflow từ báo cáo thì hai trường đó phải lấy từ file config thí nghiệm chứ không lấy được từ chính báo cáo. Hôm nay **vô hại vì chỉ có một golden set** — mọi con số `W1-13`…`W2-08` đều đo trên `golden_v1`, nên không có gì để lẫn | ⚠️ **`TD-13` sẽ tạo ra cái thứ hai với cùng đường dẫn** `data/golden/golden_v1.jsonl`, và lúc đó hai báo cáo cạnh nhau không phân biệt được — đúng cùng một họ với `TD-12` (`chunk_id` thuần vị trí). Sửa: thêm `golden`, `golden_digest`, `min_overlap_ratio` vào `config` của `EvalReport`. Additive nên báo cáo cũ vẫn đọc được; `ExperimentCell.fingerprint` **đã** băm `golden_digest` từ `W2-07` nên phía grid đã an toàn, chỉ phía file báo cáo lẻ là chưa. Làm **trước `TD-13`**, không phải sau |
| `TD-18` | **Không có nhánh khớp đúng cho mã tài liệu.** `W2-03` đo được: 25/51 mã (project ID, trust fund ID) **không nhánh nào** tìm ra ở top-10. Nguyên nhân là vocab **subword** của BGE-M3: `P171645` → `['▁P','171','645']`, và `171`/`645` có mặt khắp 15.814 chunk tài liệu thống kê. Mã tìm được thì luôn có một mảnh subword tự nó là từ hiếm (`VIE-01` → `['▁','VIE','-01']`) | Ảnh hưởng **không** đo được trên `golden_v1` — 209 câu đều là câu hỏi tự nhiên, không có câu nào tra mã. Nhưng đây là loại truy vấn có thật trong sản phẩm (người dùng dán mã dự án vào hộp tìm kiếm), ⚠️ **`W2-05` đã phản chứng nửa sau của phát biểu này.** Reranker **sửa được phần lớn**: known-item hit@1 0,0980 → **0,5490**, hit@10 0,4706 → **0,6471**, và nó **thắng cả sparse** (0,3529 → 0,5490, McNemar `p = 0,0391`). Vì vocab subword phá việc **truy hồi** một mã (điểm sparse là tích vô hướng trên *túi* subword nên mất thông tin thứ tự/liền kề) nhưng **không** phá việc **nhận ra** nó — cross-encoder có attention trên cả cặp nên nó thấy các mảnh xuất hiện liền nhau, đúng thứ tự. Reranked tìm **33/51** mã vs 26/51 của hợp dense+sparse ở top-10. Nợ còn lại **thu hẹp thành 35% mã (18/51) không vào được pool 50** — đó mới là chỗ cần một nhánh khớp đúng. RRF (`W2-04`) thì đúng là không sửa được | Hai phương án, **đo cái rẻ trước**: (a) **filter payload khớp chuỗi** — Qdrant đã có `create_payload_index` từ `W1-07`, không cần build lại index; (b) **BM25 thô mức từ** (không phải subword) làm named vector sparse thứ hai — đây là chỗ cần `modifier=Modifier.IDF`, khác nhánh BGE-M3, nhưng phải đổi schema + build lại index. `scripts/known_item_probe.py` là phép đo sẵn có để so trước/sau (đã có nhánh `--rerank` từ `W2-05`, nên phép so bốn nhánh chạy được bằng một lệnh). **Ưu tiên đã hạ** sau `W2-05`: giá của việc trì hoãn không còn là 'không tra được mã' mà là '35% mã không tra được' |

---

## 12. Changelog

| Ngày | Thay đổi |
|---|---|
| 2026-08-21 | **`W2-07` xong — và MLflow hỏng hai lần theo cùng một kiểu.** `pipeline/experiments/` (`config.py` ma trận **danh sách khối** vì không gian tham số không phải hình hộp · `runner.py` preflight + state nguyên tử + gom theo index · `tracking.py` `open_tracker`/`SafeTracker`/`NullTracker` · `backfill.py`) + `configs/eval/exp-001-retrieval.yaml` (14 ô) + `check_branch_options` tách khỏi `build_branch` + `IndexSession` tách khỏi `_eval_against_index` + 4 target Makefile. ⭐ **Resume khoá vào `fingerprint` của ô, không vào tên file**: nó nhận `index_fingerprint` và `golden_digest` **từ ngoài** vì build lại index hoặc ghi lại golden set đổi kết quả mà không đổi một ký tự YAML nào. Ba cách cài "trông đúng" đều cho grid chạy xong với số sai và không báo lỗi. Chứng minh bằng `SIGKILL` thật: state đúng 2 ô, đĩa đúng 6 file không `.tmp`, resume chạy đúng `[1/1]`. ⭐ **Preflight bắt lỗi thật ở lần `--dry-run` ĐẦU TIÊN**: ô `bgem3-sparse` trùng đúng báo cáo tiêu đề `W2-03` và sẽ ghi đè nó → thêm `run_prefix`. ⭐ **Grid tái lập 5 con số đã công bố đúng từng chữ số** (0,1621 · 0,4442 · 0,3733 · 0,4563 · 0,6481/0,5598) — cách duy nhất biết đợt refactor không đổi con số nào. ⚠️ **MLflow hỏng hai lần, cả hai "chạy xong, đúng số ô, không có dữ liệu"**: mlflow 3.15 từ chối `file:./mlruns` (14 ô chạy, Evidence DoD không tồn tại, cảnh báo là **dòng 19 trong log 2320 dòng**) và MLflow không nhận `@` trong tên metric (**14 run, 0 cột metric**, `SafeTracker` nuốt 14 lần rồi in "Đã log 14 ô"). 💡 Bài học: **khoan dung đúng với lỗi nhất thời và sai với lỗi hệ thống, mà ở chỗ gọi thì hai loại giống nhau** — sửa bằng cách xoá lớp lỗi ở nguồn (`@` → `_at_` ở tầng adapter) + test ghim, và bằng cách coi URI không mở được là **lỗi config → preflight**. ⭐ `make exp-backfill` dựng lại view MLflow **từ file báo cáo**, tức *kiểm* câu "MLflow là view không phải nguồn sự thật" thay vì tuyên bố nó. ⚠️ **`D2` sai**: `@lru_cache` đã có trên cả ba loại model từ `W1` (0,4 s cho lần mở thứ hai cùng model vs 27,8 s lần đầu), nên gom theo index mua **quét nhãn span 14 → 3** chứ không mua lần nạp model — và docstring đầu của `_release()` **sai**, `lru_cache` giữ tham chiếu mạnh nên trần VRAM do `maxsize` ở `rag_core` quyết định, không do runner (đầu vào `W0-06`). ⚠️ 97% log là nhiễu httpx. ⚠️ Phép thử crash đầu **không hợp lệ** (`timeout` giết `uv run` chứ không giết python con) — vạch ra bởi chính resume nói "Bỏ qua 3 ô" khi state có 2. **`TD-19` mới**: báo cáo không nói được nó đo trên golden set nào, phải sửa **trước** `TD-13`. Dự đoán: **2/6 đúng, 4 sai** — ba lần quá cao độ khó phần code, một lần quá thấp độ khó phần tích hợp. +76 unit = **1096 test**. Evidence: `reports/tasks/w2-07-experiment-runner.md` |
| 2026-08-21 | **`W2-06` xong — phần thiếu không phải date range mà là đường `fetch`.** `rag_core/retrieval/filters.py` (`MetadataFilter` pydantic `extra="forbid"` + `frozen`, `type FilterSpec`, `build_filter` kiểm khoá) + `published_at` vào payload phẳng và `PAYLOAD_INDEXES` (kiểu **`datetime`**, không phải `keyword`) + `fetch_chunks`/`fetch_doc_chunks` nhận `filters` + `ensure_payload_indexes()`/`backfill_flat_payload()` + `pipeline/indexing/backfill_payload.py` + `scripts/filter_probe.py`. ⭐ **Đường `fetch` bỏ qua filter hoàn toàn** cho tới hạng mục này, và đó đúng là hai method `W4-09` (giải citation) / `W4-06` (mở rộng ngữ cảnh) sẽ gọi — một `chunk_id` từ log trả về nội dung đầy đủ của tenant khác dù mọi truy vấn vector đều lọc đúng. Khó thấy vì `client.retrieve(ids=...)` của Qdrant **không nhận filter**, nên vá là chuyển sang `scroll`. 💡 Bài học phương pháp: DoD ("filter áp ở tầng Qdrant") đã được đường search thoả mãn từ trước — đọc DoD như checklist thì hạng mục chỉ còn là viết test; đọc như **câu hỏi về bề mặt tấn công** thì nó tìm ra hai lỗ. **Hướng hỏng không đối xứng** nên mọi mặc định nghiêng về fail-closed: 4 cách viết filter cho 0 kết quả mà không báo lỗi (khoá gõ sai, `[]`, khoảng ngược, chunk thiếu `tenant_id`) — ba cái đầu giờ **nổ**, cái thứ tư là hành vi Qdrant nên có test **ghim** thay vì được tin. `frozen` là quyết định **bảo mật**: `W4` kiểm rồi truyền tiếp thì filter đổi được cho phép nới ra *sau* khi đã kiểm. ⚠️ **Không đóng được**: `W2-06` không ép người gọi phải truyền `tenant_id` — `rag_core` không biết "không filter" là đúng (eval toàn corpus) hay là lỗ rò (serving quên). Chỗ ép là `W4-04`; có test ghim hành vi **hiện tại**. ⭐ **Lọc không tốn gì**: tám ca 20,5%–100% độ chọn lọc nằm trong **29,98–30,54 ms** (trải 1,8%); ca 0 point khớp **nhanh gấp đôi** (15,39 ms — Qdrant cắt sớm khi cardinality bằng 0, tức "tenant mới" là đường *nhanh*). `D4`/`D5` sai cả hai. ⚠️ **Và tôi suýt báo cáo nhiễu**: hai lượt đầu cho bảng đơn điệu theo độ chọn lọc, nhưng cùng một ca lệch ±11 ms giữa hai lượt còn p95 chỉ lệch 2,8%. Cách sửa **không** phải tăng mẫu (n=200 vẫn cho 32,5 và 43,9 cho cùng ca) mà là **phân rã** — embed ngoài vòng bấm giờ. Bài học `W2-04` §6 lần thứ hai. **Migrate không build lại index** (`D3` sai theo hướng có lợi): payload và vector là hai thứ Qdrant cập nhật độc lập, nên `backfill-payload` sửa **15.814/15.814** point và chạm **0** vector → mọi số eval `W2-01`…`W2-05` còn nguyên. `mypy --strict` bắt một chỗ **nới nửa vời** (`retrieve` vẫn khai `dict`): 25 lỗi, gồm 4 vi phạm Liskov ở `Retriever` giả. ⚠️ **`doc_type` là chiều filter CHẾT** cho `W2-08`: khớp 15.814/15.814 point. +24 unit (876) + 22 integration (131) = **1020 test**. Evidence: `reports/tasks/w2-06-metadata-filter.md` |
| 2026-08-21 | **`W2-05` xong — mức cải thiện lớn nhất của W2 sau chính BGE-M3, và hai kết quả kiến trúc không đi tìm thì không thấy.** `rag_core/reranking/` (`Reranker` ABC + `CrossEncoderReranker`) + `rag_core/retrieval/reranked.py` (`RerankedRetriever` bọc **một `Retriever` bất kỳ**, nên `--rerank-base` là chiều thật của `W2-08`) + `build_branch` **đệ quy** cho nhánh nền + `scripts/rerank_probe.py`. ⭐ **`hit_rate@1` 0,3397 → 0,5598 (+22,0 điểm)**, nDCG@10 0,4563 → **0,6481** (+42% tương đối), **15/15 metric có ý nghĩa** với cả hai nền — `hit_rate@5` từ dense là **0↔43**, tức 43 câu được sửa và **0 câu bị làm hỏng** (khác hẳn `W2-04`: 3/15). **Trần vùng phủ đo TRƯỚC khi làm**: `hit_rate@50` của nhánh nền = **0,7799**, nên reranker lấy đúng **một nửa** dư địa 44 điểm — và con số đó nói luôn giới hạn của cả tầng xếp hạng: **22% golden set không có bằng chứng trong 50 ứng viên đầu**. ⚠️ **DoD 400 ms KHÔNG đạt**: 50 cặp tốn **524 ms** trên 4060 Laptop (400 ms mua được pool 37 — sửa 2026-08-21 khi chạy lại probe, con số "38" cũ tính bằng hằng số đã làm tròn). Không sửa DoD cho khớp số đo — ba cần điều khiển tối ưu đều đã cạn và mỗi cái có số: `max_length` chỉ cắt **1/12.100 cặp**; fp16 lấy **3,52×** và đổi top-1 ở đúng 1/60 câu; `batch_size` **không mua được gì** (32/64 *tệ dần* vì `predict` gom batch theo độ dài đã sắp nên batch lớn tối đa hoá padding). Còn lại chỉ `candidates`, và nó là **trả bằng vùng phủ**. CPU fallback có, đo được **19.446 ms** cho pool 50 (chậm **38×**) — nó tồn tại để code chạy đúng ở nơi không có GPU, không phải để phục vụ; mất GPU thì cách đúng là **tắt tầng rerank**. ⭐ **Kiến trúc 1: sau khi có reranker, tầng hybrid không còn đo được** — nền dense vs nền hybrid cho **13/15 metric trong ngưỡng nhiễu** (`hit_rate@1` `p = 0,453`), trong khi `W2-04` đo hybrid hơn dense *có ý nghĩa*. Hai tầng sửa **cùng một khuyết điểm** và chồng lên nhau; vẫn giữ hybrid vì nó miễn phí (534,4 vs 538,0 ms) chứ không vì nó đo được. ⭐⭐ **Kiến trúc 2: `TD-18` là bài toán TRUY HỒI, không phải biểu diễn** — dự đoán của tôi sai hẳn. Known-item hit@1 **0,0980 → 0,5490**, và nó **thắng cả sparse** (`p = 0,0391`): vocab subword phá việc *truy hồi* một mã nhưng không phá việc *nhận ra* nó, vì cross-encoder có attention trên cả cặp còn điểm sparse là tích vô hướng trên **túi** subword. Reranked tìm 33/51 mã vs 26/51 của hợp dense+sparse — 7 mã cứu từ vùng sâu pool. `TD-18` thu hẹp thành **35% mã không vào được pool 50**. ⭐ **Ngoài DoD: hố im lặng thứ hai trong `compare.py`** — `precision@1` bằng `hit_rate@1` từng chữ số nhưng đi bootstrap, nên cùng một con số nhận hai kết luận trái nhau (`p = 0,125` vs CI95 loại 0). Đã route qua McNemar, và **bản sửa đầu tự tạo bug mới** vì `precision@10` khớp tiền tố `precision@1` — tách `BINARY_METRICS` khớp đúng tên, có test canh chính cái bẫy tiền tố. ⚠️ **Dự đoán ghi trước: 3/7 sai và cả 3 lần đều THẤP** (độ trễ, fp16, `hit_rate@1`) — thiên lệch hệ thống về hiệu chuẩn với cross-encoder, cả chi phí lẫn lợi ích; D6 (bão hoà sigmoid) cũng bị phản chứng, logit nằm trong [−10,87; +8,67] và **0,0%** bão hoà. +68 unit (839) + 7 integration (109) + 9 gpu (26). Evidence: `reports/tasks/w2-05-reranker.md` |
| 2026-08-20 | **`W2-04` xong — mặc định của bài báo gốc là lựa chọn tệ nhất, và một bug 64 ms.** RRF tự cài (`rag_core/retrieval/rrf.py`, hàm **thuần** không nhận điểm — có test canh chính chữ ký đó, vì dense là cosine ∈ [−1,1] còn sparse là dot product không trần) + `QdrantHybridRetriever` (embed truy vấn **một** lần, **một** request HTTP cho cả hai nhánh) + `--rrf-k`/`--candidate-k`/`--rrf-weights`. ⚠️ **`k=60` của bài báo kém dense một mình CÓ Ý NGHĨA** (`hit_rate@5` 0,5455 → 0,4689, `p = 0,014`; nDCG@10 CI95 không chứa 0). Quét ra `k` là cần điều khiển chính và **đơn điệu**: nDCG@10 theo `k` = 1→2→5→10→60 cho 0,4557→0,4530→0,4443→0,4305→0,4021. Cấu hình thắng **`k=1, c=20`**: `hit_rate@10` 0,6268 → **0,6555**, `recall@20` 0,6324 → **0,6754** — cả 15 metric tốt hơn dense nhưng **chỉ 3 đạt ý nghĩa**, tức cải thiện nhỏ và phần lớn dưới ngưỡng phân giải (+2,9 điểm vs ngưỡng ≥ 6). **`candidate_k` chỉ có tác dụng khi `k` lớn**: ở `k=60` thì c20/c50/c100 chênh 13 điểm, ở `k=1` thì không lệch một chữ số — ngược hẳn dự đoán của tôi và ngược cả hai test đơn vị tôi viết để biện minh cho pool sâu (số học đúng, tiên đề sai: chunk sâu-đồng-thuận được đẩy lên thật, nhưng chúng thường **không liên quan**). ⭐ **Đối chiếu với `Fusion.RRF` của Qdrant**: suy ra Qdrant dùng **`k = 1`** không phải 60, và bản của ta trùng khít điểm của nó (`rel=1e-6`, `score` của Qdrant là float32) — nhưng vẫn phải tự cài vì `k` của Qdrant không cấu hình được. ⭐⭐ **Phát hiện đáng nhất không phải RRF: bug hiệu năng 64 ms/lần gọi**, tìm ra vì **các con số không cộng lại đúng** — phân rã `retrieve_sparse` để lại 81,7 ms không thuộc thành phần nào trong khi hybrid cộng đúng. `sparse_vocab_size` gọi `len(tokenizer)` → dựng lại dict 250.002 phần tử, và nó bị đọc ở **mỗi** truy vấn sparse + **mỗi lô** upsert. Nó làm sai **hai con số đã công bố**: `W2-03` §8 (tìm sparse thật ra **15,4 ms**, RẺ HƠN dense 28,7 ms; cả hai trong một batch tốn 30,2 ms = bằng dense một mình vì Qdrant chạy song song) và `W2-02` (+8,8 s → build lại sau khi sửa cho **379,1 s**, **nhanh hơn** cả dense-only 380,4 s của `W2-01` — sparse phía ghi miễn phí hoàn toàn). Cả hai report đã đính chính tại chỗ; index bit-identical sau khi build lại (0/209 câu đổi điểm). **Known-item: hybrid giữ vùng phủ (hit@10 0,5098 = sparse, 3↔3 `p = 1`) nhưng làm hỏng thứ hạng** (hit@1 0,3529 → 0,0980, hạng trung vị 1 → 4) — RRF trọng số đều không biết nhánh nào đáng tin cho truy vấn nào. ⭐ **Kết luận kiến trúc: hybrid là bộ SINH ứng viên tốt, bộ XẾP HẠNG cuối tệ** — đúng hình dạng bài toán `W2-05` tồn tại để giải. Dự đoán ghi trước khi đo (`hybrid@10 ≈ 0,6268`, `hit_rate@1` sẽ cải thiện) **sai cả hai chiều**. +73 unit (771) + 29 integration (102) + 2 gpu (17). Evidence: `reports/tasks/w2-04-rrf.md` |
| 2026-08-20 | **`W2-03` xong — sparse có số, và hai câu trả lời ngược nhau.** `retrieve_sparse()` giờ là `Retriever` (`QdrantSparseRetriever` bọc chính store đang mở kết nối; `build_branch()` chọn nhánh, `--retrieval-mode`/`MODE=`). **Trên golden set sparse KÉM hơn dense có ý nghĩa**: nDCG@10 0,4442 → 0,3733 (CI95 [−0,1190, −0,0225]), `hit_rate@10` 0,6268 → 0,5120 (`p = 0,002`), 12/15 metric cùng chiều — kết quả **phải chờ đợi** vì `golden_v1` toàn câu hỏi tự nhiên. **Nhưng DoD thì đạt áp đảo**: `golden_v1` không đo được DoD nên dựng phép đo riêng `scripts/known_item_probe.py` (known-item search trên 51 mã tài liệu, tiêu chí kiểm bằng **so chuỗi** nên không cần nhãn người) — sparse hit@10 **0,0784 → 0,5098**, hit@1 0,0196 → 0,3529, McNemar `p = 4,8e-07`, và **0 mã nào dense tìm ra mà sparse không**. ⭐ **Con số đáng nhất: trần của `W2-04` là `hit_rate@10` 0,7033** vs dense 0,6268 — ghi lại trước khi làm RRF để lúc đó không tự diễn giải theo hướng có lợi. Chỗ sparse bù được: `en` 9↔4 (**sparse thắng** trên truy vấn tiếng Anh), `factoid` 10↔7; chỗ sparse chết hẳn: `cross_lingual` 1↔19 và **0 câu cả hai**. ⚠️ **Sparse học được KHÔNG phải index khớp đúng**: 25/51 mã không nhánh nào tìm ra, tokenizer giải thích hết (`VIE-01` → `['▁','VIE','-01']` có neo từ vựng và tìm được; `P171645` → `['▁P','171','645']` toàn chữ số chung và miss) → `TD-18`, giờ có bằng chứng chứ không phải phỏng đoán. ⚠️ Chi phí: tìm trong index sparse **97,8 ms vs dense 17,8 ms (5,5×)**, p50 toàn phần 30,2 → 113,4 ms; embed truy vấn 12,6 ms **dùng chung** cả hai nhánh. ⭐ **Ngoài DoD: đóng một hố im lặng trong `compare.py`** — harness lấy `fetch_doc_chunks` bằng `getattr`, nên retriever thiếu method đó thì nhãn rơi về nhãn cũ và hai lần chạy **cùng số nhãn nhưng khác nhãn** vẫn so được. Thêm `QueryScore.relevant_digest`; `compare.py` từ chối **toàn bộ** khi băm lệch (không lọc bỏ câu lệch rồi so phần còn lại — đó là tự chọn mẫu). Chạy thật: 209/209 có băm, **0 lệch**. **Hai giả định của tôi bị phản chứng và được giữ lại trong test**: "dense lẫn giữa các mã gần giống" (sai — trên 7 chunk cả hai đều đúng hạng 1; có test canh chính kết quả âm đó thay vì đổi assertion thành `rank_sparse <= rank_dense` để xanh) và "không trùng token thì sparse trả rỗng" (sai trên 15.814 chunk — sparse trả đủ 20 cho cả 209 câu; giới hạn thật là **xếp hạng sai**). +31 unit (697) + 14 integration (73) + 4 gpu (15). Evidence: `reports/tasks/w2-03-sparse-retriever.md`, `reports/probes/w2-03-known-item.json` |
| 2026-08-20 | **`W2-02` xong — một collection, hai loại vector.** `rag_bgem3` build lại với cả named vector `dense` (1024-d) và `sparse`, query độc lập qua `retrieve()` / `retrieve_sparse()`. **Sparse gần như miễn phí: +8,8 s trên 389 s (+2,3%)**, dung lượng +19% — hệ quả trực tiếp của việc `W2-01` cho hai loại vector ra từ **một** forward pass; gọi provider hai lần thì đây là +380 s. **Dense bit-identical sau khi build lại**: 15/15 metric không lệch một chữ số, 0/209 câu đổi điểm — kiểm tra tuyên bố "một đường code" của `W2-01` trên 15.814 chunk thật thay vì trên 4 câu test. Sparse thật: 95,9 entry/chunk (p50 100 · p95 147 · min **3**), mật độ 0,0384% của vocab 250.002, ReLU loại ~55% token. ⭐ **Ngoài DoD: `ensure_collection` giờ KIỂM TRA schema** thay vì thấy tồn tại là trả về — trước đó chạy provider sinh sparse lên collection dense-only sẽ chết ở *giữa* job 15.000 chunk. 4 ca lệch đều có test, trong đó ca "collection có sparse mà provider chỉ dense" là **hỏng im lặng**: eval ra số trông bình thường trong khi nửa index bị bỏ. `schema_problems()` thuần nên 12 ca test trong `make test`. Thêm `scripts/migrate_collection.py` — cố ý **không** migrate tại chỗ (Qdrant không cho thêm named vector), mà chẩn đoán lệch schema + in đúng lệnh phải chạy; mã trả về 0/1/2/3. `HashingEmbeddingProvider` nay sinh được sparse (mặc định **tắt** vì `name` là cache key của semantic chunker — để mặc định bật làm 2 test W1 đỏ ngay, đó là cách phát hiện). ⚠️ **Không dùng `Modifier.IDF`** cho BGE-M3: trọng số đã học, chồng IDF là nhân đôi phép hạ bậc và hỏng im lặng. ⚠️ Thang điểm hai nhánh không so được (cosine vs dot không trần) nên `W2-04` phải hợp nhất theo **thứ hạng**. +22 unit (666) + 21 integration (59). Evidence: `reports/tasks/w2-02-qdrant-hybrid.md` |
| 2026-08-20 | **`W2-01` xong — BGE-M3, và mức tăng lớn nhất của dự án tới giờ.** Đổi model embedding `vietnamese-bi-encoder` → `BAAI/bge-m3`, **giữ nguyên toàn bộ chunking của baseline**: nDCG@10 **0,1621 → 0,4442** (+174%), `hit_rate@5` 0,2153 → 0,5455, MRR 0,1660 → 0,4394, recall@10 0,2257 → 0,5813. **Cả 15 metric có ý nghĩa thống kê** (`p < 0,001` hoặc CI95 không chứa 0); ở `hit_rate@5` là 74 câu thắng vs 5 câu thua. `cross_lingual` recall@5 **0,0000 → 0,3023** — 43 câu (18% golden set) đi từ *không hoạt động* sang hoạt động. 5/6 nhóm truy vấn cải thiện có ý nghĩa. Truncation **56,9% → 0,0%**. Vì giữ `chunk_size=1000` nên nhãn **bit-identical** với baseline (`n_relevant_mean` 1,3828 cả hai bên) → recall@k/nDCG/MAP so được trực tiếp, `compare.py` không từ chối metric nào. ⚠️ **Mức tăng này là của model, KHÔNG phải của việc hết truncation**: `TD-11` đã tách riêng phần đó (`chunk550` cũng hết cắt) và nó cho `p = 0,711`; vai trò thật của cửa sổ 8192 là *cho phép giữ `chunk_size=1000`*, tức làm phép đo sạch. Hạ tầng thêm: `SparseVector` (bất biến cưỡng chế), `HybridVectors`, năng lực sparse tuỳ chọn trên `EmbeddingProvider` (mặc định `None` — `None` ≠ rỗng, bài học `TD-11` lần thứ hai), `BgeM3EmbeddingProvider` (dense + sparse **một** forward pass, có test canh dense khớp `SentenceTransformer.encode()` tới 1,5e-8), `embedding_max_batch_tokens` (knob VRAM, ngoài `fingerprint`), `make test-gpu`. ⚠️ **Sparse chưa đi vào Qdrant** — eval là dense-only, chờ `W2-02`. `G2` 2/4 điều kiện. +44 unit test (644 tổng) + 11 gpu test. Sửa lỗi sổ sách: p95 baseline là **32,8 ms** không phải 39,9 ms. Evidence: `reports/tasks/w2-01-bge-m3.md` |
| 2026-08-20 | **W2 bắt đầu — `TD-11` trả xong, kết quả ÂM.** Hạ `chunk_size` 1000 → 550 đưa truncation **56,9% → 0,4%** chunk (token mất 15,4% → 0,1%) và **không cải thiện gì đo được**: McNemar `p = 0,711`, mọi CI bootstrap chứa 0. Giả định của `TD-11` bị phản chứng — hạ `chunk_size` là **đánh đổi** (mỗi vector đọc 950 → 678 ký tự), không phải thu hồi nội dung. Ba thứ đổi cách làm phần còn lại của W2: (1) **không quét thêm `chunk_size`** với model đơn ngữ, đi thẳng `W2-01` BGE-M3 giữ `chunk_size=1000`; (2) **`golden_v1` chỉ phân giải được mức chênh ≥ 6 điểm `hit_rate`** nên mọi dòng ablation phải có `p`/CI — thêm `*-per-query.jsonl` + `pipeline/eval/compare.py` (McNemar exact + bootstrap cặp, không cần `scipy`); (3) **recall@k/nDCG/MAP không so được giữa hai `chunk_size`** vì nhãn neo theo span làm mẫu số đổi 1,38 → 1,96 nhãn/câu → tụt 29,6% kể cả khi truy hồi y nguyên, `compare.py` tự chặn. Thêm `make truncation` (đo `TD-11` tái lập được, chia theo ngôn ngữ: EN mất 19,4% token vs VI 7,5%) và `WARNING` truncation trong mọi lần build index. Thêm `TD-17` (1 tài liệu dùng `Ê` làm dấu cách). +66 unit test (600 tổng). Evidence: `reports/tasks/w2-td11-chunk-size.md` |
| 2026-08-20 | **Đóng sổ tuần 1, mở sổ tuần 2.** Điền cột **Baseline** ở §1 (recall@10 0,2257 · nDCG@10 0,1621 · MRR 0,1660 · p95 truy hồi 39,9 ms) và tách rõ metric *chưa đo* thuộc W4/W5 khỏi metric đã có số. `W0-04` `[?]` → `[x]` (đánh nhầm từ 2026-08-17, DoD đã đủ và đã dùng thật ở `W1-10`/`W1-13`). Đếm lại tổng task: **77** (71 backlog gốc + 6 `NEW-xx`), số cũ "73" là lỗi sổ sách vì `NEW-03`…`NEW-06` không được cộng vào. Chốt **thứ tự tấn công W2** vào §4: `TD-11` trước, `W2-01` sau, để cặp số trước/sau tách được phần của `chunk_size` khỏi phần của BGE-M3. Thêm `TD-13`…`TD-16` (review bằng model · 7 `reference_answer` lẫn ngữ cảnh · nhãn liên quan chưa đầy đủ · nhóm `unanswerable` thiếu đa dạng) |
| 2026-08-20 | **`W1-11` + `W1-13` xong — tuần 1 đủ 13/13.** `golden_v1` 242 câu (loại 24/266), baseline **recall@5 0,1746 · MRR 0,1660 · p95 39,9ms**, chạy lại 2 lần sai số **0,0000%**, chạy được với API key rỗng. ⚠️ Review bằng **model** (`reviewed_by_human=false`), không phải người — `G1` là PASS **có điều kiện**. Trả xong `TD-09` bằng cách tra thẳng văn bản gốc: 7/40 câu `unanswerable` sai nhãn. Bắt được `freeze` làm rơi `relevant_spans`. Thêm `reviewed_by` vào `GoldenQuery` và `--reviewer` vào `freeze`. 534 unit test. Evidence: `reports/tasks/w1-11-review.md` |
| 2026-08-20 | **`TD-12` đã trả — golden set neo theo span ký tự.** `TextSpan` + `Chunk.start_char/end_char`; `split_pieces()` thay `split_text()` làm API chính của Chunker; `pipeline/eval/spans.py` ánh xạ span → chunk_id của index đang đo; `pipeline/goldenset/anchor.py` chuyển 266 nháp sang span (299 span, 62% thu theo trích dẫn). Đo ở `chunk_size` 1000/600/400: **0 câu mất nhãn**, còn nhãn `chunk_id` cũ thì "hợp lệ" 226/226 ở cả ba mà trỏ vào văn bản khác. Bất biến: nội dung chunk khớp từng byte trước/sau (digest `b381634d51e39365`). Chạy `--recreate` để point mang offset (170s). +121 unit test (530), +5 integration (38). Evidence: `reports/tasks/w1-11-spans.md` |
| 2026-08-20 | **`W1-11` phần máy xong — triage + freeze, và ba phát hiện về retrieval.** Chạy retriever thật lên 266 câu nháp (24,3s): chỉ **65/226 = 28,8%** câu có chunk đã gán trong top-20. Điều tra ra ba nguyên nhân: (a) model embedding **đơn ngữ** → cùng ngôn ngữ 42,6% vs khác ngôn ngữ 7,8%, chênh 5,5× (`cross_lingual` trượt 95,7%); (b) giả thuyết "corpus gần trùng nên nhãn không đầy đủ" **bị loại** (0/78 câu có Jaccard ≥ 0,5); (c) **`TD-11`** — 56,8% chunk bị cắt ở 256 token, 15,7% text không tới được vector. Trả xong `TD-09` (15/40 câu unanswerable bị nghi). +71 unit test (409), +5 integration (38). Evidence: `reports/tasks/w1-11-triage.md` |
| 2026-08-20 | **`W1-09` xong — corpus đã version bằng DVC**: 60 tài liệu / 14,7 MiB, `dvc pull` trên clone sạch lấy đủ 61 file trong 1,9s và **60/60 sha256 khớp manifest**. Thêm `pipeline/corpus/dvc_state.py` canh lệch giữa hai cơ chế versioning + 23 test (317 → 338 unit). Remote đặt ở `.dvc/config.local` chứ không phải `.dvc/config`. Làm khác checklist: `data/golden` giữ trong git (lý lẽ ở `reports/tasks/w1-09-dvc.md` §3.2). Thêm `TD-10` |
| 2026-08-20 | **Dọn codebase + đổi tên repo**: bản POC chuyển vào `legacy/` (giữ lịch sử qua `git mv`), repo đổi tên `project_1.2_chatbot_rag` → `RAG-Chatbot`. Evidence: `reports/tasks/rename-workspace.md` |
| 2026-08-17 | **`W1-10` xong — 266 câu nháp golden set** ($0,5821 · 640s · 163 lời gọi). Thêm `packages/rag_core/llm/` và `pipeline/goldenset/`, 100 test mới. Phát hiện `deepseek-chat` chỉ là bí danh của `deepseek-v4-flash`; phát hiện model suy luận tiêu hết ngân sách token trước khi viết JSON; phát hiện 27,8% chunk bị trộn hai cột PDF. Thêm `NEW-05`, `NEW-06`, `TD-08`, `TD-09`; trả xong `TD-07`. `W0-04` hết bị chặn |
| 2026-08-17 | **`W1-08` xong — index baseline đã build**: 60 tài liệu → 15.814 chunk (768 chiều, cuda, 202s). Thêm `pipeline/indexing/` (config + corpus loader + build_index), `configs/indexing/{baseline,smoke}.yaml`, 59 test mới. Trả xong `TD-02`, `TD-03`; thêm `NEW-03`, `NEW-04`, `TD-06`, `TD-07`. Ghim torch CUDA (`cu126`) vì wheel PyPI trên Windows là bản CPU-only |
| 2026-08-17 | **Corpus nguồn (a) xong**: 60 tài liệu World Bank (40 EN + 20 VI) qua `scripts/fetch_corpus.py`. Thêm `pipeline/corpus/` (manifest ép giấy phép + adapter WDS) và 23 test. Phát hiện ADB chặn truy cập tự động (403) → phải tải tay qua `seed_list`. Thêm `NEW-02`, `TD-04`, `TD-05`; tổng 72 → 73 task |
| 2026-08-17 | **Hoàn thành 8/13 task tuần 1** (`W1-01`…`W1-07`, `W1-12`): 144 unit test + 18 integration test xanh, ruff + mypy strict pass, coverage `rag_core` 81%. Bằng chứng: `reports/tasks/w1-foundation.md`. Thêm `NEW-01`, `TD-01`…`TD-03`; tổng 71 → 72 task |
| 2026-08-17 | **Đổi yêu cầu Python từ ≥3.11 lên ≥3.12** — stub numpy dùng cú pháp `type` statement chỉ có từ 3.12, mypy strict không chạy được với `python_version=3.11`. Ảnh hưởng lựa chọn image ở `W0-05` |
| 2026-08-17 | **Bỏ LangChain khỏi `rag_core`** — viết lại recursive + semantic splitter thuần Python. Phụ thuộc nặng (`torch`, `qdrant-client`) chuyển sang optional extras để unit test chạy không cần GPU stack |
| 2026-08-14 | Tạo plan kỹ thuật `2026-08-14-rag-upgrade-proposal.md` (chốt 4 quyết định: LLM hybrid router · Docker Compose + HF Spaces · corpus VI–EN · 4–6 tuần) |
| 2026-08-14 | Thay diagram ASCII §2 bằng Mermaid + fallback text (ASCII bị vỡ khi trộn dấu tiếng Việt) |
| 2026-08-14 | Tạo `CHECKLIST.md` — 67 task, 6 gate, quy ước trạng thái & Definition of Done |
| 2026-08-14 | **Chuyển GPU thuê sang RunPod:** chốt Secure Cloud + RTX 4090 24GB + Pod On-Demand + Network Volume (tránh trả tiền GPU lúc tải trọng số). Spot chỉ dùng sau khi job resumable. Viết lại §3.6 mục GPU thuê; thêm rủi ro quên terminate pod |
| 2026-08-14 | **Bảo mật GPU thuê + corpus:** chốt quy tắc máy thuê không mang API key (job GPU-bound tự chứa); thứ tự nền tảng Kaggle → Colab → HF Jobs → vast.ai; bắt buộc corpus công khai (cấm tài liệu khách hàng Enigmas — rủi ro NDA). Thêm `W0-08`; siết `W0-03`, `W0-05`; tổng 69 → 70 task |
| 2026-08-14 | **Đổi nhà cung cấp LLM:** bỏ Claude (không có key) → DeepSeek primary + OpenRouter fallback + vLLM cho pipeline. Phát hiện GPU local là RTX 4060 8GB → không thể self-host generator ở serving. Thêm `W0-06`, `W0-07`, `W5-11` (generator ablation); tổng 67 → 69 task. Thêm §3.6 vào plan (phân vai LLM + ngân sách VRAM + 3 quy tắc tái lập) |
