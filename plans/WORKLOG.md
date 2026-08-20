# WORKLOG — nhật ký phiên làm việc

> Mục đích: nếu phiên Claude Code bị ngắt giữa chừng (hết quota 5h, mất mạng, đóng máy),
> file này cho biết **đang làm dở tới đâu** và **lệnh nào để tiếp tục**.
> Trạng thái chính thức của từng task vẫn nằm ở [`CHECKLIST.md`](CHECKLIST.md).
>
> **Phiên mới nhất: 2026-08-20 (cuối file)** — tuần 1 xong 13/13; W2 xong `TD-11` (kết quả âm), `W2-01` BGE-M3 (nDCG@10 0,1621 → 0,4442) và `W2-02` (Qdrant hybrid schema). Việc tiếp theo: `W2-03`.
> Sắp xếp theo thứ tự thời gian, mục mới thêm vào **cuối**.

---

## Phiên 2026-08-17 · Tuần 1 — nền móng

**Mục tiêu phiên:** hoàn thành 8 task W1 không bị chặn — `W1-01` … `W1-07` và `W1-12`.

### Lệnh để kiểm tra trạng thái hiện tại

```bash
cd D:/studioproj/RAG-Chatbot   # đổi tên 2026-08-20, xem reports/rename-workspace.md
uv sync --extra dev --extra qdrant   # cài môi trường
make lint                            # ruff check + ruff format --check + mypy strict
make test                            # 144 unit test, không cần Docker, ~3s
make up && make test-integration     # cần Docker Desktop đang chạy
```

### Bảng tiến độ phiên

| Task | Nội dung | Trạng thái | Ghi chú |
|---|---|---|---|
| `W1-01` | Monorepo skeleton + tooling | ✅ xong | `make lint` + `make test` exit 0 |
| `W1-02` | Pydantic schemas | ✅ xong | 20 test |
| `W1-03` | Port chunking (bỏ LangChain) | ✅ xong | 35 test |
| `W1-04` | SQLite cache thay pickle | ✅ xong | 16 test |
| `W1-05` | docker-compose hạ tầng | ✅ xong | 4 integration test, 3 service healthy |
| `W1-06` | Embedding provider | ✅ xong | 14 test |
| `W1-07` | Dense retriever Qdrant | ✅ xong | 14 integration test |
| `W1-12` | Retrieval metrics + report | ✅ xong | 35 + 19 test |
| `W1-08` | Build index (corpus → Qdrant) | ✅ xong | 59 test mới · index baseline 15.814 chunk |
| `W1-10` | Sinh nháp golden set | ✅ xong | 100 test mới · 266 câu · $0,5821 |

**Tổng kết phiên:** 144 unit test + 18 integration test xanh · `ruff` + `mypy --strict` sạch ·
coverage `rag_core` 81%. Bằng chứng: [`reports/w1-foundation.md`](reports/w1-foundation.md).

Ngoài scope dự kiến, đã thêm: `test_architecture_boundaries.py` (canh chiều phụ thuộc
hai plane) và `test_settings.py`.

### Quyết định kỹ thuật phát sinh trong phiên

*(đây là thứ dễ quên nhất sau khi ngắt phiên — mọi quyết định đều ghi lại đây)*

1. **Bỏ LangChain khỏi `rag_core`.** Bản POC dùng `RecursiveCharacterTextSplitter` +
   `SemanticChunker` của `langchain-experimental`. Cả hai được viết lại thuần Python
   (~150 dòng tổng cộng) trong `packages/rag_core/chunking/`. Lý do: `langchain-experimental`
   kéo cây phụ thuộc nặng và API hay đổi; tự viết thì test được từng nhánh.

2. **Phụ thuộc nặng nằm ở extras.** `sentence-transformers`/`torch` → extra `embeddings`,
   `qdrant-client` → extra `qdrant`. Unit test chạy không cần chúng nhờ
   `HashingEmbeddingProvider`. `make test` chạy ~3 giây.

3. **`HashingEmbeddingProvider` thay vì fake trả vector ngẫu nhiên.** Test của semantic
   chunking cần embedding có **tương đồng thật**; vector ngẫu nhiên deterministic sẽ
   khiến bài test pass kể cả khi thuật toán sai. Provider này là bag-of-words hashing
   đã chuẩn hoá — dùng cho test và CI, **không** dùng cho eval thật.

4. **Ba sai lệch có chủ ý so với `enhanced_chunking.py`** (ghi ở docstring
   `chunking/base.py`, ảnh hưởng tới số baseline `W1-13`):
   - `config_hash` không làm tròn tham số nữa. Bản cũ làm tròn `chunk_size` về bội 100
     → `1000` và `1049` dùng chung cache entry, đủ để một vòng ablation báo hai cấu
     hình cho kết quả y hệt.
   - Hậu xử lý áp **theo từng tài liệu**, không trên danh sách gộp (bản cũ gộp chunk
     nhỏ vào chunk của tài liệu khác).
   - Ép kích thước trước, thêm ngữ cảnh hàng xóm sau (bản cũ làm ngược).

5. **`neighbor_context_chars` mặc định TẮT.** Bản POC luôn bật ở mức 100 ký tự.
   ⚠️ **Config baseline của `W1-13` phải đặt `neighbor_context_chars=100`** để tái lập
   đúng hệ thống hiện tại.

6. **Cache theo từng tài liệu, không theo cả corpus.** Bản POC hash chuỗi nối của mọi
   tài liệu → sửa một file là mất sạch cache. Đây cũng là nền cho `W3-07`.

7. **Named vector `dense` trên Qdrant ngay từ W1.** Collection tạo bằng vector vô danh
   không thêm được named vector mà không build lại toàn bộ index — W2 sẽ thêm sparse.

8. **Point ID = UUIDv5 sinh từ `chunk_id`.** Tính idempotent nằm ở tầng store chứ không
   ở script build index → `W1-08` thừa hưởng sẵn.

9. **Yêu cầu Python ≥ 3.12** (ban đầu định 3.11). Lý do: stub của numpy dùng cú pháp
   `type` statement chỉ có từ 3.12, mypy strict không chạy được với `python_version=3.11`.
   ⚠️ Ảnh hưởng: image RunPod ở `W0-05` phải có Python ≥ 3.12.

10. **Truy vấn `unanswerable` trả `None` ở mọi metric xếp hạng, không trả `0.0`.**
    Recall trên tập rỗng là không xác định. Nhóm này đo riêng bằng refusal correctness
    ở `W5-02`.

11. **Hai lỗi hạ tầng chỉ lộ khi chạy thật** (không unit test nào bắt được — lý do
    `W1-05`/`W1-07` có integration test riêng thay vì mock Qdrant):
    - `QDRANT__SERVICE__API_KEY: ${QDRANT_API_KEY:-}` khiến biến **luôn tồn tại** với
      giá trị rỗng → Qdrant bật xác thực và trả 401 cho mọi request. Sửa bằng dạng
      danh sách không có dấu `=`.
    - `localhost` trên Windows resolve `::1` trước, Docker chỉ bind `127.0.0.1` →
      **mỗi request Qdrant chậm đúng 2 giây**. Toàn bộ default đã đổi sang `127.0.0.1`;
      integration suite từ ~5 phút xuống 30 giây. Nếu không phát hiện bây giờ thì p95
      latency đo ở W5/W6 sẽ cộng thêm 2000 ms cố định mà rất khó truy nguyên nhân.
    - `qdrant-client` ghim `>=1.15,<1.16` cho khớp major/minor với image server.

12. **Loại `*.md` khỏi ruff.** Ruff 0.16 format cả code block Python bên trong Markdown,
    và nó đã lặng lẽ sửa `README.md`, `enhanced_chunking_presentation.md`,
    `state_chat_presentation.md` — tài liệu viết tay, không phải thứ định đụng tới.
    Đã hoàn nguyên bằng `git checkout` và thêm `*.md` vào `extend-exclude`.

13. **Bộ tải corpus** (`scripts/fetch_corpus.py` + `pipeline/corpus/`). Ba điều học được:
    - World Bank WDS API: **không truyền tham số `fl`** — nó làm rụng đúng hai field
      `pdfurl`/`txturl` cần nhất. Lấy đủ rồi lọc phía mình.
    - Ưu tiên `txturl` hơn `pdfurl`: text đã trích sẵn, dùng được ngay ở W1 mà không
      phải đợi Docling ở `W3-01`.
    - **ADB trả 403 cho mọi truy cập tự động.** Không viết adapter được; tài liệu ADB
      phải tải tay rồi khai báo qua block `seed_list`.

14. **`limit` nghĩa là tổng mục tiêu của nguồn, không phải "thêm N mỗi lần chạy".**
    Bản đầu tiên chạy lần hai đã âm thầm tải thêm 30 tài liệu nữa mà manifest vẫn hợp
    lệ nên không có gì báo động. Đã sửa: trừ đi số đã có theo **tên block** (không phải
    theo loại nguồn, nếu không hai block cùng loại sẽ ăn chung hạn mức).

15. **Giấy phép được ép bằng code, không bằng lời hứa.** `CorpusEntry` từ chối mọi
    giấy phép ngoài allowlist. Đáng chú ý: **giấy phép có `ND` bị từ chối** — chunking
    + sinh context bằng LLM (`W3-04`) là tạo tác phẩm phái sinh, `ND` chỉ cho phép
    phát tán nguyên bản.

16. **`Chunker.prepare(n_documents)` — lỗi nghiêm trọng nhất của phiên này.**
    Cache chunk hoạt động theo **từng tài liệu**, nên `CachedChunker` gọi
    `inner.chunk([doc])` 60 lần. `HybridChunker` quyết định nhánh theo
    `len(documents)`, thấy `n=1` mỗi lần, ngưỡng là 5 → **luôn chọn semantic**.
    Bản POC truyền cả corpus một lượt nên với 60 tài liệu nó chạy fixed. Nếu
    không phát hiện thì baseline `W1-13` đo một chiến lược chunking khác hẳn.
    Sửa: người gọi khai báo tổng số tài liệu trước, khai báo tường minh thắng
    suy đoán theo lô. Log xác nhận `chunker hybrid->fixed`.

17. **`HybridChunker.name` giờ gồm cả nhánh đã chọn** (`hybrid->fixed:...`).
    Tên chunker là một phần khoá cache; tên cũ chỉ có `config_hash` nên hai lần
    chạy hybrid — một rơi semantic, một rơi fixed — đọc lại kết quả của nhau.
    Nhánh semantic còn kèm tên model embedding, nhờ đó ablation đổi model ở W2
    cũng không đọc nhầm.

18. **Ba tầng idempotent; point ID xác định chỉ là tầng 1.** Tầng 2: tài liệu
    **ngắn lại** để lại point mồ côi `doc::00030`… mà không upsert nào ghi đè —
    chúng không trùng lặp, chúng trỏ tới văn bản không còn tồn tại, và retriever
    vẫn trả về. Tầng 3: `fingerprint` trong state chặn việc trộn hai cấu hình
    vào một collection. Thứ tự cố ý là **upsert trước, xoá sau** — chết giữa
    chừng thì thừa chunk cũ (chạy lại dọn được), chứ không mất hẳn tài liệu.

19. **State file không phải nguồn sự thật, Qdrant mới là.** Trước khi tin state,
    script đối chiếu tổng số chunk với `collection.count()`; lệch thì bỏ state
    và index lại toàn bộ. Ca thật: `make down-clean` mà quên xoá `.cache/`.

20. **p95 độ trễ truy hồi 15.219 ms hoá ra là thời gian nạp model** (p50 chỉ
    31 ms). Truy vấn đầu tiên gánh cả việc nạp `sentence-transformers` vốn lazy.
    Thêm một lượt warm-up: p95 còn **98 ms**. Đây là con số làm ngưỡng cho gate
    hiệu năng W5/W6 — để nguyên thì gate đo thời gian khởi động.

21. **Hệ số phình 1.24x của `neighbor_context_chars=100`.** Text đem embed đi từ
    14,3 lên 17,7 triệu ký tự. Con số này chưa nói kỹ thuật đó tốt hay dở, nó
    nói **giá phải trả** — so recall giữa bật và tắt là việc của W2.

22. **`fingerprint` cố ý không gồm `device`/`batch_size`.** Chạy trên GPU thuê
    hay laptop phải ra cùng một index về mặt logic; nếu `device` vào fingerprint
    thì mỗi lần đổi máy là build lại vài giờ GPU — đúng thứ kiến trúc hai plane
    sinh ra để tránh. Có test canh cả hai chiều.

23. **Wheel `torch` mặc định trên PyPI cho Windows là bản CPU-only.** Cài từ đó
    thì mọi thứ vẫn chạy, chỉ chậm hơn nhiều và `torch.cuda.is_available()` trả
    `False` mà không báo lỗi gì. Đã ghim index `download.pytorch.org/whl/cu126`
    trong `[tool.uv.sources]`. Xác nhận `torch 2.13.0+cu126`, build chạy `cuda`.

24. **`plans/` vừa xuất hiện trong `.gitignore`** (không phải do phiên này thêm).
    File trong đó đã tracked nên vẫn commit được, nhưng file **mới** bị bỏ qua
    âm thầm — `reports/w1-08-build-index.md` phải `git add -f`. Ghi vào `TD-07`.

25. **`deepseek-chat` là BÍ DANH, không phải một model.** Xác nhận trực tiếp
    trên API: cả `deepseek-chat` lẫn `deepseek-reasoner` đều được phục vụ bởi
    `deepseek-v4-flash`. Đây đúng là vấn đề mà quy tắc cứng #1 nói về OpenRouter
    preset, chỉ kín đáo hơn — tên trông như một model cụ thể nhưng là con trỏ do
    server nắm. Mặc định dự án đổi sang slug thật; gọi bằng bí danh vẫn được
    nhưng có cảnh báo.

26. **Model suy luận làm chẩn đoán `max_tokens` sai hoàn toàn.** Triệu chứng là
    hàng loạt "Response không phải JSON hợp lệ" với nội dung rỗng. Đo lại:
    `completion_tokens=1770` nhưng `len(text)=515` ký tự — `deepseek-v4-flash`
    tiêu 1.500–3.000 token cho chuỗi suy luận KHÔNG nằm trong `content`. Nâng
    `max_tokens` lên 6.000 và tách `finish_reason=="length"` thành cảnh báo
    riêng. Vẫn còn 13,5% lời gọi bị cắt — xem `TD-08`.

27. **Chạy song song 6 luồng: >1 giờ → 640 giây.** Mỗi lời gọi ~25 giây và gần
    như toàn bộ là chờ mạng. Kết quả vẫn ráp theo đúng thứ tự lô để hai lần chạy
    cùng seed cho ra cùng một file.

28. **Job trả tiền dài phải có checkpoint.** Lượt đầu treo ở phút 40, mất sạch.
    Giờ ghi nối sau mỗi lời gọi; chạy lại bỏ qua lô đã xong. Lô mà model trả về
    rỗng cũng ghi dấu — nếu không thì mỗi lần chạy lại đều trả tiền cho cùng một
    câu trả lời rỗng.

29. **27,8% chunk của corpus bị trộn hai cột PDF.** Bản `.txt` World Bank giữ
    nguyên vị trí ký tự của trang hai cột nên cột trái/phải đan xen theo dòng.
    Nó gồm toàn từ hợp lệ nên mọi bộ lọc theo tỉ lệ chữ cái đều cho qua; chỉ
    `gutter_ratio` (khoảng trắng dài ở GIỮA dòng) lộ ra. Thêm hai bộ lọc nữa:
    `mean_words_per_line` (loại chú thích biểu đồ) và `skip_leading_chunks`
    (loại bìa/bản quyền/mục lục). Tổng cộng loại ~60% lô → `overshoot=3.0`.
    ⚠️ Ba bộ lọc này CHỈ áp cho việc chọn mẫu sinh câu hỏi, KHÔNG áp cho index.

30. **Model không được tự viết `chunk_id`.** Nó chỉ trả về chỉ số đoạn văn trong
    danh sách được đưa; code ánh xạ sang id thật. Cho model tự viết id là mở
    đường cho cả golden set trỏ vào hư không. Kèm theo: `quote` được đối chiếu
    lại với chunk (16/266 câu không kiểm chứng được), và `multi_hop` chỉ dùng
    một chunk thì bị hạ nhóm về `factoid`.

### Việc còn dở / cần làm tiếp

- [x] **Đã commit + push** lên nhánh `feat/w1-foundation` (2026-08-17):
      `f5ec22b` nền móng W1 · `c253fa5` bộ tải corpus W0-03.
      Cố ý **không** push thẳng `main` vì repo này đang là link trong CV — trang chủ
      repo giữ nguyên bản cũ cho tới khi bạn chủ động merge.
- [ ] Quyết định merge `feat/w1-foundation` vào `main` hay giữ nhánh (liên quan `W0-02`).
- [x] `W1-08` — **xong**: `reports/w1-08-build-index.md`. Collection `rag_baseline`
      có 15.814 chunk; `make index` chạy lần hai không ghi thêm gì.
- [x] `W1-10` — **xong**: `reports/w1-10-goldenset-draft.md`, 266 câu nháp.
- [x] `W1-11` — **xong ở phiên 2026-08-20 nhưng review bằng MODEL, không phải người**:
      `golden_v1` 242 câu, loại 24/266. Việc còn lại của bạn nằm ở `TD-13`.
- [x] `W1-09` (DVC) và `W1-13` (đo baseline) — **xong ở phiên 2026-08-20**.
- [ ] Corpus nguồn (b) pháp luật và (c) HOSE — cần bạn chọn tay, khai báo qua `seed_list`.
      ⚠️ Phải dựng DVC remote dùng chung **trước** đó (`TD-10`).
- [x] Gate `G1` 4/4 mục — nhưng là **PASS có điều kiện** (`TD-13`), ký hiệu vẫn 🟡.

> ⬇️ **Mọi mục "còn dở" ở trên đã được xử lý hoặc chuyển thành `TD-xx` trong phiên
> 2026-08-20.** Đọc mục "Vấn đề đang mở" ở cuối file để biết trạng thái hiện tại.

### Nếu phiên sau bắt đầu từ đây

Đọc theo thứ tự: mục "Quyết định kỹ thuật" ở trên → `reports/w1-08-build-index.md`
→ `reports/w1-foundation.md` →
`CHECKLIST.md` §1 Dashboard và §10 Đang bị chặn. Chạy `make lint && make test` để xác
nhận repo vẫn xanh trước khi làm tiếp.

---
## Phiên 2026-08-20 · Tuần 1 — golden set, DVC, baseline

**Mục tiêu phiên:** 5 task còn lại của W1 (`W1-08`…`W1-13`) → đóng gate `G1` → sẵn sàng vào W2.

### Lệnh để kiểm tra trạng thái hiện tại

```bash
cd D:/studioproj/RAG-Chatbot
uv sync --extra dev --extra qdrant --extra embeddings
make lint                            # ruff check + format --check + mypy strict (74 file)
make test                            # 534 unit test, không cần Docker
make up && make test-integration     # 38 integration test, cần Docker Desktop
make data-verify                     # đối chiếu DVC <-> corpus_manifest.csv
make goldenset-verify                # checksum golden_v1 (242 câu)
make eval-retrieval                  # đo lại baseline, ~20 giây
```

### Bảng tiến độ phiên

| Task | Nội dung | Trạng thái | Ghi chú |
|---|---|---|---|
| `W1-09` | DVC version corpus | ✅ xong | 23 test · clone sạch `dvc pull` 61 file/1,9s · 60/60 sha256 khớp |
| `TD-12` | Neo golden set theo span ký tự | ✅ trả xong | 121 test mới · 0 câu mất nhãn ở `chunk_size` 1000/600/400 |
| `W1-11` | Triage + review + freeze `golden_v1` | ⚠️ xong **có điều kiện** | 242 câu · review bằng **model**, không phải người → `TD-13` |
| `W1-13` | Đo baseline | ✅ xong | recall@5 0,1746 · MRR 0,1660 · chạy lại sai số **0,0000%** |
| `W0-04` | Credential & biến môi trường | ✅ xong | Trước bị đánh `[?]` nhầm; DoD đã đủ từ 2026-08-17 |

**Tổng kết phiên:** 534 unit + 38 integration test xanh · `ruff` + `mypy --strict` sạch trên
74 file · coverage `rag_core` 80% · 8 commit (`b48a34b` … `b9fcbce`) đã push lên
`feat/w1-foundation`. Tuần 1 **13/13**, gate `G1` 🟡 (4/4 kỹ thuật, 1 điều kiện: `TD-13`).

### Quyết định kỹ thuật phát sinh trong phiên

*(tiếp số từ phiên trước — quyết định 1–30 ở phiên 2026-08-17)*

31. **DVC remote nằm ở `.dvc/config.local`, không phải `.dvc/config`.** `dvc remote add -d`
    ghi đường dẫn tuyệt đối `D:/dvc-remote/...` vào file **được commit** → mọi clone nhận
    remote trỏ vào ổ đĩa không tồn tại. Cùng loại lỗi với đường dẫn tuyệt đối trong
    `.venv/*.pth` gặp lúc đổi tên workspace. Phải dùng `--local`.

32. **`data/golden` KHÔNG đưa vào DVC** (khác checklist gốc). Golden set là **thước đo**;
    thứ cần nhất ở nó là **diff đọc được** lúc review — ai đổi nhãn câu nào, từ gì sang gì.
    DVC thay file bằng một hash nên mất đúng thứ đó. 284 KB text không phải lý do để tránh
    git. Tính tái lập không mất: một commit ghim cả hai. Lý lẽ đầy đủ: `reports/w1-09-dvc.md` §3.2.

33. **Corpus giờ có HAI cơ chế versioning, và chỗ cả hai đều mù là phép so SỐ LƯỢNG.**
    sha256/manifest và md5/DVC. Thêm file vào `data/corpus/` rồi `dvc add` mà quên manifest
    thì **không lỗi nào nổ ra**: build index bỏ qua file lạ, `dvc status` vẫn sạch,
    `dvc push` vẫn đem nó lên remote. `pipeline/corpus/dvc_state.py` canh đúng chỗ đó.

34. **Bất đối xứng của tín hiệu triage — quyết định quan trọng nhất của phiên.**
    Câu `unanswerable` mà retriever tự tin = **bằng chứng nhãn sai** (mệnh đề về corpus bị
    phản chứng). Câu trả lời được mà retriever trượt **không** phải bằng chứng nhãn sai —
    đó chính là thứ eval tồn tại để đo, loại nó đi là **tự thổi phồng recall baseline**.
    Nên tín hiệu thứ hai xếp **cuối** hàng đợi, đề xuất `accept`, và có 2 test canh
    (`TestSignalB`). Không có luật này thì 161/226 câu "khó" bị dọn khỏi tập đo và recall@5
    nhảy từ 0,17 lên khoảng 0,6 mà không cải tiến gì cả.

35. **Ngưỡng nghi ngờ hiệu chuẩn TỪ DỮ LIỆU, không phải hằng số.** Trung vị điểm top-1 của
    các câu trả lời được = **0,5797**. Ghim `0.8` là đoán, vì "điểm cao" phụ thuộc model
    embedding và corpus. Kèm theo: điểm cao **không** đủ để phân loại lại tự động — ví dụ
    đầu hàng đợi có điểm 0,7287 mà chunk top-1 vẫn không trả lời được câu hỏi. Điểm cao
    chứng minh *cùng chủ đề*, không chứng minh *trả lời được*.

36. **`TD-12` — `chunk_id` thuần vị trí làm golden set hỏng ÂM THẦM.**
    `chunk_id` là số thứ tự trong tài liệu, nên đổi `chunk_size` khiến nhãn trỏ vào **văn bản
    khác mà vẫn hợp lệ**. Đo thật: nhãn `chunk_id` cũ "hợp lệ" 226/226 ở cả ba cấu hình
    1000/600/400 trong khi trỏ vào chỗ khác; nhãn span mất **0** câu. Sửa **trước** review vì
    6–8 giờ công người sẽ hỏng ở đúng việc đầu tiên của W2 (hạ `chunk_size`).

37. **Span là PROVENANCE, không phải chỉ thị cắt.** `chunk.content` không bằng
    `text[start:end]` — splitter nối các mảnh bằng ký tự xuống dòng bất kể nguyên bản ngăn
    nhau bằng gì, và phép split bỏ mảnh rỗng nên hai dòng trống thành một. Cảnh báo này ghi
    thẳng vào docstring của `TextSpan` và `chunking/pieces.py`, vì người đọc sau sẽ rất tự
    nhiên dùng span để slice lại văn bản.

38. **Luật chồng lấp span↔chunk phải ĐỐI XỨNG.** Bản đầu chỉ xét `overlap/span.length` →
    **mất 40/226 nhãn ở `chunk_size=400`**, vì chunk nhỏ nằm trọn trong span rộng thì tỉ lệ
    theo span rất thấp. Luật đúng: `overlap/span.length >= r` **HOẶC**
    `overlap/chunk.length >= r`. Chỉ lộ ra vì đem thử ở đúng cấu hình mà W2 sẽ dùng thật,
    không phải ở fixture tự bịa.

39. **Ánh xạ span trả rỗng thì GIỮ NHÃN CŨ.** `evaluate_run` bỏ qua câu có
    `relevant_chunk_ids` rỗng (coi là unanswerable) → nếu ánh xạ trượt mà trả rỗng thì câu
    khó nhất tự rơi khỏi tập đo và recall tăng. Cùng cái bẫy của quyết định 34, ở tầng khác.

40. **Thêm field optional vào pydantic KHÔNG làm `ValidationError`** — nên cache chunk phải
    có version trong **tên bảng** (`chunk_cache_v2`). Thêm `start_char`/`end_char` rồi đọc
    lại cache cũ thì được chunk **không có offset**, hợp lệ hoàn toàn, và `span_resolution`
    im lặng rơi về nhãn `chunk_id` cũ. Đây là lý do `chunks_by_document` WARN kèm gợi ý
    `--recreate` khi thấy chunk thiếu offset.

41. **Bất biến số một của lần refactor chunking: nội dung chunk không đổi MỘT BYTE.**
    Kiểm bằng digest trên corpus thật 60 tài liệu trước/sau (`b381634d51e39365` /
    `e00bc87aaffe792e` cho hai cấu hình) + class `TestTextUnchanged` so với biểu thức
    tiền-refactor. Không có phép này thì mọi thay đổi baseline sau đó lẫn giữa "cải tiến"
    và "refactor làm hỏng chunking".

42. **`freeze` làm rơi `relevant_spans` — lỗi tệ nhất của phiên.** `_apply()` không truyền
    field đó nên `golden_v1.jsonl` ra với `relevant_spans` rỗng: toàn bộ công của `TD-12`
    bị vô hiệu ở **đúng bước cuối**, mà **không có triệu chứng nào** — file hợp lệ, eval
    chạy, số vẫn ra. Phát hiện nhờ để ý `config.span_resolution` **vắng mặt** trong JSON
    baseline đầu tiên. Quyết định kèm theo: `edit` có điền `new_relevant_chunk_ids` thì
    **bỏ span**, vì ánh xạ span ghi đè `chunk_id` nên giữ span cũ sẽ âm thầm huỷ sửa tay
    của người review.

43. **Provenance review ghi vào DỮ LIỆU, không phải vào report.** `reviewed_by_human` (bool)
    tách khỏi `reviewed_by` (chuỗi); `freeze` chỉ đặt `true` khi `--reviewer human` với
    `HUMAN_REVIEWER` là hằng số hard-code, và in WARNING mỗi trường hợp khác. Nhờ vậy
    **không chỗ nào trong repo có thể khẳng định "người đã xác nhận"** khi thực tế là model.

44. **`quote_unverified` là dấu hiệu LỖI TRÍCH XUẤT PDF, không phải model bịa.** 15/16 trích
    dẫn có thật trong corpus; cái làm chúng không khớp là trộn hai cột chèn chữ vào giữa từ
    (`bổ sung` thành `bổ chuyên sung`), số chú thích chèn giữa câu, gạch nối cuối dòng, và
    trích dẫn vắt qua ranh giới chunk. Đọc cờ này thành "model bịa" là loại oan 16 câu.

45. **Kiểm câu `unanswerable` phải tra VĂN BẢN GỐC, không phải retriever.** Retriever đang
    đơn ngữ và chỉ nhìn 256 token đầu mỗi chunk — dùng nó để xác nhận "corpus không có" là
    lấy điểm yếu của hệ thống làm bằng chứng về dữ liệu. Tra thẳng text: **7/40 câu sai nhãn**
    (lạm phát 2025 = 4,5–5% · GDP 2030 = 5,45% trong bảng CGE của CCDR · 1.584 thủ tục tiền
    kiểm · nghèo đa chiều 2022), tỉ lệ sai 17,5% trong nhóm.

46. **Phép kiểm máy sửa lại mắt thường.** Tôi nghi `factoid-69f6fec136` (2.938 tỷ đồng PFES)
    không có căn cứ; phép kiểm grounding chứng minh số đó có thật, chỉ nằm **ngoài span đã
    thu hẹp**. Tin mắt thì đã loại oan một câu đúng.

47. **Baseline chạy lại sai số 0,0000%, không phải "<1%".** Dense retrieval trên vector đã
    ghi là phép tính xác định. Lượt hai chạy với `DEEPSEEK_API_KEY` và `OPENROUTER_API_KEY`
    **rỗng** → đồng thời là bằng chứng cho hạng mục "eval truy hồi không cần LLM API" của `G1`.

48. **recall@5 = 0,1746 thấp CÓ LÝ DO đã định lượng, và cố ý không sửa trước khi đo.**
    (1) `cross_lingual` = 43/209 câu tức **20% tập đo**, model embedding đơn ngữ → recall@5
    bằng 0 là con số **đúng**, không phải lỗi đo. (2) `TD-11`: 56,8% chunk bị cắt ở 256 token,
    15,7% text không tới được vector. Sửa trước rồi gọi kết quả là "baseline" thì mọi so sánh
    về sau đo lẫn cải tiến này vào.

### Vấn đề đang mở (đọc trước khi bắt đầu W2)

- ⚠️ **`TD-13` — golden set review bằng MODEL.** `reviewed_by_human=false`. Không chặn W2
  (so sánh tương đối giữa các cấu hình vẫn hợp lệ vì cùng một thước đo), nhưng **chặn** chữ
  "human-verified" ở README / CV / phỏng vấn, và chặn `G1` 🟡 → ✅. Việc cần làm: người đọc
  lại **33 câu `unanswerable` + 43 câu `cross_lingual`** (hai nhóm loại nhiều nhất = hai nhóm
  model kém tự tin nhất) rồi `make goldenset-freeze` với `--reviewer human`. Đo lại: 20 giây.
- ⚠️ **`TD-11` là hạng mục đầu tiên của W2, làm TRƯỚC `W2-01`.** Nếu hạ `chunk_size` và đổi
  sang BGE-M3 cùng lúc thì không tách được phần cải thiện nào của ai — mà BGE-M3 (cửa sổ
  8192 token) xoá luôn nguyên nhân truncation, nên gộp là mất hẳn một cặp số trước/sau.
- ⚠️ **Build index cho W2 phải dùng `--recreate`.** Point cũ không mang `start_char`/`end_char`
  thì `span_resolution` rơi về nhãn `chunk_id` cũ và **im lặng đo sai** (quyết định 40).
  Có WARNING, nhưng đừng dựa vào việc đọc log.
- ⚠️ **Không sửa `configs/indexing/baseline.yaml`.** Nó là mốc so sánh của cả 6 tuần; mọi
  cấu hình W2 phải là file mới + collection Qdrant riêng.
- `TD-14` — 7 `reference_answer` lẫn ngữ cảnh sinh ("Theo đoạn văn 1"). Vô hại cho eval truy
  hồi (nó chỉ đọc `relevant_spans`), phải dọn **trước `W5-01`** vì judge sẽ trừ điểm oan.
  Câu hỏi thì sạch: 0/242 câu có lỗi này.
- `TD-15` — nhãn liên quan chưa đầy đủ (một dữ kiện ở hai tài liệu, chỉ 1 chỗ được gán).
  Hiếm (1/78 theo phép thử Jaccard) và sai **theo chiều an toàn**: baseline là cận dưới.
- `TD-16` — 10/33 câu `unanswerable` cùng một khuôn "hỏi về nước khác".
- `TD-10` — DVC remote chỉ là thư mục local. **Bắt buộc dựng remote dùng chung trước khi thêm
  nguồn (b) pháp luật và (c) HOSE**, vì hai nguồn đó chọn tay, không script nào tải lại được.
- `TD-05` — đọc tay trang giấy phép của ~5 tài liệu **trước khi** push repo lên public.
- `TD-08` — 22/163 lời gọi LLM vẫn bị cắt ở `max_tokens=6000`.
- `table_lookup` chỉ 4 câu — chờ nguồn (c) HOSE + `W3-01`. Mọi số của nhóm này (recall
  0,0000) **không suy ra được gì** với n=4.
- `W0-02` — quyết định merge `feat/w1-foundation` vào `main` hay tách repo mới.

### Nếu phiên sau bắt đầu từ đây

1. `make lint && make test` để xác nhận repo vẫn xanh (534 unit test).
2. Đọc `reports/w1-11-review.md` §1 (ai review và điều đó nghĩa là gì) và §4 (baseline).
3. Đọc §4 của `CHECKLIST.md` — thứ tự tấn công W2 đã chốt: `TD-11` → `W2-01` → `W2-02` …
4. Việc đầu tiên gõ tay: tạo config mới với `chunk_size` ~600, build index bằng `--recreate`
   vào collection **riêng**, rồi `make eval-retrieval` để có cặp số trước/sau của `TD-11`
   **tách riêng** khỏi việc đổi model embedding.

---

## Phiên 2026-08-20 (tiếp) · Tuần 2 — `TD-11`

**Mục tiêu phiên:** hạng mục đầu của W2 — sửa truncation bằng cách hạ `chunk_size`,
lấy cặp số trước/sau.

**Kết quả: giả định của `TD-11` bị phản chứng.** Phép sửa chạy đúng (56,9% → 0,4%
chunk bị cắt) và không cải thiện gì đo được (`p = 0,711`). Thứ đáng giá nhất lại là
phát hiện về **thước đo**, không phải về `chunk_size`.

### Lệnh để kiểm tra trạng thái hiện tại

```bash
make truncation BUNDLE=baseline                  # đo TD-11, không cần Qdrant
make index BUNDLE=chunk550                       # ~257s trên RTX 4060
make eval-retrieval BUNDLE=chunk550              # ~20s
make eval-compare BASE=baseline CAND=chunk550    # McNemar + bootstrap
make test                                        # 600 unit test
```

### Bảng tiến độ phiên

| Việc | Trạng thái | Ghi chú |
|---|---|---|
| Đo truncation thành thứ quan sát được | ✅ | `max_sequence_tokens` + `count_tokens` + `TruncationStats`; `WARNING` trong mọi lần build |
| Hiệu chuẩn `chunk_size` từ dữ liệu | ✅ | 574 ký tự (an toàn cho mọi ngôn ngữ); chốt 550 |
| `chunk550` + `chunk550nb55` build & đo | ✅ | 31.155 chunk mỗi config |
| Kiểm định cặp giữa hai lần chạy | ✅ | `*-per-query.jsonl` + `pipeline/eval/compare.py` |
| `TD-11` | ✅ trả xong phần đo | Kết luận **âm** — xem `reports/w2-td11-chunk-size.md` |
| `W2-01` BGE-M3 | ⬜ việc tiếp theo | Giữ `chunk_size=1000`, chỉ đổi model |

**Tổng kết:** 600 unit (+66) + 38 integration test xanh · `ruff` + `mypy --strict`
sạch trên 79 file.

### Quyết định kỹ thuật phát sinh trong phiên

*(tiếp số từ 48)*

49. **Việc đầu tiên không phải sửa lỗi mà là làm cho lỗi có triệu chứng.**
    `sentence-transformers` cắt ở `max_seq_length` không cảnh báo, không lỗi — đó
    là lý do `TD-11` sống được suốt W1. Nên `BuildReport` giờ mang `truncation` và
    in ở mức **WARNING**: một dòng INFO giữa mười dòng INFO khác thì cũng trốn
    được lần nữa.

50. **`None` = "không biết giới hạn", KHÁC "không có giới hạn".** Provider không
    đếm được token thì phép đo trả `None` và người gọi phải cảnh báo. Quy ước
    thành 0 là báo "không bị cắt" cho mọi model — đúng cái cách bug ban đầu trốn
    được. Có 3 test riêng cho bất biến này.

51. **`truncation=False` khi gọi tokenizer, nếu không phép đo thành hằng số 0.**
    Mặc định tokenizer *cắt ở `model_max_length`*, tức trả về đúng con số ngưỡng
    cho mọi text dài. Bỏ sót cờ này thì mọi cấu hình đều báo "đã sửa xong".

52. **Đếm kèm `[CLS]`/`[SEP]`**, và text dài **đúng bằng** giới hạn thì **không**
    bị cắt (`>` chứ không `>=`). Hai lỗi off-by-one theo hai chiều khác nhau.

53. **`truncated_ratio` và `tokens_lost_ratio` phải tách rời.** 90% chunk bị cắt
    mất mỗi chunk 1 token là chuyện nhỏ; 10% chunk mất mỗi chunk một nửa là mất
    hẳn nội dung. Một con số duy nhất không phân biệt được hai ca đó.

54. **Hiệu chuẩn `chunk_size` phải dùng phân vị THẤP của mật độ ký tự/token, không
    phải trung bình — tôi viết sai lần đầu.** Bản trung bình cho ra 946 ký tự,
    trong khi ở 1000 ký tự đã có 56,9% chunk bị cắt. Con số vô lý mà trông hợp lý,
    vì nó trả lời "chunk *trung bình* vừa khít cửa sổ" = ngưỡng mà một nửa vẫn bị
    cắt. Bản đúng: 574 ký tự. Có test hồi quy cho đúng cái bug này.

55. **`max_chunk_size` phải hạ cùng `chunk_size`** — nó là cùng một đại lượng. Để
    nguyên 1500 thì hậu xử lý vẫn thả ra chunk 1500 ký tự và chúng vẫn bị cắt;
    phép sửa chỉ có tác dụng một nửa.

56. **Nhãn neo theo span làm mẫu số của recall thay đổi khi đổi `chunk_size`.**
    Chunk nhỏ hơn → một span phủ nhiều chunk hơn → **1,38 → 1,96 nhãn/câu**.
    `recall@k = |tìm được ∩ liên quan| / |liên quan|` nên nó tụt `1 − 1,38/1,96 =
    29,6%` **kể cả khi truy hồi y nguyên**; thực tế tụt 25,8%. Suýt viết vào báo
    cáo là "hạ `chunk_size` làm tụt recall 26%".
    Metric miễn nhiễm: `hit_rate@k` (mẫu số 1), `precision@k` (mẫu số k), `MRR`.

57. **Không có điểm từng câu thì không kiểm định được gì.** `hit_rate@5` 0,2153 →
    0,2010 trên 209 câu là **45 câu xuống 42 câu** — chênh 3 câu. Đã thêm
    `{run}-per-query.jsonl` và `pipeline/eval/compare.py`: McNemar exact cho metric
    nhị phân, bootstrap cặp (seed cố định) cho metric liên tục. Không cần `scipy`.

58. **`compare.py` TỪ CHỐI so `recall@k`/`nDCG@k`/`MAP@k` khi số nhãn/câu khác
    nhau**, và in kèm "metric này tụt X% ngay cả khi truy hồi y nguyên". Bài học
    của quyết định 56 đóng thành hàng rào, không để trong docstring.

59. **`golden_v1` chỉ phân giải được mức chênh ≥ 6 điểm `hit_rate` (≈28% tương
    đối).** 209 câu, base rate ≈ 0,20: với ~30 câu đổi chiều phải lệch ~12 câu mới
    đạt `p < 0,05`. Đây là **giới hạn của thước đo**, và nó chi phối cả W2: xếp
    hạng 12 tổ hợp ablation bằng mức chênh vài phần trăm là tung đồng xu. Tin tốt:
    ngưỡng `G2` (+0,08 nDCG trên 0,1621 = +49% tương đối) nằm trong tầm đo.

60. **Hạ `chunk_size` là ĐÁNH ĐỔI, không phải thu hồi nội dung đã mất.** Baseline
    bị cắt nhưng mỗi vector vẫn đọc **~950 ký tự**; chunk550 không bị cắt nhưng mỗi
    vector chỉ đọc **678**. Ba cấu hình xếp đúng theo con số đó (950 → 678 → 589,
    `hit_rate@5` 0,2153 → 0,2010 → 0,1770) — chiều thì đơn điệu, độ lớn thì dưới
    ngưỡng phân giải.

61. **Giả thuyết "pha loãng bởi ngữ cảnh hàng xóm" bị số liệu bác.** `chunk550`
    giữ `neighbor_context_chars: 100` nên 36% mỗi chunk là text của chunk bên cạnh
    (baseline 20%). Tôi cho rằng đó là lỗi thiết kế thí nghiệm và dựng
    `chunk550nb55` để đối chứng — nó đi tiếp xuống, không lấy lại được gì. Lập luận
    "giữ giá trị tuyệt đối làm đổi giá trị tương đối" vẫn đúng; kết luận rút ra từ
    nó thì sai. Đo vẫn rẻ hơn suy luận.

62. **1 tài liệu dùng `Ê` (U+00CA) làm dấu cách** (`TD-17`). Font PDF map glyph dấu
    cách thành `Ê` → văn bản **không có ranh giới từ** → splitter rơi xuống mức ký
    tự → tokenizer nổ ra 0,63 token/ký tự (bình thường 0,20) → 31/36 chunk vẫn bị
    cắt kể cả ở `chunk_size=550`. Phạm vi hẹp và đã đo: 1/60 tài liệu, 1/242 câu
    golden set. Phép phát hiện rẻ: **tỉ lệ ký tự whitespace** (corpus p50 37,5%,
    tài liệu này 16,4%) — nên thành cổng chất lượng corpus ở `W3-01`.

63. **Chạy lại `baseline` sau khi thêm per-query cho đúng từng chữ số** trên cả 15
    metric. Lần xác nhận thứ ba cho tính xác định của đường eval.

### Vấn đề đang mở

- **`W2-01` là việc tiếp theo, và giữ `chunk_size=1000`.** Đừng quét thêm
  `chunk_size` với `vietnamese-bi-encoder`.
- ⚠️ **Mọi so sánh W2 từ giờ phải qua `make eval-compare`.** Bảng số không kèm
  `p`/CI thì không kết luận được gì ở mức chênh dưới 6 điểm `hit_rate`.
- ⚠️ Đổi `chunk_size` thì **không** dùng recall@k/nDCG/MAP. `compare.py` tự chặn,
  nhưng bảng Markdown của `eval-retrieval` thì không — nó vẫn in mọi metric.
- `TD-17` — tài liệu `Ê`, sửa ở `W3-01`.
- `TD-13` — golden set vẫn là review bằng model. Thêm câu hỏi sẽ cải thiện **cả**
  độ phân giải (quyết định 59) **và** `TD-13`; gộp hai việc được.
- Collection `rag_chunk550` + `rag_chunk550nb55` còn trong Qdrant (~62k point) để
  đối chiếu lại. Xoá bằng `make down-clean` rồi build lại từ config nếu cần chỗ.
- Các mục còn lại của phiên trước (`TD-10`, `TD-05`, `TD-08`, `W0-02`) không đổi.

### Nếu phiên sau bắt đầu từ đây

1. `make lint && make test` (600 unit test).
2. Đọc `reports/w2-td11-chunk-size.md` §8 (kiểm định + độ phân giải) và §10 (bước tiếp).
3. `W2-01`: thêm provider BGE-M3, `max_sequence_tokens` phải báo 8192. Chạy
   `make truncation` với config mới **trước** khi eval để xác nhận truncation về 0.
4. Config mới đặt tên riêng + collection riêng; **không** sửa `baseline.yaml`.

---

## Phiên 2026-08-20 (tiếp) · Tuần 2 — `W2-01` BGE-M3

**Mục tiêu phiên:** hạng mục W2 thật đầu tiên — đổi model embedding sang BGE-M3,
giữ nguyên chunking của baseline, lấy cặp số có kiểm định.

**Kết quả: mức tăng lớn nhất của dự án tới giờ, và một bài học về quy kết nguyên
nhân.** nDCG@10 `0,1621 → 0,4442`, cả 15 metric có ý nghĩa thống kê,
`cross_lingual` từ **0** lên `0,3023`. Nhưng mức tăng đó **không** phải công của
việc sửa truncation — xem quyết định 68.

### Lệnh để kiểm tra trạng thái hiện tại

```bash
make truncation BUNDLE=bgem3                  # 0/15814 chunk bị cắt
make eval-compare BASE=baseline CAND=bgem3    # McNemar + bootstrap, 15/15 metric
make test                                     # 644 unit test (~3s)
make test-gpu                                 # 11 test cần model thật (2,2GB)
```

Build lại từ đầu (~405 s trên RTX 4060, **bắt buộc** `--recreate` vì 768 → 1024 chiều):

```bash
python -m pipeline.indexing.build_index --config configs/indexing/bgem3.yaml \
  --recreate --report plans/reports/index-bgem3.json
python -m pipeline.eval.retrieval_eval --index-config configs/indexing/bgem3.yaml \
  --run-name bgem3
```

### Bảng tiến độ phiên

| Việc | Trạng thái | Ghi chú |
|---|---|---|
| `SparseVector` + `HybridVectors` | ✅ | Kiểu bất biến, cưỡng chế 3 bất biến lúc khởi tạo |
| Năng lực sparse tuỳ chọn trên `EmbeddingProvider` | ✅ | Mặc định `None`; `None` ≠ rỗng |
| `BgeM3EmbeddingProvider` | ✅ | Dense + sparse **một** forward pass |
| Truncation với cửa sổ 8192 | ✅ | 56,9% → **0,0%** (0/15814) |
| Index + eval + kiểm định | ✅ | 15/15 metric "khác biệt thật" |
| `W2-01` | ✅ | `reports/w2-01-bge-m3.md` |
| `W2-02` Qdrant sparse | ⬜ việc tiếp theo | Sparse đã có, cần chỗ chứa |

**Tổng kết:** 644 unit (+44) + 38 integration + 11 gpu test xanh · `ruff` +
`mypy --strict` sạch trên 83 file.

### Quyết định kỹ thuật phát sinh trong phiên

*(tiếp số từ 63)*

64. **Sparse của BGE-M3 không nằm trong `sentence-transformers`, và không nạp
    model lần thứ hai để lấy nó.** `modules.json` chỉ có
    `Transformer → Pooling → Normalize`; sparse là một `Linear(1024 → 1)` đặt lên
    `last_hidden_state`, trọng số ở `sparse_linear.pt` mà ST không đọc. Cách làm:
    nạp file đó riêng và **dùng lại chính `XLMRobertaModel` mà ST đã nạp**
    (`self.model[0].auto_model`). 2,2GB nhân đôi thì 4060 8GB hết chỗ.

65. **Không thêm dependency `FlagEmbedding`.** Nó kéo theo `peft`, `datasets` và
    một vòng đời phát hành riêng, để có đúng một `Linear(1024 → 1)` và một phép
    gộp max. Tự viết là ~40 dòng và kiểm chứng được.

66. **`_forward` là đường sinh dense DUY NHẤT — `_encode` cũng đi qua nó.** Cái
    giá của việc không gọi `model.encode()` là dense có đường code riêng, và hai
    đường sinh dense song song là cách chắc chắn để hai nhánh ablation vô tình đo
    hai thứ khác nhau. Nên có test canh `embed_documents()` khớp
    `SentenceTransformer.encode()`: đo được **max |Δ| = 1,5e-8**, cosine 1,0000.

67. **Phải tự làm lại phép sắp batch theo độ dài mà `encode()` vốn tự làm.**
    Padding tới câu dài nhất trong batch, nên trộn câu 40 token với câu 500 token
    làm hầu hết phép tính đổ vào padding. Sắp **giảm dần** có lợi ích thứ hai:
    câu dài nhất rơi vào batch đầu, nên cấu hình sẽ OOM thì OOM ở giây thứ nhất
    chứ không phải ở phút thứ ba của job 15.000 chunk.

68. ⭐ **Mức tăng +174% nDCG là của MODEL, không phải của việc hết truncation —
    và `TD-11` là thứ chứng minh điều đó.** Lần chạy này đổi ba thứ cùng lúc
    (model, cửa sổ, tính đa ngữ) nên bảng số một mình không tách được. Nhưng đặt
    ba thí nghiệm cạnh nhau thì tách được:

    | | truncation | `hit_rate@5` | |
    |---|---:|---:|---|
    | baseline (PhoBERT 256) | 56,9% | 0,2153 | — |
    | `chunk550` (PhoBERT, hết cắt) | 0,4% | 0,2010 | `p = 0,711` |
    | `bgem3` (BGE-M3, hết cắt) | 0,0% | 0,5455 | `p < 0,001` |

    Hai dòng dưới **cùng** đưa truncation về ~0; một dòng không đổi gì, dòng kia
    +153%. Vai trò thật của cửa sổ 8192 là *cho phép giữ `chunk_size=1000`* — tức
    làm phép đo sạch — không phải tạo ra mức tăng. Kết quả âm của `TD-11` hoá ra
    là thứ có giá trị nhất ở đây: không có nó thì đã gán sai nguyên nhân.

69. **Giữ `chunk_size=1000` làm nhãn BIT-IDENTICAL với baseline.**
    `n_relevant_mean` 1,3828 ở cả hai lần chạy, `span_resolution` giống nhau từng
    trường (`resolved: 209`, `label_changed: 9`). Nên recall@k/nDCG/MAP so được
    trực tiếp và `compare.py` không từ chối metric nào — khác hẳn `TD-11`. Đây là
    lợi ích *đo lường* của việc không đụng vào chunking, không chỉ lợi ích ngữ cảnh.

70. **Gộp trọng số sparse theo token bằng `max`, không `sum`**, và bỏ
    `[CLS]`/`[SEP]`/`[PAD]`/`[UNK]`. Max là định nghĩa của BGE-M3 và nó có nghĩa:
    trọng số trả lời "token này quan trọng thế nào cho đoạn text", không phải "nó
    xuất hiện bao nhiêu lần". Bỏ `[CLS]` là bắt buộc — nó thường là chiều nặng
    nhất, để lại thì mọi cặp text khớp nhau ở đúng chiều đó và điểm sparse gần
    thành hằng số.

71. **Padding phải bỏ theo `attention_mask`, không theo trọng số bằng 0.** Vị trí
    bị mask vẫn đi qua matmul nên vẫn có trọng số dương. Có test riêng cho ca này.

72. **`batch_size` một mình không chặn được VRAM khi cửa sổ là 8192.** 16 câu ×
    8192 token = 131k token, OOM ngay. Thêm trần `max_batch_tokens` tính theo độ
    dài **thật** của batch; chunk ngắn vẫn chạy full batch. Là knob tốc độ nên
    `embedding_max_batch_tokens` nằm **ngoài** `fingerprint`, cùng lý do như
    `device` và `batch_size`.

73. **Chọn provider theo TÊN MODEL, không theo cờ riêng.** `BAAI/bge-m3` → provider
    hybrid. Thêm `use_bge_m3: bool` vào config thì tồn tại được cấu hình
    `model=bge-m3, use_bge_m3=false` vừa hợp lệ về cú pháp vừa vô nghĩa về nội dung.

74. **`None` ≠ `SparseVector` rỗng — bài học `TD-11` lần thứ hai.** `None` =
    "provider không sinh sparse"; rỗng = "đã tính, không token nào dương" (text
    chỉ gồm special token), một kết quả hợp lệ. Gộp lại thì `W2-03` không phân
    biệt được "provider chỉ có dense" với "sparse trả 0 kết quả", và cả hai trông
    giống nhau: im lặng.

75. **Tokenizer BGE-M3 tốn ít hơn 30% token cho text tiếng Anh** (0,172 vs 0,244
    token/ký tự của PhoBERT), và chiều bất đối xứng **đảo**: với PhoBERT thì `en`
    tốn nhiều hơn `vi`, với BGE-M3 thì `en` tốn ít hơn. Đây là gốc của việc `en`
    bị cắt nặng nhất ở baseline (65,3% chunk, mất 19,4% token).

76. **`Ê`-document (`TD-17`) hết truncation nhưng nợ vẫn mở.** Cửa sổ 8192 xoá
    *triệu chứng* (0/36 chunk bị cắt), nhưng văn bản vẫn không có ranh giới từ.
    Triệu chứng mất, nguyên nhân còn — vẫn sửa ở `W3-01`.

77. **Sửa lỗi sổ sách: p95 truy hồi của baseline là 32,8 ms, không phải 39,9 ms.**
    `reports/baseline-retrieval.md` ghi 32,8; CHECKLIST §1 chép sai. Đã sửa cả
    cột **Baseline** lẫn ghi chú dưới bảng.

### Vấn đề đang mở

- **`W2-02` là việc tiếp theo.** Sparse đã sinh được nhưng **chưa dùng**: eval
  của `W2-01` là dense-only. Qdrant **không** thêm named vector vào collection đã
  tồn tại → `rag_bgem3` phải build lại bằng `--recreate` (~405 s).
- ⚠️ **Đừng kể mức tăng của `W2-01` là "sửa được truncation".** Xem quyết định 68.
  Trong CV/interview thì câu đúng là: *"kết quả âm của thí nghiệm trước là thứ cho
  phép quy kết đúng nguyên nhân của thí nghiệm sau"*.
- ⚠️ **Cảnh báo khách quan:** BGE-M3 train trên MIRACL/mC4 và corpus dự án là tài
  liệu World Bank — thể loại model rất có thể đã thấy nhiều. Không phải rò rỉ tập
  test (nhãn sinh từ chunk của chính corpus, chunk giống nhau cả hai lần chạy),
  nhưng mức tăng chưa chắc giữ nguyên trên corpus đóng của doanh nghiệp.
- `TD-13` — golden set vẫn review bằng model, và giờ **càng đáng làm**: mức tăng
  lớn thế này thì con số *tuyệt đối* bắt đầu được dùng để kể chuyện.
- Chi phí phải theo dõi: p95 truy hồi 32,8 → 46,0 ms (+40%, gần như toàn bộ là
  embed truy vấn bằng model lớn hơn). `W2-05` (reranker) sẽ cộng thêm, và `G2` có
  điều kiện p95 < 3500 ms.
- Collection còn trong Qdrant: `rag_baseline`, `rag_chunk550`, `rag_chunk550nb55`,
  `rag_bgem3`. Xoá bằng `make down-clean` rồi build lại từ config nếu cần chỗ.
- Các mục còn lại không đổi: `TD-05`, `TD-08`, `TD-10`, `TD-14`…`TD-17`, `W0-02`.

### Nếu phiên sau bắt đầu từ đây

1. `make lint && make test` (644 unit test) · `make test-gpu` nếu có GPU.
2. Đọc `reports/w2-01-bge-m3.md` §5 (vì sao mức tăng không phải của `TD-11`) và
   §7 (thứ chưa được kiểm chứng đầu-cuối).
3. `W2-02`: thêm named vector `sparse` vào schema Qdrant, dùng
   `SparseVector.as_qdrant()`. Build lại `rag_bgem3` bằng `--recreate`.
4. Mọi so sánh vẫn phải qua `make eval-compare`. Đổi `chunk_size` thì **không**
   dùng recall@k/nDCG/MAP.

---

## Phiên 2026-08-20 (tiếp) · Tuần 2 — `W2-02` Qdrant hybrid schema

**Mục tiêu phiên:** cho sparse của `W2-01` một chỗ chứa — một collection Qdrant
mang cả `dense` và `sparse`, query được độc lập.

**Kết quả:** xong, và sparse **gần như miễn phí** (+2,3% thời gian index, +19%
dung lượng). Phần đáng nhất lại ngoài DoD: `ensure_collection` giờ **kiểm tra**
schema thay vì tin, và trong 4 ca lệch có một ca hỏng im lặng.

### Lệnh để kiểm tra trạng thái hiện tại

```bash
make test                                   # 666 unit (22 ca schema, thuần, ~3s)
make up && make test-integration            # 59 integration (21 ca hybrid)
python scripts/migrate_collection.py --config configs/indexing/bgem3.yaml
```

Build lại (~414 s trên RTX 4060, **bắt buộc** `--recreate`):

```bash
python -m pipeline.indexing.build_index --config configs/indexing/bgem3.yaml \
  --recreate --report plans/reports/index-bgem3.json
make eval-retrieval BUNDLE=bgem3            # phải khớp số cũ TỪNG CHỮ SỐ
```

### Bảng tiến độ phiên

| Việc | Trạng thái | Ghi chú |
|---|---|---|
| `SPARSE_VECTOR_NAME` + schema có sparse | ✅ | Không `Modifier.IDF` — xem quyết định 80 |
| `upsert` ghi cả hai từ một forward pass | ✅ | +8,8 s trên 389 s |
| `retrieve_sparse()` — nhánh độc lập | ✅ | Tách khỏi `retrieve()` có chủ ý |
| `ensure_collection` kiểm tra schema | ✅ | Ngoài DoD, 4 ca lệch có test |
| `scripts/migrate_collection.py` | ✅ | Cố ý **không** migrate tại chỗ |
| `HashingEmbeddingProvider` sinh sparse | ✅ | Mặc định **tắt** — `name` là cache key |
| Xác nhận dense không đổi | ✅ | 0/209 câu đổi điểm |
| `W2-02` | ✅ | `reports/w2-02-qdrant-hybrid.md` |
| `W2-03` sparse retriever | ⬜ việc tiếp theo | Bọc thành `Retriever` + đo có `p`/CI |

**Tổng kết:** 666 unit (+22) + 59 integration (+21) + 11 gpu test xanh · `ruff` +
`mypy --strict` sạch trên 85 file.

### Quyết định kỹ thuật phát sinh trong phiên

*(tiếp số từ 77)*

78. ⭐ **`ensure_collection` phải KIỂM TRA schema, không được thấy tồn tại là trả
    về.** Đây là phần không có trong DoD và là phần đáng nhất của `W2-02`. Trước
    đó, chạy config `bgem3` (provider sinh sparse) lên collection `rag_bgem3` cũ
    (dense-only) sẽ chết ở lần upsert **đầu tiên** — sau khi đã nạp 2,2GB trọng
    số và chunk xong tài liệu đầu — với một thông báo của Qdrant không nói phải
    làm gì. Giờ chết ở giây đầu và in ra `--recreate`.

79. **Trong 4 ca lệch schema có một ca HỎNG IM LẶNG, và nó không được bỏ qua:**
    collection có `sparse` mà provider chỉ sinh dense. Nghĩa là đang eval bằng
    provider dense-only trên index hybrid — con số ra trông bình thường trong khi
    **nửa index không được dùng tới**. Đây là biến thể của đúng cái bẫy `TD-11`:
    hệ thống chạy, số ra, không ai biết là sai. Ba ca kia đều gây lỗi rõ ràng.

80. **KHÔNG dùng `SparseVectorParams(modifier=Modifier.IDF)` cho BGE-M3.** Trọng
    số của model là **đã học** — phần "hạ bậc từ phổ biến" đã nằm trong đó. Chồng
    IDF của Qdrant lên là nhân đôi phép ấy, và nó hỏng theo kiểu tệ nhất: điểm vẫn
    ra số, chỉ là sai. `Modifier.IDF` dành cho nhánh **BM25 thô** ở `W2-03`, nơi
    giá trị đầu vào là tần suất chứ không phải trọng số. Có test integration canh
    `modifier` của `rag_bgem3` là `None`.

81. **`schema_problems()` là hàm THUẦN, tách khỏi client Qdrant.** 12 ca lệch test
    được trong `make test` (~3 giây) chứ không cần server. Hàng rào quan trọng nhất
    của một tầng không nên chỉ test được khi Docker đang chạy.

82. **`upsert` gọi provider MỘT lần, không hai.** Đo được: sparse thêm **8,8 giây
    trên 389** (+2,3%). Gọi `embed_documents()` rồi gọi tiếp một hàm sparse riêng
    sẽ là **+380 giây** — trả gấp đôi tiền forward pass cho đúng một kết quả. Đây
    là chỗ quyết định "một forward pass" của `W2-01` (quyết định 66) trả nợ.

83. **Xác nhận dense không đổi một chữ số sau khi đổi đường ghi.** `_embed_batch`
    → `embed_documents_hybrid` là đường code khác với `embed_documents` của
    `W2-01`. So lại trên 15.814 chunk thật: **15/15 metric không lệch, 0/209 câu
    đổi điểm**. Chỉ so bảng metric tổng thể thì hai lần chạy khác nhau vẫn có thể
    trùng số do bù trừ — nên phải so cả `*-per-query.jsonl`. Hạ tầng thêm ở
    `TD-11` dùng lại được ở đây.

84. **Bỏ `zip(strict=True)` thì phải thay bằng kiểm tra độ dài tường minh.**
    `upsert` giờ index theo offset (`dense[offset]`) nên lệch hàng sẽ **gán
    embedding cho sai chunk** thay vì báo lỗi. Có test integration chạy
    `batch_size=2` trên 3 chunk để đi qua nhiều lô rồi kiểm từng chunk.

85. **Sửa một bug tiềm ẩn: `rank` có lỗ khi bỏ point lỗi.** Bản cũ dùng
    `enumerate(points, start=1)`, nên một point thiếu payload làm dãy rank thành
    `1, 2, 4`. nDCG/MRR đọc `rank` như vị trí thật nên sẽ tính sai âm thầm. Giờ là
    `len(results) + 1`, có test canh dãy liên tục cho **cả hai** nhánh.

86. **`retrieve_sparse()` tách khỏi `retrieve()`, không gộp thành một hàm
    "hybrid".** `W2-04` (RRF) cần hai danh sách xếp hạng *độc lập* để hợp nhất, và
    một hàm trả sẵn hybrid thì không tách được đóng góp của mỗi nhánh — thứ mà
    `RetrievedChunk.dense_score`/`sparse_score` tồn tại để giữ.

87. **Thang điểm hai nhánh KHÔNG so được với nhau.** Dense là cosine (đã chuẩn
    hoá, ∈ [−1, 1]); sparse là **dot product** của trọng số không âm nên không có
    trần. Đo thật trên cùng một truy vấn: dense 0,6682 vs sparse 0,2938. Đây là lý
    do `W2-04` phải hợp nhất theo **thứ hạng**, không theo điểm.

88. **Script migrate cố ý KHÔNG migrate tại chỗ, vì không thể.** Qdrant không cho
    thêm named vector vào collection đã tồn tại. Nên
    `scripts/migrate_collection.py` làm thứ nó làm được: chẩn đoán lệch schema và
    in đúng lệnh phải chạy (mã trả về 0/1/2/3). Đây là phiên bản cho **schema** của
    câu hỏi mà `fingerprint` trả lời cho **nội dung**. Phương án "copy dense cũ rồi
    chỉ tính thêm sparse" đã cân nhắc và bỏ: tính được sparse của BGE-M3 tức là đã
    tính lại dense, nên chẳng tiết kiệm gì.

89. **`HashingEmbeddingProvider` mặc định `sparse=False`, và đó là quyết định.**
    Lần đầu tôi để `True` và hai test W1 đỏ ngay (`test_provider_name_is_
    reproducible` + một test của `W2-01`) — đó là cách phát hiện `name` của provider
    đi vào **cache key của semantic chunker** (`chunking/semantic.py`) và vào MLflow.
    Bật sẵn sẽ vô hiệu cache chunk và làm mọi test W1 đổi nghĩa âm thầm. Có test
    hồi quy canh `embed_documents()` cho kết quả y hệt khi bật/tắt sparse.

90. **`as_qdrant()` trả `TypedDict`, không phải `dict[str, list[int] | list[float]]`.**
    Kiểu union đúng về cấu trúc nhưng làm `models.SparseVector(**payload)` không
    kiểm được kiểu — và chỗ đó là ranh giới giữa code của mình với client bên
    ngoài, đúng chỗ đáng để kiểu chặt. mypy `--strict` bắt được.

91. **Sparse của BGE-M3 là một phép CHỌN, không phải bag-of-words có trọng số.**
    Đo trên 3.000 chunk: **95,9 entry/chunk** (p50 100 · p95 147 · max 195 ·
    **min 3**), mật độ 0,0384% của vocab 250.002. Chunk có p50 **218 token** nên
    ReLU loại khoảng **55%** token.

92. **`min = 3 entry` là cảnh báo cho `W2-04`.** Có chunk chỉ còn 3 token dương,
    tức nhánh sparse gần như không tìm ra được nó bằng bất kỳ truy vấn nào. Mặt bù
    này khớp đặc tính đo được trong test: **không trùng token thì sparse trả rỗng**,
    trong khi dense vẫn đoán được. Hai nhánh hỏng theo hai kiểu khác nhau — đó là
    lý do RRF đáng làm chứ không phải chỉ để có thêm một dòng trong CV.

93. **Filter metadata phải áp cho CẢ HAI nhánh ở tầng Qdrant.** Có test riêng cho
    nhánh sparse: thiếu nó thì `W2-06` (cô lập tenant) có một lỗ đúng bằng nhánh
    sparse, và lỗ đó không lộ ra ở bất kỳ test dense nào.

94. **Thêm sparse index làm p50 truy hồi dense tăng 23,7 → 31,5 ms (+33%), tái
    lập 3 lần — nhưng p95 gần như không đổi** (46,0 → 46,6 ms). Con số này cho biết
    cấu trúc của độ trễ: p95 bị chi phối bởi forward pass embed truy vấn của BGE-M3
    (biến động lớn), p50 phản ánh phép tìm trong Qdrant. ⚠️ **Chưa tách được** chi
    phí của sparse index khỏi trạng thái segment sau khi vừa build lại (8 segment,
    Qdrant tối ưu ở background). Với ngưỡng `G2` p95 < 3500 ms thì 46,6 ms còn rất
    nhiều chỗ nên chưa đáng đào — nhưng phải đo lại ở `W2-04`.

### Vấn đề đang mở

- **`W2-03` là việc tiếp theo.** `retrieve_sparse()` chạy được nhưng **chưa** là
  `Retriever`, nên eval harness chưa đo được nó. Đến `W2-03` mới có **số** cho câu
  "sparse đóng góp gì", và phải kèm `p`/CI qua `make eval-compare`.
- ⚠️ Nhánh BM25 **thô** của `W2-03` mới cần `Modifier.IDF`; nhánh BGE-M3 thì
  không (quyết định 80). Hai nhánh sparse khác nhau về bản chất đầu vào.
- ⚠️ `W2-04` hợp nhất theo **thứ hạng**, không theo điểm (quyết định 87).
- Collection trong Qdrant: `rag_baseline` (768-d), `rag_chunk550`,
  `rag_chunk550nb55`, `rag_bgem3` (1024-d + sparse). Chỉ `rag_bgem3` là hybrid.
- `TD-13` (golden set review bằng model) vẫn mở và vẫn là nợ đáng nhất.
- Các mục còn lại không đổi: `TD-05`, `TD-08`, `TD-10`, `TD-14`…`TD-17`, `W0-02`.

### Nếu phiên sau bắt đầu từ đây

1. `make lint && make test` · `make up && make test-integration`.
2. Đọc `reports/w2-02-qdrant-hybrid.md` §3 (sparse trên dữ liệu thật, và vì sao
   `min = 3` quan trọng) và §7 (thứ chưa làm).
3. `W2-03`: bọc `retrieve_sparse()` thành một `Retriever` để eval harness chạy
   được, rồi đo trên golden set. DoD là truy vấn từ khoá lạ mà dense miss thì
   sparse hit — phải có ca exact-ID lookup.
4. Dùng `HashingEmbeddingProvider(..., sparse=True)` cho test, không cần GPU.

---
