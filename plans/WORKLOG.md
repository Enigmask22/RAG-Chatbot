# WORKLOG — nhật ký phiên làm việc

> Mục đích: nếu phiên Claude Code bị ngắt giữa chừng (hết quota 5h, mất mạng, đóng máy),
> file này cho biết **đang làm dở tới đâu** và **lệnh nào để tiếp tục**.
> Trạng thái chính thức của từng task vẫn nằm ở [`CHECKLIST.md`](CHECKLIST.md).

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
- [ ] `W1-11` — **việc tiếp theo, cần BẠN đọc tay** (~6-8h). Thứ tự theo rủi ro:
      16 câu `needs_close_review` → 40 câu `unanswerable` → 34 câu `multi_hop`.
- [ ] `W1-09` (DVC) và `W1-13` (đo baseline) — sau `W1-11`.
- [ ] Corpus nguồn (b) pháp luật và (c) HOSE — cần bạn chọn tay, khai báo qua `seed_list`.
- [ ] Gate `G1` đạt 2/4 mục; hai mục còn lại chặn bởi golden set (`W1-11`) và baseline (`W1-13`).

### Nếu phiên sau bắt đầu từ đây

Đọc theo thứ tự: mục "Quyết định kỹ thuật" ở trên → `reports/w1-08-build-index.md`
→ `reports/w1-foundation.md` →
`CHECKLIST.md` §1 Dashboard và §10 Đang bị chặn. Chạy `make lint && make test` để xác
nhận repo vẫn xanh trước khi làm tiếp.

---
