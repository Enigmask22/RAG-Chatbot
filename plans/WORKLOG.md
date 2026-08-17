# WORKLOG — nhật ký phiên làm việc

> Mục đích: nếu phiên Claude Code bị ngắt giữa chừng (hết quota 5h, mất mạng, đóng máy),
> file này cho biết **đang làm dở tới đâu** và **lệnh nào để tiếp tục**.
> Trạng thái chính thức của từng task vẫn nằm ở [`CHECKLIST.md`](CHECKLIST.md).

---

## Phiên 2026-08-17 · Tuần 1 — nền móng

**Mục tiêu phiên:** hoàn thành 8 task W1 không bị chặn — `W1-01` … `W1-07` và `W1-12`.

### Lệnh để kiểm tra trạng thái hiện tại

```bash
cd D:/studioproj/project_1.2_chatbot_rag
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

### Việc còn dở / cần làm tiếp

- [ ] **Commit.** Toàn bộ thay đổi của phiên này **chưa được commit** — chỉ nằm trên đĩa.
- [ ] `W1-08` — **giờ chạy được**: đã có 60 tài liệu trong `data/corpus/`.
- [ ] `W1-09`, `W1-11`, `W1-13` — nối tiếp sau `W1-08`.
- [ ] Corpus nguồn (b) pháp luật và (c) HOSE — cần bạn chọn tay, khai báo qua `seed_list`.
- [ ] `W1-10` — `DEEPSEEK_API_KEY` đã có; chờ `W1-08` để có chunk thật.
- [ ] Gate `G1` mới đạt 1/4 mục; ba mục còn lại đều chặn bởi corpus.

### Nếu phiên sau bắt đầu từ đây

Đọc theo thứ tự: mục "Quyết định kỹ thuật" ở trên → `reports/w1-foundation.md` →
`CHECKLIST.md` §1 Dashboard và §10 Đang bị chặn. Chạy `make lint && make test` để xác
nhận repo vẫn xanh trước khi làm tiếp.

---
