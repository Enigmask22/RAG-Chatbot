# W1 — Nền móng: bằng chứng nghiệm thu

> Ngày chạy: 2026-08-17 · Máy: Windows 11, Python 3.13.11, Docker 29.4.3
> Phạm vi: `W1-01` … `W1-07` và `W1-12`. Các task còn lại của tuần 1 chờ corpus/API key.

**Commit:** `f5ec22b` (nền móng W1) · `c253fa5` (bộ tải corpus W0-03) — nhánh `feat/w1-foundation`

## Cách tái lập

```bash
uv sync --extra dev --extra qdrant
make lint                 # ruff check + ruff format --check + mypy strict
make test                 # 144 unit test, không cần Docker
make up                   # Qdrant + Postgres + Redis
make test-integration     # 18 integration test
```

## Kết quả

| Kiểm tra | Lệnh | Kết quả |
|---|---|---|
| Lint | `ruff check .` | All checks passed |
| Format | `ruff format --check .` | 45 file đã đúng format |
| Type | `mypy` (strict) | Success: no issues found in 39 source files |
| Unit test | `pytest -m "not integration and not gpu"` | **144 passed** · 2.99s |
| Integration test | `pytest -m integration` | **18 passed** · 30.3s |
| Coverage `rag_core` | `pytest --cov` (chỉ unit) | **81%** (ngưỡng `G1` là ≥ 70%) |

Coverage theo module (chạy unit test, chưa tính integration):

| Module | Cover | Ghi chú |
|---|---:|---|
| `schemas.py` | 99% | |
| `settings.py` | 100% | |
| `chunking/` (5 file) | 93–100% | |
| `embedding/hashing.py` | 100% | |
| `embedding/huggingface.py` | **0%** | Chỉ chạy khi có `torch` — nợ kỹ thuật, xem bên dưới |
| `retrieval/qdrant_store.py` | 23% | Phần còn lại phủ bởi integration test |

Hạ tầng lúc chạy integration test:

```
NAME                      IMAGE                   STATUS
rag-platform-postgres-1   postgres:16-alpine      Up (healthy)   127.0.0.1:5432->5432/tcp
rag-platform-qdrant-1     qdrant/qdrant:v1.15.1   Up (healthy)   127.0.0.1:6333-6334->6333-6334/tcp
rag-platform-redis-1      redis:7-alpine          Up (healthy)   127.0.0.1:6379->6379/tcp
```

## Hai lỗi hạ tầng tìm ra khi chạy thật

Cả hai đều không lộ ra ở unit test — đây là lý do `W1-05` và `W1-07` có integration
test riêng thay vì mock Qdrant.

**1. `QDRANT__SERVICE__API_KEY` rỗng vẫn bật xác thực → mọi request 401.**
Compose viết `QDRANT__SERVICE__API_KEY: ${QDRANT_API_KEY:-}` thì biến **luôn tồn tại**
với giá trị rỗng, và Qdrant hiểu là "có đặt key". Sửa bằng cách dùng dạng danh sách
không có dấu `=` (`- QDRANT__SERVICE__API_KEY`), khi đó biến chỉ được truyền xuống
container nếu máy host thực sự có đặt.

**2. `localhost` trên Windows làm mỗi request Qdrant chậm đúng 2 giây.**
`localhost` resolve sang `::1` trước; Docker chỉ bind `127.0.0.1` nên kết nối IPv6 thất
bại rồi mới fallback sang IPv4. Bộ integration test chạy mất **~5 phút**; sau khi đổi
toàn bộ default sang `127.0.0.1` còn **30 giây**.

Điểm đáng lưu ý cho về sau: nếu không phát hiện bây giờ thì con số p95 latency đo ở
`W5`/`W6` sẽ cộng thêm 2000 ms cố định cho mỗi lần gọi Qdrant, và sẽ rất khó truy vì
mọi thứ khác trông vẫn đúng.

Ngoài ra `qdrant-client` được ghim `>=1.15,<1.16` cho khớp major/minor với image server;
client 1.19 nói chuyện với server 1.12 vẫn chạy nhưng cảnh báo không tương thích.

## Sai lệch có chủ ý so với `enhanced_chunking.py`

Ghi ở đây vì chúng **ảnh hưởng tới con số baseline `W1-13`**. Chi tiết lý do nằm ở
docstring `packages/rag_core/chunking/base.py`.

1. `config_hash` không làm tròn tham số nữa (bản cũ làm tròn `chunk_size` về bội 100,
   khiến `1000` và `1049` dùng chung cache entry).
2. Hậu xử lý áp theo từng tài liệu, không trên danh sách gộp.
3. Ép kích thước trước, thêm ngữ cảnh hàng xóm sau.
4. `neighbor_context_chars` mặc định **tắt** (bản cũ luôn bật ở 100 ký tự).

⚠️ **Config baseline của `W1-13` phải đặt `neighbor_context_chars=100`**, nếu không thì
cái được đo không phải hệ thống hiện tại.

## Nợ kỹ thuật ghi nhận

| Nợ | Lý do chấp nhận | Kế hoạch trả |
|---|---|---|
| `HuggingFaceEmbeddingProvider` coverage 0% | Cần `torch` (~2.5GB) mà unit test cố ý chạy không có GPU stack | Test đánh dấu `@pytest.mark.gpu` khi làm `W0-06` (đo VRAM) và `W2-01` (BGE-M3) |
| ~~`retrieval_eval.py` CLI chưa nối retriever thật~~ | Cần index thật, mà index cần corpus (`W0-03`) | ✅ đã trả ở `W1-08` — xem [`w1-08-build-index.md`](w1-08-build-index.md) |
| ~~`pipeline/indexing/` còn rỗng~~ | Cùng lý do trên | ✅ đã trả ở `W1-08` |
