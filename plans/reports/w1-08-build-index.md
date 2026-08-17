# W1-08 — Build index: bằng chứng nghiệm thu

> Ngày chạy: 2026-08-17 · Windows 11 · Python 3.13.11 · RTX 4060 Laptop 8GB · Qdrant v1.15.1
> Phạm vi: `W1-08`. Kèm theo đó gỡ nợ `TD-02` (CLI eval chưa nối retriever thật) và `TD-03`.

## Cách tái lập

```bash
uv sync --extra dev --extra qdrant --extra pipeline --extra embeddings
make up                       # Qdrant + Postgres + Redis
make index                    # BUNDLE=baseline
make index                    # chạy lần hai: phải không ghi thêm gì
```

## Kết quả build baseline

```
Index `baseline` → collection `rag_baseline`
  fingerprint      72c87744d258ed2c
  embedding        bkai-foundation-models/vietnamese-bi-encoder (cuda, 768 chiều)
  chunker          hybrid->fixed:bf526f417134
  tài liệu         60 (index 60 · bỏ qua 0 · gỡ 0)
  chunk            ghi 15814 · xoá thừa 0 · tổng trong collection 15814
  độ dài chunk     trung bình 1122 · p50 1137 · p95 1193 · max 1251 ký tự
  ký tự            vào 14,284,300 → embed 17,745,511 (hệ số 1.24x)
  thời gian        nạp 0.1s · chunk 1.8s · embed+ghi 159.3s · tổng 202.3s
  thông lượng      78.2 chunk/giây
```

Chạy lần hai (không cờ gì): `tài liệu 60 (index 0 · bỏ qua 60)`, `ghi 0`,
`tổng trong collection 15814`. Đúng yêu cầu DoD.

Báo cáo máy đọc được: [`index-baseline.json`](index-baseline.json).

| Kiểm tra | Lệnh | Kết quả |
|---|---|---|
| Lint | `ruff check .` + `ruff format --check .` | All checks passed · 50 file |
| Type | `mypy` (strict) | Success: no issues found in **49** source files |
| Unit test | `pytest -m "not integration and not gpu"` | **217 passed** · 2.1s |
| Integration test | `pytest -m integration` | **33 passed** · 85s |
| Coverage `rag_core` | `pytest --cov` (chỉ unit) | **80%** (ngưỡng `G1` ≥ 70%) |

Test mới: `test_index_config.py` (**28 case**), `test_corpus_loader.py` (**16 case**),
`test_build_index.py` (**15 case**, integration), cộng 2 case warm-up ở
`test_retrieval_eval.py`.

## Ba tầng idempotent, và vì sao cần cả ba

`W1-07` đã có point ID xác định (UUIDv5 của `chunk_id`). Riêng nó **không đủ**:

| Tầng | Cơ chế | Bắt được ca nào | Test |
|---|---|---|---|
| 1 | Point ID xác định | Chạy lại y nguyên → không sinh bản trùng | `TestIdempotent` |
| 2 | Nhớ số chunk cũ, xoá phần đuôi thừa | Tài liệu **ngắn lại** → chunk `::00030` của bản cũ vẫn nằm trong index và retriever vẫn trả về | `TestDocumentChanges` |
| 3 | So `fingerprint` trong state | Chạy config A rồi config B vào cùng collection → index trộn hai cấu hình, `count()` vẫn trông hợp lý | `TestConfigGuard` |

Tầng 2 là tầng dễ bỏ sót nhất: point mồ côi không trùng lặp với gì cả, nó chỉ
đơn giản là trỏ tới văn bản không còn tồn tại. Không có triệu chứng nào ngoài
việc recall tụt.

Thứ tự cố ý là **upsert trước, xoá sau**. Chết giữa chừng thì tài liệu thừa vài
chunk cũ (chạy lại dọn được); xoá trước rồi chết là mất hẳn tài liệu khỏi index.

State file `.cache/index_state/{name}.json` **không** phải nguồn sự thật. Trước
khi tin nó, script đối chiếu tổng số chunk với `collection.count()`; lệch thì bỏ
state và index lại toàn bộ. Ca thật: chạy `make down-clean` mà quên xoá `.cache/`.

## Ba lỗi tìm ra khi chạy thật

**1. `HybridChunker` luôn chọn nhánh semantic khi có cache.**
Cache hoạt động theo từng tài liệu nên `CachedChunker` gọi `inner.chunk([doc])`
60 lần. `HybridChunker` quyết định nhánh theo `len(documents)`, thấy `n=1` mỗi
lần, mà ngưỡng là `hybrid_max_docs_for_semantic=5` → **luôn semantic**. Bản POC
truyền cả corpus một lượt nên với 60 tài liệu nó chạy fixed. Nếu không phát hiện
thì baseline `W1-13` đo một chiến lược chunking khác hẳn hệ thống hiện tại.

Sửa bằng `Chunker.prepare(n_documents)`: người gọi khai báo tổng số tài liệu
trước, khai báo tường minh luôn thắng suy đoán theo lô. Log xác nhận
`chunker hybrid->fixed`.

**2. `HybridChunker.name` không phân biệt được nhánh đã chọn.**
Tên mặc định chỉ có `config_hash`, mà tên chunker là một phần khoá cache. Hai
lần chạy hybrid — một rơi vào semantic, một vào fixed — dùng chung khoá và đọc
lại kết quả của nhau. Nhánh semantic còn phụ thuộc model embedding nữa. Giờ tên
là `hybrid->{tên nhánh thật}`, và tên nhánh semantic đã kèm tên model.

**3. p95 độ trễ truy hồi là con số của việc nạp model, không phải của truy hồi.**
Đo trên index baseline: p95 = **15.219 ms**, p50 = 31 ms. Truy vấn đầu tiên gánh
cả việc nạp `sentence-transformers` (lazy). Thêm một lượt warm-up bỏ đi trước
khi bấm giờ: p95 còn **98 ms**, mean 40 ms.

Đây là con số sẽ làm ngưỡng cho gate hiệu năng ở W5/W6. Để nguyên thì gate đo
thời gian khởi động.

## Con số đáng chú ý: hệ số 1.24x

`neighbor_context_chars=100` của bản POC làm phần text đem embed phình từ
14,3 triệu lên 17,7 triệu ký tự — **thêm 24%** token cho mỗi lần embed, và ở
serving là thêm 24% context nhét vào prompt.

Con số này chưa nói kỹ thuật đó tốt hay dở; nó nói **giá phải trả**. Việc so
recall giữa bật và tắt là của `W2`. Baseline giữ nguyên 100 để đo đúng hệ thống
đang có.

## Đường ống đã chạy được đầu-cuối

`corpus → chunk → embed → Qdrant → eval → báo cáo` đã thông. Kiểm chứng bằng một
golden set giả sinh từ chính chunk trong index (13 câu, **không** commit — golden
set thật là `W1-10`/`W1-11`):

```
recall@5 0.9167 · mrr 0.9167 · ndcg@10 0.9167 · p50 32.6ms · p95 97.7ms
```

Số này **không phải baseline** — câu hỏi chính là text của chunk nên gần như chắc
chắn tìm lại được chính nó. Nó chỉ chứng minh CLI chạy được:

```bash
python -m pipeline.eval.retrieval_eval \
    --index-config configs/indexing/baseline.yaml --golden <file>
```

Eval và index dùng **chung một file config** thay vì khai báo model/collection
hai lần. Hai bên khai báo riêng thì sớm muộn cũng lệch, và eval sẽ đo một index
được build bằng model khác mà không có gì báo lỗi.

## Quyết định về `fingerprint`

`IndexConfig.fingerprint` băm đúng những trường quyết định **nội dung** vector:
chunking config, tên model, `normalize`, prefix, bộ lọc corpus.

Cố ý **không** gồm `device`, `batch_size`, `upsert_batch_size`, đường dẫn cache.
Chạy trên GPU thuê hay trên laptop phải ra cùng một index về mặt logic — nếu
`device` vào fingerprint thì mỗi lần đổi máy là một lần build lại vài giờ GPU,
đúng thứ kiến trúc hai plane sinh ra để tránh. Có test canh cả hai chiều.

## torch CUDA trên Windows

Wheel `torch` mặc định trên PyPI cho Windows là bản **CPU-only**. Cài từ đó thì
mọi thứ vẫn chạy, chỉ chậm hơn nhiều và `torch.cuda.is_available()` trả `False`
mà không báo lỗi gì. Đã ghim index `download.pytorch.org/whl/cu126` trong
`[tool.uv.sources]`; xác nhận `torch 2.13.0+cu126`, `cuda_available True`,
build baseline chạy trên `cuda`.

## Nợ kỹ thuật ghi nhận

| Nợ | Lý do chấp nhận | Kế hoạch trả |
|---|---|---|
| Toàn bộ `Document` nằm trong RAM khi build | 60 tài liệu = 14 MB, không đáng tối ưu sớm | `W3-07` (re-index tăng dần) sẽ đọc theo luồng |
| `retrieval_eval` chưa có gate so với champion | Cần baseline đã (`W1-13`) rồi mới có champion | `W4-05` |
| Chưa đo lại độ ổn định giữa hai lần build | Cần golden set thật mới có số để so | `W1-13` (DoD: chạy lại 2 lần, sai số < 1%) |
