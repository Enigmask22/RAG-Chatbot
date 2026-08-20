# `legacy/` — bản POC gốc

Đây là toàn bộ hệ thống trước khi nâng cấp: một ứng dụng Streamlit đơn tệp dùng
LangChain + Chroma + Vicuna-7B, chạy được trên Colab T4.

## Vì sao còn giữ, thay vì xoá

**Nó là hệ thống đang được đo.** `W1-13` phải tái lập đúng hành vi của bản này để
mọi con số "cải thiện +X%" của W2/W3 có mốc so sánh thật. Xoá đi thì cái mốc đó
thành lời kể lại, không kiểm chứng được.

Ngoài ra `plans/reports/w1-08-build-index.md` ghi bốn chỗ bản mới **cố ý làm
khác** bản này. Đọc ghi chú đó mà không đọc được code gốc thì không xác minh được.

## Tương ứng với code mới

| File ở đây | Đã thay bằng | Ghi chú |
|---|---|---|
| `enhanced_chunking.py` | `packages/rag_core/chunking/` | Viết lại thuần Python, bỏ `langchain-experimental`. Bốn sai lệch có chủ ý — xem `reports/w1-08-build-index.md` |
| `enhanced_chunking.py` (`ChunkingCache`) | `packages/rag_core/chunking/cache.py` | `pickle` → SQLite + JSON. `pickle.load` thực thi mã tuỳ ý khi giải mã |
| `app.py` (Chroma) | `packages/rag_core/retrieval/qdrant_store.py` | Named vector, point ID xác định |
| `app.py` (UI + chat state) | `serving/` | Chưa làm, `W4` |
| `requirements.txt` | `pyproject.toml` | uv + extras, phụ thuộc nặng tách riêng |

## Chạy thử bản cũ

```bash
pip install -r legacy/requirements.txt
streamlit run legacy/app.py
```

Streamlit thêm thư mục của script vào `sys.path`, nên `from enhanced_chunking
import ...` trong `app.py` vẫn giải quyết được sau khi dời vào đây.

⚠️ Bản này cần **~7GB VRAM** cho Vicuna-7B 4-bit. RTX 4060 8GB chạy được nhưng
sát mép; đó chính là lý do kiến trúc mới đẩy generator sang API.

## Khi nào xoá

`W6-05`, sau khi `serving/` đã thay thế được đầu-cuối **và** `reports/baseline.md`
đã đóng băng số của bản cũ. Trước đó thì đây là chứng cứ, không phải rác.

Code trong này **cố ý không** chịu `ruff` và `mypy` (xem `extend-exclude` trong
`pyproject.toml`) — format lại nó chỉ tạo diff nhiễu trên thứ sắp bị xoá.
