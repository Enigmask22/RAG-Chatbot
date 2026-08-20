# Đổi tên workspace → `RAG-Chatbot`

> 2026-08-20 · repo GitHub đã đổi thành `Enigmask22/RAG-Chatbot`

## Việc đã làm sẵn

| Việc | Trạng thái |
|---|---|
| `git remote set-url origin https://github.com/Enigmask22/RAG-Chatbot.git` | ✅ xong, đã kiểm bằng `git ls-remote` |
| `cd` trong `README.md` và `plans/WORKLOG.md` | ✅ trỏ sang `RAG-Chatbot` |
| Script đổi tên thư mục | `D:\studioproj\rename-rag-workspace.ps1` |

## Việc phải tự chạy, và vì sao

**Không đổi tên được từ trong phiên đang chạy.** Thư mục đó là CWD của tiến trình
Claude Code và của shell; Windows từ chối đổi tên thư mục đang bị giữ. Vì vậy:

```powershell
# Đóng Claude Code + VS Code + mọi terminal đang ở trong thư mục đó, rồi:
powershell -ExecutionPolicy Bypass -File D:\studioproj\rename-rag-workspace.ps1
```

Script kiểm tra trước khi làm gì: thư mục đích chưa tồn tại, và không còn tiến
trình nào chạy từ bên trong thư mục cũ.

## Rà soát xung đột

### Vỡ — script xử lý

**`.venv` bám đường dẫn tuyệt đối.**
`.venv/Lib/site-packages/_editable_impl_rag_platform.pth` ghi thẳng
`D:\studioproj\project_1.2_chatbot_rag\packages`, và mọi `.exe` trong
`.venv/Scripts/` nhúng đường dẫn interpreter cũ. Đổi tên xong thì `uv run` vẫn
chạy nhưng `import rag_core` trỏ vào chỗ không còn tồn tại. Script xoá `.venv`
rồi `uv sync` lại. (`home = D:\miniconda3` là Python nền, không ảnh hưởng.)

**Thư mục Claude Code suy ra từ đường dẫn.**
`~/.claude/projects/D--studioproj-project-1-2-chatbot-rag/` chứa `memory/`
(MEMORY.md + 3 ghi chú) và 10+ transcript phiên. Claude Code sinh tên này **từ
đường dẫn workspace**, nên sau khi đổi tên nó sẽ tạo thư mục mới rỗng và toàn bộ
bộ nhớ dài hạn coi như mất. Script chuyển sang `D--studioproj-RAG-Chatbot`.

⚠️ Tên đó là **suy đoán** theo quy tắc `D:\ → D--`, `\ → -`. Nếu sau khi mở lại mà
Claude Code không nhớ gì về dự án, kiểm `ls ~/.claude/projects/` xem nó tạo tên
nào rồi chuyển nội dung sang.

**Cache của công cụ.** `.mypy_cache`, `.ruff_cache`, `.pytest_cache`, `.coverage`
đều lưu đường dẫn tuyệt đối. Script xoá; chúng tự sinh lại.

### Không vỡ — kiểm rồi

**Docker: containers và volume giữ nguyên.** `infra/docker-compose.yml` khai
`name: rag-platform` **tường minh**. Nếu để compose tự suy tên project từ tên
thư mục thì đổi tên thư mục = một project mới = **volume Qdrant mới rỗng**, và
15.814 chunk của index baseline biến mất mà không có lỗi nào. Dòng `name:` đó
đặt từ `W1-05` vì lý do khác, nhưng hôm nay nó cứu đúng việc này.

**`.cache/` giữ lại có chủ đích.** `chunks.sqlite3` khoá theo
`(content_hash, config_hash, chunker_name)`; `index_state/baseline.json` chỉ có
fingerprint và content hash — đã đọc kiểm, không có đường dẫn nào. Xoá đi là
build lại index 200 giây vô ích.

**`.env` đi theo thư mục.** Không cần làm gì.

**`uv` cache `D:\uv-cache`** vẫn cùng ổ NTFS với `.venv` mới → hardlink vẫn chạy.

**Remote cũ vẫn hoạt động nhờ GitHub redirect**, nhưng đã đổi tường minh: redirect
sẽ đứt nếu sau này có ai tạo repo trùng tên cũ.

### Cố ý không đổi

`pyproject.toml` giữ `name = "rag-platform"`. Đó là tên **distribution** Python,
không liên quan tên repo hay tên thư mục. Đổi nó sẽ đổi `dist-info`, prompt của
venv, và tên wheel — không được gì, mà `RAG-Chatbot` còn không phải tên
distribution hợp quy ước (PEP 508 chuẩn hoá về `rag-chatbot`).

## Kiểm tra sau khi đổi

```bash
cd D:\studioproj\RAG-Chatbot
git remote -v                              # .../RAG-Chatbot.git
uv run pytest -m "not integration and not gpu"   # 317 passed
make up && make test-integration           # 33 passed — volume Qdrant còn nguyên
uv run python -m pipeline.indexing.build_index --config configs/indexing/baseline.yaml
#   -> phải báo "index 0 · bỏ qua 60": index cũ còn đó, không build lại
```

Dòng cuối là phép thử thật sự: nó chứng minh cả Docker volume lẫn `.cache/` đều
qua được việc đổi tên.
