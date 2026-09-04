# `W4-11` — Prompt registry: YAML có version + hash, loader

> 2026-09-04 · Serving Plane (loader ở `rag_core` — Pipeline Plane dùng chung) ·
> DoD: đổi prompt = tăng version; runtime log rõ prompt version đang dùng ·
> Test: `tests/unit/test_prompt_registry.py` · Evidence: log khởi động (§4).

## 0. Câu hỏi thật của hạng mục

Trước hạng mục này dự án có **năm** prompt, không phải ba như checklist ngầm
định: `SYSTEM_PROMPT` + `NO_RETRIEVAL_SYSTEM_PROMPT` (chat.py),
`REWRITE_SYSTEM_PROMPT` (understanding.py), và `CONTEXT_SYSTEM_PROMPT` +
`BATCH_SYSTEM_PROMPT` (contextual chunking, `W3-03`). Ba cái đầu là hằng số
không đánh số — đổi một chữ là đổi hệ thống đang được đo mà không có gì trong
log nói ra. Hai cái sau **không migrate**, có chủ đích: nội dung system thật
của chúng đã được băm vào fingerprint cache của `W3-03`, và $5,47 ngữ cảnh đã
sinh gắn với fingerprint ấy — đưa vào registry không mua được gì hôm nay ngoài
rủi ro lệch fingerprint.

Và câu hỏi thật không phải "YAML để ở đâu" mà là: **"đổi prompt = tăng version"
được thi hành bằng gì?** Một quy ước "nhớ tăng version khi sửa" không chữa được
kiểu hỏng mà nó nhắm tới, vì kiểu hỏng ấy chính là *quên*.

## 1. Cơ chế: quên thì không nạp được

Mỗi prompt một file YAML trong `packages/rag_core/generation/prompts/`
(đi theo package — Docker của `W4-13` copy `packages/` là có, không cần thêm
volume): `id` (= tên file), `version`, `sha256` (của **template**, không của
file — đổi `description` không được vỡ hash, cùng lý lẽ với checksum bundle
`W4-01`), `history` (các cặp version–hash cũ), `template` (block scalar `|-`,
diff đọc được như văn bản).

Ba phép kiểm ở loader, mỗi phép chặn một cách file nói dối:

1. sha256 tính lại từ template ≠ sha256 khai → **từ chối nạp**, thông điệp chỉ
   đúng đường sửa (`scripts/prompt_stamp.py`).
2. version phải **lớn hơn mọi** version trong history — một số dùng lại là hai
   nội dung khác nhau mang cùng một tên.
3. `extra="forbid"`: gõ nhầm `verison: 2` là lỗi nạp, không phải một trường bị
   lặng lẽ bỏ qua trong khi người sửa tin rằng mình đã tăng version.

Đường sửa hợp lệ duy nhất là `stamp()`: sửa **mỗi** template rồi chạy tool — nó
tự đẩy cặp (version, sha256) cũ vào history, tăng version, ghi hash mới. Con
người không gõ tay version hay hash ở bước nào; loader từ chối mọi tổ hợp mà
stamp không sinh ra được.

⭐ Fail-fast **có chủ đích, ngược với bundle**: prompt hỏng chết ngay từ import
`chat.py`, còn bundle nạp lỗi thì tiến trình vẫn lên để `/ready` trả lời vì
sao. Khác nhau vì bản chất khác nhau: bundle là trạng thái runtime chữa được
bằng `POST /admin/bundle/reload`; prompt là package data đóng trong image — một
file hỏng là một bản build hỏng, và thứ duy nhất một tiến trình sống làm được
với nó là phục vụ traffic bằng prompt sai.

## 2. Nội dung migrate giữ nguyên BYTE — và cầu sang bundle

Ba file YAML sinh bằng script từ đúng hằng số cũ, round-trip so bằng `==` trước
khi commit: các phép đo `W4-07` (chỉ thị ngôn ngữ 8/8→0/8) và `W4-09` (block
CITATIONS) gắn với đúng nội dung này, lệch một byte là chúng thôi so được. Cả
ba khởi đầu ở `v1` — lịch sử trước registry nằm trong git, không bịa lại.

`Prompt.component()` trả về `PromptComponent(id, version, hash="sha256:…")` —
đúng cái ô mà schema bundle chừa sẵn từ `W4-01` thay vì nhét `"todo"`. Khi `W5`
dựng generation eval, bundle sẽ khai được prompt bằng một lời gọi.

## 3. Hệ quả kéo theo: cache và meta

* ⭐ **Namespace semantic cache phải mang version prompt.** Registry vừa biến
  prompt thành biến số có version thì `W4-10` lộ một lỗ: khoá cache chỉ có
  `{tenant}:{bundle_version}`, nên đổi prompt mà không đổi bundle sẽ phát lại
  câu trả lời của một hệ thống đã không còn tồn tại. `cache_namespace()` giờ
  trả `"{bundle}+chat-system@v1"` — invalidate khi đổi prompt là thuộc tính của
  khoá, cùng cơ chế với invalidate khi đổi bundle.
* Khung `meta` thêm `"prompt": "chat-system@v1"` (nhánh no-retrieval khai
  prompt của nó; CLARIFY khai `None` — không model nào được gọi). Một con số
  eval sau này truy được nó sinh dưới prompt nào mà không phải đoán từ ngày giờ.
* `LANGUAGE_DIRECTIVE`/`CLARIFY_TEXT` cố ý đứng ngoài registry: cái đầu là
  mảnh ghép theo lượt (nhét vào prompt có version là chẻ hash theo ngôn ngữ),
  cái sau là text trả cho người dùng do mã chọn, không phải prompt cho model.

## 4. Evidence DoD "runtime log rõ prompt version"

`create_app()` thật, log JSON (2026-09-04T09:51:11Z):

```json
{"level": "INFO", "logger": "serving.api.app", "msg": "prompt registry: chat-no-retrieval@v1 (sha256 917c10331d09…)"}
{"level": "INFO", "logger": "serving.api.app", "msg": "prompt registry: chat-system@v1 (sha256 aa26543c1c5e…)"}
{"level": "INFO", "logger": "serving.api.app", "msg": "prompt registry: query-rewrite@v1 (sha256 8a2bc7b51083…)"}
```

## 5. Tiêm lỗi: 6 phép, 1 sống sót — và nó chỉ ra một test tự chiếu

M1 (loader bỏ kiểm hash) · M3 (stamp quên tăng version) · M4 (stamp quên
history) · M5 (`cache_namespace` bỏ prompt) · M6 (`prompt_spec` bỏ rẽ nhánh):
**đỏ đúng test chủ đích**.

⭐ **M2 (`sha256_of` strip() nội dung trước khi băm) XANH** — kể cả với test
ghim hash literal đã thêm *đề phòng chính nó*. Giải thích: template thật không
có whitespace ở biên (YAML `|-` còn chủ động cắt newline cuối) nên strip() là
no-op trên mọi dữ liệu hiện có, và mọi test còn lại so hash **qua chính hàm bị
tiêm** — tự nhất quán, không ai thấy. Hợp đồng "băm đúng byte, không chuẩn hoá"
phải ghim trực tiếp: `sha256_of("a") != sha256_of(" a ")`. Thêm test, tiêm
lại: đỏ → 6/6. Cùng khuôn S6 của `W4-10`: phép tiêm sống sót không phải để
xoá đi, nó là một câu hỏi có đáp án.

## 6. Đính chính sổ sách phiên trước

Câu "ruff/mypy sạch" của phiên `W4-10` **sai một phần**: 6 lỗi F811 + 1 B007
(test files của `W4-10`) và 3 lỗi mypy (2 file test `W4-09`/`W4-10`) không nằm
trong scope lệnh kiểm phiên ấy. Đã sửa hết trong phiên này (fixture chuyển sang
`tests/integration/conftest.py`; cast cho kiểu union của redis-py sync). Từ giờ
lệnh kiểm là `uv run ruff check .` + `uv run mypy` **không tham số** — scope
hẹp là đúng cái đã che hai lần này.

## 7. Test

Unit **+22** (`test_prompt_registry.py`: nạp/round-trip, 7 kiểu từ chối, stamp
bump + history + idempotent, 3 prompt thật nạp được + giữ hợp đồng CITATIONS +
hash ghim literal + hợp đồng băm-đúng-byte) **+6** tầng service (meta khai
prompt theo route — kể cả nhánh cache replay; `cache_namespace` ghim chuỗi;
store ghi vào namespace có prompt). Integration: test TTL đổi sang khoá mới.

## 8. Còn lại / giới hạn nói to

* Registry **không** ngăn được hai người cùng stamp trên hai nhánh git — merge
  ra hai v2 khác nội dung. Git conflict trên `version:` sẽ bắt hầu hết ca này;
  chấp nhận cho quy mô một người.
* `MAX_HISTORY_MESSAGES` cắt theo số message chứ chưa theo ngân sách token
  (`HFTokenCounter` của `W1-10` chờ sẵn) — docstring cũ hứa "W4-07/W4-11" nhưng
  nó không thuộc DoD hạng mục nào; ghi thành `TD-51` thay vì làm chui.
* Đổi prompt version làm cache miss toàn bộ namespace cũ — đúng hành vi muốn
  có, nhưng đáng nói to: sau mỗi lần stamp, hit rate về 0 cho tới khi cache ấm lại.
