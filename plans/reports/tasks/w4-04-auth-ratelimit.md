# `W4-04` — xác thực, hạn mức, và chỗ ép tenant mà `W2-06` để lại

> **Ngày:** 2026-09-03 · **Mã:** `serving/api/security.py`,
> `serving/core/{auth,ratelimit}.py`
> **Test:** `tests/integration/test_auth_ratelimit.py` (21) + `tests/unit/test_auth.py` (21)
> **Kiểm thật:** uvicorn thật trên `:8078` với key thật — §7

## 1. ⭐⭐ Middleware chặn theo mặc định, không `Depends` từng route

`Depends(require_key)` là cách mọi ví dụ FastAPI làm, và nó hỏng theo **hướng
sai**: quên gắn dependency vào một route mới thì route đó **công khai**, im
lặng, và không có gì trong diff trông bất thường — chỉ là một endpoint mới không
có tham số thừa.

Middleware chặn-theo-mặc-định hỏng theo hướng ngược lại: quên nghĩ về một route
mới nghĩa là nó **bị khoá**, lộ ra ngay ở lần gọi thử đầu tiên. Muốn mở thì phải
viết tên nó vào `PUBLIC_PATHS` — một dòng **nhìn thấy được** trong review, thay
vì một dòng *không* được viết.

Cùng lý lẽ cho scope admin: quy tắc là **tiền tố đường dẫn** (`/admin/**` cần
`ADMIN_SCOPE`), không phải một dependency gắn tay. Một route admin mới được bảo
vệ vì nó *ở trong* `/admin`, không vì ai đó nhớ. Đây là chỗ đóng lỗ hổng 🔓 mà
`W4-03` cố ý để mở và ghim bằng test.

`PUBLIC_PATHS` là tập đường dẫn **chính xác**, không phải tiền tố: một tiền tố
`/health` cũng mở luôn `/health-internal-debug`.

## 2. ⭐⭐ Test đắt giá nhất — và lần viết đầu nó không kiểm gì cả

`test_every_route_is_either_public_on_purpose_or_locked` không kiểm một hành vi
mà kiểm rằng **không tồn tại** một route lọt ra ngoài.

⚠️ Bản đầu duyệt `app.routes`. FastAPI ở phiên bản này gói mỗi `include_router`
vào một `_IncludedRouter` **không có `.path`**, nên vòng lặp chỉ thấy 4 route tài
liệu và **không thấy một route `/admin` nào** — test xanh trong khi nó không kiểm
cái nó nói. Thứ cứu nó là một dòng tưởng thừa: `assert len(paths) >= 8`. Một
phép **đếm cận dưới** là thứ duy nhất phân biệt được "quét sạch, không có gì hở"
với "không quét gì cả".

Cùng họ với `git check-ignore -q` ở `W3-04` (lệnh trả 0 cả khi khớp dòng phủ
định, nên test luôn xanh): một phép kiểm phủ định **phải** kèm bằng chứng rằng
nó có nhìn thấy gì.

Bản đúng lấy hợp của `app.router.routes` và `app.openapi()["paths"]`.

## 3. SHA-256 chứ không bcrypt — và vì sao đó không phải cắt góc

"Băm bí mật thì phải dùng KDF chậm" đúng với **mật khẩu** và sai ở đây. KDF chậm
bù cho **entropy thấp**: người ta chọn `Hanoi2024!`, nên phải làm mỗi lần thử tốn
100 ms. API key ở đây là 32 byte từ `secrets` — không gian 2²⁵⁶, không từ điển,
không bảng cầu vồng.

Cái giá của KDF chậm thì trả **mỗi request**: bcrypt cost 12 ~250 ms sẽ là thành
phần chậm nhất của cả đường `/chat`. Một lượt SHA-256 cho phép tra **thẳng bằng
digest** trong dict — O(1), không so tuần tự với từng key. Đó cũng là lý do
không cần `hmac.compare_digest`: **không có phép so chuỗi nào chạy trên bí mật
cả**, digest là *khoá tra*, và đoán trúng khoá tra tức là đã đoán trúng key.

Bí mật thô không bao giờ chạm đĩa: `mint()` in ra đúng một lần rồi ghi digest.

## 4. Bốn hướng-hỏng được chọn có chủ đích

| tình huống | chọn | vì sao chiều kia sai |
|---|---|---|
| không có file key | **khoá hết** | mở hết biến "quên mount volume" thành API công khai, và **mọi request đều thành công** nên không gì báo |
| key sai vs không có key | **cùng một phản hồi** | phân biệt = biến endpoint thành **máy kiểm key** |
| xin tenant khác | **từ chối**, không ghi đè | ghi đè lặng lẽ an toàn về *dữ liệu* nhưng biến request sai thành kết quả rỗng hợp lệ |
| key đúng, thiếu scope | **403** | 401 nghĩa là "thử credential khác" — hướng dẫn sai |

Và lỗi từ chối cross-tenant **không** nhắc tên tenant được xin, nên nó không
thành máy đếm khách hàng.

## 5. ⭐⭐ Chỗ ép tenant — nợ của `W2-06`

`rag_core/retrieval/filters.py` ghi thẳng giới hạn của nó: *"nó không ép người
gọi phải truyền `tenant_id`. `retrieve(query)` không filter vẫn thấy tất cả, và
không có chỗ nào trong `rag_core` biết được như vậy là đúng (eval chạy toàn
corpus) hay là một lỗ rò."*

`tenant_filter(principal, requested)` là chỗ phân biệt được, vì tenant đến từ
**token đã xác thực**. Ba luật: không truyền gì ⇒ vẫn lọc theo token; truyền
tenant khác ⇒ `CrossTenantError`; mọi field khác giữ nguyên (chúng chỉ thu hẹp).

Và đây là chỗ `frozen` của `MetadataFilter` được dùng đúng mục đích `W2-06` đặt
nó: kiểm xong rồi mà filter còn sửa được thì phép kiểm chỉ là trang trí.

⚠️ Hàm này **chưa được gọi từ route nào** — route đầu tiên nhận filter là
`/chat` của `W4-06`. Cơ chế và test đã có; chỗ nối là một dòng ở đó.

## 6. Token bucket, và con số `Retry-After`

Cửa sổ cố định cho phép **120 request trong 2 giây** ở hạn mức 60/phút (60 cuối
cửa sổ này + 60 đầu cửa sổ sau), và nó không hiện ra trong bất kỳ phép đo trung
bình nào. Token bucket không có ranh giới để dồn vào, và nó tính được chính xác
câu mà 429 bắt buộc phải trả lời: bao giờ thử lại được.

⭐ `Retry-After = max(1, ceil(wait))`. `int()` hay `round()` cho **0** với mọi
khoảng chờ dưới một giây, và một client *lịch sự* đọc `Retry-After: 0` sẽ thử
lại ngay — nhận 429 tiếp, thử lại ngay. Header sinh ra để giảm tải thành một
vòng lặp nóng.

Ba điều khác đã nói rõ trong mã: sức chứa bằng hạn mức một phút nên 60 request
trong một giây đầu là **hợp lệ** (hạn mức này là "60/phút tính trung bình");
`monotonic` chứ không đồng hồ tường; và bucket đã đầy bị dọn vì với tenant sinh
động nó là một chỗ rò bộ nhớ tăng đều.

⚠️ **Trong-tiến-trình: N replica = N lần hạn mức.** 4 pod cho mỗi tenant 240/phút
chứ không phải 60, và con số đó đổi mỗi lần autoscale. → `TD-39`.

⚠️ **Request chưa xác thực không bị giới hạn** — tenant chỉ biết sau khi xác
thực. Không phải sơ suất: một lượt từ chối tốn đúng một SHA-256 và một phép tra
dict, nên thứ bị tiêu thụ là *kết nối*, và chỗ chặn theo IP là reverse proxy
phía trước, nơi có thông tin kết nối thật.

## 7. Chạy thật

`make api-key TENANT=local-dev SCOPE=admin RPM=5`, rồi uvicorn thật:

```
health (không key): 200      admin (không key): 401     docs (không key): 401
www-authenticate: Bearer     admin (có key): 200
5 lượt gọi liên tiếp: 200 200 200 200 429 429 429
HTTP/1.1 429 · x-ratelimit-limit: 5 · x-ratelimit-remaining: 0 · retry-after: 12
```

`retry-after: 12` đúng số học: 5/phút = một token mỗi 12 giây. Log JSON mang
`request_id` cho cả 401 lẫn 429:

```json
{"level":"WARNING","msg":"401 GET /admin/bundle (không có key)","request_id":"039dec47…"}
{"level":"WARNING","msg":"429 tenant local-dev vượt 5/phút","request_id":"dd32b772…"}
```

## 8. ⚠️ Một lỗi tôi đã ship ở `W4-03`, tìm ra khi thêm target `api-key`

`make` chết ngay ở dòng đầu: `*** target pattern contains no '%'`. Công thức
`serve-check` của `W4-03` chứa một **xuống dòng thật** giữa tham số `curl -w`
thay vì `\n`, nên dòng kế bắt đầu bằng dấu nháy chứ không phải TAB. Hệ quả:
**toàn bộ** Makefile không dùng được, không riêng target ấy — và nó đã nằm trong
commit `3dec677`.

Đáng nói vì đó là ca **đối xứng** với chính lỗi đã sinh ra
`tests/unit/test_makefile_recipes.py`: file đó bắt "`\n` nằm nguyên trong công
thức"; đây là "`\n` đã bị diễn dịch thành xuống dòng thật". Cả 10 test ở đó xanh,
vì chúng đọc Makefile như **văn bản** — chỉ có `make` biết `make` parse được hay
không. Thêm `test_make_can_actually_parse_the_makefile` (chạy `make -n help`);
tiêm lại đúng lỗi cũ thì nó đỏ.

## 9. Kiểm bằng tiêm lại lỗi — 8/8 đỏ đúng chỗ

| lỗi tiêm vào | test đỏ |
|---|---|
| bỏ phép kiểm principal | 8 test, gồm cả `…api_description_is_not_public` |
| bỏ phép kiểm scope admin | `…tenant_key_cannot_touch_the_admin_routes`, `…403_not_401` |
| `Retry-After` dùng `int()` thay `ceil` | `test_retry_after_is_at_least_one_second` |
| cross-tenant ghi đè thay vì ném | `…refused_not_silently_rewritten`, `…does_not_reveal_whether…` |
| filter mặc định không mang tenant | `…without_a_filter_is_still_scoped_to_its_tenant` |
| hạn mức khoá theo hằng số thay vì tenant | `test_one_tenant_cannot_starve_another` |
| thêm `/admin/bundle` vào `PUBLIC_PATHS` | 6 test |
| đảo thứ tự hai middleware | `test_a_401_still_carries_a_request_id` |
| trả `\n` thật vào công thức Makefile | `test_make_can_actually_parse_the_makefile` |

## 10. Còn nợ

* **`TD-39`** — hạn mức trong-tiến-trình; và request chưa xác thực không bị chặn.
* **`tenant_filter` chưa có route gọi** — nối ở `W4-06` (§5).
* **Chưa có đường thu hồi key** ngoài việc sửa file rồi restart: `ApiKeyStore`
  nạp một lần lúc khởi động. Với `W4-05` (Postgres) thì kho key về DB và thu hồi
  thành một câu `UPDATE`; trước đó, thu hồi = sửa file + restart, và điều đó phải
  nằm trong runbook chứ không nằm trong đầu ai.
* **Chưa có JWT** — DoD viết "API key / JWT". API key đủ cho quan hệ
  service-to-service hiện có; JWT chỉ đáng khi có người dùng cuối đăng nhập, tức
  sớm nhất là khi có UI (`W5`). Chọn một cái và nói rõ, thay vì làm nửa vời cả hai.
