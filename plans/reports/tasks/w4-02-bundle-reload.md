# `W4-02` — đổi bundle lúc đang chạy, và lùi lại được

> **Ngày:** 2026-09-03 · **Mã:** `serving/core/registry.py`
> **Test:** `tests/integration/test_bundle_reload.py` (17) — không cần Qdrant/GPU
> **Phạm vi:** lõi hot-swap. `POST /admin/bundle/reload` gắn ở `W4-03` cùng app.

## 1. Vì sao tách endpoint ra khỏi hạng mục này

DoD gộp cả `POST /admin/bundle/reload`, nhưng endpoint cần FastAPI app mà `W4-03`
mới dựng. Ghép hai thứ lại thì phần khó — tính chất về **thời điểm** — sẽ được
test qua một HTTP client, tức qua một tầng không liên quan gì tới cái đang kiểm.

Tách ra thì `BundleRegistry` nhận `RuntimeBuilder` từ ngoài, và toàn bộ những
cách hỏng đáng sợ nhất dựng lại được bằng một builder giả: dựng lỗi giữa chừng,
đổi bundle **ngay giữa** một request đang chạy, hai lệnh reload chồng nhau. Không
cần Qdrant, không cần GPU, chạy trong 2 giây.

## 2. DoD có ba câu, và chỉ câu giữa là khó

| câu | khó ở đâu |
|---|---|
| "đổi bundle không restart process" | gán một thuộc tính |
| ⭐ **"request đang chạy không bị lỗi"** | **tính chất về thời điểm** — không kiểm được bằng cách gọi hàm rồi đọc giá trị trả về |
| "rollback 1 lệnh" | một hàm — nhưng xem §5, nó ràng buộc ngược lại cả thiết kế |

## 3. ⭐⭐ Ba luật, mỗi luật chặn một cách hỏng

### Luật 1 — dựng xong rồi mới đổi

Nạp bundle là: đọc manifest → kiểm chữ ký → dựng retriever → nối Qdrant → nạp
reranker. Bước nào cũng hỏng được. Nếu gỡ bản cũ ra trước thì **một lần reload
lỗi biến `/chat` từ "đang phục vụ bản cũ" thành "không phục vụ gì"** — thao tác
nhằm cải thiện hệ thống lại là thao tác làm sập nó.

Nên phép gán tham chiếu là **việc cuối cùng**, và khoá bao **cả** phần dựng chứ
không chỉ phép gán: hai reload chồng nhau sẽ dựng hai runtime rồi ghi đè lẫn
nhau, và bản thua cuộc thành một runtime không ai tham chiếu nhưng vẫn giữ GPU.

### Luật 2 — request cầm ảnh chụp, không cầm tham chiếu tới "bản hiện tại"

Nếu người gọi đọc `registry.active.retriever` hai lần trong một request và có
reload chen vào giữa, câu trả lời sẽ **trích chunk của index này bằng điểm số của
index kia**. Không có gì đỏ.

Cách chặn không phải là khoá mà là **kiểu dữ liệu**: `active` trả về một
`ActiveBundle` bất biến gói *cả* bundle *và* runtime của nó. Đọc `bundle` từ một
chỗ và `retriever` từ chỗ khác trở thành thứ không viết ra được.

### Luật 3 — không `close()` runtime cũ khi đổi

Cám dỗ tự nhiên là thu hồi kết nối ngay. Làm thế là **kéo đổ đúng những request
mà luật 2 vừa bảo vệ** — chúng đang cầm ảnh chụp cũ và vẫn dùng chính retriever
đó. Runtime cũ sống tới khi request cuối cầm nó kết thúc, rồi GC dọn.

## 4. Hai bug mà chính test tìm ra

Không phải test viết sau khi mã chạy đúng — hai test dưới đây đỏ ngay lần chạy
đầu, và cả hai là lỗi thật:

**`_swap` so danh tính object thay vì so version.** Nạp lại **cùng một** version
(sau khi Qdrant rớt rồi lên lại) dựng ra một `ActiveBundle` *mới*, nên
`outgoing is not snapshot` đúng, nên chính nó bị đẩy vào lịch sử. Hệ quả:
`rollback()` sau đó "lùi" về **đúng bản đang chạy**, trong khi người vận hành tin
là đã lùi. Sửa: so `outgoing.version != snapshot.version`.

Test đồng thời đỏ vì cùng nguyên nhân — hai luồng cùng activate `1.2.0`.

## 5. ⭐ Rollback ràng buộc ngược lại luật 3

Rollback **không** phải là "reload bản cũ". Reload-từ-đĩa hỏng được: đĩa đổi,
mạng rớt, GPU hết chỗ. Mà một cơ chế rollback chỉ có ích **khi mọi thứ đang
hỏng**, nên nó không được phép hỏng.

Vì thế `rollback()` kích hoạt lại **chính object runtime** đã chạy trước đó,
không dựng lại gì — và điều đó chỉ khả thi vì luật 3 giữ nó sống. Test ghim
tính chất ấy bằng cách làm cho việc dựng lại bản cũ **chắc chắn hỏng** rồi đòi
rollback vẫn thành công.

⚠️ **Cái giá, nói thẳng:** giữ bản trước nghĩa là giữ **hai** runtime, và một
cross-encoder là 2,2 GB. Cứu cánh là hai bundle liên tiếp thường chỉ khác tham số
truy hồi và dùng **cùng** model — nhưng chia sẻ instance theo danh tính model là
việc của builder cụ thể (`W4-03`), **chưa làm**. Cho tới lúc đó, đổi sang bundle
khác reranker sẽ nhân đôi bộ nhớ GPU thật. Đó cũng là lý do lịch sử chỉ sâu
**một** bậc.

Hệ quả của lịch sử một bậc: gọi `rollback()` hai lần là quay lại chỗ cũ. Có chủ
đích — một lệnh rollback luôn có nghĩa xác định thay vì phụ thuộc vào đã gọi bao
nhiêu lần.

## 6. Hai lỗi được đặt tên riêng, và vì sao

* `NoBundleLoadedError` — `W4-03` ánh xạ thành `/ready` **503**. "Đang khởi động"
  là trạng thái bình thường, không phải sự cố; một `RuntimeError` chung sẽ thành
  500.
* `NothingToRollBackError` — **ném** thay vì no-op. Một no-op im lặng để người
  vận hành tin rằng hệ thống vừa quay lại, đúng lúc họ đang xử lý sự cố.

## 7. Kiểm rằng test bắt được gì

Tiêm lại lỗi mà luật 1 chặn (đổi trước, dựng sau):

```
FAILED test_a_failed_reload_leaves_the_old_bundle_serving
FAILED test_a_failed_reload_does_not_become_the_rollback_target
FAILED test_reactivating_the_same_version_does_not_make_it_its_own_rollback_target
FAILED test_concurrent_activations_do_not_interleave
```

## 8. Còn lại cho `W4-03`

* `POST /admin/bundle/reload` + `POST /admin/bundle/rollback` + `GET /admin/bundle`
  (`status()` đã trả sẵn đúng ba trường một lệnh `curl` cần).
* `RuntimeBuilder` thật: dựng retriever/reranker từ `bundle.components`, **có
  cache model theo danh tính** (§5), và **kiểm `components.index.fingerprint` với
  collection** trước khi coi là dựng xong — checksum bảo vệ manifest, không bảo
  vệ chỉ mục (`W4-01` §4).
* Endpoint admin chưa có xác thực; phải vào cùng lúc với `W4-04`, không sau.
