# `W4-01` — `RagBundle`: hợp đồng giữa hai plane

> **Ngày:** 2026-09-03 · **Mã:** `packages/rag_core/bundle/` · `pipeline/bundle/build_bundle.py`
> **Test:** `tests/unit/test_bundle.py` (56) · `tests/unit/test_build_bundle.py` (13)
> **Bằng chứng:** `bundles/rag-bundle-v0.1.0/manifest.json` — sinh từ artifact thật

## 1. Câu hỏi thật của hạng mục này

DoD viết là "schema + save/load/validate + checksum", và đọc như một checklist
thì nó là một buổi chiều gõ pydantic. Đọc như một câu hỏi thì nó là:

> **Điều gì khiến câu "serving không import pipeline, hai bên chỉ gặp nhau ở một
> artifact bất biến" đúng, thay vì chỉ là một mũi tên trên sơ đồ?**

Câu trả lời không nằm ở `extra="forbid"` hay ở `sha256`. Nó nằm ở một chỗ nhỏ
hơn nhiều: **trường nào trong schema được phép có giá trị mặc định.**

## 2. ⭐⭐ Mặc định là chỗ kiến trúc rò ra ngoài

Một trường có mặc định trong bundle schema là một hằng số cấu hình **sống trong
mã serving** thay vì trong artifact. Hệ quả không phải là một lỗi — nó là một
loại lỗi không bao giờ đỏ:

| nếu `embedding.normalize` có mặc định | chuyện gì xảy ra |
|---|---|
| bundle đo trên index `normalize=True` | manifest không ghi gì |
| serving deploy bằng image có mặc định `True` | chạy đúng |
| ai đó đổi mặc định trong mã serving thành `False` | **vẫn chạy** |
| kết quả | truy vấn trả về, điểm số ra, thứ hạng sai; gate không gác được vì gate so hai manifest và hai manifest giống nhau |

So sánh với hướng hỏng đối xứng: sai `embedding.dim` thì Qdrant từ chối ngay.
Ồn, xấu, vô hại. **Trường nguy hiểm không phải trường quan trọng nhất, mà là
trường mà sai không ai kêu.**

Nên luật đã chốt: *trường nào ảnh hưởng kết quả truy hồi hoặc sinh thì bắt
buộc*, kể cả khi giá trị của nó gần như luôn giống nhau. Luật ấy được ghim thành
một bảng 17 dòng ở `test_bundle.py::REQUIRED_NO_DEFAULT` — thêm một mặc định vào
đó là làm đỏ một test có ghi sẵn lý do.

### Ba nghĩa khác nhau của `None`, và vì sao phải tách

| trường | `None` nghĩa là | ai kiểm |
|---|---|---|
| `rerank` | **tắt rerank** — một cấu hình hoàn chỉnh | `test_absent_rerank_means_disabled_not_default` |
| `prompt` / `generation` | **bundle không mô tả tầng sinh** — retrieval-only | `serves_generation` + validator chéo |
| `embedding.revision` | **chưa ghim** — hợp lệ nhưng là nợ | `W5` nên chặn lúc promote |

Cái thứ hai là chỗ dễ thành lỗ hổng nhất, nên nó có một ràng buộc chéo: bundle
nào có `eval.generation_metrics` thì **buộc** phải có `prompt` + `generation`.
Không có ràng buộc ấy thì một bundle đo `faithfulness` bằng một prompt nào đó,
không ghi lại prompt ấy, rồi `W5` so nó với bundle khác và quy chênh lệch cho
retrieval.

## 3. Bundle hôm nay khai thiếu — và đó là lựa chọn

Tầng sinh chưa được dựng (`W4-08` router, `W4-11` prompt registry). Hai đường:

* nhét `prompt.hash = "todo"`, `generation.primary = "deepseek-chat"` vào rồi ký;
* để `None` và cho `serves_generation` trả `False`.

Đường đầu cho ra một bundle **đầy đủ về hình thức, bịa về nội dung, mang chữ ký
hợp lệ**. Nó tệ hơn đường sau, vì chữ ký là thứ khiến người đọc thôi kiểm. Bundle
mẫu `v0.1.0` vì thế ghi `"prompt": null, "generation": null` và
`serves_generation = False`, và `W4-03` sẽ để `/ready` trả 503 cho tuyến chat khi
nạp một bundle như vậy — chứ không phải nổ 500 giữa một luồng SSE đang chạy.

## 4. Checksum chứng nhận cái gì, và **không** chứng nhận cái gì

Băm dạng chuẩn hoá của **model đã validate**, không băm byte file. Hai chiều đều
là lựa chọn có chủ đích:

* ✅ format lại JSON (thụt lề, thứ tự khoá, `ensure_ascii`) **không** làm vỡ chữ
  ký. Cần thiết: manifest đi qua git, CI, `docker cp`; một chữ ký vỡ vì thụt lề
  sẽ bị tắt đi trong tuần đầu tiên.
* ✅ **xoá** một trường optional cũng bị bắt — `del raw["notes"]` cho manifest
  validate xanh nhưng chữ ký lệch. Không có tính chất này thì mọi trường optional
  là một chỗ sửa được mà checksum không thấy.
* ❌ **không** chứng nhận collection Qdrant mà bundle trỏ tới đúng là collection
  đã được đo. Checksum bảo vệ *manifest*, không bảo vệ *chỉ mục*.

Và một phép kiểm cố ý nằm **ngoài** checksum: tên thư mục phải khớp
`bundle_version` bên trong. Đây là cách một bản rollback đi nhầm chỗ trong khi
checksum khớp hoàn toàn — vì checksum bảo vệ nội dung, không bảo vệ chỗ đặt.

⚠️ **Hệ quả chưa xử lý (`TD-36`):** thêm một trường có mặc định vào schema sẽ làm
**mọi bundle cũ thành "sai chữ ký"**, vì pydantic lấp mặc định vào lúc đọc và
payload đem băm đổi theo. Đây là mặt trái trực tiếp của tính chất "xoá trường
optional cũng bị bắt" — không giữ được cả hai. Lối ra *không* phải ký lại bundle
cũ (chữ ký khi ấy chứng nhận một nội dung pipeline chưa từng sinh ra) mà là sinh
lại từ artifact nguồn. Phải xử lý trước khi có bundle thứ hai được promote.

## 5. Phần khó nằm ở bên SINH, không ở bên đọc

Checksum bảo vệ manifest khỏi bị sửa *sau* khi ký. Nó không bảo vệ được gì trước
một manifest **sai từ lúc ký**, và đó là kiểu sai dễ xảy ra hơn nhiều:

1. chạy eval trên `rag_bgem3`, đóng gói với `bgem3-contextual.yaml` vì đó là file
   đang mở trong editor;
2. build lại index sau khi eval xong — collection cùng tên, nội dung khác;
3. đổi một tham số chunking, quên chạy lại eval, đóng gói bằng số cũ.

Cả ba cho ra manifest hợp lệ, ký được, checksum khớp. Nên `build_bundle` so **ba
nguồn** (config, báo cáo build index, lượt chạy eval) và **so vân tay chứ không
so tên**: tên collection dùng lại được, vân tay thì không.

Kiểm thật trên artifact có sẵn — đóng gói chéo eval của `rag_bgem3` với config
`bgem3-contextual`:

```
BundleValidationError: vân tay index không khớp với `bgem3-contextual`
  (ff0828fecae7998ad2bf04a389fbe2194ee62506468a6ceb6e9660a981eac52f):
  lượt chạy eval: 0eaaf9265487eabb25eade5ecb6a85a74ebdbee194b1b5e215befe4bec474932
```

Tên collection vẫn được kiểm **riêng**, vì vân tay cố ý không gồm nó (đúng: đổi
tên collection không đổi vector). Bỏ phép kiểm ấy thì bundle trỏ serving vào một
collection **không tồn tại** trong khi mọi vân tay đều khớp.

### Một chỗ xấu, để nguyên có chủ đích

Tham số reranker không nằm trong `branch_options` của báo cáo eval — chúng bị nén
vào chuỗi tên nhánh (`...:L512:float16:n50`). `_parse_rerank` phân tích chuỗi ấy.
Cách làm dở; sửa đúng là đổi định dạng của 15 file báo cáo đã công bố. Đổi lại,
hàm **ném** khi không bóc được thay vì rơi về mặc định — vì `W3-04` vừa đo được
rằng cửa sổ 512 là ràng buộc thật, không phải chi tiết.

## 6. Semver: thứ tự, không chỉ định dạng

Dùng regex chính thức của semver.org thay vì bản rút gọn. Lý do không phải sự
chỉn chu: bản rút gọn thường nhận `1.2.3.4` hoặc `01.2.3`, và một version nhận
sai là một bundle **sắp xếp sai thứ tự**, tức rollback về nhầm bản.

`list_bundles` sắp theo khoá semver chứ không theo tên file. Sắp theo tên thì
`1.10.0 < 1.9.0` và "bản trước đó" — thứ `W4-02` cần — trỏ nhầm bundle. Lỗi này
xuất hiện ở **lần release thứ mười**, lâu sau khi hết ai kiểm bằng mắt.

## 7. Kiểm rằng test có bắt được gì

Test xanh chỉ nói pydantic hoạt động. Tiêm lại hai lỗi:

| lỗi tiêm vào | test đỏ |
|---|---|
| cho `EmbeddingComponent.normalize` một mặc định `True` | `test_field_that_changes_behaviour_has_no_default[EmbeddingComponent-normalize]` |
| bỏ phép kiểm tên thư mục vs `bundle_version` | `test_directory_version_must_match_manifest` |

## 8. Bundle mẫu

`bundles/rag-bundle-v0.1.0/manifest.json` — sinh bằng CLI từ ba artifact thật
(`bgem3-contextual.yaml`, `index-bgem3-contextual.json`,
`bgem3-ctx-rr-c50-retrieval.json`), không gõ tay. Nội dung: `rag_bgem3_ctx`,
15.814 chunk / 60 tài liệu, hybrid RRF `k=1` → reranker pool 50 top-6, nDCG@10
**0,6888**, p95 truy hồi 809 ms, `gate: NOT_RUN`, `serves_generation: false`.

Có hai test ghim nó: nạp lại được, và vân tay trong manifest khớp
`load_index_config(...)` hiện tại — tức nếu ai đó đổi config mà quên sinh lại
bundle mẫu thì test đỏ.

## 9. Còn nợ lại

* `TD-36` — chữ ký vỡ khi schema thêm trường (§4).
* `embedding.revision = None` trong bundle mẫu: chưa ghim commit HF của BGE-M3.
* `gate: NOT_RUN` là thật, không phải chỗ trống — gate được dựng ở `W5-08`.
* `prompt`/`generation` mở sau `W4-08`/`W4-11`; lúc đó `serves_generation` phải
  thành `True` và test §3 phải đổi theo một cách có ý thức.
