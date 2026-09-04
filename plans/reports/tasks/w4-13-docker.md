# `W4-13` — `serving/Dockerfile` + compose integration + smoke e2e

> 2026-09-04 · Serving Plane · DoD: `make up` → API healthy → `curl /chat` trả
> stream có citation · Test: `tests/e2e/test_smoke.py` · Evidence: log e2e +
> `probes/w4-13-e2e-latency.json`.

## 0. Quyết định lớn nhất được đưa ra TRƯỚC khi viết dòng nào

Bundle `v0.2.0` khai `retriever_name = …@cuda:L512:float16:n50`, và phép kiểm
danh tính của `TD-38`/`W4-02` so tên runtime dựng ra với tên đã eval. Một
container CPU dựng ra `…@cpu:…:float32:…` → **từ chối nạp bundle** → `/ready`
503 → `G4` không thể xanh, trừ khi bật `BUNDLE_ALLOW_RUNTIME_DRIFT`.

Nghĩa là câu "image serving nên nhỏ" — đúng ở mọi dự án khác — ở đây đụng thẳng
vào "demo phải phục vụ đúng hệ thống đã đo". Cái giá của một image nhỏ là một
dòng chữ trong `/admin/bundle` nói rằng mọi con số trong `plans/` đang nói về
một hệ thống khác. Với một repo mà cả `G4` lẫn CV trỏ vào, đó là cái giá sai.

## 1. Dự đoán — chấm 3/5, và cái sai lớn nhất mở ra lựa chọn tốt hơn

| | dự đoán | đo được |
|---|---|---|
| P1 image CUDA | > 6 GB | ✓ **7,35 GB** |
| P2 trọng số | phải mount, không nướng | ✓ mount `HF_HOME` |
| P3 độ trễ | vỡ ngân sách 3500 ms | ✓ **p95 5.312 ms** (§4) |
| P4 GPU passthrough Docker Desktop | **không dùng được** | ✗ **dùng được** |
| P5 smoke phải có marker riêng | đúng | ✓ marker `e2e` |

⭐ **P4 sai là cái sai có lợi nhất của hạng mục.** Tôi định viết cả một mục về
"đường lùi CPU + drift". Một lệnh xoá nó:

```
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
  → NVIDIA GeForce RTX 4060 Laptop GPU, 8188MiB
docker run --rm --gpus all rag-serving:w4-13 python -c "import torch; …"
  → torch 2.14.0+cu126 | cuda available: True | RTX 4060 Laptop GPU
```

Nên image mang torch CUDA, giữ nguyên `@cuda:float16`, và
`GET /admin/bundle` trả **`runtime_drift: null`** — container phục vụ **đúng**
hệ thống mà mọi con số của `W2` nói về. Đó là thứ đáng 4 GB.

## 2. Ba lỗi thật, cả ba chỉ lộ ra khi chạy container

Không lỗi nào trong ba lỗi này bị bất kỳ test nào của 12 hạng mục trước bắt
được, vì cả 12 đều dựng app **trong tiến trình pytest**. Đó chính là lý do
`W4-13` tồn tại như một hạng mục riêng.

1. **Thiếu `alembic/` trong image.** `/health` xanh, bundle nạp xong, Qdrant
   đếm được — chỉ `/ready` trả 503 với `Path doesn't exist: /app/alembic`. Đây
   đúng là kiểu hỏng mà phép kiểm "DB sẵn sàng nghĩa là **migration đã chạy**"
   của `W4-03` được dựng để bắt, và lần chạy container đầu tiên nó bắt thật.
2. ⭐⭐ **Khoá LLM không vào được container, và lý do ngược trực giác.** Dạng
   list không dấu `=` (`- DEEPSEEK_API_KEY`) chỉ lấy từ **shell**, không từ
   `--env-file`. Tệ hơn: khi shell không có biến ấy thì mục đó vẫn **che mất**
   giá trị `env_file` cung cấp — `environment` thắng `env_file` kể cả khi nó
   rỗng. Triệu chứng: mọi thứ xanh, `/chat` trả *"chưa cấu hình LLM"*, và
   `docker exec … echo $DEEPSEEK_API_KEY` in một dòng trống. Sửa: bỏ hẳn ba
   mục pass-through, để `env_file` là đường duy nhất.
3. **Thứ tự ưu tiên `environment` > `env_file` là thứ *phải* đúng chiều này.**
   File biến của repo viết cho đường chạy trên host nên nó khai
   `QDRANT_URL=http://127.0.0.1:6333`; bên trong mạng compose, `127.0.0.1` là
   chính container. Nếu ngược lại thì container tự gọi chính nó.

## 3. Hình dạng image

| | |
|---|---|
| base | `python:3.12-slim`, multi-stage, `uv` |
| kích thước | **7,35 GB** (torch `2.14.0+cu126` chiếm phần lớn) |
| trọng số model | **không** trong image — volume `HF_HOME` (4,4 GB) |
| user | `rag` (uid 10001), không root |
| worker | **1** — `--workers N` nhân bản BGE-M3 + reranker trong 8 GB VRAM |

`--workers 1` không phải tiết kiệm: hai worker là hai bộ trọng số trong cùng
một GPU 8 GB. Mở rộng ngang ở tầng này là thêm **container** kèm một GPU nữa.

`.dockerignore` mới: không có nó thì `docker build .` đẩy cả `data/` (78 MB),
`.venv/`, và — nghiêm trọng hơn — `secrets/` cùng file biến môi trường lên
daemon. Chúng không vào image vì không `COPY` nào chạm tới, nhưng chúng **đi
qua** daemon và nằm trong cache lớp build.

## 4. ⭐⭐ Ô "p95 end-to-end" bỏ trống từ `W1-13` — giờ có số, và nó KHÔNG ĐẠT

8 lượt `/chat` thật qua container ($0,0105), đo bằng **hai đồng hồ**:

| | ms |
|---|---|
| wall p50 (client thật chờ) | **4.118** |
| wall p95 | **5.312** |
| ngân sách mục tiêu | 3.500 |
| **phán quyết** | **KHÔNG ĐẠT** |
| trong đó `prepare()` (embed + hybrid + rerank) p50 | 725 (**18%**) |
| cache hit `W4-10` (cùng câu hỏi lượt trước) | **28,6** |

Ba điều đọc được:

* **Phần vượt ngân sách là bộ SINH, không phải truy hồi.** Truy hồi 725 ms khớp
  với 604 ms đo trên host ở `W2-05` (rc50) — quyết định `rc50` vs `rc100` của
  `W2-08` không phải thứ đang làm vỡ ngân sách. Muốn xuống dưới 3.500 ms thì
  phải chạm tầng sinh: model nhanh hơn, `max_tokens` thấp hơn, hoặc chấp nhận
  ngân sách khác. Đó là việc của `W5-11`/`W6-05`, và giờ nó có một con số.
* ⭐ **Khung `done` khai thiếu ~18% thời gian thật.** `total_ms` đếm từ
  `ChatTurn.started`, tức **sau** `prepare()`. Con số hệ thống tự nói về mình
  nhỏ hơn con số người dùng chịu — một hệ thống đo sai theo hướng có lợi cho
  chính nó là một hệ thống sẽ báo cáo SLA màu hồng → `TD-55`.
* **Cache hit 28,6 ms so với p50 4.118 ms: nhanh 144×.** `W4-10` chạy đúng
  trong container, và đây là lần đầu con số ấy đo được ở đường đầy đủ.

⚠️ Một nhãn của tôi sai và đã sửa trong file probe: dòng 0 được ghi là "lượt
lạnh", thật ra là một **cache hit** (câu ấy vừa được hỏi ở lượt `curl` thủ
công). Thống kê tính trên 7 lượt trượt nên không đổi, nhưng nhãn thì sai.

## 5. Smoke e2e: 7 test, và chúng hỏi những câu không test nào khác hỏi

`tests/e2e/test_smoke.py` **không** import `serving`, không dựng app, không
chạm SQLAlchemy — chỉ nói HTTP với `127.0.0.1:8000`, đúng những gì một người
dùng thật có. Đó là điều kiện để nó bắt được ba lỗi ở §2.

| test | câu nó hỏi |
|---|---|
| `test_ready_means_the_bundle_actually_loaded` | image có đủ thư viện + migration đã chạy |
| `test_the_container_serves_the_bundle_it_was_measured_on` | ⭐ `runtime_drift` rỗng — cửa thoát đang đóng |
| `test_chat_streams_an_answer_with_a_verified_citation` | DoD: citation **`verified: true`**, không chỉ "có citation" |
| `test_an_unauthenticated_chat_is_refused` | cảnh báo của `W4-03` khi mở cổng ra khỏi tiến trình |
| `test_reloading_a_bundle_needs_no_rebuild` | `G4` câu 2 |
| `test_chat_history_survives_a_compose_restart` | `G4` câu 3, bằng một lần restart **thật** |

Chạy thật: **7/7 xanh** (34 s), gồm cả lần restart container.

⚠️ Hai lần đỏ đầu tiên là do bộ test dùng khoá chat để gọi `/admin/bundle`. Đó
là `W4-04` làm đúng việc — scope tách bạch — nên bộ test có `_admin_key()`
riêng thay vì nới scope của khoá demo.

## 6. Tiêm lỗi: 4 phép vào chính hạ tầng, 4/4 đỏ — sau khi sửa chính harness

Khác mọi lần tiêm trước: mục tiêu là **Dockerfile và compose**, phép kiểm là
dựng lại container + chạy 7 test e2e thật.

| phép tiêm | test đỏ |
|---|---|
| D1 bỏ `COPY alembic/` | `test_ready_means_the_bundle_actually_loaded` |
| D2 bỏ `env_file` | `test_chat_streams_an_answer_with_a_verified_citation` |
| D3 `QDRANT_URL` về `127.0.0.1` | `test_ready_means_the_bundle_actually_loaded` |
| D4 bỏ khối GPU `deploy.resources` | `test_ready_means_the_bundle_actually_loaded` |

D1 và D2 tái hiện đúng hai lỗi thật đã gặp ở §2 — tức bộ e2e này thật sự bắt
được chúng, không phải chúng may mà lộ ra.

⭐ **D4 đỏ ở chỗ tôi không đoán.** Dự đoán là nó đỏ ở
`test_the_container_serves_the_bundle_it_was_measured_on` (drift xuất hiện).
Thực tế nó đỏ sớm hơn: không GPU thì phép kiểm danh tính **từ chối nạp bundle
hẳn**, `/ready` 503, chưa kịp có gì để khai drift. Hệ thống nghiêm hơn tôi mô
tả — nó không âm thầm phục vụ một hệ khác, nó không phục vụ.

⚠️⚠️ **Vòng đầu của bảng này là một kết quả giả, và suýt đi vào báo cáo.** Cả 4
phép đều "đỏ" ở cùng một chỗ — fixture không kết nối được — vì harness gọi
`docker compose up -d` rồi chạy pytest ngay: `up -d` trả về khi container được
*tạo*, không phải khi nó *phục vụ được*. Bốn dấu đỏ vì một cuộc đua, không vì
bốn phép tiêm. Thêm vòng chờ `/health` (240 s) rồi chạy lại mới ra bảng trên.
Cùng họ với hai sự cố khác của phiên này (một `replace` không `assert`, một
probe ghi đè output của chính nó): **một phép đo tự động chỉ đáng tin khi biết
nó đang đo cái gì hỏng.**

## 7. `G4` — hai câu rưỡi trên ba

| câu | trạng thái |
|---|---|
| `/chat` trả citation **đã verify** | ✅ đo qua HTTP thật |
| đổi config chỉ bằng `POST /admin/bundle/reload`, không rebuild | ✅ |
| history sống sót `docker compose restart` | ✅ |
| **"từ clone sạch ≤ 5 phút setup"** | ⚠️ **không đạt như viết** |

Nửa câu còn thiếu phải nói thẳng chứ không làm tròn lên. Với cache ấm
(layer Docker + `HF_HOME` đã có 4,4 GB), `make up-api` → `/ready` xanh mất
**~40 giây**. Từ một clone **thật sự sạch** thì đồng hồ bị chi phối bởi hai thứ
không cách nào nén xuống 5 phút: build image 7,35 GB (~10 phút ở lần đầu, phần
lớn là tải wheel torch CUDA 2,5 GB) và tải 4,4 GB trọng số model.

Nên câu ấy của `G4` đúng dưới dạng **"≤ 5 phút khi cache đã ấm"**, và sai dưới
dạng đã viết. Đề xuất: sửa câu chữ của gate thay vì sửa số đo → `TD-56`.

## 8. Còn lại / giới hạn nói to

* Image 7,35 GB **không đẩy lên registry công khai được** trong ngân sách hợp
  lý. `W6-02` (HF Spaces demo) sẽ phải là một image khác — nhiều khả năng CPU +
  một bundle được eval lại trên CPU, chứ không phải bật drift → `TD-57`.
* Chưa có `docker compose` cho Pipeline Plane: ingest vẫn chạy từ host.
* Test e2e cần hai biến môi trường (khoá chat + khoá admin) và **không** tự
  sinh khoá: server nạp key store lúc khởi động, nên một khoá mint giữa chừng
  là một khoá vô hình (`W4-09` đã trả giá bằng hai lần 401).
* Chưa dọn key store: 5 khoá đang nằm trong đó, phần lớn là rác của các lần
  probe → gộp vào `TD-58`.
