# `W5-07` — bảng "RAG Health", và một ô không đo được thứ nó nói

> **DoD**: dashboard JSON commit trong `infra/grafana/`; có RED metrics, cache
> hit rate, retrieval empty rate, refusal rate, cost/hour · **Test**:
> `tests/integration/test_metrics_endpoint.py` · **Evidence**: bảng có số thật
>
> **Kết quả**: đủ · 28 panel · chuỗi API → Prometheus → Grafana chạy thật · chi
> phí **$0,0135** (9 lượt `/chat` thật) + **$0** cho phần hiệu chỉnh.

---

## 0. Bảng, đọc bằng chính PromQL của nó

Không phải ảnh chụp — mỗi ô được hỏi lại qua `POST /api/v1/query` của
Prometheus, và một ô được hỏi thêm một lần nữa qua **datasource proxy của
Grafana** để chứng minh cả chuỗi đứng vững:

```
Scrape sống?                  1
% lượt vượt ngân sách 3,5 s   55,6 %      ← quá nửa
p50 / p95 một lượt            4,25 s / 15,5 s
rerank / truy hồi             93,6 %
cache hit rate                22,2 %
truy hồi rỗng                 0 %
từ chối (ước lượng)           50,5 %
không trích được nguồn        28,6 %
USD / giờ                     $0,0625
USD / câu                     $0,0016828  ← ngân sách ≤ $0,005 ✅
bước chưa định giá được       0
```

Tải: 9 lượt `/chat` (7 sinh mới, 1 trúng cache, 1 `404`) + 1 `401`.
Bằng chứng: [`runs/w5-07-dashboard.json`](../runs/w5-07-dashboard.json).

---

## 1. ⭐⭐ Bảng trực tuyến **không đo được** thứ mà eval đo, và phải nói ra

DoD viết *"refusal rate"*. `W5-02` đã đo đại lượng ấy — bằng một **nhãn của
judge** (`REFUSAL` trong rubric `judge-answer-relevancy`) — và docstring của
`score_refusal` nói thẳng vì sao không dùng từ khoá:

> *"một danh sách từ khoá sẽ bắt được đúng những cách nói tôi nghĩ ra được"*

Câu đó vẫn đúng nguyên si. Nhưng gọi judge cho **mỗi** request thì thêm một lời
gọi model vào đường có người đang đợi, và nhân đôi hoá đơn. Nên lựa chọn thật
sự chỉ có ba: bỏ ô này khỏi bảng, gọi judge trực tuyến, hoặc dùng một ước lượng
và **dán nhãn nó là ước lượng**.

Lối thứ ba đúng vì công việc của một bảng khác công việc của một phép eval:

> Một ước lượng **chệch nhưng ổn định** thì vô dụng để nói *mức*, và hoàn toàn
> dùng được để nói *đạo hàm*.

Metric tên `rag_refusals_suspected_total`, và dòng `HELP` của nó — chỗ người đọc
bảng lúc 3 giờ sáng nhìn thấy, không phải một docstring — viết *"ƯỚC LƯỢNG …
KHÔNG phải phép đo refusal của W5-02"*. Có test ghim câu chữ ấy.

---

## 2. ⭐⭐ Rồi thì đo luôn xem nó chệch bao nhiêu — và một chữ "có" là nguyên nhân

Nói "chệch nhưng ổn định" mà không có số thì cũng chỉ là một lời khai. `W5-02`
để lại 242 câu trả lời thật **và** nhãn judge của chúng trong cache đã đóng
băng, nên phép hiệu chỉnh này tốn **$0** và không một lời gọi mạng nào.

| | precision | recall | F1 | judge | ước lượng | lệch |
|---|---:|---:|---:|---:|---:|---:|
| danh sách từ khoá **bản đầu** (n=242) | 0,838 | **0,633** | 0,721 | 20,3 % | 15,3 % | **−24,5 %** |
| sau khi bổ sung, nửa **giữ ngoài** (n=122) | 0,870 | **0,909** | **0,889** | 18,0 % | 18,9 % | **+4,5 %** |

Phương pháp: chia 242 câu bằng `sha256(query_id) % 2`. Từ khoá bổ sung **chỉ**
được chọn bằng cách đọc ca bỏ sót ở nửa A; con số báo cáo là nửa B, chưa từng
được nhìn. Nửa A đạt F1 0,909 và nửa B đạt 0,889 — chênh 0,02, tức cải thiện
không phải là fit vào tập chỉnh.

**Nguyên nhân lớn nhất của 18 ca bỏ sót là một chữ.** Danh sách có
`"không đủ thông tin"`, còn model viết `"không có đủ thông tin"` — chữ *có* chen
vào giữa, nên chuỗi con không khớp. Hai chuỗi đứng cạnh nhau trông như cái sau
đã bao cái trước. Phần còn lại là dạng tiếng Anh phổ biến nhất trong corpus này
mà danh sách bỏ sót hoàn toàn: *"Based on the provided context, there is no
information about …"*.

⚠️ **Hướng của độ chệch quan trọng hơn độ lớn của nó.** Bản đầu báo **thiếu** một
phần tư số lần từ chối — tức bảng làm hệ thống trông **tốt hơn** thực tế, đúng
hướng chệch tệ nhất, và cùng họ với "một hệ thống đo sai theo hướng có lợi cho
chính nó" đã ghi ở `TD-55`. Sau khi sửa, độ chệch còn **+4,5%** và đổi dấu —
nghiêng về phía báo thừa.

Bằng chứng: [`runs/w5-07-refusal-calibration.json`](../runs/w5-07-refusal-calibration.json).

---

## 3. ⭐⭐ Bảng và trace đọc **cùng một** phép đo

Cách hiển nhiên để có metric là đặt một bộ đồng hồ thứ hai cạnh cây span của
`W5-06`. Đó là cách chắc chắn để tới một ngày bảng Grafana và trace Langfuse nói
hai điều khác nhau về cùng một request, rồi không ai biết cái nào đúng.

Nên `MetricsSink` là một `TraceSink`: cùng cây ấy, hai người tiêu thụ. Một span
mới trong `chat.py` là một dòng mới trên bảng, không cần ai nhớ sửa chỗ thứ hai.

Phần thưởng đo được ngay: `rerank / truy hồi` = **93,6%** ở đây, so với **92,8%**
mà `W5-06` đo trên một mẫu **khác** (5 lượt vs 8 lượt, hai lần chạy cách nhau
mấy giờ). Hai đường đo độc lập cho cùng một kết luận.

| bước | `W5-06` (trace, p50) | `W5-07` (Prometheus, mean) |
|---|---:|---:|
| `retrieval` | 738,1 ms | 739,8 ms |
| `retrieve.hybrid` | 44,8 ms | 47,0 ms |
| `rerank` | 685,3 ms | 692,4 ms |
| phần của rerank | 92,8 % | **93,6 %** |

**⚠️ Điều kiện để lối này không thành bẫy**: `MetricsSink` chạy **trước**
`LangfuseSink` trong `FanoutSink`, đồng bộ, không bao giờ lấy mẫu. Langfuse
**vứt** trace khi hàng đợi đầy (`W5-06` §7) — và hàng đợi đầy đúng lúc hệ thống
bận nhất. Đảo thứ tự thì bộ đếm RED cũng mất đúng phần tải cao, và một bảng
thiếu đúng lúc có sự cố là một bảng nói ngược. Có test đọc mã nguồn để ghim thứ
tự ấy (`M19`).

---

## 4. ⭐⭐ Hai tầng RED, và tầng dưới không thay được tầng trên

| họ metric | nguồn | thấy được |
|---|---|---|
| `rag_http_*` | `RequestContextMiddleware` | **mọi** request, kể cả 401/429/404 |
| `rag_chat_*`, `rag_stage_*` | cây span | chỉ lượt `/chat` **đã qua auth** |

Chỉ có tầng dưới thì một sự cố xác thực — file khoá nạp sai, hạn mức chặn tất cả
— hiện ra trên bảng là **traffic bằng 0**, thứ trông y hệt một đêm yên tĩnh.
Đo được trong lượt chạy thật: `rag_http_requests_total{route="/chat",status="401"}`
= 1, trong khi tầng lượt chat không có gì.

---

## 5. ⭐⭐ Một metric có nhãn **không tồn tại** cho tới lần quan sát đầu tiên

Bài `test_every_metric_the_dashboard_asks_for_exists` đỏ ngay lần chạy đầu với
ba metric: `rag_cache_lookups_total`, `rag_llm_unpriced_steps_total`,
`rag_trace_sink`. Không cái nào bị đổi tên — chúng chỉ **chưa được chạm tới lần
nào**, nên `prometheus_client` không in dòng nào và Grafana vẽ *"No data"*.

Ba chữ ấy mang **hai** nghĩa hoàn toàn khác nhau mà bảng không phân biệt được:

* *"điều kiện này chưa xảy ra"* — bình thường, thậm chí là tin tốt: chưa có
  citation nào hỏng, chưa có bước nào không định giá được;
* *"metric đã đổi tên và không ai sửa bảng"* — hỏng, và hỏng lặng lẽ.

`RagMetrics._declare_zero()` khai trước mọi tổ hợp nhãn **đóng** ở giá trị 0, nên
nghĩa thứ nhất hiện ra là con số 0 và *"No data"* chỉ còn một nghĩa. Chỉ áp cho
tập nhãn đóng: `outcome` và `model`/`step` là mở, và khai trước ở đó là đoán
trước một danh sách sẽ sai.

---

## 6. ⭐ Bài test đọc **dashboard JSON**, không đọc một danh sách viết tay

Một bài liệt kê tay `["rag_http_requests_total", …]` chỉ chứng minh rằng mã khớp
với chính nó. Chế độ hỏng thật của một bảng là **đổi tên metric rồi quên sửa
bảng** — bảng vẫn mở, panel vẫn vẽ, và nó vẽ một đường thẳng ở 0.

Nên bài test bóc mọi biểu thức PromQL trong `rag-health.json` bằng regex và đối
chiếu với những gì `/metrics` thật sự in ra. Nó tìm ra ba thiếu sót thật ngay
lần chạy đầu (§5). Kèm hai bài nữa cùng họ: `uid` datasource của bảng phải khớp
`uid` mà provisioning tạo ra, và bucket `le="3.5"` mà panel "% vượt ngân sách"
đếm phải thật sự tồn tại.

---

## 7. ⭐ Ở đây dùng thư viện, `W5-06` thì viết tay — và phép thử phân biệt hai chỗ

`W5-06` cố ý không dùng SDK Langfuse. Ở đây `prometheus-client` được thêm vào
`pyproject` mà không do dự, và điều đó không mâu thuẫn: phép thử là **dữ liệu
người dùng có đi qua không**.

* Gói tin Langfuse mang **nguyên văn prompt**, nên phải kiểm soát từng byte rời
  khỏi tiến trình — đó là chỗ `tracing.redact()` sống.
* Số đo Prometheus không mang một ký tự nào của người dùng: tên metric, nhãn có
  tập hữu hạn, và số.

Không còn lý do để viết tay, mà định dạng phơi bày thì có đủ thứ tinh vi (thứ tự
`le`, `+Inf`, `_sum`/`_count`, escaping) để một bản tự viết là một khoản nợ không
đổi lại được gì.

---

## 8. ⭐ `/metrics` có xác thực, và không nhãn nào mang tên tenant

Quy ước Prometheus là `/metrics` mở, vì nó thường nằm ở một cổng nội bộ. Ở đây
không có cổng ấy — cùng tiến trình, cùng cổng 8000. Nên `/metrics` **không** nằm
trong `PUBLIC_PATHS`; nó cần một khoá hợp lệ, và không hơn (không đòi scope
`admin`: scraper là một tiến trình hạ tầng, không phải một người vận hành).

Điều kiện đi kèm: **không nhãn nào mang tenant**. Một khoá hợp lệ bất kỳ đọc
được endpoint này, kể cả khoá của tenant khác — nhãn có `tenant` nghĩa là mọi
khách hàng đọc được danh sách khách hàng và lưu lượng của từng người. Đó là dữ
liệu kinh doanh rò qua một cửa không ai coi là cửa dữ liệu. (Nó cũng là một quả
bom số chuỗi thời gian, nhưng lý do thứ nhất đủ một mình.) Phân tách theo tenant
thuộc về Langfuse, nơi có xác thực theo project.

Khoá cho scraper: `make metrics-token` ghi ra một file mà Prometheus đọc bằng
`credentials_file`. Ghi khoá thô ra đĩa là một việc phải nói ra — một scraper
không gõ được mật khẩu — nên file nằm trong thư mục không commit, và cờ
`--token-file` mang một cảnh báo ba dòng ngay tại chỗ.

---

## 9. Hai lỗi thật gặp trong lúc dựng

**⭐⭐ Câu trả lời từ cache không bao giờ được quét từ chối** — tìm ra bởi một
phép tiêm **sống sót**. Phép quét nằm bên trong nhánh `kind == "generation"`,
nên `cache.replay` (một span thường) rơi ra ngoài. Hệ quả: mẫu số là
`rag_chat_turns_total` (có tính lượt cache) còn tử số **không thể** tăng cho
chúng — ước lượng lệch xuống đúng bằng tỉ lệ trúng cache, đo được 22,2%. Phép
tiêm "sống" vì đổi luật thành *quét mọi span* gần như không đổi hành vi, và điều
đó chỉ đúng khi luật hiện tại đang bỏ sót gần hết những gì đáng quét.

**⭐ Nghe `127.0.0.1` làm API vô hình với scraper trong container.** Mặc định
loopback là đúng cho một máy dev, và triệu chứng của nó là
`dial tcp 192.168.65.254:8000: connect: connection refused` trong Prometheus —
trông như "Prometheus hỏng" chứ không như "server nghe hẹp". Job vì thế có **hai**
đích (`api:8000` cho `--profile api`, `host.docker.internal:8000` cho uvicorn
trên host) và panel dùng `max(up{...})`: câu hỏi là *"có đọc được số đo từ ít
nhất một chỗ không"*, không phải *"cả hai chỗ có chạy không"*.

---

## 10. Bảng nói gì về hệ thống hôm nay

* **55,6% số lượt vượt ngân sách 3,5 s.** Không nội suy — panel đếm
  `rag_chat_turn_duration_seconds_bucket{le="3.5"}`, và bucket ấy tồn tại đúng vì
  3,5 s là con số của bảng mục tiêu.
* **$0,0016828 một câu**, dưới ngân sách $0,005. Đây là số đo **cost/query** đầu
  tiên của dự án; ô ấy trong bảng mục tiêu đang là *chưa đo*.
  ⚠️ Trên 9 lượt, một model, có cache — `W5-11` mới là phép đo có thẩm quyền.
* **93,6% thời gian truy hồi là cross-encoder.** Xác nhận độc lập cho `W5-06`.
* **0 bước không định giá được**, nên mọi ô tiền ở trên là tổng thật chứ không
  phải cận dưới.

---

## 11. Test và tiêm lỗi

| | số bài |
|---|---:|
| `tests/unit/test_metrics.py` | 44 |
| `tests/integration/test_metrics_endpoint.py` | 21 |
| **bộ mặc định** | **2 201** xanh, 3 skip (+44) |
| **bộ integration** | **275** xanh, 0 hỏng (+21) |

Tiêm lỗi **22 phép, 22 đỏ** — nhưng lượt một có **một phép sống sót**, và nó là
một lỗi thật (§9). Mọi phép được chọn theo cùng một khuôn: không phép nào làm
chương trình nổ, và mỗi phép để lại một ô **vẫn vẽ ra** và vẽ sai.

---

## 11b. ⚠️ Đính chính một con số của `W5-06`

Báo cáo `W5-06` viết *"`make lint` sạch cả ba lệnh"*. Không đúng. Tôi chạy
`mypy serving/ packages/ pipeline/`, còn `make lint` chạy `mypy` **không tham
số** — tức gồm cả `tests/`. Lệnh thật đỏ **8 lỗi** ở ba file test của `W5-06` và
`W5-07`:

* `unreachable` sau một khối `with` mà `__exit__` khai `Literal[False]` (mypy
  biết nó không nuốt ngoại lệ, nên dòng sau `with` là *đến được* — chỗ này thì
  ngược lại, mypy đúng và cách viết của tôi làm nó nghĩ nhầm);
* hai `Returning Any` từ `.text` và một `list` chưa chú kiểu;
* `build_runtime=lambda bundle: (FakeReranked(), None)` không khớp Protocol
  `RuntimeBuilder`;
* hai `type: ignore` thừa.

Đã sửa hết trong hạng mục này. Điều đáng ghi không phải "quên chạy" mà là **thay
một lệnh bằng một lệnh hẹp hơn rồi báo cáo bằng tên của lệnh rộng** — cùng đúng
một hình dạng lỗi mà cả `W5` đang đi tìm ở chỗ khác.

---

## 12. Nợ mới

| mã | nợ |
|---|---|
| `TD-75` | **`prometheus_client` giữ số đo trong bộ nhớ của MỘT tiến trình.** `W4-13` chạy 1 worker nên hôm nay đúng; bật worker thứ hai thì mỗi worker có sổ riêng và scraper hỏi trúng một cái ngẫu nhiên — bảng tụt đi một nửa mà không có gì báo. `TD-63` (khoá GPU) đang là lý do giữ 1 worker, nên hai nợ này phải trả cùng nhau. Lối ra: `PROMETHEUS_MULTIPROC_DIR`, hoặc chấp nhận và **ghi số worker vào một gauge** để bảng tự nói ra giả định của mình |
| `TD-76` | **p95 đọc từ histogram là phép nội suy trong bucket, không phải phép đo.** Lượt chạy này báo p95 = 15,5 s trong khi lượt chậm nhất thật khoảng 10 s — một mẫu rơi vào bucket `(10, 20]` và `histogram_quantile` nội suy tuyến tính trong đó. Đúng theo định nghĩa, sai theo trực giác, và ở lưu lượng thấp thì luôn xảy ra. Cần một dòng chú trên panel, và `W6-05` (load test) là chỗ con số này bắt đầu có nghĩa |
| `TD-77` | **Bộ dò từ chối được hiệu chỉnh trên MỘT corpus và MỘT model.** F1 0,889 là của `deepseek-v4-flash` trả lời trên corpus World Bank; `W5-11` sẽ đổi model sinh, và cách nói "tôi không tìm thấy" của model khác có thể không nằm trong danh sách. Phép hiệu chỉnh tốn $0 và tái lập được — nên nó phải **chạy lại trong `W5-11`** và F1 phải là một con số trong bảng ablation, không phải một hằng số trong báo cáo này |
