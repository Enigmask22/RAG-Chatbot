# `W5-04` — Hiệu chỉnh judge: 0,9877 đúng, nhưng nó là con số của **một** giám khảo

*2026-09-05 · `pipeline/eval/kappa.py`, `pipeline/eval/calibration.py`,
`tests/unit/test_kappa.py` (32 test), `tests/unit/test_calibration.py` (24 test)*

**Câu trả lời ngắn**: faithfulness `0,9877` mà `W5-01` báo là **đúng** — ước lượng
theo nhãn tay quy về quần thể là `0,9905` [0,9800 – 1,0000], bao con số của judge.
Ngưỡng `≥ 0,92` qua thật, có biên.

**Câu trả lời dài, và là lý do hạng mục này đáng làm**: đổi **duy nhất** model
judge, giữ nguyên hệ thống, nguyên rubric, nguyên 242 câu trả lời, thì cùng một
metric đọc ra `0,9246` hoặc `1,0000`. Khoảng dao động do giám khảo (**7,5 điểm**)
lớn hơn mọi cải thiện hệ thống mà `W2` đo được cộng lại.

| nguồn nhãn | faithfulness (quy về quần thể) | κ vs người |
|---|---|---|
| **người** (50 mẫu, nhãn tay) | **0,9905** [0,9800 – 1,0000] | — |
| `deepseek-v4-flash`, suy luận **tắt** ← đang dùng | **0,9877** | **0,737** |
| `glm-5.3-flash` (Z.ai), khác họ | **0,9246** [0,8284 – 0,9928] | 0,371 |
| `deepseek-v4-flash`, suy luận **bật** | **1,0000** ⚠️ | 0,112 |

---

## 0. Nhánh cross-check không dựng được như plan viết — và `W0-07` vì thế **hết hiệu lực**, không phải được giải

Plan: *"cross-check bằng 1 judge khác họ (OpenRouter, pin slug)"*, và bảng câu hỏi
mở liệt kê `W0-07` (*"`@preset/my-luna-pro` resolve ra model slug nào"*) là thứ
**chặn** `W5-04`.

Không có `OPENROUTER_API_KEY` trong môi trường. Checklist đã ghi điều này từ
`W3-08` (dòng 695) và tôi không tự đặt thêm được.

Nhưng đọc lại thì `W0-07` chưa bao giờ là ràng buộc thật. Nó hỏi "preset ấy trỏ
vào đâu" — trong khi **quy tắc cứng #1 cấm dùng preset trên đường eval bất kể nó
trỏ vào đâu**. Biết đáp án cũng không cho phép dùng. Thứ hạng mục này thật sự
cần chỉ là **một họ model thứ hai**, và `GLM_API_KEY` đã có sẵn:
`glm-5.3-flash` của Z.ai (Zhipu) khác họ `deepseek-v4-flash` đúng theo nghĩa cần
thiết — khác nhà, khác dữ liệu huấn luyện, khác hành vi.

Nên `W0-07` chuyển sang **đóng, không cần trả lời**: không phải vì đã tìm ra
đáp án, mà vì câu hỏi hoá ra không dẫn tới quyết định nào.

Việc mở `Judge` cho họ thứ hai là bốn thay đổi nhỏ, và một trong số đó có bẫy:

```python
@property
def family(self) -> str:      # SUY RA từ slug, không phải một field khai riêng
```

Cám dỗ là thêm `family: str` vào `JudgeConfig`. Làm vậy là mở đường cho cấu hình
khai `family="glm"` với `model="deepseek-v4-flash"` — và khi ấy `extra_body` gửi
sai họ **mà không có lỗi nào**: `W3-04` đã đo được rằng DeepSeek *nhận*
`reasoning_effort` rồi lặng lẽ bỏ qua. Phán quyết vẫn về, vẫn vào cache, chỉ là
được sinh dưới một điều kiện khác lời khai. Suy ra từ slug thì trạng thái ấy
không dựng lên được, và `family` cũng **không** cần vào khoá cache (model đã ở
trong đó) nên 1664 phán quyết của `W5-01` không bị huỷ.

⚠️ **Hai nhánh không so được ở điều kiện suy luận giống nhau, và không sửa được.**
`glm-5.3-flash` trả HTTP 400 khi bị yêu cầu tắt suy luận; mức thấp nhất nó nhận
là `reasoning_effort="low"`. DeepSeek tắt hẳn được. Mọi bất đồng giữa hai nhánh
vì thế mang lẫn một phần khác biệt về điều kiện. Ghi ra đây để không kết luận
quá tay ở §5.

---

## 1. ⭐⭐ Vì sao 50 mẫu ngẫu nhiên đều sẽ là 50 mẫu vô dụng

Phân bố nhãn faithfulness của `w5-answers-v1`:

```
SUPPORTED     402   92,84%
NO_CLAIM       26    6,00%
NOT_FOUND       5    1,15%
CONTRADICTED    0    0,00%
```

Với biên độ ấy, `Pe = 0,8657`. Mẫu số của kappa, `1 − Pe`, chỉ còn **0,134** —
nên **một bất đồng thêm trên 50 mẫu kéo κ đi khoảng 0,15**. Đó là nghịch lý
kappa ở dạng nặng nhất: hai người chấm có thể trùng nhau 96% mà κ vẫn thấp.

Và tệ hơn: lấy 50 mẫu ngẫu nhiên đều thì **kỳ vọng chỉ 0,58 mẫu `NOT_FOUND`**.
Nhánh mà judge dễ sai nhất thường sẽ không có mẫu nào, và toàn bộ công gán nhãn
tay chỉ chứng minh được một câu — *"judge gán `SUPPORTED` cho những thứ vốn là
`SUPPORTED`"*.

Nên mẫu phân tầng, phân bổ khai tường minh trong mã (`DEFAULT_ALLOCATION`) chứ
không nổi lên từ dữ liệu:

| tầng | quần thể | lấy | trọng số |
|---|---|---|---|
| `SUPPORTED` | 402 | 30 | 13,4 |
| `NO_CLAIM` | 26 | 15 | 1,733 |
| `NOT_FOUND` | 5 | 5 | 1,0 |
| `CONTRADICTED` | 0 | — | muốn 5, không có mẫu nào |

Hệ quả **bắt buộc phải công bố**: κ tính trên 50 mẫu ấy không phải κ của quần
thể. Hai con số trả lời hai câu khác nhau, và in một cái là nói dối bằng cách bỏ
bớt:

* **κ mẫu** = *"ở chỗ khó, judge và người hợp nhau tới đâu?"*
* **κ quần thể** (Horvitz–Thompson, trọng số = |tầng| / |mẫu tầng|) = *"nếu chấm
  tay cả 433 mệnh đề thì κ là bao nhiêu?"*

`kappa.py` giữ cả hai, và bootstrap lấy lại mẫu **trong từng tầng** — lấy trên
toàn bộ 50 cặp sẽ sinh ra những lần rút có 0 phần tử ở tầng `NOT_FOUND`, tức là
bootstrap trên một thiết kế khác thiết kế thật.

Một quyết định nhỏ nhưng dễ sai: **κ = `None` khi `Pe = 1`**, không phải `1.0`.
Nếu cả hai bên gán `SUPPORTED` cho cả 50 mục thì `0/0` nghĩa là *không có thông
tin*, không phải *đồng thuận hoàn hảo* — hai đồng hồ đứng yên cũng chỉ cùng một
giờ. Trả `1.0` ở đây là cách một báo cáo tự khen mình.

---

## 2. ⭐⭐ File "mù" của tôi rò rỉ, và tôi phát hiện ở mẫu thứ 18/50

`sample` ghi hai file: `*-blind.jsonl` (chỉ `ref`, `context`, `claim` — không có
nhãn nào) và `*-sealed.jsonl` (phán quyết judge, tầng, trọng số), với `sha256`
của file sealed nằm trong header file blind để `score` kiểm lại.

Thiết kế ấy đúng. **Cài đặt thì không.** Bản đầu ghi mẫu theo tầng: 5 mục
`NOT_FOUND`, rồi 15 mục `NO_CLAIM`, rồi 30 mục `SUPPORTED`. File không chứa một
nhãn nào — nhưng **vị trí dòng thì chứa**. Đang đọc tới mục 17, thấy sáu mục
liên tiếp đều là câu meta kiểu *"các nguồn chỉ đề cập…"*, tôi nhận ra ranh giới
tầng và từ đó biết trước judge đã nói gì cho 32 mục còn lại.

Cái đo được lúc ấy không còn là *"người nghĩ gì"* mà là *"người có phản đối judge
không"* — và mọi thiên lệch đều đi về phía đồng thuận, tức là về phía một con số
κ đẹp hơn sự thật.

Xử lý: trộn thứ tự bằng `random.Random(f"{seed}:order")`, **bốc lại mẫu với seed
khác** (`20260906`), vứt 18 nhãn đã gán, gán lại từ đầu cả 50.

Sắp theo `ref` không cứu được: `ref` là `query_id#chỉ_số_câu`, mà truy vấn
`unanswerable-*` sinh ra phần lớn câu meta còn `#s0` phần lớn là câu dẫn — thứ tự
ấy vẫn tương quan với nhãn. Chỉ có trộn mới làm thứ tự **không mang thông tin**
theo nghĩa xây dựng được.

Bài test hồi quy không kiểm "file có cột nhãn không" — nó đếm số lần đổi tầng khi
đi dọc danh sách. Gom cụm ⇒ đúng 2 lần đổi; trộn ⇒ hơn 10:

```python
switches = sum(1 for a, b in pairwise(strata) if a != b)
assert switches > 10, f"thứ tự vẫn gom cụm theo tầng ({switches} lần đổi)"
```

**Bài học**: một lớp bảo vệ chống thiên lệch phải được kiểm bằng *thứ có thể suy
ra được*, không phải bằng *thứ có mặt trong file*.

---

## 3. Tự kiểm gắn sẵn: trọng số phải dựng lại đúng con số đã báo cáo

Tầng **chính là** nhãn của judge, nên tỉ lệ có trọng số ở phía judge bắt buộc
trùng khít con số `W5-01`. Nếu lệch, trọng số sai — và khi ấy con số phía người,
thứ duy nhất ta thật sự muốn biết, cũng sai theo.

```
judge_reweighted   0,9877149877   CI [0,9877149877, 0,9877149877]
```

Khớp `402/407 = 0,9877149877` tới chữ số cuối. Khoảng tin cậy suy biến thành một
điểm, và đó là **đúng** chứ không phải lỗi: trong mỗi tầng mọi nhãn judge giống
hệt nhau, nên lấy lại mẫu trong tầng không đổi được biên độ của judge. Nó là bằng
chứng trọng số đúng, không phải bằng chứng phép đo chính xác.

---

## 4. Judge hiện tại vs người: κ = 0,737, và ba lỗi đều lệch về **một** phía

| | Po | κ | κ CI 95% | PABAK |
|---|---|---|---|---|
| trên 50 mẫu | 0,900 | **0,813** | [0,683 – 0,926] | 0,800 |
| quy về quần thể | 0,958 | **0,737** | [0,478 – 0,966] | 0,916 |

Ngưỡng DoD `≥ 0,6`: **qua ở cả hai cách đọc**. ⚠️ Nhưng cận dưới của khoảng tin
cậy quần thể là `0,478` — với n=50 thì dữ liệu **không loại trừ** được những giá
trị dưới ngưỡng. "Qua" ở đây là điểm ước lượng, không phải một kết luận chắc.

Ma trận nhầm lẫn trên mẫu (hàng = người, cột = judge):

| | judge S | judge NF | judge NC |
|---|---|---|---|
| **người SUPPORTED** (30) | 29 | 1 | 0 |
| **người CONTRADICTED** (1) | 0 | 0 | 1 |
| **người NOT_FOUND** (2) | 0 | 2 | 0 |
| **người NO_CLAIM** (17) | 1 | 2 | 14 |

`per_label` tồn tại vì κ là **một** con số cho **cả** ma trận: nó không phân biệt
"judge bỏ sót" với "judge báo động giả", trong khi hai lỗi ấy đẩy faithfulness về
hai hướng ngược nhau.

```
SUPPORTED     precision 0,967  recall 0,967
NO_CLAIM      precision 0,933  recall 0,824
NOT_FOUND     precision 0,400  recall 1,000     ← 5 lần gọi, đúng 2
CONTRADICTED  precision   —    recall 0,000     ← 1 ca, trượt
```

**`NOT_FOUND` precision 0,40 là phát hiện chính.** Cả ba lần báo động giả đều đẩy
faithfulness **xuống**: một ca đưa mệnh đề đúng vào mẫu số như một thất bại, hai
ca kéo câu `NO_CLAIM` (đáng lẽ bị loại khỏi cả tử lẫn mẫu) vào mẫu số. Đó chính
là lý do con số theo nhãn tay (`0,9905`) **cao hơn** con số của judge (`0,9877`):
judge hiện tại thận trọng, và nó sai về phía an toàn.

### Năm ca bất đồng, đọc từng ca

| ref | người | judge | GLM | chuyện gì |
|---|---|---|---|---|
| `aggregation-3895b9f4ae#s5` | NO_CLAIM | NOT_FOUND | NO_CLAIM | *"…but no equivalent limit is stated for MFIs"* — nói về **phạm vi của nguồn**, luật 2 |
| `cross_lingual-3e5b84a32c#s5` | NO_CLAIM | NOT_FOUND | SUPPORTED | *"…nhưng không có thông tin về cơ quan quản lý chúng"* — cùng dạng |
| `aggregation-c332ac1180#s1` | NO_CLAIM | SUPPORTED | SUPPORTED | câu dẫn kết thúc bằng dấu hai chấm |
| `unanswerable-599e160838#s3` | SUPPORTED | NOT_FOUND | CONTRADICTED | ngữ cảnh **có** "chi R&D của Việt Nam (0,5% GDP)"; judge bỏ sót |
| `factoid-38cc46d0ad#s1` | **CONTRADICTED** | NO_CLAIM | CONTRADICTED | xem §4.1 |

**Hai ca đầu là cùng một chỗ mờ trong rubric**: ranh giới giữa "câu nói nguồn
thiếu thông tin" (`NO_CLAIM`, luật 2) và "ngữ cảnh không nói gì về điều này"
(`NOT_FOUND`). Khi mệnh đề *là* một nhận xét về sự thiếu vắng, hai định nghĩa
chồng lên nhau. Judge chọn `NOT_FOUND`, tôi chọn `NO_CLAIM` vì luật 2 nêu đích
danh mẫu câu ấy.

### 4.1 ⚠️ Một lỗ hổng trong rubric v2 mà chỉ việc gán nhãn tay mới thấy

`factoid-38cc46d0ad#s1`:

> *"Các nguồn chỉ đề cập đến tăng trưởng doanh số bán lẻ tháng 9 so với cùng kỳ
> năm trước (9,4%) … **nhưng không nêu rõ số liệu so với tháng trước (m/m) cho
> riêng tháng 9**."*

Ngữ cảnh ghi rõ: *"Doanh số bán lẻ hàng hóa và dịch vụ (NSA) **tăng 2,4% (m/m)**
… vào tháng 9 năm 2023"* và *"doanh số bán hàng hóa **tăng 2,1% (m/m)** … trong
tháng 9"*.

Mệnh đề meta này **sai**. Nhưng luật 2 của rubric v2 viết:

> *"Một câu nói rằng ngữ cảnh THIẾU thông tin là NO_CLAIM… Nó không sai và cũng
> không đúng — nó không phải một mệnh đề để kiểm."*

Lý lẽ ấy đúng cho lời khai **đúng**, và hỏng cho lời khai **sai**. Ở đây câu ấy
vừa sai vừa kiểm được. Judge áp luật 2 đúng theo mặt chữ và cho nó `NO_CLAIM` —
tức là **miễn phí**, không vào tử số cũng không vào mẫu số. GLM cho
`CONTRADICTED`, và ở ca này GLM đúng.

Nghĩa là: **một model bịa ra "nguồn không nói gì về X" trong khi nguồn có nói về
X, hiện đang được chấm là vô hại.** Đây là một lớp ảo giác thật, ở đúng chỗ RAG
hay hỏng nhất — từ chối sai. Nó không thuộc `W5-04` để sửa (sửa rubric = tăng
version = mất toàn bộ 1664 phán quyết cache và mọi con số `W5-01` phải chạy lại),
nên ghi thành **`TD-65`**.

---

## 5. Nhánh khác họ: GLM **không** xác nhận — nó cho thấy luật khó nhất của rubric phụ thuộc model

| | Po | κ | κ CI 95% |
|---|---|---|---|
| GLM vs người — mẫu | 0,780 | 0,573 | [0,350 – 0,775] |
| GLM vs người — quần thể | 0,879 | **0,371** | [0,166 – 0,731] |
| GLM vs DeepSeek — quần thể | 0,903 | 0,424 | [0,219 – 0,769] |

κ = 0,371 nằm **dưới** ngưỡng 0,6, và bất đồng không rải đều:

```
NO_CLAIM   precision 1,000   recall 0,529   ← 8/17 ca NO_CLAIM bị GLM gọi là SUPPORTED
SUPPORTED  precision 0,771   recall 0,900
```

GLM về cơ bản **không thi hành luật 2**. Nó coi câu meta (*"Các nguồn chỉ đề cập
đến…"*, *"Tôi chỉ có thể trích dẫn những gì nguồn nói…"*) là mệnh đề bình thường
và thấy chúng khớp ngữ cảnh, nên gán `SUPPORTED`. Đây chính là lỗi mà rubric v1
đã mắc và `W5-01` đã phải sửa bằng cách thêm nhãn `NO_CLAIM` — chỉ khác là lần
này lỗi nằm ở model chứ không ở rubric.

**Hệ quả bằng số**: nếu đổi judge sang GLM, faithfulness đọc ra

```
0,9246  CI [0,8284 – 0,9928]     (so với 0,9877 hiện tại)
```

Cơ chế của 6,3 điểm ấy rất cụ thể: GLM gán `NOT_FOUND` cho **2** mệnh đề trong
tầng `SUPPORTED`, mà tầng ấy có trọng số 13,4 — `2 × 13,4 / 407 ≈ 6,6` điểm. Nói
cách khác, **hai bất đồng trên 30 mẫu, khuếch đại bởi thiết kế lấy mẫu, đủ để đưa
metric từ "qua ngưỡng thoải mái" xuống "qua ngưỡng 0,92 đúng 0,005"**.

Đó cũng là câu trả lời cho "50 mẫu phân giải được tới đâu": mỗi bất đồng trong
tầng `SUPPORTED` trị giá **3,3 điểm** trên metric quần thể. Đủ để khẳng định
`0,9877` vượt ngưỡng `0,92`; **không** đủ để phân biệt `0,9877` với `0,9905`.

Kết luận của nhánh này không phải "hai judge đồng ý nên số đáng tin". Nó là:
**số này gắn với `deepseek-v4-flash` + `judge-faithfulness@v2`, và cặp ấy phải đi
kèm mọi lần trích dẫn con số** — đúng như `bundle_version` đi kèm mọi kết quả
truy hồi. Ghi thành **`TD-66`**.

---

## 6. ⭐⭐ Trả lời câu `W5-03` cố tình để lại — và câu trả lời tệ hơn dự đoán

`JudgeConfig.reasoning` mang sẵn một cảnh báo do chính tôi viết ở `W5-03`:

> *"⚠️ Giới hạn của kết luận: 12 mẫu ấy do tôi soạn và đều có đáp án dứt khoát.
> Nó chứng minh suy luận là thừa trên ca dễ; nó **không** nói gì về ca khó.
> `W5-04` chấm 50 mẫu lấy từ câu trả lời thật — đó là chỗ phải hỏi lại câu này."*

Hỏi lại: cùng 50 mẫu, cùng rubric v2, cùng `max_tokens=512`, chỉ bật suy luận.

```
suy luận TẮT :  0/50 không đọc được · nhãn {SUPPORTED 30, NO_CLAIM 15, NOT_FOUND 5}
suy luận BẬT : 32/50 không đọc được · nhãn {SUPPORTED 18}
```

**Không phải "đắt hơn mà không tốt hơn". Là hỏng.** 64% phán quyết mất trắng, và
100% số còn lại là `SUPPORTED`. Lý do hỏng giống hệt `W4-06` và giống hệt probe
`W5-03`: chuỗi suy luận ăn hết `max_tokens`, `content` không bao giờ tới được
phần JSON.

Nhưng điều đáng sợ không nằm ở tỉ lệ hỏng. Nó nằm ở **cái gì bị hỏng**:

```
ngữ cảnh của ca BỊ MẤT   : trung vị 2952 ký tự
ngữ cảnh của ca ĐỌC ĐƯỢC : trung vị 1582 ký tự

nhãn tay của 32 ca BỊ MẤT  : NO_CLAIM 17 · SUPPORTED 12 · NOT_FOUND 2 · CONTRADICTED 1
nhãn tay của 18 ca CÒN LẠI : SUPPORTED 18
```

Ngữ cảnh dài ⇒ suy luận dài ⇒ chạm trần ⇒ mất phán quyết. Và ngữ cảnh dài chính
là ca khó. **Cả ba mệnh đề thất bại thật đều nằm trong nhóm bị mất.** Cơ chế cắt
bỏ có chọn lọc đúng những ca sẽ kéo điểm xuống.

Con số cuối cùng:

```
faithfulness với judge suy luận BẬT = 1,0000
```

**Một judge hỏng không cho điểm thấp. Nó cho điểm tuyệt đối.**

Đây là hệ quả trực tiếp — và không lường trước — của một quyết định đúng ở
`W5-03`: phán quyết không đọc được bị **loại** khỏi cả tử số lẫn mẫu số, vì tính
nó là thất bại sẽ quy lỗi của judge thành lỗi của hệ thống bị chấm. Quy tắc ấy
vẫn đúng. Nhưng nó chỉ an toàn khi việc "không đọc được" **độc lập** với nhãn.
Ở đây nó tương quan gần như hoàn hảo, và loại bỏ trở thành thiên lệch sống sót.

`W5-03` có ghi *"Tỉ lệ ấy cao thì kết luận đúng là 'phép đo này không dùng được'"*
— nhưng đó là một câu trong docstring, tức là trông chờ vào việc con người đọc
`n_unjudged`. Ở đây `n_unjudged = 32/50` và metric vẫn vui vẻ in ra `1,0000`.

➡️ **Yêu cầu cứng cho `W5-05`**: `gate.py` phải **FAIL** khi `n_unjudged / n`
vượt ngưỡng (đề xuất 5%), chứ không phải chỉ báo cáo nó. Một metric tính trên 36%
quần thể không phải là một metric. Ghi thành **`TD-67`**.

---

## 7. Cái đã kiểm được và cái không

**Kiểm được bằng máy:**

* Phán quyết judge cố định **trước** khi có nhãn tay — `sha256` file sealed
  (`6f0095d62558ef6e…`) nằm trong header file blind, `score` từ chối chạy nếu
  lệch.
* Mẫu bốc từ đúng tập phán quyết đã sinh ra `0,9877` — `sample` chạy ở
  `frozen_cache=True`, **50 hit / 0 miss / $0**. Trượt một lượt là lỗi, không
  phải một lời gọi mới.
* Câu hỏi gửi cho GLM **giống hệt** câu hỏi gốc: `build_questions` gọi thẳng
  `faithfulness_questions` của `generation_metrics`, có test so từng dict biến.
* Trọng số dựng lại đúng biên độ quần thể (§3).
* Mọi con số thống kê đối chiếu với giá trị tính tay trong `tests/unit/test_kappa.py`
  (ví dụ giáo khoa 20/5/10/15 ⇒ κ=0,40; nghịch lý 85/5/5/5 ⇒ Po=0,90 nhưng
  κ=0,444, PABAK=0,80).
* **17/17 phép tiêm đỏ**, không phép nào sống sót. Trong đó `C1` (bỏ trộn thứ tự)
  tái tạo đúng lỗi ở §2.

**Không kiểm được, và phải nói ra:**

1. **Người gán nhãn là tôi — một LLM, không phải một người độc lập.** Hash chứng
   minh phán quyết không bị sửa sau khi thấy nhãn tay; nó **không** chứng minh
   người gán nhãn không đọc trộm file sealed. Điều duy nhất tôi khai được là quy
   trình: đọc file mù, ghi nhãn, rồi mới chạy `score`. Và §2 cho thấy đúng loại
   rò rỉ này xảy ra được ngay cả khi không ai định gian lận.
2. **n = 50, một người chấm.** Không có κ giữa hai người, nên không biết bao
   nhiêu phần của bất đồng judge–người là do rubric mờ (§4 cho thấy 3/5 ca là
   đúng chỗ mờ ấy) chứ không do judge sai.
3. **Chỉ hiệu chỉnh `judge-faithfulness@v2`.** `judge-answer-relevancy@v1` nuôi
   refusal accuracy `0,9091` — một ngưỡng đang được coi là đã qua — và chưa có
   một mẫu nhãn tay nào. Công cụ đã sẵn (`--rubric relevancy`,
   `DEFAULT_ALLOCATION` đã khai); chỉ thiếu 50 nhãn. **`TD-68`**.
4. **Hai nhánh judge không đồng điều kiện suy luận** (§0) — GLM không tắt được.

---

## 8. Chạy lại

```bash
# 1. bốc mẫu — $0, không gọi mạng, trượt cache là lỗi
uv run python -m pipeline.eval.calibration --rubric faithfulness --seed 20260906 sample \
  --run plans/reports/runs/w5-answers-v1.jsonl --cache .cache/judge-w5.sqlite3 \
  --out data/eval/calibration/w5-04-faith.jsonl

# 2. (gán nhãn tay vào *-human.jsonl — chỉ đọc *-blind.jsonl)

# 3. judge khác họ trên đúng mẫu ấy
uv run python -m pipeline.eval.calibration --rubric faithfulness --cap-usd 0.2 cross \
  --run plans/reports/runs/w5-answers-v1.jsonl \
  --blind data/eval/calibration/w5-04-faith-blind.jsonl \
  --out plans/reports/runs/w5-04-faith-cross-glm.jsonl

# 4. ghép ba nguồn nhãn
uv run python -m pipeline.eval.calibration --rubric faithfulness score \
  --blind data/eval/calibration/w5-04-faith-blind.jsonl \
  --human data/eval/calibration/w5-04-faith-human.jsonl \
  --cross plans/reports/runs/w5-04-faith-cross-glm.jsonl \
  --out plans/reports/runs/w5-04-calibration.json
```

**Chi phí**: GLM 50 lời gọi `$0,0114` · nhánh suy luận bật 50 lời gọi `$0,0408`
(32 trong số đó mua về con số không) · lấy mẫu và chấm lại `$0`. Tổng
**`$0,0522`**.

**Hiện vật**: `plans/reports/runs/w5-04-calibration.json` ·
`w5-04-calibration-reasoning-arm.json` · `w5-04-faith-cross-glm.jsonl` ·
`w5-04-faith-arm-reasoning.jsonl` · `data/eval/calibration/w5-04-faith-{blind,sealed,human}.jsonl`

---

## 9. Nợ mới

| ID | Nợ |
|---|---|
| `TD-65` | Luật 2 rubric v2 không phân biệt lời khai *"nguồn im lặng"* **đúng** với **sai**; lời khai sai hiện được chấm `NO_CLAIM` (miễn phí). Trả cùng lúc với `TD-64` khi sửa prompt, vì cả hai đều buộc tăng version rubric và chạy lại toàn bộ cache. |
| `TD-66` | Mọi con số faithfulness phải mang theo `(judge_model, rubric_spec)` như một phần định danh, giống `bundle_version`. Đo được: đổi judge làm metric dịch 6,3–7,5 điểm. |
| `TD-67` | `gate.py` (`W5-05`) phải FAIL khi `n_unjudged / n` > 5%, không chỉ báo cáo. Judge hỏng cho điểm **1,0000**, không cho điểm thấp. |
| `TD-68` | `judge-answer-relevancy@v1` chưa hiệu chỉnh, trong khi nó nuôi refusal accuracy `0,9091` đang được tính là qua ngưỡng. Công cụ đã sẵn, thiếu 50 nhãn tay. |
