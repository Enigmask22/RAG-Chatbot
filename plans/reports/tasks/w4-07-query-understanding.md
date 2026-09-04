# `W4-07` — hiểu câu hỏi trước khi đi tìm, và ba dự đoán sai

> **Ngày:** 2026-09-04 · **Mã:** `serving/core/understanding.py` (mới),
> `serving/core/chat.py`, `serving/api/app.py`, `serving/db/models.py`,
> `alembic/versions/0003_message_query_plan.py`, `packages/rag_core/settings.py`
> **Dữ liệu:** `tests/fixtures/query_understanding_cases.jsonl` (27 case gán nhãn tay)
> **Test:** `tests/unit/test_query_understanding.py` (**33**),
> `tests/unit/test_chat_service.py` (+**8**), `tests/integration/test_chat_stream.py` (+**7**)
> **Bằng chứng:** `reports/probes/w4-07-language-directive.json`,
> `reports/probes/w4-07-real-run.json`
> **Chi phí API:** $0,0125 (8 câu × 2 nhánh + ba lần chạy thật)

## 0. Phạm vi — và một nửa của nó đã bị `TD-37` gỡ đi

Hạng mục này ban đầu có bốn việc. `TD-37` (03/09) đo được rằng việc thứ tư —
**định tuyến dense-vs-hybrid theo loại truy vấn** — đạt được bằng một tham số
cấu hình (`--rrf-weights 1 0.25`) với cùng mức lợi và **1/3,6** cái giá, nên nó
bị gỡ khỏi phạm vi. Còn lại ba việc, và chúng khác nhau về **bản chất**:

| việc | tín hiệu | cái giá khi sai |
|---|---|---|
| định tuyến `NO_RETRIEVAL`/`CLARIFY`/`RETRIEVE` | luật, miễn phí | bỏ truy hồi một câu hỏi thật |
| viết lại câu hỏi đa lượt | một lượt gọi LLM | truy hồi bằng một câu không ai hỏi |
| phát hiện ngôn ngữ | luật, miễn phí | ép model trả lời sai ngôn ngữ |

## 1. ⭐⭐ Bộ phân loại bất đối xứng — và vế thứ hai của nó ra đời từ một phép tiêm lỗi

Hai hướng sai không bằng giá nhau. Truy hồi **thừa** tốn ~800 ms và vài chunk vô
hại. Truy hồi **thiếu** cho ra một câu trả lời từ kiến thức nội tại của model,
không nguồn, giọng tự tin y hệt — đúng thứ cả hệ thống này tồn tại để tránh.

Nên luật không phải "điểm vượt ngưỡng" mà là **từ vựng đóng**: mọi token phải
nằm trong đó. Hình dạng ấy không có núm nào để vặn.

⚠️ Và bản đầu của nó **sai**, theo đúng hướng đắt tiền. Docstring tôi viết
khẳng định "một câu hỏi không bao giờ lọt qua được vì nó luôn chứa ít nhất một
từ ngoài từ vựng". Từ vựng khi ấy có `bạn`, `anh`, `chị`, `em`, `thầy`, `cô` —
để bắt `"chào bạn"`. Nhưng chúng **cũng là danh từ nội dung**:

```
classify("thầy cô")  -> no_retrieval   ❌
classify("chị em")   -> no_retrieval   ❌
```

Phép tiêm `N1` (nới trần độ dài 6 → 60) không làm đỏ test nào, và đi tìm *vì
sao* mới lộ ra chỗ này. Luật giờ có hai vế: mọi token trong từ vựng, **và** ít
nhất một token là `_SOCIAL_CORE` (lời chào thật). `"bạn"` một mình đi truy hồi;
`"chào bạn"` thì không.

💡 Bài học không phải về tiếng Việt: **một từ vựng đóng chỉ an toàn khi mọi từ
trong nó chỉ có một vai.**

## 2. Bộ 27 case — và vì sao con số 26/27 gần như không nói gì

Ba nhóm với ba tư cách khác nhau, và chỉ một nhóm là bằng chứng:

| nhóm | n | viết khi nào | route đúng |
|---|---|---|---|
| đặc tả | 18 | **trước** khi có một dòng luật | 18/18 |
| held-out | 7 | **sau** khi luật đóng băng, luật không sửa vì chúng | **6/7** |
| hồi quy | 2 | sau khi `N1` phơi ra lỗ ở §1 | 2/2 |

18 case đầu viết trước, nhưng chúng nằm trước mắt tôi trong lúc viết luật, nên
18/18 đo đúng một điều: **cài đặt khớp đặc tả**. Gọi nó là bằng chứng khái quát
là tự lừa. Con số duy nhất nói được gì về câu hỏi chưa từng thấy là **6/7**.

Ca hỏng held-out là `"who are you?"` → `clarify` thay vì `no_retrieval`. **Không
vá.** Cả hai nhánh đều không truy hồi và không gọi model sinh; khác biệt duy
nhất là người dùng nhận câu hỏi lại thay vì câu trả lời. Chữa đúng cần một nhánh
thứ tư ("câu hỏi về chính trợ lý"), và dựng một nhánh cho một ca là cách bộ phân
loại bắt đầu phình theo bộ test. Nó nằm nguyên trong fixture, có test ghim tên.

Độ chính xác được **ghim bằng con số tuyệt đối**, không phải ngưỡng sàn: một
ngưỡng `>= 0,9` cho cả hai chiều trôi qua không ai nhìn, kể cả chiều tăng — mà
tăng nghĩa là hành vi đã đổi ở những câu chưa ai xem lại nhãn.

### Lần chạy đầu: 17/18, và ca sai là một dấu lược

Trước mọi lần sửa: route 17/18, ngôn ngữ 17/18, quyết định viết lại 17/18. Ca
sai là `"thanks, that's all"`. Tách theo `[^\w]+` biến `that's` thành `that` +
`s`, và cái `s` mồ côi không thuộc từ vựng nào — nên câu bỗng có "từ nội dung".
Không phải lỗ hổng từ vựng: nó chạm **mọi** dạng rút gọn tiếng Anh.

## 3. ⭐⭐ Chỉ thị ngôn ngữ: 8/8 → 0/8, và tôi đoán sai cả hai đầu

`W4-06` thấy model trả lời tiếng Việt cho một câu hỏi tiếng Anh dù luật 4 của
`SYSTEM_PROMPT` nói ngược lại. Tôi ghi trước: chỉ thị tường minh sẽ *giảm* tỉ
lệ, từ khoảng 3/8 xuống 0–1/8.

Đo thật — 8 câu hỏi tiếng Anh, **cùng** bộ chunk đã truy hồi cho cả hai nhánh,
`deepseek-v4-flash`, `temp=0`, $0,0081:

| | trả lời sai ngôn ngữ |
|---|---|
| không chỉ thị (= hành vi `W4-06`) | **8/8** |
| có `"Answer in English."` cuối lượt người dùng | **0/8** |

Sai cả hai đầu. Tỉ lệ nền không phải "thỉnh thoảng" mà là **tất cả** — chuyện
của `W4-06` không phải xui, nó tất định. Và một dòng chữ vá được **toàn bộ**.

💡 Cơ chế nhiều khả năng không phải "model nghe lời": `SYSTEM_PROMPT` viết hoàn
toàn bằng tiếng Việt và phần lớn chunk cũng tiếng Việt, nên mặc định của prompt
kéo về tiếng Việt; chỉ thị thắng vì nó **gần nhất và tường minh nhất**. Đó là lý
do vị trí của nó là một quyết định: cuối **lượt người dùng**, không phải trong
`SYSTEM_PROMPT` — và cũng vì `W4-11` sắp băm hash prompt hệ thống, nên nhét một
chuỗi đổi theo từng lượt vào đó sẽ chẻ hash thành một bản cho mỗi ngôn ngữ.

⚠️ 8 câu, một model, một prompt. Đủ để **bật** nó, không phải một phát biểu về
việc model tuân chỉ dẫn nói chung. Nên phép **đo** ở lại: `detect_language()`
chạy trên cả câu hỏi lẫn câu trả lời, chênh lệch thành `done.language_mismatch`
— hôm nay 0/8, và nếu đổi model thì con số tự nói ra. Chỉ dẫn có thể hỏng; từ
đây chỗ hỏng đếm được.

### Vì sao không dùng `langdetect`/`fasttext`

Chúng trả một nhãn kèm điểm tin cậy cho **mọi** đầu vào, kể cả câu ba chữ —
và `langdetect` không tất định nếu không gieo hạt, thứ mà quy tắc "mọi cái trên
đường eval phải tái lập được" đã loại từ đầu. Bài học `TD-37` đúng nguyên văn:
một tín hiệu **tự tin và sai** tệ hơn một tín hiệu **thiếu**.

Nên hàm ở đây trả `"unknown"` khi không chắc, và `"unknown"` có hệ quả cụ thể:
**không sinh chỉ thị nào cả**. `"GDP per capita?"` là ca duy nhất lệch nhãn
trong 27 — và nó lệch vì bộ phát hiện *từ chối đoán*, nên fixture ghim chính sự
từ chối ấy.

⭐ Chỗ duy nhất ngôn ngữ là một **cơ chế** chứ không phải chỉ dẫn: câu hỏi lại
của `CLARIFY`. Text ấy do mã chọn từ một bảng, không do model sinh, nên nó không
thể sai ngôn ngữ — và song ngữ khi không phát hiện được.

## 4. ⭐⭐ Lần chạy thật đảo ngược một quyết định thiết kế

Thiết kế ban đầu: truy hồi chạy bằng câu **đã viết lại**, model xem câu **gốc**.
Lý lẽ nghe rất đúng — truy hồi cần một chuỗi tự đủ nghĩa để so vector, còn model
đã có lịch sử ở ngay trên và nên thấy đúng chữ người dùng gõ.

Chạy thật, `"cái đó thì sao?"` sau một lượt về di cư lao động:

```
truy hồi   5 chunk, đúng chủ đề, rerank 3,09 / 2,07 / 0,72 …
trả lời    "Xin lỗi, tôi không đủ thông tin để trả lời câu hỏi 'cái đó thì
            sao?' vì câu hỏi không nêu rõ 'cái đó' là gì."
```

Lịch sử **có** trong prompt (2.564 token). Model vẫn áp luật 3 lên chuỗi mơ hồ
trước mắt nó. Đưa **mỗi** bản viết lại thì câu trả lời lại nói về một câu hỏi
người dùng không gõ. Nên: **cả hai**, gốc trước.

```
CÂU HỎI: cái đó thì sao?
(Hiểu đầy đủ theo hội thoại: "What does the report say about labour migration?")
```

Sau khi sửa, cùng lượt ấy trả lời đầy đủ bằng tiếng Việt kèm `[1]`, 573 token.

💡 Đây là loại lỗi không một test đơn vị nào bắt được, vì cả hai phiên bản đều
"đúng" theo mọi thứ đo được ở tầng mã. Nó chỉ hiện ra khi có một model thật đọc
prompt thật.

## 5. Ba quyết định nhỏ hơn

* **Viết lại hỏng không được làm hỏng lượt.** Timeout hoặc `LLMError` ⇒ rơi về
  câu gốc, vẫn truy hồi, vẫn trả lời. Biến một bước *tăng chất lượng* thành một
  điểm chết mới là đổi ngược hướng.
  ⚠️ `wait_for` huỷ được cái **chờ**, không huỷ được cái **thread**: quá hạn thì
  người dùng đi tiếp còn thread kia vẫn chạy nốt và vẫn bị tính tiền.
* **`_clean_rewrite` chặn ba thứ**: rỗng, nhiều dòng, và **dài bất thường**
  (> 4× + 120 ký tự). Cái thứ ba là kiểu hỏng đắt nhất vì nó không tự biểu hiện:
  một câu hỏi bị nhét thêm chi tiết vẫn truy hồi ra chunk trông hợp lý.
* **`tenant_filter()` gọi trước khi rẽ nhánh**, kể cả trên lượt không truy hồi.
  Nó không chỉ *thu hẹp* filter, nó **từ chối** filter trỏ sang tenant khác — bỏ
  nó ở nhánh `no_retrieval` thì cùng một request nhận `403` hay `200` tuỳ người
  dùng có chào hỏi hay không, tức một hành vi bảo mật phụ thuộc vào bộ phân loại
  câu hỏi. Có test tích hợp ghim.

## 6. Migration `0003` — hai cột trên message của người dùng

`route` (có `CHECK` ba giá trị) và `rewritten_query` (text tự do, `NULL` = truy
hồi chạy đúng bằng `content`).

Không có `rewritten_query` thì một lượt đã viết lại là **không giải thích nổi**:
`content` ghi "cái đó thì sao?" còn `citations` nói về di cư lao động, và không
có gì trong database nối hai thứ ấy. Đúng lý lẽ đã đưa `model`/`finish_reason`
vào `0002` — một thuộc tính chỉ sống trong log là một thuộc tính biến mất theo
chính sách giữ log, và câu hỏi cần nó luôn được đặt muộn hơn thế.

**Không backfill.** Hàng cũ không đi qua bộ phân loại nào; điền `retrieve` cho
chúng thì đúng về hành vi nhưng nói dối về nguồn gốc, và một báo cáo "tỉ lệ lượt
bỏ truy hồi" sẽ tính cả những lượt chưa từng có lựa chọn.

## 7. Tiêm lỗi — 19 phép, 17 đỏ ngay, 2 phơi ra lỗ thật

| nhóm | phép | kết quả |
|---|---|---|
| định tuyến | `all`→`any`, bỏ mẫu tỉnh lược, "cứ có lịch sử là viết lại", bỏ chuẩn hoá dấu lược, bỏ vế "phải có lời chào thật", xưng hô quay lại làm stopword | 6 ✅ |
| ngôn ngữ | đoán `en` thay vì từ chối, `unknown` cũng tính là lệch, đọc ngôn ngữ từ bản viết lại | 3 ✅ |
| viết lại | bỏ chặn độ dài, lỗi LLM làm hỏng lượt, cửa sổ lịch sử cắt nhầm đầu, bản y hệt vẫn tính là đã viết lại | 4 ✅ |
| tích hợp vào lượt | `clarify` không dừng lại, `no_retrieval` dùng `SYSTEM_PROMPT`, `tenant_filter` vào trong nhánh, model xem bản viết lại, lưu bản viết lại vào `content` | 5 ✅ |
| **sống sót** | **trần số từ xã giao 6 → 60** | ⚠️ xanh |
| **sống sót** | **bỏ `\n\n` trước chỉ thị ngôn ngữ** | ⚠️ xanh |

⭐⭐ **Phép sống sót thứ nhất dẫn tới lỗi ở §1.** Không viết được test cho nó
nghĩa là phải bịa ra một đầu vào không ai gửi — dấu hiệu cái trần ấy mã hoá một
phỏng đoán chứ không phải một hành vi. Bỏ trần, và đi tìm *vì sao nó vô hại* mới
thấy lỗ thật nằm ở **từ vựng**, không ở độ dài. Sau khi sửa, phép tiêm thay thế
(bỏ vế "phải có lời chào thật") đỏ ngay.

⭐ **Phép thứ hai**: bỏ hai ký tự xuống dòng cho ra
`"...poverty line?Answer in English."` — chỉ thị dính liền câu hỏi, tức nó trở
thành một phần của chính câu hỏi. Test cũ dùng `endswith` sau `rstrip()` nên
xanh. Đã ghim bằng so sánh chuỗi đầy đủ.

## 8. Dự đoán — 4/5 sai, và cái đúng duy nhất là cái rẻ nhất

| | dự đoán | thực tế |
|---|---|---|
| P1 | route 16/18, hỏng ở `"còn Lào?"` và `"???"` | ❌ **17/18**, và cả hai ca dự đoán đều **đúng**; ca hỏng là dấu lược, thứ tôi không nghĩ tới |
| P2 | ngôn ngữ 16/18, hỏng ở câu ASCII ngắn | ⚠️ đúng **ca**, sai **số** (17/18) |
| P3 | chỉ thị giảm lệch 3/8 → 0–1/8 | ❌ **8/8 → 0/8** — sai cả tỉ lệ nền lẫn mức cải thiện |
| P4 | viết lại tốn 400–700 ms | ❌ đo được **1.019 ms** |
| P5 | viết lại **thừa** là kiểu hỏng chính | ✅ xác nhận: `"cái đó thì sao?"` bị thu hẹp thành *"…the household registration system as an administrative barrier to migrants?"* — một chủ đề con lấy từ câu **trả lời** trước, không phải câu hỏi trước |

P1 sai vì lý do đáng ghi: khi viết dự đoán tôi chưa quyết định coi mẫu tỉnh lược
(`"còn X"`, `"what about X"`) là từ chỉ trỏ, và chưa quyết định định nghĩa
`CLARIFY` bằng **số từ nội dung**. Hai quyết định ấy làm cả hai ca tôi dự đoán
hỏng trở nên đúng — tức dự đoán đã lỗi thời trước khi được kiểm.

## 9. Còn lại

* ⚠️⚠️ **`temp=0` ở DeepSeek không tất định.** Ba lần chạy cùng một prompt cho
  365 / 309 / 573 token completion, và bước viết lại cho **hai chuỗi khác nhau**
  cho cùng một hội thoại. Trước `W4-07` điều đó chỉ ảnh hưởng câu chữ; từ nay nó
  ảnh hưởng **chunk nào được truy hồi**, vì đầu ra của model là đầu vào của
  truy hồi. Một hội thoại phát lại không còn tái lập được. → nợ mới.
* **Không có cache cho bước viết lại.** Cùng một cặp (lịch sử, câu hỏi) gọi lại
  là gọi lại — `W4-10` (Redis semantic cache) là chỗ đúng, nhưng khoá phải gồm
  lịch sử chứ không chỉ câu hỏi.
* **`"who are you?"` → `clarify`** (§2). Cần một nhánh thứ tư mới chữa đúng.
* **Cửa sổ lịch sử của bước viết lại cắt theo số lượt (6), không theo token** —
  cùng món nợ của `MAX_HISTORY_MESSAGES` ở `W4-06`, giờ có ở hai chỗ.
* **`REWRITE_SYSTEM_PROMPT` là hằng số thứ ba trong mã.** `SYSTEM_PROMPT`,
  `NO_RETRIEVAL_SYSTEM_PROMPT` và nó phải cùng chuyển sang registry ở `W4-11`.
* **Lịch sử hội thoại là bề mặt prompt injection thứ hai.** Bước viết lại đọc cả
  lượt người dùng lẫn lượt trợ lý, và đầu ra của nó đi thẳng vào truy hồi. Bán
  kính nổ hẹp (một truy vấn, không phải một hành động), nhưng nó là bề mặt mới
  và `W4-12` phải tính cả nó, không chỉ nội dung tài liệu.
