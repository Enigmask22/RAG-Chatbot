# `W5-05` — Gate phát hành: chỗ duy nhất nói KHÔNG, và ba lỗi nó tìm ra ở chính mình

*2026-09-05 · `pipeline/eval/gate.py`, `configs/eval/gate.yaml`,
`pipeline/bundle/build_bundle.py`, `tests/unit/test_gate.py` (37 test)*

**Kết quả chạy thật trên bundle `0.2.1`:**

```
make gate BUNDLE=0.2.1                → INCOMPARABLE, exit 2
make gate BUNDLE=0.2.1 --no-champion  → FAIL,          exit 1
    ❌ citation_accuracy   0,8308 < 0,85       (TD-64)
    ❌ p95_end_to_end_ms   4706,5 > 3500
```

Hai lần đỏ, hai lý do khác nhau, và cả hai đều đúng.

---

## 0. Hạng mục này bắt đầu bằng việc **không có gì để gate**

DoD nói `make gate BUNDLE=x`. Nhưng cả hai bundle trên đĩa đều có
`generation_metrics: {}` và `judge: null`. Faithfulness `0,9877` của `W5-01`
không nằm trong bất kỳ artifact nào mà gate đọc được.

Lý do là một vòng lặp có thật trong kiến trúc: **metric truy hồi đo offline
trước khi deploy; metric tầng sinh đo qua HTTP trên một server đang chạy một
bundle**. Bundle được ký xong mới có số của chính nó. Nó không bao giờ mang được
số tầng sinh của mình.

Lối ra: đúc một **bản patch** — `0.2.1` — thành phần y hệt `0.2.0`, thêm phép
đo. Semver nói đúng chuyện đã xảy ra: cùng một hệ thống, nhiều phép đo hơn.

Và điều kiện để lối ra ấy không thành lỗ hổng: `answer_run` ghi
`bundle_versions: ["0.2.0"]`, nên `build_bundle` **nạp bundle ấy lên và đòi phần
đường truy hồi trùng khít**. Không có phép kiểm này thì gắn số của hệ thống này
lên hệ thống khác là một thao tác chép file.

---

## 1. ⭐⭐ "Không so được" là phán quyết **thứ ba**, không phải một kiểu FAIL

Cám dỗ là hai trạng thái. Nhưng ba câu dưới đây đòi ba hành động khác nhau:

| phán quyết | nghĩa | việc phải làm | exit |
|---|---|---|---|
| `PASS` | đo được, đạt | — | 0 |
| `FAIL` | đo được, tệ hơn | sửa **hệ thống** | 1 |
| `INCOMPARABLE` | hai con số không đặt cạnh nhau được | sửa **phép đo** | 2 |

Gộp cái thứ ba vào `FAIL` đẩy người ta đi tối ưu một hệ thống không có gì sai.
Gộp vào `PASS` thì tệ hơn nhiều: nó thả một bundle chưa ai so với gì.

Ba exit code khác nhau để CI phân biệt được mà không phải parse stdout.

Và khi `comparability` hỏng, gate **vẫn chạy nốt** ba nhóm còn lại. Người đọc
cần thấy cả hai thứ cùng lúc chứ không phải sửa một lỗi rồi mới biết còn lỗi gì:

```
GATE INCOMPARABLE · bundle 0.2.1
  ❌ [comparability] evaluated_with_generator: 'deepseek-v4-flash' ≠ 'deepseek-chat@2026-09'
  ✅ [absolute] ndcg@10: 0.7079 ≥ 0.6
  ❌ [absolute] citation_accuracy: 0.8308 < 0.85     ← vẫn thấy được
```

---

## 2. ⭐⭐ Trường sinh ra để bảo đảm danh tính đang mang một bí danh

`evaluated_with_generator` có từ `W4-01`, và docstring của nó nói đúng lý do tồn
tại: *"Gate chỉ được so like-for-like, và nó chỉ làm được thế nếu trường này
luôn có mặt — nên nó không có mặc định, kể cả chuỗi rỗng."*

Cả hai bundle đang có đều ghi:

```json
"evaluated_with_generator": "deepseek-chat@2026-09"
```

`DEEPSEEK_ALIASES` trong `rag_core.llm` — viết từ `W1-10`, đo lại ở `W5-03` —
nói rằng `deepseek-chat` là **con trỏ phía server**, hiện trỏ tới
`deepseek-v4-flash`, cùng đích với `deepseek-reasoner`.

Nghĩa là hai chuỗi bằng nhau **không** chứng minh hai lần đo dùng cùng model. Cái
nhãn `@2026-09` làm nó trông có mốc thời gian, nhưng nó vẫn là con trỏ. Đây đúng
là lý do quy tắc cứng #1 cấm OpenRouter preset — chỉ kín đáo hơn, và lần này nó
lọt vào chính cái trường được dựng để chống nó.

**Vì sao nó lọt**: trường ấy được **gõ tay** ở tầng CLI (`--generator`). Một
trường bảo đảm danh tính mà được gõ tay thì nó ổn định đúng bằng trí nhớ của
người gõ.

Hai bản vá, và bản thứ hai mới là bản thật:

1. Gate từ chối mọi danh tính là bí danh hoặc preset (`reject_alias_identity`),
   so trên **phần slug trước `@`**.
2. `build_bundle --generation-run` **đọc** model từ `models` của lần chạy —
   model đã thật sự phục vụ 242 request — thay vì nhận một chuỗi khai.

`--generator` vẫn còn, nhưng khi có `--generation-run` thì nó bị ghi đè bởi sự
thật đo được. `0.2.1` mang `deepseek-v4-flash`.

⚠️ Hệ quả: `0.2.1` **không so được** với champion `0.2.0`, và đó là kết luận
đúng chứ không phải bất tiện. Con số của `0.2.0` được sinh dưới một danh tính
không kiểm lại được. `TD-70`.

---

## 3. ⭐⭐ Kiểm **tính hợp lệ của phép đo** trước khi so với ngưỡng

`W5-04` đã trả tiền để học điều này: **một judge hỏng không cho điểm thấp — nó
cho điểm tuyệt đối.** Bật suy luận làm mất 32/50 phán quyết vì chuỗi suy luận ăn
hết `max_tokens`; ca bị mất có ngữ cảnh dài gần gấp đôi nên toàn bộ mệnh đề thất
bại nằm trong nhóm bị loại, và 18 ca sống sót đều `SUPPORTED` ⇒ faithfulness
`1,0000`.

Một gate chỉ so `faithfulness >= 0,92` sẽ **thả lần chạy ấy với điểm cao nhất có
thể**. Nên nhóm `validity` chạy **trước** `absolute`:

```python
def test_the_reasoning_arm_of_w5_04_is_caught_despite_perfect_scores():
    broken = bundle(generation_metrics={"faithfulness": 1.0},
                    unjudged_rate={"faithfulness": 0.64})
    verdict = evaluate_gate(broken, None, limits())
    assert outcome_of(verdict, "absolute", "faithfulness") is RuleOutcome.PASS   # ← vẫn qua
    assert outcome_of(verdict, "validity", "chưa chấm được · faithfulness") is RuleOutcome.FAIL
    assert verdict.status is GateStatus.FAIL
```

Ngưỡng `5%` chọn theo nguyên tắc "đủ chặt để bắt chế độ hỏng đã đo": nhánh hỏng
cho 64%, nhánh đang dùng cho 0%. Bất kỳ số nào trong `(0; 0,64)` đều bắt được.

Và **không khai `unjudged_rate` thì đỏ**, không phải bỏ qua. Lý lẽ: chế độ hỏng
đã đo lệch về phía điểm **cao**, nên "không biết" không được hưởng lợi thế nghi
ngờ. Mẫu số cũng phải đúng — là **số câu đã hỏi** (`n + n_unjudged +
n_not_a_claim`), không phải số câu chấm được; dùng mẫu số sau thì tỉ lệ tự nhỏ đi
đúng ở lần chạy hỏng nặng nhất.

`TD-67` trả xong.

---

## 4. ⭐⭐ Gate cho qua một hệ thống vượt ngân sách 34% — vì hai con số cùng tên

Lần chạy đầu tiên:

```
✅ [absolute] p95_latency_ms: 759.0347 ≤ 3500.0
```

Xanh. Và sai. `EvalReport.p95_latency_ms` được điền từ báo cáo eval **truy hồi**
— embed câu hỏi + tìm trong Qdrant + rerank. Ngân sách `3500 ms` của plan là
**end-to-end**. `W4-13` đã đo end-to-end là `5312 ms` và ghi rõ KHÔNG ĐẠT.

Gate vừa lấy một con số nhỏ hơn 7 lần, so với ngân sách của con số kia, rồi cho
qua. Không có gì báo, vì cả hai đều tên là "p95 latency".

Đây là cùng một họ lỗi với `citation_coverage` ở `W5-02` và `uncited_grounding`
ở `W5-01`: **mẫu số là toàn bộ câu chuyện**, và một cái tên chung làm hai đại
lượng khác nhau trông như một.

Sửa ba chỗ:

* `EvalReport.p95_end_to_end_ms` — trường mới, có tài liệu; `p95_latency_ms` giữ
  tên cũ (manifest đã ký) nhưng docstring nói rõ nó là truy hồi thuần.
* `generation_metrics` giờ tính khối `latency_ms` từ `wall_ms` của **242 request
  thật qua HTTP** — tốt hơn hẳn smoke test vài câu của `W4-13`:

  ```
  n=242  mean=2776  p50=2549  p95=4706  p99=5953  max=6372   (ms, kể cả 1 lượt trúng cache)
  ```
* `gate.yaml` chuyển ngưỡng `3500` sang `p95_end_to_end_ms`.

Kết quả: gate đỏ, đúng như thực tế. Và con số end-to-end của dự án từ nay là
**4706 ms trên toàn golden set**, thay cho `5312 ms` đo trên vài câu.

---

## 5. Danh tính giám khảo là một **bộ ba** (`TD-66`)

`W5-04` đo: giữ nguyên hệ thống, nguyên rubric, đổi mỗi model judge ⇒ `0,9246`
(GLM) hoặc `1,0000` (suy luận bật) thay vì `0,9877`. `W5-01` đo chiều còn lại:
cùng model, rubric v1→v2 đưa `uncited_grounding` từ `0,427` lên `0,856`.

Nên `judge_identity` là `(model, rubrics, reasoning)`, và cả ba đều vào
`comparability`:

```
'deepseek-v4-flash|judge-answer-relevancy@v1,judge-faithfulness@v2|reasoning=false'
```

Một chi tiết nhỏ có chủ đích: `reasoning=None` (bundle cũ chưa khai) in ra `?`,
**không** in ra `false`. Đọc "chưa khai" thành "đã tắt" là khai hộ một điều chưa
ai đo — và phép tiêm `G7` chứng minh có test giữ.

κ cũng không được gõ tay: `build_bundle --calibration` đọc thẳng
`agreement.judge_vs_human.population.kappa` từ báo cáo `W5-04`, và **từ chối**
nếu file hiệu chỉnh đo trên rubric hoặc model khác.

---

## 6. Ngưỡng nằm trong YAML, LUẬT nằm trong mã

`gate.yaml` chọn được con số. Nó **không** chọn được có kiểm hay không — không
có cờ nào tắt `comparability`, không có danh sách miễn trừ metric. Cái cờ ấy sẽ
được bật vào đúng ngày người ta cần nó nhất.

Mỗi ngưỡng mang một trường `why` và nó **bắt buộc** (có test):

```yaml
citation_accuracy:
  min: 0.85
  why: >-
    Ngưỡng gốc của plan, đo ở cấp quote (`W5-02` chọn quote_level làm gate_metric
    vì nó tất định). Hiện 0,8308 — `TD-64`. Cố ý KHÔNG hạ xuống cho vừa: gate tồn
    tại để nói ra chỗ chưa đạt.
```

`why` đi thẳng vào báo cáo HTML, nên người đọc kết quả FAIL thấy lý lẽ chứ không
phải một con số trần trụi. Một con số không có lý lẽ là một con số sẽ bị hạ
xuống vào lúc 11 giờ đêm để cho CI xanh.

HTML tự chứa hoàn toàn — không CDN, không font ngoài (có test cấm mọi `http`
trong output): báo cáo gate phải mở được trên một máy không mạng, ba tháng sau,
từ một artifact CI.

---

## 7. Vài chốt nhỏ nhưng đáng

**Champion là bản cao nhất *thấp hơn* ứng viên**, không phải "bản mới nhất". Ứng
viên thường **chính là** bản mới nhất, nên "mới nhất" sẽ làm nó tự so với chính
mình và mọi luật hồi quy đều qua — một gate luôn xanh. Phép tiêm `G14` (đổi `<`
thành `<=`) đỏ.

**Metric trùng tên ở hai bảng là lỗi**, không phải chuyện bên nào thắng. Nếu
`retrieval_metrics` và `generation_metrics` cùng có `faithfulness` thì không
quyết được con số nào bị gate.

**Metric champion có mà ứng viên thiếu là FAIL**, không phải bỏ qua — đó đúng là
cách một lần eval hỏng đi qua gate mà trông như không có gì xảy ra. Hạ nó xuống
`SKIP` được, nhưng chỉ bằng cách sửa `require_metrics_present` và commit.

**Chỉ metric do judge chấm mới có `unjudged_rate`.** Gán `0,0` cho
`citation_validity` (tất định) là khai một phép kiểm chưa từng chạy.

---

## 8. Kiểm chứng

**Tiêm lỗi: 18/18 đỏ, không phép nào sống sót.** Mỗi phép biến gate thành một
thủ tục trang trí ở đúng một chỗ:

| | phép tiêm | bắt bởi |
|---|---|---|
| `G1` | INCOMPARABLE gộp vào PASS | `test_a_different_generator_blocks_the_comparison` |
| `G2` | INCOMPARABLE dùng chung exit code với FAIL | `test_incomparable_has_its_own_exit_code` |
| `G3`/`G4`/`G5` | không bắt bí danh / không bỏ đuôi `@` / không bắt preset | `TestAliasIdentity` |
| `G6`/`G7` | danh tính judge chỉ gồm model / `None` đọc thành `false` | `TestJudgeIdentity` |
| `G8`/`G9`/`G10` | bỏ qua `unjudged_rate` / ngưỡng luôn đúng / judge chưa hiệu chỉnh vẫn qua | `TestValidityBeforeQuality` |
| `G11`/`G12`/`G15` | thiếu metric coi như qua / metric biến mất thành SKIP / trùng tên ghi đè | `TestMissingMetrics` |
| `G13` | hồi quy so ngược dấu | `test_a_quality_regression_is_fail_not_incomparable` |
| `G14` | champion là bản mới nhất | `test_champion_is_the_highest_version_below_the_candidate` |
| `B1` | không kiểm đường truy hồi của lần chạy tầng sinh | `test_numbers_measured_on_a_different_retrieval_path_are_refused` |
| `B2` | mẫu số `unjudged` là số câu chấm được | `test_unjudged_rate_denominator_is_questions_asked` |
| `B3` | nhận κ của rubric khác | `test_a_calibration_of_a_different_rubric_is_refused` |

**`G5` — hạng mục gate tuần 5** — có hai dòng đã trả:

* *"Gate **từ chối** so sánh 2 bundle khác `evaluated_with_generator` (có test
  chứng minh)"* → `TestIncomparableIsNotAKindOfFail`, và chứng minh trên **dữ
  liệu thật** chứ không chỉ fixture: `make gate BUNDLE=0.2.1` trả `INCOMPARABLE`.
* *"Có `reports/tasks/judge-calibration.md` với kappa ≥ 0.6"* → `0,737`, và giờ
  κ ấy nằm **trong manifest** (`eval.judge.kappa_vs_human`) chứ không chỉ trong
  một file markdown.

**Chạy lại**

```bash
# 1. gắn phép đo tầng sinh vào một bản patch của hệ thống đã đo
uv run python -m pipeline.bundle.build_bundle \
  --index-config configs/indexing/bgem3-contextual.yaml \
  --index-report plans/reports/probes/index-bgem3-contextual.json \
  --eval-run plans/reports/runs/bgem3-ctx-rr-c50-w025-retrieval.json \
  --generation-run plans/reports/runs/w5-answers-v1-generation.json \
  --calibration plans/reports/runs/w5-04-calibration.json \
  --gen-max-tokens 1024 --gen-temperature 0.0 --version 0.2.1

# 2. gate
make gate BUNDLE=0.2.1                                  # INCOMPARABLE, exit 2
uv run python -m pipeline.eval.gate --bundle 0.2.1 --no-champion   # FAIL, exit 1
```

**Chi phí**: `$0`. Tái lập báo cáo tầng sinh chạy ở `frozen_cache` (947 hit / 0
miss); không có lời gọi mạng nào trong cả hạng mục.

**Hiện vật**: `bundles/rag-bundle-v0.2.1/manifest.json` ·
`plans/reports/runs/gate-0.2.1.{html,json}` ·
`plans/reports/runs/gate-0.2.1-no-champion.{html,json}`

---

## 9. Nợ mới

| ID | Nợ |
|---|---|
| `TD-69` | **`answer_run` không ghi lại `max_tokens`/`temperature` của tầng sinh**, nên `build_bundle` phải nhận chúng qua cờ CLI — tức là một lời khai, không phải một phép đo. Trả bằng cách thêm hai trường vào header `AnswerRun` rồi đổi cờ thành phép đối chiếu. |
| `TD-70` | **`0.1.0` và `0.2.0` mang bí danh `deepseek-chat@2026-09`** trong `evaluated_with_generator`, nên không bundle nào so được với chúng. Bundle bất biến nên không sửa được; hoặc chấp nhận `0.2.1` là champion đầu tiên hợp lệ, hoặc đúc lại `0.1.1`/`0.2.2` với slug thật nếu cần chuỗi lịch sử liền mạch. |
| `TD-71` | **Gate chưa ghi phán quyết ngược vào `RagBundle.gate`.** `GateRecord` đã có sẵn từ `W4-01` (`status` + `champion_compared` + `report`) nhưng bundle bất biến và đã ký, nên ghi vào là đúc một bản mới. Cần quyết: gate ký lại bundle, hay `GateRecord` chuyển thành một artifact cạnh bundle. Chặn `W5-10` (auto-PR promote). |
