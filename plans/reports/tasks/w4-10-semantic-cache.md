# `W4-10` — Redis semantic cache

> 2026-09-04 · Serving Plane · DoD: query gần giống → cache hit, có TTL,
> invalidate khi đổi bundle · Test: `tests/integration/test_semantic_cache.py`
> (hit paraphrase, miss khác chủ đề, TTL) + `tests/unit/test_semantic_cache.py`
> · Evidence: `probes/w4-10-cosine-threshold.json` + `probes/w4-10-hitrate.json`.

## 0. Câu hỏi thật của hạng mục

"Ngưỡng cosine ~0.95" trong plan là một con số **đề xuất**, chưa ai đo trên
model đang dùng. Và ranh giới nguy hiểm của một semantic cache không nằm giữa
"paraphrase" và "khác chủ đề" — hai vùng đó cách nhau rất xa. Nó nằm giữa
paraphrase và **câu gần giống nhưng đổi đáp án**: "GDP nửa đầu 2025?" vs "GDP
nửa đầu 2024?" chỉ khác một chữ số, và trả câu trả lời của năm này cho câu hỏi
của năm kia là một câu trả lời *sai với citation đã verify* — tệ hơn mọi cache
miss cộng lại.

## 1. Dự đoán ghi trước khi đo — chấm: 2/3, và cái sai đổi thiết kế

- **P1 ✓** — paraphrase VI trên BGE-M3 phần lớn nằm 0,78–0,98 (đo: min 0,7827
  · p50 0,8717 · max 0,9757); ngưỡng 0,95 chỉ bắt được 2/10.
- **P2 ✗ một nửa** — bẫy đổi đáp án KHÔNG vượt 0,95 (max 0,9410) như dự đoán;
  nhưng phát hiện đo được còn xấu hơn dự đoán theo một chiều khác: **hai phân
  bố paraphrase và bẫy chồng lên nhau gần hết** (p50: 0,8717 vs 0,8659) — không
  tồn tại ngưỡng nào tách được chúng. Hạ ngưỡng để ăn thêm paraphrase là ăn
  luôn câu trả lời sai.
- **P3 ✓** — khác chủ đề max 0,4033; vùng 0,80–0,88 thuộc về... cả paraphrase
  lẫn bẫy (xem P2), nên biên an toàn chỉ rộng ở phía "khác chủ đề".

Ba bẫy cao nhất nói rõ vì sao không có ngưỡng cứu được: **"Thu ngân sách" vs
"Chi ngân sách" 0,9410** (không khác một chữ số nào), "đăng ký vs giải ngân"
0,9279, "7,5% vs 5,7%" 0,9112.

## 2. Ba quyết định, cả ba từ số đo

1. **Ngưỡng 0,96** — trên bẫy cao nhất đo được một biên 0,019. Giá: chỉ 1/10
   paraphrase thật vượt (cặp 0,9757; cặp 0,9506 hụt). Cache này **bảo thủ có
   chủ đích**: giá trị chính là câu hỏi lặp nguyên văn (demo, UI reload).
2. **Hàng rào token chữ số** — multiset token mang chữ số của hai câu phải
   trùng nhau, tất định, đứng SAU cosine: `7,5%` vs `5,7%` bị chặn kể cả khi
   vector trùng hệt. ("Thu vs Chi" thì hàng rào này không cứu — ngưỡng 0,96 mới
   là thứ đứng chắn, và đó là lý do không được hạ nó.)
3. **Namespace = `semcache:{tenant}:{bundle_version}`** — "invalidate khi đổi
   bundle" là thuộc tính của khoá, không phải một lệnh xoá: bundle mới nhìn vào
   namespace trống. Tenant trong khoá vì cache hit xuyên tenant là rò dữ liệu.

Kiến trúc: `serving/core/semantic_cache.py`, Redis 7 thuần (HASH + TTL +
trim LRU-theo-ts ở trần 128 entry, so cosine bằng numpy phía client — 128
entry × 1024 chiều là bài toán quá nhỏ cho một vector DB). Tra cache ở
`prepare()` **trước truy hồi** (hit tiết kiệm cả ~800 ms embed+rerank lẫn lượt
model); ghi cache chạy nền sau khung `done` như đường ghi Postgres. **Mọi lỗi
hạ tầng suy giảm thành miss** — Redis chết, entry vỡ, lệch chiều: cache không
bao giờ được phép là lý do `/chat` trả lỗi.

Điều kiện chạm cache (`cache_eligible`, hàm thuần): lượt **đầu** hội thoại, tự
đủ nghĩa, có truy hồi. Câu hỏi giữa hội thoại mang nghĩa từ lịch sử — cùng câu
chữ giữa hai hội thoại không phải cùng câu hỏi. Khung `meta.cache` khi hit mang
`matched_question` + `similarity`: một hit sai phải truy được từ **client**,
không phải chỉ từ log.

## 3. Số đo DoD: 100 query + hai bộ cặp

| phép đo | kết quả |
|---|---|
| Giao thông Zipf 100 query (23 câu phân biệt, 77 lượt lặp) | **hit 77/100**; trên câu lặp: **77/77** |
| Độ trễ lookup (embed + Redis + so khớp) | p50 **15,9 ms** · p95 **18,8 ms** |
| Paraphrase thuần (ghi A, tra B, 10 cặp) | **1/10** hit — đúng cái giá của 0,96 |
| Bẫy đổi đáp án (ghi A, tra B, 10 cặp) | **0/10** hit — không câu trả lời sai nào |

Một hit tiết kiệm ~800 ms truy hồi + ~900 ms model + toàn bộ token. Chi phí
của một miss: một lần embed thừa (~16 ms, retriever embed lại sau đó) — chấp
nhận thay vì refactor đường embed-một-lần; con số nằm cạnh 800 ms của lượt đầy.

## 4. Tiêm lỗi: 6 phép, 1 sống sót — và nó chỉ ra một ca chưa nghĩ tới

S1 (bỏ hàng rào chữ số) · S2 (bỏ ngưỡng) · S3 (namespace mất bundle_version) ·
S4 (store quên chuẩn hoá) · S5 (lịch sử hết loại trừ cache): **đỏ đúng test
chủ đích**.

⭐ **S6 (bỏ điều kiện `finish_reason == "stop"` khi ghi) XANH** — dự đoán sai.
Test "stream hỏng không được cache" đi qua nhánh `except`, nơi code ghi cache
không tồn tại; điều kiện ấy thật ra đang gác **`finish_reason = "length"`** —
câu trả lời bị cắt vì trần token, đi qua nhánh *thành công*, và nếu cache thì
một câu cụt sẽ được phát lại vĩnh viễn. Thêm
`test_a_truncated_answer_is_never_cached`, tiêm lại: đỏ. Cùng khuôn P4 của
`W4-08`: đoán đúng hướng, sai cơ chế — và cơ chế mới là thứ đáng tiền.

## 5. Test

Unit **+20** (`test_semantic_cache.py`: ngưỡng, hàng rào, namespace, suy giảm,
trim, `embedder_of`) **+10** tầng service (phát lại nguyên bộ khung không gọi
model; `meta.cache` khai câu đã khớp; lưu lịch sử với `finish_reason="cache"`;
ghi cache đúng lúc; câu cụt không ghi; `cache_eligible`). Integration **+3**
(Redis thật + HTTP thật: paraphrase-theo-từ-cuối hit và phát lại đúng text +
sources của lượt đầu; khác chủ đề miss; TTL trên namespace). App test chỉ bật
cache qua `CHAT_TEST_CACHE=1` — Redis là trạng thái chung giữa các test.

## 6. Còn lại / giới hạn nói to

* Paraphrase thật chỉ hit 1/10 — đây là **tính chất của embedding space**, không
  phải bug: BGE-M3 xếp "cùng nghĩa khác lời" và "khác nghĩa gần lời" vào cùng
  một dải cosine. Muốn cache paraphrase tử tế cần một bước so khớp khác
  (cross-encoder trên cặp câu hỏi, hoặc LLM verifier) — để ngỏ, có số đo làm nền.
* Trần 128 entry/namespace và so khớp O(N) phía client: đúng cho quy mô này,
  sẽ sai khi cache cần vạn entry — lúc đó chuyển sang RediSearch/Qdrant làm
  bài toán riêng.
* `_trim` đọc lại cả hash sau mỗi lần ghi — cùng bậc chi phí với lookup, nhưng
  hai lệnh Redis mỗi lượt ghi; nếu thành điểm nóng thì gộp bằng script Lua.
