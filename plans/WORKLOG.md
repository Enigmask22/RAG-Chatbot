# WORKLOG — nhật ký phiên làm việc

> Mục đích: nếu phiên Claude Code bị ngắt giữa chừng (hết quota 5h, mất mạng, đóng máy),
> file này cho biết **đang làm dở tới đâu** và **lệnh nào để tiếp tục**.
> Trạng thái chính thức của từng task vẫn nằm ở [`CHECKLIST.md`](CHECKLIST.md).
>
> **Phiên mới nhất: 2026-09-04 (2) (cuối file)** — `W4-08` xong, `W4` **8/13**. Bộ định tuyến LLM, chạy thật DeepSeek → GLM. ⭐⭐ Chuyển nhà cung cấp **giữa stream** nối hai câu trả lời khác nhau; ⚠️ tiêm lỗi 12 phép, 2 sống sót và cả hai là lỗ thật.
>
> Phiên trước: **2026-09-04** — `W4-07` xong, `W4` **7/13**. Hiểu câu hỏi trước khi truy hồi. ⭐⭐ Chỉ thị ngôn ngữ **8/8 → 0/8**; ⭐⭐ tiêm lỗi phơi ra `"thầy cô"` bị xếp là chào hỏi ⇒ không truy hồi; ⭐⭐ lần chạy thật đảo ngược một quyết định thiết kế. ⚠️ File này có lỗ hổng 03/09 → 04/09 — xem changelog `CHECKLIST.md`.
>
> Phiên trước: **2026-08-22 (10)** — `W3-08` xong, `W3` **7/9**, `G3` **2/3**. API điều khiển ingestion + worker arq (`pipeline/ingest/`). ✅ `POST` **p50 3,47 ms · p95 4,26 ms**, dư **47×** so với trần 200 ms — nhanh vì endpoint không làm gì, và có test **tiến trình con** ghim rằng import app không kéo theo torch. ⭐⭐ **`max_tries` KHÔNG thử lại `Exception` thường**: arq chỉ thử lại `Retry`/`RetryJob`/`CancelledError` (`arq/worker.py:613-633`) — tôi viết test để chứng minh nó hoạt động, test đỏ, và hoá ra mặc định của arq đúng; phân loại lỗi chuyển vào `is_transient`. ⚠️⚠️ **`doc_ids` suýt thành endpoint XOÁ index** — `build_index` gỡ mọi tài liệu không thấy trong lượt chạy, nên diễn đạt "index lại tài liệu này" bằng bộ lọc corpus sẽ xoá 59 tài liệu kia và trả 200. ⚠️ Ba lỗi test đều hỏng không giống nguyên nhân (`Annotated` trong hàm → 422 query param; hàng đợi Redis dùng chung; thứ tự fixture). 💡 Và `-q` tôi gõ suốt nhiều phiên thành `-qq`, tắt hẳn dòng tổng kết — `pyproject.toml` đã cảnh báo từ `W2-05`.
>
> Phiên trước: **2026-08-22 (9)** — `W3-07` xong, `W3` **6/9**, và `G3` được một tiêu chí. Re-index tăng dần ở mức **chunk**: nhớ `content_hash` từng chunk, mượn lại vector của point cũ. ⭐ Corpus thật, sửa một dòng ở tài liệu 1.082 chunk → embed lại **6** chunk, nhanh hơn **179,3×** → `G3` "reprocess ≥ 10×" **ĐẠT**. ⭐ **Không dựng cache vector — Qdrant đã là cache**; chỉ thiếu đường đọc lại (`fetch_vectors`). ⭐⭐ Ca tổng hợp cho 2,0% mượn lại còn corpus thật cho 99,4%; chênh lệch ấy nghĩa là chưa hiểu cơ chế, và chỗ hoà giải là phát hiện: `separators` có **thứ tự ưu tiên** nên mỗi `

` là một **điểm đồng bộ lại** — cùng văn bản thêm xuống dòng đoạn: **2,0% → 98,0%** → `TD-28`. ⚠️ Fixture đầu cho 99% ở **mọi** ca (lần thứ ba trong `W3`, và lần này khó nghi ngờ hơn vì kết quả có lợi). ⚠️ Phát hiện chệch ra: `chunk_size=200` cùng `min_chunk_size` mặc định 200 cho chunk **1.460 ký tự, gấp 7,3×**, im lặng.
>
> Phiên trước: **2026-08-22 (8)** — `TD-22` **đóng**, `W3-07` hết bị chặn. Manifest ghim `text_sha256` + `parse_fingerprint`; `iter_documents` parse qua đúng loader chung rồi đối chiếu. ⭐⭐ Nợ sâu hơn **hai tầng** so với lúc ghi: (1) vân tay `W3-01` ghim **tên gói ô dù** — `export_to_markdown` sống trong `docling-core`, gói mà `docling==2.121.0` thả tự do `>=2.91,<3.0`; (2) ngay cả ghim đủ version gói vẫn chưa đủ, vì **trọng số model bố cục tải theo `revision='main'`** (model bảng thì ghim `v2.3.0`) → ghim commit SHA đã phân giải, nhưng đó chỉ làm lần đổi *ồn ào* chứ không *ngăn* → `TD-27`. ⭐ Đo: `text_sha256 == sha256` **60/60** (hai cột mới hôm nay hoàn toàn thừa — và đó đúng là thứ hết hạn khi có parser); nối loader **không đổi gì** (14.284.300 ký tự y nguyên); dựng lại chế độ hỏng bằng cách chép **nguyên byte** sang `.md` → sha256 trùng khít, **−21,6% văn bản**. ⚠️ Ba assert đầu tôi viết là vô nghĩa (tautology, `or True`) — cùng hình dạng với `W3-05` §3, hai phiên liền.
>
> Phiên trước: **2026-08-22 (7)** — `W3-05` xong, `W3` **5/9**. Small-to-big (`chunking/parent_child.py` + `retrieval/context.py`), hai index thật: `rag_pc256` 10.473 point, `rag_pc128` 22.924 point. ✅ Cả ba vế DoD đạt. ⭐ **Parent KHÔNG phải một point** — nó là *tập* các child, nên 0 field payload mới, 0 field `MetadataFilter`, 0 backfill, 0 con số `W2` bị đổi; và parent không thể lọt vào kết quả tìm kiếm vì nó không nằm trong đó. ⚠️⚠️ **Protocol khớp với một method KHÔNG TỒN TẠI, 27 test vẫn xanh**: gọi `get_by_ids`, lớp thật có `fetch_chunks`, fake trong test cũng khai sai y hệt → *một Protocol cấu trúc không ràng buộc được gì nếu cả hai bên đối chiếu đều do test dựng ra*. Khuôn để bịt đã có sẵn ở `W3-06` mà tôi không dùng lại. ⭐⭐ **Độ nở ngữ cảnh là chỉ số ĐÁNH LỪA**: chia đôi child làm nó gấp đôi (3,45× → 6,98×) trong khi prompt thật không đổi (9.471 → 9.519 token) — prompt = *số parent* × *cỡ parent*, child không có trong công thức. Hệ quả: **child nhỏ hơn không đắt hơn**. ⚠️ `chunk_size=256` token **không phải "small"** (child 256 tok > chunk baseline 218 tok) → thêm `pc128`.
>
> Phiên trước: **2026-08-22 (6)** — `W3-06` xong, `W3` **4/9**. Kích thước chunk tính bằng token (`chunking/tokens.py`). ✅ **DoD đã đạt SẴN**: BGE-M3 0/15.814 chunk bị cắt, dư **11,2×** — đã có số từ `W2-01`. ⭐ **Ký tự không phải đơn vị mang đi được**: đổi tokenizer là đổi số token của cùng một chunk tới **47%**, và chiều lệch EN↔VI **đảo dấu** giữa hai model. ⭐ Giá trị thật là **san bằng kích thước giữa hai ngôn ngữ** (PhoBERT: −22,3% → +2,4%), không phải tránh bị cắt. ⚠️⚠️ **Chép quy tắc p05 của `truncation.py` là SAI** vì ở đây có bước kiểm lại (hụt 17% ngân sách). ⚠️⚠️ **Lỗi trộn đơn vị của chính tôi, 28 test xanh không thấy** — test kiểm *trần*, không kiểm *đích*.
>
> Phiên trước: **2026-08-22 (5)** — `W3-03` xong, `W3` **3/9**. Structure-aware chunker (`chunking/structure.py`). ✅ Cả hai vế DoD đạt, và "đúng" định nghĩa mạnh hơn DoD: **nhất quán với `section_path_at` trên 1061 chunk của hai báo cáo World Bank thật, 0 chunk nói dối**. ⭐ **Gộp mảnh ngắn qua ranh giới section làm `section_path` nói dối** — cùng khuôn lỗi bản POC gộp qua ranh giới *tài liệu*; lối ra là **hạ đường dẫn xuống tổ tiên chung** (giá: 6,0% chunk). ⭐ **PDF thật chỉ cho MỘT cấp heading** (128 và 83 heading, tất cả cấp 1) và **corpus 0/60 tài liệu có cấu trúc** → `TD-24`. ⚠️ **Lỗi `break` của `W3-01` lộ ra**: 6 heading hỏng làm sai **575/587 chunk** ở `wb1`, 0% ở `wb2`. ⚠️ Tự bẫy hai lần: **vị trí cắt ≠ vị trí hỏi** (486/587 chunk lệch một section), và **script kiểm chứng lặp lại đúng lỗi nó đi kiểm**.
>
> Phiên trước: **2026-08-22 (4)** — `W3-02` xong, `W3` **2/9**. Phát hiện scan (`scan.py`) + cổng OCR (`ocr.py`). ⭐⭐ **Máy OCR đi kèm docling đọc tiếng Anh nguyên văn và trả RÁC cho tiếng Việt** — cả hai model (`ch`, `latin`), cùng một ảnh, cùng một lần chạy. Nên với corpus tiếng Việt **bật OCR còn tệ hơn tắt**, và loader **từ chối** thay vì trả rác → `TD-23`. ⭐ Ngưỡng đo từ PDF World Bank thật: **báo cáo born-digital thật vẫn có trang trống** (1/129 · 4/112) nên luật "một trang trống ⇒ scan" sẽ đẩy 100% báo cáo vào OCR. ⚠️ **Đính chính `W3-01`**: 70,56 s là cold start, cận biên thật **0,35 s/trang** — lý do `W3-02` tồn tại không phải chi phí mà là đúng/sai.
>
> Phiên trước: **2026-08-22 (3)** — `W3-01` xong, `W3` **1/9**. Loader 6 định dạng (`rag_core.loaders`). ✅ Cả hai vế DoD đạt. ⭐ **Fixture "sạch" đo cái generator của tôi, không đo docling** — PDF hai cột toàn dòng ngắn bằng nhau cho kết quả **không đơn điệu** (4 ✗ · 12 ✓ · 24 ✗ · 40 ✓); đổi sang cột văn xuôi so le thì 4/4 đúng. ⭐⭐ **Chèn parser vào giữa byte và `Document.content` giết sạch golden set: 0/60 tài liệu đồng nhất byte, 0/280 span sống sót** — mà văn bản chỉ ngắn đi 8,85%, vì chữ không mất, **dòng bị dồn**. → `TD-22`. `.txt` **không** đi qua docling, và đó là điều kiện để mọi con số `W2` còn giá trị.
>
> Phiên trước: **2026-08-22 (2)** — `W2-09` xong, **`W2` đóng 10/10**. ⭐ Bảng delta baseline → đỉnh bảng **15/15 metric**, `ndcg@10` ×4,2, `hit_rate@5` **0↔120 câu**. ⭐⭐ **Câu thứ hai của DoD — "category nào cải thiện nhiều nhất" — KHÔNG có câu trả lời**: nó so `Δ_A` với `Δ_B` mà mọi bảng đã công bố chỉ kiểm `Δ ≠ 0`; dựng `contrast.py` (bootstrap **không cặp**) thì **cả 6 nhóm hoà, 0/5 phân giải được**, và **vẫn hoà khi bỏ hiệu chỉnh** — giới hạn của dữ liệu, không của ngưỡng. Cần **~440 câu**. ⚠️ **Hàng rào thứ tư**: số nhãn/câu đổi **giữa các nhóm**, nên `ndcg@10` không xếp hạng nhóm được. ⭐ **Bậc đổi embedding model chiếm phần lớn nhất ở cả 6/6 category**. ⭐⭐ Quyết định (b): **luật nâng `B` hiển nhiên cho câu trả lời SAI** — `MIN_TAIL_RESAMPLES` 30 → **128**. Việc tiếp theo: `G2` rồi `W3`.
>
> Phiên trước: `W2-08` xong: bảng ablation **14 ô** có `p`/CI từng dòng (`make ablation`). ⭐ Câu "cấu hình nào thắng" là một **phép chọn cực đại**, nên câu trả lời là một **tập** — bốn rổ: 2 không phân biệt được · 3 tranh chấp · 8 kém ở mọi metric · 1 không so được. ⭐⭐ **Người thắng từng do 6 mẫu lại trên 10.000 quyết định** (α đã hiệu chỉnh để lại đúng 6 mẫu trong đuôi; đo 6 seed thì `ndcg@10` đổi dấu, và ở B=50.000 thì khoảng **chứa 0**) — và điều đó **ngược** ghi chú tôi tự viết ở `W2-08-prep`. ⚠️⚠️ Lỗi đắt nhất: `KHÔNG SO ĐƯỢC` đi cùng đường với "hoà", nên **ô tệ nhất bảng** thành thành viên rẻ nhất của tập thắng. ⚠️ Luật `1/n` của tôi sai và 13/14 file đã công bố nói ra. Việc tiếp theo: `W2-09`.
>
> Phiên trước: `W2-08-prep` xong: `compare.py` có `--by category|lang` (quét, **có** hiệu chỉnh Bonferroni) và `--category X` (một giả thuyết nêu trước, không hiệu chỉnh); `scripts/category_compare.py` **đã xoá**. ⭐ Phát hiện `cross_lingual` của `W2-04` **sống sót** hiệu chỉnh cho cả 90 phép kiểm, nhưng **cả ba dẫn chứng đã công bố** là `p` không có lực trích cạnh một CI có nghĩa — đã sửa. ⭐ Hai cờ mới `KHÔNG ĐỦ LỰC` (trần `p` của McNemar là `2/2ⁿ`; `table_lookup` 4 câu là vĩnh viễn không đo được) và `KHÔNG KẾT LUẬN` (biên CI đúng 0 — lỗi do chính bản hiệu chỉnh của tôi tạo ra, phải đo mới thấy). Việc tiếp theo: `W2-08`.
>
> Phiên trước: `W2-07` xong: experiment runner (YAML matrix → 14 ô → MLflow), resume theo `fingerprint` của ô, preflight bắt ngay một ô sắp ghi đè bằng chứng `W2-03`. Grid tái lập **5** con số đã công bố đúng từng chữ số. ⚠️ Lượt chạy đầu chạy trọn 14 ô mà **không ghi gì lên MLflow** (mlflow 3 bỏ file store) → URI không mở được giờ là lỗi preflight, và có `make exp-backfill` dựng lại view từ file báo cáo. Việc tiếp theo: `--category` cho `compare.py`, rồi `W2-08`.
>
> Phiên trước: 2026-08-20 — tuần 1 xong 13/13; W2 xong `TD-11` (âm), `W2-01` BGE-M3, `W2-02` hybrid schema, `W2-03` sparse retriever, `W2-04` RRF (**`k=60` của bài báo kém dense có ý nghĩa**; `k=1` thắng nhưng chỉ 3/15 metric đạt ý nghĩa; và một **bug 64 ms** làm sai hai con số đã công bố — sparse thật ra miễn phí hoàn toàn). Việc tiếp theo: `W2-05`.
> Sắp xếp theo thứ tự thời gian, mục mới thêm vào **cuối**.

---

## Phiên 2026-08-17 · Tuần 1 — nền móng

**Mục tiêu phiên:** hoàn thành 8 task W1 không bị chặn — `W1-01` … `W1-07` và `W1-12`.

### Lệnh để kiểm tra trạng thái hiện tại

```bash
cd D:/studioproj/RAG-Chatbot   # đổi tên 2026-08-20, xem reports/tasks/rename-workspace.md
uv sync --extra dev --extra qdrant   # cài môi trường
make lint                            # ruff check + ruff format --check + mypy strict
make test                            # 144 unit test, không cần Docker, ~3s
make up && make test-integration     # cần Docker Desktop đang chạy
```

### Bảng tiến độ phiên

| Task | Nội dung | Trạng thái | Ghi chú |
|---|---|---|---|
| `W1-01` | Monorepo skeleton + tooling | ✅ xong | `make lint` + `make test` exit 0 |
| `W1-02` | Pydantic schemas | ✅ xong | 20 test |
| `W1-03` | Port chunking (bỏ LangChain) | ✅ xong | 35 test |
| `W1-04` | SQLite cache thay pickle | ✅ xong | 16 test |
| `W1-05` | docker-compose hạ tầng | ✅ xong | 4 integration test, 3 service healthy |
| `W1-06` | Embedding provider | ✅ xong | 14 test |
| `W1-07` | Dense retriever Qdrant | ✅ xong | 14 integration test |
| `W1-12` | Retrieval metrics + report | ✅ xong | 35 + 19 test |
| `W1-08` | Build index (corpus → Qdrant) | ✅ xong | 59 test mới · index baseline 15.814 chunk |
| `W1-10` | Sinh nháp golden set | ✅ xong | 100 test mới · 266 câu · $0,5821 |

**Tổng kết phiên:** 144 unit test + 18 integration test xanh · `ruff` + `mypy --strict` sạch ·
coverage `rag_core` 81%. Bằng chứng: [`reports/tasks/w1-foundation.md`](reports/tasks/w1-foundation.md).

Ngoài scope dự kiến, đã thêm: `test_architecture_boundaries.py` (canh chiều phụ thuộc
hai plane) và `test_settings.py`.

### Quyết định kỹ thuật phát sinh trong phiên

*(đây là thứ dễ quên nhất sau khi ngắt phiên — mọi quyết định đều ghi lại đây)*

1. **Bỏ LangChain khỏi `rag_core`.** Bản POC dùng `RecursiveCharacterTextSplitter` +
   `SemanticChunker` của `langchain-experimental`. Cả hai được viết lại thuần Python
   (~150 dòng tổng cộng) trong `packages/rag_core/chunking/`. Lý do: `langchain-experimental`
   kéo cây phụ thuộc nặng và API hay đổi; tự viết thì test được từng nhánh.

2. **Phụ thuộc nặng nằm ở extras.** `sentence-transformers`/`torch` → extra `embeddings`,
   `qdrant-client` → extra `qdrant`. Unit test chạy không cần chúng nhờ
   `HashingEmbeddingProvider`. `make test` chạy ~3 giây.

3. **`HashingEmbeddingProvider` thay vì fake trả vector ngẫu nhiên.** Test của semantic
   chunking cần embedding có **tương đồng thật**; vector ngẫu nhiên deterministic sẽ
   khiến bài test pass kể cả khi thuật toán sai. Provider này là bag-of-words hashing
   đã chuẩn hoá — dùng cho test và CI, **không** dùng cho eval thật.

4. **Ba sai lệch có chủ ý so với `enhanced_chunking.py`** (ghi ở docstring
   `chunking/base.py`, ảnh hưởng tới số baseline `W1-13`):
   - `config_hash` không làm tròn tham số nữa. Bản cũ làm tròn `chunk_size` về bội 100
     → `1000` và `1049` dùng chung cache entry, đủ để một vòng ablation báo hai cấu
     hình cho kết quả y hệt.
   - Hậu xử lý áp **theo từng tài liệu**, không trên danh sách gộp (bản cũ gộp chunk
     nhỏ vào chunk của tài liệu khác).
   - Ép kích thước trước, thêm ngữ cảnh hàng xóm sau (bản cũ làm ngược).

5. **`neighbor_context_chars` mặc định TẮT.** Bản POC luôn bật ở mức 100 ký tự.
   ⚠️ **Config baseline của `W1-13` phải đặt `neighbor_context_chars=100`** để tái lập
   đúng hệ thống hiện tại.

6. **Cache theo từng tài liệu, không theo cả corpus.** Bản POC hash chuỗi nối của mọi
   tài liệu → sửa một file là mất sạch cache. Đây cũng là nền cho `W3-07`.

7. **Named vector `dense` trên Qdrant ngay từ W1.** Collection tạo bằng vector vô danh
   không thêm được named vector mà không build lại toàn bộ index — W2 sẽ thêm sparse.

8. **Point ID = UUIDv5 sinh từ `chunk_id`.** Tính idempotent nằm ở tầng store chứ không
   ở script build index → `W1-08` thừa hưởng sẵn.

9. **Yêu cầu Python ≥ 3.12** (ban đầu định 3.11). Lý do: stub của numpy dùng cú pháp
   `type` statement chỉ có từ 3.12, mypy strict không chạy được với `python_version=3.11`.
   ⚠️ Ảnh hưởng: image RunPod ở `W0-05` phải có Python ≥ 3.12.

10. **Truy vấn `unanswerable` trả `None` ở mọi metric xếp hạng, không trả `0.0`.**
    Recall trên tập rỗng là không xác định. Nhóm này đo riêng bằng refusal correctness
    ở `W5-02`.

11. **Hai lỗi hạ tầng chỉ lộ khi chạy thật** (không unit test nào bắt được — lý do
    `W1-05`/`W1-07` có integration test riêng thay vì mock Qdrant):
    - `QDRANT__SERVICE__API_KEY: ${QDRANT_API_KEY:-}` khiến biến **luôn tồn tại** với
      giá trị rỗng → Qdrant bật xác thực và trả 401 cho mọi request. Sửa bằng dạng
      danh sách không có dấu `=`.
    - `localhost` trên Windows resolve `::1` trước, Docker chỉ bind `127.0.0.1` →
      **mỗi request Qdrant chậm đúng 2 giây**. Toàn bộ default đã đổi sang `127.0.0.1`;
      integration suite từ ~5 phút xuống 30 giây. Nếu không phát hiện bây giờ thì p95
      latency đo ở W5/W6 sẽ cộng thêm 2000 ms cố định mà rất khó truy nguyên nhân.
    - `qdrant-client` ghim `>=1.15,<1.16` cho khớp major/minor với image server.

12. **Loại `*.md` khỏi ruff.** Ruff 0.16 format cả code block Python bên trong Markdown,
    và nó đã lặng lẽ sửa `README.md`, `enhanced_chunking_presentation.md`,
    `state_chat_presentation.md` — tài liệu viết tay, không phải thứ định đụng tới.
    Đã hoàn nguyên bằng `git checkout` và thêm `*.md` vào `extend-exclude`.

13. **Bộ tải corpus** (`scripts/fetch_corpus.py` + `pipeline/corpus/`). Ba điều học được:
    - World Bank WDS API: **không truyền tham số `fl`** — nó làm rụng đúng hai field
      `pdfurl`/`txturl` cần nhất. Lấy đủ rồi lọc phía mình.
    - Ưu tiên `txturl` hơn `pdfurl`: text đã trích sẵn, dùng được ngay ở W1 mà không
      phải đợi Docling ở `W3-01`.
    - **ADB trả 403 cho mọi truy cập tự động.** Không viết adapter được; tài liệu ADB
      phải tải tay rồi khai báo qua block `seed_list`.

14. **`limit` nghĩa là tổng mục tiêu của nguồn, không phải "thêm N mỗi lần chạy".**
    Bản đầu tiên chạy lần hai đã âm thầm tải thêm 30 tài liệu nữa mà manifest vẫn hợp
    lệ nên không có gì báo động. Đã sửa: trừ đi số đã có theo **tên block** (không phải
    theo loại nguồn, nếu không hai block cùng loại sẽ ăn chung hạn mức).

15. **Giấy phép được ép bằng code, không bằng lời hứa.** `CorpusEntry` từ chối mọi
    giấy phép ngoài allowlist. Đáng chú ý: **giấy phép có `ND` bị từ chối** — chunking
    + sinh context bằng LLM (`W3-04`) là tạo tác phẩm phái sinh, `ND` chỉ cho phép
    phát tán nguyên bản.

16. **`Chunker.prepare(n_documents)` — lỗi nghiêm trọng nhất của phiên này.**
    Cache chunk hoạt động theo **từng tài liệu**, nên `CachedChunker` gọi
    `inner.chunk([doc])` 60 lần. `HybridChunker` quyết định nhánh theo
    `len(documents)`, thấy `n=1` mỗi lần, ngưỡng là 5 → **luôn chọn semantic**.
    Bản POC truyền cả corpus một lượt nên với 60 tài liệu nó chạy fixed. Nếu
    không phát hiện thì baseline `W1-13` đo một chiến lược chunking khác hẳn.
    Sửa: người gọi khai báo tổng số tài liệu trước, khai báo tường minh thắng
    suy đoán theo lô. Log xác nhận `chunker hybrid->fixed`.

17. **`HybridChunker.name` giờ gồm cả nhánh đã chọn** (`hybrid->fixed:...`).
    Tên chunker là một phần khoá cache; tên cũ chỉ có `config_hash` nên hai lần
    chạy hybrid — một rơi semantic, một rơi fixed — đọc lại kết quả của nhau.
    Nhánh semantic còn kèm tên model embedding, nhờ đó ablation đổi model ở W2
    cũng không đọc nhầm.

18. **Ba tầng idempotent; point ID xác định chỉ là tầng 1.** Tầng 2: tài liệu
    **ngắn lại** để lại point mồ côi `doc::00030`… mà không upsert nào ghi đè —
    chúng không trùng lặp, chúng trỏ tới văn bản không còn tồn tại, và retriever
    vẫn trả về. Tầng 3: `fingerprint` trong state chặn việc trộn hai cấu hình
    vào một collection. Thứ tự cố ý là **upsert trước, xoá sau** — chết giữa
    chừng thì thừa chunk cũ (chạy lại dọn được), chứ không mất hẳn tài liệu.

19. **State file không phải nguồn sự thật, Qdrant mới là.** Trước khi tin state,
    script đối chiếu tổng số chunk với `collection.count()`; lệch thì bỏ state
    và index lại toàn bộ. Ca thật: `make down-clean` mà quên xoá `.cache/`.

20. **p95 độ trễ truy hồi 15.219 ms hoá ra là thời gian nạp model** (p50 chỉ
    31 ms). Truy vấn đầu tiên gánh cả việc nạp `sentence-transformers` vốn lazy.
    Thêm một lượt warm-up: p95 còn **98 ms**. Đây là con số làm ngưỡng cho gate
    hiệu năng W5/W6 — để nguyên thì gate đo thời gian khởi động.

21. **Hệ số phình 1.24x của `neighbor_context_chars=100`.** Text đem embed đi từ
    14,3 lên 17,7 triệu ký tự. Con số này chưa nói kỹ thuật đó tốt hay dở, nó
    nói **giá phải trả** — so recall giữa bật và tắt là việc của W2.

22. **`fingerprint` cố ý không gồm `device`/`batch_size`.** Chạy trên GPU thuê
    hay laptop phải ra cùng một index về mặt logic; nếu `device` vào fingerprint
    thì mỗi lần đổi máy là build lại vài giờ GPU — đúng thứ kiến trúc hai plane
    sinh ra để tránh. Có test canh cả hai chiều.

23. **Wheel `torch` mặc định trên PyPI cho Windows là bản CPU-only.** Cài từ đó
    thì mọi thứ vẫn chạy, chỉ chậm hơn nhiều và `torch.cuda.is_available()` trả
    `False` mà không báo lỗi gì. Đã ghim index `download.pytorch.org/whl/cu126`
    trong `[tool.uv.sources]`. Xác nhận `torch 2.13.0+cu126`, build chạy `cuda`.

24. **`plans/` vừa xuất hiện trong `.gitignore`** (không phải do phiên này thêm).
    File trong đó đã tracked nên vẫn commit được, nhưng file **mới** bị bỏ qua
    âm thầm — `reports/tasks/w1-08-build-index.md` phải `git add -f`. Ghi vào `TD-07`.

25. **`deepseek-chat` là BÍ DANH, không phải một model.** Xác nhận trực tiếp
    trên API: cả `deepseek-chat` lẫn `deepseek-reasoner` đều được phục vụ bởi
    `deepseek-v4-flash`. Đây đúng là vấn đề mà quy tắc cứng #1 nói về OpenRouter
    preset, chỉ kín đáo hơn — tên trông như một model cụ thể nhưng là con trỏ do
    server nắm. Mặc định dự án đổi sang slug thật; gọi bằng bí danh vẫn được
    nhưng có cảnh báo.

26. **Model suy luận làm chẩn đoán `max_tokens` sai hoàn toàn.** Triệu chứng là
    hàng loạt "Response không phải JSON hợp lệ" với nội dung rỗng. Đo lại:
    `completion_tokens=1770` nhưng `len(text)=515` ký tự — `deepseek-v4-flash`
    tiêu 1.500–3.000 token cho chuỗi suy luận KHÔNG nằm trong `content`. Nâng
    `max_tokens` lên 6.000 và tách `finish_reason=="length"` thành cảnh báo
    riêng. Vẫn còn 13,5% lời gọi bị cắt — xem `TD-08`.

27. **Chạy song song 6 luồng: >1 giờ → 640 giây.** Mỗi lời gọi ~25 giây và gần
    như toàn bộ là chờ mạng. Kết quả vẫn ráp theo đúng thứ tự lô để hai lần chạy
    cùng seed cho ra cùng một file.

28. **Job trả tiền dài phải có checkpoint.** Lượt đầu treo ở phút 40, mất sạch.
    Giờ ghi nối sau mỗi lời gọi; chạy lại bỏ qua lô đã xong. Lô mà model trả về
    rỗng cũng ghi dấu — nếu không thì mỗi lần chạy lại đều trả tiền cho cùng một
    câu trả lời rỗng.

29. **27,8% chunk của corpus bị trộn hai cột PDF.** Bản `.txt` World Bank giữ
    nguyên vị trí ký tự của trang hai cột nên cột trái/phải đan xen theo dòng.
    Nó gồm toàn từ hợp lệ nên mọi bộ lọc theo tỉ lệ chữ cái đều cho qua; chỉ
    `gutter_ratio` (khoảng trắng dài ở GIỮA dòng) lộ ra. Thêm hai bộ lọc nữa:
    `mean_words_per_line` (loại chú thích biểu đồ) và `skip_leading_chunks`
    (loại bìa/bản quyền/mục lục). Tổng cộng loại ~60% lô → `overshoot=3.0`.
    ⚠️ Ba bộ lọc này CHỈ áp cho việc chọn mẫu sinh câu hỏi, KHÔNG áp cho index.

30. **Model không được tự viết `chunk_id`.** Nó chỉ trả về chỉ số đoạn văn trong
    danh sách được đưa; code ánh xạ sang id thật. Cho model tự viết id là mở
    đường cho cả golden set trỏ vào hư không. Kèm theo: `quote` được đối chiếu
    lại với chunk (16/266 câu không kiểm chứng được), và `multi_hop` chỉ dùng
    một chunk thì bị hạ nhóm về `factoid`.

### Việc còn dở / cần làm tiếp

- [x] **Đã commit + push** lên nhánh `feat/w1-foundation` (2026-08-17):
      `f5ec22b` nền móng W1 · `c253fa5` bộ tải corpus W0-03.
      Cố ý **không** push thẳng `main` vì repo này đang là link trong CV — trang chủ
      repo giữ nguyên bản cũ cho tới khi bạn chủ động merge.
- [ ] Quyết định merge `feat/w1-foundation` vào `main` hay giữ nhánh (liên quan `W0-02`).
- [x] `W1-08` — **xong**: `reports/tasks/w1-08-build-index.md`. Collection `rag_baseline`
      có 15.814 chunk; `make index` chạy lần hai không ghi thêm gì.
- [x] `W1-10` — **xong**: `reports/tasks/w1-10-goldenset-draft.md`, 266 câu nháp.
- [x] `W1-11` — **xong ở phiên 2026-08-20 nhưng review bằng MODEL, không phải người**:
      `golden_v1` 242 câu, loại 24/266. Việc còn lại của bạn nằm ở `TD-13`.
- [x] `W1-09` (DVC) và `W1-13` (đo baseline) — **xong ở phiên 2026-08-20**.
- [ ] Corpus nguồn (b) pháp luật và (c) HOSE — cần bạn chọn tay, khai báo qua `seed_list`.
      ⚠️ Phải dựng DVC remote dùng chung **trước** đó (`TD-10`).
- [x] Gate `G1` 4/4 mục — nhưng là **PASS có điều kiện** (`TD-13`), ký hiệu vẫn 🟡.

> ⬇️ **Mọi mục "còn dở" ở trên đã được xử lý hoặc chuyển thành `TD-xx` trong phiên
> 2026-08-20.** Đọc mục "Vấn đề đang mở" ở cuối file để biết trạng thái hiện tại.

### Nếu phiên sau bắt đầu từ đây

Đọc theo thứ tự: mục "Quyết định kỹ thuật" ở trên → `reports/tasks/w1-08-build-index.md`
→ `reports/tasks/w1-foundation.md` →
`CHECKLIST.md` §1 Dashboard và §10 Đang bị chặn. Chạy `make lint && make test` để xác
nhận repo vẫn xanh trước khi làm tiếp.

---
## Phiên 2026-08-20 · Tuần 1 — golden set, DVC, baseline

**Mục tiêu phiên:** 5 task còn lại của W1 (`W1-08`…`W1-13`) → đóng gate `G1` → sẵn sàng vào W2.

### Lệnh để kiểm tra trạng thái hiện tại

```bash
cd D:/studioproj/RAG-Chatbot
uv sync --extra dev --extra qdrant --extra embeddings
make lint                            # ruff check + format --check + mypy strict (74 file)
make test                            # 534 unit test, không cần Docker
make up && make test-integration     # 38 integration test, cần Docker Desktop
make data-verify                     # đối chiếu DVC <-> corpus_manifest.csv
make goldenset-verify                # checksum golden_v1 (242 câu)
make eval-retrieval                  # đo lại baseline, ~20 giây
```

### Bảng tiến độ phiên

| Task | Nội dung | Trạng thái | Ghi chú |
|---|---|---|---|
| `W1-09` | DVC version corpus | ✅ xong | 23 test · clone sạch `dvc pull` 61 file/1,9s · 60/60 sha256 khớp |
| `TD-12` | Neo golden set theo span ký tự | ✅ trả xong | 121 test mới · 0 câu mất nhãn ở `chunk_size` 1000/600/400 |
| `W1-11` | Triage + review + freeze `golden_v1` | ⚠️ xong **có điều kiện** | 242 câu · review bằng **model**, không phải người → `TD-13` |
| `W1-13` | Đo baseline | ✅ xong | recall@5 0,1746 · MRR 0,1660 · chạy lại sai số **0,0000%** |
| `W0-04` | Credential & biến môi trường | ✅ xong | Trước bị đánh `[?]` nhầm; DoD đã đủ từ 2026-08-17 |

**Tổng kết phiên:** 534 unit + 38 integration test xanh · `ruff` + `mypy --strict` sạch trên
74 file · coverage `rag_core` 80% · 8 commit (`b48a34b` … `b9fcbce`) đã push lên
`feat/w1-foundation`. Tuần 1 **13/13**, gate `G1` 🟡 (4/4 kỹ thuật, 1 điều kiện: `TD-13`).

### Quyết định kỹ thuật phát sinh trong phiên

*(tiếp số từ phiên trước — quyết định 1–30 ở phiên 2026-08-17)*

31. **DVC remote nằm ở `.dvc/config.local`, không phải `.dvc/config`.** `dvc remote add -d`
    ghi đường dẫn tuyệt đối `D:/dvc-remote/...` vào file **được commit** → mọi clone nhận
    remote trỏ vào ổ đĩa không tồn tại. Cùng loại lỗi với đường dẫn tuyệt đối trong
    `.venv/*.pth` gặp lúc đổi tên workspace. Phải dùng `--local`.

32. **`data/golden` KHÔNG đưa vào DVC** (khác checklist gốc). Golden set là **thước đo**;
    thứ cần nhất ở nó là **diff đọc được** lúc review — ai đổi nhãn câu nào, từ gì sang gì.
    DVC thay file bằng một hash nên mất đúng thứ đó. 284 KB text không phải lý do để tránh
    git. Tính tái lập không mất: một commit ghim cả hai. Lý lẽ đầy đủ: `reports/tasks/w1-09-dvc.md` §3.2.

33. **Corpus giờ có HAI cơ chế versioning, và chỗ cả hai đều mù là phép so SỐ LƯỢNG.**
    sha256/manifest và md5/DVC. Thêm file vào `data/corpus/` rồi `dvc add` mà quên manifest
    thì **không lỗi nào nổ ra**: build index bỏ qua file lạ, `dvc status` vẫn sạch,
    `dvc push` vẫn đem nó lên remote. `pipeline/corpus/dvc_state.py` canh đúng chỗ đó.

34. **Bất đối xứng của tín hiệu triage — quyết định quan trọng nhất của phiên.**
    Câu `unanswerable` mà retriever tự tin = **bằng chứng nhãn sai** (mệnh đề về corpus bị
    phản chứng). Câu trả lời được mà retriever trượt **không** phải bằng chứng nhãn sai —
    đó chính là thứ eval tồn tại để đo, loại nó đi là **tự thổi phồng recall baseline**.
    Nên tín hiệu thứ hai xếp **cuối** hàng đợi, đề xuất `accept`, và có 2 test canh
    (`TestSignalB`). Không có luật này thì 161/226 câu "khó" bị dọn khỏi tập đo và recall@5
    nhảy từ 0,17 lên khoảng 0,6 mà không cải tiến gì cả.

35. **Ngưỡng nghi ngờ hiệu chuẩn TỪ DỮ LIỆU, không phải hằng số.** Trung vị điểm top-1 của
    các câu trả lời được = **0,5797**. Ghim `0.8` là đoán, vì "điểm cao" phụ thuộc model
    embedding và corpus. Kèm theo: điểm cao **không** đủ để phân loại lại tự động — ví dụ
    đầu hàng đợi có điểm 0,7287 mà chunk top-1 vẫn không trả lời được câu hỏi. Điểm cao
    chứng minh *cùng chủ đề*, không chứng minh *trả lời được*.

36. **`TD-12` — `chunk_id` thuần vị trí làm golden set hỏng ÂM THẦM.**
    `chunk_id` là số thứ tự trong tài liệu, nên đổi `chunk_size` khiến nhãn trỏ vào **văn bản
    khác mà vẫn hợp lệ**. Đo thật: nhãn `chunk_id` cũ "hợp lệ" 226/226 ở cả ba cấu hình
    1000/600/400 trong khi trỏ vào chỗ khác; nhãn span mất **0** câu. Sửa **trước** review vì
    6–8 giờ công người sẽ hỏng ở đúng việc đầu tiên của W2 (hạ `chunk_size`).

37. **Span là PROVENANCE, không phải chỉ thị cắt.** `chunk.content` không bằng
    `text[start:end]` — splitter nối các mảnh bằng ký tự xuống dòng bất kể nguyên bản ngăn
    nhau bằng gì, và phép split bỏ mảnh rỗng nên hai dòng trống thành một. Cảnh báo này ghi
    thẳng vào docstring của `TextSpan` và `chunking/pieces.py`, vì người đọc sau sẽ rất tự
    nhiên dùng span để slice lại văn bản.

38. **Luật chồng lấp span↔chunk phải ĐỐI XỨNG.** Bản đầu chỉ xét `overlap/span.length` →
    **mất 40/226 nhãn ở `chunk_size=400`**, vì chunk nhỏ nằm trọn trong span rộng thì tỉ lệ
    theo span rất thấp. Luật đúng: `overlap/span.length >= r` **HOẶC**
    `overlap/chunk.length >= r`. Chỉ lộ ra vì đem thử ở đúng cấu hình mà W2 sẽ dùng thật,
    không phải ở fixture tự bịa.

39. **Ánh xạ span trả rỗng thì GIỮ NHÃN CŨ.** `evaluate_run` bỏ qua câu có
    `relevant_chunk_ids` rỗng (coi là unanswerable) → nếu ánh xạ trượt mà trả rỗng thì câu
    khó nhất tự rơi khỏi tập đo và recall tăng. Cùng cái bẫy của quyết định 34, ở tầng khác.

40. **Thêm field optional vào pydantic KHÔNG làm `ValidationError`** — nên cache chunk phải
    có version trong **tên bảng** (`chunk_cache_v2`). Thêm `start_char`/`end_char` rồi đọc
    lại cache cũ thì được chunk **không có offset**, hợp lệ hoàn toàn, và `span_resolution`
    im lặng rơi về nhãn `chunk_id` cũ. Đây là lý do `chunks_by_document` WARN kèm gợi ý
    `--recreate` khi thấy chunk thiếu offset.

41. **Bất biến số một của lần refactor chunking: nội dung chunk không đổi MỘT BYTE.**
    Kiểm bằng digest trên corpus thật 60 tài liệu trước/sau (`b381634d51e39365` /
    `e00bc87aaffe792e` cho hai cấu hình) + class `TestTextUnchanged` so với biểu thức
    tiền-refactor. Không có phép này thì mọi thay đổi baseline sau đó lẫn giữa "cải tiến"
    và "refactor làm hỏng chunking".

42. **`freeze` làm rơi `relevant_spans` — lỗi tệ nhất của phiên.** `_apply()` không truyền
    field đó nên `golden_v1.jsonl` ra với `relevant_spans` rỗng: toàn bộ công của `TD-12`
    bị vô hiệu ở **đúng bước cuối**, mà **không có triệu chứng nào** — file hợp lệ, eval
    chạy, số vẫn ra. Phát hiện nhờ để ý `config.span_resolution` **vắng mặt** trong JSON
    baseline đầu tiên. Quyết định kèm theo: `edit` có điền `new_relevant_chunk_ids` thì
    **bỏ span**, vì ánh xạ span ghi đè `chunk_id` nên giữ span cũ sẽ âm thầm huỷ sửa tay
    của người review.

43. **Provenance review ghi vào DỮ LIỆU, không phải vào report.** `reviewed_by_human` (bool)
    tách khỏi `reviewed_by` (chuỗi); `freeze` chỉ đặt `true` khi `--reviewer human` với
    `HUMAN_REVIEWER` là hằng số hard-code, và in WARNING mỗi trường hợp khác. Nhờ vậy
    **không chỗ nào trong repo có thể khẳng định "người đã xác nhận"** khi thực tế là model.

44. **`quote_unverified` là dấu hiệu LỖI TRÍCH XUẤT PDF, không phải model bịa.** 15/16 trích
    dẫn có thật trong corpus; cái làm chúng không khớp là trộn hai cột chèn chữ vào giữa từ
    (`bổ sung` thành `bổ chuyên sung`), số chú thích chèn giữa câu, gạch nối cuối dòng, và
    trích dẫn vắt qua ranh giới chunk. Đọc cờ này thành "model bịa" là loại oan 16 câu.

45. **Kiểm câu `unanswerable` phải tra VĂN BẢN GỐC, không phải retriever.** Retriever đang
    đơn ngữ và chỉ nhìn 256 token đầu mỗi chunk — dùng nó để xác nhận "corpus không có" là
    lấy điểm yếu của hệ thống làm bằng chứng về dữ liệu. Tra thẳng text: **7/40 câu sai nhãn**
    (lạm phát 2025 = 4,5–5% · GDP 2030 = 5,45% trong bảng CGE của CCDR · 1.584 thủ tục tiền
    kiểm · nghèo đa chiều 2022), tỉ lệ sai 17,5% trong nhóm.

46. **Phép kiểm máy sửa lại mắt thường.** Tôi nghi `factoid-69f6fec136` (2.938 tỷ đồng PFES)
    không có căn cứ; phép kiểm grounding chứng minh số đó có thật, chỉ nằm **ngoài span đã
    thu hẹp**. Tin mắt thì đã loại oan một câu đúng.

47. **Baseline chạy lại sai số 0,0000%, không phải "<1%".** Dense retrieval trên vector đã
    ghi là phép tính xác định. Lượt hai chạy với `DEEPSEEK_API_KEY` và `OPENROUTER_API_KEY`
    **rỗng** → đồng thời là bằng chứng cho hạng mục "eval truy hồi không cần LLM API" của `G1`.

48. **recall@5 = 0,1746 thấp CÓ LÝ DO đã định lượng, và cố ý không sửa trước khi đo.**
    (1) `cross_lingual` = 43/209 câu tức **20% tập đo**, model embedding đơn ngữ → recall@5
    bằng 0 là con số **đúng**, không phải lỗi đo. (2) `TD-11`: 56,8% chunk bị cắt ở 256 token,
    15,7% text không tới được vector. Sửa trước rồi gọi kết quả là "baseline" thì mọi so sánh
    về sau đo lẫn cải tiến này vào.

### Vấn đề đang mở (đọc trước khi bắt đầu W2)

- ⚠️ **`TD-13` — golden set review bằng MODEL.** `reviewed_by_human=false`. Không chặn W2
  (so sánh tương đối giữa các cấu hình vẫn hợp lệ vì cùng một thước đo), nhưng **chặn** chữ
  "human-verified" ở README / CV / phỏng vấn, và chặn `G1` 🟡 → ✅. Việc cần làm: người đọc
  lại **33 câu `unanswerable` + 43 câu `cross_lingual`** (hai nhóm loại nhiều nhất = hai nhóm
  model kém tự tin nhất) rồi `make goldenset-freeze` với `--reviewer human`. Đo lại: 20 giây.
- ⚠️ **`TD-11` là hạng mục đầu tiên của W2, làm TRƯỚC `W2-01`.** Nếu hạ `chunk_size` và đổi
  sang BGE-M3 cùng lúc thì không tách được phần cải thiện nào của ai — mà BGE-M3 (cửa sổ
  8192 token) xoá luôn nguyên nhân truncation, nên gộp là mất hẳn một cặp số trước/sau.
- ⚠️ **Build index cho W2 phải dùng `--recreate`.** Point cũ không mang `start_char`/`end_char`
  thì `span_resolution` rơi về nhãn `chunk_id` cũ và **im lặng đo sai** (quyết định 40).
  Có WARNING, nhưng đừng dựa vào việc đọc log.
- ⚠️ **Không sửa `configs/indexing/baseline.yaml`.** Nó là mốc so sánh của cả 6 tuần; mọi
  cấu hình W2 phải là file mới + collection Qdrant riêng.
- `TD-14` — 7 `reference_answer` lẫn ngữ cảnh sinh ("Theo đoạn văn 1"). Vô hại cho eval truy
  hồi (nó chỉ đọc `relevant_spans`), phải dọn **trước `W5-01`** vì judge sẽ trừ điểm oan.
  Câu hỏi thì sạch: 0/242 câu có lỗi này.
- `TD-15` — nhãn liên quan chưa đầy đủ (một dữ kiện ở hai tài liệu, chỉ 1 chỗ được gán).
  Hiếm (1/78 theo phép thử Jaccard) và sai **theo chiều an toàn**: baseline là cận dưới.
- `TD-16` — 10/33 câu `unanswerable` cùng một khuôn "hỏi về nước khác".
- `TD-10` — DVC remote chỉ là thư mục local. **Bắt buộc dựng remote dùng chung trước khi thêm
  nguồn (b) pháp luật và (c) HOSE**, vì hai nguồn đó chọn tay, không script nào tải lại được.
- `TD-05` — đọc tay trang giấy phép của ~5 tài liệu **trước khi** push repo lên public.
- `TD-08` — 22/163 lời gọi LLM vẫn bị cắt ở `max_tokens=6000`.
- `table_lookup` chỉ 4 câu — chờ nguồn (c) HOSE + `W3-01`. Mọi số của nhóm này (recall
  0,0000) **không suy ra được gì** với n=4.
- `W0-02` — quyết định merge `feat/w1-foundation` vào `main` hay tách repo mới.

### Nếu phiên sau bắt đầu từ đây

1. `make lint && make test` để xác nhận repo vẫn xanh (534 unit test).
2. Đọc `reports/tasks/w1-11-review.md` §1 (ai review và điều đó nghĩa là gì) và §4 (baseline).
3. Đọc §4 của `CHECKLIST.md` — thứ tự tấn công W2 đã chốt: `TD-11` → `W2-01` → `W2-02` …
4. Việc đầu tiên gõ tay: tạo config mới với `chunk_size` ~600, build index bằng `--recreate`
   vào collection **riêng**, rồi `make eval-retrieval` để có cặp số trước/sau của `TD-11`
   **tách riêng** khỏi việc đổi model embedding.

---

## Phiên 2026-08-20 (tiếp) · Tuần 2 — `TD-11`

**Mục tiêu phiên:** hạng mục đầu của W2 — sửa truncation bằng cách hạ `chunk_size`,
lấy cặp số trước/sau.

**Kết quả: giả định của `TD-11` bị phản chứng.** Phép sửa chạy đúng (56,9% → 0,4%
chunk bị cắt) và không cải thiện gì đo được (`p = 0,711`). Thứ đáng giá nhất lại là
phát hiện về **thước đo**, không phải về `chunk_size`.

### Lệnh để kiểm tra trạng thái hiện tại

```bash
make truncation BUNDLE=baseline                  # đo TD-11, không cần Qdrant
make index BUNDLE=chunk550                       # ~257s trên RTX 4060
make eval-retrieval BUNDLE=chunk550              # ~20s
make eval-compare BASE=baseline CAND=chunk550    # McNemar + bootstrap
make test                                        # 600 unit test
```

### Bảng tiến độ phiên

| Việc | Trạng thái | Ghi chú |
|---|---|---|
| Đo truncation thành thứ quan sát được | ✅ | `max_sequence_tokens` + `count_tokens` + `TruncationStats`; `WARNING` trong mọi lần build |
| Hiệu chuẩn `chunk_size` từ dữ liệu | ✅ | 574 ký tự (an toàn cho mọi ngôn ngữ); chốt 550 |
| `chunk550` + `chunk550nb55` build & đo | ✅ | 31.155 chunk mỗi config |
| Kiểm định cặp giữa hai lần chạy | ✅ | `*-per-query.jsonl` + `pipeline/eval/compare.py` |
| `TD-11` | ✅ trả xong phần đo | Kết luận **âm** — xem `reports/tasks/w2-td11-chunk-size.md` |
| `W2-01` BGE-M3 | ⬜ việc tiếp theo | Giữ `chunk_size=1000`, chỉ đổi model |

**Tổng kết:** 600 unit (+66) + 38 integration test xanh · `ruff` + `mypy --strict`
sạch trên 79 file.

### Quyết định kỹ thuật phát sinh trong phiên

*(tiếp số từ 48)*

49. **Việc đầu tiên không phải sửa lỗi mà là làm cho lỗi có triệu chứng.**
    `sentence-transformers` cắt ở `max_seq_length` không cảnh báo, không lỗi — đó
    là lý do `TD-11` sống được suốt W1. Nên `BuildReport` giờ mang `truncation` và
    in ở mức **WARNING**: một dòng INFO giữa mười dòng INFO khác thì cũng trốn
    được lần nữa.

50. **`None` = "không biết giới hạn", KHÁC "không có giới hạn".** Provider không
    đếm được token thì phép đo trả `None` và người gọi phải cảnh báo. Quy ước
    thành 0 là báo "không bị cắt" cho mọi model — đúng cái cách bug ban đầu trốn
    được. Có 3 test riêng cho bất biến này.

51. **`truncation=False` khi gọi tokenizer, nếu không phép đo thành hằng số 0.**
    Mặc định tokenizer *cắt ở `model_max_length`*, tức trả về đúng con số ngưỡng
    cho mọi text dài. Bỏ sót cờ này thì mọi cấu hình đều báo "đã sửa xong".

52. **Đếm kèm `[CLS]`/`[SEP]`**, và text dài **đúng bằng** giới hạn thì **không**
    bị cắt (`>` chứ không `>=`). Hai lỗi off-by-one theo hai chiều khác nhau.

53. **`truncated_ratio` và `tokens_lost_ratio` phải tách rời.** 90% chunk bị cắt
    mất mỗi chunk 1 token là chuyện nhỏ; 10% chunk mất mỗi chunk một nửa là mất
    hẳn nội dung. Một con số duy nhất không phân biệt được hai ca đó.

54. **Hiệu chuẩn `chunk_size` phải dùng phân vị THẤP của mật độ ký tự/token, không
    phải trung bình — tôi viết sai lần đầu.** Bản trung bình cho ra 946 ký tự,
    trong khi ở 1000 ký tự đã có 56,9% chunk bị cắt. Con số vô lý mà trông hợp lý,
    vì nó trả lời "chunk *trung bình* vừa khít cửa sổ" = ngưỡng mà một nửa vẫn bị
    cắt. Bản đúng: 574 ký tự. Có test hồi quy cho đúng cái bug này.

55. **`max_chunk_size` phải hạ cùng `chunk_size`** — nó là cùng một đại lượng. Để
    nguyên 1500 thì hậu xử lý vẫn thả ra chunk 1500 ký tự và chúng vẫn bị cắt;
    phép sửa chỉ có tác dụng một nửa.

56. **Nhãn neo theo span làm mẫu số của recall thay đổi khi đổi `chunk_size`.**
    Chunk nhỏ hơn → một span phủ nhiều chunk hơn → **1,38 → 1,96 nhãn/câu**.
    `recall@k = |tìm được ∩ liên quan| / |liên quan|` nên nó tụt `1 − 1,38/1,96 =
    29,6%` **kể cả khi truy hồi y nguyên**; thực tế tụt 25,8%. Suýt viết vào báo
    cáo là "hạ `chunk_size` làm tụt recall 26%".
    Metric miễn nhiễm: `hit_rate@k` (mẫu số 1), `precision@k` (mẫu số k), `MRR`.

57. **Không có điểm từng câu thì không kiểm định được gì.** `hit_rate@5` 0,2153 →
    0,2010 trên 209 câu là **45 câu xuống 42 câu** — chênh 3 câu. Đã thêm
    `{run}-per-query.jsonl` và `pipeline/eval/compare.py`: McNemar exact cho metric
    nhị phân, bootstrap cặp (seed cố định) cho metric liên tục. Không cần `scipy`.

58. **`compare.py` TỪ CHỐI so `recall@k`/`nDCG@k`/`MAP@k` khi số nhãn/câu khác
    nhau**, và in kèm "metric này tụt X% ngay cả khi truy hồi y nguyên". Bài học
    của quyết định 56 đóng thành hàng rào, không để trong docstring.

59. **`golden_v1` chỉ phân giải được mức chênh ≥ 6 điểm `hit_rate` (≈28% tương
    đối).** 209 câu, base rate ≈ 0,20: với ~30 câu đổi chiều phải lệch ~12 câu mới
    đạt `p < 0,05`. Đây là **giới hạn của thước đo**, và nó chi phối cả W2: xếp
    hạng 12 tổ hợp ablation bằng mức chênh vài phần trăm là tung đồng xu. Tin tốt:
    ngưỡng `G2` (+0,08 nDCG trên 0,1621 = +49% tương đối) nằm trong tầm đo.

60. **Hạ `chunk_size` là ĐÁNH ĐỔI, không phải thu hồi nội dung đã mất.** Baseline
    bị cắt nhưng mỗi vector vẫn đọc **~950 ký tự**; chunk550 không bị cắt nhưng mỗi
    vector chỉ đọc **678**. Ba cấu hình xếp đúng theo con số đó (950 → 678 → 589,
    `hit_rate@5` 0,2153 → 0,2010 → 0,1770) — chiều thì đơn điệu, độ lớn thì dưới
    ngưỡng phân giải.

61. **Giả thuyết "pha loãng bởi ngữ cảnh hàng xóm" bị số liệu bác.** `chunk550`
    giữ `neighbor_context_chars: 100` nên 36% mỗi chunk là text của chunk bên cạnh
    (baseline 20%). Tôi cho rằng đó là lỗi thiết kế thí nghiệm và dựng
    `chunk550nb55` để đối chứng — nó đi tiếp xuống, không lấy lại được gì. Lập luận
    "giữ giá trị tuyệt đối làm đổi giá trị tương đối" vẫn đúng; kết luận rút ra từ
    nó thì sai. Đo vẫn rẻ hơn suy luận.

62. **1 tài liệu dùng `Ê` (U+00CA) làm dấu cách** (`TD-17`). Font PDF map glyph dấu
    cách thành `Ê` → văn bản **không có ranh giới từ** → splitter rơi xuống mức ký
    tự → tokenizer nổ ra 0,63 token/ký tự (bình thường 0,20) → 31/36 chunk vẫn bị
    cắt kể cả ở `chunk_size=550`. Phạm vi hẹp và đã đo: 1/60 tài liệu, 1/242 câu
    golden set. Phép phát hiện rẻ: **tỉ lệ ký tự whitespace** (corpus p50 37,5%,
    tài liệu này 16,4%) — nên thành cổng chất lượng corpus ở `W3-01`.

63. **Chạy lại `baseline` sau khi thêm per-query cho đúng từng chữ số** trên cả 15
    metric. Lần xác nhận thứ ba cho tính xác định của đường eval.

### Vấn đề đang mở

- **`W2-01` là việc tiếp theo, và giữ `chunk_size=1000`.** Đừng quét thêm
  `chunk_size` với `vietnamese-bi-encoder`.
- ⚠️ **Mọi so sánh W2 từ giờ phải qua `make eval-compare`.** Bảng số không kèm
  `p`/CI thì không kết luận được gì ở mức chênh dưới 6 điểm `hit_rate`.
- ⚠️ Đổi `chunk_size` thì **không** dùng recall@k/nDCG/MAP. `compare.py` tự chặn,
  nhưng bảng Markdown của `eval-retrieval` thì không — nó vẫn in mọi metric.
- `TD-17` — tài liệu `Ê`, sửa ở `W3-01`.
- `TD-13` — golden set vẫn là review bằng model. Thêm câu hỏi sẽ cải thiện **cả**
  độ phân giải (quyết định 59) **và** `TD-13`; gộp hai việc được.
- Collection `rag_chunk550` + `rag_chunk550nb55` còn trong Qdrant (~62k point) để
  đối chiếu lại. Xoá bằng `make down-clean` rồi build lại từ config nếu cần chỗ.
- Các mục còn lại của phiên trước (`TD-10`, `TD-05`, `TD-08`, `W0-02`) không đổi.

### Nếu phiên sau bắt đầu từ đây

1. `make lint && make test` (600 unit test).
2. Đọc `reports/tasks/w2-td11-chunk-size.md` §8 (kiểm định + độ phân giải) và §10 (bước tiếp).
3. `W2-01`: thêm provider BGE-M3, `max_sequence_tokens` phải báo 8192. Chạy
   `make truncation` với config mới **trước** khi eval để xác nhận truncation về 0.
4. Config mới đặt tên riêng + collection riêng; **không** sửa `baseline.yaml`.

---

## Phiên 2026-08-20 (tiếp) · Tuần 2 — `W2-01` BGE-M3

**Mục tiêu phiên:** hạng mục W2 thật đầu tiên — đổi model embedding sang BGE-M3,
giữ nguyên chunking của baseline, lấy cặp số có kiểm định.

**Kết quả: mức tăng lớn nhất của dự án tới giờ, và một bài học về quy kết nguyên
nhân.** nDCG@10 `0,1621 → 0,4442`, cả 15 metric có ý nghĩa thống kê,
`cross_lingual` từ **0** lên `0,3023`. Nhưng mức tăng đó **không** phải công của
việc sửa truncation — xem quyết định 68.

### Lệnh để kiểm tra trạng thái hiện tại

```bash
make truncation BUNDLE=bgem3                  # 0/15814 chunk bị cắt
make eval-compare BASE=baseline CAND=bgem3    # McNemar + bootstrap, 15/15 metric
make test                                     # 644 unit test (~3s)
make test-gpu                                 # 11 test cần model thật (2,2GB)
```

Build lại từ đầu (~405 s trên RTX 4060, **bắt buộc** `--recreate` vì 768 → 1024 chiều):

```bash
python -m pipeline.indexing.build_index --config configs/indexing/bgem3.yaml \
  --recreate --report plans/reports/probes/index-bgem3.json
python -m pipeline.eval.retrieval_eval --index-config configs/indexing/bgem3.yaml \
  --run-name bgem3
```

### Bảng tiến độ phiên

| Việc | Trạng thái | Ghi chú |
|---|---|---|
| `SparseVector` + `HybridVectors` | ✅ | Kiểu bất biến, cưỡng chế 3 bất biến lúc khởi tạo |
| Năng lực sparse tuỳ chọn trên `EmbeddingProvider` | ✅ | Mặc định `None`; `None` ≠ rỗng |
| `BgeM3EmbeddingProvider` | ✅ | Dense + sparse **một** forward pass |
| Truncation với cửa sổ 8192 | ✅ | 56,9% → **0,0%** (0/15814) |
| Index + eval + kiểm định | ✅ | 15/15 metric "khác biệt thật" |
| `W2-01` | ✅ | `reports/tasks/w2-01-bge-m3.md` |
| `W2-02` Qdrant sparse | ⬜ việc tiếp theo | Sparse đã có, cần chỗ chứa |

**Tổng kết:** 644 unit (+44) + 38 integration + 11 gpu test xanh · `ruff` +
`mypy --strict` sạch trên 83 file.

### Quyết định kỹ thuật phát sinh trong phiên

*(tiếp số từ 63)*

64. **Sparse của BGE-M3 không nằm trong `sentence-transformers`, và không nạp
    model lần thứ hai để lấy nó.** `modules.json` chỉ có
    `Transformer → Pooling → Normalize`; sparse là một `Linear(1024 → 1)` đặt lên
    `last_hidden_state`, trọng số ở `sparse_linear.pt` mà ST không đọc. Cách làm:
    nạp file đó riêng và **dùng lại chính `XLMRobertaModel` mà ST đã nạp**
    (`self.model[0].auto_model`). 2,2GB nhân đôi thì 4060 8GB hết chỗ.

65. **Không thêm dependency `FlagEmbedding`.** Nó kéo theo `peft`, `datasets` và
    một vòng đời phát hành riêng, để có đúng một `Linear(1024 → 1)` và một phép
    gộp max. Tự viết là ~40 dòng và kiểm chứng được.

66. **`_forward` là đường sinh dense DUY NHẤT — `_encode` cũng đi qua nó.** Cái
    giá của việc không gọi `model.encode()` là dense có đường code riêng, và hai
    đường sinh dense song song là cách chắc chắn để hai nhánh ablation vô tình đo
    hai thứ khác nhau. Nên có test canh `embed_documents()` khớp
    `SentenceTransformer.encode()`: đo được **max |Δ| = 1,5e-8**, cosine 1,0000.

67. **Phải tự làm lại phép sắp batch theo độ dài mà `encode()` vốn tự làm.**
    Padding tới câu dài nhất trong batch, nên trộn câu 40 token với câu 500 token
    làm hầu hết phép tính đổ vào padding. Sắp **giảm dần** có lợi ích thứ hai:
    câu dài nhất rơi vào batch đầu, nên cấu hình sẽ OOM thì OOM ở giây thứ nhất
    chứ không phải ở phút thứ ba của job 15.000 chunk.

68. ⭐ **Mức tăng +174% nDCG là của MODEL, không phải của việc hết truncation —
    và `TD-11` là thứ chứng minh điều đó.** Lần chạy này đổi ba thứ cùng lúc
    (model, cửa sổ, tính đa ngữ) nên bảng số một mình không tách được. Nhưng đặt
    ba thí nghiệm cạnh nhau thì tách được:

    | | truncation | `hit_rate@5` | |
    |---|---:|---:|---|
    | baseline (PhoBERT 256) | 56,9% | 0,2153 | — |
    | `chunk550` (PhoBERT, hết cắt) | 0,4% | 0,2010 | `p = 0,711` |
    | `bgem3` (BGE-M3, hết cắt) | 0,0% | 0,5455 | `p < 0,001` |

    Hai dòng dưới **cùng** đưa truncation về ~0; một dòng không đổi gì, dòng kia
    +153%. Vai trò thật của cửa sổ 8192 là *cho phép giữ `chunk_size=1000`* — tức
    làm phép đo sạch — không phải tạo ra mức tăng. Kết quả âm của `TD-11` hoá ra
    là thứ có giá trị nhất ở đây: không có nó thì đã gán sai nguyên nhân.

69. **Giữ `chunk_size=1000` làm nhãn BIT-IDENTICAL với baseline.**
    `n_relevant_mean` 1,3828 ở cả hai lần chạy, `span_resolution` giống nhau từng
    trường (`resolved: 209`, `label_changed: 9`). Nên recall@k/nDCG/MAP so được
    trực tiếp và `compare.py` không từ chối metric nào — khác hẳn `TD-11`. Đây là
    lợi ích *đo lường* của việc không đụng vào chunking, không chỉ lợi ích ngữ cảnh.

70. **Gộp trọng số sparse theo token bằng `max`, không `sum`**, và bỏ
    `[CLS]`/`[SEP]`/`[PAD]`/`[UNK]`. Max là định nghĩa của BGE-M3 và nó có nghĩa:
    trọng số trả lời "token này quan trọng thế nào cho đoạn text", không phải "nó
    xuất hiện bao nhiêu lần". Bỏ `[CLS]` là bắt buộc — nó thường là chiều nặng
    nhất, để lại thì mọi cặp text khớp nhau ở đúng chiều đó và điểm sparse gần
    thành hằng số.

71. **Padding phải bỏ theo `attention_mask`, không theo trọng số bằng 0.** Vị trí
    bị mask vẫn đi qua matmul nên vẫn có trọng số dương. Có test riêng cho ca này.

72. **`batch_size` một mình không chặn được VRAM khi cửa sổ là 8192.** 16 câu ×
    8192 token = 131k token, OOM ngay. Thêm trần `max_batch_tokens` tính theo độ
    dài **thật** của batch; chunk ngắn vẫn chạy full batch. Là knob tốc độ nên
    `embedding_max_batch_tokens` nằm **ngoài** `fingerprint`, cùng lý do như
    `device` và `batch_size`.

73. **Chọn provider theo TÊN MODEL, không theo cờ riêng.** `BAAI/bge-m3` → provider
    hybrid. Thêm `use_bge_m3: bool` vào config thì tồn tại được cấu hình
    `model=bge-m3, use_bge_m3=false` vừa hợp lệ về cú pháp vừa vô nghĩa về nội dung.

74. **`None` ≠ `SparseVector` rỗng — bài học `TD-11` lần thứ hai.** `None` =
    "provider không sinh sparse"; rỗng = "đã tính, không token nào dương" (text
    chỉ gồm special token), một kết quả hợp lệ. Gộp lại thì `W2-03` không phân
    biệt được "provider chỉ có dense" với "sparse trả 0 kết quả", và cả hai trông
    giống nhau: im lặng.

75. **Tokenizer BGE-M3 tốn ít hơn 30% token cho text tiếng Anh** (0,172 vs 0,244
    token/ký tự của PhoBERT), và chiều bất đối xứng **đảo**: với PhoBERT thì `en`
    tốn nhiều hơn `vi`, với BGE-M3 thì `en` tốn ít hơn. Đây là gốc của việc `en`
    bị cắt nặng nhất ở baseline (65,3% chunk, mất 19,4% token).

76. **`Ê`-document (`TD-17`) hết truncation nhưng nợ vẫn mở.** Cửa sổ 8192 xoá
    *triệu chứng* (0/36 chunk bị cắt), nhưng văn bản vẫn không có ranh giới từ.
    Triệu chứng mất, nguyên nhân còn — vẫn sửa ở `W3-01`.

77. **Sửa lỗi sổ sách: p95 truy hồi của baseline là 32,8 ms, không phải 39,9 ms.**
    `reports/runs/baseline-retrieval.md` ghi 32,8; CHECKLIST §1 chép sai. Đã sửa cả
    cột **Baseline** lẫn ghi chú dưới bảng.

### Vấn đề đang mở

- **`W2-02` là việc tiếp theo.** Sparse đã sinh được nhưng **chưa dùng**: eval
  của `W2-01` là dense-only. Qdrant **không** thêm named vector vào collection đã
  tồn tại → `rag_bgem3` phải build lại bằng `--recreate` (~405 s).
- ⚠️ **Đừng kể mức tăng của `W2-01` là "sửa được truncation".** Xem quyết định 68.
  Trong CV/interview thì câu đúng là: *"kết quả âm của thí nghiệm trước là thứ cho
  phép quy kết đúng nguyên nhân của thí nghiệm sau"*.
- ⚠️ **Cảnh báo khách quan:** BGE-M3 train trên MIRACL/mC4 và corpus dự án là tài
  liệu World Bank — thể loại model rất có thể đã thấy nhiều. Không phải rò rỉ tập
  test (nhãn sinh từ chunk của chính corpus, chunk giống nhau cả hai lần chạy),
  nhưng mức tăng chưa chắc giữ nguyên trên corpus đóng của doanh nghiệp.
- `TD-13` — golden set vẫn review bằng model, và giờ **càng đáng làm**: mức tăng
  lớn thế này thì con số *tuyệt đối* bắt đầu được dùng để kể chuyện.
- Chi phí phải theo dõi: p95 truy hồi 32,8 → 46,0 ms (+40%, gần như toàn bộ là
  embed truy vấn bằng model lớn hơn). `W2-05` (reranker) sẽ cộng thêm, và `G2` có
  điều kiện p95 < 3500 ms.
- Collection còn trong Qdrant: `rag_baseline`, `rag_chunk550`, `rag_chunk550nb55`,
  `rag_bgem3`. Xoá bằng `make down-clean` rồi build lại từ config nếu cần chỗ.
- Các mục còn lại không đổi: `TD-05`, `TD-08`, `TD-10`, `TD-14`…`TD-17`, `W0-02`.

### Nếu phiên sau bắt đầu từ đây

1. `make lint && make test` (644 unit test) · `make test-gpu` nếu có GPU.
2. Đọc `reports/tasks/w2-01-bge-m3.md` §5 (vì sao mức tăng không phải của `TD-11`) và
   §7 (thứ chưa được kiểm chứng đầu-cuối).
3. `W2-02`: thêm named vector `sparse` vào schema Qdrant, dùng
   `SparseVector.as_qdrant()`. Build lại `rag_bgem3` bằng `--recreate`.
4. Mọi so sánh vẫn phải qua `make eval-compare`. Đổi `chunk_size` thì **không**
   dùng recall@k/nDCG/MAP.

---

## Phiên 2026-08-20 (tiếp) · Tuần 2 — `W2-02` Qdrant hybrid schema

**Mục tiêu phiên:** cho sparse của `W2-01` một chỗ chứa — một collection Qdrant
mang cả `dense` và `sparse`, query được độc lập.

**Kết quả:** xong, và sparse **gần như miễn phí** (+2,3% thời gian index, +19%
dung lượng). Phần đáng nhất lại ngoài DoD: `ensure_collection` giờ **kiểm tra**
schema thay vì tin, và trong 4 ca lệch có một ca hỏng im lặng.

### Lệnh để kiểm tra trạng thái hiện tại

```bash
make test                                   # 666 unit (22 ca schema, thuần, ~3s)
make up && make test-integration            # 59 integration (21 ca hybrid)
python scripts/migrate_collection.py --config configs/indexing/bgem3.yaml
```

Build lại (~414 s trên RTX 4060, **bắt buộc** `--recreate`):

```bash
python -m pipeline.indexing.build_index --config configs/indexing/bgem3.yaml \
  --recreate --report plans/reports/probes/index-bgem3.json
make eval-retrieval BUNDLE=bgem3            # phải khớp số cũ TỪNG CHỮ SỐ
```

### Bảng tiến độ phiên

| Việc | Trạng thái | Ghi chú |
|---|---|---|
| `SPARSE_VECTOR_NAME` + schema có sparse | ✅ | Không `Modifier.IDF` — xem quyết định 80 |
| `upsert` ghi cả hai từ một forward pass | ✅ | +8,8 s trên 389 s |
| `retrieve_sparse()` — nhánh độc lập | ✅ | Tách khỏi `retrieve()` có chủ ý |
| `ensure_collection` kiểm tra schema | ✅ | Ngoài DoD, 4 ca lệch có test |
| `scripts/migrate_collection.py` | ✅ | Cố ý **không** migrate tại chỗ |
| `HashingEmbeddingProvider` sinh sparse | ✅ | Mặc định **tắt** — `name` là cache key |
| Xác nhận dense không đổi | ✅ | 0/209 câu đổi điểm |
| `W2-02` | ✅ | `reports/tasks/w2-02-qdrant-hybrid.md` |
| `W2-03` sparse retriever | ⬜ việc tiếp theo | Bọc thành `Retriever` + đo có `p`/CI |

**Tổng kết:** 666 unit (+22) + 59 integration (+21) + 11 gpu test xanh · `ruff` +
`mypy --strict` sạch trên 85 file.

### Quyết định kỹ thuật phát sinh trong phiên

*(tiếp số từ 77)*

78. ⭐ **`ensure_collection` phải KIỂM TRA schema, không được thấy tồn tại là trả
    về.** Đây là phần không có trong DoD và là phần đáng nhất của `W2-02`. Trước
    đó, chạy config `bgem3` (provider sinh sparse) lên collection `rag_bgem3` cũ
    (dense-only) sẽ chết ở lần upsert **đầu tiên** — sau khi đã nạp 2,2GB trọng
    số và chunk xong tài liệu đầu — với một thông báo của Qdrant không nói phải
    làm gì. Giờ chết ở giây đầu và in ra `--recreate`.

79. **Trong 4 ca lệch schema có một ca HỎNG IM LẶNG, và nó không được bỏ qua:**
    collection có `sparse` mà provider chỉ sinh dense. Nghĩa là đang eval bằng
    provider dense-only trên index hybrid — con số ra trông bình thường trong khi
    **nửa index không được dùng tới**. Đây là biến thể của đúng cái bẫy `TD-11`:
    hệ thống chạy, số ra, không ai biết là sai. Ba ca kia đều gây lỗi rõ ràng.

80. **KHÔNG dùng `SparseVectorParams(modifier=Modifier.IDF)` cho BGE-M3.** Trọng
    số của model là **đã học** — phần "hạ bậc từ phổ biến" đã nằm trong đó. Chồng
    IDF của Qdrant lên là nhân đôi phép ấy, và nó hỏng theo kiểu tệ nhất: điểm vẫn
    ra số, chỉ là sai. `Modifier.IDF` dành cho nhánh **BM25 thô** ở `W2-03`, nơi
    giá trị đầu vào là tần suất chứ không phải trọng số. Có test integration canh
    `modifier` của `rag_bgem3` là `None`.

81. **`schema_problems()` là hàm THUẦN, tách khỏi client Qdrant.** 12 ca lệch test
    được trong `make test` (~3 giây) chứ không cần server. Hàng rào quan trọng nhất
    của một tầng không nên chỉ test được khi Docker đang chạy.

82. **`upsert` gọi provider MỘT lần, không hai.** Đo được: sparse thêm **8,8 giây
    trên 389** (+2,3%). Gọi `embed_documents()` rồi gọi tiếp một hàm sparse riêng
    sẽ là **+380 giây** — trả gấp đôi tiền forward pass cho đúng một kết quả. Đây
    là chỗ quyết định "một forward pass" của `W2-01` (quyết định 66) trả nợ.

83. **Xác nhận dense không đổi một chữ số sau khi đổi đường ghi.** `_embed_batch`
    → `embed_documents_hybrid` là đường code khác với `embed_documents` của
    `W2-01`. So lại trên 15.814 chunk thật: **15/15 metric không lệch, 0/209 câu
    đổi điểm**. Chỉ so bảng metric tổng thể thì hai lần chạy khác nhau vẫn có thể
    trùng số do bù trừ — nên phải so cả `*-per-query.jsonl`. Hạ tầng thêm ở
    `TD-11` dùng lại được ở đây.

84. **Bỏ `zip(strict=True)` thì phải thay bằng kiểm tra độ dài tường minh.**
    `upsert` giờ index theo offset (`dense[offset]`) nên lệch hàng sẽ **gán
    embedding cho sai chunk** thay vì báo lỗi. Có test integration chạy
    `batch_size=2` trên 3 chunk để đi qua nhiều lô rồi kiểm từng chunk.

85. **Sửa một bug tiềm ẩn: `rank` có lỗ khi bỏ point lỗi.** Bản cũ dùng
    `enumerate(points, start=1)`, nên một point thiếu payload làm dãy rank thành
    `1, 2, 4`. nDCG/MRR đọc `rank` như vị trí thật nên sẽ tính sai âm thầm. Giờ là
    `len(results) + 1`, có test canh dãy liên tục cho **cả hai** nhánh.

86. **`retrieve_sparse()` tách khỏi `retrieve()`, không gộp thành một hàm
    "hybrid".** `W2-04` (RRF) cần hai danh sách xếp hạng *độc lập* để hợp nhất, và
    một hàm trả sẵn hybrid thì không tách được đóng góp của mỗi nhánh — thứ mà
    `RetrievedChunk.dense_score`/`sparse_score` tồn tại để giữ.

87. **Thang điểm hai nhánh KHÔNG so được với nhau.** Dense là cosine (đã chuẩn
    hoá, ∈ [−1, 1]); sparse là **dot product** của trọng số không âm nên không có
    trần. Đo thật trên cùng một truy vấn: dense 0,6682 vs sparse 0,2938. Đây là lý
    do `W2-04` phải hợp nhất theo **thứ hạng**, không theo điểm.

88. **Script migrate cố ý KHÔNG migrate tại chỗ, vì không thể.** Qdrant không cho
    thêm named vector vào collection đã tồn tại. Nên
    `scripts/migrate_collection.py` làm thứ nó làm được: chẩn đoán lệch schema và
    in đúng lệnh phải chạy (mã trả về 0/1/2/3). Đây là phiên bản cho **schema** của
    câu hỏi mà `fingerprint` trả lời cho **nội dung**. Phương án "copy dense cũ rồi
    chỉ tính thêm sparse" đã cân nhắc và bỏ: tính được sparse của BGE-M3 tức là đã
    tính lại dense, nên chẳng tiết kiệm gì.

89. **`HashingEmbeddingProvider` mặc định `sparse=False`, và đó là quyết định.**
    Lần đầu tôi để `True` và hai test W1 đỏ ngay (`test_provider_name_is_
    reproducible` + một test của `W2-01`) — đó là cách phát hiện `name` của provider
    đi vào **cache key của semantic chunker** (`chunking/semantic.py`) và vào MLflow.
    Bật sẵn sẽ vô hiệu cache chunk và làm mọi test W1 đổi nghĩa âm thầm. Có test
    hồi quy canh `embed_documents()` cho kết quả y hệt khi bật/tắt sparse.

90. **`as_qdrant()` trả `TypedDict`, không phải `dict[str, list[int] | list[float]]`.**
    Kiểu union đúng về cấu trúc nhưng làm `models.SparseVector(**payload)` không
    kiểm được kiểu — và chỗ đó là ranh giới giữa code của mình với client bên
    ngoài, đúng chỗ đáng để kiểu chặt. mypy `--strict` bắt được.

91. **Sparse của BGE-M3 là một phép CHỌN, không phải bag-of-words có trọng số.**
    Đo trên 3.000 chunk: **95,9 entry/chunk** (p50 100 · p95 147 · max 195 ·
    **min 3**), mật độ 0,0384% của vocab 250.002. Chunk có p50 **218 token** nên
    ReLU loại khoảng **55%** token.

92. **`min = 3 entry` là cảnh báo cho `W2-04`.** Có chunk chỉ còn 3 token dương,
    tức nhánh sparse gần như không tìm ra được nó bằng bất kỳ truy vấn nào. Mặt bù
    này khớp đặc tính đo được trong test: **không trùng token thì sparse trả rỗng**,
    trong khi dense vẫn đoán được. Hai nhánh hỏng theo hai kiểu khác nhau — đó là
    lý do RRF đáng làm chứ không phải chỉ để có thêm một dòng trong CV.

93. **Filter metadata phải áp cho CẢ HAI nhánh ở tầng Qdrant.** Có test riêng cho
    nhánh sparse: thiếu nó thì `W2-06` (cô lập tenant) có một lỗ đúng bằng nhánh
    sparse, và lỗ đó không lộ ra ở bất kỳ test dense nào.

94. **Thêm sparse index làm p50 truy hồi dense tăng 23,7 → 31,5 ms (+33%), tái
    lập 3 lần — nhưng p95 gần như không đổi** (46,0 → 46,6 ms). Con số này cho biết
    cấu trúc của độ trễ: p95 bị chi phối bởi forward pass embed truy vấn của BGE-M3
    (biến động lớn), p50 phản ánh phép tìm trong Qdrant. ⚠️ **Chưa tách được** chi
    phí của sparse index khỏi trạng thái segment sau khi vừa build lại (8 segment,
    Qdrant tối ưu ở background). Với ngưỡng `G2` p95 < 3500 ms thì 46,6 ms còn rất
    nhiều chỗ nên chưa đáng đào — nhưng phải đo lại ở `W2-04`.

### Vấn đề đang mở

- **`W2-03` là việc tiếp theo.** `retrieve_sparse()` chạy được nhưng **chưa** là
  `Retriever`, nên eval harness chưa đo được nó. Đến `W2-03` mới có **số** cho câu
  "sparse đóng góp gì", và phải kèm `p`/CI qua `make eval-compare`.
- ⚠️ Nhánh BM25 **thô** của `W2-03` mới cần `Modifier.IDF`; nhánh BGE-M3 thì
  không (quyết định 80). Hai nhánh sparse khác nhau về bản chất đầu vào.
- ⚠️ `W2-04` hợp nhất theo **thứ hạng**, không theo điểm (quyết định 87).
- Collection trong Qdrant: `rag_baseline` (768-d), `rag_chunk550`,
  `rag_chunk550nb55`, `rag_bgem3` (1024-d + sparse). Chỉ `rag_bgem3` là hybrid.
- `TD-13` (golden set review bằng model) vẫn mở và vẫn là nợ đáng nhất.
- Các mục còn lại không đổi: `TD-05`, `TD-08`, `TD-10`, `TD-14`…`TD-17`, `W0-02`.

### Nếu phiên sau bắt đầu từ đây

1. `make lint && make test` · `make up && make test-integration`.
2. Đọc `reports/tasks/w2-02-qdrant-hybrid.md` §3 (sparse trên dữ liệu thật, và vì sao
   `min = 3` quan trọng) và §7 (thứ chưa làm).
3. `W2-03`: bọc `retrieve_sparse()` thành một `Retriever` để eval harness chạy
   được, rồi đo trên golden set. DoD là truy vấn từ khoá lạ mà dense miss thì
   sparse hit — phải có ca exact-ID lookup.
4. Dùng `HashingEmbeddingProvider(..., sparse=True)` cho test, không cần GPU.

---

## Phiên 2026-08-20 (tiếp) · Tuần 2 — `W2-03` sparse retriever

**Mục tiêu phiên:** cho nhánh sparse một hình dạng mà eval harness đo được, rồi
đo. Trước phiên này mọi phát biểu về sparse đều là suy luận từ thiết kế.

**Kết quả:** xong, và ra **hai câu trả lời ngược nhau** — sparse kém dense trên
golden set (có ý nghĩa thống kê) nhưng thắng áp đảo ở tra mã tài liệu
(`p = 4,8e-07`). Con số đáng nhất là con số thứ ba: **trần của `W2-04` là
`hit_rate@10` 0,7033** so với dense 0,6268.

### Lệnh để kiểm tra trạng thái hiện tại

```bash
make test                                   # 697 unit (16 ca branch, thuần)
make up && make test-integration            # 73 integration (14 ca sparse)
make test-gpu                               # 15 gpu (4 ca BGE-M3 thật)
```

Đo lại (index không cần build lại — `W2-02` đã ghi cả hai loại vector):

```bash
make eval-retrieval BUNDLE=bgem3 MODE=dense  RUN=bgem3
make eval-retrieval BUNDLE=bgem3 MODE=sparse RUN=bgem3-sparse
make eval-compare BASE=bgem3 CAND=bgem3-sparse
make known-item BUNDLE=bgem3                # ~2 phút, seed 20260820
```

### Bảng tiến độ phiên

| Việc | Trạng thái | Ghi chú |
|---|---|---|
| `QdrantSparseRetriever` bọc store | ✅ | Một kết nối, hai nhánh — xem quyết định 94 |
| `build_branch()` chọn nhánh | ✅ | `hybrid` báo "chưa cài", không báo "tên sai" |
| `--retrieval-mode` + `MODE=`/`RUN=` | ✅ | Nhánh **không** vào `IndexConfig` (quyết định 96) |
| Đo trên golden set kèm `p`/CI | ✅ | Sparse kém hơn, 12/15 metric có ý nghĩa |
| DoD: từ khoá lạ dense miss sparse hit | ✅ | Phép đo riêng, `golden_v1` không đo được |
| Hàng rào băm nhãn trong `compare.py` | ✅ | Ngoài DoD — hố im lặng, quyết định 98 |
| `scripts/known_item_probe.py` | ✅ | Tiêu chí kiểm bằng so chuỗi, không cần nhãn người |
| `W2-03` | ✅ | `reports/tasks/w2-03-sparse-retriever.md` |
| `W2-04` RRF | ⬜ việc tiếp theo | Trần đã biết: 0,7033 |

**Tổng kết:** 697 unit (+31) + 73 integration (+14) + 15 gpu (+4) test xanh ·
`ruff` + `mypy --strict` sạch trên 89 file (+ `scripts/` qua ruff).

### Quyết định kỹ thuật phát sinh trong phiên

*(tiếp số từ 93)*

94. **`QdrantSparseRetriever` BỌC store, không phải store thứ hai.** Nó giữ một
    tham chiếu tới `QdrantDenseRetriever` đang mở kết nối. Ba lý do: một
    `QdrantClient` cho cả hai nhánh; `W2-04` sẽ bọc **cùng** store đó nên RRF
    không phải tự đồng bộ hai đối tượng và tự tin rằng chúng trỏ vào cùng
    collection; và điểm mỗi nhánh vẫn về đúng ô `dense_score`/`sparse_score`.

95. **`build_branch(store, "dense")` trả về CHÍNH `store`, không bọc thêm.**
    `retriever.name` đi vào mọi report, nên bọc thêm một lớp chỉ để đối xứng sẽ
    làm mọi số cũ của W1/`W2-01`/`W2-02` không so được nữa.

96. **Nhánh truy hồi KHÔNG vào `IndexConfig`.** Nó không quyết định vector nào
    được ghi nên không thuộc `fingerprint` — và một trường nằm trong
    `IndexConfig` mà ngoài `fingerprint` là thứ phải giải thích lại mỗi lần đọc
    (đã có hai: `device`, `batch_size`). Nó là tham số của lần **đo**: cờ
    `--retrieval-mode`, biến `MODE=`, và ở `W2-07` là một chiều của ma trận.
    `RUN=` tách khỏi `BUNDLE=` để hai nhánh của cùng một index không ghi đè báo
    cáo của nhau.

97. **`QdrantSparseRetriever.__init__` chết ngay nếu provider không sinh sparse.**
    `_eval_against_index` quét toàn bộ chunk của corpus để ánh xạ span **trước**
    khi truy vấn câu đầu, nên lỗi ở truy vấn đầu là lỗi đến sau vài giây quét vô
    ích — với grid của `W2-07` thì là sau vài phút. Cùng lý lẽ với
    `ensure_collection` ở `W2-02`.

98. ⭐ **Thêm `QueryScore.relevant_digest` — hàng rào thứ ba của `compare.py`.**
    Hai hàng rào cũ canh *số* nhãn và *tập truy vấn*; không cái nào thấy được ca
    "cùng số nhãn, khác nhãn". Ca đó không phải giả thuyết: eval harness lấy
    `fetch_doc_chunks` bằng `getattr`, nên một retriever thiếu method đó **lặng
    lẽ** rơi về nhãn ghi trong file. Lớp bọc `W2-03` là chỗ đầu tiên trong dự án
    có thể rơi vào hố đó. Băm sha256 của **tập** `chunk_id` (16 hex, không phụ
    thuộc thứ tự). Đã chạy thật: 209/209 câu có băm, **0 lệch**. Đây là biến thể
    thứ ba của cùng một bài học — `TD-11` (im lặng cắt text), `W2-02` (im lặng bỏ
    nửa index), giờ là im lặng đổi nhãn.

99. **Khi băm lệch thì từ chối TOÀN BỘ, không lọc bỏ câu lệch rồi so phần còn
    lại.** Lọc theo kết quả là đúng cái tự chọn mẫu mà hàng rào #2 của module cấm.
    Câu **thiếu** băm (file trước `W2-03`) tính là "không biết" chứ không phải
    "khớp", và có cảnh báo nói rõ hàng rào đang tắt — một hàng rào tắt mà không ai
    biết thì tệ hơn không có.

100. **Chuyển tiếp `fetch_doc_chunks` TƯỜNG MINH, không `__getattr__`.** Một
     wrapper "trong suốt" thì mypy không kiểm được gì và hố ở quyết định 98 mở lại
     ngay lần đổi tên method tiếp theo ở store. Có test canh đúng phép thử
     `getattr` mà harness thực sự làm, và một test canh **không** có chuyển tiếp
     bao trùm (`not hasattr(branch, "upsert")`).

101. ⭐ **`golden_v1` KHÔNG đo được DoD của `W2-03`, nên phải dựng phép đo riêng.**
     209 câu đều là câu hỏi tự nhiên do LLM sinh, không có câu nào tra mã. Nếu
     chỉ báo cáo bảng golden set thì kết luận sẽ là "sparse kém hơn, xong" — đúng
     về số và sai về việc. `scripts/known_item_probe.py`: lấy chuỗi **xuất hiện
     nguyên văn** trong corpus làm truy vấn, rồi coi kết quả là đúng khi nội dung
     nó **chứa chính chuỗi đó**. Tiêu chí kiểm bằng so chuỗi → không cần nhãn
     người và không có chỗ cho phán đoán chen vào.

102. **Không chấm theo một `chunk_id` nguồn định trước.** Một mã xuất hiện ở 2–3
     chunk thì cả 2–3 đều là câu trả lời đúng; chấm chỉ một cái là hạ điểm cả hai
     nhánh một cách vô cớ. Nên tiêu chí là "có chứa chuỗi", không phải "đúng
     chunk".

103. ⭐ **SPARSE HỌC ĐƯỢC KHÔNG PHẢI INDEX KHỚP ĐÚNG** — phát hiện đổi kế hoạch.
     25/51 mã **không nhánh nào** tìm ra. Tokenizer giải thích hết: mã tìm được
     luôn có một mảnh subword tự nó là từ hiếm (`VIE-01` → `['▁','VIE','-01']`,
     `ASEAN-518` → `['▁ASEAN','-5','18']`); mã miss rã thành một chữ cái cộng khối
     ba chữ số (`P171645` → `['▁P','171','645']`) và `171`/`645` có mặt khắp
     15.814 chunk thống kê. Đo được: mã hit dài 7,5 ký tự / 4,2 chữ số, mã miss
     8,9 / 5,2. Hệ quả: **RRF (`W2-04`) và reranker (`W2-05`) đều không sửa được**
     — cả hai chỉ xếp lại thứ tự những gì đã tìm ra. → `TD-18`, và ưu tiên đo
     phương án rẻ (filter payload khớp chuỗi) trước phương án đắt (BM25 thô mức
     từ, cần đổi schema + build lại index).

104. ⭐ **Ghi lại trần của `W2-04` TRƯỚC KHI làm `W2-04`.** Hợp hai nhánh cho
     `hit_rate@10 = 0,7033` vs dense 0,6268. Ghi bây giờ thì lúc đó không tự diễn
     giải kết quả theo hướng có lợi: nếu RRF ra dưới 0,6268 thì nó đang **làm hại**
     so với dense một mình, và đó là kết quả phải báo cáo chứ không phải tinh chỉnh
     `k` cho tới khi số đẹp.

105. **Đo tách theo `lang`, không chỉ theo `category`.** `cross_lingual` là chỗ
     sparse chết hẳn (1↔19, và **0 câu cả hai** — câu tiếng Việt trên tài liệu
     tiếng Anh không có token nào trùng); nhưng trên truy vấn **tiếng Anh sparse
     thắng** (9↔4). Gộp lại thì hai hiệu ứng trái chiều triệt tiêu nhau và bảng
     `W2-08` sẽ nói "sparse kém" mà không nói được kém ở đâu.

106. **Hai giả định của tôi bị phản chứng, và cả hai được GIỮ trong bộ test.**
     (a) "dense lẫn giữa những mã gần giống nhau" — sai: trên corpus 7 chunk cả
     hai nhánh đặt đúng mã ở hạng 1, dense với biên rõ (0,6947 vs 0,6411). Với 6
     đối thủ thì đó là điều phải chờ đợi. Cái tôi **không** làm: đổi assertion
     thành `rank_sparse <= rank_dense` — nó sẽ *pass* vì hai bên bằng nhau và đọc
     như một chiến thắng. Thay vào đó có
     `test_dense_picks_it_too_and_that_is_the_honest_result`. (b) "không trùng
     token thì sparse trả rỗng" — đúng trên 3 chunk, sai trên 15.814: sparse trả
     đủ 20 kết quả cho **cả 209 câu**. Giới hạn thật của sparse ở đây là **xếp
     hạng sai**, không phải **không trả gì**.

107. **Ghi lại một chỗ CẢ HAI nhánh cùng sai, có test.** "thống kê ba tháng cuối
     năm 2024" = quý bốn; cả dense lẫn sparse đặt chunk quý **ba** ở hạng 1. Nó
     nói `W2-04` không cứu được gì ở đó: hợp nhất hai danh sách cùng sai một kiểu
     thì vẫn sai. Chỗ đó là việc của `W2-05`.

108. ⚠️ **Tìm trong index sparse tốn 97,8 ms vs dense 17,8 ms (5,5×).** Tách được
     nhờ đo riêng: embed truy vấn 12,6 ms (một forward pass, **dùng chung** cả hai
     nhánh — phần lợi còn lại của quyết định 66), tìm dense 17,8, tìm sparse 97,8.
     Vector sparse của truy vấn chỉ có 31,8 entry trung bình, vậy mà 32 posting
     list trên 15.814 point tốn 98 ms. Từ `W2-04` trở đi nhánh sparse là thành
     phần **nặng nhất** của đường truy hồi — không phải model embedding. Chưa tối
     ưu gì (chưa thử `on_disk`, chưa thử ngưỡng trọng số phía tài liệu) nên đọc là
     "chi phí hiện tại", không phải "chi phí tối thiểu".

109. **Thêm `store.verify_schema()` vào `_eval_against_index`.** Docstring của hàm
     đó vốn đã lo đúng chuyện này ("eval sẽ đo một index được build bằng model
     khác, và không có gì báo lỗi cả") nhưng không làm gì để chặn. Giờ kiểm
     **trước** khi quét span. Có lợi cho cả nhánh dense.

### Vấn đề đang mở

- **`W2-04` là việc tiếp theo.** Trần `hit_rate@10` = **0,7033**; hợp nhất theo
  **thứ hạng** không theo điểm (quyết định 87); đo lại độ trễ (~128 ms/truy vấn).
- ⚠️ `TD-18` mới: 25/51 mã tài liệu không tìm được bằng bất cứ nhánh nào, và RRF
  lẫn reranker đều không sửa được (quyết định 103). Đo phương án rẻ trước.
- ⚠️ Mọi số §3–§4 của report đo trên `golden_v1` — vẫn **review bằng model**
  (`TD-13`). So sánh *tương đối* hợp lệ; con số *tuyệt đối* thì không được gọi là
  "human-verified".
- Collection trong Qdrant không đổi: `rag_baseline`, `rag_chunk550`,
  `rag_chunk550nb55`, `rag_bgem3` (chỉ `rag_bgem3` là hybrid). **Không phải build
  lại index cho `W2-03`** — `W2-02` đã ghi cả hai loại vector.
- `*-per-query.jsonl` giờ có thêm trường `relevant_digest`, nên file của lần chạy
  cũ (`baseline`, `chunk550`) không có nó và `compare.py` sẽ cảnh báo là hàng rào
  đang tắt khi so với chúng. Chạy lại `make eval-retrieval` cho những run còn cần
  so, hoặc chấp nhận cảnh báo.
- Các mục còn lại không đổi: `TD-05`, `TD-08`, `TD-10`, `TD-13`…`TD-17`, `W0-02`.

### Nếu phiên sau bắt đầu từ đây

1. `make lint && make test` · `make up && make test-integration`.
2. Đọc `reports/tasks/w2-03-sparse-retriever.md` §4 (trần của RRF và phân rã theo
   nhóm/ngôn ngữ) và §6 (vì sao sparse không phải index khớp đúng).
3. `W2-04`: RRF `k=60`. Trần đã biết là 0,7033 — nếu ra dưới 0,6268 thì RRF đang
   làm hại, và đó là kết quả phải báo cáo.
4. Đo bằng `make eval-retrieval BUNDLE=bgem3 MODE=hybrid RUN=bgem3-rrf` rồi
   `make eval-compare BASE=bgem3 CAND=bgem3-rrf`. `build_branch` đã có chỗ cho
   `hybrid` (hiện raise `NotImplementedError` chỉ đúng vào `W2-04`).

---

## Phiên 2026-08-20 (tiếp) · Tuần 2 — `W2-04` RRF fusion

**Mục tiêu phiên:** hợp nhất hai nhánh bằng RRF và xem có tiến được tới trần
`hit_rate@10 = 0,7033` mà `W2-03` để lại hay không.

**Kết quả:** không tiến tới trần, và ở mặc định của bài báo gốc (`k=60`) nó còn
**kém dense một mình có ý nghĩa thống kê**. Quét `k` ra cấu hình thắng `k=1`
(`hit_rate@10` 0,6555) nhưng chỉ 3/15 metric đạt ý nghĩa. Phần đáng nhất của
phiên lại không phải RRF: kiểm một con số không khớp đã tìm ra **bug hiệu năng
64 ms/lần gọi** làm sai hai con số đã công bố ở `W2-02` và `W2-03`.

### Lệnh để kiểm tra trạng thái hiện tại

```bash
make test                                   # 771 unit (40 ca RRF thuần + 34 hybrid)
make up && make test-integration            # 102 integration (25 ca hybrid)
make test-gpu                               # 17 gpu (có test hồi quy bug 64 ms)
```

Đo lại (index **không** cần build lại nữa — đã build ở phiên này sau khi sửa bug):

```bash
make eval-retrieval BUNDLE=bgem3 MODE=hybrid RUN=bgem3-rrf-k1-c20 \
  RRF_ARGS="--rrf-k 1 --candidate-k 20"
make eval-compare BASE=bgem3 CAND=bgem3-rrf-k1
make known-item BUNDLE=bgem3                # thêm --rrf-k 1 cho cột hybrid k=1
```

### Bảng tiến độ phiên

| Việc | Trạng thái | Ghi chú |
|---|---|---|
| `reciprocal_rank_fusion` hàm thuần | ✅ | Không nhận điểm — có test canh chữ ký |
| `QdrantHybridRetriever` | ✅ | Một embed, một request HTTP |
| `build_branch(..., "hybrid", **options)` | ✅ | Tham số sai nhánh là **lỗi**, không bị bỏ qua |
| Quét `k` × `candidate_k` × weights | ✅ | 11 lần chạy — quyết định 112 |
| Đối chiếu với `Fusion.RRF` của Qdrant | ✅ | Qdrant dùng `k=1`, ta trùng khít |
| Tìm + sửa bug `len(tokenizer)` 64 ms | ✅ | Ngoài kế hoạch — quyết định 117 |
| Đính chính `W2-02` + `W2-03` | ✅ | Ghi tại chỗ, không xoá số sai |
| Known-item cho cả 3 nhánh | ✅ | Hybrid giữ phủ, mất thứ hạng |
| `W2-04` | ✅ | `reports/tasks/w2-04-rrf.md` |
| `W2-05` reranker | ⬜ việc tiếp theo | Biến `recall@20` 0,6754 thành thứ hạng |

**Tổng kết:** 771 unit (+73) + 102 integration (+29) + 17 gpu (+2) test xanh ·
`ruff` + `mypy --strict` sạch trên 94 file.

### Quyết định kỹ thuật phát sinh trong phiên

*(tiếp số từ 109)*

110. ⭐ **Ghi dự đoán TRƯỚC khi đo, và giữ lại khi nó sai.** Trước khi cài RRF tôi
     lập luận: RRF xen kẽ hai danh sách nên top-10 hợp nhất ≈ hợp của hai top-5 =
     **0,6268** = đúng bằng dense@10, tức "ngang bằng"; và chỗ RRF thắng sẽ là
     `hit_rate@1` (đồng thuận được cộng điểm). **Sai cả hai chiều**: `k=60` cho
     0,5742 (tệ hơn dự đoán) và `hit_rate@1` không cải thiện ở *bất kỳ* cấu hình
     nào. Lý do phần đầu sai: top-10 hợp nhất không phải hợp của hai top-5 mà bị
     chiếm bởi chunk *cả hai nhánh đều xếp hạng trung bình* — và chúng thường
     không liên quan.

111. ⚠️ **`k=60` — mặc định của bài báo gốc — là lựa chọn TỆ NHẤT trong khoảng đã
     quét**, và nó kém dense một mình *có ý nghĩa* (`hit_rate@5` `p = 0,014`,
     nDCG@10 CI95 không chứa 0). Nếu cài RRF theo mặc định, không quét `k`, rồi
     báo "đã làm hybrid search" thì đó là trình bày một **suy giảm** như một tính
     năng. Đây là lý do `--rrf-k` phải là cờ chứ không phải hằng số.

112. ⭐ **`k` là cần điều khiển chính và nó đơn điệu.** nDCG@10 theo `k` =
     1→2→5→10→60 cho **0,4557 → 0,4530 → 0,4443 → 0,4305 → 0,4021**. Cơ chế tính
     được: `k=60` cho chunk mà **cả hai** nhánh xếp hạng 3 (`2/63 = 0,0317`) đè
     lên chunk mà dense xếp **hạng 1** (`1/61 = 0,0164`) — gấp 1,9×. Với `k=1` thì
     hai bên bằng nhau. Tức `k` lớn cho **đồng thuận yếu lật đổ tín hiệu mạnh**,
     và trên tập này đồng thuận yếu chủ yếu là nhiễu.

113. **`candidate_k` chỉ là cần điều khiển khi `k` lớn.** Ở `k=60`: c20/c50/c100
     cho `hit_rate@10` 0,6364/0,5742/0,5024 — chênh 13 điểm. Ở `k=1`:
     0,6555/0,6555/0,6555 — không lệch một chữ số. `k` nhỏ làm ứng viên hạng sâu
     gần vô giá trị nên độ sâu pool không còn ý nghĩa. Hệ quả: **với `k=1` chọn
     `candidate_k` nhỏ nhất**, nó miễn phí về chất lượng và rẻ hơn về độ trễ.

114. **Hai test đơn vị tôi viết để biện minh `candidate_k` sâu có số học ĐÚNG
     nhưng tiên đề SAI.** `test_deep_agreement_beats_shallow_solo` chứng minh
     chunk ở hạng 45 của dense + hạng 3 của sparse được đẩy lên trên chunk chỉ
     dense tìm ra ở hạng 2 — điều đó đúng và vẫn đúng. Điều test không nói được:
     những chunk ấy thường **không liên quan**. Giữ cả hai test (chúng đặc tả đúng
     hành vi của hàm) và thêm ghi chú trỏ vào số đo — một test đúng về hành vi vẫn
     có thể bị đọc thành một lời khuyên sai.

115. **Weighted RRF không đáng quét ở `W2-08`, và biết được điều đó từ số học.**
     Muốn lật một đồng thuận **cùng độ sâu** cần tỉ lệ trọng số **> 30,5:1**, vì
     cân dense lên cũng cân luôn phần dense *của chunk đồng thuận*: `a = w₀/61` vs
     `b = (w₀+w₁)/63`. Lần đầu tôi tính sai chỗ này và test đỏ — nên nó thành một
     test (`test_same_depth_agreement_needs_an_absurd_weight_to_flip`). Đo thật
     2:1 cho +0,0287, khớp.

116. ⭐ **Đối chiếu với `Fusion.RRF` của Qdrant, và Qdrant dùng `k = 1`.** Suy `k`
     của nó từ chính điểm nó trả về: cả 7 chunk khớp `1/(1+rank)` tới 6 chữ số.
     Bản của ta ở `k=1` cho **cùng tập điểm** (`rel=1e-6` — trường `score` của
     Qdrant là float32, nó trả 0,6666667 cho chỗ ta tính 0,6666666666666666). Tự
     cài không có nghĩa là tự tin: trùng khít với một bản độc lập là bằng chứng
     mạnh hơn mọi số tính tay tôi tự đặt ra. **Và vẫn phải tự cài** vì `k` của
     Qdrant không cấu hình được, mà quyết định 112 cho thấy `k` là cần điều khiển
     quan trọng nhất.

117. ⭐⭐ **BUG 64 ms, tìm ra vì các con số KHÔNG CỘNG LẠI ĐÚNG.** `W2-03` §8 báo
     "tìm sparse 97,8 ms vs dense 17,8 ms"; `W2-04` thấy hybrid — làm *nhiều việc
     hơn* — có p50 31,3 ms. Hai số đó không thể cùng đúng. Phân rã đầy đủ trong
     một tiến trình: dense lệch −7,8 ms, hybrid lệch −4,1 ms (đều trong nhiễu),
     **sparse lệch +81,7 ms**. Khác biệt duy nhất: `retrieve_sparse` kiểm
     `self.writes_sparse` ở **mỗi lần gọi**, còn hybrid kiểm một lần lúc dựng. Và
     `writes_sparse` đọc `sparse_vocab_size` = `len(self.model.tokenizer)`, tức
     **dựng lại dict 250.002 phần tử — 63,9 ms**. Sửa bằng cách **nhớ kết quả**,
     không đổi sang `tokenizer.vocab_size` (nhanh nhưng khác nghĩa: không tính
     token thêm vào, và chặn trên sai thì hỏng im lặng).

118. **Cách phát hiện quyết định 117 khác ba lần trước.** `TD-11`, `W2-02`,
     `W2-03` đều được phát hiện bằng cách **kiểm một bất biến**. Cái này được phát
     hiện bằng cách **buộc các con số cộng lại đúng**. Ba harness khác cấu trúc
     cho ba câu trả lời lệch nhau 2–6× cho cùng một việc — khi phương sai giữa các
     cách đo lớn hơn hiệu ứng muốn quy kết thì phải đo lại, chứ không phải chọn
     con số vừa mắt. Đã bỏ ba lần đo mới để tìm ra chỗ thiếu.

119. ⚠️ **Hai con số đã công bố bị sai, và đã đính chính TẠI CHỖ chứ không xoá.**
     (a) `W2-03` §8: tìm sparse thật ra **15,4 ms**, **rẻ hơn** tìm dense 28,7 ms;
     gửi cả hai trong một request batch tốn **30,2 ms** = bằng dense một mình, vì
     Qdrant chạy song song. Kết luận cũ "nhánh sparse sẽ là thành phần nặng nhất
     của đường truy hồi" là sai, và dự đoán "~128 ms mỗi truy vấn hybrid" cũng
     sai. (b) `W2-02` "+8,8 s (+2,3%)": 124 lô × 64 ms ≈ 7,9 s là bug. Build lại
     sau khi sửa: **389,2 s → 379,1 s**, tức **nhanh hơn** cả dense-only 380,4 s
     của `W2-01` — sparse phía ghi **miễn phí hoàn toàn**. Giữ số sai kèm đính
     chính vì con số sai và lý do nó sai đều đáng đọc.

120. **Index bit-identical sau khi build lại** (cùng config, chỉ khác đã sửa bug):
     209/209 câu, **0 câu đổi điểm**, 0 câu đổi nhãn, fingerprint vẫn
     `0eaaf9265487eabb`. Nên mọi con số của phiên đo trên cùng một index.

121. **Hàm hợp nhất KHÔNG nhận điểm, và có test canh chính chữ ký đó.** Dense cho
     cosine ∈ [−1,1], sparse cho dot product không trần; mọi phép chuẩn hoá đều
     phải chọn một cửa sổ, tức đưa vào một tham số ẩn phụ thuộc kết quả. Cách chắc
     nhất để giữ tính bất biến theo thang là không cho điểm đi vào hàm —
     `test_only_rank_matters_not_score` khẳng định `"scores" not in params`.

122. **Tie-break là bốn quy tắc, mỗi quy tắc một test.** Điểm bằng nhau xảy ra
     *thường xuyên* (một chunk hạng 3 của dense và một chunk khác hạng 3 của sparse
     có cùng điểm). Thứ tự: điểm → `best_rank` → danh sách nào tìm ra trước → khoá
     theo chữ. Quy tắc thứ ba là chỗ **dense được đặt trước** vì `W2-03` đo được nó
     mạnh hơn; quy tắc thứ tư hầu như không bao giờ tới nhưng nó biến "gần như xác
     định" thành "xác định".

123. **Trùng khoá trong CÙNG một danh sách là lỗi, không im lặng bỏ bớt.** Nó nghĩa
     là hai point Qdrant mang cùng `chunk_id` — đúng thứ mà `W1-08` bỏ ba tầng công
     sức ra để chống. Bỏ bớt sẽ làm điểm RRF trông hợp lý trong khi index có bản
     trùng.

124. **Tham số RRF truyền cho nhánh không nhận là LỖI.** `--rrf-k 10
     --retrieval-mode dense` mà im lặng chạy tiếp sẽ vào bảng `W2-08` như một dòng
     hợp lệ trong khi nó không đo cái nó nói. `build_branch` phân biệt "không đặt
     cờ" (`None`, hợp lệ) với "đặt cờ cho nhánh sai" (raise).

125. **Corpus test phải KHÔNG có điểm trùng cho phép đối chiếu.** Bản đầu của
     `test_hybrid_retriever.py` đỏ *ngẫu nhiên*: corpus có điểm trùng thật (hai
     chunk chia đúng cùng tập token thì điểm sparse bằng nhau; chunk không trùng
     token nào thì điểm dense bằng 0), và thứ tự trong nhóm bằng điểm không thuộc
     hợp đồng của Qdrant — đường `prefetch` và đường `query` thường không hứa cùng
     thứ tự. Đã dựng lại corpus (6 doc chia sẻ số token giảm dần) cho một truy vấn
     phân biệt hoàn toàn ở cả hai nhánh, và mã lạ `GSO-2024-XII` đặt vào doc **đã**
     có token trùng để không tạo doc điểm-0 thứ hai.

126. **Test "một forward pass" phải canh đúng tầng.** Bản đầu ở tầng integration
     đếm cả `embed_query` và **đỏ** — vì `HashingEmbeddingProvider.
     embed_query_hybrid` gọi lại nó *bên trong*. Nội bộ provider là hợp đồng của
     provider, không phải việc của retriever. Chia làm hai: integration đếm số lần
     gọi provider (= 1), và một test GPU đếm số `_forward` bên trong
     `BgeM3EmbeddingProvider` (= 1). Không nới assertion cho nó xanh.

127. **Known-item: hybrid giữ vùng phủ, làm hỏng thứ hạng.** hit@10 0,5098 = sparse
     (3↔3, `p = 1` — phủ giống nhau về thống kê) nhưng hit@1 **0,3529 → 0,0980**,
     hạng trung vị 1 → 4. RRF trọng số đều **không biết nhánh nào đáng tin cho
     truy vấn nào**: trên truy vấn tra mã, dense đóng góp toàn nhiễu (hit@10 =
     0,0784) mà vẫn ngang quyền. `k=1` giảm nhẹ thiệt hại (hạng trung vị 4 → 2)
     nhưng không xoá được.

128. ⭐ **Kết luận kiến trúc: hybrid là bộ SINH ứng viên tốt, bộ XẾP HẠNG cuối tệ.**
     Mở rộng phủ có ý nghĩa (`recall@20` +0,0431, CI không chứa 0; known-item giữ
     nguyên hit@10) nhưng không đưa câu trả lời lên đầu (`hit_rate@1` đứng im;
     known-item MRR tụt một nửa). Đó đúng là hình dạng bài toán `W2-05` tồn tại để
     giải — nên `W2-05` là việc tiếp theo, không phải tinh chỉnh thêm RRF.

129. **Thứ reranker KHÔNG giải được: chọn nhánh theo truy vấn.** Tổn thất ở quyết
     định 127 đến từ trọng số cố định trên một tập truy vấn không đồng nhất. Một bộ
     định tuyến ("truy vấn này là tra mã → tin sparse") giữ được cả hai, nhưng nó
     cần một bộ phân loại truy vấn, tức thêm một thứ phải eval — câu hỏi của W3+,
     không phải một cờ thêm vào `W2-04`.

130. **Tách `build_filter` và `points_to_chunks` thành hàm module.** `W2-04` dựng
     hai truy vấn trong một request batch nên nó cần cả hai mà không đi qua
     `retrieve()`/`retrieve_sparse()` — gọi hai method đó sẽ embed truy vấn hai
     lần. Đây là cái seam đúng, và `W2-05` sẽ dùng lại nó.

### Vấn đề đang mở

- **`W2-05` là việc tiếp theo.** Hybrid `k=1` giao cho nó `recall@20` **0,6754**
  (dense 0,6324) và `hit_rate@1` **0,3397** (không đổi). Khoảng cách giữa hai số
  đó là phần việc của reranker.
- **Chốt cho `W2-08`:** mặc định `k=1, candidate_k=20`; quét `k ∈ {0,1,2,5}`;
  **không** quét `weights` (quyết định 115).
- ⚠️ `TD-18` (khớp đúng cho mã tài liệu) nặng thêm: RRF **không** sửa được nó và
  còn làm thứ hạng của nhánh duy nhất giải được nó tệ đi (quyết định 127).
- ⚠️ Mọi con số vẫn đo trên `golden_v1` — **review bằng model** (`TD-13`). Và mức
  cải thiện của cấu hình thắng (+2,9 điểm) **nhỏ hơn** ngưỡng phân giải (≥ 6
  điểm), nên thứ tự giữa `k=1`, `k=2`, `k=5` **không** phân biệt được.
- Collection trong Qdrant không đổi: `rag_baseline`, `rag_chunk550`,
  `rag_chunk550nb55`, `rag_bgem3` (chỉ `rag_bgem3` là hybrid, đã build lại
  2026-08-20 sau khi sửa bug — nội dung bit-identical).
- Các mục còn lại không đổi: `TD-05`, `TD-08`, `TD-10`, `TD-13`…`TD-18`, `W0-02`.

### Nếu phiên sau bắt đầu từ đây

1. `make lint && make test` · `make up && make test-integration`.
2. Đọc `reports/tasks/w2-04-rrf.md` §3 (bảng quét `k`), §6 (bug 64 ms và cách tìm ra) và
   §8 (vì sao hybrid là bộ sinh ứng viên chứ không phải bộ xếp hạng).
3. `W2-05`: `bge-reranker-v2-m3`, rerank 50 → 6 trong < 400 ms trên GPU. VRAM: đã
   biết BGE-M3 chiếm ~3,3/8 GB, reranker thêm ~2,2 GB — còn vừa, nhưng `W0-06`
   vẫn chưa đo chính thức.
4. Đầu vào của reranker là nhánh hybrid `k=1, c=20`; `build_branch` đã có chỗ cho
   `reranked` (hiện raise `NotImplementedError` chỉ đúng vào `W2-05`).

---

## Phiên 2026-08-21 · Tuần 2 — `W2-05` cross-encoder reranker

**Làm gì:** `rag_core/reranking/` (`Reranker` ABC + `CrossEncoderReranker` cho
`bge-reranker-v2-m3`), `rag_core/retrieval/reranked.py` (`RerankedRetriever`),
`build_branch` đệ quy cho nhánh nền, 8 cờ CLI mới, `scripts/rerank_probe.py`,
`known_item_probe.py` thêm nhánh thứ tư, hai target Makefile.

**Kết quả một dòng:** `hit_rate@1` **0,3397 → 0,5598 (+22,0 điểm)**, 15/15 metric
có ý nghĩa — mức cải thiện lớn nhất của W2 sau chính BGE-M3. Nhưng **DoD 400 ms
không đạt** (524 ms cho 50 cặp), và hai kết quả kiến trúc quan trọng hơn cả con
số chính.

### Số đo chính

| run | hit@1 | hit@5 | hit@10 | hit@20 | nDCG@10 | recall@20 | p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 0,3397 | 0,5455 | 0,6268 | 0,6746 | 0,4442 | 0,6324 | 30,3 |
| hybrid k=1 c=20 | 0,3397 | 0,5742 | 0,6555 | 0,7177 | 0,4563 | 0,6770 | 31,3 |
| rerank c=20 | 0,5407 | 0,7033 | 0,7129 | 0,7177 | 0,6075 | 0,6770 | 232,8 |
| **rerank c=50** | **0,5598** | 0,7512 | 0,7703 | 0,7751 | **0,6481** | 0,7424 | 534,4 |
| rerank c=100 | 0,5789 | 0,7895 | 0,8134 | **0,8182** | 0,6736 | 0,7775 | 1044,3 |
| rerank dense-c50 | 0,5455 | 0,7273 | 0,7416 | 0,7464 | 0,6268 | 0,7121 | 538,0 |

Trần vùng phủ (đo **trước** khi làm): `hit_rate@50` của nhánh nền = **0,7799**.

### Quyết định kỹ thuật phát sinh trong phiên

131. **Đo trần vùng phủ trước khi cài bất cứ thứ gì.** `retrieval_eval` chỉ chấm
     ở `k ∈ {1,5,10,20}` nên nó không nói được `hit_rate@50` — mà đó chính là
     trần cứng của mọi phép xếp lại. `scripts/rerank_probe.py` tồn tại cho đúng
     việc đó. Không có số 0,7799 thì "+22 điểm" là một câu vô nghĩa: 22 trên bao
     nhiêu điểm còn lại?

132. **`RerankedRetriever` bọc một `Retriever` bất kỳ**, không bọc
     `QdrantDenseRetriever`. Đó là điều kiện để câu hỏi "reranker có cần hybrid
     không" hỏi được mà không sửa code — và câu trả lời (quyết định 138) là loại
     kết quả rất dễ không đi tìm.

133. **`build_branch` gọi lại chính nó** cho nhánh nền của `reranked`, nên mọi
     phép kiểm tham số của `W2-03`/`W2-04` áp dụng nguyên vẹn ở tầng dưới:
     `--rerank-base dense --rrf-k 1` vẫn nổ. `RERANK_OPTIONS` là tập tham số
     chỉ nhánh này nhận; phần còn lại chuyển xuống y nguyên.

134. **`top_n` là trần của lượt TRẢ VỀ, không phải của phép đo.** DoD nói "50 →
     6" nhưng đặt `--rerank-top-n 6` rồi chấm `recall@20` thì mọi metric @k > 6
     mất nghĩa. Mặc định `None`, và có test **ghim** cái bẫy đó thay vì sửa nó —
     hành vi là đúng theo thiết kế, cái thiếu là câu giải thích ở chỗ dễ gặp.

135. **`max_length` và `dtype` vào `name`, `batch_size` thì không.** Cùng lý lẽ
     `IndexConfig.fingerprint` (`W1-06`): cái nào đổi kết quả thì phải xuất hiện
     trong nhãn. Có test canh cả hai chiều — `max_length` **đổi** điểm,
     `batch_size` **không** đổi.

136. **`dtype="auto"` phân giải NGAY lúc dựng**, không để chữ "auto" trong `name`.
     Một nhãn ghi "auto" thì đọc log xong vẫn không biết lần chạy đó ở fp16 hay
     fp32 — mà hai cái cho điểm khác nhau.

137. **Chết khi điểm không hữu hạn.** `NaN` so với mọi thứ đều `False` nên
     `sorted` trả thứ tự **tuỳ ý mà không báo gì**, và hậu quả trên bảng metric
     trông y như "model kém". Chế độ hỏng thật của fp16.

138. ⭐ **Sau khi có reranker, tầng hybrid không còn đo được.** Cùng reranker
     c=50: nền dense vs nền hybrid cho **13/15 metric trong ngưỡng nhiễu**
     (`hit_rate@1` `p = 0,453`, `recall@20` CI95 [−0,0008, +0,0630]). Ở `W2-04`
     hybrid hơn dense **có ý nghĩa**. Hai tầng sửa **cùng một khuyết điểm** của
     dense và chồng lên nhau. **Vẫn giữ hybrid** vì nó miễn phí (534,4 vs 538,0
     ms), cả 15 metric vẫn cùng chiều, và `golden_v1` không đo được chỗ sparse
     thắng — nhưng thứ tự ưu tiên giữa hai tầng thì rõ hẳn.

139. ⭐⭐ **`TD-18` là bài toán TRUY HỒI, không phải biểu diễn.** Dự đoán D4 của
     tôi sai hẳn: known-item hit@1 **0,0980 → 0,5490**, hit@10 0,4706 → 0,6471,
     và nó **thắng cả sparse** (0,3529, `p = 0,0391`, 1↔8). Vocab subword phá
     việc **truy hồi** một mã — điểm sparse là tích vô hướng trên *túi* subword
     nên `['▁P','171','645']` mất hết thông tin thứ tự/liền kề — nhưng **không**
     phá việc **nhận ra** nó, vì cross-encoder có attention trên cả cặp. Reranked
     tìm 33/51 mã vs 26/51 của hợp dense+sparse ở top-10: **7 mã cứu từ vùng sâu
     pool**. Nợ thu hẹp thành **35% mã không vào được pool 50**.

140. **DoD 400 ms không đạt, và tôi không sửa DoD cho khớp số đo.** 50 cặp tốn
     **524 ms**; 400 ms mua được pool **38** ⚠️ *(đúng là **37** — sửa 2026-08-21 khi
     chạy lại probe; "38" là kết quả của việc mang hằng số đã làm tròn đi tính tiếp)*.
     Ba cần điều khiển tối ưu đều cạn và
     mỗi cái có số: `max_length` chỉ cắt **1/12.100 cặp** (p50 287 token vs trần
     512) nên hạ trần không giảm tính toán thật; fp16 lấy **3,52×** (1794,7 →
     510,1 ms) và đổi top-1 ở đúng 1/60 câu; `batch_size` **không mua được gì** —
     8/16 là nhiễu, 32/64 **tệ dần** vì `CrossEncoder.predict` gom batch theo độ
     dài đã sắp, nên batch lớn nhét cả pool vào một batch và tối đa hoá padding.
     Còn lại chỉ `candidates`, và nó là **trả bằng vùng phủ** chứ không phải tối ưu.

141. **`candidates` tách được hai loại metric, và có kiểm định.** 20→50 và 50→100
     đều làm `hit_rate@1` cải thiện **không có ý nghĩa** (`p = 0,219` và
     `p = 0,125`) nhưng `hit_rate@10`/`recall@20` thì có. Pool sâu mua **chất
     lượng danh sách**, không mua **chất lượng hạng nhất**. Điểm vận hành khuyến
     nghị cho `W4`: **c=20 ở 233 ms** — 91% mức lợi hạng nhất, 44% chi phí.

142. **CPU fallback tồn tại để code chạy đúng, không để phục vụ.** 19.446 ms cho
     pool 50 (chậm **38×**). `dtype="auto"` cố ý **không** chọn fp16 trên CPU —
     phần lớn CPU không có kernel fp16 nên PyTorch lùi về emulate và chạy *chậm
     hơn* fp32. Mất GPU thì cách đúng là **tắt tầng rerank** và lùi về hybrid ở
     31 ms, không phải chạy reranker trên CPU rồi để truy vấn treo 20 giây.

143. ⭐ **Hố im lặng thứ hai trong `compare.py`.** `precision@1` **bằng
     `hit_rate@1` từng chữ số** nhưng đi đường bootstrap, nên cùng một con số
     nhận hai kết luận trái nhau (McNemar `p = 0,125` "nhiễu" vs CI95 loại 0
     "khác biệt thật"). Người đọc bảng sẽ trích dòng nào thuận với mình. Đã route
     qua McNemar — và **bản sửa đầu tự tạo bug mới**: `"precision@10".startswith(
     "precision@1")` là `True` nên `precision@10` (nhận 0; 0,1; 0,2…) bị đẩy sang
     McNemar ngay lần chạy sau. Tách `BINARY_METRICS` khớp **đúng tên**, và test
     thứ hai canh **chính cái bẫy tiền tố** chứ không canh cách sửa.

### Ba lần đoán, ba lần thấp

Dự đoán ghi trước khi đo: 3/7 sai, và **cả ba lần đều thấp**.

* độ trễ: đoán 150–300 ms → thật **1794,7 ms** (fp32). Sai về **bậc**, vì tôi
  nghĩ về reranker như *một* forward pass trong khi nó là `candidates` forward
  pass — điều tôi đã viết đúng trong docstring của `Reranker` rồi vẫn đoán như
  thể nó không tồn tại. Số phép tính nói ngay: 302,0M tham số encoder × 14.350
  token × 2 = **8,67 TFLOP** cho *một* truy vấn.
* fp16: đoán 1,7–2,2× → thật **3,52×**.
* `hit_rate@1`: đoán +8…+15 điểm → thật **+22,0**.

Đó là một thiên lệch về **hiệu chuẩn**, không phải lỗi tính: tôi ước lượng dè dặt
cho mọi thứ liên quan tới cross-encoder, cả chi phí lẫn lợi ích.

**D6 cũng bị phản chứng.** Tôi khẳng định sigmoid sẽ bão hoà trên ≥1% cặp và lấy
đó làm lý do mặc định dùng logit thô. Đo 2000 logit: khoảng **[−10,87; +8,67]**,
**0,0%** vượt ngưỡng. Mặc định giữ nhưng lý do phải hạ xuống mức yếu hơn và đúng:
sigmoid đơn điệu nên nó không thêm gì cho việc xếp hạng. "Nó bão hoà" là một tuyên
bố tôi chưa có quyền nói.

### Ba lần tự quy kết sai và phải sửa lại

1. **fp32 chậm không phải vì bị nhiễm.** Lần đo corpus thật đầu tiên bị một tiến
   trình pytest thứ hai tranh VRAM (7934/8188 MiB, `max_ms` chạm **171.759 ms**)
   và cho 1869,7 ms; bản synthetic sạch cho 1278,7 ms. Tôi kết luận chênh lệch là
   do nhiễm. **Sai** — lần đo sạch trên corpus thật cho 1794,7 ms. Con số "nhanh"
   1278,7 ms là số *lạc quan* vì text synthetic dài đều nhau nên batch không có
   padding thừa. Số bị nhiễm vẫn không dùng được, nhưng lý do nó khác không phải
   lý do tôi nói.
2. **Test suite OOM là do tôi.** `max_length` nằm trong khoá cache của
   `_load_cross_encoder`, nên bản `L32` là **model thứ ba** trong cùng phiên
   pytest: bge-m3 3,3GB + fp16 1,15GB + fp32 2,3GB + 2,3GB = **9,05GB trên GPU
   8,0GB**. Chuyển bản `L32` sang CPU — điều test đó khẳng định không phụ thuộc
   device, và ngưỡng 0,5 cách nhiễu fp32 giữa hai device (~1e-4) nhiều bậc.
   `lru_cache(maxsize=2)` cho reranker là một lời hứa phần cứng này **không giữ
   được** khi bge-m3 cũng thường trú. Bằng chứng thật cho `W0-06`.

3. **Một vòng chờ quay vô ích 1h47m, và tôi chẩn đoán sai nguyên nhân của nó
   trước khi tìm ra.** Tôi arm `until grep -qE "passed|failed|error" <file test>`
   để chờ suite xong. Nó không bao giờ khớp. Chẩn đoán đầu của tôi: "`tail -8`
   trong pipe đã cắt mất dòng tổng kết" — **sai**. Nguyên nhân thật:
   `pyproject.toml` đã có `addopts = "-q --strict-markers"`, nên `pytest -q` gõ
   tay thành **`-qq`**, và `-qq` **tắt hẳn** dòng tổng kết. Chuỗi ấy chưa bao giờ
   được in.

   Hai hệ quả, và cái thứ hai đáng hơn: (a) vòng chờ đó vô nghĩa từ đầu vì harness
   **đã** thông báo task xong — poll một việc harness tự theo dõi là thuần lãng
   phí; (b) **mọi con số test tôi báo trong phiên này đều là suy ra** từ
   `--collect-only` cộng exit code, chưa lần nào đọc trực tiếp từ pytest. Chạy lại
   không thêm `-q` cho: unit [not gpu] **839**, unit [gpu] **13**, integration
   [not gpu] **109**, integration [gpu] **13**, tất cả **974 passed** trong 267,14s
   — khớp đúng con số đã ghi, nhưng đường đi tới nó yếu hơn tôi trình bày. Đã ghi
   cảnh báo ngay cạnh `addopts` trong `pyproject.toml`; các target `Makefile` cố ý
   không thêm `-q`.

### Phép kiểm nội tại tình cờ, và nó rất mạnh

`rerank c=20` cho `hit_rate@20` **0,7177** và `recall@20` **0,6770** — trùng khít
từng chữ số với hybrid c=20. Đúng như bắt buộc: cùng `candidates` và `top_k` thì
tập trả về y hệt, chỉ khác thứ tự, nên metric theo **tập** phải bất biến còn
metric theo **hạng** thì đổi (và `hit_rate@1` nhảy 20 điểm thuần do đảo thứ tự).
Không thiết kế ra nó — nó rơi từ bảng. Nhưng nó canh đúng chỗ dễ sai nhất:
`retrieve()` dựng lại `RetrievedChunk` từ đầu, nên gán lệch chunk cho điểm, đánh
rơi ứng viên, hay off-by-one ở `rank` đều sẽ làm hai số đó lệch.

### VRAM (số thật, chưa phải `W0-06` chính thức)

| trạng thái | MiB / 8188 |
|---|---:|
| bge-m3 fp32 | ~3.300 |
| + reranker fp16 | ~3.900 |
| + reranker fp32 | **5.685** |
| + tiến trình pytest thứ hai nạp bge-m3 | **7.934** ⚠️ |

Với fp16 mặc định, tầng rerank thêm ~600 MiB → còn dư ~4,3 GB. **Vẫn không đủ**
cho generator local (Qwen3-8B 4-bit ~5,5 GB), nên kiến trúc "generator qua API"
đứng nguyên.

### Vấn đề đang mở

- **`W2-06` là việc tiếp theo.** Nửa phần đã có: nhánh reranked chuyển tiếp
  `filters`, và có test integration canh filter được áp **ở Qdrant trước khi vào
  pool** — lọc *sau* rerank thì cross-encoder đã đọc nội dung tenant khác.
- **Chốt cho `W2-08`:** thêm `rerank_base` và `candidates ∈ {20,50,100}` vào ma
  trận. **Không** quét `batch_size` (quyết định 140) và **không** quét
  `max_length` (đo được nó chỉ cắt 1/12.100 cặp). Ghi lại để không đốt thời gian
  vào ô trống.
- **Trần 0,7799 là câu hỏi của `W3`**: 22% golden set không có bằng chứng trong 50
  ứng viên đầu. Không tầng xếp hạng nào chạm được phần đó.
- ⚠️ `TD-18` sửa phát biểu (quyết định 139): còn 35% mã không vào được pool 50.
  Ưu tiên **đã hạ** vì giá của việc trì hoãn nhỏ đi nhiều.
- ⚠️ `W0-06` (ngân sách VRAM) giờ có bằng chứng là nó **cần thiết**, không chỉ là
  việc nên làm: một phiên pytest đã OOM thật.
- ⚠️ Mọi con số vẫn đo trên `golden_v1` — **review bằng model** (`TD-13`).
- Collection trong Qdrant không đổi.
- Các mục còn lại không đổi: `TD-05`, `TD-08`, `TD-10`, `TD-13`…`TD-18`, `W0-02`.

### Nếu phiên sau bắt đầu từ đây

1. `make lint && make test` · `make up && make test-integration` · `make test-gpu`.
2. Đọc `reports/tasks/w2-05-reranker.md` §2 (trần vùng phủ và vì sao phải đo trước),
   §5.4 (DoD không đạt và ngân sách thật), §6.4 (hybrid không còn đo được) và §7
   (`TD-18` là bài toán truy hồi).
3. `W2-06`: filter ở tầng Qdrant, ca isolation hai tenant. `create_payload_index`
   đã có từ `W1-07`.
4. Nếu cần chạy lại số của `W2-05`: `make rerank-probe BUNDLE=bgem3` cho trần +
   truncation + bão hoà + độ trễ; `make eval-rerank BUNDLE=bgem3 RUN=<tên>` cho
   eval trên cấu hình thắng; `python scripts/known_item_probe.py --rerank` cho
   bốn nhánh.

## 2026-08-21 · Tổ chức lại `plans/reports/` + hai lỗ bằng chứng lộ ra

97 file phẳng → 5 folder chia theo **ai sinh ra file** (đó cũng là ranh giới của
việc được sửa tay hay không): `tasks/` 14 · `runs/` 57 · `compare/` 12 ·
`probes/` 11 · `goldenset/` 3. Bản đồ + bảng review theo `Wx-xx`:
`reports/README.md`.

**Cách rewrite tham chiếu, và vì sao không dùng regex rộng.** 130 tham chiếu trong
21 file. Regex kiểu `reports/[\w-]+` sẽ sửa cả URL World Bank dạng
`.../reports/documentdetail/...` — mà chuỗi đó **có thật** trong
`data/corpus/*.txt`. Nên khớp theo **danh sách tên file tường minh** cộng ba hậu tố
của `runs/`, rồi kiểm bằng cách grep lại: mọi `reports/<file>.md|json|jsonl|py`
còn sót mà chưa có folder = còn việc. Cách kiểm đó tìm ra hai tham chiếu **treo từ
trước**: `reports/baseline.md` (file chưa bao giờ tồn tại; đúng là
`baseline-retrieval.md`) ở hai chỗ, và một đường dẫn cũ của
`scripts/category_compare.py`.

### Hai lỗ bằng chứng mà việc chia folder làm lộ ra

1. **`compare/cmp-baseline-vs-bgem3.md` chưa từng được commit.** Lệnh so sánh *đã*
   chạy ở `W2-01` (số nằm trong report) nhưng output không được lưu — nên phát biểu
   tiêu đề "15/15 metric có ý nghĩa" của `W2-01` không có file máy sinh nào đỡ.
   Sinh lại được (chỉ cần `runs/*-per-query.jsonl`, không cần GPU/Qdrant) và nó
   **khớp từng chữ số** với bảng đã viết: nDCG@10 CI95 [+0,2297, +0,3346].
   ⚠️ Kèm một cảnh báo thật: `baseline-per-query.jsonl` không có `relevant_digest`
   (có trước `W2-03`), nên riêng cặp này cái chốt "cùng bộ nhãn" không kiểm được
   bằng băm, chỉ suy gián tiếp qua `n_relevant_mean` 1,3828 bằng nhau.
2. **`probes/w2-05-rerank-probe.json` không tồn tại.** `make rerank-probe` ghi ra
   đó, nhưng ở `W2-05` tôi chạy script không truyền `--report`. Hệ quả cụ thể:
   **trần vùng phủ `hit_rate@50` = 0,7799** — con số chặn trên mọi phát biểu khác
   của `W2-05` — chỉ có trong bảng viết tay. Đó là vi phạm điều 3 của Định nghĩa
   Done. **Chưa chạy lại**: phần độ trễ sẽ ra số khác lần đo cũ, nên đây là quyết
   định có đánh đổi, không phải việc dọn dẹp.

### Việc còn lại từ phiên này

- `compare.py` cần `--category`/`--lang`, rồi **xoá** `scripts/category_compare.py`.
- Quyết định có chạy lại `make rerank-probe` để có file bằng chứng cho trần 0,7799.
- `W2-06`: filter ở tầng Qdrant, ca isolation hai tenant.

## 2026-08-21 · Chạy lại `make rerank-probe` — vá lỗ bằng chứng của `W2-05`

`probes/w2-05-rerank-probe.json` giờ tồn tại. Điều kiện chạy: Qdrant 200, GPU
trống hẳn (0/8188 MiB), tham số mặc định của probe trùng đúng cấu hình `W2-05`
đã đo (`--base hybrid --rrf-k 1 --candidate-k 20 --pool 50 --batch-size 16
--max-length 512`).

**Chia trước ba nhóm phải khớp và một nhóm được phép khác.** Trần vùng phủ,
truncation và bão hoà là **hàm thuần** của (index, golden set, tham số) — lệch một
chữ số ở đó nghĩa là có gì đã đổi mà tôi chưa biết, không phải nhiễu. Chỉ độ trễ là
đại lượng vật lý. Phân loại trước khi đo là cách duy nhất để phép đo lại có giá
trị: nếu coi tất cả là "chắc sẽ hơi khác" thì nó không phát hiện được gì.

| | kết quả |
|---|---|
| Trần `hit_rate` @1…@50 | ✅ khớp **tới chữ số cuối**: `@50 = 0.7799043062200957` |
| Truncation | ✅ khớp tuyệt đối: 12.100 cặp, p50 287, p95 352, max 542, **1** cặp vượt 512 |
| Bão hoà sigmoid | ⚠️ **lệch chữ số thấp** — xem dưới |
| Độ trễ | pool 50 = **529,3 ms** (n=90) vs 524,4 ms cũ, +0,9% |

### Chỗ lệch, và nó là một thiếu sót thật của cách trình bày

Logit ra −10,875 / −1,650 / +8,672 thay vì −10,873 / −1,652 / +8,668. Không phải
nhiễu: −10,875 và +8,671875 là **giá trị fp16 biểu diễn được đúng**, còn −10,873
thì không (khoảng cách fp16 ở độ lớn ~10 là 2⁻⁷ = 0,0078). Nên bảng §4 gốc đo ở
**fp32**, còn mặc định phục vụ là **fp16**. Kiểm chứ không đoán: chạy lại với
`--dtype float32` cho `−10.873137474060059` / `−1.6518374681472778` /
`8.668050765991211` — tái lập cả ba số cũ.

Cái đáng rút ra: chính hạng mục này đã kết luận "fp16 và fp32 là **hai** model khác
nhau về số" và đã đưa `dtype` vào `CrossEncoderReranker.name` vì thế. Cơ chế đó
**làm việc** — file JSON ghi `reranker = ...@cuda:L512:float16`, đọc file là biết.
Chỗ hỏng là **bảng viết tay** không mang nhãn ấy. Đã thêm cột dtype cho cả hai độ
chính xác thay vì thay số. Kết luận (0,0% bão hoà) không đổi ở cả hai — nhưng nếu
nó *có* phụ thuộc dtype thì cách trình bày cũ không cho ai phát hiện ra.

### Một con số phải sửa: pool 37, không phải 38

`(400 − 5) / 10,4 = 38,0` là mang hằng số đã làm tròn đi tính tiếp. Từ ba điểm đo
được (pool 6/20/50 → 68,0/212,3/529,3 ms) thì khớp là `10,48 × pool + 5,1`, cho
37,7 → làm tròn **xuống** 37, vì 38 sẽ vượt ngân sách. Đã sửa ở 5 chỗ
(`CHECKLIST` §4 + Changelog, report §5.4 + §11, `WORKLOG` phiên trước).

Ba phép đo độc lập cùng đại lượng: 510,1 (§5.2) · 524,4 (§5.3) · 529,3 (hôm nay) —
dải ~4%. "524 ms" nên đọc là **520–530 ms**.

### Điểm đo mới: pool 6

68,0 ms. Nó trả lời cách đọc DoD ("rerank 50 → 6"): chi phí đi theo **pool đầu
vào**, không theo `top_n`. 68,0 ms ở pool 6 so với 529,3 ms ở pool 50 — `top_n`
cắt danh sách *sau* khi đã chấm xong nên nó không rẻ đi được đồng nào. Cùng lý do
`top_n` mặc định là `None` và có test ghim cái bẫy đó.

### Còn lại

- `compare.py` cần `--category`/`--lang`, rồi xoá `scripts/category_compare.py`.
- `W2-06`: filter ở tầng Qdrant, ca isolation hai tenant.

## 2026-08-21 · `W2-06` — metadata filter, và phần thiếu không phải phần tôi đi tìm

Vào hạng mục để thêm `date range` + ca isolation hai tenant. Làm cả hai. Nhưng cái
đáng nhất đến từ việc đổi câu hỏi: thay vì "làm sao lọc theo metadata" (đã có từ
`W1-07`) thì hỏi **"còn đường nào vào dữ liệu mà không qua filter?"**

### ⭐ Đường `fetch` không hề đi qua filter

| method | dùng ở đâu | filter trước `W2-06` |
|---|---|---|
| `retrieve()` | truy vấn vector | ✅ từ `W1-07` |
| `fetch_chunks(chunk_ids)` | nhãn golden set; **`W4-09` giải citation** | ❌ không có tham số |
| `fetch_doc_chunks(doc_ids)` | ánh xạ span; **`W4-06` mở rộng ngữ cảnh** | ❌ không có tham số |

Một `chunk_id` lấy từ log hoặc từ câu trả lời cũ trả về **nội dung đầy đủ của
tenant khác**, dù mọi truy vấn vector đều lọc đúng. Khó thấy vì hai đường trông
giống nhau ở tầng gọi, nhưng `client.retrieve(ids=...)` của Qdrant **không nhận
filter** — vá nó là chuyển sang `scroll`, không phải thêm một tham số. Giữ hai
đường code có chủ ý: đường không filter dùng `retrieve` (nóng, dùng cho phân giải
nhãn, lấy đúng point theo id, không phân trang).

**Bài học phương pháp, không phải bài học Qdrant.** DoD viết "filter áp ở tầng
Qdrant" và đường search đã thoả mãn nó từ trước. Đọc DoD như checklist thì hạng
mục này chỉ còn là viết test; đọc nó như **câu hỏi về bề mặt tấn công** thì nó tìm
ra hai lỗ.

### Hướng của chế độ hỏng quyết định mọi mặc định

Với `tenant_id`, hai hướng hỏng **không đối xứng**: quá chặt → thiếu kết quả,
người dùng thấy và báo lại; quá lỏng → **rò dữ liệu**, không ai thấy, kể cả người
bị rò. `MetadataFilter` (`extra="forbid"`, `frozen`) đóng bốn cách viết filter cho
0 kết quả mà không báo lỗi — khoá gõ sai, `[]`, khoảng ngược, và chunk thiếu
`tenant_id`. Ba cái đầu giờ nổ; cái thứ tư là hành vi Qdrant nên nó có test **ghim**
thay vì được tin. `frozen` là quyết định bảo mật: `W4` kiểm rồi truyền tiếp thì
filter đổi được cho phép nới ra *sau* khi đã kiểm.

⚠️ **`W2-06` không đóng được chuyện quan trọng nhất**: nó không ép người gọi *phải*
truyền `tenant_id`. `rag_core` không biết được "không filter" là đúng (eval chạy
toàn corpus) hay là lỗ rò (serving quên). Chỗ ép là `W4-04`. Có test ghim hành vi
**hiện tại** chứ không ghim hành vi mong muốn.

### ⚠️ Suýt báo cáo nhiễu như một kết quả

Hai lượt đo đầu (chỉ cột đầu-cuối) cho một bảng **đơn điệu theo độ chọn lọc** rất
thuyết phục, và tôi đã suýt viết "lọc làm truy vấn nhanh hơn, càng chặt càng
nhanh". Ba thứ chặn lại:

1. **Đối chứng thứ tự** — lặp lại đúng ca đầu ở cuối bảng: 45,2 vs 46,1, lệch 2%.
2. **Cùng một ca, hai lượt, lệch ±11 ms.** Thứ tự trong bảng là hoán vị, không
   phải xu hướng.
3. **p50 lệch 30% trong khi p95 lệch 3%** → phần biến động không nằm ở thứ đang đo.

Cách sửa **không** phải tăng mẫu (n=200 vẫn cho cùng ca hai giá trị 32,5 và 43,9)
mà là **phân rã**: embed truy vấn một lần ngoài vòng bấm giờ. Xong thì trải từ 30%
tụt xuống **1,8%**. Bài học `W2-04` §6 lần thứ hai: số không cộng lại đúng thì tách
ra, đừng lấy thêm mẫu.

Kết quả sau khi phân rã: **lọc không tốn gì** — tám ca từ 20,5% tới 100% độ chọn
lọc nằm trong **29,98–30,54 ms**. Ca 0 point khớp **nhanh gấp đôi** (15,39 ms):
Qdrant cắt sớm khi cardinality bằng 0, tức "tenant mới chưa có tài liệu" là đường
*nhanh*. Dự đoán `D4`/`D5` sai cả hai.

### Migrate không cần build lại index

Dự đoán `D3` sai theo hướng có lợi: tôi mặc định "đổi payload = build lại index",
trong khi payload và vector là **hai** thứ Qdrant cập nhật độc lập.
`make backfill-payload BUNDLE=bgem3` cập nhật **15.814/15.814** point, tạo index
`published_at`, chạm **0** vector — nên mọi con số eval từ `W2-01` đến `W2-05` vẫn
đúng nguyên vẹn. Build lại thì chúng *có thể* đúng và tôi sẽ phải chứng minh.

### mypy bắt một chỗ nới nửa vời

`build_filter`/`fetch_*` nhận `MetadataFilter` nhưng `Retriever.retrieve` vẫn khai
`dict` → API mới chỉ dùng được ở một nửa điểm vào, và nửa còn lại chỉ hỏng khi có
người chạy `mypy`. 25 lỗi, gồm 4 chỗ vi phạm Liskov ở `Retriever` giả trong test.
Sửa bằng alias dùng chung `type FilterSpec = MetadataFilter | dict[str, Any] | None`.

### Trạng thái

**1020 test xanh** (974 → 1020: +24 unit, +22 integration) · ruff + mypy --strict
sạch trên 104 file · `W2` 6/9 · tổng 26/77.

### Nếu phiên sau bắt đầu từ đây

1. `make lint && make test` · `make up && make test-integration` · `make test-gpu`.
2. **`W2-07`** (experiment runner + MLflow) là việc tiếp theo — `W2-08` cần nó.
3. Trước `W2-08`: `--category`/`--lang` cho `compare.py`, rồi **xoá**
   `scripts/category_compare.py`. DoD `W2-09` đòi nhận xét theo category, và thiếu
   chiều đó đã để một mức tụt có ý nghĩa của `W2-04` đi qua không ai thấy.
4. ⚠️ `doc_type` **không** dùng được làm chiều `W2-08`: khớp 15.814/15.814 point.
   `lang` thì thật (59,4%/40,6%).
5. ⚠️ Bẫy tooling lặp lần thứ hai: `\` + newline trong heredoc `<<'PY'` bị ăn mất
   khi ghi Makefile. Dùng Edit cho recipe có continuation.

---

## Phiên 2026-08-21 (tiếp) · `W2-07` — experiment runner + MLflow

**Mục tiêu phiên:** `W2-07`. Một lệnh chạy hết ma trận thí nghiệm, resume được khi
crash, theo dõi trên MLflow.

```bash
make exp-dry        # in bảng grid + preflight, KHÔNG nạp model
make exp            # chạy hết grid (14 ô, ~13 phút), resume được
make exp-backfill   # dựng lại view MLflow từ báo cáo đã có
make mlflow-ui      # http://127.0.0.1:5000
```

### Phần khó không phải expand grid

`itertools.product` là một dòng. Ba thứ còn lại mất thời gian, và mỗi thứ ứng với
một cách grid "chạy xong" với số sai mà không có gì báo lỗi:

1. **Resume theo `fingerprint` của ô**, không theo tên file báo cáo. "Ô này đã
   chạy" và "ô này đã chạy *với đúng tham số hiện tại*" là hai câu khác nhau.
   `fingerprint` nhận `index_fingerprint` và `golden_digest` **từ ngoài**: build
   lại `bgem3.yaml` với `chunk_size` khác, hoặc `TD-13` ghi lại golden set ở cùng
   đường dẫn — cả hai đổi kết quả mà không đổi một ký tự nào trong YAML của ô.
2. **Preflight kiểm cả grid trước khi chạy ô đầu.** Gom mọi vấn đề rồi nổ một lần.
3. **Ghi báo cáo trước, state sau**, mỗi file qua `tmp` + `os.replace`.

### Preflight bắt lỗi thật ở lần `--dry-run` ĐẦU TIÊN

Grid sinh ô tên `bgem3-sparse`, **trùng đúng báo cáo tiêu đề của `W2-03`** đang
nằm trong `plans/reports/runs/`. Không có kiểm quyền sở hữu thì `make exp` ghi đè
bằng chứng của một hạng mục đã xong. Sửa bằng `run_prefix: e1` ở cấp config.

Đường **không** chọn: để grid *dùng lại* những lần chạy cũ trùng tên. State không
sở hữu chúng nên không biết chúng sinh bằng tham số nào — đúng câu `fingerprint`
tồn tại để trả lời.

### `check_branch_options` — refactor mà dự đoán `D3` nói là không làm được

`D3` đoán preflight không kiểm được tham số nhánh mà không nạp model, vì
`build_branch` dựng cross-encoder trong lúc kiểm. Sai: **tách phần kiểm ra khỏi
phần dựng**. `build_branch` giờ gọi `check_branch_options` nên không có bản chép
luật nào để lệch, và có test chạy **15 cặp đầu vào** qua cả hai đường khẳng định
chúng nổ ở cùng những chỗ. Phụ phẩm: `HYBRID_OPTIONS` + thông báo liệt kê tham số
hợp lệ cho hybrid (trước đó `candidat_k=100` nhận một `TypeError` trần).

### `D2` sai: model đã được chia sẻ sẵn từ `W1`

Đo: mở `baseline.yaml` **27,8 s**, mở `chunk550.yaml` (cùng model) **0,4 s**, mở
`bgem3.yaml` **11,1 s**. Con số 0,4 s dẫn tới `@lru_cache` đã có sẵn trên cả ba
loại model (`_load_model` 4, `_load_sparse_head` 4, cross-encoder 2). Nên gom ô
theo index **không** mua lần nạp model — nó mua **quét nhãn span 14 → 3**, và một
tính chất: mỗi model nạp đúng một lần bất kể `maxsize`.

⚠️ Và nó làm docstring đầu của `_release()` thành **sai**: `lru_cache` giữ tham
chiếu mạnh nên `del` + `gc.collect()` không chạm được trọng số. Đo 4517/8188 MiB
trong lúc grid chạy `reranked`. Hệ quả quan trọng hơn: **trần VRAM của grid do ba
con số `maxsize` ở `rag_core` quyết định, không do runner** — đầu vào cho `W0-06`.

### MLflow: chạy trọn 14 ô mà không ghi được gì

mlflow 3.15 **từ chối** `file:./mlruns` (file store vào maintenance mode, đòi
`sqlite:///`). `build_tracker` bắt exception, rơi về `NullTracker`, ghi một cảnh
báo — **dòng 19 trong một log 2320 dòng** — rồi grid chạy đủ 14 ô, đúng số, đủ
file, và Evidence của DoD không tồn tại.

Ba việc sửa, và cái thứ ba là cái đáng nhất:

1. **URI không mở được là lỗi config, nên nó thuộc preflight.** `open_tracker` nổ
   `TrackingUnavailable`; chỉ *thiếu mlflow* còn được rơi về `NullTracker` (extra
   `tracking` là tuỳ chọn có chủ đích). Lỗi **giữa** grid thì `SafeTracker` lo —
   một ô đã chạy 131 giây không được mất vì tracking server chết.
2. `sqlite:///mlflow.db`, và `.gitignore` thêm `mlflow.db`/`mlartifacts/`.
3. **`pipeline/experiments/backfill.py`** — dựng lại view MLflow **từ file báo
   cáo**, không chạy lại eval. `tracking.py` tuyên bố "MLflow là view, không phải
   nguồn sự thật"; module này *kiểm* câu đó thay vì để nó là một khẳng định. Và
   nó cứu 14 ô vừa chạy khỏi việc phải chạy lại 25 phút.

Để hai đường không thể lệch nhau: cả `_run_one` và `backfill` gọi
`report_params(json.loads(...))` trên **cùng byte** — có test ghim
`_write_report` ghi đúng `report.to_json()`.

### Lỗ hổng phát hiện khi viết backfill → `TD-19`

Một file `runs/*-retrieval.json` **không nói được nó đo trên golden set nào** (và
với `min_overlap_ratio` bao nhiêu). Hôm nay vô hại vì chỉ có một golden set;
`TD-13` sẽ tạo cái thứ hai với **cùng đường dẫn**. Cùng họ với `TD-12`. Phía grid
đã an toàn (`fingerprint` băm `golden_digest` từ `W2-07`), phía báo cáo lẻ chưa.

### Log: 97% là nhiễu

2320 dòng, **2270 dòng (97%) là `HTTP Request: HEAD huggingface.co/...`**. DoD "1
lệnh chạy hết grid" đạt về chức năng mà không đạt về việc đọc được nó đã chạy gì —
với một lệnh 13 phút thì đó là cùng một yêu cầu. Hạ từng logger tường minh,
**không** hạ root (cảnh báo span của `_resolve_span_labels` phải thấy được).

### Grid thật: 14 ô, và 5 con số tái lập đúng từng chữ số

| Ô | nDCG@10 | Số đã công bố |
|---|---|---|
| `e1-baseline-dense` | 0,1621 | `W2-01` baseline ✅ |
| `e1-bgem3-dense` | 0,4442 | `W2-01` BGE-M3 ✅ |
| `e1-bgem3-sparse` | 0,3733 | `W2-03` ✅ |
| `e1-rrf-bgem3-hybrid-k1` | 0,4563 | `W2-04` `k=1` ✅ |
| `e1-rr-…-onhybrid-rc50` | 0,6481 (hit@1 0,5598) | `W2-05` ✅ |

Đợt này refactor `_eval_against_index` → `IndexSession`, đổi
`_resolve_span_labels(retriever)` → `(store)`, và ghi file qua `os.replace`. Năm
dòng trên là cách duy nhất biết những thay đổi đó **không đổi một con số nào**.

Về chỗ đổi `retriever` → `store`: đã kiểm cả bốn nhánh đều forward
`fetch_doc_chunks` và mọi báo cáo `reranked` đã công bố **đều có**
`config.span_resolution` — nếu không thì số `W2-05` đã được chấm bằng nhãn cũ và
không so được với dense. Refactor biến tính chất đó thành **cấu trúc**.

### Ghi chú từ log: `chunk550` đổi TOÀN BỘ 209 nhãn

```
[1] baseline : 209 tính lại · 33 giữ nhãn cũ ·   9 đổi nhãn
[2] chunk550 : 209 tính lại · 33 giữ nhãn cũ · 209 đổi nhãn
[3] bgem3    : 209 tính lại · 33 giữ nhãn cũ ·   9 đổi nhãn
```

`G2`/`TD-11` hiện ra trực tiếp: recall@k/nDCG/MAP của `chunk550` không so được với
hai ô kia. `compare.py` tự từ chối; `n_relevant_mean` trên MLflow là chỗ nhìn thấy.

### Dự đoán: 2/6 đúng, 3 sai, 1 nửa

Ba lần sai đều là **đánh giá quá cao độ khó** — hai lần tưởng một thứ không tách
được (`D3`) hoặc không cùng tồn tại được (`D6`: mlflow + torch CUDA chạy êm,
`uv sync --all-extras` **3,9 s** nhờ hardlink từ `D:\uv-cache`), một lần tưởng
phải tự tối ưu thứ thư viện đã tối ưu (`D2`). Hướng lệch **ngược** với `W2-05`, ở
đó tôi đánh giá quá thấp chi phí cross-encoder ba lần liền.

### Trạng thái

**1096 test xanh** (1020 → 1096: +76 unit) · ruff + `mypy --strict` sạch trên 110
file · `W2` 7/9 · tổng 27/77.

### Nếu phiên sau bắt đầu từ đây

1. `make lint && make test` · `make up && make test-integration` · `make test-gpu`.
2. **Trước `W2-08`: `--category`/`--lang` cho `compare.py`**, rồi **xoá**
   `scripts/category_compare.py`. DoD `W2-09` đòi nhận xét theo category, và thiếu
   chiều đó đã để một mức tụt có ý nghĩa của `W2-04` đi qua không ai thấy.
3. **`W2-08`** đã có dữ liệu 14 ô từ `make exp`, nhưng DoD của nó đòi **`p`/CI cho
   từng dòng** — đó là `compare.py`, chưa chạy. Grid **không** phải `W2-08`.
4. ⚠️ **`TD-19` làm TRƯỚC `TD-13`**, không phải sau: thêm `golden`/`golden_digest`/
   `min_overlap_ratio` vào `config` của `EvalReport`.
5. ⚠️ Bẫy tooling lặp **lần thứ ba**: heredoc `<<'PY'` chứa docstring Python với
   `'''` làm bash báo `unexpected EOF`. Cách chạy được: ghi script ra scratchpad
   bằng Write rồi `uv run python <path>`.
6. ⚠️ Cột `p95` của grid dùng để **sàng**, không để kết luận: 13 phút chạy liên
   tục trên GPU laptop, không có đối chứng thứ tự. Số độ trễ đáng tin đến từ probe
   riêng (`W2-04` §6, `W2-06` §5).

---

## Phiên 2026-08-21 (tiếp) · `W2-08-prep` — `--category`/`--lang` cho `compare.py`

**Mục tiêu phiên:** điều kiện của DoD `W2-09`. Kiểm định được theo tập con, và xoá
`scripts/category_compare.py`.

```bash
make eval-compare-by BASE=bgem3 CAND=bgem3-rrf-k1-c20            # quét mọi category
make eval-compare-by BASE=bgem3 CAND=bgem3-rr-c50 BY=lang        # quét theo ngôn ngữ
make eval-compare-subset BASE=bgem3 CAND=bgem3-rrf-k1-c20 CAT=cross_lingual
```

### Phần khó không phải bộ lọc

Lọc theo `category` là mười dòng. Vấn đề thật: **hỏi 6 nhóm × 15 metric = 90 câu
hỏi rồi chọn cái to nhất là một quy trình chọn mẫu**, và ở α=0,05 nó kỳ vọng **4,5
kết quả "có ý nghĩa" thuần do ngẫu nhiên** — tức "category nào cải thiện nhiều
nhất" gần như *bảo đảm* tìm ra một cái, kể cả khi không có gì.

Nên **hai lệnh, vì có hai loại suy luận**:

| Lệnh | Là gì | Hiệu chỉnh |
|---|---|---|
| `--category X` | Một giả thuyết **nêu trước** | Không |
| `--by category` | Một cuộc **tìm kiếm** | Bonferroni trên `nhóm × metric` |

CLI **từ chối** dùng cả hai cùng lúc. `compare_runs` một-cặp giữ nguyên α=0,05 —
đổi ở đó sẽ lặng lẽ viết lại kết luận `W2-01`…`W2-07`, có test ghim.

Bonferroni chứ không Holm: Holm dùng ngưỡng khác nhau cho từng hạng, mà một khoảng
tin cậy bootstrap không biểu diễn được điều đó bằng một khoảng duy nhất — bảng sẽ
có hai luật trong một bảng.

### `KHÔNG ĐỦ LỰC` — phân biệt quan trọng nhất, và nó tính được

McNemar exact chỉ dùng `n` câu **đổi chiều**, nên trần `p` của nó là `2/2ⁿ`:
n=4 → **0,125**; n=5 → 0,0625; **n=6 → 0,031** (chỗ đầu qua 0,05); n=12 → chỗ đầu
qua α đã hiệu chỉnh cho 90 phép kiểm.

`golden_v1` có `table_lookup` = **4 câu**, nên nó **vĩnh viễn** không đo được trên
metric nhị phân. Không có cờ này thì "trong ngưỡng nhiễu" của nó đọc như *không có
hiệu ứng* trong khi nó là *không có lực* — hai chuyện trái ngược, cùng một dòng chữ.

### `KHÔNG KẾT LUẬN` — lỗi tôi tự tạo ra, rồi phải đo mới thấy

Sau khi thêm Bonferroni, bảng hiện `CI99,72% [+0,0000, +0,1765]` — biên **đúng 0**.
Và **4/4** dòng `recall@5` bị vậy, nên cả bốn bị dán "trong ngưỡng nhiễu" *do cấu
tạo*.

Lo ngại đầu của tôi sai: tôi cho là 10.000 iterations không đủ đọc phân vị 0,14%
và định nâng lên ~72.000. **Đo 6 seed × 3 mức trước khi sửa:**

| iterations | điểm ở đuôi | dao động biên theo seed |
|---|---|---|
| 10.000 | 13,9 | **0,0233** |
| 50.000 | 69,4 | **0,0233** |
| 200.000 | 277,8 | 0,0000 |

`0,0233` = **đúng 1/43** = một bước lưới, không phải sai số Monte Carlo. Phần lớn
câu có hiệu bằng 0 nên phân bố bootstrap nằm trên lưới bước `1/n` và phân vị cực
đoan rơi đúng lên 0. Tăng iterations gấp 5 **không đổi gì** — ghi lại để lần sau
không ai "sửa" bằng cách tăng iterations.

⚠️ Tự phê: cờ này tồn tại vì chính bản hiệu chỉnh của tôi tạo ra vấn đề. Không đo
thì tôi đã công bố một bảng toàn "trong ngưỡng nhiễu" và tưởng đó là kết quả.

### Kết quả: `W2-04` sống sót, nhưng cả ba dẫn chứng đã công bố đều sai

Phát hiện `cross_lingual` **sống sót** hiệu chỉnh cho cả 90 phép kiểm (α=0,000556):
`map@20` CI99,94% [−0,1235, −0,0193] · `mrr` [−0,1335, −0,0193] · `ndcg@10`
[−0,1630, −0,0054]. Bằng chứng **mạnh hơn** lúc phát biểu.

Nhưng ba con số CHECKLIST/`w2-05` trích đều sai **cùng một khuôn**:

| Dẫn chứng đã công bố | Vấn đề |
|---|---|
| `recall@5` CI95 [−0,1860, −0,0233] | `KHÔNG KẾT LUẬN` ở mức hiệu chỉnh (biên CI đúng 0) |
| `hit_rate@5` **4↔0** | trần `p` = **0,125** — chưa bao giờ có lực |
| "reranker vá lại (1↔0, `p`=1,000)" | trần `p` = **1,0** — không mang thông tin nào |

💡 Khuôn: **một `p` vô nghĩa trích cạnh một CI có nghĩa**, trông như bằng chứng bổ
trợ trong khi nó là số 0. Bằng chứng thật cho dòng cuối: `bgem3-rrf-k1-c20 →
bgem3-rr-c50` trên `cross_lingual`, `hit_rate@5` **0↔14, `p` = 0,00012**.

Đã sửa cả `CHECKLIST.md` và `w2-05-reranker.md`.

### Đối chứng: cờ không dán "không kết luận" cho mọi thứ

Reranker chia theo `lang`, 30 phép kiểm: **29/30 sống sót** Bonferroni (vi 15/15,
en 14/15, 1 hàng không đủ lực). So với `W2-04` (60/90 không kết luận được), sự đối
lập chính là thông tin: `W2-05` là hiệu ứng lớn thật, `W2-04` là hiệu ứng biên mà
bảng tổng che đi.

### Script tái sinh bắt được điều tôi đã cho là không xảy ra

Đổi `p` sang `{:.4g}` (vì `0.000` không phân biệt `4e-9` với `4,9e-4`) chạm 12 file
`compare/`. Tôi tự nhủ "`family_size=1` nên hành vi một-cặp không đổi" và viết script
tái sinh **so mọi ô trừ ô kiểm định**. Nó nổ ở 2 file: `α` thì không đổi, nhưng
**hai cờ mới áp cả ở `family_size=1`** → 3 hàng đổi kết luận.

Cả ba đúng và đều làm sắc thêm — đáng chú ý nhất là `hit_rate@1` của `c50 → c100`
(`0↔4`, trần `p`=0,125): docstring `BINARY_METRICS` viết ở `W2-05` **đã** lý giải
đúng ("bốn lần tung xu cùng mặt thì không kết luận được gì") trong khi cột kết luận
nói "trong ngưỡng nhiễu". Cùng một file đã tự nói hai điều khác nhau từ `W2-05`;
giờ chúng khớp. Và `13/15 trong ngưỡng nhiễu` của `W2-05` (cơ sở của kết luận kiến
trúc về hybrid) **còn nguyên 13**, không hàng nào bị gắn cờ.

💡 Bài học lặp lại từ `W2-07` §8: cách duy nhất biết một đợt sửa "không đổi con số
nào" là **có một script nói không**.

### Trạng thái

**1143 test xanh** (1100 → 1143: +43 unit, `test_eval_compare.py` 39 → 82) · ruff +
`mypy --strict` sạch trên 110 file · `W2` 8/10 · tổng 28/78.

### Nếu phiên sau bắt đầu từ đây

1. `make lint && make test` · `make up && make test-integration` · `make test-gpu`.
2. **`W2-08`** chạy được rồi: 14 ô của `make exp` + `make eval-compare-by` cho
   `p`/CI từng dòng. Ba bẫy đã đo: `e1-chunk550-dense` không so được recall/nDCG/MAP
   (`n_relevant_mean` 1,9617 vs 1,3828); `table_lookup` (4 câu) vĩnh viễn
   `KHÔNG ĐỦ LỰC`; bảng `--by` đọc để **loại** giả thuyết, muốn **chọn** người
   thắng thì nêu tên nhóm trước bằng `--category`.
3. ⚠️ **`W2-09` phải quyết một chuyện tôi cố ý không quyết**: bảng một-cặp **vẫn**
   không hiệu chỉnh cho 15 metric (kỳ vọng 0,75 dương giả/bảng). Sửa thì viết lại
   kết luận `W2-01`…`W2-07`; không sửa thì phải nói ra.
4. ⚠️ **`TD-19` trước `TD-13`**: báo cáo lẻ không nói được nó đo trên golden set nào.
5. ⚠️ Bẫy tooling lặp **lần thứ tư**: heredoc `<<'PY'` chứa `\n` trong string
   Python bị bash ăn mất một lớp escape → newline thật, file syntax error. Và
   `grep -rn` vẫn timeout vì quét `.venv`. Dùng Write ra scratchpad + `uv run
   python <path>`, và dùng Grep tool thay `grep -r`.
6. ⚠️⚠️ **Bẫy MỚI, và nó thuộc đúng họ lỗi mà cả W2 đang chống:**
   `uv run pytest | tail -4` trả exit code của **`tail`**, không của `pytest`. Tôi
   đọc "exit code 0" và suýt báo cáo "1143 test xanh" trong khi pytest thật ra
   **18 failed, 127 errors** (Docker Desktop tắt qua đêm nên mọi integration test
   mất Qdrant — không phải do thay đổi nào của tôi). Một phép kiểm báo thành công
   mà không kiểm đúng thứ cần kiểm, giống `ensure_collection` ở `W2-02` và
   `MatchAny(any=[])` ở `W2-06`.
   Cách chạy đúng: `uv run pytest > file 2>&1; echo "exit=$?"` rồi mới `tail`.
   Xác nhận lại sau khi `make up`: **986 unit + 157 integration/gpu = 1143, exit 0**.

---

## Phiên 2026-08-22 · `W2-08` — ablation 14 ô, và người thắng do 6 mẫu lại quyết định

**Mục tiêu phiên:** DoD `W2-08` — ≥ 12 tổ hợp, xác định cấu hình thắng, `p`/CI từng dòng.

```bash
make ablation                                    # bảng 14 ô + tập tương đương
make ablation RANK=hit_rate@1 RANK_SLUG=hit1     # đối chứng: đổi metric xếp hạng
```

### DoD đạt, nhưng câu hỏi của nó hỏi sai

"Xác định được cấu hình thắng" nghe như một dòng. Nó là một **phép chọn cực đại
trên 14 ước lượng nhiễu**: con số của cái thắng lệch lên có hệ thống, và `TD-11` đã
đo ngưỡng phân giải của `golden_v1` là ~6 điểm trong khi ba ô đầu chênh **2,5
điểm**. Nên câu trả lời là **bốn rổ**:

| rổ | ô |
|---|---:|
| không phân biệt được với đỉnh bảng | **2** |
| ba metric chính **không đồng ý** | **3** |
| kém ở mọi metric so được | **8** |
| không so được (nhãn khác) | **1** |

### ⚠️⚠️ Lỗi đắt nhất, và nó là khuôn lỗi thứ ba của `W2`

Bản đầu định nghĩa tập tương đương là "không bị đánh bại". `chunk550-dense` có nhãn
khác nên **mọi** phép so bị *từ chối* — và "bị từ chối" đi vào cùng đường với
"hoà". Nó vào tập thắng. Rồi vì nó dense thuần (46,1 ms) nó thành **thành viên rẻ
nhất**, tức công cụ đề xuất ô có `ndcg@10 = 0,1215` thay cho ô có **0,6736**: **ô
tệ nhất bảng làm khuyến nghị.**

💡 Cùng khuôn `ensure_collection` tin thay vì kiểm (`W2-02`) và `MatchAny(any=[])`
cho 0 kết quả mà không báo lỗi (`W2-06`): **một nhánh "không có thông tin" chảy vào
cùng đường với nhánh "thông tin là bằng nhau".** Ba rổ → bốn rổ, có test hồi quy
ghim đúng ca 46 ms.

### ⭐⭐ Người thắng của cả bảng từng do **6 mẫu lại trên 10.000** quyết định

Biên dưới của khoảng `α` là phần tử thứ `α/2 × B` của dãy đã sắp. Ở α = 0,05 đó là
phần tử **250**. Ở α đã hiệu chỉnh cho 39 phép kiểm (0,00128) đó là phần tử **6**.

Đo 6 seed × 3 mức `B` trên đúng cặp quyết định (`rc50 → rc100`):

| metric | B=10.000 (đuôi 6) | B=50.000 (đuôi 32) | B=200.000 (đuôi 128) |
|---|---|---|---|
| `ndcg@10` | **dấu {+, −}** theo seed | {−} nhất quán | {−} nhất quán |
| `recall@5` | **{+, −, 0}** | {−, 0} | {−, 0} |
| `mrr` | {+}, sd 0,00022 | {+, 0} | {+}, sd 0,00008 |

`ndcg@10` ở B = 10.000 cho "khác biệt thật"; ở 50.000 và 200.000 thì khoảng **chứa
0**. Vá không tốn thêm một lần lấy mẫu nào: số mẫu dưới biên là `B(B, α/2)`, sd
`sqrt(tail)`, nên đọc lại biên ở `tail ± sqrt(tail)` của **chính dãy đã sắp đó**.

### ⚠️ Và nó NGƯỢC ghi chú tôi tự viết ở phiên trước

`W2-08-prep` ghi *"đừng chữa bằng cách tăng iterations"*, kèm phép đo chứng minh
10.000 → 50.000 không đổi gì. Ghi chú đó **đúng cho ca của nó**: metric nhị phân
thưa, phân bố trên lưới `1/n`. Ca này là metric **liên tục** và tăng `B` **đảo**
kết luận.

💡 **Hai giới hạn khác nhau cho cùng một triệu chứng "biên sát 0", và thứ phân biệt
chúng là độ hạt của metric.** Nếu tôi tin ghi chú của chính mình mà không đo lại thì
kết luận sai vẫn đứng. Lần thứ hai trong `W2` một ghi chú đúng-cho-ca-của-nó bị áp
sang ca khác.

### ⚠️ Luật `1/n` của tôi sai, và 13/14 file đã công bố nói ra

Bản đầu dùng bước lưới `1/n` cho **mọi** metric → **13/14** file `compare/` đổi kết
luận, phần lớn là `precision@k`/`recall@k` bị gắn cờ oan. `1/n` chỉ đúng cho metric
nhị phân; `precision@20` nhận bội của 1/20 nên bước thật nhỏ hơn **20 lần**. Luật
đúng: `min_increment / n`, **đo từ dữ liệu**. Sau khi sửa: **6/14 file, 10 hàng**,
kiểm từng hàng thì cả 10 đúng.

Mặc định `min_increment` cũng phải sửa: `1,0` ("coi như nhị phân") làm chính dẫn
chứng `cross_lingual` của `W2-04` bị dán `KHÔNG KẾT LUẬN`. **Một cờ đoán ngưỡng của
chính nó là cờ bật theo phỏng đoán** — mà đó đúng là loại lỗi cờ này dựng để bắt.
Mặc định giờ là `0,0` = "chưa biết" = không gắn cờ.

### Kết quả

* **Bậc thang pool có một bước phân giải được và một bước không.** 20 → 50 là
  `hit_rate@5` **0↔18**, `p` = 7,6e-06. 50 → 100 thì `KHÔNG ĐỦ LỰC`/`TRÁI CHIỀU` và
  tốn **1,91×**. Và chỗ bị gắn cờ nói đúng bản chất: **vùng phủ tăng sạch**
  (`hit_rate@10` **0↔9**) còn **chất lượng xếp hạng thì không** (`nDCG`/`MAP` đếm
  câu 10↔11 và 10↔12 — nhiều câu bị làm hỏng hơn số câu được sửa). Nên `rc100` vs
  `rc50` là đánh đổi **vùng phủ vs chi phí**, không phải chất lượng vs chi phí.
* ⚠️ **Giới hạn của bộ ba metric chính, phải nói ra**: cả ba đều là metric *xếp
  hạng*, không cái nào đo vùng phủ — tức chúng loại trừ đúng chiều `rc100` thắng
  sạch. Chọn trước khi xem số, nên không đổi; nhưng phải ghi ra.
* **Tầng hybrid không đo được ở cả ba độ sâu pool** (0/24; `hit_rate@10` pool 20 cho
  **10↔10, `p` = 1,0**) và không đắt hơn ở mức nào. `W2-05` tái lập.
* **Chiều `chunk_size` không có `p`** → `TD-20`. Và chạy đúng cặp của `TD-11` cho
  thấy `p = 0,711` đã công bố tồn tại được vì file của nó có **trước** hàng rào băm
  `W2-03` — cùng phép so đó với file có băm thì bị **từ chối**. Hướng kết luận
  không đổi, bằng chứng **yếu hơn** mức đã công bố.

**Bốn con số đã công bố bị sửa**, cả bốn theo hướng sắc thêm: `W2-04` `k=1`
3/15 → **2/15** · `W2-05` "13/15 trong ngưỡng nhiễu" → 13/15 *không đạt ý nghĩa*
(12 nhiễu + 1 `KHÔNG KẾT LUẬN`) · `c50 → c100` `ndcg@10` → **`TRÁI CHIỀU`** ·
`c=100` "tốt hơn ở mọi metric" → tốt hơn về **vùng phủ**.

**Hai chốt kiểm soát giữ nguyên đúng từng chữ số**: `baseline → bgem3` **15/15**
(tiêu đề `W2-01`) và `bgem3 → rr-c50` **15/15** (tiêu đề `W2-05`).

### Dự đoán: 3/7 đúng

Bốn lần sai, và **ba lần sai cùng một hướng: đánh giá quá cao độ phân giải của 209
câu.** Ngược thiên lệch của `W2-05` (ở đó tôi đánh giá thấp cả chi phí lẫn lợi ích
của cross-encoder). `D6` sai kiểu không đoán trước được: tôi cho là `chunk550` kém
baseline *đo được*, thực tế là nó không so được **metric nào**.

### Trạng thái

**1187 test xanh** (1043 unit + 144 integration; 1143 → 1187 = +23 ablation + 21
compare) · ruff + `mypy --strict` sạch · `W2` **9/10** · tổng **29/78**.

⚠️ Con số "986 unit + 157 integration/gpu" ghi ở phiên trước **không tái lập được**
— đếm lại bằng `pytest --co -q` cho 999 + 144 ở `HEAD`. Tổng 1143 thì đúng, cách
chia thì sai.

### Nếu phiên sau bắt đầu từ đây

1. `make lint && make test` · `make up && make test-integration`.
2. **`W2-09`** là hạng mục cuối của `W2`. Dữ liệu đủ hẳn: `make ablation` +
   `make eval-compare-by`.
3. ⚠️ **Ba quyết định `W2-08` cố ý để lại cho `W2-09`**: (a) bảng một-cặp **vẫn**
   không hiệu chỉnh cho 15 metric — sửa thì viết lại `W2-01`…`W2-07`, không sửa thì
   phải nói ra; (b) `MIN_TAIL_RESAMPLES = 30` chỉ để **giải thích** cờ, chưa để tự
   nâng `B` — đọc được α = 0,00128 cần ~47.000 mẫu lại (~4,7× chi phí); (c) chốt
   điểm vận hành `rc50` vs `rc100` cần biết bộ sinh có dùng được thêm bằng chứng
   hay không, tức có thể thuộc `W4`.
4. ⚠️ **`TD-20` trước khi quét lại `chunk_size`**, `TD-19` trước `TD-13`.
5. ⚠️ Bẫy heredoc lặp **lần thứ năm** — lần này ăn mất dấu nối dòng `\\` trong
   `Makefile`, làm một target thành một dòng dài (và nó đã làm thế với hai target
   của `W2-08-prep` từ phiên trước mà không ai thấy). Cứ dùng Write ra scratchpad +
   `uv run python <path>`.

---

## Phiên 2026-08-22 (2) · `W2-09` — `W2` đóng, và câu thứ hai của DoD không có câu trả lời

**Mục tiêu phiên:** DoD `W2-09` — bảng delta + ≥ 2 nhận xét về *category nào cải
thiện nhiều nhất*. Kèm ba quyết định `W2-08` cố ý để lại.

```bash
make eval-compare BASE=e1-baseline-dense CAND=e1-rr-bgem3-reranked-onhybrid-rc100
make eval-compare-by BASE=e1-baseline-dense CAND=e1-rr-bgem3-reranked-onhybrid-rc100
make contrast                      # mới: nhóm nào cải thiện nhiều nhất
make contrast BY=lang
```

### Nửa đầu DoD: sạch

`e1-baseline-dense` → đỉnh bảng, **15/15 metric có ý nghĩa**, `p` từ 1,98e-22 tới
5,88e-39. `ndcg@10` 0,1621 → 0,6736 (**×4,2**). Con số đáng nhớ: `hit_rate@5`
**0↔120** — 120 câu được sửa, **0 câu bị làm hỏng**, tức 57% bộ đo đổi từ trượt
sang trúng mà không mất câu nào. Đây là bảng duy nhất của cả `W2` mà hiệu chỉnh đa
so sánh không cần bàn.

### ⭐⭐ Nửa sau: câu hỏi không có câu trả lời, và phải dựng công cụ mới mới biết

"Category nào cải thiện **nhiều nhất**" so `Δ_A` với `Δ_B`. Mọi bảng đã công bố —
kể cả `compare_by_group` dựng riêng cho DoD này ở `W2-08-prep` — chỉ kiểm
`Δ ≠ 0`. **Không hàng nào trong cả `W2` trả lời câu ấy.**

`pipeline/eval/contrast.py`, và nó khác `ablation.py` ở đúng một chỗ đổi cả phép
kiểm: hai **cấu hình** đo trên cùng bộ câu → bootstrap **cặp**; hai **nhóm** là hai
tập câu **rời nhau** → bootstrap **không cặp**. Dùng nhầm phép cặp ở đây là ghép
câu nhóm này với câu nhóm kia, tức bịa ra một tương quan không tồn tại.

Kết quả trên `hit_rate@5`:

| hạng | nhóm | n | Δ | vs đỉnh bảng |
|---:|---|---:|---:|---|
| 1 | `cross_lingual` | 43 | +0,6512 | *(đỉnh bảng)* |
| 2 | `aggregation` | 26 | +0,6154 | CI99% [−0,2728, +0,3444] |
| 6 | `multi_hop` | 34 | +0,5000 | CI99% [−0,1306, +0,4378] |

**Cả 6 nhóm hoà. 0/5 phép so phân giải được.** Lặp lại y hệt trên `hit_rate@1` và
`mrr`.

### Đối chứng bắt buộc — và nó là chỗ kết luận này có giá trị

Nếu "hoà 6/6" chỉ là hệ quả của Bonferroni thì đó là kết luận của công cụ, không
phải của bộ đo. Đo lại ở α **thô** 0,05: **vẫn 5/5 hoà**, cả ba metric. Khoảng cách
xa nhất giữa hai category (+0,1512) chỉ bằng **0,70×** nửa bề rộng CI95 **chưa**
hiệu chỉnh.

💡 **Ngưỡng phân giải giữa hai nhóm: ~±0,22 tuyệt đối** (43 câu vs 34 câu). `TD-11`
đo ~6 điểm cho cả 209 câu, nên **chia 209 câu thành 6 nhóm rồi so các nhóm với nhau
tốn ~3,6× độ phân giải**. Cần golden set **~440 câu** mới tách được khoảng cách hiện
tại — con số định lượng đầu tiên cho `TD-13`.

### ⚠️ Hàng rào thứ tư, canh một trục ba hàng rào cũ không nhìn

Ba hàng rào của `compare.py` đều canh trục *"hai lần chạy"*. Không cái nào thấy:

    factoid      1,0147 nhãn/câu
    aggregation  2,4231 nhãn/câu

Trong **một** nhóm thì hai lần chạy dùng đúng bộ nhãn ấy nên hàng rào băm `W2-03`
im lặng — **đúng**, nó nhìn trục khác. Nhưng so `Δ recall@5` của `factoid` với của
`aggregation` là so hai thang đo.

Nó lộ ra ngay trong dữ liệu, không cần lập luận: `aggregation` hạng **2**/6 theo
`hit_rate@5` và hạng **5**/6 theo `ndcg@10` — nhóm bị xê dịch nhiều nhất đúng là
nhóm nhiều nhãn nhất.

⚠️ Hệ quả: **`ndcg@10` — tiêu chí đầu của `G2` — không xếp hạng được giữa các
nhóm.** Dự đoán D2 của tôi đúng con số và **sai loại câu hỏi**.

### Hai nhận xét mà DoD đòi

1. **`cross_lingual` là nhóm duy nhất có một bậc làm nó TỆ ĐI**: bậc hybrid cho
   `ndcg@10` **−0,0831** CI95 [−0,1279, −0,0396], **17 câu tệ đi vs 1 câu tốt lên**
   (`mrr` và `map@20` là **19↔1**).
   Phát hiện `W2-04` tái lập nguyên vẹn, lần này ở bậc *giữa* của chính đường tới
   đỉnh bảng.
2. ⭐ **Bậc đổi embedding model chiếm phần lớn nhất ở cả 6/6 category** (45–88%).
   Hybrid + reranker + pool sâu cộng lại vẫn nhỏ hơn. Ngược dự đoán D7.

### ⭐⭐ Quyết định (b): luật hiển nhiên cho câu trả lời SAI

Đo bằng thứ thật sự quan trọng — **số thành viên tập thắng**, không phải bề rộng
một khoảng:

| `B` | đuôi | tập thắng |
|---:|---:|---:|
| 10.000 | 5 | **2 ô** ← đã công bố |
| 50.000 | 31 | **3 ô** ⚠️ |
| 200.000 | 127 | **2 ô** |
| 400.000 | 255 | **2 ô** |

Luật "nâng `B` cho tới khi đuôi đủ 30" đưa `B` tới ~48.400 — **đúng vùng bất ổn** —
và cho câu trả lời mà `B` mặc định đã trả lời đúng. **Nâng nửa vời tệ hơn không
nâng.**

Hàng gây chuyện là `mrr` của `rc50`: biên gần như đứng yên (−0,0007 → −0,0002), thứ
đổi là **dải dao động của chính nó**, và ở 50.000 dải ấy vắt qua 0.

💡 **Chỗ `W2-08` chọn sai hằng số, nói cho chính xác:** nó đo ba mức đuôi (6/32/128)
với tiêu chí *"**dấu** của biên có ổn định không"*, và ở đuôi 32 dấu **đúng là** ổn
định. Nhưng thứ quyết định một ô nằm rổ nào không phải dấu của biên — mà là **dải
dao động** của nó có chứa 0 hay không. **Hai đại lượng ổn định ở hai tốc độ khác
nhau, và tôi chọn hằng số bằng cái ổn định nhanh hơn.**

`MIN_TAIL_RESAMPLES` **30 → 128** · `resolve_iterations` có trần
`MAX_BOOTSTRAP = 200.000` · **chỉ nâng khi người gọi để nguyên mặc định** (một `B`
nêu tường minh là lựa chọn có chủ đích — tự nâng nó làm chính phép quét đo `B` mất
khả năng đo, tức bảng ngay trên sẽ không đo được).

✅ **Kiểm chứng: 0/15 file `compare/` đổi kết luận.** Bảng chia nhóm **thật sự** nâng
— log ghi **60 lần** `B: 10.000 → 200.000` ở α=0,000556 nơi đuôi cũ chỉ có **2
mẫu**. Ba phát hiện `cross_lingual` của `W2-08-prep` từng đọc từ đuôi 2 mẫu và
**sống sót nguyên vẹn** ở 200.000.

⚠️ **Bảng ablation thì đổi đúng một hàng**: `ndcg@10` của `rc50`, CI99,87%
[−0,0638, **−0,0003**] → [−0,0647, **+0,0004**], tức `TRÁI CHIỀU` →
`trong ngưỡng nhiễu`. **Không bất ngờ** — `W2-08` §5 đã đo được đúng chuyện đó và
ghi vào report, nhưng bảng mặc định vẫn chạy ở 10.000 nên nó **mâu thuẫn với phép
đo của chính nó**; giờ hai chỗ khớp. Tập thắng vẫn **2 ô**, `rc50` vẫn ở rổ tranh
chấp (nhờ `mrr`).

### Quyết định (a): không hiệu chỉnh — vì tiền đề của câu hỏi sai

"Kỳ vọng 0,75 dương giả mỗi bảng" tính bằng `15 × 0,05`, tức coi 15 metric là độc
lập. Chưa ai kiểm tiền đề đó. Đo (Li & Ji trên ma trận tương quan của hiệu từng câu):

| cặp | `n_eff` | `\|r\|` tb | trị riêng đầu |
|---|---:|---:|---:|
| `baseline → bgem3` | 5,0 | 0,666 | 10,42 |
| `bgem3 → bgem3-rrf` | 7,0 | 0,454 | 7,72 |
| `rc50 → rc100` | **4,0** | **0,833** | **12,69** |

⭐ Cặp bị hiệu chỉnh cắn đau nhất đúng là cặp tương quan chặt nhất — một chiều giải
thích **85%** phương sai. Con số đúng là **0,35**, không phải 0,75.

Nếu hiệu chỉnh thật thì mất **33 hàng/6 file** (`m=15`) hoặc 26 (`m=7`) — nhưng
**hai chốt kiểm soát không đổi ở cả hai mức**. Thay vào đó **mọi bảng giờ tự in
ngân sách dương giả của nó ngay dòng đầu**.

💡 Vì sao đặt trong bảng chứ không trong docstring: `W2-08-prep` đã ghi lựa chọn ấy
vào docstring `compare_by_group` — nơi người đọc bảng không bao giờ tới. Cả ba lần
trích sai đã công bố đều là **rút một hàng thuận nhất trong 15 hàng ra trích**.

### Quyết định (c): `rc50`, điều kiện lật lại nêu trước

Ở `B` hội tụ, `rc50` ở rổ **tranh chấp** (1/3 metric chính nói `rc100` hơn, 1/3
nhiễu, 1/3 không có lực). Ưu thế ~2,5 điểm `ndcg@10` ≈ **5 câu**/209, giá **+555
ms** = 16% ngân sách 3500 ms. ⚠️ Nhưng pool sâu đóng góp **39%** mức cải thiện của
`cross_lingual` và **35%** của `adversarial` — nếu `W4-13` cho thấy bộ sinh dùng
được ngữ cảnh dưới hạng 5 thì `rc100` đáng 1,91×.

### ⚠️ Một target Makefile chưa từng chạy được

`make eval-compare-subset CAT=… LANG=…` hỏng từ ngày viết (`W2-08-prep`): `LANG` là
**biến môi trường chuẩn**, make thừa kế nó, nên `LANG ?=` không bao giờ có tác dụng
và `--lang` luôn nhận `en_US.UTF-8`. Đổi thành `QLANG`. Chỉ lộ ra hôm nay vì đây là
lần đầu target ấy được gọi thật — cùng họ với `make eval EXP=exp_001` của `G2` (một
target chưa bao giờ tồn tại, tìm ra ở `W2-07`).

### Trạng thái

**1214 test** (1070 unit + 144 integration; +17 contrast, +10 compare) · ruff +
mypy sạch trên 114 file · **`W2` 10/10 ✅** · tổng **30/78**.

### Dự đoán: 4/7 đúng

Ba lần sai **không** cùng hướng như `W2-08` (ở đó cả bốn là đánh giá quá cao độ
phân giải). D5 và D7 sai vì tôi **suy từ cấu trúc mà chưa từng đo cấu trúc**: cho
rằng 15 metric gần độc lập, và cho rằng reranker là bậc lớn nhất. Cả hai kiểm được
bằng một phép đo mà tôi chưa chạy lần nào.

### Nếu phiên sau bắt đầu từ đây

1. `make lint && make test` · `make up && make test-integration`.
2. **`G2`** còn đúng một tiêu chí chưa đánh dấu được: "p95 không vượt 3500 ms" là
   ngưỡng **end-to-end** mà phần sinh chưa đo lần nào → `W4-13`.
3. ⚠️ **`TD-21` (mới)**: `contrast.py` mới có `category`/`lang`. Chiều đáng giá nhất
   cho `W4` là **document/tenant** (`TD-18`), mà `RunScores` chưa mang trường đó —
   và đọc `W2-09` §2 trước: chia theo 60 tài liệu thì mỗi nhóm 3–4 câu, không nhóm
   nào có lực.
4. ⚠️ `TD-20` trước khi quét lại `chunk_size` · `TD-19` trước `TD-13`.
5. 💡 **~440 câu** là mục tiêu định lượng cho `TD-13`, không phải một con số tròn
   chọn bừa: nó là `(0,2168/0,15)² ≈ 2,1×` kích thước nhóm hiện tại.
6. ⚠️ Bẫy heredoc lặp **lần thứ sáu** — lần này `bash` chết ngay với "unexpected EOF"
   khi viết `contrast.py`, nên không có file hỏng nào lọt qua. Cứ dùng Write.

---

## Phiên 2026-08-22 (3) · `W3-01` — Docling loader

**Mục tiêu phiên:** `W3-01` — loader 6 định dạng + heading hierarchy, DoD là *PDF
2 cột đọc đúng reading order* và *bảng giữ được cấu trúc*.

### Đã làm

* `packages/rag_core/loaders/` — 4 module:
  * `base.py` — `LoadedDocument` · `Heading` · `ParseFingerprint` · `section_path_at()`
  * `plain.py` — `.txt`, **hàm đồng nhất**, không strip/normalise gì
  * `docling_backend.py` — lazy import, `_normalise_depth`, `do_ocr=False`
  * `__init__.py` — bảng định tuyến theo đuôi file
* `scripts/make_loader_fixtures.py` + `make loader-fixtures` — sinh 6 fixture, idempotent
* `scripts/loader_probe.py` + `make loader-probe` — đo corpus trước/sau
* `tests/unit/test_loaders.py` — 63 test; `test_architecture_boundaries.py` thêm `docling`
* `pyproject.toml` — extra `ingestion` (`docling>=2.121,<3`)

### Ba thứ đáng nhớ

1. ⭐ **Fixture được dựng cho "sạch" là fixture nằm ngoài phân bố.** PDF hai cột
   bản đầu — mỗi cột 4 dòng ngắn bằng nhau, cách đều — làm docling đọc **sai**
   thứ tự. Quét theo số dòng: **4 ✗ · 12 ✓ · 24 ✗ · 40 ✓**, và ở `n=24` cột phải
   bị phân loại thành `table`. Không đơn điệu ⇒ không phải thư viện hỏng, mà là
   trang nằm ngoài phân bố huấn luyện của model bố cục. Cột văn xuôi có độ dài
   dòng so le: **4/4** biến thể đúng.
   💡 Nếu bản đầu tình cờ rơi vào `n=12` thì tôi đã có test xanh, DoD ✅, và một
   kết luận không có cơ sở nào — **và không nhìn output nào phát hiện ra được**,
   vì thứ đo nhầm là do chính tôi dựng ra.

2. ⭐⭐ **0/280 span golden sống sót, và cơ chế không phải cái tôi đoán.** Tôi
   đoán đúng "0/60 đồng nhất byte" (D1) rồi tưởng đã hiểu lý do. Ba con số:
   độ dài còn **91,15%**, dòng còn nguyên **2,70%**, span sống **0/280**. Chữ
   **không mất** — dòng bị **dồn** thành đoạn (+ code fence, bỏ `\r`), nên mọi
   offset phía sau trượt. `sha256` vẫn khớp manifest, không test nào đỏ, mọi
   `chunk_id` vẫn tồn tại → `TD-22`, làm **trước** `W3-07`.

3. ⚠️ **Cùng một tài liệu 3 cấp heading ra ba cách đánh số.** `.docx` cho
   `section_header` level 1/2/3; `.md`/`.html` cho `title` rồi level 1/2; `.pptx`
   ba `title`; `.xlsx` không có heading nào. Tin thẳng `item.level` thì cùng một
   heading là **cấp 3 khi tới từ DOCX và cấp 2 khi tới từ HTML** →
   `_normalise_depth`. `W3-03` phải đọc §4 của report trước khi bắt đầu.

### Ba chỗ tự bẫy mình

4. `SequenceMatcher.ratio()` trên hai chuỗi 500 KB **không chạy xong** (bậc hai),
   không phải chạy chậm — treo 60 tài liệu. Đổi sang tỉ lệ dòng còn nguyên đếm
   bằng `Counter`; chính con số đó mới lộ ra cơ chế "dồn dòng" ở (2). Nếu
   `SequenceMatcher` chạy xong thì nó cho ~0,9 và tôi đã kết luận "gần như không
   đổi" — sai hoàn toàn.
5. Đặt `book.properties.modified` **trước** khi save là không đủ: openpyxl ghi đè
   bằng giờ hiện tại ngay trong lúc serialize (6 tiến trình → 3 chuỗi byte).
   Phải sửa `docProps/core.xml` **sau** khi save.
6. Test đầu của tôi tìm `b"(for several sentences"` trong PDF, mà văn bản wrap ở
   34 ký tự nên chuỗi ấy không bao giờ là đầu dòng. Đỏ ngay lần chạy đầu — đúng
   kiểu lỗi mà test nên bắt, chỉ là nó bắt chính tôi.
7. ⚠️ `.gitignore` có `*.pdf` nên fixture PDF **không vào repo**: clone sạch thì
   `test_fixture_exists` đỏ, còn trên máy tôi mọi thứ xanh. Bắt bằng
   `git add -A --dry-run`, không bằng test. Thêm `!tests/fixtures/loaders/*.pdf`.
8. ⚠️⚠️ **Rồi tôi suýt ghi một lỗi chưa tái lập được vào lịch sử git.** Thấy
   `* text=auto`, tôi suy: PDF toàn ASCII không NUL → git coi là text → checkout
   Windows đổi LF→CRLF → xref ghi offset tuyệt đối → hỏng file. Lập luận chặt, đã
   commit như một bản vá. **Đo lại thì không đổi**: clone ở commit trước bản vá
   với `core.autocrlf=true` cho `w/lf` và sha256 y nguyên — heuristic của git tự
   nhận ra nó là binary. Đã amend lại commit cho đúng. Giữ dòng khai báo vì nó
   biến điều đang đúng **nhờ heuristic** thành điều đúng **vì được khai báo**, và
   heuristic ấy đọc nội dung file.
   💡 Ba lỗi 6–8 cùng hình dạng "xanh trên máy tôi, chưa biết trên clone sạch".
   Hai cái đầu thật, cái thứ ba tưởng tượng — **không phân biệt được bằng suy
   luận, chỉ bằng cách clone ra thử.**

**Kiểm chứng:** 1277 test — 1276 passed, 1 skipped, exit 0, 377,42 s ·
`make lint` sạch (ruff + `mypy` 119 file).

### Lệnh để tiếp tục

```bash
make loader-fixtures    # sinh lại 6 fixture (idempotent)
make loader-probe       # 0/60 đồng nhất byte · 0/280 span sống
uv run pytest tests/unit/test_loaders.py
```

Việc tiếp theo: **`W3-02`** (OCR fallback — thừa hưởng số 70,56 s vs 0,12–0,77 s)
hoặc **`TD-22`** (ghim `text_sha256` vào manifest, chặn `W3-07`).

---

## Phiên 2026-08-22 (4) · `W3-02` — Scan detection + OCR fallback

**Mục tiêu phiên:** `W3-02` — DoD là *PDF scan ra được text* và *có queue control
tránh OOM*.

### Đã làm

* `packages/rag_core/loaders/scan.py` — `PageText` · `ScanReport` · `detect_scan`
  · CLI `python -m rag_core.loaders.scan --per-page` (cũng là công cụ hiệu chỉnh)
* `packages/rag_core/loaders/ocr.py` — `OcrGate` · `require_ocr_support` ·
  `OCR_VERIFIED_LANGUAGES` · `SECONDS_PER_PAGE`
* `load_document(..., ocr="off"|"auto"|"force", language=…, gate=…)`
* Fixture thứ 7: `scanned-page.pdf` — ảnh bilevel 150 DPI, 7,8 KB, không text layer
* `make scan-probe` · marker `integration` nới nghĩa sang "hoặc trọng số model"
* +26 unit (`test_scan_detection.py`) + 9 integration (`test_ocr_fallback.py`)

### Ba thứ đáng nhớ

1. ⭐⭐ **OCR đọc tiếng Anh nguyên văn và trả rác cho tiếng Việt.** Fixture cố ý
   đặt hai đoạn cùng nội dung khác ngôn ngữ **trong cùng một ảnh** — cùng font,
   cùng DPI, cùng một lần chạy, nên không lẫn biến nào.

   | model rec | tiếng Anh | tiếng Việt |
   |---|---|---|
   | `ch` PP-OCRv6 (mặc định) | ✅ nguyên văn | `Tāng trng t 7,09 phān trām nām 2024` |
   | `latin` PP-OCRv3 | ✅ nguyên văn | `Tng trXXng XXt 7,09 phn trm nm 2024` |

   Gốc: `Tăng trưởng đạt 7,09 phần trăm năm 2024`. Con số sống sót cả hai bên,
   dấu thì không. Model `latin` — thứ tôi tìm tới **vì** tiếng Việt dùng chữ
   Latin — hoá ra **tệ hơn** model mặc định.
   💡 Nên `require_ocr_support("vi")` **ném lỗi**: rỗng thì có người thấy, còn
   rác thì trông như nội dung và đi thẳng vào embedding → index → citation.

2. ⭐ **Báo cáo born-digital THẬT vẫn có trang trống** — 1/129 và 4/112. Luật
   hiển nhiên "một trang thiếu text ⇒ scan" sẽ đẩy **100%** báo cáo World Bank
   vào OCR. Ngưỡng phải theo **tỉ lệ**, và cả hai hằng số nằm giữa hai cụm cách
   nhau rất xa. ⚠️ Fixture born-digital của tôi ở **8,17 ký tự/in², thấp hơn p05
   của tài liệu thật** — lần thứ hai liên tiếp fixture tự sinh không đại diện;
   lần này tôi tải PDF thật **trước** khi chọn hằng số.

3. ⚠️⚠️ **Đính chính `W3-01`, và nó đổi lý do cả hạng mục.** "OCR đắt hai bậc độ
   lớn (70,56 s vs 0,12–0,77 s)" là **cold start vs cold start**. Đo 5 lượt liên
   tiếp trong một tiến trình: **12,67 · 0,34 · 0,34 · 0,35 · 0,36** s. Cận biên
   **0,35 s/trang** — gấp ~3 lần, không gấp 500. Nên lý do `W3-02` đáng làm
   không phải chi phí mà là **đúng/sai** (không phát hiện thì PDF ảnh trả rỗng →
   `LoaderError` → tài liệu biến mất khỏi index, im lặng) và **chốt ngôn ngữ**.
   `OcrGate.max_pages` 50 → **500** theo đó.

### Hai chỗ tự bẫy mình

4. Fixture scan **không idempotent**, đúng lỗi openpyxl của phiên trước: ép
   `/CreationDate` rồi tưởng xong, còn `/ModDate`. Ba lần chạy ba hash, khác
   **đúng một byte**, và chỉ lộ ra khi hai lần chạy cách nhau **hơn một giây** —
   gọi hai lần liên tiếp trong cùng tiến trình thì trùng, nên phép thử đầu của
   tôi báo "ổn".
5. Đi tìm model `latin` mất ba lượt vì `Rec.ocr_version` là **enum** (không nhận
   chuỗi), `OCRVersion` **không có** `PPOCRV3` dù file model tên `PP-OCRv3`, và
   `rapidocr_params` truyền qua docling **không** đổi được engine (nó rơi về
   onnxruntime rồi `ImportError`). Phải gọi thẳng `RapidOCR` mới so được hai
   model — tức tách "thư viện có làm được không" khỏi "tôi cấu hình đúng chưa".

### DoD: đạt một nửa, và nửa kia là kết luận

✅ queue control · ✅ PDF scan ra text **với tiếng Anh** · ❌ **với tiếng Việt** —
và đó không phải phần chưa làm xong. Mở khoá cần `OPENROUTER_API_KEY` (→ VLM,
đúng như plan viết) hoặc Tesseract `vie` trong image. Cả hai đều ngoài tầm commit
này: đã kiểm, môi trường không có key OpenRouter, DeepSeek không có model thị
giác, máy không có `tesseract` lẫn `onnxruntime`.

### Lệnh để tiếp tục

```bash
make scan-probe SCAN=<file.pdf>          # mật độ text layer từng trang
uv run pytest tests/unit/test_scan_detection.py
uv run pytest tests/integration/test_ocr_fallback.py
```

**Kiểm chứng:** 1312 test — 1311 passed, 1 skipped, exit 0, 387,38 s ·
`make lint` sạch (`mypy` 123 file, thêm `pypdfium2.*` vào `ignore_missing_imports`).

Việc tiếp theo: **`W3-03`** (structure-aware chunker) — hạng mục đầu tiên tiêu
thụ `LoadedDocument.headings` + `section_path_at()`.

---

## Phiên 2026-08-22 (5) · `W3-03` — Structure-aware chunker

**Mục tiêu phiên:** `W3-03` — DoD là *`section_path` đúng trên tài liệu có 3 cấp
heading*.

### Đã làm

* `packages/rag_core/chunking/structure.py` — `StructureChunker` ·
  `common_ancestor` · `section_boundaries` · `ChunkingStrategy.STRUCTURE` ·
  config `structure_merge_short_sections`
* `chunking/base.py` tách thành hai điểm mở rộng `_prepare_pieces` /
  `_section_path_for(doc, index)` — mọi chunker cũ **không đổi hành vi**
* `scripts/structure_probe.py` + `make structure-probe` / `make structure-corpus`
* Sửa lỗi `break` ở `loaders/base.py::section_path_at` (lỗi của `W3-01`)
* +24 unit (`tests/unit/test_structure_chunker.py`)

### Bốn thứ đáng nhớ

1. ⭐ **Gộp mảnh ngắn qua ranh giới section thì `section_path` nói dối.** Một
   `Điều 4` ngắn bị gộp vào cuối `Điều 3` → chunk mang đường dẫn của `Điều 3` mà
   nửa sau nội dung thuộc `Điều 4`. Không lỗi, không cảnh báo, citation trỏ sai
   điều luật. **Đúng khuôn** lỗi bản POC gộp chunk qua ranh giới *tài liệu*
   (docstring `chunking/base.py` điểm 2), thấp hơn một cấp.
   💡 Cấm gộp không phải lối ra: `wb1.pdf` ra **735 chunk, nhỏ nhất 1 ký tự**.
   Lối ra là **hạ `section_path` xuống tổ tiên chung** — 64/1061 chunk (6,0%)
   đổi một đường dẫn *sai* lấy một đường dẫn *nông hơn*.

2. ⭐ **Tài liệu PDF thật chỉ cho MỘT cấp heading.** `wb1` 128 heading, `wb2`
   83 — **tất cả cấp 1**. Backend PDF của docling gán nhãn `section_header`
   nhưng không suy ra cấp từ bố cục. Nên với PDF, `section_path` là danh sách
   một phần tử: "chunk này ở mục nào", không phải một đường dẫn phân cấp.
   ⚠️ `DocType.LEGAL` muốn kiểm chứng `section_path` thì văn bản pháp luật phải
   vào corpus ở dạng **DOCX/HTML**, không phải PDF → `TD-24`.

3. ⭐ **Corpus hôm nay: 0/60 tài liệu có cấu trúc**, nên chunker thoái hoá về
   fixed ở 60/60 (có test ghim output phải **trùng khít** `FixedSizeChunker`).
   Cùng một báo cáo *Vietnam STI 2020*: bản `.txt` trong corpus **0 heading**,
   bản `.pdf` gốc qua docling **128 heading**. Ràng buộc nằm ở **định dạng
   corpus**, không ở chunker.

4. ⚠️ **Lỗi `break` của `W3-01` chỉ lộ ra khi có người tiêu thụ.**
   `section_path_at` viết `break` cho cả heading không định vị được, nên **một**
   heading hỏng làm mù toàn bộ phần sau. `wb1.pdf` có 6/128 → **575/587 chunk
   (98,0%)** nhận đường dẫn khác; `wb2.pdf` có 0/83 → **0%**. Đo một tài liệu
   thì 1/2 khả năng kết luận "không sao".

### Hai chỗ tự bẫy mình

5. **Vị trí cắt ≠ vị trí hỏi.** Ranh giới lùi về đầu dòng để `##` đi cùng
   heading, nhưng hỏi `section_path_at` **tại chỗ cắt** là hỏi tại ký tự `#`, và
   với hàm ấy heading chưa bắt đầu → **486/587 chunk** mang đường dẫn của section
   liền trước. `section_boundaries` giờ trả **cặp** `(cắt, hỏi)`.
6. **Script kiểm chứng lặp lại đúng lỗi nó đi kiểm** — bản kiểm đầu cũng hỏi tại
   *đầu* chunk, nên sau khi sửa vẫn báo 54 chunk sai, và 54 cái đó là chunk
   **đúng bị chấm sai**. Phải hỏi tại **ký tự cuối** của chunk.

### Lệnh để tiếp tục

```bash
make structure-corpus                      # 0/60 — corpus có gì cho chunker ăn?
make structure-probe STRUCT=<file.pdf>     # heading, phân bố chunk, số chunk nói dối
uv run pytest tests/unit/test_structure_chunker.py
```

Việc tiếp theo: **`W3-04`** (Contextual Retrieval — cần GPU thuê `W0-05`), hoặc
đảo lên `W3-05`/`W3-06`. Đọc `TD-24` trước: `W3-06` không đo được cho tới khi
corpus có tài liệu mang cấu trúc.

---

## Phiên 2026-08-22 (6) · `W3-06` — Token-based sizing

**Mục tiêu phiên:** `W3-06` — DoD là *không chunk nào vượt max token của model*.

### Đã làm

* `packages/rag_core/chunking/tokens.py` — `TokenCounter` (Protocol) ·
  `calibrate_density` · `fit_to_budget` · `TokenSizingUnavailable`
* `ChunkingConfig.size_unit = "chars" | "tokens"` + `Chunker.sizing` (config đã
  quy về ký tự cho **tài liệu hiện tại**) + `Chunker(token_counter=…)`
* `scripts/token_sizing_probe.py` + `make token-probe`
* +34 unit (`tests/unit/test_token_sizing.py`)

### Ba thứ đáng nhớ

1. ✅ **DoD đã đạt sẵn, và đã có số từ `W2-01`.** `truncation-bgem3.json`:
   **0/15.814** chunk bị cắt, token max **734** trên cửa sổ **8192** — dư
   **11,2×**. Nên hạng mục này không phải "làm cho hết bị cắt"; câu hỏi thật là
   *`chunk_size` bằng ký tự có phải cái núm dùng được không*.

2. ⭐ **Không, và vì hai lẽ.** Cùng một bộ 15.814 chunk: đổi tokenizer là đổi số
   token tới **47%** (p50 EN 313 với PhoBERT vs 212 với BGE-M3). Và **chiều lệch
   giữa hai ngôn ngữ đảo dấu** — BGE-M3 cho EN 5,83 ký tự/token vs VI 5,41;
   PhoBERT cho EN 4,09 vs VI 5,33. "Tiếng Việt tốn token hơn" là tính chất của
   **cặp (model, ngôn ngữ)**.
   💡 Nên giá trị của token sizing là **san bằng hai nửa corpus**, không phải
   tránh bị cắt: chênh p50 EN↔VI với PhoBERT đi từ **−22,3% xuống +2,4%**.

3. ⚠️⚠️ **Chép lại một lập luận ĐÚNG sang ngữ cảnh khác thì thành sai.**
   `truncation.py` dùng phân vị 5 cho mật độ ký tự/token, có lý do đã ghi: nó
   *gợi ý* một ngân sách mà **không ai kiểm lại**. Ở đây `fit_to_budget` kiểm
   chính xác, nên chừa biên không mua được gì mà làm `chunk_size=256` thật ra có
   nghĩa là **213** (−17%). Đổi sang trung bình: **261** (+2%), cả hai đều 0
   vượt trần.

### Hai chỗ tự bẫy mình

4. ⚠️⚠️ **Lỗi trộn đơn vị, và 28 test xanh không thấy.** Tôi viết docstring
   `sizing` dặn "mọi chỗ phải đọc `self.sizing`, không đọc `self.config`" rồi để
   `FixedSizeChunker`/`StructureChunker` đọc `self.config` — **12 chỗ**, nhận 256
   **token** dùng như 256 **ký tự**. Cả 28 test đều kiểm **trần**, mà trần do một
   đường code khác (`_fit_tokens`) bảo đảm; không test nào kiểm **đích**. Corpus
   thật mới lộ: chunk tiếng Việt p50 = **71 token** thay vì 256.
   💡 Đã thêm test kiểm đích (p50 phải trong `[0,5×; 1,2×]` ngân sách) và một test
   **quét chính source** tìm `self.config.<kích thước>`. Cấy lại lỗi thì 4 test đỏ.

5. ⚠️ **Probe đầu tiên đo một corpus không tồn tại.** `Path.read_text()` trên
   Windows chuẩn hoá CRLF → LF: 14.075.359 ký tự và **18.038 chunk** thay vì
   14.284.300 và **15.814**. Chỉ phát hiện được vì lệch `index-baseline.json`.
   Đọc byte rồi decode thì tái lập `truncation-baseline.json` **từng chữ số**.
   ⚠️ Đây đúng cái ví dụ tôi dùng làm test cho guard `_usable_structure` ở
   `W3-03` một hạng mục trước — viết được phép kiểm cho một chế độ hỏng không có
   nghĩa là nhận ra nó khi nó xảy ra với mình.

### Cố ý không làm

Không đổi `configs/indexing/bgem3.yaml` sang chế độ token: bật lên là
**15.814 → 11.190 chunk**, tức index khác và mọi con số `W2` không so được. Và
đổi `chunk_size` là chiều mà `TD-20` đã chỉ ra không kiểm định cặp được. →
`TD-25`, thuộc `W3-09`.

### Lệnh để tiếp tục

```bash
make token-probe                  # bảng ký tự vs token, cả hai model
make token-probe TOKENS=512
uv run pytest tests/unit/test_token_sizing.py
```

Việc tiếp theo: **`W3-05`** (parent-child) — không cần GPU, không cần key, không
vướng corpus.

---

## Phiên 2026-08-22 (7) · `W3-05` — small-to-big

**Mục tiêu phiên:** `W3-05` từ đầu tới cuối — code, test, index thật, phép đo, báo cáo.

**Kết quả:** xong. `W3` **5/9**. Full suite **1399 test, exit 0**; `ruff` + `mypy` sạch.

### Quyết định kiến trúc: parent không nằm trong index

Cách quen thuộc (index cả parent lẫn child, lọc parent khỏi search) tốn một field
phẳng trong payload + một field `MetadataFilter` + một payload index + một lượt
backfill `rag_bgem3` — tức chạm đúng tầng `W2-06` vừa đo latency xong. Và quên
filter một chỗ là parent lọt vào kết quả, âm thầm.

Ở đây **parent là tập các child của nó**: `parent_chunk_id` là khoá gom nhóm, id
anh em nằm trong `extra["parent_children"]`, assembly dựng lại parent bằng một
lời gọi `fetch_chunks`. Hai hệ quả bắt buộc đi kèm: child cùng parent **không
chồng lấn** (giữ overlap thì đoạn chồng lấn xuất hiện hai lần trong parent ghép
lại), và `_enforce_size` áp **trong phạm vi một parent** (gộp qua ranh giới parent
làm `parent_chunk_id` nói dối — lần **thứ ba** của cùng một hình dạng lỗi, sau
POC gộp qua ranh giới tài liệu và `W3-03` qua ranh giới section).

### ⚠️⚠️ Lỗi đắt nhất phiên: Protocol khớp với một cái tên không tồn tại

`context.py` gọi `fetcher.get_by_ids(...)`; method thật của `QdrantDenseRetriever`
là `fetch_chunks`. 27 unit test xanh và `mypy` sạch, vì `RecordingFetcher` trong
test **cũng** khai `get_by_ids`. Lỗi lộ ra ở probe đầu tiên, sau 328 giây build.

Bài học rộng hơn hạng mục: *một Protocol cấu trúc không ràng buộc được gì nếu cả
hai bên đối chiếu đều do test dựng ra.* Khó chịu nhất là **khuôn để bịt đã có sẵn
từ `W3-06`** (`test_token_sizing.py:427` ghim `TokenCounter` vào lớp thật) và tôi
không dùng lại. Quét cả 4 Protocol của repo thì `ChunkFetcher` là trường hợp duy
nhất bị hở.

Bịt: `runtime_checkable` + 2 test ghim vào lớp thật (`isinstance` và
`inspect.signature`). Tiêm lại lỗi: đúng 2 test mới đỏ, 27 test cũ vẫn xanh.

### ⭐⭐ Độ nở ngữ cảnh là một chỉ số đánh lừa

242 câu golden, hai index khác nhau **đúng một biến** (cỡ child; parent giữ
~1024 token ở cả hai):

| index | child | k | parent tb | dedupe | token child | **token parent** | nở |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pc256` | ~256 tok | 10 | 9,00 | 10,0% | 2.750 | **9.471** | 3,45× |
| `pc128` | ~117 tok | 10 | 8,70 | 13,0% | 1.370 | **9.519** | 6,98× |

Chia đôi child làm "độ nở" **gấp đôi** trong khi prompt thật lệch **+0,5%**.
Prompt = *số parent riêng biệt* × *cỡ parent*; child không có trong công thức.
Hệ quả đi ngược trực giác: **child nhỏ hơn không đắt hơn** — thêm 3 điểm dedupe
và vector mịn hơn ở cùng ngân sách prompt.

Và `chunk_size=256` token hoá ra **không phải "small"**: child `pc256` là 256
token còn chunk baseline chỉ 218 token (`W3-06` §2), tức child *to hơn* baseline
17%. 256 được chọn vì nó tròn. Đó là lý do `pc128` tồn tại.

### Dedupe: có thật, nhỏ

k=3 → 5,0% · k=5 → 7,3% · k=10 → **10,0%** (148/242 câu gộp được ≥1 lần) · k=20 →
13,3% (218/242). Không phải code chết, nhưng chỉ tiết kiệm ~14% prompt. Cái đắt
là **số token parent tuyệt đối**: ở k=10 prompt đi từ 2.750 lên 9.471 token.

→ Ràng buộc mới cho `W3-09`: so "k=10 child" với "k=10 parent" là so hai thứ
**khác giá**. Ablation phải ghép cặp theo **ngân sách token**, không theo k.

### Filter, đo trên store thật

Sau vụ Protocol thì fake không còn đủ để tin. Trên `rag_pc256`:
`MetadataFilter(lang=en)` chặn **32 anh em**, **9/10 parent** thành
`complete=False` với `missing_children` đủ id. Index sạch: **0/968 lượt** thiếu
anh em. Ranh giới đã ghi vào docstring: `filters` chỉ chắn đường **lấy thêm anh
em**, không chắn các child đã trúng — tầng serving `W4` phải truyền đúng filter
đã dùng cho lượt search.

### Cố ý không làm

Không đụng `bgem3.yaml`; không tuyên bố chất lượng truy hồi (bộ chunk khác nên
nhãn ánh xạ khác — `TD-20`/`TD-25`); không nối vào `CachedChunker`/serving →
`TD-26`.

### Lệnh để tiếp tục

```bash
make pc-probe PCCFG=pc256 PCK=10   # gộp trùng + độ nở, index child 256 token
make pc-probe PCCFG=pc128 PCK=10   # child 128 token, parent giữ nguyên ~1024
uv run pytest tests/unit/test_parent_child.py
```

Việc tiếp theo: **`W3-08`** (ingestion worker arq) — `W3-04` chờ GPU thuê
(`W0-05`), `W3-07` chờ `TD-22`.

---

## Phiên 2026-08-22 (8) · `TD-22` — ghim văn bản parse ra

**Mục tiêu:** đóng `TD-22` để `W3-07` hết bị chặn (`W3-04` để anh làm khi có GPU;
`W3-09` phải đứng sau `W3-04`).

**Kết quả:** đóng. `ruff` + `mypy` sạch (131 file). 13 test mới.

### Nợ hoá ra sâu hơn hai tầng so với lúc ghi

Nợ viết là "manifest ghim `sha256` của byte, không ghim văn bản đã parse". Đúng,
nhưng chưa đủ:

**Tầng 1 — vân tay ghim tên gói ô dù.** `W3-01` ghi đúng một số version: của
`docling`. Nhưng `DoclingDocument.export_to_markdown` — hàm sinh ra
`LoadedDocument.text` — sống trong **`docling-core`**, và `docling==2.121.0` cho
nó chạy tự do trong `>=2.91.0,<3.0.0`. `docling-core` dịch chuyển là markdown đổi
trong khi vân tay đứng yên. Cũng vậy với `docling-parse`, `pypdfium2` (thả **hai
major version**), `docling-ibm-models`, `rapidocr`. → `ParseFingerprint.components`,
ghi **theo đường parse** chứ không ghi tất (ghi thừa cũng có giá: nâng `rapidocr`
làm mọi DOCX báo "parser đã đổi" trong khi chúng chưa từng chạm OCR, và một cảnh
báo kêu suốt là một cảnh báo bị tắt).

**Tầng 2 — ghim đủ version gói vẫn chưa ghim được PDF.** Pipeline PDF nạp trọng
số từ HF, và hai model được đối xử khác nhau: model **bảng** ghim `revision="v2.3.0"`,
model **bố cục** dùng `revision="main"` — một nhánh di động. Model bố cục quyết
định thứ tự đọc trang, tức thứ tự đoạn trong markdown, tức mọi offset span. Một
lượt push lên `main` đổi văn bản mà **không con số version nào trên máy nhúc
nhích**. → ghim **commit SHA đã phân giải** đọc từ cache HF. Nhưng đó chỉ làm lần
đổi *ồn ào*, không *ngăn* được → `TD-27`.

### Đo được

* `text_sha256 == sha256`: **60/60** — hôm nay hai cột mới hoàn toàn thừa, và đó
  chính là thứ hết hạn khi có parser.
* Nối loader vào `corpus_loader` **không đổi gì**: `Document.content ==
  payload.decode("utf-8")` 60/60, tổng **14.284.300** ký tự — khớp từng chữ số
  với `W3-06` §7 và `index-pc256.json`.
* Dựng lại chế độ hỏng trên tài liệu thật (chép **nguyên byte** sang `.md`):
  sha256 byte trùng khít `8da126fbc1a136b7`, văn bản **4.688 → 3.673 ký tự
  (−21,6%)**. Kiểm-theo-byte không thấy gì.

### Quyết định thiết kế đáng ghi

"Chưa ghim" **không** phải lúc nào cũng là lỗi, và luật phải có lý do chứ không
theo giá trị: `if entry.text_sha256:` nghĩa là *rỗng = bỏ qua*, tức đúng chế độ
hỏng đang đi sửa. Luật theo **loader**: `plain` chấp nhận (hàm đồng nhất, `sha256`
đã ghim văn bản); mọi loader khác **báo lỗi**. Nên `W3-07` về mặt vật lý không
thêm được một `.docx` mà quên ghim.

Trường hợp thứ ba cố ý không phải lỗi: vân tay đổi mà văn bản không đổi — chỉ
`WARNING`, vì để manifest mang vân tay cũ nghĩa là lần lệch **thật** sau này sẽ
chỉ sai thủ phạm.

### ⚠️ Lỗi của tôi trong phiên

Ba assert đầu tiên trong test mới là vô nghĩa: một tautology `x == x`, một
`... or True`, một biến alias thừa. Đúng loại "test không bao giờ đỏ được" mà
`W3-05` §3 vừa viết cả một mục — hai phiên liền cùng một hình dạng. Đã thay bằng
phép kiểm thật.

### Lệnh để tiếp tục

```bash
make corpus-pin                 # báo cáo (không sửa gì)
make corpus-pin PIN_ARGS=--write
make parse-pin-probe            # dựng lại chế độ hỏng rồi cho xem nó bị chặn
uv run pytest tests/unit/test_parse_pin.py
```

Việc tiếp theo: **`W3-07`** (content-hash dedupe + incremental re-index), rồi
**`W3-08`**. `W3-04` chờ GPU thuê; `W3-09` chờ `W3-04`.

---

## Phiên 2026-08-22 (9) · `W3-07` — re-index tăng dần

**Mục tiêu:** `W3-07` (nợ `TD-22` đã đóng ở phiên trước nên hết chặn).

**Kết quả:** xong. `W3` **6/9**. `G3` được một tiêu chí. 7 test tích hợp mới.

### DoD, đo trên corpus thật

Chép corpus sang thư mục tạm, sửa **một dòng** ở giữa tài liệu dài nhất
(990.826 byte, 1.082 chunk), build ba lượt:

| lượt | embed | mượn lại | giây embed |
|---|---:|---:|---:|
| sạch | 15.814 | 0 | 376,6 |
| không sửa gì | 0 | 0 | 0,0 |
| **sửa một dòng** | **6** | **1.076** | **2,1** |

Embed ít hơn **2.636×**, nhanh hơn **179,3×** → `G3` "reprocess ≥ 10×" **đạt**.

### Không dựng cache vector — Qdrant đã là cache

Phản xạ đầu là một cache `(model, content_hash) → vector`: 65 MB trên đĩa cộng
một vòng đời cache nữa phải bảo trì. Nhưng vector **đã nằm trong point cũ**; chỉ
thiếu đường đọc lại. `fetch_vectors` + `upsert_reusing` + `DocState.chunk_hashes`
là đủ — không file mới, không schema mới, không backfill.

### ⭐⭐ Hai con số mâu thuẫn, và chỗ hoà giải là phát hiện

Ca tổng hợp: chèn một câu ở vị trí 5/300 → mượn lại **2,0%**. Corpus thật: chèn
62 byte vào giữa → **99,4%**. Chênh lệch ấy nghĩa là tôi chưa hiểu cơ chế, nên
không được viết báo cáo cho tới khi hiểu.

Đo thẳng vào nghi ngờ — cùng văn bản, cùng điểm chèn, chỉ khác có `\n\n` hay
không:

| văn bản | mượn lại |
|---|---:|
| một mạch, không `\n\n` | **2,0%** |
| `\n\n` mỗi 9 câu | **98,0%** |
| `\n\n` mỗi 3 câu | **98,0%** |

`separators` là `("\n\n", "\n", ". ", " ", "")` **theo thứ tự ưu tiên**, nên mỗi
`\n\n` là một **điểm đồng bộ lại**: thiệt hại của một lần chèn bị chặn trong
khoảng cách tới lần xuống dòng đoạn kế tiếp. Điều đó nói ra loại tài liệu mà kỹ
thuật này không giúp được — đầu ra OCR mất cấu trúc dòng, bảng chuyển thành văn
xuôi, transcript không chấm câu. → `TD-28`.

### ⚠️ Hai lỗi của tôi trong phiên

**Fixture đầu cho 99% ở MỌI ca.** Mỗi "trang" 146 ký tự vừa khít chunk 200, nên
ranh giới do nội dung quyết định và chèn thêm chữ không dịch được gì. Lần **thứ
ba** trong `W3` (sau `W3-01` §2, `W3-05` §9) — và lần này khó nghi ngờ hơn vì
con số *có lợi*.

**`chunk_size` có thể hoàn toàn vô tác dụng.** Truy vì sao "100 trang" chỉ ra 10
chunk: `min_chunk_size` mặc định **bằng đúng** `chunk_size=200` tôi khai, nên mọi
mảnh đều bị coi là quá ngắn và gộp tới `max_chunk_size=1500` — chunk trung bình
**1.460 ký tự, gấp 7,3×** con số đã khai, im lặng. Sửa bằng **cảnh báo chứ không
phải lỗi**: "cắt mịn rồi đóng gói to" là chiến lược hợp lệ và có test dùng đúng
nó (guard cứng làm đỏ 9 test); cái sai là sự im lặng.

### Cố ý không làm

Dedupe liên tài liệu: đo trên 15.814 chunk baseline thì chỉ **51 chunk (0,32%)**
trùng nội dung, 6 hash trùng giữa các tài liệu, và ba hash lặp nhiều nhất là
chunk mở đầu bằng 70 dấu cách. Không đáng một tầng tra cứu.

### Lệnh để tiếp tục

```bash
make incr-probe                 # sửa một dòng trong corpus thật, đo lại
uv run pytest tests/integration/test_incremental_reindex.py
```

Việc tiếp theo: **`W3-08`** (async ingestion worker, arq). Redis đã sẵn trong
`docker-compose` và `settings.redis_url` đã có; thiếu extra `fastapi`/`arq`.
`W3-04` anh làm khi có GPU; `W3-09` chờ `W3-04`.

---

## Phiên 2026-08-22 (10) · `W3-08` — worker ingestion

**Mục tiêu:** `W3-08`, hạng mục cuối của `W3` không cần GPU.

**Kết quả:** xong. `W3` **7/9**; hai hạng mục còn lại (`W3-04`, `W3-09`) đều chờ
GPU thuê. `G3` **2/3**. 20 test tích hợp mới.

### DoD

| vế | đo được |
|---|---|
| `POST` trả `job_id` < 200 ms | **p50 3,47 ms · p95 4,26 ms** (50 lượt) — dư 47× |
| `GET` có progress | có, đơn điệu tăng theo từng tài liệu |
| retry khi worker chết | có — nhưng bốn đường khác nhau, xem dưới |

### ⭐⭐ `max_tries` không phải cái tôi tưởng

Tôi viết một test để chứng minh `max_tries = 3` hoạt động: xoá manifest, chạy
worker, kỳ vọng `attempt == 2`. Nó đỏ ở `attempt == 1`.

`arq/worker.py:613-633`: arq chỉ thử lại `Retry`, `RetryJob`, `CancelledError`.
Một `Exception` bất kỳ là hỏng hẳn ngay lần đầu; `max_tries` không đụng tới.

Và đó là mặc định **đúng** — thử lại một job thiếu config chỉ cho ba lần hỏng y
hệt nhau và một hàng đợi che mất lỗi thật. Nên việc phân loại chuyển vào
`tasks.is_transient`, và bảng thành:

| chuyện gì | cơ chế | nhận ra sau |
|---|---|---|
| Qdrant sập | `is_transient` → `Retry` | ngay |
| worker tắt êm | `CancelledError` | ngay |
| worker **chết hẳn** | khoá `in-progress` hết hạn | **`job_timeout + 10s`** |
| sai config | không thử lại | ngay |

Test cho dòng đầu dùng **cổng đóng thật**, không phải exception dựng sẵn. Dòng
"chết hẳn" chỉ được xác minh bằng đọc mã nguồn → `TD-29`.

### ⚠️⚠️ `doc_ids` suýt thành một endpoint xoá index

`build_index` **xoá** tài liệu có trong state mà không thấy trong lượt chạy —
đúng với bộ lọc corpus, vì chúng nói *tài liệu nào THUỘC index*. Nhưng "index lại
đúng tài liệu này" là câu khác hẳn. Nếu tôi dùng lại `select_entries` cho
`doc_ids` như phản xạ đầu thì `POST /ingest {"doc_ids": ["d-1"]}` sẽ xoá 59 tài
liệu kia, im lặng, và trả 200.

`only_doc_ids` vì thế là **phạm vi lượt chạy**: không vào `fingerprint`, và tắt
bước gỡ. Có test đi tìm đúng chế độ hỏng ấy.

### ⚠️ Ba lỗi test, cả ba hỏng không giống nguyên nhân

* `Annotated[...]` khai **trong hàm** → `from __future__ import annotations` biến
  nó thành chuỗi, FastAPI phân giải ở namespace module, không thấy, nên coi tham
  số là **query param**: `422 Field required: store`.
* Test dùng chung hàng đợi Redis → worker test này nhặt job test kia. Sửa bằng
  `INGEST_QUEUE` cấu hình được — hai môi trường chung một Redis cũng cần nó.
* `client` dựng trước `workspace` → app nối hàng đợi mặc định, mọi job nằm im ở
  `queued`.

### Phát hiện phụ về chính công cụ đo

`addopts` đã có `-q`, nên `pytest -q` tôi gõ theo phản xạ suốt nhiều phiên thành
`-qq` — **tắt hẳn** dòng tổng kết. Repo đã ghi cảnh báo này ở `pyproject.toml`
từ `W2-05` và tôi vẫn lặp lại. Đã bỏ `-q` khỏi mọi lệnh.

### Vì sao `pipeline/ingest/` chứ không `serving/`

Endpoint **ghi vào index** là điều khiển từ xa của pipeline, không phải Serving
Plane. Đặt nhầm là làm nhoè ranh giới mà cả kiến trúc dựa vào, ngay ở hạng mục
đầu tiên chạm HTTP.

### Lệnh để tiếp tục

```bash
make ingest-api      # uvicorn ở 127.0.0.1:8001, /docs có sẵn
make ingest-worker   # arq
uv run pytest tests/integration/test_ingest_job.py
```

Việc tiếp theo: **`W3-04`** (contextual retrieval, cần GPU thuê — anh làm), rồi
**`W3-09`** (ablation #2) mới chạy được.

---

## 2026-09-03 · `W0-08` xong · `W3-04` chuẩn bị xong (lượt chạy GPU còn lại)

Mục tiêu phiên: làm **mọi thứ có thể hỏng trước khi thuê GPU**, để phiên pod chỉ
còn là chạy. Pod tính tiền theo giờ, nên mỗi lỗi phát hiện trên đó đắt hơn cùng
lỗi ấy ở laptop đúng bằng tiền thuê.

### Cái đáng nhớ nhất: tôi thiết kế sai chiều

Trước khi viết code tôi phân tích kỹ **kích thước cửa sổ ngữ cảnh** — chiều duy
nhất tính toán được từ trước — và kết luận đúng: 28/60 tài liệu vượt cửa sổ
40.960 token của Qwen3-8B, nên công thức gốc của Anthropic không áp thẳng được.

Rồi dry-run 60 request giá **$0,055** tìm ra hai lỗi mà **không suy luận thiết kế
nào bắt được**, và cả hai đều nghiêm trọng hơn chiều tôi đã phân tích:

1. **83% completion token là chuỗi suy luận**, 6/30 request trả rỗng.
2. **17% ngữ cảnh đúng ngôn ngữ** — 15 tiếng Pháp, 10 tiếng Trung trên tài liệu
   tiếng Anh.

Cả hai vô hình trong log. Cả hai chỉ lộ ra khi mở file output ra đọc. Nếu bỏ qua
bước dry-run vì "hạ tầng đã có test xanh", cả hai đi thẳng lên pod và lỗi thứ hai
thì còn đi tiếp vào index rồi vào mọi con số của `W3-09`.

### Tham số được nhận, không lỗi, và không có tác dụng

`chat_template_kwargs={"enable_thinking": false}` là cách đúng với vLLM/Qwen3.
DeepSeek **nhận nó, trả 200, và vẫn suy luận 159 token**. Nếu tôi chỉ thử đúng
tham số ấy trên DeepSeek rồi kết luận "đã tắt", con số cost/1000 sẽ sai 43% và
tôi sẽ mang giả định ấy lên pod.

Bài học chung: với tham số ngoài chuẩn, **không có phản hồi nào chứng minh nó có
tác dụng** — phải đo cái mà nó đáng lẽ đổi.

### Ranh giới laptop/pod trùng với ranh giới bảo mật

`prepare` (cần corpus, manifest, BGE-M3) chạy ở laptop; `run` chỉ nhận
`requests.jsonl.gz`. Pod không có corpus, không có manifest, không có config,
không có `.env`. Cách chắc nhất để giữ lời hứa "không mang API key lên pod" là
làm cho pod **không có gì khác để chạy**.

### Lệnh để tiếp tục

```bash
make ctx-dry              # thống kê prefill + prompt mẫu, không gọi LLM
make ctx-prepare          # → data/contexts/requests.jsonl.gz (8,5 MB)
make job-bundle           # → dist/runpod-job.tar.gz (7,6 MB)
make job-verify           # PHẢI sạch trước khi đẩy
uv run pytest tests/unit/test_contextual_chunking.py tests/security
```

Việc tiếp theo: **`W4`** (Serving Plane) ở laptop — nó không phụ thuộc `W3-04` —
song song với phiên GPU của anh. Kiểm ngay ở 50 request đầu trên pod:
`reasoning_tokens` phải bằng **0** và `cache_hit_rate` phải trên **30%**; cả hai
rẻ để phát hiện ở phút thứ ba và đắt ở giờ thứ ba (`TD-30`).

## 2026-09-03 (phiên 2) · Backend GLM, và một phép đo tự phản bội

Ngân sách API là ràng buộc thật, nên thêm GLM-5.3-Flash. Phần nối code mất mười
lăm phút — GLM nói đúng giao thức OpenAI kể cả phần `usage`. Phần đáng nhớ là
phép so sánh.

### Con số đầu tiên nói ngược bảng giá, và tôi suýt tin nó

Bảng giá: GLM rẻ hơn DeepSeek 1,8–2,3× ở cả ba trục. Đo được: rẻ **20%**.

Khoảng cách ấy là tín hiệu, không phải nhiễu. Câu trả lời in ngay trong chính
báo cáo tôi vừa đọc: **cache trúng 4,0% (GLM) vs 49,1% (DeepSeek)**.

Chạy lại đúng 40 request đó lần thứ hai — GLM lên **94,8%**, giá xuống
**$0,1818/1000**, tức rẻ **3,3×** so với lượt một của chính nó. Kết luận đúng:
GLM rẻ hơn 2,17×.

Chỗ sai không nằm ở GLM mà ở **mẫu**: 40 request ấy đều lấy từ đầu một tài liệu,
nên toàn bộ mẫu nằm trong vùng khởi động cache. Corpus thật có ~264 chunk mỗi tài
liệu, nên vùng khởi động chiếm ~2% chứ không phải 100%.

Đây là lần thứ **tư** trong `W3` cùng một hình dạng lỗi — `W3-01` §2 (fixture PDF
lặp), `W3-05` §9 (fixture prose lặp một câu), `W3-07` (mỗi "trang" vừa khít một
chunk), và giờ là đây. Khác biệt: ba lần trước sai lệch đến từ **nội dung** mẫu,
lần này đến từ **vị trí** mẫu. Cùng một câu hỏi kiểm được: *mẫu này có phủ chế độ
vận hành thật không, hay chỉ phủ cái nó tình cờ chạm tới?*

### Ba nhà, ba hành vi cho cùng một ý định

Muốn tắt suy luận:

| | vLLM/Qwen3 | DeepSeek | GLM |
|---|---|---|---|
| `chat_template_kwargs` | **có tác dụng** | nhận rồi **bỏ qua** | **từ chối** (400) |
| `thinking={"type":"disabled"}` | — | **có tác dụng** | **từ chối** (400) |
| `reasoning_effort` | — | `"none"` có tác dụng | chỉ `low`/`high`/`max` |

GLM **không tắt được** suy luận — API nói thẳng điều đó trong thông báo lỗi, và
đó là thông báo lỗi hữu ích nhất tôi gặp trong dự án này. Nên hằng số đổi tên
`NO_THINKING` → `MIN_REASONING`. Một cái tên sai ở tầng hằng số sẽ đi vào mọi báo
cáo chi phí về sau mà không ai đọc lại định nghĩa.

### Hệ quả cho kế hoạch

Cả corpus qua GLM còn **~$2,9**. `W3-04` không còn bị GPU chặn; pod chỉ còn cần
cho `W5-11`. Đã viết `RUNPOD.md` cho ai vẫn muốn chạy pod, với §0 nói thẳng rằng
có thể không cần.

### Lệnh để tiếp tục

```bash
make ctx-run-glm CONC=16     # ~$2,9, có checkpoint, đứt thì chạy lại
uv run pytest tests/unit/test_contextualize_backends.py
```

Việc tiếp theo: nối `apply_contexts` vào `build_index` (chưa làm), rồi `W3-09`.
Song song, `W4` vẫn là đường găng.


---

## 2026-09-04 · `W4-07` — hiểu câu hỏi trước khi truy hồi

> ⚠️ **File này có một lỗ hổng 03/09 → 04/09.** Từ `W3-04` tới `W4-06` + `TD-32`,
> `TD-35`…`TD-40` không có phiên nào ghi ở đây. Bản ghi đầy đủ của chúng nằm ở
> bảng changelog cuối [`CHECKLIST.md`](CHECKLIST.md) và ở `reports/tasks/`; tôi
> **không** dựng lại chúng ở đây vì viết lại nhật ký sau khi việc đã xong là viết
> văn, không phải ghi chép.

### Đã xong

`serving/core/understanding.py` (mới) + nối vào `ChatService`, `app.py`, migration
`0003`. Ba việc: định tuyến `NO_RETRIEVAL`/`CLARIFY`/`RETRIEVE` (luật thuần), viết
lại câu hỏi đa lượt (một lượt LLM, **chỉ khi** cần), phát hiện ngôn ngữ.

Ba phát hiện, và cả ba đến từ chỗ khác nhau:

1. **Đo A/B** — chỉ thị ngôn ngữ tường minh đưa tỉ lệ trả lời sai ngôn ngữ từ
   **8/8 xuống 0/8**. Tỉ lệ nền là *tất cả*, không phải *thỉnh thoảng*.
2. **Tiêm lỗi** — một phép tiêm **không đỏ** dẫn tới việc phát hiện `"thầy cô"`,
   `"chị em"` bị xếp là chào hỏi và **không được truy hồi**.
3. **Chạy thật** — model từ chối trả lời `"cái đó thì sao?"` dù truy hồi ra đúng
   chunk, đảo ngược quyết định "model xem câu gốc".

**1954 test xanh, 3 skip.** Chi phí API $0,0125. 6 nợ mới `TD-41`…`TD-46`.

Cũng trong phiên này: bổ sung **7 hàng `W4`** còn thiếu vào bản đồ
[`reports/README.md`](reports/README.md) — chúng có báo cáo đầy đủ từ 03/09 nhưng
không có hàng nào, nên ai vào bản đồ để review đường serving sẽ kết luận là nó
chưa được làm.

### Trạng thái

`W4` **7/13**. Còn `W4-08`…`W4-13`, và `G4`.

### Lệnh để tiếp tục

```bash
docker compose up -d                       # postgres + qdrant + redis
uv run pytest tests/unit/test_query_understanding.py -q
make serve                                 # rồi POST /chat để xem khung meta
```

Việc tiếp theo: **`W4-08`** (LLM Router: DeepSeek → OpenRouter slug ghim →
circuit breaker + budget cap). Nó có sẵn hai chỗ cắm không phải sửa gì:
`ChatService.llm` khai `StreamingLLM` và `QueryUnderstanding.llm` khai
`LLMProvider` — cả hai là Protocol.

⚠️ Quy tắc cứng #1 áp thẳng vào `W4-08`: **không** dùng OpenRouter preset
(`@preset/...`), luôn ghim slug tường minh và log model **thực tế** đã phục vụ.


---

## 2026-09-04 (2) · `W4-08` — bộ định tuyến LLM

### Đã xong

`packages/rag_core/llm/router.py` (mới): `CircuitBreaker`, `DailyBudget`, `Route`,
`LLMRouter` — cài **cả** `LLMProvider.complete` lẫn `StreamingLLM.astream`, nên nó
cắm vào `ChatService.llm` và `QueryUnderstanding.llm` **không sửa dòng nào** ở hai
chỗ đó (cả hai đã khai bằng Protocol từ `W4-06`/`W4-07`).

Ba điều đáng nhớ:

1. **Chuyển nhà cung cấp giữa stream** nối hai câu trả lời khác nhau thành một
   đoạn văn không model nào nói ra. Ranh giới là mẩu đầu tiên; hai test kẹp lấy.
2. **Ba loại hỏng, ba cách đối xử** — cần lớp lỗi mới `PermanentLLMError`.
3. **Tiêm lỗi 12 phép, 2 sống sót**, và một trong hai không chữa được bằng test
   hiển nhiên (máy ở UTC+7 nên test ấy xanh phần lớn thời gian kể cả khi có bug).

**1986 test xanh, 3 skip.** Chi phí API $0,00033. 3 nợ mới `TD-47`…`TD-49`.

### Lệnh để tiếp tục

```bash
docker compose up -d
uv run pytest tests/unit/test_llm_router.py -q
# bật fallback thật: CHAT_FALLBACK_PROVIDER=glm trong môi trường, rồi
make serve && curl -H "Authorization: Bearer <admin key>" localhost:8000/admin/llm
```

Việc tiếp theo: **`W4-09`** (structured output + citation verification — bỏ hẳn
`text.split("Trả lời:")`). Nó nằm ở **nửa dưới** của đường phân giới `W4-06`:
xác minh citation cần toàn bộ câu trả lời, nên một citation bịa chỉ báo được
bằng một khung SSE, không bằng HTTP status.
