# Dữ liệu

## Giấy phép — corpus KHÔNG thuộc phạm vi MIT

`LICENSE` ở gốc repo áp cho **mã nguồn**. Tài liệu trong `corpus/` là một thứ
khác: chúng do **World Bank** phát hành theo
[**CC BY 3.0 IGO**](https://creativecommons.org/licenses/by/3.0/igo/), và repo
này chỉ phân phối lại chứ không sở hữu.

Gộp hai thứ dưới một dòng "MIT" là cấp cho người khác một quyền mình không có.

`corpus_manifest.csv` ghi cho **từng** tài liệu: `source_url`, `landing_url`,
`license`, `license_url`, `sha256` của byte, và từ `TD-22` là cả `text_sha256`
với `parse_fingerprint`.

Ràng buộc được thực thi bằng code, không phải bằng lời hứa —
`pipeline/corpus/manifest.py` từ chối mọi entry thiếu `source_url` hoặc mang giấy
phép ngoài `LICENSE_ALLOWLIST`. Danh sách ấy **loại cả giấy phép có `ND`**
(NoDerivatives): pipeline này cắt tài liệu thành chunk rồi sinh context bằng LLM,
và đó là tạo tác phẩm phái sinh theo đúng nghĩa của điều khoản.

`contexts/contexts.jsonl.gz` **là** một tác phẩm phái sinh như thế, và nó có
trong repo. Nó thừa hưởng cùng nghĩa vụ ghi công: mỗi dòng mang `doc_id`, và
`corpus_manifest.csv` ánh xạ `doc_id` → `source_url` + `license` của tài liệu
gốc. Văn bản do LLM sinh ra thì repo không đòi bản quyền.

## Thư mục

| | |
|---|---|
| `corpus/` | 60 tài liệu (40 EN + 20 VI, 14,3 triệu ký tự). Không commit — lấy bằng `make data-pull` (DVC) hoặc `make corpus` (tải lại từ nguồn) |
| `corpus_manifest.csv` | Sổ đăng ký: nguồn, giấy phép, checksum. **Có** commit |
| `golden/golden_v1.jsonl` | 242 câu hỏi, nhãn neo theo **span** ký tự. Có checksum đi kèm; `make goldenset-verify` để đối chiếu |
| `golden/draft_v1.jsonl` | Bản nháp trước review, giữ lại để truy vết |
| `golden/review/` | Quyết định review từng câu |
| `contexts/contexts.jsonl.gz` | **Tác phẩm phái sinh**: 15.814 câu định vị do GLM-5.3-Flash sinh cho từng chunk (`W3-04`). **Có** commit — 1,8 MB nén, và sinh lại tốn ~$5,90 tiền API. Kiểm bằng `make ctx-verify` |
| `contexts/` (còn lại) | Gói request, dry-run, lô bị từ chối. **Không** commit — sinh lại miễn phí bằng `make ctx-prepare` |
