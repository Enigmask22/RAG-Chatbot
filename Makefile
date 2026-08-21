.DEFAULT_GOAL := help
SHELL := /bin/bash

# `uv run` tự đồng bộ môi trường nên không cần activate venv thủ công.
PY := uv run

.PHONY: help
help:  ## Liệt kê các target
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- môi trường
.PHONY: install
install:  ## Cài toàn bộ dependency (kể cả extras nặng)
	uv sync --all-extras

.PHONY: install-dev
install-dev:  ## Cài dependency tối thiểu để lint + unit test
	uv sync --extra dev

# ---------------------------------------------------------------- chất lượng
.PHONY: lint
lint:  ## ruff check + ruff format --check + mypy
	$(PY) ruff check .
	$(PY) ruff format --check .
	$(PY) mypy

.PHONY: fmt
fmt:  ## Tự sửa format + import order
	$(PY) ruff check --fix .
	$(PY) ruff format .

.PHONY: test
test:  ## Unit test (không cần Docker)
	$(PY) pytest -m "not integration and not gpu"

.PHONY: test-integration
test-integration:  ## Integration test (cần `make up` trước)
	$(PY) pytest -m integration

.PHONY: test-gpu
test-gpu:  ## Test cần GPU + trọng số model thật (BGE-M3 ~2,2GB)
	$(PY) pytest -m gpu

.PHONY: test-all
test-all:  ## Toàn bộ test
	$(PY) pytest

.PHONY: cov
cov:  ## Unit test + báo cáo coverage của rag_core
	$(PY) pytest -m "not integration and not gpu" \
		--cov --cov-report=term-missing --cov-report=html:.coverage_html

# ---------------------------------------------------------------- hạ tầng
COMPOSE := docker compose -f infra/docker-compose.yml --env-file .env

# Clone sạch chưa có `.env`; compose sẽ chết vì `--env-file`. Tự tạo từ mẫu để
# `make up` chạy được ngay, secret vẫn phải điền tay sau.
.env:
	cp .env.example .env
	@echo ">> Đã tạo .env từ .env.example. Điền API key trước khi chạy pipeline."

.PHONY: up
up: .env  ## Bật Qdrant + Postgres + Redis, đợi tới khi healthy
	$(COMPOSE) up -d --wait

.PHONY: down
down:  ## Tắt hạ tầng (giữ nguyên volume dữ liệu)
	$(COMPOSE) down

.PHONY: down-clean
down-clean:  ## Tắt hạ tầng và XÓA volume dữ liệu
	$(COMPOSE) down -v

.PHONY: ps
ps:  ## Trạng thái các service
	$(COMPOSE) ps

.PHONY: logs
logs:  ## Xem log hạ tầng
	$(COMPOSE) logs -f --tail=100

# ---------------------------------------------------------------- dữ liệu (DVC)
# Remote khai ở `.dvc/config.local` (không commit). Chưa cấu hình thì các target
# này sẽ báo lỗi của DVC — xem `plans/reports/tasks/w1-09-dvc.md`.
.PHONY: data-pull
data-pull:  ## Lấy corpus từ DVC remote về `data/corpus/`
	$(PY) dvc pull

.PHONY: data-push
data-push:  ## Đẩy corpus hiện tại lên DVC remote
	$(PY) dvc push

.PHONY: data-status
data-status:  ## So working tree với cache DVC và với remote
	$(PY) dvc status
	$(PY) dvc status -c

.PHONY: data-verify
data-verify:  ## Đối chiếu corpus DVC theo dõi với manifest (bắt lệch giữa 2 cơ chế)
	$(PY) python -m pipeline.corpus.dvc_state

.PHONY: data-track
data-track:  ## Ghi lại hash sau khi corpus đổi, rồi kiểm chéo với manifest
	$(PY) dvc add data/corpus
	$(PY) python -m pipeline.corpus.dvc_state

# ---------------------------------------------------------------- pipeline
BUNDLE ?= baseline
BASE ?= baseline
CAND ?= chunk550
INDEX_CONFIG = configs/indexing/$(BUNDLE).yaml
# Nhánh truy hồi (W2-03) — thuộc lần **đo**, không thuộc index, nên không nằm
# trong file config. `RUN` tách khỏi `BUNDLE` để hai nhánh của cùng một index ghi
# ra hai bộ báo cáo thay vì ghi đè lên nhau.
MODE ?= dense
RUN ?= $(BUNDLE)
# Tham số RRF (W2-04) — rỗng thì dùng mặc định (k=60, candidate_k=50). Ví dụ:
#   make eval-retrieval BUNDLE=bgem3 MODE=hybrid RUN=bgem3-rrf-c100 RRF_ARGS="--candidate-k 100"
RRF_ARGS ?=
# Tham số rerank (W2-05) — cùng chỗ với RRF_ARGS vì nhánh `reranked` nhận cả hai:
# nó bọc một nhánh nền, nên tham số của nền vẫn có nghĩa. Ví dụ:
#   make eval-rerank BUNDLE=bgem3 RUN=bgem3-rr-c100 RERANK_ARGS="--rerank-candidates 100"
RERANK_ARGS ?=

.PHONY: corpus
corpus:  ## Tải corpus theo config (idempotent)
	$(PY) python scripts/fetch_corpus.py --config configs/corpus/worldbank_vietnam.yaml

.PHONY: index
index:  ## Build index Qdrant (BUNDLE=baseline). Cần `make up` trước
	$(PY) python -m pipeline.indexing.build_index --config $(INDEX_CONFIG) \
		--report plans/reports/probes/index-$(BUNDLE).json

.PHONY: index-dry
index-dry:  ## Chunk thử vài tài liệu và in thống kê, không chạm Qdrant
	$(PY) python -m pipeline.indexing.build_index --config $(INDEX_CONFIG) --dry-run

.PHONY: eval-compare
eval-compare:  ## So hai lần chạy eval có kiểm định (BASE=baseline CAND=chunk550)
	$(PY) python -m pipeline.eval.compare $(BASE) $(CAND) \
		--out plans/reports/compare/cmp-$(BASE)-vs-$(CAND).md

# Chiều chia nhóm (W2-08-prep). `BY` quét mọi nhóm và TỰ hiệu chỉnh đa so sánh;
# `CAT`/`LANG` so một nhóm đã nêu trước và KHÔNG hiệu chỉnh. Hai loại suy luận
# khác nhau, nên hai target khác nhau — xem docstring `compare_by_group`.
BY ?= category
CAT ?=
LANG ?=

.PHONY: eval-compare-by
eval-compare-by:  ## So theo TỪNG nhóm, có hiệu chỉnh Bonferroni (BY=category|lang)
	$(PY) python -m pipeline.eval.compare $(BASE) $(CAND) --by $(BY) \
		--out plans/reports/compare/by$(BY)-$(BASE)-vs-$(CAND).md

# Bảng ablation N ô (W2-08). `RANK` phải nêu tường minh: thứ hạng đổi theo metric,
# và một bảng không nói nó xếp theo cái gì là một bảng mời đọc sai người thắng.
EXP_PREFIX ?= e1-
RANK ?= ndcg@10
RANK_SLUG ?= ndcg
ABL_BASE ?= e1-baseline-dense

.PHONY: ablation
ablation:  ## Bảng ablation 14 ô + tập tương đương, có p/CI từng dòng (W2-08)
	$(PY) python -m pipeline.eval.ablation --prefix $(EXP_PREFIX) \
		--baseline $(ABL_BASE) --rank-by $(RANK) \
		--out plans/reports/compare/ablation-exp-001-$(RANK_SLUG).md

.PHONY: eval-compare-subset
eval-compare-subset:  ## So trên MỘT nhóm đã nêu trước, không hiệu chỉnh (CAT=… LANG=…)
	$(PY) python -m pipeline.eval.compare $(BASE) $(CAND) \
		$(if $(CAT),--category $(CAT)) $(if $(LANG),--lang $(LANG))

.PHONY: backfill-payload
backfill-payload:  ## Vá payload phẳng + payload index của collection ĐÃ build (W2-06)
	$(PY) python -m pipeline.indexing.backfill_payload --config $(INDEX_CONFIG) \
		--report plans/reports/probes/w2-06-backfill-$(BUNDLE).json

.PHONY: backfill-payload-dry
backfill-payload-dry:  ## Chỉ báo payload index còn thiếu, không ghi gì
	$(PY) python -m pipeline.indexing.backfill_payload --config $(INDEX_CONFIG) --dry-run

.PHONY: truncation
truncation:  ## Đo phần text bị model embedding cắt (TD-11). Không cần Qdrant
	$(PY) python -m pipeline.indexing.truncation_report --config $(INDEX_CONFIG) \
		--report plans/reports/probes/truncation-$(BUNDLE).json

.PHONY: goldenset-dry
goldenset-dry:  ## Xem phân bố lô chunk + prompt mẫu, KHÔNG gọi API
	$(PY) python -m pipeline.goldenset.generate --index-config $(INDEX_CONFIG) --dry-run

.PHONY: goldenset-draft
goldenset-draft:  ## Sinh nháp golden set bằng DeepSeek (TỐN TIỀN API)
	$(PY) python -m pipeline.goldenset.generate --index-config $(INDEX_CONFIG)

.PHONY: goldenset-anchor
goldenset-anchor:  ## Neo nhãn nháp vào văn bản gốc (span thay chunk_id — TD-12)
	$(PY) python -m pipeline.goldenset.anchor --index-config $(INDEX_CONFIG) \
		--report plans/reports/goldenset/goldenset-anchor.json

.PHONY: goldenset-triage
goldenset-triage:  ## Chạy retriever thật lên tập nháp → hàng đợi review (W1-11)
	$(PY) python -m pipeline.goldenset.triage --index-config $(INDEX_CONFIG)

.PHONY: goldenset-freeze
goldenset-freeze:  ## Đóng băng golden_v1 từ decisions_v1.csv (cần review xong)
	$(PY) python -m pipeline.goldenset.freeze \
		--report plans/reports/goldenset/goldenset-v1.json

.PHONY: goldenset-verify
goldenset-verify:  ## Đối chiếu golden_v1.jsonl với checksum đi kèm
	$(PY) python -m pipeline.goldenset.freeze --verify

.PHONY: eval-retrieval
eval-retrieval:  ## Eval retrieval trên index đã build (BUNDLE=baseline MODE=dense RUN=tên)
	$(PY) python -m pipeline.eval.retrieval_eval \
		--index-config $(INDEX_CONFIG) --run-name $(RUN) --retrieval-mode $(MODE) $(RRF_ARGS) $(RERANK_ARGS)

.PHONY: known-item
known-item:  ## Known-item search: tra mã tài liệu, đo dense vs sparse (W2-03)
	$(PY) python scripts/known_item_probe.py --config $(INDEX_CONFIG) \
		--report plans/reports/probes/w2-03-known-item.json

.PHONY: eval-rerank
eval-rerank:  ## Eval reranked trên cấu hình THẮNG của W2-04 (BUNDLE=bgem3 RUN=tên)
	$(PY) python -m pipeline.eval.retrieval_eval \
		--index-config $(INDEX_CONFIG) --run-name $(RUN) --retrieval-mode reranked \
		--rerank-base hybrid --rrf-k 1 --candidate-k 20 $(RERANK_ARGS)

.PHONY: filter-probe
filter-probe:  ## Đo giá của metadata filter theo độ chọn lọc (W2-06)
	$(PY) python scripts/filter_probe.py --config $(INDEX_CONFIG) --mode $(MODE) \
		--report plans/reports/probes/w2-06-filter-probe.json

# ---------------------------------------------------------------- thí nghiệm
EXP ?= exp-001-retrieval
EXP_CONFIG = configs/eval/$(EXP).yaml

.PHONY: exp-dry
exp-dry:  ## In bảng grid + preflight, KHÔNG nạp model và không chạy ô nào (W2-07)
	$(PY) python -m pipeline.experiments.runner --config $(EXP_CONFIG) --dry-run

.PHONY: exp
exp:  ## Chạy hết grid thí nghiệm, resume được (EXP=exp-001-retrieval)
	$(PY) python -m pipeline.experiments.runner --config $(EXP_CONFIG) --keep-going

.PHONY: exp-rerun
exp-rerun:  ## Chạy lại CẢ những ô state ghi là đã xong
	$(PY) python -m pipeline.experiments.runner --config $(EXP_CONFIG) --no-resume --keep-going

.PHONY: exp-backfill
exp-backfill:  ## Dựng lại view MLflow từ báo cáo đã có, KHÔNG chạy lại eval (W2-07)
	$(PY) python -m pipeline.experiments.backfill --config $(EXP_CONFIG)

.PHONY: mlflow-ui
mlflow-ui:  ## Mở MLflow UI trên store cục bộ (http://127.0.0.1:5000)
	$(PY) mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

.PHONY: rerank-probe
rerank-probe:  ## Trần vùng phủ + truncation + bão hoà sigmoid + độ trễ rerank (W2-05)
	$(PY) python scripts/rerank_probe.py --config $(INDEX_CONFIG) \
		--report plans/reports/probes/w2-05-rerank-probe.json
