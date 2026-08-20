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
# này sẽ báo lỗi của DVC — xem `plans/reports/w1-09-dvc.md`.
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

.PHONY: corpus
corpus:  ## Tải corpus theo config (idempotent)
	$(PY) python scripts/fetch_corpus.py --config configs/corpus/worldbank_vietnam.yaml

.PHONY: index
index:  ## Build index Qdrant (BUNDLE=baseline). Cần `make up` trước
	$(PY) python -m pipeline.indexing.build_index --config $(INDEX_CONFIG) \
		--report plans/reports/index-$(BUNDLE).json

.PHONY: index-dry
index-dry:  ## Chunk thử vài tài liệu và in thống kê, không chạm Qdrant
	$(PY) python -m pipeline.indexing.build_index --config $(INDEX_CONFIG) --dry-run

.PHONY: eval-compare
eval-compare:  ## So hai lần chạy eval có kiểm định (BASE=baseline CAND=chunk550)
	$(PY) python -m pipeline.eval.compare $(BASE) $(CAND) \
		--out plans/reports/cmp-$(BASE)-vs-$(CAND).md

.PHONY: truncation
truncation:  ## Đo phần text bị model embedding cắt (TD-11). Không cần Qdrant
	$(PY) python -m pipeline.indexing.truncation_report --config $(INDEX_CONFIG) \
		--report plans/reports/truncation-$(BUNDLE).json

.PHONY: goldenset-dry
goldenset-dry:  ## Xem phân bố lô chunk + prompt mẫu, KHÔNG gọi API
	$(PY) python -m pipeline.goldenset.generate --index-config $(INDEX_CONFIG) --dry-run

.PHONY: goldenset-draft
goldenset-draft:  ## Sinh nháp golden set bằng DeepSeek (TỐN TIỀN API)
	$(PY) python -m pipeline.goldenset.generate --index-config $(INDEX_CONFIG)

.PHONY: goldenset-anchor
goldenset-anchor:  ## Neo nhãn nháp vào văn bản gốc (span thay chunk_id — TD-12)
	$(PY) python -m pipeline.goldenset.anchor --index-config $(INDEX_CONFIG) \
		--report plans/reports/goldenset-anchor.json

.PHONY: goldenset-triage
goldenset-triage:  ## Chạy retriever thật lên tập nháp → hàng đợi review (W1-11)
	$(PY) python -m pipeline.goldenset.triage --index-config $(INDEX_CONFIG)

.PHONY: goldenset-freeze
goldenset-freeze:  ## Đóng băng golden_v1 từ decisions_v1.csv (cần review xong)
	$(PY) python -m pipeline.goldenset.freeze \
		--report plans/reports/goldenset-v1.json

.PHONY: goldenset-verify
goldenset-verify:  ## Đối chiếu golden_v1.jsonl với checksum đi kèm
	$(PY) python -m pipeline.goldenset.freeze --verify

.PHONY: eval-retrieval
eval-retrieval:  ## Eval retrieval trên index đã build (BUNDLE=baseline MODE=dense RUN=tên)
	$(PY) python -m pipeline.eval.retrieval_eval \
		--index-config $(INDEX_CONFIG) --run-name $(RUN) --retrieval-mode $(MODE) $(RRF_ARGS)

.PHONY: known-item
known-item:  ## Known-item search: tra mã tài liệu, đo dense vs sparse (W2-03)
	$(PY) python scripts/known_item_probe.py --config $(INDEX_CONFIG) \
		--report plans/reports/w2-03-known-item.json
