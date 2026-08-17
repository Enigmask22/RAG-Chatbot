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

# ---------------------------------------------------------------- pipeline
BUNDLE ?= baseline
INDEX_CONFIG = configs/indexing/$(BUNDLE).yaml

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

.PHONY: eval-retrieval
eval-retrieval:  ## Eval retrieval trên index đã build (BUNDLE=baseline)
	$(PY) python -m pipeline.eval.retrieval_eval \
		--index-config $(INDEX_CONFIG) --run-name $(BUNDLE)
