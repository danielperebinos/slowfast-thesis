# --- Makefile for slowfast-thesis ---
# Usage: make [target]
#
# This is a research / thesis repository. The common targets below wrap:
#   - uv (package manager)                - ruff (lint + format)
#   - docker compose (MLflow stack)       - the four experiment train.py scripts
#   - the AVA data-prep scripts           - verify_setup.py smoke test
#
# ⚠️  VOLUME SAFETY ⚠️
# Docker volumes under ./volumes/ hold the MLflow tracking DB (postgres) and
# the trained-model artifact store (minio). There is NO `stack-wipe`, no
# `down -v`, and no `clean-volumes` target in this Makefile on purpose.
# See docs/mlflow-stack.md#volume-safety.

SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

# --- Project ---
PROJECT       ?= slowfast-thesis
PYTHON        ?= python3
SRC_DIR       ?= experiments

# --- Git ---
VERSION    ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
COMMIT     ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_TIME := $(shell date -u '+%Y-%m-%dT%H:%M:%SZ')

# --- Package Manager (uv for this repo) ---
PKG     ?= uv
PKG_RUN ?= uv run

# --- Docker Compose ---
COMPOSE_FILE ?= compose.experiments.yml
COMPOSE      := docker compose -f $(COMPOSE_FILE)

# --- Data paths (override via env or `make VAR=value <target>`) ---
AVA_VIDEO_DIR ?= $(CURDIR)/AVA
AVA_CSV       ?= $(CURDIR)/experiments/dataset/data/ava/ava_train_v2.2.csv
OUT_DIR       ?= $(CURDIR)/experiments/experiment_01

# ============================================================================
.DEFAULT_GOAL := help

##@ Setup

.PHONY: install
install: ## Install dependencies via uv (resolves pyproject.toml + uv.lock)
	$(PKG) sync

.PHONY: lock
lock: ## Refresh uv.lock without installing
	$(PKG) lock

.PHONY: env
env: ## Print the resolved Python + project environment info
	@echo "Project      : $(PROJECT)"
	@echo "Version      : $(VERSION)"
	@echo "Commit       : $(COMMIT)"
	@echo "Python       : $$($(PKG_RUN) python --version)"
	@echo "Torch+CUDA   : $$($(PKG_RUN) python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())')"
	@echo "MLflow URI   : $${MLFLOW_TRACKING_URI:-http://localhost:5000}"

##@ Code Quality

.PHONY: lint
lint: ## Run ruff linter
	$(PKG_RUN) ruff check $(SRC_DIR)

.PHONY: lint-fix
lint-fix: ## Run ruff linter with autofix
	$(PKG_RUN) ruff check --fix $(SRC_DIR)

.PHONY: fmt
fmt: ## Format code with ruff
	$(PKG_RUN) ruff format $(SRC_DIR)

.PHONY: fmt-check
fmt-check: ## Check code formatting without changing files
	$(PKG_RUN) ruff format --check $(SRC_DIR)

.PHONY: check
check: lint fmt-check ## Run lint + format-check (fast feedback)

##@ MLflow Stack (Docker Compose)

.PHONY: stack-up
stack-up: ## Start postgres + minio + mlflow + bucket init (detached)
	$(COMPOSE) up -d

.PHONY: stack-down
stack-down: ## Stop the stack. PRESERVES volumes (no -v flag, ever).
	$(COMPOSE) down

.PHONY: stack-stop
stack-stop: ## Pause the stack (keep containers, do not remove them)
	$(COMPOSE) stop

.PHONY: stack-restart
stack-restart: ## Restart the stack
	$(COMPOSE) restart

.PHONY: stack-status
stack-status: ## Show stack container status
	$(COMPOSE) ps

.PHONY: stack-logs
stack-logs: ## Tail mlflow logs (Ctrl-C to exit). Override: make stack-logs SVC=postgres
	$(COMPOSE) logs -f $${SVC:-mlflow}

.PHONY: stack-build-mlflow
stack-build-mlflow: ## Rebuild only the mlflow image (after Dockerfile.mlflow changes)
	$(COMPOSE) build mlflow
	$(COMPOSE) up -d mlflow

##@ Data

.PHONY: data-download
data-download: ## Download AVA clips listed in $(AVA_CSV) to $(AVA_VIDEO_DIR). Requires yt-dlp + ffmpeg.
	AVA_VIDEO_DIR=$(AVA_VIDEO_DIR) $(PKG_RUN) python experiments/dataset/download_all.py \
		--csv $(AVA_CSV) \
		--workers $${WORKERS:-4}

.PHONY: data-splits
data-splits: ## Build train.csv / test.csv for $(OUT_DIR) (default: experiment_01)
	AVA_VIDEO_DIR=$(AVA_VIDEO_DIR) AVA_CSV=$(AVA_CSV) OUT_DIR=$(OUT_DIR) \
		$(PKG_RUN) python experiments/dataset/build_splits.py

.PHONY: data-splits-all
data-splits-all: ## Build splits for experiment_01 and copy to experiment_02/03/04 (strict fairness)
	$(MAKE) data-splits OUT_DIR=$(CURDIR)/experiments/experiment_01
	cp $(CURDIR)/experiments/experiment_01/train.csv $(CURDIR)/experiments/experiment_02/train.csv
	cp $(CURDIR)/experiments/experiment_01/test.csv  $(CURDIR)/experiments/experiment_02/test.csv
	cp $(CURDIR)/experiments/experiment_01/train.csv $(CURDIR)/experiments/experiment_03/train.csv
	cp $(CURDIR)/experiments/experiment_01/test.csv  $(CURDIR)/experiments/experiment_03/test.csv
	cp $(CURDIR)/experiments/experiment_01/train.csv $(CURDIR)/experiments/experiment_04/train.csv
	cp $(CURDIR)/experiments/experiment_01/test.csv  $(CURDIR)/experiments/experiment_04/test.csv

##@ Experiments

.PHONY: experiment-01
experiment-01: ## Run experiment 01 — Baseline SlowFast R50
	$(PKG_RUN) python experiments/experiment_01/train.py

.PHONY: experiment-02
experiment-02: ## Run experiment 02 — Attention (Non-Local / Self-Attention)
	$(PKG_RUN) python experiments/experiment_02/train.py

.PHONY: experiment-03
experiment-03: ## Run experiment 03 — YOLO ROI Guidance
	$(PKG_RUN) python experiments/experiment_03/train.py

.PHONY: experiment-04
experiment-04: ## Run experiment 04 — Hybrid (ROI + local attention)
	$(PKG_RUN) python experiments/experiment_04/train.py

.PHONY: experiments-all
experiments-all: experiment-01 experiment-02 experiment-03 experiment-04 ## Run all four experiments sequentially

.PHONY: verify
verify: ## Dummy-data smoke test across all four variants (forward + backward)
	cd experiments && $(PKG_RUN) python verify_setup.py

##@ CI

.PHONY: ci
ci: install check verify ## Full CI: install, lint, format-check, verify (no training)

##@ Cleanup

.PHONY: clean-pyc
clean-pyc: ## Remove Python caches (pycache, pytest, ruff, mypy)
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -not -path "./.venv/*" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache

# NOTE: there is no `clean-volumes` or `clean-checkpoints` target.
# Volumes (./volumes/) and experiment checkpoints (experiments/experiment_0N/*.pth)
# are PROTECTED — see docs/mlflow-stack.md#volume-safety.

##@ Help

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make \033[36m<target>\033[0m\n"} \
		/^[a-zA-Z0-9_-]+:.*?## / {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2} \
		/^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5)}' $(MAKEFILE_LIST)
