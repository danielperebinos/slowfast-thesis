# --- Makefile for slowfast-thesis / dashboard ---
# Usage: make [target]
#
# ⚠️  SAFETY ⚠️
# NEVER run `docker compose down -v` against this stack.
# NEVER delete ../volumes, ../AVA, or any ../experiments/**/best_*.pth file.
# See .ai-factory/RULES.md and the comment at the top of compose.yml.
# There is no `clean` target here that touches anything outside dashboard/.

SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

# --- Package manager (uv, same as the rest of the repo) ---
PKG      ?= uv
PKG_RUN  ?= uv run

# --- Docker Compose ---
COMPOSE_FILE ?= compose.yml
COMPOSE      := docker compose -f $(COMPOSE_FILE)

# ============================================================================
.DEFAULT_GOAL := help

##@ Setup

.PHONY: install
install: ## Install dashboard deps via uv (reads pyproject.toml)
	$(PKG) sync

##@ Dev (local, host GPU)

.PHONY: run
run: ## Run Streamlit directly on the host (uses host CUDA)
	$(PKG_RUN) streamlit run app.py \
		--server.address=0.0.0.0 \
		--server.port=$${DASHBOARD_PORT:-8501} \
		--browser.gatherUsageStats=false

.PHONY: smoke
smoke: ## Run smoke tests (use -m "not slow" for fast subset)
	$(PKG_RUN) pytest tests/ -v $${PYTEST_ARGS:-}

##@ Code Quality

.PHONY: lint
lint: ## Run ruff linter on dashboard sources
	$(PKG_RUN) ruff check .

.PHONY: lint-fix
lint-fix: ## Run ruff linter with autofix
	$(PKG_RUN) ruff check --fix .

.PHONY: fmt
fmt: ## Format code with ruff
	$(PKG_RUN) ruff format .

.PHONY: fmt-check
fmt-check: ## Check formatting without changing files
	$(PKG_RUN) ruff format --check .

.PHONY: check
check: lint fmt-check ## Lint + format-check

##@ Docker (dashboard stack)

.PHONY: build
build: ## Build the dashboard image
	$(COMPOSE) build

.PHONY: up
up: ## Start the dashboard stack (detached)
	$(COMPOSE) up -d --build

.PHONY: down
down: ## Stop the dashboard stack. PRESERVES nothing it should not — there are no named volumes.
	$(COMPOSE) down

.PHONY: stop
stop: ## Pause the stack (keep containers, do not remove)
	$(COMPOSE) stop

.PHONY: restart
restart: ## Restart the stack
	$(COMPOSE) restart

.PHONY: status
status: ## Show stack status
	$(COMPOSE) ps

.PHONY: logs
logs: ## Tail dashboard container logs (Ctrl-C to exit)
	$(COMPOSE) logs -f dashboard

##@ Help

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make \033[36m<target>\033[0m\n"} \
		/^[a-zA-Z0-9_-]+:.*?## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2} \
		/^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5)}' $(MAKEFILE_LIST)

# NOTE: There is intentionally NO `clean`, `clean-volumes`, or `clean-checkpoints`
# target in this Makefile. Volumes at ../volumes/ and the trained checkpoints
# under ../experiments/**/best_*.pth are PROTECTED — see .ai-factory/RULES.md.
