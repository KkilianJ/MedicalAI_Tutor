# Medical AI Tutor — developer commands.
#
# Everything except `fetch-data` works offline with no API keys: the mock
# provider and the lexical retriever have no network dependency.

PYTHON  ?= python3
VENV    ?= .venv
BIN     := $(VENV)/bin
PDF     ?= data/raw/winter2023_his.pdf
PORT    ?= 8000
APPPORT ?= 8501

.DEFAULT_GOAL := help
.PHONY: help install fetch-data ingest test test-live eval eval-safety api app lint format clean check

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)

install: $(BIN)/python ## Create the venv and install the project
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[ingest,anthropic,openai,dev]"
	@echo "installed. to use a real model, put your key in medical_ai_tutor/local_settings.py"

fetch-data: ## Download the open-access textbook (CC BY 4.0)
	$(BIN)/python scripts/fetch_textbook.py --out $(PDF)

ingest: ## Parse the PDF into searchable / exercise / protected artefacts
	$(BIN)/python scripts/ingest_textbook.py --pdf $(PDF)

test: ## Run the test suite (offline, no API keys needed)
	$(BIN)/python -m pytest tests/ -q

test-live: ## Run tests against a real LLM (needs a key in local_settings.py; costs tokens)
	$(BIN)/python -m pytest -m live -s -v

eval: ## Run the offline evaluation suite
	$(BIN)/python scripts/run_evals.py --json artifacts/eval_summary.json

eval-safety: ## Run only the adversarial safety cases, failing on any leak
	$(BIN)/python scripts/run_evals.py --dimension safety --fail-on-leak

api: ## Serve the FastAPI backend
	$(BIN)/python -m uvicorn medical_ai_tutor.app.api:app --reload --port $(PORT)

app: ## Serve the Streamlit demonstration UI
	$(BIN)/streamlit run medical_ai_tutor/app/streamlit_app.py --server.port $(APPPORT)

lint: ## Lint with ruff (if installed)
	@$(BIN)/ruff check medical_ai_tutor tests scripts 2>/dev/null || echo "ruff not installed; skipping"

format: ## Format with ruff
	@$(BIN)/ruff format medical_ai_tutor tests scripts 2>/dev/null || echo "ruff not installed; skipping"

check: test eval ## Run tests and evaluations together

clean: ## Remove caches, the session database and generated artefacts
	rm -rf .pytest_cache artifacts .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -f data/tutor.sqlite3 data/tutor.sqlite3-wal data/tutor.sqlite3-shm
	@echo "cleaned (ingested corpora and the raw PDF were kept)"
