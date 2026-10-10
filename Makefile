# DataScienceBAModeler -- agent-run Titanic pipeline.
# Pipeline order is enforced here as much as in CLAUDE.md:
#   setup -> pull -> review -> split -> freeze -> explore -> tune -> ensemble
# Run `make help` for the target list.

PY ?= python3.13
VENV := .venv
BIN := $(VENV)/bin
PYTHON := $(BIN)/python
SEED ?= 20251010
CV_FOLDS ?= 5
N_ITER ?= 25
MODEL ?= all
METHOD ?= soft_vote
TOP_K ?= 4

.DEFAULT_GOAL := help
.PHONY: help setup pull review split freeze verify explore tune ensemble test lint log all clean distclean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: $(VENV)/.installed ## Create .venv and install the package (editable) + dev deps

$(VENV)/.installed: pyproject.toml
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install --quiet --upgrade pip
	$(BIN)/python -m pip install --quiet -e '.[dev]'
	@touch $@
	@echo "env ready: $(VENV) ($$($(PYTHON) --version))"

pull: setup ## Step 1: download the raw data into data/raw (gitignored)
	$(PYTHON) -m titanic_agent.pull

review: setup ## Step 2: audit raw data -> reports/01_review.md
	$(PYTHON) -m titanic_agent.review

split: setup ## Step 3: draw the train/holdout partition -> data/frozen/split_plan.json
	$(PYTHON) -m titanic_agent.split --seed $(SEED)

freeze: setup ## Step 4: seal the dataset + checksums (POINT OF NO RETURN)
	$(PYTHON) -m titanic_agent.freeze

verify: setup ## Re-check the frozen dataset against its manifest
	$(PYTHON) -m titanic_agent.freeze --verify-only

explore: setup ## Step 5: EDA on the frozen train split -> reports/02_explore.md + charts
	$(PYTHON) -m titanic_agent.explore

tune: setup ## Step 6: CV model search; appends to experiments/results.csv
	$(PYTHON) -m titanic_agent.tune --model $(MODEL) --n-iter $(N_ITER) --cv-folds $(CV_FOLDS) --seed $(SEED)

ensemble: setup ## Step 7: combine the best models -> reports/submission.csv
	$(PYTHON) -m titanic_agent.ensemble --method $(METHOD) --top-k $(TOP_K) --cv-folds $(CV_FOLDS)

test: setup ## Run the smoke tests
	$(BIN)/pytest

lint: setup ## Lint and format-check with ruff
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

log: setup ## Print the experiment ledger
	$(PYTHON) -m titanic_agent.results_log

all: review split freeze explore tune ensemble ## Everything after pull, in order

clean: ## Remove caches and generated reports (never touches data/)
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find reports -type f ! -name .gitkeep -delete 2>/dev/null || true

distclean: clean ## Also remove the virtualenv. Leaves data/frozen alone, by design.
	rm -rf $(VENV)
