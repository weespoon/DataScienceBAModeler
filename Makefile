# DataScienceBAModeler -- agent-run Titanic pipeline.
# Pipeline order is enforced here as much as in CLAUDE.md:
#   setup -> pull -> review -> split -> freeze -> explore -> tune -> ensemble
# Run `make help` for the target list.

# Setup prefers uv when it is on PATH (much faster, and it picks a compatible
# interpreter itself); it falls back to the stdlib venv + pip otherwise. Either
# way the result is the same: .venv with this package installed editable.
PY ?= python3.13
VENV := .venv
BIN := $(VENV)/bin
PYTHON := $(BIN)/python
EXTRAS ?= dev
SEED ?= 20251010
CV_FOLDS ?= 5
N_ITER ?= 25
MODEL ?= all
METHOD ?= soft_vote
TOP_K ?= 4

.DEFAULT_GOAL := help
.PHONY: help setup pull review split freeze verify explore tune features ensemble \
	test lint log status next journal receipts audit introspect leakage experiments experiment-open experiment-evaluate experiment-close findings backlog runs summary bootstrap loop clean distclean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: $(VENV)/.installed ## Create .venv and install the package (editable) + dev deps

# Re-runs whenever pyproject.toml changes, or when the sentinel is missing --
# which is also how it recovers if .venv was replaced by something else.
$(VENV)/.installed: pyproject.toml
	@if command -v uv >/dev/null 2>&1; then \
		echo "uv found: uv venv + uv pip install -e '.[$(EXTRAS)]'"; \
		uv venv --python '>=3.11' --allow-existing $(VENV); \
		uv pip install --quiet --python $(PYTHON) -e '.[$(EXTRAS)]'; \
	else \
		echo "uv not found: stdlib venv + pip install -e '.[$(EXTRAS)]'"; \
		$(PY) -m venv $(VENV); \
		$(PYTHON) -m pip install --quiet --upgrade pip; \
		$(PYTHON) -m pip install --quiet -e '.[$(EXTRAS)]'; \
	fi
	@touch $@
	@echo "env ready: $(VENV) ($$($(PYTHON) --version))"

pull: setup ## Step 1: download the raw data into data/raw (gitignored)
	@$(PYTHON) -c 'import kaggle' 2>/dev/null \
		|| { echo "step 1 needs the optional kaggle extra: make setup EXTRAS=dev,kaggle"; exit 1; }
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

MAX_TESTS ?= 50
features: setup ## Greedy add/prune feature selection vs one fixed model (needs an open experiment)
	$(PYTHON) -m titanic_agent.features --max-tests $(MAX_TESTS)

ensemble: setup ## Step 7: combine the best models -> reports/submission.csv
	$(PYTHON) -m titanic_agent.ensemble --method $(METHOD) --top-k $(TOP_K) --cv-folds $(CV_FOLDS)

test: setup ## Run the smoke tests
	$(BIN)/pytest

lint: setup ## Lint and format-check with ruff
	$(BIN)/ruff check src tests .claude/hooks
	$(BIN)/ruff format --check src tests .claude/hooks

log: setup ## Print the experiment ledger
	$(PYTHON) -m titanic_agent.results_log

# `freeze` raises on an already-frozen dir, so it must never sit in a loop that
# reruns. Hence two aggregates instead of one `all`: bootstrap is do-once, loop
# is the repeatable part and is what an autonomous pass actually iterates.
bootstrap: review split freeze ## Steps 2-4, run ONCE: audit, split, seal the data
	@echo "bootstrap complete -- data is frozen. From here on, use: make loop"

# One id per pass, shared by all three steps and used as the run directory name.
loop: export TITANIC_PASS_ID := $(shell date +%Y%m%d-%H%M%Z)
loop: explore tune ensemble ## Steps 5-7, repeatable: the autonomous iterate loop
	@$(PYTHON) -m titanic_agent.runs --summarize
	@echo "pass $(TITANIC_PASS_ID) complete -- make audit to check the rules held"

runs: setup ## List every pass and how many artifacts it left
	@$(PYTHON) -m titanic_agent.runs

summary: setup ## Rewrite this pass's runs/<id>/run.md from the ledger and journal
	@$(PYTHON) -m titanic_agent.runs --summarize

status: setup ## Which pipeline steps have completed, derived from artifacts on disk
	@$(PYTHON) -m titanic_agent.status

next: setup ## Just the next incomplete step -- what the `run` skill loops on
	@$(PYTHON) -m titanic_agent.status --next

# Make consumes a leading --flag itself, so options are Make VARIABLES.
# LEVEL=decision shows only what needed a person; PASS=<id> scopes to one pass.
JOURNAL_ARGS ?= --tail 20
journal: setup ## Agent journal. Options: LEVEL=decision PASS=<run id> ALL=1
	@$(PYTHON) -m titanic_agent.run_log $(JOURNAL_ARGS) \
		$(if $(LEVEL),--level $(LEVEL)) $(if $(PASS),--pass-id $(PASS)) $(if $(ALL),--all)

receipts: setup ## Show the holdout read receipts
	@$(PYTHON) -m titanic_agent.receipts

audit: setup ## Did the three rules hold? Reads manifest, ledger chain and receipts
	@$(PYTHON) -m titanic_agent.audit

introspect: setup ## Search spaces vs the INSTALLED sklearn. Options: UNTUNED=1
	@$(PYTHON) -m titanic_agent.introspect $(if $(UNTUNED),--show-untuned)

leakage: setup ## Rule 4: structural guards against looking up the answers
	@$(PYTHON) -m titanic_agent.leakage

experiments: setup ## List every experiment and its scores
	@$(PYTHON) -m titanic_agent.experiments

experiment-open: setup ## Start an experiment: make experiment-open NAME="wide search"
	@$(PYTHON) -m titanic_agent.experiments open "$(NAME)"

experiment-evaluate: setup ## Score the open experiment's val split, ONCE. Options: RUN=<run id>
	@$(PYTHON) -m titanic_agent.experiments evaluate $(if $(RUN),--run-id $(RUN))

experiment-close: setup ## Close the open experiment: make experiment-close SUMMARY="..."
	@$(PYTHON) -m titanic_agent.experiments close --summary "$(SUMMARY)"

findings: setup ## What we have learned, and what is still untried
	@$(PYTHON) -m titanic_agent.findings

backlog: setup ## Only the untried ideas -- what the planner picks from
	@$(PYTHON) -m titanic_agent.findings --backlog

clean: ## Remove caches and generated reports (never touches data/)
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find reports -type f ! -name .gitkeep -delete 2>/dev/null || true

distclean: clean ## Also remove the virtualenv. Leaves data/frozen alone, by design.
	rm -rf $(VENV)
