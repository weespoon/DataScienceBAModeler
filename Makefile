# DataScienceBAModeler -- agent-run Titanic pipeline.
# Pipeline order is enforced here as much as in CLAUDE.md:
#   setup -> pull -> review -> split -> freeze -> explore -> tune -> ensemble
# Run `make help` for the target list.

# Setup prefers uv when it is on PATH (much faster, and it picks a compatible
# interpreter itself); it falls back to the stdlib venv + pip otherwise. Either
# way the result is the same: .venv with this package installed editable.
PY ?= python3.13
# The interpreter and its paths are not overridable from the command line.
# Without `override`, `make <any-target> PYTHON=...` is arbitrary execution.
override VENV := .venv
override BIN := $(VENV)/bin
override PYTHON := $(BIN)/python

# Shell metacharacters. A variable containing any of these becomes EMPTY rather
# than reaching /bin/sh, so a bad value fails the command instead of running it.
# `safe` and `numeric` take the variable NAME, not its value, so a comma in the
# value (ONLY=a,b, EVIDENCE=id1,id2) never hits make's own $(call) parser.
_meta := ; | & ` > < ( ) { } ' " ! * ? ~ $$$$ \# \\
safe = $(if $(strip $(foreach c,$(_meta),$(findstring $(c),$($(1))))),,$($(1)))

# Pure-make numeric guard: strip digits, dots and signs; anything left means the
# value was not a number.
_strip_digits = $(subst 0,,$(subst 1,,$(subst 2,,$(subst 3,,$(subst 4,,$(subst 5,,$(subst 6,,$(subst 7,,$(subst 8,,$(subst 9,,$(subst .,,$(subst -,,$($(1))))))))))))))
numeric = $(if $(call _strip_digits,$(1)),,$($(1)))

# Enum whitelists. A value outside the set becomes empty rather than reaching
# the shell; the Python side then refuses a missing argument.
MODEL_KEYS := dummy logistic random_forest extra_trees gradient_boosting hist_gradient_boosting svc knn
ARM_SET_KEYS := bakeoff decompose simplify
METHOD_KEYS := soft_vote hard_vote stack
LEVEL_KEYS := debug info decision error
EVENT_KEYS := step_started step_finished step_skipped step_failed decision refused note
STATUS_KEYS := confirmed refuted inconclusive untried
CONFIDENCE_KEYS := low medium high

EXTRAS ?= dev
SEED ?= 20251010
CV_FOLDS ?= 5
N_ITER ?= 25
MODEL ?= all
METHOD ?= soft_vote
TOP_K ?= 4
OUTER_FOLDS ?= 5
# `make agent` caps. A turn is one model response; the dollar ceiling is a
# second backstop with a different failure mode -- thrashing cheaply hits
# TURNS, reasoning expensively hits BUDGET.
TURNS ?= 60
BUDGET ?= 5

.DEFAULT_GOAL := help
.PHONY: help setup pull review split freeze verify explore tune features ensemble \
	nested bakeoff bakeoff-decide permutation champion submit test lint log status next journal receipts audit introspect leakage experiments experiment-open experiment-lock-plan experiment-evaluate experiment-close finding findings backlog runs summary report note agent bootstrap loop clean distclean

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
	$(PYTHON) -m titanic_agent.split --seed $(call numeric,SEED)

freeze: setup ## Step 4: seal the dataset + checksums (POINT OF NO RETURN)
	$(PYTHON) -m titanic_agent.freeze

verify: setup ## Re-check the frozen dataset against its manifest
	$(PYTHON) -m titanic_agent.freeze --verify-only

explore: setup ## Step 5: EDA on the frozen train split -> reports/02_explore.md + charts
	$(PYTHON) -m titanic_agent.explore

tune: setup ## Step 6: CV model search; appends to experiments/results.csv
	$(PYTHON) -m titanic_agent.tune --model $(filter all $(MODEL_KEYS),$(MODEL)) --n-iter $(call numeric,N_ITER) --cv-folds $(call numeric,CV_FOLDS) --seed $(call numeric,SEED)

MAX_TESTS ?= 50
features: setup ## Greedy add/prune feature selection vs one fixed model (needs an open experiment)
	$(PYTHON) -m titanic_agent.features --max-tests $(call numeric,MAX_TESTS)

# Over the ~2-minute laptop rule on purpose, and carved out by name in
# CLAUDE.md: this is a deliberate measurement, not a search to be shrunk.
# Deliberately absent from `make loop` -- once per searching experiment.
nested: setup ## Measure a feature search's selection bias. Options: EXPERIMENT=<id> OUTER_FOLDS=5
	$(PYTHON) -m titanic_agent.nested $(if $(EXPERIMENT),--experiment $(call safe,EXPERIMENT),) --outer-folds $(call numeric,OUTER_FOLDS)

# The search half of a bake-off: every arm paired against the champion, plus
# the winner re-scored on fresh fold seeds. Inside the ~2-minute rule.
ONLY ?=
ARM_SET ?= bakeoff
bakeoff: setup ## Many arms vs the persisted champion (needs an open experiment). Options: ARM_SET=decompose ONLY=a,b
	$(PYTHON) -m titanic_agent.bakeoff run --set $(filter $(ARM_SET_KEYS),$(ARM_SET)) $(if $(ONLY),--only $(call safe,ONLY),)

# The measurement half: nested CV of pick-the-best, then the pre-declared rule.
# Over the 2-minute rule for the same reason `nested` is; once per bake-off.
bakeoff-decide: setup ## Nested-CV the bake-off, apply its declared rule, crown or hold
	$(PYTHON) -m titanic_agent.bakeoff decide $(if $(ONLY),--only $(call safe,ONLY),)

# The in-fold group rates read the label, so this is the one check between the
# programme and the classic Titanic leak. `make bakeoff` runs it itself before
# scoring any label-reading arm; this target is for re-running it on its own
# after a change to Engineer. Exits non-zero if anything beats chance.
permutation: setup ## Leak control: in-fold group features on shuffled labels
	$(PYTHON) -m titanic_agent.bakeoff control $(if $(ARM_SET),--set $(filter $(ARM_SET_KEYS),$(ARM_SET)),)

champion: setup ## The persisted champion every experiment is paired against
	@$(PYTHON) -m titanic_agent.champion

# Not step 7: no holdout, no ensemble search. The reigning champion, fitted on
# the 712 working rows, predicting Kaggle's unlabelled test set.
submit: setup ## Kaggle submission from the reigning champion (never reads the holdout)
	$(PYTHON) -m titanic_agent.champion submit

ensemble: setup ## Step 7: combine the best models -> reports/submission.csv
	$(PYTHON) -m titanic_agent.ensemble --method $(filter $(METHOD_KEYS),$(METHOD)) --top-k $(call numeric,TOP_K) --cv-folds $(call numeric,CV_FOLDS)

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

# One headless pass, end to end: `claude -p` pointed at the run skill, capped,
# with its transcript saved beside the rows the pass produced. Deliberately not
# --permission-mode bypassPermissions: the deny rules in .claude/settings.json
# are what stop a pass reaching chmod on data/frozen or --score-holdout.
agent: setup ## One autonomous pass via the run skill. Options: TURNS=60 BUDGET=5
	@$(PYTHON) -m titanic_agent.agent --turns $(call numeric,TURNS) --budget $(call numeric,BUDGET)

# One page per pass, for a human who was not watching: the experiment and its
# verdict, the champion, whether the rules held, the receipts, and what the pass
# decided or refused. Derived from the modules that own those facts, so it
# cannot disagree with them. Charges no budget -- `make summary` is the counter.
report: setup ## This pass in one page. Options: PASS=<run id>
	@$(PYTHON) -m titanic_agent.report $(if $(PASS),--pass-id $(call safe,PASS),)

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
# Writing to the journal, so an unattended pass can record why it stopped
# without needing permission to run arbitrary Python. A pass that stops without
# saying why is indistinguishable from one that crashed.
# The text goes through the environment, not the shell. `DETAIL_Q = '$(DETAIL)'`
# broke on the first apostrophe, and a journal writer that cannot take prose is
# one an agent will work around.
note: export TITANIC_DETAIL := $(DETAIL)
note: setup ## Append a journal entry: make note DETAIL="..." EVENT=decision STEP=tune
	@$(PYTHON) -m titanic_agent.run_log --record-env TITANIC_DETAIL \
		$(if $(EVENT),--as-event $(filter $(EVENT_KEYS),$(EVENT))) $(if $(STEP),--step $(call safe,STEP))

journal: setup ## Agent journal. Options: LEVEL=decision PASS=<run id> ALL=1
	@$(PYTHON) -m titanic_agent.run_log $(JOURNAL_ARGS) \
		$(if $(LEVEL),--level $(filter $(LEVEL_KEYS),$(LEVEL))) $(if $(PASS),--pass-id $(call safe,PASS)) $(if $(ALL),--all)

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

# NAME and SUMMARY hold prose, so emptying them on a metacharacter would be a
# real usability loss -- a summary legitimately contains a semicolon. They go
# through the environment instead, where no shell ever sees them.
experiment-open: export TITANIC_NAME := $(NAME)
experiment-open: setup ## Start an experiment: make experiment-open NAME="wide search"
	@$(PYTHON) -m titanic_agent.experiments open --name-env TITANIC_NAME

# Journal the hash of plan.md before anything is scored. `make bakeoff` does
# this itself; tune and features need it run by hand, and `make audit` re-checks
# every locked plan afterwards.
experiment-lock-plan: setup ## Lock this experiment's plan.md before scoring
	@$(PYTHON) -m titanic_agent.experiments lock-plan $(if $(EXPERIMENT),--experiment $(call safe,EXPERIMENT),)

experiment-evaluate: setup ## Score the open experiment's val split, ONCE. Options: RUN=<run id>
	@$(PYTHON) -m titanic_agent.experiments evaluate $(if $(RUN),--run-id $(call safe,RUN))

experiment-close: export TITANIC_SUMMARY := $(SUMMARY)
experiment-close: setup ## Close the open experiment: make experiment-close SUMMARY="..."
	@$(PYTHON) -m titanic_agent.experiments close --summary-env TITANIC_SUMMARY

# Writing to findings.jsonl, which is append-only and whose direct edits are
# denied. Same reason as `make note`: an unattended pass has to record what it
# learned without permission to run arbitrary Python.
finding: export TITANIC_CLAIM := $(CLAIM)
finding: export TITANIC_NOTE := $(NOTE)
finding: setup ## Record a finding: make finding CLAIM="..." STATUS=confirmed EVIDENCE=id1,id2
	@$(PYTHON) -m titanic_agent.findings --record-env TITANIC_CLAIM --note-env TITANIC_NOTE \
		$(if $(STATUS),--as-status $(filter $(STATUS_KEYS),$(STATUS))) $(if $(EVIDENCE),--evidence $(call safe,EVIDENCE)) \
		$(if $(CONFIDENCE),--confidence $(filter $(CONFIDENCE_KEYS),$(CONFIDENCE))) $(if $(DELTA),--delta $(call numeric,DELTA))

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
