# CLAUDE.md — operating rules for this repo

This repo is an experiment in agentic AI: Claude runs the whole Kaggle Titanic
workflow end to end, and the human reviews the artifacts afterwards. Work
autonomously, but obey the three hard rules below without exception.

## Setup

- Python **3.11+**. `make setup` creates `.venv` and installs the package editable.
- Dependencies are **pandas, numpy, scikit-learn, matplotlib**, plus `pytest`
  and `ruff` for dev and the optional `kaggle` CLI for step 1 only.
  Adding anything else requires a one-line justification in `pyproject.toml`
  next to the dependency. Do not reach for xgboost, lightgbm or optuna.
- Everything must run on a laptop CPU. If a step takes more than ~2 minutes,
  shrink the search space rather than waiting it out.

## Pipeline order

Run the steps in this order. Each one has a skill in `.claude/skills/<step>/SKILL.md`
with its inputs, outputs and done criteria — read that skill before running a step.

| # | Step | Module | Reads | Writes |
|---|------|--------|-------|--------|
| 1 | `pull` | `src/titanic_agent/pull.py` | Kaggle API | `data/raw/*.csv` |
| 2 | `review` | `review.py` | `data/raw/` | `reports/01_review.md` |
| 3 | `split` | `split.py` | `data/raw/train.csv` | `data/frozen/split_plan.json` |
| 4 | `freeze` | `freeze.py` | `data/raw/`, split plan | `data/frozen/*.csv`, `MANIFEST.json` |
| 5 | `explore` | `explore.py` | `data/frozen/train.csv` | `reports/02_explore.md`, charts |
| 6 | `tune` | `tune.py` | `data/frozen/train.csv` | rows in `experiments/results.csv` |
| 7 | `ensemble` | `ensemble.py` | ledger, `data/frozen/` | ledger row, `reports/submission.csv` |

`make <step>` runs any single step; `make all` runs steps 2–7 in order. Don't
skip ahead — step 6 is meaningless before step 4, because a score with no frozen
fingerprint cannot be compared to anything.

## Rule 1 — the frozen dataset is immutable

Once `freeze` has written `data/frozen/MANIFEST.json`, **nothing in
`data/frozen/` is ever modified, regenerated or re-split.** The files are
chmod 0444 and every one is checksummed in the manifest.

- Every step after `freeze` calls `freeze.require_frozen()` first and refuses to
  run if a checksum does not match.
- `freeze()` has no `force` flag. Re-freezing raises.
- If the frozen data genuinely must change — a corrupt pull, a bad split — that
  is a deliberate manual act (`chmod -R u+w data/frozen && rm -rf data/frozen/*`),
  and it **invalidates every score already in the ledger**. Note it in the
  `notes` column of the next run and start the ledger comparison over, because
  scores carrying a different `frozen_sha256` are not comparable.

## Rule 2 — the holdout is read once, at the end

`data/frozen/holdout.csv` exists to give one honest estimate of generalization.

- `explore` and `tune` must never load it. Model selection happens entirely
  through cross-validation on `data/frozen/train.csv`.
- Only `ensemble` may read it, only with `score_holdout=True`, and only once the
  ensemble has already been chosen by CV. The holdout number goes in the report
  and the ledger; it is **never** a signal to go back and pick a different model.
- If you find yourself wanting a second look at the holdout, that means you want
  a new frozen dataset. Say so in the report instead of peeking.

## Rule 3 — every experiment appends to the ledger

`experiments/results.csv` is append-only. A run that is not in it did not happen.

- One row per run, written through `titanic_agent.results_log.append_run()`.
  Never hand-edit the CSV, never rewrite or delete rows, never reorder columns
  (append new ones at the end so old rows stay readable).
- Columns: `run_id`, `timestamp_utc`, `step`, `model`, `params`, `cv_score`,
  `cv_std`, `cv_folds`, `seed`, `frozen_sha256`, `git_sha`, `notes`.
- Log failures too, with the error in `notes` and an empty `cv_score`. A failed
  configuration is a result; silently dropping it makes the sweep a lie.
- `make log` prints the ledger. Compare only rows sharing a `frozen_sha256`.

## Conventions

- Paths come from `titanic_agent.paths`. Don't hardcode `data/...` anywhere else.
- The default seed is `paths.SEED`. Pass it to every estimator and every split.
- Reports are numbered by step (`01_review.md`, `02_explore.md`, `03_ensemble.md`)
  and are regenerable — `make clean` deletes them and that must be harmless.
- Charts: matplotlib only, `Agg` backend, one chart per figure, always
  `plt.close()`. Keep figures legible at ~110 dpi.
- `make test` before any commit. `make lint` for style.
- Never commit `data/raw`, `data/frozen`, `.env` or `kaggle.json` — all four are
  gitignored. Credentials never appear in code, logs, reports or commit messages.

## When something is ambiguous

Prefer the cheap, reversible option, write down the decision in the step's report,
and keep going. The point of this repo is a full autonomous run with a legible
trail, not a perfect leaderboard score.
