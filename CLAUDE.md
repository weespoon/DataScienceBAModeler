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
  shrink the search space rather than waiting it out. The exception is a
  deliberate *measurement* rather than a search: `make nested` measures
  selection bias and is budgeted at ~3 minutes, declared up front and run once
  per searching experiment, never inside `make loop`. The rule exists to stop an
  agent waiting out a slow search, not to forbid a number that can only be got
  by paying for it.

## Pipeline order

Run the steps in this order. Each one has a skill in `.claude/skills/<step>/SKILL.md`
with its inputs, outputs and done criteria — read that skill before running a step.

| # | Step | Module | Reads | Writes |
|---|------|--------|-------|--------|
| 1 | `pull` | `src/titanic_agent/pull.py` | Kaggle API | `data/raw/*.csv` |
| 2 | `review` | `review.py` | `data/raw/` | `reports/01_review.md` |
| 3 | `split` | `split.py` | `data/raw/train.csv` | `data/frozen/split_plan.json` |
| 4 | `freeze` | `freeze.py` | `data/raw/`, split plan | `data/frozen/*.csv`, `MANIFEST.json` |
| 5 | `explore` | `explore.py` | `data/frozen/train.csv` | `runs/<id>/02_explore.md`, charts |
| 6 | `tune` | `tune.py` | `data/frozen/train.csv` | rows in `experiments/results.csv` |
| 7 | `ensemble` | `ensemble.py` | ledger, `data/frozen/` | ledger row, `runs/<id>/submission.csv` |

`make <step>` runs any single step. **`make bootstrap`** runs steps 2–4 once —
`freeze` refuses to run twice, so it can never sit in a loop. **`make loop`**
runs steps 5–7 and is the repeatable part. (There is no **make all** target any
more; it existed once and was removed precisely because it put `freeze` inside
something meant to be re-run.)

Don't skip ahead — step 6 is meaningless before step 4, because a score with no
frozen fingerprint cannot be compared to anything.

## Rule 1 — the frozen dataset is immutable

Once `freeze` has written `data/frozen/MANIFEST.json`, **nothing in
`data/frozen/` is ever modified, regenerated or re-split.** The files are
chmod 0444 and every one is checksummed in the manifest.

- Every step after `freeze` calls `freeze.require_frozen()` first and refuses to
  run if a checksum does not match.
- `freeze()` has no `force` flag. Re-freezing raises.
- A `PreToolUse` hook denies any write to `data/frozen/` once a manifest exists.
  If you hit that denial, the hook is working — do not look for a way around it.
- **Unfreezing is a human action. Stop and ask.** If you believe the frozen data
  must change — a corrupt pull, a structurally broken split — say so, explain
  why, and wait. Do not unfreeze, do not `chmod`, do not delete anything under
  `data/frozen/`, and do not reach for a tool that would get around the hook.
  Re-freezing invalidates every score already in the ledger, because those rows
  carry the old `frozen_sha256` and nothing can recover what they measured.

## Rule 2 — the holdout is read once, at the end

`data/frozen/holdout.csv` exists to give one honest estimate of generalization.

- `explore` and `tune` must never load it. Model selection happens entirely
  through cross-validation on `data/frozen/train.csv`.
- Only `ensemble` may read it, only with `score_holdout=True`, and only once the
  ensemble has already been chosen by CV. The holdout number goes in the report
  and the ledger; it is **never** a signal to go back and pick a different model.
- **This is the second tripwire. Stop and ask.** `--score-holdout` is a
  once-per-dataset, irreversible measurement, and it is deliberately absent from
  every `make` target. An autonomous pass does not take it on its own.
- Every read leaves a receipt in `experiments/holdout_reads.json`, keyed by
  `frozen_sha256`. `receipts.require_unread()` runs *before* the read and
  refuses a second one; `receipts.record_holdout_read()` runs *after*, so the
  read leaves a trace either way.
- No mechanism can stop a stray `pd.read_csv('data/frozen/holdout.csv')` in a
  step that has no business reading it, and file access times cannot reliably
  detect one. That is why **every ledger row attests `holdout_read`.** The
  attestation does not prevent the read; it makes a silent one into an
  attributable false statement. Sign it honestly — including when the honest
  answer is embarrassing. A wrong number you reported is recoverable; a false
  attestation makes every other row untrustworthy.
- If you want a second look at the holdout, you want a new frozen dataset, and
  that is a human decision. Say so in the report instead of peeking.

## Rule 3 — every experiment appends to the ledger

`experiments/results.csv` is append-only. A run that is not in it did not happen.

- One row per run, written through `titanic_agent.results_log.append_run()`.
  Never hand-edit the CSV, never rewrite or delete rows, never reorder columns
  (append new ones at the end so old rows stay readable).
- Columns: `run_id`, `timestamp_utc`, `step`, `model`, `params`, `cv_score`,
  `cv_std`, `cv_folds`, `seed`, `frozen_sha256`, `git_sha`, `holdout_read`,
  `notes`, `pass_id`, `prev_sha`.
- `prev_sha` chains each row to the hash of the line above it, so a deleted or
  edited row breaks the chain and `make audit` says so. Appends are locked;
  without the lock two concurrent sessions fork the chain and the audit reports
  tampering that never happened.
- Log failures too, with the error in `notes` and an empty `cv_score`. A failed
  configuration is a result; silently dropping it makes the sweep a lie.
- A row cannot be edited, so a correction is an **amendment, not an edit**:
  append a new row whose `notes` reference the original `run_id`. If a run dies
  before it can log, log that fact afterwards — an exception that kills the
  process writes nothing, and a gap in the ledger is indistinguishable from a
  run you chose not to record.
- `git_sha` carries a `-dirty` suffix when anything under
  `results_log.CODE_PATHS` has uncommitted changes — `src`, `tests`, the
  dependency files, the `Makefile`, `.claude` and this file. **Not** the whole
  tree: the pipeline appends to tracked files of its own, so opening an
  experiment dirtied the tree before its first row existed and the flag could
  never clear. Do not strip it, and do not narrow that list to make a row look
  clean. A score you cannot tie to committed code is not reproducible, and the
  column exists to say so out loud.
- `make log` prints the ledger. Compare only rows sharing a `frozen_sha256`.

## Rule 4 — do not look up the answers

The real Titanic passenger manifest is public: all 1309 passengers, with who
survived. A model that joins against it is not predicting anything.

This is not about Kaggle etiquette. **Our locked final test is 179 of those same
passengers**, so a lookup inflates the one number this whole programme is built
around. `make audit` would stay green. The ledger would chain cleanly. Rules 1–3
would all hold. Nothing else here would notice — which is exactly why this is a
rule rather than a convention.

- **External data about actual Titanic passengers is out of bounds.** The
  manifest, any derived lookup table, any "feature" that is a join against one.
- **"It scored higher" is never a justification.** If you cannot say why a
  technique should generalise, it does not go in the backlog and it does not get
  run.
- **Treat any reported score above ~0.85 as suspect.** The honest ceiling on 891
  labelled rows is roughly 0.80–0.84. Public notebooks above that are
  disproportionately doing the lookup.
- `make leakage` checks this structurally: feature provenance, external reads,
  score plausibility, stray data files. Run it after acting on anything read
  externally. It raises the cost and makes an attempt visible; it cannot make
  cheating impossible, which is why the three points above come first.

## The measurement hierarchy — use the instrument that can see it

Three instruments measure accuracy here, and they are not three grades of the
same thing. Every decision this project makes is worth **0.003 to 0.015**, and
only one of them can see a difference that small.

| instrument | rows | resolves | role |
|---|---:|---:|---|
| **paired repeated CV** (`compare.py`) | 569 | ~0.006 | **adjudicates** — compares two candidates |
| **validation read** (once per experiment) | 143 | ~0.064 | **tripwire** — catches a broken pipeline |
| **final test** (once, ever) | 179 | ~0.058 | **reports** — one generalisation number |

- **Comparisons go through `titanic_agent.compare`.** Never compare two absolute
  CV numbers from *different* experiments: the split is re-drawn each time, so
  they are unpaired, and identical configurations have drifted 0.004–0.011
  between experiments — one to three times the effect you are looking for.
- **The validation read is a tripwire, not a calibrator.** It is a pre-declared
  number compared against nothing. It cannot rank two candidates and it cannot
  measure selection bias; it catches 0.06-sized defects, which is how it caught
  exp-002's contamination. Do not read a cv→val gap as a measurement of
  anything — the five recorded gaps are +0.027, +0.048, −0.029, +0.014, +0.042.
- **A val read is no longer independent of the champion it scores.** Each split
  is a fresh draw from the same 712 rows, so ~80% of an experiment's validation
  rows were the previous experiment's *training* rows — the rows that selected
  the champion (exp-008: 116 of 143). The persisted champion bought
  comparability with exactly this. The tripwire still works; a gap read as
  evidence about selection does not.
- **Any feature fitted from the label runs `make permutation` first.** The
  in-fold group rates are the classic Titanic leak, and `make bakeoff` refuses
  to score a label-reading arm until the control lands at chance. Re-run it
  after any change to `bakeoff.Engineer` — a control you cannot re-run is a
  number, not a control.
- **Do not ask for more validation reads.** The answer is arithmetic, already
  worked out: 143 rows move in steps of 1/143 = 0.0070, so a true difference of
  0.004 cannot show up as 0.004. Pairing gets to ~0.06; Thresholdout's own noise
  on 143 rows has sd 0.020. A split that could adjudicate 0.004 needs ~59,000
  rows and we have 891. `require_val_unread` refuses, and says this.
- **`DELTA = 0.003` is about one standard error, and 1.7 passengers.** Measured
  over 10 fold seeds, not assumed. So report the sigma with every comparison,
  and treat a single test near the bar as a coin flip rather than a decision.

**`docs/MEASUREMENT.md` has the arithmetic, the measurements behind these
claims, and the open items.** Read it before proposing a new way to measure
anything — most of them have already been costed and refused there.

## Conventions

- Paths come from `titanic_agent.paths`. Don't hardcode `data/...` anywhere else.
- The default seed is `paths.SEED`. Pass it to every estimator and every split.
- Reports are numbered by step (`01_review.md`, `02_explore.md`, `03_ensemble.md`)
  and are regenerable — `make clean` deletes them and that must be harmless.
  Steps 5 and 7 write into `runs/<id>/` and publish a copy to `reports/`.
- Charts: matplotlib only, `Agg` backend, one chart per figure, always
  `plt.close()`. Keep figures legible at ~110 dpi.
- `make test` before any commit. `make lint` for style.
- Never commit `data/raw`, `data/frozen`, `.env` or `kaggle.json` — all four are
  gitignored. Credentials never appear in code, logs, reports or commit messages.

## Tripwires — never yours alone

Two actions are always a human's call, however reasonable they look in the
moment. Refuse them, say why you think they are needed, and wait:

1. **Unfreezing** — any `chmod`, delete, overwrite or re-freeze under
   `data/frozen/`. A `PreToolUse` hook will refuse you. The correct response to
   that refusal is to stop, not to look for another route.
2. **`ensemble --score-holdout`** — the one irreversible measurement, once per
   frozen dataset. It is deliberately absent from every `make` target so that
   running the loop cannot trigger it.

Record the refusal: `run_log.record("refused", step=..., detail=...)`. A refusal
that leaves no trace is indistinguishable from never having considered it.

## When something is ambiguous

Prefer the cheap, reversible option, write down the decision in the step's report
and in `experiments/run_log.jsonl`, and keep going. The point of this repo is a
full autonomous run with a legible trail, not a perfect leaderboard score.

But "keep going" stops at the tripwires above, and at anything irreversible that
this file does not cover. The test is not *is this a good idea* — it is *can a
human undo it tomorrow*. If not, ask.

## Where things are written

| | Lives in | Why |
|---|---|---|
| This pass's charts, step reports, submission, `run.md` | `runs/<id>/` | Per-pass, regenerable, safe to delete |
| Latest copy of each | `reports/` | The always-current view |
| The ledger, holdout receipts, the journal | `experiments/` | **Global and append-only** |

That last row is not a filing preference. Put holdout receipts inside a run
directory and "the holdout is read once per frozen dataset" silently becomes
"once per *run*": every pass sees an empty receipts file, reads the holdout, and
passes its own audit. Unlimited reads, all green. Split the ledger the same way
and you can no longer compare passes, while the hash chain stops noticing a
deleted run. **Artifacts are per-pass; evidence is global.**

A run id (`20261010-1446EDT`) is also the `pass_id` on every ledger row and
journal entry that pass wrote, so `ls runs/<id>/` and `make journal PASS=<id>`
name the same thing.

Options to `make` are **variables, not flags**: make consumes a leading `--flag`
itself and the target never sees it. `make journal LEVEL=decision`, not
`make journal` followed by a dash-dash option. A test enforces this across every
document, because the mistake has been made twice.

## Orchestration

`.claude/skills/run/SKILL.md` drives a full pass: preflight, pick one item from
`make backlog`, open an experiment, run the search that item calls for, read the
validation split once, close with a verdict. **A pass is one experiment, not one
lap of `make loop`.** The step skills are procedures; that one is the policy.
Read it before running the pipeline unattended.

`make agent` runs exactly that, headless — `claude -p` pointed at the skill,
capped by `TURNS` and `BUDGET` (variables, not flags), with the transcript in
`runs/<pass id>/agent.json`. It is deliberately not run with
`--permission-mode bypassPermissions`: the deny rules in `.claude/settings.json`
are what keep an unattended pass away from `chmod` under `data/frozen` and from
the one route to `--score-holdout`.

A **denied tool call is a stop condition**, not something to route around with a
different command. Say which command was denied. `make note DETAIL="..."` writes
a journal entry, so a pass can record why it stopped without needing permission
to run arbitrary Python — the first headless pass could not, and a pass that
stops without saying why is indistinguishable from one that crashed.

Every pass ends with `make report` — `runs/<pass id>/report.md` plus a copy in
`reports/`, holding the experiment and its verdict, whether its plan still
hashes to what was locked, the champion, `make audit` and `make leakage` check
by check, the receipts, and the pass's decisions and refusals. Run it even when
the pass stopped early; especially then.

Nothing retries a pass. Re-running one whose verdict you disliked is selecting
on the outcome.
