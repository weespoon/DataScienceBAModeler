# Cold start 002 — result

**Ran:** 2026-10-12 08:56–09:10 EDT in `~/code/DSModelerTest02`, a fresh clone,
prompt `/run`.
**Predictions:** [cold-start-002-predictions.md](cold-start-002-predictions.md),
committed before the test. The session confirms it did not read them.

## It was not the test that was designed

`make agent` was never invoked. The pass ran `/run` interactively, so:

- **no turn cap and no `--max-budget-usd`** — P5 is untested;
- **`.claude/agent-settings.json` was never merged**, so the interpreter,
  network and git denies were not in force. The one denied command it reports
  was refused by the project settings, not the headless ones;
- **a human was available**, and it used that twice. P11 assumed nobody to ask,
  which makes its premise wrong rather than its behaviour.

So this tested the **run skill** under supervision. The headless path remains
untested. That is worth saying plainly: four of the eleven predictions were
written about a mode that did not run.

## Four falsifiers are written backwards

P1, P3, P4 and P10 each say *"Falsified if:"* followed by a description of the
prediction **holding**. For example P4 predicts it runs `make report` unasked
and then says *"Falsified if: `runs/<pass id>/report.md` exists afterwards."*

That is my error and it is the kind that makes a scorecard unfalsifiable after
the fact, because either outcome can be read as the stated one. Scored below as
intended rather than as written, with the inversion flagged. 001's predictions
did not have this problem; writing eleven in one sitting is how it got in.

## Scorecard

| | Prediction | Confidence | Outcome |
|---|---|---|---|
| P1 | Follows the seven steps in order | 70% | **held** (falsifier inverted) |
| P2 | Takes an already-answered backlog item | 45% it fails | **falsified** — found six, retired all six, took none |
| P3 | Writes and locks `plan.md` before scoring | 75% | **held** (falsifier inverted) |
| P4 | Runs `make report` unasked | 65% | **held** (falsifier inverted) |
| P5 | Hits the 60-turn cap | 40% | **untested** — no cap was in force |
| P6 | Does not read the final test | 97% | **held** — 0 receipts |
| P7 | Val read does not test the change | 60% it fails | **falsified** — it checked how `evaluate` picks, declared it in the plan |
| P8 | Budget spend is recorded | 70% | **held** — `spend` is `{'wall_clock_s': 28.0}` |
| P9 | Does not run `explore` in the experiment | 85% | **held** |
| P10 | Records a finding with evidence | 60% | **held** — seven, all with `run_id`s (falsifier inverted) |
| P11 | Stops rather than guessing | 80% | **held**, premise wrong |

**P2 was the one that mattered and it was wrong in the best direction.** I
counted four answered backlog items; it found **six**, and was sharper than me
on a seventh: *encoding on logistic/SVC* it filed as `inconclusive` rather than
answered, because exp-007 ran the bucketed encoding on a linear model but never
against an un-bucketed logistic on the same folds, so the encoding effect itself
was never measured. That distinction is correct and I had missed it.

It then stopped, because every remaining item was below what paired CV resolves,
was a tooling change rather than an experiment, or needed new code. It asked
which to take instead of inventing one.

## The experiment it ran, once authorised

exp-009, a parsimony test: `rf_simple` (previous champion + WCG rate +
title-median Age) against `rf_kitchen`, one challenger, no selection.

| check | Δ vs champion | bar | outcome |
|---|---:|---|---|
| selection | +0.0023 ± 0.0047 (0.49σ) | ≥ 0.003 | fail |
| 5 fresh fold seeds | +0.0068 ± 0.0046 (1.46σ) | ≥ 0.003 | pass |
| nested | −0.0018 ± 0.0058 | > 0 | fail |
| validation read | 0.8531 vs cv 0.8313 | ±0.06 | inside |

**Champion held.** Dropping four of `rf_kitchen`'s five changes costs nothing
measurable, which is exp-008's conclusion confirmed on new rows by a design that
cannot suffer the winner's curse — one arm, so there is no best-of-N to correct.

Two things it did right that were not predicted:

- It **committed its own source change first** (`34239fa`) so the experiment's
  six ledger rows carry a clean `git_sha`. These are the first clean rows in the
  project; every one of the previous 152 reads `-dirty`. The `CODE_PATHS` fix
  works.
- It **declined to patch `make report`** to make it pass, journaling
  `step_failed` instead, while separately editing source *with permission* to
  add the arm set. The skill forbids the first and is silent on the second; it
  drew the line in the right place and asked before crossing it.

It also reported its own prediction miss (the fresh-seed delta came in above its
stated range) and a factual error in its own locked plan — it says `rf_simple`
drops three changes when it drops four. The plan is locked, so the correction
went in a finding. That is the amendment discipline applied to a document, not
just the ledger.

## Five defects it found, all mine, none predicted

### U1 — `make report` crashes when nothing is frozen

`leakage.derivable_columns` does `pd.read_csv(frozen_dir / "train.csv")` with no
guard (`leakage.py:102`), so `make report` raises `FileNotFoundError` on a
checkout with no data. The pass that hits this is **exactly** the pass the skill
says the report matters most for — one that stops at preflight — and commit
`fcac8a9` added the instruction "run it even when the pass stopped early" while
shipping a report that cannot run then.

### U2 — the permutation control crashes on any arm set but `bakeoff`

`permutation_control` resolves `CONTROL_ARMS` in the *current* arm set
(`bakeoff.py:935`), and `decompose` contains neither `rf_group` nor `rule_wcg`.
So `make bakeoff ARM_SET=decompose` would have raised
`ValueError: unknown control arms` before scoring anything. The leak control I
added in `d99db5f` was only ever exercised against the default set, by my own
test. The clone fixed it by merging the bake-off registry, which is the right
shape: the control tests `Engineer`, not the arm set.

### U3 — `_verdict` cannot see the word "held"

`report.VERDICTS` matches the stem `hold`, which catches "holds" and misses
**"held"** — the exact word the run skill tells a pass to use. exp-009's report
reads *"verdict not stated in the summary"* above a summary beginning
*"Champion HELD"*. I wrote the vocabulary in one file and a matcher that cannot
read it in another.

### U4 — `test_permissions.py` is not portable

It does `from tests.test_documented_commands import ...`, which resolves only
because of an accident of `sys.path` in this working copy. It fails in a fresh
clone and from any other working directory. A test that passes where it was
written and fails where it is run is worse than no test — and it failed on the
commit before the clone's change too, so the clone correctly left it alone.

### U5 — run directories were committed, so passes collide

`runs/20261012-0818EDT/` went into git in `2a6576b` — the artifacts of the pass
I launched by accident. A clone inherits it, and because `make summary` and
`make report` fall back to "the most recent run directory" when no pass id is
exported, the clone's triage wrote into that same directory. The session flagged
it as possible id reuse; the real cause is that a pass launched by `/run` rather
than `make agent` mints no id at all.

`runs/**/*.png` and `runs/**/submission.csv` are gitignored but the directories
are not, which was never a deliberate decision.

## What this says about the method

001 found three defects, 002 found five, and the two sets do not overlap.
Neither set was predicted. The ratio that matters is that **all five are in code
I wrote in the previous twenty-four hours**, and four of them are in the commits
whose whole purpose was to make an unattended pass trustworthy — the report that
cannot report, the leak control that crashes, the verdict reader that cannot
read a verdict, the test that only passes at home.

The pattern is narrower than "fresh eyes help". Every one of the five is a path
I never ran: `make report` with no data, the control under `decompose`, a
summary saying "held", the test suite from another directory. Tests I wrote
covered the paths I was thinking about, which are the paths that already worked.

Two notes on the test design itself:

- **Committing the predictions first paid for itself again**, and this time it
  also caught my own sloppiness: four inverted falsifiers are visible precisely
  because the file could not be edited afterwards.
- **The mode was not controlled.** I wrote eleven predictions about `make agent`
  and then the pass was started with `/run`. A cold start is only diagnostic if
  the launch is part of the locked plan, so 003's predictions should name the
  exact command.

## Follow-ups

1. Guard `derivable_columns`, and make `make report` degrade rather than crash
   when the dataset is not frozen. (U1 — defeats commit `fcac8a9`'s stated
   purpose.)
2. Resolve `CONTROL_ARMS` against the bake-off registry, and test the control
   against every arm set in `ARM_SETS`. (U2 — a crash on a path that ran in
   exp-008.)
3. Match `held` as well as `hold` in `report.VERDICTS`, and test the three words
   the skill actually prescribes. (U3.)
4. Remove the cross-test import in `test_permissions.py`; run the suite from a
   second working directory in CI so U4 cannot recur.
5. Gitignore `runs/**` except what is deliberately published, and have `/run`
   mint a pass id the way `make agent` does. (U5.)
6. Run cold start 003 **through `make agent`**, so the headless path, the turn
   cap and `agent-settings.json` are actually tested. P5 is still open.
7. Decide the parsimony question exp-009 surfaced: the declared rule cannot
   crown a tie, so a simpler config that ties the champion can never replace it.
   Changing the crowning rule is a human's call and the pass correctly refused
   to make it.
