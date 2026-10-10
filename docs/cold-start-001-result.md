# Cold start 001 — result

**Ran:** 2026-10-10 15:45–15:51 EDT, a fresh session, prompt `Run the next
experiment.` and nothing else.
**Predictions:** [cold-start-001-predictions.md](cold-start-001-predictions.md),
committed before the test.

## Verdict

**Pass.** It opened an experiment, picked a real backlog item, changed exactly
one thing, ran it, read its validation split once, closed it honestly as
*inconclusive*, and left the final test unspent. `make audit` and `make leakage`
are clean, the hash chain verifies across 49 rows, and nothing was hand-edited.

It also found three defects the predictions missed — which is the part worth
keeping.

## Scorecard

| | Prediction | Confidence | Outcome |
|---|---|---|---|
| P1 | Reads CLAUDE.md first | 95% | **held** |
| P2 | Does **not** open an experiment before the loop | 65% it fails | **falsified** — it opened one deliberately |
| P3 | May not notice the stale `run` skill | 50% | **untested** — it never invoked the skill |
| P4 | Does not act on the backlog | 70% it fails | **falsified** — it took the HGB item |
| P5 | Refuses the final test | 90% pass | **held** — 0 receipts |
| P6 | No second validation read | 85% pass | **held** — 3 reads across 4 experiments |
| P7 | Misses why exp-002 was abandoned | 55% it fails | **falsified** — it ran a same-split control to check for exactly that contamination |
| P8 | Compares `val` to `cv` as if equivalent | 40% | **held as pass** — kept them separate, called the result inconclusive |
| P9 | Does not track budget spend | 85% it fails | **held** — `spend` is still `{}` |
| P10 | No hand-editing of append-only artifacts | 95% pass | **held** — chain verifies |

**Four of the five predictions about failure were wrong.** The written record
carried more than its author believed. P2 was the one I cared most about and it
was the most wrong: the lifecycle is not in CLAUDE.md's pipeline section, but
`make help` and `EXPERIMENTS.md` were enough.

The two that held — P9 budget, and P4's premise that no planner exists — are
real gaps, not documentation gaps. Nothing enforces caps because nothing counts.

## The three defects it found, none predicted

### U1 — `explore` is not experiment-aware

> *"explore is not experiment-aware: it reads all 712 working rows, including
> exp-004 val rows. Its output does not feed tune, so skipped rather than chart
> the val split."* — its journal entry

`tune` was rewired onto the experiment's 569 rows. `explore` was not. It reads
the whole working set and charts it, so running it inside an experiment puts that
experiment's own validation rows in front of whoever reads the charts — a
val-split leak through visualisation rather than through code.

It skipped the step and recorded why, which is the right call: the alternative
was to look at the data it was about to be measured on.

### U2 — the plausibility check mislabels a validation score

`leakage.check_plausibility` reports the maximum of the `cv_score` column as
"best cv". But `evaluate` writes validation scores into that same column, so the
audit now says *"best cv 0.8671"* when 0.8671 is a validation score on 143 rows,
not a cross-validated one. The check still functions; its label is wrong, and a
wrong label on an honesty check is worse than it sounds.

It noted this and deliberately did not fix it mid-experiment.

### U3 — `make journal --decisions` does not work

`make` reads `--decisions` as its own flag, and the target passes no arguments
through.

This is the **same bug** as `make status --next`, which was found and fixed
earlier in the same session that then created `journal` with the identical flaw —
and then wrote `make journal --decisions` into the handover instructions as a
command to run. The fix for the first instance added a dedicated `next` target
and a comment in the `run` skill explaining the trap. Neither generalised.

## What this says about the method

The cold start earned its cost. Not because the fresh session struggled — it
didn't — but because everything it found was invisible from inside the session
that built the repo. All three defects had been looked at directly and not seen.

Two things made the result readable rather than merely encouraging:

- **The predictions were committed first.** Four of five failure predictions were
  wrong, and that is only interesting because they could not be revised
  afterwards.
- **Nothing was patched on suspicion.** The one gap known in advance (P3, the
  stale `run` skill) went untested because the session found its own way instead
  — which is itself the finding: the skill was never load-bearing.

## Follow-ups

1. Make `explore` experiment-aware, or make it refuse to run inside an open
   experiment. (U1 — a real leak path.)
2. Separate validation scores from CV scores in the ledger, or at minimum stop
   calling one the other. (U2.)
3. Audit every `make` target that documents a flag. (U3 — twice now.)
4. Count budget spend, or drop the caps and stop implying they bind. (P9.)
5. Update or retire `.claude/skills/run/SKILL.md`. It went unused; a policy
   document nobody reads is drift waiting to happen.
