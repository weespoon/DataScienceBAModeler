---
name: ensemble
description: Step 7 of the Titanic pipeline. Combine the best logged models, score the holdout once, and write reports/submission.csv. Use last, after tune has populated the ledger.
---

# Step 7 — ensemble

Combine the best tuned models, log the result, and only then take the single
holdout measurement. This is the only step permitted to read the holdout.

## Inputs

- `experiments/results.csv` — to pick members. Keep only rows whose
  `frozen_sha256` matches the current manifest; scores from an older frozen
  dataset are not comparable.
- `data/frozen/train.csv` for cross-validation and the final fit.
- `data/frozen/holdout.csv` — **only** with `--score-holdout`, and only once per
  frozen dataset.
- `data/frozen/test.csv` for the submission.

## Outputs

- One ledger row with `step="ensemble"`, recording members, weights, `cv_score`
  and `cv_std`; the holdout score goes in that same row's `notes`.
- `reports/03_ensemble.md` — members, weights, CV score, holdout score, and a
  short verdict.
- `reports/submission.csv` — `PassengerId,Survived`, 418 rows, integer labels.

## How to run

```bash
make ensemble                           # iterate safely; holdout untouched
make ensemble METHOD=stack TOP_K=5
.venv/bin/python -m titanic_agent.ensemble --score-holdout   # once, at the end
```

## Done criteria

- [ ] The ensemble's `cv_score` is logged. **Whether it beats the best single
      `tune` row is a result, not a done criterion** — an ensemble that loses to
      one good model is a real and common finding. Say so in the report and ship
      the single model. Never re-roll members until the number improves.
- [ ] Members are at least two, all sharing the current `frozen_sha256`.
- [ ] `reports/submission.csv` has exactly 418 rows plus a header, no missing
      values, and labels in `{0, 1}`.
- [ ] `experiments/holdout_reads.json` holds **exactly one** receipt for the
      current `frozen_sha256` if `--score-holdout` was used, and none otherwise.
- [ ] The ledger row attests `holdout_read` matching what actually happened.
- [ ] `reports/03_ensemble.md` reports CV and holdout side by side.

## Notes

- `soft_vote` needs `predict_proba` from every member; `svc` needs
  `probability=True`, which is slow — prefer `hard_vote` if it is a member.
- Drop candidates whose out-of-fold predictions correlate above
  `max_correlation` (0.95) with an already-selected member. Correlated members
  add cost without adding accuracy.
- The holdout number is a **report, never a selection signal**. If it disappoints,
  write that down. Going back to pick a different model after seeing it converts
  the holdout into a second validation set and the honest estimate is gone.
- Wanting a second look at the holdout means wanting a new frozen dataset. Say so
  in the report instead of peeking.
- This is the last step. Finish by printing the ledger (`make log`).
