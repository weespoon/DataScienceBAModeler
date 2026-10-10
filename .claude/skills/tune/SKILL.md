---
name: tune
description: Step 6 of the Titanic pipeline. Cross-validate hyperparameter searches on the frozen training split and append every result to experiments/results.csv. Use after explore and before ensemble.
---

# Step 6 — tune

Search hyperparameters by cross-validation on the frozen training split, and log
every run. **A run that is not in the ledger did not happen.**

## Inputs

- `data/frozen/train.csv`, reached through `freeze.require_frozen()`.
- The candidate features from `reports/02_explore.md`.
- Model keys from `tune.MODEL_KEYS`: `dummy`, `logistic`, `random_forest`,
  `extra_trees`, `gradient_boosting`, `hist_gradient_boosting`, `svc`, `knn`.

## Outputs

- One appended row in `experiments/results.csv` per model tuned, written via
  `results_log.append_run(step="tune", ...)`, carrying `params`, `cv_score`,
  `cv_std`, `cv_folds`, `seed` and the current `frozen_sha256`.
- The refit best estimator, returned in memory. Nothing is pickled into
  `data/frozen/`.

## How to run

```bash
make tune                                  # sweep every model key
make tune MODEL=random_forest N_ITER=40    # one model, deeper search
```

## Done criteria

- [ ] Every evaluated model has exactly one ledger row, `dummy` included.
- [ ] The best `cv_score` beats the `dummy` baseline (~0.616 majority class).
- [ ] Every row carries a non-empty `frozen_sha256` matching the current manifest.
- [ ] Each model's search stays under `time_budget_s` (default 120s) on one CPU.
- [ ] No step in this run read `holdout.csv`.

## Notes

- `StratifiedKFold(shuffle=True, random_state=seed)`. Always stratify — the
  classes are 62/38 and unstratified folds add noise for nothing.
- All preprocessing goes **inside** the `Pipeline`, so imputation and scaling are
  fitted per fold. Fitting a scaler on the full training split before CV leaks
  across folds and inflates the score.
- `n_jobs=-1` is fine; these models are small.
- Log failures too: empty `cv_score`, the error in `notes`. A failed
  configuration is a result, and dropping it silently makes the sweep a lie.
- Breadth across models beats depth within one. Keep grids small.
- Next step: `ensemble`.
