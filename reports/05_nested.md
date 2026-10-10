# 05 — nested CV: what the feature search is actually worth

Model **random_forest**, hyperparameters held fixed at `{'clf__max_depth': None, 'clf__min_samples_leaf': 3, 'clf__n_estimators': 400}` across both arms and every outer fold.

5 outer folds over 569 training rows. Inside each, the **same** `features.greedy_search` the real run calls, on that fold's training rows only, over 8 candidates (trimmed from 24). Arm B is the fixed 12-feature baseline. Both arms are scored on the outer test rows, which neither arm's search saw. 120s.

| outer fold | test rows | features chosen | inner cv | A search | B baseline | A − B |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 114 | 4 | 0.8462 | 0.7982 | 0.7982 | +0.0000 |
| 2 | 114 | 6 | 0.8322 | 0.8596 | 0.8596 | +0.0000 |
| 3 | 114 | 4 | 0.8352 | 0.7982 | 0.8333 | -0.0351 |
| 4 | 114 | 10 | 0.8410 | 0.8246 | 0.8246 | +0.0000 |
| 5 | 113 | 5 | 0.8326 | 0.8673 | 0.8761 | -0.0088 |

## Does searching beat not searching?

- **nested: search vs no search: -0.0088 +/- 0.0068 (naive SE, -1.29 sigma; -5.0 passengers)**
- nested(search) **0.8296**, nested(baseline) **0.8384**

## How optimistic was the published number?

- reported CV after the search: **0.8401**
- nested estimate of the same procedure: **0.8296**
- **selection bias: +0.0105**

## Scope

- Hyperparameters are **not** nested: they are held fixed from the real run, so this is the selection bias of the feature search alone.
- The candidate list is trimmed, so this measures the trimmed procedure. 19 of the 24 available candidates measured negative in exp-006; keeping all of them would cost several times as much to estimate a search nobody intends to repeat.
- Outer test folds hold ~113 rows each, so the absolute nested numbers are noisy. The paired difference is the measurement; the absolute levels are context.
