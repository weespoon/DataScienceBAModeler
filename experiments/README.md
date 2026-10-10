# experiments/

`results.csv` is the append-only experiment ledger. One row per run.

Rule 3 in [CLAUDE.md](../CLAUDE.md): a run that is not in this file did not
happen. Rows are written only through
`titanic_agent.results_log.append_run()` — never by hand.

| column | meaning |
|---|---|
| `run_id` | `<UTC timestamp>-<6 hex>`, unique per run |
| `timestamp_utc` | ISO-8601, second precision |
| `step` | `tune` or `ensemble` |
| `model` | estimator class, or `method(members)` for an ensemble |
| `params` | compact JSON of the chosen hyperparameters |
| `cv_score` | mean cross-validated accuracy; empty if the run failed |
| `cv_std` | standard deviation across folds |
| `cv_folds` | number of folds |
| `seed` | random seed used for folds and estimators |
| `frozen_sha256` | fingerprint of the frozen dataset the score was computed on |
| `git_sha` | short commit SHA of the code that produced it |
| `notes` | free text: failures, holdout scores, caveats |

**Only compare rows that share a `frozen_sha256`.** A score computed on a
different frozen dataset is not comparable to one computed on this one.

```bash
make log                                      # print everything
.venv/bin/python -m titanic_agent.results_log --best --step tune
```
