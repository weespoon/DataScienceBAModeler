# Experiments

Derived from each `experiment.json` by `experiments.write_index()`.
Do not edit by hand — it will be overwritten.

`cv` is cross-validated on that experiment's own training split.
`val` is its own 142-row validation split, read once, autonomously.
Neither is the final test, which is read once for the whole programme.

| experiment | status | train/val | best cv | val | summary |
| --- | --- | --- | --- | --- | --- |
| `exp-001-baseline` | closed | 569/143 | 0.8287 | — | Baseline sweep of 8 models + soft-vote ensemble. CV over the full 712-row working set; predates per-experiment val splits, so it has no val score. |

## The final test

`data/frozen/holdout.csv` (179 rows) is read **once for the whole**
**programme**, not once per experiment. Reusing it across experiments
would turn it into a validation set: each experiment's direction would be
chosen partly by the previous one's final score, and the number would stop
measuring generalisation and start measuring how hard we searched.

`make receipts` shows whether it has been spent.
