# Experiments

Derived from each `experiment.json` by `experiments.write_index()`.
Do not edit by hand — it will be overwritten.

`cv` is cross-validated on that experiment's own training split, and is
annotated with how many comparisons it was selected from where that is
known — a post-search `cv` is biased upward. `val` is its own 143-row
validation split, read once, autonomously.

**`gap` is not a measurement.** At 143 rows the 2 SE band is about
±0.061, so a gap smaller than that says nothing; the
column is here to make that obvious rather than to be tracked.
Comparisons belong in `compare.py`.
Neither number is the final test, which is read once for the whole programme.

| experiment | status | train/val | best cv | val | gap | summary |
| --- | --- | --- | --- | --- | --- | --- |
| `exp-001-baseline` | closed | 569/143 | 0.8287 | — | — | Baseline sweep of 8 models + soft-vote ensemble. CV over the full 712-row working set; predates per-experiment val splits, so it has no val score. |
| `exp-002-wider-search` | abandoned | 569/143 | 0.8278 | 0.8392 | +0.0114 ±0.061 (within noise) | ABANDONED: its validation score was contaminated. select_members inherited members tuned on the full working set, which includes this experiment's own validation rows. Bug fixed; a fresh split is needed, so exp-003 supersedes it. |
| `exp-003-wider-search` | closed | 569/143 | 0.8190 | 0.8462 | +0.0271 ±0.060 (within noise) | Re-run of exp-002 after fixing the member-scoping leak. CV 0.8190, val 0.8462 on 143 rows. No technique changed; this establishes a clean baseline under the experiment container. |
| `exp-004-hgb-regularisation` | closed | 569/143 | 0.8190 | 0.8671 | +0.0482 ±0.057 (within noise) | Widened HGB search with regularisation params (min_samples_leaf, l2, max_depth). Same-split control: +0.0035 CV, within noise; winner used less regularisation than default. Best CV 0.8190 (RF; ensemble ties); val 0.8671 on 143 rows scored RF, not the changed model. Inconclusive. |
| `exp-005-budget-check` | abandoned | 569/143 | — | — | — | Scratch: verifying the budget counter. No modelling. |
| `exp-006-feature-selection` | closed | 569/143 | 0.8401 <br><sub>selected from 36</sub> | 0.8112 | -0.0289 ±0.065 (within noise) | Feature selection vs one fixed RF (re-tuned once, then frozen). 24 new row-wise features each added: none cleared +0.003 (best TitleFine +0.0012). Pruning dropped Age, SibSp, Parch, IsAlone, Deck, AgeMissing: CV 0.8360 -> 0.8401 on 6 features. Val 0.8112 on 143 rows for the pre-declared set; within noise. 36/50 tests. |

## The final test

`data/frozen/holdout.csv` (179 rows) is read **once for the whole**
**programme**, not once per experiment. Reusing it across experiments
would turn it into a validation set: each experiment's direction would be
chosen partly by the previous one's final score, and the number would stop
measuring generalisation and start measuring how hard we searched.

`make receipts` shows whether it has been spent.
