# Findings

Claims with evidence, from experiment `exp-004-hgb-regularisation`. Rendered from the append-only
`experiments/findings.jsonl` — edit that through `findings.record()`,
never this file.

## Confirmed (2)

- **+0.0175** Cross-experiment CV deltas are dominated by the split, not the technique _(confidence: medium)_ — evidence: `20261010T194535-d04481`, `20261010T194604-a64610`
  - HGB rose 0.8014 (exp-003) -> 0.8189 (exp-004), but the same-split control accounts for all but 0.0035 of that; gradient_boosting, extra_trees and logistic also rose 0.005-0.012 with no change. Compare techniques within one experiment via a control row, never across experiments.
- Validation 0.8671 vs CV 0.8190 on 143 rows is noise, and measures random_forest, not the lever under test _(confidence: low)_ — evidence: `20261010T194625-226d5d`
  - SE on 143 rows ~0.032, so +0.048 is ~1.5 SE. Above the ~0.85 suspect line, but the model is an unchanged RF and make leakage is clean. Second experiment running with val > CV (exp-003 +0.027); worth watching, not yet a pattern.

## Inconclusive (1)

- **+0.0035** Widening HGB with min_samples_leaf / l2_regularization / max_depth does not measurably help _(confidence: low)_ — evidence: `20261010T194535-d04481`, `20261010T194604-a64610`
  - Same split, same folds: wide grid 0.8189 vs old grid (control) 0.8155, +0.0035 against a ~0.017 fold std. The winner chose LESS regularisation than default (min_samples_leaf=10, l2=0, no depth cap), so the regularisation hypothesis is not supported either way.

## Untried (2)

- evaluate scores whichever model has the best CV, so the val read often does not test the experiment's change _(confidence: low)_
  - exp-004 changed HGB; evaluate scored RF (0.81896 vs HGB 0.81894). A per-experiment val read is only diagnostic of the lever if it scores the changed model, or the experiment pre-declares which model it is testing.
- explore is not experiment-aware: it charts all 712 working rows, including the open experiment's val rows _(confidence: low)_
  - Skipped explore in exp-004 for this reason. Harmless while features are fixed in code, but any feature chosen from explore output would have seen val rows.
