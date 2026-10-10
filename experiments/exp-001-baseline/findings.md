# Findings

Claims with evidence, from experiment `exp-001-baseline`. Rendered from the append-only
`experiments/findings.jsonl` — edit that through `findings.record()`,
never this file.

## Confirmed (3)

- **+0.2121** random_forest is the strongest single model (0.8287 CV) _(confidence: medium)_ — evidence: `20261010T174124-7b0b5d`
  - Measured on 3 identical passes; deterministic given the frozen data and fixed seed.
- **-0.0014** Soft-vote ensemble does NOT beat the best single model _(confidence: medium)_ — evidence: `20261010T174242-12c95a`
  - 0.8273 vs 0.8287. Well inside the +/-0.015-0.030 fold spread, so 'tied' is the honest reading, not 'worse'.
- **-0.0408** knn is clearly the weakest model (0.7879) and is a poor ensemble member _(confidence: medium)_ — evidence: `20261010T184639-f6bea1`

## Untried (3)

- logistic leaves 9 hyperparameters at their defaults; only C is tuned _(confidence: low)_ — source: introspection
  - make introspect --show-untuned lists them. Widening this search is the cheapest untried lever.
- hist_gradient_boosting leaves 16 parameters at defaults; only 2 are tuned _(confidence: low)_ — source: introspection
  - Largest untuned surface of any model here.
- No model has been tried with engineered features beyond the 12 current ones _(confidence: low)_
  - explore proposed Title, Deck, FamilySize, IsAlone, AgeMissing; all are in use. Nothing newer has been proposed.
