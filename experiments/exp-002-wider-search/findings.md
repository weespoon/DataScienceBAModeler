# Findings

Claims with evidence, from experiment `exp-002-wider-search`. Rendered from the append-only
`experiments/findings.jsonl` — edit that through `findings.record()`,
never this file.

## Confirmed (2)

- select_members crossed experiment boundaries, contaminating the val score _(confidence: high)_ — evidence: `20261010T192716-de7e74`
  - It filtered on frozen_sha256 but not experiment_id, so exp-002 ensembled members tuned by the pre-container baseline on all 712 rows -- including exp-002's own 143 validation rows. Its 0.8392 was a selection statistic, not a measurement. Fixed by scoping select_members to the open experiment; exp-002 abandoned because its val split is spent and re-reading it would be the same error again.
- Ensemble members must be scoped to the current experiment, not just the frozen dataset _(confidence: high)_ — evidence: `20261010T192716-de7e74`
  - Two experiments share a frozen_sha256 by construction, so that filter alone never separates them.
