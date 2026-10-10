# Findings

Claims with evidence, from experiment `exp-003-wider-search`. Rendered from the append-only
`experiments/findings.jsonl` — edit that through `findings.record()`,
never this file.

## Confirmed (2)

- Validation 0.8462 vs CV 0.8190 on 143 rows -- the gap is within noise, not evidence of improvement _(confidence: low)_ — evidence: `see evaluate row`
  - 143 rows carries roughly +/-4%. A 0.027 gap is well inside that; reading it as a real gain would be exactly the mistake these splits exist to prevent.
- Training on 569 rows instead of 712 costs roughly 0.006-0.010 CV _(confidence: medium)_ — evidence: `exp-003 tune rows`
  - Best single fell 0.8287 (712 rows, exp-001) -> 0.8225 (569 rows, exp-002/3). That is the price of a per-experiment val split, and it is worth paying.
