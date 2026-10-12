# 05 — bake-off (exp-009-simplify-champion-to-rf-wcg-title-median-age, arm set `simplify`)

1 arms, each paired against the champion **rf_kitchen** on identical 5×3 repeated stratified folds of 569 training rows. 13s.

Champion cv on these folds: **0.8289**

**Read this first.** The best of 1 arms with no real effect shows about **+0.0 SE** from noise alone (independent-normal bound). A winner at or below that sigma is indistinguishable from the luckiest null. One passenger is 0.0018; the bar 0.003 is 1.7 passengers.

| arm | model | cv | Δ vs champion | SE | σ | passengers | s | hypothesis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `rf_simple` | random_forest | 0.8313 | +0.0023 | 0.0047 | +0.49 | +1.3 | 1 | Previous champion + WCG rate + title-median Age: rf_kitchen without the three changes exp-008 could not measure. Expected to tie. |
| `champion` | random_forest | 0.8289 | +0.0000 | 0.0000 | +0.00 | +0.0 | 1 | The reigning champion, re-scored as reference. |

## Winner and confirmation

- Winner: **`rf_simple`**, selected from 1.
- Fresh fold seeds: rf_simple vs champion, 5 fresh fold seeds: +0.0068 +/- 0.0046 (naive SE; seed sd 0.0033 over 5, +1.46 sigma; +3.9 passengers).
  These re-draw the folds but not the rows, so they remove fold luck, not row luck.

## Nested: is picking the best arm worth anything?

| outer fold | test rows | picked | A (picked) | B (champion) | A − B |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 114 | `rf_simple` | 0.8772 | 0.8947 | -0.0175 |
| 2 | 114 | `rf_simple` | 0.7895 | 0.7982 | -0.0088 |
| 3 | 114 | `rf_simple` | 0.8246 | 0.8246 | +0.0000 |
| 4 | 114 | `rf_simple` | 0.8596 | 0.8421 | +0.0175 |
| 5 | 113 | `champion` | 0.8230 | 0.8230 | +0.0000 |

Procedure vs champion: **-0.0018 ± 0.0058** (-0.30σ), 15s. Outer test folds hold ~113 rows, so this is coarse; its job is the sign, net of selection.

## Decision (rule declared before any arm ran)

- FAIL selection delta rf_simple vs champion: +0.0023 +/- 0.0047 (naive SE, +0.49 sigma; +1.3 passengers) vs bar 0.003
- PASS confirmation rf_simple vs champion, 5 fresh fold seeds: +0.0068 +/- 0.0046 (naive SE; seed sd 0.0033 over 5, +1.46 sigma; +3.9 passengers) vs bar 0.003
- FAIL nested A-B -0.0018 +/- 0.0058 vs 0

**CHAMPION HOLDS.**
