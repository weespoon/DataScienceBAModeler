# 05 — bake-off (exp-008-decompose-champion, arm set `decompose`)

9 arms, each paired against the champion **rf_kitchen** on identical 5×3 repeated stratified folds of 569 training rows. 21s.

Champion cv on these folds: **0.8284**

**Read this first.** The best of 9 arms with no real effect shows about **+1.5 SE** from noise alone (independent-normal bound). A winner at or below that sigma is indistinguishable from the luckiest null. One passenger is 0.0018; the bar 0.003 is 1.7 passengers.

| arm | model | cv | Δ vs champion | SE | σ | passengers | s | hypothesis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `without_max_features` | random_forest | 0.8325 | +0.0041 | 0.0024 | +1.70 | +2.3 | 1 | Default max_features (sqrt) instead of 0.5. exp-007's +0.0100 for 0.5 on the baseline features was unconfirmed; this asks whether it survives here. |
| `without_wcg_rate` | random_forest | 0.8313 | +0.0029 | 0.0038 | +0.76 | +1.7 | 1 | Drop the WCG rate, keep the all-member rate. The arm that tells us whether WCG is the lever or merely one encoding of it. |
| `without_group_rate` | random_forest | 0.8301 | +0.0018 | 0.0037 | +0.48 | +1.0 | 1 | Drop the all-member group rate, keep WCG. Adult men's fates may be noise the WCG rate already filtered out. |
| `without_ticket_size` | random_forest | 0.8290 | +0.0006 | 0.0025 | +0.24 | +0.3 | 1 | Drop TicketGroupSize. Group rates already know who shares a ticket; the size may add nothing to them. |
| `without_titlefine_swap` | random_forest | 0.8284 | +0.0000 | 0.0026 | +0.00 | +0.0 | 1 | Title back in place of TitleFine. Tests whether the finer titles earn anything once group rates already separate the families that 'Rare' mixed. |
| `rf_wcg` | random_forest | 0.8284 | +0.0000 | 0.0041 | +0.00 | +0.0 | 1 | exp-007's single-lever arm, unchanged: previous champion + WCG rate. Replicated so its +0.0182 can be checked on new rows. |
| `champion` | random_forest | 0.8284 | +0.0000 | 0.0000 | +0.00 | +0.0 | 3 | The reigning champion, re-scored as reference. |
| `rf_wcg_maxfeat` | random_forest | 0.8254 | -0.0029 | 0.0043 | -0.67 | -1.7 | 1 | rf_wcg plus max_features 0.5: the guess that these two changes carry all of the champion's gain and the other three are noise. |
| `without_age_by_title` | random_forest | 0.8237 | -0.0047 | 0.0022 | -2.08 | -2.7 | 1 | Median Age instead of title-median Age. Boys with missing ages may already be caught by Title and the WCG rate, which would make the imputation redundant. |
| `previous_champion` | random_forest | 0.8155 | -0.0129 | 0.0053 | -2.41 | -7.3 | 1 | The champion this one replaced. Replicates the promotion delta on rows the promotion never saw: the config x split interaction, measured. |

## Replication: earlier deltas, re-measured on these rows

Each row is one paired difference measured twice, in two experiments, on different training rows. Split drift cancels inside each; what is left is noise (paired SE ~0.003-0.008) plus the config × split interaction.

| arm | vs | earlier | Δ there | Δ here | change |
| --- | --- | --- | ---: | ---: | ---: |
| `rf_wcg` | `previous_champion` | exp-007-bakeoff | +0.0182 | +0.0129 | -0.0053 |
| `champion` | `previous_champion` | exp-007-bakeoff | +0.0264 | +0.0129 | -0.0135 |

## Winner and confirmation

- Winner: **`without_max_features`**, selected from 9.
- Fresh fold seeds: without_max_features vs champion, 5 fresh fold seeds: -0.0032 +/- 0.0046 (seed sd, -0.69 sigma; -1.8 passengers).
  These re-draw the folds but not the rows, so they remove fold luck, not row luck.

## Nested: is picking the best arm worth anything?

| outer fold | test rows | picked | A (picked) | B (champion) | A − B |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 114 | `without_ticket_size` | 0.8596 | 0.8596 | +0.0000 |
| 2 | 114 | `without_titlefine_swap` | 0.7982 | 0.7982 | +0.0000 |
| 3 | 114 | `champion` | 0.8158 | 0.8158 | +0.0000 |
| 4 | 114 | `without_max_features` | 0.8070 | 0.8070 | +0.0000 |
| 5 | 113 | `without_max_features` | 0.8407 | 0.8319 | +0.0088 |

Procedure vs champion: **+0.0018 ± 0.0018** (+1.00σ), 49s. Outer test folds hold ~113 rows, so this is coarse; its job is the sign, net of selection.

## Decision (rule declared before any arm ran)

- PASS selection delta without_max_features vs champion: +0.0041 +/- 0.0024 (naive SE, +1.70 sigma; +2.3 passengers) vs bar 0.003
- FAIL confirmation without_max_features vs champion, 5 fresh fold seeds: -0.0032 +/- 0.0046 (seed sd, -0.69 sigma; -1.8 passengers) vs bar 0.003
- PASS nested A-B +0.0018 +/- 0.0018 vs 0

**CHAMPION HOLDS.**
