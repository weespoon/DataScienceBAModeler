# 05 — bake-off (exp-007-bakeoff)

20 arms, each paired against the champion **rf_baseline12** on identical 5×3 repeated stratified folds of 569 training rows. 34s.

Champion cv on these folds: **0.8131**

**Read this first.** The best of 20 arms with no real effect shows about **+1.9 SE** from noise alone (independent-normal bound). A winner at or below that sigma is indistinguishable from the luckiest null. One passenger is 0.0018; the bar 0.003 is 1.7 passengers.

| arm | model | cv | Δ vs champion | SE | σ | passengers | s | hypothesis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `rf_kitchen` | random_forest | 0.8395 | +0.0264 | 0.0053 | +4.99 | +15.0 | 1 | Kitchen sink: TitleFine swap, title-imputed age, ticket size, both group rates, max_features 0.5. If it wins, the next experiment takes it apart. |
| `stack_group` | stack | 0.8342 | +0.0211 | 0.0054 | +3.90 | +12.0 | 9 | Stack the same four under a logistic meta-learner, which can learn to trust one member where the others disagree. |
| `vote_group` | soft_vote | 0.8325 | +0.0193 | 0.0052 | +3.75 | +11.0 | 2 | Soft vote of the four group-aware learners. |
| `rf_wcg` | random_forest | 0.8313 | +0.0182 | 0.0042 | +4.38 | +10.3 | 1 | Add woman-child-group survival only. Adult men in a group died whatever the group did; excluding them sharpens the signal for women and boys. |
| `svc_rich_wcg` | svc | 0.8307 | +0.0176 | 0.0063 | +2.81 | +10.0 | 0 | RBF SVC on the rich encoding plus WCG rates: a smooth boundary between the linear and tree arms, for ensemble diversity as much as on its own. |
| `rule_wcg` | rule_wcg | 0.8307 | +0.0176 | 0.0081 | +2.18 | +10.0 | 0 | Gender corrected by the woman-child group: the best-known hand rule for this ship. If no model beats it, the models are not learning much more. |
| `logit_rich_wcg` | logistic | 0.8289 | +0.0158 | 0.0054 | +2.95 | +9.0 | 0 | The same logistic with woman-child-group rates. |
| `et_group` | extra_trees | 0.8284 | +0.0152 | 0.0059 | +2.58 | +8.7 | 1 | ExtraTrees with group rates. Random thresholds average more smoothly over a 0-1 rate than greedy RF splits, which chase the exact LOO values. |
| `vote_base` | soft_vote | 0.8283 | +0.0152 | 0.0050 | +3.02 | +8.6 | 1 | Soft vote of champion, boosting and the rich logistic: three different inductive biases, none using group rates. |
| `gb_group` | gradient_boosting | 0.8260 | +0.0129 | 0.0063 | +2.03 | +7.3 | 1 | The same boosting with both group rates. |
| `rf_maxfeat_half` | random_forest | 0.8231 | +0.0100 | 0.0037 | +2.68 | +5.7 | 1 | max_features=0.5. With sqrt of ~25 one-hot columns, the strong features are rarely sampled per split; the untried dilution hypothesis. |
| `rf_group` | random_forest | 0.8231 | +0.0100 | 0.0034 | +2.92 | +5.7 | 1 | Add in-fold group survival (family, then ticket). Groups evacuated together, so a mate's fate is evidence about one's own location and choices. |
| `hgb_group` | hist_gradient_boosting | 0.8202 | +0.0070 | 0.0084 | +0.84 | +4.0 | 0 | Regularised histogram boosting with group rates: 20-row leaves and L2 1.0 keep it from splitting on a handful of LOO values. |
| `logit_rich` | logistic | 0.8195 | +0.0064 | 0.0054 | +1.19 | +3.6 | 0 | Logistic on bucketed/one-hot encodings and Sex x class. The untried finding: binning helps the model that cannot bin for itself. |
| `rf_ticket_size` | random_forest | 0.8172 | +0.0041 | 0.0030 | +1.39 | +2.3 | 1 | Add TicketGroupSize. FamilySize misses companions on one ticket who are not kin; group size had the same rise-then-collapse shape. |
| `rf_age_by_title` | random_forest | 0.8166 | +0.0035 | 0.0025 | +1.38 | +2.0 | 1 | Impute Age by title median in-fold. A missing-age Master is a boy, not a 28-year-old; median imputation erases the one thing Age tells us. |
| `rf_titlefine` | random_forest | 0.8155 | +0.0023 | 0.0013 | +1.74 | +1.3 | 1 | SWAP Title for TitleFine. 'Rare' mixes clergy (none survived) with nobility (all did); as an add it diluted, as a swap it should not. |
| `gb_baseline` | gradient_boosting | 0.8148 | +0.0017 | 0.0072 | +0.24 | +1.0 | 1 | Slow, subsampled boosting (300 x 0.05, depth 3, subsample 0.8). Earlier GB grids used full-sample fits only, which boost noise on 569 rows. |
| `champion` | random_forest | 0.8131 | +0.0000 | 0.0000 | +0.00 | +0.0 | 3 | The reference. Re-scored here so every delta is paired. |
| `rf_shallow` | random_forest | 0.8119 | -0.0012 | 0.0032 | -0.37 | -0.7 | 1 | max_depth=7. Fully grown trees on 569 rows memorise singletons; a cap trades a little bias for variance. |
| `rule_gender` | rule_gender | 0.7803 | -0.0328 | 0.0069 | -4.73 | -18.7 | 0 | Women live, men die. Not a contender: the floor, so 'better than the champion' can be read against 'better than nothing'. |

## Winner and confirmation

- Winner: **`rf_kitchen`**, selected from 20.
- Fresh fold seeds: rf_kitchen vs champion, 5 fresh fold seeds: +0.0163 +/- 0.0074 (seed sd, +2.20 sigma; +9.3 passengers).
  These re-draw the folds but not the rows, so they remove fold luck, not row luck.

## Nested: is picking the best arm worth anything?

| outer fold | test rows | picked | A (picked) | B (champion) | A − B |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 114 | `rule_wcg` | 0.8509 | 0.8246 | +0.0263 |
| 2 | 114 | `rf_wcg` | 0.8246 | 0.7982 | +0.0263 |
| 3 | 114 | `rf_kitchen` | 0.9211 | 0.8684 | +0.0526 |
| 4 | 114 | `vote_base` | 0.7368 | 0.7281 | +0.0088 |
| 5 | 113 | `logit_rich_wcg` | 0.8319 | 0.8230 | +0.0088 |

Procedure vs champion: **+0.0246 ± 0.0080** (+3.06σ), 111s. Outer test folds hold ~113 rows, so this is coarse; its job is the sign, net of selection.

## Decision (rule declared before any arm ran)

- PASS selection delta rf_kitchen vs champion: +0.0264 +/- 0.0053 (naive SE, +4.99 sigma; +15.0 passengers) vs bar 0.003
- PASS confirmation rf_kitchen vs champion, 5 fresh fold seeds: +0.0163 +/- 0.0074 (seed sd, +2.20 sigma; +9.3 passengers) vs bar 0.003
- PASS nested A-B +0.0246 +/- 0.0080 vs 0

**PROMOTE.**
