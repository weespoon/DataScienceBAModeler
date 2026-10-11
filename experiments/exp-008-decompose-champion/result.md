# exp-008 — result, scored against the locked predictions

`plan.md` is unchanged since its sha256 (`03bf15c6…`) was journaled before
scoring. Full table: `reports/05_bakeoff.md`.

## Per-arm ranges: 7 of 10, and the misses no longer lean one way

| arm | predicted | actual | |
|---|---|---:|---|
| previous_champion | −0.010…−0.025 | −0.0129 | ✓ |
| rf_wcg | −0.012…+0.002 | 0.0000 | ✓ |
| rf_wcg_maxfeat | −0.006…+0.003 | −0.0029 | ✓ |
| without_wcg_rate | −0.010…−0.002 | **+0.0029** | ✗ the all-member rate substitutes fully |
| without_group_rate | ±0.004 | +0.0018 | ✓ |
| without_titlefine_swap | ±0.004 | 0.0000 | ✓ |
| without_age_by_title | ±0.004 | **−0.0047** | ✗ |
| without_ticket_size | ±0.004 | +0.0006 | ✓ |
| without_max_features | −0.012…0 | **+0.0041** | ✗ (−0.0032 on fresh seeds) |

In exp-007 all nine misses ran the same way. Here there are three misses, going
both ways. That is what calibrated predictions look like.

## Event predictions: 4 of 4 resolved as the more likely outcome

| prediction | p | outcome |
|---|---:|---|
| replication drop > 0.008 vs exp-007's confirmed numbers | 35% | did not happen: −0.0034 (rf_kitchen), −0.0053 (rf_wcg) |
| an ablation is promoted | 15% | no; the champion holds |
| the winner is under 2 SE | 60% | yes, 1.70σ |
| val within ±0.06 of cv | 90% | yes, +0.042 |

## What was learned

1. **The gain replicates.** The champion beats the old RF by +0.0129 on rows
   the promotion never saw (2.4σ). exp-007's selection number (+0.0264) was
   about half winner's curse. Its fresh-seed confirmation (+0.0163) was within
   one SE of the replication.
2. **That answers exp-007's open question.** My one-sided misses there were
   mostly the winner's curse plus an underrated field. They were not a large
   config × split interaction: replicated deltas moved by 0.003–0.005, about
   one paired SE.
3. **The champion is rf_wcg with extras that do nothing measurable.** The old
   RF plus the WCG rate ties it exactly. Title-median Age is the one extra with
   a same-signed result in both experiments (exp-007 +0.0035 added,
   exp-008 −0.0047 removed). The rule correctly refused to prune the rest on
   sub-SE evidence, which is the exp-006 ratchet avoided.
4. **CV understates group models.** Coverage of labelled companions grows with
   training size (14.8% → 18.5% → 19.9% for 379/455/512 rows). So val > cv
   (+0.042) is the expected direction, and the final holdout read, fitted on
   712 rows, should come in above CV too. Pre-registering that now, so it
   cannot be rationalised afterwards: **the final holdout number is predicted
   to land at or above 0.83, and below ~0.88.**

## Done in this pass

- Submission from the champion: `reports/submission.csv`, 418 rows, predicted
  survival 0.364, fitted on 712 working rows. **Holdout untouched**, so the 179
  holdout passengers contribute no companion rates.
- Backlog: simplify the champion to the old RF + WCG + title-median Age
  (untried finding).
