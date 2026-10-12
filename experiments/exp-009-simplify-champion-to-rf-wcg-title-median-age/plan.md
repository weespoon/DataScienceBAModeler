# exp-009 — simplify the champion: plan and predictions, locked before scoring

Written 2026-10-12, after `make experiment-open` and before `make bakeoff
ARM_SET=simplify`. `make bakeoff` locks this file's sha256 before any arm is
scored. Code is committed (`34239fa`), so this experiment's rows should carry
a clean `git_sha`.

## Why

exp-008 took rf_kitchen apart. rf_wcg, which is the old RF plus the WCG rate
only, tied it at 0.0000 ± 0.0041. Of the five changes, only title-median Age
measurably carried weight: removing it cost −0.0047 ± 0.0022, at 2.1σ and not
confirmed. The other three (the TitleFine swap, ticket size and the
all-member group rate) were each within about 1 SE of zero. Under its rule,
exp-008 correctly refused to prune on that evidence. This experiment asks the
question directly: does the smallest config that should carry the gain match
the champion? The backlog item is "Simplify the champion to rf_wcg +
title-median Age".

## Arms (`python -m titanic_agent.bakeoff list --set simplify`)

- `champion`: rf_kitchen, the reference.
- `rf_simple`: rf_baseline12's 12 features and params (max_features sqrt),
  plus WCGSurvival/WCGKnown from the in-fold `group` step, plus `age_by_title`.
  There is no TitleFine, no TicketGroupSize, no all-member group rate and no
  max_features 0.5.

There is one challenger, so there is no best-of-N selection. Its selection
delta is a measurement, and the winner's-curse correction exp-007 needed does
not apply. Confirmation and nested CV still run, because the rule requires
them.

The permutation control runs first, because both arms read the label. It
borrows `rf_group`, `rf_wcg` and `rule_wcg` from the bake-off set, which is
the change in `34239fa`.

## Decision rule (unchanged, journaled by `bakeoff.run`)

`rf_simple` replaces the champion only if all three checks hold:
1. its selection delta is ≥ 0.003;
2. its delta on 5 fresh fold seeds is ≥ 0.003;
3. nested CV A − B > 0.

**A tie does not crown.** This experiment is about parsimony, and the rule I
declared cannot reward parsimony. I am not changing the rule after opening the
experiment. If the confirmation delta lands within ±0.003 (about 1 SE), the
verdict is **held**, and I will record a finding that the simpler config ties.
Swapping the champion for a simpler one that ties is left as a recommendation
for a human. It is not something this pass does.

If rf_simple loses by more than 0.003 on the confirmation, the verdict is
**held**, and the finding is that at least one of the three dropped changes
carries weight together with the others. That would contradict exp-008's
one-at-a-time reading.

## Validation read

Pre-declared: the val read scores **rf_simple's** bake-off row, whatever the
outcome. It is the change under test. Scoring the champion again would
re-measure a config that exp-007 and exp-008 have already val-read. It is a
tripwire with ~0.06 resolution, not a comparison.

The final holdout is not touched.

## Predictions

| quantity | prediction |
|---|---|
| rf_simple Δ vs champion, selection folds | −0.004 to +0.006, point +0.001 |
| rf_simple Δ vs champion, 5 fresh seeds | −0.004 to +0.005, point +0.001 |
| nested A − B | within ±0.006 |
| permutation control | at chance (all three arms ≤ majority + noise) |

- |selection Δ| < 2 SE (unresolvable tie): **~75%**
- rf_simple is crowned (all three checks): **~15%**
- rf_simple loses by more than 0.003 on fresh seeds: **~15%**
- rf_simple's val lands within ±0.06 of its cv: **~90%**
- rf_simple's val is above its cv. exp-008 found CV understates group models
  because group coverage grows with training size: **~65%**

What would surprise me: a confirmation delta below −0.006. That would mean the
three "dead" changes jointly carry about 3+ passengers, even though none is
measurable alone.
