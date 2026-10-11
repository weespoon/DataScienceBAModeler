# exp-008 — take the champion apart: plan and predictions, locked before scoring

Written 2026-10-11, after `make experiment-open` and before `make bakeoff
ARM_SET=decompose`. Its sha256 goes in the journal before any arm is scored.
Code is committed (`bbef8fc`), so this experiment's rows carry a clean
`git_sha`.

## Why

exp-007 crowned `rf_kitchen`, which bundles five changes over the old champion:
the TitleFine swap, title-median Age, ticket group size, two group-survival
rates, and max_features 0.5. Only one of those was ever measured alone (the WCG
rate, +0.0182). It is a kitchen sink, and exp-007 promised to take it apart.

It also follows up on exp-007's prediction pattern. All nine of my misses there
went the same way, which suggests the old champion was weak *on that split*.
Paired deltas cancel split drift, but not a fixed config's interaction with one
particular draw.

## Arms (`python -m titanic_agent.bakeoff list --set decompose`)

- `champion`: rf_kitchen, the reference.
- `without_<x>`, one per change, each undone alone: titlefine_swap,
  age_by_title, ticket_size, group_rate, wcg_rate, max_features. A negative
  delta is what that change is worth *in the presence of the others*. That is
  not its stand-alone value, and the report should not read it as one.
- `previous_champion`: rf_baseline12, unchanged.
- `rf_wcg`: exp-007's single-lever arm, unchanged.
- `rf_wcg_maxfeat`: rf_wcg plus max_features 0.5. This tests the guess that two
  changes carry everything.

**Replication.** Because `previous_champion` is exp-007's reference config,
`replication()` re-measures every exp-007 arm that runs again here, as a paired
delta over that same config, on different rows. Those arms are `rf_kitchen`
(now `champion`) and `rf_wcg`. The difference between the two experiments is
noise plus the config × split interaction. That is the cross-experiment
measurement exp-007's result.md asked for.

## Decision rule (journaled by `bakeoff.run` too)

This is the same three-check rule as exp-007, unchanged on purpose. An
ablation is crowned only if removing a change makes the champion *better*,
and it must pass all three checks:

1. selection delta ≥ 0.003;
2. ≥ 0.003 on 5 fresh fold seeds;
3. nested CV of pick-the-best beats the champion.

No greedy pruning. exp-006 showed that dropping whatever "costs less than the
bar" ratchets: three of its six accepted drops cost accuracy. A change that
costs between 0 and 0.003 stays.

Validation read: the winner, pre-declared, as in exp-007. It is a tripwire, ±0.06.

The final holdout is not touched.

## Predictions

| arm | predicted Δ vs champion |
|---|---|
| previous_champion | −0.010 to −0.025 (point −0.016) |
| rf_wcg | −0.012 to +0.002 |
| rf_wcg_maxfeat | −0.006 to +0.003 |
| without_wcg_rate | −0.010 to −0.002 (the all-member rate partly substitutes) |
| without_group_rate | −0.004 to +0.004 |
| without_titlefine_swap / age_by_title / ticket_size | each within ±0.004 |
| without_max_features | −0.012 to 0 (point −0.005) |

- **Replication:** rf_kitchen over rf_baseline12 is +0.0163 (fresh seeds) in
  exp-007. Here I predict +0.010 to +0.025. rf_wcg over rf_baseline12 is
  +0.0182 in exp-007; here, +0.006 to +0.020.
- A *drop* of more than 0.008 from exp-007's number on either replication
  would support the split-interaction story: **~35%**.
- An ablation gets promoted: **~15%**.
- The winner is an ablation whose delta is under 2 SE, i.e. no change is
  clearly dead weight: **~60%**.
- Val for the winner lands within ±0.06 of its cv: **~90%**.
