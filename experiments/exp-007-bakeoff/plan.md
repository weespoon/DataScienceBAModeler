# exp-007 — bake-off: plan and predictions, locked before any arm is scored

Written 2026-10-11 after `make experiment-open` and before `make bakeoff`. The
journal entry recording this file's sha256 is the timestamp that locks it.

## What is different about this experiment

The brief asked for breadth: many hunches at once, the "art" half of tuning.
Every earlier experiment pulled one lever. This one pulls twenty, so
**attribution and selection are both worse by design**, and the protocol
compensates for that rather than ignoring it.

**Cross-experiment support** is new infrastructure, not just this run. A persisted
champion (`experiments/champion.jsonl`, `make champion`) is re-scored on this
experiment's own folds beside every arm. Each ledger row carries
`delta_vs_champion`. That paired delta, not the absolute cv, is what
`EXPERIMENTS.md` now compares across experiments. This is the "top open item"
in `docs/MEASUREMENT.md`.

**Champion (best of breed through exp-006):** random_forest with
n_estimators=400, min_samples_leaf=3 and max_depth=None, on the 12 baseline
features. Nested CV in exp-006 preferred the baseline features to the searched
6. Its params were tuned on exp-006's rows, which overlap this experiment's
validation rows. As a reference arm in CV that can only flatter it, so the
challengers' deltas are conservative.

## Arms

There are 20 challengers; `make bakeoff` with no arguments, or
`python -m titanic_agent.bakeoff list`, prints them. In groups:

- **Untried backlog items:** the TitleFine *swap*, RF max_features=0.5 (the
  dilution hypothesis), and bucketed encodings on logistic and SVC rather than
  on a tree.
- **Fitted-state features, all fitted in-fold:** Age imputed by title median,
  ticket group size, travel-group survival and woman-child-group survival. The
  group rates are leave-one-out on the training rows.
- **Model hunches:** a capped-depth RF, slow subsampled GB, regularised HGB, and
  ExtraTrees.
- **Hand rules:** the gender rule (the floor) and the WCG rule.
- **Ensembles:** a soft vote without group features, a soft vote with them, and
  a stack with them.

## Decision rule (also journaled by `bakeoff.run` before scoring)

The winner is the arm with the largest delta_vs_champion. It replaces the
champion only if **all three** hold:

1. its selection delta is ≥ 0.003;
2. its delta on 5 fresh fold seeds is ≥ 0.003;
3. nested CV of the whole pick-the-best procedure (5 outer folds) beats the
   champion: A − B > 0.

The best of 20 nulls shows about +1.9 SE ≈ +0.006 from noise. A winner below
that is reported as indistinguishable from luck, whatever the rule says.

The **validation read** scores the winner, whatever it is, and is
pre-declared here. It is a tripwire. The group features read the label, and
that is the classic Titanic leak. If val exceeds the winner's cv by more than
~0.06, stop and look for a leak before reading anything else.

Before the bake-off runs, a **permutation control** scores rf_group on shuffled
labels. If the in-fold transforms are honest it scores at chance, about 0.62,
the majority rate. Anything clearly above chance means the group feature is
leaking labels across folds, and the experiment stops.

## Predictions (my priors, recorded so they can be wrong)

| arm | predicted Δ vs champion |
|---|---|
| rf_wcg, rf_group, gb_group, et_group | +0.005 to +0.020; rf_wcg ≈ +0.010 |
| vote_group, stack_group | about the best group member, ±0.005 |
| rule_wcg | −0.015 to +0.005 |
| rule_gender | −0.04 to −0.06 |
| rf_titlefine, rf_maxfeat_half, rf_shallow, rf_age_by_title, rf_ticket_size | within ±0.004 (under 2 SE) |
| gb_baseline, vote_base | within ±0.006 |
| logit_rich | −0.012 to 0 |
| hgb_group, svc_rich_wcg, logit_rich_wcg | 0 to +0.012 |

- The winner is a group-aware arm: **~75%**.
- The confirmation keeps it ≥ 0.003: **~65%**.
- Nested A − B > 0: **~65%**.
- Promotion overall: **~50%**.
- Val for the winner lands within ±0.06 of its cv: **~90%**. A miss above that
  band is a leak signal, not good news.
