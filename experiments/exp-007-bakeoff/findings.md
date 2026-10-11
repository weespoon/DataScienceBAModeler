# Findings

Claims with evidence, from experiment `exp-007-bakeoff`. Rendered from the append-only
`experiments/findings.jsonl` — edit that through `findings.record()`,
never this file.

## Confirmed (4)

- **+0.0182** In-fold woman-child-group (WCG) survival is the largest lever found so far: adding it alone to the champion RF gains +0.0182 _(confidence: medium)_ — evidence: `20261011T191755-7f475c`, `20261011T191749-975d5b`, `20261011T192021-a7a453`, `20261011T191741-5aa054`
  - rf_wcg +0.0182 +/- 0.0042 (4.4 sigma, 10 passengers), a single-lever change, well above the ~1.9 sigma best-of-20 null. Every WCG-using arm landed +0.013 to +0.026. Nested CV of pick-the-best: +0.0246 +/- 0.0080 (3.1 sigma), with 4 of 5 outer folds picking a WCG arm. Permutation control (labels shuffled) scored 0.59-0.61 vs majority 0.617, so the in-fold leave-one-out construction does not leak labels across folds. No external data: groups from surname+fare and ticket in the frozen rows.
- **+0.0176** A hand rule (gender, corrected by the woman-child group) with nothing learned ties the best learned single models _(confidence: medium)_ — evidence: `20261011T191800-0fec50`, `20261011T191755-7f475c`, `20261011T191759-8de651`
  - rule_wcg +0.0176 +/- 0.0081 vs champion; rf_wcg +0.0182, svc_rich_wcg +0.0176. The learned models' whole gain over the champion is roughly the WCG signal itself; the rest of the 12 features add ~0.001 on top of it. Implication: further model tuning on top of WCG is chasing ~1-2 passengers.
- **-0.0101** The best-of-20 selection delta regresses by ~0.010 on fresh fold seeds (winner's curse, measured) _(confidence: medium)_ — evidence: `20261011T191756-8c25a0`, `20261011T191821-b81d88`
  - rf_kitchen +0.0264 on the selection folds -> +0.0163 +/- 0.0074 (seed sd) on 5 fresh fold seeds. The seed sd (0.0074) is ~1.4x the naive SE (0.0053), above the 0.77-1.37x range measured in exp-006 -- plausibly because the group rates make fold composition matter more. NB champion.jsonl's exp-007 entry records delta_vs_previous=+0.0264, the biased selection delta; the honest number is the +0.0163 confirmation. That line is append-only; bakeoff._apply now records the confirmation delta.
- **+0.0136** Validation 0.8531 for rf_kitchen is above the ~0.85 'suspect' line but is explained, not a leak signal _(confidence: medium)_ — evidence: `20261011T192034-80371b`, `20261011T191741-5aa054`, `20261011T191756-8c25a0`
  - cv 0.8395 -> val 0.8531, gap +0.0136 inside the +/-0.059 2 SE band on 143 rows. Permutation control at chance; make leakage clean; val was pre-declared in plan.md. 0.853 on 143 rows is 122 passengers; one SE is 0.030, so 'true accuracy ~0.82-0.84' is entirely consistent. The 0.92 on nested outer fold 3 (114 rows) is an easy fold: the champion scored 0.868 on it.

## Refuted (1)

- **+0.0017** Capped-depth RF (7) and slow subsampled GB (300 x 0.05, subsample 0.8) do not beat the champion _(confidence: medium)_ — evidence: `20261011T191752-dee1bd`, `20261011T191758-efe5c6`
  - rf_shallow -0.0012 +/- 0.0032; gb_baseline +0.0017 +/- 0.0072 (0.24 sigma).

## Inconclusive (3)

- **+0.0100** RF max_features=0.5 beats the default sqrt on the 12 baseline features (the dilution hypothesis) _(confidence: low)_ — evidence: `20261011T191751-04cf87`, `20261011T191749-975d5b`
  - +0.0100 +/- 0.0037 (2.7 sigma), single lever. Above the 1.9 sigma best-of-20 null but not confirmed on fresh seeds. Supports the exp-006 untried finding that add-one tests were biased against features by max_features='sqrt'. Worth one paired confirmation.
- **+0.0152** A soft vote of champion RF + subsampled GB + bucketed logistic beats the champion RF alone, without group features _(confidence: low)_ — evidence: `20261011T191801-f29ed2`, `20261011T191749-975d5b`
  - +0.0152 +/- 0.0050 (3.0 sigma). Contradicts exp-001's 'ensemble does not beat best single', but that was unpaired and used different members. Not confirmed on fresh seeds; selected from 20.
- **+0.0041** TitleFine swap, title-median Age imputation and ticket group size each move the RF by under 2 SE _(confidence: low)_ — evidence: `20261011T191750-3f10dc`, `20261011T191753-c47082`, `20261011T191754-e2d6df`
  - titlefine swap +0.0023 +/- 0.0013 (1.7 sigma), age_by_title +0.0035 +/- 0.0025 (1.4), ticket_size +0.0041 +/- 0.0030 (1.4). All positive, none resolvable one at a time; the swap (vs exp-006's add at +0.0012) is consistent with the dilution story.

## Untried (2)

- Decompose rf_kitchen: which of its five changes (TitleFine swap, title-median Age, ticket size, both group rates, max_features 0.5) carry its gain over rf_wcg _(confidence: low)_
  - Kitchen sink won the selection by +0.0082 over rf_wcg, about 1.5 SE, which is not resolvable. Next experiment: paired arms rf_wcg + each change, and rf_wcg+max_features 0.5 alone, against the new champion. Expect most of the gap to be max_features and noise.
- Group/WCG rates for the Kaggle test set should be fitted on all labelled rows, with test passengers' companions counted _(confidence: low)_
  - Engineer already does this when fitted on the full working set: test rows are 'unseen', get rates from labelled mates. Not yet exercised because the submission path (ensemble.py) builds from tune rows, not bake-off configs. Needed before the champion can produce a submission.
