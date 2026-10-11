# Findings

Claims with evidence, from experiment `exp-008-decompose-champion`. Rendered from the append-only
`experiments/findings.jsonl` — edit that through `findings.record()`,
never this file.

## Confirmed (3)

- **+0.0129** The group-feature gain replicates on new rows, smaller: the champion beats the old RF by +0.0129 here vs +0.0264 (selection) / +0.0163 (fresh seeds) in exp-007 _(confidence: medium)_ — evidence: `20261011T193600-c52f68`, `20261011T193609-4b63b8`, `20261011T191756-8c25a0`, `20261011T191821-b81d88`
  - Paired, on exp-008's 569 rows, previous_champion -0.0129 +/- 0.0053 (2.4 sigma). The first cross-experiment replication through the champion: identical configs, two different row draws. rf_wcg vs the old RF: +0.0182 in exp-007 -> +0.0129 here (-0.0053, about one paired SE). rf_kitchen: -0.0135 vs its selection delta, -0.0034 vs its fresh-seed confirmation. So most of exp-007's apparent 'champion was weak on that split' is the winner's curse, not a config x split interaction; the interaction, if any, is within ~0.005.
- **+0.0000** Of rf_kitchen's five changes over rf_wcg, none is measurably worth anything once the WCG rate is present: rf_wcg ties the champion exactly _(confidence: medium)_ — evidence: `20261011T193601-b87bc0`, `20261011T193603-805cbd`, `20261011T193605-d9b78c`, `20261011T193606-f11a23`, `20261011T193607-00dba1`, `20261011T193609-000de2`, `20261011T193618-2b98fe`, `20261011T193724-6e9caa`
  - rf_wcg (old RF + WCG rate only) 0.0000 +/- 0.0041 vs champion. Removing each change: titlefine_swap +0.0000, ticket_size +0.0006, group_rate +0.0018, wcg_rate +0.0029 (the all-member rate substitutes), max_features +0.0041 -> -0.0032 on fresh seeds. Nested pick-best +0.0018 +/- 0.0018. Champion held under the declared rule. Exception: age_by_title, below.
- **+0.0416** CV systematically understates group-feature models, because group-rate coverage grows with training size; val > cv is the expected direction, not a leak signal _(confidence: medium)_ — evidence: `20261011T193734-46f101`, `20261011T191741-5aa054`
  - Share of held-out rows with a labelled woman/child mate, exp-008 train rows only: 0.148 fitting on ~379 rows, 0.185 on ~455 (5-fold CV), 0.199 on ~512 -- so the 569-row val fit and the 712-row submission fit see more. Val 0.8741 vs cv 0.8325 (+0.0416, inside the +/-0.056 band) is the largest val yet and above the ~0.85 line; exp-007 was +0.0136. Leak ruled out separately by exp-007's permutation control. Consequence: any comparison of a group model against a non-group model at the CV training size is biased against the group model, and the final holdout number (fit on 712) should be expected above CV.

## Refuted (1)

- **-0.0032** RF max_features=0.5 does not help (refutes exp-007's unconfirmed +0.0100) _(confidence: medium)_ — evidence: `20261011T193609-000de2`, `20261011T193618-2b98fe`, `20261011T193602-1bc461`, `20261011T191751-04cf87`
  - Removing it from the champion: +0.0041 on the selection folds, -0.0032 +/- 0.0046 on 5 fresh seeds. Adding it to rf_wcg: -0.0029. The exp-007 +0.0100 on the baseline features was a best-of-20 that did not replicate in either direction.

## Inconclusive (1)

- **+0.0047** Title-median Age imputation carries its weight inside the champion _(confidence: low)_ — evidence: `20261011T193604-efe7d9`
  - Removing it costs -0.0047 +/- 0.0022 (2.1 sigma, 2.7 passengers), the only ablation that clearly hurt. In exp-007 it was +0.0035 (1.4 sigma) added alone to the old RF. Two same-signed results on different rows; not confirmed on fresh seeds.

## Untried (1)

- Simplify the champion to rf_wcg + title-median Age _(confidence: low)_
  - exp-008 says rf_kitchen's other changes are dead weight (each within ~1 SE of zero) but its rule correctly refused to prune on that evidence. A parsimony arm -- previous RF + WCG rate + age_by_title -- paired against rf_kitchen with confirmation would settle it. Simpler config, same expected accuracy, fewer ways to overfit.
