# Findings

Claims with evidence, from experiment `exp-006-feature-selection`. Rendered from the append-only
`experiments/findings.jsonl` — edit that through `findings.record()`,
never this file.

## Confirmed (1)

- **+0.0152** Title, Pclass, FamilySize and Embarked carry most of the signal for random_forest; Title is the single most valuable feature _(confidence: medium)_ — evidence: `20261010T203637-707773`, `20261010T203628-a15c91`, `20261010T203635-dd5b3e`, `20261010T203634-903135`
  - Removal costs: Title 0.0152 +/- 0.0052 (2.9 SE), FamilySize 0.0070 +/- 0.0025 (2.8 SE), Pclass 0.0094 +/- 0.0049, Embarked 0.0082 +/- 0.0047, Fare 0.0058, Sex 0.0053 +/- 0.0048 (Sex is largely redundant with Title).

## Refuted (1)

- **+0.0012** None of 24 row-wise engineered features (age/fare bins, one-hot Pclass, Sex x class, ticket/cabin/name parsing, family flags) improves random_forest by >= 0.003 when added to the 12 baseline features _(confidence: medium)_ — evidence: `20261010T203602-d38cdc`, `20261010T203603-9a8b80`, `20261010T203604-341cc3`, `20261010T203605-1118ab`, `20261010T203606-0d1760`, `20261010T203607-89ea31`, `20261010T203608-684599`, `20261010T203609-71d05e`, `20261010T203610-7d3b96`, `20261010T203611-376bd7`, `20261010T203612-65e7bc`, `20261010T203613-76b478`, `20261010T203614-0f2f12`, `20261010T203615-40627f`, `20261010T203616-2663a8`, `20261010T203617-186695`, `20261010T203618-c5c2e6`, `20261010T203619-a1e61c`, `20261010T203620-edc442`, `20261010T203621-bf9404`, `20261010T203622-2b597b`, `20261010T203623-e3a00b`, `20261010T203624-18faa6`, `20261010T203625-063fd4`, `20261010T203626-fde84b`
  - Best add: TitleFine +0.0012 +/- 0.0030 SE; 19 of 24 adds were negative, worst TicketPrefix and CabinSide -0.0064. Paired 5x3 CV on identical folds. Refutes 'more features help this RF', not 'these features carry no signal' -- see the redundancy finding.

## Inconclusive (1)

- **+0.0041** Pruning the RF to 6 features (Sex, Pclass, Fare, Embarked, FamilySize, Title) improves on the 12-feature baseline _(confidence: low)_ — evidence: `20261010T203602-d38cdc`, `20261010T203640-986301`, `20261010T203704-9773c4`
  - CV 0.8360 -> 0.8401 (+0.0041), but each of the 6 drops (Age, SibSp, Parch, IsAlone, Deck, AgeMissing) was within ~2 SE and accepted greedily, so the gain is partly selection on noise. Val 0.8112 on 143 rows (SE ~0.033) is 0.029 below CV -- first experiment with val < CV -- and cannot distinguish the pruned set from the baseline, which was not val-scored.

## Untried (3)

- Add-one tests against a frozen random_forest are biased against redundant encodings: with max_features='sqrt' fixed, an extra correlated column dilutes how often the strong ones are sampled per split _(confidence: low)_
  - Would explain 19/24 negative adds and why dropping redundant Age/SibSp/Parch helped. Test: re-run the best candidates (TitleFine, AgeClass, WomanOrChild) as SWAPS for the column they refine, or with max_features re-tuned per feature set.
- Encoding and bucketing (one-hot Pclass, AgeBin, FareBin, interactions) should be tested on logistic / SVC, not a tree model _(confidence: low)_
  - Trees are invariant to monotone transforms and split ordinals themselves, so this experiment tested bucketing and one-hot on the model least able to benefit. Linear models cannot express a non-monotone age effect without bins.
- Features that need fitted state were not tested: Age imputed by Title median, ticket-group size, within-fold family/ticket survival rates _(confidence: low)_
  - All require fitting inside each CV fold (a custom transformer in the Pipeline). Group survival rates are target-derived and leak across folds if computed outside them; Rule 4 is not at issue because nothing external is joined.
