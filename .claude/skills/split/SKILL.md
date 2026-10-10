---
name: split
description: Step 3 of the Titanic pipeline. Draw a seeded stratified train/holdout partition and persist it as data/frozen/split_plan.json. Use after review and before freeze.
---

# Step 3 — split

Decide, once, which rows are holdout rows. This step only *proposes* the
partition — `freeze` is what seals it. That separation is deliberate: a bad
split can be redrawn cheaply right up until the moment it is frozen, and never
afterwards.

## Inputs

- `data/raw/train.csv`.
- Parameters: `holdout_frac` (default 0.2), `stratify_on` (`Survived`),
  `seed` (`paths.SEED`).

## Outputs

- `data/frozen/split_plan.json` — the seed, the fractions, the stratification
  column, and the explicit `PassengerId` lists for train and holdout.

Ids, not row offsets. Offsets break the moment anything is re-sorted.

## How to run

```bash
make split              # uses paths.SEED
make split SEED=1234    # override
```

## Done criteria

- [ ] `data/frozen/split_plan.json` exists.
- [ ] The two id lists are **disjoint**.
- [ ] Their union is exactly the set of `PassengerId` in `data/raw/train.csv`.
- [ ] Holdout size is within one row of `round(n * holdout_frac)`.
- [ ] `Survived` rate in train and holdout differ by less than 0.02.
- [ ] `split.validate(plan)` returns `True`.

## Notes

- Refuse to run if `freeze.is_frozen()` — the holdout is already sealed and
  redrawing it would silently invalidate every logged score.
- Record the seed in the plan. "Reproducible" means someone else can rebuild
  this exact partition from the JSON alone.
- Next step: `freeze`.
