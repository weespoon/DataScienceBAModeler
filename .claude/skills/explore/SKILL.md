---
name: explore
description: Step 5 of the Titanic pipeline. Chart the frozen training split and write reports/02_explore.md with candidate features. Use after freeze and before tune.
---

# Step 5 — explore

EDA on the frozen **training** split only, ending in a concrete feature shortlist.

## Inputs

- `data/frozen/train.csv`, reached through `freeze.require_frozen()`.
- Never `data/frozen/holdout.csv`. Looking at it — even in a chart — spends the
  one honest estimate this pipeline has.

## Outputs

- PNG charts in `reports/` (at most `max_charts`, default 8), ~110 dpi.
- `reports/02_explore.md` embedding every chart, ending in a
  **Candidate features** section that `tune` can act on directly.

Suggested charts: target balance; survival rate by `Sex`, `Pclass`, `Embarked`;
`Age` and `Fare` distributions split by target; a missingness bar chart; survival
by family size (`SibSp + Parch`).

## How to run

```bash
make explore
```

## Done criteria

- [ ] `reports/02_explore.md` exists and every chart it references is on disk.
- [ ] No chart or statistic in the report derives from `holdout.csv`.
- [ ] The **Candidate features** section names at least five features, including
      any derived ones (title extracted from `Name`, family size, `is_alone`,
      `Cabin` deck letter) with one line of reasoning each.
- [ ] The step completes in under ~30 seconds.

## Notes

- Use the `Agg` matplotlib backend; this must work headless.
- One chart per figure. No subplot grids pretending to be dashboards.
- `plt.close()` every figure — figures accumulate across runs otherwise.
- Charts are regenerable artifacts. `make clean` deletes them; that must be fine.
- Next step: `tune`.
