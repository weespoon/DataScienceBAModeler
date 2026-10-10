---
name: review
description: Step 2 of the Titanic pipeline. Audit the raw data for shape, missingness, cardinality and leakage, and write reports/01_review.md. Use after pull and before split.
---

# Step 2 — review

Understand the raw data before any split is drawn. Read-only.

## Inputs

- `data/raw/train.csv`, `data/raw/test.csv` (from `pull`).

## Outputs

- `reports/01_review.md` containing:
  - row/column counts for train and test;
  - a per-column table of dtype, missing fraction and distinct-value count;
  - constant columns and high-cardinality columns called out by name;
  - target balance for `Survived`;
  - leakage suspects — anything that separates the target suspiciously well;
  - a short **Decisions needed** section naming which columns need imputation
    and which need encoding.
- Return value: a `ReviewReport` later steps can branch on.

## How to run

```bash
make review
```

## Done criteria

- [ ] `reports/01_review.md` exists and is regenerable (safe to `make clean`).
- [ ] Every column in `train.csv` appears exactly once in the per-column table.
- [ ] `Age`, `Cabin` and `Embarked` missingness is quantified (they are the known
      gaps in this dataset — if they look clean, suspect the loader, not the data).
- [ ] `Name`, `Ticket` and `Cabin` are flagged high-cardinality.
- [ ] The raw CSVs are byte-identical before and after the run.

## Notes

- Do not impute, drop or transform anything here. This step only describes.
- Do not train a model to find leakage; correlation and group rates are enough
  and far cheaper.
- Next step: `split`.
