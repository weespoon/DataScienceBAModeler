---
name: pull
description: Step 1 of the Titanic pipeline. Download the raw competition CSVs into data/raw and verify them. Use when data/raw is empty or a fresh copy of the raw data is needed.
---

# Step 1 — pull

Get the raw Kaggle Titanic data onto disk, untouched, and prove it is usable.

## Inputs

- Kaggle credentials: `~/.kaggle/kaggle.json`, or `KAGGLE_USERNAME` + `KAGGLE_KEY`
  in the environment. Both are gitignored; never print or commit their contents.
- Or, with no credentials at all: a manually downloaded archive passed as
  `--local-zip path/to/titanic.zip`.

## Outputs

- `data/raw/train.csv` — 891 rows, includes the `Survived` target.
- `data/raw/test.csv` — 418 rows, no target.

Both gitignored. This is the only step that writes to `data/raw/`.

## How to run

```bash
make pull
# or, without Kaggle credentials:
.venv/bin/python -m titanic_agent.pull --local-zip ~/Downloads/titanic.zip
```

## Done criteria

- [ ] Both CSVs exist and are non-empty.
- [ ] Both parse with `pandas.read_csv` without error.
- [ ] `train.csv` has a `Survived` column; `test.csv` does not.
- [ ] `pull.verify()` returns `True`.
- [ ] No credential material appears in stdout, the ledger or any report.

## Notes

- Idempotent: re-running when `verify()` already passes is a no-op unless
  `--force` is given.
- If the download fails, say so plainly and stop. Do not synthesize substitute
  data — a fabricated dataset would poison everything downstream.
- Next step: `review`.
