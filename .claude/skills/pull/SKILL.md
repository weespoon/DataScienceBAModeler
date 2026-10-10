---
name: pull
description: Step 1 of the Titanic pipeline. Download the raw competition CSVs into data/raw and verify them. Use when data/raw is empty or a fresh copy of the raw data is needed.
---

# Step 1 — pull

Get the raw Kaggle Titanic data onto disk, untouched, and prove it is usable.

## Inputs

- The optional `kaggle` extra: `make setup EXTRAS=dev,kaggle`. It is not in the
  default environment because it pulls ~25 transitive packages.
- Kaggle credentials, in any shape the client accepts:
  - `KAGGLE_API_TOKEN` in the environment, or
  - a token saved to `~/.kaggle/access_token` (what this repo uses), or
  - an OAuth session from `kaggle auth login`, or
  - a legacy `~/.kaggle/kaggle.json` with `{username, key}`.

  All are gitignored; never print or commit their contents. Note that env vars
  do not survive between agent tool calls — the file-based token is what makes
  this step work unattended.
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
  `--force` is given (it reports `already present`).
- `verify()` never raises — a missing or malformed dataset is a `False`, so
  callers can branch on it. It also rejects a `Survived` column leaking into
  `test.csv`, which would mean this is not the competition dataset at all.
- Archive extraction is flattened to basenames, so a crafted zip cannot write
  outside `data/raw/`.
- Authentication failures report the exception type only, never its payload —
  the client can include configuration detail in the message.
- If the download fails, say so plainly and stop. Do not synthesize substitute
  data — a fabricated dataset would poison everything downstream.
- `gender_submission.csv` also lands in `data/raw/`. It is Kaggle's example
  submission, not training data; ignore it.
- Next step: `review`.
