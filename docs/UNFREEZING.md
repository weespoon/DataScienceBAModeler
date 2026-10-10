# Unfreezing the dataset — a human procedure

**This document is addressed to a person, not to the agent.** It is deliberately
not referenced from `CLAUDE.md` or from any skill, and the agent is instructed to
stop and ask rather than do any of this.

## Why this is not an agent action

Rule 1 says `data/frozen/` is immutable after `freeze`. The point of an immutable
dataset is that a score can be traced to the exact bytes it was computed on. The
moment the frozen data changes, **every score already in `experiments/results.csv`
becomes uncomparable** — the rows carry the old `frozen_sha256` and there is no
way to recover what they measured.

That makes unfreezing irreversible in the only sense that matters: it destroys the
meaning of the historical record, and no amount of care afterwards restores it.

There is also a subtler reason to keep the recipe out of the agent's reach. An
earlier version of this repo quoted the unfreeze command directly in `CLAUDE.md`,
while `make all` reliably failed at `freeze` on every run after the first. The
agent was therefore handed a broken pipeline and, three lines away, the exact
command that would make it green. That is a manufactured temptation, not a test of
anything. It has been removed.

## When unfreezing is actually warranted

Only two situations:

1. **The raw pull was corrupt** — `pull.verify()` passed but the data is wrong
   (truncated download, wrong competition, upstream change to the files).
2. **The split is structurally broken** — e.g. the stratification failed, or the
   holdout is the wrong size, in a way `split.validate()` did not catch.

Wanting a better score is **not** a reason. Neither is a disappointing holdout
number: that is the honest estimate doing its job. See Rule 2.

## The procedure

```bash
# 1. Record what you are destroying.
cat data/frozen/MANIFEST.json | tee /tmp/manifest-before-unfreeze.json

# 2. Note the fingerprint you are invalidating.
python -c "import json; print(json.load(open('data/frozen/MANIFEST.json'))['dataset_sha256'])"

# 3. Unfreeze.
chmod -R u+w data/frozen
rm -rf data/frozen/*
touch data/frozen/.gitkeep

# 4. Re-split and re-freeze.
make split
make freeze
```

Note that step 3 will be **denied by the PreToolUse hook** if you ask the agent to
run it. That is intentional. Run it yourself, in your own shell.

## Afterwards

- The new `dataset_sha256` differs, so `make audit` will show the ledger split
  across two frozen datasets. That is correct and should stay visible — do not
  delete the old rows. Rule 3 is append-only, and the historical rows are the
  record that this happened.
- Tell the agent explicitly that the dataset was re-frozen and that rows carrying
  the previous fingerprint are not to be compared against new ones.
- The holdout read receipt in `experiments/holdout_reads.json` is keyed by
  fingerprint, so the new dataset gets a fresh single read. Do not reuse this as a
  way to take a second look at a holdout you have already seen — that is laundering
  a rule-2 violation through a rule-1 violation, and `make audit` will show two
  receipts where an honest run has one.
