---
name: freeze
description: Step 4 of the Titanic pipeline. Seal the split into an immutable checksummed data/frozen directory. Use after split; this is the point of no return for the dataset.
---

# Step 4 — freeze

Materialise the split and make it immutable. **Everything after this step reads
`data/frozen/` and never writes to it.**

## Inputs

- `data/raw/train.csv`, `data/raw/test.csv`.
- `data/frozen/split_plan.json` (from `split`).

## Outputs

- `data/frozen/train.csv` — the modeling split.
- `data/frozen/holdout.csv` — the final estimate. Read once, by `ensemble` only.
- `data/frozen/test.csv` — the competition test set, copied through unchanged.
- `data/frozen/MANIFEST.json` — SHA-256 per file, a combined `dataset_sha256`
  fingerprint, row counts, the seed, the git SHA and the freeze timestamp.

All four files are chmod 0444.

## How to run

```bash
make freeze     # seal
make verify     # re-check checksums at any time
```

## Done criteria

- [ ] All four files above exist and are mode 0444.
- [ ] `freeze.verify()` returns `True`.
- [ ] `dataset_sha256` is non-empty and order-independent (it hashes the sorted
      `name:sha` lines, so file ordering cannot change it).
- [ ] Train and holdout row counts sum to the raw training row count.
- [ ] A second `freeze()` call raises `RuntimeError`.

## Notes — read before doing anything clever here

- There is **no `force` flag**, on purpose. Overwriting the frozen data
  invalidates every score already in `experiments/results.csv`, because those
  rows carry the old `frozen_sha256`.
- **Unfreezing is a human action. Stop and ask.** It is one of the two tripwires
  in this pipeline (the other is `ensemble --score-holdout`). If you think the
  data must be re-frozen, say why and wait for a person. Do not `chmod`, do not
  delete anything under `data/frozen/`, and do not work around the `PreToolUse`
  hook that denies writes there.
- `require_frozen()` is the guard every later step calls first. If it raises,
  the right move is to investigate the mismatch and report it, not to re-freeze.
  A checksum mismatch is information, not an obstacle.
- Next step: `explore`.
