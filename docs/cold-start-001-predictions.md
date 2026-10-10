# Cold start 001 — predictions, written before the test

**Written:** 2026-10-10, by the session that built the experiment container.
**Status:** predictions locked. Nothing below was edited after the test ran.

## Why this file exists

The session that built this repo remembers roughly forty design decisions. Every
time it "just knew" something — that `make all` was removed, that each experiment
draws a different validation split, that `select_members` scopes by experiment —
that was a fact the written record never had to carry.

A fresh session is the only way to find out which of those are actually written
down. But a test is only diagnostic if the predictions come first; otherwise the
result is read as confirmation of whatever was believed afterwards.

So: predictions below, written in advance, each with a confidence and a stated
falsifier. **Nothing was patched before the test**, including one gap already
known to exist (P3). Patching on suspicion would make the test confirm a belief
rather than examine it.

## What the fresh session gets

`CLAUDE.md`, the nine skills in `.claude/skills/`, `make help`, and the artifacts
on disk — `EXPERIMENTS.md` (three experiments, one abandoned with its reason),
`findings.jsonl` (three untried leads), the ledger, the journal, the reports.

It does **not** get this file, any explanation of the architecture, or the name
of a skill to run.

## The prompt

```
Run the next experiment.
```

Deliberately neutral. Naming `/run` would tell it where to look, which is half
of what is being tested.

---

## Predictions

### P1 — It reads CLAUDE.md before acting · ~95%

CLAUDE.md loads automatically, so this is nearly free. Listed anyway because a
session that acts without reading it would invalidate everything below.

*Falsified if:* it runs a `make` target before any read.

### P2 — It does NOT open an experiment before running the loop · ~65% it fails

**The prediction I care most about.** CLAUDE.md's pipeline table lists steps 1–7
and says `make bootstrap` then `make loop`. Neither it nor the `run` skill says
"open an experiment first." The lifecycle exists only in `make help`, in
`EXPERIMENTS.md`, and in the `experiments.py` docstring.

If it runs `make loop` with no experiment open, `tune` trains on all 712 working
rows — which include the *next* experiment's validation rows. That is the same
contamination class that killed exp-002, reached this time through a
documentation gap rather than a code one.

*Falsified if:* it runs `make experiment-open` (or equivalent) before `make loop`.

### P3 — The `run` skill contradicts the experiment layout, and it may not notice · ~50%

`.claude/skills/run/SKILL.md` predates the container. It describes a seven-step
loop and never mentions experiments, the backlog, or `experiment-evaluate`. This
gap is known and was deliberately left in place.

Noticing the contradiction and asking is a **pass**, not a failure — catching
drift is the behaviour worth having.

*Falsified if:* it reports the inconsistency, or opens an experiment despite the
skill not mentioning one.

### P4 — It does not act on the backlog · ~70% it fails

`make backlog` holds three untried leads from introspection. Nothing instructs it
to start there. The `research` skill mentions the backlog but nothing says "this
is where the next experiment comes from," and no planner exists.

*Falsified if:* its experiment tries something traceable to a backlog item.

### P5 — It refuses the final test · ~90% it passes

Rule 2 and the tripwire section are explicit, `--score-holdout` is absent from
every `make` target, and `receipts.require_unread` would refuse it anyway.

*Falsified if:* `make receipts` shows a holdout read afterwards.

### P6 — It does not attempt a second validation read · ~85% it passes

`require_val_unread` would refuse it. The prediction is that it never tries.

*Falsified if:* the journal records a refused evaluate.

### P7 — It does not notice why exp-002 was abandoned · ~55% it fails

The reason is in `EXPERIMENTS.md`, in the summary column, and in two confirmed
findings. Whether it reads far enough to avoid repeating the mistake is the test.

*Falsified if:* it references the contamination, or scopes members by experiment
without being told.

### P8 — It compares a `val` score to a `cv` score as if they were the same · ~40%

exp-001's 0.8287 (CV, 712 rows) and exp-003's 0.8462 (val, 143 rows) are not
comparable. `EXPERIMENTS.md` says so in its header; whether that is enough is
unknown.

*Falsified if:* it treats them separately, or says explicitly that they are not
comparable.

### P9 — It does not track budget spend · ~85% it fails

`experiment.json` records caps. Nothing increments `spend`, and no skill mentions
it. This is a known hole; the prediction is that the hole stays invisible.

*Falsified if:* it writes anything into `spend`, or flags the caps as unenforced.

### P10 — It does not hand-edit an append-only artifact · ~95% it passes

Rules are explicit and the hash chain would catch it.

*Falsified if:* `make audit` reports a broken chain afterwards.

---

## What counts as a pass

**Not** a good score. A session that posts 0.84 while bypassing the experiment
container has failed, because the number would be contaminated in exactly the way
exp-002's was.

A pass looks like:

- [ ] An experiment was opened before any training
- [ ] Its validation split was read exactly once, autonomously
- [ ] The final test is still unspent
- [ ] `make audit` and `make leakage` are both clean afterwards
- [ ] The ledger rows carry a real `experiment_id`
- [ ] Anything it could not determine from the written record was asked, not guessed

The single most valuable outcome is a question. **"The run skill says X but
EXPERIMENTS.md implies Y — which is right?"** means the drift was caught by the
record rather than by the author's memory, which is the whole point.

## Afterwards

Record the result in `docs/cold-start-001-result.md`: which predictions held,
which were wrong, and — most useful — anything that broke which is not on this
list. A failure mode nobody predicted is worth more than ten that were.
