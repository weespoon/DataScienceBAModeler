---
name: run
description: Drive the Titanic pipeline autonomously - decide the next step, run it, verify its done criteria, and know when to stop or ask. Use for an unattended or semi-attended pass rather than running a single step by hand.
---

# run — the autonomous pass

The other seven skills are **procedures**: each says how to do one step. This one
is the **policy**: it decides what to run, when to stop, and when to stop and
ask. Those three decisions are the whole of what makes this an agent rather than
a Makefile, which is why they live in a document rather than in Python.

## Before you start

```bash
make status          # where the pipeline is, derived from artifacts on disk
make verify          # the frozen data still matches its manifest
make log             # what has been tried already
make receipts        # has the holdout been read for this dataset?
```

Read `CLAUDE.md` first if you have not. If `make verify` fails, **stop** — that
is a finding, not an obstacle. Report the mismatch and wait.

## The loop

Repeat until a stop condition fires:

1. `make next` — it names the next incomplete step, or says complete.

   Options to make are **variables, not flags**: a leading dash-dash option is
   consumed by make itself and the target never sees it. That is why `next` is
   its own target, and why the journal takes `LEVEL=decision` rather than an
   option. A test checks every document for this.
2. Read that step's skill: `.claude/skills/<step>/SKILL.md`. Read it *before*
   running the step, every time. The skill is where the done criteria live, and
   they are what you will check against in (4).
3. Run it (`make <step>`). A non-zero exit is a failure — see **When a step
   fails**.
4. Check the step's done criteria from its skill, one by one. Most name a
   function that proves them (`pull.verify`, `split.validate`, `freeze.verify`).
   Run it. Do not mark a step done because the command exited zero.
5. Journal what happened:
   ```python
   from titanic_agent import run_log
   run_log.record("step_finished", step="tune", detail="8 models, best 0.8287")
   ```
6. Go to 1.

`make bootstrap` (steps 2–4) runs once, ever. `make loop` (steps 5–7) is the
repeatable part. If status says the data is frozen, the bootstrap is behind you.

## Stop conditions

Stop and report. Do not push through any of these:

- **The pipeline is complete.** Report the final state and the ledger.
- **A step fails twice with the same error.** Twice is a pattern; a third attempt
  is not analysis. Journal `step_failed` and report what you tried.
- **`freeze.verify()` returns False.** The frozen data drifted. Every score is
  suspect. Investigate and report — do not re-freeze.
- **A tripwire** (below).
- **Budget exhausted.** Default: 20 minutes of wall clock, or 3 full passes over
  `make loop`, whichever comes first. Say what you spent it on.
- **You are about to do something this document does not cover.** Ask.

Log the stop reason: `run_log.record("decision", detail="stopped: <reason>")`.
A pass that stops for a stated reason is a good pass. A pass that stops without
saying why is indistinguishable from one that crashed.

## Tripwires — refuse, and ask a human

Two actions are never yours to take alone, no matter how reasonable they look in
the moment:

1. **Unfreezing** — any `chmod`, delete, overwrite or re-freeze under
   `data/frozen/`. A `PreToolUse` hook will refuse you, and the correct response
   to that refusal is to stop, not to find another route. The procedure exists in
   `docs/UNFREEZING.md`, which is addressed to a person and deliberately not
   linked from the agent-facing docs.
2. **`ensemble --score-holdout`** — the one irreversible measurement, once per
   frozen dataset. It is deliberately absent from every `make` target so that it
   cannot happen by running the loop. An autonomous pass does not take it.

On either: `run_log.record("refused", step=..., detail="why you believe it is
needed")`, then stop and say so. A refusal that leaves no trace is
indistinguishable from never having considered it — which is why `refused` is an
event in the journal.

## Not stop conditions

These are **results**. Log them, report them, and carry on or finish cleanly:

- Nothing beats the majority-class baseline.
- The ensemble scores below the best single model. (It currently does.)
- A model fails to converge, or a search finds nothing better than its default.
- The holdout, once read, disappoints.

A disappointing number is information. The pipeline is working when it tells you
something you did not want to hear.

## Never

- **Edit source to make a step pass.** If a done criterion cannot be met, that is
  the finding. Report it.
- **Re-run a search hoping for a luckier seed.** Repeating until a number
  improves is not tuning; it is selecting on noise, and it is the specific
  dishonesty this repo exists to detect.
- **Hand-edit `experiments/results.csv`.** It is append-only. A correction is a
  new row whose `notes` reference the original `run_id` — an amendment, never an
  edit.
- **Sign `holdout_read` falsely.** Nothing verifies it. That is exactly why it
  matters: a wrong score you reported is recoverable, a false attestation makes
  every other row untrustworthy.
- **Delete or rewrite a report to hide a result.** Reports are regenerable;
  deleting one to change the story is not regeneration.

## Done criteria for a pass

- [ ] `make status` shows every step complete, or the journal says why not.
- [ ] Every model evaluated has exactly one ledger row, failures included.
- [ ] Every ledger row carries the current `frozen_sha256` and an honest
      `holdout_read`.
- [ ] `experiments/run_log.jsonl` has a `step_started` and a terminal event
      (`step_finished`, `step_failed` or `step_skipped`) for each step attempted.
- [ ] `make verify` still passes at the end.
- [ ] Holdout receipts: zero, or exactly one for this `frozen_sha256`.
- [ ] The stop reason is in the journal.

## Reporting

Finish with a short summary a human can read without opening anything:

- what ran, and what it scored against the baseline;
- what stopped the pass, and why;
- anything refused, and what you would need in order to proceed;
- anything surprising — especially results that went the wrong way.

Then print the ledger (`make log`) and the journal (`make journal`).

Each pass also writes `runs/<id>/run.md` — scores, decisions, refusals and
artifacts for that pass alone, derived from the global ledger and journal rather
than copied, because two records of the same events can disagree and then
neither can be trusted. `make runs` lists every pass.
