---
name: run
description: Drive the Titanic pipeline autonomously - run one experiment end to end, verify it, and know when to stop or ask. Use for an unattended or semi-attended pass rather than running a single step by hand.
---

# run — the autonomous pass

The other skills are **procedures**: each says how to do one step. This one is
the **policy**: it decides what to run, when to stop, and when to stop and ask.
Those three decisions are the whole of what makes this an agent rather than a
Makefile, which is why they live in a document rather than in Python.

Read `CLAUDE.md` first if you have not. Rules 1–4, both tripwires and the
measurement hierarchy apply to everything below and nothing here relaxes any of
them.

## There are two layers, and a pass works on the upper one

The **pipeline** is steps 1–7: pull, review, split, freeze, explore, tune,
ensemble. It runs once to get a frozen dataset, and then it is behind you.
`make bootstrap` is that phase.

The **experiment** is the repeatable unit: open a container with its own
train/val split, change one thing, measure it against the champion, read the
validation split once, close with a verdict. `make experiments` lists the eight
that exist.

A pass is **one experiment**. It is not a lap of `make loop`. Earlier versions of
this document described the step loop as the pass, which is why cold start 001
ignored the skill and worked out the lifecycle from `make help` instead.

## 1. Preflight

```bash
make status          # where the pipeline is, derived from artifacts on disk
make verify          # the frozen data still matches its manifest
make receipts        # has the final test been read for this dataset?
make audit           # did the rules hold on everything logged so far?
```

- **`make verify` fails → stop.** The frozen data drifted and every score in the
  ledger is suspect. That is a finding, not an obstacle. Report and wait.
- **`make audit` reports a binding failure → stop.** Do not start an experiment
  on top of a broken chain or a receipt mismatch.
- **`data/raw` is empty → `make pull`.** Needs the optional kaggle extra and
  credentials; if it cannot authenticate, stop and say so.
- **Not frozen yet → `make bootstrap`, once.** It runs steps 2–4 and `freeze`
  refuses to run twice, so it cannot sit in a loop.
- **Frozen already → skip both.** `make status` saying the data is frozen means
  the bootstrap is behind you.

## 2. Pick exactly one backlog item, and say why

```bash
make backlog         # the untried ideas
make findings        # everything learned, including what is already settled
make experiments     # what each experiment did and how it closed
```

Pick **one** item. Write down which one and why it is worth a pass now — its
expected effect, and which instrument could see an effect that size.

**An `untried` finding is not proof that it is untried.** The backlog is
append-only and nothing retires an item when an experiment happens to answer it.
`Decompose rf_kitchen` sits there today and exp-008 already did it. So before
taking an item, check it against `make experiments` and `make findings`. If it is
already answered: record that with `make finding` as a confirmed or refuted
claim citing the experiment that answered it, say so in the report, and take the
next item instead. Re-running settled work leaves a ledger that looks
healthy and teaches nothing.

If every item is answered or none is affordable, **stop and ask**. There is no
planner; choosing what to try is the last human-in-the-loop step and inventing a
hypothesis nobody asked for is not what this pass is for.

## 3. Open the experiment

```bash
make experiment-open NAME="what it tries"
```

Options to make are **variables, not flags**: a leading dash-dash option is
consumed by make itself and the target never sees it. That is why `next` is its
own target and why the journal takes `LEVEL=decision`. A test checks every
document in the repo for this.

Write the plan and its predictions into `experiments/<id>/plan.md` **before**
scoring anything, then lock it:

```bash
make experiment-lock-plan
```

`make bakeoff` locks it for you, so this is only needed on the `tune` and
`features` paths. `make audit` re-hashes every locked plan afterwards. A prediction that can be edited once the answer is known is not a
prediction, which is the only reason scoring predictions is worth doing.

## 4. Run the search the item calls for

| the item is about | run |
|---|---|
| hyperparameters of one model | `make tune` |
| which features to keep | `make features`, then `make nested` |
| many hunches at once, or a new champion | `make bakeoff`, then `make bakeoff-decide` |

- **Any feature fitted from the label → `make permutation` first.** The in-fold
  group rates are the classic Titanic leak. `make bakeoff` refuses to score a
  label-reading arm until the control lands at chance, so this is belt and
  braces rather than the only guard — and it is the one to re-run after any
  change to `bakeoff.Engineer`.
- **Never run `make explore` with an experiment open.** It reads all 712 working
  rows and charts them, including the open experiment's validation rows, so it
  puts the rows you are about to be measured on in front of whoever reads the
  charts. That is a val-split leak through visualisation. Chart before opening,
  or not at all.
- **A failed configuration is a result.** It gets a ledger row with the error in
  `notes` and an empty `cv_score`. Silently dropping it makes the search a lie.
- **Comparisons go through `titanic_agent.compare`**, and never between two
  absolute CV numbers from different experiments — the split is re-drawn each
  time. `delta_vs_champion` is the number that crosses an experiment boundary.

## 5. Read the validation split exactly once

```bash
make experiment-evaluate
```

Say **before running it** which configuration is being scored and what would
surprise you. `require_val_unread` refuses a second read within one experiment,
and the refusal is the point: a second read on the same rows is selection, not
measurement.

It is a **tripwire with ~0.06 resolution**, not a calibrator. It cannot rank two
candidates and it cannot measure selection bias. Do not read the cv→val gap as a
measurement of anything, and do not ask for more reads — `docs/MEASUREMENT.md`
has the arithmetic and `require_val_unread` will quote it back at you.

## 6. Close it with an honest verdict

```bash
make experiment-close SUMMARY="what it tried, what it measured, what it means"
```

The verdict is one of:

- **crowned** — a challenger passed all three declared checks and
  `experiments/champion.jsonl` has a new entry.
- **held** — the champion survived. Say what failed: the bar, the fresh-seed
  confirmation, or the nested estimate.
- **inconclusive** — the measurement could not resolve the effect. Say which
  instrument fell short and by how much. This is a real outcome and the most
  common one; dressing it up as either of the others is the dishonesty this repo
  exists to detect.

Record what was learned — a claim with the `run_id`s behind it, never a bare
conclusion. A conclusion handed to the next experiment is folklore it cannot
check.

```bash
make finding CLAIM="what you now believe" STATUS=confirmed EVIDENCE=<run_id>,<run_id> \
  CONFIDENCE=medium NOTE="the numbers, and what would overturn it"
```

`STATUS` is one of `confirmed`, `refuted`, `inconclusive`, `untried`; the first
two require evidence. Never edit `findings.jsonl` by hand — it is append-only
and direct writes to it are denied.

## Budget

- **One experiment per pass. Stop when it closes.** Not two, not "one more while
  the data is warm" — the second one would be chosen partly by the first one's
  result, which is selection the ledger cannot see.
- **20 minutes of wall clock**, or the caps in `experiment.json`, whichever binds
  first. `make summary` charges `loop_passes` and `bakeoff` charges
  `wall_clock_s`; `experiments.over_budget()` says when a cap is reached.
- `make nested` is budgeted at ~3 minutes and is a deliberate measurement rather
  than a search, so it does not count against the ~2-minute rule in CLAUDE.md.

## Stop conditions

Stop and report. Do not push through any of these:

- **The experiment closed.** That is the pass. Report it.
- **A step fails twice with the same error.** Twice is a pattern; a third attempt
  is not analysis. Journal `step_failed` and report what you tried.
- **`freeze.verify()` returns False**, or `make audit` reports a binding failure.
- **A tripwire** (below).
- **No affordable, unanswered backlog item.**
- **Budget exhausted.** Say what you spent it on.
- **A permission prompt you cannot satisfy.** In a headless pass a denied tool is
  a stop, not something to work around with a different command. Report the
  command that was denied.
- **You are about to do something this document does not cover.** Ask.

Log the stop reason:

```bash
make note EVENT=decision DETAIL="stopped: <reason>"
```

A pass that stops for a stated reason is a good pass. A pass that stops without
saying why is indistinguishable from one that crashed — and the first headless
pass was exactly that, because the only way to journal was to run arbitrary
Python and the permission settings denied it. `make note` exists for this; use
it rather than reaching for an interpreter.

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
   cannot happen by running a pass. An autonomous pass does not take it, however
   settled the champion looks.

On either, record the refusal and stop:

```bash
make note EVENT=refused STEP=ensemble DETAIL="why you believe it is needed"
```

A refusal that leaves no trace is indistinguishable from never having
considered it — which is why `refused` is an event in the journal.

## Not stop conditions

These are **results**. Log them, report them, and finish cleanly:

- Nothing beats the champion, or nothing beats the majority-class baseline.
- The ensemble scores below the best single model.
- A model fails to converge, or a search finds nothing better than its default.
- The validation read disappoints, or lands the other way from the prediction.
- Every prediction in `plan.md` was wrong.

A disappointing number is information. The pipeline is working when it tells you
something you did not want to hear.

## Never

- **Edit source to make a step pass.** If a done criterion cannot be met, that is
  the finding. Report it.
- **Re-run a search hoping for a luckier seed.** Repeating until a number
  improves is not tuning; it is selecting on noise, and it is the specific
  dishonesty this repo exists to detect.
- **Hand-edit `experiments/results.csv`**, `findings.jsonl`, `champion.jsonl` or
  `run_log.jsonl`. They are append-only. A correction is a new row whose `notes`
  reference the original `run_id` — an amendment, never an edit.
- **Edit a locked `plan.md`.** If the plan has to change, that is a new
  experiment, not an amendment to this one's predictions.
- **Sign `holdout_read` falsely.** Nothing verifies it. That is exactly why it
  matters: a wrong score you reported is recoverable, a false attestation makes
  every other row untrustworthy.
- **Delete or rewrite a report to hide a result.** Reports are regenerable;
  deleting one to change the story is not regeneration.
- **Open a second experiment to get a better answer.** See Budget.

## Done criteria for a pass

- [ ] Exactly one experiment was opened, and it is closed with a verdict.
- [ ] Its `plan.md` was written and locked before anything was scored, and
      `make audit` says it still hashes to what was journaled.
- [ ] Its validation split was read exactly once.
- [ ] Every configuration evaluated has exactly one ledger row, failures
      included, carrying the current `frozen_sha256` and an honest
      `holdout_read`.
- [ ] Every ledger row from this pass carries the experiment's `experiment_id`.
- [ ] What was learned is in `findings.jsonl` with its evidence.
- [ ] `make verify`, `make audit` and `make leakage` all pass at the end.
- [ ] Holdout receipts: zero, or exactly one for this `frozen_sha256`.
- [ ] The stop reason is in the journal.

## Reporting

Finish with a short summary a human can read without opening anything:

- which backlog item was taken, and why;
- what ran, and what it measured against the champion;
- the verdict, and the instrument that produced it;
- what stopped the pass;
- anything refused, and what you would need in order to proceed;
- anything surprising — especially results that went the wrong way, and
  predictions in `plan.md` that were wrong.

Then print `make log` and `make journal LEVEL=decision`.

Each pass also writes `runs/<id>/run.md` via `make summary` — scores, decisions,
refusals and artifacts for that pass alone, derived from the global ledger and
journal rather than copied, because two records of the same events can disagree
and then neither can be trusted. `make runs` lists every pass.
