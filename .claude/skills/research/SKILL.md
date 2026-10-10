---
name: research
description: Find out what we should try next - audit our own search spaces against the installed scikit-learn, and survey public Kaggle notebooks for techniques. Use between experiments, before planning the next one. Read this before reading anything external.
---

# research — find the next thing worth trying

Turns "we scored 0.8287, now what?" into a backlog of specific, testable ideas.
Output is `untried` findings; the planner picks from them.

## Read this part first

Titanic has a trap that most tutorial datasets do not, and this step walks
straight into it unless warned.

**The real passenger manifest is public.** Encyclopedia Titanica and several
mirrors carry all 1309 passengers with who actually survived. Public notebooks
scoring above roughly 0.85 — the ones a search for "better than ours" surfaces
first — are disproportionately doing a lookup against it rather than modelling
anything.

**This would corrupt our own number, not just a leaderboard.** Our locked final
test is 179 of those same passengers. A lookup scores near-perfectly on it too.
`make audit` would stay green, the ledger would chain cleanly, every rule in
CLAUDE.md would hold, and the result would be worthless. There is no check
anywhere else in this repo that would notice.

So:

- **A technique's justification may never be "it scored higher."** If you cannot
  say *why* it should generalise, it does not go in the backlog.
- **External data about actual Titanic passengers is out of bounds.** Not
  rate-limited, not budgeted — out of bounds. That includes the manifest, any
  derived lookup table, and any notebook's "feature" that is a join against one.
- **Suspect any reported score above ~0.85.** The honest ceiling on 891 labelled
  rows is roughly 0.80–0.84. A notebook at 0.95 is not better at modelling.
- When you see it, record it as a `refuted` finding naming the notebook, so the
  next experiment does not spend budget rediscovering it.

`make leakage` checks this structurally — feature provenance, external reads,
score plausibility, stray data files. Run it after any experiment that acted on
something read here. It raises the cost of cheating and makes an attempt visible;
it does not make cheating impossible, which is why the rule above matters.

## Inputs

- The installed scikit-learn, via `make introspect` — free, offline, and
  version-correct.
- Public Kaggle notebooks, via the Kaggle API (`kaggle kernels list -c titanic`).
  Sanctioned, already authenticated, no scraping.
- The current experiment's budget: `experiments/<id>/experiment.json`.

Not the general web. `web_fetches` defaults to 0: documentation found online
describes whatever version is current, while we run whatever is in `.venv`, and
that mismatch is silent. The installed package is the authority on our own API
surface.

## Outputs

`untried` findings in `experiments/findings.jsonl`, each carrying:

- a claim specific enough to test — "tune `min_samples_split` for the forests",
  not "improve the forest";
- a `source` that can be re-read later: `introspection`, `notebook:<ref>`;
- a `note` saying *why it should help*, which is the justification rule made
  concrete.

## How to run

```bash
make introspect            # what we tune vs what we could
make introspect UNTUNED=1  # ...and what is left at defaults
make backlog               # what is already queued
make leakage               # structural guards
```

## Done criteria

- [ ] Every model's search space has been checked against the installed
      estimator; no `invalid` keys, nothing `deprecated_tuned`.
- [ ] At least one new `untried` finding, each with a testable claim and a
      reason it should help.
- [ ] No finding whose justification is only that something scored higher.
- [ ] Notebooks read ≤ the experiment's `kaggle_notebooks` cap; count recorded
      in `spend`.
- [ ] `make leakage` passes.
- [ ] Any manifest-lookup technique encountered is recorded as `refuted`, with
      the notebook named, so nobody re-reads it.

## Notes

- Exhaust introspection before spending any external budget. It is free,
  instant, and answers the API-surface questions exactly; the network answers
  them approximately.
- Prefer a notebook's *reasoning* to its score. A well-argued 0.82 is worth more
  here than an unexplained 0.88.
- A technique that needs a dependency we do not have is still a finding — record
  it as `untried` with a note about the cost. CLAUDE.md's dependency rule
  decides, not this step.
- Next step: planning, which turns the backlog into runs.
