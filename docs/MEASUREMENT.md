# What each measurement can see

Three instruments measure accuracy in this project, on 569, 143 and 179 rows.
They are not three grades of the same thing. Each can resolve differences down
to a certain size and no further, and the sizes are far apart:

| instrument | rows | SE | quantum | can resolve | role |
|---|---:|---:|---:|---|---|
| **paired repeated CV** | 569 × 3 | **0.003** | 0.0018 | ~0.006 | **adjudicates** — compares two configurations |
| **validation read** | 143 | 0.032 | 0.0070 | ~0.064 | **tripwire** — catches a broken pipeline |
| **final test** | 179 | 0.029 | 0.0056 | ~0.058 | **reports** — one generalisation number |

Every decision this project makes is a comparison of configurations differing by
**0.003 to 0.015**. Only the first row of that table can see such a difference.
That single fact decides most of what follows.

---

## Why the validation split cannot adjudicate, at any number of reads

The usual objection to re-reading a holdout is adaptivity: choose a model after
looking, and the number becomes a selection statistic. True here too, and it is
why `require_val_unread` exists. But it is not the binding constraint, and
treating it as one invites the wrong fix — "read it more carefully", "correct for
multiplicity", "use a differentially private mechanism".

The binding constraint is **resolution**, and it is arithmetic.

### The quantum argument

Accuracy on 143 rows is a count over 143. It moves in steps of
**1/143 = 0.0070**. Two configurations that truly differ by 0.004 cannot differ
on this split by 0.004. They differ by 0, 0.0070, or 0.0140, depending on which
way the handful of disagreeing rows happens to fall. No statistical treatment
recovers a difference finer than the instrument's step size, because the
information is not in the measurement to begin with.

This argument needs no distributional assumption, and it kills the whole family
of proposals at once.

### The paired argument, including its favourable half

A paired test (McNemar) is more sensitive than comparing two absolute scores,
because it only looks at rows where the two models disagree. It still cannot get
there, and the trap is that the arithmetic *looks* better the fewer rows
disagree:

| disagreeing rows | paired SE | smallest resolvable difference |
|---:|---:|---:|
| 5 | 0.0156 | 0.035 |
| 20 | 0.0313 | 0.140 |
| 40 | 0.0442 | 0.280 |

Few disagreements buys a small SE *and* a resolution floor of n_disagreeing/143,
because the difference itself is quantised in those same rows. You cannot win at
either end. A paired val read resolves roughly **0.06** at best.

### Why the reusable-holdout literature does not rescue it

Dwork et al.'s Thresholdout, the Ladder mechanism and their relatives make a
holdout *reusable* by adding noise calibrated to the number of queries. On 143
rows:

- **The noise alone swamps the signal.** Laplace noise at scale ~2/n = 0.014 has
  sd **0.020** — five times the 0.004 effects at issue.
- **The sample complexity is three orders of magnitude out.** The guarantee needs
  roughly n ≈ √B · log(m/β) / T². For one adaptive round and 30 queries that is
  n ≈ 3.4/T²: resolving T = 0.004 needs ~**213,000** rows; T = 0.01 needs 34,000.
  Inverted at n = 143, the resolvable threshold is **T ≈ 0.15**. Strip every
  constant and log and take n = 1/T² and it is still 0.084.
- **It solves the wrong problem.** These mechanisms buy protection against
  *adaptivity*. Our problem is *resolution*. An infinitely reusable 143-row
  holdout still estimates accuracy to ±0.032 in steps of 0.0070.

So: no mechanism that makes reads **safe** makes them **informative**. That is
the general form, and it is why this file argues the class rather than naming one
paper.

### How large a split would adjudicate

Paired, 80% power, two-sided 0.05, assuming ~12% of rows disagree:

| effect | rows needed |
|---:|---:|
| 0.004 | ~59,000 |
| 0.010 | ~9,400 |
| 0.015 | ~4,200 |

We have 891 labelled rows in total. The discordance assumption drives these
numbers, so quote it whenever the figures are quoted.

**The conclusion is not "be careful with the val split". It is that a
split-based adjudicator is unavailable at this sample size, and cross-validation
on the training rows is the only instrument that can compare two candidates.**

---

## What the validation read is actually for

Not calibration of the selection process. That reading does not survive its own
data. The val − cv gap, over the three experiments that read it honestly:

| experiment | cv | val | gap |
|---|---:|---:|---:|
| exp-003 | 0.8190 | 0.8462 | **+0.0272** |
| exp-004 | 0.8190 | 0.8671 | **+0.0481** |
| exp-006 | 0.8401 | 0.8112 | **−0.0289** |

Mean **+0.0155**, sd **0.0398**. The point estimate says selection *deflates* CV,
which is not a thing selection does. Reading exp-006's −0.0289 as "the check
firing correctly" is confirmation bias: it sits 0.73 sd from the mean of the
other two, and on the same logic exp-004 would have reported selection as worth
−0.048.

The gap is also confounded in a direction nobody priced. CV trains on 455 rows
per fold; the val model trains on all 569. The learning curve pushes val *up* by
a few thousandths, opposing any inflation the gap is supposed to reveal.

And resolving a 0.010 bias through this gap would take **60 experiments**; 0.005
would take **243**. After five more the 95% interval is still ±0.028. An
accumulator here is guaranteed to stay inconclusive for the life of the project
while inviting exactly the over-reading above. So `EXPERIMENTS.md` prints the
±2 SE band beside the gap and no tracker is built.

What 143 rows *can* do is detect a **0.064** defect: a pipeline leak, a label
misalignment, a train/val swap. That is how the val read earned its keep —
exp-002 scored val 0.8392 against cv 0.8278 and was the contaminated one.

> **The validation read is an integrity tripwire with ~0.06 resolution: a
> pre-declared number compared against nothing.** Calling it a calibrator
> licenses precisely the over-reading its own three data points refute.

---

## The adjudicator, measured

Promoting paired repeated CV to "the adjudicator" means promoting an instrument
whose precision had never been checked. Two estimates were available from
exp-006's own ledger and they **straddled** the `DELTA = 0.003` bar:

- the naive SE this repo reports (sd of 15 paired fold differences / √15): mean
  **0.0030**
- a Nadeau–Bengio resampled-t correction, ×2.18 for fold overlap: **0.0066**
- the empirical sd of the 24 measured add-deltas, all against the same baseline:
  **0.0021**, centred at **−0.0022**

So the bar was either 1.4σ or 0.45σ — the difference between ~0 and ~5 spurious
accepts over 24 tests. Nobody knew, so it was measured: the same comparisons
re-run under **10 fold seeds** on exp-006's 569 training rows, taking the sd of
the paired delta across seeds. That is the component a single run cannot see,
because one run holds the fold draw fixed. 750 fits, 42 seconds.

| comparison | mean delta | naive SE | **seed sd** | ratio | bar in σ |
|---|---:|---:|---:|---:|---:|
| add `TitleFine` | +0.0012 | 0.0031 | 0.0024 | 0.78× | 1.23 |
| add `HasParch` | +0.0009 | 0.0025 | 0.0034 | 1.37× | 0.88 |
| remove `Title` (from 12) | +0.0043 | 0.0040 | 0.0030 | 0.77× | 0.99 |
| selected 6 vs baseline 12 | +0.0043 | 0.0050 | 0.0041 | 0.83× | 0.73 |

**Two findings.**

1. **The naive SE is approximately honest.** Seed-to-seed spread lands within
   0.77–1.37× of it, so the Nadeau–Bengio ×2.18 is too conservative for a fully
   paired statistic and is not applied. The SE this repo already reports can
   stand.
2. **`DELTA = 0.003` is about one standard error.** Between 0.73σ and 1.23σ
   across these four comparisons. A single test near the bar is close to a coin
   flip. This is a fact about 569 Titanic rows, not a bug — but it belongs in
   every report that leans on one comparison, which is why
   `PairedResult.describe` always prints the sigma.

### The quantum, here too

A paired delta is an integer number of flipped predictions. The 24 add-deltas
from exp-006 are integer multiples of 1/(569×3) = 0.00059 to within 4%: the
unique values are −11 to +2 flipped predictions. But the three repeats
re-predict the *same* 569 passengers, so a prediction that flips consistently
moves the mean by 1/569 = **0.00176**.

**`DELTA = 0.003` is 1.7 passengers.** That sentence does more for honesty than
any standard error.

---

## The ratchet, and why the endpoints get compared

The remove rule drops a feature *unless* keeping it clears the bar. Combined
with a one-σ bar, every undecidable removal goes the same way — so the errors
accumulate instead of cancelling. exp-006 shows it exactly:

| accepted drop | cost of dropping | σ |
|---|---:|---:|
| `Age` | +0.0024 | 0.85 |
| `SibSp` | +0.0006 | 0.18 |
| `Parch` | −0.0012 | −0.57 |
| `IsAlone` | +0.0012 | 0.62 |
| `Deck` | −0.0047 | −1.34 |
| `AgeMissing` | −0.0023 | −1.07 |

Sum of CV gains: **+0.00410** — exactly the reported headline. So the entire gain
is `Deck` (+0.0047) and `AgeMissing` (+0.0023); **three of the six accepted drops
cost accuracy** and were accepted only because the cost failed to reach 0.003.
`Age` was dropped on 1.3 passengers.

Also worth stating because it reads like a contradiction otherwise: `remove
Title` is published at +0.0152 but measures +0.0043 above. These are different
comparisons. exp-006 tested Title against the already-pruned 8-feature set
(test #34), not the 12-feature baseline; Title carries more unique signal once
`Age`, `SibSp`, `Parch` and `IsAlone` are gone. Both numbers are right.

Two consequences, both implemented:

- **`select` now compares the endpoints.** Each test is paired against whatever
  was current at the time, so summing the steps is not a measurement of the
  move. Comparing the selected set against the original baseline on identical
  folds is — and since the fold seed never changes, both fold vectors are
  already in hand and it costs **zero** extra fits. exp-006's own open question
  ("cannot distinguish the pruned set from the baseline") was a CV comparison it
  declined to make, not a limit of the data. The answer is **+0.0043 ± 0.0041,
  0.73σ**: probably a touch better, not resolvable.
- **The add rule is biased against acceptance before any feature is seen.** The
  24 adds centre at −0.0022 with sd 0.0021, so a candidate must overcome that
  dilution before clearing +0.003. A feature needs a true effect of **+0.0052
  for a 50% chance of acceptance and +0.0070 for 80%**. The rule advertised as a
  0.003 bar is in practice a 0.005–0.007 bar, which is the correct way to read
  "none of 24 candidates cleared 0.003".

(The likely mechanism for that dilution is recorded as an untried finding: with
`max_features='sqrt'` fixed, an extra correlated column reduces how often the
strong features are sampled per split.)

---

## The selection bias, measured

`make nested` runs the **same** `features.greedy_search` inside each of five
outer folds, on that fold's training rows only, and scores the selected model on
the outer test rows it never saw. Arm B is the fixed 12-feature baseline on
identical outer folds. Hyperparameters are held fixed from the real run, so this
is the bias of the feature search alone. 120 seconds on exp-006's 569 rows.

| outer fold | test rows | features chosen | inner cv | A search | B baseline | A − B |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 114 | 4 | 0.8462 | 0.7982 | 0.7982 | +0.0000 |
| 2 | 114 | 6 | 0.8322 | 0.8596 | 0.8596 | +0.0000 |
| 3 | 114 | 4 | 0.8352 | 0.7982 | 0.8333 | −0.0351 |
| 4 | 114 | 10 | 0.8410 | 0.8246 | 0.8246 | +0.0000 |
| 5 | 113 | 5 | 0.8326 | 0.8673 | 0.8761 | −0.0088 |

(The three exact ties are coincidence, not a collapsed comparison: those folds'
arms predict differently on 4, 4 and 2 rows respectively and the counts happen
to land equal. Checked, because three ties out of five looks like a bug.)

**Searching is worse than not searching: −0.0088 ± 0.0068, −1.29σ.**
nested(search) 0.8296 against nested(baseline) 0.8384.

And the published number was optimistic by more than it gained:

| | |
|---|---:|
| reported CV after the search | **0.8401** |
| nested estimate of the same procedure | **0.8296** |
| **selection bias** | **+0.0105** |
| CV gain the search bought (endpoints, paired) | +0.0043 |

The bias is **positive**, which is the falsifier passing — nesting cannot
improve on the number it corrects, so a negative result would have meant a
broken implementation rather than a good search. And it is larger than the
+0.0043 the search bought, which is the whole finding: *this search costs more
in selection bias than it earns in accuracy.*

At −1.29σ the sign is not certain at 2 SE. But the point estimate is negative,
it was predicted in advance to land in [−0.010, +0.002] (in the design review,
not as a committed pre-registration, so weaker evidence than cold-start-001),
and it agrees with every other measurement here: 24 of 24 candidate additions
failed, the prune's own endpoint comparison is 0.73σ, and the bar is one SE
wide. The reasonable conclusion is not "search harder" but **"feature search is
not a profitable technique at this sample size"** — which retires a class of
future experiments rather than adding one.

What the estimate does not cover: hyperparameters are not nested, and the
candidate list was trimmed from 24 to 8, so this measures the trimmed
procedure. Both are stated in the report and the ledger row.

---

## Cross-experiment comparison is not valid as absolute CV

`open_experiment` re-draws the train/val split each time, so two experiments'
CV numbers are **unpaired** — and the drift is larger than the effects. Identical
12-feature configurations, re-tuned per experiment:

| model | exp-002 | exp-003 | exp-004 | sd | range |
|---|---:|---:|---:|---:|---:|
| `gradient_boosting` | 0.8172 | 0.7979 | 0.8102 | 0.0098 | 0.0193 |
| `hist_gradient_boosting` | 0.8208 | 0.8014 | 0.8189 | 0.0107 | 0.0194 |
| `random_forest` | 0.8225 | 0.8172 | 0.8190 | 0.0097 | 0.0211 ¹ |
| `svc` | 0.8190 | 0.8190 | 0.8154 | 0.0021 | 0.0036 |
| `logistic` | 0.8102 | 0.8015 | 0.8067 | 0.0044 | 0.0088 |

¹ across four experiments, including exp-006's 0.8384.

Median sd **0.004–0.011**, i.e. **1–3× the effect sizes being argued about**.
Comparing exp-006's 0.8401 against exp-004's 0.8190 as though the difference
meant something is reading noise.

**This is the project's most likely failure over the next few experiments**, and
it is not yet fixed: each experiment is internally paired and defensible, while
every comparison *between* them is made at two to three times the effect size.
Five mutually incompatible "confirmed" findings accumulate, pass forward as
premises, and the final test cannot attribute what went wrong.

The fix is a **persisted champion**: each experiment re-scores the inherited
champion configuration on *its own* folds, paired, and records
`delta_vs_champion` rather than an absolute CV. `features.py` already does
exactly this internally — `base_run_id`, both arms on identical folds — so what
is missing is carrying it across the experiment boundary.

**Implemented in exp-007** (`champion.py`, `make champion`). The succession
lives in `experiments/champion.jsonl` — global and append-only, for the same
reason the ledger is. `bakeoff.py` re-scores the champion as arm zero on the
experiment's own folds, and every row carries `delta_vs_champion` with its
sigma. `EXPERIMENTS.md` now has a `Δ champion` column, which is the one to read
across rows.

The drift it removes was visible immediately. The same RF configuration scored
0.8360 on exp-006's folds and **0.8131** on exp-007's: a 0.023 swing from the
split alone, larger than every effect in exp-007 except the group features.
What it does not remove is that two experiments' deltas are still estimated on
different rows, so they carry the paired SE (~0.003-0.005), not zero. And a
`Δ champion` reported for the *best* arm is a best-of-N; exp-007's dropped from
+0.0264 to +0.0163 on fresh fold seeds.

---

## The honest summary

With 712 working rows you must **choose** between comparability and
per-experiment independence. The current design pays for independence — a fresh
143-row validation split every time — and then cannot spend it, because 143 rows
resolve 0.064 while the decisions are 0.004.

Saying that out loud is the honest version of the hierarchy. The val splits are
also, at the programme level, not independent of each other: they are
independent *draws* from the same 712, so they overlap, each experiment's
validation rows were earlier experiments' training rows, and earlier val scores
steered what this experiment chose to try. Making the blocks a fixed stratified
5-fold partition of the 712 would cost nothing and remove the overlap. It would
not reach 0.004. It is recorded here rather than implemented.
