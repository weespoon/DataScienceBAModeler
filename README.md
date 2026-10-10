# DataScienceBAModeler

A deliberately small, deliberately strict data science repo for the
[Kaggle Titanic](https://www.kaggle.com/competitions/titanic) competition.

**The goal is not the leaderboard.** It is to learn agentic AI by handing an
autonomous agent — Claude Code — a complete, well-fenced workflow and letting it
run the whole thing on its own: pull the data, audit it, split it, freeze it,
explore it, tune models, ensemble them, and leave behind a trail a human can
audit afterwards. Titanic is the substrate because it is small, well understood,
and runs in seconds on a laptop, which makes the *agent's* behaviour the
interesting variable rather than the compute.

What I am actually testing: can an agent stay honest over a long autonomous run
when the rules are written down but nobody is watching each step?

## The pipeline

```
pull → review → split → freeze → explore → tune → ensemble
```

| # | Step | What it does |
|---|------|--------------|
| 1 | `pull` | Download the raw CSVs into `data/raw/` and verify them |
| 2 | `review` | Audit shape, missingness, cardinality, leakage → `reports/01_review.md` |
| 3 | `split` | Draw a seeded stratified train/holdout partition |
| 4 | `freeze` | Seal it: checksummed, read-only, **immutable** |
| 5 | `explore` | Chart the frozen *training* split → `reports/02_explore.md` |
| 6 | `tune` | Cross-validated model search; every run logged |
| 7 | `ensemble` | Combine the best models, score the holdout once, write a submission |

Each step has a skill in `.claude/skills/<step>/SKILL.md` stating its inputs,
outputs and done criteria — that is what the agent reads before running it.

## The three rules

These are the point of the repo. They are the constraints that make an
autonomous run trustworthy, and they live in [CLAUDE.md](CLAUDE.md):

1. **The frozen dataset is immutable.** After `freeze`, nothing in
   `data/frozen/` is ever modified. Files are chmod 0444 and checksummed in
   `MANIFEST.json`; every later step verifies them before it runs. `freeze()`
   has no `force` flag.
2. **The holdout is read once, at the end.** Model selection happens entirely
   by cross-validation on the training split. Only `ensemble` may read
   `holdout.csv`, only explicitly, and the number it gets is a report — never a
   signal to go back and pick a different model.
3. **Every experiment appends to the ledger.** One row in
   `experiments/results.csv` per run, including failures. A run that is not in
   the ledger did not happen.

## Quickstart

Requires Python 3.11+.

```bash
make setup     # create .venv, install the package editable
make test      # smoke tests
make pull      # needs Kaggle credentials -- see .claude/skills/pull/SKILL.md
make all       # steps 2-7, in order
make log       # read the experiment ledger
```

`make help` lists every target.

## Dependencies

pandas, numpy, scikit-learn, matplotlib. `pytest` and `ruff` for development,
and the optional `kaggle` CLI for step 1 only — drop the CSVs into `data/raw/`
by hand and the rest of the pipeline runs without it.

That list is a constraint, not an accident. Nothing here needs gradient boosting
libraries or a hyperparameter framework, and every step must finish on a laptop
CPU in seconds.

## Layout

```
├── CLAUDE.md               # operating rules for the agent
├── Makefile                # one target per pipeline step
├── .claude/skills/         # one SKILL.md per step: inputs, outputs, done criteria
├── data/raw/               # gitignored -- raw download
├── data/frozen/            # gitignored -- immutable after `freeze`
├── experiments/results.csv # append-only experiment ledger
├── reports/                # generated charts and markdown summaries (committed)
├── src/titanic_agent/      # one module per step, plus paths.py and results_log.py
└── tests/                  # a smoke test per module
```

## Status

Scaffolding stage. Every pipeline module is a documented stub: real signatures,
real dataclasses, real done criteria, `NotImplementedError` bodies with a TODO
describing the implementation. `results_log.py` is implemented, because the
ledger is plumbing that everything else depends on.

The smoke tests pin each module's public contract and mark the
not-yet-implemented calls `xfail`, so they flip to passing as the steps land.

## License

MIT
