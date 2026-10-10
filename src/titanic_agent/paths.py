"""Canonical filesystem layout. Every module imports its paths from here.

Nothing in this module touches the disk except :func:`ensure_dirs`, which only
creates directories that are safe to create (never inside ``data/frozen`` once a
manifest exists).
"""

from __future__ import annotations

from pathlib import Path

#: Repository root, resolved from this file's location (src/titanic_agent/paths.py).
ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
FROZEN_DIR: Path = DATA_DIR / "frozen"
REPORTS_DIR: Path = ROOT / "reports"

#: One directory per pass over the loop. Artifacts live here; evidence does not.
#: See :mod:`titanic_agent.runs` for why that split matters.
RUNS_DIR: Path = ROOT / "runs"
EXPERIMENTS_DIR: Path = ROOT / "experiments"

#: Kaggle competition slug used by :mod:`titanic_agent.pull`.
COMPETITION: str = "titanic"

#: Files expected in ``data/raw`` after a successful pull.
RAW_TRAIN: Path = RAW_DIR / "train.csv"
RAW_TEST: Path = RAW_DIR / "test.csv"

#: Artifacts produced by ``split`` and ``freeze``, all inside ``data/frozen``.
SPLIT_PLAN: Path = FROZEN_DIR / "split_plan.json"
FROZEN_TRAIN: Path = FROZEN_DIR / "train.csv"
FROZEN_HOLDOUT: Path = FROZEN_DIR / "holdout.csv"
FROZEN_TEST: Path = FROZEN_DIR / "test.csv"
MANIFEST: Path = FROZEN_DIR / "MANIFEST.json"

#: Append-only experiment ledger. See :mod:`titanic_agent.results_log`.
RESULTS_LOG: Path = EXPERIMENTS_DIR / "results.csv"

#: Append-only record of every holdout read, keyed by frozen fingerprint. It
#: lives outside ``data/frozen`` of necessity: an immutable store cannot log its
#: own access. See :mod:`titanic_agent.receipts`.
HOLDOUT_READS: Path = EXPERIMENTS_DIR / "holdout_reads.json"

#: The agent's own journal, distinct from the experiment ledger: ``results.csv``
#: records models, this records conduct. See :mod:`titanic_agent.run_log`.
RUN_LOG: Path = EXPERIMENTS_DIR / "run_log.jsonl"

#: Append-only claims-with-evidence. What was *learned*, as opposed to what was
#: scored — the thing a later experiment inherits. See :mod:`titanic_agent.findings`.
FINDINGS: Path = EXPERIMENTS_DIR / "findings.jsonl"

#: Human index of every experiment, derived from their manifests.
EXPERIMENTS_INDEX: Path = EXPERIMENTS_DIR / "EXPERIMENTS.md"

#: Append-only record of every time the ledger's hash chain was rebuilt. A
#: schema migration rewrites every line and so necessarily breaks the chain;
#: rather than pretend otherwise, the rebuild is recorded and the audit reports
#: it. See :func:`titanic_agent.results_log.verify_chain`.
CHAIN_EPOCHS: Path = EXPERIMENTS_DIR / "chain_epochs.jsonl"

#: Default random seed. Change it only with a recorded reason — the split that
#: produced the frozen holdout is reproducible from this value plus MANIFEST.json.
SEED: int = 20251010


#: Every path a pipeline run writes into. The test suite redirects all of them
#: into tmp_path (see the autouse guard in tests/conftest.py), so a test can
#: never touch the real ledger, receipts, journal, reports or run directories.
#:
#: This list exists because the guard was first written naming three paths by
#: hand; `RUNS_DIR` was added later and the guard was not, so the suite quietly
#: created 53 stray run directories. A guard that has to be remembered is a
#: guard that will be forgotten — enumerate once, iterate everywhere.
WRITABLE_ARTIFACTS: tuple[str, ...] = (
    "EXPERIMENTS_DIR",
    "RESULTS_LOG",
    "HOLDOUT_READS",
    "RUN_LOG",
    "FINDINGS",
    "EXPERIMENTS_INDEX",
    "CHAIN_EPOCHS",
    "RUNS_DIR",
    "REPORTS_DIR",
)


def ensure_dirs() -> None:
    """Create the non-frozen working directories if they are missing.

    Safe to call repeatedly. Deliberately does *not* create ``data/frozen``:
    only :mod:`titanic_agent.freeze` may bring that directory into existence.
    """
    for directory in (RAW_DIR, REPORTS_DIR, EXPERIMENTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
