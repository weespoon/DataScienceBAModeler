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

#: Default random seed. Change it only with a recorded reason — the split that
#: produced the frozen holdout is reproducible from this value plus MANIFEST.json.
SEED: int = 20251010


def ensure_dirs() -> None:
    """Create the non-frozen working directories if they are missing.

    Safe to call repeatedly. Deliberately does *not* create ``data/frozen``:
    only :mod:`titanic_agent.freeze` may bring that directory into existence.
    """
    for directory in (RAW_DIR, REPORTS_DIR, EXPERIMENTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
