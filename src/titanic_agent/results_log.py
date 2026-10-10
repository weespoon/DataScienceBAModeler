"""Append-only experiment ledger (``experiments/results.csv``).

This is the one module that is fully implemented rather than stubbed: it is
plumbing, not modeling, and every other step depends on it. The rule from
CLAUDE.md is absolute — a run that is not in this file did not happen.

Usage::

    from titanic_agent import results_log

    run_id = results_log.append_run(
        step="tune",
        model="RandomForestClassifier",
        params={"n_estimators": 400, "max_depth": 6},
        cv_score=0.8327,
        cv_std=0.0184,
        cv_folds=5,
        seed=20251010,
        notes="randomized search, 25 candidates",
    )
"""

from __future__ import annotations

import csv
import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import paths

#: Column order of the ledger. Append new columns at the end only — older rows
#: must stay readable.
COLUMNS: tuple[str, ...] = (
    "run_id",
    "timestamp_utc",
    "step",
    "model",
    "params",
    "cv_score",
    "cv_std",
    "cv_folds",
    "seed",
    "frozen_sha256",
    "git_sha",
    "notes",
)


def _utc_now() -> str:
    """Current time as an ISO-8601 UTC timestamp with second precision."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _git_sha(short: bool = True) -> str:
    """Return the current git commit SHA, or ``"unknown"`` outside a repo."""
    cmd = ["git", "rev-parse", *(["--short"] if short else []), "HEAD"]
    try:
        out = subprocess.run(
            cmd,
            cwd=paths.ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def _frozen_sha() -> str:
    """Return the frozen-dataset fingerprint recorded in ``MANIFEST.json``.

    Returns ``"unfrozen"`` when no manifest exists yet, which makes rows logged
    before :mod:`titanic_agent.freeze` obvious in hindsight.
    """
    try:
        manifest = json.loads(paths.MANIFEST.read_text())
    except (OSError, json.JSONDecodeError):
        return "unfrozen"
    value = manifest.get("dataset_sha256") or manifest.get("sha256")
    return str(value) if value else "unfrozen"


def ensure_log(path: Path | None = None) -> Path:
    """Create the ledger with its header row if it does not exist yet."""
    log_path = Path(path) if path is not None else paths.RESULTS_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists() or log_path.stat().st_size == 0:
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(COLUMNS)
    return log_path


def append_run(
    *,
    step: str,
    model: str,
    params: dict[str, Any] | None = None,
    cv_score: float | None = None,
    cv_std: float | None = None,
    cv_folds: int | None = None,
    seed: int = paths.SEED,
    notes: str = "",
    run_id: str | None = None,
    path: Path | None = None,
) -> str:
    """Append exactly one row to the ledger and return its ``run_id``.

    ``params`` is serialised as a compact JSON string so the column stays a
    single CSV field and round-trips through :func:`read_log`.
    """
    log_path = ensure_log(path)
    resolved_run_id = run_id or f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
    row = {
        "run_id": resolved_run_id,
        "timestamp_utc": _utc_now(),
        "step": step,
        "model": model,
        "params": json.dumps(params or {}, sort_keys=True, separators=(",", ":"), default=str),
        "cv_score": "" if cv_score is None else f"{cv_score:.6f}",
        "cv_std": "" if cv_std is None else f"{cv_std:.6f}",
        "cv_folds": "" if cv_folds is None else int(cv_folds),
        "seed": seed,
        "frozen_sha256": _frozen_sha(),
        "git_sha": _git_sha(),
        "notes": notes.replace("\n", " ").strip(),
    }
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=COLUMNS).writerow(row)
    return resolved_run_id


def read_log(path: Path | None = None):
    """Return the ledger as a :class:`pandas.DataFrame` with ``params`` parsed.

    An empty (header-only or missing) ledger yields an empty frame with the
    right columns, so callers can filter without guarding for emptiness.
    """
    import pandas as pd

    log_path = Path(path) if path is not None else paths.RESULTS_LOG
    if not log_path.exists():
        return pd.DataFrame(columns=list(COLUMNS))
    frame = pd.read_csv(log_path)
    if frame.empty:
        return pd.DataFrame(columns=list(COLUMNS))
    frame["params"] = frame["params"].fillna("{}").map(json.loads)
    return frame


def best_run(step: str | None = None, path: Path | None = None):
    """Return the row with the highest ``cv_score``, optionally within a step.

    Returns ``None`` when the ledger holds no scored runs.
    """
    frame = read_log(path)
    if step is not None:
        frame = frame[frame["step"] == step]
    frame = frame.dropna(subset=["cv_score"])
    if frame.empty:
        return None
    return frame.loc[frame["cv_score"].astype(float).idxmax()]


def main(argv: list[str] | None = None) -> int:
    """``python -m titanic_agent.results_log`` — print the ledger."""
    import argparse

    parser = argparse.ArgumentParser(description="Show the experiment ledger.")
    parser.add_argument("--best", action="store_true", help="show only the best run")
    parser.add_argument("--step", default=None, help="filter to one pipeline step")
    args = parser.parse_args(argv)

    if args.best:
        row = best_run(step=args.step)
        print("no scored runs yet" if row is None else row.to_string())
        return 0

    frame = read_log()
    if args.step:
        frame = frame[frame["step"] == args.step]
    print(
        f"no runs logged yet ({paths.RESULTS_LOG})" if frame.empty else frame.to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
