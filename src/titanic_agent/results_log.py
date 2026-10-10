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
import hashlib
import io
import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import locking, paths, run_log

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
    #: The attestation. Rule 2 cannot be fully prevented or reliably detected --
    #: a stray read of holdout.csv in a step with no business reading it leaves
    #: no trace (atime is unreliable on APFS). So every run signs a claim
    #: instead, which turns a silent omission into an attributable false
    #: statement. Appended at the end so older rows stay readable.
    "holdout_read",
    "notes",
    #: Groups the rows written by one `make loop` iteration. Without it, "what
    #: did pass 2 do?" is unanswerable, because each model gets its own run_id.
    "pass_id",
    #: Tamper-evidence. SHA-256 of the *previous* data line exactly as written,
    #: so deleting a row or editing one breaks the next row's link. "Append-only"
    #: was an honour system until this column existed: nothing would have
    #: noticed a quietly removed result. See :func:`verify_chain`.
    "prev_sha",
    #: Which experiment owns this row. Scores are comparable across experiments
    #: only where `frozen_sha256` also matches — a different working set means a
    #: different question was being asked.
    #:
    #: Appended here rather than beside `pass_id`, where it reads better, because
    #: ensure_log() refuses a header that is not the old one plus new columns at
    #: the end. That rule exists so old rows stay readable, and it caught this.
    "experiment_id",
)


def _utc_now() -> str:
    """Current time as an ISO-8601 UTC timestamp with second precision."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _run_git(*args: str) -> str | None:
    """Run a git command in the repo root, or return None if it fails."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=paths.ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout


def _git_sha(short: bool = True) -> str:
    """Return the current git commit SHA, or ``"unknown"`` outside a repo.

    A ``-dirty`` suffix is appended when the working tree has uncommitted
    changes, because without it the column is a quiet lie: the bare SHA claims
    the row was produced by committed code that may not be what ran. A score
    you cannot tie to code is not reproducible, so the ledger has to say so.
    """
    raw = _run_git("rev-parse", *(["--short"] if short else []), "HEAD")
    if raw is None:
        return "unknown"
    sha = raw.strip()
    if not sha:
        return "unknown"
    status = _run_git("status", "--porcelain")
    if status is None:
        return sha
    return f"{sha}-dirty" if status.strip() else sha


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


def _data_lines(path: Path) -> list[str]:
    """Every data line of the ledger, header excluded, newlines stripped."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [line for line in lines[1:] if line.strip()]


def _line_sha(line: str) -> str:
    """SHA-256 of one ledger line, exactly as written."""
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def _current_experiment_id() -> str:
    """The open experiment's id, or "" when none is open.

    Imported lazily and failure-tolerantly: a row that cannot name its
    experiment is still worth logging. Rule 3 says a run not in the ledger did
    not happen, and that outranks tidy provenance.
    """
    try:
        from . import experiments

        current = experiments.current_experiment()
    except Exception:
        return ""
    return current.experiment_id if current else ""


def ensure_log(path: Path | None = None) -> Path:
    """Create the ledger, or migrate it when columns have been appended.

    Migration rewrites the header and pads existing rows with empty values for
    the new columns. No row is removed and no value changes — it is the
    mechanical form of the CLAUDE.md rule that new columns go at the end so old
    rows stay readable.

    A migration necessarily rewrites every line, which breaks the hash chain:
    tamper-evidence requires recomputing from content, while schema stability
    requires not recomputing, and those genuinely conflict. Rather than pretend
    otherwise, the chain is rebuilt and the rebuild is **recorded** in
    ``experiments/chain_epochs.jsonl``, which ``make audit`` reports.

    What that costs, stated plainly: rows written before a rebuild can no longer
    be verified against their original bytes. The chain proves nothing was
    deleted or edited *since the last rebuild*. A rebuild is therefore an
    auditable event with a reason attached, not a silent repair — the question
    "why did the chain restart here?" is one the audit makes someone answer.
    """
    log_path = Path(path) if path is not None else paths.RESULTS_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_path.exists() or log_path.stat().st_size == 0:
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(COLUMNS)
        return log_path

    with log_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        handle.seek(0)
        existing = next(csv.reader(handle), [])

    if existing == list(COLUMNS):
        return log_path
    if existing != list(COLUMNS[: len(existing)]):
        raise ValueError(
            f"{log_path} has an unrecognised header {existing}. Columns may only be "
            f"appended, never reordered or removed; refusing to guess."
        )

    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in COLUMNS})

    _rebuild_chain(log_path)
    _record_chain_epoch(
        reason=f"schema migration: {len(existing)} -> {len(COLUMNS)} columns",
        added=[c for c in COLUMNS if c not in existing],
        rows=len(rows),
        log_path=log_path,
    )
    return log_path


def _render_row(row: dict[str, Any]) -> str:
    """Render one row exactly as :func:`append_run` would write it, unterminated."""
    buffer = io.StringIO()
    csv.DictWriter(buffer, fieldnames=COLUMNS).writerow(
        {column: row.get(column, "") for column in COLUMNS}
    )
    return buffer.getvalue().rstrip("\r\n")


def _rebuild_chain(log_path: Path) -> None:
    """Recompute every ``prev_sha`` so the rewritten file is self-consistent.

    Done entirely in memory. The first attempt re-read the file after each row
    to find the previous line, which raced its own buffering and produced a
    chain that failed its own verification.
    """
    with log_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    rendered: list[str] = []
    previous: str | None = None
    for row in rows:
        # Only rows that already carried a chain stay chained. Rows that predate
        # it keep their blank prev_sha and remain unverifiable, which is the
        # truth about them -- back-filling would claim a guarantee we never had.
        if (row.get("prev_sha") or "").strip():
            row["prev_sha"] = _line_sha(previous) if previous is not None else "genesis"
        line = _render_row(row)
        rendered.append(line)
        previous = line

    with log_path.open("w", newline="", encoding="utf-8") as handle:
        handle.write(",".join(COLUMNS) + "\r\n")
        for line in rendered:
            handle.write(line + "\r\n")


def _record_chain_epoch(*, reason: str, added: list[str], rows: int, log_path: Path) -> None:
    """Append an auditable record that the chain was rebuilt, and why."""
    entry = {
        "timestamp_utc": _utc_now(),
        "reason": reason,
        "columns_added": added,
        "rows_at_rebuild": rows,
        "ledger": str(log_path),
    }
    try:
        paths.CHAIN_EPOCHS.parent.mkdir(parents=True, exist_ok=True)
        with (
            locking.locked(paths.CHAIN_EPOCHS),
            paths.CHAIN_EPOCHS.open("a", encoding="utf-8") as handle,
        ):
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    except OSError:
        pass  # the migration itself matters more than its footnote


def read_chain_epochs(path: Path | None = None) -> list[dict[str, Any]]:
    """Every recorded chain rebuild, oldest first."""
    target = Path(path) if path is not None else paths.CHAIN_EPOCHS
    found: list[dict[str, Any]] = []
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return found
    for line in text.splitlines():
        if line.strip():
            try:
                found.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return found


def verify_chain(path: Path | None = None) -> dict[str, Any]:
    """Check the ledger's hash chain and report what it found.

    Returns a dict with ``ok``, ``chained_rows``, ``unchained_rows`` and a list
    of ``problems``. A row whose ``prev_sha`` does not match the hash of the
    line above it means a row was deleted or edited between them.

    Rows written before the chain existed carry no ``prev_sha``; they are
    counted as unchained, not reported as tampered. Unverifiable is not the same
    as proven bad, and saying otherwise would make the audit cry wolf.
    """
    log_path = Path(path) if path is not None else paths.RESULTS_LOG
    lines = _data_lines(log_path)
    problems: list[str] = []
    chained = unchained = 0

    for index, line in enumerate(lines):
        fields = next(csv.reader([line]), [])
        record = dict(zip(COLUMNS, fields, strict=False))
        prev_sha = (record.get("prev_sha") or "").strip()
        if not prev_sha:
            unchained += 1
            continue
        chained += 1
        expected = "genesis" if index == 0 else _line_sha(lines[index - 1])
        if prev_sha != expected:
            problems.append(
                f"row {index + 1} ({record.get('run_id', '?')}): prev_sha does not match "
                f"the line above it — a row was deleted or edited"
            )

    return {
        "ok": not problems,
        "chained_rows": chained,
        "unchained_rows": unchained,
        "problems": problems,
    }


def append_run(
    *,
    step: str,
    model: str,
    params: dict[str, Any] | None = None,
    cv_score: float | None = None,
    cv_std: float | None = None,
    cv_folds: int | None = None,
    seed: int = paths.SEED,
    holdout_read: bool = False,
    notes: str = "",
    run_id: str | None = None,
    pass_id: str | None = None,
    experiment_id: str | None = None,
    path: Path | None = None,
) -> str:
    """Append exactly one row to the ledger and return its ``run_id``.

    ``params`` is serialised as a compact JSON string so the column stays a
    single CSV field and round-trips through :func:`read_log`.

    ``holdout_read`` is an attestation, not a measurement: nothing verifies it.
    Sign it honestly, including when the honest answer is inconvenient. A wrong
    score you reported is recoverable; a false attestation makes every other row
    in the file untrustworthy.
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
        "holdout_read": bool(holdout_read),
        "notes": notes.replace("\n", " ").strip(),
        "pass_id": pass_id or run_log.current_pass_id(),
        "experiment_id": experiment_id if experiment_id is not None else _current_experiment_id(),
    }
    # Read-hash-append is one critical section. Interleaved, two processes
    # compute the same prev_sha and the chain forks -- which verify_chain cannot
    # distinguish from a deleted row, so the audit would cry tampering about a
    # ledger nobody touched.
    with locking.locked(log_path):
        previous = _data_lines(log_path)
        row["prev_sha"] = _line_sha(previous[-1]) if previous else "genesis"
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
