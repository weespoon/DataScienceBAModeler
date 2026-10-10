"""The run journal (``experiments/run_log.jsonl``) — what the *agent* did.

This is not the experiment ledger. ``results.csv`` records **models**: what was
tried, what it scored. Nothing in it records the agent's own conduct — which
steps ran, in what order, what was skipped and why, what it decided when
CLAUDE.md said "prefer the cheap, reversible option and keep going."

Derived status answers *where am I*. This answers *what did you do*, and only the
second one supports the question in the README. An autonomous pass that leaves
no journal is a pass nobody can review.

Append-only, one JSON object per line, so a partially written file still parses
up to the last complete line — which matters when the thing being journalled is
a run that may die partway through.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import paths

#: Event kinds. Kept small on purpose; a vocabulary nobody can remember gets
#: used inconsistently and the journal stops being comparable across runs.
EVENTS: tuple[str, ...] = (
    "step_started",
    "step_finished",
    "step_skipped",
    "step_failed",
    "decision",
    "refused",
    "note",
)


def record(
    event: str,
    *,
    step: str = "",
    detail: str = "",
    path: Path | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Append one journal entry and return it.

    Args:
        event: One of :data:`EVENTS`. An unknown value is recorded rather than
            rejected — losing the entry would be worse than an odd label.
        step: Pipeline step the entry belongs to.
        detail: One line of human-readable context.
        path: Journal to append to; defaults to ``experiments/run_log.jsonl``.
        **fields: Anything else worth keeping, serialised as-is.

    A ``refused`` entry is the important one: it is how a tripwire
    (unfreezing, ``--score-holdout``) leaves a trace even though, by design,
    nothing happened.
    """
    journal = Path(path) if path is not None else paths.RUN_LOG
    journal.parent.mkdir(parents=True, exist_ok=True)

    entry: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "event": event,
        "step": step,
        "detail": detail.replace("\n", " ").strip(),
        "pid": os.getpid(),
    }
    entry.update(fields)

    with journal.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
    return entry


def read_journal(path: Path | None = None) -> list[dict[str, Any]]:
    """Return every parseable entry, oldest first.

    A truncated final line is skipped rather than fatal: a journal written by a
    run that was killed mid-write should still be readable up to that point.
    """
    journal = Path(path) if path is not None else paths.RUN_LOG
    entries: list[dict[str, Any]] = []
    try:
        text = journal.read_text(encoding="utf-8")
    except OSError:
        return entries
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            entries.append(parsed)
    return entries


def main(argv: list[str] | None = None) -> int:
    """``python -m titanic_agent.run_log`` — print the journal."""
    import argparse

    parser = argparse.ArgumentParser(description="Show the agent run journal.")
    parser.add_argument("--event", default=None, help="filter to one event kind")
    parser.add_argument("--step", default=None, help="filter to one pipeline step")
    parser.add_argument("--tail", type=int, default=0, help="show only the last N entries")
    args = parser.parse_args(argv)

    entries = read_journal()
    if args.event:
        entries = [e for e in entries if e.get("event") == args.event]
    if args.step:
        entries = [e for e in entries if e.get("step") == args.step]
    if args.tail:
        entries = entries[-args.tail :]

    if not entries:
        print(f"no journal entries yet ({paths.RUN_LOG})")
        return 0
    for entry in entries:
        print(
            f"{entry.get('timestamp_utc', '?')}  {entry.get('event', '?'):<14}"
            f"{entry.get('step', ''):<10} {entry.get('detail', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
