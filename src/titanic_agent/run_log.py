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
import uuid
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

#: Severity, least to most urgent. This is the channel split: `debug` is the
#: play-by-play nobody reads twice, `info` is the progress you glance at,
#: `decision` and `error` are the things that need a person.
LEVELS: tuple[str, ...] = ("debug", "info", "decision", "error")

#: Default severity per event kind, so callers get it right by not thinking
#: about it. `refused` is never below `decision`: a tripwire that fired is
#: exactly the thing a human must see.
DEFAULT_LEVEL: dict[str, str] = {
    "step_started": "debug",
    "step_finished": "info",
    "step_skipped": "info",
    "step_failed": "error",
    "decision": "decision",
    "refused": "decision",
    "note": "info",
}

#: Generated once per process. An ad-hoc single-step run genuinely is its own
#: pass; only `make loop` has reason to span several processes, and it does that
#: by exporting TITANIC_PASS_ID.
_PROCESS_PASS_ID = f"adhoc-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:4]}"

#: Environment variable `make loop` sets so one iteration shares an id.
PASS_ID_ENV = "TITANIC_PASS_ID"


def current_pass_id() -> str:
    """The id grouping one loop iteration across its separate processes.

    ``make loop`` runs explore, tune and ensemble as three python invocations.
    Without a shared id, "what did pass 2 do?" is unanswerable — which is why
    there is no per-run report yet. This is the enabler for one.
    """
    return os.environ.get(PASS_ID_ENV) or _PROCESS_PASS_ID


def record(
    event: str,
    *,
    step: str = "",
    detail: str = "",
    level: str | None = None,
    pass_id: str | None = None,
    path: Path | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Append one journal entry and return it.

    Args:
        event: One of :data:`EVENTS`. An unknown value is recorded rather than
            rejected — losing the entry would be worse than an odd label.
        step: Pipeline step the entry belongs to.
        detail: One line of human-readable context.
        level: Severity; defaults from :data:`DEFAULT_LEVEL` for the event kind.
        pass_id: Groups one loop iteration; defaults to :func:`current_pass_id`.
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
        "level": level or DEFAULT_LEVEL.get(event, "info"),
        "pass_id": pass_id or current_pass_id(),
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
    parser.add_argument("--pass-id", default=None, help="filter to one loop iteration")
    parser.add_argument(
        "--level",
        default="info",
        choices=LEVELS,
        help="minimum severity to show (default: info)",
    )
    parser.add_argument("--all", action="store_true", help="show everything, including debug")
    parser.add_argument("--decisions", action="store_true", help="only what needed a person")
    parser.add_argument("--tail", type=int, default=0, help="show only the last N entries")
    args = parser.parse_args(argv)

    minimum = "debug" if args.all else ("decision" if args.decisions else args.level)
    floor = LEVELS.index(minimum)

    def rank(entry: dict[str, Any]) -> int:
        """Severity rank; an unrecognised level sorts as `info` rather than crashing."""
        level = str(entry.get("level", "info"))
        return LEVELS.index(level) if level in LEVELS else LEVELS.index("info")

    all_entries = read_journal()
    if not all_entries:
        print(f"no journal entries yet ({paths.RUN_LOG})")
        return 0

    entries = [e for e in all_entries if rank(e) >= floor]
    if args.event:
        entries = [e for e in entries if e.get("event") == args.event]
    if args.step:
        entries = [e for e in entries if e.get("step") == args.step]
    if args.pass_id:
        entries = [e for e in entries if e.get("pass_id") == args.pass_id]
    if args.tail:
        entries = entries[-args.tail :]

    if not entries:
        # Distinguish "the journal is empty" from "everything is quieter than
        # this" — the second should say so, and name the way to see it.
        print(
            f"nothing at level {minimum} or above; "
            f"{len(all_entries)} quieter entry(s) exist. Use --all to see them."
        )
        return 0
    for entry in entries:
        print(
            f"{entry.get('timestamp_utc', '?')}  {str(entry.get('level', '?')).upper():<9}"
            f"{entry.get('event', '?'):<14}{entry.get('step', ''):<10} "
            f"{entry.get('detail', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
