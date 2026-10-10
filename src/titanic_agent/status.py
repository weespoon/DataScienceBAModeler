"""Where is the pipeline? Derived from artifacts on disk, not from stored state.

Deliberately dumb: it presence-checks the artifact each step is supposed to
leave behind. There is no DAG, no resume engine and no run database, because
seven linear steps with on-disk outputs do not need one. The artifact *is* the
state.

This answers "where am I". It does not answer "what did the agent do" — that is
the run journal's job, and only the journal supports the honesty question.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import paths


@dataclass(frozen=True)
class StepStatus:
    """Whether one step has left its artifact behind."""

    number: int
    name: str
    done: bool
    detail: str


def _manifest_sha() -> str | None:
    """The frozen fingerprint, or None when the data is not frozen."""
    try:
        return str(json.loads(paths.MANIFEST.read_text(encoding="utf-8"))["dataset_sha256"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def _ledger_steps() -> dict[str, int]:
    """Row count per pipeline step in the ledger, empty when there is none."""
    from . import results_log

    frame = results_log.read_log()
    if frame.empty:
        return {}
    return frame["step"].value_counts().to_dict()


def pipeline_status() -> list[StepStatus]:
    """One :class:`StepStatus` per pipeline step, in order."""
    from . import pull

    ledger = _ledger_steps()
    frozen_sha = _manifest_sha()

    def artifact(path: Path) -> tuple[bool, str]:
        return (True, str(path.relative_to(paths.ROOT))) if path.is_file() else (False, "missing")

    rows: list[StepStatus] = []

    raw_ok = pull.verify()
    rows.append(
        StepStatus(1, "pull", raw_ok, "data/raw verified" if raw_ok else "no usable raw data")
    )

    done, detail = artifact(paths.REPORTS_DIR / "01_review.md")
    rows.append(StepStatus(2, "review", done, detail))

    done, detail = artifact(paths.SPLIT_PLAN)
    rows.append(StepStatus(3, "split", done, detail))

    rows.append(
        StepStatus(
            4,
            "freeze",
            frozen_sha is not None,
            f"frozen sha={frozen_sha[:12]}" if frozen_sha else "not frozen",
        )
    )

    done, detail = artifact(paths.REPORTS_DIR / "02_explore.md")
    rows.append(StepStatus(5, "explore", done, detail))

    tuned = ledger.get("tune", 0)
    rows.append(StepStatus(6, "tune", tuned > 0, f"{tuned} ledger row(s)"))

    ensembled = ledger.get("ensemble", 0)
    rows.append(StepStatus(7, "ensemble", ensembled > 0, f"{ensembled} ledger row(s)"))

    return rows


def next_step() -> StepStatus | None:
    """The first step that has not left its artifact, or None when all have."""
    return next((row for row in pipeline_status() if not row.done), None)


def main(argv: list[str] | None = None) -> int:
    """``make status`` entry point."""
    import argparse

    from . import receipts

    parser = argparse.ArgumentParser(description="Show pipeline state.")
    parser.add_argument("--next", action="store_true", help="print only the next step to run")
    args = parser.parse_args(argv)

    if args.next:
        upcoming = next_step()
        print("pipeline complete" if upcoming is None else f"{upcoming.number} {upcoming.name}")
        return 0

    for row in pipeline_status():
        mark = "x" if row.done else " "
        print(f"  [{mark}] {row.number} {row.name:<9} {row.detail}")

    frozen_sha = _manifest_sha()
    if frozen_sha is not None:
        reads = len(receipts.receipts_for(frozen_sha))
        note = "unread" if reads == 0 else f"READ {reads}x"
        print(f"\n  holdout ({frozen_sha[:12]}): {note}")

    upcoming = next_step()
    print(f"  next: {'pipeline complete' if upcoming is None else f'make {upcoming.name}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
