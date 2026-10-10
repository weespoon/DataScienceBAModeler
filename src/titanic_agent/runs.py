"""Per-pass run directories — ``runs/20261010-1435EDT/``.

Every ``make loop`` used to overwrite ``reports/02_explore.md`` and all seven
charts. Three passes were in the ledger; only the last pass's artifacts existed.
A run directory keeps each pass's output beside the rows it produced.

**Artifacts move here. Evidence does not.** The split is the same one that
separates a log from a record:

- ``runs/<id>/`` holds this pass's charts, step reports, submission and summary.
  Regenerable, per-pass, safe to delete.
- ``experiments/results.csv`` and ``experiments/holdout_reads.json`` stay global
  and append-only.

That second point is load-bearing. If receipts moved into run directories,
"the holdout is read once per frozen dataset" would silently become "once per
*run*": each new pass would see an empty receipts file, read the holdout, and
pass its own audit. Unlimited reads, every one of them green. Splitting the
ledger is less catastrophic but still wrong — you could no longer compare across
passes, and the hash chain would only protect within one, so deleting a whole
run would become undetectable.

The id doubles as the ``pass_id`` stamped on ledger rows and journal entries, so
``ls runs/20261010-1435EDT/`` and ``make journal --pass-id 20261010-1435EDT``
name the same thing.

On the timestamp: local time, because the directory list is for a human to read.
The zone is the real abbreviation (``EDT``/``EST``) rather than a generic ``ET``
— which incidentally solves the problem a generic label would create, since
01:30 happens twice on the night the clocks go back and ``EDT``/``EST``
distinguishes the two. The exact UTC instant is recorded in ``metadata.json``
either way; the directory name is a label, not the authority.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import paths

#: Environment variable that pins one id across the separate processes of a
#: single ``make loop``. Set by the Makefile; see the ``loop`` target.
RUN_ID_ENV = "TITANIC_PASS_ID"

#: What a minted id looks like: local date, local time, zone abbreviation, and
#: an optional ``-2`` suffix when two passes start in the same minute.
RUN_ID_PATTERN = re.compile(r"^\d{8}-\d{4}[A-Z]{2,5}(-\d+)?$")

#: Files a pass writes into its own directory rather than into ``reports/``.
#: ``01_review.md`` is deliberately absent: review belongs to ``make bootstrap``,
#: which runs once, so it has nothing to be clobbered by.
PASS_ARTIFACTS: tuple[str, ...] = ("02_explore", "03_ensemble", "submission")


def mint_run_id(now: datetime | None = None, *, runs_dir: Path | None = None) -> str:
    """Make a fresh, human-readable id that no existing run directory uses."""
    local = (now or datetime.now()).astimezone()
    base = f"{local:%Y%m%d-%H%M}{local.strftime('%Z')}"
    parent = Path(runs_dir) if runs_dir is not None else paths.RUNS_DIR
    candidate, suffix = base, 1
    while (parent / candidate).exists():
        suffix += 1
        candidate = f"{base}-{suffix}"
    return candidate


def current_run_id() -> str:
    """The id for this pass: from the environment, or freshly minted.

    ``make loop`` exports one so explore, tune and ensemble agree. A single step
    run by hand genuinely is its own pass and gets its own id.
    """
    return os.environ.get(RUN_ID_ENV) or mint_run_id()


def run_dir(run_id: str | None = None, *, create: bool = True) -> Path:
    """The directory for one pass, created by default."""
    directory = paths.RUNS_DIR / (run_id or current_run_id())
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_metadata(run_id: str | None = None, **extra: Any) -> Path:
    """Record the exact UTC instant and anything else worth pinning.

    The directory name is local time for readability; this is the authority.
    """
    resolved = run_id or current_run_id()
    directory = run_dir(resolved)
    local = datetime.now().astimezone()
    payload = {
        "run_id": resolved,
        "started_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "started_local": local.replace(microsecond=0).isoformat(),
        "timezone": local.strftime("%Z"),
        **extra,
    }
    path = directory / "metadata.json"
    if path.exists():  # a later step in the same pass; keep the first start time
        with contextlib.suppress(json.JSONDecodeError, OSError):
            payload = {**json.loads(path.read_text(encoding="utf-8")), **extra}
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def publish(source: Path, run_id: str | None = None) -> Path:
    """Copy an artifact written in a run directory into ``reports/`` as latest.

    ``reports/`` stays the always-current view, so every path already written
    into the skills and the README keeps working. The run directory is the
    archive; this is the shortcut.
    """
    import shutil

    source = Path(source)
    paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = paths.REPORTS_DIR / source.name
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def list_runs(runs_dir: Path | None = None) -> list[Path]:
    """Every run directory, oldest first. The names sort chronologically."""
    parent = Path(runs_dir) if runs_dir is not None else paths.RUNS_DIR
    if not parent.is_dir():
        return []
    return sorted(p for p in parent.iterdir() if p.is_dir())


def summarize(run_id: str | None = None) -> Path:
    """Write ``run.md`` — what this one pass did, readable without opening anything.

    Reads the global ledger and journal and filters to this pass. The per-run
    view is derived, never a separate copy: two records of the same events can
    disagree, and then neither can be trusted.
    """
    from . import experiments, results_log, run_log

    # One pass just finished, so charge it. summarize() is the only thing that
    # runs exactly once per `make loop`, which makes it the honest counter —
    # and a cap nobody counts is a number in a file pretending to be a limit.
    try:
        experiments.record_spend("loop_passes", 1)
        over = experiments.over_budget()
        if over:
            run_log.record(
                "decision",
                step="experiment",
                detail=f"budget reached: {over} — stop, or ask the human to raise the cap",
            )
    except (OSError, ValueError):
        pass  # a pass with no experiment open still deserves its summary

    # Minting here would create an empty directory and summarise nothing. With
    # no id and no pass in flight, the useful answer is the most recent pass.
    if run_id is None and RUN_ID_ENV not in os.environ:
        existing = list_runs()
        run_id = existing[-1].name if existing else None
    resolved = run_id or current_run_id()
    directory = run_dir(resolved)

    frame = results_log.read_log()
    rows = frame[frame["pass_id"].astype(str) == resolved] if not frame.empty else frame
    entries = [e for e in run_log.read_journal() if e.get("pass_id") == resolved]

    lines = [
        f"# Run {resolved}",
        "",
        "Generated by `make loop`. One pass over steps 5-7.",
        "",
    ]

    metadata_path = directory / "metadata.json"
    if metadata_path.is_file():
        try:
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            lines += [
                f"Started {meta.get('started_local', '?')} ({meta.get('started_utc', '?')}).",
                "",
            ]
        except json.JSONDecodeError:
            pass

    if len(rows):
        scored = rows.dropna(subset=["cv_score"])
        lines += ["## Scores", "", "| step | model | CV | ± |", "| --- | --- | --- | --- |"]
        for _, row in scored.sort_values("cv_score", ascending=False).iterrows():
            std = row["cv_std"]
            lines.append(
                f"| {row['step']} | `{row['model']}` | {float(row['cv_score']):.4f} | "
                f"{'' if std != std else f'{float(std):.4f}'} |"
            )
        failed = rows[rows["cv_score"].isna()]
        if len(failed):
            lines += ["", f"**{len(failed)} run(s) failed** and were logged as such:", ""]
            lines += [f"- `{r['model']}` — {r['notes']}" for _, r in failed.iterrows()]
        lines += ["", f"{len(rows)} ledger row(s) carry this pass id.", ""]
    else:
        lines += ["## Scores", "", "No ledger rows for this pass.", ""]

    decisions = [e for e in entries if str(e.get("level")) in ("decision", "error")]
    lines += ["## Decisions and refusals", ""]
    if decisions:
        lines += [
            f"- **{e.get('event')}** ({e.get('step') or 'pass'}) — {e.get('detail')}"
            for e in decisions
        ]
    else:
        lines.append("_None — the pass ran to completion with nothing needing a person._")

    lines += [
        "",
        "## Artifacts",
        "",
        *(f"- `{p.name}`" for p in sorted(directory.iterdir()) if p.name != "run.md"),
        "",
        "## Evidence",
        "",
        "Scores live in the global append-only `experiments/results.csv`, not here —",
        "a per-run ledger could not be compared across passes, and its hash chain",
        "would not notice a whole run being deleted. Holdout receipts are global for",
        "a sharper reason: per-run receipts would turn *one read per frozen dataset*",
        "into *one read per run*.",
        "",
        f"- `make log` — the full ledger ({len(frame)} rows)",
        f"- `make journal PASS={resolved}` — this pass's journal "
        "(options to make are variables, not flags)",
        "- `make audit` — whether the three rules held",
        "",
    ]

    path = directory / "run.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """``make summary`` / ``make runs``."""
    import argparse

    parser = argparse.ArgumentParser(description="Per-pass run directories.")
    parser.add_argument("--summarize", action="store_true", help="write this pass's run.md")
    parser.add_argument("--run-id", default=None, help="which pass (default: current)")
    args = parser.parse_args(argv)

    if args.summarize:
        path = summarize(args.run_id)
        shown = path.relative_to(paths.ROOT) if path.is_relative_to(paths.ROOT) else path
        print(f"run summary -> {shown}")
        return 0

    found = list_runs()
    if not found:
        print(f"no runs yet ({paths.RUNS_DIR})")
        return 0
    for directory in found:
        artifacts = len([p for p in directory.iterdir() if p.is_file()])
        print(f"  {directory.name:<24} {artifacts} artifact(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
