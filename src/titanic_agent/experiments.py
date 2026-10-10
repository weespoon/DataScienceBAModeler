"""An experiment: one budgeted attempt at the problem, ending in its own score.

The data layer did not need to change for this. ``data/frozen/`` already held a
three-way split; we had only been reading it as two:

===================  =====  ====================================================
``train.csv``          712  the **working set** — each experiment re-splits it
``holdout.csv``        179  the **final test** — locked, one read for the whole
                            program, and still a tripwire
``test.csv``           418  Kaggle's unlabelled set, for submissions
===================  =====  ====================================================

So an experiment draws its own stratified 570/142 split from the working set,
tunes on the 570, and reads its own 142-row validation split **once, at the end,
with no human involved.** That is the autonomous per-experiment number. It is
cheap because a validation split is re-drawable; the next experiment draws a
different one.

The final test is what stays expensive, and why the two cannot be merged: if
experiment 2's direction is chosen using experiment 1's final-test score, the
final test has quietly become a validation set, and after a few experiments the
"final" number is a selection statistic rather than an estimate of
generalisation. Keeping them separate is what lets experiments chain freely
while one number stays honest.

What carries forward is deliberately not the scores alone. Every experiment
inherits :mod:`titanic_agent.findings` from all its predecessors — claims with
the ``run_id``s behind them — so experiment N+1 starts from what is known and
what is still untried, rather than re-deriving both.

Validation splits are never materialised as files. They are id lists over the
already-checksummed working set, so there is no second copy of the data to drift
out of step with the first.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import paths

#: Pins one experiment across the separate processes of a run.
EXPERIMENT_ID_ENV = "TITANIC_EXPERIMENT_ID"

#: Lifecycle. ``open`` accepts work; ``closed`` is final and keeps its number.
STATUSES: tuple[str, ...] = ("open", "closed", "abandoned")

#: Default budget. Caps exist so an autonomous program has an edge it cannot
#: wander past; the token cap is deliberately absent until we have measured one
#: research pass, because a number guessed now is a number we would change.
DEFAULT_CAPS: dict[str, Any] = {
    "kaggle_notebooks": 20,
    "papers": 10,
    "web_fetches": 0,  # v1 reads the installed package and the Kaggle API only
    "wall_clock_s": 1800,
    "loop_passes": 3,
}


@dataclass(frozen=True)
class Experiment:
    """One budgeted attempt, with its own validation split and its own score."""

    experiment_id: str = ""
    name: str = ""
    status: str = "open"
    seed: int = paths.SEED
    created_utc: str = ""
    closed_utc: str = ""
    #: Fingerprint of the frozen dataset this was drawn from. Two experiments
    #: sharing it are comparable; two that do not, are not.
    working_set_sha: str = ""
    train_ids: list[int] = field(default_factory=list)
    val_ids: list[int] = field(default_factory=list)
    caps: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_CAPS))
    spend: dict[str, Any] = field(default_factory=dict)
    best_cv: float | None = None
    val_score: float | None = None
    summary: str = ""

    @property
    def directory(self) -> Path:
        return paths.EXPERIMENTS_DIR / self.experiment_id

    @property
    def is_open(self) -> bool:
        return self.status == "open"


def _slug(name: str) -> str:
    cleaned = "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned or "experiment"


def next_experiment_id(name: str) -> str:
    """``exp-003-wide-search`` — ordinal first so the directory sorts by age."""
    existing = list_experiments()
    return f"exp-{len(existing) + 1:03d}-{_slug(name)}"


def list_experiments() -> list[Experiment]:
    """Every experiment, oldest first."""
    if not paths.EXPERIMENTS_DIR.is_dir():
        return []
    found: list[Experiment] = []
    for directory in sorted(paths.EXPERIMENTS_DIR.iterdir()):
        manifest = directory / "experiment.json"
        if directory.is_dir() and manifest.is_file():
            try:
                found.append(load_experiment(directory.name))
            except (ValueError, OSError):
                continue
    return found


def load_experiment(experiment_id: str) -> Experiment:
    """Read one experiment's manifest.

    Raises:
        FileNotFoundError: No such experiment.
        ValueError: The manifest is present but malformed.
    """
    manifest = paths.EXPERIMENTS_DIR / experiment_id / "experiment.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"no experiment {experiment_id!r} ({manifest})") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest} is not valid JSON: {exc}") from None
    try:
        return Experiment(**payload)
    except TypeError as exc:
        raise ValueError(f"{manifest} has unexpected fields: {exc}") from None


def _write(experiment: Experiment) -> Experiment:
    experiment.directory.mkdir(parents=True, exist_ok=True)
    (experiment.directory / "experiment.json").write_text(
        json.dumps(asdict(experiment), indent=2, default=str) + "\n", encoding="utf-8"
    )
    return experiment


def current_experiment() -> Experiment | None:
    """The experiment in flight: from the environment, else the newest open one."""
    pinned = os.environ.get(EXPERIMENT_ID_ENV)
    if pinned:
        return load_experiment(pinned)
    open_ones = [e for e in list_experiments() if e.is_open]
    return open_ones[-1] if open_ones else None


def draw_val_split(
    *, seed: int, val_frac: float = 0.2, frozen_dir: Path = paths.FROZEN_DIR
) -> tuple[list[int], list[int]]:
    """Split the working set into this experiment's train and validation ids.

    Stratified on the target, like the original split: the classes are 62/38 and
    an unstratified draw adds noise for nothing.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from . import freeze

    freeze.require_frozen(frozen_dir)
    working = pd.read_csv(Path(frozen_dir) / "train.csv")
    train, val = train_test_split(
        working,
        test_size=val_frac,
        stratify=working["Survived"],
        random_state=seed,
        shuffle=True,
    )
    return (
        sorted(int(i) for i in train["PassengerId"]),
        sorted(int(i) for i in val["PassengerId"]),
    )


def open_experiment(
    name: str,
    *,
    seed: int | None = None,
    val_frac: float = 0.2,
    caps: dict[str, Any] | None = None,
    frozen_dir: Path = paths.FROZEN_DIR,
) -> Experiment:
    """Start a new experiment: draw its validation split and set its budget.

    Args:
        name: Short human label, slugged into the id.
        seed: Split seed. Defaults to one derived from the ordinal, so each
            experiment gets a *different* validation split — reusing one seed
            across experiments would quietly recreate the problem the final test
            exists to avoid.
        val_frac: Share of the working set held back for this experiment.
        caps: Budget overrides; defaults to :data:`DEFAULT_CAPS`.
        frozen_dir: The frozen dataset; verified before anything is drawn.

    Raises:
        RuntimeError: Another experiment is still open. Two open experiments
            make "which val split am I using?" ambiguous, and the honest answer
            to an unfinished experiment is to finish or abandon it.
    """
    from . import freeze, run_log

    still_open = [e for e in list_experiments() if e.is_open]
    if still_open:
        raise RuntimeError(
            f"{still_open[-1].experiment_id} is still open. Close or abandon it first — "
            "two open experiments make the active validation split ambiguous."
        )

    manifest = freeze.require_frozen(frozen_dir)
    experiment_id = next_experiment_id(name)
    resolved_seed = seed if seed is not None else paths.SEED + len(list_experiments()) + 1
    train_ids, val_ids = draw_val_split(
        seed=resolved_seed, val_frac=val_frac, frozen_dir=frozen_dir
    )

    experiment = _write(
        Experiment(
            experiment_id=experiment_id,
            name=name,
            status="open",
            seed=resolved_seed,
            created_utc=datetime.now(UTC).replace(microsecond=0).isoformat(),
            working_set_sha=manifest.dataset_sha256,
            train_ids=train_ids,
            val_ids=val_ids,
            caps=dict(caps or DEFAULT_CAPS),
        )
    )
    run_log.record(
        "decision",
        step="experiment",
        detail=(
            f"opened {experiment_id}: {len(train_ids)} train / {len(val_ids)} val "
            f"(seed {resolved_seed}), caps {experiment.caps}"
        ),
    )
    return experiment


def load_frames(experiment: Experiment, *, frozen_dir: Path = paths.FROZEN_DIR):
    """This experiment's train and validation frames, sliced from the working set.

    Sliced at read time rather than written to disk: a second copy of frozen
    data is a second thing that can drift from the checksummed original.
    """
    import pandas as pd

    from . import freeze, tune

    freeze.require_frozen(frozen_dir)
    working = tune.add_derived(pd.read_csv(Path(frozen_dir) / "train.csv"))
    indexed = working.set_index("PassengerId", drop=False)
    return indexed.loc[experiment.train_ids], indexed.loc[experiment.val_ids]


def close_experiment(
    experiment_id: str | None = None,
    *,
    summary: str = "",
    status: str = "closed",
) -> Experiment:
    """Finish an experiment: freeze its numbers and render its findings.

    Raises:
        ValueError: Unknown status.
        FileNotFoundError: No such experiment.
    """
    from . import findings, results_log, run_log

    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}; expected one of {STATUSES}")

    experiment = load_experiment(experiment_id or _require_current().experiment_id)

    frame = results_log.read_log()
    rows = (
        frame[frame["experiment_id"].astype(str) == experiment.experiment_id]
        if not frame.empty and "experiment_id" in frame
        else frame.iloc[0:0]
    )
    scored = rows.dropna(subset=["cv_score"]) if len(rows) else rows
    best_cv = float(scored["cv_score"].astype(float).max()) if len(scored) else None

    closed = _write(
        Experiment(
            **{
                **asdict(experiment),
                "status": status,
                "closed_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
                "best_cv": best_cv if experiment.best_cv is None else experiment.best_cv,
                "summary": summary or experiment.summary,
            }
        )
    )

    (closed.directory / "findings.md").write_text(
        findings.render_markdown(closed.experiment_id), encoding="utf-8"
    )
    write_index()
    run_log.record(
        "decision",
        step="experiment",
        detail=f"{status} {closed.experiment_id}: best_cv={best_cv}, val={closed.val_score}",
    )
    return closed


def _require_current() -> Experiment:
    experiment = current_experiment()
    if experiment is None:
        raise RuntimeError(
            'no experiment is open. Run `make experiment-open NAME="..."` first — '
            "every score belongs to one, or it cannot be compared to anything."
        )
    return experiment


def inherited_findings(experiment_id: str | None = None):
    """Every finding from every *earlier* experiment.

    This is what makes a new experiment start informed rather than from scratch:
    what was confirmed, what was refuted, and what is still untried.
    """
    from . import findings

    ordered = [e.experiment_id for e in list_experiments()]
    current = experiment_id or (current_experiment().experiment_id if current_experiment() else "")
    cutoff = ordered.index(current) if current in ordered else len(ordered)
    earlier = set(ordered[:cutoff])
    return [f for f in findings.read_findings() if f.experiment_id in earlier]


def write_index(path: Path | None = None) -> Path:
    """Rewrite ``EXPERIMENTS.md`` — one row per experiment, newest last.

    Derived from the manifests every time, so it cannot disagree with them.
    """
    target = Path(path) if path is not None else paths.EXPERIMENTS_INDEX
    found = list_experiments()

    lines = [
        "# Experiments",
        "",
        "Derived from each `experiment.json` by `experiments.write_index()`.",
        "Do not edit by hand — it will be overwritten.",
        "",
        "`cv` is cross-validated on that experiment's own training split.",
        "`val` is its own 142-row validation split, read once, autonomously.",
        "Neither is the final test, which is read once for the whole programme.",
        "",
        "| experiment | status | train/val | best cv | val | summary |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for experiment in found:
        cv = "—" if experiment.best_cv is None else f"{experiment.best_cv:.4f}"
        val = "—" if experiment.val_score is None else f"{experiment.val_score:.4f}"
        lines.append(
            f"| `{experiment.experiment_id}` | {experiment.status} | "
            f"{len(experiment.train_ids)}/{len(experiment.val_ids)} | {cv} | {val} | "
            f"{experiment.summary or ''} |"
        )
    if not found:
        lines.append("| _none yet_ | | | | | |")

    lines += [
        "",
        "## The final test",
        "",
        "`data/frozen/holdout.csv` (179 rows) is read **once for the whole**",
        "**programme**, not once per experiment. Reusing it across experiments",
        "would turn it into a validation set: each experiment's direction would be",
        "chosen partly by the previous one's final score, and the number would stop",
        "measuring generalisation and start measuring how hard we searched.",
        "",
        "`make receipts` shows whether it has been spent.",
        "",
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    """``make experiments`` and friends."""
    import argparse

    parser = argparse.ArgumentParser(description="Experiment containers.")
    sub = parser.add_subparsers(dest="command")

    opener = sub.add_parser("open", help="start a new experiment")
    opener.add_argument("name")
    opener.add_argument("--seed", type=int, default=None)
    opener.add_argument("--val-frac", type=float, default=0.2)

    closer = sub.add_parser("close", help="finish the open experiment")
    closer.add_argument("--summary", default="")
    closer.add_argument("--abandon", action="store_true")

    args = parser.parse_args(argv)

    try:
        if args.command == "open":
            experiment = open_experiment(args.name, seed=args.seed, val_frac=args.val_frac)
            print(
                f"opened {experiment.experiment_id}: "
                f"{len(experiment.train_ids)} train / {len(experiment.val_ids)} val "
                f"(seed {experiment.seed})"
            )
            write_index()
            return 0
        if args.command == "close":
            experiment = close_experiment(
                summary=args.summary, status="abandoned" if args.abandon else "closed"
            )
            print(f"{experiment.status} {experiment.experiment_id} (best cv {experiment.best_cv})")
            return 0
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"experiment command failed: {exc}")
        return 1

    found = list_experiments()
    if not found:
        print('no experiments yet — `make experiment-open NAME="baseline"`')
        return 0
    for experiment in found:
        cv = "—" if experiment.best_cv is None else f"{experiment.best_cv:.4f}"
        val = "—" if experiment.val_score is None else f"{experiment.val_score:.4f}"
        print(
            f"  {experiment.experiment_id:<28} {experiment.status:<10} "
            f"cv={cv:<8} val={val:<8} {experiment.summary}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
