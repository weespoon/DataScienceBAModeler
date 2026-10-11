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

#: Caps that are actually counted and actually bind. Nothing else belongs here.
#:
#: The first version listed kaggle_notebooks, papers and web_fetches too, and
#: nothing incremented any of them. A cold-start session duly ignored the whole
#: block, which was the correct response: a cap nobody counts is a number in a
#: file pretending to be a constraint. They move to UNENFORCED below until the
#: research step exists to count them.
DEFAULT_CAPS: dict[str, Any] = {
    "wall_clock_s": 1800,
    "loop_passes": 3,
}

#: Budgets agreed in principle and not yet countable. Recorded separately and
#: named for what they are, so reading experiment.json never leaves the
#: impression that something is being enforced when it is not.
UNENFORCED_CAPS: dict[str, Any] = {
    "kaggle_notebooks": 20,
    "papers": 10,
    "web_fetches": 0,  # v1 reads the installed package and the Kaggle API only
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
    #: Budgets agreed but not yet counted. Separate from `caps` on purpose.
    caps_unenforced: dict[str, Any] = field(default_factory=lambda: dict(UNENFORCED_CAPS))
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


def record_spend(
    resource: str, amount: float = 1, *, experiment_id: str | None = None
) -> Experiment | None:
    """Add to the open experiment's spend. Returns None when none is open.

    Counting is the whole point: an uncounted cap is decoration. Every resource
    in :data:`DEFAULT_CAPS` must have something calling this, or it belongs in
    :data:`UNENFORCED_CAPS` instead.
    """
    target = experiment_id or (current_experiment().experiment_id if current_experiment() else None)
    if target is None:
        return None
    experiment = load_experiment(target)
    spend = dict(experiment.spend)
    spend[resource] = spend.get(resource, 0) + amount
    return _write(Experiment(**{**asdict(experiment), "spend": spend}))


def over_budget(experiment_id: str | None = None) -> dict[str, tuple[float, float]]:
    """Resources whose spend has reached or passed its cap.

    Returns ``{resource: (spent, cap)}``. Empty means there is budget left.
    Only :data:`DEFAULT_CAPS` are checked — the unenforced ones have no counter
    and reporting them would be theatre.
    """
    target = experiment_id or (current_experiment().experiment_id if current_experiment() else None)
    if target is None:
        return {}
    experiment = load_experiment(target)
    return {
        resource: (float(experiment.spend.get(resource, 0)), float(cap))
        for resource, cap in experiment.caps.items()
        if float(experiment.spend.get(resource, 0)) >= float(cap)
    }


def record_val_read(
    *, experiment_id: str, score: float, model: str, run_id: str, path: Path | None = None
) -> dict[str, Any]:
    """Record that an experiment spent its validation read."""
    target = Path(path) if path is not None else paths.VAL_READS
    target.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "experiment_id": experiment_id,
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "score": round(float(score), 6),
        "model": model,
        "run_id": run_id,
    }
    from . import locking

    with locking.locked(target), target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def val_reads_for(experiment_id: str, path: Path | None = None) -> list[dict[str, Any]]:
    """Every validation read recorded against one experiment."""
    target = Path(path) if path is not None else paths.VAL_READS
    found: list[dict[str, Any]] = []
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return found
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("experiment_id") == experiment_id:
            found.append(entry)
    return found


def require_val_unread(experiment_id: str, path: Path | None = None) -> None:
    """Refuse a second validation read within one experiment.

    Validation splits are cheap *across* experiments — the next one draws
    different rows. They are not cheap *within* one: reading the same 143 rows
    twice, with a model chosen in between, is adaptive overfitting at the
    experiment level and makes that experiment's number a selection statistic.

    The refusal is deliberately long, and it argues rather than just forbids.
    An agent that wants a second read is standing *here*, not in CLAUDE.md, and
    the reason it should not want one is arithmetic it will otherwise re-derive
    and get wrong. This repo's own lesson, from the commit that moved Rule 1
    into a hook: enforce it where the hand reaches, not only where the rule is
    written down.

    Raises:
        RuntimeError: This experiment already read its validation split.
    """
    existing = val_reads_for(experiment_id, path)
    if not existing:
        return
    first = existing[0]
    raise RuntimeError(
        f"{experiment_id} already read its validation split at "
        f"{first.get('timestamp_utc', 'an unrecorded time')} "
        f"(scored {first.get('score')}). Report that number, or open a new "
        f"experiment — a second read on the same rows is selection, not measurement."
        "\n\n"
        "Before you reach for a way around this: a second read would not answer "
        "your question either. The validation split holds 143 rows, so its "
        "accuracy moves in steps of 1/143 = 0.0070 and its standard error is "
        "0.032. Two configurations that truly differ by 0.004 cannot differ on "
        "it by 0.004 — they differ by 0, 0.0070 or 0.0140, whichever way the "
        "disagreeing rows fall. Pairing does not rescue it (a paired read "
        "resolves ~0.06 at best), and neither does the reusable-holdout "
        "literature: Thresholdout's own noise on 143 rows has sd 0.020, five "
        "times the effects argued about here. The problem is resolution, not "
        "adaptivity, so no mechanism that makes reads *safe* makes them "
        "*informative*.\n\n"
        "Comparisons belong in titanic_agent.compare — paired repeated CV on "
        "the 569 training rows resolves ~0.003, which is the only instrument "
        "here that can settle the question. The val read is a pipeline "
        "tripwire with ~0.06 resolution; it earned its keep by catching "
        "exp-002's contamination, not by ranking candidates. "
        "See docs/MEASUREMENT.md."
    )


def evaluate(
    experiment_id: str | None = None,
    *,
    run_id: str | None = None,
    frozen_dir: Path = paths.FROZEN_DIR,
    log_path: Path = paths.RESULTS_LOG,
) -> Experiment:
    """Score this experiment's chosen model on its own validation split. Once.

    Autonomous by design: no human is asked, because a validation split is
    re-drawable and the next experiment gets different rows. The locked final
    test is the one that still stops for a person.

    Picks whatever this experiment scored best in cross-validation — a tuned
    model or an ensemble — refits it on the experiment's training rows, and
    scores the held-back rows exactly once.

    ``run_id`` pre-declares the row to score instead. Without it, the val read
    measures whatever won CV — often not the thing the experiment changed, and
    in a feature search, possibly a test the selection rule *rejected*.

    Raises:
        RuntimeError: No experiment, nothing to evaluate, the validation split
            has already been read, or ``run_id`` is not a scored row of this
            experiment.
    """
    from . import ensemble as ensemble_module
    from . import results_log
    from . import tune as tune_module

    experiment = load_experiment(experiment_id or _require_current().experiment_id)
    require_val_unread(experiment.experiment_id)

    frame = results_log.read_log(log_path)
    rows = (
        frame[frame["experiment_id"].astype(str) == experiment.experiment_id]
        if not frame.empty and "experiment_id" in frame
        else frame.iloc[0:0]
    )
    scored = rows.dropna(subset=["cv_score"]) if len(rows) else rows
    if not len(scored):
        raise RuntimeError(
            f"{experiment.experiment_id} has no scored runs to evaluate. "
            "Run `make loop` first — there is nothing to measure yet."
        )

    if run_id is None:
        best = scored.loc[scored["cv_score"].astype(float).idxmax()]
    else:
        declared = scored[scored["run_id"].astype(str) == run_id]
        if not len(declared):
            raise RuntimeError(
                f"{run_id} is not a scored run of {experiment.experiment_id}; "
                "the validation read is unspent."
            )
        best = declared.iloc[0]
    train, val = load_frames(experiment, frozen_dir=frozen_dir)
    model_name = str(best["model"])
    params = best["params"] or {}
    if str(best["step"]) == "bakeoff":
        # A bake-off arm carries fitted transforms (some read the label), so it
        # is rebuilt whole from its logged config and fitted on the full frame:
        # its Engineer step learns from the training rows only, never from val.
        from . import bakeoff

        estimator = bakeoff.build(params["config"], seed=experiment.seed)
        features = list(params["config"]["features"])
        estimator.fit(bakeoff.design(train), train["Survived"])
        score = float(estimator.score(bakeoff.design(val), val["Survived"]))
        return _record_evaluation(
            experiment, best, run_id, model_name, features, score, len(val), log_path
        )

    features = tune_module.resolve_features(train, params.get("features"))
    if model_name in tune_module.MODEL_KEYS:
        estimator = tune_module.build_pipeline(model_name, seed=experiment.seed)
        tunable = {k: v for k, v in (best["params"] or {}).items() if k.startswith("clf__")}
        if tunable:
            estimator.set_params(**tunable)
    else:
        # An ensemble row: rebuild it from the members recorded in its params.
        members = ensemble_module.select_members(log_path=log_path)
        if len(members) < 2:
            raise RuntimeError(
                f"{experiment.experiment_id} scored best with {model_name}, but its "
                "members cannot be rebuilt from the ledger."
            )
        built = ensemble_module._build_members(members, seed=experiment.seed, features=features)
        from sklearn.ensemble import VotingClassifier

        estimator = VotingClassifier(estimators=built, voting="soft")

    estimator.fit(train[features], train["Survived"])
    score = float(estimator.score(val[features], val["Survived"]))
    return _record_evaluation(
        experiment, best, run_id, model_name, features, score, len(val), log_path
    )


def _record_evaluation(
    experiment: Experiment,
    best,
    run_id: str | None,
    model_name: str,
    features: list[str],
    score: float,
    val_rows: int,
    log_path: Path,
) -> Experiment:
    """Ledger row, read receipt, journal line and manifest update for one val read."""
    from . import results_log, run_log

    run_id = results_log.append_run(
        step="evaluate",
        model=model_name,
        params={"features": features, "val_rows": val_rows},
        cv_score=score,
        cv_folds=None,
        seed=experiment.seed,
        holdout_read=False,  # the FINAL test is untouched; this is the val split
        notes=(
            f"validation score for {experiment.experiment_id} on {val_rows} held-back rows; "
            + (f"pre-declared row {run_id}, " if run_id else "")
            + f"its cv was {float(best['cv_score']):.4f}"
        ),
        path=log_path,
    )
    record_val_read(
        experiment_id=experiment.experiment_id,
        score=score,
        model=model_name,
        run_id=run_id,
    )
    run_log.record(
        "decision",
        step="evaluate",
        detail=(
            f"{experiment.experiment_id} validation: {score:.4f} on {val_rows} rows "
            f"({model_name}); cv was {float(best['cv_score']):.4f}"
        ),
    )

    updated = _write(
        Experiment(**{**asdict(experiment), "val_score": score, "best_cv": float(best["cv_score"])})
    )
    write_index()
    return updated


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


def _comparison_counts(log_path: Path | None = None) -> dict[str, int]:
    """How many comparisons each experiment's reported score was selected from.

    A score published after a search is biased upward, and nothing in the
    ledger's own columns says a search happened. Feature rows carry
    ``test_number`` per comparison and a summary row carries ``selected_from``,
    so the count is derivable without a schema change -- which matters, because
    a new column means a hash-chain rebuild and a rebuild costs the chain's
    guarantee over every earlier row.
    """
    from . import results_log

    try:
        frame = results_log.read_log(log_path)
    except Exception:
        return {}
    if frame.empty or "experiment_id" not in frame:
        return {}

    counts: dict[str, int] = {}
    for _, row in frame.iterrows():
        params = row["params"]
        if not isinstance(params, dict):
            continue
        experiment_id = str(row["experiment_id"])
        if not experiment_id or experiment_id == "nan":
            continue
        declared = params.get("selection", {})
        number = declared.get("selected_from") if isinstance(declared, dict) else None
        if number is None:
            number = params.get("selected_from")
        if number is None:
            number = params.get("test_number")
        if number is not None:
            counts[experiment_id] = max(counts.get(experiment_id, 0), int(number))
    return counts


def _champion_deltas(log_path: Path | None = None) -> dict[str, tuple[str, float, float]]:
    """Each experiment's best paired delta over the champion: ``(arm, delta, sigma)``.

    This is the cross-experiment column. Absolute cv drifts 0.004-0.011 between
    experiments because each re-draws its split; a delta against a champion
    re-scored on the same folds does not carry that drift.
    """
    from . import results_log

    try:
        frame = results_log.read_log(log_path)
    except Exception:
        return {}
    if frame.empty or "experiment_id" not in frame:
        return {}
    best: dict[str, tuple[str, float, float]] = {}
    for _, row in frame.iterrows():
        params = row["params"]
        if not isinstance(params, dict) or "delta_vs_champion" not in params:
            continue
        if params.get("arm") == "champion":
            continue
        experiment_id = str(row["experiment_id"])
        delta = float(params["delta_vs_champion"])
        if experiment_id not in best or delta > best[experiment_id][1]:
            best[experiment_id] = (str(params.get("arm")), delta, float(params.get("sigma", 0)))
    return best


def write_index(path: Path | None = None) -> Path:
    """Rewrite ``EXPERIMENTS.md`` — one row per experiment, newest last.

    Derived from the manifests every time, so it cannot disagree with them.
    """
    target = Path(path) if path is not None else paths.EXPERIMENTS_INDEX
    found = list_experiments()

    searched = _comparison_counts()
    versus = _champion_deltas()
    val_rows = max((len(e.val_ids) for e in found), default=143)

    def noise_band(score: float, rows: int) -> float:
        """2 SE on an accuracy of ``score`` over ``rows`` rows.

        Printed beside every gap because the gap invites being read as a
        measurement of selection bias, and it is roughly eight times too coarse
        to be one. Taken at the observed accuracy rather than the worst case
        p=0.5, so it matches the figures in docs/MEASUREMENT.md.
        """
        return 2 * (max(score * (1 - score), 1e-9) / max(rows, 1)) ** 0.5

    lines = [
        "# Experiments",
        "",
        "Derived from each `experiment.json` by `experiments.write_index()`.",
        "Do not edit by hand — it will be overwritten.",
        "",
        "`cv` is cross-validated on that experiment's own training split, and is",
        "annotated with how many comparisons it was selected from where that is",
        f"known — a post-search `cv` is biased upward. `val` is its own {val_rows}-row",
        "validation split, read once, autonomously.",
        "",
        f"**`gap` is not a measurement.** At {val_rows} rows the 2 SE band is about",
        f"±{noise_band(0.84, val_rows):.3f}, so a gap smaller than that says nothing; the",
        "column is here to make that obvious rather than to be tracked.",
        "Comparisons belong in `compare.py`.",
        "Neither number is the final test, which is read once for the whole programme.",
        "",
        "**`Δ champion` is the column to compare across experiments.** Each experiment",
        "re-scores the persisted champion (`make champion`) on its own folds, so the",
        "delta is paired and carries none of the 0.004-0.011 split drift that makes",
        "`best cv` incomparable between rows. It is the best arm's delta, so it is a",
        "best-of-N and biased upward by the search; the experiment's report says by how much.",
        "",
        "| experiment | status | train/val | best cv | Δ champion | val | gap | summary |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for experiment in found:
        cv = "—" if experiment.best_cv is None else f"{experiment.best_cv:.4f}"
        count = searched.get(experiment.experiment_id)
        if cv != "—" and count:
            cv += f" <br><sub>selected from {count}</sub>"
        val = "—" if experiment.val_score is None else f"{experiment.val_score:.4f}"
        if experiment.best_cv is None or experiment.val_score is None:
            gap = "—"
        else:
            difference = experiment.val_score - experiment.best_cv
            band = noise_band(experiment.val_score, len(experiment.val_ids))
            marker = "" if abs(difference) >= band else " (within noise)"
            gap = f"{difference:+.4f} ±{band:.3f}{marker}"
        if experiment.experiment_id in versus:
            arm, delta, sigma = versus[experiment.experiment_id]
            versus_cell = f"{delta:+.4f} ({sigma:+.1f}σ) <br><sub>{arm}</sub>"
        else:
            versus_cell = "—"
        lines.append(
            f"| `{experiment.experiment_id}` | {experiment.status} | "
            f"{len(experiment.train_ids)}/{len(experiment.val_ids)} | {cv} | {versus_cell} | "
            f"{val} | {gap} | {experiment.summary or ''} |"
        )
    if not found:
        lines.append("| _none yet_ | | | | | | | |")

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

    evaluator = sub.add_parser("evaluate", help="score the open experiment's val split, once")
    evaluator.add_argument("--run-id", default=None, help="pre-declared ledger row to score")

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
        if args.command == "evaluate":
            experiment = evaluate(run_id=args.run_id)
            print(
                f"{experiment.experiment_id} validation: {experiment.val_score:.4f} "
                f"on {len(experiment.val_ids)} held-back rows (cv was {experiment.best_cv:.4f})"
            )
            print("The final test is untouched — that one is read once for the whole programme.")
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
