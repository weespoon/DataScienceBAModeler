"""The persisted champion: one configuration every experiment is paired against.

``docs/MEASUREMENT.md`` names the programme's most likely failure: each
experiment re-draws its train/val split, so two experiments' absolute CV scores
are *unpaired*, and identical configurations have drifted 0.004-0.011 between
experiments — one to three times the effects being argued about. Every
experiment was internally defensible while every comparison *between* them was
made at two to three times the effect size.

The fix is a reference that travels. The champion is a fixed configuration —
model, hyperparameters, features, fitted transforms — re-scored by each new
experiment on **its own** folds, beside every challenger. What crosses the
experiment boundary is then ``delta_vs_champion``, a paired difference, rather
than an absolute score whose level depends on which rows the split happened to
draw.

That does not make deltas from different experiments perfectly comparable: they
are still estimated on different rows. It removes the dominant term — the
split-level drift that moves *both* arms together — and leaves the paired SE,
which is the one this repo has measured.

The champion's hyperparameters were chosen in an earlier experiment on rows that
overlap the current one's validation split. Used as a fixed *reference arm* in
CV on the current training rows, that can only flatter the champion, which makes
challengers' deltas conservative. It is recorded in each entry rather than
silently assumed.

``experiments/champion.jsonl`` is append-only and global, like the ledger: a
champion that lived in a run directory would be "the champion of this pass",
and every pass would start from scratch. The last line is the current champion;
the lines above it are the succession, each with the evidence that crowned it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import paths

#: The best-of-breed configuration at the time this module was written, and why.
#: exp-001 found random_forest the strongest single model; exp-006 re-tuned it on
#: its own training rows (row 20261010T203601-584bbf); exp-006's nested CV then
#: showed the fixed 12-feature baseline generalises 0.0088 +/- 0.0068 *better*
#: than the searched 6-feature set. So the champion is the RF with exp-006's
#: params on the 12 baseline features, not the searched set.
INITIAL: dict[str, Any] = {
    "name": "rf_baseline12",
    "model": "random_forest",
    "params": {"clf__n_estimators": 400, "clf__min_samples_leaf": 3, "clf__max_depth": None},
    "features": [
        "Sex",
        "Pclass",
        "Age",
        "Fare",
        "SibSp",
        "Parch",
        "Embarked",
        "FamilySize",
        "IsAlone",
        "Title",
        "Deck",
        "AgeMissing",
    ],
    "engineer": [],
    "members": [],
}
INITIAL_EVIDENCE: tuple[str, ...] = ("20261010T203601-584bbf", "20261010T221207-f0d1bb")
INITIAL_REASON = (
    "best of breed through exp-006: random_forest is the strongest single model "
    "(exp-001), params re-tuned on exp-006's training rows, and nested CV showed the "
    "12-feature baseline generalises better than the searched 6-feature set "
    "(-0.0088 +/- 0.0068 for searching). Params were chosen on rows that overlap later "
    "experiments' validation splits; as a reference arm that can only flatter it."
)


@dataclass(frozen=True)
class Champion:
    """One entry in the succession."""

    #: The configuration itself, in the form :mod:`titanic_agent.bakeoff` builds.
    config: dict[str, Any] = field(default_factory=dict)
    crowned_utc: str = ""
    experiment_id: str = ""
    #: Ledger rows that justify the crowning.
    evidence: list[str] = field(default_factory=list)
    reason: str = ""
    #: Paired delta over the previous champion, when one existed.
    delta_vs_previous: float | None = None

    @property
    def name(self) -> str:
        return str(self.config.get("name", ""))


def _path(path: Path | None) -> Path:
    return Path(path) if path is not None else paths.CHAMPION


def history(path: Path | None = None) -> list[Champion]:
    """Every champion, oldest first."""
    target = _path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return []
    found: list[Champion] = []
    for line in text.splitlines():
        if line.strip():
            found.append(Champion(**json.loads(line)))
    return found


def current(path: Path | None = None) -> Champion | None:
    """The reigning champion, or None before one has been crowned."""
    found = history(path)
    return found[-1] if found else None


def crown(
    config: dict[str, Any],
    *,
    reason: str,
    evidence: list[str],
    experiment_id: str = "",
    delta_vs_previous: float | None = None,
    path: Path | None = None,
) -> Champion:
    """Append a new champion. The old ones stay; the file is the succession.

    Raises:
        ValueError: No evidence. A champion crowned without a ledger row behind
            it is folklore with a file format.
    """
    from . import locking, run_log

    if not evidence:
        raise ValueError("a champion needs evidence: the ledger rows that crowned it")
    entry = Champion(
        config=dict(config),
        crowned_utc=datetime.now(UTC).replace(microsecond=0).isoformat(),
        experiment_id=experiment_id,
        evidence=list(evidence),
        reason=reason,
        delta_vs_previous=delta_vs_previous,
    )
    target = _path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with locking.locked(target), target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(entry), sort_keys=True, default=str) + "\n")
    run_log.record(
        "decision",
        step="champion",
        detail=f"crowned {entry.name} ({experiment_id or 'seed'}): {reason}",
    )
    return entry


def ensure(path: Path | None = None) -> Champion:
    """The current champion, crowning :data:`INITIAL` if there has never been one."""
    existing = current(path)
    if existing is not None:
        return existing
    return crown(INITIAL, reason=INITIAL_REASON, evidence=list(INITIAL_EVIDENCE), path=path)


def submit(
    *,
    frozen_dir: Path = paths.FROZEN_DIR,
    log_path: Path | None = None,
    champion_path: Path | None = None,
    submission_path: Path | None = None,
) -> Path:
    """Fit the reigning champion on the working set and predict Kaggle's test set.

    Reads ``train.csv`` (712 labelled rows) and ``test.csv`` (418 unlabelled).
    **Never** ``holdout.csv``: the final test is read once, by
    ``ensemble --score-holdout``, and a submission has no business spending it.
    So the champion is fitted on 712 rows, not 891, and the 179 holdout
    passengers contribute no group rates — a test passenger whose only
    labelled companions are in the holdout gets the unknown rate. That costs a
    little accuracy on Kaggle and buys an honest final number.

    Group rates for test passengers come from their labelled companions in the
    working set, through the same in-fold :class:`~titanic_agent.bakeoff.Engineer`
    every CV score used: a test row is simply an unseen row.

    Raises:
        RuntimeError: No champion, or the frozen data fails verification.
    """
    import pandas as pd

    from . import bakeoff, ensemble, freeze, results_log, run_log, runs, tune

    freeze.require_frozen(frozen_dir)
    reigning = current(champion_path)
    if reigning is None:
        raise RuntimeError("no champion yet -- run a bake-off first")
    train = tune.add_derived(pd.read_csv(Path(frozen_dir) / "train.csv"))
    test = tune.add_derived(pd.read_csv(Path(frozen_dir) / "test.csv"))

    estimator = bakeoff.build(reigning.config, seed=paths.SEED)
    estimator.fit(bakeoff.design(train), train["Survived"])
    predictions = estimator.predict(test.drop(columns=["Survived"], errors="ignore"))

    target = (
        Path(submission_path)
        if submission_path is not None
        else runs.run_dir(run_log.current_pass_id()) / "submission.csv"
    )
    written = ensemble.write_submission(
        predictions,
        passenger_ids=test["PassengerId"],
        submission_path=target,
        expected_rows=len(test),
    )
    if submission_path is None:
        runs.publish(written)
    run_id = results_log.append_run(
        step="submit",
        model=reigning.name,
        params={
            "config": reigning.config,
            "features": reigning.config.get("features", []),
            "train_rows": len(train),
            "test_rows": len(test),
            "predicted_survival_rate": round(float(pd.Series(predictions).mean()), 4),
        },
        # A submission is not a measurement: Kaggle's labels are not ours.
        cv_score=None,
        seed=paths.SEED,
        holdout_read=False,  # attested: train.csv and test.csv only
        notes=(
            f"submission from champion {reigning.name} ({reigning.experiment_id or 'seed'}), "
            f"fitted on {len(train)} working rows; holdout untouched; {written}"
        ),
        path=log_path,
    )
    run_log.record(
        "step_finished",
        step="submit",
        detail=f"{len(test)} predictions from {reigning.name} -> {written}",
        run_id=run_id,
    )
    return written


def main(argv: list[str] | None = None) -> int:
    """``make champion`` (the succession) and ``make submit``."""
    import sys

    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["submit"]:
        try:
            written = submit()
        except RuntimeError as exc:
            print(f"submit failed: {exc}")
            return 1
        print(f"submission written: {written} (holdout untouched)")
        return 0

    found = history()
    if not found:
        print("no champion yet -- the first bakeoff crowns the exp-006 baseline RF")
        return 0
    for index, entry in enumerate(found):
        marker = "*" if index == len(found) - 1 else " "
        delta = (
            "" if entry.delta_vs_previous is None else f" (+{entry.delta_vs_previous:.4f} paired)"
        )
        print(
            f"{marker} {entry.crowned_utc}  {entry.name:<24} "
            f"{entry.experiment_id or 'seed':<28}{delta}"
        )
        print(f"    {entry.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
