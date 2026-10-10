"""Step 7 — ``ensemble``: combine the best tuned models, then score the holdout.

This is the only module permitted to read ``data/frozen/holdout.csv``, and it
may do so **once per frozen dataset**, at the very end, after the ensemble is
chosen by cross-validation on the training split. The holdout number is a
report, never a selection signal.

Inputs
    ``experiments/results.csv`` (to pick members), ``data/frozen/train.csv``,
    and — for the final estimate only — ``data/frozen/holdout.csv``.
Outputs
    A ledger row with ``step="ensemble"``, ``reports/03_ensemble.md``, and a
    Kaggle submission CSV at ``reports/submission.csv``.
Done criteria
    The ensemble's CV score is logged with an honest ``holdout_read``
    attestation, and the submission has 418 rows of ``PassengerId,Survived``
    with no missing values. Whether the ensemble *beats* the best single model
    is a result to report, not a criterion to retry until it passes.

The holdout protocol, in order, and never reordered:

1. :func:`titanic_agent.receipts.require_unread` — **before** the read, so a
   second measurement is refused rather than taken.
2. fit on all of train.csv, score holdout.csv once.
3. :func:`titanic_agent.receipts.record_holdout_read` — **after**, so the read
   leaves a trace whether or not anything else succeeds.

Checking after reading would defeat the point entirely.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths

#: Combination strategies this step supports.
METHODS: tuple[str, ...] = ("soft_vote", "hard_vote", "stack")

#: Never an ensemble member. It exists to be the number to beat.
NEVER_MEMBERS: frozenset[str] = frozenset({"dummy"})


@dataclass(frozen=True)
class EnsembleResult:
    """Outcome of one ensemble run."""

    run_id: str = ""
    method: str = ""
    members: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    cv_score: float = 0.0
    cv_std: float = 0.0
    holdout_score: float | None = None
    submission_path: Path | None = None
    report_path: Path | None = None
    estimator: Any = None
    best_single: float = 0.0


def select_members(
    *,
    log_path: Path = paths.RESULTS_LOG,
    top_k: int = 4,
    min_cv_score: float = 0.0,
    max_correlation: float = 0.95,
) -> list[dict[str, Any]]:
    """Pick ensemble members from the experiment ledger.

    Args:
        log_path: Ledger to read.
        top_k: How many models to keep, best ``cv_score`` first.
        min_cv_score: Floor; drop anything at or below it.
        max_correlation: Reserved for out-of-fold correlation pruning, which
            needs stored fold predictions the ledger does not carry yet. Until
            then members are de-duplicated by model key, which removes the
            obvious redundancy (the same estimator logged twice) if not the
            subtle kind.

    Returns:
        One dict per member with its model key and params, ready for
        :func:`titanic_agent.tune.build_pipeline`.

    Only rows whose ``frozen_sha256`` matches the current manifest are eligible:
    a score computed against different bytes is not comparable to one computed
    against these.
    """
    from . import freeze, results_log

    manifest = freeze.require_frozen()
    frame = results_log.read_log(log_path)
    if frame.empty:
        return []

    eligible = frame[
        (frame["step"] == "tune")
        & (frame["frozen_sha256"] == manifest.dataset_sha256)
        & (~frame["model"].isin(NEVER_MEMBERS))
    ].dropna(subset=["cv_score"])
    if eligible.empty:
        return []

    eligible = eligible[eligible["cv_score"].astype(float) > min_cv_score]
    eligible = eligible.sort_values("cv_score", ascending=False)

    members: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, row in eligible.iterrows():
        model = str(row["model"])
        if model in seen:
            continue
        seen.add(model)
        params = dict(row["params"]) if isinstance(row["params"], dict) else {}
        members.append(
            {
                "model": model,
                "params": {k: v for k, v in params.items() if k != "features"},
                "features": params.get("features"),
                "cv_score": float(row["cv_score"]),
                "run_id": str(row["run_id"]),
            }
        )
        if len(members) >= top_k:
            break
    return members


def _build_members(members: list[dict[str, Any]], *, seed: int, features: list[str] | None):
    """Rebuild each member's pipeline from its logged params."""
    from . import tune as tune_module

    built = []
    for member in members:
        pipeline = tune_module.build_pipeline(member["model"], seed=seed, features=features)
        if member.get("params"):
            pipeline.set_params(**member["params"])
        built.append((member["model"], pipeline))
    return built


def write_submission(
    predictions,
    *,
    passenger_ids,
    submission_path: Path = paths.REPORTS_DIR / "submission.csv",
    expected_rows: int,
) -> Path:
    """Write a Kaggle-shaped submission and sanity-check its shape.

    ``expected_rows`` is required rather than defaulting to Kaggle's 418. The
    invariant worth enforcing is "one row per row of the test set", which the
    caller knows and this function does not; a baked-in 418 would be a magic
    number that happens to be right for one dataset and silently wrong for any
    other.

    Raises:
        ValueError: Row count mismatch, missing values, or labels outside {0, 1}.
    """
    import pandas as pd

    frame = pd.DataFrame(
        {"PassengerId": list(passenger_ids), "Survived": [int(p) for p in predictions]}
    )
    if len(frame) != expected_rows:
        raise ValueError(f"submission has {len(frame)} rows, expected {expected_rows}")
    if frame.isna().any().any():
        raise ValueError("submission contains missing values")
    bad = set(frame["Survived"].unique()) - {0, 1}
    if bad:
        raise ValueError(f"submission has labels outside {{0, 1}}: {sorted(bad)}")

    submission_path = Path(submission_path)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(submission_path, index=False)
    return submission_path


def _markdown(result: EnsembleResult, members: list[dict[str, Any]]) -> str:
    """Render the ensemble write-up, CV and holdout side by side."""
    holdout = "not read" if result.holdout_score is None else f"{result.holdout_score:.4f}"
    verdict = (
        "The ensemble did not beat the best single model. That is a real and common "
        "result, not a failure to retry: ship the single model and say so."
        if result.cv_score < result.best_single
        else "The ensemble matched or beat the best single model."
    )
    lines = [
        "# 03 — Ensemble",
        "",
        "Generated by `make ensemble` (step 7). Regenerable: `make clean` removes it.",
        "",
        "## Scores",
        "",
        "| measure | value |",
        "| --- | --- |",
        f"| ensemble CV ({result.method}) | {result.cv_score:.4f} ± {result.cv_std:.4f} |",
        f"| best single model (CV) | {result.best_single:.4f} |",
        f"| holdout | {holdout} |",
        "",
        verdict,
        "",
        "## Members",
        "",
        "| model | CV score | source run |",
        "| --- | --- | --- |",
    ]
    lines += [f"| `{m['model']}` | {m['cv_score']:.4f} | `{m['run_id']}` |" for m in members]
    lines += [
        "",
        "## On the holdout number",
        "",
        "The holdout is read **once per frozen dataset**, after member selection is",
        "final, and the number is a report — never a signal to go back and pick a",
        "different model. Doing that converts the holdout into a second validation",
        "set and the honest estimate is gone. Every read leaves a receipt in",
        "`experiments/holdout_reads.json`.",
        "",
    ]
    if result.holdout_score is None:
        lines += [
            "This run did **not** read the holdout. Run with `--score-holdout` once,",
            "at the very end, when you are done iterating.",
            "",
        ]
    return "\n".join(lines)


def ensemble(
    *,
    members: list[dict[str, Any]] | None = None,
    method: str = "soft_vote",
    weights: list[float] | None = None,
    cv_folds: int = 5,
    scoring: str = "accuracy",
    seed: int = paths.SEED,
    frozen_dir: Path = paths.FROZEN_DIR,
    log_path: Path = paths.RESULTS_LOG,
    score_holdout: bool = False,
    submission_path: Path = paths.REPORTS_DIR / "submission.csv",
    report_path: Path = paths.REPORTS_DIR / "03_ensemble.md",
) -> EnsembleResult:
    """Build, cross-validate and log an ensemble, then write a submission.

    Args:
        members: Member specs; defaults to :func:`select_members`.
        method: One of :data:`METHODS`.
        weights: Per-member weights for the voting methods. ``None`` means equal.
        cv_folds: Stratified folds for the ensemble's own CV.
        scoring: sklearn scoring name.
        seed: Seeds folds and every member that takes a random state.
        frozen_dir: Frozen dataset; verified before reading.
        log_path: Ledger to append to.
        score_holdout: Read ``holdout.csv`` and report the final estimate.
            Leave False during iteration; flip it exactly once, at the end.
        submission_path: Where to write the Kaggle submission.
        report_path: Markdown write-up of members, weights and scores.

    Returns:
        The :class:`EnsembleResult`.

    Raises:
        RuntimeError: Not frozen, verification failed, or ``score_holdout`` was
            requested for a frozen dataset whose holdout was already scored.
        ValueError: Unknown ``method``, or fewer than two members.
    """
    import pandas as pd
    from sklearn.ensemble import StackingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    from . import freeze, receipts, results_log, run_log
    from . import tune as tune_module

    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")

    manifest = freeze.require_frozen(frozen_dir)
    members = select_members(log_path=log_path) if members is None else members
    if len(members) < 2:
        raise ValueError(
            f"an ensemble needs at least two members, got {len(members)}. "
            "Run `make tune` first so the ledger has something to combine."
        )

    run_log.record(
        "step_started",
        step="ensemble",
        detail=f"{method} over {[m['model'] for m in members]}",
    )

    frame = tune_module.load_training_frame(frozen_dir)
    features = tune_module.resolve_features(frame, members[0].get("features"))
    design, labels = frame[features], frame["Survived"]

    built = _build_members(members, seed=seed, features=features)
    folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    if method == "stack":
        model = StackingClassifier(
            estimators=built,
            final_estimator=LogisticRegression(max_iter=2000, random_state=seed),
            cv=folds,
        )
    else:
        model = VotingClassifier(
            estimators=built,
            voting="soft" if method == "soft_vote" else "hard",
            weights=weights,
        )

    scores = cross_val_score(model, design, labels, cv=folds, scoring=scoring, n_jobs=-1)
    cv_score, cv_std = float(scores.mean()), float(scores.std())
    best_single = max(float(m["cv_score"]) for m in members)

    model.fit(design, labels)

    # --- the holdout protocol: check, read, record. Never reordered. -------- #
    holdout_score: float | None = None
    if score_holdout:
        receipts.require_unread(manifest.dataset_sha256)
        holdout = tune_module.add_derived(pd.read_csv(Path(frozen_dir) / "holdout.csv"))
        holdout_score = float(model.score(holdout[features], holdout["Survived"]))
        receipts.record_holdout_read(
            frozen_sha256=manifest.dataset_sha256,
            run_id="pending",
            score=holdout_score,
            reason=f"final estimate for {method} over {[m['model'] for m in members]}",
        )
        run_log.record(
            "decision",
            step="ensemble",
            detail=f"read the holdout once: {holdout_score:.4f}",
        )
    # ------------------------------------------------------------------------ #

    notes = f"members={[m['model'] for m in members]}; best_single_cv={best_single:.4f}"
    if holdout_score is not None:
        notes += f"; holdout={holdout_score:.4f}"

    run_id = results_log.append_run(
        step="ensemble",
        model=f"{method}({'+'.join(m['model'] for m in members)})",
        params={"members": [m["model"] for m in members], "weights": weights, "method": method},
        cv_score=cv_score,
        cv_std=cv_std,
        cv_folds=cv_folds,
        seed=seed,
        holdout_read=score_holdout,  # attested, matching what actually happened
        notes=notes,
        path=log_path,
    )

    test = tune_module.add_derived(pd.read_csv(Path(frozen_dir) / "test.csv"))
    written = write_submission(
        model.predict(test[features]),
        passenger_ids=test["PassengerId"],
        submission_path=submission_path,
        expected_rows=len(test),
    )

    result = EnsembleResult(
        run_id=run_id,
        method=method,
        members=[m["model"] for m in members],
        weights=list(weights) if weights else [],
        cv_score=cv_score,
        cv_std=cv_std,
        holdout_score=holdout_score,
        submission_path=written,
        report_path=Path(report_path),
        estimator=model,
        best_single=best_single,
    )

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_markdown(result, members), encoding="utf-8")
    report_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "method": method,
                "members": result.members,
                "cv_score": cv_score,
                "cv_std": cv_std,
                "best_single_cv": best_single,
                "holdout_score": holdout_score,
                "holdout_read": score_holdout,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    run_log.record("step_finished", step="ensemble", detail=f"cv={cv_score:.4f}", run_id=run_id)
    return result


def main(argv: list[str] | None = None) -> int:
    """``make ensemble`` entry point."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", default="soft_vote", choices=METHODS)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--score-holdout",
        action="store_true",
        help="read the holdout for the final estimate -- once per frozen dataset",
    )
    args = parser.parse_args(argv)

    try:
        result = ensemble(
            members=select_members(top_k=args.top_k),
            method=args.method,
            cv_folds=args.cv_folds,
            score_holdout=args.score_holdout,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"ensemble failed: {exc}")
        return 1

    holdout = "not read" if result.holdout_score is None else f"{result.holdout_score:.4f}"
    print(
        f"{result.method}: cv={result.cv_score:.4f} +/- {result.cv_std:.4f} "
        f"(best single {result.best_single:.4f}) holdout={holdout}"
    )
    print(f"  members: {', '.join(result.members)}")
    print(f"  submission -> {result.submission_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
