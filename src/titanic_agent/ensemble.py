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
    The ensemble's CV score is logged and at least matches the best single
    model, the holdout score is reported alongside it, and the submission file
    has 418 rows with ``PassengerId,Survived`` and no missing values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths

#: Combination strategies this step supports.
METHODS: tuple[str, ...] = ("soft_vote", "hard_vote", "stack")


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
        max_correlation: Drop a candidate whose out-of-fold predictions
            correlate above this with an already-selected member — correlated
            members add cost without adding accuracy.

    Returns:
        One dict per member with its model key and params, ready for
        :func:`titanic_agent.tune.build_pipeline`.
    """
    # TODO(ensemble): implement. Read with results_log.read_log(), keep only
    #   rows whose frozen_sha256 matches the current manifest -- scores from an
    #   older frozen dataset are not comparable.
    raise NotImplementedError("select_members() is not implemented yet")


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
            requested for a frozen dataset whose holdout was already scored —
            see :func:`titanic_agent.receipts.require_unread`.
        ValueError: Unknown ``method``, or fewer than two members.
    """
    # TODO(ensemble): implement.
    #   1. freeze.require_frozen(frozen_dir).
    #   2. Rebuild each member via tune.build_pipeline + its logged params.
    #   3. VotingClassifier(voting="soft"/"hard") or StackingClassifier with a
    #      LogisticRegression meta-learner; cross-validate on train.csv.
    #   4. results_log.append_run(step="ensemble", model=f"{method}({members})",
    #      params={"members": ..., "weights": ...}, cv_score=..., cv_std=...,
    #      holdout_read=score_holdout).  The attestation is required: it is what
    #      makes a silent holdout read an attributable false statement.
    #   5. Only if score_holdout, and in this order:
    #        a. receipts.require_unread(manifest.dataset_sha256) -- BEFORE the
    #           read, so a second measurement is refused rather than taken;
    #        b. fit on all of train.csv, score holdout.csv once;
    #        c. receipts.record_holdout_read(frozen_sha256=..., run_id=...,
    #           score=...) -- AFTER, so the read leaves a trace either way;
    #        d. record the score in the same ledger row's notes.
    #      Never reorder (a) and (b): checking after reading defeats the point.
    #   6. Predict test.csv -> submission_path with columns
    #      PassengerId,Survived (int 0/1).
    raise NotImplementedError("ensemble() is not implemented yet")


def write_submission(
    predictions,
    *,
    passenger_ids,
    submission_path: Path = paths.REPORTS_DIR / "submission.csv",
) -> Path:
    """Write a Kaggle-shaped submission and sanity-check its shape.

    Raises:
        ValueError: Row count mismatch, missing values, or labels outside {0, 1}.
    """
    # TODO(ensemble): implement the write plus the three checks.
    raise NotImplementedError("write_submission() is not implemented yet")


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

    result = ensemble(
        members=select_members(top_k=args.top_k),
        method=args.method,
        cv_folds=args.cv_folds,
        score_holdout=args.score_holdout,
    )
    holdout = "n/a" if result.holdout_score is None else f"{result.holdout_score:.4f}"
    print(
        f"{result.method}: cv={result.cv_score:.4f} holdout={holdout} -> {result.submission_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
