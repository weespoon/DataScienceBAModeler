"""Step 6 — ``tune``: cross-validated model search on the frozen training split.

Every call appends one row per evaluated configuration's *best* result to
``experiments/results.csv`` via :mod:`titanic_agent.results_log`. A tuning run
that does not log did not happen.

Inputs
    ``data/frozen/train.csv``; candidate features from ``reports/02_explore.md``.
Outputs
    One or more rows in ``experiments/results.csv``; the fitted best estimator
    returned in memory (nothing is pickled into ``data/frozen``).
Done criteria
    ``cv_score`` beats the majority-class baseline, the run is in the ledger
    with its params and the frozen fingerprint, and wall-clock stayed under
    ``time_budget_s`` on a laptop CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths

#: Model keys this step knows how to build. Keep every one of these cheap
#: enough to cross-validate in seconds on a single CPU core.
MODEL_KEYS: tuple[str, ...] = (
    "dummy",
    "logistic",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "svc",
    "knn",
)


@dataclass(frozen=True)
class RunResult:
    """Outcome of one tuning run, mirroring a ledger row."""

    run_id: str = ""
    model: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    cv_score: float = 0.0
    cv_std: float = 0.0
    cv_folds: int = 5
    fit_seconds: float = 0.0
    estimator: Any = None


def build_pipeline(model_key: str, *, seed: int = paths.SEED):
    """Return an unfitted sklearn ``Pipeline`` for ``model_key``.

    The preprocessing half is shared across models — median impute plus scaling
    for numerics, most-frequent impute plus one-hot for categoricals — so that
    scores differ because of the estimator and not the plumbing.

    Raises:
        ValueError: ``model_key`` is not in :data:`MODEL_KEYS`.
    """
    # TODO(tune): implement with ColumnTransformer + the estimator for model_key.
    raise NotImplementedError("build_pipeline() is not implemented yet")


def search_space(model_key: str) -> dict[str, list[Any]]:
    """Return the hyperparameter grid for ``model_key``.

    Keys are ``Pipeline``-style dotted paths (e.g. ``clf__max_depth``). Keep the
    spaces small; breadth across models beats depth within one here.
    """
    # TODO(tune): implement per-model grids.
    raise NotImplementedError("search_space() is not implemented yet")


def tune(
    model_key: str,
    *,
    space: dict[str, list[Any]] | None = None,
    n_iter: int = 25,
    cv_folds: int = 5,
    scoring: str = "accuracy",
    seed: int = paths.SEED,
    frozen_dir: Path = paths.FROZEN_DIR,
    log_path: Path = paths.RESULTS_LOG,
    time_budget_s: float = 120.0,
    notes: str = "",
) -> RunResult:
    """Cross-validate a hyperparameter search and log the best configuration.

    Args:
        model_key: One of :data:`MODEL_KEYS`.
        space: Override for :func:`search_space`.
        n_iter: Candidates for the randomized search. Ignored when the grid is
            smaller than ``n_iter``, in which case the search is exhaustive.
        cv_folds: Number of stratified folds.
        scoring: sklearn scoring name; the competition metric is accuracy.
        seed: Seeds both the fold shuffling and the search sampler.
        frozen_dir: Frozen dataset; verified before reading.
        log_path: Ledger to append to.
        time_budget_s: Soft budget. Exceeding it is logged in ``notes`` rather
            than raising, so a slow run still leaves a record.
        notes: Free text stored with the ledger row.

    Returns:
        A :class:`RunResult` whose ``run_id`` matches the appended ledger row.

    Raises:
        RuntimeError: The dataset is not frozen or failed verification.
        ValueError: Unknown ``model_key``.
    """
    # TODO(tune): implement.
    #   1. freeze.require_frozen(frozen_dir); load train.csv ONLY -- never
    #      holdout.csv. Any holdout read belongs in a separate, final step.
    #   2. StratifiedKFold(cv_folds, shuffle=True, random_state=seed).
    #   3. RandomizedSearchCV (or GridSearchCV for small spaces), n_jobs=-1.
    #   4. results_log.append_run(step="tune", model=..., params=
    #      search.best_params_, cv_score=search.best_score_, cv_std=the std of
    #      the best candidate's fold scores, cv_folds, seed, notes).
    #   5. Return RunResult carrying that run_id and the refit estimator.
    raise NotImplementedError("tune() is not implemented yet")


def tune_all(
    model_keys: tuple[str, ...] = MODEL_KEYS,
    **kwargs: Any,
) -> list[RunResult]:
    """Run :func:`tune` across ``model_keys``, logging one row each.

    Continues past a failing model rather than aborting the sweep; the failure
    is recorded in that model's ledger row.
    """
    # TODO(tune): loop with try/except per model; log failures as notes.
    raise NotImplementedError("tune_all() is not implemented yet")


def main(argv: list[str] | None = None) -> int:
    """``make tune`` entry point."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="all", help=f"one of {('all',) + MODEL_KEYS}")
    parser.add_argument("--n-iter", type=int, default=25)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=paths.SEED)
    args = parser.parse_args(argv)

    if args.model == "all":
        results = tune_all(n_iter=args.n_iter, cv_folds=args.cv_folds, seed=args.seed)
    else:
        results = [tune(args.model, n_iter=args.n_iter, cv_folds=args.cv_folds, seed=args.seed)]

    for result in sorted(results, key=lambda r: r.cv_score, reverse=True):
        print(f"{result.cv_score:.4f} +/- {result.cv_std:.4f}  {result.model}  ({result.run_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
