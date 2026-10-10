"""Step 6 — ``tune``: cross-validated model search on the frozen training split.

Every call appends one row per model to ``experiments/results.csv`` via
:mod:`titanic_agent.results_log`. A tuning run that does not log did not happen.

Inputs
    ``data/frozen/train.csv``; candidate features from ``reports/02_explore.json``.
Outputs
    One or more rows in ``experiments/results.csv``; the fitted best estimator
    returned in memory (nothing is pickled into ``data/frozen``).
Done criteria
    The run is in the ledger with its params, the frozen fingerprint and an
    honest ``holdout_read`` attestation, and wall-clock stayed under
    ``time_budget_s`` on a laptop CPU. Whether anything beats the baseline is a
    *result*, not a done criterion.

This is a deliberately shallow search. Breadth across models beats depth within
one at this stage, and the loop is meant to complete in seconds so that the
autonomous pass over it can be run and re-run cheaply.
"""

from __future__ import annotations

import time
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

#: The subset the skeleton loop runs: a baseline and one real model. Enough to
#: exercise the whole pipeline end to end in seconds.
SKELETON_KEYS: tuple[str, ...] = ("dummy", "logistic")

#: Raw columns never offered as features. Step 2 flagged all four as
#: high-cardinality; `PassengerId` and `Name` are effectively row identifiers,
#: which memorise rather than generalise.
EXCLUDED: frozenset[str] = frozenset({"PassengerId", "Name", "Ticket", "Cabin", "Survived"})


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


def add_derived(frame):
    """Add the features step 5 proposed. Pure: returns a new frame.

    Kept out of the sklearn ``Pipeline`` only because it is row-wise and uses no
    fitted state — nothing here learns anything from the data, so computing it
    before the split leaks nothing. Anything that *does* learn (imputation
    statistics, encodings, scaling) lives inside the pipeline and is fitted per
    fold.
    """
    derived = frame.copy()
    derived["FamilySize"] = derived["SibSp"] + derived["Parch"] + 1
    derived["IsAlone"] = (derived["FamilySize"] == 1).astype(int)
    derived["AgeMissing"] = derived["Age"].isna().astype(int)
    derived["Title"] = (
        derived["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
        if "Name" in derived.columns
        else "Unknown"
    )
    # Collapse the long tail: titles seen a handful of times cannot support a
    # per-title estimate and just widen the one-hot matrix.
    if "Name" in derived.columns:
        common = {"Mr", "Mrs", "Miss", "Master"}
        derived["Title"] = derived["Title"].where(derived["Title"].isin(common), "Rare")
    derived["Deck"] = (
        derived["Cabin"].astype("string").str[0].fillna("Unknown")
        if "Cabin" in derived.columns
        else "Unknown"
    )
    # Candidates under test by `make features`. Declared here so every column
    # anything can model on comes from one function — which is also what lets
    # `make leakage` vouch for them.
    from . import features

    return features.add_candidates(derived)


def resolve_features(frame, features: list[str] | None = None) -> list[str]:
    """Decide which columns to model on.

    Args:
        frame: The frame *after* :func:`add_derived`.
        features: Explicit list, or None to read step 5's handoff and fall back
            to every usable column.

    Falling back rather than failing is deliberate: step 6 should still run if
    step 5 has not, so the loop can be exercised in any order.
    """
    from . import explore

    if features is None:
        features = explore.load_candidates() or None
    if features is None:
        from .features import CANDIDATES

        # Candidates are opt-in: they enter a feature list only by passing a test.
        features = [c for c in frame.columns if c not in EXCLUDED and c not in CANDIDATES]
    return [column for column in features if column in frame.columns and column not in EXCLUDED]


def build_pipeline(model_key: str, *, seed: int = paths.SEED, features: list[str] | None = None):
    """Return an unfitted sklearn ``Pipeline`` for ``model_key``.

    The preprocessing half is shared across models — median impute plus scaling
    for numerics, most-frequent impute plus one-hot for categoricals — so that
    scores differ because of the estimator and not the plumbing. All of it sits
    inside the pipeline, so it is fitted per fold: fitting a scaler or an
    imputer on the whole training split before cross-validation leaks across
    folds and inflates the score.

    Args:
        model_key: One of :data:`MODEL_KEYS`.
        seed: Random state for every estimator that takes one.
        features: Columns the pipeline will see. ``None`` builds a selector-free
            pipeline that accepts whatever :func:`resolve_features` chose.

    Raises:
        ValueError: ``model_key`` is not in :data:`MODEL_KEYS`.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.svm import SVC

    if model_key not in MODEL_KEYS:
        raise ValueError(f"unknown model_key {model_key!r}; expected one of {MODEL_KEYS}")

    estimators = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(random_state=seed, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(random_state=seed, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
        # SVC has no predict_proba of its own, which soft voting needs.
        # `SVC(probability=True)` is deprecated in scikit-learn 1.9 and removed
        # in 1.11; this is the documented replacement. It also costs a nested
        # cross-validation, which is why svc is the slowest small model here.
        "svc": CalibratedClassifierCV(SVC(random_state=seed), ensemble=False),
        "knn": KNeighborsClassifier(),
    }

    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, make_column_selector(dtype_include="number")),
            ("categorical", categorical, make_column_selector(dtype_exclude="number")),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocess), ("clf", estimators[model_key])])


def search_space(model_key: str) -> dict[str, list[Any]]:
    """Return the hyperparameter grid for ``model_key``.

    Keys are ``Pipeline``-style dotted paths (e.g. ``clf__max_depth``). Kept
    small; breadth across models beats depth within one here, and the loop has
    to stay cheap enough to rerun.
    """
    if model_key not in MODEL_KEYS:
        raise ValueError(f"unknown model_key {model_key!r}; expected one of {MODEL_KEYS}")

    spaces: dict[str, dict[str, list[Any]]] = {
        "dummy": {},  # nothing to tune; it exists to be the number to beat
        "logistic": {"clf__C": [0.1, 1.0, 10.0]},
        "random_forest": {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [4, 8, None],
            "clf__min_samples_leaf": [1, 3],
        },
        "extra_trees": {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [4, 8, None],
        },
        "gradient_boosting": {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [2, 3],
        },
        "hist_gradient_boosting": {
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_leaf_nodes": [15, 31],
            # exp-004: regularisation, untuned until now. On ~570 rows the
            # defaults (no L2, no depth cap) leave the trees free to memorise.
            "clf__min_samples_leaf": [10, 20, 40],
            "clf__l2_regularization": [0.0, 1.0],
            "clf__max_depth": [3, None],
        },
        # Note the extra `estimator__` hop: the SVC sits inside a
        # CalibratedClassifierCV, so its params are one level deeper.
        "svc": {
            "clf__estimator__C": [0.5, 1.0, 5.0],
            "clf__estimator__gamma": ["scale", "auto"],
        },
        "knn": {"clf__n_neighbors": [3, 5, 11], "clf__weights": ["uniform", "distance"]},
    }
    return spaces[model_key]


def load_training_frame(frozen_dir: Path = paths.FROZEN_DIR):
    """The rows this step may train on, verified, with derived features added.

    When an experiment is open, that is **its** training rows — a subset of the
    working set, with the rest held back as that experiment's validation split.
    With no experiment open, it is the whole working set, which is how the
    baseline ran and how a single step invoked by hand still behaves.

    Reads ``train.csv`` only. ``holdout.csv`` is the final test and this module
    has no legitimate reason to touch it; the experiment's validation rows are
    inside ``train.csv`` but excluded here, which is the point.
    """
    import pandas as pd

    from . import experiments, freeze

    freeze.require_frozen(frozen_dir)
    working = add_derived(pd.read_csv(Path(frozen_dir) / "train.csv"))

    experiment = experiments.current_experiment()
    if experiment is None or not experiment.train_ids:
        return working
    return working.set_index("PassengerId", drop=False).loc[experiment.train_ids]


def tune(
    model_key: str,
    *,
    space: dict[str, list[Any]] | None = None,
    features: list[str] | None = None,
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
        features: Columns to model on; defaults to step 5's handoff.
        n_iter: Candidates for the randomized search. When the grid is smaller
            than ``n_iter`` the search is exhaustive instead — sampling 25 draws
            from a 3-point grid would just evaluate the same points repeatedly.
        cv_folds: Number of stratified folds.
        scoring: sklearn scoring name; the competition metric is accuracy.
        seed: Seeds both the fold shuffling and the search sampler.
        frozen_dir: Frozen dataset; verified before reading.
        log_path: Ledger to append to.
        time_budget_s: Soft budget. Exceeding it is recorded in the ledger row's
            notes rather than raising, so a slow run still leaves a record.
        notes: Free text stored with the ledger row.

    Returns:
        A :class:`RunResult` whose ``run_id`` matches the appended ledger row.

    Raises:
        RuntimeError: The dataset is not frozen or failed verification.
        ValueError: Unknown ``model_key``.
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

    from . import results_log, run_log

    if model_key not in MODEL_KEYS:
        raise ValueError(f"unknown model_key {model_key!r}; expected one of {MODEL_KEYS}")

    frame = load_training_frame(frozen_dir)
    chosen = resolve_features(frame, features)
    design, labels = frame[chosen], frame["Survived"]

    grid = search_space(model_key) if space is None else space
    pipeline = build_pipeline(model_key, seed=seed, features=chosen)
    folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    combinations = 1
    for values in grid.values():
        combinations *= len(values)
    # Exhaustive when the grid is small enough; sampling would revisit points.
    if not grid or combinations <= n_iter:
        search = GridSearchCV(pipeline, grid, cv=folds, scoring=scoring, n_jobs=-1)
    else:
        search = RandomizedSearchCV(
            pipeline,
            grid,
            n_iter=n_iter,
            cv=folds,
            scoring=scoring,
            n_jobs=-1,
            random_state=seed,
        )

    run_log.record("step_started", step="tune", detail=f"{model_key}: {combinations} combos")
    started = time.perf_counter()
    try:
        search.fit(design, labels)
    except Exception as exc:
        # Rule 3: a failed configuration is a result. Dropping it silently
        # makes the sweep a lie, so the failure gets its own ledger row.
        elapsed = time.perf_counter() - started
        run_id = results_log.append_run(
            step="tune",
            model=model_key,
            params={"features": chosen},
            cv_score=None,
            cv_folds=cv_folds,
            seed=seed,
            holdout_read=False,
            notes=f"FAILED after {elapsed:.1f}s: {type(exc).__name__}: {exc}",
            path=log_path,
        )
        run_log.record("step_failed", step="tune", detail=f"{model_key}: {exc}", run_id=run_id)
        return RunResult(run_id=run_id, model=model_key, cv_folds=cv_folds, fit_seconds=elapsed)

    elapsed = time.perf_counter() - started
    best_index = int(search.best_index_)
    cv_std = float(search.cv_results_["std_test_score"][best_index])

    detail = notes
    if elapsed > time_budget_s:
        over = f"over time budget: {elapsed:.1f}s > {time_budget_s:.0f}s"
        detail = f"{notes}; {over}" if notes else over

    run_id = results_log.append_run(
        step="tune",
        model=model_key,
        params={**search.best_params_, "features": chosen},
        cv_score=float(search.best_score_),
        cv_std=cv_std,
        cv_folds=cv_folds,
        seed=seed,
        holdout_read=False,  # attested: this step has no business reading it
        notes=detail,
        path=log_path,
    )
    run_log.record(
        "step_finished",
        step="tune",
        detail=f"{model_key} cv={search.best_score_:.4f} in {elapsed:.1f}s",
        run_id=run_id,
    )

    return RunResult(
        run_id=run_id,
        model=model_key,
        params=dict(search.best_params_),
        cv_score=float(search.best_score_),
        cv_std=cv_std,
        cv_folds=cv_folds,
        fit_seconds=elapsed,
        estimator=search.best_estimator_,
    )


def tune_all(model_keys: tuple[str, ...] = MODEL_KEYS, **kwargs: Any) -> list[RunResult]:
    """Run :func:`tune` across ``model_keys``, logging one row each.

    Continues past a failing model rather than aborting the sweep; the failure
    is recorded in that model's ledger row by :func:`tune` itself.
    """
    results: list[RunResult] = []
    for model_key in model_keys:
        try:
            results.append(tune(model_key, **kwargs))
        except (ValueError, RuntimeError) as exc:
            from . import run_log

            run_log.record("step_failed", step="tune", detail=f"{model_key}: {exc}")
    return results


def main(argv: list[str] | None = None) -> int:
    """``make tune`` entry point."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        default="skeleton",
        help=f"'all', 'skeleton' ({', '.join(SKELETON_KEYS)}), or one of {MODEL_KEYS}",
    )
    parser.add_argument("--n-iter", type=int, default=25)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=paths.SEED)
    args = parser.parse_args(argv)

    if args.model == "all":
        keys = MODEL_KEYS
    elif args.model == "skeleton":
        keys = SKELETON_KEYS
    else:
        keys = (args.model,)

    try:
        results = tune_all(keys, n_iter=args.n_iter, cv_folds=args.cv_folds, seed=args.seed)
    except RuntimeError as exc:
        print(f"tune failed: {exc}")
        return 1

    if not results:
        print("tune produced no results")
        return 1

    for result in sorted(results, key=lambda r: r.cv_score, reverse=True):
        print(
            f"{result.cv_score:.4f} +/- {result.cv_std:.4f}  {result.model:<24}"
            f"{result.fit_seconds:5.1f}s  ({result.run_id})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
