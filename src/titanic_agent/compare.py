"""The adjudicator: paired repeated CV, with its own precision measured.

Every decision in this project is a comparison of two configurations whose
scores differ by 0.003 to 0.015. Three instruments could in principle settle
such a comparison, and only one of them can:

=========================  ======  ==========  ==================================
instrument                   rows   can see     role
=========================  ======  ==========  ==================================
paired repeated CV            569   ~0.003      **adjudicates** — compares two arms
validation read               143   ~0.064      **tripwire** — catches a broken pipeline
final test                    179   ~0.058      **reports** — one generalisation number
=========================  ======  ==========  ==================================

The validation split cannot adjudicate, at any number of reads. Its accuracy
moves in steps of 1/143 = 0.0070, so two configurations that truly differ by
0.004 cannot differ on it by 0.004 — they differ by 0, 0.007 or 0.014,
whichever way the disagreeing rows happen to fall. No amount of statistical
machinery recovers a difference smaller than the instrument's own quantum. See
``docs/MEASUREMENT.md`` for the full arithmetic and why the reusable-holdout
literature does not rescue it either.

So comparisons live here, paired on identical folds with identically seeded
estimators, which cancels most of the split noise out of the *difference* even
though each arm's absolute score stays noisy.

What this module measures about itself
    A comparator that reports an SE nobody has checked is a ruler with
    unlabelled gradations. :func:`seed_stability` re-runs the same comparison
    under several fold seeds and returns the spread of the paired delta — the
    component a single run's SE cannot see, because one run holds the fold draw
    fixed. Measured on exp-006's rows over 10 seeds, the naive SE came within a
    factor of 0.77-1.37 of that spread, so the SE this module reports is
    approximately honest. The Nadeau-Bengio resampled-t correction (x2.18) is
    too conservative for a fully paired statistic and is not applied.

    The same measurement found the thing that matters more: at these sample
    sizes a 0.003 bar is about **one standard error**, so accept/reject near the
    bar is close to a coin flip. That is a fact about the dataset, not a bug —
    but it belongs in every report that leans on a single comparison.

Two numbers every result carries
    ``naive_se``
        Standard error of the paired fold differences. Repeated-CV folds
        overlap, so it is not a textbook SE; it is what the measurement above
        validates.
    ``quantum``
        One passenger, 1/569 = 0.00176. A delta is always an integer number of
        flipped predictions, and reporting a difference finer than one
        passenger is reporting rounding. ``DELTA`` is 1.7 passengers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Minimum gain a configuration must buy to be preferred. About 1.7 flipped
#: predictions on 569 rows, and -- per the seed-stability measurement above --
#: roughly one standard error, so it is a weak bar by construction. Raising it
#: would reject real-but-small effects; the honest fix is to report the sigma
#: alongside, which :meth:`PairedResult.describe` does.
DELTA: float = 0.003

#: Repeated stratified K-fold: ``CV_FOLDS`` x ``CV_REPEATS`` paired scores.
CV_FOLDS: int = 5
CV_REPEATS: int = 3

#: Default estimator for a comparison. Overridable per call.
MODEL_KEY: str = "random_forest"

#: Fold seeds :func:`seed_stability` uses when the caller does not say. Ten is
#: enough to tell a factor-of-two error in the SE from agreement, which is the
#: question; it is nowhere near enough to pin the SE to two digits.
STABILITY_SEEDS: int = 10


@dataclass(frozen=True)
class PairedResult:
    """One paired comparison of two configurations on identical folds.

    ``delta`` is always *a minus b*, so a positive delta favours arm ``a``.
    Callers that think in terms of a feature's worth should name the arms so
    that "a" is the configuration that has it.
    """

    label: str
    delta: float
    naive_se: float
    score_a: float
    score_b: float
    n_folds: int
    n_rows: int
    repeats: int
    seed: int
    fold_diffs: tuple[float, ...] = ()
    #: Spread of ``delta`` across fold seeds, when :func:`seed_stability` ran.
    #: ``None`` means nobody measured it for this comparison.
    seed_sd: float | None = None

    @property
    def quantum(self) -> float:
        """One consistently flipped prediction: the finest honest resolution."""
        return 1.0 / self.n_rows

    @property
    def fine_quantum(self) -> float:
        """One prediction flipped in one repeat — the arithmetic step size."""
        return 1.0 / (self.n_rows * self.repeats)

    @property
    def se(self) -> float:
        """The best available uncertainty: measured spread if there is one."""
        return self.naive_se if self.seed_sd is None else self.seed_sd

    @property
    def sigma(self) -> float:
        """``delta`` in units of its own standard error."""
        return self.delta / self.se if self.se else 0.0

    @property
    def passengers(self) -> float:
        """``delta`` expressed as flipped predictions."""
        return self.delta / self.quantum

    def clears(self, bar: float = DELTA) -> bool:
        """Does the delta meet the declared bar? The rule, and only the rule."""
        return self.delta >= bar

    def resolves(self, effect: float) -> bool:
        """Could this instrument see an effect of this size at all (2 SE)?"""
        return abs(effect) >= 2 * self.se and abs(effect) >= self.quantum

    def describe(self) -> str:
        """One line carrying the delta, its uncertainty, and its sigma.

        Always prints the sigma, because a delta and a bar alone let a 0.6-sigma
        difference read as a decision. Marks a delta finer than one passenger,
        which is rounding rather than signal.
        """
        spread = f"{self.se:.4f}"
        source = "seed sd" if self.seed_sd is not None else "naive SE"
        note = "" if abs(self.passengers) >= 1 else "  [under one passenger]"
        return (
            f"{self.label}: {self.delta:+.4f} +/- {spread} ({source}, "
            f"{self.sigma:+.2f} sigma; {self.passengers:+.1f} passengers){note}"
        )

    def as_params(self) -> dict[str, Any]:
        """The ledger-friendly form: what was compared and how well."""
        payload: dict[str, Any] = {
            "comparison": self.label,
            "delta": round(self.delta, 6),
            "naive_se": round(self.naive_se, 6),
            "sigma": round(self.sigma, 3),
            "passengers": round(self.passengers, 2),
            "n_folds": self.n_folds,
            "n_rows": self.n_rows,
        }
        if self.seed_sd is not None:
            payload["seed_sd"] = round(self.seed_sd, 6)
        return payload


def fold_scores(
    frame,
    features: list[str],
    params: dict[str, Any],
    *,
    seed: int,
    folds: int = CV_FOLDS,
    repeats: int = CV_REPEATS,
    model_key: str = MODEL_KEY,
) -> np.ndarray:
    """Accuracy on every fold of a seeded repeated stratified K-fold.

    The fold generator and the estimator seed depend only on ``seed``, so two
    calls with different ``features`` are scored on identical splits by
    identically-seeded estimators — which is what makes the difference
    *paired*, and is the whole reason this comparator can see 0.003 at all.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

    from . import tune

    pipeline = tune.build_pipeline(model_key, seed=seed)
    # One core per estimator; the folds run in parallel instead.
    pipeline.set_params(**params, clf__n_jobs=1)
    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    return cross_val_score(
        pipeline, frame[features], frame["Survived"], cv=splitter, scoring="accuracy", n_jobs=-1
    )


def compare_scores(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    label: str,
    n_rows: int,
    seed: int,
    repeats: int = CV_REPEATS,
    seed_sd: float | None = None,
) -> PairedResult:
    """Build a :class:`PairedResult` from two fold vectors already computed.

    Separate from :func:`compare` so a caller that needs several comparisons
    against one arm — a feature search does, for every candidate — pays for
    each arm once instead of once per comparison.
    """
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"paired comparison needs matching fold vectors, got {a.shape} and {b.shape}; "
            "both arms must be scored on the same folds or the difference is not paired"
        )
    diff = a - b
    return PairedResult(
        label=label,
        delta=float(diff.mean()),
        naive_se=float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0,
        score_a=float(a.mean()),
        score_b=float(b.mean()),
        n_folds=len(diff),
        n_rows=int(n_rows),
        repeats=repeats,
        seed=seed,
        fold_diffs=tuple(float(d) for d in diff),
        seed_sd=seed_sd,
    )


def compare(
    frame,
    features_a: list[str],
    features_b: list[str],
    params: dict[str, Any],
    *,
    seed: int,
    label: str = "",
    folds: int = CV_FOLDS,
    repeats: int = CV_REPEATS,
    model_key: str = MODEL_KEY,
) -> PairedResult:
    """Score two feature sets on identical folds and return the paired result."""
    kwargs = dict(seed=seed, folds=folds, repeats=repeats, model_key=model_key)
    scores_a = fold_scores(frame, features_a, params, **kwargs)
    scores_b = fold_scores(frame, features_b, params, **kwargs)
    return compare_scores(
        scores_a,
        scores_b,
        label=label or f"{len(features_a)} features vs {len(features_b)}",
        n_rows=len(frame),
        seed=seed,
        repeats=repeats,
    )


def seed_stability(
    frame,
    arms: dict[str, list[str]],
    params: dict[str, Any],
    comparisons: Sequence[tuple[str, str, str]],
    *,
    seed: int,
    seeds: int = STABILITY_SEEDS,
    folds: int = CV_FOLDS,
    repeats: int = CV_REPEATS,
    model_key: str = MODEL_KEY,
) -> list[PairedResult]:
    """Re-run comparisons under several fold seeds; report the spread.

    This is how the adjudicator gets audited. A single run's ``naive_se``
    describes variation *across folds of one draw*; it cannot see variation
    *between draws*, which is exactly what differs between two experiments. So
    each comparison is repeated under ``seeds`` fold seeds and the returned
    result carries ``seed_sd`` — the sd of the paired delta across them — with
    ``delta`` averaged over seeds rather than taken from any one.

    Args:
        arms: Named feature sets. Each is scored once per seed, so several
            comparisons sharing an arm cost nothing extra.
        comparisons: ``(label, arm_a, arm_b)`` triples; delta is a minus b.

    Returns:
        One result per comparison, seed-averaged, with ``seed_sd`` set.
    """
    unknown = {name for _, a, b in comparisons for name in (a, b)} - set(arms)
    if unknown:
        raise ValueError(f"comparisons reference arms that were not supplied: {sorted(unknown)}")

    fold_seeds = [seed + 1000 * index for index in range(seeds)]
    scored: dict[tuple[int, str], np.ndarray] = {}
    for fold_seed in fold_seeds:
        for name, feature_list in arms.items():
            scored[(fold_seed, name)] = fold_scores(
                frame,
                feature_list,
                params,
                seed=fold_seed,
                folds=folds,
                repeats=repeats,
                model_key=model_key,
            )

    results: list[PairedResult] = []
    for label, arm_a, arm_b in comparisons:
        per_seed = [
            compare_scores(
                scored[(fold_seed, arm_a)],
                scored[(fold_seed, arm_b)],
                label=label,
                n_rows=len(frame),
                seed=fold_seed,
                repeats=repeats,
            )
            for fold_seed in fold_seeds
        ]
        deltas = np.array([result.delta for result in per_seed], dtype=float)
        results.append(
            PairedResult(
                label=label,
                delta=float(deltas.mean()),
                naive_se=float(np.mean([result.naive_se for result in per_seed])),
                score_a=float(np.mean([result.score_a for result in per_seed])),
                score_b=float(np.mean([result.score_b for result in per_seed])),
                n_folds=per_seed[0].n_folds,
                n_rows=len(frame),
                repeats=repeats,
                seed=seed,
                fold_diffs=tuple(float(value) for value in deltas),
                seed_sd=float(deltas.std(ddof=1)) if len(deltas) > 1 else None,
            )
        )
    return results
