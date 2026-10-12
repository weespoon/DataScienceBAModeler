"""The adjudicator, and the guarantee that there is only one of it.

These tests are mostly about arithmetic and about drift. The arithmetic matters
because every decision in the project is a 0.003-0.015 comparison and the
comparator is the only instrument that can see one. The drift matters because a
second copy of ``DELTA`` or of the search loop would make two experiments
incomparable without either one ever looking wrong — which is exactly the
failure ``docs/MEASUREMENT.md`` describes.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from titanic_agent import compare, features, nested


def test_delta_has_exactly_one_definition() -> None:
    # features.py owned DELTA, CV_FOLDS, CV_REPEATS and fold_scores until the
    # comparator was lifted out. It must now re-export, not redefine.
    assert features.DELTA is compare.DELTA
    assert features.CV_FOLDS is compare.CV_FOLDS
    assert features.CV_REPEATS is compare.CV_REPEATS
    assert features.fold_scores is compare.fold_scores
    source = inspect.getsource(features)
    for name in ("DELTA", "CV_FOLDS", "CV_REPEATS"):
        assert f"\n{name}: float" not in source, f"{name} redefined in features.py"
        assert f"\n{name}: int" not in source, f"{name} redefined in features.py"


def test_the_search_has_exactly_one_implementation() -> None:
    # nested CV must nest the procedure we actually run. A second copy of the
    # greedy loop would estimate a procedure that drifted from the real one.
    assert "greedy_search" in inspect.getsource(nested.run)
    assert "features.greedy_search" in inspect.getsource(nested.run)
    body = inspect.getsource(nested)
    assert "def greedy_search" not in body, "nested.py reimplements the search"


def test_pairing_requires_matching_fold_vectors() -> None:
    with pytest.raises(ValueError, match="not paired"):
        compare.compare_scores(np.zeros(15), np.zeros(10), label="x", n_rows=569, seed=1)


def test_delta_is_a_minus_b_and_sigma_scales_with_it() -> None:
    a = np.array([0.80, 0.82, 0.84])
    b = np.array([0.79, 0.81, 0.83])
    result = compare.compare_scores(a, b, label="a vs b", n_rows=569, seed=1)
    assert result.delta == pytest.approx(0.01)
    # identical fold-to-fold differences means no spread, so no uncertainty
    assert result.naive_se == pytest.approx(0.0)
    assert result.score_a > result.score_b

    flipped = compare.compare_scores(b, a, label="b vs a", n_rows=569, seed=1)
    assert flipped.delta == pytest.approx(-result.delta)


def test_the_quantum_is_one_passenger() -> None:
    result = compare.compare_scores(
        np.array([0.80, 0.81]), np.array([0.80, 0.80]), label="x", n_rows=569, seed=1, repeats=3
    )
    # A delta is an integer number of flipped predictions. Reporting finer than
    # one passenger is reporting rounding.
    assert result.quantum == pytest.approx(1 / 569)
    assert result.fine_quantum == pytest.approx(1 / (569 * 3))
    assert result.passengers == pytest.approx(result.delta * 569)


def test_delta_is_about_one_standard_error_on_this_dataset() -> None:
    """The measured fact that makes every single comparison weak.

    Seed-to-seed spread on exp-006's 569 rows came out 0.0024-0.0041 over ten
    fold seeds, so the 0.003 bar is roughly 0.7-1.2 sigma. If DELTA or the
    sample size ever moves far from that, the reports claiming "about one
    standard error" become wrong and this test should fail.
    """
    assert compare.DELTA / 0.0041 < 1.5
    assert compare.DELTA / 0.0024 > 0.5


def test_describe_always_carries_the_sigma() -> None:
    # A delta and a bar alone let a 0.6-sigma difference read as a decision.
    weak = compare.compare_scores(
        np.array([0.80, 0.84, 0.78]),
        np.array([0.79, 0.845, 0.79]),
        label="weak",
        n_rows=569,
        seed=1,
    )
    assert "sigma" in weak.describe()
    assert "passengers" in weak.describe()
    assert "sigma" in weak.as_params()


def test_resolves_refuses_an_effect_below_its_own_precision() -> None:
    result = compare.compare_scores(
        np.array([0.80, 0.84, 0.78]), np.array([0.79, 0.845, 0.79]), label="x", n_rows=569, seed=1
    )
    assert not result.resolves(0.0001)
    assert result.resolves(10.0)


def test_a_seed_spread_is_not_a_standard_error() -> None:
    """exp-007 reported the sd across 5 seeds as if it were the SE of their mean.

    Both numbers are real and the label said "seed sd", but sigma, the 0.003
    bar and ``champion.jsonl``'s reason text all divided by it, and the one
    inference drawn from the ratio came out backwards. The sd answers "what
    might one fresh seed show"; only sd/sqrt(n) answers "how well is the
    average pinned".
    """
    # exp-007's winner, its published numbers.
    result = compare.PairedResult(
        label="rf_kitchen vs champion",
        delta=0.016343,
        naive_se=0.005284,
        score_a=0.84,
        score_b=0.824,
        n_folds=15,
        n_rows=569,
        repeats=3,
        seed=20251017,
        seed_sd=0.007351,
        n_seeds=5,
    )
    assert result.seed_se == pytest.approx(0.007351 / 5**0.5, rel=1e-6)
    assert result.seed_se < result.naive_se
    # The floor, not the sd: row-draw spread does not shrink with seeds, so
    # whichever of the two is larger is the honest uncertainty.
    assert result.se == pytest.approx(result.naive_se)
    assert result.sigma == pytest.approx(3.09, abs=0.01)
    # The sd stays visible -- it is the right number for a different question.
    assert "seed sd 0.0074 over 5" in result.describe()
    params = result.as_params()
    assert params["se"] == pytest.approx(result.naive_se, abs=1e-6)
    assert params["seed_sd"] == pytest.approx(0.007351, abs=1e-6)
    assert params["n_seeds"] == 5


def test_the_decisions_the_old_denominator_made_are_unchanged() -> None:
    # The error was conservative, so no promotion was wrongly granted. If a
    # correction ever flips a recorded decision, that is a different problem
    # and this test should fail rather than let it pass quietly.
    promoted = compare.PairedResult(
        label="rf_kitchen",
        delta=0.016343,
        naive_se=0.005284,
        score_a=0.84,
        score_b=0.824,
        n_folds=15,
        n_rows=569,
        repeats=3,
        seed=1,
        seed_sd=0.007351,
        n_seeds=5,
    )
    held = compare.PairedResult(
        label="without_max_features",
        delta=-0.0032,
        naive_se=0.0024,
        score_a=0.83,
        score_b=0.8332,
        n_folds=15,
        n_rows=569,
        repeats=3,
        seed=1,
        seed_sd=0.0046,
        n_seeds=5,
    )
    assert promoted.clears(compare.DELTA)
    assert not held.clears(compare.DELTA)


def test_a_seed_spread_with_no_seed_count_cannot_be_corrected() -> None:
    # Rows written before n_seeds existed keep the old reading rather than
    # inventing a divisor for them.
    legacy = compare.PairedResult(
        label="x",
        delta=0.004,
        naive_se=0.002,
        score_a=0.84,
        score_b=0.836,
        n_folds=15,
        n_rows=569,
        repeats=3,
        seed=1,
        seed_sd=0.006,
    )
    assert legacy.seed_se is None
    assert legacy.se == 0.006
    assert "seed sd" in legacy.describe()


def test_seed_sd_overrides_the_naive_se_when_measured() -> None:
    naive_only = compare.PairedResult(
        label="x",
        delta=0.004,
        naive_se=0.002,
        score_a=0.84,
        score_b=0.836,
        n_folds=15,
        n_rows=569,
        repeats=3,
        seed=1,
    )
    assert naive_only.se == 0.002
    assert "naive SE" in naive_only.describe()

    measured = compare.PairedResult(
        label="x",
        delta=0.004,
        naive_se=0.002,
        score_a=0.84,
        score_b=0.836,
        n_folds=15,
        n_rows=569,
        repeats=3,
        seed=1,
        seed_sd=0.006,
    )
    assert measured.se == 0.006
    assert "seed sd" in measured.describe()
    assert measured.sigma < naive_only.sigma
    assert measured.as_params()["seed_sd"] == 0.006


def test_seed_stability_rejects_unknown_arms() -> None:
    with pytest.raises(ValueError, match="arms that were not supplied"):
        compare.seed_stability(None, {"a": ["X"]}, {}, [("x", "a", "missing")], seed=1, seeds=2)
