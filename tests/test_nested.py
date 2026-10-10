"""Nested CV: does it nest the real procedure, and does it refuse a bad number?

Runs against the 12-row fixture with 2 outer folds — enough to exercise the two
arms, the ledger row and the falsifier, not to measure anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from titanic_agent import experiments, features, nested, paths, results_log


@pytest.fixture
def sandbox(frozen_dir: Path, tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "experiments"
    monkeypatch.setattr(paths, "EXPERIMENTS_DIR", root)
    monkeypatch.setattr(paths, "FROZEN_DIR", frozen_dir)
    monkeypatch.setattr(paths, "FINDINGS", root / "findings.jsonl")
    monkeypatch.setattr(paths, "EXPERIMENTS_INDEX", root / "EXPERIMENTS.md")
    monkeypatch.setattr(paths, "VAL_READS", tmp_path / "val_reads.jsonl")
    monkeypatch.delenv(experiments.EXPERIMENT_ID_ENV, raising=False)
    return root


def _search_then_nest(frozen_dir: Path, log_path: Path):
    experiments.open_experiment("nested", frozen_dir=frozen_dir)
    features.select(
        baseline=["Sex", "Pclass", "Age"],
        candidates=["IsChild"],
        folds=2,
        repeats=1,
        tune_folds=2,
        frozen_dir=frozen_dir,
        log_path=log_path,
    )
    return nested.run(
        candidates=["IsChild", "FareLog"],
        outer_folds=2,
        folds=2,
        repeats=1,
        frozen_dir=frozen_dir,
        log_path=log_path,
    )


# --------------------------------------------------------------------------- #
# it needs a search to correct
# --------------------------------------------------------------------------- #


def test_nesting_refuses_when_no_search_has_happened(
    sandbox: Path, frozen_dir: Path, log_path: Path
) -> None:
    # Nesting measures the bias of a search that actually ran. Without one
    # there is no number to correct and nothing to report.
    with pytest.raises(RuntimeError, match="no `features` rows"):
        nested.run(outer_folds=2, frozen_dir=frozen_dir, log_path=log_path)


def test_it_recovers_the_real_run_params_from_the_ledger(
    sandbox: Path, frozen_dir: Path, log_path: Path
) -> None:
    result = _search_then_nest(frozen_dir, log_path)
    # The params come off the ledger rather than being restated, so the nested
    # estimate corrects a number that exists in the chain.
    assert result.baseline == ["Sex", "Pclass", "Age"]
    assert all(key.startswith("clf__") for key in result.params)
    assert result.reported_cv is not None


# --------------------------------------------------------------------------- #
# the two arms
# --------------------------------------------------------------------------- #


def test_both_arms_are_scored_on_identical_outer_folds(
    sandbox: Path, frozen_dir: Path, log_path: Path
) -> None:
    result = _search_then_nest(frozen_dir, log_path)
    assert len(result.folds) == 2
    for fold in result.folds:
        # One arm cannot see more rows than the other, or the difference is
        # not paired and the whole design collapses to two noisy estimates.
        assert fold.n_train + fold.n_test == result.n_rows
        assert fold.delta == fold.score_search - fold.score_baseline
    assert result.paired.n_folds == 2


def test_the_paired_difference_is_search_minus_baseline(
    sandbox: Path, frozen_dir: Path, log_path: Path
) -> None:
    result = _search_then_nest(frozen_dir, log_path)
    expected = np.mean([f.score_search for f in result.folds]) - np.mean(
        [f.score_baseline for f in result.folds]
    )
    assert result.paired.delta == pytest.approx(expected)
    assert result.nested_search == pytest.approx(np.mean([f.score_search for f in result.folds]))


def test_the_inner_search_never_sees_the_outer_test_rows(
    sandbox: Path, frozen_dir: Path, log_path: Path, monkeypatch
) -> None:
    # The real search first, unspied -- it also calls greedy_search, and on the
    # whole frame, which is legitimate there and would pollute the recording.
    experiments.open_experiment("nested", frozen_dir=frozen_dir)
    features.select(
        baseline=["Sex", "Pclass", "Age"],
        candidates=["IsChild"],
        folds=2,
        repeats=1,
        tune_folds=2,
        frozen_dir=frozen_dir,
        log_path=log_path,
    )

    seen: list[int] = []
    original = features.greedy_search

    def spy(frame, *args, **kwargs):
        seen.append(len(frame))
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(features, "greedy_search", spy)
    result = nested.run(
        candidates=["IsChild", "FareLog"],
        outer_folds=2,
        folds=2,
        repeats=1,
        frozen_dir=frozen_dir,
        log_path=log_path,
    )

    # Each inner search got exactly its outer fold's training rows -- never the
    # whole frame, which would leak the outer test rows into selection.
    assert seen == [fold.n_train for fold in result.folds]
    assert all(n < result.n_rows for n in seen)


# --------------------------------------------------------------------------- #
# the falsifier
# --------------------------------------------------------------------------- #


def test_a_nested_estimate_above_the_reported_cv_is_flagged_implausible() -> None:
    # Nesting cannot improve on the number it corrects. If it does, the
    # implementation is wrong and the number must not be quoted.
    result = nested.NestedResult(
        model="random_forest", params={}, baseline=[], candidates=[], seed=1, n_rows=100
    )
    result.folds = [
        nested.OuterFold(1, 80, 20, [], 0.90, 0.88, 0.90, 3),
        nested.OuterFold(2, 80, 20, [], 0.90, 0.88, 0.90, 3),
    ]
    result.reported_cv = 0.84
    assert result.selection_bias == pytest.approx(0.84 - 0.90)
    assert result.implausible
    assert "IMPLAUSIBLE" in nested.render_markdown(result) or "not trustworthy" in (
        nested.render_markdown(result)
    )

    result.reported_cv = 0.95
    assert result.selection_bias == pytest.approx(0.05)
    assert not result.implausible


def test_main_exits_non_zero_on_an_implausible_estimate(monkeypatch) -> None:
    result = nested.NestedResult(
        model="random_forest", params={}, baseline=[], candidates=[], seed=1, n_rows=100
    )
    result.folds = [nested.OuterFold(1, 80, 20, [], 0.90, 0.88, 0.90, 3)]
    result.reported_cv = 0.84
    monkeypatch.setattr(nested, "run", lambda **kwargs: result)
    monkeypatch.setattr(nested, "render_markdown", lambda r: "report")

    from titanic_agent import runs

    monkeypatch.setattr(runs, "publish", lambda *a, **k: None)
    assert nested.main([]) == 1


# --------------------------------------------------------------------------- #
# the ledger
# --------------------------------------------------------------------------- #


def test_it_logs_one_scored_row_attesting_no_holdout_read(
    sandbox: Path, frozen_dir: Path, log_path: Path
) -> None:
    result = _search_then_nest(frozen_dir, log_path)
    frame = results_log.read_log(log_path)
    rows = frame[frame["step"] == "nested"]
    assert len(rows) == 1
    row = rows.iloc[0]
    assert float(row["cv_score"]) == pytest.approx(result.nested_search)
    assert int(row["cv_folds"]) == 2
    assert not bool(row["holdout_read"])
    # the trimming is part of the procedure being measured, so it is recorded
    assert row["params"]["candidates"] == 2
    assert row["params"]["arms"] == "search vs fixed baseline"
    assert row["params"]["corrects_run_id"]


def test_the_report_states_what_is_not_nested(
    sandbox: Path, frozen_dir: Path, log_path: Path
) -> None:
    result = _search_then_nest(frozen_dir, log_path)
    report = nested.render_markdown(result)
    # Hyperparameters are held fixed, so the estimate covers the feature search
    # alone. A report that did not say so would overclaim.
    assert "Hyperparameters are **not** nested" in report
    assert "trimmed" in report
