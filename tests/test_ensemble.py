"""Smoke test: step 7, ``ensemble``.

Note ``test_holdout_scoring_is_opt_in``: the holdout read must be something the
caller asks for explicitly, so that the default path through this step cannot
spend the final estimate by accident.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from titanic_agent import ensemble, paths


def test_public_api_exists() -> None:
    for name in ("select_members", "ensemble", "write_submission", "main"):
        assert callable(getattr(ensemble, name)), name


def test_signature_matches_the_documented_contract() -> None:
    params = inspect.signature(ensemble.ensemble).parameters
    assert set(params) == {
        "members",
        "method",
        "weights",
        "cv_folds",
        "scoring",
        "seed",
        "frozen_dir",
        "log_path",
        "score_holdout",
        "submission_path",
        "report_path",
    }
    assert params["method"].default == "soft_vote"
    assert params["seed"].default == paths.SEED


def test_holdout_scoring_is_opt_in() -> None:
    assert inspect.signature(ensemble.ensemble).parameters["score_holdout"].default is False


def test_methods_are_the_documented_three() -> None:
    assert ensemble.METHODS == ("soft_vote", "hard_vote", "stack")


def test_result_dataclass_defaults_leave_holdout_unscored() -> None:
    result = ensemble.EnsembleResult()
    assert result.holdout_score is None
    assert result.members == [] and result.weights == []


def test_select_members_signature() -> None:
    params = inspect.signature(ensemble.select_members).parameters
    assert set(params) == {"log_path", "top_k", "min_cv_score", "max_correlation"}
    assert params["top_k"].default == 4


@pytest.mark.xfail(raises=NotImplementedError, reason="step 7 not implemented yet")
def test_select_members_returns_at_most_top_k(log_path: Path) -> None:
    from titanic_agent import results_log

    for index in range(6):
        results_log.append_run(
            step="tune", model=f"m{index}", cv_score=0.70 + index / 100, path=log_path
        )
    assert len(ensemble.select_members(log_path=log_path, top_k=3)) <= 3


@pytest.mark.xfail(raises=NotImplementedError, reason="step 7 not implemented yet")
def test_ensemble_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError):
        ensemble.ensemble(members=[{"model": "logistic"}, {"model": "dummy"}], method="telepathy")


@pytest.mark.xfail(raises=NotImplementedError, reason="step 7 not implemented yet")
def test_ensemble_logs_one_row_and_writes_a_submission(tmp_path: Path, log_path: Path) -> None:
    from titanic_agent import results_log

    result = ensemble.ensemble(
        cv_folds=2,
        log_path=log_path,
        submission_path=tmp_path / "submission.csv",
        report_path=tmp_path / "03_ensemble.md",
    )
    frame = results_log.read_log(log_path)
    assert len(frame) == 1 and frame.iloc[0]["step"] == "ensemble"
    assert result.submission_path.is_file()


@pytest.mark.xfail(raises=NotImplementedError, reason="step 7 not implemented yet")
def test_write_submission_rejects_bad_labels(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ensemble.write_submission(
            [0, 1, 2],
            passenger_ids=[892, 893, 894],
            submission_path=tmp_path / "submission.csv",
        )
