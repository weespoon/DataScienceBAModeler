"""Smoke test: step 6, ``tune``."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from titanic_agent import paths, tune


def test_public_api_exists() -> None:
    for name in ("build_pipeline", "search_space", "tune", "tune_all", "main"):
        assert callable(getattr(tune, name)), name


def test_signature_matches_the_documented_contract() -> None:
    params = inspect.signature(tune.tune).parameters
    assert set(params) == {
        "model_key",
        "space",
        "n_iter",
        "cv_folds",
        "scoring",
        "seed",
        "frozen_dir",
        "log_path",
        "time_budget_s",
        "notes",
    }
    assert params["cv_folds"].default == 5
    assert params["scoring"].default == "accuracy"
    assert params["seed"].default == paths.SEED
    assert params["log_path"].default == paths.RESULTS_LOG


def test_model_keys_include_a_baseline_and_stay_laptop_sized() -> None:
    assert "dummy" in tune.MODEL_KEYS  # a baseline to beat
    assert "logistic" in tune.MODEL_KEYS
    assert len(tune.MODEL_KEYS) <= 10


def test_run_result_mirrors_a_ledger_row() -> None:
    from titanic_agent import results_log

    result = tune.RunResult()
    for field_name in ("run_id", "model", "params", "cv_score", "cv_std", "cv_folds"):
        assert hasattr(result, field_name)
        assert field_name in results_log.COLUMNS


def test_tune_takes_no_holdout_argument() -> None:
    assert not any("holdout" in name for name in inspect.signature(tune.tune).parameters)


@pytest.mark.xfail(raises=(NotImplementedError, ValueError), reason="step 6 not implemented yet")
def test_build_pipeline_rejects_unknown_models() -> None:
    with pytest.raises(ValueError):
        tune.build_pipeline("not-a-model")


@pytest.mark.xfail(raises=NotImplementedError, reason="step 6 not implemented yet")
def test_search_space_keys_are_pipeline_scoped() -> None:
    space = tune.search_space("random_forest")
    assert space and all("__" in key for key in space)


@pytest.mark.xfail(raises=NotImplementedError, reason="step 6 not implemented yet")
def test_tune_appends_exactly_one_ledger_row(tmp_path: Path, log_path: Path) -> None:
    from titanic_agent import results_log

    result = tune.tune("logistic", n_iter=2, cv_folds=2, log_path=log_path)
    frame = results_log.read_log(log_path)
    assert len(frame) == 1
    assert frame.iloc[0]["run_id"] == result.run_id
    assert frame.iloc[0]["step"] == "tune"
