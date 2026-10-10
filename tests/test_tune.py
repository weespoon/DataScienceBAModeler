"""Smoke test: step 6, ``tune``.

Runs against the miniature frozen fixture with ``cv_folds=2`` — the 12-row
fixture cannot support 5 stratified folds. Every ledger write goes to a
``tmp_path`` file; the repo's own ``experiments/results.csv`` is never touched.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from _helpers import raises_real_runtime_error

from titanic_agent import paths, results_log, tune


@pytest.fixture
def tuned(frozen_dir: Path, log_path: Path):
    """One cheap real tuning run against the fixture."""
    return tune.tune("logistic", frozen_dir=frozen_dir, log_path=log_path, cv_folds=2)


# --------------------------------------------------------------------------- #
# contract
# --------------------------------------------------------------------------- #


def test_public_api_exists() -> None:
    for name in ("build_pipeline", "search_space", "tune", "tune_all", "main"):
        assert callable(getattr(tune, name)), name


def test_signature_matches_the_documented_contract() -> None:
    params = inspect.signature(tune.tune).parameters
    assert set(params) == {
        "model_key",
        "space",
        "features",
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


def test_model_keys_include_a_baseline_and_stay_laptop_sized() -> None:
    assert "dummy" in tune.MODEL_KEYS
    assert "logistic" in tune.MODEL_KEYS
    assert len(tune.MODEL_KEYS) <= 10


def test_run_result_mirrors_a_ledger_row() -> None:
    result = tune.RunResult()
    for field_name in ("run_id", "model", "params", "cv_score", "cv_std", "cv_folds"):
        assert hasattr(result, field_name)
        assert field_name in results_log.COLUMNS


def test_tune_takes_no_holdout_argument() -> None:
    assert not any("holdout" in name for name in inspect.signature(tune.tune).parameters)


def test_the_source_never_reads_the_holdout_file() -> None:
    source = Path(tune.__file__).read_text(encoding="utf-8")
    reads = [line for line in source.splitlines() if "read_csv(" in line]
    assert reads
    assert not [line for line in reads if "holdout" in line.lower()]


# --------------------------------------------------------------------------- #
# pipeline construction
# --------------------------------------------------------------------------- #


def test_build_pipeline_rejects_unknown_models() -> None:
    with pytest.raises(ValueError, match="unknown model_key"):
        tune.build_pipeline("not-a-model")


def test_search_space_rejects_unknown_models() -> None:
    with pytest.raises(ValueError, match="unknown model_key"):
        tune.search_space("not-a-model")


@pytest.mark.parametrize("model_key", tune.MODEL_KEYS)
def test_every_model_key_builds(model_key: str) -> None:
    pipeline = tune.build_pipeline(model_key)
    assert pipeline.steps[-1][0] == "clf"


@pytest.mark.parametrize("model_key", tune.MODEL_KEYS)
def test_every_search_space_is_pipeline_scoped(model_key: str) -> None:
    space = tune.search_space(model_key)
    assert all("__" in key for key in space), space


@pytest.mark.parametrize("model_key", tune.MODEL_KEYS)
def test_every_search_space_key_is_a_real_parameter(model_key: str) -> None:
    # Catches the class of bug where wrapping an estimator moves its params a
    # level deeper -- svc inside CalibratedClassifierCV needs
    # clf__estimator__C, not clf__C.
    pipeline = tune.build_pipeline(model_key)
    valid = set(pipeline.get_params(deep=True))
    assert set(tune.search_space(model_key)) <= valid


def test_preprocessing_lives_inside_the_pipeline() -> None:
    # Fitting an imputer or scaler before cross-validation leaks across folds.
    pipeline = tune.build_pipeline("logistic")
    assert pipeline.steps[0][0] == "preprocess"


# --------------------------------------------------------------------------- #
# features and the step 5 handoff
# --------------------------------------------------------------------------- #


def test_add_derived_is_pure(frozen_dir: Path) -> None:
    import pandas as pd

    frame = pd.read_csv(frozen_dir / "train.csv")
    before = frame.copy()
    tune.add_derived(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_add_derived_adds_the_promised_columns(frozen_dir: Path) -> None:
    import pandas as pd

    derived = tune.add_derived(pd.read_csv(frozen_dir / "train.csv"))
    assert {"FamilySize", "IsAlone", "AgeMissing", "Title", "Deck"} <= set(derived.columns)


def test_family_size_counts_the_passenger(frozen_dir: Path) -> None:
    import pandas as pd

    derived = tune.add_derived(pd.read_csv(frozen_dir / "train.csv"))
    assert (derived["FamilySize"] == derived["SibSp"] + derived["Parch"] + 1).all()
    assert (derived.loc[derived["FamilySize"] == 1, "IsAlone"] == 1).all()


def test_resolve_features_drops_identifier_columns(frozen_dir: Path) -> None:
    import pandas as pd

    derived = tune.add_derived(pd.read_csv(frozen_dir / "train.csv"))
    chosen = tune.resolve_features(derived, ["PassengerId", "Name", "Sex", "Pclass"])
    assert chosen == ["Sex", "Pclass"]


def test_resolve_features_never_returns_the_target(frozen_dir: Path) -> None:
    import pandas as pd

    derived = tune.add_derived(pd.read_csv(frozen_dir / "train.csv"))
    assert "Survived" not in tune.resolve_features(derived, None)


def test_resolve_features_ignores_columns_that_do_not_exist(frozen_dir: Path) -> None:
    import pandas as pd

    derived = tune.add_derived(pd.read_csv(frozen_dir / "train.csv"))
    assert tune.resolve_features(derived, ["Sex", "NoSuchColumn"]) == ["Sex"]


# --------------------------------------------------------------------------- #
# the ledger contract
# --------------------------------------------------------------------------- #


def test_tune_appends_exactly_one_ledger_row(tuned, log_path: Path) -> None:
    frame = results_log.read_log(log_path)
    assert len(frame) == 1
    assert frame.iloc[0]["run_id"] == tuned.run_id
    assert frame.iloc[0]["step"] == "tune"


def test_the_row_attests_that_the_holdout_was_not_read(tuned, log_path: Path) -> None:
    # `is False` fails here: pandas reads the CSV boolean back as numpy's
    # np.False_, which is falsy but not the Python singleton.
    assert not results_log.read_log(log_path).iloc[0]["holdout_read"]


def test_the_row_carries_the_frozen_fingerprint(tuned, log_path: Path) -> None:
    row = results_log.read_log(log_path).iloc[0]
    assert row["frozen_sha256"] not in ("", "unfrozen")


def test_the_row_records_the_features_used(tuned, log_path: Path) -> None:
    params = results_log.read_log(log_path).iloc[0]["params"]
    assert "features" in params and params["features"]


def test_a_failing_model_still_logs_a_row(frozen_dir: Path, log_path: Path) -> None:
    # Rule 3: a failed configuration is a result. Dropping it silently makes
    # the sweep a lie. An impossible grid value forces the failure.
    result = tune.tune(
        "logistic",
        space={"clf__C": ["not-a-number"]},
        frozen_dir=frozen_dir,
        log_path=log_path,
        cv_folds=2,
    )
    frame = results_log.read_log(log_path)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["run_id"] == result.run_id
    assert "FAILED" in row["notes"]
    import pandas as pd

    assert pd.isna(row["cv_score"])


def test_tune_all_logs_one_row_per_model(frozen_dir: Path, log_path: Path) -> None:
    results = tune.tune_all(
        ("dummy", "logistic"), frozen_dir=frozen_dir, log_path=log_path, cv_folds=2
    )
    assert len(results) == 2
    assert len(results_log.read_log(log_path)) == 2


def test_tune_all_continues_past_a_failing_model(frozen_dir: Path, log_path: Path) -> None:
    results = tune.tune_all(
        ("dummy", "not-a-model", "logistic"),
        frozen_dir=frozen_dir,
        log_path=log_path,
        cv_folds=2,
    )
    assert [r.model for r in results] == ["dummy", "logistic"]


def test_the_dummy_baseline_is_the_majority_class(frozen_dir: Path, log_path: Path) -> None:
    result = tune.tune("dummy", frozen_dir=frozen_dir, log_path=log_path, cv_folds=2)
    assert 0.0 < result.cv_score <= 1.0


def test_exceeding_the_time_budget_is_noted_not_raised(frozen_dir: Path, log_path: Path) -> None:
    tune.tune("logistic", frozen_dir=frozen_dir, log_path=log_path, cv_folds=2, time_budget_s=0.0)
    assert "over time budget" in results_log.read_log(log_path).iloc[0]["notes"]


def test_tune_rejects_unknown_models(frozen_dir: Path, log_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown model_key"):
        tune.tune("not-a-model", frozen_dir=frozen_dir, log_path=log_path)


def test_tune_refuses_an_unfrozen_dataset(tmp_path: Path, log_path: Path) -> None:
    with raises_real_runtime_error():
        tune.tune("logistic", frozen_dir=tmp_path, log_path=log_path, cv_folds=2)


def test_tune_refuses_a_tampered_dataset(writable_frozen_dir: Path, log_path: Path) -> None:
    (writable_frozen_dir / "train.csv").write_text("broken\n", encoding="utf-8")
    with raises_real_runtime_error():
        tune.tune("logistic", frozen_dir=writable_frozen_dir, log_path=log_path, cv_folds=2)
