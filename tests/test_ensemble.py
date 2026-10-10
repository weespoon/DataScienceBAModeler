"""Smoke test: step 7, ``ensemble``.

The tests that matter most are the Rule 2 ones. ``ensemble`` is the only module
allowed to read the holdout, so it is the only place the protocol can be got
wrong — and the order of its three steps (check, read, record) is the whole
guarantee.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from _helpers import raises_real_runtime_error

from titanic_agent import ensemble, paths, receipts, results_log, tune


@pytest.fixture
def populated_ledger(frozen_dir: Path, log_path: Path) -> Path:
    """A ledger holding two real tune rows against the fixture dataset."""
    tune.tune_all(
        ("dummy", "logistic", "random_forest"),
        frozen_dir=frozen_dir,
        log_path=log_path,
        cv_folds=2,
    )
    return log_path


@pytest.fixture
def built(frozen_dir: Path, populated_ledger: Path, tmp_path: Path):
    """One ensemble run with the holdout left alone."""
    reports = tmp_path / "reports"
    return ensemble.ensemble(
        members=ensemble.select_members(log_path=populated_ledger, top_k=2),
        method="hard_vote",
        cv_folds=2,
        frozen_dir=frozen_dir,
        log_path=populated_ledger,
        submission_path=reports / "submission.csv",
        report_path=reports / "03_ensemble.md",
    )


# --------------------------------------------------------------------------- #
# contract
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# member selection
# --------------------------------------------------------------------------- #


def test_select_members_signature() -> None:
    params = inspect.signature(ensemble.select_members).parameters
    assert set(params) == {"log_path", "top_k", "min_cv_score", "max_correlation"}
    assert params["top_k"].default == 4


def test_select_members_returns_at_most_top_k(
    frozen_dir: Path, populated_ledger: Path, monkeypatch
) -> None:
    monkeypatch.setattr(paths, "FROZEN_DIR", frozen_dir)
    assert len(ensemble.select_members(log_path=populated_ledger, top_k=2)) <= 2


def test_select_members_never_picks_the_baseline(
    frozen_dir: Path, populated_ledger: Path, monkeypatch
) -> None:
    # dummy exists to be the number to beat, not to be ensembled.
    monkeypatch.setattr(paths, "FROZEN_DIR", frozen_dir)
    members = ensemble.select_members(log_path=populated_ledger, top_k=9)
    assert "dummy" not in {m["model"] for m in members}


def test_select_members_is_empty_on_an_empty_ledger(
    frozen_dir: Path, log_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(paths, "FROZEN_DIR", frozen_dir)
    results_log.ensure_log(log_path)
    assert ensemble.select_members(log_path=log_path) == []


def test_select_members_ignores_rows_from_another_frozen_dataset(
    frozen_dir: Path, populated_ledger: Path, monkeypatch
) -> None:
    # A score computed against different bytes is not comparable to one
    # computed against these.
    monkeypatch.setattr(paths, "FROZEN_DIR", frozen_dir)
    results_log.append_run(
        step="tune",
        model="knn",
        cv_score=0.99,
        seed=1,
        path=populated_ledger,
        run_id="from-another-dataset",
    )
    rows = results_log.read_log(populated_ledger)
    rows.loc[rows["run_id"] == "from-another-dataset", "frozen_sha256"] = "d" * 64
    rows["params"] = rows["params"].map(json.dumps)
    rows.to_csv(populated_ledger, index=False)

    members = ensemble.select_members(log_path=populated_ledger, top_k=9)
    assert "from-another-dataset" not in {m["run_id"] for m in members}


# --------------------------------------------------------------------------- #
# a run that leaves the holdout alone
# --------------------------------------------------------------------------- #


def test_ensemble_logs_one_row(built, populated_ledger: Path) -> None:
    frame = results_log.read_log(populated_ledger)
    ensemble_rows = frame[frame["step"] == "ensemble"]
    assert len(ensemble_rows) == 1
    assert ensemble_rows.iloc[0]["run_id"] == built.run_id


def test_the_row_attests_no_holdout_read(built, populated_ledger: Path) -> None:
    frame = results_log.read_log(populated_ledger)
    row = frame[frame["step"] == "ensemble"].iloc[0]
    assert not row["holdout_read"]


def test_no_receipt_is_written_when_the_holdout_is_untouched(built, frozen_dir: Path) -> None:
    from titanic_agent import freeze

    manifest = freeze.load_manifest(frozen_dir)
    assert receipts.receipts_for(manifest.dataset_sha256) == []


def test_the_result_reports_no_holdout_score(built) -> None:
    assert built.holdout_score is None


def test_a_submission_is_written(built) -> None:
    assert built.submission_path.is_file()


def test_the_report_names_both_scores(built) -> None:
    text = built.report_path.read_text(encoding="utf-8")
    assert f"{built.cv_score:.4f}" in text
    assert f"{built.best_single:.4f}" in text
    assert "not read" in text


def test_the_report_is_honest_when_the_ensemble_loses(built) -> None:
    text = built.report_path.read_text(encoding="utf-8")
    if built.cv_score < built.best_single:
        assert "not a failure to retry" in text
    else:
        assert "matched or beat" in text


# --------------------------------------------------------------------------- #
# Rule 2: the holdout protocol
# --------------------------------------------------------------------------- #


def test_reading_the_holdout_writes_exactly_one_receipt(
    frozen_dir: Path, populated_ledger: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(paths, "HOLDOUT_READS", tmp_path / "holdout_reads.json")
    from titanic_agent import freeze

    manifest = freeze.load_manifest(frozen_dir)
    reports = tmp_path / "reports"
    result = ensemble.ensemble(
        members=ensemble.select_members(log_path=populated_ledger, top_k=2),
        method="hard_vote",
        cv_folds=2,
        frozen_dir=frozen_dir,
        log_path=populated_ledger,
        score_holdout=True,
        submission_path=reports / "submission.csv",
        report_path=reports / "03_ensemble.md",
    )
    assert result.holdout_score is not None
    assert len(receipts.receipts_for(manifest.dataset_sha256)) == 1


def test_a_second_holdout_read_is_refused(
    frozen_dir: Path, populated_ledger: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(paths, "HOLDOUT_READS", tmp_path / "holdout_reads.json")
    reports = tmp_path / "reports"
    kwargs = {
        "members": ensemble.select_members(log_path=populated_ledger, top_k=2),
        "method": "hard_vote",
        "cv_folds": 2,
        "frozen_dir": frozen_dir,
        "log_path": populated_ledger,
        "score_holdout": True,
        "submission_path": reports / "submission.csv",
        "report_path": reports / "03_ensemble.md",
    }
    ensemble.ensemble(**kwargs)
    with raises_real_runtime_error():
        ensemble.ensemble(**kwargs)


def test_the_check_happens_before_the_read() -> None:
    # Ordering is the whole guarantee: checking after reading would refuse the
    # second measurement only once the second measurement had been taken.
    source = Path(ensemble.__file__).read_text(encoding="utf-8")
    body = source[source.index("if score_holdout:") :]
    assert body.index("require_unread") < body.index("read_csv")
    assert body.index("read_csv") < body.index("record_holdout_read")


def test_the_attestation_matches_what_happened(
    frozen_dir: Path, populated_ledger: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(paths, "HOLDOUT_READS", tmp_path / "holdout_reads.json")
    reports = tmp_path / "reports"
    ensemble.ensemble(
        members=ensemble.select_members(log_path=populated_ledger, top_k=2),
        method="hard_vote",
        cv_folds=2,
        frozen_dir=frozen_dir,
        log_path=populated_ledger,
        score_holdout=True,
        submission_path=reports / "submission.csv",
        report_path=reports / "03_ensemble.md",
    )
    frame = results_log.read_log(populated_ledger)
    row = frame[frame["step"] == "ensemble"].iloc[0]
    assert bool(row["holdout_read"]) is True


# --------------------------------------------------------------------------- #
# refusals and submission shape
# --------------------------------------------------------------------------- #


def test_ensemble_rejects_an_unknown_method(frozen_dir: Path) -> None:
    with pytest.raises(ValueError, match="unknown method"):
        ensemble.ensemble(
            members=[{"model": "logistic"}, {"model": "dummy"}],
            method="telepathy",
            frozen_dir=frozen_dir,
        )


def test_ensemble_needs_at_least_two_members(frozen_dir: Path, log_path: Path) -> None:
    with pytest.raises(ValueError, match="at least two members"):
        ensemble.ensemble(members=[], frozen_dir=frozen_dir, log_path=log_path)


def test_ensemble_refuses_an_unfrozen_dataset(tmp_path: Path, log_path: Path) -> None:
    with raises_real_runtime_error():
        ensemble.ensemble(
            members=[{"model": "logistic", "cv_score": 0.8}, {"model": "knn", "cv_score": 0.7}],
            frozen_dir=tmp_path,
            log_path=log_path,
        )


def test_write_submission_rejects_bad_labels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        ensemble.write_submission(
            [0, 1, 2],
            passenger_ids=[892, 893, 894],
            submission_path=tmp_path / "submission.csv",
            expected_rows=3,
        )


def test_write_submission_rejects_a_wrong_row_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rows"):
        ensemble.write_submission(
            [0, 1],
            passenger_ids=[892, 893],
            submission_path=tmp_path / "submission.csv",
            expected_rows=418,
        )


def test_write_submission_writes_the_kaggle_shape(tmp_path: Path) -> None:
    import pandas as pd

    path = ensemble.write_submission(
        [0, 1, 1],
        passenger_ids=[892, 893, 894],
        submission_path=tmp_path / "submission.csv",
        expected_rows=3,
    )
    frame = pd.read_csv(path)
    assert list(frame.columns) == ["PassengerId", "Survived"]
    assert frame["Survived"].isin([0, 1]).all()
