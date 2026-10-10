"""Smoke test: feature selection against one fixed model.

Runs against the 12-row fixture with 2 folds and 1 repeat — enough to exercise
the add/prune loop and its ledger rows, not to measure anything. Ledger and
experiment files live in ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from titanic_agent import experiments, features, paths, results_log, tune


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


def _small_select(frozen_dir: Path, log_path: Path, **kwargs):
    return features.select(
        baseline=["Sex", "Pclass", "Age"],
        candidates=["IsChild", "FareLog"],
        folds=2,
        repeats=1,
        tune_folds=2,
        frozen_dir=frozen_dir,
        log_path=log_path,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# the candidates
# --------------------------------------------------------------------------- #


def test_every_candidate_is_built_by_add_derived(frozen_dir: Path) -> None:
    frame = tune.add_derived(pd.read_csv(frozen_dir / "train.csv"))
    missing = [name for name in features.CANDIDATES if name not in frame.columns]
    assert not missing


def test_every_candidate_states_why_it_should_generalise() -> None:
    # Rule 4: no reason, no run.
    for name, why in features.CANDIDATES.items():
        assert len(why) > 20, name


def test_candidates_survive_a_frame_without_name_ticket_or_cabin(frozen_dir: Path) -> None:
    frame = pd.read_csv(frozen_dir / "train.csv").drop(columns=["Name", "Ticket", "Cabin"])
    derived = tune.add_derived(frame)
    assert set(features.CANDIDATES) <= set(derived.columns)


def test_candidates_are_row_wise(frozen_dir: Path) -> None:
    # A candidate computed over the whole frame must equal the same candidate
    # computed one row at a time; otherwise it counts across rows and leaks
    # between CV folds.
    raw = pd.read_csv(frozen_dir / "train.csv")
    whole = tune.add_derived(raw)[list(features.CANDIDATES)].reset_index(drop=True)
    single = pd.concat(
        [tune.add_derived(raw.iloc[[i]])[list(features.CANDIDATES)] for i in range(len(raw))]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(whole.astype(str), single.astype(str))


def test_age_buckets_have_an_unknown_level() -> None:
    frame = pd.DataFrame({"Age": [5.0, None, 70.0]})
    bins = pd.cut(frame["Age"], features.AGE_EDGES, labels=features.AGE_LABELS, right=False)
    assert list(bins.astype("string").fillna("Unknown")) == ["child", "Unknown", "senior"]


def test_candidates_are_not_in_the_fallback_feature_list(frozen_dir: Path, monkeypatch) -> None:
    # Candidates enter a model only by passing a test, never by default.
    from titanic_agent import explore

    monkeypatch.setattr(explore, "load_candidates", lambda *a, **k: [])
    frame = tune.add_derived(pd.read_csv(frozen_dir / "train.csv"))
    assert not set(tune.resolve_features(frame)) & set(features.CANDIDATES)


# --------------------------------------------------------------------------- #
# the selection loop
# --------------------------------------------------------------------------- #


def test_select_needs_an_open_experiment(sandbox: Path, frozen_dir: Path, log_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no experiment is open"):
        _small_select(frozen_dir, log_path)


def test_select_logs_one_row_per_test(sandbox: Path, frozen_dir: Path, log_path: Path) -> None:
    experiments.open_experiment("features", frozen_dir=frozen_dir)
    selection = _small_select(frozen_dir, log_path)

    frame = results_log.read_log(log_path)
    rows = frame[frame["step"] == "features"]
    # one baseline row plus one per test, and one tune row for the re-tune
    assert len(rows) == len(selection.tests) + 1
    assert (frame["step"] == "tune").sum() == 1
    assert not rows["holdout_read"].any()
    # two adds, then a remove test for every feature that survived them
    assert [t.action for t in selection.tests][:2] == ["add", "add"]
    assert len(selection.tests) == 2 + len([t for t in selection.tests if t.action == "remove"])


def test_select_respects_the_test_cap(sandbox: Path, frozen_dir: Path, log_path: Path) -> None:
    experiments.open_experiment("features", frozen_dir=frozen_dir)
    selection = _small_select(frozen_dir, log_path, max_tests=1)
    assert len(selection.tests) == 1
    assert "cap" in selection.stopped


def test_the_rule_is_symmetric() -> None:
    # A feature's value is (with) - (without) whichever arm it was tested from.
    add = features.FeatureTest(1, "add", "X", 0.80, 0.81, 0.01, 0.0, 0.01, True)
    remove = features.FeatureTest(2, "remove", "X", 0.81, 0.80, -0.01, 0.0, 0.01, True)
    assert add.feature_value == remove.feature_value


def test_evaluate_scores_the_pre_declared_row(
    sandbox: Path, frozen_dir: Path, log_path: Path
) -> None:
    exp = experiments.open_experiment("features", frozen_dir=frozen_dir)
    selection = _small_select(frozen_dir, log_path)
    experiments.evaluate(run_id=selection.final_run_id, frozen_dir=frozen_dir, log_path=log_path)

    frame = results_log.read_log(log_path)
    row = frame[frame["step"] == "evaluate"].iloc[0]
    assert row["params"]["features"] == selection.features
    assert selection.final_run_id in row["notes"]
    assert len(experiments.val_reads_for(exp.experiment_id)) == 1


def test_evaluate_refuses_an_unknown_run_without_spending_the_read(
    sandbox: Path, frozen_dir: Path, log_path: Path
) -> None:
    exp = experiments.open_experiment("features", frozen_dir=frozen_dir)
    tune.tune("dummy", frozen_dir=frozen_dir, log_path=log_path, cv_folds=2)
    with pytest.raises(RuntimeError, match="not a scored run"):
        experiments.evaluate(run_id="nope", frozen_dir=frozen_dir, log_path=log_path)
    assert experiments.val_reads_for(exp.experiment_id) == []
