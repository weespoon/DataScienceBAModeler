"""Bake-off and champion: in-fold transforms, pairing, and the ledger trail.

The transforms are tested on hand-built frames, because the property that
matters — a row's group feature never contains its own label — is easiest to
see on four passengers. The run itself is exercised on the 12-row fixture with
2 folds and 1 repeat: enough to check the rows it writes, not to measure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from titanic_agent import bakeoff, champion, experiments, leakage, paths, results_log


@pytest.fixture
def sandbox(frozen_dir: Path, tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "experiments"
    monkeypatch.setattr(paths, "EXPERIMENTS_DIR", root)
    monkeypatch.setattr(paths, "FROZEN_DIR", frozen_dir)
    monkeypatch.setattr(paths, "FINDINGS", root / "findings.jsonl")
    monkeypatch.setattr(paths, "EXPERIMENTS_INDEX", root / "EXPERIMENTS.md")
    monkeypatch.setattr(paths, "VAL_READS", tmp_path / "val_reads.jsonl")
    monkeypatch.setattr(paths, "CHAMPION", root / "champion.jsonl")
    monkeypatch.delenv(experiments.EXPERIMENT_ID_ENV, raising=False)
    return root


def _family() -> pd.DataFrame:
    """Four passengers on one ticket and fare, plus a stranger."""
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4, 5],
            "Name": [
                "Ford, Mr. Arthur",
                "Ford, Mrs. Arthur",
                "Ford, Master. William",
                "Ford, Miss. Doris",
                "Smith, Mr. John",
            ],
            "Sex": ["male", "female", "male", "female", "male"],
            "Title": ["Mr", "Mrs", "Master", "Miss", "Mr"],
            "Age": [40.0, 38.0, np.nan, 10.0, 30.0],
            "Ticket": ["W 1"] * 4 + ["X 9"],
            "Fare": [34.375] * 4 + [8.05],
        }
    )


# --------------------------------------------------------------------------- #
# fitted transforms
# --------------------------------------------------------------------------- #


def test_group_rate_on_a_training_row_excludes_its_own_label() -> None:
    frame = _family()
    labels = pd.Series([0, 1, 1, 1, 0])
    out = bakeoff.Engineer(("group",)).fit(frame, labels).transform(frame)
    # Arthur's mates are the other three Fords, who all lived.
    assert out.loc[0, "GroupSurvival"] == 1.0
    # Mrs Ford's mates: Arthur died, the two children lived.
    assert out.loc[1, "GroupSurvival"] == pytest.approx(2 / 3)

    flipped = labels.copy()
    flipped[1] = 0
    again = bakeoff.Engineer(("group",)).fit(frame, flipped).transform(frame)
    # Changing her own label must not move her own feature.
    assert again.loc[1, "GroupSurvival"] == out.loc[1, "GroupSurvival"]


def test_group_rate_for_an_unseen_row_uses_only_fitted_labels() -> None:
    frame = _family()
    fit_rows, new_row = frame.iloc[:3], frame.iloc[[3]]
    engineer = bakeoff.Engineer(("group",)).fit(fit_rows, pd.Series([0, 1, 1]))
    out = engineer.transform(new_row)
    assert out["GroupSurvival"].iloc[0] == pytest.approx(2 / 3)
    # Women and children only: Mrs Ford and the boy both lived.
    assert out["WCGSurvival"].iloc[0] == 1.0
    assert out["WCGKnown"].iloc[0] == 1


def test_a_passenger_with_no_labelled_mates_gets_the_unknown_rate() -> None:
    frame = _family()
    out = bakeoff.Engineer(("group",)).fit(frame, pd.Series([0, 1, 1, 1, 0])).transform(frame)
    assert out.loc[4, "GroupSurvival"] == bakeoff.UNKNOWN_RATE
    assert out.loc[4, "GroupKnown"] == 0


def test_group_needs_labels() -> None:
    with pytest.raises(ValueError, match="labels"):
        bakeoff.Engineer(("group",)).fit(_family())


def test_ticket_size_counts_the_batch_without_double_counting() -> None:
    frame = _family()
    engineer = bakeoff.Engineer(("ticket_size",)).fit(frame.iloc[:2])
    # Re-transforming fitted rows does not count them twice.
    assert list(engineer.transform(frame.iloc[:2])["TicketGroupSize"]) == [2, 2]
    # New companions in the batch do count: no label is involved.
    assert list(engineer.transform(frame.iloc[2:4])["TicketGroupSize"]) == [4, 4]


def test_age_by_title_uses_medians_from_fitted_rows_only() -> None:
    frame = _family()
    fit_rows = frame.assign(Title=["Mr", "Mrs", "Master", "Master", "Mr"]).iloc[[0, 3]]
    engineer = bakeoff.Engineer(("age_by_title",)).fit(fit_rows)
    out = engineer.transform(frame.iloc[[2]])
    assert out["Age"].iloc[0] == 10.0


def test_wcg_rule_corrects_boys_and_women_by_their_group() -> None:
    frame = pd.DataFrame(
        {
            "Sex": ["male", "female", "female", "male"],
            "Title": ["Master", "Mrs", "Miss", "Mr"],
            "WCGSurvival": [1.0, 0.0, 0.5, 1.0],
            "WCGKnown": [1, 1, 0, 1],
        }
    )
    rule = bakeoff.RuleClassifier("wcg").fit(frame, [0, 0, 0, 0])
    # boy whose group lived -> lives; woman whose group died -> dies;
    # woman with no known group -> gender rule; adult man -> dies regardless.
    assert list(rule.predict(frame)) == [1, 0, 1, 0]


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #


def test_every_arm_states_a_hypothesis_and_builds() -> None:
    registry = bakeoff.arms_for(champion.INITIAL)
    assert next(iter(registry)) == "champion"
    for name, arm in registry.items():
        assert len(arm.hypothesis) > 20, name
        bakeoff.build(arm.config(registry), seed=0)


def test_ensemble_configs_inline_their_members() -> None:
    registry = bakeoff.arms_for(champion.INITIAL)
    config = registry["vote_group"].config(registry)
    assert [m["name"] for m in config["members"]] == list(registry["vote_group"].members)
    # Rebuildable with no registry: what a later experiment has to do.
    bakeoff.build(config, seed=0)


def test_fitted_outputs_are_vouched_for_by_leakage(frozen_dir: Path) -> None:
    allowed = leakage.derivable_columns(frozen_dir)
    for outputs in bakeoff.FITTED_OUTPUTS.values():
        assert set(outputs) <= allowed


def test_best_of_n_null_grows_with_n() -> None:
    assert bakeoff.best_of_n_null(1) == pytest.approx(0.0, abs=0.02)
    assert 1.7 < bakeoff.best_of_n_null(20) < 2.0


# --------------------------------------------------------------------------- #
# champion
# --------------------------------------------------------------------------- #


def test_ensure_crowns_the_initial_champion_once(sandbox: Path) -> None:
    first = champion.ensure()
    second = champion.ensure()
    assert first.name == second.name == "rf_baseline12"
    assert len(champion.history()) == 1


def test_a_champion_needs_evidence(sandbox: Path) -> None:
    with pytest.raises(ValueError, match="evidence"):
        champion.crown(champion.INITIAL, reason="trust me", evidence=[])


def test_crowning_appends_and_keeps_the_succession(sandbox: Path) -> None:
    champion.ensure()
    champion.crown({"name": "next"}, reason="won", evidence=["run-1"], delta_vs_previous=0.01)
    assert [c.name for c in champion.history()] == ["rf_baseline12", "next"]
    assert champion.current().name == "next"


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


def _small_run(log_path: Path, **kwargs):
    return bakeoff.run(
        frozen_dir=paths.FROZEN_DIR,
        only=["rule_gender", "logit_rich"],
        folds=2,
        repeats=1,
        confirm_seeds=1,
        log_path=log_path,
        **kwargs,
    )


def test_run_needs_an_open_experiment(sandbox: Path, log_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no experiment is open"):
        _small_run(log_path)


def test_run_logs_every_arm_paired_against_the_champion(
    sandbox: Path, frozen_dir: Path, log_path: Path
) -> None:
    experiments.open_experiment("bakeoff", frozen_dir=frozen_dir)
    bake = _small_run(log_path)

    frame = results_log.read_log(log_path)
    arms = frame[frame["params"].map(lambda p: "delta_vs_champion" in p)]
    assert list(arms["params"].map(lambda p: p["arm"])) == ["champion", "rule_gender", "logit_rich"]
    assert arms["params"].iloc[0]["delta_vs_champion"] == 0
    assert not frame["holdout_read"].any()
    assert bake.winner in ("rule_gender", "logit_rich")

    # The confirmation row is a comparison, so it carries no score of its own.
    confirm = frame[frame["params"].map(lambda p: "confirm" in p)]
    assert len(confirm) == 1
    assert pd.isna(confirm["cv_score"].iloc[0])
    assert confirm["params"].iloc[0]["selected_from"] == 2


def test_decide_requires_all_three_checks(sandbox: Path, frozen_dir: Path, log_path: Path) -> None:
    experiments.open_experiment("bakeoff", frozen_dir=frozen_dir)
    bake = _small_run(log_path)
    missing_nested = bakeoff.decide(bake, None)
    assert not missing_nested.promote
    assert any(
        "nested" in reason and reason.startswith("FAIL") for reason in missing_nested.reasons
    )


def test_evaluate_rebuilds_a_bakeoff_row(sandbox: Path, frozen_dir: Path, log_path: Path) -> None:
    exp = experiments.open_experiment("bakeoff", frozen_dir=frozen_dir)
    bake = _small_run(log_path)
    declared = next(r for r in bake.results if r.arm.name == "logit_rich").run_id
    updated = experiments.evaluate(
        exp.experiment_id, run_id=declared, log_path=log_path, frozen_dir=frozen_dir
    )
    assert updated.val_score is not None
    assert 0.0 <= updated.val_score <= 1.0


# --------------------------------------------------------------------------- #
# decomposition, replication, submission (the exp-007 follow-ups)
# --------------------------------------------------------------------------- #


def _kitchen() -> dict:
    registry = bakeoff.arms_for(champion.INITIAL)
    return registry["rf_kitchen"].config(registry)


def test_decompose_undoes_each_change_once() -> None:
    arms = bakeoff.decompose_arms(_kitchen(), champion.INITIAL)
    assert list(arms)[:4] == ["champion", "previous_champion", "rf_wcg", "rf_wcg_maxfeat"]
    without = {name.removeprefix("without_") for name in arms if name.startswith("without_")}
    assert without == set(bakeoff.COMPONENTS)

    wcg_gone = arms["without_wcg_rate"]
    assert not set(bakeoff.WCG) & set(wcg_gone.features)
    assert "group" in wcg_gone.engineer  # the all-member rate still needs it
    assert "clf__max_features" not in arms["without_max_features"].params
    assert "Title" in arms["without_titlefine_swap"].features


def test_decompose_skips_changes_the_champion_does_not_carry() -> None:
    arms = bakeoff.decompose_arms(champion.INITIAL)
    assert list(arms) == ["champion"]


def test_dropping_the_last_group_column_drops_the_group_step() -> None:
    config = {"features": ["Sex", "WCGSurvival", "WCGKnown"], "engineer": ["group"]}
    assert bakeoff._without(config, bakeoff.WCG)["engineer"] == []


def test_replication_pairs_configs_across_experiments(log_path: Path) -> None:
    ref = {"name": "champion", "model": "m", "params": {}, "features": ["Sex"]}
    arm = {"name": "x", "model": "m", "params": {"p": 1}, "features": ["Sex"]}
    new_ref = {"name": "new", "model": "n", "params": {}, "features": ["Sex"]}

    def log(experiment_id, name, config, delta):
        results_log.append_run(
            step="bakeoff",
            model="m",
            params={"arm": name, "config": config, "delta_vs_champion": delta},
            cv_score=0.8,
            experiment_id=experiment_id,
            path=log_path,
        )

    log("exp-a", "champion", ref, 0.0)
    log("exp-a", "x", arm, 0.02)
    # exp-b has a new champion; the old reference is re-run under another name.
    log("exp-b", "champion", new_ref, 0.0)
    log("exp-b", "previous_champion", {**ref, "name": "rf_old"}, -0.01)
    log("exp-b", "x", arm, 0.0)

    [row] = bakeoff.replication("exp-b", log_path=log_path)
    assert row["arm"] == "x" and row["reference"] == "previous_champion"
    assert row["delta_there"] == 0.02
    assert row["delta_here"] == pytest.approx(0.01)
    assert row["change"] == pytest.approx(-0.01)


def test_submit_predicts_the_test_set_without_reading_the_holdout(
    sandbox: Path, frozen_dir: Path, log_path: Path, tmp_path: Path, monkeypatch
) -> None:
    champion.ensure()
    real_read = pd.read_csv

    def guarded(path, *args, **kwargs):
        assert "holdout" not in str(path), "submit read the holdout"
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded)
    written = champion.submit(
        frozen_dir=frozen_dir, log_path=log_path, submission_path=tmp_path / "s.csv"
    )
    submission = real_read(written)
    assert len(submission) == len(real_read(frozen_dir / "test.csv"))
    assert set(submission["Survived"]) <= {0, 1}
    row = results_log.read_log(log_path).iloc[-1]
    assert row["step"] == "submit" and not row["holdout_read"]


def test_close_does_not_report_a_val_score_as_best_cv(
    sandbox: Path, frozen_dir: Path, log_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(paths, "RESULTS_LOG", log_path)
    exp = experiments.open_experiment("bakeoff", frozen_dir=frozen_dir)
    results_log.append_run(step="bakeoff", model="m", cv_score=0.80, path=log_path)
    results_log.append_run(step="evaluate", model="m", cv_score=0.95, path=log_path)
    closed = experiments.close_experiment(exp.experiment_id)
    assert closed.best_cv == pytest.approx(0.80)
