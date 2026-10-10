"""Smoke test: the experiment container and the findings that chain between them.

Two properties carry the design:

- :func:`test_each_experiment_draws_a_different_val_split` — reusing one seed
  across experiments would recreate, inside the working set, exactly the
  adaptive-overfitting problem the locked final test exists to prevent.
- :func:`test_a_confirmed_finding_needs_evidence` — a conclusion handed forward
  without the runs behind it is folklore the next experiment cannot check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from titanic_agent import experiments, findings, paths


@pytest.fixture
def sandbox(frozen_dir: Path, tmp_path: Path, monkeypatch) -> Path:
    """An experiments root with a real frozen dataset behind it."""
    root = tmp_path / "experiments"
    monkeypatch.setattr(paths, "EXPERIMENTS_DIR", root)
    monkeypatch.setattr(paths, "FROZEN_DIR", frozen_dir)
    monkeypatch.setattr(paths, "FINDINGS", root / "findings.jsonl")
    monkeypatch.setattr(paths, "EXPERIMENTS_INDEX", root / "EXPERIMENTS.md")
    monkeypatch.delenv(experiments.EXPERIMENT_ID_ENV, raising=False)
    return root


# --------------------------------------------------------------------------- #
# the split
# --------------------------------------------------------------------------- #


def test_opening_draws_a_val_split(sandbox: Path, frozen_dir: Path) -> None:
    exp = experiments.open_experiment("first", frozen_dir=frozen_dir)
    assert exp.train_ids and exp.val_ids
    assert not set(exp.train_ids) & set(exp.val_ids)


def test_the_split_covers_the_working_set(sandbox: Path, frozen_dir: Path) -> None:
    import pandas as pd

    exp = experiments.open_experiment("first", frozen_dir=frozen_dir)
    working = {int(i) for i in pd.read_csv(frozen_dir / "train.csv")["PassengerId"]}
    assert set(exp.train_ids) | set(exp.val_ids) == working


def test_the_split_never_touches_the_final_test(sandbox: Path, frozen_dir: Path) -> None:
    # The whole point: an experiment re-splits the working set, and the final
    # test is not part of it.
    import pandas as pd

    exp = experiments.open_experiment("first", frozen_dir=frozen_dir)
    final_test = {int(i) for i in pd.read_csv(frozen_dir / "holdout.csv")["PassengerId"]}
    assert not (set(exp.train_ids) | set(exp.val_ids)) & final_test


def test_each_experiment_draws_a_different_val_split(sandbox: Path, frozen_dir: Path) -> None:
    # Reusing a seed would mean every experiment validates on the same rows,
    # recreating inside the working set the exact problem the locked final test
    # exists to prevent.
    first = experiments.open_experiment("first", frozen_dir=frozen_dir)
    experiments.close_experiment(first.experiment_id)
    second = experiments.open_experiment("second", frozen_dir=frozen_dir)
    assert first.seed != second.seed
    assert first.val_ids != second.val_ids


def test_an_explicit_seed_is_reproducible(sandbox: Path, frozen_dir: Path) -> None:
    a = experiments.draw_val_split(seed=7, frozen_dir=frozen_dir)
    b = experiments.draw_val_split(seed=7, frozen_dir=frozen_dir)
    assert a == b


def test_load_frames_slices_without_writing_files(sandbox: Path, frozen_dir: Path) -> None:
    # No second copy of frozen data to drift from the checksummed original.
    exp = experiments.open_experiment("first", frozen_dir=frozen_dir)
    before = sorted(p.name for p in frozen_dir.iterdir())
    train, val = experiments.load_frames(exp, frozen_dir=frozen_dir)
    assert len(train) == len(exp.train_ids)
    assert len(val) == len(exp.val_ids)
    assert sorted(p.name for p in frozen_dir.iterdir()) == before


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #


def test_ids_are_ordinal_and_slugged(sandbox: Path, frozen_dir: Path) -> None:
    exp = experiments.open_experiment("Wide Search!", frozen_dir=frozen_dir)
    assert exp.experiment_id == "exp-001-wide-search"


def test_only_one_experiment_may_be_open(sandbox: Path, frozen_dir: Path) -> None:
    # Two open experiments make "which val split am I using?" ambiguous.
    experiments.open_experiment("first", frozen_dir=frozen_dir)
    with pytest.raises(RuntimeError, match="still open"):
        experiments.open_experiment("second", frozen_dir=frozen_dir)


def test_closing_frees_the_slot(sandbox: Path, frozen_dir: Path) -> None:
    first = experiments.open_experiment("first", frozen_dir=frozen_dir)
    experiments.close_experiment(first.experiment_id)
    second = experiments.open_experiment("second", frozen_dir=frozen_dir)
    assert second.experiment_id == "exp-002-second"


def test_abandoning_also_frees_the_slot(sandbox: Path, frozen_dir: Path) -> None:
    first = experiments.open_experiment("first", frozen_dir=frozen_dir)
    closed = experiments.close_experiment(first.experiment_id, status="abandoned")
    assert closed.status == "abandoned"
    experiments.open_experiment("second", frozen_dir=frozen_dir)


def test_closing_with_an_unknown_status_raises(sandbox: Path, frozen_dir: Path) -> None:
    exp = experiments.open_experiment("first", frozen_dir=frozen_dir)
    with pytest.raises(ValueError, match="unknown status"):
        experiments.close_experiment(exp.experiment_id, status="finished-ish")


def test_current_experiment_is_the_open_one(sandbox: Path, frozen_dir: Path) -> None:
    exp = experiments.open_experiment("first", frozen_dir=frozen_dir)
    assert experiments.current_experiment().experiment_id == exp.experiment_id


def test_current_experiment_is_none_when_all_are_closed(sandbox: Path, frozen_dir: Path) -> None:
    exp = experiments.open_experiment("first", frozen_dir=frozen_dir)
    experiments.close_experiment(exp.experiment_id)
    assert experiments.current_experiment() is None


def test_the_environment_pins_the_experiment(sandbox: Path, frozen_dir: Path, monkeypatch) -> None:
    exp = experiments.open_experiment("first", frozen_dir=frozen_dir)
    experiments.close_experiment(exp.experiment_id)
    monkeypatch.setenv(experiments.EXPERIMENT_ID_ENV, exp.experiment_id)
    assert experiments.current_experiment().experiment_id == exp.experiment_id


def test_the_manifest_records_the_working_set_fingerprint(sandbox: Path, frozen_dir: Path) -> None:
    # Two experiments are comparable only if this matches.
    from titanic_agent import freeze

    exp = experiments.open_experiment("first", frozen_dir=frozen_dir)
    assert exp.working_set_sha == freeze.load_manifest(frozen_dir).dataset_sha256


def test_caps_are_recorded_on_the_experiment(sandbox: Path, frozen_dir: Path) -> None:
    exp = experiments.open_experiment("first", frozen_dir=frozen_dir, caps={"papers": 3})
    assert exp.caps == {"papers": 3}


def test_enforced_caps_are_only_the_ones_that_are_counted(sandbox: Path, frozen_dir: Path) -> None:
    # A cap nobody counts is a number in a file pretending to be a limit. The
    # first version of `caps` listed three nobody incremented, and a cold-start
    # session correctly ignored the whole block.
    exp = experiments.open_experiment("first", frozen_dir=frozen_dir)
    assert set(exp.caps) == {"wall_clock_s", "loop_passes"}
    assert exp.caps_unenforced["web_fetches"] == 0  # v1: no general web
    assert exp.caps_unenforced["kaggle_notebooks"] == 20


def test_spend_is_counted_and_binds(sandbox: Path, frozen_dir: Path) -> None:
    exp = experiments.open_experiment("counting", frozen_dir=frozen_dir)
    assert experiments.over_budget() == {}
    for _ in range(exp.caps["loop_passes"]):
        experiments.record_spend("loop_passes", 1)
    over = experiments.over_budget()
    assert "loop_passes" in over
    assert over["loop_passes"] == (3.0, 3.0)


def test_unenforced_caps_never_report_as_over_budget(sandbox: Path, frozen_dir: Path) -> None:
    # Reporting a resource nobody counts would be theatre.
    experiments.open_experiment("first", frozen_dir=frozen_dir)
    experiments.record_spend("kaggle_notebooks", 999)
    assert "kaggle_notebooks" not in experiments.over_budget()


def test_recording_spend_with_no_experiment_open_is_harmless(sandbox: Path) -> None:
    assert experiments.record_spend("loop_passes", 1) is None
    assert experiments.over_budget() == {}


def test_loading_a_missing_experiment_raises(sandbox: Path) -> None:
    with pytest.raises(FileNotFoundError):
        experiments.load_experiment("exp-999-nope")


def test_a_malformed_manifest_raises(sandbox: Path) -> None:
    (sandbox / "exp-001-bad").mkdir(parents=True)
    (sandbox / "exp-001-bad" / "experiment.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        experiments.load_experiment("exp-001-bad")


def test_a_malformed_manifest_does_not_break_the_listing(sandbox: Path, frozen_dir: Path) -> None:
    experiments.open_experiment("good", frozen_dir=frozen_dir)
    (sandbox / "exp-999-bad").mkdir(parents=True)
    (sandbox / "exp-999-bad" / "experiment.json").write_text("{not json", encoding="utf-8")
    assert [e.name for e in experiments.list_experiments()] == ["good"]


# --------------------------------------------------------------------------- #
# the index
# --------------------------------------------------------------------------- #


def test_the_index_is_derived_from_the_manifests(sandbox: Path, frozen_dir: Path) -> None:
    exp = experiments.open_experiment("first", frozen_dir=frozen_dir)
    text = experiments.write_index().read_text(encoding="utf-8")
    assert exp.experiment_id in text
    assert "read once for the whole" in text  # the final-test caveat


def test_the_index_handles_no_experiments(sandbox: Path) -> None:
    assert "_none yet_" in experiments.write_index().read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# findings — what actually chains
# --------------------------------------------------------------------------- #


def test_a_confirmed_finding_needs_evidence() -> None:
    # A conclusion handed forward without the runs behind it is folklore.
    with pytest.raises(ValueError, match="needs evidence"):
        findings.Finding(claim="trees are best", status="confirmed")


def test_an_untried_finding_needs_no_evidence() -> None:
    findings.Finding(claim="try target encoding", status="untried")


def test_an_unknown_status_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown status"):
        findings.Finding(claim="x", status="probably")


def test_findings_round_trip(sandbox: Path) -> None:
    findings.record(findings.Finding(claim="A helps", status="untried", experiment_id="exp-001-x"))
    found = findings.read_findings()
    assert len(found) == 1
    assert found[0].claim == "A helps"
    assert found[0].recorded_utc.endswith("+00:00")


def test_findings_are_append_only(sandbox: Path) -> None:
    for claim in ("first", "second"):
        findings.record(findings.Finding(claim=claim, status="untried"))
    assert [f.claim for f in findings.read_findings()] == ["first", "second"]


def test_the_backlog_is_the_untried_ones(sandbox: Path) -> None:
    findings.record(findings.Finding(claim="untested idea", status="untried"))
    findings.record(findings.Finding(claim="known result", status="confirmed", evidence=["run-1"]))
    assert [f.claim for f in findings.backlog()] == ["untested idea"]


def test_a_truncated_findings_line_is_skipped(sandbox: Path) -> None:
    findings.record(findings.Finding(claim="intact", status="untried"))
    with paths.FINDINGS.open("a", encoding="utf-8") as handle:
        handle.write('{"claim": "trunc')
    assert [f.claim for f in findings.read_findings()] == ["intact"]


def test_findings_render_grouped_by_status(sandbox: Path) -> None:
    findings.record(
        findings.Finding(claim="measured", status="confirmed", evidence=["run-1"], delta=0.004)
    )
    findings.record(findings.Finding(claim="idea", status="untried"))
    text = findings.render_markdown()
    assert "## Confirmed (1)" in text
    assert "## Untried (1)" in text
    assert "+0.0040" in text
    assert "run-1" in text


def test_inherited_findings_exclude_the_current_experiment(sandbox: Path, frozen_dir: Path) -> None:
    first = experiments.open_experiment("first", frozen_dir=frozen_dir)
    findings.record(
        findings.Finding(claim="from one", status="untried", experiment_id=first.experiment_id)
    )
    experiments.close_experiment(first.experiment_id)

    second = experiments.open_experiment("second", frozen_dir=frozen_dir)
    findings.record(
        findings.Finding(claim="from two", status="untried", experiment_id=second.experiment_id)
    )
    inherited = experiments.inherited_findings(second.experiment_id)
    assert [f.claim for f in inherited] == ["from one"]


def test_closing_renders_the_findings_file(sandbox: Path, frozen_dir: Path) -> None:
    exp = experiments.open_experiment("first", frozen_dir=frozen_dir)
    findings.record(
        findings.Finding(claim="something", status="untried", experiment_id=exp.experiment_id)
    )
    closed = experiments.close_experiment(exp.experiment_id, summary="done")
    text = (closed.directory / "findings.md").read_text(encoding="utf-8")
    assert "something" in text
    assert closed.summary == "done"


def test_the_manifest_is_valid_json(sandbox: Path, frozen_dir: Path) -> None:
    exp = experiments.open_experiment("first", frozen_dir=frozen_dir)
    payload = json.loads((exp.directory / "experiment.json").read_text(encoding="utf-8"))
    assert payload["experiment_id"] == exp.experiment_id
    assert payload["val_ids"]


# --------------------------------------------------------------------------- #
# the val read, and the leak that made exp-002 worthless
# --------------------------------------------------------------------------- #


def test_tune_trains_on_the_experiment_rows_when_one_is_open(
    sandbox: Path, frozen_dir: Path
) -> None:
    from titanic_agent import tune

    full = len(tune.load_training_frame(frozen_dir))
    exp = experiments.open_experiment("scoped", frozen_dir=frozen_dir)
    assert len(tune.load_training_frame(frozen_dir)) == len(exp.train_ids) < full


def test_ensemble_members_never_cross_experiments(
    sandbox: Path, frozen_dir: Path, log_path: Path
) -> None:
    # The leak that made exp-002's validation score a selection statistic: an
    # experiment trains on its own 569 rows, but could ensemble members tuned by
    # an earlier experiment on the full working set -- which contains this
    # experiment's validation rows.
    from titanic_agent import ensemble, results_log

    for model in ("logistic", "knn"):
        results_log.append_run(
            step="tune",
            model=model,
            cv_score=0.99,
            experiment_id="exp-000-someone-else",
            path=log_path,
        )
    exp = experiments.open_experiment("mine", frozen_dir=frozen_dir)
    results_log.append_run(
        step="tune",
        model="logistic",
        cv_score=0.80,
        experiment_id=exp.experiment_id,
        path=log_path,
    )
    members = ensemble.select_members(log_path=log_path, top_k=9)
    assert [m["model"] for m in members] == ["logistic"]
    assert members[0]["cv_score"] == 0.80  # ours, not the other experiment's 0.99


def test_a_second_val_read_in_one_experiment_is_refused(
    sandbox: Path, frozen_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    # Val splits are cheap across experiments and expensive within one: the same
    # 142 rows, read twice with a model chosen in between, is selection.
    monkeypatch.setattr(paths, "VAL_READS", tmp_path / "val_reads.jsonl")
    exp = experiments.open_experiment("once", frozen_dir=frozen_dir)
    experiments.record_val_read(
        experiment_id=exp.experiment_id, score=0.84, model="rf", run_id="run-1"
    )
    with pytest.raises(RuntimeError, match="already read its validation"):
        experiments.require_val_unread(exp.experiment_id)


def test_val_reads_are_scoped_per_experiment(
    sandbox: Path, frozen_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(paths, "VAL_READS", tmp_path / "val_reads.jsonl")
    experiments.record_val_read(experiment_id="exp-001-a", score=0.8, model="rf", run_id="r1")
    experiments.require_val_unread("exp-002-b")  # a different experiment is unaffected
    assert len(experiments.val_reads_for("exp-001-a")) == 1
    assert experiments.val_reads_for("exp-002-b") == []


def test_evaluate_refuses_with_nothing_scored(
    sandbox: Path, frozen_dir: Path, log_path: Path, tmp_path: Path, monkeypatch
) -> None:
    from titanic_agent import results_log

    monkeypatch.setattr(paths, "VAL_READS", tmp_path / "val_reads.jsonl")
    results_log.ensure_log(log_path)
    experiments.open_experiment("empty", frozen_dir=frozen_dir)
    with pytest.raises(RuntimeError, match="no scored runs"):
        experiments.evaluate(frozen_dir=frozen_dir, log_path=log_path)


def test_evaluate_spends_exactly_one_val_read(
    sandbox: Path, frozen_dir: Path, log_path: Path, tmp_path: Path, monkeypatch
) -> None:
    from titanic_agent import tune

    monkeypatch.setattr(paths, "VAL_READS", tmp_path / "val_reads.jsonl")
    exp = experiments.open_experiment("real", frozen_dir=frozen_dir)
    tune.tune("dummy", frozen_dir=frozen_dir, log_path=log_path, cv_folds=2)

    evaluated = experiments.evaluate(frozen_dir=frozen_dir, log_path=log_path)
    assert evaluated.val_score is not None
    assert len(experiments.val_reads_for(exp.experiment_id)) == 1

    with pytest.raises(RuntimeError, match="already read its validation"):
        experiments.evaluate(frozen_dir=frozen_dir, log_path=log_path)


def test_evaluate_leaves_the_final_test_untouched(
    sandbox: Path, frozen_dir: Path, log_path: Path, tmp_path: Path, monkeypatch
) -> None:
    from titanic_agent import receipts, tune

    monkeypatch.setattr(paths, "VAL_READS", tmp_path / "val_reads.jsonl")
    monkeypatch.setattr(paths, "HOLDOUT_READS", tmp_path / "holdout_reads.json")
    experiments.open_experiment("real", frozen_dir=frozen_dir)
    tune.tune("dummy", frozen_dir=frozen_dir, log_path=log_path, cv_folds=2)
    experiments.evaluate(frozen_dir=frozen_dir, log_path=log_path)
    assert receipts.read_receipts() == []


def test_the_evaluate_row_attests_no_holdout_read(
    sandbox: Path, frozen_dir: Path, log_path: Path, tmp_path: Path, monkeypatch
) -> None:
    from titanic_agent import results_log, tune

    monkeypatch.setattr(paths, "VAL_READS", tmp_path / "val_reads.jsonl")
    experiments.open_experiment("real", frozen_dir=frozen_dir)
    tune.tune("dummy", frozen_dir=frozen_dir, log_path=log_path, cv_folds=2)
    experiments.evaluate(frozen_dir=frozen_dir, log_path=log_path)
    frame = results_log.read_log(log_path)
    row = frame[frame["step"] == "evaluate"].iloc[0]
    assert not row["holdout_read"]
