"""``make report`` — one page per pass.

What is worth testing is not the prose but the refusal to flatter: a failed
audit check has to read as failed, an unstated verdict as unstated, a spent
holdout as spent. A report that smoothed any of those over would be the pass
marking its own homework, and every other check in this repo would stay green.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from titanic_agent import experiments, paths, report, results_log, run_log, runs


@pytest.fixture
def sandbox(frozen_dir: Path, tmp_path: Path, monkeypatch, log_path: Path) -> Path:
    root = tmp_path / "experiments"
    monkeypatch.setattr(paths, "EXPERIMENTS_DIR", root)
    monkeypatch.setattr(paths, "FROZEN_DIR", frozen_dir)
    monkeypatch.setattr(paths, "FINDINGS", root / "findings.jsonl")
    monkeypatch.setattr(paths, "EXPERIMENTS_INDEX", root / "EXPERIMENTS.md")
    monkeypatch.setattr(paths, "VAL_READS", tmp_path / "val_reads.jsonl")
    monkeypatch.setattr(paths, "CHAMPION", root / "champion.jsonl")
    monkeypatch.setattr(paths, "RESULTS_LOG", log_path)
    monkeypatch.setattr(paths, "RUN_LOG", tmp_path / "run_log.jsonl")
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(paths, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(paths, "HOLDOUT_READS", tmp_path / "holdout_reads.json")
    monkeypatch.delenv(experiments.EXPERIMENT_ID_ENV, raising=False)
    monkeypatch.setenv(runs.RUN_ID_ENV, "20261012-1200EDT")
    return tmp_path


def test_a_pass_with_no_experiment_says_so(sandbox: Path) -> None:
    # A pass is one experiment. Printing nothing here would let a pass that
    # never opened one read as a pass that did.
    text = report.render()
    assert "**None.**" in text
    assert "A pass is one experiment" in text


def test_it_names_the_experiment_the_pass_opened_and_its_verdict(sandbox: Path) -> None:
    experiments.open_experiment("report test", frozen_dir=paths.FROZEN_DIR)
    experiment_id = experiments.current_experiment().experiment_id
    results_log.append_run(
        step="tune",
        model="random_forest",
        params={},
        cv_score=0.81,
        pass_id="20261012-1200EDT",
        experiment_id=experiment_id,
        path=paths.RESULTS_LOG,
    )
    experiments.close_experiment(experiment_id, summary="champion holds; nothing cleared the bar")

    text = report.render("20261012-1200EDT")
    assert experiment_id in text
    assert "**held**" in text


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        ("crowned rf_wcg after all three checks", "crowned"),
        ("champion holds", "held"),
        ("Champion HELD under the declared rule; the simpler config ties it", "held"),
        ("inconclusive: the instrument could not resolve it", "inconclusive"),
        ("ABANDONED: its val score was contaminated", "abandoned"),
        ("we did some things and some numbers happened", "verdict not stated"),
    ],
)
def test_the_verdict_is_read_from_the_summary_or_admitted_missing(
    summary: str, expected: str
) -> None:
    class Stub:
        status = "closed"

    stub = Stub()
    stub.summary = summary
    assert expected in report._verdict(stub)


@pytest.mark.parametrize("word", report.PRESCRIBED)
def test_every_word_the_run_skill_prescribes_is_readable(word: str) -> None:
    """The matcher and the vocabulary have to be checked against each other.

    VERDICTS matched the stem "hold", which catches "holds" and misses "held" --
    the word the skill actually asks for. exp-009's report therefore said
    "verdict not stated" above a summary beginning "Champion HELD". One file
    named the vocabulary and another could not read it.
    """
    skill = (paths.ROOT / ".claude" / "skills" / "run" / "SKILL.md").read_text()
    assert f"**{word}**" in skill, f"the skill no longer prescribes {word!r}"

    class Stub:
        status = "closed"

    stub = Stub()
    stub.summary = f"Champion {word.upper()} after the declared checks"
    assert report._verdict(stub) == word


def test_an_unlocked_plan_is_reported_as_unlocked(sandbox: Path) -> None:
    experiments.open_experiment("unlocked", frozen_dir=paths.FROZEN_DIR)
    experiment_id = experiments.current_experiment().experiment_id
    assert "no plan.md" in report._plan_state(experiment_id)

    experiments.plan_path_for(experiment_id).write_text("# plan\n\nI predict +0.005.\n")
    assert "never locked" in report._plan_state(experiment_id)

    experiments.lock_plan(experiment_id)
    assert "locked, unchanged" in report._plan_state(experiment_id)

    # The case the whole lock exists for.
    experiments.plan_path_for(experiment_id).write_text("# plan\n\nI predict what happened.\n")
    assert "**MOVED**" in report._plan_state(experiment_id)


def test_a_failed_audit_check_is_reported_as_failed(sandbox: Path, monkeypatch) -> None:
    from titanic_agent import audit

    broken = audit.AuditReport(
        checks=[
            audit.Check("rule 3 · ledger chain intact", False, "row 7 does not match its prev_sha")
        ]
    )
    monkeypatch.setattr(audit, "audit", lambda **kwargs: broken)
    text = report.render()
    assert "**FAILED**" in text
    assert "**FAIL**" in text
    assert "does not match its prev_sha" in text


def test_an_unread_holdout_and_a_spent_one_read_differently(sandbox: Path) -> None:
    assert "**Unread.**" in report.render()

    paths.HOLDOUT_READS.write_text(
        json.dumps([{"run_id": "r1", "read_utc": "2026-10-12T00:00:00+00:00"}]),
        encoding="utf-8",
    )
    text = report.render()
    assert "**Spent.**" in text and "r1" in text


def test_refusals_and_decisions_are_carried_verbatim(sandbox: Path) -> None:
    run_log.record(
        "refused",
        step="ensemble",
        detail="would need --score-holdout to confirm; that is a human's call",
        pass_id="20261012-1200EDT",
    )
    run_log.record("step_finished", step="tune", detail="noise", pass_id="20261012-1200EDT")
    text = report.render("20261012-1200EDT")
    assert "**REFUSED**" in text
    assert "that is a human's call" in text
    # An info-level event is not a decision and would only pad the section.
    assert "noise" not in text


def test_an_empty_journal_says_a_silent_stop_is_indistinguishable(sandbox: Path) -> None:
    text = report.render("20261012-1200EDT")
    assert "indistinguishable from one that crashed" in text


def test_the_champion_correction_travels_with_the_champion(sandbox: Path) -> None:
    from titanic_agent import champion

    champion.crown(champion.INITIAL, reason="seed " * 6, evidence=["r1"])
    champion.crown(
        {**champion.INITIAL, "name": "rf_kitchen"},
        reason="won " * 8,
        evidence=["r2"],
        experiment_id="exp-007",
        delta_vs_previous=0.0264,
    )
    champion.correct(
        corrects="rf_kitchen (exp-007)",
        field_name="delta_vs_previous",
        was=0.0264,
        now=0.0163,
        reason="selection delta, not the fresh-seed confirmation",
    )
    text = report.render()
    assert "rf_kitchen" in text
    assert "+0.0264" in text
    # The stored number is biased and the report must not print it alone.
    assert "**Corrected:**" in text
    assert "0.0163" in text


def test_writing_publishes_a_copy_and_charges_no_budget(sandbox: Path) -> None:
    experiments.open_experiment("budget", frozen_dir=paths.FROZEN_DIR)
    experiment_id = experiments.current_experiment().experiment_id
    before = dict(experiments.load_experiment(experiment_id).spend)

    written = report.write("20261012-1200EDT")
    assert written.name == "report.md"
    assert (paths.REPORTS_DIR / "report.md").is_file()

    # runs.summarize is the thing that runs exactly once per pass and so is the
    # honest counter. A second counter would double-charge a cap meant to bind.
    assert dict(experiments.load_experiment(experiment_id).spend) == before


def test_it_derives_rather_than_restates(sandbox: Path) -> None:
    """No section may hold a number this module computed for itself.

    The report composes status, audit, leakage, receipts, experiments and
    champion. If it ever grew its own copy of one of those facts, the copy and
    the original could disagree and then neither would be trustworthy -- the
    same reason run.md is derived from the global ledger.
    """
    import inspect

    source = inspect.getsource(report)
    for owner in (
        "audit.audit(frozen_dir=",
        "leakage.check_all(frozen_dir=",
        "receipts.read_receipts()",
        "status.pipeline_status()",
        "champion.current()",
        "experiments.load_experiment",
    ):
        assert owner in source, owner
    # No scoring of its own: accuracy numbers come from the ledger and the
    # experiment manifest, never recomputed here.
    assert "cross_val_score" not in source
    assert "fold_scores" not in source


def test_report_is_regenerable(sandbox: Path) -> None:
    # reports/ is deleted by make clean, and that must be harmless.
    first = report.write("20261012-1200EDT").read_text()
    (paths.REPORTS_DIR / "report.md").unlink()
    second = report.write("20261012-1200EDT").read_text()
    assert first.splitlines()[0] == second.splitlines()[0]
    assert (paths.REPORTS_DIR / "report.md").is_file()


def test_experiments_for_pass_ignores_rows_from_other_passes(sandbox: Path) -> None:
    for pass_id, experiment_id in (
        ("20261012-1200EDT", "exp-mine"),
        ("20261011-0900EDT", "exp-theirs"),
    ):
        results_log.append_run(
            step="tune",
            model="m",
            params={},
            cv_score=0.8,
            pass_id=pass_id,
            experiment_id=experiment_id,
            path=paths.RESULTS_LOG,
        )
    assert report._experiments_for_pass("20261012-1200EDT") == ["exp-mine"]


def test_replace_keeps_the_dataclass_contract(sandbox: Path) -> None:
    # Guards the assumption _verdict makes about Experiment having .status and
    # .summary, so a field rename fails here rather than silently printing
    # every experiment as "verdict not stated".
    experiments.open_experiment("contract", frozen_dir=paths.FROZEN_DIR)
    experiment = experiments.current_experiment()
    closed = replace(experiment, status="closed", summary="crowned something")
    assert report._verdict(closed) == "crowned"


def test_a_report_can_be_written_when_nothing_is_frozen(tmp_path: Path, monkeypatch) -> None:
    """The pass this matters most for is the one that stops at preflight.

    `leakage.derivable_columns` read data/frozen/train.csv unguarded, so
    `make report` raised FileNotFoundError on a checkout with no dataset -- and
    the run skill says to write the report "even when the pass stopped early;
    especially then". Cold start 002 hit exactly this.
    """
    empty = tmp_path / "frozen"
    empty.mkdir()
    monkeypatch.setattr(paths, "FROZEN_DIR", empty)
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(paths, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(paths, "RUN_LOG", tmp_path / "run_log.jsonl")
    monkeypatch.setattr(paths, "HOLDOUT_READS", tmp_path / "holdout_reads.json")
    monkeypatch.setenv(runs.RUN_ID_ENV, "20261012-1500EDT")
    # The provenance guard returns early on an empty ledger, so the crash needs
    # rows *and* no frozen dir -- the post-unfreeze state, not a fresh clone.
    results_log.append_run(
        step="tune", model="random_forest", params={"features": ["Sex"]}, cv_score=0.8
    )

    written = report.write(frozen_dir=empty)
    text = written.read_text()
    assert "Pass 20261012-1500EDT" in text
    # It must say the dataset is missing rather than crash or claim a pass...
    assert "holds no manifest" in text
    assert "**FAILED**" in text
    # ...and the guard that could not run reads as a note, not a pass. A guard
    # that cannot run is not a guard that passed.
    assert "[note] `feature provenance`" in text
    assert "UNVERIFIED" in text


def test_provenance_is_unverified_not_violated_when_nothing_is_frozen(
    tmp_path: Path, log_path: Path
) -> None:
    from titanic_agent import leakage, results_log

    results_log.append_run(
        step="tune",
        model="random_forest",
        params={"features": ["Sex", "Pclass"]},
        cv_score=0.81,
        path=log_path,
    )
    empty = tmp_path / "frozen"
    empty.mkdir()
    assert leakage.derivable_columns(empty) == set()

    check = leakage.check_feature_provenance(log_path=log_path, frozen_dir=empty)
    # An absent dataset is an absence of evidence. Reporting every logged
    # feature as an offender would turn that into an accusation.
    assert check.passed
    assert "UNVERIFIED" in check.detail
    assert check.informational
    assert not check.offenders
