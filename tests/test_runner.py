"""The shared runner record: what makes three runners comparable.

Each runner reports spend in its own vocabulary -- the Claude CLI's
``num_turns`` / ``total_cost_usd`` / ``subtype``, the Agent SDK's
``ResultMessage`` fields, and whatever a raw Messages API loop computes from
``usage`` against a price table. Three shapes is no common denominator, so each
translates into one record here.

The tests that matter are the two negative ones: a field nobody measured stays
None rather than becoming zero, and the record carries no verdict, so a retry
loop cannot be built on it without adding the thing it is missing on purpose.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from titanic_agent import paths, runner, runs


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(paths, "RUN_LOG", tmp_path / "run_log.jsonl")
    monkeypatch.delenv(runs.RUN_ID_ENV, raising=False)
    return tmp_path


def test_start_mints_one_id_and_journals_it(sandbox: Path) -> None:
    from titanic_agent import run_log

    pass_id = runner.start("claude-code", turn_cap=60, budget_usd=5)
    assert runs.RUN_ID_PATTERN.match(pass_id)
    assert [d.name for d in runs.list_runs()] == [pass_id]

    meta = json.loads((runs.run_dir(pass_id) / "metadata.json").read_text())
    assert meta["launched_by"] == "claude-code"
    assert meta["turn_cap"] == 60
    started = [e for e in run_log.read_journal() if e["event"] == "step_started"]
    assert started and started[-1]["pass_id"] == pass_id


def test_the_record_has_no_verdict_field(sandbox: Path) -> None:
    """The omission is the point.

    `make agent` promises it "does not read, parse or act on the pass's
    result", because a wrapper that re-ran a pass whose verdict it disliked
    would be selecting on the outcome. A verdict in this record is one refactor
    from exactly that, so there isn't one -- the verdict lives in the
    experiment manifest and in `make report`, written by the pass itself.
    """
    names = {f.name for f in fields(runner.PassRecord)}
    for forbidden in ("verdict", "crowned", "held", "best_cv", "val_score", "passed"):
        assert forbidden not in names, f"PassRecord gained a {forbidden!r} field"


def test_finish_records_and_does_not_branch_on_the_outcome(sandbox: Path) -> None:
    from titanic_agent import run_log

    pass_id = runner.start("sdk", turn_cap=5)
    failed = runner.PassRecord(
        runner="sdk", pass_id=pass_id, exit_code=1, turns=5, turn_cap=5, stop_reason="max_turns"
    )
    written = runner.finish(failed, raw='{"x": 1}', summary="stopped at preflight")

    assert written.name == "sdk.json"
    assert json.loads(written.read_text())["stop_reason"] == "max_turns"
    assert json.loads((written.parent / "sdk.raw.json").read_text()) == {"x": 1}
    assert (written.parent / "sdk.log").read_text() == "stopped at preflight"

    events = [e["event"] for e in run_log.read_journal()]
    assert "step_failed" in events


def test_unmeasured_is_none_and_never_zero() -> None:
    bare = runner.PassRecord(runner="raw-api", pass_id="20261012-1600EDT")
    assert bare.turns is None and bare.cost_usd is None
    # Not False: "nobody counted" and "it did not hit the cap" are different
    # claims, and conflating them is how an unenforced cap reads as enforced.
    assert bare.hit_the_cap is None

    counted = runner.PassRecord(runner="raw-api", pass_id="x", turns=60, turn_cap=60)
    assert counted.hit_the_cap is True
    assert runner.PassRecord(runner="x", pass_id="y", turns=3, turn_cap=60).hit_the_cap is False


def test_records_from_different_runners_read_back_together(sandbox: Path) -> None:
    for name in ("claude-code", "sdk", "raw-api"):
        pass_id = runner.start(name, turn_cap=10)
        runner.finish(runner.PassRecord(runner=name, pass_id=pass_id, exit_code=0, turns=4))

    found = runner.read_records()
    assert sorted(r.runner for r in found) == ["claude-code", "raw-api", "sdk"]
    assert all(r.schema == runner.SCHEMA for r in found)


def test_metadata_and_raw_output_are_not_mistaken_for_records(sandbox: Path) -> None:
    # Both are .json files in the same directory; only one is a record.
    pass_id = runner.start("claude-code", turn_cap=1)
    runner.finish(
        runner.PassRecord(runner="claude-code", pass_id=pass_id, exit_code=0),
        raw='{"num_turns": 9}',
    )
    found = runner.read_records(pass_id)
    assert len(found) == 1
    assert found[0].runner == "claude-code"


def test_the_prompt_is_shared_and_names_the_policy() -> None:
    from titanic_agent import agent, surface

    # Two runners with two prompts is a comparison of prompts.
    assert agent.PROMPT is runner.PROMPT
    assert surface.policy_path().name in runner.PROMPT
    assert "One experiment" in runner.PROMPT
    assert "--score-holdout" in runner.PROMPT
