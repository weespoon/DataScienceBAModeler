"""``make agent`` — the headless pass launcher.

Nothing here starts a real pass. What is worth testing is the shape of the
invocation and the two things that would quietly undermine it: a permission mode
that skips the deny rules, and a wrapper that reads the pass's result and acts
on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from titanic_agent import agent, paths, runs


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(paths, "RUN_LOG", tmp_path / "run_log.jsonl")
    monkeypatch.delenv(runs.RUN_ID_ENV, raising=False)
    return tmp_path


def test_the_command_caps_turns_and_dollars() -> None:
    command = agent.build_command(turns=12, budget=2.5)
    assert command[:2] == ["claude", "-p"]
    assert "--max-turns" in command and command[command.index("--max-turns") + 1] == "12"
    assert "--max-budget-usd" in command
    assert command[command.index("--max-budget-usd") + 1] == "2.5"


def test_the_permission_mode_does_not_bypass_the_deny_rules() -> None:
    """``bypassPermissions`` would skip the settings that make this safe.

    The deny rules in ``.claude/settings.json`` are what stop a headless pass
    reaching ``chmod`` under ``data/frozen`` or the one route to
    ``--score-holdout``. A mode that skips the permission system skips them, so
    the cap on what a pass may do would be the prompt alone.
    """
    command = agent.build_command(turns=1, budget=1)
    mode = command[command.index("--permission-mode") + 1]
    assert mode != "bypassPermissions"
    assert mode == agent.PERMISSION_MODE == "acceptEdits"


def test_the_prompt_points_at_the_run_skill_and_names_the_tripwire() -> None:
    assert ".claude/skills/run/SKILL.md" in agent.PROMPT
    assert "One experiment" in agent.PROMPT
    # The prompt is not the guard, but a prompt that omitted this would be
    # asking an unattended pass to rediscover Rule 2 from scratch.
    assert "--score-holdout" in agent.PROMPT
    assert "holdout.csv" in agent.PROMPT


def test_a_dry_run_leaves_no_run_directory(sandbox: Path, capsys) -> None:
    # It used to mint one before checking, so `make runs` listed a pass that
    # never happened.
    assert agent.run(dry_run=True, turns=3, budget=1) == 0
    assert "claude" in capsys.readouterr().out
    assert runs.list_runs() == []


def test_a_missing_cli_is_reported_rather_than_crashing(sandbox: Path, monkeypatch) -> None:
    monkeypatch.setattr(agent.shutil, "which", lambda name: None)
    assert agent.run(turns=3, budget=1) == 127
    assert runs.list_runs() == []


def test_the_pass_id_is_minted_once_and_exported(sandbox: Path, monkeypatch) -> None:
    """Every tool call inside the pass has to agree on one run directory.

    Without this the subprocess mints its own id per step and the pass's
    artifacts scatter across several directories.
    """
    seen: dict[str, str] = {}

    class Completed:
        returncode = 0
        stdout = json.dumps({"result": "done", "num_turns": 4, "total_cost_usd": 0.1})
        stderr = ""

    def fake_run(command, **kwargs):
        seen.update(kwargs["env"])
        return Completed()

    monkeypatch.setattr(agent.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(agent.subprocess, "run", fake_run)
    assert agent.run(turns=4, budget=1) == 0

    directories = runs.list_runs()
    assert len(directories) == 1
    assert seen[runs.RUN_ID_ENV] == directories[0].name


def test_the_transcript_and_the_metadata_land_in_the_run_directory(
    sandbox: Path, monkeypatch
) -> None:
    payload = {
        "result": "stopped at preflight: two tool calls denied",
        "num_turns": 7,
        "total_cost_usd": 0.42,
        "subtype": "success",
    }

    class Completed:
        returncode = 0
        stdout = json.dumps(payload)
        stderr = "a warning"

    monkeypatch.setattr(agent.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(agent.subprocess, "run", lambda command, **kwargs: Completed())
    agent.run(turns=7, budget=1)

    directory = runs.list_runs()[0]
    # The normalised record is what makes "it ran out of turns before closing"
    # checkable, and comparable across runners rather than only within one.
    record = json.loads((directory / "claude-code.json").read_text())
    assert record["runner"] == "claude-code"
    assert record["turns"] == 7 and record["turn_cap"] == 7
    assert record["cost_usd"] == 0.42
    assert record["stop_reason"] == "success"
    # The CLI's own output is kept beside it, un-normalised.
    assert json.loads((directory / "claude-code.raw.json").read_text())["num_turns"] == 7
    assert "two tool calls denied" in (directory / "claude-code.log").read_text()
    assert (directory / "claude-code.stderr.log").read_text() == "a warning"
    meta = json.loads((directory / "metadata.json").read_text())
    assert meta["launched_by"] == "claude-code"
    assert meta["turn_cap"] == 7


def test_a_failed_pass_is_journaled_as_failed_and_not_retried(sandbox: Path, monkeypatch) -> None:
    """One invocation, one pass, whatever it returns.

    A wrapper that retried a failed pass -- or re-ran one whose verdict it
    disliked -- would be selecting on the outcome, which is the dishonesty this
    repo exists to detect. So the exit code is passed through, not handled.
    """
    from titanic_agent import run_log

    calls = []

    class Completed:
        returncode = 1
        stdout = json.dumps({"result": "failed", "subtype": "error_max_turns"})
        stderr = ""

    def fake_run(command, **kwargs):
        calls.append(command)
        return Completed()

    monkeypatch.setattr(agent.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(agent.subprocess, "run", fake_run)
    assert agent.run(turns=2, budget=1) == 1
    assert len(calls) == 1

    events = [e["event"] for e in run_log.read_journal()]
    assert "step_failed" in events
    detail = next(e for e in run_log.read_journal() if e["event"] == "step_failed")["detail"]
    assert "error_max_turns" in detail


def test_a_field_the_cli_did_not_report_stays_none() -> None:
    # A cap nobody counts is a number pretending to be a limit; a count nobody
    # measured is worse. None means "not measured", never zero.
    class Completed:
        returncode = 0
        stdout = json.dumps({"num_turns": 3})
        stderr = ""

    record = agent._record("20261012-1600EDT", Completed(), turns=10, budget=1.0)
    assert record.turns == 3
    assert record.cost_usd is None
    assert record.duration_ms is None
    assert record.stop_reason is None
    assert record.hit_the_cap is False

    class Garbage:
        returncode = 1
        stdout = "not json"
        stderr = ""

    blank = agent._record("20261012-1600EDT", Garbage(), turns=10, budget=1.0)
    assert blank.turns is None
    assert blank.hit_the_cap is None  # unmeasured, not "did not hit it"
