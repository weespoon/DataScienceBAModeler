"""The permission settings, and the two writers an unattended pass needs.

A headless pass runs with an allowlist, so every command the ``run`` skill tells
it to use has to be on that list and every route to the two tripwires has to be
off it. Those are both mechanical, and both were wrong on the first attempt:
the first real pass stopped at preflight unable to journal why, because
recording a stop reason meant running arbitrary Python.

The settings are checked against the Makefile and the skill rather than against
a hand-written list, so a target added to one and forgotten in the other fails
here instead of at 3am in an unattended pass.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from titanic_agent import agent, findings, paths, run_log, surface

SETTINGS = paths.ROOT / ".claude" / "settings.json"
AGENT_SETTINGS = paths.ROOT / ".claude" / "agent-settings.json"
SKILL = paths.ROOT / ".claude" / "skills" / "run" / "SKILL.md"


def rules(path: Path, kind: str) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("permissions", {}).get(kind, [])


def allowed_targets() -> set[str]:
    return {
        match.group(1)
        for rule in rules(SETTINGS, "allow")
        if (match := re.fullmatch(r"Bash\(make ([a-z][a-z0-9-]*):\*\)", rule))
    }


# --------------------------------------------------------------------------- #
# the deny list is the part that matters
# --------------------------------------------------------------------------- #


def test_no_route_to_the_final_test_is_allowed() -> None:
    denied = rules(SETTINGS, "deny")
    # Rule 2's tripwire. ensemble is the only module that can read the holdout,
    # so the module, the flag and the file are each denied by name.
    assert "Bash(*--score-holdout*)" in denied
    assert "Bash(*titanic_agent.ensemble*)" in denied
    assert "Bash(*holdout.csv*)" in denied
    assert "Read(./data/frozen/holdout.csv)" in denied
    assert "make ensemble" not in {t for t in allowed_targets()}


def test_the_raw_data_is_unreadable_because_it_still_holds_the_holdout_labels() -> None:
    # data/raw/train.csv is all 891 labelled rows, 179 of which are the final
    # test. Reading it is a holdout read by another name.
    assert "Read(./data/raw/**)" in rules(SETTINGS, "deny")


def test_rule_1_is_denied_as_well_as_hooked() -> None:
    denied = rules(SETTINGS, "deny")
    assert "Bash(chmod:*)" in denied
    assert "Write(./data/frozen/**)" in denied
    assert "Edit(./data/frozen/**)" in denied
    assert "Bash(make freeze:*)" in denied
    # The hook is the guard; these are a second layer that does not replace it.
    payload = json.loads(SETTINGS.read_text(encoding="utf-8"))
    hook = payload["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert "protect_frozen.py" in hook


@pytest.mark.parametrize(
    "path",
    [
        "./experiments/results.csv",
        "./experiments/findings.jsonl",
        "./experiments/run_log.jsonl",
        "./experiments/champion.jsonl",
        "./experiments/holdout_reads.json",
    ],
)
def test_append_only_evidence_cannot_be_written_directly(path: str) -> None:
    denied = rules(SETTINGS, "deny")
    assert f"Write({path})" in denied, path
    assert f"Edit({path})" in denied, path


def test_the_interpreter_lockdown_is_headless_only() -> None:
    """``python -c`` is a route to anything, including ``--score-holdout``.

    Denying it is right for an unattended pass and wrong for an interactive
    one, where a person is watching and the tripwires are explicitly theirs to
    decide. A rule that makes the repo tedious to work in gets switched off, so
    it lives in the settings ``make agent`` merges rather than in the project's.
    """
    headless = rules(AGENT_SETTINGS, "deny")
    assert "Bash(python:*)" in headless
    assert "Bash(python3:*)" in headless
    assert "Bash(curl:*)" in headless
    assert "Bash(git push:*)" in headless
    # ...and not in the project settings, which apply to humans too.
    assert "Bash(python3:*)" not in rules(SETTINGS, "deny")
    assert agent.AGENT_SETTINGS == ".claude/agent-settings.json"
    assert AGENT_SETTINGS.is_file()
    command = agent.build_command(turns=1, budget=1)
    assert command[command.index("--settings") + 1] == agent.AGENT_SETTINGS


# --------------------------------------------------------------------------- #
# the allowlist has to cover what the skill actually tells it to run
# --------------------------------------------------------------------------- #


def test_every_make_target_the_run_skill_names_is_allowed() -> None:
    named = surface.targets_named_in(SKILL.read_text(encoding="utf-8")) & surface.make_targets()
    # explore and loop are named only to say *not* to run them inside an
    # experiment, so they are deliberately absent from the allowlist.
    expected = named - {"explore", "loop", "pull", "bootstrap"}
    missing = sorted(expected - allowed_targets())
    assert not missing, (
        f"the run skill tells a pass to use `make {missing}` but the allowlist does not "
        "cover it, so a headless pass will stall on it"
    )


def test_a_pass_can_write_a_plan_and_a_report_but_not_the_ledger() -> None:
    allowed = rules(SETTINGS, "allow")
    assert "Write(./experiments/**/plan.md)" in allowed
    assert "Write(./runs/**)" in allowed
    assert "Write(./reports/**)" in allowed
    assert "Write(./experiments/results.csv)" not in allowed


# --------------------------------------------------------------------------- #
# the two writers the allowlist makes necessary
# --------------------------------------------------------------------------- #


def test_the_journal_can_be_written_without_running_python(monkeypatch, tmp_path) -> None:
    """The gap that stopped the first headless pass dead.

    It halted at preflight on a denied call, then could not record why, because
    the journal write was `python -c "...run_log.record(...)"`. A pass that
    stops without saying why is indistinguishable from one that crashed.
    """
    monkeypatch.setattr(paths, "RUN_LOG", tmp_path / "run_log.jsonl")
    monkeypatch.setenv("TITANIC_DETAIL", "stopped: make audit reported a broken chain")
    assert run_log.main(["--record-env", "TITANIC_DETAIL", "--as-event", "decision"]) == 0
    entries = run_log.read_journal()
    assert entries[-1]["detail"] == "stopped: make audit reported a broken chain"
    assert entries[-1]["event"] == "decision"
    assert "Bash(make note:*)" in rules(SETTINGS, "allow")


def test_a_finding_can_be_recorded_without_running_python(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(paths, "FINDINGS", tmp_path / "findings.jsonl")
    monkeypatch.setenv("TITANIC_CLAIM", "rf_wcg ties the champion")
    monkeypatch.setenv("TITANIC_NOTE", "paired on identical folds")
    code = findings.main(
        [
            "--record-env",
            "TITANIC_CLAIM",
            "--note-env",
            "TITANIC_NOTE",
            "--as-status",
            "confirmed",
            "--evidence",
            "r1,r2",
            "--confidence",
            "medium",
            "--delta",
            "0.0",
        ]
    )
    assert code == 0
    recorded = findings.read_findings()[-1]
    assert recorded.claim == "rf_wcg ties the champion"
    assert recorded.evidence == ["r1", "r2"]
    assert recorded.status == "confirmed"
    assert "Bash(make finding:*)" in rules(SETTINGS, "allow")


def test_prose_with_quotes_survives_the_round_trip(monkeypatch, tmp_path) -> None:
    # `DETAIL_Q = '$(DETAIL)'` broke on the first apostrophe, which is why the
    # text goes through the environment rather than the shell.
    awkward = 'it\'s a "result", not a decision -- 0.003 is 1.7 passengers'
    monkeypatch.setattr(paths, "RUN_LOG", tmp_path / "run_log.jsonl")
    monkeypatch.setenv("TITANIC_DETAIL", awkward)
    run_log.main(["--record-env", "TITANIC_DETAIL"])
    assert run_log.read_journal()[-1]["detail"] == awkward


def test_an_empty_detail_is_refused_rather_than_journaled(monkeypatch, tmp_path) -> None:
    # `make note` with no DETAIL= would otherwise append a blank entry, which is
    # worse than no entry: it looks like something was recorded.
    monkeypatch.setattr(paths, "RUN_LOG", tmp_path / "run_log.jsonl")
    monkeypatch.setenv("TITANIC_DETAIL", "   ")
    assert run_log.main(["--record-env", "TITANIC_DETAIL"]) == 1
    assert not os.path.exists(tmp_path / "run_log.jsonl")


# --------------------------------------------------------------------------- #
# make variable injection
# --------------------------------------------------------------------------- #


def expand(target: str, *assignments: str) -> str:
    """What `make -n <target> VAR=...` would hand to /bin/sh."""
    import subprocess

    result = subprocess.run(
        ["make", "-n", target, *assignments],
        cwd=paths.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


def test_the_interpreter_cannot_be_replaced_from_the_command_line() -> None:
    """`make <any target> PYTHON=...` was arbitrary execution.

    A command-line assignment beats a Makefile `:=`, so this turned every
    allowlisted target into a shell of the caller's choosing -- regardless of
    which targets an allowlist permitted. `Bash(make champion:*)` is a prefix
    match, so nothing in .claude/settings.json stood in the way. `override` is
    the fix, and unlike a Python-side check it also protects a human typing at
    a terminal.
    """
    for variable in ("PYTHON", "BIN", "VENV"):
        recipe = expand("champion", f"{variable}=echo PWNED")
        assert "PWNED" not in recipe, f"{variable} still reaches the recipe"
        assert ".venv/bin/python" in recipe


@pytest.mark.parametrize(
    ("target", "assignment", "flag"),
    [
        ("tune", "MODEL=all; echo PWNED", "--model"),
        ("bakeoff", "ARM_SET=bakeoff; echo PWNED", "--set"),
        ("bakeoff", "ONLY=a; echo PWNED", "--only"),
        ("ensemble", "METHOD=soft_vote; echo PWNED", "--method"),
        ("journal", "LEVEL=info; echo PWNED", "--level"),
        ("tune", "N_ITER=25; echo PWNED", "--n-iter"),
        ("agent", "TURNS=60; echo PWNED", "--turns"),
        ("report", "PASS=x; echo PWNED", "--pass-id"),
    ],
)
def test_a_variable_carrying_a_shell_command_is_emptied(
    target: str, assignment: str, flag: str
) -> None:
    # Empty, not executed: the Python side then refuses a missing argument, so
    # a bad value fails the command instead of running it.
    recipe = expand(target, assignment)
    assert "PWNED" not in recipe, f"{assignment} reached the shell"


@pytest.mark.parametrize(
    ("target", "assignment", "expected"),
    [
        ("tune", "MODEL=svc", "--model svc"),
        ("bakeoff", "ARM_SET=decompose", "--set decompose"),
        ("bakeoff", "ONLY=rf_wcg,rf_group", "--only rf_wcg,rf_group"),
        ("journal", "LEVEL=decision", "--level decision"),
        ("tune", "N_ITER=40", "--n-iter 40"),
        ("agent", "BUDGET=2.5", "--budget 2.5"),
    ],
)
def test_legitimate_values_still_reach_the_command(
    target: str, assignment: str, expected: str
) -> None:
    # A guard that also blocks the intended use is one someone will remove.
    # Note the comma-separated list: `safe` takes the variable *name*, so a
    # comma in the value never hits make's own $(call) argument parser.
    assert expected in expand(target, assignment)


def test_prose_variables_never_reach_a_shell_at_all() -> None:
    """NAME and SUMMARY hold sentences, so emptying them would be the wrong fix.

    `experiments open "$(NAME)"` let a double quote close the quote early. They
    now travel in the environment, the same way DETAIL and CLAIM already did --
    that fix had been applied to three variables and not to these two.
    """
    makefile = (paths.ROOT / "Makefile").read_text(encoding="utf-8")
    assert '"$(NAME)"' not in makefile
    assert '"$(SUMMARY)"' not in makefile
    assert "experiment-open: export TITANIC_NAME" in makefile
    assert "experiment-close: export TITANIC_SUMMARY" in makefile

    recipe = expand("experiment-close", "SUMMARY=held; the bar was not cleared")
    assert "--summary-env TITANIC_SUMMARY" in recipe
    # The semicolon is prose here and must not appear as a shell separator.
    assert "the bar was not cleared" not in recipe


def test_a_semicolon_in_a_summary_survives_to_the_process(tmp_path, monkeypatch) -> None:
    from titanic_agent import experiments

    monkeypatch.setenv("TITANIC_SUMMARY", 'held; "ties" is the honest word')
    monkeypatch.setattr(paths, "FROZEN_DIR", tmp_path)
    # No experiment is open, so it fails -- but on the right thing, which proves
    # the text arrived whole rather than being split or emptied.
    assert experiments.main(["close", "--summary-env", "TITANIC_SUMMARY"]) == 1
