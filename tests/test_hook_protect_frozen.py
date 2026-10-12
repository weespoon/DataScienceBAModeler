"""Smoke test: the PreToolUse hook that enforces Rule 1 in the harness.

The hook lives outside the package (``.claude/hooks/``) because the harness runs
it as a standalone script, so it is loaded here by path. Its policy is a pure
function, :func:`decide`, which is why these tests need no frozen dataset on
disk — and why the hook has no environment-variable override, which would itself
be a documented way around the thing it enforces.

Two properties matter most:

- :func:`test_the_initial_freeze_is_allowed_through` — a deny that also blocks
  ``freeze`` would make the pipeline unrunnable. This is the whole design
  question: how the authorized writer passes a blanket deny.
- :func:`test_bash_heuristic_is_known_to_be_evadable` — documents the leak
  rather than pretending it is closed. Prevention is partial; detection and
  attestation cover the rest.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from titanic_agent import paths

HOOK_PATH = paths.ROOT / ".claude" / "hooks" / "protect_frozen.py"


def _load_hook():
    spec = importlib.util.spec_from_file_location("protect_frozen", HOOK_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hook = _load_hook()


def write(path: str) -> dict:
    return {"tool_name": "Write", "tool_input": {"file_path": path}}


def bash(command: str) -> dict:
    return {"tool_name": "Bash", "tool_input": {"command": command}}


def test_the_hook_exists_and_is_executable() -> None:
    assert HOOK_PATH.is_file()
    assert HOOK_PATH.stat().st_mode & 0o111, "hook must be executable"


def test_the_initial_freeze_is_allowed_through() -> None:
    # Before a manifest exists, freeze() must be able to write these files.
    # A hook that denied this would make the pipeline unrunnable.
    for payload in (
        write("data/frozen/train.csv"),
        write("data/frozen/holdout.csv"),
        write("data/frozen/MANIFEST.json"),
        bash("chmod 0444 data/frozen/train.csv"),
    ):
        assert hook.decide(payload, is_frozen=False) is None


@pytest.mark.parametrize(
    "path",
    [
        "data/frozen/train.csv",
        "data/frozen/holdout.csv",
        "data/frozen/MANIFEST.json",
        "data/frozen/nested/deeper.csv",
        "src/../data/frozen/sneaky.csv",
        "./data/frozen/train.csv",
    ],
)
def test_writes_into_frozen_are_denied_once_frozen(path: str) -> None:
    assert hook.decide(write(path), is_frozen=True) is not None


def test_absolute_paths_are_denied_too() -> None:
    absolute = str(paths.FROZEN_DIR / "train.csv")
    assert hook.decide(write(absolute), is_frozen=True) is not None


@pytest.mark.parametrize(
    "path",
    [
        "data/raw/train.csv",
        "experiments/results.csv",
        "experiments/holdout_reads.json",
        "reports/02_explore.md",
        "src/titanic_agent/tune.py",
        "data/frozen_notes.md",  # adjacent name, not inside the directory
    ],
)
def test_writes_elsewhere_are_untouched(path: str) -> None:
    assert hook.decide(write(path), is_frozen=True) is None


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf data/frozen/*",
        "chmod -R u+w data/frozen",
        "echo x > data/frozen/train.csv",
        "cat junk >> data/frozen/holdout.csv",
        "mv data/frozen/train.csv /tmp/",
        "cp /tmp/x.csv data/frozen/train.csv",
        "truncate -s 0 data/frozen/train.csv",
        "touch data/frozen/marker",
        "sed -i '' s/a/b/ data/frozen/train.csv",
        # Redirect forms that genuinely write, after the pattern was narrowed
        # to stop refusing `2>/dev/null`. If any of these starts passing, the
        # narrowing opened a hole.
        "echo x >data/frozen/train.csv",
        "echo x >  data/frozen/train.csv",
        "cat foo >| data/frozen/train.csv",
        "echo x > data/frozen/nested/deep.csv",
    ],
)
def test_mutating_bash_against_frozen_is_denied(command: str) -> None:
    assert hook.decide(bash(command), is_frozen=True) is not None


@pytest.mark.parametrize(
    "command",
    [
        "cat data/frozen/MANIFEST.json",
        "shasum -a 256 data/frozen/train.csv",
        "head -5 data/frozen/train.csv",
        "ls -la data/frozen",
        "make verify",
        "wc -l data/frozen/train.csv",
        # A file-descriptor redirect is not a write. The bare `>` alternation
        # refused all of these, which is why a reviewer could not even list the
        # directory while investigating.
        "ls data/frozen 2>/dev/null",
        "ls data/frozen/hold*.csv 2>/dev/null",
        "cat data/frozen/MANIFEST.json 2>&1",
        "shasum -a 256 data/frozen/*.csv 2>/dev/null | sort",
        "wc -l < data/frozen/train.csv",
    ],
)
def test_reading_frozen_from_bash_is_allowed(command: str) -> None:
    # Reading train.csv is legitimate. Reading holdout.csv is a Rule 2 matter,
    # which this hook deliberately does not police -- see receipts.py for why
    # that one is handled by attestation instead.
    assert hook.decide(bash(command), is_frozen=True) is None


def test_mutating_bash_elsewhere_is_allowed() -> None:
    assert hook.decide(bash("rm -rf .pytest_cache"), is_frozen=True) is None
    assert hook.decide(bash("echo hi > reports/notes.md"), is_frozen=True) is None


def test_unknown_tools_are_not_blocked() -> None:
    assert hook.decide({"tool_name": "Read", "tool_input": {}}, is_frozen=True) is None
    assert hook.decide({}, is_frozen=True) is None


def test_bash_heuristic_is_known_to_be_evadable() -> None:
    # Documented, not fixed. Bash gives the hook a command string rather than a
    # structured target, so a caller that does not spell the path can get past
    # it. This is why Rule 1 also has checksums (detect) and why Rule 2 relies
    # on attestation: prevention is the strongest guarantee but never complete.
    evasion = bash('python3 -c "import pathlib;pathlib.Path(chr(100)+chr(47)).write_text(1)"')
    assert hook.decide(evasion, is_frozen=True) is None


@pytest.mark.parametrize(
    "command",
    [
        # The commit that introduced this hook was itself denied, because its
        # message quoted the command the rule forbids. The tempting fix was to
        # reword the message; an over-broad constraint creates pressure to
        # degrade the record, so the pattern got fixed instead.
        "git commit -F - <<'EOF'\nthis message mentions chmod -R u+w data/frozen\nEOF",
        'git commit -m "describe chmod on data/frozen in a message"',
        "git commit --message='rm -rf data/frozen was denied, as intended'",
        "git status",
        "git log --oneline",
        "git clean -fd",  # removes untracked files, but not ignored ones
    ],
)
def test_prose_about_the_rule_is_not_an_attempt_to_break_it(command: str) -> None:
    assert hook.decide(bash(command), is_frozen=True) is None


def test_a_quoted_path_is_still_a_real_target() -> None:
    # Stripping heredocs and -m arguments must not become "ignore all quotes":
    # this one is a genuine attempt.
    assert hook.decide(bash('rm -rf "data/frozen"'), is_frozen=True) is not None


@pytest.mark.parametrize(
    "command",
    [
        "git clean -xfd",
        "git clean -X",
        "git clean --force -x",
        "git stash --all",
        "git stash -ua",
    ],
)
def test_git_commands_that_delete_ignored_files_are_denied(command: str) -> None:
    # Everything under data/frozen is gitignored. That is why no ordinary git
    # command can touch it -- and exactly why these can, without ever naming
    # the path. Found while fixing the false positive above.
    assert hook.decide(bash(command), is_frozen=True) is not None


def test_those_git_commands_are_fine_before_freezing(tmp_path) -> None:
    assert hook.decide(bash("git clean -xfd"), is_frozen=False) is None


def test_decide_is_pure_and_takes_no_env_override() -> None:
    import inspect

    source = HOOK_PATH.read_text(encoding="utf-8")
    assert "os.environ" not in source, "an env override would be a way around the hook"
    assert set(inspect.signature(hook.decide).parameters) == {"payload", "is_frozen", "root"}


# --------------------------------------------------------------------------- #
# the shim, under the interpreter the harness actually uses
# --------------------------------------------------------------------------- #


def _run_hook_bare(payload: dict) -> subprocess.CompletedProcess:
    """Run the hook as the harness does: system python3, no venv, no PYTHONPATH.

    Every other test in this file calls ``decide()`` in-process, which never
    touches the parts that talk to the harness. That gap let a refactor break
    the deny path while all 56 of them stayed green.
    """
    environment = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp")}
    return subprocess.run(
        ["/usr/bin/python3", str(HOOK_PATH)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )


@pytest.mark.skipif(not Path("/usr/bin/python3").exists(), reason="no system python3")
def test_the_hook_denies_under_a_bare_interpreter() -> None:
    """The constraint the guard move could have broken silently.

    The policy now lives in ``src/titanic_agent/guard.py``, loaded by path so
    ``titanic_agent/__init__.py`` never runs. If that import ever fails under
    the system interpreter the hook raises, and because it fails **open** —
    ``main`` calls ``allow()`` on bad input and exits 0 — Rule 1 would stop
    being enforced with nothing to notice.

    This caught a real break: the refactor dropped ``DENY_REASON``, so the deny
    path raised ``NameError`` and the hook exited 1 with a traceback instead of
    denying the write.
    """
    result = _run_hook_bare({"tool_name": "Write", "tool_input": {"file_path": "data/frozen/x"}})
    assert result.returncode == 0, f"the hook must never wedge the session: {result.stderr}"
    assert not result.stderr, result.stderr
    decision = json.loads(result.stdout)["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"
    assert "Rule 1" in decision["permissionDecisionReason"]


@pytest.mark.skipif(not Path("/usr/bin/python3").exists(), reason="no system python3")
def test_the_hook_allows_elsewhere_under_a_bare_interpreter() -> None:
    result = _run_hook_bare({"tool_name": "Write", "tool_input": {"file_path": "reports/x.md"}})
    assert result.returncode == 0
    assert "permissionDecision" not in result.stdout


def test_the_policy_file_imports_nothing_outside_the_stdlib() -> None:
    """``guard.py`` is loaded by the system interpreter, which has no pandas.

    Also: no ``from . import paths``. The hook derives the repo root from
    ``.claude/hooks/../..`` and ``paths.ROOT`` derives it from
    ``src/titanic_agent/../..`` -- the same answer today and two ways to be
    wrong later, which is why root is a parameter.
    """
    source = (Path(hook.ROOT) / "src" / "titanic_agent" / "guard.py").read_text(encoding="utf-8")
    imported = set(re.findall(r"^(?:from|import)\s+([\w.]+)", source, re.MULTILINE))
    assert imported <= {"__future__", "re", "pathlib"}, imported
    assert "from . import" not in source
