"""``make agent`` — one headless autonomous pass, start to finish.

Everything else in this package is a *step*: a human (or an agent reading
``.claude/skills/run/SKILL.md``) decides what to run and runs it. This module is
the one that decides nothing. It launches `claude -p` with a prompt pointing at
that skill, caps what the pass may spend, and keeps the transcript beside the
rows the pass produced.

Why a module rather than a line in the Makefile
    The pass id has to be minted *before* the subprocess starts and exported
    into it, or every tool call inside the pass mints its own and the artifacts
    scatter across half a dozen run directories. The Makefile's ``loop`` target
    does this with ``$(shell date)``, which cannot see whether that directory
    already exists; :func:`runs.mint_run_id` can, and it suffixes a collision
    rather than overwriting a previous pass.

What it deliberately does not do
    It does not pass ``--permission-mode bypassPermissions``. The deny rules in
    ``.claude/settings.json`` are load-bearing — they are what stops a headless
    pass from reaching ``chmod`` on ``data/frozen`` or the one route to
    ``--score-holdout`` — and a mode that skips the permission system skips
    them. ``acceptEdits`` plus the allowlist is the combination that lets a pass
    run unattended *and* still be refused something.

    It also does not read, parse or act on the pass's result. A wrapper that
    retried a failed pass, or re-ran one whose verdict it disliked, would be
    selecting on the outcome — the specific dishonesty this repo exists to
    detect. One invocation, one pass, whatever it returns.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

from . import paths, runner, runs

#: The pass's turn cap. A turn is one model response, so this bounds how long an
#: unattended pass can run without bounding what it is allowed to conclude.
#: Undocumented in ``claude --help`` as of CLI 2.1.296 but accepted; unknown
#: options are rejected outright, so a typo here fails loudly rather than
#: becoming a cap that does not bind.
DEFAULT_TURNS: int = 60

#: Dollar ceiling, as a second backstop with a different failure mode: a pass
#: that thrashes cheaply hits the turn cap, one that reasons expensively hits
#: this.
DEFAULT_BUDGET_USD: float = 5.0

#: Edits are auto-approved; Bash goes through the allowlist and deny rules in
#: ``.claude/settings.json``. See the module docstring on why not
#: ``bypassPermissions``.
#: This runner's name. It names the record file (runs/<id>/claude-code.json)
#: and the launched_by field, so a comparison can tell them apart.
RUNNER: str = "claude-code"

PERMISSION_MODE: str = "acceptEdits"

#: Extra deny rules that apply to a *headless* pass only, merged on top of the
#: project settings (``--settings`` loads additional settings; it does not
#: replace, so the Rule 1 hook stays).
#:
#: Why these are not in ``.claude/settings.json``: they ban the interpreters
#: (``python``, ``uv run``), the network and the git mutations. That is right for
#: an unattended pass, where ``python -c`` is a route to anything including
#: ``--score-holdout``. It is wrong for an interactive session, where a person
#: is watching and the tripwires are explicitly theirs to decide — and a rule
#: that makes the repo tedious to work in is a rule that gets switched off.
#: Rules that help in both places (``chmod``, ``holdout.csv``, the append-only
#: files) live in the project settings instead.
AGENT_SETTINGS: str = ".claude/agent-settings.json"

#: One copy, in runner.py, so a comparison between runners is not partly a
#: comparison of prompts.
PROMPT = runner.PROMPT


def build_command(*, turns: int, budget: float, prompt: str = PROMPT) -> list[str]:
    """The ``claude -p`` invocation for one pass."""
    return [
        "claude",
        "-p",
        prompt,
        "--max-turns",
        str(turns),
        "--max-budget-usd",
        str(budget),
        "--permission-mode",
        PERMISSION_MODE,
        "--settings",
        AGENT_SETTINGS,
        "--output-format",
        "json",
    ]


def run(
    *,
    turns: int = DEFAULT_TURNS,
    budget: float = DEFAULT_BUDGET_USD,
    run_id: str | None = None,
    dry_run: bool = False,
) -> int:
    """Launch one headless pass. Returns the subprocess exit code.

    Writes two files into the pass's run directory:

    ``agent.json``
        The CLI's own record: turn count, cost, duration and stop reason. This
        is what makes "it ran out of turns before closing the experiment" a
        checkable claim rather than a guess from a truncated transcript.
    ``agent.log``
        The final assistant message, for a human who wants the summary without
        parsing JSON.

    The ledger and the journal remain the authority on what the pass *did*;
    these two say what the pass *was*.
    """

    if shutil.which("claude") is None:
        print(
            "make agent needs the `claude` CLI on PATH. Install Claude Code, or "
            "run a pass by hand following .claude/skills/run/SKILL.md."
        )
        return 127

    # Before minting anything: a dry run that left a run directory behind would
    # make `make runs` list a pass that never happened.
    command = build_command(turns=turns, budget=budget)
    if dry_run:
        print(" ".join(command))
        return 0

    resolved = run_id or runner.start(
        RUNNER, turn_cap=turns, budget_usd=budget, permission_mode=PERMISSION_MODE
    )
    directory = runs.run_dir(resolved)

    # Exported so every tool call inside the pass agrees on one run directory.
    environment = {**os.environ, runs.RUN_ID_ENV: resolved}
    print(f"pass {resolved} -> {_shown(directory)}/")

    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        command,
        cwd=paths.ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    summary = _final_message(completed.stdout)
    written = runner.finish(
        _record(resolved, completed, turns=turns, budget=budget),
        raw=completed.stdout or "",
        summary=summary,
    )
    if completed.stderr:
        (directory / f"{RUNNER}.stderr.log").write_text(completed.stderr, encoding="utf-8")

    print(summary or "(the pass produced no final message)")
    print(f"\nrecord: {_shown(written)}  raw: {_shown(directory / f'{RUNNER}.raw.json')}")
    return completed.returncode


def _record(pass_id: str, completed, *, turns: int, budget: float) -> runner.PassRecord:
    """The Claude CLI's result, in the shape every runner reports.

    The CLI's own names (``num_turns``, ``total_cost_usd``, ``subtype``) differ
    from the SDK's ``ResultMessage`` fields and from whatever a raw loop
    computes. Translating here, once per runner, is what gives the three a
    common denominator. A field the CLI did not report stays None rather than
    becoming 0 -- a count nobody measured is worse than a cap nobody counted.
    """
    payload = _payload(completed.stdout)
    return runner.PassRecord(
        runner=RUNNER,
        pass_id=pass_id,
        exit_code=completed.returncode,
        turns=payload.get("num_turns"),
        turn_cap=turns,
        cost_usd=payload.get("total_cost_usd"),
        budget_usd=budget,
        duration_ms=payload.get("duration_ms"),
        stop_reason=payload.get("subtype"),
        extra={"permission_mode": PERMISSION_MODE, "settings": AGENT_SETTINGS},
    )


def _shown(path) -> str:
    """A path for a human: relative to the repo when it is inside it."""
    try:
        return str(path.relative_to(paths.ROOT))
    except ValueError:
        return str(path)


def _payload(stdout: str | None) -> dict:
    """The CLI's JSON result, or an empty dict if it did not produce one."""
    try:
        parsed = json.loads(stdout or "")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _final_message(stdout: str | None) -> str:
    """The pass's closing summary, falling back to raw output."""
    payload = _payload(stdout)
    result = payload.get("result")
    if isinstance(result, str) and result.strip():
        return result
    return (stdout or "").strip()


def main(argv: list[str] | None = None) -> int:
    """``make agent``. Options are make variables: ``TURNS=`` and ``BUDGET=``."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--turns", type=int, default=DEFAULT_TURNS)
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the command that would run and exit",
    )
    args = parser.parse_args(argv)
    return run(turns=args.turns, budget=args.budget, run_id=args.run_id, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
