"""The agent-facing surface: what a runner is allowed to ask this repo to do.

Today one runner drives this repo — Claude Code, through `make agent`. The
constraints that make it trustworthy all live in Claude-Code-shaped places:
an allow/deny list in ``.claude/settings.json``, an interpreter lockdown in
``.claude/agent-settings.json``, and Rule 1 in a ``PreToolUse`` hook. A second
runner built on the Claude Agent SDK inherits all three, because that SDK is
Claude Code as a library. A raw Messages API loop inherits **none** of them.

So the surface has to be declared somewhere both runners can reach, which is
here. Comparing two runners is only a comparison of *runners* if they are
constrained identically; otherwise it is a comparison of tool access wearing a
lab coat.

This module starts with the part that was already duplicated: parsing the
Makefile. That lived in ``tests/test_documented_commands.py`` and
``tests/test_permissions.py`` imported it with ``from tests.… import …``, which
resolves only when the working directory happens to be on ``sys.path``. Under
``make test`` — which runs ``$(BIN)/pytest``, not ``python -m pytest`` — it does
not, so the test guaranteeing that every target the run skill names is
allowlisted had been failing silently since it was written. A shared helper in
the package is the fix, and it is the strongest argument for this module
existing at all.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import paths

#: `make foo` — only ever matched inside a code span or fenced block. The first
#: version of this matched bare prose and flagged "make it", "make an" and
#: "make cheating impossible": the same over-broad-pattern mistake the Rule 1
#: hook and the explore holdout test each made. A checker that cries wolf is one
#: nobody runs.
MAKE_COMMAND = re.compile(r"\bmake\s+([a-z][a-z0-9-]*)")

#: Inline code spans and fenced blocks — the only places a command can live.
CODE_SPAN = re.compile(r"`([^`\n]+)`")
CODE_BLOCK = re.compile(r"```[a-z]*\n(.*?)```", re.DOTALL)

#: `make foo --bar` — the shape that silently does not work, because make
#: consumes the flag itself and the target never sees it.
MAKE_WITH_FLAG = re.compile(r"\bmake\s+[a-z][a-z0-9-]*\s+(--[a-z][a-z0-9-]*)")

#: Variable assignments are fine: `make tune MODEL=svc` works.
MAKE_WITH_VARIABLE = re.compile(r"\bmake\s+[a-z][a-z0-9-]*\s+[A-Z_]+=")

#: A target definition at the start of a line. Deliberately loose about what
#: follows the colon, so `loop: export TITANIC_PASS_ID := …` still matches.
TARGET = re.compile(r"^([a-z][a-z0-9-]*):", re.MULTILINE)


def makefile_path() -> Path:
    """The Makefile that defines the surface."""
    return paths.ROOT / "Makefile"


def make_targets(makefile: Path | None = None) -> set[str]:
    """Every target defined in the Makefile."""
    text = (makefile or makefile_path()).read_text(encoding="utf-8")
    return {match.group(1) for match in TARGET.finditer(text)}


def code_fragments(text: str) -> list[str]:
    """Every inline code span and fenced block in a document."""
    return CODE_SPAN.findall(text) + CODE_BLOCK.findall(text)


def targets_named_in(text: str) -> set[str]:
    """Make targets a document actually instructs someone to run.

    Confined to code fragments, and intersected with the real targets by the
    caller when that matters — a document may legitimately mention a target
    that was renamed, and this function is not the place to decide.
    """
    return {
        match.group(1)
        for fragment in code_fragments(text)
        for match in MAKE_COMMAND.finditer(fragment)
    }


def policy_path() -> Path:
    """The policy a pass follows, as a file any runner can load as a prompt.

    It lives under ``.claude/`` because that is where Claude Code discovers
    skills, not because the policy is Claude-Code-specific. Returned from here
    so a second runner does not hardcode the path and then drift from it.
    """
    return paths.ROOT / ".claude" / "skills" / "run" / "SKILL.md"
