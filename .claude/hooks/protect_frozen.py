#!/usr/bin/env python3
"""PreToolUse hook: enforce Rule 1 in the harness, not just in the docs.

Rule 1 says ``data/frozen/`` is immutable after ``freeze``. Until this hook
existed, that rule was enforced only by documentation plus ``chmod 0444`` — and
0444 does not stop the file's owner, who can simply ``chmod u+w``. The rule was
a speed bump, not a constraint.

The design question this forces is the interesting one: **how does the
authorized writer get through a blanket deny?** ``freeze`` itself has to write
these files exactly once.

The answer here: deny only once ``MANIFEST.json`` exists. The hook reads the
same signal ``freeze.is_frozen()`` reads, so the initial freeze passes and every
later write is refused, and the invariant lives in one place rather than being
restated in two that can drift apart. The rejected alternative was allowlisting
a command pattern (``titanic_agent.freeze``), which anything could claim by
renaming itself.

**This is deliberately imperfect, and the imperfection is the lesson.** For
``Write``/``Edit`` the check is exact: a resolved path either is under
``data/frozen/`` or is not. For ``Bash`` there is no structured target, only a
command string, so the check is a heuristic and a determined caller can evade it
(base64, a here-doc, a helper script, an unusual tool). Prevention is therefore
partial — which is exactly why the repo also detects (checksums, receipts) and
attests (the ``holdout_read`` ledger field). Prevent, detect, attest, in
descending order of strength, each covering what the last one cannot.

Exits 0 always; the decision travels in the JSON on stdout. A hook that crashes
should not be able to wedge the session.

**Keep this file compatible with Python 3.9 and dependency-free.** The harness
runs it with whatever ``python3`` is on PATH — the system interpreter, not the
project's 3.11+ virtualenv, which may not even exist yet. ``from __future__
import annotations`` is what makes the modern type syntax below safe there.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

#: Repo root, from this file's location: .claude/hooks/protect_frozen.py
ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "data" / "frozen"
MANIFEST = FROZEN / "MANIFEST.json"

#: Tools whose target is a structured file path we can check exactly.
PATH_TOOLS = ("Write", "Edit", "MultiEdit", "NotebookEdit")

#: Shell tokens that indicate intent to modify rather than read. Checked only
#: when the command also mentions data/frozen, so `rm -rf build` is unaffected.
MUTATING = re.compile(
    r"""(
      \brm\b | \bmv\b | \bcp\b | \bchmod\b | \bchown\b | \bchflags\b
    | \btruncate\b | \bdd\b | \btee\b | \btouch\b | \bln\b
    | \bunlink\b | \brmdir\b | \bmkdir\b | \binstall\b | \bshred\b
    | \bsed\b\s+(-[a-zA-Z]*\s+)*-i | \bperl\b.*\s-i
    # A redirect only mutates when it has a real target. This was a bare
    # `>> | >`, which denied any command mentioning data/frozen that happened
    # to contain a redirect -- `ls data/frozen 2>/dev/null` and
    # `cat data/frozen/MANIFEST.json 2>&1` were both refused as writes. That is
    # the over-broad-pattern mistake this file's own docstring warns about, and
    # it cost two reviewers a read before anyone noticed. So: not a file
    # descriptor duplication (`2>&1`), and not a sink under /dev/.
    | >>?\s*(?!&)(?!/dev/)\S
    )""",
    re.VERBOSE,
)

#: How the frozen directory might be named in a shell command.
MENTIONS_FROZEN = re.compile(r"data/frozen|data\\frozen")

#: Heredoc introducer, e.g. ``<<'EOF'`` or ``<<-EOF``.
HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?")

#: Message arguments: text that *describes*, never a path that gets mutated.
MESSAGE_ARG = re.compile(
    r"""(?:-m|--message)(?:=|\s+)(?:'[^']*'|"[^"]*"|\S+)""",
    re.VERBOSE,
)

#: git commands that delete *ignored* files, and so can wipe data/frozen without
#: ever naming it. Everything under data/frozen is gitignored, which is exactly
#: why no ordinary git command can touch it — and why these two can.
DANGEROUS_GIT = re.compile(
    r"""(
      \bgit\b[^|;&]*\bclean\b[^|;&]*-[a-zA-Z]*[xX]
    | \bgit\b[^|;&]*\bstash\b[^|;&]*(--all|\s-\w*a)
    )""",
    re.VERBOSE,
)


def _strip_prose(command: str) -> str:
    """Remove text that is a *message* rather than a command target.

    Heredoc bodies and ``-m``/``--message`` arguments carry prose: a commit
    message describing this very rule should not be read as an attempt to break
    it. Learned the hard way — this hook denied the commit that introduced it,
    and the tempting fix was to reword the message rather than the pattern.

    Quoted strings in general are *not* stripped: ``rm -rf "data/frozen"`` is a
    real attempt and must still be caught.
    """
    # Heredoc bodies: drop everything between the introducer's newline and the
    # terminator line.
    while (match := HEREDOC.search(command)) is not None:
        delimiter = match.group(1)
        body_start = command.find("\n", match.end())
        if body_start == -1:
            command = command[: match.start()]
            break
        remainder = command[body_start:]
        terminator = re.search(rf"^{re.escape(delimiter)}\s*$", remainder, re.MULTILINE)
        if terminator is None:
            command = command[:body_start]
            break
        command = command[: match.start()] + remainder[terminator.end() :]

    return MESSAGE_ARG.sub(" ", command)


DENY_REASON = (
    "Rule 1 (CLAUDE.md): data/frozen/ is immutable once MANIFEST.json exists. "
    "This write was denied by the PreToolUse hook, not by a convention. "
    "Re-freezing invalidates every score already in experiments/results.csv, "
    "because those rows carry the old frozen_sha256. "
    "Unfreezing is a human action: stop, explain why you believe the data must "
    "change, and wait. Do not look for a way around this hook."
)


def allow() -> None:
    """Say nothing and let the tool run."""
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse"}}))
    sys.exit(0)


def deny(detail: str) -> None:
    """Refuse the tool call, with a reason the agent will read."""
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"{DENY_REASON}\n\nBlocked: {detail}",
                },
                "systemMessage": f"Rule 1 enforced: blocked a write to data/frozen/ ({detail})",
            }
        )
    )
    sys.exit(0)


def targets_frozen(raw_path: str, root: Path = ROOT) -> bool:
    """True when ``raw_path`` resolves to somewhere inside ``<root>/data/frozen/``."""
    if not raw_path:
        return False
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    frozen = root / "data" / "frozen"
    try:
        resolved = candidate.resolve()
        # Resolve the frozen dir too, so a symlink pointing into it is caught.
        frozen = frozen.resolve()
    except (OSError, RuntimeError):
        return False
    return resolved == frozen or frozen in resolved.parents


def decide(payload: dict, *, is_frozen: bool, root: Path = ROOT) -> str | None:
    """Return a deny reason, or None to allow.

    Pure: every input is an argument, so the policy can be tested without a
    frozen dataset on disk and without an environment-variable override — an
    override would itself be a way around the hook.
    """
    # The gate: before the first freeze there is nothing to protect, and the
    # authorized writer has to get through. This is the same signal
    # freeze.is_frozen() reads, by design — one invariant, one place.
    if not is_frozen:
        return None

    tool = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}

    if tool in PATH_TOOLS:
        path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
        if targets_frozen(str(path), root):
            return f"{tool} -> {path}"
        return None

    if tool == "Bash":
        raw = str(tool_input.get("command") or "")
        command = _strip_prose(raw)
        if DANGEROUS_GIT.search(command):
            return f"Bash (deletes ignored files, which is all of data/frozen) -> {raw[:160]}"
        if MENTIONS_FROZEN.search(command) and MUTATING.search(command):
            return f"Bash -> {raw[:160]}"
        return None

    return None


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        allow()  # Malformed input is not evidence of a violation.

    detail = decide(payload, is_frozen=MANIFEST.is_file(), root=ROOT)
    deny(detail) if detail else allow()


if __name__ == "__main__":
    main()
