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

import importlib.util
import json
import sys
from pathlib import Path

#: Repo root, from this file's location: .claude/hooks/protect_frozen.py
ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "data" / "frozen"
MANIFEST = FROZEN / "MANIFEST.json"


def _load_guard():
    """Load ``src/titanic_agent/guard.py`` by path, not as a package import.

    By path on purpose. ``import titanic_agent.guard`` would execute
    ``titanic_agent/__init__.py`` first, and the day someone adds a convenience
    import there — ``from . import paths`` reaching pandas, say — this hook
    would raise under the system interpreter. It fails **open** (``main`` calls
    ``allow()`` on bad input and always exits 0), so Rule 1 would quietly stop
    being enforced with nothing to notice it. Loading one stdlib-only file keeps
    the constraint a property of that file.
    """
    source = ROOT / "src" / "titanic_agent" / "guard.py"
    spec = importlib.util.spec_from_file_location("_titanic_guard", source)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load the Rule 1 policy from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


_guard = _load_guard()
MUTATING = _guard.MUTATING


def decide(payload: dict, *, is_frozen: bool, root: Path = ROOT) -> str | None:
    """This hook's root, applied to the shared policy in ``guard.decide``.

    ``guard.decide`` takes ``root`` explicitly and on purpose: this file derives
    it from ``.claude/hooks/../..`` while ``paths.ROOT`` derives it from
    ``src/titanic_agent/../..``. Same answer today, two ways to be wrong later.
    The default belongs here, where the root is known, not there.
    """
    return _guard.decide(payload, is_frozen=is_frozen, root=root)


def targets_frozen(raw_path: str, root: Path = ROOT) -> bool:
    """See :func:`guard.targets_frozen`."""
    return _guard.targets_frozen(raw_path, root)


def read_refusal(raw_path: str, root: Path = ROOT) -> str | None:
    """Rule 2. See :func:`guard.read_refusal`."""
    return _guard.read_refusal(raw_path, root)


def write_refusal(raw_path: str, root: Path = ROOT) -> str | None:
    """Rule 1 and Rule 3. See :func:`guard.write_refusal`."""
    return _guard.write_refusal(raw_path, root)


DANGEROUS_GIT = _guard.DANGEROUS_GIT
PATH_TOOLS = _guard.PATH_TOOLS


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        allow()  # Malformed input is not evidence of a violation.

    detail = decide(payload, is_frozen=MANIFEST.is_file(), root=ROOT)
    deny(detail) if detail else allow()


if __name__ == "__main__":
    main()
