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


# --------------------------------------------------------------------------- #
# what a runner may run
# --------------------------------------------------------------------------- #


#: Every target, and whether an autonomous pass may run it. Two buckets, not
#: three: a "read" / "write" distinction would be decoration, because nothing
#: branches on it — both are allowed. What is load-bearing is allowed vs not,
#: and *why*, because the reason is what a reviewer checks and what
#: :func:`check` quotes back when it refuses.
#:
#: A target missing from here fails :func:`unclassified`, and therefore the test
#: suite. That is the only mechanism in this module that survives someone
#: forgetting: a new target is unavailable to a runner until it is classified.
ALLOWED: dict[str, str] = {
    # preflight and reading the record
    "status": "where the pipeline is, from artifacts on disk",
    "next": "the next incomplete step",
    "verify": "the frozen data still matches its manifest",
    "audit": "did the rules hold",
    "leakage": "Rule 4's structural guards",
    "receipts": "has the final test been spent",
    "log": "the append-only ledger, every run ever logged",
    "journal": "the journal: decisions, refusals and step outcomes",
    "runs": "every pass and what it left",
    "experiments": "every experiment and its scores",
    "findings": "what has been learned",
    "backlog": "the untried ideas a pass picks from",
    "champion": "the persisted champion and its corrections",
    "introspect": "our search spaces vs the installed sklearn",
    "help": "the target list, with the options each one takes",
    # the experiment lifecycle
    "experiment-open": "open the one experiment this pass runs",
    "experiment-lock-plan": "lock plan.md before anything is scored",
    "experiment-evaluate": "the one validation read, refused a second time",
    "experiment-close": "close with a verdict",
    # searches, all of which score on train rows only
    "tune": "hyperparameter search on the experiment's train rows",
    "features": "greedy add/prune against one fixed model",
    "nested": "measure a feature search's selection bias",
    "bakeoff": "arms paired against the champion",
    "bakeoff-decide": "apply the declared rule, crown or hold",
    "permutation": "the leak control for label-reading features",
    # writing the record
    "note": "append a journal entry without needing an interpreter",
    "finding": "append a finding without needing an interpreter",
    "summary": "this pass's run.md, derived from the ledger",
    "report": "this pass in one page",
    # housekeeping a pass may legitimately need
    "setup": "create the venv; a no-op once installed",
    "test": "the suite; CLAUDE.md requires it green before any commit",
    "lint": "ruff check and format, the style gate before a commit",
    "submit": "a Kaggle submission from the champion; never reads the final test",
}

#: Targets an autonomous pass must not run, and why. `check` refuses these by
#: name and quotes the reason, so a runner with no permission system of its own
#: still stops here.
FORBIDDEN: dict[str, str] = {
    "ensemble": (
        "the only module that can read the final test. --score-holdout is a "
        "human's call, once per frozen dataset (Rule 2)"
    ),
    "freeze": "re-freezing invalidates every score already in the ledger (Rule 1)",
    "pull": "downloads raw data; part of the one-time bootstrap, not a pass",
    "review": "part of the one-time bootstrap",
    "split": "re-drawing the split orphans every logged row (Rule 1)",
    "bootstrap": "runs split and freeze; one-time, and a human's call",
    "explore": (
        "charts all 712 working rows, including the open experiment's "
        "validation rows -- a val-split leak through visualisation"
    ),
    "loop": "the old step loop; a pass is one experiment, not a lap of this",
    "clean": "deletes generated reports, including this pass's own evidence",
    "distclean": "also removes the virtualenv",
    "agent": "a runner does not launch a runner; that is how a retry loop starts",
}

#: Targets that are allowed but deliberately absent from Claude Code's
#: allowlist in ``.claude/settings.json``, with the reason. Asserting the two
#: are *equal* would be a rule relaxed within a week; asserting the allowlist is
#: a subset, and that every gap is named here, is a rule that holds.
NOT_ALLOWLISTED: dict[str, str] = {}


def unclassified(makefile: Path | None = None) -> set[str]:
    """Targets in the Makefile that nobody has decided about yet."""
    return make_targets(makefile) - set(ALLOWED) - set(FORBIDDEN)


# --------------------------------------------------------------------------- #
# what a runner may pass to it
# --------------------------------------------------------------------------- #


#: Variables a target legitimately takes, and what counts as a value. The
#: Makefile enforces the same rules with ``override``, ``$(filter)`` and the
#: ``safe``/``numeric`` helpers, and that is the layer that actually prevents:
#: it protects a human at a terminal, and it holds even for a runner that never
#: calls this function. This copy exists so a runner can **refuse and say why**
#: before invoking make, instead of watching make silently empty a value and
#: then reading an argparse error.
#:
#: The decisive case is a variable that is not here at all. ``PYTHON=`` and
#: ``SHELL=`` are not options a target offers; they are recipe machinery, and a
#: command-line assignment used to beat the Makefile's ``:=``. So an unknown
#: variable name is refused outright rather than pattern-matched.
ENUMS: dict[str, tuple[str, ...]] = {
    "MODEL": (
        "all",
        "dummy",
        "logistic",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "hist_gradient_boosting",
        "svc",
        "knn",
    ),
    "ARM_SET": ("bakeoff", "decompose", "simplify"),
    "METHOD": ("soft_vote", "hard_vote", "stack"),
    "LEVEL": ("debug", "info", "decision", "error"),
    "EVENT": (
        "step_started",
        "step_finished",
        "step_skipped",
        "step_failed",
        "decision",
        "refused",
        "note",
    ),
    "STATUS": ("confirmed", "refuted", "inconclusive", "untried"),
    "CONFIDENCE": ("low", "medium", "high"),
}

#: Numbers. Matched, not eval'd.
NUMERIC: frozenset[str] = frozenset(
    {"SEED", "CV_FOLDS", "N_ITER", "TOP_K", "OUTER_FOLDS", "MAX_TESTS", "TURNS", "BUDGET", "DELTA"}
)

#: Identifiers and comma-separated lists of them: run ids, experiment ids, arm
#: names, step names, ledger row ids.
IDENTIFIER: frozenset[str] = frozenset(
    {"EXPERIMENT", "ONLY", "PASS", "RUN", "STEP", "EVIDENCE", "ARM", "EXTRAS", "ALL", "UNTUNED"}
)

#: Free text. These reach the process through the environment and never touch a
#: shell, which is the only reason they are allowed to contain anything: a
#: closing summary legitimately contains a semicolon.
PROSE: frozenset[str] = frozenset({"NAME", "SUMMARY", "DETAIL", "CLAIM", "NOTE"})

IDENT = re.compile(r"^[A-Za-z0-9_.,/-]+$")
NUMBER = re.compile(r"^-?[0-9]+(\.[0-9]+)?$")


class Refused(Exception):
    """A runner asked for something outside the surface. Carries the reason."""


def check(target: str, variables: dict[str, str] | None = None) -> None:
    """Refuse anything outside the declared surface. Returns None, or raises.

    The one function both runners call, and the whole of the enforcement for a
    runner that has no permission system of its own — a raw Messages API loop
    inherits neither ``.claude/settings.json`` nor the Rule 1 hook.

    Raises:
        Refused: Unknown target, forbidden target, unknown variable, or a value
            outside what that variable accepts.
    """
    if target in FORBIDDEN:
        raise Refused(f"`make {target}` is not a pass's to run: {FORBIDDEN[target]}")
    if target not in ALLOWED:
        known = target in make_targets()
        raise Refused(
            f"`make {target}` is not classified in titanic_agent.surface"
            if known
            else f"there is no `make {target}` target"
        )

    for name, value in (variables or {}).items():
        if name in PROSE:
            continue  # travels in the environment; no shell sees it
        if name in ENUMS:
            if value not in ENUMS[name]:
                raise Refused(f"{name}={value!r} is not one of {ENUMS[name]}")
        elif name in NUMERIC:
            if not NUMBER.match(value):
                raise Refused(f"{name}={value!r} is not a number")
        elif name in IDENTIFIER:
            if not IDENT.match(value):
                raise Refused(
                    f"{name}={value!r} is not an identifier; a shell metacharacter here "
                    "used to reach /bin/sh"
                )
        else:
            raise Refused(
                f"{name} is not a variable any target offers. PYTHON, BIN, VENV and SHELL "
                "are recipe machinery, and a command-line assignment to one of them used "
                "to turn every allowed target into arbitrary execution"
            )


def tool_schema() -> dict[str, object]:
    """The surface as a JSON-schema tool definition, for a runner that needs one.

    A raw Messages API loop should be given *this* and no Bash tool at all.
    Then a compound command, ``make -f``, ``make -C``, a ``$(...)`` expansion
    and ``SHELL=`` are **unrepresentable** rather than regex-blocked — the
    model cannot express them, so there is nothing to filter. That is a
    stronger boundary than any validator bolted onto a string channel, which is
    what the Claude Code allowlist is and why a variable assignment walked past
    it.
    """
    return {
        "name": "run_make",
        "description": (
            "Run one make target in the Titanic repo. This is the only way to act on "
            "the repo. Read .claude/skills/run/SKILL.md for what a pass does."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": sorted(ALLOWED),
                    "description": "The target to run.",
                },
                "variables": {
                    "type": "object",
                    "description": (
                        "Options as make variables, never flags. Only the variables a "
                        "target offers are accepted."
                    ),
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["target"],
            "additionalProperties": False,
        },
    }
