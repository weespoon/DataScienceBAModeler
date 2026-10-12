"""Rule 1 and Rule 2 as a decision any runner can call.

These two rules were enforced in Claude-Code-shaped places. Rule 1 lived in
``.claude/hooks/protect_frozen.py``, a ``PreToolUse`` hook. Rule 2's file-access
half lived only in ``.claude/settings.json``'s deny list. A Claude Agent SDK
runner inherits both, because that SDK is Claude Code as a library. **A raw
Messages API loop inherits neither** — and Rule 2 is the tripwire that matters
most, because the final test can only be spent once.

So the decisions live here, and the hook became a shim that loads this file.
One copy, so the hook and a runner's own check cannot drift apart.

Two constraints this file has to keep, both easy to break by accident:

**Stdlib only, and Python 3.9-compatible.** The harness runs the hook with
whatever ``python3`` is on PATH — the system interpreter, not the project's
3.11+ virtualenv, which may not exist yet. ``from __future__ import
annotations`` is what makes the modern type syntax below safe there. The hook
loads this module *by path*, never as ``titanic_agent.guard``, so
``__init__.py`` never runs and the constraint is a property of this one file
rather than of the whole package's import graph. That matters because the hook
fails **open**: it exits 0 on malformed input, so an ImportError would leave
Rule 1 unenforced with no visible signal.

**Root is a parameter, never imported.** The hook computes the repo root from
``.claude/hooks/../..``; ``paths.ROOT`` computes it from
``src/titanic_agent/../..``. Same answer today, two ways to be wrong later.

Prevention here is deliberately partial. For ``Write``/``Edit`` the check is
exact: a resolved path either is under ``data/frozen/`` or is not. For ``Bash``
there is only a command string, so it is a heuristic a determined caller can
evade — base64, a here-doc, a helper script. That is why the repo also detects
(checksums, the hash chain, receipts) and attests (``holdout_read`` on every
ledger row). Prevent, detect, attest, in descending order of strength, each
covering what the last one cannot.
"""

from __future__ import annotations

import re
from pathlib import Path

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
    # the over-broad-pattern mistake this repo keeps making, and it cost two
    # reviewers a read before anyone noticed. So: not a file descriptor
    # duplication (`2>&1`), and not a sink under /dev/.
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

#: Paths no runner may write, relative to the repo root, as glob patterns.
#: ``data/frozen`` is Rule 1. The rest are the append-only evidence files: they
#: are written through ``results_log.append_run`` and friends, which chain and
#: lock them, and a direct write is how a hash chain silently stops meaning
#: anything. A test asserts this list against ``.claude/settings.json``.
WRITE_DENIED: tuple[str, ...] = (
    "data/frozen/**",
    "experiments/results.csv",
    "experiments/findings.jsonl",
    "experiments/run_log.jsonl",
    "experiments/champion.jsonl",
    "experiments/holdout_reads.json",
    "experiments/val_reads.jsonl",
)

#: Paths no runner may read. This is Rule 2's file-access half, and it existed
#: only in ``.claude/settings.json`` until now.
#:
#: ``data/raw`` is here for a reason that is easy to miss: it holds all 891
#: labelled rows, 179 of which *are* the final test. Reading it is a holdout
#: read by another name, and it would leave no receipt.
READ_DENIED: tuple[str, ...] = (
    "data/frozen/holdout.csv",
    "data/raw/**",
)


def _strip_prose(command: str) -> str:
    """Remove text that is a *message* rather than a command target.

    Heredoc bodies and ``-m``/``--message`` arguments carry prose: a commit
    message explaining Rule 1, a journal entry quoting it. Checking those for
    mutating verbs flags the explanation as the offence.
    """
    cleaned = MESSAGE_ARG.sub(" ", command)
    for marker in HEREDOC.findall(cleaned):
        start = cleaned.find("<<")
        end = cleaned.find(marker, cleaned.find(marker) + len(marker))
        if start != -1 and end != -1:
            cleaned = cleaned[:start] + " " + cleaned[end + len(marker) :]
    return cleaned


def _relative(raw_path: str, root: Path) -> Path | None:
    """``raw_path`` as a path relative to ``root``, or None if it escapes it."""
    if not raw_path:
        return None
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve()
        base = root.resolve()
    except (OSError, RuntimeError):
        return None
    try:
        return resolved.relative_to(base)
    except ValueError:
        return None


def targets_frozen(raw_path: str, root: Path) -> bool:
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


def _matches(relative: Path, patterns: tuple[str, ...]) -> str | None:
    """The first pattern ``relative`` matches, or None."""
    text = relative.as_posix()
    for pattern in patterns:
        if pattern.endswith("/**"):
            prefix = pattern[:-3]
            if text == prefix or text.startswith(prefix + "/"):
                return pattern
        elif text == pattern:
            return pattern
    return None


def write_refusal(raw_path: str, root: Path) -> str | None:
    """Why a write to this path is refused, or None to allow.

    Rule 1 plus the append-only evidence files. Unlike :func:`decide` this does
    not consult ``is_frozen``: the evidence files are append-only from the
    moment they exist, and a caller that wants Rule 1's freeze-once exemption
    should use :func:`decide`.
    """
    relative = _relative(raw_path, root)
    if relative is None:
        return None
    pattern = _matches(relative, WRITE_DENIED)
    if pattern is None:
        return None
    if pattern.startswith("data/frozen"):
        return f"Rule 1: {relative} is under data/frozen, which is immutable once frozen"
    return (
        f"Rule 3: {relative} is append-only. Write it through its module "
        "(results_log.append_run, findings.record, run_log.record) so it stays "
        "chained and locked; a direct write is how a hash chain stops meaning anything"
    )


def read_refusal(raw_path: str, root: Path) -> str | None:
    """Why a read of this path is refused, or None to allow. Rule 2.

    The second tripwire, and the one that cannot be undone: the final test gives
    one honest estimate of generalisation, once per frozen dataset. Only
    ``ensemble --score-holdout`` may spend it, and that is a human's call.
    """
    relative = _relative(raw_path, root)
    if relative is None:
        return None
    pattern = _matches(relative, READ_DENIED)
    if pattern is None:
        return None
    if pattern.startswith("data/raw"):
        return (
            f"Rule 2: {relative} is the raw data -- all 891 labelled rows, 179 of which "
            "are the final test. Reading it is a holdout read by another name, and it "
            "would leave no receipt"
        )
    return (
        f"Rule 2: {relative} is the final test. It is read once per frozen dataset, by "
        "ensemble --score-holdout, and that is a human's call"
    )


def decide(payload: dict, *, is_frozen: bool, root: Path) -> str | None:
    """Return a deny reason for one tool call, or None to allow. Rule 1.

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
