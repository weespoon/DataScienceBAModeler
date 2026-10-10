#!/usr/bin/env python3
"""Redact a Claude Code export before it goes into a public repository.

A session transcript is a verbatim log of a terminal, and a terminal accumulates
things that were never meant for an audience: an email address printed by a
tool, an employer's name, a discussion that happened to share a window with the
work. None of it is a credential leak. All of it ships the moment it is pushed.

Two principles shape what this does and does not do:

**Redaction is honest; quiet rewriting is not.** Every removal leaves a visible
``[redacted: reason]`` marker. A reader can always tell that something was taken
out, and roughly what kind of thing it was. Silently editing a transcript so it
reads better — or so past reasoning looks sounder than it was — destroys the one
thing these files exist for.

**It fails closed.** If anything on the configured list survives the pass, it
refuses to write the output rather than producing a file that looks clean and is
not. A redactor that half-works is worse than none, because it is trusted.

What it cannot do: find a passage that is private without containing any
sensitive *string*. Employment, health, finances, opinions about named people —
those need a human to read the thing. The output says so every time, because the
most dangerous moment is when an automatic tool reports success and the reader
stops looking.

Usage::

    python scripts/redact.py export.txt -o conversations/001-topic.md
    python scripts/redact.py export.txt --check    # report, write nothing
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

#: Private terms, one per line, ``#`` for comments. Gitignored on purpose: a
#: committed list of the things being hidden would publish them.
TERMS_FILE = Path(__file__).parent / "redact_terms.txt"

#: Patterns that are sensitive by shape rather than by content, so they need no
#: configuration and work on a machine that has never seen this project.
PATTERNS: tuple[tuple[str, str, str], ...] = (
    (
        "email",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "email",
    ),
    (
        "kaggle token",
        r"\bKGAT[A-Za-z0-9_-]{8,}\b",
        "credential",
    ),
    (
        "github token",
        r"\bgh[pousr]_[A-Za-z0-9]{16,}\b",
        "credential",
    ),
    (
        "bearer token",
        r"(?i)\b(?:bearer|authorization:)\s+[A-Za-z0-9._~+/-]{20,}=*",
        "credential",
    ),
    (
        "aws key",
        r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
        "credential",
    ),
    (
        "private key block",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        "credential",
    ),
)


@dataclass(frozen=True)
class Redaction:
    """One category of thing removed, and how often."""

    label: str
    reason: str
    count: int


def load_terms(path: Path | None = None) -> list[str]:
    """Private terms to remove, longest first so substrings do not shadow."""
    source = path or TERMS_FILE
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    terms = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    return sorted(terms, key=len, reverse=True)


def redact(text: str, *, terms: list[str] | None = None) -> tuple[str, list[Redaction]]:
    """Return the cleaned text and a summary of what was removed."""
    found: list[Redaction] = []

    for label, pattern, reason in PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            text = re.sub(pattern, f"[redacted: {reason}]", text)
            found.append(Redaction(label, reason, len(matches)))

    for term in terms if terms is not None else load_terms():
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        matches = pattern.findall(text)
        if matches:
            text = pattern.sub("[redacted]", text)
            found.append(Redaction(term[:3] + "…", "private term", len(matches)))

    return text, found


def verify(text: str, *, terms: list[str] | None = None) -> list[str]:
    """Anything still present that should not be. Empty means clean."""
    survivors: list[str] = []
    for label, pattern, _ in PATTERNS:
        if re.search(pattern, text):
            survivors.append(label)
    for term in terms if terms is not None else load_terms():
        if re.search(re.escape(term), text, re.IGNORECASE):
            survivors.append(f"private term {term[:3]}…")
    return survivors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="the export to clean")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument(
        "--check", action="store_true", help="report what would change; write nothing"
    )
    args = parser.parse_args(argv)

    try:
        original = args.source.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"cannot read {args.source}: {exc}")
        return 1

    terms = load_terms()
    if not terms:
        print(f"note: {TERMS_FILE.name} is empty or missing — only shape-based")
        print("      patterns (emails, tokens) will be removed. Add employer and")
        print("      account names to it if the transcript mentions any.")

    cleaned, found = redact(original, terms=terms)

    if not found:
        print("nothing matched.")
    for item in found:
        print(f"  {item.count:>3} × {item.label:<16} -> [redacted: {item.reason}]")

    survivors = verify(cleaned, terms=terms)
    if survivors:
        # Fail closed. A file that looks clean and is not is worse than none.
        print(f"\nREFUSING TO WRITE: {survivors} survived the pass.")
        return 1

    print(
        f"\n{len(original.splitlines())} lines checked, "
        f"{sum(i.count for i in found)} redaction(s)."
    )
    print()
    print("This removed sensitive *strings*. It cannot find a passage that is")
    print("private without containing one — employment, health, finances, or")
    print("opinions about named people. Read it before you push.")

    if args.check:
        print("\n--check: nothing written.")
        return 0
    if args.output is None:
        print("\nno --output given; nothing written.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(cleaned, encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
