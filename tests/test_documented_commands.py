"""Every ``make`` command written in a document must actually work.

This exists because the same mistake shipped twice in one session.

``make status --next`` was documented in the ``run`` skill, found not to work
(make reads ``--next`` as its own flag), fixed by adding a dedicated ``next``
target, and annotated with a comment explaining the trap. Then ``make journal``
was built with the identical flaw, ``make journal --decisions`` was written into
a handover message as a command to run, and nobody noticed until a fresh session
tried it.

A comment explaining a trap does not stop the trap. A test does.

Two rules, both mechanical:

1. Every ``make <target>`` named in a markdown file is a real target.
2. No document shows ``make <target> --flag``, because make will claim the flag
   for itself and the target will not see it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from titanic_agent import paths

#: The regexes and the Makefile parsing live in the package, not here, so a
#: second runner and the other test module can share them. `test_permissions`
#: imported them from this module with `from tests.… import …`, which resolves
#: only when the cwd is on sys.path -- not under `make test`.
from titanic_agent.surface import (  # noqa: E402
    MAKE_COMMAND,
    MAKE_WITH_FLAG,
    MAKE_WITH_VARIABLE,
    code_fragments,
    make_targets,
)

#: Documents that describe history rather than instruct. A past-tense account of
#: a command that used to exist must stay accurate to the past.
HISTORICAL: frozenset[str] = frozenset(
    {
        "UNFREEZING.md",
        "cold-start-001-predictions.md",
        "cold-start-001-result.md",
    }
)


def documented_files() -> list[Path]:
    """Every markdown file that instructs, excluding historical accounts."""
    roots = (paths.ROOT / ".claude", paths.ROOT / "docs", paths.ROOT)
    found: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        pattern = "*.md" if root == paths.ROOT else "**/*.md"
        for path in root.glob(pattern):
            if path.name in HISTORICAL or "transcripts" in path.parts:
                continue
            if "conversations" in path.parts:  # verbatim logs, not instructions
                continue
            found.add(path)
    return sorted(found)


def test_the_makefile_has_targets() -> None:
    # If this breaks, everything below passes vacuously.
    assert {"test", "lint", "loop", "audit"} <= make_targets()


def test_documents_exist_to_check() -> None:
    assert len(documented_files()) >= 5


@pytest.mark.parametrize("document", documented_files(), ids=lambda p: p.name)
def test_every_documented_make_target_exists(document: Path) -> None:
    targets = make_targets()
    text = document.read_text(encoding="utf-8")
    named = {
        m.group(1) for fragment in code_fragments(text) for m in MAKE_COMMAND.finditer(fragment)
    }
    missing = sorted(named - targets)
    assert not missing, (
        f"{document.relative_to(paths.ROOT)} documents `make {missing}` but no such target exists"
    )


@pytest.mark.parametrize("document", documented_files(), ids=lambda p: p.name)
def test_no_document_passes_a_flag_to_make(document: Path) -> None:
    # `make journal --decisions` looks reasonable and does nothing: make takes
    # the flag for itself. If a target needs an option, give it its own target
    # (as `next` is to `status --next`) or invoke the module directly.
    text = document.read_text(encoding="utf-8")
    offenders = [
        fragment.strip()
        for fragment in code_fragments(text)
        if MAKE_WITH_FLAG.search(fragment) and not MAKE_WITH_VARIABLE.search(fragment)
    ]
    assert not offenders, (
        f"{document.relative_to(paths.ROOT)} passes a flag to make, which make "
        f"will consume itself: {offenders}"
    )


def test_make_variables_are_still_allowed() -> None:
    # `make tune MODEL=svc` is fine and documented; the check must not ban it.
    assert MAKE_WITH_VARIABLE.search("make tune MODEL=svc")
    assert not MAKE_WITH_FLAG.search("make tune MODEL=svc")


def test_the_check_would_catch_the_bug_it_was_written_for() -> None:
    assert MAKE_WITH_FLAG.search("run `make journal --decisions` to see")
    assert MAKE_WITH_FLAG.search("make status --next")
