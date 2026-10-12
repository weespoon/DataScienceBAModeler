"""The declared surface: what a runner may run, and what it may read or write.

The point of this module is that two runners are constrained *identically*.
Claude Code gets its constraints from ``.claude/settings.json`` and a
``PreToolUse`` hook; a Claude Agent SDK runner inherits both; a raw Messages API
loop inherits neither. So the declaration lives in the package, and these tests
assert that Claude Code's copy agrees with it. If they ever disagree, comparing
the two runners stops being a comparison of runners.
"""

from __future__ import annotations

import json
import re

import pytest

from titanic_agent import guard, paths, surface

SETTINGS = paths.ROOT / ".claude" / "settings.json"


def rules(kind: str) -> list[str]:
    payload = json.loads(SETTINGS.read_text(encoding="utf-8"))
    return payload.get("permissions", {}).get(kind, [])


def allowlisted_targets() -> set[str]:
    return {
        match.group(1)
        for rule in rules("allow")
        if (match := re.fullmatch(r"Bash\(make ([a-z][a-z0-9-]*):\*\)", rule))
    }


# --------------------------------------------------------------------------- #
# completeness: the one mechanism that survives someone forgetting
# --------------------------------------------------------------------------- #


def test_every_make_target_is_classified() -> None:
    """A new target is unavailable to a runner until someone decides about it.

    This is the anti-drift property, and it is a test rather than a convention
    because a convention is what the repo keeps discovering it did not follow.
    """
    missing = sorted(surface.unclassified())
    assert not missing, (
        f"Makefile has {missing} but titanic_agent.surface classifies neither. "
        "Add each to ALLOWED or FORBIDDEN with a reason."
    )


def test_nothing_is_both_allowed_and_forbidden() -> None:
    assert not set(surface.ALLOWED) & set(surface.FORBIDDEN)


def test_every_classification_carries_a_reason() -> None:
    # The reason is what a reviewer checks and what `check` quotes on refusal.
    for name, reason in {**surface.ALLOWED, **surface.FORBIDDEN}.items():
        assert len(reason) > 15, f"{name} has no real reason: {reason!r}"


# --------------------------------------------------------------------------- #
# the two tripwires
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("target", ["ensemble", "freeze", "split", "bootstrap", "explore"])
def test_the_tripwire_targets_are_refused(target: str) -> None:
    with pytest.raises(surface.Refused) as refusal:
        surface.check(target)
    # The refusal has to say why, or the next person relaxes it.
    assert len(str(refusal.value)) > 40


def test_ensemble_is_refused_because_it_is_the_route_to_the_final_test() -> None:
    assert "final test" in surface.FORBIDDEN["ensemble"]
    assert "Bash(make ensemble:*)" in rules("deny")
    assert "ensemble" not in allowlisted_targets()


def test_a_runner_cannot_launch_a_runner() -> None:
    # Nothing retries a pass; re-running one whose verdict you disliked is
    # selecting on the outcome.
    with pytest.raises(surface.Refused):
        surface.check("agent")


# --------------------------------------------------------------------------- #
# Claude Code's copy must agree with the declaration
# --------------------------------------------------------------------------- #


def test_the_allowlist_is_a_subset_of_what_is_allowed() -> None:
    """Subset, not equality.

    Equality would fight the deliberate exclusions this repo already has, and a
    rule that fights its own intent gets relaxed within a week. What must never
    happen is the other direction: a target Claude Code permits that the
    declaration forbids.
    """
    permitted_but_forbidden = allowlisted_targets() & set(surface.FORBIDDEN)
    assert not permitted_but_forbidden, (
        f".claude/settings.json allowlists {sorted(permitted_but_forbidden)}, which "
        "titanic_agent.surface forbids"
    )
    assert allowlisted_targets() <= set(surface.ALLOWED)


def test_every_allowed_target_absent_from_the_allowlist_is_named() -> None:
    gap = set(surface.ALLOWED) - allowlisted_targets()
    assert gap == set(surface.NOT_ALLOWLISTED), (
        f"{sorted(gap)} is allowed but not allowlisted for Claude Code. Either "
        "allowlist it or record why in surface.NOT_ALLOWLISTED."
    )


def test_the_path_rules_match_the_guard() -> None:
    """Rule 2's read deny and Rule 3's write deny, declared once.

    These existed only in ``.claude/settings.json``, which a raw Messages API
    loop never reads. ``guard`` owns them now; this asserts Claude Code's copy
    still says the same thing.
    """
    denied_reads = {r for r in rules("deny") if r.startswith("Read(")}
    for pattern in guard.READ_DENIED:
        expected = f"Read(./{pattern})"
        assert expected in denied_reads, f"{expected} missing from settings.json"

    denied_writes = {r for r in rules("deny") if r.startswith(("Write(", "Edit("))}
    for pattern in guard.WRITE_DENIED:
        for tool in ("Write", "Edit"):
            assert f"{tool}(./{pattern})" in denied_writes, f"{tool}(./{pattern}) missing"


# --------------------------------------------------------------------------- #
# values, not just target names
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("target", "variables"),
    [
        ("champion", {"PYTHON": "echo PWNED"}),
        ("audit", {"SHELL": "/bin/sh -c id"}),
        ("tune", {"MODEL": "all; echo PWNED"}),
        ("tune", {"N_ITER": "25; id"}),
        ("bakeoff", {"ARM_SET": "bakeoff; id"}),
        ("bakeoff", {"ONLY": "a; id"}),
        ("report", {"PASS": "$(id)"}),
        ("note", {"EVENT": "decision; id"}),
        ("finding", {"STATUS": "confirmed; id"}),
    ],
)
def test_a_value_that_could_reach_a_shell_is_refused(target: str, variables: dict) -> None:
    """A target-name allowlist is a label, not a boundary.

    ``make champion PYTHON='echo PWNED'`` was arbitrary execution: a
    command-line assignment beats the Makefile's ``:=``, so *every* allowlisted
    target was a shell of the caller's choosing. The Makefile now defends this
    with ``override`` and ``$(filter)``, which is the layer that prevents; this
    is the layer that refuses *and explains*, before make is ever invoked.
    """
    with pytest.raises(surface.Refused):
        surface.check(target, variables)


@pytest.mark.parametrize(
    ("target", "variables"),
    [
        ("tune", {"MODEL": "svc", "N_ITER": "40", "SEED": "20251010"}),
        ("bakeoff", {"ARM_SET": "decompose", "ONLY": "rf_wcg,rf_group"}),
        ("journal", {"LEVEL": "decision", "PASS": "20261012-0909EDT"}),
        (
            "finding",
            {"STATUS": "confirmed", "EVIDENCE": "20261011T191821-b81d88", "DELTA": "-0.003"},
        ),
        ("experiment-close", {"SUMMARY": 'held; "ties" is the honest word'}),
        ("experiment-open", {"NAME": "simplify the champion (take 2)"}),
        ("audit", None),
    ],
)
def test_legitimate_calls_are_allowed(target: str, variables: dict | None) -> None:
    # A guard that blocks the intended use is one someone removes. Prose keeps
    # its punctuation because it travels in the environment, never a shell.
    surface.check(target, variables)


def test_the_runner_caps_are_a_humans_variables_not_a_runners() -> None:
    """TURNS and BUDGET belong to `make agent`, which no runner may launch.

    So `check` never has to validate them; the Makefile's numeric guard does,
    and that is the layer protecting the human who types them. They stay in
    NUMERIC because the classification should describe the variable, not the
    current set of callers.
    """
    assert "TURNS" in surface.NUMERIC and "BUDGET" in surface.NUMERIC
    with pytest.raises(surface.Refused):
        surface.check("agent", {"TURNS": "60"})


def test_an_unknown_variable_is_refused_rather_than_pattern_matched() -> None:
    with pytest.raises(surface.Refused, match="recipe machinery"):
        surface.check("tune", {"LD_PRELOAD": "/tmp/x.so"})


def test_the_enum_sets_match_the_code_they_mirror() -> None:
    """A whitelist that has drifted from the thing it whitelists is worse than none."""
    from titanic_agent import bakeoff, ensemble, findings, run_log, tune

    assert set(surface.ENUMS["ARM_SET"]) == set(bakeoff.ARM_SETS)
    assert set(surface.ENUMS["MODEL"]) == set(tune.MODEL_KEYS) | {"all"}
    assert set(surface.ENUMS["METHOD"]) == set(ensemble.METHODS)
    assert set(surface.ENUMS["STATUS"]) == set(findings.STATUSES)
    assert set(surface.ENUMS["CONFIDENCE"]) == set(findings.CONFIDENCES)
    assert set(surface.ENUMS["EVENT"]) == set(run_log.EVENTS)
    assert set(surface.ENUMS["LEVEL"]) == set(run_log.LEVELS)


# --------------------------------------------------------------------------- #
# the tool schema a raw loop should be given instead of a shell
# --------------------------------------------------------------------------- #


def test_the_tool_schema_makes_a_shell_unrepresentable() -> None:
    """Stronger than a validator on a string channel.

    Given this schema and no Bash tool, a compound command, ``make -f``,
    ``make -C``, a ``$(...)`` expansion and ``SHELL=`` are things the model
    cannot express -- so there is nothing to filter. The Claude Code allowlist
    is a validator on a string channel, which is why a variable assignment
    walked past it.
    """
    schema = surface.tool_schema()
    properties = schema["input_schema"]["properties"]
    assert sorted(properties["target"]["enum"]) == sorted(surface.ALLOWED)
    assert not set(properties["target"]["enum"]) & set(surface.FORBIDDEN)
    assert schema["input_schema"]["additionalProperties"] is False
    # No free-text command field anywhere: that is the whole point.
    assert "command" not in properties


def test_the_policy_both_runners_load_is_one_file() -> None:
    assert surface.policy_path().is_file()
    assert surface.policy_path().name == "SKILL.md"
    text = surface.policy_path().read_text(encoding="utf-8")
    # Every target the policy tells a pass to run must be allowed, or a pass
    # following the policy stalls on a refusal.
    named = surface.targets_named_in(text) & surface.make_targets()
    forbidden = named & set(surface.FORBIDDEN)
    # explore and loop are named only to say *not* to run them.
    assert forbidden <= {"explore", "loop", "pull", "bootstrap"}, sorted(forbidden)
