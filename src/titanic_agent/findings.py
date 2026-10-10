"""What was *learned* — the thing that makes experiment N+1 smarter than N.

``results.csv`` records numbers: this model, these params, that score. Nothing in
it records what those numbers *meant*, so a second experiment reading only the
ledger would re-derive every conclusion from scratch, or worse, re-run every
dead end.

A finding is a **claim with evidence**, never a bare conclusion. The difference
matters more than it sounds: a conclusion handed to the next experiment is
folklore it cannot check, and five experiments later the agent is reasoning from
assertions nobody can trace. Every finding therefore carries the ``run_id``s
behind it and an honest confidence, so a later experiment can decide whether to
trust it, re-test it, or overturn it.

``untried`` findings are the other half. They are the backlog — a technique seen
in a notebook, a parameter left at its default, an idea nobody has tested — and
they are what the research step produces and the planning step consumes. The
same file is both knowledge base and agenda.

Storage follows the pattern the rest of the repo uses: a global append-only
``experiments/findings.jsonl`` is the evidence; per-experiment markdown is a
rendered view of it. One record, many views — two stores of the same claims can
disagree, and then neither can be trusted.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from . import locking, paths

#: What we know about a claim.
STATUSES: tuple[str, ...] = ("confirmed", "refuted", "inconclusive", "untried")

#: How much weight the next experiment should give it. ``low`` is the honest
#: default for anything measured once on one seed, which is most things here:
#: a 142-row validation split carries roughly ±4% noise, so a 0.004 difference
#: is not a result, it is a rounding error with ambition.
CONFIDENCES: tuple[str, ...] = ("low", "medium", "high")


@dataclass(frozen=True)
class Finding:
    """One claim, its evidence, and how much to trust it."""

    claim: str = ""
    status: str = "untried"
    experiment_id: str = ""
    delta: float | None = None
    evidence: list[str] = field(default_factory=list)
    confidence: str = "low"
    #: Where the idea came from: ``ours``, ``introspection``, ``notebook:<ref>``.
    #: Provenance matters when a later experiment wants to re-read the source.
    source: str = "ours"
    note: str = ""
    recorded_utc: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}; expected one of {STATUSES}")
        if self.confidence not in CONFIDENCES:
            raise ValueError(f"unknown confidence {self.confidence!r}")
        if self.status in ("confirmed", "refuted") and not self.evidence:
            # A claim asserted without a run behind it is exactly the folklore
            # this module exists to prevent.
            raise ValueError(f"a {self.status} finding needs evidence (run_ids): {self.claim!r}")


def record(finding: Finding, *, path: Path | None = None) -> Finding:
    """Append one finding. Append-only: a claim is never edited, only superseded."""
    target = Path(path) if path is not None else paths.FINDINGS
    target.parent.mkdir(parents=True, exist_ok=True)

    stamped = Finding(
        **{
            **asdict(finding),
            "recorded_utc": finding.recorded_utc
            or datetime.now(UTC).replace(microsecond=0).isoformat(),
        }
    )
    with locking.locked(target), target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(stamped), sort_keys=True, default=str) + "\n")
    return stamped


def read_findings(
    *, experiment_id: str | None = None, status: str | None = None, path: Path | None = None
) -> list[Finding]:
    """Every finding, oldest first, optionally filtered."""
    target = Path(path) if path is not None else paths.FINDINGS
    results: list[Finding] = []
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return results

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue  # a run killed mid-write; everything before it still counts
        try:
            finding = Finding(**payload)
        except (TypeError, ValueError):
            continue
        if experiment_id and finding.experiment_id != experiment_id:
            continue
        if status and finding.status != status:
            continue
        results.append(finding)
    return results


def backlog(path: Path | None = None) -> list[Finding]:
    """Untried ideas — what the planning step should pick from.

    Deliberately not deduplicated against confirmed findings: an idea tried once
    at low confidence is still worth retrying, and silently dropping it would
    hide that choice rather than make it.
    """
    return read_findings(status="untried", path=path)


def render_markdown(experiment_id: str | None = None, *, path: Path | None = None) -> str:
    """A human view of the findings, grouped by what we know about them."""
    found = read_findings(experiment_id=experiment_id, path=path)
    scope = f"experiment `{experiment_id}`" if experiment_id else "every experiment"

    lines = [
        "# Findings",
        "",
        f"Claims with evidence, from {scope}. Rendered from the append-only",
        "`experiments/findings.jsonl` — edit that through `findings.record()`,",
        "never this file.",
        "",
    ]

    for status in STATUSES:
        group = [f for f in found if f.status == status]
        if not group:
            continue
        lines += [f"## {status.title()} ({len(group)})", ""]
        for finding in group:
            delta = "" if finding.delta is None else f"**{finding.delta:+.4f}** "
            evidence = (
                f" — evidence: {', '.join(f'`{r}`' for r in finding.evidence)}"
                if finding.evidence
                else ""
            )
            source = f" — source: {finding.source}" if finding.source != "ours" else ""
            lines.append(
                f"- {delta}{finding.claim} _(confidence: {finding.confidence})_{evidence}{source}"
            )
            if finding.note:
                lines.append(f"  - {finding.note}")
        lines.append("")

    if not found:
        lines += ["_Nothing recorded yet._", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """``make findings`` — print what we know, and what is still untried."""
    import argparse

    parser = argparse.ArgumentParser(description="Show findings.")
    parser.add_argument("--experiment", default=None, help="filter to one experiment")
    parser.add_argument("--status", default=None, choices=STATUSES)
    parser.add_argument("--backlog", action="store_true", help="only untried ideas")
    args = parser.parse_args(argv)

    if args.backlog:
        items = backlog()
        if not items:
            print("backlog is empty — nothing untried has been recorded yet")
            return 0
        for finding in items:
            print(f"  [{finding.source}] {finding.claim}")
        return 0

    print(render_markdown(args.experiment) if not args.status else "", end="")
    if args.status:
        for finding in read_findings(experiment_id=args.experiment, status=args.status):
            print(f"  {finding.claim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
