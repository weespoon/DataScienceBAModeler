"""``make report`` — one page per pass, for a human who was not watching.

Every source this reads already prints something: ``make status``, ``make
audit``, ``make leakage``, ``make receipts``, ``make champion``, ``make journal
LEVEL=decision``. Reading six outputs and holding them together is work, and it
is work a person does *after* an unattended pass, when the question is narrow:
did it do one honest experiment, and did the rules hold?

**Derived, never copied.** Each section calls the module that owns the fact —
:func:`status.pipeline_status`, :func:`audit.audit`, :func:`leakage.check_all`,
:func:`experiments.load_experiment`, :func:`champion.current`. Nothing is
restated here, because two records of the same thing can disagree and then
neither can be trusted. That is the same reason ``runs.summarize`` derives
``run.md`` from the global ledger rather than keeping its own copy.

It complements ``run.md`` rather than replacing it. ``run.md`` is the pass's
scores and events; this is the pass's *verdict* plus the state of the rules
around it. In particular this one charges no budget: ``runs.summarize`` is the
thing that runs exactly once per pass and so is the honest counter, and a second
counter would double-charge a cap that is supposed to bind.

What it will not do is grade the pass. A verdict of *inconclusive* is printed as
inconclusive, a failed audit check is printed as failed, and an empty receipts
file is printed as empty. A report that scored its own pass would be the pass
marking its own homework.
"""

from __future__ import annotations

from pathlib import Path

from . import paths

#: Verdict stems, in priority order, matched against the closing summary. Stems
#: rather than whole words because a summary writes "champion holds" and
#: "crowned rf_kitchen" as readily as the bare verdict.
VERDICTS: tuple[tuple[str, str], ...] = (
    ("inconclusive", "inconclusive"),
    ("abandon", "abandoned"),
    ("crown", "crowned"),
    ("hold", "held"),
    ("refut", "refuted"),
)


def _verdict(experiment) -> str:
    """How an experiment closed, in the vocabulary the run skill uses.

    Inferred from the summary, which is prose, so the fallback says so instead
    of guessing. An experiment whose verdict cannot be read off its own summary
    is a real finding about that summary, not a formatting problem to paper
    over — the skill asks for one of these words explicitly.
    """
    if experiment.status != "closed":
        return experiment.status
    summary = (experiment.summary or "").lower()
    for stem, verdict in VERDICTS:
        if stem in summary:
            return verdict
    return "closed — verdict not stated in the summary"


def _experiments_for_pass(pass_id: str) -> list[str]:
    """Experiment ids this pass wrote ledger rows against, in order seen."""
    from . import results_log

    frame = results_log.read_log()
    if frame.empty or "pass_id" not in frame:
        return []
    rows = frame[frame["pass_id"].astype(str) == pass_id]
    seen: list[str] = []
    for value in rows.get("experiment_id", []):
        name = str(value)
        if name and name != "nan" and name not in seen:
            seen.append(name)
    return seen


def render(pass_id: str | None = None) -> str:
    """Build the report for one pass. Reads only; writes nothing."""
    from . import audit, champion, experiments, leakage, receipts, run_log, runs, status

    resolved = pass_id or runs.current_run_id()
    lines = [
        f"# Pass {resolved}",
        "",
        "Written by `make report`. Every number below is read from the module that "
        "owns it — the ledger, the journal, the manifest, the receipts — so this "
        "page cannot disagree with them.",
        "",
    ]

    # ---- the experiment, which is what a pass is ------------------------- #
    lines += ["## The experiment", ""]
    owned = _experiments_for_pass(resolved)
    if not owned:
        lines += [
            "**None.** This pass wrote no ledger row carrying an experiment id. "
            "A pass is one experiment, so either it stopped before opening one "
            "(the journal below says why) or it ran outside the container.",
            "",
        ]
    for experiment_id in owned:
        try:
            experiment = experiments.load_experiment(experiment_id)
        except (OSError, ValueError, KeyError):
            lines += [f"- `{experiment_id}` — no manifest on disk", ""]
            continue
        reads = experiments.val_reads_for(experiment_id)
        cv = "—" if experiment.best_cv is None else f"{experiment.best_cv:.4f}"
        val = "not read" if experiment.val_score is None else f"{experiment.val_score:.4f}"
        lines += [
            f"**`{experiment_id}`** — {experiment.name}",
            "",
            "| | |",
            "| --- | --- |",
            f"| verdict | **{_verdict(experiment)}** |",
            f"| rows | {len(experiment.train_ids)} train / {len(experiment.val_ids)} val |",
            f"| best cv | {cv} |",
            f"| validation read | {val} ({len(reads)} read{'' if len(reads) == 1 else 's'}) |",
            f"| plan locked | {_plan_state(experiment_id)} |",
            f"| spend | {experiment.spend or 'nothing counted'} |",
            "",
        ]
        if experiment.summary:
            lines += [f"> {experiment.summary}", ""]

    # ---- the champion, and what its delta really is ---------------------- #
    lines += ["## The champion", ""]
    reigning = champion.current()
    if reigning is None:
        lines += ["None crowned yet.", ""]
    else:
        delta = reigning.delta_vs_previous
        shown = "—" if delta is None else f"{delta:+.4f}"
        lines += [
            f"`{reigning.name}` — crowned {reigning.crowned_utc} by "
            f"{reigning.experiment_id or 'seed'}, delta over the previous champion "
            f"**{shown}**.",
            "",
        ]
        for amendment in champion.corrections():
            lines += [
                f"- **Corrected:** {amendment['corrects']}.{amendment['field']} "
                f"{amendment['was']} → {amendment['now']}. {amendment['reason']}",
                "",
            ]

    # ---- did the rules hold ---------------------------------------------- #
    lines += ["## Did the rules hold", ""]
    report = audit.audit()
    lines += [
        f"`make audit`: **{'pass' if report.ok else 'FAILED'}** "
        f"({len(report.checks)} checks, {len(report.failures)} binding failure"
        f"{'' if len(report.failures) == 1 else 's'})",
        "",
    ]
    for check in report.checks:
        mark = "pass" if check.passed else "**FAIL**"
        if check.informational:
            mark = "note"
        lines.append(f"- [{mark}] `{check.name}` — {check.detail}")
    lines.append("")

    guards = leakage.check_all()
    failed = [g for g in guards if not g.passed]
    lines += [
        f"`make leakage`: **{'pass' if not failed else 'FAILED'}** "
        f"({len(guards)} structural guards)",
        "",
    ]
    for guard in guards:
        lines.append(
            f"- [{'pass' if guard.passed else '**FAIL**'}] `{guard.name}` — {guard.detail}"
        )
    lines.append("")

    # ---- the final test -------------------------------------------------- #
    spent = receipts.read_receipts()
    lines += [
        "## The final test",
        "",
        (
            f"**Unread.** {paths.FROZEN_DIR.name}/holdout.csv has never been scored "
            "for this dataset, so the one generalisation number is still available."
            if not spent
            else f"**Spent.** {len(spent)} receipt(s): "
            + "; ".join(f"{r.get('run_id', '?')} at {r.get('read_utc', '?')}" for r in spent)
        ),
        "",
        "Reading it is a human's call and is deliberately absent from every `make` "
        "target. An autonomous pass does not take it.",
        "",
    ]

    # ---- what the pass decided and refused ------------------------------- #
    lines += ["## Decisions and refusals", ""]
    entries = [e for e in run_log.read_journal() if e.get("pass_id") == resolved]
    notable = [e for e in entries if e.get("event") == "refused" or e.get("level") == "decision"]
    if not notable:
        lines += [
            "Nothing journaled at decision level for this pass. A pass that stops "
            "without saying why is indistinguishable from one that crashed — "
            '`make note EVENT=decision DETAIL="..."` is how it says so.',
            "",
        ]
    for entry in notable:
        prefix = "**REFUSED**" if entry.get("event") == "refused" else entry.get("step") or "—"
        lines.append(f"- `{entry.get('timestamp_utc', '?')}` {prefix}: {entry.get('detail', '')}")
    lines.append("")

    # ---- where the pipeline is ------------------------------------------- #
    lines += ["## Pipeline", "", "| # | step | state |", "| --- | --- | --- |"]
    for step in status.pipeline_status():
        lines.append(f"| {step.number} | {step.name} | {'done' if step.done else step.detail} |")
    lines.append("")

    return "\n".join(lines)


def _plan_state(experiment_id: str) -> str:
    """Whether this experiment's predictions were locked, and still hash the same."""
    from . import experiments

    locked = experiments.locked_plan_sha256(experiment_id)
    now = experiments.plan_sha256(experiment_id)
    if now is None:
        return "no plan.md — predictions cannot be shown to predate the scoring"
    if locked is None:
        return "written but never locked — same caveat"
    if locked != now:
        return f"**MOVED** since locking ({locked[:12]} → {now[:12]})"
    return f"locked, unchanged (`{locked[:12]}`)"


def write(pass_id: str | None = None) -> Path:
    """Write the report into the pass's run directory and publish a copy."""
    from . import runs

    resolved = pass_id or runs.current_run_id()
    target = runs.run_dir(resolved) / "report.md"
    target.write_text(render(resolved) + "\n", encoding="utf-8")
    runs.publish(target, resolved)
    return target


def main(argv: list[str] | None = None) -> int:
    """``make report``. Options are make variables: ``PASS=<run id>``."""
    import argparse

    from . import runs

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pass-id", default=None, help="which pass to report on")
    args = parser.parse_args(argv)

    # With no pass in flight, the useful answer is the most recent pass rather
    # than a fresh empty directory. Same choice runs.summarize makes.
    resolved = args.pass_id
    if resolved is None and runs.RUN_ID_ENV not in __import__("os").environ:
        existing = runs.list_runs()
        resolved = existing[-1].name if existing else None

    written = write(resolved)
    print(f"report: {written}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
