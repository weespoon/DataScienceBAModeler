"""Step 5 — ``explore``: EDA on the frozen *training* split only.

The holdout is never loaded here. Looking at it — even in a chart — spends the
one honest estimate the pipeline has.

Inputs
    ``data/frozen/train.csv`` (via :func:`titanic_agent.freeze.require_frozen`).
Outputs
    ``reports/02_explore.md`` plus PNG charts under ``reports/``, and
    ``reports/02_explore.json`` — the machine-readable candidate-feature list
    that :mod:`titanic_agent.tune` consumes. That JSON is the handoff: without
    it, the only way for this step's findings to reach the next one would be the
    agent editing source, which is an unbounded edit surface with no provenance.
Done criteria
    The markdown summary exists, every chart it references is on disk, and the
    summary ends with a short list of candidate features for ``tune``.

Chart decisions worth recording, since they are easy to get wrong:

- **Target balance is not charted.** Two values is a stat, not a chart; a
  two-slice pie or one-bar chart says less than the number does. It appears as a
  figure in the report instead.
- Every chart carries a table of the same numbers in the markdown. A PNG has no
  hover layer, so the values have to be legible somewhere.
- Distributions by target use 2px step outlines rather than alpha-blended fills:
  translucent overlapping histograms produce a muddy third colour that encodes
  nothing.
- The surface is an explicit light colour, never transparent. These PNGs render
  on GitHub in both themes, and dark text on a transparent background is
  unreadable against a dark page.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import paths

# --------------------------------------------------------------------------- #
# Palette. Validated with the data-viz skill's checker on the light surface:
# the two-series pair passes every gate (CVD ΔE 24.7 protan, normal-vision
# ΔE 33.6, contrast >= 3:1). Do not substitute hues without re-running it.
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"  # axis labels and ticks
GRID = "#e1e0d9"  # hairline gridline
BASELINE = "#c3c2b7"  # axis rule
SERIES_1 = "#2a78d6"  # blue — single-series magnitude
SERIES_2 = "#eb6834"  # orange — second series only

#: Derived features this step proposes, with the reasoning kept next to the name
#: so ``tune`` inherits the *why* and not just the column list.
DERIVED_RATIONALE: dict[str, str] = {
    "FamilySize": "SibSp + Parch + 1; survival is non-monotonic in family size, "
    "so the raw counts split the signal in two.",
    "IsAlone": "FamilySize == 1; a single threshold that the trees find anyway "
    "but linear models cannot express.",
    "Title": "Extracted from Name (Mr/Mrs/Miss/Master/rare). Carries sex and "
    "rough age together, and survives Age being 20% missing.",
    "Deck": "First letter of Cabin, with a missing category. Cabin itself is "
    "77% missing and near-unique among observed rows, so the letter is the "
    "only part that generalises.",
    "AgeMissing": "Indicator for a missing Age. Whether the age was recorded "
    "is itself informative; imputing it away discards that.",
}


@dataclass(frozen=True)
class ExploreResult:
    """Charts and takeaways from the exploration pass."""

    charts: list[Path] = field(default_factory=list)
    candidate_features: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    report_path: Path | None = None
    baseline_accuracy: float = 0.0
    tables: dict[str, dict[str, float]] = field(default_factory=dict)


def _style(ax) -> None:
    """Recessive chrome: hairline solid grid, muted ticks, no box."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_color(MUTED)
    ax.set_axisbelow(True)


def _new_figure(dpi: int, height: float = 3.2):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, height), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    return fig, ax


def _save(fig, ax, path: Path, title: str) -> Path:
    import matplotlib.pyplot as plt

    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left", pad=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)  # figures accumulate across runs otherwise
    return path


def _rate_chart(rates, counts, *, title: str, path: Path, dpi: int) -> Path:
    """Horizontal bars for a rate across a handful of categories.

    One series, so no legend — the title names it. Bar ends are direct-labelled
    because a PNG has no tooltip, and the group size rides along so a 100% rate
    over three passengers cannot be mistaken for a finding.
    """
    fig, ax = _new_figure(dpi, height=0.5 * len(rates) + 1.4)
    labels = [f"{index}  (n={int(counts[index])})" for index in rates.index]
    positions = range(len(rates))
    # edgecolor=SURFACE with linewidth 2 is the 2px surface *gap* between
    # adjacent bars -- not a border drawn around the marks.
    ax.barh(
        list(positions),
        list(rates.values),
        height=0.62,
        color=SERIES_1,
        edgecolor=SURFACE,
        linewidth=2,
    )
    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.grid(True, color=GRID, linewidth=1.0, linestyle="-")
    ax.yaxis.grid(False)
    ax.set_xlabel("survival rate", color=MUTED, fontsize=9)
    for position, value in zip(positions, rates.values, strict=True):
        ax.text(
            min(value + 0.02, 0.97),
            position,
            f"{value:.0%}",
            va="center",
            ha="left",
            color=TEXT_SECONDARY,
            fontsize=9,
        )
    return _save(fig, ax, path, title)


def _distribution_chart(series_by_label, *, title: str, xlabel: str, path: Path, dpi: int) -> Path:
    """Two-series distribution as 2px step outlines.

    Step outlines rather than translucent fills: overlapping alpha blends into a
    third colour that encodes nothing. Two series, so a legend is required.
    """
    fig, ax = _new_figure(dpi, height=3.2)
    colors = (SERIES_1, SERIES_2)
    for (label, values), color in zip(series_by_label.items(), colors, strict=False):
        ax.hist(
            values,
            bins=24,
            histtype="step",
            linewidth=2,
            color=color,
            label=label,
        )
    ax.yaxis.grid(True, color=GRID, linewidth=1.0, linestyle="-")
    ax.xaxis.grid(False)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel("passengers", color=MUTED, fontsize=9)
    legend = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)
    return _save(fig, ax, path, title)


def _missingness_chart(fractions, *, path: Path, dpi: int) -> Path:
    """Single-series magnitude: share of rows missing, per affected column."""
    fig, ax = _new_figure(dpi, height=0.5 * len(fractions) + 1.4)
    positions = range(len(fractions))
    ax.barh(
        list(positions),
        list(fractions.values),
        height=0.62,
        color=SERIES_1,
        edgecolor=SURFACE,
        linewidth=2,
    )
    ax.set_yticks(list(positions))
    ax.set_yticklabels(list(fractions.index))
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.grid(True, color=GRID, linewidth=1.0, linestyle="-")
    ax.yaxis.grid(False)
    ax.set_xlabel("share of rows missing", color=MUTED, fontsize=9)
    for position, value in zip(positions, fractions.values, strict=True):
        ax.text(
            min(value + 0.02, 0.97),
            position,
            f"{value:.0%}",
            va="center",
            ha="left",
            color=TEXT_SECONDARY,
            fontsize=9,
        )
    return _save(fig, ax, path, "Missing values, by column")


def _markdown(result: ExploreResult, charts: list[tuple[str, Path]]) -> str:
    """Render the exploration summary, with a table beside every chart."""
    lines = [
        "# 02 — Exploration (frozen training split)",
        "",
        "Generated by `make explore` (step 5). Regenerable: `make clean` removes it.",
        "",
        "Read from `data/frozen/train.csv` only. **`holdout.csv` was not opened** —",
        "see Rule 2 in [CLAUDE.md](../CLAUDE.md). Charting the holdout would spend",
        "the one honest estimate this pipeline has.",
        "",
        "## The number to beat",
        "",
        f"Majority-class baseline accuracy: **{result.baseline_accuracy:.4f}**.",
        "",
        "That is the whole of the target-balance story, so it is a figure rather than",
        "a chart — two values do not need an axis. Whether a model beats it is a",
        "result to report, not a target to re-roll a search against.",
        "",
    ]

    for title, path in charts:
        lines += [f"## {title}", "", f"![{title}]({path.name})", ""]
        table = result.tables.get(title)
        if table:
            lines += ["| group | value |", "| --- | --- |"]
            lines += [f"| {key} | {value:.4f} |" for key, value in table.items()]
            lines += [""]

    lines += ["## Candidate features", "", "Carried to `tune` via `02_explore.json`.", ""]
    for note in result.notes:
        lines.append(f"- {note}")
    lines += [
        "",
        "All of these are computed **inside** the modelling pipeline, fitted per fold.",
        "Deriving them once over the whole training split before cross-validation",
        "leaks across folds and inflates the score.",
        "",
    ]
    return "\n".join(lines)


def explore(
    *,
    frozen_dir: Path = paths.FROZEN_DIR,
    reports_dir: Path = paths.REPORTS_DIR,
    report_path: Path = paths.REPORTS_DIR / "02_explore.md",
    target: str = "Survived",
    max_charts: int = 8,
    dpi: int = 110,
) -> ExploreResult:
    """Chart the frozen training split and summarise what matters.

    Args:
        frozen_dir: Frozen dataset directory; verified before any read.
        reports_dir: Where PNGs are written.
        report_path: Markdown summary to write. A sibling ``.json`` carries the
            candidate features to :mod:`titanic_agent.tune`.
        target: Label column to condition the charts on.
        max_charts: Cap on figures, to keep the step fast on a laptop.
        dpi: Figure resolution — 110 keeps files small and legible.

    Returns:
        An :class:`ExploreResult` listing charts and candidate features.

    Raises:
        RuntimeError: The dataset is not frozen or failed verification.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless; must precede the pyplot import

    from . import freeze, runs, tune

    freeze.require_frozen(frozen_dir)
    runs.write_metadata()

    # The rows this step may look at. Never holdout.csv -- that is the final
    # test -- and, when an experiment is open, never that experiment's
    # validation rows either.
    #
    # That second half was missing until the cold-start session caught it: tune
    # had been rewired onto the experiment's 569 rows and explore had not, so it
    # read all 712 and charted them. Nothing in the code was wrong; the charts
    # simply put the experiment's own validation rows in front of whoever read
    # the report, which is a leak through the eyes rather than through the data.
    # Shares tune.load_training_frame so the two can never drift apart again.
    train = tune.load_training_frame(frozen_dir)
    # Charts and the report land in this pass's run directory; reports/ keeps a
    # copy as the latest view so existing paths in the skills keep resolving.
    # Without this, every loop overwrote the previous pass's seven charts.
    if reports_dir == paths.REPORTS_DIR:
        reports_dir = runs.run_dir()
        report_path = reports_dir / Path(report_path).name
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    frame = train  # tune.load_training_frame already applied add_derived

    charts: list[tuple[str, Path]] = []
    tables: dict[str, dict[str, float]] = {}

    def rate_by(column: str, title: str, filename: str) -> None:
        if len(charts) >= max_charts:
            return
        grouped = frame.groupby(column)[target]
        rates, counts = grouped.mean(), grouped.size()
        path = _rate_chart(rates, counts, title=title, path=reports_dir / filename, dpi=dpi)
        charts.append((title, path))
        tables[title] = {str(k): float(v) for k, v in rates.items()}

    rate_by("Sex", "Survival rate by sex", "02_rate_sex.png")
    rate_by("Pclass", "Survival rate by passenger class", "02_rate_pclass.png")
    rate_by("Embarked", "Survival rate by port of embarkation", "02_rate_embarked.png")
    rate_by("FamilySize", "Survival rate by family size", "02_rate_family_size.png")

    for column, title, filename in (
        ("Age", "Age distribution by outcome", "02_dist_age.png"),
        ("Fare", "Fare distribution by outcome", "02_dist_fare.png"),
    ):
        if len(charts) >= max_charts:
            break
        series_by_label = {
            "did not survive": frame.loc[frame[target] == 0, column].dropna(),
            "survived": frame.loc[frame[target] == 1, column].dropna(),
        }
        path = _distribution_chart(
            series_by_label,
            title=title,
            xlabel=column,
            path=reports_dir / filename,
            dpi=dpi,
        )
        charts.append((title, path))
        tables[title] = {
            f"{label} (median)": float(values.median()) for label, values in series_by_label.items()
        }

    missing = train.isna().mean()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(charts) < max_charts and not missing.empty:
        path = _missingness_chart(missing, path=reports_dir / "02_missingness.png", dpi=dpi)
        charts.append(("Missing values, by column", path))
        tables["Missing values, by column"] = {str(k): float(v) for k, v in missing.items()}

    observed = [
        "Sex",
        "Pclass",
        "Age",
        "Fare",
        "SibSp",
        "Parch",
        "Embarked",
    ]
    candidate_features = observed + list(DERIVED_RATIONALE)
    notes = [f"`{name}` — {reason}" for name, reason in DERIVED_RATIONALE.items()]
    notes.insert(
        0,
        "Raw columns worth keeping: "
        + ", ".join(f"`{name}`" for name in observed)
        + ". `Name`, `Ticket`, `Cabin` and `PassengerId` are excluded as raw "
        "features — step 2 flagged all four as high-cardinality, and the first "
        "and last as effectively row identifiers.",
    )

    result = ExploreResult(
        charts=[path for _, path in charts],
        candidate_features=candidate_features,
        notes=notes,
        report_path=Path(report_path),
        baseline_accuracy=float(train[target].value_counts(normalize=True).max()),
        tables=tables,
    )

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_markdown(result, charts), encoding="utf-8")

    # The handoff to step 6. Without this, the only route from a finding here to
    # a change there is the agent editing tune.py.
    report_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "baseline_accuracy": result.baseline_accuracy,
                "candidate_features": result.candidate_features,
                "derived_rationale": DERIVED_RATIONALE,
                "excluded_raw": ["Name", "Ticket", "Cabin", "PassengerId"],
                "charts": [path.name for path in result.charts],
                "holdout_read": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if reports_dir.parent == paths.RUNS_DIR:
        for artifact in (*result.charts, report_path, report_path.with_suffix(".json")):
            runs.publish(artifact)
    return result


def load_candidates(report_path: Path = paths.REPORTS_DIR / "02_explore.md") -> list[str]:
    """Read the candidate features step 5 proposed, for :mod:`titanic_agent.tune`.

    Returns an empty list when step 5 has not run, so ``tune`` can fall back to
    its own defaults rather than fail.
    """
    try:
        payload = json.loads(Path(report_path).with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    features = payload.get("candidate_features")
    return [str(name) for name in features] if isinstance(features, list) else []


def main(argv: list[str] | None = None) -> int:
    """``make explore`` entry point."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-charts", type=int, default=8)
    parser.add_argument("--reports-dir", type=Path, default=paths.REPORTS_DIR)
    parser.add_argument("--frozen-dir", type=Path, default=paths.FROZEN_DIR)
    args = parser.parse_args(argv)

    try:
        result = explore(
            frozen_dir=args.frozen_dir,
            reports_dir=args.reports_dir,
            report_path=args.reports_dir / "02_explore.md",
            max_charts=args.max_charts,
        )
    except RuntimeError as exc:
        print(f"explore failed: {exc}")
        return 1

    print(f"{len(result.charts)} charts -> {result.report_path}")
    print(f"  baseline to beat: {result.baseline_accuracy:.4f}")
    print(f"  {len(result.candidate_features)} candidate features -> 02_explore.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
