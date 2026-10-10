"""Step 5 — ``explore``: EDA on the frozen *training* split only.

The holdout is never loaded here. Looking at it — even in a chart — spends the
one honest estimate the pipeline has.

Inputs
    ``data/frozen/train.csv`` (via :func:`titanic_agent.freeze.require_frozen`).
Outputs
    ``reports/02_explore.md`` plus PNG charts under ``reports/``.
Done criteria
    The markdown summary exists, every chart it references is on disk, and the
    summary ends with a short list of candidate features for ``tune``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import paths


@dataclass(frozen=True)
class ExploreResult:
    """Charts and takeaways from the exploration pass."""

    charts: list[Path] = field(default_factory=list)
    candidate_features: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    report_path: Path | None = None


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
        report_path: Markdown summary to write.
        target: Label column to condition the charts on.
        max_charts: Cap on figures, to keep the step fast on a laptop.
        dpi: Figure resolution — 110 keeps files small and legible.

    Returns:
        An :class:`ExploreResult` listing charts and candidate features.

    Raises:
        RuntimeError: The dataset is not frozen or failed verification.
    """
    # TODO(explore): implement.
    #   1. freeze.require_frozen(frozen_dir); load ONLY train.csv.
    #   2. Use matplotlib with the Agg backend (no display on CI / headless).
    #      One chart per figure, no subplots-as-dashboards.
    #   3. Suggested set: target balance; survival by Sex, Pclass, Embarked;
    #      Age and Fare distributions by target; missingness bar; family-size
    #      derived from SibSp+Parch.
    #   4. Close every figure (plt.close) -- memory adds up across runs.
    #   5. Write report_path with each chart embedded and a Candidate Features
    #      section naming what tune() should start from.
    raise NotImplementedError("explore() is not implemented yet")


def main(argv: list[str] | None = None) -> int:
    """``make explore`` entry point."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-charts", type=int, default=8)
    args = parser.parse_args(argv)

    result = explore(max_charts=args.max_charts)
    print(f"{len(result.charts)} charts -> {result.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
