"""Step 2 — ``review``: audit the raw data before any split is drawn.

Inputs
    ``data/raw/train.csv``, ``data/raw/test.csv``.
Outputs
    ``reports/01_review.md`` — shapes, dtypes, missingness, cardinality,
    duplicate and constant columns, target balance, and leakage suspects.
Outputs (return value)
    A :class:`ReviewReport` the later steps can branch on.
Done criteria
    The markdown report exists and names, explicitly, every column that needs
    imputation or encoding downstream. No data is modified by this step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import paths


@dataclass(frozen=True)
class ReviewReport:
    """Machine-readable summary of the raw-data audit."""

    n_rows: dict[str, int] = field(default_factory=dict)
    n_cols: dict[str, int] = field(default_factory=dict)
    missing_frac: dict[str, float] = field(default_factory=dict)
    high_cardinality: list[str] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)
    target_balance: dict[str, float] = field(default_factory=dict)
    leakage_suspects: list[str] = field(default_factory=list)
    report_path: Path | None = None


def review(
    raw_dir: Path = paths.RAW_DIR,
    *,
    report_path: Path = paths.REPORTS_DIR / "01_review.md",
    target: str = "Survived",
    high_cardinality_threshold: float = 0.5,
) -> ReviewReport:
    """Profile the raw dataset and write a markdown audit.

    Args:
        raw_dir: Directory holding the pulled CSVs.
        report_path: Markdown file to write. Parent dirs are created.
        target: Name of the label column in the training file.
        high_cardinality_threshold: Flag a column as high-cardinality when its
            distinct-value count exceeds this fraction of its row count.

    Returns:
        The populated :class:`ReviewReport`.

    Raises:
        FileNotFoundError: ``raw_dir`` has not been populated by ``pull``.
    """
    # TODO(review): implement.
    #   1. Load train/test with pandas; assert the target is absent from test.
    #   2. Compute per-column dtype, missing fraction, nunique.
    #   3. Flag constant columns, high-cardinality columns (Name/Ticket/Cabin),
    #      and leakage suspects (anything perfectly separating the target).
    #   4. Render it all to report_path as markdown tables.
    #   5. Read-only: never write back into raw_dir.
    raise NotImplementedError("review() is not implemented yet")


def main(argv: list[str] | None = None) -> int:
    """``make review`` entry point."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=paths.RAW_DIR)
    parser.add_argument("--report", type=Path, default=paths.REPORTS_DIR / "01_review.md")
    args = parser.parse_args(argv)

    report = review(args.raw_dir, report_path=args.report)
    print(f"review written to {report.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
