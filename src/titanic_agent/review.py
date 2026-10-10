"""Step 2 — ``review``: audit the raw data before any split is drawn.

Inputs
    ``data/raw/train.csv``, ``data/raw/test.csv``.
Outputs
    ``reports/01_review.md`` — shapes, dtypes, missingness, cardinality,
    duplicate and constant columns, target balance, and leakage suspects.
    ``reports/01_review.json`` — the same findings, machine-readable, so a later
    step can act on them without parsing prose or editing source.
Outputs (return value)
    A :class:`ReviewReport` the later steps can branch on.
Done criteria
    The markdown report exists and names, explicitly, every column that needs
    imputation or encoding downstream. No data is modified by this step.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import paths

#: Minimum rows a category needs before a 0% or 100% target rate counts as a
#: leakage suspect. Below this, an extreme rate is small-sample noise: a single
#: passenger who happened to survive is not a leak.
MIN_GROUP_SUPPORT: int = 10

#: A column with at most this many distinct values is treated as categorical for
#: the group-rate scan. Above it, per-group rates are meaningless anyway.
CATEGORICAL_MAX_UNIQUE: int = 20


@dataclass(frozen=True)
class ColumnProfile:
    """Per-column facts. One of these per column in the raw training file."""

    name: str = ""
    dtype: str = ""
    is_numeric: bool = True
    missing: int = 0
    missing_frac: float = 0.0
    n_unique: int = 0
    unique_frac: float = 0.0
    #: Distinct values as a share of *observed* rows. This is the one the
    #: high-cardinality flag uses: `Cabin` has 147 distinct values in 891 rows
    #: (0.17) but 147 in the 204 rows that are not missing (0.72), and the
    #: second number is the one that says whether it can be one-hot encoded.
    unique_frac_observed: float = 0.0
    example: str = ""


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
    columns: list[ColumnProfile] = field(default_factory=list)
    needs_imputation: list[str] = field(default_factory=list)
    needs_encoding: list[str] = field(default_factory=list)
    duplicate_rows: int = 0
    notes: list[str] = field(default_factory=list)


def _profile(frame, column: str) -> ColumnProfile:
    """Describe one column without altering it."""
    from pandas.api.types import is_numeric_dtype

    series = frame[column]
    n_rows = len(frame)
    missing = int(series.isna().sum())
    non_null = series.dropna()
    n_observed = len(non_null)
    n_unique = int(series.nunique(dropna=True))
    return ColumnProfile(
        name=column,
        dtype=str(series.dtype),
        # Asked of pandas rather than inferred from the dtype string: pandas 2.x
        # reports text columns as `str` where earlier versions said `object`, so
        # comparing dtype names silently misclassifies every categorical column.
        is_numeric=bool(is_numeric_dtype(series)),
        missing=missing,
        missing_frac=round(missing / n_rows, 4) if n_rows else 0.0,
        n_unique=n_unique,
        unique_frac=round(n_unique / n_rows, 4) if n_rows else 0.0,
        unique_frac_observed=round(n_unique / n_observed, 4) if n_observed else 0.0,
        example="" if non_null.empty else str(non_null.iloc[0])[:40],
    )


def _leakage_suspects(frame, target: str) -> tuple[list[str], list[str]]:
    """Find columns that separate the target suspiciously well.

    Deliberately cheap: group rates and uniqueness, no model. A column is a
    suspect when some category with real support is entirely one class, or when
    the column is a per-row identifier (which cannot generalise but will happily
    memorise).

    Returns:
        ``(suspects, notes)`` — the column names, and one human-readable line
        per finding for the report.
    """
    suspects: list[str] = []
    notes: list[str] = []
    labels = frame[target]

    for column in frame.columns:
        if column == target:
            continue
        series = frame[column]
        n_unique = int(series.nunique(dropna=True))

        # A near-unique column can memorise the target row by row.
        if n_unique >= len(frame) * 0.95:
            suspects.append(column)
            notes.append(
                f"`{column}` has {n_unique} distinct values in {len(frame)} rows: "
                f"effectively a row identifier. Exclude it from features."
            )
            continue

        if n_unique == 0 or n_unique > CATEGORICAL_MAX_UNIQUE:
            continue

        grouped = labels.groupby(series.fillna("<missing>"))
        rates = grouped.mean()
        sizes = grouped.size()
        for value, rate in rates.items():
            support = int(sizes[value])
            if support >= MIN_GROUP_SUPPORT and rate in (0.0, 1.0):
                suspects.append(column)
                notes.append(
                    f"`{column}` = `{value}` is {rate:.0%} `{target}` across "
                    f"{support} rows. Check this is a real effect, not a leak."
                )
                break

    return suspects, notes


def _markdown(report: ReviewReport, target: str) -> str:
    """Render the report as markdown."""
    lines = [
        "# 01 — Raw data review",
        "",
        "Generated by `make review` (step 2). Regenerable: `make clean` removes it.",
        "Read-only audit — this step changes no data.",
        "",
        "## Shape",
        "",
        "| file | rows | columns |",
        "| --- | --- | --- |",
    ]
    for name in sorted(report.n_rows):
        lines.append(f"| `{name}` | {report.n_rows[name]} | {report.n_cols.get(name, 0)} |")

    lines += [
        "",
        f"Duplicate rows in training data: **{report.duplicate_rows}**",
        "",
        "## Target balance",
        "",
        "| class | share |",
        "| --- | --- |",
    ]
    for label, share in sorted(report.target_balance.items()):
        lines.append(f"| `{target}={label}` | {share:.4f} |")
    majority = max(report.target_balance.values()) if report.target_balance else 0.0
    lines += [
        "",
        f"Majority-class baseline accuracy is **{majority:.4f}**. Any model that does "
        "not beat this has learned nothing — but note that failing to beat it is a "
        "result to report, not a reason to re-roll a search.",
        "",
        "## Columns",
        "",
        "| column | dtype | numeric | missing % | distinct | distinct % of observed | example |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for profile in report.columns:
        lines.append(
            f"| `{profile.name}` | {profile.dtype} | {'yes' if profile.is_numeric else 'no'} | "
            f"{profile.missing_frac:.1%} | {profile.n_unique} | "
            f"{profile.unique_frac_observed:.1%} | {profile.example} |"
        )

    def bullet_list(items: list[str]) -> list[str]:
        return [f"- `{item}`" for item in items] if items else ["- _none_"]

    lines += ["", "## Flags", "", "**High cardinality**"]
    lines += bullet_list(report.high_cardinality)
    lines += ["", "**Constant columns**"]
    lines += bullet_list(report.constant_columns)
    lines += ["", "**Leakage suspects**"]
    lines += bullet_list(report.leakage_suspects)

    if report.notes:
        lines += ["", "### Why those were flagged", ""]
        lines += [f"- {note}" for note in report.notes]

    lines += [
        "",
        "## Decisions needed",
        "",
        "Carried forward to `split` and `tune`. Nothing is acted on here.",
        "",
        "**Needs imputation** (missing values present)",
    ]
    lines += bullet_list(report.needs_imputation)
    lines += [
        "",
        "**Needs encoding** (non-numeric, low enough cardinality to one-hot)",
    ]
    lines += bullet_list(report.needs_encoding)
    lines += [
        "",
        "All imputation and encoding belongs **inside** the modelling pipeline, fitted "
        "per fold. Fitting either on the full training split before cross-validation "
        "leaks across folds and inflates the score.",
        "",
    ]
    return "\n".join(lines)


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
        report_path: Markdown file to write. Parent dirs are created. A sibling
            ``.json`` with the same stem is written next to it.
        target: Name of the label column in the training file.
        high_cardinality_threshold: Flag a column as high-cardinality when its
            distinct-value count exceeds this fraction of its row count.

    Returns:
        The populated :class:`ReviewReport`.

    Raises:
        FileNotFoundError: ``raw_dir`` has not been populated by ``pull``.
        ValueError: The target is missing from train, or present in test.
    """
    import pandas as pd

    raw_dir = Path(raw_dir)
    train_path, test_path = raw_dir / "train.csv", raw_dir / "test.csv"
    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing — run `make pull` first")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if target not in train.columns:
        raise ValueError(f"{train_path} has no `{target}` column")
    if target in test.columns:
        raise ValueError(
            f"{test_path} contains `{target}`. This is not the competition test set, "
            "and every downstream score would be meaningless."
        )

    columns = [_profile(train, column) for column in train.columns]
    by_name = {profile.name: profile for profile in columns}

    high_cardinality = [
        p.name
        for p in columns
        if p.name != target and p.unique_frac_observed > high_cardinality_threshold
    ]
    constant = [p.name for p in columns if p.n_unique <= 1]
    suspects, notes = _leakage_suspects(train, target)

    balance = train[target].value_counts(normalize=True)
    target_balance = {str(label): round(float(share), 4) for label, share in balance.items()}

    needs_imputation = [p.name for p in columns if p.missing > 0]
    # Non-numeric, few enough distinct values to one-hot, and not already
    # flagged high-cardinality. That last clause matters: on 891 rows `Name` has
    # 891 distinct values and is excluded by the count alone, but on a small
    # sample it would slip under the threshold and this step would cheerfully
    # propose one-hot encoding a free-text name.
    needs_encoding = [
        p.name
        for p in columns
        if p.name != target
        and not p.is_numeric
        and p.n_unique <= CATEGORICAL_MAX_UNIQUE
        and p.name not in high_cardinality
    ]

    report = ReviewReport(
        n_rows={"train.csv": len(train), "test.csv": len(test)},
        n_cols={"train.csv": train.shape[1], "test.csv": test.shape[1]},
        missing_frac={p.name: p.missing_frac for p in columns},
        high_cardinality=high_cardinality,
        constant_columns=constant,
        target_balance=target_balance,
        leakage_suspects=suspects,
        report_path=report_path,
        columns=columns,
        needs_imputation=needs_imputation,
        needs_encoding=needs_encoding,
        duplicate_rows=int(train.duplicated().sum()),
        notes=notes,
    )

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_markdown(report, target), encoding="utf-8")

    # Machine-readable twin, so `split` and `tune` can act on these findings
    # without parsing markdown or (worse) having the agent edit source to encode
    # what it learned.
    payload = asdict(report)
    payload["report_path"] = str(report_path)
    report_path.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )

    # Belt and braces for the "read-only" claim: `by_name` exists only to make
    # this assertion cheap, and a mismatch means a column was lost in profiling.
    assert len(by_name) == train.shape[1], "every column must be profiled exactly once"
    return report


def main(argv: list[str] | None = None) -> int:
    """``make review`` entry point."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=paths.RAW_DIR)
    parser.add_argument("--report", type=Path, default=paths.REPORTS_DIR / "01_review.md")
    args = parser.parse_args(argv)

    try:
        report = review(args.raw_dir, report_path=args.report)
    except (FileNotFoundError, ValueError) as exc:
        print(f"review failed: {exc}")
        return 1

    print(f"review written to {report.report_path}")
    print(f"  rows: {report.n_rows}  duplicates: {report.duplicate_rows}")
    print(f"  needs imputation: {report.needs_imputation}")
    print(f"  needs encoding:   {report.needs_encoding}")
    print(f"  high cardinality: {report.high_cardinality}")
    print(f"  leakage suspects: {report.leakage_suspects}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
