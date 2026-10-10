"""Smoke test: step 2, ``review``.

Step 2 is implemented, so these are real tests. They run against the miniature
fixture in ``conftest.py``, never the repo's own ``data/`` — a test run must not
be able to disturb the frozen dataset or the ledger.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from titanic_agent import paths, review


@pytest.fixture
def reviewed(raw_dir: Path, tmp_path: Path):
    """Run the audit once and hand back (report, markdown path)."""
    report_path = tmp_path / "01_review.md"
    return review.review(raw_dir, report_path=report_path), report_path


# --------------------------------------------------------------------------- #
# contract
# --------------------------------------------------------------------------- #


def test_public_api_exists() -> None:
    assert callable(review.review)
    assert callable(review.main)


def test_signature_matches_the_documented_contract() -> None:
    params = inspect.signature(review.review).parameters
    assert set(params) == {"raw_dir", "report_path", "target", "high_cardinality_threshold"}
    assert params["raw_dir"].default == paths.RAW_DIR
    assert params["target"].default == "Survived"


def test_report_dataclass_defaults_are_empty() -> None:
    report = review.ReviewReport()
    assert report.missing_frac == {}
    assert report.high_cardinality == []
    assert report.report_path is None


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #


def test_writes_markdown_and_a_machine_readable_twin(reviewed) -> None:
    report, report_path = reviewed
    assert report_path.is_file()
    assert report.report_path == report_path
    json_path = report_path.with_suffix(".json")
    assert json_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["needs_imputation"] == report.needs_imputation


def test_markdown_has_the_sections_the_skill_promises(reviewed) -> None:
    _, report_path = reviewed
    text = report_path.read_text(encoding="utf-8")
    for heading in (
        "## Shape",
        "## Target balance",
        "## Columns",
        "## Flags",
        "## Decisions needed",
    ):
        assert heading in text


def test_every_column_appears_exactly_once(reviewed) -> None:
    report, _ = reviewed
    names = [profile.name for profile in report.columns]
    assert len(names) == len(set(names)) == 12


def test_shapes_are_reported(reviewed) -> None:
    report, _ = reviewed
    assert report.n_rows == {"train.csv": 12, "test.csv": 4}
    assert report.n_cols == {"train.csv": 12, "test.csv": 11}


def test_missingness_is_quantified(reviewed) -> None:
    report, _ = reviewed
    # The fixture leaves Age blank for one row and Cabin blank for several.
    assert report.missing_frac["Age"] > 0
    assert report.missing_frac["Cabin"] > 0
    assert "Age" in report.needs_imputation
    assert "Cabin" in report.needs_imputation
    assert report.missing_frac["Sex"] == 0.0


def test_target_balance_sums_to_one(reviewed) -> None:
    report, _ = reviewed
    assert abs(sum(report.target_balance.values()) - 1.0) < 1e-6
    assert set(report.target_balance) == {"0", "1"}


# --------------------------------------------------------------------------- #
# flags
# --------------------------------------------------------------------------- #


def test_cardinality_is_measured_over_observed_rows(reviewed) -> None:
    # Cabin is mostly missing. Judging cardinality against all rows understates
    # it; against observed rows it is near-unique, which is the number that says
    # whether it can be encoded.
    report, _ = reviewed
    cabin = next(p for p in report.columns if p.name == "Cabin")
    assert cabin.unique_frac_observed > cabin.unique_frac
    assert "Cabin" in report.high_cardinality


def test_free_text_columns_are_flagged_high_cardinality(reviewed) -> None:
    report, _ = reviewed
    assert {"Name", "Ticket"} <= set(report.high_cardinality)


def test_high_cardinality_columns_are_never_proposed_for_encoding(reviewed) -> None:
    # One-hot encoding a passenger name is absurd; on a small sample the raw
    # distinct-count test alone would allow it.
    report, _ = reviewed
    assert not set(report.needs_encoding) & set(report.high_cardinality)
    assert "Name" not in report.needs_encoding


def test_low_cardinality_text_columns_are_proposed_for_encoding(reviewed) -> None:
    report, _ = reviewed
    assert "Sex" in report.needs_encoding
    assert "Embarked" in report.needs_encoding


def test_categoricals_are_detected_without_comparing_dtype_names(reviewed) -> None:
    # pandas 2.x reports text columns as `str` where older versions said
    # `object`; is_numeric is asked of pandas so the check survives either.
    report, _ = reviewed
    by_name = {p.name: p for p in report.columns}
    assert by_name["Sex"].is_numeric is False
    assert by_name["Age"].is_numeric is True
    assert by_name["Pclass"].is_numeric is True


def test_row_identifiers_are_flagged_as_leakage_suspects(reviewed) -> None:
    report, _ = reviewed
    assert "PassengerId" in report.leakage_suspects
    assert report.notes, "a flagged suspect must come with a reason"


def test_a_constant_column_is_flagged(raw_dir: Path, tmp_path: Path) -> None:
    import pandas as pd

    frame = pd.read_csv(raw_dir / "train.csv")
    frame["AllSame"] = 7
    frame.to_csv(raw_dir / "train.csv", index=False)
    report = review.review(raw_dir, report_path=tmp_path / "r.md")
    assert "AllSame" in report.constant_columns


def test_small_groups_do_not_count_as_leakage(raw_dir: Path, tmp_path: Path) -> None:
    # A single passenger who happened to survive is noise, not a leak. The
    # fixture has 12 rows, far below MIN_GROUP_SUPPORT, so no low-cardinality
    # column should be flagged on group rate alone.
    report = review.review(raw_dir, report_path=tmp_path / "r.md")
    assert "Sex" not in report.leakage_suspects
    assert "Pclass" not in report.leakage_suspects


# --------------------------------------------------------------------------- #
# read-only guarantee and failure modes
# --------------------------------------------------------------------------- #


def test_the_raw_data_is_byte_identical_afterwards(raw_dir: Path, tmp_path: Path) -> None:
    before = {p.name: p.read_bytes() for p in raw_dir.glob("*.csv")}
    review.review(raw_dir, report_path=tmp_path / "r.md")
    after = {p.name: p.read_bytes() for p in raw_dir.glob("*.csv")}
    assert before == after


def test_missing_raw_data_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        review.review(tmp_path, report_path=tmp_path / "r.md")


def test_a_target_leaking_into_test_is_refused(raw_dir: Path, tmp_path: Path) -> None:
    import pandas as pd

    frame = pd.read_csv(raw_dir / "test.csv")
    frame["Survived"] = 0
    frame.to_csv(raw_dir / "test.csv", index=False)
    with pytest.raises(ValueError, match="not the competition test set"):
        review.review(raw_dir, report_path=tmp_path / "r.md")


def test_training_data_without_the_target_is_refused(raw_dir: Path, tmp_path: Path) -> None:
    import pandas as pd

    pd.read_csv(raw_dir / "train.csv").drop(columns=["Survived"]).to_csv(
        raw_dir / "train.csv", index=False
    )
    with pytest.raises(ValueError, match="no `Survived` column"):
        review.review(raw_dir, report_path=tmp_path / "r.md")


def test_main_returns_zero_and_reports(raw_dir: Path, tmp_path: Path, capsys) -> None:
    code = review.main(["--raw-dir", str(raw_dir), "--report", str(tmp_path / "r.md")])
    assert code == 0
    assert "needs encoding" in capsys.readouterr().out


def test_main_reports_failure_without_a_traceback(tmp_path: Path, capsys) -> None:
    code = review.main(["--raw-dir", str(tmp_path / "nope"), "--report", str(tmp_path / "r.md")])
    assert code == 1
    assert "review failed" in capsys.readouterr().out
