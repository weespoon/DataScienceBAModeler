"""Smoke test: step 2, ``review``."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from titanic_agent import paths, review


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


@pytest.mark.xfail(raises=NotImplementedError, reason="step 2 not implemented yet")
def test_review_writes_a_markdown_report(raw_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "01_review.md"
    report = review.review(raw_dir, report_path=out)
    assert out.is_file()
    assert report.report_path == out


@pytest.mark.xfail(raises=NotImplementedError, reason="step 2 not implemented yet")
def test_review_flags_missing_age_and_high_cardinality_name(raw_dir: Path, tmp_path: Path) -> None:
    report = review.review(raw_dir, report_path=tmp_path / "r.md")
    assert report.missing_frac.get("Age", 0) > 0
    assert "Name" in report.high_cardinality


@pytest.mark.xfail(raises=NotImplementedError, reason="step 2 not implemented yet")
def test_review_leaves_the_raw_data_untouched(raw_dir: Path, tmp_path: Path) -> None:
    before = (raw_dir / "train.csv").read_bytes()
    review.review(raw_dir, report_path=tmp_path / "r.md")
    assert (raw_dir / "train.csv").read_bytes() == before
