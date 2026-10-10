"""Smoke test: step 5, ``explore``."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from _helpers import raises_real_runtime_error

from titanic_agent import explore, paths


def test_public_api_exists() -> None:
    assert callable(explore.explore)
    assert callable(explore.main)


def test_signature_matches_the_documented_contract() -> None:
    params = inspect.signature(explore.explore).parameters
    assert set(params) == {
        "frozen_dir",
        "reports_dir",
        "report_path",
        "target",
        "max_charts",
        "dpi",
    }
    assert params["frozen_dir"].default == paths.FROZEN_DIR
    assert params["max_charts"].default == 8


def test_result_dataclass_defaults_are_empty() -> None:
    result = explore.ExploreResult()
    assert result.charts == [] and result.candidate_features == []
    assert result.report_path is None


def test_explore_takes_no_holdout_argument() -> None:
    # The holdout must be unreachable from this step, not merely unused.
    params = inspect.signature(explore.explore).parameters
    assert not any("holdout" in name for name in params)


@pytest.mark.xfail(raises=NotImplementedError, reason="step 5 not implemented yet")
def test_explore_refuses_an_unfrozen_dataset(tmp_path: Path) -> None:
    with raises_real_runtime_error():
        explore.explore(frozen_dir=tmp_path, reports_dir=tmp_path, report_path=tmp_path / "r.md")


@pytest.mark.xfail(raises=NotImplementedError, reason="step 5 not implemented yet")
def test_explore_respects_the_chart_cap(tmp_path: Path) -> None:
    result = explore.explore(reports_dir=tmp_path, report_path=tmp_path / "r.md", max_charts=3)
    assert len(result.charts) <= 3
    assert all(chart.is_file() for chart in result.charts)
