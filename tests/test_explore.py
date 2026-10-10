"""Smoke test: step 5, ``explore``.

The load-bearing tests here are the Rule 2 ones: this step must be unable to
reach the holdout, not merely uninterested in it.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from _helpers import raises_real_runtime_error

from titanic_agent import explore, paths


@pytest.fixture
def explored(frozen_dir: Path, tmp_path: Path):
    """Run the exploration once against the miniature frozen fixture."""
    reports = tmp_path / "reports"
    result = explore.explore(
        frozen_dir=frozen_dir,
        reports_dir=reports,
        report_path=reports / "02_explore.md",
    )
    return result, reports


# --------------------------------------------------------------------------- #
# contract
# --------------------------------------------------------------------------- #


def test_public_api_exists() -> None:
    assert callable(explore.explore)
    assert callable(explore.load_candidates)
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


# --------------------------------------------------------------------------- #
# Rule 2: the holdout must be unreachable from here
# --------------------------------------------------------------------------- #


def test_explore_takes_no_holdout_argument() -> None:
    params = inspect.signature(explore.explore).parameters
    assert not any("holdout" in name for name in params)


def test_the_source_never_reads_the_holdout_file() -> None:
    # Stronger than "no test noticed a holdout read": no read in this module
    # targets it. Checked against read calls specifically, not the whole file --
    # the first version of this test asserted the string was absent entirely and
    # failed on the report line that says the holdout was *not* opened. An
    # over-broad pattern flags the prose describing the rule, which is the same
    # mistake the data/frozen hook made.
    source = Path(explore.__file__).read_text(encoding="utf-8")
    reads = [
        line
        for line in source.splitlines()
        if "read_csv(" in line or "open(" in line or "read_text(" in line
    ]
    assert reads, "expected this module to read something"
    offenders = [line.strip() for line in reads if "holdout" in line.lower()]
    assert not offenders, f"a read in explore.py targets the holdout: {offenders}"


def test_the_handoff_attests_that_the_holdout_was_not_read(explored) -> None:
    _, reports = explored
    payload = json.loads((reports / "02_explore.json").read_text(encoding="utf-8"))
    assert payload["holdout_read"] is False


def test_explore_refuses_an_unfrozen_dataset(tmp_path: Path) -> None:
    with raises_real_runtime_error():
        explore.explore(
            frozen_dir=tmp_path,
            reports_dir=tmp_path,
            report_path=tmp_path / "r.md",
        )


def test_explore_refuses_a_tampered_dataset(writable_frozen_dir: Path, tmp_path: Path) -> None:
    (writable_frozen_dir / "train.csv").write_text("broken\n", encoding="utf-8")
    with raises_real_runtime_error():
        explore.explore(
            frozen_dir=writable_frozen_dir,
            reports_dir=tmp_path / "reports",
            report_path=tmp_path / "reports" / "r.md",
        )


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #


def test_every_referenced_chart_is_on_disk(explored) -> None:
    result, reports = explored
    text = (reports / "02_explore.md").read_text(encoding="utf-8")
    assert result.charts
    for chart in result.charts:
        assert chart.is_file()
        assert chart.name in text, f"{chart.name} is on disk but not referenced"


def test_charts_are_pngs_with_content(explored) -> None:
    result, _ = explored
    for chart in result.charts:
        assert chart.suffix == ".png"
        assert chart.stat().st_size > 1000


def test_the_chart_cap_is_respected(frozen_dir: Path, tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    result = explore.explore(
        frozen_dir=frozen_dir,
        reports_dir=reports,
        report_path=reports / "r.md",
        max_charts=3,
    )
    assert len(result.charts) <= 3


def test_target_balance_is_a_figure_not_a_chart(explored) -> None:
    # Two values is a stat, not a chart. It belongs in the prose.
    result, reports = explored
    text = (reports / "02_explore.md").read_text(encoding="utf-8")
    assert f"{result.baseline_accuracy:.4f}" in text
    assert not any("balance" in chart.name for chart in result.charts)


def test_the_baseline_is_the_majority_class(explored) -> None:
    result, _ = explored
    assert 0.5 <= result.baseline_accuracy <= 1.0


def test_candidate_features_are_proposed_with_reasons(explored) -> None:
    result, _ = explored
    assert len(result.candidate_features) >= 5
    assert len(result.notes) >= 5


def test_derived_features_are_named_in_the_handoff(explored) -> None:
    _, reports = explored
    payload = json.loads((reports / "02_explore.json").read_text(encoding="utf-8"))
    assert {"FamilySize", "IsAlone", "Title"} <= set(payload["candidate_features"])
    assert payload["derived_rationale"]


def test_identifier_columns_are_excluded(explored) -> None:
    result, _ = explored
    assert not {"PassengerId", "Name", "Ticket", "Cabin"} & set(result.candidate_features)


def test_every_chart_has_a_table_beside_it(explored) -> None:
    # A PNG has no hover layer, so the numbers must be legible somewhere.
    result, _ = explored
    assert len(result.tables) == len(result.charts)


def test_load_candidates_round_trips(explored) -> None:
    result, reports = explored
    assert explore.load_candidates(reports / "02_explore.md") == result.candidate_features


def test_load_candidates_is_empty_when_step_5_has_not_run(tmp_path: Path) -> None:
    # tune falls back to its own defaults rather than failing.
    assert explore.load_candidates(tmp_path / "nope.md") == []


def test_main_returns_zero(frozen_dir: Path, tmp_path: Path, capsys) -> None:
    code = explore.main(
        ["--frozen-dir", str(frozen_dir), "--reports-dir", str(tmp_path / "reports")]
    )
    assert code == 0
    assert "baseline to beat" in capsys.readouterr().out


def test_main_reports_failure_without_a_traceback(tmp_path: Path, capsys) -> None:
    code = explore.main(["--frozen-dir", str(tmp_path), "--reports-dir", str(tmp_path)])
    assert code == 1
    assert "explore failed" in capsys.readouterr().out
