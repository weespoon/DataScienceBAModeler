"""Smoke test: the Rule 4 guards.

Every test here constructs the cheat it is meant to catch. A structural guard
that has never tripped is a guard nobody has checked, and this one protects the
number the whole programme is built around: our locked final test is 179 of the
same passengers the public manifest covers, so a lookup inflates it while the
ledger chains cleanly and every other check stays green.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from titanic_agent import leakage, results_log

# --------------------------------------------------------------------------- #
# feature provenance — a column from somewhere else
# --------------------------------------------------------------------------- #


def test_a_feature_not_derivable_from_frozen_data_is_caught(
    frozen_dir: Path, log_path: Path
) -> None:
    results_log.append_run(
        step="tune",
        model="rf",
        cv_score=0.83,
        params={"features": ["Sex", "Pclass", "ActualSurvivalFromManifest"]},
        path=log_path,
    )
    check = leakage.check_feature_provenance(log_path=log_path, frozen_dir=frozen_dir)
    assert not check.passed
    assert check.offenders == ["ActualSurvivalFromManifest"]


def test_legitimate_engineered_features_are_allowed(frozen_dir: Path, log_path: Path) -> None:
    # Derived columns are derived from data we already sealed, so they are fine
    # the moment they exist in code. The check is about provenance, not novelty.
    results_log.append_run(
        step="tune",
        model="rf",
        cv_score=0.83,
        params={"features": ["Sex", "Title", "FamilySize", "Deck", "AgeMissing"]},
        path=log_path,
    )
    assert leakage.check_feature_provenance(log_path=log_path, frozen_dir=frozen_dir).passed


def test_derivable_columns_include_raw_and_engineered(frozen_dir: Path) -> None:
    columns = leakage.derivable_columns(frozen_dir)
    assert {"Sex", "Pclass", "Age"} <= columns  # raw
    assert {"Title", "FamilySize", "IsAlone"} <= columns  # engineered


def test_an_empty_ledger_passes(frozen_dir: Path, log_path: Path) -> None:
    results_log.ensure_log(log_path)
    assert leakage.check_feature_provenance(log_path=log_path, frozen_dir=frozen_dir).passed


# --------------------------------------------------------------------------- #
# external reads — a lookup hidden inside feature engineering
# --------------------------------------------------------------------------- #


def test_a_url_read_is_caught(tmp_path: Path) -> None:
    # The column this produces would look perfectly ordinary. The read does not.
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "features.py").write_text(
        "import pandas as pd\n"
        "def enrich(df):\n"
        "    m = pd.read_csv('https://encyclopedia-titanica.example/passengers.csv')\n"
        "    return df.merge(m, on='Name')\n",
        encoding="utf-8",
    )
    check = leakage.check_external_reads(package)
    assert not check.passed
    assert "reads a URL" in check.offenders[0]


def test_a_read_outside_the_sanctioned_roots_is_caught(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "features.py").write_text(
        "import pandas as pd\nx = pd.read_csv('/Users/someone/Downloads/manifest.csv')\n",
        encoding="utf-8",
    )
    assert not leakage.check_external_reads(package).passed


@pytest.mark.parametrize(
    "source",
    [
        "x = pd.read_csv('data/frozen/train.csv')",
        "x = pd.read_csv('data/raw/test.csv')",
        "x = pd.read_csv(frozen_dir / 'train.csv')",  # not a literal; nothing to judge
        "x = pd.read_csv(some_path)",
    ],
)
def test_sanctioned_reads_pass(tmp_path: Path, source: str) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "m.py").write_text(f"import pandas as pd\n{source}\n", encoding="utf-8")
    assert leakage.check_external_reads(package).passed


def test_the_real_package_has_no_external_reads() -> None:
    assert leakage.check_external_reads().passed


# --------------------------------------------------------------------------- #
# plausibility — a score that is not a modelling achievement
# --------------------------------------------------------------------------- #


def test_an_implausible_score_is_caught(log_path: Path) -> None:
    results_log.append_run(step="tune", model="rf", cv_score=0.97, path=log_path)
    check = leakage.check_plausibility(log_path=log_path)
    assert not check.passed
    assert "honest ceiling" in check.detail


def test_a_realistic_score_passes(log_path: Path) -> None:
    # Our actual best is 0.8287. The line sits far above the honest ceiling so a
    # genuinely strong model is never flagged.
    results_log.append_run(step="tune", model="rf", cv_score=0.8287, path=log_path)
    assert leakage.check_plausibility(log_path=log_path).passed


def test_the_threshold_leaves_headroom_above_the_honest_ceiling() -> None:
    assert leakage.IMPLAUSIBLE_ACCURACY > 0.86


def test_unscored_rows_do_not_trip_it(log_path: Path) -> None:
    results_log.append_run(step="tune", model="rf", cv_score=None, path=log_path)
    assert leakage.check_plausibility(log_path=log_path).passed


# --------------------------------------------------------------------------- #
# stray data — how an external manifest arrives
# --------------------------------------------------------------------------- #


def test_an_unexpected_data_file_is_caught(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "train.csv").write_text("x", encoding="utf-8")
    (tmp_path / "raw" / "titanica_full_manifest.csv").write_text("x", encoding="utf-8")
    check = leakage.check_stray_data(tmp_path)
    assert not check.passed
    assert check.offenders == ["raw/titanica_full_manifest.csv"]


def test_the_expected_files_pass(tmp_path: Path) -> None:
    (tmp_path / "frozen").mkdir()
    for name in ("train.csv", "holdout.csv", "test.csv", "MANIFEST.json"):
        (tmp_path / "frozen" / name).write_text("x", encoding="utf-8")
    assert leakage.check_stray_data(tmp_path).passed


def test_the_real_data_directory_is_clean() -> None:
    assert leakage.check_stray_data().passed


# --------------------------------------------------------------------------- #
# the suite as a whole
# --------------------------------------------------------------------------- #


def test_check_all_runs_every_guard(frozen_dir: Path, log_path: Path) -> None:
    names = {c.name for c in leakage.check_all(log_path=log_path, frozen_dir=frozen_dir)}
    assert names == {
        "feature provenance",
        "external reads",
        "score plausibility",
        "stray data files",
    }


def test_main_passes_on_the_real_repo(capsys) -> None:
    assert leakage.main([]) == 0
    assert "not proof of honesty" in capsys.readouterr().out


def test_the_audit_includes_rule_4(frozen_dir: Path, log_path: Path) -> None:
    from titanic_agent import audit, tune

    tune.tune("dummy", frozen_dir=frozen_dir, log_path=log_path, cv_folds=2)
    report = audit.audit(
        frozen_dir=frozen_dir, log_path=log_path, receipts_path=log_path.parent / "r.json"
    )
    assert any(c.name.startswith("rule 4") for c in report.checks)
