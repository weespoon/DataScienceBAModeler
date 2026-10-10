"""Smoke test: per-pass run directories.

The load-bearing tests are the carve-out ones. Moving artifacts into run
directories is a convenience; moving *evidence* in there would be a silent
disaster, and :func:`test_receipts_are_never_per_run` is what stops a future
refactor from doing it by accident.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from titanic_agent import paths, run_log, runs


@pytest.fixture
def runs_root(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS_DIR", root)
    return root


# --------------------------------------------------------------------------- #
# the carve-out: artifacts move, evidence does not
# --------------------------------------------------------------------------- #


#: These assert on how paths.py *defines* the artifacts, not on the live values.
#: The autouse guard in conftest redirects the live ones into tmp_path for every
#: test, which is exactly right for isolation and exactly wrong for checking
#: where they are declared to live.
PATHS_SOURCE = Path(paths.__file__).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "declaration",
    [
        "HOLDOUT_READS: Path = EXPERIMENTS_DIR /",
        "RESULTS_LOG: Path = EXPERIMENTS_DIR /",
        "RUN_LOG: Path = EXPERIMENTS_DIR /",
    ],
)
def test_evidence_is_declared_outside_run_directories(declaration: str) -> None:
    # The sharpest case is holdout_reads.json. If it ever lands inside a run
    # directory, "one read per frozen dataset" silently becomes "one read per
    # run": each pass sees an empty receipts file, reads the holdout, and passes
    # its own audit. Unlimited reads, every one green.
    #
    # The ledger is less catastrophic but still wrong: split per run and you
    # cannot compare passes, and the hash chain would only protect within one,
    # so deleting a whole run becomes undetectable.
    assert declaration in PATHS_SOURCE, f"{declaration!r} moved — see runs.py on why not"


def test_no_artifact_path_is_declared_under_runs() -> None:
    for line in PATHS_SOURCE.splitlines():
        if line.startswith(("HOLDOUT_READS", "RESULTS_LOG", "RUN_LOG", "FROZEN_")):
            assert "RUNS_DIR" not in line, line


def test_the_frozen_dataset_is_never_per_run() -> None:
    assert "FROZEN_DIR: Path = DATA_DIR /" in PATHS_SOURCE


# --------------------------------------------------------------------------- #
# minting ids
# --------------------------------------------------------------------------- #


def test_an_id_is_human_readable_and_sorts_chronologically(runs_root: Path) -> None:
    early = runs.mint_run_id(datetime(2026, 3, 1, 9, 5), runs_dir=runs_root)
    late = runs.mint_run_id(datetime(2026, 10, 10, 14, 35), runs_dir=runs_root)
    assert early < late
    assert early.startswith("20260301-0905")


def test_an_id_matches_the_documented_shape(runs_root: Path) -> None:
    assert runs.RUN_ID_PATTERN.match(runs.mint_run_id(runs_dir=runs_root))


def test_the_zone_abbreviation_disambiguates_the_dst_hour(runs_root: Path) -> None:
    # 01:30 happens twice on the night the clocks go back. A generic "ET" would
    # collide; the real abbreviation (EDT vs EST) does not. This asserts the id
    # carries letters rather than a fixed label, whatever zone the test runs in.
    minted = runs.mint_run_id(runs_dir=runs_root)
    zone = minted.split("-")[1][4:]
    assert zone.isalpha() and zone.isupper()


def test_a_collision_gets_a_suffix(runs_root: Path) -> None:
    when = datetime(2026, 10, 10, 14, 35)
    first = runs.mint_run_id(when, runs_dir=runs_root)
    (runs_root / first).mkdir(parents=True)
    second = runs.mint_run_id(when, runs_dir=runs_root)
    assert second != first
    assert second.endswith("-2")


def test_the_environment_pins_the_id_across_processes(monkeypatch) -> None:
    monkeypatch.setenv(runs.RUN_ID_ENV, "20261010-1435EDT")
    assert runs.current_run_id() == "20261010-1435EDT"


def test_the_run_id_and_the_pass_id_are_the_same_string(monkeypatch) -> None:
    # One identifier, so `ls runs/X` and `make journal --pass-id X` agree.
    monkeypatch.setenv(runs.RUN_ID_ENV, "20261010-1435EDT")
    assert run_log.current_pass_id() == runs.current_run_id()


# --------------------------------------------------------------------------- #
# directories and metadata
# --------------------------------------------------------------------------- #


def test_run_dir_is_created(runs_root: Path) -> None:
    directory = runs.run_dir("20261010-1435EDT")
    assert directory.is_dir()
    assert directory.parent == runs_root


def test_metadata_records_the_authoritative_utc_instant(runs_root: Path) -> None:
    # The directory name is local time for readability; this is the authority.
    path = runs.write_metadata("20261010-1435EDT")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["started_utc"].endswith("+00:00")
    assert payload["run_id"] == "20261010-1435EDT"
    assert payload["timezone"]


def test_metadata_keeps_the_first_start_time(runs_root: Path) -> None:
    # explore, tune and ensemble each call it; the pass started once.
    first = json.loads(runs.write_metadata("x").read_text(encoding="utf-8"))
    second = json.loads(runs.write_metadata("x").read_text(encoding="utf-8"))
    assert first["started_utc"] == second["started_utc"]


def test_metadata_survives_a_corrupt_file(runs_root: Path) -> None:
    runs.run_dir("x")
    (runs_root / "x" / "metadata.json").write_text("{not json", encoding="utf-8")
    assert runs.write_metadata("x").is_file()


def test_list_runs_is_chronological(runs_root: Path) -> None:
    for name in ("20261010-1435EDT", "20260301-0905EST", "20261010-1600EDT"):
        runs.run_dir(name)
    assert [p.name for p in runs.list_runs()] == [
        "20260301-0905EST",
        "20261010-1435EDT",
        "20261010-1600EDT",
    ]


def test_list_runs_is_empty_before_any_pass(runs_root: Path) -> None:
    assert runs.list_runs() == []


# --------------------------------------------------------------------------- #
# publishing to reports/
# --------------------------------------------------------------------------- #


def test_publish_copies_into_reports(runs_root: Path, tmp_path: Path, monkeypatch) -> None:
    reports = tmp_path / "reports"
    monkeypatch.setattr(paths, "REPORTS_DIR", reports)
    source = runs.run_dir("x") / "02_explore.md"
    source.write_text("latest", encoding="utf-8")
    published = runs.publish(source)
    assert published == reports / "02_explore.md"
    assert published.read_text(encoding="utf-8") == "latest"


def test_publishing_twice_overwrites_with_the_newer(
    runs_root: Path, tmp_path: Path, monkeypatch
) -> None:
    # reports/ is the latest view, so a second pass must replace the first.
    reports = tmp_path / "reports"
    monkeypatch.setattr(paths, "REPORTS_DIR", reports)
    for name, body in (("a", "first"), ("b", "second")):
        source = runs.run_dir(name) / "02_explore.md"
        source.write_text(body, encoding="utf-8")
        runs.publish(source)
    assert (reports / "02_explore.md").read_text(encoding="utf-8") == "second"


def test_publishing_a_file_already_in_reports_is_a_noop(tmp_path: Path, monkeypatch) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr(paths, "REPORTS_DIR", reports)
    source = reports / "01_review.md"
    source.write_text("body", encoding="utf-8")
    assert runs.publish(source).read_text(encoding="utf-8") == "body"


# --------------------------------------------------------------------------- #
# the per-run summary
# --------------------------------------------------------------------------- #


def test_summarize_writes_run_md(runs_root: Path, monkeypatch) -> None:
    monkeypatch.setenv(runs.RUN_ID_ENV, "20261010-1435EDT")
    path = runs.summarize()
    assert path.name == "run.md"
    assert "20261010-1435EDT" in path.read_text(encoding="utf-8")


def test_the_summary_is_derived_not_a_second_copy(runs_root: Path, monkeypatch) -> None:
    # Two records of the same events can disagree, and then neither can be
    # trusted. The summary filters the global ledger and journal every time.
    monkeypatch.setenv(runs.RUN_ID_ENV, "20261010-1435EDT")
    text = runs.summarize().read_text(encoding="utf-8")
    assert "experiments/results.csv" in text
    assert "make audit" in text


def test_the_summary_names_decisions_and_refusals(
    runs_root: Path, journal_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(paths, "RUN_LOG", journal_path)
    monkeypatch.setenv(runs.RUN_ID_ENV, "20261010-1435EDT")
    run_log.record("refused", step="ensemble", detail="declined the holdout read")
    text = runs.summarize().read_text(encoding="utf-8")
    assert "declined the holdout read" in text


def test_the_summary_says_so_when_nothing_needed_a_person(
    runs_root: Path, journal_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(paths, "RUN_LOG", journal_path)
    monkeypatch.setenv(runs.RUN_ID_ENV, "20261010-1435EDT")
    text = runs.summarize().read_text(encoding="utf-8")
    assert "nothing needing a person" in text


def test_main_lists_runs(runs_root: Path, capsys) -> None:
    runs.run_dir("20261010-1435EDT")
    assert runs.main([]) == 0
    assert "20261010-1435EDT" in capsys.readouterr().out


def test_main_reports_no_runs_yet(runs_root: Path, capsys) -> None:
    assert runs.main([]) == 0
    assert "no runs yet" in capsys.readouterr().out


def test_main_summarize(runs_root: Path, capsys, monkeypatch) -> None:
    monkeypatch.setenv(runs.RUN_ID_ENV, "20261010-1435EDT")
    assert runs.main(["--summarize"]) == 0
    assert "run summary" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# the guard that stops a test run touching the real repo
# --------------------------------------------------------------------------- #


def test_every_writable_artifact_is_redirected_during_tests() -> None:
    # The autouse guard in conftest redirects these. If a new artifact path is
    # added to paths.py and not to WRITABLE_ARTIFACTS, the suite starts writing
    # into the real repo -- which is how 53 stray run directories appeared.
    repo = Path(paths.__file__).resolve().parents[2]
    for attribute in paths.WRITABLE_ARTIFACTS:
        live = getattr(paths, attribute)
        assert not live.is_relative_to(repo), (
            f"paths.{attribute} is not redirected during tests: {live}"
        )


def test_the_writable_list_names_real_attributes() -> None:
    for attribute in paths.WRITABLE_ARTIFACTS:
        assert isinstance(getattr(paths, attribute), Path), attribute


def test_summarize_does_not_mint_an_empty_run(runs_root: Path, monkeypatch) -> None:
    # `make summary` on its own should describe the most recent pass, not
    # create a new empty directory and summarise nothing.
    monkeypatch.delenv(runs.RUN_ID_ENV, raising=False)
    runs.run_dir("20261010-1400EDT")
    runs.run_dir("20261010-1500EDT")
    path = runs.summarize()
    assert path.parent.name == "20261010-1500EDT"
    assert len(runs.list_runs()) == 2
