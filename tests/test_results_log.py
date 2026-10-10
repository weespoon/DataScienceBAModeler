"""Smoke test: the append-only ledger. Fully implemented, so fully tested."""

from __future__ import annotations

import csv
from pathlib import Path

from titanic_agent import results_log


def test_ensure_log_writes_the_header(log_path: Path) -> None:
    results_log.ensure_log(log_path)
    assert log_path.is_file()
    with log_path.open(encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == list(results_log.COLUMNS)


def test_ensure_log_is_idempotent(log_path: Path) -> None:
    results_log.ensure_log(log_path)
    results_log.append_run(step="tune", model="dummy", cv_score=0.5, path=log_path)
    results_log.ensure_log(log_path)
    assert len(results_log.read_log(log_path)) == 1


def test_append_run_adds_exactly_one_row(log_path: Path) -> None:
    run_id = results_log.append_run(
        step="tune",
        model="RandomForestClassifier",
        params={"n_estimators": 400, "max_depth": 6},
        cv_score=0.8327,
        cv_std=0.0184,
        cv_folds=5,
        notes="smoke",
        path=log_path,
    )
    frame = results_log.read_log(log_path)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["run_id"] == run_id
    assert row["model"] == "RandomForestClassifier"
    assert row["params"] == {"max_depth": 6, "n_estimators": 400}
    assert abs(float(row["cv_score"]) - 0.8327) < 1e-9
    assert row["timestamp_utc"].endswith("+00:00")


def test_appends_accumulate_and_never_rewrite(log_path: Path) -> None:
    first = results_log.append_run(step="tune", model="a", cv_score=0.70, path=log_path)
    second = results_log.append_run(step="tune", model="b", cv_score=0.80, path=log_path)
    frame = results_log.read_log(log_path)
    assert list(frame["run_id"]) == [first, second]


def test_read_log_on_missing_file_is_empty_not_an_error(tmp_path: Path) -> None:
    frame = results_log.read_log(tmp_path / "nope.csv")
    assert frame.empty
    assert list(frame.columns) == list(results_log.COLUMNS)


def test_best_run_picks_the_highest_score(log_path: Path) -> None:
    results_log.append_run(step="tune", model="a", cv_score=0.70, path=log_path)
    winner = results_log.append_run(step="tune", model="b", cv_score=0.84, path=log_path)
    results_log.append_run(step="tune", model="c", cv_score=0.81, path=log_path)
    assert results_log.best_run(path=log_path)["run_id"] == winner


def test_best_run_filters_by_step(log_path: Path) -> None:
    results_log.append_run(step="tune", model="a", cv_score=0.90, path=log_path)
    target = results_log.append_run(step="ensemble", model="vote", cv_score=0.85, path=log_path)
    assert results_log.best_run(step="ensemble", path=log_path)["run_id"] == target


def test_best_run_is_none_when_nothing_is_scored(log_path: Path) -> None:
    results_log.ensure_log(log_path)
    assert results_log.best_run(path=log_path) is None


def test_notes_stay_on_one_csv_line(log_path: Path) -> None:
    results_log.append_run(step="tune", model="a", notes="line one\nline two", path=log_path)
    assert log_path.read_text(encoding="utf-8").count("\n") == 2
