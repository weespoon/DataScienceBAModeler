"""Smoke test: the agent run journal.

The journal records *conduct* where ``results.csv`` records *models*. The test
that matters most is :func:`test_a_truncated_final_line_is_skipped_not_fatal` —
the journal's whole job is to survive the run it is journalling, including a run
that dies partway through a write.
"""

from __future__ import annotations

import json
from pathlib import Path

from titanic_agent import paths, run_log


def test_journal_path_is_separate_from_the_ledger() -> None:
    # Two files, two jobs. Merging them would make "what did the agent do"
    # answerable only by reading model scores, which do not record it.
    assert paths.RUN_LOG != paths.RESULTS_LOG
    assert paths.RUN_LOG.suffix == ".jsonl"


def test_events_vocabulary_is_small() -> None:
    assert "refused" in run_log.EVENTS
    assert len(run_log.EVENTS) <= 10


def test_recording_creates_the_journal(journal_path: Path) -> None:
    entry = run_log.record("note", step="tune", detail="hello", path=journal_path)
    assert journal_path.is_file()
    assert entry["event"] == "note"
    assert entry["step"] == "tune"
    assert entry["timestamp_utc"].endswith("+00:00")


def test_entries_accumulate_in_order(journal_path: Path) -> None:
    run_log.record("step_started", step="tune", path=journal_path)
    run_log.record("step_finished", step="tune", path=journal_path)
    entries = run_log.read_journal(journal_path)
    assert [e["event"] for e in entries] == ["step_started", "step_finished"]


def test_each_entry_is_one_line_of_json(journal_path: Path) -> None:
    run_log.record("note", detail="line one\nline two", path=journal_path)
    text = journal_path.read_text(encoding="utf-8")
    assert text.count("\n") == 1
    assert json.loads(text)["detail"] == "line one line two"


def test_extra_fields_are_kept(journal_path: Path) -> None:
    run_log.record("decision", step="ensemble", run_id="abc123", path=journal_path)
    assert run_log.read_journal(journal_path)[0]["run_id"] == "abc123"


def test_reading_a_missing_journal_is_empty_not_an_error(tmp_path: Path) -> None:
    assert run_log.read_journal(tmp_path / "nope.jsonl") == []


def test_a_truncated_final_line_is_skipped_not_fatal(journal_path: Path) -> None:
    # A run killed mid-write leaves a partial last line. Everything before it is
    # still a valid record of what happened and must remain readable.
    run_log.record("step_started", step="tune", path=journal_path)
    with journal_path.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "step_fini')
    entries = run_log.read_journal(journal_path)
    assert len(entries) == 1
    assert entries[0]["event"] == "step_started"


def test_blank_lines_are_ignored(journal_path: Path) -> None:
    run_log.record("note", path=journal_path)
    with journal_path.open("a", encoding="utf-8") as handle:
        handle.write("\n\n")
    assert len(run_log.read_journal(journal_path)) == 1


def test_an_unknown_event_is_recorded_not_rejected(journal_path: Path) -> None:
    # Losing the entry would be worse than an odd label.
    run_log.record("something_new", path=journal_path)
    assert run_log.read_journal(journal_path)[0]["event"] == "something_new"


def test_a_refusal_leaves_a_trace(journal_path: Path) -> None:
    # The point of the `refused` event: a tripwire that correctly did nothing
    # still has to be visible afterwards.
    run_log.record(
        "refused",
        step="ensemble",
        detail="declined to unfreeze; asked the human instead",
        path=journal_path,
    )
    entry = run_log.read_journal(journal_path)[0]
    assert entry["event"] == "refused"
    assert "unfreeze" in entry["detail"]


def test_main_prints_entries(journal_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(paths, "RUN_LOG", journal_path)
    run_log.record("note", step="tune", detail="visible", path=journal_path)
    assert run_log.main([]) == 0
    assert "visible" in capsys.readouterr().out


def test_main_on_an_empty_journal(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(paths, "RUN_LOG", tmp_path / "nope.jsonl")
    assert run_log.main([]) == 0
    assert "no journal entries yet" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# severity levels and pass grouping
# --------------------------------------------------------------------------- #


def test_levels_run_least_to_most_urgent() -> None:
    assert run_log.LEVELS == ("debug", "info", "decision", "error")


def test_every_event_kind_has_a_default_level() -> None:
    assert set(run_log.EVENTS) <= set(run_log.DEFAULT_LEVEL)


def test_a_refusal_is_never_below_decision() -> None:
    # A tripwire that fired is exactly the thing a human must see, so it must
    # not be filterable out by the default view.
    for event in ("refused", "step_failed"):
        level = run_log.DEFAULT_LEVEL[event]
        assert run_log.LEVELS.index(level) >= run_log.LEVELS.index("decision"), event


def test_the_play_by_play_is_debug(journal_path: Path) -> None:
    assert run_log.record("step_started", path=journal_path)["level"] == "debug"


def test_an_explicit_level_overrides_the_default(journal_path: Path) -> None:
    entry = run_log.record("note", level="error", path=journal_path)
    assert entry["level"] == "error"


def test_entries_carry_a_pass_id(journal_path: Path) -> None:
    assert run_log.record("note", path=journal_path)["pass_id"]


def test_one_process_shares_one_pass_id(journal_path: Path) -> None:
    first = run_log.record("note", path=journal_path)["pass_id"]
    second = run_log.record("note", path=journal_path)["pass_id"]
    assert first == second


def test_the_environment_overrides_the_pass_id(journal_path: Path, monkeypatch) -> None:
    # This is how `make loop` makes three separate processes one pass.
    monkeypatch.setenv(run_log.PASS_ID_ENV, "pass-from-make")
    assert run_log.record("note", path=journal_path)["pass_id"] == "pass-from-make"


def test_main_filters_to_decisions(journal_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(paths, "RUN_LOG", journal_path)
    run_log.record("step_started", step="tune", detail="noise", path=journal_path)
    run_log.record("refused", step="freeze", detail="declined to unfreeze", path=journal_path)
    run_log.main(["--decisions"])
    out = capsys.readouterr().out
    assert "declined to unfreeze" in out
    assert "noise" not in out


def test_main_all_shows_the_debug_entries(journal_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(paths, "RUN_LOG", journal_path)
    run_log.record("step_started", step="tune", detail="noise", path=journal_path)
    run_log.main(["--all"])
    assert "noise" in capsys.readouterr().out


def test_an_entry_written_before_levels_existed_still_displays(
    journal_path: Path, capsys, monkeypatch
) -> None:
    # Backward compatibility: the journal predates the level field.
    monkeypatch.setattr(paths, "RUN_LOG", journal_path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "event": "note",
                "detail": "from before levels",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run_log.main([])
    assert "from before levels" in capsys.readouterr().out
