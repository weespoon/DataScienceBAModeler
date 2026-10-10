"""Smoke test: pipeline status.

Status reads the real repo (it derives state from artifacts on disk), so these
tests assert the *shape* of what it reports rather than specific done-flags,
which legitimately change as the pipeline advances.
"""

from __future__ import annotations

from titanic_agent import status


def test_reports_all_seven_steps_in_order() -> None:
    rows = status.pipeline_status()
    assert [row.number for row in rows] == [1, 2, 3, 4, 5, 6, 7]
    assert [row.name for row in rows] == [
        "pull",
        "review",
        "split",
        "freeze",
        "explore",
        "tune",
        "ensemble",
    ]


def test_every_step_carries_a_detail_string() -> None:
    assert all(row.detail for row in status.pipeline_status())


def test_next_step_is_the_first_incomplete_one() -> None:
    rows = status.pipeline_status()
    upcoming = status.next_step()
    incomplete = [row for row in rows if not row.done]
    if not incomplete:
        assert upcoming is None
    else:
        assert upcoming == incomplete[0]


def test_status_never_raises_on_a_partial_pipeline() -> None:
    # Most steps are unimplemented stubs; status must still answer. It presence-
    # checks artifacts rather than calling the step functions, which is the
    # whole reason it can run at any point.
    assert len(status.pipeline_status()) == 7


def test_main_prints_a_checklist_and_a_next_action(capsys) -> None:
    assert status.main([]) == 0
    out = capsys.readouterr().out
    assert "pull" in out
    assert "next:" in out


def test_main_next_prints_a_single_step(capsys) -> None:
    assert status.main(["--next"]) == 0
    out = capsys.readouterr().out.strip()
    assert "\n" not in out
