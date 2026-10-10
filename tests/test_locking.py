"""Smoke test: concurrency on the append-only artifacts.

These tests spawn real processes. Two Claude Code sessions in one repo is how
the races were found, so the tests reproduce that rather than simulating it —
an in-process mock of a file lock mostly tests the mock.

Each test asserts the *outcome* that matters, not that a lock was taken:

- the ledger's hash chain survives concurrent appends, because a forked chain is
  indistinguishable from a deleted row and would make ``make audit`` cry wolf;
- no holdout receipt is lost, because a lost receipt is how a second read passes
  itself off as the first.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from titanic_agent import locking, paths, receipts, results_log

WRITERS = 6
ROWS_EACH = 4


def _spawn(script: str, count: int) -> None:
    """Run ``count`` copies of ``script`` at once and wait for all of them."""
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(index)],
            cwd=paths.ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for index in range(count)
    ]
    for proc in procs:
        _, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, err.decode()[-2000:]


def test_the_lock_is_available_on_this_platform() -> None:
    # If this is ever False, the races below are unguarded and the tests that
    # follow would pass vacuously.
    assert locking.HAVE_FLOCK, "no fcntl: the append-only artifacts are unprotected here"


def test_concurrent_appends_keep_the_chain_intact(tmp_path: Path) -> None:
    ledger = tmp_path / "results.csv"
    script = textwrap.dedent(f"""
        import sys
        from titanic_agent import results_log
        for i in range({ROWS_EACH}):
            results_log.append_run(
                step="tune",
                model=f"writer{{sys.argv[1]}}-{{i}}",
                cv_score=0.5,
                path={str(ledger)!r},
            )
    """)
    _spawn(script, WRITERS)

    frame = results_log.read_log(ledger)
    assert len(frame) == WRITERS * ROWS_EACH, "an append was lost"

    report = results_log.verify_chain(ledger)
    assert report["ok"], report["problems"][:3]
    assert report["chained_rows"] == WRITERS * ROWS_EACH


def test_every_concurrent_row_is_distinct(tmp_path: Path) -> None:
    ledger = tmp_path / "results.csv"
    script = textwrap.dedent(f"""
        import sys
        from titanic_agent import results_log
        for i in range({ROWS_EACH}):
            results_log.append_run(
                step="tune", model="m", cv_score=0.5, path={str(ledger)!r}
            )
    """)
    _spawn(script, WRITERS)
    frame = results_log.read_log(ledger)
    assert frame["run_id"].nunique() == len(frame)
    # A forked chain shows up as a repeated prev_sha.
    assert frame["prev_sha"].nunique() == len(frame)


def test_no_holdout_receipt_is_lost_under_concurrency(tmp_path: Path) -> None:
    # The nastier race: a whole-array read-modify-write. A lost receipt makes a
    # second holdout read look like the first, which defeats Rule 2 outright.
    path = tmp_path / "holdout_reads.json"
    script = textwrap.dedent(f"""
        import sys
        from titanic_agent import receipts
        receipts.record_holdout_read(
            frozen_sha256="a" * 64,
            run_id=f"writer{{sys.argv[1]}}",
            path={str(path)!r},
        )
    """)
    _spawn(script, WRITERS)

    found = receipts.receipts_for("a" * 64, path)
    assert len(found) == WRITERS, f"{WRITERS - len(found)} receipt(s) lost"
    assert len({r["run_id"] for r in found}) == WRITERS


# --------------------------------------------------------------------------- #
# the lock itself
# --------------------------------------------------------------------------- #


def test_the_lock_is_reentrant_across_sequential_calls(tmp_path: Path) -> None:
    target = tmp_path / "artifact.csv"
    for _ in range(3):
        with locking.locked(target):
            target.write_text("ok", encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "ok"


def test_the_lock_file_sits_beside_the_target(tmp_path: Path) -> None:
    target = tmp_path / "artifact.csv"
    with locking.locked(target):
        assert (tmp_path / "artifact.csv.lock").exists()


def test_the_lock_is_released_after_the_block(tmp_path: Path) -> None:
    target = tmp_path / "artifact.csv"
    with locking.locked(target):
        pass
    with locking.locked(target, timeout_s=1.0):
        pass  # would raise LockTimeout if the first had not released


def test_an_exception_still_releases_the_lock(tmp_path: Path) -> None:
    target = tmp_path / "artifact.csv"
    with pytest.raises(ValueError), locking.locked(target):
        raise ValueError("boom")
    with locking.locked(target, timeout_s=1.0):
        pass


def test_waiting_too_long_raises_rather_than_hanging(tmp_path: Path) -> None:
    # A stuck holder should produce a named error that says what to do, not an
    # unattended run that blocks forever.
    target = tmp_path / "artifact.csv"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(f"""
                import time
                from titanic_agent import locking
                with locking.locked({str(target)!r}):
                    print("held", flush=True)
                    time.sleep(30)
            """),
        ],
        cwd=paths.ROOT,
        stdout=subprocess.PIPE,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == b"held"
        with (
            pytest.raises(locking.LockTimeout, match="another session"),
            locking.locked(target, timeout_s=0.3),
        ):
            pass
    finally:
        holder.kill()
        holder.wait(timeout=10)


def test_a_dead_holder_does_not_wedge_the_repo(tmp_path: Path) -> None:
    # flock releases on process death, so a crashed run cannot leave the repo
    # locked the way a stale lockfile would.
    target = tmp_path / "artifact.csv"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(f"""
                import time
                from titanic_agent import locking
                with locking.locked({str(target)!r}):
                    print("held", flush=True)
                    time.sleep(30)
            """),
        ],
        cwd=paths.ROOT,
        stdout=subprocess.PIPE,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == b"held"
    holder.kill()
    holder.wait(timeout=10)

    with locking.locked(target, timeout_s=5.0):
        pass  # the lock is free again despite the lockfile still existing
