"""A file lock for the append-only artifacts.

Two Claude Code sessions in the same repo is not a hypothetical — it happened,
and it exposed two races that had been sitting in the code since the artifacts
were written:

**The ledger forks.** :func:`titanic_agent.results_log.append_run` reads the last
line, hashes it into ``prev_sha``, and appends. Two processes interleaved there
both see the same last line and write the same ``prev_sha``. The chain forks,
and :func:`titanic_agent.results_log.verify_chain` cannot tell a fork from a
deletion — so ``make audit`` reports *"a row was deleted or edited"* about a
ledger nobody touched. A false alarm in the one mechanism whose entire job is
being trustworthy is worse than no mechanism: an audit that cries wolf gets
ignored.

**A holdout receipt is lost.** :func:`titanic_agent.receipts.record_holdout_read`
is a read-modify-write of a whole JSON array. Interleaved, the second write
clobbers the first and a receipt vanishes. That one is worse than a false alarm:
Rule 2 allows one holdout read per frozen dataset, and a lost receipt makes the
*second* read look like the first.

``flock`` is advisory and released automatically when the process dies, so a
crashed run cannot wedge the repo the way a stale lockfile would. On a platform
without ``fcntl`` the lock degrades to a no-op rather than failing closed: a
single-session run is the overwhelmingly common case, and refusing to log at all
would be a worse failure than a race nobody is currently causing.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

try:  # pragma: no cover - platform dependent
    import fcntl

    HAVE_FLOCK = True
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]
    HAVE_FLOCK = False

#: How long to wait for another process to finish its append. The critical
#: section is a read of one line plus a write of one line — microseconds — so a
#: wait this long means something is genuinely stuck, not merely busy.
DEFAULT_TIMEOUT_S: float = 10.0


class LockTimeout(RuntimeError):
    """Another process held the lock for longer than the timeout."""


@contextmanager
def locked(target: Path, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> Iterator[None]:
    """Hold an exclusive lock for ``target`` while the block runs.

    The lock lives in a sibling ``.lock`` file rather than on ``target`` itself,
    so that locking does not require opening the artifact for writing — a reader
    that only wants a consistent view can take the same lock.

    Args:
        target: The artifact being protected.
        timeout_s: How long to wait before giving up.

    Raises:
        LockTimeout: The lock could not be acquired in time.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_suffix(target.suffix + ".lock")

    if not HAVE_FLOCK:  # pragma: no cover - platform dependent
        yield
        return

    deadline = time.monotonic() + timeout_s
    handle = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise LockTimeout(
                        f"could not lock {target.name} within {timeout_s:g}s — another "
                        f"session is probably mid-write. Wait for it, or check for a "
                        f"stuck process; do not delete {lock_path.name} while one runs."
                    ) from None
                time.sleep(0.02)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        os.close(handle)
