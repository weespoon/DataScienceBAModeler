"""Holdout read receipts — the mechanism that makes Rule 2 enforceable.

Rule 2 says the holdout is read exactly once per frozen dataset. That rule was
unenforceable as originally written: ``ensemble`` promised to refuse a second
read, but nothing anywhere recorded that a read had happened. The receipt cannot
live in ``data/frozen/`` either, because Rule 1 forbids writing there.

**An immutable store cannot log its own access.** So the receipt lives here, in
``experiments/holdout_reads.json``, keyed by the frozen dataset's fingerprint.

What this does and does not buy you, stated honestly:

- It **prevents** a second read through :func:`require_unread`, which
  :mod:`titanic_agent.ensemble` calls before it touches ``holdout.csv``.
- It **detects** a second read after the fact, because two receipts under one
  fingerprint stay visible in ``make receipts`` where an honest run shows one.
- It does **not** prevent a stray ``pd.read_csv('data/frozen/holdout.csv')``
  somewhere it does not belong, and nothing cheap can: file access times are
  unreliable on APFS. That gap is why every ledger row also *attests*
  ``holdout_read``. Attestation does not stop the read; it converts a silent
  omission into an attributable false statement, which is the strongest
  guarantee available when prevention and detection both run out.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import paths


def read_receipts(path: Path | None = None) -> list[dict[str, Any]]:
    """Return every receipt on file, oldest first.

    A missing or unreadable file yields an empty list: the absence of receipts
    is a meaningful state (nothing has been read yet), not an error.
    """
    receipt_path = Path(path) if path is not None else paths.HOLDOUT_READS
    try:
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def receipts_for(frozen_sha256: str, path: Path | None = None) -> list[dict[str, Any]]:
    """Return the receipts recorded against one frozen fingerprint."""
    return [r for r in read_receipts(path) if r.get("frozen_sha256") == frozen_sha256]


def has_been_read(frozen_sha256: str, path: Path | None = None) -> bool:
    """True when this frozen dataset's holdout has already been read."""
    return bool(receipts_for(frozen_sha256, path))


def require_unread(frozen_sha256: str, path: Path | None = None) -> None:
    """Refuse to proceed if this holdout has already been read.

    Raises:
        RuntimeError: A receipt already exists for ``frozen_sha256``. The
            message names the earlier read, because the useful response is to
            report the existing number, not to take a second measurement.
    """
    existing = receipts_for(frozen_sha256, path)
    if not existing:
        return
    first = existing[0]
    raise RuntimeError(
        f"the holdout for frozen dataset {frozen_sha256[:12]} was already read at "
        f"{first.get('timestamp_utc', 'an unrecorded time')} by run "
        f"{first.get('run_id', 'unknown')}. Rule 2 allows one read per frozen "
        f"dataset. Report that score; do not take a second measurement."
    )


def record_holdout_read(
    *,
    frozen_sha256: str,
    run_id: str,
    score: float | None = None,
    reason: str = "",
    path: Path | None = None,
) -> dict[str, Any]:
    """Append a receipt for a holdout read and return it.

    Call this **after** the read succeeds, from the code that did the reading.
    Appending rather than overwriting is deliberate: a second read must leave a
    second receipt, so that a rule-2 violation is visible afterwards even though
    :func:`require_unread` was supposed to make it impossible.
    """
    receipt_path = Path(path) if path is not None else paths.HOLDOUT_READS
    receipt_path.parent.mkdir(parents=True, exist_ok=True)

    receipt = {
        "frozen_sha256": frozen_sha256,
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "score": None if score is None else round(float(score), 6),
        "reason": reason.replace("\n", " ").strip(),
    }
    existing = read_receipts(receipt_path)
    existing.append(receipt)
    receipt_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv: list[str] | None = None) -> int:
    """``python -m titanic_agent.receipts`` — show the holdout read receipts."""
    import argparse

    parser = argparse.ArgumentParser(description="Show holdout read receipts.")
    parser.add_argument("--frozen-sha", default=None, help="filter to one fingerprint")
    args = parser.parse_args(argv)

    found = receipts_for(args.frozen_sha) if args.frozen_sha else read_receipts()
    if not found:
        print(f"no holdout reads recorded ({paths.HOLDOUT_READS})")
        return 0
    for receipt in found:
        score = "n/a" if receipt.get("score") is None else f"{receipt['score']:.4f}"
        print(
            f"{receipt.get('timestamp_utc', '?')}  "
            f"frozen={str(receipt.get('frozen_sha256', '?'))[:12]}  "
            f"run={receipt.get('run_id', '?')}  score={score}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
