"""Step 4 — ``freeze``: seal the dataset. This is the point of no return.

Everything after this step reads ``data/frozen/`` and never writes to it. The
manifest records a SHA-256 per file plus a combined dataset fingerprint, which
every logged experiment carries so a score can always be traced to the exact
bytes it was computed on.

Inputs
    ``data/raw/*.csv`` and ``data/frozen/split_plan.json``.
Outputs
    ``data/frozen/train.csv``, ``data/frozen/holdout.csv``,
    ``data/frozen/test.csv``, ``data/frozen/MANIFEST.json`` — all mode 0444.
Done criteria
    :func:`verify` returns True: every manifest entry exists and its checksum
    matches. Re-running ``freeze`` on an already-frozen directory **raises
    RuntimeError** — it is not a no-op, and there is no flag that makes it one.
    Use :func:`verify` (or ``make verify``) to re-check an existing freeze.

Because ``freeze`` refuses to run twice, it must never sit inside a loop that
reruns. ``make bootstrap`` covers steps 2-4 and is run once; ``make loop``
covers the repeatable steps 5-7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import paths


@dataclass(frozen=True)
class Manifest:
    """Contents of ``data/frozen/MANIFEST.json``."""

    frozen_at_utc: str = ""
    seed: int = paths.SEED
    git_sha: str = ""
    files: dict[str, str] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    dataset_sha256: str = ""
    manifest_path: Path | None = None


def freeze(
    *,
    raw_dir: Path = paths.RAW_DIR,
    frozen_dir: Path = paths.FROZEN_DIR,
    plan_path: Path = paths.SPLIT_PLAN,
    read_only: bool = True,
) -> Manifest:
    """Materialise the frozen dataset and write its manifest.

    Args:
        raw_dir: Source of the raw CSVs.
        frozen_dir: Destination. Must not already contain a valid manifest.
        plan_path: The split plan produced by :mod:`titanic_agent.split`.
        read_only: chmod the written files to 0444 as a guard rail against
            accidental edits. Set False only on filesystems that refuse it.

    Returns:
        The written :class:`Manifest`.

    Raises:
        RuntimeError: ``frozen_dir`` is already frozen. There is deliberately
            no ``force`` flag — unfreezing is a manual, deliberate act
            (``chmod -R u+w data/frozen && rm -rf data/frozen/*``) and it
            invalidates every score already in the results log.
    """
    # TODO(freeze): implement.
    #   1. If is_frozen(frozen_dir): raise RuntimeError -- no silent overwrite.
    #   2. Load the split plan; slice raw train into train.csv / holdout.csv
    #      by PassengerId; copy test.csv through unchanged.
    #   3. sha256 each written file; dataset_sha256 = sha256 of the sorted
    #      "name:sha" lines, so it is order-independent.
    #   4. Write MANIFEST.json, then chmod 0444 every file if read_only.
    #   5. Assert verify(frozen_dir) before returning.
    raise NotImplementedError("freeze() is not implemented yet")


def verify(frozen_dir: Path = paths.FROZEN_DIR) -> bool:
    """Re-hash the frozen files and compare against the manifest.

    Every downstream step calls this first. A False here means the frozen data
    drifted and all existing scores are suspect.
    """
    # TODO(freeze): implement manifest-vs-disk checksum comparison.
    raise NotImplementedError("verify() is not implemented yet")


def is_frozen(frozen_dir: Path = paths.FROZEN_DIR) -> bool:
    """True when ``frozen_dir`` holds a readable manifest. Never raises."""
    # TODO(freeze): return (frozen_dir / "MANIFEST.json").is_file() and parses.
    raise NotImplementedError("is_frozen() is not implemented yet")


def load_manifest(frozen_dir: Path = paths.FROZEN_DIR) -> Manifest:
    """Read ``MANIFEST.json`` into a :class:`Manifest`."""
    # TODO(freeze): implement JSON -> Manifest.
    raise NotImplementedError("load_manifest() is not implemented yet")


def require_frozen(frozen_dir: Path = paths.FROZEN_DIR) -> Manifest:
    """Guard used by explore/tune/ensemble: verify, or refuse to run.

    Raises:
        RuntimeError: The data is not frozen, or its checksums do not match.
    """
    # TODO(freeze): if not is_frozen() or not verify(): raise RuntimeError.
    raise NotImplementedError("require_frozen() is not implemented yet")


def main(argv: list[str] | None = None) -> int:
    """``make freeze`` entry point."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify-only", action="store_true", help="check checksums, write nothing")
    args = parser.parse_args(argv)

    if args.verify_only:
        ok = verify()
        print("frozen dataset OK" if ok else "FROZEN DATASET MISMATCH")
        return 0 if ok else 1

    manifest = freeze()
    print(
        f"frozen at {manifest.frozen_at_utc} "
        f"sha={manifest.dataset_sha256[:12]} rows={manifest.row_counts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
