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

Note that ``chmod 0444`` is a guard rail, not the enforcement: it does not stop
the file's owner, who can simply ``chmod u+w``. The actual constraint is the
``PreToolUse`` hook in ``.claude/hooks/protect_frozen.py``, which denies the
agent any write here once ``MANIFEST.json`` exists.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import stat
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from . import paths

#: Files the manifest covers. ``split_plan.json`` is included on purpose: the
#: provenance of the partition is part of what is being sealed, so a later edit
#: to the plan shows up as a checksum mismatch rather than passing unnoticed.
MANIFEST_FILES: tuple[str, ...] = (
    "train.csv",
    "holdout.csv",
    "test.csv",
    "split_plan.json",
)

#: Read-only for everyone.
READ_ONLY = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH  # 0o444


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


def _sha256(path: Path) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_fingerprint(files: dict[str, str]) -> str:
    """Combine per-file hashes into one order-independent fingerprint.

    Hashes the sorted ``name:sha`` lines, so the order files happened to be
    written in cannot change the result.
    """
    body = "\n".join(f"{name}:{sha}" for name, sha in sorted(files.items()))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def is_frozen(frozen_dir: Path = paths.FROZEN_DIR) -> bool:
    """True when ``frozen_dir`` holds a readable manifest. Never raises.

    This is the single source of truth for "is the dataset sealed", read by
    :mod:`titanic_agent.split`, by :func:`require_frozen`, and — deliberately —
    by the ``PreToolUse`` hook, so the invariant lives in one place.
    """
    manifest_path = Path(frozen_dir) / paths.MANIFEST.name
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(isinstance(payload, dict) and payload.get("dataset_sha256"))


def load_manifest(frozen_dir: Path = paths.FROZEN_DIR) -> Manifest:
    """Read ``MANIFEST.json`` into a :class:`Manifest`.

    Raises:
        FileNotFoundError: The dataset is not frozen.
        ValueError: The manifest is present but malformed.
    """
    manifest_path = Path(frozen_dir) / paths.MANIFEST.name
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{manifest_path} is missing — the dataset is not frozen. Run `make freeze`."
        ) from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest_path} is not valid JSON: {exc}") from None

    try:
        return Manifest(
            frozen_at_utc=str(payload["frozen_at_utc"]),
            seed=int(payload["seed"]),
            git_sha=str(payload.get("git_sha", "unknown")),
            files={str(k): str(v) for k, v in payload["files"].items()},
            row_counts={str(k): int(v) for k, v in payload.get("row_counts", {}).items()},
            dataset_sha256=str(payload["dataset_sha256"]),
            manifest_path=manifest_path,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{manifest_path} has a missing or malformed field: {exc}") from None


def verify(frozen_dir: Path = paths.FROZEN_DIR) -> bool:
    """Re-hash the frozen files and compare against the manifest.

    Every downstream step calls this first. A False here means the frozen data
    drifted and all existing scores are suspect. Never raises — an unfrozen or
    malformed directory is a False, so callers can branch on it.
    """
    frozen_dir = Path(frozen_dir)
    try:
        manifest = load_manifest(frozen_dir)
    except (FileNotFoundError, ValueError):
        return False

    for name, expected in manifest.files.items():
        path = frozen_dir / name
        if not path.is_file():
            return False
        try:
            if _sha256(path) != expected:
                return False
        except OSError:
            return False

    # The fingerprint must also agree with the per-file hashes, so editing the
    # manifest's own `files` map cannot quietly re-bless tampered data.
    return _dataset_fingerprint(manifest.files) == manifest.dataset_sha256


def require_frozen(frozen_dir: Path = paths.FROZEN_DIR) -> Manifest:
    """Guard used by explore/tune/ensemble: verify, or refuse to run.

    Raises:
        RuntimeError: The data is not frozen, or its checksums do not match.
    """
    frozen_dir = Path(frozen_dir)
    if not is_frozen(frozen_dir):
        raise RuntimeError(
            f"{frozen_dir} is not frozen. Steps 5-7 read frozen data only — "
            "run `make bootstrap` (review, split, freeze) first."
        )
    if not verify(frozen_dir):
        raise RuntimeError(
            f"the frozen dataset in {frozen_dir} does not match its manifest. "
            "Every score computed against it is suspect. This is information, not "
            "an obstacle: investigate and report the mismatch. Do not re-freeze — "
            "that is a human decision."
        )
    return load_manifest(frozen_dir)


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
            no ``force`` flag — unfreezing is a manual, deliberate act and it
            invalidates every score already in the results log. See
            ``docs/UNFREEZING.md``, which is addressed to a human.
        FileNotFoundError: The raw data or the split plan is missing.
        ValueError: The plan's ids do not match the raw training data.
    """
    import pandas as pd

    from . import results_log
    from . import split as split_module

    frozen_dir = Path(frozen_dir)
    if is_frozen(frozen_dir):
        manifest = load_manifest(frozen_dir)
        raise RuntimeError(
            f"{frozen_dir} was already frozen at {manifest.frozen_at_utc} "
            f"(sha={manifest.dataset_sha256[:12]}). Re-freezing would invalidate every "
            "score in experiments/results.csv. There is no force flag. Use "
            "`make verify` to re-check the existing freeze."
        )

    raw_dir = Path(raw_dir)
    raw_train, raw_test = raw_dir / "train.csv", raw_dir / "test.csv"
    for path in (raw_train, raw_test):
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing — run `make pull` first")

    plan = split_module.load_plan(plan_path)

    train_raw = pd.read_csv(raw_train)
    indexed = train_raw.set_index(plan.id_column, drop=False)
    all_ids = {int(i) for i in train_raw[plan.id_column]}
    planned = set(plan.train_ids) | set(plan.holdout_ids)
    if planned != all_ids:
        missing, extra = all_ids - planned, planned - all_ids
        raise ValueError(
            f"the split plan does not cover {raw_train}: "
            f"{len(missing)} unplanned id(s), {len(extra)} unknown id(s). "
            "Redraw the split before freezing."
        )

    frozen_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    row_counts: dict[str, int] = {}

    for name, ids in (("train.csv", plan.train_ids), ("holdout.csv", plan.holdout_ids)):
        part = indexed.loc[ids].reset_index(drop=True)
        destination = frozen_dir / name
        part.to_csv(destination, index=False)
        written[name] = destination
        row_counts[name] = len(part)

    # The competition test set passes through unchanged, so a byte comparison
    # against data/raw stays meaningful.
    test_destination = frozen_dir / "test.csv"
    test_destination.write_bytes(raw_test.read_bytes())
    written["test.csv"] = test_destination
    row_counts["test.csv"] = len(pd.read_csv(test_destination))

    # The plan itself is sealed alongside the data it describes.
    written["split_plan.json"] = Path(plan_path)

    files = {name: _sha256(path) for name, path in written.items()}
    manifest = Manifest(
        frozen_at_utc=datetime.now(UTC).replace(microsecond=0).isoformat(),
        seed=plan.seed,
        git_sha=results_log._git_sha(),
        files=files,
        row_counts=row_counts,
        dataset_sha256=_dataset_fingerprint(files),
        manifest_path=frozen_dir / paths.MANIFEST.name,
    )

    manifest.manifest_path.write_text(
        json.dumps(
            {
                "frozen_at_utc": manifest.frozen_at_utc,
                "seed": manifest.seed,
                "git_sha": manifest.git_sha,
                "dataset_sha256": manifest.dataset_sha256,
                "row_counts": manifest.row_counts,
                "files": manifest.files,
                "note": (
                    "Immutable. See CLAUDE.md rule 1. Unfreezing is a human action; "
                    "it invalidates every ledger row carrying this dataset_sha256."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if read_only:
        # Guard rail only. The real constraint is the PreToolUse hook, which
        # starts denying writes here the moment the manifest above exists.
        for path in (*written.values(), manifest.manifest_path):
            # Some filesystems refuse chmod; the manifest and the hook still
            # protect us, so a failure here is not fatal.
            with contextlib.suppress(OSError):
                path.chmod(READ_ONLY)

    if not verify(frozen_dir):
        raise RuntimeError(
            "freeze wrote the dataset but verification failed immediately afterwards; "
            "do not trust this directory"
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    """``make freeze`` entry point."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify-only", action="store_true", help="check checksums, write nothing")
    parser.add_argument("--frozen-dir", type=Path, default=paths.FROZEN_DIR)
    args = parser.parse_args(argv)

    if args.verify_only:
        ok = verify(args.frozen_dir)
        if ok:
            manifest = load_manifest(args.frozen_dir)
            print(
                f"frozen dataset OK: sha={manifest.dataset_sha256[:12]} "
                f"frozen_at={manifest.frozen_at_utc} rows={manifest.row_counts}"
            )
            return 0
        print(
            "FROZEN DATASET MISMATCH (or not frozen). Every score computed against it "
            "is suspect. Investigate and report — do not re-freeze."
        )
        return 1

    try:
        manifest = freeze(frozen_dir=args.frozen_dir)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"freeze failed: {exc}")
        return 1

    print(
        f"frozen at {manifest.frozen_at_utc} "
        f"sha={manifest.dataset_sha256[:12]} rows={manifest.row_counts}"
    )
    print("data/frozen is now immutable. Steps 5-7 read it; nothing writes to it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
