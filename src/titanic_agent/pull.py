"""Step 1 — ``pull``: fetch the raw competition data into ``data/raw/``.

Inputs
    Kaggle credentials, in any of the shapes the client supports:
    ``KAGGLE_API_TOKEN``, ``~/.kaggle/access_token``, a ``kaggle auth login``
    session, or a legacy ``~/.kaggle/kaggle.json``. Or no credentials at all,
    with a manually downloaded archive handed in via ``local_zip``.
Outputs
    ``data/raw/train.csv``, ``data/raw/test.csv`` (gitignored).
Done criteria
    Both files exist, are non-empty, parse as CSV, and ``train.csv`` carries a
    ``Survived`` column. Nothing downstream of this step writes to ``data/raw``.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from . import paths

#: Files this step must produce. ``gender_submission.csv`` also ships in the
#: archive; it is extracted but not required.
REQUIRED_FILES: tuple[str, ...] = ("train.csv", "test.csv")

#: Row counts Kaggle currently serves. Used for a warning, not a hard check --
#: the competition data is stable, but failing a pull over an upstream change
#: would be worse than saying so and carrying on.
EXPECTED_ROWS: dict[str, int] = {"train.csv": 891, "test.csv": 418}


@dataclass(frozen=True)
class PullResult:
    """What a pull produced."""

    files: dict[str, Path] = field(default_factory=dict)
    source: str = ""
    n_rows: dict[str, int] = field(default_factory=dict)
    skipped: bool = False


def _row_counts(raw_dir: Path) -> dict[str, int]:
    """Row count per required file, excluding the header."""
    import pandas as pd

    return {name: len(pd.read_csv(raw_dir / name)) for name in REQUIRED_FILES}


def _extract(archive: Path, dest: Path) -> None:
    """Extract ``archive`` into ``dest``, flattening any directory structure.

    Guards against path traversal: entries are written by basename only, so a
    crafted archive cannot escape ``dest``.
    """
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if member.is_dir():
                continue
            name = Path(member.filename).name
            if not name or name.startswith("."):
                continue
            with bundle.open(member) as src, (dest / name).open("wb") as out:
                out.write(src.read())


def _download_from_kaggle(competition: str, dest: Path) -> None:
    """Download and unpack the competition archive via the Kaggle API.

    Imported lazily so the rest of the pipeline runs without the optional
    ``kaggle`` extra installed.

    Raises:
        FileNotFoundError: The ``kaggle`` package is not installed.
        RuntimeError: Authentication or download failed.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise FileNotFoundError(
            "the kaggle client is not installed; run `make setup EXTRAS=dev,kaggle` "
            "or pass --local-zip to skip the download"
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        # Deliberately not echoing the exception's payload: the client can
        # include configuration detail, and credentials must never reach stdout.
        raise RuntimeError(
            f"Kaggle authentication failed ({type(exc).__name__}). Set KAGGLE_API_TOKEN, "
            "save a token to ~/.kaggle/access_token, or run `kaggle auth login`."
        ) from None

    api.competition_download_files(competition, path=str(dest), quiet=True)

    archives = sorted(dest.glob("*.zip"))
    if not archives:
        raise RuntimeError(f"Kaggle reported success but no archive landed in {dest}")
    for archive in archives:
        _extract(archive, dest)
        archive.unlink()


def pull(
    dest: Path = paths.RAW_DIR,
    *,
    competition: str = paths.COMPETITION,
    local_zip: Path | None = None,
    force: bool = False,
) -> PullResult:
    """Download the Titanic dataset into ``dest``.

    Args:
        dest: Directory to write the raw CSVs into. Created if absent.
        competition: Kaggle competition slug to download.
        local_zip: Use this already-downloaded archive instead of calling the
            Kaggle API. Lets the pipeline run with no credentials at all.
        force: Re-download even when ``dest`` already holds valid files.

    Returns:
        A :class:`PullResult` describing the files on disk.

    Raises:
        FileNotFoundError: No credentials and no ``local_zip``, or the given
            ``local_zip`` does not exist.
        RuntimeError: The download succeeded but failed :func:`verify`.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if not force and verify(dest):
        return PullResult(
            files={name: dest / name for name in REQUIRED_FILES},
            source="cache",
            n_rows=_row_counts(dest),
            skipped=True,
        )

    if local_zip is not None:
        local_zip = Path(local_zip).expanduser()
        if not local_zip.is_file():
            raise FileNotFoundError(f"no such archive: {local_zip}")
        _extract(local_zip, dest)
        source = f"local_zip:{local_zip.name}"
    else:
        _download_from_kaggle(competition, dest)
        source = f"kaggle:{competition}"

    if not verify(dest):
        missing = [name for name in REQUIRED_FILES if not (dest / name).is_file()]
        detail = f"missing {missing}" if missing else "files are present but unusable"
        raise RuntimeError(f"pull from {source} did not produce a usable dataset: {detail}")

    n_rows = _row_counts(dest)
    for name, expected in EXPECTED_ROWS.items():
        if n_rows[name] != expected:
            print(f"warning: {name} has {n_rows[name]} rows, expected {expected}")

    return PullResult(
        files={name: dest / name for name in REQUIRED_FILES},
        source=source,
        n_rows=n_rows,
    )


def verify(raw_dir: Path = paths.RAW_DIR) -> bool:
    """Return True when ``raw_dir`` holds a usable raw dataset.

    Checks existence, non-emptiness, parseability and the presence of the
    ``Survived`` target in the training file. Never raises — a malformed or
    absent dataset is a False, not an exception, so callers can branch on it.
    """
    import pandas as pd

    raw_dir = Path(raw_dir)
    for name in REQUIRED_FILES:
        path = raw_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            return False

    try:
        train = pd.read_csv(raw_dir / "train.csv")
        test = pd.read_csv(raw_dir / "test.csv")
    except (OSError, ValueError, pd.errors.ParserError):
        return False

    if train.empty or test.empty:
        return False
    if "Survived" not in train.columns:
        return False
    # The target must not leak into the test file -- if it is there, this is not
    # the competition dataset and every downstream score would be meaningless.
    return "Survived" not in test.columns


def main(argv: list[str] | None = None) -> int:
    """``make pull`` entry point."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=paths.RAW_DIR)
    parser.add_argument("--local-zip", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = pull(args.dest, local_zip=args.local_zip, force=args.force)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"pull failed: {exc}")
        return 1

    verb = "already present" if result.skipped else f"pulled from {result.source}"
    rows = ", ".join(f"{name}={count}" for name, count in sorted(result.n_rows.items()))
    print(f"{verb}: {rows} -> {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
