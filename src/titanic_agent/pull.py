"""Step 1 — ``pull``: fetch the raw competition data into ``data/raw/``.

Inputs
    Kaggle credentials (``~/.kaggle/kaggle.json`` or ``KAGGLE_USERNAME`` /
    ``KAGGLE_KEY``), or a manually downloaded zip handed in via ``local_zip``.
Outputs
    ``data/raw/train.csv``, ``data/raw/test.csv`` (gitignored).
Done criteria
    Both files exist, are non-empty, parse as CSV, and ``train.csv`` carries a
    ``Survived`` column. Nothing downstream of this step writes to ``data/raw``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import paths


@dataclass(frozen=True)
class PullResult:
    """What a pull produced."""

    files: dict[str, Path] = field(default_factory=dict)
    source: str = ""
    n_rows: dict[str, int] = field(default_factory=dict)
    skipped: bool = False


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
        FileNotFoundError: No credentials and no ``local_zip``.
        RuntimeError: The download succeeded but failed :func:`verify`.
    """
    # TODO(pull): implement.
    #   1. If not force and verify(dest) passes -> return PullResult(skipped=True).
    #   2. If local_zip: extract it into dest.
    #      Else: shell out to `kaggle competitions download -c {competition} -p {dest}`
    #      (the Kaggle CLI is an optional extra, see pyproject.toml) and unzip.
    #   3. Never commit or log the credential contents.
    #   4. Call verify(dest) and raise RuntimeError on failure.
    raise NotImplementedError("pull() is not implemented yet")


def verify(raw_dir: Path = paths.RAW_DIR) -> bool:
    """Return True when ``raw_dir`` holds a usable raw dataset.

    Checks existence, non-emptiness, parseability and the presence of the
    ``Survived`` target in the training file.
    """
    # TODO(pull): implement the four checks named in the docstring.
    raise NotImplementedError("verify() is not implemented yet")


def main(argv: list[str] | None = None) -> int:
    """``make pull`` entry point."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=paths.RAW_DIR)
    parser.add_argument("--local-zip", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    result = pull(args.dest, local_zip=args.local_zip, force=args.force)
    print(f"pulled from {result.source}: {sorted(result.files)} rows={result.n_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
