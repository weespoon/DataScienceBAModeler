"""Smoke test: step 1, ``pull``.

Step 1 is implemented, so these are real tests. None of them touch the network:
every path through :func:`titanic_agent.pull.pull` that these exercise goes
through ``local_zip`` or the cache short-circuit. The Kaggle download itself is
covered by actually running ``make pull``, not by a mock that would only assert
that the mock was called.
"""

from __future__ import annotations

import inspect
import zipfile
from pathlib import Path

import pytest

from titanic_agent import paths, pull


def _zip_of(source_dir: Path, archive: Path, names: list[str] | None = None) -> Path:
    """Bundle the named files from ``source_dir`` into ``archive``."""
    names = names if names is not None else [p.name for p in source_dir.glob("*.csv")]
    with zipfile.ZipFile(archive, "w") as bundle:
        for name in names:
            bundle.write(source_dir / name, arcname=name)
    return archive


# --------------------------------------------------------------------------- #
# contract
# --------------------------------------------------------------------------- #


def test_public_api_exists() -> None:
    assert callable(pull.pull)
    assert callable(pull.verify)
    assert callable(pull.main)


def test_signature_matches_the_documented_contract() -> None:
    params = inspect.signature(pull.pull).parameters
    assert set(params) == {"dest", "competition", "local_zip", "force"}
    assert params["dest"].default == paths.RAW_DIR
    assert params["competition"].default == paths.COMPETITION
    assert params["local_zip"].default is None
    assert params["force"].default is False


def test_result_dataclass_defaults_are_empty() -> None:
    result = pull.PullResult()
    assert result.files == {} and result.n_rows == {} and result.skipped is False


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #


def test_verify_accepts_a_well_formed_raw_dir(raw_dir: Path) -> None:
    assert pull.verify(raw_dir) is True


def test_verify_rejects_an_empty_dir(tmp_path: Path) -> None:
    assert pull.verify(tmp_path) is False


def test_verify_rejects_a_missing_dir(tmp_path: Path) -> None:
    assert pull.verify(tmp_path / "nope") is False


def test_verify_rejects_a_zero_byte_file(raw_dir: Path) -> None:
    (raw_dir / "test.csv").write_text("", encoding="utf-8")
    assert pull.verify(raw_dir) is False


def test_verify_rejects_a_header_only_file(raw_dir: Path) -> None:
    (raw_dir / "test.csv").write_text("PassengerId,Pclass\n", encoding="utf-8")
    assert pull.verify(raw_dir) is False


def test_verify_rejects_training_data_with_no_target(raw_dir: Path) -> None:
    import pandas as pd

    frame = pd.read_csv(raw_dir / "train.csv").drop(columns=["Survived"])
    frame.to_csv(raw_dir / "train.csv", index=False)
    assert pull.verify(raw_dir) is False


def test_verify_rejects_a_target_leaking_into_test(raw_dir: Path) -> None:
    import pandas as pd

    frame = pd.read_csv(raw_dir / "test.csv")
    frame["Survived"] = 0
    frame.to_csv(raw_dir / "test.csv", index=False)
    assert pull.verify(raw_dir) is False


def test_verify_rejects_unparseable_csv(raw_dir: Path) -> None:
    (raw_dir / "train.csv").write_text('a,b\n"unterminated,1\n2,3,4,5\n', encoding="utf-8")
    assert pull.verify(raw_dir) is False


def test_verify_never_raises(tmp_path: Path) -> None:
    # A directory where a CSV should be is the nastiest shape to hand pandas.
    (tmp_path / "train.csv").mkdir()
    (tmp_path / "test.csv").mkdir()
    assert pull.verify(tmp_path) is False


# --------------------------------------------------------------------------- #
# pull, via local_zip
# --------------------------------------------------------------------------- #


def test_pull_extracts_a_local_zip(raw_dir: Path, tmp_path: Path) -> None:
    archive = _zip_of(raw_dir, tmp_path / "titanic.zip")
    dest = tmp_path / "out"
    result = pull.pull(dest, local_zip=archive)
    assert set(result.files) == {"train.csv", "test.csv"}
    assert all(path.is_file() for path in result.files.values())
    assert result.source == "local_zip:titanic.zip"
    assert result.skipped is False
    assert result.n_rows == {"train.csv": 12, "test.csv": 4}


def test_pull_creates_a_missing_destination(raw_dir: Path, tmp_path: Path) -> None:
    archive = _zip_of(raw_dir, tmp_path / "titanic.zip")
    dest = tmp_path / "deep" / "nested" / "raw"
    pull.pull(dest, local_zip=archive)
    assert (dest / "train.csv").is_file()


def test_pull_is_a_noop_when_already_valid(raw_dir: Path, tmp_path: Path) -> None:
    before = (raw_dir / "train.csv").read_bytes()
    result = pull.pull(raw_dir, local_zip=tmp_path / "does-not-exist.zip")
    assert result.skipped is True
    assert result.source == "cache"
    assert (raw_dir / "train.csv").read_bytes() == before


def test_force_bypasses_the_cache(raw_dir: Path, tmp_path: Path) -> None:
    archive = _zip_of(raw_dir, tmp_path / "titanic.zip")
    result = pull.pull(raw_dir, local_zip=archive, force=True)
    assert result.skipped is False
    assert result.source == "local_zip:titanic.zip"


def test_pull_rejects_a_missing_archive(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pull.pull(tmp_path / "out", local_zip=tmp_path / "nope.zip")


def test_pull_raises_when_the_archive_is_incomplete(raw_dir: Path, tmp_path: Path) -> None:
    archive = _zip_of(raw_dir, tmp_path / "partial.zip", names=["train.csv"])
    with pytest.raises(RuntimeError, match="test.csv"):
        pull.pull(tmp_path / "out", local_zip=archive)


def test_extraction_flattens_nested_paths_and_cannot_escape(raw_dir: Path, tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(raw_dir / "train.csv", arcname="nested/dir/train.csv")
        bundle.write(raw_dir / "test.csv", arcname="../../../../escaped/test.csv")
    dest = tmp_path / "out"
    pull.pull(dest, local_zip=archive)
    assert (dest / "train.csv").is_file()
    assert (dest / "test.csv").is_file()
    assert not (tmp_path.parent / "escaped").exists()


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def test_main_returns_zero_on_success(raw_dir: Path, tmp_path: Path, capsys) -> None:
    archive = _zip_of(raw_dir, tmp_path / "titanic.zip")
    dest = tmp_path / "out"
    assert pull.main(["--dest", str(dest), "--local-zip", str(archive)]) == 0
    assert "train.csv=12" in capsys.readouterr().out


def test_main_reports_failure_without_a_traceback(tmp_path: Path, capsys) -> None:
    code = pull.main(["--dest", str(tmp_path / "out"), "--local-zip", str(tmp_path / "nope.zip")])
    assert code == 1
    assert "pull failed" in capsys.readouterr().out
