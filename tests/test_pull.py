"""Smoke test: step 1, ``pull``.

The step is still a stub, so these tests pin the contract rather than the
behaviour: the public API exists with the documented signature, and the
not-yet-implemented calls are marked xfail so they flip to passing on their own
once the step is written.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from titanic_agent import paths, pull


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


@pytest.mark.xfail(raises=NotImplementedError, reason="step 1 not implemented yet")
def test_pull_accepts_a_local_zip(tmp_path: Path) -> None:
    result = pull.pull(tmp_path, local_zip=tmp_path / "titanic.zip")
    assert {"train.csv", "test.csv"} <= set(result.files)


@pytest.mark.xfail(raises=NotImplementedError, reason="step 1 not implemented yet")
def test_verify_accepts_a_well_formed_raw_dir(raw_dir: Path) -> None:
    assert pull.verify(raw_dir) is True


@pytest.mark.xfail(raises=NotImplementedError, reason="step 1 not implemented yet")
def test_verify_rejects_an_empty_dir(tmp_path: Path) -> None:
    assert pull.verify(tmp_path) is False
