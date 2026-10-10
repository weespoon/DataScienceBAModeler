"""Smoke test: step 4, ``freeze`` — the immutability guarantee.

The most important assertion in this file is the one about ``force``: there is
no such parameter, on purpose. Overwriting a frozen dataset invalidates every
score already in the ledger, so it cannot be a flag.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from _helpers import raises_real_runtime_error

from titanic_agent import freeze, paths


def test_public_api_exists() -> None:
    for name in ("freeze", "verify", "is_frozen", "load_manifest", "require_frozen", "main"):
        assert callable(getattr(freeze, name)), name


def test_signature_matches_the_documented_contract() -> None:
    params = inspect.signature(freeze.freeze).parameters
    assert set(params) == {"raw_dir", "frozen_dir", "plan_path", "read_only"}
    assert params["frozen_dir"].default == paths.FROZEN_DIR
    assert params["read_only"].default is True


def test_freeze_has_no_force_flag() -> None:
    assert "force" not in inspect.signature(freeze.freeze).parameters


def test_manifest_defaults_carry_a_fingerprint_field() -> None:
    manifest = freeze.Manifest()
    assert manifest.dataset_sha256 == ""
    assert manifest.files == {} and manifest.row_counts == {}
    assert manifest.seed == paths.SEED


@pytest.mark.xfail(raises=NotImplementedError, reason="step 4 not implemented yet")
def test_freeze_writes_a_manifest_and_verifies(tmp_path: Path, raw_dir: Path) -> None:
    frozen = tmp_path / "frozen"
    manifest = freeze.freeze(
        raw_dir=raw_dir, frozen_dir=frozen, plan_path=frozen / "split_plan.json"
    )
    assert (frozen / "MANIFEST.json").is_file()
    assert manifest.dataset_sha256
    assert freeze.verify(frozen) is True


@pytest.mark.xfail(raises=NotImplementedError, reason="step 4 not implemented yet")
def test_refreezing_is_refused(tmp_path: Path, raw_dir: Path) -> None:
    frozen = tmp_path / "frozen"
    freeze.freeze(raw_dir=raw_dir, frozen_dir=frozen, plan_path=frozen / "split_plan.json")
    with pytest.raises(RuntimeError):
        freeze.freeze(raw_dir=raw_dir, frozen_dir=frozen, plan_path=frozen / "split_plan.json")


@pytest.mark.xfail(raises=NotImplementedError, reason="step 4 not implemented yet")
def test_verify_catches_tampering(tmp_path: Path, raw_dir: Path) -> None:
    frozen = tmp_path / "frozen"
    freeze.freeze(
        raw_dir=raw_dir, frozen_dir=frozen, plan_path=frozen / "split_plan.json", read_only=False
    )
    target = frozen / "train.csv"
    target.write_text(target.read_text() + "\n999,1,3,X,male,30,0,0,T,1.0,,S\n", encoding="utf-8")
    assert freeze.verify(frozen) is False


@pytest.mark.xfail(raises=NotImplementedError, reason="step 4 not implemented yet")
def test_is_frozen_is_false_for_an_empty_dir(tmp_path: Path) -> None:
    assert freeze.is_frozen(tmp_path) is False


@pytest.mark.xfail(raises=NotImplementedError, reason="step 4 not implemented yet")
def test_require_frozen_refuses_an_unfrozen_dir(tmp_path: Path) -> None:
    with raises_real_runtime_error():
        freeze.require_frozen(tmp_path)
