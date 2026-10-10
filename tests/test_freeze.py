"""Smoke test: step 4, ``freeze`` — the immutability guarantee.

Every test freezes into ``tmp_path``. The repo's own ``data/frozen`` is never
written, re-frozen or chmod'd by a test run.

Three tests carry most of the weight:

- :func:`test_freeze_has_no_force_flag` — there is no such parameter, on
  purpose. Overwriting a frozen dataset invalidates every score already in the
  ledger, so it cannot be a flag.
- :func:`test_verify_catches_tampering` — the detect half of the guarantee,
  for everything prevention does not cover.
- :func:`test_editing_the_manifest_cannot_re_bless_tampered_data` — the
  fingerprint is checked against the per-file hashes, so a tamperer has to
  forge both and still fails.
"""

from __future__ import annotations

import inspect
import json
import stat
from pathlib import Path

import pytest
from _helpers import raises_real_runtime_error

from titanic_agent import freeze, paths, split

# --------------------------------------------------------------------------- #
# contract
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# a successful freeze
# --------------------------------------------------------------------------- #


def test_freeze_writes_every_declared_artifact(frozen_dir: Path) -> None:
    for name in ("train.csv", "holdout.csv", "test.csv", "split_plan.json", "MANIFEST.json"):
        assert (frozen_dir / name).is_file(), name


def test_freeze_verifies_immediately_afterwards(frozen_dir: Path) -> None:
    assert freeze.verify(frozen_dir) is True
    assert freeze.is_frozen(frozen_dir) is True


def test_files_are_read_only(frozen_dir: Path) -> None:
    for name in freeze.MANIFEST_FILES + ("MANIFEST.json",):
        mode = stat.S_IMODE((frozen_dir / name).stat().st_mode)
        assert mode == 0o444, f"{name} is {oct(mode)}"


def test_the_split_plan_is_sealed_too(frozen_dir: Path) -> None:
    # The provenance of the partition is part of what is frozen, so a later
    # edit to the plan shows up as a mismatch rather than passing unnoticed.
    manifest = freeze.load_manifest(frozen_dir)
    assert "split_plan.json" in manifest.files


def test_row_counts_sum_to_the_raw_training_rows(frozen_dir: Path) -> None:
    manifest = freeze.load_manifest(frozen_dir)
    assert manifest.row_counts["train.csv"] + manifest.row_counts["holdout.csv"] == 12
    assert manifest.row_counts["test.csv"] == 4


def test_the_test_set_passes_through_unchanged(frozen_dir: Path, raw_dir: Path) -> None:
    assert (frozen_dir / "test.csv").read_bytes() == (raw_dir / "test.csv").read_bytes()


def test_train_and_holdout_hold_exactly_the_planned_rows(frozen_dir: Path) -> None:
    import pandas as pd

    plan = split.load_plan(frozen_dir / "split_plan.json")
    train = pd.read_csv(frozen_dir / "train.csv")
    holdout = pd.read_csv(frozen_dir / "holdout.csv")
    assert sorted(train["PassengerId"]) == plan.train_ids
    assert sorted(holdout["PassengerId"]) == plan.holdout_ids


def test_the_manifest_records_provenance(frozen_dir: Path) -> None:
    manifest = freeze.load_manifest(frozen_dir)
    assert manifest.frozen_at_utc.endswith("+00:00")
    assert manifest.dataset_sha256
    assert manifest.git_sha


def test_the_fingerprint_is_order_independent(frozen_dir: Path) -> None:
    manifest = freeze.load_manifest(frozen_dir)
    shuffled = dict(reversed(list(manifest.files.items())))
    assert freeze._dataset_fingerprint(shuffled) == manifest.dataset_sha256


def test_the_fingerprint_changes_when_a_file_hash_does(frozen_dir: Path) -> None:
    manifest = freeze.load_manifest(frozen_dir)
    altered = dict(manifest.files, **{"train.csv": "0" * 64})
    assert freeze._dataset_fingerprint(altered) != manifest.dataset_sha256


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #


def test_refreezing_is_refused(frozen_dir: Path, raw_dir: Path) -> None:
    with pytest.raises(RuntimeError, match="already frozen"):
        freeze.freeze(
            raw_dir=raw_dir,
            frozen_dir=frozen_dir,
            plan_path=frozen_dir / "split_plan.json",
        )


def test_refreezing_leaves_the_dataset_intact(frozen_dir: Path, raw_dir: Path) -> None:
    before = (frozen_dir / "train.csv").read_bytes()
    with pytest.raises(RuntimeError):
        freeze.freeze(
            raw_dir=raw_dir,
            frozen_dir=frozen_dir,
            plan_path=frozen_dir / "split_plan.json",
        )
    assert (frozen_dir / "train.csv").read_bytes() == before
    assert freeze.verify(frozen_dir) is True


def test_freezing_without_a_plan_raises(raw_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        freeze.freeze(
            raw_dir=raw_dir,
            frozen_dir=tmp_path / "frozen",
            plan_path=tmp_path / "frozen" / "split_plan.json",
        )


def test_freezing_without_raw_data_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        freeze.freeze(
            raw_dir=tmp_path / "empty",
            frozen_dir=tmp_path / "frozen",
            plan_path=tmp_path / "plan.json",
        )


def test_a_plan_that_does_not_cover_the_data_is_refused(raw_dir: Path, tmp_path: Path) -> None:
    target = tmp_path / "frozen"
    plan_path = target / "split_plan.json"
    plan = split.split(raw_dir / "train.csv", plan_path=plan_path)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["train_ids"] = plan.train_ids[:-1]  # drop a row from the plan
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not cover"):
        freeze.freeze(raw_dir=raw_dir, frozen_dir=target, plan_path=plan_path)


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #


def test_is_frozen_is_false_for_an_empty_dir(tmp_path: Path) -> None:
    assert freeze.is_frozen(tmp_path) is False


def test_is_frozen_is_false_for_a_corrupt_manifest(tmp_path: Path) -> None:
    (tmp_path / "MANIFEST.json").write_text("{not json", encoding="utf-8")
    assert freeze.is_frozen(tmp_path) is False


def test_is_frozen_never_raises(tmp_path: Path) -> None:
    (tmp_path / "MANIFEST.json").mkdir()  # a directory where a file belongs
    assert freeze.is_frozen(tmp_path) is False


def test_verify_catches_tampering(writable_frozen_dir: Path) -> None:
    target = writable_frozen_dir / "train.csv"
    target.write_text(
        target.read_text(encoding="utf-8") + "999,1,3,X,male,30,0,0,T,1.0,,S\n",
        encoding="utf-8",
    )
    assert freeze.verify(writable_frozen_dir) is False


def test_verify_catches_a_deleted_file(writable_frozen_dir: Path) -> None:
    (writable_frozen_dir / "holdout.csv").unlink()
    assert freeze.verify(writable_frozen_dir) is False


def test_verify_catches_an_edited_split_plan(writable_frozen_dir: Path) -> None:
    path = writable_frozen_dir / "split_plan.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["seed"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert freeze.verify(writable_frozen_dir) is False


def test_editing_the_manifest_cannot_re_bless_tampered_data(writable_frozen_dir: Path) -> None:
    # Tamper with the data, then rewrite the manifest's per-file hash to match.
    # The fingerprint is derived from those hashes and checked separately, so
    # both have to be forged -- and the honest path is simply shorter.
    target = writable_frozen_dir / "train.csv"
    target.write_text("PassengerId,Survived\n1,0\n", encoding="utf-8")
    manifest_path = writable_frozen_dir / "MANIFEST.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["files"]["train.csv"] = freeze._sha256(target)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    assert freeze.verify(writable_frozen_dir) is False


def test_verify_is_false_for_an_unfrozen_dir(tmp_path: Path) -> None:
    assert freeze.verify(tmp_path) is False


def test_require_frozen_refuses_an_unfrozen_dir(tmp_path: Path) -> None:
    with raises_real_runtime_error():
        freeze.require_frozen(tmp_path)


def test_require_frozen_refuses_a_tampered_dir(writable_frozen_dir: Path) -> None:
    (writable_frozen_dir / "train.csv").write_text("broken\n", encoding="utf-8")
    with raises_real_runtime_error():
        freeze.require_frozen(writable_frozen_dir)


def test_require_frozen_returns_the_manifest_when_sound(frozen_dir: Path) -> None:
    manifest = freeze.require_frozen(frozen_dir)
    assert manifest.dataset_sha256


def test_load_manifest_on_a_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        freeze.load_manifest(tmp_path)


def test_load_manifest_on_an_incomplete_manifest_raises(tmp_path: Path) -> None:
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"dataset_sha256": "x"}), encoding="utf-8")
    with pytest.raises(ValueError):
        freeze.load_manifest(tmp_path)


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def test_main_verify_only_reports_success(frozen_dir: Path, capsys) -> None:
    assert freeze.main(["--verify-only", "--frozen-dir", str(frozen_dir)]) == 0
    assert "frozen dataset OK" in capsys.readouterr().out


def test_main_verify_only_fails_loudly_on_mismatch(writable_frozen_dir: Path, capsys) -> None:
    (writable_frozen_dir / "train.csv").write_text("broken\n", encoding="utf-8")
    code = freeze.main(["--verify-only", "--frozen-dir", str(writable_frozen_dir)])
    assert code == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_main_reports_failure_without_a_traceback(tmp_path: Path, capsys) -> None:
    code = freeze.main(["--frozen-dir", str(tmp_path / "frozen")])
    assert code == 1
    assert "freeze failed" in capsys.readouterr().out
