"""Smoke test: the filesystem contract every other module imports."""

from __future__ import annotations

from pathlib import Path

from titanic_agent import paths


def test_root_points_at_the_repo() -> None:
    assert (paths.ROOT / "pyproject.toml").is_file()
    assert (paths.ROOT / "CLAUDE.md").is_file()


def test_unredirected_paths_are_anchored_under_root() -> None:
    # Only the paths the test guard leaves alone. Anything in
    # paths.WRITABLE_ARTIFACTS is deliberately pointed at tmp_path for the
    # duration of a test, so asserting it sits under ROOT would be asserting
    # that the isolation is broken.
    for path in (paths.RAW_DIR, paths.FROZEN_DIR, paths.DATA_DIR):
        assert path.is_relative_to(paths.ROOT)


def test_writable_paths_are_declared_under_root() -> None:
    # Checked against the declarations, since the live values are redirected.
    source = Path(paths.__file__).read_text(encoding="utf-8")
    for attribute in paths.WRITABLE_ARTIFACTS:
        declaration = next(line for line in source.splitlines() if line.startswith(f"{attribute}:"))
        assert "ROOT /" in declaration or "_DIR /" in declaration, declaration


def test_frozen_artifacts_live_in_the_frozen_dir() -> None:
    for path in (
        paths.FROZEN_TRAIN,
        paths.FROZEN_HOLDOUT,
        paths.FROZEN_TEST,
        paths.MANIFEST,
        paths.SPLIT_PLAN,
    ):
        assert path.parent == paths.FROZEN_DIR


def test_seed_is_a_fixed_int() -> None:
    assert isinstance(paths.SEED, int)
