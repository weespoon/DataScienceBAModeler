"""Smoke test: the filesystem contract every other module imports."""

from __future__ import annotations

from titanic_agent import paths


def test_root_points_at_the_repo() -> None:
    assert (paths.ROOT / "pyproject.toml").is_file()
    assert (paths.ROOT / "CLAUDE.md").is_file()


def test_paths_are_anchored_under_root() -> None:
    for path in (paths.RAW_DIR, paths.FROZEN_DIR, paths.REPORTS_DIR, paths.EXPERIMENTS_DIR):
        assert path.is_relative_to(paths.ROOT)


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
