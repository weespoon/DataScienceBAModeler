"""Smoke test: step 3, ``split``."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from titanic_agent import paths, split


def test_public_api_exists() -> None:
    assert callable(split.split)
    assert callable(split.validate)
    assert callable(split.load_plan)
    assert callable(split.main)


def test_signature_matches_the_documented_contract() -> None:
    params = inspect.signature(split.split).parameters
    assert set(params) == {
        "raw_train",
        "holdout_frac",
        "stratify_on",
        "id_column",
        "seed",
        "plan_path",
        "force",
    }
    assert params["holdout_frac"].default == 0.2
    assert params["seed"].default == paths.SEED
    assert params["plan_path"].default == paths.SPLIT_PLAN


def test_plan_defaults_describe_a_stratified_holdout() -> None:
    plan = split.SplitPlan()
    assert plan.stratify_on == "Survived"
    assert plan.id_column == "PassengerId"
    assert plan.holdout_frac == 0.2
    assert plan.train_ids == [] and plan.holdout_ids == []


def test_the_plan_is_written_inside_the_frozen_dir() -> None:
    # split proposes, freeze seals -- but both artifacts live together.
    assert paths.SPLIT_PLAN.parent == paths.FROZEN_DIR


@pytest.mark.xfail(raises=NotImplementedError, reason="step 3 not implemented yet")
def test_split_produces_disjoint_complete_id_lists(raw_dir: Path, tmp_path: Path) -> None:
    plan = split.split(raw_dir / "train.csv", plan_path=tmp_path / "split_plan.json")
    train, holdout = set(plan.train_ids), set(plan.holdout_ids)
    assert not train & holdout
    assert len(train | holdout) == 12


@pytest.mark.xfail(raises=NotImplementedError, reason="step 3 not implemented yet")
def test_split_is_reproducible_from_the_seed(raw_dir: Path, tmp_path: Path) -> None:
    first = split.split(raw_dir / "train.csv", seed=7, plan_path=tmp_path / "a.json")
    second = split.split(raw_dir / "train.csv", seed=7, plan_path=tmp_path / "b.json")
    assert first.holdout_ids == second.holdout_ids


@pytest.mark.xfail(raises=NotImplementedError, reason="step 3 not implemented yet")
def test_plan_round_trips_through_json(raw_dir: Path, tmp_path: Path) -> None:
    plan_path = tmp_path / "split_plan.json"
    written = split.split(raw_dir / "train.csv", plan_path=plan_path)
    assert split.load_plan(plan_path).holdout_ids == written.holdout_ids
