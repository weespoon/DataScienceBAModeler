"""Smoke test: step 3, ``split``.

Every test writes its plan into ``tmp_path``, never ``data/frozen`` — the real
frozen dataset must be unreachable from a test run.

The load-bearing test is :func:`test_split_refuses_once_the_data_is_frozen`.
Everything else here checks that a drawn split is sound; that one checks the
rule that makes the holdout worth anything.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from titanic_agent import paths, split


@pytest.fixture
def plan_path(tmp_path: Path) -> Path:
    return tmp_path / "frozen" / "split_plan.json"


@pytest.fixture
def drawn(raw_dir: Path, plan_path: Path) -> split.SplitPlan:
    return split.split(raw_dir / "train.csv", plan_path=plan_path)


# --------------------------------------------------------------------------- #
# contract
# --------------------------------------------------------------------------- #


def test_public_api_exists() -> None:
    for name in ("split", "validate", "load_plan", "main"):
        assert callable(getattr(split, name)), name


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


# --------------------------------------------------------------------------- #
# soundness of a drawn split
# --------------------------------------------------------------------------- #


def test_id_lists_are_disjoint_and_complete(drawn: split.SplitPlan, raw_dir: Path) -> None:
    import pandas as pd

    train, holdout = set(drawn.train_ids), set(drawn.holdout_ids)
    assert not train & holdout
    expected = {int(i) for i in pd.read_csv(raw_dir / "train.csv")["PassengerId"]}
    assert train | holdout == expected


def test_no_row_appears_twice_within_a_part(drawn: split.SplitPlan) -> None:
    assert len(drawn.train_ids) == len(set(drawn.train_ids))
    assert len(drawn.holdout_ids) == len(set(drawn.holdout_ids))


def test_holdout_size_matches_the_requested_fraction(raw_dir: Path, plan_path: Path) -> None:
    plan = split.split(raw_dir / "train.csv", holdout_frac=0.25, plan_path=plan_path)
    total = len(plan.train_ids) + len(plan.holdout_ids)
    assert abs(len(plan.holdout_ids) - round(total * 0.25)) <= 1


def test_target_balance_is_recorded_and_close(drawn: split.SplitPlan) -> None:
    assert set(drawn.target_balance) == {"train", "holdout", "gap"}
    assert drawn.target_balance["gap"] <= 0.5  # tiny fixture; real data is ~0.002


def test_the_same_seed_reproduces_the_same_draw(raw_dir: Path, tmp_path: Path) -> None:
    first = split.split(raw_dir / "train.csv", seed=7, plan_path=tmp_path / "a.json")
    second = split.split(raw_dir / "train.csv", seed=7, plan_path=tmp_path / "b.json")
    assert first.holdout_ids == second.holdout_ids


def test_a_different_seed_draws_differently(raw_dir: Path, tmp_path: Path) -> None:
    first = split.split(raw_dir / "train.csv", seed=7, plan_path=tmp_path / "a.json")
    second = split.split(raw_dir / "train.csv", seed=99, plan_path=tmp_path / "b.json")
    assert first.holdout_ids != second.holdout_ids


def test_validate_accepts_a_drawn_plan(drawn: split.SplitPlan, raw_dir: Path) -> None:
    assert split.validate(drawn, raw_train=raw_dir / "train.csv") is True


# --------------------------------------------------------------------------- #
# validate rejects unsound plans
# --------------------------------------------------------------------------- #


def test_validate_rejects_overlapping_parts() -> None:
    plan = split.SplitPlan(train_ids=[1, 2, 3, 4], holdout_ids=[4], holdout_frac=0.2)
    assert split.validate(plan) is False


def test_validate_rejects_an_empty_part() -> None:
    assert split.validate(split.SplitPlan(train_ids=[1, 2], holdout_ids=[])) is False


def test_validate_rejects_duplicates_within_a_part() -> None:
    plan = split.SplitPlan(train_ids=[1, 1, 2, 3], holdout_ids=[4], holdout_frac=0.2)
    assert split.validate(plan) is False


def test_validate_rejects_a_wrong_sized_holdout() -> None:
    plan = split.SplitPlan(train_ids=[1, 2], holdout_ids=[3, 4, 5, 6, 7, 8], holdout_frac=0.2)
    assert split.validate(plan) is False


def test_validate_rejects_an_imbalanced_draw() -> None:
    # 20 holdout rows, so one row is worth 0.05 and a 0.4 gap is far beyond
    # anything the quantisation can excuse.
    plan = split.SplitPlan(
        train_ids=list(range(1, 81)),
        holdout_ids=list(range(81, 101)),
        holdout_frac=0.2,
        target_balance={"train": 0.5, "holdout": 0.9, "gap": 0.4},
    )
    assert split.validate(plan) is False


def test_the_balance_tolerance_respects_holdout_granularity() -> None:
    # With 3 holdout rows the only achievable target rates are 0, 1/3, 2/3 and
    # 1, so demanding 0.02 would be arithmetically impossible however good the
    # stratification is. The check allows one row's worth instead.
    tiny = split.SplitPlan(
        train_ids=list(range(1, 13)),
        holdout_ids=[13, 14, 15],
        holdout_frac=0.2,
        target_balance={"train": 0.5, "holdout": 0.667, "gap": 0.167},
    )
    assert split.validate(tiny) is True

    # The same gap on a large holdout is not excusable.
    large = split.SplitPlan(
        train_ids=list(range(1, 801)),
        holdout_ids=list(range(801, 1001)),
        holdout_frac=0.2,
        target_balance={"train": 0.5, "holdout": 0.667, "gap": 0.167},
    )
    assert split.validate(large) is False


def test_validate_rejects_a_plan_that_misses_rows(drawn: split.SplitPlan, raw_dir: Path) -> None:
    truncated = split.SplitPlan(
        holdout_frac=drawn.holdout_frac,
        train_ids=drawn.train_ids[:-1],
        holdout_ids=drawn.holdout_ids,
        target_balance=drawn.target_balance,
    )
    assert split.validate(truncated, raw_train=raw_dir / "train.csv") is False


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #


def test_the_plan_round_trips_through_json(drawn: split.SplitPlan, plan_path: Path) -> None:
    reloaded = split.load_plan(plan_path)
    assert reloaded.train_ids == drawn.train_ids
    assert reloaded.holdout_ids == drawn.holdout_ids
    assert reloaded.seed == drawn.seed
    assert reloaded.stratify_on == drawn.stratify_on


def test_the_json_records_everything_needed_to_rebuild_it(plan_path: Path, drawn) -> None:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    for key in ("seed", "holdout_frac", "stratify_on", "id_column", "train_ids", "holdout_ids"):
        assert key in payload, key
    assert payload["n_train"] == len(payload["train_ids"])


def test_load_plan_on_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        split.load_plan(tmp_path / "nope.json")


def test_load_plan_on_malformed_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        split.load_plan(path)


def test_load_plan_on_an_incomplete_plan_raises(tmp_path: Path) -> None:
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"seed": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        split.load_plan(path)


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #


def test_split_refuses_once_the_data_is_frozen(raw_dir: Path, tmp_path: Path) -> None:
    # The rule that makes the holdout worth anything: a sealed split is never
    # redrawn. A manifest in the target directory is the signal, which is the
    # same one freeze.is_frozen() and the PreToolUse hook read.
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "MANIFEST.json").write_text(
        json.dumps({"dataset_sha256": "deadbeef"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="already frozen"):
        split.split(raw_dir / "train.csv", plan_path=frozen / "split_plan.json")


def test_split_refuses_to_overwrite_a_plan_without_force(raw_dir: Path, plan_path: Path) -> None:
    split.split(raw_dir / "train.csv", plan_path=plan_path)
    with pytest.raises(FileExistsError):
        split.split(raw_dir / "train.csv", plan_path=plan_path)


def test_force_redraws_while_unfrozen(raw_dir: Path, plan_path: Path) -> None:
    first = split.split(raw_dir / "train.csv", seed=1, plan_path=plan_path)
    second = split.split(raw_dir / "train.csv", seed=2, plan_path=plan_path, force=True)
    assert first.holdout_ids != second.holdout_ids


def test_missing_raw_data_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        split.split(tmp_path / "nope.csv", plan_path=tmp_path / "plan.json")


def test_a_non_unique_id_column_is_refused(raw_dir: Path, tmp_path: Path) -> None:
    import pandas as pd

    frame = pd.read_csv(raw_dir / "train.csv")
    frame["PassengerId"] = 1
    frame.to_csv(raw_dir / "train.csv", index=False)
    with pytest.raises(ValueError, match="not unique"):
        split.split(raw_dir / "train.csv", plan_path=tmp_path / "plan.json")


def test_nothing_is_written_when_the_draw_is_refused(raw_dir: Path, tmp_path: Path) -> None:
    import pandas as pd

    frame = pd.read_csv(raw_dir / "train.csv")
    frame["PassengerId"] = 1
    frame.to_csv(raw_dir / "train.csv", index=False)
    plan_path = tmp_path / "plan.json"
    with pytest.raises(ValueError):
        split.split(raw_dir / "train.csv", plan_path=plan_path)
    assert not plan_path.exists()


def test_main_returns_zero_and_reports(raw_dir: Path, tmp_path: Path, capsys) -> None:
    code = split.main(
        ["--raw-train", str(raw_dir / "train.csv"), "--plan", str(tmp_path / "plan.json")]
    )
    assert code == 0
    assert "train /" in capsys.readouterr().out


def test_main_reports_failure_without_a_traceback(tmp_path: Path, capsys) -> None:
    code = split.main(
        ["--raw-train", str(tmp_path / "nope.csv"), "--plan", str(tmp_path / "plan.json")]
    )
    assert code == 1
    assert "split failed" in capsys.readouterr().out
