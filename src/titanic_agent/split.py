"""Step 3 — ``split``: decide, once, which rows are holdout rows.

This step only *proposes* the partition; :mod:`titanic_agent.freeze` is what
seals it. Keeping them separate means a bad split can be redrawn cheaply right
up until the moment it is frozen — and never afterwards.

Inputs
    ``data/raw/train.csv``.
Outputs
    ``data/frozen/split_plan.json`` — seed, fractions, stratification column,
    and the explicit ``PassengerId`` lists for train and holdout.
Done criteria
    The plan exists, the two id lists are disjoint, their union is exactly the
    raw training ids, and the target proportions match within ``tolerance``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import paths


@dataclass(frozen=True)
class SplitPlan:
    """A reproducible train/holdout partition, before freezing."""

    seed: int = paths.SEED
    holdout_frac: float = 0.2
    stratify_on: str = "Survived"
    id_column: str = "PassengerId"
    train_ids: list[int] = field(default_factory=list)
    holdout_ids: list[int] = field(default_factory=list)
    target_balance: dict[str, float] = field(default_factory=dict)
    plan_path: Path | None = None


def split(
    raw_train: Path = paths.RAW_TRAIN,
    *,
    holdout_frac: float = 0.2,
    stratify_on: str = "Survived",
    id_column: str = "PassengerId",
    seed: int = paths.SEED,
    plan_path: Path = paths.SPLIT_PLAN,
    force: bool = False,
) -> SplitPlan:
    """Draw a stratified train/holdout partition and persist it as JSON.

    Args:
        raw_train: The pulled training CSV.
        holdout_frac: Share of rows held back for the final honest estimate.
        stratify_on: Column to stratify on, keeping class balance in both parts.
        id_column: Stable row identifier written into the plan.
        seed: Seed for ``sklearn.model_selection.train_test_split``.
        plan_path: Where to write the plan.
        force: Overwrite an existing plan. Refused once the data is frozen —
            see :func:`titanic_agent.freeze.is_frozen`.

    Returns:
        The persisted :class:`SplitPlan`.

    Raises:
        RuntimeError: The dataset is already frozen, so the split is immutable.
    """
    # TODO(split): implement.
    #   1. Refuse outright if freeze.is_frozen() -- the holdout is sealed.
    #   2. Load raw_train; stratified train_test_split on `stratify_on`.
    #   3. Write plan_path as JSON: seed, fracs, id lists, target balance.
    #   4. Call validate() before returning.
    raise NotImplementedError("split() is not implemented yet")


def validate(plan: SplitPlan, *, tolerance: float = 0.02) -> bool:
    """Check a plan for disjointness, completeness and balance.

    Args:
        plan: The plan to check.
        tolerance: Allowed absolute difference in target rate between the
            train and holdout parts.
    """
    # TODO(split): implement the three checks named in the module docstring.
    raise NotImplementedError("validate() is not implemented yet")


def load_plan(plan_path: Path = paths.SPLIT_PLAN) -> SplitPlan:
    """Read a previously written split plan back into a :class:`SplitPlan`."""
    # TODO(split): implement JSON -> SplitPlan.
    raise NotImplementedError("load_plan() is not implemented yet")


def main(argv: list[str] | None = None) -> int:
    """``make split`` entry point."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=paths.SEED)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    plan = split(holdout_frac=args.holdout_frac, seed=args.seed, force=args.force)
    print(
        f"split: {len(plan.train_ids)} train / {len(plan.holdout_ids)} holdout -> {plan.plan_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
