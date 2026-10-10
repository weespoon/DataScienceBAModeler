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

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import paths

#: Allowed absolute difference in target rate between the train and holdout
#: parts. 0.02 is loose enough for an 891-row dataset split 80/20 and tight
#: enough that a genuinely skewed draw fails.
DEFAULT_TOLERANCE: float = 0.02


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


def _target_rates(
    frame,
    *,
    stratify_on: str,
    id_column: str,
    train_ids: list[int],
    holdout_ids: list[int],
) -> dict[str, float]:
    """Mean target in each part, plus the gap between them."""
    indexed = frame.set_index(id_column)[stratify_on]
    train_rate = float(indexed.loc[train_ids].mean())
    holdout_rate = float(indexed.loc[holdout_ids].mean())
    return {
        "train": round(train_rate, 6),
        "holdout": round(holdout_rate, 6),
        "gap": round(abs(train_rate - holdout_rate), 6),
    }


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
        FileExistsError: A plan exists and ``force`` was not given.
        ValueError: The drawn split failed :func:`validate`.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from . import freeze

    # The holdout is already sealed; redrawing it would silently invalidate
    # every score in the ledger. This is a hard stop, not a warning.
    if freeze.is_frozen(Path(plan_path).parent):
        raise RuntimeError(
            "the dataset is already frozen — redrawing the split would invalidate "
            "every score in experiments/results.csv. Unfreezing is a human action: "
            "stop and ask. See docs/UNFREEZING.md (for a person, not for you)."
        )

    plan_path = Path(plan_path)
    if plan_path.exists() and not force:
        raise FileExistsError(
            f"{plan_path} already exists. Pass force=True (or --force) to redraw it; "
            "that is safe only because the data is not frozen yet."
        )

    raw_train = Path(raw_train)
    if not raw_train.is_file():
        raise FileNotFoundError(f"{raw_train} is missing — run `make pull` first")

    frame = pd.read_csv(raw_train)
    for column in (id_column, stratify_on):
        if column not in frame.columns:
            raise ValueError(f"{raw_train} has no `{column}` column")
    if frame[id_column].duplicated().any():
        raise ValueError(f"`{id_column}` is not unique — it cannot identify rows")

    train_part, holdout_part = train_test_split(
        frame,
        test_size=holdout_frac,
        stratify=frame[stratify_on],
        random_state=seed,
        shuffle=True,
    )

    train_ids = sorted(int(i) for i in train_part[id_column])
    holdout_ids = sorted(int(i) for i in holdout_part[id_column])
    plan = SplitPlan(
        seed=seed,
        holdout_frac=holdout_frac,
        stratify_on=stratify_on,
        id_column=id_column,
        train_ids=train_ids,
        holdout_ids=holdout_ids,
        target_balance=_target_rates(
            frame,
            stratify_on=stratify_on,
            id_column=id_column,
            train_ids=train_ids,
            holdout_ids=holdout_ids,
        ),
        plan_path=plan_path,
    )

    if not validate(plan, raw_train=raw_train):
        raise ValueError("the drawn split failed validation; nothing was written")

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": plan.seed,
        "holdout_frac": plan.holdout_frac,
        "stratify_on": plan.stratify_on,
        "id_column": plan.id_column,
        "n_train": len(plan.train_ids),
        "n_holdout": len(plan.holdout_ids),
        "target_balance": plan.target_balance,
        "source": str(raw_train.relative_to(paths.ROOT))
        if raw_train.is_relative_to(paths.ROOT)
        else str(raw_train),
        "train_ids": plan.train_ids,
        "holdout_ids": plan.holdout_ids,
    }
    plan_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return plan


def validate(
    plan: SplitPlan,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    raw_train: Path | None = None,
) -> bool:
    """Check a plan for disjointness, completeness and balance.

    Args:
        plan: The plan to check.
        tolerance: Allowed absolute difference in target rate between the
            train and holdout parts.
        raw_train: When given, also check that the two id lists together are
            exactly the ids in this file. Without it, completeness cannot be
            checked — the plan alone does not know what it was drawn from.

    Returns:
        True when every check passes.
    """
    train, holdout = set(plan.train_ids), set(plan.holdout_ids)

    if not train or not holdout:
        return False
    # Disjoint, and no duplicates inside either part.
    if train & holdout:
        return False
    if len(train) != len(plan.train_ids) or len(holdout) != len(plan.holdout_ids):
        return False

    total = len(train) + len(holdout)
    expected_holdout = round(total * plan.holdout_frac)
    if abs(len(holdout) - expected_holdout) > 1:
        return False

    # The achievable gap is quantised by the holdout's size: with 3 holdout rows
    # the only possible target rates are 0, 1/3, 2/3 and 1, so demanding 0.02
    # is arithmetically impossible however good the stratification is. Allow
    # whichever is larger — the absolute tolerance, or one row's worth. On the
    # real 179-row holdout one row is 0.0056, so `tolerance` still governs and
    # the check stays as strict as it was meant to be.
    gap = plan.target_balance.get("gap")
    if gap is not None:
        granularity = 1.0 / len(holdout)
        if gap > max(tolerance, granularity):
            return False

    if raw_train is not None:
        import pandas as pd

        ids = pd.read_csv(Path(raw_train))[plan.id_column]
        if train | holdout != {int(i) for i in ids}:
            return False

    return True


def load_plan(plan_path: Path = paths.SPLIT_PLAN) -> SplitPlan:
    """Read a previously written split plan back into a :class:`SplitPlan`.

    Raises:
        FileNotFoundError: No plan at ``plan_path``.
        ValueError: The plan is present but malformed.
    """
    plan_path = Path(plan_path)
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{plan_path} is missing — run `make split` first") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{plan_path} is not valid JSON: {exc}") from None

    try:
        return SplitPlan(
            seed=int(payload["seed"]),
            holdout_frac=float(payload["holdout_frac"]),
            stratify_on=str(payload["stratify_on"]),
            id_column=str(payload["id_column"]),
            train_ids=[int(i) for i in payload["train_ids"]],
            holdout_ids=[int(i) for i in payload["holdout_ids"]],
            target_balance=dict(payload.get("target_balance") or {}),
            plan_path=plan_path,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{plan_path} is missing or has a malformed field: {exc}") from None


def main(argv: list[str] | None = None) -> int:
    """``make split`` entry point."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-train", type=Path, default=paths.RAW_TRAIN)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=paths.SEED)
    parser.add_argument("--plan", type=Path, default=paths.SPLIT_PLAN)
    parser.add_argument("--force", action="store_true", help="redraw an existing plan")
    args = parser.parse_args(argv)

    try:
        plan = split(
            args.raw_train,
            holdout_frac=args.holdout_frac,
            seed=args.seed,
            plan_path=args.plan,
            force=args.force,
        )
    except (RuntimeError, FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"split failed: {exc}")
        return 1

    balance = plan.target_balance
    print(
        f"split: {len(plan.train_ids)} train / {len(plan.holdout_ids)} holdout "
        f"(seed={plan.seed}) -> {plan.plan_path}"
    )
    print(
        f"  {plan.stratify_on} rate: train={balance.get('train', 0):.4f} "
        f"holdout={balance.get('holdout', 0):.4f} gap={balance.get('gap', 0):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
