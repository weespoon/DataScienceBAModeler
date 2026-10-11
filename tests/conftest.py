"""Shared fixtures for the smoke tests.

The suite never touches the real ``data/`` or ``experiments/`` directories: every
fixture writes into pytest's ``tmp_path``. That is a hard rule here — a test run
must not be able to disturb a frozen dataset or the experiment ledger.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

import pytest

from titanic_agent import paths

#: A tiny, hand-written stand-in for the Kaggle training file. Twelve rows is
#: enough to exercise shapes, dtypes, missingness and a two-class target.
TRAIN_ROWS: list[dict[str, object]] = [
    {
        "PassengerId": 1,
        "Survived": 0,
        "Pclass": 3,
        "Name": "Braund, Mr. Owen",
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Ticket": "A/5 21171",
        "Fare": 7.25,
        "Cabin": "",
        "Embarked": "S",
    },
    {
        "PassengerId": 2,
        "Survived": 1,
        "Pclass": 1,
        "Name": "Cumings, Mrs. John",
        "Sex": "female",
        "Age": 38,
        "SibSp": 1,
        "Parch": 0,
        "Ticket": "PC 17599",
        "Fare": 71.28,
        "Cabin": "C85",
        "Embarked": "C",
    },
    {
        "PassengerId": 3,
        "Survived": 1,
        "Pclass": 3,
        "Name": "Heikkinen, Miss. Laina",
        "Sex": "female",
        "Age": 26,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "STON/O2",
        "Fare": 7.92,
        "Cabin": "",
        "Embarked": "S",
    },
    {
        "PassengerId": 4,
        "Survived": 1,
        "Pclass": 1,
        "Name": "Futrelle, Mrs. Jacques",
        "Sex": "female",
        "Age": 35,
        "SibSp": 1,
        "Parch": 0,
        "Ticket": "113803",
        "Fare": 53.1,
        "Cabin": "C123",
        "Embarked": "S",
    },
    {
        "PassengerId": 5,
        "Survived": 0,
        "Pclass": 3,
        "Name": "Allen, Mr. William",
        "Sex": "male",
        "Age": 35,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "373450",
        "Fare": 8.05,
        "Cabin": "",
        "Embarked": "S",
    },
    {
        "PassengerId": 6,
        "Survived": 0,
        "Pclass": 3,
        "Name": "Moran, Mr. James",
        "Sex": "male",
        "Age": "",
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "330877",
        "Fare": 8.46,
        "Cabin": "",
        "Embarked": "Q",
    },
    {
        "PassengerId": 7,
        "Survived": 0,
        "Pclass": 1,
        "Name": "McCarthy, Mr. Timothy",
        "Sex": "male",
        "Age": 54,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "17463",
        "Fare": 51.86,
        "Cabin": "E46",
        "Embarked": "S",
    },
    {
        "PassengerId": 8,
        "Survived": 0,
        "Pclass": 3,
        "Name": "Palsson, Master. Gosta",
        "Sex": "male",
        "Age": 2,
        "SibSp": 3,
        "Parch": 1,
        "Ticket": "349909",
        "Fare": 21.08,
        "Cabin": "",
        "Embarked": "S",
    },
    {
        "PassengerId": 9,
        "Survived": 1,
        "Pclass": 3,
        "Name": "Johnson, Mrs. Oscar",
        "Sex": "female",
        "Age": 27,
        "SibSp": 0,
        "Parch": 2,
        "Ticket": "347742",
        "Fare": 11.13,
        "Cabin": "",
        "Embarked": "S",
    },
    {
        "PassengerId": 10,
        "Survived": 1,
        "Pclass": 2,
        "Name": "Nasser, Mrs. Nicholas",
        "Sex": "female",
        "Age": 14,
        "SibSp": 1,
        "Parch": 0,
        "Ticket": "237736",
        "Fare": 30.07,
        "Cabin": "",
        "Embarked": "C",
    },
    {
        "PassengerId": 11,
        "Survived": 1,
        "Pclass": 3,
        "Name": "Sandstrom, Miss. Marguerite",
        "Sex": "female",
        "Age": 4,
        "SibSp": 1,
        "Parch": 1,
        "Ticket": "PP 9549",
        "Fare": 16.7,
        "Cabin": "G6",
        "Embarked": "S",
    },
    {
        "PassengerId": 12,
        "Survived": 1,
        "Pclass": 1,
        "Name": "Bonnell, Miss. Elizabeth",
        "Sex": "female",
        "Age": 58,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "113783",
        "Fare": 26.55,
        "Cabin": "C103",
        "Embarked": "S",
    },
]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """A ``data/raw``-shaped directory holding a miniature train.csv/test.csv."""
    directory = tmp_path / "raw"
    _write_csv(directory / "train.csv", TRAIN_ROWS)
    test_rows = [{k: v for k, v in row.items() if k != "Survived"} for row in TRAIN_ROWS[:4]]
    for offset, row in enumerate(test_rows):
        row["PassengerId"] = 892 + offset
    _write_csv(directory / "test.csv", test_rows)
    return directory


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    """Path to a throwaway experiment ledger (not yet created)."""
    return tmp_path / "experiments" / "results.csv"


@pytest.fixture
def frozen_dir(raw_dir: Path, tmp_path: Path) -> Path:
    """A sealed dataset in tmp_path, built from the miniature fixture.

    Every test that touches steps 4-7 goes through this. The repo's own
    ``data/frozen`` is never read or written by a test run.
    """
    from titanic_agent import freeze, split

    target = tmp_path / "frozen"
    split.split(raw_dir / "train.csv", plan_path=target / "split_plan.json")
    freeze.freeze(raw_dir=raw_dir, frozen_dir=target, plan_path=target / "split_plan.json")
    return target


@pytest.fixture
def writable_frozen_dir(raw_dir: Path, tmp_path: Path) -> Path:
    """Same, but mode 0644 so a test can tamper with the files."""
    from titanic_agent import freeze, split

    target = tmp_path / "frozen"
    split.split(raw_dir / "train.csv", plan_path=target / "split_plan.json")
    freeze.freeze(
        raw_dir=raw_dir,
        frozen_dir=target,
        plan_path=target / "split_plan.json",
        read_only=False,
    )
    return target


@pytest.fixture
def journal_path(tmp_path: Path) -> Path:
    """Path to a throwaway run journal (not yet created)."""
    return tmp_path / "experiments" / "run_log.jsonl"


@pytest.fixture(autouse=True)
def _never_touch_real_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect every append-only artifact path into tmp_path, for every test.

    Autouse and unconditional, because the failure it prevents is silent. Tests
    that pass ``log_path=`` explicitly were safe; ``run_log.record()`` inside
    ``tune`` and ``ensemble`` takes no such argument from its callers and fell
    through to the real ``experiments/run_log.jsonl``. The suite wrote 167
    fixture-sized entries into the production journal before ``make journal``
    showed cv=0.4500 on a 12-row dataset and gave it away.

    Belt and braces: a test should not be able to reach these paths even by
    accident, and relying on every call site remembering to pass an override is
    exactly the kind of discipline that fails quietly.

    It failed a second time, too. This guard originally named three paths by
    hand; ``RUNS_DIR`` arrived later and was not added, so the suite created 53
    stray run directories before ``make runs`` showed them. The list now lives
    in :data:`titanic_agent.paths.WRITABLE_ARTIFACTS` and is iterated, so a new
    artifact is covered by declaring it rather than by remembering this file.
    """
    sandbox = tmp_path / "artifacts"
    for attribute in paths.WRITABLE_ARTIFACTS:
        real = getattr(paths, attribute)
        monkeypatch.setattr(paths, attribute, sandbox / real.name)


@pytest.fixture(scope="session", autouse=True)
def _repo_artifacts_unchanged() -> Iterator[None]:
    """Fail the session if any test wrote into the repo's artifact trees.

    The per-path guard above redirects everything named in
    ``paths.WRITABLE_ARTIFACTS``. It cannot catch a path nobody remembered to
    add — and that has now happened twice: ``RUNS_DIR`` (53 stray run
    directories) and ``EXPERIMENTS_DIR``. A list you must remember to update is
    not a guard, it is a habit.

    This one needs no list. It snapshots the real trees before the session and
    compares afterwards, so a new artifact leaks loudly the first time rather
    than quietly for an hour.
    """
    from titanic_agent import paths as real_paths

    watched = (
        real_paths.ROOT / "experiments",
        real_paths.ROOT / "runs",
        real_paths.ROOT / "reports",
    )

    # Editor swap and backup files. A human reading an experiment in vim while
    # the suite runs creates one, and no test ever writes one -- flagging it
    # failed a green suite during exp-007's follow-ups.
    editor_droppings = (".swp", ".swo", "~")

    def snapshot() -> dict[str, float]:
        seen: dict[str, float] = {}
        for root in watched:
            if not root.exists():
                continue
            for item in root.rglob("*"):
                if item.is_file() and not item.name.endswith(editor_droppings):
                    seen[str(item)] = item.stat().st_mtime
        return seen

    before = snapshot()
    yield
    after = snapshot()

    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    assert not (added or removed or changed), (
        "the test suite wrote into the real repo:\n"
        f"  added:   {added[:5]}\n  removed: {removed[:5]}\n  changed: {changed[:5]}\n"
        "Add the path to paths.WRITABLE_ARTIFACTS so the autouse guard redirects it."
    )
