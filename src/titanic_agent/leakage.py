"""Structural guards against the one way this project can cheat itself.

Titanic has a specific trap that most tutorial datasets do not. The real
passenger manifest, with who actually survived, is public — Encyclopedia
Titanica and several mirrors carry all 1309 passengers. A model that joins
against it does not predict anything; it looks the answer up.

The reason this matters here is not Kaggle etiquette. **Our own final test is
drawn from those same passengers.** A lookup would score near-perfectly on the
locked 179 rows too, so the number the whole programme is built around would be
inflated and nothing else in the repo would notice. ``make audit`` would stay
green. The ledger would chain cleanly. Every rule would hold, and the result
would be worthless.

Written rules do not survive an autonomous programme that reads public notebooks
looking for techniques that scored better — because the techniques that score
better are, disproportionately, that one. So this module checks structurally:

``feature_provenance``
    Every feature in every ledger row must be derivable from the frozen data.
    A column from somewhere else is the signature of a join.
``external_reads``
    No pipeline module may read data from outside the sanctioned directories.
    This is what catches a lookup hidden inside feature engineering, where the
    resulting column name would look perfectly innocent.
``plausibility``
    Titanic's realistic ceiling on 891 labelled rows is roughly 0.80-0.84. A
    cross-validated score far above that is not a breakthrough. It is the single
    most reliable signal that something is being looked up rather than learned.
``stray_data``
    An unexpected file under ``data/`` is how an external manifest arrives.

None of these prove honesty. A determined agent could defeat each one. They
raise the cost of cheating from "copy a highly-rated notebook" to "deliberately
evade four named checks", and they make the attempt visible afterwards — which
is the most any structural guard offers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import paths

#: Cross-validated accuracy above which a Titanic result stops being plausible.
#: Published work on the 891 labelled rows clusters around 0.80-0.84; the public
#: leaderboard's perfect scores come from the manifest lookup, not from models.
#: Set well above the honest ceiling so a genuinely good model is never flagged.
IMPLAUSIBLE_ACCURACY: float = 0.90

#: Directories a pipeline module may legitimately read data from.
SANCTIONED_READ_ROOTS: tuple[str, ...] = (
    "data/frozen",
    "data/raw",
    "experiments",
    "reports",
    "runs",
)

#: Files expected to exist under ``data/``. Anything else is worth explaining.
EXPECTED_DATA_FILES: frozenset[str] = frozenset(
    {
        "train.csv",
        "test.csv",
        "holdout.csv",
        "gender_submission.csv",
        "split_plan.json",
        "MANIFEST.json",
        ".gitkeep",
    }
)

#: A read call in source, and the literal path it targets if there is one.
READ_CALL = re.compile(r"(read_csv|read_json|read_parquet|read_excel|read_html)\s*\(([^)]*)")


@dataclass(frozen=True)
class LeakageCheck:
    """One structural finding."""

    name: str
    passed: bool
    detail: str = ""
    offenders: list[str] = field(default_factory=list)


def derivable_columns(frozen_dir: Path = paths.FROZEN_DIR) -> set[str]:
    """Every column obtainable from the frozen data alone.

    The raw columns plus whatever :func:`titanic_agent.tune.add_derived` builds
    from them. Self-maintaining on purpose: a new engineered feature is allowed
    the moment it is declared in code, because it is derived from data we have
    already sealed. What this cannot see is a join *inside* ``add_derived`` —
    that is what :func:`check_external_reads` is for.
    """
    import pandas as pd

    from . import tune

    working = pd.read_csv(Path(frozen_dir) / "train.csv", nrows=5)
    return set(tune.add_derived(working).columns)


def check_feature_provenance(
    *, log_path: Path | None = None, frozen_dir: Path = paths.FROZEN_DIR
) -> LeakageCheck:
    """Every feature any run used must be derivable from the frozen data."""
    from . import results_log

    frame = results_log.read_log(log_path)
    if frame.empty:
        return LeakageCheck("feature provenance", True, "no runs logged yet")

    allowed = derivable_columns(frozen_dir)
    offenders: set[str] = set()
    for params in frame["params"]:
        if isinstance(params, dict):
            for feature in params.get("features") or []:
                if feature not in allowed:
                    offenders.add(str(feature))

    return LeakageCheck(
        "feature provenance",
        not offenders,
        f"{len(allowed)} derivable columns; every logged feature is one of them"
        if not offenders
        else f"features not derivable from the frozen data: {sorted(offenders)}",
        sorted(offenders),
    )


def check_external_reads(package_dir: Path | None = None) -> LeakageCheck:
    """No pipeline module may read data from outside the sanctioned directories.

    This is the check that catches a lookup disguised as feature engineering: the
    column it produces would look ordinary, but the read that built it cannot.
    """
    root = Path(package_dir) if package_dir is not None else Path(__file__).parent
    offenders: list[str] = []

    for module in sorted(root.glob("*.py")):
        if module.name == "leakage.py":  # this file names the patterns it hunts
            continue
        for number, line in enumerate(module.read_text(encoding="utf-8").splitlines(), 1):
            match = READ_CALL.search(line)
            if not match:
                continue
            argument = match.group(2)
            quoted = re.findall(r"""['"]([^'"]+)['"]""", argument)
            for literal in quoted:
                if literal.startswith(("http://", "https://", "s3://", "gs://", "ftp://")):
                    offenders.append(f"{module.name}:{number} reads a URL: {literal}")
                elif "/" in literal and not any(
                    literal.startswith(root_dir) for root_dir in SANCTIONED_READ_ROOTS
                ):
                    offenders.append(f"{module.name}:{number} reads {literal}")

    return LeakageCheck(
        "external reads",
        not offenders,
        "no module reads data from outside the sanctioned directories"
        if not offenders
        else "a module reads data from outside the pipeline",
        offenders,
    )


#: Which kind of number a ledger row's ``cv_score`` column actually holds. The
#: column is shared, so the *step* is what distinguishes them — and conflating
#: the two is how this check spent a while reporting a 143-row validation score
#: as "best cv", which the cold-start session caught.
METRIC_BY_STEP: dict[str, str] = {
    "tune": "cv",
    "features": "cv",
    "ensemble": "cv",
    "evaluate": "val",
}


def metric_for(step: str) -> str:
    """What kind of score a row of this step carries."""
    return METRIC_BY_STEP.get(str(step), "score")


def check_plausibility(*, log_path: Path | None = None) -> LeakageCheck:
    """Flag scores too high to be honest on this dataset.

    Not a proof of cheating, and deliberately set far above the realistic
    ceiling so a genuinely strong model never trips it. But on 891 labelled
    rows, a 0.95 is not a modelling achievement — it means the answers got in
    somehow, and that deserves a human rather than a shrug.

    Cross-validated and validation scores are reported **separately**. They
    share the ``cv_score`` column but are not the same measurement: a CV score
    averages folds over hundreds of rows, a validation score is one draw over
    143. Calling the second one "cv" is a small lie in an honesty check, which
    is the worst place to keep one.
    """
    from . import results_log

    frame = results_log.read_log(log_path)
    if frame.empty:
        return LeakageCheck("score plausibility", True, "no runs logged yet")

    scored = frame.dropna(subset=["cv_score"])
    if scored.empty:
        return LeakageCheck("score plausibility", True, "no scored runs yet")

    suspicious = scored[scored["cv_score"].astype(float) > IMPLAUSIBLE_ACCURACY]

    bests: list[str] = []
    for metric in ("cv", "val"):
        rows = scored[scored["step"].map(metric_for) == metric]
        if len(rows):
            bests.append(f"best {metric} {float(rows['cv_score'].astype(float).max()):.4f}")

    return LeakageCheck(
        "score plausibility",
        suspicious.empty,
        f"{', '.join(bests)} — below the {IMPLAUSIBLE_ACCURACY:.2f} implausibility line"
        if suspicious.empty
        else (
            f"{len(suspicious)} run(s) scored above {IMPLAUSIBLE_ACCURACY:.2f} on a dataset "
            f"whose honest ceiling is ~0.84. Stop and explain before trusting anything here"
        ),
        sorted(str(r) for r in suspicious["run_id"]),
    )


def check_stray_data(data_dir: Path | None = None) -> LeakageCheck:
    """An unexpected file under ``data/`` is how an external manifest arrives."""
    root = Path(data_dir) if data_dir is not None else paths.DATA_DIR
    if not root.is_dir():
        return LeakageCheck("stray data files", True, "no data directory")

    strays = [
        str(item.relative_to(root))
        for item in sorted(root.rglob("*"))
        if item.is_file() and item.name not in EXPECTED_DATA_FILES and item.suffix != ".lock"
    ]
    return LeakageCheck(
        "stray data files",
        not strays,
        "only the expected files are present" if not strays else f"unexpected: {strays}",
        strays,
    )


def check_all(
    *, log_path: Path | None = None, frozen_dir: Path = paths.FROZEN_DIR
) -> list[LeakageCheck]:
    """Run every structural guard."""
    return [
        check_feature_provenance(log_path=log_path, frozen_dir=frozen_dir),
        check_external_reads(),
        check_plausibility(log_path=log_path),
        check_stray_data(),
    ]


def main(argv: list[str] | None = None) -> int:
    """``make leakage`` — exits non-zero if any structural guard trips."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)

    checks = check_all()
    for check in checks:
        print(f"  [{'pass' if check.passed else 'FAIL'}] {check.name:<22} {check.detail}")
        for offender in check.offenders[:5]:
            print(f"         {offender}")

    print()
    failures = [c for c in checks if not c.passed]
    if failures:
        print(f"{len(failures)} structural guard(s) tripped. Stop and explain before continuing.")
        print("The real Titanic manifest is public; a lookup would inflate our own")
        print("locked final test, not just a leaderboard, and nothing else here would notice.")
        return 1
    print("No structural sign of leakage. That is not proof of honesty —")
    print("these raise the cost of cheating and make an attempt visible, nothing more.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
