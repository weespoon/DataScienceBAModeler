"""Feature selection against one fixed model: does each feature earn its place?

Every earlier experiment varied the *model* and held the features fixed at the
twelve step 5 proposed. This holds the model fixed instead — one estimator, one
set of hyperparameters, chosen once — and asks of each feature, one at a time,
whether the model is measurably better with it than without it.

The rule is a single number, declared before any test runs::

    keep a feature  iff  score(with) - score(without) >= delta

It applies to additions and removals alike. A new candidate is tried on top of
the current set and kept only if it clears ``delta``; an existing feature is
tried *removed*, and dropped unless removing it costs at least ``delta``. A
feature that does not help is not neutral — it widens the one-hot matrix and
gives the trees another way to fit noise — so the burden of proof is on it.

Why repeated CV, and why paired
    One 5-fold run on ~570 rows has a fold std near 0.017, an order of magnitude
    larger than most feature effects. Three repeats of five folds, with both arms
    of every comparison scored on the *same* 25 folds and the same estimator
    seed, cancels most of the split noise out of the difference. It does not
    cancel all of it: 40 sequential greedy decisions against a 0.003 bar will
    accept some features on luck. That is what the experiment's single
    validation read, on rows none of this saw, is for.

Every candidate is row-wise
    Each is a pure function of one passenger's own row — fixed bin edges, string
    parsing, arithmetic — so computing it before the CV split leaks nothing.
    Nothing here counts across rows (ticket group sizes, surname frequencies) or
    touches the label (family survival rates); those would need fitting inside
    each fold and are recorded as untried instead.

Rule 4 applies to the list itself: every candidate carries a one-line reason
it should generalise, in :data:`CANDIDATES`. "It scored higher in a notebook"
is not a reason, and nothing here came from one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import paths

#: Minimum CV gain a feature must buy to stay. About a third of one row per
#: 114-row fold; small on purpose, so that it is the noise and not the bar that
#: rejects a weak-but-real feature.
DELTA: float = 0.003

#: Hard cap on feature tests in one selection run. The baseline score and the
#: one-off hyperparameter search do not count; every add or remove test does.
MAX_TESTS: int = 50

#: Repeated stratified K-fold: ``CV_FOLDS`` x ``CV_REPEATS`` paired scores.
CV_FOLDS: int = 5
CV_REPEATS: int = 3

#: The model every feature is tested against: exp-004's best single model.
MODEL_KEY: str = "random_forest"

#: Life-stage boundaries. Round numbers chosen from what "child" and "elderly"
#: meant for lifeboat loading, not fitted to this data.
AGE_EDGES: tuple[float, ...] = (0, 13, 18, 30, 50, np.inf)
AGE_LABELS: tuple[str, ...] = ("child", "teen", "young_adult", "adult", "senior")

#: Fare bands in round pounds. Fixed, not quantiles, so the bins cannot carry
#: any information about which rows are in the training fold.
FARE_EDGES: tuple[float, ...] = (-0.01, 10, 25, 50, 100, np.inf)
FARE_LABELS: tuple[str, ...] = ("lt10", "10-25", "25-50", "50-100", "100+")

#: Titles kept distinct by ``TitleFine``. The coarse ``Title`` lumps Rev (no
#: survivors among clergy) with Countess and Lady (all survived) as "Rare".
TITLE_FINE: dict[str, str] = {
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Dr": "Professional",
    "Rev": "Clergy",
    "Col": "Officer",
    "Major": "Officer",
    "Capt": "Officer",
    "Lady": "Noble",
    "the Countess": "Noble",
    "Countess": "Noble",
    "Sir": "Noble",
    "Don": "Noble",
    "Dona": "Noble",
    "Jonkheer": "Noble",
}

#: Every candidate feature and why it should generalise. Order is test order.
CANDIDATES: dict[str, str] = {
    "AgeBin": "Life-stage buckets; survival is non-linear in age (children first), "
    "and a bucket with an explicit Unknown level does not depend on median imputation.",
    "IsChild": "Age < 16. 'Women and children first' was the loading rule; a hard "
    "threshold says that directly.",
    "WomanOrChild": "Female, or a boy (Master / under 16): the evacuation rule itself "
    "as one binary, rather than something the trees must reassemble.",
    "PclassCat": "Pclass one-hot instead of ordinal. Tests whether 1->2->3 being "
    "evenly spaced is a fair assumption.",
    "SexPclass": "Sex x class interaction. Third-class women survived at roughly half "
    "the rate of first-class women; the effect of sex depends on class.",
    "FamilyBucket": "Alone / small (2-4) / large (5+). Family-size survival rises then "
    "collapses; three buckets capture the shape with few parameters.",
    "FareLog": "log1p(Fare). Fare is heavily right-skewed; the log spreads the dense "
    "low end where most of the class-3 variation lives.",
    "FareBin": "Fare bands in round pounds. A coarse wealth signal less sensitive to "
    "the handful of extreme fares.",
    "FarePerPerson": "Fare / FamilySize. Fares were paid per ticket, often for a whole "
    "family, so the raw fare overstates individual wealth for large groups.",
    "FareZero": "Fare == 0. Free passage (crew-adjacent staff, guarantee group) is a "
    "distinct population, not just a cheap ticket.",
    "TitleFine": "Finer titles: Mlle/Ms->Miss, Mme->Mrs, and Clergy, Officer, Noble "
    "and Professional split out of 'Rare', which mixes 0% and 100% survival groups.",
    "TicketPrefix": "Alphabetic ticket prefix (PC, CA, SOTON...). Encodes the issuing "
    "office / route, which tracks class and embarkation point.",
    "TicketNumeric": "Ticket is purely numeric. A coarse version of the prefix with no "
    "rare levels.",
    "TicketLength": "Digits in the ticket number. Ticket series were issued in ranges "
    "by office, so length is a proxy for where and how the ticket was sold.",
    "HasCabin": "Cabin recorded at all. Cabin records survived mostly for first class "
    "and for survivors' accounts; presence is the signal, the value is sparse.",
    "CabinCount": "Number of cabins on one booking. Several cabins means a wealthy "
    "family travelling together.",
    "CabinSide": "Odd cabin numbers were starboard, even port. The two sides were "
    "loaded under different officers with different rules for men.",
    "DeckGroup": "Decks pooled ABC / DE / FG / Unknown. Individual deck letters are too "
    "sparse to estimate; pooled decks share distance to the boat deck.",
    "NameLength": "Characters in Name. Long names are formal ones: married women listed "
    "under the husband's name plus their own, and titled passengers.",
    "NameHasParens": "Name contains a parenthesised part — almost always a married "
    "woman's own name, i.e. a wife travelling with her husband.",
    "IsMother": "Female, not Miss, travelling with children (Parch > 0). Mothers were "
    "boarded with their children.",
    "AgeClass": "Age x Pclass. Older passengers in third class fared worst; a product "
    "lets a single split separate young first-class from old third-class.",
    "HasSibSp": "Any sibling or spouse aboard. Someone to look for, or to help.",
    "HasParch": "Any parent or child aboard; distinct from siblings/spouse.",
}


def _first_cabin_number(cabin: Any) -> float:
    if not isinstance(cabin, str) or not cabin:
        return np.nan
    for token in cabin.split():
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            return float(digits)
    return np.nan


def _ticket_prefix(ticket: Any) -> str:
    if not isinstance(ticket, str) or " " not in ticket.strip():
        return "None"
    head = ticket.strip().rsplit(" ", 1)[0]
    cleaned = "".join(ch for ch in head.upper() if ch.isalnum())
    return cleaned or "None"


def _ticket_digits(ticket: Any) -> int:
    if not isinstance(ticket, str):
        return 0
    tail = ticket.strip().rsplit(" ", 1)[-1]
    return sum(ch.isdigit() for ch in tail)


def add_candidates(frame):
    """Add every column in :data:`CANDIDATES`. Row-wise only; mutates ``frame``.

    Called from :func:`titanic_agent.tune.add_derived`, after ``FamilySize`` and
    ``Title`` exist, so that the candidates are derivable columns everywhere the
    pipeline looks — including ``make leakage``'s provenance check.
    """
    import pandas as pd

    age, fare = frame["Age"], frame["Fare"]
    female = frame["Sex"] == "female"
    title = frame["Title"] if "Title" in frame.columns else pd.Series("Unknown", frame.index)

    frame["AgeBin"] = (
        pd.cut(age, AGE_EDGES, labels=AGE_LABELS, right=False)
        .astype("string")
        .fillna("Unknown")
        .astype(object)
    )
    frame["IsChild"] = (age < 16).astype(int)
    frame["WomanOrChild"] = (female | (title == "Master") | (age < 16)).astype(int)
    frame["PclassCat"] = "class" + frame["Pclass"].astype(str)
    frame["SexPclass"] = frame["Sex"].astype(str) + "_" + frame["Pclass"].astype(str)
    frame["FamilyBucket"] = np.select(
        [frame["FamilySize"] == 1, frame["FamilySize"] <= 4], ["alone", "small"], "large"
    )
    frame["FareLog"] = np.log1p(fare)
    frame["FareBin"] = (
        pd.cut(fare, FARE_EDGES, labels=FARE_LABELS).astype("string").fillna("Unknown")
    ).astype(object)
    frame["FarePerPerson"] = fare / frame["FamilySize"]
    frame["FareZero"] = (fare == 0).astype(int)

    if "Name" in frame.columns:
        raw_title = frame["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
        frame["TitleFine"] = raw_title.map(TITLE_FINE).fillna("Other")
        frame["NameLength"] = frame["Name"].str.len()
        frame["NameHasParens"] = frame["Name"].str.contains("(", regex=False).astype(int)
    else:
        frame["TitleFine"] = "Unknown"
        frame["NameLength"] = np.nan
        frame["NameHasParens"] = 0

    if "Ticket" in frame.columns:
        frame["TicketPrefix"] = frame["Ticket"].map(_ticket_prefix)
        frame["TicketNumeric"] = frame["Ticket"].astype(str).str.isdigit().astype(int)
        frame["TicketLength"] = frame["Ticket"].map(_ticket_digits)
    else:
        frame["TicketPrefix"] = "None"
        frame["TicketNumeric"] = 0
        frame["TicketLength"] = 0

    cabin = frame["Cabin"] if "Cabin" in frame.columns else pd.Series(np.nan, frame.index)
    # map() rather than .str: a slice whose cabins are all missing is float, and
    # the .str accessor refuses it — which a single-row frame hits immediately.
    cabin = cabin.map(lambda value: value.strip() if isinstance(value, str) else "")
    has_cabin = cabin != ""
    frame["HasCabin"] = has_cabin.astype(int)
    frame["CabinCount"] = cabin.map(lambda value: len(value.split())).astype(int)
    number = cabin.map(_first_cabin_number)
    frame["CabinSide"] = np.select(
        [number.isna(), number % 2 == 1], ["Unknown", "starboard"], "port"
    ).astype(object)
    deck = frame["Deck"].astype(str) if "Deck" in frame.columns else pd.Series("U", frame.index)
    frame["DeckGroup"] = np.select(
        [deck.isin(list("ABC")), deck.isin(list("DE")), deck.isin(list("FG"))],
        ["ABC", "DE", "FG"],
        "Unknown",
    ).astype(object)

    frame["IsMother"] = (female & (frame["Parch"] > 0) & (title != "Miss")).astype(int)
    frame["AgeClass"] = age * frame["Pclass"]
    frame["HasSibSp"] = (frame["SibSp"] > 0).astype(int)
    frame["HasParch"] = (frame["Parch"] > 0).astype(int)
    return frame


@dataclass(frozen=True)
class FeatureTest:
    """One add-or-remove comparison, mirroring its ledger row."""

    number: int
    action: str  # "add" or "remove"
    feature: str
    base_score: float
    score: float
    delta_vs_base: float
    se: float
    feature_value: float
    kept: bool
    run_id: str = ""


@dataclass
class Selection:
    """The whole run: what was tested, what survived, and the final row."""

    experiment_id: str
    params: dict[str, Any]
    baseline_features: list[str]
    baseline_score: float
    baseline_run_id: str
    features: list[str] = field(default_factory=list)
    score: float = 0.0
    final_run_id: str = ""
    tests: list[FeatureTest] = field(default_factory=list)
    stopped: str = ""
    seconds: float = 0.0


def fold_scores(
    frame, features: list[str], params: dict[str, Any], *, seed: int, folds: int, repeats: int
) -> np.ndarray:
    """Accuracy on every fold of a seeded repeated stratified K-fold.

    The fold generator and the estimator seed depend only on ``seed``, so two
    calls with different ``features`` are scored on identical splits by
    identically-seeded forests — which is what makes the difference *paired*.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

    from . import tune

    pipeline = tune.build_pipeline(MODEL_KEY, seed=seed)
    # One core per forest; the folds run in parallel instead.
    pipeline.set_params(**params, clf__n_jobs=1)
    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    return cross_val_score(
        pipeline, frame[features], frame["Survived"], cv=splitter, scoring="accuracy", n_jobs=-1
    )


def plan(baseline: list[str], candidates: list[str]) -> list[tuple[str, str]]:
    """The add phase in candidate order. The remove phase is planned later,
    from whatever survives the add phase."""
    return [("add", name) for name in candidates if name not in baseline]


def select(
    *,
    baseline: list[str] | None = None,
    candidates: list[str] | None = None,
    delta: float = DELTA,
    max_tests: int = MAX_TESTS,
    folds: int = CV_FOLDS,
    repeats: int = CV_REPEATS,
    tune_folds: int = 5,
    frozen_dir: Path = paths.FROZEN_DIR,
    log_path: Path = paths.RESULTS_LOG,
) -> Selection:
    """Greedy add-then-prune feature selection against one re-tuned model.

    1. Re-tune :data:`MODEL_KEY` on this experiment's training rows with the
       baseline features (one ``tune`` row). Params tuned in an earlier
       experiment are not reused: that experiment's training rows overlap this
       one's validation rows, the exact leak that sank exp-002.
    2. Score the baseline under repeated CV with those params frozen.
    3. Add each candidate in turn; keep it if it gains ``delta``.
    4. Try removing each surviving feature; drop it unless that costs ``delta``.

    Stops at ``max_tests`` tests. Every test is one ledger row.

    Raises:
        RuntimeError: No experiment is open — feature scores need a val split
            that none of this has seen.
    """
    from . import experiments, results_log, run_log, tune

    experiment = experiments.current_experiment()
    if experiment is None:
        raise RuntimeError(
            'no experiment is open. Run `make experiment-open NAME="..."` first — '
            "selection needs a validation split it has never seen."
        )
    seed = experiment.seed
    frame = tune.load_training_frame(frozen_dir)
    baseline = list(baseline or tune.resolve_features(frame))
    candidates = list(candidates or CANDIDATES)
    unknown = [c for c in candidates if c not in frame.columns]
    if unknown:
        raise ValueError(f"candidates not built by add_derived: {unknown}")

    started = time.perf_counter()
    cv_label = f"{folds}x{repeats} repeated stratified"
    run_log.record(
        "decision",
        step="features",
        detail=(
            f"{experiment.experiment_id}: rule declared before testing -- keep a feature iff "
            f"score(with) - score(without) >= {delta}; {cv_label} CV, paired; "
            f"model {MODEL_KEY} re-tuned once on baseline features then frozen; "
            f"cap {max_tests} tests"
        ),
    )

    tuned = tune.tune(
        MODEL_KEY,
        features=baseline,
        cv_folds=tune_folds,
        seed=seed,
        frozen_dir=frozen_dir,
        log_path=log_path,
        notes=f"features: re-tune {MODEL_KEY} once on baseline features; params then frozen",
    )
    if not tuned.params:
        raise RuntimeError(f"re-tuning {MODEL_KEY} failed; see ledger row {tuned.run_id}")
    params = {k: v for k, v in tuned.params.items() if k.startswith("clf__")}

    def log_row(features: list[str], scores: np.ndarray, extra: dict[str, Any], notes: str):
        return results_log.append_run(
            step="features",
            model=MODEL_KEY,
            params={**params, "features": features, "cv": cv_label, **extra},
            cv_score=float(scores.mean()),
            cv_std=float(scores.std()),
            cv_folds=folds * repeats,
            seed=seed,
            holdout_read=False,  # attested: selection reads the experiment's train rows only
            notes=notes,
            path=log_path,
        )

    current = list(baseline)
    current_scores = fold_scores(frame, current, params, seed=seed, folds=folds, repeats=repeats)
    baseline_run_id = log_row(
        current, current_scores, {"test": "baseline"}, "features: baseline under repeated CV"
    )
    selection = Selection(
        experiment_id=experiment.experiment_id,
        params=params,
        baseline_features=list(baseline),
        baseline_score=float(current_scores.mean()),
        baseline_run_id=baseline_run_id,
    )
    current_run_id = baseline_run_id

    def run_test(action: str, feature: str) -> None:
        nonlocal current, current_scores, current_run_id
        number = len(selection.tests) + 1
        trial = current + [feature] if action == "add" else [f for f in current if f != feature]
        try:
            scores = fold_scores(frame, trial, params, seed=seed, folds=folds, repeats=repeats)
        except Exception as exc:  # a failed test is a result too
            run_id = results_log.append_run(
                step="features",
                model=MODEL_KEY,
                params={**params, "features": trial, "test": f"{action} {feature}"},
                cv_score=None,
                cv_folds=folds * repeats,
                seed=seed,
                holdout_read=False,
                notes=f"features #{number} {action} {feature} FAILED: {type(exc).__name__}: {exc}",
                path=log_path,
            )
            run_log.record("step_failed", step="features", detail=str(exc), run_id=run_id)
            selection.tests.append(
                FeatureTest(number, action, feature, 0.0, float("nan"), 0, 0, 0, False, run_id)
            )
            return

        diff = scores - current_scores
        delta_vs_base = float(diff.mean())
        se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
        # The feature's value is always (with) - (without), whichever arm it is.
        value = delta_vs_base if action == "add" else -delta_vs_base
        kept = value >= delta
        accepted = (action == "add" and kept) or (action == "remove" and not kept)
        verdict = "KEEP" if kept else "DROP"
        run_id = log_row(
            trial,
            scores,
            {
                "test": f"{action} {feature}",
                "test_number": number,
                "feature_value": round(value, 6),
                "paired_se": round(se, 6),
                "decision": verdict.lower(),
                "base_run_id": current_run_id,
            },
            (
                f"features #{number}: {action} {feature} -> {verdict} "
                f"(value {value:+.4f} +/- {se:.4f} SE vs {current_run_id}; bar {delta})"
            ),
        )
        selection.tests.append(
            FeatureTest(
                number,
                action,
                feature,
                float(current_scores.mean()),
                float(scores.mean()),
                delta_vs_base,
                se,
                value,
                kept,
                run_id,
            )
        )
        run_log.record(
            "step_finished",
            step="features",
            detail=f"#{number} {action} {feature}: value {value:+.4f} (se {se:.4f}) -> {verdict}",
            run_id=run_id,
        )
        if accepted:
            current, current_scores, current_run_id = trial, scores, run_id

    for action, feature in plan(baseline, candidates):
        if len(selection.tests) >= max_tests:
            selection.stopped = f"test cap {max_tests} reached during the add phase"
            break
        run_test(action, feature)

    if not selection.stopped:
        for feature in list(current):
            if len(selection.tests) >= max_tests:
                selection.stopped = f"test cap {max_tests} reached during the remove phase"
                break
            if len(current) == 1:
                break
            run_test("remove", feature)

    selection.features = current
    selection.score = float(current_scores.mean())
    selection.final_run_id = current_run_id
    selection.stopped = selection.stopped or "every planned test ran"
    selection.seconds = time.perf_counter() - started
    experiments.record_spend("wall_clock_s", round(selection.seconds, 1))
    run_log.record(
        "decision",
        step="features",
        detail=(
            f"selected {len(current)} features, cv {selection.score:.4f} vs baseline "
            f"{selection.baseline_score:.4f}; final row {current_run_id} is the one "
            f"evaluate will score ({selection.stopped})"
        ),
        run_id=current_run_id,
    )
    return selection


def render_markdown(selection: Selection) -> str:
    """The step report: every test, its verdict, and what survived."""
    lines = [
        f"# 04 — feature selection ({selection.experiment_id})",
        "",
        f"Model **{MODEL_KEY}**, params re-tuned once on this experiment's training rows "
        f"and then frozen: `{selection.params}`.",
        "",
        f"Rule, declared before testing: keep a feature iff score(with) − score(without) "
        f"≥ **{DELTA}**, on {CV_FOLDS}×{CV_REPEATS} repeated stratified CV with both arms "
        "on identical folds. SE is the standard error of the "
        f"{CV_FOLDS * CV_REPEATS} paired fold differences; "
        "repeated-CV folds overlap, so it understates the true uncertainty.",
        "",
        f"- Baseline ({len(selection.baseline_features)} features): "
        f"**{selection.baseline_score:.4f}** (`{selection.baseline_run_id}`)",
        f"- Selected ({len(selection.features)} features): **{selection.score:.4f}** "
        f"(`{selection.final_run_id}`)",
        f"- Tests run: {len(selection.tests)} of a {MAX_TESTS} cap — {selection.stopped}; "
        f"{selection.seconds:.0f}s",
        "",
        "| # | test | value (with − without) | paired SE | verdict | cv after |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for test in selection.tests:
        lines.append(
            f"| {test.number} | {test.action} `{test.feature}` | {test.feature_value:+.4f} | "
            f"{test.se:.4f} | {'keep' if test.kept else 'drop'} | {test.score:.4f} |"
        )
    added = [t.feature for t in selection.tests if t.action == "add" and t.kept]
    removed = [t.feature for t in selection.tests if t.action == "remove" and not t.kept]
    lines += [
        "",
        f"**Added:** {', '.join(added) or 'none'}  ",
        f"**Removed:** {', '.join(removed) or 'none'}  ",
        f"**Final set:** {', '.join(selection.features)}",
        "",
        "## Candidate rationale",
        "",
    ]
    lines += [f"- `{name}` — {why}" for name, why in CANDIDATES.items()]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """``make features`` entry point."""
    import argparse

    from . import runs

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-tests", type=int, default=MAX_TESTS)
    args = parser.parse_args(argv)

    try:
        selection = select(max_tests=args.max_tests)
    except (RuntimeError, ValueError) as exc:
        print(f"features failed: {exc}")
        return 1

    report = render_markdown(selection)
    written = runs.run_dir() / "04_features.md"
    written.write_text(report, encoding="utf-8")
    runs.publish(written)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
