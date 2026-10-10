"""Nested CV: what the feature search is worth once it pays for its own bias.

A score reported after a search is biased upward, because the search chose the
configuration *using* the number it then reports. exp-006 published cv 0.8401
after ~36 comparisons and nothing here measured how much of that was selection.
The cv→val gap cannot: at SE 0.032 it is eight times too coarse, and split
variance and selection bias are confounded inside it (``docs/MEASUREMENT.md``).

Nested CV measures it directly. An outer fold runs the **entire** selection
procedure on its own training portion, never seeing its test portion, and scores
the selected model there. The average over outer folds is what the *procedure*
earns on fresh data, as opposed to what the chosen configuration scored on the
data that chose it.

Two arms, not one number
    A standalone nested estimate is ceremony. Outer test folds hold ~114 rows,
    so the estimate's own SE is ~0.015-0.020 — no better than the validation
    read it would be compared against, for five times the compute. Trading a
    0.032-SE number for a 0.018-SE one resolves nothing and changes no decision.

    So this runs two arms on *identical* outer folds and reports the paired
    difference:

    **A — search**
        ``features.greedy_search`` re-run inside each outer fold's training rows.
    **B — no search**
        The fixed baseline feature set, unchanged.

    ``A - B`` answers the question that matters: *does searching, net of its own
    selection bias, beat not searching at this sample size?* Paired, so outer
    fold noise largely cancels; and it needs no validation read at all.

What it does not cover
    Hyperparameters are held fixed across both arms and every outer fold, at
    whatever the real run tuned them to. So this is the selection bias of the
    **feature search**, which is the open question. Nesting the hyperparameter
    search too would measure a larger procedure than the one we run, and cost
    several times as much.

Why the inner loop is not thinned
    A cheaper inner loop estimates a cheaper procedure. The inner comparator is
    the same ``5x3`` paired repeated CV against the same ``DELTA``, through the
    same :func:`titanic_agent.features.greedy_search` the real run calls — not a
    second copy that could drift from it. What *is* trimmed is the candidate
    list, which is the actual cost driver: 24 candidates times five outer folds
    is affordable once but not per experiment, and 19 of those 24 were measured
    negative in exp-006.

Cost and the ~2-minute rule
    Budgeted at ~3 minutes, over the laptop rule in CLAUDE.md, which carves this
    step out by name: the rule exists to stop an agent waiting out a slow
    *search*, not to forbid a number that can only be had by paying for it. Run
    once per searching experiment, never inside ``make loop``.

Falsifier
    The nested estimate must come out **below** the selected CV. Nesting cannot
    help a procedure; if it comes out above, the implementation is wrong and the
    number should be thrown away rather than reported.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import compare as compare_module
from . import paths
from .compare import CV_FOLDS, CV_REPEATS, DELTA, MODEL_KEY

#: Outer folds. Five keeps outer test folds at ~114 rows and the whole run
#: inside its budget; the paired design is what buys the precision, not the
#: number of folds.
OUTER_FOLDS: int = 5

#: Candidates the inner search may consider. Trimmed from the 24 in
#: :data:`titanic_agent.features.CANDIDATES` to the ones with surviving prior
#: plausibility after exp-006 measured 19 of 24 negative: the best-measured add
#: (``TitleFine``), the three recorded as worth re-testing as swaps, and the
#: handful whose rationale does not depend on the tree being able to use them.
#: Trimming the candidate list changes the procedure being measured, which is
#: why the count goes in the report and the ledger row.
NESTED_CANDIDATES: tuple[str, ...] = (
    "TitleFine",
    "WomanOrChild",
    "AgeClass",
    "FamilyBucket",
    "FarePerPerson",
    "IsChild",
    "SexPclass",
    "HasCabin",
)


@dataclass
class OuterFold:
    """One outer fold: both arms, scored on rows neither arm's search saw."""

    number: int
    n_train: int
    n_test: int
    selected: list[str]
    score_search: float
    score_baseline: float
    inner_cv_selected: float
    inner_tests: int

    @property
    def delta(self) -> float:
        return self.score_search - self.score_baseline


@dataclass
class NestedResult:
    """The whole nested run."""

    model: str
    params: dict[str, Any]
    baseline: list[str]
    candidates: list[str]
    seed: int
    n_rows: int
    #: How many candidates the untrimmed procedure would have considered.
    candidates_available: int = 0
    folds: list[OuterFold] = field(default_factory=list)
    reported_cv: float | None = None
    seconds: float = 0.0

    @property
    def nested_search(self) -> float:
        return float(np.mean([f.score_search for f in self.folds]))

    @property
    def nested_baseline(self) -> float:
        return float(np.mean([f.score_baseline for f in self.folds]))

    @property
    def paired(self) -> compare_module.PairedResult:
        """Arm A minus arm B, paired across outer folds."""
        return compare_module.compare_scores(
            np.array([f.score_search for f in self.folds]),
            np.array([f.score_baseline for f in self.folds]),
            label="nested: search vs no search",
            n_rows=self.n_rows,
            seed=self.seed,
            repeats=1,
        )

    @property
    def selection_bias(self) -> float | None:
        """Reported (post-search) CV minus the nested estimate of the procedure.

        Positive means the published number was optimistic by this much, which
        is the only direction nesting can legitimately produce.
        """
        if self.reported_cv is None:
            return None
        return self.reported_cv - self.nested_search

    @property
    def implausible(self) -> bool:
        """The falsifier: nesting cannot improve on the number it corrects."""
        bias = self.selection_bias
        return bias is not None and bias < 0


def _fit_and_score(frame, train_index, test_index, features: list[str], params, *, seed: int):
    """Fit on the outer training rows, score on the outer test rows."""
    from sklearn.metrics import accuracy_score

    from . import tune

    pipeline = tune.build_pipeline(MODEL_KEY, seed=seed)
    pipeline.set_params(**params, clf__n_jobs=-1)
    train, test = frame.iloc[train_index], frame.iloc[test_index]
    pipeline.fit(train[features], train["Survived"])
    return float(accuracy_score(test["Survived"], pipeline.predict(test[features])))


def resolve_params(
    experiment_id: str | None = None, log_path: Path | None = None
) -> tuple[dict[str, Any], list[str], float | None, str]:
    """Recover the real run's frozen params, baseline features and reported CV.

    Reads them off the ledger rather than restating them, so the nested estimate
    corrects a number that actually exists in the chain.
    """
    from . import results_log

    frame = results_log.read_log(log_path)
    rows = frame[frame["step"] == "features"] if not frame.empty else frame
    if experiment_id and len(rows):
        rows = rows[rows["experiment_id"].astype(str) == experiment_id]
    if not len(rows):
        raise RuntimeError(
            "no `features` rows in the ledger to correct. Run `make features` inside an "
            "experiment first — nesting measures the bias of a search that happened."
        )

    baselines = [
        row
        for _, row in rows.iterrows()
        if isinstance(row["params"], dict) and row["params"].get("test") == "baseline"
    ]
    if not baselines:
        raise RuntimeError("found `features` rows but none marked test=baseline")
    base = baselines[-1]
    params = {k: v for k, v in base["params"].items() if k.startswith("clf__")}
    baseline_features = list(base["params"].get("features") or [])

    # The reported CV is the best scored row of that search -- the number the
    # search chose and then published.
    scored = rows.dropna(subset=["cv_score"])
    reported = float(scored["cv_score"].astype(float).max()) if len(scored) else None
    return params, baseline_features, reported, str(base["run_id"])


def run(
    *,
    experiment_id: str | None = None,
    candidates: list[str] | None = None,
    outer_folds: int = OUTER_FOLDS,
    delta: float = DELTA,
    folds: int = CV_FOLDS,
    repeats: int = CV_REPEATS,
    frozen_dir: Path = paths.FROZEN_DIR,
    log_path: Path = paths.RESULTS_LOG,
) -> NestedResult:
    """Run the two-armed nested estimate and append one ledger row.

    Reads the experiment's training rows only. No validation read, no holdout.
    """
    from sklearn.model_selection import StratifiedKFold

    from . import experiments, features, results_log, run_log, tune

    params, baseline, reported_cv, base_run_id = resolve_params(experiment_id, log_path)
    chosen = list(candidates or NESTED_CANDIDATES)

    experiment = experiments.current_experiment()
    if experiment_id and (experiment is None or experiment.experiment_id != experiment_id):
        # Measuring a closed experiment: use *its* training rows, so the outer
        # folds partition the same data the search actually saw.
        import pandas as pd

        from . import freeze

        freeze.require_frozen(frozen_dir)
        working = tune.add_derived(pd.read_csv(Path(frozen_dir) / "train.csv"))
        target = experiments.load_experiment(experiment_id)
        frame = working.set_index("PassengerId", drop=False).loc[target.train_ids]
        seed = target.seed
    else:
        frame = tune.load_training_frame(frozen_dir)
        seed = experiment.seed if experiment is not None else paths.SEED

    missing = [name for name in chosen + baseline if name not in frame.columns]
    if missing:
        raise ValueError(f"not built by add_derived: {missing}")

    started = time.perf_counter()
    run_log.record(
        "decision",
        step="nested",
        detail=(
            f"nested CV declared before running: {outer_folds} outer folds, inner "
            f"{folds}x{repeats} paired CV against bar {delta} through "
            f"features.greedy_search; arm A = search, arm B = fixed {len(baseline)}-feature "
            f"baseline, identical outer folds; {len(chosen)} candidates (trimmed from "
            f"{len(features.CANDIDATES)}); hyperparameters held fixed from {base_run_id}; "
            f"correcting reported cv {reported_cv}"
        ),
    )

    result = NestedResult(
        model=MODEL_KEY,
        params=params,
        baseline=list(baseline),
        candidates=chosen,
        seed=seed,
        n_rows=len(frame),
        candidates_available=len(features.CANDIDATES),
        reported_cv=reported_cv,
    )

    splitter = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    labels = frame["Survived"]
    for number, (train_index, test_index) in enumerate(splitter.split(frame, labels), 1):
        inner = frame.iloc[train_index]
        search = features.greedy_search(
            inner,
            list(baseline),
            chosen,
            params,
            seed=seed,
            delta=delta,
            folds=folds,
            repeats=repeats,
            max_tests=features.MAX_TESTS,
        )
        fold = OuterFold(
            number=number,
            n_train=len(train_index),
            n_test=len(test_index),
            selected=list(search.selected),
            score_search=_fit_and_score(
                frame, train_index, test_index, search.selected, params, seed=seed
            ),
            score_baseline=_fit_and_score(
                frame, train_index, test_index, list(baseline), params, seed=seed
            ),
            inner_cv_selected=float(search.selected_scores.mean()),
            inner_tests=len(search.steps),
        )
        result.folds.append(fold)
        run_log.record(
            "step_finished",
            step="nested",
            detail=(
                f"outer fold {number}/{outer_folds}: {len(search.selected)} features from "
                f"{fold.inner_tests} inner tests, inner cv {fold.inner_cv_selected:.4f}, "
                f"outer search {fold.score_search:.4f} vs baseline "
                f"{fold.score_baseline:.4f} ({fold.delta:+.4f})"
            ),
        )

    result.seconds = time.perf_counter() - started
    paired = result.paired
    bias = result.selection_bias
    notes = (
        f"nested CV, {outer_folds} outer folds, inner {folds}x{repeats} paired against "
        f"{delta}; arm A search vs arm B fixed baseline on identical outer folds. "
        f"{paired.describe()}. nested(search) {result.nested_search:.4f}, "
        f"nested(baseline) {result.nested_baseline:.4f}"
    )
    if bias is not None:
        notes += (
            f"; reported cv {result.reported_cv:.4f} - nested {result.nested_search:.4f} = "
            f"selection bias {bias:+.4f}"
        )
    if result.implausible:
        notes += (
            " -- IMPLAUSIBLE: nesting cannot improve on the number it corrects, so this "
            "estimate is not trustworthy; treat the implementation as suspect"
        )

    run_id = results_log.append_run(
        step="nested",
        model=MODEL_KEY,
        params={
            **params,
            "features": list(baseline),
            "outer_folds": outer_folds,
            "inner_cv": f"{folds}x{repeats} repeated stratified",
            "candidates": len(chosen),
            "candidates_trimmed_from": len(features.CANDIDATES),
            "arms": "search vs fixed baseline",
            "nested_baseline": round(result.nested_baseline, 6),
            "reported_cv": result.reported_cv,
            "selection_bias": None if bias is None else round(bias, 6),
            "corrects_run_id": base_run_id,
            **paired.as_params(),
        },
        cv_score=result.nested_search,
        cv_std=float(np.std([f.score_search for f in result.folds])),
        cv_folds=outer_folds,
        seed=seed,
        holdout_read=False,  # attested: outer folds partition the training rows only
        notes=notes,
        path=log_path,
    )
    if experiment is not None:
        experiments.record_spend("wall_clock_s", round(result.seconds, 1))
    run_log.record("decision", step="nested", detail=notes, run_id=run_id)
    return result


def render_markdown(result: NestedResult) -> str:
    """The step report."""
    paired = result.paired
    bias = result.selection_bias
    lines = [
        "# 05 — nested CV: what the feature search is actually worth",
        "",
        f"Model **{result.model}**, hyperparameters held fixed at `{result.params}` across "
        "both arms and every outer fold.",
        "",
        f"{len(result.folds)} outer folds over {result.n_rows} training rows. Inside each, "
        f"the **same** `features.greedy_search` the real run calls, on that fold's training "
        f"rows only, over {len(result.candidates)} candidates "
        f"(trimmed from {result.candidates_available}). Arm B is the fixed "
        f"{len(result.baseline)}-feature baseline. Both arms are scored on the outer test "
        f"rows, which neither arm's search saw. {result.seconds:.0f}s.",
        "",
        "| outer fold | test rows | features chosen | inner cv | A search | B baseline | A − B |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in result.folds:
        lines.append(
            f"| {fold.number} | {fold.n_test} | {len(fold.selected)} | "
            f"{fold.inner_cv_selected:.4f} | {fold.score_search:.4f} | "
            f"{fold.score_baseline:.4f} | {fold.delta:+.4f} |"
        )
    lines += [
        "",
        "## Does searching beat not searching?",
        "",
        f"- **{paired.describe()}**",
        f"- nested(search) **{result.nested_search:.4f}**, "
        f"nested(baseline) **{result.nested_baseline:.4f}**",
        "",
    ]
    if bias is not None:
        lines += [
            "## How optimistic was the published number?",
            "",
            f"- reported CV after the search: **{result.reported_cv:.4f}**",
            f"- nested estimate of the same procedure: **{result.nested_search:.4f}**",
            f"- **selection bias: {bias:+.4f}**",
            "",
        ]
        if result.implausible:
            lines += [
                "> **This estimate is not trustworthy.** Nesting cannot improve on the "
                "number it corrects — a negative bias means the implementation is wrong, "
                "not that the search was better than reported. Do not quote it.",
                "",
            ]
    lines += [
        "## Scope",
        "",
        "- Hyperparameters are **not** nested: they are held fixed from the real run, so "
        "this is the selection bias of the feature search alone.",
        "- The candidate list is trimmed, so this measures the trimmed procedure. 19 of the "
        f"{result.candidates_available} available candidates measured negative in exp-006; "
        "keeping all of them would cost "
        "several times as much to estimate a search nobody intends to repeat.",
        "- Outer test folds hold ~"
        f"{result.n_rows // max(len(result.folds), 1)} rows each, so the absolute nested "
        "numbers are noisy. The paired difference is the measurement; the absolute "
        "levels are context.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """``make nested`` entry point."""
    import argparse

    from . import runs

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--experiment", default=None, help="measure this experiment's search (may be closed)"
    )
    parser.add_argument("--outer-folds", type=int, default=OUTER_FOLDS)
    args = parser.parse_args(argv)

    try:
        result = run(experiment_id=args.experiment, outer_folds=args.outer_folds)
    except (RuntimeError, ValueError) as exc:
        print(f"nested failed: {exc}")
        return 1

    report = render_markdown(result)
    written = runs.run_dir() / "05_nested.md"
    written.write_text(report, encoding="utf-8")
    runs.publish(written)
    print(report)
    return 1 if result.implausible else 0


if __name__ == "__main__":
    raise SystemExit(main())
