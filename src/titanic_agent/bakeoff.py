"""A bake-off: many hunches in one experiment, each paired against the champion.

Every earlier experiment pulled one lever and held everything else still. That
is the right way to learn what a lever is worth, and it is slow: at one
hypothesis per experiment, and with a single comparison resolving ~0.003 at
best, the backlog grows faster than it drains. This module trades attribution
for breadth on purpose. It runs a list of **arms** — model, hyperparameters,
features and fitted transforms chosen by hand, each with a one-line reason it
should generalise — and scores every one of them against the reigning
:mod:`titanic_agent.champion` on identical folds.

Breadth has a price, and it is paid here rather than ignored
    The best of N arms is a selection statistic. With twenty arms whose true
    effect is zero and a paired SE of ~0.003, the best one still shows about
    +1.9 SE — roughly +0.006, twice the bar — from noise alone. So the
    winner's delta is never the decision. Three things are, all declared
    before any arm runs:

    1. **The selection delta** — winner vs champion on the experiment's folds.
       Reported with the null expectation for the best of N beside it.
    2. **Confirmation on fresh fold seeds** — winner and champion re-scored
       under seeds the selection never saw. This removes the luck of the fold
       draw. It cannot remove the luck of the *rows*, which is why there is a
       third.
    3. **Nested CV of the whole procedure** — ``pick the best arm`` run inside
       each outer fold, scored on outer rows it never saw, paired against the
       champion on the same outer folds. This is the only one of the three that
       charges the search for its own selection bias.

    The champion is replaced only if all three agree. See :func:`decide`.

Arms are not single-lever, and the report says so
    Most arms differ from the champion by one change, so a large delta is
    attributable. A few are deliberate kitchen sinks; if one wins, the next
    experiment's job is to take it apart.

Fitted state lives inside the fold
    Three transforms here learn from rows: Age imputed by title median, ticket
    group size, and travel-group survival. The last one reads the **label**, so
    it is the classic Titanic leak if computed outside the fold. All three are
    fitted in :class:`Engineer`, which is the first step of every pipeline, so
    cross-validation refits them per fold and a validation or test row never
    contributes its own label. On training rows the group rate is
    leave-one-out: a passenger's own survival is excluded from their feature.

    Rule 4 is not at issue: nothing here joins anything external. Groups are
    formed from columns of the frozen data — surname, fare, ticket — and rates
    from the training fold's own labels.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from . import compare as compare_module
from . import paths
from .compare import CV_FOLDS, CV_REPEATS, DELTA

#: Fresh fold seeds for confirming the winner. Offsets from the experiment seed
#: start at 1, because offset 0 *is* the selection draw.
CONFIRM_SEEDS: int = 5

#: Outer folds for the nested measurement of the whole bake-off.
OUTER_FOLDS: int = 5

#: Columns each fitted transform produces. Declared so ``make leakage`` can vouch
#: for them: they are derived from frozen columns, inside the fold, and appear in
#: no frame :func:`titanic_agent.tune.add_derived` builds.
FITTED_OUTPUTS: dict[str, tuple[str, ...]] = {
    "age_by_title": (),  # rewrites Age in place
    "ticket_size": ("TicketGroupSize",),
    "group": ("GroupSurvival", "GroupKnown", "WCGSurvival", "WCGKnown"),
}

#: Default for a group survival rate when no labelled mate exists. The midpoint
#: rather than the base rate, so "unknown" is not quietly "probably died".
UNKNOWN_RATE: float = 0.5


# --------------------------------------------------------------------------- #
# fitted transforms
# --------------------------------------------------------------------------- #


def _surname(names) -> Any:
    return names.astype(str).str.split(",").str[0].str.strip()


def _group_keys(frame) -> dict[str, Any]:
    """Two ways passengers travelled together, both from columns we hold.

    ``family``: surname and fare. A family booked together paid one fare, so
    surname alone over-merges unrelated Smiths and fare alone merges strangers.
    ``ticket``: the ticket itself, which also catches servants, friends and
    in-laws travelling under another name.
    """
    fare = frame["Fare"].round(2).astype(str)
    return {
        "family": _surname(frame["Name"]) + "|" + fare,
        "ticket": frame["Ticket"].astype(str),
    }


def _woman_or_child(frame):
    """Female, or a boy by title: the evacuation rule as one flag."""
    boy = frame["Name"].astype(str).str.contains(r"\bMaster\.", regex=True)
    return (frame["Sex"] == "female") | boy


class Engineer(BaseEstimator, TransformerMixin):
    """Apply fitted transforms, learning only from the rows passed to ``fit``.

    Args:
        steps: Any of ``age_by_title``, ``ticket_size``, ``group``, applied in
            that order regardless of how they are listed.
    """

    def __init__(self, steps: tuple[str, ...] = ()):
        self.steps = steps

    def fit(self, X, y=None):
        import pandas as pd

        unknown = set(self.steps) - set(FITTED_OUTPUTS)
        if unknown:
            raise ValueError(f"unknown fitted transforms: {sorted(unknown)}")
        self.fit_ids_ = pd.Index(X["PassengerId"].astype(int))
        if "age_by_title" in self.steps:
            self.age_global_ = float(X["Age"].median())
            self.age_by_title_ = X.groupby("Title")["Age"].median().to_dict()
        if "ticket_size" in self.steps:
            self.tickets_ = pd.Series(X["Ticket"].astype(str).to_numpy(), index=self.fit_ids_)
        if "group" in self.steps:
            if y is None:
                raise ValueError("the group transform needs labels at fit time")
            labels = pd.Series(np.asarray(y, dtype=float), index=self.fit_ids_)
            keys = _group_keys(X)
            wc = _woman_or_child(X).to_numpy()
            self.labels_ = labels
            self.wc_ = pd.Series(wc, index=self.fit_ids_)
            self.stats_ = {}
            for name, key in keys.items():
                key = pd.Series(key.to_numpy(), index=self.fit_ids_)
                self.stats_[name] = {
                    "all": labels.groupby(key).agg(["sum", "count"]),
                    "wc": labels[wc].groupby(key[wc]).agg(["sum", "count"]),
                }
        return self

    def _rate(self, X, *, subset: str):
        """Leave-one-out survival of each row's labelled mates; NaN if none.

        Family key first; the ticket key fills rows whose family has no
        labelled mate.
        """
        import pandas as pd

        ids = X["PassengerId"].astype(int).to_numpy()
        in_fit = pd.Index(ids).isin(self.fit_ids_)
        own_label = self.labels_.reindex(ids).fillna(0).to_numpy()
        own_counts = in_fit.copy()
        if subset == "wc":
            own_counts = in_fit & self.wc_.reindex(ids).fillna(False).to_numpy().astype(bool)
        rate = np.full(len(X), np.nan)
        known = np.zeros(len(X), dtype=int)
        for name, key in _group_keys(X).items():
            table = self.stats_[name][subset]
            total = key.map(table["sum"]).fillna(0).to_numpy() - np.where(own_counts, own_label, 0)
            count = key.map(table["count"]).fillna(0).to_numpy() - own_counts.astype(int)
            fill = np.isnan(rate) & (count > 0)
            rate[fill] = total[fill] / count[fill]
            known = np.maximum(known, (count > 0).astype(int))
        return rate, known

    def transform(self, X):
        import pandas as pd

        out = X.copy()
        if "age_by_title" in self.steps:
            fill = out["Title"].map(self.age_by_title_).fillna(self.age_global_)
            out["Age"] = out["Age"].fillna(fill)
        if "ticket_size" in self.steps:
            # Counting needs no label, so the rows being transformed count too:
            # a test passenger's companions in the test batch are still companions.
            ids = out["PassengerId"].astype(int)
            batch = pd.Series(out["Ticket"].astype(str).to_numpy(), index=ids.to_numpy())
            union = pd.concat([self.tickets_, batch[~batch.index.isin(self.tickets_.index)]])
            out["TicketGroupSize"] = out["Ticket"].astype(str).map(union.value_counts()).to_numpy()
        if "group" in self.steps:
            rate, known = self._rate(out, subset="all")
            out["GroupSurvival"] = np.where(np.isnan(rate), UNKNOWN_RATE, rate)
            out["GroupKnown"] = known
            rate, known = self._rate(out, subset="wc")
            out["WCGSurvival"] = np.where(np.isnan(rate), UNKNOWN_RATE, rate)
            out["WCGKnown"] = known
        return out


class Select(BaseEstimator, TransformerMixin):
    """Keep only the model-facing columns. Identifiers stop here."""

    def __init__(self, columns: tuple[str, ...] = ()):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[list(self.columns)]


class RuleClassifier(ClassifierMixin, BaseEstimator):
    """A hand-written rule, so a model has something simple to be compared with.

    ``gender``
        Women survive, men do not. The floor every Titanic model must clear.
    ``wcg``
        Gender, corrected by the woman-child group: a boy survives if every
        labelled woman or child travelling with him did, and a woman dies if
        every one of them died. Learns nothing beyond what :class:`Engineer`
        computed in-fold.
    """

    def __init__(self, kind: str = "gender"):
        self.kind = kind

    def fit(self, X, y):
        if self.kind not in ("gender", "wcg"):
            raise ValueError(f"unknown rule {self.kind!r}")
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        survive = (X["Sex"] == "female").to_numpy()
        if self.kind == "wcg":
            known = X["WCGKnown"].to_numpy() > 0
            rate = X["WCGSurvival"].to_numpy()
            boy = (X["Title"] == "Master").to_numpy()
            survive = np.where(boy & known & (rate >= 1.0), True, survive)
            survive = np.where(~boy & survive & known & (rate <= 0.0), False, survive)
        return survive.astype(int)

    def predict_proba(self, X):
        p = self.predict(X).astype(float)
        return np.column_stack([1 - p, p])


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Arm:
    """One configuration and the reason it might beat the champion."""

    name: str
    hypothesis: str
    model: str
    features: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    engineer: tuple[str, ...] = ()
    #: For ``soft_vote`` and ``stack``: names of other arms.
    members: tuple[str, ...] = ()

    def config(self, registry: dict[str, Arm] | None = None) -> dict[str, Any]:
        """The JSON form logged to the ledger and stored by the champion.

        Ensemble members are inlined, so a config rebuilds without the registry
        that produced it — which is what lets a later experiment re-score it.
        """
        payload: dict[str, Any] = {
            "name": self.name,
            "model": self.model,
            "params": dict(self.params),
            "features": list(self.features),
            "engineer": list(self.engineer),
            "members": [],
        }
        if self.members:
            if registry is None:
                raise ValueError(f"{self.name} has members; pass the registry to inline them")
            payload["members"] = [registry[m].config(registry) for m in self.members]
            payload["features"] = sorted({f for m in payload["members"] for f in m["features"]})
        return payload


def from_config(config: dict[str, Any]) -> Arm:
    """Rebuild an :class:`Arm` from its logged form (members stay inlined)."""
    return Arm(
        name=config["name"],
        hypothesis="",
        model=config["model"],
        features=tuple(config.get("features", ())),
        params=dict(config.get("params", {})),
        engineer=tuple(config.get("engineer", ())),
    )


GROUP = ("GroupSurvival", "GroupKnown")
WCG = ("WCGSurvival", "WCGKnown")
#: Bucketed and one-hot encodings for the linear models. exp-006 tested these on
#: a random forest, the model least able to benefit; a linear model cannot
#: express a non-monotone age or family effect without them.
RICH = (
    "Sex",
    "PclassCat",
    "SexPclass",
    "AgeBin",
    "FamilyBucket",
    "TitleFine",
    "FareLog",
    "Embarked",
    "HasCabin",
    "IsChild",
)


def arms_for(champion_config: dict[str, Any]) -> dict[str, Arm]:
    """Every arm in this bake-off, champion first. Order is run order."""
    base = tuple(champion_config["features"])
    rf = (
        dict(champion_config["params"])
        if champion_config["model"] == "random_forest"
        else {
            "clf__n_estimators": 400,
            "clf__min_samples_leaf": 3,
            "clf__max_depth": None,
        }
    )
    swap_title = tuple("TitleFine" if f == "Title" else f for f in base)

    def a(name, hypothesis, model, features=(), params=None, engineer=(), members=()):
        return Arm(name, hypothesis, model, tuple(features), dict(params or {}), engineer, members)

    listed = [
        a(
            "champion",
            "The reference. Re-scored here so every delta is paired.",
            champion_config["model"],
            base,
            champion_config["params"],
            tuple(champion_config.get("engineer", ())),
        ),
        a(
            "rf_titlefine",
            "SWAP Title for TitleFine. 'Rare' mixes clergy (none survived) with "
            "nobility (all did); as an add it diluted, as a swap it should not.",
            "random_forest",
            swap_title,
            rf,
        ),
        a(
            "rf_maxfeat_half",
            "max_features=0.5. With sqrt of ~25 one-hot columns, the strong features "
            "are rarely sampled per split; the untried dilution hypothesis.",
            "random_forest",
            base,
            {**rf, "clf__max_features": 0.5},
        ),
        a(
            "rf_shallow",
            "max_depth=7. Fully grown trees on 569 rows memorise singletons; a cap "
            "trades a little bias for variance.",
            "random_forest",
            base,
            {**rf, "clf__max_depth": 7},
        ),
        a(
            "rf_age_by_title",
            "Impute Age by title median in-fold. A missing-age Master is a boy, not "
            "a 28-year-old; median imputation erases the one thing Age tells us.",
            "random_forest",
            base,
            rf,
            ("age_by_title",),
        ),
        a(
            "rf_ticket_size",
            "Add TicketGroupSize. FamilySize misses companions on one ticket who are "
            "not kin; group size had the same rise-then-collapse shape.",
            "random_forest",
            base + ("TicketGroupSize",),
            rf,
            ("ticket_size",),
        ),
        a(
            "rf_group",
            "Add in-fold group survival (family, then ticket). Groups evacuated "
            "together, so a mate's fate is evidence about one's own location and choices.",
            "random_forest",
            base + GROUP,
            rf,
            ("group",),
        ),
        a(
            "rf_wcg",
            "Add woman-child-group survival only. Adult men in a group died whatever "
            "the group did; excluding them sharpens the signal for women and boys.",
            "random_forest",
            base + WCG,
            rf,
            ("group",),
        ),
        a(
            "rf_kitchen",
            "Kitchen sink: TitleFine swap, title-imputed age, ticket size, both group "
            "rates, max_features 0.5. If it wins, the next experiment takes it apart.",
            "random_forest",
            swap_title + ("TicketGroupSize",) + GROUP + WCG,
            {**rf, "clf__max_features": 0.5},
            ("age_by_title", "ticket_size", "group"),
        ),
        a(
            "et_group",
            "ExtraTrees with group rates. Random thresholds average more smoothly over "
            "a 0-1 rate than greedy RF splits, which chase the exact LOO values.",
            "extra_trees",
            base + GROUP + WCG,
            {"clf__n_estimators": 400, "clf__min_samples_leaf": 3, "clf__max_features": 0.5},
            ("group",),
        ),
        a(
            "gb_baseline",
            "Slow, subsampled boosting (300 x 0.05, depth 3, subsample 0.8). Earlier GB "
            "grids used full-sample fits only, which boost noise on 569 rows.",
            "gradient_boosting",
            base,
            {
                "clf__n_estimators": 300,
                "clf__learning_rate": 0.05,
                "clf__max_depth": 3,
                "clf__subsample": 0.8,
            },
        ),
        a(
            "gb_group",
            "The same boosting with both group rates.",
            "gradient_boosting",
            base + GROUP + WCG,
            {
                "clf__n_estimators": 300,
                "clf__learning_rate": 0.05,
                "clf__max_depth": 3,
                "clf__subsample": 0.8,
            },
            ("group",),
        ),
        a(
            "hgb_group",
            "Regularised histogram boosting with group rates: 20-row leaves and L2 1.0 "
            "keep it from splitting on a handful of LOO values.",
            "hist_gradient_boosting",
            base + GROUP + WCG,
            {
                "clf__learning_rate": 0.05,
                "clf__max_iter": 200,
                "clf__max_leaf_nodes": 15,
                "clf__min_samples_leaf": 20,
                "clf__l2_regularization": 1.0,
            },
            ("group",),
        ),
        a(
            "logit_rich",
            "Logistic on bucketed/one-hot encodings and Sex x class. The untried "
            "finding: binning helps the model that cannot bin for itself.",
            "logistic",
            RICH,
            {"clf__C": 0.5},
        ),
        a(
            "logit_rich_wcg",
            "The same logistic with woman-child-group rates.",
            "logistic",
            RICH + WCG,
            {"clf__C": 0.5},
            ("group",),
        ),
        a(
            "svc_rich_wcg",
            "RBF SVC on the rich encoding plus WCG rates: a smooth boundary between "
            "the linear and tree arms, for ensemble diversity as much as on its own.",
            "svc",
            RICH + WCG,
            {"clf__estimator__C": 1.0, "clf__estimator__gamma": "scale"},
            ("group",),
        ),
        a(
            "rule_gender",
            "Women live, men die. Not a contender: the floor, so 'better than the "
            "champion' can be read against 'better than nothing'.",
            "rule_gender",
            ("Sex",),
        ),
        a(
            "rule_wcg",
            "Gender corrected by the woman-child group: the best-known hand rule for "
            "this ship. If no model beats it, the models are not learning much more.",
            "rule_wcg",
            ("Sex", "Title") + WCG,
            {},
            ("group",),
        ),
        a(
            "vote_base",
            "Soft vote of champion, boosting and the rich logistic: three different "
            "inductive biases, none using group rates.",
            "soft_vote",
            members=("champion", "gb_baseline", "logit_rich"),
        ),
        a(
            "vote_group",
            "Soft vote of the four group-aware learners.",
            "soft_vote",
            members=("rf_wcg", "gb_group", "logit_rich_wcg", "svc_rich_wcg"),
        ),
        a(
            "stack_group",
            "Stack the same four under a logistic meta-learner, which can learn to "
            "trust one member where the others disagree.",
            "stack",
            members=("rf_wcg", "gb_group", "logit_rich_wcg", "svc_rich_wcg"),
        ),
    ]
    return {arm.name: arm for arm in listed}


def _without(config: dict[str, Any], columns: tuple[str, ...]) -> dict[str, Any]:
    """``config`` minus some features, dropping the ``group`` step if it is now unused."""
    undone = {**config, "features": [f for f in config["features"] if f not in columns]}
    if "group" in undone["engineer"] and not set(undone["features"]) & set(GROUP + WCG):
        undone["engineer"] = [e for e in undone["engineer"] if e != "group"]
    return undone


#: Each change a champion may carry, how to undo it, and why it might matter.
#: ``undo`` returns None when the champion does not carry the change.
COMPONENTS: dict[str, tuple[str, Any]] = {
    "titlefine_swap": (
        "Title back in place of TitleFine. Tests whether the finer titles earn anything "
        "once group rates already separate the families that 'Rare' mixed.",
        lambda c: (
            {**c, "features": ["Title" if f == "TitleFine" else f for f in c["features"]]}
            if "TitleFine" in c["features"] and "Title" not in c["features"]
            else None
        ),
    ),
    "age_by_title": (
        "Median Age instead of title-median Age. Boys with missing ages may already be "
        "caught by Title and the WCG rate, which would make the imputation redundant.",
        lambda c: (
            {**c, "engineer": [e for e in c["engineer"] if e != "age_by_title"]}
            if "age_by_title" in c["engineer"]
            else None
        ),
    ),
    "ticket_size": (
        "Drop TicketGroupSize. Group rates already know who shares a ticket; the size may "
        "add nothing to them.",
        lambda c: (
            {
                **_without(c, ("TicketGroupSize",)),
                "engineer": [e for e in c["engineer"] if e != "ticket_size"],
            }
            if "ticket_size" in c["engineer"]
            else None
        ),
    ),
    "group_rate": (
        "Drop the all-member group rate, keep WCG. Adult men's fates may be noise the "
        "WCG rate already filtered out.",
        lambda c: _without(c, GROUP) if set(GROUP) & set(c["features"]) else None,
    ),
    "wcg_rate": (
        "Drop the WCG rate, keep the all-member rate. The arm that tells us whether WCG "
        "is the lever or merely one encoding of it.",
        lambda c: _without(c, WCG) if set(WCG) & set(c["features"]) else None,
    ),
    "max_features": (
        "Default max_features (sqrt) instead of 0.5. exp-007's +0.0100 for 0.5 on the "
        "baseline features was unconfirmed; this asks whether it survives here.",
        lambda c: (
            {
                **c,
                "params": {k: v for k, v in c["params"].items() if k != "clf__max_features"},
            }
            if "clf__max_features" in c["params"]
            else None
        ),
    ),
}


def decompose_arms(
    champion_config: dict[str, Any], previous_config: dict[str, Any] | None = None
) -> dict[str, Arm]:
    """Take the champion apart: one arm per change it carries, each undone alone.

    Also re-runs, unchanged, the configs a decomposition needs to be read
    against: the previous champion and exp-007's single-lever ``rf_wcg``. Their
    configs are identical to the ones exp-007 logged, which is what lets
    :func:`replication` compare the two experiments' deltas directly.
    """

    def arm(name: str, hypothesis: str, config: dict[str, Any]) -> Arm:
        return Arm(
            name,
            hypothesis,
            config["model"],
            tuple(config["features"]),
            dict(config.get("params", {})),
            tuple(config.get("engineer", ())),
        )

    listed = [arm("champion", "The reigning champion, re-scored as reference.", champion_config)]
    if previous_config is not None:
        listed.append(
            arm(
                "previous_champion",
                "The champion this one replaced. Replicates the promotion delta on rows the "
                "promotion never saw: the config x split interaction, measured.",
                previous_config,
            )
        )
        earlier = arms_for(previous_config)
        if "rf_wcg" in earlier:
            wcg = earlier["rf_wcg"].config(earlier)
            listed.append(
                arm(
                    "rf_wcg",
                    "exp-007's single-lever arm, unchanged: previous champion + WCG rate. "
                    "Replicated so its +0.0182 can be checked on new rows.",
                    wcg,
                )
            )
            listed.append(
                arm(
                    "rf_wcg_maxfeat",
                    "rf_wcg plus max_features 0.5: the guess that these two changes carry "
                    "all of the champion's gain and the other three are noise.",
                    {**wcg, "params": {**wcg["params"], "clf__max_features": 0.5}},
                )
            )
    for component, (hypothesis, undo) in COMPONENTS.items():
        undone = undo(champion_config)
        if undone is not None:
            listed.append(arm(f"without_{component}", hypothesis, undone))
    return {a.name: a for a in listed}


#: Arm sets an experiment can run. ``bakeoff`` is exp-007's breadth search;
#: ``decompose`` takes the reigning champion apart.
ARM_SETS: tuple[str, ...] = ("bakeoff", "decompose")


def registry_for(arm_set: str, champion_path: Path | None = None) -> dict[str, Arm]:
    """The arms for ``arm_set``, built against the reigning champion."""
    from . import champion as champion_module

    reigning = champion_module.ensure(champion_path)
    if arm_set == "bakeoff":
        return arms_for(reigning.config)
    if arm_set == "decompose":
        succession = champion_module.history(champion_path)
        previous = succession[-2].config if len(succession) > 1 else None
        return decompose_arms(reigning.config, previous)
    raise ValueError(f"unknown arm set {arm_set!r}; expected one of {ARM_SETS}")


def _comparable(config: dict[str, Any]) -> str:
    """A config's identity for replication: everything but its display name."""
    import json

    return json.dumps({k: v for k, v in config.items() if k != "name"}, sort_keys=True)


def replication(experiment_id: str, *, log_path: Path | None = None) -> list[dict[str, Any]]:
    """Re-measure earlier experiments' deltas on this experiment's rows.

    For each earlier experiment whose *reference* config was re-run here, every
    other config run in both experiments gets two numbers: its delta over that
    reference there, and the same paired delta here. Both are differences of
    two arms on one experiment's identical folds, so the comparison carries no
    split drift — what remains is the config x split interaction plus noise.
    That is exactly what a cross-experiment claim has to survive.
    """
    from . import results_log

    frame = results_log.read_log(log_path)
    rows = frame[frame["step"] == "bakeoff"].dropna(subset=["cv_score"])
    by_experiment: dict[str, dict[str, tuple[str, float]]] = {}
    for _, row in rows.iterrows():
        params = row["params"]
        if not isinstance(params, dict) or "config" not in params:
            continue
        key = _comparable(params["config"])
        by_experiment.setdefault(str(row["experiment_id"]), {})[key] = (
            str(params["arm"]),
            float(params["delta_vs_champion"]),
        )
    here = by_experiment.get(experiment_id, {})
    found: list[dict[str, Any]] = []
    for earlier_id, there in by_experiment.items():
        if earlier_id == experiment_id:
            continue
        reference = next((k for k, (name, _) in there.items() if name == "champion"), None)
        if reference is None or reference not in here:
            continue
        for key, (name_there, delta_there) in there.items():
            if key == reference or key not in here:
                continue
            name_here, delta_here = here[key]
            found.append(
                {
                    "earlier": earlier_id,
                    "reference": here[reference][0],
                    "arm": name_here,
                    "arm_there": name_there,
                    "delta_there": round(delta_there, 6),
                    "delta_here": round(delta_here - here[reference][1], 6),
                    "change": round(delta_here - here[reference][1] - delta_there, 6),
                }
            )
    return found


def build(config: dict[str, Any], *, seed: int):
    """An unfitted estimator for a logged arm config. Takes the whole frame."""
    from sklearn.ensemble import StackingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from . import tune

    model = config["model"]
    if model in ("soft_vote", "stack"):
        members = [(m["name"], build(m, seed=seed)) for m in config["members"]]
        if model == "soft_vote":
            return VotingClassifier(members, voting="soft")
        return StackingClassifier(members, final_estimator=LogisticRegression(max_iter=2000), cv=5)

    head = [
        ("engineer", Engineer(tuple(config.get("engineer", ())))),
        ("select", Select(tuple(config["features"]))),
    ]
    if model.startswith("rule_"):
        return Pipeline(head + [("clf", RuleClassifier(model.removeprefix("rule_")))])

    base = tune.build_pipeline(model, seed=seed)
    pipeline = Pipeline(head + base.steps)
    params = dict(config.get("params", {}))
    # One core per estimator; the folds run in parallel instead.
    if model in ("random_forest", "extra_trees", "knn"):
        params["clf__n_jobs"] = 1
    if params:
        pipeline.set_params(**params)
    return pipeline


def design(frame):
    """What an arm's estimator is fitted on: every column but the label."""
    return frame.drop(columns=["Survived"])


def fold_scores(
    frame, config: dict[str, Any], *, seed: int, folds: int = CV_FOLDS, repeats: int = CV_REPEATS
) -> np.ndarray:
    """Accuracy on each fold, on the same splitter :func:`compare.fold_scores` uses.

    Identical ``seed`` gives identical folds across arms — and across this module
    and :mod:`titanic_agent.compare` — which is what makes every delta paired.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    return cross_val_score(
        build(config, seed=seed),
        design(frame),
        frame["Survived"],
        cv=splitter,
        scoring="accuracy",
        n_jobs=-1,
        error_score="raise",
    )


def best_of_n_null(n: int, draws: int = 20000) -> float:
    """Expected maximum of ``n`` independent standard normals.

    The delta, in SE units, that the best of ``n`` arms shows when *none* of
    them is better than the champion. Independence overstates it slightly —
    arms sharing a model are correlated — so treat it as a ceiling on the
    noise, not a precise threshold.
    """
    rng = np.random.default_rng(0)
    return float(rng.standard_normal((draws, max(n, 1))).max(axis=1).mean())


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


@dataclass
class ArmResult:
    arm: Arm
    run_id: str = ""
    scores: Any = None
    paired: compare_module.PairedResult | None = None
    seconds: float = 0.0
    error: str = ""


@dataclass
class Bakeoff:
    experiment_id: str
    champion_name: str
    results: list[ArmResult] = field(default_factory=list)
    winner: str = ""
    confirm: compare_module.PairedResult | None = None
    confirm_run_id: str = ""
    seconds: float = 0.0
    n_rows: int = 0
    arm_set: str = "bakeoff"
    replicated: list[dict[str, Any]] = field(default_factory=list)


def _require_experiment():
    from . import experiments

    experiment = experiments.current_experiment()
    if experiment is None:
        raise RuntimeError(
            'no experiment is open. Run `make experiment-open NAME="..."` first — '
            "a bake-off needs a validation split none of it has seen."
        )
    return experiment


def run(
    *,
    arm_set: str = "bakeoff",
    only: list[str] | None = None,
    folds: int = CV_FOLDS,
    repeats: int = CV_REPEATS,
    confirm_seeds: int = CONFIRM_SEEDS,
    frozen_dir: Path = paths.FROZEN_DIR,
    log_path: Path = paths.RESULTS_LOG,
    champion_path: Path | None = None,
) -> Bakeoff:
    """Score every arm against the champion on identical folds; confirm the winner.

    Every arm is one ledger row, failures included, carrying
    ``delta_vs_champion`` and its sigma. The champion's own row carries delta 0
    by construction and is what a later experiment's deltas are relative to.
    """
    from . import champion as champion_module
    from . import experiments, results_log, run_log, tune

    experiment = _require_experiment()
    seed = experiment.seed
    reigning = champion_module.ensure(champion_path)
    registry = registry_for(arm_set, champion_path)
    names = list(registry) if not only else ["champion"] + [n for n in only if n != "champion"]
    frame = tune.load_training_frame(frozen_dir)
    cv_label = f"{folds}x{repeats} repeated stratified"
    challengers = len(names) - 1

    run_log.record(
        "decision",
        step="bakeoff",
        detail=(
            f"{experiment.experiment_id}: {challengers} {arm_set} arms vs champion "
            f"{reigning.name}, {cv_label} CV, paired. Declared before scoring: the "
            f"winner is the arm with the largest delta_vs_champion; it replaces the "
            f"champion only if (1) its selection delta >= {DELTA}, (2) its delta on "
            f"{confirm_seeds} fresh fold seeds >= {DELTA}, and (3) nested CV of the "
            f"whole pick-the-best procedure beats the champion (A - B > 0). The "
            f"validation read scores the winner, whatever it is."
        ),
    )

    started = time.perf_counter()
    bake = Bakeoff(experiment.experiment_id, reigning.name, n_rows=len(frame), arm_set=arm_set)
    champion_scores = None
    for number, name in enumerate(names):
        arm = registry[name]
        config = arm.config(registry)
        result = ArmResult(arm)
        tick = time.perf_counter()
        try:
            result.scores = fold_scores(frame, config, seed=seed, folds=folds, repeats=repeats)
        except Exception as exc:  # a failed arm is a result too
            result.error = f"{type(exc).__name__}: {exc}"
        result.seconds = time.perf_counter() - tick

        extra: dict[str, Any] = {
            "arm": name,
            "config": config,
            "features": config["features"],
            "cv": cv_label,
            "champion": reigning.name,
            "arm_set": arm_set,
            "test_number": number,
        }
        if result.error:
            result.run_id = results_log.append_run(
                step="bakeoff",
                model=arm.model,
                params=extra,
                cv_score=None,
                cv_folds=folds * repeats,
                seed=seed,
                holdout_read=False,  # attested: train rows of this experiment only
                notes=f"bakeoff #{number} {name} FAILED: {result.error}",
                path=log_path,
            )
            run_log.record("step_failed", step="bakeoff", detail=result.error, run_id=result.run_id)
            bake.results.append(result)
            continue

        if champion_scores is None:
            champion_scores = result.scores
        result.paired = compare_module.compare_scores(
            result.scores,
            champion_scores,
            label=f"{name} vs champion",
            n_rows=len(frame),
            seed=seed,
            repeats=repeats,
        )
        extra.update(
            {
                "delta_vs_champion": round(result.paired.delta, 6),
                "paired_se": round(result.paired.naive_se, 6),
                "sigma": round(result.paired.sigma, 3),
                "passengers": round(result.paired.passengers, 2),
            }
        )
        result.run_id = results_log.append_run(
            step="bakeoff",
            model=arm.model,
            params=extra,
            cv_score=float(result.scores.mean()),
            cv_std=float(result.scores.std()),
            cv_folds=folds * repeats,
            seed=seed,
            holdout_read=False,  # attested: train rows of this experiment only
            notes=f"bakeoff #{number} {name}: {result.paired.describe()}. {arm.hypothesis}",
            path=log_path,
        )
        run_log.record(
            "step_finished",
            step="bakeoff",
            detail=f"#{number} {name}: {result.paired.describe()} ({result.seconds:.1f}s)",
            run_id=result.run_id,
        )
        bake.results.append(result)

    bake.replicated = replication(experiment.experiment_id, log_path=log_path)
    if bake.replicated:
        results_log.append_run(
            step="bakeoff",
            model="replication",
            params={"replication": bake.replicated, "features": []},
            # Differences of differences, not an accuracy.
            cv_score=None,
            cv_folds=folds * repeats,
            seed=seed,
            holdout_read=False,
            notes="bakeoff replication: earlier experiments' paired deltas re-measured here; "
            + "; ".join(
                f"{r['arm']} vs {r['reference']}: {r['delta_there']:+.4f} in "
                f"{r['earlier']} -> {r['delta_here']:+.4f} here"
                for r in bake.replicated
            ),
            path=log_path,
        )

    scored = [r for r in bake.results if r.paired is not None and r.arm.name != "champion"]
    if scored:
        top = max(scored, key=lambda r: r.paired.delta)
        bake.winner = top.arm.name
        bake.confirm = confirm(
            frame,
            top.arm.config(registry),
            registry["champion"].config(registry),
            seed=seed,
            seeds=confirm_seeds,
            folds=folds,
            repeats=repeats,
            label=f"{top.arm.name} vs champion, {confirm_seeds} fresh fold seeds",
        )
        bake.confirm_run_id = results_log.append_run(
            step="bakeoff",
            model=top.arm.model,
            params={
                "arm": top.arm.name,
                "features": top.arm.config(registry)["features"],
                "confirm": bake.confirm.as_params(),
                "selection_run_id": top.run_id,
                "selected_from": challengers,
            },
            # A comparison, not an accuracy: unscored so no "best cv" picks it up.
            cv_score=None,
            cv_folds=folds * repeats * confirm_seeds,
            seed=seed,
            holdout_read=False,
            notes=(
                f"bakeoff confirmation of winner {top.arm.name} (selected from "
                f"{challengers}) on fresh fold seeds: {bake.confirm.describe()}"
            ),
            path=log_path,
        )

    bake.seconds = time.perf_counter() - started
    experiments.record_spend("wall_clock_s", round(bake.seconds, 1))
    run_log.record(
        "decision",
        step="bakeoff",
        detail=(
            f"winner {bake.winner or 'none'} of {challengers}; "
            f"{bake.confirm.describe() if bake.confirm else 'no confirmation'} "
            f"({bake.seconds:.0f}s)"
        ),
        run_id=bake.confirm_run_id,
    )
    return bake


def confirm(
    frame,
    config_a: dict[str, Any],
    config_b: dict[str, Any],
    *,
    seed: int,
    seeds: int = CONFIRM_SEEDS,
    folds: int = CV_FOLDS,
    repeats: int = CV_REPEATS,
    label: str = "",
) -> compare_module.PairedResult:
    """``a`` vs ``b`` on fold seeds the selection never used; seed-averaged.

    Offsets start at 1: offset 0 is the selection draw, and confirming on it
    would re-measure the luck being checked for.
    """
    deltas, se, a_means, b_means = [], [], [], []
    for offset in range(1, seeds + 1):
        fold_seed = seed + 1000 * offset
        a = fold_scores(frame, config_a, seed=fold_seed, folds=folds, repeats=repeats)
        b = fold_scores(frame, config_b, seed=fold_seed, folds=folds, repeats=repeats)
        paired = compare_module.compare_scores(
            a, b, label=label, n_rows=len(frame), seed=fold_seed, repeats=repeats
        )
        deltas.append(paired.delta)
        se.append(paired.naive_se)
        a_means.append(paired.score_a)
        b_means.append(paired.score_b)
    deltas_arr = np.asarray(deltas)
    return compare_module.PairedResult(
        label=label,
        delta=float(deltas_arr.mean()),
        # The SE of the seed-averaged delta. Seeds share rows, so averaging does
        # not shrink it by sqrt(seeds); the per-seed SE is the honest floor.
        naive_se=float(np.mean(se)),
        score_a=float(np.mean(a_means)),
        score_b=float(np.mean(b_means)),
        n_folds=folds * repeats,
        n_rows=len(frame),
        repeats=repeats,
        seed=seed,
        fold_diffs=tuple(float(d) for d in deltas_arr),
        seed_sd=float(deltas_arr.std(ddof=1)) if len(deltas_arr) > 1 else None,
    )


# --------------------------------------------------------------------------- #
# nested measurement and the decision
# --------------------------------------------------------------------------- #


@dataclass
class NestedBakeoff:
    experiment_id: str
    picks: list[str] = field(default_factory=list)
    a_scores: list[float] = field(default_factory=list)
    b_scores: list[float] = field(default_factory=list)
    test_rows: list[int] = field(default_factory=list)
    seconds: float = 0.0
    run_id: str = ""

    @property
    def diffs(self) -> np.ndarray:
        return np.asarray(self.a_scores) - np.asarray(self.b_scores)

    @property
    def delta(self) -> float:
        return float(self.diffs.mean()) if len(self.diffs) else 0.0

    @property
    def se(self) -> float:
        d = self.diffs
        return float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0


def nested(
    *,
    arm_set: str = "bakeoff",
    only: list[str] | None = None,
    outer_folds: int = OUTER_FOLDS,
    folds: int = CV_FOLDS,
    repeats: int = CV_REPEATS,
    frozen_dir: Path = paths.FROZEN_DIR,
    log_path: Path = paths.RESULTS_LOG,
    champion_path: Path | None = None,
) -> NestedBakeoff:
    """What "pick the best arm" is worth once it pays for its own selection.

    Inside each outer fold the whole bake-off runs on that fold's training rows
    — same arms, same inner paired CV — and the arm it picks is scored on the
    outer test rows beside the champion. ``A - B`` is the procedure's real gain.
    The inner comparator is not thinned: a cheaper inner loop would measure a
    cheaper procedure than the one that ran.
    """
    from sklearn.model_selection import StratifiedKFold

    from . import experiments, results_log, run_log, tune

    experiment = _require_experiment()
    seed = experiment.seed
    registry = registry_for(arm_set, champion_path)
    names = list(registry) if not only else ["champion"] + [n for n in only if n != "champion"]
    configs = {name: registry[name].config(registry) for name in names}
    frame = tune.load_training_frame(frozen_dir)

    started = time.perf_counter()
    result = NestedBakeoff(experiment.experiment_id)
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    for train_index, test_index in outer.split(frame, frame["Survived"]):
        inner, held = frame.iloc[train_index], frame.iloc[test_index]
        means: dict[str, float] = {}
        for name, config in configs.items():
            try:
                means[name] = float(
                    fold_scores(inner, config, seed=seed, folds=folds, repeats=repeats).mean()
                )
            except Exception:  # an arm that fails inside a fold cannot be picked
                continue
        # Ties go to the champion: a challenger must be strictly better to be picked.
        pick = max(means, key=lambda n: (means[n], n == "champion"))
        result.picks.append(pick)
        for name, sink in ((pick, result.a_scores), ("champion", result.b_scores)):
            estimator = build(configs[name], seed=seed)
            estimator.fit(design(inner), inner["Survived"])
            sink.append(float(estimator.score(design(held), held["Survived"])))
        result.test_rows.append(len(held))

    result.seconds = time.perf_counter() - started
    experiments.record_spend("wall_clock_s", round(result.seconds, 1))
    sigma = result.delta / result.se if result.se else 0.0
    result.run_id = results_log.append_run(
        step="bakeoff",
        model="nested(pick best arm)",
        params={
            "nested": {
                "outer_folds": outer_folds,
                "inner": f"{folds}x{repeats} repeated stratified",
                "picks": result.picks,
                "a_search": [round(s, 6) for s in result.a_scores],
                "b_champion": [round(s, 6) for s in result.b_scores],
                "delta": round(result.delta, 6),
                "se": round(result.se, 6),
                "sigma": round(sigma, 3),
            },
            "selected_from": len(names) - 1,
        },
        # A comparison of procedures, not an accuracy to rank.
        cv_score=None,
        cv_folds=outer_folds,
        seed=seed,
        holdout_read=False,
        notes=(
            f"bakeoff nested: pick-best-of-{len(names) - 1} "
            f"{np.mean(result.a_scores):.4f} vs champion {np.mean(result.b_scores):.4f}, "
            f"A-B {result.delta:+.4f} +/- {result.se:.4f} ({sigma:+.2f} sigma); "
            f"picks {result.picks}; {result.seconds:.0f}s"
        ),
        path=log_path,
    )
    run_log.record(
        "step_finished",
        step="bakeoff",
        detail=(f"nested: A-B {result.delta:+.4f} +/- {result.se:.4f}, picks {result.picks}"),
        run_id=result.run_id,
    )
    return result


@dataclass(frozen=True)
class Decision:
    promote: bool
    reasons: list[str]


def decide(bake: Bakeoff, nested_result: NestedBakeoff | None) -> Decision:
    """Apply the rule declared in :func:`run`. All three must hold."""
    winner = next((r for r in bake.results if r.arm.name == bake.winner), None)
    checks = []
    if winner is None or winner.paired is None:
        return Decision(False, ["no scored challenger"])
    checks.append(
        (winner.paired.delta >= DELTA, f"selection delta {winner.paired.describe()} vs bar {DELTA}")
    )
    checks.append(
        (
            bake.confirm is not None and bake.confirm.delta >= DELTA,
            f"confirmation {bake.confirm.describe() if bake.confirm else 'missing'} vs bar {DELTA}",
        )
    )
    checks.append(
        (
            nested_result is not None and nested_result.delta > 0,
            "nested A-B "
            + (
                f"{nested_result.delta:+.4f} +/- {nested_result.se:.4f} vs 0"
                if nested_result
                else "not measured"
            ),
        )
    )
    reasons = [("PASS " if ok else "FAIL ") + text for ok, text in checks]
    return Decision(all(ok for ok, _ in checks), reasons)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def render_markdown(
    bake: Bakeoff, nested_result: NestedBakeoff | None = None, decision: Decision | None = None
) -> str:
    challengers = [r for r in bake.results if r.arm.name != "champion"]
    null = best_of_n_null(len(challengers))
    champion_row = next((r for r in bake.results if r.arm.name == "champion"), None)
    lines = [
        f"# 05 — bake-off ({bake.experiment_id}, arm set `{bake.arm_set}`)",
        "",
        f"{len(challengers)} arms, each paired against the champion **{bake.champion_name}** "
        f"on identical {CV_FOLDS}×{CV_REPEATS} repeated stratified folds of "
        f"{bake.n_rows} training rows. {bake.seconds:.0f}s.",
        "",
        f"Champion cv on these folds: **{champion_row.scores.mean():.4f}**"
        if champion_row is not None and champion_row.scores is not None
        else "Champion failed to score.",
        "",
        f"**Read this first.** The best of {len(challengers)} arms with no real effect "
        f"shows about **+{null:.1f} SE** from noise alone (independent-normal bound). "
        f"A winner at or below that sigma is indistinguishable from the luckiest null. "
        f"One passenger is {1 / max(bake.n_rows, 1):.4f}; the bar {DELTA} is "
        f"{DELTA * bake.n_rows:.1f} passengers.",
        "",
        "| arm | model | cv | Δ vs champion | SE | σ | passengers | s | hypothesis |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    ordered = sorted(
        bake.results,
        key=lambda r: -1e9 if r.paired is None else -r.paired.delta,
    )
    for r in ordered:
        if r.paired is None:
            lines.append(
                f"| `{r.arm.name}` | {r.arm.model} | FAILED | | | | | {r.seconds:.0f} | {r.error} |"
            )
            continue
        p = r.paired
        lines.append(
            f"| `{r.arm.name}` | {r.arm.model} | {p.score_a:.4f} | {p.delta:+.4f} | "
            f"{p.naive_se:.4f} | {p.sigma:+.2f} | {p.passengers:+.1f} | {r.seconds:.0f} | "
            f"{r.arm.hypothesis} |"
        )
    if bake.replicated:
        lines += [
            "",
            "## Replication: earlier deltas, re-measured on these rows",
            "",
            "Each row is one paired difference measured twice, in two experiments, on "
            "different training rows. Split drift cancels inside each; what is left is "
            "noise (paired SE ~0.003-0.008) plus the config × split interaction.",
            "",
            "| arm | vs | earlier | Δ there | Δ here | change |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
        for r in bake.replicated:
            lines.append(
                f"| `{r['arm']}` | `{r['reference']}` | {r['earlier']} | "
                f"{r['delta_there']:+.4f} | {r['delta_here']:+.4f} | {r['change']:+.4f} |"
            )
    lines += ["", "## Winner and confirmation", ""]
    if bake.winner and bake.confirm is not None:
        lines += [
            f"- Winner: **`{bake.winner}`**, selected from {len(challengers)}.",
            f"- Fresh fold seeds: {bake.confirm.describe()}.",
            "  These re-draw the folds but not the rows, so they remove fold luck, not row luck.",
            "",
        ]
    if nested_result is not None:
        lines += [
            "## Nested: is picking the best arm worth anything?",
            "",
            "| outer fold | test rows | picked | A (picked) | B (champion) | A − B |",
            "| ---: | ---: | --- | ---: | ---: | ---: |",
        ]
        for i, (pick, a, b, n) in enumerate(
            zip(
                nested_result.picks,
                nested_result.a_scores,
                nested_result.b_scores,
                nested_result.test_rows,
                strict=True,
            ),
            start=1,
        ):
            lines.append(f"| {i} | {n} | `{pick}` | {a:.4f} | {b:.4f} | {a - b:+.4f} |")
        sigma = nested_result.delta / nested_result.se if nested_result.se else 0.0
        lines += [
            "",
            f"Procedure vs champion: **{nested_result.delta:+.4f} ± {nested_result.se:.4f}** "
            f"({sigma:+.2f}σ), {nested_result.seconds:.0f}s. Outer test folds hold ~"
            f"{int(np.mean(nested_result.test_rows))} rows, so this is coarse; its job is "
            "the sign, net of selection.",
            "",
        ]
    if decision is not None:
        lines += [
            "## Decision (rule declared before any arm ran)",
            "",
            *[f"- {reason}" for reason in decision.reasons],
            "",
            f"**{'PROMOTE' if decision.promote else 'CHAMPION HOLDS'}.**",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """``make bakeoff`` and ``make bakeoff-decide``."""
    import argparse
    import pickle

    from . import run_log, runs

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", nargs="?", default="run", choices=("run", "decide", "list"))
    parser.add_argument("--only", default="", help="comma-separated arm names")
    parser.add_argument("--set", dest="arm_set", default="bakeoff", choices=ARM_SETS)
    args = parser.parse_args(argv)
    only = [n for n in args.only.split(",") if n] or None

    if args.command == "list":
        from . import champion as champion_module

        # Listing must not crown anything, so with no champion yet it shows the
        # arms the first bake-off would build rather than calling ensure().
        registry = (
            arms_for(champion_module.INITIAL)
            if champion_module.current() is None
            else registry_for(args.arm_set)
        )
        for name, arm in registry.items():
            print(f"  {name:<18} {arm.model:<24} {arm.hypothesis}")
        return 0

    # Resolved once: with no pass id exported, every call to run_dir() mints a
    # fresh directory, and the report would land beside nothing it describes.
    directory = runs.run_dir(run_log.current_pass_id())
    try:
        if args.command == "run":
            bake = run(arm_set=args.arm_set, only=only)
            # Hands the in-memory result to `decide`. A cache, not evidence:
            # every number in it is also in the ledger.
            (directory / "bakeoff.pickle").write_bytes(pickle.dumps(bake))
            report = render_markdown(bake)
        else:
            bake = _latest_state()
            if bake is None:
                print("bakeoff decide: no bake-off for the open experiment; run `make bakeoff`")
                return 1
            directory = runs.run_dir(_latest_state_dir())
            nested_result = nested(arm_set=bake.arm_set, only=only)
            decision = decide(bake, nested_result)
            report = render_markdown(bake, nested_result, decision)
            _apply(bake, nested_result, decision)
    except (RuntimeError, ValueError) as exc:
        print(f"bakeoff failed: {exc}")
        return 1

    written = directory / "05_bakeoff.md"
    written.write_text(report, encoding="utf-8")
    runs.publish(written)
    print(report)
    return 0


def _state_files():
    """Bake-off caches for the open experiment, newest last."""
    import pickle

    from . import experiments

    experiment = experiments.current_experiment()
    if experiment is None:
        return []
    found = []
    for path in sorted(paths.RUNS_DIR.glob("*/bakeoff.pickle"), key=lambda p: p.stat().st_mtime):
        bake = pickle.loads(path.read_bytes())
        if bake.experiment_id == experiment.experiment_id:
            found.append((path, bake))
    return found


def _latest_state() -> Bakeoff | None:
    found = _state_files()
    return found[-1][1] if found else None


def _latest_state_dir() -> str:
    return _state_files()[-1][0].parent.name


def _apply(bake: Bakeoff, nested_result: NestedBakeoff, decision: Decision) -> None:
    """Journal the decision; crown the winner if the rule says so."""
    from . import champion as champion_module
    from . import run_log

    run_log.record(
        "decision",
        step="bakeoff",
        detail=f"{'PROMOTE' if decision.promote else 'HOLD'} {bake.winner}: "
        + "; ".join(decision.reasons),
        run_id=nested_result.run_id,
    )
    if not decision.promote:
        return
    winner = next(r for r in bake.results if r.arm.name == bake.winner)
    registry = registry_for(bake.arm_set)
    champion_module.crown(
        winner.arm.config(registry),
        reason=f"won {bake.experiment_id} bake-off: " + "; ".join(decision.reasons),
        evidence=[winner.run_id, bake.confirm_run_id, nested_result.run_id],
        experiment_id=bake.experiment_id,
        # The confirmation delta, not the selection one: the selection delta is
        # the best of N and carries the winner's curse (exp-007: +0.0264 on the
        # selection folds, +0.0163 on fresh ones). exp-007's own entry predates
        # this line and recorded the biased number; see its findings.
        delta_vs_previous=bake.confirm.delta if bake.confirm else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
