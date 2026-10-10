"""Audit our search spaces against the estimators we actually installed.

The question this answers: *are we tuning the right things, or leaving most of
each estimator at its defaults and calling it a search?*

It asks the installed package rather than the web, deliberately. The web
documents whatever version is current; we run whatever is in ``.venv``. That gap
is not hypothetical here — ``SVC(probability=True)`` is deprecated in the
scikit-learn 1.9 we have and removed in 1.11, and web docs for a different
version could have confirmed the wrong thing with total confidence. Introspection
is also free, offline, and instant, which is why it should exhaust what it can
answer before any budgeted external reading begins.

Four findings it can produce, in rough order of how much they matter:

``invalid``
    A search-space key that is not a parameter of its pipeline. The search
    silently tunes nothing. This is the failure that hides: a sweep runs, logs
    scores, and the grid never applied.
``deprecated_tuned``
    A parameter we actually put in a search space that the installed version has
    deprecated. A countdown on a search we rely on.
``deprecated_available``
    Deprecated, but we never touch it. Reported, not failed. A deprecation in a
    parameter nobody uses is news, not a defect, and failing on it would bury
    the two findings above in noise.
``untuned``
    A parameter left at its default that plausibly matters. Not a bug — most
    defaults are fine — but it is the list worth arguing with.
``unreachable``
    A parameter that cannot be set through the pipeline at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: Parameters that are plumbing, not modelling choices. Reporting these as
#: "untuned" would bury the real findings in noise.
BORING = frozenset(
    {
        "copy",
        "copy_X",
        "verbose",
        "n_jobs",
        "random_state",
        "warm_start",
        "memory",
        "cache_size",
        "estimator",
        "steps",
        "transformers",
        "remainder",
        "transformer_weights",
        "sparse_threshold",
        "verbose_feature_names_out",
        "force_int_remainder_cols",
        "n_features_to_select",
    }
)

#: numpydoc's deprecation directive, which carries the version it landed in.
DEPRECATION = re.compile(r"deprecated::\s*([\d.]+)", re.IGNORECASE)


@dataclass(frozen=True)
class ModelAudit:
    """What we tune on one model versus what we could."""

    model_key: str = ""
    tuned: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)
    deprecated_tuned: dict[str, str] = field(default_factory=dict)
    deprecated_available: dict[str, str] = field(default_factory=dict)
    untuned: list[str] = field(default_factory=list)
    n_candidates: int = 0

    @property
    def ok(self) -> bool:
        """True when nothing we rely on is broken.

        Parameters left at defaults are a judgement call, and a deprecation in a
        parameter we never set is news rather than a defect. Only a search key
        that tunes nothing, or a deprecation in something we do tune, fails.
        """
        return not self.invalid and not self.deprecated_tuned


def _estimator_params(pipeline) -> dict[str, Any]:
    """Every settable parameter of the classifier half of a pipeline."""
    classifier = pipeline.named_steps["clf"]
    return classifier.get_params(deep=False)


def _param_docs(estimator) -> dict[str, str]:
    """Pull the per-parameter blurbs out of a numpydoc ``Parameters`` section."""
    doc = estimator.__doc__ or ""
    section = re.split(r"\n\s*Parameters\s*\n\s*-+\s*\n", doc, maxsplit=1)
    if len(section) < 2:
        return {}
    body = re.split(r"\n\s*(?:Attributes|Returns|Examples|See Also)\s*\n\s*-+", section[1])[0]

    docs: dict[str, str] = {}
    current: str | None = None
    for line in body.splitlines():
        header = re.match(r"^\s{0,4}(\w+)\s*:\s*(.*)$", line)
        if header:
            current = header.group(1)
            docs[current] = header.group(2)
        elif current:
            docs[current] += " " + line.strip()
    return docs


def audit_model(model_key: str) -> ModelAudit:
    """Compare one model's search space against its installed estimator."""
    from . import tune

    pipeline = tune.build_pipeline(model_key)
    space = tune.search_space(model_key)
    valid = set(pipeline.get_params(deep=True))

    invalid = sorted(key for key in space if key not in valid)
    tuned_leaves = {key.rsplit("__", 1)[-1] for key in space if key not in invalid}

    classifier = pipeline.named_steps["clf"]
    settable = _estimator_params(pipeline)
    docs = _param_docs(type(classifier))

    deprecations: dict[str, str] = {}
    for name, blurb in docs.items():
        match = DEPRECATION.search(blurb or "")
        if name in settable and match:
            deprecations[name] = match.group(1).rstrip(".")
    deprecated_tuned = {n: v for n, v in deprecations.items() if n in tuned_leaves}
    deprecated_available = {n: v for n, v in deprecations.items() if n not in tuned_leaves}
    untuned = sorted(
        name
        for name in settable
        if name not in tuned_leaves and name not in BORING and not name.startswith("_")
    )

    candidates = 1
    for values in space.values():
        candidates *= len(values)

    return ModelAudit(
        model_key=model_key,
        tuned=sorted(space),
        invalid=invalid,
        deprecated_tuned=deprecated_tuned,
        deprecated_available=deprecated_available,
        untuned=untuned,
        n_candidates=candidates if space else 0,
    )


def audit_all(model_keys: tuple[str, ...] | None = None) -> list[ModelAudit]:
    """Audit every model the tuner knows about."""
    from . import tune

    return [audit_model(key) for key in (model_keys or tune.MODEL_KEYS)]


def main(argv: list[str] | None = None) -> int:
    """``make introspect`` — exits non-zero if anything is actively broken."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None, help="audit one model key")
    parser.add_argument(
        "--show-untuned", action="store_true", help="list parameters left at defaults"
    )
    args = parser.parse_args(argv)

    import sklearn

    print(f"scikit-learn {sklearn.__version__} (the installed one, not the documented one)")
    print()

    audits = audit_all((args.model,) if args.model else None)
    broken = [a for a in audits if not a.ok]

    for audit in audits:
        mark = "ok  " if audit.ok else "FAIL"
        print(
            f"  [{mark}] {audit.model_key:<24} {len(audit.tuned)} tuned, "
            f"{audit.n_candidates} candidate(s), {len(audit.untuned)} left at default"
        )
        for key in audit.invalid:
            print(f"         INVALID  {key} is not a parameter — this search tunes nothing")
        for name, version in sorted(audit.deprecated_tuned.items()):
            print(f"         DEPRECATED (we tune this)  {name} — since {version}")
        for name, version in sorted(audit.deprecated_available.items()):
            print(f"         note: {name} deprecated since {version}; we do not set it")
        if args.show_untuned and audit.untuned:
            print(f"         untuned: {', '.join(audit.untuned)}")

    print()
    if broken:
        print(f"{len(broken)} model(s) tune something invalid or deprecated.")
        return 1
    print("No invalid or deprecated search-space keys.")
    print("Parameters left at defaults are a judgement call, not a defect —")
    print("run with --show-untuned to argue with the list.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
