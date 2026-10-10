"""Smoke test: auditing our search spaces against the installed estimators.

The test that earns this module's existence is
:func:`test_it_catches_a_search_key_that_tunes_nothing` — the real bug it was
written for shipped in this repo for a few minutes, when wrapping ``SVC`` in
``CalibratedClassifierCV`` moved its parameters a level deeper and the grid
still said ``clf__C``.

The other half is about *not* crying wolf: a deprecation in a parameter nobody
sets is news, not a defect, and failing on it would bury the findings that
matter.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from titanic_agent import introspect, tune


@pytest.fixture(scope="module")
def audits() -> list[introspect.ModelAudit]:
    return introspect.audit_all()


# --------------------------------------------------------------------------- #
# the failure it exists to catch
# --------------------------------------------------------------------------- #


def test_it_catches_a_search_key_that_tunes_nothing() -> None:
    # Exactly the bug that shipped: SVC inside CalibratedClassifierCV needs
    # clf__estimator__C, not clf__C.
    broken = {"clf__C": [0.5, 1.0], "clf__gamma": ["scale"]}
    with patch.object(
        tune, "search_space", lambda key: broken if key == "svc" else tune.search_space(key)
    ):
        audit = introspect.audit_model("svc")
    assert audit.ok is False
    assert audit.invalid == ["clf__C", "clf__gamma"]


def test_every_current_search_space_is_valid(audits) -> None:
    offenders = {a.model_key: a.invalid for a in audits if a.invalid}
    assert not offenders, f"these searches tune nothing: {offenders}"


def test_nothing_we_tune_is_deprecated(audits) -> None:
    offenders = {a.model_key: a.deprecated_tuned for a in audits if a.deprecated_tuned}
    assert not offenders, f"deprecated parameters in live search spaces: {offenders}"


# --------------------------------------------------------------------------- #
# not crying wolf
# --------------------------------------------------------------------------- #


def test_a_deprecation_we_do_not_touch_is_not_a_failure(audits) -> None:
    # scikit-learn 1.9 deprecates LogisticRegression.penalty, l1_ratio and
    # n_jobs. We set none of them, so this must stay informational -- otherwise
    # the audit is red permanently and stops being read.
    logistic = next(a for a in audits if a.model_key == "logistic")
    assert logistic.deprecated_available
    assert logistic.ok is True


def test_untuned_parameters_are_not_failures(audits) -> None:
    assert any(a.untuned for a in audits)
    assert all(a.ok for a in audits if a.untuned and not a.invalid)


def test_plumbing_is_not_reported_as_untuned(audits) -> None:
    # random_state and n_jobs are not modelling choices; listing them would
    # bury the parameters worth arguing about.
    for audit in audits:
        assert "random_state" not in audit.untuned
        assert "n_jobs" not in audit.untuned


# --------------------------------------------------------------------------- #
# what it reports
# --------------------------------------------------------------------------- #


def test_every_model_is_audited(audits) -> None:
    assert {a.model_key for a in audits} == set(tune.MODEL_KEYS)


def test_candidate_counts_match_the_grid(audits) -> None:
    for audit in audits:
        space = tune.search_space(audit.model_key)
        expected = 1
        for values in space.values():
            expected *= len(values)
        assert audit.n_candidates == (expected if space else 0), audit.model_key


def test_the_baseline_has_nothing_to_tune(audits) -> None:
    dummy = next(a for a in audits if a.model_key == "dummy")
    assert dummy.tuned == []
    assert dummy.ok


def test_param_docs_are_extracted(audits) -> None:
    from sklearn.ensemble import RandomForestClassifier

    docs = introspect._param_docs(RandomForestClassifier)
    assert "n_estimators" in docs
    assert docs["n_estimators"].strip()


def test_param_docs_survive_an_undocumented_estimator() -> None:
    class Bare:
        """No parameters section at all."""

    assert introspect._param_docs(Bare) == {}


def test_auditing_an_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="unknown model_key"):
        introspect.audit_model("not-a-model")


def test_main_reports_the_installed_version(capsys) -> None:
    # The whole premise: the installed version, not the documented one.
    import sklearn

    introspect.main([])
    assert sklearn.__version__ in capsys.readouterr().out


def test_main_exits_zero_when_nothing_is_broken(capsys) -> None:
    assert introspect.main([]) == 0


def test_main_can_show_untuned(capsys) -> None:
    introspect.main(["--model", "random_forest", "--show-untuned"])
    assert "untuned:" in capsys.readouterr().out
