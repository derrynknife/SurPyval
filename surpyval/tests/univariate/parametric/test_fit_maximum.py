"""A fit records whether it reached a verified maximum (follow-up to #492).

``Parametric.maximum`` says what a fit's log-likelihood stands on:
``"verified"`` (a zero gradient and a positive-definite Hessian, or an
exact closed form), ``"unverified"`` (the search stopped short; the fit
warned), ``"no finite maximum"`` (the likelihood has none; the fit warned
so), ``"not applicable"`` (not a maximum-likelihood fit) or
``"unknown"`` (restored from a dictionary saved before it existed). It
agrees with the fit's warnings, survives ``to_dict`` / ``from_dict``, and
is what ``fit_best`` reads to set a candidate aside -- not the text of its
warnings.
"""

import json
import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.univariate.parametric.fitters import mle as mle_module
from surpyval.univariate.parametric.parametric import (
    MAXIMUM_STATES,
    Parametric,
)
from surpyval.utils.no_maximum import quiet_maximum_warnings

SEVEN = np.arange(1, 8.0)
WEIBULL_50 = np.random.default_rng(5).weibull(2, 50) * 100
# The 3-parameter Weibull profile likelihood rises all the way to the
# first failure (#487)
OFFSET_X = [55, 60, 70, 80, 95, 120, 140]


def _fit(fit, *args, **kwargs):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        model = fit(*args, **kwargs)
    return model, [str(w.message) for w in rec]


def test_an_ordinary_fit_is_a_verified_maximum():
    model, messages = _fit(sp.Weibull.fit, WEIBULL_50)
    assert model.maximum == "verified"
    assert messages == []


@pytest.mark.parametrize(
    "dist, x",
    [
        (sp.Exponential, WEIBULL_50),  # closed form
        (sp.Normal, WEIBULL_50),  # closed form
        (sp.Uniform, WEIBULL_50),  # the extreme observations
        (sp.Gamma, WEIBULL_50),
        (sp.LogNormal, WEIBULL_50),
    ],
)
def test_exact_and_optimised_maxima_are_verified(dist, x):
    assert _fit(dist.fit, x)[0].maximum == "verified"


def test_the_exact_discrete_fits_are_verified():
    assert sp.Bernoulli.fit([0, 1, 1]).maximum == "verified"
    assert sp.Binomial.fit([2, 3, 1, 4], n_trials=5).maximum == "verified"


def test_a_beta4_with_no_finite_maximum_says_so():
    model, messages = _fit(sp.Beta4.fit, SEVEN)
    assert model.maximum == "no finite maximum"
    assert [m for m in messages if m.startswith("No finite maximum")]


def test_an_offset_run_onto_the_first_failure_has_no_finite_maximum():
    model, messages = _fit(sp.Weibull.fit, OFFSET_X, offset=True)
    assert model.maximum == "no finite maximum"
    assert len(messages) == 1
    assert messages[0].startswith("No finite maximum: the offset gamma")


def test_an_offset_exponential_has_a_genuine_maximum():
    # Its density is finite at the origin: a maximum at gamma = x(1)
    model, messages = _fit(sp.Exponential.fit, OFFSET_X, offset=True)
    assert model.maximum != "no finite maximum"
    assert not [m for m in messages if m.startswith("No finite maximum")]


def test_a_search_that_stops_short_is_unverified():
    # The ExpoWeibull runs towards a limit of its shapes on this sample
    model, messages = _fit(sp.ExpoWeibull.fit, WEIBULL_50)
    assert model.maximum == "unverified"
    assert [m for m in messages if "did not reach a verified" in m]


def test_a_starved_search_is_unverified(monkeypatch):
    # No point the search reaches passes the check: every start fails,
    # and the fit warns and records it.
    monkeypatch.setattr(mle_module, "is_local_minimum", lambda *a, **k: False)
    model, messages = _fit(sp.Weibull.fit, WEIBULL_50)
    assert model.maximum == "unverified"
    assert len(messages) == 1


def test_the_flag_agrees_with_the_warnings_when_they_are_held_back():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with quiet_maximum_warnings():
            beta4 = sp.Beta4.fit(SEVEN)
            expo = sp.ExpoWeibull.fit(WEIBULL_50)
            offset = sp.Weibull.fit(OFFSET_X, offset=True)
    assert rec == []
    assert beta4.maximum == "no finite maximum"
    assert expo.maximum == "unverified"
    assert offset.maximum == "no finite maximum"


@pytest.mark.parametrize("how", ["MPS", "MSE", "MPP", "MOM"])
def test_other_estimators_do_not_maximise_the_likelihood(how):
    model, _ = _fit(sp.Weibull.fit, WEIBULL_50, how=how)
    assert model.maximum == "not applicable"


def test_models_not_fitted_by_maximum_likelihood():
    assert sp.Weibull.from_params([10, 2]).maximum == "not applicable"
    assert sp.Weibull.from_params([10, 2], p=0.9).maximum == "not applicable"
    fitted = sp.Weibull.fit(WEIBULL_50)
    assert fitted.with_params([90, 2]).maximum == "not applicable"
    x = np.array([1.0, 2.0, 3.0, 4.0])
    F = np.array([0.1, 0.3, 0.6, 0.9])
    assert sp.Weibull.fit_from_ecdf(x, F).maximum == "not applicable"


@pytest.mark.parametrize(
    "fit",
    [
        lambda: sp.Weibull.fit(WEIBULL_50),
        lambda: sp.Beta4.fit(SEVEN),
        lambda: sp.ExpoWeibull.fit(WEIBULL_50),
        lambda: sp.Weibull.fit(WEIBULL_50, how="MPS"),
        lambda: sp.Weibull.from_params([10, 2]),
    ],
)
def test_the_flag_round_trips(fit):
    model, _ = _fit(fit)
    stored = json.loads(json.dumps(model.to_dict()))
    assert stored["maximum"] == model.maximum
    assert stored["schema"] == 1  # an older reader ignores it
    assert sp.from_dict(stored).maximum == model.maximum


def test_a_dictionary_saved_before_the_flag_reads_as_unknown():
    old = sp.Weibull.fit(WEIBULL_50).to_dict()
    del old["maximum"]
    assert sp.from_dict(old).maximum == "unknown"
    old = sp.Weibull.fit(WEIBULL_50, how="MPS").to_dict()
    del old["maximum"]
    assert sp.from_dict(old).maximum == "not applicable"


def test_a_bad_stored_flag_is_refused():
    stored = sp.Weibull.fit(WEIBULL_50).to_dict()
    stored["maximum"] = "maybe"
    with pytest.raises(ValueError, match="'maximum'"):
        Parametric.from_dict(stored)


def test_every_state_is_documented():
    doc = Parametric.__doc__
    for state in MAXIMUM_STATES:
        assert f'``"{state}"``' in doc


# ---------------------------------------------------------------------------
# fit_best reads the flag, not the warnings
# ---------------------------------------------------------------------------
def _with_flag(monkeypatch, dist, maximum, message=None):
    """``dist.fit`` returning its model with ``maximum`` set, and giving
    ``message`` as a warning if one is given."""
    fit = dist.fit

    def patched(*args, **kwargs):
        model = fit(*args, **kwargs)
        model.maximum = maximum
        if message is not None:
            warnings.warn(message)
        return model

    monkeypatch.setattr(dist, "fit", patched)


def test_fit_best_sets_aside_a_candidate_by_its_flag(monkeypatch):
    # The Weibull wins on this sample among these three (a Weibull with
    # shape 2); flagged, it is set aside although it warned nothing.
    include = ["Weibull", "Gamma", "LogNormal"]
    assert sp.fit_best(WEIBULL_50, include=include).dist.name == "Weibull"
    _with_flag(monkeypatch, sp.Weibull, "unverified")
    model, messages = _fit(sp.fit_best, WEIBULL_50, include=include)
    assert model.dist.name != "Weibull"
    aside = [m for m in messages if m.startswith("fit_best set aside")]
    assert len(aside) == 1
    assert "Weibull (its fit is not a verified maximum)" in aside[0]


def test_fit_best_reads_no_finite_maximum_from_the_flag(monkeypatch):
    _with_flag(monkeypatch, sp.Weibull, "no finite maximum")
    model, messages = _fit(
        sp.fit_best, WEIBULL_50, include=["Weibull", "Gamma"]
    )
    assert model.dist.name == "Gamma"
    assert [m for m in messages if "Weibull (its likelihood has no" in m]


def test_fit_best_ignores_the_text_of_a_verified_candidate(monkeypatch):
    # A warning that merely reads like a no-maximum one does not set a
    # verified candidate aside; it is passed on as any other warning is.
    text = "No finite maximum: said, but not so"
    _with_flag(monkeypatch, sp.Weibull, "verified", message=text)
    model, messages = _fit(
        sp.fit_best, WEIBULL_50, include=["Weibull", "Gamma", "LogNormal"]
    )
    assert model.dist.name == "Weibull"
    assert text in messages
    assert not [m for m in messages if m.startswith("fit_best set aside")]


def test_fit_best_holds_back_the_candidates_own_warnings():
    model, messages = _fit(
        sp.fit_best, WEIBULL_50, include=["ExpoWeibull", "Weibull", "Beta4"]
    )
    assert model.dist.name == "Weibull"
    assert not [m for m in messages if "did not reach a verified" in m]
    assert not [m for m in messages if m.startswith("No finite maximum")]
    assert len(messages) == 1 and messages[0].startswith("fit_best set")
