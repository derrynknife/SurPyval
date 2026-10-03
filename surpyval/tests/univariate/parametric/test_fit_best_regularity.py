"""fit_best ranks only regular maxima (#492).

AIC, AIC_c and BIC assume a regular maximum of the likelihood. The Uniform
and the Beta4 (support ends among the parameters) are left out of the
default candidates, and any candidate that is not a verified maximum (it
warns "No finite maximum", or that its search did not reach a verified
maximum) is set aside: ranked only when no regular candidate fitted, and
named in one warning.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
import surpyval as surv
from surpyval.tests.conformance.registry import CASE_BY_NAME

SEVEN = np.arange(1, 8.0)
WEIBULL_50 = np.random.default_rng(5).weibull(2, 50) * 100


def _fit_best(*args, **kwargs):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        model = sp.fit_best(*args, **kwargs)
    return model, [str(w.message) for w in rec]


@pytest.mark.parametrize("metric", ["aic", "aic_c", "bic"])
def test_beta4_no_longer_wins_on_seven_points(metric):
    # Before: 'Beta4' under every metric, neg_ll -24.75 at a shape of 0.11
    model, _ = _fit_best(SEVEN, metric=metric)
    assert model.dist.name not in ("Beta4", "Uniform")


def test_uniform_no_longer_wins_on_weibull_data():
    # Before: 'Uniform' on [3.8, 147.7], neg_ll 248.5 against 255.3
    model, _ = _fit_best(WEIBULL_50)
    assert model.dist.name == "Rayleigh"  # a Weibull with shape 2


def test_a_named_non_regular_family_is_set_aside_with_a_warning():
    model, messages = _fit_best(WEIBULL_50, include=["Uniform", "Weibull"])
    assert model.dist.name == "Weibull"
    aside = [m for m in messages if m.startswith("fit_best set aside")]
    assert len(aside) == 1
    assert "Uniform (its support ends are parameters" in aside[0]


def test_a_fit_with_no_maximum_is_set_aside_and_its_warning_replaced():
    model, messages = _fit_best(SEVEN, include=["Beta4", "Weibull"])
    assert model.dist.name == "Weibull"
    assert not [m for m in messages if m.startswith("No finite maximum")]
    assert [m for m in messages if "Beta4 (its likelihood has no" in m]


def test_a_runaway_fit_is_set_aside():
    # The ExpoWeibull runs towards a limit of its shapes on this sample
    # (beta to infinity, mu to 0; the profile log-likelihood rises from
    # -253.47 at beta = 3 to -248.64 at beta = 1000) and its AIC (503.5)
    # used to beat every regular fit. Its search used to end "unverified"
    # after the whole ladder; it now finds the runaway (#584).
    model, messages = _fit_best(WEIBULL_50, include=["ExpoWeibull", "Weibull"])
    assert model.dist.name == "Weibull"
    assert [m for m in messages if "ExpoWeibull (its likelihood has no" in m]
    assert not [m for m in messages if m.startswith("No finite maximum")]


def test_an_unverified_fit_is_set_aside():
    # The Beta4 on its conformance fixture stops short of a verified
    # maximum (its likelihood is unbounded at a support end, #385).
    d = CASE_BY_NAME["Beta4"].data()
    model, messages = _fit_best(d["x"], n=d["n"], include=["Beta4", "Weibull"])
    assert model.dist.name == "Weibull"
    assert [m for m in messages if "Beta4 (its fit is not a verif" in m]
    assert not [m for m in messages if "did not reach a verified" in m]


def test_set_aside_candidates_are_ranked_when_nothing_else_fits():
    model, messages = _fit_best(SEVEN, include=["Uniform"])
    assert model.dist.name == "Uniform"
    assert [m for m in messages if "Chosen: Uniform" in m]


def test_an_ordinary_call_warns_nothing_extra():
    x = sp.Weibull.random(40, 10, 2, random_state=3)
    model, messages = _fit_best(x, include=["Weibull", "Gamma", "LogNormal"])
    assert messages == []
    assert model.dist.name in ("Weibull", "Gamma", "LogNormal")


# ---------------------------------------------------------------------------
# ``fit_best`` checks the distribution names.
# ---------------------------------------------------------------------------


def test_fit_best_checks_distribution_names():
    x = [1.0, 2, 3, 4, 5, 6]
    assert surv.fit_best(x, include=["weibull"]).dist.name == "Weibull"
    assert surv.fit_best(x, include="Weibull").dist.name == "Weibull"
    with pytest.raises(ValueError, match="Unknown distribution"):
        surv.fit_best(x, include=["Weibul"])
    with pytest.raises(ValueError, match="Unknown distribution"):
        surv.fit_best(x, exclude=["Geometric"])
