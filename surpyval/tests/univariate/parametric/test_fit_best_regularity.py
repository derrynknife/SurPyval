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


def test_an_unverified_fit_is_set_aside():
    # The ExpoWeibull runs towards a limit of its shapes on this sample
    # (beta = 468, mu = 0.0027) and warned that its search did not reach
    # a verified maximum; its AIC (503.5) used to beat every regular fit.
    model, messages = _fit_best(WEIBULL_50, include=["ExpoWeibull", "Weibull"])
    assert model.dist.name == "Weibull"
    assert [m for m in messages if "ExpoWeibull (its fit is not a verif" in m]
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
