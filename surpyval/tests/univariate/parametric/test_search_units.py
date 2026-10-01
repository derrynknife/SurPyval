"""Principle 6, units don't matter, for the optimiser's search (#366).

A parameter with one bound is searched as the log of its distance from the
bound below one *unit* and linearly above it. Offset fits took each unit
from the parameter's own starting distance from its bound; every other fit
used a unit of 1, so a scale was searched as a log in data recorded in
millionths and linearly in data recorded in millions: a different search
in every set of units. Every fit now takes its units from its start, so
the search is the same whatever the units, and a fit to ``k x`` reaches the
maximum of the fit to ``x``, moved by the change of units.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.univariate.parametric.parametric_fitter import _search_units

LOCATION = ("Normal", "Gumbel", "GumbelLEV", "Logistic")
FAMILIES = (
    "Weibull",
    "Exponential",
    "Gamma",
    "LogNormal",
    "LogLogistic",
    "ExpoWeibull",
    "Rayleigh",
) + LOCATION
SCALES = (1e-6, 1e6)


def _data(name):
    rng = np.random.default_rng(366)
    n = 40
    c = (rng.uniform(size=n) < 0.25).astype(int)
    if name in LOCATION:
        return rng.normal(50, 8, n), c
    return rng.weibull(1.7, n) * 10 + 0.5, c


def _rtol(name, how):
    # The same search to rounding (1e-16 to 3e-11 measured on these data;
    # 1e-7 to 2e-6 before #366), except where the objective itself is not
    # computed identically in the two sets of units: the LogNormal's
    # unbounded log-location mu moves by log k, which the search's scale
    # (its own magnitude) sees (1e-7), and the Gamma's MPS spacings, from
    # the incomplete gamma function at rounded arguments (2e-7). Both
    # reach the same optimum to 1e-13 of the objective.
    if name == "LogNormal" or (name, how) == ("Gamma", "MPS"):
        return 1e-6
    return 1e-8


def _quantiles(model, k):
    return model.qf(np.array([0.1, 0.5, 0.9])) / k


@pytest.mark.parametrize("how", ["MLE", "MSE", "MPS"])
@pytest.mark.parametrize("name", FAMILIES)
def test_fits_in_any_units_reach_the_same_answer(name, how):
    if (name, how) == ("ExpoWeibull", "MSE"):
        pytest.skip(
            "no finite optimum on these data (alpha -> 0, mu -> inf); the "
            "search stops at its evaluation limit"
        )
    dist = getattr(sp, name)
    x, c = _data(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = dist.fit(x, c=c, how=how)
        for k in SCALES:
            got = dist.fit(x * k, c=c, how=how)
            np.testing.assert_allclose(
                _quantiles(got, k), _quantiles(ref, 1.0), rtol=_rtol(name, how)
            )
            if how == "MLE":
                # The same maximum: the log-likelihood moves by n_obs log k.
                shift = np.sum(c == 0) * np.log(k)
                assert -got.neg_ll() + shift == pytest.approx(
                    -ref.neg_ll(), abs=1e-9
                )
            else:
                # The MSE and MPS objectives do not depend on the units.
                assert got.res.fun == pytest.approx(ref.res.fun, rel=1e-11)


def test_units_are_each_parameters_start_distance_from_its_bound():
    bounds = ((0, None), (None, None), (None, 5.0), (0, 1))
    units = _search_units(np.array([2e-6, 3.0, 4.5, 0.3]), bounds)
    assert units == [2e-6, 1.0, 0.5, 1.0]
    # A start on (or beyond) its bound keeps a unit of 1.
    assert _search_units(np.array([0.0, 1.0]), ((0, None), (0, None))) == [
        1.0,
        1.0,
    ]


def test_a_scale_starts_at_the_switch_in_any_units():
    # The fit's own record of the search: every one-sided parameter starts
    # at a searched value of 0 whatever the data's units.
    x, c = _data("Weibull")
    for k in (1e-6, 1.0, 1e6):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sp.Weibull.fit(x * k, c=c)
        np.testing.assert_allclose(model.fitting_info["init"], 0.0, atol=0)
