"""The starting point of an offset maximum-likelihood fit (#622)."""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.utils.surpyval_data import SurpyvalData


def test_622_offset_interval_fit_in_a_lower_tail_window_is_searchable():
    # A start far below the data, with a tiny shape, puts every interval
    # in the distribution's lower tail. Its stand-in value was taken with
    # ``float`` of the bounds less the offset, which the search traces:
    # TypeError: float() argument must be ... not 'ArrayBox'.
    xl = np.array([7.3, 8.0, 10.2, 11.7, 12.4, 13.1, 13.9])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.LogNormal.fit(
            xl=xl, xr=xl + 0.7, offset=True, init=[-6.4e9, 22.6, 4e-10]
        )
    assert np.isfinite(model.neg_ll())
    assert model.gamma < xl.min()


OFFSET_FAMILIES = [
    sp.Weibull,
    sp.Gamma,
    sp.LogNormal,
    sp.LogLogistic,
    sp.Exponential,
    sp.Rayleigh,
    sp.ExpoWeibull,
]


def _seeds(dist, data):
    # (``_parameter_initialiser`` on code without ``_shifted_initialiser``)
    seed = getattr(dist, "_shifted_initialiser", dist._parameter_initialiser)
    return seed(data)


def _shifted(x, gamma, c=None, n=None):
    x = np.asarray(x, dtype=float) - gamma
    c = np.zeros(len(x), dtype=int) if c is None else np.asarray(c)
    n = np.ones(len(x)) if n is None else np.asarray(n)
    return SurpyvalData(x, c, n, group_and_sort=False)


@pytest.mark.parametrize("dist", OFFSET_FAMILIES, ids=lambda d: d.name)
def test_622_an_offset_start_is_one_distribution_at_its_offset(dist):
    # The start's other parameters are the distribution's own seeds for
    # the data shifted by the start's offset. The Weibull's were its
    # probability plot's, fitted with the plot's own offset (-5.1 for
    # data from 15.8 to 30) and kept when the offset was replaced by one
    # just below the data (15.3); the Exponential's rate was that of the
    # unshifted data.
    rng = np.random.default_rng(9)
    x = 14.3 + sp.Weibull.random(30, 10, 2.7, random_state=rng)
    data = SurpyvalData(x)
    init = dist._initial_guess(data, True, False, False, "Nelson-Aalen")
    gamma = init[0]
    assert gamma < x.min()
    expected = _seeds(dist, _shifted(data.x, gamma))
    np.testing.assert_allclose(init[1:], expected, rtol=1e-12)


@pytest.mark.parametrize(
    "dist", [sp.Weibull, sp.LogNormal, sp.Gamma], ids=lambda d: d.name
)
def test_622_an_interval_offset_start_is_shifted_by_its_own_offset(dist):
    # The seeds were taken from the imputed midpoints shifted by an
    # offset below *them*, and the offset then replaced by one below the
    # intervals' left ends.
    rng = np.random.default_rng(4)
    t = 20.0 + sp.Weibull.random(25, 10, 2.0, random_state=rng)
    xl = np.floor(t / 3.0) * 3.0
    data = SurpyvalData(xl=xl, xr=xl + 3.0)
    init = dist._initial_guess(data, True, False, False, "Nelson-Aalen")
    gamma = init[0]
    assert gamma < xl.min()
    mid = np.asarray(data.x, dtype=float).mean(axis=1)
    expected = _seeds(dist, _shifted(mid, gamma, n=np.asarray(data.n)))
    np.testing.assert_allclose(init[1:], expected, rtol=1e-12)


def test_622_a_left_skewed_weibull_offset_fit_is_not_a_garbage_point():
    # The issue's seed 14: 15 points between 22.1 and 30.5 with a long
    # left tail, on which the offset Weibull runs to its limit, the
    # Gumbel (log-likelihood -30.169). From the inconsistent start
    # (scale 1.7e5, shape 8.5e4 at an offset of 21.5; -1.3e7) BFGS
    # stopped at once at -4.8e6, and that point was returned as "No
    # finite maximum".
    rng = np.random.default_rng(14)
    n = int(rng.choice([15, 30, 100]))
    gamma = float(rng.uniform(0, 50))
    beta = float(rng.uniform(0.8, 4))
    x = gamma + sp.Weibull.random(n, 10, beta, random_state=rng)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        model = sp.Weibull.fit(x, offset=True)
    gumbel = sp.Gumbel.fit(x)
    assert model.maximum == "no finite maximum"
    assert len(rec) == 1
    assert "surpyval.Gumbel" in str(rec[0].message)
    # on the way to the limit: below the data, and within a hair of it
    assert model.gamma < x.min() - 100
    assert model.neg_ll() == pytest.approx(gumbel.neg_ll(), abs=0.01)


@pytest.mark.parametrize("dist", OFFSET_FAMILIES, ids=lambda d: d.name)
def test_622_an_offset_fit_refuses_a_failure_at_infinity(dist):
    # As the fit without an offset does. The offset fits took it in: the
    # Exponential returned a rate, the Weibull warned "MLE Failed" with an
    # infinite likelihood, the Gamma leaked RuntimeWarnings, and the
    # LogNormal raised with the Normal's message (from its start).
    from surpyval.univariate.parametric._fit_inputs import (
        OutsideSupportError,
    )

    x = [1.0, 2.0, 3.0, np.inf, 5.0]
    with pytest.raises(OutsideSupportError, match=f"offset {dist.name} "):
        dist.fit(x, offset=True)
