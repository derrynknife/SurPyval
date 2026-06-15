"""
Tests for the ``MixtureModel`` fitter.
"""

import numpy as np
import pytest

import surpyval as sp


def _fitted_model(m=2, seed=0):
    np.random.seed(seed)
    x = np.concatenate(
        [sp.Weibull.random(100, 10, 3), sp.Weibull.random(100, 50, 4)]
    )
    mm = sp.MixtureModel(dist=sp.Weibull, m=m)
    mm.fit(x=x)
    return mm


def test_unfitted_attributes_are_none():
    mm = sp.MixtureModel(dist=sp.Weibull, m=2)
    assert mm.params is None
    assert mm.w is None
    assert mm.data is None
    assert repr(mm) == "Unable to fit values"


def test_fit_populates_params_and_weights():
    mm = _fitted_model()
    assert mm.params is not None
    assert mm.params.shape == (2, sp.Weibull.k)
    # Weights are a valid probability vector.
    assert np.isclose(mm.w.sum(), 1.0)
    assert np.all(mm.w >= 0)


def test_distribution_functions_are_consistent():
    mm = _fitted_model()
    grid = np.array([1.0, 5.0, 10.0, 25.0, 50.0])
    # sf and ff are complements.
    assert np.allclose(mm.sf(grid), 1 - mm.ff(grid))
    # ff is a valid CDF: non-decreasing and within [0, 1].
    ff = mm.ff(grid)
    assert np.all(np.diff(ff) >= 0)
    assert np.all(ff >= 0) and np.all(ff <= 1)
    # df is non-negative.
    assert np.all(mm.df(grid) >= 0)


def test_random_returns_requested_size_for_m_gt_2():
    # Regression test for the slice-accumulation bug in random(): for
    # m > 2 the per-component samples must be laid down contiguously
    # without overwriting earlier components.
    mm = _fitted_model(m=3)
    rvs = mm.random(500)
    assert rvs.shape == (500,)
    # All draws should be populated (Weibull draws are strictly positive),
    # i.e. no slot was left as the initial zero from an overwritten slice.
    assert np.all(rvs > 0)


def test_r_cb_is_removed():
    # R_cb was dead code that raised AttributeError on every fitted model;
    # it has been removed rather than silently shipped.
    mm = _fitted_model()
    assert not hasattr(mm, "R_cb")


def test_too_few_data_points_raises():
    mm = sp.MixtureModel(dist=sp.Weibull, m=2)
    with pytest.raises(ValueError):
        mm.fit(x=[1.0, 2.0])
