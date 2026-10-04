"""
A missing (NaN) time at prediction gives NaN for that time only (#375).

The non-parametric estimators and non-parametric competing risks looked a
NaN time up with ``searchsorted``, which sorts it past the last step, so
it returned the value at t = inf; parametric ``sf_tvc`` / ``Hf_tvc``
raised an IndexError. Other times must be unaffected.
"""

import warnings

import numpy as np
import pytest

import surpyval as surv
from surpyval.univariate.competing_risks import CompetingRisks

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0])
C = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
E = np.array([1, 2, None, 1, 2, None, 1, 1, 2, None], dtype=object)
T = np.array([np.nan, 5.0, 8.0, 1e9])


def _check(with_nan, without_nan):
    with_nan = np.asarray(with_nan, dtype=float)
    assert np.isnan(with_nan[..., 0]).all()
    np.testing.assert_allclose(with_nan[..., 1:], without_nan)


@pytest.mark.parametrize(
    "fitter",
    [
        surv.KaplanMeier,
        surv.NelsonAalen,
        surv.FlemingHarrington,
        surv.Turnbull,
    ],
)
@pytest.mark.parametrize("method", ["sf", "ff", "Hf", "hf", "df"])
def test_nonparametric_function_at_a_missing_time(fitter, method):
    model = fitter.fit(X, C)
    f = getattr(model, method)
    _check(f(T), f(T[1:]))


def test_nonparametric_hf_at_a_single_missing_time():
    model = surv.KaplanMeier.fit(X, C)
    assert np.isnan(model.hf(np.nan)).all()
    assert np.isnan(model.hf(np.array([np.nan, np.nan]))).all()


@pytest.mark.parametrize("bound", ["two-sided", "upper", "lower"])
def test_nonparametric_bounds_at_a_missing_time(bound):
    model = surv.KaplanMeier.fit(X, C)
    # (T goes past the last time, where the bounds warn, #665.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with_nan = np.atleast_2d(model.cb(T, bound=bound).T).T  # a row a time
        without = np.atleast_2d(model.cb(T[1:], bound=bound).T).T
    assert np.isnan(with_nan[0]).all()
    np.testing.assert_allclose(with_nan[1:], without)


def test_nonparametric_band_at_a_missing_time():
    model = surv.KaplanMeier.fit(X, C)
    band = model.band(T)
    assert np.isnan(band[0]).all()
    np.testing.assert_allclose(band[1:], model.band(T[1:]))


@pytest.mark.parametrize("method", ["Nelson-Aalen", "Kaplan-Meier"])
def test_competing_risks_at_a_missing_time(method):
    model = CompetingRisks.fit(X, E, C, how=method)
    _check(model.cif(T, event=1), model.cif(T[1:], event=1))
    _check(model.sf(T, event=1), model.sf(T[1:], event=1))
    _check(model.sf(T), model.sf(T[1:]))
    _check(model.Hf(T, event=2), model.Hf(T[1:], event=2))
    _check(model.hf(T), model.hf(T[1:]))


@pytest.mark.parametrize(
    "fitter", [surv.WeibullPH, surv.WeibullAFT, surv.WeibullPO]
)
def test_tvc_evaluation_at_a_missing_time(fitter):
    Z = np.random.default_rng(0).normal(size=(10, 1))
    model = fitter.fit(X, Z=Z, c=C)
    path = np.array([[0.1], [0.4]])
    xl = np.array([0.0, 4.0])
    t = np.array([np.nan, 2.0, 6.0])
    _check(model.sf_tvc(t, path, xl=xl), model.sf_tvc(t[1:], path, xl=xl))
    _check(model.Hf_tvc(t, path, xl=xl), model.Hf_tvc(t[1:], path, xl=xl))
    assert np.isnan(model.sf_tvc(np.array([np.nan]), path, xl=xl)).all()
