"""Recurrent fits from a start far from the maximum (#429).

A user's ``init`` is followed by the default start, and the better answer
kept (principles 12 and 13): from a start far from the maximum the NHPP
and renewal searches used to stay where they began, silently.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.tests.conformance.registry import recurrent_data


def _silent(fit, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fit(**kwargs)


def _ll(model):
    return -float(model._neg_ll(model._mle))


@pytest.mark.parametrize("fitter", ["Duane", "CoxLewis", "CrowAMSAA"])
def test_an_nhpp_fit_from_a_far_start(fitter):
    # Duane from alpha x1e6 kept alpha at 7.8e5, where the intensity
    # overflows: cif(5) inf and a NaN log-likelihood, no warning.
    fitter = getattr(sp.recurrent, fitter)
    data = recurrent_data()
    ref = _silent(fitter.fit, **data)
    far = np.array(ref.params, dtype=float)
    far[0] *= 1e6
    got = _silent(fitter.fit, **data, init=far)
    np.testing.assert_allclose(_ll(got), _ll(ref), rtol=1e-6)
    np.testing.assert_allclose(got.params, ref.params, rtol=1e-3)


def test_a_proportional_intensity_fit_from_a_far_start():
    data = recurrent_data(with_Z=True)
    fitter = sp.recurrent.ProportionalIntensityNHPP
    ref = _silent(fitter.fit, **data)
    far = np.r_[ref.params, ref.coeffs]
    far[0] *= 1e6
    got = _silent(fitter.fit, **data, init=far)
    np.testing.assert_allclose(_ll(got), _ll(ref), rtol=1e-6)
    np.testing.assert_allclose(got.params, ref.params, rtol=1e-3)


@pytest.mark.parametrize("fitter", ["GeneralizedOneRenewal", "ARI"])
def test_a_renewal_fit_from_a_far_start(fitter):
    # ARI from its baseline scale x1e6 kept it at 4.2e6: a log-likelihood
    # of -264.1 against -37.4, and mcf(55) 0 against 4.8.
    fitter = getattr(sp.recurrent, fitter)
    data = recurrent_data()
    ref = fitter.fit(**data)
    far = np.r_[ref.restoration, ref.model.params]
    far[1] *= 1e6
    got = fitter.fit(**data, init=far)
    np.testing.assert_allclose(_ll(got), _ll(ref), rtol=1e-6)
    np.testing.assert_allclose(
        np.r_[got.restoration, got.model.params],
        np.r_[ref.restoration, ref.model.params],
        rtol=1e-3,
    )


def test_an_unconverged_nhpp_fit_warns(monkeypatch):
    # Where the kept answer's optimiser did not converge, the fit says so
    from scipy.optimize import OptimizeResult

    from surpyval.recurrent.parametric import nhpp_fitter

    real = nhpp_fitter.minimize

    def capped(*args, **kwargs):
        res = real(*args, **kwargs)
        return OptimizeResult({**res, "success": False})

    monkeypatch.setattr(nhpp_fitter, "minimize", capped)
    with pytest.warns(UserWarning, match="did not converge"):
        sp.recurrent.Duane.fit(**recurrent_data())
