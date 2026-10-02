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
    with pytest.warns(UserWarning, match="did not reach a verified maximum"):
        sp.recurrent.Duane.fit(**recurrent_data())


def _rossi():
    from surpyval.datasets import load_rossi_static

    data = load_rossi_static()
    covariates = ["fin", "age", "race", "wexp", "mar", "paro", "prio"]
    return {
        "x": data["week"].values,
        "Z": data[covariates].values,
        "i": np.arange(len(data)),
        "c": 1 - data["arrest"].values,
    }


@pytest.mark.parametrize(
    "init",
    [
        [0.017, 5, 5, 5, 5, 5, 5, 5],
        [0.017, 50, 0, 0, 0, 0, 0, 0],
        [0.017, 0, 0, 0, 0, 0, 0, 30],
    ],
)
def test_554_an_hpp_proportional_intensity_fit_from_a_poor_start(init):
    # The HPP fit kept its one BFGS search: from these starts exp(beta'Z)
    # overflows, and it returned a rate of 0 with nan coefficients, or
    # its start (a log-likelihood 2e234 below the maximum), in silence.
    # It now also searches from the default start and keeps the better.
    fitter = sp.recurrent.ProportionalIntensityHPP
    data = _rossi()
    ref = _silent(fitter.fit, **data)
    got = _silent(fitter.fit, **data, init=init)
    np.testing.assert_allclose(_ll(got), _ll(ref), rtol=1e-10)
    np.testing.assert_allclose(
        np.r_[got.params, got.coeffs], np.r_[ref.params, ref.coeffs], rtol=1e-5
    )


def test_554_an_unverified_hpp_proportional_intensity_fit_warns(
    monkeypatch,
):
    # A search that stops short of the maximum (here: never moves, and
    # says it succeeded) is not kept in silence: one warning, at the
    # caller's line.
    from scipy.optimize import OptimizeResult

    from surpyval.recurrent.regression import hpp_proportional_intensity

    def stuck(fun, x0, *args, **kwargs):
        return OptimizeResult(
            x=np.asarray(x0, dtype=float), fun=fun(x0), success=True
        )

    monkeypatch.setattr(hpp_proportional_intensity, "minimize", stuck)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sp.recurrent.ProportionalIntensityHPP.fit(**_rossi())
    assert len(caught) == 1, [str(w.message) for w in caught]
    message = str(caught[0].message)
    assert message.startswith("The proportional intensity fit did not")
    assert "verified maximum" in message
    assert caught[0].filename == __file__
