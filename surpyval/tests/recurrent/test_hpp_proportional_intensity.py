import numpy as np

from surpyval.recurrent import ProportionalIntensityHPP
from surpyval.utils.recurrent_utils import handle_xicn

# Shared single-covariate design used across the censoring cases below.
Z = np.array([[0.1], [0.1], [0.5], [0.5], [0.9], [0.9]])
I = np.array([1, 1, 2, 2, 3, 3])


def test_left_censored_likelihood_evaluates():
    # Regression test: Z_left was only bound in the no-censoring else branch,
    # so any left-censored data raised a NameError inside the likelihood.
    x = np.array([5.0, 8.0, 6.0, 10.0, 7.0, 9.0])
    c = np.array([-1, 0, -1, 0, -1, 0])
    n = np.array([2, 1, 3, 1, 1, 1])
    data = handle_xicn(x, I, c, n, Z=Z, as_recurrent_data=True)
    negll = ProportionalIntensityHPP.create_negll_func(data)
    assert np.isfinite(negll(np.array([0.5, 0.2])))


def test_interval_censored_likelihood_evaluates():
    # Regression test: Z_i had the same scoping bug for interval-censored data.
    x = np.array(
        [
            [5.0, 5.0],
            [5.0, 9.0],
            [6.0, 6.0],
            [6.0, 11.0],
            [7.0, 7.0],
            [7.0, 12.0],
        ]
    )
    c = np.array([0, 2, 0, 2, 0, 2])
    n = np.array([1, 2, 1, 3, 1, 2])
    data = handle_xicn(x, I, c, n, Z=Z, as_recurrent_data=True)
    negll = ProportionalIntensityHPP.create_negll_func(data)
    assert np.isfinite(negll(np.array([0.5, 0.2])))


def test_left_censored_fit_succeeds():
    x = np.array([5.0, 8.0, 6.0, 10.0, 7.0, 9.0])
    c = np.array([-1, 0, -1, 0, -1, 0])
    n = np.array([2, 1, 3, 1, 1, 1])
    model = ProportionalIntensityHPP.fit(x, Z, i=I, c=c, n=n)
    assert model.res.success
    assert np.all(np.isfinite(model.params))
    assert np.all(np.isfinite(model.coeffs))


def test_interval_censored_fit_succeeds():
    x = np.array(
        [
            [5.0, 5.0],
            [5.0, 9.0],
            [6.0, 6.0],
            [6.0, 11.0],
            [7.0, 7.0],
            [7.0, 12.0],
        ]
    )
    c = np.array([0, 2, 0, 2, 0, 2])
    n = np.array([1, 2, 1, 3, 1, 2])
    model = ProportionalIntensityHPP.fit(x, Z, i=I, c=c, n=n)
    assert model.res.success
    assert np.all(np.isfinite(model.params))
    assert np.all(np.isfinite(model.coeffs))


def test_right_censored_fit_still_works():
    # The previously working observed/right-censored path must be unaffected.
    x = np.array([5.0, 8.0, 6.0, 10.0])
    i = np.array([1, 1, 2, 2])
    c = np.array([0, 1, 0, 1])
    Z_rc = np.array([[0.1], [0.1], [0.5], [0.5]])
    model = ProportionalIntensityHPP.fit(x, Z_rc, i=i, c=c)
    assert model.res.success
    assert np.all(np.isfinite(model.params))
