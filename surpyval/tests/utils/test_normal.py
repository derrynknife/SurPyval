"""The normal functions from ``scipy.special`` (#469).

``Normal`` and ``LogNormal`` called ``scipy.stats.norm`` (through
autograd's wrapper), whose generic argument handling cost 4 to 7 times
the maths. ``surpyval.utils.normal`` evaluates what ``scipy.stats.norm``
evaluates once its checks are done, so the values must be the same to
the last bit -- tails, infinities, NaN and bad scales included -- and the
derivatives the fits take must be those of the old wrapper.
"""

import warnings

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import elementwise_grad, hessian
from autograd.scipy.stats import norm as autograd_norm
from scipy.stats import norm as scipy_norm

import surpyval as sp
from surpyval.utils import normal

_RNG = np.random.default_rng(0)
X = np.concatenate(
    [
        _RNG.normal(0.0, 3.0, 5000),
        _RNG.uniform(-40.0, 40.0, 2000),
        [-np.inf, np.inf, np.nan, 0.0, 1.0, -1.0, 1e300, -1e300, 38.5],
        [-38.5, 1e-300, -5e-324, 37.0, 8.3, -8.3],
    ]
)
Q = np.concatenate(
    [
        _RNG.uniform(0.0, 1.0, 5000),
        [0.0, 1.0, -0.1, 1.1, np.nan, 1e-300, 5e-324, 1 - 1e-16],
    ]
)
PARAMS = [
    (0.0, 1.0),
    (3.0, 4.0),
    (-2.5, 0.01),
    (1e3, 1e-3),
    (0.0, 0.0),
    (0.0, -1.0),
    (np.nan, 1.0),
    (0.0, np.nan),
]
FUNCTIONS = ("pdf", "logpdf", "cdf", "sf", "logcdf", "logsf")


def _identical(a, b):
    assert type(a) is type(b)
    np.testing.assert_array_equal(a, b, strict=True)


@pytest.mark.parametrize("loc, scale", PARAMS)
@pytest.mark.parametrize("name", FUNCTIONS)
def test_same_bits_as_scipy_stats(name, loc, scale):
    with np.errstate(all="ignore"):
        _identical(
            getattr(normal, name)(X, loc, scale),
            getattr(scipy_norm, name)(X, loc, scale),
        )
        for x in (0.3, -np.inf, np.nan, 40.0):
            _identical(
                getattr(normal, name)(x, loc, scale),
                getattr(scipy_norm, name)(x, loc, scale),
            )


@pytest.mark.parametrize("loc, scale", PARAMS)
@pytest.mark.parametrize("name", ("ppf", "isf"))
def test_quantiles_same_bits_as_scipy_stats(name, loc, scale):
    with np.errstate(all="ignore"):
        _identical(
            getattr(normal, name)(Q, loc, scale),
            getattr(scipy_norm, name)(Q, loc, scale),
        )
        for q in (0.3, 0.0, 1.0, np.nan, 2.0):
            _identical(
                getattr(normal, name)(q, loc, scale),
                getattr(scipy_norm, name)(q, loc, scale),
            )


@pytest.mark.parametrize("name", FUNCTIONS)
@pytest.mark.parametrize("argnum", (0, 1, 2))
def test_gradients_match_autograds_wrapper(name, argnum):
    # Into the far tails, where log_ndtr's derivative is taken from logs.
    x = np.linspace(-45.0, 45.0, 181)

    def grad(lib):
        g = elementwise_grad(getattr(lib, name), argnum)
        if argnum == 0:
            return g(x, 1.5, 2.0)
        return np.array([g(xi, 1.5, 2.0) for xi in x])

    np.testing.assert_allclose(
        grad(normal), grad(autograd_norm), rtol=1e-13, atol=0
    )


def test_hessian_of_a_censored_likelihood():
    x = np.linspace(-10.0, 30.0, 80)

    def ll(p, lib):
        return anp.sum(lib.logpdf(x[:50], p[0], p[1])) + anp.sum(
            lib.logsf(x[50:], p[0], p[1])
        )

    p = np.array([1.0, 3.0])
    np.testing.assert_allclose(
        hessian(lambda p: ll(p, normal))(p),
        hessian(lambda p: ll(p, autograd_norm))(p),
        rtol=1e-12,
    )


@pytest.mark.parametrize("dist", ["Normal", "LogNormal"])
def test_distributions_unchanged(dist):
    # The families' own functions, as they were through scipy.stats.
    D = getattr(sp, dist)
    x = np.abs(X) if dist == "LogNormal" else X
    with np.errstate(all="ignore"):
        if dist == "Normal":
            want_sf = scipy_norm.sf(x, 3.0, 4.0)
            want_qf = scipy_norm.ppf(Q, 3.0, 4.0)
        else:
            want_sf = scipy_norm.sf(np.log(x), 3.0, 4.0)
            want_qf = np.exp(scipy_norm.ppf(Q, 3.0, 4.0))
        np.testing.assert_array_equal(D.sf(x, 3.0, 4.0), want_sf)
        np.testing.assert_array_equal(D.qf(Q, 3.0, 4.0), want_qf)


def test_no_overflow_warning_far_out():
    # scipy.stats let numpy's "overflow encountered in square" through at
    # |x| = 1e300 (principle 22); the density there is 0 and its log -inf.
    x = np.array([-1e300, 1e300])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_array_equal(sp.Normal.df(x, 3.0, 4.0), [0.0, 0.0])
        np.testing.assert_array_equal(
            sp.Normal.log_df(x, 3.0, 4.0), [-np.inf, -np.inf]
        )
