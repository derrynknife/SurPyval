"""Truncated rows in the Gaussian and Student-t copulas of three or more
series (``elliptical_nd.py``).

A truncated row's likelihood is divided by the copula's mass over its
truncation window. With the window in one series only that mass is the
series' own probability over it (``ur - ul``), and with it in two series
it is the bivariate copula's mass over the window, as the bivariate code
computes it. Seeds, sample sizes and tolerances were fixed before any
result was seen.
"""

import numpy as np
import pytest

import surpyval as sp
from surpyval.multivariate import Gaussian, MultivariateSurpyvalData, StudentT
from surpyval.multivariate.parametric.copula.elliptical_nd import for_axes

R3 = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.4], [0.2, 0.4, 1.0]])
NU = 5.0


def _margins(D=3):
    return [
        sp.Weibull.from_params([10.0 * (d + 1), 1.5 + 0.5 * d])
        for d in range(D)
    ]


def _dims(copula, margins, data):
    return [
        copula._prepare_dim(margins[d], *data.dimension(d))
        for d in range(data.D)
    ]


def _data(seed, truncate, size=40, censor=True):
    """Rows of three series, some censored, each truncated on the left in
    the series ``truncate`` at a fraction of its value."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(3.0, 30.0, (size, 3))
    c = np.zeros(X.shape, int)
    if censor:
        c[: size // 2, 2] = 1
        c[size // 4 : size // 2, 0] = -1
    t = np.full(X.shape + (2,), np.inf)
    t[..., 0] = -np.inf
    for d in truncate:
        t[:, d, 0] = X[:, d] * rng.uniform(0.1, 0.6, size)
    t[::5, truncate[0], 1] = X[::5, truncate[0]] * 1.5  # some on the right
    plain = MultivariateSurpyvalData(X, c=c)
    return plain, MultivariateSurpyvalData(X, c=c, t=t)


@pytest.mark.parametrize("family", [Gaussian, StudentT])
def test_truncation_in_one_series_divides_by_its_probability(family):
    margins = _margins()
    copula = for_axes(family.name, 3, "unstructured")
    params = [R3[i, j] for i, j in copula.pairs] + (
        [NU] if family is StudentT else []
    )
    plain, truncated = _data(4731, [1])
    dims = _dims(copula, margins, truncated)
    expected = copula._rows_loglik(params, _dims(copula, margins, plain))
    expected = expected - np.log(dims[1]["ur"] - dims[1]["ul"])
    np.testing.assert_allclose(
        copula._rows_loglik(params, dims), expected, rtol=0, atol=1e-10
    )


@pytest.mark.parametrize("family", [Gaussian, StudentT])
def test_truncation_in_two_series_divides_by_the_pairs_mass(family):
    margins = _margins()
    copula = for_axes(family.name, 3, "unstructured")
    nu = [NU] if family is StudentT else []
    params = [R3[i, j] for i, j in copula.pairs] + nu
    plain, truncated = _data(4732, [0, 2])
    dims = _dims(copula, margins, truncated)
    untruncated = copula._rows_loglik(params, _dims(copula, margins, plain))
    pair = family._trunc_logmass([R3[0, 2]] + nu, dims[0], dims[2])
    np.testing.assert_allclose(
        copula._rows_loglik(params, dims), untruncated - pair, atol=1e-8
    )


def test_truncation_in_four_series_uses_the_lattice():
    """Four series, the fourth independent of the others: the window's
    mass is the three series' mass times the fourth's probability."""
    margins = _margins(4)
    R = np.eye(4)
    R[:3, :3] = R3
    copula = for_axes("Gaussian", 4, "unstructured")
    rng = np.random.default_rng(4733)
    X = rng.uniform(3.0, 30.0, (20, 4))
    t = np.full(X.shape + (2,), np.inf)
    t[..., 0] = X * rng.uniform(0.1, 0.6, X.shape)
    data = MultivariateSurpyvalData(X, t=t)
    dims = _dims(copula, margins, data)
    logmass = copula._truncation_logmass(R, None, dims)
    three = for_axes("Gaussian", 3, "unstructured")
    expected = three._truncation_logmass(R3, None, dims[:3]) + np.log(
        1.0 - dims[3]["ul"]
    )
    np.testing.assert_allclose(logmass, expected, atol=2e-5)


# (a fit to late entrants: calibration/test_recovery_copula_nd_truncated.py)
