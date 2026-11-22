"""Rectangle probabilities of the multivariate normal and t distributions
(``_rectangle.py``), which the likelihood and the functions of an
elliptical copula of more than two series are made of.

The references are exact where there is one: the orthant probability of
three centred coordinates, :math:`\\frac18 + \\frac{1}{4\\pi}(\\arcsin
\\rho_{12} + \\arcsin\\rho_{13} + \\arcsin\\rho_{23})` (the same for the
normal and every t, David 1953), the equicorrelated orthant of
:math:`\\rho = 1/2`, :math:`1/(k + 1)`, and products for independent
coordinates. scipy's multivariate normal CDF is a second, numerical,
reference (a randomised quasi-Monte Carlo rule, accurate to about 1e-7
at the settings used).
"""

import numpy as np
import pytest
from scipy.stats import multivariate_normal, multivariate_t, norm
from scipy.stats import t as student_t

from surpyval.multivariate.parametric.copula._rectangle import (
    MAX_AXES,
    MAX_LIKELIHOOD_AXES,
    rectangle,
)

INF = np.inf


def _correlations(k, seed, count=5):
    """Random correlation matrices, then a strongly positive and a nearly
    singular negative equicorrelated one."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        a = rng.standard_normal((k, k))
        s = a @ a.T
        d = np.sqrt(np.diag(s))
        out.append(s / np.outer(d, d))
    for r in (0.95, -0.9 / (k - 1)):
        out.append(r * np.ones((k, k)) + (1 - r) * np.eye(k))
    return out


def _orthant3(corr):
    return 1 / 8 + (
        np.arcsin(corr[0, 1]) + np.arcsin(corr[0, 2]) + np.arcsin(corr[1, 2])
    ) / (4 * np.pi)


def test_one_coordinate_is_the_univariate_cdf():
    lo = np.array([[-INF], [0.5], [-1.0], [7.0]])
    hi = np.array([[0.3], [INF], [2.0], [9.0]])
    one = np.eye(1)
    expected = [
        norm.cdf(0.3),
        norm.sf(0.5),
        norm.cdf(2.0) - norm.cdf(-1.0),
        norm.sf(7.0) - norm.sf(9.0),
    ]
    np.testing.assert_allclose(rectangle(lo, hi, one), expected, rtol=1e-13)
    expected_t = [
        student_t.cdf(0.3, 3.5),
        student_t.sf(0.5, 3.5),
        student_t.cdf(2.0, 3.5) - student_t.cdf(-1.0, 3.5),
        student_t.sf(7.0, 3.5) - student_t.sf(9.0, 3.5),
    ]
    np.testing.assert_allclose(
        rectangle(lo, hi, one, nu=3.5), expected_t, rtol=1e-12
    )


@pytest.mark.parametrize("nu", [None, 2.5, 7.0])
def test_two_coordinates_orthants_are_exact(nu):
    # P(Z1 <= 0, Z2 <= 0) = 1/4 + arcsin(rho) / (2 pi), normal or t
    for rho in (-0.9, -0.3, 0.0, 0.4, 0.99):
        corr = np.array([[1.0, rho], [rho, 1.0]])
        lower = rectangle([[-INF, -INF]], [[0.0, 0.0]], corr, nu)
        upper = rectangle([[0.0, 0.0]], [[INF, INF]], corr, nu)
        mixed = rectangle([[0.0, -INF]], [[INF, 0.0]], corr, nu)
        exact = 0.25 + np.arcsin(rho) / (2 * np.pi)
        np.testing.assert_allclose(lower, exact, atol=1e-10)
        np.testing.assert_allclose(upper, exact, atol=1e-10)
        np.testing.assert_allclose(mixed, 0.5 - exact, atol=1e-10)


@pytest.mark.parametrize("nu", [None, 1.5, 4.0, 30.0])
def test_three_coordinates_orthants_are_exact(nu):
    for corr in _correlations(3, seed=11):
        exact = _orthant3(corr)
        lo = np.array([[-INF] * 3, [0.0] * 3])
        hi = np.array([[0.0] * 3, [INF] * 3])
        np.testing.assert_allclose(
            rectangle(lo, hi, corr, nu), [exact, exact], atol=1e-9
        )
        # one coordinate above 0, two below: the orthant with that
        # coordinate's correlations negated
        flip = np.diag([-1.0, 1.0, 1.0])
        np.testing.assert_allclose(
            rectangle([[0.0, -INF, -INF]], [[INF, 0.0, 0.0]], corr, nu),
            _orthant3(flip @ corr @ flip),
            atol=1e-9,
        )


def test_three_coordinates_agree_with_scipy():
    rng = np.random.default_rng(12)
    worst = 0.0
    for i, corr in enumerate(_correlations(3, seed=13)):
        lo = rng.normal(-1.0, 1.0, (3, 3))
        hi = lo + rng.exponential(1.5, (3, 3))
        if i % 3 == 1:
            hi[:] = INF
        if i % 3 == 2:
            lo[:] = -INF
        ours = rectangle(lo, hi, corr)
        for row in range(3):
            ref = multivariate_normal.cdf(
                hi[row],
                mean=np.zeros(3),
                cov=corr,
                lower_limit=lo[row],
                abseps=1e-10,
                releps=1e-10,
                maxpts=3_000_000,
                allow_singular=True,
            )
            worst = max(worst, abs(ours[row] - ref))
    assert worst < 2e-6, worst


def test_three_t_coordinates_tend_to_the_normal_and_agree_with_scipy():
    rng = np.random.default_rng(14)
    corr = _correlations(3, seed=15)[0]
    lo = rng.normal(-1.0, 1.0, (4, 3))
    hi = lo + rng.exponential(1.5, (4, 3))
    np.testing.assert_allclose(
        rectangle(lo, hi, corr, nu=1e7), rectangle(lo, hi, corr), atol=1e-6
    )
    ours = rectangle(np.full((4, 3), -INF), hi, corr, nu=4.0)
    ref = [
        multivariate_t(shape=corr, df=4.0, seed=1).cdf(h, maxpts=2_000_000)
        for h in hi
    ]
    # (scipy's t CDF is good to about 1e-4)
    np.testing.assert_allclose(ours, ref, atol=5e-4)


def test_independent_coordinates_multiply():
    rng = np.random.default_rng(16)
    for k in (3, 4, 5):
        lo = rng.normal(-1.0, 1.0, (3, k))
        hi = lo + rng.exponential(1.5, (3, k))
        exact = np.prod(norm.cdf(hi) - norm.cdf(lo), axis=1)
        np.testing.assert_allclose(
            rectangle(lo, hi, np.eye(k)), exact, atol=1e-9 if k == 3 else 1e-5
        )


@pytest.mark.parametrize("k", [4, 5, 6])
def test_many_coordinates_by_the_lattice(k):
    # Equicorrelated 1/2: P(all <= 0) = 1 / (k + 1), normal or t
    corr = 0.5 * np.ones((k, k)) + 0.5 * np.eye(k)
    lo, hi = np.full((1, k), -INF), np.zeros((1, k))
    np.testing.assert_allclose(rectangle(lo, hi, corr), 1 / (k + 1), atol=1e-4)
    np.testing.assert_allclose(
        rectangle(lo, hi, corr, nu=5.0), 1 / (k + 1), atol=1e-4
    )


def test_four_coordinates_agree_with_scipy():
    rng = np.random.default_rng(17)
    worst = 0.0
    for corr in _correlations(4, seed=18, count=3):
        hi = rng.normal(0.5, 1.0, (2, 4))
        ours = rectangle(np.full((2, 4), -INF), hi, corr)
        for row in range(2):
            ref = multivariate_normal.cdf(
                hi[row],
                mean=np.zeros(4),
                cov=corr,
                abseps=1e-8,
                releps=1e-8,
                maxpts=2_000_000,
                allow_singular=True,
            )
            worst = max(worst, abs(ours[row] - ref))
    assert worst < 5e-4, worst


def test_the_rule_is_deterministic_and_rows_are_independent():
    rng = np.random.default_rng(19)
    for k in (3, 4):
        corr = _correlations(k, seed=20)[1]
        lo = rng.normal(-1.0, 1.0, (5, k))
        hi = lo + rng.exponential(1.5, (5, k))
        whole = rectangle(lo, hi, corr)
        np.testing.assert_array_equal(whole, rectangle(lo, hi, corr))
        one_by_one = [rectangle(lo[[i]], hi[[i]], corr)[0] for i in range(5)]
        np.testing.assert_allclose(whole, one_by_one, rtol=0, atol=1e-15)
        # repeated rows (units censored at one time) give one value
        twice = rectangle(np.r_[lo, lo], np.r_[hi, hi], corr)
        np.testing.assert_array_equal(twice[:5], twice[5:])


def test_too_many_coordinates_are_refused():
    k = MAX_AXES + 1
    with pytest.raises(ValueError, match="fewer axes at a time"):
        rectangle(np.full((1, k), -INF), np.zeros((1, k)), np.eye(k))
    assert MAX_LIKELIHOOD_AXES == 3
