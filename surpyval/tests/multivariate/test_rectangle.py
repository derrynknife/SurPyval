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
    for k in (3, 4, 5, 8):
        lo = rng.normal(-1.0, 1.0, (3, k))
        hi = lo + rng.exponential(1.5, (3, k))
        exact = np.prod(norm.cdf(hi) - norm.cdf(lo), axis=1)
        np.testing.assert_allclose(
            rectangle(lo, hi, np.eye(k)), exact, rtol=1e-9
        )


@pytest.mark.parametrize("k", [4, 5, 6, 8, 10])
def test_many_coordinates_by_the_lattice(k):
    # Equicorrelated 1/2: P(all <= 0) = 1 / (k + 1), normal or t
    corr = 0.5 * np.ones((k, k)) + 0.5 * np.eye(k)
    lo, hi = np.full((1, k), -INF), np.zeros((1, k))
    np.testing.assert_allclose(rectangle(lo, hi, corr), 1 / (k + 1), atol=1e-6)
    np.testing.assert_allclose(
        rectangle(lo, hi, corr, nu=5.0), 1 / (k + 1), atol=1e-5
    )


# A one-factor correlation, R = l l' off the diagonal, makes the
# coordinates independent given one standard normal W (Z_i = l_i W +
# sqrt(1 - l_i^2) e_i), so the rectangle is a one-dimensional integral over
# W of a product of univariate CDFs (and, for the t, a second integral over
# the scale); the trapezoidal rule on [-12, 12] gives it to about 1e-13.
_W = np.linspace(-12.0, 12.0, 24001)
_PHI = np.exp(-0.5 * _W**2) / np.sqrt(2.0 * np.pi)


def _one_factor(lo, hi, lam):
    s = np.sqrt(1.0 - lam**2)
    out = []
    for a, b in zip(lo, hi):
        A = (a[None, :] - lam * _W[:, None]) / s
        B = (b[None, :] - lam * _W[:, None]) / s
        flip = (A + B) > 0
        p = np.where(flip, norm.sf(A) - norm.sf(B), norm.cdf(B) - norm.cdf(A))
        out.append(np.trapezoid(_PHI * np.prod(p, axis=1), _W))
    return np.array(out)


def _one_factor_t(lo, hi, lam, nu):
    from scipy.stats import chi2

    from surpyval.multivariate.parametric.copula.elliptical import (
        _tanh_sinh,
    )

    z, zc, w = _tanh_sinh(1.0 / 32.0)
    scale = np.sqrt(np.where(z <= 0.5, chi2.ppf(z, nu), chi2.isf(zc, nu)) / nu)
    return sum(
        wj * _one_factor(lo * sj, hi * sj, lam) for sj, wj in zip(scale, w)
    )


def _censored_limits(rng, m, k):
    """Each row's limits as a censored row's: right, left and interval
    censored coordinates, mixed."""
    lo = np.where(rng.random((m, k)) < 0.6, rng.normal(0, 1.5, (m, k)), -INF)
    hi = np.where(np.isfinite(lo), INF, rng.normal(0, 1.5, (m, k)))
    both = (rng.random((m, k)) < 0.2) & np.isfinite(lo)
    hi = np.where(both, lo + rng.uniform(0.2, 2.0, (m, k)), hi)
    return lo, hi


@pytest.mark.parametrize("k", [4, 5, 6, 8, 10])
def test_many_coordinates_agree_with_one_factor_integrals(k):
    rng = np.random.default_rng(30 + k)
    worst_abs = worst_rel = 0.0
    for trial in range(3):
        lam = (
            rng.uniform(-0.95, 0.95, k) if trial else rng.uniform(0.3, 0.97, k)
        )
        corr = np.outer(lam, lam)
        np.fill_diagonal(corr, 1.0)
        lo, hi = _censored_limits(rng, 8, k)
        ref = _one_factor(lo, hi, lam)
        ours = rectangle(lo, hi, corr)
        worst_abs = max(worst_abs, np.max(np.abs(ours - ref)))
        worst_rel = max(worst_rel, np.max(np.abs(ours / ref - 1.0)))
    # (measured: below 3.2e-6 and 4e-4 for k = 4 to 10)
    assert worst_abs < 5e-6, worst_abs
    assert worst_rel < 1e-3, worst_rel


@pytest.mark.parametrize("nu", [1.5, 4.0])
def test_many_t_coordinates_agree_with_one_factor_integrals(nu):
    rng = np.random.default_rng(41)
    worst = 0.0
    for k in (4, 6):
        lam = rng.uniform(-0.9, 0.9, k)
        corr = np.outer(lam, lam)
        np.fill_diagonal(corr, 1.0)
        lo, hi = _censored_limits(rng, 6, k)
        ref = _one_factor_t(lo, hi, lam, nu)
        worst = max(worst, np.max(np.abs(rectangle(lo, hi, corr, nu) - ref)))
    # (measured: below 3e-6)
    assert worst < 5e-6, worst


def test_a_block_of_three_and_an_independent_coordinate():
    """Four coordinates, the last independent of the others: the lattice
    rule gives the three-coordinate quadrature's value times the
    univariate probability (a one-factor block, whose integral is also
    exact, checks both)."""
    rng = np.random.default_rng(42)
    lam = np.array([0.8, -0.6, 0.7, 0.0])
    corr4 = np.outer(lam, lam)
    np.fill_diagonal(corr4, 1.0)
    lo, hi = _censored_limits(rng, 10, 4)
    exact = _one_factor(lo, hi, lam)
    three = rectangle(lo[:, :3], hi[:, :3], corr4[:3, :3]) * rectangle(
        lo[:, 3:], hi[:, 3:], np.eye(1)
    )
    np.testing.assert_allclose(three, exact, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(
        rectangle(lo, hi, corr4), exact, rtol=1e-5, atol=1e-7
    )


def test_a_nearly_singular_correlation_matrix():
    """A correlation matrix with smallest eigenvalue 7e-4, against R's
    mvtnorm (Miwa's algorithm, which Genz-Bretz confirms to 1e-9): the
    prioritised order keeps the lattice rule to 1e-5 of the probability,
    where a fixed order was wrong by a factor of 8."""
    corr = _correlations(3, seed=43)[0]
    corr4 = np.eye(4)
    corr4[:3, :3] = corr
    lo = np.array([[1.01837034, 0.1013686, 0.4336791, -INF]])
    hi = np.array([[1.74202245, INF, INF, INF]])
    # pmvnorm(lo, hi, corr, algorithm = Miwa(steps = 4096)) of the three
    np.testing.assert_allclose(
        rectangle(lo, hi, corr4), 4.380972e-07, rtol=1e-4
    )


def test_the_lattice_rule_is_smooth_in_the_correlations():
    """A likelihood search and its finite-difference Hessian need the
    rule to be a smooth function of the correlations: its second
    differences agree from a step of 1e-2 down to 1e-5."""
    k = 5
    base = 0.3 * np.ones((k, k)) + 0.7 * np.eye(k)
    lo = np.array([[0.2, -INF, -0.5, 0.1, -1.0]])
    hi = np.array([[INF, 0.4, 0.7, INF, INF]])

    def f(r, nu):
        corr = base.copy()
        corr[0, 1] = corr[1, 0] = r
        corr[2, 3] = corr[3, 2] = 0.3 + 0.5 * (r - 0.3)
        return np.log(rectangle(lo, hi, corr, nu)[0])

    for nu in (None, 4.0):
        second = [
            (f(0.3 + h, nu) - 2 * f(0.3, nu) + f(0.3 - h, nu)) / h**2
            for h in (1e-2, 1e-3, 1e-4, 1e-5)
        ]
        np.testing.assert_allclose(second, second[2], rtol=2e-4)


def test_the_order_changes_the_result_only_within_the_rules_accuracy():
    rng = np.random.default_rng(44)
    corr = _correlations(5, seed=45)[2]
    lo, hi = _censored_limits(rng, 6, 5)
    default = rectangle(lo, hi, corr)
    for perm in ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0]):
        given = rectangle(lo, hi, corr, order=np.tile(perm, (6, 1)))
        np.testing.assert_allclose(given, default, rtol=1e-3, atol=1e-5)


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
    assert worst < 1e-5, worst


def test_the_rule_is_deterministic_and_rows_are_independent():
    rng = np.random.default_rng(19)
    for k in (3, 4, 6):
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
    with pytest.raises(ValueError, match="fewer axes at a time"):
        rectangle(
            np.full((1, k - 1), -INF), np.zeros((1, k - 1)), np.eye(k - 1), 4.0
        )


def _nearly_singular(rng, smallest):
    """A random correlation matrix of three coordinates whose smallest
    eigenvalue is ``smallest``."""
    a = rng.normal(size=(3, 3))
    s = a @ a.T
    for _ in range(50):
        d = np.sqrt(np.diag(s))
        s = s / np.outer(d, d)
        w, v = np.linalg.eigh(s)
        w[0] = smallest
        s = v @ np.diag(w) @ v.T
    d = np.sqrt(np.diag(s))
    s = s / np.outer(d, d)
    return (s + s.T) / 2


@pytest.mark.parametrize("nu", [None, 4.0])
def test_three_coordinates_of_a_nearly_singular_matrix(monkeypatch, nu):
    """With a smallest eigenvalue of 1e-4 or 1e-3 the other two
    coordinates given the first are almost perfectly correlated, and their
    rectangle changes sharply where their standardised limits meet. The
    quadrature agrees with one of a ten times finer step (measured: within
    4e-10; the rule before it split its panels at those points was off
    by up to 1.3e-4, and 10% of the probability)."""
    from surpyval.multivariate.parametric.copula import _rectangle
    from surpyval.multivariate.parametric.copula.elliptical import (
        _tanh_sinh,
    )

    rng = np.random.default_rng(675)
    cases = [
        (_nearly_singular(rng, smallest), *_censored_limits(rng, 8, 3))
        for smallest in (1e-4, 1e-4, 1e-3, 1e-3)
    ]
    ours = [rectangle(lo, hi, corr, nu) for corr, lo, hi in cases]
    fine = _tanh_sinh(step=1.0 / 128.0, reach=4.5)
    monkeypatch.setattr(_rectangle, "_Z", fine[0])
    monkeypatch.setattr(_rectangle, "_ZC", fine[1])
    monkeypatch.setattr(_rectangle, "_W", fine[2])
    for (corr, lo, hi), got in zip(cases, ours):
        assert np.linalg.eigvalsh(corr)[0] < 2e-3
        ref = rectangle(lo, hi, corr, nu)
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=2e-9)
