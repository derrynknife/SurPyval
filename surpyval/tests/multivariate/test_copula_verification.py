"""Numerical verification of the copula families (#291).

Each test pins an identity or a reference value the review checked; the
review's full tables (recovery bias and coverage, the reference grids from
vinecopulib) are in the #291 report.
"""

import warnings

import numpy as np
import pytest

from surpyval import Weibull
from surpyval.multivariate import Clayton, Copula, Frank, Gaussian, Gumbel

# Spearman's rho, 12 * int int C(u, v) du dv - 3, by adaptive quadrature
# (scipy.integrate.quad, the inner integral split at the diagonal, tolerance
# 1e-13); independent of the package's own quadrature.
SPEARMAN_REFERENCE = [
    (Clayton, 0.5, 0.2949437385539322),
    (Clayton, 2.0, 0.6822338332806566),
    (Clayton, 8.0, 0.9409181387560075),
    (Clayton, 30.0, 0.9937920719095037),
    (Gumbel, 1.2, 0.2456600516279348),
    (Gumbel, 2.0, 0.6822338332806597),
    (Gumbel, 5.0, 0.9431899253578759),
    (Gumbel, 20.0, 0.9963519447237799),
]


@pytest.mark.parametrize(
    "family, theta, expected",
    SPEARMAN_REFERENCE,
    ids=[f"{f.name}-{t}" for f, t, _ in SPEARMAN_REFERENCE],
)
def test_spearman_rho_is_the_integral_not_a_simulation(
    family, theta, expected
):
    # Clayton and Gumbel have no closed form; the default estimated it
    # from 50 000 simulated pairs, 4.8e-3 off at Clayton theta = 0.5
    # (0.2901 for 0.2949) and 4.4e-3 at Gumbel theta = 1.2.
    assert family.spearman_rho(theta) == pytest.approx(
        expected, rel=0, abs=1e-10
    )


class _NoClosedForms(Copula):
    """A Clayton copula with only its CDF: every other quantity comes from
    the base class's defaults."""

    name = "ClaytonByCDF"
    bounds = ((0, None),)
    parameter_names = ["theta"]

    def cdf(self, u, v, theta):
        return (u**-theta + v**-theta - 1) ** (-1 / theta)


@pytest.mark.parametrize("theta", [0.5, 2.0, 8.0])
def test_default_kendall_tau_is_the_integral(theta):
    # tau = 1 - 4 int int dC/du dC/dv; the default simulated 50 000 pairs
    # (Clayton theta = 0.5: 0.2009 for 0.2, error 9e-4).
    family = _NoClosedForms()
    # (the integrand has a ridge along the diagonal: 1e-10 at tau = 0.5,
    # 1.1e-8 at tau = 0.8)
    assert family.kendall_tau(theta) == pytest.approx(
        theta / (theta + 2), rel=0, abs=1e-7
    )
    assert family.spearman_rho(theta) == pytest.approx(
        Clayton.spearman_rho(theta), rel=0, abs=1e-10
    )


@pytest.mark.parametrize(
    "family, theta",
    [(Clayton, 2.0), (Gumbel, 2.0), (Frank, -5.0), (Gaussian, 0.7)],
)
def test_closed_forms_agree_with_the_integrals(family, theta):
    # Frank's Debye-function forms and the Gaussian's arcsine forms against
    # the base class's quadrature
    assert family.kendall_tau(theta) == pytest.approx(
        Copula.kendall_tau(family, theta), rel=0, abs=1e-6
    )
    assert family.spearman_rho(theta) == pytest.approx(
        Copula.spearman_rho(family, theta), rel=0, abs=1e-9
    )


# Gumbel (cdf, du, pdf) by mpmath at 50 digits, from the formula
# C = exp(-((-log u)^theta + (-log v)^theta)^(1/theta)) differentiated
# numerically by mpmath.diff.
GUMBEL_EXACT = [
    (
        0.98,
        0.98,
        100.0,
        0.97986229915196533,
        0.5034070308277594,
        1258.8618161328284,
    ),
    (
        0.999999,
        0.999999,
        50.0,
        0.99999898604052726,
        0.50697973281783765,
        12421009.921215461,
    ),
    (
        0.9999999999,
        0.9999999999,
        20.0,
        0.9999999998964735,
        0.51763246191886333,
        49175079816.254366,
    ),
    (
        0.3,
        0.6,
        2.0,
        0.27039854940488131,
        0.82973438317288735,
        0.95312149796093535,
    ),
    (
        1e-06,
        0.5,
        5.0,
        9.9999912160612538e-7,
        0.99999886728520244,
        1.6341608762024989e-5,
    ),
    (
        0.5,
        0.5,
        1.0001,
        0.2500240205701968,
        0.50001338511810982,
        1.0000295996784588,
    ),
]


@pytest.mark.parametrize("u, v, theta, cdf, du, pdf", GUMBEL_EXACT)
def test_gumbel_primitives_near_the_upper_corner(u, v, theta, cdf, du, pdf):
    # The autograd derivatives of the old CDF took (-log u)^theta, which
    # underflows to 0 near u = 1: the density was inf at (0.98, 0.98) for
    # theta = 100 and at (1 - 1e-10, 1 - 1e-10) for theta = 20, and du
    # was NaN at (1 - 1e-6, 1 - 1e-6) for theta = 100.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = [
            float(f(u, v, theta)) for f in (Gumbel.cdf, Gumbel.du, Gumbel.pdf)
        ]
    assert got == pytest.approx([cdf, du, pdf], rel=1e-9)
    assert float(Gumbel.dv(v, u, theta)) == pytest.approx(du, rel=1e-9)


@pytest.fixture(scope="module")
def clayton_sample():
    margins = [
        Weibull.from_params([10.0, 2.0]),
        Weibull.from_params([20.0, 3.0]),
    ]
    X = Clayton.from_params([2.0], margins).random(100, random_state=0)
    return X, margins


def _bad_series(X):
    nan = X.copy()
    nan[3, 0] = np.nan
    c = np.zeros(X.shape, int)
    c[0] = 2
    t = np.zeros(X.shape + (2,))
    t[..., 1] = np.inf
    t[0, 1, 0] = X[0, 1] + 1.0
    return {
        "NaN": ({"x": nan}, "Series 0: .*NaN"),
        "negative n": (
            {"x": X, "n": np.r_[-5, np.ones(99)]},
            "Series 0: count array can't be 0 or less",
        ),
        "fractional n": (
            {"x": X, "n": np.r_[0.5, np.ones(99)]},
            "Series 0: Count array 'n' must contain integer values",
        ),
        "xl > xr": (
            {"x": X, "c": c, "xl": X + 1.0, "xr": X - 1.0},
            "Series 0: All left intervals must be less than or equal",
        ),
        "below tl": ({"x": X, "t": t}, "Series 1: All left truncated"),
    }


@pytest.mark.parametrize(
    "case", ["NaN", "negative n", "fractional n", "xl > xr", "below tl"]
)
def test_data_checked_with_fitted_margins(clayton_sample, case):
    # A margin fitted by the copula fit checked its series, but a margin
    # passed already fitted did not: a NaN gave theta = 0.01 (the start)
    # with a NaN likelihood, a count of -5 a negative weight (theta 2.25
    # for 2.34), and xl > xr or a value below its truncation were used.
    X, margins = clayton_sample
    kwargs, match = _bad_series(X)[case]
    with pytest.raises(ValueError, match=match):
        Clayton.fit(margins=margins, **kwargs)


# -- #541: the elliptical copulas near rho = +-1 -----------------------------
# rho was clipped to +-0.9999 inside every formula, so from_params(0.99995)
# silently evaluated another copula: a density of 81.11 at (0.3, 0.3001)
# for the true 114.68.
_NEAR_ONE = [0.99995, -0.99995, 1 - 1e-8, -(1 - 1e-12)]
_POINTS = [(0.3, 0.3001), (0.5, 0.5), (0.9, 0.2), (0.02, 0.0201), (0.7, 0.3)]


def _gaussian_density(u, v, rho):
    """The bivariate normal density over its margins' in long double, from
    the textbook form."""
    from scipy.special import ndtri

    a = np.longdouble(ndtri(u))
    b = np.longdouble(ndtri(v))
    r = np.longdouble(rho)
    d = 1 - r * r
    quad = (r * r * (a * a + b * b) - 2 * r * a * b) / (2 * d)
    return float(np.exp(-quad) / np.sqrt(d))


def _t_density(u, v, rho, nu):
    """The bivariate t density over its margins', the quadratic form in
    long double."""
    from scipy.special import gammaln, stdtrit

    x = np.longdouble(stdtrit(nu, u))
    y = np.longdouble(stdtrit(nu, v))
    r = np.longdouble(rho)
    d = 1 - r * r
    q = (x * x + y * y - 2 * r * x * y) / (nu * d)
    log_c = (
        gammaln((nu + 2) / 2)
        + gammaln(nu / 2)
        - 2 * gammaln((nu + 1) / 2)
        - np.log(np.sqrt(d))
        - (nu + 2) / 2 * np.log1p(q)
        + (nu + 1) / 2 * (np.log1p(x * x / nu) + np.log1p(y * y / nu))
    )
    return float(np.exp(log_c))


def _assert_log_close(value, reference):
    if reference < 1e-300:  # underflows in double precision
        assert value < 1e-290, (value, reference)
        return
    log_ref = np.log(reference)
    tol = 1e-11 * max(1.0, abs(log_ref))
    assert abs(np.log(value) - log_ref) <= tol, (value, reference)


@pytest.mark.parametrize("rho", _NEAR_ONE)
def test_541_elliptical_density_near_rho_one(rho):
    from surpyval.multivariate import StudentT

    margins = [Weibull.from_params([10, 2]), Weibull.from_params([20, 3])]
    gauss = Gaussian.from_params([rho], margins)
    t = StudentT.from_params([rho, 4.0], margins)
    for u, v in _POINTS:
        if rho < 0:
            v = 1 - v
        # On the log scale the likelihood uses: far from the diagonal the
        # density is as small as 1e-42, its exponent good to 1e-12
        _assert_log_close(
            gauss.copula.pdf(u, v, rho), _gaussian_density(u, v, rho)
        )
        _assert_log_close(
            t.copula.pdf(u, v, rho, 4.0), _t_density(u, v, rho, 4.0)
        )


@pytest.mark.parametrize("rho", _NEAR_ONE)
def test_541_gaussian_cdf_and_h_function_near_rho_one(rho):
    from scipy.special import ndtr, ndtri

    # C(1/2, 1/2) = 1/4 + arcsin(rho) / (2 pi) exactly; scipy's check of
    # the covariance refused rho within 1e-10 of 1
    np.testing.assert_allclose(
        Gaussian.cdf(0.5, 0.5, rho),
        0.25 + np.arcsin(rho) / (2 * np.pi),
        rtol=1e-13,
        atol=1e-15,
    )
    # Within 1e-8 of +-1, and off the diagonal, the CDF is a Frechet
    # bound to double precision
    if 1 - abs(rho) <= 1e-8:
        v, expected = (0.31, 0.3) if rho > 0 else (0.69, 0.0)
        assert Gaussian.cdf(0.3, v, rho) == pytest.approx(expected, abs=1e-15)
    for u, v in _POINTS:
        a = np.longdouble(ndtri(u))
        b = np.longdouble(ndtri(v))
        r = np.longdouble(rho)
        z = float((b - r * a) / np.sqrt(1 - r * r))
        np.testing.assert_allclose(
            Gaussian.du(u, v, rho), ndtr(z), rtol=1e-11, atol=1e-300
        )


def test_541_samples_near_rho_one_have_the_right_dependence():
    from scipy.stats import kendalltau

    from surpyval.multivariate import StudentT

    margins = [Weibull.from_params([10, 2]), Weibull.from_params([20, 3])]
    expected = 2 / np.pi * np.arcsin(0.99995)  # 0.99363; 0.99100 at 0.9999
    for model in (
        Gaussian.from_params([0.99995], margins),
        StudentT.from_params([0.99995, 3.0], margins),
    ):
        assert model.kendall_tau() == pytest.approx(expected, rel=1e-14)
        X = model.random(5000, random_state=0)
        tau = kendalltau(X[:, 0], X[:, 1]).statistic
        assert tau == pytest.approx(expected, abs=5e-4)


@pytest.mark.parametrize("rho", [1.0, -1.0, 1.5])
def test_541_rho_of_one_is_not_an_elliptical_copula(rho):
    from surpyval.multivariate import StudentT

    margins = [Weibull.from_params([10, 2]), Weibull.from_params([20, 3])]
    with pytest.raises(ValueError, match="rho = .* is outside its bounds"):
        Gaussian.from_params([rho], margins)
    with pytest.raises(ValueError, match="rho = .* is outside its bounds"):
        StudentT.from_params([rho, 4.0], margins)
