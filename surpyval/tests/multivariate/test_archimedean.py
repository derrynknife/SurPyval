import warnings
from decimal import Decimal, getcontext

import numpy as np
import pytest
from scipy.stats import kendalltau, kstest, spearmanr

import surpyval as surv
from surpyval.multivariate import (
    Clayton,
    Copula,
    Frank,
    Gaussian,
    Gumbel,
    Independence,
)
from surpyval.tests._helpers import WEIBULL_MARGINS

MARGINS = [
    surv.Weibull.from_params([10, 2]),
    surv.LogNormal.from_params([3, 0.4]),
]


def test_independence_cdf_is_product_of_margins():
    ind = Independence.from_params([], margins=MARGINS)
    x = np.array([[10.0, 18.0], [5.0, 25.0], [15.0, 12.0]])
    prod = MARGINS[0].ff(x[:, 0]) * MARGINS[1].ff(x[:, 1])
    assert np.allclose(ind.cdf(x), prod)
    assert np.allclose(
        ind.pdf(x), MARGINS[0].df(x[:, 0]) * MARGINS[1].df(x[:, 1])
    )


def test_joint_sf_definition():
    m = Clayton.from_params(2.0, margins=MARGINS)
    x = np.array([[10.0, 18.0], [7.0, 20.0]])
    u = MARGINS[0].ff(x[:, 0])
    v = MARGINS[1].ff(x[:, 1])
    expected = 1.0 - u - v + m.cdf(x)
    assert np.allclose(m.sf(x), expected)


def test_cdf_bounds_and_monotonicity():
    m = Gumbel.from_params(2.5, margins=MARGINS)
    x = np.array([[1.0, 1.0], [10.0, 18.0], [1e3, 1e3]])
    c = m.cdf(x)
    assert np.all(c >= 0) and np.all(c <= 1)
    assert c[0] < c[1] < c[2]


def _recover(cop, true_p, seed):
    m = cop.from_params(true_p, margins=MARGINS)
    data = m.random(4000, random_state=seed)
    fit = cop.fit(
        [data[:, 0], data[:, 1]],
        margins=[surv.Weibull, surv.LogNormal],
        how="IFM",
    )
    return fit.params[0]


def test_ifm_recovers_clayton():
    assert np.isclose(_recover(Clayton, 2.5, 1), 2.5, rtol=0.15)


def test_ifm_recovers_gumbel():
    assert np.isclose(_recover(Gumbel, 2.0, 2), 2.0, rtol=0.15)


def test_ifm_recovers_frank():
    assert np.isclose(_recover(Frank, 5.0, 3), 5.0, rtol=0.15)


def test_ifm_recovers_gaussian():
    assert np.isclose(_recover(Gaussian, 0.6, 4), 0.6, rtol=0.15)


def test_mle_joint_recovers_clayton_and_margins():
    m = Clayton.from_params(2.5, margins=MARGINS)
    data = m.random(3000, random_state=5)
    fit = Clayton.fit(
        [data[:, 0], data[:, 1]],
        margins=[surv.Weibull, surv.LogNormal],
        how="MLE",
    )
    assert np.isclose(fit.params[0], 2.5, rtol=0.2)
    assert np.allclose(fit.margins[0].params, [10, 2], rtol=0.15)


def test_from_params_roundtrips_through_to_dict():
    m = Clayton.from_params(2.0, margins=MARGINS)
    d = m.to_dict()
    assert d["copula"] == "Clayton"
    assert np.isclose(d["params"][0], 2.0)
    assert len(d["margins"]) == 2


# ---------------------------------------------------------------------------
# Frank's primitives are evaluated in log space: finite and
# accurate for large ``|theta|`` (the textbook form overflowed
# above ``theta ~ 37``), with a closed-form sampler and
# Spearman's rho. Clayton does not collapse to 0 in the far
# lower corner at large ``theta``. The autograd
# ``du``/``dv``/``pdf`` broadcast their arguments.
# ---------------------------------------------------------------------------


getcontext().prec = 200


def _frank_cdf_exact(u, v, theta):
    u, v, t = Decimal(u), Decimal(v), Decimal(theta)
    g = lambda s: (-t * s).exp() - 1  # noqa: E731
    return float(-1 / t * (1 + g(u) * g(v) / g(Decimal(1))).ln())


def _frank_pdf_exact(u, v, theta):
    u, v, t = Decimal(u), Decimal(v), Decimal(theta)
    g = lambda s: (-t * s).exp() - 1  # noqa: E731
    one = g(Decimal(1))
    return float(-t * one * (-t * (u + v)).exp() / (one + g(u) * g(v)) ** 2)


@pytest.mark.parametrize("theta", [38.0, 60.0, 300.0, -20.0, -60.0])
@pytest.mark.parametrize(
    "u, v", [(0.999999, 0.999999), (0.999, 0.999), (1e-4, 1e-4), (0.3, 0.7)]
)
def test_frank_primitives_accurate_for_large_theta(theta, u, v):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c = float(Frank.cdf(u, v, theta))
        p = float(Frank.pdf(u, v, theta))
        h = float(Frank.du(u, v, theta))
    assert c == pytest.approx(_frank_cdf_exact(u, v, theta), rel=1e-10, abs=0)
    exact_pdf = _frank_pdf_exact(u, v, theta)
    if exact_pdf > 1e-300:
        assert p == pytest.approx(exact_pdf, rel=1e-10, abs=0)
    assert 0.0 <= h <= 1.0


@pytest.mark.parametrize("theta", [38.3, 60.0])
def test_frank_fit_at_strong_dependence_is_finite(theta):
    X = Frank.from_params([theta], WEIBULL_MARGINS).random(
        1000, random_state=1
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = Frank.fit(X, margins=WEIBULL_MARGINS)
        ll = model.log_likelihood
    assert np.isfinite(ll)
    assert model.params[0] == pytest.approx(theta, rel=0.1)


@pytest.mark.parametrize("theta", [45.0, 60.0, -45.0])
def test_frank_sampler_at_strong_dependence(theta):
    u, v = Frank.sample_uv(20000, [theta], random_state=3)
    assert kstest(v, "uniform").pvalue > 0.01
    assert kendalltau(u, v).statistic == pytest.approx(
        Frank.kendall_tau(theta), abs=0.01
    )


@pytest.mark.parametrize("theta", [-8.0, 2.0, 10.0])
def test_frank_spearman_closed_form_matches_simulation(theta):
    u, v = Frank.sample_uv(50000, [theta], random_state=0)
    assert Frank.spearman_rho(theta) == pytest.approx(
        spearmanr(u, v).statistic, abs=0.01
    )


def test_frank_dependence_measures_near_independence():
    # tau ~ theta / 9 and rho ~ theta / 6 without the closed forms'
    # cancellation
    assert Frank.kendall_tau(1e-6) == pytest.approx(1e-6 / 9, rel=1e-9)
    assert Frank.spearman_rho(-1e-6) == pytest.approx(-1e-6 / 6, rel=1e-9)


def test_frank_theta_zero_is_independence():
    u = np.array([0.2, 0.5])
    v = np.array([0.7, 0.4])
    np.testing.assert_allclose(Frank.cdf(u, v, 0.0), u * v)
    np.testing.assert_allclose(Frank.du(u, v, 0.0), v)
    np.testing.assert_allclose(Frank.pdf(u, v, 0.0), 1.0)


def _clayton_cdf_exact(u, v, theta):
    u, v, t = Decimal(u), Decimal(v), Decimal(theta)
    base = (-t * u.ln()).exp() + (-t * v.ln()).exp() - 1
    return float((-base.ln() / t).exp())


@pytest.mark.parametrize("theta", [31.0, 100.0, 1000.0])
def test_clayton_far_lower_corner_at_large_theta(theta):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c = float(Clayton.cdf(1e-10, 1e-10, theta))
        d = float(Clayton.pdf(1e-10, 2e-10, theta))
    exact = _clayton_cdf_exact(1e-10, 1e-10, theta)
    assert c == pytest.approx(exact, rel=1e-12, abs=0)
    assert np.isfinite(d) and d > 0


class _AMH(Copula):
    name = "AMH"
    bounds = ((-1, 1),)
    parameter_names = ["theta"]

    def cdf(self, u, v, theta):
        return u * v / (1 - theta * (1 - u) * (1 - v))


@pytest.mark.parametrize(
    "family, theta", [(Gumbel, 2.0), (_AMH(), 0.5)], ids=["Gumbel", "AMH"]
)
def test_autograd_derivatives_broadcast(family, theta):
    vs = np.array([0.2, 0.5, 0.9])
    per_point = [
        [
            float(np.ravel(f(np.array([0.3]), np.array([v]), theta))[0])
            for v in vs
        ]
        for f in (family.du, family.dv, family.pdf)
    ]
    np.testing.assert_allclose(family.du(0.3, vs, theta), per_point[0])
    np.testing.assert_allclose(family.dv(0.3, vs, theta), per_point[1])
    np.testing.assert_allclose(family.pdf(0.3, vs, theta), per_point[2])
    grid = family.pdf(np.array([[0.3], [0.6]]), vs, theta)
    assert grid.shape == (2, 3)
