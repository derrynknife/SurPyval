"""Regression tests for the third docs-review bug-fix round (copulas).

- Frank's primitives are evaluated in log space: finite and accurate for
  large ``|theta|`` (the textbook form overflowed above ``theta ~ 37``), with
  a closed-form sampler and Spearman's rho. Clayton no longer collapses to 0
  in the far lower corner at large ``theta``.
- The autograd ``du``/``dv``/``pdf`` broadcast their arguments instead of
  summing over a broadcast axis.
- ``from_params`` validates the parameters and the margins.
- ``how="MLE"`` re-fits an already-fitted margin with its own configuration
  (offset, limited failure, zero inflation, fixed parameters).
- ``CopulaModel.random`` honours a tuple size; ``conditional_cdf`` checks
  ``given_dim``.
"""

import warnings
from decimal import Decimal, getcontext

import numpy as np
import pytest
from scipy.stats import kendalltau, kstest, spearmanr

import surpyval as surv
from surpyval.multivariate import Clayton, Copula, Frank, Gaussian, Gumbel

getcontext().prec = 200

MARGINS = [
    surv.Weibull.from_params([10, 2]),
    surv.Weibull.from_params([20, 3]),
]


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
    X = Frank.from_params([theta], MARGINS).random(1000, random_state=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = Frank.fit(X, margins=MARGINS)
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
    param_names = ("theta",)

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


@pytest.mark.parametrize(
    "family, params, match",
    [
        (Clayton, [2.0, 3.0], "takes 1 parameter"),
        (Clayton, [-2.0], "outside its bounds"),
        (Gumbel, [0.5], "outside its bounds"),
        (Gaussian, [1.5], "outside its bounds"),
        (Frank, [np.nan], "outside its bounds"),
    ],
)
def test_from_params_validates_parameters(family, params, match):
    with pytest.raises(ValueError, match=match):
        family.from_params(params, MARGINS)


def test_from_params_validates_margins():
    with pytest.raises(ValueError, match="needs 2 margins"):
        Clayton.from_params([2.0], MARGINS[:1])
    with pytest.raises(ValueError, match="not a fitted univariate model"):
        Clayton.from_params([2.0], [MARGINS[0], 3.0])


def test_from_params_accepts_gumbel_independence():
    model = Gumbel.from_params([1.0], MARGINS)
    assert model.kendall_tau() == 0.0


def _clayton_sample(shift=0.0):
    M = [
        surv.Weibull.from_params([10, 2]),
        surv.LogNormal.from_params([2.5, 0.5]),
    ]
    X = Clayton.from_params([2.0], M).random(300, random_state=1)
    return np.column_stack([X[:, 0] + shift, X[:, 1]])


def test_mle_keeps_an_offset_margin():
    X = _clayton_sample(shift=5.0)
    margins = [
        surv.Weibull.fit(X[:, 0], offset=True),
        surv.LogNormal.fit(X[:, 1]),
    ]
    ifm = Clayton.fit(X, margins=margins)
    mle = Clayton.fit(X, margins=margins, how="MLE")
    assert mle.margins[0].offset
    assert 0 < mle.margins[0].gamma < X[:, 0].min()
    assert mle.neg_ll() <= ifm.neg_ll()
    assert mle.k == 1 + 3 + 2


def test_mle_keeps_limited_failure_zero_inflation_and_fixed():
    X = _clayton_sample()
    c = np.zeros(X.shape, dtype=int)
    over = X[:, 0] > 12
    c[over, 0] = 1
    Xc = X.copy()
    Xc[over, 0] = 12
    lfp = [surv.Weibull.fit(Xc[:, 0], c=c[:, 0], lfp=True), surv.LogNormal]
    mle = Clayton.fit(Xc, c=c, margins=lfp, how="MLE")
    assert mle.margins[0].lfp and 0 < mle.margins[0].p < 1

    Xz = X.copy()
    Xz[:20, 0] = 0
    zi = [surv.Weibull.fit(Xz[:, 0], zi=True), surv.LogNormal]
    mle = Clayton.fit(Xz, margins=zi, how="MLE")
    assert mle.margins[0].zi and 0 < mle.margins[0].f0 < 1

    fixed = [surv.Weibull.fit(X[:, 0], fixed={"beta": 2.0}), surv.LogNormal]
    mle = Clayton.fit(X, margins=fixed, how="MLE")
    assert mle.margins[0].params[1] == 2.0
    assert mle.k == 1 + 1 + 2


def test_mle_refuses_a_non_parametric_margin():
    X = _clayton_sample()
    km = surv.KaplanMeier.fit(X[:, 0])
    with pytest.raises(ValueError, match="how='IFM'"):
        Clayton.fit(X, margins=[km, surv.LogNormal], how="MLE")


def test_random_tuple_size_and_given_dim():
    model = Clayton.from_params([2.0], MARGINS)
    draws = model.random((2, 3), random_state=0)
    assert draws.shape == (2, 3, 2)
    np.testing.assert_allclose(
        draws.reshape(6, 2), model.random(6, random_state=0)
    )
    with pytest.raises(ValueError, match="given_dim"):
        model.conditional_cdf([[10, 18]], given_dim=2)
