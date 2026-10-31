"""The closed-form (conjugate) cases, exactly, and the grid against them.

- The conjugate path's posterior parameters are the textbook formulas,
  checked in exact arithmetic (``fractions.Fraction``).
- The grid, forced with ``method="grid"``, matches the closed form's
  posterior mean, quantiles and predictive ``sf`` to 1e-4 (relative),
  and its evidence.
- The Weibull with a known shape is conjugate too (a Gamma prior on
  ``alpha ** -beta``): the grid matches that closed form, which no
  engine uses (a test case only).
"""

from fractions import Fraction

import numpy as np
import pytest
from scipy import integrate, stats

import surpyval as sp
from surpyval.bayes.prior import Prior

A, B = 2.0, 500.0
X = np.array([100.0, 250.0, 400.0, 520.0, 700.0])
C = np.array([0, 0, 1, 0, 1])
TL = np.array([0.0, 50.0, 0.0, 120.0, 0.0])
RATE = {"failure_rate": sp.Gamma.from_params([A, B])}


def _posterior_exact():
    r = sum(Fraction(1) for c in C if c == 0)
    T = sum(Fraction(x) - Fraction(t) for x, t in zip(X, TL))
    return Fraction(A) + r, Fraction(B) + T


def test_exponential_gamma_is_the_textbook_posterior():
    model = sp.Exponential.fit_bayes(X, C, tl=TL, priors=RATE)
    assert model.method == "conjugate"
    a, b = _posterior_exact()
    assert model.posterior.exact["params"] == [float(a), float(b)]
    post = stats.gamma(float(a), scale=1 / float(b))
    np.testing.assert_allclose(
        model.param_cri("failure_rate"), post.ppf([0.025, 0.975]), rtol=1e-14
    )
    # Predictive: Lomax, S(t) = (b / (b + t)) ** a
    t = np.array([10.0, 100.0, 1000.0])
    np.testing.assert_allclose(
        model.sf(t), (float(b) / (float(b) + t)) ** float(a), rtol=1e-14
    )
    assert model.mean() == pytest.approx(float(b / (a - 1)), rel=1e-14)
    # B10 of the population: qf(0.1 | rate) = -log(0.9) / rate, which
    # falls with the rate, so its interval is the rate's flipped
    np.testing.assert_allclose(
        model.quantile_cri(0.1),
        -np.log(0.9) / post.ppf([0.975, 0.025]),
        rtol=1e-12,
    )


def test_exponential_gamma_evidence_is_the_integral():
    model = sp.Exponential.fit_bayes(X, C, tl=TL, priors=RATE)
    a0, b0 = A, B

    def integrand(lam):
        loglik = np.sum(np.where(C == 0, np.log(lam), 0.0)) - lam * np.sum(
            X - TL
        )
        return np.exp(loglik) * stats.gamma(a0, scale=1 / b0).pdf(lam)

    # (relative accuracy only: the integrand is about 1e-11)
    z, _ = integrate.quad(
        integrand, 0, np.inf, limit=500, epsabs=0, epsrel=1e-12
    )
    assert model.log_evidence == pytest.approx(np.log(z), rel=1e-8)


def test_grid_matches_exponential_gamma():
    exact = sp.Exponential.fit_bayes(X, C, tl=TL, priors=RATE)
    grid = sp.Exponential.fit_bayes(X, C, tl=TL, priors=RATE, method="grid")
    assert grid.method == "grid"
    q = [0.025, 0.25, 0.5, 0.75, 0.975]
    np.testing.assert_allclose(
        grid.posterior.quantile("failure_rate", q),
        exact.posterior.quantile("failure_rate", q),
        rtol=1e-4,
    )
    assert grid.posterior.mean("failure_rate") == pytest.approx(
        exact.posterior.mean("failure_rate"), rel=1e-4
    )
    t = np.array([10.0, 100.0, 500.0, 2000.0])
    np.testing.assert_allclose(grid.sf(t), exact.sf(t), rtol=1e-4)
    assert grid.log_evidence == pytest.approx(exact.log_evidence, rel=1e-4)


def test_interval_data_are_not_conjugate():
    model = sp.Exponential.fit_bayes(
        [[100, 200], 300.0], c=[2, 0], priors=RATE
    )
    assert model.method == "grid"
    with pytest.raises(ValueError, match="No closed form"):
        sp.Exponential.fit_bayes(
            [[100, 200], 300.0], c=[2, 0], priors=RATE, method="conjugate"
        )


Y = np.array([4.1, 5.3, 3.8, 6.0, 5.1, 4.7])
SIGMA = 0.8
M0, S0 = 3.0, 2.0


def test_normal_normal_is_the_textbook_posterior():
    model = sp.Normal.fit_bayes(
        Y,
        priors={
            "mu": sp.Normal.from_params([M0, S0]),
            "sigma": sp.prior.fixed(SIGMA),
        },
    )
    assert model.method == "conjugate"
    n = Fraction(len(Y))
    prec = 1 / Fraction(S0) ** 2 + n / Fraction(SIGMA) ** 2
    mean = (
        Fraction(M0) / Fraction(S0) ** 2
        + sum(Fraction(y) for y in Y) / Fraction(SIGMA) ** 2
    ) / prec
    m, s = model.posterior.exact["params"]
    assert m == pytest.approx(float(mean), rel=1e-15)
    assert s == pytest.approx(float(1 / prec) ** 0.5, rel=1e-15)
    # predictive: Normal(mean, sqrt(sigma^2 + s^2))
    pred = stats.norm(float(mean), np.hypot(SIGMA, s))
    np.testing.assert_allclose(model.sf([4, 5, 6]), pred.sf([4, 5, 6]))
    np.testing.assert_allclose(model.qf([0.1, 0.9]), pred.ppf([0.1, 0.9]))
    # evidence: the data's marginal, a multivariate normal
    cov = SIGMA**2 * np.eye(len(Y)) + S0**2
    z = stats.multivariate_normal(np.full(len(Y), M0), cov).logpdf(Y)
    assert model.log_evidence == pytest.approx(z, rel=1e-12)


def test_grid_matches_normal_normal():
    pri = {
        "mu": sp.Normal.from_params([M0, S0]),
        "sigma": sp.prior.fixed(SIGMA),
    }
    exact = sp.Normal.fit_bayes(Y, priors=pri)
    grid = sp.Normal.fit_bayes(Y, priors=pri, method="grid")
    q = [0.025, 0.5, 0.975]
    np.testing.assert_allclose(
        grid.posterior.quantile("mu", q),
        exact.posterior.quantile("mu", q),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        grid.sf([4.0, 5.0, 6.0]), exact.sf([4.0, 5.0, 6.0]), rtol=1e-4
    )
    assert grid.log_evidence == pytest.approx(exact.log_evidence, rel=1e-4)


def test_bernoulli_beta():
    outcomes = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
    model = sp.Bernoulli.fit_bayes(
        outcomes, priors={"p": sp.Beta.from_params([1.0, 1.0])}
    )
    assert model.method == "conjugate"
    assert model.posterior.exact["params"] == [9.0, 3.0]
    # predictive: Bernoulli(E p) = Bernoulli(9 / 12)
    assert float(model.sf(0)) == pytest.approx(0.75, rel=1e-15)
    np.testing.assert_allclose(
        model.param_cri("p"), stats.beta(9, 3).ppf([0.025, 0.975])
    )
    # evidence: B(9, 3) / B(1, 1)
    from scipy.special import betaln

    assert model.log_evidence == pytest.approx(betaln(9, 3), rel=1e-14)


def test_bernoulli_needs_a_beta_prior():
    with pytest.raises(ValueError, match="Beta prior"):
        sp.Bernoulli.fit_bayes([1, 0, 1])


class _GammaOnAlphaPower(Prior):
    """A Gamma(a, b) prior on lambda = alpha ** -beta, beta known, as a
    density on log alpha: lambda = exp(-beta phi), |d lambda / d phi| =
    beta lambda."""

    def __init__(self, a, b, beta):
        self.a, self.b, self.beta = a, b, beta
        self.said = "Gamma on alpha ** -beta"

    def log_density_phi(self, phi, t):
        lam = np.exp(-self.beta * phi)
        return stats.gamma(self.a, scale=1 / self.b).logpdf(lam) + np.log(
            self.beta * lam
        )

    def centre_phi(self, t):
        return float(-np.log(self.a / self.b) / self.beta)


def test_grid_matches_fixed_shape_weibull_gamma():
    beta = 1.7
    a, b = 3.0, 2e4
    x = np.array([300.0, 520.0, 610.0, 900.0, 1200.0])
    c = np.array([0, 0, 1, 0, 1])
    model = sp.Weibull.fit_bayes(
        x,
        c,
        priors={
            "alpha": _GammaOnAlphaPower(a, b, beta),
            "beta": sp.prior.fixed(beta),
        },
    )
    assert model.method == "grid"
    # lambda | data ~ Gamma(a + r, b + sum x ** beta)
    post = stats.gamma(a + 3, scale=1 / (b + np.sum(x**beta)))
    lam_q = post.ppf([0.975, 0.5, 0.025])
    np.testing.assert_allclose(
        model.posterior.quantile("alpha", [0.025, 0.5, 0.975]),
        lam_q ** (-1 / beta),
        rtol=1e-4,
    )
    # predictive: S(t) = ((b + T) / (b + T + t ** beta)) ** (a + r)
    T = np.sum(x**beta)
    t = np.array([200.0, 800.0, 2000.0])
    np.testing.assert_allclose(
        model.sf(t), ((b + T) / (b + T + t**beta)) ** (a + 3), rtol=1e-4
    )


def test_lomax_predictive_mean_is_infinite_when_the_shape_is_at_most_1():
    """audit B25: the Exponential-Gamma predictive is a Lomax with mean
    ``b / (a - 1)``, infinite where the posterior shape ``a`` is at most
    1 (a Gamma(0.5, 100) prior and no failures: a = 0.5, b = 700)."""
    model = sp.Exponential.fit_bayes(
        [100.0, 200.0, 300.0],
        c=[1, 1, 1],
        priors={"failure_rate": sp.Gamma.from_params([0.5, 100.0])},
    )
    assert model.method == "conjugate"
    assert model.posterior.exact["params"] == [0.5, 700.0]
    assert model.mean() == np.inf


def test_lomax_predictive_variance_is_the_closed_form():
    """The Exponential-Gamma predictive is a Lomax of shape ``a`` and
    scale ``b`` (the posterior's): its variance is ``b**2 a / ((a - 1)**2
    (a - 2))``, checked in exact arithmetic, and against integrating the
    predictive survival function (``E[T^2] = 2 int t S(t) dt``)."""
    model = sp.Exponential.fit_bayes(X, C, tl=TL, priors=RATE)
    a, b = _posterior_exact()
    exact = b**2 * a / ((a - 1) ** 2 * (a - 2))
    assert model.var() == pytest.approx(float(exact), rel=1e-14)
    af, bf = float(a), float(b)
    second, _ = integrate.quad(
        lambda t: 2 * t * (bf / (bf + t)) ** af, 0, np.inf, epsabs=0
    )
    assert model.var() == pytest.approx(second - model.mean() ** 2, rel=1e-8)


def test_lomax_predictive_variance_is_infinite_when_the_shape_is_at_most_2():
    """With a posterior shape in (1, 2] the Lomax mean is finite and its
    variance infinite (a Gamma(0.5, 100) prior and one failure: a = 1.5);
    at most 1 both are."""
    prior_ = {"failure_rate": sp.Gamma.from_params([0.5, 100.0])}
    one = sp.Exponential.fit_bayes(
        [100.0, 200.0, 300.0], c=[0, 1, 1], priors=prior_
    )
    assert one.posterior.exact["params"] == [1.5, 700.0]
    assert one.mean() == pytest.approx(700.0 / 0.5, rel=1e-14)
    assert one.var() == np.inf
    none = sp.Exponential.fit_bayes(
        [100.0, 200.0, 300.0], c=[1, 1, 1], priors=prior_
    )
    assert none.mean() == np.inf and none.var() == np.inf


def _normal_normal():
    model = sp.Normal.fit_bayes(
        Y,
        priors={
            "mu": sp.Normal.from_params([M0, S0]),
            "sigma": sp.prior.fixed(SIGMA),
        },
    )
    n = Fraction(len(Y))
    prec = 1 / Fraction(S0) ** 2 + n / Fraction(SIGMA) ** 2
    mean = (
        Fraction(M0) / Fraction(S0) ** 2
        + sum(Fraction(y) for y in Y) / Fraction(SIGMA) ** 2
    ) / prec
    return model, mean, 1 / prec


def test_normal_normal_predictive_mean_is_the_posterior_mean():
    """audit B25: the Normal-normal predictive is Normal with the
    posterior mean of mu as its mean."""
    model, mean, _ = _normal_normal()
    assert model.mean() == pytest.approx(float(mean), rel=1e-14)


def test_normal_normal_predictive_variance_is_exact():
    """audit B25: the predictive variance is ``sigma**2 + s**2``, with
    ``s**2`` the posterior variance of mu (exact, as ``mean()`` is)."""
    model, _, post_var = _normal_normal()
    assert model.var() == pytest.approx(
        float(Fraction(SIGMA) ** 2 + post_var), rel=1e-12
    )
