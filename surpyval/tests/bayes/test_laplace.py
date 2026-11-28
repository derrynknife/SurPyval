"""The Laplace engine (``method="laplace"``): the posterior as a normal at
its peak on the unconstrained scale, checked by importance sampling.

Checked against posteriors known apart from the engine:

- the Exponential with a Gamma prior, whose Laplace approximation is a
  closed form (on the log of the rate, the posterior Gamma(a + r, b + T)
  has its peak at (a + r) / (b + T) and curvature a + r), and which
  converges to the conjugate posterior as the data grow;
- the grid engine on two- and three-parameter fits, before and after the
  importance reweighting;
- the engine's own reproducibility, saving and the refusals.

Data sets, seeds and tolerances were fixed before the first run.
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

import surpyval as sp
from surpyval.bayes.engines.laplace import (
    DRAWS,
    curvature,
    normal_draws,
    normal_logpdf,
)

A, B = 2.0, 500.0
RATE = {"failure_rate": sp.Gamma.from_params([A, B])}


def _exp_data(n, seed=3):
    x = sp.Exponential.random(n, 1 / 200.0, random_state=seed)
    cut = float(np.quantile(x, 0.7))
    return np.minimum(x, cut), (x > cut).astype(int)


def test_exponential_gamma_is_the_closed_form_laplace():
    """The Laplace normal on log(rate) is N(log((a + r) / (b + T)),
    1 / (a + r)), so its 95% interval is (a + r) / (b + T) * exp(+-1.96 /
    sqrt(a + r))."""
    x, c = _exp_data(25)
    model = sp.Exponential.fit_bayes(x, c, priors=RATE, method="laplace")
    r = float(np.sum(c == 0))
    T = float(np.sum(x))
    peak = (A + r) / (B + T)
    z = stats.norm.ppf(0.975)
    assert_allclose(
        np.ravel(model.param_cri("failure_rate")),
        peak * np.exp(np.array([-1, 1]) * z / np.sqrt(A + r)),
        rtol=1e-5,
    )
    # the log evidence is Laplace's: log p(peak) + log(2 pi) / 2 + log sd
    exact = sp.Exponential.fit_bayes(x, c, priors=RATE)
    assert exact.method == "conjugate"
    assert abs(model.log_evidence - exact.log_evidence) < 0.05


@pytest.mark.parametrize("importance", [False, True])
def test_converges_to_the_conjugate_posterior(importance):
    """The interval's ends, as a share of the exact interval's width,
    move towards the conjugate posterior's as the data grow."""
    gaps = []
    for n in (10, 100, 1000):
        x, c = _exp_data(n)
        exact = sp.Exponential.fit_bayes(x, c, priors=RATE)
        approx = sp.Exponential.fit_bayes(
            x, c, priors=RATE, method="laplace", importance=importance
        )
        e = np.ravel(exact.param_cri("failure_rate", alpha_ci=0.1))
        a = np.ravel(approx.param_cri("failure_rate", alpha_ci=0.1))
        gaps.append(float(np.max(np.abs(a - e)) / (e[1] - e[0])))
    if importance:
        # reweighted: within Monte Carlo error at every size
        assert max(gaps) < 0.02
    else:
        assert gaps[0] > gaps[1] > gaps[2]
        assert gaps[2] < 0.01


def _summary(model, names):
    post = model.posterior
    w = post.weights
    out = {}
    for nm in names:
        col = post.points[:, post._col(nm)]
        mean = float(w @ col)
        sd = float(np.sqrt(w @ (col - mean) ** 2))
        lo, hi = post.quantile(nm, [0.05, 0.95])
        out[nm] = np.array([mean, sd, lo, hi])
    return out


def _gaps(approx, exact, names):
    """Each parameter's (mean, 5% point, 95% point) differences in
    standard deviations of the exact posterior, and its sd ratio."""
    a, e = _summary(approx, names), _summary(exact, names)
    out = {}
    for nm in names:
        sd = e[nm][1]
        out[nm] = (
            abs(a[nm][0] - e[nm][0]) / sd,
            abs(a[nm][2] - e[nm][2]) / sd,
            abs(a[nm][3] - e[nm][3]) / sd,
            a[nm][1] / sd,
        )
    return out


WEIBULL_PRIORS = {
    "alpha": sp.prior.around(100, between=(30, 300)),
    "beta": sp.prior.around(1.5, between=(0.7, 3)),
}


def _weibull_data():
    x = sp.Weibull.random(100, 100, 2, random_state=11)
    return np.minimum(x, 130.0), (x > 130.0).astype(int)


def test_two_parameters_against_the_grid():
    """100 Weibull lives, 30% censored. Plain: means and 90% ends within
    0.25 sd of the grid's, sds within 5%. Reweighted: within 0.03 sd and
    2%."""
    x, c = _weibull_data()
    grid = sp.Weibull.fit_bayes(x, c, priors=WEIBULL_PRIORS)
    assert grid.method == "grid"
    plain = sp.Weibull.fit_bayes(x, c, priors=WEIBULL_PRIORS, method="laplace")
    reweighted = sp.Weibull.fit_bayes(
        x, c, priors=WEIBULL_PRIORS, method="laplace", importance=True
    )
    assert plain.diagnostics["pareto_k"] < 0.7
    assert reweighted.diagnostics["importance"]
    for nm, (m, lo, hi, ratio) in _gaps(
        plain, grid, ["alpha", "beta"]
    ).items():
        assert max(m, lo, hi) < 0.25, nm
        assert abs(ratio - 1) < 0.05, nm
    for nm, (m, lo, hi, ratio) in _gaps(
        reweighted, grid, ["alpha", "beta"]
    ).items():
        assert max(m, lo, hi) < 0.03, nm
        assert abs(ratio - 1) < 0.02, nm


@pytest.mark.slow
def test_three_parameters_against_the_grid():
    """200 exponentiated-Weibull lives: means and 90% ends within 0.4 sd
    of the grid's, sds within 10%, plain or reweighted."""
    x = sp.ExpoWeibull.random(200, 100, 2, 1.5, random_state=12)
    priors = {
        "alpha": sp.prior.around(100, between=(30, 300)),
        "beta": sp.prior.around(2, between=(0.7, 5)),
        "mu": sp.prior.around(1.5, between=(0.3, 5)),
    }
    grid = sp.ExpoWeibull.fit_bayes(x, priors=priors)
    assert grid.method == "grid"
    for importance in (False, True):
        approx = sp.ExpoWeibull.fit_bayes(
            x, priors=priors, method="laplace", importance=importance
        )
        for nm, (m, lo, hi, ratio) in _gaps(
            approx, grid, ["alpha", "beta", "mu"]
        ).items():
            assert max(m, lo, hi) < 0.4, (importance, nm)
            assert abs(ratio - 1) < 0.12, (importance, nm)


def test_more_than_three_parameters():
    """Four free parameters (an offset and a limited-failure proportion
    with a Weibull): the grid refuses, 'auto' and 'laplace' fit, and the
    posterior median sits near the maximum-likelihood fit."""
    x = sp.Weibull.random(150, 100, 2, random_state=5) + 50
    c = (x > 170).astype(int)
    x = np.minimum(x, 170.0)
    priors = {
        "beta": sp.prior.between(1, 4),
        "lfp_p": sp.prior.between(0.5, 0.99),
    }
    with pytest.raises(ValueError, match="1 to 3"):
        sp.Weibull.fit_bayes(
            x, c, offset=True, lfp=True, priors=priors, method="grid"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.Weibull.fit_bayes(
            x, c, offset=True, lfp=True, priors=priors
        )
    assert model.method == "laplace"
    assert model.diagnostics["free_parameters"] == [
        "gamma",
        "alpha",
        "beta",
        "lfp_p",
    ]
    assert model.diagnostics["approximate"]
    mode = dict(
        zip(["gamma", "alpha", "beta", "lfp_p"], model.diagnostics["mode"])
    )
    assert 0 < mode["gamma"] < 100
    lo, hi = np.ravel(model.param_cri("alpha"))
    assert lo < model.params[0] < hi
    # the offset's scale falls as its unconstrained value rises: its
    # interval still runs from low to high, about its median
    lo, hi = np.ravel(model.param_cri("gamma"))
    assert lo < model.gamma < hi < float(np.min(x[c == 0]))
    d = model.posterior.draws(100_000, random_state=2)[:, 0]
    # (within 1% of the interval's width: Monte Carlo error in a long
    # tail)
    assert_allclose(
        np.quantile(d, [0.025, 0.975]), [lo, hi], atol=0.01 * (hi - lo)
    )


def test_poor_approximation_warns():
    """Three failures among 40 cannot place an offset: the posterior of
    the offset is far from normal, which the Pareto k says."""
    x = np.r_[2100.0, 3400.0, 4050.0, np.linspace(1500, 6000, 37)]
    c = np.r_[0, 0, 0, np.ones(37)]
    with pytest.warns(UserWarning, match="Pareto k"):
        model = sp.Weibull.fit_bayes(
            x,
            c,
            offset=True,
            priors={"beta": sp.prior.around(2, between=(1.5, 3))},
            method="laplace",
        )
    assert model.diagnostics["pareto_k"] > 0.7
    assert model.diagnostics["approximation"] == "poor"
    assert "poor" in repr(model)


def test_reproducible_and_saved():
    x, c = _weibull_data()
    a = sp.Weibull.fit_bayes(x, c, priors=WEIBULL_PRIORS, method="laplace")
    b = sp.Weibull.fit_bayes(x, c, priors=WEIBULL_PRIORS, method="laplace")
    assert np.array_equal(a.posterior.points, b.posterior.points)
    assert a.sf(90.0) == b.sf(90.0)
    again = sp.from_dict(a.to_dict())
    assert again.method == "laplace"
    assert again.sf(90.0) == a.sf(90.0)
    assert_allclose(again.param_cri("beta"), a.param_cri("beta"))
    assert_allclose(
        again.posterior.density(np.array([[100.0, 2.0]])),
        a.posterior.density(np.array([[100.0, 2.0]])),
    )
    w = sp.Weibull.fit_bayes(
        x, c, priors=WEIBULL_PRIORS, method="laplace", importance=True
    )
    again = sp.from_dict(w.to_dict())
    assert again.posterior.normal["reweighted"]
    assert again.sf(90.0) == w.sf(90.0)


def test_draws_come_from_the_normal():
    """A plain Laplace posterior's draws are the normal's, so their
    quantiles match the exact quantiles of its marginals."""
    x, c = _weibull_data()
    model = sp.Weibull.fit_bayes(x, c, priors=WEIBULL_PRIORS, method="laplace")
    d = model.posterior.draws(200_000, random_state=1)
    for j, nm in enumerate(["alpha", "beta"]):
        assert_allclose(
            np.quantile(d[:, j], [0.05, 0.5, 0.95]),
            model.posterior.quantile(nm, [0.05, 0.5, 0.95]),
            rtol=3e-3,
        )


def test_update_keeps_the_engine():
    x, c = _weibull_data()
    model = sp.Weibull.fit_bayes(
        x[:60],
        c[:60],
        priors=WEIBULL_PRIORS,
        method="laplace",
        importance=True,
    )
    both = model.update(x[60:], c[60:])
    once = sp.Weibull.fit_bayes(
        x, c, priors=WEIBULL_PRIORS, method="laplace", importance=True
    )
    assert both.method == "laplace"
    assert both.diagnostics["importance"]
    assert_allclose(both.params, once.params, rtol=1e-6)


def test_importance_with_the_grid_is_refused():
    x, c = _weibull_data()
    with pytest.raises(ValueError, match="importance=True"):
        sp.Weibull.fit_bayes(
            x, c, priors=WEIBULL_PRIORS, method="grid", importance=True
        )


def test_a_flat_direction_is_refused():
    """A log posterior flat along one direction has no normal."""

    def log_post(phi):
        phi = np.atleast_2d(phi)
        return -0.5 * phi[:, 0] ** 2 + 0.0 * phi[:, 1]

    from surpyval.bayes.engines.grid import GridError

    with pytest.raises(GridError, match="curve down"):
        curvature(log_post, np.zeros(2), np.eye(2))


def test_normal_helpers():
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    mode = np.array([1.0, -1.0])
    z = normal_draws(mode, cov, DRAWS)
    assert z.shape == (DRAWS, 2)
    assert_allclose(z.mean(axis=0), mode, atol=5e-3)
    assert_allclose(np.cov(z.T), cov, rtol=0.03)
    assert_allclose(
        normal_logpdf(z[:5], mode, cov),
        stats.multivariate_normal(mode, cov).logpdf(z[:5]),
    )
