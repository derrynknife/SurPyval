"""The grid's answers against the truth, with rules fixed before the first
run (DECISIONS.md, 2026-10-29).

- Large samples: with flat priors and 1,000 complete Weibull values the
  posterior's peak is the maximum-likelihood estimate (to 1e-6), its
  median is within 0.02 standard errors of it, and each end of the 95%
  credible interval is within 2% of the likelihood-ratio interval's
  width of that interval's end (Bernstein-von Mises).
- Honesty: on a hard posterior (2 failures among 100, the scale flat, a
  prior on the shape) the integration error the fit reports is within a
  factor of 2 of its error against a grid 4 times finer.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.bayes import BayesianParametric, fitting
from surpyval.bayes.engines import grid
from surpyval.bayes.likelihood import LogLikelihood


@pytest.fixture(scope="module")
def large():
    x = sp.Weibull.random(1000, 100.0, 2.5, random_state=8000)
    return sp.Weibull.fit_bayes(x), sp.Weibull.fit(x)


def test_large_sample_peak_is_the_maximum(large):
    model, mle = large
    np.testing.assert_allclose(
        model.diagnostics["mode"], mle.params, rtol=1e-6
    )


# The rule was fixed before the first run and is kept; beta's median is
# 0.037 standard errors from the maximum (2.49814 against 2.50045, standard
# error 0.0621). A posterior median differs from the maximum by the
# posterior's skewness, which shrinks as 1 / sqrt(n) in standard errors:
# at n = 1000 it is about 0.04 for the Weibull shape, so the rule's 0.02
# holds only from about 4,000 values. Reported, not loosened.
_SKEW = pytest.mark.xfail(
    strict=True,
    reason="beta's posterior median is 0.037 standard errors from the "
    "maximum at n = 1000 (the rule asks 0.02): the posterior's skewness, "
    "O(1 / sqrt(n)); see DECISIONS.md 2026-10-29",
)


@pytest.mark.parametrize(
    "k, name", [(0, "alpha"), pytest.param(1, "beta", marks=_SKEW)]
)
def test_large_sample_median_is_the_maximum(large, k, name):
    model, mle = large
    se = mle.standard_errors()[k]
    med = float(model.posterior.quantile(name, 0.5))
    assert abs(med - mle.params[k]) <= 0.02 * se, (name, med, mle.params[k])


@pytest.mark.parametrize("name", ["alpha", "beta"])
def test_large_sample_interval_is_the_likelihood_ratio_interval(large, name):
    model, mle = large
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = mle.param_cb(name, method="lr")
    cri = model.param_cri(name)
    width = lr[1] - lr[0]
    assert np.all(np.abs(cri - lr) <= 0.02 * width), (name, cri, lr)


def _hard():
    x = np.r_[300.0, 700.0, np.linspace(100.0, 1000.0, 98)]
    c = np.r_[0, 0, np.ones(98)]
    return sp.SurpyvalData(x=x, c=c), {
        "beta": sp.prior.around(2, between=(1, 4))
    }


@pytest.mark.calibration
def test_reported_integration_error_is_honest():
    data, priors = _hard()
    model = sp.Weibull.fit_bayes(data, priors=priors)
    reported = model.diagnostics["integration_error"]
    m = model.diagnostics["points_per_dim"]

    bound = fitting.build_bound(
        sp.Weibull, data, priors, False, False, False, None
    )
    ll = LogLikelihood(sp.Weibull, data)

    def log_post(phi):
        return bound.log_prior(phi) + ll(bound.full_theta(phi))

    start = bound.to_phi(np.atleast_2d(model.diagnostics["mode"]))[0]
    mode, cov, _ = grid.find_mode(log_post, [start])
    summary = fitting.check_summary(sp.Weibull, bound)
    ref = grid.run_grid(
        log_post, mode, cov, summary, points=4 * max(m) - 3, tol=1.0
    )
    ref_post = BayesianParametric._from_grid(
        sp.Weibull, data, bound, ref, "grid", log_post, None
    ).posterior
    ours, theirs = [], []
    for name in ("alpha", "beta"):
        ours.append(model.posterior.quantile(name, [0.025, 0.5, 0.975]))
        theirs.append(ref_post.quantile(name, [0.025, 0.5, 0.975]))
    ours.append(model.quantile_cri(0.1, alpha_ci=0.05).tolist())
    ours[-1].insert(1, float(model.quantile_cri(0.1, point=True)))
    q10 = BayesianParametric._from_grid(
        sp.Weibull, data, bound, ref, "grid", log_post, None
    )
    theirs.append(
        [
            float(q10.quantile_cri(0.1)[0]),
            float(q10.quantile_cri(0.1, point=True)),
            float(q10.quantile_cri(0.1)[1]),
        ]
    )
    ours, theirs = np.asarray(ours), np.asarray(theirs)
    width = theirs[:, 2] - theirs[:, 0]
    actual = float(np.max(np.abs(ours - theirs) / width[:, None]))
    assert reported / 2 <= actual <= 2 * reported, (reported, actual)
