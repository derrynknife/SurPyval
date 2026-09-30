"""The Wald band on sf / ff follows the function's direction (#477).

It was a Wald bound on the logit of sf for every family, whose standard
error grows in the tails faster than the logit itself, so on small samples
the lower bound on F fell back towards 0: 0.39 at t = 5, 0.00004 at t = 10
on the issue's interval data. It is now a Wald bound on the family's
straight-line (probability-plot) scale -- log(-log sf) for the Weibull,
Exponential, Rayleigh and Gumbel, the normal quantile for the Normal and
LogNormal, the logit for the Logistic and LogLogistic -- where the band is
the envelope of the Wald ellipsoid's lines and is monotone whenever the
shape's Wald interval excludes 0. Large-sample bounds are essentially
unchanged.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.univariate.parametric.parametric import z

ISSUE_X = [[1, 2], [2, 3], [3, 5], 4, [4, 6], [5, 8]]
ISSUE_T = np.array([3, 5, 6, 8, 10])


def _fit(dist, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dist.fit(*args, **kwargs)


def _monotone(values):
    values = values[np.isfinite(values)]
    return bool(np.all(np.diff(values) >= -1e-10))


def test_the_issues_band_is_monotone():
    model = _fit(sp.Weibull, ISSUE_X)
    band = model.cb(ISSUE_T, on="ff")
    # Before: lower [0.0895, 0.3874, 0.3907, 0.0488, 0.00004]
    np.testing.assert_allclose(
        band[:, 0], [0.0987, 0.4711, 0.6027, 0.7476, 0.8258], atol=1e-4
    )
    assert _monotone(band[:, 0]) and _monotone(band[:, 1])
    # close to the likelihood-ratio band [0.076, 0.442, 0.599, 0.799, 0.903]
    lr = model.cb(ISSUE_T, on="ff", method="lr")
    assert np.max(np.abs(band[:, 0] - lr[:, 0])) < 0.1
    sf = model.cb(ISSUE_T, on="sf")
    assert _monotone(-sf[:, 0]) and _monotone(-sf[:, 1])


# (family, index of the parameter whose Wald interval must exclude 0 for
# the band to be monotone -- the slope of the straight line -- or None)
FAMILIES = [
    (sp.Weibull, 1),
    (sp.LogNormal, 1),
    (sp.LogLogistic, 1),
    (sp.Exponential, None),
    (sp.Rayleigh, None),
    (sp.Normal, 1),
    (sp.Gumbel, 1),
]


@pytest.mark.parametrize(
    "dist, slope", FAMILIES, ids=[d.name for d, _ in FAMILIES]
)
def test_small_sample_bands_are_monotone(dist, slope):
    # Three to eight values, 30% censored, on a wide grid of times. With
    # the logit band all 39 Weibull, LogNormal, Normal and Gumbel fits of
    # such a sweep turned back somewhere (14 of 40 Exponentials).
    rng = np.random.default_rng(0)
    location = dist.name in ("Normal", "Gumbel")
    fits = 0
    for _ in range(30):
        n = int(rng.integers(3, 9))
        x = rng.weibull(1.5, n) * 10 + (20 if location else 0)
        c = (rng.random(n) < 0.3).astype(int)
        if c.sum() == n:
            continue
        try:
            model = _fit(dist, x, c)
        except ValueError:
            continue
        if model.hess_inv is None:
            continue
        if slope is not None:
            se = np.sqrt(model.hess_inv[slope, slope])
            if model.params[slope] - 1.96 * se <= 0:
                continue  # the Wald ellipsoid holds a slope of 0
        fits += 1
        if location:
            spread = np.ptp(x)
            t = np.linspace(x.min() - 3 * spread, x.max() + 3 * spread, 300)
        else:
            t = np.geomspace(x.min() / 20, x.max() * 5, 300)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            band = model.cb(t, on="ff")
        assert _monotone(band[:, 0]) and _monotone(band[:, 1]), model.params
    assert fits >= 8


def test_large_sample_bounds_are_essentially_unchanged():
    np.random.seed(1)
    x = sp.Weibull.random(2000, 10, 3)
    model = sp.Weibull.fit(x)
    t = np.array([5.0, 10, 15])
    # the old logit band, from the same delta-method variance
    ctx = model._cb_context()
    sd = model._cb_sd(
        model._cb_delta_var(lambda p: model._cb_full_sf(t, p, ctx), ctx),
        t,
        "sf",
    )
    R = model.sf(t)
    d = z(0.025) * sd * np.array([1.0, -1]).reshape(2, 1)
    old = np.fliplr((R / (R + (1 - R) * np.exp(d / (R * (1 - R))))).T)
    np.testing.assert_allclose(model.cb(t, on="sf"), old, atol=3e-4)


def test_the_normal_band_is_the_quantile_scale_band():
    # On the probit scale the Normal is a line in t: the band on F at t is
    # Phi((t - mu) / sigma -/+ z se), se from the delta method.
    np.random.seed(2)
    model = sp.Normal.fit(sp.Normal.random(40, 10, 2))
    t = np.array([6.0, 10, 13])
    mu, sigma = model.params
    cov = model.hess_inv
    u = (t - mu) / sigma
    grad = np.stack([-1 / sigma * np.ones_like(t), -u / sigma], axis=1)
    se = np.sqrt(np.einsum("ij,jk,ik->i", grad, cov, grad))
    from scipy.stats import norm

    want = np.stack([norm.cdf(u - 1.959964 * se), norm.cdf(u + 1.959964 * se)])
    np.testing.assert_allclose(model.cb(t, on="ff"), want.T, rtol=1e-6)


def test_plot_can_draw_the_likelihood_ratio_band():
    model = _fit(sp.Weibull, ISSUE_X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wald = model.get_plot_data(heuristic="Turnbull")
        lr = model.get_plot_data(heuristic="Turnbull", method="lr")
        want = model.cb(lr["x_model"], on="ff", method="lr")
    np.testing.assert_allclose(lr["cbs"], want)
    assert not np.allclose(wald["cbs"], lr["cbs"])
