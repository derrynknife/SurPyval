import numpy as np

import surpyval as surv
from surpyval.multivariate import Clayton
from surpyval.multivariate.parametric.data import MultivariateSurpyvalData

MARGINS = [
    surv.Weibull.from_params([10, 2]),
    surv.Weibull.from_params([20, 1.5]),
]


def test_interval_censoring_matches_brute_force_integral():
    # Likelihood of a both-interval-censored pair equals the integral of the
    # joint density over the rectangle (inclusion-exclusion on C).
    theta = 2.0
    m = Clayton.from_params(theta, margins=MARGINS)
    a1, b1, a2, b2 = 8.0, 11.0, 15.0, 22.0
    data = MultivariateSurpyvalData(
        x=[[(a1 + b1) / 2], [(a2 + b2) / 2]],
        c=[[2], [2]],
        xl=[[a1], [a2]],
        xr=[[b1], [b2]],
    )
    dims = [
        Clayton._prepare_dim(m.margins[d], *data.dimension(d))
        for d in range(2)
    ]
    L = np.exp(Clayton._pair_loglik([theta], dims[0], dims[1]))[0]

    gx = np.linspace(a1, b1, 400)
    gy = np.linspace(a2, b2, 400)
    GX, GY = np.meshgrid(gx, gy)
    pdf = m.pdf(np.column_stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    integral = np.trapezoid(np.trapezoid(pdf, gy, axis=0), gx)

    assert np.isclose(L, integral, rtol=1e-3)


def test_uninformative_right_censoring_reduces_to_marginal_density():
    # If dim 2 is right-censored at the support floor it carries no info, so
    # each row's likelihood collapses to the dim-1 density.
    theta = 3.0
    m = Clayton.from_params(theta, margins=MARGINS)
    x1 = np.array([5.0, 10.0, 15.0])
    data = MultivariateSurpyvalData(
        x=[x1, np.zeros_like(x1)],
        c=[np.zeros_like(x1, dtype=int), np.ones_like(x1, dtype=int)],
    )
    dims = [
        Clayton._prepare_dim(m.margins[d], *data.dimension(d))
        for d in range(2)
    ]
    L = np.exp(Clayton._pair_loglik([theta], dims[0], dims[1]))
    assert np.allclose(L, MARGINS[0].df(x1), rtol=1e-4)


def test_right_censored_fit_recovers_theta():
    true_p = 3.0
    m = Clayton.from_params(true_p, margins=MARGINS)
    data = m.random(4000, random_state=4)
    thr = np.array([12.0, 25.0])
    c = (data > thr).astype(int)
    xobs = np.minimum(data, thr)
    fit = Clayton.fit(
        [xobs[:, 0], xobs[:, 1]],
        c=[c[:, 0], c[:, 1]],
        margins=[surv.Weibull, surv.Weibull],
        how="IFM",
    )
    # a quarter of each dimension is censored
    assert 0.15 < c.mean() < 0.35
    assert np.isclose(fit.params[0], true_p, rtol=0.2)


def test_interval_censored_margin_fit_recovers_theta():
    # One dimension is interval (grouped) censored via the public fit API.
    true_p = 2.0
    m = Clayton.from_params(true_p, margins=MARGINS)
    data = m.random(3000, random_state=9)
    lo = np.floor(data[:, 0] / 2.0) * 2.0
    hi = lo + 2.0
    c0 = np.full(3000, 2)
    c1 = np.zeros(3000, dtype=int)
    fit = Clayton.fit(
        [lo, data[:, 1]],
        c=[c0, c1],
        xl=[lo, data[:, 1]],
        xr=[hi, data[:, 1]],
        margins=[surv.Weibull, surv.Weibull],
        how="IFM",
    )
    assert np.isclose(fit.params[0], true_p, rtol=0.2)
    assert np.allclose(fit.margins[0].params, [10, 2], rtol=0.15)


def test_left_truncation_normalisation_is_neutral_when_window_is_full():
    # A truncation window of (-inf, inf) must not change the likelihood.
    theta = 2.0
    m = Clayton.from_params(theta, margins=MARGINS)
    x = np.array([[10.0, 18.0], [6.0, 25.0]])
    base = MultivariateSurpyvalData(x=[x[:, 0], x[:, 1]])
    t = np.empty((2, 2, 2))
    t[..., 0] = -np.inf
    t[..., 1] = np.inf
    trunc = MultivariateSurpyvalData(x=[x[:, 0], x[:, 1]], t=t)
    d0 = [
        Clayton._prepare_dim(m.margins[i], *base.dimension(i))
        for i in range(2)
    ]
    d1 = [
        Clayton._prepare_dim(m.margins[i], *trunc.dimension(i))
        for i in range(2)
    ]
    assert np.allclose(
        Clayton._pair_loglik([theta], d0[0], d0[1]),
        Clayton._pair_loglik([theta], d1[0], d1[1]),
    )
