"""Copula fitting: counts and truncation in the margins, Clayton near
independence, and the data checks."""

import warnings

import numpy as np
import pytest

import surpyval as surv
from surpyval.multivariate import Clayton, Frank, Independence
from surpyval.multivariate.parametric.data import MultivariateSurpyvalData


def _negatively_dependent(n=500, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    z[:, 1] = -0.6 * z[:, 0] + 0.8 * z[:, 1]
    return np.exp(z + 2)


def test_clayton_near_zero_is_the_independence_copula():
    u = np.array([0.1, 0.4, 0.8])
    v = np.array([0.3, 0.9, 0.2])
    for theta in (1e-21, 1e-12):
        assert np.allclose(Clayton.cdf(u, v, theta), u * v)
        assert np.allclose(Clayton.pdf(u, v, theta), 1.0)
        assert np.allclose(Clayton.du(u, v, theta), v)
    # unchanged away from zero
    theta = 2.0
    direct = (u**-theta + v**-theta - 1) ** (-1 / theta)
    assert np.allclose(Clayton.cdf(u, v, theta), direct, rtol=1e-12)


def test_clayton_on_negative_dependence_is_no_better_than_independence():
    x = _negatively_dependent()
    margins = [surv.LogNormal, surv.LogNormal]
    clayton = Clayton.fit(x, margins=margins)
    dims = [
        Clayton._prepare_dim(clayton.margins[d], *clayton.data.dimension(d))
        for d in range(2)
    ]
    ll_i = -Independence.neg_ll([], dims, clayton.data.n)
    # at theta ~ 1e-21 the old closed form gave C = 1 and a density of
    # 1 / (u v): a likelihood ~1000 units above independence
    for theta in (1e-21, 1e-12, clayton.params[0]):
        ll_c = -Clayton.neg_ll([theta], dims, clayton.data.n)
        assert ll_c == pytest.approx(ll_i, abs=1e-3)
    assert Frank.fit(x, margins=margins).params[0] < 0


def test_ifm_margins_use_the_counts():
    x = _negatively_dependent(60)
    n = np.random.default_rng(1).integers(1, 6, size=60)
    weighted = Clayton.fit(x, n=n, margins=[surv.Weibull, surv.Weibull])
    expanded = Clayton.fit(
        np.repeat(x, n, axis=0), margins=[surv.Weibull, surv.Weibull]
    )
    for a, b in zip(weighted.margins, expanded.margins):
        assert np.allclose(a.params, b.params, rtol=1e-4)
    assert weighted.params[0] == pytest.approx(expanded.params[0], rel=1e-3)


def test_ifm_margins_use_the_truncation():
    rng = np.random.default_rng(3)
    x = rng.weibull(2.0, size=(4000, 2)) * 10
    keep = (x[:, 0] > 6) & (x[:, 1] > 6)
    x = x[keep]
    t = np.empty((len(x), 2, 2))
    t[..., 0] = 6.0
    t[..., 1] = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model = Independence.fit(x, t=t, margins=[surv.Weibull, surv.Weibull])
    for margin in model.margins:
        assert margin.params == pytest.approx([10.0, 2.0], rel=0.08)


def test_data_checks():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    with pytest.raises(ValueError, match="need xl and xr"):
        MultivariateSurpyvalData(x, c=np.array([[2, 0], [0, 0], [0, 0]]))
    # one row of codes applies to every row
    data = MultivariateSurpyvalData(x, c=np.array([0, 1]))
    assert (data.c == [[0, 1]] * 3).all()
    data = MultivariateSurpyvalData(x, c=[0, 1])
    assert (data.c == [[0, 1]] * 3).all()
