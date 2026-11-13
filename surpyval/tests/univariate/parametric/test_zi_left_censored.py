"""Zero inflation (``fit(zi=True)``) with rows that say only "failed by
``t``".

With the share ``f0`` failing at 0 and the rest failing at a time with
distribution function ``F`` (of which a limited-failure ``p`` fail at
all, ``F_mix(t) = f0 + (p - f0) F(t)``), a unit known only to have failed
by ``t`` may have failed at 0, so it contributes ``f0 + (p - f0) F(t)``:
left censoring at ``t`` and an interval ``[0, t]`` alike (Moulton &
Halsey 1995; Braekers & Grouwels 2016). An interval opening above 0 does
not hold the mass, a late entry at ``tl > 0`` divides by
``1 - f0 - (p - f0) F(tl)``, and a ``tl`` of 0 is no truncation. These
tests check the fit against that likelihood written out with scipy, against
the zero-inflated regression with no covariates, and that data with no
such row are fitted exactly as before.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit

import surpyval as sp


def _data(seed=11, n=600):
    """Exact failures (some at 0), right censoring, left censoring at 3,
    intervals ``[0, 3]`` and ``(a, a + 2]`` with ``a > 0``, and late
    entries at 1.5 (only of units then still working)."""
    rng = np.random.default_rng(seed)
    life = 10.0 * rng.weibull(1.5, n)
    x = np.where(rng.random(n) < 0.15, 0.0, life)
    lo, hi = x.copy(), x.copy()
    c = np.zeros(n, dtype=int)
    stop = rng.uniform(8.0, 20.0, n)
    right = x > stop
    lo[right], hi[right], c[right] = stop[right], stop[right], 1
    kind = rng.integers(0, 3, n)
    early = ~right & (x < 3.0)
    as_left = early & (kind == 1)
    as_from_zero = early & (kind == 2)
    c[as_left], lo[as_left], hi[as_left] = -1, 3.0, 3.0
    c[as_from_zero], lo[as_from_zero], hi[as_from_zero] = 2, 0.0, 3.0
    window = ~right & (x >= 3.0) & (kind == 1)
    lo[window] = np.floor(x[window] / 2.0) * 2.0
    hi[window] = lo[window] + 2.0
    c[window] = 2
    tl = np.where(rng.random(n) < 0.2, 1.5, 0.0)
    keep = (tl == 0) | ((lo > tl) & (c != -1))
    xx = np.column_stack([lo, hi])[keep]
    return xx, c[keep], tl[keep]


def _hand_neg_ll(alpha, beta, f0, p, xx, c, tl):
    """The likelihood written out, with scipy's Weibull."""
    d = stats.weibull_min(beta, scale=alpha)
    lo, hi = xx[:, 0], xx[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = np.select(
            [
                (c == 0) & (lo == 0),
                c == 0,
                c == 1,
                (c == -1) | ((c == 2) & (lo == 0)),
                c == 2,
            ],
            [
                np.log(f0),
                np.log(p - f0) + d.logpdf(lo),
                np.log(1 - f0 - (p - f0) * d.cdf(lo)),
                np.log(f0 + (p - f0) * d.cdf(hi)),
                np.log(p - f0) + np.log(d.cdf(hi) - d.cdf(lo)),
            ],
        )
        late = tl > 0
        ll = ll - np.where(
            late, np.log(1 - f0 - (p - f0) * d.cdf(np.where(late, tl, 1))), 0
        )
    return -np.sum(ll)


def _fit(xx, c, tl, lfp):
    return sp.Weibull.fit(
        xx, c=c, tl=np.where(tl > 0, tl, -np.inf), zi=True, lfp=lfp
    )


@pytest.mark.parametrize("lfp", [False, True])
def test_matches_the_likelihood_by_hand(lfp):
    xx, c, tl = _data()
    assert np.any(c == -1) and np.any((c == 2) & (xx[:, 0] == 0))
    assert np.any((c == 2) & (xx[:, 0] > 0)) and np.any(tl > 0)
    model = _fit(xx, c, tl, lfp)
    p = model.lfp_p if lfp else 1.0

    # The same likelihood, at the answer and away from it ...
    for alpha, beta, f0, pp in [
        (*model.params, model.f0, p),
        (8.0, 1.2, 0.3, 0.9 if lfp else 1.0),
    ]:
        own = sp.Weibull._neg_ll_func(model.surv_data, alpha, beta, 0, f0, pp)
        hand = _hand_neg_ll(alpha, beta, f0, pp, xx, c, tl)
        assert_allclose(own, hand, rtol=1e-12)

    # ... and the same maximum, found by scipy
    def neg_ll(u):
        f0 = expit(u[2])
        pp = f0 + (1 - f0) * expit(u[3]) if lfp else 1.0
        return _hand_neg_ll(np.exp(u[0]), np.exp(u[1]), f0, pp, xx, c, tl)

    u0 = [np.log(10.0), 0.0, -1.0] + ([2.0] if lfp else [])
    res = minimize(neg_ll, u0, method="Nelder-Mead")
    res = minimize(
        neg_ll,
        res.x,
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000},
    )
    assert res.fun >= model.neg_ll() - 1e-7
    assert_allclose(res.fun, model.neg_ll(), rtol=1e-9)
    assert_allclose(np.exp(res.x[:2]), model.params, rtol=1e-4)
    assert_allclose(expit(res.x[2]), model.f0, rtol=1e-4)


@pytest.mark.parametrize("lfp", [False, True])
def test_an_interval_from_zero_is_left_censoring(lfp):
    xx, c, tl = _data()
    from_zero = (c == 2) & (xx[:, 0] == 0)
    as_left = np.where(from_zero[:, None], xx[:, 1:], xx)
    interval = _fit(xx, c, tl, lfp)
    left = _fit(as_left, np.where(from_zero, -1, c), tl, lfp)
    assert_allclose(interval.neg_ll(), left.neg_ll(), rtol=1e-12)
    assert_allclose(interval.params, left.params, rtol=1e-6)
    assert_allclose(interval.f0, left.f0, rtol=1e-6)


def test_a_truncation_at_zero_is_no_truncation():
    xx, c, _ = _data()
    none = sp.Weibull.fit(xx, c=c, zi=True)
    at_zero = sp.Weibull.fit(xx[xx[:, 0] > 0], c=c[xx[:, 0] > 0], zi=True)
    zeros = sp.Weibull.fit(
        xx[xx[:, 0] > 0],
        c=c[xx[:, 0] > 0],
        tl=np.zeros(int(np.sum(xx[:, 0] > 0))),
        zi=True,
    )
    assert_allclose(zeros.neg_ll(), at_zero.neg_ll(), rtol=1e-12)
    assert_allclose(zeros.f0, at_zero.f0, rtol=1e-6)
    assert none.f0 > at_zero.f0


@pytest.mark.parametrize("latency", [sp.Weibull, sp.LogNormal])
@pytest.mark.parametrize("as_left", [False, True])
def test_is_the_regression_with_no_covariates(latency, as_left):
    xx, c, tl = _data(seed=12, n=800)
    if as_left:
        from_zero = (c == 2) & (xx[:, 0] == 0)
        xx = np.where(from_zero[:, None], xx[:, 1:], xx)
        c = np.where(from_zero, -1, c)
    Z = np.zeros((len(c), 1))
    regression = sp.ZeroInflated(latency).fit(xx, Z, c=c, tl=tl, covariates={})
    assert regression.fit_route == "joint"
    univariate = latency.fit(
        xx, c=c, tl=np.where(tl > 0, tl, -np.inf), zi=True
    )
    assert_allclose(regression.neg_ll(), univariate.neg_ll(), rtol=1e-9)
    assert_allclose(regression.params[:-1], univariate.params, rtol=1e-4)
    assert_allclose(regression.params[-1], univariate.f0, rtol=1e-4)


def _reference_data():
    """Data with no left censoring and no interval from 0, whose fits
    the change to such rows must leave alone."""
    rng = np.random.default_rng(7)
    life = rng.weibull(1.5, 120) * 10
    x = np.where(rng.random(120) < 0.15, 0.0, life)
    out = {"exact": dict(x=x)}
    c = (x > 12).astype(int)
    out["right"] = dict(x=np.minimum(x, 12), c=c)
    cl = np.where((x > 0) & (x < 3), -1, c)
    out["left_pos"] = dict(x=np.where(cl == -1, 3.0, np.minimum(x, 12)), c=cl)
    xx = np.column_stack([x, x])
    win = (x > 2) & (x < 12)
    xx[win, 0] = np.floor(x[win] / 2) * 2
    xx[win, 1] = xx[win, 0] + 2
    out["interval_pos"] = dict(x=xx, c=np.where(win, 2, 0))
    tl = np.where(rng.random(120) < 0.3, 0.5, 0.0)
    keep = x > tl
    out["late_entry"] = dict(x=x[keep], tl=np.where(tl > 0, tl, -np.inf)[keep])
    out["tl_negative"] = dict(x=x, tl=np.full(120, -1.0))
    return out


#: ``(neg_ll at alpha 9, beta 1.4, f0 0.12 (p 0.95 with lfp), fitted
#: neg_ll, fitted f0)`` before the change, which these data do not touch.
_REFERENCE = {
    ("exact", False): (375.3140284153682, 373.541063665918, 0.13333333333),
    ("exact", True): (381.3976339102555, 373.5410760199729, 0.13333306746),
    ("right", False): (303.6090719609826, 302.15166788417804, 0.13333333333),
    ("right", True): (303.5010252627216, 302.1516679124658, 0.13333303308),
    ("left_pos", False): (
        275.4037980625836,
        258.98792803151304,
        0.26287048762,
    ),
    ("left_pos", True): (274.8773255636821, 258.98515497499, 0.26312391225),
    ("interval_pos", False): (
        329.2453834876613,
        327.38787435600455,
        0.13333333333,
    ),
    ("interval_pos", True): (
        335.3289889825486,
        327.38787436953874,
        0.13333334653,
    ),
    ("late_entry", False): (332.61018585466525, 322.3084318229629, 0.0),
    ("late_entry", True): (338.6723539961537, 322.3084312807524, 0.0),
    ("tl_negative", False): (
        375.3140284153682,
        373.541063665918,
        0.13333333333,
    ),
    ("tl_negative", True): (
        381.3976339102555,
        373.5410760199729,
        0.13333306746,
    ),
}


@pytest.mark.parametrize("key", sorted(_REFERENCE))
def test_data_without_such_rows_are_unchanged(key):
    name, lfp = key
    at, fitted, f0 = _REFERENCE[key]
    model = sp.Weibull.fit(**_reference_data()[name], zi=True, lfp=lfp)
    value = sp.Weibull._neg_ll_func(
        model.surv_data, 9.0, 1.4, 0.0, 0.12, 0.95 if lfp else 1.0
    )
    assert_allclose(float(value), at, rtol=1e-14)
    assert_allclose(model.neg_ll(), fitted, rtol=1e-9)
    assert_allclose(model.f0, f0, rtol=1e-6, atol=1e-6)


def test_bayes_likelihood_agrees():
    from surpyval.bayes.likelihood import LogLikelihood

    xx, c, tl = _data()
    data = sp.SurpyvalData(xx, c, tl=np.where(tl > 0, tl, -np.inf))
    ll = LogLikelihood(sp.Weibull, data, zi=True)
    theta = np.array([[8.0, 1.2, 0.3], [11.0, 1.6, 0.1]])
    own = [
        -sp.Weibull._neg_ll_func(data, a, b, 0, f0, 1.0) for a, b, f0 in theta
    ]
    hand = [-_hand_neg_ll(a, b, f0, 1.0, xx, c, tl) for a, b, f0 in theta]
    assert_allclose(ll(theta), own, rtol=1e-10)
    assert_allclose(ll(theta), hand, rtol=1e-10)
