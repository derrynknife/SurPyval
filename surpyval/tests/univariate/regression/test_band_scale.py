"""The parametric regression Wald bands on ``sf``/``ff``/``Hf`` are on the
baseline family's probability-plot scale, as the univariate and degradation
bands are (#504, after #477).

Before #504 every regression band was formed on the logit of the survival:
a regression with its coefficient fixed at 0 gave a band up to 120 % away
from the univariate fit's at the same point (LogNormal), and on ten-point
samples the Weibull PH and AFT bands on ``ff`` turned back in the left tail
in 200 of 200 fits (the Normal AFT's in 16).
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval import StepSchedule
from surpyval.utils.linalg import (
    link_band,
    sf_from_link,
    sf_link_bound,
    sf_link_from_H,
    sf_link_from_sf,
)

T = np.array([0.5, 2.0, 5.0, 10.0, 20.0])


def _data(n=12, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.weibull(1.5, n) * 10
    Z = rng.normal(size=(n, 1))
    return x, Z


@pytest.mark.parametrize(
    "fitter, dist, link",
    [
        ("WeibullPH", "Weibull", "loglog"),
        ("WeibullAFT", "Weibull", "loglog"),
        ("LogNormalAFT", "LogNormal", "probit"),
        ("LogisticPO", "Logistic", "logit"),
    ],
)
@pytest.mark.parametrize("on", ["sf", "ff", "Hf"])
def test_no_effect_gives_the_univariate_band(fitter, dist, link, on):
    # With the coefficient fixed at 0 the fit is the univariate one, and
    # so is its covariance: the band at Z = 0 must be the univariate band.
    x, Z = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = getattr(sp, fitter).fit(x, Z, fixed={"coef_0": 0.0})
    uni = getattr(sp, dist).fit(x)
    assert reg._cb_link == link
    np.testing.assert_allclose(reg.params[:-1], uni.params, rtol=1e-6)
    for bound in ("two-sided", "lower", "upper"):
        np.testing.assert_allclose(
            reg.cb(T, [0.0], on=on, bound=bound),
            uni.cb(T, on=on, bound=bound),
            rtol=1e-6,
        )


@pytest.mark.parametrize(
    "fitter", ["WeibullPH", "WeibullAFT", "LogNormalAFT", "NormalAFT"]
)
def test_small_sample_band_is_monotone(fitter):
    # Ten points and a binary covariate; the old logit band on ff turned
    # back in the left tail of every Weibull fit here.
    t = np.geomspace(0.01, 100, 400)
    for seed in range(100, 120):
        rng = np.random.default_rng(seed)
        Z = rng.binomial(1, 0.5, (10, 1)).astype(float)
        if fitter == "NormalAFT":
            x = rng.normal(20, 4, 10) + 2 * Z[:, 0]
        else:
            x = rng.weibull(1.5, 10) * 10 * np.exp(0.3 * Z[:, 0])
        model = getattr(sp, fitter).fit(x, Z)
        for on in ("sf", "ff", "Hf"):
            band = model.cb(t, [0.0], on=on)
            step = np.diff(band, axis=0)
            if on == "sf":
                step = -step
            assert np.all(step >= -1e-12 * np.abs(band[1:])), (seed, on)


@pytest.mark.parametrize(
    "fitter", ["WeibullPH", "LogNormalAFT", "LogisticPO", "GammaPH"]
)
def test_hf_bound_has_no_ceiling(fitter):
    # Far in the right tail sf underflows; the bounds come from H, so the
    # Hf band stays finite and around the estimate on every scale (#418).
    x, Z = _data(60, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = getattr(sp, fitter).fit(x, Z)
    t = np.array([1.0, 1e3, 1e5])
    H = model.Hf(t, [0.2])
    band = model.cb(t, [0.2], on="Hf")
    assert np.all(np.isfinite(band))
    assert np.all((band[:, 0] <= H) & (H <= band[:, 1]))
    sf_band = model.cb(t, [0.2], on="sf")
    assert np.all((sf_band >= 0) & (sf_band <= 1))


def test_cb_tvc_on_a_constant_path_is_cb():
    x, Z = _data(40, seed=5)
    for fitter in ("WeibullPH", "LogNormalAFT"):
        model = getattr(sp, fitter).fit(x, Z)
        schedule = StepSchedule.constant([0.4])
        t = np.array([1.0, 5.0, 12.0])
        for on in ("sf", "ff", "Hf"):
            np.testing.assert_allclose(
                model.cb_tvc(t, schedule, on=on),
                model.cb(t, [0.4], on=on),
                rtol=1e-10,
            )


@pytest.mark.parametrize("link", ["loglog", "probit", "logit"])
def test_link_scale_round_trips_in_both_tails(link):
    H = np.array([1e-300, 1e-20, 1e-3, 0.7, 5.0, 40.0, 800.0, 1e5])
    u = sf_link_from_H(H, link)
    # (scipy ndtri_exp keeps about 1e-12 at the extremes)
    np.testing.assert_allclose(sf_from_link(u, link, "Hf"), H, rtol=1e-11)
    sf, ff = np.exp(-H), -np.expm1(-H)
    keep = sf > 1e-300
    np.testing.assert_allclose(
        sf_link_from_sf(sf, ff, link)[keep], u[keep], rtol=1e-12
    )
    np.testing.assert_allclose(sf_from_link(u, link, "ff"), ff, rtol=1e-12)
    # the edges: sf exactly 1 or 0 are the estimate, with no width
    edges = link_band(
        sf_link_from_H(np.array([0.0, np.inf]), link),
        np.array([np.nan, np.nan]),
        0.05,
        "two-sided",
        link,
        "sf",
    )
    np.testing.assert_array_equal(edges, [[1.0, 1.0], [0.0, 0.0]])


def test_sf_link_bound_orders_its_ends():
    sf = np.array([0.999, 0.6, 0.01])
    se = np.array([1e-4, 0.05, 0.004])
    for link in ("loglog", "probit", "logit"):
        two = sf_link_bound(sf, se, 0.1, "two-sided", link)
        assert np.all(two[:, 0] < sf) and np.all(sf < two[:, 1])
        np.testing.assert_allclose(
            two[:, 0], sf_link_bound(sf, se, 0.05, "lower", link)
        )
        hf = sf_link_bound(sf, se, 0.1, "two-sided", link, on="Hf")
        np.testing.assert_allclose(hf[:, ::-1], -np.log(two), rtol=1e-12)
