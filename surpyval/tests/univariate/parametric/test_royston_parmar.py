"""Tests for the Royston-Parmar flexible parametric model."""

import json

import numpy as np
import pytest

import surpyval as surv
from surpyval import LogNormal, RoystonParmar, Weibull


def test_df1_hazard_scale_is_weibull():
    np.random.seed(0)
    x = Weibull.random(800, 10.0, 1.8)
    rp = RoystonParmar.fit(x, df=1, scale="hazard")
    wb = Weibull.fit(x)
    t = np.array([2.0, 6.0, 12.0, 20.0, 30.0])
    assert np.allclose(rp.sf(t), wb.sf(t), atol=3e-3)


def test_df1_normal_scale_is_lognormal():
    np.random.seed(1)
    x = LogNormal.random(800, 2.0, 0.5)
    rp = RoystonParmar.fit(x, df=1, scale="normal")
    ln = LogNormal.fit(x)
    t = np.array([3.0, 6.0, 10.0, 18.0])
    assert np.allclose(rp.sf(t), ln.sf(t), atol=5e-3)


def test_flexible_shape_beats_weibull_by_aic():
    # A bimodal mixture no single Weibull can fit; more spline df should win.
    np.random.seed(2)
    x = np.concatenate(
        [Weibull.random(400, 3, 5), Weibull.random(400, 30, 1.2)]
    )
    aic = {d: RoystonParmar.fit(x, df=d).aic() for d in (1, 2, 3, 4)}
    assert min(aic, key=aic.get) > 1  # a flexible df beats the Weibull (df=1)
    assert aic[3] < aic[1] - 50  # and by a clear margin


@pytest.mark.parametrize("scale", ["hazard", "odds", "normal"])
def test_all_scales_give_valid_survival(scale):
    np.random.seed(3)
    x = Weibull.random(500, 10.0, 1.6)
    rp = RoystonParmar.fit(x, df=3, scale=scale)
    t = np.linspace(0.5, 40, 200)
    s = rp.sf(t)
    assert np.all(s <= 1.0 + 1e-9) and np.all(s >= -1e-9)
    assert np.all(np.diff(s) <= 1e-9)  # monotone non-increasing


def test_function_identities():
    np.random.seed(4)
    x = Weibull.random(500, 10.0, 1.8)
    rp = RoystonParmar.fit(x, df=3)
    t = np.array([3.0, 8.0, 15.0])
    assert np.allclose(rp.Hf(t), -np.log(rp.sf(t)))
    assert np.allclose(rp.hf(t), rp.df(t) / rp.sf(t))
    assert np.allclose(rp.ff(t), 1.0 - rp.sf(t))


def test_qf_inverts_ff():
    np.random.seed(5)
    x = Weibull.random(500, 10.0, 1.8)
    rp = RoystonParmar.fit(x, df=3)
    for q in (0.1, 0.5, 0.9):
        t = rp.qf(q)
        assert float(rp.ff(np.array([t]))[0]) == pytest.approx(q, abs=1e-4)


def test_mean_matches_survival_integral():
    np.random.seed(6)
    x = Weibull.random(600, 10.0, 1.8)
    rp = RoystonParmar.fit(x, df=3)
    wb = Weibull.fit(x)
    assert rp.mean() == pytest.approx(float(wb.mean()), rel=0.05)


def test_cb_band_brackets_and_valid():
    np.random.seed(7)
    x = Weibull.random(400, 10.0, 1.8)
    rp = RoystonParmar.fit(x, df=3)
    t = np.linspace(1, 30, 20)
    band = rp.cb(t, on="sf")
    point = rp.sf(t)
    assert np.all(band[:, 0] <= point + 1e-9)
    assert np.all(point <= band[:, 1] + 1e-9)
    assert np.all(band >= -1e-9) and np.all(band <= 1 + 1e-9)


def test_extrapolates_linearly_beyond_boundary_knots():
    # Beyond the upper boundary knot log H is linear in log t (Weibull tail),
    # so its second derivative in log-time is ~0.
    np.random.seed(8)
    x = Weibull.random(500, 10.0, 1.8)
    rp = RoystonParmar.fit(x, df=3, scale="hazard")
    far = np.exp(rp.knots[-1] + np.array([1.0, 1.5, 2.0]))
    logH = np.log(rp.Hf(far))
    second_diff = np.diff(np.diff(logH))
    assert np.all(np.abs(second_diff) < 1e-6)


def test_right_censoring_and_truncation():
    np.random.seed(9)
    x = Weibull.random(500, 12.0, 1.5)
    c = (x > 20).astype(int)
    xc = np.minimum(x, 20.0)
    rp = RoystonParmar.fit(xc, c=c, df=3)
    assert rp.n_events == int((c == 0).sum())
    tl = np.full_like(x, 1.0)
    rp_t = RoystonParmar.fit(x, tl=tl, df=2)
    assert np.isfinite(rp_t._neg_ll)


def test_serialisation_round_trip():
    np.random.seed(10)
    x = Weibull.random(400, 10.0, 1.8)
    rp = RoystonParmar.fit(x, df=3, scale="odds")
    restored = surv.from_dict(json.loads(json.dumps(rp.to_dict())))
    assert type(restored).__name__ == "RoystonParmarModel"
    t = np.array([4.0, 10.0, 20.0])
    assert np.allclose(rp.sf(t), restored.sf(t))
    assert restored.scale == "odds"
    assert np.allclose(restored.knots, rp.knots)


def test_explicit_knots():
    np.random.seed(11)
    x = Weibull.random(300, 10.0, 1.8)
    knots = np.log(np.array([1.0, 8.0, 30.0]))  # 1 internal knot
    rp = RoystonParmar.fit(x, knots=knots)
    assert np.allclose(rp.knots, knots)
    assert rp.k == 3


def test_guards():
    x = Weibull.random(100, 10.0, 1.8)
    with pytest.raises(ValueError, match="scale must be"):
        RoystonParmar.fit(x, scale="weird")
    with pytest.raises(ValueError, match="positive"):
        RoystonParmar.fit(np.array([-1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="right-censored"):
        RoystonParmar.fit(np.array([1.0, 2.0, 3.0]), c=np.array([0, 2, 1]))
    with pytest.raises(ValueError, match="event"):
        RoystonParmar.fit(np.array([1.0, 2.0, 3.0]), c=np.array([1, 1, 1]))
