"""Restricted mean survival time and the two-group RMST difference (#213).

RMST is the area under the survival curve to a horizon ``tau``; the
RMST-difference is the assumption-light alternative to the hazard ratio when
proportional hazards fails. These tests pin the point estimate against an
analytic value, and validate the two-group test for null behaviour, power and
calibration.
"""

import numpy as np
import pytest

import surpyval as sp


def _km(seed, scale, n=150, cens=0.2):
    rng = np.random.default_rng(seed)
    x = rng.exponential(scale, n)
    c = (rng.random(n) < cens).astype(int)
    return sp.KaplanMeier.fit(x, c)


def test_rmst_matches_mean_and_analytic():
    # Uncomplicated: RMST of Exponential(scale) to tau is
    #   scale * (1 - exp(-tau/scale)).
    m = _km(0, 8.0, n=4000, cens=0.0)
    tau = 10.0
    theo = 8.0 * (1 - np.exp(-tau / 8.0))
    out = m.rmst(tau=tau)
    assert out["rmst"] == pytest.approx(m.mean(tau=tau))
    assert out["rmst"] == pytest.approx(theo, abs=0.2)
    assert out["se"] > 0
    assert out["lower"] < out["rmst"] < out["upper"]
    assert out["tau"] == tau


def test_rmst_diff_identical_groups_not_significant():
    a, b = _km(1, 10.0), _km(2, 10.0)
    res = sp.rmst_diff(a, b)
    assert abs(res["difference"]) < 3.0
    assert res["p_value"] > 0.05
    assert res["se"] > 0
    assert res["lower"] < res["difference"] < res["upper"]


def test_rmst_diff_detects_real_difference():
    a, b = _km(3, 15.0), _km(4, 6.0)
    res = sp.rmst_diff(a, b)
    assert res["difference"] > 0  # group a survives longer
    assert res["p_value"] < 0.01
    assert res["ratio"] > 1.0


def test_rmst_diff_default_tau_is_common_support():
    a = sp.KaplanMeier.fit([1.0, 2.0, 3.0, 4.0, 5.0])
    b = sp.KaplanMeier.fit([1.0, 2.0, 3.0])
    res = sp.rmst_diff(a, b)
    # tau defaults to the smaller of the two groups' largest times.
    assert res["tau"] == 3.0


def test_rmst_diff_is_calibrated_under_null():
    def one(s):
        return sp.rmst_diff(_km(200 + s, 10.0), _km(700 + s, 10.0))["p_value"]

    pvals = np.array([one(s) for s in range(100)])
    assert (pvals < 0.05).mean() < 0.15  # not over-rejecting
    assert 0.35 < pvals.mean() < 0.65  # centred near 0.5


def test_rmst_diff_antisymmetric():
    a, b = _km(5, 12.0), _km(6, 8.0)
    ab = sp.rmst_diff(a, b, tau=15.0)
    ba = sp.rmst_diff(b, a, tau=15.0)
    assert ab["difference"] == pytest.approx(-ba["difference"])
    assert ab["p_value"] == pytest.approx(ba["p_value"])


def test_rmst_requires_variance_estimate():
    # A model built from an ECDF has no risk-set variance.
    m = sp.NonParametric.fit_from_ecdf([1.0, 2.0, 3.0], [0.9, 0.6, 0.2])
    with pytest.raises(ValueError, match="variance"):
        m.rmst()
