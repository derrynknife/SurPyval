"""
Consistency of the LFP (``p``), zero-inflation (``f0``) and offset
(``gamma``) conventions across the fitted-model surface (#256): the
mixture constant is ``(p - f0)`` everywhere, the zero-inflation mass
sits at 0, functions clamp to their boundary values below the (offset)
support, and the boundary does not produce NaN confidence bounds.
"""

import warnings

import numpy as np
import pytest
from scipy.integrate import quad

from surpyval import LogNormal, Weibull


def test_df_matches_numeric_ff_derivative_for_lfp_zi():
    m = Weibull.from_params([10, 3], lfp_p=0.8, f0=0.1)
    h = 1e-5
    for x in (2.0, 5.0, 9.0):
        num = (m.ff(x + h) - m.ff(x - h)) / (2 * h)
        assert float(m.df(x)) == pytest.approx(float(num), abs=1e-6)


def test_hf_is_df_over_sf_for_lfp_zi():
    m = Weibull.from_params([10, 3], lfp_p=0.8, f0=0.1)
    x = np.array([2.0, 5.0, 9.0])
    assert np.allclose(m.hf(x), m.df(x) / m.sf(x))


def test_mean_and_moment_include_f0():
    m = Weibull.from_params([10, 3], f0=0.2)
    integral, _ = quad(lambda t: t * float(np.ravel(m.df(t))[0]), 0, 200)
    assert m.mean() == pytest.approx(integral, abs=1e-3)
    assert m.moment(1) == pytest.approx(m.mean(), abs=1e-12)


def test_qf_inverts_ff_for_offset_zi():
    m = Weibull.from_params([10, 3], gamma=2, f0=0.2)
    # The zero-inflation mass sits at 0 (consistent with ff(0) == f0).
    assert float(m.ff(0)) == pytest.approx(0.2)
    assert m.qf(0.1) == 0.0
    q = m.qf(0.5)
    assert float(m.ff(q)) == pytest.approx(0.5, abs=1e-9)


def test_offset_model_clamps_below_gamma():
    m = Weibull.from_params([10, 3], gamma=5)
    x = np.array([0.0, 2.0, 4.0])
    assert np.all(m.ff(x) == 0.0)
    assert np.all(m.sf(x) == 1.0)
    assert np.all(m.df(x) == 0.0)
    assert np.all(m.Hf(x) == 0.0)
    assert np.all(m.hf(x) == 0.0)


def test_cb_finite_at_boundary():
    np.random.seed(2)
    x = Weibull.random(80, 10, 3)
    plain = Weibull.fit(x)
    cb = plain.cb([0.0, 5.0])
    assert np.all(np.isfinite(cb))
    assert cb[0, 0] == 1.0 and cb[0, 1] == 1.0

    offset = Weibull.fit(x, offset=True)
    cb_o = offset.cb([0.0, 5.0, 15.0])
    assert np.all(np.isfinite(cb_o))


def test_lfp_random_with_no_failures_drawn():
    np.random.seed(3)
    m = Weibull.from_params([10, 3], lfp_p=0.05)
    # With p = 0.05 and size 3 the draw usually has no failures; the
    # survival-data draw crashed on np.max of an empty array (#256). It is
    # random_data since #403 (random draws the lifetimes, here all inf).
    x, c, n, _ = m.random_data(3)
    assert np.all(c == 1) and n.sum() == 3 and np.all(np.isfinite(x))


def test_left_censored_fit_uses_stable_path():
    # The numerically stable log_ff branch was unreachable (inverted
    # ``f0 == 1`` condition, #256).
    x = np.array([1.0, 2.0, 3.0, 4.0, 25.0, 30.0])
    c = np.array([-1, -1, 0, 0, 0, 1])
    m = Weibull.fit(x, c=c)
    assert np.isfinite(m.neg_ll())


def test_aic_c_uses_full_parameter_count():
    np.random.seed(4)
    x = Weibull.random(100, 10, 3)
    m = Weibull.fit(x, offset=True)
    k = m.k
    n = m.data["n"].sum()
    expected = m.aic() + (2 * k**2 + 2 * k) / (n - k - 1)
    assert m.aic_c() == pytest.approx(expected, abs=1e-12)


def test_710_zero_inflated_Hf_is_finite_in_the_far_tail():
    # Hf was -log sf, inf once sf underflowed; it is -log(1 - f0) + H
    # from 0 on, and its Wald bound follows it.
    m = Weibull.from_params([10, 3], gamma=2, f0=0.2)
    x = np.array([-1.0, 0.0, 1.0, 5.0, 30.0, 1000.0])
    expected = np.r_[0.0, -np.log(0.8), -np.log(0.8), np.zeros(3)]
    expected[3:] = -np.log(0.8) + ((x[3:] - 2) / 10) ** 3
    np.testing.assert_allclose(m.Hf(x), expected, rtol=1e-12)
    assert m.sf(1000.0) == 0
    np.testing.assert_allclose(m.Hf(x[:5]), -np.log(m.sf(x[:5])), rtol=1e-12)

    np.random.seed(0)
    data = np.r_[np.zeros(10), Weibull.random(40, 10, 2)]
    fit = Weibull.fit(data, zi=True)
    t = np.array([5.0, 100.0, 1000.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        band = fit.cb(t, on="Hf")
        missing = fit.cb([5.0, np.nan], on="Hf")
    H = fit.Hf(t)
    assert np.all(np.isfinite(H)) and np.all(np.isfinite(band))
    assert np.all((band[:, 0] < H) & (H < band[:, 1]))
    # Where sf is a normal number the band is -log of the sf band.
    np.testing.assert_allclose(
        band[0], -np.log(fit.cb(5.0, on="sf"))[::-1], rtol=1e-9
    )
    assert np.isnan(missing[1]).all()


@pytest.mark.parametrize("lfp", [False, True])
def test_728_zero_inflated_hazard_at_zero_is_f0_and_bounded(lfp):
    # The point mass at 0 is a discrete hazard, f0 / sf(0-) = f0 (it was
    # df / sf(0) = f0 / (1 - f0)); its df and hf bounds were [0, 0].
    m = Weibull.from_params([10, 3], lfp_p=0.8 if lfp else 1, f0=0.2)
    assert m.hf(0.0) == pytest.approx(0.2)
    assert m.df(0.0) == pytest.approx(m.hf(0.0) * m.sf(-1.0))
    assert m.Hf(0.0) == pytest.approx(-np.log(0.8))

    np.random.seed(0)
    data = np.r_[np.zeros(10), Weibull.random(40, 10, 2)]
    c = np.zeros_like(data)
    if lfp:
        data, c = np.r_[data, np.full(5, 1e4)], np.r_[c, np.ones(5)]
    fit = Weibull.fit(data, c=c, zi=True, lfp=lfp)
    f0_band = fit.param_cb("f0")
    for on in ("hf", "df"):
        assert getattr(fit, on)(0.0) == pytest.approx(fit.f0)
        band = fit.cb([0.0, 5.0], on=on)
        # At 0 it is f0's own (logit) Wald interval, as the f0 bound.
        np.testing.assert_allclose(band[0], f0_band, rtol=1e-6)
        assert np.all(band[:, 0] < getattr(fit, on)([0.0, 5.0]))
        assert np.all(getattr(fit, on)([0.0, 5.0]) < band[:, 1])
    for on in ("sf", "ff", "Hf"):
        lo, hi = np.sort(fit.cb(0.0, on=on))
        assert lo < getattr(fit, on)(0.0) < hi


@pytest.mark.parametrize(
    "kw", [{}, {"f0": 0.1}, {"gamma": 2.0}, {"f0": 0.1, "gamma": 2.0}]
)
def test_728_lfp_Hf_keeps_its_digits_where_ff_is_tiny(kw):
    # -log sf was 0 once sf rounded to 1 (Hf = 9e-40 at x = 1e-12): it is
    # -log1p(-ff) there, and -log sf in the upper tail (sf -> 1 - p).
    m = Weibull.from_params([10, 3], lfp_p=0.9, **kw)
    g, f0 = kw.get("gamma", 0.0), kw.get("f0", 0.0)
    x = np.array([1e-12, 1e-4, 3.0, 30.0, 1e4]) + g
    F = -np.expm1(-(((x - g) / 10) ** 3))
    expected = -np.log1p(-(f0 + (0.9 - f0) * F))
    np.testing.assert_allclose(m.Hf(x), expected, rtol=1e-13)
    assert m.Hf(1e4 + g) == pytest.approx(-np.log(0.1), rel=1e-14)
    if not kw:
        assert m.Hf(1e-12) == pytest.approx(0.9e-39, rel=1e-13)
    H0 = m.Hf([-1.0, g / 2])
    np.testing.assert_array_equal(H0, [0.0, -np.log1p(-f0)])
    assert not np.any(np.signbit(H0))


def test_728_left_tail_ff_and_Hf_bounds_keep_their_digits():
    # 1 - R and -log R of the survival band were 0 (or 2e-16) where R
    # rounds to 1; they are now mapped from the band's own scale.
    np.random.seed(1)
    x = Weibull.random(60, 10, 2)
    lfp_x, lfp_c = np.r_[x, np.full(15, 50.0)], np.r_[x * 0, np.ones(15)]
    t = np.array([1e-4, 1e-2, 1.0])
    for model in (
        LogNormal.fit(x),
        LogNormal.fit(lfp_x, c=lfp_c, lfp=True),
    ):
        for on in ("ff", "Hf"):
            est = getattr(model, on)(t)
            band = model.cb(t, on=on)
            assert np.all((band[:, 0] < est) & (est < band[:, 1]))
        # ff and Hf agree to first order where they are small.
        np.testing.assert_allclose(
            model.cb(t[:2], on="ff"), model.cb(t[:2], on="Hf"), rtol=1e-9
        )
