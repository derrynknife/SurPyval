"""
Consistency of the LFP (``p``), zero-inflation (``f0``) and offset
(``gamma``) conventions across the fitted-model surface (#256): the
mixture constant is ``(p - f0)`` everywhere, the zero-inflation mass
sits at 0, functions clamp to their boundary values below the (offset)
support, and the boundary does not produce NaN confidence bounds.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from surpyval import Weibull


def test_df_matches_numeric_ff_derivative_for_lfp_zi():
    m = Weibull.from_params([10, 3], p=0.8, f0=0.1)
    h = 1e-5
    for x in (2.0, 5.0, 9.0):
        num = (m.ff(x + h) - m.ff(x - h)) / (2 * h)
        assert float(m.df(x)) == pytest.approx(float(num), abs=1e-6)


def test_hf_is_df_over_sf_for_lfp_zi():
    m = Weibull.from_params([10, 3], p=0.8, f0=0.1)
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
    m = Weibull.from_params([10, 3], p=0.05)
    # With p = 0.05 and size 3 the binomial draw is usually 0 failures;
    # this crashed on np.max of an empty array (#256).
    out = m.random(3)
    assert out is not None


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
