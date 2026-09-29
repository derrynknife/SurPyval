"""The stable forms behind the tail accuracy of the continuous
distributions (#410, #442-#445, #447), beyond the grid of
``reference/test_tails.py``: the branches of the new forms that grid does
not reach, their support edges without raw warnings, and their autograd
gradients, which the fits rely on.
"""

import warnings

import autograd
import autograd.numpy as anp
import numpy as np
import pytest

import surpyval as sp


def _quiet(f, *args):
    """``f(*args)``, failing on any warning (a raw numpy one included)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return f(*args)


# ---------------------------------------------------------------------------
# Branches the tail grid does not reach
# ---------------------------------------------------------------------------
# 50-digit mpmath values of the standard normal hazard phi(z) / Q(z).
NORMAL_HAZARD = {
    40.0: 40.024968847207263723,
    99.5: 99.510048221978376256,
    100.5: 100.50994827943335702,
    150.0: 150.00666607420571802,
    1000.0: 1000.00099999800001,
    1e6: 1000000.000001,
}


def test_normal_hazard_far_right():
    # pdf / sf was 0 / 0 = NaN from z = 38.5 on (#444); above z = 100 the
    # asymptotic series takes over from the difference of the logs.
    z = np.array(list(NORMAL_HAZARD))
    true = np.array(list(NORMAL_HAZARD.values()))
    np.testing.assert_allclose(_quiet(sp.Normal.hf, z, 0.0, 1.0), true, 1e-12)
    np.testing.assert_allclose(
        _quiet(sp.Normal.hf, 3.0 + 2.0 * z, 3.0, 2.0), true / 2.0, 1e-12
    )
    # the limits
    hf = _quiet(sp.Normal.hf, np.array([np.inf, -np.inf]), 0.0, 1.0)
    np.testing.assert_array_equal(hf, [np.inf, 0.0])


def test_lognormal_hazard_far_right_and_at_the_edges():
    z = np.array(list(NORMAL_HAZARD))[:-1]
    x = np.exp(1.0 + 0.5 * z)
    true = np.array(list(NORMAL_HAZARD.values()))[:-1] / (0.5 * x)
    np.testing.assert_allclose(
        _quiet(sp.LogNormal.hf, x, 1.0, 0.5), true, 1e-12
    )
    hf = _quiet(sp.LogNormal.hf, np.array([0.0, np.inf]), 1.0, 0.5)
    np.testing.assert_array_equal(hf, [0.0, 0.0])


def test_hypoexponential_one_stage_is_the_exponential():
    # One stage reaches F = 1 - 1/e inside the series region, where the
    # survival function must still come from 1 - F.
    x = np.array([0.5, 0.999999, 1.000001, 3.0, 50.0, 800.0]) / 0.3
    for fn in ("sf", "ff", "df", "hf", "Hf", "log_sf", "log_ff", "log_df"):
        np.testing.assert_allclose(
            getattr(sp.Hypoexponential, fn)(x, 0.3),
            getattr(sp.Exponential, fn)(x, 0.3),
            rtol=1e-13,
            err_msg=fn,
        )


@pytest.mark.parametrize("rates", [(1.0, 2.0), (0.5, 1.5, 3.0)])
def test_hypoexponential_quantile_of_a_tiny_probability(rates):
    # Near 0, F(x) = prod(rates) x^m / m! to double precision, so the
    # quantile of u is (m! u / prod(rates))^(1/m); it was about 1e-60
    # whatever u (#447).
    m = len(rates)
    u = np.array([1e-300, 1e-100])
    q = _quiet(sp.Hypoexponential.qf, u, *rates)
    factorial = np.prod(np.arange(1.0, m + 1.0))
    expected = (factorial * u / np.prod(rates)) ** (1.0 / m)
    np.testing.assert_allclose(q, expected, rtol=1e-13)
    # and the quantile inverts the CDF there and at 1e-30
    u = np.array([1e-300, 1e-100, 1e-30])
    q = _quiet(sp.Hypoexponential.qf, u, *rates)
    np.testing.assert_allclose(
        sp.Hypoexponential.log_ff(q, *rates), np.log(u), rtol=1e-13
    )


def test_gamma_log_cdf_where_the_cdf_underflows():
    # log P(a, x) from its series once P underflows: P(3, x) = x^3 / 6
    # (1 - 3x/4 + ...) at small x, and log P(0.5, 0) is -inf, not the
    # -708 of a clipped log (#443).
    x = np.array([1e-200, 1e-120])
    np.testing.assert_allclose(
        _quiet(sp.Gamma.log_ff, x, 3.0, 1.0),
        3.0 * np.log(x) - np.log(6.0),
        rtol=1e-14,
    )
    assert sp.Gamma.log_ff(0.0, 0.5, 1.0) == -np.inf


# ---------------------------------------------------------------------------
# The support edge: exact limits, without a raw warning
# ---------------------------------------------------------------------------
EDGE_CASES = [
    # (distribution, params, log_df at 0, log_ff at 0)
    ("Weibull", (2.0, 0.5), np.inf, -np.inf),
    ("Weibull", (2.0, 1.0), -np.log(2.0), -np.inf),
    ("Weibull", (2.0, 3.0), -np.inf, -np.inf),
    ("LogLogistic", (2.0, 1.0), -np.log(2.0), -np.inf),
    ("Gamma", (1.0, 3.0), np.log(3.0), -np.inf),
    ("Gamma", (2.0, 3.0), -np.inf, -np.inf),
    ("LogNormal", (0.0, 1.0), -np.inf, -np.inf),
    ("Rayleigh", (2.0,), -np.inf, -np.inf),
    ("Exponential", (2.0,), np.log(2.0), -np.inf),
]


@pytest.mark.parametrize(
    "name, params, log_df, log_ff",
    EDGE_CASES,
    ids=["{}{}".format(c[0], c[1]) for c in EDGE_CASES],
)
def test_log_functions_at_zero(name, params, log_df, log_ff):
    dist = getattr(sp, name)
    x = np.array([0.0, 1.0, np.nan])
    got_df = _quiet(dist.log_df, x, *params)
    got_ff = _quiet(dist.log_ff, x, *params)
    assert got_df[0] == log_df and got_ff[0] == log_ff
    assert np.isfinite(got_df[1]) and np.isfinite(got_ff[1])
    # a NaN in, a NaN out
    assert np.isnan(got_df[2]) and np.isnan(got_ff[2])
    # the density itself at the edge (0 * log 0 was NaN, #444)
    df = _quiet(dist.df, np.array([0.0]), *params)[0]
    assert df == np.exp(log_df)


def test_uniform_logs_at_the_edges():
    lf = _quiet(sp.Uniform.log_ff, np.array([0.0, 1e-300, 1.0]), 0.0, 1.0)
    ls = _quiet(sp.Uniform.log_sf, np.array([0.0, 1e-300, 1.0]), 0.0, 1.0)
    np.testing.assert_allclose(lf, [-np.inf, -690.7755278982137, 0.0])
    np.testing.assert_array_equal(ls, [0.0, -1e-300, -np.inf])
    assert _quiet(sp.Uniform.Hf, 1.0, 0.0, 1.0) == np.inf


# ---------------------------------------------------------------------------
# autograd: the fits differentiate these functions
# ---------------------------------------------------------------------------
GRADIENT_CASES = [
    ("Weibull", (2.0, 1.5), [0.3, 2.0, 7.0]),
    ("Exponential", (0.5,), [0.3, 2.0, 30.0]),
    ("Rayleigh", (2.0,), [0.3, 2.0, 9.0]),
    ("Uniform", (0.5, 11.0), [0.6, 3.0, 10.9]),
    ("Gumbel", (5.0, 2.0), [-20.0, 5.0, 8.0]),
    ("GumbelLEV", (5.0, 2.0), [-2.0, 5.0, 30.0]),
    ("Normal", (5.0, 2.0), [-30.0, 5.0, 30.0]),
    ("LogNormal", (1.0, 0.5), [0.3, 2.7, 80.0]),
    ("Logistic", (5.0, 2.0), [-50.0, 5.0, 60.0]),
    ("LogLogistic", (3.0, 2.0), [0.3, 3.0, 1e4]),
    ("Gamma", (2.0, 0.5), [0.3, 3.0, 80.0]),
]
FUNCTIONS = ["sf", "ff", "df", "hf", "Hf", "log_sf", "log_ff", "log_df"]


@pytest.mark.parametrize(
    "name, params, xs", GRADIENT_CASES, ids=[c[0] for c in GRADIENT_CASES]
)
def test_gradients_match_central_differences(name, params, xs):
    dist = getattr(sp, name)
    x = np.array(xs)
    p0 = np.array(params, dtype=float)
    for fn in FUNCTIONS:
        f = getattr(dist, fn)

        def total(p, f=f):
            return anp.sum(f(x, *[p[k] for k in range(len(params))]))

        grad = autograd.grad(total)(p0)
        for k in range(len(params)):
            h = 1e-6 * max(abs(p0[k]), 1.0)
            up, down = p0.copy(), p0.copy()
            up[k] += h
            down[k] -= h
            fd = (total(up) - total(down)) / (2 * h)
            assert np.isfinite(grad[k]), (fn, k)
            np.testing.assert_allclose(
                grad[k],
                fd,
                rtol=1e-5,
                atol=1e-6,
                err_msg="{} {}".format(fn, k),
            )


@pytest.mark.parametrize(
    "name", [c[0] for c in GRADIENT_CASES if c[0] != "Uniform"]
)
def test_censored_and_truncated_fits_give_finite_bounds(name):
    dist = getattr(sp, name)
    if name in ("Gumbel", "GumbelLEV", "Normal", "Logistic"):
        x = sp.Normal.random(200, 10.0, 3.0, random_state=1)
    else:
        x = sp.Weibull.random(200, 10.0, 2.0, random_state=1)
    c = (np.random.default_rng(1).random(200) < 0.2).astype(int)
    for kw in (dict(c=c), dict(c=-c), dict(tl=np.full(200, 0.9 * x.min()))):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = dist.fit(x, **kw)
        for param in dist.param_names:
            assert np.all(np.isfinite(model.param_cb(param))), (kw, param)
