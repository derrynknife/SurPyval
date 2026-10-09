"""Likelihood-ratio bounds where the profile levels off or follows a
valley (#421).

The NegativeBinomial and ExpoWeibull fits of the conformance registry
have profile likelihoods that the old searches did not follow: their
nuisance parameters run off along curved valleys (a NegativeBinomial
``p`` of ``1 - lambda / r`` as ``r`` grows; an ExpoWeibull ``beta``
running off to infinity with ``alpha`` at the largest observation), and
some profiles level off below the critical value, where the bound is the
edge of the parameter's space.
"""

import itertools
import warnings

import numpy as np
import pytest
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import expit, logit
from scipy.special import ndtri as z
from scipy.stats import poisson

import surpyval as surv
import surpyval.univariate.parametric._likelihood_ratio as likelihood_ratio
from surpyval.tests._helpers import (
    fresh_conformance_fit,
    neg_ll_at,
    no_warnings,
)
from surpyval.tests.conformance.registry import CASE_BY_NAME
from surpyval.univariate.parametric import _likelihood_ratio

CRIT_95 = z(0.975) ** 2
CRIT_80 = z(0.9) ** 2


@pytest.fixture(scope="module")
def nb():
    case = CASE_BY_NAME["NegativeBinomial"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return case.fit(case.data())


@pytest.fixture(scope="module")
def ew():
    case = CASE_BY_NAME["ExpoWeibull"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return case.fit(case.data())


def _deviance(model, theta):
    return 2.0 * (neg_ll_at(model, theta) - neg_ll_at(model, model.params))


def _brute_profile(model, name, value, points=7):
    """The profile deviance at ``name = value`` by many restarts: a grid
    of starts, 6 units either side of the fit on the log (logit) scale
    of each other parameter, each run by Nelder-Mead to convergence."""
    idx = model.dist.param_map[name]
    free = [j for j in range(len(model.params)) if j != idx]
    to_u = [
        (logit, expit) if model.dist.bounds[j] == (0, 1) else (np.log, np.exp)
        for j in free
    ]
    u_hat = np.array([f(model.params[j]) for j, (f, _) in zip(free, to_u)])

    def dev(u):
        theta = np.array(model.params, float)
        theta[idx] = value
        theta[free] = [g(v) for v, (_, g) in zip(u, to_u)]
        with np.errstate(all="ignore"):
            d = _deviance(model, theta)
        return d if np.isfinite(d) else 1e10

    best = np.inf
    grid = np.linspace(-6, 6, points)
    for shift in itertools.product(grid, repeat=len(free)):
        res = minimize(
            dev,
            u_hat + np.array(shift),
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-12, "maxfev": 3000},
        )
        best = min(best, res.fun)
    return best


# ---------------------------------------------------------------------------
# NegativeBinomial: r -> infinity is the shifted Poisson
# ---------------------------------------------------------------------------
def test_nb_profile_of_r_levels_off_at_the_shifted_poisson(nb):
    # As r -> inf with the mean held, the NegativeBinomial on {1, 2, ...}
    # tends to 1 + Poisson. Its best fit, computed here from scipy's
    # Poisson, has deviance 2.345 from the NegativeBinomial's: between
    # the 80% and the 95% critical values.
    d = nb.data
    x, c, n = (np.asarray(d[k], float) for k in ("x", "c", "n"))

    def nll(lam):
        y = x - 1.0
        ll = np.where(c == 0, poisson.logpmf(y, lam), poisson.logsf(y, lam))
        return -np.sum(n * ll)

    res = minimize_scalar(nll, bounds=(0.1, 20), method="bounded")
    nll_hat = neg_ll_at(nb, nb.params)
    limit = 2 * (res.fun - nll_hat)
    assert limit == pytest.approx(2.34507, abs=1e-4)
    assert CRIT_80 < limit < CRIT_95
    # The profile itself reaches it (it was 2.43 of rounding at 1e16).
    assert 2 * (nb._profile_neg_ll(0, 1e8) - nll_hat) == pytest.approx(
        limit, abs=1e-5
    )


def test_nb_upper_bound_on_r_is_infinite_where_the_limit_is_inside(nb):
    # 95% and 99%: the limit is inside the region, so no r is excluded.
    # These were 1.7e16 and 1.9e16, rounding noise.
    for alpha in (0.05, 0.01):
        lo, hi = nb.param_cb("r", alpha_ci=alpha, method="lr")
        assert np.isfinite(lo) and 0 < lo < nb.params[0]
        assert hi == np.inf
    # 80%: the critical value is below the limit, so the bound is where
    # the profile crosses it (checked by restarts).
    lo, hi = nb.param_cb("r", alpha_ci=0.2, method="lr")
    assert hi == pytest.approx(36.1983, rel=1e-4)
    assert _brute_profile(nb, "r", hi) == pytest.approx(CRIT_80, abs=1e-4)


def test_nb_upper_bound_on_p_is_one_at_95(nb):
    # p -> 1 is the same Poisson limit: the edge (was 0.99999686).
    lo, hi = nb.param_cb("p", method="lr")
    assert hi == 1.0
    assert _brute_profile(nb, "p", lo) == pytest.approx(CRIT_95, abs=1e-4)


def test_nb_bands_reach_the_poisson_valley(nb):
    # The band's extremes lie far out along the valley (r ~ 1e8): the
    # df(5) upper bound was 0.1909 and the sf(8) lower one 0.00918.
    df = nb.cb(5.0, on="df", method="lr")
    sf = nb.cb(8.0, on="sf", method="lr")
    assert df[1] == pytest.approx(0.194406, rel=1e-4)
    assert sf[0] == pytest.approx(0.0058723, rel=1e-4)
    # Each is attained at a point of the region: r large, p near 1.
    r = 1.4086584e8
    theta = [r, 0.999999973]
    assert _deviance(nb, theta) <= CRIT_95 + 1e-3
    assert nb.dist.df(np.array([5.0]), *theta)[0] > 0.1941


# ---------------------------------------------------------------------------
# ExpoWeibull: valleys to alpha -> 0 and to beta -> infinity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name, value, want",
    [
        # beta -> 0 with alpha -> 0 and mu -> inf: the profile the old
        # search put at 6.82 is 3.53
        ("beta", 0.1, 3.5307),
        # mu -> 0 with beta -> inf and alpha at max(x): 42.4 before
        ("mu", 1e-4, 0.28776),
        ("alpha", 1e-6, 3.19328),
    ],
)
def test_ew_profile_follows_its_valleys(ew, name, value, want):
    # A walk to the point, each step started from the one before, as
    # param_cb solves it; the value is that of 49 restarts of
    # Nelder-Mead over a grid of starts (_brute_profile), recorded.
    idx = ew.dist.param_map[name]
    coord = ew._lr_coords()[0][idx]
    path = likelihood_ratio._LRPath()
    nll_hat = neg_ll_at(ew, ew.params)
    for w in np.linspace(coord.to_u(ew.params[idx]), coord.to_u(value), 25):
        nll = ew._profile_neg_ll(idx, coord.from_u(w), path=path)
    assert 2 * (nll - nll_hat) == pytest.approx(want, abs=1e-3)


def test_ew_bounds_are_nested_across_levels(ew):
    # beta's upper bound was 367.5 at 95% but 372.0 at 80%; its profile
    # never reaches 1.64 (it peaks at 1.43 near beta = 20 and falls to
    # 0.27 as beta -> inf), so both are inf, and at 50% it is 4.906.
    ends = [
        ew.param_cb("beta", alpha_ci=a, method="lr") for a in (0.05, 0.2, 0.5)
    ]
    for wide, narrow in zip(ends[:-1], ends[1:]):
        assert wide[0] <= narrow[0] and narrow[1] <= wide[1]
    assert ends[0][1] == ends[1][1] == np.inf
    assert ends[2][1] == pytest.approx(4.90591, rel=1e-4)
    # mu's 80% lower bound was 0.0078, where its profile goes on falling
    # to 0.29 at 1e-4 (the same valley): 0.
    assert ew.param_cb("mu", alpha_ci=0.1, bound="lower", method="lr") == 0


def test_ew_alpha_lower_bound_is_far_down_its_valley(ew):
    # The profile of alpha rises slowly as alpha -> 0 (3.19 at 1e-6) and
    # crosses 3.84 at 5e-28: not 0, nor the 1.6e-8 the old search found.
    lo = ew.param_cb("alpha", bound="lower", alpha_ci=0.025, method="lr")
    assert 1e-29 < lo[0] < 1e-26
    assert _brute_profile(ew, "alpha", lo[0], points=5) == pytest.approx(
        CRIT_95, abs=2e-3
    )


def test_ew_band_reaches_the_power_function_valley(ew):
    # As beta -> inf with beta * mu = c and alpha at max(x), the model is
    # a power function; with c = 1.78 its deviance is 1.638 < 1.642, and
    # sf(13) there is 0.3797. The 80% band's upper bound was 0.3155.
    theta = [17.0, 1e4, 1.78e-4]
    assert _deviance(ew, theta) <= CRIT_80
    at = ew.dist.sf(np.array([13.0]), *theta)[0]
    assert at == pytest.approx(0.37967, abs=1e-4)
    upper = ew.cb(13.0, alpha_ci=0.2, method="lr")[1]
    assert upper >= at


def test_ew_99_band_contains_the_95_band(ew):
    # The 99% hf(13) lower bound was 0.1046, above the 95% one of 0.1017,
    # and the 99% qf(0.95) upper bound 36.44, below the 95% one of 40.37:
    # the searches stopped on local extremes of the long, curved region
    # (alpha -> 0, beta -> 0, mu -> inf), and none of the walks' points
    # they could start from lay in the far extreme's basin (#535). The
    # 95% extreme is a point of the 99% region, so the 99% bound must
    # reach past it.
    theta = [4.06915818e-04, 2.00959888e-01, 1.20416620e03]
    assert _deviance(ew, theta) <= CRIT_95
    at = ew.dist.hf(np.array([13.0]), *theta)[0]
    assert at == pytest.approx(0.101709, rel=1e-5)

    # One-sided at alpha / 2: the ends of the two-sided 95% and 99% bands.
    def end(f, a, x, **kw):
        return float(np.ravel(f(x, alpha_ci=a, method="lr", **kw))[0])

    lo = {
        a: end(ew.cb, a, 13.0, on="hf", bound="lower") for a in (0.025, 0.005)
    }
    hi = {
        a: end(ew.quantile_cb, a, 0.95, bound="upper") for a in (0.025, 0.005)
    }
    assert lo[0.025] == pytest.approx(at, rel=1e-4)
    assert hi[0.025] == pytest.approx(40.367, rel=1e-4)
    assert lo[0.005] < lo[0.025] and hi[0.005] > hi[0.025]
    # At 99% the region runs down the valley to alpha's edge, and the
    # extremes are approached only there, at the end of alpha's search
    # coordinate (alpha = 2.2e-308): the hazard's 0.07077, found by
    # tracing the region's two-parameter slices at 100 values of
    # log(alpha), and the quantile's extreme over that face (checked in
    # test_601_99_quantile_bound_is_the_extreme_on_alphas_face).
    assert 0.07077 <= lo[0.005] < 0.07077 * 1.01


# ---------------------------------------------------------------------------
# #601: bounds far down the ExpoWeibull's valleys, whatever the CPU
# ---------------------------------------------------------------------------
# The bounds below were found by searches that ran out of iterations on
# the region's long, flat valleys (beta -> inf with alpha at the largest
# observation; alpha -> 0), stopped inside the region, and depended on
# the BLAS thread count and CPU. Each is checked against the extreme
# found independently: with F(x) = P held, mu is solved for, and the
# least deviance is a one-dimensional search, in alpha with beta far down
# its valley (log beta = 40, where the deviance has reached its limit to
# 1e-10), or in beta with alpha at the end of its coordinate. The bound
# is that extreme to 1e-6 of it (the searches' tolerance: the extremality
# check looks 1e-6 beyond on the function's scale, and the valley narrows
# as 1 / beta), and beyond it only where the deviance there is within the
# slack the searches allow (1e-6). These run alike with one BLAS thread
# or several, and on the AVX2 path.
LOG_TINY = float(np.log(np.finfo(float).tiny))
CRIT_90 = z(0.95) ** 2
CRIT_99 = z(0.995) ** 2


def _ew_mu(log_alpha, log_beta, x, prob):
    """mu with F(x) = prob at alpha, beta: (1 - exp(-z))^mu = prob."""
    with np.errstate(all="ignore"):
        log_z = np.exp(log_beta) * (np.log(x) - log_alpha)
        if log_z < -30:
            # 1 - exp(-z) = z (1 - z / 2 + ...)
            log_g = log_z - 0.5 * np.exp(log_z)
        else:
            log_g = np.log(-np.expm1(-np.exp(log_z)))
            if log_g == 0.0:
                # 1 - exp(-z) rounds to 1: log(1 - e) = -e
                log_g = -np.exp(-np.exp(log_z))
        return np.log(prob) / log_g


def _ew_level_deviance(ew, x, prob, log_alpha=None, log_beta=None):
    """The least deviance with F(x) = prob, one of log alpha and log
    beta held and the other searched over a grid and refined."""
    grid = (
        np.linspace(2.0, 3.5, 61)
        if log_alpha is None
        else np.linspace(-12.0, 8.0, 81)
    )

    def dev(v):
        la, lb = (v, log_beta) if log_alpha is None else (log_alpha, v)
        mu = _ew_mu(la, lb, x, prob)
        if not (np.isfinite(mu) and mu > 0):
            return 1e10
        d = _deviance(ew, [np.exp(la), np.exp(lb), mu])
        return d if np.isfinite(d) else 1e10

    values = [dev(v) for v in grid]
    k = int(np.argmin(values))
    res = minimize_scalar(
        dev,
        bounds=(grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]),
        method="bounded",
        options={"xatol": 1e-13},
    )
    return min(values[k], res.fun)


def _ew_check(ew, bound, direction, crit, x_of, **held):
    """``bound`` against the extreme of the region's level F(x) = P,
    ``x_of(v) = (x, P)``, as the root near it of the least deviance less
    ``crit``: as far out as that root, to the searches' tolerance, and
    no further than the deviance's slack (``_LR_NOISE``) allows. Returns
    the extreme."""

    def excess(v):
        return _ew_level_deviance(ew, *x_of(v), **held) - crit

    # bracketed from the bound outwards
    near, far = bound * (1 - direction * 1e-3), bound * (1 + direction * 1e-3)
    while excess(far) < 0 and abs(far / bound - 1) < 0.5:
        near, far = far, bound + 2 * (far - bound)
    extreme = brentq(excess, near, far, xtol=1e-14, rtol=1e-14)
    # Short of it by no more than a search stopping in a valley that
    # flattens out to beta -> inf reaches: 4e-6 of qf(0.05) (a deviance
    # 1.3e-5 below the critical value) on one CI runner, 2e-7 locally,
    # against the 54% (0.849 for 0.552) of #601
    assert direction * (bound - extreme) >= -1e-5 * abs(extreme)
    if direction * (bound - extreme) > 0:
        assert excess(bound) <= 2 * likelihood_ratio._LR_NOISE
    return extreme


def test_601_band_bound_is_the_extreme_down_the_valley(ew):
    # The 95% sf(8) upper bound was 0.7758 with multi-threaded BLAS or
    # the AVX2 path, 0.79698 with one thread: its extreme is approached
    # as beta -> inf, where the profile deviance falls to its limit.
    upper = ew.cb(8.0, method="lr")[1]

    def level(s):
        return 8.0, 1.0 - s

    extreme = _ew_check(ew, upper, 1.0, CRIT_95, level, log_beta=40.0)
    # (log beta = 30 is already there)
    assert _ew_check(
        ew, upper, 1.0, CRIT_95, level, log_beta=30.0
    ) == pytest.approx(extreme, rel=1e-12)


def test_601_quantile_bounds_are_the_extremes_down_the_valley(ew):
    # The 95% qf(0.2) band was [2.72, 7.63], and the one-sided 95%
    # qf(0.05) and qf(0.2) lower bounds 0.849 and 3.24: the searches
    # stopped on the near side of the region, and its extremes lie far
    # down the valley to beta -> inf.
    band = np.ravel(ew.quantile_cb(0.2, method="lr"))
    lower = ew.quantile_cb([0.05, 0.2], bound="lower", method="lr")
    for bound, prob, crit, direction in (
        (band[0], 0.2, CRIT_95, -1.0),
        (band[1], 0.2, CRIT_95, 1.0),
        (lower[0], 0.05, CRIT_90, -1.0),
        (lower[1], 0.2, CRIT_90, -1.0),
    ):
        _ew_check(
            ew,
            bound,
            direction,
            crit,
            lambda q, prob=prob: (q, prob),
            log_beta=40.0,
        )


def test_601_99_quantile_bound_is_the_extreme_on_alphas_face(ew):
    # The 99% qf(0.95) upper bound varied from 80.6 to 86.2 with the BLAS
    # thread count and CPU: its extreme is on the face of the search box
    # where alpha is at the end of its coordinate (2.2e-308), and the
    # region's slices nearer it reach less far (log alpha = -600: 86.83).
    upper = ew.quantile_cb(0.95, alpha_ci=0.005, bound="upper", method="lr")

    def level(q):
        return q, 0.95

    extreme = _ew_check(ew, upper, 1.0, CRIT_99, level, log_alpha=LOG_TINY)
    nearer = brentq(
        lambda q: _ew_level_deviance(ew, q, 0.95, log_alpha=-600.0) - CRIT_99,
        85.0,
        88.0,
    )
    assert nearer < extreme * (1 - 1e-3)
    # and the 99% bound contains the 95% one
    assert upper > ew.quantile_cb(
        0.95, alpha_ci=0.025, bound="upper", method="lr"
    )


def test_601_unsettled_bound_warns_once_at_the_caller(monkeypatch):
    # A bound whose search was still moving out when its continuations
    # ran out is the most extreme point found, with one warning saying
    # so (principle: warn, don't refuse).
    model = fresh_conformance_fit("Weibull")
    expected = model.cb([4.0, 8.0], method="lr")
    model = fresh_conformance_fit("Weibull")
    far = likelihood_ratio._PsiBoundSearch.extreme_far

    def unsettled(self, *args, **kwargs):
        x = far(self, *args, **kwargs)
        self.converged = False
        return x

    monkeypatch.setattr(
        likelihood_ratio._PsiBoundSearch, "extreme_far", unsettled
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = model.cb([4.0, 8.0], method="lr")
    messages = [w for w in caught if "did not converge" in str(w.message)]
    assert len(messages) == 1
    assert messages[0].filename == __file__
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# Both
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "family, name, alpha", [("nb", "r", 0.05), ("ew", "mu", 0.2)]
)
def test_lr_bounds_raise_no_raw_warnings(family, name, alpha, request):
    # The NegativeBinomial's r bounds leaked 7000 numpy warnings ("divide
    # by zero encountered in log1p"), the ExpoWeibull's mu bounds 60
    # (principle 22).
    model = request.getfixturevalue(family)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.param_cb(name, alpha_ci=alpha, method="lr")
        model.cb([4.0, 8.0], on="hf", alpha_ci=alpha, method="lr")
    assert [str(w.message) for w in caught] == []


# ---------------------------------------------------------------------------
# The walk's levelling-off test (_lr_walk)
# ---------------------------------------------------------------------------
def _walk(deviance, crit, end=709.78):
    return likelihood_ratio._lr_walk(deviance, 0.0, 0.5, 1.0, crit, end)


def test_walk_reads_a_converging_profile_as_unbounded():
    # A profile tending to 2.345 is unbounded at 3.84 and crosses 1.64.
    def deviance(w):
        return 2.345 * (1 - np.exp(-w))

    assert _walk(deviance, CRIT_95) == ("edge", 709.78)
    status, w = _walk(deviance, CRIT_80)
    assert status == "root"
    assert deviance(w) == pytest.approx(CRIT_80, abs=1e-8)


@pytest.mark.parametrize(
    "deviance",
    [
        lambda w: (w / 50.0) ** 2,  # quadratic, with a large scale
        lambda w: 0.024 * w,  # rising slowly, at a steady rate
        lambda w: 3.0 + 0.9 * (1 - np.exp(-w / 200.0)),  # slow approach
    ],
)
def test_walk_finds_a_crossing_however_slowly_the_profile_rises(deviance):
    status, w = _walk(deviance, CRIT_95)
    assert status == "root"
    assert deviance(w) == pytest.approx(CRIT_95, abs=1e-8)


# ---------------------------------------------------------------------------
# #421: likelihood-ratio bands.
# ---------------------------------------------------------------------------


# -- #421: likelihood-ratio bands ------------------------------------------
def test_lr_band_does_not_stall_on_the_far_side_of_the_estimate():
    # A density at x peaks in the scale, and the warm-started search for
    # the lower df bound stopped on that peak: Rayleigh's 99% df band at
    # 14.6 was [0.0502, 0.0504], above the estimate 0.0359.
    model = fresh_conformance_fit("Rayleigh")
    x = np.array([3.2, 8.0, 14.6])
    df = model.df(x)
    for alpha in (0.01, 0.05, 0.2):
        cb = no_warnings(model.cb, x, on="df", alpha_ci=alpha, method="lr")
        assert np.all(cb[:, 0] <= df) and np.all(df <= cb[:, 1]), cb


def test_lr_band_of_one_parameter_is_the_extreme_over_its_interval():
    # With one free parameter the likelihood region is the profile
    # interval, and the band is the extreme of the function over it. The
    # search found one end or the other: Geometric's df(5) lower bound
    # was 0.0740 in a sweep over [2, 5, 8] and 0.0652 queried alone.
    model = fresh_conformance_fit("Geometric")
    x = np.array([2.0, 5.0, 8.0])
    for alpha in (0.05, 0.2):
        band = model.cb(x, on="df", alpha_ci=alpha, method="lr")
        lo, hi = model.param_cb("p", alpha_ci=alpha, method="lr")
        grid = np.linspace(lo, hi, 2001)
        want = np.array(
            [
                [f.min(), f.max()]
                for f in (surv.Geometric.df(k, grid) for k in x)
            ]
        )
        np.testing.assert_allclose(band, want, rtol=1e-5)
        for k in range(x.size):
            np.testing.assert_allclose(
                model.cb(x[k], on="df", alpha_ci=alpha, method="lr"),
                band[k],
                rtol=1e-8,
            )


def test_lr_band_of_the_uniform_follows_the_support_edge():
    # The likelihood is 0 once a > min(x) or b < max(x), a cliff the
    # constrained search could not follow: the 95% Hf band at 14.6 was
    # [1.533, 1.821], its lower end the 80% band's and its upper end the
    # estimate. The searches now keep a and b beyond the data's
    # extremes. A brute-force grid over (a, b) of the likelihood region
    # of the (exact, #460) fixture gives [1.337, 1.949] (95%) and
    # [1.578, 1.876] (80%).
    model = fresh_conformance_fit("Uniform")
    x = np.array([3.2, 8.0, 14.6])
    wide = no_warnings(model.cb, x, on="Hf", alpha_ci=0.05, method="lr")
    narrow = no_warnings(model.cb, x, on="Hf", alpha_ci=0.2, method="lr")
    np.testing.assert_allclose(wide[2], [1.337, 1.949], rtol=5e-3)
    np.testing.assert_allclose(narrow[2], [1.578, 1.876], rtol=5e-3)
    Hf = model.Hf(x)
    assert np.all(wide[:, 0] < narrow[:, 0]) and np.all(narrow[:, 0] < Hf)
    assert np.all(Hf < narrow[:, 1]) and np.all(narrow[:, 1] < wide[:, 1])
    # The profile bound on a ends at the smallest observation.
    lo, hi = model.param_cb("a", method="lr")
    assert lo < hi == model.params[0] == 2.411


# ---------------------------------------------------------------------------
# The band does not collapse onto the estimate; unsolved
# searches are nan with a warning; bounds after a restore.
# ---------------------------------------------------------------------------


W, E, G = surv.Weibull, surv.Exponential, surv.Geometric


def test_lr_band_does_not_collapse_onto_the_estimate():
    np.random.seed(1000)
    model = G.fit(G.random(30, 0.3))
    lower, upper = model.cb([2.0], method="lr")[0]
    estimate = float(model.sf(2.0))
    assert lower < estimate - 0.05 < estimate < upper
    # The band's lower sf is the sf at the parameter's upper LR bound
    p_hi = model.param_cb("p", method="lr")[1]
    assert lower == pytest.approx((1 - p_hi) ** 2, rel=1e-3)


def test_lr_band_is_nan_with_a_warning_when_every_search_fails(monkeypatch):
    np.random.seed(1)
    model = W.fit(W.random(30, 10, 3))

    class Failed:
        success = False
        x = np.array([np.nan, np.nan])
        fun = np.nan

    monkeypatch.setattr(
        _likelihood_ratio, "minimize", lambda *a, **k: Failed()
    )
    with pytest.warns(RuntimeWarning, match="could not be found"):
        band = model.cb([5.0, 10.0], method="lr")
    assert np.isnan(band).all()


def test_lr_param_bound_is_nan_with_a_warning_when_unsolved(monkeypatch):
    np.random.seed(1)
    model = W.fit(W.random(30, 10, 3))
    monkeypatch.setattr(
        model, "_profile_neg_ll", lambda idx, v, path=None: np.nan
    )
    with pytest.warns(RuntimeWarning, match="could not be found"):
        bound = model.param_cb("beta", method="lr")
    assert np.isnan(bound).all()


def test_lr_bounds_after_restoring_with_the_data():
    np.random.seed(1)
    model = W.fit(W.random(30, 10, 3))
    restored = surv.from_dict(model.to_dict(with_data=True))
    assert np.allclose(
        restored.cb([5.0, 10.0], method="lr"),
        model.cb([5.0, 10.0], method="lr"),
    )
    assert np.allclose(
        restored.param_cb("beta", method="lr"),
        model.param_cb("beta", method="lr"),
    )


def test_652_lr_hf_band_far_in_the_tail_contains_the_estimate():
    # The sf, ff and Hf bands are one band on the logit of sf, which has
    # no room past sf = 1e-308: the Hf band was [inf, inf] around Hf(1e6)
    # = 85302. There it is found on the scale of log Hf, the same extreme
    # of the same region.
    rng = np.random.default_rng(9)
    t = 500 * rng.weibull(1.8, 25)
    cen = rng.uniform(200, 900, 25)
    model = surv.Weibull.fit(np.minimum(t, cen), (t > cen).astype(int))
    x = np.array([300.0, 1e4, 1e6])
    band = no_warnings(model.cb, x, on="Hf", method="lr")
    Hf = model.Hf(x)
    assert np.all(band[:, 0] < Hf) and np.all(Hf < band[:, 1])
    assert np.all(np.isfinite(band))
    sf = model.cb([300.0], on="sf", method="lr")
    np.testing.assert_allclose(band[0], -np.log(sf[0, ::-1]), rtol=1e-10)
    upper = no_warnings(model.cb, x, on="Hf", method="lr", bound="upper")
    assert np.all(Hf < upper) and np.all(upper < band[:, 1])
    lower = no_warnings(model.cb, x, on="Hf", method="lr", bound="lower")
    assert np.all(band[:, 0] < lower) and np.all(lower < Hf)
