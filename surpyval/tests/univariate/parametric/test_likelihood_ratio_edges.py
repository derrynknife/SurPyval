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
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from scipy.special import ndtri as z
from scipy.stats import poisson

import surpyval.univariate.parametric.parametric as parametric_module
from surpyval.tests.conformance.registry import CASE_BY_NAME

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


def _nll(model, theta):
    with np.errstate(all="ignore"):
        return float(
            model.dist._neg_ll_func(
                model.surv_data, *theta, model.gamma, model.f0, model.p
            )
        )


def _deviance(model, theta):
    return 2.0 * (_nll(model, theta) - _nll(model, model.params))


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
    nll_hat = _nll(nb, nb.params)
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
    path = parametric_module._LRPath()
    nll_hat = _nll(ew, ew.params)
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
    return parametric_module._lr_walk(deviance, 0.0, 0.5, 1.0, crit, end)


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
