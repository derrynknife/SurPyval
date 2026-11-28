"""Likelihood-ratio bounds of offset, limited-failure and zero-inflated
models (``ExtendedLikelihoodRatio``).

They used to raise ``NotImplementedError``. Each bound is checked against
a brute-force profile computed here from the distribution's likelihood
with scipy's optimisers (not the package's searches): the profile
deviance of a parameter's bound, and of a function's bound, is the
critical value. The closed edges of ``lfp_p`` (1) and ``f0`` (0), the
time before an estimated offset, the offset's unbounded corner and the
summaries of a limited-failure model are checked as documented.
"""

import pickle
import warnings

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import ndtri

import surpyval as sp

CRIT = ndtri(0.975) ** 2  # two-sided 95%


def _lfp_data(p=0.6, n=150, seed=1):
    rng = np.random.default_rng(seed)
    t = 10 * rng.weibull(2.0, n)
    t = np.where(rng.uniform(size=n) > p, np.inf, t)
    cens = rng.uniform(0, 30, n)
    return np.minimum(t, cens), (cens < t).astype(int)


def _zi_data(n=150, seed=2):
    rng = np.random.default_rng(seed)
    t = 10 * rng.weibull(2.0, n)
    t = np.where(rng.uniform(size=n) < 0.1, 0.0, t)
    cens = rng.uniform(0, 30, n)
    return np.minimum(t, cens), (cens < t).astype(int)


def _offset_data(n=150, seed=3):
    rng = np.random.default_rng(seed)
    t = 5 + 10 * rng.weibull(3.0, n)
    cens = 5 + rng.uniform(0, 25, n)
    return np.minimum(t, cens), (cens < t).astype(int)


def _fit(kind, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "lfp":
            x, c = _lfp_data(**kw)
            return sp.Weibull.fit(x, c=c, lfp=True)
        if kind == "zi":
            x, c = _zi_data(**kw)
            return sp.Weibull.fit(x, c=c, zi=True)
        x, c = _offset_data(**kw)
        return sp.Weibull.fit(x, c=c, offset=True)


def _vector(model):
    """The fitted parameters in ``param_map`` order, with a function
    giving the negative log-likelihood of such a vector."""
    names = sorted(model.param_map, key=model.param_map.__getitem__)
    values = []
    for name in names:
        if name in model.dist.param_map:
            values.append(model.params[model.dist.param_map[name]])
        else:
            values.append(model.extras[name])

    def nll(phi):
        d = dict(zip(names, phi))
        core = [d[n] for n in model.dist.parameter_names]
        return float(
            model.dist._neg_ll_func(
                model.surv_data,
                *core,
                d.get("gamma", 0.0),
                d.get("f0", 0.0),
                d.get("lfp_p", 1.0),
            )
        )

    return names, np.array(values, dtype=float), nll


def _brute_profile(model, constraint, start=None):
    """``min nll`` subject to ``constraint(phi) == 0``, by SLSQP from the
    estimate (and ``start``), in the parameters themselves: an
    independent check of the package's searches."""
    names, phi_hat, nll = _vector(model)
    nll_hat = nll(phi_hat)
    # Inside the parameters' spaces (and the offset below the data)
    box = [
        (
            None if lo is None else lo + 1e-12,
            None if hi is None else hi - 1e-12 * max(1.0, abs(hi)),
        )
        for lo, hi in model.bounds
    ]
    best = np.inf
    for x0 in [phi_hat] + ([] if start is None else [start]):
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                lambda v: nll(v) if np.isfinite(nll(v)) else 1e10,
                x0,
                method="SLSQP",
                bounds=box,
                constraints=[{"type": "eq", "fun": constraint}],
                options={"ftol": 1e-12, "maxiter": 500},
            )
        if res.success and abs(constraint(res.x)) < 1e-8:
            best = min(best, float(res.fun))
    return 2.0 * (best - nll_hat)


# ---------------------------------------------------------------------------
# Parameters: the profile deviance at each bound is the critical value
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kind, name",
    [
        ("lfp", "lfp_p"),
        ("lfp", "alpha"),
        ("zi", "f0"),
        ("zi", "beta"),
        ("offset", "gamma"),
        ("offset", "beta"),
    ],
)
def test_parameter_bounds_are_where_the_profile_reaches_the_critical_value(
    kind, name
):
    model = _fit(kind)
    lower, upper = model.param_cb(name, method="lr")
    names, phi_hat, _ = _vector(model)
    j = names.index(name)
    assert lower < phi_hat[j] < upper
    for bound in (lower, upper):
        start = phi_hat.copy()
        start[j] = bound
        dev = _brute_profile(model, lambda v: v[j] - bound, start)
        assert dev == pytest.approx(CRIT, abs=2e-3), (name, bound, dev)


def test_the_extra_parameters_bounds_differ_from_wald_but_agree_roughly():
    model = _fit("lfp")
    wald = model.param_cb("lfp_p")
    lr = model.param_cb("lfp_p", method="lr")
    assert lr == pytest.approx(wald, abs=0.05)
    assert not np.allclose(lr, wald, rtol=1e-3)


def test_the_core_bounds_include_the_extra_parameters_uncertainty():
    # Profiling lfp_p as well widens alpha's interval: the plain model's
    # region is the slice of the LFP model's at its estimate
    model = _fit("lfp")
    lr = model.param_cb("alpha", method="lr")
    held = sp.Weibull.fit(
        *_lfp_data(), lfp=True, fixed={"lfp_p": float(model.lfp_p)}
    )
    lr_held = held.param_cb("alpha", method="lr")
    assert lr[0] < lr_held[0] and lr_held[1] < lr[1]


# ---------------------------------------------------------------------------
# The closed edges: lfp_p = 1 and f0 = 0
# ---------------------------------------------------------------------------
def _lfp_on_edge():
    # Every unit can fail (p = 1): the fit's p is at, or a hair below, 1
    x, c = _lfp_data(p=1.0, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.Weibull.fit(x, c=c, lfp=True)


def test_an_interval_reaching_the_edge_is_the_edge():
    model = _lfp_on_edge()
    assert model.lfp_p > 0.999
    lower, upper = model.param_cb("lfp_p", method="lr")
    # The plain model, p = 1, is not excluded: the edge exactly
    assert upper == 1.0
    names, phi_hat, _ = _vector(model)
    j = names.index("lfp_p")
    start = phi_hat.copy()
    start[j] = lower
    dev = _brute_profile(model, lambda v: v[j] - lower, start)
    assert dev == pytest.approx(CRIT, abs=2e-3)


def test_a_one_sided_bound_from_the_edge_is_the_two_sided_end():
    model = _lfp_on_edge()
    one = model.param_cb("lfp_p", alpha_ci=0.05, bound="lower", method="lr")
    two = model.param_cb("lfp_p", alpha_ci=0.10, method="lr")
    assert one[0] == pytest.approx(two[0], rel=1e-8)
    assert model.param_cb("lfp_p", bound="upper", method="lr")[0] == 1.0


def test_an_interval_away_from_the_edge_does_not_reach_it():
    model = _fit("lfp")  # p = 0.6
    upper = model.param_cb("lfp_p", method="lr")[1]
    assert upper < 0.85


def test_f0_reaches_its_edge_without_zeros_in_the_data():
    # No unit dead on arrival: f0's interval starts at the edge, 0
    x, c = _zi_data()
    x = np.where(x == 0, 0.5, x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.Weibull.fit(x, c=c, zi=True)
    lower, upper = model.param_cb("f0", method="lr")
    assert lower == 0.0
    names, phi_hat, _ = _vector(model)
    j = names.index("f0")
    start = phi_hat.copy()
    start[j] = upper
    dev = _brute_profile(model, lambda v: v[j] - upper, start)
    assert dev == pytest.approx(CRIT, abs=2e-3)


# ---------------------------------------------------------------------------
# Functions: the bound is the extreme over the region
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kind, t", [("lfp", 8.0), ("zi", 8.0), ("offset", 12.0)]
)
def test_sf_band_ends_are_where_the_profile_reaches_the_critical_value(
    kind, t
):
    model = _fit(kind)
    lower, upper = model.cb(t, method="lr")
    assert lower < model.sf(t) < upper
    names, phi_hat, _ = _vector(model)

    def sf_of(phi):
        d = dict(zip(names, phi))
        kw = {k: d[k] for k in ("gamma", "lfp_p", "f0") if k in d}
        core = [d[n] for n in model.dist.parameter_names]
        return sp.Weibull.from_params(core, **kw).sf(t)

    for bound in (lower, upper):
        dev = _brute_profile(model, lambda v: (sf_of(v) - bound) * 100)
        assert dev == pytest.approx(CRIT, abs=5e-3), (bound, dev)


def test_the_lfp_band_holds_the_survivors_that_never_fail():
    model = _fit("lfp")
    band = model.cb([30.0, 100.0], method="lr")
    lower_p, upper_p = model.param_cb("lfp_p", method="lr")
    # Far out the survival is 1 - p: its band is 1 - p's interval
    assert band[1] == pytest.approx([1 - upper_p, 1 - lower_p], abs=1e-4)


def test_a_time_before_the_estimated_offset_has_a_band_below_1():
    model = _fit("offset")
    gamma_lo = model.param_cb("gamma", alpha_ci=0.2, method="lr")[0]
    t = 0.5 * (gamma_lo + model.gamma)
    assert model.sf(t) == 1.0
    lower, upper = model.cb(t, alpha_ci=0.2, method="lr")
    assert lower < 1.0 and upper == 1.0
    # Before every offset in the region, the survival is 1 throughout
    early = model.cb(gamma_lo - 1.0, alpha_ci=0.2, method="lr")
    np.testing.assert_array_equal(early, [1.0, 1.0])
    # and the band at a narrower level is inside
    narrow = model.cb(t, alpha_ci=0.5, method="lr")
    assert lower <= narrow[0]


def test_quantile_and_mean_bounds():
    model = _fit("zi")
    q = model.quantile_cb([0.05, 0.5], method="lr")
    # f0 is about 0.1: the 5% quantile is 0 (dead on arrival) at the
    # estimate and its lower bound is that edge
    assert q[0, 0] == 0.0
    assert q[1, 0] < model.qf(0.5) < q[1, 1]
    m = model.mean_cb(method="lr")
    assert m[0] < model.mean() < m[1]
    mrl = model.mrl_cb(5.0, method="lr")
    assert mrl[0] < model.mrl(5.0) < mrl[1]


def test_lfp_quantile_above_the_proportion_is_infinite():
    model = _fit("lfp")
    lower_p = model.param_cb("lfp_p", method="lr")[0]
    q = model.quantile_cb([0.1, lower_p + 0.02], method="lr")
    assert np.all(np.isfinite(q[0]))
    assert np.isfinite(q[1, 0]) and q[1, 1] == np.inf


def test_lfp_mean_is_infinite_and_its_lower_bound_is_on_the_edge():
    model = _fit("lfp")
    np.testing.assert_array_equal(model.mean_cb(method="lr"), [np.inf] * 2)
    # With the proportion at 1 the mean is finite there, and its lower
    # bound is the least over the region where it stays at 1: the
    # region holds p < 1, where the mean is infinite
    edge = _lfp_on_edge()
    edge.lfp_p = 1.0
    lower, upper = edge.mean_cb(method="lr")
    assert upper == np.inf and lower < edge.mean()
    # That slice is the plain model's region (its maximum is the same)
    x, c = _lfp_data(p=1.0, seed=2)
    plain = sp.Weibull.fit(x, c=c)
    assert lower == pytest.approx(plain.mean_cb(method="lr")[0], rel=1e-3)


# ---------------------------------------------------------------------------
# The offset's corner, where the likelihood is unbounded
# ---------------------------------------------------------------------------
def _small_offset_data():
    # Twelve values: at 95% the region reaches the offset at the first
    # failure, where a shape below 1 makes the likelihood infinite
    x = np.array(
        [
            7.411,
            8.84,
            9.956,
            10.953,
            11.903,
            12.846,
            13.816,
            14.85,
            15.997,
            17.347,
            19.096,
            21.954,
        ]
    )
    c = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0])
    n = np.array([1, 1, 2, 1, 1, 1, 1, 3, 1, 1, 1, 1])
    return x, c, n


def _small_offset():
    x, c, n = _small_offset_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.Weibull.fit(x, c=c, n=n, offset=True)


def test_a_region_reaching_the_unbounded_corner_gives_nan_and_says_so():
    model = _small_offset()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        band = model.cb([9.0, 13.0], method="lr")
    assert np.isnan(band).all()
    assert len(caught) == 1
    assert "unbounded" in str(caught[0].message)
    assert caught[0].filename == __file__
    # The offset's own interval runs up to the first failure
    assert model.param_cb("gamma", method="lr")[1] == pytest.approx(7.411)
    # At 80% the region stops short of it, and the band is found
    assert np.isfinite(model.cb([9.0, 13.0], alpha_ci=0.2, method="lr")).all()


def _small_gamma_offset():
    x, c, n = _small_offset_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.Gamma.fit(x, c=c, n=n, offset=True)


@pytest.mark.parametrize("alpha_ci", [0.01, 0.05, 0.2])
def test_a_side_with_unbounded_likelihood_points_is_the_edge(alpha_ci):
    # The region reaches the corner at each level, and holds every point
    # with a shape below 1 and the offset a hair below the first failure,
    # where the likelihood is unbounded: a Gamma alpha below 1, and any
    # beta. The profile walk stopped at whichever wall the refused
    # likelihood above the maximum left it, which changed with the last
    # bits of the arithmetic (a 95% alpha bound of 0.081 and an 80% one of
    # 0 on Linux x86-64).
    model = _small_gamma_offset()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alpha = model.param_cb("alpha", alpha_ci=alpha_ci, method="lr")
        beta = model.param_cb("beta", alpha_ci=alpha_ci, method="lr")
    assert alpha[0] == 0.0
    assert alpha[1] > model.params[0]
    assert list(beta) == [0.0, np.inf]


def test_flat_gamma_offset_bounds_do_not_depend_on_rounding():
    # The profile of the Gamma offset model's alpha tends to the Normal
    # limit, whose deviance (0.854) is below the 80% two-sided critical
    # value (1.64): the upper bound is infinite at every level. Its
    # offset's lower side runs the same way, to the same limit: the
    # lower bound is minus infinity at every level, with no warning. The
    # searches lost the profile's narrow valley there, and returned a
    # finite lower bound (-3.5e6 at 99%, -4.6e5 at 80%) (#1417).
    model = _small_gamma_offset()
    for alpha_ci in (0.01, 0.05, 0.2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            alpha = model.param_cb("alpha", alpha_ci=alpha_ci, method="lr")
            gamma = model.param_cb("gamma", alpha_ci=alpha_ci, method="lr")
        assert alpha[1] == np.inf
        assert gamma[0] == -np.inf
        assert gamma[1] == pytest.approx(7.411)
        assert [str(w.message) for w in caught] == []


def _normal_limit_deviance(x, c, n):
    """The deviance of the Normal fit against the Gamma offset fit: the
    limit of the Gamma offset's profile deviance as the offset goes to
    minus infinity (the shifted Gamma tends to the Normal)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gamma = sp.Gamma.fit(x, c=c, n=n, offset=True)
        normal = sp.Normal.fit(x, c=c, n=n)
    return 2.0 * (normal.neg_ll() - gamma.neg_ll())


def test_the_gamma_offset_profile_follows_its_valley_to_the_normal_limit():
    # Far out, the profile minimum over alpha and beta lies along a
    # valley about sd / (mean - offset) wide; the searches stopped on its
    # wall (a deviance of 8.9 at an offset of -1e4, 93 at -1e6). Followed
    # down it, the deviance rises towards the Normal limit, 0.854, and
    # stays below it.
    x, c, n = _small_offset_data()
    limit = _normal_limit_deviance(x, c, n)
    assert limit == pytest.approx(0.854, abs=1e-3)
    model = _small_gamma_offset()
    searches = model._lr_extended()
    nll_hat = searches._lr_neg_ll(np.asarray(searches.params, dtype=float))
    j = searches._gamma_idx
    devs = []
    for offset in (-1e2, -1e4, -1e6):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nll = searches._profile_neg_ll(j, offset)
        devs.append(2.0 * (nll - nll_hat))
    assert devs[0] < devs[1] < devs[2] < limit + 1e-6
    assert devs[2] == pytest.approx(limit, abs=1e-3)


def test_a_gamma_offset_whose_normal_limit_is_excluded_has_a_finite_bound():
    # The other way round: data far from Normal (a strong right skew)
    # have a Normal limit far above the critical value, and the offset's
    # lower bound is a finite crossing the profile deviance reaches.
    rng = np.random.default_rng(7)
    x = 3.0 + rng.gamma(1.5, 4.0, 60)
    c = np.zeros_like(x)
    n = np.ones_like(x)
    assert _normal_limit_deviance(x, c, n) > 10.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.Gamma.fit(x, offset=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lower, upper = model.param_cb("gamma", alpha_ci=0.05, method="lr")
    assert [str(w.message) for w in caught] == []
    assert -np.inf < lower < model.gamma < upper <= x.min()
    # The bound is where the profile deviance is the critical value
    searches = model._lr_extended()
    nll_hat = searches._lr_neg_ll(np.asarray(searches.params, dtype=float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nll = searches._profile_neg_ll(searches._gamma_idx, lower)
    assert 2.0 * (nll - nll_hat) == pytest.approx(3.841459, abs=1e-3)


def _limit_deviance(family, x, c=None, n=None):
    """The deviance of ``family``'s offset limit (the smallest extreme
    value distribution for a Weibull, the Logistic for a LogLogistic,
    the Normal for a Gamma or a LogNormal) against its offset fit."""
    limit = {
        "Weibull": sp.Gumbel,
        "LogLogistic": sp.Logistic,
        "Gamma": sp.Normal,
        "LogNormal": sp.Normal,
    }[family]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = getattr(sp, family).fit(x, c=c, n=n, offset=True)
        other = limit.fit(x, c=c, n=n)
    return model, 2.0 * (other.neg_ll() - model.neg_ll())


def _lower(model, alpha_ci):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lower = model.param_cb(
            "gamma", alpha_ci=alpha_ci, bound="lower", method="lr"
        )[0]
    return lower, [str(w.message) for w in caught]


@pytest.mark.parametrize(
    "family, limit, alpha_ci",
    [("Weibull", 4.164, 0.01), ("LogLogistic", 0.616, 0.2)],
)
def test_an_offset_whose_limit_is_inside_the_region_is_minus_infinity(
    family, limit, alpha_ci
):
    # The 12-unit data's offset profile rises towards its limit's
    # deviance (a Weibull's Gumbel, 4.164; a LogLogistic's Logistic,
    # 0.616), below the one-sided critical value (5.41 at 99%, 0.708 at
    # 80%): no offset down to -inf is excluded. The searches lose the
    # valley far out (a Weibull's shape and scale run to infinity
    # together), and the bound was nan with a warning.
    x, c, n = _small_offset_data()
    model, d_lim = _limit_deviance(family, x, c, n)
    assert d_lim == pytest.approx(limit, abs=1e-3)
    assert ndtri(1 - alpha_ci) ** 2 > d_lim
    assert _lower(model, alpha_ci) == (-np.inf, [])


def test_an_offset_whose_limit_is_outside_the_region_keeps_its_crossing():
    # At 95% one-sided (2.71) the Weibull's Gumbel limit, 4.164, is
    # excluded, and the bound is the crossing the profile reaches, as
    # before: -16.39, where the profile deviance is the critical value.
    x, c, n = _small_offset_data()
    model, _ = _limit_deviance("Weibull", x, c, n)
    lower, caught = _lower(model, 0.05)
    assert caught == []
    assert lower == pytest.approx(-16.39, abs=0.01)
    searches = model._lr_extended()
    nll_hat = searches._lr_neg_ll(np.asarray(searches.params, dtype=float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nll = searches._profile_neg_ll(searches._gamma_idx, lower)
    assert 2.0 * (nll - nll_hat) == pytest.approx(ndtri(0.95) ** 2, abs=1e-3)


def _near_normal(seed):
    rng = np.random.default_rng(seed)
    return 20.0 + 2.0 * rng.standard_normal(40)


def test_a_lognormal_offset_profile_falls_to_its_better_normal_limit():
    # The reference for the next test, in closed form: with the offset
    # held, a LogNormal's maximum-likelihood mu and sigma are the mean
    # and standard deviation of log(x - gamma). On 40 near-Normal values
    # the profile deviance falls from the fitted offset (-68.1) to the
    # Normal's, -0.375, without rising again: the Normal fits better
    # than any offset, and the likelihood's supremum is at gamma = -inf.
    from scipy import stats

    x = _near_normal(1)
    model, d_lim = _limit_deviance("LogNormal", x)
    # Where the offset search stops on this flat valley (it is not a
    # maximum) depends on the platform's last digits: the deviance from
    # the limit is -0.375 on arm64 macOS and -0.385 to -0.389 on x86-64
    # Linux. What holds everywhere is its sign: the limit fits better.
    assert d_lim < 0

    def nll(g):
        logs = np.log(x - g)
        return -stats.lognorm.logpdf(
            x - g, logs.std(), scale=np.exp(logs.mean())
        ).sum()

    offsets = -np.logspace(2, 8, 25)
    devs = np.array([2.0 * (nll(g) - model.neg_ll()) for g in offsets])
    assert np.all(np.diff(devs) <= 1e-9)
    assert devs[-1] == pytest.approx(d_lim, abs=1e-3)


@pytest.mark.parametrize(
    "family, seed, limit",
    [
        ("Gamma", 1, -0.498),
        ("LogNormal", 1, -0.375),
        ("LogNormal", 2, -0.341),
        ("LogLogistic", 4, -0.005),
    ],
)
def test_an_offset_fit_beaten_by_its_limit_has_a_lower_bound_of_minus_inf(
    family, seed, limit
):
    # Near-Normal data whose offset fit is not the maximum: the limit
    # (the Normal, the Logistic) fits better, so no offset down to -inf
    # is excluded. The searches could not settle the profile far out,
    # and gave nan with a warning, or read noise as a crossing (a
    # LogNormal, seed 2, -3.4e14 at 99%; a LogLogistic, seed 4, -6.0e14
    # at 99% and 95%).
    model, d_lim = _limit_deviance(family, _near_normal(seed))
    # ``limit`` is the deviance on arm64 macOS; where the offset search
    # stops on the flat valley depends on the platform's last digits
    # (-0.389 for LogNormal seed 1 and -0.515 for Gamma on x86-64 Linux),
    # so only its sign is checked: the limit fits better.
    assert d_lim < 0 and limit < 0
    for alpha_ci in (0.01, 0.05, 0.2):
        assert _lower(model, alpha_ci) == (-np.inf, [])


def test_a_side_without_unbounded_likelihood_points_is_searched():
    # A Weibull shape above 1 has a density of 0 at the offset: its upper
    # side is walked to the critical value, while its lower side and both
    # sides of the scale (a shape below 1 at any scale) are the edges.
    model = _small_offset()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        beta = model.param_cb("beta", method="lr")
        alpha = model.param_cb("alpha", method="lr")
    assert beta[0] == 0.0
    assert model.params[1] < beta[1] < np.inf
    assert list(alpha) == [0.0, np.inf]


def test_an_exponential_offset_on_the_first_failure_is_a_genuine_maximum():
    # Its density is finite and positive at the offset: no corner
    rng = np.random.default_rng(4)
    x = 5 + rng.exponential(10, 60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.Exponential.fit(x, offset=True)
        band = model.cb([8.0, 20.0], method="lr")
        gamma = model.param_cb("gamma", method="lr")
    assert np.isfinite(band).all()
    assert gamma[0] < model.gamma <= gamma[1] <= x.min()


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
def test_a_fixed_parameter_is_held_in_the_profiles():
    x, c = _lfp_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.Weibull.fit(x, c=c, lfp=True, fixed={"beta": 2.0})
    np.testing.assert_array_equal(
        model.param_cb("beta", method="lr"), [2.0, 2.0]
    )
    lower, upper = model.param_cb("lfp_p", method="lr")
    assert lower < model.lfp_p < upper


def test_an_unknown_name_still_names_the_valid_ones():
    model = _fit("lfp")
    with pytest.raises(ValueError, match="lfp_p"):
        model.param_cb("p0", method="lr")


def test_a_model_pickles_after_its_searches():
    model = _fit("zi")
    band = model.cb(8.0, method="lr")
    again = pickle.loads(pickle.dumps(model))
    assert not any(k.startswith("_lr_") for k in again.__dict__)
    np.testing.assert_allclose(again.cb(8.0, method="lr"), band, rtol=1e-6)


_LIMIT = {
    "LogNormal": sp.Normal,
    "Gamma": sp.Normal,
    "LogLogistic": sp.Logistic,
}


@pytest.mark.parametrize("family", ["LogNormal", "Gamma", "LogLogistic"])
def test_an_offset_fit_beaten_by_its_limit_says_so_at_fit_time(family):
    # The fit stops at a point that is not a maximum (a LogNormal at
    # gamma = -68.1): it warns that the data do not support an offset,
    # names the limit to fit instead, and says the likelihood-ratio
    # bounds are measured from that limit's fit.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(sp, family).fit(_near_normal(1), offset=True)
    limit = _LIMIT[family].name
    messages = [str(w.message) for w in caught]
    assert len(messages) == 1
    assert messages[0].startswith("No finite maximum")
    assert "the data do not support an offset" in messages[0]
    assert f"fit surpyval.{limit} instead" in messages[0]
    assert f"measured from the {limit} fit" in messages[0]


@pytest.mark.parametrize(
    "family, seed, upper, d_hat",
    [
        # (before: 3.233, 8.469, 14.58, measured from the fit itself)
        ("LogNormal", 1, 1.7068, 0.3748),
        ("Gamma", 1, 7.1814, 0.4984),
        ("LogLogistic", 1, 7.3489, 0.00262),
    ],
)
def test_an_offset_fit_beaten_by_its_limit_is_bounded_from_the_limit(
    family, seed, upper, d_hat
):
    # A likelihood-ratio region is the parameters whose likelihood is
    # within crit / 2 of the best the family can do. Where the fit is
    # not the maximum, the best is its limit (the Normal, the Logistic),
    # and the offset's upper bound is where the profile deviance from
    # the limit's fit, not from the fit, is the critical value.
    x = _near_normal(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = getattr(sp, family).fit(x, offset=True)
        limit = _LIMIT[family].fit(x)
    searches = model._lr_extended()
    assert searches._lr_reference_nll() == pytest.approx(limit.neg_ll())
    # The fit's deviance from its limit (``d_hat`` on arm64 macOS) depends
    # on where the offset search stopped, which is platform-dependent (0.389
    # for LogNormal and 0.515 for Gamma on x86-64 Linux); the reference is
    # that deviance by definition, wherever the search stopped.
    assert d_hat > 0
    assert searches._lr_reference_deviance() == pytest.approx(
        2.0 * (model.neg_ll() - limit.neg_ll()), rel=1e-12
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bound = model.param_cb(
            "gamma", alpha_ci=0.05, bound="upper", method="lr"
        )[-1]
        nll = searches._profile_neg_ll(searches._gamma_idx, bound)
    assert [str(w.message) for w in caught] == []
    assert bound == pytest.approx(upper, abs=2e-3)
    assert 2.0 * (nll - limit.neg_ll()) == pytest.approx(
        ndtri(0.95) ** 2, abs=1e-3
    )


def test_an_offset_lognormal_beaten_by_the_normal_has_its_bounds():
    # Measured from the fit (not a maximum), mu's upper and sigma's lower
    # bounds were nan with a warning, the profile too flat out along the
    # valley to the Normal. From the Normal's fit they are the ends the
    # region reaches, mu running to inf and sigma to 0 as gamma runs to
    # -inf; and a function's bound is the Normal's own (the region holds
    # LogNormals as close to that Normal as you like): sf(18) at 90%,
    # 0.7745 to 0.9204, was 0.7680 to 0.9232.
    x = _near_normal(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.LogNormal.fit(x, offset=True)
        normal = sp.Normal.fit(x)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mu = model.param_cb("mu", alpha_ci=0.1, method="lr")
        sigma = model.param_cb("sigma", alpha_ci=0.1, method="lr")
        band = model.cb(18.0, on="sf", alpha_ci=0.1, method="lr")
    assert [str(w.message) for w in caught] == []
    assert mu[0] == pytest.approx(2.9011, abs=1e-3)
    assert mu[1] == np.inf
    assert sigma[0] == 0.0
    assert sigma[1] == pytest.approx(0.10696, abs=1e-4)
    np.testing.assert_allclose(
        band, normal.cb(18.0, on="sf", alpha_ci=0.1, method="lr"), atol=1e-4
    )


def test_a_supported_offset_is_bounded_from_its_own_fit():
    # Where the offset fit beats its limit (a Weibull's Gumbel, 4.164
    # worse), the deviances are measured from the fit as before.
    x, c, n = _small_offset_data()
    model, _ = _limit_deviance("Weibull", x, c, n)
    searches = model._lr_extended()
    nll_hat = searches._lr_neg_ll(np.asarray(searches.params, dtype=float))
    assert searches._lr_reference_nll() == nll_hat
    assert searches._lr_reference_deviance() == 0.0


def test_a_bound_whose_level_excludes_the_fit_is_nan_with_a_warning():
    # At 60% one-sided (critical value 0.064) the LogNormal fit, 0.375
    # from the Normal's, is outside the region itself: there is no
    # stretch around it to bound, and the bound is nan with a warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.LogNormal.fit(_near_normal(1), offset=True)
    with pytest.warns(RuntimeWarning, match="not the maximum"):
        bound = model.param_cb(
            "gamma", alpha_ci=0.4, bound="upper", method="lr"
        )[-1]
    assert np.isnan(bound)
