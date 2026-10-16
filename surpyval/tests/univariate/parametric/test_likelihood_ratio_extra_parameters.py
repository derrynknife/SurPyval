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
def _small_offset():
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
