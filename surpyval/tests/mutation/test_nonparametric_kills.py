"""Checks that kill the mutants that survived the non-parametric test
suite (#396).

A mutation run (``scripts/mutation/run.sh nonparametric``) changes
``surpyval/univariate/nonparametric/nonparametric.py`` and the three
estimator modules one small edit at a time and reruns the tests; an edit
no test notices is a check the suite lacks. Each test here names the
mutants it kills (file:line at commit 85ed4c5, the one mutated, then the
edit) and, where it is a general property, the conformance check it
should become. The data are small and the tests fast: they run in the
normal suite.

Queries are compared through ``np.ravel`` so the tests hold whether a
scalar query returns a scalar or a length-one array.
"""

import importlib
import json
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402
from scipy.stats import norm  # noqa: E402

import surpyval as sp  # noqa: E402
from surpyval import NonParametric  # noqa: E402
from surpyval.univariate import nonparametric as nonp  # noqa: E402

# The module; the package's name 'fleming_harrington' is the function.
fh = importlib.import_module(
    "surpyval.univariate.nonparametric.fleming_harrington"
)

# Ten items, three of them censored (c=1), the last one a failure.
X = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
C = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
ESTIMATORS = ("KaplanMeier", "NelsonAalen", "FlemingHarrington")


def _fit(name="KaplanMeier", x=X, c=C, **kw):
    return getattr(sp, name).fit(x, c=c, **kw)


def _flat(a):
    return np.ravel(np.asarray(a, dtype=float))


def _truncated_sample():
    # 60 items, a quarter censored, half of them entering late (tl > 0):
    # enough that a bootstrap's median sits within 0.01 of the estimate,
    # and truncation moves the estimate by 0.05 to 0.1.
    rng = np.random.default_rng(1)
    x = np.round(rng.weibull(1.5, 60) * 10, 2) + 0.01
    tl = np.where(rng.random(60) < 0.5, np.round(x * rng.random(60), 2), 0.0)
    c = (rng.random(60) < 0.25).astype(int)
    return x, c, tl


# --- the functions of a model agree with each other (principle 8) ------------


@pytest.mark.parametrize("name", ESTIMATORS)
@pytest.mark.parametrize("interp", ["step", "linear"])
def test_density_is_the_drop_over_the_hazard_step(name, interp):
    # df is the drop in sf over the step whose increment hf gives (#408):
    # where that is the step from the previous point, df = sf(q) (e^hf - 1)
    # exactly, the discrete form of df = hf * sf. Kills a wrong sign or
    # operator in either, and either 'interp=interp' dropped.
    model = _fit(name)
    q = np.array([1.5, 2.5, 3.5, 5, 6.2, 8, 9.5])
    hf = model.hf(q, interp=interp)
    own = np.append(False, np.diff(model.Hf(q, interp=interp)) > 0)
    assert own[1:].sum() >= 4
    assert_allclose(
        model.df(q, interp=interp)[own],
        (model.sf(q, interp=interp) * np.expm1(hf))[own],
        rtol=1e-12,
    )


@pytest.mark.parametrize("name", ESTIMATORS)
def test_stored_ladder_agrees_with_the_functions(name):
    # The arrays a model stores (F, H) are the functions at its own times;
    # qf and smoothed_hf read them directly. A conformance property for
    # every non-parametric model ("ladder_matches_functions").
    model = _fit(name)
    assert_allclose(model.F, _flat(model.ff(model.x)), rtol=1e-12)
    assert_allclose(model.H, _flat(model.Hf(model.x)), rtol=1e-12)


def test_fit_from_ecdf_ladder_agrees_with_the_functions():
    # Kills nonparametric.py:2107-2112: the model name, F = 1 - R
    # (mutated to None, 1 + R, 2 - R) and H = -log R (None, +log R).
    x, R = [1.0, 2, 3, 4], [0.9, 0.6, 0.3, 0.1]
    model = NonParametric.fit_from_ecdf(x, R)
    assert model.model == "from_ecdf"
    assert_allclose(model.F, 1 - np.array(R), rtol=1e-12)
    assert_allclose(model.H, -np.log(R), rtol=1e-12)
    # qf inverts ff at the steps.
    assert_allclose(_flat(model.qf(model.ff(x))), x)
    assert np.all(np.isfinite(model.smoothed_hf([2, 3], bandwidth=1.5)))


@pytest.mark.parametrize(
    "x, R",
    [
        ([5.0], [0.5]),  # a single point
        ([1.0, 2, 2, 3], [0.9, 0.7, 0.6, 0.2]),  # a repeated value
        ([1.0, 2, 3], [0.8, 0.8, 0.5]),  # a flat stretch
        ([1.0, 2, 3], [1.0, 0.5, 0.0]),  # reaching 1 and 0
    ],
)
def test_fit_from_ecdf_accepts_its_documented_edges(x, R):
    # The docstring allows a repeated x, a non-increasing (so possibly
    # flat) R, and R within [0, 1] inclusive. Kills nonparametric.py:2091
    # 'size == 0' -> '== 1', :2097 '< 0' -> '<= 0' / '< 1', :2099
    # '> 0' -> '>= 0' and :2103 '< 0' -> '<= 0', '> 1' -> '>= 1'.
    model = NonParametric.fit_from_ecdf(x, R)
    # At a repeated value the curve has settled on the last R given there.
    last = np.searchsorted(x, x, side="right") - 1
    assert_allclose(_flat(model.sf(x)), np.asarray(R)[last])


# --- the estimators' low-level functions ------------------------------------


def test_kaplan_meier_step_with_no_one_at_risk_keeps_its_value():
    # Documented: a step with no events and r zero (0 / 0) leaves the
    # estimate unchanged (#425; it took it to zero), without a raw numpy
    # warning (principle 22; it leaked "invalid value" until #450).
    r, d = np.array([2.0, 1, 0]), np.array([1.0, 0, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        R = nonp.kaplan_meier(r, d)
        var = nonp.greenwood_variance(r, d)
    assert_allclose(R, [0.5, 0.5, 0.5])
    assert_allclose(var, [0.5, 0.5, 0.5])


def test_nelson_aalen_step_with_no_one_at_risk_keeps_its_value_quietly():
    # Documented: no one at risk and no events (0 / 0) leaves it unchanged
    # (#425), without a raw numpy warning (principle 22). Kills the
    # errstates relaxed.
    r, d = np.array([2.0, 1, 0]), np.array([1.0, 0, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        R = nonp.nelson_aalen(r, d)
        var = nonp.nelson_aalen_variance(r, d)
    assert_allclose(R, np.exp(-0.5) * np.ones(3))
    assert_allclose(var, [0.25, 0.25, 0.25])


def test_snap_is_relative_and_one_in_a_billion():
    # Counts that are whole numbers up to round-off (relative 1e-9, at
    # least 1e-9 absolute) are taken as those numbers, and no others.
    # Kills fleming_harrington.py:28 and :38 ('*' -> '/', 1.0 -> 2.0,
    # 'v - nearest' -> 'v + nearest').
    v = np.array([1000 + 5e-7, 1 + 1e-15, 3 - 1e-12, 1 + 1.5e-9, 0.5])
    expected = np.array([1000, 1, 3, 1 + 1.5e-9, 0.5])
    assert_allclose(fh._snap_array(v), expected, rtol=0, atol=0)
    assert_allclose([fh._snap(u) for u in v], expected, rtol=0, atol=0)


def test_kaplan_meier_survives_underflow():
    # #450: 1100 staggered entries with a risk set of 2 at each failure
    # (R = 0.5**k): once the product fell below 1e-308 the fit raised
    # FloatingPointError ('underflow encountered in exp') from a log-space
    # fallback run under errstate(under='raise'), and leaked 'divide by
    # zero in log'. An estimate below the smallest float is 0, quietly.
    # Found from kaplan_meier.py:100-101: the fallback for an underflowing
    # product never ran in the suite (its five mutants survived).
    x = np.arange(1, 1101) + 0.5
    tl = np.r_[0, x[:-1] - 0.75]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = sp.KaplanMeier.fit(x, tl=tl)
        bounds = model.cb([5.0, 1060.5, 1100.5])
    assert np.all(np.diff(model.R) <= 0) and model.R[-1] == 0.0
    # Exact while the product is a normal float: 0.5**1020 is 8.9e-308.
    assert model.R[1020] == 0.5**1021 and model.R[1080] == 0.0
    assert_allclose(model.sf(5.0), 0.5**4)
    assert np.isfinite(bounds).all()


# --- hf, the discrete hazard -------------------------------------------------


@pytest.mark.parametrize("name", ["KaplanMeier", "NelsonAalen"])
def test_hf_of_one_point_is_the_jump_of_its_step(name):
    # "A single point returns the jump of the step it falls in", carried
    # over the censored steps, which have no jump. Here the jumps are at
    # the failure times 1, 3, 4, 6, 7, 9, 10 (the last one infinite for
    # the Kaplan-Meier, which reaches zero). Kills nonparametric.py:485-491
    # (the start of the ladder 0 -> 1, side='right' dropped, pos - 1 ->
    # pos - 2, pos < 0 -> pos <= 0 / pos < 1, the slice pos + 1 -> pos - 1
    # / pos + 2, sub > 0 -> sub >= 0 / sub > 1, nz[-1] -> nz[+1] / nz[-2]
    # and the empty-case conditions).
    model = _fit(name)
    fails = X[C == 0]
    H = _flat(model.Hf(fails))
    jumps = np.diff(np.concatenate([[0.0], H]))
    for t in [0.5, 1, 1.5, 2, 2.5, 3, 5, 5.5, 8, 9.99, 10, 12]:
        k = np.searchsorted(fails, t, side="right") - 1
        expected = np.nan if k < 0 else jumps[k]
        got = _flat(model.hf(t))
        assert got.size == 1
        assert_allclose(got[0], expected, rtol=1e-12, err_msg=f"t={t}")


def test_hf_of_one_point_before_any_failure_is_nan():
    # "Where there is no previous non-zero increment, e.g. before the
    # first failure, the result is NaN": here the first time is censored.
    # Kills nonparametric.py:491 (the empty case forced to nz[-1]).
    model = _fit(c=np.r_[1, C[1:]])
    assert np.isnan(_flat(model.hf(1.0))).all()
    assert np.isnan(_flat(model.hf(1.5))).all()


def test_hf_of_two_points_repeats_the_second():
    # "The first point ... repeats the second point's value": for two
    # points both are the increment between them. Kills
    # nonparametric.py:502 'hf.size > 1' -> '> 2'.
    model = _fit("NelsonAalen")
    H = _flat(model.Hf([2.5, 6.5]))
    assert_allclose(_flat(model.hf([2.5, 6.5])), [H[1] - H[0]] * 2)


def test_hf_missing_query_leaves_the_others_alone():
    # Missing in, missing out (principle 3), and the other points get what
    # they get without it. Conformance's missing-value check does not
    # reach hf with a mixed query. Kills nonparametric.py:468-470.
    model = _fit("NelsonAalen")
    raw = model.hf([np.nan, 3.5, 6.5, np.nan])
    assert raw.dtype == np.float64
    got = _flat(raw)
    assert np.isnan(got[[0, 3]]).all()
    assert_allclose(got[1:3], _flat(model.hf([3.5, 6.5])))


def test_hf_linear_differences_the_linear_cumulative_hazard():
    # Kills nonparametric.py:494 'self.Hf(x, interp=interp)' losing interp
    # (the linear hazard then equals the step one).
    model = _fit("KaplanMeier")
    q = np.array([2.5, 3.5, 4.5, 5.5])
    d = np.diff(_flat(model.Hf(q, interp="linear")))
    assert_allclose(
        _flat(model.hf(q, interp="linear")), np.r_[d[0], d], rtol=1e-12
    )


# --- quantiles ---------------------------------------------------------------


def test_qf_accepts_p_of_one():
    # p is documented in (0, 1]: qf(1) is the time the estimate reaches 0.
    # Kills nonparametric.py:1015 'p > 1' -> 'p >= 1'.
    model = sp.KaplanMeier.fit(X)
    assert _flat(model.qf(1.0))[0] == 10.0


@pytest.mark.parametrize("p", [-0.1, 1.5, 2.5])
def test_quantile_cb_is_nan_for_p_outside_0_1(p):
    # As qf (#626; it raised): NaN with one warning, the other p bounded.
    with pytest.warns(UserWarning, match=r"quantile_cb: .* outside \[0, 1\]"):
        got = _fit().quantile_cb([0.5, p])
    assert np.isnan(got[1]).all() and not np.isnan(got[0, 0])


@pytest.mark.parametrize("alpha_ci", [0.05, 0.3])
@pytest.mark.parametrize("bound_type", ["exp", "normal"])
def test_quantile_cb_inverts_the_pointwise_bounds(alpha_ci, bound_type):
    # Brookmeyer-Crowley: the interval of the p-quantile is the times at
    # which the pointwise interval of sf, at the same alpha_ci, contains
    # 1 - p. Kills nonparametric.py:1108 (alpha_ci not passed on to cb).
    x, c, _ = _truncated_sample()
    model = sp.KaplanMeier.fit(x, c=c)
    bounds = model.cb(model.x, alpha_ci=alpha_ci, bound_type=bound_type)
    for p in [0.25, 0.5]:
        lo = model.x[np.argmax(bounds[:, 0] <= 1 - p)]
        hi = model.x[np.argmax(bounds[:, 1] < 1 - p)]
        got = model.quantile_cb(p, alpha_ci=alpha_ci, bound_type=bound_type)
        assert_allclose(_flat(got), [lo, hi])


def test_quantile_cb_open_ends_are_nan():
    # A bound that never reaches the level gives NaN, at either end; and
    # at p = 1 the lower end is the last time, where the lower bound of
    # a curve falling to 0 is 0. Kills nonparametric.py:1120 '<=' -> '<'
    # and :1123 (the NaN case forced to x[0]).
    # Two failures then censoring: the lower bound of sf stays above 0.4.
    model = _fit(c=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    assert np.isnan(_flat(model.quantile_cb(0.8))).all()
    complete = sp.KaplanMeier.fit(X)
    assert _flat(complete.quantile_cb(1.0))[0] == 10.0


# --- the (restricted) mean ---------------------------------------------------


def test_mean_with_an_observation_at_zero():
    # Zero is a valid (non-negative) time. Kills nonparametric.py:1165
    # '< 0' -> '<= 0'.
    model = sp.KaplanMeier.fit([0.0, 1, 2, 3])
    assert model.mean() == pytest.approx(0.75 + 0.5 + 0.25)


@pytest.mark.parametrize("tau", [None, 4.5])
@pytest.mark.parametrize("alpha_ci", [0.05, 0.2])
def test_mean_cb_is_the_rmst_interval(tau, alpha_ci):
    # Two entry points for one interval (principle 14). Kills
    # nonparametric.py:1224 (tau or alpha_ci not passed on).
    model = _fit()
    res = model.rmst(tau=tau, alpha_ci=alpha_ci)
    assert_allclose(
        model.mean_cb(tau=tau, alpha_ci=alpha_ci), [res["lower"], res["upper"]]
    )


@pytest.mark.parametrize("alpha_ci", [0.05, 0.2])
def test_rmst_diff_agrees_with_each_groups_rmst(alpha_ci):
    # The difference of two independent groups' RMSTs: its parts are the
    # groups' own rmst(), its variance their sum. Kills
    # nonparametric.py:2321 (alpha_ci / 2 -> * 2, / 3), :2330-2335
    # ('z * se' -> 'z / se', the ratio, the keys 'rmst_a', 'rmst_b').
    a = _fit()
    b = sp.KaplanMeier.fit(
        [1.0, 2, 2, 3, 4, 5, 6, 8], c=[0, 0, 1, 0, 0, 0, 1, 0]
    )
    tau = 6.0
    ra = a.rmst(tau=tau, alpha_ci=alpha_ci)
    rb = b.rmst(tau=tau, alpha_ci=alpha_ci)
    res = sp.rmst_diff(a, b, tau=tau, alpha_ci=alpha_ci)
    diff = ra["rmst"] - rb["rmst"]
    se = np.hypot(ra["se"], rb["se"])
    z = norm.ppf(1 - alpha_ci / 2)
    assert_allclose(
        [res[k] for k in ("rmst_a", "rmst_b", "difference", "se")],
        [ra["rmst"], rb["rmst"], diff, se],
    )
    assert_allclose(
        [res["lower"], res["upper"]], [diff - z * se, diff + z * se]
    )
    assert_allclose(res["ratio"], ra["rmst"] / rb["rmst"])
    assert_allclose(res["p_value"], 2 * norm.sf(abs(diff) / se))


def test_rmst_diff_edge_cases():
    # A group whose RMST is exactly 1 still has a ratio; a zero horizon has
    # no variance, so no p-value and no ratio (NaN, not an error). Kills
    # nonparametric.py:2322 'se > 0' -> '>= 0', :2325 and :2333 (the NaN
    # cases).
    a = sp.KaplanMeier.fit([0.5, 2.0, 3.0])
    b = sp.KaplanMeier.fit([2.0, 3.0, 4.0])
    res = sp.rmst_diff(a, b, tau=1.0)
    assert res["rmst_b"] == 1.0
    assert_allclose(res["ratio"], res["rmst_a"])
    zero = sp.rmst_diff(a, b, tau=0.0)
    assert zero["difference"] == 0.0 and zero["se"] == 0.0
    assert np.isnan(zero["p_value"]) and np.isnan(zero["ratio"])


# --- random ------------------------------------------------------------------


def test_random_takes_a_shape():
    # size may be a tuple (documented). Kills nonparametric.py:980
    # 'reshape(np.shape(u))' -> 'reshape(None)'.
    model = _fit()
    draws = model.random((2, 3), random_state=0)
    assert draws.shape == (2, 3)
    assert np.isin(draws, np.r_[X, np.inf]).all()


# --- bootstrap_cb ------------------------------------------------------------


def test_bootstrap_is_centred_on_the_estimate_with_truncation():
    # The median of the resampled curves is the estimate, to within the
    # resampling noise (0.01 here), including for left truncated data.
    # With alpha_ci = 0.98 the two-sided bounds are the 49% and 51%
    # quantiles. Kills nonparametric.py:1454 (the truncation dropped from
    # the refits) and :1459 (the first step clipped to the second).
    x, c, tl = _truncated_sample()
    model = sp.KaplanMeier.fit(x, c=c, tl=tl)
    q = np.quantile(x, [0.1, 0.3, 0.5, 0.7, 0.9])
    median = model.bootstrap_cb(q, alpha_ci=0.98, n_boot=300, random_state=0)
    assert_allclose(median, np.c_[model.sf(q), model.sf(q)], atol=0.03)


def test_bootstrap_bounds_are_right_continuous():
    # Like the estimate, the bounds take the value of the step at a step
    # time (up to the last, past which they are NaN). Kills
    # nonparametric.py:1457 (side='right' dropped).
    model = sp.KaplanMeier.fit(X)
    at = model.bootstrap_cb(X[:-1], n_boot=50, random_state=0)
    after = model.bootstrap_cb(X[:-1] + 1e-9, n_boot=50, random_state=0)
    assert_allclose(at, after)


def test_bootstrap_at_the_first_and_last_times():
    # A resample that misses the first failure (0.9^10 = 35% of them)
    # has survival 1 there, one that has it less than 1; and every
    # resample of complete data ends at 0, so at the last time both bounds
    # are 0. Kills nonparametric.py:1459 (idx < 0 -> None, <= 0, < 1; the
    # 1.0 -> 2.0; the last index len - 1 -> len - 2).
    model = sp.KaplanMeier.fit(X)
    lower, upper = _flat(model.bootstrap_cb(1.0, n_boot=200, random_state=0))
    assert upper == 1.0
    assert lower < 1.0
    assert_allclose(model.bootstrap_cb(10.0, n_boot=100, random_state=0), 0.0)


def test_bootstrap_default_and_smallest_B():
    # B defaults to 200 (documented) and B = 1 is a positive integer.
    # Kills nonparametric.py:1319 (B = 201) and :1410 ('B < 1' -> 'B <= 1',
    # 'B < 2').
    model = _fit()
    assert_allclose(
        model.bootstrap_cb([3, 6], random_state=4),
        model.bootstrap_cb([3, 6], n_boot=200, random_state=4),
    )
    one = model.bootstrap_cb([3, 6], n_boot=1, random_state=4)
    assert_allclose(one[:, 0], one[:, 1])


def test_turnbull_bootstrap_equals_kaplan_meier_on_right_censored_data():
    # On right censored data the Turnbull NPMLE is the Kaplan-Meier, and
    # both resample the same rows with the same stream, so the bootstrap
    # bounds agree to the EM's tolerance. Kills nonparametric.py:1446
    # (the censoring dropped from the Turnbull refits).
    km = _fit()
    tb = sp.Turnbull.fit(X, c=C, turnbull_estimator="Kaplan-Meier")
    q = [1.5, 3, 5, 8]
    assert_allclose(
        tb.bootstrap_cb(q, n_boot=100, random_state=3),
        km.bootstrap_cb(q, n_boot=100, random_state=3),
        atol=1e-8,
    )


@pytest.mark.parametrize("bound", ["two-sided", "lower", "upper"])
def test_bootstrap_cb_is_nan_outside_the_data_like_cb(bound):
    # #452: Kaplan-Meier of 1..10 (last censored) gave bootstrap_cb([0.5,
    # 11]) = [[1, 1], [0, 0.548]] where cb is NaN; and a missing time sorted
    # past the last step. Found while triaging the bootstrap mutants
    # (principle 11: behaviour outside the data is the same for all of a
    # model's bounds; principle 3: missing in, missing out).
    model = _fit(c=np.r_[C[:-1], 1])
    q = [0.5, 11, np.nan, 5]
    kw = dict(bound=bound, n_boot=50, random_state=0)
    assert np.isnan(model.cb(q, bound=bound)[:3]).all()
    got = model.bootstrap_cb(q, **kw)
    assert np.isnan(got[:3]).all() and np.isfinite(got[3]).all()
    # Inside the data, the same bounds as before.
    assert_allclose(got[3:], model.bootstrap_cb([5], **kw))


def test_bootstrap_cb_follows_cb_with_a_support():
    # With a support: the start value (1) from lower to the first time,
    # the bounds at the last time carried to upper, NaN outside, exactly
    # where cb gives them.
    model = _fit(c=np.r_[C[:-1], 1]).set_support(0, 20)
    q = [-1, 0.5, 10, 15, 21, np.nan]
    got = model.bootstrap_cb(q, n_boot=50, random_state=0)
    ref = model.cb(q)
    assert_allclose(np.isnan(got), np.isnan(ref))
    assert_allclose(got[1], [1.0, 1.0])
    assert_allclose(got[3], got[2])


# --- serialisation (principle 20) --------------------------------------------


def test_restored_model_bootstraps_as_the_original():
    # A dictionary written with its data restores a model that can
    # bootstrap, and gives the original's bounds. Conformance's round trip
    # compares predictions only; bootstrap_cb could join them. Kills
    # nonparametric.py:2238 ('or' -> 'and', either 'in' -> 'not in').
    model = _fit()
    restored = sp.from_dict(model.to_dict(with_data=True))
    assert_allclose(
        restored.bootstrap_cb([2, 5, 8], n_boot=50, random_state=2),
        model.bootstrap_cb([2, 5, 8], n_boot=50, random_state=2),
    )


def test_restored_turnbull_keeps_its_estimator():
    # Without its data a Turnbull model keeps its estimator settings (they
    # show in its repr and drive bootstrap_cb). Kills nonparametric.py:2238
    # (the 'estimator' test mutated).
    model = sp.Turnbull.fit(X, c=C)
    restored = sp.from_dict(model.to_dict())
    assert repr(restored) == repr(model)
    assert restored.data["estimator"] == "Fleming-Harrington"


def test_turnbull_saved_before_tol_was_recorded_bootstraps_as_fitted():
    # A dictionary from before 'tol' and 'max_iter' were stored refits
    # with the fit() defaults, which are what this model used. Kills
    # nonparametric.py:1429-1430 (the fallback tol and max_iter).
    model = sp.Turnbull.fit(X, c=C)
    old = model.to_dict(with_data=True)
    del old["tol"], old["max_iter"]
    restored = sp.from_dict(old)
    assert_allclose(
        restored.bootstrap_cb([2, 5, 8], n_boot=20, random_state=2),
        model.bootstrap_cb([2, 5, 8], n_boot=20, random_state=2),
    )


@pytest.mark.parametrize("method", ["hall-wellner", "nair"])
@pytest.mark.parametrize("bound_type", ["exp", "normal"])
@pytest.mark.parametrize("alpha_ci", [0.05, 0.2])
def test_band_over_a_single_time_is_the_pointwise_interval(
    method, bound_type, alpha_ci
):
    # One failure, then censoring: the band covers one value of a = N
    # sigma^2 / (1 + N sigma^2), where the supremum is a single normal
    # variable, so both bands reduce to the pointwise interval (for
    # Hall-Wellner, z sqrt(a (1 - a)) (1 + N sigma^2) / sqrt(N) = z sigma).
    # A property conformance could check for every band. The critical
    # value is found numerically, to about 1e-6. Kills band's half width
    # ('/ sqrt(N)' -> '* sqrt(N)', 'N * sigma2' -> 'N / sigma2'), which
    # the containment test passes.
    model = sp.KaplanMeier.fit([1.0, 2, 3, 4], c=[0, 1, 1, 1])
    q = [1, 2.5]
    assert_allclose(
        model.band(q, method=method, bound_type=bound_type, alpha_ci=alpha_ci),
        model.cb(q, bound_type=bound_type, alpha_ci=alpha_ci),
        atol=1e-5,
    )


def test_band_round_trips_without_its_data():
    # A model restored without its data takes the sample size for the
    # band from the risk set (N = max r), which is the same number for
    # untruncated data. Kills nonparametric.py:1730 (the fallback N
    # mutated to None).
    model = _fit()
    restored = sp.from_dict(model.to_dict())
    assert_allclose(restored.band([3, 6]), model.band([3, 6]), rtol=1e-12)


@pytest.mark.parametrize("name", ["KaplanMeier", "Turnbull"])
def test_band_round_trips_with_truncation(name):
    # #451: without its data a restored model took the band's N from the
    # risk set: for this Kaplan-Meier N = n.sum() = 60 became max(r) = 37,
    # and the band at the 20% time went from [0.4900, 0.8917] to [0.4535,
    # 0.9018]. Principle 20 (identical predictions after a round trip);
    # found triaging the band mutants that choose N (nonparametric.py:1727).
    x, c, tl = _truncated_sample()
    model = getattr(sp, name).fit(x, c=c, tl=tl)
    q = np.quantile(x, [0.2, 0.5, 0.8])
    d = model.to_dict()
    assert d["band_n"] == 60.0 and d["schema"] == 2
    restored = sp.from_dict(json.loads(json.dumps(d)))
    assert_allclose(restored.band(q), model.band(q), rtol=1e-12)
    if name == "KaplanMeier":
        assert_allclose(
            restored.band(q, bound_type="exp")[0],
            [0.4900, 0.8917],
            atol=5e-5,
            rtol=0,
        )


def test_band_n_is_stored_only_where_the_risk_set_differs():
    # Untruncated, the fallback N = max(r) is the number of items (up to
    # the Turnbull EM's round-off), so nothing is added and the dictionary
    # stays schema 1, readable by v0.20; a dictionary written before
    # "band_n" was stored keeps the fallback.
    x, c, tl = _truncated_sample()
    em = sp.Turnbull.fit(x, c=c, turnbull_algorithm="EM")
    for model in (sp.KaplanMeier.fit(x, c=c), em):
        d = model.to_dict()
        assert "band_n" not in d and d["schema"] == 1
    # (the default Turnbull fit of untruncated data is the EM-ICM, #620,
    # whose dictionary names it and so is schema 2; still no band_n)
    d = sp.Turnbull.fit(x, c=c).to_dict()
    assert "band_n" not in d and d["algorithm"] == "EMICM"
    model = sp.KaplanMeier.fit(x, c=c, tl=tl)
    old = model.to_dict()
    del old["band_n"]
    q = np.quantile(x, [0.2])
    assert_allclose(
        sp.from_dict(old).band(q, bound_type="exp"),
        [[0.4535, 0.9018]],
        atol=5e-5,
        rtol=0,
    )
    old["band_n"] = "60"
    with pytest.raises(ValueError, match="'band_n' must be a number"):
        sp.from_dict(old)


def test_band_does_not_warn_and_its_retired_arguments_are_gone():
    # No warning by default; n_sims and random_state, unused since the
    # critical value stopped being simulated, were removed in v0.22.
    model = _fit()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.band([3, 6])
    for kw in ({"n_sims": 100}, {"random_state": 1}):
        with pytest.raises(TypeError, match="unexpected keyword"):
            model.band([3, 6], **kw)


# --- smoothed_hf -------------------------------------------------------------


def test_smoothed_hf_by_hand():
    # Nelson-Aalen of 1, 2, 3: jumps 1/3, 1/2, 1. With b = 4 at t = 2 the
    # Epanechnikov weights 0.75 (1 - u^2) at u = 1/4, 0, -1/4 are
    # 0.703125, 0.75, 0.703125, so the sum is 1.3125 / 4 = 0.328125; the
    # share of the kernel inside [1, 3] is F(1/4) - F(-1/4) with
    # F(v) = (2 + 3v - v^3) / 4, i.e. 0.3671875. Kills
    # nonparametric.py:1880 ('-' -> '+' in the share).
    model = sp.NelsonAalen.fit([1.0, 2, 3])
    got = _flat(model.smoothed_hf(2.0, bandwidth=4.0))
    assert_allclose(got, [0.328125 / 0.3671875], rtol=1e-12)


def test_smoothed_hf_leaves_out_an_infinite_jump():
    # A Kaplan-Meier ending in a failure has an infinite last jump, left
    # out of the sum: the estimate is the one with that item censored.
    # Kills nonparametric.py:1847 (the infinite jump counted as 1).
    ends_failed = sp.KaplanMeier.fit(X)
    ends_censored = sp.KaplanMeier.fit(X, c=(X == 10).astype(int))
    q = np.linspace(1, 10, 19)
    assert_allclose(
        ends_failed.smoothed_hf(q, bandwidth=3),
        ends_censored.smoothed_hf(q, bandwidth=3),
        rtol=1e-12,
    )


def test_smoothed_hf_moves_with_the_data():
    # Shifting every time by a constant shifts the estimate (the
    # non-parametric estimate depends on the order and spacing of the
    # times only): a metamorphic property conformance could check for
    # every non-parametric model. With the default bandwidth, (max - min)
    # / 8. Kills nonparametric.py:1861 ('/ 8' -> '* 8', '/ 9',
    # 'max - min' -> 'max + min') and :1879 ('x - min' -> 'x + min').
    model = _fit("NelsonAalen")
    shifted = _fit("NelsonAalen", x=X + 100)
    q = np.linspace(1, 10, 19)
    assert_allclose(
        shifted.smoothed_hf(q + 100), model.smoothed_hf(q), rtol=1e-9
    )
    assert_allclose(
        model.smoothed_hf(q), model.smoothed_hf(q, bandwidth=9 / 8)
    )


def test_smoothed_hf_is_defined_on_the_closed_range_only():
    # Finite at the first and last times, NaN outside them (documented).
    # Kills nonparametric.py:1884 (the mask dropped, '|' -> '&', and the
    # ends excluded) and :1862 (a zero bandwidth accepted).
    model = _fit("NelsonAalen")
    got = _flat(model.smoothed_hf([0.9, 1.0, 10.0, 10.1], bandwidth=2))
    assert np.isnan(got[[0, 3]]).all() and np.isfinite(got[[1, 2]]).all()
    with pytest.raises(ValueError, match="'bandwidth' must be positive"):
        model.smoothed_hf(5, bandwidth=0)


# --- messages name what was wrong (principle 2) ------------------------------


@pytest.mark.parametrize(
    "call, words",
    [
        (lambda m: m.set_support(5, 2), ["lower=5.0", "upper=2.0"]),
        (lambda m: m.set_support(0, 9), ["of at least 10.0"]),
        (lambda m: m.smoothed_hf(5, bandwidth=-1), ["got -1"]),
        (lambda m: m.mean(tau=-2), ["got -2"]),
    ],
)
def test_messages_quote_the_value(call, words):
    # Kills the mutants that drop the value from the message, e.g.
    # nonparametric.py:59 ('.format(lo, hi)' -> '.format(None, hi)').
    with pytest.raises(ValueError) as info:
        call(_fit())
    for word in words:
        assert word in str(info.value)


def test_from_dict_message_quotes_the_bad_support():
    # Kills nonparametric.py:84 ('.format(value)' -> '.format(None)').
    d = _fit().to_dict()
    d["support"] = [1.0]
    with pytest.raises(ValueError, match=r"got \[1.0\]"):
        sp.from_dict(d)


# --- the aliases of ``on`` (principle 21) ------------------------------------


@pytest.mark.parametrize("bound", ["upper", "two-sided"])
def test_bounds_of_a_single_failure_are_the_estimate(bound):
    # No value has a finite variance, so the upper bound falls back to the
    # estimate, 0 (documented in cb's Notes). Kills nonparametric.py:888
    # (the fallback mutated).
    model = sp.KaplanMeier.fit([5.0])
    assert_allclose(_flat(model.cb(5.0, bound=bound)), 0.0)
    assert_allclose(_flat(model.cb(5.0, on="ff", bound="lower")), 1.0)


@pytest.mark.parametrize("bound", ["upper", "two-sided"])
def test_cb_of_a_missing_query_is_nan(bound):
    # Missing in, missing out (principle 3); a NaN query used to sort past
    # the last step. Kills nonparametric.py:908 (the NaN mask dropped).
    model = _fit(c=np.r_[C[:-1], 1])
    got = model.cb([np.nan, 3.0], bound=bound)
    assert np.isnan(got[0]).all()
    assert_allclose(got[1], model.cb([3.0], bound=bound)[0])


def test_R_cb_defaults_are_those_of_cb():
    # R_cb is cb(on='sf') with the columns as [upper, lower] (documented),
    # with the same defaults. Kills nonparametric.py:803 (alpha_ci = 1.05).
    model = _fit()
    assert_allclose(model.R_cb([2, 5]), np.fliplr(model.cb([2, 5])))


def test_support_keeps_the_estimate_up_to_the_last_time():
    # set_support changes nothing within the data and carries the value at
    # the last time (principle 11). Kills nonparametric.py:295 (the last
    # time taken as x[-2]).
    model = _fit()
    bounded = _fit().set_support(0, 20)
    assert_allclose(
        _flat(bounded.sf([9.5, 10, 15])), _flat(model.sf([9.5, 10, 10]))
    )
    assert_allclose(bounded.cb([10, 15]), model.cb([10, 10]))


def test_cb_aliases_agree_with_a_support_set():
    # 'R' and 'F' are 'sf' and 'ff', also where the support gives the start
    # value (1 for survival). Kills nonparametric.py:755 (the 'R' in the
    # start value's test mutated).
    model = _fit().set_support(0, 20)
    q = [0.5, 3, 12]
    assert_allclose(model.cb(q, on="R"), model.cb(q, on="sf"))
    assert_allclose(model.cb(q, on="F"), model.cb(q, on="ff"))


# --- repr and plot -----------------------------------------------------------


def test_repr_names_the_estimator():
    # Kills nonparametric.py:198-208 (28 mutants of __repr__).
    head = "Non-Parametric SurPyval Model\n" + "=" * 29 + "\n"
    # and the data it was fitted to (#508)
    data = (
        "\nData             : 10 units: 7 events at 7 unique times, "
        "3 right censored"
    )
    assert repr(_fit()) == head + "Model            : Kaplan-Meier" + data
    assert repr(sp.Turnbull.fit(X, c=C)) == (
        head + "Model            : Turnbull\n"
        "Estimator        : Fleming-Harrington" + data
    )


def _plotted(model, **kw):
    fig, ax = plt.subplots()
    try:
        assert model.plot(ax=ax, **kw) is ax
        lines = [(ln.get_xdata(), ln.get_ydata(), ln) for ln in list(ax.lines)]
        bands = [c.get_paths()[0].vertices for c in ax.collections]
        colours = [c.get_facecolor()[0] for c in ax.collections]
        return lines, bands, colours, ax.get_ylim()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("interp", ["step", "linear"])
@pytest.mark.parametrize("alpha_ci", [0.05, 0.4])
def test_plot_draws_the_estimate_its_bounds_and_the_censored_items(
    interp, alpha_ci
):
    # What plot draws, not only how many artists: the curve at the model's
    # own times, the band between the R_cb bounds at alpha_ci, as steps for
    # the step curve, in the curve's colour, and a tick on the curve at
    # each censored item. Kills most of the 122 surviving mutants of plot
    # and get_plot_data (the data, the bounds' options, the censoring
    # marks); those of the title, the labels and the marker styling are
    # left.
    x = np.array([4.0, 7, 9, 13, 16, 21, 28, 33, 41, 50])
    c = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 1])
    model = sp.KaplanMeier.fit(x, c=c)
    lines, bands, colours, ylim = _plotted(
        model, interp=interp, alpha_ci=alpha_ci
    )
    assert ylim == (0.0, 1.0)
    (cx, cy, curve), (tx, ty, ticks) = lines
    assert_allclose(cx, model.x)
    assert_allclose(cy, model.R)
    assert (curve.get_drawstyle() == "steps-post") == (interp == "step")
    assert_allclose(tx, x[c == 1])
    assert_allclose(ty, _flat(model.sf(x[c == 1], interp=interp)))
    assert ticks.get_marker() == "|"
    (band,) = bands
    bounds = model.R_cb(model.x, interp=interp, alpha_ci=alpha_ci)
    assert_allclose(np.unique(band[:, 1]), np.unique(bounds))
    assert (len(band) > 3 * len(x)) == (interp == "step")
    assert_allclose(to_rgb(colours[0]), to_rgb(curve.get_color()))
    assert to_rgb(ticks.get_color()) == to_rgb(curve.get_color())


@pytest.mark.parametrize("interp", ["step", "linear"])
def test_plot_of_a_one_sided_bound_is_a_dashed_line(interp):
    model = _fit()
    lines, bands, _, _ = _plotted(
        model, bound="upper", interp=interp, show_censors=False
    )
    assert not bands
    (_, _, curve), (bx, by, bound) = lines
    assert_allclose(bx, model.x)
    assert_allclose(by, model.R_cb(model.x, bound="upper", interp=interp))
    assert bound.get_linestyle() == "--"
    assert bound.get_color() == curve.get_color()
    assert (bound.get_drawstyle() == "steps-post") == (interp == "step")


def test_plot_passes_on_the_bound_options_and_line_style():
    # bound_type and dist go to the bounds, anything else to the curve.
    # Kills nonparametric.py:1968-1969 (the options left in kwargs) and
    # :2003 (the curve's kwargs dropped for an interpolated curve).
    model = _fit(c=np.r_[C[:-1], 1])
    lines, bands, _, _ = _plotted(
        model, bound_type="normal", dist="z", interp="linear", color="C3"
    )
    assert to_rgb(lines[0][2].get_color()) == to_rgb("C3")
    expected = model.R_cb(model.x, bound_type="normal", interp="linear")
    assert_allclose(np.unique(bands[0][:, 1]), np.unique(expected))


def test_plot_marks_nothing_without_censoring_or_data():
    # No tick line for complete data; and a model without data plots with
    # show_censors=False. Kills nonparametric.py:1978 ('and' -> 'or') and
    # :2029 ('and' -> 'or').
    lines, _, _, _ = _plotted(sp.KaplanMeier.fit(X))
    assert len(lines) == 1
    ecdf = NonParametric.fit_from_ecdf([1.0, 2, 3], [0.8, 0.5, 0.1])
    lines, _, _, _ = _plotted(ecdf, plot_bounds=False, show_censors=False)
    assert len(lines) == 1


def test_plot_defaults_to_the_current_axes():
    # Kills nonparametric.py:1958 ('ax is None' -> 'is not None') and :1961.
    fig, ax = plt.subplots()
    try:
        assert _fit().plot() is ax
    finally:
        plt.close(fig)


def test_get_plot_data():
    # The axis limits: from min(0, first time) to the last time plus a
    # tenth of that range. Kills nonparametric.py:1897-1921.
    model = sp.KaplanMeier.fit(X + 10, c=C)
    d = model.get_plot_data(alpha_ci=0.3)
    assert (d["x_scale_min"], d["x_scale_max"]) == (0, 20 + 2.0)
    assert (d["y_scale_min"], d["y_scale_max"]) == (0, 1)
    assert_allclose(d["F"], model.F)
    assert_allclose(d["cbs"], model.R_cb(model.x, alpha_ci=0.3))
    negative = NonParametric.fit_from_ecdf([-5.0, 5.0], [0.5, 0.0])
    d = negative.get_plot_data(plot_bounds=False)
    assert (d["x_scale_min"], d["x_scale_max"], d["cbs"]) == (-5, 6, None)
