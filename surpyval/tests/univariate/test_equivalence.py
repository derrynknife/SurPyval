"""Equivalent time scales (``Equivalence``): one life on two clocks,
``T = x1 + k x2``, fitted by maximum likelihood with the clock-rate
Jacobian.

The simulations below have fixed seeds and sizes, chosen before their
results were seen.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar

import surpyval as sp
from surpyval.univariate.equivalence import _Data, _log_lik

K_TRUE = 10.0
ALPHA, BETA = 2000.0, 2.0
RATES = (0.0, 0.02, 0.1, 0.5)  # starts per hour


def _fleet(n, seed, k=K_TRUE, follow_up=(500.0, 3000.0), tl=False):
    """``n`` units at mixed start rates, each followed for a uniform number
    of hours; a unit's life on the combined clock is Weibull(2000, 2)."""
    rng = np.random.default_rng(seed)
    rate = rng.choice(RATES, n)
    T = ALPHA * rng.weibull(BETA, n)
    life = T / (1.0 + k * rate)
    end = rng.uniform(*follow_up, n)
    c = (life > end).astype(int)
    h = np.minimum(life, end)
    out = {"x1": h, "x2": rate * h, "c": c, "rate": rate}
    if tl:
        # entered observation at a uniform fraction of its age; kept only
        # if it had not failed before then (left truncation)
        frac = rng.uniform(0.0, 0.6, n)
        out["tl"] = np.column_stack([frac * h, frac * rate * h])
    return out


def _fit(d, dist=sp.Weibull, **kw):
    return sp.Equivalence(dist).fit(
        d["x1"], d["x2"], c=d["c"], tl=d.get("tl"), **kw
    )


def test_recovers_k_and_the_distribution():
    d = _fleet(3000, seed=101)
    model = _fit(d)
    assert model.maximum == "verified"
    alpha, beta, k = model.params
    assert k == pytest.approx(K_TRUE, rel=0.1)
    assert alpha == pytest.approx(ALPHA, rel=0.05)
    assert beta == pytest.approx(BETA, rel=0.05)
    lo, hi = model.param_cb("k")
    assert lo < K_TRUE < hi
    lo, hi = model.param_cb("k", method="lr")
    assert lo < K_TRUE < hi


@pytest.mark.parametrize(
    "dist", [sp.LogNormal, sp.LogLogistic, sp.Gamma, sp.Exponential]
)
def test_recovers_k_with_every_distribution(dist):
    rng = np.random.default_rng(202)
    n = 2000
    rate = rng.choice(RATES, n)
    T = np.asarray(
        dist.random(n, *dist.fit(ALPHA * rng.weibull(BETA, 50)).params),
        dtype=float,
    )
    h = T / (1.0 + K_TRUE * rate)
    model = sp.Equivalence(dist).fit(h, rate * h)
    assert model.maximum == "verified"
    assert model.k == pytest.approx(K_TRUE, rel=0.15)


def test_the_jacobian_removes_the_bias_of_the_naive_fit():
    """Fitting the distribution to T = h + k s without the factor
    (1 + k rho) pulls k towards 0; with it, k is recovered. 20 fleets of
    1,000 units, k = 10."""
    ours, naive = [], []
    for seed in range(20):
        d = _fleet(1000, seed=1000 + seed)
        ours.append(_fit(d).k)

        def neg_ll(log_k, d=d):
            k = np.exp(log_k)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sp.Weibull.fit(d["x1"] + k * d["x2"], d["c"])
            return fit.neg_ll()

        res = minimize_scalar(
            neg_ll, bounds=(np.log(1e-3), np.log(1e3)), method="bounded"
        )
        naive.append(np.exp(res.x))
    ours, naive = np.array(ours), np.array(naive)
    se = ours.std(ddof=1) / np.sqrt(ours.size)
    # ours: unbiased within 3 standard errors of the mean
    assert abs(ours.mean() - K_TRUE) < 3 * se
    # the naive estimate is nowhere near
    assert naive.mean() < 0.5 * K_TRUE


def test_k_zero_is_the_plain_fit_on_the_first_clock():
    d = _fleet(500, seed=7, k=0.0)
    hours = sp.Weibull.fit(d["x1"], d["c"])
    # the likelihood at k = 0 is the distribution's on hours, exactly
    data = _Data(d["x1"], d["x2"], d["c"])
    for params in ([1500.0, 1.5], [2500.0, 3.0]):
        ours = float(_log_lik(sp.Weibull, data, params, 0.0))
        theirs = float(
            np.sum(
                np.where(
                    d["c"] == 0,
                    sp.Weibull.log_df(d["x1"], *params),
                    sp.Weibull.log_sf(d["x1"], *params),
                )
            )
        )
        assert ours == pytest.approx(theirs, rel=1e-13)
    # and where k is fitted at 0, the model is that fit
    model = _fit(d)
    if model.k == 0.0:
        np.testing.assert_allclose(model.params[:-1], hours.params, rtol=1e-12)
        assert model.neg_ll() == pytest.approx(hours.neg_ll(), rel=1e-12)
        assert model.k_lr_test()["p_value"] == 1.0
        assert model.param_cb("k", method="lr")[0] == 0.0
    else:
        assert model.k_lr_test()["p_value"] > 0.05


def test_k_fitted_at_zero_matches_the_hours_fit():
    """A fleet where starts truly do not matter, and k lands on 0."""
    hits = 0
    for seed in range(10):
        d = _fleet(300, seed=500 + seed, k=0.0)
        model = _fit(d)
        if model.k > 0:
            continue
        hits += 1
        hours = sp.Weibull.fit(d["x1"], d["c"])
        np.testing.assert_allclose(model.params[:-1], hours.params, rtol=1e-12)
        assert model.neg_ll() == pytest.approx(hours.neg_ll(), rel=1e-12)
        with pytest.warns(RuntimeWarning, match="method='lr'"):
            assert np.isnan(model.param_cb("k")).all()
        lo, hi = model.param_cb("k", method="lr")
        assert lo == 0.0 and 0 < hi < np.inf
    # about half the fits land on the boundary (Self and Liang)
    assert hits >= 2


def test_equals_an_aft_fit_on_log_of_the_clock_rate_at_fixed_k():
    """At fixed k the model is the Weibull AFT with covariate
    log(1 + k rho) and coefficient 1: the same likelihood, so the same
    fit."""
    d = _fleet(400, seed=11)
    model = _fit(d)
    Z = np.log1p(model.k * d["rate"])[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aft = sp.WeibullAFT.fit(d["x1"], Z, c=d["c"], fixed={"coef_0": 1.0})
    np.testing.assert_allclose(aft.params[:2], model.params[:2], rtol=1e-5)
    assert aft.neg_ll() == pytest.approx(model.neg_ll(), rel=1e-9)
    # and the hours-scale predictions agree
    t = np.array([100.0, 400.0, 900.0])
    for rate in (0.0, 0.1, 0.5):
        z = np.full((t.size, 1), np.log1p(model.k * rate))
        np.testing.assert_allclose(
            model.sf_at_usage(t, rate), aft.sf(t, z), rtol=1e-4
        )


def test_censoring_and_late_entry():
    d = _fleet(3000, seed=303, tl=True)
    # late entry: keep the units that had not failed by entry
    model = _fit(d)
    assert model.maximum == "verified"
    assert model.k == pytest.approx(K_TRUE, rel=0.15)
    # left-censored rows are taken
    c = d["c"].copy()
    c[:50] = np.where(c[:50] == 0, -1, c[:50])
    left = sp.WeibullEquivalence.fit(d["x1"], d["x2"], c=c)
    assert left.k == pytest.approx(K_TRUE, rel=0.2)


def test_counts_are_repeated_rows():
    d = _fleet(200, seed=5)
    n = np.where(np.arange(200) % 3 == 0, 2, 1)
    counted = sp.WeibullEquivalence.fit(d["x1"], d["x2"], d["c"], n=n)
    repeated = sp.WeibullEquivalence.fit(
        np.repeat(d["x1"], n), np.repeat(d["x2"], n), np.repeat(d["c"], n)
    )
    np.testing.assert_allclose(counted.params, repeated.params, rtol=1e-6)


def test_swapping_the_clocks_inverts_k():
    d = _fleet(800, seed=17)
    model = _fit(d)
    swapped = sp.WeibullEquivalence.fit(d["x2"], d["x1"], c=d["c"])
    assert 1.0 / swapped.k == pytest.approx(model.k, rel=1e-5)
    assert swapped.params[1] == pytest.approx(model.params[1], rel=1e-5)


def test_units_who_failed_on_starts_alone():
    d = _fleet(500, seed=19)
    h, s = d["x1"].copy(), d["x2"].copy()
    failed = np.flatnonzero(d["c"] == 0)[:5]
    s[failed] = h[failed] / K_TRUE
    h[failed] = 0.0
    model = sp.WeibullEquivalence.fit(h, s, c=d["c"])
    assert model.maximum == "verified"
    assert model.k > 0
    assert model.k_lr_test()["p_value"] == 0.0


def test_fit_from_df_names_the_clocks():
    d = _fleet(300, seed=23)
    df = pd.DataFrame({"hours": d["x1"], "starts": d["x2"], "c": d["c"]})
    model = sp.WeibullEquivalence.fit_from_df(
        df, x_cols=["hours", "starts"], c_col="c"
    )
    direct = _fit(d)
    np.testing.assert_allclose(model.params, direct.params, rtol=1e-9)
    assert model.clocks == ("hours", "starts")
    assert "one unit of 'starts' is worth" in repr(model)
    assert "95% CI" in repr(model)


def test_to_dict_round_trip():
    d = _fleet(300, seed=29)
    model = _fit(d)
    restored = sp.from_dict(model.to_dict())
    assert isinstance(restored, sp.EquivalenceModel)
    np.testing.assert_array_equal(restored.params, model.params)
    t = np.array([100.0, 1000.0])
    np.testing.assert_array_equal(restored.sf(t), model.sf(t))
    np.testing.assert_array_equal(restored.cb(t), model.cb(t))
    np.testing.assert_array_equal(
        restored.cb_at_usage(t, 0.1), model.cb_at_usage(t, 0.1)
    )
    np.testing.assert_array_equal(restored.param_cb("k"), model.param_cb("k"))
    assert restored.aic() == model.aic()
    assert restored.bic() == model.bic()
    assert restored.k_lr_test() == model.k_lr_test()
    assert restored.clocks == model.clocks
    with pytest.raises(ValueError, match="need the data"):
        restored.param_cb("k", method="lr")


def test_usage_functions():
    model = sp.WeibullEquivalence.from_params([2000.0, 2.0, 10.0])
    t = np.array([100.0, 500.0])
    np.testing.assert_allclose(
        model.sf_at_usage(t, 0.1), model.sf(t * 2.0), rtol=1e-15
    )
    np.testing.assert_allclose(
        model.hf_at_usage(t, 0.1), model.hf(t * 2.0) * 2.0, rtol=1e-15
    )
    # B10 along the path is the combined clock's, divided by the rate
    assert model.qf_at_usage(0.1, 0.1) == pytest.approx(model.qf(0.1) / 2.0)
    assert model.sf_at_usage(model.qf_at_usage(0.1, 0.5), 0.5) == (
        pytest.approx(0.9)
    )
    assert model.mean_at_usage(0.0) == pytest.approx(model.mean())
    assert model.equivalent(100.0, 3.0) == 130.0
    hours, starts = model.random(5, rate=0.2, random_state=3)
    np.testing.assert_allclose(starts, 0.2 * hours)


def test_usage_bounds_include_the_uncertainty_in_k():
    d = _fleet(500, seed=31)
    model = _fit(d)
    t, rate = 400.0, 0.5
    full = model.cb_at_usage(t, rate)
    combined = model.cb(model.equivalent(t, rate * t))
    # k's variance and covariances make the difference: with them set to
    # 0 the usage band is the combined-scale band at T = t (1 + k rate)
    assert not np.allclose(full, combined, rtol=1e-3)
    cov = model.covariance().copy()
    cov[-1, :] = cov[:, -1] = 0.0
    model._covariance = cov
    np.testing.assert_allclose(model.cb_at_usage(t, rate), combined, rtol=1e-9)
    lo, hi = model.quantile_cb_at_usage(0.1, rate)
    assert lo < model.qf_at_usage(0.1, rate) < hi
    # a one-sided bound is that end of the two-sided one at twice alpha
    assert model.quantile_cb_at_usage(0.1, rate, bound="lower") == (
        pytest.approx(model.quantile_cb_at_usage(0.1, rate, alpha_ci=0.1)[0])
    )


@pytest.mark.parametrize(
    "x2, message",
    [
        (lambda h: 0.0 * h, "Every unit has x2 = 0"),
        (lambda h: 0.1 * h, "at the same rate"),
    ],
)
def test_refuses_data_that_do_not_identify_k(x2, message):
    d = _fleet(100, seed=37)
    with pytest.raises(ValueError, match=message):
        sp.WeibullEquivalence.fit(d["x1"], x2(d["x1"]), c=d["c"])


def test_refusals_say_what_to_do():
    d = _fleet(50, seed=41)
    h, s, c = d["x1"], d["x2"], d["c"]
    with pytest.raises(ValueError, match="Interval-censored"):
        sp.WeibullEquivalence.fit(h, s, c=np.full(50, 2))
    with pytest.raises(ValueError, match="missing"):
        sp.WeibullEquivalence.fit(np.r_[np.nan, h[1:]], s, c=c)
    with pytest.raises(ValueError, match="negative"):
        sp.WeibullEquivalence.fit(-h, s, c=c)
    with pytest.raises(ValueError, match="no failures"):
        sp.WeibullEquivalence.fit(h, s, c=np.ones(50))
    with pytest.raises(ValueError, match="not available with the Normal"):
        sp.Equivalence(sp.Normal)
    with pytest.raises(ValueError, match="method='lr' is available for k"):
        _fit(d).param_cb("alpha", method="lr")
    with pytest.raises(ValueError, match="Wald"):
        _fit(d).cb([100.0], method="lr")
    with pytest.raises(ValueError, match="does not take weights"):
        sp.WeibullEquivalence.fit(h, s, w=np.ones(50))
