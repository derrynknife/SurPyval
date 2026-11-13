"""Time-varying covariates in the proportional-intensity models.

``fit_tvc`` (start-stop rows), ``fit_tvc_timeline`` and their DataFrame
forms fit ``ProportionalIntensityHPP`` and ``ProportionalIntensityNHPP``
to items whose covariates are piecewise constant in time; ``cif_tvc``,
``iif_tvc`` and ``cb_tvc`` predict along a covariate schedule.

The checks: a closed form (exact arithmetic), a likelihood written here
with numpy and scipy, recovery of a known truth, and the identities the
fit and the predictions must keep (constant covariates are the
fixed-covariate model; splitting a row, reordering rows and changing the
time unit change nothing they should not; saving and loading keeps every
prediction).
"""

import json
import math
import pickle
import warnings
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime, minimize
from scipy.special import gammaln
from scipy.stats import norm

import surpyval
from surpyval import CovariatePath, StepSchedule
from surpyval.recurrent import (
    CoxLewis,
    CrowAMSAA,
    Duane,
    ProportionalIntensityHPP,
    ProportionalIntensityNHPP,
)
from surpyval.recurrent.regression.tvc import (
    handle_recurrent_tvc,
    handle_recurrent_tvc_timeline,
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Three items whose binary covariate switches within them; the times are
# whole numbers so the closed form below is exact.
HAND = {
    "i": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3],
    "xl": [0, 6, 10, 13, 17, 0, 3, 7, 10, 16, 21, 2, 9, 12],
    "xr": [6, 10, 13, 17, 20, 3, 7, 10, 16, 21, 25, 9, 12, 19],
    "c": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    "Z": [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
}


def _hand_closed_form(d=HAND):
    """Events and exposure at each level, as exact fractions."""
    events = {0: Fraction(0), 1: Fraction(0)}
    exposure = {0: Fraction(0), 1: Fraction(0)}
    counts = d.get("n", [1] * len(d["i"]))
    for xl, xr, c, z, n in zip(d["xl"], d["xr"], d["c"], d["Z"], counts):
        exposure[z] += Fraction(xr) - Fraction(xl)
        events[z] += Fraction(1) if c == 0 else Fraction(n) * (c == 2)
    return events, exposure


def _fixed_as_rows(x, i, c, Z, tl=0.0):
    """Recurrent event rows with fixed covariates as start-stop rows: the
    interval from each of an item's rows to its next."""
    x, i, c = np.asarray(x, float), np.asarray(i), np.asarray(c)
    Z = np.asarray(Z, float).reshape(len(x), -1)
    xl = np.empty_like(x)
    for item in np.unique(i):
        rows = np.flatnonzero(i == item)
        rows = rows[np.argsort(x[rows], kind="stable")]
        xl[rows] = np.r_[tl, x[rows][:-1]]
    return {"i": i, "xl": xl, "xr": x, "c": c, "Z": Z}


FIXED = {
    "x": [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60, 5, 18, 30, 50, 60.0],
    "i": [1] * 6 + [2] * 5 + [3] * 5,
    "c": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    "Z": [0.0] * 6 + [1.0] * 5 + [0.5] * 5,
}


def _mixed_rows():
    """Start-stop rows with everything the fit takes: two covariates that
    change, delayed entry (item 2), a gap (item 3, from 14 to 20), rows
    cut where nothing changes, and counted failures (c = 2)."""
    rows = [
        # item, xl, xr, c, n, z1, z2
        (1, 0.0, 4.0, 0, 1, 0.0, 0.2),
        (1, 4.0, 9.0, 1, 1, 0.0, 0.2),
        (1, 9.0, 11.5, 0, 1, 1.0, 0.2),
        (1, 11.5, 18.0, 0, 1, 1.0, 0.7),
        (1, 18.0, 30.0, 2, 3, 1.0, 0.7),
        (2, 5.0, 7.5, 0, 1, 1.0, 0.5),
        (2, 7.5, 16.0, 0, 1, 1.0, 0.5),
        (2, 16.0, 22.0, 1, 1, 0.0, 0.9),
        (2, 22.0, 26.0, 0, 1, 0.0, 0.9),
        (2, 26.0, 28.0, 1, 1, 0.0, 0.1),
        (3, 0.0, 6.0, 2, 2, 0.0, 0.4),
        (3, 6.0, 14.0, 0, 1, 1.0, 0.4),
        (3, 20.0, 23.0, 0, 1, 1.0, 0.8),
        (3, 23.0, 31.0, 2, 0, 0.0, 0.8),
        (4, 0.0, 12.0, 0, 1, 0.0, 0.6),
        (4, 12.0, 21.0, 0, 1, 1.0, 0.3),
        (4, 21.0, 27.0, 0, 1, 1.0, 0.3),
        (4, 27.0, 33.0, 1, 1, 0.0, 0.3),
    ]
    a = np.array(rows)
    return {
        "i": a[:, 0].astype(int),
        "xl": a[:, 1],
        "xr": a[:, 2],
        "c": a[:, 3].astype(int),
        "n": a[:, 4],
        "Z": a[:, 5:],
    }


def _fit(fitter, d, **kwargs):
    return fitter.fit_tvc(
        d["i"], d["xl"], d["xr"], d["c"], d["Z"], d.get("n"), **kwargs
    )


BASELINES = {
    "HPP": (ProportionalIntensityHPP, {}),
    "Duane": (ProportionalIntensityNHPP, {"baseline": Duane}),
    "CrowAMSAA": (ProportionalIntensityNHPP, {"baseline": CrowAMSAA}),
    "CoxLewis": (ProportionalIntensityNHPP, {"baseline": CoxLewis}),
}


def _baseline_functions(name):
    """Each baseline's cumulative and log intensity, written out here."""
    if name == "HPP":
        return (lambda t, lam: lam * t), (lambda t, lam: np.log(lam) + 0 * t)
    if name == "Duane":
        return (
            lambda t, alpha, b: b * t**alpha,
            lambda t, alpha, b: np.log(alpha * b) + (alpha - 1) * np.log(t),
        )
    if name == "CrowAMSAA":
        return (
            lambda t, alpha, beta: (t / alpha) ** beta,
            lambda t, alpha, beta: (
                np.log(beta / alpha) + (beta - 1) * np.log(t / alpha)
            ),
        )
    raise KeyError(name)


def _oracle(d, name):
    """The negative log-likelihood of start-stop rows, written directly:
    the sum over failures of the log intensity, less the sum over rows of
    the expected count, plus the Poisson terms of the counted rows."""
    cif, log_iif = _baseline_functions(name)
    xl, xr = np.asarray(d["xl"], float), np.asarray(d["xr"], float)
    c = np.asarray(d["c"])
    Z = np.asarray(d["Z"], float).reshape(len(xl), -1)
    n = np.asarray(d.get("n", np.ones(len(xl))), float)
    k = 1 if name == "HPP" else 2

    def neg_ll(params):
        theta, beta = params[:k], params[k:]
        ll = 0.0
        for row in range(len(xl)):
            eta = float(Z[row] @ beta)
            mean = (cif(xr[row], *theta) - cif(xl[row], *theta)) * math.exp(
                eta
            )
            ll -= mean
            if c[row] == 0:
                ll += log_iif(xr[row], *theta) + eta
            elif c[row] == 2:
                ll += n[row] * math.log(mean) - gammaln(n[row] + 1)
        return -ll

    return neg_ll


# ---------------------------------------------------------------------------
# A hand-checked case: the HPP with one binary covariate
# ---------------------------------------------------------------------------


def test_hpp_binary_covariate_matches_the_closed_form():
    # The rate at each level is its events over its exposure, and the
    # coefficient the log of their ratio.
    events, exposure = _hand_closed_form()
    assert events == {0: 4, 1: 5}
    assert exposure == {0: 32, 1: 30}
    rate_0 = events[0] / exposure[0]
    rate_1 = events[1] / exposure[1]
    assert rate_0 == Fraction(1, 8) and rate_1 == Fraction(1, 6)

    model = _fit(ProportionalIntensityHPP, HAND)
    assert model.maximum == "verified"
    assert model.is_tvc and model.n_items == 3
    np.testing.assert_allclose(model.params, [float(rate_0)], rtol=1e-6)
    np.testing.assert_allclose(
        model.coeffs, [math.log(rate_1 / rate_0)], rtol=1e-6
    )
    # At the maximum each level's expected count is its number of events.
    log_likelihood = (
        events[0] * math.log(rate_0)
        + events[1] * math.log(rate_1)
        - (events[0] + events[1])
    )
    assert model.log_likelihood == pytest.approx(log_likelihood, abs=1e-9)
    assert model.aic() == pytest.approx(4 - 2 * log_likelihood, abs=1e-8)
    assert model.bic() == pytest.approx(
        2 * math.log(9) - 2 * log_likelihood, abs=1e-8
    )
    # The information: var(log rate_0) = 1 / events_0, and the two
    # levels' rates are independent.
    cov = model.covariance()
    var_rate = float(rate_0**2 / events[0])
    var_beta = float(1 / events[0] + 1 / events[1])
    np.testing.assert_allclose(cov[0, 0], var_rate, rtol=1e-4)
    np.testing.assert_allclose(cov[1, 1], var_beta, rtol=1e-4)
    np.testing.assert_allclose(cov[0, 1], -float(rate_0 / events[0]), 1e-4)
    summary = model.summary()
    assert list(summary.index) == ["lambda", "coef_0"]
    np.testing.assert_allclose(
        summary["se"], [math.sqrt(var_rate), math.sqrt(var_beta)], rtol=1e-4
    )


def test_hpp_counted_failures_match_the_closed_form():
    # The same closed form with failures counted over an interval: a
    # count adds its failures and its exposure to its level.
    d = {
        "i": [1, 1, 1, 2, 2, 2],
        "xl": [0, 8, 14, 0, 5, 9],
        "xr": [8, 14, 20, 5, 9, 16],
        "c": [2, 0, 2, 0, 2, 1],
        "n": [3, 1, 4, 1, 0, 1],
        "Z": [0, 1, 1, 1, 0, 0],
    }
    events, exposure = _hand_closed_form(d)
    assert events == {0: 3, 1: 6} and exposure == {0: 19, 1: 17}
    rate_0, rate_1 = events[0] / exposure[0], events[1] / exposure[1]
    model = _fit(ProportionalIntensityHPP, d)
    np.testing.assert_allclose(model.params, [float(rate_0)], rtol=1e-6)
    np.testing.assert_allclose(
        model.coeffs, [math.log(rate_1 / rate_0)], rtol=1e-6
    )
    assert model._n_obs == 9


def test_cb_tvc_matches_the_closed_form_delta_method():
    # Along a schedule spending a at level 0 and b at level 1 the expected
    # count is rate_0 (a + e^beta b); its log has gradient (1, w) in
    # (log rate_0, beta), w the share of the count at level 1.
    events, exposure = _hand_closed_form()
    rate_0 = float(events[0] / exposure[0])
    ratio = float((events[1] / exposure[1]) / (events[0] / exposure[0]))
    a, b = 12.0, 18.0
    count = rate_0 * (a + ratio * b)
    w = ratio * b / (a + ratio * b)
    e0, e1 = float(events[0]), float(events[1])
    var_log = 1 / e0 - 2 * w / e0 + w**2 * (1 / e0 + 1 / e1)
    z = norm.ppf(0.975)
    expected = count * np.exp(np.array([-1, 1]) * z * math.sqrt(var_log))

    model = _fit(ProportionalIntensityHPP, HAND)
    plan = StepSchedule.from_changepoints([0, 12], [0.0, 1.0])
    assert model.cif_tvc(30.0, plan) == pytest.approx(count, rel=1e-6)
    np.testing.assert_allclose(model.cb_tvc(30.0, plan), expected, rtol=1e-4)
    # The intensity at level 1: var(log rate_1) = 1 / events_1.
    rate_1 = rate_0 * ratio
    expected = rate_1 * np.exp(np.array([-1, 1]) * z / math.sqrt(e1))
    np.testing.assert_allclose(
        model.cb_tvc(20.0, plan, on="iif"), expected, rtol=1e-4
    )


# ---------------------------------------------------------------------------
# An independent likelihood: the fit's maximum must be its maximum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["HPP", "Duane", "CrowAMSAA"])
def test_the_fit_maximises_an_independent_likelihood(name):
    fitter, kwargs = BASELINES[name]
    d = _mixed_rows()
    model = _fit(fitter, d, **kwargs)
    assert model.maximum == "verified"
    oracle = _oracle(d, name)
    fitted = np.r_[model.params, model.coeffs]
    # The same likelihood value, constants included ...
    assert model.neg_ll() == pytest.approx(oracle(fitted), abs=1e-9)
    # ... a stationary point of the oracle ...
    scale = np.maximum(np.abs(fitted), 1e-3)
    gradient = approx_fprime(fitted, oracle, 1e-7 * scale) * scale
    np.testing.assert_allclose(gradient, 0.0, atol=2e-4)
    # ... and nothing higher found by searching the oracle from it, or
    # from points around it.
    rng = np.random.default_rng(5)
    best = oracle(fitted)
    for _ in range(4):
        start = fitted * np.exp(rng.normal(0, 0.05, fitted.size))
        res = minimize(oracle, start, method="Nelder-Mead")
        assert res.fun >= best - 1e-7, (res.x, res.fun, best)


# ---------------------------------------------------------------------------
# Recovery of a known truth
# ---------------------------------------------------------------------------


def _simulate_switching(rng, n_items, alpha, beta, coef):
    """Items with a power-law baseline ``(t / alpha) ** beta`` and one
    binary covariate that switches at known times: start-stop rows cut at
    every failure and every switch."""
    out = {k: [] for k in ("i", "xl", "xr", "c", "Z")}
    for item in range(n_items):
        end = rng.uniform(30.0, 60.0)
        switches = np.sort(rng.uniform(0.0, end, rng.integers(1, 4)))
        edges = np.r_[0.0, switches, end]
        level = float(rng.integers(0, 2))
        for a, b in zip(edges[:-1], edges[1:]):
            lo, hi = (a / alpha) ** beta, (b / alpha) ** beta
            k = rng.poisson((hi - lo) * math.exp(coef * level))
            fails = np.sort(alpha * rng.uniform(lo, hi, k) ** (1 / beta))
            cuts = np.r_[a, fails, b]
            out["i"] += [item] * (k + 1)
            out["xl"] += list(cuts[:-1])
            out["xr"] += list(cuts[1:])
            out["c"] += [0] * k + [1]
            out["Z"] += [level] * (k + 1)
            level = 1.0 - level
    return {k: np.asarray(v) for k, v in out.items()}


def test_recovers_a_trend_and_a_switching_covariate():
    # Seed, sizes and truth fixed before the result was seen.
    truth = {"alpha": 15.0, "beta": 1.5, "coef_0": 0.7}
    rng = np.random.default_rng(20261111)
    d = _simulate_switching(rng, 60, truth["alpha"], truth["beta"], 0.7)
    model = _fit(ProportionalIntensityNHPP, d, baseline=CrowAMSAA)
    assert model.maximum == "verified"
    assert model.parameter_names == ["alpha", "beta", "coef_0"]
    for name, value in truth.items():
        lower, upper = model.param_cb(name)
        assert lower < value < upper, (name, lower, upper)
    # The expected count along a schedule, against the truth's.
    plan = StepSchedule.from_changepoints([0, 20, 35], [0.0, 1.0, 0.0])
    t = np.array([20.0, 35.0, 50.0])
    base = (np.r_[0.0, t] / 15.0) ** 1.5
    true_count = np.cumsum(np.diff(base) * np.exp(0.7 * np.array([0, 1, 0])))
    bounds = model.cb_tvc(t, plan)
    assert np.all(bounds[:, 0] < true_count), (bounds, true_count)
    assert np.all(true_count < bounds[:, 1]), (bounds, true_count)


# ---------------------------------------------------------------------------
# Identities of the fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(BASELINES))
@pytest.mark.parametrize("tl", [0.0, 2.0])
def test_constant_covariates_are_the_fixed_covariate_fit(name, tl):
    fitter, kwargs = BASELINES[name]
    fixed = fitter.fit(
        FIXED["x"], FIXED["Z"], i=FIXED["i"], c=FIXED["c"], tl=tl, **kwargs
    )
    model = _fit(fitter, _fixed_as_rows(tl=tl, **FIXED), **kwargs)
    assert model.maximum == fixed.maximum == "verified"
    assert model.neg_ll() == pytest.approx(fixed.neg_ll(), abs=1e-7)
    np.testing.assert_allclose(model.params, fixed.params, rtol=2e-4)
    np.testing.assert_allclose(model.coeffs, fixed.coeffs, rtol=2e-4)
    # The same likelihood function, not only the same maximum.
    for shift in (0.9, 1.0, 1.2):
        at = np.r_[fixed.params, fixed.coeffs] * shift
        assert model._neg_ll(at) == pytest.approx(fixed._neg_ll(at), abs=1e-9)
    assert model.aic() == pytest.approx(fixed.aic(), abs=1e-6)
    assert model.bic() == pytest.approx(fixed.bic(), abs=1e-6)
    np.testing.assert_allclose(
        model.standard_errors(), fixed.standard_errors(), rtol=1e-3
    )
    x, z = np.array([5.0, 20.0, 45.0]), [0.5]
    np.testing.assert_allclose(model.cif(x, z), fixed.cif(x, z), rtol=2e-4)
    np.testing.assert_allclose(
        model.cif_cb(x, z), fixed.cif_cb(x, z), rtol=1e-3
    )


def test_a_failure_truncated_item_is_the_fixed_covariate_fit():
    # No end-of-record row: the item's record ends at its last failure.
    x = [4, 9, 15, 22, 3, 11, 19, 30.0]
    i = [1, 1, 1, 1, 2, 2, 2, 2]
    c = [0, 0, 0, 0, 0, 0, 0, 1]
    Z = [0.0] * 4 + [1.0] * 4
    fixed = ProportionalIntensityNHPP.fit(x, Z, i=i, c=c, baseline=CrowAMSAA)
    model = _fit(
        ProportionalIntensityNHPP,
        _fixed_as_rows(x, i, c, Z),
        baseline=CrowAMSAA,
    )
    assert model.neg_ll() == pytest.approx(fixed.neg_ll(), abs=1e-7)
    np.testing.assert_allclose(model.params, fixed.params, rtol=2e-4)


@pytest.mark.parametrize("name", ["HPP", "CrowAMSAA"])
def test_splitting_a_row_changes_nothing(name):
    fitter, kwargs = BASELINES[name]
    d = _mixed_rows()
    ref = _fit(fitter, d, **kwargs)
    # Every row with no count cut at its middle: the first piece ends
    # with no failure, the second as the row did.
    cut = np.flatnonzero(d["c"] != 2)
    mid = (d["xl"][cut] + d["xr"][cut]) / 2
    split = {
        "i": np.r_[d["i"], d["i"][cut]],
        "xl": np.r_[d["xl"], mid],
        "xr": np.r_[np.where(d["c"] != 2, 0, d["xr"]), d["xr"][cut]],
        "c": np.r_[np.where(d["c"] != 2, 1, d["c"]), d["c"][cut]],
        "n": np.r_[d["n"], d["n"][cut]],
        "Z": np.r_[d["Z"], d["Z"][cut]],
    }
    split["xr"][cut] = mid
    assert len(split["i"]) == len(d["i"]) + len(cut)
    model = _fit(fitter, split, **kwargs)
    at = np.r_[ref.params, ref.coeffs]
    assert model._neg_ll(at) == pytest.approx(ref._neg_ll(at), abs=1e-9)
    assert model.neg_ll() == pytest.approx(ref.neg_ll(), abs=1e-8)
    np.testing.assert_allclose(model.params, ref.params, rtol=1e-4)
    np.testing.assert_allclose(model.coeffs, ref.coeffs, rtol=1e-4, atol=1e-6)
    # The criteria count failures, not rows.
    assert model.bic() == pytest.approx(ref.bic(), abs=1e-7)
    assert model.aic_c() == pytest.approx(ref.aic_c(), abs=1e-7)


def test_a_counted_row_of_no_failures_is_a_row_with_no_failure():
    d = _mixed_rows()
    assert d["n"][13] == 0 and d["c"][13] == 2
    ref = _fit(ProportionalIntensityHPP, d)
    plain = {**d, "c": d["c"].copy(), "n": d["n"].copy()}
    plain["c"][13], plain["n"][13] = 1, 1
    model = _fit(ProportionalIntensityHPP, plain)
    at = np.r_[ref.params, ref.coeffs]
    assert model._neg_ll(at) == pytest.approx(ref._neg_ll(at), abs=1e-12)


@pytest.mark.parametrize("name", ["HPP", "Duane", "CrowAMSAA"])
def test_the_order_of_the_rows_does_not_matter(name):
    fitter, kwargs = BASELINES[name]
    d = _mixed_rows()
    ref = _fit(fitter, d, **kwargs)
    order = np.random.default_rng(3).permutation(len(d["i"]))
    model = _fit(fitter, {k: v[order] for k, v in d.items()}, **kwargs)
    np.testing.assert_array_equal(model.params, ref.params)
    np.testing.assert_array_equal(model.coeffs, ref.coeffs)


@pytest.mark.parametrize(
    "name, bounds",
    [
        ("HPP", True),
        ("CrowAMSAA", True),
        ("Duane", True),
    ],
)
def test_a_change_of_time_unit_rescales_time_only(name, bounds):
    # Months to hours: the coefficients, the expected counts and the
    # criteria's differences stay; rates and scales move by the factor.
    fitter, kwargs = BASELINES[name]
    k = 730.0
    d = _mixed_rows()
    ref = _fit(fitter, d, **kwargs)
    model = _fit(fitter, {**d, "xl": d["xl"] * k, "xr": d["xr"] * k}, **kwargs)
    assert model.maximum == "verified"
    np.testing.assert_allclose(model.coeffs, ref.coeffs, rtol=1e-4, atol=1e-6)
    # Each exactly timed failure's density is per unit of time.
    exact = int((d["c"] == 0).sum())
    assert model.neg_ll() == pytest.approx(
        ref.neg_ll() + exact * math.log(k), abs=1e-6
    )
    if name == "HPP":
        np.testing.assert_allclose(model.params * k, ref.params, rtol=1e-5)
    elif name == "CrowAMSAA":
        np.testing.assert_allclose(
            model.params, ref.params * [k, 1.0], rtol=1e-4
        )
    plan = StepSchedule.from_changepoints([0, 10], [[0, 0.5], [1, 0.5]])
    plan_k = StepSchedule.from_changepoints([0, 10 * k], [[0, 0.5], [1, 0.5]])
    t = np.array([4.0, 10.0, 25.0])
    np.testing.assert_allclose(
        model.cif_tvc(t * k, plan_k), ref.cif_tvc(t, plan), rtol=1e-4
    )
    np.testing.assert_allclose(
        model.iif_tvc(t * k, plan_k) * k, ref.iif_tvc(t, plan), rtol=1e-4
    )
    if bounds:
        np.testing.assert_allclose(
            model.cb_tvc(t * k, plan_k), ref.cb_tvc(t, plan), rtol=1e-3
        )


def test_a_covariate_scale_rescales_its_coefficient_only():
    d = _mixed_rows()
    ref = _fit(ProportionalIntensityNHPP, d, baseline=CrowAMSAA)
    scaled = {**d, "Z": d["Z"] * [1e-3, 40.0]}
    model = _fit(ProportionalIntensityNHPP, scaled, baseline=CrowAMSAA)
    assert model.neg_ll() == pytest.approx(ref.neg_ll(), abs=1e-7)
    np.testing.assert_allclose(
        model.coeffs * [1e-3, 40.0], ref.coeffs, rtol=1e-3, atol=1e-5
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _timeline_of(d):
    """Contiguous start-stop rows (no gaps, c of 0 or 1) as a timeline."""
    rows = []
    for item in np.unique(d["i"]):
        at = np.flatnonzero(np.asarray(d["i"]) == item)
        rows.append((item, d["xl"][at[0]], 1, d["Z"][at[0]]))
        for j, k in enumerate(at):
            nxt = d["Z"][at[j + 1]] if j + 1 < len(at) else -99.0
            rows.append((item, d["xr"][k], d["c"][k], nxt))
    i, x, c, Z = zip(*rows)
    return np.array(i), np.array(x, float), np.array(c), np.array(Z, float)


@pytest.mark.parametrize("name", ["HPP", "CrowAMSAA"])
def test_a_timeline_is_its_start_stop_rows(name):
    fitter, kwargs = BASELINES[name]
    ref = _fit(fitter, HAND, **kwargs)
    i, x, c, Z = _timeline_of(HAND)
    # Shuffled: the timeline is sorted within each item.
    order = np.random.default_rng(1).permutation(len(i))
    model = fitter.fit_tvc_timeline(
        i[order], x[order], Z[order], c[order], **kwargs
    )
    np.testing.assert_array_equal(model.params, ref.params)
    np.testing.assert_array_equal(model.coeffs, ref.coeffs)
    assert model.is_tvc


def test_a_timeline_takes_counted_failures():
    i, xl, xr, c, Z, n = handle_recurrent_tvc_timeline(
        i=[7, 7, 7, 7],
        x=[2.0, 5.0, 9.0, 12.0],
        Z=[[0.0], [1.0], [1.0], [np.nan]],
        c=[1, 2, 0, 1],
        n=[1, 3, 1, 1],
    )
    np.testing.assert_array_equal(xl, [2.0, 5.0, 9.0])
    np.testing.assert_array_equal(xr, [5.0, 9.0, 12.0])
    np.testing.assert_array_equal(c, [2, 0, 1])
    np.testing.assert_array_equal(n, [3, 1, 1])
    np.testing.assert_array_equal(Z.ravel(), [0.0, 1.0, 1.0])


@pytest.mark.parametrize("name", ["HPP", "Duane"])
def test_the_data_frame_entry_points_are_the_array_fits(name):
    fitter, kwargs = BASELINES[name]
    d = _mixed_rows()
    ref = _fit(fitter, d, **kwargs)
    df = pd.DataFrame(
        {
            "unit": d["i"],
            "start": d["xl"],
            "stop": d["xr"],
            "status": d["c"],
            "count": d["n"],
            "load": d["Z"][:, 0],
            "dust": d["Z"][:, 1],
        }
    )
    model = fitter.fit_tvc_from_df(
        df,
        i_col="unit",
        xl_col="start",
        xr_col="stop",
        c_col="status",
        Z_cols=["load", "dust"],
        n_col="count",
        unit="months",
        **kwargs,
    )
    np.testing.assert_array_equal(model.params, ref.params)
    np.testing.assert_array_equal(model.coeffs, ref.coeffs)
    assert model.feature_names == ["load", "dust"]
    assert model.parameter_names[-2:] == ["load", "dust"]
    assert model.unit == "months"
    assert list(model.summary().index[-2:]) == ["load", "dust"]
    # A DataFrame Z names the coefficients too.
    named = fitter.fit_tvc(
        d["i"], d["xl"], d["xr"], d["c"], df[["load", "dust"]], d["n"]
    )
    assert named.feature_names == ["load", "dust"]

    ref = _fit(fitter, HAND, **kwargs)
    i, x, c, Z = _timeline_of(HAND)
    log = pd.DataFrame({"unit": i, "t": x, "status": c, "load": Z})
    model = fitter.fit_tvc_timeline_from_df(
        log, i_col="unit", x_col="t", Z_cols="load", c_col="status", **kwargs
    )
    np.testing.assert_array_equal(model.params, ref.params)
    np.testing.assert_array_equal(model.coeffs, ref.coeffs)
    assert model.feature_names == ["load"]
    with pytest.raises(ValueError, match="Z_cols entry 'nope'"):
        fitter.fit_tvc_from_df(
            df, "unit", "start", "stop", "status", Z_cols=["nope"]
        )
    with pytest.raises(ValueError, match="xl_col='nope'"):
        fitter.fit_tvc_from_df(df, "unit", "nope", "stop", "status", "load")
    with pytest.raises(TypeError, match="'xl_col'"):
        fitter.fit_tvc_from_df(
            df, "unit", "start", "stop", "status", "load", xl=df["start"]
        )


def test_init_is_taken_and_checked():
    d = _mixed_rows()
    ref = _fit(ProportionalIntensityNHPP, d, baseline=CrowAMSAA)
    model = _fit(
        ProportionalIntensityNHPP,
        d,
        baseline=CrowAMSAA,
        init=[20.0, 1.0, 0.1, 0.1],
    )
    assert model.neg_ll() == pytest.approx(ref.neg_ll(), abs=1e-7)
    with pytest.raises(ValueError, match="init must have 4 values"):
        _fit(ProportionalIntensityNHPP, d, baseline=CrowAMSAA, init=[1.0])
    with pytest.raises(ValueError, match="init must have 3 values"):
        _fit(ProportionalIntensityHPP, d, init=[1.0])
    with pytest.raises(TypeError, match="baseline"):
        _fit(ProportionalIntensityNHPP, d, baseline="Duane")


def test_a_column_the_data_cannot_determine_is_aliased():
    d = _mixed_rows()
    twice = {**d, "Z": np.c_[d["Z"], d["Z"][:, 0], np.ones(len(d["i"]))]}
    with pytest.warns(UserWarning, match=r"column\(s\) 2, 3 of Z"):
        model = _fit(ProportionalIntensityHPP, twice)
    ref = _fit(ProportionalIntensityHPP, d)
    np.testing.assert_array_equal(model.aliased, [2, 3])
    assert np.isnan(model.coeffs[2:]).all()
    np.testing.assert_allclose(model.coeffs[:2], ref.coeffs, rtol=1e-6)
    assert model.aic() == pytest.approx(ref.aic(), abs=1e-8)
    plan = StepSchedule.from_changepoints([0, 8], [[0, 0.3], [1, 0.3]])
    wide = StepSchedule.from_changepoints(
        [0, 8], [[0, 0.3, 0, 1], [1, 0.3, 1, 1]]
    )
    np.testing.assert_allclose(
        model.cif_tvc([5.0, 20.0], wide), ref.cif_tvc([5.0, 20.0], plan)
    )
    np.testing.assert_allclose(
        model.cb_tvc([5.0, 20.0], wide),
        ref.cb_tvc([5.0, 20.0], plan),
        rtol=1e-4,
    )


# ---------------------------------------------------------------------------
# Prediction along a schedule
# ---------------------------------------------------------------------------


def _models():
    """A model fitted to changing covariates and one to fixed ones, for
    each baseline: the path methods serve both."""
    out = []
    for name, (fitter, kwargs) in BASELINES.items():
        d = {**_mixed_rows()}
        d["Z"] = d["Z"][:, :1]
        out.append(pytest.param(_fit(fitter, d, **kwargs), id=f"{name}-tvc"))
        fixed = fitter.fit(
            FIXED["x"], FIXED["Z"], i=FIXED["i"], c=FIXED["c"], **kwargs
        )
        out.append(pytest.param(fixed, id=f"{name}-fixed"))
    return out


MODELS = _models()
TIMES = np.array([0.0, 1.0, 7.5, 12.0, 30.0, 55.0])


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("start", [-np.inf, -5.0, 0.0])
def test_a_constant_schedule_is_the_fixed_covariate_prediction(model, start):
    z = [0.6]
    flat = (
        StepSchedule.constant(z)
        if start == -np.inf
        else StepSchedule.from_changepoints([start], [z])
    )
    np.testing.assert_allclose(
        model.cif_tvc(TIMES, flat), model.cif(TIMES, z), rtol=1e-12
    )
    np.testing.assert_allclose(
        model.iif_tvc(TIMES[1:], flat), model.iif(TIMES[1:], z), rtol=1e-12
    )
    for kwargs in (
        {},
        {"alpha_ci": 0.2},
        {"bound": "lower"},
        {"bound": "upper", "alpha_ci": 0.1},
    ):
        np.testing.assert_allclose(
            model.cb_tvc(TIMES, flat, **kwargs),
            model.cif_cb(TIMES, z, **kwargs),
            rtol=1e-9,
        )
        np.testing.assert_allclose(
            model.cb_tvc(TIMES[1:], flat, on="iif", **kwargs),
            model.iif_cb(TIMES[1:], z, **kwargs),
            rtol=1e-9,
        )
    # The array form of the same schedule.
    if np.isfinite(start):
        np.testing.assert_array_equal(
            model.cif_tvc(TIMES, [z], xl=[start]), model.cif_tvc(TIMES, flat)
        )


@pytest.mark.parametrize("model", MODELS)
def test_a_schedule_adds_up_its_segments(model):
    # Low to 10, high to 25, low after: each segment adds the fixed
    # covariate's expected count over it.
    plan = StepSchedule.from_changepoints([0, 10, 25], [0.0, 1.0, 0.0])
    low, high = [0.0], [1.0]
    t = np.array([4.0, 10.0, 18.0, 25.0, 40.0])
    expected = np.array(
        [
            model.cif(4.0, low),
            model.cif(10.0, low),
            model.cif(10.0, low) + model.cif(18.0, high) - model.cif(10, high),
            model.cif(10.0, low) + model.cif(25.0, high) - model.cif(10, high),
            model.cif(10.0, low)
            + model.cif(25.0, high)
            - model.cif(10.0, high)
            + model.cif(40.0, low)
            - model.cif(25.0, low),
        ]
    )
    np.testing.assert_allclose(model.cif_tvc(t, plan), expected, rtol=1e-12)
    np.testing.assert_allclose(
        model.cif_tvc(t, [[0.0], [1.0], [0.0]], xl=[0, 10, 25]),
        expected,
        rtol=1e-12,
    )
    # The intensity is the current covariate's; at a step, the new one.
    np.testing.assert_allclose(
        model.iif_tvc([4.0, 10.0, 18.0, 25.0, 40.0], plan),
        [
            model.iif(4.0, low),
            model.iif(10.0, high),
            model.iif(18.0, high),
            model.iif(25.0, low),
            model.iif(40.0, low),
        ],
        rtol=1e-12,
    )
    # The cumulative count never decreases and the query order is kept.
    grid = np.linspace(0.5, 50.0, 200)
    assert np.all(np.diff(model.cif_tvc(grid, plan)) >= 0)
    order = np.random.default_rng(0).permutation(len(t))
    np.testing.assert_array_equal(
        model.cif_tvc(t[order], plan), model.cif_tvc(t, plan)[order]
    )


@pytest.mark.parametrize("model", MODELS)
def test_given_counts_from_an_age(model):
    plan = StepSchedule.from_changepoints([0, 10, 25], [0.0, 1.0, 0.0])
    t = np.array([5.0, 12.0, 12.5, 30.0, 44.0])
    total = model.cif_tvc(t, plan)
    at_given = model.cif_tvc(12.0, plan)
    expected = np.where(t <= 12.0, 0.0, total - at_given)
    np.testing.assert_allclose(
        model.cif_tvc(t, plan, given=12.0), expected, rtol=1e-10, atol=1e-12
    )
    # The schedule need only start by the age given.
    later = StepSchedule.from_changepoints([12, 25], [1.0, 0.0])
    np.testing.assert_allclose(
        model.cif_tvc(t, later, given=12.0), expected, rtol=1e-10, atol=1e-12
    )
    with pytest.raises(ValueError, match="after the age given"):
        model.cif_tvc(t, later, given=11.0)
    with pytest.raises(ValueError, match="after the time origin"):
        model.cif_tvc(t, later)
    with pytest.raises(ValueError, match="after the time origin"):
        model.iif_tvc(t, later)
    assert np.isnan(model.cif_tvc(t, plan, given=np.nan)).all()
    with pytest.raises(ValueError, match="given must be one age"):
        model.cif_tvc(t, plan, given="soon")
    # The bounds contain the count, and have no width where it is 0.
    cb = model.cb_tvc(t, later, given=12.0)
    assert cb.shape == (5, 2)
    np.testing.assert_array_equal(cb[:2], 0.0)
    assert np.all(cb[2:, 0] < expected[2:]) and np.all(
        expected[2:] < cb[2:, 1]
    )
    with pytest.raises(ValueError, match="given applies to on='cif'"):
        model.cb_tvc(t, plan, given=12.0, on="iif")


@pytest.mark.parametrize("model", MODELS)
def test_a_repeating_schedule_is_its_pattern_repeated(model):
    season = StepSchedule.cyclic([0.0, 8.0], [[0.0], [1.0]], period=12.0)
    spelt = StepSchedule.from_changepoints(
        [0, 8, 12, 20, 24, 32, 36], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    )
    t = np.array([3.0, 9.0, 14.0, 22.0, 35.5])
    np.testing.assert_allclose(
        model.cif_tvc(t, season), model.cif_tvc(t, spelt), rtol=1e-12
    )
    np.testing.assert_allclose(
        model.iif_tvc(t, season), model.iif_tvc(t, spelt), rtol=1e-12
    )
    with pytest.raises(ValueError, match="infinite time"):
        model.cif_tvc([np.inf], season)


@pytest.mark.parametrize("model", MODELS)
def test_path_bounds_behave(model):
    plan = StepSchedule.from_changepoints([0, 10, 25], [0.0, 1.0, 0.0])
    t = np.array([2.0, 10.0, 18.0, 40.0])
    for on, point in (("cif", model.cif_tvc), ("iif", model.iif_tvc)):
        estimate = point(t, plan)
        two = model.cb_tvc(t, plan, on=on)
        assert np.all(two[:, 0] < estimate) and np.all(estimate < two[:, 1])
        np.testing.assert_allclose(
            model.cb_tvc(t, plan, on=on, bound="lower", alpha_ci=0.025),
            two[:, 0],
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            model.cb_tvc(t, plan, on=on, bound="upper", alpha_ci=0.025),
            two[:, 1],
            rtol=1e-12,
        )
        wide = model.cb_tvc(t, plan, on=on, alpha_ci=0.01)
        assert np.all(wide[:, 0] < two[:, 0]) and np.all(
            wide[:, 1] > two[:, 1]
        )
    with pytest.raises(ValueError, match="alpha_ci"):
        model.cb_tvc(t, plan, alpha_ci=1.5)
    with pytest.raises(ValueError, match="on"):
        model.cb_tvc(t, plan, on="mcf")
    with pytest.raises(ValueError, match="bound"):
        model.cb_tvc(t, plan, bound="both")


@pytest.mark.parametrize("model", MODELS[:2] + MODELS[4:6])
def test_shapes_and_missing_times(model):
    plan = StepSchedule.from_changepoints([0, 10], [0.0, 1.0])
    for method in (model.cif_tvc, model.iif_tvc, model.cb_tvc):
        two_sided = (2,) if method == model.cb_tvc else ()
        assert np.shape(method(5.0, plan)) == two_sided
        assert method([[5.0, 12.0], [20.0, 30.0]], plan).shape == (
            (2, 2) + two_sided
        )
        assert method([], plan).shape == (0,) + two_sided
        out = method([5.0, np.nan, 20.0], plan)
        assert np.isnan(out[1]).all() and np.isfinite(out[[0, 2]]).all()
    assert np.shape(model.cb_tvc(5.0, plan, bound="lower")) == ()
    # Held to infinity on the last value (a power law or a constant rate
    # never stops counting).
    if model.dist.name != "Cox-Lewis":
        assert model.cif_tvc(np.inf, plan) == np.inf


def test_the_schedule_is_checked():
    model = MODELS[0].values[0]
    with pytest.raises(ValueError, match="2 covariate"):
        model.cif_tvc([5.0], StepSchedule.constant([0.0, 1.0]))
    with pytest.raises(ValueError, match="pass xl"):
        model.cif_tvc([5.0], [[0.0], [1.0]])
    with pytest.raises(ValueError, match="xl must not be given"):
        model.cif_tvc([5.0], StepSchedule.constant([0.0]), xl=[0.0])
    ramp = CovariatePath.from_points([0, 10], [0.0, 1.0])
    for method in (model.cif_tvc, model.iif_tvc, model.cb_tvc):
        with pytest.raises(NotImplementedError, match="StepSchedule"):
            method([5.0], ramp)


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(BASELINES))
def test_a_round_trip_keeps_every_prediction(name):
    fitter, kwargs = BASELINES[name]
    d = _mixed_rows()
    df = pd.DataFrame({"load": d["Z"][:, 0], "dust": d["Z"][:, 1]})
    model = fitter.fit_tvc(
        d["i"], d["xl"], d["xr"], d["c"], df, d["n"], **kwargs
    )
    plan = StepSchedule.from_changepoints(
        [0, 9, 20], [[0, 0.2], [1, 0.2], [1, 0.9]]
    )
    t = np.array([3.0, 9.0, 15.0, 28.0])
    z = [1.0, 0.4]

    def predictions(m):
        return [
            m.cif(t, z),
            m.iif(t, z),
            m.inv_cif(2.0, np.array(z)),
            m.cif_tvc(t, plan),
            m.iif_tvc(t, plan),
            m.cif_tvc(t, plan, given=9.0),
        ]

    ref = predictions(model)
    saved = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    restored = surpyval.from_dict(saved)
    for got, want in zip(predictions(restored), ref):
        np.testing.assert_array_equal(got, want)
    assert restored.feature_names == ["load", "dust"]
    assert restored.parameter_names == model.parameter_names
    assert restored.maximum == "verified"
    np.testing.assert_array_equal(restored.params, model.params)
    assert surpyval.from_json(model.to_json()).to_dict() == model.to_dict()
    # A restored model keeps the covariance, so the Wald bounds, but no
    # likelihood: what needs it says so.
    np.testing.assert_array_equal(
        restored.cb_tvc(t, plan), model.cb_tvc(t, plan)
    )
    with pytest.raises(ValueError, match="restored|from_dict"):
        restored.aic()

    # pickle keeps the likelihood, so the inference too.
    clone = pickle.loads(pickle.dumps(model))
    assert clone.is_tvc
    for got, want in zip(predictions(clone), ref):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(clone.cb_tvc(t, plan), model.cb_tvc(t, plan))
    np.testing.assert_array_equal(clone.covariance(), model.covariance())
    assert clone.aic() == model.aic() and clone.bic() == model.bic()
    pd.testing.assert_frame_equal(clone.summary(), model.summary())


# ---------------------------------------------------------------------------
# What a time-varying fit refuses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["HPP", "CrowAMSAA"])
def test_methods_reading_one_covariate_row_per_item_refuse(name):
    fitter, kwargs = BASELINES[name]
    model = _fit(fitter, _mixed_rows(), **kwargs)
    for call in (
        model.residuals,
        lambda: model.residuals(kind="martingale"),
        model.trend_test,
        lambda: model.cramer_von_mises(n_boot=5, random_state=0),
        model.diagnose,
        model.plot,
    ):
        with pytest.raises(ValueError, match="fit_tvc") as error:
            call()
        assert "time-varying covariates" in str(error.value)
        assert "cif_tvc" in str(error.value) or "aic()" in str(error.value)


def test_simulation_is_at_one_covariate_vector():
    model = _fit(ProportionalIntensityNHPP, _mixed_rows(), baseline=CrowAMSAA)
    # At a fixed covariate vector the model simulates as any other.
    data = model.time_terminated_simulation_data(
        30.0, [1.0, 0.5], items=3, random_state=1
    )
    assert len(np.unique(data.i)) == 3
    assert np.isfinite(model.mcf([10.0], [1.0, 0.5], items=20, random_state=1))
    plan = StepSchedule.from_changepoints([0, 9], [[0, 0.2], [1, 0.2]])
    for call in (
        lambda: model.mcf([10.0], plan),
        lambda: model.time_terminated_simulation(30.0, plan),
        lambda: model.time_terminated_simulation_data(30.0, plan),
        lambda: model.count_terminated_simulation(3, plan),
        lambda: model.count_terminated_simulation_data(3, plan),
    ):
        with pytest.raises(ValueError, match="StepSchedule") as error:
            call()
        assert "cif_tvc" in str(error.value)


def _bad(**changes):
    d = {k: list(v) for k, v in HAND.items()}
    d.update(changes)
    return d


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"xl": [0, 5] + HAND["xl"][2:]}, "item 1 has overlapping"),
        ({"xl": [6] + HAND["xl"][1:]}, "xl < xr"),
        ({"xl": [np.nan] + HAND["xl"][1:]}, "xl and xr must be finite"),
        ({"xr": [np.inf] + HAND["xr"][1:]}, "xl and xr must be finite"),
        ({"c": [-1] + HAND["c"][1:]}, "c = -1"),
        ({"c": [3] + HAND["c"][1:]}, "c must be 0"),
        ({"c": [bool(v) for v in HAND["c"]]}, "censoring flag"),
        ({"c": HAND["c"][1:]}, "same length"),
        ({"Z": [np.nan] + HAND["Z"][1:]}, "Z has a missing"),
        ({"Z": HAND["Z"][1:]}, "Z must have one row per interval"),
        ({"Z": ["a"] * 14}, "Z must be numeric"),
        ({"n": [2] + [1] * 13}, "n must be 1 there"),
        ({"c": [2] + HAND["c"][1:], "n": [1.5] + [1] * 13}, "whole number"),
        ({"c": [1] * 14}, "no failures"),
    ],
)
def test_invalid_rows_are_refused(changes, message):
    d = _bad(**changes)
    for fitter, kwargs in (BASELINES["HPP"], BASELINES["CrowAMSAA"]):
        with pytest.raises(ValueError, match=message):
            _fit(fitter, d, **kwargs)


def test_rows_outside_the_baseline_are_refused():
    d = _bad(xl=[-1] + HAND["xl"][1:])
    with pytest.raises(ValueError, match="reaching outside"):
        _fit(ProportionalIntensityNHPP, d, baseline=CrowAMSAA)
    with pytest.raises(ValueError, match="no interval rows"):
        ProportionalIntensityHPP.fit_tvc([], [], [], [], [])


@pytest.mark.parametrize(
    "i, x, c, message",
    [
        ([1, 1, 2], [0, 5, 3], [1, 0, 1], "item 2 has a single timeline row"),
        ([1, 1, 1], [0, 5, 9], [0, 0, 1], "first row is its entry"),
        ([1, 1, 1], [0, 5, 5], [1, 0, 1], "two timeline rows at the same"),
        ([1, 1, 1], [0, np.nan, 5], [1, 0, 1], "must be finite"),
        ([1, 1, 1], [0, 5, 9], [1, -1, 1], "c = -1"),
    ],
)
def test_invalid_timelines_are_refused(i, x, c, message):
    with pytest.raises(ValueError, match=message):
        ProportionalIntensityHPP.fit_tvc_timeline(i, x, np.zeros(len(i)), c)
    with pytest.raises(ValueError, match="Z has a missing"):
        ProportionalIntensityHPP.fit_tvc_timeline(
            [1, 1, 1], [0, 5, 9], [0.0, np.nan, 0.0], [1, 0, 1]
        )


def test_rows_are_sorted_within_an_item():
    rows = handle_recurrent_tvc(
        i=["b", "a", "a"], xl=[0, 4, 0], xr=[5, 9, 4], c=[1, 1, 0], Z=[0, 1, 0]
    )
    assert list(rows["i"]) == ["a", "a", "b"]
    np.testing.assert_array_equal(rows["xl"], [0.0, 4.0, 0.0])
    np.testing.assert_array_equal(rows["n"], 1.0)


def test_no_warning_escapes_a_fit_or_a_prediction():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for fitter, kwargs in BASELINES.values():
            model = _fit(fitter, _mixed_rows(), **kwargs)
            plan = StepSchedule.from_changepoints([0, 9], [[0, 0.2], [1, 0.2]])
            model.cif_tvc([0.0, 5.0, 20.0], plan)
            model.iif_tvc([5.0, 20.0], plan)
            model.cb_tvc([0.0, 5.0, 20.0], plan)
            model.summary()
