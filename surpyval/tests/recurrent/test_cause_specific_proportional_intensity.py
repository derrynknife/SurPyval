"""Cause-specific proportional intensity: several event types (failure
modes) of a repairable item, each with its own baseline and its own
covariate coefficients.

The checks: a closed form in exact arithmetic (constant rates, two types,
one binary covariate: each rate is events over exposure), recovery of a
known truth from simulated data, and the identities the model must keep
(each type is the single-type regression of its own events; without
covariate effects it is ``CauseSpecificNHPP``; the types sum to the
totals and their variances add; a one-type model's totals are that
type's; constant covariates through the time-varying fit are the fixed
fit; saving and loading keeps every prediction).
"""

import json
import math
import pickle
import warnings
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import surpyval
from surpyval import StepSchedule
from surpyval.recurrent import (
    HPP,
    CauseSpecificNHPP,
    CauseSpecificProportionalIntensity,
    CoxLewis,
    CrowAMSAA,
    Duane,
    ProportionalIntensityHPP,
    ProportionalIntensityNHPP,
)

CSPI = CauseSpecificProportionalIntensity

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Five items watched for different lengths of time, two at covariate 0 and
# three at 1, failing in two ways. Whole numbers, so the closed form below
# is exact.
HAND = {
    "x": [4, 9, 15, 20, 7, 22, 30, 2, 5, 11, 16, 3, 8, 13, 19, 24, 6, 10],
    "i": [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5],
    "c": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    "e": ["a", "b", "a", None, "b", "a", None, "a", "a", "b", None]
    + ["a", "b", "a", "a", None, "a", None],
    "Z": [0] * 7 + [1] * 11,
}

# Two items whose binary covariate switches within them (start-stop rows).
HAND_TVC = {
    "i": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3],
    "xl": [0, 6, 10, 13, 17, 0, 3, 7, 10, 16, 21, 2, 9, 12],
    "xr": [6, 10, 13, 17, 20, 3, 7, 10, 16, 21, 25, 9, 12, 19],
    "c": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    "e": ["a", None, "a", "b", None, "a", "b", None, "a", "b", None]
    + ["b", None, "a"],
    "Z": [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
}


def _closed_form(d=HAND):
    """Events of each type and exposure at each covariate level, exact."""
    events = {(k, z): Fraction(0) for k in "ab" for z in (0, 1)}
    exposure = {0: Fraction(0), 1: Fraction(0)}
    for x, c, e, z in zip(d["x"], d["c"], d["e"], d["Z"]):
        if c == 0:
            events[e, z] += 1
        else:
            exposure[z] += x
    return events, exposure


def _closed_form_tvc(d=HAND_TVC):
    events = {(k, z): Fraction(0) for k in "ab" for z in (0, 1)}
    exposure = {0: Fraction(0), 1: Fraction(0)}
    for xl, xr, c, e, z in zip(d["xl"], d["xr"], d["c"], d["e"], d["Z"]):
        exposure[z] += xr - xl
        if c == 0:
            events[e, z] += 1
    return events, exposure


def _fit_hand(**kwargs):
    d = HAND
    return CSPI.fit(
        d["x"], d["Z"], i=d["i"], c=d["c"], e=d["e"], baseline=HPP, **kwargs
    )


def _fit_hand_tvc(**kwargs):
    d = HAND_TVC
    return CSPI.fit_tvc(
        d["i"], d["xl"], d["xr"], d["c"], d["Z"], e=d["e"], baseline=HPP
    )


def simulate(truth, n_items, end, seed, two_covariates=True):
    """Marked power-law processes: ``truth`` maps each type to ``(alpha,
    beta, coefficients)``. Returns the arrays of ``fit``."""
    rng = np.random.default_rng(seed)
    x, i, c, e, Z = [], [], [], [], []
    for item in range(n_items):
        z = [float(item % 2)]
        if two_covariates:
            z.append(float(rng.uniform()))
        for mode, (alpha, beta, coef) in truth.items():
            mean = (end / alpha) ** beta * math.exp(float(np.dot(coef, z)))
            times = end * rng.uniform(size=rng.poisson(mean)) ** (1 / beta)
            for t in np.sort(times):
                x.append(float(t))
                i.append(item)
                c.append(0)
                e.append(mode)
                Z.append(z)
        x.append(end)
        i.append(item)
        c.append(1)
        e.append(None)
        Z.append(z)
    return {
        "x": np.array(x),
        "i": np.array(i),
        "c": np.array(c),
        "e": np.array(e, dtype=object),
        "Z": np.array(Z),
    }


TRUTH = {
    "bearing": (25.0, 1.5, [0.0, 0.7]),
    "motor": (12.0, 1.0, [-0.5, 0.0]),
    "seal": (30.0, 1.3, [0.9, 0.0]),
}


@pytest.fixture(scope="module")
def simulated():
    return simulate(TRUTH, n_items=80, end=50.0, seed=1138)


@pytest.fixture(scope="module")
def fitted(simulated):
    d = simulated
    return CSPI.fit(
        d["x"], d["Z"], i=d["i"], c=d["c"], e=d["e"], baseline=CrowAMSAA
    )


def _start_stop(d):
    """Fixed-covariate arrays as start-stop rows: one row per gap between
    an item's consecutive rows."""
    out = {k: [] for k in ("i", "xl", "xr", "c", "e", "Z")}
    for item in np.unique(d["i"]):
        rows = np.flatnonzero(d["i"] == item)
        rows = rows[np.lexsort((d["c"][rows], d["x"][rows]))]
        start = 0.0
        for k in rows:
            if d["x"][k] == start:
                continue
            out["i"].append(item)
            out["xl"].append(start)
            out["xr"].append(d["x"][k])
            out["c"].append(d["c"][k])
            out["e"].append(d["e"][k])
            out["Z"].append(d["Z"][k])
            start = d["x"][k]
    return {
        k: np.array(v, dtype=object if k == "e" else None)
        for k, v in out.items()
    }


# ---------------------------------------------------------------------------
# The hand-checked case
# ---------------------------------------------------------------------------


def test_hand_checked_rates_and_coefficients():
    """Constant rates, two types, one binary covariate: each type's rate
    at each level is its events over the exposure there, and its
    coefficient the log of the ratio of the two."""
    events, exposure = _closed_form()
    assert exposure == {0: Fraction(50), 1: Fraction(50)}
    assert events == {
        ("a", 0): 3,
        ("a", 1): 6,
        ("b", 0): 2,
        ("b", 1): 2,
    }
    model = _fit_hand()
    assert model.event_types == ["a", "b"]
    for k in "ab":
        rate0 = events[k, 0] / exposure[0]
        rate1 = events[k, 1] / exposure[1]
        sub = model.models[k]
        np.testing.assert_allclose(sub.params, [float(rate0)], rtol=1e-4)
        np.testing.assert_allclose(
            sub.coeffs, [math.log(rate1 / rate0)], rtol=1e-4, atol=1e-5
        )
        # Predictions at each level are the rates there
        for z, rate in ((0, rate0), (1, rate1)):
            np.testing.assert_allclose(
                model.iif(10.0, [z], k), float(rate), rtol=1e-4
            )
            np.testing.assert_allclose(
                model.cif([10.0, 40.0], [z], k),
                [float(10 * rate), float(40 * rate)],
                rtol=1e-4,
            )
    for z in (0, 1):
        total = sum(events[k, z] for k in "ab") / exposure[z]
        np.testing.assert_allclose(
            model.total_iif(10.0, [z]), float(total), rtol=1e-4
        )
        np.testing.assert_allclose(
            model.total_cif(40.0, [z]), float(40 * total), rtol=1e-4
        )


def test_hand_checked_likelihood_and_criteria():
    events, exposure = _closed_form()
    model = _fit_hand()
    log_lik = 0.0
    for (k, z), count in events.items():
        rate = count / exposure[z]
        log_lik += float(count) * math.log(rate) - float(rate * exposure[z])
    assert model.neg_ll() == pytest.approx(-log_lik, abs=1e-7)
    assert model.log_likelihood == pytest.approx(log_lik, abs=1e-7)
    n = float(sum(events.values()))
    assert n == 13
    # Two parameters per type
    assert model.aic() == pytest.approx(2 * 4 - 2 * log_lik, abs=1e-6)
    assert model.bic() == pytest.approx(
        4 * math.log(n) - 2 * log_lik, abs=1e-6
    )
    assert model.aic_c() == pytest.approx(
        model.aic() + (2 * 16 + 8) / (n - 4 - 1), abs=1e-9
    )
    assert model.maximum == "verified"


def test_hand_checked_table_and_bounds():
    """The saturated Poisson model: the variance of a log rate is one over
    its events, of a log rate ratio the sum of the two; and the total at a
    level has the bounds of one constant rate fitted to every event
    there."""
    events, exposure = _closed_form()
    model = _fit_hand()
    table = model.coefficients()
    q = norm.ppf(0.975)
    for k in "ab":
        n0, n1 = float(events[k, 0]), float(events[k, 1])
        coef = math.log(n1 / n0)
        se = math.sqrt(1 / n0 + 1 / n1)
        row = table.loc[(k, "coef_0")]
        assert row["coef"] == pytest.approx(coef, rel=1e-4, abs=1e-5)
        assert row["exp(coef)"] == pytest.approx(n1 / n0, rel=1e-4)
        assert row["se(coef)"] == pytest.approx(se, rel=1e-3)
        assert row["coef lower 95%"] == pytest.approx(
            coef - q * se, rel=1e-3, abs=1e-3
        )
        assert row["exp(coef) upper 95%"] == pytest.approx(
            math.exp(coef + q * se), rel=2e-3
        )
        assert row["z"] == pytest.approx(coef / se, rel=1e-3, abs=1e-3)
        assert row["p"] == pytest.approx(
            2 * norm.sf(abs(coef / se)), rel=1e-2, abs=1e-4
        )
        for z, count in ((0, n0), (1, n1)):
            rate = count / float(exposure[z])
            expected = 40 * rate * np.exp(np.array([-q, q]) / math.sqrt(count))
            np.testing.assert_allclose(
                model.cif_cb(40.0, [z], k), expected, rtol=2e-3
            )
    for z in (0, 1):
        count = float(events["a", z] + events["b", z])
        rate = count / float(exposure[z])
        spread = np.exp(np.array([-q, q]) / math.sqrt(count))
        np.testing.assert_allclose(
            model.total_cif_cb(40.0, [z]), 40 * rate * spread, rtol=2e-3
        )
        np.testing.assert_allclose(
            model.total_iif_cb(40.0, [z]), rate * spread, rtol=2e-3
        )


def test_hand_checked_time_varying():
    """The same closed form when the covariate switches within items."""
    events, exposure = _closed_form_tvc()
    assert exposure == {0: Fraction(32), 1: Fraction(30)}
    assert events == {
        ("a", 0): 2,
        ("a", 1): 3,
        ("b", 0): 2,
        ("b", 1): 2,
    }
    model = _fit_hand_tvc()
    assert model.is_tvc
    for k in "ab":
        rate0 = events[k, 0] / exposure[0]
        rate1 = events[k, 1] / exposure[1]
        sub = model.models[k]
        assert sub.is_tvc
        np.testing.assert_allclose(sub.params, [float(rate0)], rtol=1e-4)
        np.testing.assert_allclose(
            sub.coeffs, [math.log(rate1 / rate0)], rtol=1e-4, atol=1e-5
        )
        # 12 time units at 0 then 12 at 1
        np.testing.assert_allclose(
            model.cif_tvc([12, 24], [[0], [1]], k, xl=[0, 12]),
            [float(12 * rate0), float(12 * rate0 + 12 * rate1)],
            rtol=1e-4,
        )
    total0 = (events["a", 0] + events["b", 0]) / exposure[0]
    total1 = (events["a", 1] + events["b", 1]) / exposure[1]
    np.testing.assert_allclose(
        model.total_cif_tvc([12, 24], [[0], [1]], xl=[0, 12]),
        [float(12 * total0), float(12 * total0 + 12 * total1)],
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        model.total_iif_tvc([6, 18], [[0], [1]], xl=[0, 12]),
        [float(total0), float(total1)],
        rtol=1e-4,
    )
    n = float(sum(events.values()))
    assert model.bic() == pytest.approx(
        4 * math.log(n) + 2 * model.neg_ll(), abs=1e-9
    )


# ---------------------------------------------------------------------------
# Recovery of a known truth
# ---------------------------------------------------------------------------


def test_recovers_simulated_truth(simulated, fitted):
    """Three types with different coefficients on two covariates, some of
    them zero (80 items to time 50, seed 1138; the seed, the sizes and the
    criterion were fixed before the result was seen). Every parameter is
    within 3.5 standard errors of the truth, on the scale of its interval;
    every effect that is there is found at the 1% level."""
    assert fitted.event_types == ["bearing", "motor", "seal"]
    assert fitted.maximum == "verified"
    summary = fitted.summary()
    table = fitted.coefficients()
    for mode, (alpha, beta, coef) in TRUTH.items():
        for name, true in (("alpha", alpha), ("beta", beta)):
            row = summary.loc[(mode, name)]
            z = math.log(row["estimate"] / true) / (
                row["se"] / row["estimate"]
            )
            assert abs(z) < 3.5, (mode, name, row["estimate"], true, z)
        for k, true in enumerate(coef):
            row = table.loc[(mode, f"coef_{k}")]
            z = (row["coef"] - true) / row["se(coef)"]
            assert abs(z) < 3.5, (mode, k, row["coef"], true, z)
            if true != 0:
                assert row["p"] < 0.01, (mode, k, row["p"])
    # The totals at a covariate row, against the truth
    Z = [1.0, 0.5]
    true_total = sum(
        (40.0 / alpha) ** beta * math.exp(float(np.dot(coef, Z)))
        for alpha, beta, coef in TRUTH.values()
    )
    lower, upper = fitted.total_cif_cb(40.0, Z, alpha_ci=0.001)
    assert lower < true_total < upper


# ---------------------------------------------------------------------------
# Each type is the single-type regression of its own events
# ---------------------------------------------------------------------------


def _alone(d, mode, fitter, **kwargs):
    """The single-type regression fitted to ``mode``'s events, the other
    types' events removed."""
    keep = (d["c"] == 1) | (d["e"] == mode)
    return fitter.fit(
        d["x"][keep], d["Z"][keep], i=d["i"][keep], c=d["c"][keep], **kwargs
    )


@pytest.mark.parametrize("baseline", [HPP, Duane, CrowAMSAA, CoxLewis])
def test_each_type_is_the_single_type_fit(simulated, baseline):
    d = simulated
    model = CSPI.fit(
        d["x"], d["Z"], i=d["i"], c=d["c"], e=d["e"], baseline=baseline
    )
    assert model.baseline is baseline
    for mode in model.event_types:
        if baseline is HPP:
            alone = _alone(d, mode, ProportionalIntensityHPP)
        else:
            alone = _alone(
                d, mode, ProportionalIntensityNHPP, baseline=baseline
            )
        sub = model.models[mode]
        np.testing.assert_allclose(sub.params, alone.params, rtol=1e-8)
        np.testing.assert_allclose(sub.coeffs, alone.coeffs, rtol=1e-8)
        assert sub.neg_ll() == pytest.approx(alone.neg_ll(), rel=1e-12)
        np.testing.assert_allclose(
            model.cif([10.0, 30.0], [1.0, 0.3], mode),
            alone.cif([10.0, 30.0], [1.0, 0.3]),
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            model.cif_cb([10.0, 30.0], [1.0, 0.3], mode),
            alone.cif_cb([10.0, 30.0], [1.0, 0.3]),
            rtol=1e-6,
        )


def test_each_type_is_the_single_type_fit_with_delayed_entry():
    d = simulate(TRUTH, n_items=30, end=40.0, seed=7)
    entry = np.where(d["i"] % 3 == 0, 5.0, 0.0)
    keep = d["x"] > entry
    d = {k: v[keep] for k, v in d.items()}
    entry = entry[keep]
    model = CSPI.fit(
        d["x"],
        d["Z"],
        i=d["i"],
        c=d["c"],
        e=d["e"],
        tl=entry,
        baseline=CrowAMSAA,
    )
    for mode in model.event_types:
        rows = (d["c"] == 1) | (d["e"] == mode)
        alone = ProportionalIntensityNHPP.fit(
            d["x"][rows],
            d["Z"][rows],
            i=d["i"][rows],
            c=d["c"][rows],
            tl=entry[rows],
            baseline=CrowAMSAA,
        )
        np.testing.assert_allclose(
            model.models[mode].params, alone.params, rtol=1e-8
        )
        np.testing.assert_allclose(
            model.models[mode].coeffs, alone.coeffs, rtol=1e-8
        )


def test_time_varying_type_is_the_single_type_fit():
    d = HAND_TVC
    model = CSPI.fit_tvc(
        d["i"], d["xl"], d["xr"], d["c"], d["Z"], e=d["e"], baseline=CrowAMSAA
    )
    for mode in "ab":
        c = [
            0 if (flag == 0 and mark == mode) else 1
            for flag, mark in zip(d["c"], d["e"])
        ]
        alone = ProportionalIntensityNHPP.fit_tvc(
            d["i"], d["xl"], d["xr"], c, d["Z"], baseline=CrowAMSAA
        )
        np.testing.assert_allclose(
            model.models[mode].params, alone.params, rtol=1e-10
        )
        np.testing.assert_allclose(
            model.models[mode].coeffs, alone.coeffs, rtol=1e-10
        )


# ---------------------------------------------------------------------------
# Without covariate effects: CauseSpecificNHPP
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dist", [HPP, CrowAMSAA])
def test_reduces_to_cause_specific_nhpp(simulated, dist):
    """A covariate column of zeros has no effect to estimate (it is
    aliased, with a warning), and the model is then the one without
    covariates."""
    d = simulated
    with pytest.warns(UserWarning, match="aliased"):
        model = CSPI.fit(
            d["x"],
            np.zeros(d["x"].size),
            i=d["i"],
            c=d["c"],
            e=d["e"],
            baseline=dist,
        )
    plain = CauseSpecificNHPP.fit(d["x"], d["i"], d["c"], e=d["e"], dist=dist)
    assert model.event_types == plain.event_types
    x = np.array([10.0, 25.0, 50.0])
    for mode in model.event_types:
        assert np.isnan(model.models[mode].coeffs).all()
        np.testing.assert_allclose(
            model.models[mode].params, plain.models[mode].params, rtol=1e-4
        )
        np.testing.assert_allclose(
            model.cif(x, [0.0], mode), plain.cif(x, mode), rtol=1e-4
        )
        np.testing.assert_allclose(
            model.cif_cb(x, [0.0], mode), plain.cif_cb(x, mode), rtol=1e-3
        )
    np.testing.assert_allclose(
        model.total_cif(x, [0.0]), plain.total_cif(x), rtol=1e-4
    )
    np.testing.assert_allclose(
        model.total_iif(x, [0.0]), plain.total_iif(x), rtol=1e-4
    )
    np.testing.assert_allclose(
        model.total_cif_cb(x, [0.0]), plain.total_cif_cb(x), rtol=1e-3
    )
    np.testing.assert_allclose(
        model.total_iif_cb(x, [0.0]), plain.total_iif_cb(x), rtol=1e-3
    )
    assert model.neg_ll() == pytest.approx(plain.neg_ll(), rel=1e-7)
    # An aliased coefficient is not a parameter
    assert model.aic() == pytest.approx(plain.aic(), rel=1e-7)
    assert model.bic() == pytest.approx(plain.bic(), rel=1e-7)


# ---------------------------------------------------------------------------
# Totals
# ---------------------------------------------------------------------------

PLAN = StepSchedule.from_changepoints([0, 20], [[0.0, 0.2], [1.0, 0.6]])
X = np.array([5.0, 20.0, 35.0, 50.0])


def test_types_sum_to_totals(fitted):
    Z = [1.0, 0.4]
    for total, each in (
        (fitted.total_cif(X, Z), lambda k: fitted.cif(X, Z, k)),
        (fitted.total_iif(X, Z), lambda k: fitted.iif(X, Z, k)),
        (fitted.total_cif_tvc(X, PLAN), lambda k: fitted.cif_tvc(X, PLAN, k)),
        (fitted.total_iif_tvc(X, PLAN), lambda k: fitted.iif_tvc(X, PLAN, k)),
        (
            fitted.total_cif_tvc(X, PLAN, given=10.0),
            lambda k: fitted.cif_tvc(X, PLAN, k, given=10.0),
        ),
    ):
        np.testing.assert_allclose(
            total, sum(each(k) for k in fitted.event_types), rtol=1e-12
        )
    # One covariate row per time
    rows = np.array([[0.0, 0.1], [1.0, 0.2], [0.0, 0.9], [1.0, 0.5]])
    np.testing.assert_allclose(
        fitted.total_cif(X, rows),
        sum(fitted.cif(X, rows, k) for k in fitted.event_types),
        rtol=1e-12,
    )


def _log_scale_variance(estimate, cb):
    """The variance behind log-scale Wald bounds, on the natural scale."""
    q = norm.ppf(0.975)
    return (np.log(cb[..., 1] / estimate) * estimate / q) ** 2


def test_total_variance_is_the_sum_of_the_types(fitted):
    Z = [1.0, 0.4]
    cases = (
        (
            fitted.total_cif(X, Z),
            fitted.total_cif_cb(X, Z),
            lambda k: (fitted.cif(X, Z, k), fitted.cif_cb(X, Z, k)),
        ),
        (
            fitted.total_iif(X, Z),
            fitted.total_iif_cb(X, Z),
            lambda k: (fitted.iif(X, Z, k), fitted.iif_cb(X, Z, k)),
        ),
        (
            fitted.total_cif_tvc(X, PLAN),
            fitted.total_cb_tvc(X, PLAN),
            lambda k: (fitted.cif_tvc(X, PLAN, k), fitted.cb_tvc(X, PLAN, k)),
        ),
        (
            fitted.total_iif_tvc(X, PLAN),
            fitted.total_cb_tvc(X, PLAN, on="iif"),
            lambda k: (
                fitted.iif_tvc(X, PLAN, k),
                fitted.cb_tvc(X, PLAN, k, on="iif"),
            ),
        ),
    )
    for total, total_cb, each in cases:
        assert np.all(total_cb[:, 0] < total) and np.all(
            total < total_cb[:, 1]
        )
        parts = sum(_log_scale_variance(*each(k)) for k in fitted.event_types)
        np.testing.assert_allclose(
            _log_scale_variance(total, total_cb), parts, rtol=1e-8
        )


@pytest.mark.parametrize("baseline", [HPP, Duane, CrowAMSAA, CoxLewis])
@pytest.mark.parametrize("bound", ["two-sided", "lower", "upper"])
def test_one_type_total_bounds_are_the_types(simulated, baseline, bound):
    d = simulated
    only = (d["c"] == 1) | (d["e"] == "seal")
    model = CSPI.fit(
        d["x"][only],
        d["Z"][only],
        i=d["i"][only],
        c=d["c"][only],
        e=d["e"][only],
        baseline=baseline,
    )
    assert model.event_types == ["seal"]
    sub = model.models["seal"]
    Z = [1.0, 0.4]
    kw = {"alpha_ci": 0.1, "bound": bound}
    np.testing.assert_allclose(
        model.total_cif_cb(X, Z, **kw), sub.cif_cb(X, Z, **kw), rtol=1e-10
    )
    np.testing.assert_allclose(
        model.total_cif_cb(X, Z, **kw),
        model.cif_cb(X, Z, "seal", **kw),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        model.total_iif_cb(X, Z, **kw), sub.iif_cb(X, Z, **kw), rtol=1e-10
    )
    for on in ("cif", "iif"):
        np.testing.assert_allclose(
            model.total_cb_tvc(X, PLAN, on=on, **kw),
            sub.cb_tvc(X, PLAN, on=on, **kw),
            rtol=1e-10,
        )
    np.testing.assert_allclose(
        model.total_cb_tvc(X, PLAN, given=10.0, **kw),
        model.cb_tvc(X, PLAN, "seal", given=10.0, **kw),
        rtol=1e-10,
    )
    np.testing.assert_allclose(model.total_cif(X, Z), sub.cif(X, Z))


def test_constant_schedule_totals_are_the_fixed_totals(fitted):
    flat = StepSchedule.constant([1.0, 0.4])
    Z = [1.0, 0.4]
    np.testing.assert_allclose(
        fitted.total_cif_tvc(X, flat), fitted.total_cif(X, Z), rtol=1e-12
    )
    np.testing.assert_allclose(
        fitted.total_iif_tvc(X, flat), fitted.total_iif(X, Z), rtol=1e-12
    )
    np.testing.assert_allclose(
        fitted.total_cb_tvc(X, flat), fitted.total_cif_cb(X, Z), rtol=1e-9
    )
    np.testing.assert_allclose(
        fitted.total_cb_tvc(X, flat, on="iif"),
        fitted.total_iif_cb(X, Z),
        rtol=1e-9,
    )
    for mode in fitted.event_types:
        np.testing.assert_allclose(
            fitted.cb_tvc(X, flat, mode), fitted.cif_cb(X, Z, mode), rtol=1e-9
        )


# ---------------------------------------------------------------------------
# Entry points agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("baseline", [HPP, CrowAMSAA])
def test_constant_covariates_through_the_time_varying_fit(baseline):
    """Start-stop rows whose covariates never change within an item give
    the fixed-covariate fit."""
    d = simulate(TRUTH, n_items=30, end=40.0, seed=11)
    fixed = CSPI.fit(
        d["x"], d["Z"], i=d["i"], c=d["c"], e=d["e"], baseline=baseline
    )
    ss = _start_stop(d)
    tvc = CSPI.fit_tvc(
        ss["i"],
        ss["xl"],
        ss["xr"],
        ss["c"],
        np.array(list(ss["Z"]), dtype=float),
        e=ss["e"],
        baseline=baseline,
    )
    assert tvc.is_tvc and not fixed.is_tvc
    assert tvc.event_types == fixed.event_types
    Z = [1.0, 0.4]
    for mode in fixed.event_types:
        np.testing.assert_allclose(
            tvc.models[mode].params, fixed.models[mode].params, rtol=1e-4
        )
        np.testing.assert_allclose(
            tvc.models[mode].coeffs,
            fixed.models[mode].coeffs,
            rtol=1e-4,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            tvc.cif(X, Z, mode), fixed.cif(X, Z, mode), rtol=1e-4
        )
    np.testing.assert_allclose(
        tvc.total_cif_cb(X, Z), fixed.total_cif_cb(X, Z), rtol=1e-3
    )
    assert tvc.neg_ll() == pytest.approx(fixed.neg_ll(), rel=1e-8)
    assert tvc.bic() == pytest.approx(fixed.bic(), rel=1e-8)


def test_timeline_is_start_stop():
    d = HAND_TVC
    start_stop = _fit_hand_tvc()
    # The same histories as a timeline: an entry row, then each row's end
    i, x, c, e, Z = [], [], [], [], []
    items = np.array(d["i"])
    for item in (1, 2, 3):
        rows = np.flatnonzero(items == item)
        i.append(item)
        x.append(d["xl"][rows[0]])
        c.append(1)
        e.append(None)
        Z.append(d["Z"][rows[0]])
        for pos, k in enumerate(rows):
            i.append(item)
            x.append(d["xr"][k])
            c.append(d["c"][k])
            e.append(d["e"][k])
            last = pos == len(rows) - 1
            Z.append(d["Z"][k] if last else d["Z"][rows[pos + 1]])
    # Shuffled: the timeline is sorted by item and time
    order = np.random.default_rng(3).permutation(len(i))
    timeline = CSPI.fit_tvc_timeline(
        np.array(i)[order],
        np.array(x)[order],
        np.array(Z)[order],
        np.array(c)[order],
        e=np.array(e, dtype=object)[order],
        baseline=HPP,
    )
    for mode in "ab":
        np.testing.assert_allclose(
            timeline.models[mode].params,
            start_stop.models[mode].params,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            timeline.models[mode].coeffs,
            start_stop.models[mode].coeffs,
            rtol=1e-10,
        )
    frame = pd.DataFrame(
        {"unit": i, "t": x, "status": c, "mode": e, "load": Z}
    )
    from_df = CSPI.fit_tvc_timeline_from_df(
        frame, "unit", "t", "load", "status", "mode", baseline=HPP
    )
    assert from_df.feature_names == ["load"]
    assert from_df.models["a"].parameter_names == ["lambda", "load"]
    np.testing.assert_allclose(
        from_df.models["a"].coeffs, start_stop.models["a"].coeffs, rtol=1e-10
    )


def test_dataframe_forms_are_the_array_fits(simulated, fitted):
    d = simulated
    frame = pd.DataFrame(
        {
            "t": d["x"],
            "unit": d["i"],
            "status": d["c"],
            "mode": d["e"],
            "harsh": d["Z"][:, 0],
            "duty": d["Z"][:, 1],
        }
    )
    from_df = CSPI.fit_from_df(
        frame,
        x_col="t",
        Z_cols=["harsh", "duty"],
        e_col="mode",
        i_col="unit",
        c_col="status",
        baseline=CrowAMSAA,
    )
    assert from_df.feature_names == ["harsh", "duty"]
    assert list(from_df.coefficients().index) == [
        (mode, name)
        for mode in ("bearing", "motor", "seal")
        for name in ("harsh", "duty")
    ]
    np.testing.assert_allclose(
        from_df.coefficients().to_numpy(),
        fitted.coefficients().to_numpy(),
        rtol=1e-10,
    )
    # Predictions read a DataFrame, a dict or an array of covariates
    new = pd.DataFrame({"duty": [0.4], "harsh": [1.0]})
    np.testing.assert_allclose(
        from_df.total_cif(30.0, new), fitted.total_cif(30.0, [1.0, 0.4])
    )
    np.testing.assert_allclose(
        from_df.total_cif_cb(30.0, {"harsh": 1.0, "duty": 0.4}),
        fitted.total_cif_cb(30.0, [1.0, 0.4]),
    )
    # A DataFrame Z in fit names the coefficients as well
    named = CSPI.fit(
        d["x"],
        frame[["harsh", "duty"]],
        i=d["i"],
        c=d["c"],
        e=d["e"],
        baseline=CrowAMSAA,
    )
    assert named.feature_names == ["harsh", "duty"]
    assert named.models["seal"].parameter_names == [
        "alpha",
        "beta",
        "harsh",
        "duty",
    ]
    # Start-stop rows from a DataFrame
    ss = HAND_TVC
    rows = pd.DataFrame(
        {
            "unit": ss["i"],
            "start": ss["xl"],
            "stop": ss["xr"],
            "status": ss["c"],
            "mode": ss["e"],
            "load": ss["Z"],
        }
    )
    tvc_df = CSPI.fit_tvc_from_df(
        rows, "unit", "start", "stop", "status", "load", "mode", baseline=HPP
    )
    assert tvc_df.feature_names == ["load"]
    np.testing.assert_allclose(
        tvc_df.coefficients()["coef"].to_numpy(),
        _fit_hand_tvc().coefficients()["coef"].to_numpy(),
        rtol=1e-10,
    )
    with pytest.raises(TypeError, match="e_col"):
        CSPI.fit_from_df(frame, x_col="t", Z_cols="harsh", e="mode")


def test_event_history_data_and_row_order(simulated, fitted):
    d = simulated
    data = surpyval.handle_xicn(d["x"], d["i"], d["c"], e=d["e"], Z=d["Z"])
    from_data = CSPI.fit(data, baseline=CrowAMSAA)
    order = np.random.default_rng(5).permutation(d["x"].size)
    shuffled = CSPI.fit(
        d["x"][order],
        d["Z"][order],
        i=d["i"][order],
        c=d["c"][order],
        e=d["e"][order],
        baseline=CrowAMSAA,
    )
    for other in (from_data, shuffled):
        np.testing.assert_allclose(
            other.summary().to_numpy(), fitted.summary().to_numpy(), rtol=1e-6
        )


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def test_summary_stacks_the_types(fitted):
    summary = fitted.summary(alpha_ci=0.1)
    assert list(summary.columns) == [
        "estimate",
        "se",
        "lower 90%",
        "upper 90%",
    ]
    assert summary.index.names == ["event", "parameter"]
    for mode in fitted.event_types:
        own = fitted.models[mode].summary(alpha_ci=0.1)
        assert list(summary.loc[mode].index) == list(own.index)
        np.testing.assert_allclose(
            summary.loc[mode].to_numpy(), own.to_numpy(), rtol=1e-12
        )


def test_coefficient_table(fitted):
    table = fitted.coefficients(alpha_ci=0.1)
    assert table.index.names == ["event", "covariate"]
    assert list(table.columns) == [
        "coef",
        "exp(coef)",
        "se(coef)",
        "coef lower 90%",
        "coef upper 90%",
        "exp(coef) lower 90%",
        "exp(coef) upper 90%",
        "z",
        "p",
    ]
    assert len(table) == 6
    for mode in fitted.event_types:
        sub = fitted.models[mode]
        np.testing.assert_allclose(
            table.loc[mode]["coef"].to_numpy(), sub.coeffs, rtol=1e-12
        )
        np.testing.assert_allclose(
            table.loc[mode]["se(coef)"].to_numpy(),
            sub.standard_errors()[2:],
            rtol=1e-12,
        )
        for k in range(2):
            lower, upper = sub.param_cb(f"coef_{k}", alpha_ci=0.1)
            row = table.loc[(mode, f"coef_{k}")]
            assert row["coef lower 90%"] == pytest.approx(lower, rel=1e-10)
            assert row["coef upper 90%"] == pytest.approx(upper, rel=1e-10)
            assert row["exp(coef) lower 90%"] == pytest.approx(
                math.exp(lower), rel=1e-10
            )
            assert row["exp(coef)"] == pytest.approx(
                math.exp(sub.coeffs[k]), rel=1e-12
            )


def test_repr_and_labels():
    model = _fit_hand()
    assert repr(model) == (
        "Cause-specific proportional intensity (Constant baseline) with "
        "causes: ['a', 'b']"
    )
    # Labels of mixed types, a tuple among them
    d = HAND
    marks = np.empty(len(d["e"]), dtype=object)
    for k, mark in enumerate(d["e"]):
        marks[k] = {"a": ("pump", 1), "b": 2, None: None}[mark]
    mixed = CSPI.fit(d["x"], d["Z"], i=d["i"], c=d["c"], e=marks, baseline=HPP)
    assert set(mixed.event_types) == {("pump", 1), 2}
    np.testing.assert_allclose(
        mixed.cif(10.0, [1], ("pump", 1)), model.cif(10.0, [1], "a")
    )
    assert list(mixed.summary().index.get_level_values("event").unique()) == (
        mixed.event_types
    )
    restored = CSPI.from_dict(json.loads(json.dumps(mixed.to_dict())))
    assert restored.event_types == mixed.event_types
    np.testing.assert_allclose(
        restored.cif_cb(10.0, [1], ("pump", 1)),
        mixed.cif_cb(10.0, [1], ("pump", 1)),
    )


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def test_shapes(fitted):
    Z = [1.0, 0.4]
    assert np.ndim(fitted.cif(10.0, Z, "seal")) == 0
    assert np.ndim(fitted.total_cif(10.0, Z)) == 0
    assert np.ndim(fitted.total_iif(10.0, Z)) == 0
    assert np.ndim(fitted.total_cif_tvc(10.0, PLAN)) == 0
    assert fitted.total_cif_cb(10.0, Z).shape == (2,)
    assert fitted.total_cb_tvc(10.0, PLAN).shape == (2,)
    assert np.ndim(fitted.total_cif_cb(10.0, Z, bound="lower")) == 0
    grid = X.reshape(2, 2)
    assert fitted.total_cif(grid, Z).shape == (2, 2)
    assert fitted.total_cif_cb(grid, Z).shape == (2, 2, 2)
    assert fitted.total_cb_tvc(grid, PLAN, on="iif").shape == (2, 2, 2)
    assert fitted.total_cif(np.array([]), Z).shape == (0,)
    assert fitted.total_cif_cb(np.array([]), Z).shape == (0, 2)
    assert np.isnan(fitted.total_cif([np.nan, 10.0], Z)[0])


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------


def _everything(model):
    Z = [1.0, 0.4]
    out = {}
    for mode in model.event_types:
        out[f"cif {mode}"] = model.cif(X, Z, mode)
        out[f"iif {mode}"] = model.iif(X, Z, mode)
        out[f"cif_cb {mode}"] = model.cif_cb(X, Z, mode)
        out[f"iif_cb {mode}"] = model.iif_cb(X, Z, mode, bound="upper")
        out[f"cif_tvc {mode}"] = model.cif_tvc(X, PLAN, mode)
        out[f"cif_tvc given {mode}"] = model.cif_tvc(X, PLAN, mode, given=8.0)
        out[f"iif_tvc {mode}"] = model.iif_tvc(X, PLAN, mode)
        out[f"cb_tvc {mode}"] = model.cb_tvc(X, PLAN, mode)
        out[f"cb_tvc iif {mode}"] = model.cb_tvc(
            X, PLAN, mode, on="iif", alpha_ci=0.2, bound="lower"
        )
        out[f"cb_tvc given {mode}"] = model.cb_tvc(X, PLAN, mode, given=8.0)
    out["total_cif"] = model.total_cif(X, Z)
    out["total_iif"] = model.total_iif(X, Z)
    out["total_cif_cb"] = model.total_cif_cb(X, Z)
    out["total_iif_cb"] = model.total_iif_cb(X, Z, alpha_ci=0.1)
    out["total_cif_tvc"] = model.total_cif_tvc(X, PLAN)
    out["total_iif_tvc"] = model.total_iif_tvc(X, PLAN)
    out["total_cb_tvc"] = model.total_cb_tvc(X, PLAN)
    out["total_cb_tvc iif"] = model.total_cb_tvc(X, PLAN, on="iif")
    out["total_cb_tvc given"] = model.total_cb_tvc(X, PLAN, given=8.0)
    out["summary"] = model.summary().to_numpy()
    out["coefficients"] = model.coefficients().to_numpy()
    return out


@pytest.mark.parametrize(
    "restore",
    [
        lambda m: CSPI.from_dict(m.to_dict()),
        lambda m: CSPI.from_dict(json.loads(json.dumps(m.to_dict()))),
        lambda m: CSPI.from_json(m.to_json()),
        lambda m: surpyval.from_dict(m.to_dict()),
        lambda m: pickle.loads(pickle.dumps(m)),
        # A restored model saves what it was given
        lambda m: CSPI.from_dict(CSPI.from_dict(m.to_dict()).to_dict()),
    ],
    ids=["dict", "json dict", "json", "surpyval", "pickle", "twice"],
)
def test_round_trip_keeps_every_prediction(fitted, restore):
    restored = restore(fitted)
    assert isinstance(restored, CSPI)
    assert restored.event_types == fitted.event_types
    assert restored.baseline is fitted.baseline
    assert repr(restored) == repr(fitted)
    before, after = _everything(fitted), _everything(restored)
    assert before.keys() == after.keys()
    for name, value in before.items():
        np.testing.assert_allclose(after[name], value, rtol=1e-9, err_msg=name)


def test_round_trip_of_a_time_varying_and_a_constant_rate_fit():
    for model in (_fit_hand_tvc(), _fit_hand()):
        restored = CSPI.from_json(model.to_json())
        assert restored.baseline is HPP
        for mode in "ab":
            np.testing.assert_allclose(
                restored.cb_tvc([12, 24], [[0], [1]], mode, xl=[0, 12]),
                model.cb_tvc([12, 24], [[0], [1]], mode, xl=[0, 12]),
                rtol=1e-9,
            )
        np.testing.assert_allclose(
            restored.total_cb_tvc([12, 24], [[0], [1]], xl=[0, 12]),
            model.total_cb_tvc([12, 24], [[0], [1]], xl=[0, 12]),
            rtol=1e-9,
        )
        np.testing.assert_allclose(
            restored.coefficients().to_numpy(),
            model.coefficients().to_numpy(),
            rtol=1e-9,
        )


def test_restored_model_keeps_names_and_refuses_the_likelihood(fitted):
    frame = pd.DataFrame(
        {
            "t": HAND["x"],
            "unit": HAND["i"],
            "status": HAND["c"],
            "mode": HAND["e"],
            "harsh": HAND["Z"],
        }
    )
    model = CSPI.fit_from_df(
        frame, "t", "harsh", "mode", i_col="unit", c_col="status", baseline=HPP
    )
    restored = CSPI.from_json(model.to_json())
    assert restored.feature_names == ["harsh"]
    assert list(restored.coefficients().index) == [
        ("a", "harsh"),
        ("b", "harsh"),
    ]
    np.testing.assert_allclose(
        restored.total_cif(10.0, pd.DataFrame({"harsh": [1.0]})),
        model.total_cif(10.0, [1.0]),
    )
    for method in ("neg_ll", "aic", "bic", "aic_c"):
        with pytest.raises(ValueError, match="restored"):
            getattr(restored, method)()
    # A dict without the covariances predicts, and refuses its bounds
    old = model.to_dict()
    del old["covariances"]
    bare = CSPI.from_dict(old)
    np.testing.assert_allclose(
        bare.total_cif(10.0, [1.0]), model.total_cif(10.0, [1.0])
    )
    for call in (
        lambda: bare.total_cif_cb(10.0, [1.0]),
        lambda: bare.cif_cb(10.0, [1.0], "a"),
        lambda: bare.total_cb_tvc(10.0, [[1.0]], xl=[0]),
        lambda: bare.summary(),
        lambda: bare.coefficients(),
    ):
        with pytest.raises(ValueError, match="restored"):
            call()


def test_information_criteria_are_sums(fitted):
    subs = [fitted.models[k] for k in fitted.event_types]
    assert fitted.neg_ll() == pytest.approx(
        sum(m.neg_ll() for m in subs), rel=1e-14
    )
    assert fitted.aic() == pytest.approx(sum(m.aic() for m in subs), rel=1e-12)
    n = sum(m._n_obs for m in subs)
    assert fitted.bic() == pytest.approx(
        12 * math.log(n) + 2 * fitted.neg_ll(), rel=1e-12
    )


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

NOT_AVAILABLE = [
    "diagnose",
    "residuals",
    "cramer_von_mises",
    "trend_test",
    "plot",
    "count_terminated_simulation",
    "count_terminated_simulation_data",
    "time_terminated_simulation",
    "time_terminated_simulation_data",
    "mcf",
    "forecast",
    "growth",
    "mtbf",
    "mtbf_cb",
    "total_mtbf",
    "total_mtbf_cb",
]


@pytest.mark.parametrize("name", NOT_AVAILABLE)
def test_what_is_not_carried_over_refuses_saying_what_to_use(name):
    model = _fit_hand()
    with pytest.raises(NotImplementedError) as error:
        getattr(model, name)()
    message = str(error.value)
    assert message.startswith(f"{name}() is not available")
    # What to use instead: a type's own model, or a method of this one
    assert (
        "model.models[event]" in message
        or "iif" in message
        or ("cif" in message)
    )
    assert getattr(CSPI, name).__doc__


def test_the_alternatives_the_refusals_name_work():
    model = _fit_hand()
    sub = model.models["a"]
    assert sub.residuals().size > 0
    assert sub.trend_test().p_value > 0
    data = sub.time_terminated_simulation_data(
        20.0, [1.0], items=3, random_state=1
    )
    assert data.x.size >= 3
    one = surpyval.forecast(sub, age=10.0, horizon=5.0, Z=[1.0])
    np.testing.assert_allclose(
        one.expected, model.cif_tvc(15.0, [[1.0]], "a", xl=[10.0], given=10.0)
    )
    np.testing.assert_allclose(
        model.total_cif_tvc(15.0, [[1.0]], xl=[10.0], given=10.0),
        model.total_cif(15.0, [1.0]) - model.total_cif(10.0, [1.0]),
    )


def test_surpyval_forecast_refuses():
    with pytest.raises(NotImplementedError, match="total_cif_tvc"):
        surpyval.forecast(_fit_hand(), age=10.0, horizon=5.0, Z=[1.0])


def test_fit_refusals():
    d = HAND
    fit = lambda **kw: CSPI.fit(  # noqa: E731
        **{
            "x": d["x"],
            "Z": d["Z"],
            "i": d["i"],
            "c": d["c"],
            "e": d["e"],
            "baseline": HPP,
            **kw,
        }
    )
    with pytest.raises(ValueError, match="pass `e`"):
        fit(e=None)
    with pytest.raises(ValueError, match="baseline must be"):
        fit(baseline="HPP")
    with pytest.raises(ValueError, match="right truncation"):
        fit(tr=40.0)
    with pytest.raises(ValueError):
        fit(n=np.full(len(d["x"]), 2))
    counted = np.array(d["c"])
    counted[0] = 2
    with pytest.raises(ValueError):
        fit(c=counted)
    data = surpyval.handle_xicn(d["x"], d["i"], d["c"], e=d["e"])
    # (the data wrapper's own refusal, a TypeError as on every fitter)
    with pytest.raises((TypeError, ValueError), match="covariates"):
        CSPI.fit_from_recurrent_data(data, baseline=HPP)
    data = surpyval.handle_xicn(d["x"], d["i"], d["c"], Z=d["Z"])
    with pytest.raises(ValueError, match="event-type marks"):
        CSPI.fit_from_recurrent_data(data, baseline=HPP)
    # Covariates that change within an item belong to fit_tvc
    changing = np.array(d["Z"], dtype=float)
    changing[1] = 1.0
    with pytest.raises(ValueError):
        fit(Z=changing)
    model = fit()
    with pytest.raises(ValueError, match="Unknown cause 'c'"):
        model.cif(10.0, [1], "c")
    with pytest.raises(ValueError, match="Unknown cause"):
        model.cb_tvc(10.0, [[1]], "seal", xl=[0])
    with pytest.raises(ValueError, match="alpha_ci"):
        model.total_cif_cb(10.0, [1], alpha_ci=1.5)
    with pytest.raises(ValueError, match="bound"):
        model.total_cif_cb(10.0, [1], bound="both")
    with pytest.raises(ValueError, match="given applies to on='cif'"):
        model.total_cb_tvc(10.0, [[1]], xl=[0], given=2.0, on="iif")
    with pytest.raises(ValueError, match="on"):
        model.total_cb_tvc(10.0, [[1]], xl=[0], on="mcf")
    with pytest.raises(ValueError, match="2 covariate"):
        model.total_cif_tvc(10.0, [[1, 0]], xl=[0])


def test_time_varying_fit_refusals():
    d = HAND_TVC
    fit = lambda **kw: CSPI.fit_tvc(  # noqa: E731
        **{
            "i": d["i"],
            "xl": d["xl"],
            "xr": d["xr"],
            "c": d["c"],
            "Z": d["Z"],
            "e": d["e"],
            "baseline": HPP,
            **kw,
        }
    )
    counted = list(d["c"])
    counted[1] = 2
    with pytest.raises(ValueError, match="c = 2") as error:
        fit(c=counted)
    assert "ProportionalIntensityNHPP.fit_tvc" in str(error.value)
    with pytest.raises(ValueError, match="e .the event type"):
        fit(e=None)
    with pytest.raises(ValueError, match="same length"):
        fit(e=d["e"][:-1])
    with pytest.raises(ValueError, match="baseline must be"):
        fit(baseline=ProportionalIntensityHPP)
    with pytest.raises(ValueError):
        fit(n=[2] * len(d["i"]))
    with pytest.raises(ValueError, match="no failures"):
        fit(c=[1] * len(d["i"]))
    # A failure without a type warns, with the count, and counts for none
    unmarked = list(d["e"])
    unmarked[0] = None
    with pytest.warns(UserWarning, match="1 row.s. that end in a failure"):
        model = fit(e=unmarked)
    events, exposure = _closed_form_tvc()
    np.testing.assert_allclose(
        model.models["a"].params,
        [float((events["a", 0] - 1) / exposure[0])],
        rtol=1e-4,
    )
    # NaN marks (a DataFrame column) are missing marks
    nan_marks = [np.nan if mark is None else mark for mark in d["e"]]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        same = fit(e=nan_marks)
    assert same.event_types == ["a", "b"]
