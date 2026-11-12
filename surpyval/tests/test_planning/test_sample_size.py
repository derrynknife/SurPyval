"""Sample size and power beyond the plain log-rank calculation: a Cox
model's coefficient, non-inferiority and equivalence of a hazard ratio,
a difference in RMST, and a log-rank test with cured units. Closed forms,
consistency of each design's functions with each other, and the
refusals. The R references are in
``surpyval/tests/reference/test_sample_size_references.py``; the
simulated power in ``calibration/test_power_sample_size_designs.py``."""

from typing import Any

import numpy as np
import pytest
from scipy import integrate, stats

import surpyval as sp
from surpyval.test_planning import (
    cox_detectable,
    cox_events,
    cox_power,
    cox_sample_size,
    cure_logrank_events,
    cure_logrank_power,
    cure_logrank_sample_size,
    equivalence_events,
    equivalence_power,
    equivalence_sample_size,
    logrank_events,
    noninferiority_events,
    noninferiority_power,
    noninferiority_sample_size,
    rmst_power,
    rmst_sample_size,
)
from surpyval.test_planning.sample_size import _rmst_moments

Z = stats.norm.ppf

# Wrong types, passed on purpose to the checks below.
THREE: Any = (0.1, 0.2, 0.3)
NOT_A_CURVE: Any = "weibull"


# --- Cox ---------------------------------------------------------------------


@pytest.mark.parametrize("proportion", [0.2, 0.5, 0.7])
@pytest.mark.parametrize("hazard_ratio", [0.5, 1.5, 2.0])
def test_cox_two_levels_is_schoenfeld(proportion, hazard_ratio):
    ratio = proportion / (1 - proportion)
    assert cox_events(hazard_ratio, proportion=proportion) == logrank_events(
        hazard_ratio, ratio=ratio
    )


def test_cox_events_closed_form():
    d = (Z(0.975) + Z(0.9)) ** 2 / (2.0**2 * 0.6 * np.log(1.3) ** 2)
    assert cox_events(1.3, 0.9, sd=2.0, r_squared=0.4) == int(np.ceil(d))


def test_r_squared_inflates_by_its_complement():
    plain = (Z(0.975) + Z(0.8)) ** 2 / np.log(1.2) ** 2
    for r2 in (0.25, 0.5, 0.75):
        adjusted = cox_events(1.2, sd=1.0, r_squared=r2)
        assert adjusted == int(np.ceil(plain / (1 - r2)))


def test_cox_three_ways_agree():
    kw = dict(sd=0.8, r_squared=0.2)
    n = cox_sample_size(1.4, 0.35, 0.85, **kw)
    assert cox_power(1.4, n, 0.35, **kw) >= 0.85
    assert cox_power(1.4, n - 1, 0.35, **kw) < 0.85
    hr = cox_detectable(n, 0.35, 0.85, **kw)
    assert 1.0 < hr <= 1.4
    assert cox_power(hr, n, 0.35, **kw) == pytest.approx(0.85)
    lower = cox_detectable(n, 0.35, 0.85, direction="decrease", **kw)
    assert lower == pytest.approx(1 / hr)


def test_cox_broadcasts():
    out = cox_events(np.array([1.5, 2.0]), sd=1.0)
    assert out.tolist() == [cox_events(1.5, sd=1.0), cox_events(2.0, sd=1.0)]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({}, "Give one of sd"),
        ({"sd": 1.0, "proportion": 0.5}, "Give one of sd"),
        ({"sd": -1.0}, "sd must be positive"),
        ({"proportion": 1.0}, "proportion must be between"),
        ({"sd": 1.0, "r_squared": 1.0}, "r_squared"),
        ({"sd": 1.0, "r_squared": -0.1}, "r_squared"),
    ],
)
def test_cox_refusals(kwargs, match):
    with pytest.raises(ValueError, match=match):
        cox_events(1.5, **kwargs)


def test_cox_refuses_no_effect():
    with pytest.raises(ValueError, match="differ from 1"):
        cox_events(1.0, sd=1.0)


# --- Non-inferiority ---------------------------------------------------------


@pytest.mark.parametrize("margin", [1.1, 1.3, 1.5])
@pytest.mark.parametrize("ratio", [1.0, 2.0])
def test_noninferiority_at_one_is_schoenfeld_at_the_margin(margin, ratio):
    # H0: HR = margin against a truth of 1 is Schoenfeld's comparison of
    # the margin with 1, one-sided at alpha / 2.
    assert noninferiority_events(margin, ratio=ratio) == logrank_events(
        margin, ratio=ratio
    )


def test_noninferiority_closed_form():
    d = (Z(0.975) + Z(0.9)) ** 2 / (0.25 * (np.log(1.25) - np.log(0.95)) ** 2)
    got = noninferiority_events(1.25, 0.9, hazard_ratio=0.95)
    assert got == int(np.ceil(d))


def test_noninferiority_three_ways_agree():
    n = noninferiority_sample_size(1.3, (0.3, 0.4), 0.9, ratio=2.0)
    assert noninferiority_power(1.3, n, (0.3, 0.4)) >= 0.9
    assert n[0] == pytest.approx(2 * n[1], abs=1)


def test_noninferiority_refuses_a_truth_at_the_margin():
    with pytest.raises(ValueError, match="must be below margin"):
        noninferiority_events(1.2, hazard_ratio=1.2)


# --- Equivalence -------------------------------------------------------------


@pytest.mark.parametrize("margin", [1.2, 1.25, 1.5])
@pytest.mark.parametrize("power", [0.8, 0.9])
def test_equivalence_symmetric_closed_form(margin, power):
    # Chow, Shao and Wang: at a true ratio of 1 the two one-sided tests
    # need (z_{1 - a/2} + z_{1 - b/2})^2 / (pi (1 - pi) ln(m)^2) events,
    # to within the second tail (below 1e-6 of a unit here).
    beta = 1 - power
    d = (Z(0.975) + Z(1 - beta / 2)) ** 2 / (0.25 * np.log(margin) ** 2)
    assert equivalence_events(margin, power) == int(np.ceil(d - 1e-6))


def test_equivalence_pair_and_off_centre_truth():
    symmetric = equivalence_events(1.25)
    assert equivalence_events((0.8, 1.25)) == symmetric
    assert equivalence_events(1.25, hazard_ratio=1.05) > symmetric


def test_equivalence_three_ways_agree():
    n = equivalence_sample_size((0.85, 1.2), 0.6, hazard_ratio=1.02)
    power = equivalence_power((0.85, 1.2), n, 0.6, hazard_ratio=1.02)
    assert 0.8 <= power < 0.81


@pytest.mark.parametrize(
    "margin, hazard_ratio, match",
    [
        (0.9, 1.0, "margin must be above 1"),
        (THREE, 1.0, "or a pair"),
        (1.25, 1.3, "strictly between the margins"),
        ((0.8, 1.25), 0.8, "strictly between the margins"),
    ],
)
def test_equivalence_refusals(margin, hazard_ratio, match):
    with pytest.raises(ValueError, match=match):
        equivalence_events(margin, hazard_ratio=hazard_ratio)


# --- RMST --------------------------------------------------------------------


@pytest.mark.parametrize("rate, tau", [(0.1, 5.0), (0.5, 3.0), (1.0, 1.0)])
def test_rmst_variance_uncensored_is_that_of_the_truncated_life(rate, tau):
    # With no censoring before tau the Kaplan-Meier RMST is the mean of
    # min(T, tau): its variance is that of min(T, tau).
    sf = sp.Exponential.from_params([rate]).sf
    mean, variance = _rmst_moments(sf, tau, 0.0, tau, 0.0, "s")
    first = (1 - np.exp(-rate * tau)) / rate
    second = 2 * (
        1 / rate**2 - np.exp(-rate * tau) * (tau / rate + 1 / rate**2)
    )
    assert mean == pytest.approx(first, rel=1e-8)
    assert variance == pytest.approx(second - first**2, rel=1e-6)


def test_rmst_variance_censored_quadrature():
    # Weibull, uniform entry over 2, follow-up 1, dropout 0.1: the grid
    # against adaptive quadrature of the same integral.
    alpha, beta, tau, a, f, d = 4.0, 1.5, 2.5, 2.0, 1.0, 0.1
    w = sp.Weibull.from_params([alpha, beta])

    def g(t):
        return min(1.0, (a + f - t) / a) * np.exp(-d * t)

    def tail(t):
        return integrate.quad(lambda u: w.sf(u), t, tau)[0]

    def integrand(t):
        return tail(t) ** 2 * w.hf(t) / (w.sf(t) * g(t))

    expected = integrate.quad(integrand, 0, tau, points=[f], limit=200)[0]
    _, variance = _rmst_moments(w.sf, tau, a, f, d, "s")
    assert variance == pytest.approx(expected, rel=1e-6)


def test_rmst_sample_size_and_power_agree():
    curves = (sp.Weibull.from_params([10.0, 1.2]), lambda t: np.exp(-t / 7))
    kw = dict(accrual_time=1.0, follow_up=3.0, ratio=1.5)
    n = rmst_sample_size(curves, 4.0, 0.9, **kw)
    kw.pop("ratio")
    assert rmst_power(curves, 4.0, n, **kw) >= 0.9
    assert rmst_power(curves, 4.0, (n[0] - 2, n[1] - 1), **kw) < 0.9


def test_rmst_follow_up_defaults_to_tau():
    curves = (sp.Exponential.from_params([0.1]), lambda t: np.exp(-t / 6))
    assert rmst_sample_size(curves, 5.0) == rmst_sample_size(
        curves, 5.0, follow_up=5.0
    )


@pytest.mark.parametrize(
    "survival, kwargs, match",
    [
        (NOT_A_CURVE, {}, "must be a pair"),
        ((NOT_A_CURVE, NOT_A_CURVE), {}, "sf method"),
        (
            (lambda t: np.exp(-t), lambda t: np.exp(-t)),
            {},
            "same RMST",
        ),
        (
            (lambda t: np.exp(-t), lambda t: np.exp(-2 * t)),
            {"follow_up": 1.0},
            "beyond the end of the study",
        ),
        (
            (lambda t: 1 + 0 * t, lambda t: np.exp(t)),
            {},
            "between 0 and 1",
        ),
        (
            (lambda t: np.exp(-t), lambda t: np.exp(-t)),
            {"dropout_rate": -1.0},
            "dropout_rate",
        ),
    ],
)
def test_rmst_refusals(survival, kwargs, match):
    with pytest.raises(ValueError, match=match):
        rmst_sample_size(survival, 2.0, **kwargs)


# --- Cure --------------------------------------------------------------------


@pytest.mark.parametrize("hazard_ratio", [0.8, 1.25])
def test_cure_with_no_cure_and_ph_is_close_to_schoenfeld(hazard_ratio):
    # No cured units, proportional hazards and everyone followed until
    # nearly all fail: the general formula is Schoenfeld's to within the
    # difference between the variance under the alternative and under the
    # null (a few per cent at these ratios).
    latency = sp.Exponential.from_params([1.0])
    events = cure_logrank_events(
        0.0, latency, latency_hazard_ratio=hazard_ratio, follow_up=30.0
    )
    assert events == pytest.approx(logrank_events(hazard_ratio), rel=0.03)


def test_cure_three_ways_agree():
    latency = sp.Weibull.from_params([2.0, 1.2])
    kw = dict(accrual_time=1.0, follow_up=3.0)
    n = cure_logrank_sample_size((0.4, 0.3), latency, 0.9, ratio=2.0, **kw)
    assert n[0] == pytest.approx(2 * n[1], abs=1)
    assert cure_logrank_power((0.4, 0.3), latency, n, **kw) >= 0.9


def test_cure_latency_pair_matches_hazard_ratio():
    latency = sp.Weibull.from_params([2.0, 1.2])
    hr = 0.7
    # S(t)^hr is a Weibull with scale alpha * hr^(-1/beta).
    scaled = sp.Weibull.from_params([2.0 * hr ** (-1 / 1.2), 1.2])
    kw = dict(follow_up=3.0, accrual_time=1.0)
    by_ratio = cure_logrank_sample_size(
        0.3, latency, latency_hazard_ratio=hr, **kw
    )
    by_pair = cure_logrank_sample_size(0.3, (scaled, latency), **kw)
    assert by_ratio == by_pair


def test_cure_needs_more_follow_up_for_a_cure_difference():
    # A difference only in the cured fraction shows late: a short study
    # needs far more units than a long one.
    latency = sp.Exponential.from_params([1.0])
    short = cure_logrank_sample_size((0.4, 0.3), latency, follow_up=0.5)
    long = cure_logrank_sample_size((0.4, 0.3), latency, follow_up=5.0)
    assert short[0] > 2 * long[0]


@pytest.mark.parametrize(
    "cured, latency_hr, match",
    [
        (0.3, 1.0, "no test can detect"),
        (1.0, 0.7, "cured"),
        (THREE, 0.7, "pair"),
    ],
)
def test_cure_refusals(cured, latency_hr, match):
    latency = sp.Exponential.from_params([1.0])
    with pytest.raises(ValueError, match=match):
        cure_logrank_sample_size(
            cured, latency, latency_hazard_ratio=latency_hr, follow_up=2.0
        )


def test_cure_refuses_a_ratio_with_a_pair():
    latency = sp.Exponential.from_params([1.0])
    with pytest.raises(ValueError, match="latency_hazard_ratio"):
        cure_logrank_sample_size(
            0.3,
            (latency, latency),
            latency_hazard_ratio=0.7,
            follow_up=2.0,
        )


def test_cure_needs_follow_up():
    latency = sp.Exponential.from_params([1.0])
    with pytest.raises(TypeError, match="follow_up"):
        cure_logrank_sample_size((0.4, 0.3), latency)
