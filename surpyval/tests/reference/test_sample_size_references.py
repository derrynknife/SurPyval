"""The sample-size and power calculations of
``surpyval.test_planning.sample_size`` against R.

``scripts/reference/reference_r_sample_size.R`` writes
``data/r_sample_size.json``:

- **Cox model** (powerSurvEpi): ``ssizeEpiCont`` (units, continuous
  covariate, with Hsieh and Lavori's 2000 published example),
  ``powerEpiCont`` (power) and ``numDEpi`` (events, two-level covariate,
  with Latouche, Porcher and Chevret's 2004 example). The same formula,
  so the rounded counts must be equal and the power equal to 1e-12.
- **Non-inferiority** (gsDesign ``nEvents(hr0=...)`` and rpact
  ``getSampleSizeSurvival(thetaH0=...)``): the events, unrounded in R,
  equal to SurPyval's before rounding (to 1e-9 relative), so the rounded
  counts agree.
- **RMST difference and the log-rank test with cured units** (npsurvSS,
  ``size_two_arm`` and ``power_two_arm``, Yung and Liu 2020): the same
  asymptotic mean and variance, which npsurvSS integrates with R's
  ``integrate`` and SurPyval on a fixed grid of 40,000 intervals, so the
  unrounded units and events agree to ``rtol=1e-5`` and the power to
  ``atol=1e-5``. npsurvSS's loss to follow-up of rate 1e-12 stands for
  none. The events are compared with npsurvSS's general integral of the
  event density (``d_integral``): its reported ``d`` uses a closed form
  for exponential arms that leaves out the cured fraction, and is wrong
  for them (``test_npsurvss_closed_form_events_leave_out_the_cure``).

Equivalence of a hazard ratio has no R function among these (gsDesign
and rpact plan non-inferiority and superiority only): it is checked
against the closed form of Chow, Shao and Wang (2008) in
``surpyval/tests/test_planning/test_sample_size.py``.
"""

import numpy as np
import pytest
from scipy import stats

from surpyval.test_planning import (
    cox_events,
    cox_power,
    cox_sample_size,
    cure_logrank_power,
    noninferiority_events,
    rmst_power,
)
from surpyval.test_planning import sample_size as module

from ._data import reference

SOURCE = "r_sample_size"


def _entries(prefix):
    from ._data import _load

    return sorted(
        k for k in _load(SOURCE)["references"] if k.startswith(prefix)
    )


@pytest.mark.parametrize("ref_id", _entries("ssize_cont_"))
def test_cox_sample_size_continuous(ref_id):
    v = reference(SOURCE, ref_id)["values"]
    n = cox_sample_size(
        v["theta"],
        v["psi"],
        v["power"],
        sd=np.sqrt(v["sigma2"]),
        r_squared=v["rho2"],
        alpha_ci=v["alpha"],
    )
    assert n == v["n"]


@pytest.mark.parametrize("ref_id", _entries("power_cont_"))
def test_cox_power_continuous(ref_id):
    v = reference(SOURCE, ref_id)["values"]
    power = cox_power(
        v["theta"],
        int(v["n"]),
        v["psi"],
        sd=np.sqrt(v["sigma2"]),
        r_squared=v["rho2"],
        alpha_ci=v["alpha"],
    )
    assert power == pytest.approx(v["power"], abs=1e-12)


@pytest.mark.parametrize("ref_id", _entries("events_binary_"))
def test_cox_events_two_levels(ref_id):
    v = reference(SOURCE, ref_id)["values"]
    events = cox_events(
        v["theta"],
        v["power"],
        proportion=v["p"],
        r_squared=v["rho2"],
        alpha_ci=v["alpha"],
    )
    assert events == int(np.ceil(v["events"] - 1e-9))


@pytest.mark.parametrize("ref_id", _entries("noninferiority_"))
def test_noninferiority_events(ref_id):
    v = reference(SOURCE, ref_id)["values"]
    share = v["ratio"] / (1 + v["ratio"])
    z = stats.norm.ppf(0.975) + stats.norm.ppf(1 - v["beta"])
    unrounded = z**2 / (share * (1 - share) * np.log(v["hr0"] / v["hr"]) ** 2)
    assert unrounded == pytest.approx(v["events"], rel=1e-9)
    events = noninferiority_events(
        v["hr0"], 1 - v["beta"], hazard_ratio=v["hr"], ratio=v["ratio"]
    )
    assert events == int(np.ceil(v["events"] - 1e-9))


def _npsurvss_curve(arm):
    """The survival function of an npsurvSS arm: a cured fraction and a
    Weibull, or piecewise exponential, latency, S(t) = exp(-(scale t) ^
    shape)."""
    cure = arm["cure"]
    knots = np.array([float(k) for k in np.atleast_1d(arm["interval"])])
    scale = np.atleast_1d(np.asarray(arm["scale"], dtype=float))
    shape = arm["shape"]

    def sf(t):
        t = np.asarray(t, dtype=float)
        if scale.size == 1:
            latency = np.exp(-((scale[0] * t) ** shape))
        else:
            widths = np.clip(t[..., None] - knots[:-1], 0.0, np.diff(knots))
            latency = np.exp(-(widths * scale).sum(axis=-1))
        return cure + (1 - cure) * latency

    return sf


def _design(v):
    return dict(
        accrual_time=v["accrual_time"],
        follow_up=v["follow_up"],
        dropout_rate=v["loss_rate"],
    )


def _z():
    return stats.norm.ppf(0.975) + stats.norm.ppf(0.8)


@pytest.mark.parametrize("ref_id", _entries("rmst_"))
def test_rmst_against_npsurvss(ref_id):
    entry = reference(SOURCE, ref_id)
    v = entry["values"]
    tau = float(entry["settings"]["test"].split("milestone = ")[1][:-1])
    curves = (_npsurvss_curve(v["arm1"]), _npsurvss_curve(v["arm0"]))
    difference, var_1, var_2 = module._rmst_setup(
        curves, tau, v["accrual_time"], v["follow_up"], v["loss_rate"]
    )
    per_second = _z() ** 2 * (var_1 / v["ratio"] + var_2) / difference**2
    assert per_second == pytest.approx(v["n0"], rel=1e-5)
    assert v["ratio"] * per_second == pytest.approx(v["n1"], rel=1e-5)
    power = rmst_power(curves, tau, tuple(v["power_at"]), **_design(v))
    assert power == pytest.approx(v["power"], abs=1e-5)


@pytest.mark.parametrize("ref_id", _entries("cure_"))
def test_cure_logrank_against_npsurvss(ref_id):
    v = reference(SOURCE, ref_id)["values"]
    sf_1, sf_2 = _npsurvss_curve(v["arm1"]), _npsurvss_curve(v["arm0"])
    share = v["ratio"] / (1 + v["ratio"])
    mean, variance, events_1, events_2 = module._logrank_moments(
        sf_1, sf_2, share, v["accrual_time"], v["follow_up"], v["loss_rate"]
    )
    total = _z() ** 2 * variance / mean**2
    assert share * total == pytest.approx(v["n1"], rel=1e-5)
    events = total * (share * events_1 + (1 - share) * events_2)
    assert events == pytest.approx(v["d_integral"], rel=1e-5)
    # The public function, with the latency as a function of time and no
    # cure (the cure is in the curve).
    power = cure_logrank_power(
        0.0, (sf_1, sf_2), tuple(v["power_at"]), **_design(v)
    )
    assert power == pytest.approx(v["power"], abs=1e-5)


def test_npsurvss_closed_form_events_leave_out_the_cure():
    # The one entry with exponential latencies and a cured fraction:
    # npsurvSS's reported events count the cured as failing, so they are
    # far above the integral (and SurPyval's); without a cure, or with a
    # Weibull latency, the two agree.
    for ref_id in _entries("cure_") + _entries("rmst_"):
        v = reference(SOURCE, ref_id)["values"]
        cured = v["arm1"]["cure"] > 0 or v["arm0"]["cure"] > 0
        if v["lachin"] and cured:
            assert v["d"] > 1.2 * v["d_integral"], ref_id
        else:
            assert v["d"] == pytest.approx(v["d_integral"], rel=1e-6)
