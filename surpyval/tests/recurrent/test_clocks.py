"""``EffortClock`` and ``ChangePoint``: intensity models on another
clock.

The likelihoods against sums written out, the two identities the models
rest on (steady effort is the calendar model; a change after the data is
no change), the clocks against their definitions, recovery from
simulated data and the test of no change (seeds fixed before the results
were seen), saving, their part in the growth models' other tools, and
the refusals.

No published fit is checked: none could be read (Yamada, Ohtera and
Narihisa's and Zhao's papers, and the data sets they fit, are behind
publishers' walls). See DECISIONS.md. The size of the test of no change
is studied in ``calibration/test_change_point_size.py``.
"""

import json
import pickle
import warnings

import numpy as np
import pytest
from scipy.optimize import minimize

import surpyval as sp
from surpyval.recurrent import (
    HPP,
    ChangePoint,
    CoxLewis,
    CrowAMSAA,
    Duane,
    EffortClock,
    EffortTable,
    GoelOkumoto,
    InflectionS,
    MusaOkumoto,
    WeibullEffort,
    YamadaDelayedS,
    change_point_cb,
    change_point_test,
    compare_predictions,
)
from surpyval.recurrent.parametric.nhpp_fitter import NHPPFitter

BASES = [
    (GoelOkumoto, [80.0, 0.03]),
    (YamadaDelayedS, [80.0, 0.05]),
    (InflectionS, [80.0, 0.05, 2.0]),
    (MusaOkumoto, [3.0, 0.02]),
    (CrowAMSAA, [0.5, 1.3]),
    (Duane, [0.6, 0.7]),
    (CoxLewis, [0.2, -0.01]),
    (HPP, [0.7]),
]
IDS = [type(b).__name__.rstrip("_") for b, _ in BASES]

# One system's failure times and the end of its test, for the
# likelihoods: no model generated them.
X = np.array([2.0, 5.0, 9.0, 12.0, 14.0, 21.0, 27.0, 33.0, 46.0, 58.0, 70.0])
C = np.array([0] * 10 + [1])
ENDS, AMOUNTS = [10.0, 30.0, 70.0], [50.0, 60.0, 40.0]

EFFORT_SEED, CHANGE_SEED, TEST_SEED = 5361, 5362, 5363


def _data(x=X, c=C):
    return sp.handle_xicn(x, c=c, as_recurrent_data=True)


# -- the effort over time -----------------------------------------------


def test_effort_table_is_its_definition():
    table = EffortTable(ENDS, AMOUNTS)
    t = np.array([0.0, 4.0, 10.0, 20.0, 30.0, 50.0, 70.0, 90.0])
    # 5 a unit of time to 10, then 3 to 30, then 1 (and on after 70)
    expected = [0.0, 20.0, 50.0, 80.0, 110.0, 130.0, 150.0, 170.0]
    np.testing.assert_allclose(table.cumulative(t), expected, rtol=1e-14)
    np.testing.assert_array_equal(
        table.rate([4.0, 10.0, 10.5, 30.0, 31.0, 90.0]),
        [5.0, 5.0, 3.0, 3.0, 1.0, 1.0],
    )
    np.testing.assert_allclose(table.inverse(expected), t, rtol=1e-14)


def test_effort_table_with_an_idle_period():
    table = EffortTable([10.0, 20.0, 30.0], [10.0, 0.0, 30.0])
    np.testing.assert_array_equal(
        table.cumulative([10.0, 15.0, 20.0, 25.0]), [10.0, 10.0, 10.0, 25.0]
    )
    # the earliest time by which the effort was spent
    np.testing.assert_array_equal(table.inverse([10.0, 25.0]), [10.0, 25.0])


def test_weibull_effort_is_its_definition_and_is_found_from_a_table():
    n, beta, m = 100.0, 0.02, 2.0
    curve = WeibullEffort(n, beta, m)
    t = np.array([0.0, 0.5, 3.0, 9.0, 20.0])
    total = n * (1 - np.exp(-beta * t**m))
    np.testing.assert_allclose(curve.cumulative(t), total, rtol=1e-13)
    np.testing.assert_allclose(
        curve.rate(t), n * beta * m * t * np.exp(-beta * t**m), rtol=1e-13
    )
    np.testing.assert_allclose(curve.inverse(total), t, rtol=1e-9, atol=1e-12)
    assert curve.inverse(n) == np.inf
    ends = np.arange(1.0, 11.0)
    amounts = np.diff(n * (1 - np.exp(-beta * ends**m)), prepend=0.0)
    fitted = WeibullEffort.fit(ends, amounts)
    np.testing.assert_allclose(
        [fitted.N, fitted.beta, fitted.m], [n, beta, m], rtol=1e-5
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(ends=[1, 2, 3], amounts=[5, -1, 2]), "cannot fall"),
        (dict(ends=[1, 3, 2], amounts=[5, 1, 2]), "increasing order"),
        (dict(ends=[0, 1], amounts=[5, 1]), "increasing order"),
        (dict(ends=[1, 2], amounts=[0, 0]), "no testing was done"),
        (dict(ends=[1, 2], amounts=[1, 2, 3]), "of one length"),
    ],
)
def test_effort_table_refusals(kwargs, match):
    with pytest.raises(ValueError, match=match):
        EffortTable(**kwargs)


def test_weibull_effort_refusals():
    with pytest.raises(ValueError, match="m must be a number above 0"):
        WeibullEffort(10.0, 0.1, 0.0)


# -- the effort clock ---------------------------------------------------


def test_effort_likelihood_is_the_sum_written_out():
    a, b = 40.0, 0.012
    table = EffortTable(ENDS, AMOUNTS)
    negll = EffortClock(GoelOkumoto, table).create_negll_func(_data())
    events = X[:-1]
    w, rate = table.cumulative(events), table.rate(events)
    direct = np.sum(np.log(a * b * rate) - b * w) - a * (
        1 - np.exp(-b * table.cumulative(X[-1]))
    )
    assert negll(np.array([a, b])) == pytest.approx(-direct, rel=1e-13)


def test_effort_likelihood_of_counts_is_the_sum_written_out():
    from scipy.special import gammaln

    a, b = 60.0, 0.004
    weeks = np.arange(6.0)
    hours = np.array([40.0, 120, 160, 160, 80, 40])
    counts = np.array([6, 15, 14, 9, 3, 1])
    data = sp.handle_xicn(
        np.column_stack([weeks, weeks + 1]),
        c=[2] * 6,
        n=counts,
        as_recurrent_data=True,
    )
    negll = EffortClock(GoelOkumoto, (weeks + 1, hours)).create_negll_func(
        data
    )
    total = np.concatenate([[0.0], np.cumsum(hours)])
    mean = a * (np.exp(-b * total[:-1]) - np.exp(-b * total[1:]))
    direct = np.sum(counts * np.log(mean) - mean - gammaln(counts + 1))
    assert negll(np.array([a, b])) == pytest.approx(-direct, rel=1e-13)


@pytest.mark.parametrize("base, params", BASES, ids=IDS)
def test_steady_effort_is_the_calendar_model(base, params):
    """Effort of 1 per unit time is time: the same functions and the same
    likelihood, to the last bits."""
    steady = EffortClock(base, ([35.0, 70.0], [35.0, 35.0]))
    t = np.array([0.5, 3.0, 20.0, 35.0, 60.0, 70.0, 95.0])
    for name in ("cif", "iif", "log_iif"):
        np.testing.assert_allclose(
            getattr(steady, name)(t, *params),
            getattr(base, name)(t, *params),
            rtol=1e-14,
            err_msg=name,
        )
    counts = base.cif(t, *params)
    np.testing.assert_allclose(
        steady.inv_cif(counts, *params),
        base.inv_cif(counts, *params),
        rtol=1e-12,
    )
    p = np.array(params)
    assert steady.create_negll_func(_data())(p) == pytest.approx(
        NHPPFitter.create_negll_func(base, _data())(p), rel=1e-13
    )


def test_steady_effort_fits_as_the_calendar_model():
    """At 2 of effort a unit of time, the detection rate is per unit of
    effort: half the calendar model's. The same maximised likelihood."""
    truth = GoelOkumoto.from_params([80.0, 0.03])
    d = truth.time_terminated_simulation_data(100.0, random_state=EFFORT_SEED)
    plain = GoelOkumoto._fit_from_recurrent_data(d)
    doubled = EffortClock(
        GoelOkumoto, ([50.0, 100.0], [100.0, 100.0])
    )._fit_from_recurrent_data(d)
    np.testing.assert_allclose(
        doubled.params, plain.params * [1.0, 0.5], rtol=1e-4
    )
    assert doubled.aic() == pytest.approx(plain.aic(), abs=1e-6)
    np.testing.assert_allclose(doubled.cif(40.0), plain.cif(40.0), rtol=1e-4)
    np.testing.assert_allclose(doubled.iif(40.0), plain.iif(40.0), rtol=1e-4)


@pytest.mark.parametrize(
    "effort",
    [EffortTable(ENDS, AMOUNTS), WeibullEffort(160.0, 0.001, 1.8)],
    ids=["table", "curve"],
)
def test_the_clock_in_time(effort):
    """The curve is the model's at the effort spent, the intensity its
    slope in time, and ``inv_cif`` its inverse."""
    params = [40.0, 0.012]
    clock = EffortClock(GoelOkumoto, effort)
    t = np.array([3.0, 9.0, 17.0, 29.0, 44.0, 66.0])
    np.testing.assert_allclose(
        clock.cif(t, *params),
        GoelOkumoto.cif(effort.cumulative(t), *params),
        rtol=1e-14,
    )
    h = 1e-5
    slope = (clock.cif(t + h, *params) - clock.cif(t - h, *params)) / (2 * h)
    np.testing.assert_allclose(clock.iif(t, *params), slope, rtol=1e-7)
    np.testing.assert_allclose(
        clock.inv_cif(clock.cif(t, *params), *params), t, rtol=1e-9
    )


def test_effort_recovery_from_simulated_data():
    """150 faults found at 0.004 per unit of effort, with effort that
    rises and falls over ten periods (600 in all, so about 136 failures):
    each estimate within three standard errors, and the effort model
    ahead of the calendar model, which has the same parameters."""
    truth = [150.0, 0.004]
    table = EffortTable(
        np.arange(10.0, 101.0, 10.0),
        [20.0, 40, 80, 120, 120, 80, 60, 40, 20, 20],
    )
    clock = EffortClock(GoelOkumoto, table)
    d = clock.from_params(truth).time_terminated_simulation_data(
        100.0, random_state=EFFORT_SEED
    )
    model = clock._fit_from_recurrent_data(d)
    assert model.maximum == "verified"
    z = (model.params - truth) / model.standard_errors()
    assert np.all(np.abs(z) < 3.0), z
    assert model.aic() < GoelOkumoto._fit_from_recurrent_data(d).aic()


def test_effort_fit_refusals():
    table = EffortTable([10.0, 20.0, 30.0], [10.0, 0.0, 30.0])
    with pytest.raises(ValueError, match="effort's rate is 0"):
        EffortClock(GoelOkumoto, table).fit([5.0, 15.0, 25.0, 30.0])
    with pytest.raises(ValueError, match="takes an intensity model"):
        EffortClock("GoelOkumoto", table)
    with pytest.raises(ValueError, match="takes an intensity model"):
        EffortClock(EffortClock(GoelOkumoto, table), table)
    with pytest.raises(ValueError, match="effort must be an EffortTable"):
        EffortClock(GoelOkumoto, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="cannot fall"):
        EffortClock(GoelOkumoto, ([1, 2], [3, -1]))


def test_no_growth_is_judged_on_the_effort_clock():
    """Failures at a steady pace in time show growth to the calendar
    model's eye only if they slow; with effort that was cut back, they
    came no slower per unit of effort, and the model is refused."""
    x = np.array([4.0, 9.0, 14.0, 20.0, 27.0, 35.0, 45.0, 57.0, 72.0, 90.0])
    c = [0] * 9 + [1]
    assert GoelOkumoto.fit(x, c=c).params[0] > 0
    cut_back = EffortClock(GoelOkumoto, ([30.0, 60.0, 90.0], [300.0, 60, 15]))
    with pytest.raises(ValueError, match="no reliability growth") as err:
        cut_back.fit(x, c=c)
    assert "mean effort" in str(err.value)
    assert "testing effort, not time" in str(err.value)


# -- the change point ---------------------------------------------------


def test_change_point_likelihood_is_the_sum_written_out():
    """Zhao's model: the detection rate ``b`` before ``tau`` and ``rho
    b`` after, the curve continuous."""
    a, b, rho, tau = 40.0, 0.012, 2.5, 20.0
    negll = ChangePoint(GoelOkumoto, tau=tau).create_negll_func(_data())
    events = X[:-1]
    late = events > tau
    exponent = np.where(late, b * tau + rho * b * (events - tau), b * events)
    log_rate = np.log(a * b) + np.where(late, np.log(rho), 0.0) - exponent
    end = b * tau + rho * b * (X[-1] - tau)
    direct = np.sum(log_rate) - a * (1 - np.exp(-end))
    assert negll(np.array([a, b, rho])) == pytest.approx(-direct, rel=1e-13)


@pytest.mark.parametrize("base, params", BASES, ids=IDS)
def test_a_change_after_the_data_is_no_change(base, params):
    """With ``tau`` beyond every time, and with ``ratio = 1`` at any
    ``tau``, the model is the plain one: the same functions and the same
    likelihood, whatever the other of the two is."""
    t = np.array([0.5, 3.0, 20.0, 35.0, 60.0, 70.0])
    p = np.array(params)
    plain = NHPPFitter.create_negll_func(base, _data())(p)
    for tau, ratio in ((80.0, 0.2), (80.0, 7.0), (10.0, 1.0), (33.0, 1.0)):
        changed = ChangePoint(base, tau=tau)
        for name in ("cif", "iif", "log_iif"):
            np.testing.assert_allclose(
                getattr(changed, name)(t, *params, ratio),
                getattr(base, name)(t, *params),
                rtol=1e-14,
                err_msg=name,
            )
        assert changed.create_negll_func(_data())(np.append(p, ratio)) == (
            pytest.approx(plain, rel=1e-13)
        )


def test_a_change_after_the_data_is_refused_by_the_fit():
    with pytest.raises(ValueError, match="beyond the data leaves the model"):
        ChangePoint(GoelOkumoto, tau=80.0).fit(X, c=C)
    with pytest.raises(ValueError, match="no failure before it"):
        ChangePoint(GoelOkumoto, tau=1.0).fit(X, c=C)
    with pytest.raises(ValueError, match="no failure after it"):
        ChangePoint(GoelOkumoto, tau=60.0).fit(X, c=C)


def _two_rates(tau=30.0):
    events = X[:-1]
    before, after = np.sum(events <= tau), np.sum(events > tau)
    return before, after, before / tau, after / (X[-1] - tau)


def test_a_constant_rate_that_changes_is_its_closed_form():
    """``ChangePoint(HPP, tau)``: the failures over the time, on each
    side of ``tau``. The likelihood at those rates is the closed form's,
    and the fit's is no higher and as high to 1e-9."""
    before, after, rate, later = _two_rates()
    closed = before * np.log(rate) + after * np.log(later) - (before + after)
    model = ChangePoint(HPP, tau=30.0).fit(X, c=C)
    at_closed = -model._neg_ll(np.array([rate, later / rate]))
    assert at_closed == pytest.approx(closed, rel=1e-13)
    fitted = -model._neg_ll(model.params)
    assert fitted <= at_closed + 1e-12
    assert fitted == pytest.approx(closed, rel=1e-9)


def test_a_constant_rate_that_changes_fits_to_its_closed_form():
    before, after, rate, later = _two_rates()
    model = ChangePoint(HPP, tau=30.0).fit(X, c=C)
    np.testing.assert_allclose(model.params, [rate, later / rate], rtol=1e-5)
    assert model.cif(X[-1]) == pytest.approx(before + after, rel=1e-6)
    assert model.maximum == "verified"


def test_a_constant_rate_that_changes_on_several_systems_is_exact():
    """Systems that start late (``tl``) and end before or after ``tau``:
    the failures over the time observed on each side. No search finds a
    higher likelihood."""
    x = [3.0, 8.0, 15.0, 25.0, 40.0, 12.0, 19.0, 33.0, 50.0, 7.0, 22.0]
    i = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3]
    c = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1]
    tl = [0.0] * 5 + [10.0] * 4 + [0.0] * 2
    tau = 20.0
    # Failures 3 + 2 + 1 before tau, over 20 + 10 + 20 observed there;
    # 1 + 1 + 0 after, over 20 + 30 + 2
    rate = 6 / 50.0
    later = 2 / 52.0
    model = ChangePoint(HPP, tau=tau).fit(x, i=i, c=c, tl=tl)
    np.testing.assert_allclose(model.params, [rate, later / rate], rtol=1e-13)
    searched = minimize(
        lambda u: model._neg_ll(np.exp(u)),
        np.log(model.params) + 0.1,
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-14},
    )
    # To rounding: the two agree to the last bits
    assert model._neg_ll(model.params) <= searched.fun + 1e-12


def test_a_constant_rate_that_changes_from_counts_is_searched():
    """Counts in periods have no closed form split at ``tau`` (a period
    may straddle it): searched, to the counts' rates where none does."""
    x = [[0.0, 10.0], [10.0, 20.0], [20.0, 40.0], [40.0, 60.0]]
    model = ChangePoint(HPP, tau=20.0).fit(x, c=[2] * 4, n=[4, 6, 3, 1])
    np.testing.assert_allclose(model.params, [0.5, 0.2], rtol=1e-3)
    assert "closed-form" not in str(model.res.message)


# The NTDS failures (the guide's data): steady to day 105, then few
NTDS = np.cumsum(
    [9, 12, 11, 4, 7, 2, 5, 8, 5, 7, 1, 6, 1]
    + [9, 4, 1, 3, 3, 6, 1, 11, 33, 7, 91, 2, 1]
)


def test_a_fit_at_the_models_limit_is_refused():
    """Where the failures are steady on each side of the change, a
    growth model with a change point fits no better than a constant rate
    that changes: its faults run off to infinity, and it is refused,
    naming the model that is its limit. That model's likelihood is the
    limit's, in closed form."""
    for fitter in (
        ChangePoint(GoelOkumoto, tau=105.0),
        ChangePoint(GoelOkumoto),
    ):
        with pytest.raises(ValueError, match="no reliability growth") as err:
            fitter.fit(NTDS)
        assert "ChangePoint(HPP, tau=...)" in str(err.value)
        assert "tau = 105" in str(err.value)
    steps = ChangePoint(HPP).fit(NTDS)
    assert steps.change_point["tau"] == 105.0
    limit = ChangePoint(GoelOkumoto, tau=105.0)._limit_log_likelihood(
        sp.handle_xicn(NTDS, as_recurrent_data=True), 0.5
    )
    closed = 20 * np.log(20 / 105) + 6 * np.log(6 / 145) - 26
    assert limit == pytest.approx(closed, rel=1e-10)
    assert -steps._neg_ll(steps.params) == pytest.approx(closed, rel=1e-8)
    # a model with growth to show is well above its limit
    delayed = ChangePoint(YamadaDelayedS, tau=105.0).fit(NTDS)
    assert delayed.params[0] < 1e4


@pytest.mark.parametrize("base, params", BASES, ids=IDS)
def test_the_curve_is_continuous_and_the_rate_jumps(base, params):
    tau, ratio = 20.0, 3.0
    changed = ChangePoint(base, tau=tau)
    below, above = np.nextafter(tau, 0.0), np.nextafter(tau, np.inf)
    assert changed.cif(above, *params, ratio) == pytest.approx(
        changed.cif(below, *params, ratio), rel=1e-12
    )
    assert changed.iif(above, *params, ratio) == pytest.approx(
        ratio * changed.iif(below, *params, ratio), rel=1e-12
    )
    t = np.array([5.0, 20.0, 26.0, 50.0])
    np.testing.assert_allclose(
        changed.inv_cif(changed.cif(t, *params, ratio), *params, ratio),
        t,
        rtol=1e-9,
    )
    # the same curve, sooner: where the plain model is at 20 + 3 (t - 20)
    np.testing.assert_allclose(
        changed.cif(26.0, *params, ratio), base.cif(38.0, *params), rtol=1e-14
    )


def test_the_total_is_unchanged():
    model = ChangePoint(GoelOkumoto, tau=30.0).from_params([80, 0.01, 4.0])
    assert model.cif(np.inf) == 80.0
    assert model.remaining_faults(30.0) == pytest.approx(
        80 * np.exp(-0.3), rel=1e-13
    )
    assert model.change_point == {"tau": 30.0}


@pytest.fixture(scope="module")
def changed_data():
    """80 faults, found at 0.006 a unit of time to 40 and four times as
    fast after: about 17 failures before the change and 62 after, in a
    test to 200 (long enough for the failures to slow: at 100 they have
    not, and the model without a change has no fit; see
    ``test_a_change_with_no_growth_overall``)."""
    truth = ChangePoint(GoelOkumoto, tau=40.0).from_params([80.0, 0.006, 4.0])
    return truth.time_terminated_simulation_data(
        200.0, random_state=CHANGE_SEED
    )


@pytest.fixture(scope="module")
def estimated(changed_data):
    return ChangePoint(GoelOkumoto)._fit_from_recurrent_data(changed_data)


def test_change_point_recovery_with_tau_given(changed_data):
    model = ChangePoint(GoelOkumoto, tau=40.0)._fit_from_recurrent_data(
        changed_data
    )
    assert model.maximum == "verified"
    z = (model.params - [80.0, 0.006, 4.0]) / model.standard_errors()
    assert np.all(np.abs(z) < 3.0), z


def test_change_point_recovery_with_tau_estimated(changed_data, estimated):
    info = estimated.change_point
    assert info["estimated"]
    assert abs(info["tau"] - 40.0) < 15.0, info["tau"]
    lower, upper = change_point_cb(estimated)
    assert lower <= 40.0 <= upper, (lower, upper)
    assert lower <= info["tau"] <= upper
    # wider at a higher level
    wide = change_point_cb(estimated, alpha_ci=0.01)
    assert wide[0] <= lower and upper <= wide[1]
    # no candidate time does better than the estimate, and the estimate
    # no worse than the fit at the true time
    best = -estimated._neg_ll(estimated.params)
    assert best >= info["log_likelihood"].max() - 1e-6
    at_truth = ChangePoint(GoelOkumoto, tau=40.0)._fit_from_recurrent_data(
        changed_data
    )
    assert best >= -at_truth._neg_ll(at_truth.params) - 1e-6


def test_the_profile_is_the_fit_at_each_time(changed_data, estimated):
    """The profile's values are those of the full fit with ``tau``
    given, at a few of its candidate times."""
    info = estimated.change_point
    for k in (0, len(info["candidates"]) // 2, len(info["candidates"]) - 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ChangePoint(
                GoelOkumoto, tau=float(info["candidates"][k])
            )._fit_from_recurrent_data(changed_data)
        assert info["log_likelihood"][k] == pytest.approx(
            -fit._neg_ll(fit.params), abs=1e-5
        )


def test_the_test_of_no_change_finds_the_change(estimated):
    out = change_point_test(estimated, n_boot=19, random_state=TEST_SEED)
    again = change_point_test(estimated, n_boot=19, random_state=TEST_SEED)
    assert out == again
    assert out["n_boot"] + out["skipped"] == 19
    assert out["p_value"] == 1 / (1 + out["n_boot"])
    info = estimated.change_point
    assert out["statistic"] == pytest.approx(
        2 * (-estimated._neg_ll(estimated.params))
        - 2 * info["no_change_log_likelihood"],
        rel=1e-12,
    )
    assert out["tau"] == info["tau"]
    assert out["ratio"] == estimated.params[-1]


def test_a_change_with_no_growth_overall():
    """A detection rate that rose, seen before the failures slow: the
    model without a change is refused, the change point is still
    estimated, and the test of no change says there is nothing to test
    against."""
    truth = ChangePoint(GoelOkumoto, tau=40.0).from_params([80.0, 0.006, 4.0])
    d = truth.time_terminated_simulation_data(100.0, random_state=CHANGE_SEED)
    with pytest.raises(ValueError, match="no reliability growth"):
        GoelOkumoto._fit_from_recurrent_data(d)
    model = ChangePoint(GoelOkumoto)._fit_from_recurrent_data(d)
    assert model.change_point["no_change_log_likelihood"] is None
    assert model.params[-1] > 1.0
    with pytest.raises(ValueError, match="no fit without a change"):
        change_point_test(model, random_state=TEST_SEED)


def test_change_point_refusals(estimated):
    with pytest.raises(ValueError, match="tau is the time of the change"):
        ChangePoint(GoelOkumoto, tau=0.0)
    with pytest.raises(ValueError, match="takes an intensity model"):
        ChangePoint(ChangePoint(GoelOkumoto, tau=3.0))
    free = ChangePoint(GoelOkumoto)
    with pytest.raises(ValueError, match="has no tau"):
        free.from_params([80, 0.01, 2.0])
    with pytest.raises(ValueError, match="needs each failure's time"):
        weeks = np.arange(6.0)
        free.fit(
            np.column_stack([weeks, weeks + 1]),
            c=[2] * 6,
            n=[6, 15, 14, 9, 3, 1],
        )
    with pytest.raises(ValueError, match="one system's failures"):
        free.fit(np.tile(X, 2), i=np.repeat([1, 2], X.size), c=np.tile(C, 2))
    with pytest.raises(ValueError, match="at least 7 failures"):
        free.fit([1.0, 2.0, 4.0, 7.0, 11.0, 16.0])
    with pytest.raises(ValueError, match="how='MSE' needs"):
        free.fit(X, c=C, how="MSE")
    with pytest.raises(ValueError, match="takes no frailty"):
        free.fit(X, c=C, frailty="gamma")
    with pytest.raises(ValueError, match="init must have 3 values"):
        free.fit(X, c=C, init=[40.0, 0.01])
    given = ChangePoint(GoelOkumoto, tau=20.0).fit(X, c=C)
    for func in (change_point_cb, change_point_test):
        with pytest.raises(ValueError, match="change point estimated"):
            func(given)
    with pytest.raises(ValueError, match="n_boot is the number"):
        change_point_test(estimated, n_boot=5)
    with pytest.raises(ValueError):
        change_point_cb(estimated, alpha_ci=1.5)


def test_a_change_of_units_rescales_the_clock_with_the_data():
    """Times in another unit, with the table's periods and ``tau`` in it
    too, give the same curve (the conformance suite's property, which
    cannot rescale a fitter's own times)."""
    k = 7.3
    t = np.array([5.0, 20.0, 45.0, 70.0])
    pairs = [
        (
            EffortClock(GoelOkumoto, ([35.0, 70.0], [30.0, 40.0])),
            EffortClock(GoelOkumoto, ([35.0 * k, 70.0 * k], [30.0, 40.0])),
        ),
        (
            ChangePoint(GoelOkumoto, tau=20.0),
            ChangePoint(GoelOkumoto, tau=20.0 * k),
        ),
    ]
    for fitter, rescaled in pairs:
        ref = fitter.fit(X, c=C)
        got = rescaled.fit(X * k, c=C)
        np.testing.assert_allclose(got.cif(t * k), ref.cif(t), rtol=1e-3)
        np.testing.assert_allclose(got.iif(t * k) * k, ref.iif(t), rtol=1e-3)


# -- with the rest of the package --------------------------------------


def _clock_fits(changed_data):
    table = EffortTable([40.0, 200.0], [40.0, 640.0])
    return [
        EffortClock(GoelOkumoto, table)._fit_from_recurrent_data(changed_data),
        ChangePoint(GoelOkumoto, tau=40.0)._fit_from_recurrent_data(
            changed_data
        ),
    ]


def test_saving_and_loading(changed_data, estimated):
    t = np.array([10.0, 40.0, 75.0, 230.0])
    for model in (*_clock_fits(changed_data), estimated):
        saved = json.loads(json.dumps(model.to_dict(), allow_nan=False))
        back = sp.from_dict(saved)
        assert back.dist.name == model.dist.name
        np.testing.assert_array_equal(back.cif(t), model.cif(t))
        np.testing.assert_array_equal(back.iif(t), model.iif(t))
        np.testing.assert_allclose(back.cif_cb(t), model.cif_cb(t), rtol=1e-12)
        kept = pickle.loads(pickle.dumps(model))
        assert kept.aic() == model.aic()
        np.testing.assert_array_equal(kept.cif(t), model.cif(t))
    assert sp.from_dict(estimated.to_dict()).change_point == {
        "tau": estimated.change_point["tau"]
    }
    curve = EffortClock(GoelOkumoto, WeibullEffort(160.0, 0.001, 1.8))
    model = curve.from_params([40.0, 0.012])
    back = sp.from_dict(json.loads(model.to_json()))
    np.testing.assert_array_equal(back.cif(t), model.cif(t))


def test_the_growth_models_answers_are_in_time(changed_data):
    """Faults left, the testing to a target intensity and the release
    time, from a model on a clock."""
    for model in _clock_fits(changed_data):
        a = model.params[0]
        assert model.remaining_faults(200.0) == pytest.approx(
            a - model.cif(200.0), rel=1e-12
        )
        lower, upper = model.remaining_faults_cb(200.0)
        assert lower < model.remaining_faults(200.0) < upper
        target = 0.5 * float(model.iif(200.0))
        when = model.time_to_intensity(target)
        assert when > 200.0
        assert model.iif(when) == pytest.approx(target, rel=1e-6)
        plan = model.release_time(1.0, 2.0, 12.0)
        # testing pays while the intensity is above 1 / (12 - 2)
        assert model.iif(plan["time"]) == pytest.approx(0.1, rel=1e-5)


def test_they_are_compared_by_their_predictions(changed_data):
    x = changed_data.x[changed_data.c == 0]
    table = EffortTable([40.0, 200.0], [40.0, 640.0])
    out = compare_predictions(
        x,
        [
            GoelOkumoto,
            EffortClock(GoelOkumoto, table),
            ChangePoint(GoelOkumoto, tau=40.0),
        ],
        start=40,
    )
    text = repr(out)
    assert "EffortClock(GoelOkumoto)" in text
    assert "ChangePoint(GoelOkumoto)" in text


def test_repr_and_a_constant_rate_per_unit_of_effort():
    table = EffortTable(ENDS, AMOUNTS)
    assert repr(EffortClock(GoelOkumoto, table)).startswith(
        "EffortClock(GoelOkumoto): "
    )
    hours = EffortClock(HPP, table).fit(X, c=C)
    np.testing.assert_allclose(hours.params, [10 / 150.0], rtol=1e-5)
    assert repr(ChangePoint(MusaOkumoto, tau=3.0)).startswith(
        "ChangePoint(MusaOkumoto): "
    )
