"""The software reliability growth models as NHPPs: ``GoelOkumoto``,
``YamadaDelayedS``, ``InflectionS`` and ``MusaOkumoto``.

- Their four functions against the closed forms worked in 50-digit
  decimal arithmetic (``decimal``, so the references are the same on
  every platform), from ``b t`` of 1e-9 to 800, and against each other.
- The likelihood against a sum written out here, for failure times and
  for counts per period.
- The fit against the model's own likelihood equations, solved here.
- Recovery of the parameters from simulated data (seeds fixed before the
  results were seen), the units rule, saving and loading, and the refusal
  of data that show no growth.

No published fit is quoted: see DECISIONS.md ("Software reliability
growth models") for the references that could not be read.
"""

import warnings
from decimal import Decimal, getcontext

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import gammaln

import surpyval as sp
from surpyval.recurrent import (
    HPP,
    CrowAMSAA,
    GoelOkumoto,
    InflectionS,
    MusaOkumoto,
    YamadaDelayedS,
)
from surpyval.recurrent.parametric.parametric_recurrence import (
    ParametricRecurrenceModel,
)
from surpyval.recurrent.parametric.software_growth import limit_centroids

getcontext().prec = 50

#: 20 times: ``b t`` from 1e-9, where ``1 - exp(-b t)`` written directly
#: rounds, to 800, where ``exp(-b t)`` is below the smallest double
TIMES = [
    1e-8, 1e-6, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0,
    4.9, 5.1, 10.0, 25.0, 60.0, 150.0, 400.0, 1000.0, 3000.0, 8000.0,
]  # fmt: skip

PARAMS = {
    "GoelOkumoto": (GoelOkumoto, [80.0, 0.1]),
    "YamadaDelayedS": (YamadaDelayedS, [80.0, 0.1]),
    "InflectionS": (InflectionS, [80.0, 0.1, 3.0]),
    "InflectionS[psi<0]": (InflectionS, [80.0, 0.1, -0.5]),
    "MusaOkumoto": (MusaOkumoto, [2.0, 0.05]),
}


def _exact(name, t, params):
    """(cif, iif) at ``t`` in decimal arithmetic."""
    t = Decimal(repr(t))
    p = [Decimal(repr(v)) for v in params]
    one = Decimal(1)
    if name == "GoelOkumoto":
        a, b = p
        decay = (-b * t).exp()
        return a * (one - decay), a * b * decay
    if name == "YamadaDelayedS":
        a, b = p
        decay = (-b * t).exp()
        return a * (one - (one + b * t) * decay), a * b * b * t * decay
    if name.startswith("InflectionS"):
        a, b, psi = p
        decay = (-b * t).exp()
        return (
            a * (one - decay) / (one + psi * decay),
            a * b * (one + psi) * decay / (one + psi * decay) ** 2,
        )
    lambda0, theta = p
    grown = one + lambda0 * theta * t
    return grown.ln() / theta, lambda0 / grown


@pytest.mark.parametrize("name", sorted(PARAMS))
def test_functions_match_the_closed_forms(name):
    fitter, params = PARAMS[name]
    t = np.array(TIMES)
    exact = [_exact(name, v, params) for v in TIMES]
    cif = np.array([float(c) for c, _ in exact])
    iif = np.array([float(i) for _, i in exact])
    log_iif = np.array([float(i.ln()) for _, i in exact])
    np.testing.assert_allclose(fitter.cif(t, *params), cif, rtol=4e-15)
    # exp(-800) is 0 in doubles, so the last intensities are compared
    # through their logs
    seen = fitter.iif(t, *params)
    np.testing.assert_allclose(seen[iif > 0], iif[iif > 0], rtol=4e-14)
    assert np.all(seen[iif == 0] == 0)
    np.testing.assert_allclose(fitter.log_iif(t, *params), log_iif, rtol=4e-15)


@pytest.mark.parametrize("name", sorted(PARAMS))
def test_iif_is_the_slope_of_cif(name):
    fitter, params = PARAMS[name]
    # Where a difference of counts resolves the slope: by t = 150 the
    # count is its total to 1e-8, and the difference is rounding.
    t = np.array([v for v in TIMES if 1e-3 <= v <= 60.0])
    h = 1e-5 * t
    slope = (fitter.cif(t + h, *params) - fitter.cif(t - h, *params)) / (2 * h)
    np.testing.assert_allclose(fitter.iif(t, *params), slope, rtol=1e-6)
    np.testing.assert_allclose(
        np.exp(fitter.log_iif(t, *params)),
        fitter.iif(t, *params),
        rtol=1e-13,
    )


@pytest.mark.parametrize("name", sorted(PARAMS))
def test_inv_cif_inverts_cif(name):
    fitter, params = PARAMS[name]
    # Up to where the curve still moves in doubles: past b t of about 30
    # a finite model's count is its total to the last bit.
    t = np.array([v for v in TIMES if v <= 150.0])
    back = fitter.inv_cif(fitter.cif(t, *params), *params)
    # The count at a time is known to a relative 1e-16, which is 1e-16 /
    # (slope of the log count) of the time: all of it for a small t,
    # and more as the curve flattens.
    flat = fitter.cif(t, *params) / (t * fitter.iif(t, *params))
    np.testing.assert_allclose(back, t, rtol=1e-9)
    assert np.all(np.abs(back / t - 1) <= 2e-14 * np.maximum(flat, 1.0))
    assert fitter.inv_cif(0.0, *params) == 0.0


@pytest.mark.parametrize(
    "name", ["GoelOkumoto", "YamadaDelayedS", "InflectionS"]
)
def test_a_count_of_the_total_is_never_reached(name):
    fitter, params = PARAMS[name]
    model = fitter.from_params(params)
    assert fitter.total_parameter == "a"
    assert model.cif(np.inf) == params[0]
    assert model.iif(np.inf) == 0.0
    out = model.inv_cif([79.0, 80.0, 120.0])
    assert np.isfinite(out[0]) and np.all(np.isinf(out[1:]))
    # so a draw to a number of events is refused, and one to a time made
    with pytest.raises(ValueError, match="levels off"):
        model.count_terminated_simulation(200, random_state=1)
    drawn = model.time_terminated_simulation_data(1e4, random_state=1)
    assert (drawn.c == 0).sum() < 130
    # and a unit may never fail again
    assert model.next_failure_qf(0.999999, age=[300.0])[0] == np.inf


def test_musa_okumoto_has_no_total():
    fitter, params = PARAMS["MusaOkumoto"]
    assert fitter.total_parameter is None
    model = fitter.from_params(params)
    assert model.cif(np.inf) == np.inf
    assert np.isfinite(model.inv_cif(500.0))
    assert model.count_terminated_simulation(20, random_state=1).x.size


def test_inflection_s_at_psi_zero_is_goel_okumoto():
    t = np.array(TIMES)
    np.testing.assert_allclose(
        InflectionS.cif(t, 80.0, 0.1, 0.0),
        GoelOkumoto.cif(t, 80.0, 0.1),
        rtol=1e-15,
    )
    np.testing.assert_allclose(
        InflectionS.log_iif(t, 80.0, 0.1, 0.0),
        GoelOkumoto.log_iif(t, 80.0, 0.1),
        rtol=1e-15,
    )
    # psi > 1 has its peak intensity at log(psi) / b
    peak = np.log(3.0) / 0.1
    around = InflectionS.iif(np.array([peak - 1, peak, peak + 1]), 80, 0.1, 3)
    assert around[1] > around[0] and around[1] > around[2]


# ---------------------------------------------------------------------------
# The likelihood
# ---------------------------------------------------------------------------
#: Two systems: failure times, each to the end of its test (c = 1)
TIMES_X = np.array([1.5, 4.0, 9.0, 17.0, 31.0, 50.0, 2.5, 8.0, 20.0, 40.0])
TIMES_I = np.array([1] * 6 + [2] * 4)
TIMES_C = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 1])

#: One system: the failures counted in each of eight weeks
WEEKS = np.column_stack([np.arange(8.0), np.arange(1.0, 9.0)])
COUNTS = np.array([12, 9, 8, 5, 4, 4, 2, 1])


@pytest.mark.parametrize("name", sorted(PARAMS))
def test_likelihood_of_failure_times_is_the_sum_written_out(name):
    fitter, params = PARAMS[name]
    data = sp.handle_xicn(TIMES_X, TIMES_I, TIMES_C, as_recurrent_data=True)
    direct = 0.0
    for item in (1, 2):
        rows = TIMES_I == item
        events = TIMES_X[rows & (TIMES_C == 0)]
        end = TIMES_X[rows].max()
        direct += float(np.sum(np.log(fitter.iif(events, *params))))
        direct -= float(fitter.cif(end, *params))
    seen = -fitter.create_negll_func(data)(np.array(params))
    assert seen == pytest.approx(direct, rel=1e-13)


@pytest.mark.parametrize("name", sorted(PARAMS))
def test_likelihood_of_counts_is_the_sum_written_out(name):
    fitter, params = PARAMS[name]
    data = sp.handle_xicn(WEEKS, c=[2] * 8, n=COUNTS, as_recurrent_data=True)
    mean = fitter.cif(WEEKS[:, 1], *params) - fitter.cif(WEEKS[:, 0], *params)
    direct = float(
        np.sum(COUNTS * np.log(mean) - mean - gammaln(COUNTS + 1.0))
    )
    seen = -fitter.create_negll_func(data)(np.array(params))
    assert seen == pytest.approx(direct, rel=1e-13)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------
def _same_fit(model, params):
    """The fitted parameters are ``params`` to the search's tolerance:
    within a thousandth of a standard error (the search stops where the
    likelihood is flat to its tolerance, which is a distance in standard
    errors, not in digits of a parameter) and 1e-4 of the value."""
    gap = np.abs(np.asarray(model.params) - np.asarray(params))
    assert np.all(gap < 1e-3 * model.standard_errors()), gap
    np.testing.assert_allclose(model.params, params, rtol=2e-4)


def test_goel_okumoto_fit_solves_its_likelihood_equations():
    # One system's failure times to T: a = n / (1 - exp(-b T)), and b
    # is the root of n / b - sum(t) - n T / (exp(b T) - 1).
    t = TIMES_X[:5]
    n, T = t.size, 50.0
    model = GoelOkumoto.fit(np.append(t, T), c=[0] * n + [1])

    def score(b):
        return n / b - t.sum() - n * T / np.expm1(b * T)

    b = brentq(score, 1e-6, 10.0, xtol=1e-15, rtol=1e-15)
    a = n / -np.expm1(-b * T)
    _same_fit(model, [a, b])
    assert model.maximum == "verified"


def test_musa_okumoto_fit_solves_its_likelihood_equations():
    # With k = lambda0 theta: theta = log(1 + k T) / n at the maximum, and
    # k is the root of n / k - sum(t / (1 + k t)) - n T / ((1 + k T)
    # log(1 + k T)).
    t = TIMES_X[:5]
    n, T = t.size, 50.0
    model = MusaOkumoto.fit(np.append(t, T), c=[0] * n + [1])

    def score(k):
        grown = 1 + k * T
        return (
            n / k - np.sum(t / (1 + k * t)) - n * T / (grown * np.log(grown))
        )

    k = brentq(score, 1e-6, 100.0, xtol=1e-15, rtol=1e-15)
    theta = np.log1p(k * T) / n
    _same_fit(model, [k / theta, theta])


def test_fit_to_counts_and_to_the_times_behind_them_agree_in_the_limit():
    # Counts over ever finer periods carry the times: the fits meet.
    truth = GoelOkumoto.from_params([150.0, 0.04])
    d = truth.time_terminated_simulation_data(80.0, random_state=11)
    exact = GoelOkumoto.fit(d.x, c=d.c)
    edges = np.linspace(0.0, 80.0, 1601)
    counts = np.histogram(d.x[d.c == 0], edges)[0]
    grouped = GoelOkumoto.fit(
        np.column_stack([edges[:-1], edges[1:]]),
        c=np.full(1600, 2),
        n=counts,
    )
    np.testing.assert_allclose(grouped.params, exact.params, rtol=2e-3)


#: name -> (true parameters, end of test); the sizes are the systems
RECOVERY = {
    "GoelOkumoto": ([60.0, 0.03], 100.0),
    "YamadaDelayedS": ([60.0, 0.05], 100.0),
    "InflectionS": ([60.0, 0.06], 100.0),
    "MusaOkumoto": ([3.0, 0.04], 100.0),
}
RECOVERY["InflectionS"] = ([60.0, 0.06, 4.0], 100.0)


@pytest.mark.parametrize("systems", [1, 4, 16])
@pytest.mark.parametrize("name", sorted(RECOVERY))
def test_recovery_from_simulated_data(name, systems):
    # Seeds fixed before the results were seen: each estimate within 3
    # of its standard errors of the truth (a 99.7% statement for each;
    # 12 fits of 2 or 3 parameters), and the standard errors falling as
    # the systems are added.
    fitter = PARAMS[name][0]
    params, end = RECOVERY[name]
    truth = fitter.from_params(params)
    d = truth.time_terminated_simulation_data(
        end, items=systems, random_state=20261120
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = fitter.fit(d.x, d.i, d.c, d.n)
    assert model.maximum == "verified"
    se = model.standard_errors()
    z = (model.params - np.array(params)) / se
    print(name, systems, model.params, se, z)
    assert np.all(np.abs(z) < 3.0), z


@pytest.mark.parametrize("name", sorted(RECOVERY))
def test_recovery_from_counts_per_period(name):
    fitter = PARAMS[name][0]
    params, end = RECOVERY[name]
    truth = fitter.from_params(params)
    d = truth.time_terminated_simulation_data(
        end, items=8, random_state=20261121
    )
    # The eight systems' failures counted together in 20 periods, as one
    # process with eight times the faults (or, for Musa-Okumoto, eight
    # times lambda0 and an eighth of theta).
    edges = np.linspace(0.0, end, 21)
    counts = np.histogram(d.x[d.c == 0], edges)[0]
    model = fitter.fit(
        np.column_stack([edges[:-1], edges[1:]]), c=[2] * 20, n=counts
    )
    pooled = np.array(params)
    if name == "MusaOkumoto":
        pooled = pooled * [8.0, 1 / 8.0]
    else:
        pooled[0] *= 8.0
    z = (model.params - pooled) / model.standard_errors()
    print(name, model.params, z)
    assert np.all(np.abs(z) < 3.0), z


@pytest.mark.parametrize("scale", [1e-3, 1e3])
@pytest.mark.parametrize("name", sorted(RECOVERY))
def test_fit_does_not_depend_on_the_unit_of_time(name, scale):
    fitter = PARAMS[name][0]
    params, end = RECOVERY[name]
    d = fitter.from_params(params).time_terminated_simulation_data(
        end, items=4, random_state=7
    )
    base = fitter.fit(d.x, d.i, d.c, d.n)
    scaled = fitter.fit(d.x * scale, d.i, d.c, d.n)
    # Each rate (b, lambda0) is per unit of time; the others have none.
    rates = {"b": 1.0, "lambda0": 1.0}
    back = [
        v * scale ** rates.get(p, 0.0)
        for p, v in zip(fitter.parameter_names, scaled.params)
    ]
    _same_fit(base, back)
    np.testing.assert_allclose(
        scaled.cif(np.array([10.0, 50.0]) * scale),
        base.cif([10.0, 50.0]),
        rtol=1e-4,
    )
    # and so do the bounds
    np.testing.assert_allclose(
        scaled.cif_cb(50.0 * scale), base.cif_cb(50.0), rtol=1e-3
    )


@pytest.mark.parametrize("name", sorted(RECOVERY))
def test_bounds_and_save_and_load(name):
    fitter = PARAMS[name][0]
    params, end = RECOVERY[name]
    d = fitter.from_params(params).time_terminated_simulation_data(
        end, items=4, random_state=7
    )
    model = fitter.fit(d.x, d.i, d.c, d.n)
    t = np.array([20.0, 100.0, 200.0])
    for point, bound in (
        ("cif", "cif_cb"),
        ("iif", "iif_cb"),
        ("mtbf", "mtbf_cb"),
    ):
        value = getattr(model, point)(t)
        cb = getattr(model, bound)(t)
        assert np.all(cb[:, 0] < value) and np.all(value < cb[:, 1])
    for p, value in zip(model.parameter_names, model.params):
        lo, hi = model.param_cb(p)
        assert lo < value < hi
    back = ParametricRecurrenceModel.from_dict(model.to_dict())
    assert back.dist is fitter
    np.testing.assert_array_equal(back.params, model.params)
    np.testing.assert_allclose(back.cif_cb(t), model.cif_cb(t), rtol=1e-12)
    assert sp.from_json(model.to_json()).dist is fitter
    assert type(model.trend_test().statistic) is float


# ---------------------------------------------------------------------------
# Data that show no growth
# ---------------------------------------------------------------------------
def test_mean_times_of_the_limits():
    data = sp.handle_xicn(
        np.array([[0.0, 2.0], [2.0, 6.0]]),
        c=[2, 2],
        n=[3, 1],
        as_recurrent_data=True,
    )
    s = data.split_for_nhpp_likelihood()
    # p = 1: the periods' midpoints, 1 and 4, and the midpoint of (0, 6]
    assert limit_centroids(s, 1) == pytest.approx((1.75, 3.0))
    # p = 2: (2 / 3) (r^3 - l^3) / (r^2 - l^2) per period, and of (0, 6]
    means = (2 / 3) * np.array([8.0 / 4.0, 208.0 / 32.0])
    assert limit_centroids(s, 2) == pytest.approx(
        ((3 * means[0] + means[1]) / 4, 4.0)
    )


@pytest.mark.parametrize("fitter", [GoelOkumoto, MusaOkumoto])
def test_constant_rate_data_are_refused(fitter):
    d = HPP.from_params([0.5]).time_terminated_simulation_data(
        100.0, random_state=1
    )
    assert d.x[d.c == 0].mean() > 50.0
    with pytest.raises(ValueError) as error:
        fitter.fit(d.x, c=d.c)
    message = str(error.value)
    assert "show no reliability growth" in message
    assert "50.3325, is not earlier than the 50 " in message
    assert "a constant rate" in message
    assert "Fit HPP" in message and "CrowAMSAA" in message
    lost = "theta" if fitter is MusaOkumoto else "number of faults a"
    assert lost + " has no estimate" in message
    # and the search itself, were it run, would not beat the limit
    with pytest.raises(ValueError, match="show no reliability growth"):
        fitter.fit(d.x, c=d.c, how="MSE")


def test_the_mean_time_rule_is_where_the_likelihood_stops_having_a_maximum():
    # Failure times to T = 10 with their mean just below and just above
    # the limit's 5: below, the profile likelihood in b has a maximum
    # inside; above, it rises all the way to b = 0 (a constant rate).
    T = 10.0
    for shift, has_maximum in ((-0.05, True), (0.05, False)):
        t = np.array([1.0, 3.0, 5.0, 7.0, 9.0]) + shift
        data = sp.handle_xicn(
            np.append(t, T), c=[0] * 5 + [1], as_recurrent_data=True
        )
        negll = GoelOkumoto.create_negll_func(data)

        def profile(b):
            return -negll(np.array([5 / -np.expm1(-b * T), b]))

        b = np.exp(np.linspace(np.log(1e-6), np.log(1.0), 400))
        values = np.array([profile(v) for v in b])
        hpp = 5 * np.log(5 / T) - 5
        if has_maximum:
            assert values.max() > hpp and 0 < values.argmax() < 399
            assert GoelOkumoto.fit(np.append(t, T), c=[0] * 5 + [1])
        else:
            assert np.all(np.diff(values) < 0) and values.max() < hpp
            with pytest.raises(ValueError, match="no reliability growth"):
                GoelOkumoto.fit(np.append(t, T), c=[0] * 5 + [1])


def test_yamada_refuses_a_rate_still_rising():
    # A constant rate is growth enough for the delayed S-shaped curve (it
    # is past its peak); an intensity rising faster than in proportion to
    # time is not.
    flat = HPP.from_params([0.5]).time_terminated_simulation_data(
        100.0, random_state=1
    )
    assert YamadaDelayedS.fit(flat.x, c=flat.c).maximum == "verified"
    rising = CrowAMSAA.from_params(
        [10.0, 3.0]
    ).time_terminated_simulation_data(40.0, random_state=1)
    with pytest.raises(ValueError) as error:
        YamadaDelayedS.fit(rising.x, c=rising.c)
    assert "an intensity that rises in proportion to time" in str(error.value)
    assert "Yamada delayed S-shaped" in str(error.value)


def test_inflection_s_refuses_a_rate_that_rises_ever_faster():
    # 40 failures at the evenly spread quantiles of an intensity that
    # rises faster than exponentially, Lambda(t) = exp((t / 20)**2) - 1:
    # the best inflection S-shaped curve is its exponential limit.
    count = np.arange(1.0, 41.0)
    t = 20.0 * np.sqrt(np.log1p(count - 0.5))
    end = 20.0 * np.sqrt(np.log1p(40.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError) as error:
            InflectionS.fit(np.append(t, end), c=[0] * 40 + [1])
    message = str(error.value)
    assert "the fitted likelihood is no higher than that of the" in message
    assert "a constant or an exponentially rising rate" in message
    assert "Fit HPP" in message


def test_no_events_is_said_before_no_growth():
    with pytest.raises(ValueError, match="no events"):
        GoelOkumoto.fit([10.0], c=[1])


def test_one_period_of_counts_cannot_show_growth():
    with pytest.raises(ValueError, match="no reliability growth"):
        GoelOkumoto.fit(np.array([[0.0, 10.0]]), c=[2], n=[7])
