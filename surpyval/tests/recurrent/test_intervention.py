"""The intervention test: did the rate of failures change at an
intervention, and was the evidence enough?

The exact HPP test is checked against closed forms (Clopper-Pearson, the
binomial tails), against brute-force enumeration of the stratified
distribution, and for its exact size and coverage by summing over every
outcome (no simulation needed: the test is exact). The simulation studies
of size, power and coverage, the power law's included, are in
``calibration/test_size_power.py`` and
``calibration/test_coverage_recurrent.py``.
"""

import itertools
import json
import math

import numpy as np
import pytest
from scipy import stats

from surpyval.recurrent import InterventionTestResult, intervention_test
from surpyval.recurrent.intervention import _exact_bound, _Strata
from surpyval.test_planning import (
    rate_ratio_detectable,
    rate_ratio_exposure,
    rate_ratio_power,
)
from surpyval.test_planning.rate_ratio import (
    critical_counts,
    expected_power,
    limit_power,
)

PUMP = [3, 8, 15, 19, 26, 30, 37, 44, 52, 77, 95]

# A fleet of three pumps, changed at different times.
FLEET_X = [5, 11, 18, 26, 41, 4, 9, 13, 17, 22, 30, 52, 7, 33]
FLEET_I = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3]
FLEET_T = {1: 60, 2: 70, 3: 50}
FLEET_TAU = {1: 30, 2: 25, 3: 20}


def _events(n_before, n_after, tau=50.0, end=100.0):
    """``n_before`` failures spread over (0, tau] and ``n_after`` over
    (tau, end]."""
    before = np.linspace(0, tau, n_before + 2)[1:-1]
    after = np.linspace(tau, end, n_after + 2)[1:-1]
    return list(before) + list(after)


# --- the exact HPP test: closed forms ---------------------------------------


@pytest.mark.parametrize("n_before, n_after", [(8, 3), (12, 0), (2, 9)])
@pytest.mark.parametrize("tau", [30.0, 50.0, 80.0])
def test_single_system_matches_clopper_pearson(n_before, n_after, tau):
    # One system: given N failures, those after are Bin(N, E_a / (E_b +
    # E_a)) under no change; the interval is Clopper-Pearson's,
    # transformed by rho = pi / (1 - pi) * E_b / E_a.
    x = _events(n_before, n_after, tau)
    result = intervention_test(x, T=100.0, intervention=tau)
    n = n_before + n_after
    e_b, e_a = tau, 100.0 - tau
    p0 = e_a / (e_b + e_a)
    lower = stats.binom.cdf(n_after, n, p0)
    upper = stats.binom.sf(n_after - 1, n, p0)
    assert result.p_value == pytest.approx(min(1, 2 * min(lower, upper)))
    pi_lo = stats.beta.ppf(0.025, n_after, n - n_after + 1) if n_after else 0
    pi_hi = stats.beta.ppf(0.975, n_after + 1, n - n_after)
    to_ratio = lambda p: p / (1 - p) * e_b / e_a  # noqa: E731
    assert result.ratio_cb[0] == pytest.approx(to_ratio(pi_lo), abs=1e-9)
    assert result.ratio_cb[1] == pytest.approx(to_ratio(pi_hi), rel=1e-7)
    assert result.ratio == pytest.approx(
        (n_after / e_a) / (n_before / e_b), abs=1e-12
    )
    assert result.expected_after == pytest.approx(n * p0)


def test_one_sided_tests_and_bounds():
    x = PUMP
    down = intervention_test(x, T=100, intervention=50, alternative="decrease")
    up = intervention_test(x, T=100, intervention=50, alternative="increase")
    assert down.p_value == pytest.approx(stats.binom.cdf(3, 11, 0.5))
    assert up.p_value == pytest.approx(stats.binom.sf(2, 11, 0.5))
    assert down.ratio_cb[0] == 0 and math.isinf(up.ratio_cb[1])
    # The one-sided 95% bound is the two-sided 90% bound's end.
    ninety = intervention_test(x, T=100, intervention=50, alpha_ci=0.1)
    assert down.ratio_cb[1] == pytest.approx(ninety.ratio_cb[1])
    assert up.ratio_cb[0] == pytest.approx(ninety.ratio_cb[0])


def test_failure_at_the_intervention_counts_before():
    result = intervention_test([10, 20, 50, 70], T=100, intervention=50)
    assert (result.events_before, result.events_after) == (3, 1)


def test_single_system_per_system_equals_common():
    a = intervention_test(PUMP, T=100, intervention=50)
    b = intervention_test(PUMP, T=100, intervention=50, baseline="common")
    assert a.p_value == pytest.approx(b.p_value)
    np.testing.assert_allclose(a.ratio_cb, b.ratio_cb)


# --- the stratified (per-system) distribution ------------------------------


def test_stratified_distribution_matches_enumeration():
    totals, weights = [3, 4, 2], np.array([0.5, 1.2, 2.0])
    strata = _Strata(np.array(totals), weights, midp=False)
    rho = 0.7
    p = rho * weights / (1 + rho * weights)
    pmf = np.zeros(sum(totals) + 1)
    for counts in itertools.product(*(range(n + 1) for n in totals)):
        pmf[sum(counts)] += np.prod(
            [stats.binom.pmf(k, n, q) for k, n, q in zip(counts, totals, p)]
        )
    for s in range(sum(totals) + 1):
        lower, upper = strata.tails(s, rho)
        assert lower == pytest.approx(pmf[: s + 1].sum())
        assert upper == pytest.approx(pmf[s:].sum())


def test_conditional_estimate_solves_its_score():
    strata = _Strata(np.array([5, 3, 6]), np.array([0.4, 1.0, 2.5]), False)
    rho = strata.estimate(7)
    p = rho * strata.weights / (1 + rho * strata.weights)
    assert np.sum(strata.totals * p) == pytest.approx(7)


@pytest.mark.parametrize("method", ["exact", "mid-p"])
@pytest.mark.parametrize("alternative", ["two-sided", "decrease", "increase"])
def test_test_and_interval_agree(method, alternative):
    # Inverting the test gives the interval: p < alpha exactly when the
    # interval excludes 1, outcome by outcome.
    for n_before, n_after in [(8, 1), (8, 2), (12, 3), (5, 11), (9, 8)]:
        result = intervention_test(
            _events(n_before, n_after),
            T=100,
            intervention=50,
            method=method,
            alternative=alternative,
        )
        lo, hi = result.ratio_cb
        excluded = (lo > 1) or (hi < 1)
        assert (result.p_value < 0.05) == excluded, (n_before, n_after)


def _poisson_outcomes(mean, tail=1e-10):
    k = np.arange(0, stats.poisson.isf(tail, mean) + 1)
    return k, stats.poisson.pmf(k, mean)


@pytest.mark.parametrize("method", ["exact", "mid-p"])
def test_exact_size_by_enumeration(method):
    # The size of the test, summed over every outcome rather than
    # simulated: at most alpha for the exact test; the mid-p test is near
    # alpha but not bounded by it.
    e_b, e_a, rate = 40.0, 25.0, 0.3
    midp = method == "mid-p"
    k_b, w_b = _poisson_outcomes(rate * e_b)
    k_a, w_a = _poisson_outcomes(rate * e_a)
    size = 0.0
    for nb, pb in zip(k_b, w_b):
        for na, pa in zip(k_a, w_a):
            if nb + na == 0:
                continue
            strata = _Strata(np.array([nb + na]), np.array([e_a / e_b]), midp)
            lower, upper = strata.tails(int(na), 1.0)
            size += pb * pa * (min(1, 2 * min(lower, upper)) < 0.05)
    if method == "exact":
        assert 0.02 < size <= 0.05  # 0.028: exact tests are conservative
    else:
        assert abs(size - 0.05) < 0.01


def test_exact_coverage_by_enumeration():
    # The exact interval covers the true ratio at least 95% of the time,
    # summed over every outcome.
    e_b, e_a, rate, ratio = 30.0, 30.0, 0.25, 0.5
    k_b, w_b = _poisson_outcomes(rate * e_b, 1e-7)
    k_a, w_a = _poisson_outcomes(rate * ratio * e_a, 1e-7)
    covered = 0.0
    for nb, pb in zip(k_b, w_b):
        for na, pa in zip(k_a, w_a):
            if nb + na == 0:
                covered += pb * pa  # [0, inf] covers
                continue
            strata = _Strata(np.array([nb + na]), np.array([e_a / e_b]), False)
            lo = _exact_bound(strata, int(na), 0.025, "lower", 0.0)
            hi = _exact_bound(strata, int(na), 0.025, "upper", 0.0)
            covered += pb * pa * (lo <= ratio <= hi)
    assert covered >= 0.95 - 1e-6
    assert covered < 0.995


# --- fleets ----------------------------------------------------------------


def test_fleet_per_system_and_common():
    per = intervention_test(
        FLEET_X, FLEET_I, T=FLEET_T, intervention=FLEET_TAU
    )
    common = intervention_test(
        FLEET_X, FLEET_I, T=FLEET_T, intervention=FLEET_TAU, baseline="common"
    )
    assert (per.events_before, per.events_after) == (10, 4)
    assert per.exposure_before == 30 + 25 + 20
    assert per.exposure_after == 30 + 45 + 30
    # The pooled test is the single-stratum test on the totals.
    p0 = 105 / 180
    lower = stats.binom.cdf(4, 14, p0)
    assert common.p_value == pytest.approx(2 * lower)
    assert common.ratio == pytest.approx((4 / 105) / (10 / 75))
    assert per.n_systems == per.n_systems_compared == 3


def test_per_system_is_not_confounded_by_rate_differences():
    # System "a" fails often and is mostly observed before the change;
    # system "b" fails rarely and is mostly observed after it. Neither
    # rate changed at all, but pooling them makes the rate look as if it
    # fell. The per-system test compares each system with itself.
    a = np.linspace(0, 100, 42)[1:-1]  # 40 failures in (0, 100]
    b = np.array([40.0, 60.0, 100.0, 130.0, 160.0, 190.0])
    x = np.concatenate([a, b])
    i = ["a"] * a.size + ["b"] * b.size
    T = {"a": 100.0, "b": 200.0}
    tau = {"a": 80.0, "b": 20.0}
    pooled = intervention_test(x, i, T=T, intervention=tau, baseline="common")
    per = intervention_test(x, i, T=T, intervention=tau)
    assert pooled.verdict == "decrease" and pooled.p_value < 1e-4
    assert per.change == "none" and per.p_value > 0.5


def test_systems_seen_on_one_side_only():
    # System 3 was never changed (inf), system 4 was changed before its
    # observation began: per-system, neither informs the ratio, and the
    # result says so; pooled, both count.
    x = FLEET_X + [6, 19, 31, 12]
    i = FLEET_I + [4, 4, 4, 5]
    T = {**FLEET_T, 4: 40, 5: 40}
    tau = {**FLEET_TAU, 4: np.inf, 5: -1.0}
    per = intervention_test(x, i, T=T, intervention=tau)
    assert per.n_systems == 5 and per.n_systems_compared == 3
    assert any("not compared" in note for note in per.notes)
    alone = intervention_test(
        FLEET_X, FLEET_I, T=FLEET_T, intervention=FLEET_TAU
    )
    assert per.p_value == pytest.approx(alone.p_value)
    common = intervention_test(x, i, T=T, intervention=tau, baseline="common")
    assert common.events_before == 13 and common.events_after == 5


def test_systems_without_failures_count_their_exposure():
    with_quiet = intervention_test(
        FLEET_X,
        FLEET_I,
        T={**FLEET_T, 9: 80},
        intervention={**FLEET_TAU, 9: 40},
        baseline="common",
    )
    assert with_quiet.exposure_before == 75 + 40
    assert with_quiet.exposure_after == 105 + 40


# --- the forms of the data ---------------------------------------------------


def test_c_form_equals_T_form():
    x, i, c = [], [], []
    for q, end in FLEET_T.items():
        times = [t for t, k in zip(FLEET_X, FLEET_I) if k == q]
        x += times + [end]
        i += [q] * (len(times) + 1)
        c += [0] * len(times) + [1]
    a = intervention_test(x, i, c=c, intervention=FLEET_TAU)
    b = intervention_test(FLEET_X, FLEET_I, T=FLEET_T, intervention=FLEET_TAU)
    assert a.to_dict() == b.to_dict()


def test_delayed_entry_and_windows():
    # A system watched over (0, 10] and (20, 40], changed at 30: before is
    # (0, 10] and (20, 30], after (30, 40].
    x = [2, 7, 22, 28, 35]
    gapped = intervention_test(
        x, windows={1: [(0, 10), (20, 40)]}, intervention=30
    )
    assert gapped.exposure_before == 20 and gapped.exposure_after == 10
    assert (gapped.events_before, gapped.events_after) == (4, 1)
    # Delayed entry: observed over (20, 40].
    late = intervention_test([22, 28, 35], T=40, tl=20, intervention=30)
    assert late.exposure_before == 10 and late.exposure_after == 10


def test_order_and_units_do_not_matter():
    order = np.random.default_rng(0).permutation(len(FLEET_X))
    x = np.asarray(FLEET_X, float)
    i = np.asarray(FLEET_I)
    base = intervention_test(x, i, T=FLEET_T, intervention=FLEET_TAU)
    shuffled = intervention_test(
        x[order], i[order], T=FLEET_T, intervention=FLEET_TAU
    )
    assert base.to_dict() == shuffled.to_dict()
    for model in ("hpp", "power-law"):
        a = intervention_test(
            x, i, T=FLEET_T, intervention=FLEET_TAU, model=model
        )
        b = intervention_test(
            x * 1000,
            i,
            T={q: t * 1000 for q, t in FLEET_T.items()},
            intervention={q: t * 1000 for q, t in FLEET_TAU.items()},
            model=model,
        )
        assert a.ratio == pytest.approx(b.ratio, rel=1e-6)
        assert a.p_value == pytest.approx(b.p_value, rel=1e-6)
        np.testing.assert_allclose(a.ratio_cb, b.ratio_cb, rtol=1e-5)
        assert b.exposure_needed == pytest.approx(
            1000 * a.exposure_needed, rel=1e-6
        ) or not math.isfinite(a.exposure_needed)


# --- refusals and untestable data --------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"intervention": 50}, "observation in one way"),
        ({"T": 100, "c": [0] * 11, "intervention": 50}, "one way"),
        ({"T": 100, "intervention": float("nan")}, "inf for never"),
        ({"T": 40, "intervention": 20}, "outside its observation"),
        ({"T": 100, "intervention": 50, "model": "weibull"}, "'model'"),
        ({"T": 100, "intervention": 50, "baseline": "pooled"}, "baseline"),
        ({"T": 100, "intervention": 50, "method": "wald"}, "'method'"),
        (
            {
                "T": 100,
                "intervention": 50,
                "model": "power-law",
                "method": "exact",
            },
            "likelihood ratio",
        ),
        ({"T": 100, "intervention": 50, "target_ratio": 1}, "target_ratio"),
        (
            {
                "T": 100,
                "intervention": 50,
                "alternative": "decrease",
                "target_ratio": 2,
            },
            "below 1",
        ),
        ({"T": 100, "intervention": 50, "power": 80}, "'power'"),
        ({"T": 100, "intervention": 50, "alpha_ci": 0.95 * 2}, "alpha_ci"),
        (
            {"T": 100, "tl": -10, "intervention": 50, "model": "power-law"},
            "age of 0",
        ),
    ],
)
def test_invalid_input_raises(kwargs, match):
    with pytest.raises(ValueError, match=match):
        intervention_test(PUMP, **kwargs)


def test_every_system_needs_its_end():
    with pytest.raises(ValueError, match="c=1 row"):
        intervention_test(
            [5, 9, 20, 4, 30],
            [1, 1, 1, 2, 2],
            c=[0, 0, 1, 0, 0],
            intervention=10,
        )


def test_dates_are_refused():
    times = np.array([3, 8, 15], dtype="timedelta64[D]")
    with pytest.raises(ValueError, match="x"):
        intervention_test(times, T=100, intervention=50)


@pytest.mark.parametrize(
    "x, T, tau, match",
    [
        ([5, 12], 60, 70, "nothing to compare"),
        ([5, 12], 60, -5, "no earlier rate"),
        ([], 60, 30, "no failure in the periods"),
    ],
)
@pytest.mark.parametrize("model", ["hpp", "power-law"])
def test_untestable_data_give_a_result_that_says_why(x, T, tau, match, model):
    result = intervention_test(x, T=T, intervention=tau, model=model)
    assert result.verdict == "untestable"
    assert math.isnan(result.p_value) and math.isnan(result.ratio)
    if model == "hpp" or "failure in" not in match:
        assert match in result.notes[0]


def test_no_failure_before_cannot_show_a_reduction():
    result = intervention_test([45, 50, 55], T=60, intervention=40)
    assert result.verdict == "untestable"
    assert math.isinf(result.ratio)
    assert "no failure rate to show a reduction" in result.notes[0]
    # ...but a large enough increase is still found.
    worse = intervention_test(
        [41, 42, 44, 45, 47, 50, 52, 55, 57, 59], T=60, intervention=40
    )
    assert worse.verdict == "increase"


def test_no_failure_after():
    x = _events(12, 0, tau=50)
    result = intervention_test(x, T=100, intervention=50)
    assert result.ratio == 0 and result.ratio_cb[0] == 0
    assert result.verdict == "decrease"
    assert any("No failure since" in note for note in result.notes)
    # Too little exposure after: inconclusive, and says how much more
    # failure-free exposure would show a decrease.
    short = intervention_test(_events(12, 0, 90), T=100, intervention=90)
    assert short.verdict == "inconclusive"
    assert short.zero_failure_exposure == pytest.approx(
        90 * (0.025 ** (-1 / 12) - 1)
    )
    assert any("no further failure" in note for note in short.notes)
    longer = intervention_test(
        _events(12, 0, 90),
        T=90 + short.zero_failure_exposure * 1.001,
        intervention=90,
    )
    assert longer.verdict == "decrease"


# --- the evidence ------------------------------------------------------------


def test_evidence_fields():
    result = intervention_test(PUMP, T=100, intervention=50)
    rate = 8 / 50
    assert result.power == pytest.approx(rate_ratio_power(0.5, rate, 50, 50))
    assert result.detectable_ratio == pytest.approx(
        rate_ratio_detectable(rate, 50, 50)
    )
    assert math.isinf(result.exposure_needed)  # 8 failures before: never
    assert result.verdict == "inconclusive" and not result.sufficient
    assert any("No exposure after" in note for note in result.notes)


def test_enough_evidence_of_no_change():
    # 60 failures in 100 before, 58 in 100 after: no change, and plenty of
    # power to have seen a halving.
    result = intervention_test(_events(60, 58), T=100, intervention=50)
    assert result.verdict == "no change detected" and result.sufficient
    assert result.power > 0.85
    assert any("smallest change" in note for note in result.notes)


def test_more_exposure_reaches_the_power():
    result = intervention_test(_events(30, 5, 50, 60), T=60, intervention=50)
    assert result.verdict == "inconclusive"
    needed = result.exposure_needed
    assert result.additional_exposure == pytest.approx(needed - 10)
    assert rate_ratio_power(0.5, 30 / 50, 50, needed) >= 0.8
    assert result.events_needed == pytest.approx(0.5 * 30 / 50 * needed)
    assert any("more), with about" in note for note in result.notes)


def test_trend_within_the_periods_is_flagged():
    # Wear-out: failures bunch towards the end of each period.
    rng = np.random.default_rng(3)
    x = np.sort(100 * rng.uniform(size=60) ** (1 / 3))
    hpp = intervention_test(x, T=100, intervention=50)
    assert hpp.trend_p_value < 0.05
    assert any("model='power-law'" in note for note in hpp.notes)
    pl = intervention_test(x, T=100, intervention=50, model="power-law")
    assert pl.shape == pytest.approx(3, rel=0.35)
    assert pl.trend_p_value < 0.05
    # No change happened: the power law does not find one, while the HPP
    # comparison mistakes the wear-out for a worsening.
    assert pl.change == "none"
    assert hpp.verdict == "increase"


# --- the power law -----------------------------------------------------------


def test_power_law_with_a_known_answer():
    # Shape 1 (an HPP) and a common scale: the profile likelihood of the
    # ratio is that of two Poisson rates, so the estimate is the ratio of
    # the average rates.
    x = _events(20, 8)
    result = intervention_test(
        x, T=100, intervention=50, model="power-law", baseline="common"
    )
    assert result.shape == pytest.approx(1.0, abs=0.25)
    assert result.method == "likelihood-ratio"
    assert result.ratio < 1 and result.ratio_cb[0] < result.ratio
    assert result.ratio_cb[1] > result.ratio


def test_power_law_without_failures_before_is_the_reciprocal():
    x = [45, 50, 52, 55, 58]
    after = intervention_test(x, T=60, intervention=40, model="power-law")
    assert math.isinf(after.ratio)
    # The same data with the periods' roles swapped (failures before 20,
    # none after) give the reciprocal ratio.
    mirror = [60 - t for t in x]
    before = intervention_test(
        mirror, T=60, intervention=20, model="power-law"
    )
    assert before.ratio == 0


def test_power_law_profile_interval_holds_its_level():
    # The likelihood-ratio statistic at the bound is the chi-squared
    # quantile.
    from surpyval.recurrent.intervention import _PowerLaw, _split, _systems

    systems = _systems(FLEET_X, FLEET_I, FLEET_T, None, None, None, FLEET_TAU)
    split = _split(systems)
    model = _PowerLaw(split, np.arange(3))
    result = intervention_test(
        FLEET_X,
        FLEET_I,
        T=FLEET_T,
        intervention=FLEET_TAU,
        model="power-law",
    )
    top = model.profile(result.ratio)
    for bound in result.ratio_cb:
        assert 2 * (top - model.profile(bound)) == pytest.approx(
            stats.chi2.isf(0.05, 1), abs=1e-5
        )
    assert 2 * (top - model.profile(1.0)) == pytest.approx(
        result.statistic, abs=1e-6
    )


# --- the result --------------------------------------------------------------


def test_result_round_trips_through_strict_json():
    result = intervention_test(
        FLEET_X,
        FLEET_I,
        T=FLEET_T,
        intervention=FLEET_TAU,
        model="power-law",
    )
    text = json.dumps(result.to_dict(), allow_nan=False)
    again = InterventionTestResult.from_dict(json.loads(text))
    for name, value in vars(result).items():
        other = getattr(again, name)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, other)
        elif isinstance(value, float) and math.isnan(value):
            assert math.isnan(other)
        else:
            assert value == other, name
    with pytest.raises(ValueError, match="InterventionTestResult"):
        InterventionTestResult.from_dict({"result": "Forecast"})


def test_repr_and_properties():
    result = intervention_test(PUMP, T=100, intervention=50)
    text = repr(result)
    assert "Rate ratio" in text and "Verdict          : inconclusive" in text
    assert result.reduction == pytest.approx(1 - result.ratio)
    np.testing.assert_allclose(result.reduction_cb, 1 - result.ratio_cb[::-1])
    assert result.rate_before == pytest.approx(8 / 50)
    assert result.rate_after == pytest.approx(3 / 50)


# --- the planning functions --------------------------------------------------


def test_power_is_the_exact_rejection_probability():
    # The power function against a brute-force sum, over both Poisson
    # counts, of the test's own rejections (no simulation: both exact).
    e_b, e_a, rate, ratio = 40.0, 30.0, 0.4, 0.5
    for method in ("exact", "mid-p"):
        midp = method == "mid-p"
        k_b, w_b = _poisson_outcomes(rate * e_b, 1e-12)
        k_a, w_a = _poisson_outcomes(rate * ratio * e_a, 1e-12)
        rejected = 0.0
        for nb, pb in zip(k_b, w_b):
            for na, pa in zip(k_a, w_a):
                strata = _Strata(
                    np.array([nb + na]), np.array([e_a / e_b]), midp
                )
                lower = strata.tails(int(na), 1.0)[0]
                rejected += pb * pa * (lower < 0.025)
        assert rate_ratio_power(
            ratio, rate, e_b, e_a, method=method
        ) == pytest.approx(rejected, abs=1e-9)


def test_critical_counts():
    n = np.arange(0, 40)
    for midp in (False, True):
        c = critical_counts(n, 0.4, 0.025, midp)
        for total, crit in zip(n, c):
            p = stats.binom.cdf(np.arange(total + 1), total, 0.4)
            if midp:
                p = p - 0.5 * stats.binom.pmf(np.arange(total + 1), total, 0.4)
            assert crit == (np.flatnonzero(p < 0.025).max(initial=-1))


def test_power_rises_to_its_limit():
    # More exposure after the change cannot beat the uncertainty in the
    # rate before it.
    for ratio in (0.5, 2.0):
        limit = limit_power(12.0, ratio, 0.025, False)
        far = expected_power(12.0, 12.0 * 1e5, ratio, 0.025, False)
        assert far == pytest.approx(limit, abs=1e-6)
    assert rate_ratio_exposure(0.5, 12 / 1e4, 1e4) == math.inf


def test_detectable_and_exposure_hit_the_power():
    rate, e_b, e_a = 30 / 1e4, 1e4, 2e4
    ratio = rate_ratio_detectable(rate, e_b, e_a)
    assert rate_ratio_power(ratio, rate, e_b, e_a) == pytest.approx(
        0.8, abs=2e-3
    )
    assert rate_ratio_power(ratio, rate, e_b, e_a) >= 0.8
    worse = rate_ratio_detectable(rate, e_b, e_a, direction="increase")
    assert worse > 1
    assert rate_ratio_power(worse, rate, e_b, e_a) >= 0.8
    needed = rate_ratio_exposure(0.5, rate, e_b)
    assert rate_ratio_power(0.5, rate, e_b, needed) >= 0.8
    assert rate_ratio_power(0.5, rate, e_b, 0.9 * needed) < 0.8


def test_planning_takes_arrays():
    powers = rate_ratio_power([0.3, 0.5, 0.7], 30 / 1e4, 1e4, 2e4)
    assert powers.shape == (3,) and np.all(np.diff(powers) < 0)


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda: rate_ratio_power(1.0, 0.1, 10, 10), "differ from 1"),
        (lambda: rate_ratio_power(0.5, -0.1, 10, 10), "rate"),
        (lambda: rate_ratio_power(0.5, 0.1, 0, 10), "exposure_before"),
        (lambda: rate_ratio_power(0.5, 0.1, 10, 10, method="z"), "method"),
        (
            lambda: rate_ratio_power(0.5, 0.1, 10, 10, alternative="less"),
            "alternative",
        ),
        (lambda: rate_ratio_detectable(0.1, 10, 10, 1.5), "power"),
        (
            lambda: rate_ratio_detectable(0.1, 10, 10, direction="down"),
            "direction",
        ),
        (lambda: rate_ratio_exposure(0.5, 0.1, 10, 0), "power"),
    ],
)
def test_planning_refuses_bad_arguments(call, match):
    with pytest.raises(ValueError, match=match):
        call()
