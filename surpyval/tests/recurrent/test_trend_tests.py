import numpy as np
import pytest
from scipy.stats import chi2, norm

from surpyval.recurrent import laplace, mil_hdbk_189c
from surpyval.recurrent.tests import TrendTestResult


def test_exported_from_recurrent():
    # Both the dotted module path used in the docs and the package-level
    # re-export should reach the same functions.
    from surpyval.recurrent import tests as trend_tests

    assert trend_tests.laplace is laplace
    assert trend_tests.mil_hdbk_189c is mil_hdbk_189c


# ----------------------------------------------------------------------------
# Laplace test
# ----------------------------------------------------------------------------


def test_laplace_matches_closed_form_single_system_time_truncated():
    # For a single time-truncated system the statistic is the textbook
    # U = (sum t - n T / 2) / (T sqrt(n / 12)).
    x = np.array([10.0, 19.0, 27.0, 34.0, 40.0, 45.0, 49.0, 52.0, 54.0])
    T = 60.0
    n = x.size
    expected = (x.sum() - n * T / 2.0) / (T * np.sqrt(n / 12.0))
    res = laplace(x, T=T)
    assert res.statistic == pytest.approx(expected)
    assert res.n_events == n
    assert res.n_systems == 1


def _power_law_times(T=100.0, n=20, beta=2.5):
    # Deterministic power-law (Crow-AMSAA) event times: evenly spaced in the
    # mean value function N(t), so t_k = T (k/(n+1))**(1/beta). beta > 1 gives
    # an increasing intensity, beta < 1 a decreasing one.
    k = np.arange(1, n + 1)
    return T * (k / (n + 1)) ** (1.0 / beta)


def test_laplace_increasing_intensity_positive_statistic():
    # Power-law with beta > 1: failures speed up -> U > 0, increasing.
    x = _power_law_times(beta=2.5)
    res = laplace(x, T=100.0, alternative="increasing")
    assert res.statistic > 0
    assert res.trend == "increasing"
    assert res.p_value < 0.05


def test_laplace_decreasing_intensity_negative_statistic():
    # Inter-arrival times grow (reliability growth) -> U < 0, decreasing.
    x = np.cumsum(np.arange(1, 11, dtype=float))
    res = laplace(x, T=x[-1] + 11.0, alternative="decreasing")
    assert res.statistic < 0
    assert res.trend == "decreasing"
    assert res.p_value < 0.05


def test_laplace_hpp_data_no_significant_trend():
    # Genuine HPP data should not raise a significant trend on average.
    rng = np.random.default_rng(0)
    T = 1000.0
    rejections = 0
    trials = 200
    for _ in range(trials):
        n = rng.poisson(50)
        x = np.sort(rng.uniform(0, T, size=max(n, 2)))
        if laplace(x, T=T).p_value < 0.05:
            rejections += 1
    # Roughly the nominal 5% level; allow generous slack for randomness.
    assert rejections / trials < 0.15


def test_laplace_pvalue_directions_consistent():
    x = np.array([10.0, 19.0, 27.0, 34.0, 40.0, 45.0, 49.0, 52.0, 54.0])
    u = laplace(x, T=60.0).statistic
    assert laplace(x, T=60.0, alternative="increasing").p_value == (
        pytest.approx(norm.sf(u))
    )
    assert laplace(x, T=60.0, alternative="decreasing").p_value == (
        pytest.approx(norm.cdf(u))
    )
    assert laplace(x, T=60.0, alternative="two-sided").p_value == (
        pytest.approx(2.0 * norm.sf(abs(u)))
    )


def test_laplace_failure_truncated_drops_last_event():
    # Without T the last event is the truncation point and is excluded.
    x = np.array([10.0, 19.0, 27.0, 34.0, 40.0, 45.0, 49.0, 52.0, 54.0])
    res = laplace(x)
    assert res.n_events == x.size - 1
    # Failure-truncation at the n-th event equals time-truncating the first
    # n-1 events at that event time.
    res_T = laplace(x[:-1], T=x[-1])
    assert res.statistic == pytest.approx(res_T.statistic)


def test_laplace_multiple_systems():
    # Two systems pooled; statistic uses the combined numerator/variance.
    x = np.array([5.0, 12.0, 18.0, 7.0, 15.0, 22.0])
    i = np.array([1, 1, 1, 2, 2, 2])
    T = {1: 25.0, 2: 30.0}
    num = (5 + 12 + 18 - 3 * 25 / 2) + (7 + 15 + 22 - 3 * 30 / 2)
    var = 3 * 25**2 / 12 + 3 * 30**2 / 12
    res = laplace(x, i, T)
    assert res.statistic == pytest.approx(num / np.sqrt(var))
    assert res.n_systems == 2
    assert res.n_events == 6


def test_laplace_scalar_and_array_T_equivalent():
    x = np.array([5.0, 12.0, 18.0, 7.0, 15.0, 20.0])
    i = np.array([1, 1, 1, 2, 2, 2])
    res_scalar = laplace(x, i, 25.0)
    res_array = laplace(x, i, [25.0, 25.0])
    res_dict = laplace(x, i, {1: 25.0, 2: 25.0})
    assert res_scalar.statistic == pytest.approx(res_array.statistic)
    assert res_scalar.statistic == pytest.approx(res_dict.statistic)


# ----------------------------------------------------------------------------
# MIL-HDBK-189C test
# ----------------------------------------------------------------------------


def test_mil_matches_closed_form_single_system():
    x = np.array([10.0, 19.0, 27.0, 34.0, 40.0, 45.0, 49.0, 52.0, 54.0])
    T = 60.0
    expected = 2.0 * np.sum(np.log(T / x))
    res = mil_hdbk_189c(x, T=T)
    assert res.statistic == pytest.approx(expected)
    assert res.dof == 2 * x.size


def test_mil_increasing_intensity_lower_tail():
    # Crow-AMSAA power-law with beta > 1 (deterioration). The statistic equals
    # 2N / beta_hat, so a small value -> increasing intensity.
    x = np.array([10.0, 19.0, 27.0, 34.0, 40.0, 45.0, 49.0, 52.0, 54.0])
    res = mil_hdbk_189c(x, T=60.0, alternative="increasing")
    # p is about 0.1: the statistic points up but no trend is concluded
    assert res.direction == "increasing" and res.trend == "none"
    assert res.statistic < res.dof
    assert res.p_value == pytest.approx(chi2.cdf(res.statistic, res.dof))


def test_mil_decreasing_intensity_upper_tail():
    x = np.cumsum(np.arange(1, 11, dtype=float))
    T = x[-1] + 11.0
    res = mil_hdbk_189c(x, T=T, alternative="decreasing")
    assert res.trend == "decreasing"
    assert res.statistic > res.dof
    assert res.p_value == pytest.approx(chi2.sf(res.statistic, res.dof))


def test_mil_failure_truncated_dof():
    x = np.array([10.0, 19.0, 27.0, 34.0, 40.0, 45.0, 49.0, 52.0, 54.0])
    res = mil_hdbk_189c(x)
    # Last event is the truncation point: 2(n-1) dof.
    assert res.dof == 2 * (x.size - 1)
    res_T = mil_hdbk_189c(x[:-1], T=x[-1])
    assert res.statistic == pytest.approx(res_T.statistic)
    assert res.dof == res_T.dof


def test_mil_multiple_systems_pooled():
    x = np.array([5.0, 12.0, 18.0, 7.0, 15.0, 22.0])
    i = np.array([1, 1, 1, 2, 2, 2])
    T = {1: 25.0, 2: 30.0}
    expected = 2.0 * (
        np.sum(np.log(25.0 / np.array([5.0, 12.0, 18.0])))
        + np.sum(np.log(30.0 / np.array([7.0, 15.0, 22.0])))
    )
    res = mil_hdbk_189c(x, i, T)
    assert res.statistic == pytest.approx(expected)
    assert res.dof == 12


def test_mil_hpp_data_no_significant_trend():
    rng = np.random.default_rng(1)
    T = 1000.0
    rejections = 0
    trials = 200
    for _ in range(trials):
        n = rng.poisson(50)
        x = np.sort(rng.uniform(0, T, size=max(n, 2)))
        if mil_hdbk_189c(x, T=T).p_value < 0.05:
            rejections += 1
    assert rejections / trials < 0.15


# ----------------------------------------------------------------------------
# Validation and result object
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("func", [laplace, mil_hdbk_189c])
def test_rejects_bad_alternative(func):
    with pytest.raises(ValueError):
        func([1.0, 2.0, 3.0], alternative="up")


@pytest.mark.parametrize("func", [laplace, mil_hdbk_189c])
def test_rejects_nonpositive_times(func):
    with pytest.raises(ValueError):
        func([0.0, 1.0, 2.0], T=3.0)
    with pytest.raises(ValueError):
        func([-1.0, 1.0, 2.0], T=3.0)


@pytest.mark.parametrize("func", [laplace, mil_hdbk_189c])
def test_rejects_events_after_T(func):
    with pytest.raises(ValueError):
        func([1.0, 2.0, 5.0], T=3.0)


@pytest.mark.parametrize("func", [laplace, mil_hdbk_189c])
def test_rejects_too_few_events(func):
    with pytest.raises(ValueError):
        func([5.0])  # single failure-truncated event -> 0 usable events
    with pytest.raises(ValueError):
        func([], T=10.0)


@pytest.mark.parametrize("func", [laplace, mil_hdbk_189c])
def test_rejects_mismatched_i(func):
    with pytest.raises(ValueError):
        func([1.0, 2.0, 3.0], i=[1, 1], T=5.0)


@pytest.mark.parametrize("func", [laplace, mil_hdbk_189c])
def test_rejects_missing_dict_window(func):
    with pytest.raises(ValueError):
        func([1.0, 2.0, 3.0], i=[1, 1, 2], T={1: 5.0})


@pytest.mark.parametrize("func", [laplace, mil_hdbk_189c])
def test_rejects_nonfinite(func):
    with pytest.raises(ValueError):
        func([1.0, np.nan, 3.0], T=5.0)


def test_result_repr_contains_fields():
    res = laplace([10.0, 19.0, 27.0, 34.0], T=40.0)
    text = repr(res)
    assert "Laplace Trend Test" in text
    assert "p-value" in text
    assert "Direction" in text and "Trend" in text
    assert isinstance(res, TrendTestResult)

    res_mil = mil_hdbk_189c([10.0, 19.0, 27.0, 34.0], T=40.0)
    assert "MIL-HDBK-189C Trend Test" in repr(res_mil)
    assert "DoF" in repr(res_mil)


# ----------------------------------------------------------------------------
# #481: a trend is named only when the test is significant
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("func", [laplace, mil_hdbk_189c])
def test_trend_is_none_when_not_significant(func):
    # The statistic leans towards an increasing rate (p about 0.25 and 0.20),
    # which the result used to report as "Suggested trend: increasing".
    x = [10, 19, 27, 34, 40, 45, 49, 52, 54]
    res = func(x, T=60)
    assert res.p_value > 0.05
    assert res.direction == "increasing"
    assert res.trend == "none"
    assert "no trend detected (p >= 0.05)" in repr(res)
    assert "Trend            : increasing" not in repr(res)


@pytest.mark.parametrize("func", [laplace, mil_hdbk_189c])
def test_trend_is_named_when_significant(func):
    x = [20, 32, 41, 48, 54, 59, 63, 67, 70, 73, 76, 78, 80, 82, 84]
    res = func(x, T=85)
    assert res.p_value < 0.01
    assert res.direction == res.trend == "increasing"
    assert "increasing (p < 0.05)" in repr(res)


@pytest.mark.parametrize("func", [laplace, mil_hdbk_189c])
def test_alpha_ci_sets_the_level(func):
    x = [10, 19, 27, 34, 40, 45, 49, 52, 54]
    p = func(x, T=60).p_value
    assert func(x, T=60, alpha_ci=p * 1.01).trend == "increasing"
    assert func(x, T=60, alpha_ci=p * 0.99).trend == "none"
    assert func(x, T=60, alpha_ci=0.3).alpha_ci == 0.3


@pytest.mark.parametrize("func", [laplace, mil_hdbk_189c])
@pytest.mark.parametrize("bad", [0, 1, -0.1, 1.5, [0.05], "0.05"])
def test_alpha_ci_is_validated(func, bad):
    with pytest.raises(ValueError, match="alpha_ci"):
        func([10, 19, 27, 34], T=40, alpha_ci=bad)


@pytest.mark.parametrize("func", [laplace, mil_hdbk_189c])
def test_one_sided_test_against_the_data_names_no_trend(func):
    # Failures speeding up, tested against a decreasing alternative: the
    # p-value is near 1 and no trend (least of all "increasing") is named.
    x = [20, 32, 41, 48, 54, 59, 63, 67, 70, 73, 76, 78, 80, 82, 84]
    res = func(x, T=85, alternative="decreasing")
    assert res.p_value > 0.9
    assert res.direction == "increasing" and res.trend == "none"


def test_model_trend_test_passes_alpha_ci():
    # The fitted models' ``trend_test`` takes the same level as the
    # standalone tests (parametric, renewal and proportional intensity all
    # delegate through ``diagnostics.trend_test``).
    from surpyval import Weibull
    from surpyval.recurrent import CrowAMSAA, GeneralizedRenewal

    x = [10.0, 19.0, 27.0, 34.0, 40.0, 45.0, 49.0, 52.0, 54.0, 60.0]
    c = [0] * 9 + [1]
    direct = laplace(x[:-1], T=60.0)
    for model in (
        CrowAMSAA.fit(x, c=c),
        GeneralizedRenewal.fit(x, c=c, dist=Weibull),
    ):
        res = model.trend_test()
        assert res.p_value == pytest.approx(direct.p_value)
        assert res.trend == "none" and res.direction == "increasing"
        loose = model.trend_test(alpha_ci=0.3)
        assert loose.alpha_ci == 0.3 and loose.trend == "increasing"


# ----------------------------------------------------------------------------
# #575: per-system observation windows (s_q, T_q] (delayed entry)
# ----------------------------------------------------------------------------

_X1 = [10.0, 19.0, 27.0, 34.0, 40.0, 45.0, 49.0, 52.0, 54.0]
_X2 = [35.0, 48.0, 60.0, 66.0, 71.0, 75.0, 78.0]
_I = [1] * 9 + [2] * 7


def test_575_laplace_delayed_entry_hand_computation():
    # System 1 on (0, 60], system 2 on (30, 80]: each event's null mean is
    # its window's centre and its variance the window length squared / 12.
    res = laplace(_X1 + _X2, i=_I, T={1: 60, 2: 80}, tl={1: 0, 2: 30})
    num = sum(_X1) - 9 * 30.0 + sum(_X2) - 7 * 55.0
    hand = num / np.sqrt(9 * 60.0**2 / 12 + 7 * 50.0**2 / 12)
    assert res.statistic == pytest.approx(hand, rel=1e-12)
    assert res.statistic == pytest.approx(1.674804448250529, rel=1e-12)
    assert res.p_value == pytest.approx(2 * norm.sf(hand), rel=1e-12)
    assert res.n_events == 16 and res.n_systems == 2


def test_575_mil_delayed_entry_hand_computation():
    # Time is measured from each system's start: -log((t - s) / (T - s)) is
    # Exp(1) under the HPP null, so the statistic is chi2 on 2N exactly.
    res = mil_hdbk_189c(_X1 + _X2, i=_I, T={1: 60, 2: 80}, tl={1: 0, 2: 30})
    hand = 2 * (
        np.log(60 / np.array(_X1)).sum()
        + np.log(50 / (np.array(_X2) - 30)).sum()
    )
    assert res.statistic == pytest.approx(hand, rel=1e-12)
    assert res.dof == 32
    assert res.p_value == pytest.approx(
        2 * min(chi2.cdf(hand, 32), chi2.sf(hand, 32)), rel=1e-12
    )


@pytest.mark.parametrize("func", [laplace, mil_hdbk_189c])
@pytest.mark.parametrize("T", [None, {1: 60.0, 2: 80.0}])
def test_575_zero_start_is_the_plain_test(func, T):
    # tl = 0 (scalar, per system or dict) gives the plain test bit for bit.
    plain = func(_X1 + _X2, i=_I, T=T)
    for tl in (0.0, [0.0, 0.0], {1: 0.0, 2: 0.0}):
        res = func(_X1 + _X2, i=_I, T=T, tl=tl)
        assert res.statistic == plain.statistic
        assert res.p_value == plain.p_value


@pytest.mark.parametrize("func", [laplace, mil_hdbk_189c])
def test_575_shifting_a_window_changes_nothing(func):
    # Under the windowed statistics only the position of each event within
    # its window matters: moving system 2 and its window by 1000 leaves the
    # test unchanged (the plain test, from time 0, would change).
    T = {1: 60.0, 2: 80.0}
    res = func(_X1 + _X2, i=_I, T=T, tl={1: 0.0, 2: 30.0})
    shifted = func(
        _X1 + [t + 1000 for t in _X2],
        i=_I,
        T={1: 60.0, 2: 1080.0},
        tl={1: 0.0, 2: 1030.0},
    )
    assert shifted.statistic == pytest.approx(res.statistic, rel=1e-9)


@pytest.mark.parametrize("func", [laplace, mil_hdbk_189c])
def test_575_tl_in_the_fitters_form(func):
    # With c, tl may be one value per row, as CrowAMSAA.fit takes it.
    x = _X1 + [60.0] + _X2 + [80.0]
    i = [1] * 10 + [2] * 8
    c = [0] * 9 + [1] + [0] * 7 + [1]
    tl = [0.0] * 10 + [30.0] * 8
    direct = func(_X1 + _X2, i=_I, T={1: 60, 2: 80}, tl={1: 0, 2: 30})
    res = func(x, i=i, c=c, tl=tl)
    assert res.statistic == pytest.approx(direct.statistic, rel=1e-12)
    assert func(x, i=i, c=c, tl={1: 0, 2: 30}).statistic == res.statistic


@pytest.mark.parametrize("func", [laplace, mil_hdbk_189c])
def test_575_tl_validation(func):
    T = {1: 60.0, 2: 80.0}
    with pytest.raises(ValueError, match="at or before its observation"):
        func(_X1 + _X2, i=_I, T=T, tl={1: 0.0, 2: 35.0})
    with pytest.raises(ValueError, match="one entry per system"):
        func(_X1 + _X2, i=_I, T=T, tl=[0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="must be finite"):
        func(_X1 + _X2, i=_I, T=T, tl={1: 0.0, 2: np.nan})
    x = _X1 + [60.0]
    c = [0] * 9 + [1]
    with pytest.raises(ValueError, match="same on every row"):
        func(x, c=c, tl=[0.0] * 9 + [5.0])
    with pytest.raises(ValueError, match="one entry per row"):
        func(x, c=c, tl=[0.0, 0.0])
