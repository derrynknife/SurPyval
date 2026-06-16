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
    assert res.trend == "increasing"
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
    assert "Suggested trend" in text
    assert isinstance(res, TrendTestResult)

    res_mil = mil_hdbk_189c([10.0, 19.0, 27.0, 34.0], T=40.0)
    assert "MIL-HDBK-189C Trend Test" in repr(res_mil)
    assert "DoF" in repr(res_mil)
