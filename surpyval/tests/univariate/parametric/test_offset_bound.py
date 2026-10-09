"""The offset's upper bound and starts on censored data (#631, #632,
#633)."""

import warnings

import numpy as np

import surpyval as surv


def _quiet(fit, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit(**kwargs)


def _inspected_weibull():
    # True offset 5; every failure seen in its 3-wide inspection interval
    rng = np.random.default_rng(1)
    T = 5 + surv.Weibull.random(500, 10, 2, random_state=rng)
    lo = np.floor(T / 3) * 3
    return lo, lo + 3, np.full(len(T), 2)


def test_633_interval_starts_do_not_cap_the_offset():
    # The bound was the smallest value of any row, the first interval
    # start 3.0: the fit stopped there (-904.69) and called it verified,
    # though the likelihood rises to -896.85 at gamma = 5.2.
    xl, xr, c = _inspected_weibull()
    model = _quiet(surv.Weibull.fit, xl=xl, xr=xr, c=c, offset=True)
    assert model.maximum == "verified"
    assert model.gamma > 5.0
    assert model.neg_ll() < 896.86


def test_633_right_censored_times_do_not_cap_the_offset():
    # Withdrawals before the first failure meet S = 1 below the offset
    rng = np.random.default_rng(2)
    x = 20 + surv.Weibull.random(60, 10, 2, random_state=rng)
    x = np.r_[x, [5.0, 8.0]]
    c = np.r_[np.zeros(60), [1, 1]]
    with_early = _quiet(surv.Weibull.fit, x=x, c=c, offset=True)
    assert with_early.gamma > 8.0
    without = _quiet(surv.Weibull.fit, x=x[:60], offset=True)
    assert np.isclose(with_early.gamma, without.gamma, rtol=1e-5)


def test_633_a_custom_distribution_sees_no_negative_shifted_time():
    # A cumulative hazard that is NaN below 0 (as (x / alpha) ** beta is)
    # with right-censored times below the offset
    from autograd import numpy as anp

    Wb = surv.CustomDistribution(
        "Wb",
        lambda x, a, b: (anp.array(x) / a) ** b,
        ["a", "b"],
        ((0, None), (0, None)),
        (0, anp.inf),
    )
    rng = np.random.default_rng(3)
    x = np.r_[20 + surv.Weibull.random(60, 10, 2, random_state=rng), [5.0]]
    c = np.r_[np.zeros(60), [1]]
    model = _quiet(Wb.fit, x=x, c=c, offset=True)
    assert np.isfinite(model.neg_ll())
    assert model.gamma > 5.0


def test_632_an_interior_offset_maximum_is_found_on_interval_data():
    # The bound was the first interval start 4.49, and the search, run
    # onto it, said "no finite maximum"; the maximum is at 4.2765.
    rows = [
        (0.0, 0.0, 0, 38),
        (4.49, 6.73, 2, 9),
        (6.73, 8.98, 2, 23),
        (8.98, 11.22, 2, 24),
        (11.22, 13.47, 2, 40),
        (13.47, 15.71, 2, 48),
        (15.71, 17.95, 2, 43),
        (17.95, 20.2, 2, 20),
        (19.35, 19.35, 1, 155),
    ]
    xl, xr, c, n = map(np.array, zip(*rows))
    model = _quiet(
        surv.Rayleigh.fit,
        xl=xl,
        xr=xr,
        c=c,
        n=n,
        offset=True,
        lfp=True,
        zi=True,
    )
    assert model.maximum == "verified"
    assert abs(model.gamma - 4.2765) < 1e-3
    assert model.neg_ll() < 746.453


def test_631_left_censored_rows_with_zero_inflation_fit():
    # A left-censored row is imputed half way to the smallest value, an
    # exact zero with zi, below the start taken from the nonzero data:
    # shifted by it, the seed's data went negative and the initialiser
    # raised.
    rows = [
        (0.0, 0.0, 0, -np.inf, np.inf),
        (16.06, 18.07, 2, 6.95, np.inf),
        (10.92, 10.92, -1, -np.inf, np.inf),
        (14.05, 16.06, 2, 7.18, np.inf),
        (15.07, 15.07, 0, -np.inf, np.inf),
    ]
    xl, xr, c, tl, tr = map(list, zip(*rows))
    model = _quiet(
        surv.Weibull.fit,
        xl=xl,
        xr=xr,
        c=c,
        tl=tl,
        tr=tr,
        offset=True,
        lfp=True,
        zi=True,
    )
    assert np.isfinite(model.neg_ll())

    rows = [
        (0.0, 0.0, 0, -np.inf, 26.31),
        (9.92, 9.92, 0, -np.inf, np.inf),
        (11.97, 11.97, -1, 8.25, 21.78),
        (10.3, 12.36, 2, -np.inf, np.inf),
    ]
    xl, xr, c, tl, tr = map(list, zip(*rows))
    model = _quiet(
        surv.LogNormal.fit,
        xl=xl,
        xr=xr,
        c=c,
        tl=tl,
        tr=tr,
        offset=True,
        zi=True,
    )
    assert np.isfinite(model.neg_ll())
