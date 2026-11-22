"""The two-clock fit's data (0.29): univariate data on the main clock
(``x``, ``c``, ``n``, ``t``) plus ``x2``, the subordinate clock read at the
same moments, as an array or a ``SecondaryClock``.

The simulations below have fixed seeds and sizes, chosen before their
results were seen.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

import surpyval as sp
from surpyval import SecondaryClock, WeibullEquivalence
from surpyval.univariate.equivalence import _Data, _log_lik


def _arrays(size=250, seed=7):
    rng = np.random.default_rng(seed)
    rate = rng.choice([0.02, 0.1, 0.5], size)
    T = 2000 * rng.weibull(2.0, size)
    hours = T / (1 + 10 * rate)
    c = (hours > 1500).astype(int)
    hours = np.minimum(hours, 1500)
    starts = rate * hours
    n = rng.integers(1, 3, size)
    tl = np.column_stack([hours * 0.1, starts * 0.1])
    tl[::3] = 0.0
    return hours, starts, c, n, tl


def _same(a, b):
    np.testing.assert_array_equal(a.params, b.params)
    np.testing.assert_array_equal(a.covariance(), b.covariance())
    assert a.neg_ll() == b.neg_ll()
    assert a.maximum == b.maximum


# ---------------------------------------------------------------------------
# The array and SecondaryClock forms are the 0.28 call
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dist", [sp.Weibull, sp.LogNormal])
def test_array_and_secondary_clock_give_the_0_28_numbers(dist):
    hours, starts, c, n, _ = _arrays()
    fitter = sp.Equivalence(dist)
    array = fitter.fit(hours, starts, c, n)
    clock = fitter.fit(hours, SecondaryClock(starts), c=c, n=n)
    keyword = fitter.fit(x=hours, x2=starts, c=c, n=n)
    _same(array, clock)
    _same(array, keyword)
    # 0.28's keyword x1 is the main clock x, with a deprecation warning
    with pytest.warns(DeprecationWarning, match="'x1' is deprecated.*'x'"):
        old = fitter.fit(x1=hours, x2=starts, c=c, n=n)
    _same(array, old)
    assert array.clocks == ("x1", "x2")


def test_the_clock_name_names_the_model_clock():
    hours, starts, c, n, _ = _arrays()
    named = WeibullEquivalence.fit(
        hours, SecondaryClock(starts, name="starts"), c=c, n=n
    )
    assert named.clocks == ("x1", "starts")
    both = WeibullEquivalence.fit(
        hours,
        SecondaryClock(starts, name="starts"),
        c=c,
        n=n,
        clocks=("hours", "starts"),
    )
    assert both.clocks == ("hours", "starts")
    assert "one unit of 'starts' is worth" in repr(both)
    _same(named, both)


# ---------------------------------------------------------------------------
# Late entry, per clock
# ---------------------------------------------------------------------------
def test_late_entry_with_secondary_clock_tl_is_the_entry_path_of_0_28():
    hours, starts, c, n, tl = _arrays()
    with pytest.warns(DeprecationWarning, match="tl as a pair per unit"):
        pair = WeibullEquivalence.fit(hours, starts, c, n, tl)
    clock = WeibullEquivalence.fit(
        hours, SecondaryClock(starts, tl=tl[:, 1]), c=c, n=n, tl=tl[:, 0]
    )
    _same(pair, clock)
    # t, the univariate window, with no upper bound
    t = np.column_stack([tl[:, 0], np.full(len(hours), np.inf)])
    windowed = WeibullEquivalence.fit(
        hours, SecondaryClock(starts, tl=tl[:, 1]), c=c, n=n, t=t
    )
    _same(pair, windowed)
    # the entry changes the fit
    plain = WeibullEquivalence.fit(hours, starts, c=c, n=n)
    assert not np.allclose(plain.params, clock.params)


def test_entry_term_is_the_survival_at_the_combined_entry_age():
    hours, starts, c, n, tl = _arrays(60, seed=3)
    d = _Data(hours, SecondaryClock(starts, tl=tl[:, 1]), c, n, tl[:, 0])
    plain = _Data(hours, starts, c, n)
    theta, k = [1800.0, 1.7], 8.0
    Tl = tl[:, 0] + k * tl[:, 1]
    expected = _log_lik(sp.Weibull, plain, theta, k) - np.sum(
        n * sp.Weibull.log_sf(np.where(Tl > 0, Tl, 1.0), *theta) * (Tl > 0)
    )
    assert _log_lik(sp.Weibull, d, theta, k) == pytest.approx(
        expected, rel=1e-12
    )


# ---------------------------------------------------------------------------
# Shapes that do not line up, said plainly
# ---------------------------------------------------------------------------
def test_shape_mismatches_say_what_is_off():
    hours, starts, c, n, tl = _arrays(12)
    with pytest.raises(
        ValueError,
        match=r"tl was given for 8 units but the secondary clock "
        r"\(starts\) has no entry reading: set SecondaryClock\(\.\.\., "
        r"tl=\.\.\.\)",
    ):
        WeibullEquivalence.fit(
            hours, SecondaryClock(starts, name="starts"), c=c, tl=tl[:, 0]
        )
    with pytest.raises(
        ValueError, match="x has 12 rows but the secondary clock has 11"
    ):
        WeibullEquivalence.fit(hours, starts[:-1], c=c)
    with pytest.raises(
        ValueError, match="entry readings .* but tl, the main clock's"
    ):
        WeibullEquivalence.fit(hours, SecondaryClock(starts, tl=tl[:, 1]))
    with pytest.raises(ValueError, match="one number per row: 12 values"):
        WeibullEquivalence.fit(hours, starts, tl=tl[:5, 0])
    with pytest.raises(ValueError, match=r"one value per row, or an N x 2"):
        SecondaryClock(np.ones((12, 3)))
    with pytest.raises(ValueError, match="negative readings, at rows"):
        SecondaryClock(-starts)
    with pytest.raises(ValueError, match="missing or infinite reading"):
        SecondaryClock(np.r_[np.nan, starts[1:]])
    with pytest.raises(ValueError, match="'starts', but the SecondaryClock"):
        WeibullEquivalence.fit(
            hours,
            SecondaryClock(starts, name="landings"),
            clocks=("hours", "starts"),
        )
    with pytest.raises(ValueError, match="name names the second clock"):
        SecondaryClock(starts, name="")
    with pytest.raises(ValueError, match="right truncated"):
        WeibullEquivalence.fit(hours, starts, tr=np.full(12, 5000.0))
    with pytest.raises(ValueError, match="Give the truncation once"):
        WeibullEquivalence.fit(
            hours, starts, t=np.zeros((12, 2)), tl=np.zeros(12)
        )
    with pytest.raises(ValueError, match="x2, the second clock's reading"):
        WeibullEquivalence.fit(hours)
    with pytest.raises(ValueError, match="entry readings too"):
        WeibullEquivalence.fit(
            hours, SecondaryClock(starts, tl=tl[:, 1]), tl=tl
        )
    # a window on a row that is not an interval
    x = np.column_stack([hours, hours])
    x[2, 0] -= 1.0
    with pytest.raises(ValueError, match=r"Rows \[2\] have a window"):
        WeibullEquivalence.fit(x, starts, c=c)


# ---------------------------------------------------------------------------
# The multivariate path is gone; other data objects are refused
# ---------------------------------------------------------------------------
def test_multivariate_data_is_refused_pointing_to_x2():
    from surpyval.multivariate import MultivariateSurpyvalData

    hours, starts, c, n, _ = _arrays()
    data = MultivariateSurpyvalData(
        [hours, starts], c=np.column_stack([c, c]), names=("hours", "starts")
    )
    message = r"not a MultivariateSurpyvalData.*x2.*SecondaryClock"
    with pytest.raises(ValueError, match=message):
        WeibullEquivalence.fit(data)
    with pytest.raises(ValueError, match=message):
        WeibullEquivalence.fit(hours, data)
    assert not hasattr(WeibullEquivalence, "n_axes")
    with pytest.raises(ValueError, match="groups and sorts its rows"):
        WeibullEquivalence.fit(sp.SurpyvalData(hours, c, n), starts)


def test_other_fitters_refuse_a_second_clock():
    hours, starts, c, n, _ = _arrays()
    with pytest.raises(ValueError, match="Weibull.fit does not take x2"):
        sp.Weibull.fit(hours, c, x2=starts)
    with pytest.raises(ValueError, match="KaplanMeier.fit does not take x2"):
        sp.KaplanMeier.fit(hours, c, x2=starts)
    with pytest.raises(ValueError, match="WeibullPH.fit does not take x2"):
        sp.WeibullPH.fit(hours, starts[:, None], c, x2=starts)
    df = pd.DataFrame({"hours": hours, "starts": starts, "c": c})
    with pytest.raises(
        ValueError, match="Weibull.fit_from_df does not take x2_col"
    ):
        sp.Weibull.fit_from_df(df, x_col="hours", c_col="c", x2_col="starts")
    with pytest.raises(ValueError, match="does not take tl2_col, x2_name"):
        sp.Weibull.fit_from_df(df, x_col="hours", tl2_col=0, x2_name="s")


# ---------------------------------------------------------------------------
# fit_from_df: every column
# ---------------------------------------------------------------------------
def test_fit_from_df_reads_every_column():
    hours, starts, c, n, tl = _arrays()
    df = pd.DataFrame(
        {
            "hours": hours,
            "starts": starts,
            "failed": c,
            "count": n,
            "hours_in": tl[:, 0],
            "starts_in": tl[:, 1],
        }
    )
    model = WeibullEquivalence.fit_from_df(
        df,
        x_col="hours",
        x2_col="starts",
        c_col="failed",
        n_col="count",
        tl_col="hours_in",
        tl2_col="starts_in",
        x2_name="engine starts",
    )
    direct = WeibullEquivalence.fit(
        hours,
        SecondaryClock(starts, tl=tl[:, 1], name="engine starts"),
        c=c,
        n=n,
        tl=tl[:, 0],
        clocks=("hours", "engine starts"),
    )
    _same(model, direct)
    assert model.clocks == ("hours", "engine starts")
    # one number for every row
    zero = WeibullEquivalence.fit_from_df(
        df,
        x_col="hours",
        x2_col="starts",
        c_col="failed",
        tl_col=0.0,
        tl2_col=0.0,
    )
    _same(zero, WeibullEquivalence.fit(hours, starts, c=c))
    # 0.28's columns, deprecated
    with pytest.warns(DeprecationWarning, match="x_cols and tl_cols"):
        old = WeibullEquivalence.fit_from_df(
            df,
            x_cols=["hours", "starts"],
            c_col="failed",
            n_col="count",
            tl_cols=["hours_in", "starts_in"],
        )
    _same(old, model)
    assert old.clocks == ("hours", "starts")
    with pytest.warns(DeprecationWarning):
        positional = WeibullEquivalence.fit_from_df(
            df, ["hours", "starts"], "failed", "count"
        )
    _same(positional, WeibullEquivalence.fit(hours, starts, c=c, n=n))
    with pytest.raises(ValueError, match="'nope' is not in the DataFrame"):
        WeibullEquivalence.fit_from_df(df, x_col="hours", x2_col="nope")
    with pytest.raises(ValueError, match="x2_col, the column of the second"):
        WeibullEquivalence.fit_from_df(df, x_col="hours")


def test_fit_from_df_reads_interval_columns():
    x, s, c = _interval_data()
    df = pd.DataFrame(
        {
            "h0": x[:, 0],
            "h1": x[:, 1],
            "s0": s[:, 0],
            "s1": s[:, 1],
            "c": c,
        }
    )
    model = WeibullEquivalence.fit_from_df(
        df,
        xl_col="h0",
        xr_col="h1",
        x2l_col="s0",
        x2r_col="s1",
        c_col="c",
        clocks=("hours", "starts"),
    )
    direct = WeibullEquivalence.fit(x, s, c=c, clocks=("hours", "starts"))
    _same(model, direct)
    assert model.clocks == ("hours", "starts")


# ---------------------------------------------------------------------------
# Interval rows (c = 2): S(T_lo) - S(T_hi) on the combined clock
# ---------------------------------------------------------------------------
def _interval_data(size=400, seed=3, inspect=200.0):
    """Units inspected every ``inspect`` hours: a failure is known to lie
    between two inspections (on both clocks); a unit still running at the
    end of its follow-up is right censored there."""
    rng = np.random.default_rng(seed)
    rate = rng.choice([0.02, 0.1, 0.5], size)
    life = 2000 * rng.weibull(2.0, size) / (1 + 10 * rate)
    hi = np.ceil(life / inspect) * inspect
    end = rng.uniform(800, 3000, size)
    running = hi > end
    c = np.where(running, 1, 2)
    lo = np.where(running, end, hi - inspect)
    hi = np.where(running, end, hi)
    x = np.column_stack([lo, hi])
    return x, rate[:, None] * x, c


def _weibull_nll(x, s, c, n=None, tl=None, sl=None):
    """The two-clock Weibull likelihood written out in numpy, rows of
    every kind, for a brute-force check."""
    n = np.ones(len(c)) if n is None else n

    def nll(alpha, beta, k):
        def S(t):
            return np.exp(-((t / alpha) ** beta))

        lo, hi = x[:, 0] + k * s[:, 0], x[:, 1] + k * s[:, 1]
        h = x[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = np.where(h > 0, s[:, 1] / h, 0.0)
            f = beta / alpha * (hi / alpha) ** (beta - 1) * S(hi)
            # np.select works out every branch on every row
            ll = np.select(
                [c == 0, c == 1, c == -1, c == 2],
                [
                    np.log(f) + np.log1p(k * rho),
                    np.log(S(hi)),
                    np.log(1 - S(hi)),
                    np.log(S(lo) - S(hi)),
                ],
            )
        if tl is not None:
            ll = ll - np.log(S(tl + k * sl))
        return -np.sum(n * ll)

    return nll


def test_interval_fit_is_the_brute_force_maximum():
    x, s, c = _interval_data()
    model = WeibullEquivalence.fit(x, SecondaryClock(s, name="starts"), c=c)
    assert model.maximum == "verified"
    nll = _weibull_nll(x, s, c)
    assert model.neg_ll() == pytest.approx(nll(*model.params), rel=1e-12)
    found = minimize(
        lambda p: nll(*np.exp(p)),
        np.log([2000.0, 2.0, 10.0]),
        method="Nelder-Mead",
        options=dict(xatol=1e-10, fatol=1e-12, maxiter=20000),
    )
    np.testing.assert_allclose(np.exp(found.x), model.params, rtol=1e-5)
    assert model.neg_ll() <= found.fun + 1e-8
    # the truth (k = 10, Weibull(2000, 2)) inside the intervals
    lo, hi = model.param_cb("k")
    assert lo < 10.0 < hi
    lo, hi = model.param_cb("k", method="lr")
    assert lo < 10.0 < hi


def test_every_kind_of_row_together_is_the_written_out_likelihood():
    x, s, c = _interval_data(300, seed=11)
    rng = np.random.default_rng(12)
    exact = rng.random(300) < 0.3
    x = np.where(exact[:, None] & (c == 2)[:, None], x[:, [1]], x)
    s = np.where(exact[:, None] & (c == 2)[:, None], s[:, [1]], s)
    c = np.where(exact & (c == 2), 0, c)
    c[:15] = np.where(c[:15] == 0, -1, c[:15])
    n = rng.integers(1, 3, 300)
    frac = rng.uniform(0, 0.5, 300) * (rng.random(300) < 0.5)
    hl, sl = frac * x[:, 0], frac * s[:, 0]
    d = _Data(x, SecondaryClock(s, tl=sl), c, n, hl)
    nll = _weibull_nll(x, s, c, n, hl, sl)
    for theta, k in (([1500.0, 1.5], 4.0), ([2500.0, 2.5], 15.0)):
        assert -_log_lik(sp.Weibull, d, theta, k) == pytest.approx(
            nll(*theta, k), rel=1e-12
        )
    model = WeibullEquivalence.fit(x, SecondaryClock(s, tl=sl), c, n, hl)
    assert model.maximum == "verified"
    assert model.neg_ll() == pytest.approx(nll(*model.params), rel=1e-12)
    # equal columns where a row is not an interval: 1-D readings of the
    # same rows are the same data
    point = c != 2
    assert np.array_equal(x[point, 0], x[point, 1])


def test_an_interval_on_starts_alone_rules_out_k_zero():
    # Failed between two inspections with no running hours in between:
    # impossible if starts did not matter.
    x, s, c = _interval_data()
    i = np.flatnonzero(c == 2)[:5]
    x[i, 0] = x[i, 1]
    s[i, 0] = np.maximum(s[i, 1] - 20.0, 0.0)
    model = WeibullEquivalence.fit(x, s, c=c)
    assert model.k > 0
    assert model.k_lr_test()["p_value"] == 0.0


def test_interval_rows_need_both_readings_of_the_second_clock():
    x, s, c = _interval_data(50)
    with pytest.raises(
        ValueError,
        match=r"are intervals \(c = 2\), but the secondary clock has one "
        r"reading per row",
    ):
        WeibullEquivalence.fit(x, s[:, 1], c=c)
    # c defaults to 2 where x's window has width
    _same(
        WeibullEquivalence.fit(x, s),
        WeibullEquivalence.fit(x, s, c=2 * (x[:, 0] < x[:, 1])),
    )


def test_interval_model_round_trips():
    x, s, c = _interval_data()
    model = WeibullEquivalence.fit(x, SecondaryClock(s, name="starts"), c=c)
    restored = sp.from_json(model.to_json())
    np.testing.assert_array_equal(restored.params, model.params)
    np.testing.assert_array_equal(restored.cb([500.0]), model.cb([500.0]))
    assert restored.clocks == model.clocks
    assert restored.aic() == model.aic()


# ---------------------------------------------------------------------------
# Saving the clock
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "clock",
    [
        SecondaryClock([3.0, 5.0, 0.0]),
        SecondaryClock([3.0, 5.0, 0.0], tl=[0.0, 1.0, 0.0], name="starts"),
        SecondaryClock([[1.0, 3.0], [5.0, 5.0]], tl=0.5, name="landings"),
    ],
)
def test_secondary_clock_round_trips(clock, tmp_path):
    again = SecondaryClock.from_dict(clock.to_dict())
    for other in (
        again,
        sp.from_dict(clock.to_dict()),
        sp.from_json(clock.to_json()),
    ):
        assert isinstance(other, SecondaryClock)
        np.testing.assert_array_equal(other.values, clock.values)
        assert other.name == clock.name
        if clock.tl is None:
            assert other.tl is None
        else:
            np.testing.assert_array_equal(other.tl, clock.tl)
    fp = tmp_path / "clock.json"
    clock.to_json(fp)
    np.testing.assert_array_equal(
        SecondaryClock.from_json(fp).values, clock.values
    )
    with pytest.raises(ValueError, match="written by SurpyvalData"):
        SecondaryClock.from_dict({"class": "SurpyvalData", "values": [1]})
    with pytest.raises(ValueError, match="needs 'values'"):
        SecondaryClock.from_dict({"class": "SecondaryClock"})


def test_secondary_clock_repr():
    assert repr(SecondaryClock([1.0, 2.0], tl=0, name="starts")) == (
        "SecondaryClock('starts', 2 rows, with entry readings)"
    )


def test_no_warning_from_the_new_forms():
    hours, starts, c, n, tl = _arrays()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        WeibullEquivalence.fit(
            hours, SecondaryClock(starts, tl=tl[:, 1]), c, n, tl[:, 0]
        )
