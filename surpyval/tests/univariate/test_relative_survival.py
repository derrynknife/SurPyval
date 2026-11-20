"""Relative survival: rate tables, net survival (Pohar Perme and Ederer II)
and excess hazard regression."""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.stats import norm

import surpyval as sp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _gompertz_table(scale=1.0):
    """Rates rising 9% a year of age, in annual bands to 120, in a time
    unit of 1 / scale years (scale=12: months)."""
    ages = np.arange(0, 121) * scale
    return sp.RateTable(5e-5 * np.exp(0.09 * ages / scale) / scale, ages=ages)


def _population_times(table, age, u):
    """Population lifetimes from a table by age alone: the expected
    survival inverted at u, through the cumulative hazard by attained age
    (linear between the band edges)."""
    edges = np.r_[table.ages, table.ages[-1] + 1e3]
    rates = table.rates[0, :, 0]
    C = np.r_[0.0, np.cumsum(rates * np.diff(edges))]
    target = np.interp(age, edges, C) - np.log(u)
    return np.interp(target, C, edges) - age


def _small():
    """Five units, ages 60 to 75, with a rate table by age only."""
    table = sp.RateTable([0.01, 0.03], ages=[0, 70])
    return {
        "x": np.array([2.0, 3.0, 5.0, 6.0, 8.0]),
        "c": np.array([0, 0, 1, 0, 1]),
        "rate_table": table,
        "age": np.array([60.0, 64.0, 66.0, 70.0, 75.0]),
    }


# ---------------------------------------------------------------------------
# RateTable
# ---------------------------------------------------------------------------
def test_rate_table_integrates_along_age_and_calendar_year():
    table = sp.RateTable(
        [[0.01, 0.02], [0.03, 0.04]], ages=[0, 50], years=[2000, 2005]
    )
    # Aged 48 in 2003: two years at 0.01, then over 50 and past 2005.
    assert table.Hf(4, age=48, year=2003) == pytest.approx(0.02 + 0.08)
    # Aged 49 in 2003: one year at 0.01, one at 0.03, two at 0.04.
    assert table.Hf(4, age=49, year=2003) == pytest.approx(0.01 + 0.03 + 0.08)
    np.testing.assert_allclose(
        table.hf([0.5, 1.5, 3.0], age=49, year=2003), [0.01, 0.03, 0.04]
    )
    np.testing.assert_allclose(
        table.sf([1, 4], age=48, year=2003),
        np.exp(-np.array([0.01, 0.1])),
        rtol=1e-12,
    )


def test_rate_table_bands_extend_beyond_the_table():
    table = sp.RateTable([[0.01, 0.02]], ages=[40], years=[2000, 2010])
    # Before the first year band, and below the first age edge, the first
    # band's rate; after the last year, the last.
    assert table.Hf(5, age=20, year=1990) == pytest.approx(0.05)
    assert table.Hf(5, age=20, year=2020) == pytest.approx(0.10)


def test_rate_table_shapes_and_missing_values():
    table = sp.RateTable([0.01, 0.02], ages=[0, 50])
    assert np.ndim(table.Hf(5, age=45)) == 0
    out = table.Hf([[1, 2], [3, np.nan]], age=45)
    assert out.shape == (2, 2) and np.isnan(out[1, 1])
    np.testing.assert_allclose(table.Hf(5, age=[45, 50]), [0.05, 0.1])


def test_rate_table_strata():
    table = sp.RateTable({"F": [0.01], "M": [0.02]}, ages=[0])
    np.testing.assert_allclose(
        table.Hf(2, age=50, stratum=["F", "M"]), [0.02, 0.04]
    )
    with pytest.raises(ValueError, match="not a stratum"):
        table.Hf(2, age=50, stratum="X")
    with pytest.raises(ValueError, match="stratified"):
        table.Hf(2, age=50)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"rates": [0.01, -0.01], "ages": [0, 50]}, "at least 0"),
        ({"rates": [0.01], "ages": [0, 50]}, "shape"),
        ({"rates": [0.01, 0.02], "ages": [50, 0]}, "increasing"),
        ({"rates": [[0.01]], "ages": [0], "years": [2000, 2010]}, "shape"),
        ({"rates": {}, "ages": [0]}, "empty"),
    ],
)
def test_rate_table_refuses_bad_tables(kwargs, match):
    with pytest.raises(ValueError, match=match):
        sp.RateTable(**kwargs)


def test_rate_table_year_rules():
    by_year = sp.RateTable([[0.01, 0.02]], ages=[0], years=[2000, 2010])
    with pytest.raises(ValueError, match="by calendar year"):
        by_year.Hf(1, age=50)
    plain = sp.RateTable([0.01], ages=[0])
    with pytest.raises(ValueError, match="no calendar-year"):
        plain.Hf(1, age=50, year=2000)


def test_rate_table_from_df_and_round_trip():
    df = pd.DataFrame(
        {
            "age": [0, 50, 0, 50, 0, 50, 0, 50],
            "year": [2000, 2000, 2010, 2010] * 2,
            "sex": ["F"] * 4 + ["M"] * 4,
            "rate": [0.01, 0.02, 0.008, 0.015, 0.012, 0.03, 0.01, 0.02],
        }
    )
    table = sp.RateTable.from_df(df, "age", "rate", "year", "sex")
    direct = sp.RateTable(
        {
            "F": [[0.01, 0.008], [0.02, 0.015]],
            "M": [[0.012, 0.01], [0.03, 0.02]],
        },
        ages=[0, 50],
        years=[2000, 2010],
    )
    assert table == direct
    assert sp.RateTable.from_dict(table.to_dict()) == table
    assert sp.from_dict(table.to_dict()) == table
    with pytest.raises(ValueError, match="needs a rate"):
        sp.RateTable.from_df(df.iloc[1:], "age", "rate", "year", "sex")


# ---------------------------------------------------------------------------
# RelativeSurvival
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["pohar-perme", "ederer2"])
def test_no_background_hazard_is_nelson_aalen(method):
    x = np.array([1.0, 2.0, 2.0, 3.5, 4.0, 5.0, 7.0])
    c = np.array([0, 0, 1, 0, 1, 0, 0])
    n = np.array([1, 2, 1, 1, 1, 3, 1])
    table = sp.RateTable([0.0], ages=[0])
    model = sp.RelativeSurvival.fit(
        x, c, n, rate_table=table, age=60, method=method
    )
    na = sp.NelsonAalen.fit(x, c, n)
    np.testing.assert_allclose(model.Hf(x), na.Hf(x), rtol=1e-12)
    # The variance is Aalen's, sum d / Y**2.
    Y = np.array([10.0, 9.0, 6.0, 5.0, 4.0, 1.0])
    d = np.array([1.0, 2.0, 1.0, 0.0, 3.0, 1.0])
    times = np.array([1.0, 2.0, 3.5, 4.0, 5.0, 7.0])
    np.testing.assert_allclose(
        model.standard_errors(times), np.sqrt(np.cumsum(d / Y**2)), rtol=1e-12
    )


@pytest.mark.parametrize("method", ["pohar-perme", "ederer2"])
def test_one_background_rate_for_everyone_subtracts_it(method):
    # With the same constant population hazard for every unit, the
    # weights are the same for every unit at risk, and both estimators
    # are the Nelson-Aalen estimate less rate * t.
    x = np.array([1.0, 2.0, 3.0, 4.5, 6.0])
    c = np.array([0, 1, 0, 0, 1])
    table = sp.RateTable([0.05], ages=[0])
    model = sp.RelativeSurvival.fit(
        x, c, rate_table=table, age=[30, 40, 50, 60, 70], method=method
    )
    na = sp.NelsonAalen.fit(x, c)
    t = np.array([0.5, 1.0, 2.5, 3.0, 4.5, 5.0, 6.0])
    np.testing.assert_allclose(
        model.Hf(t), na.Hf(t) - 0.05 * t, rtol=1e-12, atol=1e-15
    )


def test_ederer2_by_hand():
    model = sp.RelativeSurvival.fit(**_small(), method="ederer2")
    # At 2: 1 event of 5 at risk, whose expected hazards over (0, 2] are
    # 0.02, 0.02, 0.02, 0.06 and 0.06. At 3: 1 of 4, expected over (2, 3]
    # 0.01, 0.01, 0.03 and 0.03.
    np.testing.assert_allclose(
        model.Hf([2, 3]),
        [1 / 5 - 0.18 / 5, 1 / 5 - 0.18 / 5 + 1 / 4 - 0.08 / 4],
        rtol=1e-12,
    )


def test_pohar_perme_by_hand():
    table = sp.RateTable([0.1, 0.2], ages=[0, 50])
    model = sp.RelativeSurvival.fit(
        [2.0, 3.0], [0, 1], rate_table=table, age=[49.0, 30.0]
    )
    # Unit A (aged 49, event at 2) has an expected cumulative hazard of
    # 0.1 at 1 and 0.3 at 2; unit B (aged 30) one of 0.1 t.
    w_end = np.exp(0.3) + np.exp(0.2)
    at_2 = np.exp(0.3) / w_end - np.log(w_end / 2.0)
    np.testing.assert_allclose(
        model.Hf([2.0, 3.0]), [at_2, at_2 - 0.1], rtol=1e-12
    )
    np.testing.assert_allclose(
        model.standard_errors(2.0), np.exp(0.3) / w_end, rtol=1e-12
    )


def test_net_survival_is_one_before_time_zero_and_holds_after_the_data():
    model = sp.RelativeSurvival.fit(**_small())
    assert model.sf(-1.0) == 1.0 and model.sf(0.0) == 1.0
    assert model.sf(8.0) == model.sf(80.0)
    got = model.sf([1.0, np.nan, 3.0])
    assert np.isnan(got[1]) and not np.isnan(got[[0, 2]]).any()
    # The expected part accrues between the events: net survival rises.
    assert model.sf(4.9) > model.sf(3.0)


def test_pohar_perme_is_consistent_where_ederer2_is_not():
    # The excess hazard rises with age, as the population's does, so the
    # units still followed drift towards the younger: Ederer II
    # overestimates net survival late in follow-up; Pohar Perme does not.
    rng = np.random.default_rng(20261120)
    table = _gompertz_table()
    size = 3000
    age = rng.uniform(50, 85, size)
    lam = 0.08 * np.exp(0.06 * (age - 65))
    excess = rng.exponential(1 / lam)
    background = _population_times(table, age, rng.uniform(size=size))
    time = np.minimum(np.minimum(excess, background), 12.0)
    c = (time == 12.0).astype(int)
    truth = np.mean(np.exp(-lam * 10.0))
    pp = sp.RelativeSurvival.fit(time, c, rate_table=table, age=age)
    e2 = sp.RelativeSurvival.fit(
        time, c, rate_table=table, age=age, method="ederer2"
    )
    lower, upper = pp.cb(10.0)
    assert lower < truth < upper
    assert abs(e2.sf(10.0) - truth) > 2 * abs(pp.sf(10.0) - truth)
    assert e2.sf(10.0) > truth


def test_units_do_not_matter():
    # Times, ages and the table in years, then in months: the rates are
    # per month, the same curve at twelve times the times.
    rng = np.random.default_rng(3)
    age = rng.uniform(50, 80, 60)
    x = rng.exponential(5.0, 60)
    c = (rng.uniform(size=60) < 0.2).astype(int)
    for method in ("pohar-perme", "ederer2"):
        years = sp.RelativeSurvival.fit(
            x, c, rate_table=_gompertz_table(), age=age, method=method
        )
        months = sp.RelativeSurvival.fit(
            12 * x,
            c,
            rate_table=_gompertz_table(12.0),
            age=12 * age,
            method=method,
        )
        t = np.array([0.5, 2.0, 5.0, 9.0])
        np.testing.assert_allclose(months.sf(12 * t), years.sf(t), rtol=1e-9)
        np.testing.assert_allclose(months.cb(12 * t), years.cb(t), rtol=1e-9)


def test_counts_are_repeated_rows():
    d = _small()
    n = np.array([1, 3, 1, 2, 1])
    model = sp.RelativeSurvival.fit(**d, n=n)
    rows = sp.RelativeSurvival.fit(
        np.repeat(d["x"], n),
        np.repeat(d["c"], n),
        rate_table=d["rate_table"],
        age=np.repeat(d["age"], n),
    )
    t = np.linspace(0, 9, 19)
    np.testing.assert_allclose(model.sf(t), rows.sf(t), rtol=1e-12)
    np.testing.assert_allclose(model.cb(t), rows.cb(t), rtol=1e-12)


def test_bounds():
    model = sp.RelativeSurvival.fit(**_small())
    t = np.array([2.0, 3.0, 6.0])
    H, se = model.Hf(t), model.standard_errors(t)
    z = norm.ppf(0.975)
    np.testing.assert_allclose(
        model.cb(t), np.exp(-np.c_[H + z * se, H - z * se]), rtol=1e-12
    )
    S = np.exp(-H)
    np.testing.assert_allclose(
        model.cb(t, bound_type="normal"),
        np.c_[S - z * S * se, S + z * S * se],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        model.cb(t, on="Hf"), np.c_[H - z * se, H + z * se], rtol=1e-12
    )
    np.testing.assert_allclose(
        model.cb(t, bound="lower", alpha_ci=0.05),
        model.cb(t, alpha_ci=0.1)[:, 0],
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="bound_type"):
        model.cb(t, bound_type="log-log")


def test_relative_survival_round_trips():
    model = sp.RelativeSurvival.fit(**_small(), method="ederer2")
    t = np.linspace(0, 10, 21)
    for restored in (
        sp.from_dict(model.to_dict()),
        sp.from_json(model.to_json()),
        pickle.loads(pickle.dumps(model)),
    ):
        assert restored.method == "ederer2"
        np.testing.assert_array_equal(restored.sf(t), model.sf(t))
        np.testing.assert_array_equal(restored.cb(t), model.cb(t))


def test_relative_survival_from_df():
    d = _small()
    df = pd.DataFrame({"t": d["x"], "cens": d["c"], "age": d["age"]})
    model = sp.RelativeSurvival.fit_from_df(
        df, "t", "age", d["rate_table"], c_col="cens", unit="years"
    )
    ref = sp.RelativeSurvival.fit(**d)
    np.testing.assert_array_equal(model.sf(d["x"]), ref.sf(d["x"]))
    assert model.unit == "years"
    assert "Pohar Perme" in repr(model)


def test_relative_survival_drops_a_row_with_a_missing_age():
    d = _small()
    age = d["age"].copy()
    age[1] = np.nan
    with pytest.warns(UserWarning, match="Dropped 1 of 5 rows"):
        model = sp.RelativeSurvival.fit(**{**d, "age": age})
    keep = np.arange(5) != 1
    ref = sp.RelativeSurvival.fit(
        d["x"][keep], d["c"][keep], rate_table=d["rate_table"], age=age[keep]
    )
    np.testing.assert_array_equal(model.sf([2, 5]), ref.sf([2, 5]))


@pytest.mark.parametrize(
    "change, match",
    [
        ({"c": [0, 2, 1, 0, 1]}, "right-censored"),
        ({"x": [2.0, np.nan, 5.0, 6.0, 8.0]}, "missing"),
        ({"rate_table": None}, "rate_table"),
        ({"age": None}, "'age' is required"),
        ({"x": [0.0, 3.0, 5.0, 6.0, 8.0]}, "time 0"),
        ({"c": [1, 1, 1, 1, 1]}, "no events"),
        ({"stratum": "F"}, "no strata"),
        ({"method": "ederer1"}, "method"),
    ],
)
def test_relative_survival_refusals(change, match):
    with pytest.raises(ValueError, match=match):
        sp.RelativeSurvival.fit(**{**_small(), **change})


def test_relative_survival_refuses_surpyval_data():
    data = sp.SurpyvalData([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="take arrays"):
        sp.RelativeSurvival.fit(
            data, rate_table=sp.RateTable([0.01], ages=[0]), age=60
        )


# ---------------------------------------------------------------------------
# ExcessHazard
# ---------------------------------------------------------------------------
def _excess_sample(size=3000, seed=1):
    rng = np.random.default_rng(seed)
    table = _gompertz_table()
    age = rng.uniform(50, 80, size)
    treated = rng.binomial(1, 0.5, size)
    excess = rng.exponential(1 / (0.15 * np.exp(-0.7 * treated)))
    background = _population_times(table, age, rng.uniform(size=size))
    time = np.minimum(np.minimum(excess, background), 8.0)
    return {
        "x": time,
        "Z": treated.reshape(-1, 1),
        "c": (time == 8.0).astype(int),
        "rate_table": table,
        "age": age,
    }


def test_no_background_hazard_is_the_exponential_ph_model():
    # Without a population hazard the model is a piecewise exponential
    # proportional hazards model: one interval and a binary covariate
    # give each group its events over its time at risk.
    x = np.array([1.0, 2.5, 3.0, 4.0, 4.5, 6.0, 7.5, 8.0, 9.5, 11.0])
    c = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 1])
    z = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    table = sp.RateTable([0.0], ages=[0])
    model = sp.ExcessHazard.fit(x, z, c=c, rate_table=table, age=50)
    rate = [np.sum((c == 0) & (z == g)) / np.sum(x[z == g]) for g in (0, 1)]
    np.testing.assert_allclose(
        model.params, [rate[0], np.log(rate[1] / rate[0])], rtol=1e-7
    )
    assert model.maximum == "verified"


def test_excess_hazard_reaches_an_independent_maximum():
    d = _excess_sample(400, seed=5)
    table = d["rate_table"]
    lam_P = table.hf(d["x"], d["age"])
    H_P = table.Hf(d["x"], d["age"])
    event = d["c"] == 0
    z = d["Z"][:, 0]
    cut = 2.0

    def nll(v):
        rates = np.exp(v[:2])
        H0 = rates[0] * np.minimum(d["x"], cut) + rates[1] * np.maximum(
            d["x"] - cut, 0
        )
        h0 = np.where(d["x"] <= cut, rates[0], rates[1])
        risk = np.exp(v[2] * z)
        return -np.sum(event * np.log(lam_P + h0 * risk) - risk * H0 - H_P)

    oracle = minimize(
        nll,
        [np.log(0.1), np.log(0.1), 0.0],
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000},
    )
    model = sp.ExcessHazard.fit(**d, breaks=[cut])
    np.testing.assert_allclose(
        model.params, np.r_[np.exp(oracle.x[:2]), oracle.x[2]], rtol=1e-4
    )
    assert model.neg_ll() == pytest.approx(oracle.fun, abs=1e-6)
    assert model.neg_ll() <= oracle.fun + 1e-9


def test_excess_hazard_recovers_the_truth():
    model = sp.ExcessHazard.fit(**_excess_sample())
    lower, upper = model.param_cb("coef_0")
    assert lower < -0.7 < upper
    lower, upper = model.param_cb("rate_0")
    assert lower < 0.15 < upper
    table = model.summary()
    assert table.loc["coef_0", "exp(coef)"] == pytest.approx(
        np.exp(model.coef[0])
    )


def test_excess_hazard_functions():
    model = sp.ExcessHazard.fit(**_excess_sample(500, seed=2), breaks=3)
    assert model.n_intervals == 3 and len(model.intervals) == 3
    edges = [a for a, _ in model.intervals][1:]
    np.testing.assert_allclose(edges, model.breaks)
    t = np.array([0.5, 1.0, 2.0, 4.0, 7.0, 12.0])
    bounds = np.r_[0.0, model.breaks, np.inf]
    H0 = np.array(
        [np.sum(model.rates * np.diff(np.clip(bounds, 0, ti))) for ti in t]
    )
    for z in ([0.0], [1.0]):
        risk = np.exp(model.coef[0] * z[0])
        np.testing.assert_allclose(model.Hf(t, z), H0 * risk, rtol=1e-12)
        np.testing.assert_allclose(model.sf(t, z) + model.ff(t, z), 1.0)
        np.testing.assert_allclose(
            model.df(t, z), model.hf(t, z) * model.sf(t, z), rtol=1e-12
        )
        p = np.array([0.05, 0.3, 0.6, 0.9])
        np.testing.assert_allclose(model.ff(model.qf(p, z), z), p, rtol=1e-10)
    assert model.sf(-1.0, [1.0]) == 1.0
    with pytest.warns(UserWarning, match="outside"):
        assert np.isnan(model.qf(1.5, [0.0]))


def test_excess_hazard_bounds():
    model = sp.ExcessHazard.fit(**_excess_sample(500, seed=2))
    t = np.array([1.0, 3.0, 6.0])
    sf = model.sf(t, [1.0])
    cb = model.cb(t, [1.0])
    assert np.all((cb[:, 0] < sf) & (sf < cb[:, 1]))
    np.testing.assert_allclose(
        model.cb(t, [1.0], on="ff"), 1 - cb[:, ::-1], rtol=1e-12
    )
    # One interval: log H = log rate + beta z + log t, so its variance is
    # the log rate's, the coefficient's and twice their covariance.
    cov = model.covariance()
    rate = model.rates[0]
    se = np.sqrt(cov[0, 0] / rate**2 + cov[1, 1] + 2 * cov[0, 1] / rate)
    H = model.Hf(t, [1.0])
    z = norm.ppf(0.975)
    np.testing.assert_allclose(
        model.cb(t, [1.0], on="Hf"),
        np.c_[H * np.exp(-z * se), H * np.exp(z * se)],
        rtol=1e-10,
    )


def test_excess_hazard_aliases_a_repeated_column():
    d = _excess_sample(400, seed=4)
    ref = sp.ExcessHazard.fit(**d)
    Z = np.c_[d["Z"], d["Z"]]
    with pytest.warns(UserWarning, match=r"column\(s\) 1 of Z cannot"):
        model = sp.ExcessHazard.fit(**{**d, "Z": Z})
    assert np.isnan(model.coef[1])
    np.testing.assert_array_equal(model.aliased, [1])
    np.testing.assert_allclose(model.coef[0], ref.coef[0], rtol=1e-8)


def test_excess_hazard_warns_without_a_finite_maximum():
    d = _excess_sample(400, seed=4)
    # A group with no events at all: its excess hazard runs to 0.
    z = (d["c"] == 1).astype(float).reshape(-1, 1)
    with pytest.warns(UserWarning, match="No finite maximum"):
        model = sp.ExcessHazard.fit(**{**d, "Z": z})
    assert model.maximum == "no finite maximum"


def test_excess_hazard_refuses_an_interval_without_events():
    with pytest.raises(ValueError, match="has no events"):
        sp.ExcessHazard.fit(**_excess_sample(200), breaks=[7.99])


def test_excess_hazard_units_do_not_matter():
    d = _excess_sample(400, seed=6)
    years = sp.ExcessHazard.fit(**d, breaks=[2.0])
    months = sp.ExcessHazard.fit(
        **{
            **d,
            "x": 12 * d["x"],
            "age": 12 * d["age"],
            "rate_table": _gompertz_table(12.0),
        },
        breaks=[24.0],
    )
    np.testing.assert_allclose(months.rates * 12, years.rates, rtol=1e-6)
    np.testing.assert_allclose(months.coef, years.coef, rtol=1e-6, atol=1e-9)


def test_excess_hazard_round_trips_and_observed_survival():
    d = _excess_sample(400, seed=7)
    df = pd.DataFrame(
        {"t": d["x"], "c": d["c"], "treated": d["Z"][:, 0], "age": d["age"]}
    )
    model = sp.ExcessHazard.fit_from_df(
        df, "t", ["treated"], "age", d["rate_table"], c_col="c"
    )
    assert model.parameter_names == ["rate_0", "treated"]
    t = np.array([1.0, 4.0])
    for restored in (
        sp.from_dict(model.to_dict()),
        pickle.loads(pickle.dumps(model)),
    ):
        np.testing.assert_array_equal(restored.sf(t, [1]), model.sf(t, [1]))
        np.testing.assert_array_equal(restored.cb(t, [1]), model.cb(t, [1]))
        assert restored.aic() == model.aic()
    expected = d["rate_table"].sf(t, 70.0)
    np.testing.assert_allclose(
        model.observed_sf(t, [1], age=70.0),
        expected * model.sf(t, [1]),
        rtol=1e-12,
    )
    draws, rows = model.random(5, [[0], [1]], random_state=3)
    again, _ = model.random(5, [[0], [1]], random_state=3)
    np.testing.assert_array_equal(draws, again)
    assert draws.shape == (10,) and rows.shape == (10, 1)


def test_excess_hazard_needs_its_inputs():
    d = _excess_sample(100)
    with pytest.raises(ValueError, match="rate_table"):
        sp.ExcessHazard.fit(d["x"], d["Z"], c=d["c"], age=d["age"])
    with pytest.raises(ValueError, match="'init' has"):
        sp.ExcessHazard.fit(**d, init=[0.1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sp.ExcessHazard.fit(**d, init=[0.2, -0.5])
