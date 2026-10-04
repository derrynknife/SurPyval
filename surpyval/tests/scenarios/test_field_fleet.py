"""Card: gearbox lives from a maintenance system's records.

Persona: a reliability engineer at a two-site wind farm with an export
of gearbox removals from the CMMS (computerised maintenance management
system). Questions, in the order of a field-data study (Meeker and
Escobar ch. 3-4 and 17; Abernethy ch. 2 and 5):

1. What is a gearbox's life, from the date table as it is: main-bearing
   failures found at six-monthly vibration surveys (interval censored),
   gear failures that trip the turbine (exact), gearboxes still running
   (right censored), and records only from the day the CMMS went live
   (left truncated)?
2. What is each failure mode's life, the other mode a suspension?
3. Do the coastal site and the turbine's load shorten the bearing life,
   and by how much (B10 by condition)?
4. Is the fleet's removal rate trending, in calendar time?
5. How many removals next year, and which turbines?

Truth: bearing Weibull(eta = 9 years x 0.7 at the coastal site / load
index squared, beta = 2.5); gear Weibull(15 years, 1.3), independent.
"""

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval.tests.scenarios._oracles import best_of, contains, weibull_sf

DAY = 365.25
BEARING_BETA, GEAR_ETA, GEAR_BETA = 2.5, 15.0, 1.3
COASTAL_FACTOR, LOAD_POWER = 0.7, 2.0
SURVEY = 0.5  # years between vibration surveys


def _cmms():
    """The CMMS export as the engineer would prepare it, in years: one
    row per gearbox with its age at the last OK survey and at removal
    (or the data cut-off), the age at which the records start, its site
    and load index, and the removal reason."""
    rng = np.random.default_rng(11)
    start, live, end = 0.0, 5.0, 12.75  # years from the first install
    rows = []
    for site, n_turbines, factor in [
        ("coastal", 45, COASTAL_FACTOR),
        ("inland", 35, 1.0),
    ]:
        for k in range(n_turbines):
            commissioned = start + rng.uniform(0, 8)
            load = rng.normal(1.0, 0.1)
            t = commissioned
            while t < end:
                bearing = 9 * factor * rng.weibull(BEARING_BETA) / load**2
                gear = GEAR_ETA * rng.weibull(GEAR_BETA)
                found = np.ceil(bearing / SURVEY) * SURVEY
                life = min(found, gear)
                mode = "bearing" if found < gear else "gear"
                removed = t + life
                if removed >= live:  # what the CMMS can see
                    seen = removed < end
                    rows.append(
                        dict(
                            turbine=f"{site}-{k}",
                            coastal=float(site == "coastal"),
                            log_load=np.log(load),
                            reason=mode if seen else None,
                            age=min(removed, end) - t,
                            last_ok=(life - SURVEY) if seen else np.nan,
                            entry=max(0.0, live - t),
                        )
                    )
                t = removed + rng.uniform(10, 60) / DAY
    df = pd.DataFrame(rows)
    bearing = df.reason == "bearing"
    # A bearing failure is only *found* at a survey, so a record is
    # selected on its detection time: its truncation point is the last
    # survey before the records start, at most its last OK survey.
    df.loc[bearing, "entry"] = np.minimum(
        df.loc[bearing, "entry"], df.loc[bearing, "last_ok"]
    )
    df["xl"] = np.where(bearing, df.last_ok, df.age)
    df["xr"] = np.where(df.reason.isna(), np.inf, df.age)
    df["b_xl"] = np.where(bearing, df.last_ok, df.age)
    df["b_xr"] = np.where(bearing, df.age, np.inf)
    df["g_xr"] = np.where(df.reason == "gear", df.age, np.inf)
    return df


DF = _cmms()


def _truncated_interval_neg_ll(alpha, beta, xl, xr, tl):
    """Weibull negative log-likelihood of [xl, xr] (xl == xr exact,
    xr = inf right censored), each row left truncated at tl."""
    if alpha <= 0 or beta <= 0:
        return np.inf
    exact = xl == xr
    s_tl = weibull_sf(tl, alpha, beta)
    xe = np.where(exact, xl, 1.0)
    dens = (
        (beta / alpha)
        * (xe / alpha) ** (beta - 1)
        * weibull_sf(xe, alpha, beta)
    )
    lik = np.where(
        exact,
        dens,
        weibull_sf(xl, alpha, beta)
        - np.where(
            np.isinf(xr),
            0.0,
            weibull_sf(np.where(np.isinf(xr), 1.0, xr), alpha, beta),
        ),
    )
    return -(np.log(lik) - np.log(s_tl)).sum()


def test_life_from_the_date_table_is_the_mle():
    model = sp.Weibull.fit_from_df(
        DF, xl_col="xl", xr_col="xr", tl_col="entry"
    )
    ref = best_of(
        lambda p: _truncated_interval_neg_ll(
            np.exp(p[0]),
            np.exp(p[1]),
            DF.xl.values,
            DF.xr.values,
            DF.entry.values,
        ),
        [np.log(model.params), [np.log(5.0), 0.0]],
    )
    assert model.maximum == "verified"
    assert model.neg_ll() == pytest.approx(ref.fun, abs=1e-6)
    np.testing.assert_allclose(model.params, np.exp(ref.x), rtol=1e-4)


def test_gear_mode_recovers_its_life():
    # The gear mode is the same for every turbine, so its marginal life is
    # the Weibull it was drawn from, with the bearing removals as
    # suspensions. (The bearing mode's marginal mixes the sites' and the
    # loads' scales, so its truth is checked through the regression.)
    gear = sp.Weibull.fit_from_df(
        DF.assign(g_xl=DF.age), xl_col="g_xl", xr_col="g_xr", tl_col="entry"
    )
    assert gear.maximum == "verified"
    assert contains(gear.param_cb("beta", alpha_ci=0.05), GEAR_BETA)
    assert contains(gear.param_cb("alpha", alpha_ci=0.05), GEAR_ETA)


def _bearing_regression_data():
    c = np.where(DF.reason == "bearing", 2, 1)
    x = np.column_stack([DF.b_xl, np.where(c == 2, DF.b_xr, DF.b_xl)])
    Z = DF[["coastal", "log_load"]].values
    t = np.column_stack([DF.entry, np.full(len(DF), np.inf)])
    return x, c, Z, t


def test_site_and_load_effects():
    x, c, Z, t = _bearing_regression_data()
    model = sp.WeibullAFT.fit(x=x, Z=Z, c=c, t=t)
    assert model.maximum == "verified"
    # The coefficients accelerate time: life / exp(Z beta).
    site = model.param_cb("coef_0", alpha_ci=0.05)
    load = model.param_cb("coef_1", alpha_ci=0.05)
    assert contains(site, -np.log(COASTAL_FACTOR))
    assert contains(load, LOAD_POWER)
    assert contains(model.param_cb("beta", alpha_ci=0.05), BEARING_BETA)


# ---------------------------------------------------------------------------
# Questions 3-5 and the field-data paths. Each was a gap when the card was
# written (#570, #571, #575, #576, #581); each is now answered with the
# API its fix chose.
# ---------------------------------------------------------------------------
def test_regression_fit_from_df_takes_interval_columns():
    # #571: the date table goes in as it is, its interval ends as columns,
    # and gives the model the arrays do (with each coefficient named by
    # its column).
    x, c, Z, t = _bearing_regression_data()
    arrays = sp.WeibullAFT.fit(x=x, Z=Z, c=c, t=t)
    model = sp.WeibullAFT.fit_from_df(
        DF,
        xl_col="b_xl",
        xr_col="b_xr",
        Z_cols=["coastal", "log_load"],
        tl_col="entry",
    )
    assert model.maximum == "verified"
    assert model.neg_ll() == pytest.approx(arrays.neg_ll(), abs=1e-8)
    np.testing.assert_allclose(model.params, arrays.params, rtol=1e-6)
    assert contains(
        model.param_cb("coastal", alpha_ci=0.05), -np.log(COASTAL_FACTOR)
    )


def test_competing_risks_with_delayed_entry():
    # #571: with independent risks the likelihood factorises by cause, so
    # each cause's fit is the single-distribution fit with the other
    # cause's removals as suspensions, truncated at the same entry.
    e = DF.reason.fillna("none").values
    model = sp.ParametricCompetingRisks.fit(
        x=DF.age.values,
        e=np.where(e == "none", None, e),
        c=(e == "none").astype(int),
        tl=DF.entry.values,
    )
    gear = sp.Weibull.fit(
        x=DF.age.values,
        c=(DF.reason != "gear").astype(int).values,
        tl=DF.entry.values,
    )
    np.testing.assert_allclose(
        model.models["gear"].params, gear.params, rtol=1e-6
    )
    assert contains(gear.param_cb("beta", alpha_ci=0.05), GEAR_BETA)
    assert contains(gear.param_cb("alpha", alpha_ci=0.05), GEAR_ETA)


def test_b10_by_operating_condition():
    # #571: B10 by condition from the regression's own qf, and its bounds
    # cover the true B10 at each condition.
    x, c, Z, t = _bearing_regression_data()
    model = sp.WeibullAFT.fit(x=x, Z=Z, c=c, t=t)
    load = 1.15
    conditions = {
        "coastal, high load": (
            np.array([[1.0, np.log(load)]]),
            9 * COASTAL_FACTOR / load**LOAD_POWER,
        ),
        "inland, nominal load": (np.array([[0.0, 0.0]]), 9.0),
    }
    b10 = {}
    for name, (z, eta) in conditions.items():
        b10[name] = np.ravel(model.qf(0.1, z))[0]
        assert np.ravel(model.ff(b10[name], z))[0] == pytest.approx(0.1)
        truth = eta * (-np.log(0.9)) ** (1 / BEARING_BETA)
        assert contains(model.quantile_cb(0.1, z, alpha_ci=0.05), truth)
    assert b10["coastal, high load"] < b10["inland, nominal load"]


def test_next_year_forecast_of_units_in_service():
    # #581: for each gearbox still running, at its own age and condition,
    # the chance of a bearing removal in the next year is
    # 1 - S(age + 1 | Z) / S(age | Z); the expected count is their sum.
    x, c, Z, t = _bearing_regression_data()
    model = sp.WeibullAFT.fit(x=x, Z=Z, c=c, t=t)
    live = DF.reason.isna().values
    age, z = DF.age.values[live], Z[live]
    p = 1 - np.asarray(model.cs(1.0, age, z), dtype=float)
    alpha, beta = model.params[:2]
    eta = alpha / np.exp(z @ model.params[2:])
    expected = 1 - weibull_sf(age + 1.0, eta, beta) / weibull_sf(
        age, eta, beta
    )
    np.testing.assert_allclose(p, expected, rtol=1e-8)
    # The fleet's count, with its prediction interval, is forecast().
    fleet = sp.forecast(model, age=age, Z=z, horizon=1.0)
    np.testing.assert_allclose(np.ravel(fleet.probability), p, rtol=1e-8)
    assert fleet.expected[0] == pytest.approx(p.sum(), rel=1e-8)
    assert fleet.lower[0] <= p.sum() <= fleet.upper[0] < live.sum()


def test_best_distribution_for_the_date_table():
    # #570: fit_best takes the interval ends and the truncation as fit
    # does; it chooses the candidate with the least AIC, and its Weibull
    # is the MLE of the first question.
    best, table = sp.fit_best(
        xl=DF.xl, xr=DF.xr, tl=DF.entry, return_table=True
    )
    ranked = table[table.status.isin(["chosen", "ranked"])]
    assert best.dist.name == ranked.loc[ranked.aic.idxmin(), "model"]
    weibull = sp.Weibull.fit_from_df(
        DF, xl_col="xl", xr_col="xr", tl_col="entry"
    )
    row = table[table.model == "Weibull"].iloc[0]
    assert row.aic == pytest.approx(weibull.aic(), rel=1e-8)


def _logrank_statistic(x, group, event, entry):
    """The two-sample log-rank statistic written out, a unit at risk at
    t when entry < t <= x."""
    o1 = e1 = v = 0.0
    for t in np.unique(x[event]):
        risk = (entry < t) & (x >= t)
        n, n1 = risk.sum(), (risk & group).sum()
        d = (event & (x == t)).sum()
        o1 += (event & (x == t) & group).sum()
        e1 += d * n1 / n
        if n > 1:
            v += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    return (o1 - e1) ** 2 / v


def test_sites_compared_with_delayed_entry():
    # #576: logrank takes the truncation, so each risk set holds only the
    # gearboxes the records can see; the coastal site's shorter bearing
    # life is detected.
    result = sp.logrank(
        x=DF.age.values,
        Z=DF.coastal.values,
        c=(DF.reason != "bearing").astype(int).values,
        tl=DF.entry.values,
    )
    expected = _logrank_statistic(
        DF.age.values,
        DF.coastal.values == 1,
        (DF.reason == "bearing").values,
        DF.entry.values,
    )
    assert result.statistic == pytest.approx(expected, rel=1e-10)
    assert result.p_value < 0.05


def test_calendar_trend_with_records_from_the_cmms_start():
    # #575: the Laplace test over each turbine's own window [entry, end].
    from surpyval.recurrent import CrowAMSAA

    rng = np.random.default_rng(3)
    # Turbines watched from their own entry into the records, each with
    # removals in calendar years: a homogeneous rate, so no trend.
    x, i, c, tl = [], [], [], []
    for k in range(40):
        entry = rng.uniform(0, 3)
        events = entry + np.cumsum(rng.exponential(5.0, 10))
        events = events[events < 8.0]
        x += list(events) + [8.0]
        i += [k] * (events.size + 1)
        c += [0] * events.size + [1]
        tl += [entry] * (events.size + 1)
    result = CrowAMSAA.fit(x=x, i=i, c=c, tl=tl).trend_test()
    assert result.trend == "none"
    x, i, c, tl = map(np.asarray, (x, i, c, tl))
    num = den = 0.0
    for k in np.unique(i):
        rows = i == k
        events, start, end = x[rows & (c == 0)], tl[rows][0], x[rows].max()
        num += events.sum() - events.size * (start + end) / 2
        den += events.size * (end - start) ** 2 / 12
    assert result.statistic == pytest.approx(num / np.sqrt(den), rel=1e-10)
