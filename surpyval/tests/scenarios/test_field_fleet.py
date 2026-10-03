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
    site = model.param_cb("beta_0", alpha_ci=0.05)
    load = model.param_cb("beta_1", alpha_ci=0.05)
    assert contains(site, -np.log(COASTAL_FACTOR))
    assert contains(load, LOAD_POWER)
    assert contains(model.param_cb("beta", alpha_ci=0.05), BEARING_BETA)


# ---------------------------------------------------------------------------
# Gaps: what the study needs that the package does not give yet
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="#571: the regression fit_from_df takes no xl_col / xr_col",
)
def test_regression_fit_from_df_takes_interval_columns():
    sp.WeibullAFT.fit_from_df(
        DF,
        x_col="b_xl",
        xr_col="b_xr",
        Z_cols=["coastal", "log_load"],
        tl_col="entry",
    )


@pytest.mark.xfail(
    strict=True,
    reason="#571: ParametricCompetingRisks takes no truncation",
)
def test_competing_risks_with_delayed_entry():
    e = DF.reason.fillna("none").values
    sp.ParametricCompetingRisks.fit(
        x=DF.age.values,
        e=np.where(e == "none", None, e),
        c=(e == "none").astype(int),
        tl=DF.entry.values,
    )


@pytest.mark.xfail(
    strict=True,
    reason="#571: a regression model has no qf (B10 by condition)",
)
def test_b10_by_operating_condition():
    x, c, Z, t = _bearing_regression_data()
    model = sp.WeibullAFT.fit(x=x, Z=Z, c=c, t=t)
    coastal_high = np.array([[1.0, np.log(1.15)]])
    inland = np.array([[0.0, 0.0]])
    assert model.qf(0.1, coastal_high) < model.qf(0.1, inland)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#581: a regression model has no cs, for the at-risk forecast "
        "of the units in service"
    ),
)
def test_next_year_forecast_of_units_in_service():
    x, c, Z, t = _bearing_regression_data()
    model = sp.WeibullAFT.fit(x=x, Z=Z, c=c, t=t)
    live = DF.reason.isna().values
    p = 1 - model.cs(1.0, DF.age.values[live], Z[live])
    assert 0 < p.sum() < live.sum()


@pytest.mark.xfail(
    strict=True,
    reason="#570: fit_best takes no tl / tr (fit and fit_from_df do)",
)
def test_best_distribution_for_the_date_table():
    x, c, n, _ = sp.xcnt_handler(xl=DF.xl, xr=DF.xr)
    model = sp.fit_best(x=x, c=c, n=n, tl=0.0)
    assert model is not None


@pytest.mark.xfail(
    strict=True,
    reason="#576: logrank takes no truncation (delayed entry)",
)
def test_sites_compared_with_delayed_entry():
    sp.logrank(
        x=DF.age.values,
        Z=DF.coastal.values,
        c=(DF.reason != "bearing").astype(int).values,
        tl=DF.entry.values,
    )


@pytest.mark.xfail(
    strict=True,
    reason="#575: the recurrent trend tests refuse delayed entry",
)
def test_calendar_trend_with_records_from_the_cmms_start():
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
    model = CrowAMSAA.fit(x=x, i=i, c=c, tl=tl)
    assert model.trend_test().trend == "none"
