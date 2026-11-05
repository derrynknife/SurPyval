"""The distributional regression family of ``diagnose()``: its statistics
against stored R results, its checks on data built to pass or to fail
each of them, the advice each failure calls for, and its refusals.

The R values (``scripts/reference/reference_r_distributional_diagnostics.R``,
stored in ``tests/reference/data/r_distributional_diagnostics.json``) are
flexsurv's (``flexsurvreg`` with ``anc=``, covariates on the Weibull's
scale and shape): the likelihood-ratio statistics of a quadratic term on
each parameter, the AIC of the other distributions and the Cox-Snell
residuals (``coxsnell_flexsurvreg``), with and without delayed entry; and
gamlss's normalised quantile residuals of a log-normal fit with
covariates on ``mu`` and ``sigma``. The statistics are differences of
maximised log-likelihoods, which agree with R's to the precision of the
two maximisations: ``rtol=1e-6``.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm

import surpyval as sp
from surpyval.diagnostics import Diagnosis, distributional, ph_aft
from surpyval.tests.reference._data import fixture, values
from surpyval.utils.weights import WeightedRefusal

FITTED = dict(rtol=1e-6, atol=1e-8)
REF = "r_distributional_diagnostics"


def _quiet(fit, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit(*args, **kwargs)


def _rows(d, check, verdict=None):
    table = d.table[d.table["check"] == check]
    if verdict is not None:
        table = table[table["verdict"] == verdict]
    return table


def _stat(d, check, parameter, covariate):
    t = _rows(d, check)
    keep = (t["parameter"] == parameter) & (t["covariates"] == (covariate,))
    return t.loc[keep, "statistic"].iloc[0]


# ---------------------------------------------------------------------------
# Against R
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def lung():
    d = fixture("lung")
    keep = ~np.isnan(d["ph_ecog"])
    return pd.DataFrame(
        {k: d[k][keep] for k in ("time", "c", "age", "sex", "ph_ecog")}
    )


@pytest.fixture(scope="module")
def lung_weibull(lung):
    model = _quiet(
        sp.WeibullDistributional.fit_from_df,
        lung,
        x_col="time",
        c_col="c",
        Z_cols=["age", "sex", "ph_ecog"],
    )
    return model.diagnose()


def test_the_quadratic_terms_match_flexsurv(lung_weibull):
    r = values(REF, "lung_weibull")
    assert_allclose(
        -lung_weibull._context.model._neg_ll, r["loglik"], rtol=1e-10
    )
    for param in ("alpha", "beta"):
        for v in ("age", "ph_ecog"):
            assert_allclose(
                _stat(lung_weibull, "functional_form", param, v),
                r[f"square_{param}_{v}"],
                **FITTED,
            )
    # sex has two values: not tested, and the note says so
    form = _rows(lung_weibull, "functional_form")
    assert ("sex",) not in set(form["covariates"])
    assert any("sex has two" in n for n in lung_weibull.notes)


def test_the_cox_snell_residuals_match_flexsurv(lung_weibull):
    r = values(REF, "lung_weibull")
    ours = ph_aft._residual_data(lung_weibull._context, "PH")["lo"]
    # At the model's parameters, which the two fits find to about 1e-5
    # (relative) of each other in the likelihood's flattest directions
    assert_allclose(np.sort(ours), np.sort(r["cox_snell"]), rtol=1e-4)


def test_the_aic_comparison_matches_flexsurv(lung_weibull):
    r = values(REF, "lung_aic")
    aic = {
        c["name"]: c["aic"]
        for c in distributional._aic_candidates(lung_weibull._context)
    }
    for ours, theirs in [
        ("Weibull", "weibull"),
        ("LogLogistic", "llogis"),
        ("Exponential", "exp"),
    ]:
        assert_allclose(aic[ours], r[theirs], **FITTED)
    # flexsurv's log-normal and gamma searches stop short of the maximum
    # (by 0.03 and 0.23 in AIC); ours reach a higher likelihood
    assert aic["LogNormal"] <= r["lnorm"] + 1e-6
    assert aic["Gamma"] <= r["gamma"] + 1e-6
    row = _rows(lung_weibull, "distribution").iloc[0]
    assert row["verdict"] == "ok"
    assert_allclose(row["statistic"], aic["Weibull"] - aic["Gamma"])


def test_cox_snell_residuals_with_delayed_entry_match_flexsurv():
    lt = fixture("left_truncation")
    model = _quiet(
        sp.WeibullDistributional.fit, lt["x"], lt["z"], c=lt["c"], tl=lt["tl"]
    )
    r = values(REF, "left_truncation_weibull")
    assert_allclose(-model._neg_ll, r["loglik"], rtol=1e-10)
    d = model.diagnose()
    res = ph_aft._residual_data(d._context, "PH")
    entry = np.where(np.isfinite(res["tl"]), res["tl"], 0.0)
    # flexsurv's Cox-Snell residual is H(x) - H(tl)
    assert_allclose(
        np.sort(res["lo"] - entry), np.sort(r["cox_snell"]), rtol=1e-4
    )
    assert_allclose(np.sort(entry), np.sort(r["entry"]), rtol=1e-4)


def test_quantile_residuals_match_gamlss(lung):
    failed = lung[lung["c"] == 0]
    model = _quiet(
        sp.LogNormalDistributional.fit_from_df,
        failed,
        x_col="time",
        Z_cols=["age", "sex", "ph_ecog"],
        covariates={"mu": ["age", "sex", "ph_ecog"], "sigma": ["age", "sex"]},
    )
    r = values(REF, "lung_failed_lognormal")
    assert_allclose(-model._neg_ll, r["loglik"], rtol=1e-10)
    d = model.diagnose()
    H = ph_aft._residual_data(d._context, "PH")["lo"]
    quantile = norm.ppf(-np.expm1(-H))
    assert_allclose(
        np.sort(quantile), np.sort(r["quantile_residuals"]), atol=1e-5
    )
    # sigma holds no ph_ecog, so ph_ecog is not tested on it
    form = _rows(d, "functional_form")
    pairs = set(zip(form["parameter"], form["covariates"]))
    assert pairs == {
        ("mu", ("age",)),
        ("mu", ("ph_ecog",)),
        ("sigma", ("age",)),
    }


# ---------------------------------------------------------------------------
# Residuals and influence
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def simulated():
    rng = np.random.default_rng(11)
    load = rng.uniform(0, 1, 200)
    x = 10 * np.exp(-0.5 * load) * rng.weibull(1.2 * np.exp(load))
    cens = rng.uniform(0, 25, 200)
    c = (cens < x).astype(int)
    return np.minimum(x, cens), load, c


def test_the_scale_score_residual_is_the_martingale_one(simulated):
    x, load, c = simulated
    model = _quiet(sp.WeibullDistributional.fit, x, load, c=c)
    ctx = model.diagnose()._context
    alpha, beta = model.parameters_at(load[:, None]).T
    # model.data is sorted: compare on its own rows
    data = model.data
    alpha, beta = model.parameters_at(np.asarray(data.Z)).T
    H = (data.x / alpha) ** beta
    delta = (data.c == 0).astype(float)
    score = distributional._score_residuals(ctx)
    # d log L / d log(alpha) = -beta (delta - H): -beta times the
    # martingale residual
    assert_allclose(score[:, 0], -beta * (delta - H), rtol=1e-8, atol=1e-10)


def test_dfbeta_is_close_to_the_refit_without_the_unit(simulated):
    x, load, c = simulated
    model = _quiet(sp.WeibullDistributional.fit, x, load, c=c)
    d = model.diagnose()
    dfbeta, se = distributional._dfbeta(d._context)
    data = model.data
    rows = [0, 50, 100, 150, *np.argsort(-np.abs(dfbeta[:, 1]))[:3]]
    for i in rows:
        keep = np.arange(len(data.x)) != i
        refit = _quiet(
            sp.WeibullDistributional.fit,
            data.x[keep],
            np.asarray(data.Z)[keep],
            c=data.c[keep],
        )
        change = model.params[2:] - refit.params[2:]
        # The one-step approximation: to a few per cent for a typical
        # row, and within 0.15 standard errors (the screen's limit is 1)
        # for the most influential ones, where the likelihood is least
        # quadratic
        assert np.all(np.abs(dfbeta[i] - change) < 0.15 * se), i


# ---------------------------------------------------------------------------
# Data built to pass or to fail
# ---------------------------------------------------------------------------
def test_a_correct_model_passes(simulated):
    x, load, c = simulated
    d = _quiet(sp.WeibullDistributional.fit, x, load, c=c).diagnose()
    assert d.ok
    assert set(d.table["verdict"]) == {"ok"}
    assert not d.advice


def test_a_non_linear_effect_on_the_shape_fails_functional_form():
    rng = np.random.default_rng(5)
    load = rng.uniform(0, 1, 300)
    shape = 1.2 * np.exp(12 * (load - 0.5) ** 2)
    x = 10 * np.exp(-0.5 * load) * rng.weibull(shape)
    d = _quiet(sp.WeibullDistributional.fit, x, load).diagnose()
    form = _rows(d, "functional_form")
    # The shape's square is by far the strongest; the scale's form may
    # fail too, as a shape that bends with load also moves the
    # quantiles the scale sets
    worst = form.loc[form["statistic"].idxmax()]
    assert worst["parameter"] == "beta" and worst["verdict"] == "fail"
    (advice,) = (
        a for a in d.advice if a.check == "functional_form" and '"beta"' in a
    )
    assert 'formulas={"beta": "np.log(coef_0)' in str(advice)
    assert advice.features == ("WeibullDistributional.fit_from_df(formulas=)",)


def test_the_wrong_distribution_is_flagged():
    rng = np.random.default_rng(6)
    load = rng.uniform(0, 1, 300)
    x = np.exp(2 - 0.5 * load + 0.8 * rng.normal(size=300))
    d = _quiet(sp.WeibullDistributional.fit, x, load).diagnose()
    row = _rows(d, "distribution").iloc[0]
    assert row["verdict"] == "warn"
    assert "LogNormalDistributional fits better" in row["detail"]
    (advice,) = (a for a in d.advice if a.check == "distribution")
    assert advice.features == ("LogNormalDistributional.fit(covariates=)",)


def test_an_influential_unit_is_flagged():
    rng = np.random.default_rng(7)
    load = rng.uniform(0, 1, 40)
    x = 10 * np.exp(-0.5 * load) * rng.weibull(2.0, 40)
    # A failure far earlier than the model allows at its load. (A unit the
    # model can fit by bending a coefficient -- far out on load and long
    # lived -- is masked: its own score is small at the fit it pulled,
    # as for every one-step dfbeta.)
    load[0], x[0] = 1.0, 0.05
    d = _quiet(sp.WeibullDistributional.fit, x, load).diagnose()
    warned = _rows(d, "influence", "warn")
    assert len(warned)
    assert any(a.check == "influence" and "Row" in a for a in d.advice)


# ---------------------------------------------------------------------------
# Every kind of data, the options and the report
# ---------------------------------------------------------------------------
def test_interval_and_left_censored_data_are_diagnosed():
    rng = np.random.default_rng(8)
    load = rng.uniform(0, 1, 200)
    x = 10 * np.exp(-0.5 * load) * rng.weibull(1.5 * np.exp(0.5 * load))
    lo, hi = np.floor(x), np.floor(x) + 1
    c = np.where(lo == 0, -1, 2)
    xx = np.column_stack([np.where(c == -1, hi, lo), hi])
    d = _quiet(sp.WeibullDistributional.fit, xx, load, c=c).diagnose()
    assert np.isfinite(_rows(d, "functional_form")["p"]).all()
    assert _rows(d, "influence")["statistic"].notna().all()


def test_columns_parameters_and_fixed(simulated):
    x, load, c = simulated
    Z = np.column_stack([load, np.random.default_rng(1).normal(size=200)])
    model = _quiet(
        sp.WeibullDistributional.fit,
        x,
        Z,
        c=c,
        covariates={"alpha": [0, 1], "beta": [0]},
        fixed={"beta_coef_0": 1.0},
    )
    d = model.diagnose(columns=["coef_1"])
    assert set(d.table["covariates"]) - {()} == {("coef_1",)}
    assert set(_rows(d, "influence")["parameter"]) == {"alpha"}
    # A fixed parameter leaves the AIC comparison out, with a note
    assert _rows(d, "distribution").empty
    assert any("beta_coef_0 fixed" in n for n in d.notes)
    # the fixed coefficient has no influence row
    d = model.diagnose()
    infl = _rows(d, "influence")
    assert set(zip(infl["parameter"], infl["covariates"])) == {
        ("alpha", ("coef_0",)),
        ("alpha", ("coef_1",)),
    }


def test_report_round_trips_and_plots(simulated, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    x, load, c = simulated
    d = _quiet(sp.WeibullDistributional.fit, x, load, c=c).diagnose()
    assert isinstance(d, Diagnosis)
    text = json.dumps(d.to_dict(), allow_nan=False)
    assert '"parameter"' in text
    fig = d.plot(n_boot=0)
    # Cox-Snell, AIC, two functional forms, two influences
    assert len(fig.axes) == 6
    fig = d.plot(which="functional_form", n_boot=0)
    assert len(fig.axes) == 2


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def test_a_weighted_fit_is_refused(simulated):
    x, load, c = simulated
    w = np.random.default_rng(2).uniform(0.5, 2, 200)
    model = _quiet(sp.WeibullDistributional.fit, x, load, c=c, w=w)
    with pytest.raises(WeightedRefusal, match="without w"):
        model.diagnose()


def test_a_penalised_fit_is_refused(simulated):
    x, load, c = simulated
    model = _quiet(
        sp.WeibullDistributional.fit_penalised, x, load, c=c, penalizer=0.1
    )
    with pytest.raises(ValueError, match="penalised"):
        model.diagnose()


def test_a_restored_model_is_refused(simulated):
    x, load, c = simulated
    model = _quiet(sp.WeibullDistributional.fit, x, load, c=c)
    restored = type(model).from_dict(model.to_dict())
    with pytest.raises(ValueError, match="from_dict"):
        restored.diagnose()


def test_aliased_coefficients_are_refused(simulated):
    x, load, c = simulated
    Z = np.column_stack([load, 2 * load])
    model = _quiet(sp.WeibullDistributional.fit, x, Z, c=c)
    with pytest.raises(ValueError, match="aliased"):
        model.diagnose()
