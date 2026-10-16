"""The parametric proportional hazards and accelerated failure time
families of ``diagnose()`` (#869): their statistics against stored R
results, their checks on data built to pass or to fail each of them, the
advice each failure calls for, and their refusals.

The R values (``scripts/reference/reference_r_ph_aft.R``, stored in
``tests/reference/data/r_ph_aft.json``) are survreg's likelihood-ratio
statistics of a quadratic term and of a scale per stratum, its AIC, its
dfbeta and residuals, and a Weibull PH step test computed as a Poisson
model of the split follow-up (independently of survreg). The statistics
are differences of maximised log-likelihoods, which agree with R's to the
precision of the two maximisations: ``rtol=1e-6``.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.diagnostics import Diagnosis, ph_aft
from surpyval.tests.reference._data import fixture, values

FITTED = dict(rtol=1e-6, atol=1e-8)


def _rows(d, check, verdict=None):
    table = d.table[d.table["check"] == check]
    if verdict is not None:
        table = table[table["verdict"] == verdict]
    return table


def _stat(d, check, covariates):
    """The statistic of the first row of ``check`` about ``covariates``
    (with one covariate, its own row comes before the GLOBAL one)."""
    table = _rows(d, check)
    return table.loc[table["covariates"] == covariates, "statistic"].iloc[0]


def _quiet(fit, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit(*args, **kwargs)


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


def _lung_fit(fitter, lung):
    return fitter.fit_from_df(
        lung, x_col="time", c_col="c", Z_cols=["age", "sex", "ph_ecog"]
    )


@pytest.fixture(scope="module")
def lung_weibull_ph(lung):
    return _lung_fit(sp.WeibullPH, lung).diagnose()


@pytest.fixture(scope="module")
def lung_weibull_aft(lung):
    return _lung_fit(sp.WeibullAFT, lung).diagnose()


@pytest.fixture(scope="module")
def lung_lognormal_aft(lung):
    return _lung_fit(sp.LogNormalAFT, lung).diagnose()


def test_quadratic_terms_match_survreg(
    lung_weibull_ph, lung_weibull_aft, lung_lognormal_aft
):
    weibull = values("r_ph_aft", "lung_weibull")
    lognormal = values("r_ph_aft", "lung_lognormal")
    # A Weibull PH model is a Weibull AFT model: the same statistic
    for d in (lung_weibull_ph, lung_weibull_aft):
        for v in ("age", "ph_ecog"):
            assert_allclose(
                _stat(d, "functional_form", (v,)),
                weibull["square_" + v],
                **FITTED,
            )
    for v in ("age", "ph_ecog"):
        assert_allclose(
            _stat(lung_lognormal_aft, "functional_form", (v,)),
            lognormal["square_" + v],
            **FITTED,
        )


def test_a_shape_per_level_matches_survreg_strata(
    lung_weibull_aft, lung_lognormal_aft
):
    for d, ref, invert in (
        (lung_weibull_aft, "lung_weibull", True),
        (lung_lognormal_aft, "lung_lognormal", False),
    ):
        r = values("r_ph_aft", ref)
        assert_allclose(
            _stat(d, "common_shape", ("sex",)), r["shape_sex"], **FITTED
        )
        # The shapes of the larger model are R's per-stratum scales
        # (the Weibull's shape is 1 / scale)
        ctx = d._context
        (factor,) = [f for f in ph_aft._factors(ctx) if f["name"] == "sex"]
        res = ph_aft._shape_test(ctx, factor)
        s = ctx.model.distribution_param_map["beta" if invert else "sigma"]
        theta = res["fit"].theta
        k = ctx.model.params.size
        shapes = theta[s] * np.exp(np.concatenate([[0.0], theta[k:]]))
        expected = r["shape_sex_scales"]
        assert_allclose(
            shapes, 1 / expected if invert else expected, rtol=1e-5
        )
    # ph_ecog's level 3 has one unit: not computed, and says why
    row = _rows(lung_weibull_aft, "common_shape")
    row = row[row["covariates"] == ("ph_ecog",)].iloc[0]
    assert np.isnan(row["p"]) and row["verdict"] == "warn"
    assert "ph_ecog=3" in row["detail"] and "two distinct" in row["detail"]


@pytest.mark.parametrize("dist", ["weibull", "lognormal"])
def test_a_formula_factor_and_interval_censoring_match_survreg(dist):
    fitter = sp.WeibullAFT if dist == "weibull" else sp.LogNormalAFT
    v = fixture("veteran")
    df = pd.DataFrame({k: v[k] for k in v})
    df["c"] = 1 - df["status"]
    model = fitter.fit_from_df(
        df, x_col="time", c_col="c", formula="C(celltype) + karno"
    )
    d = model.diagnose()
    r = values("r_ph_aft", "veteran_" + dist)
    assert_allclose(model.aic(), r["aic"], **FITTED)
    row = _rows(d, "common_shape").iloc[0]
    assert len(row["covariates"]) == 3 and row["df"] == 3
    assert_allclose(row["statistic"], r["shape_celltype"], **FITTED)
    assert_allclose(
        _stat(d, "functional_form", ("karno",)), r["square_karno"], **FITTED
    )
    assert "celltype reference" in row["detail"]
    assert "celltype=2" in row["detail"]

    iv = fixture("interval")
    x, c = [], []
    for left, right in zip(iv["left"], iv["right"]):
        if np.isnan(right):
            x.append([left, left])
            c.append(1)
        elif left == 0:
            x.append([right, right])
            c.append(-1)
        elif left == right:
            x.append([left, left])
            c.append(0)
        else:
            x.append([left, right])
            c.append(2)
    model = fitter.fit(np.array(x), iv["z"][:, None], c=np.array(c))
    d = model.diagnose()
    r = values("r_ph_aft", "interval_" + dist)
    assert_allclose(model.aic(), r["aic"], **FITTED)
    assert_allclose(
        _stat(d, "common_shape", ("coef_0",)), r["shape_z"], **FITTED
    )


def test_the_better_error_distribution_and_ph_model_are_advised(
    lung_lognormal_aft,
):
    d = lung_lognormal_aft
    rows = _rows(d, "error_distribution")
    assert (rows["verdict"] == "warn").all() and len(rows) == 2
    assert [a.features for a in d.advice] == [
        ("WeibullAFT.fit",),
        ("LogNormalPH.fit",),
    ]
    assert d.advice[0].startswith("Another error distribution fits")
    # The gap is survreg's: the log-normal's AIC less the Weibull's
    gap = values("r_ph_aft", "lung_lognormal")["aic"] - (
        values("r_ph_aft", "lung_weibull")["aic"]
    )
    assert_allclose(rows["statistic"].iloc[0], gap, **FITTED)


def test_the_aic_comparison_refits_as_survreg(lung_weibull_aft):
    ctx = lung_weibull_aft._context
    candidates = {c["label"]: c for c in ph_aft._aic_candidates(ctx, "AFT")}
    for label, ref in (
        ("WeibullAFT", "lung_weibull"),
        ("ExponentialAFT", "lung_exponential"),
        ("LogNormalAFT", "lung_lognormal"),
    ):
        assert_allclose(
            candidates[label]["aic"], values("r_ph_aft", ref)["aic"], **FITTED
        )
    assert candidates["WeibullAFT"]["status"] == "fitted"
    assert set(candidates) == {b + "AFT" for b in ph_aft.BASELINES}


@pytest.mark.parametrize(
    "fitter, ref",
    [
        (sp.WeibullPH, "step_lung_weibull"),
        (sp.ExponentialPH, "step_lung_exponential"),
    ],
)
def test_the_step_test_matches_a_poisson_model_of_the_split_data(
    lung, fitter, ref
):
    d = _lung_fit(fitter, lung).diagnose()
    r = values("r_ph_aft", ref)
    for v in ("age", "sex", "ph_ecog"):
        assert_allclose(
            _stat(d, "proportional_hazards", (v,)), r["step_" + v], **FITTED
        )
    assert_allclose(
        _stat(d, "proportional_hazards", ("age", "sex", "ph_ecog")),
        r["step_global"],
        **FITTED,
    )
    assert ph_aft._cut(d._context) == r["tau"]
    # The Poisson model without the steps is the fitted model
    assert_allclose(-d._context.model._neg_ll, r["loglik_fitted"], rtol=1e-9)


def test_the_step_test_with_delayed_entry_matches_r():
    lt = fixture("left_truncation")
    model = sp.WeibullPH.fit(lt["x"], lt["z"][:, None], c=lt["c"], tl=lt["tl"])
    d = model.diagnose()
    r = values("r_ph_aft", "step_left_truncation_weibull")
    assert_allclose(
        _stat(d, "proportional_hazards", ("coef_0",)), r["step_z"], **FITTED
    )
    assert ph_aft._cut(d._context) == r["tau"]
    assert _rows(d, "influence")["statistic"].notna().all()


def _row_order(model, columns):
    """The rows of the model's (sorted) data in the order of R's rows:
    each set of keys sorted the same way."""
    data = model.data
    keys = np.column_stack([data.x, data.c, data.Z])
    return np.lexsort(keys.T[::-1]), np.lexsort(columns.T[::-1])


def test_dfbeta_and_residuals_match_survreg(lung, lung_weibull_ph):
    ctx = lung_weibull_ph._context
    model = ctx.model
    r = values("r_ph_aft", "lung_weibull")
    # survreg's dfbeta is of (intercept, coefficients, log scale); the PH
    # coefficients are -coef / scale, so their one-step change is
    # -d coef / scale + coef / scale * d log(scale).
    coef, scale = r["coef"][1:], r["scale"]
    raw = r["dfbeta"]
    expected = -raw[:, 1:4] / scale + np.outer(raw[:, 4], coef / scale)
    dfbeta, se = ph_aft._dfbeta(ctx)
    ours, theirs = _row_order(
        model,
        lung[["time", "c", "age", "sex", "ph_ecog"]].to_numpy(dtype=float),
    )
    # The one-step changes are taken at the model's parameters, which the
    # fit found to about 1e-6 (relative) of survreg's (tightened)
    # maximum: the scores there differ by that times the curvature.
    scale_of = np.abs(expected).max(axis=0)
    assert_allclose(
        dfbeta[ours] / scale_of, expected[theirs] / scale_of, atol=1e-4
    )
    # The model's covariance is computed where its fit's search ended
    # (on centred covariates, carried back): the same to about 1e-5.
    assert_allclose(se, model.standard_errors()[model.k_dist :], rtol=1e-4)
    residuals = ph_aft._residual_data(ctx, "PH")["lo"]
    # The residuals, too, are at the model's parameters
    assert_allclose(np.sort(residuals), np.sort(r["cox_snell"]), rtol=1e-5)
    lognormal = values("r_ph_aft", "lung_lognormal")
    aft = _lung_fit(sp.LogNormalAFT, lung).diagnose()._context
    e = ph_aft._residual_data(aft, "AFT")["lo"]
    mu, sigma = aft.model.params[:2]
    assert_allclose(
        np.sort((np.log(e) - mu) / sigma),
        np.sort(lognormal["standardised"]),
        rtol=1e-5,
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# Constructed data: the checks and their advice
# ---------------------------------------------------------------------------
def _weibull_ph(rng, n, beta, Z, shape=1.5, scale=10.0):
    e = rng.exponential(size=n)
    return scale * (e / np.exp(Z @ beta)) ** (1 / shape)


def _frame(t, cens, **covariates):
    return pd.DataFrame(
        dict(x=np.minimum(t, cens), c=(t > cens).astype(int), **covariates)
    )


def _fit(fitter, df, Z_cols=None, formula=None, **kwargs):
    return _quiet(
        fitter.fit_from_df,
        df,
        x_col="x",
        c_col="c",
        Z_cols=Z_cols,
        formula=formula,
        **kwargs,
    )


@pytest.fixture(scope="module")
def meets_ph():
    rng = np.random.default_rng(5)
    n = 300
    z, w = rng.binomial(1, 0.5, n), rng.normal(size=n)
    t = _weibull_ph(rng, n, np.array([0.7, 0.5]), np.column_stack([z, w]))
    return _fit(
        sp.WeibullPH, _frame(t, rng.uniform(5, 30, n), z=z, w=w), ["z", "w"]
    )


def test_a_ph_model_that_meets_its_assumptions_passes(meets_ph):
    d = meets_ph.diagnose()
    assert d.ok, d
    assert not d.advice, d.advice
    assert set(d.table["check"]) == {
        "proportional_hazards",
        "functional_form",
        "influence",
        "outliers",
        "baseline_distribution",
    }
    assert d.family == "Parametric proportional hazards"
    assert d.model == "ParametricRegressionModel (Proportional Hazard)"
    assert d.columns == ["z", "w"]
    # z is binary: no functional-form row, and a note says why
    assert set(_rows(d, "functional_form")["covariates"]) == {("w",)}
    assert any("two-valued" in n for n in d.notes)
    # The Weibull's PH and AFT models are one model: not compared
    assert len(_rows(d, "baseline_distribution")) == 1
    assert any("WeibullPH and WeibullAFT fit the same" in n for n in d.notes)
    glob = _rows(d, "proportional_hazards").iloc[-1]
    assert glob["covariates"] == ("z", "w") and glob["df"] == 2


def _crossing(rng, n):
    """A binary covariate whose hazard ratio reverses at t = 0.5."""
    z = rng.binomial(1, 0.5, n).astype(float)
    e = rng.exponential(size=n)
    exposed = np.where(e < 1.5, e / 3, 0.5 + 3 * (e - 1.5))
    return z, np.where(z == 1, exposed, e)


def test_a_binary_ph_violation_is_advised_a_step_an_aft_or_strata():
    rng = np.random.default_rng(11)
    z, t = _crossing(rng, 400)
    w = rng.normal(size=400)
    t = t * np.exp(-0.3 * w)
    d = _fit(
        sp.WeibullPH, _frame(t, np.full(400, 3.0), z=z, w=w), ["z", "w"]
    ).diagnose()
    failed = _rows(d, "proportional_hazards", "fail")
    assert ("z",) in set(failed["covariates"])
    assert ("w",) not in set(failed["covariates"])
    detail = failed.loc[failed["covariates"] == ("z",), "detail"].item()
    assert detail.startswith("its coefficient falls after t = ")
    (advice,) = [
        a
        for a in d.advice
        if a.check == "proportional_hazards" and "of z" in a
    ]
    assert "WeibullPH.fit_tvc" in advice and "LogNormalAFT.fit" in advice
    assert "CoxPH.fit(..., strata=<the z column>)" in advice
    assert advice.features == (
        "WeibullPH.fit_tvc",
        "LogNormalAFT.fit",
        "CoxPH.fit(strata=)",
    )
    assert advice.covariates == ("z",)


@pytest.mark.parametrize(
    "dist, aft",
    [(sp.Rayleigh, "LogNormalAFT"), (sp.LogLogistic, "AFT(LogLogistic)")],
)
def test_a_baseline_with_no_fitter_of_its_own_is_named_by_the_factory(
    dist, aft
):
    rng = np.random.default_rng(11)
    z, t = _crossing(rng, 400)
    df = _frame(t, np.full(400, 3.0), z=z)
    d = _fit(sp.PH(dist), df, ["z"]).diagnose()
    (advice,) = [a for a in d.advice if a.check == "proportional_hazards"]
    own = "PH({})".format(dist.name)
    assert own + ".fit_tvc" in advice and aft + ".fit" in advice
    feature = "LogNormalAFT.fit" if aft == "LogNormalAFT" else "AFT"
    assert advice.features == ("PH", feature, "CoxPH.fit(strata=)")
    # The fitted baseline is ranked against the built-in ones
    ranked = ph_aft._aic_candidates(d._context, "PH")
    assert ranked[0]["label"] == own
    assert {c["label"] for c in ranked[1:]} >= {
        b + "PH" for b in ph_aft.BASELINES
    }
    # The Rayleigh's PH and AFT models are one model: not compared
    twin = [c for c in ranked if c["kind"] == "AFT"]
    assert bool(twin) == (dist is sp.LogLogistic)
    if dist is sp.Rayleigh:
        assert any("PH(Rayleigh) and AFT(Rayleigh)" in n for n in d.notes)
        d = _fit(sp.AFT(dist), df, ["z"]).diagnose()
        assert _rows(d, "common_shape").empty
        assert any("no shape parameter the check" in n for n in d.notes)


def test_a_continuous_ph_violation_is_not_advised_strata():
    rng = np.random.default_rng(3)
    n = 500
    w = rng.normal(size=n)
    e = rng.exponential(size=n)
    early, late = np.exp(w), np.exp(-w)
    t = np.where(e < 0.5 * early, e / early, 0.5 + (e - 0.5 * early) / late)
    d = _fit(sp.LogNormalPH, _frame(t, np.full(n, 4.0), w=w), ["w"]).diagnose()
    assert _rows(d, "proportional_hazards").iloc[0]["verdict"] == "fail"
    (advice,) = [a for a in d.advice if a.check == "proportional_hazards"]
    # The AFT model named is the LogNormal's own
    assert advice.features == ("LogNormalPH.fit_tvc", "LogNormalAFT.fit")
    assert "strata" not in advice


def test_a_joint_failure_alone_has_its_own_advice(monkeypatch, meets_ph):
    def fake(ctx, J):
        p = 0.01 if len(J) > 1 else 0.3
        return {
            "statistic": 9.0,
            "df": len(J),
            "p": p,
            "fit": None,
            "reason": "",
        }

    monkeypatch.setattr(ph_aft, "_step_test", fake)
    d = meets_ph.diagnose()
    (advice,) = [a for a in d.advice if a.check == "proportional_hazards"]
    assert "GLOBAL" in advice and "no single covariate" in advice
    assert advice.covariates == ("z", "w")
    assert advice.features == ("WeibullPH.fit_tvc", "LogNormalAFT.fit")


@pytest.mark.parametrize(
    "fitter, scale, feature",
    [
        (sp.WeibullPH, "the log hazard", "WeibullPH.fit_from_df(formula=)"),
        (sp.WeibullAFT, "log time", "WeibullAFT.fit_from_df(formula=)"),
    ],
)
def test_a_non_linear_effect_fails_functional_form(fitter, scale, feature):
    rng = np.random.default_rng(2)
    n = 400
    u = rng.uniform(-2, 2, n)
    t = _weibull_ph(rng, n, np.array([0.8]), (u**2)[:, None])
    df = _frame(t, rng.uniform(2, 20, n), u=u)
    d = _fit(fitter, df, ["u"]).diagnose()
    form = _rows(d, "functional_form")
    assert form["verdict"].tolist() == ["fail"]
    assert form["detail"].item() == "its effect on {} is not linear".format(
        scale
    )
    (advice,) = [a for a in d.advice if a.check == "functional_form"]
    assert 'formula="np.log(u) + ..."' in advice
    assert "linearly on {}".format(scale) in advice
    assert advice.features == (feature,)
    # ... and the fix it names is a formula the fit takes
    _fit(fitter, df, formula="np.square(u) + u")


def test_an_influential_row_warns():
    rng = np.random.default_rng(4)
    n = 30
    z = rng.normal(size=n)
    t = _weibull_ph(rng, n, np.array([0.8]), z[:, None])
    z[0], t[0] = 4.0, 500.0  # far out, and lives far longer than it should
    d = _fit(sp.WeibullPH, _frame(t, np.full(n, 600.0), z=z), ["z"]).diagnose()
    row = _rows(d, "influence").iloc[0]
    assert row["verdict"] == "warn" and row["statistic"] > 1
    assert pd.isna(row["p"])
    (advice,) = [a for a in d.advice if a.check == "influence"]
    assert advice.startswith("Row ") and "WeibullPH.fit" in advice
    # The row is model.data's (the fitted rows, sorted): the unit put out
    assert row["detail"].split()[2:4] == ["of", "model.data"]
    assert d._context.model.data.Z[int(row["detail"].split()[1]), 0] == 4.0
    assert advice.features == ("WeibullPH.fit",)
    assert "fail" not in set(_rows(d, "influence")["verdict"])


def test_an_outlier_warns_with_the_bonferroni_cutoff():
    from scipy.stats import norm

    rng = np.random.default_rng(6)
    n = 200
    z = rng.normal(size=n)
    t = _weibull_ph(rng, n, np.array([1.0]), z[:, None], shape=2.0)
    z[0], t[0] = 3.0, 60.0  # high risk, censored very late
    d = _fit(sp.WeibullPH, _frame(t, np.full(n, 50.0), z=z), ["z"]).diagnose()
    row = _rows(d, "outliers").iloc[0]
    cutoff = norm.isf(0.05 / (2 * n))
    assert row["verdict"] == "warn" and row["statistic"] > cutoff
    assert "{:.3g}".format(cutoff) in row["detail"]
    (advice,) = [a for a in d.advice if a.check == "outliers"]
    assert "recording errors" in advice and advice.features == ()


def test_a_better_baseline_is_advised():
    rng = np.random.default_rng(7)
    n = 400
    z = rng.binomial(1, 0.5, n)
    t = np.exp(rng.normal(2.0, 0.4, n))  # LogNormal times, no effect
    d = _fit(sp.WeibullPH, _frame(t, np.full(n, 30.0), z=z), ["z"]).diagnose()
    row = _rows(d, "baseline_distribution").iloc[0]
    assert row["verdict"] == "warn" and row["statistic"] > ph_aft.AIC_MARGIN
    assert row["detail"].startswith("LogNormalPH fits better")
    (advice,) = [a for a in d.advice if a.check == "baseline_distribution"]
    assert "Refit with LogNormalPH.fit" in advice
    assert advice.features == ("LogNormalPH.fit",)


def _lognormal_ph_times(rng, n, beta, z):
    """LogNormal-baseline PH times: S(t | z) = S0(t)^exp(beta z)."""
    u = rng.uniform(size=n)
    return sp.LogNormal.qf(1 - u ** np.exp(-beta * z), 2.0, 0.6)


def test_ph_against_aft_is_advised_both_ways():
    rng = np.random.default_rng(8)
    n = 600
    z = rng.binomial(1, 0.5, n).astype(float)
    # LogNormal AFT data, fitted as LogNormal PH: the AFT model is better
    t = np.exp(rng.normal(2.0, 0.5, n) - 1.0 * z)
    d = _fit(
        sp.LogNormalPH, _frame(t, np.full(n, 60.0), z=z), ["z"]
    ).diagnose()
    row = _rows(d, "baseline_distribution")
    row = row[row["test"].str.contains("AFT")].iloc[0]
    assert row["verdict"] == "warn"
    (advice,) = [a for a in d.advice if a.features == ("LogNormalAFT.fit",)]
    assert advice.check == "baseline_distribution"
    assert "The accelerated failure time model" in advice
    assert "rescale time rather than multiply the hazard" in advice
    # LogNormal PH data, fitted as LogNormal AFT: the PH model is better
    t = _lognormal_ph_times(rng, n, 2.0, z)
    d = _fit(
        sp.LogNormalAFT, _frame(t, np.full(n, 60.0), z=z), ["z"]
    ).diagnose()
    row = _rows(d, "error_distribution")
    row = row[row["test"].str.contains("PH")].iloc[0]
    assert row["verdict"] == "warn"
    (advice,) = [a for a in d.advice if a.features == ("LogNormalPH.fit",)]
    assert advice.check == "error_distribution"
    assert "The proportional hazards model" in advice
    assert "multiply the hazard rather than rescale time" in advice


@pytest.fixture(scope="module")
def shape_violation():
    rng = np.random.default_rng(9)
    n = 400
    g = rng.binomial(1, 0.5, n)
    w = rng.normal(size=n)
    shape = np.where(g == 1, 3.0, 1.0)
    t = 10 * rng.exponential(size=n) ** (1 / shape) * np.exp(-0.4 * w)
    return _frame(t, np.full(n, 40.0), g=g, w=w)


def test_a_shape_that_varies_by_group_fails_common_shape(shape_violation):
    d = _fit(sp.WeibullAFT, shape_violation, ["g", "w"]).diagnose()
    assert d.family == "Accelerated failure time"
    assert set(d.table["check"]) == {
        "error_distribution",
        "common_shape",
        "functional_form",
    }
    row = _rows(d, "common_shape").iloc[0]
    assert row["covariates"] == ("g",) and row["verdict"] == "fail"
    assert row["detail"].startswith("the shape (beta) differs by level")
    assert any("w have more" in n or "w has more" in n for n in d.notes)
    (advice,) = [a for a in d.advice if a.check == "common_shape"]
    assert "WeibullAFT.fit on that level's rows" in advice
    assert 'strata_col="g"' in advice
    assert advice.features == (
        "WeibullAFT.fit",
        "Weibull.fit",
        "CoxPH.fit(strata=)",
        "CoxPH.fit_from_df(strata_col=)",
    )


def test_a_formula_factor_is_one_shape_test(shape_violation):
    df = shape_violation.copy()
    rng = np.random.default_rng(1)
    df["arm"] = np.where(df["g"] == 1, "b", rng.choice(["a", "c"], len(df)))
    d = _fit(sp.LogNormalAFT, df, formula="C(arm) + w").diagnose()
    (row,) = _rows(d, "common_shape").itertuples()
    assert row.covariates == ("C(arm)[T.b]", "C(arm)[T.c]") and row.df == 2
    assert "arm reference" in row.detail and "arm=b" in row.detail
    if row.verdict == "fail":
        assert any('strata_col="arm"' in a for a in d.advice)


def test_the_exponential_has_no_shape_to_test():
    rng = np.random.default_rng(2)
    n = 200
    z = rng.binomial(1, 0.5, n)
    t = rng.exponential(np.exp(0.5 * z))
    d = _fit(
        sp.ExponentialAFT, _frame(t, np.full(n, 5.0), z=z), ["z"]
    ).diagnose()
    assert _rows(d, "common_shape").empty
    assert any("no shape parameter" in n for n in d.notes)
    assert any("ExponentialPH.fit on the same data" in n for n in d.notes)


# ---------------------------------------------------------------------------
# What the checks support
# ---------------------------------------------------------------------------
def test_columns_and_alpha(meets_ph):
    d = meets_ph.diagnose(columns=["w"])
    single = {c for c in d.table["covariates"] if len(c) == 1}
    assert single == {("w",)}
    glob = _rows(d, "proportional_hazards").iloc[-1]
    assert glob["covariates"] == ("z", "w")
    loose = meets_ph.diagnose(alpha=0.5)
    tests = loose.table[loose.table["p"].notna()]
    assert ((tests["verdict"] == "fail") == (tests["p"] < 0.5)).all()


def test_the_step_test_does_not_depend_on_a_covariates_origin(meets_ph):
    # Principle 6: a constant added to a covariate leaves the tests as
    # they were (the baseline's own step absorbs it)
    df = pd.DataFrame(
        {
            "x": meets_ph.data.x,
            "c": meets_ph.data.c,
            "z": meets_ph.data.Z[:, 0],
            "w": meets_ph.data.Z[:, 1],
        }
    )
    shifted = df.assign(w=df["w"] + 3.0)
    a = _fit(sp.WeibullPH, df, ["z", "w"]).diagnose()
    b = _fit(sp.WeibullPH, shifted, ["z", "w"]).diagnose()
    c = _fit(sp.WeibullPH, shifted, ["z", "w"], center=True).diagnose()
    for other in (b, c):
        assert_allclose(
            other.table["statistic"].to_numpy(dtype=float)[:6],
            a.table["statistic"].to_numpy(dtype=float)[:6],
            rtol=1e-5,
        )


def test_counts_equal_repetition():
    rng = np.random.default_rng(3)
    m = 60
    z = rng.normal(size=m)
    w = rng.integers(0, 2, m).astype(float)
    x = np.ceil(
        _weibull_ph(rng, m, np.array([0.5, 0.3]), np.column_stack([z, w]))
    )
    c = (rng.uniform(size=m) < 0.2).astype(int)
    n = rng.integers(1, 4, m)
    Z = np.column_stack([z, w])
    counted = sp.WeibullAFT.fit(x, Z, c=c, n=n).diagnose()
    written = sp.WeibullAFT.fit(
        np.repeat(x, n), np.repeat(Z, n, axis=0), c=np.repeat(c, n)
    ).diagnose()
    columns = ["check", "covariates", "df", "verdict"]
    pd.testing.assert_frame_equal(
        counted.table[columns], written.table[columns]
    )
    assert_allclose(
        counted.table["statistic"], written.table["statistic"], rtol=1e-5
    )
    counted = sp.WeibullPH.fit(x, Z, c=c, n=n).diagnose()
    written = sp.WeibullPH.fit(
        np.repeat(x, n), np.repeat(Z, n, axis=0), c=np.repeat(c, n)
    ).diagnose()
    tests = counted.table["check"] != "influence"
    assert_allclose(
        counted.table.loc[tests, "statistic"],
        written.table.loc[tests, "statistic"],
        rtol=1e-5,
    )
    assert_allclose(
        _rows(counted, "influence")["statistic"],
        _rows(written, "influence")["statistic"],
        rtol=1e-5,
    )


def test_interval_and_left_censoring_and_truncation():
    rng = np.random.default_rng(12)
    n = 150
    z = rng.binomial(1, 0.5, n).astype(float)
    t = _weibull_ph(rng, n, np.array([0.6]), z[:, None], shape=2.0)
    lo = np.floor(t)
    x = np.column_stack([lo, lo + 1])
    c = np.full(n, 2)
    c[lo == 0] = -1
    x[lo == 0] = 1
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for fitter in (sp.WeibullPH, sp.LogNormalAFT):
        d = _quiet(fitter.fit, x, z[:, None], c=c).diagnose()
        assert d.table.loc[d.table["p"].notna(), "p"].between(0, 1).all()
        assert not d.table["detail"].str.contains("could not").any(), d
        # The probability plots take Turnbull's estimate of the residuals
        plt.close(d.plot())
    # The residual plots leave out the interval rows, and say so
    d = _quiet(sp.WeibullPH.fit, x, z[:, None], c=c).diagnose()
    assert _rows(d, "outliers").empty
    assert any("left out of them" in n for n in d.notes)
    # Right truncation: units seen only if they failed by t = 15
    seen = t < 15
    d = _quiet(sp.WeibullPH.fit, t[seen], z[seen, None], tr=15.0).diagnose()
    assert d.table.loc[d.table["p"].notna(), "p"].between(0, 1).all()
    assert not d.table["detail"].str.contains("could not").any(), d
    assert any("right-truncated" in n for n in d.notes)
    plt.close(d.plot("baseline_distribution"))


def test_covariate_names_come_from_the_fit(meets_ph):
    rng = np.random.default_rng(1)
    Z = pd.DataFrame(
        {"dose": rng.normal(size=80), "arm": rng.integers(0, 2, 80)}
    )
    t = _weibull_ph(rng, 80, np.array([0.3, 0.3]), Z.to_numpy(dtype=float))
    assert sp.WeibullAFT.fit(t, Z).diagnose().columns == ["dose", "arm"]
    assert sp.WeibullAFT.fit(t, Z.values).diagnose().columns == [
        "coef_0",
        "coef_1",
    ]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def test_a_restored_model_refuses(meets_ph):
    restored = sp.from_dict(meets_ph.to_dict())
    with pytest.raises(ValueError, match="from_dict"):
        restored.diagnose()


def test_a_time_varying_fit_refuses_naming_the_issue():
    i = [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 9]
    xl = [0, 2, 0, 0, 1, 0, 0, 3, 0, 0, 0, 2, 0, 0, 4]
    xr = [2, 5, 3, 1, 4, 6, 3, 7, 2, 8, 2, 5, 4, 4, 9]
    c = [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    Z = np.array([0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1.0])[:, None]
    model = _quiet(sp.WeibullPH.fit_tvc, i, xl, xr, c, Z)
    with pytest.raises(NotImplementedError, match="time-varying.*#869"):
        model.diagnose()


def test_a_custom_link_refuses_naming_the_issue():
    import autograd.numpy as anp

    def linear_rr(Z, *params):
        return 1.0 + anp.dot(Z, anp.array(params))

    fitter = sp.ProportionalHazardsFitter(
        "WeibullLinearRR",
        sp.Weibull,
        linear_rr,
        "Linear [1 + beta'Z]",
        phi_bounds=lambda Z: ((0, None),) * Z.shape[1],
        phi_param_map=lambda Z: {f"beta_{i}": i for i in range(Z.shape[1])},
        phi_init=lambda Z: np.full(Z.shape[1], 0.5),
    )
    rng = np.random.default_rng(0)
    dose = rng.uniform(0, 4, 200)
    x = 10 * (-np.log(rng.uniform(size=200)) / (1 + 0.5 * dose)) ** 0.5
    with pytest.raises(NotImplementedError, match="link.*#869"):
        fitter.fit(x=x, Z=dose).diagnose()


def test_an_aliased_fit_refuses():
    rng = np.random.default_rng(1)
    z = rng.normal(size=60)
    model = _quiet(
        sp.WeibullPH.fit, rng.exponential(size=60), np.column_stack([z, z])
    )
    with pytest.raises(ValueError, match="aliased"):
        model.diagnose()


def test_a_fit_without_a_verified_maximum_refuses(meets_ph):
    import copy

    model = copy.copy(meets_ph)
    model.maximum = "no finite maximum"
    with pytest.raises(ValueError, match="finite maximum"):
        model.diagnose()
    model.maximum = "unverified"
    with pytest.raises(ValueError, match="verified maximum.*init="):
        model.diagnose()


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------
def test_to_dict_round_trips(shape_violation):
    d = _fit(sp.WeibullAFT, shape_violation, ["g", "w"]).diagnose()
    out = d.to_dict()
    back = Diagnosis.from_dict(json.loads(json.dumps(out, allow_nan=False)))
    assert back.to_dict() == out
    assert back.advice == d.advice
    assert [a.features for a in back.advice] == [a.features for a in d.advice]


def test_plot_draws_every_panel(meets_ph, shape_violation):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = meets_ph.diagnose().plot()
    titles = [ax.get_title() for ax in fig.axes]
    assert titles[0].startswith("z: p = ") and titles[0].endswith("(step)")
    assert "w: functional form" in titles
    assert "Cox-Snell residuals (exponential paper)" in titles
    assert "AIC by model" in titles
    # 2 PH, 1 functional form, 2 influence, outliers, Cox-Snell, AIC
    assert len(titles) == 2 + 1 + 2 + 1 + 1 + 1
    form = fig.axes[titles.index("w: functional form")]
    assert form.get_xlabel() == "w"
    plt.close(fig)

    d = _fit(sp.WeibullAFT, shape_violation, ["g", "w"]).diagnose()
    fig = d.plot()
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == [
        "standardised residuals (Weibull paper)",
        "AIC by model",
        "g: residuals by level",
        "w: functional form",
    ]
    plt.close(fig)
    fig = d.plot("common_shape")
    assert len(fig.axes[0].collections) == 2  # one level each
    plt.close(fig)
