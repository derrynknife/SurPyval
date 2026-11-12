"""The mixed-effects Cox model and the log-normal Cox frailty against R's
``coxme`` (2.2.22) and ``survival::coxph(... + frailty(dist =
"gaussian"))``: the stored values of scripts/reference/reference_r_coxme.R
on the kidney, lung and seeded clustered fixtures.

coxme's own search over the variances stops early with its default
tolerance; the script runs it to ``reltol = 1e-14``, where it agrees with
SurPyval's maximum to the printed digits. The tolerances below are those
of two optimisers of the same smooth function reaching the same maximum:
SurPyval's integrated log-likelihood is at least R's and at most 1e-4
above it (coxme still stops up to 6e-6 short on the correlated random
slope), and the coefficients are within 1e-2 of R's. Refitted at R's own
variances (``variance=``), every value R reports is reproduced to 1e-6 or
better, which checks the likelihoods themselves without the optimisers.
"""

import numpy as np
import pandas as pd
import pytest

from surpyval import CoxFrailty, CoxMixedEffects, RandomEffect

from ._data import fixture, reference

# R's maximum is short of SurPyval's by up to this much (coxme's search
# stops on a flat ridge), and never above it.
LOGLIK = 1e-4


def _kidney():
    d = fixture("kidney")
    Z = np.column_stack([d["age"], d["female"]])
    return d, Z


def _r_variances(model, v, names):
    """R's variances, one per term of ``model``, in its order."""
    out = []
    for term in model.terms:
        r_name = next(r for r, py in names.items() if py == term["name"])
        out.append(np.atleast_2d(np.asarray(v["variance"][r_name], float)))
    fixed = []
    for S in out:
        if S.shape[0] > 1:
            # VarCorr: variances on the diagonal, correlations off it
            sd = np.sqrt(np.diag(S))
            R = S.copy()
            np.fill_diagonal(R, 1.0)
            S = R * np.outer(sd, sd)
        fixed.append(S if S.shape[0] > 1 else float(S[0, 0]))
    return fixed


def _check(model, ref, names, refit):
    """The free fit (SurPyval's maximum at least R's, the estimates near
    R's), then the fit at R's own variances, which must reproduce every
    one of R's values closely: the same penalised and integrated
    likelihoods."""
    v = ref["values"]
    assert v["loglik_integrated"] - 1e-9 <= model.log_likelihood
    assert model.log_likelihood <= v["loglik_integrated"] + LOGLIK
    np.testing.assert_allclose(model.beta, v["beta"], rtol=1e-2)
    at_r = refit(_r_variances(model, v, names))
    np.testing.assert_allclose(at_r.beta, v["beta"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(at_r.standard_errors(), v["se"], rtol=1e-6)
    for got, key in (
        (at_r.log_likelihood, "loglik_integrated"),
        (at_r.log_likelihood_null, "loglik_null"),
        (at_r.log_likelihood_penalised, "loglik_penalised"),
        (at_r.log_likelihood_fitted, "loglik_fitted"),
    ):
        assert got == pytest.approx(v[key], abs=1e-7), key
    assert at_r.df_penalised == pytest.approx(v["df_penalised"], rel=1e-6)
    ranef = at_r.random_effects
    for r_name, py_name in names.items():
        r = v["ranef"][r_name]
        labels = [str(lab).replace("/", ":") for lab in r["labels"]]
        values = np.asarray(r["values"], dtype=float).reshape(len(labels), -1)
        mine = ranef[py_name].loc[labels].to_numpy()
        np.testing.assert_allclose(mine, values, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_kidney_random_intercept_matches_coxme(ties):
    d, Z = _kidney()

    def fit(variance=None):
        return CoxMixedEffects.fit(
            d["time"],
            Z=Z,
            c=d["c"],
            groups=d["id"].astype(int),
            tie_method=ties,
            variance=variance,
        )

    ref = reference("r_coxme", "kidney_coxme_" + ties)
    _check(fit(), ref, {"id": "group"}, fit)


def test_lung_by_institution_matches_coxme():
    d = fixture("lung")
    df = pd.DataFrame(d)
    with pytest.warns(UserWarning, match="missing group"):
        model = CoxMixedEffects.fit_from_df(
            df,
            x_col="time",
            c_col="c",
            Z_cols=["age", "sex"],
            random_effects="(1 | inst)",
        )
    ref = reference("r_coxme", "lung_coxme_inst")
    v = ref["values"]
    # The variance is on its boundary: R's search stops at a variance of
    # rounding size, SurPyval's at 0.
    assert model.log_likelihood == pytest.approx(
        v["loglik_integrated"], abs=1e-8
    )
    np.testing.assert_allclose(model.beta, v["beta"], rtol=1e-6)
    np.testing.assert_allclose(model.standard_errors(), v["se"], rtol=1e-6)
    assert model.variance_components()["variance"].iloc[0] < 1e-6


def _clustered():
    d = fixture("clustered")
    df = pd.DataFrame(d)
    for col in ("hospital", "ward", "batch", "sex"):
        df[col] = df[col].astype(int)
    return df


def test_nested_and_crossed_terms_with_truncation_and_strata_match_coxme():
    df = _clustered()

    def fit(variance=None):
        return CoxMixedEffects.fit_from_df(
            df,
            x_col="x",
            c_col="c",
            tl_col="tl",
            Z_cols=["trt", "age"],
            random_effects="(1 | hospital/ward) + (1 | batch)",
            strata_col="sex",
            variance=variance,
        )

    names = {
        "hospital": "hospital",
        "hospital/ward": "hospital:ward",
        "batch": "batch",
    }
    _check(fit(), reference("r_coxme", "clustered_nested_crossed"), names, fit)


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_correlated_random_slope_matches_coxme(ties):
    df = _clustered()

    def fit(variance=None):
        return CoxMixedEffects.fit_from_df(
            df,
            x_col="x",
            c_col="c",
            tl_col="tl",
            Z_cols=["trt", "age"],
            random_effects="(1 + trt | hospital)",
            tie_method=ties,
            variance=variance,
        )

    model = fit()
    ref = reference("r_coxme", "clustered_random_slope_" + ties)
    _check(model, ref, {"hospital": "hospital"}, fit)
    # The same model through RandomEffect, without a formula
    plain = CoxMixedEffects.fit(
        df["x"],
        Z=df[["trt", "age"]].to_numpy(),
        c=df["c"],
        tl=df["tl"],
        random_effects=RandomEffect(df["hospital"], slopes=[0]),
        tie_method=ties,
    )
    assert plain.log_likelihood == pytest.approx(
        model.log_likelihood, abs=1e-9
    )


def test_penalised_fit_at_a_fixed_variance_is_coxphs_gaussian_frailty():
    # coxph(... + frailty(id, dist = "gaussian", theta = 0.5)): the same
    # penalised partial likelihood as coxme's at that variance.
    d, Z = _kidney()
    v = reference("r_coxme", "kidney_coxph_gaussian_fixed")["values"]
    model = CoxMixedEffects.fit(
        d["time"], Z=Z, c=d["c"], groups=d["id"].astype(int), variance=[0.5]
    )
    np.testing.assert_allclose(model.beta, v["beta"], rtol=1e-6)
    np.testing.assert_allclose(model.standard_errors(), v["se"], rtol=1e-6)
    blups = model.random_effects["group"].to_numpy()[:, 0]
    np.testing.assert_allclose(blups, v["frail"], rtol=1e-6, atol=1e-9)
    assert model.log_likelihood_fitted == pytest.approx(
        v["loglik_partial"], abs=1e-7
    )


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_lognormal_cox_frailty_is_coxmes_random_intercept(ties):
    # CoxFrailty(family="lognormal") is coxme's (1 | id): theta is the
    # variance of the log-frailty, the frailties exp of coxme's ranef.
    d, Z = _kidney()
    v = reference("r_coxme", "kidney_coxme_" + ties)["values"]
    model = CoxFrailty.fit(
        d["time"],
        Z=Z,
        c=d["c"],
        groups=d["id"].astype(int),
        tie_method=ties,
        family="lognormal",
    )
    assert v["loglik_integrated"] - 1e-9 <= model.log_likelihood
    assert model.log_likelihood <= v["loglik_integrated"] + LOGLIK
    variance = float(np.asarray(v["variance"]["id"], dtype=float).ravel()[0])
    at_r = CoxFrailty.fit(
        d["time"],
        Z=Z,
        c=d["c"],
        groups=d["id"].astype(int),
        tie_method=ties,
        family="lognormal",
        theta=variance,
    )
    np.testing.assert_allclose(at_r.beta, v["beta"], rtol=1e-6)
    np.testing.assert_allclose(at_r.standard_errors()[:2], v["se"], rtol=1e-6)
    assert at_r.log_likelihood == pytest.approx(
        v["loglik_integrated"], abs=1e-7
    )
    r = v["ranef"]["id"]
    frailties = [at_r.frailties[str(lab)] for lab in r["labels"]]
    np.testing.assert_allclose(
        np.log(frailties), np.ravel(r["values"]), rtol=1e-5, atol=1e-8
    )
