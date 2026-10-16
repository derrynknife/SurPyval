"""The shared-frailty families of ``diagnose()`` (#872): their statistics
against stored R results, the calibration of the test of no frailty under
its null, their checks on data built to pass or to fail each of them, the
advice each failure calls for, and their refusals.

The R values (``scripts/reference/reference_r_frailty_diag.R``, stored in
``tests/reference/data/r_frailty_diag.json``) are, on R's kidney catheter
data: coxph's I-likelihood of the gamma-frailty Cox model against the Cox
partial likelihood; lme4's glmer fits of the log-normal-frailty Weibull
and exponential models (the Poisson GLMM of the events, of the split
follow-up for the step test) against survreg's fits without frailty; and
the closed-form gamma-frailty Weibull likelihood maximised with optim and
nlminb. The statistics are differences of maximised log-likelihoods,
which agree with R's to the precision of the two maximisations:
``rtol=1e-6``.
"""

import copy
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.datasets import load_kidney
from surpyval.diagnostics import Diagnosis, frailty
from surpyval.tests.reference._data import fixture, values

FITTED = dict(rtol=1e-6, atol=1e-8)
LOGNORMAL = sp.Frailty(sp.Weibull, family="lognormal")


def _quiet(fit, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit(*args, **kwargs)


def _rows(d, check, verdict=None):
    table = d.table[d.table["check"] == check]
    if verdict is not None:
        table = table[table["verdict"] == verdict]
    return table


def _row(d, check, covariates=()):
    table = _rows(d, check)
    return table[table["covariates"] == covariates].iloc[0]


def _features(d):
    return [a.features for a in d.advice]


# ---------------------------------------------------------------------------
# Against R
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def kidney():
    d = fixture("kidney")
    return pd.DataFrame(
        {k: d[k] for k in ("time", "c", "id", "age", "female")}
    )


def _kidney_fit(fitter, kidney, **kwargs):
    return _quiet(
        fitter.fit_from_df,
        kidney,
        x_col="time",
        c_col="c",
        group_col="id",
        Z_cols=["age", "female"],
        **kwargs,
    )


@pytest.fixture(scope="module")
def gamma_weibull(kidney):
    return _kidney_fit(sp.WeibullFrailty, kidney)


@pytest.fixture(scope="module")
def gamma_weibull_d(gamma_weibull):
    return gamma_weibull.diagnose()


@pytest.fixture(scope="module")
def lognormal_weibull_d(kidney):
    return _kidney_fit(LOGNORMAL, kidney).diagnose()


@pytest.fixture(scope="module")
def cox_d(kidney):
    return _kidney_fit(sp.CoxFrailty, kidney).diagnose()


@pytest.mark.parametrize(
    "which, ref",
    [
        ("gamma_weibull_d", "kidney_gamma_weibull"),
        ("lognormal_weibull_d", "kidney_lognormal_weibull"),
    ],
)
def test_parametric_statistics_match_r(which, ref, request):
    d = request.getfixturevalue(which)
    r = values("r_frailty_diag", ref)
    row = _row(d, "frailty_variance")
    assert_allclose(row["statistic"], r["statistic"], **FITTED)
    assert_allclose(row["p"], r["p"], **FITTED)
    assert row["df"] == 1 and "WeibullPH" in row["test"]
    for covariates, key in [
        (("age",), "step_age"),
        (("female",), "step_female"),
        (("age", "female"), "step_global"),
    ]:
        assert_allclose(
            _row(d, "proportional_hazards", covariates)["statistic"],
            r[key],
            **FITTED,
        )
    assert (
        "t = {:g}".format(r["tau"])
        in _row(d, "proportional_hazards", ("age",))["test"]
    )
    assert _row(d, "proportional_hazards", ("age", "female"))["df"] == 2
    assert_allclose(
        _row(d, "functional_form", ("age",))["statistic"],
        r["square_age"],
        **FITTED,
    )
    # female is binary: no functional-form row
    assert len(_rows(d, "functional_form")) == 1


def test_frailty_family_aic_matches_r(gamma_weibull_d, lognormal_weibull_d):
    gamma = values("r_frailty_diag", "kidney_gamma_weibull")["aic"]
    lognormal = values("r_frailty_diag", "kidney_lognormal_weibull")["aic"]
    for d, delta in [
        (gamma_weibull_d, gamma - lognormal),
        (lognormal_weibull_d, lognormal - gamma),
    ]:
        row = _row(d, "frailty_family")
        assert_allclose(row["statistic"], delta, **FITTED)
        assert row["verdict"] == "ok"  # 1.7 apart: within the margin of 2


def test_lognormal_exponential_matches_r(kidney):
    r = values("r_frailty_diag", "kidney_lognormal_exponential")
    model = _kidney_fit(sp.Frailty(sp.Exponential, family="lognormal"), kidney)
    res = frailty.zero_variance_test(model)
    assert res["null"] == "ExponentialPH"
    assert_allclose(res["statistic"], r["statistic"], **FITTED)
    assert_allclose(res["loglik_null"], r["loglik_null"], **FITTED)
    assert_allclose(model.aic(), r["aic"], **FITTED)


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_cox_statistic_matches_r(kidney, ties):
    r = values("r_frailty_diag", "kidney_cox_gamma_" + ties)
    model = _kidney_fit(sp.CoxFrailty, kidney, tie_method=ties)
    row = _row(model.diagnose(), "frailty_variance")
    assert_allclose(row["statistic"], r["statistic"], **FITTED)
    assert_allclose(row["p"], r["p"], **FITTED)
    assert row["verdict"] == "ok" and "CoxPH" in row["test"]


def test_cox_without_frailty_matches_r():
    # With the disease type the frailty vanishes (R's theta is 5e-9)
    r = values("r_frailty_diag", "kidney_cox_gamma_no_frailty")
    df = load_kidney()
    df["female"] = (df["sex"] == 2).astype(float)
    df["pkd"] = (df["disease"] == "PKD").astype(float)
    df["censored"] = 1 - df["status"]
    model = _quiet(
        sp.CoxFrailty.fit_from_df,
        df,
        x_col="time",
        c_col="censored",
        group_col="id",
        Z_cols=["age", "female", "pkd"],
    )
    d = model.diagnose()
    row = _row(d, "frailty_variance")
    assert row["statistic"] == r["statistic"] == 0
    assert row["p"] == r["p"] == 1
    assert row["verdict"] == "warn" and d.ok
    assert _features(d) == [("CoxPH.fit", "CoxPH.fit_from_df")]
    assert "tie_method='efron'" in d.advice[0]


def test_the_null_is_the_fit_without_frailty(kidney, gamma_weibull):
    # Parametric: WeibullPH on the same rows. Cox: the partial likelihood
    # of CoxPH.fit on the same rows, which CoxFrailty.fit stores.
    res = frailty.zero_variance_test(gamma_weibull)
    Z = kidney[["age", "female"]].values
    ph = sp.WeibullPH.fit(kidney["time"].values, Z, c=kidney["c"].values)
    assert_allclose(res["loglik_null"], -ph.neg_ll(), rtol=1e-10)
    assert_allclose(res["loglik"], -gamma_weibull.neg_ll(), rtol=1e-12)
    cox = _kidney_fit(sp.CoxFrailty, kidney)
    refit = sp.CoxPH.fit(kidney["time"].values, Z, c=kidney["c"].values)
    assert_allclose(
        frailty.zero_variance_test(cox)["loglik_null"],
        refit.log_likelihood,
        rtol=1e-10,
    )


@pytest.mark.parametrize("fitter", [sp.WeibullFrailty, LOGNORMAL])
def test_the_larger_models_likelihood_is_the_models(kidney, fitter):
    # The likelihood the tests maximise is the fit's own at the fit
    from surpyval.diagnostics.registry import Context

    model = _kidney_fit(fitter, kidney)
    ctx = Context(model, 0.05, ["age", "female"], [0, 1])
    f = frailty._Marginal(model, frailty._data(ctx))
    assert_allclose(f(frailty._start(model)), model.neg_ll(), rtol=1e-10)


@pytest.mark.parametrize("theta", [1e-7, 9.9e-6, 1e-5, 1e-3])
def test_the_gamma_series_meets_the_closed_form(theta):
    from scipy.special import gammaln

    D = np.array([0.0, 1.0, 2.0, 3.0, 2.5])
    H = np.array([0.1, 0.8, 2.0, 1.5, 3.0])
    a = 1 / theta
    exact = (
        gammaln(D + a)
        - gammaln(a)
        - D * np.log(a)
        - (D + a) * np.log1p(H * theta)
    )
    # (the closed form loses ~1e-10 to cancellation at theta 1e-5)
    assert_allclose(frailty._gamma_term(D, H, theta), exact, atol=2e-9)


# ---------------------------------------------------------------------------
# The test of no frailty under its null
# ---------------------------------------------------------------------------
def _no_frailty(rng, G, m):
    g = np.repeat(np.arange(G), m)
    z = rng.normal(size=g.size)
    t = 10 * (rng.exponential(size=g.size) / np.exp(0.5 * z)) ** 0.5
    cens = rng.uniform(5, 30, g.size)
    return np.minimum(t, cens), (t > cens).astype(int), z[:, None], g


def _null_p(fitter, reps, G, m, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        x, c, Z, g = _no_frailty(rng, G, m)
        model = _quiet(fitter.fit, x, Z, c, groups=g)
        out.append(frailty.zero_variance_test(model)["p"])
    return np.array(out)


def test_no_frailty_test_is_calibrated_quick():
    # 20 data sets without frailty: the boundary test rejects at about
    # its level, and about half (or, in small samples, more) of the fits
    # put theta on its boundary, T = 0 and p = 1. (The opt-in study below
    # has the size.)
    p = _null_p(sp.WeibullFrailty, 20, 30, 4, 7)
    assert np.sum(p < 0.05) <= 4  # P(Binomial(20, 0.05) > 4) = 0.003
    assert 0.3 <= np.mean(p == 1) <= 0.9
    assert np.all((p > 0) & (p <= 1))
    assert np.all((p == 1) | (p <= 0.5))  # the mixture: p > 0.5 is T = 0


@pytest.mark.calibration
@pytest.mark.parametrize(
    "fitter, reps",
    [(sp.WeibullFrailty, 1000), (LOGNORMAL, 400), (sp.CoxFrailty, 400)],
)
def test_no_frailty_test_is_calibrated(fitter, reps):
    # The size at 0.05, 0.10 and 0.25 is within the 99.9% binomial range
    # of the level, or below it: the mixture's chi2(0) share is at least
    # half in finite samples, which makes the test conservative.
    from scipy.stats import binom

    p = _null_p(fitter, reps, 50, 4, 2026)
    for level in (0.05, 0.10, 0.25):
        rejected = int(np.sum(p < level))
        assert rejected <= binom.isf(0.0005, reps, level), (level, rejected)
        assert rejected >= binom.ppf(0.0005, reps, level / 2), (
            level,
            rejected,
        )
    assert 0.4 <= np.mean(p == 1) <= 0.8


# ---------------------------------------------------------------------------
# Constructed data: each check passing and failing, and its advice
# ---------------------------------------------------------------------------
def _simulate(
    rng,
    G=60,
    m=5,
    theta=0.5,
    family="gamma",
    before=lambda c: 0.5 * c["z"],
    after=None,
    cut=None,
    covariates=None,
    baseline=None,
):
    """A shared-frailty data set: ``G`` groups of ``m`` rows, frailty of
    ``family`` with variance ``theta``, a Weibull baseline (scale 10,
    shape 1.5) or ``baseline`` (the time of each baseline cumulative
    hazard), log hazard ratio ``before(covariates)``, and ``after`` after
    the time ``cut``; censored uniformly on (5, 40)."""
    g = np.repeat(np.arange(G), m)
    n = g.size
    cov = covariates(rng, n) if covariates else {"z": rng.normal(size=n)}
    if theta == 0:
        u = np.ones(n)
    elif family == "gamma":
        u = rng.gamma(1 / theta, theta, G)[g]
    else:
        u = np.exp(rng.normal(0, np.sqrt(theta), G))[g]
    lin0 = before(cov)
    lin1 = lin0 if after is None else after(cov)
    E = rng.exponential(size=n) / u
    s0 = np.inf if cut is None else (cut / 10) ** 1.5
    early = E < np.exp(lin0) * s0
    s = np.where(
        early, E / np.exp(lin0), s0 + (E - np.exp(lin0) * s0) / np.exp(lin1)
    )
    t = 10 * s ** (1 / 1.5) if baseline is None else baseline(s)
    cens = rng.uniform(5, 40, n)
    return pd.DataFrame(
        dict(x=np.minimum(t, cens), c=(t > cens).astype(int), g=g, **cov)
    )


def _fit(df, fitter=sp.WeibullFrailty, cols=("z",), **kwargs):
    return _quiet(
        fitter.fit_from_df,
        df,
        x_col="x",
        c_col="c",
        group_col="g",
        Z_cols=list(cols) if cols else None,
        **kwargs,
    )


def _binary(rng, n):
    return {"z": rng.binomial(1, 0.5, n).astype(float)}


@pytest.fixture(scope="module")
def meets_assumptions():
    return _fit(_simulate(np.random.default_rng(12))).diagnose()


def test_data_that_meet_the_assumptions_pass(meets_assumptions):
    d = meets_assumptions
    assert d.ok and not d.advice
    assert set(d.table["verdict"]) == {"ok"}
    assert list(d.table["check"]) == [
        "frailty_variance",
        "frailty_family",
        "proportional_hazards",
        "functional_form",
        "baseline_distribution",
    ]
    assert d.family == "Parametric shared frailty"
    assert d.model == "FrailtyModel"


def test_no_frailty_advises_the_model_without_it():
    df = _simulate(np.random.default_rng(6), G=40, m=4, theta=0.0)
    d = _fit(df).diagnose()
    row = _row(d, "frailty_variance")
    assert row["verdict"] == "warn" and row["p"] == 1
    assert "boundary" in row["detail"]
    assert ("WeibullPH.fit", "WeibullPH.fit_from_df") in _features(d)
    # Without covariates the model without frailty is the distribution
    plain = _fit(df, cols=None).diagnose()
    assert _rows(plain, "proportional_hazards").empty
    assert ("Weibull.fit",) in _features(plain)
    assert "Weibull.fit(x, c, n)" in plain.advice[0]


def test_a_significant_frailty_gives_no_advice(gamma_weibull_d):
    row = _row(gamma_weibull_d, "frailty_variance")
    assert row["verdict"] == "ok" and row["p"] < 0.01
    assert not gamma_weibull_d.advice


def test_the_other_family_fitting_better_is_advised():
    df = _simulate(
        np.random.default_rng(1), G=80, m=6, theta=1.5, family="lognormal"
    )
    d = _fit(df).diagnose()
    row = _row(d, "frailty_family")
    assert row["verdict"] == "warn" and row["statistic"] > 2
    assert _features(d) == [("Frailty(family=)",)]
    assert "family='lognormal'" in d.advice[0]
    # and the log-normal fit is not advised the gamma
    d = _fit(df, LOGNORMAL).diagnose()
    assert _row(d, "frailty_family")["verdict"] == "ok"


@pytest.fixture(scope="module")
def binary_violation():
    df = _simulate(
        np.random.default_rng(2),
        before=lambda c: 1.0 * c["z"],
        after=lambda c: -1.0 * c["z"],
        cut=8,
        covariates=_binary,
    )
    return _fit(df).diagnose()


def test_a_binary_effect_that_changes_is_advised_strata(binary_violation):
    d = binary_violation
    row = _row(d, "proportional_hazards", ("z",))
    assert row["verdict"] == "fail" and "falls" in row["detail"]
    assert (
        "FrailtyFitter.fit",
        "CoxPH.fit(strata=)",
        "summary(robust=)",
    ) in _features(d)
    assert "WeibullFrailty.fit on its rows" in d.advice[0]
    assert not d.ok


def test_a_continuous_effect_that_changes_is_advised_fit_tvc():
    df = _simulate(
        np.random.default_rng(3),
        before=lambda c: 1.0 * c["z"],
        after=lambda c: -0.3 * c["z"],
        cut=8,
    )
    d = _fit(df).diagnose()
    assert _row(d, "proportional_hazards", ("z",))["verdict"] == "fail"
    assert ("CoxPH.fit_tvc", "summary(robust=)") in _features(d)


def test_a_joint_change_alone_is_advised():
    two = {}

    def covariates(rng, n):
        two.update(
            a=rng.binomial(1, 0.5, n).astype(float),
            b=rng.binomial(1, 0.5, n).astype(float),
        )
        return dict(two)

    df = _simulate(
        np.random.default_rng(101),
        G=40,
        before=lambda c: 0.4 * c["a"] + 0.4 * c["b"],
        after=lambda c: 0.0 * c["a"],
        cut=8,
        covariates=covariates,
    )
    d = _fit(df, cols=("a", "b")).diagnose()
    ph = _rows(d, "proportional_hazards")
    assert list(ph["verdict"]) == ["warn", "warn", "fail"]
    joint = [a for a in d.advice if a.check == "proportional_hazards"]
    assert [a.features for a in joint] == [
        ("CoxPH.fit(strata=)", "CoxPH.fit_tvc", "summary(robust=)")
    ]
    assert "GLOBAL" in joint[0]


def test_a_curved_effect_is_advised_a_transform():
    df = _simulate(
        np.random.default_rng(4), before=lambda c: 0.8 * c["z"] ** 2
    )
    d = _fit(df).diagnose()
    row = _row(d, "functional_form", ("z",))
    assert row["verdict"] == "fail" and "not linear" in row["detail"]
    assert _features(d) == [("FrailtyFitter.fit_from_df(formula=)",)]
    assert "WeibullFrailty.fit_from_df" in d.advice[0]


def test_a_better_baseline_is_advised():
    from scipy.stats import norm

    df = _simulate(
        np.random.default_rng(5),
        baseline=lambda s: np.exp(2 + 0.5 * norm.isf(np.exp(-s))),
    )
    d = _fit(df).diagnose()
    row = _row(d, "baseline_distribution")
    assert row["verdict"] == "warn" and row["statistic"] > 2
    assert _features(d) == [("Frailty(family=)",)]
    assert "Frailty(LogNormal, family='gamma')" in d.advice[0]


# ---------------------------------------------------------------------------
# The Cox frailty model
# ---------------------------------------------------------------------------
def test_cox_frailty_report(cox_d):
    d = cox_d
    assert d.family == "Cox shared frailty" and d.model == "CoxFrailtyModel"
    assert list(d.table["check"]) == ["frailty_variance"]
    assert d.ok and not d.advice
    text = " ".join(d.notes)
    assert "does not keep" in text and "gamma frailty only" in text


def test_a_fixed_frailty_variance_is_not_tested(kidney):
    model = _kidney_fit(sp.CoxFrailty, kidney, theta=0.5)
    d = model.diagnose()
    assert d.table.empty
    assert any("fixed (theta=0.5)" in n for n in d.notes)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fitter", [sp.WeibullFrailty, sp.CoxFrailty])
def test_a_restored_model_is_refused(kidney, fitter):
    model = _kidney_fit(fitter, kidney)
    restored = type(model).from_dict(model.to_dict())
    with pytest.raises(ValueError, match="from_dict.*Refit"):
        restored.diagnose()
    with pytest.raises(ValueError, match="from_dict"):
        frailty.zero_variance_test(restored)


@pytest.mark.parametrize(
    "maximum, match",
    [("no finite maximum", "finite maximum"), ("unverified", "verified")],
)
@pytest.mark.parametrize("fitter", [sp.WeibullFrailty, sp.CoxFrailty])
def test_a_fit_without_a_verified_maximum_is_refused(
    kidney, fitter, maximum, match
):
    model = copy.copy(_kidney_fit(fitter, kidney))
    model.maximum = maximum
    with pytest.raises(ValueError, match=match):
        model.diagnose()


@pytest.mark.parametrize("fitter", [sp.WeibullFrailty, sp.CoxFrailty])
def test_an_aliased_coefficient_is_refused(kidney, fitter):
    df = kidney.assign(age2=2 * kidney["age"])
    model = _quiet(
        fitter.fit_from_df,
        df,
        x_col="time",
        c_col="c",
        group_col="id",
        Z_cols=["age", "female", "age2"],
    )
    assert np.isnan(model.beta).any()
    with pytest.raises(ValueError, match="aliased.*Drop"):
        model.diagnose()


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------
def test_plot_draws_every_panel(gamma_weibull_d, cox_d):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # the frailties, age's functional form, the AIC of each baseline and
    # the Cox-Snell residuals
    fig = gamma_weibull_d.plot()
    assert len(fig.axes) == 4
    plt.close(fig)
    fig = gamma_weibull_d.plot("overall_fit")
    assert len(fig.axes) == 1
    plt.close(fig)
    fig = cox_d.plot()
    assert len(fig.axes) == 2
    plt.close(fig)
    with pytest.raises(ValueError, match="which"):
        cox_d.plot("functional_form")


def test_to_dict_round_trips(gamma_weibull_d, cox_d):
    for d in (gamma_weibull_d, cox_d):
        out = d.to_dict()
        back = Diagnosis.from_dict(
            json.loads(json.dumps(out, allow_nan=False))
        )
        assert back.to_dict() == out
        pd.testing.assert_frame_equal(back.table, d.table)
        assert back.notes == d.notes and back.ok == d.ok


def test_columns_and_alpha(gamma_weibull, gamma_weibull_d):
    d = gamma_weibull.diagnose(columns=["age"])
    ph = _rows(d, "proportional_hazards")
    assert list(ph["covariates"]) == [("age",), ("age", "female")]
    assert_allclose(
        ph["statistic"],
        _rows(gamma_weibull_d, "proportional_hazards")["statistic"].iloc[
            [0, 2]
        ],
    )
    assert d.columns == ["age"]
    # theta's p-value is 0.0016: not significant at 0.001
    strict = gamma_weibull.diagnose(alpha=0.001)
    assert _row(strict, "frailty_variance")["verdict"] == "warn"
    assert ("WeibullPH.fit", "WeibullPH.fit_from_df") in _features(strict)


def test_a_formula_fit_names_its_columns():
    df = load_kidney()
    df["censored"] = 1 - df["status"]
    model = _quiet(
        sp.WeibullFrailty.fit_from_df,
        df,
        x_col="time",
        c_col="censored",
        group_col="id",
        formula="age + C(sex)",
    )
    d = model.diagnose()
    assert d.columns == ["age", "C(sex)[T.2]"]
    assert ("C(sex)[T.2]",) in set(d.table["covariates"])


def test_the_fitter_names():
    model = sp.Frailty(sp.Gumbel, family="lognormal")
    m = copy.copy(_fit(_simulate(np.random.default_rng(9), G=20), model))
    assert frailty.frailty_fitter_name(m) == (
        "Frailty(Gumbel, family='lognormal')"
    )
    assert frailty.null_fitter_name(m) == "GumbelPH"
