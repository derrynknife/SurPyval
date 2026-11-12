"""The mixed-effects Cox model (``CoxMixedEffects``) and the log-normal Cox
frailty (``CoxFrailty.fit(family="lognormal")``). The comparisons with R's
coxme are in surpyval/tests/reference/test_coxme.py."""

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import chi2, norm

from surpyval import CoxFrailty, CoxMixedEffects, CoxPH, RandomEffect
from surpyval.datasets import load_kidney
from surpyval.univariate.regression.frailty.cox_mixed_effects import (
    CoxMixedEffectsModel,
    parse_random,
)


@pytest.fixture(scope="module")
def kidney():
    df = load_kidney()
    Z = np.column_stack([df["age"], df["sex"] == 2]).astype(float)
    return df["time"].to_numpy(float), Z, 1 - df["status"].to_numpy(), df


@pytest.fixture(scope="module")
def kidney_fit(kidney):
    x, Z, c, df = kidney
    return CoxMixedEffects.fit(x, Z=Z, c=c, groups=df["id"])


def _simulate(seed, G=30, per=6, sd=0.6, beta=0.7, slope_sd=0.0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(G), per)
    z = rng.binomial(1, 0.5, g.size).astype(float)
    b = rng.normal(0, sd, G)[g] + rng.normal(0, slope_sd, G)[g] * z
    t = rng.exponential(1 / (0.1 * np.exp(beta * z + b)))
    u = rng.uniform(0, 20, g.size)
    return np.minimum(t, u), z[:, None], (u < t).astype(int), g


def test_zero_variance_is_the_cox_model(kidney):
    x, Z, c, df = kidney
    model = CoxMixedEffects.fit(x, Z=Z, c=c, groups=df["id"], variance=[0])
    cox = CoxPH.fit(x, Z, c)
    np.testing.assert_allclose(model.beta, cox.beta, rtol=1e-7)
    np.testing.assert_allclose(
        model.standard_errors(), cox.standard_errors(), rtol=1e-6
    )
    assert model.log_likelihood == pytest.approx(
        model.log_likelihood_no_random, abs=1e-9
    )
    np.testing.assert_allclose(
        model.sf([50, 150], Z[:2]), cox.sf([50, 150], Z[:2]), rtol=1e-7
    )


def test_kidney_estimates_and_likelihoods(kidney_fit):
    m = kidney_fit
    assert m.variance_components()["variance"].iloc[0] == pytest.approx(
        0.45736, abs=1e-4
    )
    assert m.log_likelihood_penalised <= m.log_likelihood_fitted
    assert m.log_likelihood_null < m.log_likelihood_no_random
    assert m.k == 3
    assert m.aic() == pytest.approx(-2 * m.log_likelihood + 6)
    test = m.lr_test()
    stat = 2 * (m.log_likelihood - m.log_likelihood_no_random)
    assert test["statistic"] == pytest.approx(stat)
    assert test["p_value"] == pytest.approx(0.5 * chi2.sf(stat, 1))
    assert test["reference"] == "0.5 chi2(0) + 0.5 chi2(1)"


def test_lognormal_cox_frailty_is_the_one_term_model(kidney, kidney_fit):
    x, Z, c, df = kidney
    frailty = CoxFrailty.fit(x, Z=Z, c=c, groups=df["id"], family="lognormal")
    m = kidney_fit
    assert frailty.family == "lognormal"
    np.testing.assert_allclose(frailty.beta, m.beta, rtol=1e-6)
    assert frailty.theta == pytest.approx(
        m.variance_components()["variance"].iloc[0], rel=1e-5
    )
    assert frailty.log_likelihood == pytest.approx(m.log_likelihood, abs=1e-8)
    assert frailty.log_likelihood_no_frailty == pytest.approx(
        m.log_likelihood_no_random, abs=1e-8
    )
    assert np.isfinite(frailty.standard_errors()[-1])
    # The marginal curve integrates the same log-normal frailty out
    np.testing.assert_allclose(
        frailty.sf([30, 100], [45, 1]),
        m.sf([30, 100], [45, 1], marginal=True),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        frailty.sf([30, 100], [45, 1], group=21),
        m.sf([30, 100], [45, 1], group=21),
        rtol=1e-6,
    )
    restored = type(frailty).from_dict(
        json.loads(json.dumps(frailty.to_dict()))
    )
    assert restored.family == "lognormal"
    np.testing.assert_allclose(
        restored.sf([30, 100], [45, 1]), frailty.sf([30, 100], [45, 1])
    )
    d = frailty.diagnose()
    assert "frailty_variance" in set(d.table["check"])


def test_lognormal_cox_frailty_fixed_theta_and_refusals(kidney):
    x, Z, c, df = kidney
    zero = CoxFrailty.fit(
        x, Z=Z, c=c, groups=df["id"], family="lognormal", theta=0
    )
    np.testing.assert_allclose(zero.beta, CoxPH.fit(x, Z, c).beta, rtol=1e-7)
    with pytest.raises(ValueError, match="family"):
        CoxFrailty.fit(x, Z=Z, c=c, groups=df["id"], family="weibull")
    with pytest.raises(ValueError, match="at most"):
        CoxFrailty.fit(
            x, Z=Z, c=c, groups=df["id"], family="lognormal", theta=1e4
        )


def test_penalised_maximum_and_laplace_term(kidney_fit):
    # At the fit the penalised likelihood's gradient is 0, and the
    # integrated likelihood is the penalised one less half the log
    # determinant of its information in the random effects.
    from surpyval.univariate.regression.frailty.cox_mixed_effects import (
        solution_of,
    )

    m = kidney_fit
    sol = solution_of(m)
    engine = m._engine
    L = engine.factor(sol.psi)
    grad, H, _ = engine.derivatives(sol.beta, sol.a, L)
    assert np.max(np.abs(grad)) < 1e-8
    p = engine.p
    logdet = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(H[p:, p:]))))
    assert m.log_likelihood == pytest.approx(
        m.log_likelihood_penalised - 0.5 * logdet, abs=1e-10
    )


def test_row_order_and_counts_do_not_matter():
    x, Z, c, g = _simulate(1)
    model = CoxMixedEffects.fit(x, Z=Z, c=c, groups=g)
    order = np.random.default_rng(2).permutation(x.size)
    shuffled = CoxMixedEffects.fit(
        x[order], Z=Z[order], c=c[order], groups=g[order]
    )
    assert shuffled.log_likelihood == pytest.approx(
        model.log_likelihood, abs=1e-8
    )
    np.testing.assert_allclose(shuffled.beta, model.beta, rtol=1e-5)
    # A row given twice is a row with a count of 2
    twice = np.r_[np.arange(x.size), np.arange(10)]
    n = np.ones(x.size)
    n[:10] = 2
    doubled = CoxMixedEffects.fit(
        x[twice], Z=Z[twice], c=c[twice], groups=g[twice]
    )
    counted = CoxMixedEffects.fit(x, Z=Z, c=c, n=n, groups=g)
    assert counted.log_likelihood == pytest.approx(
        doubled.log_likelihood, abs=1e-8
    )


def test_conditional_and_marginal_predictions(kidney_fit, kidney):
    m = kidney_fit
    t = np.array([20.0, 80.0, 300.0])
    z = np.array([45.0, 1.0])
    beta = m.beta
    base = m.baselines[0]
    H0 = np.array([base["H0"][base["x"] <= ti][-1] for ti in t])
    lp = z @ beta
    # A new group at random effect 0
    np.testing.assert_allclose(m.Hf(t, z), np.exp(lp) * H0, rtol=1e-12)
    # An observed group: its predicted random effect
    b = m.random_effects["group"].loc["21"].iloc[0]
    np.testing.assert_allclose(
        m.Hf(t, z, group=21), np.exp(lp + b) * H0, rtol=1e-12
    )
    # Marginal: the random effect integrated out
    v = m.variance_components()["variance"].iloc[0]
    for ti, h in zip(t, np.exp(lp) * H0):
        exact = quad(
            lambda w: np.exp(-h * np.exp(w)) * norm.pdf(w, 0, np.sqrt(v)),
            -12,
            12,
            epsabs=1e-13,
        )[0]
        assert m.sf(ti, z, marginal=True) == pytest.approx(exact, rel=1e-8)
    # hf is the jump of Hf; qf inverts ff at a step
    jumps = m.hf(base["x"][:5], z)
    np.testing.assert_allclose(
        np.cumsum(jumps), m.Hf(base["x"][:5], z), rtol=1e-12
    )
    q = m.qf(0.5, z)
    assert m.ff(q, z) >= 0.5
    assert m.df(t, z) == pytest.approx(m.hf(t, z) * m.sf(t, z))
    # Missing and unknown groups
    assert np.all(np.isnan(m.sf(t, z, group=np.nan)))
    with pytest.raises(KeyError, match="Unknown group"):
        m.sf(t, z, group="nobody")


def test_nested_crossed_slopes_and_strata_predictions():
    rng = np.random.default_rng(5)
    x, Z, c, g = _simulate(7, G=12, per=20, slope_sd=0.3)
    ward = rng.integers(0, 3, x.size)
    batch = rng.integers(0, 4, x.size)
    sex = rng.integers(0, 2, x.size)
    model = CoxMixedEffects.fit(
        x,
        Z=Z,
        c=c,
        strata=sex,
        random_effects=[
            RandomEffect(g, slopes=[0], name="site"),
            RandomEffect((g, ward), name="site:ward"),
            RandomEffect(batch, name="batch"),
        ],
    )
    assert [t["name"] for t in model.terms] == ["site", "site:ward", "batch"]
    assert list(model.random_effects["site"].columns) == [
        "(Intercept)",
        "Z[:, 0]",
    ]
    assert model.random_effects["site:ward"].shape == (36, 1)
    table = model.variance_components()
    assert ("site", "corr((Intercept), Z[:, 0])") in table.index
    with pytest.raises(ValueError, match="stratum"):
        model.sf(5.0, [1.0])
    with pytest.raises(ValueError, match="dict"):
        model.sf(5.0, [1.0], group=0, stratum=0)
    s_new = model.sf(5.0, [1.0], stratum=0)
    s_site = model.sf(5.0, [1.0], group={"site": 3}, stratum=0)
    b = model.random_effects["site"].loc["3"].to_numpy()
    assert -np.log(s_site) == pytest.approx(-np.log(s_new) * np.exp(b.sum()))
    # The marginal over every term: the random part's variance adds up
    marginal = model.sf(5.0, [1.0], marginal=True, stratum=1)
    assert (
        np.isfinite(marginal)
        and marginal > model.sf(5.0, [1.0], stratum=1) - 1
    )
    d = model.to_dict()
    restored = CoxMixedEffectsModel.from_dict(json.loads(json.dumps(d)))
    for kw in ({}, {"marginal": True}, {"group": {"site:ward": "3:1"}}):
        np.testing.assert_allclose(
            restored.sf([2.0, 9.0], [1.0], stratum=1, **kw),
            model.sf([2.0, 9.0], [1.0], stratum=1, **kw),
        )
    with pytest.raises(ValueError, match="Refit"):
        restored.variance_cb()


def test_formula_paths_give_the_same_fit():
    x, Z, c, g = _simulate(3, slope_sd=0.3)
    df = pd.DataFrame({"x": x, "c": c, "z": Z[:, 0], "g": g})
    a = CoxMixedEffects.fit_from_df(
        df,
        x_col="x",
        c_col="c",
        Z_cols=["z"],
        random_effects="(1 + z || g)",
    )
    b = CoxMixedEffects.fit(
        x,
        Z=Z,
        c=c,
        random_effects=RandomEffect(g, slopes=[0], correlated=False),
    )
    assert a.log_likelihood == pytest.approx(b.log_likelihood, abs=1e-9)
    assert a.random_effects["g"].columns.tolist() == ["(Intercept)", "z"]
    c_formula = CoxMixedEffects.fit_from_df(
        df, x_col="x", c_col="c", formula="z", random_effects="(1 + z || g)"
    )
    assert c_formula.log_likelihood == pytest.approx(
        a.log_likelihood, abs=1e-9
    )
    new = pd.DataFrame({"z": [0.0, 1.0]})
    np.testing.assert_allclose(
        a.sf([3.0, 3.0], new), b.sf([3.0, 3.0], [[0.0], [1.0]])
    )


def test_parse_random():
    terms = parse_random("(1 + trt | centre) + (1 | hosp/ward) + (0 + x || b)")
    assert [t["groups"] for t in terms] == [
        ["centre"],
        ["hosp"],
        ["hosp", "ward"],
        ["b"],
    ]
    assert terms[0]["slopes"] == ["trt"] and terms[0]["intercept"]
    assert not terms[3]["intercept"] and not terms[3]["correlated"]
    assert parse_random("(1 | a:b)")[0]["groups"] == ["a", "b"]
    with pytest.raises(ValueError, match="intercept"):
        parse_random("(trt | centre)")
    with pytest.raises(ValueError, match="bar"):
        parse_random("1 | centre")
    with pytest.raises(ValueError, match="grouping"):
        parse_random("(1 | )")


def test_refusals(kidney, kidney_fit):
    x, Z, c, df = kidney
    with pytest.raises(ValueError, match="groups= .*random_effects="):
        CoxMixedEffects.fit(x, Z=Z, c=c)
    with pytest.raises(ValueError, match="right-censored"):
        CoxMixedEffects.fit(x, Z=Z, c=np.where(c == 1, 2, 0), groups=df["id"])
    with pytest.raises(ValueError, match="slope"):
        CoxMixedEffects.fit(
            x, Z=Z, c=c, random_effects=RandomEffect(df["id"], slopes=[5])
        )
    with pytest.raises(ValueError, match="one entry per"):
        CoxMixedEffects.fit(x, Z=Z, c=c, groups=df["id"], variance=0.5)
    with pytest.raises(ValueError, match="non-negative"):
        CoxMixedEffects.fit(x, Z=Z, c=c, groups=df["id"], variance=[-1])
    with pytest.raises(ValueError, match="at least two groups"):
        CoxMixedEffects.fit(x, Z=Z, c=c, groups=np.ones(x.size))
    with pytest.raises(ValueError, match="tie_method"):
        CoxMixedEffects.fit(x, Z=Z, c=c, groups=df["id"], tie_method="exact")
    with pytest.raises(ValueError, match="intercept or at least one slope"):
        RandomEffect(df["id"], intercept=False)
    with pytest.raises(ValueError, match="group_col= .* random_effects="):
        CoxMixedEffects.fit_from_df(df, x_col="time")
    with pytest.raises(NotImplementedError, match="Frailty"):
        kidney_fit.random(5, [45, 1])
    with pytest.raises(NotImplementedError, match="lr_test"):
        kidney_fit.diagnose()
    with pytest.raises(ValueError, match="variance_cb"):
        kidney_fit.param_cb("coef_0", method="lr")
    fixed = CoxMixedEffects.fit(x, Z=Z, c=c, groups=df["id"], variance=[0.5])
    with pytest.raises(ValueError, match="fixed"):
        fixed.variance_cb()
    assert np.isnan(fixed.lr_test()["p_value"])
    assert fixed.k == 2


def test_missing_labels_are_dropped_with_a_warning(kidney):
    x, Z, c, df = kidney
    ids = df["id"].to_numpy(dtype=object).copy()
    ids[:3] = None
    with pytest.warns(UserWarning, match="Dropped 3 of 76"):
        model = CoxMixedEffects.fit(x, Z=Z, c=c, groups=ids)
    assert model.n_obs == 73


def test_variance_bounds_are_where_the_profile_drops(kidney, kidney_fit):
    x, Z, c, df = kidney
    m = kidney_fit
    table = m.variance_cb()
    row = table.iloc[0]
    assert row["lower"] < row["estimate"] < row["upper"]
    crit = chi2.ppf(0.95, 1) / 2
    for bound in (row["lower"], row["upper"]):
        at = CoxMixedEffects.fit(
            x, Z=Z, c=c, groups=df["id"], variance=[bound]
        )
        assert m.log_likelihood - at.log_likelihood == pytest.approx(
            crit, abs=1e-4
        )


def test_no_group_effect_sits_on_the_boundary():
    x, Z, c, g = _simulate(11, sd=0.0)
    model = CoxMixedEffects.fit(x, Z=Z, c=c, groups=g)
    var = model.variance_components()["variance"].iloc[0]
    if var == 0:
        assert model.lr_test()["p_value"] == 1.0
        assert model.variance_cb()["lower"].iloc[0] == 0.0
    assert model.lr_test()["p_value"] > 0.05


def test_correlated_terms_use_the_conservative_reference():
    x, Z, c, g = _simulate(4, slope_sd=0.4)
    model = CoxMixedEffects.fit(
        x, Z=Z, c=c, random_effects=RandomEffect(g, slopes=[0])
    )
    test = model.lr_test()
    assert test["df"] == 3
    assert "conservative" in test["reference"]
    table = model.variance_cb()
    assert len(table) == 3


def test_pickle_and_repr(kidney_fit):
    text = repr(kidney_fit)
    assert "Mixed-Effects Cox" in text and "var((Intercept))" in text
    again = pickle.loads(pickle.dumps(kidney_fit))
    np.testing.assert_allclose(
        again.sf([30, 100], [45, 1]), kidney_fit.sf([30, 100], [45, 1])
    )
    assert repr(RandomEffect([1, 2], name="g")) == "RandomEffect(g: intercept)"
