"""Directly adjusted survival curves (``direct_adjustment``) and the
checks on weights (``weight_diagnostics``).

The comparisons with R are in ``tests/reference/test_adjusted_curves.py``.
"""

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import surpyval as sp


def _confounded(seed, n):
    """Units whose covariate ``a`` raises both the chance of group 1 and
    the hazard: Weibull PH, shape 1.5, scale 10, log hazard ratio 0.5
    for the group and 0.6 per unit of ``a``; censored at 12."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    propensity = 1.0 / (1.0 + np.exp(-1.2 * a))
    g = rng.binomial(1, propensity)
    x = 10.0 * rng.weibull(1.5, n) * np.exp(-(0.5 * g + 0.6 * a) / 1.5)
    c = (x > 12.0).astype(int)
    return np.minimum(x, 12.0), c, g, a, propensity


def _truth(t, g):
    """The survival at ``t`` of the whole population in group ``g``:
    the model's survival averaged over ``a ~ N(0, 1)`` (Gauss-Hermite)."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(80)
    weights = weights / weights.sum()
    H = (np.asarray(t)[:, None] / 10.0) ** 1.5 * np.exp(0.5 * g + 0.6 * nodes)
    return np.exp(-H) @ weights


# -- a hand-checked case ------------------------------------------------------


def test_exponential_two_groups_closed_form():
    # With the group as the only covariate every unit has the same
    # prediction, so the adjusted curve is the group's own exponential
    # curve, rate = failures / time on test, and log(rate) has variance
    # 1 / failures.
    x = np.array([2.0, 3.0, 5.0, 8.0, 4.0, 1.0, 2.0, 2.0, 3.0, 6.0])
    c = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
    g = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    model = sp.ExponentialPH.fit(x, g.reshape(-1, 1), c=c)
    adjusted = sp.direct_adjustment(model, group=0)
    assert adjusted.levels == [0.0, 1.0]

    rate = {0: Fraction(4, 22), 1: Fraction(4, 14)}
    t, tau = 3.0, 5.0
    S, se_S, mu, se_mu = {}, {}, {}, {}
    for k in (0, 1):
        lam = float(rate[k])
        S[k] = np.exp(-lam * t)
        se_S[k] = S[k] * lam * t / np.sqrt(4)
        mu[k] = (1 - np.exp(-lam * tau)) / lam
        se_mu[k] = abs(tau * np.exp(-lam * tau) - mu[k]) / np.sqrt(4)
        assert_allclose(adjusted.sf(t, k), S[k], rtol=1e-6)
        assert_allclose(adjusted.ff(t, k), 1 - S[k], rtol=1e-6)
        res = adjusted.rmst(tau, k)
        assert_allclose(res["rmst"], mu[k], rtol=1e-6)
        assert_allclose(res["se"], se_mu[k], rtol=1e-4)
        assert res["lower"] < res["rmst"] < res["upper"] < tau

    res = adjusted.sf_diff(t, 1, 0)
    assert_allclose(res["difference"], S[1] - S[0], rtol=1e-6)
    assert_allclose(res["se"], np.hypot(se_S[0], se_S[1]), rtol=1e-4)
    assert_allclose(res["lower"], res["difference"] - 1.959964 * res["se"])
    assert_allclose(res["sf_a"] - res["sf_b"], res["difference"])
    res = adjusted.rmst_diff(tau, 1, 0)
    assert_allclose(res["difference"], mu[1] - mu[0], rtol=1e-6)
    assert_allclose(res["se"], np.hypot(se_mu[0], se_mu[1]), rtol=1e-4)
    assert_allclose(res["ratio"], mu[1] / mu[0], rtol=1e-6)
    assert res["tau"] == tau


def test_curve_is_the_weighted_mean_of_the_predictions():
    x, c, g, a, _ = _confounded(7, 60)
    n = np.arange(60) % 3 + 1
    Z = np.column_stack([g, a])
    model = sp.WeibullPH.fit(x, Z, c=c, n=n)
    adjusted = sp.direct_adjustment(model, group=0)
    t = np.array([2.0, 6.0, 11.0])
    Z1 = Z.astype(float).copy()
    Z1[:, 0] = 1.0
    by_hand = [np.sum(n * model.sf(np.full(60, s), Z1)) / n.sum() for s in t]
    assert_allclose(adjusted.sf(t, 1), by_hand, rtol=1e-12)
    # counts are the rows repeated
    rep = np.repeat(np.arange(60), n)
    repeated = sp.direct_adjustment(
        sp.WeibullPH.fit(x[rep], Z[rep], c=c[rep]), group=0
    )
    assert_allclose(repeated.sf(t, 1), adjusted.sf(t, 1), rtol=1e-6)
    assert_allclose(
        repeated.sf_diff(t, 1, 0)["se"],
        adjusted.sf_diff(t, 1, 0)["se"],
        rtol=1e-4,
    )
    assert_allclose(
        repeated.rmst_diff(10.0, 1, 0)["se"],
        adjusted.rmst_diff(10.0, 1, 0)["se"],
        rtol=1e-4,
    )


def test_one_row_population_is_the_models_own_curve():
    x, c, g, a, _ = _confounded(8, 150)
    Z = np.column_stack([g, a])
    t = np.array([3.0, 7.0])
    cox = sp.CoxPH.fit(x, Z, c=c)
    one = sp.direct_adjustment(
        cox, group=0, levels=[1], Z=[[0.0, 0.4]], covariates="fixed"
    )
    assert_allclose(one.sf(t, 1), cox.sf(t, [1.0, 0.4]), rtol=1e-12)
    plain = cox.cb(t, [1.0, 0.4], bound_type="plain")
    se_model = (plain[:, 1] - plain[:, 0]) / (2 * 1.959963984540054)
    _, var = one._sf_and_var(t, np.array([1.0]))
    assert_allclose(np.sqrt(var), se_model, rtol=1e-9)

    weibull = sp.WeibullPH.fit(x, Z, c=c)
    one = sp.direct_adjustment(
        weibull, group=0, levels=[1], Z=[[0.0, 0.4]], covariates="fixed"
    )
    assert_allclose(
        one.rmst(9.0, 1)["rmst"], weibull.rmst(9.0, [1.0, 0.4])["rmst"]
    )
    assert_allclose(
        one.rmst(9.0, 1)["se"], weibull.rmst(9.0, [1.0, 0.4])["se"], rtol=1e-8
    )


# -- recovery -----------------------------------------------------------------


@pytest.mark.parametrize("fitter", ["WeibullPH", "CoxPH"])
def test_recovers_the_population_curves(fitter):
    x, c, g, a, propensity = _confounded(135, 2000)
    Z = np.column_stack([g, a])
    model = getattr(sp, fitter).fit(x, Z, c=c)
    adjusted = sp.direct_adjustment(model, group=0)
    t = np.array([3.0, 6.0, 9.0])
    for k in (0, 1):
        cb = adjusted.cb(t, k)
        truth = _truth(t, k)
        assert np.all((cb[:, 0] < truth) & (truth < cb[:, 1]))
    res = adjusted.sf_diff(t, 1, 0)
    truth = _truth(t, 1) - _truth(t, 0)
    assert np.all((res["lower"] < truth) & (truth < res["upper"]))
    # the unadjusted difference is confounded: further from the truth
    km = [sp.KaplanMeier.fit(x[g == k], c=c[g == k]) for k in (0, 1)]
    naive = km[1].sf(t) - km[0].sf(t)
    assert np.all(np.abs(naive - truth) > np.abs(res["difference"] - truth))
    # and the weighted Kaplan-Meier difference agrees with the truth too
    w = np.where(g == 1, 1 / propensity, 1 / (1 - propensity))
    ipw = [
        sp.KaplanMeier.fit(x[g == k], c=c[g == k], w=w[g == k]) for k in (0, 1)
    ]
    for s, target in zip(t, truth):
        weighted = sp.sf_diff(ipw[1], ipw[0], s, transform="linear")
        assert weighted["lower"] < target < weighted["upper"]


def test_cox_standard_errors_match_the_spread_over_replicates():
    # 400 data sets of 200 units from a model the Cox fit is right for:
    # the mean standard error should be the standard deviation of the
    # estimates (whose own relative error over 400 draws is about 3.5%).
    rng = np.random.default_rng(20261111)
    n, reps = 200, 400
    t = np.array([5.0, 10.0])
    est, se, diff, diff_se, rm, rm_se = [], [], [], [], [], []
    for _ in range(reps):
        a = rng.normal(size=n)
        g = rng.binomial(1, 1 / (1 + np.exp(-a)))
        x = 10 * rng.weibull(1.5, n) * np.exp(-(0.5 * g + 0.4 * a) / 1.5)
        cens = rng.uniform(5, 25, n)
        c = (x > cens).astype(int)
        model = sp.CoxPH.fit(np.minimum(x, cens), np.c_[g, a], c=c)
        adjusted = sp.direct_adjustment(model, 0)
        S, var = adjusted._sf_and_var(t, np.array([0.0, 1.0]))
        est.append(S)
        se.append(np.sqrt(var))
        res = adjusted.sf_diff(t, 1, 0)
        diff.append(res["difference"])
        diff_se.append(res["se"])
        res = adjusted.rmst_diff(12.0, 1, 0)
        rm.append(res["difference"])
        rm_se.append(res["se"])
    assert_allclose(np.mean(se, axis=0), np.std(est, axis=0, ddof=1), rtol=0.1)
    assert_allclose(
        np.mean(diff_se, axis=0), np.std(diff, axis=0, ddof=1), rtol=0.1
    )
    assert_allclose(np.mean(rm_se), np.std(rm, ddof=1), rtol=0.1)


# -- how the group and the population are given -------------------------------


def test_named_columns_formula_and_interaction():
    x, c, g, a, _ = _confounded(9, 300)
    df = pd.DataFrame({"x": x, "c": c, "g": g, "a": a})
    by_position = sp.direct_adjustment(
        sp.WeibullPH.fit(x, np.column_stack([g, a]), c=c), group=0
    )
    named = sp.direct_adjustment(
        sp.WeibullPH.fit_from_df(df, x_col="x", c_col="c", Z_cols=["g", "a"]),
        group="g",
    )
    formula_model = sp.WeibullPH.fit_from_df(
        df, x_col="x", c_col="c", formula="g + a"
    )
    formula = sp.direct_adjustment(formula_model, group="g", Z=df)
    t = [2.0, 8.0]
    assert_allclose(named.sf(t, 1), by_position.sf(t, 1), rtol=1e-6)
    assert_allclose(formula.sf(t, 1), by_position.sf(t, 1), rtol=1e-6)
    assert_allclose(
        formula.sf_diff(t, 1, 0)["se"],
        by_position.sf_diff(t, 1, 0)["se"],
        rtol=1e-4,
    )
    with pytest.raises(ValueError, match="fitted with a formula"):
        sp.direct_adjustment(formula_model, group="g")

    # An interaction with the group is rebuilt at each level
    inter = sp.WeibullPH.fit_from_df(df, x_col="x", c_col="c", formula="g * a")
    adjusted = sp.direct_adjustment(inter, group="g", Z=df)
    at_one = df.assign(g=1)
    by_hand = np.mean(inter.sf(np.full(300, 5.0), at_one))
    assert_allclose(adjusted.sf(5.0, 1), by_hand, rtol=1e-12)


def test_categorical_group_in_a_formula():
    x, c, g, a, _ = _confounded(10, 300)
    df = pd.DataFrame(
        {"x": x, "c": c, "a": a, "supplier": np.where(g == 1, "new", "old")}
    )
    model = sp.WeibullAFT.fit_from_df(
        df, x_col="x", c_col="c", formula="C(supplier) + a"
    )
    adjusted = sp.direct_adjustment(model, group="supplier", Z=df)
    assert adjusted.levels == ["new", "old"]
    by_hand = np.mean(model.sf(np.full(300, 5.0), df.assign(supplier="new")))
    assert_allclose(adjusted.sf(5.0, "new"), by_hand, rtol=1e-12)
    assert adjusted.sf_diff(5.0, "new", "old")["se"] > 0
    with pytest.raises(ValueError, match="not one of the adjusted levels"):
        adjusted.sf(5.0, "other")


def test_reference_population_and_weights():
    x, c, g, a, _ = _confounded(11, 200)
    model = sp.WeibullPH.fit(x, np.column_stack([g, a]), c=c)
    pop = np.array([[0.0, -1.0], [0.0, 0.0], [0.0, 2.0]])
    w = np.array([1.0, 2.0, 1.0])
    adjusted = sp.direct_adjustment(
        model, group=0, levels=[0, 1], Z=pop, w=w, covariates="fixed"
    )
    rows = np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 2.0]])
    by_hand = np.sum(w * model.sf(np.full(3, 4.0), rows)) / 4.0
    assert_allclose(adjusted.sf(4.0, 1), by_hand, rtol=1e-12)
    sampled = sp.direct_adjustment(model, group=0, levels=[0, 1], Z=pop, w=w)
    assert sampled.sf_diff(4.0, 1, 0)["se"] > adjusted.sf_diff(4.0, 1, 0)["se"]


def test_shapes_and_bounds():
    x, c, g, a, _ = _confounded(12, 200)
    model = sp.LogNormalAFT.fit(x, np.column_stack([g, a]), c=c)
    adjusted = sp.direct_adjustment(model, group=0)
    assert np.shape(adjusted.sf(3.0, 1)) == ()
    assert adjusted.sf([[1.0, 2.0], [3.0, 4.0]], 1).shape == (2, 2)
    assert adjusted.cb([1.0, 2.0, 3.0], 1).shape == (3, 2)
    assert adjusted.cb(2.0, 1).shape == (2,)
    assert adjusted.cb([1.0, 2.0], 1, bound="lower").shape == (2,)
    two = adjusted.cb(4.0, 1, alpha_ci=0.1)
    assert two[0] < adjusted.sf(4.0, 1) < two[1]
    assert_allclose(adjusted.cb(4.0, 1, alpha_ci=0.05, bound="lower"), two[0])
    assert_allclose(
        adjusted.cb(4.0, 1, on="ff", alpha_ci=0.1), 1 - two[::-1], rtol=1e-12
    )
    res = adjusted.sf_diff([2.0, 5.0], 1, 0)
    assert res["difference"].shape == (2,) and res["p_value"].shape == (2,)
    assert isinstance(adjusted.sf_diff(2.0, 1, 0)["difference"], float)
    assert "levels [0.0, 1.0]" in repr(adjusted)
    # the default horizon is the largest observed time
    assert adjusted.rmst(None, 1)["tau"] == np.max(x)


def test_robust_covariance_changes_only_the_errors():
    x, c, g, a, _ = _confounded(13, 200)
    Z = np.column_stack([g, a])
    for fitter in (sp.WeibullPH, sp.CoxPH):
        model = fitter.fit(x, Z, c=c)
        plain = sp.direct_adjustment(model, group=0).sf_diff(5.0, 1, 0)
        robust = sp.direct_adjustment(model, group=0, robust=True).sf_diff(
            5.0, 1, 0
        )
        assert robust["difference"] == plain["difference"]
        assert robust["se"] != plain["se"]
        assert_allclose(robust["se"], plain["se"], rtol=0.3)


# -- refusals -----------------------------------------------------------------


def test_refusals():
    x, c, g, a, _ = _confounded(14, 200)
    Z = np.column_stack([g, a])
    model = sp.WeibullPH.fit(x, Z, c=c)
    with pytest.raises(ValueError, match="not a column of Z"):
        sp.direct_adjustment(model, group=2)
    with pytest.raises(ValueError, match="covariates"):
        sp.direct_adjustment(model, group=0, covariates="random")
    with pytest.raises(ValueError, match="missing or infinite"):
        sp.direct_adjustment(model, group=0, Z=[[0.0, np.nan]])
    with pytest.raises(ValueError, match="'levels' must be a list"):
        sp.direct_adjustment(model, group=0, levels=1)
    with pytest.raises(ValueError, match="repeats a value"):
        sp.direct_adjustment(model, group=0, levels=[1, 1])
    adjusted = sp.direct_adjustment(model, group=0)
    with pytest.raises(ValueError, match="not one of the adjusted levels"):
        adjusted.sf(1.0, 2)
    with pytest.raises(ValueError, match="alpha_ci"):
        adjusted.cb(1.0, 1, alpha_ci=1.5)
    with pytest.raises(ValueError, match="not a finite time"):
        adjusted.rmst(-1.0, 1)
    with pytest.raises(ValueError, match="takes one tau"):
        adjusted.rmst([1.0, 2.0], 1)

    cox = sp.direct_adjustment(sp.CoxPH.fit(x, Z, c=c), group=0)
    with pytest.raises(ValueError, match="past the largest observed time"):
        cox.rmst(13.0, 1)
    stratified = sp.CoxPH.fit(x, Z, c=c, strata=(a > 0).astype(int))
    with pytest.raises(ValueError, match="stratified Cox"):
        sp.direct_adjustment(stratified, group=0)
    with pytest.raises(TypeError, match="fitted regression model"):
        sp.direct_adjustment(sp.Weibull.fit(x, c=c), group=0)


def test_restored_and_penalised_models():
    x, c, g, a, _ = _confounded(15, 200)
    Z = np.column_stack([g, a])
    model = sp.WeibullPH.fit(x, Z, c=c)
    full = sp.direct_adjustment(model, group=0, covariates="fixed")
    # A restored model keeps its covariance but not its rows
    restored = sp.from_dict(model.to_dict())
    with pytest.raises(ValueError, match="pass Z"):
        sp.direct_adjustment(restored, group=0)
    again = sp.direct_adjustment(restored, group=0, Z=Z, covariates="fixed")
    assert_allclose(again.sf([3.0, 6.0], 1), full.sf([3.0, 6.0], 1))
    assert_allclose(
        again.sf_diff(3.0, 1, 0)["se"],
        full.sf_diff(3.0, 1, 0)["se"],
        rtol=1e-6,
    )
    with pytest.raises(ValueError, match="pass tau"):
        again.rmst(None, 1)
    # A penalised fit has no covariance: curves, and a refusal for bounds
    penalised = sp.WeibullPH.fit_penalised(x, Z, c=c, penalizer=0.01)
    curves = sp.direct_adjustment(penalised, group=0)
    assert 0 < curves.sf(3.0, 1) < 1
    with pytest.raises(ValueError, match="[Pp]enalised"):
        curves.cb(3.0, 1)
    with pytest.raises(ValueError, match="[Pp]enalised"):
        curves.sf_diff(3.0, 1, 0)


def test_another_model_family_gives_curves_without_bounds():
    x, c, g, a, _ = _confounded(16, 200)
    Z = np.column_stack([g, a])
    model = sp.ProportionalOdds.fit(x, Z, c=c)
    adjusted = sp.direct_adjustment(model, group=0)
    Z1 = Z.astype(float).copy()
    Z1[:, 0] = 1.0
    assert_allclose(
        adjusted.sf(5.0, 1), np.mean(model.sf(np.full(200, 5.0), Z1))
    )
    with pytest.raises(ValueError, match="have no bounds"):
        adjusted.cb(5.0, 1)
    with pytest.raises(ValueError, match="restricted mean"):
        adjusted.rmst(5.0, 1)


def test_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, c, g, a, _ = _confounded(17, 150)
    Z = np.column_stack([g, a])
    for fitter in (sp.WeibullPH, sp.CoxPH):
        fig, ax = plt.subplots()
        adjusted = sp.direct_adjustment(fitter.fit(x, Z, c=c), group=0)
        out = adjusted.plot(ax=ax)
        assert out is ax
        assert ax.get_legend_handles_labels()[1] == ["0 = 0.0", "0 = 1.0"]
        assert ax.get_ylabel() == "Adjusted survival probability"
        assert len(ax.collections) == 2
        plt.close(fig)
    fig, ax = plt.subplots()
    adjusted.plot(x=[1.0, 2.0, 3.0], alpha_ci=None, ax=ax)
    assert len(ax.collections) == 0
    plt.close(fig)


# -- weight diagnostics -------------------------------------------------------


def test_weight_diagnostics_by_hand():
    z = [1, 1, 2, 1, 2, 2]
    group = ["a", "a", "a", "b", "b", "b"]
    w = [1, 1, 2, 2, 1, 1]
    res = sp.weight_diagnostics(w, group, z)
    groups = res["groups"]
    assert list(groups.index) == ["a", "b"]
    assert list(groups["n"]) == [3, 3]
    # Kish: (1 + 1 + 2)^2 / (1 + 1 + 4) = 16 / 6
    assert_allclose(groups["ess"], [16 / 6, 16 / 6])
    assert_allclose(groups["ess_ratio"], [8 / 9, 8 / 9])
    assert_allclose(groups["max_share"], [0.5, 0.5])
    assert list(groups["n_extreme"]) == [0, 0]
    balance = res["balance"]
    # means 4/3 and 5/3, both sample variances 1/3: -(1/3) / sqrt(1/3)
    assert_allclose(balance["smd_unweighted"], [-1 / np.sqrt(3)])
    # weighted means (1 + 1 + 4) / 4 and (2 + 2 + 2) / 4
    assert_allclose(balance["smd_weighted"], [0.0], atol=1e-15)
    assert_allclose(balance["variance_ratio"], [1.0])
    assert res["flags"] == []


def test_weight_diagnostics_flags_and_names():
    rng = np.random.default_rng(18)
    n = 300
    a = rng.normal(size=n)
    b = rng.binomial(1, 0.4, n)
    g = rng.binomial(1, 1 / (1 + np.exp(-1.5 * a)))
    p = 1 / (1 + np.exp(-1.5 * a))
    w = np.where(g == 1, 1 / p, 1 / (1 - p))
    Z = pd.DataFrame({"a": a, "b": b})
    res = sp.weight_diagnostics(w, g, Z)
    balance = res["balance"].set_index("covariate")
    assert list(balance.index) == ["a", "b"]
    # the true propensity weights balance the covariate that drove them
    assert abs(balance.loc["a", "smd_unweighted"]) > 0.5
    assert abs(balance.loc["a", "smd_weighted"]) < abs(
        balance.loc["a", "smd_unweighted"]
    )
    assert np.all(res["groups"]["ess"] < res["groups"]["n"])

    # equal weights: nothing to report
    res = sp.weight_diagnostics(np.ones(n), g)
    assert res["balance"] is None and res["flags"] == []
    assert_allclose(res["groups"]["ess"], res["groups"]["n"])

    # one unit carrying the group, and a covariate left unbalanced
    w = np.ones(n)
    w[np.flatnonzero(g == 1)[0]] = 500.0
    res = sp.weight_diagnostics(w, g, Z)
    text = " ".join(res["flags"])
    assert "effective sample size" in text
    assert "more than 10 times" in text
    assert "still unbalanced" in text
    assert res["groups"].loc[1, "n_extreme"] == 1

    # three groups: every pair
    three = np.arange(n) % 3
    res = sp.weight_diagnostics(np.ones(n), three, a)
    assert len(res["balance"]) == 3
    assert list(res["balance"]["covariate"]) == ["Z[:, 0]"] * 3


def test_weight_diagnostics_refusals():
    with pytest.raises(ValueError, match="positive and finite"):
        sp.weight_diagnostics([1.0, -1.0], [0, 1])
    with pytest.raises(ValueError, match="2 weights for 3 rows"):
        sp.weight_diagnostics([1.0, 1.0], [0, 1, 1])
    with pytest.raises(ValueError, match="one row of covariates per unit"):
        sp.weight_diagnostics([1.0, 1.0], [0, 1], [[1.0], [2.0], [3.0]])
    with pytest.raises(ValueError, match="must be numeric"):
        sp.weight_diagnostics([1.0, 1.0], [0, 1], [["x"], ["y"]])
    with pytest.raises(ValueError, match="missing or infinite"):
        sp.weight_diagnostics([1.0, 1.0], [0, 1], [1.0, np.nan])
    with pytest.raises(ValueError, match="'extreme' must be a positive"):
        sp.weight_diagnostics([1.0, 1.0], [0, 1], extreme=0)
    with pytest.raises(ValueError, match="one-dimensional"):
        sp.weight_diagnostics([1.0], [[0]])
