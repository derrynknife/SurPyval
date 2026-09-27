"""Regression tests for the third round of regression-model bug fixes."""

import json
import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

import surpyval
from surpyval import (
    AFT,
    PO,
    AdditiveHazards,
    BuckleyJames,
    CoxPH,
    Exponential,
    Gamma,
    GammaFrailty,
    LogisticPO,
    LogNormal,
    LogNormalPH,
    Weibull,
    WeibullAFT,
    WeibullAH,
    WeibullFrailty,
    WeibullPH,
)
from surpyval.datasets import load_tires_data
from surpyval.univariate.regression import (
    AcceleratedLife,
    DualPower,
    ExponentialLifeModel,
    InversePower,
    Linear,
    Power,
)
from surpyval.univariate.regression import _fit_skeleton
from surpyval.univariate.regression.additive_hazards.additive_hazards import (
    AdditiveHazardsModel,
)
from surpyval.univariate.regression.semi_parametric_regression_model import (
    SemiParametricRegressionModel,
)

matplotlib.use("Agg")

DROPPED = "Dropped 1 of"


def _ph_data(n: int = 200, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 1))
    x = 10 * rng.weibull(1.5, n) * np.exp(-0.5 * Z[:, 0] / 1.5)
    return x, Z


def _with_nan(Z: np.ndarray, value: float = np.nan) -> np.ndarray:
    Zn = np.array(Z, dtype=float, copy=True)
    Zn[0, 0] = value
    return Zn


# -- 1. Non-finite covariates are dropped, with a warning, everywhere --------


@pytest.mark.parametrize(
    "fitter", [WeibullPH, WeibullAFT, PO(Weibull), WeibullAH]
)
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_parametric_fitters_drop_nonfinite_covariate_rows(fitter, value):
    x, Z = _ph_data()
    with pytest.warns(UserWarning, match=DROPPED):
        model = fitter.fit(x=x, Z=_with_nan(Z, value))
    np.testing.assert_allclose(
        model.params, fitter.fit(x=x[1:], Z=Z[1:]).params, rtol=1e-6
    )
    assert model.res.success


def test_accelerated_life_drops_nonfinite_stress_rows():
    stress = np.repeat([20.0, 30.0, 40.0], 20)
    rng = np.random.default_rng(1)
    x = 10 * rng.weibull(3, 60) * (100.0 / stress)
    bad = stress.copy()
    bad[0] = np.nan
    with pytest.warns(UserWarning, match=DROPPED):
        model = AcceleratedLife(Weibull, Power).fit(x, Z=bad)
    ref = AcceleratedLife(Weibull, Power).fit(x[1:], Z=stress[1:])
    np.testing.assert_allclose(model.params, ref.params, rtol=1e-6)


def test_frailty_drops_nonfinite_covariate_rows_with_their_groups():
    x, Z = _ph_data()
    groups = np.repeat(np.arange(40), 5)
    with pytest.warns(UserWarning, match=DROPPED):
        model = WeibullFrailty.fit(x=x, Z=_with_nan(Z), groups=groups)
    ref = WeibullFrailty.fit(x=x[1:], Z=Z[1:], groups=groups[1:])
    np.testing.assert_allclose(model.beta, ref.beta, rtol=1e-6)
    assert np.isfinite(model.neg_ll())


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_semiparametric_fitters_warn_when_dropping_rows(value):
    x, Z = _ph_data()
    c = (x > 12).astype(int)
    Zn = _with_nan(Z, value)
    with pytest.warns(UserWarning, match=DROPPED):
        cox = CoxPH.fit(x, Zn, c=c)
    np.testing.assert_allclose(cox.beta, CoxPH.fit(x[1:], Z[1:], c=c[1:]).beta)
    with pytest.warns(UserWarning, match=DROPPED):
        ly = AdditiveHazards.fit(x, Zn, c=c)
    np.testing.assert_allclose(
        ly.beta, AdditiveHazards.fit(x[1:], Z[1:], c=c[1:]).beta
    )
    with pytest.warns(UserWarning, match=DROPPED):
        bj = BuckleyJames.fit(x, Zn, c=c)
    np.testing.assert_allclose(
        bj.beta, BuckleyJames.fit(x[1:], Z[1:], c=c[1:]).beta
    )


def test_optimise_ph_warns_instead_of_returning_a_failed_start(monkeypatch):
    from scipy.optimize import OptimizeResult

    def stuck(fun, x0, **kwargs):
        return OptimizeResult(
            x=np.asarray(x0), fun=fun(x0), success=False, message="forced"
        )

    monkeypatch.setattr(_fit_skeleton, "preconditioned_bfgs", stuck)
    monkeypatch.setattr(_fit_skeleton, "minimize", stuck)

    def fun(p):
        return ((p - 3.0) ** 2).sum()

    with pytest.warns(UserWarning, match="did not converge"):
        _fit_skeleton.optimise_ph(fun, np.zeros(2))


def test_nm_tnc_ladder_is_polished_to_the_maximum():
    # Nelder-Mead ran out of iterations (Gamma AFT, logistic PO) or TNC
    # reported success with a non-zero gradient (Weibull PO), and the fit
    # came back tenths of a nat -- or five nats -- short of the maximum.
    data = load_tires_data()
    x = data["Survival"].values
    c = data["Censoring"].values
    Z = data[
        [
            "Wedge gauge",
            "Interbelt gauge",
            "Peel force",
            "Wedge gauge×peel force",
        ]
    ].values
    expected = {
        "gamma_aft": -4.5104,
        "logistic_po": -5.9190,
        "weibull_po": -5.5831,
    }
    fitters = {
        "gamma_aft": AFT(Gamma),
        "logistic_po": LogisticPO,
        "weibull_po": PO(Weibull),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for key, fitter in fitters.items():
            model = fitter.fit(x=x, Z=Z, c=c)
            assert model.neg_ll() == pytest.approx(expected[key], abs=1e-3)


# -- 2. Stratified Cox time-varying prediction uses the stratum's baseline --


def _stratified_cox() -> SemiParametricRegressionModel:
    st = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    xs = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 4.5, 4.0])
    zs = np.array([[0.0], [1.0], [0.5], [0.0], [1.0], [0.2], [0.7], [0.3]])
    return CoxPH.fit(x=xs, Z=zs, strata=st)


def test_stratified_predict_tvc_requires_and_uses_stratum():
    ms = _stratified_cox()
    with pytest.raises(ValueError, match="stratum"):
        ms.predict_tvc([0.0], [10.0], [[0.0]], times=[3.0])
    for s in (0, 1):
        _, sf, _ = ms.predict_tvc(
            [0.0], [10.0], [[0.0]], times=[3.0], stratum=s
        )
        np.testing.assert_allclose(sf, ms.sf([3.0], [0.0], stratum=s))
    assert not np.isclose(
        ms.sf([3.0], [0.0], stratum=0), ms.sf([3.0], [0.0], stratum=1)
    )


# -- 3. Lin-Ying predictions do not depend on covariate centring -----------


def test_lin_ying_hf_is_invariant_to_covariate_centring():
    rng = np.random.default_rng(21)
    Z = rng.uniform(0, 2, size=(200, 1))
    T = rng.exponential(1 / (0.1 + 0.05 * Z[:, 0]))
    C = rng.uniform(2, 20, 200)
    x, c = np.minimum(T, C), (T > C).astype(int)
    m1 = AdditiveHazards.fit(x=x, Z=Z, c=c)
    m2 = AdditiveHazards.fit(x=x, Z=Z + 3, c=c)
    t = np.array([0.01, 1.0, 3.0, 5.5, 40.0])
    np.testing.assert_allclose(m1.Hf(t, [0.5]), m2.Hf(t, [3.5]), rtol=1e-10)
    # The drift is stored, so a restored model predicts the same.
    restored = AdditiveHazardsModel.from_dict(
        json.loads(json.dumps(m1.to_dict()))
    )
    np.testing.assert_allclose(restored.Hf(t, [0.5]), m1.Hf(t, [0.5]))


# -- 4. Counts are frequency weights ---------------------------------------


def _counted_data() -> tuple:
    rng = np.random.default_rng(7)
    x = np.round(rng.exponential(5, 50)) + 1
    Z = rng.normal(size=(50, 1))
    n = rng.integers(1, 4, 50)
    c = (rng.uniform(size=50) < 0.2).astype(int)
    return x, Z, n, c


def test_cox_robust_se_and_rank_test_equal_expanded_data():
    x, Z, n, c = _counted_data()
    a = CoxPH.fit(x=x, Z=Z, c=c, n=n)
    b = CoxPH.fit(
        x=np.repeat(x, n), Z=np.repeat(Z, n, axis=0), c=np.repeat(c, n)
    )
    np.testing.assert_allclose(
        a.robust_summary()["se"], b.robust_summary()["se"], rtol=1e-8
    )
    for transform in ("rank", "km", "identity"):
        np.testing.assert_allclose(
            a.check_ph(transform)["global"]["statistic"],
            b.check_ph(transform)["global"]["statistic"],
            rtol=1e-8,
        )


def test_buckley_james_bootstrap_equals_expanded_data():
    x, Z, n, c = _counted_data()
    a = BuckleyJames.fit(x, Z, c=c, n=n)
    b = BuckleyJames.fit(
        np.repeat(x, n), np.repeat(Z, n, axis=0), c=np.repeat(c, n)
    )
    np.testing.assert_allclose(
        a.bootstrap_ci(seed=3, n_boot=50), b.bootstrap_ci(seed=3, n_boot=50)
    )


# -- 5. Formulas with a missing value ---------------------------------------


def _formula_frame() -> pd.DataFrame:
    x, Z = _ph_data()
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "x": x,
            "age": Z[:, 0],
            "site": rng.choice(["a", "b", "c"], x.shape[0]),
            "c": (x > 12).astype(int),
        }
    )
    df.loc[3, "age"] = np.nan
    df.loc[5, "site"] = None
    return df


@pytest.mark.parametrize(
    "fitter",
    [WeibullPH, WeibullAFT, PO(Weibull), WeibullAH, CoxPH, AdditiveHazards],
)
def test_formula_fit_drops_missing_rows_from_every_column(fitter):
    df = _formula_frame()
    with pytest.warns(UserWarning, match="Dropped 2 of 200"):
        model = fitter.fit_from_df(
            df, x_col="x", c_col="c", formula="age + site"
        )
    ref = fitter.fit_from_df(
        df.drop(index=[3, 5]), x_col="x", c_col="c", formula="age + site"
    )
    np.testing.assert_allclose(model.params, ref.params, rtol=1e-6)


def test_formula_fit_buckley_james_and_frailty():
    df = _formula_frame()
    df["g"] = np.repeat(np.arange(40), 5)
    clean = df.drop(index=[3, 5])
    with pytest.warns(UserWarning, match="Dropped 2 of 200"):
        bj = BuckleyJames.fit_from_df(
            df, x_col="x", c_col="c", formula="age + site"
        )
    np.testing.assert_allclose(
        bj.beta,
        BuckleyJames.fit_from_df(
            clean, x_col="x", c_col="c", formula="age + site"
        ).beta,
    )
    with pytest.warns(UserWarning, match="Dropped 2 of 200"):
        fr = WeibullFrailty.fit_from_df(
            df, x_col="x", group_col="g", c_col="c", formula="age + site"
        )
    ref = WeibullFrailty.fit_from_df(
        clean, x_col="x", group_col="g", c_col="c", formula="age + site"
    )
    np.testing.assert_allclose(fr.beta, ref.beta, rtol=1e-6)


@pytest.mark.parametrize("fitter", [WeibullPH, CoxPH])
def test_prediction_from_frame_with_missing_value_is_nan_in_place(fitter):
    df = _formula_frame()
    with pytest.warns(UserWarning):
        model = fitter.fit_from_df(
            df, x_col="x", c_col="c", formula="age + site"
        )
    rows = df.iloc[[2, 3, 4, 5, 6]]
    sf = model.sf(np.full(5, 5.0), rows)
    assert sf.shape == (5,)
    assert np.isnan(sf[[1, 3]]).all()
    keep = [0, 2, 4]
    np.testing.assert_allclose(
        sf[keep], model.sf(np.full(3, 5.0), rows.iloc[keep])
    )


# -- 6. Restored models say they need the data -----------------------------


def test_restored_parametric_model_explains_missing_data():
    x, Z = _ph_data()
    fitted = WeibullPH.fit(x=x, Z=Z)
    restored = surpyval.from_dict(fitted.to_dict())
    assert np.isfinite(restored.aic())
    # The dict stores the criteria's sample size, so bic and aic_c work.
    assert restored.bic() == pytest.approx(fitted.bic())
    assert restored.aic_c() == pytest.approx(fitted.aic_c())
    with pytest.raises(ValueError, match="needs the data"):
        restored.plot()
    # A dict written before the sample size was stored needs the data.
    old = fitted.to_dict()
    del old["ic_n"]
    for call in (surpyval.from_dict(old).bic, surpyval.from_dict(old).aic_c):
        with pytest.raises(ValueError, match="needs the data"):
            call()


def test_restored_cox_residuals_explain_missing_data():
    x, Z = _ph_data()
    restored = SemiParametricRegressionModel.from_dict(
        CoxPH.fit(x, Z).to_dict()
    )
    with pytest.raises(ValueError, match="restored"):
        restored.compute_residuals()


# -- 7. Covariate shapes ----------------------------------------------------


def test_two_stress_accelerated_life_accepts_a_1d_row():
    rng = np.random.default_rng(0)
    T = np.repeat([300.0, 350.0, 400.0], 30)
    V = np.tile(np.repeat([1.0, 2.0, 3.0], 10), 3)
    x = 100 * rng.weibull(2.0, 90) * (T / 300) ** -2 * V**-1
    model = AcceleratedLife(Weibull, DualPower).fit(
        x, Z=np.column_stack([T, V])
    )
    row = [320.0, 2.0]
    np.testing.assert_allclose(model.sf([5.0], row), model.sf([5.0], [row]))
    np.testing.assert_allclose(model.hf([5.0], row), model.hf([5.0], [row]))
    np.testing.assert_allclose(model.cb([5.0], row), model.cb([5.0], [row]))
    draws, Z_out = model.random(3, row)
    assert draws.shape == (3,) and Z_out.shape == (3, 2)


def test_single_stress_accelerated_life_accepts_a_scalar_stress():
    stress = np.repeat([300.0, 350.0, 400.0], 30)
    rng = np.random.default_rng(0)
    x = rng.weibull(2.0, 90) * 1e-3 * np.exp(3000 / stress)
    model = AcceleratedLife(Weibull, ExponentialLifeModel).fit(x, Z=stress)
    np.testing.assert_allclose(
        model.sf([10.0, 5.0], 300.0), model.sf([10.0, 5.0], [300.0, 300.0])
    )
    np.testing.assert_allclose(
        model.cb([10.0, 5.0], 300.0), model.cb([10.0, 5.0], [300.0, 300.0])
    )


def test_cox_accepts_a_scalar_covariate():
    x, Z = _ph_data()
    model = CoxPH.fit(x, Z)
    for fn in ("sf", "hf", "Hf"):
        np.testing.assert_allclose(
            getattr(model, fn)([3.0], 0.5), getattr(model, fn)([3.0], [0.5])
        )
    np.testing.assert_allclose(model.phi(0.5), np.exp(0.5 * model.beta[0]))


# -- 8. Clear errors ---------------------------------------------------------


def test_wrong_number_of_covariate_rows_is_named():
    x, Z = _ph_data()
    c = (x > 12).astype(int)
    fits = [
        lambda: WeibullPH.fit(x=x, Z=Z[:-1]),
        lambda: AcceleratedLife(Weibull, Power).fit(x, Z=np.abs(Z[:-1]) + 1),
        lambda: CoxPH.fit(x, Z[:-1], c=c),
        lambda: CoxPH.fit(x, Z[:-1], c=c, strata=np.arange(200) % 2),
        lambda: AdditiveHazards.fit(x, Z[:-1], c=c),
        lambda: BuckleyJames.fit(x, Z[:-1], c=c),
        lambda: WeibullFrailty.fit(
            x, Z=Z[:-1], groups=np.repeat(np.arange(40), 5)
        ),
    ]
    for fit in fits:
        with pytest.raises(ValueError, match="Z has 199 row"):
            fit()


def test_bad_fixed_and_init_are_named():
    x, Z = _ph_data()
    with pytest.raises(ValueError, match="Unknown parameter.*gamma"):
        WeibullPH.fit(x=x, Z=Z, fixed={"gamma": 1.0})
    with pytest.raises(ValueError, match="Every parameter is fixed"):
        WeibullPH.fit(
            x=x, Z=Z, fixed={"alpha": 10, "beta": 1.5, "beta_0": 0.5}
        )
    with pytest.raises(ValueError, match="`init` has 2 value"):
        WeibullPH.fit(x=x, Z=Z, init=[10, 1.5])
    stress = np.repeat([1.0, 2.0], 100)
    with pytest.raises(ValueError, match="Unknown parameter"):
        AcceleratedLife(Weibull, Power).fit(x, Z=stress, fixed={"b": 1.0})
    with pytest.raises(ValueError, match="`init` has 2 value"):
        AcceleratedLife(Weibull, Power).fit(x, Z=stress, init=[1.0, 2.0])
    with pytest.raises(ValueError, match="`init` has 2 value"):
        WeibullFrailty.fit(
            x, Z=Z, groups=np.repeat(np.arange(40), 5), init=[1.0, 2.0]
        )


def test_frailty_groups_length_and_param_cb_name():
    x, Z = _ph_data()
    groups = np.repeat(np.arange(40), 5)
    with pytest.raises(ValueError, match="'groups' has 199"):
        WeibullFrailty.fit(x, Z=Z, groups=groups[:-1])
    model = WeibullFrailty.fit(x, Z=Z, groups=groups)
    with pytest.raises(ValueError, match="Unknown parameter 'gamma'"):
        model.param_cb("gamma")


# -- 9-10. Accelerated life: tiny parameters and start-up ------------------


def _arrhenius_data() -> tuple:
    stress = np.repeat([300.0, 350.0, 400.0], 40)
    rng = np.random.default_rng(0)
    x = rng.weibull(2.0, 120) * 1e-3 * np.exp(3000 / stress)
    return x, stress


def test_accelerated_life_covariance_with_tiny_parameter():
    x, stress = _arrhenius_data()
    model = AcceleratedLife(Weibull, InversePower).fit(x, Z=stress)
    assert model.params[2] < 1e-15
    se = model.standard_errors()
    assert np.all(np.isfinite(se))
    power = AcceleratedLife(Weibull, Power).fit(x, Z=stress)
    # InversePower's n is Power's -n, with the same standard error.
    assert se[3] == pytest.approx(power.standard_errors()[3], rel=1e-2)
    restored = surpyval.from_dict(model.to_dict())
    np.testing.assert_allclose(restored.standard_errors(), se)


def test_restored_model_without_covariance_says_so():
    x, Z = _ph_data()
    d = WeibullPH.fit(x=x, Z=Z).to_dict()
    del d["covariance"]
    restored = surpyval.from_dict(d)
    with pytest.raises(ValueError, match="restored from a dict"):
        restored.cb([3.0], [0.0])


def test_power_life_model_refuses_non_positive_stress(capfd):
    x, stress = _arrhenius_data()
    with pytest.raises(ValueError, match="strictly positive stresses"):
        AcceleratedLife(Weibull, Power).fit(x, Z=stress - 350)
    assert capfd.readouterr().err == ""


@pytest.mark.parametrize("dist", [Weibull, LogNormal, Exponential, Gamma])
def test_linear_life_model_finds_a_feasible_start(dist):
    x, stress = _arrhenius_data()
    model = AcceleratedLife(dist, Linear).fit(x, Z=stress)
    assert np.all(model.phi(np.array([300.0, 350.0, 400.0])) > 0)
    assert np.isfinite(model.neg_ll())


def test_accelerated_life_and_gamma_frailty_accept_ragged_x():
    x = [10, [11, 13], 12, 9, [20, 25], 30, 14, 15, 16]
    c = [0, 2, 0, 0, 2, 0, 0, 0, 0]
    model = AcceleratedLife(Weibull, Power).fit(
        x, Z=[1, 1, 1, 2, 2, 2, 3, 3, 3], c=c
    )
    assert np.all(np.isfinite(model.params))
    # The frailty likelihood has no interval term, so an interval row is
    # refused by name; the ragged form itself no longer crashes.
    with pytest.raises(ValueError, match="right-censored"):
        GammaFrailty.fit(x, groups=[0, 0, 0, 0, 0, 1, 1, 1, 1], c=c)
    exact = [10, 11, 12, 9, 20, 30, 14, 15, 16, 3]
    two_col = np.column_stack([exact, exact])
    groups = [0] * 5 + [1] * 5
    np.testing.assert_allclose(
        GammaFrailty.fit(two_col, groups=groups).dist_params,
        GammaFrailty.fit(exact, groups=groups).dist_params,
    )


# -- 11. Degenerate data ------------------------------------------------------


def test_degenerate_semiparametric_data_are_refused():
    x, Z = _ph_data()
    c = (x > 12).astype(int)
    for fit in (BuckleyJames.fit, AdditiveHazards.fit):
        with pytest.raises(ValueError, match="constant|vary"):
            fit(x, np.ones(200), c=c)
        with pytest.raises(ValueError, match="constant|vary"):
            fit([3.0], [[1.0]])
        with pytest.raises(ValueError, match="collinear"):
            fit(x, np.column_stack([Z, 2 * Z]), c=c)
    with pytest.raises(ValueError, match="at least one event"):
        AdditiveHazards.fit(x, Z, c=np.ones(200))
    with pytest.raises(ValueError, match="non-negative"):
        AdditiveHazards.fit(np.r_[-1.0, x[1:]], Z, c=c)


# -- 12-14. Warnings --------------------------------------------------------


def test_truncated_fit_inference_and_lognormal_path_are_quiet():
    x, Z = _ph_data()
    n = 100
    i = np.r_[np.arange(n), np.arange(50)]
    xl = np.r_[np.zeros(n), x[:50]]
    xr = np.r_[x[:n], x[:50] + 5]
    c = np.r_[np.ones(50), np.zeros(50), np.zeros(50)]
    Zt = np.r_[np.zeros(n), np.ones(50)]
    model = WeibullPH.fit_tvc(i, xl, xr, c, Zt)
    lognormal = LogNormalPH.fit(x=x, Z=Z)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.standard_errors()
        model.cb([3.0], [0.0])
        model.param_cb("beta_0")
        lognormal.sf_tvc([1.0, 3.0], [[0.0], [1.0]], xl=[0.0, 2.0])


def test_cox_warns_on_monotone_likelihood():
    rng = np.random.default_rng(2)
    x = np.r_[np.full(10, 1.0), np.full(10, 5.0)] + rng.uniform(0, 0.1, 20)
    Z = np.r_[np.ones(10), np.zeros(10)]
    c = np.r_[np.zeros(10), np.ones(10)]
    with pytest.warns(UserWarning, match="Monotone partial likelihood"):
        CoxPH.fit(x=x, Z=Z, c=c)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CoxPH.fit(x=x, Z=Z, c=c, tl=np.r_[np.full(10, 0.5), np.zeros(10)])
    kinds = {type(w.message) for w in caught}
    assert kinds == {UserWarning}  # no RuntimeWarnings alongside it


def test_quiet_functions_outside_the_support():
    x, Z = _ph_data()
    c = (x > 12).astype(int)
    cox = CoxPH.fit(x, Z, c=c)
    with pytest.raises(ValueError, match="positive event times"):
        CoxPH.fit(np.r_[0.0, x[1:]], Z, c=c).check_ph("log")
    bj = BuckleyJames.fit(x, Z, c=c)
    model = WeibullPH.fit(x=x, Z=Z)
    lognormal = LogNormalPH.fit(x=x, Z=Z)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_allclose(bj.sf([0.0, -1.0], [0.0]), [1.0, 1.0])
        for m in (model, lognormal):
            np.testing.assert_allclose(m.sf([-1.0, 0.0], [0.0]), [1.0, 1.0])
            np.testing.assert_allclose(m.ff([-1.0], [0.0]), [0.0])
            np.testing.assert_allclose(m.Hf([-1.0], [0.0]), [0.0])
            np.testing.assert_allclose(m.df([-1.0], [0.0]), [0.0])
        assert np.isfinite(cox.check_ph("rank")["global"]["statistic"])


# -- 15. aic_c of a time-varying fit counts subjects -------------------------


@pytest.mark.parametrize("fitter", [WeibullPH, WeibullAH])
def test_tvc_aic_c_is_invariant_to_episode_splitting(fitter):
    rng = np.random.default_rng(5)
    n = 80
    x = 10 * rng.weibull(1.5, n)
    z = rng.binomial(1, 0.5, n).astype(float)
    whole = fitter.fit_tvc(np.arange(n), np.zeros(n), x, np.zeros(n), z)
    i = np.r_[np.arange(n), np.arange(n)]
    split = fitter.fit_tvc(
        i,
        np.r_[np.zeros(n), x / 2],
        np.r_[x / 2, x],
        np.r_[np.ones(n), np.zeros(n)],
        np.r_[z, z],
    )
    assert split.aic_c() == pytest.approx(whole.aic_c(), rel=1e-6)
    assert split.n_subjects == n


# -- 16. Cox serialises to strict JSON -------------------------------------


def test_cox_to_dict_is_strict_json_and_reads_old_dicts():
    x, Z = _ph_data()
    model = CoxPH.fit(x, Z)
    text = json.dumps(model.to_dict(), allow_nan=False)
    restored = SemiParametricRegressionModel.from_dict(json.loads(text))
    np.testing.assert_allclose(
        restored.sf([3.0], [0.5]), model.sf([3.0], [0.5])
    )
    old = model.to_dict()
    old["tl"] = [-np.inf] * 200
    assert np.all(SemiParametricRegressionModel.from_dict(old).tl == -np.inf)
    tl = np.r_[np.zeros(100), np.full(100, -np.inf)]
    delayed = CoxPH.fit(x, Z, tl=tl)
    text = json.dumps(delayed.to_dict(), allow_nan=False)
    np.testing.assert_array_equal(
        SemiParametricRegressionModel.from_dict(json.loads(text)).tl, tl
    )
