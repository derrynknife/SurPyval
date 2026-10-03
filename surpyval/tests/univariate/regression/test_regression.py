"""
This code was created for and sponsored by Cartiga (www.cartiga.com).
Cartiga makes no representations or warranties in connection with the code
and waives any and all liability in connection therewith. Your use of the
code constitutes acceptance of these terms.

Copyright 2022 Cartiga LLC
"""

import warnings
from unittest import mock

import numpy as np
import pytest
from scipy.optimize import minimize

import surpyval as sp
from surpyval import (
    AFT,
    PO,
    AdditiveHazards,
    BuckleyJames,
    Gamma,
    LogisticPO,
    LogNormalPH,
    Weibull,
    WeibullFrailty,
    WeibullPH,
)
from surpyval.datasets import load_lung, load_rossi_static, load_tires_data
from surpyval.tests._helpers import weibull_ph_data
from surpyval.univariate.regression import (
    AcceleratedLife,
    CoxPH,
    Power,
    _fit_skeleton,
)
from surpyval.univariate.regression._fit_skeleton import finite_start
from surpyval.univariate.regression.accelerated_failure_time import aft_fitter
from surpyval.univariate.regression.proportional_odds import (
    proportional_odds_fitter,
)


def test_coxph_against_ll_rossi_static():
    ll_answer = np.array(
        [
            -0.37942216,
            -0.05743772,
            0.31389978,
            -0.14979572,
            -0.43370385,
            -0.08487107,
            0.09149708,
        ]
    )

    rossi = load_rossi_static().assign(c=lambda d: 1 - d["arrest"])
    model = CoxPH.fit_from_df(
        rossi,
        x_col="week",
        c_col="c",
        Z_cols=["fin", "age", "race", "wexp", "mar", "paro", "prio"],
        tie_method="efron",
    )

    assert np.allclose(model.beta, ll_answer)


# Examples taken from:
# http://www.sthda.com/english/wiki/cox-proportional-hazards-model


def test_coxph_against_r_lung_1():
    r_answer = np.array([-0.5310235])

    lung = load_lung()
    x = lung["time"].values
    c = 1 - lung["status"].values
    Z = lung[["sex"]].values

    model = CoxPH.fit(x=x, Z=Z, c=c, tie_method="efron")

    assert np.allclose(model.beta, r_answer)


def test_coxph_against_r_lung_2():
    r_answer = np.array([0.01106676, -0.55261240, 0.46372848])

    lung = load_lung().assign(c=lambda d: 1 - d["status"])
    model = CoxPH.fit_from_df(
        lung,
        x_col="time",
        c_col="c",
        Z_cols=["age", "sex", "ph.ecog"],
        tie_method="efron",
    )

    assert np.allclose(model.beta, r_answer)


def test_breslow_betas_rossi():
    # Hardcoded breslow answer for the Rossi dataset.
    # lifelines does not support breslow for this dataset, so the expected
    # values were generated from surpyval and kept as a regression guard.
    expected = np.array(
        [
            -0.37902189,
            -0.05724593,
            0.31412977,
            -0.15111460,
            -0.43278257,
            -0.08498284,
            0.09111154,
        ]
    )

    rossi = load_rossi_static().assign(c=lambda d: 1 - d["arrest"])
    model = CoxPH.fit_from_df(
        rossi,
        x_col="week",
        c_col="c",
        Z_cols=["fin", "age", "race", "wexp", "mar", "paro", "prio"],
        tie_method="breslow",
    )

    assert np.allclose(model.beta, expected)


def test_breslow_p_values_rossi():
    # Breslow method returns a p_value per covariate; Efron returns None.
    rossi = load_rossi_static().assign(c=lambda d: 1 - d["arrest"])
    model = CoxPH.fit_from_df(
        rossi,
        x_col="week",
        c_col="c",
        Z_cols=["fin", "age", "race", "wexp", "mar", "paro", "prio"],
        tie_method="breslow",
    )

    assert model.p_values is not None
    assert model.p_values.shape == (7,)
    assert np.all((model.p_values >= 0) & (model.p_values <= 1))


def test_formula_interface_matches_Z_cols():
    # fit_from_df with formula= must give the same betas as Z_cols=.
    # Formulaic preserves the order covariates appear in the formula, so
    # the betas can be compared directly.
    rossi = load_rossi_static().assign(c=lambda d: 1 - d["arrest"])
    Z_cols = ["fin", "age", "race", "wexp", "mar", "paro", "prio"]

    model_z = CoxPH.fit_from_df(
        rossi, x_col="week", c_col="c", Z_cols=Z_cols, tie_method="efron"
    )
    model_f = CoxPH.fit_from_df(
        rossi,
        x_col="week",
        c_col="c",
        formula="fin + age + race + wexp + mar + paro + prio",
        tie_method="efron",
    )

    assert np.allclose(model_z.beta, model_f.beta)


def test_parametric_ph_aic_bic():
    # ParametricRegressionModel.aic/bic crashed: the PH fitter never set
    # model.k, and bic/aic_c indexed SurpyvalData like a dict.
    from surpyval import Weibull, WeibullPH

    np.random.seed(1)
    n = 100
    Z = np.random.binomial(1, 0.5, n).reshape(-1, 1)
    x = Weibull.random(n, 10, 2) * np.exp(-0.5 * Z[:, 0])
    model = WeibullPH.fit(x=x, Z=Z)

    assert model.k == 3
    assert np.isfinite(model.aic())
    assert np.isfinite(model.aic_c())
    assert np.isfinite(model.bic())
    assert model.aic() < model.aic_c()


# ---------------------------------------------------------------------------
# AFT and PO fits take the gradient ladder first (#499): with a
# differentiable objective ``optimise_nm_tnc`` tries
# ``optimise_ph`` first and keeps that answer when it is a
# verified optimum.
# ---------------------------------------------------------------------------


FITTERS = [
    (sp.WeibullAFT, aft_fitter),
    (sp.LogNormalAFT, aft_fitter),
    (sp.WeibullPO, proportional_odds_fitter),
]


def _data(n=400, seed=499):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 3))
    x = rng.weibull(1.5, n) * 50 * np.exp(Z @ np.array([0.3, -0.2, 0.1]))
    c = (rng.random(n) < 0.3).astype(int)
    return x, Z, c


def _capture_objective(fitter, module):
    """The objective and start the fitter hands to ``optimise_nm_tnc``."""
    captured = []
    real = _fit_skeleton.optimise_nm_tnc

    def spy(fun, init_t, quiet=False, **kwargs):
        captured.append((fun, np.array(init_t, dtype=float)))
        return real(fun, init_t, quiet=quiet, **kwargs)

    with mock.patch.object(module, "optimise_nm_tnc", spy):
        fitter.fit(*_data())
    assert captured, "the fitter did not call optimise_nm_tnc"
    return captured[0]


def _legacy_ladder(fun, init_t):
    res = minimize(
        fun, init_t, method="Nelder-Mead", options={"maxiter": 1000}
    )
    res2 = minimize(fun, res.x, method="TNC")
    return res2 if res2.success else res


@pytest.mark.parametrize("fitter,module", FITTERS)
def test_reaches_at_least_the_old_ladders_optimum(fitter, module):
    fun, init_t = _capture_objective(fitter, module)
    new = _fit_skeleton.optimise_nm_tnc(fun, init_t, quiet=True)
    old = _legacy_ladder(fun, init_t)
    assert new.fun <= old.fun + 1e-6 * max(1.0, abs(old.fun))


@pytest.mark.parametrize("fitter,module", FITTERS)
def test_no_nelder_mead_when_the_gradient_ladder_converges(fitter, module):
    fun, init_t = _capture_objective(fitter, module)
    methods = []
    real_minimize = _fit_skeleton.minimize

    def recording(*args, **kwargs):
        methods.append(kwargs.get("method"))
        return real_minimize(*args, **kwargs)

    with mock.patch.object(_fit_skeleton, "minimize", recording):
        res = _fit_skeleton.optimise_nm_tnc(fun, init_t, quiet=True)
    assert np.isfinite(res.fun)
    assert not getattr(res, "stopped_short", False)
    assert "Nelder-Mead" not in methods, methods


# ---------------------------------------------------------------------------
# ``random`` dispatches to the fitter (#261).
# ---------------------------------------------------------------------------


def test_regression_random_dispatches_to_fitter():
    np.random.seed(3)
    x = Weibull.random(150, 10, 3)
    Z = np.random.normal(size=(150, 1))
    m = WeibullPH.fit(x, Z=Z)
    out = m.random(20, Z[:1])
    # The PH fitter's sampler returns (x, Z) arrays.
    assert np.all(np.asarray(out[0]) > 0)


# ---------------------------------------------------------------------------
# A non-finite start falls back to the default start.
# ---------------------------------------------------------------------------


def _tires():
    tires = load_tires_data()
    x = tires["Survival"].values
    c = tires["Censoring"].values
    Z = tires[
        [
            "Wedge gauge",
            "Interbelt gauge",
            "Peel force",
            "Wedge gauge×peel force",
        ]
    ].values
    return x, Z, c


def test_po_non_finite_init_falls_back_to_default_start():
    x, Z, c = _tires()
    ph = WeibullPH.fit(x=x, Z=Z, c=c)
    init = np.concatenate([ph.params[:2], -ph.params[2:]])
    with pytest.warns(UserWarning, match="not finite at the supplied") as w:
        po = PO(Weibull).fit(x=x, Z=Z, c=c, init=init)
    # The warning points at the caller, through the shared fit function.
    assert [r.filename for r in w] == [__file__]
    default = PO(Weibull).fit(x=x, Z=Z, c=c)
    # Previously the init came back unchanged, with neg_ll() == inf.
    assert not np.allclose(po.params, init)
    assert np.isfinite(po.neg_ll())
    assert po.neg_ll() == pytest.approx(default.neg_ll(), abs=1e-6)


def test_finite_start_raises_when_no_finite_start():
    def never_finite(p):
        return np.inf

    with pytest.raises(ValueError, match="not finite"):
        finite_start(never_finite, np.zeros(2), lambda: np.ones(2))
    with pytest.raises(ValueError, match="not finite"):
        finite_start(never_finite, np.zeros(2), None)


# ---------------------------------------------------------------------------
# The fit ladders warn rather than return a failed start, and
# are polished to the maximum; argument errors are named; the
# functions are quiet outside the support.
# ---------------------------------------------------------------------------


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

    with pytest.warns(UserWarning, match="did not reach a verified maximum"):
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


def test_wrong_number_of_covariate_rows_is_named():
    x, Z = weibull_ph_data()
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
    x, Z = weibull_ph_data()
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


def test_quiet_functions_outside_the_support():
    x, Z = weibull_ph_data()
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
        assert np.isfinite(cox.check_ph("rank").loc["GLOBAL", "statistic"])
