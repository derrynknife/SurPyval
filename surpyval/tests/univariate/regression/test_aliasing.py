"""Aliased coefficients: a covariate column the data cannot determine
(#476).

A constant column, or one that is a linear combination of the others,
wrecked the fit or passed silently: Cox ran a coefficient off to 5.5e14
(every prediction nan) or split an effect between collinear columns, and
the parametric regressions split it wherever their optimiser stopped
(WeibullAFT's scale moved from 54.1 to 51.0). As R's ``coxph`` and
``lm`` do, such a coefficient is now aliased: the other columns are
fitted as they are without it, it is reported as nan (with its standard
error and p-value), predictions take it as 0, and one warning names it.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval.datasets import load_rossi_static
from surpyval.univariate.competing_risks import (
    CompetingRisksProportionalHazards,
    FineGray,
)

COLS = ["fin", "age", "race", "wexp", "mar", "paro", "prio"]


def _rossi_df():
    # ``arrest`` is 1 for an arrest (#479); the censoring flag is 1 - arrest.
    df = load_rossi_static()
    return df.assign(censored=1 - df["arrest"])


def _rossi():
    df = _rossi_df()
    return (
        df.week.to_numpy(),
        df[COLS].to_numpy(float),
        df.censored.to_numpy().astype(int),
    )


def _fit(fit):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = fit()
    return model, [str(w.message) for w in caught], caught


def _aliased(columns):
    return "Covariate column(s) {} of Z cannot be estimated".format(columns)


def _extra(Z, kind):
    one = np.ones(len(Z))
    return np.c_[Z, one] if kind == "constant" else np.c_[Z, 2 * Z[:, 0]]


@pytest.mark.parametrize("kind", ["constant", "collinear"])
def test_cox(kind):
    # The issue's data: a constant column ran its coefficient off to
    # 5.46e14 (sf nan) with a misleading "monotone" warning; the doubled
    # "fin" split the effect (-0.643 and 0.132) without a word.
    x, Z, c = _rossi()
    ref = sp.CoxPH.fit(x, Z, c)
    model, messages, caught = _fit(lambda: sp.CoxPH.fit(x, _extra(Z, kind), c))
    assert len(messages) == 1 and messages[0].startswith(_aliased(7))
    assert caught[0].filename == __file__
    np.testing.assert_allclose(model.beta[:7], ref.beta, rtol=1e-10)
    np.testing.assert_allclose(model.p_values[:7], ref.p_values, rtol=1e-8)
    assert np.isnan(model.beta[7]) and np.isnan(model.params[7])
    assert np.isnan(model.p_values[7])
    np.testing.assert_array_equal(model.aliased, [7])
    q = _extra(Z[:3], kind)
    np.testing.assert_allclose(
        model.sf([20, 40, 52], q), ref.sf([20, 40, 52], Z[:3]), rtol=1e-10
    )
    assert model.beta[0] == pytest.approx(-0.3794, abs=1e-4)


@pytest.mark.parametrize("fitter", ["WeibullPH", "WeibullAFT", "LogNormalAFT"])
@pytest.mark.parametrize("kind", ["constant", "collinear"])
def test_parametric(fitter, kind):
    # WeibullPH gave the constant column 0.316 and moved alpha from 54.1 to
    # 67.7 (in the issue's run), WeibullAFT gave it -0.058 and alpha 51.0;
    # the doubled column split its effect. All silently.
    x, Z, c = _rossi()
    F = getattr(sp, fitter)
    ref = F.fit(x, Z, c)
    model, messages, caught = _fit(lambda: F.fit(x, _extra(Z, kind), c))
    assert len(messages) == 1 and messages[0].startswith(_aliased(7))
    assert caught[0].filename == __file__
    np.testing.assert_allclose(model.params[:-1], ref.params, rtol=1e-5)
    assert np.isnan(model.params[-1]) and np.isnan(model.phi_params[-1])
    np.testing.assert_array_equal(model.aliased, [7])
    se = model.standard_errors()
    assert np.isnan(se[-1])
    np.testing.assert_allclose(se[:-1], ref.standard_errors(), rtol=1e-3)
    q = _extra(Z[:3], kind)
    np.testing.assert_allclose(
        model.sf([20, 40, 52], q), ref.sf([20, 40, 52], Z[:3]), rtol=1e-5
    )
    np.testing.assert_allclose(
        model.cb([20.0], q[:1]), ref.cb([20.0], Z[:1]), rtol=1e-3
    )
    # Not estimated: it costs the model no degree of freedom.
    assert model.k == ref.k


def test_additive_hazards_constant_column_is_identified():
    # h0(x) + b is not a Weibull hazard: the constant has no intercept to
    # be aliased with, and is fitted.
    x, Z, c = _rossi()
    model, messages, _ = _fit(
        lambda: sp.WeibullAH.fit(x, np.c_[Z[:, [0]], np.ones(len(x))], c)
    )
    assert not any("cannot be estimated" in m for m in messages)
    assert np.isfinite(model.params).all()


def test_the_warning_names_the_columns_of_a_data_frame():
    df = _rossi_df().assign(one=1.0)
    model, messages, caught = _fit(
        lambda: sp.CoxPH.fit_from_df(
            df, x_col="week", c_col="censored", Z_cols=["fin", "one", "age"]
        )
    )
    assert messages[0].startswith(_aliased("1 ('one')"))
    assert caught[0].filename == __file__
    model, messages, _ = _fit(
        lambda: sp.WeibullPH.fit_from_df(
            df, x_col="week", c_col="censored", Z_cols=["fin", "one", "age"]
        )
    )
    assert messages[0].startswith(_aliased("1 ('one')"))


def test_every_level_of_a_factor():
    # "0 + C(race)" codes every level; with the model's intercept (the Cox
    # baseline, the Weibull scale) their sum is aliased, as in R.
    df = _rossi_df()
    for fitter in (sp.CoxPH, sp.WeibullPH):
        model, messages, _ = _fit(
            lambda: fitter.fit_from_df(
                df, x_col="week", c_col="censored", formula="age + 0 + C(race)"
            )
        )
        ref = fitter.fit_from_df(
            df, x_col="week", c_col="censored", formula="age + C(race)"
        )
        assert len(messages) == 1
        assert "('C(race)[1]')" in messages[0]
        new = pd.DataFrame({"age": [20.0, 30.0], "race": [0, 1]})
        np.testing.assert_allclose(
            model.sf([20, 40], new), ref.sf([20, 40], new), rtol=1e-5
        )


def test_a_column_constant_within_each_stratum_and_a_column_of_zeros():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(50, 2))
    x = rng.exponential(size=50) * np.exp(-Z[:, 0])
    strata = np.repeat([0, 1], 25)
    ref = sp.CoxPH.fit(x, Z, strata=strata)
    Z4 = np.column_stack([Z[:, 0], strata, np.zeros(50), Z[:, 1]])
    model, messages, _ = _fit(lambda: sp.CoxPH.fit(x, Z4, strata=strata))
    assert len(messages) == 1 and messages[0].startswith(_aliased("1, 2"))
    np.testing.assert_allclose(model.beta[[0, 3]], ref.beta, rtol=1e-10)


def test_every_coefficient_aliased():
    # Every unit at risk fails at once: the exact partial likelihood is
    # flat (it was refused, #409).
    x = np.ones(20)
    Z = np.random.default_rng(1).normal(size=(20, 2))
    model, messages, _ = _fit(lambda: sp.CoxPH.fit(x, Z, tie_method="exact"))
    assert messages[0].startswith(_aliased("0, 1"))
    assert np.isnan(model.beta).all()
    assert np.all(model.sf([0.5, 1.0], Z[:2]) <= 1)


def test_fine_gray_and_competing_risks():
    # Fine-Gray gave a constant column 0 (se nan) and split collinear
    # columns (se nan), silently.
    rng = np.random.default_rng(0)
    n = 200
    Z = rng.normal(size=(n, 2))
    x = rng.exponential(size=n) * np.exp(-Z[:, 0])
    e = rng.choice(["a", "b"], n)
    e = np.where(rng.uniform(size=n) < 0.2, None, e)
    Z3 = np.c_[Z, Z[:, 0] - Z[:, 1]]
    ref = FineGray.fit(x, Z, e, event="a")
    model, messages, _ = _fit(lambda: FineGray.fit(x, Z3, e, event="a"))
    assert len(messages) == 1 and messages[0].startswith(_aliased(2))
    np.testing.assert_allclose(model.beta[:2], ref.beta, rtol=1e-6)
    assert np.isnan(model.beta[2]) and np.isnan(model.se[2])
    np.testing.assert_allclose(
        model.cif([0.5, 1.0], Z3[:2]), ref.cif([0.5, 1.0], Z[:2]), rtol=1e-6
    )
    for kind in ("Cox", "Fine-Gray"):
        ref = CompetingRisksProportionalHazards.fit(x, Z, e, model=kind)
        model, messages, caught = _fit(
            lambda: CompetingRisksProportionalHazards.fit(x, Z3, e, model=kind)
        )
        # One warning for both causes' fits.
        assert len(messages) == 1 and messages[0].startswith(_aliased(2))
        assert caught[0].filename == __file__
        assert np.isnan(model.betas[:, 2]).all()
        np.testing.assert_array_equal(model.aliased, [2])
        np.testing.assert_allclose(
            model.cif([0.5, 1.0], Z3[:1], "a"),
            ref.cif([0.5, 1.0], Z[:1], "a"),
            rtol=1e-6,
        )


def test_frailty():
    # The constant column took -0.49 from the scale (1.00 -> 0.63), with a
    # "No finite maximum" warning.
    rng = np.random.default_rng(0)
    n = 200
    Z = rng.normal(size=(n, 2))
    x = rng.exponential(size=n) * np.exp(-Z[:, 0])
    groups = np.arange(n) // 10
    ref = sp.WeibullFrailty.fit(x, Z=Z, groups=groups)
    model, messages, _ = _fit(
        lambda: sp.WeibullFrailty.fit(x, Z=np.c_[Z, np.ones(n)], groups=groups)
    )
    assert len(messages) == 1 and messages[0].startswith(_aliased(2))
    np.testing.assert_allclose(model.dist_params, ref.dist_params, rtol=1e-5)
    np.testing.assert_allclose(model.beta[:2], ref.beta, rtol=1e-5)
    assert np.isnan(model.beta[2])
    np.testing.assert_array_equal(model.aliased, [2])
    assert ref.aliased.size == 0
    np.testing.assert_allclose(
        model.sf([0.5], [0.1, 0.2, 1.0]), ref.sf([0.5], [0.1, 0.2]), rtol=1e-5
    )


@pytest.mark.parametrize("fitter", ["CoxPH", "WeibullPH"])
def test_round_trip(fitter):
    x, Z, c = _rossi()
    ZZ = _extra(Z, "constant")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = getattr(sp, fitter).fit(x, ZZ, c)
    back = sp.from_dict(
        json.loads(json.dumps(model.to_dict(), allow_nan=False))
    )
    assert np.isnan(back.params[-1])
    np.testing.assert_array_equal(back.aliased, [7])
    np.testing.assert_allclose(
        back.sf([20, 40], ZZ[:2]), model.sf([20, 40], ZZ[:2]), rtol=1e-12
    )


def test_cox_diagnostics_refuse_an_aliased_model():
    x, Z, c = _rossi()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.CoxPH.fit(x, _extra(Z, "constant"), c)
    with pytest.raises(ValueError, match="aliased"):
        model.compute_residuals("martingale")


def test_full_rank_fits_do_not_warn():
    x, Z, c = _rossi()
    for fitter in (sp.CoxPH, sp.WeibullPH, sp.WeibullAFT):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = fitter.fit(x, Z, c)
        assert model.aliased.size == 0


def _tvc_data(n=120):
    # Start-stop rows: a stress switches on at a random time (column 0)
    # and a fixed covariate acts throughout (column 1).
    rng = np.random.default_rng(0)
    switch = rng.uniform(0.3, 1.5, n)
    z = rng.normal(size=n)
    t_low = rng.exponential(2.0, n) * np.exp(-0.3 * z)
    t_high = switch + rng.exponential(2.0 / np.e, n) * np.exp(-0.3 * z)
    T = np.where(t_low > switch, t_high, t_low)
    one = T <= switch
    i = np.r_[np.arange(n), np.flatnonzero(~one)]
    xl = np.r_[np.zeros(n), switch[~one]]
    xr = np.r_[np.where(one, T, switch), T[~one]]
    c = np.r_[np.where(one, 0, 1), np.zeros((~one).sum(), dtype=int)]
    Z = np.c_[np.r_[np.zeros(n), np.ones((~one).sum())], np.r_[z, z[~one]]]
    return i, xl, xr, c, Z


# (A constant column is identified in a Weibull PO model: no intercept.)
_TVC = [
    (fitter, kind)
    for fitter in ("WeibullPH", "WeibullPO", "WeibullAFT", "LogNormalAFT")
    for kind in ("constant", "collinear")
    if (fitter, kind) != ("WeibullPO", "constant")
]


@pytest.mark.parametrize("fitter, kind", _TVC)
def test_fit_tvc(fitter, kind):
    # The AFT fit_tvc has its own likelihood and did not alias: a repeated
    # column split WeibullAFT's 0.338 into 1.685 and -1.346, and a constant
    # one moved alpha from 2.33 to 0.79 (to -0.195 for LogNormalAFT's mu),
    # silently. The PH / PO fit_tvc refits through ``fit`` and aliased.
    i, xl, xr, c, Z = _tvc_data()
    F = getattr(sp, fitter)
    ref = F.fit_tvc(i, xl, xr, c, Z)
    extra = np.ones(len(i)) if kind == "constant" else Z[:, 1]
    model, messages, caught = _fit(
        lambda: F.fit_tvc(i, xl, xr, c, np.c_[Z, extra])
    )
    assert len(messages) == 1 and messages[0].startswith(_aliased(2))
    assert caught[0].filename == __file__
    np.testing.assert_allclose(model.params[:-1], ref.params, rtol=1e-6)
    assert np.isnan(model.params[-1])
    np.testing.assert_array_equal(model.aliased, [2])
    assert model.k == ref.k
    q = [[0.0, 0.3, 5.0], [1.0, 0.3, -2.0]]
    np.testing.assert_allclose(
        model.sf_tvc([1.0, 2.0], q, xl=[0.0, 1.0]),
        ref.sf_tvc([1.0, 2.0], [r[:2] for r in q], xl=[0.0, 1.0]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        model.cb([1.0], q[:1]), ref.cb([1.0], [q[0][:2]]), rtol=1e-4
    )


@pytest.mark.parametrize("fitter", ["AdditiveHazards", "BuckleyJames"])
@pytest.mark.parametrize("kind", ["constant", "collinear"])
def test_lin_ying_and_buckley_james(fitter, kind):
    # Both refused such a column with a ValueError, where every other
    # regression aliases it; the baseline hazard (Lin-Ying) and the
    # profiled-out intercept (Buckley-James) absorb a constant.
    x, Z, c = _rossi()
    Z = Z[:, [0, 1, 6]]  # fin, age, prio
    F = getattr(sp, fitter)
    ref = F.fit(x, Z, c=c)
    model, messages, caught = _fit(lambda: F.fit(x, _extra(Z, kind), c=c))
    assert len(messages) == 1 and messages[0].startswith(_aliased(3))
    assert caught[0].filename == __file__
    np.testing.assert_allclose(model.beta[:3], ref.beta, rtol=1e-10)
    assert np.isnan(model.beta[3])
    np.testing.assert_array_equal(model.aliased, [3])
    q = _extra(Z[:3], kind)
    for k in range(3):
        np.testing.assert_allclose(
            model.sf([20, 40, 52], q[k]),
            ref.sf([20, 40, 52], Z[k]),
            rtol=1e-10,
        )
    if fitter == "AdditiveHazards":
        np.testing.assert_allclose(model.se[:3], ref.se, rtol=1e-10)
        assert np.isnan(model.se[3]) and np.isnan(model.p_values[3])
    else:
        ci = model.bootstrap_ci(n_boot=10, random_state=0)
        ref_ci = ref.bootstrap_ci(n_boot=10, random_state=0)
        np.testing.assert_allclose(ci[:3], ref_ci, rtol=1e-10)
        assert np.isnan(ci[3]).all()


def _stress_data():
    s = np.repeat([1.0, 2.0, 3.0], 10)
    u = (np.arange(1, 31) - 0.3) / 30.4
    life = (-np.log1p(-u)) ** 0.5
    x = np.round(30.0 * s**-1.2 * life[(np.arange(30) * 7 + 3) % 30], 3)
    return x, s


@pytest.mark.parametrize(
    "dual, single, kept, dropped",
    [
        ("DualPower", "Power", ["c", "m"], "n"),
        ("DualExponential", "ExponentialLifeModel", ["a", "c"], "b"),
    ],
)
def test_dual_stress_life_model_with_equal_stresses(
    dual, single, kept, dropped
):
    # #503: with s1 == s2, DualPower's c s1^m s2^n is Power's c s^(m + n),
    # and DualExponential's c exp(a / s1 + b / s2) is c exp((a + b) / s):
    # only the sum is determined. DualPower split Power's exponent -1.174
    # into -0.568 and -0.605, DualExponential Exponential's 1.877 into
    # 0.912 and 0.965, silently. The second stress's parameter is now
    # aliased, and the others are those of the single-stress fit.
    x, s = _stress_data()
    ref = sp.AcceleratedLife(sp.Weibull, getattr(sp, single)).fit(x, Z=s)
    F = sp.AcceleratedLife(sp.Weibull, getattr(sp, dual))
    model, messages, caught = _fit(lambda: F.fit(x, Z=np.c_[s, s]))
    assert len(messages) == 1 and messages[0].startswith(_aliased(1))
    assert dual in messages[0]
    assert caught[0].filename == __file__
    names = list(model.parameter_names)
    phi = list(model.reg_model.phi_param_map)
    assert np.isnan(model.params[names.index(dropped)])
    assert np.isnan(model.params).sum() == 1
    np.testing.assert_array_equal(model.aliased, [phi.index(dropped)])
    # The single-stress model's parameters, (a, n) for Power and (a, c)
    # for ExponentialLifeModel, are the kept ones.
    got = [model.params[names.index(k)] for k in kept]
    np.testing.assert_allclose(got, ref.phi_params, rtol=1e-4)
    np.testing.assert_allclose(model.dist_params, ref.dist_params, rtol=1e-4)
    assert model.neg_ll() == pytest.approx(ref.neg_ll(), rel=1e-8)
    assert model.k == ref.k
    np.testing.assert_allclose(
        model.sf([5.0, 10.0], [[1.5, 1.5], [2.5, 2.5]]),
        ref.sf([5.0, 10.0], [1.5, 2.5]),
        rtol=1e-4,
    )
    # A query with any value in the aliased column: its effect is 0.
    np.testing.assert_allclose(
        model.sf([5.0, 10.0], [[1.5, 7.0], [2.5, 0.3]]),
        ref.sf([5.0, 10.0], [1.5, 2.5]),
        rtol=1e-4,
    )
    se = model.standard_errors()
    assert np.isnan(se[names.index(dropped)])
    np.testing.assert_allclose(
        [se[names.index(k)] for k in kept],
        ref.standard_errors()[ref.k_dist :],
        rtol=1e-3,
    )
    restored = sp.from_dict(json.loads(json.dumps(model.to_dict())))
    np.testing.assert_array_equal(restored.aliased, model.aliased)


def test_dual_power_with_a_constant_stress_aliases_its_exponent():
    # A constant second stress: s2^n is absorbed by c, so n is aliased and
    # the fit is Power's.
    x, s = _stress_data()
    ref = sp.AcceleratedLife(sp.Weibull, sp.Power).fit(x, Z=s)
    F = sp.AcceleratedLife(sp.Weibull, sp.DualPower)
    model, messages, _ = _fit(lambda: F.fit(x, Z=np.c_[s, np.full(30, 4.0)]))
    assert len(messages) == 1 and messages[0].startswith(_aliased(1))
    assert "constant" in messages[0]
    np.testing.assert_allclose(model.phi_params[:2], ref.phi_params, rtol=1e-4)
    assert np.isnan(model.phi_params[2])


def test_power_exponential_with_equal_stresses_is_identified():
    # PowerExponential's log-life is a / s1 + n log s2: with s1 == s2 the
    # two terms are not proportional over three levels, so both effects
    # are determined and nothing is aliased.
    x, s = _stress_data()
    F = sp.AcceleratedLife(sp.Weibull, sp.PowerExponential)
    model, messages, _ = _fit(lambda: F.fit(x, Z=np.c_[s, s]))
    assert not [m for m in messages if "cannot be estimated" in m]
    assert np.isfinite(model.params).all() and model.aliased.size == 0


def test_dual_stress_fixed_parameter_is_an_offset_not_aliased():
    # With n fixed by the caller, the second stress is an offset: no
    # aliasing, and m is fitted (about Power's -1.174 less n).
    x, s = _stress_data()
    F = sp.AcceleratedLife(sp.Weibull, sp.DualPower)
    model, messages, _ = _fit(
        lambda: F.fit(x, Z=np.c_[s, s], fixed={"n": -0.2})
    )
    assert not [m for m in messages if "cannot be estimated" in m]
    assert model.params[-1] == -0.2 and np.isfinite(model.params).all()
    assert model.params[-2] == pytest.approx(-1.174 + 0.2, abs=2e-3)
