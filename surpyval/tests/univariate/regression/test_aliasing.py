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


def _rossi():
    df = load_rossi_static()
    return (
        df.week.to_numpy(),
        df[COLS].to_numpy(float),
        df.arrest.to_numpy().astype(int),
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
    df = load_rossi_static().assign(one=1.0)
    model, messages, caught = _fit(
        lambda: sp.CoxPH.fit_from_df(
            df, x_col="week", c_col="arrest", Z_cols=["fin", "one", "age"]
        )
    )
    assert messages[0].startswith(_aliased("1 ('one')"))
    assert caught[0].filename == __file__
    model, messages, _ = _fit(
        lambda: sp.WeibullPH.fit_from_df(
            df, x_col="week", c_col="arrest", Z_cols=["fin", "one", "age"]
        )
    )
    assert messages[0].startswith(_aliased("1 ('one')"))


def test_every_level_of_a_factor():
    # "0 + C(race)" codes every level; with the model's intercept (the Cox
    # baseline, the Weibull scale) their sum is aliased, as in R.
    df = load_rossi_static()
    for fitter in (sp.CoxPH, sp.WeibullPH):
        model, messages, _ = _fit(
            lambda: fitter.fit_from_df(
                df, x_col="week", c_col="arrest", formula="age + 0 + C(race)"
            )
        )
        ref = fitter.fit_from_df(
            df, x_col="week", c_col="arrest", formula="age + C(race)"
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
