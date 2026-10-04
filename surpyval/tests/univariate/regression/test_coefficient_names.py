"""Regression coefficients are named by their covariate (#614).

The coefficients were ``beta_0``, ``beta_1``, ..., next to the Weibull
baseline's shape ``beta`` in every summary and in ``fixed=``. A
coefficient is now named by its covariate's column where the fit has one
(a formula, ``fit_from_df`` or a DataFrame ``Z``), else ``coef_j``, in
every family; a name that clashes with another parameter's, or with an
earlier column's, gets the first free suffix ``.1``, ``.2``, ... (R's
``make.unique``). The old names still work in ``fixed=`` and
``param_cb`` until v0.24, with a ``DeprecationWarning``, and a dictionary
saved with them loads with the new ones.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval.life_models import GeneralLogLinear
from surpyval.recurrent import ProportionalIntensityHPP
from surpyval.univariate.competing_risks import FineGray
from surpyval.univariate.regression import AcceleratedLife

DEPRECATED = "coefficient names beta_0, beta_1, ... are deprecated"


def _data(n=120, seed=3):
    rng = np.random.default_rng(seed)
    Z = np.column_stack([rng.binomial(1, 0.5, n), rng.normal(0, 1, n)])
    x = 10 * rng.weibull(2.0, n) * np.exp(-0.4 * Z[:, 0] + 0.3 * Z[:, 1])
    c = (rng.uniform(size=n) < 0.2).astype(int)
    groups = np.repeat(np.arange(n // 4), 4)
    return x, Z, c, groups


def _quiet(fit):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        return fit()


FITS = {
    "WeibullPH": lambda x, Z, c, g: sp.WeibullPH.fit(x, Z, c),
    "WeibullAFT": lambda x, Z, c, g: sp.WeibullAFT.fit(x, Z, c),
    "WeibullPO": lambda x, Z, c, g: sp.WeibullPO.fit(x, Z, c),
    "WeibullAH": lambda x, Z, c, g: sp.WeibullAH.fit(x, Z, c),
    "WeibullFrailty": lambda x, Z, c, g: sp.WeibullFrailty.fit(
        x, Z=Z, c=c, groups=g
    ),
    "CoxPH": lambda x, Z, c, g: sp.CoxPH.fit(x, Z, c),
    "CoxFrailty": lambda x, Z, c, g: sp.CoxFrailty.fit(x, Z=Z, c=c, groups=g),
    "ProportionalOdds": lambda x, Z, c, g: sp.ProportionalOdds.fit(x, Z, c),
    "AdditiveHazards": lambda x, Z, c, g: sp.AdditiveHazards.fit(x, Z, c),
    "BuckleyJames": lambda x, Z, c, g: sp.BuckleyJames.fit(x, Z, c),
}


def _coefficients(model):
    names = list(model.parameter_names)
    return [n for n in names if n not in ("alpha", "beta", "theta")]


@pytest.mark.parametrize("family", sorted(FITS))
def test_614_coefficients_are_coef_j_without_column_names(family):
    x, Z, c, g = _data()
    model = _quiet(lambda: FITS[family](x, Z, c, g))
    assert _coefficients(model) == ["coef_0", "coef_1"]
    assert "beta_0" not in repr(model)


@pytest.mark.parametrize("family", sorted(FITS))
def test_614_a_dataframe_Z_names_the_coefficients_by_column(family):
    x, Z, c, g = _data()
    frame = pd.DataFrame(Z, columns=["treated", "dose"])
    model = _quiet(lambda: FITS[family](x, frame, c, g))
    assert _coefficients(model) == ["treated", "dose"]
    assert model.feature_names == ["treated", "dose"]
    array = FITS[family](x, Z, c, g)
    np.testing.assert_array_equal(model.params, array.params)
    # It reads a DataFrame of new covariates by name
    rows = frame.iloc[:3]
    with warnings.catch_warnings():
        # (the additive model's negative hazard there is not the point)
        warnings.simplefilter("ignore", RuntimeWarning)
        np.testing.assert_allclose(
            model.sf(np.full(3, 5.0), rows[["dose", "treated"]]),
            array.sf(np.full(3, 5.0), Z[:3]),
        )


def test_614_fit_from_df_and_formula_name_the_coefficients():
    x, Z, c, _ = _data()
    df = pd.DataFrame({"x": x, "c": c, "treated": Z[:, 0], "dose": Z[:, 1]})
    by_cols = sp.WeibullAFT.fit_from_df(
        df, x_col="x", c_col="c", Z_cols=["treated", "dose"]
    )
    assert by_cols.parameter_names == ["alpha", "beta", "treated", "dose"]
    table = by_cols.summary()
    assert list(table.loc["coefficients"].index) == ["treated", "dose"]
    by_formula = sp.WeibullAFT.fit_from_df(
        df, x_col="x", c_col="c", formula="treated + dose"
    )
    assert by_formula.parameter_names == by_cols.parameter_names
    fixed = sp.WeibullAFT.fit_from_df(
        df,
        x_col="x",
        c_col="c",
        Z_cols=["treated", "dose"],
        fixed={"dose": 0.3},
    )
    assert fixed.params[3] == 0.3
    assert fixed.fixed == {"dose": 0.3}
    assert "dose = 0.3" in repr(fixed)


def test_614_a_clash_gets_the_first_free_suffix():
    # A column named like a baseline parameter, and two of one name
    x, Z, c, g = _data()
    frame = pd.DataFrame(Z, columns=["alpha", "alpha"])
    model = _quiet(lambda: sp.WeibullPH.fit(x, frame, c))
    assert model.parameter_names == ["alpha", "beta", "alpha.1", "alpha.2"]
    # The frailty variance is a parameter too
    theta = pd.DataFrame(Z, columns=["theta", "beta"])
    frailty = sp.WeibullFrailty.fit(x, Z=theta, c=c, groups=g)
    assert frailty.parameter_names == [
        "alpha",
        "beta",
        "theta.1",
        "beta.1",
        "theta",
    ]


def test_614_accelerated_life_names_its_column_coefficients():
    x, Z, c, _ = _data()
    stress = np.column_stack([np.repeat([1.0, 2.0, 3.0], 40), Z[:, 0]])
    al = AcceleratedLife(sp.Weibull, GeneralLogLinear)
    model = _quiet(lambda: al.fit(x, stress, c))
    assert model.parameter_names == ["alpha", "beta", "c", "coef_0", "coef_1"]
    frame = pd.DataFrame(stress, columns=["volts", "c"])
    named = _quiet(lambda: al.fit(x, frame, c))
    assert named.parameter_names == ["alpha", "beta", "c", "volts", "c.1"]
    np.testing.assert_array_equal(named.params, model.params)
    back = sp.from_dict(named.to_dict())
    assert back.parameter_names == named.parameter_names


def test_614_recurrent_and_competing_risks_name_their_coefficients():
    rng = np.random.default_rng(2)
    n = 60
    z = rng.binomial(1, 0.5, n).astype(float)
    x = rng.exponential(10 / np.exp(0.5 * z))
    hpp = ProportionalIntensityHPP.fit(x, z.reshape(-1, 1), i=np.arange(n))
    assert hpp.parameter_names == ["lambda", "coef_0"]
    assert "coef_0" in repr(hpp)
    named = ProportionalIntensityHPP.fit(
        x, pd.DataFrame({"z": z}), i=np.arange(n)
    )
    assert named.parameter_names == ["lambda", "z"]
    back = type(named).from_dict(named.to_dict())
    assert back.parameter_names == ["lambda", "z"]

    e = np.where(rng.uniform(size=n) < 0.5, "a", "b")
    fg = FineGray.fit(x, Z=z.reshape(-1, 1), e=e, event="a")
    assert "coef_0" in repr(fg)
    df = pd.DataFrame({"x": x, "e": e, "z": z})
    fg = FineGray.fit_from_df(df, x_col="x", e_col="e", Z_cols="z", event="a")
    assert "   z  :" in repr(fg)
    assert type(fg).from_dict(fg.to_dict()).feature_names == ["z"]


def test_614_old_names_in_fixed_and_param_cb_warn_and_work():
    x, Z, c, _ = _data()
    new = sp.WeibullPH.fit(x, Z, c, fixed={"coef_1": 0.25})
    with pytest.warns(DeprecationWarning, match=DEPRECATED) as caught:
        old = sp.WeibullPH.fit(x, Z, c, fixed={"beta_1": 0.25})
    assert caught[0].filename == __file__
    assert "'coef_1' for 'beta_1'" in str(caught[0].message)
    np.testing.assert_array_equal(old.params, new.params)
    assert old.fixed == {"coef_1": 0.25}
    with pytest.warns(DeprecationWarning, match=DEPRECATED):
        np.testing.assert_array_equal(
            new.param_cb("beta_0"), new.param_cb("coef_0")
        )
    # The Weibull's own shape is not a coefficient
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        sp.WeibullPH.fit(x, Z, c, fixed={"beta": 2.0})
    stress = np.column_stack([np.repeat([1.0, 2.0, 3.0], 40), Z[:, 0]])
    al = AcceleratedLife(sp.Weibull, GeneralLogLinear)
    with pytest.warns(DeprecationWarning, match=DEPRECATED):
        old = al.fit(x, stress, c, fixed={"beta_1": -0.4})
    assert old.fixed["coef_1"] == -0.4


def test_614_a_dict_saved_with_the_old_names_loads_with_the_new():
    x, Z, c, g = _data()
    model = sp.WeibullPH.fit(x, Z, c, fixed={"coef_1": 0.25})
    saved = model.to_dict()
    saved["phi_param_map"] = {"beta_0": 0, "beta_1": 1}
    saved["fixed"] = {"beta_1": 0.25}
    back = sp.from_dict(saved)
    assert back.parameter_names == ["alpha", "beta", "coef_0", "coef_1"]
    assert back.fixed == {"coef_1": 0.25}
    np.testing.assert_array_equal(back.sf(5.0, Z[:2]), model.sf(5.0, Z[:2]))
    # With the columns it was fitted from, by their names
    saved["feature_names"] = ["treated", "dose"]
    back = sp.from_dict(saved)
    assert back.parameter_names == ["alpha", "beta", "treated", "dose"]

    frailty = sp.WeibullFrailty.fit(x, Z=Z, c=c, groups=g).to_dict()
    frailty["param_names"] = ["alpha", "beta", "beta_0", "beta_1", "theta"]
    back = sp.from_dict(frailty)
    assert back.parameter_names == [
        "alpha",
        "beta",
        "coef_0",
        "coef_1",
        "theta",
    ]
    cox = sp.CoxFrailty.fit(x, Z=Z, c=c, groups=g).to_dict()
    cox["parameter_names"] = ["beta_0", "beta_1", "theta"]
    assert sp.from_dict(cox).parameter_names == ["coef_0", "coef_1", "theta"]

    stress = np.column_stack([np.repeat([1.0, 2.0, 3.0], 40), Z[:, 0]])
    al = AcceleratedLife(sp.Weibull, GeneralLogLinear).fit(x, stress, c)
    saved = al.to_dict()
    saved["phi_param_map"] = {"c": 0, "beta_0": 1, "beta_1": 2}
    back = sp.from_dict(saved)
    assert back.parameter_names == ["alpha", "beta", "c", "coef_0", "coef_1"]
