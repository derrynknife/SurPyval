"""Missing and non-finite covariates in the parametric, frailty and
semi-parametric regressions: the rows are dropped at fit with one warning,
formula fits drop them from every column, and prediction gives nan in
place.
"""

import numpy as np
import pandas as pd
import pytest

from surpyval import (
    PO,
    AdditiveHazards,
    BuckleyJames,
    CoxPH,
    Weibull,
    WeibullAFT,
    WeibullAH,
    WeibullFrailty,
    WeibullPH,
)
from surpyval.tests._helpers import weibull_ph_data
from surpyval.univariate.regression import AcceleratedLife, Power

DROPPED = "Dropped 1 of"


def _with_nan(Z: np.ndarray, value: float = np.nan) -> np.ndarray:
    Zn = np.array(Z, dtype=float, copy=True)
    Zn[0, 0] = value
    return Zn


@pytest.mark.parametrize(
    "fitter", [WeibullPH, WeibullAFT, PO(Weibull), WeibullAH]
)
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_parametric_fitters_drop_nonfinite_covariate_rows(fitter, value):
    x, Z = weibull_ph_data()
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
    x, Z = weibull_ph_data()
    groups = np.repeat(np.arange(40), 5)
    with pytest.warns(UserWarning, match=DROPPED):
        model = WeibullFrailty.fit(x=x, Z=_with_nan(Z), groups=groups)
    ref = WeibullFrailty.fit(x=x[1:], Z=Z[1:], groups=groups[1:])
    np.testing.assert_allclose(model.beta, ref.beta, rtol=1e-6)
    assert np.isfinite(model.neg_ll())


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_semiparametric_fitters_warn_when_dropping_rows(value):
    x, Z = weibull_ph_data()
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


def _formula_frame() -> pd.DataFrame:
    x, Z = weibull_ph_data()
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
