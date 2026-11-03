"""Weighted Cox fits from a DataFrame (``fit_from_df(w_col=)``) and with
time-varying covariates (``fit_tvc(w=)``).

The time-varying reference is R 4.6.2, ``survival`` 3.8.12:
``coxph(Surv(start, stop, event) ~ z1 + z2t, weights = w, robust = TRUE,
id = id, ties = ...)`` on ``_helpers.weighted_reference_data()`` with
each unit's follow-up split at its midpoint and 1 added to ``z2`` on the
second episode (``scripts/reference/reference_r_weighted_cox_tvc.R``).
Tolerances as for the other weighted Cox references: ``1e-7`` on the
coefficients (the Newton search's), ``1e-6`` relative on the standard
errors.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.tests._helpers import weighted_reference_data
from surpyval.utils.weights import WeightedRefusal

R_TVC = {
    "efron": (
        [1.13764523712327, 1.20169196825796],
        [0.417589536263549, 0.202035733017177],
    ),
    "breslow": (
        [1.15115399614827, 1.19933941991423],
        [0.415679805483814, 0.19903432867117],
    ),
}


@pytest.fixture(scope="module")
def data():
    return weighted_reference_data()


def _split(data, change=0.0):
    """Each unit's follow-up split at its midpoint, ``change`` added to
    ``z2`` on the second episode, the unit's weight on both."""
    x, c, w, tl, Z = data["x"], data["c"], data["w"], data["tl"], data["Z"]
    mid = (tl + x) / 2
    k = x.size
    return dict(
        i=np.repeat(np.arange(k), 2),
        xl=np.ravel(np.column_stack([tl, mid])),
        xr=np.ravel(np.column_stack([mid, x])),
        c=np.ravel(np.column_stack([np.ones(k), c])).astype(int),
        Z=np.column_stack(
            [
                np.repeat(Z[:, 0], 2),
                np.ravel(np.column_stack([Z[:, 1], Z[:, 1] + change])),
            ]
        ),
        w=np.repeat(w, 2),
    )


def _df(data):
    return pd.DataFrame(
        {
            "x": data["x"],
            "c": data["c"],
            "w": data["w"],
            "tl": data["tl"],
            "z1": data["Z"][:, 0],
            "z2": data["Z"][:, 1],
        }
    )


@pytest.mark.parametrize("tie_method", ["efron", "breslow"])
def test_fit_from_df_with_w_col_is_fit_with_w(data, tie_method):
    model = sp.CoxPH.fit_from_df(
        _df(data),
        "x",
        ["z1", "z2"],
        c_col="c",
        tl_col="tl",
        w_col="w",
        tie_method=tie_method,
    )
    arrays = sp.CoxPH.fit(
        data["x"],
        data["Z"],
        c=data["c"],
        tl=data["tl"],
        w=data["w"],
        tie_method=tie_method,
    )
    assert model.weighted
    assert_allclose(model.params, arrays.params, rtol=0, atol=0)
    assert_allclose(model.covariance(), arrays.covariance(), rtol=0, atol=0)
    assert_allclose(model.H0, arrays.H0, rtol=0, atol=0)


def test_fit_from_df_drops_the_weights_of_dropped_rows(data):
    df = _df(data)
    df.loc[[3, 17], "z2"] = np.nan
    with pytest.warns(UserWarning, match="Dropped 2 of 40 rows"):
        model = sp.CoxPH.fit_from_df(df, "x", ["z1", "z2"], "c", w_col="w")
    keep = ~np.isin(np.arange(40), [3, 17])
    arrays = sp.CoxPH.fit(
        data["x"][keep], data["Z"][keep], c=data["c"][keep], w=data["w"][keep]
    )
    assert_allclose(model.params, arrays.params, rtol=0, atol=0)
    assert_allclose(model.covariance(), arrays.covariance(), rtol=0, atol=0)


def test_fit_from_df_formula_takes_w_col(data):
    df = _df(data)
    model = sp.CoxPH.fit_from_df(
        df, "x", formula="z1 + z2", c_col="c", w_col="w"
    )
    arrays = sp.CoxPH.fit(data["x"], data["Z"], c=data["c"], w=data["w"])
    assert_allclose(model.params, arrays.params, rtol=1e-12)


def test_fit_from_df_missing_weight_is_refused(data):
    df = _df(data)
    df.loc[5, "w"] = np.nan
    with pytest.raises(ValueError, match="'w' cannot contain NaN"):
        sp.CoxPH.fit_from_df(df, "x", ["z1", "z2"], "c", w_col="w")


@pytest.mark.parametrize("tie_method", ["efron", "breslow"])
def test_split_rows_give_the_plain_weighted_fit(data, tie_method):
    """With no covariate changing, splitting each unit's follow-up into
    two episodes changes nothing: the coefficients, and the robust
    standard errors, which cluster the episodes by unit."""
    plain = sp.CoxPH.fit(
        data["x"],
        data["Z"],
        c=data["c"],
        tl=data["tl"],
        w=data["w"],
        tie_method=tie_method,
    )
    tvc = sp.CoxPH.fit_tvc(**_split(data), tie_method=tie_method)
    assert tvc.weighted and tvc.is_tvc
    assert_allclose(tvc.params, plain.params, rtol=1e-12, atol=1e-14)
    assert_allclose(tvc.standard_errors(), plain.standard_errors(), rtol=1e-10)
    assert_allclose(tvc.robust_covariance(), plain.covariance(), rtol=1e-10)
    # The episodes as independent rows would be a different (wrong) answer
    rows = tvc.robust_covariance(cluster=np.arange(80))
    assert not np.allclose(rows, plain.covariance(), rtol=1e-3)


@pytest.mark.parametrize("tie_method", ["efron", "breslow"])
def test_tvc_matches_r(data, tie_method):
    model = sp.CoxPH.fit_tvc(**_split(data, change=1.0), tie_method=tie_method)
    coef, se = R_TVC[tie_method]
    assert_allclose(model.params, coef, rtol=0, atol=1e-7)
    assert_allclose(model.standard_errors(), se, rtol=1e-6)


def test_tvc_row_order_does_not_matter(data):
    arrays = _split(data, change=1.0)
    base = sp.CoxPH.fit_tvc(**arrays)
    order = np.random.default_rng(5).permutation(80)
    shuffled = {k: np.asarray(v)[order] for k, v in arrays.items()}
    again = sp.CoxPH.fit_tvc(**shuffled)
    assert_allclose(again.params, base.params, rtol=1e-12)
    assert_allclose(again.covariance(), base.covariance(), rtol=1e-10)


def test_tvc_from_df_takes_w_col(data):
    arrays = _split(data, change=1.0)
    df = pd.DataFrame(
        {
            "id": arrays["i"],
            "start": arrays["xl"],
            "stop": arrays["xr"],
            "c": arrays["c"],
            "z1": arrays["Z"][:, 0],
            "z2": arrays["Z"][:, 1],
            "w": arrays["w"],
        }
    )
    model = sp.CoxPH.fit_tvc_from_df(
        df, "id", "start", "stop", "c", ["z1", "z2"], w_col="w"
    )
    assert_allclose(model.params, sp.CoxPH.fit_tvc(**arrays).params)


def test_tvc_weights_with_strata_refused(data):
    arrays = _split(data)
    with pytest.raises(WeightedRefusal, match="stratified"):
        sp.CoxPH.fit_tvc(**arrays, strata=arrays["i"] % 2)
