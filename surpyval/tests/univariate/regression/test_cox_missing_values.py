"""Missing values in Cox and Buckley-James prediction and stratified fits
(#375 items 1 and 7).

The package rule: at prediction a missing time or covariate gives nan in
its place and leaves the other rows alone; at fit a row that is an
independent observation is dropped with one warning giving the count,
and an observation with a missing stratum label is dropped the same way.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval.tests._helpers import dropped_row_messages


def _data(seed=0, n=150):
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rng.uniform(0, 1, size=n)
    g = rng.choice(["a", "b", "c"], size=n)
    eff = 0.5 * z1 - 0.8 * z2 + np.select([g == "b", g == "c"], [0.4, -0.4])
    t = 10 * rng.weibull(1.5, size=n) * np.exp(-eff / 1.5)
    cens = rng.uniform(5, 30, size=n)
    x = np.minimum(t, cens)
    c = (t > cens).astype(int)
    return x, np.column_stack([z1, z2]), c, g


# -- CoxPH: a missing time at prediction ---------------------------------


@pytest.mark.parametrize("method", ["hf", "Hf", "sf", "ff", "df"])
def test_cox_nan_time_predicts_nan(method):
    # The baseline step lookup placed a nan time after every event time and
    # returned the value at t = inf (sf 0.0102 here; hf and df 0).
    x, Z, c, _ = _data()
    model = sp.CoxPH.fit(x, Z, c)
    t = np.array([np.nan, 5.0, 1e9])
    out = getattr(model, method)(t, np.zeros((3, 2)))
    assert np.isnan(out[0])
    # The other rows are unaffected.
    clean = getattr(model, method)(t[1:], np.zeros((2, 2)))
    np.testing.assert_array_equal(out[1:], clean)


@pytest.mark.parametrize("method", ["hf", "Hf", "sf", "ff", "df"])
def test_stratified_cox_nan_time_predicts_nan(method):
    x, Z, c, g = _data()
    model = sp.CoxPH.fit(x, Z, c, strata=g)
    out = getattr(model, method)(np.array([np.nan, 5.0]), Z[:2], stratum="b")
    assert np.isnan(out[0]) and np.isfinite(out[1])


def test_cox_nan_time_before_first_event_is_nan_not_zero():
    # A nan time must not read as "before the first event" either.
    x, Z, c, _ = _data()
    model = sp.CoxPH.fit(x, Z, c)
    assert np.isnan(model.Hf(np.nan, [0.0, 0.0])).all()
    assert model.Hf(0.0, [0.0, 0.0]) == 0.0


@pytest.mark.parametrize("stratum", [np.nan, pd.NA])
def test_stratified_cox_missing_stratum_predicts_nan(stratum):
    # A missing stratum label is a missing covariate: nan, where it used to
    # raise "unknown stratum nan". None still means "no stratum given".
    x, Z, c, g = _data()
    model = sp.CoxPH.fit(x, Z, c, strata=g)
    for method in ["hf", "Hf", "sf", "ff", "df"]:
        out = getattr(model, method)([0.001, 5.0], Z[:2], stratum=stratum)
        assert out.shape == (2,) and np.isnan(out).all(), method
    with pytest.raises(ValueError, match="pass stratum"):
        model.sf(5.0, Z[:1])


# -- CoxPH: time-varying covariate prediction ----------------------------


def test_cox_tvc_nan_time_predicts_nan():
    # Hf_tvc / sf_tvc raised a bare IndexError for a nan time; predict_tvc
    # returned the survival at t = inf (0.0134).
    x, Z, c, _ = _data()
    model = sp.CoxPH.fit(x, Z, c)
    path = np.array([[0.1, 0.2], [0.3, 0.4]])
    H = model.Hf_tvc([np.nan, 5.0, 10.0], path, xl=[0, 4])
    np.testing.assert_array_equal(
        H[1:], model.Hf_tvc([5.0, 10.0], path, xl=[0, 4])
    )
    assert np.isnan(H[0])
    assert np.isnan(model.Hf_tvc([np.nan], path, xl=[0, 4])).all()
    S = model.sf_tvc([5.0, np.nan], path, xl=[0, 4])
    assert np.isfinite(S[0]) and np.isnan(S[1])
    assert np.isnan(model.sf_tvc([5.0], path, xl=[0, 4], given=np.nan)).all()
    _, sf, _ = model.predict_tvc([0, 4], [4, 20], path, times=[np.nan, 5.0])
    assert np.isnan(sf[0]) and np.isfinite(sf[1])


@pytest.mark.parametrize("bad", ["xl", "xr", "Z"])
def test_cox_predict_tvc_refuses_missing_path(bad):
    # The covariate path is one subject's history: a nan in it raises,
    # naming the input (a nan xr was ignored; a nan Z made every time nan).
    x, Z, c, _ = _data()
    model = sp.CoxPH.fit(x, Z, c)
    args = {
        "xl": [0.0, 4.0],
        "xr": [4.0, 20.0],
        "Z": np.array([[0.1, 0.2], [0.3, 0.4]]),
    }
    args[bad] = np.array(args[bad], dtype=float)
    args[bad].flat[1] = np.nan
    with pytest.raises(ValueError, match="'{}'".format(bad)):
        model.predict_tvc(args["xl"], args["xr"], args["Z"], times=[3.0])


# -- Stratified CoxPH: missing stratum labels at fit ---------------------


@pytest.mark.parametrize(
    "missing", [None, np.nan, pd.NA], ids=["None", "nan", "NA"]
)
def test_stratified_fit_drops_missing_stratum_array(missing):
    # A None / nan label raised TypeError ("'<' not supported between
    # instances of 'str' and 'NoneType'") from np.unique.
    x, Z, c, g = _data()
    strata = g.astype(object)
    strata[[0, 7]] = missing
    keep = np.ones(len(x), bool)
    keep[[0, 7]] = False
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model = sp.CoxPH.fit(x, Z, c, strata=strata)
    assert dropped_row_messages(record) == [
        "Dropped 2 of 150 rows with a missing stratum label."
    ]
    ref = sp.CoxPH.fit(x[keep], Z[keep], c[keep], strata=g[keep])
    np.testing.assert_allclose(model.params, ref.params, rtol=1e-10)
    assert list(model.strata_labels) == ["a", "b", "c"]


def test_stratified_fit_drops_missing_stratum_list():
    # np.asarray(["a", nan]) makes the nan the string "nan", a stratum of
    # its own: a list is read element by element.
    x, Z, c, g = _data()
    strata = list(g)
    strata[0], strata[7] = np.nan, None
    keep = np.ones(len(x), bool)
    keep[[0, 7]] = False
    with pytest.warns(UserWarning, match="Dropped 2 of 150 rows with a miss"):
        model = sp.CoxPH.fit(x, Z, c, strata=strata)
    ref = sp.CoxPH.fit(x[keep], Z[keep], c[keep], strata=g[keep])
    np.testing.assert_allclose(model.params, ref.params, rtol=1e-10)
    assert [str(s) for s in model.strata_labels] == ["a", "b", "c"]


def test_stratified_fit_drops_missing_numeric_stratum():
    # A nan in a float stratum array became a stratum no row matches
    # (nan != nan): "'x' is empty".
    x, Z, c, _ = _data()
    strata = np.random.default_rng(1).integers(0, 3, len(x)).astype(float)
    strata[0] = np.nan
    with pytest.warns(UserWarning, match="Dropped 1 of 150 rows"):
        model = sp.CoxPH.fit(x, Z, c, strata=strata)
    ref = sp.CoxPH.fit(x[1:], Z[1:], c[1:], strata=strata[1:])
    np.testing.assert_allclose(model.params, ref.params, rtol=1e-10)
    assert model.strata_labels == [0.0, 1.0, 2.0]


@pytest.mark.parametrize("dtype", [object, "string"])
def test_stratified_fit_from_df_drops_missing_stratum(dtype):
    x, Z, c, g = _data()
    df = pd.DataFrame({"x": x, "c": c, "z1": Z[:, 0], "z2": Z[:, 1]})
    df["g"] = pd.Series(g).astype(dtype)
    df.loc[0, "g"] = None if dtype is object else pd.NA
    df.loc[7, "g"] = np.nan if dtype is object else pd.NA
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model = sp.CoxPH.fit_from_df(
            df, "x", Z_cols=["z1", "z2"], c_col="c", strata_col="g"
        )
    assert dropped_row_messages(record) == [
        "Dropped 2 of 150 rows with a missing stratum label."
    ]
    keep = np.ones(len(x), bool)
    keep[[0, 7]] = False
    ref = sp.CoxPH.fit_from_df(
        df[keep], "x", Z_cols=["z1", "z2"], c_col="c", strata_col="g"
    )
    np.testing.assert_allclose(model.params, ref.params, rtol=1e-10)


def test_stratified_fit_every_stratum_missing_raises():
    x, Z, c, _ = _data()
    with pytest.raises(ValueError, match="Every stratum label is missing"):
        sp.CoxPH.fit(x, Z, c, strata=[None] * len(x))


def test_stratified_array_fit_warns_once_for_missing_covariates():
    # Each stratum's validation dropped its own rows: one warning per
    # stratum ("Dropped 1 of 49 ...", "Dropped 1 of 55 ...").
    x, Z, c, g = _data()
    Zn = Z.copy()
    Zn[0, 0] = np.nan
    Zn[7, 1] = np.nan
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model = sp.CoxPH.fit(x, Zn, c, strata=g)
    assert dropped_row_messages(record) == [
        "Dropped 2 of 150 rows with a missing (NaN) or infinite covariate "
        "value."
    ]
    keep = np.isfinite(Zn).all(axis=1)
    ref = sp.CoxPH.fit(x[keep], Z[keep], c[keep], strata=g[keep])
    np.testing.assert_allclose(model.params, ref.params, rtol=1e-10)


def test_stratified_fit_missing_covariate_and_stratum():
    x, Z, c, g = _data()
    Zn = Z.copy()
    Zn[[0, 5], 0] = np.nan
    strata = g.astype(object)
    strata[[0, 3]] = None
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model = sp.CoxPH.fit(x, Zn, c, strata=strata)
    # Row 0 misses both: counted once, with the covariates.
    assert sorted(dropped_row_messages(record)) == [
        "Dropped 1 of 148 rows with a missing stratum label.",
        "Dropped 2 of 150 rows with a missing (NaN) or infinite covariate "
        "value.",
    ]
    keep = np.ones(len(x), bool)
    keep[[0, 3, 5]] = False
    ref = sp.CoxPH.fit(x[keep], Z[keep], c[keep], strata=g[keep])
    np.testing.assert_allclose(model.params, ref.params, rtol=1e-10)


# -- BuckleyJames: a missing time at prediction ---------------------------


def test_buckley_james_nan_time_predicts_nan():
    # A nan time failed ``x > 0`` and was given survival 1 (Hf -0.0).
    x, Z, c, _ = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.BuckleyJames.fit(x, Z, c)
    z = [0.2, 0.1]
    t = np.array([np.nan, 5.0, 1e9])
    for method in ["sf", "ff", "Hf"]:
        out = getattr(model, method)(t, z)
        assert np.isnan(out[0]), method
        np.testing.assert_array_equal(
            out[1:], getattr(model, method)(t[1:], z)
        )
    # Time 0 and below keep survival 1.
    np.testing.assert_array_equal(model.sf([0.0, -1.0], z), [1.0, 1.0])
