"""Missing values in competing-risks regression (#375 items 1 and 5).

At prediction a missing time gives nan in its place (it used to give the
value at t = inf). At fit, rows with a missing covariate are dropped with
the package's single warning on the array path too (they used to be
dropped silently there).
"""

import warnings

import numpy as np
import pytest

from surpyval.univariate.competing_risks import (
    CompetingRisksProportionalHazards as CR,
)
from surpyval.univariate.competing_risks import FineGray


def _data(seed=0, n=150):
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rng.uniform(0, 1, size=n)
    t = 10 * rng.weibull(1.5, size=n) * np.exp(-(0.5 * z1 - 0.8 * z2) / 1.5)
    cens = rng.uniform(5, 30, size=n)
    x = np.minimum(t, cens)
    e = np.where(t > cens, None, rng.choice([1, 2], n)).astype(object)
    return x, np.column_stack([z1, z2]), e


def _dropped_warnings(record):
    return [str(w.message) for w in record if "Dropped" in str(w.message)]


@pytest.fixture(scope="module")
def cox_model():
    x, Z, e = _data()
    return CR.fit(x, Z, e)


@pytest.fixture(scope="module")
def fg_model():
    x, Z, e = _data()
    return FineGray.fit(x, Z, e, event=1)


# -- A missing time at prediction ----------------------------------------


@pytest.mark.parametrize("event", [1, 2, None])
@pytest.mark.parametrize("method", ["hf", "Hf", "sf", "ff", "df"])
def test_cr_cox_nan_time_predicts_nan(cox_model, method, event):
    # sf(nan) was the all-cause survival at t = inf (0.0116 here).
    x, Z, _ = _data()
    t = np.array([np.nan, 5.0, 1e9])
    out = getattr(cox_model, method)(t, Z[:3], event=event)
    assert np.isnan(out[0])
    ref = [
        getattr(cox_model, method)([t[i]], Z[i], event=event)[0]
        for i in (1, 2)
    ]
    np.testing.assert_allclose(out[1:], ref, rtol=1e-12)


def test_cr_cox_cif_nan_time_predicts_nan(cox_model):
    # cif(nan) was the incidence at t = inf (0.3853, equal to cif(1e9)).
    x, Z, _ = _data()
    out = cox_model.cif(np.array([np.nan, 5.0, 1e9]), Z[0], event=1)
    assert np.isnan(out[0])
    np.testing.assert_allclose(
        out[1:], cox_model.cif(np.array([5.0, 1e9]), Z[0], event=1)
    )


def test_cr_cox_pairs_unsorted_times_with_their_rows(cox_model):
    # With one covariate row per time the step values were read at the
    # sorted times but multiplied by the rows in the given order: for
    # x = [10, 1] Hf paired H0(1) with the first row and H0(10) with the
    # second (0.14 and 0.59 on the data of the report, for 4.96 and 0.017).
    Z = np.array([[2.0, 0.5], [-2.0, 0.5]])
    t = np.array([10.0, 1.0])
    for method in ["hf", "Hf", "sf", "ff", "df"]:
        for event in [1, None]:
            paired = getattr(cox_model, method)(t, Z, event=event)
            one_by_one = [
                getattr(cox_model, method)([t[i]], Z[i], event=event)[0]
                for i in range(2)
            ]
            np.testing.assert_allclose(paired, one_by_one, rtol=1e-12)


def test_cr_cox_nan_covariate_before_first_event_is_nan(cox_model):
    # The step value before the first event was set to 0 after the
    # multiplier was applied, so a nan covariate row read as Hf = 0 there.
    early = cox_model.x[0] / 2
    out = cox_model.Hf(np.array([early, early]), [[np.nan, 0.5], [0.2, 0.1]])
    assert np.isnan(out[0]) and out[1] == 0.0


@pytest.mark.parametrize("method", ["cif", "sf"])
def test_fine_gray_nan_time_predicts_nan(fg_model, method):
    # cif(nan) was the incidence at t = inf (0.3541, equal to cif(1e9)).
    x, Z, _ = _data()
    t = np.array([np.nan, 5.0, 1e9])
    out = getattr(fg_model, method)(t, Z[:3])
    assert np.isnan(out[0])
    ref = [getattr(fg_model, method)([t[i]], Z[i])[0] for i in (1, 2)]
    np.testing.assert_allclose(out[1:], ref, rtol=1e-12)


def test_cr_fine_gray_nan_time_predicts_nan():
    x, Z, e = _data()
    model = CR.fit(x, Z, e, model="Fine-Gray")
    t = np.array([np.nan, 5.0])
    for method in ["cif", "sf", "ff", "Hf"]:
        if method == "cif":
            out = model.cif(t, Z[0], 1)
        else:
            out = getattr(model, method)(t, Z[0], event=1)
        assert np.isnan(out[0]) and np.isfinite(out[1]), method


# -- Fit: rows with a missing covariate are dropped with a warning ---------


def _nan_rows():
    x, Z, e = _data()
    Zn = Z.copy()
    Zn[0, 0] = np.nan
    Zn[7, 1] = np.nan
    keep = np.ones(len(x), bool)
    keep[[0, 7]] = False
    return x, Z, Zn, e, keep


MESSAGE = (
    "Dropped 2 of 150 rows with a missing (NaN) or infinite covariate value."
)


def test_fine_gray_fit_warns_when_dropping_missing_covariates():
    x, Z, Zn, e, keep = _nan_rows()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model = FineGray.fit(x, Zn, e, event=1)
    assert _dropped_warnings(record) == [MESSAGE]
    ref = FineGray.fit(x[keep], Z[keep], e[keep], event=1)
    np.testing.assert_allclose(model.beta, ref.beta, rtol=1e-10)


@pytest.mark.parametrize("how", ["Cox", "Fine-Gray"])
def test_cr_fit_warns_once_when_dropping_missing_covariates(how):
    x, Z, Zn, e, keep = _nan_rows()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model = CR.fit(x, Zn, e, model=how)
    assert _dropped_warnings(record) == [MESSAGE]
    ref = CR.fit(x[keep], Z[keep], e[keep], model=how)
    np.testing.assert_allclose(model.betas, ref.betas, rtol=1e-10)


def test_fine_gray_fit_drops_infinite_and_none_covariates():
    # An infinite covariate is dropped as "missing or infinite", as in every
    # other regression fitter; None in an object array is a missing value.
    x, Z, e = _data()
    Zo = Z.astype(object)
    Zo[3, 0] = None
    Zo[4, 1] = np.inf
    keep = np.ones(len(x), bool)
    keep[[3, 4]] = False
    with pytest.warns(UserWarning, match="Dropped 2 of 150 rows"):
        model = FineGray.fit(x, Zo, e, event=1)
    ref = FineGray.fit(x[keep], Z[keep], e[keep], event=1)
    np.testing.assert_allclose(model.beta, ref.beta, rtol=1e-10)


def test_fine_gray_fit_covariate_row_count_checked():
    x, Z, e = _data()
    with pytest.raises(ValueError, match="Z has 149 row"):
        FineGray.fit(x, Z[:-1], e, event=1)
