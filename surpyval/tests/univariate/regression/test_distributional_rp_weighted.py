"""Distributional and Royston-Parmar regression with sampling weights
``w``: the properties beyond the conformance suite's
(``conformance/test_weights.py``) and R's values
(``reference/test_weighted_distributional_rp.py``)."""

import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.tests._helpers import quietly, weighted_reference_data
from surpyval.utils.weights import WeightedRefusal

COVARIATES = {"alpha": [0, 1], "beta": [0]}


@pytest.fixture(scope="module")
def data():
    return weighted_reference_data()


def _distributional(data, **kwargs):
    return quietly(
        sp.WeibullDistributional.fit,
        data["x"],
        data["Z"],
        c=data["c"],
        tl=data["tl"],
        covariates=COVARIATES,
        **kwargs,
    )


def _rp(data, knots=None, **kwargs):
    fitter = sp.RoystonParmarRegression(df=2, knots=knots)
    return quietly(
        fitter.fit, data["x"], data["Z"], c=data["c"], tl=data["tl"], **kwargs
    )


def _whole_weights(k):
    return np.where(np.arange(k) % 3 == 0, 3, 1)


def test_distributional_integer_weights_equal_counts(data):
    k = _whole_weights(len(data["x"]))
    weighted = _distributional(data, w=k.astype(float))
    counted = _distributional(data, n=k)
    assert weighted.weighted and not counted.weighted
    assert_allclose(weighted.params, counted.params, rtol=1e-5, atol=1e-8)
    assert_allclose(weighted.log_likelihood, counted.log_likelihood, rtol=1e-9)


def test_rp_integer_weights_equal_counts_on_the_same_knots(data):
    knots = _rp(data).model.dist.knots
    k = _whole_weights(len(data["x"]))
    weighted = _rp(data, knots=knots, w=k.astype(float))
    counted = _rp(data, knots=knots, n=k)
    assert_allclose(weighted.params, counted.params, rtol=1e-5, atol=1e-8)
    assert_allclose(weighted.log_likelihood, counted.log_likelihood, rtol=1e-9)


def test_rp_default_knots_do_not_depend_on_the_weights(data):
    plain = _rp(data)
    weighted = _rp(data, w=data["w"])
    assert_allclose(weighted.model.dist.knots, plain.model.dist.knots)


@pytest.mark.parametrize("fit", [_distributional, _rp])
def test_unit_weights_and_scale(data, fit):
    plain = fit(data)
    ones = fit(data, w=np.ones(len(data["x"])))
    assert_allclose(ones.params, plain.params, rtol=1e-6, atol=1e-9)
    base = fit(data, w=data["w"])
    scaled = fit(data, w=data["w"] * 37.0)
    assert_allclose(scaled.params, base.params, rtol=1e-6, atol=1e-9)
    assert_allclose(
        scaled.standard_errors(), base.standard_errors(), rtol=1e-5
    )
    # The covariance is the robust one, not the inverse information
    assert not np.allclose(base.standard_errors(), plain.standard_errors())


@pytest.mark.parametrize("fit", [_distributional, _rp])
def test_effective_sample_size_and_its_warning(data, fit):
    model = fit(data, w=data["w"])
    w = data["w"]
    assert_allclose(model.effective_n, w.sum() ** 2 / (w**2).sum())
    assert "effective sample size" in repr(model)
    rows = slice(0, 15)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x, c, Z = data["x"][rows], data["c"][rows], data["Z"][rows]
        if fit is _distributional:
            sp.WeibullDistributional.fit(
                x, Z[:, :1], c=c, w=w[rows], covariates={"alpha": [0]}
            )
        else:
            sp.RoystonParmarRegression(df=1).fit(x, Z[:, :1], c=c, w=w[rows])
    said = [m for m in caught if "effective sample size" in str(m.message)]
    assert len(said) == 1


def test_distributional_refusals(data):
    model = _distributional(data, w=data["w"])
    with pytest.raises(WeightedRefusal, match="likelihood_ratio_test"):
        model.likelihood_ratio_test(parameter="beta")
    with pytest.raises(WeightedRefusal, match="likelihood-ratio"):
        model.parameter_cb("alpha", [[0.0, 0.0]], method="lr")
    with pytest.raises(WeightedRefusal, match="pseudo-likelihood"):
        model.bic()
    with pytest.raises(ValueError, match="w"):
        sp.WeibullDistributional.fit_penalised(
            data["x"], data["Z"], c=data["c"], w=data["w"]
        )


def test_distributional_round_trip_keeps_the_weights(data):
    model = _distributional(data, w=data["w"])
    d = model.to_dict()
    assert d["weighted"] is True and d["schema"] >= 3
    restored = sp.from_dict(d)
    assert restored.weighted
    assert_allclose(restored.effective_n, model.effective_n)
    assert_allclose(restored.covariance(), model.covariance(), rtol=1e-12)
    with pytest.raises(WeightedRefusal):
        restored.aic()


@pytest.mark.parametrize("which", ["distributional", "rp"])
def test_fit_from_df_w_col(data, which):
    df = pd.DataFrame(
        {
            "x": data["x"],
            "c": data["c"],
            "w": data["w"],
            "z1": data["Z"][:, 0],
            "z2": data["Z"][:, 1],
        }
    )
    # A missing covariate drops its row, and its weight with it
    df.loc[3, "z2"] = np.nan
    keep = df["z2"].notna().to_numpy()
    if which == "distributional":
        fitter = sp.WeibullDistributional
        kwargs = {"covariates": {"alpha": ["z1", "z2"], "beta": ["z1"]}}
        array_kwargs = {"covariates": COVARIATES}
    else:
        fitter = sp.RoystonParmarRegression(df=2)
        kwargs = array_kwargs = {}
    from_df = quietly(
        fitter.fit_from_df,
        df,
        x_col="x",
        Z_cols=["z1", "z2"],
        c_col="c",
        w_col="w",
        **kwargs,
    )
    arrays = quietly(
        fitter.fit,
        data["x"][keep],
        data["Z"][keep],
        c=data["c"][keep],
        w=data["w"][keep],
        **array_kwargs,
    )
    assert from_df.weighted
    assert_allclose(from_df.params, arrays.params, rtol=1e-10, atol=1e-12)
    assert_allclose(
        from_df.covariance(), arrays.covariance(), rtol=1e-8, atol=1e-12
    )


def test_rp_entry_at_zero_is_no_truncation(data):
    """A left truncation at 0 is none (S(0) = 1); it used to make the
    spline's truncation term nan at log(0), the fit unverified and a
    weighted fit's covariance nan."""
    at_zero = _rp(data, w=data["w"])
    tl = np.where(data["tl"] > 0, data["tl"], -np.inf)
    fitter = sp.RoystonParmarRegression(df=2)
    none = quietly(
        fitter.fit, data["x"], data["Z"], c=data["c"], tl=tl, w=data["w"]
    )
    assert at_zero.maximum == "verified"
    assert_allclose(at_zero.params, none.params, rtol=1e-10)
    assert np.isfinite(at_zero.standard_errors()).all()
