"""Aalen's additive hazards model with sampling weights ``w``.

The estimate and its robust variance are checked against R's
``timereg::aalen(..., weights = w, robust = 1)`` in
``tests/reference/test_weighted_aalen.py``; this file checks the
properties every weighted fit keeps (whole-number weights are counts,
the scale of the weights does not matter, unit weights are the
unweighted estimate), the refusals, the effective sample size and
saving.
"""

import json
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import surpyval as sp
from surpyval.tests._helpers import weighted_reference_data
from surpyval.utils.weights import WeightedRefusal, effective_sample_size

QUERY = [1.5, 3.0, 6.0]


@pytest.fixture(scope="module")
def data():
    return weighted_reference_data()


def _fit(data, **kwargs):
    kwargs.setdefault("tl", data["tl"])
    return sp.AalenAdditive.fit(data["x"], data["Z"], c=data["c"], **kwargs)


@pytest.mark.parametrize("penalizer", [0.0, 0.5])
def test_whole_number_weights_are_counts(data, penalizer):
    """``w = k`` gives the point estimates of ``n = k`` bit for bit (each
    row enters as ``n w``, the same products) without a penalty; with
    one, the penalty's variance divides by the rows' ``sum n - 1`` (so
    that it scales with the weights) and the two differ, by design."""
    k = data["x"].size
    w = np.where(np.arange(k) % 2 == 0, 2, 1)
    weighted = _fit(data, w=w.astype(float), penalizer=penalizer)
    counted = _fit(data, n=w, penalizer=penalizer)
    assert weighted.weighted and not counted.weighted
    assert_array_equal(weighted.x, counted.x)
    if penalizer == 0:
        assert_array_equal(weighted.B, counted.B)
    else:
        assert not np.allclose(weighted.B, counted.B, rtol=1e-6)
    # The rows written out that many times: the same to rounding
    rows = np.r_[np.arange(k), np.flatnonzero(w == 2)]
    written = sp.AalenAdditive.fit(
        data["x"][rows],
        data["Z"][rows],
        c=data["c"][rows],
        tl=data["tl"][rows],
        penalizer=penalizer,
    )
    assert_allclose(counted.B, written.B, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("scale", [0.01, 7.3, 1e4])
@pytest.mark.parametrize("penalizer", [0.0, 0.5])
def test_the_scale_of_the_weights_changes_nothing(data, scale, penalizer):
    base = _fit(data, w=data["w"], penalizer=penalizer)
    scaled = _fit(data, w=data["w"] * scale, penalizer=penalizer)
    assert_allclose(scaled.B, base.B, rtol=1e-10, atol=1e-13)
    assert_allclose(scaled.B_var, base.B_var, rtol=1e-10, atol=1e-15)
    assert_allclose(
        scaled.cumulative_coefficients_cb(),
        base.cumulative_coefficients_cb(),
        rtol=1e-10,
        atol=1e-13,
    )
    Z = data["Z"][:3]
    assert_allclose(scaled.cb(QUERY, Z), base.cb(QUERY, Z), rtol=1e-10)
    assert scaled.effective_n == pytest.approx(base.effective_n, rel=1e-12)


def test_unit_weights_give_the_unweighted_estimate(data):
    """``w = 1`` is the unweighted ``B`` exactly; its variance is the
    robust one, not the optional variation."""
    plain = _fit(data)
    ones = _fit(data, w=np.ones(data["x"].size))
    assert_array_equal(ones.B, plain.B)
    assert ones.effective_n == data["x"].size
    assert not np.allclose(ones.B_var, plain.B_var)


def test_robust_variance_is_the_units_influences(data):
    """The robust variance is ``sum_i n_i psi_i psi_i'``: with each row's
    count 2 it is the variance of the rows written twice (each copy a
    unit of its own)."""
    k = data["x"].size
    n = np.full(k, 2)
    counted = _fit(data, n=n, w=data["w"])
    rows = np.r_[np.arange(k), np.arange(k)]
    twice = sp.AalenAdditive.fit(
        data["x"][rows],
        data["Z"][rows],
        c=data["c"][rows],
        tl=data["tl"][rows],
        w=data["w"][rows],
    )
    assert_allclose(counted.B, twice.B, rtol=1e-12, atol=1e-14)
    assert_allclose(counted._B_cov, twice._B_cov, rtol=1e-10, atol=1e-15)


def test_constant_effects_test_refuses_weights(data):
    model = _fit(data, w=data["w"])
    with pytest.raises(WeightedRefusal, match="weights") as info:
        model.constant_effects_test(n_sim=10, random_state=1)
    assert "Use the robust bands" in str(info.value)


def test_effective_sample_size_is_shown_and_warned(data):
    model = _fit(data, w=data["w"])
    ess = effective_sample_size(np.ones(data["x"].size), data["w"])
    assert model.effective_n == pytest.approx(ess)
    assert f"effective sample size {ess:.4g}; robust covariance" in repr(model)
    # A few rows carrying most of the weight
    w = np.ones(data["x"].size)
    w[:3] = 200.0
    with pytest.warns(UserWarning, match="effective sample size"):
        _fit(data, w=w)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _fit(data, w=data["w"])


def test_saved_weighted_model_is_schema_4(data):
    model = _fit(data, w=data["w"])
    saved = model.to_dict()
    assert saved["schema"] == 4 and saved["weighted"] is True
    assert "weighted" not in _fit(data).to_dict()
    back = sp.from_dict(json.loads(json.dumps(saved, allow_nan=False)))
    for again in (back, pickle.loads(pickle.dumps(model))):
        assert again.weighted
        assert again.effective_n == model.effective_n
        assert_array_equal(again.B_var, model.B_var)
        Z = data["Z"][:3]
        assert_array_equal(again.cb(QUERY, Z), model.cb(QUERY, Z))
        with pytest.raises(WeightedRefusal):
            again.constant_effects_test()


def test_fit_from_df_takes_w_col(data):
    df = pd.DataFrame(
        {
            "x": data["x"],
            "c": data["c"],
            "tl": data["tl"],
            "w": data["w"],
            "z1": data["Z"][:, 0],
            "z2": data["Z"][:, 1],
        }
    )
    from_df = sp.AalenAdditive.fit_from_df(
        df, "x", ["z1", "z2"], c_col="c", tl_col="tl", w_col="w"
    )
    arrays = _fit(data, w=data["w"])
    assert from_df.weighted
    assert_array_equal(from_df.B, arrays.B)
    assert_array_equal(from_df.B_var, arrays.B_var)
    assert from_df.parameter_names == ["z1", "z2"]
    with pytest.raises(ValueError, match="w_col"):
        sp.AalenAdditive.fit_from_df(df, "x", ["z1"], w_col="nope")


def test_surpyval_data_with_weights(data):
    sd = sp.SurpyvalData(
        data["x"], data["c"], tl=data["tl"], Z=data["Z"], w=data["w"]
    )
    model = sp.AalenAdditive.fit(sd)
    assert_array_equal(model.B, _fit(data, w=data["w"]).B)


def test_a_dropped_row_drops_its_weight(data):
    Z = data["Z"].copy()
    Z[4, 1] = np.nan
    with pytest.warns(UserWarning):
        dropped = sp.AalenAdditive.fit(
            data["x"], Z, c=data["c"], tl=data["tl"], w=data["w"]
        )
    keep = np.arange(data["x"].size) != 4
    left_out = sp.AalenAdditive.fit(
        data["x"][keep],
        data["Z"][keep],
        c=data["c"][keep],
        tl=data["tl"][keep],
        w=data["w"][keep],
    )
    assert_array_equal(dropped.B, left_out.B)
    assert_array_equal(dropped.B_var, left_out.B_var)


@pytest.mark.parametrize(
    "w, match",
    [
        ([1.0, 2.0], "2 weights for 40 rows"),
        (-1.0, "positive and finite"),
        (np.nan, "NaN"),
    ],
)
def test_bad_weights_are_refused_by_name(data, w, match):
    with pytest.raises(ValueError, match=match):
        _fit(data, w=w)
