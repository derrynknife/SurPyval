"""Sampling weights ``w`` in the data model: checks, grouping, printouts,
reshapes, serialisation and the routes into the fitters."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import surpyval as sp
from surpyval.serialisation import required_schema
from surpyval.utils.weights import (
    WeightedRefusal,
    check_weights,
    effective_sample_size,
)


def test_weights_are_checked_by_name():
    for bad, match in (
        ([1, 0, 2], "positive and finite"),
        ([1, -1, 2], "positive and finite"),
        ([1, np.inf, 2], "positive and finite"),
        ([1, np.nan, 2], "NaN"),
        ([1, 2], "2 weights for 3 rows"),
        ([[1, 2], [3, 4], [5, 6]], "one dimensional"),
        (["a", "b", "c"], "must be numbers"),
    ):
        with pytest.raises(ValueError, match=match):
            sp.SurpyvalData([1, 2, 3], w=bad)
    assert_array_equal(check_weights(2.5, 3), [2.5, 2.5, 2.5])


def test_a_decimal_count_points_to_w():
    with pytest.raises(ValueError, match="pass them as 'w'"):
        sp.Weibull.fit([1, 2, 3], n=[1.5, 1, 1])


def test_rows_group_only_with_equal_weights():
    data = sp.SurpyvalData([3, 1, 3, 3], c=[0, 0, 0, 0], w=[2, 1, 2, 5])
    assert_array_equal(data.x, [1, 3, 3])
    assert_array_equal(data.n, [1, 2, 1])
    assert_array_equal(data.w, [1, 2, 5])
    # The likelihood's counts are n * w, the weights scaled to sum to the
    # units (4 units, total weight 10)
    assert_allclose(data.likelihood_weights, np.array([1, 2, 5]) * 0.4)
    assert_allclose(data.n_o, np.array([1, 4, 5]) * 0.4)
    assert data.n_items == 4


def test_grouping_does_not_depend_on_row_order():
    x, c, w = [3, 1, 3, 3, 2], [0, 1, 0, 1, 0], [2, 1, 2, 5, 3]
    a = sp.SurpyvalData(x, c, w=w)
    order = [4, 2, 0, 3, 1]
    b = sp.SurpyvalData(
        np.take(x, order), np.take(c, order), w=np.take(w, order)
    )
    for part in ("x", "c", "n", "w"):
        assert_array_equal(getattr(a, part), getattr(b, part))


def test_unweighted_data_is_unchanged():
    data = sp.SurpyvalData([3, 1, 3], [0, 1, 0])
    assert data.w is None and not data.weighted
    assert "w" not in data.to_dict()
    assert " w" not in repr(data).splitlines()[1]
    assert "weights" not in data.summary()


def test_effective_sample_size_in_the_printouts():
    data = sp.SurpyvalData([1, 2, 3, 4], w=[1, 1, 2, 4])
    ess = effective_sample_size(data.n, data.w)
    assert_allclose(ess, 64 / 22)
    assert "effective sample size 2.909" in data.summary()
    assert repr(data).splitlines()[1].split() == ["x", "c", "n", "w"]


def test_indexing_keeps_the_weights():
    data = sp.SurpyvalData([1, 2, 3, 4], c=[0, 1, 0, 0], w=[1, 2, 3, 4])
    part = data[1:3]
    assert_array_equal(part.w, [2, 3])
    assert_allclose(part.n_r, [2 * 2 / 5])


def test_reshapes_refuse_weighted_data():
    data = sp.SurpyvalData([1, 2, 3, 4], c=[0, 1, 0, 0], w=[1, 2, 3, 4])
    for call in (
        lambda: data.censor_at(2.5),
        lambda: data.enter_at(0.5),
        lambda: data.without("truncation"),
        lambda: data.to_items(),
        lambda: data.item(0),
    ):
        with pytest.raises(ValueError, match="weighted data"):
            call()


def test_serialisation_keeps_the_weights_with_schema_3():
    data = sp.SurpyvalData([1, 2, 2, 5], c=[0, 0, 0, 1], w=[1, 2, 2, 0.5])
    d = json.loads(json.dumps(data.to_dict(), allow_nan=False))
    again = sp.SurpyvalData.from_dict(d)
    assert_array_equal(again.w, data.w)
    assert_array_equal(again.n_o, data.n_o)
    assert required_schema(d) == 3
    # (the infinite truncation bounds already make unweighted data 2)
    assert required_schema(sp.SurpyvalData([1, 2]).to_dict()) == 2


def test_frames_carry_the_weights():
    df = pd.DataFrame({"t": [1, 2, 3], "w": [0.5, 2.0, 1.5]})
    data = sp.SurpyvalData.from_dataframe(df, x_col="t", w_col="w")
    assert_array_equal(data.w, [0.5, 2.0, 1.5])
    assert data.to_dataframe().columns.tolist() == ["x", "c", "n", "w"]
    a = sp.KaplanMeier.fit_from_df(df, x_col="t", w_col="w")
    b = sp.KaplanMeier.fit([1, 2, 3], w=[0.5, 2.0, 1.5])
    assert_allclose(a.sf([1.5, 2.5]), b.sf([1.5, 2.5]))
    with pytest.raises(ValueError, match="takes no `w`"):
        sp.Bernoulli.fit_from_df(df, x_col="t", w_col="w")


def test_to_xrd_is_weighted():
    data = sp.SurpyvalData([1, 2, 2, 3], c=[0, 0, 1, 0], w=[2, 1, 3, 1])
    x, r, d = data.to_xrd()
    assert_array_equal(x, [1, 2, 3])
    assert_array_equal(r, [7, 5, 1])
    assert_array_equal(d, [2, 1, 1])


def test_fit_of_weighted_data_passes_the_weights():
    x, c, w = [1, 2, 3, 4, 5, 6], [0, 0, 1, 0, 0, 1], [1, 2, 1, 3, 1, 2]
    data = sp.SurpyvalData(x, c, w=w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fitter in (sp.KaplanMeier, sp.NelsonAalen):
            a, b = fitter.fit(data), fitter.fit(x, c, w=w)
            assert a.weighted
            assert_allclose(a.sf([2.5, 4.5]), b.sf([2.5, 4.5]))
    with pytest.raises(TypeError, match="leave out 'w'"):
        sp.KaplanMeier.fit(data, w=w)
    with pytest.raises(ValueError, match="takes no weights"):
        sp.MixtureModel(sp.Weibull, 2).fit(data)


def test_few_effective_units_warn():
    with pytest.warns(UserWarning, match="effective sample size of 2.9"):
        sp.KaplanMeier.fit([1, 2, 3, 4], w=[1, 1, 2, 4])


def test_refusals_are_value_errors():
    assert issubclass(WeightedRefusal, ValueError)
