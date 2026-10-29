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
from surpyval.tests._helpers import quietly, weighted_reference_data
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
    a = sp.Weibull.fit_from_df(df, x_col="t", w_col="w")
    b = sp.Weibull.fit([1, 2, 3], w=[0.5, 2.0, 1.5])
    assert_allclose(a.params, b.params)
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
        for fitter in (sp.KaplanMeier, sp.Weibull):
            a, b = fitter.fit(data), fitter.fit(x, c, w=w)
            assert a.weighted
            assert_allclose(a.sf([2.5, 4.5]), b.sf([2.5, 4.5]))
    with pytest.raises(TypeError, match="leave out 'w'"):
        sp.Weibull.fit(data, w=w)
    with pytest.raises(ValueError, match="takes no weights"):
        sp.MixtureModel(sp.Weibull, 2).fit(data)


def test_few_effective_units_warn():
    with pytest.warns(UserWarning, match="effective sample size of 2.9"):
        sp.KaplanMeier.fit([1, 2, 3, 4], w=[1, 1, 2, 4])


_FEW_X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
_FEW_C = np.array([0, 0, 1, 0, 0, 1])
_FEW_Z = np.array(
    [[0, 0.5], [1, -0.2], [0, 1.1], [1, 0.3], [0, -0.7], [1, 0.9]]
)
_FITTERS = {
    "KaplanMeier": lambda x, c, Z, w: sp.KaplanMeier.fit(x, c, w=w),
    "Weibull": lambda x, c, Z, w: sp.Weibull.fit(x, c, w=w),
    "WeibullPH": lambda x, c, Z, w: sp.WeibullPH.fit(x, Z, c=c, w=w),
    "CoxPH": lambda x, c, Z, w: sp.CoxPH.fit(x, Z, c=c, w=w),
}
_TWICE = (
    "WeibullPH warns twice: once from optimised_fit.py and again from "
    "_fit_skeleton.py. Reproduce: sp.WeibullPH.fit([1, 2, 3, 4, 5, 6], Z, "
    "c=[0, 0, 1, 0, 0, 1], w=[1, 1, 1, 1, 1, 40]) records two 'effective "
    "sample size of 1.26' warnings."
)
_STACKLEVEL = (
    "warn_if_few_effective's stacklevel=3 points inside the library, not "
    "at the caller: the warning's filename is {} for {}."
)
_WHERE = {
    "KaplanMeier": "nonparametric_fitter.py",
    "Weibull": "optimised_fit.py",
    "WeibullPH": "optimised_fit.py",
    "CoxPH": "cox_ph.py",
}


def _effective_warnings(name, x, c, Z, w):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _FITTERS[name](x, c, Z, np.asarray(w, dtype=float))
    return [r for r in record if "effective sample size" in str(r.message)]


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name,
            marks=(
                [pytest.mark.xfail(strict=True, reason=_TWICE)]
                if name == "WeibullPH"
                else []
            ),
        )
        for name in _FITTERS
    ],
)
def test_few_effective_units_warn_exactly_once(name):
    """audit W-G: a fit whose weights leave few effective units (here
    1.26 of 6) warns exactly once (principle 22)."""
    found = _effective_warnings(
        name, _FEW_X, _FEW_C, _FEW_Z, [1, 1, 1, 1, 1, 40]
    )
    assert len(found) == 1, [str(r.message) for r in found]
    assert "1.26" in str(found[0].message)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name,
            marks=[
                pytest.mark.xfail(
                    strict=True,
                    reason=_STACKLEVEL.format(_WHERE[name], name),
                )
            ],
        )
        for name in _FITTERS
    ],
)
def test_few_effective_units_warning_points_at_the_caller(name):
    """audit W-G: the few-effective-units warning is attributed to the
    line that called ``fit`` (its stacklevel), not to library code."""
    found = _effective_warnings(
        name, _FEW_X, _FEW_C, _FEW_Z, [1, 1, 1, 1, 1, 40]
    )
    assert found
    assert all(r.filename == __file__ for r in found), [
        r.filename for r in found
    ]


@pytest.mark.parametrize("name", list(_FITTERS))
def test_equal_weights_on_enough_units_do_not_warn(name):
    """audit W-G: equal weights on 30 rows (an effective sample size of
    30) give no few-effective-units warning. (With equal weights on the
    six rows above the effective sample size is 6, below 20, and the
    warning is right to fire.)"""
    rng = np.random.default_rng(20261030)
    x = np.round(rng.weibull(1.5, 30) * 5 + 0.1, 2)
    c = (rng.uniform(size=30) < 0.2).astype(int)
    Z = np.column_stack([rng.integers(0, 2, 30), rng.normal(size=30)])
    assert not _effective_warnings(name, x, c, Z, np.full(30, 2.5))


def test_refusals_are_value_errors():
    assert issubclass(WeightedRefusal, ValueError)


@pytest.mark.parametrize("how", ["MPP", "MOM", "MSE", "MPS"])
def test_every_method_but_mle_refuses_weights(how):
    """audit W-H: a weighted fit by probability plotting, moments, mean
    square error or product spacings refuses, naming the method and MLE."""
    data = weighted_reference_data()
    with pytest.raises(WeightedRefusal, match=f"how='{how}'") as info:
        sp.Weibull.fit(data["x"], data["c"], w=data["w"], how=how)
    assert "MLE" in str(info.value)


def test_regression_frames_carry_the_weights():
    """audit W-L: a parametric regression's ``fit_from_df(w_col=)`` gives
    the ``params`` and robust covariance of ``fit(..., w=w)``."""
    data = weighted_reference_data()
    df = pd.DataFrame(
        {
            "t": data["x"],
            "cens": data["c"],
            "z1": data["Z"][:, 0],
            "z2": data["Z"][:, 1],
            "w": data["w"],
        }
    )
    a = quietly(
        sp.WeibullPH.fit_from_df,
        df,
        x_col="t",
        c_col="cens",
        Z_cols=["z1", "z2"],
        w_col="w",
    )
    b = quietly(
        sp.WeibullPH.fit, data["x"], data["Z"], c=data["c"], w=data["w"]
    )
    assert a.weighted
    assert_allclose(a.params, b.params, rtol=1e-10)
    assert_allclose(a.covariance(), b.covariance(), rtol=1e-10)


def test_a_weighted_estimate_saved_with_its_data_keeps_its_rmst():
    """audit W-N: a weighted Kaplan-Meier saved with ``with_data=True``
    restores with the data its restricted mean's jackknife needs, and
    gives the original's ``rmst(6.0)`` (estimate, se and bounds)."""
    data = weighted_reference_data()
    model = quietly(sp.KaplanMeier.fit, data["x"], data["c"], w=data["w"])
    d = model.to_dict(with_data=True)
    restored = sp.from_dict(json.loads(json.dumps(d, allow_nan=False)))
    assert restored.weighted
    original, again = model.rmst(6.0), restored.rmst(6.0)
    for key in ("rmst", "se", "lower", "upper", "tau"):
        assert_allclose(again[key], original[key], rtol=1e-12, err_msg=key)
