"""MultivariateSurpyvalData with any number of axes of 2 or more, axis
names, serialisation, and the fitters that say how many axes they take."""

import json

import numpy as np
import pytest

import surpyval as sp
from surpyval.multivariate import Clayton, Gaussian, MultivariateSurpyvalData

ROWS = np.array(
    [
        [100.0, 3.0, 40.0],
        [250.0, 9.0, 80.0],
        [175.0, 4.0, 61.0],
        [320.0, 12.0, 90.0],
    ]
)
NAMES = ("hours", "landings", "cycles")


def _three():
    t = np.empty((4, 3, 2))
    t[..., 0] = [[10.0, 0.0, -np.inf]] * 4
    t[..., 1] = np.inf
    c = np.array([[0, 1, 0], [2, 0, 1], [0, 0, 0], [-1, 1, 1]])
    xl = ROWS.copy()
    xr = ROWS.copy()
    xl[1, 0], xr[1, 0] = 200.0, 260.0
    return MultivariateSurpyvalData(
        ROWS, c=c, n=[1, 2, 1, 3], t=t, xl=xl, xr=xr, names=NAMES
    )


def test_three_axes_both_layouts_and_shared_row():
    by_array = MultivariateSurpyvalData(ROWS, c=[0, 1, 0])
    by_cols = MultivariateSurpyvalData(list(ROWS.T), c=[0, 1, 0])
    assert (by_array.N, by_array.D) == (4, 3)
    np.testing.assert_array_equal(by_array.x, by_cols.x)
    np.testing.assert_array_equal(by_array.c, np.tile([0, 1, 0], (4, 1)))
    assert by_array.t.shape == (4, 3, 2)
    assert by_array.xl.shape == by_array.xr.shape == (4, 3)
    assert by_array.n.shape == (4,)


def test_dimension_any_axis_by_position_or_name():
    data = _three()
    for d, name in enumerate(NAMES):
        by_pos = data.dimension(d)
        by_name = data.dimension(name)
        for a, b in zip(by_pos, by_name):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(by_pos[0], ROWS[:, d])
    np.testing.assert_array_equal(data.dimension(-1)[0], ROWS[:, 2])
    with pytest.raises(ValueError, match="3 axes"):
        data.dimension(3)
    with pytest.raises(ValueError, match="No axis is named 'starts'"):
        data.dimension("starts")


def test_one_axis_is_refused_pointing_to_surpyval_data():
    with pytest.raises(ValueError, match="SurpyvalData"):
        MultivariateSurpyvalData([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="SurpyvalData"):
        MultivariateSurpyvalData(np.array([1.0, 2.0, 3.0]))


def test_names_are_checked():
    with pytest.raises(ValueError, match="one name per axis"):
        MultivariateSurpyvalData(ROWS, names=("hours", "landings"))
    with pytest.raises(ValueError, match="distinct"):
        MultivariateSurpyvalData(ROWS, names=("a", "a", "b"))


def test_messages_name_the_axis():
    x = ROWS.copy()
    x[2, 1] = np.nan
    with pytest.raises(ValueError, match=r"Series 1 \(landings\): .*NaN"):
        MultivariateSurpyvalData(x, names=NAMES)
    # without names, the message is unchanged
    with pytest.raises(ValueError, match=r"Series 1: .*NaN"):
        MultivariateSurpyvalData(x)
    with pytest.raises(ValueError, match=r"Series 2 \(cycles\) has \[5\]"):
        MultivariateSurpyvalData(ROWS, c=[0, 0, 5], names=NAMES)


@pytest.mark.parametrize("via", ["dict", "json", "file", "package"])
def test_round_trip_keeps_every_array_and_the_names(via, tmp_path):
    data = _three()
    if via == "dict":
        again = MultivariateSurpyvalData.from_dict(data.to_dict())
    elif via == "json":
        text = data.to_json()
        json.loads(text)  # strict JSON: no Infinity literals
        assert "Infinity" not in text
        again = MultivariateSurpyvalData.from_json(text)
    elif via == "file":
        path = tmp_path / "data.json"
        assert data.to_json(path) is None
        again = MultivariateSurpyvalData.from_json(path)
    else:
        again = sp.from_json(data.to_json())
    assert isinstance(again, MultivariateSurpyvalData)
    assert again.names == NAMES
    for key in ("x", "c", "n", "t", "xl", "xr"):
        np.testing.assert_array_equal(getattr(again, key), getattr(data, key))


def test_round_trip_without_names():
    data = MultivariateSurpyvalData(ROWS[:, :2])
    again = sp.from_dict(data.to_dict())
    assert again.names is None
    assert "names" not in data.to_dict()


def test_from_dict_refuses_another_class():
    with pytest.raises(ValueError, match="written by SurpyvalData"):
        MultivariateSurpyvalData.from_dict({"class": "SurpyvalData"})


def test_copulas_take_exactly_two_axes():
    family = Clayton
    assert family.n_axes == 2
    with pytest.raises(ValueError, match=r"exactly 2 axes .*has 3 \(hours"):
        family.fit(_three(), margins=[sp.Weibull] * 3)
    with pytest.raises(ValueError, match="exactly 2 axes .*has 3"):
        family.fit(ROWS, margins=[sp.Weibull] * 3)


def test_the_elliptical_copulas_take_three_axes():
    # (the Gaussian and Student-t copulas of more than two series; their
    # own tests are in test_elliptical_nd.py)
    # Truncation of three or more series is taken since #1387 (it used to
    # be refused); the fit's own checks are in test_elliptical_nd.py.
    assert Gaussian.n_axes == 2
    # Four rows, two of them perfectly concordant in series 0 and 2: the fit
    # runs, and says the correlation has no finite maximum.
    with pytest.warns(UserWarning, match="No finite maximum"):
        model = Gaussian.fit(_three(), margins=[sp.Weibull] * 3)
    assert len(model.margins) == 3


def test_copula_fit_of_the_data_object_equals_the_arrays():
    margins = [
        sp.Weibull.from_params([10, 2]),
        sp.Weibull.from_params([20, 3]),
    ]
    X = Clayton.from_params([2.0], margins).random(200, random_state=0)
    c = (X > [14.0, 25.0]).astype(int)
    X = np.minimum(X, [14.0, 25.0])
    arrays = Clayton.fit(X, c=c, margins=[sp.Weibull, sp.Weibull])
    data = MultivariateSurpyvalData(X, c=c, names=("pump", "motor"))
    by_data = Clayton.fit(data, margins=[sp.Weibull, sp.Weibull])
    np.testing.assert_array_equal(by_data.params, arrays.params)
    with pytest.raises(ValueError, match="leave out c"):
        Clayton.fit(data, c=c, margins=[sp.Weibull, sp.Weibull])
