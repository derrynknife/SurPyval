"""``SplinePath``: a natural cubic spline as a degradation path.

The basis is checked against its definition worked out in exact
arithmetic (``fractions.Fraction``), so the reference values do not
depend on the platform.
"""

import json
import pickle
import warnings
from fractions import Fraction

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

import surpyval
from surpyval.degradation import (
    DegradationAnalysis,
    LinearPath,
    PathModel,
    SplinePath,
    get_path_model,
)
from surpyval.degradation.path_models import path_model_key

KNOTS = [Fraction(2), Fraction(4), Fraction(7)]
LO, HI = Fraction(1, 2), Fraction(10)
PATH = SplinePath(knots=[2.0, 4.0, 7.0], boundary=(0.5, 10.0))
THETA = (2.0, 0.1, 5.0, -3.0, 4.0)


def _exact(t, k, order=0):
    """``N_k`` (``k`` from 0) at the Fraction ``t`` from its definition,
    or its derivative of ``order`` (-1: its integral from age 0), in
    exact arithmetic."""
    span = HI - LO
    xi = [Fraction(0)] + [(q - LO) / span for q in KNOTS] + [Fraction(1)]

    def power(u, start):
        """The derivative of ``order`` of ``(u - start)_+^3`` in u (or
        its antiderivative)."""
        v = max(u - start, Fraction(0))
        if order == -1:
            return v**4 / 4
        factor = {0: 1, 1: 3, 2: 6, 3: 6}[order]
        if order == 3:
            return Fraction(6) if u > start else Fraction(0)
        return factor * v ** (3 - order)

    def d(u, j):
        return (power(u, xi[j]) - power(u, xi[-1])) / (xi[-1] - xi[j])

    def at(u):
        return d(u, k) - d(u, len(xi) - 2)

    u = (t - LO) / span
    if order == -1:
        return span * (at(u) - at((0 - LO) / span))
    return at(u) / span**order


def test_is_a_path_model_with_named_parameters():
    assert isinstance(PATH, PathModel)
    assert PATH.parameter_names == ["level", "rate", "s1", "s2", "s3"]
    assert PATH.linear_in_parameters
    assert SplinePath().parameter_names == ["level", "rate"]
    assert SplinePath.natural(df=3).parameter_names == [
        "level",
        "rate",
        "s1",
        "s2",
    ]


def test_basis_is_its_definition_in_exact_arithmetic():
    ages = [Fraction(n, 4) for n in range(0, 57)]
    x = np.array([float(a) for a in ages])
    basis = PATH.basis(x)
    slopes = PATH.slope_jacobian(x, *THETA)
    areas = PATH.area_jacobian(x, *THETA)
    assert np.array_equal(basis[:, 0], np.ones_like(x))
    assert np.array_equal(basis[:, 1], x)
    for k in range(3):
        value = np.array([float(_exact(a, k)) for a in ages])
        slope = np.array([float(_exact(a, k, 1)) for a in ages])
        area = np.array([float(_exact(a, k, -1)) for a in ages])
        np.testing.assert_allclose(basis[:, 2 + k], value, rtol=1e-12, atol=0)
        np.testing.assert_allclose(slopes[:, 2 + k], slope, rtol=1e-12, atol=0)
        np.testing.assert_allclose(areas[:, 2 + k], area, rtol=1e-12, atol=0)


def test_pieces_join_smoothly_and_are_lines_outside_the_boundary():
    """Exactly: the value, slope and curvature of every basis function
    are continuous at every knot (each piece is a polynomial, so one
    side's limit is its value at the knot from a formula with the knot's
    term left out, the other's with it in: they agree when the term and
    its first two derivatives are 0 there), and the curvature is 0 at and
    beyond the boundary knots."""
    for k in range(3):
        for knot in [LO, *KNOTS, HI]:
            for order in (0, 1, 2):
                near = [
                    _exact(knot + step, k, order)
                    for step in (Fraction(-1, 10**9), Fraction(1, 10**9))
                ]
                at = _exact(knot, k, order)
                # each side is within the pieces' slope times the step
                assert abs(near[0] - at) < Fraction(1, 10**6)
                assert abs(near[1] - at) < Fraction(1, 10**6)
        for outside in (Fraction(0), LO, HI, Fraction(12), Fraction(40)):
            assert _exact(outside, k, 2) == 0
        # a line before the first knot: 0; a line after the last
        assert _exact(Fraction(1, 4), k) == 0
        assert _exact(Fraction(12), k, 1) == _exact(Fraction(40), k, 1)


def test_path_slope_and_area_agree_with_each_other():
    x = np.linspace(0.0, 14.0, 57)
    np.testing.assert_allclose(
        PATH.path(x, *THETA), PATH.basis(x) @ np.array(THETA), rtol=1e-13
    )
    h = 1e-5
    numeric = (PATH.path(x + h, *THETA) - PATH.path(x - h, *THETA)) / (2 * h)
    np.testing.assert_allclose(PATH.slope(x, *THETA), numeric, atol=1e-8)
    np.testing.assert_allclose(
        PATH.slope_jacobian(x, *THETA) @ np.array(THETA),
        PATH.slope(x, *THETA),
        rtol=1e-13,
    )
    area = [
        quad(
            lambda t: float(PATH.path(t, *THETA)),
            0.0,
            upper,
            points=[0.5, 2.0, 4.0, 7.0, 10.0],
            epsabs=1e-13,
            epsrel=1e-13,
        )[0]
        for upper in x
    ]
    np.testing.assert_allclose(PATH.area(x, *THETA), area, atol=1e-11)
    np.testing.assert_allclose(
        PATH.area_jacobian(x, *THETA) @ np.array(THETA),
        PATH.area(x, *THETA),
        rtol=1e-13,
    )


def test_area_starts_at_age_zero_when_the_boundary_is_below_it():
    path = SplinePath(knots=[1.0], boundary=(-2.0, 5.0))
    theta = (1.0, 0.2, 3.0)
    for upper in (0.0, 0.5, 3.0, 8.0):
        reference = quad(
            lambda t: float(path.path(t, *theta)),
            0.0,
            upper,
            points=[1.0, 5.0],
            epsabs=1e-13,
        )[0]
        assert float(path.area(upper, *theta)) == pytest.approx(
            reference, abs=1e-12
        )


def test_it_spans_the_natural_cubic_splines_on_its_knots():
    """The natural cubic spline through any values at the knots (scipy's)
    is in the family: fitted to that curve, the path is that curve."""
    knots = np.array([0.5, 2.0, 4.0, 7.0, 10.0])
    values = np.array([1.0, 1.2, 0.7, 2.5, 6.0])
    reference = CubicSpline(knots, values, bc_type="natural")
    x = np.linspace(0.5, 10.0, 77)
    theta = PATH.fit(x, reference(x))
    np.testing.assert_allclose(
        PATH.path(x, *theta), reference(x), rtol=0, atol=1e-11
    )
    np.testing.assert_allclose(
        PATH.slope(x, *theta), reference(x, 1), rtol=0, atol=1e-10
    )


def test_no_interior_knots_is_the_linear_path_bit_for_bit():
    line = SplinePath(boundary=(0.0, 10.0))
    x = np.array([0.0, 0.7, 1.9, 3.2, 6.6, 9.1])
    y = np.array([2.1, 2.0, 2.9, 3.1, 5.2, 6.4])
    assert np.array_equal(line.fit(x, y), LinearPath.fit(x, y))
    a, b = LinearPath.fit(x, y)
    for own, linear in (
        (line.path(x, a, b), LinearPath.path(x, a, b)),
        (line.jacobian(x, a, b), LinearPath.jacobian(x, a, b)),
        (line.slope(x, a, b), LinearPath.slope(x, a, b)),
        (line.slope_jacobian(x, a, b), LinearPath.slope_jacobian(x, a, b)),
        (line.inv_path(4.0, a, b), LinearPath.inv_path(4.0, a, b)),
    ):
        assert np.array_equal(own, linear)
    assert line.breaks == ()
    # no boundary is needed for a line
    assert np.array_equal(SplinePath().path(x, a, b), LinearPath.path(x, a, b))


def test_inv_path_is_the_first_crossing():
    # rises to about 3.5 near 4.4, falls back, then climbs for good
    theta = (2.0, 0.1, 5.0, -3.0, 4.0)
    grid = np.linspace(0.0, 10.0, 200001)
    values = PATH.path(grid, *theta)
    for level in (2.2, 3.0, 4.5, 6.0):
        first = grid[np.argmax(values >= level)]
        found = float(PATH.inv_path(level, *theta))
        assert found == pytest.approx(first, abs=1e-4)
        assert float(PATH.path(found, *theta)) == pytest.approx(
            level, abs=1e-12
        )
    # beyond the last knot the line's crossing is exact
    far = float(PATH.inv_path(60.0, *theta))
    assert far > 10.0
    assert float(PATH.path(far, *theta)) == pytest.approx(60.0, rel=1e-13)
    # a level the path never reaches
    assert np.isnan(PATH.inv_path(1.0, *theta))
    # a path that turns down for good never reaches a level above its top
    down = (5.0, -0.3, 0.2, 0.0, 0.0)
    assert PATH.slope(12.0, *down) < 0
    assert np.isnan(PATH.inv_path(50.0, *down))


def test_inv_path_takes_arrays_of_parameters_and_levels():
    rng = np.random.default_rng(4)
    theta = np.array(THETA) + 0.1 * rng.normal(size=(25, 5))
    together = PATH.inv_path(3.0, *theta.T)
    assert together.shape == (25,)
    one_by_one = [float(PATH.inv_path(3.0, *row)) for row in theta]
    assert np.array_equal(together, one_by_one)
    levels = np.array([2.2, 3.0, 1.0])
    out = PATH.inv_path(levels, *THETA)
    assert out.shape == (3,) and np.isnan(out[2])
    assert np.isnan(PATH.inv_path(np.nan, *THETA))


def test_natural_places_its_knots_at_quantiles_of_the_times():
    x = np.array([0.0, 1.0, 1.5, 2.0, 4.0, 4.5, 7.0, 9.0, 12.0, 20.0])
    placed = SplinePath.natural(df=4).placed(x)
    assert placed.boundary == (0.0, 20.0)
    assert placed.knots == tuple(np.quantile(x, [0.25, 0.5, 0.75]).tolist())
    assert placed.parameter_names == ["level", "rate", "s1", "s2", "s3"]
    assert placed.placed(x + 1.0) is placed
    assert SplinePath.natural(df=1).placed(x).knots == ()
    given = SplinePath(knots=[3.0]).placed(x)
    assert given.knots == (3.0,) and given.boundary == (0.0, 20.0)


def test_refusals_say_what_to_do():
    with pytest.raises(ValueError, match="ascending"):
        SplinePath(knots=[4.0, 2.0])
    with pytest.raises(ValueError, match="lo < hi"):
        SplinePath(knots=[], boundary=(5.0, 5.0))
    with pytest.raises(ValueError, match="strictly between"):
        SplinePath(knots=[12.0], boundary=(0.0, 10.0))
    with pytest.raises(ValueError, match="whole number"):
        SplinePath.natural(df=2.5)
    with pytest.raises(ValueError, match="1 \\(a straight line\\) or more"):
        SplinePath.natural(df=0)
    unplaced = SplinePath.natural(df=3)
    for call in (
        lambda: unplaced.path([1.0], 1.0, 1.0, 1.0, 1.0),
        lambda: unplaced.fit([1.0, 2.0, 3.0, 4.0, 5.0], [1.0] * 5),
        unplaced.to_dict,
    ):
        with pytest.raises(ValueError, match=r"\.placed\(x\)"):
            call()
    with pytest.raises(ValueError, match="smaller df"):
        SplinePath.natural(df=4).placed([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="more than one time"):
        SplinePath.natural(df=2).placed([3.0, 3.0])
    with pytest.raises(ValueError, match="median of 3"):
        PATH.check_readings([3, 3, 9])
    with pytest.raises(ValueError, match="use LinearPath"):
        SplinePath(knots=[1.0], boundary=(0.0, 2.0)).check_readings([2, 2])
    PATH.check_readings([5, 5, 2])


def test_saves_as_a_dictionary_and_pickles():
    saved = path_model_key(PATH)
    assert saved == {
        "name": "Spline",
        "knots": [2.0, 4.0, 7.0],
        "boundary": [0.5, 10.0],
    }
    again = get_path_model(json.loads(json.dumps(saved)))
    assert again.knots == PATH.knots and again.boundary == PATH.boundary
    copy = pickle.loads(pickle.dumps(PATH))
    x = np.linspace(0.0, 12.0, 9)
    assert np.array_equal(copy.path(x, *THETA), PATH.path(x, *THETA))
    with pytest.raises(ValueError, match="'Spline'"):
        get_path_model({"name": "Cubic", "knots": []})
    with pytest.raises(ValueError, match="boundary"):
        get_path_model({"name": "Spline", "knots": [1.0]})
    assert "knots: 2, 4, 7" in repr(PATH)
    assert "df=3" in repr(SplinePath.natural(df=3))
    assert "knots at 2, 4, 7 between 0.5 and 10" in PATH.formula


def _knee_units(units=8, seed=11):
    """Units whose reading is flat to age 4 and then climbs, each at its
    own level and rate, read 13 times with a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 13.0)
    x, y, i = [], [], []
    for k in range(units):
        level = 2.0 + 0.3 * rng.normal()
        climb = 0.8 + 0.1 * rng.normal()
        values = level + climb * np.maximum(t - 4.0, 0.0)
        x.append(t)
        y.append(values + 0.05 * rng.normal(size=t.size))
        i.append(np.full(t.size, k))
    return np.concatenate(x), np.concatenate(y), np.concatenate(i)


def test_degradation_analysis_follows_a_knee_a_line_cannot():
    x, y, i = _knee_units()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spline = DegradationAnalysis.fit(
            x, y, i, threshold=12.0, path=SplinePath.natural(df=3)
        )
        line = DegradationAnalysis.fit(x, y, i, threshold=12.0, path="linear")
    assert spline.path_model.knots == (4.0, 8.0)
    assert spline.path_model.boundary == (0.0, 12.0)
    assert spline.path_params.shape == (8, 4)
    # each unit truly crosses 12 at 4 + (12 - level) / climb, past the data
    assert spline.measurement_var < 0.1 * line.measurement_var
    assert np.all(spline.pseudo_failure_times > 12.0)
    # the line, pulled down by the flat years, crosses later than the
    # last pieces' climb says
    assert np.all(line.pseudo_failure_times > spline.pseudo_failure_times)
    assert 0.0 < float(spline.sf(16.0)) < 1.0


def test_degradation_model_on_a_spline_saves_loads_and_pickles():
    x, y, i = _knee_units()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DegradationAnalysis.fit(
            x, y, i, threshold=12.0, path=SplinePath.natural(df=3)
        )
    saved = json.loads(json.dumps(model.to_dict()))
    # a reader before the spline path would fail on the dictionary
    assert saved["schema"] == 6
    assert saved["path_model"]["knots"] == [4.0, 8.0]
    ages = np.array([13.0, 15.0, 18.0])
    for restored in (
        surpyval.from_dict(saved),
        pickle.loads(pickle.dumps(model)),
    ):
        assert restored.path_model.knots == (4.0, 8.0)
        assert np.array_equal(restored.sf(ages), model.sf(ages))
        assert np.array_equal(restored.path_params, model.path_params)
    # a linear model's file is what it was
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        line = DegradationAnalysis.fit(x, y, i, threshold=12.0, path="linear")
    assert line.to_dict()["path_model"] == "linear"
    assert line.to_dict()["schema"] < 5


def test_degradation_analysis_with_no_knots_is_the_linear_fit():
    x, y, i = _knee_units()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        line = DegradationAnalysis.fit(x, y, i, threshold=12.0, path="linear")
        same = DegradationAnalysis.fit(
            x, y, i, threshold=12.0, path=SplinePath()
        )
    assert np.array_equal(same.path_params, line.path_params)
    assert np.array_equal(same.pseudo_failure_times, line.pseudo_failure_times)
    ages = np.array([13.0, 15.0, 18.0])
    assert np.array_equal(same.sf(ages), line.sf(ages))


def test_degradation_analysis_refusals():
    x, y, i = _knee_units()
    with pytest.raises(ValueError, match="distinct"):
        # 13 readings a unit cannot fit 15 parameters
        DegradationAnalysis.fit(
            x, y, i, threshold=12.0, path=SplinePath.natural(df=14)
        )
    with pytest.raises(ValueError, match="reference-stress time"):
        DegradationAnalysis.fit(
            x,
            y,
            i,
            threshold=12.0,
            path=SplinePath.natural(df=2),
            acceleration="clock",
            Z=np.repeat(np.arange(8.0) % 2 + 1.0, 13),
            stress_ref=1.0,
        )
