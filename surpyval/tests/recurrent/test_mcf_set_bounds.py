"""``set_bounds`` on the non-parametric MCFs (``NonParametricCounting``
and ``CauseSpecificMCF``): 0 from ``lower`` to the origin, the value at
the last observed time carried to ``upper``, NaN outside (principle 11)."""

import copy
import json
import warnings
from typing import Any

import numpy as np
import pytest

import surpyval
from surpyval.recurrent import CauseSpecificMCF, NonParametricCounting

X = [3, 9, 20, 35, 56, 60, 11, 44, 60]
I = [1, 1, 1, 1, 1, 1, 2, 2, 2]  # noqa: E741
C = [0, 0, 0, 0, 0, 1, 0, 0, 1]
E = ["a", "b", "a", "a", "b", None, "b", "a", None]
INTERPS = ("step", "linear")
# The singleton fitter looks like a class to mypy.
NPC: Any = NonParametricCounting
CB = [
    {"bound": bound, "bound_type": bound_type, "interp": interp}
    for bound in ("two-sided", "upper", "lower")
    for bound_type in ("exp", "normal")
    for interp in INTERPS
]


def _fit():
    return NPC.fit(X, i=I, c=C)


def test_a_fit_has_no_support_and_set_bounds_chains():
    model = _fit()
    assert model.support is None
    assert model.set_bounds(-1, 100) is model
    assert model.support == (-1.0, 100.0)


@pytest.mark.parametrize("interp", INTERPS)
def test_mcf_regions(interp):
    model = _fit()
    bounded = copy.deepcopy(model).set_bounds(-10.0, 100.0)
    inside = np.linspace(0.0, 60.0, 31)
    np.testing.assert_array_equal(
        bounded.mcf(inside, interp=interp), model.mcf(inside, interp=interp)
    )
    for q in ([-10.1], [100.1], [np.nan], [-np.inf], [np.inf]):
        assert np.isnan(bounded.mcf(q, interp=interp)).all()
    start = bounded.mcf([-10.0, -5.0, -1e-9], interp=interp)
    np.testing.assert_array_equal(start, 0.0)
    assert not np.signbit(start).any()
    # Unbounded, NaN before the origin and after the last observed time.
    assert np.isnan(model.mcf([-5.0, 61.0], interp=interp)).all()
    after = bounded.mcf([60.5, 100.0], interp=interp)
    np.testing.assert_array_equal(after, model.mcf_hat[-1])


@pytest.mark.parametrize("kw", CB)
def test_mcf_cb_regions(kw):
    model = _fit()
    bounded = copy.deepcopy(model).set_bounds(-10.0, 100.0)
    inside = np.linspace(0.0, 60.0, 31)
    np.testing.assert_array_equal(
        bounded.mcf_cb(inside, **kw), model.mcf_cb(inside, **kw)
    )
    assert np.isnan(bounded.mcf_cb([-11.0, 101.0, np.nan], **kw)).all()
    np.testing.assert_array_equal(bounded.mcf_cb([-10.0, -1.0], **kw), 0.0)
    at_last = model.mcf_cb([60.0], **kw)
    np.testing.assert_array_equal(
        bounded.mcf_cb([61.0, 100.0], **kw),
        np.concatenate([at_last, at_last]),
    )


def test_infinite_bounds():
    bounded = _fit().set_bounds(-np.inf, np.inf)
    for interp in INTERPS:
        np.testing.assert_array_equal(
            bounded.mcf([-np.inf, np.inf], interp=interp),
            [0.0, bounded.mcf_hat[-1]],
        )
    assert bounded.mcf_cb([np.inf]).shape == (1, 2)


def test_negative_times():
    model = NPC.fit([-5, -3, -1, 0, 2], c=[0, 0, 0, 0, 1], tl=-10)
    assert model._origin() == -10.0
    bounded = copy.deepcopy(model).set_bounds(-20.0, 5.0)
    np.testing.assert_array_equal(
        bounded.mcf([-21.0, -20.0, -15.0, -4.0, 2.0, 5.0, 6.0]),
        [np.nan, 0.0, 0.0, 1.0, 4.0, 4.0, np.nan],
    )
    with pytest.raises(ValueError, match="origin of the MCF"):
        model.set_bounds(-9.0, 5.0)


@pytest.mark.parametrize(
    "lower, upper, match",
    [
        (1.0, 100.0, r"'lower' \(1.0\) is above the origin .*\(0.0\)"),
        (0.0, 59.0, r"'upper' \(59.0\) is below the last observed time"),
        (0.0, 0.0, "'lower' must be below 'upper'"),
        (0.0, np.nan, "not NaN"),
    ],
)
def test_invalid_bounds_are_refused(lower, upper, match):
    model = _fit()
    with pytest.raises(ValueError, match=match):
        model.set_bounds(lower, upper)
    assert model.support is None


def test_no_raw_numpy_warning():
    bounded = _fit().set_bounds(-np.inf, 100)
    q = np.array([-np.inf, -1.0, 0.0, 3.0, 30.0, 60.0, 100.0, 101.0, np.nan])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for interp in INTERPS:
            bounded.mcf(q, interp=interp)
        for kw in CB:
            bounded.mcf_cb(q, **kw)


@pytest.mark.parametrize("support", [(-1.0, 100.0), (-np.inf, np.inf)])
def test_serialisation_round_trip(support):
    bounded = _fit().set_bounds(*support)
    d = bounded.to_dict()
    assert d["schema"] == 2
    restored = surpyval.from_dict(json.loads(json.dumps(d, allow_nan=False)))
    assert restored.support == support
    q = np.array([-5.0, -1.0, 10.0, 70.0, 150.0, np.inf])
    for interp in INTERPS:
        np.testing.assert_array_equal(
            restored.mcf(q, interp=interp), bounded.mcf(q, interp=interp)
        )
        np.testing.assert_array_equal(
            restored.mcf_cb(q, interp=interp),
            bounded.mcf_cb(q, interp=interp),
        )


def test_without_bounds_the_dictionary_is_unchanged():
    d = _fit().to_dict()
    assert "support" not in d
    assert d["schema"] == 1


# -- CauseSpecificMCF ------------------------------------------------------


def _fit_cs():
    return CauseSpecificMCF.fit(X, i=I, c=C, e=E)


def test_cause_specific_bounds_every_cause():
    model = _fit_cs()
    assert model.support is None
    assert model.set_bounds(-10, 100) is model
    assert model.support == (-10.0, 100.0)
    for cause in model.event_types:
        assert model.models[cause].support == (-10.0, 100.0)


@pytest.mark.parametrize("interp", INTERPS)
@pytest.mark.parametrize("cause", ["a", "b"])
def test_cause_specific_regions(cause, interp):
    model = _fit_cs()
    bounded = copy.deepcopy(model).set_bounds(-10.0, 100.0)
    q = np.array([-11.0, -10.0, -1.0, 0.0, 20.0, 60.0, 61.0, 100.0, 101.0])
    q = np.append(q, np.nan)
    v = bounded.mcf(q, cause, interp=interp)
    last = model.mcf([60.0], cause, interp=interp)[0]
    np.testing.assert_array_equal(v[[0, -2, -1]], np.nan)
    np.testing.assert_array_equal(v[1:3], 0.0)
    np.testing.assert_array_equal(
        v[3:6], model.mcf(q[3:6], cause, interp=interp)
    )
    np.testing.assert_array_equal(v[6:8], last)
    cb = bounded.mcf_cb(q, cause, interp=interp)
    assert np.isnan(cb[[0, -2, -1]]).all()
    np.testing.assert_array_equal(cb[1:3], 0.0)
    np.testing.assert_array_equal(
        cb[6:8], np.repeat(model.mcf_cb([60.0], cause, interp=interp), 2, 0)
    )


def test_cause_specific_invalid_bounds_leave_the_model_unchanged():
    model = _fit_cs()
    with pytest.raises(ValueError, match="last observed time"):
        model.set_bounds(0, 50)
    assert model.support is None
    assert all(m.support is None for m in model.models.values())


@pytest.mark.parametrize("support", [(-1.0, 100.0), (-np.inf, np.inf)])
def test_cause_specific_serialisation_round_trip(support):
    bounded = _fit_cs().set_bounds(*support)
    d = bounded.to_dict()
    assert d["schema"] == 2
    restored = surpyval.from_dict(json.loads(json.dumps(d, allow_nan=False)))
    assert restored.support == support
    q = np.array([-5.0, -1.0, 10.0, 70.0, 150.0, np.inf])
    for cause in ("a", "b"):
        assert restored.models[cause].support == support
        np.testing.assert_array_equal(
            restored.mcf(q, cause), bounded.mcf(q, cause)
        )
        np.testing.assert_array_equal(
            restored.mcf_cb(q, cause), bounded.mcf_cb(q, cause)
        )
    unbounded = _fit_cs().to_dict()
    assert "support" not in unbounded and unbounded["schema"] == 1
