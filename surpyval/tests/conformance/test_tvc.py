"""Time-varying-covariate evaluation (principles 3, 7 and 11; #433, #435).

``sf_tvc`` / ``Hf_tvc`` follow a covariate along a step schedule. A
constant schedule is the ordinary time-fixed covariate, so for every
registered model that has them:

- a constant ``StepSchedule`` gives ``sf(x, z)`` and ``Hf(x, z)``, at
  every time the query can hold -- inside the data, at ``0`` and below
  it (1 and 0 for a positive baseline, the baseline's own value for one
  with mass below ``0``);
- the query's shape is kept: a scalar gives a scalar, a 2-D query keeps
  its shape, an empty query gives an empty result;
- the path is measured from ``0``: the part of a schedule before ``0`` is
  ignored and a schedule starting after ``0`` has its first value held
  back to it, so a constant path starting anywhere is still ``sf(x, z)``;
- conditioning on survival to ``given`` divides by ``sf(given, z)``, and
  a missing ``given`` gives NaN.

A model whose family has no exact form along a step path (accelerated
life) refuses with ``NotImplementedError`` instead.

The same properties are checked along a constant ``CovariatePath``
(#172), which is integrated rather than summed.
"""

import numpy as np
import pytest

import surpyval as sp
from surpyval import CovariatePath, StepSchedule
from surpyval.tests.conformance.registry import CASES, REGRESSION, fitted

# Case name -> reason, for a property of this file that fails and is not
# fixed yet; each becomes a strict xfail (the reason starts with its issue).
KNOWN_FAILURES: dict[str, str] = {}


def _has_tvc(case):
    cls = sp
    for part in case.model_class.split(".")[1:]:
        cls = getattr(cls, part, None)
    return hasattr(cls, "sf_tvc")


def _tvc_cases():
    params = []
    for case in CASES:
        if case.interface != REGRESSION or not _has_tvc(case):
            continue
        marks = []
        if case.name in KNOWN_FAILURES:
            marks.append(
                pytest.mark.xfail(
                    strict=True, reason=KNOWN_FAILURES[case.name]
                )
            )
        params.append(pytest.param(case, id=case.name, marks=marks))
    return params


TVC_CASES = _tvc_cases()


def _rows(case):
    # Two covariate rows of the query, one each side of the middle.
    Z = np.asarray(case.Z, dtype=float)
    return [Z[0], Z[len(Z) // 2]]


def _times(case):
    # The case's times with 0 and a negative time in front.
    return np.concatenate([[-1.0, 0.0], np.asarray(case.x, dtype=float)])


def _evaluable(case, model):
    """Whether the model evaluates a step path; if not, check it refuses."""
    if getattr(model, "kind", None) != "Accelerated Life":
        return True
    with pytest.raises(NotImplementedError):
        model.sf_tvc([1.0], StepSchedule.constant(_rows(case)[0]))
    return False


def _close(got, ref, err_msg=""):
    np.testing.assert_allclose(
        np.asarray(got, float),
        np.asarray(ref, float),
        rtol=1e-9,
        atol=1e-12,
        err_msg=err_msg,
    )


@pytest.mark.parametrize("case", TVC_CASES)
def test_constant_schedule_is_the_fixed_covariate(case):
    model = fitted(case)
    if not _evaluable(case, model):
        return
    kw = case.call_kwargs
    x = _times(case)
    for z in _rows(case):
        schedule = StepSchedule.constant(z)
        _close(model.sf_tvc(x, schedule, **kw), model.sf(x, z, **kw), "sf")
        _close(model.Hf_tvc(x, schedule, **kw), model.Hf(x, z, **kw), "Hf")


@pytest.mark.parametrize("case", TVC_CASES)
def test_tvc_keeps_the_query_shape(case):
    model = fitted(case)
    if not _evaluable(case, model):
        return
    kw = case.call_kwargs
    z = _rows(case)[1]
    schedule = StepSchedule.constant(z)
    x = np.asarray(case.x, dtype=float)
    for x0 in (x[len(x) // 2], 0.0):
        got = model.sf_tvc(x0, schedule, **kw)
        assert np.shape(got) == (), np.shape(got)
        _close(got, model.sf(x0, z, **kw), f"scalar {x0}")
    grid = x[: 2 * (len(x) // 2)].reshape(2, -1)
    got = model.sf_tvc(grid, schedule, **kw)
    assert np.shape(got) == grid.shape, np.shape(got)
    _close(got, model.sf(grid.ravel(), z, **kw).reshape(grid.shape), "2-D")
    for empty in (np.array([]), np.empty((0, 3))):
        got = model.sf_tvc(empty, schedule, **kw)
        assert np.shape(got) == empty.shape, np.shape(got)


@pytest.mark.parametrize("case", TVC_CASES)
def test_path_is_measured_from_zero(case):
    model = fitted(case)
    if not _evaluable(case, model):
        return
    kw = case.call_kwargs
    x = _times(case)
    z, other = _rows(case)
    ref = model.sf(x, z, **kw)
    paths = {
        "starts below 0": StepSchedule.from_changepoints([-5.0], [z]),
        "differs below 0": StepSchedule.from_changepoints(
            [-5.0, 0.0], [other, z]
        ),
        "starts after 0": StepSchedule.from_changepoints([3.0], [z]),
    }
    for name, schedule in paths.items():
        _close(model.sf_tvc(x, schedule, **kw), ref, name)


@pytest.mark.parametrize("case", TVC_CASES)
def test_given(case):
    model = fitted(case)
    if not _evaluable(case, model):
        return
    kw = case.call_kwargs
    x = np.asarray(case.x, dtype=float)
    z = _rows(case)[1]
    schedule = StepSchedule.constant(z)
    given = float(x[1])
    later = x[x >= given]
    _close(
        model.sf_tvc(later, schedule, given=given, **kw),
        model.sf(later, z, **kw) / model.sf(given, z, **kw),
        "given",
    )
    got = model.sf_tvc(x, schedule, given=np.nan, **kw)
    assert np.isnan(got).all(), got


# -- the same properties along a CovariatePath (#172) ----------------------
#
# A constant CovariatePath is integrated by quadrature (or, for Cox, summed
# over the baseline jumps), not summed over segments, so these check that
# the two methods agree and that a path keeps every convention above.


def _constant_path(z):
    return CovariatePath.from_points([0.0], [z])


@pytest.mark.parametrize("case", TVC_CASES)
def test_constant_path_is_the_fixed_covariate(case):
    model = fitted(case)
    if not _evaluable(case, model):
        return
    kw = case.call_kwargs
    x = _times(case)
    for z in _rows(case):
        path = _constant_path(z)
        _close(model.sf_tvc(x, path, **kw), model.sf(x, z, **kw), "sf")
        _close(model.Hf_tvc(x, path, **kw), model.Hf(x, z, **kw), "Hf")


@pytest.mark.parametrize("case", TVC_CASES)
def test_path_keeps_the_query_shape(case):
    model = fitted(case)
    if not _evaluable(case, model):
        return
    kw = case.call_kwargs
    z = _rows(case)[1]
    path = _constant_path(z)
    x = np.asarray(case.x, dtype=float)
    for x0 in (x[len(x) // 2], 0.0):
        got = model.sf_tvc(x0, path, **kw)
        assert np.shape(got) == (), np.shape(got)
        _close(got, model.sf(x0, z, **kw), f"scalar {x0}")
    grid = x[: 2 * (len(x) // 2)].reshape(2, -1)
    got = model.sf_tvc(grid, path, **kw)
    assert np.shape(got) == grid.shape, np.shape(got)
    _close(got, model.sf(grid.ravel(), z, **kw).reshape(grid.shape), "2-D")
    for empty in (np.array([]), np.empty((0, 3))):
        got = model.sf_tvc(empty, path, **kw)
        assert np.shape(got) == empty.shape, np.shape(got)
    got = model.sf_tvc(np.array([x[0], np.nan]), path, **kw)
    assert np.isnan(got[1]) and not np.isnan(got[0]), got


@pytest.mark.parametrize("case", TVC_CASES)
def test_covariate_path_is_measured_from_zero(case):
    model = fitted(case)
    if not _evaluable(case, model):
        return
    kw = case.call_kwargs
    x = _times(case)
    z, other = _rows(case)
    ref = model.sf(x, z, **kw)
    paths = {
        "starts below 0": CovariatePath.from_points([-5.0], [z]),
        "differs below 0": CovariatePath.from_points(
            [-5.0, 0.0, 0.0], [other, other, z]
        ),
        "starts after 0": CovariatePath.from_points([3.0], [z]),
    }
    for name, path in paths.items():
        _close(model.sf_tvc(x, path, **kw), ref, name)


@pytest.mark.parametrize("case", TVC_CASES)
def test_path_given(case):
    model = fitted(case)
    if not _evaluable(case, model):
        return
    kw = case.call_kwargs
    x = np.asarray(case.x, dtype=float)
    z = _rows(case)[1]
    path = _constant_path(z)
    given = float(x[1])
    later = x[x >= given]
    _close(
        model.sf_tvc(later, path, given=given, **kw),
        model.sf(later, z, **kw) / model.sf(given, z, **kw),
        "given",
    )
    got = model.sf_tvc(x, path, given=np.nan, **kw)
    assert np.isnan(got).all(), got
