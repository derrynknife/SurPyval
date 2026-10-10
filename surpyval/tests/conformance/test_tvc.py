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
  a missing ``given`` gives NaN;
- along a changing step schedule and a ramp, conditional survival is 1
  at and before ``given``, ``S(x) / S(given)`` after it, and at most 1
  (but for the additive hazards' documented exception, #376; #523).

An accelerated life model is evaluated by cumulative exposure where its
life parameter scales time (Weibull, Exponential, Gamma, LogNormal); one
whose life parameter is a location (Normal, Gumbel, Logistic) refuses with
``NotImplementedError`` instead.

The same properties are checked along a constant ``CovariatePath``
(#172), which is integrated rather than summed.
"""

import warnings

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


# Accelerated life distributions whose life parameter scales time, and so
# evaluate along a path by cumulative exposure (#172 phase 2).
_SCALE_LIFE = ("Weibull", "Exponential", "Gamma", "LogNormal")


def _evaluable(case, model):
    """Whether the model evaluates a step path; if not, check it refuses."""
    if getattr(model, "kind", None) != "Accelerated Life":
        return True
    if model.distribution.name in _SCALE_LIFE:
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


# -- conditional survival before and after given (#523) --------------------


def _changing_paths(case, x):
    # A step change a quarter of the way along, and a ramp over the whole
    # range: H(x) - H(given) then differs from zero for x < given.
    z, other = _rows(case)
    top = float(np.max(x))
    return {
        "step": StepSchedule.from_changepoints([0.0, top / 4], [z, other]),
        "ramp": CovariatePath.from_points([0.0, top], [z, other]),
    }


@pytest.mark.parametrize("case", TVC_CASES)
def test_conditional_survival_is_one_up_to_given(case):
    # S(x | given) is 1 for x <= given -- survival to x is certain -- and
    # S(x) / S(given) after it. It exceeded 1 before given (#523: 1.18 for
    # WeibullPO, 1.21 for CoxPH).
    model = fitted(case)
    if not _evaluable(case, model):
        return
    kw = case.call_kwargs
    x = _times(case)
    given = float(np.median(case.x))
    after = x[x > given]
    for name, path in _changing_paths(case, x).items():
        got = model.sf_tvc(x, path, given=given, **kw)
        np.testing.assert_array_equal(got[x <= given], 1.0, err_msg=name)
        ratio = model.sf_tvc(after, path, **kw) / model.sf_tvc(
            given, path, **kw
        )
        np.testing.assert_allclose(
            got[x > given], ratio, rtol=1e-8, atol=1e-12, err_msg=name
        )
        if getattr(model, "kind", None) != "Additive Hazard":
            # The additive hazard can fall below 0, and survival exceed 1,
            # after given too (principle 9's documented exception, #376).
            assert np.all(got <= 1.0), (name, got.max())
        got = model.sf_tvc(x, path, given=np.nan, **kw)
        assert np.isnan(got).all(), (name, got)


# -- bounds and the mean along a path (#172 phase 2) -----------------------
#
# cb_tvc bounds sf_tvc as cb bounds sf, and mean_tvc integrates sf_tvc, so
# along a constant path both are the time-fixed model's: cb, and the
# integral of sf. (Cox has neither: it has no cb, and its baseline ends at
# the last event.)


def _has(model, name):
    return hasattr(model, name) and _evaluable(None, model)


@pytest.mark.parametrize("case", TVC_CASES)
def test_constant_path_bounds_are_cb(case):
    model = fitted(case)
    if not _has(model, "cb_tvc"):
        return
    x = _times(case)
    z = _rows(case)[1]
    for Z in (_constant_path(z), StepSchedule.constant(z)):
        for on in ("sf", "ff", "Hf"):
            for bound in ("two-sided", "lower"):
                got = model.cb_tvc(x, Z, on=on, bound=bound)
                ref = model.cb(x, z, on=on, bound=bound)
                np.testing.assert_allclose(
                    got,
                    ref,
                    rtol=1e-7,
                    atol=1e-10,
                    err_msg=f"{type(Z).__name__} {on} {bound}",
                )


@pytest.mark.parametrize("case", TVC_CASES)
def test_bounds_keep_the_query_shape(case):
    model = fitted(case)
    if not _has(model, "cb_tvc"):
        return
    z = _rows(case)[1]
    x = np.asarray(case.x, dtype=float)
    top = float(np.max(x))
    path = CovariatePath.from_points([0.0, top], [z, _rows(case)[0]])
    assert np.shape(model.cb_tvc(x[1], path)) == (2,)
    assert np.shape(model.cb_tvc(x[1], path, bound="upper")) == ()
    grid = x[: 2 * (len(x) // 2)].reshape(2, -1)
    got = model.cb_tvc(grid, path)
    assert got.shape == grid.shape + (2,), got.shape
    np.testing.assert_allclose(
        got.reshape(-1, 2), model.cb_tvc(grid.ravel(), path), rtol=1e-12
    )
    assert model.cb_tvc(np.array([]), path).shape == (0, 2)
    # Given survival to g, the bound at and before g is [1, 1].
    given = float(np.median(x))
    got = model.cb_tvc(x, path, given=given)
    np.testing.assert_array_equal(got[x <= given], 1.0)
    assert np.isnan(model.cb_tvc(x, path, given=np.nan)).all()


@pytest.mark.parametrize("case", TVC_CASES)
def test_constant_path_mean_is_the_integral_of_sf(case):
    from scipy.integrate import quad

    model = fitted(case)
    if not _has(model, "mean_tvc"):
        return
    kw = case.call_kwargs
    z = _rows(case)[1]

    def sf(t):
        return float(model.sf(np.array([t]), z, **kw)[0])

    with warnings.catch_warnings():
        # An additive hazard negative somewhere warns (#376).
        warnings.simplefilter("ignore")
        mean = model.mean_tvc(_constant_path(z), **kw)
        mean_step = model.mean_tvc(StepSchedule.constant(z), **kw)
        given = float(np.median(case.x))
        mrl = model.mean_tvc(_constant_path(z), given=given, **kw)
        ref_mrl = quad(sf, given, np.inf, epsabs=0, epsrel=1e-12, limit=500)[
            0
        ] / sf(given)
        if np.isfinite(mean):
            ref = quad(sf, 0, np.inf, epsabs=0, epsrel=1e-12, limit=500)[0]
            if model.distribution.support[0] < 0:
                # (an additive model's sf is nan before its support starts,
                # where nothing has failed yet, #828)
                ref -= quad(
                    lambda t: 1 - np.nan_to_num(sf(t), nan=1.0),
                    -np.inf,
                    0,
                    epsabs=0,
                    epsrel=1e-12,
                    limit=500,
                )[0]
            assert abs(mean / ref - 1) < 1e-8, (mean, ref)
    # An additive hazard that turns negative can leave the mean infinite
    # (survival above 1) or undefined (nan, F growing before 0).
    if np.isfinite(mean_step):
        assert abs(mean / mean_step - 1) < 1e-12, (mean, mean_step)
    else:
        np.testing.assert_array_equal(mean, mean_step)
    if np.isfinite(ref_mrl):
        assert abs(mrl / ref_mrl - 1) < 1e-8, (mrl, ref_mrl)
    else:
        assert mrl == ref_mrl, (mrl, ref_mrl)
    assert np.isnan(model.mean_tvc(_constant_path(z), given=np.nan, **kw))
