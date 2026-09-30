"""Conditional survival, ``sf(x, given=g)`` (#514).

For every registered univariate model whose ``sf`` takes ``given``:
:math:`S(x \\mid T > g) = S(x) / S(g)` for ``x > g`` and 1 for ``x <= g``
(the meaning of the regression models' ``sf_tvc(..., given=)``), ``ff``
its complement, ``cs(x - g, g)`` the same where the model has ``cs``, and
shape in, shape out (principle 7) with ``given`` a scalar or an array that
broadcasts against ``x``.

The parametric, non-parametric and mixture models must take ``given``;
the other univariate models (the degradation models' induced
distributions, Royston-Parmar, the point masses) do not yet (#514).
"""

import inspect

import numpy as np
import pytest

from surpyval.tests.conformance.registry import (
    CASES,
    UNIVARIATE,
    fitted,
)

MUST = ("Parametric", "NonParametric", "MixtureModel")


def _takes_given(model, fname="sf"):
    try:
        params = inspect.signature(getattr(model, fname)).parameters
    except (TypeError, ValueError, AttributeError):
        return False
    return "given" in params


def _cases():
    out = []
    for case in CASES:
        if case.interface != UNIVARIATE:
            continue
        if case.model_class.rsplit(".", 1)[-1] in MUST:
            out.append(pytest.param(case, id=case.name))
    return out


CASES_GIVEN = _cases()


def _points(case, model):
    """Query times and a conditioning time inside the data, where the
    survival to it is positive."""
    x = np.asarray(case.x, dtype=float)
    x = x[np.isfinite(x)]
    # at least six points (a Bernoulli's are 0 and 1), repeated in turn
    x = np.resize(x, max(x.size, 6))
    # one of the query times, so that it is a valid value of a discrete
    # model's variable too
    g = float(np.sort(x)[int(0.3 * len(x))])
    if not model.sf(g) > 0:
        g = float(np.min(x))
    return x, g


@pytest.mark.parametrize("case", CASES_GIVEN)
def test_sf_and_ff_take_given(case):
    model = fitted(case)
    assert _takes_given(model, "sf") and _takes_given(model, "ff")


@pytest.mark.parametrize("case", CASES_GIVEN)
def test_given_is_the_ratio_of_survivals(case):
    model = fitted(case)
    x, g = _points(case, model)
    got = np.asarray(model.sf(x, given=g), float)
    ratio = np.asarray(model.sf(x), float) / float(model.sf(g))
    want = np.where(x <= g, 1.0, ratio)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)
    ff = np.asarray(model.ff(x, given=g), float)
    np.testing.assert_allclose(ff + got, 1.0, rtol=0, atol=1e-12)
    assert np.all((got >= 0) & (got <= 1))
    if hasattr(model, "cs"):
        later = x[x > g]
        np.testing.assert_allclose(
            np.asarray(model.cs(later - g, g), float),
            np.asarray(model.sf(later, given=g), float),
            rtol=1e-9,
            atol=1e-12,
        )


@pytest.mark.parametrize("case", CASES_GIVEN)
@pytest.mark.parametrize("fname", ["sf", "ff"])
def test_given_keeps_the_query_shape(case, fname):
    model = fitted(case)
    x, g = _points(case, model)
    f = getattr(model, fname)
    # scalar in, scalar out
    k = len(x) // 2
    assert np.shape(f(x[k], given=g)) == ()
    # 2-D in, 2-D out, agreeing with the flat query
    grid = x[:4].reshape(2, 2)
    got = np.asarray(f(grid, given=g))
    assert got.shape == (2, 2)
    np.testing.assert_array_equal(got.ravel(), f(x[:4], given=g))
    # empty in, empty out
    assert np.shape(f(np.array([]), given=g)) == (0,)
    # given broadcasts against x: one per column, and one per point
    gs = np.array([g, x[0]])
    got = np.asarray(f(grid, given=gs))
    assert got.shape == (2, 2)
    for j in range(2):
        np.testing.assert_array_equal(got[:, j], f(grid[:, j], given=gs[j]))
    np.testing.assert_array_equal(
        np.asarray(f(grid, given=np.full((2, 2), g))), f(grid, given=g)
    )


@pytest.mark.parametrize("case", CASES_GIVEN)
def test_given_that_does_not_broadcast_is_refused(case):
    model = fitted(case)
    x, g = _points(case, model)
    with pytest.raises(ValueError, match="given"):
        model.sf(x[:3], given=[g, g])


@pytest.mark.parametrize("case", CASES_GIVEN)
def test_missing_given_is_missing(case):
    model = fitted(case)
    x, _ = _points(case, model)
    got = np.asarray(model.sf(x[:3], given=[np.nan, x[0], np.nan]))
    assert np.isnan(got[[0, 2]]).all() and not np.isnan(got[1])
