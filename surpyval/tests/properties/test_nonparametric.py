"""Properties of the non-parametric estimators on generated data (#379).

For Kaplan-Meier, Nelson-Aalen and Fleming-Harrington (exact and right
censored data, with left truncation) and Turnbull (any censoring, left
and right truncation):

- the curve is valid: ``sf`` and ``ff`` in [0, 1] and monotone,
  ``sf + ff == 1``, ``Hf == -log(sf)``, the jumps ``hf`` and ``df``
  non-negative;
- the fit does not depend on the order of the rows, nor on whether a
  count ``n`` is given or the row repeated;
- Turnbull with the Kaplan-Meier update is a local maximum of the
  likelihood: moving a little probability from any piece of its curve to
  any other does not raise the likelihood, computed here from the
  definition rather than by the estimator;
- on exact and right censored data (with or without left truncation)
  Turnbull with the Kaplan-Meier update is the Kaplan-Meier estimate.
"""

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

import surpyval as sp
from surpyval.tests.conformance.checks import (
    RULES,
    check_valid,
    compare,
    expanded,
    permuted,
)
from surpyval.tests.conformance.registry import predictions
from surpyval.tests.properties import known
from surpyval.tests.properties import strategies as gen
from surpyval.tests.properties.common import (
    case_for,
    query_points,
    quietly,
    step_log_likelihood,
)

KM_FAMILY = ("KaplanMeier", "NelsonAalen", "FlemingHarrington")
ESTIMATORS = KM_FAMILY + ("Turnbull",)


def _turnbull_data(**kw):
    return gen.xcnt(**kw).filter(
        lambda d: not known.turnbull_all_right_truncated(d)
    )


def _data(name):
    if name == "Turnbull":
        return _turnbull_data()
    return gen.xcnt(censoring=gen.RIGHT_CENSORING, right_truncation=False)


def _fit(name, data, **kw):
    return quietly(getattr(sp, name).fit, **data, **kw)


def _check_curve(case, model, x):
    for fname in ("sf", "ff", "Hf", "hf", "df"):
        values = np.asarray(getattr(model, fname)(x), float)
        if fname in case.jump_functions:
            # The jumps are NaN before the first failure (NonParametric.hf)
            values = values[~np.isnan(values)]
        check_valid(fname, values, RULES[fname])
    sf, ff, Hf = model.sf(x), model.ff(x), model.Hf(x)
    np.testing.assert_allclose(sf + ff, 1.0, rtol=0, atol=1e-12)
    keep = sf > 0
    np.testing.assert_allclose(Hf[keep], -np.log(sf[keep]), rtol=1e-12)
    assert np.all(np.isposinf(Hf[~keep]))


@pytest.mark.parametrize("name", ESTIMATORS)
@given(data=st.data())
def test_curve_is_valid(name, data):
    d = data.draw(_data(name), label="data")
    case = case_for(name, d)
    _check_curve(case, _fit(name, d), query_points(d))


@pytest.mark.parametrize("name", ESTIMATORS)
@given(data=st.data())
def test_row_order(name, data):
    d = data.draw(_data(name), label="data")
    perm = data.draw(gen.permutations(len(d["x"])), label="perm")
    case = case_for(name, d)
    x = query_points(d)
    ref = predictions(case, _fit(name, d), x=x)
    got = predictions(case, _fit(name, permuted(case, d, perm)), x=x)
    compare(case, got, ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("name", ESTIMATORS)
@given(data=st.data())
def test_counts_equal_repeated_rows(name, data):
    d = data.draw(_data(name), label="data")
    case = case_for(name, d)
    x = query_points(d)
    ref = predictions(case, _fit(name, d), x=x)
    got = predictions(case, _fit(name, expanded(case, d)), x=x)
    compare(case, got, ref, rtol=1e-9, atol=1e-12)


def _masses(R):
    """The probability on each ladder point, and past the last one."""
    return -np.diff(np.r_[1.0, R, 0.0])


def _curve(p):
    return 1.0 - np.cumsum(p)[:-1]


@given(data=_turnbull_data())
def test_turnbull_is_a_local_maximum(data):
    model = _fit("Turnbull", data, turnbull_estimator="Kaplan-Meier")
    # Only where the maximum exists and the EM reached it.
    assume(model.npmle == "exists" and model.converged)
    assume(not model.degenerate)
    ladder, R = model.x, model.R
    best = step_log_likelihood(ladder, R, data)
    assert np.isfinite(best)
    p = _masses(R)
    total = np.sum(data["n"])
    for i in np.flatnonzero(p > 1e-9):
        step = min(p[i], 1e-4)
        for j in range(p.size):
            if j == i:
                continue
            q = p.copy()
            q[i] -= step
            q[j] += step
            other = step_log_likelihood(ladder, _curve(q), data)
            assert other <= best + 1e-7 * total, (i, j, other - best)


@given(data=gen.xcnt(censoring=gen.RIGHT_CENSORING, right_truncation=False))
def test_turnbull_equals_kaplan_meier(data):
    x = query_points(data)
    km = _fit("KaplanMeier", data)
    tb = _fit("Turnbull", data, turnbull_estimator="Kaplan-Meier")
    assume(tb.npmle == "exists" and tb.converged)
    np.testing.assert_allclose(tb.sf(x), km.sf(x), rtol=0, atol=1e-7)
