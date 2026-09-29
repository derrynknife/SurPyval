"""Properties of the parametric (maximum likelihood) fits on generated
data (#379).

Data: any mix of exact, right, left and interval censoring, counts, and
left and right truncation consistent with each row.

- **no crash**: a fit either returns a model with finite parameters and
  a valid survival curve, or refuses the data with a ``ValueError``
  saying why (too few distinct failures, only censored rows); never
  another exception type, and never NaN parameters;
- **local optimum**: no point a small step from the fitted parameters,
  and not the optimiser's starting point, has a higher likelihood. The
  likelihood is computed here from the fitted model's ``sf``, ``ff`` and
  ``df`` (see ``common.parametric_log_likelihood``), not by the fitter;
- **row order**, **units** and **counts**: permuting the rows, changing
  the time unit (every family here is closed under rescaling: a scale,
  location-scale or log-location-scale family) and replacing counts by
  repeated rows give the same fit.

An optimiser stops at its own tolerance, and a flat likelihood (a small
sample) turns that into a visible difference in the parameters. So two
fits count as the same when their predictions agree to ``RTOL``, or,
failing that, when they reach the same likelihood (the same optimum,
reached from another side).
"""

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

import surpyval as sp
from surpyval.tests.conformance.checks import expanded, permuted, rescaled
from surpyval.tests.conformance.registry import predictions
from surpyval.tests.properties import known
from surpyval.tests.properties import strategies as gen
from surpyval.tests.properties.common import (
    case_for,
    outcome,
    parametric_log_likelihood,
    query_points,
    rows,
)

# Every family here is closed under a change of unit. A fit to truncated
# data can take seconds, so the default profile runs two families (a scale
# and a location-scale one); the nightly profile runs them all.
DISTRIBUTIONS: tuple[str, ...] = ("Weibull", "Normal")
if gen.THOROUGH:
    DISTRIBUTIONS += ("Exponential", "LogNormal", "Gamma", "LogLogistic")
    DISTRIBUTIONS += ("Gumbel", "Logistic")
K = 7.3  # the unit change
RTOL = 1e-3
# Log-likelihoods of two fits of one data set are equal to this, per
# observation, when both are at the optimum.
LL_TOL = 1e-6


def _case(name, data):
    return case_for(name, data, rtol=RTOL)


def _ll(model, data):
    return parametric_log_likelihood(model, data)


def _same_fit(case, ref_data, ref, got_data, got, scale=None):
    """Assert the fit ``got`` (to ``got_data``, in a unit ``scale`` times
    the original) is the fit ``ref`` (see the module docstring).

    Two degenerate fits (see :func:`_degenerate`) are not compared: both
    stopped somewhere on the way to a limit."""
    dead = (_degenerate(ref, ref_data), _degenerate(got, got_data))
    if all(dead):
        return
    assert not any(dead), "one fit is degenerate and the other not"
    x = query_points(ref_data)
    a = predictions(case, ref, x=x)
    b = predictions(case, got, x=x if scale is None else x * scale)
    try:
        for key in ("sf", "ff", "Hf"):
            np.testing.assert_allclose(b[key], a[key], rtol=RTOL, atol=1e-6)
        return
    except AssertionError:
        pass
    # The same optimum reached from another side: equal likelihoods (on
    # the data each was fitted to; a density picks up 1/K per exact row).
    _, _, c, n, _, _ = rows(ref_data)
    ll_ref = _ll(ref, ref_data)
    ll_got = _ll(got, got_data)
    if scale is not None:
        ll_got += np.sum(n[c == gen.EXACT]) * np.log(scale)
    assert abs(ll_got - ll_ref) <= LL_TOL * np.sum(n), (ll_ref, ll_got)


def _fit(name, data):
    return outcome(getattr(sp, name).fit, **data)


def _has_optimum(name, data):
    """Whether the likelihood can have a maximum: not where a family that
    can approach a point mass has one in reach (see
    ``known.point_mass_supremum``). The Exponential cannot."""
    return name == "Exponential" or not known.point_mass_supremum(data)


def _degenerate(model, data):
    """Whether the fit sits at the edge of the parameter space: a spike
    (the central 98% of the distribution narrower than a thousandth of
    the data's grid) or a spread (a 1% or 99% quantile a thousand times
    beyond the data). The likelihood's supremum is then a limit, not a
    point, and a local-optimum or invariance property does not apply."""
    times = np.concatenate(
        [np.ravel(data[k]) for k in ("x", "tl", "tr") if k in data]
    )
    top = np.max(np.abs(times[np.isfinite(times)]))
    with np.errstate(all="ignore"):
        q01, q99 = np.asarray(model.qf([0.01, 0.99]), dtype=float)
    return bool(
        not (np.isfinite(q01) and np.isfinite(q99))
        or q99 - q01 < 1e-3 * gen.STEP
        or q99 > 1e3 * top
        or q01 < -1e3 * top
        or 0 <= q01 < 1e-9 * top
    )


@pytest.mark.parametrize("name", DISTRIBUTIONS)
@given(data=gen.xcnt())
def test_fit_succeeds_or_refuses(name, data):
    # Data whose likelihood has no maximum are accepted, a known failure
    # (see known.point_mass_supremum), and some take tens of seconds.
    assume(_has_optimum(name, data))
    status, model = _fit(name, data)
    if status != "ok":
        return
    params = np.asarray(model.params, dtype=float)
    assert np.all(np.isfinite(params)), params
    sf = np.asarray(model.sf(query_points(data)), dtype=float)
    assert not np.any(np.isnan(sf)), sf
    assert np.all((sf >= 0) & (sf <= 1)), sf
    assert np.all(np.diff(sf) <= 1e-12), sf


def _nearby(fitter, params):
    """Parameter vectors a small step from ``params``, one coordinate at a
    time, inside the parameter bounds."""
    out = []
    for j, (lo, hi) in enumerate(fitter.bounds):
        for sign in (-1, 1):
            step = 1e-4 * max(abs(params[j]), 1e-3)
            p = np.array(params, dtype=float)
            p[j] += sign * step
            if (lo is not None and p[j] <= lo) or (
                hi is not None and p[j] >= hi
            ):
                continue
            out.append(p)
    return out


def _start(model):
    """The optimiser's starting point, where there was one."""
    info = getattr(model, "fitting_info", None) or {}
    if info.get("init") is None or "inv_trans" not in info:
        return None
    return np.asarray(info["inv_trans"](info["const"](info["init"])), float)


@pytest.mark.parametrize("name", DISTRIBUTIONS)
@given(data=gen.xcnt())
def test_fit_is_a_local_optimum(name, data):
    assume(_has_optimum(name, data))
    status, model = _fit(name, data)
    assume(status == "ok" and not _degenerate(model, data))
    fitter = getattr(sp, name)
    best = _ll(model, data)
    assert np.isfinite(best), best
    total = np.sum(rows(data)[3])
    candidates = _nearby(fitter, model.params)
    start = _start(model)
    if start is not None:
        candidates.append(start)
    for params in candidates:
        other = _ll(fitter.from_params(params), data)
        assert other <= best + LL_TOL * total, (params, other - best)


@pytest.mark.parametrize("name", DISTRIBUTIONS)
@given(data=st.data())
def test_row_order(name, data):
    d = data.draw(gen.xcnt(min_rows=2), label="data")
    assume(_has_optimum(name, d))
    perm = data.draw(gen.permutations(len(d["x"])), label="perm")
    case = _case(name, d)
    other = permuted(case, d, perm)
    status, ref = _fit(name, d)
    status2, got = _fit(name, other)
    assert status == status2, (ref, got)
    if status == "ok":
        _same_fit(case, d, ref, other, got)


@pytest.mark.parametrize("name", DISTRIBUTIONS)
@given(data=gen.xcnt(left_truncation=False, right_truncation=False))
def test_units(name, data):
    # Untruncated data only: with truncation the fit depends on the unit,
    # a known failure (see known.truncated).
    assume(_has_optimum(name, data))
    case = _case(name, data)
    other = rescaled(case, data, K)
    status, ref = _fit(name, data)
    status2, got = _fit(name, other)
    assert status == status2, (ref, got)
    if status == "ok":
        _same_fit(case, data, ref, other, got, scale=K)


@pytest.mark.parametrize("name", DISTRIBUTIONS)
@given(data=gen.xcnt())
def test_counts_equal_repeated_rows(name, data):
    assume(np.any(data["n"] > 1) and _has_optimum(name, data))
    case = _case(name, data)
    other = expanded(case, data)
    status, ref = _fit(name, data)
    status2, got = _fit(name, other)
    assert status == status2, (ref, got)
    if status == "ok":
        _same_fit(case, data, ref, other, got)
