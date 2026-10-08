"""Helpers for the property-based tests: registry cases on generated data,
query points, quiet fits, and log-likelihoods computed independently of
the fitters (#379)."""

import warnings
from dataclasses import replace

import numpy as np

from surpyval.tests.conformance.registry import CASE_BY_NAME
from surpyval.tests.properties.strategies import (
    EXACT,
    LEFT,
    RIGHT,
    STEP,
)

# The keys of a generated data dict that hold one entry per row, and those
# that are times (rescaled by the units property).
ROW_KEYS = ("x", "c", "n", "tl", "tr", "Z", "e", "i", "g")
TIME_KEYS = ("x", "tl", "tr")


def case_for(name, data, **changes):
    """The registered case ``name`` with its rows and times set to those
    ``data`` has, so the conformance checks rewrite ``data`` correctly."""
    case = CASE_BY_NAME[name]
    return replace(
        case,
        data=lambda: data,
        rows=tuple(k for k in ROW_KEYS if k in data),
        times=tuple(k for k in TIME_KEYS if k in data),
        **changes,
    )


def query_points(data):
    """Increasing times from 0 to past the largest value in ``data``, on a
    grid twice as fine as the data's, so every value and every gap
    between two values is visited."""
    top = max(
        float(np.max(np.asarray(data[k], dtype=float)[np.isfinite(data[k])]))
        for k in TIME_KEYS
        if k in data and np.isfinite(data[k]).any()
    )
    return np.arange(0.0, top + 2 * STEP, STEP / 2)


def quietly(fit, *args, **kwargs):
    """``fit(*args, **kwargs)`` with warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            return fit(*args, **kwargs)


def outcome(fit, *args, **kwargs):
    """``("ok", model)`` or ``("ValueError", message)`` for a quiet fit.

    Any other exception propagates: a valid input may be refused with a
    ``ValueError``, never with another type.
    """
    try:
        return "ok", quietly(fit, *args, **kwargs)
    except ValueError as e:
        return "ValueError", str(e)


def rows(data):
    """``xl, xr, c, n, tl, tr`` as arrays, one entry per row."""
    x = np.asarray(data["x"], dtype=float)
    xl = x if x.ndim == 1 else x[:, 0]
    xr = x if x.ndim == 1 else x[:, 1]
    size = xl.size
    c = np.asarray(data.get("c", np.zeros(size)), dtype=int)
    n = np.asarray(data.get("n", np.ones(size)), dtype=float)
    tl = np.asarray(data.get("tl", np.full(size, -np.inf)), dtype=float)
    tr = np.asarray(data.get("tr", np.full(size, np.inf)), dtype=float)
    return xl, xr, c, n, tl, tr


# ---------------------------------------------------------------------------
# Log-likelihoods, written from the definitions (not the fitters' code)
# ---------------------------------------------------------------------------
def _log(p):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p > 0, np.log(np.where(p > 0, p, 1.0)), -np.inf)


def step_log_likelihood(ladder, R, data):
    """Log-likelihood of ``data`` under a step survival curve.

    ``ladder`` is increasing (an exactly observed time may appear twice,
    as on a Turnbull ladder) and ``R[k]`` is the survival just after
    ``ladder[k]``. ``S(v)`` is ``R`` at the last point ``<= v`` (1 before
    the first); the survival just before an exact time ``v``, ``S(v-)``,
    is ``R`` at its first copy (the mass there is the open piece up to
    ``v``, the second copy's drop is the atom at ``v``). A row
    contributes ``S(x-) - S(x)`` (exact), ``S(x) - S(tr)`` (right
    censored), ``S(tl) - S(x)`` (left censored) or ``S(xl) - S(xr)``
    (interval), divided by ``S(tl) - S(tr)`` for its truncation window.
    """
    ladder = np.asarray(ladder, dtype=float)
    R = np.asarray(R, dtype=float)

    def S(v):
        idx = np.searchsorted(ladder, v, side="right") - 1
        out = np.where(idx < 0, 1.0, R[np.maximum(idx, 0)])
        out = np.where(np.isposinf(v), 0.0, out)
        return np.where(np.isneginf(v), 1.0, out)

    def S_before(v):
        idx = np.searchsorted(ladder, v, side="left")
        # The first copy of v, if v is on the ladder; else the last point
        # below v.
        on = (idx < ladder.size) & (
            ladder[np.minimum(idx, ladder.size - 1)] == v
        )
        idx = np.where(on, idx, idx - 1)
        return np.where(idx < 0, 1.0, R[np.maximum(idx, 0)])

    xl, xr, c, n, tl, tr = rows(data)
    # A censored row's event lies in its censoring interval *and* its
    # window: right censored under right truncation is ``x < X <= tr``.
    p = np.where(
        c == EXACT,
        S_before(xl) - S(xl),
        np.where(
            c == RIGHT,
            S(xl) - S(tr),
            np.where(c == LEFT, S(tl) - S(xr), S(xl) - S(xr)),
        ),
    )
    window = S(tl) - S(tr)
    return float(np.sum(n * (_log(p) - _log(window))))


def parametric_log_likelihood(model, data):
    """Log-likelihood of ``data`` under the fitted univariate ``model``,
    from its ``sf``, ``ff`` and ``df``: ``df(x)`` (exact), ``P(x < X <=
    tr)`` (right), ``P(tl < X <= x)`` (left) or ``P(xl < X <= xr)``
    (interval), each divided by ``P(tl < X <= tr)``. Differences are
    taken on whichever of ``sf`` and ``ff`` is smaller there, to keep
    their precision."""
    xl, xr, c, n, tl, tr = rows(data)

    def mass(a, b):
        # P(a < X <= b)
        with np.errstate(all="ignore"):
            upper = np.where(
                np.isposinf(b), 0.0, model.sf(np.where(np.isinf(b), 1, b))
            )
            lower = np.where(
                np.isneginf(a), 1.0, model.sf(np.where(np.isinf(a), 1, a))
            )
            by_sf = lower - upper
            F_b = np.where(
                np.isposinf(b), 1.0, model.ff(np.where(np.isinf(b), 1, b))
            )
            F_a = np.where(
                np.isneginf(a), 0.0, model.ff(np.where(np.isinf(a), 1, a))
            )
            by_ff = F_b - F_a
        return np.where(lower < 0.5, by_sf, by_ff)

    with np.errstate(all="ignore"):
        exact = np.asarray(model.df(xl), dtype=float)
    p = np.where(
        c == EXACT,
        exact,
        np.where(
            c == RIGHT,
            mass(xl, tr),
            np.where(c == LEFT, mass(tl, xr), mass(xl, xr)),
        ),
    )
    window = mass(tl, tr)
    with np.errstate(invalid="ignore"):
        terms = _log(p) - _log(window)
    # Far in a tail ``df``, ``sf`` and ``ff`` all underflow to 0, and a
    # row's ratio reads 0 / 0 (or 0) although its log is finite: an exact
    # 0.5 seen only in (0, 0.5] under a LogNormal with mu 1.32 and sigma
    # 0.047 (z = -43) contributes log(f(0.5) / F(0.5)) = 7.5 (#714). There
    # the terms are taken from the family's ``log_df``, ``log_ff`` and
    # ``log_sf`` instead.
    bad = ~np.isfinite(terms)
    if np.any(bad):
        logs = _log_space_terms(model, xl, xr, c, tl, tr)
        terms = np.where(bad & ~np.isnan(logs), logs, terms)
    return float(np.sum(n * terms))


def _log_space_terms(model, xl, xr, c, tl, tr):
    """Each row's log-likelihood from ``model.dist``'s ``log_df``,
    ``log_ff`` and ``log_sf``: a difference of two values of ``F`` (of
    ``R``) as ``log F(b) + log(1 - F(a) / F(b))``, on whichever tail is
    the smaller, as for ``mass`` above."""
    dist, params = model.dist, model.params

    def at(f, v, inf_value, neginf_value):
        out = f(np.where(np.isinf(v), 1.0, v), *params)
        out = np.where(np.isposinf(v), inf_value, out)
        return np.where(np.isneginf(v), neginf_value, out)

    def log_mass(a, b):
        # log P(a < X <= b)
        a_ff = at(dist.log_ff, a, 0.0, -np.inf)
        b_ff = at(dist.log_ff, b, 0.0, -np.inf)
        a_sf = at(dist.log_sf, a, -np.inf, 0.0)
        b_sf = at(dist.log_sf, b, -np.inf, 0.0)
        lower = b_ff + np.log1p(
            -np.exp(a_ff - np.where(b_ff == -np.inf, 0, b_ff))
        )
        upper = a_sf + np.log1p(
            -np.exp(b_sf - np.where(a_sf == -np.inf, 0, a_sf))
        )
        return np.where(a_sf < np.log(0.5), upper, lower)

    with np.errstate(all="ignore"):
        exact = np.asarray(dist.log_df(xl, *params), dtype=float)
        p = np.where(
            c == EXACT,
            exact,
            np.where(
                c == RIGHT,
                log_mass(xl, tr),
                np.where(c == LEFT, log_mass(tl, xr), log_mass(xl, xr)),
            ),
        )
        return p - log_mass(tl, tr)
