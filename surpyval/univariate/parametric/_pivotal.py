"""The pivotal Monte Carlo bounds of a probability-plotting fit
(``method="pivotal"`` on ``cb``, ``quantile_cb`` and ``param_cb`` of a
model fitted with ``how="MPP"``).

In plain words
--------------
A probability-plotting (rank regression) fit draws a straight line through
the plotted points. Its error has no likelihood to read a standard error
from, but for a family that is a location and a scale on the time or
log-time scale (the Weibull, LogNormal, LogLogistic, Normal, Logistic,
Gumbel, GumbelLEV and Exponential), the error of the line *relative to its own
slope* has the same distribution whatever the true parameters are: a
**pivotal quantity**. That distribution can be simulated once, from the
standard member of the family (location 0, scale 1), by drawing samples of
the same size with the same failures and suspensions in the same rank
positions, and fitting each by the same regression (the same plotting
positions, the same direction). Reading its percentiles and undoing the
standardisation gives the bounds. For complete samples and for samples
censored after the ``r``-th failure (type II) the bounds are exact up to
the Monte Carlo error of the percentiles; with other censoring they hold
the observed rank pattern fixed (as Abernethy's *New Weibull Handbook* and
R's ``WeibullR`` do) and are approximate.

The formulas
------------
On the transformed scale ``y = log t`` (or ``t``), ``y = u + b Z`` with
``Z`` the family's standard variable. The fit's ``(u_hat, b_hat)`` obey
``u_hat = u + b U*`` and ``b_hat = b B*``, where ``(U*, B*)`` is the fit of
a standard sample; so ``B* = b_hat / b``, ``T = (u_hat - u) / b_hat =
U* / B*``, and for the ``p`` quantile ``y_p = u + b z_p``, ``W_p =
(u_hat - y_p) / b_hat = (U* - z_p) / B*`` are pivotal (Lawless 2003,
section 5.1). Their percentiles over the ``n_sim`` simulated fits give

- ``b`` in ``[b_hat / B*_(1 - a), b_hat / B*_(a)]``,
- ``u`` in ``[u_hat - b_hat T_(1 - a), u_hat - b_hat T_(a)]``,
- ``y_p`` in ``[u_hat - b_hat W_(1 - a), u_hat - b_hat W_(a)]``,

and, inverting the last over ``p``, the failure probability at ``t``,
``F(t) = G((y - u) / b)``, in ``[G(D_(a)), G(D_(1 - a))]`` with ``D = U* -
v B*`` and ``v = (u_hat - y) / b_hat`` (``P(W_z <= v) = P(D <= z)``, so
the band on ``F`` and the bounds on the quantiles are one band). Each
percentile is the ``(n_sim + 1) a``-th smallest draw, the rule of the
bootstrap bounds (``utils.linalg.bootstrap_quantile``), which makes the
interval's coverage exact over the simulation too when ``(n_sim + 1) a``
is a whole number: hence the default ``n_sim = 9999``.

The Exponential's line goes through the origin (one parameter): it is the
same with ``b = 1`` known, ``u = -log(failure_rate)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.utils.linalg import bootstrap_quantile
from surpyval.utils.rng import as_generator

#: The default number of simulated fits: ``(n_sim + 1) a`` is a whole
#: number for every level that is a multiple of 1e-4.
DEFAULT_N_SIM = 9999

#: The simulated fits are drawn in blocks of at most this many numbers.
_BLOCK = 2**21


@dataclass(frozen=True)
class _Family:
    """How a location-scale family's parameters map to the location
    ``u`` and scale ``b`` of ``y = log t`` (``log_time``) or ``y = t``."""

    log_time: bool
    #: ``params -> (u, b)``.
    to_ub: Callable[[Any], tuple[float, float]]
    #: For each parameter, in order: which of ``u`` and ``b`` it is a
    #: function of, and that function.
    roles: tuple[tuple[str, Callable[[Any], Any]], ...]
    #: The line through the origin of the Exponential (``b = 1``).
    origin: bool = False


def _inv(v: Any) -> Any:
    return 1.0 / v


def _identity(v: Any) -> Any:
    return v


def _neg_exp(v: Any) -> Any:
    return np.exp(-v)


_FAMILIES = {
    "Weibull": _Family(
        True,
        lambda p: (math.log(p[0]), 1.0 / p[1]),
        (("u", np.exp), ("b", _inv)),
    ),
    "LogLogistic": _Family(
        True,
        lambda p: (math.log(p[0]), 1.0 / p[1]),
        (("u", np.exp), ("b", _inv)),
    ),
    "LogNormal": _Family(
        True,
        lambda p: (float(p[0]), float(p[1])),
        (("u", _identity), ("b", _identity)),
    ),
    "Normal": _Family(
        False,
        lambda p: (float(p[0]), float(p[1])),
        (("u", _identity), ("b", _identity)),
    ),
    "Logistic": _Family(
        False,
        lambda p: (float(p[0]), float(p[1])),
        (("u", _identity), ("b", _identity)),
    ),
    "GumbelLEV": _Family(
        False,
        lambda p: (float(p[0]), float(p[1])),
        (("u", _identity), ("b", _identity)),
    ),
    "Gumbel": _Family(
        False,
        lambda p: (float(p[0]), float(p[1])),
        (("u", _identity), ("b", _identity)),
    ),
    "Exponential": _Family(
        True, lambda p: (-math.log(p[0]), 1.0), (("u", _neg_exp),), origin=True
    ),
}

#: The families with pivotal bounds, by name, for the refusals.
PIVOTAL_FAMILIES = tuple(_FAMILIES)


def _refuse(why: str, instead: str) -> ValueError:
    return ValueError(f"Pivotal bounds {why}. {instead}")


def _family(model: Any) -> _Family:
    """The model's family, after the checks that the pivotal argument
    holds for its fit (each refusal says what to use instead)."""
    if model.method != "MPP":
        how = model.method
        if how == "MLE":
            instead = (
                "For a maximum-likelihood fit use method='wald', 'lr' or "
                "'bootstrap'; or refit with how='MPP' for pivotal bounds."
            )
        else:
            instead = (
                "Refit with how='MPP' for pivotal bounds, or with "
                "how='MLE' for the Wald, likelihood-ratio and bootstrap "
                "bounds."
            )
        raise _refuse(
            f"are for probability-plotting (how='MPP') fits; this model "
            f"was {'built' if how == 'given parameters' else 'fitted'} by "
            f"{how}",
            instead,
        )
    name = model.dist.canonical_name
    family = _FAMILIES.get(name)
    if family is None:
        raise _refuse(
            f"are built for the families that are a location and a scale "
            f"on the time or log-time scale, where the rank-regression "
            f"error has a distribution free of the parameters "
            f"({', '.join(PIVOTAL_FAMILIES)}); the {name} is not among "
            f"them",
            "Fit by maximum likelihood (how='MLE') and use method='lr' or "
            "method='bootstrap'.",
        )
    if model.offset:
        raise _refuse(
            "do not hold for an offset (3-parameter) fit: the offset is not "
            "a location or scale of the log-time scale, so the fit's error "
            "depends on the true parameters",
            "Fit by maximum likelihood with offset=True and use "
            "method='bootstrap' or method='lr'.",
        )
    if model.lfp or model.zi or getattr(model, "weighted", False):
        raise _refuse(
            "are for a plain probability-plotting fit",
            "Use an MLE fit with method='bootstrap' or method='lr'.",
        )
    if model._user_fixed_idx():
        raise _refuse(
            "are for a fit with every parameter estimated",
            "Refit without 'fixed'.",
        )
    return family


def _units(model: Any) -> npt.NDArray:
    """The censoring of each unit in time order (0 failed, 1 still
    running), a failure before a suspension at the same time."""
    data = getattr(model, "data", None)
    if data is None:
        raise _refuse(
            "need the data the model was fitted to (their size and which "
            "ranks are suspensions); this model was built from parameters, "
            "or restored from a dict saved without its data",
            "Save it with to_dict(with_data=True) to keep them.",
        )
    c = np.asarray(data["c"])
    if np.any((c != 0) & (c != 1)):
        raise _refuse(
            "need exact failure times and right censoring; this model was "
            "fitted to left- or interval-censored data, whose ranks are not "
            "known",
            "Fit by maximum likelihood (how='MLE') and use method='lr'.",
        )
    t = data.get("t")
    if t is not None and np.isfinite(np.asarray(t, dtype=float)).any():
        raise _refuse(
            "need data without truncation: a truncated sample's ranks do "
            "not have the distribution of a full sample's",
            "Fit by maximum likelihood (how='MLE') and use method='lr' or "
            "method='bootstrap'.",
        )
    x = np.asarray(data["x"], dtype=float)
    n = np.asarray(data["n"], dtype=int)
    order = np.lexsort((c, x))
    return np.repeat(c[order], n[order])


@dataclass
class _Design:
    """The regression a simulated sample is fitted by: the zero-based
    ranks (in the ordered sample of ``size``) whose points enter it, and
    their plotted values on the regression's ``y`` scale."""

    size: int
    ranks: npt.NDArray
    y: npt.NDArray
    rr: str


def _design(model: Any, family: _Family) -> _Design:
    """The fit's regression on a sample of the same size and rank
    pattern, as ``fitters.mpp`` builds it."""
    from surpyval import Exponential
    from surpyval.univariate.nonparametric.plotting_positions import (
        plotting_positions,
    )

    pattern = _units(model)
    info = getattr(model, "fitting_info", None) or {}
    heuristic = info.get("heuristic", "Nelson-Aalen")
    rr = info.get("rr", "y")
    on_d_is_0 = bool(info.get("on_d_is_0", False))
    size = pattern.size
    ranks = np.arange(1.0, size + 1.0)
    x_, _, d, F = plotting_positions(ranks, c=pattern, heuristic=heuristic)
    keep = np.isfinite(x_)
    if not on_d_is_0:
        keep &= d > 0
    keep &= (F != 0) & (F != 1)
    with np.errstate(all="ignore"):
        y = (Exponential if family.origin else model.dist).mpp_y_transform(
            np.asarray(F[keep], dtype=float)
        )
    return _Design(
        size=size,
        ranks=(np.asarray(x_[keep]) - 1).astype(int),
        y=np.asarray(y, dtype=float),
        rr=rr,
    )


@dataclass
class Pivots:
    """The fits of ``n_sim`` standard samples: each one's location
    ``u`` and scale ``b`` (``b = 1`` for the Exponential)."""

    u: npt.NDArray
    b: npt.NDArray
    n_sim: int
    #: The model's parameters the draws were made for.
    point: tuple


def _standard_fits(
    model: Any, family: _Family, design: _Design, n_sim: int, rng: Any
) -> tuple[npt.NDArray, npt.NDArray]:
    """``(u, b)`` of the regression on ``n_sim`` standard samples."""
    tiny = np.finfo(float).tiny
    rows = max(1, _BLOCK // max(design.size, 1))
    us, bs = [], []
    y = design.y
    ybar = y.mean()
    yc = y - ybar
    syy = float(np.sum(yc**2))
    done = 0
    while done < n_sim:
        m = min(rows, n_sim - done)
        U = rng.uniform(size=(m, design.size))
        U.sort(axis=1)
        U = np.clip(U[:, design.ranks], tiny, 1.0 - np.finfo(float).eps)
        with np.errstate(all="ignore"):
            if family.origin:
                # Standard exponential order statistics; the Exponential's
                # line through the origin of -log(1 - F) on t.
                t = -np.log1p(-U)
                sxy = t @ y
                if design.rr == "y":
                    rate = sxy / np.sum(t**2, axis=1)
                else:
                    rate = float(np.sum(y**2)) / sxy
                u = -np.log(rate)
                b = np.ones(m)
            else:
                z = np.asarray(model.dist.mpp_y_transform(U), dtype=float)
                zc = z - z.mean(axis=1, keepdims=True)
                sxy = zc @ yc
                if design.rr == "y":
                    slope = sxy / np.sum(zc**2, axis=1)
                    b = 1.0 / slope
                    u = z.mean(axis=1) - ybar * b
                else:
                    b = sxy / syy
                    u = z.mean(axis=1) - b * ybar
        us.append(u)
        bs.append(b)
        done += m
    return np.concatenate(us), np.concatenate(bs)


def check_n_sim(n_sim: Any) -> int:
    """``n_sim`` as an int of at least 1, else the refusal."""
    if (
        isinstance(n_sim, bool)
        or not isinstance(n_sim, (int, np.integer))
        or n_sim < 1
    ):
        raise ValueError(
            f"'n_sim' must be a whole number of simulations of at least 1 "
            f"(the default, {DEFAULT_N_SIM}, is enough for 95% bounds); "
            f"got {n_sim!r}"
        )
    return int(n_sim)


def pivots(model: Any, n_sim: Any, random_state: Any) -> Pivots:
    """The standard fits behind the model's pivotal bounds: drawn, or
    those kept on the model for this ``n_sim`` and integer
    ``random_state`` (``None`` or a ``Generator`` draws afresh)."""
    family = _family(model)
    n_sim = check_n_sim(n_sim)
    point = tuple(float(p) for p in np.ravel(model.params))
    key = None
    if isinstance(random_state, (int, np.integer)) and not isinstance(
        random_state, bool
    ):
        key = (n_sim, int(random_state))
        kept = getattr(model, "_pivotal_draws", None)
        if kept is not None and key in kept and kept[key].point == point:
            return kept[key]
    design = _design(model, family)
    if design.ranks.size < (1 if family.origin else 2):
        raise _refuse(
            "need at least two plotted failures",
            "Fit by maximum likelihood (how='MLE').",
        )
    u, b = _standard_fits(
        model, family, design, n_sim, as_generator(random_state)
    )
    out = Pivots(u=u, b=b, n_sim=n_sim, point=point)
    if key is not None:
        if getattr(model, "_pivotal_draws", None) is None:
            model._pivotal_draws = {}
        model._pivotal_draws[key] = out
    return out


def _tail(alpha_ci: float, bound: str, n_sim: int) -> float:
    """The tail probability ``a`` of each end of the interval read (the
    one-sided bounds are the ends of the two-sided interval at
    ``1 - 2 alpha_ci``), refused where ``n_sim`` draws cannot give it."""
    a = alpha_ci / 2.0 if bound == "two-sided" else alpha_ci
    if (n_sim + 1) * a < 1:
        need = math.ceil(1.0 / a) - 1
        raise ValueError(
            f"'n_sim' = {n_sim} simulations cannot give a bound with tail "
            f"probability {a:g}: the bound would be beyond the most "
            f"extreme simulated fit. Use n_sim of at least {need} (the "
            f"default, {DEFAULT_N_SIM}, gives tails down to 1e-4)."
        )
    return a


def _ends(lo: npt.NDArray, hi: npt.NDArray, bound: str) -> npt.NDArray:
    """``[lo, hi]`` on a last axis, or the end ``bound`` asks for."""
    if bound == "lower":
        return lo
    if bound == "upper":
        return hi
    return np.stack([lo, hi], axis=-1)


def location_ends(
    u_star: npt.NDArray,
    b_star: npt.NDArray,
    u_hat: float,
    b_hat: float,
    a: float,
) -> npt.NDArray:
    """The ends of the interval on the location ``u`` with tail ``a``
    each side, from the pivot ``T = (u_hat - u) / b_hat``, simulated as
    ``U* / B*``."""
    q = bootstrap_quantile(u_star / b_star, [a, 1.0 - a])
    return np.array([u_hat - b_hat * q[1], u_hat - b_hat * q[0]])


def scale_ends(b_star: npt.NDArray, b_hat: float, a: float) -> npt.NDArray:
    """The ends of the interval on the scale ``b``, from the pivot
    ``b_hat / b``, simulated as ``B*``."""
    q = bootstrap_quantile(b_star, [a, 1.0 - a])
    return np.array([b_hat / q[1], b_hat / q[0]])


def quantile_ends(
    u_star: npt.NDArray,
    b_star: npt.NDArray,
    z: npt.NDArray,
    u_hat: float,
    b_hat: float,
    a: float,
) -> tuple[npt.NDArray, npt.NDArray]:
    """The ends of the intervals on ``y_p = u + b z_p`` at each ``z``,
    from the pivot ``W = (u_hat - y_p) / b_hat``, simulated as
    ``(U* - z_p) / B*``."""
    W = (u_star[:, None] - z[None, :]) / b_star[:, None]
    q = bootstrap_quantile(W, [a, 1.0 - a])
    return u_hat - b_hat * q[1], u_hat - b_hat * q[0]


def standardised_ends(
    u_star: npt.NDArray,
    b_star: npt.NDArray,
    y: npt.NDArray,
    u_hat: float,
    b_hat: float,
    a: float,
) -> tuple[npt.NDArray, npt.NDArray]:
    """The ends of the intervals on ``(y - u) / b`` at each finite ``y``
    (the inversion of :func:`quantile_ends` over ``p``): the percentiles
    of ``U* - v B*`` with ``v = (u_hat - y) / b_hat``."""
    v = (u_hat - y) / b_hat
    D = u_star[:, None] - v[None, :] * b_star[:, None]
    q = bootstrap_quantile(D, [a, 1.0 - a])
    return q[0], q[1]


def _ub_hat(model: Any, family: _Family) -> tuple[float, float]:
    return family.to_ub([float(p) for p in np.ravel(model.params)])


def param_cb_pivotal(
    model: Any,
    name: str,
    alpha_ci: float,
    bound: str,
    n_sim: Any,
    random_state: Any,
) -> npt.NDArray:
    """``param_cb(method="pivotal")``."""
    family = _family(model)
    names = list(model.dist.parameter_names)
    if name not in names:
        raise ValueError(
            f"Unknown parameter {name!r}; expected one of {names}"
        )
    piv = pivots(model, n_sim, random_state)
    a = _tail(alpha_ci, bound, piv.n_sim)
    u_hat, b_hat = _ub_hat(model, family)
    role, f = family.roles[names.index(name)]
    with np.errstate(all="ignore"):
        if role == "u":
            ends = location_ends(piv.u, piv.b, u_hat, b_hat, a)
        else:
            ends = scale_ends(piv.b, b_hat, a)
        values = np.sort(np.asarray(f(ends), dtype=float))
    return np.atleast_1d(_ends(values[0], values[1], bound))


def quantile_cb_pivotal(
    model: Any,
    p: npt.NDArray,
    alpha_ci: float,
    bound: str,
    n_sim: Any,
    random_state: Any,
) -> npt.NDArray:
    """``quantile_cb(method="pivotal")``: ``p`` in (0, 1), checked."""
    family = _family(model)
    piv = pivots(model, n_sim, random_state)
    a = _tail(alpha_ci, bound, piv.n_sim)
    u_hat, b_hat = _ub_hat(model, family)
    p = np.asarray(p, dtype=float)
    with np.errstate(all="ignore"):
        z = _standard_quantile(model, family, p.ravel())
        lo, hi = quantile_ends(piv.u, piv.b, z, u_hat, b_hat, a)
        if family.log_time:
            lo, hi = np.exp(lo), np.exp(hi)
    return _ends(lo.reshape(p.shape), hi.reshape(p.shape), bound)


def _standard_quantile(
    model: Any, family: _Family, p: npt.NDArray
) -> npt.NDArray:
    """``z_p``: the standard variable's ``p`` quantile on the
    transformed scale."""
    if family.origin:
        return np.log(-np.log1p(-p))
    return np.asarray(model.dist.mpp_y_transform(p), dtype=float)


def _standard_model(model: Any, family: _Family) -> Any:
    """The family's standard member, ``u = 0`` and ``b = 1``."""
    if family.origin:
        return model.dist.from_params([1.0])
    params = model.dist.unpack_rr(np.array([1.0, 0.0]), "x")
    return model.dist.from_params([float(v) for v in params])


def cb_pivotal(
    model: Any,
    x: npt.NDArray,
    on: str,
    alpha_ci: float,
    bound: str,
    n_sim: Any,
    random_state: Any,
) -> npt.NDArray:
    """``cb(method="pivotal")`` on ``sf``, ``ff`` or ``Hf``."""
    from surpyval.utils.validation import CB_ON, option_error

    family = _family(model)
    on = {"R": "sf", "F": "ff"}.get(on, on)
    if on in ("hf", "df"):
        raise ValueError(
            f"Pivotal bounds are on sf, ff and Hf (and on the quantiles and "
            f"parameters); on='{on}' is not a monotone function of the "
            "fitted line's error, so it has no pivotal bound. Bound it with "
            "an MLE fit: cb(x, on=..., method='lr') or method='wald'."
        )
    if on not in ("sf", "ff", "Hf"):
        raise option_error("on", on, CB_ON)
    piv = pivots(model, n_sim, random_state)
    a = _tail(alpha_ci, bound, piv.n_sim)
    u_hat, b_hat = _ub_hat(model, family)
    t = np.asarray(x, dtype=float)
    flat = t.ravel()
    z_lo = np.full(flat.shape, np.nan)
    z_hi = np.full(flat.shape, np.nan)
    with np.errstate(all="ignore"):
        y = np.log(flat) if family.log_time else flat
    below = np.isneginf(y) | (family.log_time & (flat <= 0))
    above = np.isposinf(y)
    z_lo[below], z_hi[below] = -np.inf, -np.inf
    z_lo[above], z_hi[above] = np.inf, np.inf
    inside = np.isfinite(y)
    idx = np.flatnonzero(inside)
    rows = max(1, _BLOCK // piv.n_sim)
    for start in range(0, idx.size, rows):
        k = idx[start : start + rows]
        z_lo[k], z_hi[k] = standardised_ends(
            piv.u, piv.b, y[k], u_hat, b_hat, a
        )
    std = _standard_model(model, family)
    t_lo = np.exp(z_lo) if family.log_time else z_lo
    t_hi = np.exp(z_hi) if family.log_time else z_hi
    with np.errstate(all="ignore"):
        if on == "ff":
            lo = np.asarray(std.ff(t_lo), dtype=float)
            hi = np.asarray(std.ff(t_hi), dtype=float)
        elif on == "sf":
            lo = np.asarray(std.sf(t_hi), dtype=float)
            hi = np.asarray(std.sf(t_lo), dtype=float)
        else:
            lo = np.asarray(std.Hf(t_lo), dtype=float)
            hi = np.asarray(std.Hf(t_hi), dtype=float)
    lo = lo.reshape(t.shape)
    hi = hi.reshape(t.shape)
    return _ends(lo, hi, bound)
