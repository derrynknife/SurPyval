"""
The joint frailty model of recurrent events and a terminal event.

Many units have events that repeat -- repairs of a machine, readmissions
of a patient -- until an event that ends the history for good: the
machine is scrapped, the patient dies. Where the units that have many
recurrences also tend to reach the terminal event sooner, the terminal
event is not independent censoring of the recurrent events, and a model
of the recurrent events alone (which treats it as such) is biased. The
joint frailty model (Liu, Wolfe and Huang 2004; Rondeau, Mathoulin-Pelissier,
Jacqmin-Gadda, Brouste and Soubeyran 2007) gives each unit one
unobserved frailty ``u`` that acts on both processes:

.. math::
    r_i(t \\mid u) = u \\, r_0(t) \\, e^{\\beta' Z_i}, \\qquad
    \\lambda_i(t \\mid u) = u^{\\alpha} \\, \\lambda_0(t) \\,
    e^{\\gamma' Z_i},

the recurrent intensity and the terminal hazard. ``theta``, the variance
of ``u``, says how much the units differ; ``alpha`` says how the frailty
acts on the terminal event: ``alpha > 0`` when units with more
recurrences reach it sooner, ``alpha = 0`` when it is independent of
them. Both baselines are Weibull, :math:`R_0(t) = (t / \\eta_r)^{k_r}`
and :math:`\\Lambda_0(t) = (t / \\eta_t)^{k_t}`; the recurrent baseline is
then the power-law (Crow-AMSAA) intensity.

The marginal likelihood integrates the frailty out of each unit's
likelihood by adaptive Gauss-Hermite quadrature over ``w = log u``,
centred on the mode of each unit's integrand and scaled by its curvature
there, for the gamma frailty (mean 1, variance ``theta``) and the
log-normal one (``log u ~ N(0, theta)``).
"""

from __future__ import annotations

import warnings
from typing import Any

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from autograd import grad as _ag_grad
from autograd import hessian as _ag_hessian
from autograd.scipy.special import gammaln, logsumexp
from autograd.tracer import getval
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.special import roots_legendre
from scipy.stats import chi2

from surpyval.recurrent.inference import (
    LikelihoodInferenceMixin,
    check_alpha_ci,
)
from surpyval.recurrent.regression.proportional_intensity import (
    alias_covariates,
)
from surpyval.serialisation import (
    SerialisableMixin,
    dict_numbers,
    require_model_tag,
    stamp_schema,
)
from surpyval.univariate.regression._aliasing import (
    covariate_columns,
    dataframe_covariates,
    fit_columns,
)
from surpyval.utils.fitter import singleton_fitter
from surpyval.utils.fitter_repr import FitterRepr
from surpyval.utils.no_maximum import (
    maximum_entry,
    restored_maximum,
    warn_unverified,
)
from surpyval.utils.recurrent_data import (
    accepts_terminal_surpyval_data,
    notes_text,
)
from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import check_option
from surpyval.utils.warnings import caller_stacklevel

__all__ = ["JointFrailty", "JointFrailtyModel"]

FAMILIES = ("gamma", "lognormal")
TIME_SCALES = ("calendar", "gap")
BOUNDS = ("two-sided", "lower", "upper")
BASELINE_NAMES = [
    "recurrent_scale",
    "recurrent_shape",
    "terminal_scale",
    "terminal_shape",
]
#: The number of nodes of the frailty integral, a trapezoid rule in
#: ``w = log u`` over each unit's own range: from its integrand's mode out
#: to where the integrand has fallen by a factor ``e^-_DROP`` on either
#: side. The gamma frailty's integrand has an exponential (not Gaussian)
#: left tail in ``w`` -- ``u^(D + 1/theta)`` near 0 -- which Gauss-Hermite
#: quadrature, even adaptive, follows badly (an error of 4e-4 in the
#: log-integral at theta = 2 with no events); the trapezoid rule on a
#: smooth integrand that has died away at both ends converges
#: geometrically. Against scipy's adaptive quadrature the log-integral is
#: within 1e-10 (``test_joint_frailty.py``).
N_NODES = 64
_DROP = 36.0
#: Gauss-Legendre nodes of the integral over time in ``mcf``.
N_TIME_NODES = 64
_EXP_CAP = 700.0


# ---------------------------------------------------------------------------
# The frailty integral
# ---------------------------------------------------------------------------
def _expm1_minus(w: Any) -> Any:
    """``exp(w) - 1 - w`` without cancellation near 0."""
    wv = getval(w)
    if np.all(np.abs(wv) >= 0.05):
        return anp.expm1(w) - w
    series = w**2 * (
        0.5
        + w
        * (
            1.0 / 6
            + w * (1.0 / 24 + w * (1.0 / 120 + w * (1.0 / 720 + w / 5040)))
        )
    )
    small = np.abs(wv) < 0.05
    return anp.where(small, series, anp.expm1(w) - w)


def _gamma_constant(k: Any) -> Any:
    """``log Gamma(k) - k log k + k``, the normalising constant of the
    gamma frailty's log-density in ``w = log u`` (shape and rate ``k =
    1 / theta``), by its Stirling series where ``k`` is large, where the
    direct form loses its digits to cancellation."""
    if float(getval(k)) < 30.0:
        return gammaln(k) - k * anp.log(k) + k
    inv = 1.0 / k
    inv2 = inv * inv
    return (
        0.5 * np.log(2 * np.pi)
        - 0.5 * anp.log(k)
        + inv
        * (
            1.0 / 12
            - inv2 * (1.0 / 360 - inv2 * (1.0 / 1260 - inv2 * (1.0 / 1680)))
        )
    )


def _log_density(w: Any, theta: Any, family: str) -> Any:
    """The log-density of ``w = log u``."""
    if family == "gamma":
        k = 1.0 / theta
        return -k * _expm1_minus(w) - _gamma_constant(k)
    return -(w**2) / (2.0 * theta) - 0.5 * anp.log(2.0 * np.pi * theta)


def _mode(
    D: npt.NDArray,
    A: npt.NDArray,
    B: npt.NDArray,
    alpha: float,
    theta: float,
    family: str,
) -> tuple[npt.NDArray, npt.NDArray]:
    """The mode ``w*`` of ``D w - A e^w - B e^{alpha w} + log g(w)`` and
    its curvature ``-h''(w*)`` (the function is concave), by safeguarded
    Newton steps."""
    if family == "gamma":
        k = 1.0 / theta
        w = np.log((D + k) / (A + k))
    else:
        k = 0.0
        w = np.zeros_like(D)
    for _ in range(200):
        ew = np.exp(np.clip(w, -_EXP_CAP, _EXP_CAP))
        eaw = np.exp(np.clip(alpha * w, -_EXP_CAP, _EXP_CAP))
        if family == "gamma":
            g1 = D - A * ew - alpha * B * eaw + k * (1.0 - ew)
            g2 = -(A + k) * ew - alpha**2 * B * eaw
        else:
            g1 = D - A * ew - alpha * B * eaw - w / theta
            g2 = -A * ew - alpha**2 * B * eaw - 1.0 / theta
        step = np.clip(-g1 / g2, -2.0, 2.0)
        w = w + step
        if np.all(np.abs(step) < 1e-12 * (1.0 + np.abs(w))):
            break
    ew = np.exp(np.clip(w, -_EXP_CAP, _EXP_CAP))
    eaw = np.exp(np.clip(alpha * w, -_EXP_CAP, _EXP_CAP))
    if family == "gamma":
        curv = (A + k) * ew + alpha**2 * B * eaw
    else:
        curv = A * ew + alpha**2 * B * eaw + 1.0 / theta
    return w, curv


def log_frailty_integral(
    D: Any,
    A: Any,
    B: Any,
    alpha: Any,
    theta: Any,
    family: str = "gamma",
    n_nodes: int = N_NODES,
) -> Any:
    """
    The frailty integral of the joint frailty model, elementwise,

    .. math::
        \\log \\int_0^\\infty u^D e^{-A u - B u^\\alpha} \\, dG_\\theta(u),

    with ``G`` the gamma (mean 1, variance ``theta``) or log-normal
    (``log u ~ N(0, theta)``) frailty distribution, by an ``n_nodes``-point
    trapezoid rule in ``w = log u`` over the range where the integrand
    matters: from its mode (found by Newton's method; the log-integrand is
    concave) out to where it has fallen by ``e^-36`` on either side. The
    mode and the range only place the nodes, so they are not
    differentiated: autograd's derivatives are those of the rule with the
    nodes held, which is a valid quadrature wherever they are.
    ``theta <= 0`` is the model without frailty, ``u = 1``:
    ``-A - B``.

    Parameters
    ----------
    D, A, B : array_like
        Broadcast together: a unit's count of events (``n + alpha delta``,
        its recurrent events and ``alpha`` times its terminal flag), its
        cumulative recurrent intensity at ``u = 1`` and its cumulative
        terminal hazard at ``u = 1``.
    alpha : float
        The power of the frailty in the terminal hazard.
    theta : float
        The variance of the frailty (gamma) or of its logarithm
        (log-normal).
    family : {'gamma', 'lognormal'}, optional
        The frailty distribution. Default ``'gamma'``.
    n_nodes : int, optional
        The least number of nodes. Default 64; more are taken where a
        unit's integrand is wide, for steps of at most 0.3 in ``w``.

    Returns
    -------
    numpy.ndarray
        The log-integral, of the broadcast shape.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.recurrent.joint_frailty import log_frailty_integral
    >>> # A gamma frailty with alpha = 1 has a closed form:
    >>> # log E[u^D e^{-(A + B) u}] for theta = 0.5, D = 2, A + B = 1.5
    >>> value = log_frailty_integral(2.0, 1.0, 0.5, 1.0, 0.5)
    >>> from scipy.special import gammaln
    >>> k = 2.0
    >>> exact = (gammaln(k + 2) - gammaln(k) + k * np.log(k)
    ...          - (k + 2) * np.log(k + 1.5))
    >>> bool(abs(value - exact) < 1e-12)
    True
    """
    Dv, Av, Bv = np.broadcast_arrays(
        np.asarray(getval(D), dtype=float),
        np.asarray(getval(A), dtype=float),
        np.asarray(getval(B), dtype=float),
    )
    missing = ~(np.isfinite(Dv) & np.isfinite(Av) & np.isfinite(Bv))
    if (
        np.any(missing)
        and not np.all(missing)
        and Dv.shape == np.shape(D)
        and Dv.shape == np.shape(A)
        and Dv.shape == np.shape(B)
    ):
        # A missing value (a nan covariate in a query) gives nan there
        # only: the others are computed on their own, so one row's nan
        # does not place the nodes of the rest.
        out = np.full(Dv.shape, np.nan)
        keep = ~missing
        out[keep] = log_frailty_integral(
            Dv[keep],
            Av[keep],
            Bv[keep],
            getval(alpha),
            getval(theta),
            family,
            n_nodes,
        )
        return out
    tv = np.float64(getval(theta))
    av = np.float64(getval(alpha))
    if not (np.isfinite(tv) and np.isfinite(av)):
        return 0.0 * D + 0.0 * A + 0.0 * B + np.nan
    if tv <= 0.0:
        return -1.0 * A - 1.0 * B + 0.0 * D
    shape = Dv.shape
    Dr, Ar, Br = Dv.ravel(), Av.ravel(), Bv.ravel()
    with np.errstate(all="ignore"):
        m, curv = _mode(Dr, Ar, Br, av, tv, family)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = 1.0 / np.sqrt(curv)
    if not (np.all(np.isfinite(m)) and np.all(np.isfinite(s) & (s > 0))):
        # Parameters so extreme (an optimiser's trial step) that the
        # integrand overflows: no value, which a search steps back from.
        return 0.0 * D + 0.0 * A + 0.0 * B + np.nan
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        top = _h_values(m, Dr, Ar, Br, av, tv, family)
        ends = []
        for side in (-1.0, 1.0):
            t = np.ones_like(m)
            for _ in range(14):
                drop = top - _h_values(
                    m + side * t * s, Dr, Ar, Br, av, tv, family
                )
                short = ~(drop > _DROP)
                if not np.any(short):
                    break
                t = np.where(short, 2.0 * t, t)
            ends.append(t)
    left, right = ends
    # The trapezoid rule's error falls as exp(-2 pi d / step), with d the
    # half-width of the strip about the real line where the integrand is
    # analytic and bounded: pi / 2 for the e^w term, pi / (2 |alpha|) for
    # the e^(alpha w) one. Steps of 0.3 / max(1, |alpha|) keep it below
    # 1e-14, and a quarter of the integrand's scale at its mode resolves
    # its peak where the frailty is narrow.
    h_max = np.minimum(0.3 / max(1.0, abs(av)), 0.25 * s)
    want = (left + right) * s / h_max
    if not np.all(np.isfinite(want)):
        return 0.0 * D + 0.0 * A + 0.0 * B + np.nan
    n_nodes = int(np.clip(np.ceil(np.max(want, initial=1.0)), n_nodes, 4000))
    grid = np.linspace(-1.0, 1.0, n_nodes)
    half = 0.5 * (left + right) * s
    centre = m + 0.5 * (right - left) * s
    nodes = centre[:, None] + half[:, None] * grid
    step = half * (grid[1] - grid[0])
    nodes = nodes.reshape(shape + (n_nodes,))
    log_step = np.log(step).reshape(shape)
    Dx = anp.expand_dims(D + 0.0 * Dv, -1)
    Ax = anp.expand_dims(A + 0.0 * Av, -1)
    Bx = anp.expand_dims(B + 0.0 * Bv, -1)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        e_w = np.exp(np.clip(nodes, -_EXP_CAP, _EXP_CAP))
        e_aw = anp.exp(anp.clip(alpha * nodes, -_EXP_CAP, _EXP_CAP))
        h = Dx * nodes - Ax * e_w - Bx * e_aw
        h = h + _log_density(nodes, theta, family)
        return log_step + logsumexp(h, axis=-1)


def _h_values(
    w: npt.NDArray,
    D: npt.NDArray,
    A: npt.NDArray,
    B: npt.NDArray,
    alpha: float,
    theta: float,
    family: str,
) -> npt.NDArray:
    """The log-integrand ``D w - A e^w - B e^{alpha w} + log g(w)``."""
    e_w = np.exp(np.clip(w, -_EXP_CAP, _EXP_CAP))
    e_aw = np.exp(np.clip(alpha * w, -_EXP_CAP, _EXP_CAP))
    return D * w - A * e_w - B * e_aw + _log_density(w, theta, family)


# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------
class _Units:
    """Each unit's history, as the likelihood reads it."""

    def __init__(
        self,
        ids: npt.NDArray,
        entry: npt.NDArray,
        end: npt.NDArray,
        delta: npt.NDArray,
        Z: npt.NDArray,
        event_times: npt.NDArray,
        event_unit: npt.NDArray,
        time_scale: str,
    ) -> None:
        self.ids = ids
        self.entry = entry
        self.end = end
        self.delta = delta
        self.Z = Z
        self.event_times = event_times
        self.event_unit = event_unit
        self.time_scale = time_scale
        m = ids.size
        self.n_events = np.bincount(event_unit, minlength=m).astype(float)
        if time_scale == "gap":
            # the gaps of each unit, then its open gap to the end
            starts = np.zeros_like(event_times)
            same = np.r_[False, event_unit[1:] == event_unit[:-1]]
            starts[same] = event_times[:-1][same[1:]]
            gaps = event_times - starts
            last = np.zeros(m)
            has = self.n_events > 0
            last_idx = np.cumsum(self.n_events).astype(int) - 1
            last[has] = event_times[last_idx[has]]
            self.gaps = gaps
            self.open_gap = end - last
            order_units = np.r_[event_unit, np.arange(m)]
            pieces = np.r_[gaps, self.open_gap]
            order = np.argsort(order_units, kind="stable")
            self.pieces = pieces[order]
            counts = np.bincount(order_units, minlength=m)
            self.piece_end = np.cumsum(counts) - 1
            self.piece_start = self.piece_end - counts + 1

    @property
    def n_units(self) -> int:
        return int(self.ids.size)


def _rows_of(arg: str, values: Any, n: int) -> npt.NDArray:
    out = np.asarray(values)
    if out.ndim == 0:
        out = np.full(n, out.item())
    if out.shape[0] != n:
        raise ValueError(
            f"'{arg}' must have one value per row of x ({n}); got "
            f"{out.shape[0]}."
        )
    return out


def _checked_rows(
    x: Any, i: Any, c: Any, terminal: Any
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """``x``, ``i`` and ``c`` as arrays, checked."""
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    if n == 0:
        raise ValueError(
            "'x' is empty: give each unit's event times and "
            "the end of its observation."
        )
    if i is None:
        raise ValueError(
            "'i' is needed: the unit each row belongs to. The joint frailty "
            "model compares units, so give each unit's rows its id."
        )
    if c is None:
        raise ValueError(
            "'c' is needed: 0 on a recurrent event, 1 on the row that "
            "closes a unit's observation (its end, or its terminal event)."
        )
    if terminal is None:
        raise ValueError(
            "'terminal' is needed: 1 on the row that closes a unit's "
            "observation where the terminal event closed it, 0 where the "
            "observation simply ended (or a dict {unit: 0 or 1})."
        )
    i = _rows_of("i", i, n)
    c = _rows_of("c", c, n)
    if np.any(np.isnan(x)):
        raise ValueError(
            "'x' has missing values at rows "
            f"{np.flatnonzero(np.isnan(x)).tolist()}: each row is part of a "
            "unit's history, so a missing time cannot be dropped; remove "
            "the unit or give the time."
        )
    try:
        c = c.astype(float)
    except (TypeError, ValueError):
        raise ValueError(
            "'c' must be 0 (an event) or 1 (the end of a "
            "unit's observation)."
        ) from None
    bad = ~np.isin(c, (0.0, 1.0))
    if np.any(bad):
        raise ValueError(
            "'c' must be 0 (a recurrent event) or 1 (the end of a unit's "
            f"observation); rows {np.flatnonzero(bad).tolist()} are not. "
            "Left- and interval-censored counts are not modelled by the "
            "joint frailty model: give the exact event times."
        )
    c = c.astype(int)
    if not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise ValueError(
            "'x' must be finite and positive: the ages of the events and "
            "of the end of each unit's observation, from its origin at 0."
        )
    return x, i, c


def _ends(
    x: npt.NDArray,
    c: npt.NDArray,
    terminal: Any,
    ids: npt.NDArray,
    inv: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Each unit's end of observation and terminal flag."""
    n, m = x.size, ids.size
    if isinstance(terminal, dict):
        missing = [u for u in ids.tolist() if u not in terminal]
        if missing:
            raise ValueError(
                f"'terminal' has no entry for units {missing}: give 0 or 1 "
                "for every unit."
            )
        delta = np.array([terminal[u] for u in ids.tolist()], dtype=float)
        term_rows = None
    else:
        term_rows = _rows_of("terminal", terminal, n).astype(float)
        delta = None
    ends = np.flatnonzero(c == 1)
    count_end = np.bincount(inv[ends], minlength=m)
    if np.any(count_end != 1):
        wrong = ids[count_end != 1].tolist()
        raise ValueError(
            f"Units {wrong} do not have exactly one row with c=1: each unit "
            "needs one row closing its observation, at the age it ended "
            "(with terminal=1 where the terminal event ended it)."
        )
    end = np.zeros(m)
    end[inv[ends]] = x[ends]
    if term_rows is not None:
        on_events = (c == 0) & (term_rows != 0)
        if np.any(on_events):
            raise ValueError(
                "'terminal' is 1 on recurrent-event rows (c=0) "
                f"{np.flatnonzero(on_events).tolist()}: the terminal event "
                "goes on the row that closes the unit's observation (c=1). "
                "Where a recurrent event also ended the unit, give both "
                "rows at that age."
            )
        delta = np.zeros(m)
        delta[inv[ends]] = term_rows[ends]
    assert delta is not None
    if not np.all(np.isin(delta, (0.0, 1.0))):
        raise ValueError("'terminal' must be 0 or 1.")
    return end, delta


def _entries(
    tl: Any,
    x: npt.NDArray,
    c: npt.NDArray,
    inv: npt.NDArray,
    end: npt.NDArray,
    time_scale: str,
) -> npt.NDArray:
    """Each unit's age at entry, checked."""
    n, m = x.size, end.size
    events = np.flatnonzero(c == 0)
    if tl is None:
        entry = np.zeros(m)
    else:
        tl_arr = np.asarray(tl, dtype=float)
        if tl_arr.ndim == 0:
            entry = np.full(m, float(tl_arr))
        elif tl_arr.size == n:
            entry = np.zeros(m)
            entry[inv] = tl_arr
            if np.any(entry[inv] != tl_arr):
                raise ValueError(
                    "'tl' must be the same on every row of a unit: its age "
                    "when its observation began."
                )
        elif tl_arr.size == m:
            entry = tl_arr.reshape(-1)
        else:
            raise ValueError(
                f"'tl' must be a scalar, one value per row ({n}) or one per "
                f"unit ({m}); got {tl_arr.size}."
            )
        if not np.all(np.isfinite(entry)) or np.any(entry < 0):
            raise ValueError("'tl' must be finite and at least 0.")
        if time_scale == "gap" and np.any(entry > 0):
            raise ValueError(
                "Left truncation ('tl') is not available with "
                "time_scale='gap': the time since the last event before "
                "entry is not known. Use time_scale='calendar'."
            )
        early = x[events] <= entry[inv[events]]
        if np.any(early) or np.any(end <= entry):
            raise ValueError(
                "Every event and end of observation must come after the "
                "unit's entry 'tl'."
            )
    return entry


def _unit_covariates(Z: Any, inv: npt.NDArray, m: int) -> npt.NDArray:
    """One covariate row per unit, checked."""
    n = inv.size
    if Z is None:
        Zr = np.zeros((n, 0))
    else:
        Zr = np.asarray(Z, dtype=float)
        if Zr.ndim == 1:
            Zr = Zr[:, None]
        if Zr.shape[0] != n:
            raise ValueError(
                f"'Z' must have one row per row of x ({n}); got "
                f"{Zr.shape[0]}."
            )
    p = Zr.shape[1]
    if np.any(np.isnan(Zr)):
        rows = np.flatnonzero(np.isnan(Zr).any(axis=1)).tolist()
        raise ValueError(
            f"'Z' has missing values at rows {rows}: a covariate describes "
            "the whole unit, so its rows cannot be dropped one by one; "
            "remove the unit or give the value."
        )
    if not np.all(np.isfinite(Zr)):
        raise ValueError("'Z' must be finite.")
    Zu = np.zeros((m, p))
    Zu[inv] = Zr
    if np.any(Zu[inv] != Zr):
        raise ValueError(
            "'Z' must be the same on every row of a unit: the joint frailty "
            "model's covariates describe the unit (they are fixed in time)."
        )
    return Zu


def _prepare(
    x: Any,
    Z: Any,
    i: Any,
    c: Any,
    terminal: Any,
    tl: Any,
    time_scale: str,
) -> tuple[_Units, int]:
    """Check the rows and gather each unit's history, with the number of
    covariate columns."""
    x, i, c = _checked_rows(x, i, c, terminal)
    ids, inv = np.unique(i, return_inverse=True)
    m = ids.size
    end, delta = _ends(x, c, terminal, ids, inv)
    events = np.flatnonzero(c == 0)
    late = x[events] > end[inv[events]]
    if np.any(late):
        raise ValueError(
            f"Rows {events[late].tolist()} are events after the end of their "
            "unit's observation (its row with c=1): the end must be at or "
            "after every event of the unit."
        )
    entry = _entries(tl, x, c, inv, end, time_scale)
    Zu = _unit_covariates(Z, inv, m)
    p = Zu.shape[1]
    order = np.lexsort((x[events], inv[events]))
    event_times = x[events][order]
    event_unit = inv[events][order]
    if time_scale == "gap":
        same = np.r_[False, event_unit[1:] == event_unit[:-1]]
        tied = same & (np.r_[0.0, np.diff(event_times)] == 0)
        if np.any(tied):
            raise ValueError(
                "Two events of one unit at the same age have a gap of 0, "
                "which a Weibull gap-time baseline cannot take: use "
                "time_scale='calendar', or separate the ages."
            )
    units = _Units(
        ids, entry, end, delta, Zu, event_times, event_unit, time_scale
    )
    if units.n_events.sum() == 0:
        raise ValueError(
            "There are no recurrent events (rows with c=0), so the "
            "recurrent part has no maximum: the joint frailty model needs "
            "both kinds of event."
        )
    if delta.sum() == 0:
        raise ValueError(
            "No unit had the terminal event (terminal=1), so the terminal "
            "part has no maximum: fit the recurrent events alone "
            "(ProportionalIntensityNHPP with the CrowAMSAA baseline), or "
            "check the 'terminal' flags."
        )
    return units, p


# ---------------------------------------------------------------------------
# The likelihood
# ---------------------------------------------------------------------------
def _power(t: npt.NDArray, scale: Any, shape: Any) -> Any:
    """``(t / scale) ** shape``, 0 at ``t = 0`` with finite derivatives
    there (``0 ** shape`` has a ``nan`` derivative in ``shape``)."""
    t = np.asarray(t, dtype=float)
    positive = t > 0
    if np.all(positive):
        return (t / scale) ** shape
    safe = np.where(positive, t, 1.0)
    return anp.where(positive, (safe / scale) ** shape, 0.0)


def _window(
    start: npt.NDArray, end: npt.NDArray, scale: Any, shape: Any
) -> Any:
    """``(end / scale) ** shape - (start / scale) ** shape`` without the
    cancellation of the difference where ``start`` is near ``end`` or the
    shape is small: ``(start / scale) ** shape * expm1(shape *
    log(end / start))``."""
    start = np.asarray(start, dtype=float)
    positive = start > 0
    if not np.any(positive):
        return (end / scale) ** shape
    safe = np.where(positive, start, end)
    late = _power(safe, scale, shape) * anp.expm1(shape * np.log(end / safe))
    return anp.where(positive, late, (end / scale) ** shape)


class _Likelihood:
    """The negative log marginal likelihood of the natural parameters
    ``[recurrent_scale, recurrent_shape, terminal_scale, terminal_shape,
    theta, alpha, *beta, *gamma]`` of the data ``units`` with covariates
    ``Z`` (one row per unit), differentiable by autograd. A class rather
    than a closure, so that a fitted model pickles."""

    def __init__(self, units: _Units, Z: npt.NDArray, family: str) -> None:
        self.units = units
        self.Z = np.asarray(Z, dtype=float)
        self.family = family

    def log_terms(self, params: Any) -> tuple[Any, Any, Any, Any, Any]:
        """``(constant, D, A, B)`` per unit, and ``C``, the terminal
        cumulative hazard at entry."""
        d = self.units
        p = self.Z.shape[1]
        rs, rk, ts, tk, theta, alpha = (params[j] for j in range(6))
        beta = params[6 : 6 + p]
        gamma = params[6 + p : 6 + 2 * p]
        eta_r = anp.dot(self.Z, beta) if p else 0.0 * d.end
        eta_t = anp.dot(self.Z, gamma) if p else 0.0 * d.end
        if d.time_scale == "calendar":
            at = d.event_times
            R = _window(d.entry, d.end, rs, rk)
        else:
            at = d.gaps
            cs = anp.cumsum(_power(d.pieces, rs, rk))
            prior = anp.concatenate([anp.zeros(1), cs])
            R = prior[d.piece_end + 1] - prior[d.piece_start]
        log_r0 = (
            anp.log(rk)
            - anp.log(rs)
            + (rk - 1.0) * (anp.log(at) - anp.log(rs))
        )
        log_lam_end = (
            anp.log(tk)
            - anp.log(ts)
            + (tk - 1.0) * (anp.log(d.end) - anp.log(ts))
        )
        H_end = (d.end / ts) ** tk
        H_entry = _power(d.entry, ts, tk)
        const = (
            anp.sum(d.n_events * eta_r)
            + anp.sum(log_r0)
            + anp.sum(d.delta * (eta_t + log_lam_end))
        )
        D = d.n_events + alpha * d.delta
        A = anp.exp(eta_r) * R
        # The terminal hazard from age 0: the units are those that had not
        # reached the terminal event by their entry, which the likelihood
        # divides out (``__call__``) -- given the frailty, the ratio is
        # exp(-u^alpha (Lambda_0(X) - Lambda_0(entry))).
        B = anp.exp(eta_t) * H_end
        C = anp.exp(eta_t) * H_entry
        return const, D, A, B, C

    def __call__(self, params: Any) -> Any:
        theta, alpha = params[4], params[5]
        const, D, A, B, C = self.log_terms(params)
        ll = const + anp.sum(
            log_frailty_integral(D, A, B, alpha, theta, self.family)
        )
        entered = self.units.entry > 0
        if np.any(entered):
            ll = ll - anp.sum(
                log_frailty_integral(
                    0.0 * C[entered],
                    0.0 * C[entered],
                    C[entered],
                    alpha,
                    theta,
                    self.family,
                )
            )
        return -ll

    def posterior_frailty(self, params: npt.NDArray) -> npt.NDArray:
        """Each unit's posterior mean frailty ``E[u | its history]``."""
        theta, alpha = float(params[4]), float(params[5])
        _, D, A, B, _ = self.log_terms(params)
        if theta <= 0:
            return np.ones(self.units.n_units)
        return np.exp(
            log_frailty_integral(D + 1.0, A, B, alpha, theta, self.family)
            - log_frailty_integral(D, A, B, alpha, theta, self.family)
        )


class _FloatLikelihood:
    """The negative log-likelihood as a float of a plain array: what the
    likelihood-ratio searches call (a class, so a model pickles after a
    likelihood-ratio bound)."""

    def __init__(self, lik: _Likelihood) -> None:
        self.lik = lik

    def __call__(self, params: Any) -> float:
        return float(self.lik(np.asarray(params, dtype=float)))


class _Search:
    """The negative log-likelihood on the search scale: logs of the
    positive parameters, the coefficients of the standardised columns;
    ``fixed`` holds entries of the natural vector at given values."""

    def __init__(
        self,
        lik: _Likelihood,
        p: int,
        fixed: dict[int, float] | None = None,
    ) -> None:
        self.lik = lik
        self.p = p
        self.fixed = dict(fixed or {})
        self.free = [j for j in range(6 + 2 * p) if j not in self.fixed]

    def natural(self, u: Any) -> Any:
        out = []
        k = 0
        for j in range(6 + 2 * self.p):
            if j in self.fixed:
                out.append(self.fixed[j] + 0.0 * u[0])
                continue
            out.append(anp.exp(u[k]) if j < 5 else u[k] * 1.0)
            k += 1
        return anp.array(out)

    def search(self, natural: npt.NDArray) -> npt.NDArray:
        nat = np.asarray(natural, dtype=float)
        return np.array(
            [np.log(nat[j]) if j < 5 else nat[j] for j in self.free]
        )

    def __call__(self, u: Any) -> Any:
        with np.errstate(all="ignore"):
            value = self.lik(self.natural(u))
        return value


# ---------------------------------------------------------------------------
# The fitter
# ---------------------------------------------------------------------------
def _standardise(Z: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """The columns' means and standard deviations (1 where constant)."""
    if Z.shape[1] == 0:
        return np.zeros(0), np.ones(0)
    mean = Z.mean(axis=0)
    sd = Z.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return mean, sd


def _to_natural(
    std_params: npt.NDArray, mean: npt.NDArray, sd: npt.NDArray
) -> npt.NDArray:
    """Parameters of the standardised columns as those of the original
    columns: the coefficients divided by the scales, the baselines' scales
    moved by the means."""
    out = np.array(std_params, dtype=float)
    p = mean.size
    beta = out[6 : 6 + p] / sd
    gamma = out[6 + p :] / sd
    out[6 : 6 + p] = beta
    out[6 + p :] = gamma
    out[0] = out[0] * np.exp(float(mean @ beta) / out[1])
    out[2] = out[2] * np.exp(float(mean @ gamma) / out[3])
    return out


def _to_standard(
    params: npt.NDArray, mean: npt.NDArray, sd: npt.NDArray
) -> npt.NDArray:
    """The inverse of :func:`_to_natural`."""
    out = np.array(params, dtype=float)
    p = mean.size
    beta = out[6 : 6 + p]
    gamma = out[6 + p :]
    out[0] = out[0] * np.exp(-float(mean @ beta) / out[1])
    out[2] = out[2] * np.exp(-float(mean @ gamma) / out[3])
    out[6 : 6 + p] = beta * sd
    out[6 + p :] = gamma * sd
    return out


def _minimise(search: _Search, start: npt.NDArray) -> Any:
    """BFGS on autograd's gradient, finite answers only."""
    jac = _ag_grad(search)

    def fun(u: npt.NDArray) -> float:
        value = float(search(u))
        return value if np.isfinite(value) else 1e300

    def gradient(u: npt.NDArray) -> npt.NDArray:
        with np.errstate(all="ignore"):
            g = np.asarray(jac(u), dtype=float)
        return np.where(np.isfinite(g), g, 0.0)

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = minimize(
            fun,
            start,
            jac=gradient,
            method="BFGS",
            options={"gtol": 1e-8, "maxiter": 2000},
        )
        if not np.isfinite(res.fun) or res.fun >= 1e300:
            return res
        # A Newton polish on autograd's Hessian: BFGS stops on its
        # gradient tolerance a little short of the maximum.
        hess = _ag_hessian(search)
        x = np.asarray(res.x, dtype=float)
        f = res.fun
        for _ in range(20):
            g = gradient(x)
            H = np.asarray(hess(x), dtype=float)
            if not np.all(np.isfinite(H)):
                break
            try:
                if np.any(np.linalg.eigvalsh(H) <= 0):
                    break
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            t = 1.0
            while t > 1e-4:
                trial = x - t * step
                ft = fun(trial)
                if ft <= f:
                    break
                t /= 2
            else:
                break
            if (
                f - ft < 1e-13 * max(1.0, abs(f))
                and np.max(np.abs(t * step)) < 1e-10
            ):
                x, f = trial, ft
                break
            x, f = trial, ft
        res.x, res.fun = x, f
    return res


def _is_maximum(search: _Search, u: npt.NDArray, n_obs: float) -> bool:
    """A zero gradient (relative to the number of events) and a
    positive-definite Hessian of the negative log-likelihood."""
    with np.errstate(all="ignore"):
        g = np.asarray(_ag_grad(search)(u), dtype=float)
        H = np.asarray(_ag_hessian(search)(u), dtype=float)
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(H))):
        return False
    try:
        if np.any(np.linalg.eigvalsh(0.5 * (H + H.T)) <= 0):
            return False
        # The Newton decrement: the log-likelihood still to gain
        decrement = float(g @ np.linalg.solve(H, g))
    except np.linalg.LinAlgError:
        return False
    return decrement < 1e-6 * max(1.0, 1e-3 * n_obs)


def _boundary_p(statistic: float) -> float:
    """The p-value of the likelihood-ratio statistic of no frailty: a
    50:50 mixture of chi-square on one and on two degrees of freedom.

    Near ``theta = 0`` the frailty enters the likelihood through the
    variance of the recurrent part's random effect, ``theta``, and that of
    the terminal part's, ``alpha^2 theta``, which the free ``alpha`` lets
    differ: two variances on the edge of their space, not one. The 50:50
    mixture of a point mass at 0 and chi-square(1) (Self and Liang 1987),
    the reference of a single variance, rejected a true null 18% of the
    time at the 5% level in the calibration study; this mixture is that
    of a variance on its edge tested together with a second parameter
    (Self and Liang 1987, case 7; Stram and Lee 1994)."""
    if not np.isfinite(statistic):
        return float("nan")
    if statistic <= 1e-9:
        return 1.0
    return float(0.5 * chi2.sf(statistic, 1) + 0.5 * chi2.sf(statistic, 2))


@singleton_fitter
class JointFrailty(FitterRepr):
    """
    The joint frailty model of recurrent events and a terminal event
    (Liu, Wolfe and Huang 2004; Rondeau et al. 2007), with Weibull
    baselines for both. ``JointFrailty`` is an instance of this class;
    its ``fit`` returns a :class:`JointFrailtyModel`.

    In plain words: each unit (a machine, a patient) has repeated events
    until its history is ended by a terminal event (scrapping, death) or
    by the end of follow-up. Every unit carries an unobserved frailty
    ``u`` that raises its rate of recurrent events by the factor ``u`` and
    its hazard of the terminal event by ``u ** alpha``. ``theta``, the
    frailty's variance, measures how much units differ beyond their
    covariates; ``alpha`` measures how the terminal event depends on that
    difference: positive when the units with many recurrences reach the
    terminal event sooner. With the terminal event treated as ordinary
    censoring that dependence biases the recurrent-event rate, which this
    model corrects.

    .. math::
        r_i(t \\mid u) = u \\, r_0(t) \\, e^{\\beta' Z_i}, \\qquad
        \\lambda_i(t \\mid u) = u^{\\alpha} \\lambda_0(t) \\,
        e^{\\gamma' Z_i},

    with :math:`R_0(t) = (t / \\eta_r)^{k_r}` (the power-law intensity)
    and :math:`\\Lambda_0(t) = (t / \\eta_t)^{k_t}`.

    **How the data are given.** The rows of each unit (named by ``i``) are
    its recurrent events (``c = 0``, at the age they happened) and one row
    that closes its observation (``c = 1``, at the age it ended). The
    ``terminal`` flag is 1 on that closing row when the terminal event
    closed it, and 0 when observation simply stopped (the end of the
    study, a unit lost to follow-up). Covariates ``Z`` describe the unit
    and are the same on all its rows; they enter both parts, with their
    own coefficients. ``tl`` is a unit's age when its observation began
    (delayed entry), on the calendar time scale.

    **From a SurpyvalData.** ``fit(data)`` takes a ``SurpyvalData`` with
    units ``i``, in which the terminal event is an event row with the
    terminal flag ``a = 1``: a unit's rows are its recurrent events
    (``c = 0``, ``a = 0``), then its terminal event (``c = 0``, ``a = 1``)
    or a row ``c = 1`` at the end of its record without one. They are
    read as the arrays above (the terminal event becomes the closing row
    with ``terminal = 1``), so the fit is the same. An event on a unit's
    last row with no ``a`` is read as its terminal event, and the
    model's ``notes`` say so: give ``a = 0`` on a last event that was an
    ordinary one. ``fit_from_df(..., a_col=...)`` reads such a table.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.recurrent import JointFrailty
    >>> # Draw 150 units from a joint frailty model and fit it back.
    >>> truth = JointFrailty.from_params(
    ...     recurrent=(10.0, 1.2), terminal=(20.0, 1.5),
    ...     theta=0.5, alpha=1.0, beta=[0.5], gamma=[0.3],
    ... )
    >>> rows = truth.random(150, end=15.0, Z=np.tile([[0.0], [1.0]], (75, 1)),
    ...                     random_state=1)
    >>> model = JointFrailty.fit(**rows)
    >>> model.parameter_names[4:6]
    ['theta', 'alpha']
    >>> round(model.theta, 2), round(model.alpha, 2)
    (0.54, 1.68)
    """

    fitter_kind = "joint frailty fitter (recurrent and terminal events)"

    def _repr_details(self) -> list[str]:
        return ["Weibull baselines"]

    @dataframe_covariates
    @takes_unit
    @accepts_terminal_surpyval_data
    def fit(
        self,
        x: ArrayLike,
        Z: ArrayLike | None = None,
        i: ArrayLike | None = None,
        c: ArrayLike | None = None,
        terminal: Any = None,
        tl: ArrayLike | None = None,
        time_scale: str = "calendar",
        family: str = "gamma",
        init: ArrayLike | None = None,
    ) -> JointFrailtyModel:
        """
        Fit the joint frailty model by maximising the marginal likelihood
        (the frailty integrated out by adaptive Gauss-Hermite quadrature).

        Parameters
        ----------
        x : array_like or SurpyvalData
            The ages of the recurrent events and of the end of each unit's
            observation, pooled over the units. Or a ``SurpyvalData`` with
            units ``i`` holding the whole of the data, the terminal event
            marked by ``a = 1`` on its event row (see the class
            description); the other data arguments are then left out.
        Z : array_like, optional
            Covariates: one row per row of ``x`` (a 1-D array is one
            covariate), the same on every row of a unit, or a DataFrame
            whose column names name the coefficients. They enter both the
            recurrent intensity (``beta``) and the terminal hazard
            (``gamma``). Default: none.
        i : array_like
            The unit each row belongs to.
        c : array_like
            0 on a recurrent event, 1 on the one row per unit that closes
            its observation.
        terminal : array_like or dict
            1 on a unit's closing row (``c = 1``) where the terminal event
            closed its observation, 0 where observation simply stopped;
            0 on every event row. Or a dict ``{unit: 0 or 1}``.
        tl : array_like or float, optional
            Each unit's age when its observation began (delayed entry):
            a scalar, one value per row (the same on a unit's rows) or one
            per unit in the sorted order of the ids. A unit is seen only
            if it had not reached the terminal event by then, which the
            likelihood takes into account. Calendar time scale only.
        time_scale : {'calendar', 'gap'}, optional
            The clock of the recurrent intensity: ``'calendar'`` (the
            default), the unit's age (the Andersen-Gill counting-process
            form, R ``frailtypack``'s ``recurrentAG=TRUE``), or ``'gap'``,
            the time since the unit's last event (each event renews the
            baseline, ``recurrentAG=FALSE``). The terminal hazard is
            always on the unit's age.
        family : {'gamma', 'lognormal'}, optional
            The frailty distribution: gamma with mean 1 and variance
            ``theta`` (the default), or log-normal with ``log u ~ N(0,
            theta)``.
        init : array_like, optional
            Starting values of the natural parameters, in the order of
            ``parameter_names``. The default start (the fits without a
            frailty) is tried as well, and the better answer kept.

        Returns
        -------
        JointFrailtyModel
            The fitted model.
        """
        check_option("family", family, FAMILIES)
        check_option("time_scale", time_scale, TIME_SCALES)
        units, p = _prepare(x, Z, i, c, terminal, tl, time_scale)
        names = fit_columns()
        aliased: npt.NDArray = np.asarray(
            alias_covariates(units.Z, intercept=True) if p else [], dtype=int
        )
        kept = np.setdiff1d(np.arange(p), aliased)
        Zk = units.Z[:, kept]
        mean, sd = _standardise(Zk)
        Zs = (Zk - mean) / sd
        q = kept.size
        lik_std = _Likelihood(units, Zs, family)
        n_obs = float(units.n_events.sum() + units.delta.sum())

        null_std, ll_null = self._null_fit(units, Zs, family)
        starts = []
        for alpha0 in (1.0, 0.0):
            s = null_std.copy()
            s[4], s[5] = 0.5, alpha0
            starts.append(s)
        if init is not None:
            init = np.asarray(init, dtype=float).reshape(-1)
            if init.size != 6 + 2 * p:
                raise ValueError(
                    f"'init' must have {6 + 2 * p} values, in the order of "
                    "parameter_names (four baseline parameters, theta, "
                    f"alpha, then {p} recurrent and {p} terminal "
                    f"coefficients); got {init.size}."
                )
            if np.any(init[:5] <= 0) or not np.all(np.isfinite(init)):
                raise ValueError(
                    "'init' must be finite, with the four baseline "
                    "parameters and theta positive."
                )
            mask = np.r_[np.arange(6), 6 + kept, 6 + p + kept]
            starts.insert(0, _to_standard(init[mask], mean, sd))
        search = _Search(lik_std, q)
        best = None
        for s in starts:
            res = _minimise(search, search.search(s))
            if best is None or (np.isfinite(res.fun) and res.fun < best.fun):
                best = res
        assert best is not None
        verified = bool(
            np.isfinite(best.fun)
            and best.fun < 1e300
            and _is_maximum(search, best.x, n_obs)
        )
        std = np.asarray(getval(search.natural(best.x)), dtype=float)
        ll_full = -float(best.fun)
        boundary = False
        if not (ll_full > ll_null + 1e-8) or std[4] < 1e-6:
            # theta at its boundary 0: the model without frailty, where
            # alpha has no meaning.
            boundary = True
            std = null_std.copy()
            std[4], std[5] = 0.0, 0.0
            ll_full = ll_null
            verified = True
            warnings.warn(
                "The frailty variance theta is estimated at its boundary "
                "0: these data show no unit-to-unit differences shared by "
                "the recurrent and the terminal events, so alpha is not "
                "identified and is reported as nan. The model is the two "
                "parts fitted independently; theta's standard error and "
                "bounds are nan.",
                UserWarning,
                stacklevel=caller_stacklevel(),
            )
        out = JointFrailtyModel()
        nat_kept = _to_natural(std, mean, sd)
        params = np.full(6 + 2 * p, np.nan)
        params[:6] = nat_kept[:6]
        params[6 + kept] = nat_kept[6 : 6 + q]
        params[6 + p + kept] = nat_kept[6 + q :]
        if boundary:
            params[5] = np.nan
        out.family = family
        out.time_scale = time_scale
        out.params = params
        out.feature_names = names
        out.n_covariates = p
        out.maximum = "verified" if verified else "unverified"
        if not verified:
            warn_unverified("The joint frailty fit")
        lik = _Likelihood(units, units.Z, family)
        out._lik = lik
        out._neg_ll = lik
        out._mle = params.copy()
        out._held = np.r_[
            [False] * 4, [boundary, boundary], np.isnan(params[6:])
        ].astype(bool)
        out._n_obs = n_obs
        out._ll_null = ll_null
        out.data = {
            "ids": units.ids,
            "n_units": units.n_units,
            "n_events": int(units.n_events.sum()),
            "n_terminal": int(units.delta.sum()),
        }
        with np.errstate(all="ignore"):
            post = lik.posterior_frailty(np.nan_to_num(params, nan=0.0))
        out.frailties = dict(zip(units.ids.tolist(), post.tolist()))
        return out

    def _null_fit(
        self, units: _Units, Zs: npt.NDArray, family: str
    ) -> tuple[npt.NDArray, float]:
        """The model without frailty (theta = 0): the recurrent and the
        terminal parts fitted independently, from starts read off the
        data. Returns its parameters (theta and alpha 0) and its
        log-likelihood."""
        q = Zs.shape[1]
        lik = _Likelihood(units, Zs, family)
        scale = float(np.max(units.end))
        rate_r = units.n_events.sum() / np.sum(units.end - units.entry)
        rate_t = units.delta.sum() / np.sum(units.end - units.entry)
        start = np.r_[
            1.0 / rate_r,
            1.0,
            1.0 / rate_t,
            1.0,
            0.0,
            0.0,
            np.zeros(2 * q),
        ]
        if not np.all(np.isfinite(start[:4])):
            start[:4] = scale
        fixed = {4: 0.0, 5: 0.0}
        search = _Search(lik, q, fixed)
        res = _minimise(search, search.search(start))
        out = np.asarray(getval(search.natural(res.x)), dtype=float)
        return out, -float(res.fun)

    def fit_from_df(
        self,
        df: Any,
        x_col: str,
        i_col: str,
        c_col: str,
        terminal_col: str | None = None,
        Z_cols: str | list[str] | None = None,
        tl_col: str | None = None,
        *,
        a_col: str | None = None,
        **fit_options: Any,
    ) -> JointFrailtyModel:
        """
        Fit to an event log held in the columns of a
        :class:`pandas.DataFrame`: the column names in place of the arrays
        of :meth:`fit`, every other option passed to it unchanged.

        Parameters
        ----------
        df : pandas.DataFrame
            The event log: one row per recurrent event and one per unit
            closing its observation.
        x_col, i_col, c_col, terminal_col : str
            The columns of ``x``, ``i``, ``c`` and ``terminal``. Give
            ``terminal_col`` or ``a_col``, not both.
        Z_cols : str or list of str, optional
            The covariate columns, which name the coefficients.
        tl_col : str, optional
            The column of each unit's entry age.
        a_col : str, optional
            In place of ``terminal_col``: the column of terminal flags of
            the data model, 1 on the event row (``c = 0``) of a unit's
            terminal event, 0 on its recurrent events, with a row
            ``c = 1`` at the end of the record of a unit that had none.
            A blank on a unit's last event reads as 1 (its terminal
            event, which the model's ``notes`` say) and as 0 before.
            The table is read by ``SurpyvalData.from_dataframe`` (each
            unit's rows put in time order) and fitted as ``fit(data)``.
        **fit_options
            ``time_scale``, ``family``, ``init``, ``unit``.

        Returns
        -------
        JointFrailtyModel
            The model :meth:`fit` returns.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval.recurrent import JointFrailty
        >>> truth = JointFrailty.from_params(
        ...     recurrent=(10.0, 1.2), terminal=(20.0, 1.5), theta=0.5,
        ...     alpha=1.0,
        ... )
        >>> log = pd.DataFrame(truth.random(100, end=15.0, random_state=2))
        >>> model = JointFrailty.fit_from_df(log, "x", "i", "c", "terminal")
        >>> model.n_covariates
        0
        """
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            raise ValueError("'df' must be a pandas DataFrame.")
        if (terminal_col is None) == (a_col is None):
            raise TypeError(
                "Give terminal_col (1 on the row c = 1 that closes a "
                "unit's observation where the terminal event closed it) or "
                "a_col (1 on the event row of the terminal event), one of "
                "the two."
            )
        if a_col is not None:
            return self._fit_from_a_col(
                df, x_col, i_col, c_col, a_col, Z_cols, tl_col, fit_options
            )
        cols = [x_col, i_col, c_col, terminal_col]
        if isinstance(Z_cols, str):
            Z_cols = [Z_cols]
        for name in (
            cols
            + list(Z_cols or [])
            + ([tl_col] if tl_col is not None else [])
        ):
            if name not in df.columns:
                raise ValueError(
                    f"{name!r} is not a column of df; its columns are "
                    f"{list(df.columns)}."
                )
        Z = None if not Z_cols else df[Z_cols].to_numpy(dtype=float)
        names = None if not Z_cols else [str(n) for n in Z_cols]
        with covariate_columns(names):
            model = self.fit(
                df[x_col].to_numpy(dtype=float),
                Z=Z,
                i=df[i_col].to_numpy(),
                c=df[c_col].to_numpy(),
                terminal=df[terminal_col].to_numpy(),
                tl=None if tl_col is None else df[tl_col].to_numpy(),
                **fit_options,
            )
        if names is not None:
            model.feature_names = names
        return model

    def _fit_from_a_col(
        self,
        df: Any,
        x_col: str,
        i_col: str,
        c_col: str,
        a_col: str,
        Z_cols: str | list[str] | None,
        tl_col: str | None,
        fit_options: dict,
    ) -> JointFrailtyModel:
        """``fit_from_df`` with ``a_col``: the table as a ``SurpyvalData``
        (each unit's rows in time order, a terminal event after the
        recurrent events at its age), fitted as ``fit(data)``."""
        from surpyval.utils.surpyval_data import SurpyvalData

        if isinstance(Z_cols, str):
            Z_cols = [Z_cols]
        cols = [x_col, i_col, c_col, a_col] + list(Z_cols or [])
        for name in cols + ([tl_col] if tl_col is not None else []):
            if name not in df.columns:
                raise ValueError(
                    f"{name!r} is not a column of df; its columns are "
                    f"{list(df.columns)}."
                )
        x = df[x_col].to_numpy(dtype=float)
        c = df[c_col].to_numpy()
        flags = df[a_col].to_numpy(dtype=float, na_value=np.nan)
        i = df[i_col].to_numpy()
        # A unit's rows in time order: at one age, its recurrent events,
        # then its terminal event or the end of its record.
        last = np.where(np.asarray(c) == 1, 2, np.where(flags == 1, 1, 0))
        try:
            order = np.lexsort((last, x, i))
        except TypeError:
            order = np.lexsort((last, x, np.asarray([str(v) for v in i])))
        ids = i[order]
        kw: dict = {}
        if tl_col is not None:
            # The unit's entry, on its first row (a later row starts where
            # the previous one ended).
            tl = df[tl_col].to_numpy(dtype=float)[order]
            first = np.r_[True, ids[1:] != ids[:-1]]
            kw["tl"] = np.where(first, tl, -np.inf)
        if Z_cols:
            kw["Z"] = df[Z_cols].to_numpy(dtype=float)[order]
            kw["covariate_names"] = [str(n) for n in Z_cols]
        data = SurpyvalData(
            x=x[order],
            c=np.asarray(c)[order],
            i=[v.item() if isinstance(v, np.generic) else v for v in ids],
            a=[None if np.isnan(v) else int(v) for v in flags[order]],
            **kw,
        )
        return self.fit(data, **fit_options)

    def from_params(
        self,
        recurrent: ArrayLike,
        terminal: ArrayLike,
        theta: float,
        alpha: float,
        beta: ArrayLike | None = None,
        gamma: ArrayLike | None = None,
        family: str = "gamma",
        time_scale: str = "calendar",
    ) -> JointFrailtyModel:
        """
        A joint frailty model with the given parameters, to predict from
        or to simulate units with (:meth:`JointFrailtyModel.random`).

        Parameters
        ----------
        recurrent, terminal : (scale, shape)
            The Weibull baselines: ``R_0(t) = (t / scale) ** shape`` and
            ``Lambda_0(t) = (t / scale) ** shape``.
        theta : float
            The frailty's variance (of its log for the log-normal), at
            least 0.
        alpha : float
            The power of the frailty in the terminal hazard.
        beta, gamma : array_like, optional
            The coefficients of the recurrent and the terminal parts, of
            one length. Default: no covariates.
        family : {'gamma', 'lognormal'}, optional
            Default ``'gamma'``.
        time_scale : {'calendar', 'gap'}, optional
            Default ``'calendar'``.

        Returns
        -------
        JointFrailtyModel
            A model with no data, so no likelihood.

        Examples
        --------
        >>> from surpyval.recurrent import JointFrailty
        >>> model = JointFrailty.from_params(
        ...     (10.0, 1.2), (20.0, 1.5), 0.5, 1.0
        ... )
        >>> float(model.cif(10.0, None))
        1.0
        """
        check_option("family", family, FAMILIES)
        check_option("time_scale", time_scale, TIME_SCALES)
        rec = np.asarray(recurrent, dtype=float).reshape(-1)
        term = np.asarray(terminal, dtype=float).reshape(-1)
        if rec.size != 2 or term.size != 2:
            raise ValueError(
                "'recurrent' and 'terminal' are each (scale, shape)."
            )
        if np.any(np.r_[rec, term] <= 0) or not np.all(
            np.isfinite(np.r_[rec, term])
        ):
            raise ValueError(
                "The Weibull scales and shapes must be finite and positive."
            )
        if not (np.isfinite(theta) and theta >= 0):
            raise ValueError("'theta' must be finite and at least 0.")
        if not np.isfinite(alpha):
            raise ValueError("'alpha' must be finite.")
        b = np.zeros(0) if beta is None else np.asarray(beta, float).ravel()
        g = np.zeros(0) if gamma is None else np.asarray(gamma, float).ravel()
        if beta is None and gamma is not None:
            b = np.zeros(g.size)
        if gamma is None and beta is not None:
            g = np.zeros(b.size)
        if b.size != g.size:
            raise ValueError(
                "'beta' and 'gamma' must have one coefficient per "
                f"covariate each; got {b.size} and {g.size}."
            )
        out = JointFrailtyModel()
        out.family = family
        out.time_scale = time_scale
        out.params = np.r_[rec, term, float(theta), float(alpha), b, g]
        out.n_covariates = b.size
        out.feature_names = None
        out.maximum = "unknown"
        out.frailties = None
        out.data = None
        return out


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class JointFrailtyModel(SerialisableMixin, LikelihoodInferenceMixin):
    """
    A fitted joint frailty model of recurrent events and a terminal event
    (see :data:`JointFrailty`).

    Its predictions are for a unit with covariates ``Z``: either for the
    population of such units (``frailty=None``, the frailty averaged
    over its distribution) or for one unit of known frailty
    (``frailty=u``; ``model.frailties`` holds each fitted unit's
    estimate). Recurrent events: :meth:`cif` and :meth:`iif`, the
    cumulative intensity and the intensity with no terminal event to
    stop them, and :meth:`mcf`, the expected number of recurrent events
    by each age with the terminal event stopping them (the calendar time
    scale). The terminal event: :meth:`sf`, :meth:`ff`, :meth:`hf`,
    :meth:`Hf` and :meth:`df`.

    Attributes
    ----------
    params : numpy.ndarray
        The parameters, named by ``parameter_names``: the recurrent and
        terminal Weibull scales and shapes, ``theta``, ``alpha``, then the
        recurrent coefficients ``beta`` and the terminal ones ``gamma``.
    frailties : dict
        Each fitted unit's predicted frailty, its posterior mean given its
        history (empirical Bayes).
    family, time_scale : str
        As fitted.
    maximum : str
        ``"verified"`` where the fit reached a verified maximum.

    Examples
    --------
    >>> from surpyval.recurrent import JointFrailty
    >>> model = JointFrailty.from_params(
    ...     (10.0, 1.2), (20.0, 1.5), 0.5, 1.0
    ... )
    >>> # The population's terminal survival, and that of a frail unit
    >>> s_all = float(model.sf(10.0))
    >>> s_frail = float(model.sf(10.0, frailty=2.0))
    >>> round(s_all, 4), round(s_frail, 4)
    (0.7221, 0.4931)
    """

    params: npt.NDArray
    family: str
    time_scale: str
    n_covariates: int
    feature_names: list[str] | None
    maximum: str
    frailties: dict | None
    data: dict | None
    _held: npt.NDArray
    _lik: _Likelihood
    _ll_null: float
    #: What ``fit(data)`` made of the data's terminal flags ``a`` (empty
    #: from arrays); not serialised.
    notes: tuple = ()

    def __getstate__(self) -> dict:
        """The state to pickle, without the likelihood-ratio searches'
        cache, which is rebuilt when needed."""
        state = dict(vars(self))
        state.pop("_lr_region", None)
        return state

    # -- names and parts ----------------------------------------------------
    def _coefficient_names(self) -> list[str]:
        p = self.n_covariates
        names = self.feature_names
        if names is None or len(names) != p:
            names = [f"coef_{j}" for j in range(p)]
        return [str(n) for n in names]

    def _parameter_names(self) -> list[str]:
        cols = self._coefficient_names()
        return (
            BASELINE_NAMES
            + ["theta", "alpha"]
            + [f"recurrent: {c}" for c in cols]
            + [f"terminal: {c}" for c in cols]
        )

    def _parameter_bounds(self) -> list:
        return [(0, None)] * 5 + [(None, None)] * (1 + 2 * self.n_covariates)

    def _estimated(self) -> npt.NDArray:
        held = getattr(self, "_held", None)
        if held is None:
            return ~np.isnan(np.asarray(self._mle, dtype=float))
        return ~np.asarray(held, dtype=bool)

    @property
    def theta(self) -> float:
        """The frailty variance (of ``log u`` for the log-normal)."""
        return float(self.params[4])

    @property
    def alpha(self) -> float:
        """The power of the frailty in the terminal hazard: 0 where the
        terminal event is independent of the recurrences given the
        covariates, positive where the units with many recurrences reach
        it sooner. ``nan`` where ``theta`` is 0."""
        return float(self.params[5])

    @property
    def beta(self) -> npt.NDArray:
        """The recurrent intensity's coefficients (``nan`` where
        aliased)."""
        return self.params[6 : 6 + self.n_covariates].copy()

    @property
    def gamma(self) -> npt.NDArray:
        """The terminal hazard's coefficients (``nan`` where aliased)."""
        return self.params[6 + self.n_covariates :].copy()

    @property
    def aliased(self) -> npt.NDArray:
        """The columns of ``Z`` whose coefficients the data cannot
        determine (a constant column, the baselines' scales being the
        intercepts, or a combination of the others): ``nan`` in ``beta``
        and ``gamma``, taken as 0 in predictions."""
        return np.flatnonzero(np.isnan(self.beta))

    def _p(self) -> npt.NDArray:
        """The parameters as the predictions take them: an aliased
        coefficient, or alpha on the boundary, as 0."""
        return np.nan_to_num(np.asarray(self.params, dtype=float), nan=0.0)

    # -- covariates and frailty --------------------------------------------
    def _rows(self, Z: Any, n: int) -> npt.NDArray:
        p = self.n_covariates
        if p == 0:
            return np.zeros((n, 0))
        if Z is None:
            raise ValueError(
                f"'Z' is needed: the model has {p} covariate(s); give one "
                "row, or one row per time."
            )
        if type(Z).__name__ == "DataFrame":
            names = self.feature_names
            if names is not None:
                Z = Z[names]
            Z = Z.to_numpy(dtype=float)
        Zr = np.asarray(Z, dtype=float)
        if Zr.ndim <= 1:
            Zr = Zr.reshape(1, -1)
        if Zr.shape[1] != p:
            raise ValueError(
                f"'Z' must have {p} column(s), one per covariate; got "
                f"{Zr.shape[1]}."
            )
        if Zr.shape[0] == 1:
            return np.repeat(Zr, n, axis=0)
        if Zr.shape[0] != n and n != 1:
            raise ValueError(
                f"'Z' has {Zr.shape[0]} rows for {n} times: give one row, "
                "or one row per time."
            )
        return Zr

    def _paired(
        self, x: npt.NDArray, Z: Any
    ) -> tuple[npt.NDArray, npt.NDArray]:
        x = np.asarray(x, dtype=float).reshape(-1)
        Zr = self._rows(Z, x.size)
        if Zr.shape[0] != x.size:
            if x.size == 1:
                x = np.repeat(x, Zr.shape[0])
        return x, Zr

    def _frailty(self, frailty: Any, n: int) -> npt.NDArray | None:
        if frailty is None:
            return None
        u = np.asarray(frailty, dtype=float).reshape(-1)
        if u.size == 1:
            u = np.repeat(u, n)
        if u.size != n:
            raise ValueError("'frailty' must be one value, or one per time.")
        if np.any(u < 0):
            raise ValueError("'frailty' must be at least 0.")
        return u

    def _mean_frailty(self, params: npt.NDArray) -> float:
        theta = float(params[4])
        return float(np.exp(theta / 2)) if self.family == "lognormal" else 1.0

    # -- the functions (for given parameters) ------------------------------
    def _f(
        self,
        name: str,
        params: npt.NDArray,
        x: npt.NDArray,
        Zr: npt.NDArray,
        u: npt.NDArray | None,
    ) -> npt.NDArray:
        p = self.n_covariates
        rs, rk, ts, tk, theta, alpha = (float(v) for v in params[:6])
        beta = params[6 : 6 + p]
        gamma = params[6 + p :]
        er = np.exp(Zr @ beta) if p else np.ones(x.size)
        et = np.exp(Zr @ gamma) if p else np.ones(x.size)
        nan = np.isnan(x)
        xs = np.where(nan | (x < 0), 0.0, x)
        with np.errstate(all="ignore"):
            if name in ("cif", "iif"):
                mult = self._mean_frailty(params) if u is None else u
                if name == "cif":
                    out = mult * er * (xs / rs) ** rk
                else:
                    out = mult * er * rk / rs * (xs / rs) ** (rk - 1.0)
            elif name == "mcf":
                out = self._mcf(params, xs, er, et, u)
            else:
                H = et * (xs / ts) ** tk
                h0 = et * tk / ts * (xs / ts) ** (tk - 1.0)
                if u is None:
                    zero = np.zeros_like(H)
                    if theta <= 0:
                        log_s = -H
                        ratio = np.ones_like(H)
                    else:
                        log_s = log_frailty_integral(
                            zero, zero, H, alpha, theta, self.family
                        )
                        ratio = np.exp(
                            log_frailty_integral(
                                zero + alpha,
                                zero,
                                H,
                                alpha,
                                theta,
                                self.family,
                            )
                            - log_s
                        )
                    hf = h0 * ratio
                else:
                    ua = u**alpha
                    log_s = -ua * H
                    hf = ua * h0
                if name == "sf":
                    out = np.exp(log_s)
                elif name == "ff":
                    out = -np.expm1(log_s)
                elif name == "Hf":
                    out = -log_s
                elif name == "hf":
                    out = hf
                else:
                    out = hf * np.exp(log_s)
            if name in ("iif", "hf", "df"):
                out = np.where(x < 0, 0.0, out)
            else:
                out = np.where(x < 0, 0.0 if name != "sf" else 1.0, out)
        return np.where(nan, np.nan, out)

    def _mcf(
        self,
        params: npt.NDArray,
        x: npt.NDArray,
        er: npt.NDArray,
        et: npt.NDArray,
        u: npt.NDArray | None,
    ) -> npt.NDArray:
        """``E[N(x)]``: ``e^{beta'Z} int_0^{R_0(x)} E[u S_T(s | u)] dv``
        with ``v = R_0(s)``, by Gauss-Legendre in ``y`` with ``v = V
        y^2`` (smooth where the integrand has a power of ``v`` at 0)."""
        if self.time_scale != "calendar":
            raise ValueError(
                "mcf is the expected count of recurrent events on the "
                "calendar time scale; with time_scale='gap' each event "
                "renews the intensity, and the expected count has no "
                "closed form: simulate it with random()."
            )
        rs, rk, ts, tk, theta, alpha = (float(v) for v in params[:6])
        V = (x / rs) ** rk
        y, wt = roots_legendre(N_TIME_NODES)
        y = 0.5 * (y + 1.0)
        wt = 0.5 * wt
        v = V[:, None] * y**2
        s = rs * v ** (1.0 / rk)
        H = et[:, None] * (s / ts) ** tk
        if u is None:
            if theta <= 0:
                inner = np.exp(-H)
            else:
                zero = np.zeros_like(H)
                inner = np.exp(
                    log_frailty_integral(
                        zero + 1.0, zero, H, alpha, theta, self.family
                    )
                )
            mult = er
        else:
            inner = np.exp(-(u[:, None] ** alpha) * H)
            mult = er * u
        integral = np.sum(wt * inner * 2.0 * V[:, None] * y, axis=1)
        return mult * integral

    def _evaluate(
        self, name: str, x: Any, Z: Any, frailty: Any, params: Any = None
    ) -> npt.NDArray:
        xs, Zr = self._paired(x, Z)
        u = self._frailty(frailty, xs.size)
        par = self._p() if params is None else params
        return self._f(name, par, xs, Zr, u)

    # -- the public functions ---------------------------------------------
    @keeps_query_shape
    def cif(self, x: ArrayLike, Z: Any = None, frailty: Any = None) -> Any:
        """
        The cumulative intensity of recurrent events at age ``x`` (the
        time since the last event with ``time_scale='gap'``): the
        expected number of recurrent events by ``x`` were there no
        terminal event to stop them. ``frailty=None`` averages over the
        frailty (``E[u] R_0(x) e^{beta' Z}``, ``E[u]`` 1 for the gamma
        frailty and ``e^{theta / 2}`` for the log-normal); ``frailty=u``
        is that of a unit of frailty ``u``.

        Parameters
        ----------
        x : array_like
            Ages.
        Z : array_like, optional
            Covariates: one row for every age, or one per age.
        frailty : float or array_like, optional
            A unit's frailty; default the population average.

        Returns
        -------
        numpy.ndarray
            The cumulative intensity, of the shape of ``x``.

        Examples
        --------
        >>> from surpyval.recurrent import JointFrailty
        >>> model = JointFrailty.from_params(
        ...     (10.0, 2.0), (20.0, 1.5), 0.5, 1.0
        ... )
        >>> float(model.cif(20.0))
        4.0
        >>> float(model.cif(20.0, frailty=2.0))
        8.0
        """
        return self._evaluate("cif", x, Z, frailty)

    @keeps_query_shape
    def iif(self, x: ArrayLike, Z: Any = None, frailty: Any = None) -> Any:
        """
        The intensity of recurrent events at age ``x`` (or time since the
        last event, ``time_scale='gap'``), averaged over the frailty
        (``frailty=None``) or for a unit of frailty ``u``: the derivative
        of :meth:`cif`.

        Examples
        --------
        >>> from surpyval.recurrent import JointFrailty
        >>> model = JointFrailty.from_params(
        ...     (10.0, 2.0), (20.0, 1.5), 0.5, 1.0
        ... )
        >>> float(model.iif(10.0))
        0.2
        """
        return self._evaluate("iif", x, Z, frailty)

    @keeps_query_shape
    def mcf(self, x: ArrayLike, Z: Any = None, frailty: Any = None) -> Any:
        """
        The mean cumulative function: the expected number of recurrent
        events by age ``x``, the terminal event stopping them,

        .. math::
            E[N(x)] = e^{\\beta' Z} \\int_0^x r_0(s)\\,
            E\\big[u\\, e^{-u^\\alpha e^{\\gamma' Z} \\Lambda_0(s)}\\big]\\,ds

        (for one unit of frailty ``u``, ``u`` in place of the
        expectation). Below :meth:`cif` wherever units reach the terminal
        event. Calendar time scale only.

        Examples
        --------
        >>> from surpyval.recurrent import JointFrailty
        >>> model = JointFrailty.from_params(
        ...     (10.0, 1.0), (20.0, 1.0), 0.0, 1.0
        ... )
        >>> # no frailty, constant rates 0.1 and 0.05: 2 (1 - e^{-0.05 x})
        >>> round(float(model.mcf(20.0)), 6)
        1.264241
        """
        return self._evaluate("mcf", x, Z, frailty)

    @keeps_query_shape
    def sf(self, x: ArrayLike, Z: Any = None, frailty: Any = None) -> Any:
        """
        The survival function of the terminal event: the probability that
        a unit has not reached it by age ``x``, averaged over the frailty
        (``frailty=None``, :math:`E[e^{-u^\\alpha e^{\\gamma' Z}
        \\Lambda_0(x)}]`) or for a unit of frailty ``u``.

        Examples
        --------
        >>> from surpyval.recurrent import JointFrailty
        >>> model = JointFrailty.from_params(
        ...     (10.0, 1.2), (20.0, 1.0), 0.5, 1.0
        ... )
        >>> # gamma frailty, alpha = 1: (1 + theta H)^(-1 / theta)
        >>> round(float(model.sf(20.0)), 6), round((1 + 0.5) ** -2, 6)
        (0.444444, 0.444444)
        """
        return self._evaluate("sf", x, Z, frailty)

    @keeps_query_shape
    def ff(self, x: ArrayLike, Z: Any = None, frailty: Any = None) -> Any:
        """The probability of the terminal event by age ``x``, ``1 -
        sf``.

        Examples
        --------
        >>> from surpyval.recurrent import JointFrailty
        >>> model = JointFrailty.from_params(
        ...     (10.0, 1.2), (20.0, 1.0), 0.5, 1.0
        ... )
        >>> round(float(model.ff(20.0)), 6)
        0.555556
        """
        return self._evaluate("ff", x, Z, frailty)

    @keeps_query_shape
    def Hf(self, x: ArrayLike, Z: Any = None, frailty: Any = None) -> Any:
        """The cumulative hazard of the terminal event, ``-log sf``.

        Examples
        --------
        >>> from surpyval.recurrent import JointFrailty
        >>> model = JointFrailty.from_params(
        ...     (10.0, 1.2), (20.0, 1.0), 0.5, 1.0
        ... )
        >>> round(float(model.Hf(20.0)), 6)
        0.81093
        """
        return self._evaluate("Hf", x, Z, frailty)

    @keeps_query_shape
    def hf(self, x: ArrayLike, Z: Any = None, frailty: Any = None) -> Any:
        """The hazard of the terminal event: for the population of units
        alive at ``x`` (``frailty=None``), which falls behind the
        conditional hazard as the frail units leave, or for one unit.

        Examples
        --------
        >>> from surpyval.recurrent import JointFrailty
        >>> model = JointFrailty.from_params(
        ...     (10.0, 1.2), (20.0, 1.0), 0.5, 1.0
        ... )
        >>> # gamma, alpha = 1: h0 / (1 + theta H) = 0.05 / 1.5
        >>> round(float(model.hf(20.0)), 6)
        0.033333
        """
        return self._evaluate("hf", x, Z, frailty)

    @keeps_query_shape
    def df(self, x: ArrayLike, Z: Any = None, frailty: Any = None) -> Any:
        """The density of the age at the terminal event, ``hf * sf``.

        Examples
        --------
        >>> from surpyval.recurrent import JointFrailty
        >>> model = JointFrailty.from_params(
        ...     (10.0, 1.2), (20.0, 1.0), 0.5, 1.0
        ... )
        >>> round(float(model.df(20.0)), 6)
        0.014815
        """
        return self._evaluate("df", x, Z, frailty)

    # -- confidence bounds ---------------------------------------------------
    def _delta_cb(
        self,
        name: str,
        x: Any,
        Z: Any,
        alpha_ci: float,
        bound: str,
        frailty: Any,
    ) -> npt.NDArray:
        """Delta-method bounds, on the log scale of the function (of the
        cumulative hazard, for ``sf``), the gradient by central
        differences over the estimated parameters."""
        from scipy.stats import norm

        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        self._check_inference()
        on = "Hf" if name == "sf" else name
        xs, Zr = self._paired(x, Z)
        u = self._frailty(frailty, xs.size)
        base = self._p()
        value = self._f(on, base, xs, Zr, u)
        cov, _ = self._covariance_matrix()
        free = np.flatnonzero(self._estimated())
        grad = np.zeros((xs.size, free.size))
        with np.errstate(all="ignore"):
            log_value = np.log(value)
            for k, j in enumerate(free):
                h = 1e-5 * max(abs(base[j]), 1e-3)
                up, down = base.copy(), base.copy()
                up[j] += h
                down[j] -= h
                grad[:, k] = (
                    np.log(self._f(on, up, xs, Zr, u))
                    - np.log(self._f(on, down, xs, Zr, u))
                ) / (2 * h)
            var = np.einsum("ij,jk,ik->i", grad, cov[np.ix_(free, free)], grad)
            se = np.sqrt(np.where(var >= 0, var, np.nan))
            if bound == "two-sided":
                q = norm.ppf(1 - alpha_ci / 2)
                lo = np.exp(log_value - q * se)
                hi = np.exp(log_value + q * se)
                out = np.stack([lo, hi], axis=-1)
            else:
                q = norm.ppf(1 - alpha_ci)
                sign = -1.0 if bound == "lower" else 1.0
                if name == "sf":
                    sign = -sign
                out = np.exp(log_value + sign * q * se)
            # where the function is 0 (age 0) it is known exactly
            zero = value == 0
            if bound == "two-sided":
                out[zero] = 0.0
            else:
                out = np.where(zero, 0.0, out)
            if name == "sf":
                out = np.exp(-out)
                if bound == "two-sided":
                    out = out[:, ::-1]
        return out

    @keeps_query_shape
    def cif_cb(
        self,
        x: ArrayLike,
        Z: Any = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        frailty: Any = None,
    ) -> Any:
        """
        Confidence bounds on :meth:`cif` by the delta method on its log,
        from the parameters' covariance.

        Parameters
        ----------
        x : array_like
            Ages.
        Z : array_like, optional
            Covariates, as for :meth:`cif`.
        alpha_ci : float, optional
            The total tail probability. Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds add a last ``[lower, upper]`` axis.
        frailty : float, optional
            As for :meth:`cif`.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import JointFrailty
        >>> truth = JointFrailty.from_params(
        ...     (10.0, 1.2), (20.0, 1.5), 0.5, 1.0
        ... )
        >>> rows = truth.random(150, end=15.0, random_state=3)
        >>> model = JointFrailty.fit(**rows)
        >>> lo, hi = model.cif_cb(10.0)
        >>> bool(lo < model.cif(10.0) < hi)
        True
        """
        return self._delta_cb("cif", x, Z, alpha_ci, bound, frailty)

    @keeps_query_shape
    def iif_cb(
        self,
        x: ArrayLike,
        Z: Any = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        frailty: Any = None,
    ) -> Any:
        """Confidence bounds on :meth:`iif`, as :meth:`cif_cb`.

        Examples
        --------
        >>> from surpyval.recurrent import JointFrailty
        >>> truth = JointFrailty.from_params(
        ...     (10.0, 1.2), (20.0, 1.5), 0.5, 1.0
        ... )
        >>> rows = truth.random(150, end=15.0, random_state=3)
        >>> model = JointFrailty.fit(**rows)
        >>> model.iif_cb([5.0, 10.0]).shape
        (2, 2)
        """
        return self._delta_cb("iif", x, Z, alpha_ci, bound, frailty)

    @keeps_query_shape
    def mcf_cb(
        self,
        x: ArrayLike,
        Z: Any = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        frailty: Any = None,
    ) -> Any:
        """Confidence bounds on :meth:`mcf`, as :meth:`cif_cb`.

        Examples
        --------
        >>> from surpyval.recurrent import JointFrailty
        >>> truth = JointFrailty.from_params(
        ...     (10.0, 1.2), (20.0, 1.5), 0.5, 1.0
        ... )
        >>> rows = truth.random(150, end=15.0, random_state=3)
        >>> model = JointFrailty.fit(**rows)
        >>> model.mcf_cb(10.0, bound="upper") > model.mcf(10.0)
        np.True_
        """
        return self._delta_cb("mcf", x, Z, alpha_ci, bound, frailty)

    @keeps_query_shape
    def sf_cb(
        self,
        x: ArrayLike,
        Z: Any = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        frailty: Any = None,
    ) -> Any:
        """Confidence bounds on the terminal event's :meth:`sf`, by the
        delta method on the log of its cumulative hazard, so that they
        stay in [0, 1].

        Examples
        --------
        >>> from surpyval.recurrent import JointFrailty
        >>> truth = JointFrailty.from_params(
        ...     (10.0, 1.2), (20.0, 1.5), 0.5, 1.0
        ... )
        >>> rows = truth.random(150, end=15.0, random_state=3)
        >>> model = JointFrailty.fit(**rows)
        >>> lo, hi = model.sf_cb(10.0)
        >>> bool(lo < model.sf(10.0) < hi)
        True
        """
        return self._delta_cb("sf", x, Z, alpha_ci, bound, frailty)

    # -- parameters ----------------------------------------------------------
    def _covariance_matrix(self) -> tuple[npt.NDArray, bool]:
        """The inverse of autograd's Hessian of the negative marginal
        log-likelihood at the estimate, over the estimated parameters."""
        self._check_inference()
        lik = getattr(self, "_lik", None)
        if lik is None:
            return super()._covariance_matrix()
        cached = getattr(self, "_cov_cache", None)
        if cached is not None:
            return cached
        full = self._p()
        free = self._estimated()
        n = full.size

        def neg_ll_free(values: Any) -> Any:
            parts = []
            k = 0
            for j in range(n):
                if free[j]:
                    parts.append(values[k])
                    k += 1
                else:
                    parts.append(full[j] + 0.0 * values[0])
            return lik(anp.array(parts))

        out = np.full((n, n), np.nan)
        with np.errstate(all="ignore"):
            H = np.asarray(_ag_hessian(neg_ll_free)(full[free]), dtype=float)
        found = bool(np.all(np.isfinite(H)))
        if found:
            try:
                out[np.ix_(free, free)] = np.linalg.inv(0.5 * (H + H.T))
            except np.linalg.LinAlgError:
                found = False
        self._cov_cache = (out, found)
        return out, found

    def param_cb(
        self,
        name: str,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str | None = "wald",
    ) -> npt.NDArray:
        """
        Confidence bound(s) on a named parameter.

        - ``"wald"`` (the default) -- from the covariance, on the log
          scale for the positive parameters (the baselines' and
          ``theta``) and the natural scale for ``alpha`` and the
          coefficients.
        - ``"lr"`` -- the profile-likelihood interval: the values whose
          deviance, every other parameter re-fitted, stays below the
          :math:`\\chi^2_1` critical value. Where it stays below it down
          to ``theta = 0`` the lower bound on ``theta`` is 0. It needs the
          data, which a model restored from a dict does not keep.

        Parameters
        ----------
        name : str
            One of :attr:`parameter_names`.
        alpha_ci : float, optional
            Total tail probability. Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are ``[lower, upper]``.
        method : {'wald', 'lr'}, optional
            As above.

        Examples
        --------
        >>> from surpyval.recurrent import JointFrailty
        >>> truth = JointFrailty.from_params(
        ...     (10.0, 1.2), (20.0, 1.5), 0.5, 1.0
        ... )
        >>> rows = truth.random(150, end=15.0, random_state=3)
        >>> model = JointFrailty.fit(**rows)
        >>> model.param_cb("alpha").round(2)
        array([-0.15,  1.82])
        >>> model.param_cb("alpha", method="lr").round(2)
        array([0.06, 2.57])
        """
        from surpyval.univariate.regression._likelihood_ratio import (
            LikelihoodRegion,
            is_lr,
            profile_interval,
        )

        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        if not is_lr(method):
            check_option("method", method, ("wald", "lr"))
            names = self.parameter_names
            if name not in names:
                raise ValueError(
                    "Unknown parameter {!r}; expected one of {}".format(
                        name, names
                    )
                )
            idx = names.index(name)
            if not self._estimated()[idx]:
                value = float(self.params[idx])
                return np.array(
                    [value, value] if bound == "two-sided" else [value]
                )
            return super().param_cb(name, alpha_ci, bound)
        if name not in self.parameter_names:
            raise ValueError(
                "Unknown parameter {!r}; expected one of {}".format(
                    name, self.parameter_names
                )
            )
        lik = getattr(self, "_lik", None)
        if lik is None:
            raise ValueError(
                "Likelihood-ratio bounds need the data the model was fitted "
                "to, which a model restored from a dict (or built from "
                "parameters) does not keep; use method='wald'."
            )
        region = getattr(self, "_lr_region", None)
        if region is None:
            held = set(np.flatnonzero(~self._estimated()).tolist())
            cov, _ = self._covariance_matrix()
            region = LikelihoodRegion(
                _FloatLikelihood(lik),
                self._p(),
                list(self.parameter_names),
                [(0, None)] * 4
                + [(0, None), (None, None)]
                + [(None, None)] * (2 * self.n_covariates),
                held,
                cov,
            )
            self._lr_region = region
        return profile_interval(region, name, alpha_ci, bound)

    def frailty_tests(self) -> Any:
        """
        Two likelihood-ratio tests of the dependence between the
        recurrent and the terminal events.

        - ``theta = 0``: no frailty, so the units differ only through
          their covariates and the two parts are independent. The model
          without frailty is the two parts fitted separately. Near the
          null the frailty acts through two variances on the edge of their
          space, ``theta`` in the recurrent part and ``alpha^2 theta`` in
          the terminal part (``alpha`` is not identified under the null),
          so the p-value is that of a 50:50 mixture of
          :math:`\\chi^2_1` and :math:`\\chi^2_2` (Self and Liang 1987;
          Stram and Lee 1994), an approximation whose size is checked in
          the calibration study. (The single-variance reference, half a
          :math:`\\chi^2_1`, rejected a true null 18% of the time at 5%;
          this mixture held 5.0% in that study and 4.6% in a second one.)
        - ``alpha = 0``: the frailty acts on the recurrent events only, so
          the terminal event does not depend on how often a unit has
          recurrences. The model is refitted with ``alpha`` held at 0
          (``theta`` free); the statistic is compared with
          :math:`\\chi^2_1`.

        Returns
        -------
        pandas.DataFrame
            One row per test: ``statistic`` (twice the difference of the
            log-likelihoods), ``df``, ``p`` and ``reference`` (the null
            distribution).

        Examples
        --------
        >>> from surpyval.recurrent import JointFrailty
        >>> truth = JointFrailty.from_params(
        ...     (10.0, 1.2), (20.0, 1.5), 0.5, 1.0
        ... )
        >>> rows = truth.random(150, end=15.0, random_state=3)
        >>> model = JointFrailty.fit(**rows)
        >>> model.frailty_tests()["p"].round(5).tolist()
        [0.00012, 0.03289]
        """
        import pandas as pd

        lik = getattr(self, "_lik", None)
        if lik is None:
            raise ValueError(
                "The tests need the data the model was fitted to, which a "
                "model restored from a dict (or built from parameters) does "
                "not keep; refit it."
            )
        ll = -float(lik(self._p()))
        t_theta = max(0.0, 2.0 * (ll - self._ll_null))
        if np.isnan(self.params[5]):
            t_alpha = float("nan")
        else:
            t_alpha = max(0.0, 2.0 * (ll - self._ll_alpha_zero()))
        return pd.DataFrame(
            {
                "statistic": [t_theta, t_alpha],
                "df": [2, 1],
                "p": [
                    _boundary_p(t_theta),
                    (
                        float(chi2.sf(t_alpha, 1))
                        if np.isfinite(t_alpha)
                        else np.nan
                    ),
                ],
                "reference": [
                    "50:50 mixture of chi2(1) and chi2(2)",
                    "chi2(1)",
                ],
            },
            index=["theta = 0", "alpha = 0"],
        )

    def _ll_alpha_zero(self) -> float:
        """The maximised log-likelihood with ``alpha`` held at 0."""
        lik = self._lik
        full = self._p()
        free = self._estimated()
        fixed = {j: float(full[j]) for j in range(full.size) if not free[j]}
        fixed[5] = 0.0
        search = _Search(lik, self.n_covariates, fixed)
        best = np.inf
        for theta0 in (max(self.theta, 1e-3), 0.5):
            start = full.copy()
            start[5] = 0.0
            start[4] = theta0
            res = _minimise(search, search.search(start))
            if np.isfinite(res.fun):
                best = min(best, float(res.fun))
        return -best

    def summary(self, alpha_ci: float = 0.05) -> Any:
        """
        The parameters with their standard errors and Wald intervals, one
        row per entry of :attr:`parameter_names` (``nan`` where not
        estimated).

        Examples
        --------
        >>> from surpyval.recurrent import JointFrailty
        >>> truth = JointFrailty.from_params(
        ...     (10.0, 1.2), (20.0, 1.5), 0.5, 1.0
        ... )
        >>> rows = truth.random(150, end=15.0, random_state=3)
        >>> model = JointFrailty.fit(**rows)
        >>> list(model.summary().columns)
        ['estimate', 'se', 'lower 95%', 'upper 95%']
        """
        out = super().summary(alpha_ci)
        out["estimate"] = np.asarray(self.params, dtype=float)
        return out

    # -- simulation --------------------------------------------------------
    def random(
        self,
        size: int,
        end: Any,
        Z: Any = None,
        random_state: Any = None,
    ) -> dict:
        """
        Simulate ``size`` units from the model, each followed from age 0
        until its terminal event or the end of its observation ``end``,
        whichever comes first.

        Each unit draws its frailty ``u``; its terminal age from the
        Weibull hazard ``u^alpha lambda_0(t) e^{gamma' Z}``; and its
        recurrent events from the intensity ``u r_0 e^{beta' Z}`` (on its
        age, or renewed at each event with ``time_scale='gap'``) until the
        earlier of the two.

        Parameters
        ----------
        size : int
            The number of units.
        end : float or array_like
            The age at which each unit's observation ends (one value, or
            one per unit).
        Z : array_like, optional
            The units' covariates: one row for all, or one per unit.
        random_state : optional
            A seed or ``numpy.random.Generator``; ``None`` draws from
            numpy's global generator.

        Returns
        -------
        dict
            ``x``, ``i``, ``c``, ``terminal`` (and ``Z`` with covariates),
            ready for :meth:`JointFrailty.fit` (``JointFrailty.fit(**rows)``).

        Examples
        --------
        >>> from surpyval.recurrent import JointFrailty
        >>> model = JointFrailty.from_params(
        ...     (10.0, 1.2), (20.0, 1.5), 0.5, 1.0
        ... )
        >>> rows = model.random(3, end=15.0, random_state=0)
        >>> sorted(rows)
        ['c', 'i', 'terminal', 'x']
        """
        if not (isinstance(size, (int, np.integer)) and size >= 1):
            raise ValueError("'size' must be a whole number of units, >= 1.")
        ends = np.asarray(end, dtype=float).reshape(-1)
        if ends.size == 1:
            ends = np.repeat(ends, size)
        if (
            ends.size != size
            or not np.all(np.isfinite(ends))
            or np.any(ends <= 0)
        ):
            raise ValueError(
                "'end' must be one positive, finite age, or one per unit."
            )
        rng = as_generator(random_state)
        par = self._p()
        p = self.n_covariates
        Zr = self._rows(Z, size)
        rs, rk, ts, tk, theta, alpha = (float(v) for v in par[:6])
        er = np.exp(Zr @ par[6 : 6 + p]) if p else np.ones(size)
        et = np.exp(Zr @ par[6 + p :]) if p else np.ones(size)
        if theta <= 0:
            u = np.ones(size)
        elif self.family == "gamma":
            u = rng.gamma(1.0 / theta, theta, size)
        else:
            u = np.exp(rng.normal(0.0, np.sqrt(theta), size))
        e = rng.exponential(1.0, size)
        with np.errstate(divide="ignore", over="ignore"):
            T = ts * (e / (u**alpha * et)) ** (1.0 / tk)
        stop = np.minimum(T, ends)
        delta = (T <= ends).astype(int)
        rate = u * er
        xs, ids = [], []
        t = np.zeros(size)
        cum = np.zeros(size)
        active = np.flatnonzero(rate > 0)
        limit = 100000
        count = 0
        while active.size:
            draws = rng.exponential(1.0, active.size) / rate[active]
            if self.time_scale == "calendar":
                cum[active] += draws
                t[active] = rs * cum[active] ** (1.0 / rk)
            else:
                t[active] = t[active] + rs * draws ** (1.0 / rk)
            keep = t[active] < stop[active]
            xs.append(t[active][keep])
            ids.append(active[keep])
            active = active[keep]
            count += 1
            if count > limit:
                raise ValueError(
                    "A simulated unit had more than 100000 recurrent "
                    "events: the model's intensity is too high for its "
                    "'end'; give a shorter 'end'."
                )
        ev_x = np.concatenate(xs) if xs else np.zeros(0)
        ev_i = np.concatenate(ids) if ids else np.zeros(0, int)
        x = np.r_[ev_x, stop]
        i = np.r_[ev_i, np.arange(size)]
        c = np.r_[np.zeros(ev_x.size, int), np.ones(size, int)]
        term = np.r_[np.zeros(ev_x.size, int), delta]
        order = np.lexsort((c, x, i))
        out = {
            "x": x[order],
            "i": i[order],
            "c": c[order],
            "terminal": term[order],
        }
        if p:
            out["Z"] = Zr[i[order]]
        return out

    # -- serialisation -------------------------------------------------------
    def to_dict(self) -> dict:
        """
        A plain, JSON-safe dict of the model: its family, time scale,
        parameters and their names, the covariance (so that the Wald
        bounds survive), and each fitted unit's predicted frailty. The data
        are not stored: the likelihood, the likelihood-ratio bounds and
        the tests need a refit.

        Examples
        --------
        >>> from surpyval.recurrent import JointFrailty, JointFrailtyModel
        >>> model = JointFrailty.from_params(
        ...     (10.0, 1.2), (20.0, 1.5), 0.5, 1.0
        ... )
        >>> again = JointFrailtyModel.from_dict(model.to_dict())
        >>> float(again.sf(10.0)) == float(model.sf(10.0))
        True
        """
        from surpyval.serialisation import to_native

        frail = self.frailties
        out = {
            "model": "JointFrailtyModel",
            "family": self.family,
            "time_scale": self.time_scale,
            "params": np.asarray(self.params, dtype=float).tolist(),
            "n_covariates": int(self.n_covariates),
            "feature_names": (
                None
                if self.feature_names is None
                else [str(n) for n in self.feature_names]
            ),
            "held": (
                None
                if getattr(self, "_held", None) is None
                else [bool(v) for v in self._held]
            ),
            "covariance": self._covariance_entry(),
            "frailty_ids": (
                None if frail is None else [to_native(k) for k in frail]
            ),
            "frailty_values": (
                None if frail is None else [float(v) for v in frail.values()]
            ),
            **maximum_entry(self.maximum),
        }
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> JointFrailtyModel:
        """
        Rebuild a model from :meth:`to_dict`'s dict.

        Examples
        --------
        >>> from surpyval.recurrent import JointFrailty, JointFrailtyModel
        >>> model = JointFrailty.from_params(
        ...     (10.0, 1.2), (20.0, 1.5), 0.5, 1.0
        ... )
        >>> JointFrailtyModel.from_dict(model.to_dict()).theta
        0.5
        """
        require_model_tag(
            model_dict, "JointFrailtyModel", "a joint frailty model"
        )
        out = cls()
        check_option("family", model_dict["family"], FAMILIES)
        check_option("time_scale", model_dict["time_scale"], TIME_SCALES)
        out.family = str(model_dict["family"])
        out.time_scale = str(model_dict["time_scale"])
        out.params = np.asarray(dict_numbers(model_dict, "params"), float)
        out.n_covariates = int(model_dict["n_covariates"])
        if out.params.size != 6 + 2 * out.n_covariates:
            raise ValueError(
                "The serialised 'params' must have 6 + 2 * n_covariates "
                f"values; got {out.params.size}."
            )
        names = model_dict.get("feature_names")
        out.feature_names = None if names is None else list(names)
        held = model_dict.get("held")
        if held is not None:
            out._held = np.asarray(held, dtype=bool)
        out.maximum = restored_maximum(model_dict)
        ids = model_dict.get("frailty_ids")
        values = model_dict.get("frailty_values")
        out.frailties = (
            None
            if ids is None or values is None
            else dict(zip(ids, [float(v) for v in values]))
        )
        out.data = None
        out._restore_covariance(model_dict, np.nan_to_num(out.params, nan=0.0))
        if out._saved_covariance is not None:
            out._mle = out.params.copy()
        return out

    def __repr__(self) -> str:
        lines = [
            "Joint Frailty Model (recurrent and terminal events)",
            "=" * 51,
            f"Frailty             : {self.family}",
            f"Time scale          : {self.time_scale}",
            "Baselines           : Weibull (recurrent), Weibull (terminal)",
            "Parameters          :",
        ]
        for name, value in zip(self.parameter_names, self.params):
            lines.append(f"    {name:<22}: {value:.6g}")
        return "\n".join(lines) + notes_text(self) + "\n"
