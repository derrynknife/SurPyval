"""The likelihood-ratio bounds of a fitted parametric model.

``LikelihoodRatioMixin`` holds them for
:class:`~surpyval.univariate.parametric.parametric.Parametric`, which
inherits it: the profile likelihood and the parameters' intervals
(``param_cb(method="lr")``), and the bands and summary bounds of the
functions (``cb``, ``quantile_cb`` and ``mean_cb`` with ``method="lr"``).
The searches move each parameter in an unbounded coordinate
(``_LRCoord``), continue along the solved points (``_LRPath``) and walk
out to the critical value (``_lr_walk``); see #421, #519, #535 and #587.
"""

from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING, Any, Callable

import autograd.numpy as np
import numpy as onp
import numpy.typing as npt
from autograd import grad
from scipy.optimize import (
    brentq,
    minimize,
    minimize_scalar,
)
from scipy.special import expit
from scipy.special import ndtri as z

from surpyval.utils.validation import BOUNDS, CB_ON, check_option
from surpyval.utils.warnings import caller_stacklevel

if TYPE_CHECKING:
    from surpyval.utils.surpyval_data import SurpyvalData

# The likelihood-ratio searches (#421) move each parameter in a coordinate
# that is unbounded over its space (see ``_LRCoord``). These are the
# coordinates' ends: past them a parameter is no longer a double distinct
# from the edge of its space.
_LN_MAX = float(np.log(np.finfo(float).max))  # 709.78: exp overflows
_LN_TINY = float(np.log(np.finfo(float).tiny))  # -708.40: exp underflows
_FLOAT_MAX = float(np.finfo(float).max)
# The profile deviance is solved to about 1e-8 (the searches' tolerances);
# this is the slack the likelihood-ratio walk allows it (``_lr_walk``).
_LR_NOISE = 1e-6
# The scale of the deviance constraint in the scaled search for a
# function's extreme (``_PsiBoundSearch.extreme``): SLSQP holds it to its
# ftol of 1e-10, which is 1e-8 of deviance.
_LR_DEV_SCALE = 1e-2
# A deviance that is not finite, or no search reaching the target at all,
# is a failure; a target beyond a data-derived edge (the Uniform's) is not
# reachable, and reads as this deviance, above any critical value.
_LR_UNREACHABLE = 1e6
# The most likelihoods a model keeps for its likelihood-ratio searches
# (``_lr_raw_neg_ll``; a 50-point Weibull band asks for 16,000).
_LR_MEMO_SIZE = 100_000
# How many times a search that stops short of converging is continued
# (``_PsiBoundSearch.extreme_far``), and the relative steps of its
# central differences: ``_LR_FD_FINE`` where a continued search stalls.
_LR_CONTINUE = 10
_LR_FD_STEP = 1e-6
_LR_FD_FINE = 1e-8
# A continued search that moves psi out by less than this (relative to
# psi beyond 1) has stalled: a tenth of the hair ``checks_out`` tests.
_LR_GAIN = 1e-7


def _lr_hair(psi: float) -> float:
    """How far beyond an answer ``psi`` the extremality check looks
    (``_PsiBoundSearch.checks_out``): 1e-6 of it, beyond 1."""
    return 1e-6 * max(1.0, abs(psi))


def warn_unsettled(where: str) -> None:
    """Warn, once at the caller, that a likelihood-ratio bound is the
    most extreme point of the likelihood region its search found, but
    the search did not converge there (#601)."""
    warnings.warn(
        f"The likelihood-ratio bound {where} is the most extreme point of "
        "the likelihood region its search found, but the search did not "
        "converge there (the region runs out along a long, flat valley), "
        "so it may fall a little short of the region's extreme.",
        RuntimeWarning,
        stacklevel=caller_stacklevel(),
    )


class _LRCoord:
    """The coordinate a likelihood-ratio search moves one parameter in.

    Unbounded over the parameter's declared space ``(lo, hi)``: the log of
    its distance from a one-sided bound, the logit of its position within
    a finite interval, or the parameter itself when it has no bound -- as
    the fitter searches it, but a plain log throughout, which is the same
    at every scale. ``ends`` are the coordinate's values beyond which the
    parameter is no longer a double distinct from the edge of its space.
    """

    def __init__(self, lo: Any, hi: Any) -> None:
        self.lo = -np.inf if lo is None else float(lo)
        self.hi = np.inf if hi is None else float(hi)
        eps = np.finfo(float).eps

        def floor(edge: float) -> float:
            # The log of the smallest distance from ``edge`` that is still
            # a distinct double.
            return float(np.log(max(np.finfo(float).tiny, eps * abs(edge))))

        if np.isfinite(self.lo) and np.isfinite(self.hi):
            self.kind = "logit"
            log_width = float(np.log(self.hi - self.lo))
            # expit rounds to 1 above -log(eps): 36.04 at most.
            self.ends = (
                floor(self.lo) - log_width,
                min(-float(np.log(eps)), log_width - floor(self.hi)),
            )
        elif np.isfinite(self.lo):
            self.kind = "log"
            self.ends = (floor(self.lo), _LN_MAX)
        elif np.isfinite(self.hi):
            self.kind = "neglog"
            self.ends = (-_LN_MAX, -floor(self.hi))
        else:
            self.kind = "identity"
            self.ends = (-_FLOAT_MAX, _FLOAT_MAX)

    def to_u(self, theta: float) -> float:
        with np.errstate(all="ignore"):
            if self.kind == "log":
                return float(np.log(theta - self.lo))
            if self.kind == "neglog":
                return float(-np.log(self.hi - theta))
            if self.kind == "logit":
                return float(np.log(theta - self.lo) - np.log(self.hi - theta))
        return float(theta)

    def from_u(self, u: float) -> float:
        # Plain numpy where the exponential cannot overflow: the searches
        # map every point they evaluate, and autograd's wrapper and the
        # error state took longer than the exponential (#519).
        if self.kind == "log" and u < _LN_MAX:
            return float(self.lo + onp.exp(u))
        if self.kind == "neglog" and -u < _LN_MAX:
            return float(self.hi - onp.exp(-u))
        with np.errstate(all="ignore"):
            if self.kind == "log":
                return float(self.lo + np.exp(u))
            if self.kind == "neglog":
                return float(self.hi - np.exp(-u))
            if self.kind == "logit":
                return float(self.lo + (self.hi - self.lo) * expit(u))
        return float(u)

    def slope(self, theta: float) -> float:
        """d theta / d u at ``theta``."""
        if self.kind == "log":
            return float(theta - self.lo)
        if self.kind == "neglog":
            return float(self.hi - theta)
        if self.kind == "logit":
            return float(
                (theta - self.lo) * (self.hi - theta) / (self.hi - self.lo)
            )
        return 1.0

    def edge(self, direction: float) -> float:
        """The edge of the space the parameter goes to in ``direction``."""
        return self.hi if direction > 0 else self.lo


class _LRPath:
    """The points a profile search has solved, for continuation.

    A profile point is started from the solved point nearest to it and
    from the straight line through the two nearest, as well as from the
    estimate: the minimising nuisance parameters move along a curved
    valley (a NegativeBinomial ``p`` of ``1 - lambda / r`` as ``r`` grows;
    an ExpoWeibull ``alpha`` of 1e-28 at a ``beta`` of 0.05), which a
    search started from the estimate each time does not follow.
    """

    def __init__(self) -> None:
        self.w: list[float] = []
        self.u: list[npt.NDArray] = []
        # The minimum at each point (a negative log-likelihood).
        self.f: list[float] = []

    def add(self, w: float, u: npt.NDArray, f: float = np.nan) -> None:
        if np.isfinite(w) and np.all(np.isfinite(u)):
            self.w.append(float(w))
            self.u.append(np.array(u, dtype=float))
            self.f.append(float(f))

    def starts(self, w: float) -> list[npt.NDArray]:
        if not self.w:
            return []
        order = np.argsort(np.abs(np.asarray(self.w) - w))
        near = self.u[order[0]]
        out = [near]
        if len(order) > 1:
            w0, w1 = self.w[order[0]], self.w[order[1]]
            if w0 != w1:
                slope = (near - self.u[order[1]]) / (w0 - w1)
                out.append(near + slope * (w - w0))
        return out


def _lr_walk(
    deviance: Callable[[float], float],
    w_hat: float,
    step: float,
    direction: float,
    crit: float,
    end: float,
    stop: float | None = None,
    d_hat: float = 0.0,
) -> tuple[str, float]:
    """Walk out from ``w_hat`` to where ``deviance`` first reaches ``crit``.

    Steps of ``step``, growing by 1.6 each time, bracket the crossing; the
    bracket is walked again in eight equal steps (a long step can lose the
    valley the profile's minimum follows, and overstate the deviance: an
    ExpoWeibull ``mu`` bound of 1e-3 where the deviance falls to 0.3 at
    1e-4), and ``brentq`` solves the first of them to reach ``crit``.
    Returns ``("root", w)``; ``("stop", stop)`` when the walk reaches
    ``stop`` (a data-derived limit, beyond which the likelihood is 0)
    below ``crit``; ``("edge", end)`` when the deviance stays below
    ``crit`` to ``end`` (the end of the coordinate) or levels off below
    it; and ``("fail", nan)`` where the deviance is not finite, or the
    walk runs out of steps. ``d_hat`` is the deviance at ``w_hat``: 0 at
    the estimate.

    *Levelling off.* The deviance of a parameter that tends to a limiting
    model at the edge of its space (a NegativeBinomial ``r`` to infinity,
    the shifted Poisson) converges to that model's deviance. Where that is
    below ``crit`` no value of the parameter out to the edge is excluded,
    and the bound is the edge. The walk says so once, over its last three
    steps, the deviance has risen ever more slowly -- each rise per unit
    of ``w`` (a fall counting as no rise) no more than the one before, to
    within ``_LR_NOISE`` -- and a straight line from the latest point, at
    the latest of those slopes, stays below ``crit`` all the way to
    ``end``. A deviance rising ever more slowly lies below that line, so
    it cannot reach ``crit`` before the end of the representable range.
    A deviance rising at a steady or growing rate (a quadratic one, the
    usual case) fails the first test, and one rising at a slowing rate
    that would still reach ``crit`` before ``end`` fails the second: an
    ExpoWeibull ``alpha`` whose deviance rises by 0.024 per unit of
    ``log(alpha)`` at ``alpha`` = 7e-11 goes on to cross 3.84 at 5e-28,
    and the walk finds it there. The one profile the test misreads is one
    that falls for three steps and later climbs back above ``crit`` (a
    second, lower mode of the likelihood further out); the bound is then
    the edge, wider than it need be, never narrower.
    """
    ws, ds = [w_hat], [d_hat]
    # brentq starts from the bracket's ends, which the walk has solved.
    solved: dict[float, float] = {float(w_hat): float(d_hat)}

    def dev_at(v: float) -> float:
        v = float(v)
        if v not in solved:
            solved[v] = deviance(v)
        return solved[v]

    def f(v: float) -> float:
        d = dev_at(v)
        return (_LR_UNREACHABLE if d == np.inf else d) - crit

    def levels_off() -> bool:
        if len(ds) < 4:
            return False
        with np.errstate(all="ignore"):
            h = np.abs(np.diff(ws[-4:]))
            rate = np.maximum(np.diff(ds[-4:]), 0.0) / h
            slowing = np.all(rate[1:] <= rate[:-1] + _LR_NOISE / h[1:])
            reach = ds[-1] + rate[-1] * abs(end - ws[-1]) + _LR_NOISE
        return bool(slowing and reach < crit)

    for _ in range(80):
        w = ws[-1] + direction * step
        step *= 1.6
        at_stop = stop is not None and direction * (w - stop) >= 0
        at_end = direction * (w - end) >= 0
        if stop is not None and at_stop and direction * (stop - end) <= 0:
            # The data's limit comes before the end of the coordinate.
            w, at_end = float(stop), False
        elif at_end:
            w, at_stop = float(end), False
        dev = dev_at(w)
        if dev >= crit:
            # Walk the bracket again, in eight steps from its near end.
            for v in np.linspace(ws[-1], w, 9)[1:]:
                dev = dev_at(v)
                if dev >= crit:
                    if f(ws[-1]) >= 0:
                        # A start on the boundary (``d_hat`` at crit).
                        return "root", float(ws[-1])
                    a, b = sorted((ws[-1], v))
                    try:
                        root = brentq(f, a, b, xtol=1e-9, rtol=1e-9)
                    except ValueError:
                        return "fail", np.nan
                    return "root", float(root)
                if not np.isfinite(dev):
                    return "fail", np.nan
                if v != w:
                    ws.append(float(v))
                    ds.append(dev)
                    if levels_off():
                        return "edge", float(end)
            # The far end, solved again from the steps before it, is
            # below crit after all: the walk goes on from it.
        if not np.isfinite(dev):
            return "fail", np.nan
        if at_stop:
            return "stop", float(w)
        if at_end:
            return "edge", float(end)
        ws.append(w)
        ds.append(dev)
        if levels_off():
            return "edge", float(end)
    return "fail", np.nan


def central_gradient(
    f: Callable[..., Any], u: npt.NDArray, rel_step: float = 1e-6
) -> npt.NDArray:
    """Central-difference gradient of ``f`` at ``u``, in steps of
    ``rel_step`` (relative to each coordinate beyond 1)."""
    grad = np.empty(len(u))
    for j in range(len(u)):
        h = rel_step * max(1.0, abs(u[j]))
        up, down = np.array(u, dtype=float), np.array(u, dtype=float)
        up[j] += h
        down[j] -= h
        grad[j] = (f(up) - f(down)) / (2 * h)
    return grad


def _wald_sd(model: Any, coords: list[_LRCoord], free: list[int]) -> Any:
    """The Wald standard errors of the core parameters ``free`` in their
    search coordinates, or ``None`` without a usable covariance.

    The searches run in the coordinates divided by these: SLSQP and
    L-BFGS-B start from the identity as the likelihood's curvature, which
    is near its curvature there, where in the raw coordinates it can be
    out by the sample size (a Weibull's ``log alpha`` on 1000 units: 4,000
    against 1). A constrained search that took 40 to 90 evaluations from
    a point beside its answer takes 5.
    """
    hess_inv = getattr(model, "hess_inv", None)
    if hess_inv is None or np.ndim(hess_inv) != 2:
        return None
    theta = np.asarray(model.params, dtype=float)
    slopes = np.array([coords[j].slope(theta[j]) for j in free], dtype=float)
    with np.errstate(all="ignore"):
        sd = np.sqrt(np.diag(np.asarray(hess_inv, dtype=float))[free])
        sd = sd / np.abs(slopes)
    if not (np.all(np.isfinite(sd)) and np.all(sd > 0)):
        return None
    return sd


def _scaled_bounds(
    bounds: list[tuple[Any, Any]] | None,
    origin: npt.NDArray,
    scale: npt.NDArray,
) -> list[tuple[Any, Any]] | None:
    """``bounds`` on ``u`` as bounds on ``z``, ``u = origin + scale z``."""
    if bounds is None:
        return None
    return [
        (
            None if lo is None else (lo - origin[k]) / scale[k],
            None if hi is None else (hi - origin[k]) / scale[k],
        )
        for k, (lo, hi) in enumerate(bounds)
    ]


def _unguarded(dist: Any, name: str) -> Callable[..., Any]:
    """The distribution's function ``name`` without the wrappers of
    ``parametric_fitter`` (``_array_inputs`` and ``_support_guarded``),
    unbound: called as ``f(dist, x, *params)``."""
    fn = getattr(type(dist), name)
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _lean_neg_ll(dist: Any, lean: tuple, theta: npt.NDArray) -> float:
    """The negative log-likelihood of ``Parametric._lr_lean_data``'s data
    ``lean`` at core parameters ``theta``: the terms of
    ``ParametricFitter._log_likelihood`` for a distribution without an
    offset, zero inflation or a limited failure population, as they
    compute them (a term with no data is exactly 0 there and is left
    out), but without the guards, which these data pass."""
    return float(-_lean_log_likelihood(dist, lean, theta))


def _lean_log_likelihood(dist: Any, lean: tuple, theta: Any) -> Any:
    """The log-likelihood of ``_lean_neg_ll``, in a form autograd can
    follow (``_lean_ll_gradient``)."""
    observed, right, left, interval, truncated, extra = lean
    params = tuple(theta)
    ll: Any = 0
    if observed is not None:
        x, n = observed
        ll = (n * _unguarded(dist, "log_df")(dist, x, *params)).sum()
    if right is not None:
        x, n = right
        ll = ll + np.sum(n * _unguarded(dist, "log_sf")(dist, x, *params))
    if left is not None:
        x, n = left
        ll = ll + np.sum(n * _unguarded(dist, "log_ff")(dist, x, *params))
    if interval is not None or truncated is not None:
        fns = tuple(
            functools.partial(_unguarded(dist, name), dist)
            for name in ("ff", "log_sf", "log_ff")
        )
    if interval is not None:
        windows, n = interval
        ll = ll + dist._window_log_likelihood(windows, n, params, extra, fns)
    if truncated is not None:
        windows, n = truncated
        ll = ll - dist._window_log_likelihood(windows, n, params, extra, fns)
    return ll


#: The gradient of ``_lean_log_likelihood`` in ``theta`` (autograd's).
_lean_ll_gradient = grad(_lean_log_likelihood, 2)


class _PsiBoundSearch:
    """The search for the likelihood-ratio bounds on one function of the
    free core parameters, ``psi_of(theta)``, for
    ``LikelihoodRatioMixin._cb_lr_psi_bounds``.

    A bound is the extreme of psi over the likelihood region
    ``{deviance <= crit}``, searched in ``box`` in the search coordinates
    ``u`` of the free parameters (see ``_LRCoord``). Each side is tried
    in turn, cheapest first, until an answer checks out:

    1. ``from_trace``: from the traced point of a two-parameter region
       where psi is most extreme (``_lr_trace``), or from where the
       neighbouring time's bound was found (``hints``);
    2. ``ladder`` then ``direct`` from the estimate and the walks' tips:
       the extreme sought directly (SLSQP), checked against every point
       of the region known;
    3. ``direct`` again from the known points beyond the answer;
    4. ``walk_profile``: the profile of psi walked out to ``crit``.

    The points of the region found on the way are shared by the steps:
    ``known`` (with their psi), ``reached`` (where each direct search
    ended, inside the region or not) and ``walks`` (the parameters' walks'
    points inside the region).
    """

    def __init__(
        self,
        model: Any,
        psi_of: Callable[[npt.NDArray], float],
        free: list[int],
        crit: float,
        ends: tuple[float, float],
        box: list[tuple[Any, Any]],
    ) -> None:
        self.model = model
        self.psi_of = psi_of
        self.free = free
        self.crit = crit
        self.ends = ends
        self.box = box
        self.theta_hat = np.array(model.params, dtype=float)
        self.nll_hat = model._lr_neg_ll(self.theta_hat)
        self.coords, _ = model._lr_coords()
        self.free_coords = [self.coords[j] for j in free]
        self.bounds = box if any(b != (None, None) for b in box) else None
        self.u_hat = np.array(
            [self.coords[j].to_u(self.theta_hat[j]) for j in free]
        )
        self.psi_hat = np.nan
        self.u_start: npt.NDArray | None = None
        self.step = np.nan
        # The Cholesky factor of the Wald covariance in the search
        # coordinates: the level ladder also searches in the coordinates
        # it whitens (``ladder``).
        self.chol: npt.NDArray | None = None
        self.known: list[tuple[float, npt.NDArray]] = []
        self.reached: list[tuple[float, npt.NDArray]] = []
        self.walks: list[list[tuple[float, npt.NDArray]]] = []
        # The parameters' walks' points, one list per parameter and side.
        self.seeds: list[list[npt.NDArray]] = []
        self.traced: list[tuple[float, npt.NDArray]] | None = None
        # The points the level ladders found (``ladder``).
        self.ladder_ids: set[int] = set()
        # psi at each point asked for (``psi_u``).
        self.psi_kept: dict[bytes, float] = {}
        # The searches' scale (``_wald_sd``): ``solve`` and the search
        # from the trace run in z, u = u_hat + scale z.
        self.scale = _wald_sd(model, self.coords, free)
        # Where each side's bound was found at the neighbouring time of a
        # band (``from_trace``), and where it is found here.
        self.hints: dict[float, npt.NDArray] = {}
        self.answers: dict[float, npt.NDArray] = {}
        self.last_status = 0
        # A point of the region beyond the answer ``checks_out`` found.
        self.beyond: npt.NDArray | None = None
        # Whether the last ``extreme_far`` converged; the answers whose
        # search did not; and the sides whose bound is such an answer.
        self.converged = True
        self.fd_step = _LR_FD_STEP
        self.unsettled: set[float] = set()
        self.unsettled_sides: set[float] = set()
        # The most extreme answer that has checked out on each side
        # (``checks_out``)
        self.checked: dict[float, float] = {}

    # -- the functions of the search coordinates --------------------------
    def theta_of(self, u: npt.NDArray) -> npt.NDArray:
        theta = self.theta_hat.copy()
        theta[self.free] = [c.from_u(v) for c, v in zip(self.free_coords, u)]
        return theta

    def nll_of(self, u: npt.NDArray) -> float:
        nll = self.model._lr_neg_ll(self.theta_of(u))
        return nll if np.isfinite(nll) else np.inf

    def psi_u(self, u: npt.NDArray) -> float:
        if not onp.isfinite(u).all():
            return np.nan
        # Kept by the point's bytes, as the likelihood is: SLSQP asks
        # for the constraint again where it has just evaluated it.
        key = np.asarray(u, dtype=float).tobytes()
        if key not in self.psi_kept:
            # Held to the ends of its scale where the function reaches
            # the edge of its range (a density that underflows to 0), so
            # that a search can still step there.
            # (``min`` and ``max``, as ``np.clip``, keep a nan)
            low, high = self.ends
            psi = float(self.psi_of(self.theta_of(u)))
            self.psi_kept[key] = min(max(psi, low), high)
        return self.psi_kept[key]

    def dev_u(self, u: npt.NDArray) -> float:
        return 2.0 * (self.nll_of(u) - self.nll_hat)

    def dev_grad_u(self, u: npt.NDArray) -> npt.NDArray | None:
        """The gradient of ``dev_u`` at ``u``, autograd's through the
        lean likelihood (``_lean_ll_gradient``); ``None`` without one, or
        where it is not finite."""
        lean = self.model._lr_lean_data()
        if lean is None:
            return None
        theta = self.theta_of(u)
        try:
            with np.errstate(all="ignore"):
                g = np.asarray(
                    _lean_ll_gradient(self.model.dist, lean, theta),
                    dtype=float,
                )
        except (ValueError, TypeError, ArithmeticError):
            return None
        slopes = np.array(
            [c.slope(theta[j]) for c, j in zip(self.free_coords, self.free)]
        )
        out = -2.0 * g[self.free] * slopes
        return out if np.all(np.isfinite(out)) else None

    # -- the whole search ---------------------------------------------------
    def run(
        self,
        want_lower: bool,
        want_upper: bool,
        seeds: list[list[npt.NDArray]],
        trace: list[npt.NDArray] | None,
    ) -> tuple[float, float]:
        """``(lower, upper)``; see ``_cb_lr_psi_bounds``."""
        psi_hat = self.psi_of(self.theta_of(self.u_hat))
        if not np.isfinite(psi_hat):
            # The function is at the edge of its range at the estimate (a
            # rate of 0 below the support): the bounds are that value.
            return psi_hat, psi_hat
        self.psi_hat = psi_hat

        # The search is checked where its answer is known, at the
        # estimate: if it fails there, it cannot be trusted anywhere.
        _, u_start = self.solve(psi_hat, [self.u_hat])
        if u_start is None:
            return np.nan, np.nan
        self.u_start = u_start

        self._wald_scale()
        self.known = [(psi_hat, u_start)]
        self.seeds = seeds
        self._take_walks(seeds)
        if trace is not None:
            traced = [(self.psi_u(u), u) for u in trace]
            if all(np.isfinite(k[0]) for k in traced):
                self.traced = traced

        lower = self.solve_side(-1.0) if want_lower else np.nan
        upper = self.solve_side(1.0) if want_upper else np.nan
        return lower, upper

    def _wald_scale(self) -> None:
        """The first step (the Wald standard error on the psi scale) and
        the Cholesky factor of the Wald covariance."""
        hess_inv = getattr(self.model, "hess_inv", None)
        step = np.nan
        if hess_inv is not None and np.ndim(hess_inv) == 2:
            slopes = np.array(
                [self.coords[j].slope(self.theta_hat[j]) for j in self.free],
                dtype=float,
            )
            cov_u = np.asarray(hess_inv, dtype=float)[
                np.ix_(self.free, self.free)
            ]
            cov_u = cov_u / np.outer(slopes, slopes)
            grad = central_gradient(self.psi_u, self.u_hat)
            step = float(np.sqrt(grad @ cov_u @ grad))
            try:
                chol = np.linalg.cholesky(cov_u)
            except np.linalg.LinAlgError:
                chol = None
            if chol is not None and not np.all(np.isfinite(chol)):
                chol = None
            self.chol = chol
        if not (np.isfinite(step) and step > 0):
            step = 1.0
        self.step = step

    def _take_walks(self, seeds: list[list[npt.NDArray]]) -> None:
        """The points of the parameters' walks inside the region."""
        for group in seeds:
            walk = []
            for seed in group:
                seed = self.model._lr_start(seed, self.box)
                psi_seed = self.psi_u(seed)
                if np.isfinite(psi_seed) and self.dev_u(seed) <= self.crit:
                    walk.append((psi_seed, seed))
            self.known.extend(walk)
            if walk:
                self.walks.append(walk)

    # -- the searches ---------------------------------------------------
    def solve(
        self, target: float, starts: list[npt.NDArray]
    ) -> tuple[float, npt.NDArray | None]:
        """The least negative log-likelihood with psi at ``target``.

        Searched in the coordinates scaled by the Wald standard errors
        (``_wald_sd``), where SLSQP's first guess at the curvature is
        near right, and in the search coordinates without a covariance.
        """
        best, best_u = np.inf, None
        if self.scale is None:
            origin, scale = np.zeros(len(self.u_hat)), np.ones(len(self.u_hat))
        else:
            origin, scale = self.u_hat, self.scale

        def psi_z(z: npt.NDArray) -> float:
            return self.psi_u(origin + scale * z)

        def nll_z(z: npt.NDArray) -> float:
            return self.nll_of(origin + scale * z)

        constraint = {
            "type": "eq",
            "fun": lambda z: psi_z(z) - target,
            "jac": lambda z: central_gradient(psi_z, z),
        }
        tol = 1e-7 * max(1.0, abs(target))
        for x0 in starts:
            z0 = (self.model._lr_start(x0, self.box) - origin) / scale
            # Continued from where SLSQP stops short of converging just
            # outside the region (within 0.05 of deviance), for as long
            # as the likelihood rises there: whether a point is in the
            # region turns on it (#601).
            for _ in range(_LR_CONTINUE):
                try:
                    res = minimize(
                        nll_z,
                        z0,
                        method="SLSQP",
                        jac=lambda z: central_gradient(nll_z, z),
                        bounds=_scaled_bounds(self.bounds, origin, scale),
                        constraints=[constraint],
                        options={"ftol": 1e-10, "maxiter": 60},
                    )
                except (ValueError, np.linalg.LinAlgError):
                    break
                if not np.all(np.isfinite(res.x)):
                    break
                u = origin + scale * np.asarray(res.x)
                gap = abs(self.psi_u(u) - target)
                nll = self.nll_of(u)
                if not (gap <= tol and nll < best):
                    break
                best, best_u = nll, u
                excess = 2.0 * (nll - self.nll_hat) - self.crit
                if res.status == 0 or not 0.0 <= excess < 0.05:
                    break
                z0 = np.asarray(res.x)
        return best, best_u

    def extreme(
        self,
        direction: float,
        start: npt.NDArray,
        level: float,
        whiten: bool = False,
        scaled: bool = False,
        face: tuple[int, float] | None = None,
    ) -> npt.NDArray | None:
        """Where SLSQP stops in its search for the extreme of psi over
        the region {deviance <= level}; ``None`` where it fails. Its
        status is kept in ``last_status``.

        ``whiten``: searched in z, u = u_hat + chol z (held in the box),
        where the region is near a ball about the estimate. ``scaled``:
        searched in z, u = u_hat + scale z (``_wald_sd``), where the
        box is a box. ``face``, ``(k, end)``: over the face of the box
        ``u[k] = end`` only (``_face_coords``).
        """
        u_hat, box = self.u_hat, self.box
        if face is not None:
            to_u, z0, z_bounds = self._face_coords(face, start)
        elif whiten:
            if self.chol is None:
                return None
            L = self.chol

            def to_u(z: npt.NDArray) -> npt.NDArray:
                return self.model._lr_start(u_hat + L @ z, box)

            z0 = np.linalg.solve(L, np.asarray(start) - u_hat)
            z_bounds = None
        elif scaled and self.scale is not None:
            scale = self.scale

            def to_u(z: npt.NDArray) -> npt.NDArray:
                return u_hat + scale * z

            z0 = (np.asarray(start) - u_hat) / scale
            z_bounds = _scaled_bounds(self.bounds, u_hat, scale)
        else:

            def to_u(z: npt.NDArray) -> npt.NDArray:
                return z

            z0, z_bounds = start, self.bounds

        def f(z: npt.NDArray) -> float:
            return self.psi_u(to_u(z))

        def g(z: npt.NDArray) -> float:
            return self.dev_u(to_u(z))

        # SLSQP holds a constraint to its ftol. Held so to the deviance's
        # last digits, the scaled search cycled up to 90 times at a point
        # outside the region by 5e-10, trading that against psi (a
        # Weibull band on 1000 units); scaled, the deviance is held to
        # 1e-8, as the walks solve it, and ``from_trace`` takes a point
        # outside back onto the boundary.
        c = _LR_DEV_SCALE if scaled or face is not None else 1.0
        h = self.fd_step
        # Where a continued search has stalled (``extreme_far``), the
        # deviance's gradient is autograd's, where it can be had: the
        # region narrows down its valleys faster than differences resolve
        # (to_u is linear here: u = to_u(0) + J z)
        exact = h == _LR_FD_FINE and not whiten
        n_z = len(z0)
        zero = to_u(np.zeros(n_z))
        J = np.column_stack([to_u(e) - zero for e in np.eye(n_z)])

        def g_jac(z: npt.NDArray) -> npt.NDArray:
            grad_u = self.dev_grad_u(to_u(z)) if exact else None
            if grad_u is None:
                return central_gradient(g, z, h)
            return J.T @ grad_u

        try:
            res = minimize(
                lambda z: -direction * f(z),
                z0,
                method="SLSQP",
                jac=lambda z: -direction * central_gradient(f, z, h),
                bounds=z_bounds,
                constraints=[
                    {
                        "type": "ineq",
                        "fun": lambda z: c * (level - g(z)),
                        "jac": lambda z: -c * g_jac(z),
                    }
                ],
                options={"ftol": 1e-10, "maxiter": 100},
            )
        except (ValueError, np.linalg.LinAlgError):
            return None
        if not np.all(np.isfinite(res.x)):
            return None
        self.last_status = int(res.status)
        return to_u(np.asarray(res.x))

    def _face_coords(
        self, face: tuple[int, float], start: npt.NDArray
    ) -> tuple[
        Callable[[npt.NDArray], npt.NDArray],
        npt.NDArray,
        list[tuple[Any, Any]] | None,
    ]:
        """The coordinates of a search over the face ``u[k] = end`` of
        the box, from ``start`` moved onto it: ``(to_u, z0, z_bounds)``,
        the other coordinates scaled by the Wald standard errors
        (``_wald_sd``) about ``start``."""
        k, end = face
        others = [j for j in range(len(self.u_hat)) if j != k]
        scale = (
            np.ones(len(others))
            if self.scale is None
            else np.asarray(self.scale)[others]
        )
        origin = self.model._lr_start(np.asarray(start, dtype=float), self.box)
        origin[k] = end

        def to_u(z: npt.NDArray) -> npt.NDArray:
            u = origin.copy()
            u[others] = origin[others] + scale * z
            return u

        z_bounds = _scaled_bounds(
            [self.box[j] for j in others], origin[others], scale
        )
        return to_u, np.zeros(len(others)), z_bounds

    def extreme_far(
        self,
        direction: float,
        start: npt.NDArray,
        level: float,
        whiten: bool = False,
        scaled: bool = False,
        face: tuple[int, float] | None = None,
    ) -> npt.NDArray | None:
        """``extreme`` from ``start``, a point of the region, continued
        where SLSQP stops short of converging (its iteration limit, a
        failed line search): from where it stopped, or from where the
        line to there from its start meets the boundary if that is
        outside the region, for as long as psi moves out. ``None`` where
        it fails.

        On a long, flat valley of the region (an ExpoWeibull's as beta
        -> inf, #601) SLSQP spent its 100 iterations creeping along it;
        stopped there, outside the region by 7e-5 of deviance, its point
        was dropped, and a bound 2.7% short of the extreme taken. Where
        a continued search stalls, it goes on with gradients in steps of
        ``_LR_FD_FINE``: the valley narrows as it goes (the ExpoWeibull's
        ``alpha`` closes on the largest observation to within 1 /
        ``beta``), and steps of 1e-6 stop resolving it at ``beta`` ~ 1e6,
        where the deviance is still 1e-4 above its limit (a bound 1e-6
        to 1e-5 short); steps of 1e-8 follow it to 1e-8.
        """
        self.converged = True
        x = self.extreme(direction, start, level, whiten, scaled, face)
        if x is None or self.last_status == 0:
            return x
        # The most extreme point of the region the search has reached
        u_from = np.asarray(start, dtype=float)

        def inside(x: npt.NDArray | None) -> npt.NDArray:
            # x, or the nearest point to it on the boundary (or where the
            # line to it from u_from meets the boundary), if that is
            # further out than u_from; otherwise u_from.
            if x is not None and not self.dev_u(x) <= level + _LR_NOISE:
                back = self.back_onto_boundary(x, level)
                if back is None:
                    back = self.onto_boundary(u_from, x, level)
                x = back
            if (
                x is None
                or direction * (self.psi_u(x) - self.psi_u(u_from)) < 0
            ):
                return u_from
            return x

        try:
            for _ in range(_LR_CONTINUE):
                # (SLSQP failed, rather than ran out of iterations)
                failed = self.last_status != 9
                x = inside(x)
                gain = direction * (self.psi_u(x) - self.psi_u(u_from))
                stalled = not gain > _LR_GAIN * max(1.0, abs(self.psi_u(x)))
                if (failed or stalled) and self.fd_step != _LR_FD_FINE:
                    # At a vertex of the box (a Uniform's support edge),
                    # or where the differences no longer resolve the
                    # valley: on in the finer gradients.
                    self.fd_step = _LR_FD_FINE
                elif stalled:
                    return self.out_to_boundary(direction, x, level)
                u_from = x
                x = self.extreme(direction, x, level, whiten, scaled, face)
                if x is None or self.last_status == 0:
                    return self.out_to_boundary(direction, inside(x), level)
            # Still moving out after every continuation: not settled.
            self.converged = False
            return self.out_to_boundary(direction, inside(x), level)
        finally:
            self.fd_step = _LR_FD_STEP

    def direct(
        self,
        direction: float,
        start: npt.NDArray,
        scaled: bool = False,
        persist: bool = True,
    ) -> float | None:
        """The extreme of psi over the region, sought directly (SLSQP;
        ``scaled`` as ``extreme``), when it checks out (``checks_out``);
        otherwise ``None``.

        ``persist``: continued where it stops short (``extreme_far``),
        and from a point further out that the check finds. Otherwise one
        search, whose end, if it stops outside the region, is kept
        (taken onto the boundary) as a point the bound must reach
        (``keep``)."""
        if not persist:
            x = self.extreme(direction, start, self.crit, scaled=scaled)
            if x is None:
                return None
            if not self.dev_u(x) <= self.crit + _LR_NOISE:
                self.keep(start, x)
                return None
            return self.checks_out(direction, x)
        x = self.extreme_far(direction, start, self.crit, scaled=scaled)
        for _ in range(3):
            if x is None:
                return None
            quick = self.checks_out(direction, x)
            if quick is not None or self.beyond is None:
                return quick
            # The check found a point of the region further out: the
            # search stopped short of the extreme, and goes on from there.
            u_from = self.beyond
            x = self.extreme_far(direction, u_from, self.crit, scaled=scaled)
            if (
                x is None
                or not direction * (self.psi_u(x) - self.psi_u(u_from)) >= 0
            ):
                x = u_from
        return None

    def checks_out(self, direction: float, x: npt.NDArray) -> float | None:
        """psi at ``x``, where a search for its extreme ended, if that is
        the extreme; otherwise ``None``.

        It checks out when its deviance is at crit or below, and none
        with psi a hair further out (1e-6 of it) is at crit or below.
        That is where the profile of psi crosses crit, which the walk
        would find at many times the cost. The check searches from the
        estimate, and from beside the answer: the answer moved along the
        gradient of psi to the target, where the least likelihood with
        psi there is a step away. (From the answer itself SLSQP spent 40
        to 90 evaluations crawling to the target, its line search
        trading the gap against the likelihood.)

        An answer within that hair of one already checked out on the same
        side (``checked``) is the same extreme, found again from another
        start: it is taken without a check, as the more extreme of the
        two. Each side's searches from the estimate, the walks' tips and
        the edge valleys mostly end on one extreme, and the checks were
        two thirds of a NegativeBinomial band's likelihood evaluations
        (#609).
        """
        crit, psi_hat = self.crit, self.psi_hat
        self.beyond = None
        psi_star = self.psi_u(x)
        if np.isfinite(psi_star):
            self.reached.append((psi_star, x))
        if not (
            np.isfinite(psi_star)
            and self.dev_u(x) <= crit + _LR_NOISE
            and direction * (psi_star - psi_hat) >= 0
        ):
            return None
        self.known.append((psi_star, x))
        done = self.checked.get(direction)
        if done is not None and abs(psi_star - done) <= _lr_hair(done):
            if direction * (psi_star - done) <= 0:
                return done
        else:
            beyond = psi_star + direction * _lr_hair(psi_star)
            nll, u = self.solve(beyond, [self._towards(x, beyond), self.u_hat])
            if u is not None and 2.0 * (nll - self.nll_hat) < crit:
                # (kept, for the search to go on from: ``direct``)
                self.beyond = u
                self.known.append((self.psi_u(u), u))
                return None
        if not self.converged:
            self.unsettled.add(psi_star)
        if done is None or direction * (psi_star - done) > 0:
            self.checked[direction] = psi_star
        return psi_star

    def _towards(self, x: npt.NDArray, target: float) -> npt.NDArray:
        """``x`` moved along the gradient of psi to where its linear
        approximation is ``target``; ``x`` where that cannot be found."""
        grad = central_gradient(self.psi_u, x)
        size = float(grad @ grad)
        if not (np.isfinite(size) and size > 0):
            return x
        moved = x + (target - self.psi_u(x)) * grad / size
        return moved if np.all(np.isfinite(moved)) else x

    def from_trace(self, direction: float) -> float | None:
        """The search from the traced point of the region where psi is
        most extreme, taken when it is at least as far out as every
        traced point and every point known (see ``_lr_trace``).

        In a band, it starts instead from where the bound was found at
        the neighbouring time (``hints``) when psi is at least as far out
        there: the extreme moves little from one time to the next, and
        the search from it takes fewer steps. The answer is the same
        extreme, and is taken on the same terms.
        """
        if self.traced is None:
            return None
        u_hat, crit = self.u_hat, self.crit
        top_psi, top_u = max(self.traced, key=lambda k: direction * k[0])
        start = top_u
        hint = self.hints.get(direction)
        if hint is not None:
            psi_hint = self.psi_u(hint)
            if np.isfinite(psi_hint) and direction * (psi_hint - top_psi) >= 0:
                start = hint
        quick = self.direct(
            direction, self.model._lr_start(start, self.box), scaled=True
        )
        if quick is None:
            return None
        # SLSQP may stop outside the region by its tolerance (a
        # deviance up to _LR_NOISE over crit, which ``direct`` allows):
        # psi is then taken where the ray to that point meets the
        # boundary. Where psi is steep that is the difference between
        # 2e-8 and 1e-6 of the bound (a LogNormal hf(0.5) lower bound
        # against its boundary's minimum, found by brute force).
        end = self.reached[-1][1]
        if self.dev_u(end) > crit:
            ray = end - u_hat
            r = brentq(
                lambda r: self.dev_u(u_hat + r * ray) - crit,
                0.0,
                1.0,
                xtol=1e-14,
                rtol=1e-14,
            )
            quick = self.psi_u(u_hat + r * ray)
            self.known[-1] = (quick, u_hat + r * ray)
        far = max(direction * k[0] for k in self.known)
        slack = 1e-9 * max(1.0, abs(top_psi))
        if direction * quick >= max(far, direction * top_psi - slack):
            self.answers[direction] = self.known[-1][1]
            return quick
        return None

    def ladder(self, direction: float) -> float:
        """The extreme of psi over the narrower regions {deviance <= f
        crit}, f in _LR_LADDER, and then over this one, each sought from
        the one before (continuation in the level): points of this region
        that the bound must reach, as far as the farthest of them
        (``-inf`` if none).

        A region grows with its level, and the extreme moves out with
        it; followed so, the search stays in its valley, where one from
        the estimate or from a walk's tip can stop on a nearer local
        extreme of a long, curved region (an ExpoWeibull hf(13) lower
        bound of 0.1046 at 99%, though 0.1017 is in the 95% region,
        #535). Each rung is searched in the search coordinates and in
        those the Wald covariance whitens (SLSQP is sensitive to the
        scaling, and in a narrow valley neither alone is reliable: the
        99% qf(0.95) upper bound stops at 25.3 in one and at 35.3 in the
        other, and at 82.3 taking the better of the two at each rung, of
        the 87.0 that tracing the region slice by slice finds), and the
        more extreme answer taken, or the rung's start where neither is
        further out. An answer
        outside its region (SLSQP's tolerance, or its iteration limit) is
        taken back onto the boundary along the line from the rung's
        start.
        """
        assert self.u_start is not None
        u_step, psi_step = self.u_start, self.psi_hat
        far = -np.inf
        for frac in (*self.model._LR_LADDER, 1.0):
            level = frac * self.crit
            best_x, best_psi = None, psi_step
            for whiten in (False, True):
                x = self.extreme(direction, u_step, level, whiten)
                if x is None:
                    continue
                if not self.dev_u(x) <= level:
                    x = self.onto_boundary(u_step, x, level)
                    if x is None:
                        continue
                psi_x = self.psi_u(x)
                if np.isfinite(psi_x) and (direction * (psi_x - best_psi) > 0):
                    best_x, best_psi = x, psi_x
            if best_x is None:
                continue
            self.known.append((best_psi, best_x))
            self.ladder_ids.add(id(best_x))
            far = max(far, direction * best_psi)
            u_step, psi_step = best_x, best_psi
        return far

    def onto_boundary(
        self, inside: npt.NDArray, outside: npt.NDArray, level: float
    ) -> npt.NDArray | None:
        """Where the line from ``inside`` (deviance <= level) to
        ``outside`` meets {deviance = level}; ``None`` if it cannot be
        found."""
        if not self.dev_u(inside) <= level:
            return None
        ray = outside - inside

        def excess(r: float) -> float:
            return min(self.dev_u(inside + r * ray), _LR_UNREACHABLE) - level

        try:
            r = brentq(excess, 0.0, 1.0, xtol=1e-14, rtol=1e-14)
        except ValueError:
            return None
        # On the boundary from inside: brentq's root can be either side
        # of it by its tolerance, and a point outside cannot be started
        # from (#601).
        for _ in range(5):
            if excess(r) <= 0.0:
                return inside + r * ray
            r *= 1.0 - 1e-12
        return None

    def keep(self, start: npt.NDArray, x: npt.NDArray) -> None:
        """``x``, where a search from ``start`` (a point of the region)
        ended, added to the points known, taken onto the boundary if it
        is outside the region: the bound must reach at least as far,
        and where that is further out than the answer the other
        searches give, the search goes on from it (``_retry_beyond``)."""
        if not self.dev_u(x) <= self.crit:
            back = self.back_onto_boundary(x, self.crit)
            if back is None:
                back = self.onto_boundary(start, x, self.crit)
            if back is None:
                return
            x = back
        psi_x = self.psi_u(x)
        if np.isfinite(psi_x):
            self.known.append((psi_x, x))

    def out_to_boundary(
        self, direction: float, x: npt.NDArray, level: float
    ) -> npt.NDArray:
        """``x``, a point inside {deviance <= level}, moved along the
        gradient of psi out to the boundary (or to the box): a ray
        search, for a search that stopped short of the boundary (SLSQP
        stops once psi changes by less than its tolerance, which far
        down a flat valley it can do inside the region: an ExpoWeibull
        sf(8) upper bound 1.5e-5 of deviance inside, #601). ``x`` where
        that is no further out."""
        if not self.dev_u(x) < level - _LR_NOISE:
            return x
        grad_psi = central_gradient(self.psi_u, x, _LR_FD_FINE)
        grad_dev = self.dev_grad_u(x)
        if grad_dev is None:
            grad_dev = central_gradient(self.dev_u, x, _LR_FD_FINE)
        d = direction * grad_psi
        rise = float(grad_dev @ d)
        if not (np.all(np.isfinite(d)) and np.isfinite(rise) and rise > 0):
            return x
        # Newton's step to the boundary on the deviance's tangent, doubled
        # until it is outside
        t = (level - self.dev_u(x)) / rise
        y: npt.NDArray | None = x
        for _ in range(40):
            y = self.model._lr_start(x + t * d, self.box)
            if not self.dev_u(y) <= level:
                y = self.onto_boundary(x, y, level)
                break
            if np.array_equal(
                y, self.model._lr_start(x + 2 * t * d, self.box)
            ):
                break  # held at the box
            t *= 2.0
        if y is None or not direction * (self.psi_u(y) - self.psi_u(x)) > 0:
            return x
        return y

    def back_onto_boundary(
        self, outside: npt.NDArray, level: float
    ) -> npt.NDArray | None:
        """The point of {deviance <= level} nearest ``outside``, a point
        just outside it, by Newton's steps along the gradient of the
        deviance (in steps of ``_LR_FD_FINE``); ``None`` where they do
        not reach it in five.

        Where a search ends just outside the region, far down a valley,
        the line back to its start can cross the boundary far from it:
        an ExpoWeibull's sf(8) upper bound lost 3e-6 that way (#601).
        """
        x = np.asarray(outside, dtype=float)
        for _ in range(5):
            excess = self.dev_u(x) - level
            if excess <= 0.0:
                return x
            grad = central_gradient(self.dev_u, x, _LR_FD_FINE)
            size = float(grad @ grad)
            if not (np.isfinite(size) and size > 0):
                return None
            # (overshooting a hair, so as to land inside)
            x = x - 1.000001 * excess * grad / size
            if not np.all(np.isfinite(x)):
                return None
        return x if self.dev_u(x) <= level else None

    def faces(self) -> list[tuple[int, float]]:
        """The faces of the box at the end of a parameter's coordinate:
        where the parameter's interval reaches the edge of its space."""
        out = []
        for k, (coord, (b_lo, b_hi)) in enumerate(
            zip(self.free_coords, self.box)
        ):
            if coord.kind == "identity":
                continue
            if b_lo is not None and b_lo == coord.ends[0]:
                out.append((k, float(b_lo)))
            if b_hi is not None and b_hi == coord.ends[1]:
                out.append((k, float(b_hi)))
        return out

    def probe_edges(self, direction: float) -> float | None:
        """The extreme of psi down the valleys to the edges of the
        parameters' spaces: the most extreme that checks out, or
        ``None``; the points found on the way are added to the points
        known, which the bound must reach.

        Where a parameter's interval reaches the edge of its space, its
        profile has been solved further out (``_lr_walk_on``), to where
        it levels off or to the end of its coordinate. The extreme is
        sought over the slice of the region through each of the two
        deepest of those points, the parameter held there (``extreme``
        over a ``face``): a search over the whole region from them runs
        out of the valley, where the function takes values far outside
        the region (an ExpoWeibull quantile of 0 as ``mu -> 0``), and
        back to a nearer extreme. From the slice's extreme the check
        (``checks_out``) looks further out over the whole region, and the
        search goes on from what it finds. The extremes so found lie far
        down the valley, which the searches from the estimate and the
        walks' tips missed: the ExpoWeibull's 95% ``qf(0.2)`` band was
        [2.72, 7.63] for [2.22, 7.94], found as ``beta -> inf``, and
        where the profile is still changing at the end of the coordinate,
        on the face of the box there (its 99% ``qf(0.95)`` upper bound,
        87.02 at ``alpha`` = 2.2e-308; #601).
        """
        best = None
        for k, end in self.faces():
            side = int(end == self.free_coords[k].ends[1])
            walk = self.seeds[2 * k + side] if self.seeds else []
            for point in walk[-2:]:
                if not self.dev_u(point) <= self.crit:
                    continue
                face = (k, float(point[k]))
                x = self.extreme_far(direction, point, self.crit, face=face)
                if x is None:
                    continue
                quick = self.checks_out(direction, x)
                if quick is None and self.beyond is not None:
                    quick = self.direct(direction, self.beyond)
                if quick is not None and (
                    best is None or direction * quick > direction * best
                ):
                    best = quick
        return best

    # -- one side -----------------------------------------------------------
    def solve_side(self, direction: float) -> float:
        """The bound on one side: ``direction`` -1 lower, 1 upper. A side
        whose answer's search did not converge is kept in
        ``unsettled_sides``."""
        answer = self._solve_side(direction)
        if answer in self.unsettled:
            self.unsettled_sides.add(direction)
        return answer

    def _solve_side(self, direction: float) -> float:
        quick = self.from_trace(direction)
        if quick is not None:
            return quick
        far_ladder = self.ladder(direction)
        on_face = self.probe_edges(direction)
        # The search starts from the estimate and from the farthest
        # points of the two walks that reach farthest (the region can
        # have more than one local extreme), and the most extreme
        # result that checks out is taken.
        tips = sorted(
            (max(walk, key=lambda k: direction * k[0]) for walk in self.walks),
            key=lambda k: -direction * k[0],
        )
        best = self._best_direct(direction, tips)
        if on_face is not None and (
            best is None or direction * on_face > direction * best
        ):
            best = on_face
        # As far out as every point known, and as the ladder's to
        # within its tolerance (a ladder that ends on the same extreme
        # by another path leaves the answer as it was).
        far = max(
            direction * k[0]
            for k in self.known
            if id(k[1]) not in self.ladder_ids
        )
        if (
            best is not None
            and direction * best >= far
            and direction * best >= far_ladder - 1e-6 * max(1.0, abs(best))
        ):
            return best
        quick = self._retry_beyond(direction, tips)
        if quick is not None:
            return quick
        return self.walk_profile(direction)

    def _best_direct(self, direction: float, tips: list) -> float | None:
        """The most extreme direct search from the estimate and the two
        farthest walks' tips that checks out, or ``None``. (Each is one
        search: one that stops short leaves its end among the points
        known, and ``_retry_beyond`` goes on from there.)"""
        assert self.u_start is not None
        best = None
        for start in [self.u_start] + [k[1] for k in tips[:2]]:
            quick = self.direct(direction, start, persist=False)
            if quick is not None and (
                best is None or direction * quick > direction * best
            ):
                best = quick
        return best

    def _retry_beyond(self, direction: float, tips: list) -> float | None:
        """The bound is at least as far out as every point of the region
        known: a search that stops short of one has stopped at a local
        extreme, and is tried again from the points beyond it, farthest
        first. ``None`` if none checks out."""
        tried: list[int] = [id(k[1]) for k in tips[:2]]
        for _ in range(3):
            beyond = [
                k
                for k in sorted(self.known, key=lambda k: -direction * k[0])
                if id(k[1]) not in tried and k[1] is not self.u_start
            ]
            if not beyond:
                break
            start = beyond[0][1]
            tried.append(id(start))
            quick = self.direct(direction, start)
            far = max(direction * k[0] for k in self.known)
            if quick is not None and direction * quick >= far:
                return quick
        return None

    def walk_profile(self, direction: float) -> float:
        """The profile of psi walked out from the farthest point known."""
        far_psi, far_u = max(self.known, key=lambda k: direction * k[0])
        path = _LRPath()
        path.add(far_psi, far_u)

        def deviance(target: float) -> float:
            nll, u = self.solve(target, path.starts(target))
            if u is None and self.reached:
                # From where a direct search ended nearest the target:
                # it can have run down a valley the walk's own points
                # do not reach, and stopped just outside the region
                # because the extreme there is only approached (an
                # ExpoWeibull sf(13) as beta -> inf with alpha at the
                # largest observation, #472).
                near = min(self.reached, key=lambda k: abs(k[0] - target))
                nll, u = self.solve(target, [near[1]])
            if u is None:
                nll, u = self.solve(target, [self.u_hat])
            if u is None:
                # No start reaches the target: the function does not
                # take that value near where the walk has come from
                # (a NegativeBinomial df(5) above 0.195, its value
                # in the Poisson limit; a Uniform hazard above
                # 1 / (max(x) - x), which b >= max(x) caps). It reads
                # as beyond any critical value.
                return np.inf
            path.add(target, u)
            return 2.0 * (nll - self.nll_hat)

        status, w = _lr_walk(
            deviance,
            far_psi,
            self.step,
            direction,
            self.crit,
            end=self.ends[1] if direction > 0 else self.ends[0],
            d_hat=self.dev_u(far_u),
        )
        if status == "root":
            return w
        if status == "edge":
            return np.inf if direction > 0 else -np.inf
        if direction * (far_psi - self.psi_hat) > 0:
            # The walk failed: the most extreme point of the region found,
            # flagged (principle: warn, don't refuse).
            self.unsettled.add(far_psi)
            return far_psi
        return np.nan


class LikelihoodRatioMixin:
    """The likelihood-ratio bounds of a :class:`Parametric` model.

    Separated from ``Parametric``, which inherits it, to keep the search
    machinery in one place; every method here reads the fitted model
    through the attributes and helpers ``Parametric`` defines.
    """

    if TYPE_CHECKING:
        # Supplied by Parametric, the one class that inherits this mixin.
        # Declared rather than defined so the methods below type check
        # without the mixin pretending to own them.
        dist: Any
        method: str
        params: npt.NDArray
        gamma: float
        p: float
        f0: float
        offset: bool
        lfp: bool
        zi: bool
        param_map: dict[str, int]
        surv_data: "SurpyvalData"

        def _resolve_param_name(self, name: str) -> tuple[bool, int]: ...
        def _ensure_surv_data(self) -> None: ...
        def _is_fixed_param(self, name: str) -> bool: ...
        def _user_fixed_idx(self) -> set: ...
        def _summary_scale(self, zero_floor: bool = False) -> tuple: ...

    def _lr_neg_ll(self, theta: npt.NDArray) -> float:
        """The negative log-likelihood at core parameters ``theta``, as
        the likelihood-ratio searches see it.

        ``nan`` where it cannot be right: at parameters that are not
        finite (a search that has stepped off to nan; some likelihoods
        iterate to their limit on nan, a NegativeBinomial's incomplete
        beta for 2 s a call), and below the fit's own minimum by more
        than the fit's precision (``_LR_NOISE`` in deviance, or 1e-8 of
        the log-likelihood; the registry's fits are within 2e-11 of
        their profiles' minima). The fit is the maximum, so a likelihood
        above it is the likelihood failing at extreme parameters, which a
        search would take for the best point there. It guards every
        family: an inaccurate density at ``beta`` = 1e14 and ``mu`` =
        1e-14 gave an ExpoWeibull a deviance of -1e21 before #472 made it
        accurate.
        """
        if not onp.isfinite(theta).all():
            return np.nan
        nll = self._lr_raw_neg_ll(theta)
        params = np.asarray(self.params, dtype=float)
        kept = self.__dict__.get("_lr_nll_hat")
        if kept is None or kept[0] != params.tobytes():
            kept = (params.tobytes(), self._lr_raw_neg_ll(params))
            self.__dict__["_lr_nll_hat"] = kept
        slack = max(_LR_NOISE, 2e-8 * abs(kept[1]))
        if 2.0 * (nll - kept[1]) < -slack:
            return np.nan
        return nll

    def _lr_raw_neg_ll(self, theta: npt.NDArray) -> float:
        # Kept by the parameters' bytes: the searches ask for the same
        # point again (SLSQP's function and gradient calls, a result
        # checked after the search), a quarter of a band's evaluations,
        # each O(n).
        key = np.asarray(theta, dtype=float).tobytes()
        memo = self.__dict__.get("_lr_nll_memo")
        if memo is None or memo[0] is not self.surv_data:
            memo = (self.surv_data, {})
            self.__dict__["_lr_nll_memo"] = memo
        kept: dict[bytes, float] = memo[1]
        if key in kept:
            return kept[key]
        with np.errstate(all="ignore"):
            lean = self._lr_lean_data()
            if lean is not None:
                nll = _lean_neg_ll(self.dist, lean, theta)
            else:
                nll = self._lr_full_neg_ll(theta)
        if len(kept) >= _LR_MEMO_SIZE:
            kept.clear()
        kept[key] = nll
        return nll

    def _lr_full_neg_ll(self, theta: npt.NDArray) -> float:
        """The negative log-likelihood at ``theta`` where there is no lean
        one (``_lr_lean_data``): the distribution's own. (A regression
        model's searches supply theirs, ``RegressionLikelihoodRatio``.)"""
        return float(
            self.dist._neg_ll_func(
                self.surv_data, *theta, self.gamma, self.f0, self.p
            )
        )

    def _lr_declared_bounds(self) -> list[tuple[Any, Any]]:
        """The declared ``(lower, upper)`` of each parameter the searches
        move (``None`` for no bound): the distribution's."""
        return list(self.dist.bounds)

    def _lr_plain(self) -> bool:
        """Whether the likelihood-ratio searches may evaluate the
        distribution's formulas without their guards where the data or
        query are inside its support and not missing (see
        ``_lr_lean_data``): a continuous distribution of
        ``ParametricFitter``'s, with a support that does not depend on its
        parameters, and no offset, zero inflation or limited failure
        population."""
        from .parametric_fitter import ParametricFitter

        dist = self.dist
        if not isinstance(dist, ParametricFitter) or dist.discrete:
            return False
        lo, hi = (float(v) for v in dist.support)
        return (
            not (np.isnan(lo) or np.isnan(hi))
            and self.gamma == 0
            and self.f0 == 0
            and self.p == 1
        )

    def _lr_function(self, name: str) -> Callable[..., Any]:
        """The distribution's function ``name`` as the likelihood-ratio
        band evaluates it, ``f(x, *theta)``: without its guards where
        every ``x`` is inside the support (which they leave unchanged
        there; see ``_lr_lean_data``), and as it is otherwise."""
        full = getattr(self.dist, name)
        if not self._lr_plain():
            return full
        dist = self.dist
        raw = _unguarded(dist, name)
        lo, hi = (float(v) for v in dist.support)

        def f(x: npt.NDArray, *theta: Any) -> Any:
            # (False for a missing x, which takes the full path.)
            if (x >= lo).all() and (x <= hi).all():
                return raw(dist, x, *theta)
            return full(x, *theta)

        return f

    def _lr_lean_data(self) -> tuple | None:
        """The data of the lean likelihood the likelihood-ratio searches
        evaluate (``_lean_neg_ll``), or ``None`` where they use the
        distribution's own ``_neg_ll_func``.

        The searches take thousands of evaluations, and most of each one's
        time went on checks that cannot change its value on these data:
        every observation inside the support, none missing (the guards of
        ``_support_guarded`` and ``_array_inputs``), and no offset, zero
        inflation or limited failure population (the terms they add are
        exactly 0). The lean likelihood calls the distribution's own
        formulas without them, on the same arrays, in the same order, so
        its value is the same to the last bit, about 2.5 times as fast
        (#519). Interval-censored and truncated windows are evaluated as
        ``ll_interval_or_truncated`` evaluates them, with its tail forms
        (``_window_log_likelihood``), from the data's part of it kept once
        (``_window_inputs``) and with the functions unwrapped: a Weibull's
        likelihood on six interval-censored units took 243 us, against 43
        us on 1000 exact ones (#602). A discrete distribution, a support
        that depends on the parameters (the Uniform's), or data outside
        the support or missing take the full path.
        """
        data = self.surv_data
        kept = self.__dict__.get("_lr_lean")
        if kept is not None and kept[0] is data:
            return kept[1]
        lean = None
        plain = self._lr_plain()
        if plain:
            lo, hi = (float(v) for v in self.dist.support)
            terms = []
            for x, n in (
                (data.x_o, data.n_o),
                (data.x_r, data.n_r),
                (data.x_l, data.n_l),
            ):
                x = np.atleast_1d(np.array(x))
                inside = not np.any(np.isnan(x)) and not (
                    np.any(x < lo) or np.any(x > hi)
                )
                plain = plain and inside
                terms.append(
                    None if x.size == 0 else (x - self.gamma, np.asarray(n))
                )
            windows: list[tuple | None] = []
            for xl, xr, n in (
                (data.x_il, data.x_ir, data.n_i),
                (data.tl_unique, data.tr_unique, data.n_t_unique),
            ):
                if np.size(xl) == 0:
                    windows.append(None)
                    continue
                # As ``ll_interval_or_truncated`` takes them (its
                # ``_check_x_not_empty``), and its data part kept
                xl = np.atleast_1d(np.array(xl))
                inputs = self.dist._window_inputs(
                    xl, xr, self.gamma, self.params
                )
                safe = np.concatenate(inputs[-2:])
                plain = plain and not (
                    np.any(np.isnan(safe))
                    or np.any(safe < lo)
                    or np.any(safe > hi)
                )
                windows.append((inputs, n))
            if plain:
                lean = (*terms, *windows, (self.gamma, self.f0, self.p))
        self.__dict__["_lr_lean"] = (data, lean)
        return lean

    def _lr_coords(self) -> tuple[list[_LRCoord], list[tuple[Any, Any]]]:
        """Each core parameter's likelihood-ratio search coordinate
        (``_LRCoord``), and its box in that coordinate: the data-derived
        limits of ``_lr_limits``, ``None`` where the limit is the declared
        bound (which the coordinate maps to infinity)."""
        coords, boxes = [], []
        for (lo, hi), (l_lo, l_hi) in zip(
            self._lr_declared_bounds(), self._lr_limits(), strict=True
        ):
            coord = _LRCoord(lo, hi)
            coords.append(coord)
            boxes.append(
                (
                    None if l_lo == coord.lo else coord.to_u(l_lo),
                    None if l_hi == coord.hi else coord.to_u(l_hi),
                )
            )
        return coords, boxes

    @staticmethod
    def _lr_box(coords: list, limits: list) -> list[tuple[Any, Any]]:
        """The box a likelihood-ratio search runs in: the data-derived
        limits, and elsewhere the ends of each coordinate, so that a
        search cannot step off to where the parameter overflows (the
        identity coordinate's ends are the doubles' own)."""
        box = []
        for coord, (b_lo, b_hi) in zip(coords, limits, strict=True):
            if coord.kind != "identity":
                b_lo = coord.ends[0] if b_lo is None else b_lo
                b_hi = coord.ends[1] if b_hi is None else b_hi
            box.append((b_lo, b_hi))
        return box

    @staticmethod
    def _lr_start(x0: npt.NDArray, box: list) -> npt.NDArray:
        """``x0`` inside ``box``."""
        x0 = np.array(x0, dtype=float)
        for k, (b_lo, b_hi) in enumerate(box):
            if b_lo is not None:
                x0[k] = max(x0[k], b_lo)
            if b_hi is not None:
                x0[k] = min(x0[k], b_hi)
        return x0

    def _profile_neg_ll(
        self, idx: int, value: Any, path: _LRPath | None = None
    ) -> float:
        """Profile negative log-likelihood with core parameter ``idx`` fixed.

        Holds the ``idx``-th distribution parameter at ``value`` and
        minimises the negative log-likelihood over the remaining core
        parameters, each in its unbounded search coordinate
        (``_LRCoord``). The search starts from the fit and, given the
        ``path`` of the points already solved, from the nearest of them
        and the line through the two nearest (continuation); the lowest
        minimum is kept, and added to ``path``. The raw parameters within
        box limits, started from the fit every time, did not follow the
        minimum out along its valley: a NegativeBinomial profile deviance
        of 3.05 at ``p`` = 0.999999 where it is 2.35, and ExpoWeibull ones
        of 49 and 42 where they are 3.9 and 0.29 (#421).

        ``nan`` if every search fails; ``inf`` where the likelihood is 0
        from every start. ``gamma``, ``f0`` and ``p`` are held at their
        fitted values -- likelihood-ratio bounds for offset / LFP / ZI models
        are not yet supported, so the public entry point rejects them before
        this is reached.
        """
        theta = np.array(self.params, dtype=float)
        theta[idx] = value
        # Parameters the user fixed at fit time stay fixed during the
        # profile — re-freeing them makes the profile drop below the fitted
        # nll and silently inflates the interval (#255).
        user_fixed = self._user_fixed_idx()
        free = [
            j for j in range(len(theta)) if j != idx and j not in user_fixed
        ]
        if not free:
            # Single-parameter distribution: nothing left to profile over.
            return self._lr_neg_ll(theta)

        coords, limits = self._lr_coords()
        free_coords = [coords[j] for j in free]
        box = [limits[j] for j in free]
        bounds = box if any(b != (None, None) for b in box) else None
        # Starts are held inside the coordinates' ends as well.
        start_box = self._lr_box(free_coords, box)

        def obj(u: npt.NDArray) -> float:
            th = theta.copy()
            th[free] = [c.from_u(v) for c, v in zip(free_coords, u)]
            nll = self._lr_neg_ll(th)
            return nll if np.isfinite(nll) else np.inf

        u_hat = np.array([coords[j].to_u(self.params[j]) for j in free])
        w = coords[idx].to_u(value)
        starts = [] if path is None else path.starts(w)
        best, best_u, zero = np.inf, None, False
        # Searched in the coordinates divided by the Wald standard errors
        # (``_wald_sd``), where L-BFGS-B's first steps are to scale.
        scale = _wald_sd(self, coords, free)
        if scale is None:
            scale = np.ones(len(free))
        z_bounds = _scaled_bounds(bounds, np.zeros(len(free)), scale)

        def obj_z(z: npt.NDArray) -> float:
            return obj(scale * z)

        with np.errstate(all="ignore"):
            for x0 in starts + [u_hat]:
                res = minimize(
                    obj_z,
                    self._lr_start(x0, start_box) / scale,
                    method="L-BFGS-B",
                    jac="3-point",
                    bounds=z_bounds,
                    options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 1000},
                )
                zero = zero or res.fun == np.inf
                if np.isfinite(res.fun) and res.fun < best:
                    best, best_u = float(res.fun), scale * res.x
            if best_u is None:
                # Every gradient search failed: derivative free from the
                # fit, as a last resort.
                res = minimize(
                    obj,
                    self._lr_start(u_hat, start_box),
                    method="Nelder-Mead",
                    bounds=bounds,
                )
                zero = zero or res.fun == np.inf
                if np.isfinite(res.fun):
                    best, best_u = float(res.fun), np.asarray(res.x)
        if best_u is None:
            return np.inf if zero else np.nan
        if path is not None:
            path.add(w, best_u, best)
        return best

    def _lr_limits(self) -> list[tuple[float, float]]:
        """``(lower, upper)`` of each core parameter for the
        likelihood-ratio searches: its declared bounds, and for a
        parameter that is an edge of the support (the Uniform's and the
        4-parameter Beta's ``a`` and ``b``) the data's extremes, beyond
        which the likelihood is 0. The searches could not follow that
        cliff: a Uniform band stalled at the estimate (#421)."""
        limits = [
            (-np.inf if lo is None else lo, np.inf if hi is None else hi)
            for lo, hi in self._lr_declared_bounds()
        ]
        support = np.asarray(
            getattr(self.dist, "support", (0.0, 0.0)), dtype=float
        )
        if np.any(np.isnan(support)):
            x = np.asarray(self.surv_data.x, dtype=float)
            x = x[np.isfinite(x)]
            i_lo, i_hi = self.dist.support_param_index
            if np.isnan(support[0]):
                lo, hi = limits[i_lo]
                limits[i_lo] = (lo, min(hi, float(x.min())))
            if np.isnan(support[1]):
                lo, hi = limits[i_hi]
                limits[i_hi] = (max(lo, float(x.max())), hi)
        return limits

    def _param_cb_lr(
        self, name: str, alpha_ci: float, bound: str
    ) -> npt.NDArray:
        """Profile-likelihood (likelihood-ratio) bound on a parameter.

        The bound(s) solve ``2[nll_p(v) - nll_hat] = c`` where ``nll_p`` is the
        profile negative log-likelihood, ``nll_hat`` the fitted value, and
        ``c`` the chi-squared critical value (``z**2``) at the requested level.
        The deviance is zero at the estimate, so each side walks out from it
        in the parameter's search coordinate (``_LRCoord``: its log, or
        logit, ...), in steps that start at the Wald standard error there,
        and the first crossing is solved by ``brentq`` (``_lr_walk``). The
        interval is thus the piece of the likelihood-ratio confidence set
        that contains the estimate. Where the deviance stays below ``c`` to
        the edge of the parameter's space, or levels off below it on the
        way, the bound is that edge (0, 1 or ``inf``): no value up to it is
        excluded. A data-derived limit (a Uniform's ``a`` at the smallest
        observation) reached first is the bound.
        """
        if self.method != "MLE":
            raise ValueError("Only MLE has confidence bounds")
        self._ensure_surv_data()
        if self.offset or self.lfp or self.zi:
            raise NotImplementedError(
                "Likelihood-ratio bounds are not yet available for offset, "
                "limited-failure-population or zero-inflated models; use "
                "method='wald'."
            )
        is_core, idx = self._resolve_param_name(name)
        if not is_core:
            raise NotImplementedError(
                "Likelihood-ratio bounds on 'p' / 'f0' are not yet "
                "available; use method='wald'."
            )

        check_option("bound", bound, BOUNDS)
        if self._is_fixed_param(name):
            # A parameter fixed at fit time is known, not estimated: the
            # degenerate interval at its value, as the Wald method gives
            # (its variance is zero).
            value = float(self.params[idx])
            if bound == "two-sided":
                return np.array([value, value])
            return np.array([value])
        if bound == "two-sided":
            crit = z(1.0 - alpha_ci / 2.0) ** 2
        else:
            crit = z(1.0 - alpha_ci) ** 2

        def solve_side(direction: Any) -> Any:
            value = self._lr_param_side(idx, crit, direction)
            if np.isnan(value):
                # Never fall back on the last candidate or the estimate:
                # an unfound bound is reported as such.
                side = "upper" if direction > 0 else "lower"
                warnings.warn(
                    f"The likelihood-ratio {side} bound on '{name}' could "
                    "not be found (the profile deviance never crossed the "
                    "critical value, or was not finite); nan is returned "
                    "for it. method='wald' gives a bound in its place.",
                    RuntimeWarning,
                    stacklevel=4,
                )
            return value

        if bound == "two-sided":
            return np.array([solve_side(-1), solve_side(1)])
        elif bound == "lower":
            return np.array([solve_side(-1)])
        else:
            return np.array([solve_side(1)])

    def _lr_key(self, idx: int, crit: float, direction: Any) -> tuple:
        """The key a parameter's likelihood-ratio side is kept under."""
        return (
            idx,
            float(crit),
            float(np.sign(direction)),
            np.asarray(self.params, dtype=float).tobytes(),
            id(self.surv_data),
        )

    def _lr_param_side(self, idx: int, crit: float, direction: Any) -> float:
        """One side of the likelihood-ratio interval on core parameter
        ``idx`` at the critical value ``crit``: the bound, the edge of the
        parameter's space, or ``nan`` where it cannot be found. See
        ``_param_cb_lr``.

        Each side has a path of its own, so a bound does not depend on
        whether the other side was asked for. A side is solved once per
        critical value and kept (a one-sided bound at ``alpha`` is the end
        of the two-sided one at ``2 alpha``, and ``cb`` reads the interval
        of every parameter).
        """
        theta_hat = float(self.params[idx])
        key = self._lr_key(idx, crit, direction)
        cache = self.__dict__.setdefault("_lr_sides", {})
        if key in cache:
            return cache[key]
        nll_hat = self._lr_neg_ll(np.asarray(self.params, dtype=float))

        hess_inv = getattr(self, "hess_inv", None)
        if (
            hess_inv is not None
            and np.ndim(hess_inv) == 2
            and np.isfinite(hess_inv[idx, idx])
            and hess_inv[idx, idx] > 0
        ):
            se = float(np.sqrt(hess_inv[idx, idx]))
        else:
            se = 0.5 * abs(theta_hat) if theta_hat != 0 else 1.0

        coords, limits = self._lr_coords()
        coord = coords[idx]
        w_hat = float(np.clip(coord.to_u(theta_hat), *coord.ends))
        # The first step: the standard error in the search coordinate.
        with np.errstate(all="ignore"):
            step = se / coord.slope(theta_hat)
        if not (np.isfinite(step) and step > 0):
            step = 1.0

        path = _LRPath()

        def deviance(w: float) -> float:
            nll = self._profile_neg_ll(idx, coord.from_u(w), path=path)
            return 2.0 * (nll - nll_hat)

        status, w = _lr_walk(
            deviance,
            w_hat,
            step,
            direction,
            crit,
            end=coord.ends[1] if direction > 0 else coord.ends[0],
            stop=limits[idx][1] if direction > 0 else limits[idx][0],
        )
        if status in ("root", "stop"):
            value = coord.from_u(w)
        elif status == "edge":
            value = float(coord.edge(direction))
        else:
            value = np.nan
        cache[key] = value
        # The points the walk solved inside the region, for the band's
        # searches to start from (see ``_cb_lr``).
        user_fixed = self._user_fixed_idx()
        free = [
            j
            for j in range(len(self.params))
            if j != idx and j not in user_fixed
        ]
        points = []
        for w_i, u_i, f_i in zip(path.w, path.u, path.f):
            if 2.0 * (f_i - nll_hat) <= crit:
                theta = np.array(self.params, dtype=float)
                theta[idx] = coord.from_u(w_i)
                theta[free] = [coords[j].from_u(v) for j, v in zip(free, u_i)]
                points.append(theta)
        self.__dict__.setdefault("_lr_points", {})[key] = points
        return value

    def _summary_cb_lr(
        self,
        fns: list,
        alpha_ci: float,
        bound: str,
        what: str,
    ) -> npt.NDArray:
        """Likelihood-ratio bounds on the functions ``fns`` of the core
        parameters (a quantile, the mean): the extreme of each over the
        parameters' likelihood region, searched as ``_cb_lr`` searches for
        a function of time, on the scale of ``_summary_scale``."""
        self._ensure_surv_data()
        if self.offset or self.lfp or self.zi:
            raise NotImplementedError(
                "Likelihood-ratio confidence bounds are not yet available "
                "for offset, limited-failure-population or zero-inflated "
                "models; use method='wald'."
            )
        if bound == "two-sided":
            crit = z(1.0 - alpha_ci / 2.0) ** 2
        else:
            crit = z(1.0 - alpha_ci) ** 2
        want_lower = bound in ("two-sided", "lower")
        want_upper = bound in ("two-sided", "upper")
        theta_hat = np.array(self.params, dtype=float)
        user_fixed = self._user_fixed_idx()
        free = [j for j in range(len(theta_hat)) if j not in user_fixed]
        n = len(fns)

        def shape(lower: npt.NDArray, upper: npt.NDArray) -> npt.NDArray:
            if bound == "two-sided":
                return np.column_stack([lower, upper])
            return lower if bound == "lower" else upper

        if not free:
            at = np.array([f(theta_hat) for f in fns], dtype=float)
            return shape(at, at)
        if len(free) == 1:
            band = self._cb_lr_one_param(
                np.arange(n, dtype=float),
                lambda i, theta: fns[int(i)](theta),
                free[0],
                alpha_ci,
                bound,
            )
            if band is not None:
                return band

        to_psi_, to_value, _, ends = self._summary_scale()

        def to_psi(v: Any) -> float:
            return float(to_psi_(v))

        lower = np.full(n, np.nan)
        upper = np.full(n, np.nan)
        failed = []
        unsettled = 0
        with np.errstate(all="ignore"):
            region = self._lr_region(free, crit)
            for i, f in enumerate(fns):

                def psi(theta: npt.NDArray, f: Callable = f) -> float:
                    return to_psi(f(theta))

                sides: set[float] = set()
                lo, hi = self._cb_lr_psi_bounds(
                    psi,
                    free,
                    crit,
                    want_lower,
                    want_upper,
                    ends,
                    *region,
                    unsettled=sides,
                )
                if (want_lower and np.isnan(lo)) or (
                    want_upper and np.isnan(hi)
                ):
                    failed.append(i)
                unsettled += bool(sides)
                lower[i], upper[i] = to_value(lo), to_value(hi)
        if unsettled:
            warn_unsettled(f"on {what} for {unsettled} of {n} value(s)")
        if failed:
            warnings.warn(
                f"The likelihood-ratio bound on {what} could not be found "
                f"for {len(failed)} of {n} value(s) (the constrained "
                "optimiser failed from every start); nan is returned there. "
                "method='wald' gives a bound in its place.",
                RuntimeWarning,
                stacklevel=4,
            )
        return shape(lower, upper)

    def _cb_lr_on_func(self, on: str) -> Any:
        """Return ``g(t, theta)`` for the requested ``on`` function.

        Evaluates the chosen distribution function at a single time for a
        candidate core-parameter vector, so the profile optimiser can push it
        to the edge of the likelihood region.
        """
        check_option("on", on, CB_ON)

        def g(t: Any, theta: npt.NDArray) -> Any:
            xt = np.atleast_1d(t) - self.gamma
            if on in ("sf", "R"):
                return self.dist.sf(xt, *theta)[0]
            if on in ("ff", "F"):
                return self.dist.ff(xt, *theta)[0]
            if on == "Hf":
                return self.dist.Hf(xt, *theta)[0]
            if on == "hf":
                return self.dist.hf(xt, *theta)[0]
            return self.dist.df(xt, *theta)[0]

        return g

    def _cb_lr(self, t: Any, on: str, alpha_ci: float, bound: str) -> Any:
        """Profile-likelihood (likelihood-ratio) band on a model function.

        At each time ``x`` the bound is the extreme value of the ``on``
        function over the parameter confidence region ``{theta :
        deviance(theta) <= crit}`` -- the piece of it that contains the
        estimate. It is found as ``param_cb`` finds a parameter's bound,
        with the function's value ``psi`` in the parameter's place: the
        bound is where the function's own profile deviance, ``2[min{
        nll(theta) : g(x, theta) = psi} - nll_hat]``, first reaches
        ``crit``. ``psi`` is on the scale the Wald band uses -- the logit
        of ``sf`` (from which the ``sf``, ``ff`` and ``Hf`` bands all
        come, so they agree exactly), the log of a continuous hazard or
        density, the logit of a discrete one.

        Each side is sought first directly (the extreme of ``psi`` over
        the region, by SLSQP, from the estimate and then from the points
        of the region farther out that the parameters' own walks found),
        and a result is taken only if it checks out as that crossing and
        is at least as far out as every point of the region known. The
        points known include the extremes over the narrower regions at
        1/4, 1/2 and 3/4 of the critical value and at the value itself,
        each searched from the one before (continuation in the level,
        ``_LR_LADDER``), so the search follows the extreme out as the
        region grows rather than stopping on a nearer local extreme of a
        long, curved region. Without it an ExpoWeibull 99% ``hf(13)`` lower
        bound comes out at 0.1046, above the 95% one of 0.1017, and a
        ``qf(0.95)`` upper bound at 36.4, below the 95% one of 40.4; with it
        they are 0.0714 and 80.6, near the 0.0708 and 87.0 that the region
        approaches at the end of ``alpha``'s coordinate (#535). Failing
        that, the profile of ``psi`` is walked out from the farthest point
        known (``_lr_walk``); where it stays below ``crit``, or levels off
        below it, to the end of the scale, the band reaches the edge of
        the function's range. Each end is solved once per level and kept.
        With two free parameters the region's boundary is traced once
        (``_lr_trace``), and each side is first sought from its most
        extreme traced point alone, taken when the answer is at least as
        far out as every traced point (and moved onto the boundary where
        SLSQP stopped just outside it), which makes a Weibull band at 20
        times on 1000 units 2.4 s rather than 9.4 s (#519). The region is
        found once per level, for every band and summary bound at that
        level; a band's times are searched in order, each side's search
        starting from where the neighbouring time's bound was found when
        that is as far out as the trace; and the searches run in the
        coordinates scaled by the Wald standard errors (``_wald_sd``),
        the check beyond each answer starting beside it. That takes a
        two-parameter band from about 250 likelihood evaluations a time
        to about 100 (#587).

        A search for the extreme from a warm start alone would stop
        wherever it first meets the region's boundary: ExpoWeibull and
        NegativeBinomial bands would be ``nan`` where a search fails, or
        short of the region's far corners (a NegativeBinomial ``sf(8)``
        lower bound of 0.00918 for 0.00587), and 95% and 80% bands would
        not be nested (#421).
        """
        self._ensure_surv_data()
        if self.offset or self.lfp or self.zi:
            raise NotImplementedError(
                "Likelihood-ratio confidence bounds are not yet available "
                "for offset, limited-failure-population or zero-inflated "
                "models; use method='wald'."
            )
        check_option("bound", bound, BOUNDS)
        check_option("on", on, CB_ON)

        if bound == "two-sided":
            crit = z(1.0 - alpha_ci / 2.0) ** 2
        else:
            crit = z(1.0 - alpha_ci) ** 2

        t = np.atleast_1d(t).astype(float)
        theta_hat = np.array(self.params, dtype=float)
        user_fixed = self._user_fixed_idx()
        free = [j for j in range(len(theta_hat)) if j not in user_fixed]
        if not free:
            # Every parameter was fixed at fit time: the region is the
            # estimate, and so is the band.
            g = self._cb_lr_on_func(on)
            at = np.array([g(time, theta_hat) for time in t], dtype=float)
            if bound == "two-sided":
                return np.column_stack([at, at])
            return at
        if len(free) == 1:
            band = self._cb_lr_one_param(
                t, self._cb_lr_on_func(on), free[0], alpha_ci, bound
            )
            if band is not None:
                return band

        survival = on in ("sf", "R", "ff", "F", "Hf")
        # ff and Hf fall as sf rises: their lower bound is sf's upper.
        falling = on in ("ff", "F", "Hf")
        want_lower = bound in ("two-sided", "lower")
        want_upper = bound in ("two-sided", "upper")
        if falling:
            want_lower, want_upper = want_upper, want_lower

        if survival:
            sf, ff = self._lr_function("sf"), self._lr_function("ff")

            def psi_of(time: Any, theta: npt.NDArray) -> float:
                x = np.atleast_1d(time) - self.gamma
                return float(
                    np.log(sf(x, *theta)[0]) - np.log(ff(x, *theta)[0])
                )

            ends = (_LN_TINY, -_LN_TINY)
            if on in ("sf", "R"):
                value: Callable[[Any], Any] = expit
            elif on in ("ff", "F"):
                value = lambda v: expit(-v)  # noqa: E731
            else:
                value = lambda v: np.logaddexp(0.0, -v)  # noqa: E731
        else:
            rate = self._lr_function("hf" if on == "hf" else "df")
            discrete = bool(self.dist.discrete)

            def psi_of(time: Any, theta: npt.NDArray) -> float:
                g = rate(np.atleast_1d(time) - self.gamma, *theta)[0]
                if discrete:
                    # A discrete hazard and mass are probabilities: the
                    # logit scale, as the Wald band's.
                    return float(np.log(g) - np.log1p(-g))
                return float(np.log(g))

            if discrete:
                ends = (_LN_TINY, -_LN_TINY)
                value = expit
            else:
                ends = (_LN_TINY, _LN_MAX)
                value = np.exp

        lower, upper, failed, unsettled_at = self._lr_band(
            t,
            np.argsort(t, kind="stable"),
            psi_of,
            lambda time: (float(time),),
            "survival" if survival else on,
            free,
            crit,
            ends,
            value,
            (want_lower, want_upper),
        )
        if unsettled_at:
            at = sorted({float(t[i]) for i in unsettled_at})
            warn_unsettled(f"at t = {at}")
        if failed:
            at = sorted({float(t[i]) for i in failed})
            warnings.warn(
                "The likelihood-ratio bound could not be found at "
                f"t = {at} (the constrained optimiser "
                "failed from every start); nan is returned there. "
                "method='wald' gives a bound in its place.",
                RuntimeWarning,
                # _cb_lr -> cb -> the query-shape wrapper -> the caller
                stacklevel=4,
            )

        if falling:
            lower, upper = upper, lower
        if bound == "two-sided":
            return np.column_stack([lower, upper])
        elif bound == "lower":
            return lower
        else:
            return upper

    def _lr_band(
        self,
        queries: Any,
        order: Any,
        psi_of: Callable[[Any, npt.NDArray], float],
        key_of: Callable[[Any], tuple],
        kind: str,
        free: list[int],
        crit: float,
        ends: tuple[float, float],
        value: Callable[[Any], Any],
        want: tuple[bool, bool],
    ) -> tuple[npt.NDArray, npt.NDArray, list[int], list[int]]:
        """The likelihood-ratio bounds on ``psi_of(query, theta)`` at
        each of ``queries`` (a band's times; a regression model's pairs of
        time and covariates), searched in ``order``: ``(lower, upper)``,
        each carried from the psi scale by ``value``, and the positions
        of the queries where a side asked for (``want``) was not found,
        and where a side's search did not converge (for the caller's
        warnings).

        A bound is solved once per function (``kind``), query (by
        ``key_of``), level and side, and kept: the sf, ff and Hf bands
        are one band, and a one-sided bound at alpha is an end of the
        two-sided one at 2 alpha. Each side's search starts from where
        the query before it found its bound (``hints``; see
        ``_cb_lr``)."""
        want_lower, want_upper = want
        cache = self.__dict__.setdefault("_lr_bands", {})
        # The keys of the bounds whose search did not converge
        unsure = self.__dict__.setdefault("_lr_unsettled", set())
        n = len(queries)
        lower = np.full(n, np.nan)
        upper = np.full(n, np.nan)
        failed: list[int] = []
        unsettled_at: list[int] = []
        hints: dict[float, npt.NDArray] = {}
        with np.errstate(all="ignore"):
            for i in order:
                q = queries[i]
                key_lo = (kind, *key_of(q), *self._lr_key(-1, crit, -1))
                key_hi = (kind, *key_of(q), *self._lr_key(-1, crit, 1))
                need_lo = want_lower and key_lo not in cache
                need_hi = want_upper and key_hi not in cache
                if need_lo or need_hi:
                    self._cb_lr_time(
                        lambda theta: psi_of(q, theta),
                        free,
                        crit,
                        ends,
                        (key_lo, key_hi),
                        (need_lo, need_hi),
                        hints,
                    )
                lo = cache[key_lo] if want_lower else np.nan
                hi = cache[key_hi] if want_upper else np.nan
                if (want_lower and np.isnan(lo)) or (
                    want_upper and np.isnan(hi)
                ):
                    failed.append(int(i))
                if (want_lower and key_lo in unsure) or (
                    want_upper and key_hi in unsure
                ):
                    unsettled_at.append(int(i))
                lower[i], upper[i] = value(lo), value(hi)
        return lower, upper, failed, unsettled_at

    def _cb_lr_time(
        self,
        psi: Callable[[npt.NDArray], float],
        free: list[int],
        crit: float,
        ends: tuple[float, float],
        keys: tuple[tuple, tuple],
        needs: tuple[bool, bool],
        hints: dict[float, npt.NDArray],
    ) -> None:
        """The sides ``needs`` (lower, upper) of one time of ``_cb_lr``'s
        band, kept under ``keys`` in the model's band cache, and in
        ``_lr_unsettled`` where the search did not converge."""
        cache = self.__dict__.setdefault("_lr_bands", {})
        unsure = self.__dict__.setdefault("_lr_unsettled", set())
        box, seeds, trace = self._lr_region(free, crit)
        sides: set[float] = set()
        found = self._cb_lr_psi_bounds(
            psi,
            free,
            crit,
            needs[0],
            needs[1],
            ends,
            box,
            seeds,
            trace,
            hints=hints,
            unsettled=sides,
        )
        for key, need, value, side in zip(keys, needs, found, (-1.0, 1.0)):
            if need:
                cache[key] = value
                if side in sides:
                    unsure.add(key)

    def _lr_region(self, free: list[int], crit: float) -> tuple[
        list[tuple[Any, Any]],
        list[list[npt.NDArray]],
        list[npt.NDArray] | None,
    ]:
        """The box a likelihood-ratio band's searches run in, the points
        they may start from (one list per walk), and the region's boundary
        traced where it can be (``_lr_trace``), at the critical value
        ``crit``.

        The box is the one the parameters' own intervals at this level
        make: the region's extent in each parameter is that parameter's
        likelihood-ratio interval, so the box holds every point the band
        can come from, and it keeps a search for a value the function
        never takes from wandering off to where the likelihood is slow or
        not defined (a NegativeBinomial ``r`` of 1e-308, whose incomplete
        beta takes 4.5 s a call). The points are those of the region that
        the intervals' walks passed through: they reach its far corners
        (an ExpoWeibull ``beta`` running off to infinity with ``alpha`` at
        the largest observation), which a search from the estimate does
        not find.

        The region is found once per level and kept: every band, quantile
        and mean bound at that level searches the same one.
        """
        key = (tuple(free), *self._lr_key(-1, crit, 0))
        cache = self.__dict__.setdefault("_lr_regions", {})
        if key not in cache:
            cache[key] = self._lr_find_region(free, crit)
        return cache[key]

    def _lr_find_region(self, free: list[int], crit: float) -> tuple[
        list[tuple[Any, Any]],
        list[list[npt.NDArray]],
        list[npt.NDArray] | None,
    ]:
        """The region of ``_lr_region``, found."""
        coords, limits = self._lr_coords()
        free_coords = [coords[j] for j in free]
        box = self._lr_box(free_coords, [limits[j] for j in free])
        with np.errstate(all="ignore"):
            for k, j in enumerate(free):
                b_lo, b_hi = box[k]
                lo_j = coords[j].to_u(self._lr_param_side(j, crit, -1))
                hi_j = coords[j].to_u(self._lr_param_side(j, crit, 1))
                if np.isfinite(lo_j):
                    b_lo = lo_j if b_lo is None else max(b_lo, lo_j)
                if np.isfinite(hi_j):
                    b_hi = hi_j if b_hi is None else min(b_hi, hi_j)
                box[k] = (b_lo, b_hi)
            # One list per walk.
            seeds = [
                [
                    np.array([coords[i].to_u(theta[i]) for i in free])
                    for theta in self.__dict__.get("_lr_points", {}).get(
                        self._lr_key(j, crit, d), []
                    )
                ]
                for j in free
                for d in (-1.0, 1.0)
            ]
            for k, j in enumerate(free):
                for s, d in enumerate((-1.0, 1.0)):
                    edge = coords[j].to_u(self._lr_param_side(j, crit, d))
                    if not np.isfinite(edge):
                        seeds[2 * k + s] += self._lr_walk_on(
                            free, k, d, crit, seeds[2 * k + s]
                        )
            trace = self._lr_trace(free, crit, seeds)
        return box, seeds, trace

    def _lr_walk_on(
        self,
        free: list[int],
        k: int,
        direction: float,
        crit: float,
        walk: list[npt.NDArray],
    ) -> list[npt.NDArray]:
        """More points of the region along the valley the profile of the
        free parameter ``free[k]`` follows to the edge of its space, where
        its likelihood-ratio interval ends: the profile solved further
        out than its walk went, in steps doubling to the end of the
        coordinate (continued from the points before), for as long as it
        is inside the region and still changing (by more than ``_LR_NOISE``).

        The walk stops once the profile levels off below the critical
        value (``_lr_walk``), and a band's extreme can lie far beyond
        that, down the same valley: an ExpoWeibull's ``qf(0.05)`` lower
        bound is 0.5517 there, as ``beta -> inf``, and 0.8487 on the
        nearer side, which a search from the walk's points found (#601).
        """
        j = free[k]
        coords, _ = self._lr_coords()
        coord = coords[j]
        others = [i for i in range(len(free)) if i != k]
        end = coord.ends[1] if direction > 0 else coord.ends[0]
        path = _LRPath()
        for u in walk:
            path.add(u[k], u[others])
        nll_hat = self._lr_neg_ll(np.asarray(self.params, dtype=float))
        w = coord.to_u(float(self.params[j]))
        dev = 0.0
        if path.w:
            i = int(np.argmax(direction * np.asarray(path.w)))
            w, dev = path.w[i], 2.0 * (path.f[i] - nll_hat)
        out = []
        step = 1.0
        for _ in range(12):
            if not direction * (end - w) > 0:
                break
            w_next = w + direction * step
            if direction * (w_next - end) >= 0:
                w_next = end
            step *= 2.0
            dev_next = 2.0 * (
                self._profile_neg_ll(j, coord.from_u(w_next), path=path)
                - nll_hat
            )
            if not dev_next <= crit or len(path.w) == 0:
                break
            u = np.empty(len(free))
            u[k], u[others] = w_next, path.u[-1]
            out.append(u)
            if abs(dev_next - dev) < _LR_NOISE:
                break
            w, dev = w_next, dev_next
        return out

    #: Rays along which ``_lr_trace`` finds a two-parameter region's
    #: boundary, at angles evenly spaced in the Wald metric.
    _LR_TRACE_RAYS = 64

    #: The fractions of the critical value at which ``_cb_lr_psi_bounds``
    #: finds a function's extreme over the narrower regions first, each
    #: search started from the one before (continuation in the level).
    _LR_LADDER = (0.25, 0.5, 0.75)

    def _lr_trace(
        self, free: list[int], crit: float, seeds: list[list[npt.NDArray]]
    ) -> list[npt.NDArray] | None:
        """Points on the boundary of a two-parameter likelihood-ratio
        region, ``{u : deviance(u) = crit}`` in the search coordinates, or
        ``None`` where it is not traced.

        The boundary is found by bracketing and ``brentq`` along rays from
        the estimate, at angles evenly spaced in the Wald metric (in which
        the region is near a circle), and along the ray through each point
        of the region that the parameters' walks found (``seeds``) inside
        it, whose tips reach its far corners. ``_cb_lr_psi_bounds`` starts
        its search for a function's extreme over the region from the
        traced point where the function is most extreme, and takes the
        answer without the searches from the walks' tips when it is at
        least as far out as every traced point (#519; Meeker and Escobar,
        *Statistical Methods for Reliability Data*, trace the region so to
        draw it).

        ``None`` with more or fewer than two free parameters (see
        ``_lr_traces``), without a positive definite covariance, where a
        ray does not meet the boundary or the deviance cannot be computed
        on it, or where a walk's point lies beyond the boundary on its
        ray: the region is not star-shaped about the estimate, and the
        rays can miss part of it.
        """
        if not self._lr_traces(free):
            return None
        hess_inv = getattr(self, "hess_inv", None)
        if hess_inv is None or np.ndim(hess_inv) != 2:
            return None
        L = self._lr_wald_factor(free)
        excess, u_hat, nll_hat = self._lr_ray_excess(free, crit)
        if L is None or not np.isfinite(nll_hat):
            return None

        def crossing(d: npt.NDArray) -> float | None:
            return self._lr_crossing(excess, u_hat, d, crit)

        trace = []
        with np.errstate(all="ignore"):
            for d in self._lr_trace_rays(L):
                r = crossing(d)
                if r is None:
                    return None
                trace.append(u_hat + r * d)
            for group in seeds:
                for seed in group:
                    if not excess(np.asarray(seed, dtype=float)) <= 0.0:
                        continue
                    offset = np.asarray(seed, dtype=float) - u_hat
                    size = float(np.linalg.norm(np.linalg.solve(L, offset)))
                    if not (np.isfinite(size) and size > 0.0):
                        continue
                    r = crossing(offset / size)
                    if r is None or r < size * (1.0 - 1e-6):
                        return None
                    trace.append(u_hat + r * offset / size)
        return trace

    def _lr_traces(self, free: list[int]) -> bool:
        """Whether ``_lr_trace`` traces the region of the parameters
        ``free``: with two of them, where 64 rays cover its boundary. (A
        regression model's searches trace more, ``_lr_trace_rays``.)"""
        return len(free) == 2

    def _lr_trace_rays(self, L: npt.NDArray) -> list[npt.NDArray]:
        """The directions of ``_lr_trace``'s rays, in the search
        coordinates: ``_LR_TRACE_RAYS`` angles evenly spaced in the Wald
        metric, whose Cholesky factor is ``L``."""
        rays = self._LR_TRACE_RAYS
        out = []
        for k in range(rays):
            angle = 2.0 * np.pi * k / rays
            out.append(L @ np.array([np.cos(angle), np.sin(angle)]))
        return out

    def _lr_wald_factor(self, free: list[int]) -> npt.NDArray | None:
        """The Cholesky factor of the Wald covariance of the parameters
        ``free`` in their search coordinates, or ``None`` where it is not
        positive definite."""
        theta_hat = np.array(self.params, dtype=float)
        coords, _ = self._lr_coords()
        slopes = np.array(
            [coords[j].slope(theta_hat[j]) for j in free], dtype=float
        )
        hess_inv = getattr(self, "hess_inv", None)
        if hess_inv is None:
            return None
        cov_u = np.asarray(hess_inv, dtype=float)[np.ix_(free, free)]
        cov_u = cov_u / np.outer(slopes, slopes)
        try:
            L = np.linalg.cholesky(cov_u)
        except np.linalg.LinAlgError:
            return None
        return L if np.all(np.isfinite(L)) else None

    def _lr_ray_excess(
        self, free: list[int], crit: float
    ) -> tuple[Callable[[npt.NDArray], float], npt.NDArray, float]:
        """``(excess, u_hat, nll_hat)``: the deviance less ``crit`` at a
        point ``u`` of the search coordinates of the parameters ``free``
        (``inf`` where the likelihood is 0, ``nan`` where it cannot be
        right, see ``_lr_neg_ll``), the estimate in those coordinates and
        the negative log-likelihood there."""
        theta_hat = np.array(self.params, dtype=float)
        nll_hat = self._lr_neg_ll(theta_hat)
        coords, _ = self._lr_coords()
        free_coords = [coords[j] for j in free]
        u_hat = np.array([coords[j].to_u(theta_hat[j]) for j in free])

        def excess(u: npt.NDArray) -> float:
            theta = theta_hat.copy()
            theta[free] = [c.from_u(v) for c, v in zip(free_coords, u)]
            nll = self._lr_neg_ll(theta)
            if nll == np.inf:
                return np.inf
            return 2.0 * (nll - nll_hat) - crit

        return excess, u_hat, nll_hat

    @staticmethod
    def _lr_crossing(
        excess: Callable[[npt.NDArray], float],
        u_hat: npt.NDArray,
        d: npt.NDArray,
        crit: float,
    ) -> float | None:
        """The radius along ``u_hat + r d`` at which ``excess`` (see
        ``_lr_ray_excess``) is 0, a hair inside; ``None`` where the ray
        does not meet the boundary, or the deviance is not finite on
        the way."""
        r_lo, r_hi = 0.0, float(np.sqrt(crit))
        f_hi = excess(u_hat + r_hi * d)
        for _ in range(60):
            if not np.isfinite(f_hi):
                return None
            if f_hi > 0.0:
                break
            r_lo, r_hi = r_hi, 2.0 * r_hi
            f_hi = excess(u_hat + r_hi * d)
        else:
            return None
        r = brentq(
            lambda r: excess(u_hat + r * d),
            r_lo,
            r_hi,
            xtol=1e-12 * r_hi,
            rtol=1e-12,
        )
        # Just inside: a point of the region.
        return float(r) * (1.0 - 1e-10)

    def _cb_lr_psi_bounds(
        self,
        psi_of: Callable[[npt.NDArray], float],
        free: list[int],
        crit: float,
        want_lower: bool,
        want_upper: bool,
        ends: tuple[float, float],
        box: list[tuple[Any, Any]],
        seeds: list[list[npt.NDArray]],
        trace: list[npt.NDArray] | None = None,
        hints: dict[float, npt.NDArray] | None = None,
        unsettled: set[float] | None = None,
    ) -> tuple[float, float]:
        """The likelihood-ratio bounds on a function ``psi_of(theta)`` of
        the free core parameters, searched in ``box``: ``(lower,
        upper)``, ``nan`` for a side not asked for or not found, ``-inf``
        / ``inf`` for one at the edge of the scale. See ``_cb_lr``, and
        ``_PsiBoundSearch`` for the search. The sides (-1 lower, 1
        upper) whose search did not converge are added to
        ``unsettled``."""
        search = _PsiBoundSearch(self, psi_of, free, crit, ends, box)
        if hints is not None:
            search.hints = dict(hints)
        out = search.run(want_lower, want_upper, seeds, trace)
        if hints is not None:
            hints.update(search.answers)
        if unsettled is not None:
            unsettled.update(search.unsettled_sides)
        return out

    def _cb_lr_one_param(
        self, t: Any, g: Any, j: int, alpha_ci: float, bound: str
    ) -> Any:
        """The likelihood-ratio band of a model with one free parameter.

        Its likelihood region is an interval, the profile bound on the
        parameter, so the band at each time is the extreme of ``g`` over
        that interval: at an end, or at an interior stationary point of
        ``g`` (a density at ``x`` peaks in the scale). A constrained
        search from a warm start found one end or the other, not always
        the more extreme: a Geometric df(5) lower bound of 0.0740 in a
        sweep but 0.0652 queried alone (#421). ``None`` (the general
        search is used instead) where the interval cannot be found.
        """
        name = self.dist.parameter_names[j]
        # A one-sided bound at alpha is an end of the two-sided region at
        # 2 alpha: the same chi-squared critical value.
        level = alpha_ci if bound == "two-sided" else 2.0 * alpha_ci
        if not 0 < level < 1:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ends = self._param_cb_lr(name, level, "two-sided")
        if not np.all(np.isfinite(ends)):
            return None
        lo, hi = (float(e) for e in ends)
        # The likelihood at a support edge (p = 0 or 1) is typically nan,
        # so g is evaluated just inside it, as the search is.
        lo_b, hi_b = self.dist.bounds[j]
        lo = 1e-10 if lo_b == 0 and lo == 0 else lo
        hi = hi - 1e-10 if hi_b == 1 and hi == 1 else hi
        theta = np.array(self.params, dtype=float)

        def at(time: Any, v: Any) -> Any:
            th = theta.copy()
            th[j] = v
            return g(time, th)

        lower = np.empty(t.shape)
        upper = np.empty(t.shape)
        with np.errstate(all="ignore"):
            for i, time in enumerate(t):
                values = [at(time, lo), at(time, hi), at(time, theta[j])]
                for sign in (1.0, -1.0):
                    res = minimize_scalar(
                        lambda v: sign * at(time, v),
                        bounds=(lo, hi),
                        method="bounded",
                    )
                    if np.isfinite(res.fun):
                        values.append(sign * res.fun)
                values = np.asarray(values, dtype=float)
                if not np.all(np.isfinite(values[:3])):
                    return None
                values = values[np.isfinite(values)]
                lower[i], upper[i] = values.min(), values.max()
        if bound == "two-sided":
            return np.column_stack([lower, upper])
        return lower if bound == "lower" else upper
