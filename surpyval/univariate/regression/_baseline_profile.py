"""Whether a regression's baseline shape or scale runs to a limit (#710).

The no-maximum check of a regression fit (``judge_search`` in
``_fit_skeleton.py``) is Newton's, along each parameter's profile from the
point the search stopped (``runaways_in_units``), and it reads that profile
from the Hessian there. A baseline parameter that runs to a limit of its
family can take the search where the Hessian cannot be had: a LogNormalPH
whose sigma ran to 1e-83 (the hazard rising from 0 at a threshold, its
scale ``1 / sigma^2`` taken up by a coefficient that runs off with it) has
derivatives in sigma that overflow, and the fit could only say
"unverified". So could one that stopped short of a maximum where its
sigma or a WeibullPO's beta is far from 1.

The check here walks the profile itself: the parameter, on the log scale of
its distance from its bound, is held at one value after another, and the
others are brought to their best values for it by Newton's method (their
own Hessian stays finite where the parameter's column does not). Where the
profile log-likelihood rises on the way to the limit, all the way to where
the parameter can no longer be represented, the likelihood has no finite
maximum along it; where the profile rises and then falls, its maximum is
found between and the search finished from there, and verified as any
other fit's, its Hessian taken by differences of the gradient in the
parameters whose second derivatives overflow (a WeibullPO's alpha at
1e-127, :func:`filled_derivatives`). Only a fit that is not otherwise
verified is walked.
"""

from typing import Any, Callable, NamedTuple

import autograd.numpy as np
import numpy.typing as npt
from autograd import grad
from scipy.optimize import OptimizeResult, minimize_scalar

from surpyval.univariate.parametric.fitters.runaway import (
    LOG_MAX,
    search_derivatives,
)

#: The profile is walked in steps that start at this many e-folds of the
#: parameter's distance from its bound and double,
FIRST_STEP = 0.25
#: as far as this, past where any such distance can be represented.
WALK_EFOLDS = 1500.0
#: The least distance in e-folds over which a profile must rise, unbroken,
#: to be called a run to the limit where the walk ends before the
#: representable range does (the likelihood cannot be computed further).
FEWEST_EFOLDS = 7.0
#: Newton steps on the other parameters at each point of the profile, and
#: the gain (per unit of the objective) their quadratic model must promise
#: for another.
NEWTON_STEPS = 30
INNER_TOL = 1e-14


class ProfileVerdict(NamedTuple):
    """What the walk along a baseline parameter's profile found."""

    #: ``"run-off"``, or ``"interior"`` where the profile has a maximum.
    kind: str
    #: For an interior maximum, the point of the profile at it.
    res: Any = None
    #: For a run-off, the positions in the search vector of the parameters
    #: that run off with it (it included).
    running: "tuple[int, ...]" = ()


def to_log(u: float) -> float:
    """A one-bounded parameter's search value ``u`` as the log of its
    distance from its bound: ``bounds_convert`` searches the log within a
    unit of the bound and the distance less 1 beyond."""
    return u if u < 0.0 else float(np.log1p(u))


def from_log(w: float) -> float:
    """The inverse of :func:`to_log`; ``inf`` beyond the largest float."""
    if w < 0.0:
        return w
    if w > LOG_MAX:
        return float("inf")
    return float(np.expm1(w))


def filled_derivatives(
    fun: Callable, x: npt.ArrayLike, floor: "float | npt.ArrayLike" = 1.0
) -> "tuple[npt.NDArray, npt.NDArray] | None":
    """:func:`search_derivatives` of ``fun`` at ``x``, with each column of
    the Hessian that autograd cannot give finitely taken by central
    differences of its gradient instead (each component stepped by 1e-5
    of ``max(|x|, floor)``, as ``is_local_minimum`` does for a univariate
    fit). A baseline parameter far along its search scale overflows in
    autograd's second derivatives, not its first: a WeibullPO's alpha at
    1e-127, ``1 / alpha^2`` in the chain rule (#710). ``None`` where the
    gradient is not finite either."""
    at = np.asarray(x, dtype=float)
    derivatives = search_derivatives(fun, at)
    if derivatives is None:
        return None
    H, g = derivatives
    if not np.all(np.isfinite(g)):
        return None
    bad = np.flatnonzero(~np.all(np.isfinite(H), axis=0))
    if not bad.size:
        return derivatives
    scale = np.maximum(
        np.abs(at), np.broadcast_to(np.asarray(floor, dtype=float), at.shape)
    )
    H = np.array(H, dtype=float)
    gradient = grad(fun)
    for i in bad:
        step = 1e-5 * scale[i]
        e = np.zeros(at.size)
        e[i] = step
        try:
            up, down = (
                np.asarray(gradient(at + s * e), dtype=float)
                for s in (1.0, -1.0)
            )
        except (TypeError, ValueError, ArithmeticError):
            return None
        column = (up - down) / (2.0 * step)
        H[:, i] = column
        H[i, :] = column
    if not np.all(np.isfinite(H)):
        return None
    return H, g


def _scaled_newton_step(H: npt.NDArray, g: npt.NDArray) -> npt.NDArray:
    """A Newton step ``-H^{-1} g`` in units where ``H``'s diagonal is 1,
    with each eigenvalue of ``H`` taken as its size, at least ``1e-12`` of
    the largest: where ``H`` is not positive definite the step still goes
    downhill."""
    d = np.sqrt(np.abs(np.diag(H)))
    d = np.where(d > 0.0, d, 1.0)
    lam, V = np.linalg.eigh(H / np.outer(d, d))
    size = np.maximum(np.abs(lam), 1e-12 * np.max(np.abs(lam), initial=1.0))
    return -(V @ ((V.T @ (g / d)) / size)) / d


def profile_point(
    fun: Callable, guess: npt.ArrayLike, j: int, w: float
) -> "tuple[float, npt.NDArray] | None":
    """``(f, y)``: the least value of ``fun`` with parameter ``j`` held at
    ``from_log(w)``, and the point where it is, by Newton's method on the
    other parameters from ``guess`` (each step halved until it lowers
    ``fun``). ``None`` where ``fun`` is not finite there, or the
    others' derivatives are not, or Newton's method has not converged in
    ``NEWTON_STEPS``."""
    u = from_log(w)
    y = np.array(guess, dtype=float)
    if not np.isfinite(u):
        return None
    y[j] = u
    others = [i for i in range(y.size) if i != j]

    def holding(v: Any) -> Any:
        full = [v[k - (k > j)] if k != j else u for k in range(y.size)]
        return fun(np.array(full))

    with np.errstate(all="ignore"):
        f = float(fun(y))
        if not np.isfinite(f):
            return None
        for _ in range(NEWTON_STEPS if others else 0):
            v = y[others]
            derivatives = filled_derivatives(holding, v)
            if derivatives is None:
                return None
            H, g = derivatives
            try:
                step = _scaled_newton_step(H, g)
            except np.linalg.LinAlgError:
                return None
            if -0.5 * float(g @ step) <= INNER_TOL * max(1.0, abs(f)):
                # Within rounding of the others' best values
                break
            for _ in range(40):
                trial = v + step
                f_trial = float(holding(trial))
                if np.isfinite(f_trial) and f_trial < f:
                    break
                step = 0.5 * step
            else:
                # No step lowers it: at its least value, to rounding
                break
            y[others], f = trial, f_trial
        else:
            if others:
                # Not converged: the others' best values are not found
                return None
    return f, y


def _bracket_minimum(
    point: Callable, a: "tuple", b: "tuple", c: "tuple"
) -> "tuple[float, npt.NDArray]":
    """The profile's least value between ``a`` and ``c`` (each ``(w, f,
    y)``), ``b`` between them and below both, by Brent's method on ``w``,
    each point's search started from the nearest point already found."""
    found = [a, b, c]

    def value(w: float) -> float:
        near = min(found, key=lambda p: abs(p[0] - w))
        out = point(near[2], w)
        if out is None:
            return float("inf")
        found.append((w, out[0], out[1]))
        return out[0]

    try:
        minimize_scalar(
            value,
            bracket=(a[0], b[0], c[0]),
            method="brent",
            options={"xtol": 1e-6, "maxiter": 40},
        )
    except (ValueError, ArithmeticError):
        pass
    best = min(found, key=lambda p: p[1])
    return best[1], best[2]


def _running(
    points: "list[tuple]", one_sided: "tuple[int, ...]", floor: npt.NDArray
) -> "tuple[int, ...]":
    """The positions of the parameters that run off along the walk's last
    three ``points``: each one-bounded parameter on the log scale of its
    distance from its bound, the others as they are, a parameter that
    moved at least half as fast (per e-fold of the walked parameter) over
    the last step as over the one before, and by more than a millionth of
    its size, is on its way to infinity with the profile (one that runs
    off with it moves in proportion, or faster); one that has converged
    moves less and less."""
    (w0, _, y0), (w1, _, y1), (w2, _, y2) = points[-3:]

    def units(y: npt.NDArray) -> npt.NDArray:
        return np.array(
            [to_log(v) if k in one_sided else v for k, v in enumerate(y)]
        )

    u0, u1, u2 = units(y0), units(y1), units(y2)
    last, before = np.abs(u2 - u1), np.abs(u1 - u0)
    size = np.maximum(np.abs(u2), floor)
    rate_last, rate_before = last / abs(w2 - w1), before / abs(w1 - w0)
    moving = (rate_last >= 0.5 * rate_before) & (last > 1e-6 * size)
    return tuple(int(k) for k in np.flatnonzero(moving))


def _tangent(
    fun: Callable, y: npt.NDArray, j: int, w: float
) -> "npt.NDArray | None":
    """How the other parameters' best values for parameter ``j`` move
    with ``w`` (``j``'s log distance from its bound) at ``y``, a point of
    its profile: ``-H_oo^{-1} H_oj du/dw`` (``j``'s own entry 0), or
    ``None`` where the Hessian there is not finite (far along a run-off,
    where the walk follows the line through its last two points)."""
    derivatives = filled_derivatives(fun, y)
    if derivatives is None:
        return None
    H = derivatives[0]
    others = [i for i in range(y.size) if i != j]
    if not np.all(np.isfinite(H)):
        return None
    out = np.zeros(y.size)
    try:
        out[others] = -np.linalg.solve(H[np.ix_(others, others)], H[others, j])
    except np.linalg.LinAlgError:
        return None
    out = out * (1.0 if w < 0.0 else np.exp(w))
    return out if np.all(np.isfinite(out)) else None


def walk_profile(
    fun: Callable,
    x: npt.ArrayLike,
    j: int,
    one_sided: "tuple[int, ...]" = (),
    floor: "float | npt.ArrayLike" = 1.0,
) -> "ProfileVerdict | None":
    """Walk the profile of the one-bounded parameter ``j`` of ``fun`` (the
    negative log-likelihood the search minimised) from ``x``, the point
    where the search stopped, on the log scale of the parameter's distance
    from its bound (see the module's notes). ``None`` where it says
    nothing: the profile cannot be computed, or the walk ends before the
    profile has risen for ``FEWEST_EFOLDS``.

    The steps start at a quarter of an e-fold and double while the
    profile rises; each point's search starts from the line along the
    profile through the last (its tangent, or the line through the last
    two points), and a step after which the profile falls is halved before
    the fall is taken as a maximum passed. ``one_sided`` are the positions
    of all the parameters with one bound and ``floor`` each parameter's
    least unit (as for ``runaways_in_units``), for naming the parameters
    that run off with ``j``."""
    at = np.asarray(x, dtype=float)
    floors = np.broadcast_to(np.asarray(floor, dtype=float), at.shape)

    def point(guess: npt.NDArray, w: float) -> "tuple | None":
        return profile_point(fun, guess, j, w)

    def towards(points: "list[tuple]", w: float) -> "tuple | None":
        """The profile's point at ``w``, from the line through the last of
        ``points``."""
        wb, _, yb = points[-1]
        slope = _tangent(fun, yb, j, wb)
        if slope is None and len(points) > 1:
            wa, _, ya = points[-2]
            slope = (yb - ya) / (wb - wa)
        if slope is None:
            slope = np.zeros(yb.size)
        guess = yb + slope * (w - wb)
        out = point(np.where(np.isfinite(guess), guess, yb), w)
        if out is None or (out[0] > points[-1][1] and slope.any()):
            # The line led astray: from the last point itself
            plain = point(yb, w)
            if plain is not None and (out is None or plain[0] < out[0]):
                out = plain
        return None if out is None else (w, *out)

    w0 = to_log(float(at[j]))
    start = point(at, w0)
    if start is None:
        return None
    here = (w0, *start)
    tol = 64.0 * np.finfo(float).eps * max(1.0, abs(here[1]))
    sides = {}
    for s in (1.0, -1.0):
        found = towards([here], w0 + s * FIRST_STEP)
        if found is not None:
            sides[s] = found
    lower = [s for s in sides if sides[s][1] < here[1] - tol]
    if not lower:
        if len(sides) < 2:
            return None
        f, y = _bracket_minimum(point, sides[-1.0], here, sides[1.0])
        return ProfileVerdict("interior", OptimizeResult(x=y, fun=f))
    s = min(lower, key=lambda s: sides[s][1])
    points = [here, sides[s]]
    step = FIRST_STEP
    while abs(points[-1][0] - w0) < WALK_EFOLDS:
        step *= 2.0
        w = points[-1][0] + s * step
        if not np.isfinite(from_log(w)):
            return _run_off(points, j, one_sided, floors, tol)
        found = towards(points, w)
        risen = None
        for _ in range(3):
            if found is not None and not found[1] > points[-1][1] + tol:
                break
            # Risen again, or a step too long to follow: a shorter one
            risen = found or risen
            step *= 0.5
            found = towards(points, points[-1][0] + s * step)
        if found is not None and found[1] > points[-1][1] + tol:
            found, risen = None, found
        if found is None:
            if risen is None:
                break
            # A maximum passed
            f, y = _bracket_minimum(point, points[-2], points[-1], risen)
            return ProfileVerdict("interior", OptimizeResult(x=y, fun=f))
        points.append(found)
    else:
        return _run_off(points, j, one_sided, floors, tol)
    if abs(points[-1][0] - w0) < FEWEST_EFOLDS:
        return None
    return _run_off(points, j, one_sided, floors, tol)


def _run_off(
    points: "list[tuple]",
    j: int,
    one_sided: "tuple[int, ...]",
    floors: npt.NDArray,
    tol: float,
) -> "ProfileVerdict | None":
    """The run-off the walk's ``points`` show, with the parameters that run
    with ``j``; ``None`` where the profile has not risen at all."""
    if not points[0][1] - points[-1][1] > tol:
        return None
    running = {j}
    if len(points) >= 3:
        running.update(_running(points, one_sided, floors))
    return ProfileVerdict("run-off", running=tuple(sorted(running)))
