"""Whether a likelihood has no finite maximum along a parameter (#392).

Newton's test, along each parameter's profile, of whether the point an
optimiser returned is on the way to a supremum the likelihood never
reaches, a parameter running off to infinity (or a limit of its range).
The regression fits apply it to their covariate coefficients
(``judge_search`` in ``univariate/regression/_fit_skeleton.py``), and the
univariate maximum-likelihood fit to every free parameter of a search
that stopped short of a verified maximum (``fitters/mle.py``, #584). The
reasoning below is written for a coefficient. Newton's test itself holds
for any parameter; the gate (``_cleared``) leans on ``exp`` keeping a
linear predictor finite, which another parameter need not obey (an
ExpoWeibull's ``mu``, searched linearly, ran to 8e19), and there it can
only let a runaway pass unseen, never call a maximum a runaway. The
univariate fit adds its own conditions (``_Judge.keep`` in ``mle.py``).
"""

import warnings
from typing import Any, Callable

import autograd.numpy as np
import numpy.typing as npt
from autograd import grad, hessian, value_and_grad
from autograd.differential_operators import make_hvp

#: The largest linear predictor exp can take, log(largest float).
LOG_MAX = float(np.log(np.finfo(float).max))


# -- no finite maximum (#392) -------------------------------------------------
#
# A covariate that separates the events from the survivors -- one level of it
# with no events, say -- gives a likelihood that keeps increasing as its
# coefficient grows, towards a supremum it never reaches. The optimisers stop
# wherever the rise has become too small for their tolerances (a WeibullPH
# coefficient of -16, a PO one of +33), report success, and the fit used to
# be returned silently. CoxPH detects its own case from the collapse of the
# information (``cox_ph._warn_if_monotone``); its root-finder runs on until
# the collapse is complete, whereas these optimisers stop part-way, at a
# point set by their tolerances, so a fixed collapse ratio cannot tell.
#
# Newton's method can. Along a coefficient's profile the log-likelihood of
# such data approaches its supremum like C - A exp(-s t) (or with a Gaussian
# tail, for a LogNormal AFT), and at every point on the way the Newton step
# is as long as the distance over which the curvature itself falls away:
# the next step is the same length again, and Newton's method never
# converges. Kantorovich's theorem makes that the test. For the negative
# log-likelihood f along the profile, the Newton step from the fit is
# -f'/f'', and Newton's method is guaranteed to converge to a minimum within
# twice that distance if h = |f'''| |f'| / f''^2 <= 1/2 (the relative change
# of the curvature over one step, with |f'''| its local bound). At a fit that
# has reached a maximum the step is at the level of the optimiser's
# tolerance, and so is h (at most 2e-5 on the ordinary fits of the
# conformance registry, and 2e-4 over the 1360 refits of their calibration
# study); on the way to a supremum h is 1 (exactly, for an exponential
# tail) wherever the optimiser stopped, and the curvature falls in the
# direction the likelihood rises (f' f''' > 0). A curvature that is zero or
# negative there is no maximum either: the additive hazards likelihood rises
# linearly as a no-event level's coefficient falls, without bound.
#
# Reading a profile costs about three gradients, traced for Hessian-vector
# products at the polished point and a quarter of a Newton step either side
# (#501: it took a third derivative and two full Hessians, two to five times
# as long on a 100,000-row AFT), so a coefficient's is read only when
# Newton's method has not already shown the fit to be a maximum in it
# (``_cleared``, which costs one Hessian, needed for the covariance
# anyway). The coefficient's part of the
# Newton step -H^{-1} g is at the level of the optimiser's tolerance at a
# maximum. On the way to a supremum it is 1/s, however far the optimiser
# went: write the gradient as H d plus the tail's s A e^{-st} along the flat
# direction u, d the optimiser's leftover displacement of the other
# parameters; along u, H d is the curvature s^2 A e^{-st} times d's small
# component, so the step along u is 1/s plus that component, and the
# leftover error cannot hide the runaway (which it does on the profile line
# until polished, see ``_profile``). The coefficient itself is then about t,
# and s t is the linear predictor the runaway drives, which exp keeps within
# log(largest float) = 709.8 of 0: beyond it the rows it moves underflow and
# the likelihood no longer depends on the coefficient at all. So a runaway's
# step is at least 1/709.8 of its size (measured: 1/100 to 1/5 on every
# runaway in the conformance registry), and a coefficient with a smaller
# step has converged. A larger step (at most 1/5900 of the coefficient on
# the registry's ordinary fits, and on a few of the calibration refits
# more), or a Hessian that is not positive definite, has the profile read,
# which only costs time.


def search_derivatives(
    neg_ll: Callable, x: npt.ArrayLike
) -> "tuple[npt.NDArray, npt.NDArray] | None":
    """``(H, g)``, the Hessian and gradient of ``neg_ll`` at ``x`` by
    autograd, from one trace; ``None`` for an objective autograd cannot
    differentiate. The no-maximum check reads them, and the fitted model
    keeps the Hessian for its covariance (:func:`keep_information`)."""
    at = np.asarray(x, dtype=float)
    try:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Output seems independent")
            hvp, g = make_hvp(neg_ll)(at)
            H = np.array([hvp(e) for e in np.eye(at.size)], dtype=float)
            return H, np.asarray(g, dtype=float)
    except (TypeError, ValueError, ArithmeticError):
        return None


def runaway_coefficients(
    neg_ll: Callable,
    x: npt.ArrayLike,
    coefs: "list[int]",
    start: "npt.ArrayLike | None" = None,
    derivatives: "tuple[npt.NDArray, npt.NDArray] | None" = None,
    floor: "float | npt.ArrayLike" = 0.0,
    keep: "Callable[[int, float], bool] | None" = None,
) -> "list[int]":
    """The positions in ``coefs`` of the parameters along which the
    likelihood has no finite maximum near ``x``.

    ``neg_ll`` is the negative log-likelihood the optimiser minimised,
    differentiable by autograd, ``x`` the point it returned (in its search
    space), ``coefs`` the positions in ``x`` of the parameters to check
    (the covariate coefficients) and ``start`` the point the search began
    from. Each is checked along its profile: the coefficient moves by 1 and
    the other parameters by the amounts that keep them at their best values
    for it (to first order), after the other parameters are first brought
    to their best values for the fitted coefficient (the optimiser stops
    them only as close as its tolerance, and on a flat profile that residue
    would swamp its derivatives). Where the profile cannot be formed -- the
    other parameters have no curvature either, in a fit that has run off in
    several directions at once -- the coefficient's own axis is used. It
    runs away when Newton's method cannot converge along that line, the
    Kantorovich test described above.

    There is no verdict (an empty list) for an objective autograd cannot
    differentiate, nor along a line on which the derivatives are not
    finite, or on which the likelihood does not change at all: at ``x``,
    or, with ``start``, at the start either (see
    :func:`_flat_at_start`), as it does not along a combination of
    collinear covariates, whose coefficients are not identified rather than
    infinite. ``derivatives`` are those of :func:`search_derivatives` at
    ``x``, if the caller has them, and ``floor`` the least size of each
    parameter for :func:`_cleared` (none by default). ``keep(j, slope)``,
    where given, must also hold of a parameter ``j`` that runs away, with
    ``slope`` the derivative of ``neg_ll`` along the line it was judged
    on: a caller whose answers are not all where an optimiser stopped on a
    flat rise says so there (``fitters.mle``).
    """
    at = np.asarray(x, dtype=float)
    if derivatives is None:
        derivatives = search_derivatives(neg_ll, at)
    if derivatives is None:
        return []
    H, g = derivatives
    cleared = _cleared(at, H, g, floor)
    out = []
    at_start: "tuple[Any] | None" = None  # derivatives at start, if needed
    for k, j in enumerate(coefs):
        if cleared[j]:
            continue
        axis = np.zeros(at.size)
        axis[j] = 1.0
        lines = [(at, axis)]
        runaway = None
        slope = 0.0  # along the line judged on
        with np.errstate(all="ignore"):
            # The profile's trial points can sit where the hazard is 0
            # (log 0): a non-finite value ends its polish, quietly.
            if np.all(np.isfinite(H)):
                lines.insert(0, _profile(neg_ll, at, H, j))
            for point, v in lines:
                d = _line_derivatives(neg_ll, point, v)
                if d is None:
                    continue
                slope = d[0]
                if d[0] != 0.0:
                    runaway = _no_convergence(neg_ll, point, v, d, j, H)
                if runaway is not None or d[0] == 0.0:
                    break
        if runaway and keep is not None and not keep(j, slope):
            continue
        if runaway:
            if start is None:
                out.append(k)
                continue
            if at_start is None:
                at_start = (_start_derivatives(neg_ll, start),)
            if not _flat_at_start(neg_ll, start, v, at_start[0]):
                out.append(k)
    return out


# -- the units the check is made in (#628) -------------------------------------
#
# Newton's test is unchanged by a linear change of units in exact arithmetic,
# but its parts are not in floating point: the pseudo-inverse of the other
# parameters' Hessian that forms a profile drops any parameter whose curvature
# is below 1e-15 of the largest, and a regression's centred scale searched
# linearly at 3.6e6 (curvature 1e-17, against 5e2 for the shape) was dropped
# that way, so that a WeibullPH coefficient running off had its profile
# formed without it and the test said nothing. Nor is it unchanged by a
# nonlinear one: a scale searched linearly runs off exponentially along a
# separating direction (the log of the scale moves with the coefficients'
# linear predictor), so the run-off is a curve in the search space, and an
# accelerated life ``c`` searched linearly at 1e22, a fit merely stopped
# short of a finite maximum, read as running off (Newton's step along a line
# in ``c`` is not Newton's step along ``log c``). So the regressions make
# the test with each parameter that has one bound (searched linearly beyond
# a unit from it, see ``bounds_convert``) as the log of its distance from the
# bound, where a run-off is a straight line, and every parameter in units of
# its size there (at least its unit, ``floor``: a covariate coefficient's is
# its covariate's, ``coefficient_floor``), where the Hessian is conditioned.


def runaways_in_units(
    neg_ll: Callable,
    x: npt.ArrayLike,
    coefs: "list[int]",
    start: "npt.ArrayLike | None" = None,
    derivatives: "tuple[npt.NDArray, npt.NDArray] | None" = None,
    floor: "float | npt.ArrayLike" = 1.0,
    one_sided: "tuple[int, ...]" = (),
) -> "list[int]":
    """The positions in ``coefs`` of the parameters along which the
    likelihood has no finite maximum near ``x``, each running off alone
    (:func:`runaway_coefficients`) or with others (:func:`joint_runaway`),
    judged in the units described above.

    ``neg_ll``, ``x``, ``coefs`` and ``start`` are as for
    :func:`runaway_coefficients`, in the search space; ``one_sided`` are the
    positions in ``x`` of the parameters searched as ``bounds_convert`` maps
    one with a single bound (``log`` within a unit of it, linear beyond),
    which are judged as the log of that distance; ``floor`` is each
    parameter's least unit (one for those); ``derivatives`` are those of
    :func:`search_derivatives` at ``x``, if the caller has them, which are
    carried to the new units by the chain rule rather than taken again."""
    at = np.asarray(x, dtype=float)
    log = np.zeros(at.size, dtype=bool)
    log[list(one_sided)] = True
    with np.errstate(all="ignore"):
        u = np.where(log & (at >= 0.0), np.log1p(np.abs(at)), at)
        unit = np.where(
            log, 1.0, np.broadcast_to(np.asarray(floor, dtype=float), at.shape)
        )
        size = np.maximum(np.abs(u), unit)
    if not np.all(np.isfinite(size) & (size > 0.0)):
        return []

    def from_units(v: Any) -> Any:
        w = size * v
        return np.where(log & (w >= 0.0), np.expm1(np.minimum(w, LOG_MAX)), w)

    def in_units(v: Any) -> Any:
        return neg_ll(from_units(v))

    v0 = u / size
    if derivatives is None:
        derivatives = search_derivatives(neg_ll, at)
    if derivatives is None:
        return []
    H, g = derivatives
    # The chain rule for x = from_units(v): dx/dv and d2x/dv2, one
    # coordinate at a time.
    linear = log & (at >= 0.0)
    d1 = size * np.where(linear, at + 1.0, 1.0)
    d2 = size**2 * np.where(linear, at + 1.0, 0.0)
    with np.errstate(all="ignore"):
        units_derivatives = (
            np.outer(d1, d1) * H + np.diag(g * d2),
            d1 * g,
        )
    v_start = None
    if start is not None:
        s = np.asarray(start, dtype=float)
        with np.errstate(all="ignore"):
            s = np.where(log & (s >= 0.0), np.log1p(np.abs(s)), s)
        v_start = s / size
    out = runaway_coefficients(
        in_units, v0, coefs, v_start, units_derivatives
    )
    if not out:
        out = joint_runaway(
            in_units, v0, coefs, v_start, units_derivatives, unit / size
        )
    if not out:
        out = flat_profiles(in_units, v0, coefs, v_start, units_derivatives)
    return out


def flat_profiles(
    neg_ll: Callable,
    x: npt.ArrayLike,
    coefs: "list[int]",
    start: "npt.ArrayLike | None",
    derivatives: "tuple[npt.NDArray, npt.NDArray]",
) -> "list[int]":
    """The positions in ``coefs`` of the parameters whose profile has no
    curvature at ``x`` to rounding, though the likelihood depends on them at
    ``start``: they have run so far that the rows they move no longer count
    (see above), where neither Newton's test nor any other made with
    derivatives can say more.

    A maximum's profile curves down: its curvature is the estimate's
    precision. Here the curvature, the Schur complement of the Hessian
    ``H`` in the parameter, is within the rounding of ``H`` itself (the
    tolerance of ``numpy.linalg.matrix_rank``, ``size * eps * ||H||``), as
    on a WeibullPO run along a separating direction to coefficients of
    1.6e6 and -668 (a profile curvature of 1e-13 beside a largest of 1e8).
    A parameter that does not enter the likelihood at all is flat at the
    start too, and is left out (:func:`_flat_at_start`).

    ``_cleared`` is no guide here: a Newton step computed from a Hessian
    singular to rounding is rounding itself, and it cleared an accelerated
    life ``a`` of 1.2e5 running off with ``log c`` at -266 (a profile
    curvature of 2e-9 beside a largest of 6e6)."""
    H, g = derivatives
    at = np.asarray(x, dtype=float)
    if start is None or not (np.all(np.isfinite(H)) and np.all(np.isfinite(g))):
        return []
    tol = at.size * float(np.finfo(float).eps) * np.linalg.norm(H, 2)
    out = []
    for k, j in enumerate(coefs):
        others = [i for i in range(at.size) if i != j]
        v = np.zeros(at.size)
        v[j] = 1.0
        if others:
            pinv = np.linalg.pinv(H[np.ix_(others, others)])
            v[others] = -pinv @ H[others, j]
        curvature = float(v @ H @ v)
        if abs(curvature) <= tol and not _flat_at_start(neg_ll, start, v):
            out.append(k)
    return out


# -- several coefficients running off together (#628) --------------------------
#
# A coefficient's profile is the likelihood with every other parameter at its
# best value for it, and the test above assumes the others have one. Where
# two or more coefficients run off together -- all the failures in one corner
# cell of a two-stress design, so that any direction in a cone of the two
# coefficients raises the likelihood -- the other's best value is at infinity
# too, its "profile" is not a curve, and Newton's test along it said nothing
# (a WeibullAFT with both coefficients at -52674 and 70, the scale at 1e134,
# was returned as a verified maximum). The test is then made along the line
# Newton's method itself would move those coefficients on: the joint profile
# of the direction ``w`` of their part of the Newton step ``-H^{-1} g``,
# every other parameter (the distribution's, and the coefficients that have
# converged) at its best value for each point on it. On the way to a
# supremum each coefficient's part of the step is ``1/s_j`` for its own
# tail, the step's tails all fall together along ``w``, and the joint profile
# is ``C - A exp(-t)``: Kantorovich's ``h`` is 1, as for one coefficient (for
# a sum of exponential tails all falling along ``w``, ``h >= 1`` by the
# Cauchy-Schwarz inequality). At a maximum the step is at the level of the
# optimiser's tolerance and the test cannot fire.


def joint_runaway(
    neg_ll: Callable,
    x: npt.ArrayLike,
    coefs: "list[int]",
    start: "npt.ArrayLike | None" = None,
    derivatives: "tuple[npt.NDArray, npt.NDArray] | None" = None,
    floor: "float | npt.ArrayLike" = 0.0,
) -> "list[int]":
    """The positions in ``coefs`` of the parameters that run off together
    from ``x``: Newton's method cannot converge along their joint profile
    in the direction of their part of the Newton step (see above), and
    the likelihood is not flat along it at ``start`` (as
    :func:`runaway_coefficients`); an empty list otherwise.

    The parameters tested are those ``_cleared`` (with each one's size at
    least ``floor``) does not show to be at a maximum, and only where there
    are two or more of them: one alone is
    :func:`runaway_coefficients`'s. There is no verdict where the Hessian
    is not finite and positive definite (a fit run into the limits of
    floating point), as there is no Newton step. ``derivatives`` are those
    of :func:`search_derivatives` at ``x``, if the caller has them."""
    at = np.asarray(x, dtype=float)
    if derivatives is None:
        derivatives = search_derivatives(neg_ll, at)
    if derivatives is None:
        return []
    H, g = derivatives
    if not (np.all(np.isfinite(H)) and np.all(np.isfinite(g))):
        return []
    cleared = _cleared(at, H, g, floor)
    loose = [k for k, j in enumerate(coefs) if not cleared[j]]
    if len(loose) < 2:
        return []
    try:
        np.linalg.cholesky(H)
        step = -np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        return []
    moving = np.array([coefs[k] for k in loose])
    rest = np.array([i for i in range(at.size) if i not in moving], dtype=int)
    w = np.zeros(at.size)
    w[moving] = step[moving] / np.max(np.abs(step[moving]))
    if not np.all(np.isfinite(w)):
        return []

    def along(y: Any) -> Any:
        # The likelihood with the moving coefficients on the line through
        # ``at`` along ``w`` (the last coordinate of ``y`` the distance
        # along it) and the other parameters free.
        tau = y[rest.size]
        full = [None] * at.size
        for r, i in enumerate(rest):
            full[i] = y[r]
        for i in moving:
            full[i] = at[i] + tau * w[i]
        return neg_ll(np.array(full))

    y0 = np.append(at[rest], 0.0)
    with np.errstate(all="ignore"):
        if not runaway_coefficients(along, y0, [rest.size]):
            return []
        if start is not None:
            # The joint profile's direction, to first order, for the check
            # that the likelihood depends on it at the start (collinear
            # covariates do not run off; they are not identified).
            v = w.copy()
            if rest.size:
                H_rr = H[np.ix_(rest, rest)]
                v[rest] = -np.linalg.pinv(H_rr) @ (H[np.ix_(rest, moving)] @ w[moving])
            if _flat_at_start(neg_ll, start, v):
                return []
    return loose


def _cleared(
    x: npt.NDArray,
    H: npt.NDArray,
    g: npt.NDArray,
    floor: "float | npt.ArrayLike" = 0.0,
) -> npt.NDArray:
    """Which parameters Newton's method shows to be at a maximum at ``x``,
    ``H`` and ``g`` the Hessian and gradient there: those whose part of the
    Newton step ``-H^{-1} g`` is no more than ``1 / log(largest float)`` of
    their size, which no parameter running off to a supremum can be (see
    above).

    Parameters the likelihood does not depend on at ``x`` to second order
    (a zero gradient and Hessian row, as a frailty variance held at its
    limit of 0 has) are left out of the step. None is cleared where the
    Hessian of the others is not finite and positive definite: a maximum
    has one, and a runaway's may not (a linear rise has no curvature).

    A parameter's size is taken as at least ``floor`` (per parameter, or
    one for all), the unit the search measures it in: a caller that judges
    every answer, most of them maxima with parameters near 0 in their
    search units, then reads the profiles of no more of them than the
    curvature calls for. It can only clear more: a runaway that has not
    gone that far is not caught."""
    cleared = np.zeros(x.size, dtype=bool)
    if not (np.all(np.isfinite(H)) and np.all(np.isfinite(g))):
        return cleared
    used = np.flatnonzero(np.any(H != 0, axis=1) | (g != 0))
    H_u = H[np.ix_(used, used)]
    try:
        np.linalg.cholesky(H_u)
        step = np.linalg.solve(H_u, g[used])
    except np.linalg.LinAlgError:
        return cleared
    with np.errstate(all="ignore"):
        # A runaway's Newton step is at least 1 / LOG_MAX of its
        # coefficient's size (see above).
        size = np.maximum(np.abs(x), np.asarray(floor, dtype=float))
        cleared[used] = np.abs(step) * LOG_MAX <= size[used]
    return cleared


def _no_convergence(
    neg_ll: Callable,
    point: npt.NDArray,
    v: npt.NDArray,
    d: "tuple[float, ...]",
    j: int,
    H: npt.NDArray,
) -> "bool | None":
    """Whether Newton's method cannot be shown to converge along the
    profile of parameter ``j`` through ``point`` (direction ``v``, with
    ``v[j] = 1``), where the objective's first two derivatives are ``d``:
    it has no curvature, or Kantorovich's ``h = |f'''| |f'| / f''^2`` is
    above 1/2 with the curvature falling the way the likelihood rises (see
    above). ``H`` is the Hessian at the fit, near ``point``.

    ``f'''`` is the rate of change of the profile's curvature, the Schur
    complement of the Hessian in ``j``, from the curvature a quarter of a
    Newton step either side. Autograd's third derivative along the line
    was rounding noise at a stopped runaway, where the derivatives are
    near 1e-7: its sign changed with the build, so a WeibullAFT with a
    fixed coefficient warned on one Python and not on another. The
    curvature needs second derivatives only, which are well conditioned
    there. Where it cannot be formed, the line's own third derivative is
    used, and ``None`` (no verdict along this line) is returned where that
    is not finite either."""
    d1, d2 = d[:2]
    if not d2 > 0.0:
        return True
    half = 0.125 * abs(d1 / d2)
    d3 = None
    if half > 0.0:
        ahead = _profile_curvature(neg_ll, point + half * v, j, H, v)
        behind = _profile_curvature(neg_ll, point - half * v, j, H, v)
        if ahead is not None and behind is not None:
            d3 = (ahead - behind) / (2.0 * half)
    if d3 is None:
        line = _line_derivatives(neg_ll, point, v, order=3)
        if line is None:
            return None
        d3 = line[2]
    try:
        return d1 * d3 > 0.5 * d2**2
    except OverflowError:
        # A curvature past 1e154 (an ExpoWeibull's shape at 1e3) is no
        # flat profile: Newton's method converges there.
        return False


def _profile_curvature(
    neg_ll: Callable,
    point: npt.NDArray,
    j: int,
    H: "npt.NDArray | None" = None,
    v: "npt.NDArray | None" = None,
) -> "float | None":
    """The curvature of the profile of parameter ``j`` at ``point``: the
    Schur complement ``H_jj - H_jo H_oo^+ H_oj`` of its Hessian, over the
    other parameters the likelihood depends on there (a frailty variance
    held at its limit has a zero row). ``None`` where the Hessian is not
    finite or cannot be taken.

    With ``H``, the Hessian at a point near ``point`` (the fit), and ``v``
    the profile direction there, the complement is found without forming
    the Hessian at ``point`` (:func:`_schur_by_products`), which costs
    one or two Hessian-vector products instead of one per parameter; the
    Hessian is formed where that does not converge."""
    if H is not None and v is not None:
        S = _schur_by_products(neg_ll, point, j, H, v)
        if S is not None:
            return S
    derivatives = search_derivatives(neg_ll, point)
    if derivatives is None or not np.all(np.isfinite(derivatives[0])):
        return None
    H = derivatives[0]
    others = [i for i in range(H.shape[0]) if i != j and np.any(H[i] != 0.0)]
    if not others:
        return float(H[j, j])
    H_oo = H[np.ix_(others, others)]
    H_oj = H[others, j]
    return float(H[j, j] - H_oj @ np.linalg.pinv(H_oo) @ H_oj)


#: The relative accuracy to which :func:`_schur_by_products` finds a
#: profile's curvature: far below the relative change of the curvature over
#: a quarter of a Newton step that the Kantorovich test reads (``h / 4``,
#: about 1/4 at a runaway, against a threshold of 1/8), and at the level of
#: the rounding of a full Hessian's Schur complement, which loses digits to
#: cancellation where the profile is flat (1e-8 of it on a runaway in
#: ``test_no_maximum.py``, where the products are exact).
_SCHUR_RTOL = 1e-12


def _schur_by_products(
    neg_ll: Callable,
    point: npt.NDArray,
    j: int,
    H: npt.NDArray,
    v: npt.NDArray,
) -> "float | None":
    """The Schur complement of :func:`_profile_curvature` at ``point``, by
    Hessian-vector products there. It is the minimum of ``w' H(point) w``
    over the ``w`` with ``w_j = 1`` (and 0 for a parameter whose row of
    ``H`` is zero), which conjugate gradients find from the fit's profile
    direction ``v``, preconditioned by ``H``'s block of the other
    parameters, the Hessian a short step away. Near a maximum ``v`` is the
    answer to rounding and one product is taken; on a runaway's plateau,
    a few. The error in the minimum is ``r' H_oo^{-1} r`` for the residual
    ``r``, which is run down to :data:`_SCHUR_RTOL` of it. ``None`` where
    a product is not finite or it has not converged in as many steps as
    there are parameters, plus one."""
    others = [i for i in range(H.shape[0]) if i != j and np.any(H[i] != 0.0)]
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Output seems independent")
            hvp = make_hvp(neg_ll)(point)[0]
            w = np.zeros(point.size)
            w[j] = 1.0
            w[others] = v[others]
            Hw = np.asarray(hvp(w), dtype=float)
            if not others:
                return float(Hw[j]) if np.isfinite(Hw[j]) else None
            M = np.linalg.pinv(H[np.ix_(others, others)])
            r = -Hw[others]
            z = M @ r
            rz = float(r @ z)
            p = z
            for _ in range(len(others) + 2):
                S = float(w @ Hw)
                if not (np.isfinite(S) and np.all(np.isfinite(Hw))):
                    return None
                if abs(rz) <= _SCHUR_RTOL * abs(S):
                    return S
                u = np.zeros(point.size)
                u[others] = p
                Hu = np.asarray(hvp(u), dtype=float)
                pAp = float(p @ Hu[others])
                if not (np.isfinite(pAp) and pAp > 0.0 and rz > 0.0):
                    return None
                alpha = rz / pAp
                w = w + alpha * u
                Hw = Hw + alpha * Hu
                r = r - alpha * Hu[others]
                z = M @ r
                rz, rz_old = float(r @ z), rz
                p = z + (rz / rz_old) * p
    except (TypeError, ValueError, ArithmeticError, np.linalg.LinAlgError):
        return None
    return None


def _start_derivatives(
    neg_ll: Callable, start: npt.ArrayLike
) -> "tuple[npt.NDArray, npt.NDArray] | None":
    """The gradient and Hessian of ``neg_ll`` at ``start``, for
    :func:`_flat_at_start`; ``None`` where they cannot be taken or are not
    finite."""
    x0 = np.asarray(start, dtype=float)
    try:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Output seems independent")
            g0 = np.asarray(grad(neg_ll)(x0), dtype=float)
            H0 = np.asarray(hessian(neg_ll)(x0), dtype=float)
    except (TypeError, ValueError, ArithmeticError):
        return None
    if not (np.all(np.isfinite(g0)) and np.all(np.isfinite(H0))):
        return None
    return g0, H0


def _flat_at_start(
    neg_ll: Callable,
    start: npt.ArrayLike,
    v: npt.NDArray,
    derivatives: Any = False,
) -> bool:
    """Whether ``neg_ll`` has neither slope nor curvature along ``v`` at
    ``start``, to rounding: its derivatives along ``v`` within ``size *
    eps`` of the size of its gradient and Hessian there (the tolerance of
    ``numpy.linalg.matrix_rank``), so that ``v`` is a direction the
    likelihood does not depend on at all. A likelihood running off to a
    supremum does depend on it, most of all near the start; one whose
    covariates are collinear (each level of a factor coded, with no
    intercept) does not, anywhere, and is no concern of this check (CoxPH
    warns of it as collinear). ``derivatives`` are those of
    :func:`_start_derivatives` at ``start``, if the caller has them."""
    if derivatives is False:
        derivatives = _start_derivatives(neg_ll, start)
    if derivatives is None:
        return False
    g0, H0 = derivatives
    tol = g0.size * float(np.finfo(float).eps)
    size = float(np.dot(v, v))
    slope = abs(float(np.dot(g0, v)))
    curvature = abs(float(v @ H0 @ v))
    return bool(
        slope <= tol * np.linalg.norm(g0) * np.sqrt(size)
        and curvature <= tol * np.linalg.norm(H0, 2) * size
    )


def _profile(
    neg_ll: Callable, x: npt.NDArray, H: npt.NDArray, j: int
) -> "tuple[npt.NDArray, npt.NDArray]":
    """``(point, direction)``: parameter ``j``'s profile line (see
    :func:`runaway_coefficients`), with ``H`` the Hessian at ``x``. The
    direction is not finite where it cannot be found."""
    rest = [i for i in range(x.size) if i != j]
    v = np.zeros(x.size)
    v[j] = 1.0
    point = x.copy()
    if not rest:
        return point, v
    try:
        # The pseudo-inverse: a parameter with no curvature to rounding (a
        # frailty variance at its boundary of 0) is not moved.
        inv_rr = np.linalg.pinv(H[np.ix_(rest, rest)])
        v[rest] = -inv_rr @ H[rest, j]
        # Newton steps on the other parameters, with the coefficient held
        # and their Hessian held at its value at x; a step that does not
        # lower the objective ends it (it has converged to rounding, or the
        # Hessian is no guide there).
        gradient = grad(neg_ll)
        f0 = float(neg_ll(point))
        for _ in range(_POLISH_STEPS):
            step = inv_rr @ gradient(point)[rest]
            trial = point.copy()
            trial[rest] = trial[rest] - step
            f1 = float(neg_ll(trial))
            if not (np.isfinite(f1) and f1 < f0):
                break
            point, f0 = trial, f1
    except (TypeError, ValueError, ArithmeticError, np.linalg.LinAlgError):
        v[rest] = np.nan
    return point, v


def _line_derivatives(
    neg_ll: Callable, point: npt.NDArray, v: npt.NDArray, order: int = 2
) -> "tuple[float, ...] | None":
    """The first ``order`` (2 or 3) derivatives of ``neg_ll`` at ``point``
    along ``v``, or ``None`` where they are not finite (or cannot be
    taken). The third is taken only where :func:`_no_convergence` cannot
    form the profile's curvature: it costs several times the first two
    (#501)."""
    if not np.all(np.isfinite(v)):
        return None
    moving = v != 0

    def line(t: Any) -> Any:
        return neg_ll(point + t * v)

    def moving_only(t: Any) -> Any:
        # The parameters that do not move held as constants, so that a
        # derivative that is not finite in one of them (a Gamma baseline's
        # shape near 0) cannot reach the line's.
        return neg_ll(
            np.array(
                [p + t * u if m else p for p, u, m in zip(point, v, moving)]
            )
        )

    out: "tuple[float, ...]"
    for along in (line,) if moving.all() else (line, moving_only):
        try:
            with warnings.catch_warnings():
                # autograd says so of a derivative that is constant (a
                # likelihood linear along the line); it is 0, not a fault
                warnings.filterwarnings("ignore", "Output seems independent")
                if order == 2:
                    d1, d2 = value_and_grad(grad(along))(0.0)
                    out = float(d1), float(d2)
                else:
                    d1 = grad(along)(0.0)
                    d2, d3 = value_and_grad(grad(grad(along)))(0.0)
                    out = float(d1), float(d2), float(d3)
        except (TypeError, ValueError, ArithmeticError):
            return None
        if np.all(np.isfinite(out)):
            return out
    return None


#: The most Newton steps taken on the other parameters before a profile is
#: read (see ``_profile``); they start within the optimiser's
#: tolerance of their optimum, and two or three reach rounding.
_POLISH_STEPS = 10
