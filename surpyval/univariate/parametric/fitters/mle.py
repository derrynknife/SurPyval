import warnings
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

if TYPE_CHECKING:
    from ..parametric import Parametric

import autograd.numpy as np
import numpy.typing as npt
from autograd import hessian, jacobian
from autograd.numpy.linalg import inv
from scipy.optimize import OptimizeResult

from surpyval.univariate.parametric.fitters import (
    OPTIMUM_GTOL,
    Gradient,
    _usable,
    is_local_minimum,
    minimize_with_gradient,
    preconditioned_bfgs,
    search_floor,
)
from surpyval.univariate.parametric.fitters.runaway import (
    runaway_coefficients,
    search_derivatives,
)

# The optimiser ladder: gradient methods first, then the derivative-free
# fallbacks. ``None`` stands for "no derivative"; "jac" and "hess" for the
# autograd ones.
_LADDER = (
    ("BFGS", "jac", None),
    ("TNC", "jac", None),
    ("Newton-CG", "jac", "hess"),
    ("Nelder-Mead", None, None),
    ("Powell", None, None),
)

_MLE_FAILED = (
    "MLE Failed; returning the optimiser's starting point "
    "(a probability-plot fit, or a rougher initial guess where "
    "the distribution has none) instead. "
    "Try making the values of the data closer to "
    "1 by dividing or multiplying by some constant."
    "\n\nAlternately try setting the `init` keyword in"
    " the `fit()`"
    " method to a value you believe is closer."
    "A good way to do this is to set any shape parameter to 1. "
    "and any scale parameter to be the mean of the data "
    "(or it's inverse)"
    "\n\nModel returned with the initial guesses."
)


class _Search(NamedTuple):
    """What the optimiser ladder found."""

    res: Any
    optimizer: str
    verified: bool
    #: The positions in the search vector of the parameters along which
    #: the likelihood has no finite maximum (see ``_runaway``); empty
    #: where none was found.
    runaway: tuple[int, ...] = ()
    #: A start off the bound of a parameter whose likelihood rises off it
    #: (``_OnBounds.off``), for the caller to search from (#579).
    off_bound: "npt.NDArray | None" = None
    #: Whether the runaway is the offset's, found at the end of the ladder
    #: by the family's limit fitting the data at least as well as the
    #: answer, rather than by Newton's test (see ``_search``, #616).
    by_limit: bool = False


def _negative_log_likelihood(model: "Parametric") -> Callable[..., Any]:
    """The objective: the negative log-likelihood of the search vector.

    With ``transform`` the vector is in the unbounded search space and
    the fixed parameters are filled in (``const``) before it is mapped to
    the bounded parameters (``inv_trans``). The offset, zero-inflation and
    LFP parameters are then split off the ends.
    """
    # Function that adds in any fixed parameters
    const = model.fitting_info["const"]
    # Inverse transform function for parameters. i.e. from (None, None) to
    # correct bounded values
    inv_trans = model.fitting_info["inv_trans"]

    def fun(
        params: Any,
        offset: bool = False,
        lfp: bool = False,
        zi: bool = False,
        transform: bool = True,
        gamma: float = 0,
        f0: float = 0,
        p: float = 1,
    ) -> Any:
        # Transform parameters from (-Inf, Inf) range to parameter
        # to correct bounded values
        if transform:
            params = inv_trans(const(params))

        # Unpack offset, zi, lfp parameters
        if offset:
            gamma, *params = params

        if zi:
            *params, f0 = params

        if lfp:
            *params, p = params

        return model.dist._neg_ll_func(model.surv_data, *params, gamma, f0, p)

    return fun


def _kept_hessian(
    hess: Callable[..., Any],
) -> tuple[Callable[..., Any], dict]:
    """``hess`` (or any derivative) that keeps its last value, and the
    dict it keeps it in.

    The Hessian at the verified answer (see ``is_local_minimum``) is the
    one the covariance needs too, where no parameter is held: kept, not
    taken twice. On a large sample it is the dearest part of the check.
    """
    hess_at: dict = {}

    def hess_kept(x: npt.NDArray, *args: Any) -> Any:
        key = np.asarray(x, dtype=float).tobytes()
        if key not in hess_at:
            hess_at.clear()
            hess_at[key] = hess(x, *args)
        return hess_at[key]

    return hess_kept, hess_at


def _rung_starts(method: str, init: npt.NDArray, first_success: Any) -> list:
    """Where a rung starts its search.

    From the first rung's answer that reported success, or from the
    initial guess (Nelder-Mead from both, Powell from the guess).
    """
    if method == "Powell" or first_success is None:
        return [init]
    if method == "Nelder-Mead":
        return [init, first_success[0].x]
    return [first_success[0].x]


def _run_rung(
    fun: Callable[..., Any],
    method: str,
    x0: npt.NDArray,
    args: tuple,
    jac_i: Any,
    hess_i: Any,
    floor: Any,
    obj_scale: float,
    callback: "Callable[[npt.NDArray], None] | None" = None,
) -> Any:
    """One search of one rung of the ladder from ``x0``; ``callback``, for
    BFGS, watches its iterates (``_Judge.watch``)."""
    opts = {"maxfun": 1000} if method == "TNC" else {"maxiter": 1000}
    if method == "BFGS":
        # Scaled per parameter (see ``search_floor``) and per
        # observation: the negative log-likelihood itself
        # moves with the data's units, so ``|f(x0)|`` is not a
        # scale free normaliser for it (see
        # ``preconditioned_bfgs``).
        return preconditioned_bfgs(
            fun,
            x0,
            args,
            jac_i,
            opts,
            floor=floor,
            obj_scale=obj_scale,
            callback=callback,
        )
    return minimize_with_gradient(
        fun, x0, args, jac_i, method=method, hess=hess_i, options=opts
    )


#: How many BFGS iterations pass before ``_Judge.watch`` first checks one.
_WATCH_EVERY = 100

#: The relative tolerance of comparing a point's negative log-likelihood
#: with the offset family's limit's (``_Judge.toward_limit``, #627).
_LIMIT_RTOL = 1e-7

#: The most offsets the profile before a runaway by the limit is read at
#: (``_profile_offsets``), the least ratio of their distances below the
#: data, and the most BFGS iterations of each one's search.
_PROFILE_POINTS = 8
_PROFILE_RATIO = 4.0
_PROFILE_ITERATIONS = 100


def _profile_offsets(high: float, start: float, reached: float) -> list:
    """The offsets ``_Judge.interior_beats_limit`` reads the profile at:
    from the search's ``start`` towards the offset it ``reached`` (both
    below ``high``, the data's least value), spaced geometrically in
    their distance below ``high`` (a flat maximum is wide in it), at most
    ``_PROFILE_POINTS`` of them, ``reached`` itself left out (the search's
    own point is there); none where the search did not move down."""
    d_start, d_reached = high - start, high - reached
    if not (0 < d_start < d_reached and np.isfinite(d_reached)):
        return []
    span = np.log(d_reached / d_start)
    ratio = max(_PROFILE_RATIO, np.exp(span / _PROFILE_POINTS))
    count = int(np.ceil(span / np.log(ratio)))
    return [high - d_start * ratio**i for i in range(count)]


def _seed_data(surv_data: Any) -> Any:
    """The data as the distributions' starts read them (as
    ``_initial_guess`` imputes them): an interval at its midpoint, a
    left-censored value as observed, the truncation dropped; ``None``
    where no value is finite."""
    from surpyval.univariate.parametric._fit_inputs import imputed_data

    x = np.asarray(surv_data.x, dtype=float)
    c = np.asarray(surv_data.c)
    n = np.asarray(surv_data.n)
    if x.ndim == 2:
        with np.errstate(all="ignore"):
            x = np.where(np.isfinite(x[:, 1]), x.mean(axis=1), x[:, 0])
    c = np.where(c == 1, 1, 0)
    finite = np.isfinite(x)
    if not np.any(finite & (c == 0)):
        return None
    return imputed_data(x[finite], c[finite], n[finite])


def _runaway(
    fun: Callable[..., Any],
    args: tuple,
    x: npt.NDArray,
    init: npt.NDArray,
    derivatives: "tuple[npt.NDArray, npt.NDArray] | None" = None,
    floor: "float | npt.ArrayLike" = 0.0,
    keep: "Callable[[int, float], bool] | None" = None,
) -> tuple[int, ...]:
    """The positions of the parameters along which the likelihood has no
    finite maximum near ``x``, a point a rung stopped at, searched from
    ``init``: Newton's method cannot converge along their profiles
    (``runaway_coefficients``, the regression fits' check, #392), and
    ``keep`` holds (see ``_Judge.keep``). ``derivatives`` are the Hessian
    and gradient at ``x``, where the caller has them, and ``floor`` the
    parameters' least sizes for its gate (see ``_cleared``)."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Output seems independent")
        runaway = tuple(
            runaway_coefficients(
                lambda u: fun(u, *args),
                x,
                list(range(len(x))),
                init,
                derivatives,
                floor,
                keep,
            )
        )
    if runaway and not _on_the_floor(
        fun, args, x, runaway, derivatives, floor
    ):
        return ()
    return runaway


def _on_the_floor(
    fun: Callable[..., Any],
    args: tuple,
    x: npt.NDArray,
    runaway: tuple[int, ...],
    derivatives: "tuple[npt.NDArray, npt.NDArray] | None",
    floor: "float | npt.ArrayLike",
) -> bool:
    """Whether the parameters other than ``runaway`` are at their best
    values at ``x``, as they are on a search following the likelihood's
    rise along the parameters running off: the likelihood curves down in
    each of their directions (to rounding), and Newton's step to their
    best values moves none by as much as its own size.

    A search stopped against a kink or a jump in the likelihood is not
    following a rise, whatever Newton's test says along a profile there:
    a custom spline whose cumulative hazard jumps at its knot stopped
    BFGS after eight iterations with the other parameters' Hessian
    indefinite and their Newton step 12 times their sizes, read as a
    runaway, and the search ended there, 9 below the maximum the ladder's
    later rungs reached. At each runaway in the tests the step was at most
    a tenth of the sizes. Where the derivatives cannot be taken, the
    runaway stands."""
    others = [i for i in range(len(x)) if i not in runaway]
    if not others:
        return True
    if derivatives is None:
        derivatives = search_derivatives(
            lambda u: fun(u, *args), np.asarray(x, dtype=float)
        )
    if derivatives is None:
        return True
    H, g = (np.asarray(d, dtype=float) for d in derivatives)
    H_o, g_o = H[np.ix_(others, others)], g[others]
    if not (np.all(np.isfinite(H_o)) and np.all(np.isfinite(g_o))):
        return True
    size = np.maximum(np.abs(np.asarray(x, dtype=float)), floor)[others]
    # (a parameter at 0 with no floor given is measured in its unit)
    size = np.where(size > 0, size, 1.0)
    # In units of the parameters' sizes, where the Hessian is conditioned
    H_o = H_o * np.outer(size, size)
    g_o = g_o * size
    eig = np.linalg.eigvalsh(H_o)
    if eig[0] < -1e-8 * max(np.max(np.abs(eig)), 1.0):
        return False
    step = np.linalg.lstsq(H_o, g_o, rcond=None)[0]
    return bool(np.max(np.abs(step)) < 1.0)


class _OnBounds(NamedTuple):
    """The parameters of a search's point that are on a bound of their
    space (``_Judge.on_bounds``)."""

    #: Their positions in the search vector.
    held: tuple[int, ...] = ()
    #: The steepest rise of the likelihood off its bound among them, per
    #: observation and per unit of the parameter (0 where it falls off
    #: every bound).
    rise: float = 0.0
    #: The natural parameters with each one the likelihood rises off moved
    #: off its bound to the middle of its range, where its searched value
    #: moves it most (a start just inside the bound, ``p = 0.99``, is where
    #: the search's tolerance is met at once, the likelihood so flat in
    #: it): a start for another search (``optimised_fit``); ``None`` where
    #: it rises off none.
    off: "npt.NDArray | None" = None


class _Judge(NamedTuple):
    """How ``_search`` judges a rung's best point.

    A likelihood with no finite maximum keeps rising towards a supremum as
    a parameter runs off, and no rung can verify a point on the way: each
    runs until its own limit, and the ladder ran them all (an ExpoWeibull
    whose ``mu`` ran off took 23 s, every rung; #584). So after the first
    rung that stops short of a verified maximum, the point it reached is
    checked as the regression fits check theirs (``_runaway``), and a
    runaway ends the search. Otherwise the ladder goes on as before, and
    a later rung's point is checked too where it is on the way to the
    family's limit as the offset runs to -inf (``toward_limit``): the
    first rung can stop somewhere unrelated to the runaway a later one
    finds (#616).
    """

    fun: Callable[..., Any]
    jac: Callable[..., Any]
    hess_kept: Callable[..., Any]
    args: tuple
    init: npt.NDArray
    floor: Any
    obj_scale: float
    #: ``(natural, bounds, free, edge, corner)``: the map from the search
    #: vector to the full vector of natural parameters, their bounds, the
    #: position in it of each searched (free) parameter, and the positions
    #: of those at an edge where the likelihood is unbounded at a point of
    #: the search: a family's own edges, and an offset run onto the first
    #: failure (see ``_space``).
    space: tuple
    #: The runaways found while watching a search, by the point's bytes.
    found: dict
    #: The last few iterates of the search watched (``watch``).
    last: list
    #: The negative log-likelihood of the family's limit as an offset
    #: runs to -inf, fitted to the data, or ``None`` (``_offset_limit``).
    limit: Callable[[], "float | None"]
    #: The model being fitted.
    model: Any = None

    def watch(self) -> Callable[[npt.NDArray], None]:
        """A BFGS callback that checks its iterates for a runaway
        (``_runaway``) at iterations 100, 200, 400, ..., and ends the
        search on one. A search running off spends its iterations on the
        way out (1000 of them on the ExpoWeibull of #584, 98% of its fit)
        and an ordinary one converges in tens, before the first check;
        doubling the interval keeps the checks' cost below a fixed share
        of the iterations between them."""
        count = [0]

        def callback(x: npt.NDArray) -> None:
            self.last[:] = self.last[-2:] + [np.array(x, dtype=float)]
            count[0] += 1
            k, rest = divmod(count[0], _WATCH_EVERY)
            if rest or k & (k - 1):
                return
            runaway = _runaway(
                self.fun, self.args, x, self.init, keep=self.keep(x)
            )
            if runaway:
                self.found[np.asarray(x, dtype=float).tobytes()] = runaway
                raise StopIteration

        return callback

    def last_iterate(self) -> "OptimizeResult | None":
        """The last of the last few iterates of the search watched
        (``watch``) whose likelihood is finite, as a result; ``None``
        where there is none. (BFGS can step onto a point where it is not:
        an offset LogLogistic's search vector from 5e5 to -3e10, #627.)"""
        for x in self.last[::-1]:
            with np.errstate(all="ignore"):
                f = float(self.fun(x, *self.args))
            if np.all(np.isfinite(x)) and np.isfinite(f):
                return OptimizeResult(x=x, fun=f, success=False, message="")
        return None

    def keep(self, x: npt.NDArray) -> Callable[[int, float], bool]:
        """What else a parameter ``j`` that Newton's method cannot
        converge along at ``x`` must show to be running off there, with
        ``slope`` the derivative along its profile:

        - The profile is flat, to the verification's own tolerance
          (``is_local_minimum``: per observation, in the parameter's
          search unit). A search on its way to a supremum stops only where
          the rise has become too small to follow; one that stopped
          anywhere else (its line search failed against a wall where the
          likelihood is not defined, its iterations ran out on a slope)
          says nothing about where the likelihood goes. Or, for an
          offset running to -inf, the family's limit there fits the data
          at least as well as the point reached, and as any offset on
          the way (``runs_to_limit``): the rise then goes on to that
          limit, though it may never look flat on the way (#599).
        - For an offset moved down, the point fits the data no better
          than that limit (``beats_limit``), flat or not: one that fits
          better is short of the limit, and so of a run to it (#627).
        - The likelihood rises towards an infinite end of the parameter's
          range. Towards a finite bound the rise ends at the bound, a
          maximum on the edge of the space (an Exponential's offset at
          the first failure), not a runaway; the families that can have
          no maximum there check it themselves (``_warn_if_at_limit``,
          ``_warn_if_offset_at_limit``).
        """
        natural, bounds, free = self.space[:3]
        size = np.maximum(np.abs(x), np.asarray(self.floor, dtype=float))

        def keep(j: int, slope: float) -> bool:
            flat = abs(slope) * size[j] / self.obj_scale < OPTIMUM_GTOL
            if self.beats_limit(x):
                return False
            if not flat and not self.runs_to_limit(x):
                return False
            ahead = np.array(x, dtype=float)
            # (each parameter's map is monotone and its own)
            ahead[j] -= np.sign(slope) * size[j]
            with np.errstate(all="ignore"):
                now = float(natural(x)[free[j]])
                then = float(natural(ahead)[free[j]])
            low, high = bounds[free[j]]
            if then > now:
                return high is None or not np.isfinite(high)
            if then < now:
                return low is None or not np.isfinite(low)
            return False

        return keep

    def toward_limit(self, x: npt.NDArray) -> bool:
        """Whether ``x`` is on the way to the family's limit as its offset
        runs to -inf (``_offset_limit``): the search has moved the offset
        down from where it started, and the limit fits the data at least
        as well as ``x`` does. ``keep`` checks that the parameter running
        off rises towards an infinite end: the offset itself, or the
        shape that makes up for it.

        An offset LogNormal or Gamma fitted to data with a long left tail
        (an observation at -1 below the rest at 9 to 22) runs its offset
        down towards their limit, the Normal (the Gamma's shape up with
        it, the LogNormal's sigma down to 0); but the likelihood
        approaches the Normal's only as ``1 / |gamma|``, so a search
        stopped at gamma = -2765 was still 7 times the verification's
        tolerance from flat, every rung of the ladder ran, and the fits
        ended "unverified" after 4-17 s (#599).

        "At least as well" is to a relative tolerance (``_LIMIT_RTOL``):
        far out the family's likelihood is computed to rounding, and the
        limit's own fit is a maximum only to its verification's
        tolerance, so a point on the way can come out a hair better than
        the limit (a Weibull 1.8e-7 below the Gumbel's, #627)."""
        gap = self._limit_gap(x)
        return gap is not None and gap[0] >= -gap[1]

    def beats_limit(self, x: npt.NDArray) -> bool:
        """Whether ``x`` has its offset moved down from the start and
        fits the data better than the family's limit (beyond
        ``toward_limit``'s tolerance): the search is then not running off
        towards the limit, whose likelihood the family only approaches,
        however flat the profile looks there. An offset LogNormal stopped
        at gamma = -141, 0.012 better than the Normal, was called a
        runaway though its maximum is near -150 (#627)."""
        gap = self._limit_gap(x)
        return gap is not None and gap[0] < -gap[1]

    def runs_to_limit(self, x: npt.NDArray) -> bool:
        """Whether the search reaching ``x`` runs off towards the family's
        limit: ``toward_limit``, and no offset on its way there, between
        its start and ``x``, fits the data better than the limit
        (``interior_beats_limit``). The likelihood can have a very flat
        maximum just below the limit's, past which the search ran (a
        LogNormal's 0.003 better than the Normal's, #627); this is the
        test before calling a runaway by the limit."""
        return self.toward_limit(x) and not self.interior_beats_limit(x)

    def _limit_gap(self, x: npt.NDArray) -> "tuple[float, float] | None":
        """``(here - limit, tolerance)``: how much worse ``x`` fits the
        data than the family's limit as the offset runs to -inf, and the
        tolerance of the comparison; ``None`` for a fit without an offset
        (or one whose limit is not known), and where the search has not
        moved the offset down from its start."""
        if not self.args[0]:
            return None
        natural = self.space[0]
        with np.errstate(all="ignore"):
            moved_down = float(natural(x)[0]) < float(natural(self.init)[0])
        if not moved_down:
            return None
        limit = self.limit()
        if limit is None:
            return None
        with np.errstate(all="ignore"):
            here = float(self.fun(x, *self.args))
        if not np.isfinite(here):
            return None
        return here - limit, _LIMIT_RTOL * max(abs(limit), 1.0)

    def interior_beats_limit(self, x: npt.NDArray) -> bool:
        """Whether an offset between the search's start and ``x``'s fits
        the data better than the family's limit (beyond ``toward_limit``'s
        tolerance), its other parameters at their best for it: a profile
        of the likelihood over the offset, at points spaced geometrically
        in their distance below the data (``_profile_offsets``). Where
        one does, the likelihood has a maximum on the way, short of the
        limit, and the search ran past it.

        The search ran its offset down from the start, so a maximum it
        passed is between the two; the profile is not read nearer the
        data than the start, where the likelihood rises towards the
        first failure without bound for a density infinite at its origin
        (``corner``), a different end of the search."""
        limit = self.limit()
        if limit is None:
            return False
        tol = _LIMIT_RTOL * max(abs(limit), 1.0)
        for value in self._offset_profile(x):
            if value < limit - tol:
                return True
        return False

    def _offset_profile(self, x: npt.NDArray) -> Any:
        """The profile negative log-likelihood at the offsets
        ``_profile_offsets`` gives between the start and ``x``, one at a
        time: at each, the distribution's own parameters searched with
        the offset held, from the distribution's start for the data
        shifted by it (``_offset_seed``). A point whose search fails is
        skipped."""
        natural, _, free = self.space[:3]
        model = self.model
        if model is None or not free or free[0] != 0:
            return
        with np.errstate(all="ignore"):
            values = np.array(natural(x), dtype=float)
            start = float(natural(self.init)[0])
        high = model.bounds[0][1]
        if high is None or not np.isfinite(high):
            return
        seed_data = _seed_data(model.surv_data)
        if seed_data is None:
            return
        transform = model.fitting_info["transform"]
        k = len(model.dist.parameter_names)
        for gamma in _profile_offsets(float(high), start, float(values[0])):
            point = values.copy()
            try:
                with np.errstate(all="ignore"):
                    seed = model.dist._offset_seed(seed_data, gamma)
                    point[: 1 + k] = seed
                    u = np.asarray(transform(point), dtype=float)[free]
            except (ValueError, ArithmeticError, TypeError, IndexError):
                continue
            if not np.all(np.isfinite(u)):
                continue
            value = self._held_offset_minimum(u)
            if value is not None:
                yield value

    def _held_offset_minimum(self, u: npt.NDArray) -> "float | None":
        """The least negative log-likelihood a BFGS search over the search
        vector's other parameters finds from ``u``, its offset (first)
        held; ``None`` where it is not finite."""
        head = u[:1]
        fun, args = self.fun, self.args

        def held(v: Any) -> Any:
            return fun(np.concatenate([head, v]), *args)

        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                f0 = float(held(u[1:]))
                res = minimize_with_gradient(
                    held,
                    u[1:],
                    jac=Gradient(held),
                    method="BFGS",
                    options={"maxiter": _PROFILE_ITERATIONS},
                )
                f = min(f0, float(res.fun)) if np.isfinite(res.fun) else f0
            except (ValueError, ArithmeticError, TypeError):
                return None
        return f if np.isfinite(f) else None

    def corner(self, x: npt.NDArray) -> "OptimizeResult | None":
        """Where a search that stopped short of a verified maximum at
        ``x`` is going, if that is onto the first failure with the offset
        (#622): the result to end the search with, else ``None``.

        That is ``x`` itself where its offset is already there and the
        density at its origin is not finite and positive
        (``_offset_corner``): the likelihood is unbounded there. And it is
        the point on the first failure with the other parameters as at
        ``x`` (within half the distance ``_offset_corner`` allows), where
        the search has moved the offset up from its start with a density
        infinite at its origin (a Weibull, Gamma or LogLogistic shape
        below 1) and the likelihood there is higher than at ``x``: the
        search was stopped on its way there by the steepening rise (BFGS's
        line search fails on it), and with such a density the likelihood
        of data without truncation keeps rising as the offset moves up
        however the other parameters are set (each density, survival and
        interval term rises as its point moves towards the origin), so
        the corner is where the search goes and the rest of the ladder
        took it there in 5,000 to 15,000 evaluations. (An interior maximum,
        where there is one, has a shape above 1, since below 1 the
        likelihood rises with the offset everywhere.)"""
        natural, _, free = self.space[:3]
        if not self.args[0] or 0 not in free:
            return None
        model = self.model
        if self.space[4](x):
            with np.errstate(all="ignore"):
                f = float(self.fun(x, *self.args))
            return OptimizeResult(x=x, fun=f, success=False, message="")
        gap = model.dist._first_failure_gap(model.surv_data)
        if gap is None:
            return None
        x1, close = gap
        with np.errstate(all="ignore"):
            values = np.array(natural(x), dtype=float)
            start = float(natural(self.init)[0])
            core = values[1 : 1 + len(model.dist.parameter_names)]
            f0 = float(np.asarray(model.dist.df(np.array([0.0]), *core))[0])
        if not (np.isposinf(f0) and start < values[0] < x1):
            return None
        values[0] = x1 - 0.5 * close
        u = np.asarray(model.fitting_info["transform"](values), dtype=float)
        u = u[free]
        with np.errstate(all="ignore"):
            f_x = float(self.fun(x, *self.args))
            f_u = float(self.fun(u, *self.args))
        if not (np.all(np.isfinite(u)) and f_u < f_x):
            return None
        return OptimizeResult(x=u, fun=f_u, success=False, message="")

    def on_bounds(self, x: npt.NDArray) -> _OnBounds:
        """The parameters at ``x`` on a bound of a range bounded at both
        ends (a limited-failure ``p`` of 1, a zero-inflation ``f0`` of 0),
        and whether the likelihood rises off it.

        Such a parameter is searched as a scaled arctanh, whose bounds are
        at infinity: on its way to a bound the parameter reaches it in
        floating point, the likelihood stops depending on its searched
        value, and its gradient and curvature there are zero or rounding.
        So a zero gradient says nothing about it, and the Hessian's
        positive rounding passed the verification: a Weibull with
        ``lfp=True`` on monthly return counts ran ``p`` to 1, where the
        likelihood rises as ``p`` moves off it, and reported a verified
        maximum 3.8 below the one at ``p = 0.059`` (#579). A parameter is
        on its bound where the likelihood is the same, to rounding, a
        millionth of the way closer to it (as ``verified_maximum`` tests
        it), and the likelihood rises off the bound where it is higher
        (beyond rounding) a millionth of the range into it, the other
        parameters as they are."""
        natural, bounds, free = self.space[:3]
        offset, lfp, zi = self.args[:3]
        if not any(None not in bounds[i] for i in free):
            # No parameter has a range bounded at both ends
            return _OnBounds()

        def at(values: npt.NDArray) -> float:
            # The likelihood of the natural parameters
            return float(self.fun(values, offset, lfp, zi, False))

        with np.errstate(all="ignore"):
            values = natural(x)
            f = at(values)
        if not np.isfinite(f):
            return _OnBounds()
        level = 1e-12 * max(abs(f), 1.0)
        held, rise, off = [], 0.0, None
        for k, i in enumerate(free):
            low, high = bounds[i]
            if low is None or high is None:
                continue
            width = float(high) - float(low)
            for bound, inward in ((low, 1.0), (high, -1.0)):
                toward, away = values.copy(), values.copy()
                toward[i] = bound + (values[i] - bound) * 1e-6
                away[i] = bound + inward * 1e-6 * width
                with np.errstate(all="ignore"):
                    f_toward, f_away = at(toward), at(away)
                if not abs(f_toward - f) <= level:
                    continue
                held.append(k)
                if f_away < f - level:
                    slope = (f - f_away) / (1e-6 * width) / self.obj_scale
                    rise = max(rise, slope)
                    off = np.array(values if off is None else off)
                    off[i] = bound + inward * 0.5 * width
                break
        return _OnBounds(tuple(held), rise, off)

    def _verified_without(self, x: npt.NDArray, held: tuple) -> bool:
        """Whether ``x`` is a verified maximum in its components other
        than ``held`` (``is_local_minimum`` on them)."""
        keep = [k for k in range(len(x)) if k not in held]
        if not keep:
            return True
        with np.errstate(all="ignore"):
            g = np.asarray(self.jac(x, *self.args), dtype=float)[keep]
            H = np.atleast_2d(
                np.asarray(self.hess_kept(x, *self.args), dtype=float)
            )[np.ix_(keep, keep)]
        floor = np.broadcast_to(
            np.asarray(self.floor, dtype=float), np.shape(x)
        )
        return is_local_minimum(
            lambda _: 0.0,  # (only the derivatives are read)
            lambda _: g,
            lambda _: H,
            np.asarray(x, dtype=float)[keep],
            floor=floor[keep],
            obj_scale=self.obj_scale,
        )

    def verdict(self, x: npt.NDArray, check: bool) -> tuple[bool, tuple]:
        """``(verified, runaway)`` at ``x``, a rung's best point: whether
        it is a verified maximum (``is_local_minimum``) and, if not and
        ``check``, the parameters running off there (``_runaway``).

        A verified point is checked too, through the gate alone where it
        is a maximum (the Hessian it was verified with, and its Newton
        step against the parameters' sizes): a likelihood that flattens
        towards a supremum can pass the verification far out on the way
        to it (a Normal at mu = -2.9e8, #594). Then it is a runaway, not
        a maximum.

        A parameter on a bound of its space (``on_bounds``) is held out
        of the test, and it is a maximum there where the likelihood does
        not rise off the bound by more than the verification's tolerance
        (``OPTIMUM_GTOL`` per observation, as ``at_boundary_maximum``).
        Where it rises at all, the fit also searches from off the bound
        (``optimised_fit``) and keeps the better answer."""
        fun, args = self.fun, self.args
        keep = self.keep(x)
        # The gradient the verification takes, kept for the check
        jac_kept, _ = _kept_hessian(self.jac)
        on = self.on_bounds(x)
        if on.held:
            if on.rise < OPTIMUM_GTOL and self._verified_without(x, on.held):
                return True, ()
        elif is_local_minimum(
            fun,
            jac_kept,
            self.hess_kept,
            x,
            args,
            floor=self.floor,
            obj_scale=self.obj_scale,
        ):
            with np.errstate(all="ignore"):
                H = np.asarray(self.hess_kept(x, *args), dtype=float)
                g = np.asarray(jac_kept(x, *args), dtype=float)
            runaway = _runaway(
                fun, args, x, self.init, (H, g), self.floor, keep
            )
            return not runaway, runaway
        if not check:
            return False, ()
        seen = self.found.get(np.asarray(x, dtype=float).tobytes())
        if seen:
            return False, seen
        # A search that ran onto an edge where the family's likelihood is
        # unbounded (``_at_unbounded_edge``) cannot be verified either.
        at_edge = self.space[3](x)
        if at_edge:
            return False, at_edge
        return False, _runaway(fun, args, x, self.init, keep=keep)


def _space(model: "Parametric") -> tuple:
    """``_Judge.space`` for ``model``'s fit."""
    const = model.fitting_info["const"]
    inv_trans = model.fitting_info["inv_trans"]
    fixed_idx = model.fitting_info["fixed_idx"]

    def natural(u: npt.NDArray) -> npt.NDArray:
        return np.asarray(inv_trans(const(u)), dtype=float)

    free = [i for i in range(len(model.bounds)) if i not in fixed_idx]
    names = sorted(model.param_map, key=model.param_map.__getitem__)

    def edge(u: npt.NDArray) -> tuple[int, ...]:
        with np.errstate(all="ignore"):
            values = dict(zip(names, natural(u)))
        at = model.dist._at_unbounded_edge(model.surv_data, values)
        return tuple(k for k, i in enumerate(free) if names[i] in at)

    def corner(u: npt.NDArray) -> tuple[int, ...]:
        # The offset's position in the search vector, where it has run
        # onto the first failure (``_offset_corner``)
        if not model.offset or 0 not in free:
            return ()
        with np.errstate(all="ignore"):
            values = natural(u)
        core = values[1 : 1 + len(model.dist.parameter_names)]
        at = model.dist._offset_corner(model.surv_data, values[0], core)
        return (0,) if at is not None else ()

    return natural, model.bounds, free, edge, corner


def _offset_limit(model: "Parametric") -> Callable[[], "float | None"]:
    """The negative log-likelihood, on the model's data, of the family its
    distribution tends to as an offset runs to -inf
    (``_offset_limit_family``: the Normal for a LogNormal or a Gamma),
    fitted by maximum likelihood when first asked for; ``None`` for a fit
    without an offset (or with a limited failure population or zero
    inflation, which the limit has not), for a family with no such limit,
    and where the limit's own fit is not a verified maximum."""
    kept: list = []

    def neg_ll() -> "float | None":
        if not kept:
            kept.append(None)
            family = getattr(model.dist, "_offset_limit_family", None)
            family = family() if family is not None else None
            if not model.offset or model.lfp or model.zi or family is None:
                return None
            from surpyval.utils.no_maximum import quiet_maximum_warnings

            with warnings.catch_warnings(), quiet_maximum_warnings():
                warnings.simplefilter("ignore")
                try:
                    limit = family.fit_from_surpyval_data(model.surv_data)
                except (ValueError, ArithmeticError):
                    return None
            if limit.maximum == "verified":
                kept[0] = float(limit._neg_ll)
        return kept[0]

    return neg_ll


def _at_corner(res: Any, judge: _Judge) -> bool:
    """Whether a rung's result stopped with its offset run onto the first
    failure (``_offset_corner``), its likelihood finite or, where a density
    infinite at its origin is evaluated there, infinite (a negative
    log-likelihood of ``-inf``)."""
    if not np.all(np.isfinite(res.x)):
        return False
    if not (np.isfinite(res.fun) or np.isneginf(res.fun)):
        return False
    return bool(judge.space[4](res.x))


def _search(
    model: "Parametric",
    fun: Callable[..., Any],
    derivatives: tuple[Callable[..., Any], Callable[..., Any]],
    hess_kept: Callable[..., Any],
    init: npt.NDArray,
    args: tuple,
) -> _Search:
    """Run the optimiser ladder, stopping at the first verified rung.

    ``derivatives`` is ``(jac, hess)`` of ``fun``.

    Gradient methods go first and the derivative-free pair (Nelder-Mead,
    Powell) is the fallback for when they do not converge. Running every
    rung on every fit would mostly confirm what an earlier rung had
    already found: over 102 fits across eleven distributions the whole
    ladder agreed on the objective to 1e-10. Nelder-Mead and Powell pay
    for their robustness in function evaluations -- 50 and 22 of them
    against BFGS's 21, each O(n) -- which on a million observations is
    42% of the fit. The order and the early exit go together: stopping
    early with the derivative-free methods first would halt at
    Nelder-Mead, the most expensive rung and the one with the worst
    objective. The derivative-free methods start from the cold initial
    guess when they are reached, so the fits that need a second start
    still get one.

    "Converges" means the answer is verifiably a maximum -- its gradient
    ~0 and its Hessian positive definite (see ``is_local_minimum``) --
    not that the optimiser reported success: from a start far from the
    maximum a gradient method reports success where the likelihood first
    looks flat (a Weibull started at alpha = 1e7 stopped at beta = 0.099,
    40 below the maximum; #427). So a rung stops the ladder only when the
    best point so far is verified, which costs one gradient and one
    Hessian; a fit the first rung solves stops there. If no rung is
    verified, the answer is the first rung that reported success, or
    failing that the best point found; ``verified`` is then False and the
    caller tries other starts and, failing those, warns.

    A likelihood with no finite maximum ends the search where it is found
    (see ``_Judge``): at the first rung that stops short of a verified
    maximum, inside BFGS (``_Judge.watch``), at a later rung's point on
    the way to an offset family's limit, or at a verified answer that is
    really on the way to a supremum. The answer is then the point
    checked, and ``runaway`` names the parameters running off. An offset
    fit whose rung ends unverified at a best point no better than the
    family's limit, with no offset on the way there better either
    (``_Judge.runs_to_limit``), has its offset running off too, and the
    search ends there (``by_limit``, #616, #627); so does a first BFGS
    search that overshot to a point where the likelihood is not finite,
    whose last finite iterate is such a point. And a rung that runs the
    offset onto the first failure, where the likelihood is unbounded, ends
    the search there, as does the first rung stopping on its way there
    (``_Judge.corner``, #622); the offset is then the parameter named.
    """
    if len(init) == 0:
        # Every parameter is fixed; there is nothing to optimise, and
        # the answer is exact
        res = OptimizeResult(
            x=init,
            success=True,
            fun=fun(init, *args),
            message="",
        )
        return _Search(res, "all parameters fixed", True)

    jac, hess = derivatives
    by_name = {None: None, "jac": jac, "hess": hess}
    floor = search_floor(model)
    obj_scale = float(np.sum(model.data["n"]))
    best = np.inf
    best_result = None
    best_method = None
    verified = False
    first_success = None
    runaway: tuple[int, ...] = ()
    by_limit = False
    checked = False
    judge = _Judge(
        fun,
        jac,
        hess_kept,
        args,
        init,
        floor,
        obj_scale,
        _space(model),
        {},
        [],
        _offset_limit(model),
        model,
    )
    for method, jac_name, hess_name in _LADDER:
        jac_i, hess_i = by_name[jac_name], by_name[hess_name]
        for x0 in _rung_starts(method, init, first_success):
            res = _run_rung(
                fun,
                method,
                x0,
                args,
                jac_i,
                hess_i,
                floor,
                obj_scale,
                judge.watch() if not checked else None,
            )
            # A search that ran onto the first failure with the offset,
            # where the likelihood is unbounded (#622; see
            # ``_Judge.corner``), ends the ladder: no rung can verify a
            # maximum there, and the rest followed it into the corner in
            # thousands of evaluations
            if _at_corner(res, judge):
                best_result, best_method = res, method
                runaway = (0,)
                break
            if not _usable(res):
                continue
            if res.success and first_success is None:
                first_success = (res, method)
            if res.fun < best:
                best_result, best_method, best = res, method, res.fun
        if runaway:
            break
        if best_result is None and not checked:
            # The first search ended where the likelihood is not finite
            # (BFGS overshot by 1e10 on its way to the family's limit):
            # its last iterate tells whether it was running off there,
            # rather than the next rung's 1,000 evaluations (#627)
            last = judge.last_iterate()
            if last is not None and judge.runs_to_limit(last.x):
                best_result, best_method = last, method
                runaway, by_limit = (0,), True
                break
        if best_result is None:
            continue
        # After the first rung that stops short of a verified maximum: is
        # the likelihood running off? Then no rung can verify it. A later
        # rung's point is checked where it is on the way to the family's
        # limit (#616).
        verified, runaway = judge.verdict(
            best_result.x, not checked or judge.toward_limit(best_result.x)
        )
        first, checked = not checked, True
        if verified or runaway:
            break
        # Or onto the first failure with the offset (#622)
        corner = judge.corner(best_result.x) if first else None
        if corner is not None:
            best_result, runaway = corner, (0,)
            break
        # Or towards the family's limit as the offset runs to -inf: the
        # best point reached is no better than the limit, nor is any
        # offset on the way there (``_Judge.runs_to_limit``). No member
        # the search found fits the data better than the limit, which the
        # family only approaches, though Newton's test could not show the
        # rise there (its derivatives are rounding far out, #616). The
        # offset is the parameter that runs off. This was only asked at
        # the end of the ladder, whose rest took 1,000 to 3,000
        # evaluations to reach the same verdict (#627).
        if judge.runs_to_limit(best_result.x):
            runaway, by_limit = (0,), True
            break

    if not (verified or runaway) and first_success is not None:
        best_result, best_method = first_success
    off_bound = None
    if best_result is not None:
        res = best_result
        # A verified answer stands whatever its rung reported: BFGS
        # often stops with "precision loss" at the maximum.
        res.success = res.success or verified
        if not runaway:
            off_bound = judge.on_bounds(res.x).off
    return _Search(
        res,
        best_method if best_method is not None else method,
        verified,
        runaway,
        off_bound,
        by_limit,
    )


def _unverified_outcome(search: _Search) -> tuple[Any, Any, bool]:
    """``(warning, unverified_reason, use_initial)`` for the answer.

    The warning is the caller's to give (``results["_warning"]``, or
    ``warn_unverified`` with ``results["_unverified_reason"]``): it may
    try other starts, and only the answer it keeps speaks.
    """
    res = search.res
    if search.verified or search.runaway:
        # (A runaway is said by the caller: "No finite maximum")
        return None, None, False
    if "Desired error not necessarily" in res.get("message", ""):
        return (
            None,
            (
                "the optimiser stopped on a loss of precision; data "
                "rescaled closer to 1 may help"
            ),
            False,
        )
    if (not res.success) or (np.isnan(res.x).any()):
        return _MLE_FAILED, None, True
    return None, None, False


def _split_parameters(
    params: Any, offset: bool, zi: bool, lfp: bool
) -> tuple[Any, Any, Any, Any]:
    """``(gamma, f0, p, params)``: the extra parameters split off the ends."""
    if offset:
        gamma = params[0]
        params = params[1:]
    else:
        gamma = 0.0

    if zi:
        f0 = params[-1]
        params = params[0:-1]
    else:
        f0 = 0.0

    if lfp:
        p = params[-1]
        params = params[0:-1]
    else:
        p = 1.0
    return gamma, f0, p, params


def _covariance(
    model: "Parametric",
    u_full: npt.NDArray,
    n_core: int,
    extras: tuple[Any, Any, Any],
    flags: tuple[bool, bool, bool],
    hess_at: dict,
) -> tuple[Any, Any]:
    """``(cov_matrix, hess_inv)`` of the fitted parameters.

    The covariance of the parameters is found from the Hessian in the
    transformed (unbounded) space used during optimisation, then mapped
    back to the bounded parameter space with the delta method. p and f0
    are estimated parameters and are included in the covariance.
    User-fixed parameters are known, not estimated, so they carry no
    variance and the free parameters get their conditional variance.
    gamma is also held at its estimate since the threshold parameter of
    an offset model is non-regular and a Wald variance for it would be
    misleading. ``extras`` is ``(gamma, f0, p)`` and ``flags`` is
    ``(offset, zi, lfp)``.
    """
    from numdifftools import Hessian  # type: ignore

    gamma, f0, p = extras
    offset, zi, lfp = flags
    inv_trans = model.fitting_info["inv_trans"]
    fixed_idx = model.fitting_info["fixed_idx"]
    n_head = 1 if offset else 0
    n_total = len(u_full)
    var_idx = np.array(
        [i for i in range(n_head, n_total) if i not in fixed_idx],
        dtype=int,
    )

    # Embed the variance-carrying sub-vector into the full
    # transformed vector; the matrix form keeps the held entries
    # constant under autograd
    embed = np.zeros((n_total, len(var_idx)))
    embed[var_idx, np.arange(len(var_idx))] = 1.0
    u_held = np.where(embed.sum(axis=1) == 0, u_full, 0.0)

    def transformed_fun(u: npt.NDArray) -> Any:
        theta = inv_trans(embed @ u + u_held)[n_head:]
        if zi:
            *theta, f0_i = theta
        else:
            f0_i = f0
        if lfp:
            *theta, p_i = theta
        else:
            p_i = p
        return model.dist._neg_ll_func(
            model.surv_data, *theta, gamma, f0_i, p_i
        )

    def u_to_phi(u: npt.NDArray) -> Any:
        return inv_trans(embed @ u + u_held)[n_head:]

    try:
        if len(var_idx) == 0:
            cov_matrix = np.zeros((n_total - n_head, n_total - n_head))
        else:
            u_var = u_full[var_idx]
            kept = hess_at.get(np.asarray(u_var, dtype=float).tobytes())
            if n_head == 0 and len(var_idx) == n_total and kept is not None:
                hess_u = kept
            else:
                hess_u = hessian(transformed_fun)(u_var)
            # A corrupted autograd Hessian (e.g. a primitive whose
            # VJP silently drops second-order terms) shows up as
            # asymmetry; recompute numerically rather than invert
            # garbage (#270).
            asym = np.max(np.abs(hess_u - hess_u.T)) > 1e-4 * max(
                np.max(np.abs(hess_u)), 1.0
            )
            if np.isnan(hess_u).any() or asym:
                hess_u = Hessian(transformed_fun)(u_var)
            cov_u = inv(hess_u)
            if np.isnan(cov_u).any():
                cov_u = inv(Hessian(transformed_fun)(u_var))
            jac_u = jacobian(u_to_phi)(u_var)
            # Covariance of the extended vector (*params, p?, f0?);
            # fixed parameters have zero rows and columns
            cov_matrix = jac_u @ cov_u @ jac_u.T
        hess_inv = cov_matrix[:n_core, :n_core]
    except np.linalg.LinAlgError:
        cov_matrix = None
        hess_inv = None
    return cov_matrix, hess_inv


def _runaway_names(model: "Parametric", runaway: tuple[int, ...]) -> list:
    """The names of the parameters at ``runaway``, positions in the search
    vector (the free parameters: ``gamma``, the distribution's, ``p``,
    ``f0``)."""
    if not runaway:
        return []
    names = sorted(model.param_map, key=model.param_map.__getitem__)
    fixed_idx = model.fitting_info["fixed_idx"]
    free = [name for i, name in enumerate(names) if i not in fixed_idx]
    return [free[k] for k in runaway]


def mle(model: "Parametric") -> Any:
    """
    Maximum Likelihood Estimation (MLE)

    """
    const = model.fitting_info["const"]
    inv_trans = model.fitting_info["inv_trans"]
    # Initial guess
    init = model.fitting_info["init"]
    # Offset, Limited Failure Population, Zero Inflated logic.
    offset, lfp, zi = model.offset, model.lfp, model.zi

    results = {}

    fun = _negative_log_likelihood(model)
    # The value and the gradient from one pass (#593)
    jac = Gradient(fun)
    hess = hessian(fun)
    hess_kept, hess_at = _kept_hessian(hess)
    args = (offset, lfp, zi, True)

    with np.errstate(all="ignore"):
        search = _search(model, fun, (jac, hess), hess_kept, init, args)
        res = search.res
        warning, unverified_reason, use_initial = _unverified_outcome(search)

        u_full = const(init) if use_initial else const(res.x)
        gamma, f0, p, params = _split_parameters(
            inv_trans(u_full), offset, zi, lfp
        )
        results["gamma"] = gamma
        results["f0"] = f0
        results["lfp_p"] = p
        results["params"] = params

        cov_matrix, hess_inv = _covariance(
            model,
            u_full,
            len(params),
            (gamma, f0, p),
            (offset, zi, lfp),
            hess_at,
        )
        results["_covariance"] = cov_matrix
        results["hess_inv"] = hess_inv
        # On the fallback path the returned parameters are the initial
        # guess, so the reported likelihood must be evaluated there — not
        # taken from the failed optimizer (#261).
        if use_initial:
            with np.errstate(all="ignore"):
                neg_ll_val = float(fun(init, offset, lfp, zi, True))
        else:
            neg_ll_val = float(res["fun"])
        results["_neg_ll"] = neg_ll_val
        results["log_likelihood"] = -neg_ll_val
        results["res"] = res
        results["_verified"] = bool(search.verified) and not use_initial
        results["_warning"] = warning
        results["_unverified_reason"] = unverified_reason
        results["_runaway"] = _runaway_names(model, search.runaway)
        results["_runaway_by_limit"] = search.by_limit
        results["_off_bound"] = search.off_bound
        results["optimizer"] = search.optimizer

    return results
