import warnings
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

if TYPE_CHECKING:
    from ..parametric import Parametric

import autograd.numpy as np
import numpy.typing as npt
from autograd import hessian, jacobian
from autograd.numpy.linalg import inv
from numdifftools import Hessian  # type: ignore
from scipy.optimize import OptimizeResult, minimize

from surpyval.univariate.parametric.fitters import (
    OPTIMUM_GTOL,
    _usable,
    is_local_minimum,
    preconditioned_bfgs,
    search_floor,
)
from surpyval.univariate.parametric.fitters.runaway import (
    runaway_coefficients,
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
    """``hess`` that keeps its last value, and the dict it keeps it in.

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
    return minimize(
        fun,
        x0,
        args=args,
        method=method,
        jac=jac_i,
        hess=hess_i,
        options=opts,
    )


#: How many BFGS iterations pass before ``_Judge.watch`` first checks one.
_WATCH_EVERY = 100


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
        return tuple(
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


class _Judge(NamedTuple):
    """How ``_search`` judges a rung's best point.

    A likelihood with no finite maximum keeps rising towards a supremum as
    a parameter runs off, and no rung can verify a point on the way: each
    runs until its own limit, and the ladder ran them all (an ExpoWeibull
    whose ``mu`` ran off took 23 s, every rung; #584). So after the first
    rung that stops short of a verified maximum, the point it reached is
    checked as the regression fits check theirs (``_runaway``), and a
    runaway ends the search. Otherwise the ladder goes on as before.
    """

    fun: Callable[..., Any]
    jac: Callable[..., Any]
    hess_kept: Callable[..., Any]
    args: tuple
    init: npt.NDArray
    floor: Any
    obj_scale: float
    #: ``(natural, bounds, free)``: the map from the search vector to
    #: the full vector of natural parameters, their bounds, and the
    #: position in it of each searched (free) parameter.
    space: tuple
    #: The runaways found while watching a search, by the point's bytes.
    found: dict

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
          says nothing about where the likelihood goes.
        - The likelihood rises towards an infinite end of the parameter's
          range. Towards a finite bound the rise ends at the bound, a
          maximum on the edge of the space (an Exponential's offset at
          the first failure), not a runaway; the families that can have
          no maximum there check it themselves (``_warn_if_at_limit``,
          ``_warn_if_offset_at_limit``).
        """
        natural, bounds, free = self.space
        size = np.maximum(np.abs(x), np.asarray(self.floor, dtype=float))

        def keep(j: int, slope: float) -> bool:
            if not abs(slope) * size[j] / self.obj_scale < OPTIMUM_GTOL:
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

    def verdict(self, x: npt.NDArray, check: bool) -> tuple[bool, tuple]:
        """``(verified, runaway)`` at ``x``, a rung's best point: whether
        it is a verified maximum (``is_local_minimum``) and, if not and
        ``check``, the parameters running off there (``_runaway``).

        A verified point is checked too, through the gate alone where it
        is a maximum (the Hessian it was verified with, and its Newton
        step against the parameters' sizes): a likelihood that flattens
        towards a supremum can pass the verification far out on the way
        to it (a Normal at mu = -2.9e8, #594). Then it is a runaway, not
        a maximum."""
        fun, args = self.fun, self.args
        keep = self.keep(x)
        if is_local_minimum(
            fun,
            self.jac,
            self.hess_kept,
            x,
            args,
            floor=self.floor,
            obj_scale=self.obj_scale,
        ):
            with np.errstate(all="ignore"):
                H = np.asarray(self.hess_kept(x, *args), dtype=float)
                g = np.asarray(self.jac(x, *args), dtype=float)
            runaway = _runaway(
                fun, args, x, self.init, (H, g), self.floor, keep
            )
            return not runaway, runaway
        if not check:
            return False, ()
        seen = self.found.get(np.asarray(x, dtype=float).tobytes())
        if seen:
            return False, seen
        return False, _runaway(fun, args, x, self.init, keep=keep)


def _space(model: "Parametric") -> tuple:
    """``_Judge.space`` for ``model``'s fit."""
    const = model.fitting_info["const"]
    inv_trans = model.fitting_info["inv_trans"]
    fixed_idx = model.fitting_info["fixed_idx"]

    def natural(u: npt.NDArray) -> npt.NDArray:
        return np.asarray(inv_trans(const(u)), dtype=float)

    free = [i for i in range(len(model.bounds)) if i not in fixed_idx]
    return natural, model.bounds, free


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
    checked = False
    judge = _Judge(
        fun, jac, hess_kept, args, init, floor, obj_scale, _space(model), {}
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
            if not _usable(res):
                continue
            if res.success and first_success is None:
                first_success = (res, method)
            if res.fun < best:
                best_result, best_method, best = res, method, res.fun
        if best_result is None:
            continue
        # After the first rung that stops short of a verified maximum: is
        # the likelihood running off? Then no rung can verify it.
        verified, runaway = judge.verdict(best_result.x, not checked)
        checked = True
        if verified or runaway:
            break

    if not (verified or runaway) and first_success is not None:
        best_result, best_method = first_success
    if best_result is not None:
        res = best_result
        # A verified answer stands whatever its rung reported: BFGS
        # often stops with "precision loss" at the maximum.
        res.success = res.success or verified
    return _Search(
        res,
        best_method if best_method is not None else method,
        verified,
        runaway,
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
    jac = jacobian(fun)
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
        results["p"] = p
        results["params"] = params

        cov_matrix, hess_inv = _covariance(
            model,
            u_full,
            len(params),
            (gamma, f0, p),
            (offset, zi, lfp),
            hess_at,
        )
        results["cov_matrix"] = cov_matrix
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
        results["optimizer"] = search.optimizer

    return results
