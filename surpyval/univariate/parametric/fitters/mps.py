import warnings
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..parametric import Parametric

import autograd.numpy as np
import numpy.typing as npt
from autograd import hessian

from surpyval.utils.no_maximum import warn_no_maximum

from . import (
    OPTIMUM_GTOL,
    Gradient,
    fallback_minimize,
    is_local_minimum,
    search_floor,
)


def _shifted_window(dist: Any, tl: Any, tr: Any, gamma: Any) -> tuple:
    """The truncation window ``(tl, tr)`` of an offset fit, on the scale
    of ``x - gamma``.

    The truncation bounds live on the same (observed) timescale as x, so
    they must be shifted with it; leaving them unshifted evaluated F(tl)
    on the unshifted distribution and made the objective infinite at the
    true parameters (#268). Infinite bounds are unaffected by the shift.
    A shifted left bound at or below the base distribution's support
    carries F = 0 exactly -- evaluate it as -inf rather than feeding a
    negative time to the base CDF (NaN for e.g. Weibull, which fails the
    whole fit)."""
    s0 = dist.support[0]
    tl = tl - gamma
    tr = tr - gamma
    if np.isfinite(tl) and tl <= s0:
        tl = -np.inf
    if np.isfinite(tr) and tr <= s0:
        # The whole window sits below the support: F(tr) = 0 makes the
        # spacings denominator collapse, which neg_mean_D reports as an
        # infinite objective.
        tr = s0
    return tl, tr


def offset_start(
    dist: Any, data: Any, tl: float, tr: float, init: npt.NDArray
) -> npt.NDArray:
    """A start for an offset MPS fit, ``(gamma, *params)``, at which the
    objective is finite: ``init`` where it is, else the distribution's
    own initialiser on the data shifted by ``init``'s ``gamma``.

    A distribution's offset start seeds its shape and scale from a
    probability-plot fit with its own offset (the Weibull's) and then has
    that offset replaced by one just below the data (``_offset_start``):
    on data with a value far below the rest (-1 below 9 to 22) the seeds
    were a scale of 8.5e4 and a shape of 1.5e4, under which every spacing
    is 0 at the replaced offset. The objective was infinite at the start,
    every rung failed there, and the fit returned it (#616)."""
    init = np.asarray(init, dtype=float)
    gamma, params = init[0], init[1:]
    x, c, n = data.x, data.c, data.n
    with np.errstate(all="ignore"):
        window = _shifted_window(dist, tl, tr, gamma)
        if np.isfinite(dist.neg_mean_D(x - gamma, c, n, *window, *params)):
            return init
        from surpyval.utils.surpyval_data import SurpyvalData

        shifted = SurpyvalData(x - gamma, c, n, group_and_sort=False)
        try:
            base = np.atleast_1d(dist._parameter_initialiser(shifted))
        except (ValueError, ArithmeticError):
            return init
    start = np.concatenate([[gamma], np.asarray(base, dtype=float)])
    return start if np.all(np.isfinite(start)) else init


def mps_fun(
    params: npt.NDArray,
    dist: Any,
    x: npt.NDArray,
    inv_trans: Callable[..., Any],
    const: Callable[..., Any],
    c: npt.NDArray | None,
    n: npt.NDArray,
    tl: npt.NDArray | None,
    tr: npt.NDArray | None,
    offset: bool,
) -> Any:
    if offset:
        gamma = inv_trans(const(params))[0]
        x_new = x - gamma
        tl, tr = _shifted_window(dist, tl, tr, gamma)
        params = inv_trans(const(params))[1:]
    else:
        params = inv_trans(const(params))
        x_new = x.copy()
        # The same for an unshifted fit. With no truncation the left
        # bound arrives here as the support's edge -- ``fit`` clamps
        # -inf onto it -- and F there is 0 whatever the parameters, but
        # evaluating the CDF *at* the edge taped a nan derivative for
        # every distribution with a singular derivative there (the
        # LogNormal's log(0), the ExpoWeibull's t**mu at mu < 1). The
        # objective was right, the gradient was nan, and BFGS handed
        # over to Nelder-Mead, whose absolute tolerances are neither
        # tight nor scale free: a censored ExpoWeibull MPS fit came back
        # 1% short of its optimum.
        s0, s1 = dist.support
        if np.isfinite(tl) and tl <= s0:
            tl = -np.inf
        if np.isfinite(tr) and tr >= s1:
            tr = np.inf
    D = dist.neg_mean_D(x_new, c, n, tl, tr, *params)
    return D


class _TowardLimit:
    """Whether an offset MPS search that stopped short of an optimum has
    run its offset towards the family's limit as it goes to -inf
    (``_offset_limit_family``: the Normal for a LogNormal or a Gamma, the
    Logistic for a LogLogistic, the smallest extreme value ``Gumbel`` for
    a Weibull), as the maximum-likelihood fit checks its own (#599).

    It has where the offset has moved down from its start and the limit's
    own MPS fit spaces the data at least as well as the point reached
    (its objective no higher): the family only approaches its limit, so
    its product of spacings has no finite maximum along the way, and on
    data with a long left tail (a value at -1 below the rest at 9 to 22)
    the search ran on for 12 s and ended "MPS FAILED" (#616). Called on a
    search's result; :meth:`warn` then says so.
    """

    def __init__(self, model: "Parametric", init: npt.NDArray) -> None:
        self.model = model
        self.start = self._gamma(init)
        family = getattr(model.dist, "_offset_limit_family", None)
        self.family: Any = family() if family is not None else None
        self._limit: list = []

    def _gamma(self, u: npt.NDArray) -> float:
        info = self.model.fitting_info
        with np.errstate(all="ignore"):
            return float(info["inv_trans"](info["const"](u))[0])

    def limit(self) -> "float | None":
        """The limit's MPS objective on the data, fitted when first asked
        for; ``None`` where its search failed."""
        if not self._limit:
            self._limit.append(None)
            from surpyval.utils.no_maximum import quiet_maximum_warnings

            with warnings.catch_warnings(), quiet_maximum_warnings():
                warnings.simplefilter("ignore")
                try:
                    fit = self.family.fit_from_surpyval_data(
                        self.model.surv_data, how="MPS"
                    )
                except (ValueError, ArithmeticError):
                    return None
            if fit.res.success and np.isfinite(fit.res.fun):
                self._limit[0] = float(fit.res.fun)
        return self._limit[0]

    def __call__(self, res: Any) -> bool:
        return not res.success and self.reached(res)

    def reached(self, res: Any) -> bool:
        """The check, whatever the search reported: BFGS can report
        success on the flat objective along the way (#630)."""
        if self.family is None:
            return False
        if not (np.all(np.isfinite(res.x)) and np.isfinite(res.fun)):
            return False
        if not self._gamma(res.x) < self.start:
            return False
        limit = self.limit()
        return limit is not None and limit <= float(res.fun)

    def warn(self, params: npt.NDArray) -> None:
        name, limit = self.model.dist.name, self.family.name
        warn_no_maximum(
            f"the {name}'s product of spacings keeps increasing as gamma "
            f"({float(params[0]):.4g}) runs on to -inf, towards a {limit} "
            "distribution that none of its members reaches: no "
            f"{name} the search reached spaces the data better than "
            "that limit",
            "The reported parameters are where the search stopped and are "
            "meaningless",
            f"fit surpyval.{limit} instead",
        )


def _hessian_or_nan(hess: Callable[..., Any]) -> Callable[..., Any]:
    """``hess``, NaN where autograd cannot take it: on a truncated
    LogLogistic's objective it raises ``TypeError('first operand must be
    array')`` from inside its backward pass. ``is_local_minimum`` then
    takes the Hessian from central differences of the gradient."""

    def safe(u: npt.NDArray, *args: Any) -> Any:
        try:
            return hess(u, *args)
        except TypeError:
            size = np.size(u)
            return np.full((size, size), np.nan)

    return safe


def _usable(res: Any) -> bool:
    """A result with finite parameters and a finite objective."""
    return bool(np.all(np.isfinite(res.x)) and np.isfinite(res.fun))


class _RunsOff:
    """Whether an MPS search that stopped short of a verified optimum has
    parameters running off, along which the product of spacings has no
    finite maximum (#630): Newton's method cannot converge along their
    profiles, the profile is flat to the verification's tolerance, and it
    rises towards an infinite end of the parameter's range. The check
    maximum likelihood makes of its own searches (``mle._runaway``).

    Asked of a BFGS result that failed (``fallback_minimize``'s
    ``give_up``), it ends the search there rather than escalate. A BFGS
    search that diverged (a non-finite objective where it stopped) goes
    straight to the derivative-free rung from the start, whose result is
    then :attr:`result`: Newton-CG from the same cold start follows the
    same derivatives at the price of the Hessian, and on an offset
    ExpoWeibull on the #599 data, whose spacings run off with no limit
    family to compare with, it spent 32 s of a 34 s fit before ending
    "MPS FAILED" anyway.
    """

    def __init__(
        self,
        model: "Parametric",
        args: tuple,
        init: npt.NDArray,
        floor: Any,
    ) -> None:
        self.model = model
        self.args = args
        self.init = np.asarray(init, dtype=float)
        self.floor = floor
        self.found: tuple = ()
        self.result: Any = None

    def check(self, x: npt.NDArray) -> tuple:
        """The positions in the search vector of the parameters running
        off at ``x`` (empty for none)."""
        from .mle import _runaway, _space

        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)):
            return ()
        natural, bounds, free = _space(self.model)[:3]
        size = np.maximum(np.abs(x), np.asarray(self.floor, dtype=float))

        def keep(j: int, slope: float) -> bool:
            if not abs(slope) * size[j] < OPTIMUM_GTOL:
                return False
            ahead = np.array(x, dtype=float)
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

        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                found = _runaway(
                    mps_fun,
                    self.args,
                    x,
                    self.init,
                    floor=self.floor,
                    keep=keep,
                )
            except (ValueError, ArithmeticError, TypeError):
                # (TypeError: autograd's second derivative of some
                # truncated objectives, see ``_hessian_or_nan``)
                found = ()
        return tuple(found)

    def __call__(self, res: Any) -> bool:
        from scipy.optimize import minimize

        if _usable(res):
            self.found = self.check(res.x)
            return bool(self.found)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nm = minimize(
                mps_fun, self.init, method="Nelder-Mead", args=self.args
            )
        nm.optimizer = "Nelder-Mead"
        self.result = nm
        self.found = self.check(nm.x) if _usable(nm) else ()
        return True

    def warn(self, params: npt.NDArray) -> None:
        from .mle import _runaway_names

        names = _runaway_names(self.model, self.found)
        values = dict(
            zip(
                sorted(
                    self.model.param_map, key=self.model.param_map.__getitem__
                ),
                params,
            )
        )
        running = " and ".join(
            (
                "{} ({:.4g})".format(name, float(values[name]))
                if name in values
                else name
            )
            for name in names
        )
        what = "runs" if len(names) == 1 else "run"
        warn_no_maximum(
            f"the {self.model.dist.name}'s product of spacings keeps "
            f"increasing as {running} {what} on, towards a limit of the "
            "family that none of its members reaches",
            "The reported parameters are where the search stopped and are "
            "meaningless",
            "fit by maximum likelihood (how='MLE'), or with a family that "
            "contains the limit",
        )


def mps(model: "Parametric") -> Any:
    """
    MPS: Maximum Product Spacing

    This is the method to get the largest (geometric) average distance
    between all points. This method works really well when all points are
    unique. Some complication comes in when using repeated data. This method
    is quite good for offset distributions.

    The answer is checked as maximum likelihood checks its own: a point
    that is not verifiably an optimum warns, as "No finite maximum" where
    a parameter runs off (#630).
    """

    dist = model.dist
    x, c, n = model.data["x"], model.data["c"], model.data["n"]
    const = model.fitting_info["const"]
    inv_trans = model.fitting_info["inv_trans"]
    init = model.fitting_info["init"]
    offset = model.offset
    tl = model.tl
    tr = model.tr

    jac = Gradient(mps_fun)
    hess = hessian(mps_fun)

    args = (dist, x, inv_trans, const, c, n, tl, tr, offset)
    floor = search_floor(model)
    toward_limit = _TowardLimit(model, init) if offset else None
    runs_off = _RunsOff(model, args, init, floor)

    def give_up(res: Any) -> bool:
        if toward_limit is not None and toward_limit(res):
            return True
        return runs_off(res)

    res = fallback_minimize(
        mps_fun,
        init,
        args,
        jac,
        hess,
        newton_tol=1e-15,
        floor=floor,
        give_up=give_up,
    )
    if runs_off.result is not None:
        res = runs_off.result

    params = inv_trans(const(res.x))
    # An optimiser's success is not an optimum: BFGS reports success on a
    # flat objective, and an offset run far down ended there in silence
    # (#630). The answer is checked as maximum likelihood checks its own.
    verified = _usable(res) and is_local_minimum(
        mps_fun, jac, _hessian_or_nan(hess), res.x, args, floor=floor
    )
    if not (verified or runs_off.found) and _usable(res):
        runs_off.found = runs_off.check(res.x)
    if not verified and toward_limit is not None and toward_limit.reached(res):
        toward_limit.warn(params)
    elif not verified and runs_off.found:
        runs_off.warn(params)
    elif (res.success is False) or (np.isnan(res.x).any()):
        from surpyval.utils.warnings import caller_stacklevel

        reason = str(res.get("message", "")).rstrip(".")
        warnings.warn(
            "MPS FAILED: the maximum product of spacings search found no "
            "optimum{}; the parameters returned are where it stopped. Try "
            "alternate estimation method (how='MLE').".format(
                f" ({reason})" if reason else ""
            ),
            UserWarning,
            stacklevel=caller_stacklevel(),
        )
    elif not verified:
        from surpyval.utils.warnings import caller_stacklevel

        warnings.warn(
            "The maximum product of spacings search did not reach a "
            "verified optimum of its objective (a point where the "
            "gradient is zero and the product of spacings curves down in "
            "every direction); the parameters returned are the best point "
            "it found: check the fit, or use another method (how='MLE').",
            UserWarning,
            stacklevel=caller_stacklevel(),
        )

    results = {}
    results["res"] = res

    if offset:
        results["gamma"] = params[0]
        results["params"] = params[1::]
    else:
        results["gamma"] = 0.0
        results["params"] = params

    return results
