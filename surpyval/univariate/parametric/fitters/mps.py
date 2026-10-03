import warnings
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..parametric import Parametric

import autograd.numpy as np
import numpy.typing as npt
from autograd import hessian

from surpyval.utils.no_maximum import warn_no_maximum

from . import Gradient, fallback_minimize, search_floor


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
        self.family = family() if family is not None else None
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
        if self.family is None or res.success:
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


def mps(model: "Parametric") -> Any:
    """
    MPS: Maximum Product Spacing

    This is the method to get the largest (geometric) average distance
    between all points. This method works really well when all points are
    unique. Some complication comes in when using repeated data. This method
    is quite good for offset distributions.
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
    toward_limit = _TowardLimit(model, init) if offset else None
    res = fallback_minimize(
        mps_fun,
        init,
        args,
        jac,
        hess,
        newton_tol=1e-15,
        floor=search_floor(model),
        give_up=toward_limit,
    )

    params = inv_trans(const(res.x))
    if toward_limit is not None and toward_limit(res):
        toward_limit.warn(params)
    elif (res.success is False) or (np.isnan(res.x).any()):
        warnings.warn("MPS FAILED: Try alternate estimation method")

    results = {}
    results["res"] = res

    if offset:
        results["gamma"] = params[0]
        results["params"] = params[1::]
    else:
        results["gamma"] = 0.0
        results["params"] = params

    return results
