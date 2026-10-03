"""The likelihood-ratio bounds of a fitted parametric regression model.

``RegressionLikelihoodRatio`` gives the searches of the univariate
models' likelihood-ratio bounds
(:class:`~surpyval.univariate.parametric._likelihood_ratio.LikelihoodRatioMixin`,
#421, #519, #535, #601) a regression model's likelihood and parameters:
the bounds are the same searches, over all the model's free parameters
(the distribution's and the covariate coefficients, or the life model's),
of a function evaluated at the covariates asked for (#583). ``cb_lr`` and
``param_cb_lr`` are ``cb(method="lr")`` and ``param_cb(method="lr")`` of
:class:`~surpyval.univariate.regression.parametric_regression_model.ParametricRegressionModel`.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable

import autograd.numpy as np
import numpy.typing as npt
from scipy.special import expit
from scipy.special import ndtri as z

from surpyval.univariate.parametric._likelihood_ratio import (
    LikelihoodRatioMixin,
    central_gradient,
    warn_unsettled,
)
from surpyval.utils.shapes import check_paired_rows, covariate_rows
from surpyval.utils.validation import BOUNDS, CB_ON, check_option
from surpyval.utils.warnings import caller_stacklevel

if TYPE_CHECKING:
    from .parametric_regression_model import ParametricRegressionModel

# The ends of the scales the searches move a function on (as the
# univariate band's): past them it is no longer a double distinct from the
# edge of its range.
_LN_MAX = float(np.log(np.finfo(float).max))
_LN_TINY = float(np.log(np.finfo(float).tiny))
_FLOAT_MAX = float(np.finfo(float).max)

#: The names ``method`` takes for the likelihood-ratio bounds (as the
#: univariate models'; case does not matter).
LR_NAMES = ("lr", "likelihood", "likelihood-ratio", "profile")


def is_lr(method: str) -> bool:
    """Whether ``method`` asks for the likelihood-ratio bounds: ``"lr"``
    or one of its aliases; ``"wald"`` is the other option, and anything
    else is refused."""
    m = str(method).lower()
    if m in LR_NAMES:
        return True
    if m != "wald":
        check_option(
            "method",
            method,
            ("wald", "lr"),
            "Case does not matter, and 'likelihood', "
            "'likelihood-ratio' and 'profile' also mean 'lr'.",
        )
    return False


def critical_value(alpha_ci: float, bound: str) -> float:
    """The chi-squared (one degree of freedom) critical value of a
    likelihood-ratio bound at ``alpha_ci``: one side of a two-sided bound
    takes ``alpha_ci / 2``."""
    tail = alpha_ci / 2.0 if bound == "two-sided" else alpha_ci
    return float(z(1.0 - tail) ** 2)


class RegressionLikelihoodRatio(LikelihoodRatioMixin):
    """The likelihood-ratio searches of a fitted
    :class:`ParametricRegressionModel`, at the parameters ``params`` with
    the baseline at the covariates ``center``.

    The searches are the univariate models' (``LikelihoodRatioMixin``):
    the parameters' profiles walked out to the critical value, and a
    function's extreme over the region they bound found by
    ``_PsiBoundSearch``. This class gives them the regression model's
    likelihood (``model.neg_ll`` on the data, centred at ``center``), its
    parameters' declared bounds (the distribution's, then the life
    model's, or none for a coefficient), the parameters it held fixed or
    could not determine, and its covariance ``cov`` as the Wald scale the
    searches start from. A ``cb`` searches in the parameterisation of the
    centred fit behind a model that reports its baseline at ``Z = 0``
    (#463), where the coefficients and the baseline are not nearly
    collinear: the likelihood, and so the region and its bounds, are the
    same in both.
    """

    method = "MLE"
    offset = False
    lfp = False
    zi = False

    def __init__(
        self,
        model: "ParametricRegressionModel",
        params: npt.NDArray,
        center: "npt.NDArray | None",
        cov: "npt.NDArray | None",
        point: tuple,
    ) -> None:
        from ._fit_skeleton import centred_copy

        self.fitted = model
        self.point = point
        self.params = np.array(params, dtype=float)
        self.center = center
        data = model.data
        if center is not None and np.any(center):
            data = centred_copy(data, center)
        self.surv_data = data
        self.dist = model.distribution
        names = model.parameter_names
        held = model._held()
        self._held_idx = {i for i, nm in enumerate(names) if nm in held}
        self._bounds = list(model._parameter_bounds())
        free = [i for i in range(len(names)) if i not in self._held_idx]
        if cov is not None:
            cov = np.asarray(cov, dtype=float)
            if not np.all(np.isfinite(cov[np.ix_(free, free)])):
                cov = None
        self.hess_inv = cov

    # -- what the searches read --------------------------------------------
    def _user_fixed_idx(self) -> set:
        return self._held_idx

    def _lr_lean_data(self) -> None:
        return None

    def _lr_full_neg_ll(self, theta: npt.NDArray) -> float:
        return float(self.fitted.model.neg_ll(self.surv_data, *theta))

    def _lr_declared_bounds(self) -> list[tuple[Any, Any]]:
        return self._bounds

    def _lr_limits(self) -> list[tuple[float, float]]:
        # No data-derived limits: with covariates, the data do not bound a
        # baseline's support parameter as they do a univariate one's.
        return [
            (-np.inf if lo is None else lo, np.inf if hi is None else hi)
            for lo, hi in self._bounds
        ]

    def _free(self) -> list[int]:
        return [j for j in range(len(self.params)) if j not in self._held_idx]

    # -- the region --------------------------------------------------------
    def _lr_find_region(self, free: list[int], crit: float) -> tuple[
        list[tuple[Any, Any]],
        list[list[npt.NDArray]],
        list[npt.NDArray] | None,
    ]:
        """The region the searches start from (``_lr_region``): the box
        of the coordinates' ends, no walks' points, and the boundary
        traced along rays in every direction of the Wald metric
        (``_lr_trace_rays``).

        The univariate searches walk every parameter's profile out to the
        critical value first, for the box and the points far out along
        the region's valleys. A regression model has more parameters, and
        a likelihood many times slower (an accelerated life model's, about
        1 ms on 72 units): the walks of the four parameters of #583's
        model were 85% of an 85 s bound. Its region is near an
        ellipsoid in the search coordinates (its log-life is linear in
        them), and each bound is found from the traced point where the
        function is most extreme, which includes the boundary along the
        function's own Wald direction (``_cb_lr_psi_bounds``). The answer
        is checked as the univariate one is (``_PsiBoundSearch``), and
        where it does not check out the searches go on as theirs do."""
        coords, limits = self._lr_coords()
        box = self._lr_box(
            [coords[j] for j in free], [limits[j] for j in free]
        )
        seeds: list[list[npt.NDArray]] = [[] for _ in range(2 * len(free))]
        with np.errstate(all="ignore"):
            trace = self._lr_trace(free, crit, seeds)
        return box, seeds, trace

    def _lr_traces(self, free: list[int]) -> bool:
        return len(free) >= 1

    def _lr_trace_rays(self, L: npt.NDArray) -> list[npt.NDArray]:
        """Rays along each axis of the Wald metric (whose Cholesky factor
        is ``L``) both ways, and, with up to five parameters, along each
        diagonal: ``2 k + 2**k`` rays for ``k`` parameters."""
        k = L.shape[0]
        whitened = [s * e for e in np.eye(k) for s in (1.0, -1.0)]
        if 1 < k <= 5:
            for signs in np.ndindex(*(2,) * k):
                whitened.append((1.0 - 2.0 * np.array(signs)) / np.sqrt(k))
        return [L @ w for w in whitened]

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
        """``LikelihoodRatioMixin._cb_lr_psi_bounds``, with the boundary
        along the Wald direction of ``psi_of`` added to the traced points:
        where the region is an ellipsoid, the extreme is there."""
        extra = self._psi_rays(psi_of, free, crit)
        trace = (
            None if trace is None and not extra else [*(trace or []), *extra]
        )
        return super()._cb_lr_psi_bounds(
            psi_of,
            free,
            crit,
            want_lower,
            want_upper,
            ends,
            box,
            seeds,
            trace,
            hints=hints,
            unsettled=unsettled,
        )

    def _psi_rays(
        self,
        psi_of: Callable[[npt.NDArray], float],
        free: list[int],
        crit: float,
    ) -> list[npt.NDArray]:
        """The boundary of the region along the Wald direction of
        ``psi_of`` both ways: ``± Sigma g``, ``g`` its gradient in the
        search coordinates at the estimate and ``Sigma`` the Wald
        covariance there (the points of the Wald ellipsoid where its
        linear approximation is most extreme)."""
        if self.hess_inv is None:
            return []
        L = self._lr_wald_factor(free)
        if L is None:
            return []
        excess, u_hat, _ = self._lr_ray_excess(free, crit)
        coords, _ = self._lr_coords()

        def psi_u(u: npt.NDArray) -> float:
            theta = self.params.copy()
            theta[free] = [coords[j].from_u(v) for j, v in zip(free, u)]
            return float(psi_of(theta))

        out = []
        with np.errstate(all="ignore"):
            g = central_gradient(psi_u, u_hat)
            d = L @ (L.T @ g)
            size = float(np.sqrt(g @ d))
            if not (np.all(np.isfinite(d)) and np.isfinite(size) and size > 0):
                return []
            for sign in (1.0, -1.0):
                ray = sign * d / size
                r = self._lr_crossing(excess, u_hat, ray, crit)
                if r is not None:
                    out.append(u_hat + r * ray)
        return out

    # -- the bounds --------------------------------------------------------
    def param_sides(
        self, idx: int, crit: float, want: tuple[bool, bool]
    ) -> tuple[float, float]:
        """``(lower, upper)`` of the profile-likelihood interval on
        parameter ``idx`` at the critical value ``crit`` (``nan`` for a
        side not asked for, or not found): the extreme of the parameter
        over the region, which is where its profile deviance reaches
        ``crit`` -- the same bound as the univariate walk out along the
        profile (``_lr_param_side``) finds, by the searches of a band
        (``_PsiBoundSearch``), searched in the parameter's coordinate
        (its log, for a positive one). A side whose region reaches the
        edge of the parameter's space is that edge. Each side is kept."""
        cache = self.__dict__.setdefault("_lr_params", {})
        keys = [(idx, float(crit), side) for side in (-1.0, 1.0)]
        need = [w and key not in cache for w, key in zip(want, keys)]
        if any(need):
            coords, _ = self._lr_coords()
            coord = coords[idx]
            free = self._free()
            box, seeds, trace = self._lr_region(free, crit)
            sides: set[float] = set()
            with np.errstate(all="ignore"):
                found = self._cb_lr_psi_bounds(
                    lambda theta: coord.to_u(theta[idx]),
                    free,
                    crit,
                    need[0],
                    need[1],
                    coord.ends,
                    box,
                    seeds,
                    trace,
                    unsettled=sides,
                )
            for key, n_, v in zip(keys, need, found):
                if n_:
                    cache[key] = (coord.from_u(v), key[2] in sides)
        out = []
        for w, key in zip(want, keys):
            out.append(cache[key] if w else (np.nan, False))
        if any(unsure for _, unsure in out):
            warn_unsettled("on '{}'".format(self.fitted.parameter_names[idx]))
        return out[0][0], out[1][0]

    def band(
        self,
        x: npt.NDArray,
        rows: npt.NDArray,
        on: str,
        crit: float,
        want: tuple[bool, bool],
    ) -> tuple[npt.NDArray, npt.NDArray, list[int], list[int]]:
        """The likelihood-ratio bounds on ``on`` at each pair of a time
        ``x[i]`` and covariate row ``rows[i]``: ``(lower, upper)`` on the
        scale of ``on`` (``nan`` for a side not asked for), and the
        positions where a side was not found, and where its search did
        not converge (``_lr_band``).

        Searched on the scale of the univariate band: the logit of ``sf``
        (from which the ``sf``, ``ff`` and ``Hf`` bounds all come), the
        log of ``hf`` or ``df``. An additive hazards model's hazard and
        cumulative hazard can be negative (documented), so its bounds are
        searched on ``-Hf`` and on ``hf`` or ``df`` themselves, and a
        bound on ``sf``, ``ff`` or ``Hf`` that reaches past the range of
        a probability (parameters in the region with a negative ``Hf``
        there) ends at it, as the Wald bound stays inside it. The bounds
        are those of the region whatever the scale: it moves only where
        the searches step.
        """
        model = self.fitted.model
        additive = self.fitted._is_additive()
        survival = on in ("sf", "ff", "Hf")
        free = self._free()
        value: Callable[[Any], Any]
        if survival:

            def psi_of(i: int, theta: npt.NDArray) -> float:
                H = model.Hf(x[i : i + 1], rows[i : i + 1], *theta)
                H = float(np.asarray(H, dtype=float).reshape(-1)[0])
                if additive:
                    return -H
                # The logit of sf, exp(-H) / (1 - exp(-H)), from H
                return float(-H - np.log(-np.expm1(-H)))

            ends = (_LN_TINY, -_LN_TINY)
            if additive:
                # The region can hold parameters with a negative H here
                # (sf above 1, the additive model's own): a bound on a
                # probability ends at 1, as the Wald bound keeps inside it.
                value = {
                    "sf": lambda v: np.exp(min(v, 0.0)),
                    "ff": lambda v: -np.expm1(min(v, 0.0)),
                    "Hf": lambda v: -min(v, 0.0),
                }[on]
            else:
                value = {
                    "sf": expit,
                    "ff": lambda v: expit(-v),
                    "Hf": lambda v: np.logaddexp(0.0, -v),
                }[on]
        else:
            fn = model.hf if on == "hf" else model.df

            def psi_of(i: int, theta: npt.NDArray) -> float:
                g = fn(x[i : i + 1], rows[i : i + 1], *theta)
                g = float(np.asarray(g, dtype=float).reshape(-1)[0])
                return g if additive else float(np.log(g))

            if additive:
                ends, value = (-_FLOAT_MAX, _FLOAT_MAX), lambda v: v
            else:
                ends, value = (_LN_TINY, _LN_MAX), np.exp
        # ff and Hf fall as sf rises: their lower bound is sf's upper.
        falling = on in ("ff", "Hf")
        if falling:
            want = (want[1], want[0])
        n = len(x)
        if not free:
            # Every parameter was held: the region is the estimate.
            at = np.array([value(psi_of(i, self.params)) for i in range(n)])
            return at, at, [], []
        lower, upper, failed, unsettled = self._lr_band(
            np.arange(n),
            np.argsort(x, kind="stable"),
            psi_of,
            lambda i: (float(x[i]), rows[i].tobytes()),
            "survival" if survival else on,
            free,
            crit,
            ends,
            value,
            want,
        )
        if falling:
            lower, upper = upper, lower
        return lower, upper, failed, unsettled

    def quantiles(
        self,
        p: npt.NDArray,
        rows: npt.NDArray,
        t_hat: npt.NDArray,
        crit: float,
        want: tuple[bool, bool],
    ) -> tuple[npt.NDArray, npt.NDArray, list[int], list[int]]:
        """The likelihood-ratio bounds on the quantile ``qf(p[i])`` at
        each covariate row ``rows[i]`` (centred as the searches' are),
        ``t_hat`` the quantiles at the estimate: the extreme of the
        quantile over the region, searched on its log above the start of
        the support (the quantile itself for a baseline on the whole
        line), as the univariate ``quantile_cb(method="lr")`` searches
        it. Returned as ``band`` returns its bounds."""
        model = self.fitted.model
        lower_edge = float(self.dist.support[0])
        logged = np.isfinite(lower_edge)
        target = -np.log1p(-np.asarray(p, dtype=float))

        def psi_of(i: int, theta: npt.NDArray) -> float:
            t = _invert_H(
                lambda t: model.Hf(t, rows[i : i + 1], *theta),
                lambda t: model.hf(t, rows[i : i + 1], *theta),
                float(target[i]),
                float(t_hat[i]),
                lower_edge,
            )
            if logged:
                return float(np.log(t - lower_edge))
            return t

        if logged:
            ends = (_LN_TINY, _LN_MAX)

            def value(v: Any) -> Any:
                return lower_edge + np.exp(v)

        else:
            ends = (-_FLOAT_MAX, _FLOAT_MAX)

            def value(v: Any) -> Any:
                return v

        n = len(p)
        free = self._free()
        if not free:
            at = np.array([value(psi_of(i, self.params)) for i in range(n)])
            return at, at, [], []
        return self._lr_band(
            np.arange(n),
            np.argsort(p, kind="stable"),
            psi_of,
            lambda i: (float(p[i]), rows[i].tobytes()),
            "qf",
            free,
            crit,
            ends,
            value,
            want,
        )


def _invert_H(
    H: Callable[[npt.NDArray], Any],
    h: Callable[[npt.NDArray], Any],
    target: float,
    start: float,
    lower_edge: float,
) -> float:
    """The time at which the cumulative hazard ``H`` reaches ``target``:
    Newton's steps on ``log H`` in ``log(t - lower_edge)`` (or in ``t``
    for a support with no start) from ``start``, the quantile at the
    estimate, near which the searches ask; ``nan`` where they do not
    settle in 100 steps."""
    logged = np.isfinite(lower_edge)
    s = np.log(start - lower_edge) if logged else start
    for _ in range(100):
        t = lower_edge + np.exp(s) if logged else s
        H_t = float(np.asarray(H(np.array([t])), dtype=float).reshape(-1)[0])
        h_t = float(np.asarray(h(np.array([t])), dtype=float).reshape(-1)[0])
        if not (np.isfinite(H_t) and H_t > 0 and np.isfinite(h_t) and h_t > 0):
            return np.nan
        # d log H / ds = h dt/ds / H
        slope = h_t * (t - lower_edge if logged else 1.0) / H_t
        step = (np.log(H_t) - np.log(target)) / slope
        # (at most a factor of e a step, on the log scale)
        step = float(np.clip(step, -1.0, 1.0)) if logged else step
        s = s - step
        if abs(step) <= 1e-13 * max(1.0, abs(s)):
            return float(lower_edge + np.exp(s) if logged else s)
    return np.nan


def lr_search(model: Any, reported: bool) -> RegressionLikelihoodRatio:
    """The likelihood-ratio searches of ``model``: in the
    parameterisation its bounds on the functions are computed in
    (``_inference_state``), or, with ``reported``, in that of the
    parameters it reports (whose profiles ``param_cb`` walks). Kept on
    the model, with what they have found, while its parameters and data
    are as they are."""
    from ._inference import _same_point

    if getattr(model, "data", None) is None or model._restored:
        raise ValueError(
            "Likelihood-ratio bounds need the data the model was fitted "
            "to, which a model restored from a dict does not keep; use "
            "method='wald'."
        )
    with warnings.catch_warnings():
        # The searches need no covariance (it only scales their steps):
        # without one, they go on without it.
        warnings.simplefilter("ignore")
        if reported or model._fit_centring is None:
            params = np.asarray(model._eval_params(), dtype=float)
            center = model.center
            cov = model.covariance()
        else:
            params, center, cov = model._inference_state()
    point = model._covariance_point(params, center)
    kept = model._lr_searches
    if kept is None:
        kept = model._lr_searches = []
    for search in kept:
        if _same_point(search.point, point):
            return search
    search = RegressionLikelihoodRatio(model, params, center, cov, point)
    # (one per parameterisation: the model's own, and the centred fit's)
    del kept[:-1]
    kept.append(search)
    return search


def param_cb_lr(
    model: Any,
    name: str,
    alpha_ci: float,
    bound: str,
) -> npt.NDArray:
    """``param_cb(method="lr")``: the profile-likelihood interval on the
    parameter ``name`` of ``model`` (already checked to be one it
    estimated or held), as the univariate ``param_cb(method="lr")``
    defines it: the values whose profile deviance, the other parameters
    re-fitted, stays below the critical value (``RegressionLikelihoodRatio
    .param_sides``); the edge of the parameter's space where it stays
    below it to there."""
    check_option("bound", bound, BOUNDS)
    names = model.parameter_names
    idx = names.index(name)
    search = lr_search(model, reported=True)
    if idx in search._held_idx:
        # Held fixed in the fit, or not determined by the data: the
        # degenerate interval at its value, or nan, as the Wald bound.
        value = float(search.params[idx])
        if names[idx] not in model.fixed:
            value = np.nan
        return np.array([value, value] if bound == "two-sided" else [value])
    want = (bound in ("two-sided", "lower"), bound in ("two-sided", "upper"))
    found = search.param_sides(idx, critical_value(alpha_ci, bound), want)
    for which, w, value in zip(("lower", "upper"), want, found):
        if w and np.isnan(value):
            warnings.warn(
                f"The likelihood-ratio {which} bound on '{name}' could "
                "not be found (the constrained optimiser failed from every "
                "start); nan is returned for it. method='wald' gives a "
                "bound in its place.",
                RuntimeWarning,
                stacklevel=caller_stacklevel(),
            )
    return np.array([v for w, v in zip(want, found) if w])


def cb_lr(
    model: Any,
    x: npt.ArrayLike,
    Z: Any,
    on: str,
    alpha_ci: float,
    bound: str,
) -> npt.NDArray:
    """``cb(method="lr")``: at each time and covariate row, the extreme
    of ``on`` over the likelihood region of all the model's free
    parameters, ``{theta : 2[nll(theta) - nll_hat] <= chi2_1}``, as the
    univariate ``cb(method="lr")`` finds it (``RegressionLikelihoodRatio
    .band``). Below the support the bound is the estimate (``sf`` is 1
    there), as the Wald bound's."""
    check_option("on", on, CB_ON)
    check_option("bound", bound, BOUNDS)
    on = {"R": "sf", "F": "ff"}.get(on, on)
    search = lr_search(model, reported=False)
    Zp = model._centred(model._prepare_Z(Z), search.center)
    rows = covariate_rows(Zp, model._n_covariates())
    t = np.atleast_1d(np.asarray(x, dtype=float)).reshape(-1)
    check_paired_rows(t.size, rows.shape[0], grid=False)
    n = max(t.size, rows.shape[0])
    t = np.broadcast_to(t, (n,)).copy()
    rows = np.ascontiguousarray(np.broadcast_to(rows, (n, rows.shape[1])))
    want = (bound in ("two-sided", "lower"), bound in ("two-sided", "upper"))
    crit = critical_value(alpha_ci, bound)

    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    below = t < model.distribution.support[0]
    inside = np.flatnonzero(~below)
    failed: list[float] = []
    unsettled: list[float] = []
    if inside.size:
        lo, hi, bad, unsure = search.band(
            t[inside], rows[inside], on, crit, want
        )
        lower[inside], upper[inside] = lo, hi
        failed = [float(t[inside][i]) for i in bad]
        unsettled = [float(t[inside][i]) for i in unsure]
    if below.any():
        # Nothing has happened yet: the bound is the estimate.
        at = 1.0 if on == "sf" else 0.0
        lower[below] = at if want[0] else np.nan
        upper[below] = at if want[1] else np.nan
    if unsettled:
        warn_unsettled(f"at x = {sorted(set(unsettled))}")
    if failed:
        warnings.warn(
            "The likelihood-ratio bound could not be found at "
            f"x = {sorted(set(failed))} (the constrained optimiser "
            "failed from every start); nan is returned there. "
            "method='wald' gives a bound in its place.",
            RuntimeWarning,
            stacklevel=caller_stacklevel(),
        )
    if bound == "two-sided":
        return np.column_stack([lower, upper])
    return lower if bound == "lower" else upper


def quantile_cb_lr(
    model: Any,
    p: npt.NDArray,
    rows: npt.NDArray,
    t_hat: npt.NDArray,
    alpha_ci: float,
    bound: str,
) -> npt.NDArray:
    """``quantile_cb(method="lr")``: at each probability ``p[i]`` and
    covariate row ``rows[i]`` (as given), the extreme of the quantile
    over the likelihood region (``RegressionLikelihoodRatio.quantiles``),
    ``t_hat`` the quantiles at the estimate."""
    search = lr_search(model, reported=False)
    rows_c = np.asarray(model._centred(rows, search.center), dtype=float)
    rows_c = np.ascontiguousarray(rows_c)
    want = (bound in ("two-sided", "lower"), bound in ("two-sided", "upper"))
    crit = critical_value(alpha_ci, bound)
    lower, upper, failed, unsettled = search.quantiles(
        p, rows_c, t_hat, crit, want
    )
    if unsettled:
        at = sorted({float(p[i]) for i in unsettled})
        warn_unsettled(f"on qf at p = {at}")
    if failed:
        at = sorted({float(p[i]) for i in failed})
        warnings.warn(
            f"The likelihood-ratio bound on qf could not be found at p = {at} "
            "(the constrained optimiser failed from every start); nan is "
            "returned there. method='wald' gives a bound in its place.",
            RuntimeWarning,
            stacklevel=caller_stacklevel(),
        )
    if bound == "two-sided":
        return np.column_stack([lower, upper])
    return lower if bound == "lower" else upper
