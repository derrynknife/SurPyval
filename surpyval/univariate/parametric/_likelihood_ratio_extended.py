"""The likelihood-ratio bounds of an offset, limited-failure-population or
zero-inflated model.

``ExtendedLikelihoodRatio`` runs the searches of ``LikelihoodRatioMixin``
over every fitted parameter of such a model -- the offset ``gamma``, the
distribution's own, the limited-failure proportion ``lfp_p`` and the
zero-inflation fraction ``f0`` -- in the order of the model's
``param_map``. A :class:`Parametric` model with any of the three hands
its ``param_cb``, ``cb``, ``quantile_cb``, ``mean_cb`` and ``mrl_cb``
with ``method="lr"`` to one, so that the bounds profile the extra
parameters as well, and bound them.

Two things differ from a plain model's searches:

- ``lfp_p`` and ``f0`` have a closed edge (``lfp_p = 1``, every unit can
  fail; ``f0 = 0``, none is dead on arrival) at which the model is the
  plain one, and their estimates can sit on it. A side of the interval
  towards the edge is the edge itself wherever the profile deviance there
  is within the critical value (``_lr_param_side``); with the estimate on
  the edge the other side is walked out in the parameter itself, where
  the deviance has curvature, rather than in its logit, which maps the
  edge to infinity.
- A function of time is the model's own (``sf`` with the extra mass and
  the offset), and at a time before the estimated offset it is at the
  edge of its range at the estimate but not over the whole region: its
  bound is searched from a point of the region where the offset is below
  that time (``_OffsetStartSearch``).
"""

from __future__ import annotations

import warnings
from copy import copy
from typing import Any, Callable

import autograd.numpy as np
import numpy as onp
import numpy.typing as npt
from scipy.optimize import brentq
from scipy.special import ndtri as z

from surpyval.utils.validation import BOUNDS, CB_ON, check_option
from surpyval.utils.warnings import caller_stacklevel

from ._likelihood_ratio import (
    _LR_UNREACHABLE,
    LikelihoodRatioMixin,
    _LRCoord,
    _LRPath,
    _PsiBoundSearch,
)

#: A profile deviance at the closed edge of ``lfp_p`` or ``f0`` below this
#: means the estimate is on the edge (``ExtendedLikelihoodRatio``): the
#: side away from it is walked out in the parameter itself.
_LR_ON_EDGE = 1e-3


def critical_value(alpha_ci: float, bound: str) -> float:
    """The chi-squared critical value of a bound at ``alpha_ci``: one
    degree of freedom, at ``2 alpha_ci`` for a one-sided bound."""
    if bound == "two-sided":
        return float(z(1.0 - alpha_ci / 2.0) ** 2)
    return float(z(1.0 - alpha_ci) ** 2)


#: The first step of that walk where the parameter has no Wald standard
#: error (an estimate on the edge has none).
_LR_EDGE_STEP = 1e-3


class _OffsetStartSearch(_PsiBoundSearch):
    """``_PsiBoundSearch`` for a function that is at an edge of its range
    at the estimate but not over the whole likelihood region: the
    survival at a time before the estimated offset is 1 there (its logit
    is infinite), and below 1 wherever the region holds an offset below
    that time.

    The bound on the side away from the edge is searched as usual, from
    the point of the region found so far (the parameters' walks and the
    traced boundary) where the function is most extreme on that side; the
    side at the edge is the edge. Where no point of the region found has
    the function inside its range, both bounds are the edge, as for a
    plain model.
    """

    def run(
        self,
        want_lower: bool,
        want_upper: bool,
        seeds: list[list[npt.NDArray]],
        trace: list[npt.NDArray] | None,
    ) -> tuple[float, float]:
        psi_hat = self.psi_of(self.theta_of(self.u_hat))
        if not onp.isinf(psi_hat):
            return super().run(want_lower, want_upper, seeds, trace)
        side = 1.0 if psi_hat > 0 else -1.0
        best, best_psi = None, np.nan
        points = [u for group in seeds for u in group] + list(trace or [])
        with np.errstate(all="ignore"):
            for u in points:
                u = self.model._lr_start(u, self.box)
                psi = self.psi_of(self.theta_of(u))
                if not onp.isfinite(psi) or not self.dev_u(u) <= self.crit:
                    continue
                if best is None or side * (psi - best_psi) < 0:
                    best, best_psi = u, float(psi)
        if best is None:
            return psi_hat, psi_hat
        # The searches' origin and first start: a point of the region
        # where the function is inside its range.
        self.u_hat = np.asarray(best, dtype=float)
        lower, upper = super().run(
            want_lower and side > 0, want_upper and side < 0, seeds, trace
        )
        if side > 0:
            return lower, psi_hat if want_upper else np.nan
        return psi_hat if want_lower else np.nan, upper


class ExtendedLikelihoodRatio(LikelihoodRatioMixin):
    """The likelihood-ratio searches of a fitted offset, limited-failure
    or zero-inflated :class:`Parametric` model ``model``, over the vector
    ``phi`` of its fitted parameters in the order of ``model.param_map``:
    ``gamma`` first with an offset, the distribution's parameters, then
    ``lfp_p`` and ``f0``.

    ``held`` adds parameters (by position in ``phi``) to those the user
    fixed: the summary bounds of a limited-failure model with its
    proportion at 1 search the slice of the region where it stays there
    (see ``Parametric._lr_summary``).
    """

    method = "MLE"
    # What the mixin's own entry points check: the extra parameters are
    # in ``phi`` here, and its functions of time take ``phi`` whole.
    offset = False
    lfp = False
    zi = False
    gamma = 0.0
    f0 = 0.0
    lfp_p = 1.0
    _psi_search_class = _OffsetStartSearch

    def __init__(self, model: Any, held: "set[int] | None" = None) -> None:
        self.model = model
        self.dist = model.dist
        self.surv_data = model.surv_data
        pmap = model.param_map
        self.names = sorted(pmap, key=pmap.__getitem__)
        core_names = sorted(
            model.dist.param_map, key=model.dist.param_map.__getitem__
        )
        self._core_idx = [pmap[name] for name in core_names]
        self._gamma_idx = pmap.get("gamma") if model.offset else None
        self._p_idx = pmap.get(model.lfp_name) if model.lfp else None
        self._f0_idx = pmap.get("f0") if model.zi else None
        phi = onp.empty(len(self.names))
        phi[self._core_idx] = onp.asarray(model.params, dtype=float)
        for idx, value in (
            (self._gamma_idx, model.gamma),
            (self._p_idx, model.lfp_p),
            (self._f0_idx, model.f0),
        ):
            if idx is not None:
                phi[idx] = float(value)
        self.params = phi
        self._bounds = [
            (lo, hi) for lo, hi in model.bounds  # gamma's upper: the data's
        ]
        # The closed edges: the plain model is there.
        self._closed_edges: dict[int, float] = {}
        if self._p_idx is not None:
            self._closed_edges[self._p_idx] = 1.0
        if self._f0_idx is not None:
            self._closed_edges[self._f0_idx] = 0.0
        self._held = set(model._user_fixed_idx()) | set(held or ())
        # A copy of the model whose parameters are set to the point a
        # function is evaluated at (``_at``): its own functions, with the
        # offset, the extra mass and their guards.
        self._scratch = copy(model)
        self.hess_inv = self._wald_covariance()

    # -- what the searches read --------------------------------------------
    def _user_fixed_idx(self) -> set:
        return self._held

    def _lr_lean_data(self) -> None:
        return None

    def _lr_plain(self) -> bool:
        return False

    def _split(self, phi: npt.NDArray) -> tuple[npt.NDArray, Any, Any, Any]:
        """``(core, gamma, p, f0)`` of the vector ``phi``."""
        core = phi[self._core_idx]
        gamma = 0.0 if self._gamma_idx is None else phi[self._gamma_idx]
        p = 1.0 if self._p_idx is None else phi[self._p_idx]
        f0 = 0.0 if self._f0_idx is None else phi[self._f0_idx]
        return core, gamma, p, f0

    def _lr_full_neg_ll(self, theta: npt.NDArray) -> float:
        core, gamma, p, f0 = self._split(np.asarray(theta, dtype=float))
        return float(
            self.dist._neg_ll_func(self.surv_data, *core, gamma, f0, p)
        )

    def _lr_declared_bounds(self) -> list[tuple[Any, Any]]:
        return self._bounds

    def _lr_limits(self) -> list[tuple[float, float]]:
        # The offset's upper bound is already the data's (``_offset_upper``)
        return [
            (-np.inf if lo is None else lo, np.inf if hi is None else hi)
            for lo, hi in self._bounds
        ]

    def _at(self, phi: npt.NDArray) -> Any:
        """The model at ``phi`` (the copy ``_scratch``, set there)."""
        core, gamma, p, f0 = self._split(onp.asarray(phi, dtype=float))
        s = self._scratch
        s.params = onp.array(core, dtype=float)
        s.gamma = float(gamma)
        s.lfp_p = float(p)
        s.f0 = float(f0)
        # (``mean`` keeps what it computed)
        s.__dict__.pop("_mean", None)
        return s

    def _lr_function(self, name: str) -> Callable[..., Any]:
        """The model's own function ``name`` at ``phi``, ``f(x, *phi)``,
        of the time itself (``gamma`` is in ``phi``: the mixin's searches
        subtract this object's ``gamma``, 0)."""

        def f(x: npt.NDArray, *phi: Any) -> Any:
            with np.errstate(all="ignore"):
                value = getattr(self._at(onp.asarray(phi)), name)(x)
                return onp.atleast_1d(onp.asarray(value, dtype=onp.float64))

        return f

    def _wald_covariance(self) -> npt.NDArray:
        """The scale the searches start from (``hess_inv``): the inverse
        of the likelihood's Hessian over the free parameters, the offset
        among them, taken in the search coordinates and carried to
        ``phi``; ``nan`` for a parameter on the closed edge of its space,
        or wherever the Hessian is not positive definite there (the
        offset's likelihood is not regular for a shape below 2), where
        the model's own covariance is used for the others. The searches
        go on without a scale where it is ``nan``."""
        k = len(self.params)
        out = onp.full((k, k), onp.nan)
        coords, _ = self._lr_coords()
        free = [
            j
            for j in range(k)
            if j not in self._held
            and onp.isfinite(coords[j].to_u(self.params[j]))
            and abs(coords[j].to_u(self.params[j])) < 30.0
        ]
        if free:
            u_hat = onp.array([coords[j].to_u(self.params[j]) for j in free])

            def nll_u(u: npt.NDArray) -> float:
                phi = onp.array(self.params, dtype=float)
                phi[free] = [coords[j].from_u(v) for j, v in zip(free, u)]
                return float(self._lr_full_neg_ll(phi))

            try:
                from numdifftools import Hessian  # type: ignore

                with np.errstate(all="ignore"):
                    hess = onp.asarray(Hessian(nll_u)(u_hat), dtype=float)
                    cov_u = onp.linalg.inv(hess)
                    onp.linalg.cholesky(cov_u)
                slopes = onp.array(
                    [coords[j].slope(self.params[j]) for j in free]
                )
                cov = cov_u * onp.outer(slopes, slopes)
                if onp.all(onp.isfinite(cov)):
                    out[onp.ix_(free, free)] = cov
                    return out
            except (onp.linalg.LinAlgError, ValueError, ArithmeticError):
                pass
        # The model's covariance (the distribution's parameters, lfp_p,
        # f0), without the offset's
        kept: Any = getattr(self.model, "_covariance", None)
        if kept is not None:
            cov = kept
            order = list(self._core_idx)
            for idx in (self._p_idx, self._f0_idx):
                if idx is not None:
                    order.append(idx)
            cov = onp.asarray(cov, dtype=float)
            if cov.shape == (len(order), len(order)):
                out[onp.ix_(order, order)] = cov
        for j, edge in self._closed_edges.items():
            if self.params[j] == edge:
                out[j, :] = onp.nan
                out[:, j] = onp.nan
        return out

    # -- the parameters' sides -------------------------------------------
    def _edge_deviance(self, idx: int) -> float:
        """The profile deviance with ``lfp_p`` or ``f0`` (``idx``) on its
        closed edge, kept."""
        kept = self.__dict__.setdefault("_lr_edge_dev", {})
        if idx not in kept:
            nll_hat = self._lr_neg_ll(onp.asarray(self.params, dtype=float))
            edge = self._closed_edges[idx]
            with np.errstate(all="ignore"):
                nll = self._profile_neg_ll(idx, edge)
            dev = 2.0 * (nll - nll_hat)
            # A profile a hair below the estimate's is the estimate on the
            # edge, where the fit stopped short of it
            kept[idx] = max(float(dev), 0.0) if not onp.isnan(dev) else dev
        return kept[idx]

    def _lr_param_side(self, idx: int, crit: float, direction: Any) -> float:
        """One side of the likelihood-ratio interval on ``phi[idx]``; see
        ``LikelihoodRatioMixin._lr_param_side``.

        For ``lfp_p`` and ``f0`` a side towards their closed edge (1 and
        0) is the edge wherever the profile deviance there is within
        ``crit``: the plain model is not excluded. With the estimate on
        the edge (the deviance there below ``_LR_ON_EDGE``) the other side
        is walked out in the parameter itself (``_lr_side_from_edge``)."""
        edge = self._closed_edges.get(idx)
        if edge is None:
            return super()._lr_param_side(idx, crit, direction)
        key = self._lr_key(idx, crit, direction)
        cache = self.__dict__.setdefault("_lr_sides", {})
        if key in cache:
            return cache[key]
        theta_hat = float(self.params[idx])
        d_edge = self._edge_deviance(idx)
        toward = direction * (edge - theta_hat) >= 0
        if toward and d_edge <= crit:
            cache[key] = float(edge)
            self.__dict__.setdefault("_lr_points", {})[key] = []
            return float(edge)
        if not toward and d_edge < _LR_ON_EDGE:
            value = self._lr_side_from_edge(idx, crit, direction, key)
            cache[key] = value
            return value
        return super()._lr_param_side(idx, crit, direction)

    def _lr_side_from_edge(
        self, idx: int, crit: float, direction: Any, key: tuple
    ) -> float:
        """The side ``direction`` of the interval on ``lfp_p`` or ``f0``
        (``idx``) whose estimate is on its closed edge, walked out in the
        parameter itself: steps doubling from its Wald standard error
        (``_LR_EDGE_STEP`` without one) to the first that reaches
        ``crit``, that step walked again in eight, and the crossing solved
        by ``brentq``; the far edge of the space where the deviance stays
        below ``crit`` to it (it cannot: there the data have no
        likelihood); ``nan`` where the deviance is not finite.

        In the logit, which the other parameters' walks move in, the edge
        is at infinity, and the deviance rises by about 1e-8 per unit
        there: the walk took that for a profile levelling off below the
        critical value, and the bound for the far edge."""
        coords, _ = self._lr_coords()
        coord: _LRCoord = coords[idx]
        far = coord.lo if direction < 0 else coord.hi
        theta_hat = float(self.params[idx])
        nll_hat = self._lr_neg_ll(onp.asarray(self.params, dtype=float))
        path = _LRPath()
        solved: dict[float, float] = {}

        def dev(v: float) -> float:
            v = float(v)
            if v not in solved:
                nll = self._profile_neg_ll(idx, v, path=path)
                solved[v] = 2.0 * (nll - nll_hat)
            return solved[v]

        def f(v: float) -> float:
            d = dev(v)
            return (_LR_UNREACHABLE if d == np.inf else d) - crit

        var = self.hess_inv[idx, idx]
        step = float(onp.sqrt(var)) if onp.isfinite(var) and var > 0 else 0.0
        step = step if step > 0 else _LR_EDGE_STEP
        prev = theta_hat
        value = onp.nan
        with np.errstate(all="ignore"):
            for _ in range(80):
                v = theta_hat + direction * step
                step *= 2.0
                at_far = direction * (v - far) >= 0
                if at_far:
                    v = float(far)
                if dev(v) >= crit:
                    for s in onp.linspace(prev, v, 9)[1:]:
                        if dev(s) >= crit:
                            a, b = sorted((prev, float(s)))
                            try:
                                value = float(
                                    brentq(f, a, b, xtol=1e-12, rtol=1e-10)
                                )
                            except ValueError:
                                value = onp.nan
                            break
                        if not onp.isfinite(dev(s)):
                            break
                        prev = float(s)
                    break
                if not onp.isfinite(dev(v)):
                    break
                if at_far:
                    value = float(far)
                    break
                prev = v
        self._lr_keep_points(idx, key, coord, path, crit, nll_hat)
        return value

    def _ensure_surv_data(self) -> None:
        # (the model's, which it made available)
        return None

    # -- the offset's corner ------------------------------------------------
    def _lr_corner(self, crit: float) -> bool:
        """Whether the likelihood region at ``crit`` reaches the corner of
        an offset model where the likelihood is unbounded: the offset on
        the first failure, with a base density at its origin that is not
        finite and positive (a Weibull, Gamma or LogLogistic shape below
        1 makes it infinite; the LogNormal reaches it along a path on
        which the likelihood grows without bound; Smith, 1985, Hill,
        1963). The offset's interval then runs up to the first failure
        (its profile deviance stays below ``crit`` there), and the region
        holds points of any likelihood: the extreme of a function over it
        is not defined. An Exponential's density is finite and positive
        at its origin, and its maximum on the first failure is genuine.
        """
        j = self._gamma_idx
        if j is None or j in self._held:
            return False
        hi = self._bounds[j][1]
        if hi is None:
            return False
        core = onp.asarray(self.params[self._core_idx], dtype=float)
        with np.errstate(all="ignore"):
            origin = onp.ravel(
                onp.asarray(self.dist.df(onp.array([0.0]), *core), float)
            )[0]
        if onp.isfinite(origin) and origin > 0:
            return False
        # Reached at a lower level, it is reached at this one: the region
        # grows with the level (the walk there can stop a hair short of
        # the first failure, where its searches lose the narrow valley)
        reached = self.__dict__.setdefault("_lr_corner_at", set())
        if any(c <= crit for c in reached):
            return True
        gap = float(hi) - self._lr_param_side(j, crit, 1.0)
        if gap <= 1e-6 * (float(hi) - float(self.params[j])):
            reached.add(float(crit))
            return True
        return False

    def _warn_corner(self, what: str) -> None:
        warnings.warn(
            f"The likelihood-ratio region of this offset model at this "
            f"level reaches the first failure with its offset, where the "
            f"likelihood is unbounded (the density at the offset is not "
            f"finite and positive, and a smaller shape makes it "
            f"infinite): the data cannot rule out that degenerate fit at "
            f"this level, and the extreme of {what} over the region is "
            f"not defined, so nan is returned. The interval on the offset "
            f"itself, param_cb('gamma', method='lr'), runs up to the first "
            f"failure. A larger alpha_ci, more failures, or "
            f"method='bootstrap' gives a bound.",
            RuntimeWarning,
            stacklevel=caller_stacklevel(),
        )

    def _cb_lr(self, t: Any, on: str, alpha_ci: float, bound: str) -> Any:
        check_option("bound", bound, BOUNDS)
        check_option("on", on, CB_ON)
        if self._lr_corner(critical_value(alpha_ci, bound)):
            self._warn_corner(f"{on} at each time")
            t = onp.atleast_1d(t)
            shape = (t.size, 2) if bound == "two-sided" else (t.size,)
            return onp.full(shape, onp.nan)
        return super()._cb_lr(t, on, alpha_ci, bound)

    def _summary_cb_lr(
        self,
        fns: list,
        alpha_ci: float,
        bound: str,
        what: str,
        scale: tuple | None = None,
    ) -> npt.NDArray:
        if self._lr_corner(critical_value(alpha_ci, bound)):
            self._warn_corner(what)
            n = len(fns)
            shape = (n, 2) if bound == "two-sided" else (n,)
            return onp.full(shape, onp.nan)
        return super()._summary_cb_lr(fns, alpha_ci, bound, what, scale)
