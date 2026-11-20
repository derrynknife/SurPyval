"""The log-posterior and its gradient at one point, fast, for a sampler.

:func:`surpyval.bayes.likelihood.log_posterior_and_grad` builds it. The
public :class:`~surpyval.bayes.likelihood.LogLikelihood` checks and
reshapes on every call and takes many parameter points at once, which is
right for the grid and wrong for a gradient-based sampler, which asks for
one point and its gradient hundreds of thousands of times. Here the data
are checked and split once (by censoring kind and, for the windows, by
which ends are present), and each call is one ``autograd`` pass over the
same terms, built from each distribution's own ``log_df``, ``log_sf``,
``log_ff``, ``ff`` and ``sf``.

What a sampler may assume
-------------------------
- The argument ``z`` is the free parameters on their unconstrained scale
  (``BoundPriors.free_names`` order): the scale ``fit_bayes`` and its
  priors use, so the value is a log density with respect to ``z`` (the
  log-Jacobian of the transform is inside each prior's density on that
  scale; a flat prior is flat in ``z``).
- The value equals ``bound.log_prior(z) + LogLikelihood(...)(theta(z))``,
  the posterior the grid uses, to rounding; ``nan`` there is ``-inf``
  here.
- Where the value is ``-inf`` the gradient is all zeros (a sampler
  rejects the point on its value). Where the value is finite, so is the
  gradient.
- No randomness, no state: the same ``z`` gives the same answer.
- The gradient is exact (automatic differentiation) except for a prior
  factor whose quantity has no differentiable form (an ``on_quantities``
  mean through a special function); that factor's gradient is by central
  differences, said in :attr:`LogPosterior.numeric_factors`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from autograd import grad, value_and_grad
from autograd.extend import defvjp, primitive
from autograd.tracer import getval

from surpyval.bayes.likelihood import model_parameter_names
from surpyval.bayes.transforms import Transform
from surpyval.utils.surpyval_data import SurpyvalData

_LOG_2 = float(np.log(2.0))
_LOG_SQRT_2PI = float(np.log(np.sqrt(2 * np.pi)))
#: Errors autograd raises where a function cannot be traced (a SciPy
#: special function, an in-place write), which send a prior factor to
#: central differences.
_UNTRACEABLE = (
    TypeError,
    ValueError,
    AttributeError,
    IndexError,
    NotImplementedError,
)


def _log1mexp(d: Any) -> Any:
    """``log(1 - exp(-d))`` for ``d >= 0``, as the public likelihood has
    it (the branch chosen per element on the value, so an unused branch
    never puts a NaN into the gradient)."""
    small = getval(d) < _LOG_2
    safe_small = anp.where(small, d, 1.0)
    safe_big = anp.where(small, 1.0, d)
    return anp.where(
        small,
        anp.log(-anp.expm1(-safe_small)),
        anp.log1p(-anp.exp(-safe_big)),
    )


def _sum(a: Any, n: npt.NDArray) -> Any:
    return anp.sum(a * n)


class _Windows:
    """Rows ``(xl, xr]`` with counts: interval-censored rows, or the
    truncation windows."""

    def __init__(self, xl: npt.NDArray, xr: npt.NDArray, n: npt.NDArray):
        self.xl = np.asarray(xl, dtype=float)
        self.xr = np.asarray(xr, dtype=float)
        self.n = np.asarray(n, dtype=float)
        self.lo_finite = np.isfinite(self.xl)
        self.hi_finite = np.isfinite(self.xr)
        present = np.concatenate(
            [self.xl[self.lo_finite], self.xr[self.hi_finite]]
        )
        self.present_max = float(present.max()) if present.size else None
        self.empty = len(self.n) == 0


class LeanLikelihood:
    """The log-likelihood at one parameter point, traceable by autograd.

    Called with the parameters (a sequence of scalars, in
    ``model_parameter_names`` order), it returns the same number as
    :class:`~surpyval.bayes.likelihood.LogLikelihood` at that point, with
    no checks. ``nan`` is returned as it arises (the caller maps it).
    """

    def __init__(
        self,
        dist: Any,
        data: SurpyvalData,
        offset: bool = False,
        lfp: bool = False,
        zi: bool = False,
    ) -> None:
        self.dist = dist
        self.offset, self.lfp, self.zi = bool(offset), bool(lfp), bool(zi)
        self.names = model_parameter_names(dist, offset, lfp, zi)
        self.k = len(self.names)
        lo = float(dist.support[0])
        self._lo = lo if np.isfinite(lo) else -np.inf
        d = data
        x_o = np.asarray(d.x_o, dtype=float)
        n_o = np.asarray(d.n_o, dtype=float)
        self._n0 = 0.0
        if zi:
            zero = x_o == 0
            self._n0 = float(np.sum(n_o[zero]))
            x_o, n_o = x_o[~zero], n_o[~zero]
        self._x_o, self._n_o = x_o, n_o
        self._N_o = float(n_o.sum())
        self._x_r = np.asarray(d.x_r, dtype=float)
        self._n_r = np.asarray(d.n_r, dtype=float)
        self._x_l = np.asarray(d.x_l, dtype=float)
        self._n_l = np.asarray(d.n_l, dtype=float)
        if not self.offset and np.isfinite(self._lo):
            # Fixed for every call without an offset.
            self._x_r = np.maximum(self._x_r, self._lo)
            self._x_l = np.maximum(self._x_l, self._lo)
        self._interval = _Windows(d.x_il, d.x_ir, d.n_i)
        self._trunc = _Windows(d.tl_unique, d.tr_unique, d.n_t_unique)
        self._has_o = len(x_o) > 0
        self._has_r = len(self._x_r) > 0
        self._has_l = len(self._x_l) > 0
        self._static: dict[int, tuple] = {}
        if not self.offset:
            for w in (self._interval, self._trunc):
                self._static[id(w)] = self._masks(w, 0.0)

    # -- the pieces -----------------------------------------------------
    def _split(self, P: Any) -> tuple:
        k_dist = self.dist.k
        j = 0
        gamma: Any = 0.0
        if self.offset:
            gamma = P[0]
            j = 1
        dp = list(P[j : j + k_dist])
        j += k_dist
        p: Any = 1.0
        if self.lfp:
            p = P[j]
            j += 1
        f0: Any = P[j] if self.zi else 0.0
        return dp, gamma, p, f0

    def _censored(self, x: npt.NDArray, gamma: Any) -> Any:
        if not self.offset:
            return x
        xs = x - gamma
        if np.isfinite(self._lo):
            xs = anp.maximum(xs, self._lo)
        return xs

    def _observed(self, dp: list, gamma: Any, p: Any, f0: Any) -> Any:
        out: Any = 0.0
        if self.zi and self._n0:
            out = self._n0 * anp.log(f0)
        if not self._has_o:
            return out
        x = self._x_o - gamma if self.offset else self._x_o
        out = out + _sum(self.dist.log_df(x, *dp), self._n_o)
        if self.lfp or self.zi:
            out = out + self._N_o * anp.log(p - f0)
        return out

    def _right(self, dp: list, gamma: Any, p: Any, f0: Any) -> Any:
        if not self._has_r:
            return 0.0
        xs = self._censored(self._x_r, gamma)
        if self.lfp:
            terms = anp.log((1 - p) + (p - f0) * self.dist.sf(xs, *dp))
        elif self.zi:
            terms = anp.log1p(-f0) + self.dist.log_sf(xs, *dp)
        else:
            terms = self.dist.log_sf(xs, *dp)
        return _sum(terms, self._n_r)

    def _left(self, dp: list, gamma: Any, p: Any, f0: Any) -> Any:
        if not self._has_l:
            return 0.0
        xs = self._censored(self._x_l, gamma)
        if self.zi:
            terms = anp.log(f0 + (p - f0) * self.dist.ff(xs, *dp))
        elif self.lfp:
            terms = self.dist.log_ff(xs, *dp) + anp.log(p)
        else:
            terms = self.dist.log_ff(xs, *dp)
        return _sum(terms, self._n_l)

    def _masks(self, w: _Windows, g: float) -> tuple:
        """Which rows have a lower end inside the support (``lo_eval``),
        and the four groups of rows by which ends are evaluated."""
        with np.errstate(invalid="ignore"):
            lo_eval = w.lo_finite & ((w.xl - g) > self._lo)
        hi = w.hi_finite
        return (
            lo_eval,
            np.flatnonzero(lo_eval & hi),
            np.flatnonzero(lo_eval & ~hi),
            np.flatnonzero(~lo_eval & hi),
        )

    def _windows(
        self, w: _Windows, dp: list, gamma: Any, p: Any, f0: Any
    ) -> Any:
        """``sum n log P(xl < T <= xr)``, term for term as the public
        likelihood forms it."""
        if w.empty:
            return 0.0
        if self.offset:
            masks = self._masks(w, float(getval(gamma)))
        else:
            masks = self._static[id(w)]
        if self.lfp or self.zi:
            return self._windows_mixture(w, masks[0], dp, gamma, p, f0)
        dist = self.dist
        _, both, lower_only, upper_only = masks
        out: Any = 0.0
        if both.size:
            n = w.n[both]
            ls = w.xl[both] - gamma
            rs = w.xr[both] - gamma
            log_sl = dist.log_sf(ls, *dp)
            tail = getval(log_sl) < np.log(0.5)
            t_idx, b_idx = np.flatnonzero(tail), np.flatnonzero(~tail)
            if t_idx.size:
                lsl = log_sl[t_idx] if b_idx.size else log_sl
                log_sr = dist.log_sf(rs[t_idx], *dp)
                term = lsl + _log1mexp(anp.maximum(lsl - log_sr, 0.0))
                out = out + _sum(term, n[t_idx])
            if b_idx.size:
                log_fl = dist.log_ff(ls[b_idx], *dp)
                log_fr = dist.log_ff(rs[b_idx], *dp)
                term = log_fr + _log1mexp(anp.maximum(log_fr - log_fl, 0.0))
                out = out + _sum(term, n[b_idx])
        if lower_only.size:
            # no upper end: P(T > xl)
            n = w.n[lower_only]
            ls = w.xl[lower_only] - gamma
            log_sl = dist.log_sf(ls, *dp)
            tail = getval(log_sl) < np.log(0.5)
            t_idx, b_idx = np.flatnonzero(tail), np.flatnonzero(~tail)
            if t_idx.size:
                term = log_sl[t_idx] if b_idx.size else log_sl
                out = out + _sum(term, n[t_idx])
            if b_idx.size:
                log_fl = dist.log_ff(ls[b_idx], *dp)
                term = _log1mexp(anp.maximum(-log_fl, 0.0))
                out = out + _sum(term, n[b_idx])
        if upper_only.size:
            # no lower end inside the support: P(T <= xr)
            rs = w.xr[upper_only] - gamma
            out = out + _sum(dist.log_ff(rs, *dp), w.n[upper_only])
        return out

    def _windows_mixture(
        self,
        w: _Windows,
        lo_eval: npt.NDArray,
        dp: list,
        gamma: Any,
        p: Any,
        f0: Any,
    ) -> Any:
        """The windows of a limited-failure or zero-inflated model:
        ``log(F_mix(xr) - F_mix(xl))``."""
        dist = self.dist
        # stand-ins inside the support where an end is absent; their
        # values are masked out (and finite, so their gradients are too)
        if w.present_max is not None:
            inside: Any = w.present_max - gamma
        else:
            lo, hi = self._lo, float(dist.support[1])
            if np.isfinite(lo) and np.isfinite(hi):
                inside = 0.5 * (lo + hi)
            elif np.isfinite(lo):
                inside = lo + 1.0
            else:
                inside = 0.0
        l_safe = anp.where(lo_eval, w.xl - gamma, inside)
        r_safe = anp.where(w.hi_finite, w.xr - gamma, inside)
        upper = anp.where(
            w.hi_finite,
            f0 * (w.xr >= 0) + (p - f0) * dist.ff(r_safe, *dp),
            1.0,
        )
        lower = anp.where(
            w.lo_finite,
            f0 * (w.xl > 0)
            + (p - f0) * anp.where(lo_eval, dist.ff(l_safe, *dp), 0.0),
            0.0,
        )
        return _sum(anp.log(anp.maximum(upper - lower, 0.0)), w.n)

    def __call__(self, params: Any) -> Any:
        dp, gamma, p, f0 = self._split(params)
        return (
            self._observed(dp, gamma, p, f0)
            + self._right(dp, gamma, p, f0)
            + self._left(dp, gamma, p, f0)
            + self._windows(self._interval, dp, gamma, p, f0)
            - self._windows(self._trunc, dp, gamma, p, f0)
        )


# -- the transforms, traceable ------------------------------------------
def _to_theta(t: Transform, z: Any) -> Any:
    if t.kind == "identity":
        return z
    if t.kind == "log":
        return t.lower + anp.exp(z)
    if t.kind == "neglog":
        return t.upper - anp.exp(z)
    u = 0.5 * (1.0 + anp.tanh(0.5 * z))
    return t.lower + (t.upper - t.lower) * u


def _to_phi(t: Transform, theta: Any) -> Any:
    if t.kind == "identity":
        return theta
    if t.kind == "log":
        return anp.log(theta - t.lower)
    if t.kind == "neglog":
        return anp.log(t.upper - theta)
    u = (theta - t.lower) / (t.upper - t.lower)
    return anp.log(u) - anp.log1p(-u)


def _log_jacobian(t: Transform, z: Any) -> Any:
    if t.kind == "identity":
        return 0.0
    if t.kind in ("log", "neglog"):
        return z
    return (
        np.log(t.upper - t.lower)
        - anp.logaddexp(0.0, -z)
        - anp.logaddexp(0.0, z)
    )


# -- prior factors ------------------------------------------------------
def _exact_with_gradient(
    value: Callable[[npt.NDArray], float],
    gradient: Callable[[npt.NDArray], npt.NDArray],
) -> Callable[[Any], Any]:
    """A traceable function whose value is ``value(z)`` (the public
    path's own number) and whose gradient is ``gradient(z)``."""

    @primitive
    def f(z: npt.NDArray) -> float:
        return value(z)

    defvjp(f, lambda ans, z: lambda g: g * gradient(z))
    return f


def _central_gradient(
    fn: Callable[[npt.NDArray], npt.NDArray], h: float = 1e-4
) -> Callable[[npt.NDArray], npt.NDArray]:
    """The gradient of a vectorised ``fn`` (``(N, d) -> (N,)``) by central
    differences, all ``2 d`` points in one call."""

    def gradient(z: npt.NDArray) -> npt.NDArray:
        d = z.shape[0]
        steps = np.eye(d) * h
        pts = np.concatenate([z + steps, z - steps])
        vals = np.asarray(fn(pts), dtype=float)
        return (vals[:d] - vals[d:]) / (2 * h)

    return gradient


def _single_log_density(p: Any, t: Transform) -> Callable[[Any], Any] | None:
    """A one-parameter prior's log density on ``phi``, traceable; None for
    a prior with none to trace (it is given by value and gradient)."""
    from surpyval.bayes.prior import Flat, NormalOnScale

    if isinstance(p, Flat):
        return lambda phi: 0.0
    if isinstance(p, NormalOnScale):
        m, s = p._resolved(t)
        log_s = float(np.log(s))

        def normal(phi: Any) -> Any:
            r = (phi - m) / s
            return -(r**2) / 2.0 - _LOG_SQRT_2PI - log_s

        return normal
    return None


def _distribution_prior_gradient(p: Any, t: Transform) -> Callable:
    """The derivative in ``phi`` of a ``DistributionPrior``'s log density:
    the model's ``log_df`` at ``theta - gamma`` plus the log-Jacobian
    (the constants, ``log(p - f0)`` and a truncation's mass, have none)."""
    model = p.model
    params = [float(v) for v in np.atleast_1d(model.params)]
    gamma = float(getattr(model, "gamma", 0.0) or 0.0)

    def log_density(phi: Any) -> Any:
        theta = _to_theta(t, phi)
        return model.dist.log_df(theta - gamma, *params) + _log_jacobian(
            t, phi
        )

    return log_density


def _theta_and_slope(t: Transform, z: float) -> tuple[float, float]:
    """``theta`` at ``z`` and ``d theta / d z``, in plain floats (the same
    formulas as :meth:`Transform.to_theta`)."""
    if t.kind == "identity":
        return z, 1.0
    if t.kind == "log":
        e = float(np.exp(z))
        return t.lower + e, e
    if t.kind == "neglog":
        e = float(np.exp(z))
        return t.upper - e, -e
    u = 0.5 * (1.0 + float(np.tanh(0.5 * z)))
    width = t.upper - t.lower
    return t.lower + width * u, width * u * (1.0 - u)


Term = Callable[[npt.NDArray], tuple[Any, npt.NDArray]]


class LogPosterior:
    """``f(z) -> (value, gradient)``: the log-posterior of a Bayesian fit
    and its gradient at one point ``z`` of the free parameters'
    unconstrained scale. Built by
    :func:`surpyval.bayes.likelihood.log_posterior_and_grad`; see
    :mod:`surpyval.bayes._lean` for what a sampler may assume.

    Attributes
    ----------
    bound : BoundPriors
        The priors bound to the fit (names, transforms, fixed values).
    free_names : list of str
        The order of ``z``.
    hand_written : bool
        Whether the likelihood's gradient is the hand-written one (the
        Weibull, Exponential, Normal, LogNormal and Gamma without an
        offset, limited-failure or zero-inflated form) rather than
        autograd's.
    numeric_factors : list of str
        The prior factors whose gradient is by central differences (none
        in the usual case).
    """

    def __init__(self, dist: Any, data: SurpyvalData | None, bound: Any):
        self.dist = dist
        self.bound = bound
        self.free_names = list(bound.free_names)
        self.d = bound.d
        self.numeric_factors: list[str] = []
        self.hand_written = False
        self._free = list(bound.free_idx)
        self._ts = list(bound.transforms)
        self._fixed = {
            bound.names.index(k): float(v) for k, v in bound.fixed.items()
        }
        self._terms: list[Term] = []
        if data is not None:
            term, fast = self._likelihood_term(data, 1.0)
            self.hand_written = fast
            self._terms.append(term)
        self._build_prior(bound)

    # -- building -------------------------------------------------------
    def theta(self, z: Any) -> list:
        """Every parameter (fixed ones too) at ``z``, as a list (traceable
        by autograd)."""
        out: list = [None] * len(self._ts)
        for j, i in enumerate(self._free):
            out[i] = _to_theta(self._ts[i], z[j])
        for i, v in self._fixed.items():
            out[i] = v
        return out

    def _theta_numeric(self, z: npt.NDArray) -> tuple[list, npt.NDArray]:
        theta: list = [0.0] * len(self._ts)
        slope = np.empty(self.d)
        for j, i in enumerate(self._free):
            theta[i], slope[j] = _theta_and_slope(self._ts[i], float(z[j]))
        for i, v in self._fixed.items():
            theta[i] = v
        return theta, slope

    def _likelihood_term(
        self, data: SurpyvalData, weight: float
    ) -> tuple[Term, bool]:
        from surpyval.bayes._lean_fast import FastLikelihood

        b = self.bound
        fast = None
        if not (b.offset or b.lfp or b.zi):
            fast = FastLikelihood.build(self.dist, data)
        if fast is not None:
            free = self._free

            def hand(z: npt.NDArray) -> tuple[Any, npt.NDArray]:
                theta, slope = self._theta_numeric(z)
                v, g = fast(theta)
                return weight * v, weight * g[free] * slope

            return hand, True
        ll = LeanLikelihood(self.dist, data, b.offset, b.lfp, b.zi)
        return value_and_grad(lambda z: weight * ll(self.theta(z))), False

    def _build_prior(self, bound: Any) -> None:
        from surpyval.bayes.prior import (
            DistributionPrior,
            Flat,
            MultivariateNormal,
            NormalOnScale,
            OnQuantities,
            PowerPrior,
        )

        for k, p in bound.spec.single.items():
            if p.is_fixed or isinstance(p, Flat):
                continue
            j = bound.free_names.index(k)
            t = bound.transforms[bound.names.index(k)]
            if isinstance(p, NormalOnScale):
                self._terms.append(self._normal(p, t, j))
            elif isinstance(p, DistributionPrior):
                twin = _distribution_prior_gradient(p, t)
                factor = self._by_value(
                    f"{k}: {p.said}",
                    lambda Z, p=p, t=t, j=j: p.log_density_phi(Z[:, j], t),
                    lambda z, twin=twin, j=j: twin(z[j]),
                )
                self._terms.append(value_and_grad(factor))
            else:  # pragma: no cover - every prior kind above
                raise ValueError(
                    f"No gradient for the prior on {k} ({type(p).__name__})."
                )
        for jp in bound.spec.joint:
            if isinstance(jp, MultivariateNormal):
                self._terms.append(self._mvn(jp, bound))
            elif isinstance(jp, PowerPrior):
                term, _ = self._likelihood_term(jp.data, jp.weight)
                self._terms.append(term)
            elif isinstance(jp, OnQuantities):
                self._terms.append(
                    value_and_grad(self._on_quantities(jp, bound))
                )
            else:  # pragma: no cover - every joint prior kind above
                raise ValueError(
                    f"No gradient for the joint prior {type(jp).__name__}."
                )

    def _normal(self, p: Any, t: Transform, j: int) -> Term:
        m, s = p._resolved(t)
        log_s = float(np.log(s))
        d = self.d

        def normal(z: npt.NDArray) -> tuple[Any, npt.NDArray]:
            r = (z[j] - m) / s
            g = np.zeros(d)
            g[j] = -r / s
            return -(r**2) / 2.0 - _LOG_SQRT_2PI - log_s, g

        return normal

    def _by_value(
        self, label: str, value_fn: Callable, twin: Callable | None
    ) -> Callable:
        """A factor whose value is the public path's own (``value_fn`` on an
        ``(N, d)`` array) and whose gradient is ``twin``'s by autograd, or
        by central differences where ``twin`` cannot be traced."""

        def vec(Z: npt.NDArray) -> npt.NDArray:
            with np.errstate(all="ignore"):
                out = np.asarray(value_fn(Z), dtype=float)
            return np.where(np.isnan(out), -np.inf, out)

        gradient: Callable | None = None
        if twin is not None:
            g = grad(twin)
            probe = np.zeros(self.d)
            try:
                with np.errstate(all="ignore"):
                    g(probe)
                gradient = g
            except _UNTRACEABLE:
                gradient = None
        if gradient is None:
            self.numeric_factors.append(label)
            gradient = _central_gradient(vec)
        return _exact_with_gradient(
            lambda z: float(vec(z[None, :])[0]), gradient
        )

    def _mvn(self, jp: Any, bound: Any) -> Term:
        idx = bound.indices(jp.names, "MultivariateNormal")
        mean = jp.mean
        cov = 0.5 * (jp.cov + jp.cov.T)
        chol = np.linalg.cholesky(cov)
        inv_chol = np.linalg.inv(chol)
        precision = inv_chol.T @ inv_chol
        const = -0.5 * (
            len(idx) * np.log(2 * np.pi) + 2 * np.sum(np.log(np.diag(chol)))
        )
        d = self.d

        def mvn(z: npt.NDArray) -> tuple[Any, npt.NDArray]:
            r = z[idx] - mean
            u = inv_chol @ r
            g = np.zeros(d)
            g[idx] = -(precision @ r)
            return const - 0.5 * float(u @ u), g

        return mvn

    def _on_quantities(self, jp: Any, bound: Any) -> Callable:
        """The value is the public prior's (its Jacobian by central
        differences, step 1e-5, as the grid has it); the gradient is that
        same expression's, by autograd through each quantity."""
        log_density = jp.bind(bound)
        ts = [jp.transform_of(spec, bound) for _, spec, _ in jp.statements]
        dens = [
            _single_log_density(p, t)
            for (_, _, p), t in zip(jp.statements, ts)
        ]
        twin: Callable | None = None
        if all(d is not None for d in dens):

            def q_phi(z: Any) -> list:
                theta = dict(zip(bound.names, self.theta(z)))
                return [
                    _to_phi(t, _quantity(jp.dist, spec, theta))
                    for (_, spec, _), t in zip(jp.statements, ts)
                ]

            def twin_fn(z: Any) -> Any:
                psi = q_phi(z)
                out = sum(f(v) for f, v in zip(dens, psi))  # type: ignore
                h = 1e-5
                k = len(z)
                cols = []
                for i in range(k):
                    step = np.zeros(k)
                    step[i] = h
                    up, dn = q_phi(z + step), q_phi(z - step)
                    cols.append([(a - b) / (2 * h) for a, b in zip(up, dn)])
                return out + _log_abs_det(cols)

            twin = twin_fn
        return self._by_value("on_quantities: " + jp.said, log_density, twin)

    # -- evaluating -----------------------------------------------------
    def __call__(self, z: npt.ArrayLike) -> tuple[float, npt.NDArray]:
        z = np.asarray(z, dtype=float)
        if z.shape != (self.d,):
            raise ValueError(
                f"z must hold {self.d} values ({', '.join(self.free_names)}"
                f", on the unconstrained scale); got shape {z.shape}."
            )
        value = 0.0
        gradient = np.zeros(self.d)
        with np.errstate(all="ignore"):
            for term in self._terms:
                v, g = term(z)
                value += float(v)
                gradient += np.asarray(g, dtype=float)
        if np.isnan(value) or value == -np.inf:
            return -np.inf, np.zeros(self.d)
        return value, gradient

    def value(self, z: npt.ArrayLike) -> float:
        """The value alone (the same number ``__call__`` returns)."""
        return self(z)[0]


def _log_abs_det(J: list) -> Any:
    """``log |det J|`` for a small square ``J`` given as nested lists of
    traceable scalars (by cofactors: one to three parameters)."""
    k = len(J)
    if k == 1:
        det = J[0][0]
    elif k == 2:
        det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
    elif k == 3:
        det = (
            J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
            - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
            + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0])
        )
    else:
        det = anp.linalg.det(anp.array(J))
    return anp.log(anp.abs(det))


def _quantity(dist: Any, spec: tuple, theta: dict) -> Any:
    """``quantity_at`` at one point, traceable (its gradient is what is
    used; the value comes from ``quantity_at`` itself)."""
    dp = [theta[n] for n in dist.parameter_names]
    gamma = theta.get("gamma", 0.0)
    lfp = theta.get("lfp_p", 1.0)
    f0 = theta.get("f0", 0.0)
    kind = spec[0]
    if kind == "param":
        return theta[spec[1]]
    if kind == "q":
        u = (spec[1] - f0) / (lfp - f0)
        return gamma + dist.qf(u, *dp)
    s0 = getattr(dist, "support", (-np.inf, np.inf))[0]
    below = float(spec[1] - getval(gamma)) < s0
    if kind == "R":
        sf = 1.0 if below else dist.sf(spec[1] - gamma, *dp)
        return 1 - lfp + (lfp - f0) * sf
    if kind == "F":
        ff = 0.0 if below else dist.ff(spec[1] - gamma, *dp)
        return f0 + (lfp - f0) * ff
    if kind == "mean":
        return gamma + lfp * dist.mean(*dp)
    raise ValueError(f"Unknown quantity {spec!r}.")
