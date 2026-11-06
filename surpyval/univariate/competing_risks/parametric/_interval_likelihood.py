"""
The joint likelihood of a parametric competing-risks model whose data hold
left-censored (``c = -1``) or interval-censored (``c = 2``) rows.

With exact and right-censored rows only, the likelihood is a product of one
factor per cause, and :class:`ParametricCompetingRisks` fits each cause on
its own. A row known to have failed from cause :math:`k` somewhere in
:math:`(a, b]` contributes instead the probability of that,

.. math::
    \\mathrm{CIF}_k(b) - \\mathrm{CIF}_k(a) = \\int_a^b f_k(u)
    \\prod_{j \\neq k} S_j(u)\\,du,

which holds every cause's parameters at once, so the causes are fitted
together here: one maximum-likelihood search over all of their parameters,
with the joint inverse Hessian as the covariance.

The integral is taken as :meth:`ParametricCompetingRisks.cif` takes it, on
the cause's own probability scale :math:`p = F_k(u)`, where the integrand
is the other causes' survival, bounded by 1. The quadrature is
Gauss-Legendre on panels refined until the integral at a reference value
of the parameters holds to about :math:`10^{-12}`; the nodes are then held
fixed, so the likelihood is a smooth function of the parameters that
autograd differentiates exactly. The fit refines the nodes at its answer
and searches again until the answer no longer moves, so the reported
likelihood, gradient and Hessian are those with nodes refined at the
estimate.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from autograd import hessian, jacobian, value_and_grad
from scipy.optimize import minimize

from surpyval.univariate.competing_risks.labels import label_mask
from surpyval.univariate.information_criteria import ic_sample_size
from surpyval.univariate.parametric.fitters import is_local_minimum
from surpyval.univariate.parametric.parametric_fitter import ParametricFitter
from surpyval.utils.no_maximum import quiet_maximum_warnings, warn_unverified

# Gauss-Legendre nodes and weights on (0, 1).
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)
_GL_X = 0.5 * (_GL_X + 1.0)
_GL_W = 0.5 * _GL_W
# The first panels of a window on its probability scale; a window from
# the start of the support (p = 0) is graded towards it in half decades,
# and one reaching the far tail towards p = 1, where the derivatives of
# the integrand in the parameters (a log t in a shape's) are integrably
# singular. The panels are then halved where the integral at the
# reference parameters is not yet accurate.
_BASE = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
_TO_ZERO = 10.0 ** np.arange(-16.0, -1.0, 0.5)
_TO_ONE = 1.0 - 10.0 ** np.arange(-2.0, -9.0, -1.0)
_RTOL = 1e-12
_MAX_PANELS = 400


def _gl(lo: npt.NDArray, hi: npt.NDArray) -> tuple:
    """Gauss-Legendre nodes and weights over each panel ``(lo, hi)``,
    one row per panel."""
    width = hi - lo
    return lo[:, None] + width[:, None] * _GL_X, width[:, None] * _GL_W


def _panels(phi: Any, lo: float, hi: float) -> npt.NDArray:
    """Breakpoints on ``(lo, hi)``, graded, then halved where the
    eight-point rule on a panel and on its two halves differ by more than
    ``_RTOL`` of the whole integral of ``phi`` (a vectorised function)."""
    edges = lo + (hi - lo) * _BASE
    if lo == 0:
        edges = np.concatenate([edges, hi * _TO_ZERO])
    edges = np.concatenate([edges, _TO_ONE[(_TO_ONE > lo) & (_TO_ONE < hi)]])
    edges = np.unique(edges)
    while True:
        a, b = edges[:-1], edges[1:]
        mid = 0.5 * (a + b)
        x, w = _gl(a, b)
        whole = np.sum(w * phi(x), axis=1)
        xl, wl = _gl(a, mid)
        xr, wr = _gl(mid, b)
        halves = np.sum(wl * phi(xl), axis=1) + np.sum(wr * phi(xr), axis=1)
        total = max(abs(float(np.sum(halves))), 1e-300)
        split = np.abs(halves - whole) > _RTOL * total
        if not split.any() or edges.size > _MAX_PANELS:
            return edges
        edges = np.sort(np.concatenate([edges, mid[split]]))


@dataclass
class _Above:
    """Points ``x`` with the mask of those strictly above a lower support
    end ``lo`` and the points with the others moved inside, so a function
    and its gradient are finite at every point (the masked ones are then
    replaced)."""

    mask: npt.NDArray
    safe: npt.NDArray

    @classmethod
    def of(cls, x: npt.NDArray, lo: float) -> _Above:
        x = np.asarray(x, dtype=float)
        mask = np.isfinite(x) & (x > lo)
        inside = (lo + 1.0) if np.isfinite(lo) else 0.0
        return cls(mask=mask, safe=np.where(mask, x, inside))


def _H(dist: Any, at: _Above, p: Any) -> Any:
    """The cumulative hazard at the points of ``at``, 0 at or below the
    support's lower end."""
    if not at.mask.any():
        return anp.zeros(at.mask.shape)
    return anp.where(at.mask, dist.Hf(at.safe, *p), 0.0)


@dataclass
class _CauseRows:
    """The rows of one cause's events, grouped by kind."""

    # exact failures: count and every cause's view of the time
    n_exact: npt.NDArray
    exact_at: list
    # left- and interval-censored failures: count and window
    n_window: npt.NDArray
    a: npt.NDArray
    b: npt.NDArray
    # the quadrature (set by JointData.adapt): every cause's view of the
    # nodes, and the weights, one row per window, zero-padded
    U_at: list = field(default_factory=list)
    W: npt.NDArray = field(default_factory=lambda: np.zeros((0, 0)))


@dataclass
class JointData:
    """The data of a joint fit, prepared once: rows with the same values
    merged, every point moved inside each cause's support."""

    causes: list
    dists: list
    sizes: list
    los: list
    rows: list  # one _CauseRows per cause
    censored_at: list  # every cause's view of the right-censored times
    n_censored: npt.NDArray
    entry_at: list  # every cause's view of the entry times
    n_entry: npt.NDArray
    n_obs: float
    n_ic: float
    bounds: list = field(default_factory=list)

    def adapt(self, theta: npt.NDArray) -> None:
        """Set every window's quadrature for the parameters ``theta``."""
        params = split(np.asarray(theta, dtype=float), self.sizes)
        for i, rows in enumerate(self.rows):
            if not rows.n_window.size:
                continue
            nodes, weights = [], []
            for a, b in zip(rows.a, rows.b):
                U, W = _window_nodes(self, i, params, a, b)
                nodes.append(U)
                weights.append(W)
            M = max(u.size for u in nodes)
            U = np.zeros((len(nodes), M))
            W = np.zeros((len(nodes), M))
            for r, (u, w) in enumerate(zip(nodes, weights)):
                U[r, : u.size] = u
                U[r, u.size :] = u[-1] if u.size else 1.0
                W[r, : w.size] = w
            rows.U_at = [_Above.of(U, lo) for lo in self.los]
            rows.W = W


def _window_nodes(
    data: JointData, i: int, params: list, a: float, b: float
) -> tuple:
    """Nodes in time and weights for the integral over ``(a, b]`` of
    cause ``i``'s sub-density, on its probability scale at ``params``."""
    dist, p_i = data.dists[i], params[i]
    with np.errstate(all="ignore"):
        Pa = (
            float(np.ravel(dist.ff(np.array([a]), *p_i))[0])
            if (np.isfinite(a))
            else 0.0
        )
        Pb = float(np.ravel(dist.ff(np.array([b]), *p_i))[0])

    def time_at(p: npt.NDArray) -> npt.NDArray:
        with np.errstate(all="ignore"):
            return np.asarray(dist.qf(np.clip(p, 0.0, 1.0), *p_i), float)

    def others(u: npt.NDArray) -> npt.NDArray:
        H = np.zeros(u.shape)
        for j, d in enumerate(data.dists):
            if j != i:
                at = _Above.of(u, data.los[j])
                with np.errstate(all="ignore"):
                    H = H + np.where(
                        at.mask, np.asarray(d.Hf(at.safe, *params[j])), 0.0
                    )
        return np.exp(-H)

    if Pb - Pa > 1e-13 * max(Pb, 1e-300):
        edges = _panels(lambda p: others(time_at(p)), Pa, Pb)
        P, Wp = _gl(edges[:-1], edges[1:])
        U = time_at(P).ravel()
        with np.errstate(all="ignore"):
            f = np.asarray(dist.df(U, *p_i), dtype=float)
        ok = np.isfinite(U) & (f > 1e-300) & np.isfinite(f)
        W = np.where(ok, Wp.ravel() / np.where(ok, f, 1.0), 0.0)
        U = np.where(ok, U, b)
        return U, W
    # A window too far in the cause's tail for its probability scale to
    # resolve (F(a) and F(b) equal in double precision): in time.
    lo = a if np.isfinite(a) else b - 1.0
    U, W = _gl(np.array([lo]), np.array([b]))
    return U.ravel(), W.ravel()


def _support_lo(dist: Any) -> float:
    return float(getattr(dist, "support", (-np.inf, np.inf))[0])


def prepare(
    causes: list,
    dists: list,
    x: npt.NDArray,
    c: npt.NDArray,
    n: npt.NDArray,
    e: npt.NDArray,
    t: npt.NDArray,
) -> JointData:
    """Check and arrange the data of a joint fit (see the module
    docstring); raise a ``ValueError`` saying what to do where a row
    cannot be used. The quadrature is set by :meth:`JointData.adapt`."""
    for k, dist in zip(causes, dists):
        if not isinstance(dist, ParametricFitter):
            raise ValueError(
                f"cause {k!r}: left- or interval-censored rows are fitted "
                "jointly over every cause's parameters, which needs a "
                "SurPyval parametric distribution (Weibull, LogNormal, "
                f"...) for each cause, not {type(dist).__name__}."
            )
        if dist.discrete:
            raise ValueError(
                f"cause {k!r}: left- or interval-censored rows are "
                "supported with continuous distributions only; "
                f"{dist.name} is discrete. Use a continuous distribution, "
                "or drop those rows."
            )
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        xl = xr = x
    else:
        xl, xr = x[:, 0], x[:, 1]
    tl = np.asarray(t, dtype=float)[:, 0]
    c = np.asarray(c, dtype=float)
    n = np.asarray(n, dtype=float)
    index = {k: i for i, k in enumerate(causes)}
    ci = np.array([-1 if v is None else index[v] for v in e], dtype=float)
    window = (c == -1) | (c == 2)
    if (window & (ci == -1)).any():
        raise ValueError(
            "A left- or interval-censored row (c = -1 or 2) needs its "
            "cause in 'e': a failure in a window from an unknown cause is "
            "not supported. Give each such row its cause, or drop it."
        )

    # Rows with the same values are one row with their counts summed.
    key = np.column_stack([c, ci, xl, xr, tl])
    key = np.where(np.isfinite(key), key, np.sign(key) * 1e308)
    uniq, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = np.ravel(inverse)
    counts = np.bincount(inverse, weights=n, minlength=uniq.shape[0])
    first = np.zeros(uniq.shape[0], dtype=int)
    first[inverse[::-1]] = np.arange(inverse.size)[::-1]
    c, ci, xl, xr, tl = c[first], ci[first], xl[first], xr[first], tl[first]
    n = counts
    window = (c == -1) | (c == 2)

    los = [_support_lo(d) for d in dists]

    def views(points: npt.NDArray) -> list:
        return [_Above.of(points, lo) for lo in los]

    rows = []
    for i in range(len(causes)):
        mine = ci == i
        ex = mine & (c == 0)
        win = mine & window
        rows.append(
            _CauseRows(
                n_exact=n[ex],
                exact_at=views(xl[ex]),
                n_window=n[win],
                a=np.where(c[win] == -1, tl[win], xl[win]),
                b=xr[win],
            )
        )
    cens = c == 1
    entry = np.isfinite(tl)
    bounds: list = []
    for d in dists:
        bounds += list(d.bounds)
    return JointData(
        causes=list(causes),
        dists=list(dists),
        sizes=[d.k for d in dists],
        los=los,
        rows=rows,
        censored_at=views(xl[cens]),
        n_censored=n[cens],
        entry_at=views(tl[entry]),
        n_entry=n[entry],
        n_obs=float(n.sum()),
        n_ic=float(ic_sample_size(c, n)),
        bounds=bounds,
    )


def split(theta: Any, sizes: list) -> list:
    """``theta`` cut into each cause's parameters."""
    out, start = [], 0
    for size in sizes:
        out.append([theta[start + j] for j in range(size)])
        start += size
    return out


def neg_ll(theta: Any, data: JointData) -> Any:
    """The joint negative log-likelihood at ``theta``, every cause's
    parameters in turn (autograd differentiates it), with the quadrature
    last set by :meth:`JointData.adapt`."""
    params = split(theta, data.sizes)
    dists = data.dists
    K = len(dists)
    ll = 0.0
    for j in range(K):
        # Right-censored rows: -H(x), every cause's cumulative hazard;
        # delayed entry: + H(tl), every row conditioned on reaching tl.
        ll = ll - anp.sum(
            data.n_censored * _H(dists[j], data.censored_at[j], params[j])
        )
        ll = ll + anp.sum(
            data.n_entry * _H(dists[j], data.entry_at[j], params[j])
        )
    for i, rows in enumerate(data.rows):
        if rows.n_exact.size:
            # h_i(x) S(x) = f_i(x) prod_{j != i} S_j(x)
            ll = ll + anp.sum(
                rows.n_exact
                * dists[i].log_df(rows.exact_at[i].safe, *params[i])
            )
            for j in range(K):
                if j != i:
                    ll = ll - anp.sum(
                        rows.n_exact
                        * _H(dists[j], rows.exact_at[j], params[j])
                    )
        if rows.n_window.size:
            ll = ll + anp.sum(
                rows.n_window
                * anp.log(window_probability(i, rows, data, params))
            )
    return -ll


def window_probability(
    i: int, rows: _CauseRows, data: JointData, params: list
) -> Any:
    """Each window's probability of a failure from cause ``i`` in it,
    :math:`\\int_a^b f_i(u) \\prod_{j \\neq i} S_j(u)\\,du`."""
    dists = data.dists
    U_i = rows.U_at[i]
    f = anp.where(U_i.mask, dists[i].df(U_i.safe, *params[i]), 0.0)
    H_o = 0.0
    for j in range(len(dists)):
        if j != i:
            H_o = H_o + _H(dists[j], rows.U_at[j], params[j])
    return anp.sum(rows.W * f * anp.exp(-H_o), axis=1)


# -- the fit ---------------------------------------------------------------


def _to_natural(bounds: list) -> Any:
    """The map from the unbounded search space to the parameters: log of
    the distance from a one-sided bound, logit between two."""

    def natural(u: Any) -> Any:
        out = []
        for j, (lo, hi) in enumerate(bounds):
            v = u[j]
            if lo is not None and hi is not None:
                out.append(lo + (hi - lo) / (1.0 + anp.exp(-v)))
            elif lo is not None:
                out.append(lo + anp.exp(v))
            elif hi is not None:
                out.append(hi - anp.exp(v))
            else:
                out.append(v)
        return anp.array(out)

    return natural


def _to_search(theta: npt.NDArray, bounds: list) -> npt.NDArray:
    out = []
    for v, (lo, hi) in zip(theta, bounds):
        if lo is not None and hi is not None:
            q = (v - lo) / (hi - lo)
            out.append(np.log(q) - np.log1p(-q))
        elif lo is not None:
            out.append(np.log(v - lo))
        elif hi is not None:
            out.append(np.log(hi - v))
        else:
            out.append(v)
    return np.array(out, dtype=float)


@dataclass
class JointFit:
    """What :func:`fit` found: the parameters, each cause's, the joint
    covariance (``None`` where the Hessian is not positive definite), the
    negative log-likelihood and whether it is a verified maximum."""

    theta: npt.NDArray
    params: list
    covariance: npt.NDArray | None
    neg_ll: float
    maximum: str
    k: int
    n_ic: float


def fit(data: JointData, init: npt.NDArray) -> JointFit:
    """Maximise the joint likelihood from ``init`` (the causes' own
    approximate fits): BFGS in the unbounded space, then Newton steps,
    with the quadrature refined at each answer until the answer no longer
    moves; checked to be a verified maximum (zero gradient,
    positive-definite Hessian), as every SurPyval maximum-likelihood fit
    is."""
    natural = _to_natural(data.bounds)
    scale = max(data.n_obs, 1.0)

    def objective(u: Any) -> Any:
        return neg_ll(natural(u), data) / scale

    vg = value_and_grad(objective)
    hess_u = hessian(objective)

    def safe_vg(u: npt.NDArray) -> tuple:
        with np.errstate(all="ignore"):
            val, g = vg(u)
        if not np.isfinite(val) or not np.all(np.isfinite(g)):
            return np.inf, np.zeros_like(u)
        return float(val), np.asarray(g, dtype=float)

    def newton(u: npt.NDArray) -> npt.NDArray:
        # Newton steps, halved until the objective falls.
        for _ in range(20):
            f0, g = safe_vg(u)
            with np.errstate(all="ignore"):
                h = np.asarray(hess_u(u), dtype=float)
            try:
                step = np.linalg.solve(h, g)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(step)):
                break
            # A step is taken where the objective falls or, as near the
            # optimum it stops falling measurably, where its gradient does.
            t = 1.0
            while t > 1e-6:
                f1, g1 = safe_vg(u - t * step)
                if f1 < f0 or (
                    f1 <= f0 + 1e-12 * abs(f0)
                    and np.max(np.abs(g1)) < np.max(np.abs(g))
                ):
                    break
                t /= 2
            if t <= 1e-6:
                break
            u = u - t * step
            if np.max(np.abs(t * step)) < 1e-12:
                break
        return u

    theta = np.asarray(init, dtype=float)
    u = _to_search(theta, data.bounds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for round_ in range(6):
            data.adapt(theta)
            if round_ == 0:
                res = minimize(
                    safe_vg, u, jac=True, method="BFGS", options={"gtol": 1e-9}
                )
                if np.isfinite(res.fun):
                    u = np.asarray(res.x, dtype=float)
            u_new = newton(u)
            moved = np.max(np.abs(u_new - u))
            u = u_new
            theta = np.asarray(natural(u), dtype=float)
            if round_ > 0 and moved < 1e-9:
                break
    theta = np.asarray(natural(u), dtype=float)

    def nll_natural(v: Any) -> Any:
        return neg_ll(v, data)

    jac_nat = jacobian(nll_natural)
    hess_nat = hessian(nll_natural)
    with np.errstate(all="ignore"):
        value = float(nll_natural(theta))
        verified = np.isfinite(value) and is_local_minimum(
            nll_natural,
            jac_nat,
            hess_nat,
            theta,
            obj_scale=data.n_obs,
        )
        try:
            H = np.asarray(hess_nat(theta), dtype=float)
            np.linalg.cholesky(H)
            cov: npt.NDArray | None = np.linalg.inv(H)
        except (np.linalg.LinAlgError, ValueError):
            cov = None
    if not verified:
        warn_unverified(
            "The joint competing-risks fit (left- or interval-censored "
            "rows)",
            advice="check that the data hold enough failures of each "
            "cause to estimate its parameters",
        )
    return JointFit(
        theta=theta,
        params=[np.asarray(p, dtype=float) for p in split(theta, data.sizes)],
        covariance=cov,
        neg_ll=value,
        maximum="verified" if verified else "unverified",
        k=int(theta.size),
        n_ic=data.n_ic,
    )


def start(
    causes: list,
    dists: list,
    x: npt.NDArray,
    c: npt.NDArray,
    n: npt.NDArray,
    e: npt.NDArray,
    t: npt.NDArray,
) -> npt.NDArray:
    """Starting values: each cause fitted on its own, its own windows kept
    as they are, another cause's window right censored at its lower end
    (a left-censored row of another cause, which says only that the unit
    reached its entry, left out), as if the causes did not compete inside
    the windows."""
    x = np.asarray(x, dtype=float)
    X = np.column_stack([x, x]) if x.ndim == 1 else x
    c = np.asarray(c)
    out = []
    for k, dist in zip(causes, dists):
        mine = label_mask(e, k)
        c_k = np.where(mine & (c != 1), c, 1).astype(int)
        X_k = X.copy()
        other = ~mine & ((c == 2) | (c == -1))
        X_k[other & (c == 2), 1] = X_k[other & (c == 2), 0]
        # (one starting at the unit's entry says nothing either)
        keep = ~(other & ((c == -1) | (X[:, 0] <= t[:, 0])))
        with warnings.catch_warnings(), quiet_maximum_warnings():
            warnings.simplefilter("ignore")
            model = dist.fit(
                x=X_k[keep], c=c_k[keep], n=np.asarray(n)[keep], t=t[keep]
            )
        out.append(np.ravel(np.asarray(model.params, dtype=float)))
    return np.concatenate(out)
