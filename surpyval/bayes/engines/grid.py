"""The adaptive grid: the posterior of one to three free parameters
integrated on a fine grid, with a check of its own accuracy.

1. Find the posterior's peak and curvature on the unconstrained scale.
2. Lay a grid over the peak, along the curvature's axes, and widen it
   until the density on every face is below ``1e-8`` of the peak (long
   tails and banana shapes with few failures widen it). The grid is
   stretched (``STRETCH``): fine near the peak, coarser in the tails.
3. Evaluate the log posterior at every point at once.
4. Lay a second grid, shifted by half a cell in every direction, and
   compare the two grids' summaries (each parameter's median and 2.5%
   and 97.5% points, and those of the 10% quantile). Where half their
   difference is more than ``tol`` of the 95% interval's width, refine
   both and repeat; past the budget, refuse with the numbers.
5. The posterior is the two grids together, every point weighted by its
   density; the integration error reported is half the largest
   difference between the two. Quantiles spread each point's mass over
   its cell (:func:`lattice_quantile`), which makes them accurate to the
   square of the cell size.

No random numbers are used: the same data and prior give the same
numbers on every run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import optimize
from scipy.special import logsumexp

LogPost = Callable[[npt.NDArray], npt.NDArray]

#: Points per dimension of the first grid, and the most it may grow to.
START_POINTS = {1: 401, 2: 61, 3: 25}
MAX_POINTS = {1: 8001, 2: 241, 3: 61}
#: The accuracy each grid must reach: its error estimate (half the
#: difference of the two offset grids' summaries) as a fraction of the
#: 95% interval's width. Three parameters are held to 0.5%: a 61^3 grid
#: is the most that stays near a few seconds.
TOLERANCE = {1: 1e-3, 2: 1e-3, 3: 5e-3}
#: Half-width of the first grid, in standard deviations of the peak's
#: curvature, and the most it may be widened to.
START_HALF_WIDTH = 6.0
MAX_HALF_WIDTH = 60.0
#: The face density, relative to the peak, below which the grid holds
#: the posterior.
EDGE = 1e-8
#: The half-width (standard deviations of the peak's curvature) past
#: which the grid is laid again along the spread it has found.
REPLACE_AT = 15.0
#: The grid is even in ``u``, and a point's distance from the peak in
#: standard deviations is ``z = STRETCH * sinh(u / STRETCH)``: about
#: ``u`` within a few standard deviations, growing exponentially beyond,
#: so the cells are fine at the centre and coarse in a long tail.
STRETCH = 3.0


def _z_of(u: npt.NDArray) -> npt.NDArray:
    return STRETCH * np.sinh(u / STRETCH)


def _u_of(z: float) -> float:
    return float(STRETCH * np.arcsinh(z / STRETCH))


def _log_dz(u: npt.NDArray) -> npt.NDArray:
    """``log |dz/du|`` summed over the dimensions of each point."""
    a = np.abs(u / STRETCH)
    return np.sum(a + np.log1p(np.exp(-2 * a)) - np.log(2.0), axis=-1)


class GridError(ValueError):
    """The grid could not hold the posterior to the stated accuracy, or
    the posterior has no finite mass."""


@dataclass
class GridResult:
    """A posterior on a grid: the points (free parameters, unconstrained
    scale), their normalised log weights and the evidence."""

    phi: npt.NDArray
    log_w: npt.NDArray
    log_z: float
    error: float
    points_per_dim: list
    cell: npt.NDArray
    refinements: int
    mode: npt.NDArray
    cov: npt.NDArray
    half_widths: npt.NDArray
    checks: dict = field(default_factory=dict)
    #: (lattices, m, ..., m): how the points are laid out.
    shape: tuple = ()


def weighted_quantile(
    values: npt.NDArray, weights: npt.NDArray, q: npt.ArrayLike
) -> npt.NDArray:
    """Quantiles ``q`` of ``values`` (``(N,)`` or ``(N, m)``, one set per
    column) weighted by ``weights`` (``(N,)``, summing to 1), by linear
    interpolation between the sorted values at their mid-weights; the
    result is ``q``'s shape, or ``(m,) + q.shape``.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.bayes.engines.grid import weighted_quantile
    >>> float(weighted_quantile(np.array([1.0, 2, 3]), np.ones(3) / 3, 0.5))
    2.0
    """
    q = np.asarray(q, dtype=float)
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    one = v.ndim == 1
    if one:
        v = v[:, None]
    order = np.argsort(v, axis=0, kind="stable")
    vs = np.take_along_axis(v, order, axis=0)
    ws = w[order]
    cum = np.cumsum(ws, axis=0) - 0.5 * ws
    out = np.empty((v.shape[1],) + q.shape)
    for j in range(v.shape[1]):
        out[j] = np.interp(q, cum[:, j], vs[:, j])
    if one:
        return out[0]
    return out


def lattice_quantile(
    values: npt.NDArray,
    weights: npt.NDArray,
    q: npt.ArrayLike,
    shape: tuple[int, ...],
) -> npt.NDArray:
    """Quantiles ``q`` of a function's ``values`` at the points of
    lattices of ``shape`` (``(lattices, m, ..., m)``, the points in that
    order), each point's mass spread over its cell.

    A point's mass is not all at its value: across its cell the function
    changes by about its difference to the neighbouring points, and the
    mass is spread evenly over that change (a uniform with the cell's
    variance, ``sum(a_k ** 2) / 12`` for changes ``a_k`` along the
    lattice's axes). The plain weighted quantile of the values converges
    only as fast as the cells shrink. The spread one converges as their
    square, but its error is the same on every lattice of that size (the
    spread adds the cells' variance), so it is taken off by extrapolation:
    the quantile spread twice as wide moves four times as far, and ``(4
    Q1 - Q2) / 3`` is left.

    ``values`` is ``(N,)`` or ``(N, s)``; the result is ``q``'s shape, or
    ``(s,) + q.shape``.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.bayes.engines.grid import lattice_quantile
    >>> z = np.linspace(-6, 6, 201)
    >>> w = np.exp(-0.5 * z**2); w /= w.sum()
    >>> round(float(lattice_quantile(z, w, 0.975, (1, 201))), 3)
    1.96
    """
    q_shape = np.shape(q)
    q = np.atleast_1d(np.asarray(q, dtype=float))
    v = np.asarray(values, dtype=float)
    one = v.ndim == 1
    if one:
        v = v[:, None]
    s = v.shape[1]
    finite = np.isfinite(v)
    # Each column on a scale where it is about even: logit for values in
    # (0, 1), log for positive ones (a long tail, or values spanning many
    # decades, otherwise swamps the sums); quantiles carry back exactly.
    scales = []
    v = v.copy()
    with np.errstate(all="ignore"):
        for j in range(s):
            col = v[finite[:, j], j]
            if col.size and np.all(col > 0) and np.all(col < 1):
                v[:, j] = np.log(v[:, j]) - np.log1p(-v[:, j])
                scales.append("logit")
            elif col.size and np.all(col > 0):
                v[:, j] = np.log(v[:, j])
                scales.append("log")
            else:
                scales.append("identity")
    finite = np.isfinite(v)
    vf = np.where(finite, v, 0.0)
    V = vf.reshape(tuple(shape) + (s,))
    var = np.zeros(V.shape)
    with np.errstate(all="ignore"):
        for k in range(1, len(shape)):
            if shape[k] > 1:
                g = np.gradient(V, axis=k)
                var = var + g**2
        sd = np.sqrt(var / 12.0).reshape(v.shape)
    sd = np.where(finite & np.isfinite(sd), sd, 0.0)
    w = np.asarray(weights, dtype=float)
    keep = w > 1e-14 * w.max()
    w, vk, sdk, fin = w[keep], v[keep], sd[keep], finite[keep]
    w = w / w.sum()
    out = np.empty((s, q.size))
    for j in range(s):
        wj = w * fin[:, j]
        half = np.sqrt(3.0) * sdk[:, j]
        # The spread adds each cell's variance (sigma^2) to the points',
        # which already have the right spread (a lattice's sums are
        # accurate to high order), and so moves a quantile by about
        # sigma^2 times a constant. Spread over two cells (4 sigma^2)
        # moves it four times as far, so (4 Q1 - Q2) / 3 takes the move
        # off. (Widths of one and two cells: a spread over a whole number
        # of cells leaves no ripple where the points are in rows.)
        once = _spread_quantiles(vk[:, j], half, wj, q)
        twice = _spread_quantiles(vk[:, j], 2.0 * half, wj, q)
        with np.errstate(invalid="ignore"):
            out[j] = np.where(
                np.isfinite(once) & np.isfinite(twice),
                (4 * once - twice) / 3,
                once,
            )
    with np.errstate(all="ignore"):
        for j, scale in enumerate(scales):
            if scale == "logit":
                out[j] = 1.0 / (1.0 + np.exp(-out[j]))
            elif scale == "log":
                out[j] = np.exp(out[j])
    shaped: npt.NDArray = out.reshape((s,) + q_shape)
    return shaped[0] if one else shaped


def _spread_quantiles(
    c: npt.NDArray, h: npt.NDArray, w: npt.NDArray, q: npt.NDArray
) -> npt.NDArray:
    """Quantiles of the mixture of uniforms on ``[c - h, c + h]`` with
    weights ``w`` (a point mass where ``h`` is 0), exactly: its CDF is
    piecewise linear, built from the sorted ends."""
    ok = (w > 0) & np.isfinite(c)
    c, h, w = c[ok], h[ok], w[ok]
    finite_mass = float(w.sum())
    if c.size == 0:
        return np.full(np.shape(q), np.inf)
    # spreads far narrower than the values' range are points (their
    # slopes would swamp the sums)
    span = float(np.ptp(c))
    wide = h > 1e-9 * max(span, 1e-300)
    slope = np.where(wide, w / np.where(wide, 2 * h, 1.0), 0.0)
    t = np.concatenate([np.where(wide, c - h, c), np.where(wide, c + h, c)])
    d_slope = np.concatenate([slope, -slope])
    jump = np.concatenate([np.where(wide, 0.0, w), np.zeros_like(w)])
    order = np.argsort(t, kind="stable")
    t, d_slope, jump = t[order], d_slope[order], jump[order]
    run = np.cumsum(d_slope)
    area = run[:-1] * np.diff(t)
    F = np.cumsum(np.concatenate([[jump[0]], area + jump[1:]]))
    F = np.maximum.accumulate(np.clip(F, 0.0, None))
    out = np.interp(q, F, t)
    return np.where(q > finite_mass * (1 + 1e-12), np.inf, out)


def _vector_gradient(f: LogPost, x: npt.NDArray, h: float) -> npt.NDArray:
    d = x.size
    pts = np.vstack([x + h * np.eye(d), x - h * np.eye(d)])
    v = f(pts)
    return (v[:d] - v[d:]) / (2 * h)


def hessian(f: LogPost, x: npt.NDArray, h: npt.NDArray) -> npt.NDArray:
    """The Hessian of ``f`` at ``x`` by central differences with steps
    ``h``, all points evaluated in one call."""
    d = x.size
    pts = [x]
    for i in range(d):
        for s in (1, -1):
            e = np.zeros(d)
            e[i] = s * h[i]
            pts.append(x + e)
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    for i, j in pairs:
        for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            e = np.zeros(d)
            e[i], e[j] = si * h[i], sj * h[j]
            pts.append(x + e)
    v = f(np.vstack(pts))
    f0 = v[0]
    H = np.empty((d, d))
    for i in range(d):
        H[i, i] = (v[1 + 2 * i] - 2 * f0 + v[2 + 2 * i]) / h[i] ** 2
    base = 1 + 2 * d
    for k, (i, j) in enumerate(pairs):
        a, b, c, e = v[base + 4 * k : base + 4 * k + 4]
        H[i, j] = H[j, i] = (a - b - c + e) / (4 * h[i] * h[j])
    return H


def find_mode(
    log_post: LogPost, starts: list[npt.NDArray]
) -> tuple[npt.NDArray, npt.NDArray, float]:
    """The peak of ``log_post`` from the best of ``starts``, with the
    covariance of the normal approximation there (the inverse of minus
    the Hessian, made positive definite where it is not)."""

    def neg(x: npt.NDArray) -> float:
        v = float(log_post(x[None, :])[0])
        return -v if np.isfinite(v) else 1e300

    def grad(x: npt.NDArray) -> npt.NDArray:
        g = -_vector_gradient(log_post, x, 1e-6)
        return np.where(np.isfinite(g), g, 0.0)

    best: tuple[float, npt.NDArray] | None = None
    for s in starts:
        s = np.asarray(s, dtype=float)
        if not np.all(np.isfinite(s)) or neg(s) >= 1e300:
            continue
        with np.errstate(all="ignore"):
            res = optimize.minimize(
                neg, s, jac=grad, method="BFGS", options={"gtol": 1e-7}
            )
            x = res.x
            if not res.success:
                res2 = optimize.minimize(
                    neg,
                    x,
                    method="Nelder-Mead",
                    options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 2000},
                )
                if res2.fun < neg(x):
                    x = res2.x
        v = neg(x)
        if best is None or v < best[0]:
            best = (v, x)
    if best is None:
        raise GridError(
            "No start gives a finite posterior density: the prior and the "
            "data have no parameter values in common."
        )
    x = best[1]
    d = x.size
    cov = np.eye(d)
    h = np.full(d, 1e-3)
    for _ in range(2):
        with np.errstate(all="ignore"):
            H = -hessian(log_post, x, h)
        H = 0.5 * (H + H.T)
        if not np.all(np.isfinite(H)):
            H = np.eye(d)
        vals, vecs = np.linalg.eigh(H)
        vals = np.where(vals > 1e-8, vals, 1.0 / 16.0)
        cov = (vecs / vals) @ vecs.T
        h = np.clip(0.2 * np.sqrt(np.diag(cov)), 1e-5, 1.0)
    return x, cov, -best[0]


def _lattice(
    lo: npt.NDArray, hi: npt.NDArray, m: Any, shift: bool
) -> tuple[npt.NDArray, npt.NDArray]:
    """The z points of a lattice of ``m`` points a dimension over
    ``[lo, hi]`` (shifted by half a cell when ``shift``), and the cell
    sizes."""
    axes = []
    steps = []
    ms = np.broadcast_to(np.asarray(m, dtype=int), np.shape(lo))
    for a, b, mj in zip(lo, hi, ms):
        ax, step = np.linspace(a, b, int(mj), retstep=True)
        if shift:
            ax = ax + 0.5 * step
        axes.append(ax)
        steps.append(step)
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in mesh], axis=1), np.asarray(steps)


def _faces(
    z: npt.NDArray, lo: npt.NDArray, hi: npt.NDArray, step: Any
) -> dict[tuple[int, int], npt.NDArray]:
    """Masks of the points on each face: (dim, side) -> mask."""
    out = {}
    for j in range(z.shape[1]):
        out[(j, 0)] = z[:, j] <= lo[j] + 0.75 * step[j]
        out[(j, 1)] = z[:, j] >= hi[j] - 0.75 * step[j]
    return out


def run_grid(
    log_post: LogPost,
    mode: npt.NDArray,
    cov: npt.NDArray,
    summary: Callable[..., npt.NDArray],
    *,
    points: int | None = None,
    max_refinements: int = 3,
    tol: float | None = None,
    flat_names: list[str] | None = None,
) -> GridResult:
    """The posterior on the self-checking grid (see the module's
    docstring).

    ``summary(phi, weights, shape)`` gives the numbers to check, shape
    ``(3, s)``: the 2.5%, 50% and 97.5% points of ``s`` quantities on a
    lattice of ``shape``.
    """
    d = mode.size
    if d not in START_POINTS:
        raise GridError(
            f"The grid takes 1 to 3 free parameters; this fit has {d}."
        )
    if tol is None:
        tol = TOLERANCE[d]
    L = np.linalg.cholesky(cov)
    log_det_l = float(np.sum(np.log(np.diag(L))))
    lo = np.full(d, -_u_of(START_HALF_WIDTH))
    hi = np.full(d, _u_of(START_HALF_WIDTH))
    u_max = _u_of(MAX_HALF_WIDTH)
    m0 = int(points) if points is not None else START_POINTS[d]
    m_max = max(MAX_POINTS[d], m0)
    m = np.full(d, m0)
    target = (hi[0] - lo[0]) / (m0 - 1)

    centre = np.asarray(mode, dtype=float).copy()
    replaced = False

    def evaluate(shift: bool) -> tuple:
        u, step = _lattice(lo, hi, m, shift)
        phi = centre + _z_of(u) @ L.T
        with np.errstate(all="ignore"):
            lp = np.asarray(log_post(phi), dtype=float)
        lp = np.where(np.isfinite(lp), lp, -np.inf)
        return u, step, phi, lp

    # Widen until every face is below EDGE of the peak.
    widened = 0
    while True:
        u, step, phi, lp = evaluate(False)
        peak = lp.max()
        if not np.isfinite(peak):
            raise GridError(
                "The posterior density is zero everywhere near its peak."
            )
        faces = _faces(u, lo, hi, step)
        grow = False
        for (j, side), mask in faces.items():
            if lp[mask].max() - peak > np.log(EDGE):
                if side == 0:
                    lo[j] = lo[j] * 1.25
                else:
                    hi[j] = hi[j] * 1.25
                grow = True
        if not grow:
            break
        widened += 1
        if not replaced and max(np.max(-lo), np.max(hi)) > _u_of(REPLACE_AT):
            # The curvature at the peak understates the posterior's
            # spread (a long or curved ridge): lay the grid again along
            # the spread the grid has found so far, once.
            replaced = True
            lw = lp + _log_dz(u)
            w = np.exp(lw - logsumexp(lw))
            mean = w @ phi
            spread = (phi - mean).T @ ((phi - mean) * w[:, None])
            spread = 0.5 * (spread + spread.T)
            if np.all(np.isfinite(spread)) and np.all(
                np.linalg.eigvalsh(spread) > 0
            ):
                centre = mean
                L = np.linalg.cholesky(spread)
                log_det_l = float(np.sum(np.log(np.diag(L))))
                lo = np.full(d, -_u_of(START_HALF_WIDTH))
                hi = np.full(d, _u_of(START_HALF_WIDTH))
                m = np.full(d, m0)
                continue
        if max(np.max(-lo), np.max(hi)) > u_max:
            raise GridError(_improper_message(flat_names))
        # each dimension keeps its cells' size as it widens
        m = np.minimum(
            m_max, np.maximum(m, np.ceil((hi - lo) / target) + 1)
        ).astype(int)

    refinements = 0
    checks: dict = {}
    while True:
        ua, step, phia, lpa = u, step, phi, lp
        ub, _, phib, lpb = evaluate(True)
        la = lpa + _log_dz(ua)
        lb = lpb + _log_dz(ub)
        wa = np.exp(la - logsumexp(la))
        wb = np.exp(lb - logsumexp(lb))
        one = (1,) + tuple(int(v) for v in m)
        sa = np.asarray(summary(phia, wa, one), dtype=float)
        sb = np.asarray(summary(phib, wb, one), dtype=float)
        width = np.abs(0.5 * (sa[2] + sb[2]) - 0.5 * (sa[0] + sb[0]))
        width = np.where(width > 0, width, 1.0)
        rel = np.abs(sa - sb) / width
        rel = np.where(np.isfinite(rel), rel, 0.0)
        err = float(rel.max()) if rel.size else 0.0
        checks = {"difference": err}
        # The answer is the two grids together; half their difference is
        # its error estimate, which must be within ``tol``.
        if 0.5 * err <= tol:
            break
        if refinements >= max_refinements or np.all(m >= m_max):
            raise GridError(
                f"The grid did not reach its accuracy: the summaries of two "
                f"grids of {' x '.join(str(int(v)) for v in m)} points, "
                "offset by half a cell, "
                f"differ by {err:.2%} of the 95% interval's width, an "
                f"error of about {0.5 * err:.2%} (the target is "
                f"{tol:.2%}). Pass points= larger (up to {m_max}), give "
                "the parameters with long tails a prior, or use "
                "method='laplace' for an approximation."
            )
        refinements += 1
        m = np.minimum(m_max, 2 * m - 1).astype(int)
        u, step, phi, lp = evaluate(False)

    u_all = np.vstack([ua, ub])
    phi_all = np.vstack([phia, phib])
    l_all = np.concatenate([la, lb])
    log_cell = float(np.sum(np.log(step))) + log_det_l
    log_z = float(logsumexp(l_all) - np.log(2.0) + log_cell)
    log_w = l_all - logsumexp(l_all)
    # each point's cell on the unconstrained scale (for draws)
    dz = np.cosh(u_all / STRETCH)
    cells = L[None, :, :] * (dz * step[None, :])[:, None, :]
    return GridResult(
        phi=phi_all,
        log_w=log_w,
        log_z=log_z,
        error=0.5 * err,
        points_per_dim=[int(v) for v in m],
        cell=cells,
        refinements=refinements,
        mode=mode,
        cov=cov,
        half_widths=_z_of(np.maximum(-lo, hi)),
        checks={**checks, "widened": widened},
        shape=(2,) + tuple(int(v) for v in m),
    )


def _improper_message(flat_names: list[str] | None) -> str:
    if flat_names:
        names = ", ".join(flat_names)
        which = "which has" if len(flat_names) == 1 else "which have"
        return (
            f"The posterior does not fall away from its peak, so it has no "
            f"finite mass: the data alone do not determine {names}, "
            f"{which} no prior (flat). Give {names} a prior, for example "
            "prior.around(value, between=(low, high))."
        )
    return (
        "The posterior does not fall away from its peak within "
        f"{MAX_HALF_WIDTH:g} standard deviations of its curvature: the "
        "prior is too wide for these data. Give a narrower prior."
    )


def laplace_grid(
    mode: npt.NDArray,
    cov: npt.NDArray,
    points: int | None = None,
) -> GridResult:
    """The normal approximation at the peak (Laplace), laid on a grid so
    that every summary is computed as the grid's are."""
    d = mode.size
    m = int(points) if points is not None else START_POINTS.get(d, 25)
    L = np.linalg.cholesky(cov)
    lo = np.full(d, -START_HALF_WIDTH)
    hi = np.full(d, START_HALF_WIDTH)
    za, step = _lattice(lo, hi, m, False)
    zb, _ = _lattice(lo, hi, m, True)
    z = np.vstack([za, zb])
    lp = -0.5 * np.sum(z**2, axis=1)
    log_w = lp - logsumexp(lp)
    return GridResult(
        phi=mode + z @ L.T,
        log_w=log_w,
        log_z=float("nan"),
        error=float("nan"),
        points_per_dim=[m] * d,
        cell=np.repeat((L * step[None, :])[None], len(z), axis=0),
        refinements=0,
        mode=mode,
        cov=cov,
        half_widths=hi.copy(),
        shape=(2,) + (m,) * d,
    )
