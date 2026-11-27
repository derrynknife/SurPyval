"""The joint likelihood of a marker and a lifetime, by adaptive
Gauss-Hermite quadrature over each item's path parameters (#891).

Item ``i``'s marker follows a degradation path model (``LinearPath``,
``ExponentialPath``, ... or a ``PathModel`` of one's own) with
parameters ``theta_i``, read with normal noise:

    y_ij = path(t_ij; theta_i) + e_ij,  e_ij ~ N(0, sigma^2).

The path parameters that vary between items (``varying``) are
multivariate normal about a mean that the item's covariates shift,
``theta_iV ~ N(mu_V + Delta_V z_i, D)``; the others are common,
``theta_iC = mu_C + Delta_C z_i``. The hazard is Weibull proportional
hazards in the path's current value and the covariates (a covariate may
be in the hazard, the path or both: the entries of ``Delta`` and
``gamma`` it is not in are 0, not estimated),

    h_i(t) = h0(t) exp(gamma'z_i + zeta (path(t; theta_i) - c)).

The hazard may follow, in place of the path's value or beside it, the
path's slope or the area under it from age 0 (the links, each with an
effect of its own): ``zeta (path - c)`` is then the sum over the links
(:func:`predictor`), and everything below holds with it. The area may
count older exposure for less (:class:`WeightedArea`: an exponential
weight, or a window), and each link may be followed a lag later
(:class:`LaggedPath`).

Given ``theta_i`` the readings and the lifetime are independent, so an
item's likelihood is the integral over its varying parameters

    L_i = int p(y_i | theta) h(T_i | theta)^d_i S(T_i | theta)
          phi(theta_V; mu_V + Delta_V z_i, D) d theta_V,

evaluated by adaptive Gauss-Hermite quadrature: a product grid
(7 points a dimension; on request without the nodes whose weight is
negligible, :func:`gh_grid`; beyond five varying parameters there is no
grid, and the integral is Laplace's approximation, :mod:`._laplace`)
centred at the mode of the integrand and scaled by its curvature there
(Gauss-Newton: the path's Jacobian, which is the exact curvature for a
path linear in its parameters). The nodes are
placed in the space of the varying parameters themselves and held fixed
while the parameters are searched, so the path is evaluated at fixed
nodes (any path model works, and autograd differentiates the rest),
then placed again at the new parameters until the log-likelihood stops
moving. A common path parameter is searched through the path: its
derivatives are the path model's Jacobian (its own analytic one, else
central differences), and the second derivatives central differences
of that.

An item known only to have failed in an interval ``(l_i, u_i]``
(interval censored; ``l_i = 0`` for left censored) has ``S(l_i |
theta) - S(u_i | theta)`` in place of ``h^d S``, the hazard integrated
over the interval directly (below). An item that came under
observation at ``L_i > 0`` is in the data because it reached ``L_i``,
and one right truncated at ``R_i`` because it failed by then: its
likelihood is divided by ``int (S(L_i | theta) - S(R_i | theta))
phi(theta_V) d theta_V`` over the population's paths, by adaptive
quadrature of its own.

The cumulative hazard along a path is, with ``s = t u`` and ``u = w^4``,

    H(t) = exp(gamma'z) (t / alpha)^beta
           int_0^1 4 beta w^(4 beta - 1) exp(zeta (path(t w^4) - c)) dw,

by 15-point Gauss-Legendre in ``w``: the nodes do not move with
``beta``, so the path is evaluated at fixed ages, and the error is
below 3e-6 relative for shapes 0.5 to 5 with the hazard changing by a
factor of up to e^8 over ``(0, t]``.

That is the proportional hazards survival part. The accelerated one
(``Layout.survival == "AFT"``: the marker speeds up the item's ageing)
has its own lifetime terms in :mod:`._aft`, which the integrand and the
search for each item's mode take in place of these
(:func:`lifetime_functions`); the rest is shared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from autograd.extend import defjvp, defvjp, primitive
from autograd.scipy.special import logsumexp

#: Gauss-Hermite points a dimension: in the fit (adaptive; 2-3 fail on
#: sparse data), and in a prediction (by the number of varying path
#: parameters: 15, 15, then 9 for three and 7 for four and five).
GH_FIT = 7
GH_PREDICT = {1: 15, 2: 15, 3: 9, 4: 7, 5: 7}
#: Gauss-Legendre points of the cumulative hazard.
GL_POINTS = 15
#: The most path parameters that may vary on a grid: a full grid has
#: 7^q nodes (2,401 for four, 16,807 for five). Beyond it the integral
#: is Laplace's approximation (:mod:`._laplace`).
MAX_VARYING = 5
#: How the integral over the varying parameters is done: on the full
#: product grid (``"grid"``), on it without the nodes of negligible
#: weight (``"pruned"``), or by Laplace's approximation (``"laplace"``:
#: no grid). The default is the full grid, the more accurate one,
#: wherever there is one (up to ``MAX_VARYING`` varying parameters), and
#: Laplace's approximation beyond; the others are options the caller
#: chooses.
INTEGRATIONS = ("grid", "pruned", "laplace")
#: The pruned grid leaves out the lowest-weight nodes of the product
#: grid, as many as have less than this share of the grid's total weight
#: between them (nodes of equal weight are kept or left out together).
PRUNE_SHARE = 1e-9


def gh_grid(
    dim: int, points: int, prune: bool = False
) -> tuple[npt.NDArray, npt.NDArray]:
    """The ``dim``-dimensional product Gauss-Hermite grid: nodes
    ``(K, dim)`` and log weights ``(K,)`` for ``int exp(-|z|^2) f(z)
    dz``.

    With ``prune`` the nodes of negligible weight are left out: the
    lowest-weight nodes, as many as together hold less than
    ``PRUNE_SHARE`` (1e-9) of the grid's total weight, nodes of equal
    weight kept or left out together (so the grid keeps its symmetries).
    They are the corners, where several coordinates are extreme at once:
    none of 7 points in up to three dimensions, 80 of the 2,401 nodes
    in four and 1,392 of the 16,807 in five. The rule depends on ``dim``
    and ``points`` only, never on the data. The kept nodes are in the
    full grid's order and
    their weights are not rescaled."""
    x, w = np.polynomial.hermite.hermgauss(points)
    grids = np.meshgrid(*([x] * dim), indexing="ij")
    z = np.stack([g.ravel() for g in grids], axis=-1)
    logw = np.zeros(z.shape[0])
    for g in np.meshgrid(*([np.log(w)] * dim), indexing="ij"):
        logw = logw + g.ravel()
    if prune:
        keep = _heavy_enough(logw)
        z, logw = z[keep], logw[keep]
    return z, logw


def _heavy_enough(logw: npt.NDArray) -> npt.NDArray:
    """The mask of the nodes a pruned grid keeps (:func:`gh_grid`)."""
    share = np.exp(logw - np.max(logw))
    share = share / share.sum()
    ordered = np.sort(share)
    # the nodes that could go one at a time, lightest first
    below = int(np.searchsorted(np.cumsum(ordered), PRUNE_SHARE, "left"))
    if below == 0:
        return np.ones(share.size, dtype=bool)
    if below == share.size:  # pragma: no cover - PRUNE_SHARE < 1
        below -= 1
    # ... but every node as heavy as the first one kept stays (a tie is
    # equal to a few ulps: the same product in another order)
    return share >= ordered[below] * (1.0 - 1e-9)


def check_integration(integration: Any, q: int) -> str:
    """The name of the integration over ``q`` varying parameters:
    ``integration`` itself where it is given, else the default for
    ``q``."""
    if integration is None:
        return "grid" if q <= MAX_VARYING else "laplace"
    if not isinstance(integration, str) or integration not in INTEGRATIONS:
        raise ValueError(
            "'integration' is how each item's likelihood is integrated "
            "over its varying path parameters: 'grid' (the full grid of "
            f"{GH_FIT} points in each), 'pruned' (the grid without the "
            "nodes of negligible weight) or 'laplace' (Laplace's "
            "approximation, with no grid), or leave it out for the "
            f"default; got {integration!r}."
        )
    if integration != "laplace" and q > MAX_VARYING:
        raise ValueError(
            f"integration={integration!r} is a grid of {GH_FIT} points in "
            f"each varying path parameter, too many beyond {MAX_VARYING} "
            f"of them ({GH_FIT}^{q} nodes an item for the {q} here): use "
            "integration='laplace' (the default for this many), or name "
            f"up to {MAX_VARYING} in varying=(...), the rest common to "
            "all items."
        )
    return integration


#: The covariance structures of the varying parameters named in words:
#: every pair correlated (one block), or none (each in a block of its
#: own). Any other is given as its blocks.
RANDOM_EFFECTS = ("full", "diagonal")


def check_random_effects(random_effects: Any, varying: Any) -> Any:
    """``random_effects`` for the varying parameters ``varying`` (in the
    path's order), checked and in one form: ``"full"`` (one block: every
    pair correlated), ``"diagonal"`` (each its own block: none
    correlated), or a tuple of blocks, each a tuple of the names of the
    parameters correlated with each other, in ``varying``'s order and the
    blocks by their first. Blocks that make one of the two are named by
    it (so one varying parameter is ``"full"``). Every varying parameter
    is in exactly one block; anything else is refused, saying which."""
    varying = list(varying)
    usage = (
        "'full' (every pair of varying path parameters correlated), "
        "'diagonal' (none), or a list of blocks of the names correlated "
        "with each other, such as [('a', 'b'), ('c',)], each varying "
        f"parameter in exactly one; the varying parameters are {varying!r}"
    )
    if isinstance(random_effects, str):
        if random_effects not in RANDOM_EFFECTS:
            raise ValueError(
                f"'random_effects' is {usage}; got {random_effects!r}."
            )
        if random_effects == "full":
            blocks = [list(varying)]
        else:
            blocks = [[v] for v in varying]
    else:
        try:
            if isinstance(random_effects, dict):
                raise TypeError
            given = list(random_effects)
        except TypeError:
            raise ValueError(
                f"'random_effects' is {usage}; got {random_effects!r}."
            ) from None
        blocks = []
        for block in given:
            if isinstance(block, str):
                block = [block]
            try:
                names = list(block)
            except TypeError:
                names = [block]
            if not names:
                raise ValueError(
                    "'random_effects' has an empty block: each block names "
                    f"one or more varying parameters, of {varying!r}."
                )
            blocks.append(names)
        named = [name for block in blocks for name in block]
        unknown = [name for name in named if name not in varying]
        if unknown:
            raise ValueError(
                f"'random_effects' names {unknown!r}, which are not varying "
                f"path parameters; those are {varying!r} (a parameter "
                "common to all items has no spread to correlate)."
            )
        twice = [v for v in varying if named.count(v) > 1]
        if twice:
            raise ValueError(
                f"'random_effects' names {twice!r} more than once: put each "
                "varying parameter in exactly one block."
            )
        missing = [v for v in varying if v not in named]
        if missing:
            raise ValueError(
                f"'random_effects' leaves out {missing!r}: put each varying "
                "parameter in exactly one block, one correlated with none "
                f"in a block of its own, such as ({missing[0]!r},)."
            )
    ordered = sorted(
        (tuple(sorted(block, key=varying.index)) for block in blocks),
        key=lambda block: varying.index(block[0]),
    )
    if len(ordered) == 1:
        return "full"
    if all(len(block) == 1 for block in ordered):
        return "diagonal"
    return tuple(ordered)


def block_labels(random_effects: Any, varying: Any) -> tuple:
    """The block of each of ``varying`` (``0, 1, ...``) in the checked
    ``random_effects`` (:func:`check_random_effects`)."""
    q = len(varying)
    if random_effects == "full":
        return (0,) * q
    if random_effects == "diagonal":
        return tuple(range(q))
    label = {
        name: k for k, block in enumerate(random_effects) for name in block
    }
    return tuple(label[name] for name in varying)


def _gl_grid() -> tuple[npt.NDArray, npt.NDArray]:
    x, w = np.polynomial.legendre.leggauss(GL_POINTS)
    nodes = (x + 1.0) / 2.0
    return nodes, np.log(w / 2.0 * 4.0)


#: Gauss-Legendre nodes ``w`` on (0, 1) and the log of their weights
#: times the 4 of ``u = w^4``; the ages of the nodes are ``t w^4``.
GL_NODES, GL_LOGW = _gl_grid()
GL_U = GL_NODES**4


# -- the path, twice differentiable by autograd ------------------------------


def _numeric_jacobian(path_model: Any, x: npt.NDArray, P: npt.NDArray) -> Any:
    """``d path / d theta`` at the flat ages ``x`` (M,) and parameters
    ``P`` (M, p), by central differences of the (broadcasting) path."""
    cols = []
    for j in range(P.shape[1]):
        h = 1e-6 * np.maximum(np.abs(P[:, j]), 1.0)
        up, down = P.copy(), P.copy()
        up[:, j] += h
        down[:, j] -= h
        cols.append(
            (path_model.path(x, *up.T) - path_model.path(x, *down.T)) / (2 * h)
        )
    return np.stack(cols, axis=-1)


def path_jacobian(path_model: Any, x: npt.NDArray, P: npt.NDArray) -> Any:
    """``d path / d theta`` ``(..., p)`` at ages ``x`` ``(...)`` and
    parameters ``P`` ``(..., p)``: the path model's own Jacobian where it
    takes arrays of parameters, else central differences."""
    shape = np.shape(x)
    xf = np.asarray(x, dtype=float).ravel()
    Pf = np.asarray(P, dtype=float).reshape(-1, P.shape[-1])
    try:
        with np.errstate(all="ignore"):
            J = np.asarray(path_model.jacobian(xf, *Pf.T), dtype=float)
        if J.shape != Pf.shape:
            raise ValueError
    except Exception:
        with np.errstate(all="ignore"):
            J = _numeric_jacobian(path_model, xf, Pf)
    return J.reshape(shape + (Pf.shape[1],))


#: What of the marker's path the hazard may follow (``link``), in the
#: order of their effects in the parameters, and each effect's name.
LINKS = ("value", "slope", "cumulative")
EFFECTS = {
    "value": "marker_effect",
    "slope": "slope_effect",
    "cumulative": "cumulative_effect",
}


#: The nodes (as fractions of the age) and weights of the area under a
#: path from age 0: the cumulative hazard's rule, ``s = x w^4`` with
#: 15-point Gauss-Legendre in ``w``, which is exact for a polynomial path
#: up to the sixth degree and keeps its accuracy for a path with a
#: logarithmic or root singularity at 0.
AREA_NODES = GL_U
AREA_W = np.exp(GL_LOGW) * GL_NODES**3


class LinkPath:
    """What the hazard follows of a marker's path, as a path of its own
    (``path`` and ``jacobian``, as a path model's): the path's rate of
    change (``"slope"``: the path model's ``slope`` and
    ``slope_jacobian``) or the area under it from age 0
    (``"cumulative"``: 15-point Gauss-Legendre of the path and of its
    Jacobian over ``(0, x)``, ``AREA_NODES``; for a path of joined
    pieces, the path model's own ``area`` and ``area_jacobian``, a
    closed form)."""

    def __init__(self, path_model: Any, link: str) -> None:
        self.base = path_model
        self.link = link
        self.name = f"{path_model.name} ({link})"
        self.parameter_names = list(path_model.parameter_names)
        #: Whether the area is the path model's own closed form: for a
        #: path of joined pieces (one with ``breaks``, a spline), which
        #: the rule below would not follow across its knots.
        self.closed = (
            link == "cumulative"
            and type(self) is LinkPath
            and bool(getattr(path_model, "breaks", ()))
            and hasattr(path_model, "area")
        )

    def path(self, x: Any, *params: Any) -> Any:
        x = np.asarray(x, dtype=float)
        if self.link == "slope":
            return np.asarray(self.base.slope(x, *params), dtype=float)
        if self.closed:
            return np.asarray(self.base.area(x, *params), dtype=float)
        spread = [np.asarray(v, dtype=float)[..., None] for v in params]
        m = np.asarray(
            self.base.path(x[..., None] * AREA_NODES, *spread), dtype=float
        )
        return x * np.sum(AREA_W * m, axis=-1)

    def jacobian(self, x: Any, *params: Any) -> Any:
        x = np.asarray(x, dtype=float)
        if self.link == "slope":
            return np.asarray(
                self.base.slope_jacobian(x, *params), dtype=float
            )
        if self.closed:
            return np.asarray(self.base.area_jacobian(x, *params), dtype=float)
        K = AREA_NODES.size
        P = np.stack(
            [
                np.repeat(np.broadcast_to(np.asarray(v, float), x.shape), K)
                for v in params
            ],
            axis=-1,
        )
        ages = (x[:, None] * AREA_NODES).ravel()
        J = path_jacobian(self.base, ages, P).reshape(x.size, K, -1)
        return x[:, None] * np.einsum("k,mkp->mp", AREA_W, J)


#: The weights the accumulated exposure may take (``weight``).
WEIGHTS = ("exponential", "window")
#: An exponentially weighted area is summed over the last ``EXP_SPAN``
#: scales of the weight (what is older has ``exp(-30)``, 1e-13, of the
#: newest exposure's weight), in pieces that end at these fractions of
#: that span back from the age (the oldest first).
EXP_SPAN = 30.0
EXP_EDGES = (1.0, 0.5, 0.2, 0.0)
#: Plain Gauss-Legendre on (0, 1), for a piece with nothing at its ends.
PLAIN_NODES = GL_NODES
PLAIN_W = np.exp(GL_LOGW) / 4.0


def check_weight(weight: Any) -> tuple | None:
    """``weight`` as ``(kind, scale)`` with a float scale, or ``None``;
    refused unless it is ``None`` or one of ``WEIGHTS`` with one finite
    scale above 0."""
    if weight is None:
        return None
    try:
        kind, scale = weight
        value = float(scale)
        fine = (
            isinstance(kind, str)
            and kind in WEIGHTS
            and not np.ndim(scale)
            and np.isfinite(value)
            and value > 0
        )
    except (TypeError, ValueError):
        fine = False
    if not fine:
        raise ValueError(
            "'weight' says how much of the accumulated exposure (the "
            "'cumulative' link) still counts, by how long ago it was: "
            "('exponential', scale) for exposure that fades as "
            "exp(-(time since) / scale), or ('window', width) to count "
            "only the last 'width' of it, the scale or width one finite "
            "number above 0 in the lifetimes' unit of time; or None for "
            f"all of it, equally; got {weight!r}."
        )
    return (kind, value)


class WeightedArea(LinkPath):
    """The area under a marker's path with older exposure counting for
    less, as a path of its own: at age ``x``

        int_0^x w(x - s) path(s) ds,

    with ``w(u) = exp(-u / scale)`` (``"exponential"``) or ``w(u) = 1``
    for ``u <= scale`` and 0 beyond (``"window"``). :meth:`rule` gives
    the ages and weights of the sum, which the path and its Jacobian
    share."""

    def __init__(self, path_model: Any, weight: tuple) -> None:
        super().__init__(path_model, "cumulative")
        self.kind, self.scale = weight[0], float(weight[1])
        self.name = (
            f"{path_model.name} (cumulative, {self.kind} weight "
            f"{self.scale:g})"
        )

    def rule(self, x: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """The ages ``(..., K)`` and weights ``(..., K)`` (the kernel
        ``w`` included) of the weighted area at the ages ``x`` ``(...)``.

        A window is one piece, ``(max(x - scale, 0), x)``, by the plain
        area's rule from its older end (``AREA_NODES``: the plain area's
        own sum while the window reaches back to age 0).

        An exponential weight is summed over ``(max(x - 30 scale, 0),
        x)`` in three pieces, which end 100%, 50%, 20% and 0% of that
        span back from ``x`` (``EXP_EDGES``), each by 15-point
        Gauss-Legendre: the kernel changes by a factor of at most
        ``e^15`` over a piece, which the rule follows to 1e-10 of the
        area. The oldest piece is by the plain area's rule where it
        starts at age 0 (a path may have a logarithmic or root
        singularity there), and by plain Gauss-Legendre where it does
        not (there the path is smooth, and what lies at its older end
        has a weight of 1e-13)."""
        x = np.asarray(x, dtype=float)[..., None]
        if self.kind == "window":
            lo = np.maximum(x - self.scale, 0.0)
            width = x - lo
            return lo + width * AREA_NODES, width * AREA_W
        reach = EXP_SPAN * self.scale
        span = np.minimum(x, reach)
        from_zero = x <= reach
        ages, weights = [], []
        for k in range(len(EXP_EDGES) - 1):
            a = x - span * EXP_EDGES[k]
            width = span * (EXP_EDGES[k] - EXP_EDGES[k + 1])
            if k == 0:
                nodes = np.where(from_zero, AREA_NODES, PLAIN_NODES)
                w = np.where(from_zero, AREA_W, PLAIN_W)
            else:
                nodes, w = PLAIN_NODES, PLAIN_W
            s = a + width * nodes
            ages.append(s)
            weights.append(width * w * np.exp(-(x - s) / self.scale))
        return np.concatenate(ages, axis=-1), np.concatenate(weights, axis=-1)

    def path(self, x: Any, *params: Any) -> Any:
        ages, weights = self.rule(x)
        spread = [np.asarray(v, dtype=float)[..., None] for v in params]
        m = np.asarray(self.base.path(ages, *spread), dtype=float)
        return np.sum(weights * m, axis=-1)

    def jacobian(self, x: Any, *params: Any) -> Any:
        x = np.asarray(x, dtype=float)
        ages, weights = self.rule(x)
        K = ages.shape[-1]
        P = np.stack(
            [
                np.repeat(np.broadcast_to(np.asarray(v, float), x.shape), K)
                for v in params
            ],
            axis=-1,
        )
        J = path_jacobian(self.base, ages.ravel(), P).reshape(x.size, K, -1)
        return np.einsum("mk,mkp->mp", weights, J)


class LaggedPath:
    """What the hazard follows of a marker's path ``lag`` later (a path
    model, or a :class:`LinkPath`, as ``inner``): at age ``x`` the
    inner path at ``max(x - lag, 0)``. The lagged path is the path held
    at its age-0 value until ``lag``, so its value there is the path's
    at age 0 and its slope and the area under it are 0 (``held_at_zero``
    is ``False`` for the value, ``True`` for the slope and the area)."""

    def __init__(self, inner: Any, lag: float, held_at_zero: bool) -> None:
        self.inner = inner
        self.lag = float(lag)
        self.held_at_zero = held_at_zero
        self.name = f"{inner.name} ({lag:g} earlier)"
        self.parameter_names = list(inner.parameter_names)

    def _ages(self, x: Any) -> tuple:
        """The inner path's ages, and where ``x`` is past the lag."""
        x = np.asarray(x, dtype=float)
        past = x > self.lag
        if self.held_at_zero:
            # 0 before the lag, whatever the inner path is at age 0
            return np.where(past, x - self.lag, 1.0), past
        return np.maximum(x - self.lag, 0.0), past

    def path(self, x: Any, *params: Any) -> Any:
        ages, past = self._ages(x)
        with np.errstate(all="ignore"):
            out = np.asarray(self.inner.path(ages, *params), dtype=float)
        return np.where(past, out, 0.0) if self.held_at_zero else out

    def jacobian(self, x: Any, *params: Any) -> Any:
        ages, past = self._ages(x)
        P = np.stack(
            [
                np.broadcast_to(np.asarray(v, float), ages.shape)
                for v in params
            ],
            axis=-1,
        )
        J = path_jacobian(self.inner, ages, P)
        return np.where(past[..., None], J, 0.0) if self.held_at_zero else J


def lag_rule(
    lo: npt.NDArray, hi: npt.NDArray, lag: Any
) -> tuple[npt.NDArray, npt.NDArray]:
    """The ages ``(N, 2 GL_POINTS)`` and log weights of the rule for
    ``int_lo^hi f(s) ds`` (``0 <= lo < hi``, finite) when the hazard
    follows the marker ``lag`` later: the interval split at ``lag``,
    where the lagged path has a kink (its slope a jump), and each piece
    by the cumulative hazard's own rule, ``s = a + (b - a) w^4`` and
    15-point Gauss-Legendre in ``w``. A piece of no width has weights
    ``-inf``. ``lag`` may be several ages, ascending
    (``PathFunction.cuts``: the lag, and where a window on the
    accumulated exposure starts to move): the interval is split at each,
    ``GL_POINTS`` ages a piece."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    edges = [lo]
    for cut in np.atleast_1d(np.asarray(lag, dtype=float)):
        edges.append(np.clip(cut, lo, hi))
    edges.append(hi)
    ages, logw = [], []
    with np.errstate(divide="ignore"):
        for a, b in zip(edges[:-1], edges[1:]):
            width = b - a
            ages.append(a[:, None] + width[:, None] * GL_U[None, :])
            logw.append(
                np.log(width)[:, None]
                + (GL_LOGW + 3 * np.log(GL_NODES))[None, :]
            )
    return np.concatenate(ages, axis=1), np.concatenate(logw, axis=1)


def check_lag(lag: Any) -> float:
    """``lag`` as a float, refused unless it is one finite number of 0
    or more."""
    try:
        value = float(lag)
    except (TypeError, ValueError):
        value = np.nan
    if np.ndim(lag) or not np.isfinite(value) or value < 0:
        raise ValueError(
            "'lag' is how long after a reading the hazard follows it (the "
            "hazard at age t follows the marker's path at t - lag, and at "
            "age 0 before the lag), one finite number of 0 or more in the "
            f"lifetimes' unit of time; got {lag!r}."
        )
    return value


class PathFunction:
    """``path(params, x)`` of a path model for autograd: ``params``
    ``(..., p)`` and ``x`` ``(...)`` of one shape, its derivatives the
    path model's Jacobian and, for the second, central differences of
    that. ``terms`` holds what the hazard follows, one ``(link, path,
    function)`` for each of ``links``: the path itself (``"value"``),
    or its :class:`LinkPath`; with a ``lag`` above 0, each of those
    ``lag`` later (:class:`LaggedPath`). With a ``weight`` (``(kind,
    scale)``, :func:`check_weight`) the cumulative link is the
    :class:`WeightedArea`. The readings are always of the path itself
    (``values``).

    ``cuts`` are the ages, ascending, at which what the hazard follows
    has a kink, where the integrals of the hazard over age are split
    (:func:`lag_rule`): the lag, with a window on the accumulated
    exposure the age at which the window stops reaching back to age 0
    (the lag plus its width), and the ages a path of joined pieces
    names as its ``breaks`` (a spline's knots, where its third
    derivative jumps), the lag later."""

    def __init__(
        self,
        path_model: Any,
        links: tuple = ("value",),
        lag: float = 0.0,
        weight: tuple | None = None,
    ) -> None:
        self.model = path_model
        self.values, self.jac = _differentiable(path_model)
        self.links = tuple(links)
        self.lag = float(lag)
        self.weight = weight if "cumulative" in self.links else None
        cuts = [self.lag] if self.lag > 0 else []
        if self.weight is not None and self.weight[0] == "window":
            cuts.append(self.lag + float(self.weight[1]))
        # a path of joined pieces (a spline): its knots, the lag later
        pieces = getattr(path_model, "base", path_model)
        for knot in getattr(pieces, "breaks", ()):
            if knot + self.lag > 0:
                cuts.append(float(knot) + self.lag)
        self.cuts = tuple(sorted(set(cuts)))
        self.terms: list = []
        for link in self.links:
            followed: Any
            if link == "value":
                followed = path_model
            elif link == "cumulative" and self.weight is not None:
                followed = WeightedArea(path_model, self.weight)
            else:
                followed = LinkPath(path_model, link)
            if self.lag > 0:
                followed = LaggedPath(followed, self.lag, link != "value")
                self.terms.append(
                    (link, followed, _differentiable(followed)[0])
                )
            elif link == "value":
                self.terms.append((link, path_model, self.values))
            else:
                self.terms.append(
                    (link, followed, _differentiable(followed)[0])
                )

    def __call__(self, params: Any, x: Any) -> Any:
        return self.values(params, x)


def _differentiable(path_model: Any) -> tuple:
    """``(values, jac)`` of a path model for autograd."""

    @primitive
    def values(params: Any, x: Any) -> Any:
        with np.errstate(all="ignore"):
            return np.asarray(
                path_model.path(x, *np.moveaxis(params, -1, 0)),
                dtype=float,
            )

    @primitive
    def jac(params: Any, x: Any) -> Any:
        return path_jacobian(path_model, x, params)

    def jac_vjp(ans: Any, params: Any, x: Any) -> Any:
        p = params.shape[-1]
        h = 1e-5 * np.maximum(np.abs(params), 1.0)
        second = []
        for j in range(p):
            up, down = params.copy(), params.copy()
            up[..., j] += h[..., j]
            down[..., j] -= h[..., j]
            second.append(
                (
                    path_jacobian(path_model, x, up)
                    - path_jacobian(path_model, x, down)
                )
                / (2 * h[..., j : j + 1])
            )
        S = np.stack(second, axis=-1)  # (..., p, p)
        return lambda g: np.einsum("...k,...kj->...j", g, S)

    defvjp(
        values,
        lambda ans, params, x: lambda g: g[..., None] * jac(params, x),
    )
    defvjp(jac, jac_vjp)
    # forward mode, which a prediction's confidence bound takes through
    # the path parameters common to all items (their means)
    defjvp(
        values,
        lambda g, ans, params, x: anp.sum(g * jac(params, x), axis=-1),
    )
    return values, jac


class _OneMarker:
    """What the hazard follows of one marker of several, as a path of
    the joint parameter vector: the marker's own (``inner``, a path or
    its :class:`LinkPath`) of its slice of the parameters, less
    ``shift`` (the marker's centre, for its value)."""

    def __init__(self, inner: Any, part: slice, shift: float, P: int):
        self.inner = inner
        self.part = part
        self.shift = float(shift)
        self.P = P
        self.name = getattr(inner, "name", "path")

    def path(self, x: Any, *params: Any) -> Any:
        with np.errstate(all="ignore"):
            out = np.asarray(
                self.inner.path(x, *params[self.part]), dtype=float
            )
        return out - self.shift

    def jacobian(self, x: Any, *params: Any) -> Any:
        x = np.asarray(x, dtype=float)
        own = np.stack(
            [
                np.broadcast_to(np.asarray(v, dtype=float), x.shape)
                for v in params[self.part]
            ],
            axis=-1,
        )
        J = path_jacobian(self.inner, x, own)
        out = np.zeros(x.shape + (self.P,))
        out[..., self.part] = J
        return out


class MarkerPaths:
    """The paths of several markers as the likelihood's ``path``: the
    parameter vector is each marker's path parameters in turn (``parts``
    of the :class:`~._multi.MarkerSet` ``model``), each marker's
    readings are of its own path (:meth:`reading_means`), and ``terms``
    holds what the hazard follows of each, one ``(link, path,
    function)`` for each ``(marker, link)`` of ``links`` (a marker with
    none is read, and not in the hazard), each of the whole vector: a
    marker's value is less its own ``centre`` (the likelihood's centre is
    then 0). ``cuts`` are every marker's (:class:`PathFunction`)."""

    def __init__(
        self,
        model: Any,
        links: tuple,
        lag: float = 0.0,
        weight: tuple | None = None,
        centres: Any = None,
    ) -> None:
        self.model = model
        from ._multi import lag_of, weight_of

        self.links = tuple(tuple(pair) for pair in links)
        self.lag = lag
        self.weight = weight
        names = list(model.names)
        P = len(model.parameter_names)
        centres = np.zeros(len(names)) if centres is None else centres
        self.centres = np.asarray(centres, dtype=float).reshape(len(names))
        self.parts: list = []
        self.slices: list = list(model.slices)
        cuts: set = set()
        for name, path_model in zip(names, model.models):
            own = tuple(link for marker, link in self.links if marker == name)
            part = PathFunction(
                path_model, own, lag_of(lag, name), weight_of(weight, name)
            )
            self.parts.append(part)
            if own:
                cuts.update(part.cuts)
        self.cuts = tuple(sorted(cuts))
        self.terms: list = []
        for marker, link in self.links:
            k = names.index(marker)
            part, piece = self.parts[k], self.slices[k]
            j = part.links.index(link)
            _, followed, function = part.terms[j]
            shift = self.centres[k] if link == "value" else 0.0
            self.terms.append(
                (
                    link,
                    _OneMarker(followed, piece, shift, P),
                    _sliced(function, piece, shift),
                )
            )

    def reading_means(self, params: Any, arrays: JointArrays) -> Any:
        """Each reading's path value ``(N, K, R)`` at the parameters
        ``(N, K, P)``, each marker's at its own block of columns."""
        out = []
        for k, (part, piece) in enumerate(zip(self.parts, self.slices)):
            cols = arrays.columns(k)
            out.append(_at(part, params[..., piece], arrays.ages[:, cols]))
        return anp.concatenate(out, axis=-1)

    def reading_jacobians(
        self, params: npt.NDArray, arrays: JointArrays
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """The paths at the readings ``(N, R)`` and their Jacobian ``(N,
        R, P)`` at the parameters ``(N, P)`` (numbers, for the search for
        each item's mode)."""
        N, R = arrays.ages.shape
        m = np.zeros((N, R))
        J = np.zeros((N, R, params.shape[-1]))
        for k, (part, piece) in enumerate(zip(self.parts, self.slices)):
            cols = arrays.columns(k)
            ages = arrays.ages[:, cols]
            own = np.broadcast_to(
                params[:, None, piece],
                ages.shape + (params[:, piece].shape[-1],),
            )
            m[:, cols] = part.model.path(ages, *np.moveaxis(own, -1, 0))
            J[:, cols, piece] = path_jacobian(part.model, ages, np.array(own))
        return m, J

    def noise(self, sigma: Any, arrays: JointArrays) -> Any:
        """The noise sd of each reading column ``(R,)``."""
        return anp.reshape(sigma, (-1,))[arrays.channel[0]]


def _sliced(function: Any, piece: slice, shift: float) -> Any:
    """``function`` (of one marker's parameters) of the whole vector,
    less ``shift``."""

    def of_all(params: Any, x: Any) -> Any:
        out = function(params[..., piece], x)
        return out - shift if shift else out

    return of_all


def several_markers(path: Any) -> bool:
    """Whether ``path`` is several markers' (:class:`MarkerPaths`)."""
    return isinstance(path, MarkerPaths)


# -- the parameters ---------------------------------------------------------


def scatter(free: Any, mask: npt.NDArray) -> Any:
    """The vector of ``mask``'s length with ``free`` at its ``True``
    entries, in order, and 0 elsewhere (differentiable by autograd)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.all():
        return free
    if not mask.any():
        return anp.zeros(mask.size)
    zero = 0.0 * free[0]
    out, k = [], 0
    for on in mask:
        if on:
            out.append(free[k])
            k += 1
        else:
            out.append(zero)
    return anp.stack(out)


@dataclass
class Layout:
    """The internal parameter vector for a path of ``P`` parameters, of
    which those at ``varying`` vary between items, and ``p`` covariates:
    the path means ``mu`` (at the covariate means), their covariate
    shifts ``Delta`` ``(P, p)`` (the entries of ``shifts``, by rows), the
    log-Cholesky factor of ``D`` (log diagonal, then the lower triangle by
    rows), log ``sigma``, the log Weibull ``alpha`` (at the covariate
    means and the marker's centre) and ``beta``, the hazard's
    coefficients ``gamma`` (the covariates of ``hazard``) and the
    marker's effects ``zeta``, one for each of ``links`` (what of the
    path the hazard follows: its value, slope or cumulative area).
    ``survival`` is the survival part, ``"PH"`` (proportional hazards)
    or ``"AFT"`` (accelerated: the covariates and the links speed up
    the item's ageing); the parameters are the same in both.

    ``shifts`` ``(P, p)`` and ``hazard`` ``(p,)`` say which covariates
    enter which part (default every one in both): an entry that is off is
    held at 0, not estimated.

    ``blocks`` ``(q,)`` labels each varying parameter with its block of
    correlated ones (default one block: every pair correlated). An entry
    of the Cholesky factor between two blocks is held at 0, not
    estimated, so ``D`` is block diagonal: the factor of a matrix that is
    block diagonal, in any order of its rows, is 0 between the blocks
    too, and a factor that is gives such a matrix.

    ``n_causes`` above 1 is competing failure modes (causes): each has a
    hazard of its own, with its own log ``alpha`` and ``beta``, ``gamma``
    and ``zeta``, and the vector holds, after log ``sigma``, each cause's
    own entries in that order and then the entries the causes share, one
    each: ``beta`` where ``shared_beta``, the hazard coefficients of
    ``shared_hazard`` ``(p,)`` and the effects of ``shared_links``
    ``(n_links,)``. ``unpack`` then gives ``alpha`` and ``beta``
    ``(n_causes,)``, ``gamma`` ``(n_causes, p)`` and ``zeta``
    ``(n_causes, n_links)`` (:func:`cause_thetas` gives each cause's).
    With one cause the vector and ``unpack`` are as described above.

    ``n_markers`` above 1 is several markers (:class:`MarkerPaths`): the
    path is their paths' parameters one marker after another, ``links``
    are ``(marker, link)`` pairs, and log ``sigma`` is one entry a marker
    (``unpack`` gives ``sigma`` ``(n_markers,)``); the rest is as
    above.

    ``loaded`` ``(j, place, at, sign)`` holds the items' spread on an
    edge of the valid covariance matrices: the path parameter ``j`` (by
    place in the path, not in ``varying``) varies between items as an
    exact multiple ``lam`` of the varying one at ``place`` (of
    ``varying``), about their means, ``theta_j = mean_j + lam (theta_v -
    mean_v)``: a correlation of ``sign`` (+-1) between the two. ``lam``
    is one entry of the vector after the spread's (``unpack`` gives it as
    ``"lam"``); ``at`` is ``j``'s place among the varying parameters of
    the model with ``j`` varying freely (:attr:`full_varying`), whose
    order the reported parameters keep. ``None``: no such parameter."""

    P: int
    p: int
    varying: tuple
    shifts: Any = None
    hazard: Any = None
    links: tuple = ("value",)
    survival: str = "PH"
    blocks: Any = None
    n_causes: int = 1
    shared_beta: bool = False
    shared_hazard: Any = None
    shared_links: Any = None
    n_markers: int = 1
    loaded: Any = None

    def __post_init__(self) -> None:
        self.links = tuple(self.links)
        if self.loaded is not None:
            j, place, at, sign = self.loaded
            self.loaded = (int(j), int(place), int(at), float(sign))
        self.shifts = (
            np.ones((self.P, self.p), dtype=bool)
            if self.shifts is None
            else np.asarray(self.shifts, dtype=bool).reshape(self.P, self.p)
        )
        self.hazard = (
            np.ones(self.p, dtype=bool)
            if self.hazard is None
            else np.asarray(self.hazard, dtype=bool).reshape(self.p)
        )
        self.blocks = (
            (0,) * len(self.varying)
            if self.blocks is None
            else tuple(int(b) for b in self.blocks)
        )
        self.shared_hazard = (
            np.zeros(self.p, dtype=bool)
            if self.shared_hazard is None
            else np.asarray(self.shared_hazard, dtype=bool).reshape(self.p)
        ) & self.hazard
        self.shared_links = (
            np.zeros(len(self.links), dtype=bool)
            if self.shared_links is None
            else np.asarray(self.shared_links, dtype=bool).reshape(
                len(self.links)
            )
        )

    @property
    def q(self) -> int:
        return len(self.varying)

    @property
    def correlated(self) -> npt.NDArray:
        """``(q, q)``: whether two varying parameters are in one block
        (their correlation estimated, else held at 0)."""
        labels = np.asarray(self.blocks, dtype=int)
        return labels[:, None] == labels[None, :]

    @property
    def lower(self) -> list[tuple[int, int]]:
        """``(row, column)`` of each estimated entry below the diagonal
        of the Cholesky factor (and of each correlation), by rows."""
        same = self.correlated
        return [(r, c) for r in range(self.q) for c in range(r) if same[r, c]]

    @property
    def n_spread(self) -> int:
        """The number of estimated spread parameters: ``q`` sds and the
        correlations within the blocks (and ``lam``, :attr:`loaded`)."""
        return self.q + len(self.lower) + self.n_loaded

    @property
    def n_loaded(self) -> int:
        return 0 if self.loaded is None else 1

    @property
    def full_varying(self) -> tuple:
        """The varying parameters (by place in the path) with the one
        :attr:`loaded` among them where it is, as the model with it
        varying freely has them."""
        if self.loaded is None:
            return tuple(self.varying)
        j, _, at, _ = self.loaded
        out = list(self.varying)
        out.insert(at, j)
        return tuple(out)

    @property
    def full_blocks(self) -> tuple:
        """:attr:`blocks` of :attr:`full_varying`: the parameter
        :attr:`loaded` in the block of the one it is a multiple of."""
        if self.loaded is None:
            return tuple(self.blocks)
        _, place, at, _ = self.loaded
        out = list(self.blocks)
        out.insert(at, self.blocks[place])
        return tuple(out)

    @property
    def common(self) -> tuple:
        return tuple(j for j in range(self.P) if j not in self.varying)

    @property
    def n_shifts(self) -> int:
        return int(self.shifts.sum())

    @property
    def n_hazard(self) -> int:
        return int(self.hazard.sum())

    @property
    def n_links(self) -> int:
        return len(self.links)

    @property
    def n_survival(self) -> int:
        """The number of estimated parameters of the survival part: the
        Weibull ``alpha`` and ``beta``, the hazard's coefficients and
        the effects, of every cause (one each where shared)."""
        if self.n_causes == 1:
            return 2 + self.n_hazard + self.n_links
        return int(self.cause_slots.max()) + 1

    @property
    def cause_slots(self) -> npt.NDArray:
        """``(n_causes, 2 + n_hazard + n_links)``: where in the survival
        part of the vector each cause's log ``alpha``, log ``beta``,
        estimated hazard coefficients and effects are: its own entries,
        cause by cause, then the shared ones."""
        shared = np.concatenate(
            [
                [False, self.shared_beta],
                self.shared_hazard[self.hazard],
                self.shared_links,
            ]
        ).astype(bool)
        own = int((~shared).sum())
        slots = np.zeros((self.n_causes, shared.size), dtype=int)
        for k in range(self.n_causes):
            slots[k, ~shared] = k * own + np.arange(own)
            slots[k, shared] = self.n_causes * own + np.arange(shared.sum())
        return slots

    @property
    def size(self) -> int:
        return (
            self.P
            + self.n_shifts
            + self.n_spread
            + self.n_markers
            + self.n_survival
        )

    def shift_positions(self) -> list[tuple[int, int, int]]:
        """``(position in u, path parameter, covariate column)`` of each
        estimated shift."""
        rows, cols = np.nonzero(self.shifts)
        return [
            (self.P + k, int(j), int(c))
            for k, (j, c) in enumerate(zip(rows, cols))
        ]

    def hazard_positions(self) -> list[tuple[int, int]]:
        """``(position in u, covariate column)`` of each estimated
        hazard coefficient."""
        start = self.P + self.n_shifts + self.n_spread + self.n_markers + 2
        cols = np.flatnonzero(self.hazard)
        return [(start + k, int(c)) for k, c in enumerate(cols)]

    def _delta(self, free: Any) -> Any:
        """``Delta`` ``(P, p)`` with the estimated entries ``free`` and 0
        elsewhere."""
        P, p = self.P, self.p
        if self.shifts.all():
            return anp.reshape(free, (P, p))
        if p == 0 or not self.shifts.any():
            return anp.zeros((P, p))
        rows, k = [], 0
        for j in range(P):
            n = int(self.shifts[j].sum())
            rows.append(scatter(free[k : k + n], self.shifts[j]))
            k += n
        return anp.stack(rows)

    def _gamma(self, free: Any) -> Any:
        """``gamma`` ``(p,)`` with the estimated entries ``free`` and 0
        elsewhere."""
        return scatter(free, self.hazard)

    def unpack(self, u: Any) -> dict:
        P, q = self.P, self.q
        n_shifts, n_hazard = self.n_shifts, self.n_hazard
        i = 0
        mu = u[i : i + P]
        i += P
        Delta = self._delta(u[i : i + n_shifts])
        i += n_shifts
        diag = anp.exp(u[i : i + q])
        i += q
        n_lower = len(self.lower)
        lower = u[i : i + n_lower]
        i += n_lower
        lam = u[i] if self.loaded is not None else None
        i += self.n_loaded
        same = self.correlated
        rows = []
        k = 0
        for r in range(q):
            row = []
            for col in range(q):
                if col < r and same[r, col]:
                    row.append(lower[k])
                    k += 1
                elif col < r:
                    row.append(0.0 * diag[r])
                elif col == r:
                    row.append(diag[r])
                else:
                    row.append(0.0 * diag[r])
            rows.append(anp.stack(row))
        L = anp.stack(rows) if q else anp.zeros((0, 0))
        sigma = (
            anp.exp(u[i])
            if self.n_markers == 1
            else anp.exp(u[i : i + self.n_markers])
        )
        # the survival part starts after the noise
        i += self.n_markers - 1
        loaded = {} if lam is None else {"lam": lam}
        if self.n_causes > 1:
            slots = self.cause_slots
            block = u[i + 1 : i + 1 + self.n_survival]
            each = [block[slots[c]] for c in range(self.n_causes)]
            return {
                **loaded,
                "mu": mu,
                "Delta": Delta,
                "L": L,
                "sigma": sigma,
                "alpha": anp.exp(anp.stack([v[0] for v in each])),
                "beta": anp.exp(anp.stack([v[1] for v in each])),
                "gamma": anp.stack(
                    [self._gamma(v[2 : 2 + n_hazard]) for v in each]
                ),
                "zeta": anp.stack([v[2 + n_hazard :] for v in each]),
            }
        return {
            **loaded,
            "mu": mu,
            "Delta": Delta,
            "L": L,
            "sigma": sigma,
            "alpha": anp.exp(u[i + 1]),
            "beta": anp.exp(u[i + 2]),
            "gamma": self._gamma(u[i + 3 : i + 3 + n_hazard]),
            "zeta": u[i + 3 + n_hazard : i + 3 + n_hazard + self.n_links],
        }

    def pack(self, theta: dict) -> npt.NDArray:
        L = np.asarray(theta["L"], dtype=float)
        q = self.q
        lower = [L[r, c] for r, c in self.lower]
        if self.loaded is not None:
            lower.append(float(theta["lam"]))
        Delta = np.asarray(theta["Delta"], dtype=float).reshape(self.P, self.p)
        if self.n_causes > 1:
            C = self.n_causes
            each = np.column_stack(
                [
                    np.log(np.asarray(theta["alpha"], dtype=float)),
                    np.log(np.asarray(theta["beta"], dtype=float)),
                    np.asarray(theta["gamma"], dtype=float).reshape(C, self.p)[
                        :, self.hazard
                    ],
                    np.asarray(theta["zeta"], dtype=float).reshape(
                        C, self.n_links
                    ),
                ]
            )
            # a shared entry is its mean over the causes (equal in a
            # theta that unpack gave)
            slots = self.cause_slots
            block = np.zeros(self.n_survival)
            np.add.at(block, slots.ravel(), each.ravel())
            block /= np.bincount(slots.ravel(), minlength=block.size)
            return np.concatenate(
                [
                    np.asarray(theta["mu"], dtype=float),
                    Delta[self.shifts],
                    np.log(np.diag(L)) if q else np.zeros(0),
                    np.asarray(lower, dtype=float),
                    np.log(np.atleast_1d(np.asarray(theta["sigma"], float))),
                    block,
                ]
            )
        gamma = np.atleast_1d(np.asarray(theta["gamma"], dtype=float))
        return np.concatenate(
            [
                np.asarray(theta["mu"], dtype=float),
                Delta[self.shifts],
                np.log(np.diag(L)) if q else np.zeros(0),
                np.asarray(lower, dtype=float),
                np.log(np.atleast_1d(np.asarray(theta["sigma"], float))),
                [np.log(theta["alpha"]), np.log(theta["beta"])],
                gamma.reshape(self.p)[self.hazard],
                np.asarray(theta["zeta"], dtype=float).reshape(self.n_links),
            ]
        )


def cause_thetas(theta: dict, layout: Layout) -> list[dict]:
    """``theta`` for each cause's hazard: itself with that cause's
    ``alpha``, ``beta``, ``gamma`` and ``zeta`` (the one ``theta`` where
    the layout has one cause)."""
    if layout.n_causes == 1:
        return [theta]
    return [
        dict(
            theta,
            alpha=theta["alpha"][k],
            beta=theta["beta"][k],
            gamma=theta["gamma"][k],
            zeta=theta["zeta"][k],
        )
        for k in range(layout.n_causes)
    ]


# -- the data --------------------------------------------------------------


@dataclass
class JointArrays:
    """The items: counts ``w``, the age ``T`` each left observation (the
    lower end of an interval, 0 for a left censored item observed from
    0), the event flag ``d``, the upper end ``U`` of an interval (``inf``
    where the item has none), the entry age ``L`` (0 for an item observed
    from 0) and the right truncation ``Rt`` (``inf`` for none), the
    centred covariates ``Z`` ``(N, p)``, and the readings padded to ``(N,
    R)``: ``ages``, ``values`` and their ``mask``. ``e`` is the cause of
    each item's failure, its index among the model's causes (competing
    failure modes), and -1 where the item has not failed or its failure
    is of any cause (every item's in a model of one cause).

    ``channel`` ``(N, R)`` is the marker each reading column is of (0
    throughout for one marker). With several markers the columns are in
    blocks, a marker's readings in its own block (padded with its first
    reading's age), the blocks in the markers' order and the same in
    every row, so each marker's path is evaluated only at its own
    readings (:meth:`columns`)."""

    w: npt.NDArray
    T: npt.NDArray
    d: npt.NDArray
    L: npt.NDArray
    Z: npt.NDArray
    ages: npt.NDArray
    values: npt.NDArray
    mask: npt.NDArray
    U: npt.NDArray
    Rt: npt.NDArray
    e: Any = None
    channel: Any = None

    def __post_init__(self) -> None:
        if self.e is None:
            self.e = np.full(len(self.w), -1, dtype=int)
        if self.channel is None:
            self.channel = np.zeros(np.shape(self.mask), dtype=int)

    def columns(self, k: int) -> npt.NDArray:
        """The reading columns of marker ``k`` (a block)."""
        row = self.channel[0] if len(self.channel) else np.zeros(0, int)
        return np.flatnonzero(row == k)

    @classmethod
    def build(
        cls,
        w: Any,
        T: Any,
        d: Any,
        L: Any,
        Z: Any,
        ages: list,
        values: list,
        U: Any = None,
        Rt: Any = None,
        e: Any = None,
        channels: Any = None,
        n_markers: int = 1,
    ) -> JointArrays:
        """With ``channels`` (several markers), ``ages`` and ``values``
        hold each item's readings of every marker and ``channels`` the
        marker of each (``0`` to ``n_markers - 1``)."""
        N = len(ages)
        if channels is not None:
            A, V, M, C = _marker_blocks(ages, values, channels, n_markers, T)
        else:
            R = max([len(a) for a in ages] + [1])
            A = np.zeros((N, R))
            V = np.zeros((N, R))
            M = np.zeros((N, R), dtype=bool)
            C = np.zeros((N, R), dtype=int)
            for i, (a, v) in enumerate(zip(ages, values)):
                A[i, : len(a)] = a
                V[i, : len(v)] = v
                M[i, : len(a)] = True
            # padded ages repeat the first (or 0), so the path is defined
            # there
            for i in range(N):
                if len(ages[i]):
                    A[i, len(ages[i]) :] = ages[i][0]
                else:
                    A[i, :] = max(float(np.asarray(T, dtype=float)[i]), 1e-12)
        return cls(
            w=np.asarray(w, dtype=float),
            T=np.asarray(T, dtype=float),
            d=np.asarray(d, dtype=float),
            L=np.asarray(L, dtype=float),
            Z=np.asarray(Z, dtype=float).reshape(N, -1),
            ages=A,
            values=V,
            mask=M,
            U=(
                np.full(N, np.inf)
                if U is None
                else np.asarray(U, dtype=float).reshape(N)
            ),
            Rt=(
                np.full(N, np.inf)
                if Rt is None
                else np.asarray(Rt, dtype=float).reshape(N)
            ),
            e=(
                np.full(N, -1, dtype=int)
                if e is None
                else np.asarray(e, dtype=int).reshape(N)
            ),
            channel=C,
        )

    def of_any_cause(self) -> JointArrays:
        """The items with no cause given (``e`` -1): for a term that is
        of failing by any cause, such as a truncated item's probability
        of being in the data."""
        fields = {k: getattr(self, k) for k in self.__dataclass_fields__}
        fields["e"] = np.full(len(self.w), -1, dtype=int)
        return JointArrays(**fields)

    def take(self, index: Any) -> JointArrays:
        """The items ``index``."""
        return JointArrays(
            **{k: getattr(self, k)[index] for k in self.__dataclass_fields__}
        )

    @property
    def n_read(self) -> npt.NDArray:
        return self.mask.sum(axis=1).astype(float)

    @property
    def truncated(self) -> npt.NDArray:
        """The items in the data because they reached an age (left
        truncated) or failed by one (right truncated)."""
        return (self.L > 0) | np.isfinite(self.Rt)


def _marker_blocks(
    ages: list, values: list, channels: list, n_markers: int, T: Any
) -> tuple:
    """The padded ``(ages, values, mask, channel)`` of several markers'
    readings: a block of columns a marker, as wide as the most readings
    of it an item has (at least 1), each item's readings of it in age
    order there and padded with the first of them (with ``T``, as one
    marker's, where the item has none of it)."""
    N = len(ages)
    T = np.asarray(T, dtype=float)
    split = []
    for a, v, ch in zip(ages, values, channels):
        a = np.asarray(a, dtype=float)
        v = np.asarray(v, dtype=float)
        ch = np.asarray(ch, dtype=int)
        mine = []
        for k in range(n_markers):
            on = ch == k
            order = np.argsort(a[on], kind="stable")
            mine.append((a[on][order], v[on][order]))
        split.append(mine)
    widths = [
        max([len(item[k][0]) for item in split] + [1])
        for k in range(n_markers)
    ]
    R = int(sum(widths))
    A = np.zeros((N, R))
    V = np.zeros((N, R))
    M = np.zeros((N, R), dtype=bool)
    C = np.repeat(
        np.repeat(np.arange(n_markers), widths)[None, :], N, axis=0
    ).astype(int)
    for i, item in enumerate(split):
        start = 0
        for k, (a, v) in enumerate(item):
            width = widths[k]
            if len(a):
                A[i, start : start + width] = a[0]
                A[i, start : start + len(a)] = a
                V[i, start : start + len(v)] = v
                M[i, start : start + len(a)] = True
            else:
                A[i, start : start + width] = max(float(T[i]), 1e-12)
            start += width
    return A, V, M, C


# -- the integrand ---------------------------------------------------------


def _path_params(theta: dict, layout: Layout, Z: Any, nodes: Any) -> Any:
    """Each item's full path parameters at the nodes ``(N, K, q)`` of its
    varying ones: ``(N, K, P)``, the common ones its mean."""
    means = theta["mu"][None, :] + anp.dot(Z, anp.transpose(theta["Delta"]))
    N, K = nodes.shape[0], nodes.shape[1]
    loaded = layout.loaded
    cols = []
    for j in range(layout.P):
        if j in layout.varying:
            cols.append(nodes[..., layout.varying.index(j)])
        elif loaded is not None and j == loaded[0]:
            # a multiple of a varying one's departure from its mean
            v = layout.varying[loaded[1]]
            cols.append(
                means[:, j : j + 1]
                + theta["lam"] * (nodes[..., loaded[1]] - means[:, v : v + 1])
            )
        else:
            cols.append(anp.broadcast_to(means[:, j : j + 1], (N, K)))
    return anp.stack(cols, axis=-1)


def varying_jacobian(J: Any, layout: Layout, theta: dict) -> Any:
    """The Jacobian in the varying parameters from ``J`` ``(..., P)``,
    the one in every path parameter: its columns of the varying ones,
    and, where a parameter is a multiple of one of them
    (:attr:`Layout.loaded`), that parameter's column times the
    multiple added to that one's."""
    out = np.array(np.asarray(J)[..., list(layout.varying)])
    if layout.loaded is not None:
        j, place, _, _ = layout.loaded
        out[..., place] += float(theta["lam"]) * np.asarray(J)[..., j]
    return out


def _at(path: Any, params: Any, ages: npt.NDArray) -> Any:
    """``path`` at ``ages`` ``(N, A)`` for parameters ``(N, K, P)``:
    ``(N, K, A)``."""
    N, K, P = params.shape
    A = ages.shape[1]
    full = anp.broadcast_to(params[:, :, None, :], (N, K, A, P))
    x = np.broadcast_to(ages[:, None, :], (N, K, A))
    return path(full, x)


def predictor(
    theta: dict, path: PathFunction, params: Any, ages: npt.NDArray, c: float
) -> Any:
    """The marker's part of the log hazard at ``ages`` ``(N, A)`` for
    parameters ``(N, K, P)``, ``(N, K, A)``: each of the path's links
    times its effect, the value relative to the centre ``c``."""
    zeta = anp.reshape(theta["zeta"], (-1,))
    out: Any = 0.0
    for k, (link, _, function) in enumerate(path.terms):
        followed = _at(function, params, ages)
        if link == "value":
            followed = followed - c
        out = out + zeta[k] * followed
    return out


def log_prior(theta: dict, layout: Layout, Z: Any, nodes: Any) -> Any:
    """``log phi(theta_V; mu_V + Delta_V z, D)`` at the nodes. ``D`` is
    block diagonal where the layout has blocks (``Layout.blocks``): the
    Cholesky factor's entries between them, held at 0, are left out of
    the substitution."""
    q = layout.q
    if q == 0:
        return anp.zeros(nodes.shape[:2])
    V = list(layout.varying)
    mean = theta["mu"][V][None, :] + anp.dot(
        Z, anp.transpose(theta["Delta"][V])
    )
    diff = nodes - mean[:, None, :]
    L = theta["L"]
    same = layout.correlated
    # forward substitution: L x = diff
    xs: list = []
    for r in range(q):
        acc = diff[..., r]
        for c in range(r):
            if same[r, c]:
                acc = acc - L[r, c] * xs[c]
        xs.append(acc / L[r, r])
    quad = sum(x_**2 for x_ in xs)
    log_det = anp.sum(anp.log(anp.stack([L[r, r] for r in range(q)])))
    return -0.5 * q * np.log(2 * np.pi) - log_det - 0.5 * quad


def log_readings(
    theta: dict, path: Any, params: Any, arrays: JointArrays
) -> Any:
    """``log p(y_i | theta)`` at the nodes ``(N, K)``."""
    if several_markers(path):
        m = path.reading_means(params, arrays)
        r = arrays.values[:, None, :] - m
        s2 = path.noise(theta["sigma"], arrays) ** 2
        each = -0.5 * anp.log(2 * np.pi * s2) - r**2 / (2 * s2)
        return anp.sum(anp.where(arrays.mask[:, None, :], each, 0.0), axis=-1)
    m = _at(path, params, arrays.ages)
    r = arrays.values[:, None, :] - m
    ss = anp.sum(anp.where(arrays.mask[:, None, :], r**2, 0.0), axis=-1)
    s2 = theta["sigma"] ** 2
    n = arrays.n_read[:, None]
    return -0.5 * n * anp.log(2 * np.pi * s2) - ss / (2 * s2)


def log_cumhaz(
    theta: dict,
    path: PathFunction,
    params: Any,
    arrays: JointArrays,
    t: npt.NDArray,
    centre: float,
) -> Any:
    """``log H(t_i | theta)`` at the nodes ``(N, K)`` (``-inf`` where
    ``t_i`` is 0)."""
    positive = t > 0
    safe = np.where(positive, t, 1.0)
    if path.cuts:
        # from age 0, split at the lag (and a window's start)
        out = log_cumhaz_between(
            theta, path, params, arrays, np.zeros(safe.size), safe, centre
        )
        return anp.where(positive[:, None], out, -anp.inf)
    eta = predictor(theta, path, params, safe[:, None] * GL_U[None, :], centre)
    beta = theta["beta"]
    inner = logsumexp(
        GL_LOGW + (4 * beta - 1) * np.log(GL_NODES) + eta,
        axis=-1,
    )
    lin = anp.dot(arrays.Z, theta["gamma"])
    base = lin + beta * (anp.log(safe) - anp.log(theta["alpha"]))
    out = base[:, None] + anp.log(beta) + inner
    return anp.where(positive[:, None], out, -anp.inf)


def _between(lo: npt.NDArray, hi: npt.NDArray) -> tuple:
    """``(rows, lo, hi)``: the rows with a finite ``hi`` above ``lo``,
    and ``lo`` and ``hi`` made safe (``0`` and ``1``) elsewhere."""
    rows = np.isfinite(hi) & (hi > lo)
    return rows, np.where(rows, lo, 0.0), np.where(rows, hi, 1.0)


def log_cumhaz_between(
    theta: dict,
    path: PathFunction,
    params: Any,
    arrays: JointArrays,
    lo: npt.NDArray,
    hi: npt.NDArray,
    centre: float,
) -> Any:
    """``log (H(hi_i | theta) - H(lo_i | theta))`` at the nodes ``(N,
    K)``, for ``0 <= lo < hi`` (finite): the hazard integrated over
    ``(lo, hi]`` directly, with ``s = lo + (hi - lo) w^4``, by the same
    15-point Gauss-Legendre in ``w`` (with ``lo = 0`` this is the
    cumulative hazard's own rule), so a narrow interval late in life
    loses no precision to a difference of two cumulative hazards. With
    a lag (or a window on the accumulated exposure), split at
    ``path.cuts`` (:func:`lag_rule`)."""
    width = hi - lo
    if path.cuts:
        s, logw = lag_rule(lo, hi, path.cuts)
    else:
        s = lo[:, None] + width[:, None] * GL_U[None, :]
    eta = predictor(theta, path, params, s, centre)
    alpha, beta = theta["alpha"], theta["beta"]
    lin = anp.dot(arrays.Z, theta["gamma"])
    log_h = (
        anp.log(beta)
        - anp.log(alpha)
        + (beta - 1) * (np.log(s)[:, None, :] - anp.log(alpha))
        + lin[:, None, None]
        + eta
    )
    if path.cuts:
        return logsumexp(logw[:, None, :] + log_h, axis=-1)
    inner = logsumexp(GL_LOGW + 3 * np.log(GL_NODES) + log_h, axis=-1)
    return np.log(width)[:, None] + inner


def log_fails_between(
    theta: dict,
    path: PathFunction,
    params: Any,
    arrays: JointArrays,
    lo: npt.NDArray,
    hi: npt.NDArray,
    centre: float,
) -> Any:
    """``log (1 - S(hi | theta) / S(lo | theta))`` at the nodes ``(N,
    K)``: the probability of failing in ``(lo, hi]`` having reached
    ``lo``, so that with ``S(lo)`` it is ``S(lo) - S(hi)``; 0 for a row
    whose ``hi`` is ``inf`` (no upper end)."""
    rows, lo_s, hi_s = _between(lo, hi)
    if not rows.any():
        return anp.zeros(params.shape[:2])
    log_H = log_cumhaz_between(theta, path, params, arrays, lo_s, hi_s, centre)
    return anp.where(rows[:, None], log1mexp(log_H), 0.0)


#: ``log1mexp`` takes its series below ``H = 0.05``, and holds ``H`` at
#: 50 above (where ``1 - exp(-H)`` is 1 to 22 digits).
_SERIES_CUT = float(np.log(0.05))
_FLAT_CUT = float(np.log(50.0))


def log1mexp(log_H: Any) -> Any:
    """``log(1 - exp(-H))`` of ``H = exp(log_H)``, accurate and with a
    finite derivative everywhere: for a small ``H`` by the series of
    ``(1 - exp(-H)) / H``, which keeps an interval's probability right
    when it is tiny, and held flat for a large one."""
    small_x = anp.minimum(log_H, _SERIES_CUT)
    h = anp.exp(small_x)
    small = small_x + anp.log1p(-h / 2 + h**2 / 6 - h**3 / 24 + h**4 / 120)
    big_x = anp.minimum(anp.maximum(log_H, _SERIES_CUT), _FLAT_CUT)
    big = anp.log(-anp.expm1(-anp.exp(big_x)))
    return anp.where(log_H < _SERIES_CUT, small, big)


def log_event(
    theta: dict,
    path: PathFunction,
    params: Any,
    arrays: JointArrays,
    centre: float,
) -> Any:
    """``log h(T_i | theta)`` at the nodes (at age 1 where ``T_i`` is 0,
    a left censored item's, which has no event there)."""
    T = np.where(arrays.T > 0, arrays.T, 1.0)
    eta = predictor(theta, path, params, T[:, None], centre)[..., 0]
    alpha, beta = theta["alpha"], theta["beta"]
    lin = anp.dot(arrays.Z, theta["gamma"])
    base = (
        anp.log(beta)
        - anp.log(alpha)
        + (beta - 1) * (anp.log(T) - anp.log(alpha))
        + lin
    )
    return base[:, None] + eta


# -- competing failure modes (causes) ---------------------------------------
#
# With several causes each has a hazard of its own (its own Weibull
# baseline, coefficients and effects; :func:`cause_thetas`), and an
# item's lifetime term is
#
#     prod_k h_k(T)^(d_k) exp(-sum_k H_k(T)):
#
# the hazard of the cause it failed by and the survival of every cause.
# An item found failed by cause k in ``(l, u]`` has, in place of the
# hazard, ``int_l^u h_k(s) exp(-(H(s) - H(l))) ds`` with ``H`` the sum of
# the causes' cumulative hazards: the cause's hazard and the survival of
# all of them to each age of the interval, which does not factor by
# cause.


def log_hazard(
    theta: dict,
    path: PathFunction,
    params: Any,
    arrays: JointArrays,
    s: npt.NDArray,
    centre: float,
) -> Any:
    """``log h(s | theta)`` at the ages ``s`` ``(N, A)`` (each above 0)
    and the nodes: ``(N, K, A)``."""
    eta = predictor(theta, path, params, s, centre)
    alpha, beta = theta["alpha"], theta["beta"]
    lin = anp.dot(arrays.Z, theta["gamma"])
    return (
        anp.log(beta)
        - anp.log(alpha)
        + (beta - 1) * (np.log(s)[:, None, :] - anp.log(alpha))
        + lin[:, None, None]
        + eta
    )


def log_event_of(
    thetas: list, path: PathFunction, params: Any, arrays: JointArrays, c: Any
) -> Any:
    """The log of each item's hazard at its age ``T``, at the nodes
    ``(N, K)``: its cause's, and for an item with no cause (``e`` -1) the
    sum of the causes' (the hazard of failing at all)."""
    each = [log_event(th, path, params, arrays, c) for th in thetas]
    out = logsumexp(anp.stack(each), axis=0)
    for k, log_h in enumerate(each):
        out = anp.where((arrays.e == k)[:, None], log_h, out)
    return out


def log_cause_fails_between(
    thetas: list,
    path: PathFunction,
    params: Any,
    arrays: JointArrays,
    lo: npt.NDArray,
    hi: npt.NDArray,
    centre: float,
) -> Any:
    """The log of the probability, at the nodes ``(N, K)``, that an item
    which reached ``lo`` fails in ``(lo, hi]``, by its cause where it
    has one (``arrays.e``): ``int_lo^hi h_k(s) exp(-(H(s) - H(lo)))
    ds``, by the cumulative hazard's rule over the interval (``s = lo +
    (hi - lo) w^4``, 15-point Gauss-Legendre in ``w``, split at the
    path's cuts), the causes' cumulative hazards from ``lo`` to each of
    its ages by the same rule. An item with no cause (-1) fails by any:
    ``1 - exp(-(H(hi) - H(lo)))``. 0 for a row whose ``hi`` is ``inf``
    (no upper end)."""
    rows, lo_s, hi_s = _between(lo, hi)
    if not rows.any():
        return anp.zeros(params.shape[:2])
    total = logsumexp(
        anp.stack(
            [
                log_cumhaz_between(
                    th, path, params, arrays, lo_s, hi_s, centre
                )
                for th in thetas
            ]
        ),
        axis=0,
    )
    out = anp.where(rows[:, None], log1mexp(total), 0.0)
    known = np.flatnonzero(rows & (arrays.e >= 0))
    if not known.size:
        return out
    sub = arrays.take(known)
    sub_params = params[known]
    a, b = lo_s[known], hi_s[known]
    if path.cuts:
        s, logw = lag_rule(a, b, path.cuts)
    else:
        s = a[:, None] + (b - a)[:, None] * GL_U[None, :]
        logw = np.log(b - a)[:, None] + (GL_LOGW + 3 * np.log(GL_NODES))
    n, A = s.shape
    # every age of every row's rule as a row of its own
    again = np.repeat(np.arange(n), A)
    inner = sub.take(again)
    inner_params = sub_params[again]
    start = np.repeat(a, A)
    ends = s.ravel()
    # a piece of no width (a cut at an end) has no age above the start
    live = ends > start
    ends_safe = np.where(live, ends, start + 1.0)
    H_to = sum(
        anp.exp(
            log_cumhaz_between(
                th, path, inner_params, inner, start, ends_safe, centre
            )
        )
        for th in thetas
    )
    # (n A, K) -> (n, K, A)
    H_to = anp.transpose(anp.reshape(H_to, (n, A, params.shape[1])), (0, 2, 1))
    log_h: Any = 0.0
    for k, th in enumerate(thetas):
        own = (sub.e == k).astype(float)[:, None, None]
        log_h = log_h + own * log_hazard(th, path, sub_params, sub, s, centre)
    with np.errstate(divide="ignore"):
        logw = np.where(live.reshape(n, A), logw, -np.inf)
    found = logsumexp(logw[:, None, :] + log_h - H_to, axis=-1)
    # back among the items
    place_ = np.zeros((len(arrays.w), n))
    place_[known, np.arange(n)] = 1.0
    is_known = np.zeros(len(arrays.w), dtype=bool)
    is_known[known] = True
    return anp.where(is_known[:, None], anp.dot(place_, found), out)


def lifetime_functions(layout: Layout) -> tuple:
    """``(log_cumhaz, log_event, log_fails_between)`` of the layout's
    survival part: this module's (proportional hazards), or the
    accelerated part's (:mod:`._aft`), which take the same arguments."""
    if layout.survival == "AFT":
        from . import _aft

        return _aft.log_cumhaz, _aft.log_event, _aft.log_fails_between
    return log_cumhaz, log_event, log_fails_between


# -- where the nodes go -------------------------------------------------------


def _mode_objective(
    theta: dict,
    layout: Layout,
    path: Any,
    arrays: JointArrays,
    b: npt.NDArray,
    d: npt.NDArray,
    t: npt.NDArray,
    centre: float,
    readings: bool,
    upper: npt.NDArray | None = None,
    prior: bool = True,
) -> tuple:
    """``-log`` of the integrand at the varying parameters ``b`` ``(N,
    q)``, its gradient and its Gauss-Newton curvature (``prior``
    false: without the prior's terms). With ``upper``,
    a row with a finite one fails in ``(t, upper]``: its lifetime term is
    ``S(t) - S(upper)``, whose curvature is taken as that of ``-log(1 -
    exp(-Delta))`` in the interval's cumulative hazard ``Delta`` along
    the first derivatives (a Gauss-Newton term, an upper bound on it)."""
    params = np.asarray(
        _path_params(theta, layout, arrays.Z, b[:, None, :]), dtype=float
    )[:, 0, :]
    V = list(layout.varying)
    q = layout.q
    zeta = np.asarray(theta["zeta"], dtype=float).reshape(-1)

    def followed(ages: npt.NDArray) -> list:
        """``(effect, value, Jacobian in the varying parameters)`` of
        each link at ``ages`` ``(N, A)``."""
        P = np.broadcast_to(params[:, None, :], ages.shape + (layout.P,))
        out = []
        for k, (_, model, _) in enumerate(path.terms):
            m = model.path(ages, *np.moveaxis(P, -1, 0))
            J = varying_jacobian(
                path_jacobian(model, ages, np.array(P)), layout, theta
            )
            out.append((zeta[k], m, J))
        return out

    def log_effect(terms: list) -> Any:
        """The marker's part of the log hazard (the value's centred)."""
        out: Any = 0.0
        for (link, _, _), (effect, m, _) in zip(path.terms, terms):
            out = out + effect * ((m - centre) if link == "value" else m)
        return out

    def weighted(lam: npt.NDArray, terms: list) -> tuple:
        """The gradient and Gauss-Newton curvature of ``sum lam`` in
        the varying parameters, through the links."""
        grad: Any = 0.0
        curve: Any = 0.0
        for effect, _, J in terms:
            grad = grad + effect * np.einsum("nq,nqj->nj", lam, J)
            for other, _, J2 in terms:
                curve = curve + effect * other * np.einsum(
                    "nq,nqi,nqj->nij", lam, J, J2
                )
        return grad, curve

    with np.errstate(all="ignore"):
        f = np.zeros(len(b))
        g = np.zeros((len(b), q))
        H = np.zeros((len(b), q, q))
        if readings and several_markers(path):
            s2m = np.asarray(path.noise(theta["sigma"], arrays)) ** 2
            m, J = path.reading_jacobians(params, arrays)
            J = varying_jacobian(J, layout, theta)
            r = np.where(arrays.mask, arrays.values - m, 0.0)
            J = np.where(arrays.mask[..., None], J, 0.0)
            f = f + np.sum(r**2 / (2 * s2m), axis=1)
            g = g - np.einsum("nr,nrj->nj", r / s2m, J)
            H = H + np.einsum("nri,nrj->nij", J / s2m[:, None], J)
        elif readings:
            s2 = float(theta["sigma"]) ** 2
            ages = arrays.ages
            P = np.broadcast_to(params[:, None, :], ages.shape + (layout.P,))
            m = path.model.path(ages, *np.moveaxis(P, -1, 0))
            J = varying_jacobian(
                path_jacobian(path.model, ages, np.array(P)), layout, theta
            )
            r = np.where(arrays.mask, arrays.values - m, 0.0)
            J = np.where(arrays.mask[..., None], J, 0.0)
            f = f + np.sum(r**2, axis=1) / (2 * s2)
            g = g - np.einsum("nr,nrj->nj", r, J) / s2
            H = H + np.einsum("nri,nrj->nij", J, J) / s2
        if layout.survival == "AFT":
            from ._aft import mode_terms

            life = mode_terms(
                theta,
                arrays,
                d,
                t,
                upper,
                followed,
                log_effect,
                weighted,
                lag=path.cuts if path.cuts else 0.0,
            )
            f, g, H = f + life[0], g + life[1], H + life[2]
        elif layout.n_causes > 1:
            life = _cause_mode_terms(
                theta,
                layout,
                path,
                arrays,
                d,
                t,
                upper,
                (followed, log_effect, weighted),
            )
            f, g, H = f + life[0], g + life[1], H + life[2]
        else:
            # the lifetime: an event flag d at age t, survival to t
            positive = t > 0
            safe = np.where(positive, t, 1.0)
            beta = float(theta["beta"])
            lin = arrays.Z @ np.asarray(theta["gamma"], dtype=float)
            if path.cuts:
                # from age 0, split at the lag (as log_cumhaz)
                s_ages, logw = lag_rule(np.zeros(safe.size), safe, path.cuts)
                terms = followed(s_ages)
                alpha = float(theta["alpha"])
                log_lam = (
                    (lin + np.log(beta / alpha))[:, None]
                    + (beta - 1) * (np.log(s_ages) - np.log(alpha))
                    + logw
                    + log_effect(terms)
                )
            else:
                s_ages = safe[:, None] * GL_U[None, :]
                terms = followed(s_ages)
                log_lam = (
                    (
                        lin
                        + beta * (np.log(safe) - np.log(float(theta["alpha"])))
                    )[:, None]
                    + np.log(beta)
                    + GL_LOGW
                    + (4 * beta - 1) * np.log(GL_NODES)
                    + log_effect(terms)
                )
            lam = np.where(positive[:, None], np.exp(log_lam), 0.0)
            f = f + lam.sum(axis=1)
            grad, curve = weighted(lam, terms)
            g = g + grad
            H = H + curve
            if np.any(d > 0):
                for effect, mT, JT in followed(safe[:, None]):
                    f = f - d * effect * mT[:, 0]
                    g = g - (d * effect)[:, None] * JT[:, 0]
            if upper is not None:
                rows, lo, hi = _between(t, upper)
                if rows.any():
                    width = hi - lo
                    alpha = float(theta["alpha"])
                    if path.cuts:
                        s_ages, logw = lag_rule(lo, hi, path.cuts)
                        terms = followed(s_ages)
                        log_lam = (
                            (lin + np.log(beta / alpha))[:, None]
                            + (beta - 1) * (np.log(s_ages) - np.log(alpha))
                            + logw
                            + log_effect(terms)
                        )
                    else:
                        s_ages = lo[:, None] + width[:, None] * GL_U[None, :]
                        terms = followed(s_ages)
                        log_lam = (
                            (lin + np.log(width) + np.log(beta / alpha))[
                                :, None
                            ]
                            + (beta - 1) * (np.log(s_ages) - np.log(alpha))
                            + GL_LOGW
                            + 3 * np.log(GL_NODES)
                            + log_effect(terms)
                        )
                    lam = np.where(rows[:, None], np.exp(log_lam), 0.0)
                    delta = np.where(rows, lam.sum(axis=1), 1.0)
                    g_delta, _ = weighted(lam, terms)
                    small = -np.expm1(-delta)
                    f = f - np.where(rows, np.log(small), 0.0)
                    g = (
                        g
                        - np.where(rows, 1.0 / np.expm1(delta), 0.0)[:, None]
                        * g_delta
                    )
                    curve = np.where(rows, np.exp(-delta) / small**2, 0.0)
                    H = H + curve[:, None, None] * np.einsum(
                        "ni,nj->nij", g_delta, g_delta
                    )
        if not prior:
            return f, g, H
        # the prior
        Lc = np.asarray(theta["L"], dtype=float)
        Dinv = np.linalg.inv(Lc @ Lc.T)
        mean = (
            np.asarray(theta["mu"], dtype=float)[V][None, :]
            + arrays.Z @ np.asarray(theta["Delta"], dtype=float)[V].T
        )
        diff = b - mean
        f = f + 0.5 * np.einsum("ni,ij,nj->n", diff, Dinv, diff)
        g = g + diff @ Dinv
        H = H + Dinv[None]
    return f, g, H


def _cause_mode_terms(
    theta: dict,
    layout: Layout,
    path: PathFunction,
    arrays: JointArrays,
    d: npt.NDArray,
    t: npt.NDArray,
    upper: npt.NDArray | None,
    helpers: tuple,
) -> tuple:
    """The lifetime's part of :func:`_mode_objective` with several
    causes: ``(f, g, H)``. Every cause's cumulative hazard to ``t``, the
    hazard of the item's own cause at ``t`` (of any, for an item with
    none), and for a row that fails in ``(t, upper]`` the term ``(1 -
    exp(-Delta)) Delta_k / Delta`` with ``Delta`` the interval's
    cumulative hazard and ``Delta_k`` its cause's (all of it for an
    item with no cause): the interval's probability where the causes'
    hazards are in proportion over it, which places the nodes; the
    integral itself is :func:`log_cause_fails_between`'s. The curvature
    is the cumulative hazards' and that of ``-log(1 - exp(-Delta))``,
    as with one cause."""
    followed, log_effect, weighted = helpers
    thetas = cause_thetas(theta, layout)
    N, q = len(arrays.w), layout.q
    positive = t > 0
    safe = np.where(positive, t, 1.0)
    cause = np.asarray(arrays.e, dtype=int)

    def effects(raw: list, th: dict) -> list:
        zeta = np.asarray(th["zeta"], dtype=float).reshape(-1)
        return [(zeta[j], m, J) for j, (_, m, J) in enumerate(raw)]

    def log_base(th: dict, s_ages: npt.NDArray, logw: npt.NDArray) -> Any:
        """The log of the baseline hazard at the ages times the rule's
        weights, the covariates' part included."""
        alpha, beta = float(th["alpha"]), float(th["beta"])
        lin = arrays.Z @ np.asarray(th["gamma"], dtype=float)
        return (
            (lin + np.log(beta / alpha))[:, None]
            + (beta - 1) * (np.log(s_ages) - np.log(alpha))
            + logw
        )

    def rule(lo: npt.NDArray, hi: npt.NDArray) -> tuple:
        if path.cuts:
            return lag_rule(lo, hi, path.cuts)
        width = hi - lo
        return (
            lo[:, None] + width[:, None] * GL_U[None, :],
            np.log(width)[:, None] + (GL_LOGW + 3 * np.log(GL_NODES)),
        )

    f = np.zeros(N)
    g = np.zeros((N, q))
    H = np.zeros((N, q, q))
    s_ages, logw = rule(np.zeros(N), safe)
    raw = followed(s_ages)
    at_T = followed(safe[:, None])
    failed = d > 0
    log_h, grads = [], []
    for k, th in enumerate(thetas):
        terms = effects(raw, th)
        lam = np.where(
            positive[:, None],
            np.exp(log_base(th, s_ages, logw) + log_effect(terms)),
            0.0,
        )
        f = f + lam.sum(axis=1)
        grad, curve = weighted(lam, terms)
        g = g + grad
        H = H + curve
        if failed.any():
            terms_T = effects(at_T, th)
            log_h.append(
                log_base(th, safe[:, None], np.zeros((N, 1)))[:, 0]
                + log_effect(terms_T)[:, 0]
            )
            grads.append(sum(effect * JT[:, 0] for effect, _, JT in terms_T))
    if failed.any():
        stacked = np.stack(log_h)
        top = stacked.max(axis=0)
        share = np.exp(stacked - top[None])
        total = share.sum(axis=0)
        # an item with no cause: the hazard of failing at all
        log_own = top + np.log(total)
        g_own = sum(
            (share[k] / total)[:, None] * grads[k] for k in range(len(thetas))
        )
        for k in range(len(thetas)):
            mine = cause == k
            log_own = np.where(mine, stacked[k], log_own)
            g_own = np.where(mine[:, None], grads[k], g_own)
        f = f - np.where(failed, d * log_own, 0.0)
        g = g - np.where(failed, d, 0.0)[:, None] * g_own
    if upper is not None:
        rows, lo, hi = _between(t, upper)
        if rows.any():
            s_ages, logw = rule(lo, hi)
            raw = followed(s_ages)
            deltas, g_deltas = [], []
            for th in thetas:
                terms = effects(raw, th)
                lam = np.where(
                    rows[:, None],
                    np.exp(log_base(th, s_ages, logw) + log_effect(terms)),
                    0.0,
                )
                deltas.append(lam.sum(axis=1))
                g_deltas.append(weighted(lam, terms)[0])
            delta = np.where(rows, sum(deltas), 1.0)
            g_delta = sum(g_deltas)
            small = -np.expm1(-delta)
            f = f - np.where(rows, np.log(small), 0.0)
            g = (
                g
                - np.where(rows, 1.0 / np.expm1(delta), 0.0)[:, None] * g_delta
            )
            curve = np.where(rows, np.exp(-delta) / small**2, 0.0)
            H = H + curve[:, None, None] * np.einsum(
                "ni,nj->nij", g_delta, g_delta
            )
            for k in range(len(thetas)):
                mine = rows & (cause == k)
                if not mine.any():
                    continue
                own = np.where(mine, deltas[k], 1.0)
                f = f - np.where(mine, np.log(own) - np.log(delta), 0.0)
                g = g - np.where(mine, 1.0, 0.0)[:, None] * (
                    g_deltas[k] / own[:, None] - g_delta / delta[:, None]
                )
    return f, g, H


def numeric_theta(theta: dict) -> dict:
    """``theta`` with plain arrays (no autograd boxes)."""
    return {
        k: np.asarray(getattr(v, "_value", v), dtype=float)
        for k, v in theta.items()
    }


def _prior_scale(theta: dict) -> npt.NDArray:
    """The prior spread of each varying parameter, ``sqrt(diag D)``,
    the scale the modes are searched and their curvature inverted in
    (1 where it is not positive and finite)."""
    Lc = np.asarray(theta["L"], dtype=float)
    scale = np.sqrt(np.sum(Lc**2, axis=1)) if Lc.size else np.ones(0)
    return np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)


def modes(
    theta: dict,
    layout: Layout,
    path: PathFunction,
    arrays: JointArrays,
    d: npt.NDArray,
    t: npt.NDArray,
    centre: float,
    readings: bool = True,
    start: npt.NDArray | None = None,
    upper: npt.NDArray | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Each item's mode of the integrand over its varying parameters and
    the Cholesky factor of the inverse of its (Gauss-Newton) curvature
    there, by damped Gauss-Newton steps (``upper``: as for
    :func:`_mode_objective`)."""
    theta = numeric_theta(theta)
    V = list(layout.varying)
    N = len(arrays.w)
    scale = _prior_scale(theta)
    if start is None:
        b = np.asarray(theta["mu"], dtype=float)[V][None, :] + (
            arrays.Z @ np.asarray(theta["Delta"], dtype=float)[V].T
        )
    else:
        b = np.array(start, dtype=float)
    f, g, H = _mode_objective(
        theta, layout, path, arrays, b, d, t, centre, readings, upper
    )
    for _ in range(200):
        with np.errstate(all="ignore"):
            step = -np.linalg.solve(H, g[..., None])[..., 0]
        step = np.where(np.isfinite(step), step, 0.0)
        size = np.ones(N)
        for _ in range(40):
            trial = b + size[:, None] * step
            f_new, g_new, H_new = _mode_objective(
                theta,
                layout,
                path,
                arrays,
                trial,
                d,
                t,
                centre,
                readings,
                upper,
            )
            worse = ~(f_new <= f + 1e-12 * np.abs(f))
            if not worse.any():
                break
            size = np.where(worse, size / 2, size)
        accept = ~worse
        b = np.where(accept[:, None], trial, b)
        f = np.where(accept, f_new, f)
        g = np.where(accept[:, None], g_new, g)
        H = np.where(accept[:, None, None], H_new, H)
        moved = np.abs(step * np.where(accept, size, 0.0)[:, None])
        if np.all(moved <= 1e-10 * (scale + np.abs(b))):
            break
    # A row whose cumulative hazard overflows (an age so far beyond the
    # data that the item cannot have survived to it whatever its path:
    # its survival there is 0) has no finite curvature: its nodes are
    # placed by the prior's alone, and its integrand is 0 on them.
    bad = ~np.isfinite(H).all(axis=(-2, -1))
    if bad.any():
        Lc = np.asarray(theta["L"], dtype=float)
        prior = np.linalg.inv(Lc @ Lc.T)
        H = np.where(bad[:, None, None], prior[None], H)
    # The curvature in each varying parameter's own scale (its prior
    # spread), so that neither the inverse nor the floor depends on the
    # unit of time: a slope a second is ten million times smaller than
    # one an hour, and its curvature that squared larger.
    H = H * np.outer(scale, scale)[None]
    with np.errstate(all="ignore"):
        cov = np.linalg.pinv(H)
    cov = 0.5 * (cov + np.swapaxes(cov, -1, -2))
    if not np.all(np.isfinite(cov)):
        raise np.linalg.LinAlgError("the integrand's curvature is not finite")
    # floored, so the grid has a scale in every direction
    vals, vecs = np.linalg.eigh(cov)
    floor = np.maximum(vals.max(axis=-1, keepdims=True), 1e-300) * 1e-12
    floored = np.any(vals < floor, axis=-1) & ~bad
    vals = np.maximum(vals, floor)
    root = np.einsum("nij,nj->nij", vecs, np.sqrt(vals))
    cov = np.einsum("nij,nkj->nik", root, root)
    root = scale[None, :, None] * np.linalg.cholesky(cov)
    if floored.any():
        # The floor widens the grid in a direction the integrand is
        # far narrower in, and the grid's sum is then wrong (by 1.8 in
        # one item's log-likelihood on a fixture with a level, a rate
        # and a curvature varying, near a correlation of +-1). There the
        # curvature is too ill-conditioned to invert as it is: it is
        # taken in the prior's own coordinates instead.
        root[floored] = _whitened_root(
            theta,
            layout,
            path,
            arrays.take(floored),
            b[floored],
            d[floored],
            t[floored],
            centre,
            readings,
            None if upper is None else upper[floored],
            root[floored],
        )
    return b, root


def _whitened_root(
    theta: dict,
    layout: Layout,
    path: Any,
    arrays: JointArrays,
    b: npt.NDArray,
    d: npt.NDArray,
    t: npt.NDArray,
    centre: float,
    readings: bool,
    upper: npt.NDArray | None,
    fallback: npt.NDArray,
) -> npt.NDArray:
    """The lower Cholesky factor ``(N, q, q)`` of the inverse curvature
    of the integrand at its modes ``b`` taken in the prior's own
    coordinates: with ``L`` the prior's factor, the curvature there is
    ``I + L' H L`` (``H`` the readings' and the lifetime's, without the
    prior's), well conditioned however near singular the prior is; its
    inverse in the parameters is ``A A'`` with ``A = L C^-T`` (``C C'``
    its factorisation), whose lower factor is the transpose of the
    triangle of the QR factorisation of ``A'``: no matrix is formed
    that squares ``L``'s condition number. ``fallback`` where the
    curvature there is not positive definite."""
    _, _, H = _mode_objective(
        theta, layout, path, arrays, b, d, t, centre, readings, upper, False
    )
    Lc = np.asarray(theta["L"], dtype=float)
    out = np.array(fallback, dtype=float)
    q = Lc.shape[0]
    for n in range(len(b)):
        Hw = Lc.T @ H[n] @ Lc + np.eye(q)
        try:
            C = np.linalg.cholesky(0.5 * (Hw + Hw.T))
        except np.linalg.LinAlgError:
            continue
        A = Lc @ np.linalg.inv(C).T
        R = np.linalg.qr(A.T, mode="r")
        lower = R.T * np.sign(np.diag(R))[None, :]
        if np.all(np.isfinite(lower)) and np.all(np.diag(lower) > 0):
            out[n] = lower
    return out


def continued_modes(
    theta: dict,
    layout: Layout,
    path: PathFunction,
    arrays: JointArrays,
    t: npt.NDArray,
    centre: float,
) -> tuple[npt.NDArray, npt.NDArray]:
    """:func:`modes` of items censored at ``t`` (each row at its own age),
    by continuation in the age: from the smallest age, at most doubled at
    each stage, each search started from the last stage's modes. Far
    beyond the data the mode is where the hazard stays small, which a
    search from the modes of nearer ages reaches; from the readings
    alone, the cumulative hazard there overflows."""
    t = np.asarray(t, dtype=float)
    smallest = np.min(t[t > 0]) if np.any(t > 0) else 1.0
    current = np.minimum(t, smallest)
    zero = np.zeros(t.size)
    b, root = modes(theta, layout, path, arrays, zero, current, centre)
    while np.any(current < t):
        current = np.minimum(t, 2.0 * current)
        b, root = modes(
            theta, layout, path, arrays, zero, current, centre, start=b
        )
    return b, root


def adaptive_nodes(
    mode: npt.NDArray, root: npt.NDArray, points: int, prune: bool = False
) -> tuple[npt.NDArray, npt.NDArray]:
    """The nodes ``(N, K, q)`` of each item's adaptive grid and their log
    weights ``(N, K)``, the change of variables included: ``int f db ~
    sum_k exp(logw_k) f(b_k)``. ``prune``: without the nodes of
    negligible weight (:func:`gh_grid`), so ``K`` is below ``points **
    q``; nothing downstream needs a full product."""
    q = mode.shape[1]
    z, logw = gh_grid(q, points, prune)
    nodes = mode[:, None, :] + np.sqrt(2.0) * np.einsum("nij,kj->nki", root, z)
    log_det = q * 0.5 * np.log(2.0) + np.sum(
        np.log(np.diagonal(root, axis1=1, axis2=2)), axis=1
    )
    return (
        nodes,
        logw[None, :] + np.sum(z**2, axis=1)[None, :] + log_det[:, None],
    )


# -- the likelihood --------------------------------------------------------


@dataclass
class Model:
    """What the likelihood needs besides the parameters: the layout, the
    path, the marker's centre ``c`` in the hazard, and how the integral
    over the varying parameters is done (``INTEGRATIONS``)."""

    layout: Layout
    #: A :class:`PathFunction`, or several markers' :class:`MarkerPaths`.
    path: Any
    centre: float
    integration: str = "grid"

    @property
    def prune(self) -> bool:
        return self.integration == "pruned"

    @property
    def laplace(self) -> bool:
        return self.integration == "laplace"

    def prediction_nodes(
        self,
        mode: npt.NDArray,
        root: npt.NDArray,
        theta: dict,
        arrays: JointArrays,
        t: npt.NDArray,
    ) -> Any:
        """The nodes of a prediction's integrals, of items still running
        at the ages ``t`` with their readings (their ``mode`` and
        ``root`` at ``theta``, :func:`continued_modes`'): the adaptive
        grid (``GH_PREDICT`` points a dimension, pruned as the fit's
        were) wherever there is one, a Laplace fit's too (one item's
        grid is cheap, and exact for an item with few readings or
        none); beyond ``MAX_VARYING`` varying parameters, Laplace's
        stencil (:mod:`._laplace`)."""
        if self.layout.q in GH_PREDICT:
            return adaptive_nodes(
                mode, root, GH_PREDICT[self.layout.q], self.prune
            )
        from ._laplace import laplace_nodes

        return laplace_nodes(
            theta, self, arrays, mode, root, t, np.zeros(len(t))
        )


def log_integrals(
    theta: dict,
    model: Model,
    arrays: JointArrays,
    nodes: tuple,
    t: npt.NDArray,
    d: npt.NDArray,
    readings: bool,
    upper: npt.NDArray | None = None,
) -> Any:
    """``log int p(y | theta)^readings h(t | theta)^d S(t | theta)
    phi(theta_V) d theta_V`` of each item, by the fixed nodes ``(b,
    logw)``; with ``upper``, a row with a finite one has ``S(t) -
    S(upper)`` (it failed in ``(t, upper]``) in place of ``S(t)``. Nodes
    that are a Laplace stencil give Laplace's approximation
    (:func:`._laplace.log_integrals`)."""
    if not isinstance(nodes, tuple):
        from . import _laplace

        return _laplace.log_integrals(
            theta, model, arrays, nodes, t, d, readings, upper
        )
    return logsumexp(
        log_integrand(theta, model, arrays, nodes, t, d, readings, upper),
        axis=1,
    )


def log_integrand(
    theta: dict,
    model: Model,
    arrays: JointArrays,
    nodes: tuple,
    t: npt.NDArray,
    d: npt.NDArray,
    readings: bool,
    upper: npt.NDArray | None = None,
) -> Any:
    """The terms ``(N, K)`` of :func:`log_integrals`' sums: the log of
    the integrand at each node plus the node's log weight."""
    b, logw = nodes
    layout, path, c = model.layout, model.path, model.centre
    cumhaz, event, fails_between = lifetime_functions(layout)
    params = _path_params(theta, layout, arrays.Z, b)
    log_g = log_prior(theta, layout, arrays.Z, b) + logw
    if readings:
        log_g = log_g + log_readings(theta, path, params, arrays)
    if layout.n_causes > 1:
        thetas = cause_thetas(theta, layout)
        for th in thetas:
            log_g = log_g - anp.exp(cumhaz(th, path, params, arrays, t, c))
        if np.any(d > 0):
            log_g = log_g + d[:, None] * log_event_of(
                thetas, path, params, arrays, c
            )
        if upper is not None and np.any(np.isfinite(upper)):
            log_g = log_g + log_cause_fails_between(
                thetas, path, params, arrays, t, upper, c
            )
        return log_g
    H = anp.exp(cumhaz(theta, path, params, arrays, t, c))
    log_g = log_g - H
    if np.any(d > 0):
        log_g = log_g + d[:, None] * event(theta, path, params, arrays, c)
    if upper is not None and np.any(np.isfinite(upper)):
        log_g = log_g + fails_between(theta, path, params, arrays, t, upper, c)
    return log_g


def place(
    u: npt.NDArray, model: Model, arrays: JointArrays, starts: tuple
) -> tuple:
    """The adaptive nodes of every item at ``u``, for its likelihood and
    (truncated items) its probability of being in the data -- of reaching
    its entry age and, if right truncated, failing by its right
    truncation -- and the modes to start the next placement from."""
    theta = model.layout.unpack(np.asarray(u, dtype=float))
    lik_start, entry_start = starts
    mode, root = modes(
        theta,
        model.layout,
        model.path,
        arrays,
        arrays.d,
        arrays.T,
        model.centre,
        start=lik_start,
        upper=arrays.U,
    )
    if model.laplace:
        if arrays.truncated.any():
            raise ValueError(
                "Laplace's approximation does not take truncated items "
                "(the fit refuses them): use integration='grid'."
            )
        from ._laplace import laplace_nodes

        stencils = laplace_nodes(
            theta,
            model,
            arrays,
            mode,
            root,
            arrays.T,
            arrays.d,
            True,
            arrays.U,
        )
        return (stencils, None), (mode, None)
    nodes = adaptive_nodes(mode, root, GH_FIT, model.prune)
    entry = None
    entered = arrays.truncated
    e_mode = None
    if entered.any():
        sub = arrays.take(entered).of_any_cause()
        e_mode, e_root = modes(
            theta,
            model.layout,
            model.path,
            sub,
            np.zeros(len(sub.w)),
            sub.L,
            model.centre,
            readings=False,
            start=entry_start,
            upper=sub.Rt,
        )
        entry = adaptive_nodes(e_mode, e_root, GH_FIT, model.prune)
    return (nodes, entry), (mode, e_mode)


def neg_ll(u: Any, model: Model, arrays: JointArrays, placed: tuple) -> Any:
    """The negative log-likelihood at ``u`` by the fixed nodes."""
    theta = model.layout.unpack(u)
    nodes, entry = placed
    ll = anp.sum(
        arrays.w
        * log_integrals(
            theta,
            model,
            arrays,
            nodes,
            arrays.T,
            arrays.d,
            readings=True,
            upper=arrays.U,
        )
    )
    entered = arrays.truncated
    if entry is not None:
        sub = arrays.take(entered).of_any_cause()
        ll = ll - anp.sum(
            sub.w
            * log_integrals(
                theta,
                model,
                sub,
                entry,
                sub.L,
                np.zeros(len(sub.w)),
                readings=False,
                upper=sub.Rt,
            )
        )
    return -ll
