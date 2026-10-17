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

Given ``theta_i`` the readings and the lifetime are independent, so an
item's likelihood is the integral over its varying parameters

    L_i = int p(y_i | theta) h(T_i | theta)^d_i S(T_i | theta)
          phi(theta_V; mu_V + Delta_V z_i, D) d theta_V,

evaluated by adaptive Gauss-Hermite quadrature: a product grid
(7 points a dimension) centred at the mode of the integrand and scaled
by its curvature there (Gauss-Newton: the path's Jacobian, which is the
exact curvature for a path linear in its parameters). The nodes are
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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from autograd.extend import defvjp, primitive
from autograd.scipy.special import logsumexp

#: Gauss-Hermite points a dimension: in the fit (adaptive; 2-3 fail on
#: sparse data), and in a prediction (by the number of varying path
#: parameters: 15, 15, then 9 for three).
GH_FIT = 7
GH_PREDICT = {1: 15, 2: 15, 3: 9}
#: Gauss-Legendre points of the cumulative hazard.
GL_POINTS = 15
#: The most path parameters that may vary: a full grid has 7^q nodes.
MAX_VARYING = 3


def gh_grid(dim: int, points: int) -> tuple[npt.NDArray, npt.NDArray]:
    """The ``dim``-dimensional product Gauss-Hermite grid: nodes
    ``(K, dim)`` and log weights ``(K,)`` for ``int exp(-|z|^2) f(z)
    dz``."""
    x, w = np.polynomial.hermite.hermgauss(points)
    grids = np.meshgrid(*([x] * dim), indexing="ij")
    z = np.stack([g.ravel() for g in grids], axis=-1)
    logw = np.zeros(z.shape[0])
    for g in np.meshgrid(*([np.log(w)] * dim), indexing="ij"):
        logw = logw + g.ravel()
    return z, logw


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


class PathFunction:
    """``path(params, x)`` of a path model for autograd: ``params``
    ``(..., p)`` and ``x`` ``(...)`` of one shape, its derivatives the
    path model's Jacobian and, for the second, central differences of
    that."""

    def __init__(self, path_model: Any) -> None:
        self.model = path_model

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
        self.values = values
        self.jac = jac

    def __call__(self, params: Any, x: Any) -> Any:
        return self.values(params, x)


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
    marker's effect ``zeta``.

    ``shifts`` ``(P, p)`` and ``hazard`` ``(p,)`` say which covariates
    enter which part (default every one in both): an entry that is off is
    held at 0, not estimated."""

    P: int
    p: int
    varying: tuple
    shifts: Any = None
    hazard: Any = None

    def __post_init__(self) -> None:
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

    @property
    def q(self) -> int:
        return len(self.varying)

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
    def size(self) -> int:
        q = self.q
        return (
            self.P + self.n_shifts + q * (q + 1) // 2 + 3 + self.n_hazard + 1
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
        q = self.q
        start = self.P + self.n_shifts + q * (q + 1) // 2 + 3
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
        lower = u[i : i + q * (q - 1) // 2]
        i += q * (q - 1) // 2
        rows = []
        k = 0
        for r in range(q):
            row = []
            for col in range(q):
                if col < r:
                    row.append(lower[k])
                    k += 1
                elif col == r:
                    row.append(diag[r])
                else:
                    row.append(0.0 * diag[r])
            rows.append(anp.stack(row))
        L = anp.stack(rows) if q else anp.zeros((0, 0))
        return {
            "mu": mu,
            "Delta": Delta,
            "L": L,
            "sigma": anp.exp(u[i]),
            "alpha": anp.exp(u[i + 1]),
            "beta": anp.exp(u[i + 2]),
            "gamma": self._gamma(u[i + 3 : i + 3 + n_hazard]),
            "zeta": u[i + 3 + n_hazard],
        }

    def pack(self, theta: dict) -> npt.NDArray:
        L = np.asarray(theta["L"], dtype=float)
        q = self.q
        lower = [L[r, c] for r in range(q) for c in range(r)]
        Delta = np.asarray(theta["Delta"], dtype=float).reshape(self.P, self.p)
        gamma = np.atleast_1d(np.asarray(theta["gamma"], dtype=float))
        return np.concatenate(
            [
                np.asarray(theta["mu"], dtype=float),
                Delta[self.shifts],
                np.log(np.diag(L)) if q else np.zeros(0),
                np.asarray(lower, dtype=float),
                [
                    np.log(theta["sigma"]),
                    np.log(theta["alpha"]),
                    np.log(theta["beta"]),
                ],
                gamma.reshape(self.p)[self.hazard],
                [float(theta["zeta"])],
            ]
        )


# -- the data --------------------------------------------------------------


@dataclass
class JointArrays:
    """The items: counts ``w``, the age ``T`` each left observation (the
    lower end of an interval, 0 for a left censored item observed from
    0), the event flag ``d``, the upper end ``U`` of an interval (``inf``
    where the item has none), the entry age ``L`` (0 for an item observed
    from 0) and the right truncation ``Rt`` (``inf`` for none), the
    centred covariates ``Z`` ``(N, p)``, and the readings padded to ``(N,
    R)``: ``ages``, ``values`` and their ``mask``."""

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
    ) -> JointArrays:
        N = len(ages)
        R = max([len(a) for a in ages] + [1])
        A = np.zeros((N, R))
        V = np.zeros((N, R))
        M = np.zeros((N, R), dtype=bool)
        for i, (a, v) in enumerate(zip(ages, values)):
            A[i, : len(a)] = a
            V[i, : len(v)] = v
            M[i, : len(a)] = True
        # padded ages repeat the first (or 0), so the path is defined there
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
        )

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


# -- the integrand ---------------------------------------------------------


def _path_params(theta: dict, layout: Layout, Z: Any, nodes: Any) -> Any:
    """Each item's full path parameters at the nodes ``(N, K, q)`` of its
    varying ones: ``(N, K, P)``, the common ones its mean."""
    means = theta["mu"][None, :] + anp.dot(Z, anp.transpose(theta["Delta"]))
    N, K = nodes.shape[0], nodes.shape[1]
    cols = []
    for j in range(layout.P):
        if j in layout.varying:
            cols.append(nodes[..., layout.varying.index(j)])
        else:
            cols.append(anp.broadcast_to(means[:, j : j + 1], (N, K)))
    return anp.stack(cols, axis=-1)


def _at(path: PathFunction, params: Any, ages: npt.NDArray) -> Any:
    """``path`` at ``ages`` ``(N, A)`` for parameters ``(N, K, P)``:
    ``(N, K, A)``."""
    N, K, P = params.shape
    A = ages.shape[1]
    full = anp.broadcast_to(params[:, :, None, :], (N, K, A, P))
    x = np.broadcast_to(ages[:, None, :], (N, K, A))
    return path(full, x)


def log_prior(theta: dict, layout: Layout, Z: Any, nodes: Any) -> Any:
    """``log phi(theta_V; mu_V + Delta_V z, D)`` at the nodes."""
    q = layout.q
    if q == 0:
        return anp.zeros(nodes.shape[:2])
    V = list(layout.varying)
    mean = theta["mu"][V][None, :] + anp.dot(
        Z, anp.transpose(theta["Delta"][V])
    )
    diff = nodes - mean[:, None, :]
    L = theta["L"]
    # forward substitution: L x = diff
    xs: list = []
    for r in range(q):
        acc = diff[..., r]
        for c in range(r):
            acc = acc - L[r, c] * xs[c]
        xs.append(acc / L[r, r])
    quad = sum(x_**2 for x_ in xs)
    log_det = anp.sum(anp.log(anp.stack([L[r, r] for r in range(q)])))
    return -0.5 * q * np.log(2 * np.pi) - log_det - 0.5 * quad


def log_readings(
    theta: dict, path: PathFunction, params: Any, arrays: JointArrays
) -> Any:
    """``log p(y_i | theta)`` at the nodes ``(N, K)``."""
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
    m = _at(path, params, safe[:, None] * GL_U[None, :])
    beta, zeta = theta["beta"], theta["zeta"]
    inner = logsumexp(
        GL_LOGW + (4 * beta - 1) * np.log(GL_NODES) + zeta * (m - centre),
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
    loses no precision to a difference of two cumulative hazards."""
    width = hi - lo
    s = lo[:, None] + width[:, None] * GL_U[None, :]
    m = _at(path, params, s)
    alpha, beta, zeta = theta["alpha"], theta["beta"], theta["zeta"]
    lin = anp.dot(arrays.Z, theta["gamma"])
    log_h = (
        anp.log(beta)
        - anp.log(alpha)
        + (beta - 1) * (np.log(s)[:, None, :] - anp.log(alpha))
        + lin[:, None, None]
        + zeta * (m - centre)
    )
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
    m = _at(path, params, T[:, None])[..., 0]
    alpha, beta = theta["alpha"], theta["beta"]
    lin = anp.dot(arrays.Z, theta["gamma"])
    base = (
        anp.log(beta)
        - anp.log(alpha)
        + (beta - 1) * (anp.log(T) - anp.log(alpha))
        + lin
    )
    return base[:, None] + theta["zeta"] * (m - centre)


# -- where the nodes go -------------------------------------------------------


def _mode_objective(
    theta: dict,
    layout: Layout,
    path: PathFunction,
    arrays: JointArrays,
    b: npt.NDArray,
    d: npt.NDArray,
    t: npt.NDArray,
    centre: float,
    readings: bool,
    upper: npt.NDArray | None = None,
) -> tuple:
    """``-log`` of the integrand at the varying parameters ``b`` ``(N,
    q)``, its gradient and its Gauss-Newton curvature. With ``upper``,
    a row with a finite one fails in ``(t, upper]``: its lifetime term is
    ``S(t) - S(upper)``, whose curvature is taken as that of ``-log(1 -
    exp(-Delta))`` in the interval's cumulative hazard ``Delta`` along
    the first derivatives (a Gauss-Newton term, an upper bound on it)."""
    params = np.asarray(
        _path_params(theta, layout, arrays.Z, b[:, None, :]), dtype=float
    )[:, 0, :]
    V = list(layout.varying)
    q = layout.q
    zeta = float(theta["zeta"])
    with np.errstate(all="ignore"):
        f = np.zeros(len(b))
        g = np.zeros((len(b), q))
        H = np.zeros((len(b), q, q))
        if readings:
            s2 = float(theta["sigma"]) ** 2
            ages = arrays.ages
            P = np.broadcast_to(params[:, None, :], ages.shape + (layout.P,))
            m = path.model.path(ages, *np.moveaxis(P, -1, 0))
            J = path_jacobian(path.model, ages, np.array(P))[..., V]
            r = np.where(arrays.mask, arrays.values - m, 0.0)
            J = np.where(arrays.mask[..., None], J, 0.0)
            f = f + np.sum(r**2, axis=1) / (2 * s2)
            g = g - np.einsum("nr,nrj->nj", r, J) / s2
            H = H + np.einsum("nri,nrj->nij", J, J) / s2
        # the lifetime: an event flag d at age t, survival to t
        positive = t > 0
        safe = np.where(positive, t, 1.0)
        s_ages = safe[:, None] * GL_U[None, :]
        P = np.broadcast_to(params[:, None, :], s_ages.shape + (layout.P,))
        m = path.model.path(s_ages, *np.moveaxis(P, -1, 0))
        J = path_jacobian(path.model, s_ages, np.array(P))[..., V]
        beta = float(theta["beta"])
        lin = arrays.Z @ np.asarray(theta["gamma"], dtype=float)
        log_lam = (
            (lin + beta * (np.log(safe) - np.log(float(theta["alpha"]))))[
                :, None
            ]
            + np.log(beta)
            + GL_LOGW
            + (4 * beta - 1) * np.log(GL_NODES)
            + zeta * (m - centre)
        )
        lam = np.where(positive[:, None], np.exp(log_lam), 0.0)
        f = f + lam.sum(axis=1)
        g = g + zeta * np.einsum("nq,nqj->nj", lam, J)
        H = H + zeta**2 * np.einsum("nq,nqi,nqj->nij", lam, J, J)
        if np.any(d > 0):
            P1 = params[:, None, :]
            mT = path.model.path(safe[:, None], *np.moveaxis(P1, -1, 0))[:, 0]
            JT = path_jacobian(path.model, safe[:, None], np.array(P1))[
                :, 0, V
            ]
            f = f - d * zeta * mT
            g = g - (d * zeta)[:, None] * JT
        if upper is not None:
            rows, lo, hi = _between(t, upper)
            if rows.any():
                width = hi - lo
                s_ages = lo[:, None] + width[:, None] * GL_U[None, :]
                P = np.broadcast_to(
                    params[:, None, :], s_ages.shape + (layout.P,)
                )
                m = path.model.path(s_ages, *np.moveaxis(P, -1, 0))
                J = path_jacobian(path.model, s_ages, np.array(P))[..., V]
                alpha = float(theta["alpha"])
                log_lam = (
                    (lin + np.log(width) + np.log(beta / alpha))[:, None]
                    + (beta - 1) * (np.log(s_ages) - np.log(alpha))
                    + GL_LOGW
                    + 3 * np.log(GL_NODES)
                    + zeta * (m - centre)
                )
                lam = np.where(rows[:, None], np.exp(log_lam), 0.0)
                delta = np.where(rows, lam.sum(axis=1), 1.0)
                g_delta = zeta * np.einsum("nq,nqj->nj", lam, J)
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


def numeric_theta(theta: dict) -> dict:
    """``theta`` with plain arrays (no autograd boxes)."""
    return {
        k: np.asarray(getattr(v, "_value", v), dtype=float)
        for k, v in theta.items()
    }


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
        if np.all(moved <= 1e-10 * (1 + np.abs(b))):
            break
    with np.errstate(all="ignore"):
        cov = np.linalg.pinv(H)
    cov = 0.5 * (cov + np.swapaxes(cov, -1, -2))
    if not np.all(np.isfinite(cov)):
        raise np.linalg.LinAlgError("the integrand's curvature is not finite")
    # floored, so the grid has a scale in every direction
    vals, vecs = np.linalg.eigh(cov)
    floor = np.maximum(vals.max(axis=-1, keepdims=True), 1e-300) * 1e-12
    vals = np.maximum(vals, floor)
    root = np.einsum("nij,nj->nij", vecs, np.sqrt(vals))
    cov = np.einsum("nij,nkj->nik", root, root)
    return b, np.linalg.cholesky(cov)


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
    mode: npt.NDArray, root: npt.NDArray, points: int
) -> tuple[npt.NDArray, npt.NDArray]:
    """The nodes ``(N, K, q)`` of each item's adaptive grid and their log
    weights ``(N, K)``, the change of variables included: ``int f db ~
    sum_k exp(logw_k) f(b_k)``."""
    q = mode.shape[1]
    z, logw = gh_grid(q, points)
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
    path, and the marker's centre ``c`` in the hazard."""

    layout: Layout
    path: PathFunction
    centre: float


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
    S(upper)`` (it failed in ``(t, upper]``) in place of ``S(t)``."""
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
    params = _path_params(theta, layout, arrays.Z, b)
    log_g = log_prior(theta, layout, arrays.Z, b) + logw
    if readings:
        log_g = log_g + log_readings(theta, path, params, arrays)
    H = anp.exp(log_cumhaz(theta, path, params, arrays, t, c))
    log_g = log_g - H
    if np.any(d > 0):
        log_g = log_g + d[:, None] * log_event(theta, path, params, arrays, c)
    if upper is not None and np.any(np.isfinite(upper)):
        log_g = log_g + log_fails_between(
            theta, path, params, arrays, t, upper, c
        )
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
    nodes = adaptive_nodes(mode, root, GH_FIT)
    entry = None
    entered = arrays.truncated
    e_mode = None
    if entered.any():
        sub = arrays.take(entered)
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
        entry = adaptive_nodes(e_mode, e_root, GH_FIT)
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
        sub = arrays.take(entered)
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
