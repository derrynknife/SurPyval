"""The joint likelihood of a marker and a lifetime, by adaptive
Gauss-Hermite quadrature over each item's random effects (#891).

Item ``i``'s marker is a straight line in age with its own intercept and
slope, read with normal noise:

    m_i(t) = (b0 + u_i0) + (b1 + u_i1) t + delta'z_i,
    y_ij = m_i(t_ij) + e_ij,  e_ij ~ N(0, sigma^2),  u_i ~ N(0, D),

and its hazard is Weibull proportional hazards in the line's current
value and the covariates,

    h_i(t) = h0(t) exp(gamma'z_i + a m_i(t)).

Everything here works on an internal scale on which the problem is well
conditioned -- ages divided by a time scale, the marker centred and
divided by its spread, the covariates centred (``joint_ph`` makes the
scales and maps the parameters back) -- so the marker's centre is 0 here.

Given ``u_i`` the readings and the lifetime are independent, so an item's
likelihood is the two-dimensional integral

    L_i = int p(y_i | u) p(T_i, d_i | u) phi(u; 0, D) du,

which adaptive Gauss-Hermite quadrature evaluates on a grid of nodes
centred at the mode of the integrand and scaled by its curvature there.
The nodes are held fixed while the parameters are searched, so the
approximation is a smooth function of the parameters, differentiated by
autograd (the gradient is then the posterior-weighted complete-data
score), and placed again at the new parameters between searches.

An item that came under observation at ``L_i > 0`` (left truncation) is
in the data because it reached ``L_i``: its likelihood is divided by the
probability of that, ``int S(L_i | u) phi(u; 0, D) du``, over the
population's random effects (the item's own are not yet selected at
entry). That integrand is bounded and smooth, so plain Gauss-Hermite
over the prior evaluates it.

The cumulative hazard ``H(t | u) = int_0^t h(s | u) ds`` is, with
``s = t v^(1/beta)`` and ``v = w^3``,

    H = exp(gamma'z + a A) (t / alpha)^beta
        int_0^1 3 w^2 exp(a B t w^(3/beta)) dw,

``A`` and ``B`` the line's intercept and slope: 15-point Gauss-Legendre
in ``w`` is accurate to about 1e-5 relative even where the hazard grows
by a factor of e^8 over the life (the substitution takes out the
singularity of the Weibull hazard at 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from autograd.scipy.special import logsumexp

#: Gauss-Hermite points per dimension of the random effects: in the fit
#: (adaptive; 2-3 fail on sparse data), in a prediction, and for the
#: probability of reaching the entry age (over the prior, not adaptive).
GH_FIT = 7
GH_PREDICT = 15
GH_ENTRY = 10
#: Gauss-Legendre points of the cumulative hazard.
GL_POINTS = 15


def _gh_grid(points: int) -> tuple[npt.NDArray, npt.NDArray]:
    """The two-dimensional product Gauss-Hermite grid: nodes ``(K, 2)``
    and log weights ``(K,)`` for ``int exp(-|z|^2) f(z) dz``."""
    x, w = np.polynomial.hermite.hermgauss(points)
    z = np.array(np.meshgrid(x, x, indexing="ij")).reshape(2, -1).T
    logw = np.add.outer(np.log(w), np.log(w)).ravel()
    return z, logw


def _gl_grid() -> tuple[npt.NDArray, npt.NDArray]:
    """The Gauss-Legendre nodes ``w`` on [0, 1] and log weights, the
    weights including the ``3 w^2`` of ``v = w^3``."""
    x, w = np.polynomial.legendre.leggauss(GL_POINTS)
    nodes = (x + 1.0) / 2.0
    return nodes, np.log(w / 2.0 * 3.0 * nodes**2)


_GL_NODES, _GL_LOGW = _gl_grid()


@dataclass
class Layout:
    """Where each part is in the internal parameter vector, for ``p``
    covariates: the marker's intercept, slope and covariate coefficients,
    the log-Cholesky factor of ``D`` (log ``L00``, ``L10``, log ``L11``),
    log ``sigma``, log Weibull ``alpha`` and ``beta``, the hazard's
    covariate coefficients, and the marker's effect ``a``."""

    p: int

    @property
    def size(self) -> int:
        return 9 + 2 * self.p

    def unpack(self, u: Any) -> dict:
        p = self.p
        i = 2 + p
        return {
            "b0": u[0],
            "b1": u[1],
            "delta": u[2:i],
            "L00": anp.exp(u[i]),
            "L10": u[i + 1],
            "L11": anp.exp(u[i + 2]),
            "sigma": anp.exp(u[i + 3]),
            "scale": anp.exp(u[i + 4]),
            "shape": anp.exp(u[i + 5]),
            "gamma": u[i + 6 : i + 6 + p],
            "assoc": u[i + 6 + p],
        }

    def pack(self, theta: dict) -> npt.NDArray:
        return np.concatenate(
            [
                [theta["b0"], theta["b1"]],
                np.atleast_1d(theta["delta"]),
                [
                    np.log(theta["L00"]),
                    theta["L10"],
                    np.log(theta["L11"]),
                    np.log(theta["sigma"]),
                    np.log(theta["scale"]),
                    np.log(theta["shape"]),
                ],
                np.atleast_1d(theta["gamma"]),
                [theta["assoc"]],
            ]
        ).astype(float)


@dataclass
class JointArrays:
    """The items on the internal scale: counts ``w``, the age ``T`` each
    left observation (its failure or censoring), the event flag ``d``,
    the entry age ``L`` (0 for an item observed from 0), the covariates
    ``Z`` ``(N, p)``, and each item's readings as the sums the marker's
    likelihood needs: their number, and the sums of ``y``, ``t``, ``y^2``,
    ``t y`` and ``t^2``."""

    w: npt.NDArray
    T: npt.NDArray
    d: npt.NDArray
    L: npt.NDArray
    Z: npt.NDArray
    n_read: npt.NDArray
    Sy: npt.NDArray
    St: npt.NDArray
    Syy: npt.NDArray
    Sty: npt.NDArray
    Stt: npt.NDArray

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
    ) -> "JointArrays":
        def sums(f: Any) -> npt.NDArray:
            return np.array(
                [float(np.sum(f(t, y))) for t, y in zip(ages, values)]
            )

        return cls(
            w=np.asarray(w, dtype=float),
            T=np.asarray(T, dtype=float),
            d=np.asarray(d, dtype=float),
            L=np.asarray(L, dtype=float),
            Z=np.asarray(Z, dtype=float).reshape(len(ages), -1),
            n_read=np.array([float(len(t)) for t in ages]),
            Sy=sums(lambda t, y: y),
            St=sums(lambda t, y: t),
            Syy=sums(lambda t, y: y * y),
            Sty=sums(lambda t, y: t * y),
            Stt=sums(lambda t, y: t * t),
        )


# -- the pieces of an item's log-integrand ----------------------------------


def _lines(theta: dict, arrays: JointArrays, b: Any) -> tuple:
    """The intercept ``A`` and slope ``B`` of each item's line at the
    random effects ``b`` (``(N, K, 2)`` or ``(N, 2)``)."""
    fixed = theta["b0"] + anp.dot(arrays.Z, theta["delta"])
    if anp.ndim(b) == 3:
        return fixed[:, None] + b[..., 0], theta["b1"] + b[..., 1]
    return fixed + b[:, 0], theta["b1"] + b[:, 1]


def _expand(v: Any, b: Any) -> Any:
    """A per-item array lined up with ``b``'s leading axes."""
    return v[:, None] if anp.ndim(b) == 3 else v


def log_readings(theta: dict, arrays: JointArrays, b: Any) -> Any:
    """``log p(y_i | b)``, each item's readings given its line."""
    A, B = _lines(theta, arrays, b)
    a = arrays
    n, Sy, St = _expand(a.n_read, b), _expand(a.Sy, b), _expand(a.St, b)
    Syy, Sty, Stt = _expand(a.Syy, b), _expand(a.Sty, b), _expand(a.Stt, b)
    ss = (
        Syy
        - 2 * A * Sy
        - 2 * B * Sty
        + n * A * A
        + 2 * A * B * St
        + B * B * Stt
    )
    s2 = theta["sigma"] ** 2
    return -0.5 * n * anp.log(2 * np.pi * s2) - ss / (2 * s2)


def log_cumhaz(theta: dict, arrays: JointArrays, b: Any, t: Any) -> Any:
    """``log H(t_i | b)``, the cumulative hazard to each item's ``t_i``
    (``-inf`` where ``t_i`` is 0)."""
    A, B = _lines(theta, arrays, b)
    shape, scale, a = theta["shape"], theta["scale"], theta["assoc"]
    positive = t > 0
    safe = anp.where(positive, t, 1.0)
    lin = anp.dot(arrays.Z, theta["gamma"])
    # the times of the quadrature nodes: s = t w^(3 / beta), (N, Q)
    s = safe[:, None] * _GL_NODES[None, :] ** (3.0 / shape)
    if anp.ndim(b) == 3:
        inner = logsumexp(_GL_LOGW + a * B[..., None] * s[:, None, :], axis=-1)
        base = (lin + shape * (anp.log(safe) - anp.log(scale)))[:, None]
        out = base + a * A + inner
        return anp.where(positive[:, None], out, -anp.inf)
    inner = logsumexp(_GL_LOGW + a * B[:, None] * s, axis=-1)
    out = lin + shape * (anp.log(safe) - anp.log(scale)) + a * A + inner
    return anp.where(positive, out, -anp.inf)


def log_lifetime(theta: dict, arrays: JointArrays, b: Any) -> Any:
    """``log p(T_i, d_i | b)``: the density at a failure, the survival to
    a censoring, from age 0."""
    A, B = _lines(theta, arrays, b)
    shape, scale, a = theta["shape"], theta["scale"], theta["assoc"]
    T, d = _expand(arrays.T, b), _expand(arrays.d, b)
    lin = _expand(anp.dot(arrays.Z, theta["gamma"]), b)
    log_h = (
        anp.log(shape)
        - anp.log(scale)
        + (shape - 1) * (anp.log(T) - anp.log(scale))
        + lin
        + a * (A + B * T)
    )
    return d * log_h - anp.exp(log_cumhaz(theta, arrays, b, arrays.T))


def log_prior(theta: dict, b: Any) -> Any:
    """``log phi(b; 0, D)`` with ``D = L L'``."""
    L00, L10, L11 = theta["L00"], theta["L10"], theta["L11"]
    x0 = b[..., 0] / L00
    x1 = (b[..., 1] - L10 * x0) / L11
    return (
        -anp.log(2 * np.pi)
        - anp.log(L00)
        - anp.log(L11)
        - 0.5 * (x0**2 + x1**2)
    )


# -- where the nodes go: each item's posterior mode and curvature ----------


def _numeric(theta: dict) -> dict:
    """``theta`` with plain floats and arrays (no autograd boxes)."""
    return {
        k: np.asarray(getattr(v, "_value", v), dtype=float)
        for k, v in theta.items()
    }


def _objective(
    theta: dict,
    arrays: JointArrays,
    b: npt.NDArray,
    d: npt.NDArray,
    T: npt.NDArray,
) -> tuple:
    """``-log`` of the integrand (less constants) at ``b`` ``(N, 2)``,
    with its gradient and Hessian in ``b``: the readings, the lifetime
    (an event flag ``d`` at age ``T``) and the prior."""
    A, B = _lines(theta, arrays, b)
    a, s2 = float(theta["assoc"]), float(theta["sigma"]) ** 2
    n, Sy, St = arrays.n_read, arrays.Sy, arrays.St
    Syy, Sty, Stt = arrays.Syy, arrays.Sty, arrays.Stt
    ss = (
        Syy
        - 2 * A * Sy
        - 2 * B * Sty
        + n * A * A
        + 2 * A * B * St
        + B * B * Stt
    )
    g_read = np.stack([-Sy + n * A + B * St, -Sty + A * St + B * Stt], -1) / s2
    h_read = (
        np.stack([np.stack([n, St], -1), np.stack([St, Stt], -1)], -2) / s2
    )
    # the cumulative hazard to T, node by node
    shape, scale = float(theta["shape"]), float(theta["scale"])
    positive = T > 0
    safe = np.where(positive, T, 1.0)
    s = safe[:, None] * _GL_NODES[None, :] ** (3.0 / shape)
    lin = arrays.Z @ theta["gamma"]
    with np.errstate(over="ignore", under="ignore"):
        lam = np.exp(
            (lin + shape * (np.log(safe) - np.log(scale)) + a * A)[:, None]
            + _GL_LOGW
            + a * B[:, None] * s
        )
    lam = np.where(positive[:, None], lam, 0.0)
    H0, H1, H2 = lam.sum(1), (lam * s).sum(1), (lam * s * s).sum(1)
    g_surv = np.stack([a * H0 - d * a, a * H1 - d * a * T], -1)
    h_surv = (
        a * a * np.stack([np.stack([H0, H1], -1), np.stack([H1, H2], -1)], -2)
    )
    D = _covariance(theta)
    Dinv = np.linalg.inv(D)
    f = (
        ss / (2 * s2)
        - d * a * (A + B * T)
        + H0
        + 0.5 * np.einsum("ni,ij,nj->n", b, Dinv, b)
    )
    grad = g_read + g_surv + b @ Dinv
    hess = h_read + h_surv + Dinv[None]
    return f, grad, hess


def _covariance(theta: dict) -> npt.NDArray:
    L = np.array(
        [
            [float(theta["L00"]), 0.0],
            [float(theta["L10"]), float(theta["L11"])],
        ]
    )
    return L @ L.T


def posterior_modes(
    theta: dict,
    arrays: JointArrays,
    d: npt.NDArray,
    T: npt.NDArray,
    start: "npt.NDArray | None" = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Each item's mode of ``p(y | b) p(T, d | b) phi(b)`` and the
    Cholesky factor of the inverse of its curvature there, by damped
    Newton steps (the integrand is log-concave in ``b``, so they
    converge from anywhere)."""
    theta = _numeric(theta)
    N = len(arrays.w)
    b = np.zeros((N, 2)) if start is None else np.array(start, dtype=float)
    f, g, h = _objective(theta, arrays, b, d, T)
    for _ in range(100):
        step = -np.linalg.solve(h, g[..., None])[..., 0]
        size = np.ones(N)
        for _ in range(30):
            trial = b + size[:, None] * step
            f_new, g_new, h_new = _objective(theta, arrays, trial, d, T)
            worse = ~(f_new <= f + 1e-12 * np.abs(f))
            if not worse.any():
                break
            size = np.where(worse, size / 2, size)
        accept = ~worse
        b = np.where(accept[:, None], trial, b)
        f = np.where(accept, f_new, f)
        g = np.where(accept[:, None], g_new, g)
        h = np.where(accept[:, None, None], h_new, h)
        if np.all(np.abs(step * size[:, None]) <= 1e-10 * (1 + np.abs(b))):
            break
    cov = np.linalg.inv(h)
    # symmetric, and positive definite (the curvature is)
    cov = 0.5 * (cov + np.swapaxes(cov, -1, -2))
    return b, np.linalg.cholesky(cov)


def censored_modes(
    theta: dict, arrays: JointArrays, times: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """:func:`posterior_modes` of items censored at ``times`` (each row of
    ``arrays`` at its own time), found by continuation: from age 0, the
    time is at most doubled at each stage and the search started from the
    last stage's modes. Far beyond the data the integrand's mode is where
    the hazard stays small, which a search started from the modes of
    nearer ages reaches; started from the readings alone, the cumulative
    hazard there overflows."""
    times = np.asarray(times, dtype=float)
    smallest = np.min(times[times > 0]) if np.any(times > 0) else 1.0
    current = np.minimum(times, smallest)
    mode, root = posterior_modes(theta, arrays, np.zeros(times.size), current)
    while np.any(current < times):
        current = np.minimum(times, 2.0 * current)
        mode, root = posterior_modes(
            theta, arrays, np.zeros(times.size), current, start=mode
        )
    return mode, root


def adaptive_nodes(
    mode: npt.NDArray, root: npt.NDArray, points: int
) -> tuple[npt.NDArray, npt.NDArray]:
    """The nodes ``(N, K, 2)`` of each item's adaptive grid and their log
    weights ``(N, K)``, including the change of variables: ``int f(b) db
    ~ sum_k exp(logw_k) f(b_k)``."""
    z, logw = _gh_grid(points)
    nodes = mode[:, None, :] + np.sqrt(2.0) * np.einsum("nij,kj->nki", root, z)
    log_det = np.log(2.0) + np.log(root[:, 0, 0]) + np.log(root[:, 1, 1])
    return (
        nodes,
        logw[None, :] + np.sum(z**2, axis=1)[None, :] + log_det[:, None],
    )


# -- the likelihood ----------------------------------------------------------


def log_entry(theta: dict, arrays: JointArrays) -> Any:
    """``log P(T > L_i)`` over the population's random effects: the
    probability of reaching the entry age (0 for an item observed from
    0)."""
    z, logw = _gh_grid(GH_ENTRY)
    L00, L10, L11 = theta["L00"], theta["L10"], theta["L11"]
    b0 = np.sqrt(2.0) * L00 * z[:, 0]
    b1 = np.sqrt(2.0) * (L10 * z[:, 0] + L11 * z[:, 1])
    # the same nodes for every item: (1, K, 2)
    b = anp.stack([b0, b1], axis=-1)[None, :, :]
    log_H = log_cumhaz(theta, arrays, b, arrays.L)
    return logsumexp(logw - np.log(np.pi) - anp.exp(log_H), axis=1)


def item_log_likelihoods(
    u: Any, layout: Layout, arrays: JointArrays, nodes: tuple
) -> Any:
    """Each item's log-likelihood at the parameters ``u``, by the fixed
    adaptive nodes ``(b, logw)``."""
    theta = layout.unpack(u)
    b, logw = nodes
    log_g = (
        log_readings(theta, arrays, b)
        + log_lifetime(theta, arrays, b)
        + log_prior(theta, b)
    )
    out = logsumexp(logw + log_g, axis=1)
    if np.any(arrays.L > 0):
        out = out - anp.where(arrays.L > 0, log_entry(theta, arrays), 0.0)
    return out


def neg_ll(u: Any, layout: Layout, arrays: JointArrays, nodes: tuple) -> Any:
    """The negative log-likelihood (internal scale) at ``u``."""
    return -anp.sum(arrays.w * item_log_likelihoods(u, layout, arrays, nodes))


def place_nodes(
    u: npt.NDArray,
    layout: Layout,
    arrays: JointArrays,
    start: "npt.NDArray | None" = None,
    points: int = GH_FIT,
) -> tuple:
    """The adaptive nodes of every item at the parameters ``u``, and the
    modes (to start the next placement from)."""
    theta = layout.unpack(np.asarray(u, dtype=float))
    mode, root = posterior_modes(theta, arrays, arrays.d, arrays.T, start)
    return adaptive_nodes(mode, root, points), mode
