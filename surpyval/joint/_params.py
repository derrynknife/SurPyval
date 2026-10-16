"""The joint model's parameters: their names, and the map between the
natural parameters a model reports and the internal search vector its fit
works in (#891).

The fit works in internal units (:class:`Scaling`): the times divided by
a time scale, the readings centred at the marker centre and divided by
their spread, and each covariate column standardised. The natural
parameters are those of the data's own units, and are what a model
stores: the map below is exact, so a model's predictions do not depend on
the internal units, and autograd differentiates it for the covariance.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt

from ._likelihood import (
    Layout,
    Q,
    basis_time_scale,
    chol_entries,
    chol_factor,
    unpack,
)

#: The names of the path's basis functions, as its parameters use them.
PATH_TERMS = ("intercept", "slope")
SD_NAMES = tuple(f"sd_{term}" for term in PATH_TERMS)
CORR_NAMES = ("corr",)


class Scaling(NamedTuple):
    """The internal units: ``t_s`` the time scale, ``y_c`` the marker
    centre (the ``marker_centre`` of the model, which is not a unit
    choice: the Weibull scale is that of a unit whose marker stays there)
    and ``y_s`` the marker scale; ``z_m``, ``z_s`` and ``w_m``, ``w_s``
    the means and spreads of the hazard and path covariates."""

    t_s: float
    y_c: float
    y_s: float
    z_m: npt.NDArray
    z_s: npt.NDArray
    w_m: npt.NDArray
    w_s: npt.NDArray

    @classmethod
    def identity(cls, centre: float, p: int, pw: int) -> "Scaling":
        """The natural units themselves (for prediction)."""
        return cls(
            1.0,
            float(centre),
            1.0,
            np.zeros(p),
            np.ones(p),
            np.zeros(pw),
            np.ones(pw),
        )


def parameter_names(z_names: list[str], w_names: list[str]) -> list[str]:
    """The model's parameters, in the order of ``params``: the Weibull
    scale and shape, the hazard coefficients (named by their columns), the
    association, the path's coefficients (``path_intercept``,
    ``path_slope`` and, for each path covariate ``w``,
    ``path_intercept:w`` and ``path_slope:w``), the random effects'
    standard deviations and correlation, and the noise ``sigma``."""
    names = ["alpha", "beta", *z_names, "association"]
    for col in ["", *[":" + w for w in w_names]]:
        names += [f"path_{term}{col}" for term in PATH_TERMS]
    return names + list(SD_NAMES) + list(CORR_NAMES) + ["sigma"]


def natural_from_internal(theta: Any, layout: Layout, sc: Scaling) -> Any:
    """The natural parameters (in the order of :func:`parameter_names`,
    the aliased columns left out) of the internal vector ``theta``.
    Written in ``autograd.numpy``, so that it can be differentiated."""
    parts = unpack(theta, layout)
    k = anp.exp(parts.lk)
    gamma = parts.gamma / sc.z_s
    assoc = parts.assoc / sc.y_s
    log_alpha = parts.la + np.log(sc.t_s) + anp.dot(gamma, sc.z_m) / k
    S = sc.y_s * basis_time_scale(sc.t_s)
    P = parts.path  # (Q, 1 + pw)
    cols = [P[:, j + 1] / sc.w_s[j] for j in range(layout.pw)]
    shift = P[:, 0]
    for j in range(layout.pw):
        shift = shift - cols[j] * sc.w_m[j]
    centre = np.zeros(Q)
    centre[0] = sc.y_c
    path = [anp.dot(S, shift) + centre] + [anp.dot(S, c) for c in cols]
    L = anp.dot(S, chol_factor(parts.chol))
    D = anp.dot(L, L.T)
    sd = anp.sqrt(anp.diag(D))
    corr = D[0, 1] / (sd[0] * sd[1])
    sigma = sc.y_s * anp.exp(parts.lsig)
    return anp.concatenate(
        [
            anp.reshape(anp.exp(log_alpha), (1,)),
            anp.reshape(k, (1,)),
            gamma,
            anp.reshape(assoc, (1,)),
            *path,
            sd,
            anp.reshape(corr, (1,)),
            anp.reshape(sigma, (1,)),
        ]
    )


class Natural(NamedTuple):
    """The natural parameters, unpacked."""

    alpha: float
    beta: float
    gamma: npt.NDArray
    assoc: float
    path: npt.NDArray  # (Q, 1 + pw)
    sd: npt.NDArray
    corr: float
    sigma: float

    @property
    def D(self) -> npt.NDArray:
        """The covariance of the random effects."""
        R = np.array([[1.0, self.corr], [self.corr, 1.0]])
        return R * np.outer(self.sd, self.sd)


def unpack_natural(values: npt.ArrayLike, p: int, pw: int) -> Natural:
    """:class:`Natural` from a vector in the order of
    :func:`parameter_names` (with ``p`` hazard and ``pw`` path
    covariates)."""
    v = np.asarray(values, dtype=float)
    at = 3 + p
    n_path = Q * (1 + pw)
    path = v[at : at + n_path].reshape(1 + pw, Q).T
    rest = v[at + n_path :]
    return Natural(
        alpha=float(v[0]),
        beta=float(v[1]),
        gamma=v[2 : 2 + p],
        assoc=float(v[2 + p]),
        path=path,
        sd=rest[:Q],
        corr=float(rest[Q]),
        sigma=float(rest[Q + 1]),
    )


def internal_from_natural(
    values: npt.ArrayLike, layout: Layout, sc: Scaling
) -> npt.NDArray:
    """The internal vector of natural parameters ``values`` (the inverse
    of :func:`natural_from_internal`)."""
    nat = unpack_natural(values, layout.p, layout.pw)
    k = nat.beta
    gamma_i = nat.gamma * sc.z_s
    la = np.log(nat.alpha) - np.log(sc.t_s) - np.dot(nat.gamma, sc.z_m) / k
    S_inv = np.linalg.inv(sc.y_s * basis_time_scale(sc.t_s))
    centre = np.zeros(Q)
    centre[0] = sc.y_c
    cols = [S_inv @ nat.path[:, j + 1] * sc.w_s[j] for j in range(layout.pw)]
    first = S_inv @ (nat.path[:, 0] - centre)
    for j in range(layout.pw):
        first = first + cols[j] * sc.w_m[j] / sc.w_s[j]
    D = S_inv @ nat.D @ S_inv.T
    return np.concatenate(
        [
            [la, np.log(k)],
            gamma_i,
            [nat.assoc * sc.y_s],
            np.concatenate([first, *cols]),
            [np.log(nat.sigma / sc.y_s)],
            chol_entries(D),
        ]
    )


def kept_positions(
    p: int, pw: int, z_kept: npt.NDArray, w_kept: npt.NDArray
) -> npt.NDArray:
    """The positions in ``params`` of the estimated parameters."""
    pos = [0, 1, *(2 + z_kept).tolist(), 2 + p]
    start = 3 + p
    pos += [start, start + 1]
    for j in w_kept.tolist():
        pos += [start + Q * (1 + j), start + Q * (1 + j) + 1]
    rest = 3 + p + Q * (1 + pw)
    pos += list(range(rest, rest + Q + 2))
    return np.asarray(pos, dtype=int)
