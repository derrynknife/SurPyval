r"""Lin and Ying's (1993) estimator of the joint survival of two lifetimes
under univariate (common) censoring.

When both lifetimes of a unit are followed until the same time (both eyes
seen until the patient's last visit, both components of a machine until it
is retired), one censoring time :math:`C` stops both. Then a unit is seen
past :math:`(s, t)` in both series exactly when both lifetimes last past
their times *and* the follow-up lasts past the later of the two, so

.. math::
    P(X_1 > s, X_2 > t) = S(s, t)\, G(\max(s, t)),

with :math:`X_k = \min(T_k, C)` the observed times and :math:`G(u) = P(C >
u)`. The estimate turns this around:

.. math::
    \hat S(s, t) = \frac{\hat H(s, t)}{\hat G(\max(s, t))},

with :math:`\hat H(s, t)` the fraction of units whose two observed times
are past :math:`s` and :math:`t` (failed or censored), and :math:`\hat G`
the Kaplan-Meier estimate of the follow-up time: each unit gives the later
of its two times, an "event" of follow-up if either lifetime is censored
there, else censored (follow-up lasted at least that long).

It holds only when the censoring is common. It is refused for data whose
censoring is not (see ``check_common_censoring``).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__: list[str] = []


def _inv(a: npt.NDArray) -> npt.NDArray:
    out = np.zeros_like(a, dtype=float)
    np.divide(1.0, a, out=out, where=a > 0)
    return out


def check_common_censoring(x: npt.NDArray, c: npt.NDArray) -> None:
    """Refuse pairs that cannot share one censoring time.

    Under common censoring a unit's censored lifetimes are censored at the
    same time :math:`C`, and a lifetime seen to fail failed by then
    (:math:`T \\le C`). So a pair is refused when its two lifetimes are
    censored at different times, or when one is censored at a time before
    the other's failure.
    """
    cens = c == 1
    both = cens.all(axis=1)
    differ = both & (x[:, 0] != x[:, 1])
    one = cens.any(axis=1) & ~both
    C = np.where(cens[:, 0], x[:, 0], x[:, 1])
    other = np.where(cens[:, 0], x[:, 1], x[:, 0])
    late = one & (other > C)
    if differ.any() or late.any():
        parts = []
        if differ.any():
            i = int(np.flatnonzero(differ)[0])
            parts.append(
                f"{int(differ.sum())} unit(s) have both lifetimes censored "
                f"at different times (row {i}: {x[i, 0]:g} and "
                f"{x[i, 1]:g})"
            )
        if late.any():
            i = int(np.flatnonzero(late)[0])
            parts.append(
                f"{int(late.sum())} unit(s) have one lifetime censored "
                "before the other's failure, which a shared follow-up "
                f"time cannot give (row {i}: censored at {C[i]:g}, failed "
                f"at {other[i]:g})"
            )
        raise ValueError(
            "Lin and Ying's estimator needs one censoring time shared by "
            "both lifetimes of each unit (common censoring), and these "
            "data do not have one: " + "; ".join(parts) + ". Use "
            "estimator='dabrowska' (or 'prentice-cai'), which allow each "
            "lifetime its own censoring time."
        )


class _LinYingGrid:
    """Lin and Ying's estimate on the grid of every observed time of each
    series and every follow-up time (it steps at censoring times as well
    as at failures), as
    ``_Grid`` is for Dabrowska's: the counts are a function of the rows'
    weights, so a bootstrap resample reuses the grid."""

    def __init__(self, x: npt.NDArray, c: npt.NDArray) -> None:
        self.x = x
        # The follow-up time: the later of the two times, seen to end
        # ("event") when either lifetime is censored
        m = np.max(x, axis=1)
        self.cens = (c == 1).any(axis=1)
        self.W = np.unique(m[self.cens])
        # The estimate steps where either series' time passes an observed
        # time of its own, and where the later of the two passes a
        # follow-up time: so each grid holds both
        self.U = np.unique(np.concatenate([x[:, 0], self.W]))
        self.V = np.unique(np.concatenate([x[:, 1], self.W]))
        self.r1 = np.searchsorted(self.U, x[:, 0])
        self.r2 = np.searchsorted(self.V, x[:, 1])
        self.rm = np.searchsorted(self.W, m, side="right") - 1
        # Where G is read: at max(s, t) on the grid, and at each margin
        self.g = (
            np.searchsorted(self.W, np.maximum.outer(self.U, self.V), "right")
            - 1
        )
        self.g1 = np.searchsorted(self.W, self.U, side="right") - 1
        self.g2 = np.searchsorted(self.W, self.V, side="right") - 1

    def upto(self, s: float, t: float) -> _LinYingGrid:
        # Every unit's times enter every point of the estimate (through
        # H's counts beyond the point and G), so there is no corner to cut
        return self

    def counts(self, w: npt.NDArray) -> dict:
        K, L = len(self.U), len(self.V)
        total = float(np.sum(w))
        hist = np.zeros((K, L))
        np.add.at(hist, (self.r1, self.r2), w)
        # H[i, j]: the weight of units with X1 > U[i] and X2 > V[j]
        ge = hist[::-1, ::-1].cumsum(axis=0).cumsum(axis=1)[::-1, ::-1]
        H = np.zeros((K, L))
        H[:-1, :-1] = ge[1:, 1:]
        H1 = np.concatenate([ge[1:, 0], [0.0]])
        H2 = np.concatenate([ge[0, 1:], [0.0]])
        # G: Kaplan-Meier of the follow-up time
        J_ = len(self.W)
        at = self.rm >= 0
        R = np.zeros(J_)
        np.add.at(R, self.rm[at], w[at])
        R = R[::-1].cumsum()[::-1]
        D = np.zeros(J_)
        np.add.at(D, self.rm[self.cens], w[self.cens])
        G = np.concatenate([[1.0], np.cumprod(1 - D * _inv(R))])
        inv_total = 1.0 / total if total > 0 else 0.0
        # 0 / 0 (nobody followed that long) is 0
        J = H * inv_total * _inv(G[self.g + 1])
        S1 = H1 * inv_total * _inv(G[self.g1 + 1])
        S2 = H2 * inv_total * _inv(G[self.g2 + 1])
        return dict(J=J, S1=S1, S2=S2, G=G)
