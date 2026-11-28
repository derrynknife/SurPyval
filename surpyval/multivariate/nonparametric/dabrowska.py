r"""Dabrowska's non-parametric estimator of the joint survival of two
lifetimes under right censoring (Dabrowska, 1988).

Two lifetimes measured on the same unit (two eyes, twins, two components
of one machine, the hours and the landings of one part) are usually
dependent. Their joint survival :math:`S(s, t) = P(T_1 > s, T_2 > t)` is
the chance that the first lasts past :math:`s` *and* the second past
:math:`t`. A copula estimates it by choosing a family for the dependence;
this estimator chooses none: it is the two-dimensional analogue of the
Kaplan-Meier estimate.

It builds the joint survival from the two margins (each a Kaplan-Meier
estimate) and a correction for the dependence at each pair of failure
times. At a pair :math:`(u, v)` of failure times of the two series, of the
:math:`Y` units still at risk in both (``T_1 >= u`` and ``T_2 >= v``),
:math:`a` fail in the first series at :math:`u`, :math:`b` in the second at
:math:`v` and :math:`c` in both; :math:`m = Y - a - b + c` fail in
neither. The factor

.. math::
    f(u, v) = \frac{Y m}{(Y - a)(Y - b)}

is the odds-ratio-like cross-ratio of that two-by-two table: 1 when the
failures in the two series are unrelated, above 1 when they come together.
The estimate is

.. math::
    \hat S(s, t) = \hat S_1(s)\, \hat S_2(t)
        \prod_{u \le s,\ v \le t} f(u, v).

Written as Dabrowska wrote it, :math:`f = 1 - (\Lambda_{10}\Lambda_{01} -
\Lambda_{11}) / ((1 - \Lambda_{10})(1 - \Lambda_{01}))`, with the
single and double hazards :math:`\Lambda_{10} = a / Y`,
:math:`\Lambda_{01} = b / Y`, :math:`\Lambda_{11} = c / Y`. With no
censoring the product telescopes and the estimate is exactly the empirical
joint survival function, as Kaplan-Meier is the empirical survival
function.

``Dabrowska.fit(..., estimator="prentice-cai")`` gives Prentice and Cai's
(1992) estimate on the same grid (see ``_VolterraGrid``), and
``estimator="lin-ying"`` Lin and Ying's (1993) for common censoring (see
``surpyval.multivariate.nonparametric.lin_ying``).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from surpyval.multivariate.nonparametric.lin_ying import (
    _LinYingGrid,
    check_common_censoring,
)
from surpyval.serialisation import SerialisableMixin
from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.validation import check_alpha_ci
from surpyval.utils.warnings import caller_stacklevel

__all__ = ["Dabrowska", "BivariateNonParametric"]

# Elements of a (units x query points) influence matrix built at once.
_CHUNK = 2_000_000


def _inv(a: npt.NDArray) -> npt.NDArray:
    """``1 / a``, 0 where ``a`` is 0."""
    out = np.zeros_like(a, dtype=float)
    np.divide(1.0, a, out=out, where=a > 0)
    return out


def _cum2(a: npt.NDArray) -> npt.NDArray:
    return np.cumsum(np.cumsum(a, axis=0), axis=1)


def _rev2(a: npt.NDArray) -> npt.NDArray:
    return a[::-1, ::-1].cumsum(axis=0).cumsum(axis=1)[::-1, ::-1]


class _Grid:
    """The fixed part of the estimate: the failure-time grids of the two
    series and where each row sits on them. The counts are a function of
    the rows' weights, so a bootstrap resample or a perturbed weight
    reuses the grid."""

    def __init__(self, x: npt.NDArray, c: npt.NDArray) -> None:
        self.x = x
        self.f1 = c[:, 0] == 0
        self.f2 = c[:, 1] == 0
        self.U = np.unique(x[self.f1, 0])
        self.V = np.unique(x[self.f2, 1])
        # The largest grid time each row's time reaches (at risk there)
        self.r1 = np.searchsorted(self.U, x[:, 0], side="right") - 1
        self.r2 = np.searchsorted(self.V, x[:, 1], side="right") - 1

    def upto(self, s: float, t: float) -> _Grid:
        """The grid cut to the failure times up to ``s`` and ``t``: the
        estimate on it is the full grid's wherever both coordinates are at
        most ``s`` and ``t``, so a bootstrap asked about points there
        builds only that corner. A row whose time is past the cut is at
        risk at every time kept, and its failure is past them."""
        K = int(np.searchsorted(self.U, s, side="right"))
        L = int(np.searchsorted(self.V, t, side="right"))
        if K == len(self.U) and L == len(self.V):
            return self
        cut = object.__new__(type(self))
        cut.x = self.x
        cut.U, cut.V = self.U[:K], self.V[:L]
        cut.f1 = self.f1 & (self.r1 < K)
        cut.f2 = self.f2 & (self.r2 < L)
        cut.r1 = np.minimum(self.r1, K - 1)
        cut.r2 = np.minimum(self.r2, L - 1)
        return cut

    def counts(self, w: npt.NDArray) -> dict:
        K, L = len(self.U), len(self.V)
        r1, r2, f1, f2 = self.r1, self.r2, self.f1, self.f2
        both = (r1 >= 0) & (r2 >= 0)
        H = np.zeros((K, L))
        np.add.at(H, (r1[both], r2[both]), w[both])
        HA = np.zeros((K, L))
        sel = both & f1
        np.add.at(HA, (r1[sel], r2[sel]), w[sel])
        HB = np.zeros((K, L))
        sel = both & f2
        np.add.at(HB, (r1[sel], r2[sel]), w[sel])
        HC = np.zeros((K, L))
        sel = both & f1 & f2
        np.add.at(HC, (r1[sel], r2[sel]), w[sel])
        Y = _rev2(H) if K and L else H
        A = HA[:, ::-1].cumsum(axis=1)[:, ::-1]
        B = HB[::-1, :].cumsum(axis=0)[::-1, :]
        C = HC
        M = Y - A - B + C
        # Rounding of weighted sums: a count that is zero is zero
        tiny = 1e-12 * max(float(np.sum(w)), 1.0)
        for arr in (M, Y):
            arr[np.abs(arr) < tiny] = 0.0
        YA, YB = Y - A, Y - B
        YA[np.abs(YA) < tiny] = 0.0
        YB[np.abs(YB) < tiny] = 0.0
        factor = np.ones((K, L))
        live = (Y > 0) & (M > 0)
        factor[live] = Y[live] * M[live] / (YA[live] * YB[live])
        # No unit left in both series (m = 0), or none at risk in both
        # (Y = 0, a corner past the data where nothing is known): 0, as
        # R's mhazard; with no censoring that is the empirical answer.
        factor[~live] = 0.0
        # The margins: Kaplan-Meier of each series on its own grid
        Y1 = np.zeros(K)
        np.add.at(Y1, r1[r1 >= 0], w[r1 >= 0])
        Y1 = Y1[::-1].cumsum()[::-1]
        D1 = np.zeros(K)
        np.add.at(D1, r1[f1], w[f1])
        Y2 = np.zeros(L)
        np.add.at(Y2, r2[r2 >= 0], w[r2 >= 0])
        Y2 = Y2[::-1].cumsum()[::-1]
        D2 = np.zeros(L)
        np.add.at(D2, r2[f2], w[f2])
        S1 = np.cumprod(1 - D1 * _inv(Y1))
        S2 = np.cumprod(1 - D2 * _inv(Y2))
        P = np.cumprod(np.cumprod(factor, axis=0), axis=1)
        return dict(
            Y=Y,
            A=A,
            B=B,
            C=C,
            M=M,
            YA=YA,
            YB=YB,
            live=live,
            P=P,
            Y1=Y1,
            D1=D1,
            Y2=Y2,
            D2=D2,
            S1=S1,
            S2=S2,
            J=S1[:, None] * S2[None, :] * P,
        )


class _VolterraGrid(_Grid):
    r"""Prentice and Cai's (1992) estimate on Dabrowska's grid: the same
    counts, then the joint survival from the Volterra equation

    .. math::
        S(s, t) = S_1(s) S_2(t) \Big[1 + \int_0^s\!\!\int_0^t
        \frac{S(u^-, v^-)}{S_1(u) S_2(v)}\, \big(\Lambda_{11}(du, dv)
        - \Lambda_{10}(du, v^-) \Lambda_2(dv)
        - \Lambda_1(du) \Lambda_{01}(u^-, dv)
        + \Lambda_1(du) \Lambda_2(dv)\big)\Big],

    solved one anti-diagonal of the grid at a time through its
    double difference (which divides by nothing):

    .. math::
        S_{ij} = (1 - \lambda_{1i}) S_{i-1,j} + (1 - \lambda_{2j}) S_{i,j-1}
        - (1 - \lambda_{1i})(1 - \lambda_{2j}) S_{i-1,j-1}
        + Q_{ij} S_{i-1,j-1},

    :math:`Q = (c - a \lambda_2 - b \lambda_1) / Y + \lambda_1 \lambda_2`,
    with :math:`\lambda_1 = d_1 / Y_1` and :math:`\lambda_2` the
    margins' Nelson-Aalen increments."""

    def counts(self, w: npt.NDArray) -> dict:
        e = super().counts(w)
        e["J"] = _volterra(e)
        return e


def _volterra(e: dict) -> npt.NDArray:
    S1, S2 = e["S1"], e["S2"]
    K, L = len(S1), len(S2)
    if not (K and L):
        return np.zeros((K, L))
    lam1 = e["D1"] * _inv(e["Y1"])
    lam2 = e["D2"] * _inv(e["Y2"])
    # The local hazards a/Y, b/Y, c/Y; 0 where nobody is at risk in both
    # (then a = b = c = 0 too)
    iY = _inv(e["Y"])
    Q = (e["C"] - e["A"] * lam2[None, :] - e["B"] * lam1[:, None]) * iY + lam1[
        :, None
    ] * lam2[None, :]
    a1, a2 = 1 - lam1, 1 - lam2
    P = np.empty((K + 1, L + 1))
    P[0, 0] = 1.0
    P[1:, 0] = S1
    P[0, 1:] = S2
    for d in range(2, K + L + 1):
        i = np.arange(max(1, d - L), min(K, d - 1) + 1)
        j = d - i
        b1, b2 = a1[i - 1], a2[j - 1]
        prev = P[i - 1, j - 1]
        P[i, j] = (
            b1 * P[i - 1, j]
            + b2 * P[i, j - 1]
            - b1 * b2 * prev
            + Q[i - 1, j - 1] * prev
        )
    return P[1:, 1:]


def _evaluate(
    grid: _Grid, est: dict, x: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """The joint survival and the two margins at the points ``x``."""
    k = np.searchsorted(grid.U, x[:, 0], side="right") - 1
    l_ = np.searchsorted(grid.V, x[:, 1], side="right") - 1
    # 1 before a series' first failure (an index of -1)
    S1 = np.concatenate([[1.0], est["S1"]])[k + 1]
    S2 = np.concatenate([[1.0], est["S2"]])[l_ + 1]
    if est["J"].size:
        S = np.where(
            (k >= 0) & (l_ >= 0),
            est["J"][np.maximum(k, 0), np.maximum(l_, 0)],
            S1 * S2,
        )
    else:
        S = S1 * S2
    missing = np.isnan(x).any(axis=1)
    if missing.any():
        S, S1, S2 = (np.where(missing, np.nan, a) for a in (S, S1, S2))
    return S, S1, S2


class BivariateNonParametric(SerialisableMixin):
    r"""
    A non-parametric estimate of the joint survival of two lifetimes,
    from :meth:`Dabrowska.fit`.

    Attributes
    ----------
    estimator : str
        ``"Dabrowska"``, ``"Prentice-Cai"`` or ``"Lin-Ying"`` (the
        ``estimator`` given to :meth:`Dabrowska.fit`).
    common_censoring : bool
        Whether the data were declared to have one censoring time shared
        by both lifetimes of each unit (required by Lin-Ying).
    x, c, n : numpy.ndarray
        The data: ``x`` and ``c`` of shape ``(N, 2)``, ``n`` the count of
        each row.
    t1, t2 : numpy.ndarray
        The failure times of each series, the grid on which the
        Dabrowska and Prentice-Cai estimates step (Lin-Ying's steps at
        censoring times too).
    names : tuple of str or None
        The names of the two series, if the data had them.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.multivariate import Dabrowska
    >>> x = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])
    >>> model = Dabrowska.fit(x)
    >>> model.estimator
    'Dabrowska'
    >>> model.t1, model.t2
    (array([1., 2., 3., 4.]), array([1., 2., 3., 4.]))
    >>> model.sf([[2.5, 2.5]])
    array([0.5])
    """

    def __init__(
        self,
        x: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        names: Any = None,
        estimator: str = "dabrowska",
        common_censoring: bool = False,
    ) -> None:
        self.estimator = _check_estimator(estimator)
        self.common_censoring = bool(common_censoring)
        if self.estimator == "Lin-Ying" and not self.common_censoring:
            raise ValueError(
                "Lin and Ying's estimator is right only when both lifetimes "
                "of each unit share one censoring time (common censoring: "
                "for example both followed until the unit leaves the "
                "study). Declare it with common_censoring=True if the data "
                "were collected that way; otherwise use "
                "estimator='dabrowska', which allows each lifetime its own "
                "censoring time."
            )
        if self.common_censoring:
            check_common_censoring(x, c)
        self.x = x
        self.c = c
        self.n = n
        self.names = names
        kind: Any = {
            "Dabrowska": _Grid,
            "Prentice-Cai": _VolterraGrid,
            "Lin-Ying": _LinYingGrid,
        }[self.estimator]
        self._grid: Any = kind(x, c)
        self._est = self._grid.counts(n.astype(float))
        self.t1 = np.unique(x[c[:, 0] == 0, 0])
        self.t2 = np.unique(x[c[:, 1] == 0, 1])

    # ------------------------------------------------------------------
    # The estimate

    @keeps_query_shape(point_ndim=1)
    def sf(self, x: npt.ArrayLike) -> npt.NDArray:
        r"""
        The joint survival :math:`P(T_1 > x_1, T_2 > x_2)` at the points
        ``x``: a pair, or one pair per row. A coordinate before the first
        failure of its series leaves the other's margin (``sf([[s, 0]])``
        is the first series' Kaplan-Meier estimate at ``s``; for Lin-Ying
        it is that estimator's own margin, see :meth:`Dabrowska.fit`).
        """
        pts = _points(x)
        return _evaluate(self._grid, self._est, pts)[0]

    @keeps_query_shape(point_ndim=1)
    def ff(self, x: npt.ArrayLike) -> npt.NDArray:
        r"""
        The joint distribution function :math:`P(T_1 \le x_1, T_2 \le
        x_2) = 1 - S_1(x_1) - S_2(x_2) + S(x_1, x_2)` at the points ``x``.
        (Not the complement of :meth:`sf`.)
        """
        pts = _points(x)
        S, S1, S2 = _evaluate(self._grid, self._est, pts)
        return 1 - S1 - S2 + S

    @keeps_query_shape(point_ndim=1)
    def cdf(self, x: npt.ArrayLike) -> npt.NDArray:
        """Alias of :meth:`ff`, the name the copula models use."""
        return self.ff(x)

    def sf_grid(self, t1: npt.ArrayLike, t2: npt.ArrayLike) -> npt.NDArray:
        """
        The joint survival on a grid: ``out[i, j]`` is
        :math:`P(T_1 > t_1[i], T_2 > t_2[j])`.
        """
        a = np.atleast_1d(np.asarray(t1, dtype=float)).ravel()
        b = np.atleast_1d(np.asarray(t2, dtype=float)).ravel()
        A, B = np.meshgrid(a, b, indexing="ij")
        pts = np.column_stack([A.ravel(), B.ravel()])
        return self.sf(pts).reshape(len(a), len(b))

    @property
    def margins(self) -> list:
        """The two margins as fitted ``KaplanMeier`` models. For the
        Dabrowska and Prentice-Cai estimates the joint survival at ``(s,
        0)`` and ``(0, t)`` is exactly theirs; Lin-Ying's own margin
        (``sf([[s, 0]])``) is a weighted empirical one that differs from
        Kaplan-Meier's a little where there is censoring."""
        from surpyval import KaplanMeier

        return [
            KaplanMeier.fit(self.x[:, d], c=self.c[:, d], n=self.n)
            for d in range(2)
        ]

    # ------------------------------------------------------------------
    # Uncertainty

    def influence(self, x: npt.ArrayLike, on: str = "sf") -> npt.NDArray:
        r"""
        The empirical influence of each data row on the estimate at the
        points ``x``: ``out[i, j]`` is the derivative of the estimate at
        point ``j`` in the weight of one unit of row ``i``.

        The estimate is a smooth function of the rows' weights, and these
        are its exact derivatives (worked out from the counts, not by
        finite differences). The infinitesimal-jackknife variance
        :math:`\sum_i n_i\, \text{out}[i, j]^2` is the variance of the
        estimator's influence function, which :meth:`cb` uses.

        Only for Dabrowska's estimator: the derivatives are worked out for
        its product form. The other estimators refuse, naming the
        bootstrap.
        """
        self._check_wald("influence")
        on = _check_on(on)
        pts = _points(x)
        if pts.ndim != 2:
            pts = pts.reshape(-1, 2)
        out = np.empty((len(self.x), len(pts)))
        step = max(1, _CHUNK // max(len(self.x), 1))
        for s in range(0, len(pts), step):
            out[:, s : s + step] = self._influence(pts[s : s + step], on)
        return out

    def _influence(self, pts: npt.NDArray, on: str) -> npt.NDArray:
        g, e = self._grid, self._est
        ks = np.searchsorted(g.U, pts[:, 0], side="right") - 1
        lt = np.searchsorted(g.V, pts[:, 1], side="right") - 1
        S, S1, S2 = _evaluate(g, e, pts)
        r1, r2 = g.r1[:, None], g.r2[:, None]
        f1, f2 = g.f1[:, None], g.f2[:, None]
        k_ = ks[None, :]
        l_ = lt[None, :]

        # The margins: d log S1 / dw_i
        def margin(Y: Any, D: Any, r: Any, f: Any, q: Any) -> npt.NDArray:
            iYD, iY = _inv(Y - D), _inv(Y)
            cum = np.concatenate([[0.0], np.cumsum(iYD - iY)])
            kmin = np.minimum(q, r)
            term = cum[kmin + 1]
            own = np.where(f & (r <= q) & (r >= 0), iYD[np.maximum(r, 0)], 0.0)
            return np.where(q >= 0, term - own, 0.0)

        d1 = margin(e["Y1"], e["D1"], r1, f1, k_)
        d2 = margin(e["Y2"], e["D2"], r2, f2, l_)
        live = e["live"]
        iY = np.where(live, _inv(e["Y"]), 0.0)
        iM = np.where(live, _inv(e["M"]), 0.0)
        iA = np.where(live, _inv(e["YA"]), 0.0)
        iB = np.where(live, _inv(e["YB"]), 0.0)
        Z = iY + iM - iA - iB
        K, L = len(g.U), len(g.V)
        dp = np.zeros((len(g.x), len(pts)))
        if K and L:
            pad = np.zeros((K + 1, L + 1))
            pad[1:, 1:] = _cum2(Z)
            kk = np.minimum(k_, r1)
            ll = np.minimum(l_, r2)
            ok = (kk >= 0) & (ll >= 0)
            dp += np.where(ok, pad[kk + 1, ll + 1], 0.0)
            # A unit failing in the first series at its own row of the grid
            row = np.zeros((K, L + 1))
            row[:, 1:] = np.cumsum(iA - iM, axis=1)
            sel = f1 & (r1 <= k_) & (r1 >= 0) & (ll >= 0)
            dp += np.where(
                sel, row[np.maximum(r1, 0), np.maximum(ll, -1) + 1], 0.0
            )
            col = np.zeros((K + 1, L))
            col[1:, :] = np.cumsum(iB - iM, axis=0)
            sel = f2 & (r2 <= l_) & (r2 >= 0) & (kk >= 0)
            dp += np.where(
                sel, col[np.maximum(kk, -1) + 1, np.maximum(r2, 0)], 0.0
            )
            sel = f1 & f2 & (r1 <= k_) & (r2 <= l_) & (r1 >= 0) & (r2 >= 0)
            dp += np.where(sel, iM[np.maximum(r1, 0), np.maximum(r2, 0)], 0.0)
        dS = S[None, :] * (d1 + d2 + dp)
        if on == "sf":
            return dS
        return -S1[None, :] * d1 - S2[None, :] * d2 + dS

    def standard_errors(self, x: npt.ArrayLike, on: str = "sf") -> Any:
        r"""
        The standard error of the estimate at the points ``x`` from its
        influence function (the infinitesimal jackknife,
        :math:`\sqrt{\sum_i n_i\, \text{IF}_i^2}`; see :meth:`influence`).
        """
        self._check_wald("standard_errors")
        single = np.ndim(x) == 1
        pts = _points(x)
        shape = np.shape(pts)[:-1]
        flat = pts.reshape(-1, 2)
        infl = self.influence(flat, on)
        se = np.sqrt(np.sum(self.n[:, None] * infl**2, axis=0))
        return se[0] if single else se.reshape(shape)

    def cb(
        self,
        x: npt.ArrayLike,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str | None = None,
        n_boot: int = 200,
        random_state: Any = None,
    ) -> npt.NDArray:
        r"""
        Confidence bounds on the joint survival ``sf`` or the joint
        distribution function ``ff`` at the points ``x``.

        ``method="wald"`` (the default for Dabrowska's estimator) uses
        the standard error from the estimator's influence function
        (:meth:`standard_errors`) on the logit scale of the probability,
        so the bound stays in (0, 1), as the copula models' bounds do. At
        a point where the estimate is 0 or 1 the bound is that value.

        ``method="bootstrap"`` gives the percentile interval of the
        estimate over ``n_boot`` resamples of the units (the case
        bootstrap: pairs resampled together, keeping their dependence).
        It is the default, and the only method, for the Prentice-Cai and
        Lin-Ying estimators, whose influence functions are not built.

        Parameters
        ----------
        x : array like
            The points ``(x_1, x_2)``: a pair, or one pair per row.
        on : {'sf', 'ff'}, optional
            The joint survival (default) or the joint distribution
            function.
        alpha_ci : float, optional
            The significance level: 0.05 (the default) gives a 95% bound.
        bound : {'two-sided', 'upper', 'lower'}, optional
            Defaults to two-sided.
        method : {'wald', 'bootstrap'}, optional
            ``'wald'`` for Dabrowska's estimator, ``'bootstrap'`` for the
            others, by default.
        n_boot : int, optional
            The number of bootstrap resamples. Default 200.
        random_state : int or numpy Generator, optional
            The seed of the resamples.

        Returns
        -------
        numpy array
            One row per point holding ``[lower, upper]`` for a two-sided
            bound, else the one bound per point.
        """
        check_alpha_ci(alpha_ci)
        on = _check_on(on)
        if method is None:
            method = "wald" if self.estimator == "Dabrowska" else "bootstrap"
        if bound not in ("two-sided", "upper", "lower"):
            raise ValueError(
                "'bound' must be 'two-sided', 'upper' or 'lower', not "
                f"{bound!r}."
            )
        if method not in ("wald", "bootstrap"):
            raise ValueError(
                f"'method' must be 'wald' or 'bootstrap', not {method!r}. "
                "A likelihood-ratio bound needs a likelihood, which a "
                "non-parametric joint survival does not have; a copula "
                "(surpyval.multivariate.Clayton and the others) has one."
            )
        single = np.ndim(x) == 1
        pts = _points(x)
        shape = np.shape(pts)[:-1]
        flat = pts.reshape(-1, 2)
        value = self.sf(flat) if on == "sf" else self.ff(flat)
        sides = (
            [alpha_ci / 2, 1 - alpha_ci / 2]
            if bound == "two-sided"
            else [alpha_ci if bound == "lower" else 1 - alpha_ci]
        )
        if method == "wald":
            self._check_wald("cb(method='wald')")
            se = np.atleast_1d(self.standard_errors(flat, on))
            inside = (value > 0) & (value < 1)
            p = np.where(inside, value, 0.5)
            logit = np.log(p / (1 - p))
            scale = se / (p * (1 - p))
            ends = []
            for q in sides:
                z = norm.ppf(q)
                e = 1 / (1 + np.exp(-(logit + z * scale)))
                ends.append(np.where(inside, e, value))
            out = np.column_stack(ends)
        else:
            if int(n_boot) < 1:
                raise ValueError(
                    f"'n_boot' must be a positive whole number; got {n_boot}."
                )
            # Each resample builds only the corner of the grid the points
            # are in
            grid = self._grid
            seen = flat[~np.isnan(flat).any(axis=1)]
            if len(seen):
                grid = grid.upto(*np.max(seen, axis=0))

            def stat(est: dict) -> npt.NDArray:
                S, S1, S2 = _evaluate(grid, est, flat)
                return S if on == "sf" else 1 - S1 - S2 + S

            draws = self._bootstrap(grid, stat, n_boot, random_state)
            out = np.quantile(draws, sides, axis=0).T
        if bound != "two-sided":
            out = out[:, 0]
            return out[0] if single else out.reshape(shape)
        return out[0] if single else out.reshape(shape + (2,))

    def _check_wald(self, what: str) -> None:
        if self.estimator != "Dabrowska":
            raise ValueError(
                f"{what} needs the estimator's influence function, which is "
                f"worked out only for Dabrowska's estimator, not "
                f"{self.estimator}'s. Use cb(..., method='bootstrap') (the "
                "case bootstrap), or estimator='dabrowska'."
            )

    def _bootstrap(
        self, grid: Any, stat: Any, n_boot: int, random_state: Any
    ) -> npt.NDArray:
        """``stat`` of the estimate over ``n_boot`` case resamples."""
        if int(n_boot) < 1:
            raise ValueError(
                f"'n_boot' must be a positive whole number; got {n_boot}."
            )
        rng = as_generator(random_state)
        total = int(np.sum(self.n))
        prob = self.n / total
        draws = []
        for _ in range(int(n_boot)):
            w = rng.multinomial(total, prob).astype(float)
            draws.append(stat(grid.counts(w)))
        return np.array(draws)

    # ------------------------------------------------------------------
    # Dependence

    def _region(self, t1_max: Any, t2_max: Any) -> tuple[float, float]:
        out = []
        for d, (given, t) in enumerate(
            zip((t1_max, t2_max), (self.t1, self.t2))
        ):
            if given is None:
                if not len(t):
                    raise ValueError(
                        f"Series {d} has no failures, so no region of the "
                        "joint distribution is seen; Kendall's tau cannot "
                        "be estimated from these data."
                    )
                out.append(float(t[-1]))
            else:
                out.append(float(given))
        return out[0], out[1]

    def _tau(self, grid: Any, est: dict, a: float, b: float) -> float:
        g1 = grid.U[grid.U <= a]
        g2 = grid.V[grid.V <= b]
        if not (len(g1) and len(g2)):
            return float("nan")
        p1 = np.concatenate([[-np.inf], g1])
        p2 = np.concatenate([[-np.inf], g2])
        A, B = np.meshgrid(p1, p2, indexing="ij")
        flat = np.column_stack([A.ravel(), B.ravel()])
        S = _evaluate(grid, est, flat)[0].reshape(len(p1), len(p2))
        # The mass at each grid point: the double difference of S
        p = S[:-1, :-1] - S[1:, :-1] - S[:-1, 1:] + S[1:, 1:]
        mass = float(np.sum(p))
        count = mass * float(np.sum(self.n))
        if not mass > 0 or count <= 1:
            return float("nan")
        q = p / mass
        # Mass strictly below-left, and strictly above-left, of each point
        cum = np.cumsum(np.cumsum(q, axis=0), axis=1)
        below = np.zeros_like(q)
        below[1:, 1:] = cum[:-1, :-1]
        col = np.cumsum(q, axis=0)  # rows <= i
        above_left = np.zeros_like(q)
        # rows < i, columns > j
        rows_before = np.zeros_like(q)
        rows_before[1:, :] = col[:-1, :]
        right = rows_before[:, ::-1].cumsum(axis=1)[:, ::-1]
        above_left[:, :-1] = right[:, 1:]
        concordant = 2 * float(np.sum(q * below))
        discordant = 2 * float(np.sum(q * above_left))
        return (concordant - discordant) * count / (count - 1)

    def kendall_tau(
        self, t1_max: float | None = None, t2_max: float | None = None
    ) -> float:
        r"""
        Kendall's tau of the two lifetimes from the estimated joint
        distribution, within the region the data support.

        Kendall's tau is the chance that two units are concordant (the one
        that fails first in one series fails first in the other too) less
        the chance they are discordant: 0 for unrelated lifetimes, up to 1
        (or down to -1) for lifetimes that always fall in the same (the
        opposite) order.

        Censoring hides the joint distribution past the last failures, so
        this is the tau of the lifetimes *given that both fail by*
        ``t1_max`` and ``t2_max`` (by default the last failure time of
        each series): the estimate's probability mass on the grid points
        in that rectangle, normalised to 1, and

        .. math::
            \hat\tau = \frac{\hat n}{\hat n - 1}
            \sum_{(i, j)} \sum_{(k, l)} q_{ij} q_{kl}
            \operatorname{sign}(u_i - u_k) \operatorname{sign}(v_j - v_l),

        with :math:`q` the normalised masses and :math:`\hat n` the
        estimated number of units in the rectangle (the total count times
        its mass). With no censoring the masses are the units' shares,
        :math:`\hat n` is the count in the rectangle, and this is exactly
        the sample's Kendall's tau-a (ties in either series count as
        neither concordant nor discordant); it is scipy's
        ``kendalltau`` (tau-b) when neither series has ties. With
        censoring it differs from the whole population's tau, which the
        data do not reach; a copula's ``kendall_tau()`` gives that by
        assuming a family. The estimate's negative masses (see the
        estimator's notes) are used as they are.

        Parameters
        ----------
        t1_max, t2_max : float, optional
            The corner of the region. Default: each series' last failure
            time.

        Returns
        -------
        float
            ``nan`` if the region holds no mass or about one unit.
        """
        a, b = self._region(t1_max, t2_max)
        return self._tau(self._grid, self._est, a, b)

    def kendall_tau_cb(
        self,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        t1_max: float | None = None,
        t2_max: float | None = None,
        n_boot: int = 200,
        random_state: Any = None,
    ) -> Any:
        """
        The case-bootstrap percentile bound on :meth:`kendall_tau` (pairs
        resampled together), over the same region in every resample.

        Returns
        -------
        numpy array or float
            ``[lower, upper]`` for a two-sided bound, else the one bound.
        """
        check_alpha_ci(alpha_ci)
        if bound not in ("two-sided", "upper", "lower"):
            raise ValueError(
                "'bound' must be 'two-sided', 'upper' or 'lower', not "
                f"{bound!r}."
            )
        a, b = self._region(t1_max, t2_max)
        grid = self._grid.upto(a, b)
        draws = self._bootstrap(
            grid, lambda e: self._tau(grid, e, a, b), n_boot, random_state
        )
        draws = draws[np.isfinite(draws)]
        if not len(draws):
            return (
                np.array([np.nan, np.nan])
                if bound == "two-sided"
                else float("nan")
            )
        if bound == "two-sided":
            return np.quantile(draws, [alpha_ci / 2, 1 - alpha_ci / 2])
        q = alpha_ci if bound == "lower" else 1 - alpha_ci
        return float(np.quantile(draws, q))

    # ------------------------------------------------------------------
    # Plotting and saving

    def plot(
        self,
        ax: Any = None,
        t1: npt.ArrayLike | None = None,
        t2: npt.ArrayLike | None = None,
        levels: Any = 10,
        filled: bool = True,
    ) -> Any:
        """
        A contour plot of the joint survival over the two lifetimes.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Draw on this; a new figure by default.
        t1, t2 : array_like, optional
            The grid of each series' times; by default from 0 to its
            largest observed time.
        levels : int or array_like, optional
            The contour levels (matplotlib's ``levels``). Default 10.
        filled : bool, optional
            Filled contours (default) or lines.

        Returns
        -------
        matplotlib Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        if t1 is None:
            t1 = np.linspace(0, float(np.max(self.x[:, 0])), 101)
        if t2 is None:
            t2 = np.linspace(0, float(np.max(self.x[:, 1])), 101)
        a, b = np.asarray(t1, float), np.asarray(t2, float)
        Z = self.sf_grid(a, b)
        if filled:
            # (values outside the levels are shaded, not left blank)
            cs = ax.contourf(a, b, Z.T, levels=levels, extend="both")
        else:
            cs = ax.contour(a, b, Z.T, levels=levels)
        if filled:
            ax.figure.colorbar(cs, ax=ax, label="joint survival")
        else:
            ax.clabel(cs, inline=True, fontsize=8)
        names = self.names or ("series 0", "series 1")
        ax.set_xlabel(str(names[0]))
        ax.set_ylabel(str(names[1]))
        ax.set_title(f"Joint survival ({self.estimator})")
        return ax

    def to_dict(self) -> dict:
        """
        Serialise to a plain dictionary. A non-parametric estimate is its
        data, so the dictionary holds the rows (``x``, ``c``, ``n``), and
        :meth:`from_dict` (or ``surpyval.from_dict``) recomputes the same
        estimate from them.
        """
        from surpyval.serialisation import stamp_schema

        out = {
            "parameterization": "bivariate-non-parametric",
            "estimator": self.estimator,
            "x": self.x.tolist(),
            "c": self.c.tolist(),
            "n": self.n.tolist(),
        }
        if self.common_censoring:
            out["common_censoring"] = True
        if self.names is not None:
            out["names"] = list(self.names)
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> BivariateNonParametric:
        """Rebuild the estimate from a :meth:`to_dict` dictionary."""
        names = model_dict.get("names")
        return cls(
            np.asarray(model_dict["x"], dtype=float),
            np.asarray(model_dict["c"], dtype=int),
            np.asarray(model_dict["n"], dtype=int),
            names=tuple(names) if names is not None else None,
            estimator=model_dict.get("estimator", "Dabrowska"),
            common_censoring=bool(model_dict.get("common_censoring", False)),
        )

    def __repr__(self) -> str:
        N = int(np.sum(self.n))
        cens = [int(np.sum(self.n[self.c[:, d] == 1])) for d in range(2)]
        names = self.names or ("series 0", "series 1")
        return (
            "Non-parametric joint survival of two lifetimes\n"
            f"Estimator          : {self.estimator}\n"
            f"Units              : {N}\n"
            f"Censored           : {cens[0]} in {names[0]}, "
            f"{cens[1]} in {names[1]}\n"
            f"Failure times      : {len(self.t1)} and {len(self.t2)}"
        )


def _points(x: npt.ArrayLike) -> npt.NDArray:
    pts = np.asarray(x, dtype=float)
    if pts.ndim == 1 and pts.shape[0] == 2:
        pts = pts.reshape(1, 2)
    if pts.ndim < 1 or pts.shape[-1] != 2:
        raise ValueError(
            "The points must have two coordinates each, a row per point "
            f"and a column per series: got shape {pts.shape}."
        )
    return pts


_ESTIMATORS = {
    "dabrowska": "Dabrowska",
    "prentice-cai": "Prentice-Cai",
    "volterra": "Prentice-Cai",
    "lin-ying": "Lin-Ying",
}


def _check_estimator(estimator: str) -> str:
    key = str(estimator).lower().replace("_", "-").replace(" ", "-")
    if key not in _ESTIMATORS:
        raise ValueError(
            "'estimator' must be 'dabrowska' (the default), 'prentice-cai' "
            f"(or 'volterra') or 'lin-ying', not {estimator!r}."
        )
    return _ESTIMATORS[key]


def _check_on(on: str) -> str:
    aliases = {"sf": "sf", "R": "sf", "ff": "ff", "F": "ff"}
    if on not in aliases:
        raise ValueError(
            f"'on' must be 'sf' (the joint survival) or 'ff' (the joint "
            f"distribution function), not {on!r}."
        )
    return aliases[on]


class Dabrowska:
    r"""
    Dabrowska's (1988) non-parametric estimator of the joint survival of
    two right-censored lifetimes.

    Two lifetimes measured on each unit (two eyes of a patient, twins, two
    components of one machine) are usually dependent: the chance both last
    past their times is not the product of the two chances. This estimates
    that joint survival, :math:`S(s, t) = P(T_1 > s, T_2 > t)`, without
    choosing a family for the dependence, as Kaplan-Meier estimates one
    lifetime's survival without choosing a distribution. Its margins are
    the two Kaplan-Meier estimates, and with no censoring it is exactly
    the empirical joint survival function: the fraction of units that
    outlived both times.

    The estimate is the product of the two margins and, at each pair of
    failure times, a factor for how much more (or less) often the two
    series fail together than they would if they were unrelated; see
    :mod:`surpyval.multivariate.nonparametric.dabrowska` for the formula.

    Notes
    -----
    Like every non-parametric estimator of a bivariate survival function,
    it can give a little negative probability to some rectangles (its
    implied density is not always positive) in small or heavily censored
    samples; this is a known property of the estimator, not an error, and
    shrinks as the sample grows. The estimate is a step function on the
    grid of the two series' failure times, and past the last time of a
    series it stays at its last value.

    Censoring may differ between the two series of a unit (each has its
    own censoring time). Only right censoring is taken: a copula with
    non-parametric margins handles none of the others either, but a
    copula with parametric margins (``surpyval.multivariate.Clayton.fit``
    and the other families) takes left and interval censoring and
    truncation.

    References
    ----------
    Dabrowska, D. M. (1988). Kaplan-Meier estimate on the plane. *The
    Annals of Statistics*, 16(4), 1475-1489.

    Examples
    --------
    With no censoring it is the empirical joint survival:

    >>> import numpy as np
    >>> from surpyval.multivariate import Dabrowska
    >>> x = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])
    >>> model = Dabrowska.fit(x)
    >>> model.sf([[1.5, 1.5], [2.5, 2.5], [0.0, 3.5]])
    array([0.5 , 0.5 , 0.25])

    Paired lifetimes with censoring, and the joint survival with a 95%
    bound:

    >>> c = np.array([[0, 0], [0, 1], [1, 0], [0, 0]])
    >>> model = Dabrowska.fit(x, c=c)
    >>> model.sf([[2.5, 2.5]])
    array([0.5])
    >>> model.margins[0].sf(2.5), model.margins[1].sf(2.5).round(4)
    (np.float64(0.5), np.float64(0.6667))
    >>> model.cb([[2.5, 2.5]]).round(4)
    array([[0.1235, 0.8765]])

    Both lasting past 2.5 is more likely (0.5) than the product of the
    margins (1/3): the two lifetimes are positively dependent.
    """

    @classmethod
    def fit_from_df(
        cls,
        df: Any,
        x_cols: list[str],
        c_cols: list[str] | None = None,
        n_col: str | None = None,
        estimator: str = "dabrowska",
        common_censoring: bool = False,
    ) -> BivariateNonParametric:
        """
        Estimate the joint survival from the columns of a
        :class:`pandas.DataFrame`, one row per unit: ``x_cols`` names the
        column of each series' times, ``c_cols`` of each series'
        censoring flags (all failures by default) and ``n_col`` the
        counts, as the copulas' ``fit_from_df`` does. ``estimator`` and
        ``common_censoring`` are as for :meth:`fit`.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval.multivariate import Dabrowska
        >>> df = pd.DataFrame({"left": [1.0, 2.0, 3.0, 4.0],
        ...                    "right": [2.0, 1.0, 4.0, 3.0]})
        >>> Dabrowska.fit_from_df(df, x_cols=["left", "right"]).sf(
        ...     [[2.5, 2.5]]
        ... )
        array([0.5])
        """
        from surpyval.utils.dataframe import (
            call_fit,
            frame_column,
            frame_columns,
            require_frame,
        )

        df = require_frame(df)
        arrays: dict[str, Any] = {
            "x": frame_columns(df, x_cols, "x_cols", time=True).astype(float)
        }
        if c_cols is not None:
            arrays["c"] = frame_columns(df, c_cols, "c_cols", time=False)
        if n_col is not None:
            arrays["n"] = frame_column(df, n_col, "n_col")
        names = {"x": "x_cols", "c": "c_cols", "n": "n_col"}
        options = {
            "estimator": estimator,
            "common_censoring": common_censoring,
        }
        return call_fit(cls, arrays, names, options)

    @classmethod
    def fit(
        cls,
        x: Any,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        estimator: str = "dabrowska",
        common_censoring: bool = False,
    ) -> BivariateNonParametric:
        r"""
        Estimate the joint survival of two right-censored lifetimes.

        Parameters
        ----------
        x : array_like, shape (N, 2), or MultivariateSurpyvalData
            The two lifetimes of each unit: an array with a row per unit
            and a column per series, or a list of the two series (as the
            copulas take them), or a
            :class:`~surpyval.multivariate.MultivariateSurpyvalData` with
            two series, when the other arguments are left out.
        c : array_like, shape (N, 2) or (2,), optional
            The censoring flag of each lifetime: 0 a failure, 1 right
            censored. One pair applies to every row. All failures by
            default.
        n : array_like, shape (N,), optional
            The count of each row (1 each by default).
        estimator : {'dabrowska', 'prentice-cai', 'lin-ying'}, optional
            Which estimate (all three are the empirical joint survival
            when nothing is censored):

            - ``'dabrowska'`` (the default; Dabrowska 1988): the two
              Kaplan-Meier margins times a cross-ratio factor at each
              pair of failure times. Bounds: Wald (influence function)
              and bootstrap.
            - ``'prentice-cai'`` (or ``'volterra'``; Prentice and Cai
              1992): the two Kaplan-Meier margins and the covariance of
              the two series' failure counts, accumulated through a
              Volterra integral equation. Same data as Dabrowska's (each
              lifetime its own censoring time). Bounds: bootstrap.
            - ``'lin-ying'`` (Lin and Ying 1993): the fraction of units
              seen past both times, divided by the chance follow-up lasts
              past the later one. Only for *common* censoring (one
              censoring time shared by both lifetimes of a unit), which
              must be declared with ``common_censoring=True``. Bounds:
              bootstrap. Its margins are not the Kaplan-Meier estimates
              but a weighted empirical estimate, :math:`\#\{X_1 > s\} /
              (N \hat G(s))`, which agrees with them where nothing is
              censored before ``s``.
        common_censoring : bool, optional
            Declare that both lifetimes of each unit were followed until
            the same time, so a unit's censored lifetimes share one
            censoring time and a failure was seen no later than it.
            Required by ``'lin-ying'`` (the data cannot show it: two
            failures say nothing of the follow-up). When ``True``, data
            that contradict it (a pair censored at two different times, or
            censored before the other lifetime's failure) are refused.

        Truncation and left or interval censoring are not taken: a
        ``MultivariateSurpyvalData`` holding them is refused, naming the
        copula fits that take them.

        Returns
        -------
        BivariateNonParametric
            The estimate, with ``sf``, ``ff``, ``sf_grid``, ``cb``,
            ``margins`` and ``plot``.

        Raises
        ------
        ValueError
            If the data do not have exactly two series, or have left or
            interval censoring or truncation, or the data's own checks
            fail (see ``MultivariateSurpyvalData``); for ``'lin-ying'``,
            if common censoring is not declared or the data contradict it.
        """
        from surpyval.multivariate.parametric.data import (
            MultivariateSurpyvalData,
        )

        if isinstance(x, MultivariateSurpyvalData):
            if any(v is not None for v in (c, n)):
                raise ValueError(
                    "Give either a MultivariateSurpyvalData as 'x' or the "
                    "arrays (x, c, n), not both."
                )
            data = x
        else:
            data = MultivariateSurpyvalData(x, c=c, n=n)
        if data.D != 2:
            raise ValueError(
                f"Dabrowska's estimator is of two lifetimes; the data have "
                f"{data.D} series. Fit it to two at a time, or a Gaussian "
                "or Student-t copula or a vine to all of them "
                "(surpyval.multivariate.Gaussian, Vine)."
            )
        other = np.isin(data.c, (-1, 2))
        if other.any():
            d = int(np.flatnonzero(other.any(axis=0))[0])
            kind = "left" if (data.c[:, d] == -1).any() else "interval"
            raise ValueError(
                f"Series {d} has {kind}-censored values; Dabrowska's "
                "estimator takes right censoring only (c of 0 or 1). A "
                "copula with parametric margins takes left and interval "
                "censoring: surpyval.multivariate.Clayton.fit(x, c=c, "
                "xl=xl, xr=xr, margins=[surpyval.Weibull, "
                "surpyval.Weibull]), or another family."
            )
        if np.isfinite(data.t).any():
            raise ValueError(
                "The data are truncated (t); Dabrowska's estimator takes "
                "none. A copula with parametric margins takes truncation: "
                "surpyval.multivariate.Clayton.fit(x, c=c, t=t, "
                "margins=[surpyval.Weibull, surpyval.Weibull]), or another "
                "family."
            )
        if (data.x < 0).any():
            warnings.warn(
                "Some lifetimes are negative; the estimate is computed as "
                "given, but a lifetime is usually at least 0. Check the "
                "time origin.",
                UserWarning,
                stacklevel=caller_stacklevel(),
            )
        return BivariateNonParametric(
            data.x.astype(float),
            data.c.astype(int),
            np.asarray(data.n).astype(int),
            names=data.names,
            estimator=estimator,
            common_censoring=common_censoring,
        )
