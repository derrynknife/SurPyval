"""Kaplan-Meier, Nelson-Aalen and Fleming-Harrington with weights ``w``.

The estimate is the usual one on weighted numbers at risk and of deaths:
each unit counts ``w`` instead of 1. Its variance is the robust
(infinitesimal jackknife) one, as R's ``survfit(..., weights = w,
robust = TRUE)`` computes it: the derivative of the estimate in each unit's
weight, ``U_i(t)``, its influence, and ``Var = sum_i w_i^2 U_i(t)^2``
(``n_i`` identical units contribute ``n_i`` such terms). Greenwood's and
Aalen's formulas would treat the weights as counts of units seen.

For the log of the survival estimate, with ``r_k`` and ``d_k`` the
weighted numbers at risk and of deaths at time ``t_k`` and ``h_k = d_k /
r_k``:

- Kaplan-Meier, ``log S = sum log(1 - h_k)``:
  ``U_i(t) = -sum_{t_k <= t} (dN_i(t_k) - Y_i(t_k) h_k) / (r_k - d_k)``;
- Nelson-Aalen, ``log S = -sum h_k``:
  ``U_i(t) = -sum_{t_k <= t} (dN_i(t_k) - Y_i(t_k) h_k) / r_k``;
- Fleming-Harrington, whose increment with ``m_k`` tied deaths is
  ``g_k = sum_{l < m_k} a_k / (r_k - l a_k)``, ``a_k = d_k / m_k`` (each
  tied death taken at the mean weight, as R's ``ctype = 2`` does):
  ``U_i(t) = -sum_{t_k <= t} (dN_i(t_k) / m_k dg/da + Y_i(t_k) dg/dr)``,

where ``Y_i`` is 1 while unit ``i`` is at risk (``tl < t <= x``) and
``dN_i`` 1 at its failure.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

# Rows by times worked on at once in the influence sums.
_BLOCK = 2_000_000


class WeightedLadder:
    """The weighted ladder of a right censored, left truncated sample:
    ``x`` the distinct times, ``r`` and ``d`` the weighted numbers at risk
    and of deaths, ``m`` the number of units dying (their counts), and the
    units' own rows (``exit``, ``event``, ``entry``, ``n``, ``w``)."""

    def __init__(
        self,
        x: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        t: npt.NDArray,
        w: npt.NDArray,
    ) -> None:
        x = np.asarray(x, dtype=float)
        c = np.asarray(c)
        n = np.asarray(n, dtype=float)
        w = np.asarray(w, dtype=float)
        tl = np.asarray(t, dtype=float)[:, 0]
        times, idx = np.unique(x, return_inverse=True)
        nw = n * w
        died = c == 0
        self.x = times
        self.d = np.bincount(idx, weights=nw * died, minlength=times.size)
        self.m = np.bincount(idx, weights=n * died, minlength=times.size)
        leaving = np.bincount(idx, weights=nw, minlength=times.size)
        # Entered before each time: the (entry, exit] convention
        entry = np.searchsorted(times, tl, side="right")
        entered = np.cumsum(
            np.bincount(entry, weights=nw, minlength=times.size + 1)
        )[: times.size]
        self.r = entered - (np.cumsum(leaving) - leaving)
        self.exit = idx
        self.event = died
        self.entry = entry
        self.n = n
        self.w = w

    def log_sf_increments(self, estimator: str) -> npt.NDArray:
        """The increment of ``log S`` at each time."""
        r, d = self.r, self.d
        with np.errstate(all="ignore"):
            if estimator == "Kaplan-Meier":
                return np.where(d > 0, np.log1p(-d / r), 0.0)
            if estimator == "Nelson-Aalen":
                return np.where(d > 0, -d / r, 0.0)
            g, _, _ = self._fh_terms()
            return -g

    def _fh_terms(
        self,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Fleming-Harrington's ``g``, ``dg/da`` and ``dg/dr`` per time."""
        r, d, m = self.r, self.d, self.m
        g = np.zeros_like(r)
        dg_da = np.zeros_like(r)
        dg_dr = np.zeros_like(r)
        for k in np.flatnonzero(m > 0):
            a = d[k] / m[k]
            ell = np.arange(int(round(m[k])))
            den = r[k] - ell * a
            with np.errstate(all="ignore"):
                g[k] = np.sum(a / den)
                dg_da[k] = np.sum(r[k] / den**2)
                dg_dr[k] = -np.sum(a / den**2)
        return g, dg_da, dg_dr

    def _influence_coefficients(
        self, estimator: str
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """``(alpha, beta)``: a unit's influence on ``log S`` gains
        ``alpha_k`` at its failure at ``t_k`` and ``beta_k`` at each time
        ``t_k`` it is at risk."""
        r, d, m = self.r, self.d, self.m
        with np.errstate(all="ignore"):
            if estimator == "Kaplan-Meier":
                alpha = -1.0 / (r - d)
                beta = d / (r * (r - d))
            elif estimator == "Nelson-Aalen":
                alpha = -1.0 / r
                beta = d / r**2
            else:
                _, dg_da, dg_dr = self._fh_terms()
                alpha = -dg_da / np.where(m > 0, m, 1.0)
                beta = -dg_dr
        alpha = np.where(d > 0, alpha, 0.0)
        beta = np.where(d > 0, beta, 0.0)
        return alpha, beta

    def influence_blocks(
        self, estimator: str
    ) -> Iterator[tuple[npt.NDArray, npt.NDArray]]:
        """Each block of units' influence on ``log S`` at every time:
        ``(rows, U)``, ``U`` of shape ``(len(rows), len(x))``."""
        alpha, beta = self._influence_coefficients(estimator)
        B = np.concatenate([[0.0], np.cumsum(beta)])
        k = self.x.size
        cols = np.arange(k)
        step = max(1, _BLOCK // max(k, 1))
        for start in range(0, self.exit.size, step):
            rows = np.arange(start, min(start + step, self.exit.size))
            e = self.exit[rows][:, None]
            s = self.entry[rows][:, None]
            j = cols[None, :]
            with np.errstate(all="ignore"):
                at_risk = B[np.minimum(j, e) + 1] - B[s]
                U = np.where(j >= s, at_risk, 0.0)
                failed = (self.event[rows][:, None]) & (j >= e)
                U = U + np.where(failed, alpha[e], 0.0)
            yield rows, U

    def log_sf_variance(self, estimator: str) -> npt.NDArray:
        """The robust variance of ``log S`` at each time (``inf`` from a
        time at which every unit at risk fails, under Kaplan-Meier)."""
        var = np.zeros(self.x.size)
        factor = self.n * self.w**2
        for rows, U in self.influence_blocks(estimator):
            with np.errstate(all="ignore"):
                var += np.nansum(factor[rows][:, None] * U**2, axis=0)
        if estimator == "Kaplan-Meier":
            gone = np.cumsum((self.d > 0) & (self.r - self.d <= 0)) > 0
            var = np.where(gone, np.inf, var)
        return var

    def area_variance(
        self, estimator: str, R: npt.NDArray, tau: float
    ) -> float:
        """The robust variance of the area under the survival estimate
        ``R`` (on this ladder) from 0 to ``tau``: the influence of the area
        is ``sum_k R_k U_i(t_k) width_k``, the area of each step times the
        influence on ``log S`` there."""
        xs = self.x
        upper = np.minimum(np.concatenate([xs[1:], [np.inf]]), tau)
        widths = np.clip(upper - np.minimum(xs, tau), 0, None)
        area = widths * R
        factor = self.n * self.w**2
        total = 0.0
        for rows, U in self.influence_blocks(estimator):
            with np.errstate(all="ignore"):
                U_area = np.where(area > 0, U * area[None, :], 0.0).sum(1)
            total += float(np.sum(factor[rows] * U_area**2))
        return total
