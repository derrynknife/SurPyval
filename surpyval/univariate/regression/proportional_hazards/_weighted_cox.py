"""The Cox proportional hazards model with weights ``w``.

The weighted partial likelihood, with each row's risk-set and death terms
multiplied by ``n w``, and ties handled as R's ``coxph`` handles them with
weights: Breslow's denominator, or Efron's, in which each of the ``m``
tied deaths (``m`` the units, their counts) is taken at their mean weight,

    l = sum_j [ sum_{i in D_j} n_i w_i eta_i
                - wbar_j sum_{k < m_j} log(S0_j - (k / m_j) D0_j) ],

``S0_j`` the weighted sum of ``exp(eta)`` over the risk set at the event
time ``t_j``, ``D0_j`` over its deaths and ``wbar_j`` the deaths' mean
weight. The covariance is the robust (Lin and Wei, 1989) sandwich, with
each unit's score residual ``L_i``, its dfbeta ``D_i = L_i I^-1`` and
``V = sum_i n_i w_i^2 D_i' D_i``, or the sums of ``n_i w_i D_i`` over the
clusters given (R's ``coxph(..., weights = w, robust = TRUE)``). It is the
only covariance a weighted fit offers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


class WeightedCox:
    """The weighted partial likelihood of right censored, left truncated
    rows (``x``, ``c``, ``n``, ``w``, ``tl``, ``Z``) and its derivatives."""

    def __init__(
        self,
        x: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        w: npt.NDArray,
        tl: npt.NDArray,
        Z: npt.NDArray,
        tie_method: str,
    ) -> None:
        self.x = np.asarray(x, dtype=float)
        self.c = np.asarray(c)
        self.n = np.asarray(n, dtype=float)
        self.w = np.asarray(w, dtype=float)
        self.tl = np.asarray(tl, dtype=float)
        self.Z = np.asarray(Z, dtype=float)
        self.omega = self.n * self.w
        self.efron = tie_method == "efron"
        died = self.c == 0
        self.died = died
        self.times = np.unique(self.x[died])
        k = self.times.size
        # Each event time's deaths: their weighted count, their number of
        # units and the index of each row's own event time.
        self.event_index = np.searchsorted(self.times, self.x)
        on_time = died & (self.event_index < k)
        idx = self.event_index[on_time]
        self.m = np.bincount(idx, weights=self.n[on_time], minlength=k)
        self.d_w = np.bincount(idx, weights=self.omega[on_time], minlength=k)
        self.wbar = self.d_w / self.m
        # Where each row leaves and enters, on the event-time axis: at
        # risk at t_j for tl < t_j <= x.
        self.last = np.searchsorted(self.times, self.x, side="right")
        self.first = np.searchsorted(self.times, self.tl, side="right")

    def _risk_sums(self, values: npt.NDArray) -> npt.NDArray:
        """``sum`` of ``values`` (one row per data row, any trailing
        shape) over the risk set at each event time."""
        k = self.times.size
        shape = (k + 1,) + values.shape[1:]
        out = np.zeros(shape)
        np.add.at(out, self.last, values)
        leave = np.cumsum(out[::-1], axis=0)[::-1][1:]  # x >= t_j
        inn = np.zeros(shape)
        np.add.at(inn, self.first, values)
        entered = np.cumsum(inn[::-1], axis=0)[::-1][1:]  # tl >= t_j
        # Rows with x >= t_j, less those not yet entered (tl >= t_j)
        return leave - entered

    def _death_sums(self, values: npt.NDArray) -> npt.NDArray:
        k = self.times.size
        out = np.zeros((k,) + values.shape[1:])
        rows = self.died
        np.add.at(out, self.event_index[rows], values[rows])
        return out

    def _ladders(self, j: int) -> npt.NDArray:
        """The Efron fractions ``k / m`` of event time ``j`` (zeros under
        Breslow)."""
        m = int(round(self.m[j]))
        if not self.efron or m <= 1:
            return np.zeros(max(m, 1))
        return np.arange(m) / m

    def derivatives(
        self, beta: npt.NDArray
    ) -> tuple[float, npt.NDArray, npt.NDArray]:
        """``(log-likelihood, score, information)`` at ``beta``."""
        Z = self.Z
        eta = Z @ beta
        r = self.omega * np.exp(eta)
        S0 = self._risk_sums(r)
        S1 = self._risk_sums(r[:, None] * Z)
        S2 = self._risk_sums(r[:, None, None] * Z[:, :, None] * Z[:, None, :])
        D0 = self._death_sums(r)
        D1 = self._death_sums(r[:, None] * Z)
        D2 = self._death_sums(r[:, None, None] * Z[:, :, None] * Z[:, None, :])
        died = self.died
        ll = float(np.sum(self.omega[died] * eta[died]))
        score = (self.omega[died][:, None] * Z[died]).sum(axis=0)
        p = Z.shape[1]
        info = np.zeros((p, p))
        for j in range(self.times.size):
            f = self._ladders(j)
            den = S0[j] - f * D0[j]
            num = S1[j][None, :] - f[:, None] * D1[j][None, :]
            sq = S2[j][None] - f[:, None, None] * D2[j][None]
            wbar = self.wbar[j]
            ll -= wbar * float(np.sum(np.log(den)))
            score -= wbar * (num / den[:, None]).sum(axis=0)
            mean = num / den[:, None]
            info += wbar * (
                (sq / den[:, None, None]).sum(axis=0)
                - (mean[:, :, None] * mean[:, None, :]).sum(axis=0)
            )
        return ll, score, info

    def neg_ll(self, beta: npt.ArrayLike) -> float:
        """The negative weighted partial log-likelihood at ``beta``."""
        return -self.derivatives(np.asarray(beta, dtype=float))[0]

    def score_residuals(self, beta: npt.NDArray) -> npt.NDArray:
        """Each row's score residual for one unit of it, ``L_i``: its
        derivative of the partial likelihood's score in its weight (R's
        ``residuals(fit, type = "score")``)."""
        Z = self.Z
        eta = Z @ beta
        risk = np.exp(eta)
        r = self.omega * risk
        S0 = self._risk_sums(r)
        S1 = self._risk_sums(r[:, None] * Z)
        D0 = self._death_sums(r)
        D1 = self._death_sums(r[:, None] * Z)
        k, p = self.times.size, Z.shape[1]
        # Per event time: the hazard increments and their means, for the
        # units at risk that do not die there (c0, c1) and those that do
        # (e0, e1, the Efron share 1 - k/m), and the deaths' mean
        # covariate (zbar).
        c0, e0 = np.zeros(k), np.zeros(k)
        c1, e1, zbar = np.zeros((k, p)), np.zeros((k, p)), np.zeros((k, p))
        for j in range(k):
            f = self._ladders(j)
            den = S0[j] - f * D0[j]
            mean = (S1[j][None, :] - f[:, None] * D1[j][None, :]) / den[
                :, None
            ]
            h = self.wbar[j] / den
            c0[j] = h.sum()
            c1[j] = (h[:, None] * mean).sum(axis=0)
            e0[j] = ((1 - f) * h).sum()
            e1[j] = (((1 - f) * h)[:, None] * mean).sum(axis=0)
            zbar[j] = mean.mean(axis=0)
        C0 = np.concatenate([[0.0], np.cumsum(c0)])
        C1 = np.vstack([np.zeros((1, p)), np.cumsum(c1, axis=0)])
        # Event times strictly before each row's exit, from its entry
        own = (self.event_index < k) & (
            self.times[np.minimum(self.event_index, k - 1)] == self.x
        )
        before = np.where(own, self.event_index, self.last)
        start = self.first
        before = np.maximum(before, start)
        H0 = C0[before] - C0[start]
        H1 = C1[before] - C1[start]
        # The row's own event time, if it is at risk there
        j_own = np.minimum(self.event_index, k - 1)
        at_own = own & (start <= self.event_index)
        died = self.died
        own0 = np.where(died, e0[j_own], c0[j_own]) * at_own
        own1 = np.where(died[:, None], e1[j_own], c1[j_own]) * at_own[:, None]
        resid = -risk[:, None] * (Z * (H0 + own0)[:, None] - (H1 + own1))
        resid[died] += Z[died] - zbar[self.event_index[died]]
        return resid

    def baseline(
        self, beta: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """``(times, at risk, deaths, hazard increments)`` of the baseline
        of a unit at ``Z = 0`` (the rows' ``Z`` as given): Breslow's, or
        Efron's increments ``sum_k wbar / (S0 - (k/m) D0)``."""
        r = self.omega * np.exp(self.Z @ beta)
        S0 = self._risk_sums(r)
        D0 = self._death_sums(r)
        h0 = np.zeros(self.times.size)
        for j in range(self.times.size):
            f = self._ladders(j)
            h0[j] = np.sum(self.wbar[j] / (S0[j] - f * D0[j]))
        at_risk = self._risk_sums(self.omega)
        return self.times, at_risk, self.d_w, h0


def newton(
    lik: WeightedCox, p: int, tol: float, max_iter: int = 100
) -> tuple[npt.NDArray, bool, int]:
    """Newton-Raphson with step halving on the weighted partial likelihood,
    from 0, stopping when a step is at most ``tol`` standard errors long
    (as the unweighted fit): ``(beta, converged, iterations)``."""
    beta = np.zeros(p)
    ll, score, info = lik.derivatives(beta)
    for it in range(1, max_iter + 1):
        try:
            step = np.linalg.solve(info, score)
        except np.linalg.LinAlgError:
            return beta, False, it
        length = float(np.sqrt(abs(step @ info @ step)))
        trial = beta + step
        for _ in range(30):
            ll_t, score_t, info_t = lik.derivatives(trial)
            if np.isfinite(ll_t) and ll_t >= ll - 1e-12 * abs(ll):
                break
            step = step / 2
            trial = beta + step
        beta, ll, score, info = trial, ll_t, score_t, info_t
        if length <= tol:
            return beta, True, it
    return beta, False, max_iter


def robust_covariance(
    lik: WeightedCox,
    beta: npt.NDArray,
    bread: npt.NDArray,
    cluster: npt.ArrayLike | None = None,
) -> npt.NDArray:
    """The robust (sandwich) covariance of a weighted Cox fit; ``bread``
    the inverse of its information. With ``cluster`` (one label per row)
    the dfbetas are summed within each cluster first."""
    dfbeta = lik.score_residuals(beta) @ bread
    if cluster is None:
        factor = lik.n * lik.w**2
        return (dfbeta * factor[:, None]).T @ dfbeta
    labels = np.asarray(cluster)
    if labels.shape[0] != dfbeta.shape[0]:
        raise ValueError(
            f"cluster has {labels.shape[0]} labels for {dfbeta.shape[0]} "
            "rows: give one label per row of the data fitted."
        )
    _, group = np.unique(labels, return_inverse=True)
    summed = np.zeros((group.max() + 1, dfbeta.shape[1]))
    np.add.at(summed, group, lik.omega[:, None] * dfbeta)
    return summed.T @ summed


def fit_weighted_cox(
    x: Any,
    Z: Any,
    c: Any,
    n: Any,
    tl: Any,
    w: Any,
    tie_method: str,
    tol: float,
    center: bool,
) -> Any:
    """``CoxPH.fit(..., w=w)``: the weighted fit and its model."""
    from surpyval.utils.weights import (
        effective_sample_size,
        refuse_weighted,
        warn_if_few_effective,
    )

    from ..semi_parametric_regression_model import (
        SemiParametricRegressionModel,
    )
    from .cox_ph import _require_an_event, validate_coxph

    if tie_method not in ("efron", "breslow"):
        raise refuse_weighted(
            f"tie_method={tie_method!r}",
            "the exact partial likelihoods have no weighted form here",
            "Use tie_method='efron' (the default) or 'breslow'.",
        )
    from surpyval.utils.weights import check_weights

    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    w_arr = check_weights(w, x_arr.shape[0])
    Z_in = np.asarray(Z, dtype=float)
    Z_rows = Z_in.reshape(-1, 1) if Z_in.ndim <= 1 else Z_in
    if Z_rows.shape[0] != w_arr.shape[0]:
        raise ValueError(
            f"Z has {Z_rows.shape[0]} rows for {w_arr.shape[0]} values of "
            "x: give one row of covariates per value."
        )
    # The weights travel as a last column of Z, so that the rows the
    # validation drops (a missing covariate, with its warning) or
    # reorders take their weights with them.
    x, c, n, tl, Z_aug = validate_coxph(
        x, c, n, np.column_stack([Z_rows, w_arr]), tl, tie_method
    )
    Z, w_arr = Z_aug[:, :-1], Z_aug[:, -1]
    _require_an_event(c, n)
    n = np.asarray(n, dtype=float)
    warn_if_few_effective(n, w_arr)
    Z = np.asarray(Z, dtype=float)
    omega = n * w_arr
    mean = (omega[:, None] * Z).sum(axis=0) / omega.sum()
    Zc = Z - mean
    lik = WeightedCox(x, c, n, w_arr, tl, Zc, tie_method)
    beta, converged, iters = newton(lik, Z.shape[1], tol)
    ll, score, info = lik.derivatives(beta)
    try:
        bread = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        raise ValueError(
            "The weighted partial likelihood's information is singular: a "
            "covariate column is constant or a combination of the others. "
            "Drop it and fit again."
        ) from None
    if not converged:
        import warnings

        warnings.warn(
            f"The weighted Cox fit did not converge in {iters} Newton "
            "steps: a covariate may separate the events from the "
            "survivors, so the likelihood has no finite maximum. The "
            "coefficients and their standard errors are not reliable.",
            UserWarning,
            stacklevel=3,
        )
    cov = robust_covariance(lik, beta, bread)
    se = np.sqrt(np.diag(cov))
    from scipy.stats import norm

    with np.errstate(all="ignore"):
        p_values = 2 * norm.sf(np.abs(beta / se))

    model = SemiParametricRegressionModel("Cox", "Semi-Parametric")
    model._neg_ll = -ll
    model.p_values = p_values
    model._se = se
    model._covariance = cov
    model.tie_method = tie_method
    model.baseline_method = (
        "fleming-harrington" if tie_method == "efron" else "breslow"
    )
    model.maximum = "verified" if converged else "unverified"
    model.beta = beta.copy()
    model.params = beta.copy()
    model._fit_data = {
        "x": np.asarray(x, dtype=float),
        "c": np.asarray(c, dtype=int),
        "n": n,
        "Z": Z,
        "tl": np.asarray(tl, dtype=float),
        "w": w_arr,
    }
    model._weighted_cox = lik
    model.neg_ll_of = lik.neg_ll
    model._ic_n = float(np.sum(n[np.asarray(c) == 0]))
    times, r, d, h0 = lik.baseline(beta)
    if center:
        model.center = mean
    else:
        # From the centre to Z = 0: h0(0) = h0(centre) exp(-beta' centre)
        shift = float(np.exp(-beta @ mean))
        h0 = h0 * shift
        model.center = np.zeros_like(mean)
    model._fit_center = mean
    model.x = times
    model.r = r
    model.d = d
    model.tl = np.asarray(tl, dtype=float)
    model.h0 = h0
    model.H0 = np.cumsum(h0)
    model.weighted = True
    model.effective_n = effective_sample_size(n, w_arr)
    model.res = None
    return model
