"""The variance of a Cox model's predicted cumulative hazard.

The cumulative hazard of a unit with covariates ``z`` is ``H(t | z) =
e^{beta'(z - c)} H_0(t)``, ``H_0`` the Breslow (or, after an Efron fit,
the Efron) baseline at the centre ``c``. Its variance has two parts
(Tsiatis 1981; Link 1984; Therneau and Grambsch 2000, section 10.2):

* the variance of the baseline with the coefficients known, the sum over
  the event times ``s <= t`` of ``dN(s) / S_0(s)^2`` (Efron's form spreads
  each time's ``m`` tied deaths over the ``m`` shrinking risk sets), and
* the coefficients' part, ``q(t)' V q(t)``, ``V`` their covariance and
  ``q(t) = sum_{s <= t} (z - xbar(s)) dH_0(s)``, ``xbar`` the risk-weighted
  mean of the covariates over the risk set,

each multiplied by ``e^{2 beta'(z - c)}``. It is what R's
``survfit.coxph`` computes as ``std.err`` (survival's ``agsurv`` and
``coxsurv.fit``). Each row of the data counts ``n w`` in every sum (``w``
the case weights of a weighted fit, 1 otherwise); the covariance is the
fit's own (the robust one for a weighted fit, as R's ``survfit`` uses
``fit$var``).

:func:`curve_components` returns, at each event time, the running sums
``A`` (the baseline cumulative hazard), ``V`` (the baseline's variance)
and ``B`` (``sum dH_0 (xbar - c)``, one column per covariate), from
which :func:`cumulative_hazard_se` forms the standard error at any
``z``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ._weighted_cox import WeightedCox


def curve_components(
    x: npt.NDArray,
    c: npt.NDArray,
    n: npt.NDArray,
    w: npt.NDArray | None,
    tl: npt.NDArray | None,
    Z: npt.NDArray,
    beta: npt.NDArray,
    mean: npt.NDArray,
    center: npt.NDArray,
    tie_method: str,
) -> dict[str, npt.NDArray]:
    """The running sums of the variance of ``H(t | z)`` at the event times
    of the rows (``x``, ``c``, ``n``, ``w``, ``tl``, ``Z``), for the
    coefficients ``beta`` (0 at an aliased column) and the baseline at
    ``center``. The sums are taken on the covariates less ``mean`` (where
    the fit centred, so ``exp`` cannot overflow) and moved to ``center``.

    Returns ``{"times", "A", "V", "B"}``: the event times, and at each the
    cumulative baseline hazard, its variance with the coefficients known,
    and ``sum dH_0 (xbar - center)`` (one column per covariate).
    """
    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    w_arr = np.ones_like(n) if w is None else np.asarray(w, dtype=float)
    tl_arr = np.full(x.shape, -np.inf) if tl is None else np.asarray(tl, float)
    Z = np.asarray(Z, dtype=float)
    beta = np.asarray(beta, dtype=float)
    mean = np.asarray(mean, dtype=float)
    center = np.asarray(center, dtype=float)
    Zc = Z - mean
    lik = WeightedCox(x, c, n, w_arr, tl_arr, Zc, tie_method)
    _, risk, shift = lik._shifted_risks(beta)
    r = lik.omega * risk
    S0 = lik._risk_sums(r)
    S1 = lik._risk_sums(r[:, None] * Zc)
    m = lik.m
    # The weighted deaths at each time, d_w = m * wbar
    d_w = lik.d_w
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        haz = d_w / S0
        var = d_w / S0**2
        xb = var[:, None] * S1
        if lik.efron:
            D0 = lik._death_sums(r)
            D1 = lik._death_sums(r[:, None] * Zc)
            for j in np.flatnonzero(m > 1):
                f = lik._ladders(int(j))
                den = S0[j] - f * D0[j]
                wbar = lik.wbar[j]
                haz[j] = wbar * np.sum(1.0 / den)
                var[j] = wbar * np.sum(1.0 / den**2)
                xb[j] = wbar * np.sum(
                    (S1[j] - f[:, None] * D1[j]) / den[:, None] ** 2, axis=0
                )
        # The sums were of exp(eta - shift) at the mean; at ``center``
        # every hazard is exp(beta'(center - mean)) times the one at the
        # mean.
        log_k = float((center - mean) @ beta) - shift
        k = np.exp(log_k)
        A_mean = np.cumsum(haz)
        A = A_mean * k
        V = np.cumsum(var) * k**2
        B = (np.cumsum(xb, axis=0) + np.outer(A_mean, mean - center)) * k
    return {"times": lik.times, "A": A, "V": V, "B": B}


def cumulative_hazard_se(
    parts: dict[str, npt.NDArray],
    t: npt.NDArray,
    rows: npt.NDArray,
    log_risk: npt.NDArray,
    center: npt.NDArray,
    covariance: npt.NDArray,
) -> npt.NDArray:
    """The standard error of ``H(t | z)`` at the times ``t`` paired with
    the covariate rows ``rows`` (``log_risk`` their ``beta'(z -
    center)``), from :func:`curve_components`'s ``parts``: 0 before the
    first event time, ``nan`` at a missing time or covariate."""
    times = parts["times"]
    idx = np.searchsorted(times, t, side="right") - 1
    before = idx < 0
    at = np.maximum(idx, 0)
    if times.size == 0:
        return np.where(np.isnan(t + log_risk), np.nan, 0.0)
    A, V, B = parts["A"][at], parts["V"][at], parts["B"][at]
    q = (rows - center) * A[:, None] - B
    # An aliased coefficient's row and column of the covariance are nan;
    # its coefficient is 0 in the predictions, and has no variance here
    # (R's survfit takes coef as 0 and its variance as 0).
    cov = np.where(np.isnan(covariance), 0.0, covariance)
    quad = np.einsum("ij,jk,ik->i", q, cov, q)
    with np.errstate(over="ignore", invalid="ignore"):
        var = np.exp(2.0 * log_risk) * (V + quad)
        se = np.sqrt(np.maximum(var, 0.0))
    se = np.where(before, 0.0, se)
    return np.where(np.isnan(t + log_risk), np.nan, se)


def survival_bounds(
    H: npt.NDArray,
    se: npt.NDArray,
    z: Any,
    bound_type: str,
) -> tuple[npt.NDArray, npt.NDArray]:
    """The bounds ``(H_lower, H_upper)`` on the cumulative hazard ``H``
    with standard error ``se``, ``z`` the normal quantile, of R's
    ``survfit`` ``conf.type``: ``"log"`` (``log S +- z se``, the upper
    survival bound at most 1), ``"log-log"`` (``log H +- z se / H``) or
    ``"plain"`` (``S +- z S se``, inside [0, 1]). Where ``se`` is 0 both
    are ``H``, as in R."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if bound_type == "log":
            lo = np.maximum(H - z * se, 0.0)
            hi = H + z * se
        elif bound_type == "log-log":
            ratio = np.where(H > 0, se / H, 0.0)
            lo = H * np.exp(-z * ratio)
            hi = H * np.exp(z * ratio)
        else:  # plain
            S = np.exp(-H)
            S_hi = np.minimum(S + z * S * se, 1.0)
            S_lo = np.maximum(S - z * S * se, 0.0)
            lo = -np.log(S_hi)
            hi = -np.log(S_lo)
        flat = se == 0
        lo = np.where(flat, H, lo)
        hi = np.where(flat, H, hi)
    return lo, hi
