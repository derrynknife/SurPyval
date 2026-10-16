"""Residuals and the proportional-hazards test for a fitted Cox model.

All quantities are computed from the per-observation training data retained
on the fitted model (``model._fit_data``) and its baseline
(``model.x`` / ``model.H0``). Risk sets respect delayed entry (``tl``) and
count weights (``n``), so the residuals are correct for left-truncated /
start-stop data too.

References
----------
- Grambsch & Therneau (1994), "Proportional hazards tests and diagnostics
  based on weighted residuals", Biometrika 81(3).
- Therneau & Grambsch (2000), *Modeling Survival Data*, ch. 4-6.
- Therneau, ``survival`` 3.0 (2019): ``cox.zph`` as the score test of the
  time-dependent term, which :func:`check_ph` computes.
"""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

from surpyval.utils.linalg import safe_inv
from surpyval.utils.validation import check_option, option_error

if TYPE_CHECKING:  # pragma: no cover
    from ..semi_parametric_regression_model import (
        SemiParametricRegressionModel,
    )


_RESIDUAL_KINDS = (
    "schoenfeld",
    "scaled_schoenfeld",
    "martingale",
    "deviance",
    "score",
    "dfbeta",
)

_TRANSFORMS = ("km", "rank", "identity", "log")


def _require_cox(model: "SemiParametricRegressionModel") -> dict:
    if getattr(model, "is_stratified", False):
        raise NotImplementedError(
            "Residuals, the proportional-hazards test and robust standard "
            "errors are not available for stratified Cox models (each stratum "
            "has its own baseline hazard)."
        )
    if getattr(model, "kind", None) != "Cox":
        raise NotImplementedError(
            "Residuals and the proportional-hazards test are implemented for "
            "Cox proportional-hazards models fit with CoxPH.fit / fit_from_df."
        )
    if not hasattr(model, "_fit_data"):
        # A Cox model rebuilt by from_dict keeps its coefficients and
        # baseline, not the per-observation data; it used to be told it
        # was not "fit with CoxPH.fit".
        raise ValueError(
            "Residuals, the proportional-hazards test and robust standard "
            "errors need the per-observation data the model was fitted to, "
            "which a model restored with from_dict / from_json does not "
            "carry. Call them on the fitted model, or refit."
        )
    aliased = np.flatnonzero(np.isnan(np.asarray(model.beta, dtype=float)))
    if aliased.size:
        raise ValueError(
            "Residuals, the proportional-hazards test and robust standard "
            "errors are not available for a model with aliased "
            "coefficients (column(s) {} of Z, whose coefficients are "
            "nan, #476). Refit without those columns.".format(aliased.tolist())
        )
    data = model._fit_data
    # The covariates centred as the fit centred them (#459), whichever
    # centre the model reports its baseline at (#463): every quantity here
    # depends on Z only through differences within a risk set, or through
    # exp(beta'Z) times a baseline recomputed here from the same rows;
    # centred, exp(beta'Z) cannot overflow either.
    center = getattr(model, "_fit_center", None)
    if center is None:
        center = getattr(model, "center", None)
    if center is None:
        return data
    return dict(data, Z=np.asarray(data["Z"], dtype=float) - center)


def _bucket_sums(
    bucket: np.ndarray, values: np.ndarray, size: int
) -> np.ndarray:
    """The sums of ``values`` (rows) by ``bucket``, in ``size`` buckets."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return np.bincount(bucket, weights=values, minlength=size)
    return np.column_stack(
        [
            np.bincount(bucket, weights=values[:, j], minlength=size)
            for j in range(values.shape[1])
        ]
    ).reshape(size, values.shape[1])


def _risk_set_means(
    data: dict, beta: np.ndarray, tie_method: str = "breslow"
) -> tuple:
    """Per distinct event time: risk-set covariate means and baseline-hazard
    step aggregates, weighted by ``n * exp(beta'Z)`` and respecting delayed
    entry.

    Under Efron ties the ``m`` tied deaths at a time are spread over ``m``
    pseudo-risk-sets ``S0 - (l/m) * S0_D`` (Therneau & Grambsch ch. 3), and
    every returned quantity is the corresponding step aggregate; with
    Breslow (or no ties) each event time is a single step and the classic
    forms are recovered. Using Breslow forms for an Efron fit made the
    residuals, ``check_ph`` and the robust covariance disagree with R /
    lifelines under heavy ties (#279).

    Returns ``(event_times, Zbar, S0, d, A, B, A_own, B_own)`` where per
    event time ``k``:

    - ``Zbar[k]``: tie-averaged risk-set covariate mean ``mean_l(E_l)``
      (the Schoenfeld expectation),
    - ``S0[k]``: full risk weight, ``d[k]``: n-weighted deaths,
    - ``A[k] = sum_l 1 / S0_l``: the baseline-hazard increment seen by a
      subject at risk through the whole tie group,
    - ``B[k] = sum_l E_l / S0_l`` (p-vector),
    - ``A_own[k] = sum_l (1 - l/m) / S0_l``: the partial increment a tied
      death itself accrues, and ``B_own[k]`` its E-weighted analogue.
    """
    x, c, n, Z, tl = (
        data["x"],
        data["c"],
        data["n"],
        data["Z"],
        data["tl"],
    )
    w = n * np.exp(Z @ beta)  # risk weight per observation
    efron = str(tie_method).lower() == "efron"

    event_times = np.unique(x[c == 0])
    K = event_times.size
    # The risk sums at every event time at once, as the fit and its
    # baseline compute them (``cox_at_risk_mask``'s convention, #299):
    # everyone with ``x >= tau`` (a suffix sum over the rows bucketed by
    # the number of event times at or before their exit) less those not
    # yet entered, ``tl >= tau`` (valid because ``tl < x`` on every row).
    # A loop over the event times with a mask of every row was O(n K):
    # 29 s for the martingale residuals of 3e4 rows with distinct times.
    wZ = w[:, None] * Z
    exit_k = np.searchsorted(event_times, x, side="right")
    entry_k = np.searchsorted(event_times, tl, side="right")

    def suffix(bucket: np.ndarray, values: np.ndarray) -> np.ndarray:
        # sum of ``values`` over the rows whose bucket is > k, for each k
        out = _bucket_sums(bucket, values, K + 1)
        return out[::-1].cumsum(axis=0)[::-1][1:]

    S0 = suffix(exit_k, w) - suffix(entry_k, w)
    S1 = suffix(exit_k, wZ) - suffix(entry_k, wZ)

    is_event = c == 0
    event_k = np.searchsorted(event_times, x[is_event])
    d = _bucket_sums(event_k, n[is_event], K)
    Zbar = S1 / S0[:, None]
    A = d / S0
    B = Zbar * (d / S0)[:, None]
    A_own = A.copy()
    B_own = B.copy()
    if efron:
        S0D = _bucket_sums(event_k, w[is_event], K)
        S1D = _bucket_sums(event_k, wZ[is_event], K)
        for k in np.flatnonzero(np.round(d) > 1):
            m = int(round(d[k]))
            fracs = np.arange(m) / m  # l/m for l = 0..m-1
            s0_l = S0[k] - fracs * S0D[k]
            e_l = (S1[k][None, :] - fracs[:, None] * S1D[k][None, :]) / s0_l[
                :, None
            ]
            inv_l = 1.0 / s0_l
            own_coef = 1.0 - fracs  # (m - l)/m
            Zbar[k] = e_l.mean(axis=0)
            A[k] = inv_l.sum()
            B[k] = (inv_l[:, None] * e_l).sum(axis=0)
            A_own[k] = (own_coef * inv_l).sum()
            B_own[k] = ((own_coef * inv_l)[:, None] * e_l).sum(axis=0)
    return event_times, Zbar, S0, d, A, B, A_own, B_own


def _information(model: "SemiParametricRegressionModel") -> np.ndarray:
    """The observed information (Hessian of the negative partial
    log-likelihood) at the fitted ``beta``."""
    # ``model.jac`` is the jac/hess closure returned by the fitter (the class
    # attribute is loosely annotated as an array); [1] is the Hessian.
    _, hess = model.jac(model.beta)
    return np.atleast_2d(hess)


def compute_residuals(
    model: "SemiParametricRegressionModel", kind: str = "martingale"
) -> np.ndarray:
    """
    Residuals for a fitted Cox proportional-hazards model.

    Parameters
    ----------
    kind : str
        One of ``"schoenfeld"``, ``"scaled_schoenfeld"`` (one row per event,
        ``p`` columns), ``"martingale"``, ``"deviance"`` (one value per
        observation), ``"score"`` or ``"dfbeta"`` (one row per observation,
        ``p`` columns).

    Returns
    -------
    numpy.ndarray
        The requested residuals. Schoenfeld residuals have one row per
        event, in the input order of the event rows (the rows with
        ``c == 0``) -- pair them with ``x[c == 0]``, and sort by it to plot
        them against time; the per-observation residuals follow the input
        order.

    Notes
    -----
    - **Schoenfeld** (``Z_i - E[Z | risk set]`` at each event) check the
      proportional-hazards assumption; see :func:`check_ph`.
    - **Martingale** = observed minus expected events; used for functional
      form. **Deviance** is a symmetrised martingale for outlier/influence
      screening.
    - **Score** residuals are each observation's contribution to the score;
      **dfbeta** (score residual times the covariance) approximates each
      observation's influence on the coefficients and underlies the
      cluster-robust variance.
    """
    if not (isinstance(kind, str) and kind.lower() in _RESIDUAL_KINDS):
        raise option_error(
            "kind", kind, _RESIDUAL_KINDS, "Case does not matter."
        )
    kind = kind.lower()
    data = _require_cox(model)
    beta = np.asarray(model.beta, dtype=float)
    x, c, n, Z, tl = (
        data["x"],
        data["c"],
        data["n"],
        data["Z"],
        data["tl"],
    )
    tie_method = getattr(model, "tie_method", "breslow")

    event_times, Zbar, S0, d, A, B, A_own, B_own = _risk_set_means(
        data, beta, tie_method
    )

    if kind in ("martingale", "deviance"):
        # Expected events from the tie-method-consistent baseline
        # increments (A); a tied death accrues only its own partial step
        # (A_own) at its event time under Efron (#279). With Breslow (or
        # no ties) A_own == A and this equals H0(x) - H0(tl).
        delta = (c == 0).astype(float)
        Lam = np.concatenate([[0.0], np.cumsum(A)])

        def _Lam_at(t: np.ndarray) -> np.ndarray:
            return Lam[np.searchsorted(event_times, t, side="right")]

        cum_haz = _Lam_at(x) - _Lam_at(tl)
        event_rows = np.flatnonzero(c == 0)
        own_idx = np.searchsorted(event_times, x[event_rows])
        cum_haz[event_rows] += A_own[own_idx] - A[own_idx]
        cum_haz = np.exp(Z @ beta) * cum_haz
        martingale = delta - cum_haz
        if kind == "martingale":
            return martingale
        # Deviance residual: a symmetrising transform of the martingale.
        with np.errstate(invalid="ignore", divide="ignore"):
            inner = martingale + delta * np.log(delta - martingale)
            dev = np.sign(martingale) * np.sqrt(-2.0 * inner)
        # delta - martingale == cum_haz; for a censored obs with 0 hazard the
        # log term vanishes (delta == 0), so guard the 0*log(0) case.
        dev = np.where(delta == 0, -np.sqrt(2.0 * cum_haz), dev)
        return dev

    if kind in ("schoenfeld", "scaled_schoenfeld"):
        # One residual per event observation: covariate minus the
        # (tie-averaged) risk-set mean at its event time.
        event_rows = np.flatnonzero(c == 0)
        zbar_idx = np.searchsorted(event_times, x[event_rows])
        sch = Z[event_rows] - Zbar[zbar_idx]
        if kind == "schoenfeld":
            return sch
        # Scaled Schoenfeld (Grambsch-Therneau): beta + d * V * s, where V is
        # the coefficient covariance and d the number of events. Centring on
        # beta makes the plot read directly as beta(t).
        n_events = int(d.sum())
        V = safe_inv(_information(model))
        return beta + n_events * (sch @ V)

    # Score / dfbeta residuals. The score residual for observation i is
    #   delta_i (Z_i - Zbar(x_i))
    #     - exp(beta'Z_i) * sum_{k in window} (Z_i * A_k - B_k)
    # where A_k / B_k are the per-event step aggregates from
    # _risk_set_means, and a tied death's own event time contributes its
    # partial (A_own, B_own) instead (#279). Breslow reduces to the
    # classic (Z_i - Zbar_k) d_k / S0_k form.
    w = np.exp(Z @ beta)
    score = np.zeros_like(Z, dtype=float)
    # First term (only for events).
    event_rows = np.flatnonzero(c == 0)
    zbar_idx = np.searchsorted(event_times, x[event_rows])
    score[event_rows] += Z[event_rows] - Zbar[zbar_idx]
    # Second term: subtract the expected score accrued while at risk, over
    # the event times in each row's window (tl, x]: differences of the
    # cumulative A and B (a loop over the rows was O(n K)).
    lo = np.searchsorted(event_times, tl, side="right")
    hi = np.searchsorted(event_times, x, side="right")
    in_window = hi > lo
    cum_A = np.concatenate([[0.0], np.cumsum(A)])
    cum_B = np.concatenate([np.zeros((1, B.shape[1])), np.cumsum(B, axis=0)])
    A_sum = np.where(in_window, cum_A[hi] - cum_A[lo], 0.0)
    B_sum = np.where(in_window[:, None], cum_B[hi] - cum_B[lo], 0.0)
    # A tied death accrues only its own partial step at its event time.
    own = in_window[event_rows]
    rows_own, k_own = event_rows[own], zbar_idx[own]
    A_sum[rows_own] += A_own[k_own] - A[k_own]
    B_sum[rows_own] += B_own[k_own] - B[k_own]
    score -= w[:, None] * (Z * A_sum[:, None] - B_sum)
    score = score * n[:, None]
    if kind == "score":
        return score
    return score @ safe_inv(_information(model))  # dfbeta


def robust_covariance(
    model: "SemiParametricRegressionModel",
    cluster: "npt.ArrayLike | None" = None,
) -> np.ndarray:
    """
    Cluster-robust ("sandwich") covariance of the Cox coefficients.

    The model-based covariance assumes independent observations. For
    clustered or correlated data (repeated events per subject, grouped
    sampling) the Lin-Wei (1989) robust variance is

    .. math::
        V_{robust} = \\sum_c D_c D_c^{\\top},

    where ``D_c`` is the sum over cluster ``c`` of the per-observation dfbeta
    residuals (score residual times the model-based covariance). With no
    clustering each observation is its own cluster and this is the ordinary
    robust (Lin-Wei / HC) variance.

    Parameters
    ----------
    cluster : array_like, optional
        A cluster label per observation (same length and order as the fitting
        data). ``None`` treats every observation as its own cluster. Counts
        ``n`` are frequency weights: without ``cluster`` a row with ``n = 3``
        is three independent observations (three clusters), with it the
        three share the row's label -- either way the result equals that of
        the data written out one row per observation.

    Returns
    -------
    numpy.ndarray
        The ``p x p`` robust covariance matrix. Its diagonal square-roots are
        the robust standard errors (see :meth:`robust_standard_errors`).
    """
    data = _require_cox(model)
    dfbeta = compute_residuals(model, "dfbeta")  # (n_obs, p)
    n_obs = dfbeta.shape[0]
    # The counts ``n`` are frequency weights: a row with count 3 is three
    # independent observations, each its own cluster, exactly as the fit and
    # the model-based covariance treat it. A row's dfbeta is its count times
    # one observation's, so the default (unclustered) sandwich sums
    # n * (dfbeta / n)^2 per row -- squaring the row total counted the row
    # as a single cluster of n perfectly correlated copies.
    n_w = np.asarray(data["n"], dtype=float)

    if cluster is None and getattr(model, "is_tvc", False):
        # Start-stop rows of one subject are correlated by construction, so
        # a TVC fit defaults to clustering by subject (#259). A subject's
        # count is again a frequency weight: that many independent subjects
        # with the same history.
        subject_ids = getattr(model, "tvc_subject_ids", None)
        if subject_ids is not None:
            labels, inv_idx = np.unique(subject_ids, return_inverse=True)
            unit = np.zeros((labels.size, dfbeta.shape[1]))
            np.add.at(unit, inv_idx, dfbeta / n_w[:, None])
            weight = np.zeros(labels.size)
            np.maximum.at(weight, inv_idx, n_w)
            return (weight[:, None] * unit).T @ unit

    if cluster is None:
        unit = dfbeta / n_w[:, None]
        return (n_w[:, None] * unit).T @ unit
    else:
        cluster = np.asarray(cluster)
        if cluster.shape[0] != n_obs:
            raise ValueError(
                "`cluster` must have one label per observation "
                f"({n_obs}); got {cluster.shape[0]}."
            )
        row_order = getattr(model, "tvc_row_order", None)
        subject_ids = getattr(model, "tvc_subject_ids", None)
        if row_order is not None and (
            subject_ids is None or not np.array_equal(cluster, subject_ids)
        ):
            # The residuals are in the internal (subject, entry)-sorted
            # order; user labels arrive in the caller's original row order
            # and must be permuted the same way (#259).
            cluster = cluster[row_order]
        labels, inv_idx = np.unique(cluster, return_inverse=True)
        grouped = np.zeros((labels.size, dfbeta.shape[1]))
        np.add.at(grouped, inv_idx, dfbeta)

    return grouped.T @ grouped


def robust_summary(
    model: "SemiParametricRegressionModel",
    cluster: "npt.ArrayLike | None" = None,
) -> dict:
    """
    Cluster-robust standard errors, z-scores and p-values for the
    coefficients (see :func:`robust_covariance`).

    Returns
    -------
    dict
        ``{"covariance", "se", "z", "p_value"}`` plus ``"covariate"`` names
        when the model was fit from a DataFrame.
    """
    from scipy.stats import norm

    beta = np.asarray(model.beta, dtype=float)
    cov = robust_covariance(model, cluster)
    se = np.sqrt(np.diag(cov))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = beta / se
        p = 2.0 * (1.0 - norm.cdf(np.abs(z)))
    out = {"covariance": cov, "se": se, "z": z, "p_value": p}
    names = getattr(model, "feature_names", None)
    if names is not None:
        out["covariate"] = list(names)
    return out


# ---------------------------------------------------------------------------
# Score tests at the fit
# ---------------------------------------------------------------------------
# The proportional-hazards test, as R's ``survival::cox.zph`` computes it
# since version 3.0, and the score test of an added covariate (the
# functional-form check of ``diagnose()``). Both need, at each distinct
# event time, the risk-set variance of the covariates summed over that
# time's deaths: the time's contribution to the information matrix.


def _risk_set_variances(
    data: dict, beta: np.ndarray, tie_method: str = "breslow"
) -> tuple:
    """Per distinct event time ``k``: ``W[k]``, the ``p x p`` contribution
    of the time's deaths to the observed information -- the n-weighted
    risk-set variance of ``Z`` times the deaths (Breslow), or its sum over
    the ``m`` Efron pseudo-risk-sets of a tie. ``W.sum(axis=0)`` is the
    information of the partial likelihood at ``beta``.

    Returns ``(event_times, W)``, with the risk sets as
    :func:`_risk_set_means` builds them (delayed entry, count weights).
    """
    x, c, n, tl = data["x"], data["c"], data["n"], data["tl"]
    Z = np.asarray(data["Z"], dtype=float)
    p = Z.shape[1]
    w = n * np.exp(Z @ beta)
    efron = str(tie_method).lower() == "efron"
    event_times = np.unique(x[c == 0])
    K = event_times.size
    ZZ = (Z[:, :, None] * Z[:, None, :]).reshape(Z.shape[0], p * p)
    exit_k = np.searchsorted(event_times, x, side="right")
    entry_k = np.searchsorted(event_times, tl, side="right")

    def at_risk(values: np.ndarray) -> np.ndarray:
        # As in _risk_set_means: those still in (x >= tau) less those
        # not yet entered (tl >= tau), by suffix sums over the buckets.
        def suffix(bucket: np.ndarray) -> np.ndarray:
            out = _bucket_sums(bucket, values, K + 1)
            return out[::-1].cumsum(axis=0)[::-1][1:]

        return suffix(exit_k) - suffix(entry_k)

    S0 = at_risk(w)
    S1 = at_risk(w[:, None] * Z)
    S2 = at_risk(w[:, None] * ZZ).reshape(K, p, p)
    is_event = c == 0
    event_k = np.searchsorted(event_times, x[is_event])
    d = _bucket_sums(event_k, n[is_event], K)
    mean = S1 / S0[:, None]
    W = d[:, None, None] * (
        S2 / S0[:, None, None] - mean[:, :, None] * mean[:, None, :]
    )
    if efron:
        S0D = _bucket_sums(event_k, w[is_event], K)
        S1D = _bucket_sums(event_k, (w[:, None] * Z)[is_event], K)
        S2D = _bucket_sums(event_k, (w[:, None] * ZZ)[is_event], K)
        S2D = S2D.reshape(K, p, p)
        for k in np.flatnonzero(np.round(d) > 1):
            m = int(round(d[k]))
            fracs = np.arange(m) / m
            s0 = S0[k] - fracs * S0D[k]
            s1 = S1[k][None, :] - fracs[:, None] * S1D[k][None, :]
            s2 = S2[k][None, :, :] - fracs[:, None, None] * S2D[k][None]
            e = s1 / s0[:, None]
            var = s2 / s0[:, None, None] - e[:, :, None] * e[:, None, :]
            # Each pseudo-risk-set carries d/m of the deaths (one with
            # unit counts), as in the Efron information.
            W[k] = var.sum(axis=0) * (d[k] / m)
    return event_times, W


def _zph_times(data: dict, transform: str) -> np.ndarray:
    """The transformed time of every row, as R's ``cox.zph`` (survival
    3.0 and later) computes it: ``"km"`` is ``1 - S(t-)``, the
    left-continuous Kaplan-Meier estimate from all the data (delayed entry
    respected); ``"rank"`` the average rank of the row's time among *all*
    the rows, event and censored, each counted ``n`` times; ``"identity"``
    and ``"log"`` the time and its log. Only the event rows' values are
    used, and only up to an additive constant.

    Until v0.24 ``check_ph`` used a right-continuous ``"km"`` (``1 -
    S(t)``) and ranked the event times only (lifelines' conventions)."""
    from surpyval.utils import xcnt_to_xrd

    x = np.asarray(data["x"], dtype=float)
    n = np.asarray(data["n"], dtype=float)
    c = np.asarray(data["c"], dtype=int)
    if transform == "identity":
        return x
    if transform == "log":
        events = x[c == 0]
        if np.any(events <= 0):
            raise ValueError(
                "The 'log' transform needs positive event times; there is "
                "an event at time {}. Use transform='km' (the default), "
                "'rank' or 'identity'.".format(float(np.min(events)))
            )
        return np.log(np.where(x > 0, x, np.nan))
    if transform == "rank":
        uniq, inv = np.unique(x, return_inverse=True)
        w_at = np.bincount(inv.ravel(), weights=n, minlength=uniq.size)
        before = np.concatenate([[0.0], np.cumsum(w_at)[:-1]])
        return (before + (w_at + 1.0) / 2.0)[inv.ravel()]
    if transform == "km":
        tl = np.asarray(data["tl"], dtype=float)
        t_mat = np.column_stack([tl, np.full(x.shape[0], np.inf)])
        xk, rk, dk = xcnt_to_xrd(x, c, n, t_mat)
        with np.errstate(all="ignore"):
            surv = np.cumprod(1.0 - dk / rk)
        before = np.searchsorted(xk, x, side="left")  # the times < x
        return 1.0 - np.concatenate([[1.0], surv])[before]
    raise option_error("transform", transform, _TRANSFORMS)


def _score_statistic(u: np.ndarray, info: np.ndarray) -> float:
    """``u' info^{-1} u``; nan where ``info`` is singular."""
    try:
        return float(u @ np.linalg.solve(info, u))
    except np.linalg.LinAlgError:
        return float("nan")


def check_ph(
    model: "SemiParametricRegressionModel", transform: str = "km"
) -> dict:
    """
    Test the proportional-hazards assumption (Grambsch-Therneau), as R's
    ``survival::cox.zph`` does since version 3.0.

    For each covariate, the score test of adding ``Z_j * g(t)`` to the
    model, at the fitted coefficients and a zero coefficient for the added
    term: a non-zero coefficient of ``Z_j * g(t)`` is an effect that
    changes with time, a violation of proportional hazards. The ``GLOBAL``
    test is the score test of adding all of them. Each is the efficient
    score test, with the full information matrix of the enlarged model at
    every event time.

    Parameters
    ----------
    transform : {"km", "rank", "identity", "log"}
        The function ``g`` of time, as R defines it: ``"km"`` (the
        default, the scale-free choice) is ``1 - S(t-)``, one minus the
        left-continuous Kaplan-Meier estimate from all the data; ``"rank"``
        the rank of each row's time among every row, censored ones too;
        ``"identity"`` the time and ``"log"`` its log (positive event times
        only). See :func:`_zph_times`.

    Returns
    -------
    dict
        ``{"global": {"statistic", "df", "p_value"},
           "per_covariate": [{"statistic", "df", "p_value"}, ...],
           "transform": transform}``. Per-covariate tests have 1 degree of
        freedom; the global test is the joint ``p``-d.f. test. A small
        ``p_value`` is evidence *against* proportional hazards. When the
        model was fit from a DataFrame, per-covariate entries carry the
        feature name.

    Notes
    -----
    The statistics are R's ``cox.zph`` (survival 3.0 and later) to the
    precision of the fit, under Efron and Breslow ties, delayed entry and
    time-varying covariates (``surpyval/tests/reference``). Under Efron
    ties the variances are those of the Efron pseudo-risk sets; under any
    other tie method (``"breslow"``, ``"exact"``,
    ``"kalbfleisch-prentice"``) they are Breslow's, as R uses for every
    method but Efron's. ``g`` is centred on its mean over the events, as R
    centres it, which matters only where the fit's score is not exactly 0
    (the exact methods). Counts ``n`` are frequency weights: a row with
    ``n = 3`` is three tied observations, as in the data written out row by
    row. Where the transformed event times have no spread (every event at
    one time) the test is undefined and every statistic is ``nan``.

    Until v0.24 this was the approximation of R before version 3.0 and of
    lifelines, which puts the average covariate variance in place of the
    variance at each event time; the two agree closely in large samples
    and can differ in small ones.

    Examples
    --------
    >>> from surpyval import CoxPH
    >>> from surpyval.datasets import load_rossi_static
    >>> from surpyval.univariate.regression.proportional_hazards import (
    ...     diagnostics,
    ... )
    >>> df = load_rossi_static()
    >>> x, c = df["week"].values, 1 - df["arrest"].values
    >>> model = CoxPH.fit(x, df[["fin", "age"]].values, c=c)
    >>> res = diagnostics.check_ph(model, "km")
    >>> round(res["global"]["statistic"], 3)
    6.095
    """
    data = _require_cox(model)
    check_option("transform", transform, _TRANSFORMS)
    beta = np.asarray(model.beta, dtype=float)
    p = beta.size
    x, c, n = data["x"], data["c"], data["n"]
    tie_method = getattr(model, "tie_method", "breslow")
    event_rows = np.flatnonzero(c == 0)
    w_events = n[event_rows]
    g = _zph_times(data, transform)[event_rows]
    g = g - np.average(g, weights=w_events)
    names = getattr(model, "feature_names", None)
    if not np.any(np.abs(g) > 1e-12 * max(1.0, float(np.abs(g).max()))):
        # No spread in the transformed times (e.g. a single event time):
        # the test is undefined.
        undefined = [{"statistic": np.nan, "df": 1, "p_value": np.nan}]
        per_cov = [dict(undefined[0]) for _ in range(p)]
        for j, entry in enumerate(per_cov):
            if names is not None and j < len(names):
                entry["covariate"] = names[j]
        return {
            "global": {"statistic": np.nan, "df": p, "p_value": np.nan},
            "per_covariate": per_cov,
            "transform": transform,
        }
    sch = compute_residuals(model, "schoenfeld")
    u2 = (w_events[:, None] * g[:, None] * sch).sum(axis=0)

    event_times, W = _risk_set_variances(data, beta, tie_method)
    # g is a function of time: the same at every death of a time.
    g_k = np.zeros(event_times.size)
    g_k[np.searchsorted(event_times, x[event_rows])] = g
    i11 = W.sum(axis=0)
    i12 = (g_k[:, None, None] * W).sum(axis=0)
    i22 = (g_k[:, None, None] ** 2 * W).sum(axis=0)
    info = np.block([[i11, i12], [i12.T, i22]])
    zeros = np.zeros(p)

    per_cov = []
    for j in range(p):
        keep = list(range(p)) + [p + j]
        stat = _score_statistic(
            np.concatenate([zeros, u2[j : j + 1]]), info[np.ix_(keep, keep)]
        )
        entry = {
            "statistic": stat,
            "df": 1,
            "p_value": float(chi2.sf(stat, df=1)),
        }
        if names is not None and j < len(names):
            entry["covariate"] = names[j]
        per_cov.append(entry)
    global_stat = _score_statistic(np.concatenate([zeros, u2]), info)
    return {
        "global": {
            "statistic": global_stat,
            "df": p,
            "p_value": float(chi2.sf(global_stat, df=p)),
        },
        "per_covariate": per_cov,
        "transform": transform,
    }


#: The name ``diagnose()``'s Cox family first used for :func:`check_ph`;
#: the same function.
zph_score_test = check_ph


def added_covariate_score_test(
    model: "SemiParametricRegressionModel", extra: "npt.ArrayLike"
) -> dict:
    """
    The score test of adding the columns ``extra`` to a fitted Cox model:
    the score of their coefficients at 0, with the model's own at the
    fit, against the information of the enlarged model there -- a
    chi-squared test on as many degrees of freedom as ``extra`` has
    columns. It is what R's ``coxph(..., init = c(coef(fit), 0),
    iter.max = 0)$score`` reports, without a refit.

    ``extra`` has one row per row of the data the model was fitted to, in
    the order the model keeps them (``model._fit_data``, the input order
    less any dropped rows). ``diagnose()`` uses it with a covariate's
    square, a test of the linearity of its effect.

    Returns
    -------
    dict
        ``{"statistic", "df", "p_value"}``; the statistic is nan where the
        enlarged information is singular (an added column that is a
        combination of the model's own).

    Examples
    --------
    >>> from surpyval import CoxPH
    >>> from surpyval.datasets import load_rossi_static
    >>> from surpyval.univariate.regression.proportional_hazards import (
    ...     diagnostics,
    ... )
    >>> df = load_rossi_static()
    >>> x, c = df["week"].values, 1 - df["arrest"].values
    >>> model = CoxPH.fit(x, df[["fin", "age"]].values, c=c)
    >>> res = diagnostics.added_covariate_score_test(model, df["age"] ** 2)
    >>> round(res["statistic"], 3), res["df"]
    (3.363, 1)
    """
    data = _require_cox(model)
    Z = np.asarray(data["Z"], dtype=float)
    extra = np.asarray(extra, dtype=float)
    if extra.ndim == 1:
        extra = extra[:, None]
    if extra.ndim != 2 or extra.shape[0] != Z.shape[0]:
        raise ValueError(
            "'extra' must have one row per row the model was fitted to "
            "({}); got shape {}.".format(Z.shape[0], extra.shape)
        )
    # Centred and scaled, which the test does not depend on (a linear
    # change of a column is absorbed by the model's coefficients and the
    # column's own), so that the information is well conditioned.
    weights = np.asarray(data["n"], dtype=float)
    extra = extra - np.average(extra, axis=0, weights=weights)
    scale = np.sqrt(np.average(extra**2, axis=0, weights=weights))
    extra = extra / np.where(scale > 0, scale, 1.0)
    p, q = Z.shape[1], extra.shape[1]
    beta = np.concatenate([np.asarray(model.beta, dtype=float), np.zeros(q)])
    enlarged = dict(data, Z=np.column_stack([Z, extra]))
    tie_method = getattr(model, "tie_method", "breslow")
    x, c, n = data["x"], data["c"], data["n"]
    event_times, Zbar = _risk_set_means(enlarged, beta, tie_method)[:2]
    event_rows = np.flatnonzero(c == 0)
    k = np.searchsorted(event_times, x[event_rows])
    resid = extra[event_rows] - Zbar[k, p:]
    u = np.concatenate([np.zeros(p), (n[event_rows][:, None] * resid).sum(0)])
    info = _risk_set_variances(enlarged, beta, tie_method)[1].sum(axis=0)
    stat = _score_statistic(u, info)
    return {"statistic": stat, "df": q, "p_value": float(chi2.sf(stat, df=q))}
