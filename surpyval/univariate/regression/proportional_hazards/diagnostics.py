"""Residuals and the proportional-hazards test for a fitted Cox model.

All quantities are computed from the per-observation training data retained
on the fitted model (``model._fit_data``) and the Breslow baseline
(``model.x`` / ``model.H0``). Risk sets respect delayed entry (``tl``) and
count weights (``n``), so the residuals are correct for left-truncated /
start-stop data too.

References
----------
- Grambsch & Therneau (1994), "Proportional hazards tests and diagnostics
  based on weighted residuals", Biometrika 81(3).
- Therneau & Grambsch (2000), *Modeling Survival Data*, ch. 4-6.
"""

from typing import TYPE_CHECKING

import numpy as np
from numpy.linalg import inv, pinv
from scipy.stats import chi2

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
    if getattr(model, "kind", None) != "Cox" or not hasattr(
        model, "_fit_data"
    ):
        raise NotImplementedError(
            "Residuals and the proportional-hazards test are implemented for "
            "Cox proportional-hazards models fit with CoxPH.fit / fit_from_df."
        )
    return model._fit_data


def _H0_at(
    model: "SemiParametricRegressionModel", t: np.ndarray
) -> np.ndarray:
    """Breslow cumulative baseline hazard ``H0(t)`` as a right-continuous
    step function: the sum of the baseline hazard increments at event times
    ``<= t`` (0 below the first event time)."""
    idx = np.searchsorted(model.x, t, side="right")
    H0_padded = np.concatenate([[0.0], model.H0])
    return H0_padded[idx]


def _risk_set_means(data: dict, beta: np.ndarray):
    """For every distinct event time, the covariate mean over the risk set,
    weighted by ``n * exp(beta'Z)`` and respecting delayed entry.

    Returns ``(event_times, Zbar, S0, d)`` where ``Zbar[k]`` is the risk-set
    covariate mean at ``event_times[k]``, ``S0[k]`` the summed risk weight and
    ``d[k]`` the ``n``-weighted number of events there.
    """
    x, c, n, Z, tl = (
        data["x"],
        data["c"],
        data["n"],
        data["Z"],
        data["tl"],
    )
    w = n * np.exp(Z @ beta)  # risk weight per observation

    event_times = np.unique(x[c == 0])
    p = Z.shape[1]
    Zbar = np.empty((event_times.size, p))
    S0 = np.empty(event_times.size)
    d = np.empty(event_times.size)
    for k, tau in enumerate(event_times):
        at_risk = (tl < tau) & (x >= tau)
        s0 = w[at_risk].sum()
        S0[k] = s0
        Zbar[k] = (w[at_risk, None] * Z[at_risk]).sum(axis=0) / s0
        is_event = (x == tau) & (c == 0)
        d[k] = n[is_event].sum()
    return event_times, Zbar, S0, d


def _information(model: "SemiParametricRegressionModel") -> np.ndarray:
    """The observed information (Hessian of the negative partial
    log-likelihood) at the fitted ``beta``."""
    # ``model.jac`` is the jac/hess closure returned by the fitter (the class
    # attribute is loosely annotated as an array); [1] is the Hessian.
    _, hess = model.jac(model.beta)  # type: ignore[operator]
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
        The requested residuals. Schoenfeld residuals are ordered by event
        time; the per-observation residuals follow the input order.

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
    kind = kind.lower()
    if kind not in _RESIDUAL_KINDS:
        raise ValueError(
            f"Unknown residual kind {kind!r}; expected one of "
            f"{_RESIDUAL_KINDS}."
        )
    data = _require_cox(model)
    beta = np.asarray(model.beta, dtype=float)
    x, c, n, Z, tl = (
        data["x"],
        data["c"],
        data["n"],
        data["Z"],
        data["tl"],
    )

    if kind in ("martingale", "deviance"):
        delta = (c == 0).astype(float)
        cum_haz = np.exp(Z @ beta) * (_H0_at(model, x) - _H0_at(model, tl))
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

    event_times, Zbar, S0, d = _risk_set_means(data, beta)

    if kind in ("schoenfeld", "scaled_schoenfeld"):
        # One residual per event observation: covariate minus the risk-set
        # mean at its event time.
        event_rows = np.flatnonzero(c == 0)
        zbar_idx = np.searchsorted(event_times, x[event_rows])
        sch = Z[event_rows] - Zbar[zbar_idx]
        if kind == "schoenfeld":
            return sch
        # Scaled Schoenfeld (Grambsch-Therneau): beta + d * V * s, where V is
        # the coefficient covariance and d the number of events. Centring on
        # beta makes the plot read directly as beta(t).
        n_events = int(d.sum())
        V = _safe_inv(_information(model))
        return beta + n_events * (sch @ V)

    # Score / dfbeta residuals. The score residual for observation i is
    #   delta_i (Z_i - Zbar(x_i)) - exp(beta'Z_i) * sum_k in-window
    #       (Z_i - Zbar_k) (d_k / S0_k)
    # summing over event times k in the observation's at-risk window.
    w = np.exp(Z @ beta)
    delta = (c == 0).astype(float)
    increments = d / S0  # baseline-hazard increment per event time
    score = np.zeros_like(Z, dtype=float)
    # First term (only for events).
    event_rows = np.flatnonzero(c == 0)
    zbar_idx = np.searchsorted(event_times, x[event_rows])
    score[event_rows] += Z[event_rows] - Zbar[zbar_idx]
    # Second term: subtract the expected score accrued while at risk.
    for i in range(Z.shape[0]):
        in_window = (event_times > tl[i]) & (event_times <= x[i])
        if not in_window.any():
            continue
        contrib = (Z[i] - Zbar[in_window]) * increments[in_window, None]
        score[i] -= w[i] * contrib.sum(axis=0)
    score = score * n[:, None]
    if kind == "score":
        return score
    return score @ _safe_inv(_information(model))  # dfbeta


def _safe_inv(m: np.ndarray) -> np.ndarray:
    try:
        out = inv(m)
        if not np.all(np.isfinite(out)):
            raise np.linalg.LinAlgError
        return out
    except np.linalg.LinAlgError:
        return pinv(m)


def robust_covariance(
    model: "SemiParametricRegressionModel", cluster=None
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
        data). ``None`` treats every observation as its own cluster.

    Returns
    -------
    numpy.ndarray
        The ``p x p`` robust covariance matrix. Its diagonal square-roots are
        the robust standard errors (see :meth:`robust_standard_errors`).
    """
    _require_cox(model)
    dfbeta = compute_residuals(model, "dfbeta")  # (n_obs, p)
    n_obs = dfbeta.shape[0]

    if cluster is None:
        grouped = dfbeta
    else:
        cluster = np.asarray(cluster)
        if cluster.shape[0] != n_obs:
            raise ValueError(
                "`cluster` must have one label per observation "
                f"({n_obs}); got {cluster.shape[0]}."
            )
        labels, inv_idx = np.unique(cluster, return_inverse=True)
        grouped = np.zeros((labels.size, dfbeta.shape[1]))
        np.add.at(grouped, inv_idx, dfbeta)

    return grouped.T @ grouped


def robust_summary(
    model: "SemiParametricRegressionModel", cluster=None
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


def _transform_times(t: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return t.astype(float)
    if transform == "log":
        return np.log(t)
    if transform == "rank":
        return np.argsort(np.argsort(t)).astype(float)
    if transform == "km":
        # Scale-free transform (Grambsch-Therneau default): the empirical
        # CDF of the event times, a monotone stand-in for 1 - KM(t).
        ranks = np.searchsorted(np.sort(t), t, side="right")
        return ranks.astype(float) / t.size
    raise ValueError(
        f"Unknown transform {transform!r}; expected one of {_TRANSFORMS}."
    )


def check_ph(
    model: "SemiParametricRegressionModel", transform: str = "km"
) -> dict:
    """
    Test the proportional-hazards assumption (Grambsch-Therneau).

    Regresses the scaled Schoenfeld residuals on a transform of the event
    times; a non-zero slope is evidence of a time-varying coefficient, i.e.
    a violation of proportional hazards.

    Parameters
    ----------
    transform : {"km", "rank", "identity", "log"}
        The function of time to test against. ``"km"`` (the default) is the
        scale-free choice; ``"rank"`` matches Kaplan-Meier-rank usage.

    Returns
    -------
    dict
        ``{"global": {"statistic", "df", "p_value"},
           "per_covariate": [{"statistic", "df", "p_value"}, ...],
           "transform": transform}``. Per-covariate tests are 1-d.f. marginal
        screens; the global test is the joint ``p``-d.f. test. A small
        ``p_value`` is evidence *against* proportional hazards. When the model
        was fit from a DataFrame, per-covariate entries carry the feature name.

    Notes
    -----
    Per-covariate tests use the information diagonal (a marginal working
    covariance) and so do not adjust for non-proportionality in the *other*
    covariates; the global test uses the full information and does. For a
    single covariate the two coincide.
    """
    _require_cox(model)
    if transform not in _TRANSFORMS:
        raise ValueError(
            f"Unknown transform {transform!r}; expected one of {_TRANSFORMS}."
        )
    beta = np.asarray(model.beta, dtype=float)
    data = model._fit_data
    x, c, n = data["x"], data["c"], data["n"]

    sch = compute_residuals(model, "schoenfeld")  # (n_events, p)
    event_rows = np.flatnonzero(c == 0)
    t_events = x[event_rows]
    w_events = n[event_rows]  # count weight per event record

    g = _transform_times(t_events, transform)
    g_bar = np.average(g, weights=w_events)
    gc = g - g_bar

    # u = sum_i n_i (g_i - gbar) s_i ; Sgc2 = sum_i n_i (g_i - gbar)^2
    u = (w_events[:, None] * gc[:, None] * sch).sum(axis=0)
    Sgc2 = float((w_events * gc**2).sum())
    n_events = float(w_events.sum())

    info = _information(model)  # observed information at beta
    p = beta.size

    if Sgc2 <= 0:
        # No spread in the transformed times (e.g. a single event time):
        # the test is undefined.
        nan_entry = {"statistic": np.nan, "df": 1, "p_value": np.nan}
        return {
            "global": {"statistic": np.nan, "df": p, "p_value": np.nan},
            "per_covariate": [dict(nan_entry) for _ in range(p)],
            "transform": transform,
        }

    # Global joint test: T = (d / Sgc2) u' I^{-1} u ~ chi2_p.
    V = _safe_inv(info)
    global_stat = float((n_events / Sgc2) * (u @ V @ u))
    global_p = float(chi2.sf(global_stat, df=p))

    # Per-covariate marginal test: T_j = d u_j^2 / (Sgc2 * I_jj) ~ chi2_1.
    info_diag = np.diag(info)
    names = getattr(model, "feature_names", None)
    per_cov = []
    for j in range(p):
        stat_j = float(n_events * u[j] ** 2 / (Sgc2 * info_diag[j]))
        entry = {
            "statistic": stat_j,
            "df": 1,
            "p_value": float(chi2.sf(stat_j, df=1)),
        }
        if names is not None and j < len(names):
            entry["covariate"] = names[j]
        per_cov.append(entry)

    return {
        "global": {
            "statistic": global_stat,
            "df": p,
            "p_value": global_p,
        },
        "per_covariate": per_cov,
        "transform": transform,
    }
