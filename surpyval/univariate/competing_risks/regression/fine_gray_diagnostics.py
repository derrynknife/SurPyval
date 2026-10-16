"""The proportional subdistribution hazards test and the residuals of a
fitted Fine-Gray model, for ``diagnose()`` (#874).

The Fine-Gray fit maximises a weighted partial likelihood: a unit that
failed from a competing cause stays in the risk set after its failure,
with the weight ``G(t-)/G(x_i-)`` (``fine_gray._risk_sets``). Holding the
weights as fitted, that is the partial likelihood of a weighted Cox model
of counting-process data (``survival::finegray`` builds them for R's
``coxph``), so the Cox model's tests carry over: the Grambsch-Therneau
score test of adding ``Z * g(t)`` (:func:`fine_gray_zph_test`, R's
``cox.zph`` on those data), the score test of an added covariate
(:func:`fine_gray_added_covariate_test`) and the subdistribution
martingale residuals (:func:`fine_gray_martingale_residuals`). Each
takes the fitted model, or the :class:`FineGrayData` of
:func:`fine_gray_data`, which the diagnostics compute once per model.

References
----------
- Fine & Gray (1999), "A proportional hazards model for the
  subdistribution of a competing risk", JASA 94(446).
- Grambsch & Therneau (1994), "Proportional hazards tests and diagnostics
  based on weighted residuals", Biometrika 81(3).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from scipy.stats import chi2

from surpyval.utils.ipcw import censoring_survival, step_left_limit
from surpyval.utils.validation import check_option, option_error

from .fine_gray import _risk_set_sums, _risk_set_weights, _risk_sets

#: The time scales of the proportional subdistribution hazards test.
TRANSFORMS = ("cif", "rank")


def _restored() -> ValueError:
    return ValueError(
        "The diagnostics of a Fine-Gray model need the data it was fitted "
        "to, which a Fine-Gray model restored with from_dict / from_json "
        "does not carry. Call them on the fitted model, or refit."
    )


class FineGrayData(NamedTuple):
    """What the tests of a fitted Fine-Gray model share: its rows and
    risk sets, as :func:`fine_gray_data` rebuilds them."""

    #: The rows as fitted: times, counts, censoring flags and the
    #: covariates centred at their ``n``-weighted means.
    x: np.ndarray
    n: np.ndarray
    c: np.ndarray
    Z: np.ndarray
    #: The covariates as given (not centred).
    Z_raw: np.ndarray
    beta: np.ndarray
    #: The subdistribution risk sets (``fine_gray._risk_sets``).
    sets: Any
    #: The rows that are events of the cause, and the index in
    #: ``sets.times`` of each one's time.
    events: np.ndarray
    k: np.ndarray


def fine_gray_data(model: Any) -> FineGrayData:
    """
    The rows and the subdistribution risk sets of a fitted
    ``FineGrayModel``, rebuilt as its fit built them; every test here
    takes it in place of the model.

    Raises a ``ValueError`` for a model restored with ``from_dict``, which
    has no data, and for one with an aliased coefficient.
    """
    if isinstance(model, FineGrayData):
        return model
    data = getattr(model, "_fit_data", None)
    if data is None:
        raise _restored()
    beta = np.asarray(model.beta, dtype=float)
    aliased = np.flatnonzero(np.isnan(beta))
    if aliased.size:
        raise ValueError(
            "The diagnostics are not available for a Fine-Gray model with "
            "aliased coefficients (column(s) {} of Z, whose coefficients "
            "are nan, #476). Refit without those columns.".format(
                aliased.tolist()
            )
        )
    x = np.asarray(data["x"], dtype=float)
    n = np.asarray(data["n"], dtype=float)
    c = np.asarray(data["c"])
    Z_raw = np.asarray(data["Z"], dtype=float).reshape(x.size, -1)
    is_event = np.asarray(data["is_event"], dtype=bool)
    is_competing = np.asarray(data["is_competing"], dtype=bool)
    g_times, g_vals = censoring_survival(x, c == 1, n)
    G_x = step_left_limit(g_times, g_vals, x, before=1.0)
    sets = _risk_sets(x, n, is_event, is_competing, G_x, g_times, g_vals)
    Z = Z_raw - (n @ Z_raw) / n.sum()
    events = np.flatnonzero(is_event)
    k = np.searchsorted(sets.times, x[events])
    return FineGrayData(x, n, c, Z, Z_raw, beta, sets, events, k)


def _moments(fd: FineGrayData, A: np.ndarray, b: np.ndarray) -> tuple:
    """The weighted mean and covariance of the columns of ``A`` over the
    subdistribution risk set at each event time, with weights ``n
    W exp(A b)`` (``W`` the censoring weights): arrays of shape ``(T,
    q)`` and ``(T, q, q)``."""

    sets = fd.sets
    eta = A @ b
    r = (fd.n * np.exp(eta - eta.max()))[sets.order]
    As = A[sets.order]
    N, q = As.shape
    S0 = np.asarray(_risk_set_sums(r, sets))
    S1 = np.asarray(_risk_set_sums(r * As.T, sets))
    outer = (As[:, :, None] * As[:, None, :]).reshape(N, q * q)
    S2 = np.asarray(_risk_set_sums(r * outer.T, sets))
    mean = (S1 / S0).T
    second = (S2 / S0).T.reshape(-1, q, q)
    return mean, second - mean[:, :, None] * mean[:, None, :]


def _times(fd: FineGrayData, transform: str) -> np.ndarray:
    """The value of ``g`` at each event time of the cause: ``"cif"`` the
    cumulative incidence just before it, ``1 - prod_{t_j < t} (1 - d_j /
    R_j)``, the product-limit over the subdistribution risk sets (with
    the censoring weights), which is the Aalen-Johansen estimate;
    ``"rank"`` the time's average rank among all the rows, each counted
    ``n`` times, as for the Cox test."""
    sets = fd.sets
    if transform == "cif":
        at_risk = np.asarray(_risk_set_sums(fd.n[sets.order], sets))
        surv = np.cumprod(1.0 - sets.d / at_risk)
        return 1.0 - np.concatenate([[1.0], surv])[: sets.times.size]
    if transform == "rank":
        uniq, inv = np.unique(fd.x, return_inverse=True)
        inv = inv.ravel()
        w_at = np.bincount(inv, weights=fd.n, minlength=uniq.size)
        before = np.concatenate([[0.0], np.cumsum(w_at)[:-1]])
        rank = (before + (w_at + 1.0) / 2.0)[inv]
        out = np.zeros(sets.times.size)
        out[fd.k] = rank[fd.events]
        return out
    raise option_error("transform", transform, TRANSFORMS)


def _score_statistic(u: np.ndarray, info: np.ndarray) -> float:
    """``u' info^{-1} u``; nan where ``info`` is singular."""
    try:
        return float(u @ np.linalg.solve(info, u))
    except np.linalg.LinAlgError:
        return float("nan")


def _zph(fd: FineGrayData, transform: str) -> dict:
    mean, var = _moments(fd, fd.Z, fd.beta)
    w = fd.n[fd.events]
    sch = fd.Z[fd.events] - mean[fd.k]
    g_raw = _times(fd, transform)
    g_k = g_raw - np.average(g_raw[fd.k], weights=w)
    u2 = (w[:, None] * g_k[fd.k][:, None] * sch).sum(axis=0)
    W = fd.sets.d[:, None, None] * var
    i11 = W.sum(axis=0)
    i12 = (g_k[:, None, None] * W).sum(axis=0)
    i22 = (g_k[:, None, None] ** 2 * W).sum(axis=0)
    info = np.block([[i11, i12], [i12.T, i22]])
    p = fd.beta.size
    zeros = np.zeros(p)
    per_cov = []
    for j in range(p):
        keep = list(range(p)) + [p + j]
        stat = _score_statistic(
            np.concatenate([zeros, u2[j : j + 1]]), info[np.ix_(keep, keep)]
        )
        per_cov.append(
            {"statistic": stat, "df": 1, "p_value": float(chi2.sf(stat, 1))}
        )
    stat = _score_statistic(np.concatenate([zeros, u2]), info)
    try:
        scaled = fd.beta + w.sum() * sch @ np.linalg.inv(i11)
    except np.linalg.LinAlgError:
        scaled = np.full(sch.shape, np.nan)
    return {
        "global": {
            "statistic": stat,
            "df": p,
            "p_value": float(chi2.sf(stat, p)),
        },
        "per_covariate": per_cov,
        "transform": transform,
        "information": i11,
        "g": g_raw[fd.k],
        "time": fd.x[fd.events],
        "scaled_schoenfeld": scaled,
    }


def fine_gray_zph_test(model: Any, transform: str = "cif") -> dict:
    """
    The proportional subdistribution hazards test of a fitted Fine-Gray
    model (``FineGray.fit``): for each covariate, the score test of adding
    ``Z_j * g(t)`` to the weighted partial likelihood, at the fitted
    coefficients and a zero coefficient for the added term, and a
    ``GLOBAL`` score test of adding all of them (#874).

    The censoring weights ``G(t-)/G(x_i-)`` of the fit are held fixed, so
    this is the test of R's ``survival::cox.zph`` (version 3.0 and later)
    on the weighted counting-process data on which ``coxph`` reproduces
    the Fine-Gray fit (as ``survival::finegray`` builds them), with
    Breslow's ties, as the Fine-Gray fit has: a Grambsch-Therneau test of
    the weighted Schoenfeld residuals. The information is the model-based
    one, as ``FineGray``'s standard errors are.

    Parameters
    ----------
    model : FineGrayModel or FineGrayData
        The fitted model (``FineGray.fit``, or one cause's model of a
        ``CompetingRisksProportionalHazards`` with ``model="Fine-Gray"``,
        ``models[cause]``), or its :func:`fine_gray_data`.
    transform : {"cif", "rank"}
        The function ``g`` of time: ``"cif"`` the cumulative incidence of
        the cause just before ``t`` (the Aalen-Johansen estimate, from the
        fit's own risk sets), the subdistribution's analogue of
        ``cox.zph``'s Kaplan-Meier scale; ``"rank"`` the average rank of
        ``t`` among all the rows' times.

    Returns
    -------
    dict
        ``{"global": {"statistic", "df", "p_value"}, "per_covariate":
        [...], "transform"}``, as the Cox model's ``zph_score_test``,
        and the event rows' ``time``, ``g`` and ``scaled_schoenfeld``
        residuals (``beta + n_events * V^{-1} r``, ``r`` the weighted
        Schoenfeld residual and ``V`` the information ``information``,
        the inverse of the model's covariance).

    Examples
    --------
    Proportional cause-specific hazards: ``z`` doubles the hazard of cause
    ``a`` and leaves ``b``'s alone, so the subdistribution hazard ratio of
    ``a`` is not constant (it falls as the units with ``z = 1`` are used
    up), and the test sees it:

    >>> import numpy as np
    >>> from surpyval import FineGray
    >>> from surpyval.univariate.competing_risks.regression.fine_gray_diagnostics import (  # noqa: E501
    ...     fine_gray_zph_test,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> z = rng.binomial(1, 0.5, 300).astype(float)
    >>> t_a = rng.exponential(1 / (0.1 * np.exp(0.7 * z)))
    >>> t_b = rng.exponential(1 / 0.05, 300)
    >>> x = np.minimum(t_a, t_b)
    >>> e = np.where(t_a < t_b, "a", "b")
    >>> model = FineGray.fit(x, z[:, None], e, event="a")
    >>> res = fine_gray_zph_test(model, "cif")
    >>> glob = res["global"]
    >>> round(glob["statistic"], 3), round(glob["p_value"], 4)
    (6.371, 0.0116)
    """
    check_option("transform", transform, TRANSFORMS)
    return _zph(fine_gray_data(model), transform)


def _added(fd: FineGrayData, extra: np.ndarray) -> dict:
    extra = np.asarray(extra, dtype=float)
    if extra.ndim == 1:
        extra = extra[:, None]
    if extra.ndim != 2 or extra.shape[0] != fd.Z.shape[0]:
        raise ValueError(
            "'extra' must have one row per row the model was fitted to "
            "({}); got shape {}.".format(fd.Z.shape[0], extra.shape)
        )
    # Centred and scaled, which the test does not depend on, so that the
    # information is well conditioned.
    extra = extra - np.average(extra, axis=0, weights=fd.n)
    scale = np.sqrt(np.average(extra**2, axis=0, weights=fd.n))
    extra = extra / np.where(scale > 0, scale, 1.0)
    p, q = fd.Z.shape[1], extra.shape[1]
    A = np.column_stack([fd.Z, extra])
    b = np.concatenate([fd.beta, np.zeros(q)])
    mean, var = _moments(fd, A, b)
    w = fd.n[fd.events]
    resid = extra[fd.events] - mean[fd.k, p:]
    u = np.concatenate([np.zeros(p), (w[:, None] * resid).sum(axis=0)])
    info = (fd.sets.d[:, None, None] * var).sum(axis=0)
    stat = _score_statistic(u, info)
    return {"statistic": stat, "df": q, "p_value": float(chi2.sf(stat, q))}


def fine_gray_added_covariate_test(model: Any, extra: Any) -> dict:
    """
    The score test of adding the columns ``extra`` to a fitted Fine-Gray
    model: the score of their coefficients at 0, with the model's own at
    the fit, against the information of the enlarged weighted partial
    likelihood there (the censoring weights held fixed) -- a chi-squared
    test on as many degrees of freedom as ``extra`` has columns (#874).
    It is what R's ``coxph(..., init = c(coef(fit), 0), iter.max =
    0)$score`` reports on the weighted counting-process data of the fit.
    ``diagnose()`` uses it with a covariate's square, a test of the
    linearity of its effect on the log subdistribution hazard.

    ``model`` is the fitted model or its :func:`fine_gray_data`;
    ``extra`` has one row per row the model was fitted to, in the order
    given to the fit (less any rows dropped for a missing value).

    Returns
    -------
    dict
        ``{"statistic", "df", "p_value"}``; the statistic is nan where the
        enlarged information is singular.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import FineGray
    >>> from surpyval.univariate.competing_risks.regression.fine_gray_diagnostics import (  # noqa: E501
    ...     fine_gray_added_covariate_test,
    ... )
    >>> rng = np.random.default_rng(1)
    >>> z = rng.normal(size=300)
    >>> t_a = rng.exponential(1 / (0.1 * np.exp(0.5 * z)))
    >>> t_b = rng.exponential(1 / 0.05, 300)
    >>> e = np.where(t_a < t_b, "a", "b")
    >>> model = FineGray.fit(np.minimum(t_a, t_b), z[:, None], e, event="a")
    >>> res = fine_gray_added_covariate_test(model, z**2)
    >>> res["df"], bool(res["p_value"] > 0.05)
    (1, True)
    """
    return _added(fine_gray_data(model), extra)


def fine_gray_martingale_residuals(model: Any) -> np.ndarray:
    """
    Each row's subdistribution martingale residual, per unit: its event
    indicator for the cause less its weighted cumulative subdistribution
    hazard, ``exp(beta'Z_i) sum_t W_i(t) dLambda_0(t)``, ``W_i(t)`` the
    row's censoring weight in the risk set at ``t`` and ``Lambda_0``
    Breslow's baseline. Plotted against a covariate it shows the
    covariate's functional form, as the martingale residuals of a Cox
    model do. ``model`` is the fitted model or its :func:`fine_gray_data`;
    one residual per row, in the order given to the fit.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import FineGray
    >>> from surpyval.univariate.competing_risks.regression.fine_gray_diagnostics import (  # noqa: E501
    ...     fine_gray_martingale_residuals,
    ... )
    >>> model = FineGray.fit([1, 2, 3, 4, 5, 6], [[0], [1], [0], [1], [1], [0]],
    ...                      ["a", "b", "a", "a", "b", "a"], event="a")
    >>> fine_gray_martingale_residuals(model).shape
    (6,)
    """
    fd = fine_gray_data(model)
    sets = fd.sets
    eta = fd.Z @ fd.beta
    shift = eta.max()
    r = np.exp(eta - shift)
    S0 = np.asarray(_risk_set_sums((fd.n * r)[sets.order], sets))
    exposure = np.empty(fd.x.size)
    exposure[sets.order] = _risk_set_weights(sets.d / S0, sets)
    delta = np.zeros(fd.x.size)
    delta[fd.events] = 1.0
    return delta - r * exposure
