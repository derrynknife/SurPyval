"""The restricted mean time lost (RMTL) to a cause of failure: the area
under its cumulative incidence up to a horizon, with its variance, and the
comparison of two groups (:func:`rmtl_diff`)."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.utils.validation import check_alpha_ci


def aalen_johansen_rmtl(
    x: npt.ArrayLike,
    r: npt.ArrayLike,
    d: npt.ArrayLike,
    d_k: npt.ArrayLike,
    tau: float,
) -> tuple[float, float]:
    r"""The restricted mean time lost to one cause up to ``tau``, from the
    Aalen-Johansen estimate, and its delta-method variance.

    The time lost is the area under the cause's cumulative incidence,
    :math:`\int_0^\tau \hat F_k(t)\,dt`, exact for the step function. Its
    variance is the delta method's on the hazard increments at each
    distinct time, each a multinomial proportion of the :math:`r_j` at
    risk (the cause's, :math:`p_{kj} = d_{kj} / r_j`, and the other
    causes', :math:`p_{oj}`): with :math:`G_{kj}` and :math:`G_{oj}` the
    derivatives of the area in them,

    .. math::
        \widehat{Var} = \sum_{x_j \le \tau} \frac{G_{kj}^2 p_{kj}(1 -
            p_{kj}) + G_{oj}^2 p_{oj}(1 - p_{oj}) - 2 G_{kj} G_{oj}
            p_{kj} p_{oj}}{r_j},

    .. math::
        G_{oj} = -\frac{\hat S(x_{j-1})}{\hat S(x_j)}\left(A_j - \hat
            F_k(x_j) L_j\right), \quad G_{kj} = \hat S(x_{j-1}) L_j +
            G_{oj},

    where :math:`L_j = \tau - x_j`, :math:`A_j = \int_{x_j}^\tau \hat
    F_k(t)\,dt` and :math:`\hat S` is the all-cause Kaplan-Meier survival.
    With one cause it is the Greenwood-based variance of the restricted
    mean survival time (the time lost is :math:`\tau` less it). Where
    :math:`\hat S(x_j) = 0` the incidence is flat from there on, and the
    ratio is taken as 0.

    Parameters
    ----------
    x, r, d : array_like
        The distinct times, the numbers at risk and of events (all
        causes) at each.
    d_k : array_like
        The events of the cause at each time.
    tau : float
        The horizon.

    Returns
    -------
    tuple of float
        ``(rmtl, variance)``.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.univariate.competing_risks.rmtl import (
    ...     aalen_johansen_rmtl,
    ... )
    >>> x, r = np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])
    >>> d, d_a = np.array([1.0, 1.0, 1.0]), np.array([1.0, 0.0, 1.0])
    >>> rmtl, var = aalen_johansen_rmtl(x, r, d, d_a, 4.0)
    >>> round(rmtl, 4), round(var, 4)
    (1.3333, 0.5185)
    """
    x = np.asarray(x, dtype=float)
    r = np.asarray(r, dtype=float)
    d = np.asarray(d, dtype=float)
    d_k = np.asarray(d_k, dtype=float)
    keep = x <= tau
    x, r, d, d_k = x[keep], r[keep], d[keep], d_k[keep]
    if x.size == 0:
        return 0.0, 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(r > 0, d / r, 0.0)
        p_k = np.where(r > 0, d_k / r, 0.0)
    p_o = p - p_k
    S = np.cumprod(1.0 - p)
    S_prev = np.concatenate([[1.0], S[:-1]])
    F = np.cumsum(S_prev * p_k)
    # The area under the step incidence from each time to tau.
    upper = np.append(x[1:], tau)
    A = np.cumsum((F * (upper - x))[::-1])[::-1]
    L = tau - x
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(S > 0, S_prev / S, 0.0)
    G_o = -ratio * (A - F * L)
    G_k = S_prev * L + G_o
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = (
            G_k**2 * p_k * (1.0 - p_k)
            + G_o**2 * p_o * (1.0 - p_o)
            - 2.0 * G_k * G_o * p_k * p_o
        ) / r
    terms = np.where(r > 0, terms, 0.0)
    return float(A[0]), float(max(np.sum(terms), 0.0))


def rmtl_diff(
    model_a: Any,
    model_b: Any,
    event: Any,
    tau: float | None = None,
    alpha_ci: float = 0.05,
) -> dict:
    r"""
    Compare the restricted mean time lost (RMTL) to a cause between two
    groups.

    The RMTL to a cause up to ``tau`` is the expected time, within the
    first ``tau``, lost to that cause: the area under its cumulative
    incidence. The difference reads directly as how much more time
    group a loses to the cause than group b within the horizon, and needs
    no proportional-hazards assumption; it is the competing-risks
    counterpart of :func:`surpyval.rmst_diff`.

    Parameters
    ----------
    model_a, model_b : CompetingRisks or ParametricCompetingRisks
        Two fitted competing-risks models, one per group, each with the
        cause ``event``: non-parametric (``CompetingRisks.fit`` per group)
        or parametric fits, or one of each. Their RMTL and standard error
        are those of their ``rmtl``.
    event : hashable
        The cause.
    tau : scalar, optional
        Common horizon. Defaults to the smaller of the two groups' largest
        observed times, so both incidences are supported by data up to
        ``tau``. A non-parametric group refuses a ``tau`` past its largest
        observed time while some units are still event-free there (as
        :func:`surpyval.rmst_diff`); a parametric fit extrapolates.
    alpha_ci : scalar, optional
        Significance level for the intervals and the test (default 0.05).

    Returns
    -------
    dict
        ``{"difference", "se", "lower", "upper", "p_value", "ratio",
        "ratio_lower", "ratio_upper", "rmtl_a", "rmtl_b", "tau"}``.
        ``difference`` is ``RMTL_a - RMTL_b``, ``se`` the square root of
        the sum of the groups' variances (the groups are independent),
        ``lower`` and ``upper`` are ``difference +- z se``, and ``p_value``
        the two-sided z-test of no difference. ``ratio`` is ``RMTL_a /
        RMTL_b`` with its interval from the delta method on the log scale,
        :math:`\exp(\log \text{ratio} \pm z \sqrt{v_a / \mu_a^2 + v_b /
        \mu_b^2})` (NaN where a group's RMTL is 0).

    Raises
    ------
    ValueError
        If ``event`` is not a cause of both models, a model cannot give a
        standard error (see their ``rmtl``), or ``tau`` is past a
        non-parametric group's largest observed time while it still has
        units event-free there.

    Examples
    --------
    >>> import surpyval as sp
    >>> e_a = ["x", "y", "x", None, "x", "y", "x", None]
    >>> e_b = ["y", "x", None, "y", "x", "y", None, "x"]
    >>> a = sp.CompetingRisks.fit([1, 2, 3, 4, 5, 6, 7, 8], e_a)
    >>> b = sp.CompetingRisks.fit([2, 3, 4, 5, 6, 7, 8, 9], e_b)
    >>> res = sp.rmtl_diff(a, b, "x")
    >>> print(res["tau"], round(res["difference"], 4), round(res["se"], 4))
    8.0 1.2 1.0895
    >>> print(round(res["p_value"], 4))
    0.2707
    """
    from scipy.stats import norm

    check_alpha_ci(alpha_ci)
    if tau is None:
        tau = float(
            min(
                model_a._rmtl_default_tau("rmtl_diff"),
                model_b._rmtl_default_tau("rmtl_diff"),
            )
        )
    mu_a, var_a = model_a._rmtl_estimate(event, float(tau), "model_a")
    mu_b, var_b = model_b._rmtl_estimate(event, float(tau), "model_b")
    diff = mu_a - mu_b
    se = float(np.sqrt(var_a + var_b))
    z = norm.ppf(1.0 - alpha_ci / 2.0)
    p_value = float(2.0 * norm.sf(abs(diff) / se)) if se > 0 else float("nan")
    if mu_a > 0 and mu_b > 0:
        ratio = mu_a / mu_b
        se_log = np.sqrt(var_a / mu_a**2 + var_b / mu_b**2)
        ratio_lower = float(ratio * np.exp(-z * se_log))
        ratio_upper = float(ratio * np.exp(z * se_log))
    else:
        ratio = ratio_lower = ratio_upper = float("nan")
    return {
        "difference": float(diff),
        "se": se,
        "lower": float(diff - z * se),
        "upper": float(diff + z * se),
        "p_value": p_value,
        "ratio": float(ratio),
        "ratio_lower": ratio_lower,
        "ratio_upper": ratio_upper,
        "rmtl_a": float(mu_a),
        "rmtl_b": float(mu_b),
        "tau": float(tau),
    }


def rmtl_result(
    mu: float, var: float, tau: float, alpha_ci: float, scale: str
) -> dict:
    """The ``rmtl`` result: the estimate, its standard error and the
    two-sided interval, plain (``scale="plain"``, ``mu +- z se``, as the
    non-parametric ``rmst``) or on the logit of ``mu / tau``
    (``"logit"``, as the parametric ``rmst``, inside ``(0, tau)``)."""
    from scipy.special import expit
    from scipy.stats import norm

    se = float(np.sqrt(var))
    z = norm.ppf(1.0 - alpha_ci / 2.0)
    if scale == "plain" or not 0 < mu < tau:
        lower, upper = mu - z * se, mu + z * se
    else:
        psi = np.log(mu) - np.log(tau - mu)
        se_psi = se * (1.0 / mu + 1.0 / (tau - mu))
        lower = tau * expit(psi - z * se_psi)
        upper = tau * expit(psi + z * se_psi)
    return {
        "rmtl": float(mu),
        "se": se,
        "lower": float(lower),
        "upper": float(upper),
        "tau": float(tau),
    }
