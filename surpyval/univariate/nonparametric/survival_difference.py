"""The test of a difference between two survival curves at one time."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from surpyval.utils.validation import check_alpha_ci, check_option

if TYPE_CHECKING:
    from .nonparametric import NonParametric

#: The ``transform`` values of :func:`sf_diff`.
TRANSFORMS = ("log-log", "log", "linear", "arcsine-sqrt", "logit")


def _at(model: NonParametric, name: str, x: float) -> tuple[float, float]:
    """The survival estimate of ``model`` at ``x`` and the variance of its
    logarithm there (Greenwood's for the Kaplan-Meier, the estimator's own
    for the others): the estimate is a step function, so both are those at
    the last observed time at or before ``x``."""
    if getattr(model, "greenwood", None) is None:
        raise ValueError(
            "{} has no variance estimate, so the survival difference "
            "cannot be tested. This occurs for models created with "
            "'fit_from_ecdf' since the at risk and death counts are "
            "unknown; fit the model to the data instead.".format(name)
        )
    s_last = model._survival_past_data(x)
    if s_last is not None:
        raise ValueError(
            "x = {:g} is past {}'s largest observed time, {:g}, where its "
            "survival estimate is still {:.3g}: the curve beyond it is not "
            "estimated. Use an x at or below {:g}.".format(
                x, name, float(np.max(model.x)), s_last, float(np.max(model.x))
            )
        )
    xs = np.asarray(model.x, dtype=float)
    i = int(np.searchsorted(xs, x, side="right")) - 1
    if i < 0:
        return 1.0, 0.0
    sf = float(np.asarray(model.R, dtype=float)[i])
    var_log = float(np.asarray(model.greenwood, dtype=float)[i])
    return sf, var_log


def _contrast(
    transform: str, s: float, v: float
) -> tuple[float, float] | None:
    """The transformed survival ``g(s)`` and the delta-method variance of
    ``g(S)`` from ``v``, the variance of ``log S``; ``None`` where ``g`` is
    not defined at ``s``."""
    if transform == "linear":
        return s, v * s**2
    if s <= 0:
        return None
    if transform == "log":
        return np.log(s), v
    if s >= 1:
        return None
    if transform == "log-log":
        return np.log(-np.log(s)), v / np.log(s) ** 2
    if transform == "arcsine-sqrt":
        return np.arcsin(np.sqrt(s)), v * s / (4 * (1 - s))
    # logit
    return np.log(s / (1 - s)), v / (1 - s) ** 2


def sf_diff(
    model_a: NonParametric,
    model_b: NonParametric,
    x: float,
    transform: str = "log-log",
    alpha_ci: float = 0.05,
) -> dict:
    r"""
    Test whether two groups' survival differs at one time ``x``: is
    :math:`S_a(x) = S_b(x)`?

    It answers a question such as "is the chance of surviving 5,000 hours
    different between the two designs?" without assuming anything about
    the curves at other times, as the log-rank test (which compares the
    whole curves) or a hazard ratio do. Each group's survival at ``x``
    comes from its own non-parametric fit, with its variance (Greenwood's
    for the Kaplan-Meier); the groups are independent, so the variance of
    the difference is the sum of the two.

    The test is computed on a transformed scale, :math:`g(S)`, on which the
    estimate is closer to normal in small samples:

    .. math::
        X^2 = \frac{(g(\hat S_a) - g(\hat S_b))^2}
                   {\widehat{Var}\,g(\hat S_a) + \widehat{Var}\,g(\hat S_b)}

    which is chi-squared on one degree of freedom when the two survival
    probabilities are equal. Klein et al. (2007) compared the transforms
    by simulation and recommend the log-log (complementary log-log) or
    the arcsine-square-root transform over the untransformed difference,
    whose test rejects too often in small samples.

    Parameters
    ----------
    model_a, model_b : NonParametric
        The two groups' fitted non-parametric estimates (e.g.
        ``KaplanMeier.fit`` on each group), with a variance estimate.
    x : scalar
        The time at which to compare the survival.
    transform : str, optional
        The scale of the test: "log-log" (the default;
        :math:`\log(-\log S)`), "log", "linear" (the plain difference),
        "arcsine-sqrt" (:math:`\arcsin\sqrt{S}`) or "logit".
    alpha_ci : scalar, optional
        Significance level for the interval on the difference (default
        0.05).

    Returns
    -------
    dict
        ``{"statistic", "p_value", "difference", "se", "lower", "upper",
        "sf_a", "sf_b", "x", "transform"}``. ``statistic`` is :math:`X^2`
        and ``p_value`` its two-sided p-value; ``difference`` is
        :math:`\hat S_a(x) - \hat S_b(x)`, with its standard error ``se``
        and the normal interval ``lower``, ``upper`` = ``difference +- z
        se`` (on the survival scale whatever ``transform`` is).

    Raises
    ------
    ValueError
        If ``transform`` is not listed above, a model has no variance
        estimate, ``x`` is past a group's last observed time while its
        estimate is above zero there, the transform is not defined at a
        group's estimate (the log-log, arcsine and logit need it strictly
        between 0 and 1, the log above 0), or neither estimate has any
        variance at ``x`` (no failure by ``x`` in either group).

    Examples
    --------
    The AML maintenance data (``aml`` in R's ``survival``): is the chance
    of staying in remission for 30 weeks different with and without
    maintenance chemotherapy?

    >>> import surpyval as sp
    >>> x = [9, 13, 13, 18, 23, 28, 31, 34, 45, 48, 161,
    ...      5, 5, 8, 8, 12, 16, 23, 27, 30, 33, 43, 45]
    >>> c = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1,
    ...      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    >>> a = sp.KaplanMeier.fit(x[:11], c=c[:11])  # maintained
    >>> b = sp.KaplanMeier.fit(x[11:], c=c[11:])  # not maintained
    >>> res = sp.sf_diff(a, b, 30)
    >>> print(round(res["sf_a"], 4), round(res["sf_b"], 4))
    0.6136 0.2917
    >>> print(round(res["statistic"], 4), round(res["p_value"], 4))
    2.0972 0.1476

    Notes
    -----
    The estimate and its variance are both step functions of time, taken
    at the last observed time at or before ``x``; the variance is not
    interpolated between observed times.

    References
    ----------
    Klein, J. P., Logan, B., Harhoff, M. and Andersen, P. K. (2007),
    "Analyzing survival curves at a fixed point in time", Statistics in
    Medicine 26, 4505-4519.
    """
    from scipy.stats import chi2, norm

    check_option("transform", transform, TRANSFORMS)
    check_alpha_ci(alpha_ci)
    try:
        x = float(x)
    except (TypeError, ValueError):
        raise ValueError(
            f"'x' must be one time, a number; got {x!r}"
        ) from None
    if not np.isfinite(x):
        raise ValueError(f"'x' must be a finite time; got {x!r}")

    parts: list[Any] = []
    for name, model in (("model_a", model_a), ("model_b", model_b)):
        s, v = _at(model, name, x)
        g = _contrast(transform, s, v)
        if g is None:
            raise ValueError(
                "The {} transform is not defined at {}'s survival "
                "estimate at x = {:g}, {:g}. Compare at a time where both "
                "estimates are between 0 and 1, or use transform="
                "'linear'.".format(transform, name, x, s)
            )
        parts.append((s, v, g))
    (s_a, v_a, (g_a, var_a)), (s_b, v_b, (g_b, var_b)) = parts
    if not var_a + var_b > 0:
        raise ValueError(
            "Neither group has a failure by x = {:g}, so the survival "
            "estimates there have no variance and nothing to compare. "
            "Compare at a later time.".format(x)
        )

    statistic = float((g_a - g_b) ** 2 / (var_a + var_b))
    se = float(np.sqrt(v_a * s_a**2 + v_b * s_b**2))
    z = norm.ppf(1 - alpha_ci / 2)
    difference = s_a - s_b
    return {
        "statistic": statistic,
        "p_value": float(chi2.sf(statistic, 1)),
        "difference": float(difference),
        "se": se,
        "lower": float(difference - z * se),
        "upper": float(difference + z * se),
        "sf_a": float(s_a),
        "sf_b": float(s_b),
        "x": x,
        "transform": transform,
    }
