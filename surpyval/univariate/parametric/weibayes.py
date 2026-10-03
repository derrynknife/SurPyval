"""Weibayes: the zero- (or few-) failure bound on a Weibull scale (#493)."""

from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.utils import xcnt_handler
from surpyval.utils.validation import alpha_ci_error

from .distributions import Weibull
from .parametric import Parametric


def weibayes(
    x: npt.ArrayLike,
    c: "npt.ArrayLike | None" = None,
    n: "npt.ArrayLike | None" = None,
    beta: float = 1.0,
    alpha_ci: float = 0.05,
) -> Parametric:
    r"""
    Weibayes: the lower confidence bound on the scale of a Weibull whose
    shape is known, from data with few or no failures.

    With every unit suspended the likelihood has no maximum -- it keeps
    rising as the scale grows past every suspension -- so ``Weibull.fit``
    refuses such data, with the shape fixed or not. With the shape
    :math:`\beta` known, though, :math:`t^\beta` is exponentially
    distributed, and the standard zero-failure (Weibayes) analysis gives
    an exact lower bound on the scale :math:`\alpha` at confidence
    :math:`1 - \alpha_{ci}` from the total "transformed time on test"
    :math:`T = \sum_i n_i x_i^\beta` of every unit, failed or suspended,
    and the number of failures :math:`r`:

    .. math::
        \alpha_L = \left(\frac{2T}{\chi^2_{1-\alpha_{ci}}(2r + 2)}
                   \right)^{1/\beta}

    (Nelson, 1985; Abernethy, 2006, ch. 6). With no failures,
    :math:`\chi^2_{C}(2) = -2\ln(1 - C)`, so
    :math:`\alpha_L = (T / -\ln\alpha_{ci})^{1/\beta}`; at
    :math:`1 - \alpha_{ci} = 63.2\%` this is :math:`T^{1/\beta}`, the
    maximum-likelihood scale with one failure assumed -- Abernethy's
    "Weibayes line". With failures, the :math:`2r + 2` degrees of
    freedom make it the conservative bound of a time-terminated test.

    The bound on the scale is a bound on the whole curve: every quantile
    and the reliability at every time increase with the scale, so the
    returned Weibull's ``sf(t)`` is the lower bound on the reliability at
    ``t``, and its ``qf(p)`` the lower bound on the B-life at ``p``, at
    the same confidence. With ``beta=1`` it is the exponential
    zero-failure bound: the scale is the lower bound on the mean life
    (MTTF or MTBF), :math:`2T / \chi^2_{1-\alpha_{ci}}(2r + 2)`, and its
    inverse the upper bound on the failure rate. For a pass/fail test
    with no times, see :func:`surpyval.success_run`.

    Parameters
    ----------
    x : array like
        The time of each failure or suspension.
    c : array like, optional
        The censoring flags: 0 for a failure, 1 for a suspension (right
        censored). Defaults to all failures. Left or interval censoring
        is not supported.
    n : array like, optional
        The number of units with each ``x`` and ``c``. Defaults to 1.
    beta : float, optional
        The known Weibull shape. Defaults to 1 (the exponential).
    alpha_ci : float, optional
        The significance level: the bound holds with confidence
        ``1 - alpha_ci``. Defaults to 0.05.

    Returns
    -------
    Parametric
        The Weibull with scale :math:`\alpha_L` and shape ``beta``, built
        with ``Weibull.from_params``; ``model.params[0]`` is the bound.

    Raises
    ------
    ValueError
        If the data has left or interval censoring or truncation, if
        ``beta`` is not a positive number, or ``alpha_ci`` is not in
        (0, 1).

    Examples
    --------
    Ten units ran 500 hours each without a failure; with a known shape of
    2, the scale is at least 913.5 hours with 95% confidence:

    >>> from surpyval import weibayes
    >>> model = weibayes([500] * 10, c=[1] * 10, beta=2)
    >>> round(float(model.params[0]), 1)
    913.5

    and the reliability at 500 hours at least 0.741 (the same as
    ``success_run(10)``: ten units passing a test of that length).

    >>> round(float(model.sf(500)), 3)
    0.741

    Five units ran 1000 hours each with no failure: the MTBF is at least
    5457 hours with 60% confidence.

    >>> mtbf = weibayes([1000] * 5, c=[1] * 5, alpha_ci=0.4).params[0]
    >>> round(float(mtbf))
    5457

    References
    ----------
    Nelson, W. (1985). Weibull analysis of reliability data with few or
    no failures. *Journal of Quality Technology*, 17(3), 140-146.

    Abernethy, R. B. (2006). *The New Weibull Handbook*, 5th ed.,
    chapter 6 (Weibayes and Weibest).
    """
    from scipy.stats import chi2

    x_arr, c_arr, n_arr, t_arr = xcnt_handler(x, c, n)
    if x_arr.ndim != 1 or not np.isin(c_arr, (0, 1)).all():
        raise ValueError(
            "weibayes takes failures (c = 0) and suspensions (c = 1) "
            "only; left and interval censoring are not supported"
        )
    if np.isfinite(t_arr).any():
        raise ValueError("weibayes does not support truncated data")
    if (x_arr <= 0).any():
        raise ValueError("Every time 'x' must be positive for weibayes")
    beta_f: Any = beta
    if not (np.isscalar(beta) and np.isfinite(beta_f) and beta_f > 0):
        raise ValueError(f"'beta' must be a positive number; got {beta}")
    if not 0 < alpha_ci < 1:
        raise alpha_ci_error(alpha_ci)

    beta_f = float(beta_f)
    # Scaled by the largest time so that a large shape cannot overflow
    scale = float(x_arr.max())
    total = float(np.sum(n_arr * (x_arr / scale) ** beta_f))
    r = int(np.sum(n_arr[c_arr == 0]))
    quantile = chi2.ppf(1 - alpha_ci, 2 * r + 2)
    alpha_lower = scale * (2 * total / quantile) ** (1 / beta_f)
    return Weibull.from_params([alpha_lower, beta_f])
