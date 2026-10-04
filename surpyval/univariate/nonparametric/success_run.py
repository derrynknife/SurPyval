import warnings

import numpy as np

from surpyval.utils.deprecation import REMOVED_IN


def success_run(
    n: int,
    confidence: float | None = None,
    alpha: float | None = None,
    *,
    alpha_ci: float | None = None,
) -> float:
    r"""
    Calculate the minimum success probability of a run of 'n' independent
    events for a given confidence level. Useful when you want to know, with a
    certain amount of confidence what the probability of success is to be
    higher than a certain value.

    After :math:`n` successes and no failures, the lower
    :math:`1 - \alpha` confidence bound on the per-trial success
    probability is the smallest :math:`R` for which :math:`n` successes in
    a row still have probability at least :math:`\alpha`:

    .. math::
        R_L = \alpha^{1/n}

    It is the complement of the exact (Clopper-Pearson) upper bound on the
    failure probability that ``Bernoulli.fit([0], n=[n]).param_cb("p",
    alpha_ci, bound="upper")`` gives.

    Parameters
    ----------

    n : int
        The number of independent successes in the run.
    confidence : float, optional
        Deprecated in v0.23 (it will be removed in v0.24): the confidence
        level, ``1 - alpha_ci``. Use ``alpha_ci``.
    alpha : float, optional
        Deprecated in v0.23 (it will be removed in v0.24): the old name of
        ``alpha_ci``.
    alpha_ci : float, optional
        Keyword only: the significance level, the total tail probability
        of the bound, as every bound in SurPyval takes it (default 0.05,
        a 95% bound).

    Returns
    -------

    float
        The lower confidence bound on the success probability.

    Raises
    ------

    ValueError
        If more than one of ``alpha_ci``, ``confidence`` and ``alpha`` is
        given, ``n`` is not a positive number, or the significance level
        is not in [0, 1].

    Examples
    --------

    >>> from surpyval import success_run
    >>> success_run(10)
    np.float64(0.7411344491069477)

    59 successes demonstrate 95% reliability with 95% confidence:

    >>> print(round(success_run(59, alpha_ci=0.05), 4))
    0.9505
    """
    # Tested against None rather than for truthiness: `confidence=0` and
    # `alpha=0` are both falsy.
    given = [
        name
        for name, value in (
            ("alpha_ci", alpha_ci),
            ("confidence", confidence),
            ("alpha", alpha),
        )
        if value is not None
    ]
    if len(given) > 1:
        raise ValueError(
            "Give only one of alpha_ci, confidence and alpha; got "
            "{}".format(", ".join(given))
        )
    # `confidence` and `alpha` were the odd ones out among the bounds,
    # which all take `alpha_ci` (#580). The positional slot stays
    # `confidence` for the release they are deprecated in, so an old
    # success_run(59, 0.95) still means 95% confidence.
    if confidence is not None:
        warnings.warn(
            "success_run: 'confidence' is deprecated and will be removed in "
            "v{}; use alpha_ci = 1 - confidence.".format(REMOVED_IN),
            DeprecationWarning,
            stacklevel=2,
        )
        alpha_ci = 1 - confidence
    elif alpha is not None:
        warnings.warn(
            "success_run: 'alpha' is deprecated and will be removed in "
            "v{}; use 'alpha_ci'.".format(REMOVED_IN),
            DeprecationWarning,
            stacklevel=2,
        )
        alpha_ci = alpha
    elif alpha_ci is None:
        alpha_ci = 0.05
    # A run of no successes demonstrates nothing; n = 0 used to fail as a
    # ZeroDivisionError and a negative n returned a "probability" above 1.
    if not n > 0:
        raise ValueError(
            "'n' must be a positive number of successes; got {}".format(n)
        )
    if not 0 <= alpha_ci <= 1:
        raise ValueError(
            "The significance level alpha_ci must be between 0 and 1; got "
            "alpha_ci = {}".format(alpha_ci)
        )

    return np.power(alpha_ci, 1.0 / n)
