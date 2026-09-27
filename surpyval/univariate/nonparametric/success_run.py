import numpy as np


def success_run(
    n: int,
    confidence: float | None = None,
    alpha: float | None = None,
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

    Parameters
    ----------

    n : int
        The number of independent successes in the run.
    confidence : float, optional
        The desired confidence level, a value between 0 and 1. If neither
        ``confidence`` nor ``alpha`` is given, 0.95 is used.
    alpha : float, optional
        The significance level, ``1 - confidence``. Pass one of
        ``confidence`` and ``alpha``, not both.

    Returns
    -------

    float
        The lower confidence bound on the success probability.

    Raises
    ------

    ValueError
        If both ``confidence`` and ``alpha`` are given, ``n`` is not a
        positive number, or the significance level is not in [0, 1].

    Examples
    --------

    >>> from surpyval import success_run
    >>> success_run(10)
    np.float64(0.7411344491069477)

    59 successes demonstrate 95% reliability with 95% confidence:

    >>> print(round(success_run(59, confidence=0.95), 4))
    0.9505
    """
    # Tested against None rather than for truthiness: `confidence=0` and
    # `alpha=0` are both falsy, so the truthiness form let a caller pass
    # both without the raise firing and then left `alpha` as None for
    # `confidence=0`.
    if confidence is not None and alpha is not None:
        raise ValueError("Only one of confidence or alpha can be specified")
    if confidence is not None:
        alpha = 1 - confidence
    elif alpha is None:
        alpha = 0.05
    # A run of no successes demonstrates nothing; n = 0 used to fail as a
    # ZeroDivisionError and a negative n returned a "probability" above 1.
    if not n > 0:
        raise ValueError(
            "'n' must be a positive number of successes; got {}".format(n)
        )
    if not 0 <= alpha <= 1:
        raise ValueError(
            "The confidence (and alpha) must be between 0 and 1; got "
            "alpha = {}".format(alpha)
        )

    return np.power(alpha, 1.0 / n)
