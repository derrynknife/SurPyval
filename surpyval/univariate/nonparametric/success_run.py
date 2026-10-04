import numpy as np


def success_run(n: int, *, alpha_ci: float = 0.05) -> float:
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
    alpha_ci : float, optional
        Keyword only: the significance level, the total tail probability
        of the bound, as every bound in SurPyval takes it (default 0.05,
        a 95% bound). It replaces ``confidence`` (``1 - alpha_ci``) and
        ``alpha``, removed in v0.24.

    Returns
    -------

    float
        The lower confidence bound on the success probability.

    Raises
    ------

    ValueError
        If ``n`` is not a positive number, or the significance level is
        not in [0, 1].

    Examples
    --------

    >>> from surpyval import success_run
    >>> success_run(10)
    np.float64(0.7411344491069477)

    59 successes demonstrate 95% reliability with 95% confidence:

    >>> print(round(success_run(59, alpha_ci=0.05), 4))
    0.9505
    """
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
