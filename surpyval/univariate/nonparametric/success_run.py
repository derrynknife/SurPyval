import numpy as np


def success_run(n, confidence=None, alpha=None):
    """
    Calculate the minimum success probability of a run of 'n' independent
    events for a given confidence level. Useful when you want to know, with a
    certain amount of confidence what the probability of success is to be
    higher than a certain value.

    Parameters
    ----------

        n : int
            The number of independent events in the run.
        confidence : float, optional
            The desired confidence level, a value between 0 and 1. Default is
            0.95, which corresponds to a 95% confidence level.
        alpha : float, optional
            The significance level, a value between 0 and
            1. Only used if confidence is not specified. Default is None.

    Returns
    -------

        float:
            The minimum success probability of a run of 'n' independent trials
            for the given confidence level.

    Example
    -------

        >>> from surpyval import success_run
        >>> success_run(10)
        0.7411344491069477
    """
    if confidence and alpha:
        raise ValueError("Only one of confidence or alpha can be specified")
    if confidence is None and alpha is None:
        alpha = 0.05
    if confidence:
        alpha = 1 - confidence

    return np.power(alpha, 1.0 / n)
