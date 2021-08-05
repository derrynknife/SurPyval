import numpy as np


def success_run(n, confidence=0.95, alpha=None):
    r"""
    Function that can be used to estimte the confidence given n samples
    all survive a test.
    """
    if alpha is None:
        alpha = 1 - confidence
    return np.power(alpha, 1. / n)
