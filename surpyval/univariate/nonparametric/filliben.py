import numpy as np

from surpyval.univariate.nonparametric import rank_adjust
from surpyval.utils import xcnt_handler


def filliben(x, c, n, t):
    """
    Method From:
    Filliben, J. J. (February 1975),
    "The Probability Plot Correlation Coefficient Test for Normality",
    Technometrics, American Society for Quality, 17 (1): 111-117
    """
    x, c, n, t = xcnt_handler(x, c, n, t)

    x = np.repeat(x, n)
    c = np.repeat(c, n)
    n = np.ones_like(x)

    idx = np.argsort(c, kind="stable")
    x = x[idx]
    c = c[idx]

    idx2 = np.argsort(x, kind="stable")
    x = x[idx2]
    c = c[idx2]
    N = len(x)

    ranks = rank_adjust(x, c)
    d = 1 - c
    r = np.linspace(N, 1, num=N)

    F = (ranks - 0.3175) / (N + 0.365)
    F[0] = 1 - (0.5 ** (1.0 / N))
    F[-1] = 0.5 ** (1.0 / N)

    out = {k: v for k, v in zip(["x", "r", "d"], (x, r, d))}
    out["R"] = 1 - F
    return out
