import numpy as np
import numpy.typing as npt
from pandas import Series

from surpyval.univariate.nonparametric.rank_adjust import rank_adjust
from surpyval.utils import xcnt_handler


def filliben(
    x: npt.ArrayLike,
    c: npt.ArrayLike | None,
    n: npt.ArrayLike | None,
    t: npt.ArrayLike | None,
) -> dict:
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

    # Filliben's medians of the uniform order statistics: the general
    # formula, with exact end-point values for the smallest and largest
    # of the N. With right censoring the failures carry mean order numbers
    # (Johnson's adjusted ranks, as for the other rank heuristics), so an
    # end-point value belongs to a failure whose adjusted rank *is* 1 or N
    # -- the first or last failure when no item is censored before or
    # after it respectively. Assigning them to the first and last rows
    # instead gave the censored unit the largest position when the last
    # item was censored, and a failure at rank 4 of 5 kept the general
    # value only by luck of the row order.
    F = (ranks - 0.3175) / (N + 0.365)
    F[np.isclose(ranks, 1.0)] = 1 - (0.5 ** (1.0 / N))
    F[np.isclose(ranks, N)] = 0.5 ** (1.0 / N)
    # Censored rows carry the previous failure's value (0 before the first
    # failure), as for the other rank heuristics, rather than NaN.
    F = Series(F).ffill().fillna(0).values

    out = {k: v for k, v in zip(["x", "r", "d"], (x, r, d))}
    out["R"] = 1 - F
    return out
