from typing import Callable

import numpy as np
import numpy.typing as npt
from pandas import Series

from surpyval.univariate import nonparametric as nonp
from surpyval.univariate.nonparametric.fleming_harrington import (
    fleming_harrington,
)
from surpyval.univariate.nonparametric.kaplan_meier import kaplan_meier
from surpyval.univariate.nonparametric.nelson_aalen import nelson_aalen
from surpyval.utils import xcnt_handler, xcnt_to_xrd

# Estimator-form heuristics share one dispatch; the (A, B) constants
# define the rank-based plotting-position formula F = (rank - A) / (N + B).
# Imported from their defining modules rather than through the package
# namespace. Each shares its name with the submodule that defines it, so
# ``nonp.nelson_aalen`` is the function only once the package __init__
# has bound it over the submodule -- and this dict is built at import
# time, while that __init__ is still running. The other ``nonp.`` uses
# below are inside functions, so they resolve after it has finished.
ESTIMATOR_FUNCS: dict[str, Callable[..., npt.NDArray]] = {
    "Nelson-Aalen": nelson_aalen,
    "Kaplan-Meier": kaplan_meier,
    "Fleming-Harrington": fleming_harrington,
}

HEURISTIC_AB = {
    "Blom": (0.375, 0.25),
    "Median": (0.3, 0.4),
    "ECDF": (0.0, 0.0),
    "ECDF_Adj": (0.0, 1.0),
    "Modal": (1.0, -1.0),
    "Midpoint": (0.5, 0.0),
    "Mean": (0.0, 1.0),
    "Weibull": (0.0, 1.0),
    # Benard's median-rank approximation, (i - 0.3) / (N + 0.4): the same
    # formula as "Median" (B was 0.2, which is no published heuristic).
    "Benard": (0.3, 0.4),
    "Beard": (0.31, 0.38),
    "Hazen": (0.5, 0.0),
    "Gringorten": (0.44, 0.12),
    "None": (0.0, 0.0),
    "Larsen": (0.567, -0.134),
    "Tukey": (1.0 / 3.0, 1.0 / 3.0),
    "DPW": (1.0, 0.0),
}


def plotting_positions(
    x: npt.ArrayLike,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    t: npt.ArrayLike | None = None,
    heuristic: str = "Blom",
    turnbull_estimator: str = "Fleming-Harrington",
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Empirical estimates of the CDF, F, at each observation: the points a
    probability plot draws, and the values the ``MPP`` (probability
    plotting) fitting method regresses on.

    Two kinds of estimate are available. The classical *rank heuristics*
    place the i-th of N ordered observations at

    .. math::
        F_i = \\frac{i - A}{N + B}

    with constants ``(A, B)`` that depend on the heuristic (Blom uses
    ``A = 0.375, B = 0.25``, Median ``A = 0.3, B = 0.4``, and so on).
    With right-censored data the ranks ``i`` are adjusted by the mean
    order number (Johnson's method) before the formula is applied. The
    *estimator* heuristics (``'Nelson-Aalen'``, ``'Kaplan-Meier'``,
    ``'Fleming-Harrington'``, ``'Turnbull'``) instead return one minus the
    non-parametric survival estimate, and ``'Filliben'`` uses Filliben's
    order-statistic medians.

    Parameters
    ----------

    x : array like
        Array of observations of the random variables.
    c : array like, optional
        Array of censoring flag. -1 is left censored, 0 is observed, 1 is
        right censored, and 2 is intervally censored. If not provided will
        assume all values are observed. Left- and interval-censored data
        need ``heuristic='Turnbull'``.
    n : array like, optional
        Array of counts for each x. If data is provided as counts, then this
        can be provided. If :code:`None` will assume each observation is 1.
    t : 2D-array like, optional
        2D array like of the left and right values at which the respective
        observation was truncated. If not provided it assumes that no
        truncation occurs. Left truncation needs one of the estimator
        heuristics; right truncation needs ``'Turnbull'``.
    heuristic : str, optional
        The method used to compute F. One of ``'Blom'`` (the default),
        ``'Median'``, ``'ECDF'``, ``'ECDF_Adj'``, ``'Modal'``,
        ``'Midpoint'``, ``'Mean'``, ``'Weibull'``, ``'Benard'``,
        ``'Beard'``, ``'Hazen'``, ``'Gringorten'``, ``'None'``,
        ``'Larsen'``, ``'Tukey'``, ``'DPW'``, ``'Filliben'``,
        ``'Nelson-Aalen'``, ``'Kaplan-Meier'``, ``'Fleming-Harrington'``
        or ``'Turnbull'``.
    turnbull_estimator : str, optional
        If using the Turnbull heuristic, the estimator used with the
        Turnbull estimates of the risk and death sets: one of
        ``'Fleming-Harrington'`` (the default), ``'Nelson-Aalen'`` or
        ``'Kaplan-Meier'``.

    Returns
    -------

    x : numpy array
        x values for the plotting points. The rank heuristics and
        ``'Filliben'`` return one row per item (a value with ``n = 3``
        appears three times); the estimator heuristics return the ``x``
        of the corresponding fitted model (one row per distinct value, or
        for ``'Turnbull'`` the endpoints of the Turnbull pieces).
    r : numpy array
        risk set at each x
    d : numpy array
        death set at each x (for the rank heuristics, 1 for a failure and
        0 for a censored item)
    F : numpy array
        estimate of F to use in plotting positions. Only the rows with
        ``d > 0`` are meant to be plotted: for the rank heuristics a
        censored row carries the previous failure's value (0 before the
        first failure), and for ``'Filliben'`` it is NaN.

    Examples
    --------

    >>> from surpyval.univariate.nonparametric import plotting_positions
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> x, r, d, F = plotting_positions(x, heuristic="Filliben")
    >>> F
    array([0.08299596, 0.20113568, 0.32068141, 0.44022714, 0.55977286,
           0.67931859, 0.79886432, 0.91700404])
    """

    x, c, n, t = xcnt_handler(x, c, n, t)

    if heuristic not in nonp.PLOTTING_METHODS:
        raise ValueError("Must use available heuristic")

    if ((-1 in c) or (2 in c)) & (heuristic != "Turnbull"):
        raise ValueError(
            "Left or interval censored data requires "
            + "the use of the Turnbull estimator"
        )

    if (np.isfinite(t[:, 0]).any()) & (
        heuristic
        not in [
            "Nelson-Aalen",
            "Kaplan-Meier",
            "Fleming-Harrington",
            "Turnbull",
        ]
    ):
        raise ValueError(
            "Left truncated data can only be used with "
            + "'Nelson-Aalen', 'Kaplan-Meier', "
            + "'Fleming-Harrington', and 'Turnbull' estimators"
        )

    if (np.isfinite(t[:, 1]).any()) & (heuristic != "Turnbull"):
        raise ValueError(
            "Right truncated data can only be used with "
            + "'Turnbull' estimator"
        )

    N = n.sum()

    if heuristic == "Filliben":
        out = nonp.filliben(x, c, n, t)
    elif heuristic in ESTIMATOR_FUNCS:
        x_e, r, d = xcnt_to_xrd(x, c, n, t)
        R = ESTIMATOR_FUNCS[heuristic](r, d)
        return x_e, r, d, 1 - R
    elif heuristic == "Turnbull":
        out = nonp.turnbull(x, c, n, t, estimator=turnbull_estimator)
    else:
        # Reformat for plotting point style
        x_ = np.repeat(x, n)
        c = np.repeat(c, n)
        n = np.ones_like(x_)

        idx = np.argsort(c, kind="stable")
        x_ = x_[idx]
        c = c[idx]

        idx2 = np.argsort(x_, kind="stable")
        x_ = x_[idx2]
        c = c[idx2]

        ranks = nonp.rank_adjust(x_, c=c)
        d = 1 - c
        r = np.linspace(N, 1, num=N)

        A, B = HEURISTIC_AB[heuristic]

        F = (ranks - A) / (N + B)
        R = 1 - Series(F).ffill().fillna(0).values
        out = {}
        out["x"] = x_
        out["r"] = r
        out["d"] = d
        out["R"] = R

    return out["x"], out["r"], out["d"], 1 - out["R"]
