import numpy as np
import pandas as pd

from scipy.stats import t, norm
from scipy.stats import rankdata
from scipy.special import ndtri as z
from itertools import tee

import surpyval
from surpyval import nonparametric as nonp

def plotting_positions(x, c=None, n=None, t=None, heuristic="Blom", 
                       turnbull_estimator='Fleming-Harrington'):
    r"""
    This function takes in data in the xcnt format and outputs an approximation
    of the CDF. This function can be used to produce estimates of F using the 
    Nelson-Aalen, Kaplan-Meier, Fleming-Harrington, and the Turnbull estimates.
    Additionally, it can be used to create 'plotting heuristics.'

    Plotting heuristics are the values that are used to plot on probability paper
    and can be used to estiamte the parameters of a distribution. The use of 
    probability plots is one of the traditional ways to estimate the parameters
    of a distribution.

    If right censored data can be used by the regular plotting positions. If
    there is right censored data this method adjusts the ranks of the values
    using the mean order number.

    Parameters
    ----------

    x : array like, optional
        Array of observations of the random variables. If x is :code:`None`, xl and xr must be provided.
    c : array like, optional
        Array of censoring flag. -1 is left censored, 0 is observed, 1 is right censored, 
        and 2 is intervally censored. If not provided will assume all values are observed.
    n : array like, optional
        Array of counts for each x. If data is proivded as counts, then this can be provided. If :code:`None`
        will assume each observation is 1.
    t : 2D-array like, optional
        2D array like of the left and right values at which the respective observation was truncated. If
        not provided it assumes that no truncation occurs.
    heuristic : ("Blom", "Median", "ECDF", "ECDF_Adj", "Modal", "Midpoint", "Mean", "Weibull", 
                 "Benard", "Beard", "Hazen", "Gringorten", "None", "Larsen", "Tukey", "DPW"), str, optional
        Method to use to compute the heuristic of F. See details of each heursitic in the
        `probability plotting section <https://surpyval.readthedocs.io/en/latest/Parametric%20Modelling.html#method-of-probability-plotting-mpp>`_.
    turnbull_estimator : ('Nelson-Aalen', 'Kaplan-Meier'), str, optional
        If using the Turnbull heuristic, you can elect to use the NA or KM method to compute R
        with the Turnbull estimates of the risk and deat sets.

    Returns
    -------

    x : numpy array
        x values for the plotting points
    r : numpy array
        risk set at each x
    d : numpy array
        death set at each x
    F : numpy array
        estimate of F to use in plotting positions.

    Examples
    --------

    >>> from surpyval.nonparametric import plotting_positions
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> x, r, d, F = plotting_positions(x, heuristic="Filliben")
    >>> F
    array([0.08299596, 0.20113568, 0.32068141, 0.44022714, 0.55977286,
           0.67931859, 0.79886432, 0.91700404])
    """

    x, c, n, t = surpyval.xcnt_handler(x, c, n, t)

    if heuristic not in nonp.PLOTTING_METHODS:
        raise ValueError("Must use available heuristic")

    if ((-1 in c) or (2 in c)) & (heuristic != 'Turnbull'):
        raise ValueError("Left or interval censored data requires the use of the Turnbull estimator")

    if (np.isfinite(t[:, 0]).any()) & (heuristic not in ['Nelson-Aalen', 'Kaplan-Meier', 'Fleming-Harrington', 'Turnbull']):
        raise ValueError("Left truncated data can only be used with 'Nelson-Aalen', 'Kaplan-Meier', 'Fleming-Harrington', and 'Turnbull' estimators")

    if (np.isfinite(t[:, 1]).any()) & (heuristic != 'Turnbull'):
        raise ValueError("Right truncated data can only be used with 'Turnbull' estimator")

    N = n.sum()

    if heuristic == 'Filliben':
        # Needs work
        x_, r, d, R = nonp.filliben(x, c, n)
        F = 1 - R 
        return x_, r, d, F
    elif heuristic == 'Nelson-Aalen':
        x_, r, d, R = nonp.nelson_aalen(x, c, n, t=t)
        F = 1 - R
        return x_, r, d, F
    elif heuristic == 'Kaplan-Meier':
        x_, r, d, R = nonp.kaplan_meier(x, c, n, t=t)
        F = 1 - R
        return x_, r, d, F
    elif heuristic == 'Fleming-Harrington':
        x_, r, d, R = nonp.fleming_harrington(x, c, n, t=t)
        F = 1 - R
        return x_, r, d, F
    elif heuristic == 'Turnbull':
        x_, r, d, R = nonp.turnbull(x, c, n, t, estimator=turnbull_estimator)
        F = 1 - R
        return x_, r, d, F
    else:
        # Reformat for plotting point style
        x_ = np.repeat(x, n)
        c = np.repeat(c, n)
        n = np.ones_like(x_)

        idx = np.argsort(c, kind='stable')
        x_ = x_[idx]
        c  = c[idx]

        idx2 = np.argsort(x_, kind='stable')
        x_ = x_[idx2]
        c  = c[idx2]

        ranks = nonp.rank_adjust(x_, c=c)
        d = 1 - c
        r = np.linspace(N, 1, num=N)

        if   heuristic == "Blom":       A, B = 0.375, 0.25
        elif heuristic == "Median":     A, B = 0.3, 0.4
        elif heuristic == "ECDF":       A, B = 0, 0
        elif heuristic == "ECDF_Adj":   A, B = 0, 1
        elif heuristic == "Modal":      A, B = 1.0, -1.0
        elif heuristic == "Midpoint":   A, B = 0.5, 0.0
        elif heuristic == "Mean":       A, B = 0.0, 1.0
        elif heuristic == "Weibull":    A, B = 0.0, 1.0
        elif heuristic == "Benard":     A, B = 0.3, 0.2
        elif heuristic == "Beard":      A, B = 0.31, 0.38
        elif heuristic == "Hazen":      A, B = 0.5, 0.0
        elif heuristic == "Gringorten": A, B = 0.44, 0.12
        elif heuristic == "None":       A, B = 0.0, 0.0
        elif heuristic == "Larsen":     A, B = 0.567, -0.134
        elif heuristic == "Tukey":      A, B = 1./3., 1./3.
        elif heuristic == "DPW":        A, B = 1.0, 0.0

        F = (ranks - A)/(N + B)
        F = pd.Series(F).ffill().fillna(0).values
        return x_, r, d, F