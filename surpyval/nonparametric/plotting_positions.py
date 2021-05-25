import numpy as np
import pandas as pd

from scipy.stats import t, norm
from scipy.stats import rankdata
from scipy.special import ndtri as z
from itertools import tee

import surpyval
from surpyval import nonparametric as nonp

def plotting_positions(x, c=None, n=None, heuristic="Blom", A=None, B=None):
    """
    Good reference for heuristics:
    https://en.wikipedia.org/wiki/Q–Q_plot
    """
    x, c, n = surpyval.xcn_handler(x, c, n)
    assert heuristic in nonp.PLOTTING_METHODS, "Must use available heuristic"

    N = n.sum()

    if heuristic == 'Filliben':
        # Needs work
        x_, r, d, R = nonp.filliben(x, c=c, n=n)
        F = 1 - R 
        return x_, r, d, F
    elif heuristic == 'Nelson-Aalen':
        x_, r, d, R = nonp.nelson_aalen(x, c, n)
        F = 1 - R
        return x_, r, d, F
    elif heuristic == 'Kaplan-Meier':
        x_, r, d, R = nonp.kaplan_meier(x, c, n)
        F = 1 - R
        return x_, r, d, F
    elif heuristic == 'Fleming-Harrington':
        x_, r, d, R = nonp.fleming_harrington(x, c, n)
        F = 1 - R
        return x_, r, d, F
    elif heuristic == 'Turnbull':
        x_, r, d, R = nonp.turnbull(x, c, n)
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
        elif heuristic == "ECDF":       A, B = 0, 1
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