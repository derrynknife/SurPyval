import numpy as np
import surpyval.nonparametric as nonp

def filliben(x, c=None, n=None):
    """
    Method From:
    Filliben, J. J. (February 1975), 
    "The Probability Plot Correlation Coefficient Test for Normality", 
    Technometrics, American Society for Quality, 17 (1): 111-117
    """
    if n is None:
        n = np.ones_like(x)

    if c is None:
        c = np.zeros_like(x)
        
    x_ = np.repeat(x, n)
    c = np.repeat(c, n)
    n = np.ones_like(x_)

    idx = np.argsort(c, kind='stable')
    x_ = x_[idx]
    c  = c[idx]

    idx2 = np.argsort(x_, kind='stable')
    x_ = x_[idx2]
    c  = c[idx2]
    N = len(x_)
    
    ranks = nonp.rank_adjust(x_, c)
    d = 1 - c
    r = np.linspace(N, 1, num=N)
    
    F     = (ranks - 0.3175) / (N + 0.365)
    F[0]  = 1 - (0.5 ** (1./N))
    F[-1] = 0.5 ** (1./N)
    R = 1 - F
    return x, r, d, R