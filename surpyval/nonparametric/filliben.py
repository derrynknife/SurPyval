import numpy as np
from surpyval import nonparametric as nonp
import surpyval

def filliben(x, c=None, n=None):
    """
    Method From:
    Filliben, J. J. (February 1975), 
    "The Probability Plot Correlation Coefficient Test for Normality", 
    Technometrics, American Society for Quality, 17 (1): 111-117
    """
    x, c, n = surpyval.xcn_handler(x, c, n)
        
    x = np.repeat(x, n)
    c = np.repeat(c, n)
    n = np.ones_like(x)

    idx = np.argsort(c, kind='stable')
    x = x[idx]
    c = c[idx]

    idx2 = np.argsort(x, kind='stable')
    x = x[idx2]
    c  = c[idx2]
    N = len(x)
    
    ranks = nonp.rank_adjust(x, c)
    d = 1 - c
    r = np.linspace(N, 1, num=N)
    
    F     = (ranks - 0.3175) / (N + 0.365)
    F[0]  = 1 - (0.5 ** (1./N))
    F[-1] = 0.5 ** (1./N)
    R = 1 - F
    return x, r, d, R