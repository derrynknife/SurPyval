import numpy as np

def success_run(n, confidence=0.95, alpha=None):
    if alpha is None: alpha = 1 - confidence
    return np.power(alpha, 1./n)