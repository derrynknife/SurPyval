import numpy as np
from itertools import tee

def pairwise(iterable):
    """
    s -> (s0, s1), (s1, s2), (s2, s3), ...
    """
    a, b = tee(iterable, 2)
    next(b, None)
    return zip(a, b)

def find_turnbull_bounds(left, right):
    left = np.unique(left[left > 0])
    right = np.unique(right)
    left.sort()
    right.sort()
    flag = np.hstack([np.zeros_like(left), np.ones_like(right)]).astype(int)
    tau = np.hstack([left, right])
    idx = np.argsort(flag, kind='stable')[::-1]
    flag = flag[idx]
    tau = tau[idx]
    idx = np.argsort(tau, kind='stable')
    flag = flag[idx]
    tau = tau[idx]
    bounds = []
    for i, (l, r) in enumerate(pairwise(flag)):
        if ((l == 0) and (r == 1)):
            bounds.append((tau[i], tau[i+1]))
    return bounds

def turnbull(left, right):
    """
    WARNING: I DO NOT KNOW IF THIS IS CORRECT
    Using this I do get a survival curve looking output. But I am not confident
    it is accurate
    """
    max_iter = 1000
    intervals = find_turnbull_bounds(left, right)
    
    m = len(intervals)
    n = len(left)
    p = np.ones(m)/m
    alphas = np.zeros((n, m))
    for j, (l, r) in enumerate(intervals):
        #alphas[:, j] = (((left  < r) & (left  > l)) |
        #                ((right < r) & (right > l)) |
        #                ((right > r) & (left  < r)) |
        #                ((left  < l) & (right > l)) |
                        #((l == left)) |
        #                ((r == right))).astype(int)
        alphas[:, j] = (((left <  r) & (right >= r)) |
                        ((left <  l) & (right >  l)) |
                        ((left >= l) & (right <= r))
                        ).astype(int)
    d = np.zeros(m)
    iters = 0
    p_1 = np.zeros_like(p)
    while (not np.isclose(p, p_1).all()) and (iters < max_iter):
        p_1 = p
        iters += 1
        conditional_p = np.zeros_like(alphas)
        denom = np.zeros(n)
        for i in range(n):
            denom[i] = (alphas[i, :] * p).sum()
        for j in range(m):
            conditional_p[:, j] = alphas[:, j] * p[j]
        d = (conditional_p / np.atleast_2d(denom).T).sum(axis=0)
        
        r = np.ones_like(d) * n + d
        r = r - d.cumsum()
        R = np.cumprod(1 - d/r)
        p = np.abs(np.diff(np.hstack([[1], R])))
    
    return intervals, p, d, r