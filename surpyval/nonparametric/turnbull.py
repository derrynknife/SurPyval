import numpy as np
from surpyval.nonparametric.nonparametric_fitter import NonParametricFitter

def na(d, r):
	H = np.cumsum(d/r)
	H[np.isnan(H)] = np.inf
	R = np.exp(-H)
	p = np.abs(np.diff(np.hstack([[1], R])))
	return R
	
def km(d, r):
	R = 1 - d/r
	R[np.isnan(R)] = 0
	R = np.cumprod(R)
	return R

def turnbull(x, c, n, estimator='Kaplan-Meier'):
	bounds = np.unique(x)
	N = n.sum()

	if x.ndim == 1:
		x_new = np.empty(shape=(x.shape[0], 2))
		x_new[:, 0] = x
		x_new[:, 1] = x
		x = x_new
	# Unpack x array
	xl = x[:, 0]
	xr = x[:, 1]
	
	# Separate observed
	xl[c == -1] = -np.inf
	xr[c ==  1] =  np.inf
	
	mask = c == 0
	
	x_obs = xl[mask]
	n_obs =  n[mask]
	
	d_obs = np.zeros_like(bounds)
	for xv, nv in zip(x_obs, n_obs):
		idx, = np.where(bounds == np.array(xv))
		d_obs[idx] = nv
	
	xl = xl[~mask]
	xr = xr[~mask]
	ni =  n[~mask]
	
	mask_obs = np.isin(bounds, x_obs)
	
	m = bounds.size
	n = xl.size
	
	alpha = np.zeros(shape=(n, m))
	for i in range(0, n):
		l = xl[i]
		u = xr[i]
		alpha[i, :] = ((bounds >= l) & (bounds < u)).astype(int) * ni[i]
		
	d = np.zeros(m)
	p = np.ones(m)/m
	
	iters = 0
	p_1 = np.zeros_like(p)
		
	if estimator == 'Kaplan-Meier':
		func = km
	else:
		func = na

	while (not np.isclose(p, p_1, atol=1e-30).all()) and (iters < 10000):
			p_1 = p
			iters += 1
			conditional_p = np.zeros_like(alpha)
			denom = np.zeros(n)
			for i in range(n):
				denom[i] = (alpha[i, :] * p).sum()
			for j in range(m):
				conditional_p[:, j] = alpha[:, j] * p[j]

			d = (conditional_p / np.atleast_2d(denom).T).sum(axis=0)
			d += d_obs

			r = np.ones_like(d) * N + d
			r = r - d.cumsum()
			R = func(d, r)
			p = np.abs(np.diff(np.hstack([[1], R])))
	
	return bounds, r, d, R

class Turnbull_(NonParametricFitter):
	def __init__(self):
		self.how = 'Turnbull'

Turnbull = Turnbull_()
