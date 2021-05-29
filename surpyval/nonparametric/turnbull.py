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

def turnbull(x, c, n, t, estimator='Kaplan-Meier'):
	x = x.astype(float)
	t = t.astype(float)
	bounds = np.unique(np.concatenate([np.unique(x), np.unique(t)]))
	N = n.sum()

	if x.ndim == 1:
		x_new = np.empty(shape=(x.shape[0], 2))
		x_new[:, 0] = x
		x_new[:, 1] = x
		x = x_new

	# Unpack x array
	xl = x[:, 0]
	xr = x[:, 1]

	# Unpack t array
	tl = t[:, 0]
	tr = t[:, 1]
	
	# If there are left and right censored observations, convert them to interval censored observations
	xl[c == -1] = -np.inf
	xr[c ==  1] =  np.inf
	
	M = bounds.size
	N = xl.size
	
	alpha = np.zeros(shape=(N, M))
	beta  = np.zeros(shape=(N, M))
	for i in range(0, N):
		x1, x2 = xl[i], xr[i]
		t1, t2 = tl[i], tr[i]
		if x1 == x2:
			alpha[i, :] = (bounds == x1).astype(int) * n[i]
		else:
			alpha[i, :] = ((bounds >= x1) & (bounds < x2)).astype(int) * n[i]
		beta[i, :]  = 1 - ((bounds >= t1) & (bounds <= t2)).astype(int)

	d = np.zeros(M)
	p = np.ones(M)/M
	
	iters = 0
	p_prev = np.zeros_like(p)
		
	if estimator == 'Kaplan-Meier':
		func = km
	else:
		func = na
	
	while (not np.isclose(p, p_prev, atol=1e-100).all()) and (iters < 100000):
		p_prev = p
		ap = alpha * p
		bp = beta * p
		# Expected failures
		mu = ap / ap.sum(axis=1).reshape(-1, 1)
		# Expected additional failures due to truncation
		nu = bp / (1 - bp).sum(axis=1).reshape(-1, 1)
		d = (nu + mu).sum(axis=0)
		r = d.sum() - d.cumsum() + d
		R = func(d, r)
		p = np.abs(np.diff(np.hstack([[1], R])))
		# The 'official' way to do it. This is the Kaplan-Meier
		# p = (nu + mu).sum(axis=0)/(nu + mu).sum()
	mask = np.isfinite(bounds)
	x = bounds[mask]
	r = r[mask]
	d = d[mask]
	R = R[mask]
	return x, r, d, R

class Turnbull_(NonParametricFitter):
	def __init__(self):
		self.how = 'Turnbull'

Turnbull = Turnbull_()