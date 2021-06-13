import pytest
import numpy as np
from surpyval import *

DISTS = [Gumbel, Normal, Weibull, LogNormal, Logistic, LogLogistic]
parameter_sample_bounds = [((1, 20), (0.5, 5)),
						   ((1, 100), (0.5, 100)),
						   ((1, 100), (0.5, 20)),
						   ((1, 3), (0.2, 1)),
						   ((1, 100), (0.5, 20)),
						   ((1, 100), (0.5, 20)),
						   ]
FIT_SIZES = [5000, 10000, 20000, 50000]

def generate_test_cases():
	for idx, dist in enumerate(DISTS):
		bounds = parameter_sample_bounds[idx]
		for kind in ['full', 'censored', 'truncated', 'interval']:
			yield dist, bounds, kind

def idfunc(x):
	if type(x) is tuple:
		return 'bounds'
	elif type(x) is str:
		return x
	else:
		return x.name


def interval_censor(x, n=100):
	n, xx = np.histogram(x, bins=n)
	x = np.vstack([xx[0:-1], xx[1::]]).T
	x = x[n > 0]
	n = n[n > 0]
	return x, n

def censor_at(x, q, where='right'):
	c = np.zeros_like(x)
	x = np.copy(x)
	if where == 'right':
		x_q = np.quantile(x, 1 - q)
		mask = x > x_q
		c[mask] = 1
		x[mask] = x_q
		return x, c
	elif where == 'left':
		x_q = np.quantile(x, q)
		mask = x < x_q
		c[mask] = -1
		x[mask] = x_q
		return x, c
	elif where == 'both':
		x_u = np.quantile(x, 1 - q)
		x_l = np.quantile(x, q)
		mask_l = x < x_l
		mask_u = x > x_u
		c[mask_l] = -1
		c[mask_u] =  1
		x[mask_l] = x_l
		x[mask_u] = x_u
		return x, c
	else:
		raise ValueError("'where' parameter not correctly defined")

def truncate_at(x, q, where='right'):
	x = np.copy(x)
	if where == 'right':
		x_q = np.quantile(x, 1 - q)
		x = x[x < x_q]
		return x, None, x_q
	elif where == 'left':
		x_q = np.quantile(x, q)
		x = x[x > x_q]
		return x, x_q, None
	elif where == 'both':
		x_u = np.quantile(x, 1 - q)
		x_l = np.quantile(x, q)
		x = x[x < x_u]
		x = x[x > x_l]
		return x, x_l, x_u
	else:
		raise ValueError("'where' parameter not correctly defined")


@pytest.mark.parametrize("dist,bounds,kind", generate_test_cases(), ids=idfunc)
def test_mle(dist, bounds, kind):
	for n in FIT_SIZES:
		test_params = []
		for b in bounds:
			test_params.append(np.random.uniform(*b))
		test_params = np.array(test_params)
		x = dist.random(n, *test_params)
		if kind == 'full':
			model = dist.fit(x)
			tol = 0.1
		elif kind == 'censored':
			x, c = censor_at(x, 0.025, 'right')
			tol = 0.1
			model = dist.fit(x, c=c)
		elif kind == 'truncated':
			x, tl, tr = truncate_at(x, 0.05, 'both')
			model = dist.fit(x, tl=tl, tr=tr)
			tol = 0.15
		elif kind == 'interval':
			x, n = interval_censor(x)
			model = dist.fit(x=x, n=n)
			tol = 0.15
		if model.params == []:
			continue
		fitted_params = np.array(model.params)
		max_params = np.max([fitted_params, test_params], axis=0)
		diff = np.abs(fitted_params - test_params) / max_params
		if (diff < tol).all():
			break
	else:
		raise AssertionError('fit not very good in %s\n' % dist.name)

