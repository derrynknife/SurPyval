import autograd.numpy as np
from scipy.stats import pearsonr
from surpyval import nonparametric as nonp
from scipy.optimize import minimize

def mpp(dist, x, c, n, t=None, heuristic="Turnbull", rr='y', on_d_is_0=False, offset=False):
	"""
	MPP: Method of Probability Plotting
	Yes, the order of this language was invented to keep MXX format consistent
	This is the classic probability plotting paper method.

	This method creates the plotting points, transforms it to Weibull scale and then fits the line of best fit.

	Fit a two parameter Weibull distribution from data.
	
	Fits a Weibull model using cumulative probability from x values. 
	"""
	
	if rr not in ['x', 'y']:
		raise ValueError("rr must be either 'x' or 'y'")

	if hasattr(dist, 'mpp'):
		return dist.mpp(x, c, n, heuristic=heuristic, rr=rr, on_d_is_0=on_d_is_0, offset=offset)
	
	
	x_, r, d, F = nonp.plotting_positions(x, c=c, n=n, t=t, heuristic=heuristic)

	x_mask = np.isfinite(x_)
	x_ = x_[x_mask]
	F = F[x_mask]
	d = d[x_mask]
	r = r[x_mask]
	
	if not on_d_is_0:
		x_ = x_[d > 0]
		y_ = F[d > 0]
	else:
		y_ = F

	mask = (y_ != 0) & (y_ != 1)
	y_pp = y_[mask]
	x_pp = x_[mask]
	y_pp = dist.mpp_y_transform(y_pp)

	if offset:		
		# I think this should be x[c != 1] and not in xl (left boundary of intervals)
		x_min = np.min(x_pp)

		# fun = lambda gamma : -pearsonr(np.log(x - gamma), y_)[0]
		def fun(gamma):
			g =  x_min - np.exp(-gamma)
			out = -pearsonr(dist.mpp_x_transform(x_pp - g), y_pp)[0]
			return out

		res = minimize(fun, 0.)
		gamma = x_min - np.exp(-res.x[0])
		x_pp = x_pp - gamma
	
	x_pp = dist.mpp_x_transform(x_pp)
		
	if   rr == 'y':
		params = np.polyfit(x_pp, y_pp, 1)
	elif rr == 'x':
		params = np.polyfit(y_pp, x_pp, 1)

	params = dist.unpack_rr(params, rr)

	if offset:
		return [gamma, *params]
	else:
		return params