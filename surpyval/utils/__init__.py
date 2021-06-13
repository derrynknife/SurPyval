import numpy as np
from pandas import Series
from collections import defaultdict
import sys

def check_no_censoring(c):
	return any(c != 0)
def no_left_or_int(c):
	return any((c == -1) | (c == 2))
def surv_tolist(x):
	"""
	supplement to the .tolist() function where there is mixed scalars and arrays.

	Survival data can be confusing sometimes. I'm probably not helping by trying
	to make it fit with one input.
	"""
	if x.ndim == 2:        
		return [v[0] if v[0] == v[1] else v.tolist() for v in x]
	else:
		return x.tolist()

def group_xcn(x, c, n):
	"""
	Ensures that all unique combinations of x and c are grouped. This should 
	reduce the chance of having arrays that are too long for the MLE fit.
	"""
	grouped = defaultdict(lambda: defaultdict(int))
	if x.ndim == 2:
		for vx, vc, vn in zip(x, c, n):
			grouped[tuple(vx)][vc] += vn
	else:
		for vx, vc, vn in zip(x, c, n):
			grouped[vx][vc] += vn

	x_out = []
	c_out = []
	n_out = []

	for k, v in grouped.items():
		for cv, nv in v.items():
			x_out.append(k)
			c_out.append(cv)
			n_out.append(nv)
	return np.array(x_out), np.array(c_out), np.array(n_out)
def xcn_sort(x, c, n):
	"""
	Sorts the x, c, and n arrays so that x is increasing, and where there are ties they are ordered as left censored, failure, right censored, then intervally censored.

	Parameters
	----------
	x: array
		array of values of variable for which observations were made.
	c: array
		array of censoring values (-1, 0, 1, 2) corrseponding to x
	n: array
		array of count of observations at each x and with censoring c

	Returns
	----------
	x: array
		sorted array of values of variable for which observations were made.
	c: array
		sorted array of censoring values (-1, 0, 1, 2) corrseponding to rearranged x
	n: array
		array of count of observations at each rearranged x and with censoring c
	"""
	
	idx_c = np.argsort(c, kind='stable')
	x = x[idx_c]
	c = c[idx_c]
	n = n[idx_c]

	if x.ndim == 1:
		idx = np.argsort(x, kind='stable')
	else:
		idx = np.argsort(x[:, 0], kind='stable')

	x = x[idx]
	c = c[idx]
	n = n[idx]
	return x, c, n
def xcn_handler(x, c=None, n=None):
	"""
	Main handler that ensures any input to a surpyval fitter meets the requirements to be used in one of the parametric or nonparametric fitters.

	Parameters
	----------
	x: array
		array of values of variable for which observations were made.
	c: array, optional (default: None)
		array of censoring values (-1, 0, 1, 2) corrseponding to x
	n: array, optional (default: None)
		array of count of observations at each x and with censoring c

	Returns
	----------

	x: array
		sorted array of values of variable for which observations were made.
	c: array
		array of censoring values (-1, 0, 1, 2) corrseponding to output array x. If c was None, defaults to creating array of zeros the length of x.
	n: array
		array of count of observations at to output array x and with censoring c. If n was None, count array assumed to be all one observation.
	"""
	if type(x) == list:
		if any([list == type(v) for v in x]):
			x_ndarray = np.empty(shape=(len(x), 2))
			for idx, val in enumerate(x):
				x_ndarray[idx, :] = np.array(val)
			x = x_ndarray
		else:
			x = np.array(x).astype(float)
	elif type(x) == Series:
		x = np.array(x)

	if x.ndim > 2:
		raise ValueError("Variable (x) array must be one or two dimensional")

	if x.ndim == 2:
		if x.shape[1] != 2:
			raise ValueError("Dimension 1 must be equalt to 2, try transposing data, or do you have a 1d array in a 2d array?")

		if not (x[:, 0] <= x[:, 1]).all():
			raise ValueError("All left intervals must be less than right intervals")

	if c is not None:
		c = np.array(c)
		if c.ndim != 1:
			raise ValueError("censoring array must be one dimensional")

		if c.shape[0] != x.shape[0]:
			raise ValueError("censoring array must be same length as variable array")

		if x.ndim == 2:
			if any(c[x[:, 0] == x[:, 1]] == 2):
				raise ValueError("Censor flag indicates interval censored but only has one failure time")

			if not all(c[x[:, 0] != x[:, 1]] == 2):
				raise ValueError("Censor flag provided, but case where interval flagged as non interval censoring")

			if any((c !=  0) & (c !=  1) & (c != -1) & (c !=  2)):
				raise ValueError("Censoring value must only be one of -1, 0, 1, or 2")

		else:
			if any((c !=  0) & (c !=  1) & (c != -1)):
				raise ValueError("Censoring value must only be one of -1, 0, 1 for single dimension input")
	else:
		c = np.zeros(x.shape[0])
		if x.ndim != 1:
			c[x[:, 0] != x[:, 1]] = 2

	if n is not None:
		n = np.array(n)
		if n.ndim != 1:
			raise ValueError("Count array must be one dimensional")
		if n.shape[0] != x.shape[0]:
			raise ValueError("count array must be same length as variable array.")
		if not (n > 0).all():
			raise ValueError("count array can't be 0")
	else:
		# Do check here for groupby and binning
		n = np.ones(x.shape[0])

	n = n.astype(int)
	c = c.astype(int)
	
	x, c, n = group_xcn(x, c, n)
	x, c, n = xcn_sort(x, c, n)

	return x, c, n

def group_xcnt(x, c, n, t):
	"""
	Ensures that all unique combinations of x and c are grouped. This should 
	reduce the chance of having arrays that are too long for the MLE fit.
	"""
	grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
	if x.ndim == 2:
		for vx, vc, vn, vt in zip(x, c, n, t):
			grouped[tuple(vx)][vc][tuple(vt)] += vn
	else:
		for vx, vc, vn, vt in zip(x, c, n, t):
			grouped[vx][vc][tuple(vt)] += vn

	x_out = []
	c_out = []
	n_out = []
	t_out = []

	for xv, level2 in grouped.items():
		for cv, level3 in level2.items():
			for tv, nv in level3.items():
				x_out.append(xv)
				c_out.append(cv)
				n_out.append(nv)
				t_out.append(tv)
	return np.array(x_out), np.array(c_out), np.array(n_out), np.array(t_out)
def xcnt_sort(x, c, n, t):
	"""
	Sorts the x, c, and n arrays so that x is increasing, and where there are ties they are ordered as left censored, failure, right censored, then intervally censored.

	Parameters
	----------
	x: array
		array of values of variable for which observations were made.
	c: array
		array of censoring values (-1, 0, 1, 2) corrseponding to x
	n: array
		array of count of observations at each x and with censoring c
	t: array, optional (default: None)
		array of values with shape (?, 2) with the left and right value of truncation	

	Returns
	----------
	x: array
		sorted array of values of variable for which observations were made.
	c: array
		sorted array of censoring values (-1, 0, 1, 2) corrseponding to rearranged x
	n: array
		array of count of observations at each rearranged x and with censoring c
	t: array, optional (default: None)
		array of values with shape (?, 2) with the left and right value of truncation	
	"""
	
	idx_c = np.argsort(c, kind='stable')
	x = x[idx_c]
	c = c[idx_c]
	n = n[idx_c]
	t = t[idx_c]

	if t.ndim == 1:
		idx = np.argsort(t, kind='stable')
	else:
		idx = np.argsort(t.min(axis=1), kind='stable')
	x = x[idx]
	c = c[idx]
	n = n[idx]
	t = t[idx]
	
	if x.ndim == 1:
		idx = np.argsort(x, kind='stable')
	else:
		idx = np.argsort(x.mean(axis=1), kind='stable')

	x = x[idx]
	c = c[idx]
	n = n[idx]
	t = t[idx]
	
	return x, c, n, t
def xcnt_handler(x, c=None, n=None, t=None, tl=None, tr=None):
	"""
	Main handler that ensures any input to a surpyval fitter meets the requirements to be used in one of the parametric or nonparametric fitters.

	Parameters
	----------
	x: array
		array of values of variable for which observations were made.
	c: array, optional (default: None)
		array of censoring values (-1, 0, 1, 2) corrseponding to x
	n: array, optional (default: None)
		array of count of observations at each x and with censoring c
	t: array, optional (default: None)
		array of values with shape (?, 2) with the left and right value of truncation
	tl:	array or scalar, optional (default: None)
		array of values of the left value of truncation. If scalar, all values will be treated as left truncated by that value
		cannot be used with 't' parameter but can be used with the 'tr' parameter
	tr: array or scalar, optional (default: None)
		array of values of the right value of truncation. If scalar, all values will be treated as right truncated by that value
		cannot be used with 't' parameter but can be used with the 'tl' parameter

	Returns
	----------

	x: array
		sorted array of values of variable for which observations were made.
	c: array
		array of censoring values (-1, 0, 1, 2) corrseponding to output array x. If c was None, defaults to creating array of zeros the length of x.
	n: array
		array of count of observations at output array x and with censoring c. If n was None, count array assumed to be all one observation.
	t: array
		array of truncation values of observations at output array x and with censoring c.
	"""
	if type(x) == list:
		if any([list == type(v) for v in x]):
			x_ndarray = np.empty(shape=(len(x), 2))
			for idx, val in enumerate(x):
				x_ndarray[idx, :] = np.array(val)
			x = x_ndarray
		else:
			x = np.array(x)
	elif type(x) == Series:
		x = np.array(x)

	if x.ndim > 2:
		raise ValueError("Variable (x) array must be one or two dimensional")

	if x.ndim == 2:
		if x.shape[1] != 2:
			raise ValueError("Dimension 1 must be equalt to 2, try transposing data, or do you have a 1d array in a 2d array?")

		if not (x[:, 0] <= x[:, 1]).all():
			raise ValueError("All left intervals must be less than (or equal to) right intervals")

	if c is not None:
		c = np.array(c)
		if c.ndim != 1:
			raise ValueError("Censoring flag array must be one dimensional")

		if c.shape[0] != x.shape[0]:
			raise ValueError("censoring flag array must be same length as variable array")

		if x.ndim == 2:
			if any(c[x[:, 0] == x[:, 1]] == 2):
				raise ValueError("Censor flag indicates interval censored but only has one failure time")

			if any((c == 2) & (x[:, 0] == x[:, 1])):
				raise ValueError("Censor flag provided, but case where interval flagged as non interval censoring")

			if any((c !=  0) & (c !=  1) & (c != -1) & (c !=  2)):
				raise ValueError("Censoring value must only be one of -1, 0, 1, or 2")

		else:
			if any((c !=  0) & (c !=  1) & (c != -1)):
				raise ValueError("Censoring value must only be one of -1, 0, 1 for single dimension input")

	else:
		c = np.zeros(x.shape[0])
		if x.ndim != 1:
			c[x[:, 0] != x[:, 1]] = 2

	if n is not None:
		n = np.array(n)
		if n.ndim != 1:
			raise ValueError("Count array must be one dimensional")
		if n.shape[0] != x.shape[0]:
			raise ValueError("count array must be same length as variable array.")
		if not (n > 0).all():
			raise ValueError("count array can't be 0")
	else:
		# Do check here for groupby and binning
		n = np.ones(x.shape[0])

	if t is not None and ((tl is not None) or (tr is not None)):
		raise ValueError("Cannot use 't' with 'tl' or 'tr'. Use either 't' or any combination of 'tl' and 'tr'")

	elif (t is None) & (tl is None) & (tr is None):
		tl = np.ones(x.shape[0]) * -np.inf
		tr = np.ones(x.shape[0]) *  np.inf
		t  = np.vstack([tl, tr]).T
	elif (tl is not None) or (tr is not None):
		if tl is None:
			tl = np.ones(x.shape[0]) * -np.inf
		elif np.isscalar(tl):
			tl = np.ones(x.shape[0]) * tl
		else:
			tl = np.array(tl)
		
		if tr is None:
			tr = np.ones(x.shape[0]) * np.inf
		elif np.isscalar(tr):
			tr = np.ones(x.shape[0]) * tr
		else:
			tr = np.array(tr)
			
		if tl.ndim > 1:
			raise ValueError("Left truncation array must be one dimensional, did you mean to use 't'")
		if tr.ndim > 1:
			raise ValueError("Left truncation array must be one dimensional, did you mean to use 't'")
		if tl.shape[0] != x.shape[0]:
			raise ValueError("Left truncation array must be same length as variable array")
		if tr.shape[0] != x.shape[0]:
			raise ValueError("Right truncation array must be same length as variable array")
		if tl.shape != tr.shape:
			raise ValueError("Left truncation array and right truncation array must be the same length")
		t = np.vstack([tl, tr]).T

	else:
		t = np.array(t)
		if t.ndim != 2:
			raise ValueError("Truncation ndarray must be 2 dimensional")
		if t.shape[0] != x.shape[0]:
			raise ValueError("Truncation ndarray must be same shape as variable array")
			
	if (t[:, 1] <= t[:, 0]).any():
		raise ValueError("All left truncated values must be less than right truncated values")
	if x.ndim == 2:
		if ((t[:, 0] > x[:, 0]) & (np.isfinite(t[:, 0]))).any():
			raise ValueError("All left truncated values must be less than the respective observed values")
		elif ((t[:, 1] < x[:, 1]) & (np.isfinite(t[:, 1]))).any():            
			raise ValueError("All right truncated values must be greater than the respective observed values")
	else:
		if (t[:, 0] >= x).any():            
			raise ValueError("All left truncated values must be less than the respective observed values")
		elif (t[:, 1] <= x).any():            
			raise ValueError("All right truncated values must be greater than the respective observed values")

	x = x.astype(float)
	c = c.astype(int)
	n = n.astype(int)
	t = t.astype(float)
	
	x, c, n, t = group_xcnt(x, c, n, t)
	x, c, n, t = xcnt_sort(x, c, n, t)

	return x, c, n, t

def xcnt_to_xrd(x, c=None, n=None, **kwargs):
	"""
	Converts the xcn format to the xrd format.

	Parameters
	----------
	x: array
		array of values of variable for which observations were made.
	c: array, optional (default: None)
		array of censoring values (-1, 0, 1, 2) corrseponding to x. If None, an array of 0s is created corresponding to each x.
	n: array, optional (default: None)
		array of count of observations at each x and with censoring c. If None, an array of ones is created.
	kwargs: keywords for truncation can be either 't' or a combo of 'tl' and 'tr'

	Returns
	----------
	x: array
		sorted array of values of variable for which observations were made.
	r: array
		array of count of units/people at risk at time x.
	d: array
		array of the count of failures/deaths at each time x.
	"""
	x, c, n, t = xcnt_handler(x, c, n, **kwargs)
        
	if np.isfinite(t[:, 1]).any():
		raise ValueError("xrd format can't be used right truncated data")
        
	if (t[:, 0] == t[0, 0]).all() & np.isfinite(t[0, 0]):
		print("Ignoring left truncated values as all observations truncated at same value", file=sys.stderr)
        
	if ((c != 1) & (c != 0)).any():
		raise ValueError("xrd format can't be used with left (c=-1) or interval (c=2) censoring")

	x = np.repeat(x, n)
	c = np.repeat(c, n)
	t = np.repeat(t[:, 0], n)
	n = np.ones_like(x).astype(int)

	x, idx = np.unique(x, return_inverse=True)

	d = np.bincount(idx, weights=1 - c)
	le = (t.reshape(-1, 1) <= x).sum(axis=0)
	# do is drop outs
	do = np.bincount(idx, weights=c)
	r = le + d - d.cumsum() + do - do.cumsum()
	r = r.astype(int)
	d = d.astype(int)
	return x, r, d
def xcn_to_xrd(x, c=None, n=None):
	"""
	Converts the xcn format to the xrd format.

	Parameters
	----------
	x: array
		array of values of variable for which observations were made.
	c: array, optional (default: None)
		array of censoring values (-1, 0, 1, 2) corrseponding to x. If None, an array of 0s is created corresponding to each x.
	n: array, optional (default: None)
		array of count of observations at each x and with censoring c. If None, an array of ones is created.

	Returns
	----------
	x: array
		sorted array of values of variable for which observations were made.
	r: array
		array of count of units/people at risk at time x.
	d: array
		array of the count of failures/deaths at each time x.
	"""
	x, c, n = xcn_handler(x, c, n)

	if ((c != 1) & (c != 0)).any():
		raise ValueError("xrd format can't handle left (c=-1) or interval (c=2) censoring")

	x = np.repeat(x, n)
	c = np.repeat(c, n)
	n = np.ones_like(x).astype(int)

	x, idx = np.unique(x, return_inverse=True)

	d = np.bincount(idx, weights=1 - c)
	# do is drop outs
	do = np.bincount(idx, weights=c)
	r = n.sum() + d - d.cumsum() + do - do.cumsum()
	r = r.astype(int)
	d = d.astype(int)
	return x, r, d
def xrd_to_xcn(x, r, d):
	"""
	Exact inverse of the xcn_to_xrd function.
	"""
	n_f = np.copy(d)
	x_f = np.copy(x)
	mask = n_f != 0
	n_f = n_f[mask]
	x_f = x_f[mask]

	delta = np.abs(np.diff(np.hstack([r, [0]])))

	sus = (delta - d)
	x_s = x[sus > 0]
	n_s = sus[sus > 0]

	x_f = np.repeat(x_f, n_f)
	x_s = np.repeat(x_s, n_s)

	return fs_to_xcn(x_f, x_s)
def fsli_to_xcn(f, s, l, i):
	"""
	Main handler that ensures any input to a surpyval fitter meets the requirements to be used in
	 one of the parametric or nonparametric fitters.

	Parameters
	----------
	f: array
		array of values for which the failure/death was observed
	s: array
		array of right censored observation values
	l: array
		array of left censored observation values
	i: array
		array of interval censored data

	Returns
	----------
	x: array
		sorted array of values of variable for which observations were made.
	c: array
		array of censoring values (-1, 0, 1, 2) corrseponding to output array x.
	n: array
		array of count of observations at to output array x and with censoring c.
	"""
	x_f, n_f = np.unique(f, return_counts=True)
	x_f = np.array([x_f, x_f]).T
	c_f = np.zeros_like(x_f)

	x_s, n_s = np.unique(s, return_counts=True)
	x_s = np.array([x_s, x_s]).T
	c_s = np.ones_like(x_s)

	x_l, n_l = np.unique(l, return_counts=True)
	x_l = np.array([x_l, x_l]).T
	c_l = -np.ones_like(x_l)

	x_i, n_i = np.unique(i, axis=0, return_counts=True)
	c_i = -np.ones_like(x_i)

	x = np.concatenate([x_f, x_s, x_l, x_i])
	c = np.hstack([c_f, c_s, c_l]).astype(int)
	n = np.hstack([n_f, n_s, n_l]).astype(int)

	x, c, n = xcn_sort(x, c, n)

	return x, c, n
def fsl_to_xcn(f, s, l):
	"""
	Main handler that ensures any input to a surpyval fitter meets the requirements to be used in one of the parametric or nonparametric fitters.

	Parameters
	----------
	f: array
		array of values for which the failure/death was observed
	s: array
		array right censored observation values
	l: array
		array left censored observation values

	Returns
	----------
	x: array
		sorted array of values of variable for which observations were made.
	c: array
		array of censoring values (-1, 0, 1, 2) corrseponding to output array x.
	n: array
		array of count of observations at to output array x and with censoring c.
	"""
	x_f, n_f = np.unique(f, return_counts=True)
	c_f = np.zeros_like(x_f)

	x_s, n_s = np.unique(s, return_counts=True)
	c_s = np.ones_like(x_s)

	x_l, n_l = np.unique(l, return_counts=True)
	c_l = -np.ones_like(x_l)

	x = np.hstack([x_f, x_s, x_l])
	c = np.hstack([c_f, c_s, c_l]).astype(int)
	n = np.hstack([n_f, n_s, n_l]).astype(int)

	x, c, n = xcn_sort(x, c, n)

	return x, c, n
def fs_to_xcn(f, s):
	"""
	Main handler that ensures any input to a surpyval fitter meets the requirements to be used in one of the parametric or nonparametric fitters.

	Parameters
	----------
	f: array
		array of values for which the failure/death was observed
	s: array
		array right censored observation values

	Returns
	----------
	x: array
		sorted array of values of variable for which observations were made.
	c: array
		array of censoring values (-1, 0, 1, 2) corrseponding to output array x.
	n: array
		array of count of observations at to output array x and with censoring c.
	"""
	x_f, n_f = np.unique(f, return_counts=True)
	c_f = np.zeros_like(x_f)

	x_s, n_s = np.unique(s, return_counts=True)
	c_s = np.ones_like(x_s)

	x = np.hstack([x_f, x_s])
	c = np.hstack([c_f, c_s]).astype(int)
	n = np.hstack([n_f, n_s]).astype(int)

	x, c, n = xcn_sort(x, c, n)

	return x, c, n
def tl_tr_to_t(tl=None, tr=None):
	if tl is not None: tl = np.array(tl)
	if tr is not None: tr = np.array(tr)
	
	if (tl is None) & (tr is None):
		raise ValueError("One input must not be None")
	elif (tl is not None) and (tr is not None):
		if tl.shape != tr.shape:
			raise ValueError("Left array must be the same shape as right array")
	elif tl is None:
		tl = np.ones_like(tr) * -np.inf
	elif tr is None:
		tr = np.ones_like(tl) *  np.inf

	if (tr < tl).any():
		raise ValueError("All left truncated values must be less than right truncated values")

	t = np.vstack([tl, tr]).T
	return t
def fs_to_xrd(f, s):
	"""
	Chain of the fs_to_xrd and xrd_to_xcn functions.
	"""
	x, c, n = fs_to_xcn(f, s)
	return xcn_to_xrd(x, c, n)
def fsl_to_xrd(f, s, l):
	"""
	Chain of the fsl_to_xrd and xrd_to_xcn functions.
	"""
	x, c, n = fsl_to_xcn(f, s, l)
	return xcn_to_xrd(x, c, n)
def round_sig(points, sig=2):
	"""
	Used to round to sig significant figures.
	"""
	places = sig - np.floor(np.log10(np.abs(points))) - 1
	output = []
	for p, i in zip(points, places):
		output.append(np.round(p, int(i)))
	return output
