"""

Surpyval Again
===============

More stuff about the surpyval package

"""

import numpy as np
import surpyval.datasets

import surpyval.parametric
import surpyval.nonparametric

from surpyval.parametric import Weibull
from surpyval.parametric import Gumbel
from surpyval.parametric import LogLogistic
from surpyval.parametric import Logistic
from surpyval.parametric import Exponential
from surpyval.parametric import Gamma
from surpyval.parametric import LFP
from surpyval.parametric import MixtureModel
from surpyval.parametric import Weibull3p
from surpyval.parametric import LogNormal
from surpyval.parametric import Normal
from surpyval.parametric import Uniform
from surpyval.parametric import Weibull_Mix_Two

from surpyval.nonparametric import KaplanMeier
from surpyval.nonparametric import NelsonAalen
from surpyval.nonparametric import FlemingHarrington

NUM = np.float64
TINIEST = np.finfo(np.float64).tiny
EPS = np.sqrt(np.finfo(NUM).eps)

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
	x = np.array(x)
	assert x.ndim < 3, "variable array must be one or two dimensional"

	if x.ndim == 2:
		assert x.shape[1] == 2, "Dim 1 must be 2, try transposing data, or do you have a 1d array in a 2d array?"

	if c is not None:
		c = np.array(c)
		assert c.ndim == 1, "censoring array must be one dimensional"
		assert c.shape[0] == x.shape[0], "censoring array must be same length as variable array"
		assert not any(
				(c !=  0) &
				(c !=  1) &
				(c != -1) &
				(c !=  2)
			), "Censoring value must only be one of -1, 0, 1, or 2"
	else:
		c = np.zeros_like(x)

	if n is not None:
		n = np.array(n)
		assert n.ndim == 1, "count array must be one dimensional"
		assert n.shape[0] == x.shape[0], "count array must be same length as variable array."
		assert (n > 0).all(), "count array can't be 0"
	else:
		n = np.ones_like(x)

	n = n.astype(np.int64)
	c = c.astype(np.int64)

	x, c, n = xcn_sort(x, c, n)

	return x, c, n

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
	x, c, n = surpyval.xcn_handler(x, c, n)
	assert not ((c != 1) & (c != 0)).any(), "xrd format can't handle left (c=-1) or interval (c=2) censoring"

	x = np.repeat(x, n)
	c = np.repeat(c, n)
	n = np.ones_like(x).astype(np.int64)

	x, idx = np.unique(x, return_inverse=True)

	d = np.bincount(idx, weights=1 - c)
	# do is drop outs
	do = np.bincount(idx, weights=c)
	r = n.sum() + d - d.cumsum() + do - do.cumsum()
	r = r.astype(np.int64)
	d = d.astype(np.int64)
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
	c = np.hstack([c_f, c_s, c_l]).astype(np.int64)
	n = np.hstack([n_f, n_s, n_l]).astype(np.int64)

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
	c = np.hstack([c_f, c_s]).astype(np.int64)
	n = np.hstack([n_f, n_s]).astype(np.int64)

	x, c, n = xcn_sort(x, c, n)

	return x, c, n

def fs_to_xrd(f, s):
	"""
	Chain of the fs_to_xrd and xrd_to_xcn functions.
	"""
	x, c, n = fs_to_xcn(f, s)
	return xcn_to_xrd(x, c, n)

def round_sig(points, sig=2):
	"""
	Used to round to sig significant figures.
	"""
	places = sig - np.floor(np.log10(np.abs(points))) - 1
	output = []
	for p, i in zip(points, places):
		output.append(np.round(p, np.int(i)))
	return output