import numpy as np
import surpyval.datasets

import surpyval.parametric
import surpyval.nonparametric

NUM = np.float64
TINIEST = np.finfo(np.float64).tiny
EPS = np.sqrt(np.finfo(NUM).eps)

def xcn_sort(x, c, n):
	idx_c = np.argsort(c, kind='stable')
	x = x[idx_c]
	c = c[idx_c]
	n = n[idx_c]

	idx = np.argsort(x, kind='stable')
	x = x[idx]
	c = c[idx]
	n = n[idx]
	return x, c, n

def xcn_handler(x, c=None, n=None):
	x = np.array(x)
	assert x.ndim == 1

	if c is not None:
		c = np.array(c)
		assert c.ndim == 1
		assert c.shape == x.shape
		assert not any(
				(c !=  0) &
				(c !=  1) &
				(c != -1) &
				(c !=  2)
			)
	else:
		c = np.zeros_like(x)

	if n is not None:
		n = np.array(n)
		assert n.ndim == 1
		assert n.shape == x.shape
	else:
		n = np.ones_like(x)

	n = n.astype(np.int64)
	c = c.astype(np.int64)

	x, c, n = xcn_sort(x, c, n)

	return x, c, n

def xcn_to_xrd(x, c=None, n=None):
    x = x.copy()
    # Handle censoring
    x, c, n = surpyval.xcn_handler(x, c, n)

    assert not ((c != 1) & (c != 0)).any()
    
    # Handle counts
    if n is not None:
        n = n.astype(np.int64)
        x = np.repeat(x, n)
        c = np.repeat(c, n)
    n = np.ones_like(x).astype(np.int64)
    assert n.shape == x.shape
    assert (n > 0).all()

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
	Converts the x, r, d format to the x, c, n format
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
	x_f, n_f = np.unique(f, return_counts=True)
	c_f = np.zeros_like(x_f)

	x_s, n_s = np.unique(s, return_counts=True)
	c_s = np.ones_like(x_s)

	x_l, n_l = np.unique(l, return_counts=True)
	c_l = -np.ones_like(x_l)

	x = np.hstack([x_f, x_s, x_l])
	c = np.hstack([c_f, c_s, c_l])
	n = np.hstack([n_f, n_s, n_l])

	x, c, n = xcn_sort(x, c, n)

	return x, c, n


def fs_to_xcn(f, s):
	x_f, n_f = np.unique(f, return_counts=True)
	c_f = np.zeros_like(x_f)

	x_s, n_s = np.unique(s, return_counts=True)
	c_s = np.ones_like(x_s)

	x = np.hstack([x_f, x_s])
	c = np.hstack([c_f, c_s])
	n = np.hstack([n_f, n_s])

	x, c, n = xcn_sort(x, c, n)

	return x, c, n

def fs_to_xrd(f, s):
	x, c, n = fs_to_xcn(f, s)
	return xcn_to_xrd(x, c, n)

def round_sig(points, sig=2):
    places = sig - np.floor(np.log10(np.abs(points))) - 1
    output = []
    for p, i in zip(points, places):
        output.append(np.round(p, np.int(i)))
    return output