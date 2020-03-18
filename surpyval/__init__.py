import numpy as np
import surpyval.datasets

import surpyval.parametric
import surpyval.nonparametric

NUM = np.float64

def xcn_to_xrd(x, c=None, n=None):
    x = x.copy()
    # Handle censoring
    if c is None: c = np.zeros_like(x).astype(np.int64)
    else: c.astype(np.int64, casting='safe')
    assert c.shape == x.shape
    # xrd format can't be done with left censoring
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

	idx_c = np.argsort(c, kind='stable')
	x = x[idx_c]
	c = c[idx_c]
	n = n[idx_c]

	idx = np.argsort(x, kind='stable')
	x = x[idx]
	c = c[idx]
	n = n[idx]

	return x, c, n


def fs_to_xcn(f, s):
	x_f, n_f = np.unique(f, return_counts=True)
	c_f = np.zeros_like(x_f)

	x_s, n_s = np.unique(s, return_counts=True)
	c_s = np.ones_like(x_s)

	x = np.hstack([x_f, x_s])
	c = np.hstack([c_f, c_s])
	n = np.hstack([n_f, n_s])

	idx_c = np.argsort(c, kind='stable')
	x = x[idx_c]
	c = c[idx_c]
	n = n[idx_c]

	idx = np.argsort(x, kind='stable')
	x = x[idx]
	c = c[idx]
	n = n[idx]

	return x, c, n