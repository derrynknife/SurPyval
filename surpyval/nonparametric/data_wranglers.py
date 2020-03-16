import numpy as np
import pandas as pd
from surpyval.nonparametric import NUM

def get_x_r_d(t, c=None, n=None):
    x = t.copy()
    # Handle censoring
    if c is None: c = np.zeros_like(x).astype(np.int64)
    else: c.astype(np.int64, casting='safe')
    assert c.shape == t.shape
    assert not ((c != 1) & (c != 0)).any()
    
    # Handle counts
    if n is not None:
        n = n.astype(np.int64, casting='safe')
        x = np.repeat(x, n)
        c = np.repeat(c, n)
    else:
        n = np.ones_like(x).astype(np.int64)
    assert n.shape == t.shape
    assert (n > 0).all()

    x, idx = np.unique(x, return_inverse=True)
    
    d = np.bincount(idx, weights=1 - c)
    # do is drop outs
    do = np.bincount(idx, weights=c)
    r = n.sum() + d - d.cumsum() + do - do.cumsum()
    r = r.astype(np.int64)
    d = d.astype(np.int64)
    return x, r, d

def xrd_to_tcn(x, r, d):
	"""
	Converts the x, r, d format to the t, c, n format
	"""
	df = pd.DataFrame({'x' : x, 'r' : r, 'd' : d})
	df['sus'] = (np.abs((df['r'].diff().shift(-1).fillna(df['r'].iloc[-1] - df['d'].iloc[-1]) + df['d']))).astype(np.int64)
	df_tcn = pd.melt(df, id_vars=['x'], value_vars=['d', 'sus']).replace(0, np.nan).dropna().sort_values('x')
	df_tcn = df_tcn.replace('d', 0).replace('sus', 1)
	c = df_tcn['variable'].astype(np.int64).values
	t = df_tcn['x'].astype(NUM).values
	n = df_tcn['value'].astype(np.int64).values
	return t, c, n