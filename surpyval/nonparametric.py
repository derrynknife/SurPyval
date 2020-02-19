import numpy as np
import pandas as pd

from scipy.stats import t, norm
from scipy.stats import rankdata
from scipy.special import ndtri as z

NUM = np.float64

PLOTTING_METHODS = [ "Blom", "Median", "ECDF", "Modal", "Midpoint", 
"Mean", "Weibull", "Benard", "Beard", "Hazen", "Gringorten", 
"None", "Tukey", "DPW", "Fleming-Harrington", "Kaplan-Meier",
"Nelson-Aalen", "Filiben"]

"""
Conventions for surpyval package
- c = censoring
- x = random variable (time, stress etc.)
- n = counts
- r = risk set
- d = deaths

- ff / F = Failure Function
- sf / R = Survival Function
- h = hazard rate
- H = Cumulative hazard function

- Censoring: -1 = left
              0 = failure / event
              1 = right
This is done to give an intuitive feel for when the 
event happened on the timeline.

- Censoring vectors. Pass repeat x's not censoring counts.
- Count doesn't assume unique x, it assumes repeats of censoring possible
"""

def plotting_positions(x, c=None, n=None, heuristic="Blom", A=None, B=None):
    # Need some error catching on the A and B thing

    if n is None:
        n = np.ones_like(x).astype(np.int64)

    if c is None:
        c = np.zeros_like(x).astype(np.int64)

    assert x.ndim == 1
    assert (x.ndim == c.ndim) & (x.ndim == n.ndim)
    assert (x.size == n.size) & (x.size == c.size)

    N = n.sum()

    # Some repeated models with different names (for readability...)

    if heuristic == 'Filliben':
        # Needs work
        x_, r, d, R = filliben(x, c=c, n=n)
        F = 1 - R 
        return x_, r, d, F
    elif heuristic == 'Nelson-Aalen':
        x_, r, d, R = nelson_aalen(x, c, n)
        F = 1 - R
        return x_, r, d, F
    elif heuristic == 'Kaplan-Meier':
        x_, r, d, R = kaplan_meier(x, c, n)
        F = 1 - R
        return x_, r, d, F
    elif heuristic == 'Fleming-Harrington':
        x_, r, d, R = fleming_harrington(x, c, n)
        F = 1 - R
        return x_, r, d, F
    else:
        # Reformat for plotting point style
        x_ = np.repeat(x, n)
        c = np.repeat(c, n)
        n = np.ones_like(x_)

        idx = np.argsort(c, kind='stable')
        x_ = x_[idx]
        c  = c[idx]

        idx2 = np.argsort(x_, kind='stable')
        x_ = x_[idx2]
        c  = c[idx2]

        ranks = rank_adjust(x_, c=c)
        d = 1 - c
        r = np.linspace(N, 1, num=N)

        if   heuristic == "Blom":       A, B = 0.375, 0.25
        elif heuristic == "Median":     A, B = 0.3, 0.4
        elif heuristic == "ECDF":       A, B = 0, 1
        elif heuristic == "Modal":      A, B = 1.0, -1.0
        elif heuristic == "Midpoint":   A, B = 0.5, 0.0
        elif heuristic == "Mean":       A, B = 0.0, 1.0
        elif heuristic == "Weibull":    A, B = 0.0, 1.0
        elif heuristic == "Benard":     A, B = 0.3, 0.2
        elif heuristic == "Beard":      A, B = 0.31, 0.38
        elif heuristic == "Hazen":      A, B = 0.5, 0.0
        elif heuristic == "Gringorten": A, B = 0.44, 0.12
        elif heuristic == "None":       A, B = 0.0, 0.0
        elif heuristic == "Tukey":      A, B = 1./3., 1./3.
        elif heuristic == "DPW":        A, B = 1.0, 0.0
        F = (ranks - A)/(N + B)
        F = pd.Series(F).ffill().fillna(0).values
        return x_, r, d, F
def filliben(x, c=None, n=None):
    """
    Method From:
    Filliben, J. J. (February 1975), 
    "The Probability Plot Correlation Coefficient Test for Normality", 
    Technometrics, American Society for Quality, 17 (1): 111-117
    """
    if n is None:
        n = np.ones_like(x)

    if c is None:
        c = np.zeros_like(x)
        
    x_ = np.repeat(x, n)
    c = np.repeat(c, n)
    n = np.ones_like(x_)

    idx = np.argsort(c, kind='stable')
    x_ = x_[idx]
    c  = c[idx]

    idx2 = np.argsort(x_, kind='stable')
    x_ = x_[idx2]
    c  = c[idx2]
    N = len(x_)
    
    ranks = rank_adjust(x_, c)
    d = 1 - c
    r = np.linspace(N, 1, num=N)
    
    F     = (ranks - 0.3175) / (N + 0.365)
    F[0]  = 1 - (0.5 ** (1./N))
    F[-1] = 0.5 ** (1./N)
    R = 1 - F
    return x, r, d, R
def nelson_aalen(x, c=None, n=None):
	"""
	Nelson-Aalen estimation of Reliability function
    Nelson, W.: Theory and Applications of Hazard Plotting for Censored Failure Data. 
    Technometrics, Vol. 14, #4, 1972
    Technically the NA estimate is for the Cumulative Hazard Function,
    The reliability (survival) curve that is output is also known as the Breslow estimate.
    I will leave it as Nelson-Aalen for this library.

    return_all is called by the fit method to ensure h, x, c, d are all saved

    Hazard Rate
	h = d/r
	Cumulative Hazard Function
	H = cumsum(h)
	Reliability Function
	R = exp(-H)
	"""	
	x, r, d = get_x_r_d(x, c, n)

	h = d/r
	H = np.cumsum(h)
	R = np.exp(-H)
	return x, r, d, R
def fleming_harrington(x, c=None, n=None):
	"""
	Fleming Harrington estimation of Reliability function
   
    return_all is called by the fit method to ensure h, x, c, d are all saved

    Hazard Rate:
    at each x, for each d:
	h = 1/r + 1/(r-1) + ... + 1/(r-d)
	Cumulative Hazard Function
	H = cumsum(h)
	Reliability Function
	R = exp(-H)
	"""
	x, r, d = get_x_r_d(x, c, n)

	h = [np.sum([1./(r[i]-j) for j in range(d[i])]) for i in range(len(x))]
	H = np.cumsum(h)
	R = np.exp(-H)
	return x, r, d, R
def kaplan_meier(x, c=None, n=None):
	"""
	Kaplan-Meier estimate of survival
	Good explanation of K-M reason can be found at:
	http://reliawiki.org/index.php/Non-Parametric_Life_Data_Analysis#Kaplan-Meier_Example
	Data given not necessarily in order
	"""
	x, r, d = get_x_r_d(x, c, n)
	
	R = np.cumprod(1 - d/r)
	return x, r, d, R
def success_run(n, confidence=0.95, alpha=None):
    if alpha is None: alpha = 1 - confidence
    return np.power(alpha, 1./n)
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
def rank_adjust(t, c=None):
	"""
	Currently limited to only Mean Order Number
	Room to expand to:
	Modal Order Number, and
	Median Order Number
	Uses mean order statistic to conduct rank adjustment
	For further reading see:
	http://reliawiki.org/index.php/Parameter_Estimation
	Above reference provides excellent explanation of how this method is derived
	This function currently assumes good input - Use if you know how
	15 Mar 2015
	"""
	# Total items in test/population
	N = len(t)
	# Preallocate adjusted ranks array
	ranks = np.zeros(N)

	if c is None:
	    c = np.zeros(N)

	# Rank adjustment for [right] censored data
	# PMON - "Previous Mean Order Number"
	# NIPBSS - "Number of Items Before Present Suspended Set"
	PMON = 0
	for i in range(0, N):
	    if c[i] == 0:
	        NIBPSS = N - i
	        ranks[i] = PMON + (N + 1 - PMON)/(1 + NIBPSS)
	        PMON = ranks[i]
	    elif c[i] == 1:
	        ranks[i] = np.nan
	    else:
	        # ERROR
	        pass
	# Return adjusted ranks
	return ranks

class NonParametric(object):
	"""
	Class to capture all data and meta data on non-parametric sur(py)val model

	Needs to have:
	f = None - or empirical
	confidence bounds

	method for pp/F:
	Turnbull

	TODO: add confidence bounds
	standard: h_u, H_u or Ru, Rl

	"""
	def __init__(self):
		pass

	def __str__(self):
		# Used to automate print(NonParametric()) call
		return "%s Reliability Model" % self.model
	# TODO: This
	def sf(self, x, how='step'):
		x = np.atleast_1d(x)
		# Let's not assume we can predict above the highest measurement
		if how == 'step':
			idx = np.searchsorted(self.x, x, side='right') - 1
			R = self.R[idx]
			R[np.where(x < self.x.min())] = 1
			R[np.where(x > self.x.max())] = np.nan
			R[np.where(x < 0)] = np.nan
			return R
		elif how == 'interp':
			R = np.hstack([[1], self.R])
			x_data = np.hstack([[0], self.x])
			R = np.interp(x, x_data, R)
			R[np.where(x > self.x.max())] = np.nan
			return R

	def ff(self, x, how='step'):
		return 1 - self.sf(x, how=how)

	def hf(self, x, how='step'):
		return np.diff(self.Hf(x, how=how))

	def Hf(self, x, how='step'):
		H = -np.log(self.sf(x, how=how))
		H[H == 0] = 0
		return H

	def R_cb(self, x, bound='upper', how='step', confidence=0.95, bound_type='exp', dist='t'):
		# Greenwoods variance using t-stat. Ref found:
		# http://reliawiki.org/index.php/Non-Parametric_Life_Data_Analysis
		assert bound_type in ['exp', 'normal']
		assert dist in ['t', 'z']
		x = np.atleast_1d(x)
		if bound in ['upper', 'lower']:
			if dist == 't':
				stat = t.ppf(1 - confidence, self.r - 1)
			else:
				stat = norm.ppf(1 - confidence, 0, 1)
			if bound == 'upper' : stat = -stat
		elif bound == 'two-sided':
			if dist == 't':
				stat = t.ppf((1 - confidence)/2, self.r - 1)
			else:
				stat = norm.ppf((1 - confidence)/2, 0, 1)
			stat = np.array([-1, 1]).reshape(2, 1) * stat

		if bound_type == 'exp':
			# Exponential Greenwood confidence
			R_out = self.greenwood * 1./(np.log(self.R)**2)
			R_out = np.log(-np.log(self.R)) - stat * np.sqrt(R_out)
			R_out = np.exp(-np.exp(R_out))
		else:
			# Normal Greenwood confidence
			R_out = self.R + np.sqrt(self.greenwood * self.R**2) * stat
		# Let's not assume we can predict above the highest measurement
		if how == 'step':
			R_out[np.where(x < self.x.min())] = 1
			R_out[np.where(x > self.x.max())] = np.nan
			R_out[np.where(x < 0)] = np.nan
			idx = np.searchsorted(self.x, x, side='right') - 1
			if bound == 'two-sided':
				R_out = R_out[:, idx].T
			else:
				R_out = R_out[idx]
		elif how == 'interp':
			if bound == 'two-sided':
				R1 = np.interp(x, self.x, R_out[0, :])
				R2 = np.interp(x, self.x, R_out[1, :])
				R_out = np.vstack([R1, R2]).T
			else:
				R_out = np.interp(x, self.x, R_out)
			R_out[np.where(x > self.x.max())] = np.nan
		return R_out

	def random(self, size):
		return np.random.choice(self.x, size=size)

	@classmethod
	def fit(cls, x, how='Nelson-Aalen', 
			c=None, n=None, sig=0.05):
		assert how in PLOTTING_METHODS
		data = {}
		data['x'] = x
		data['c'] = c
		data['n'] = n
		out = cls()
		out.data = data
		out.model = how
		if   how == 'Nelson-Aalen':
			x_, r, d, R = nelson_aalen(x, c=c, n=n)
		elif how == 'Kaplan-Meier':
			x_, r, d, R = kaplan_meier(x, c=c, n=n)
		elif how == 'Fleming-Harrington':
			x_, r, d, R = fleming_harrington(x, c=c, n=n)

		out.x = x_
		out.max_x = np.max(out.x)
		out.r = r
		out.d = d
		with np.errstate(divide='ignore'):
			out.H = -np.log(R)
		out.R = R
		out.F = 1 - out.R

		with np.errstate(divide='ignore'):
			var = out.d / (out.r * (out.r - out.d))
		
		with np.errstate(invalid='ignore'):
			greenwood = np.cumsum(var)
		out.greenwood = greenwood

		return out




