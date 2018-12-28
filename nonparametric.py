import numpy as np
from scipy.stats import rankdata
from scipy.special import ndtri as z
from scipy.stats import t

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

TODO: implement a return_x where repeated values are returned.
"""

def plotting_positions(x, 
					   c = None, 
					   n = None, 
					   heuristic = "Blom",
					   A = None,
					   B = None,
					   return_x=False):
	"""
	This is really a parametric function....
	Plotting positions should really only be used when estimating parameters

    Numbers from "Effect of Ranking Selection on the Weibull Modulus Estimation"
    Authors: Kirtay, S; Dispinar, D.
    From: Gazi University Journal of Science 25(1):175-187, 2012.

    Also see:
    https://en.wikipedia.org/wiki/Q-Q_plot

    for a good summary of points
    TODO: censoring for ranking methods - i.e. rank adjust
    TODO: Add Turnbull
    """
	ranks = rankdata(x, method='ordinal')
	N = len(ranks)

	# Some repeated models with different names (for readability)

	if heuristic == 'Filiben':
		return filiben(x, ranks)
	elif heuristic == 'Nelson-Aalen':
		x_, r, d = get_x_r_d(x, c, n)
		F = 1 - np.exp(-np.cumsum(d/r))
		return F[d != 0]
	elif heuristic == 'Kaplan-Meier':
		x_, r, d = get_x_r_d(x, c, n)
		F = 1 - np.cumprod(1 - d/r)
		return F[d != 0]
	elif heuristic == 'Fleming-Harrington':
		x_, r, d = get_x_r_d(x, c, n)
		h = [np.sum([1./(r[i]-j) for j in np.arange(d[i])]) for i in np.arange(len(x_))]
		F = 1 - np.exp(-np.cumsum(np.array(h)))
		return F[d != 0]
	elif ((A is None) & (B is None)):
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
		return (ranks - A)/(N + B)
	else:
		return None

def ecdf(x, c=None, n=None):
	"""
	Returns the ECDF of the data.
	"""
	R = 1 - plotting_positions(x, c=c, n=n, heuristic="ECDF")
	H = -np.log(R)
	h = np.diff(H)
	return x, r, d, R

def filiben(x, ranks):
	"""
	Method From:
	Filliben, J. J. (February 1975), 
	"The Probability Plot Correlation Coefficient Test for Normality", 
	Technometrics, American Society for Quality, 17 (1): 111-117
	"""
	N = len(x)
	out     = (ranks - 0.3175) / (N + 0.365)
	out[0]  = 1 - (0.5 ** (1./N))
	out[-1] = 0.5 ** (1./N)
	return out

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

def get_x_r_d(x, c=None, n=None):
    x_ = x.copy()
    if c is None:
        c = np.zeros_like(x_)
    if n is not None:
        x_ = np.repeat(x_, n)
        c = np.repeat(c, n)
    else:
        n = np.ones_like(x_)
    x_, idx = np.unique(x_, return_inverse=True)
    d = np.bincount(idx, weights=1 - c)
    r = n.sum() - d.cumsum() + d
    r = r.astype(np.int)
    d = d.astype(np.int)
    return x_, r, d

def rank_adjust(t, censored=None):
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
	idx = np.argsort(t)
	t = t[idx]
	censored = censored[idx]

	# Total items in test/population
	n = len(t)
	# Preallocate adjusted ranks array
	ranks = np.zeros(n)

	if censored is None:
	    censored = np.zeros(n)

	# Rank increment for [right] censored data
	# Previous Mean Order Number
	PMON = 0

	# Implemented in loop:
	# "Number of Items Before Present Suspended Set"
	# NIBPSS = n - (i - 1)
	# Denominator of rank increment = 1 + NIBPSS = n - i + 2
	for i in range(0, n):
	    if censored[i] == 0:
	        ranks[i] = PMON + (n + 1 - PMON)/(n - i + 2)
	        PMON = ranks[i]
	    else:
	        ranks[i] = np.nan
	# Return adjusted ranks
	return ranks

class NonParametric(object):
	"""
	Class to capture all data and meta data on non-parametric sur(py)vival model

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
			R_out = self.R[idx]
			R_out[np.where(x < self.x.min())] = 1
			R_out[np.where(x > self.x.max())] = np.nan
			R_out[np.where(x < 0)] = np.nan
		elif how == 'interp':
			R = np.hstack([[1], self.R])
			x_data = np.hstack([[0], self.x])
			R = np.interp(x, x_data, R)
			R[np.where(x > self.x.max())] = np.nan
			return R
		else:
			pass
		return R_out

	def ff(self, x, how='step'):
		return 1 - self.sf(x, how=how)

	def hf(self, x, how='step'):
		return np.diff(self.Hf(x, how=how))

	def Hf(self, x, how='step'):
		H = -np.log(self.sf(x, how=how))
		H[H == 0] = 0
		return H

	def R_cb(self, x, bound='upper', how='step'):
		x = np.atleast_1d(x)
		if bound == 'upper':
			R_out = self.cb_u
		elif bound == 'lower':
			R_out = self.cb_l
		# Let's not assume we can predict above the highest measurement
		if how == 'step':
			idx = np.searchsorted(self.x, x, side='right') - 1
			R_out = R_out[idx]
			R_out[np.where(x < self.x.min())] = 1
			R_out[np.where(x > self.x.max())] = np.nan
			R_out[np.where(x < 0)] = np.nan
		elif how == 'interp':
			R_out = np.hstack([[1], R_out])
			x_data = np.hstack([[0], self.x])
			R_out = np.interp(x, x_data, R_out)
			R_out[np.where(x > self.x.max())] = np.nan
		return R_out

	def random(self, size):
		return np.random.choice(self.x,size=size)

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
		elif how == 'ECDF':
			x_, r, d, R = ecdf(x, c=c, n=n)

		out.x = x_
		out.max_x = np.max(out.x)
		out.r = r
		out.d = d
		with np.errstate(divide='ignore'):
			out.H = -np.log(R)
		out.R = R
		out.F = 1 - out.R

		# TODO: Also add more options?
		# Happy to ignore the infinite variance expected when all fail
		# Greenwoods variance using t-stat. Ref found:
		# http://reliawiki.org/index.php/Non-Parametric_Life_Data_Analysis
		with np.errstate(divide='ignore'):
			var = out.d / (out.r * (out.r - out.d))
			t_stat = t.ppf(sig/2, r - 1)
		var = np.cumsum(var)
		with np.errstate(invalid='ignore'):
			var = R**2 * var

		R_l = R + t_stat * np.sqrt(var)
		R_u = R - t_stat * np.sqrt(var)
		out.cb_u = R_u
		out.cb_l = R_l
		
		return out




