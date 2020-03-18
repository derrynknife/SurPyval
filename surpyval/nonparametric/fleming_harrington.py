import numpy as np
import surpyval
import surpyval.nonparametric as nonp

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
	x, c, n = surpyval.xcn_handler(x, c, n)
	x, r, d = surpyval.xcn_to_xrd(x, c, n)

	h = [np.sum([1./(r[i]-j) for j in range(d[i])]) for i in range(len(x))]
	H = np.cumsum(h)
	R = np.exp(-H)
	return x, r, d, R